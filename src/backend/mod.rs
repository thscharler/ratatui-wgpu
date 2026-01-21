pub(crate) mod builder;
pub(crate) mod wgpu_backend;

use std::any::Any;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use crate::fonts::FontBox;
use crate::utils::text_atlas::Atlas;
use wgpu::CommandEncoder;
use wgpu::Device;
use wgpu::Extent3d;
use wgpu::Queue;
use wgpu::RenderPipeline;
use wgpu::Surface;
use wgpu::SurfaceConfiguration;
use wgpu::SurfaceTexture;
use wgpu::TextureDescriptor;
use wgpu::TextureDimension;
use wgpu::TextureFormat;
use wgpu::TextureUsages;
use wgpu::TextureView;
use wgpu::TextureViewDescriptor;
use wgpu::{Adapter, Buffer, Texture};
use wgpu::{BindGroup, BindGroupLayout, Sampler};
use crate::backend::wgpu_backend::ImageInfo;

pub trait PostProcessorBuilder {
    /// Resulting postprocessor.
    type PostProcessor<'a>: PostProcessor + 'a;

    /// Called during initialization of the backend. This should fully
    /// initialize the post processor for rendering. Note that you are expected
    /// to render to the final surface during [`PostProcessor::process`].
    fn compile(
        self,
        device: &Device,
        text_view: &TextureView,
        surface_config: &SurfaceConfiguration,
    ) -> Self::PostProcessor<'static>;
}

/// A pipeline for post-processing rendered text.
pub trait PostProcessor : Any {
    /// Map the screen-coordinates to cell-coordinates.
    fn map_to_cell(
        &self,
        scr_x: u32,
        scr_y: u32,
        font_box: FontBox,
    ) -> (u16, u16);

    /// Called after the drawing dimensions have changed (e.g. the surface was
    /// resized).
    fn resize(
        &mut self,
        device: &Device,
        text_view: &TextureView,
        surface_config: &SurfaceConfiguration,
    );

    /// Called after text has finished compositing. The provided `text_view` is
    /// the composited text. The final output of your implementation should
    /// render to the provided `surface_view`.
    ///
    /// <div class="warning">
    ///
    /// Retaining a reference to the provided surface view will cause a panic if
    /// the swapchain is recreated.
    ///
    /// </div>
    fn process(
        &mut self,
        margin_color: u32,
        encoder: &mut CommandEncoder,
        queue: &Queue,
        text_view: &TextureView,
        surface_config: &SurfaceConfiguration,
        surface_view: &TextureView,
    );

    /// Called to see if this post processor wants to update the screen. By
    /// default, the backend only runs the compositor and post processor when
    /// the text changes. Returning true from this will override that behavior
    /// and cause the processor to be invoked after a call to flush, even if no
    /// text changes occurred.
    fn needs_update(&self) -> bool {
        false
    }
}

/// The surface dimensions of the backend in pixels.
pub struct Dimensions {
    pub width: NonZeroU32,
    pub height: NonZeroU32,
}

impl From<(NonZeroU32, NonZeroU32)> for Dimensions {
    fn from((width, height): (NonZeroU32, NonZeroU32)) -> Self {
        Self { width, height }
    }
}

pub(crate) enum RenderTarget {
    Surface {
        texture: SurfaceTexture,
        view: TextureView,
    },
    #[cfg(test)]
    Headless { view: TextureView },
}

pub(crate) enum RenderSurface<'s> {
    Surface(Surface<'s>),
    #[cfg(test)]
    Headless(Headless),
}

#[cfg(test)]
pub(crate) struct Headless {
    pub(crate) texture: Option<wgpu::Texture>,
    pub(crate) buffer: Option<wgpu::Buffer>,
    pub(crate) buffer_width: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: TextureFormat,
}

impl RenderTarget {
    pub(crate) fn get_view(&self) -> &TextureView {
        match self {
            RenderTarget::Surface { view, .. } => view,
            #[cfg(test)]
            RenderTarget::Headless { view } => view,
        }
    }

    pub(crate) fn present(self) {
        match self {
            RenderTarget::Surface { texture, .. } => texture.present(),
            #[cfg(test)]
            RenderTarget::Headless { .. } => {
                // noop
            }
        }
    }
}

impl<'s> RenderSurface<'s> {
    pub(crate) fn new_surface(surface: Surface<'s>) -> Self {
        Self::Surface(surface)
    }

    #[cfg(test)]
    pub(crate) fn new_headless() -> Self {
        Self::Headless(Headless {
            texture: Default::default(),
            buffer: Default::default(),
            buffer_width: Default::default(),
            width: Default::default(),
            height: Default::default(),
            format: TextureFormat::Rgba8Unorm,
        })
    }

    #[cfg(test)]
    pub(crate) fn new_headless_with_format(format: TextureFormat) -> Self {
        Self::Headless(Headless {
            texture: Default::default(),
            buffer: Default::default(),
            buffer_width: Default::default(),
            width: Default::default(),
            height: Default::default(),
            format,
        })
    }

    pub(crate) fn wgpu_surface(&self) -> Option<&Surface<'s>> {
        match self {
            RenderSurface::Surface(surface) => Some(surface),
            #[cfg(test)]
            RenderSurface::Headless(_) => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn headless(&self) -> Option<&Headless> {
        match self {
            RenderSurface::Surface(_) => None,
            #[cfg(test)]
            RenderSurface::Headless(headless) => Some(headless),
        }
    }

    pub(crate) fn get_default_config(
        &self,
        adapter: &Adapter,
        width: u32,
        height: u32,
    ) -> Option<SurfaceConfiguration> {
        match self {
            RenderSurface::Surface(surface) => surface.get_default_config(adapter, width, height),
            #[cfg(test)]
            RenderSurface::Headless(Headless { format, .. }) => Some(SurfaceConfiguration {
                usage: TextureUsages::RENDER_ATTACHMENT,
                format: *format,
                width,
                height,
                present_mode: wgpu::PresentMode::Immediate,
                desired_maximum_frame_latency: 2,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
            }),
        }
    }

    pub(crate) fn configure(
        &mut self,
        device: &Device,
        config: &SurfaceConfiguration,
    ) {
        match self {
            RenderSurface::Surface(surface) => {
                Surface::configure(surface, device, config);
            }
            #[cfg(test)]
            RenderSurface::Headless(Headless {
                texture,
                buffer,
                buffer_width,
                width,
                height,
                format,
            }) => {
                *texture = Some(device.create_texture(&TextureDescriptor {
                    label: None,
                    size: Extent3d {
                        width: config.width,
                        height: config.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: *format,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
                    view_formats: &[],
                }));

                *buffer_width = config.width * 4;
                *buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (*buffer_width * config.height) as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }));
                *width = config.width;
                *height = config.height;
            }
        }
    }

    pub(crate) fn get_current_texture(&self) -> Option<RenderTarget> {
        match self {
            RenderSurface::Surface(surface) => {
                let output = match surface.get_current_texture() {
                    Ok(output) => output,
                    Err(err) => {
                        error!("{err}");
                        return None;
                    }
                };

                let view = output
                    .texture
                    .create_view(&TextureViewDescriptor::default());

                Some(RenderTarget::Surface {
                    texture: output,
                    view,
                })
            }
            #[cfg(test)]
            RenderSurface::Headless(Headless { texture, .. }) => {
                texture.as_ref().map(|t| RenderTarget::Headless {
                    view: t.create_view(&TextureViewDescriptor::default()),
                })
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TextBgVertexMember {
    vertex: [f32; 2],
    bg_color: u32,
}

// Vertex + UVCoord + Color
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TextVertexMember {
    vertex: [f32; 2],
    uv: [f32; 2],
    uv_x0: f32,
    fg_color: u32,
    color_glyph: u32,
    underline_pos: u32,
    strikeout_pos: u32,
    cursor_pos: u32,
    cursor_color: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ImgVertexMember {
    vertex: [f32; 2],
    uv: [f32; 2],
}

struct ImgPipeline {
    pipeline: RenderPipeline,
    fs_uniforms: BindGroup,
    fragment_shader_layout: BindGroupLayout,
    image_shader_layout: BindGroupLayout,
}

struct TextCacheBgPipeline {
    pipeline: RenderPipeline,
    fs_uniforms: BindGroup,
}

struct TextCacheFgPipeline {
    pipeline: RenderPipeline,
    fs_uniforms: BindGroup,
    atlas_bindings: BindGroup,
}

struct WgpuBase<'s> {
    surface: RenderSurface<'s>,
    surface_config: SurfaceConfiguration,
    device: Device,
    queue: Queue,
    text_dest_view: TextureView,
}

struct WgpuAtlas {
    cached: Atlas,
    text_cache: Texture,
}

struct WgpuImage {
    texture: TextureView,
    width: u32,
    height: u32,
    dropped: Arc<AtomicBool>,
}

struct WgpuImages {
    pub(super) img_id: usize,
    pub(super) img: HashMap<usize, WgpuImage>,
}

struct WgpuVertices {
    pub(super) text_indices: Vec<[u32; 6]>,
    pub(super) bg_vertices: Vec<TextBgVertexMember>,
    pub(super) text_vertices: Vec<TextVertexMember>,

    pub(super) img_render: Vec<ImageInfo>,
    pub(super) img_indices: Vec<[u32; 6]>,
    pub(super) img_vertices: Vec<ImgVertexMember>,
}

struct WgpuPipeline {
    sampler: Sampler,

    text_screen_size_buffer: Buffer,

    text_bg_compositor: TextCacheBgPipeline,
    text_fg_compositor: TextCacheFgPipeline,

    img_compositor: ImgPipeline,
}

impl WgpuVertices {
    pub fn is_empty(&self) -> bool {
        self.bg_vertices.is_empty() && self.text_vertices.is_empty() && self.img_vertices.is_empty()
    }

    pub fn clear(&mut self) {
        self.text_indices.clear();
        self.bg_vertices.clear();
        self.text_vertices.clear();
        self.img_vertices.clear();
        self.img_indices.clear();
        self.img_render.clear();
    }
}

fn build_wgpu_state(
    device: &Device,
    drawable_width: u32,
    drawable_height: u32,
) -> TextureView {
    let text_dest = device.create_texture(&TextureDescriptor {
        label: Some("Text Compositor Out"),
        size: Extent3d {
            width: drawable_width.max(1),
            height: drawable_height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let text_dest_view = text_dest.create_view(&TextureViewDescriptor::default());

    text_dest_view
}
