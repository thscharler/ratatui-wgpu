pub(crate) mod builder;
pub(crate) mod wgpu_backend;

use std::num::NonZeroU32;

use wgpu::Adapter;
use wgpu::BindGroup;
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
pub trait PostProcessor {
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

/// Controls the area the text is rendered to relative to the presentation
/// surface.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub enum Viewport {
    /// Render to the entire surface.
    #[default]
    Full,
    /// Render to a reduced area starting at the top right and rendering up to
    /// the bottom left - (width, height).
    Shrink { width: u32, height: u32 },
}

pub(crate) enum RenderTarget {
    Surface {
        texture: SurfaceTexture,
        view: TextureView,
    },
    #[cfg(test)]
    Headless {
        view: TextureView,
    },
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
    fg_color: u32,
    underline_pos: u32,
    underline_color: u32,
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

struct WgpuState {
    text_dest_view: TextureView,
}

fn build_wgpu_state(
    device: &Device,
    drawable_width: u32,
    drawable_height: u32,
) -> WgpuState {
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

    WgpuState { text_dest_view }
}
