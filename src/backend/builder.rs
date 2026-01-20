use crate::backend::wgpu_backend::{ImageBuffer, TuiSurface, WgpuBackend};
use crate::backend::{build_wgpu_state, PostProcessorBuilder, WgpuImages};
use crate::backend::{Dimensions, RenderSurface};
use crate::backend::{ImgPipeline, TextCacheFgPipeline};
use crate::backend::{ImgVertexMember, TextVertexMember};
use crate::backend::{TextBgVertexMember, WgpuAtlas, WgpuBase, WgpuVertices};
use crate::backend::{TextCacheBgPipeline, WgpuPipeline};
use crate::colors::named;
use crate::colors::ColorTable;
use crate::fonts::Font;
use crate::fonts::Fonts;
use crate::shaders::DefaultPostProcessorBuilder;
use crate::utils::plan_cache::PlanCache;
use crate::utils::text_atlas::Atlas;
use crate::Result;
use crate::{CursorStyle, Error};
use ratatui_core::style::Color;
use rustybuzz::UnicodeBuffer;
use std::num::NonZeroU32;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};
use wgpu::util::BufferInitDescriptor;
use wgpu::util::DeviceExt;
use wgpu::AddressMode;
use wgpu::Backends;
use wgpu::BindGroupDescriptor;
use wgpu::BindGroupEntry;
use wgpu::BindGroupLayoutDescriptor;
use wgpu::BindGroupLayoutEntry;
use wgpu::BindingResource;
use wgpu::BindingType;
use wgpu::BlendState;
use wgpu::Buffer;
use wgpu::BufferBindingType;
use wgpu::BufferDescriptor;
use wgpu::BufferUsages;
use wgpu::ColorTargetState;
use wgpu::ColorWrites;
use wgpu::Device;
use wgpu::Extent3d;
use wgpu::FilterMode;
use wgpu::FragmentState;
use wgpu::Instance;
use wgpu::InstanceDescriptor;
use wgpu::InstanceFlags;
use wgpu::Limits;
use wgpu::MipmapFilterMode;
use wgpu::MultisampleState;
use wgpu::PipelineCompilationOptions;
use wgpu::PipelineLayoutDescriptor;
use wgpu::PresentMode;
use wgpu::PrimitiveState;
use wgpu::PrimitiveTopology;
use wgpu::RenderPipelineDescriptor;
use wgpu::Sampler;
use wgpu::SamplerBindingType;
use wgpu::SamplerDescriptor;
use wgpu::ShaderStages;
use wgpu::Surface;
use wgpu::SurfaceTarget;
use wgpu::TextureDescriptor;
use wgpu::TextureDimension;
use wgpu::TextureFormat;
use wgpu::TextureSampleType;
use wgpu::TextureUsages;
use wgpu::TextureView;
use wgpu::TextureViewDescriptor;
use wgpu::TextureViewDimension;
use wgpu::VertexBufferLayout;
use wgpu::VertexState;
use wgpu::VertexStepMode;
use wgpu::{include_wgsl, MemoryHints};
use wgpu::{vertex_attr_array, BindGroup};

const CACHE_WIDTH: u32 = 1800;
const CACHE_HEIGHT: u32 = 1200;

/// Builds a [`WgpuBackend`] instance.
///
/// Height and width will default to 1x1, so don't forget to call
/// [`Builder::with_dimensions`] to configure the backend presentation
/// dimensions.
pub struct Builder<'a, P> {
    postprocessor: P,
    fonts: Fonts<'a>,
    backends: Backends,
    instance: Option<Instance>,
    limits: Option<Limits>,
    present_mode: Option<PresentMode>,
    width: u32,
    height: u32,
    colors: ColorTable,
    reset_fg: Color,
    reset_bg: Color,
    fast_blink: u8,
    slow_blink: u8,
    cursor_blink: u8,
    cursor_style: CursorStyle,
    cursor_color: Color,
}

pub type DefaultBuilder<'a> = Builder<'a, DefaultPostProcessorBuilder>;

impl<'a, P> Builder<'a, P>
where
    P: PostProcessorBuilder + Default,
{
    /// Create a new Builder from a specified [`Font`] and default
    /// [`PostProcessor::UserData`].
    pub fn from_font(font: Font<'a>) -> Self {
        Self {
            postprocessor: Default::default(),
            instance: None,
            fonts: Fonts::new(font, 22),
            limits: None,
            present_mode: None,
            width: 100,
            height: 100,
            colors: named::DEFAULT_COLORS,
            reset_fg: Color::Black,
            reset_bg: Color::White,
            fast_blink: 1,
            slow_blink: 5,
            cursor_blink: 5,
            cursor_style: Default::default(),
            cursor_color: Color::Reset,
            backends: Default::default(),
        }
    }

    /// Create a new Builder with a list of fallback fonts.
    pub fn from_fonts(fonts: Fonts<'a>) -> Self {
        Self {
            postprocessor: Default::default(),
            instance: None,
            fonts,
            limits: None,
            present_mode: None,
            width: 100,
            height: 100,
            colors: named::DEFAULT_COLORS,
            reset_fg: Color::Black,
            reset_bg: Color::White,
            fast_blink: 1,
            slow_blink: 5,
            cursor_blink: 5,
            cursor_style: Default::default(),
            cursor_color: Color::Reset,
            backends: Default::default(),
        }
    }
}

impl<'a, P> Builder<'a, P>
where
    P: PostProcessorBuilder,
{
    /// Create a new Builder from a specified [`Font`] and supplied
    /// [`PostProcessor::UserData`].
    pub fn from_font_and_user_data(
        font: Font<'a>,
        postprocessor_builder: P,
    ) -> Self {
        Self {
            postprocessor: postprocessor_builder,
            instance: None,
            fonts: Fonts::new(font, 22),
            limits: None,
            present_mode: None,
            width: 100,
            height: 100,
            colors: named::DEFAULT_COLORS,
            reset_fg: Color::Black,
            reset_bg: Color::White,
            fast_blink: 1,
            slow_blink: 5,
            cursor_blink: 5,
            cursor_style: Default::default(),
            cursor_color: Color::Reset,
            backends: Default::default(),
        }
    }

    /// Use one of the given Backends.
    #[must_use]
    pub fn with_backends(
        mut self,
        backends: Backends,
    ) -> Self {
        self.backends = backends;
        self
    }

    /// Use the supplied [`wgpu::Instance`] when building the backend.
    #[must_use]
    pub fn with_instance(
        mut self,
        instance: Instance,
    ) -> Self {
        self.instance = Some(instance);
        self
    }

    /// Use the specified font size in pixels. Defaults to 24px.
    ///
    /// __Note__
    ///
    /// Size 0 is ignored.
    #[must_use]
    pub fn with_font_size_px(
        mut self,
        size: u32,
    ) -> Self {
        if size > 0 {
            self.fonts.set_size_px(size);
        }
        self
    }

    /// Use the specified list of fonts for rendering. You may call this
    /// multiple times to extend the list of fallback fonts. Note that this will
    /// automatically organize fonts by relative width in order to optimize
    /// fallback rendering quality. The ordering of already provided fonts will
    /// remain unchanged.
    ///
    /// See also [`Fonts::add_fonts`].
    pub fn with_fonts<I: IntoIterator<Item = Font<'a>>>(
        mut self,
        fonts: I,
    ) -> Self {
        self.fonts.add_fonts(fonts);
        self
    }

    /// Use the specified list of regular fonts for rendering. You may call this
    /// multiple times to extend the list of fallback fonts.
    ///
    /// See also [`Fonts::add_regular_fonts`].
    #[must_use]
    pub fn with_regular_fonts<I: IntoIterator<Item = Font<'a>>>(
        mut self,
        fonts: I,
    ) -> Self {
        self.fonts.add_regular_fonts(fonts);
        self
    }

    /// Use the specified list of bold fonts for rendering. You may call this
    /// multiple times to extend the list of fallback fonts.
    ///
    /// See also [`Fonts::add_bold_fonts`].
    #[must_use]
    pub fn with_bold_fonts<I: IntoIterator<Item = Font<'a>>>(
        mut self,
        fonts: I,
    ) -> Self {
        self.fonts.add_bold_fonts(fonts);
        self
    }

    /// Use the specified list of italic fonts for rendering. You may call this
    /// multiple times to extend the list of fallback fonts.
    ///
    /// See also [`Fonts::add_italic_fonts`].
    #[must_use]
    pub fn with_italic_fonts<I: IntoIterator<Item = Font<'a>>>(
        mut self,
        fonts: I,
    ) -> Self {
        self.fonts.add_italic_fonts(fonts);
        self
    }

    /// Use the specified list of bold italic fonts for rendering. You may call
    /// this multiple times to extend the list of fallback fonts.
    ///
    /// See also [`Fonts::add_bold_italic_fonts`].
    #[must_use]
    pub fn with_bold_italic_fonts<I: IntoIterator<Item = Font<'a>>>(
        mut self,
        fonts: I,
    ) -> Self {
        self.fonts.add_bold_italic_fonts(fonts);
        self
    }

    /// Use the specified [`wgpu::Limits`]. Defaults to
    /// [`wgpu::Adapter::limits`].
    #[must_use]
    pub fn with_limits(
        mut self,
        limits: Limits,
    ) -> Self {
        self.limits = Some(limits);
        self
    }

    /// Use the specified [`wgpu::PresentMode`].
    #[must_use]
    pub fn with_present_mode(
        mut self,
        mode: PresentMode,
    ) -> Self {
        self.present_mode = Some(mode);
        self
    }

    /// Use the specified height and width when creating the surface. Defaults
    /// to 1x1.
    #[must_use]
    pub fn with_dimensions(
        mut self,
        dimensions: Dimensions,
    ) -> Self {
        self.width = dimensions.width.get();
        self.height = dimensions.height.get();
        self
    }

    /// Use the specified height and width when creating the surface.
    ///
    /// Defaults to 100x100.
    /// Minimum size depends on the font. The window is at least 1x1 cells sized.
    #[must_use]
    pub fn with_width_and_height(
        mut self,
        width: u32,
        height: u32,
    ) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Use the specified [`ColorTable`] for the base-16 colors.
    /// There is a default value for this.
    pub fn with_color_table(
        mut self,
        colors: ColorTable,
    ) -> Self {
        self.colors = colors;
        self
    }

    /// Use the specified [`ratatui::style::Color`] for the default foreground
    /// color. Defaults to Black.
    #[must_use]
    pub fn with_fg_color(
        mut self,
        fg: Color,
    ) -> Self {
        self.reset_fg = fg;
        self
    }

    /// Use the specified [`ratatui::style::Color`] for the default background
    /// color. Defaults to White.
    #[must_use]
    pub fn with_bg_color(
        mut self,
        bg: Color,
    ) -> Self {
        self.reset_bg = bg;
        self
    }

    /// Initial cursor-color.
    #[must_use]
    pub fn with_cursor_color(
        mut self,
        color: Color,
    ) -> Self {
        self.cursor_color = color;
        self
    }

    /// Initial cursor-style.
    #[must_use]
    pub fn with_cursor_style(
        mut self,
        style: CursorStyle,
    ) -> Self {
        self.cursor_style = style;
        self
    }

    /// This library doesn't control the cursor blink timer by itself, instead
    /// it relies on [blink] being called. Every call to blink increases an
    /// internal counter. Every time `internal % counter == 0` the blink-state
    /// is switched.
    ///
    /// So. To switch with every call to blink give a counter 1.
    /// To blink half as fast give a counter 2.
    #[must_use]
    pub fn with_cursor_blink(
        mut self,
        counter: u8,
    ) -> Self {
        self.cursor_blink = counter;
        self
    }

    /// This library doesn't control the blink timer by itself, instead
    /// it relies on [blink] being called. Every call to blink increases an
    /// internal counter. Every time `internal % counter == 0` the blink-state
    /// is switched.
    ///
    /// So. To switch with every call to blink give a counter 1.
    /// To blink half as fast give a counter 2.
    #[must_use]
    pub fn with_rapid_blink(
        mut self,
        counter: u8,
    ) -> Self {
        self.fast_blink = counter;
        self
    }

    /// This library doesn't control the blink timer by itself, instead
    /// it relies on [blink] being called. Every call to blink increases an
    /// internal counter. Every time `internal % counter == 0` the blink-state
    /// is switched.
    ///
    /// So. To switch with every call to blink give a counter 1.
    /// To blink half as fast give a counter 2.
    #[must_use]
    pub fn with_slow_blink(
        mut self,
        counter: u8,
    ) -> Self {
        self.slow_blink = counter;
        self
    }
}

impl<'a, P> Builder<'a, P>
where
    P: PostProcessorBuilder,
{
    /// Build a new backend with the provided surface target - e.g. a winit
    /// `Window`.
    pub async fn build_with_target<'s>(
        mut self,
        target: impl Into<SurfaceTarget<'s>>,
    ) -> Result<WgpuBackend<'a, 's>> {
        let instance = self.instance.get_or_insert_with(|| {
            Instance::new(&InstanceDescriptor {
                backends: self.backends,
                flags: InstanceFlags::default(),
                ..Default::default()
            })
        });

        let surface = instance
            .create_surface(target)
            .map_err(Error::SurfaceCreationFailed)?;

        self.build_with_render_surface(RenderSurface::new_surface(surface))
            .await
    }

    /// Build a new backend from this builder with the supplied surface. You
    /// almost certainly want to call this with the instance you used to create
    /// the provided surface - see [`Builder::with_instance`]. If one is not
    /// provided, a default instance will be created.
    pub async fn build_with_surface<'s>(
        self,
        surface: Surface<'s>,
    ) -> Result<WgpuBackend<'a, 's>> {
        self.build_with_render_surface(RenderSurface::new_surface(surface))
            .await
    }

    #[cfg(test)]
    pub(crate) async fn build_headless(self) -> Result<WgpuBackend<'a, 'static>> {
        self.build_with_render_surface(RenderSurface::new_headless())
            .await
    }

    #[cfg(test)]
    pub(crate) async fn build_headless_with_format(
        self,
        format: TextureFormat,
    ) -> Result<WgpuBackend<'a, 'static>> {
        self.build_with_render_surface(RenderSurface::new_headless_with_format(format))
            .await
    }

    async fn build_with_render_surface<'s>(
        mut self,
        mut surface: RenderSurface<'s>,
    ) -> Result<WgpuBackend<'a, 's>> {
        let instance = self.instance.get_or_insert_with(|| {
            Instance::new(&InstanceDescriptor {
                backends: self.backends,
                flags: InstanceFlags::default(),
                ..Default::default()
            })
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: surface.wgpu_surface(),
                ..Default::default()
            })
            .await
            .map_err(Error::AdapterRequestFailed)?;

        let limits = if let Some(limits) = self.limits {
            limits
        } else {
            Limits::downlevel_defaults()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ratatui-wgpu Device"),
                required_features: Default::default(),
                required_limits: limits,
                experimental_features: Default::default(),
                memory_hints: MemoryHints::MemoryUsage,
                trace: Default::default(),
            })
            .await
            .map_err(Error::DeviceRequestFailed)?;

        // this may create a surface that is bigger than the window.
        let width = self.width.max(self.fonts.min_width_px());
        let height = self.height.max(self.fonts.height_px());

        let mut surface_config = surface
            .get_default_config(&adapter, width, height)
            .ok_or(Error::SurfaceConfigurationRequestFailed)?;

        if let Some(mode) = self.present_mode {
            surface_config.present_mode = mode;
        }

        surface.configure(&device, &surface_config);

        let drawable_width = surface_config.width;
        let drawable_height = surface_config.height;

        info!(
            "char width x height: {}x{}",
            self.fonts.min_width_px(),
            self.fonts.height_px()
        );

        let text_cache = device.create_texture(&TextureDescriptor {
            label: Some("Text Atlas"),
            size: Extent3d {
                width: CACHE_WIDTH,
                height: CACHE_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let text_cache_view = text_cache.create_view(&TextureViewDescriptor::default());

        let sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let text_screen_size_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Text Uniforms Buffer"),
            size: size_of::<[f32; 4]>() as u64,
            mapped_at_creation: false,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let atlas_size_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Atlas Size buffer"),
            contents: bytemuck::cast_slice(&[CACHE_WIDTH as f32, CACHE_HEIGHT as f32, 0.0, 0.0]),
            usage: BufferUsages::UNIFORM,
        });

        let text_bg_compositor = build_text_bg_compositor(
            &device, //
            &text_screen_size_buffer,
        );

        let text_fg_compositor = build_text_fg_compositor(
            &device,
            &text_screen_size_buffer,
            &atlas_size_buffer,
            &text_cache_view,
            &sampler,
        );

        let img_compositor = build_img_compositor(&device, &text_screen_size_buffer);

        let wgpu_view = build_wgpu_state(
            &device,
            (drawable_width / self.fonts.min_width_px()) * self.fonts.min_width_px(),
            (drawable_height / self.fonts.height_px()) * self.fonts.height_px(),
        );

        let reset_fg = self.colors.c2c(self.reset_fg, [255; 3]);
        let reset_bg = self.colors.c2c(self.reset_bg, [0; 3]);

        let font_box = self.fonts.font_box();
        let font_count = self.fonts.count();

        let post_process = self
            .postprocessor
            .compile(&device, &wgpu_view, &surface_config);

        Ok(WgpuBackend {
            fonts: self.fonts,
            tui_surface: TuiSurface {
                image_buffer: ImageBuffer {
                    font_box: Arc::new(Mutex::new(font_box)),
                    image_size: Arc::new(Mutex::new(Default::default())),
                    images: Default::default(),
                },
                images: vec![],
                cells: vec![],
                cell_font: vec![],
                cell_remap: vec![],
                dirty_rows: Default::default(),
                dirty_cells: Default::default(),
                dirty_img: vec![],
                fast_blinking: Default::default(),
                slow_blinking: Default::default(),
                cursor: (0, 0),
                colors: self.colors,
                reset_fg,
                reset_bg,
                cursor_color: self.cursor_color,
                cursor_style: self.cursor_style,
                cursor_visible: true,
                cursor_blink: 0,
                cursor_divisor: self.cursor_blink,
                cursor_showing: true,
                blink: 0,
                fast_blink_divisor: self.fast_blink,
                fast_blink_showing: true,
                slow_blink_divisor: self.slow_blink,
                slow_blink_showing: true,
            },
            rendered: vec![],

            tmp_plan_cache: PlanCache::new(font_count.max(2)),
            tmp_buffer: UnicodeBuffer::new(),
            tmp_rowbuf: String::new(),
            tmp_rowbuf_to_cell: vec![],

            wgpu_base: WgpuBase {
                surface,
                surface_config,
                device,
                queue,
                text_dest_view: wgpu_view,
            },
            wgpu_vertices: WgpuVertices {
                bg_vertices: vec![],
                text_indices: vec![],
                text_vertices: vec![],
                img_render: vec![],
                img_indices: vec![],
                img_vertices: vec![],
            },
            wgpu_atlas: WgpuAtlas {
                cached: Atlas::new(font_box, CACHE_WIDTH, CACHE_HEIGHT),
                text_cache,
            },
            wgpu_images: WgpuImages {
                img_id: 1,
                img: Default::default(),
            },
            wgpu_post_process: Box::new(post_process),
            wgpu_pipeline: WgpuPipeline {
                sampler,
                text_screen_size_buffer,
                text_bg_compositor,
                text_fg_compositor,
                img_compositor,
            },
        })
    }
}

pub(super) fn build_img_size_bindings(
    img_pipeline: &ImgPipeline,
    device: &Device,
    img_size: &Buffer,
    view_size: &Buffer,
    uv_transform: &Buffer,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("Img Size Binding"),
        layout: &img_pipeline.image_shader_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: img_size.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: view_size.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: uv_transform.as_entire_binding(),
            },
        ],
    })
}

pub(super) fn build_img_bindings(
    img_pipeline: &ImgPipeline,
    device: &Device,
    sampler: &Sampler,
    img_texture: &TextureView,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("Img Compositor Fragment Binding"),
        layout: &img_pipeline.fragment_shader_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Sampler(sampler),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(img_texture),
            },
        ],
    })
}

fn build_img_compositor(
    device: &Device,
    screen_size: &Buffer,
) -> ImgPipeline {
    let shader = device.create_shader_module(include_wgsl!("shaders/img.wgsl"));

    let vertex_shader_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Image Compositor Uniforms Binding Layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(size_of::<[f32; 4]>() as u64).unwrap()),
            },
            count: None,
        }],
    });

    let image_shader_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Image Size Uniforms Binding Layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(size_of::<[f32; 2]>() as u64).unwrap()),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(size_of::<[f32; 2]>() as u64).unwrap()),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(size_of::<[[f32; 4]; 2]>() as u64).unwrap()),
                },
                count: None,
            },
        ],
    });

    let fs_uniforms = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Image Compositor Uniforms Binding"),
        layout: &vertex_shader_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: screen_size.as_entire_binding(),
        }],
    });

    let fragment_shader_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Img Compositor Fragment Binding Layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Img Compositor Layout"),
        bind_group_layouts: &[
            &vertex_shader_layout,
            &image_shader_layout,
            &fragment_shader_layout,
        ],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("Img Compositor Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            buffers: &[VertexBufferLayout {
                array_stride: size_of::<ImgVertexMember>() as u64,
                step_mode: VertexStepMode::Vertex,
                attributes: &vertex_attr_array![0 => Float32x2, 1 => Float32x2],
            }],
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            targets: &[Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    ImgPipeline {
        image_shader_layout,
        fragment_shader_layout,
        pipeline,
        fs_uniforms,
    }
}

fn build_text_bg_compositor(
    device: &Device,
    screen_size: &Buffer,
) -> TextCacheBgPipeline {
    let shader = device.create_shader_module(include_wgsl!("shaders/composite_bg.wgsl"));

    let vertex_shader_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Text Bg Compositor Uniforms Binding Layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(size_of::<[f32; 4]>() as u64).unwrap()),
            },
            count: None,
        }],
    });

    let fs_uniforms = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Text Bg Compositor Uniforms Binding"),
        layout: &vertex_shader_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: screen_size.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Text Bg Compositor Layout"),
        bind_group_layouts: &[&vertex_shader_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("Text Bg Compositor Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            buffers: &[VertexBufferLayout {
                array_stride: size_of::<TextBgVertexMember>() as u64,
                step_mode: VertexStepMode::Vertex,
                attributes: &vertex_attr_array![0 => Float32x2, 1 => Uint32],
            }],
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            targets: &[Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    TextCacheBgPipeline {
        pipeline,
        fs_uniforms,
    }
}

fn build_text_fg_compositor(
    device: &Device,
    screen_size: &Buffer,
    atlas_size: &Buffer,
    cache_view: &TextureView,
    sampler: &Sampler,
) -> TextCacheFgPipeline {
    let shader = device.create_shader_module(include_wgsl!("shaders/composite_fg.wgsl"));

    let vertex_shader_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Text Compositor Uniforms Binding Layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(size_of::<[f32; 4]>() as u64).unwrap()),
            },
            count: None,
        }],
    });

    let fragment_shader_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Text Compositor Fragment Binding Layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(size_of::<[f32; 4]>() as u64).unwrap()),
                },
                count: None,
            },
        ],
    });

    let fs_uniforms = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Text Compositor Uniforms Binding"),
        layout: &vertex_shader_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: screen_size.as_entire_binding(),
        }],
    });

    let atlas_bindings = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Text Compositor Fragment Binding"),
        layout: &fragment_shader_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(cache_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(sampler),
            },
            BindGroupEntry {
                binding: 2,
                resource: atlas_size.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Text Compositor Layout"),
        bind_group_layouts: &[&vertex_shader_layout, &fragment_shader_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("Text Compositor Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            buffers: &[VertexBufferLayout {
                array_stride: size_of::<TextVertexMember>() as u64,
                step_mode: VertexStepMode::Vertex,
                attributes: &vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32, 3 => Uint32, 4 => Uint32, 5 => Uint32, 6 => Uint32, 7 => Uint32, 8 => Uint32 ],
            }],
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            targets: &[Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    TextCacheFgPipeline {
        pipeline,
        fs_uniforms,
        atlas_bindings,
    }
}
