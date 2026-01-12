use crate::backend::build_wgpu_state;
use crate::backend::TextBgVertexMember;
use crate::backend::TextVertexMember;
use crate::backend::{PostProcessor, WgpuAtlas, WgpuBase, WgpuPipeline, WgpuVertices};
use crate::colors::ColorTable;
use crate::colors::Rgb;
use crate::fonts::{Font, FontBox};
use crate::fonts::{Fonts, RenderedFont};
use crate::utils::plan_cache::PlanCache;
use crate::utils::text_atlas::CacheRect;
use crate::utils::text_atlas::Entry;
use crate::utils::text_atlas::Key;
use crate::utils::Outline;
use crate::utils::Painter;
use crate::{CursorStyle, PostProcessorBuilder, RandomState};
use bitvec::order::Lsb0;
use bitvec::slice::BitSlice;
use bitvec::vec::BitVec;
use indexmap::IndexMap;
use raqote::DrawOptions;
use raqote::DrawTarget;
use raqote::SolidSource;
use raqote::StrokeStyle;
use raqote::Transform;
use ratatui_core::backend::Backend;
use ratatui_core::backend::ClearType;
use ratatui_core::backend::WindowSize;
use ratatui_core::buffer::Cell;
use ratatui_core::style::Modifier;
use rustybuzz::shape_with_plan;
use rustybuzz::ttf_parser::RasterGlyphImage;
use rustybuzz::ttf_parser::RasterImageFormat;
use rustybuzz::ttf_parser::RgbaColor;
use rustybuzz::ttf_parser::{GlyphId, OutlineBuilder};
use rustybuzz::GlyphBuffer;
use rustybuzz::UnicodeBuffer;
use std::mem::size_of;
use std::num::NonZeroU64;
use std::{iter, mem};
use unicode_bidi::Level;
use unicode_bidi::ParagraphBidiInfo;
use unicode_properties::UnicodeEmoji;
use unicode_properties::UnicodeGeneralCategory;
use unicode_properties::{GeneralCategory, GeneralCategoryGroup};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
use wgpu::util::BufferInitDescriptor;
use wgpu::util::DeviceExt;
use wgpu::CommandEncoderDescriptor;
use wgpu::Extent3d;
use wgpu::IndexFormat;
use wgpu::LoadOp;
use wgpu::Operations;
use wgpu::Origin3d;
use wgpu::RenderPassColorAttachment;
use wgpu::RenderPassDescriptor;
use wgpu::StoreOp;
use wgpu::TextureAspect;
use wgpu::{BufferUsages, Queue};

const NULL_CELL: Cell = {
    let mut c = Cell::new("");
    c.skip = true;
    c
};

const ONE_CELL: Cell = Cell::new(" ");

#[derive(Debug)]
pub(super) struct RenderInfo {
    cached: CacheRect,
    fg: ratatui_core::style::Color,
    bg: ratatui_core::style::Color,
    modifier: Modifier,
    underline_pos_min: u16,
    underline_pos_max: u16,
    strikeout_pos_min: u16,
    strikeout_pos_max: u16,
    cursor_pos_min: u16,
    cursor_pos_max: u16,
}
/// Map from (x, y, glyph) -> (cell index, cache entry).
/// We use an IndexMap because we want a consistent rendering order for
/// vertices.
type Rendered = IndexMap<(i32, i32, GlyphId), RenderInfo, RandomState>;

pub(crate) struct TuiSurface {
    pub(super) cells: Vec<Cell>,
    pub(super) cell_remap: Vec<u16>,
    pub(super) dirty_rows: BitVec,
    pub(super) dirty_cells: BitVec,
    pub(super) fast_blinking: BitVec,
    pub(super) slow_blinking: BitVec,

    pub(super) cursor: (u16, u16),
    pub(super) cursor_color: ratatui_core::style::Color,
    pub(super) cursor_style: CursorStyle,
    pub(super) cursor_visible: bool,
    pub(super) cursor_blink: u8,
    pub(super) cursor_divisor: u8,
    pub(super) cursor_showing: bool,

    pub(super) blink: u8,
    pub(super) fast_blink_divisor: u8,
    pub(super) fast_blink_showing: bool,
    pub(super) slow_blink_divisor: u8,
    pub(super) slow_blink_showing: bool,

    pub(super) colors: ColorTable,
    pub(super) reset_fg: Rgb,
    pub(super) reset_bg: Rgb,
}

/// A ratatui backend leveraging wgpu for rendering.
///
/// Constructed using a [`Builder`](crate::Builder).
///
/// The first lifetime parameter is the lifetime of the data for referenced
/// [`Font`] objects. The second lifetime parameter is the lifetime of the
/// referenced [`Surface`] (typically the lifetime of your window object).
///
/// Limitations:
/// - The cursor is tracked but not rendered.
/// - No builtin accessibilty, although [`WgpuBackend::get_text`] is provided to
///   access the screen's contents.
pub struct WgpuBackend<'f, 's> {
    pub(super) fonts: Fonts<'f>,

    // ratatui state
    pub(super) tui_surface: TuiSurface,

    // positioned glyphs.
    pub(super) rendered: Vec<Rendered>,

    // temporaries for shaping
    pub(super) tmp_plan_cache: PlanCache,
    pub(super) tmp_rowbuf: String,
    pub(super) tmp_rowbuf_to_cell: Vec<u16>,
    pub(super) tmp_buffer: UnicodeBuffer,

    // wgpu input
    pub(super) wgpu_base: WgpuBase<'s>,
    pub(super) wgpu_vertices: WgpuVertices,
    pub(super) wgpu_atlas: WgpuAtlas,
    pub(super) wgpu_post_process: Box<dyn PostProcessor + 'static>,
    pub(super) wgpu_pipeline: WgpuPipeline,
}

impl<'f, 's> WgpuBackend<'f, 's> {
    pub fn set_bg_color(
        &mut self,
        color: ratatui_core::style::Color,
    ) {
        self.tui_surface.reset_bg = self.tui_surface.colors.c2c(color, [0; 3]);
    }

    pub fn set_fg_color(
        &mut self,
        color: ratatui_core::style::Color,
    ) {
        self.tui_surface.reset_fg = self.tui_surface.colors.c2c(color, [255; 3]);
    }

    /// Set the cursor style
    pub fn set_cursor_style(
        &mut self,
        style: CursorStyle,
    ) {
        self.tui_surface.cursor_style = style;
    }

    /// Current cursor style.
    pub fn cursor_style(&self) -> CursorStyle {
        self.tui_surface.cursor_style
    }

    /// Set the cursor color.
    pub fn set_cursor_color(
        &mut self,
        color: ratatui_core::style::Color,
    ) {
        self.tui_surface.cursor_color = color;
    }

    /// Current cursor color.
    pub fn cursor_color(&self) -> ratatui_core::style::Color {
        self.tui_surface.cursor_color
    }

    /// Map a physical cursor position to a col/row position.
    pub fn pos_to_cell(
        &self,
        pos: (u32, u32),
    ) -> (u16, u16) {
        let font_box = self.fonts.font_box();
        if font_box.width == 0 || font_box.height == 0 {
            // might happen during resize or before the first render.
            return (0, 0);
        }

        let (cell_x, cell_y) =
            self.wgpu_post_process
                .map_to_cell(pos.0, pos.1, self.fonts.font_box());

        let bounds = self.size().unwrap();
        let offset = (cell_y * bounds.width) as usize;
        if self.tui_surface.cell_remap.len() < offset + bounds.width as usize {
            // might happen during resize or before the first render.
            return (cell_x, cell_y);
        }
        for cell in 0..bounds.width {
            if self.tui_surface.cell_remap[offset + cell as usize] == cell_x {
                return (cell, cell_y);
            }
        }

        (cell_x, cell_y)
    }

    /// Get the [`PostProcessor`] associated with this backend.
    pub fn post_processor(&self) -> &dyn PostProcessor {
        self.wgpu_post_process.as_ref()
    }

    /// Get a mutable reference to the [`PostProcessor`] associated with this
    /// backend.
    pub fn post_processor_mut(&mut self) -> &mut dyn PostProcessor {
        self.wgpu_post_process.as_mut()
    }

    /// Changes the post-processor.
    pub fn update_post_processor<P: PostProcessorBuilder>(
        &mut self,
        builder: P,
    ) {
        let post_process = builder.compile(
            &self.wgpu_base.device,
            &self.wgpu_base.text_dest_view,
            &self.wgpu_base.surface_config,
        );
        self.wgpu_post_process = Box::new(post_process);
    }

    /// Resize the rendering surface. This should be called e.g. to keep the
    /// backend in sync with your window size.
    pub fn resize(
        &mut self,
        width: u32,
        height: u32,
    ) {
        let limits = self.wgpu_base.device.limits();
        let width = width.min(limits.max_texture_dimension_2d);
        let height = height.min(limits.max_texture_dimension_2d);

        if width == self.wgpu_base.surface_config.width
            && height == self.wgpu_base.surface_config.height
            || width == 0
            || height == 0
        {
            return;
        }

        self.wgpu_base.surface_config.width = width;
        self.wgpu_base.surface_config.height = height;

        rebuild_surface(
            self.fonts.font_box(),
            &mut self.tui_surface,
            &mut self.rendered,
            &mut self.wgpu_base,
            self.wgpu_post_process.as_mut(),
        );
    }

    /// Get the text currently displayed on the screen.
    pub fn get_text(&self) -> String {
        let bounds = self.size().unwrap();
        self.tui_surface.cells.chunks(bounds.width as usize).fold(
            String::with_capacity((bounds.width + 1) as usize * bounds.height as usize),
            |dest, row| {
                let mut dest = row.iter().fold(dest, |mut dest, s| {
                    dest.push_str(s.symbol());
                    dest
                });
                dest.push('\n');
                dest
            },
        )
    }

    /// Update the color-table used for rendering. This will cause a full
    /// repaint of the screen the next time [`WgpuBackend::flush`] is
    /// called.
    pub fn update_color_table(
        &mut self,
        new_colors: ColorTable,
    ) {
        self.tui_surface.dirty_rows.clear();
        self.tui_surface.dirty_cells.clear();
        self.tui_surface.colors = new_colors;
    }

    /// Update the fonts used for rendering. This will cause a full repaint of
    /// the screen the next time [`WgpuBackend::flush`] is called.
    pub fn update_fonts(
        &mut self,
        new_fonts: Fonts<'f>,
    ) {
        self.fonts = new_fonts;
        self.tui_surface.dirty_rows.clear();
        self.tui_surface.dirty_cells.clear();
        self.wgpu_atlas
            .cached
            .update_font_box(self.fonts.font_box());

        rebuild_surface(
            self.fonts.font_box(),
            &mut self.tui_surface,
            &mut self.rendered,
            &mut self.wgpu_base,
            self.wgpu_post_process.as_mut(),
        );
    }

    /// Replace the fonts used for rendering. This will keep the fallback fonts.
    /// If you want to replace those too, use [update_fonts].
    ///
    /// This will cause a full repaint of the screen the next
    /// time [`WgpuBackend::flush`] is called.
    pub fn update_font_vec(
        &mut self,
        new_fonts: Vec<Font<'f>>,
    ) {
        self.fonts.clear_fonts();
        self.fonts.add_fonts(new_fonts);
        self.tui_surface.dirty_rows.clear();
        self.tui_surface.dirty_cells.clear();
        self.wgpu_atlas
            .cached
            .update_font_box(self.fonts.font_box());

        rebuild_surface(
            self.fonts.font_box(),
            &mut self.tui_surface,
            &mut self.rendered,
            &mut self.wgpu_base,
            self.wgpu_post_process.as_mut(),
        );
    }

    /// Update the font-size used for rendering. This will cause a full repaint of
    /// the screen the next time [`WgpuBackend::flush`] is called.
    pub fn update_font_size(
        &mut self,
        new_font_size: u32,
    ) {
        self.tui_surface.dirty_rows.clear();
        self.tui_surface.dirty_cells.clear();
        self.fonts.set_size_px(new_font_size);
        self.wgpu_atlas
            .cached
            .update_font_box(self.fonts.font_box());

        rebuild_surface(
            self.fonts.font_box(),
            &mut self.tui_surface,
            &mut self.rendered,
            &mut self.wgpu_base,
            self.wgpu_post_process.as_mut(),
        );
    }

    /// Toggle blink.
    pub fn blink(&mut self) {
        let bounds = self.size().unwrap();

        flush_blink(
            bounds,
            &mut self.tui_surface,
            &self.rendered,
            &mut self.wgpu_vertices,
            &self.wgpu_base.queue,
        );

        render(
            self.window_size().expect("window_size"),
            self.fonts.font_box(),
            self.tui_surface.reset_bg,
            &self.wgpu_base,
            &self.wgpu_pipeline,
            self.wgpu_post_process.as_mut(),
            &self.wgpu_vertices,
        );
    }
}

/// Resize the rendering surface. This should be called e.g. to keep the
/// backend in sync with your window size.
fn rebuild_surface(
    font_box: FontBox,
    tui_surface: &mut TuiSurface,
    rendered: &mut Vec<Rendered>,
    wgpu_base: &mut WgpuBase,
    wgpu_post_process: &mut dyn PostProcessor,
) {
    let width = wgpu_base.surface_config.width;
    let height = wgpu_base.surface_config.height;
    wgpu_base
        .surface
        .configure(&wgpu_base.device, &wgpu_base.surface_config);

    let chars_wide = width / font_box.width;
    let chars_high = height / font_box.height;

    tui_surface.cells.clear();
    tui_surface.cell_remap.clear();
    tui_surface.fast_blinking.clear();
    tui_surface.slow_blinking.clear();
    // This always needs to be cleared because the surface is cleared when it is
    // resized. If we don't re-render the rows, we end up with a blank surface when
    // the resize is less than a character dimension.
    tui_surface.dirty_rows.clear();
    tui_surface.dirty_cells.clear();

    rendered.clear();

    wgpu_base.text_dest_view = build_wgpu_state(
        &wgpu_base.device,
        chars_wide * font_box.width,
        chars_high * font_box.height,
    );

    wgpu_post_process.resize(
        &wgpu_base.device,
        &wgpu_base.text_dest_view,
        &wgpu_base.surface_config,
    );
}

fn render(
    bounds: WindowSize,
    font_box: FontBox,
    reset_bg: Rgb,
    base: &WgpuBase,
    pipeline: &WgpuPipeline,
    post_process: &mut dyn PostProcessor,
    vertices: &WgpuVertices,
) {
    let mut encoder = base
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Draw Encoder"),
        });

    if !vertices.text_vertices.is_empty() {
        {
            let mut uniforms = base
                .queue
                .write_buffer_with(
                    &pipeline.text_screen_size_buffer,
                    0,
                    NonZeroU64::new(size_of::<[f32; 4]>() as u64).unwrap(),
                )
                .unwrap();
            uniforms.copy_from_slice(bytemuck::cast_slice(&[
                bounds.columns_rows.width as f32 * font_box.width as f32,
                bounds.columns_rows.height as f32 * font_box.height as f32,
                0.0,
                0.0,
            ]));
        }

        let bg_vertices = base.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Text Bg Vertices"),
            contents: bytemuck::cast_slice(&vertices.bg_vertices),
            usage: BufferUsages::VERTEX,
        });

        let fg_vertices = base.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Text Vertices"),
            contents: bytemuck::cast_slice(&vertices.text_vertices),
            usage: BufferUsages::VERTEX,
        });

        let indices = base.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Text Indices"),
            contents: bytemuck::cast_slice(&vertices.text_indices),
            usage: BufferUsages::INDEX,
        });

        {
            let mut text_render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Text Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &base.text_dest_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });

            text_render_pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);

            text_render_pass.set_pipeline(&pipeline.text_bg_compositor.pipeline);
            text_render_pass.set_bind_group(0, &pipeline.text_bg_compositor.fs_uniforms, &[]);
            text_render_pass.set_vertex_buffer(0, bg_vertices.slice(..));
            text_render_pass.draw_indexed(0..(vertices.bg_vertices.len() as u32 / 4) * 6, 0, 0..1);

            text_render_pass.set_pipeline(&pipeline.text_fg_compositor.pipeline);
            text_render_pass.set_bind_group(0, &pipeline.text_fg_compositor.fs_uniforms, &[]);
            text_render_pass.set_bind_group(1, &pipeline.text_fg_compositor.atlas_bindings, &[]);

            text_render_pass.set_vertex_buffer(0, fg_vertices.slice(..));
            text_render_pass.draw_indexed(
                0..(vertices.text_vertices.len() as u32 / 4) * 6,
                0,
                0..1,
            );
        }
    }

    let Some(texture) = base.surface.get_current_texture() else {
        return;
    };

    let bg_color_u32 = u32::from_le_bytes([reset_bg[0], reset_bg[1], reset_bg[2], 255]);

    post_process.process(
        bg_color_u32,
        &mut encoder,
        &base.queue,
        &base.text_dest_view,
        &base.surface_config,
        texture.get_view(),
    );

    base.queue.submit(Some(encoder.finish()));
    texture.present();
}

fn draw_tui(
    bounds: ratatui_core::layout::Size,
    content: &mut dyn Iterator<Item = (u16, u16, &'_ Cell)>,
    tui_surface: &mut TuiSurface,
    rendered: &mut Vec<Rendered>,
) {
    tui_surface
        .cells
        .resize(bounds.height as usize * bounds.width as usize, Cell::EMPTY);
    tui_surface
        .cell_remap
        .resize(bounds.height as usize * bounds.width as usize, 0);
    tui_surface
        .fast_blinking
        .resize(bounds.height as usize * bounds.width as usize, false);
    tui_surface
        .slow_blinking
        .resize(bounds.height as usize * bounds.width as usize, false);
    tui_surface.dirty_rows.resize(bounds.height as usize, true);
    tui_surface
        .dirty_cells
        .resize(bounds.height as usize * bounds.width as usize, true);

    rendered.resize_with(
        bounds.height as usize * bounds.width as usize,
        Rendered::default,
    );

    for (x, y, cell) in content {
        let offset = y as usize * bounds.width as usize;
        let index = offset + x as usize;

        tui_surface
            .fast_blinking
            .set(index, cell.modifier.contains(Modifier::RAPID_BLINK));
        tui_surface
            .slow_blinking
            .set(index, cell.modifier.contains(Modifier::SLOW_BLINK));

        for i in 1..tui_surface.cells[index].symbol().width() {
            tui_surface.cells[index + i] = ONE_CELL;
            tui_surface.dirty_cells.set(index + i, true);
        }
        tui_surface.cells[index] = cell.clone();
        tui_surface.dirty_cells.set(index, true);
        for i in 1..tui_surface.cells[index].symbol().width() {
            tui_surface.cells[index + i] = NULL_CELL;
            tui_surface.dirty_cells.set(index + i, true);
        }

        tui_surface.dirty_rows.set(y as usize, true);
    }
}

fn flush_tui(
    bounds: ratatui_core::layout::Size,
    fonts: &Fonts<'_>,
    tui_surface: &mut TuiSurface,
    rendered: &mut Vec<Rendered>,
    wgpu_atlas: &mut WgpuAtlas,
    queue: &Queue,
    //
    tmp_plan_cache: &mut PlanCache,
    tmp_rowbuf: &mut String,
    tmp_rowbuf_to_cell: &mut Vec<u16>,
    tmp_buffer: &mut UnicodeBuffer,
) {
    // always show cursor on flush.
    tui_surface.cursor_showing = true;
    // reset blink, removes flickering.
    tui_surface.cursor_blink = 0;

    for (row_idx, row_cells) in tui_surface.cells.chunks(bounds.width as usize).enumerate() {
        if !tui_surface.dirty_rows[row_idx] {
            continue;
        }

        let row_offset = row_idx.min(bounds.height as usize - 1) * bounds.width as usize;

        // This block concatenates the strings for the row into one string for bidi
        // resolution, then maps bytes for the string to their associated cell index. It
        // also maps the row's cell index to the font that can source all glyphs for
        // that cell.
        tmp_rowbuf.clear();
        tmp_rowbuf_to_cell.clear();

        let mut fontmap = Vec::with_capacity(tmp_rowbuf_to_cell.capacity());
        for (cell_idx, cell) in row_cells.iter().enumerate() {
            if !cell.skip {
                tmp_rowbuf.push_str(cell.symbol());
                tmp_rowbuf_to_cell.resize(
                    tmp_rowbuf_to_cell.len() + cell.symbol().len(),
                    cell_idx as u16,
                );
            }

            tui_surface.cell_remap[row_offset + cell_idx] = cell_idx as u16;

            fontmap.push(fonts.font_for_cell(cell));
        }

        // run text shaping
        let bidi = ParagraphBidiInfo::new(&tmp_rowbuf, None);
        let (levels, runs) = bidi.visual_runs(0..bidi.levels.len());

        // when bidi kicks in dirty_cell ceases to work...
        if runs.len() > 1 {
            for cell_idx in 0..bounds.width as usize {
                tui_surface.dirty_cells.set(row_offset + cell_idx, true);
            }
        }

        // rebuild from scratch
        for cell_idx in 0..bounds.width as usize {
            if tui_surface.dirty_cells[row_offset + cell_idx] {
                rendered[row_offset + cell_idx].clear();
            }
        }

        let (
            mut current_font,
            mut current_fake_bold,
            mut current_fake_italic,
            mut current_is_fallback,
        ) = fontmap[0];
        let mut current_level = Level::ltr();

        for (level, range) in runs.into_iter().map(|run| (levels[run.start], run)) {
            let bidi_run_chars = &tmp_rowbuf[range.clone()];
            let bidi_run_cells = &tmp_rowbuf_to_cell[range.clone()];
            let min_cell_idx = *bidi_run_cells.first().expect("first") as usize;
            let max_cell_idx = *bidi_run_cells.last().expect("last") as usize;

            for (ch_idx, ch) in bidi_run_chars.char_indices() {
                let cell_idx = bidi_run_cells[ch_idx] as usize;

                if ch.general_category() == GeneralCategory::Format {
                    // skip Format, no longer needed after bidi. probably?
                    continue;
                }

                let (font, fake_bold, fake_italic, is_fallback) = fontmap[cell_idx];
                if font.id() != current_font.id()
                    || current_fake_bold != fake_bold
                    || current_fake_italic != fake_italic
                    || current_is_fallback != is_fallback
                    || current_level != level
                {
                    let mut buffer = mem::take(tmp_buffer);

                    *tmp_buffer = shape(
                        row_idx,
                        row_cells,
                        &tui_surface.dirty_cells[row_offset..row_offset + bounds.width as usize],
                        &tui_surface.cell_remap[row_offset..row_offset + bounds.width as usize],
                        &tmp_rowbuf,
                        &tmp_rowbuf_to_cell,
                        shape_with_plan(
                            current_font.font(),
                            tmp_plan_cache.get(current_font, &mut buffer),
                            buffer,
                        ),
                        RenderedFont {
                            font_box: fonts.font_box(),
                            font: current_font,
                            fake_bold: current_fake_bold,
                            fake_italic: current_fake_italic,
                            is_fallback: current_is_fallback,
                        },
                        tui_surface.cursor_visible,
                        tui_surface.cursor,
                        &mut rendered[row_offset..row_offset + bounds.width as usize],
                        wgpu_atlas,
                        queue,
                    );
                }

                if level.is_rtl() {
                    // rtl flip visible cell index for this run.
                    let view_idx = (max_cell_idx - (cell_idx - min_cell_idx)) as u16;

                    if (cell_idx as u16, row_idx as u16) == tui_surface.cursor {
                        tui_surface.cursor_style = tui_surface.cursor_style.to_rtl();
                    }

                    tui_surface.cell_remap[row_offset + cell_idx] = view_idx;
                } else {
                    if (cell_idx as u16, row_idx as u16) == tui_surface.cursor {
                        tui_surface.cursor_style = tui_surface.cursor_style.to_ltr();
                    }
                }

                tmp_buffer.add(ch, (range.start + ch_idx) as u32);

                current_font = font;
                current_fake_bold = fake_bold;
                current_fake_italic = fake_italic;
                current_is_fallback = is_fallback;
                current_level = level;
            }
        }

        let mut buffer = mem::take(tmp_buffer);
        *tmp_buffer = shape(
            row_idx,
            row_cells,
            &tui_surface.dirty_cells[row_offset..row_offset + bounds.width as usize],
            &tui_surface.cell_remap[row_offset..row_offset + bounds.width as usize],
            &tmp_rowbuf,
            tmp_rowbuf_to_cell,
            shape_with_plan(
                current_font.font(),
                tmp_plan_cache.get(current_font, &mut buffer),
                buffer,
            ),
            RenderedFont {
                font_box: fonts.font_box(),
                font: current_font,
                fake_bold: current_fake_bold,
                fake_italic: current_fake_italic,
                is_fallback: current_is_fallback,
            },
            tui_surface.cursor_visible,
            tui_surface.cursor,
            &mut rendered[row_offset..row_offset + bounds.width as usize],
            wgpu_atlas,
            queue,
        );
    }
}

fn flush_blink(
    bounds: ratatui_core::layout::Size,
    tui_surface: &mut TuiSurface,
    rendered: &Vec<Rendered>,
    wgpu_vertices: &mut WgpuVertices,
    queue: &Queue,
) {
    wgpu_vertices.bg_vertices.clear();
    wgpu_vertices.text_vertices.clear();
    wgpu_vertices.text_indices.clear();

    tui_surface.blink = tui_surface.blink.wrapping_add(1);
    tui_surface.cursor_blink = tui_surface.cursor_blink.wrapping_add(1);

    if tui_surface.fast_blink_divisor != 0
        && tui_surface.blink % tui_surface.fast_blink_divisor == 0
    {
        tui_surface.fast_blink_showing = !tui_surface.fast_blink_showing;
    }
    if tui_surface.slow_blink_divisor != 0
        && tui_surface.blink % tui_surface.slow_blink_divisor == 0
    {
        tui_surface.slow_blink_showing = !tui_surface.slow_blink_showing;
    }
    if tui_surface.cursor_divisor != 0 && tui_surface.cursor_blink % tui_surface.cursor_divisor == 0
    {
        tui_surface.cursor_showing = !tui_surface.cursor_showing;
    }

    let mut index_offset = 0;

    let cell_indexes = tui_surface
        .fast_blinking
        .iter_ones()
        .chain(tui_surface.slow_blinking.iter_ones())
        .chain(iter::once(
            tui_surface.cursor.1 as usize * bounds.width as usize + tui_surface.cursor.0 as usize,
        ))
        .collect::<Vec<_>>();
    for index in cell_indexes {
        if let Some(to_render) = rendered.get(index) {
            append_rendered(&tui_surface, to_render, &mut index_offset, wgpu_vertices);
        }
    }

    queue.submit([]);
}

fn append_dirty_rows(
    bounds: ratatui_core::layout::Size,
    tui_surface: &mut TuiSurface,
    wgpu_post_process: &dyn PostProcessor,
    rendered: &Vec<Rendered>,
    wgpu_vertices: &mut WgpuVertices,
    queue: &Queue,
) {
    if wgpu_post_process.needs_update() || tui_surface.dirty_rows.any() {
        wgpu_vertices.bg_vertices.clear();
        wgpu_vertices.text_vertices.clear();
        wgpu_vertices.text_indices.clear();

        let mut index_offset = 0;
        for row in tui_surface.dirty_rows.iter_ones() {
            let row_index = row * bounds.width as usize;
            for col_index in 0..bounds.width as usize {
                let index = row_index + col_index;

                let to_render = &rendered[index];
                append_rendered(&tui_surface, to_render, &mut index_offset, wgpu_vertices);
            }
        }

        tui_surface
            .dirty_rows
            .iter_mut()
            .for_each(|mut v| *v = false);
        tui_surface
            .dirty_cells
            .iter_mut()
            .for_each(|mut v| *v = false);

        queue.submit([]);
    }
}

fn append_rendered(
    tui_surface: &TuiSurface,
    to_render: &Rendered,
    index_offset: &mut u32,
    vertices: &mut WgpuVertices,
) {
    for (
        (x, y, _),
        RenderInfo {
            cached,
            fg,
            bg,
            modifier,
            underline_pos_min,
            underline_pos_max,
            strikeout_pos_min,
            strikeout_pos_max,
            cursor_pos_min,
            cursor_pos_max,
        },
    ) in to_render.iter()
    {
        let alpha = if modifier.contains(Modifier::HIDDEN)
            | (modifier.contains(Modifier::RAPID_BLINK) && !tui_surface.fast_blink_showing)
            | (modifier.contains(Modifier::SLOW_BLINK) && !tui_surface.slow_blink_showing)
        {
            0
        } else if modifier.contains(Modifier::DIM) {
            127
        } else {
            255
        };

        let reverse = modifier.contains(Modifier::REVERSED);
        let fg_color = if reverse {
            tui_surface.colors.c2c(*bg, tui_surface.reset_bg)
        } else {
            tui_surface.colors.c2c(*fg, tui_surface.reset_fg)
        };
        let fg_color_u32: u32 = u32::from_le_bytes([fg_color[0], fg_color[1], fg_color[2], alpha]);

        let cursor_color_u32 = if tui_surface.cursor_color != ratatui_core::style::Color::Reset {
            let cur_color = tui_surface
                .colors
                .c2c(tui_surface.cursor_color, tui_surface.reset_fg);
            u32::from_le_bytes([cur_color[0], cur_color[1], cur_color[2], 99])
        } else {
            u32::from_le_bytes([fg_color[0], fg_color[1], fg_color[2], 99])
        };

        let bg_color = if reverse {
            tui_surface.colors.c2c(*fg, tui_surface.reset_fg)
        } else {
            tui_surface.colors.c2c(*bg, tui_surface.reset_bg)
        };
        let bg_color_u32 = u32::from_le_bytes([bg_color[0], bg_color[1], bg_color[2], 255]);

        let underline_pos =
            ((*underline_pos_min as u32 + cached.y) << 16) | (*underline_pos_max as u32 + cached.y);
        let strikeout_pos =
            ((*strikeout_pos_min as u32 + cached.y) << 16) | (*strikeout_pos_max as u32 + cached.y);

        let mut cursor_pos = 0x0000_0000;
        if tui_surface.cursor_visible
            && tui_surface.cursor_showing
            && cursor_pos_min != cursor_pos_max
        {
            match tui_surface.cursor_style {
                CursorStyle::Block => {
                    cursor_pos = 0x0002_0000 | cached.width << 8 | 0x0000_0000;
                    // horizontal
                }
                CursorStyle::Underscore => {
                    cursor_pos = 0x0003_0000
                        | (*cursor_pos_max as u32 + cached.y + 1) << 8
                        | (*cursor_pos_min as u32 + cached.y);
                }
                CursorStyle::BoldUnderscore => {
                    cursor_pos = 0x0003_0000
                        | (*cursor_pos_max as u32 + cached.y + 3) << 8
                        | (*cursor_pos_min as u32 + cached.y);
                }
                CursorStyle::Bar => {
                    let cursor_width = (*cursor_pos_max).abs_diff(*cursor_pos_min) as u32;
                    cursor_pos = 0x0002_0000 | (cursor_width + 1) << 8 | 0x0000_0000;
                }
                CursorStyle::BoldBar => {
                    let cursor_width = (*cursor_pos_max).abs_diff(*cursor_pos_min) as u32;
                    cursor_pos = 0x0002_0000 | (cursor_width + 3) << 8 | 0x0000_0000;
                }
                CursorStyle::RtlBar => {
                    let cursor_width = (*cursor_pos_max).abs_diff(*cursor_pos_min) as u32;
                    cursor_pos = 0x0002_0000
                        | cached.width << 8
                        | (cached.width.saturating_sub(cursor_width + 1));
                }
                CursorStyle::RtlBoldBar => {
                    let cursor_width = (*cursor_pos_max).abs_diff(*cursor_pos_min) as u32;
                    cursor_pos = 0x0002_0000
                        | cached.width << 8
                        | (cached.width.saturating_sub(cursor_width + 3))
                }
            }
        }

        vertices.text_indices.push([
            *index_offset,     // x, y
            *index_offset + 1, // x + w, y
            *index_offset + 2, // x, y + h
            *index_offset + 2, // x, y + h
            *index_offset + 3, // x + w, y + h
            *index_offset + 1, // x + w, y
        ]);
        *index_offset += 4;

        let x = *x as f32;
        let y = *y as f32;
        let width = cached.width as f32;
        let height = cached.height as f32;
        let uvx = cached.x as f32;
        let uvy = cached.y as f32;

        vertices.bg_vertices.push(TextBgVertexMember {
            vertex: [x, y],
            bg_color: bg_color_u32,
        });
        vertices.bg_vertices.push(TextBgVertexMember {
            vertex: [x + width, y],
            bg_color: bg_color_u32,
        });
        vertices.bg_vertices.push(TextBgVertexMember {
            vertex: [x, y + height],
            bg_color: bg_color_u32,
        });
        vertices.bg_vertices.push(TextBgVertexMember {
            vertex: [x + width, y + height],
            bg_color: bg_color_u32,
        });

        vertices.text_vertices.push(TextVertexMember {
            vertex: [x, y],
            uv: [uvx, uvy],
            uv_x0: uvx,
            fg_color: fg_color_u32,
            color_glyph: cached.color as u32,
            underline_pos,
            strikeout_pos,
            cursor_pos,
            cursor_color: cursor_color_u32,
        });
        vertices.text_vertices.push(TextVertexMember {
            vertex: [x + width, y],
            uv: [uvx + width, uvy],
            uv_x0: uvx,
            fg_color: fg_color_u32,
            color_glyph: cached.color as u32,
            underline_pos,
            strikeout_pos,
            cursor_pos,
            cursor_color: cursor_color_u32,
        });
        vertices.text_vertices.push(TextVertexMember {
            vertex: [x, y + height],
            uv: [uvx, uvy + height],
            uv_x0: uvx,
            fg_color: fg_color_u32,
            color_glyph: cached.color as u32,
            underline_pos,
            strikeout_pos,
            cursor_pos,
            cursor_color: cursor_color_u32,
        });
        vertices.text_vertices.push(TextVertexMember {
            vertex: [x + width, y + height],
            uv: [uvx + width, uvy + height],
            uv_x0: uvx,
            fg_color: fg_color_u32,
            color_glyph: cached.color as u32,
            underline_pos,
            strikeout_pos,
            cursor_pos,
            cursor_color: cursor_color_u32,
        });
    }
}

impl<'s> Backend for WgpuBackend<'_, 's> {
    fn draw<'a, I>(
        &mut self,
        mut content: I,
    ) -> std::io::Result<()>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        let bounds = self.size()?;
        draw_tui(
            bounds,
            &mut content,
            &mut self.tui_surface,
            &mut self.rendered,
        );
        Ok(())
    }

    fn hide_cursor(&mut self) -> std::io::Result<()> {
        self.tui_surface.cursor_visible = false;
        Ok(())
    }

    fn show_cursor(&mut self) -> std::io::Result<()> {
        self.tui_surface.cursor_visible = true;
        Ok(())
    }

    fn get_cursor_position(&mut self) -> std::io::Result<ratatui_core::layout::Position> {
        Ok(ratatui_core::layout::Position::new(
            self.tui_surface.cursor.0,
            self.tui_surface.cursor.1,
        ))
    }

    fn set_cursor_position<Pos: Into<ratatui_core::layout::Position>>(
        &mut self,
        position: Pos,
    ) -> std::io::Result<()> {
        let bounds = self.size()?;
        let pos = position.into();

        // old cursor
        self.tui_surface
            .dirty_rows
            .set(self.tui_surface.cursor.1 as usize, true);
        self.tui_surface.dirty_cells.set(
            self.tui_surface.cursor.1 as usize * bounds.width as usize
                + self.tui_surface.cursor.0 as usize,
            true,
        );

        self.tui_surface.cursor = (pos.x.min(bounds.width - 1), pos.y.min(bounds.height - 1));
        self.tui_surface
            .dirty_rows
            .set(self.tui_surface.cursor.1 as usize, true);
        self.tui_surface.dirty_cells.set(
            self.tui_surface.cursor.1 as usize * bounds.width as usize
                + self.tui_surface.cursor.0 as usize,
            true,
        );

        Ok(())
    }

    fn clear(&mut self) -> std::io::Result<()> {
        self.tui_surface.cells.clear();
        self.tui_surface.dirty_rows.clear();
        self.rendered.clear();
        self.tui_surface.fast_blinking.clear();
        self.tui_surface.slow_blinking.clear();
        self.tui_surface.cursor = (0, 0);

        Ok(())
    }

    fn size(&self) -> std::io::Result<ratatui_core::layout::Size> {
        let font_box = self.fonts.font_box();
        let width = self.wgpu_base.surface_config.width;
        let height = self.wgpu_base.surface_config.height;

        Ok(ratatui_core::layout::Size {
            width: (width / font_box.width) as u16,
            height: (height / font_box.height) as u16,
        })
    }

    fn window_size(&mut self) -> std::io::Result<WindowSize> {
        let font_box = self.fonts.font_box();
        let width = self.wgpu_base.surface_config.width;
        let height = self.wgpu_base.surface_config.height;

        Ok(WindowSize {
            columns_rows: ratatui_core::layout::Size {
                width: (width / font_box.width) as u16,
                height: (height / font_box.height) as u16,
            },
            pixels: ratatui_core::layout::Size {
                width: width as u16,
                height: height as u16,
            },
        })
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let bounds = self.size()?;

        flush_tui(
            bounds,
            &self.fonts,
            &mut self.tui_surface,
            &mut self.rendered,
            &mut self.wgpu_atlas,
            &self.wgpu_base.queue,
            &mut self.tmp_plan_cache,
            &mut self.tmp_rowbuf,
            &mut self.tmp_rowbuf_to_cell,
            &mut self.tmp_buffer,
        );

        append_dirty_rows(
            bounds,
            &mut self.tui_surface,
            self.wgpu_post_process.as_ref(),
            &self.rendered,
            &mut self.wgpu_vertices,
            &self.wgpu_base.queue,
        );

        render(
            self.window_size().expect("window_size"),
            self.fonts.font_box(),
            self.tui_surface.reset_bg,
            &self.wgpu_base,
            &self.wgpu_pipeline,
            self.wgpu_post_process.as_mut(),
            &self.wgpu_vertices,
        );

        Ok(())
    }

    fn clear_region(
        &mut self,
        clear_type: ClearType,
    ) -> std::io::Result<()> {
        let bounds = self.size()?;
        let line_start = self.tui_surface.cursor.1 as usize * bounds.width as usize;
        let idx = line_start + self.tui_surface.cursor.0 as usize;

        match clear_type {
            ClearType::All => self.clear(),
            ClearType::AfterCursor => {
                self.tui_surface.cells.truncate(idx + 1);
                Ok(())
            }
            ClearType::BeforeCursor => {
                self.tui_surface.cells[..idx].fill(Cell::EMPTY);
                Ok(())
            }
            ClearType::CurrentLine => {
                self.tui_surface.cells[line_start..line_start + bounds.width as usize]
                    .fill(Cell::EMPTY);
                Ok(())
            }
            ClearType::UntilNewLine => {
                let remain = (bounds.width - self.tui_surface.cursor.0) as usize;
                self.tui_surface.cells[idx..idx + remain].fill(Cell::EMPTY);
                Ok(())
            }
        }
    }

    type Error = std::io::Error;
}

// shape a part of one row.
//
// the glyphs come as a GlyphBuffer provided by the bidi algorithm.
// each glyph is mapped to a cell, which in turn might be mapped to a
// visible cell if there is any reordering during bidi.
//
// then the glyph is positioned and rendered if it is not already in the
// glyph-cache.
//
// Positioning of glyphs always restarts with each new cell.
// This ensures that the output is mostly cell-aligned and makes
// the final result more predictable.
fn shape(
    row_idx: usize,
    row: &[Cell],
    dirty_cells: &BitSlice,
    cell_remap: &[u16],
    buf_str: &str,
    buf_to_cell: &[u16],
    buffer: GlyphBuffer,
    font: RenderedFont<'_>,
    cursor_visible: bool,
    cursor: (u16, u16),
    rendered: &mut [Rendered],
    wgpu_atlas: &mut WgpuAtlas,
    queue: &Queue,
) -> UnicodeBuffer {
    let metrics = font.font();
    let advance_scale = font.font_box.scale;

    let mut x = 0;
    let mut default_chars_wide = 1;
    let mut chars_wide = 1;
    let mut last_cell_idx: Option<usize> = None;
    let mut last_advance = 0;
    for (info, position) in buffer
        .glyph_infos()
        .iter()
        .zip(buffer.glyph_positions().iter())
    {
        let cell_idx = buf_to_cell[info.cluster as usize] as usize;

        if !dirty_cells[cell_idx] {
            continue;
        }

        let cell = &row[cell_idx];
        let ch = buf_str[info.cluster as usize..]
            .chars()
            .next()
            .unwrap_or_default();

        // Every cell has it's defined position on the grid.
        // This position is used as a starting point from which
        // every glyph in the cell is positioned.
        let mut first_glyph = false;
        if last_cell_idx != Some(cell_idx) {
            x = cell_remap[cell_idx] as i32 * font.font_box.width as i32;
            default_chars_wide = ch.width().unwrap_or(1).max(1);
            chars_wide = default_chars_wide;
            last_advance = 0;
            first_glyph = true;
        } else {
            chars_wide = ch
                .width()
                .unwrap_or(default_chars_wide)
                .max(default_chars_wide);
        }

        // if we have a combining '.undef'. skip it completely.
        if last_cell_idx == Some(cell_idx) {
            if ch.general_category_group() == GeneralCategoryGroup::Mark && info.glyph_id == 0 {
                continue;
            }
        }

        last_cell_idx = Some(cell_idx);

        let glyph_advance = (position.x_advance as f32 * advance_scale) as i32;
        let glyph_offset = (position.x_offset as f32 * advance_scale) as i32;

        let basey = row_idx as i32 * font.font_box.height as i32
            + (position.y_offset as f32 * advance_scale) as i32;

        let mut basex = x + glyph_offset;
        // special case: combining glyphs with offset == 0 && advance == 0
        if glyph_advance == 0 && glyph_offset == 0 {
            basex -= last_advance;
        }
        if glyph_advance > 0 {
            last_advance = glyph_advance;
        }

        // advance
        x += glyph_advance;

        // This assumes that we only want to underline the first character in the
        // cluster, and that the remaining characters are all combining characters
        // which don't need an underline.
        let limit_modifiers = if first_glyph {
            Modifier::BOLD | Modifier::ITALIC | Modifier::UNDERLINED | Modifier::CROSSED_OUT
        } else {
            Modifier::BOLD | Modifier::ITALIC
        };

        let key = Key {
            style: cell.modifier.intersection(limit_modifiers),
            glyph: info.glyph_id,
            width: chars_wide as u8,
            font: font.font.id(),
        };

        let cached = wgpu_atlas.cached.get(
            &key,
            chars_wide as u32 * font.font_box.width,
            font.font_box.height,
        );

        let cursor_pos =
            if first_glyph && cursor_visible && (cell_idx as u16, row_idx as u16) == cursor {
                font.underline(font.font_box.height, cached.height)
            } else {
                (0, 0)
            };

        let underline_pos = if key.style.contains(Modifier::UNDERLINED) {
            font.underline(font.font_box.height, cached.height)
        } else {
            (0, 0)
        };
        let strikeout_pos = if key.style.contains(Modifier::CROSSED_OUT) {
            font.strikeout(font.font_box.height, cached.height)
        } else {
            (0, 0)
        };

        if cached.cached() {
            rendered[cell_idx].insert(
                (basex, basey, GlyphId(info.glyph_id as _)),
                RenderInfo {
                    cached: *cached,
                    fg: cell.fg,
                    bg: cell.bg,
                    modifier: cell.modifier,
                    underline_pos_min: underline_pos.0 as u16,
                    underline_pos_max: underline_pos.1 as u16,
                    strikeout_pos_min: strikeout_pos.0 as u16,
                    strikeout_pos_max: strikeout_pos.1 as u16,
                    cursor_pos_min: cursor_pos.0 as u16,
                    cursor_pos_max: cursor_pos.1 as u16,
                },
            );

            continue;
        }

        let is_emoji = ch.is_emoji_char();
        let (cached, image) = rasterize_glyph(
            cached,
            metrics,
            info,
            font.fake_italic & !is_emoji,
            font.fake_bold,
            advance_scale,
            font.font_box.ascender,
            is_emoji,
            font.is_fallback,
        );

        // remember colored flag for the glyph.
        wgpu_atlas.cached.update_colored(&key, cached.color);

        rendered[cell_idx].insert(
            (basex, basey, GlyphId(info.glyph_id as _)),
            RenderInfo {
                cached,
                fg: cell.fg,
                bg: cell.bg,
                modifier: cell.modifier,
                underline_pos_min: underline_pos.0 as u16,
                underline_pos_max: underline_pos.1 as u16,
                strikeout_pos_min: strikeout_pos.0 as u16,
                strikeout_pos_max: strikeout_pos.1 as u16,
                cursor_pos_min: cursor_pos.0 as u16,
                cursor_pos_max: cursor_pos.1 as u16,
            },
        );

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &wgpu_atlas.text_cache,
                mip_level: 0,
                origin: Origin3d {
                    x: cached.x,
                    y: cached.y,
                    z: 0,
                },
                aspect: TextureAspect::All,
            },
            bytemuck::cast_slice(&image),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(cached.width * size_of::<u32>() as u32),
                rows_per_image: Some(cached.height),
            },
            Extent3d {
                width: cached.width,
                height: cached.height,
                depth_or_array_layers: 1,
            },
        );
    }

    buffer.clear()
}

fn rasterize_glyph(
    cached: Entry,
    metrics: &rustybuzz::Face,
    info: &rustybuzz::GlyphInfo,
    fake_italic: bool,
    fake_bold: bool,
    advance_scale: f32,
    ascender: f32,
    emoji: bool,
    is_fallback: bool,
) -> (CacheRect, Vec<u32>) {
    let actual_width = metrics
        .glyph_hor_advance(GlyphId(info.glyph_id as _))
        .unwrap_or_default();
    let actual_width_px = if actual_width == 0 {
        cached.width
    } else {
        (actual_width as f32 * advance_scale) as u32
    };

    let computed_offset_x;
    let computed_offset_y;

    let scale;
    let scale_y;
    if is_fallback {
        // glyphs from a fallback font will probably not fit.
        // scale them down either vertically or horizontally, whatever fits.
        // then align them centered.
        // and later render them at the same baseline as the regular font.

        let mut rect_scale_x = cached.width as f32 / (actual_width as f32);
        let rect_scale_y = cached.height as f32 / metrics.height() as f32;

        if rect_scale_x / rect_scale_y > 1.0 {
            rect_scale_x = rect_scale_y;
            computed_offset_x = (cached.width as f32 - actual_width as f32 * rect_scale_y) / 2.0;
        } else {
            computed_offset_x = 0.0;
        }
        computed_offset_y = 0.0;

        scale = rect_scale_x * 2.0;
        scale_y = rect_scale_y * 2.0;
    } else if !metrics.is_monospaced() {
        let mut rect_scale_x = cached.width as f32 / (actual_width as f32);

        if rect_scale_x / advance_scale > 1.0 {
            rect_scale_x = advance_scale;
            computed_offset_x = (cached.width as f32 - actual_width as f32 * advance_scale) / 2.0;
        } else {
            computed_offset_x = 0.0;
        }
        computed_offset_y = 0.0;

        scale = rect_scale_x * 2.0;
        scale_y = advance_scale * 2.0;
    } else {
        // regular fonts will probably be from one font family and therefore have
        // more regular properties.
        let rect_scale = cached.width as f32 / actual_width_px as f32;

        // don't offset. font should fit.
        computed_offset_x = 0.0;
        computed_offset_y = 0.0;

        scale = rect_scale * advance_scale * 2.0;
        scale_y = scale;
    }

    let skew = if fake_italic {
        Transform::new(
            /* scale x */ 1.0,
            /* skew x */ 0.0,
            /* skew y */ -0.25,
            /* scale y */ 1.0,
            /* translate x */ -0.25 * cached.width as f32,
            /* translate y */ 0.0,
        )
    } else {
        Transform::default()
    };

    if info.glyph_id == 0 {
        // the glyph provided by the font is ugly most of the time.
        let width = cached.width as usize;
        let height = cached.height as usize;

        let mut image = vec![0u32; width * height];

        let mut target = DrawTarget::from_backing(width as i32, height as i32, &mut image[..]);

        let w1 = width as f32 * 0.33;
        let w2 = width as f32 * 0.67;
        let h1 = height as f32 * 0.33;
        let h2 = height as f32 * 0.67;

        let mut render = Outline::default();
        render.move_to(w1, h1);
        render.line_to(w2, h1);
        render.line_to(w2, h2);
        render.line_to(w1, h2);
        render.close();
        let path = render.finish();

        target.stroke(
            &path,
            &raqote::Source::Solid(SolidSource::from_unpremultiplied_argb(255, 255, 255, 255)),
            &StrokeStyle {
                width: 1.5,
                ..Default::default()
            },
            &DrawOptions::new(),
        );

        return (
            CacheRect {
                color: false,
                ..*cached
            },
            image,
        );
    }

    let mut image = vec![0u32; cached.width as usize * 2 * cached.height as usize * 2];
    let mut target = DrawTarget::from_backing(
        cached.width as i32 * 2,
        cached.height as i32 * 2,
        &mut image[..],
    );

    let mut painter = Painter::new(
        metrics,
        &mut target,
        skew,
        scale,
        ascender * advance_scale * 2.0 + computed_offset_y,
        computed_offset_x,
    );
    if metrics
        .paint_color_glyph(
            GlyphId(info.glyph_id as _),
            0,
            RgbaColor::new(255, 255, 255, 255),
            &mut painter,
        )
        .is_some()
    {
        let mut final_image = DrawTarget::new(cached.width as i32, cached.height as i32);
        final_image.draw_image_with_size_at(
            cached.width as f32,
            cached.height as f32,
            0.,
            0.,
            &raqote::Image {
                width: cached.width as i32 * 2,
                height: cached.height as i32 * 2,
                data: &image,
            },
            &DrawOptions {
                blend_mode: raqote::BlendMode::Src,
                antialias: raqote::AntialiasMode::None,
                ..Default::default()
            },
        );

        let mut final_image = final_image.into_vec();
        for argb in final_image.iter_mut() {
            let [a, r, g, b] = argb.to_be_bytes();
            *argb = u32::from_le_bytes([r, g, b, a]);
        }

        return (
            CacheRect {
                color: true,
                ..*cached
            },
            final_image,
        );
    }

    if let Some(raster) = metrics.glyph_raster_image(GlyphId(info.glyph_id as _), u16::MAX) {
        if let Some((cache_rect, image)) =
            extract_color_image(&mut image, raster, cached, advance_scale)
        {
            return (
                CacheRect {
                    color: true,
                    ..cache_rect
                },
                image,
            );
        }
    }

    let mut render = Outline::default();
    if let Some(bounds) = metrics.outline_glyph(GlyphId(info.glyph_id as _), &mut render) {
        let path = render.finish();

        // Some fonts return bounds that are entirely negative. I'm not sure why this
        // is, but it means the glyph won't render at all. We check for this here and
        // offset it if so. This seems to let those fonts render correctly.
        let x_off = if bounds.x_max < 0 {
            -bounds.x_min as f32
        } else {
            0.
        };
        let x_off = x_off * scale + computed_offset_x;
        let y_off = ascender * advance_scale * 2.0 + computed_offset_y;

        let mut target = DrawTarget::from_backing(
            cached.width as i32 * 2,
            cached.height as i32 * 2,
            &mut image[..],
        );
        target.set_transform(
            &Transform::scale(scale, -scale_y)
                .then(&skew)
                .then_translate((x_off, y_off).into()),
        );

        target.fill(
            &path,
            &raqote::Source::Solid(SolidSource::from_unpremultiplied_argb(255, 255, 255, 255)),
            &DrawOptions::default(),
        );

        if fake_bold {
            target.stroke(
                &path,
                &raqote::Source::Solid(SolidSource::from_unpremultiplied_argb(255, 255, 255, 255)),
                &StrokeStyle {
                    width: 1.5 / scale,
                    ..Default::default()
                },
                &DrawOptions::new(),
            );
        } else if emoji && is_fallback {
            // noto-emoji and open-moji need this.
            target.stroke(
                &path,
                &raqote::Source::Solid(SolidSource::from_unpremultiplied_argb(255, 255, 255, 255)),
                &StrokeStyle {
                    width: 1.0 / scale,
                    ..Default::default()
                },
                &DrawOptions::new(),
            );
        }

        let mut final_image = DrawTarget::new(cached.width as i32, cached.height as i32);
        final_image.draw_image_with_size_at(
            cached.width as f32,
            cached.height as f32,
            0.,
            0.,
            &raqote::Image {
                width: cached.width as i32 * 2,
                height: cached.height as i32 * 2,
                data: &image,
            },
            &DrawOptions {
                blend_mode: raqote::BlendMode::Src,
                antialias: raqote::AntialiasMode::None,
                ..Default::default()
            },
        );

        return (
            CacheRect {
                color: false,
                ..*cached
            },
            final_image.into_vec(),
        );
    }

    if let Some(raster) = metrics.glyph_raster_image(GlyphId(info.glyph_id as _), u16::MAX) {
        if raster.width != 0 && raster.height != 0 {
            if let Some((cached, image)) =
                extract_bw_image(&mut image, raster, cached, advance_scale)
            {
                return (
                    CacheRect {
                        color: false,
                        ..cached
                    },
                    image,
                );
            }
        }
    }

    (
        CacheRect {
            color: false,
            ..*cached
        },
        vec![0u32; cached.width as usize * cached.height as usize],
    )
}

fn extract_color_image(
    image: &mut Vec<u32>,
    raster: RasterGlyphImage,
    cached: Entry,
    scale: f32,
) -> Option<(CacheRect, Vec<u32>)> {
    match raster.format {
        RasterImageFormat::PNG => {
            #[cfg(feature = "png")]
            {
                let decoder = png::Decoder::new(std::io::Cursor::new(raster.data));
                if let Ok(mut info) = decoder.read_info() {
                    image.resize(
                        info.output_buffer_size().unwrap_or_default() / size_of::<u32>(),
                        0,
                    );
                    if info.next_frame(bytemuck::cast_slice_mut(image)).is_err() {
                        return None;
                    }

                    for rgba in image.iter_mut() {
                        let [r, g, b, a] = rgba.to_be_bytes();
                        *rgba = u32::from_be_bytes([a, r, g, b]);
                    }
                } else {
                    return None;
                }
            }
            #[cfg(not(feature = "png"))]
            return None;
        }
        RasterImageFormat::BitmapPremulBgra32 => {
            image.resize(raster.width as usize * raster.height as usize, 0);
            for (y, row) in raster.data.chunks(raster.width as usize * 4).enumerate() {
                for (x, pixel) in row.chunks(4).enumerate() {
                    let pixel: &[u8; 4] = pixel.try_into().expect("Invalid chunk size");
                    let [b, g, r, a] = *pixel;
                    let pixel = u32::from_be_bytes([
                        a,
                        r.saturating_mul(255 / a),
                        g.saturating_mul(255 / a),
                        b.saturating_mul(255 / a),
                    ]);
                    image[y * raster.width as usize + x] = pixel;
                }
            }
        }
        _ => return None,
    }

    let mut final_image = DrawTarget::new(cached.width as i32, cached.height as i32);
    final_image.draw_image_with_size_at(
        cached.width as f32,
        cached.height as f32,
        raster.x as f32 * scale,
        raster.y as f32 * scale,
        &raqote::Image {
            width: raster.width as i32,
            height: raster.height as i32,
            data: &*image,
        },
        &DrawOptions {
            blend_mode: raqote::BlendMode::Src,
            antialias: raqote::AntialiasMode::None,
            ..Default::default()
        },
    );

    let mut final_image = final_image.into_vec();
    for argb in final_image.iter_mut() {
        let [a, r, g, b] = argb.to_be_bytes();
        *argb = u32::from_le_bytes([r, g, b, a]);
    }

    Some((*cached, final_image))
}

fn extract_bw_image(
    image: &mut Vec<u32>,
    raster: RasterGlyphImage,
    cached: Entry,
    scale: f32,
) -> Option<(CacheRect, Vec<u32>)> {
    image.resize(raster.width as usize * raster.height as usize, 0);

    match raster.format {
        RasterImageFormat::BitmapMono => {
            from_gray_unpacked::<1, 2>(image, raster, LUT_1);
        }
        RasterImageFormat::BitmapMonoPacked => {
            from_gray_packed::<1, 2>(image, raster, LUT_1);
        }
        RasterImageFormat::BitmapGray2 => {
            from_gray_unpacked::<2, 4>(image, raster, LUT_2);
        }
        RasterImageFormat::BitmapGray2Packed => {
            from_gray_packed::<2, 4>(image, raster, LUT_2);
        }
        RasterImageFormat::BitmapGray4 => {
            from_gray_unpacked::<4, 16>(image, raster, LUT_4);
        }
        RasterImageFormat::BitmapGray4Packed => {
            from_gray_packed::<4, 16>(image, raster, LUT_4);
        }
        RasterImageFormat::BitmapGray8 => {
            for (byte, dst) in raster.data.iter().zip(image.iter_mut()) {
                *dst = u32::from_be_bytes([*byte, 255, 255, 255]);
            }
        }
        _ => return None,
    }

    let mut final_image = DrawTarget::new(cached.width as i32, cached.height as i32);
    final_image.draw_image_with_size_at(
        cached.width as f32,
        cached.height as f32,
        raster.x as f32 * scale,
        raster.y as f32 * scale,
        &raqote::Image {
            width: raster.width as i32,
            height: raster.height as i32,
            data: &*image,
        },
        &DrawOptions {
            blend_mode: raqote::BlendMode::Src,
            antialias: raqote::AntialiasMode::None,
            ..Default::default()
        },
    );

    let mut final_image = final_image.into_vec();
    for argb in final_image.iter_mut() {
        let [a, r, g, b] = argb.to_be_bytes();
        *argb = u32::from_le_bytes([r, g, b, a]);
    }

    Some((*cached, final_image))
}

fn from_gray_unpacked<const BITS: usize, const ENTRIES: usize>(
    image: &mut [u32],
    raster: RasterGlyphImage,
    steps: [u8; ENTRIES],
) {
    for (bits, dst) in raster
        .data
        .chunks((raster.width as usize / (8 / BITS)) + 1)
        .zip(image.chunks_mut(raster.width as usize))
    {
        let bits = BitSlice::<_, Lsb0>::from_slice(bits);
        for (bits, dst) in bits.chunks(BITS).zip(dst.iter_mut()) {
            let mut index = 0;
            for idx in bits.iter_ones() {
                index |= 1 << (BITS - idx - 1);
            }
            let value = steps[index as usize];
            *dst = u32::from_be_bytes([value, 255, 255, 255]);
        }
    }
}

fn from_gray_packed<const BITS: usize, const ENTRIES: usize>(
    image: &mut [u32],
    raster: RasterGlyphImage,
    steps: [u8; ENTRIES],
) {
    let bits = BitSlice::<_, Lsb0>::from_slice(raster.data);
    for (bits, dst) in bits.chunks(BITS).zip(image.iter_mut()) {
        let mut index = 0;
        for idx in bits.iter_ones() {
            index |= 1 << (BITS - idx - 1);
        }
        let value = steps[index as usize];
        *dst = u32::from_be_bytes([value, 255, 255, 255]);
    }
}

const LUT_1: [u8; 2] = [0, 255];
const LUT_2: [u8; 4] = [0, 255 / 3, 2 * (255 / 3), 255];
const LUT_4: [u8; 16] = [
    0,
    (255 / 15),
    2 * (255 / 15),
    3 * (255 / 15),
    4 * (255 / 15),
    5 * (255 / 15),
    6 * (255 / 15),
    7 * (255 / 15),
    8 * (255 / 15),
    9 * (255 / 15),
    10 * (255 / 15),
    11 * (255 / 15),
    12 * (255 / 15),
    13 * (255 / 15),
    14 * (255 / 15),
    255,
];

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use image::load_from_memory;
    use image::GenericImageView;
    use image::ImageBuffer;
    use image::Rgba;
    use ratatui_core::style::Color;
    use ratatui_core::style::Stylize;
    use ratatui_core::terminal::Terminal;
    use ratatui_core::text::Line;
    use ratatui_widgets::block::Block;
    use ratatui_widgets::paragraph::Paragraph;
    use rustybuzz::ttf_parser::RasterGlyphImage;
    use rustybuzz::ttf_parser::RasterImageFormat;
    use serial_test::serial;
    use wgpu::wgt::PollType;
    use wgpu::CommandEncoderDescriptor;
    use wgpu::Device;
    use wgpu::Extent3d;
    use wgpu::Queue;
    use wgpu::TextureFormat;

    use crate::backend::wgpu_backend::extract_bw_image;
    use crate::backend::wgpu_backend::LUT_2;
    use crate::backend::wgpu_backend::LUT_4;
    use crate::backend::RenderSurface;
    use crate::shaders::DefaultPostProcessorBuilder;
    use crate::utils::text_atlas::CacheRect;
    use crate::utils::text_atlas::Entry;
    use crate::Builder;
    use crate::Dimensions;
    use crate::Font;

    fn tex2buffer(
        device: &Device,
        queue: &Queue,
        surface: &RenderSurface,
    ) {
        let surface = surface.headless().expect("headless");
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            surface.texture.as_ref().unwrap().as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: surface.buffer.as_ref().unwrap(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(surface.buffer_width),
                    rows_per_image: Some(surface.height),
                },
            },
            Extent3d {
                width: surface.width,
                height: surface.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
    }

    #[test]
    #[serial]
    fn a_z() {
        let mut terminal = Terminal::new(
            futures_lite::future::block_on(
                Builder::<DefaultPostProcessorBuilder>::from_font(
                    Font::new(include_bytes!("fonts/CascadiaMono-Regular.ttf"))
                        .expect("Invalid font file"),
                )
                .with_width_and_height(Dimensions {
                    width: NonZeroU32::new(512).unwrap(),
                    height: NonZeroU32::new(72).unwrap(),
                })
                .build_headless(),
            )
            .unwrap(),
        )
        .unwrap();

        terminal
            .draw(|f: &mut ratatui_core::terminal::Frame| {
                let block = Block::bordered();
                let area = block.inner(f.area());
                f.render_widget(block, f.area());
                f.render_widget(Paragraph::new("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), area);
            })
            .unwrap();

        let surface = &terminal.backend().wgpu_base.surface;
        tex2buffer(
            &terminal.backend().wgpu_base.device,
            &terminal.backend().wgpu_base.queue,
            surface,
        );
        let surface = surface.headless().expect("headless");
        {
            let buffer = surface.buffer.as_ref().unwrap().slice(..);

            let (send, recv) = oneshot::channel();
            buffer.map_async(wgpu::MapMode::Read, move |data| {
                send.send(data).unwrap();
            });
            terminal
                .backend()
                .wgpu_base
                .device
                .poll(PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            recv.recv().unwrap().unwrap();

            let data = buffer.get_mapped_range();
            let image =
                ImageBuffer::<Rgba<u8>, _>::from_raw(surface.width, surface.height, data).unwrap();

            let pixels = image.pixels().copied().collect::<Vec<_>>();
            let golden = load_from_memory(include_bytes!("goldens/a_z.png")).unwrap();
            let golden_pixels = golden.pixels().map(|(_, _, px)| px).collect::<Vec<_>>();

            assert!(
                pixels == golden_pixels,
                "Rendered image differs from golden"
            );
        }

        surface.buffer.as_ref().unwrap().unmap();
    }

    #[test]
    #[serial]
    fn arabic() {
        let mut terminal = Terminal::new(
            futures_lite::future::block_on(
                Builder::<DefaultPostProcessorBuilder>::from_font(
                    Font::new(include_bytes!("fonts/CascadiaMono-Regular.ttf"))
                        .expect("Invalid font file"),
                )
                .with_width_and_height(Dimensions {
                    width: NonZeroU32::new(256).unwrap(),
                    height: NonZeroU32::new(72).unwrap(),
                })
                .build_headless(),
            )
            .unwrap(),
        )
        .unwrap();

        terminal
            .draw(|f: &mut ratatui_core::terminal::Frame| {
                let block = Block::bordered();
                let area = block.inner(f.area());
                f.render_widget(block, f.area());
                f.render_widget(Paragraph::new("مرحبا بالعالم"), area);
            })
            .unwrap();

        let surface = &terminal.backend().wgpu_base.surface;
        tex2buffer(
            &terminal.backend().wgpu_base.device,
            &terminal.backend().wgpu_base.queue,
            surface,
        );
        let surface = surface.headless().expect("headless");
        {
            let buffer = surface.buffer.as_ref().unwrap().slice(..);

            let (send, recv) = oneshot::channel();
            buffer.map_async(wgpu::MapMode::Read, move |data| {
                send.send(data).unwrap();
            });
            terminal
                .backend()
                .wgpu_base
                .device
                .poll(PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            recv.recv().unwrap().unwrap();

            let data = buffer.get_mapped_range();
            let image =
                ImageBuffer::<Rgba<u8>, _>::from_raw(surface.width, surface.height, data).unwrap();

            let pixels = image.pixels().copied().collect::<Vec<_>>();
            let golden = load_from_memory(include_bytes!("goldens/arabic.png")).unwrap();
            let golden_pixels = golden.pixels().map(|(_, _, px)| px).collect::<Vec<_>>();

            assert!(
                pixels == golden_pixels,
                "Rendered image differs from golden"
            );
        }

        surface.buffer.as_ref().unwrap().unmap();
    }

    #[test]
    #[serial]
    fn really_wide() {
        let mut terminal = Terminal::new(
            futures_lite::future::block_on(
                Builder::<DefaultPostProcessorBuilder>::from_font(
                    Font::new(include_bytes!("fonts/Fairfax.ttf")).expect("Invalid font file"),
                )
                .with_width_and_height(Dimensions {
                    width: NonZeroU32::new(512).unwrap(),
                    height: NonZeroU32::new(72).unwrap(),
                })
                .build_headless(),
            )
            .unwrap(),
        )
        .unwrap();

        terminal
            .draw(|f: &mut ratatui_core::terminal::Frame| {
                let block = Block::bordered();
                let area = block.inner(f.area());
                f.render_widget(block, f.area());
                f.render_widget(Paragraph::new("Ｈｅｌｌｏ, ｗｏｒｌｄ!"), area);
            })
            .unwrap();

        let surface = &terminal.backend().wgpu_base.surface;
        tex2buffer(
            &terminal.backend().wgpu_base.device,
            &terminal.backend().wgpu_base.queue,
            surface,
        );
        let surface = surface.headless().expect("headless");
        {
            let buffer = surface.buffer.as_ref().unwrap().slice(..);

            let (send, recv) = oneshot::channel();
            buffer.map_async(wgpu::MapMode::Read, move |data| {
                send.send(data).unwrap();
            });
            terminal
                .backend()
                .wgpu_base
                .device
                .poll(PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            recv.recv().unwrap().unwrap();

            let data = buffer.get_mapped_range();
            let image =
                ImageBuffer::<Rgba<u8>, _>::from_raw(surface.width, surface.height, data).unwrap();

            let pixels = image.pixels().copied().collect::<Vec<_>>();
            let golden = load_from_memory(include_bytes!("goldens/really_wide.png")).unwrap();
            let golden_pixels = golden.pixels().map(|(_, _, px)| px).collect::<Vec<_>>();

            assert!(
                pixels == golden_pixels,
                "Rendered image differs from golden"
            );
        }

        surface.buffer.as_ref().unwrap().unmap();
    }

    #[test]
    #[serial]
    fn mixed() {
        let mut terminal = Terminal::new(
            futures_lite::future::block_on(
                Builder::<DefaultPostProcessorBuilder>::from_font(
                    Font::new(include_bytes!("fonts/CascadiaMono-Regular.ttf"))
                        .expect("Invalid font file"),
                )
                .with_width_and_height(Dimensions {
                    width: NonZeroU32::new(512).unwrap(),
                    height: NonZeroU32::new(72).unwrap(),
                })
                .build_headless(),
            )
            .unwrap(),
        )
        .unwrap();

        terminal
            .draw(|f: &mut ratatui_core::terminal::Frame| {
                let block = Block::bordered();
                let area = block.inner(f.area());
                f.render_widget(block, f.area());
                f.render_widget(
                    Paragraph::new("Hello World! مرحبا بالعالم 0123456789000000000"),
                    area,
                );
            })
            .unwrap();

        let surface = &terminal.backend().wgpu_base.surface;
        tex2buffer(
            &terminal.backend().wgpu_base.device,
            &terminal.backend().wgpu_base.queue,
            surface,
        );
        let surface = surface.headless().expect("headless");
        {
            let buffer = surface.buffer.as_ref().unwrap().slice(..);

            let (send, recv) = oneshot::channel();
            buffer.map_async(wgpu::MapMode::Read, move |data| {
                send.send(data).unwrap();
            });
            terminal
                .backend()
                .wgpu_base
                .device
                .poll(PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            recv.recv().unwrap().unwrap();

            let data = buffer.get_mapped_range();
            let image =
                ImageBuffer::<Rgba<u8>, _>::from_raw(surface.width, surface.height, data).unwrap();

            let pixels = image.pixels().copied().collect::<Vec<_>>();
            let golden = load_from_memory(include_bytes!("goldens/mixed.png")).unwrap();
            let golden_pixels = golden.pixels().map(|(_, _, px)| px).collect::<Vec<_>>();

            assert!(
                pixels == golden_pixels,
                "Rendered image differs from golden"
            );
        }

        surface.buffer.as_ref().unwrap().unmap();
    }

    #[test]
    #[serial]
    fn mixed_colors() {
        let mut terminal = Terminal::new(
            futures_lite::future::block_on(
                Builder::<DefaultPostProcessorBuilder>::from_font(
                    Font::new(include_bytes!("fonts/CascadiaMono-Regular.ttf"))
                        .expect("Invalid font file"),
                )
                .with_width_and_height(Dimensions {
                    width: NonZeroU32::new(512).unwrap(),
                    height: NonZeroU32::new(72).unwrap(),
                })
                .build_headless(),
            )
            .unwrap(),
        )
        .unwrap();

        terminal
            .draw(|f: &mut ratatui_core::terminal::Frame| {
                let block = Block::bordered();
                let area = block.inner(f.area());
                f.render_widget(block, f.area());
                f.render_widget(
                    Paragraph::new(Line::from(vec![
                        "Hello World!".green(),
                        "مرحبا بالعالم".blue(),
                        "0123456789".dim(),
                    ])),
                    area,
                );
            })
            .unwrap();

        let surface = &terminal.backend().wgpu_base.surface;
        tex2buffer(
            &terminal.backend().wgpu_base.device,
            &terminal.backend().wgpu_base.queue,
            surface,
        );
        let surface = surface.headless().expect("headless");
        {
            let buffer = surface.buffer.as_ref().unwrap().slice(..);

            let (send, recv) = oneshot::channel();
            buffer.map_async(wgpu::MapMode::Read, move |data| {
                send.send(data).unwrap();
            });
            terminal
                .backend()
                .wgpu_base
                .device
                .poll(PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            recv.recv().unwrap().unwrap();

            let data = buffer.get_mapped_range();
            let image =
                ImageBuffer::<Rgba<u8>, _>::from_raw(surface.width, surface.height, data).unwrap();

            let pixels = image.pixels().copied().collect::<Vec<_>>();
            let golden = load_from_memory(include_bytes!("goldens/mixed_colors.png")).unwrap();
            let golden_pixels = golden.pixels().map(|(_, _, px)| px).collect::<Vec<_>>();

            assert!(
                pixels == golden_pixels,
                "Rendered image differs from golden"
            );
        }

        surface.buffer.as_ref().unwrap().unmap();
    }

    #[test]
    #[serial]
    fn overlap() {
        let mut terminal = Terminal::new(
            futures_lite::future::block_on(
                Builder::<DefaultPostProcessorBuilder>::from_font(
                    Font::new(include_bytes!("fonts/Fairfax.ttf")).expect("Invalid font file"),
                )
                .with_width_and_height(Dimensions {
                    width: NonZeroU32::new(256).unwrap(),
                    height: NonZeroU32::new(72).unwrap(),
                })
                .build_headless(),
            )
            .unwrap(),
        )
        .unwrap();

        terminal
            .draw(|f: &mut ratatui_core::terminal::Frame| {
                let block = Block::bordered();
                let area = block.inner(f.area());
                f.render_widget(block, f.area());
                f.render_widget(Paragraph::new("H̴̢͕̠͖͇̻͓̙̞͔͕͓̰͋͛͂̃̌͂͆͜͠".underlined()), area);
            })
            .unwrap();

        let surface = &terminal.backend().wgpu_base.surface;
        tex2buffer(
            &terminal.backend().wgpu_base.device,
            &terminal.backend().wgpu_base.queue,
            surface,
        );
        let surface = surface.headless().expect("headless");
        {
            let buffer = surface.buffer.as_ref().unwrap().slice(..);

            let (send, recv) = oneshot::channel();
            buffer.map_async(wgpu::MapMode::Read, move |data| {
                send.send(data).unwrap();
            });
            terminal
                .backend()
                .wgpu_base
                .device
                .poll(PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            recv.recv().unwrap().unwrap();

            let data = buffer.get_mapped_range();
            let image =
                ImageBuffer::<Rgba<u8>, _>::from_raw(surface.width, surface.height, data).unwrap();

            let pixels = image.pixels().copied().collect::<Vec<_>>();
            let golden = load_from_memory(include_bytes!("goldens/overlap_initial.png")).unwrap();
            let golden_pixels = golden.pixels().map(|(_, _, px)| px).collect::<Vec<_>>();

            assert!(
                pixels == golden_pixels,
                "Rendered image differs from golden"
            );
        }
        surface.buffer.as_ref().unwrap().unmap();

        terminal
            .draw(|f: &mut ratatui_core::terminal::Frame| {
                let block = Block::bordered();
                let area = block.inner(f.area());
                f.render_widget(block, f.area());
                f.render_widget(Paragraph::new("H".underlined()), area);
            })
            .unwrap();

        let surface = &terminal.backend().wgpu_base.surface;
        tex2buffer(
            &terminal.backend().wgpu_base.device,
            &terminal.backend().wgpu_base.queue,
            surface,
        );
        let surface = surface.headless().expect("headless");
        {
            let buffer = surface.buffer.as_ref().unwrap().slice(..);

            let (send, recv) = oneshot::channel();
            buffer.map_async(wgpu::MapMode::Read, move |data| {
                send.send(data).unwrap();
            });
            terminal
                .backend()
                .wgpu_base
                .device
                .poll(PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            recv.recv().unwrap().unwrap();

            let data = buffer.get_mapped_range();
            let image =
                ImageBuffer::<Rgba<u8>, _>::from_raw(surface.width, surface.height, data).unwrap();

            let pixels = image.pixels().copied().collect::<Vec<_>>();
            let golden = load_from_memory(include_bytes!("goldens/overlap_post.png")).unwrap();
            let golden_pixels = golden.pixels().map(|(_, _, px)| px).collect::<Vec<_>>();

            assert!(
                pixels == golden_pixels,
                "Rendered image differs from golden"
            );
        }

        surface.buffer.as_ref().unwrap().unmap();
    }

    #[test]
    #[serial]
    fn overlap_colors() {
        let mut terminal = Terminal::new(
            futures_lite::future::block_on(
                Builder::<DefaultPostProcessorBuilder>::from_font(
                    Font::new(include_bytes!("fonts/Fairfax.ttf")).expect("Invalid font file"),
                )
                .with_width_and_height(Dimensions {
                    width: NonZeroU32::new(256).unwrap(),
                    height: NonZeroU32::new(72).unwrap(),
                })
                .build_headless(),
            )
            .unwrap(),
        )
        .unwrap();

        terminal
            .draw(|f: &mut ratatui_core::terminal::Frame| {
                let block = Block::bordered();
                let area = block.inner(f.area());
                f.render_widget(block, f.area());
                f.render_widget(Paragraph::new("H̴̢͕̠͖͇̻͓̙̞͔͕͓̰͋͛͂̃̌͂͆͜͠".blue().on_red().underlined()), area);
            })
            .unwrap();

        let surface = &terminal.backend().wgpu_base.surface;
        tex2buffer(
            &terminal.backend().wgpu_base.device,
            &terminal.backend().wgpu_base.queue,
            surface,
        );
        let surface = surface.headless().expect("headless");
        {
            let buffer = surface.buffer.as_ref().unwrap().slice(..);

            let (send, recv) = oneshot::channel();
            buffer.map_async(wgpu::MapMode::Read, move |data| {
                send.send(data).unwrap();
            });
            terminal
                .backend()
                .wgpu_base
                .device
                .poll(PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            recv.recv().unwrap().unwrap();

            let data = buffer.get_mapped_range();
            let image =
                ImageBuffer::<Rgba<u8>, _>::from_raw(surface.width, surface.height, data).unwrap();

            let pixels = image.pixels().copied().collect::<Vec<_>>();
            let golden = load_from_memory(include_bytes!("goldens/overlap_colors.png")).unwrap();
            let golden_pixels = golden.pixels().map(|(_, _, px)| px).collect::<Vec<_>>();

            assert!(
                pixels == golden_pixels,
                "Rendered image differs from golden"
            );
        }
        surface.buffer.as_ref().unwrap().unmap();
    }

    #[test]
    #[serial]
    fn rgb_conversion() {
        let mut terminal = Terminal::new(
            futures_lite::future::block_on(
                Builder::<DefaultPostProcessorBuilder>::from_font(
                    Font::new(include_bytes!("fonts/Fairfax.ttf")).expect("Invalid font file"),
                )
                .with_width_and_height(Dimensions {
                    width: NonZeroU32::new(256).unwrap(),
                    height: NonZeroU32::new(72).unwrap(),
                })
                .with_bg_color(Color::Rgb(0x1E, 0x23, 0x26))
                .with_fg_color(Color::White)
                .build_headless_with_format(TextureFormat::Rgba8Unorm),
            )
            .unwrap(),
        )
        .unwrap();

        terminal
            .draw(|f: &mut ratatui_core::terminal::Frame| {
                let block = Block::bordered();
                let area = block.inner(f.area());
                f.render_widget(block, f.area());
                f.render_widget(Paragraph::new("TEST"), area);
            })
            .unwrap();

        let surface = &terminal.backend().wgpu_base.surface;
        tex2buffer(
            &terminal.backend().wgpu_base.device,
            &terminal.backend().wgpu_base.queue,
            surface,
        );
        let surface = surface.headless().expect("headless");
        {
            let buffer = surface.buffer.as_ref().unwrap().slice(..);

            let (send, recv) = oneshot::channel();
            buffer.map_async(wgpu::MapMode::Read, move |data| {
                send.send(data).unwrap();
            });
            terminal
                .backend()
                .wgpu_base
                .device
                .poll(PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            recv.recv().unwrap().unwrap();

            let data = buffer.get_mapped_range();
            let image =
                ImageBuffer::<Rgba<u8>, _>::from_raw(surface.width, surface.height, data).unwrap();

            let pixels = image.pixels().copied().collect::<Vec<_>>();
            let golden = load_from_memory(include_bytes!("goldens/rgb_conversion.png")).unwrap();
            let golden_pixels = golden.pixels().map(|(_, _, px)| px).collect::<Vec<_>>();

            assert!(
                pixels == golden_pixels,
                "Rendered image differs from golden"
            );
        }
        surface.buffer.as_ref().unwrap().unmap();
    }

    #[test]
    #[serial]
    fn srgb_conversion() {
        let mut terminal = Terminal::new(
            futures_lite::future::block_on(
                Builder::<DefaultPostProcessorBuilder>::from_font(
                    Font::new(include_bytes!("fonts/Fairfax.ttf")).expect("Invalid font file"),
                )
                .with_width_and_height(Dimensions {
                    width: NonZeroU32::new(256).unwrap(),
                    height: NonZeroU32::new(72).unwrap(),
                })
                .with_bg_color(Color::Rgb(0x1E, 0x23, 0x26))
                .with_fg_color(Color::White)
                .build_headless_with_format(TextureFormat::Rgba8UnormSrgb),
            )
            .unwrap(),
        )
        .unwrap();

        terminal
            .draw(|f: &mut ratatui_core::terminal::Frame| {
                let block = Block::bordered();
                let area = block.inner(f.area());
                f.render_widget(block, f.area());
                f.render_widget(Paragraph::new("TEST"), area);
            })
            .unwrap();

        let surface = &terminal.backend().wgpu_base.surface;
        tex2buffer(
            &terminal.backend().wgpu_base.device,
            &terminal.backend().wgpu_base.queue,
            surface,
        );
        let surface = surface.headless().expect("headless");
        {
            let buffer = surface.buffer.as_ref().unwrap().slice(..);

            let (send, recv) = oneshot::channel();
            buffer.map_async(wgpu::MapMode::Read, move |data| {
                send.send(data).unwrap();
            });
            terminal
                .backend()
                .wgpu_base
                .device
                .poll(PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            recv.recv().unwrap().unwrap();

            let data = buffer.get_mapped_range();
            let image =
                ImageBuffer::<Rgba<u8>, _>::from_raw(surface.width, surface.height, data).unwrap();

            let pixels = image.pixels().copied().collect::<Vec<_>>();
            let golden = load_from_memory(include_bytes!("goldens/srgb_conversion.png")).unwrap();
            let golden_pixels = golden.pixels().map(|(_, _, px)| px).collect::<Vec<_>>();

            assert!(
                pixels == golden_pixels,
                "Rendered image differs from golden"
            );
        }
        surface.buffer.as_ref().unwrap().unmap();
    }

    #[test]
    #[cfg(feature = "png")]
    fn png() {
        use crate::backend::wgpu_backend::extract_color_image;
        let golden = load_from_memory(include_bytes!("goldens/A.png")).unwrap();
        let raster = RasterGlyphImage {
            x: 0,
            y: 0,
            width: golden.width() as u16,
            height: golden.height() as u16,
            pixels_per_em: 0,
            format: RasterImageFormat::PNG,
            data: include_bytes!("goldens/A.png"),
        };

        let mut image = vec![];
        let extracted = extract_color_image(
            &mut image,
            raster,
            Entry::Cached(CacheRect {
                color: false,
                x: 0,
                y: 0,
                width: golden.width(),
                height: golden.height(),
            }),
            1.0,
        )
        .expect("Didn't extract png")
        .1;

        for (l, r) in bytemuck::cast_slice::<_, u8>(&extracted)
            .chunks(4)
            .zip(golden.pixels())
        {
            let [r, g, b, a] = r.2 .0;
            assert_eq!(l, [a, b, g, r]);
        }
    }

    #[test]
    fn bgra() {
        use crate::backend::wgpu_backend::extract_color_image;

        const BLUE: u8 = 2;
        const GREEN: u8 = 4;
        const RED: u8 = 8;
        const ALPHA: u8 = 127;
        let data = [BLUE, GREEN, RED, ALPHA];
        let raster = RasterGlyphImage {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
            pixels_per_em: 0,
            format: RasterImageFormat::BitmapPremulBgra32,
            data: &data,
        };

        let mut image = vec![];
        let extracted = extract_color_image(
            &mut image,
            raster,
            Entry::Cached(CacheRect {
                color: false,
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            }),
            1.0,
        )
        .expect("Didn't extract bgra")
        .1;

        assert_eq!(
            bytemuck::bytes_of(&extracted[0]),
            [RED * 2, GREEN * 2, BLUE * 2, ALPHA]
        );
    }

    #[test]
    fn bmp1() {
        let data0 = 0b1000_0001;
        let data1 = 0b0001_1000;
        let raster = RasterGlyphImage {
            x: 0,
            y: 0,
            width: 4,
            height: 2,
            pixels_per_em: 0,
            format: RasterImageFormat::BitmapMono,
            data: &[data0, data1],
        };

        let mut image = vec![];
        let extracted = extract_bw_image(
            &mut image,
            raster,
            Entry::Cached(CacheRect {
                color: false,
                x: 0,
                y: 0,
                width: 4,
                height: 2,
            }),
            1.0,
        )
        .expect("Didn't extract bmp1")
        .1;

        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&extracted),
            bytemuck::cast_slice(&[
                [
                    [255u8, 255, 255, 255,],
                    [255, 255, 255, 0,],
                    [255, 255, 255, 0,],
                    [255, 255, 255, 0,],
                ],
                [
                    [255, 255, 255, 0,],
                    [255, 255, 255, 0,],
                    [255, 255, 255, 0,],
                    [255, 255, 255, 255,],
                ],
            ])
        );
    }

    #[test]
    fn bmp1_packed() {
        let data0 = 0b1000_0001;
        let data1 = 0b0001_1000;
        let raster = RasterGlyphImage {
            x: 0,
            y: 0,
            width: 8,
            height: 2,
            pixels_per_em: 0,
            format: RasterImageFormat::BitmapMonoPacked,
            data: &[data0, data1],
        };

        let mut image = vec![];
        let extracted = extract_bw_image(
            &mut image,
            raster,
            Entry::Cached(CacheRect {
                color: false,
                x: 0,
                y: 0,
                width: 8,
                height: 2,
            }),
            1.0,
        )
        .expect("Didn't extract bmp1")
        .1;

        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&extracted),
            bytemuck::cast_slice(&[
                [
                    [255u8, 255, 255, 255,],
                    [255, 255, 255, 0,],
                    [255, 255, 255, 0,],
                    [255, 255, 255, 0,],
                    [255, 255, 255, 0,],
                    [255, 255, 255, 0,],
                    [255, 255, 255, 0,],
                    [255, 255, 255, 255,],
                ],
                [
                    [255, 255, 255, 0,],
                    [255, 255, 255, 0,],
                    [255, 255, 255, 0,],
                    [255, 255, 255, 255,],
                    [255, 255, 255, 255,],
                    [255, 255, 255, 0,],
                    [255, 255, 255, 0,],
                    [255, 255, 255, 0,],
                ],
            ])
        );
    }

    #[test]
    fn bmp2() {
        let data0 = 0b1010_1101u8.reverse_bits();
        let data1 = 0b0000_1000u8.reverse_bits();
        let data2 = 0b0111_1010u8.reverse_bits();
        let data3 = 0b0000_1000u8.reverse_bits();
        let raster = RasterGlyphImage {
            x: 0,
            y: 0,
            width: 6,
            height: 2,
            pixels_per_em: 0,
            format: RasterImageFormat::BitmapGray2,
            data: &[data0, data1, data2, data3],
        };

        let mut image = vec![];
        let extracted = extract_bw_image(
            &mut image,
            raster,
            Entry::Cached(CacheRect {
                color: false,
                x: 0,
                y: 0,
                width: 6,
                height: 2,
            }),
            1.0,
        )
        .expect("Didn't extract bmp2")
        .1;

        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&extracted),
            bytemuck::cast_slice(&[
                [
                    [255, 255, 255, LUT_2[0b10],],
                    [255, 255, 255, LUT_2[0b10],],
                    [255, 255, 255, LUT_2[0b11],],
                    [255, 255, 255, LUT_2[0b01],],
                    [255, 255, 255, LUT_2[0b00],],
                    [255, 255, 255, LUT_2[0b00],],
                ],
                [
                    [255, 255, 255, LUT_2[0b01],],
                    [255, 255, 255, LUT_2[0b11],],
                    [255, 255, 255, LUT_2[0b10],],
                    [255, 255, 255, LUT_2[0b10],],
                    [255, 255, 255, LUT_2[0b00],],
                    [255, 255, 255, LUT_2[0b00],],
                ],
            ])
        );
    }

    #[test]
    fn bmp2_packed() {
        let data0 = 0b1010_1101u8.reverse_bits();
        let data1 = 0b0000_1000u8.reverse_bits();
        let data2 = 0b0111_1010u8.reverse_bits();
        let data3 = 0b0000_1000u8.reverse_bits();
        let data4 = 0b0000_1000u8.reverse_bits();
        let data5 = 0b0000_1000u8.reverse_bits();
        let raster = RasterGlyphImage {
            x: 0,
            y: 0,
            width: 6,
            height: 4,
            pixels_per_em: 0,
            format: RasterImageFormat::BitmapGray2Packed,
            data: &[data0, data1, data2, data3, data4, data5],
        };

        let mut image = vec![];
        let extracted = extract_bw_image(
            &mut image,
            raster,
            Entry::Cached(CacheRect {
                color: false,
                x: 0,
                y: 0,
                width: 6,
                height: 4,
            }),
            1.0,
        )
        .expect("Didn't extract bmp2")
        .1;

        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&extracted),
            bytemuck::cast_slice(&[
                [
                    [255, 255, 255, LUT_2[0b10],],
                    [255, 255, 255, LUT_2[0b10],],
                    [255, 255, 255, LUT_2[0b11],],
                    [255, 255, 255, LUT_2[0b01],],
                    [255, 255, 255, LUT_2[0b00],],
                    [255, 255, 255, LUT_2[0b00],],
                ],
                [
                    [255, 255, 255, LUT_2[0b10],],
                    [255, 255, 255, LUT_2[0b00],],
                    [255, 255, 255, LUT_2[0b01],],
                    [255, 255, 255, LUT_2[0b11],],
                    [255, 255, 255, LUT_2[0b10],],
                    [255, 255, 255, LUT_2[0b10],],
                ],
                [
                    [255, 255, 255, LUT_2[0b00],],
                    [255, 255, 255, LUT_2[0b00],],
                    [255, 255, 255, LUT_2[0b10],],
                    [255, 255, 255, LUT_2[0b00],],
                    [255, 255, 255, LUT_2[0b00],],
                    [255, 255, 255, LUT_2[0b00],],
                ],
                [
                    [255, 255, 255, LUT_2[0b10],],
                    [255, 255, 255, LUT_2[0b00],],
                    [255, 255, 255, LUT_2[0b00],],
                    [255, 255, 255, LUT_2[0b00],],
                    [255, 255, 255, LUT_2[0b10],],
                    [255, 255, 255, LUT_2[0b00],],
                ]
            ])
        );
    }

    #[test]
    fn bmp4() {
        let data0 = 0b1010_1000u8.reverse_bits();
        let data1 = 0b0000_1000u8.reverse_bits();
        let raster = RasterGlyphImage {
            x: 0,
            y: 0,
            width: 1,
            height: 2,
            pixels_per_em: 0,
            format: RasterImageFormat::BitmapGray4,
            data: &[data0, data1],
        };

        let mut image = vec![];
        let extracted = extract_bw_image(
            &mut image,
            raster,
            Entry::Cached(CacheRect {
                color: false,
                x: 0,
                y: 0,
                width: 1,
                height: 2,
            }),
            1.0,
        )
        .expect("Didn't extract bmp4")
        .1;

        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&extracted),
            bytemuck::cast_slice(&[
                [[255, 255, 255, LUT_4[0b1010],],],
                [[255, 255, 255, LUT_4[0b0000],],],
            ])
        );
    }

    #[test]
    fn bmp4_packed() {
        let data0 = 0b1111_0001u8.reverse_bits();
        let data1 = 0b0011_1100u8.reverse_bits();
        let raster = RasterGlyphImage {
            x: 0,
            y: 0,
            width: 2,
            height: 2,
            pixels_per_em: 0,
            format: RasterImageFormat::BitmapGray4Packed,
            data: &[data0, data1],
        };

        let mut image = vec![];
        let extracted = extract_bw_image(
            &mut image,
            raster,
            Entry::Cached(CacheRect {
                color: false,
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            }),
            1.0,
        )
        .expect("Didn't extract bmp4")
        .1;

        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&extracted),
            bytemuck::cast_slice(&[
                [
                    [255, 255, 255, LUT_4[0b1111],],
                    [255, 255, 255, LUT_4[0b0001],],
                ],
                [
                    [255, 255, 255, LUT_4[0b0011],],
                    [255, 255, 255, LUT_4[0b1100],],
                ],
            ])
        );
    }
}
