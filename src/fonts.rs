use ratatui_core::buffer::Cell;
use ratatui_core::style::Modifier;
use rustybuzz::Face;

/// A Font which can be used for rendering.
#[derive(Clone)]
pub struct Font<'a> {
    font: Face<'a>,
    fallback: bool,
    advance: f32,
    id: u64,
}

/// The metrics needed for rendering.
#[derive(Debug, Default, Clone, Copy)]
pub struct FontBox {
    /// Width in px.
    pub width: u32,
    /// Height in px.
    pub height: u32,
    /// Baseline for glyphs. Measured from the top of the box.
    pub ascender: f32,
    /// Scaling factor from font-coords to px for the primary font.
    pub scale: f32,
}

impl FontBox {
    /// Pixel to cell-size. Rounded up.
    pub fn cell_size(
        &self,
        width: u32,
        height: u32,
    ) -> ratatui_core::layout::Size {
        let w = width / self.width + if width % self.width == 0 { 0 } else { 1 };
        let h = height / self.height + if height % self.height == 0 { 0 } else { 1 };

        ratatui_core::layout::Size::new(w as u16, h as u16)
    }

    /// Cell-size to pixel.
    pub fn px_size(
        &self,
        width: u16,
        height: u16,
    ) -> (u32, u32) {
        (width as u32 * self.width, height as u32 * self.height)
    }
}

impl<'a> Font<'a> {
    /// Create a new Font from data. Returns [`None`] if the font cannot
    /// be parsed.
    pub fn new(data: &'a [u8]) -> Option<Self> {
        Face::from_slice(data, 0).map(|font| {
            let em_idx;
            let advance;
            if font.is_monospaced() {
                em_idx = font.glyph_index('m').unwrap_or_default();
                advance = font.glyph_hor_advance(em_idx).unwrap_or_default() as f32;
            } else {
                em_idx = font.glyph_index('n').unwrap_or_default();
                advance = font.glyph_hor_advance(em_idx).unwrap_or_default() as f32;
            }

            Self {
                font,
                fallback: false,
                advance,
                id: 0,
            }
        })
    }
}

impl Font<'_> {
    pub(crate) fn font(&'_ self) -> &'_ Face<'_> {
        &self.font
    }

    pub(crate) fn is_fallback(&self) -> bool {
        self.fallback
    }

    pub(crate) fn ascender(&self) -> f32 {
        self.font.ascender() as f32
    }

    pub(crate) fn em_advance(&self) -> f32 {
        self.advance
    }

    pub(crate) fn scale(
        &self,
        height_px: u32,
    ) -> f32 {
        height_px as f32 / self.font.height() as f32
    }

    pub(crate) fn char_width(
        &self,
        height_px: u32,
    ) -> u32 {
        (self.advance * self.scale(height_px)) as u32
    }

    pub fn underline_metrics(
        &self,
        height_px: u32,
        box_height_px: u32,
    ) -> (u32, u32) {
        let scale = self.scale(height_px);

        let ascender = self.font.ascender() as f32;

        let underline_position = self
            .font
            .underline_metrics()
            .map(|m| m.position as f32)
            .unwrap_or(0.0);
        let underline_position = ascender - underline_position;

        let underline_thickness = self
            .font
            .underline_metrics()
            .map(|m| m.thickness as f32)
            .unwrap_or(100.0); /* observed average */
        // default underlines are a bit thin for larger font-sizes.
        let underline_thickness = underline_thickness * 1.3;

        let underline_position = (underline_position * scale) as u32;
        let underline_thickness = ((underline_thickness * scale) as u32).max(1);

        // might overflow the box
        if underline_position + underline_thickness < box_height_px {
            (underline_position, underline_position + underline_thickness)
        } else {
            (
                box_height_px.saturating_sub(underline_thickness),
                box_height_px,
            )
        }
    }

    pub fn strikeout_metrics(
        &self,
        height_px: u32,
        _box_height: u32,
    ) -> (u32, u32) {
        let scale = self.scale(height_px);

        let ascender = self.font.ascender() as f32;

        let strikeout_position = self
            .font
            .strikeout_metrics()
            .map(|m| m.position as f32)
            .unwrap_or_default();
        let strikeout_position = if strikeout_position > 0.0 {
            ascender - strikeout_position
        } else {
            ascender as f32 * 0.7 /* observed average */
        };

        let strikeout_thickness = self
            .font
            .strikeout_metrics()
            .map(|m| m.thickness as f32)
            .unwrap_or(100.0); /* observed average */
        // default strikeout lines are a bit thin for larger font-sizes.
        let strikeout_thickness = strikeout_thickness * 1.8;

        (
            (strikeout_position * scale) as u32,
            ((strikeout_position + strikeout_thickness) * scale) as u32,
        )
    }
}

/// A collection of fonts to use for rendering. Supports font fallback.
///
/// It is recommended, but not required, that all fonts have the same/very
/// similar aspect ratio, or you may get unexpected results during rendering due
/// to fallback.
pub struct Fonts<'a> {
    char_width_px: u32,
    char_height_px: u32,
    scale: f32,
    ascender: f32,
    em_advance: f32,

    last_resort: Vec<Font<'a>>,

    regular: Vec<Font<'a>>,
    bold: Vec<Font<'a>>,
    italic: Vec<Font<'a>>,
    bold_italic: Vec<Font<'a>>,
    // give an id in insertion order.
    id_count: u64,
}

impl<'a> Fonts<'a> {
    /// Create a new, empty set of fonts. The provided font will be used as a
    /// last-resort fallback if no other fonts can render a particular
    /// character. Rendering will attempt to fake bold/italic styles using this
    /// font where appropriate.
    ///
    /// The provided size_px will be the rendered height in pixels of all fonts
    /// in this collection.
    pub fn new(
        mut font: Font<'a>,
        size_px: u32,
    ) -> Self {
        font.fallback = true;
        font.id = 0;

        Self {
            char_width_px: font.char_width(size_px),
            char_height_px: size_px,
            scale: font.scale(size_px),
            ascender: font.ascender(),
            em_advance: font.em_advance(),
            last_resort: vec![font],
            regular: vec![],
            bold: vec![],
            italic: vec![],
            bold_italic: vec![],
            id_count: 1,
        }
    }

    /// Create a new, empty set of fonts. The provided fonts will be used as a
    /// last-resort fallback if no other fonts can render a particular
    /// character. Rendering will attempt to fake bold/italic styles using this
    /// font where appropriate.
    ///
    /// The expectation is that the fallback fonts accommodate for missing symbols
    /// and emojis. Any fonts used for actual text display should use [add_fonts]
    ///
    /// The provided size_px will be the rendered height in pixels of all fonts
    /// in this collection.
    pub fn new_vec(
        mut fonts: Vec<Font<'a>>,
        size_px: u32,
    ) -> Self {
        fonts.iter_mut().enumerate().for_each(|(n, f)| {
            f.fallback = true;
            f.id = n as u64
        });
        let id_count = fonts.len() as u64;

        Self {
            char_width_px: size_px / 2,
            char_height_px: size_px,
            scale: 1.0,
            ascender: size_px as f32,
            em_advance: size_px as f32 / 2.0,
            last_resort: fonts,
            regular: vec![],
            bold: vec![],
            italic: vec![],
            bold_italic: vec![],
            id_count,
        }
    }

    /// The height (in pixels) of all fonts.
    #[inline]
    pub fn height_px(&self) -> u32 {
        self.char_height_px
    }

    #[inline]
    pub fn ascender(&self) -> f32 {
        self.ascender
    }

    #[inline]
    pub fn em_advance(&self) -> f32 {
        self.em_advance
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Change the height of all fonts in this collection to the specified
    /// height in pixels.
    pub fn set_size_px(
        &mut self,
        height_px: u32,
    ) {
        self.char_height_px = height_px;

        if !self.regular.is_empty()
            || !self.bold.is_empty()
            || !self.italic.is_empty()
            || !self.bold_italic.is_empty()
        {
            (
                self.char_width_px,
                self.scale,
                self.ascender,
                self.em_advance,
            ) = self
                .regular
                .iter()
                .chain(self.bold.iter())
                .chain(self.italic.iter())
                .chain(self.bold_italic.iter())
                .map(|font| {
                    (
                        font.char_width(height_px),
                        font.scale(height_px),
                        font.ascender(),
                        font.em_advance(),
                    )
                })
                .next() /* first is fine */
                .expect("font");
        } else {
            self.char_width_px = self.char_height_px / 2;
            self.scale = 1.0;
            self.ascender = self.char_height_px as f32;
            self.em_advance = self.char_height_px as f32 / 2.0;
        }

        assert!(self.char_height_px != 0);
        assert!(self.char_width_px != 0);
    }

    /// Remove the non-fallback fonts.
    pub fn clear_fonts(&mut self) {
        self.bold_italic.clear();
        self.italic.clear();
        self.bold.clear();
        self.regular.clear();
    }

    /// Add a collection of fonts for various styles. They will automatically be
    /// added to the appropriate fallback font list based on the font's
    /// bold/italic properties. Note that this will automatically organize fonts
    /// by relative width in order to optimize fallback rendering quality. The
    /// ordering of already provided fonts will remain unchanged.
    ///
    /// Adding more fonts will not have any effect, if the text can be rendered
    /// with a prior font.
    pub fn add_fonts(
        &mut self,
        fonts: impl IntoIterator<Item = Font<'a>>,
    ) {
        for mut font in fonts {
            font.id = self.id_count;
            self.id_count += 1;

            if !font.font().is_monospaced() {
                warn!("Non monospace font used in add_fonts, this may cause unexpected rendering.");
            }
            if font.font().is_italic() && font.font().is_bold() {
                self.bold_italic.push(font);
            } else if font.font().is_italic() {
                self.italic.push(font);
            } else if font.font().is_bold() {
                self.bold.push(font);
            } else {
                self.regular.push(font);
            }
        }
        self.set_size_px(self.char_height_px);
    }

    /// Add a new collection of fonts for regular styled text. These fonts will
    /// come _after_ previously provided fonts in the fallback order.
    pub fn add_regular_fonts(
        &mut self,
        fonts: impl IntoIterator<Item = Font<'a>>,
    ) {
        for mut font in fonts {
            font.id = self.id_count;
            self.id_count += 1;
            self.regular.push(font);
        }
        self.set_size_px(self.char_height_px);
    }

    /// Add a new collection of fonts for bold styled text. These fonts will
    /// come _after_ previously provided fonts in the fallback order.
    ///
    /// You do not have to provide these for bold text to be supported. If no
    /// bold fonts are supplied, rendering will fallback to the regular fonts
    /// with fake bolding.
    pub fn add_bold_fonts(
        &mut self,
        fonts: impl IntoIterator<Item = Font<'a>>,
    ) {
        for mut font in fonts {
            font.id = self.id_count;
            self.id_count += 1;
            self.bold.push(font);
        }
        self.set_size_px(self.char_height_px);
    }

    /// Add a new collection of fonts for italic styled text. These fonts will
    /// come _after_ previously provided fonts in the fallback order.
    ///
    /// It is recommended, but not required, that you provide italic fonts if
    /// your application intends to make use of italics. If no italic fonts
    /// are supplied, rendering will fallback to the regular fonts with fake
    /// italics.
    pub fn add_italic_fonts(
        &mut self,
        fonts: impl IntoIterator<Item = Font<'a>>,
    ) {
        for mut font in fonts {
            font.id = self.id_count;
            self.id_count += 1;
            self.italic.push(font);
        }
        self.set_size_px(self.char_height_px);
    }

    /// Add a new collection of fonts for bold italic styled text. These fonts
    /// will come _after_ previously provided fonts in the fallback order.
    ///
    /// You do not have to provide these for bold text to be supported. If no
    /// bold fonts are supplied, rendering will fallback to the italic fonts
    /// with fake bolding.
    pub fn add_bold_italic_fonts(
        &mut self,
        fonts: impl IntoIterator<Item = Font<'a>>,
    ) {
        for mut font in fonts {
            font.id = self.id_count;
            self.id_count += 1;
            self.bold_italic.push(font);
        }
        self.set_size_px(self.char_height_px);
    }

    /// Size of a cell with the current font in px.
    pub fn font_box(&self) -> FontBox {
        FontBox {
            width: self.min_width_px(),
            height: self.height_px(),
            ascender: self.ascender(),
            scale: self.scale(),
        }
    }

    /// The minimum width (in pixels) across all fonts.
    pub fn min_width_px(&self) -> u32 {
        self.char_width_px
    }

    pub(crate) fn count(&self) -> usize {
        1 + self.bold.len() + self.italic.len() + self.bold_italic.len() + self.regular.len()
    }

    pub(crate) fn get_by_id(
        &'a self,
        id: u64,
    ) -> &'a Font<'a> {
        self.regular
            .iter()
            .chain(self.bold.iter())
            .chain(self.italic.iter())
            .chain(self.bold_italic.iter())
            .chain(self.last_resort.iter())
            .find(|v| v.id == id)
            .expect("font")
    }

    pub(crate) fn font_for_cell(
        &'_ self,
        cell: &Cell,
    ) -> u64 {
        if cell.modifier.contains(Modifier::BOLD | Modifier::ITALIC) {
            self.select_font(
                cell.symbol(),
                self.bold_italic
                    .iter()
                    .map(|f| f)
                    .chain(self.italic.iter().map(|f| f))
                    .chain(self.bold.iter().map(|f| f))
                    .chain(self.regular.iter().map(|f| f))
                    .chain(self.last_resort.iter().map(|f| f)),
            )
        } else if cell.modifier.contains(Modifier::BOLD) {
            self.select_font(
                cell.symbol(),
                self.bold
                    .iter()
                    .map(|f| f)
                    .chain(self.regular.iter().map(|f| f))
                    .chain(self.last_resort.iter().map(|f| f)),
            )
        } else if cell.modifier.contains(Modifier::ITALIC) {
            self.select_font(
                cell.symbol(),
                self.italic
                    .iter()
                    .map(|f| f)
                    .chain(self.regular.iter().map(|f| f))
                    .chain(self.last_resort.iter().map(|f| f)),
            )
        } else {
            self.select_font(
                cell.symbol(),
                self.regular
                    .iter()
                    .map(|f| f)
                    .chain(self.last_resort.iter().map(|f| f)),
            )
        }
    }

    fn select_font<'fonts>(
        &'fonts self,
        cluster: &str,
        fonts: impl IntoIterator<Item = &'fonts Font<'a>>,
    ) -> u64 {
        let mut max = 0;
        let mut font = None;
        let mut last_resort = None;

        for candidate in fonts.into_iter() {
            // try to map the complete cluster to a single font.
            // the first font that can map it completely wins, otherwise
            // the one with the max matched glyphs.
            let (count, last_idx) =
                cluster
                    .chars()
                    .enumerate()
                    .fold((0, 0), |(mut count, _), (idx, ch)| {
                        count += usize::from(candidate.font().glyph_index(ch).is_some());
                        (count, idx)
                    });

            if count > max {
                max = count;
                font = Some(candidate.id);
            }

            if count == last_idx + 1 {
                break;
            }

            last_resort = Some(candidate.id);
        }

        font.unwrap_or_else(|| {
            if let Some(last_resort) = last_resort {
                last_resort
            } else {
                panic!("at least one font must be set.");
            }
        })
    }
}
