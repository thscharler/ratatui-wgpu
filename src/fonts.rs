use ratatui_core::buffer::Cell;
use ratatui_core::style::Modifier;
use rustybuzz::Face;
use std::hash::BuildHasher;
use std::hash::Hasher;
use std::hash::RandomState;

/// A Font which can be used for rendering.
#[derive(Clone)]
pub struct Font<'a> {
    font: Face<'a>,
    advance: f32,
    id: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct FontBox {
    pub width: u32,
    pub height: u32,
    pub ascender: f32,
    pub scale: f32,
}

pub(crate) struct RenderedFont<'a> {
    pub font_box: FontBox,
    pub font: &'a Font<'a>,
    pub fake_bold: bool,
    pub fake_italic: bool,
    pub is_fallback: bool,
}

impl<'a> Font<'a> {
    /// Create a new Font from data. Returns [`None`] if the font cannot
    /// be parsed.
    pub fn new(data: &'a [u8]) -> Option<Self> {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write(data);

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
                advance,
                id: hasher.finish(),
            }
        })
    }
}

impl Font<'_> {
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn font(&'_ self) -> &'_ Face<'_> {
        &self.font
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
}

impl<'a> RenderedFont<'a> {
    pub(crate) fn font(&self) -> &'_ Face<'_> {
        &self.font.font
    }

    pub(crate) fn underline(
        &self,
        height_px: u32,
        box_height_px: u32,
    ) -> (u32, u32) {
        let scale = self.font.scale(height_px);

        let ascender = self.font.ascender() as f32;

        let underline_position = self
            .font
            .font
            .underline_metrics()
            .map(|m| m.position as f32)
            .unwrap_or(0.0);
        let underline_position = ascender - underline_position;

        let underline_thickness = self
            .font
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

    pub(crate) fn strikeout(
        &self,
        height_px: u32,
        _box_height: u32,
    ) -> (u32, u32) {
        let scale = self.font.scale(height_px);

        let ascender = self.font.ascender() as f32;

        let strikeout_position = self
            .font
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

    has_fonts: bool,
    regular: Vec<Font<'a>>,
    bold: Vec<Font<'a>>,
    italic: Vec<Font<'a>>,
    bold_italic: Vec<Font<'a>>,
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
        font: Font<'a>,
        size_px: u32,
    ) -> Self {
        Self {
            char_width_px: font.char_width(size_px),
            char_height_px: size_px,
            scale: font.scale(size_px),
            ascender: font.ascender(),
            em_advance: font.em_advance(),
            last_resort: vec![font],
            has_fonts: false,
            regular: vec![],
            bold: vec![],
            italic: vec![],
            bold_italic: vec![],
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
        fonts: Vec<Font<'a>>,
        size_px: u32,
    ) -> Self {
        Self {
            char_width_px: size_px / 2,
            char_height_px: size_px,
            scale: 1.0,
            ascender: size_px as f32,
            em_advance: size_px as f32 / 2.0,
            last_resort: fonts,
            has_fonts: false,
            regular: vec![],
            bold: vec![],
            italic: vec![],
            bold_italic: vec![],
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

        if self.has_fonts {
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
                .unwrap_or_default();
        } else {
            self.char_width_px = self.char_height_px / 2;
            self.scale = 1.0;
            self.ascender = self.char_height_px as f32;
            self.em_advance = self.char_height_px as f32 / 2.0;
        }
    }

    /// Remove the non-fallback fonts.
    pub fn clear_fonts(&mut self) {
        self.bold_italic.clear();
        self.italic.clear();
        self.bold.clear();
        self.regular.clear();
        self.has_fonts = false;
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
        let bold_italic_len = self.bold_italic.len();
        let italic_len = self.italic.len();
        let bold_len = self.bold.len();
        let regular_len = self.regular.len();

        for font in fonts {
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

        self.bold_italic[bold_italic_len..]
            .sort_by_key(|font| font.char_width(self.char_height_px));
        self.italic[italic_len..].sort_by_key(|font| font.char_width(self.char_height_px));
        self.bold[bold_len..].sort_by_key(|font| font.char_width(self.char_height_px));
        self.regular[regular_len..].sort_by_key(|font| font.char_width(self.char_height_px));

        self.has_fonts = !self.bold_italic.is_empty()
            || !self.italic.is_empty()
            || !self.bold.is_empty()
            || !self.regular.is_empty();

        self.set_size_px(self.char_height_px);
    }

    /// Add a new collection of fonts for regular styled text. These fonts will
    /// come _after_ previously provided fonts in the fallback order.
    pub fn add_regular_fonts(
        &mut self,
        fonts: impl IntoIterator<Item = Font<'a>>,
    ) {
        self.regular.extend(fonts.into_iter());
        self.set_size_px(self.char_height_px);
        self.has_fonts = self.has_fonts || !self.regular.is_empty();
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
        self.bold.extend(fonts.into_iter());
        self.set_size_px(self.char_height_px);
        self.has_fonts = self.has_fonts || !self.bold.is_empty();
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
        self.italic.extend(fonts.into_iter());
        self.set_size_px(self.char_height_px);
        self.has_fonts = self.has_fonts || !self.italic.is_empty();
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
        self.bold_italic.extend(fonts.into_iter());
        self.set_size_px(self.char_height_px);
        self.has_fonts = self.has_fonts || !self.bold_italic.is_empty();
    }
}

impl<'a> Fonts<'a> {
    /// Size of a cell with the current font in px.
    pub(crate) fn font_box(&self) -> FontBox {
        FontBox {
            width: self.min_width_px(),
            height: self.height_px(),
            ascender: self.ascender(),
            scale: self.scale(),
        }
    }

    /// The minimum width (in pixels) across all fonts.
    pub(crate) fn min_width_px(&self) -> u32 {
        self.char_width_px
    }

    pub(crate) fn count(&self) -> usize {
        1 + self.bold.len() + self.italic.len() + self.bold_italic.len() + self.regular.len()
    }

    pub(crate) fn font_for_cell(
        &'_ self,
        cell: &Cell,
    ) -> (&'_ Font<'_>, bool, bool, bool) {
        if cell.modifier.contains(Modifier::BOLD | Modifier::ITALIC) {
            self.select_font(
                cell.symbol(),
                self.bold_italic
                    .iter()
                    .map(|f| (f, false, false, false))
                    .chain(self.italic.iter().map(|f| (f, true, false, false)))
                    .chain(self.bold.iter().map(|f| (f, false, true, false)))
                    .chain(self.regular.iter().map(|f| (f, true, true, false)))
                    .chain(self.last_resort.iter().map(|v| (v, true, true, true))),
            )
        } else if cell.modifier.contains(Modifier::BOLD) {
            self.select_font(
                cell.symbol(),
                self.bold
                    .iter()
                    .map(|f| (f, false, false, false))
                    .chain(self.regular.iter().map(|f| (f, true, false, false)))
                    .chain(self.last_resort.iter().map(|v| (v, true, false, true))),
            )
        } else if cell.modifier.contains(Modifier::ITALIC) {
            self.select_font(
                cell.symbol(),
                self.italic
                    .iter()
                    .map(|f| (f, false, false, false))
                    .chain(self.regular.iter().map(|f| (f, false, true, false)))
                    .chain(self.last_resort.iter().map(|v| (v, false, true, true))),
            )
        } else {
            self.select_font(
                cell.symbol(),
                self.regular
                    .iter()
                    .map(|f| (f, false, false, false))
                    .chain(self.last_resort.iter().map(|v| (v, false, false, true))),
            )
        }
    }

    fn select_font<'fonts>(
        &'fonts self,
        cluster: &str,
        fonts: impl IntoIterator<Item = (&'fonts Font<'a>, bool, bool, bool)>,
    ) -> (&'fonts Font<'a>, bool, bool, bool) {
        let mut max = 0;
        let mut font = None;
        let mut last_resort = None;

        for (candidate, fake_bold, fake_italic, is_fallback) in fonts.into_iter() {
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
                font = Some((candidate, fake_bold, fake_italic, is_fallback));
            }

            if count == last_idx + 1 {
                break;
            }

            last_resort = Some((candidate, fake_bold, fake_italic, is_fallback));
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
