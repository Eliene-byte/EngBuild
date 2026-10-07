//! Named text styles (AutoCAD `STYLE` table).

use cad_core::Vec2;

/// How a text glyph is stretched to fit a defined width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextWidthStyle {
    #[default]
    Auto,
    /// Scale each glyph so the whole string fits `width`.
    Fit,
    /// Scale X only.
    FitWidth,
}

/// A single-line text style definition.
#[derive(Debug, Clone, PartialEq)]
pub struct TextStyle {
    pub name: String,
    /// Height of a capital letter.
    pub height: f32,
    /// Uniform width factor (X scale).
    pub width: f32,
    /// Oblique angle in degrees.
    pub oblique: f32,
    /// Extra spacing between characters, as a fraction of height.
    pub tracking: f32,
    /// For `Fit`/`FitWidth`: the target width.
    pub fixed_width: f32,
    pub width_style: TextWidthStyle,
    /// Index into the renderer's built-in font table.
    pub font_index: u32,
    /// Optional TTF/OTF path for externally supplied shapes.
    pub font_file: Option<String>,
    pub big: bool,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            name: "Standard".to_string(),
            height: 0.0,
            width: 1.0,
            oblique: 0.0,
            tracking: 0.0,
            fixed_width: 0.0,
            width_style: TextWidthStyle::Auto,
            font_index: 0,
            font_file: None,
            big: false,
        }
    }
}

impl TextStyle {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Resolve the effective cap height for a string.
    ///
    /// With `Auto` the width factor is 1 and the height is whatever the caller
    /// asked for; with `Fit`/`FitWidth` the size is derived from `fixed_width`.
    pub fn resolve_size(
        &self,
        requested_height: f32,
        requested_width_factor: f32,
        text_len: usize,
    ) -> (f32, f32) {
        match self.width_style {
            TextWidthStyle::Auto | TextWidthStyle::FitWidth => {
                let h = if requested_height > 0.0 {
                    requested_height
                } else {
                    self.height.max(0.1)
                };
                let w = if requested_width_factor > 0.0 {
                    requested_width_factor
                } else {
                    self.width.max(0.01)
                };
                (h, w)
            }
            TextWidthStyle::Fit => {
                if self.fixed_width <= 0.0 || text_len == 0 {
                    let h = if requested_height > 0.0 {
                        requested_height
                    } else {
                        self.height.max(0.1)
                    };
                    return (h, 1.0);
                }
                // Monospace-ish approximation: 0.6 em per glyph.
                let h = self.fixed_width / (text_len as f32 * 0.6);
                (h, 1.0)
            }
        }
    }

    /// Total advance width of `text`.
    pub fn measure(&self, text: &str, height: f32, width_factor: f32) -> f32 {
        let n = text.chars().count().max(1) as f32;
        let base = n * height * 0.6 * width_factor;
        base * (1.0 + self.tracking)
    }

    /// Apply the oblique shear to a glyph offset.
    pub fn shear(&self, offset: Vec2, height: f32) -> Vec2 {
        if self.oblique.abs() < 1e-6 {
            return offset;
        }
        let t = (self.oblique * cad_core::RAD).tan();
        Vec2::new(offset.x + offset.y * t * height, offset.y)
    }
}

/// The style table, with a `Standard` entry always present.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextStyleTable {
    items: Vec<Option<TextStyle>>,
    names: std::collections::HashMap<String, usize>,
}

impl TextStyleTable {
    pub fn new() -> Self {
        let mut t = Self::default();
        t.insert(TextStyle::new("Standard"));
        t
    }
    pub fn insert(&mut self, s: TextStyle) -> usize {
        if let Some(i) = self.names.get(&s.name.to_ascii_uppercase()) {
            self.items[*i] = Some(s);
            return *i;
        }
        let i = self.items.len();
        self.names.insert(s.name.to_ascii_uppercase(), i);
        self.items.push(Some(s));
        i
    }
    pub fn get(&self, i: usize) -> Option<&TextStyle> {
        self.items.get(i).and_then(|o| o.as_ref())
    }
    pub fn by_name(&self, name: &str) -> Option<usize> {
        self.names.get(&name.to_ascii_uppercase()).copied()
    }
    pub fn index_of_standard(&self) -> Option<usize> {
        self.by_name("Standard")
    }
    pub fn len(&self) -> usize {
        self.items.iter().filter(|o| o.is_some()).count()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn iter(&self) -> impl Iterator<Item = (usize, &TextStyle)> {
        self.items
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.as_ref().map(|s| (i, s)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_has_standard() {
        let t = TextStyleTable::new();
        assert_eq!(t.len(), 1);
        assert!(t.index_of_standard().is_some());
        assert_eq!(t.by_name("standard"), t.index_of_standard());
    }

    #[test]
    fn fit_derives_height_from_width() {
        let mut s = TextStyle::new("fit");
        s.width_style = TextWidthStyle::Fit;
        s.fixed_width = 12.0;
        let (h, w) = s.resolve_size(0.0, 0.0, 10);
        assert!((h - 2.0).abs() < 1e-5, "h={h}");
        assert_eq!(w, 1.0);
    }

    #[test]
    fn auto_uses_requested_values() {
        let s = TextStyle::new("a");
        assert_eq!(s.resolve_size(2.5, 0.8, 5), (2.5, 0.8));
        assert_eq!(s.resolve_size(0.0, 0.0, 5), (0.1, 1.0));
    }

    #[test]
    fn measure_scales_with_tracking() {
        let mut s = TextStyle::new("m");
        assert!((s.measure("abcd", 1.0, 1.0) - 2.4).abs() < 1e-5);
        s.tracking = 0.5;
        assert!((s.measure("abcd", 1.0, 1.0) - 3.6).abs() < 1e-5);
    }

    #[test]
    fn oblique_shear_only_when_set() {
        let mut s = TextStyle::new("o");
        let p = Vec2::new(1.0, 1.0);
        assert_eq!(s.shear(p, 1.0), p);
        s.oblique = 45.0;
        let q = s.shear(p, 1.0);
        assert!(q.x > 1.5, "{q:?}");
        assert!((q.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn insert_is_idempotent_by_name() {
        let mut t = TextStyleTable::new();
        let a = t.insert(TextStyle::new("Dup"));
        let b = t.insert(TextStyle {
            height: 3.0,
            ..TextStyle::new("Dup")
        });
        assert_eq!(a, b);
        assert_eq!(t.get(a).unwrap().height, 3.0);
        assert_eq!(t.len(), 2);
    }
}
