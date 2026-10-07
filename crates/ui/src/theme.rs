//! Colours and metrics for the UI, in one place so the whole app can be
//! re-skinned without touching a widget.

use cad_core::Rgba;

/// A flat colour set. Deliberately not a struct with named fields per widget:
/// a palette is easier to keep coherent as a whole.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Theme {
    /// Window / panel background.
    pub background: Rgba,
    /// Slightly raised surfaces: toolbars, headers, inputs.
    pub surface: Rgba,
    pub surface_hover: Rgba,
    pub surface_active: Rgba,
    /// Drawing canvas. This is the colour the user spends the most time
    /// staring at, so it is near-black and low-chroma.
    pub canvas: Rgba,
    pub border: Rgba,
    pub border_focused: Rgba,
    pub text: Rgba,
    pub text_dim: Rgba,
    pub text_disabled: Rgba,
    pub accent: Rgba,
    pub accent_text: Rgba,
    pub danger: Rgba,
    pub warning: Rgba,
    pub success: Rgba,
    /// Grid, axes and construction lines drawn *over* the canvas.
    pub grid: Rgba,
    pub axis_x: Rgba,
    pub axis_y: Rgba,
    pub selection: Rgba,
    pub highlight: Rgba,
    /// Crosshair / rubber band.
    pub rubber_band: Rgba,
    /// Font size in pixels.
    pub font_size: f32,
    pub row_height: f32,
    pub button_height: f32,
    pub border_radius: f32,
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

impl Theme {
    /// The default dark theme: what a CAD user expects (AutoCAD dark model
    /// space is near-black, not white).
    pub fn dark() -> Self {
        Self {
            background: Rgba::new(0.094, 0.102, 0.118, 1.0),
            surface: Rgba::new(0.145, 0.157, 0.180, 1.0),
            surface_hover: Rgba::new(0.196, 0.212, 0.243, 1.0),
            surface_active: Rgba::from_srgb_u8(0x3F, 0x45, 0x51, 0xFF),
            canvas: Rgba::new(0.055, 0.063, 0.078, 1.0),
            border: Rgba::new(0.243, 0.263, 0.302, 1.0),
            border_focused: Rgba::new(0.302, 0.541, 0.949, 1.0),
            text: Rgba::new(0.878, 0.894, 0.918, 1.0),
            text_dim: Rgba::new(0.596, 0.627, 0.671, 1.0),
            text_disabled: Rgba::new(0.400, 0.420, 0.455, 1.0),
            accent: Rgba::new(0.294, 0.549, 0.949, 1.0),
            accent_text: Rgba::new(1.0, 1.0, 1.0, 1.0),
            danger: Rgba::new(0.902, 0.310, 0.310, 1.0),
            warning: Rgba::new(0.953, 0.706, 0.239, 1.0),
            success: Rgba::new(0.310, 0.780, 0.451, 1.0),
            grid: Rgba::new(0.220, 0.235, 0.267, 1.0),
            axis_x: Rgba::new(0.800, 0.290, 0.310, 1.0),
            axis_y: Rgba::new(0.310, 0.740, 0.400, 1.0),
            selection: Rgba::new(0.294, 0.549, 0.949, 0.85),
            highlight: Rgba::new(0.949, 0.686, 0.239, 0.95),
            rubber_band: Rgba::new(0.400, 0.780, 0.980, 0.90),
            font_size: 13.0,
            row_height: 24.0,
            button_height: 28.0,
            border_radius: 4.0,
        }
    }

    /// Light theme, for drafting on paper.
    pub fn light() -> Self {
        Self {
            background: Rgba::new(0.933, 0.937, 0.945, 1.0),
            surface: Rgba::new(1.0, 1.0, 1.0, 1.0),
            surface_hover: Rgba::new(0.925, 0.937, 0.953, 1.0),
            surface_active: Rgba::new(0.867, 0.886, 0.914, 1.0),
            canvas: Rgba::new(1.0, 1.0, 1.0, 1.0),
            border: Rgba::new(0.769, 0.792, 0.824, 1.0),
            border_focused: Rgba::new(0.157, 0.424, 0.808, 1.0),
            text: Rgba::new(0.078, 0.094, 0.125, 1.0),
            text_dim: Rgba::new(0.325, 0.353, 0.404, 1.0),
            text_disabled: Rgba::new(0.580, 0.608, 0.651, 1.0),
            accent: Rgba::new(0.157, 0.424, 0.808, 1.0),
            accent_text: Rgba::new(1.0, 1.0, 1.0, 1.0),
            danger: Rgba::new(0.780, 0.141, 0.141, 1.0),
            warning: Rgba::new(0.749, 0.478, 0.027, 1.0),
            success: Rgba::new(0.153, 0.545, 0.259, 1.0),
            grid: Rgba::new(0.808, 0.827, 0.855, 1.0),
            axis_x: Rgba::new(0.800, 0.200, 0.200, 1.0),
            axis_y: Rgba::new(0.150, 0.600, 0.250, 1.0),
            selection: Rgba::new(0.157, 0.424, 0.808, 0.80),
            highlight: Rgba::new(0.851, 0.478, 0.043, 0.90),
            rubber_band: Rgba::new(0.157, 0.514, 0.784, 0.85),
            font_size: 13.0,
            row_height: 24.0,
            button_height: 28.0,
            border_radius: 4.0,
        }
    }

    /// Surface colour for a control in a given state.
    pub fn surface_for(&self, hovered: bool, active: bool) -> Rgba {
        if active {
            self.surface_active
        } else if hovered {
            self.surface_hover
        } else {
            self.surface
        }
    }

    /// Text colour for a disabled control.
    pub fn text_for(&self, disabled: bool) -> Rgba {
        if disabled {
            self.text_disabled
        } else {
            self.text
        }
    }

    /// Monospace advance width, used by the immediate-mode text layout before
    /// the glyph atlas is queried.
    pub fn text_width(&self, s: &str) -> f32 {
        s.chars().count() as f32 * self.font_size * 0.55
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_themes_are_complete() {
        for t in [Theme::dark(), Theme::light()] {
            assert!(t.font_size > 0.0);
            assert!(t.row_height > 0.0);
            assert!(t.border_radius >= 0.0);
            assert_eq!(t.text.a, 1.0, "text must be opaque");
        }
    }

    #[test]
    fn dark_canvas_is_darker_than_surfaces() {
        let t = Theme::dark();
        assert!(
            t.canvas.r < t.surface.r,
            "the canvas must sit behind panels: {:?} vs {:?}",
            t.canvas,
            t.surface
        );
        assert!(
            t.text.luminance() > t.background.luminance(),
            "text must contrast"
        );
    }

    #[test]
    fn light_canvas_is_lighter_than_surfaces() {
        let t = Theme::light();
        assert!(t.canvas.luminance() > t.background.luminance());
        assert!(t.text.luminance() < t.canvas.luminance());
    }

    #[test]
    fn light_and_dark_have_opposite_contrast() {
        let d = Theme::dark();
        let l = Theme::light();
        assert!(d.text.luminance() > d.canvas.luminance());
        assert!(l.text.luminance() < l.canvas.luminance());
        // Selection must be readable on both.
        assert!(d.selection.luminance() > 0.2);
        assert!(l.selection.luminance() < 0.5);
    }

    #[test]
    fn surface_state_resolution() {
        let t = Theme::dark();
        assert_eq!(t.surface_for(false, false), t.surface);
        assert_eq!(t.surface_for(true, false), t.surface_hover);
        assert_eq!(t.surface_for(true, true), t.surface_active);
        assert_eq!(t.surface_for(false, true), t.surface_active);
    }

    #[test]
    fn text_width_is_monotonic_and_nonzero() {
        let t = Theme::dark();
        assert!(t.text_width("A") > 0.0);
        assert!(t.text_width("AB") > t.text_width("A"));
        assert_eq!(t.text_width(""), 0.0);
        // Multibyte text must measure by glyph count, not bytes.
        assert_eq!(t.text_width("ação"), t.text_width("acao"));
    }
}
