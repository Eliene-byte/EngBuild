//! Packed color types. `Srgba` is 4 unorm floats ready for a GPU vertex buffer.

/// Linear-space float color in `[0, 1]` with alpha.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba {
    pub const TRANSPARENT: Self = Self::new(0.0, 0.0, 0.0, 0.0);
    pub const BLACK: Self = Self::new(0.0, 0.0, 0.0, 1.0);
    pub const WHITE: Self = Self::new(1.0, 1.0, 1.0, 1.0);

    #[inline(always)]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    #[inline(always)]
    pub const fn from_srgb_u8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self::new(
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            a as f32 / 255.0,
        )
    }

    #[inline(always)]
    pub const fn to_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// sRGB -> linear transfer (IEC 61966-2-1), used before lighting.
    #[inline(always)]
    pub fn to_linear(self) -> Self {
        fn f(c: f32) -> f32 {
            if c <= 0.040_45 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        }
        Self::new(f(self.r), f(self.g), f(self.b), self.a)
    }

    #[inline(always)]
    pub fn with_alpha(self, a: f32) -> Self {
        Self::new(self.r, self.g, self.b, a)
    }

    #[inline(always)]
    pub fn to_srgba(self) -> Srgba {
        Srgba(self)
    }

    /// Packed `0xRRGGBBAA`, the on-disk format for layers/linetypes.
    #[inline]
    pub fn to_u32(self) -> u32 {
        let c = |x: f32| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
        (c(self.r) << 24) | (c(self.g) << 16) | (c(self.b) << 8) | c(self.a)
    }

    #[inline]
    pub fn from_u32(v: u32) -> Self {
        Self::from_srgb_u8(
            ((v >> 24) & 0xff) as u8,
            ((v >> 16) & 0xff) as u8,
            ((v >> 8) & 0xff) as u8,
            (v & 0xff) as u8,
        )
    }

    /// Relative luminance, for contrast decisions in the UI.
    #[inline]
    pub fn luminance(self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    /// Pick black or white text for legibility on this background.
    #[inline]
    pub fn contrasting_text(self) -> Srgba {
        if self.luminance() > 0.45 {
            Srgba::new(0.05, 0.05, 0.05, 1.0)
        } else {
            Srgba::WHITE
        }
    }
}

/// The GPU-facing color: 4 x `unorm32` floats, exactly the WGSL `vec4<f32>`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Srgba(pub Rgba);

impl Srgba {
    pub const TRANSPARENT: Self = Self(Rgba::TRANSPARENT);
    pub const BLACK: Self = Self(Rgba::BLACK);
    pub const WHITE: Self = Self(Rgba::WHITE);

    #[inline(always)]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self(Rgba::new(r, g, b, a))
    }
    #[inline(always)]
    pub const fn from_srgb_u8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(Rgba::from_srgb_u8(r, g, b, a))
    }
    #[inline(always)]
    pub const fn to_array(self) -> [f32; 4] {
        self.0.to_array()
    }
    #[inline(always)]
    pub fn with_alpha(self, a: f32) -> Self {
        Self(self.0.with_alpha(a))
    }
    #[inline(always)]
    pub const fn rgb(self) -> [f32; 3] {
        [self.0.r, self.0.g, self.0.b]
    }
}

impl From<Rgba> for Srgba {
    #[inline(always)]
    fn from(r: Rgba) -> Self {
        Self(r)
    }
}

/// The classic AutoCAD-ish 7-color ACI palette (first 10 indices), kept because
/// DXF files reference colors by index.
pub const ACI_PALETTE: [u32; 9] = [
    0xFFFF_0000,  // 1 red
    0xFFFF_FF00,  // 2 yellow
    0xFF00_FF00,  // 3 green
    0xFF00_FFFF,  // 4 cyan
    0xFF00_00FF,  // 5 blue
    0xFFFF_00FF,  // 6 magenta
    0xFF00_0000,  // 7 white/black (by layer)
    0xFF80_80_80, // 8 dark gray
    0xFFC0_C0C0,  // 9 light gray
];

/// Resolve an ACI color index to RGBA. `0` means "by block", `256` "by layer",
/// and 7 is theme-dependent (white on dark UI).
pub fn aci_to_rgba(index: i16, by_layer: Rgba) -> Rgba {
    match index {
        0 | 256 => by_layer,
        7 => Rgba::WHITE,
        i => Rgba::from_u32(
            *ACI_PALETTE
                .get((i as usize).saturating_sub(1))
                .unwrap_or(&0xFFFF_FFFF),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_transfer() {
        let l = Rgba::from_srgb_u8(255, 255, 255, 255).to_linear();
        assert!((l.r - 1.0).abs() < 1e-4);
        let m = Rgba::from_srgb_u8(128, 128, 128, 255).to_linear();
        assert!(m.r > 0.2 && m.r < 0.3, "{}", m.r);
    }

    #[test]
    fn u32_round_trip() {
        let c = Rgba::from_srgb_u8(18, 52, 86, 255);
        let p = c.to_u32();
        let back = Rgba::from_u32(p);
        assert!((c.r - back.r).abs() < 1.0 / 255.0);
        assert_eq!(c.a, 1.0);
    }

    #[test]
    fn aci_mapping() {
        assert_eq!(aci_to_rgba(1, Rgba::BLACK).to_u32(), 0xFFFF0000);
        assert_eq!(aci_to_rgba(256, Rgba::WHITE), Rgba::WHITE);
        assert_eq!(aci_to_rgba(0, Rgba::WHITE), Rgba::WHITE);
    }
}
