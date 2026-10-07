//! DXF "bulge": the arc fraction stored inside LWPOLYLINE vertices.
//!
//! `bulge = tan(θ/4)` where `θ` is the included angle of the arc. Positive is
//! counter-clockwise. This tiny value is why most CAD programs got polylines
//! wrong on import — we handle it explicitly.

use cad_core::{EPS, TAU, Vec2, wrap_pi};

/// Arc bulge factor of a segment, defined by `tan(included_angle / 4)`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Bulge(pub f32);

impl Bulge {
    pub const NONE: Self = Self(0.0);

    #[inline]
    pub fn is_arc(self) -> bool {
        self.0.abs() > 1e-9
    }

    /// Included angle of the arc in radians (signed).
    #[inline]
    pub fn included_angle(self) -> f32 {
        4.0 * self.0.atan()
    }

    /// Build a bulge that sweeps `sweep` radians from `p0` to `p1`.
    #[inline]
    pub fn from_sweep(sweep: f32) -> Self {
        Self((sweep * 0.25).tan())
    }

    /// Center of the arc from its two endpoints.
    ///
    /// Uses the chord midpoint plus the perpendicular offset
    /// `|d| / (2 * tan(θ/2))`, which is exact for every sweep < 2π.
    /// The offset points to the **left** of travel for a positive (CCW) bulge.
    #[inline]
    pub fn center(&self, p0: Vec2, p1: Vec2) -> Vec2 {
        let chord = p1 - p0;
        let d2 = chord.length_squared();
        if d2 < EPS * EPS {
            return p0;
        }
        let theta = self.included_angle();
        let half = (theta * 0.5).tan();
        let k = if half.abs() < 1e-12 {
            0.0
        } else {
            0.5 * d2.sqrt() / half
        };
        (p0 + p1) * 0.5 + chord.normalize().perp() * k
    }

    /// Radius of the arc from its two endpoints.
    #[inline]
    pub fn radius(&self, p0: Vec2, p1: Vec2) -> f32 {
        let d = p1.distance(p0);
        let theta = self.included_angle();
        let h = (theta * 0.5).sin().abs();
        if h < 1e-7 {
            f32::INFINITY
        } else {
            (d * 0.5) / h
        }
    }

    /// Sweep angle in `(-2π, 2π)`.
    #[inline]
    pub fn sweep(&self) -> f32 {
        let t = self.included_angle();
        if t > TAU { wrap_pi(t) } else { t }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::PI;

    #[test]
    fn quarter_circle_center() {
        // p0=(1,0) -> p1=(0,1) is a 90 deg CCW arc centered at the origin.
        let b = Bulge::from_sweep(PI / 2.0);
        // bulge = tan(θ/4) = tan(22.5°)
        assert!((b.0 - 0.414_213_6).abs() < 1e-6, "{}", b.0);
        let c = b.center(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0));
        assert!(c.distance(Vec2::ZERO) < 1e-4, "{c:?}");
        assert!((b.radius(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn clockwise_bulge() {
        // p0=(1,0) -> p1=(0,-1): 90 deg CW, still centered at origin.
        let b = Bulge::from_sweep(-PI / 2.0);
        let c = b.center(Vec2::new(1.0, 0.0), Vec2::new(0.0, -1.0));
        assert!(c.distance(Vec2::ZERO) < 1e-4, "{c:?}");
    }

    #[test]
    fn semicircle_radius_is_half_the_chord() {
        let b = Bulge::from_sweep(PI);
        let r = b.radius(Vec2::ZERO, Vec2::new(2.0, 0.0));
        assert!((r - 1.0).abs() < 1e-4, "r={r}");
        let c = b.center(Vec2::ZERO, Vec2::new(2.0, 0.0));
        assert!(c.distance(Vec2::new(1.0, 0.0)) < 1e-4, "{c:?}");
    }

    #[test]
    fn bulge_larger_than_one_gives_major_arc() {
        // θ > 180° needs bulge > 1.
        let b = Bulge::from_sweep(3.0 * PI / 2.0);
        assert!(b.0 > 1.0);
        assert!((b.included_angle() - 3.0 * PI / 2.0).abs() < 1e-4);
    }

    #[test]
    fn zero_bulge_is_straight() {
        let b = Bulge::NONE;
        assert!(!b.is_arc());
        assert!(b.center(Vec2::ZERO, Vec2::X).distance(Vec2::new(0.5, 0.0)) < 1e-5);
    }
}
