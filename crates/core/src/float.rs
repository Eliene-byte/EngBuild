//! Scalar helpers and tolerances.

/// Generic "close enough" epsilon used across the geometry kernel.
pub const EPS: f32 = 1.0e-6;
/// Tighter epsilon for exact/algebraic tests (angle ordering, boolean ops).
pub const EPS_TIGHT: f32 = 1.0e-9;

pub const TAU: f32 = core::f32::consts::PI * 2.0;
pub const PI: f32 = core::f32::consts::PI;
pub const FRAC_PI_2: f32 = core::f32::consts::FRAC_PI_2;
pub const DEG: f32 = 180.0 / PI;
pub const RAD: f32 = PI / 180.0;

/// Squared distance below which two points are considered coincident.
#[inline]
pub fn dist2(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

#[inline]
pub fn nearly_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

#[inline]
pub fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[inline(always)]
pub const fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Wrap an angle into `(-PI, PI]`.
#[inline]
pub fn wrap_pi(mut a: f32) -> f32 {
    a = a % TAU;
    if a > PI {
        a -= TAU;
    } else if a <= -PI {
        a += TAU;
    }
    a
}

/// Wrap an angle into `[0, TAU)`.
#[inline]
pub fn wrap_tau(mut a: f32) -> f32 {
    a %= TAU;
    if a < 0.0 {
        a += TAU;
    }
    a
}

/// Wrap a value into `[lo, lo + span)`.
#[inline]
pub fn wrap_range(v: f32, lo: f32, span: f32) -> f32 {
    if span <= 0.0 {
        return lo;
    }
    lo + wrap_tau(v - lo) * (span / TAU)
}

/// Shortest signed delta from `from` to `to` in radians.
#[inline]
pub fn angle_delta(from: f32, to: f32) -> f32 {
    wrap_pi(to - from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_helpers() {
        assert!(nearly_eq(wrap_pi(0.0), 0.0, EPS));
        assert!(nearly_eq(wrap_pi(PI / 2.0), PI / 2.0, EPS));
        // +PI is kept as-is, -PI is mapped to +PI
        assert!(wrap_pi(-PI - 0.01) > 0.0);
        assert!((wrap_pi(-PI - 0.01) - (PI - 0.01)).abs() < 1e-4);
        assert!(nearly_eq(wrap_tau(-0.5), TAU - 0.5, EPS));
        assert!(nearly_eq(wrap_tau(TAU + 0.5), 0.5, EPS));
        // radians in, radians out: wrap_range(370 rad) mod TAU
        assert!(nearly_eq(wrap_range(370.0, 0.0, TAU), 370.0 % TAU, 1e-3));
        assert!(nearly_eq(wrap_range(-0.5, 0.0, TAU), TAU - 0.5, 1e-3));
    }

    #[test]
    fn delta_is_shortest_path() {
        assert!(angle_delta(0.1, 0.2).abs() < 0.2);
        assert!(angle_delta(0.1, -0.1) < 0.0);
    }
}
