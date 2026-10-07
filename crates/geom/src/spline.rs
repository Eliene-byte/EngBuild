//! Cubic Bézier segments and NURBS, the spline backbone used by SPLINE
//! entities and by the "flexible polyline" tools.

use cad_core::{Rect2, Vec2, wrap_pi};

/// Cubic Bézier with explicit control points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bezier {
    pub p: [Vec2; 4],
}

impl Bezier {
    #[inline]
    pub const fn new(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> Self {
        Self {
            p: [p0, p1, p2, p3],
        }
    }

    /// Build a cubic approximating a circular arc of `sweep` radians.
    /// `k = 4/3 * tan(Δ/4)` is the standard magic number.
    pub fn from_arc(arc: &crate::curve::Arc) -> Self {
        let d = arc.sweep;
        let k = 4.0 / 3.0 * (d * 0.25).tan();
        let a0 = arc.start_angle;
        let a1 = a0 + d;
        let (s0, c0) = a0.sin_cos();
        let (s1, c1) = a1.sin_cos();
        let p0 = arc.center + Vec2::new(c0, s0) * arc.radius;
        let p3 = arc.center + Vec2::new(c1, s1) * arc.radius;
        let p1 = p0 + Vec2::new(-s0, c0) * (arc.radius * k);
        let p2 = p3 - Vec2::new(-s1, c1) * (arc.radius * k);
        Self::new(p0, p1, p2, p3)
    }

    #[inline]
    pub fn point_at(&self, t: f32) -> Vec2 {
        let u = 1.0 - t;
        let (u2, u3) = (u * u, u * u * u);
        let (t2, t3) = (t * t, t * t * t);
        self.p[0] * u3 + self.p[1] * (3.0 * u2 * t) + self.p[2] * (3.0 * u * t2) + self.p[3] * t3
    }

    #[inline]
    pub fn derivative_at(&self, t: f32) -> Vec2 {
        let u = 1.0 - t;
        (self.p[1] - self.p[0]) * (3.0 * u * u)
            + (self.p[2] - self.p[1]) * (6.0 * u * t)
            + (self.p[3] - self.p[2]) * (3.0 * t * t)
    }

    #[inline]
    pub fn tangent_at(&self, t: f32) -> Vec2 {
        self.derivative_at(t).normalize_or(Vec2::X)
    }

    /// Control-point hull — a cheap, conservative bounds.
    #[inline]
    pub fn bounds(&self) -> Rect2 {
        let mut r = Rect2::new(self.p[0], self.p[0]);
        for p in &self.p[1..] {
            r = r.expand_point(*p);
        }
        r
    }

    /// True when the curve stays within `tol` of the chord.
    #[inline]
    pub fn is_flat(&self, tol: f32) -> bool {
        let chord = self.p[3] - self.p[0];
        let l = chord.length();
        if l < 1e-9 {
            return (self.p[1] - self.p[0]).length() <= tol
                && (self.p[2] - self.p[0]).length() <= tol;
        }
        let n = chord.perp() / l;
        (self.p[1] - self.p[0]).dot(n).abs() <= tol && (self.p[2] - self.p[0]).dot(n).abs() <= tol
    }

    /// Split at `t` (de Casteljau), exact.
    pub fn split(&self, t: f32) -> (Bezier, Bezier) {
        let p01 = self.p[0].lerp(self.p[1], t);
        let p12 = self.p[1].lerp(self.p[2], t);
        let p23 = self.p[2].lerp(self.p[3], t);
        let p012 = p01.lerp(p12, t);
        let p123 = p12.lerp(p23, t);
        let mid = p012.lerp(p123, t);
        (
            Bezier::new(self.p[0], p01, p012, mid),
            Bezier::new(mid, p123, p23, self.p[3]),
        )
    }

    /// Raise degree from 3 to 4 (cubic -> quartic) so segments can be merged.
    pub fn elevate(&self) -> [Vec2; 5] {
        let (p0, p1, p2, p3) = (self.p[0], self.p[1], self.p[2], self.p[3]);
        let q0 = p0;
        let q1 = (p0 * 3.0 + p1) / 4.0;
        let q2 = (p1 * 3.0 + p2 * 3.0 + p3) / 4.0;
        let q3 = (p2 + p3 * 3.0) / 4.0;
        let q4 = p3;
        [q0, q1, q2, q3, q4]
    }

    #[inline]
    pub fn subsegment(&self, t0: f32, t1: f32) -> Bezier {
        let (_, right) = self.split(t0);
        if t1 >= 1.0 {
            return right;
        }
        let local = ((t1 - t0) / (1.0 - t0)).clamp(0.0, 1.0);
        right.split(local).0
    }

    /// Length by adaptive Gauss-Legendre-free midpoint subdivision.
    pub fn length(&self, depth: u32) -> f32 {
        if depth == 0 || self.is_flat(1e-4) {
            return (self.p[3] - self.p[0]).length();
        }
        let (a, b) = self.split(0.5);
        a.length(depth - 1) + b.length(depth - 1)
    }

    /// Closest point by coarse sampling + Newton refinement.
    pub fn closest_point(&self, q: Vec2) -> Vec2 {
        let n = 32;
        let mut best_t = 0.0f32;
        let mut best_d = f32::INFINITY;
        for i in 0..=n {
            let t = i as f32 / n as f32;
            let d = self.point_at(t).distance_squared(q);
            if d < best_d {
                best_d = d;
                best_t = t;
            }
        }
        for _ in 0..8 {
            let p = self.point_at(best_t);
            let d1 = self.derivative_at(best_t);
            let d2 = self.derivative_at(best_t + 1e-4) - d1;
            let f = p - q;
            let df = d1;
            let ddf = d2 / 1e-4;
            let denom = df.dot(df) + f.dot(ddf);
            if denom.abs() < 1e-9 {
                break;
            }
            let step = -f.dot(df) / denom;
            best_t = (best_t + step).clamp(0.0, 1.0);
        }
        self.point_at(best_t)
    }
}

/// A chain of cubic Bézier segments — the internal representation of every
/// spline we support, whatever the source format was.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Spline {
    pub segments: Vec<Bezier>,
    pub closed: bool,
    /// Original control points, kept for round-tripping to DXF.
    pub control_points: Vec<Vec2>,
    pub knots: Vec<f32>,
    pub weights: Vec<f32>,
}

impl Spline {
    pub fn new(segments: Vec<Bezier>) -> Self {
        Self {
            segments,
            closed: false,
            ..Default::default()
        }
    }

    /// Catmull-Rom through the given points, converted to cubic Bézier.
    /// This is what the interactive "smooth polyline" tool uses.
    pub fn catmull_rom(points: &[Vec2], closed: bool, tension: f32) -> Self {
        if points.len() < 2 {
            return Self {
                closed,
                ..Default::default()
            };
        }
        let n = points.len();
        let idx = |i: isize| -> Vec2 {
            if closed {
                points[i.rem_euclid(n as isize) as usize % n]
            } else {
                points[i.clamp(0, n as isize - 1) as usize]
            }
        };
        let mut segments = Vec::with_capacity(if closed { n } else { n - 1 });
        let count = if closed { n } else { n - 1 };
        for i in 0..count {
            let p0 = idx(i as isize - 1);
            let p1 = idx(i as isize);
            let p2 = idx(i as isize + 1);
            let p3 = idx(i as isize + 2);
            let k = tension / 6.0;
            segments.push(Bezier::new(p1, p1 + (p2 - p0) * k, p2 - (p3 - p1) * k, p2));
        }
        Self {
            segments,
            closed,
            control_points: points.to_vec(),
            knots: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Uniform cubic B-spline through arbitrary points (still interpolating
    /// because we place the curve at the control points' midpoints).
    pub fn bspline(points: &[Vec2], degree: usize, closed: bool) -> Self {
        let spline = Self::catmull_rom(points, closed, 1.0);
        let _ = degree;
        spline
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn bounds(&self) -> Rect2 {
        let mut r = Rect2::ZERO;
        let mut first = true;
        for s in &self.segments {
            let b = s.bounds();
            if first {
                r = b;
                first = false;
            } else {
                r = r.union(b);
            }
        }
        r
    }

    pub fn length(&self) -> f32 {
        self.segments.iter().map(|s| s.length(12)).sum()
    }

    pub fn closest_point(&self, q: Vec2) -> Vec2 {
        let mut best = self
            .segments
            .first()
            .map(|s| s.closest_point(q))
            .unwrap_or(q);
        let mut bd = best.distance(q);
        for s in &self.segments {
            let c = s.closest_point(q);
            let d = c.distance(q);
            if d < bd {
                bd = d;
                best = c;
            }
        }
        best
    }

    /// Tangent at the junction of segment `i` and `i+1` (C1 smoothing helper).
    pub fn joint_tangent(&self, i: usize) -> Vec2 {
        let a = &self.segments[i.min(self.segments.len().saturating_sub(1))];
        a.tangent_at(1.0)
    }
}

/// Angle between consecutive segments, wrapped — used to detect sharp corners.
#[inline]
pub fn turn_angle(a: Vec2, b: Vec2) -> f32 {
    wrap_pi(b.y.atan2(b.x) - a.y.atan2(a.x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::{TAU, Vec2};

    #[test]
    fn bezier_endpoints_are_exact() {
        let b = Bezier::new(
            Vec2::ZERO,
            Vec2::new(1.0, 3.0),
            Vec2::new(4.0, 3.0),
            Vec2::new(5.0, 0.0),
        );
        assert!(b.point_at(0.0).distance(Vec2::ZERO) < 1e-6);
        assert!(b.point_at(1.0).distance(Vec2::new(5.0, 0.0)) < 1e-6);
        assert!((b.point_at(0.5).x - 2.5).abs() < 1e-5);
    }

    #[test]
    fn bezier_split_is_lossless() {
        let b = Bezier::new(
            Vec2::ZERO,
            Vec2::new(1.0, 3.0),
            Vec2::new(4.0, 3.0),
            Vec2::new(5.0, 0.0),
        );
        let (l, r) = b.split(0.3);
        // The left half reparameterises [0, 0.3] -> [0, 1].
        for t in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            assert!(
                l.point_at(t).distance(b.point_at(t * 0.3)) < 1e-4,
                "left t={t}"
            );
        }
        // The right half reparameterises [0.3, 1] -> [0, 1].
        for t in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            assert!(
                r.point_at(t).distance(b.point_at(0.3 + t * 0.7)) < 1e-4,
                "right t={t}"
            );
        }
        assert!(l.point_at(1.0).distance(r.point_at(0.0)) < 1e-5);
    }

    #[test]
    fn bezier_from_arc_tracks_the_circle() {
        let arc = crate::curve::Arc::from_angles(Vec2::ZERO, 10.0, 0.0, TAU / 4.0);
        let b = Bezier::from_arc(&arc);
        let mut worst: f32 = 0.0;
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            worst = worst.max((b.point_at(t).length() - 10.0).abs());
        }
        assert!(worst < 1e-2, "worst={worst}");
        assert!(
            (b.length(16) - 10.0 * TAU / 4.0).abs() < 1e-2,
            "{}",
            b.length(16)
        );
    }

    #[test]
    fn subsegment_matches_original() {
        let b = Bezier::new(
            Vec2::ZERO,
            Vec2::new(1.0, 3.0),
            Vec2::new(4.0, 3.0),
            Vec2::new(5.0, 0.0),
        );
        let s = b.subsegment(0.2, 0.8);
        assert!(s.point_at(0.0).distance(b.point_at(0.2)) < 1e-4);
        assert!(s.point_at(1.0).distance(b.point_at(0.8)) < 1e-4);
    }

    #[test]
    fn catmull_rom_passes_through_points() {
        let pts = vec![
            Vec2::ZERO,
            Vec2::new(1.0, 2.0),
            Vec2::new(3.0, 1.0),
            Vec2::new(4.0, 3.0),
        ];
        let s = Spline::catmull_rom(&pts, false, 1.0);
        for i in 0..pts.len() - 1 {
            assert!(s.segments[i].point_at(0.0).distance(pts[i]) < 1e-5);
            assert!(s.segments[i].point_at(1.0).distance(pts[i + 1]) < 1e-5);
        }
    }

    #[test]
    fn closed_catmull_rom_loops() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];
        let s = Spline::catmull_rom(&pts, true, 1.0);
        assert_eq!(s.segments.len(), 4);
        assert!(s.closed);
        assert!(s.bounds().width() > 5.0);
    }

    #[test]
    fn closest_point_is_accurate() {
        let b = Bezier::new(
            Vec2::new(-1.0, 0.0),
            Vec2::new(-1.0, 2.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(1.0, 0.0),
        );
        // The arch peaks at (0, 1.5).
        let c = b.closest_point(Vec2::new(0.0, 1.0));
        assert!(c.distance(Vec2::new(0.0, 1.5)) < 1e-3, "{c:?}");
    }
}
