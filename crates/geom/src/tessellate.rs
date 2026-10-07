//! Adaptive flattening: turning exact curves into GPU-friendly polylines while
//! keeping the maximum deviation under the caller's tolerance.
//!
//! Flatness is driven by the *chord sagitta*, not by a fixed segment count, so
//! a tiny arc stays cheap and a huge one stays smooth.

use crate::curve::{Arc, Circle, Curve, Ellipse, Line, Polyline};
use crate::spline::{Bezier, Spline};
use cad_core::{Rect2, TAU, Vec2};

/// Flatness/simplification controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TessellationOptions {
    /// Max perpendicular deviation between the curve and the chord, in world units.
    pub tolerance: f32,
    /// Hard cap on generated points per curve (guards against huge zoom-outs).
    pub max_points: usize,
    /// Extra points inserted at curve endpoints so joins are exact.
    pub keep_endpoints: bool,
}

impl Default for TessellationOptions {
    fn default() -> Self {
        // 0.01 drawing units: sub-pixel at any sane zoom.
        Self {
            tolerance: 0.01,
            max_points: 4096,
            keep_endpoints: true,
        }
    }
}

impl TessellationOptions {
    pub fn with_tolerance(tolerance: f32) -> Self {
        Self {
            tolerance,
            ..Self::default()
        }
    }
}

/// A tessellated run of 2D points. `Vec<Vec2>` keeps the renderer allocation-free.
pub type Points = Vec<Vec2>;

/// Maximum angle step that keeps the sagitta under `tol` on a circle of radius `r`.
///
/// Solves `r (1 - cos(θ/2)) = tol` for `θ`.
#[inline]
pub fn arc_angle_step(radius: f32, tol: f32) -> f32 {
    if radius <= tol || !radius.is_finite() {
        return TAU / 8.0;
    }
    let c = (1.0 - tol / radius).clamp(-1.0, 1.0);
    let step = 2.0 * c.acos();
    // Never degenerate.
    step.clamp(1e-3, TAU / 4.0)
}

#[inline]
fn segments_for(radius: f32, sweep: f32, tol: f32, max_points: usize) -> usize {
    if !radius.is_finite() {
        return 1;
    }
    let step = arc_angle_step(radius, tol);
    let n = (sweep.abs() / step).ceil() as usize;
    n.clamp(1, max_points.max(2) - 1)
}

/// Flatten a straight line (trivially 2 points, but kept for uniformity).
pub fn tessellate_line(l: Line) -> Points {
    if l.p0.distance_squared(l.p1) < 1e-14 {
        vec![l.p0]
    } else {
        vec![l.p0, l.p1]
    }
}

pub fn tessellate_arc(a: Arc, opts: &TessellationOptions) -> Points {
    if a.sweep.abs() < 1e-9 {
        return vec![a.start_point()];
    }
    let n = segments_for(a.radius, a.sweep, opts.tolerance, opts.max_points);
    let mut out = Points::with_capacity(n + 1);
    for i in 0..=n {
        let t = i as f32 / n as f32;
        out.push(a.point_at(t));
    }
    dedup_consecutive(&mut out);
    out
}

pub fn tessellate_circle(c: Circle, opts: &TessellationOptions) -> Points {
    let a = c.as_arc();
    let mut pts = tessellate_arc(a, opts);
    // A closed ring must NOT repeat the first point at the end.
    if pts.len() > 1 {
        let first = pts[0];
        let last = *pts.last().unwrap();
        // Scale-aware tolerance: at radius 1e6 the wrap-around error is ~0.1.
        let eps = (1.0 + c.radius) * 1e-5;
        if first.distance(last) < eps {
            pts.pop();
        }
    }
    pts
}

/// Recursive flattening of a cubic Bézier driven by the true control-polygon
/// deviation. This is what makes very eccentric ellipses (1000 x 10) come out
/// smooth instead of faceted.
pub fn flatten_bezier(b: &Bezier, tol: f32, max_depth: u32, out: &mut Points) {
    fn rec(b: Bezier, tol: f32, depth: u32, out: &mut Points) {
        if depth == 0 || b.is_flat(tol) {
            out.push(b.p[3]);
            return;
        }
        let (l, r) = b.split(0.5);
        out.push(l.p[0]);
        rec(l, tol, depth - 1, out);
        rec(r, tol, depth - 1, out);
    }
    if depth_limit(max_depth) == 0 {
        return;
    }
    out.push(b.p[0]);
    rec(*b, tol, max_depth, out);
    dedup_consecutive(out);
}

#[inline]
fn depth_limit(d: u32) -> u32 {
    d
}

/// Four cubic Béziers approximate a full ellipse to within ~1e-4 of its radii,
/// which is far below any sane drawing tolerance.
pub fn ellipse_to_beziers(e: Ellipse) -> [Bezier; 4] {
    use cad_core::TAU as TWO_PI;
    // The circular-arc magic number for a quarter sweep:
    // k = 4/3 * tan(Δ/4) with Δ = π/2, i.e. tan(π/8) ≈ 0.5522847.
    let k = 4.0 / 3.0 * ((TWO_PI / 4.0) * 0.25).tan();
    let (u, v) = (e.major_dir(), e.minor_dir());
    let (a, b) = (e.semi_major(), e.semi_minor());
    let mut out = [Bezier::new(e.center, e.center, e.center, e.center); 4];
    for i in 0..4 {
        let t0 = TWO_PI * i as f32 / 4.0;
        let t1 = TWO_PI * (i as f32 + 1.0) / 4.0;
        let (s0, c0) = t0.sin_cos();
        let (s1, c1) = t1.sin_cos();
        let p0 = e.center + u * (c0 * a) + v * (s0 * b);
        let p3 = e.center + u * (c1 * a) + v * (s1 * b);
        let d0 = u * (-s0 * a) + v * (c0 * b);
        let d1 = u * (-s1 * a) + v * (c1 * b);
        out[i] = Bezier::new(p0, p0 + d0 * k, p3 - d1 * k, p3);
    }
    out
}

pub fn tessellate_ellipse(e: Ellipse, opts: &TessellationOptions) -> Points {
    let depth = 16u32.min(opts.max_points.ilog2().max(4));
    let mut out = Points::with_capacity(opts.max_points.min(256));
    for b in ellipse_to_beziers(e) {
        flatten_bezier(&b, opts.tolerance, depth, &mut out);
    }
    dedup_consecutive(&mut out);
    if out.len() > 1 && out[0].distance(*out.last().unwrap()) < 1e-5 {
        out.pop();
    }
    out
}

/// Flatten a spline chain (used by the renderer and by export code).
pub fn tessellate_spline(s: &Spline, opts: &TessellationOptions) -> Points {
    let mut out = Points::new();
    for b in &s.segments {
        flatten_bezier(b, opts.tolerance, 16, &mut out);
    }
    dedup_consecutive(&mut out);
    out
}

pub fn tessellate_polyline(p: &Polyline, opts: &TessellationOptions) -> Points {
    let n = p.segment_count();
    if n == 0 {
        return p.vertices.clone();
    }
    let mut out = Points::with_capacity(n * 4 + 1);
    if opts.keep_endpoints {
        out.push(p.point(0));
    }
    for i in 0..n {
        let p0 = p.point(i);
        let p1 = p.point(i + 1);
        let b = p.bulge_at(i);
        if b.is_arc() {
            let arc = p.segment_arc(i);
            let m = segments_for(arc.radius, arc.sweep, opts.tolerance, opts.max_points);
            for k in 1..=m {
                let t = k as f32 / m as f32;
                out.push(arc.point_at(t));
            }
        } else if p0.distance_squared(p1) > 1e-14 {
            out.push(p1);
        }
    }
    dedup_consecutive(&mut out);
    out
}

/// Flatten any curve.
pub fn tessellate(c: &Curve, opts: &TessellationOptions) -> Points {
    match c {
        Curve::Line(l) => tessellate_line(*l),
        Curve::Arc(a) => tessellate_arc(*a, opts),
        Curve::Circle(ci) => tessellate_circle(*ci, opts),
        Curve::Ellipse(e) => tessellate_ellipse(*e, opts),
        Curve::Polyline(p) => tessellate_polyline(p, opts),
    }
}

/// Drop points that are closer than a hair — keeps vertex buffers small.
pub fn dedup_consecutive(pts: &mut Points) {
    const MIN: f32 = 1e-7;
    pts.dedup_by(|a, b| a.distance_squared(*b) < MIN * MIN);
}

/// Append a 3D polyline built from a 2D tessellation (Z lifted by `z`).
pub fn lift_z(pts: &[Vec2], z: f32) -> Vec<[f32; 3]> {
    pts.iter().map(|p| [p.x, p.y, z]).collect()
}

/// Bounds of a point run, for quick culling before tessellation.
pub fn points_bounds(pts: &[Vec2]) -> Rect2 {
    let mut r = Rect2::ZERO;
    for p in pts {
        r = r.expand_point(*p);
    }
    r
}

/// Shortest distance from `p` to the segment `a..b`.
pub fn distance_point_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let d = b - a;
    let l2 = d.length_squared();
    if l2 < 1e-18 {
        return p.distance(a);
    }
    let t = ((p - a).dot(d) / l2).clamp(0.0, 1.0);
    (a + d * t).distance(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::PI;

    fn max_sagitta(radius: f32, sweep: f32, opts: &TessellationOptions) -> f32 {
        let pts = tessellate_arc(
            Arc {
                center: Vec2::ZERO,
                radius,
                start_angle: 0.0,
                sweep,
            },
            opts,
        );
        let mut worst: f32 = 0.0;
        for i in 1..pts.len().saturating_sub(1) {
            // distance from the exact circle at this angle
            let exact = Vec2::ZERO + pts[i].normalize_or(Vec2::X) * radius;
            worst = worst.max(exact.distance(pts[i]));
        }
        // also measure against the chord midpoints
        for w in pts.windows(2) {
            let mid = (w[0] + w[1]) * 0.5;
            let ang = mid.y.atan2(mid.x);
            let exact = Vec2::ZERO + Vec2::new(ang.cos(), ang.sin()) * radius;
            worst = worst.max(exact.distance(mid));
        }
        worst
    }

    #[test]
    fn arc_flatness_respects_tolerance() {
        let opts = TessellationOptions::with_tolerance(0.05);
        for radius in [0.5f32, 5.0, 500.0, 5000.0] {
            let s = max_sagitta(radius, PI, &opts);
            assert!(s <= 0.05 + 1e-3, "r={radius} sagitta={s}");
        }
    }

    #[test]
    fn circle_ring_has_no_duplicate_endpoint() {
        let pts = tessellate_circle(
            Circle::new(Vec2::ZERO, 10.0),
            &TessellationOptions::default(),
        );
        assert!(pts.len() >= 32);
        assert!(pts[0].distance(*pts.last().unwrap()) > 1e-3);
    }

    #[test]
    fn arc_endpoints_are_exact() {
        let a = Arc::from_angles(Vec2::new(1.0, 2.0), 3.0, 0.3, 2.1);
        let pts = tessellate_arc(a, &TessellationOptions::with_tolerance(0.01));
        assert!(pts[0].distance(a.start_point()) < 1e-6);
        assert!(pts.last().unwrap().distance(a.end_point()) < 1e-6);
    }

    #[test]
    fn polyline_keeps_all_vertices() {
        let p = Polyline::new(
            vec![
                Vec2::ZERO,
                Vec2::new(10.0, 0.0),
                Vec2::new(10.0, 10.0),
                Vec2::new(20.0, 10.0),
            ],
            false,
        );
        let pts = tessellate_polyline(&p, &TessellationOptions::default());
        assert_eq!(pts.len(), 4);
    }

    #[test]
    fn tiny_arcs_are_cheap() {
        let a = Arc::from_angles(Vec2::ZERO, 100.0, 0.0, 0.001);
        let pts = tessellate_arc(a, &TessellationOptions::default());
        assert_eq!(pts.len(), 2);
    }

    #[test]
    fn max_points_is_respected() {
        let opts = TessellationOptions {
            tolerance: 1e-9,
            max_points: 64,
            keep_endpoints: true,
        };
        let pts = tessellate_circle(Circle::new(Vec2::ZERO, 1.0), &opts);
        assert!(pts.len() <= 64, "{}", pts.len());
    }

    #[test]
    fn ellipse_flatness() {
        let e = Ellipse::new(Vec2::ZERO, Vec2::new(1000.0, 0.0), 0.01);
        let opts = TessellationOptions::with_tolerance(0.05);
        let pts = tessellate_ellipse(e, &opts);
        // Sample the curve and confirm the polyline stays within tolerance.
        let mut worst: f32 = 0.0;
        for i in 0..=400 {
            let t = TAU * i as f32 / 400.0;
            let p = e.point_at(t);
            let mut best = f32::INFINITY;
            for w in pts.windows(2) {
                best = best.min(distance_point_segment(p, w[0], w[1]));
            }
            // The ring is closed, so the wrap-around segment matters a lot on
            // eccentric ellipses - do not forget it.
            best = best.min(distance_point_segment(p, pts[pts.len() - 1], pts[0]));
            worst = worst.max(best);
        }
        assert!(worst <= 0.05 + 1e-2, "worst={worst}");
    }
}
