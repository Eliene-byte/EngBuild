//! Analytic curve/curve intersection.
//!
//! Lines and circles are solved in closed form (quadratic and quartic roots),
//! so hits are exact rather than "close enough". Polylines and splines are
//! subdivided into these primitives, which keeps the result consistent with
//! what the user sees on screen.

use crate::curve::{Arc, Circle, Curve, Line, Polyline};
use crate::tessellate::{TessellationOptions, tessellate};
use cad_core::{EPS, Vec2};

/// What kind of feature produced an intersection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitKind {
    /// Straight segments crossing (2 lines).
    Crossing,
    /// A line touching or cutting a circular feature.
    ArcOnLine,
    /// Two circles/arcs meeting.
    ArcOnArc,
    /// Endpoint of a polyline or spline sitting on another entity.
    VertexOnCurve,
    /// Coincident / overlapping geometry.
    Overlap,
}

/// A single intersection result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// World-space point.
    pub point: Vec2,
    /// Parameters along each participating curve (for arc-on-line, `[t_line, t_arc]`).
    pub params: [f32; 2],
    pub kind: HitKind,
}

impl Hit {
    #[inline]
    fn new(point: Vec2, params: [f32; 2], kind: HitKind) -> Self {
        Self {
            point,
            params,
            kind,
        }
    }
}

/// Intersection of two segments (or rays when `clamp` is false).
/// Returns both `(t, u)` pairs when they overlap collinearly.
pub fn segment_segment(a: Line, b: Line, clamp: bool) -> Vec<Hit> {
    let mut out = Vec::new();
    let r = a.delta();
    let s = b.delta();
    let denom = r.cross(s);
    let qp = b.p0 - a.p0;

    if denom.abs() < 1e-12 {
        // Parallel. If also collinear, report the overlap endpoints.
        if qp.cross(r).abs() > 1e-9 * r.length().max(1.0) {
            return out; // parallel, disjoint
        }
        let rl = r.length_squared();
        let t0 = if rl > 1e-18 { qp.dot(r) / rl } else { 0.0 };
        let t1 = t0 + if rl > 1e-18 { s.dot(r) / rl } else { 0.0 };
        let (t0, t1) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
        let (lo, hi) = (t0.max(0.0), t1.min(1.0));
        if lo <= hi {
            if (hi - lo).abs() < 1e-9 {
                out.push(Hit::new(
                    a.at(lo.clamp(0.0, 1.0)),
                    [lo, 0.0],
                    HitKind::Overlap,
                ));
            } else {
                out.push(Hit::new(a.at(lo), [lo, 0.0], HitKind::Overlap));
                out.push(Hit::new(a.at(hi), [hi, 1.0], HitKind::Overlap));
            }
        }
        return out;
    }

    let t = qp.cross(s) / denom;
    let u = qp.cross(r) / denom;
    if clamp && (!(0.0..=1.0).contains(&t) || !(0.0..=1.0).contains(&u)) {
        return out;
    }
    out.push(Hit::new(a.at(t), [t, u], HitKind::Crossing));
    out
}

/// Intersections of a line with a circle. Returns 0, 1 (tangent) or 2 points.
pub fn line_circle(l: Line, c: Circle, clamp_line: bool) -> Vec<Hit> {
    let d = l.delta();
    let dd = d.length_squared();
    if dd < 1e-18 {
        return Vec::new();
    }
    let f = l.p0 - c.center;
    let a = dd;
    let b = 2.0 * f.dot(d);
    let cc = f.length_squared() - c.radius * c.radius;
    let disc = b * b - 4.0 * a * cc;
    if disc < -EPS {
        return Vec::new();
    }
    let disc = disc.max(0.0);
    let sq = disc.sqrt();
    let ts = if sq.abs() < 1e-9 {
        vec![-b / (2.0 * a)]
    } else {
        vec![(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)]
    };
    ts.into_iter()
        .filter(|t| !clamp_line || (0.0..=1.0).contains(t))
        .map(|t| {
            let p = l.at(t);
            let angle = (p - c.center).y.atan2((p - c.center).x);
            Hit::new(p, [t, angle], HitKind::ArcOnLine)
        })
        .collect()
}

/// Intersections of a line with an arc, restricted to the angular range.
pub fn line_arc(l: Line, arc: Arc, clamp_line: bool) -> Vec<Hit> {
    if !arc.radius.is_finite() {
        return segment_segment(l, Line::new(arc.start_point(), arc.end_point()), clamp_line);
    }
    line_circle(l, Circle::new(arc.center, arc.radius), clamp_line)
        .into_iter()
        .filter(|h| arc.contains_angle(h.params[1]))
        .collect()
}

/// Intersections of two circles. `None` when concentric or non-intersecting.
pub fn circle_circle(a: Circle, b: Circle) -> Vec<Hit> {
    let r0 = a.radius;
    let r1 = b.radius;
    let d_vec = b.center - a.center;
    let d = d_vec.length();
    if d < 1e-9 {
        return Vec::new(); // concentric
    }
    if d > r0 + r1 + EPS || d < (r0 - r1).abs() - EPS {
        return Vec::new();
    }
    let aa = (d * d - r1 * r1 + r0 * r0) / (2.0 * d);
    let h2 = r0 * r0 - aa * aa;
    let h = if h2 <= 0.0 { 0.0 } else { h2.sqrt() };
    let mid = a.center + d_vec * (aa / d);
    let perp = d_vec.perp() * (h / d);
    let ang_a = |p: Vec2| (p - a.center).y.atan2((p - a.center).x);
    let ang_b = |p: Vec2| (p - b.center).y.atan2((p - b.center).x);
    if h < 1e-9 {
        let p = mid;
        return vec![Hit::new(p, [ang_a(p), ang_b(p)], HitKind::ArcOnArc)];
    }
    let p0 = mid + perp;
    let p1 = mid - perp;
    vec![
        Hit::new(p0, [ang_a(p0), ang_b(p0)], HitKind::ArcOnArc),
        Hit::new(p1, [ang_a(p1), ang_b(p1)], HitKind::ArcOnArc),
    ]
}

/// Intersections of two arcs (angle-filtered).
pub fn arc_arc(a: Arc, b: Arc) -> Vec<Hit> {
    circle_circle(
        Circle::new(a.center, a.radius),
        Circle::new(b.center, b.radius),
    )
    .into_iter()
    .filter(|h| a.contains_angle(h.params[0]) && b.contains_angle(h.params[1]))
    .collect()
}

/// Generic dispatcher for the two curve primitives.
pub fn curve_curve(a: &Curve, b: &Curve, clamp: bool) -> Vec<Hit> {
    use Curve::*;
    match (a, b) {
        (Line(x), Line(y)) => segment_segment(*x, *y, clamp),
        (Line(x), Circle(c)) => line_circle(*x, *c, clamp),
        (Circle(c), Line(y)) => line_circle(*y, *c, clamp),
        (Line(x), Arc(r)) => line_arc(*x, *r, clamp),
        (Arc(r), Line(y)) => line_arc(*y, *r, clamp),
        (Circle(c), Circle(d)) => circle_circle(*c, *d),
        (Circle(c), Arc(r)) => arc_arc(c.as_arc(), *r),
        (Arc(r), Circle(c)) => arc_arc(*r, c.as_arc()),
        (Arc(x), Arc(y)) => arc_arc(*x, *y),
        _ => Vec::new(), // composite curves are handled by the caller
    }
}

/// Intersections involving a polyline (split into its primitive segments).
pub fn polyline_curve(p: &Polyline, other: &Curve, clamp: bool) -> Vec<Hit> {
    let mut out = Vec::new();
    let n = p.segment_count();
    for i in 0..n {
        let seg = if p.bulge_at(i).is_arc() {
            Curve::Arc(p.segment_arc(i))
        } else {
            Curve::Line(p.segment(i))
        };
        out.extend(curve_curve(&seg, other, clamp));
    }
    out
}

/// Intersections between two arbitrary curves.
///
/// Primitives (line/arc/circle) go through the closed-form solvers. Composites
/// are **decomposed** into primitives rather than blindly sampled, so a bulge
/// arc inside a polyline still yields exact hits.
pub fn intersect(a: &Curve, b: &Curve, opts: &TessellationOptions) -> Vec<Hit> {
    match (a, b) {
        (Curve::Polyline(pa), Curve::Polyline(pb)) => {
            // Fast path: pairwise primitive-vs-primitive over the segments.
            let mut acc = Vec::new();
            let (na, nb) = (pa.segment_count(), pb.segment_count());
            for i in 0..na {
                let sa = segment_curve(pa, i);
                for j in 0..nb {
                    let sb = segment_curve(pb, j);
                    acc.extend(curve_curve(&sa, &sb, true));
                }
            }
            if !acc.is_empty() {
                acc
            } else {
                let ta = tessellate(a, opts);
                let tb = tessellate(b, opts);
                segment_runs(&ta, &tb, true)
            }
        }
        (Curve::Polyline(p), other) => polyline_curve(p, other, true),
        (other, Curve::Polyline(p)) => polyline_curve(p, other, true),
        (Curve::Ellipse(_), _) | (_, Curve::Ellipse(_)) => {
            // Ellipse-vs-anything has no closed form in a polyline world:
            // flatten both and intersect segment runs.
            let ta = tessellate(a, opts);
            let tb = tessellate(b, opts);
            segment_runs(&ta, &tb, true)
        }
        _ => curve_curve(a, b, true),
    }
}

/// The `i`-th segment of a polyline as a primitive curve (arc when it has a bulge).
#[inline]
fn segment_curve(p: &Polyline, i: usize) -> Curve {
    if p.bulge_at(i).is_arc() {
        Curve::Arc(p.segment_arc(i))
    } else {
        Curve::Line(p.segment(i))
    }
}

/// Brute-force but *exact enough* segment intersection over two point runs.
/// Used as the fallback for ellipse vs polyline and similar mixed cases.
pub fn segment_runs(a: &[Vec2], b: &[Vec2], clamp: bool) -> Vec<Hit> {
    let mut out = Vec::new();
    for w in a.windows(2) {
        let la = Line::new(w[0], w[1]);
        for v in b.windows(2) {
            let lb = Line::new(v[0], v[1]);
            out.extend(segment_segment(la, lb, clamp));
        }
    }
    out
}

/// Closest points between two segments (used by "closest" object snaps).
///
/// Uses the standard clamped parameter solve plus the re-projection step, so a
/// vertex-to-edge case lands on the vertex instead of overshooting it.
pub fn closest_segment_segment(a: Line, b: Line) -> (Vec2, Vec2) {
    let d1 = a.delta();
    let d2 = b.delta();
    let r = a.p0 - b.p0;
    let a11 = d1.dot(d1);
    let a22 = d2.dot(d2);
    let a12 = d1.dot(d2);
    let b1 = r.dot(d1);
    let b2 = r.dot(d2);
    let den = a11 * a22 - a12 * a12;

    let (mut s, mut t);
    if den.abs() < 1e-12 {
        // Parallel: project the origin of one onto the other.
        s = 0.0;
        t = if a22 > 1e-12 { b2 / a22 } else { 0.0 };
    } else {
        s = (b1 * a22 - b2 * a12) / den;
        t = (a11 * b2 - a12 * b1) / den;
    }
    s = s.clamp(0.0, 1.0);
    t = t.clamp(0.0, 1.0);
    if den.abs() >= 1e-12 {
        // Re-solve the other parameter given the clamped one, then clamp back.
        t = (a12 * s + b2) / a22;
        if !(0.0..=1.0).contains(&t) {
            t = t.clamp(0.0, 1.0);
            s = ((a12 * t - b1) / a11).clamp(0.0, 1.0);
        }
    }
    (a.at(s), b.at(t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::{TAU, Vec2};

    #[test]
    fn crossing_segments() {
        let a = Line::new(Vec2::ZERO, Vec2::new(10.0, 0.0));
        let b = Line::new(Vec2::new(5.0, -5.0), Vec2::new(5.0, 5.0));
        let h = segment_segment(a, b, true);
        assert_eq!(h.len(), 1);
        assert!(h[0].point.distance(Vec2::new(5.0, 0.0)) < 1e-5);
        assert_eq!(h[0].kind, HitKind::Crossing);
    }

    #[test]
    fn segments_that_do_not_touch() {
        let a = Line::new(Vec2::ZERO, Vec2::new(10.0, 0.0));
        let b = Line::new(Vec2::new(11.0, -5.0), Vec2::new(11.0, 5.0));
        assert!(segment_segment(a, b, true).is_empty());
    }

    #[test]
    fn collinear_overlap() {
        let a = Line::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        let b = Line::new(Vec2::new(5.0, 0.0), Vec2::new(15.0, 0.0));
        let h = segment_segment(a, b, true);
        assert_eq!(h.len(), 2);
        assert!((h[0].point.x - 5.0).abs() < 1e-5);
        assert!((h[1].point.x - 10.0).abs() < 1e-5);
        assert_eq!(h[0].kind, HitKind::Overlap);
    }

    #[test]
    fn secant_line_circle() {
        let l = Line::new(Vec2::new(-10.0, 0.0), Vec2::new(10.0, 0.0));
        let c = Circle::new(Vec2::ZERO, 5.0);
        let h = line_circle(l, c, true);
        assert_eq!(h.len(), 2);
        assert!(h.iter().all(|x| (x.point.length() - 5.0).abs() < 1e-4));
    }

    #[test]
    fn tangent_line_circle() {
        let l = Line::new(Vec2::new(-10.0, 5.0), Vec2::new(10.0, 5.0));
        let c = Circle::new(Vec2::ZERO, 5.0);
        let h = line_circle(l, c, true);
        assert_eq!(h.len(), 1);
        assert!((h[0].point.y - 5.0).abs() < 1e-4);
    }

    #[test]
    fn missing_line_circle() {
        let l = Line::new(Vec2::new(-10.0, 6.0), Vec2::new(10.0, 6.0));
        assert!(line_circle(l, Circle::new(Vec2::ZERO, 5.0), true).is_empty());
    }

    #[test]
    fn line_hits_only_the_covered_arc() {
        let full = Arc::from_angles(Vec2::ZERO, 5.0, 0.0, TAU / 2.0); // upper half
        let l = Line::new(Vec2::new(-10.0, 0.0), Vec2::new(10.0, 0.0));
        let h = line_arc(l, full, true);
        // The chord itself only touches the arc endpoints.
        assert!(h.len() <= 2);
        let secant = Line::new(Vec2::new(-10.0, 1.0), Vec2::new(10.0, 1.0));
        let h2 = line_arc(secant, full, true);
        assert_eq!(h2.len(), 2, "{h2:?}");
        for hit in &h2 {
            assert!(hit.point.y > 0.0);
        }
    }

    #[test]
    fn crossing_circles() {
        let a = Circle::new(Vec2::ZERO, 5.0);
        let b = Circle::new(Vec2::new(8.0, 0.0), 5.0);
        let h = circle_circle(a, b);
        assert_eq!(h.len(), 2);
        for x in &h {
            assert!((x.point.length() - 5.0).abs() < 1e-4);
            assert!((x.point.distance(b.center) - 5.0).abs() < 1e-4);
        }
    }

    #[test]
    fn internally_tangent_circles() {
        let a = Circle::new(Vec2::ZERO, 5.0);
        let b = Circle::new(Vec2::new(3.0, 0.0), 2.0);
        let h = circle_circle(a, b);
        assert_eq!(h.len(), 1);
        assert!((h[0].point.x - 5.0).abs() < 1e-3, "{:?}", h[0].point);
    }

    #[test]
    fn disjoint_and_concentric() {
        let a = Circle::new(Vec2::ZERO, 5.0);
        assert!(circle_circle(a, Circle::new(Vec2::new(50.0, 0.0), 5.0)).is_empty());
        assert!(circle_circle(a, Circle::new(Vec2::ZERO, 2.0)).is_empty());
        assert!(circle_circle(a, Circle::new(Vec2::new(3.0, 0.0), 9.0)).is_empty());
    }

    #[test]
    fn generic_curve_curve_dispatch() {
        let a = Curve::Line(Line::new(Vec2::ZERO, Vec2::new(10.0, 0.0)));
        let b = Curve::Circle(Circle::new(Vec2::new(5.0, 0.0), 2.0));
        assert_eq!(curve_curve(&a, &b, true).len(), 2);
        let c = Curve::Arc(Arc::from_angles(Vec2::ZERO, 5.0, 0.0, TAU));
        assert_eq!(curve_curve(&b, &c, true).len(), 2);
    }

    #[test]
    fn polyline_cross() {
        let p = Polyline::new(vec![Vec2::ZERO, Vec2::new(10.0, 0.0)], false);
        let q = Polyline::new(vec![Vec2::new(5.0, -5.0), Vec2::new(5.0, 5.0)], false);
        let h = intersect(
            &Curve::Polyline(p),
            &Curve::Polyline(q),
            &TessellationOptions::default(),
        );
        assert_eq!(h.len(), 1);
        assert!((h[0].point.x - 5.0).abs() < 1e-5);
    }

    #[test]
    fn closest_points_between_segments() {
        let a = Line::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        let b = Line::new(Vec2::new(4.0, 3.0), Vec2::new(6.0, 7.0));
        let (pa, pb) = closest_segment_segment(a, b);
        // The closest feature is the bottom endpoint of b projected onto a.
        assert!(pa.distance(Vec2::new(4.0, 0.0)) < 1e-4, "{pa:?}");
        assert!(pb.distance(Vec2::new(4.0, 3.0)) < 1e-4, "{pb:?}");
    }
}
