//! The 2D curve vocabulary of the drawing model.

use crate::bulge::Bulge;
use cad_core::{EPS, Rect2, TAU, Vec2, wrap_pi};

/// A straight segment between two points.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Line {
    pub p0: Vec2,
    pub p1: Vec2,
}

impl Line {
    #[inline]
    pub const fn new(p0: Vec2, p1: Vec2) -> Self {
        Self { p0, p1 }
    }
    #[inline]
    pub fn delta(self) -> Vec2 {
        self.p1 - self.p0
    }
    #[inline]
    pub fn length(self) -> f32 {
        self.p1.distance(self.p0)
    }
    /// Unit direction from `p0` to `p1`.
    #[inline]
    pub fn dir(self) -> Vec2 {
        self.delta().normalize_or(Vec2::X)
    }
    /// Unit normal (left-hand side of travel direction).
    #[inline]
    pub fn normal(self) -> Vec2 {
        self.dir().perp()
    }
    #[inline]
    pub fn midpoint(self) -> Vec2 {
        (self.p0 + self.p1) * 0.5
    }
    /// Point at parameter `t` (unclamped).
    #[inline]
    pub fn at(self, t: f32) -> Vec2 {
        self.p0 + self.delta() * t
    }
    #[inline]
    pub fn bounds(self) -> Rect2 {
        Rect2::new(self.p0.min(self.p1), self.p0.max(self.p1))
    }
    /// Squared distance from `p` to the infinite line.
    #[inline]
    pub fn distance_squared(self, p: Vec2) -> f32 {
        let n = self.normal();
        n.dot(p - self.p0) * n.dot(p - self.p0)
    }
    /// Closest point on the *segment* (clamped to the endpoints).
    #[inline]
    pub fn closest_point(self, p: Vec2) -> Vec2 {
        let d = self.delta();
        let l2 = d.length_squared();
        if l2 < EPS * EPS {
            return self.p0;
        }
        self.at(((p - self.p0).dot(d) / l2).clamp(0.0, 1.0))
    }
    #[inline]
    pub fn distance_to(self, p: Vec2) -> f32 {
        self.closest_point(p).distance(p)
    }
    /// Extend (or trim) the line by `d` at each end.
    #[inline]
    pub fn extended(self, d: f32) -> Self {
        let dir = self.dir();
        Self::new(self.p0 - dir * d, self.p1 + dir * d)
    }
    /// Flip direction.
    #[inline]
    pub fn reversed(self) -> Self {
        Self::new(self.p1, self.p0)
    }
    /// True when the infinite lines intersect; returns the parameters.
    pub fn intersect(self, o: Line) -> Option<(f32, f32)> {
        let r = self.delta();
        let s = o.delta();
        let denom = r.cross(s);
        if denom.abs() < 1e-12 {
            return None; // parallel or degenerate
        }
        let qp = o.p0 - self.p0;
        let t = qp.cross(s) / denom;
        let u = qp.cross(r) / denom;
        Some((t, u))
    }
}

/// A circular arc: center, radius and a **signed** sweep starting at `start_angle`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arc {
    pub center: Vec2,
    pub radius: f32,
    pub start_angle: f32,
    /// Signed sweep in radians; positive = counter-clockwise.
    pub sweep: f32,
}

impl Arc {
    #[inline]
    pub const fn new(center: Vec2, radius: f32, start_angle: f32, sweep: f32) -> Self {
        Self {
            center,
            radius,
            start_angle,
            sweep,
        }
    }

    /// Build from two angles through a center (always CCW from `a0` to `a1`).
    pub fn from_angles(center: Vec2, radius: f32, a0: f32, a1: f32) -> Self {
        let mut sweep = a1 - a0;
        while sweep <= 0.0 {
            sweep += TAU;
        }
        while sweep > TAU {
            sweep -= TAU;
        }
        Self {
            center,
            radius,
            start_angle: a0,
            sweep,
        }
    }

    /// Build from the DXF form: center, radius, start **and** end point angles.
    pub fn from_endpoints(center: Vec2, radius: f32, a0: f32, a1: f32) -> Self {
        let sweep = wrap_pi(a1 - a0);
        Self {
            center,
            radius,
            start_angle: a0,
            sweep,
        }
    }

    #[inline]
    pub fn start_point(self) -> Vec2 {
        self.center + Vec2::new(self.start_angle.cos(), self.start_angle.sin()) * self.radius
    }
    #[inline]
    pub fn end_point(self) -> Vec2 {
        self.center
            + Vec2::new(
                (self.start_angle + self.sweep).cos(),
                (self.start_angle + self.sweep).sin(),
            ) * self.radius
    }
    #[inline]
    pub fn end_angle(self) -> f32 {
        self.start_angle + self.sweep
    }
    #[inline]
    pub fn is_ccw(self) -> bool {
        self.sweep >= 0.0
    }
    #[inline]
    pub fn point_at(self, t: f32) -> Vec2 {
        let a = self.start_angle + self.sweep * t;
        self.center + Vec2::new(a.cos(), a.sin()) * self.radius
    }
    #[inline]
    pub fn tangent_at(self, t: f32) -> Vec2 {
        let a = self.start_angle + self.sweep * t;
        let s = if self.sweep >= 0.0 { 1.0 } else { -1.0 };
        Vec2::new(-a.sin() * s, a.cos() * s)
    }
    /// Angle of the point relative to the center, wrapped into the arc's range.
    #[inline]
    pub fn angle_of(self, p: Vec2) -> f32 {
        (p - self.center).y.atan2((p - self.center).x)
    }
    #[inline]
    pub fn is_full_circle(self) -> bool {
        self.sweep.abs() >= TAU - 1e-4
    }
    /// Is `angle` inside the swept range? A full circle contains everything.
    #[inline]
    pub fn contains_angle(self, angle: f32) -> bool {
        if self.is_full_circle() {
            return true;
        }
        let rel = wrap_pi(angle - self.start_angle);
        if self.sweep >= 0.0 {
            rel >= -1e-6 && rel <= self.sweep + 1e-6
        } else {
            rel <= 1e-6 && rel >= self.sweep - 1e-6
        }
    }
    /// Signed area contribution (Green's theorem) — used by polyline closing.
    #[inline]
    pub fn signed_area(self) -> f32 {
        let a = self.start_angle;
        let b = self.end_angle();
        0.5 * self.radius * self.radius * (b - a).sin()
    }
    #[inline]
    pub fn length(self) -> f32 {
        self.radius * self.sweep.abs()
    }
    /// Axis-aligned bounds of the arc, accounting for the cardinal extremes
    /// that actually lie inside the sweep.
    pub fn bounds(self) -> Rect2 {
        let mut min = Vec2::new(self.center.x + self.radius, self.center.y + self.radius);
        let mut max = Vec2::new(self.center.x - self.radius, self.center.y - self.radius);
        let consider = |ang: f32, min: &mut Vec2, max: &mut Vec2| {
            if self.contains_angle(ang) {
                let p = self.center + Vec2::new(ang.cos(), ang.sin()) * self.radius;
                *min = min.min(p);
                *max = max.max(p);
            }
        };
        consider(wrap_pi(self.start_angle), &mut min, &mut max);
        consider(wrap_pi(self.end_angle()), &mut min, &mut max);
        for k in -4..=4 {
            let ang = cad_core::FRAC_PI_2 * k as f32;
            consider(ang, &mut min, &mut max);
        } // Safety net: if the sweep covers a full turn the whole circle is in.
        if self.is_full_circle() {
            min = self.center - Vec2::splat(self.radius);
            max = self.center + Vec2::splat(self.radius);
        }
        Rect2::new(min, max)
    }
    pub fn closest_point(self, p: Vec2) -> Vec2 {
        if self.is_full_circle() {
            return self.center + (p - self.center).normalize_or(Vec2::X) * self.radius;
        }
        let ang = self.angle_of(p);
        // Try the raw angle and the +-2π shifted variants; keep the one on the arc.
        let mut best = None::<(f32, Vec2)>;
        for k in -1..=1 {
            let a = ang + k as f32 * TAU;
            if self.contains_angle(a) {
                let q = self.center + Vec2::new(a.cos(), a.sin()) * self.radius;
                let d = q.distance(p);
                if best.is_none_or(|(_, qb)| d < qb.distance(p)) {
                    best = Some((a, q));
                }
            }
        }
        match best {
            Some((_, q)) => q,
            None => {
                // Outside the angular range: clamp to the nearer endpoint.
                let d0 = self.start_point().distance(p);
                let d1 = self.end_point().distance(p);
                if d0 <= d1 {
                    self.start_point()
                } else {
                    self.end_point()
                }
            }
        }
    }
}

/// Full circle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    pub center: Vec2,
    pub radius: f32,
}

impl Circle {
    #[inline]
    pub const fn new(center: Vec2, radius: f32) -> Self {
        Self { center, radius }
    }
    #[inline]
    pub fn as_arc(self) -> Arc {
        Arc {
            center: self.center,
            radius: self.radius,
            start_angle: 0.0,
            sweep: TAU,
        }
    }
    #[inline]
    pub fn bounds(self) -> Rect2 {
        Rect2::new(
            self.center - Vec2::splat(self.radius),
            self.center + Vec2::splat(self.radius),
        )
    }
    #[inline]
    pub fn area(self) -> f32 {
        self.radius * self.radius * cad_core::PI
    }
    #[inline]
    pub fn perimeter(self) -> f32 {
        TAU * self.radius
    }
    #[inline]
    pub fn closest_point(self, p: Vec2) -> Vec2 {
        self.center + (p - self.center).normalize_or(Vec2::X) * self.radius
    }
    #[inline]
    pub fn contains(self, p: Vec2) -> bool {
        p.distance_squared(self.center) <= self.radius * self.radius
    }
}

/// Axis-aligned ellipse (an *ellipse* is stored rotated; see `xform`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse {
    pub center: Vec2,
    pub major_axis: Vec2,
    /// Ratio in `(0, 1]`.
    pub ratio: f32,
}

impl Ellipse {
    #[inline]
    pub fn new(center: Vec2, major: Vec2, ratio: f32) -> Self {
        Self {
            center,
            major_axis: major,
            ratio: ratio.clamp(1e-4, 1.0),
        }
    }
    #[inline]
    pub fn semi_major(self) -> f32 {
        self.major_axis.length()
    }
    #[inline]
    pub fn semi_minor(self) -> f32 {
        self.semi_major() * self.ratio
    }
    #[inline]
    pub fn rotation(self) -> f32 {
        self.major_axis.y.atan2(self.major_axis.x)
    }
    /// Unit vector along the major axis.
    #[inline]
    pub fn major_dir(self) -> Vec2 {
        self.major_axis.normalize_or(Vec2::X)
    }
    /// Unit vector along the minor axis (major rotated by +90°).
    #[inline]
    pub fn minor_dir(self) -> Vec2 {
        self.major_dir().perp()
    }
    /// Point at eccentric angle `t`.
    #[inline]
    pub fn point_at(self, t: f32) -> Vec2 {
        let (s, c) = t.sin_cos();
        self.center
            + self.major_dir() * (c * self.semi_major())
            + self.minor_dir() * (s * self.semi_minor())
    }
    /// Parametric derivative (not unit length) — used for tangents.
    #[inline]
    pub fn derivative_at(self, t: f32) -> Vec2 {
        let (s, c) = t.sin_cos();
        self.major_dir() * (-s * self.semi_major()) + self.minor_dir() * (c * self.semi_minor())
    }
    #[inline]
    pub fn tangent_at(self, t: f32) -> Vec2 {
        self.derivative_at(t).normalize_or(Vec2::X)
    }
    #[inline]
    pub fn bounds(self) -> Rect2 {
        let (u, v) = (self.major_dir(), self.minor_dir());
        let (a, b) = (self.semi_major(), self.semi_minor());
        let ext = Vec2::new(
            ((u.x * a) * (u.x * a) + (v.x * b) * (v.x * b)).sqrt(),
            ((u.y * a) * (u.y * a) + (v.y * b) * (v.y * b)).sqrt(),
        );
        Rect2::new(self.center - ext, self.center + ext)
    }

    /// Ramanujan's second approximation for the perimeter of a full ellipse.
    /// Accurate to about 1e-9 relative, which is far below drawing precision.
    pub fn perimeter_ramanujan(self) -> f32 {
        let (a, b) = (self.semi_major(), self.semi_minor());
        if a <= 0.0 || b <= 0.0 {
            return 0.0;
        }
        let sum = a + b;
        let h = ((a - b) * (a - b)) / (sum * sum);
        cad_core::PI * sum * (1.0 + (3.0 * h) / (10.0 + (4.0 - 3.0 * h).sqrt()))
    }

    /// Alias used where "how long is this curve" is meant generically.
    pub fn perimeter_numeric(self) -> f32 {
        self.perimeter_ramanujan()
    }
}

/// Polyline with optional per-vertex bulges (exactly DXF LWPOLYLINE).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Polyline {
    pub vertices: Vec<Vec2>,
    /// `bulges.len() == vertices.len()`; the bulge at `i` describes the segment
    /// from vertex `i` to vertex `i+1` (wrapping when closed).
    pub bulges: Vec<Bulge>,
    pub closed: bool,
}

impl Polyline {
    pub fn new(vertices: Vec<Vec2>, closed: bool) -> Self {
        let n = vertices.len();
        Self {
            bulges: vec![Bulge::NONE; n],
            vertices,
            closed,
        }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.vertices.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
    /// Segment count, accounting for closure.
    #[inline]
    pub fn segment_count(&self) -> usize {
        if self.vertices.len() < 2 {
            return 0;
        }
        if self.closed {
            self.vertices.len()
        } else {
            self.vertices.len() - 1
        }
    }
    #[inline]
    pub fn bulge_at(&self, i: usize) -> Bulge {
        self.bulges.get(i).copied().unwrap_or(Bulge::NONE)
    }
    #[inline]
    pub fn point(&self, i: usize) -> Vec2 {
        self.vertices[i % self.vertices.len()]
    }
    /// The `i`-th segment as a straight line (ignores bulge).
    #[inline]
    pub fn segment(&self, i: usize) -> Line {
        Line::new(self.point(i), self.point(i + 1))
    }
    /// The `i`-th segment as an `Arc`, which is also how a straight run is
    /// represented (`radius = INF` -> draw as a line).
    pub fn segment_arc(&self, i: usize) -> Arc {
        let p0 = self.point(i);
        let p1 = self.point(i + 1);
        let b = self.bulge_at(i);
        let center = b.center(p0, p1);
        let radius = b.radius(p0, p1);
        let a0 = (p0 - center).y.atan2((p0 - center).x);
        Arc {
            center,
            radius,
            start_angle: a0,
            sweep: b.sweep(),
        }
    }
    pub fn bounds(&self) -> Rect2 {
        let mut r = Rect2::ZERO;
        for v in &self.vertices {
            r = r.expand_point(*v);
        }
        for i in 0..self.segment_count() {
            if self.bulge_at(i).is_arc() {
                r = r.union(self.segment_arc(i).bounds());
            }
        }
        r
    }
    /// Shoelace area including the circular-segment correction for bulges.
    pub fn signed_area(&self) -> f32 {
        let n = self.segment_count();
        if n == 0 {
            return 0.0;
        }
        let mut acc = 0.0;
        for i in 0..n {
            let p0 = self.point(i);
            let p1 = self.point(i + 1);
            let b = self.bulge_at(i);
            if b.is_arc() {
                let c = b.center(p0, p1);
                let r = b.radius(p0, p1);
                // Green's theorem: the two triangles from the arc center plus
                // the (signed) circular sector. Exact for any sweep < 2π.
                acc += c.cross(p0) + c.cross(p1) + r * r * b.sweep();
            } else {
                acc += p0.cross(p1);
            }
        }
        acc * 0.5
    }
    pub fn perimeter(&self) -> f32 {
        let mut total = 0.0;
        for i in 0..self.segment_count() {
            let b = self.bulge_at(i);
            let p0 = self.point(i);
            let p1 = self.point(i + 1);
            if b.is_arc() {
                total += b.radius(p0, p1) * b.sweep().abs();
            } else {
                total += p0.distance(p1);
            }
        }
        total
    }
    pub fn reversed(&self) -> Self {
        let n = self.vertices.len();
        let mut vertices = self.vertices.clone();
        vertices.reverse();
        let mut bulges = vec![Bulge::NONE; n];
        if n > 1 {
            for i in 0..n.saturating_sub(1) {
                bulges[i] = Bulge(-self.bulge_at(n - 2 - i).0);
            }
        }
        if self.closed {
            bulges[n - 1] = Bulge(-self.bulge_at(n - 1).0);
        }
        Self {
            vertices,
            bulges,
            closed: self.closed,
        }
    }
}

/// Anything that can be tessellated into a polyline in the XY plane.
#[derive(Debug, Clone, PartialEq)]
pub enum Curve {
    Line(Line),
    Arc(Arc),
    Circle(Circle),
    Ellipse(Ellipse),
    Polyline(Polyline),
}

impl Curve {
    #[inline]
    pub fn start_point(&self) -> Vec2 {
        match self {
            Curve::Line(l) => l.p0,
            Curve::Arc(a) => a.start_point(),
            Curve::Circle(c) => c.center + Vec2::X * c.radius,
            Curve::Ellipse(e) => e.point_at(0.0),
            Curve::Polyline(p) => *p.vertices.first().unwrap_or(&Vec2::ZERO),
        }
    }
    #[inline]
    pub fn end_point(&self) -> Vec2 {
        match self {
            Curve::Line(l) => l.p1,
            Curve::Arc(a) => a.end_point(),
            Curve::Circle(c) => c.center + Vec2::X * c.radius,
            Curve::Ellipse(e) => e.point_at(TAU),
            Curve::Polyline(p) => {
                if p.closed {
                    p.point(0)
                } else {
                    *p.vertices.last().unwrap_or(&Vec2::ZERO)
                }
            }
        }
    }
    #[inline]
    pub fn bounds(&self) -> Rect2 {
        match self {
            Curve::Line(l) => l.bounds(),
            Curve::Arc(a) => a.bounds(),
            Curve::Circle(c) => c.bounds(),
            Curve::Ellipse(e) => e.bounds(),
            Curve::Polyline(p) => p.bounds(),
        }
    }
    #[inline]
    pub fn is_closed(&self) -> bool {
        match self {
            Curve::Circle(_) => true,
            Curve::Polyline(p) => p.closed,
            _ => false,
        }
    }
    #[inline]
    pub fn length(&self) -> f32 {
        match self {
            Curve::Line(l) => l.length(),
            Curve::Arc(a) => a.length(),
            Curve::Circle(c) => c.perimeter(),
            Curve::Ellipse(e) => e.perimeter_ramanujan(),
            Curve::Polyline(p) => p.perimeter(),
        }
    }
    pub fn closest_point(&self, p: Vec2) -> Vec2 {
        match self {
            Curve::Line(l) => l.closest_point(p),
            Curve::Arc(a) => a.closest_point(p),
            Curve::Circle(c) => c.closest_point(p),
            Curve::Ellipse(e) => {
                // Newton on the eccentric anomaly is overkill here; sampling the
                // four quadrants and refining converges in a handful of steps.
                let mut best = e.point_at(0.0);
                let mut best_d = best.distance(p);
                for i in 0..64 {
                    let t = TAU * i as f32 / 64.0;
                    let q = e.point_at(t);
                    let d = q.distance(p);
                    if d < best_d {
                        best_d = d;
                        best = q;
                    }
                }
                best
            }
            Curve::Polyline(pl) => {
                let mut best = pl.point(0);
                let mut best_d = best.distance(p);
                for i in 0..pl.segment_count() {
                    let q = if pl.bulge_at(i).is_arc() {
                        pl.segment_arc(i).closest_point(p)
                    } else {
                        pl.segment(i).closest_point(p)
                    };
                    let d = q.distance(p);
                    if d < best_d {
                        best_d = d;
                        best = q;
                    }
                }
                best
            }
        }
    }
}

/// A closed region: an outer boundary with optional holes, in 2D.
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    pub outer: Polyline,
    pub holes: Vec<Polyline>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::PI;

    #[test]
    fn line_basics() {
        let l = Line::new(Vec2::ZERO, Vec2::new(10.0, 0.0));
        assert_eq!(l.length(), 10.0);
        assert_eq!(l.dir(), Vec2::X);
        assert_eq!(l.closest_point(Vec2::new(-5.0, 3.0)), Vec2::ZERO);
        assert_eq!(l.closest_point(Vec2::new(5.0, 3.0)), Vec2::new(5.0, 0.0));
        assert!((l.distance_to(Vec2::new(5.0, 3.0)) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn line_intersection() {
        let a = Line::new(Vec2::ZERO, Vec2::new(10.0, 0.0));
        let b = Line::new(Vec2::new(5.0, -5.0), Vec2::new(5.0, 5.0));
        let (t, u) = a.intersect(b).unwrap();
        assert!((a.at(t) - Vec2::new(5.0, 0.0)).length() < 1e-5);
        assert!((b.at(u) - Vec2::new(5.0, 0.0)).length() < 1e-5);
        // A parallel offset line never meets it.
        assert!(
            a.intersect(Line::new(Vec2::new(0.0, 5.0), Vec2::new(10.0, 5.0)))
                .is_none()
        );
    }

    #[test]
    fn arc_endpoints_and_angles() {
        let a = Arc::from_angles(Vec2::ZERO, 5.0, 0.0, PI / 2.0);
        assert!(a.start_point().distance(Vec2::new(5.0, 0.0)) < 1e-5);
        assert!(a.end_point().distance(Vec2::new(0.0, 5.0)) < 1e-5);
        assert!((a.length() - 5.0 * PI / 2.0).abs() < 1e-5);
        assert!(a.contains_angle(PI / 4.0));
        assert!(!a.contains_angle(-PI / 4.0));
    }

    #[test]
    fn arc_bounds_cover_only_the_sweep() {
        let a = Arc::from_angles(Vec2::ZERO, 10.0, 0.0, PI / 2.0);
        let b = a.bounds();
        assert!((b.min.x - 0.0).abs() < 1e-4, "{:?}", b.min);
        assert!((b.min.y - 0.0).abs() < 1e-4, "{:?}", b.min);
        assert!((b.max.x - 10.0).abs() < 1e-4, "{:?}", b.max);
        assert!((b.max.y - 10.0).abs() < 1e-4, "{:?}", b.max);

        let full = Circle::new(Vec2::ZERO, 3.0).as_arc();
        let fb = full.bounds();
        assert!((fb.size() - Vec2::splat(6.0)).length() < 1e-4);
    }

    #[test]
    fn circle_area_and_membership() {
        let c = Circle::new(Vec2::ZERO, 2.0);
        assert!((c.area() - 4.0 * PI).abs() < 1e-4);
        assert!(c.contains(Vec2::new(1.0, 1.0)));
        assert!(!c.contains(Vec2::new(3.0, 0.0)));
    }

    #[test]
    fn ellipse_bounds_respect_rotation() {
        let e = Ellipse::new(Vec2::ZERO, Vec2::new(10.0, 0.0), 0.5);
        let b = e.bounds();
        assert!(
            (b.size() - Vec2::new(20.0, 10.0)).length() < 1e-3,
            "{:?}",
            b
        );
        // Semi-major axis of 10 rotated 45 degrees.
        let d = 10.0f32 / 2f32.sqrt();
        let e45 = Ellipse::new(Vec2::ZERO, Vec2::new(d, d), 0.5);
        let b45 = e45.bounds();
        // Rotated 45 deg: extent = sqrt((a*cos45)^2 + (b*sin45)^2) on each axis.
        let expect = 2.0f32 * ((100.0f32) / 2.0 + (25.0f32) / 2.0).sqrt();
        assert!(
            (b45.size().x - expect).abs() < 1e-2,
            "{:?} vs {expect}",
            b45
        );
    }

    #[test]
    fn polyline_perimeter_with_bulge() {
        let mut p = Polyline::new(
            vec![Vec2::ZERO, Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0)],
            true,
        );
        // Make the first edge a quarter circle bulging left (upwards).
        p.bulges[0] = Bulge::from_sweep(PI / 2.0);
        // Segments 1 and 2 stay straight; only segment 0 becomes an arc.
        let straight = 10.0 + Vec2::ZERO.distance(Vec2::new(10.0, 10.0));
        let arc_len = p.bulge_at(0).radius(Vec2::ZERO, Vec2::new(10.0, 0.0)) * PI / 2.0;
        assert!((p.perimeter() - (straight + arc_len)).abs() < 1e-3);
    }

    #[test]
    fn polyline_square_area() {
        let p = Polyline::new(
            vec![
                Vec2::ZERO,
                Vec2::new(10.0, 0.0),
                Vec2::new(10.0, 10.0),
                Vec2::new(0.0, 10.0),
            ],
            true,
        );
        assert!(
            (p.signed_area() - 100.0).abs() < 1e-3,
            "{}",
            p.signed_area()
        );
    }

    #[test]
    fn polyline_half_disk_area() {
        // Semicircle of radius 1 bulging up, closed by the diameter.
        let mut p = Polyline::new(vec![Vec2::ZERO, Vec2::new(2.0, 0.0)], true);
        p.bulges[0] = Bulge::from_sweep(PI);
        p.bulges[1] = Bulge::NONE;
        assert!(
            (p.signed_area() - PI / 2.0).abs() < 1e-4,
            "{}",
            p.signed_area()
        );
    }

    #[test]
    fn polyline_full_circle_area() {
        // A 360 deg bulge between two points degenerates; use 3 arcs instead.
        let r = 2.0f32;
        let mut p = Polyline::new(
            vec![
                Vec2::new(r, 0.0),
                Vec2::new(0.0, r),
                Vec2::new(-r, 0.0),
                Vec2::new(0.0, -r),
            ],
            true,
        );
        for i in 0..4 {
            p.bulges[i] = Bulge::from_sweep(PI / 2.0);
        }
        assert!(
            (p.signed_area() - PI * r * r).abs() < 1e-3,
            "{}",
            p.signed_area()
        );
    }

    #[test]
    fn polyline_reverse_preserves_length() {
        let mut p = Polyline::new(
            vec![Vec2::ZERO, Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0)],
            true,
        );
        p.bulges[0] = Bulge::from_sweep(PI / 3.0);
        p.bulges[2] = Bulge::from_sweep(-PI / 2.0);
        let r = p.reversed();
        assert!(
            (r.perimeter() - p.perimeter()).abs() < 1e-3,
            "{} vs {}",
            r.perimeter(),
            p.perimeter()
        );
        assert!(r.closed);
    }
}
