//! Affine (and similarity) transforms applied to 2D curves, with `Arc` radius
//! correction so that scaling a drawing never silently distorts arc geometry.

use crate::curve::{Arc, Circle, Curve, Ellipse, Line, Polyline};
use crate::spline::Bezier;
use cad_core::{Mat3, Rect2, Vec2};

/// A 2D affine transform stored as a 3x3 matrix (last row is `0 0 1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine2 {
    pub m: Mat3,
}

impl Default for Affine2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Affine2 {
    pub const IDENTITY: Self = Self { m: Mat3::IDENTITY };

    #[inline]
    pub fn new(m: Mat3) -> Self {
        Self { m }
    }
    #[inline]
    pub fn translation(t: Vec2) -> Self {
        Self {
            m: Mat3::from_cols(v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0), v3(t.x, t.y, 1.0)),
        }
    }
    #[inline]
    pub fn scaling(s: Vec2) -> Self {
        Self {
            m: Mat3::from_cols(v3(s.x, 0.0, 0.0), v3(0.0, s.y, 0.0), v3(0.0, 0.0, 1.0)),
        }
    }
    #[inline]
    pub fn rotation(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m: Mat3::from_cols(v3(c, s, 0.0), v3(-s, c, 0.0), v3(0.0, 0.0, 1.0)),
        }
    }
    #[inline]
    pub fn skew(k: Vec2) -> Self {
        Self {
            m: Mat3::from_cols(v3(1.0, k.y, 0.0), v3(k.x, 1.0, 0.0), v3(0.0, 0.0, 1.0)),
        }
    }
    #[inline]
    pub fn compose(self, o: Affine2) -> Self {
        Self { m: self.m * o.m }
    }
    #[inline]
    pub fn then(self, o: Affine2) -> Self {
        o.compose(self)
    }
    #[inline]
    pub fn apply(&self, p: Vec2) -> Vec2 {
        let v = self.m.mul_vec3(cad_core::Vec3::new(p.x, p.y, 1.0));
        Vec2::new(v.x, v.y)
    }
    #[inline]
    pub fn apply_dir(&self, d: Vec2) -> Vec2 {
        let v = self.m.mul_vec3(cad_core::Vec3::new(d.x, d.y, 0.0));
        Vec2::new(v.x, v.y)
    }
    #[inline]
    pub fn inverse(&self) -> Self {
        Self {
            m: self.m.inverse(),
        }
    }
    #[inline]
    pub fn determinant(&self) -> f32 {
        self.m.determinant()
    }
    /// Average linear scale factor — used to correct arc radii.
    #[inline]
    pub fn scale_factor(&self) -> f32 {
        self.determinant().abs().sqrt()
    }
    /// X axis image, as a vector.
    #[inline]
    pub fn axis_x(&self) -> Vec2 {
        self.apply_dir(Vec2::X)
    }
    /// Y axis image, as a vector.
    #[inline]
    pub fn axis_y(&self) -> Vec2 {
        self.apply_dir(Vec2::Y)
    }
    /// Rotation angle implied by the transform (assumes uniform scale).
    #[inline]
    pub fn rotation_angle(&self) -> f32 {
        self.axis_x().y.atan2(self.axis_x().x)
    }
    /// Transform a rectangle by taking its 4 transformed corners.
    pub fn apply_rect(&self, r: Rect2) -> Rect2 {
        let mut out = Rect2::new(self.apply(r.min), self.apply(r.min));
        for c in r.corners() {
            out = out.expand_point(self.apply(c));
        }
        out
    }
    /// Mirror across an arbitrary line through the origin.
    pub fn mirror(dir: Vec2) -> Self {
        let d = dir.normalize_or(Vec2::X);
        let (nx, ny) = (d.x, d.y);
        Self {
            m: Mat3::from_cols(
                v3(nx * nx - ny * ny, 2.0 * nx * ny, 0.0),
                v3(2.0 * nx * ny, ny * ny - nx * nx, 0.0),
                v3(0.0, 0.0, 1.0),
            ),
        }
    }
    /// Build from the images of the X and Y axes plus the origin.
    pub fn from_basis(origin: Vec2, x: Vec2, y: Vec2) -> Self {
        Self {
            m: Mat3::from_cols(
                v3(x.x, x.y, 0.0),
                v3(y.x, y.y, 0.0),
                v3(origin.x, origin.y, 1.0),
            ),
        }
    }
    /// True when the transform preserves angles up to a uniform scale + mirror,
    /// i.e. circles stay circles. Arcs/ellipses are only promoted to polylines
    /// when this is false.
    #[inline]
    pub fn is_similarity(&self) -> bool {
        let x = self.axis_x();
        let y = self.axis_y();
        if x.length_squared() < 1e-18 || y.length_squared() < 1e-18 {
            return false;
        }
        (x.length() - y.length()).abs() <= 1e-4 * x.length().max(y.length())
            && x.dot(y).abs() <= 1e-4 * x.length() * y.length()
    }
    /// True when the similarity also flips handedness (mirror).
    #[inline]
    pub fn is_mirroring(&self) -> bool {
        self.determinant() < 0.0
    }
}

#[inline]
fn v3(x: f32, y: f32, z: f32) -> cad_core::Vec3 {
    cad_core::Vec3::new(x, y, z)
}

impl Curve {
    /// Transform a curve. Arcs/circles only stay circular under **similarity**
    /// transforms; under a general affine we promote them to an elliptical
    /// Bézier approximation, which is what AutoCAD/FreeCAD do as well.
    pub fn transformed(&self, t: &Affine2) -> Curve {
        match self {
            Curve::Line(l) => Curve::Line(Line::new(t.apply(l.p0), t.apply(l.p1))),
            Curve::Polyline(p) => {
                let mut out = p.clone();
                out.vertices = p.vertices.iter().map(|v| t.apply(*v)).collect();
                // Bulges are angles: they only survive similarity transforms.
                if t.is_similarity() {
                    if t.is_mirroring() {
                        for b in out.bulges.iter_mut() {
                            *b = crate::bulge::Bulge(-b.0);
                        }
                    }
                } else {
                    out.bulges = vec![crate::bulge::Bulge::NONE; out.vertices.len()];
                }
                Curve::Polyline(out)
            }
            Curve::Circle(c) => {
                if t.is_similarity() {
                    // Still a circle: radius scales uniformly.
                    Curve::Circle(Circle::new(t.apply(c.center), c.radius * t.scale_factor()))
                } else {
                    // Affine image of a circle is an ellipse; approximate it.
                    Curve::Polyline(ellipse_from_affine(c, t))
                }
            }
            Curve::Arc(a) => {
                if t.is_similarity() {
                    let flip = t.is_mirroring();
                    Curve::Arc(Arc {
                        center: t.apply(a.center),
                        radius: a.radius * t.scale_factor(),
                        start_angle: t.rotation_angle() + a.start_angle,
                        sweep: if flip { -a.sweep } else { a.sweep },
                    })
                } else {
                    Curve::Polyline(arc_from_affine(a, t))
                }
            }
            Curve::Ellipse(e) => {
                let center = t.apply(e.center);
                let major = t.apply_dir(e.major_axis);
                // Under non-uniform scale the minor axis scales by a different
                // factor; recover the ratio from the Y axis image.
                let y = t.apply_dir(Vec2::Y);
                let ratio = {
                    let a_img = t.apply_dir(Vec2::X).length();
                    let b_img = y.length();
                    if a_img < 1e-12 {
                        1.0
                    } else {
                        (b_img / a_img) * e.ratio
                    }
                };
                Curve::Ellipse(Ellipse::new(center, major, ratio.clamp(1e-4, 1.0)))
            }
        }
    }

    /// Convenience: translate.
    pub fn translated(&self, v: Vec2) -> Curve {
        self.transformed(&Affine2::translation(v))
    }
    /// Convenience: mirror.
    pub fn mirrored(&self, axis: Vec2) -> Curve {
        self.transformed(&Affine2::mirror(axis))
    }
    pub fn rotated_about(&self, center: Vec2, angle: f32) -> Curve {
        let t = Affine2::translation(-center)
            .then(Affine2::rotation(angle))
            .then(Affine2::translation(center));
        self.transformed(&t)
    }
}

/// Affine image of a circle, as a closed polyline of Bézier-fitted points.
fn ellipse_from_affine(c: &Circle, t: &Affine2) -> Polyline {
    let center = t.apply(c.center);
    let ux = t.apply_dir(Vec2::X) * c.radius;
    let uy = t.apply_dir(Vec2::Y) * c.radius;
    // Sample the parametric ellipse densely enough for a fill/edge.
    let n = 64;
    let mut vertices = Vec::with_capacity(n);
    for i in 0..n {
        let a = cad_core::TAU * i as f32 / n as f32;
        let (s, cs) = a.sin_cos();
        vertices.push(center + ux * cs + uy * s);
    }
    Polyline::new(vertices, true)
}

fn arc_from_affine(a: &Arc, t: &Affine2) -> Polyline {
    let mut pts = Vec::new();
    let n = 48;
    let start = t.apply(a.start_point());
    for i in 0..=n {
        let tt = i as f32 / n as f32;
        pts.push(t.apply(a.point_at(tt)));
    }
    let _ = start;
    Polyline::new(pts, false)
}

/// Transform a Bézier — control points transform linearly, so this is exact.
pub fn transform_bezier(b: &Bezier, t: &Affine2) -> Bezier {
    Bezier::new(
        t.apply(b.p[0]),
        t.apply(b.p[1]),
        t.apply(b.p[2]),
        t.apply(b.p[3]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::Arc;
    use cad_core::Vec2;

    #[test]
    fn translation_and_rotation() {
        let t = Affine2::translation(Vec2::new(10.0, 0.0));
        assert_eq!(t.apply(Vec2::ZERO), Vec2::new(10.0, 0.0));

        let r = Affine2::rotation(cad_core::FRAC_PI_2);
        let p = r.apply(Vec2::X);
        assert!(p.distance(Vec2::Y) < 1e-5, "{p:?}");
    }

    #[test]
    fn compose_then_invert_is_identity() {
        let t = Affine2::translation(Vec2::new(3.0, -2.0))
            .then(Affine2::rotation(0.7))
            .then(Affine2::scaling(Vec2::new(2.0, 0.5)));
        let inv = t.inverse();
        let p = Vec2::new(12.0, -5.0);
        assert!(inv.apply(t.apply(p)).distance(p) < 1e-4);
    }

    #[test]
    fn rotate_line_about_point() {
        let l = Curve::Line(Line::new(Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0)));
        let r = l.rotated_about(Vec2::ZERO, cad_core::FRAC_PI_2);
        match r {
            Curve::Line(m) => {
                assert!((m.p0.y - 1.0).abs() < 1e-5, "{:?}", m.p0);
                assert!((m.p1.y - 2.0).abs() < 1e-5, "{:?}", m.p1);
            }
            _ => panic!("expected a line"),
        }
    }

    #[test]
    fn scale_circle_scales_radius() {
        let c = Curve::Circle(Circle::new(Vec2::ZERO, 2.0));
        let s = c.transformed(&Affine2::scaling(Vec2::splat(3.0)));
        match s {
            Curve::Circle(out) => assert!((out.radius - 6.0).abs() < 1e-4),
            other => panic!("expected a circle, got {other:?}"),
        }
    }

    #[test]
    fn mirror_circle_flips_arc_direction() {
        let a = Curve::Arc(Arc::from_angles(Vec2::ZERO, 5.0, 0.0, cad_core::FRAC_PI_2));
        let m = a.mirrored(Vec2::X);
        match m {
            Curve::Arc(out) => {
                assert!(out.sweep < 0.0, "sweep should flip: {:?}", out.sweep);
                assert!(out.contains_angle(-cad_core::FRAC_PI_2 / 2.0));
            }
            other => panic!("expected an arc, got {other:?}"),
        }
    }

    #[test]
    fn non_uniform_scale_circle_becomes_polyline() {
        let c = Curve::Circle(Circle::new(Vec2::ZERO, 1.0));
        let s = c.transformed(&Affine2::scaling(Vec2::new(4.0, 1.0)));
        match s {
            Curve::Polyline(p) => {
                assert!(p.closed);
                let b = p.bounds();
                assert!((b.size().x - 8.0).abs() < 0.1, "{:?}", b.size());
                assert!((b.size().y - 2.0).abs() < 0.1, "{:?}", b.size());
            }
            other => panic!("expected a polyline, got {other:?}"),
        }
    }

    #[test]
    fn polyline_arc_bulges_drop_under_anisotropic_scale() {
        let mut p = Polyline::new(vec![Vec2::ZERO, Vec2::new(4.0, 0.0)], false);
        p.bulges[0] = crate::bulge::Bulge::from_sweep(cad_core::FRAC_PI_2);
        let c = Curve::Polyline(p);
        let s = c.transformed(&Affine2::scaling(Vec2::new(2.0, 1.0)));
        match s {
            Curve::Polyline(out) => {
                assert!(out.bulges.iter().all(|b| !b.is_arc()));
                assert_eq!(out.vertices[1].x, 8.0);
            }
            _ => panic!("expected a polyline"),
        }
    }

    #[test]
    fn rotation_preserves_bulges() {
        let mut p = Polyline::new(vec![Vec2::ZERO, Vec2::new(4.0, 0.0)], false);
        p.bulges[0] = crate::bulge::Bulge::from_sweep(cad_core::FRAC_PI_2);
        let c = Curve::Polyline(p).rotated_about(Vec2::ZERO, cad_core::FRAC_PI_2);
        match c {
            Curve::Polyline(out) => {
                assert!((out.bulges[0].0 - 0.414_213_6).abs() < 1e-4);
                assert!((out.vertices[1].y - 4.0).abs() < 1e-4);
            }
            _ => panic!("expected a polyline"),
        }
    }

    #[test]
    fn mirror_polyline_area_is_preserved_in_magnitude() {
        let p = Polyline::new(
            vec![
                Vec2::ZERO,
                Vec2::new(10.0, 0.0),
                Vec2::new(10.0, 4.0),
                Vec2::new(0.0, 4.0),
            ],
            true,
        );
        let a0 = p.signed_area();
        let m = Curve::Polyline(p).mirrored(Vec2::X);
        match m {
            Curve::Polyline(out) => {
                assert!((out.signed_area().abs() - a0).abs() < 1e-3);
            }
            _ => panic!("expected a polyline"),
        }
    }

    #[test]
    fn rect_transform_covers_corners() {
        let t = Affine2::rotation(0.785398);
        let r = Rect2::from_xywh(0.0, 0.0, 1.0, 1.0);
        let o = t.apply_rect(r);
        let expect = 2f32.sqrt();
        assert!((o.size().x - expect).abs() < 1e-4, "{:?}", o.size());
    }
}
