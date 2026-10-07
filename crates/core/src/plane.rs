//! Normalised 3D plane in Hessian form `dot(n, p) = d`.

use crate::vec::Vec3;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Plane3 {
    pub n: Vec3,
    pub d: f32,
}

impl Plane3 {
    #[inline(always)]
    pub const fn new(n: Vec3, d: f32) -> Self {
        Self { n, d }
    }

    #[inline(always)]
    pub const fn from_point_normal(p: Vec3, n: Vec3) -> Self {
        Self { n, d: n.dot(p) }
    }

    #[inline]
    pub fn normalized(self) -> Self {
        let l = self.n.length();
        if l < 1e-9 {
            return Self { n: Vec3::Z, d: 0.0 };
        }
        let i = 1.0 / l;
        Self {
            n: self.n * i,
            d: self.d * i,
        }
    }

    #[inline(always)]
    pub const fn signed_distance(self, p: Vec3) -> f32 {
        self.n.dot(p) - self.d
    }

    #[inline(always)]
    pub fn project(self, p: Vec3) -> Vec3 {
        p - self.n * self.signed_distance(p)
    }

    /// Intersection with a segment, returning the parameter in `0..=1`.
    #[inline]
    pub fn intersect_segment(self, a: Vec3, b: Vec3) -> Option<f32> {
        let da = self.signed_distance(a);
        let db = self.signed_distance(b);
        let denom = da - db;
        if denom.abs() < 1e-12 {
            return None;
        }
        let t = da / denom;
        if (0.0..=1.0).contains(&t) {
            Some(t)
        } else {
            None
        }
    }

    /// Intersection of three planes via Cramer's rule.
    /// Solves `na·p = da`, `nb·p = db`, `nc·p = dc`.
    pub fn intersect3(a: Self, b: Self, c: Self) -> Option<Vec3> {
        let det = a.n.dot(b.n.cross(c.n));
        if det.abs() < 1e-12 {
            return None;
        }
        let inv = 1.0 / det;
        Some((b.n.cross(c.n) * a.d + c.n.cross(a.n) * b.d + a.n.cross(b.n) * c.d) * inv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_projection_and_hit() {
        let pl = Plane3::from_point_normal(Vec3::ZERO, Vec3::Z);
        assert!((pl.signed_distance(Vec3::new(5.0, 5.0, 3.0)) - 3.0).abs() < 1e-6);
        let t = pl.intersect_segment(Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0));
        assert!((t.unwrap() - 0.5).abs() < 1e-6);
        assert!(
            pl.intersect_segment(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 2.0))
                .is_none()
        );
    }

    #[test]
    fn three_plane_intersection() {
        let x = Plane3::new(Vec3::X, 1.0);
        let y = Plane3::new(Vec3::Y, 2.0);
        let z = Plane3::new(Vec3::Z, 3.0);
        let p = Plane3::intersect3(x, y, z).unwrap();
        assert!(p.distance(Vec3::new(1.0, 2.0, 3.0)) < 1e-5);
    }
}
