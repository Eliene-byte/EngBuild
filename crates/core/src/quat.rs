//! Unit quaternion used for orientations and camera arcs.

use crate::mat::Mat3;
use crate::vec::Vec3;
use core::ops::{Mul, Neg};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    #[inline(always)]
    pub fn from_axis_angle(axis: Vec3, radians: f32) -> Self {
        let h = radians * 0.5;
        let s = h.sin();
        let a = axis.normalize();
        Self::new(a.x * s, a.y * s, a.z * s, h.cos())
    }

    /// Intrinsic XYZ (roll, pitch, yaw) in radians.
    #[inline]
    pub fn from_euler(rx: f32, ry: f32, rz: f32) -> Self {
        let (sx, cx) = rx.sin_cos();
        let (sy, cy) = ry.sin_cos();
        let (sz, cz) = rz.sin_cos();
        Self::new(
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        )
    }

    #[inline(always)]
    pub fn from_mat3(m: Mat3) -> Self {
        // Shepperd's method: pick the largest diagonal term for numerical stability.
        let tr = m.c0.x + m.c1.y + m.c2.z;
        if tr > 0.0 {
            let s = (tr + 1.0).sqrt() * 2.0;
            Self::new(
                (m.c1.z - m.c2.y) / s,
                (m.c2.x - m.c0.z) / s,
                (m.c0.y - m.c1.x) / s,
                0.25 * s,
            )
        } else if m.c0.x > m.c1.y && m.c0.x > m.c2.z {
            let s = (1.0 + m.c0.x - m.c1.y - m.c2.z).sqrt() * 2.0;
            Self::new(
                0.25 * s,
                (m.c1.x + m.c0.y) / s,
                (m.c2.x + m.c0.z) / s,
                (m.c1.z - m.c2.y) / s,
            )
        } else if m.c1.y > m.c2.z {
            let s = (1.0 + m.c1.y - m.c0.x - m.c2.z).sqrt() * 2.0;
            Self::new(
                (m.c1.x + m.c0.y) / s,
                0.25 * s,
                (m.c2.y + m.c1.z) / s,
                (m.c2.x - m.c0.z) / s,
            )
        } else {
            let s = (1.0 + m.c2.z - m.c0.x - m.c1.y).sqrt() * 2.0;
            Self::new(
                (m.c2.x + m.c0.z) / s,
                (m.c2.y + m.c1.z) / s,
                0.25 * s,
                (m.c0.y - m.c1.x) / s,
            )
        }
        .normalize()
    }

    #[inline(always)]
    pub const fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w
    }

    #[inline(always)]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l < 1e-9 {
            Self::IDENTITY
        } else {
            let i = 1.0 / l;
            Self::new(self.x * i, self.y * i, self.z * i, self.w * i)
        }
    }

    #[inline(always)]
    pub fn conjugate(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    #[inline(always)]
    pub fn inverse(self) -> Self {
        self.conjugate().normalize()
    }

    #[inline(always)]
    pub fn to_mat3(self) -> Mat3 {
        Mat3::from_quat(self)
    }

    #[inline(always)]
    pub fn rotate_vec3(self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let s = self.w;
        u * (2.0 * u.dot(v)) + v * (s * s - u.dot(u)) + u.cross(v) * (2.0 * s)
    }

    /// Shortest-arc rotation taking `from` to `to` (both normalized).
    #[inline]
    pub fn rotation_between(from: Vec3, to: Vec3) -> Self {
        let a = from.normalize_or(Vec3::X);
        let b = to.normalize_or(Vec3::Y);
        let d = a.dot(b);
        if d > 0.999_999 {
            return Self::IDENTITY;
        }
        if d < -0.999_999 {
            // 180°: any orthogonal axis works.
            let axis = a.any_orthonormal();
            return Self::from_axis_angle(axis, core::f32::consts::PI);
        }
        let c = a.cross(b);
        Self::new(c.x, c.y, c.z, 1.0 + d).normalize()
    }

    #[inline]
    pub fn slerp(self, o: Self, t: f32) -> Self {
        let mut b = o;
        let mut cos = self.dot(b);
        if cos < 0.0 {
            b = -b;
            cos = -cos;
        }
        if cos > 0.999_5 {
            return Self::new(
                self.x + (b.x - self.x) * t,
                self.y + (b.y - self.y) * t,
                self.z + (b.z - self.z) * t,
                self.w + (b.w - self.w) * t,
            )
            .normalize();
        }
        let theta = cos.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let c = (t * theta).sin() / sin_theta;
        Self::new(
            self.x * a + b.x * c,
            self.y * a + b.y * c,
            self.z * a + b.z * c,
            self.w * a + b.w * c,
        )
        .normalize()
    }
}

impl Mul for Quat {
    type Output = Quat;
    #[inline(always)]
    fn mul(self, o: Quat) -> Quat {
        Quat::new(
            self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
            self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
        )
    }
}
impl Neg for Quat {
    type Output = Quat;
    #[inline(always)]
    fn neg(self) -> Quat {
        Quat::new(-self.x, -self.y, -self.z, -self.w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_angle_rotates_x_to_y() {
        let q = Quat::from_axis_angle(Vec3::Z, core::f32::consts::FRAC_PI_2);
        let v = q.rotate_vec3(Vec3::X);
        assert!(v.distance(Vec3::Y) < 1e-5, "{v:?}");
    }

    #[test]
    fn rotation_between_works_for_opposite() {
        let q = Quat::rotation_between(Vec3::X, Vec3::new(-1.0, 0.0, 0.0));
        let v = q.rotate_vec3(Vec3::X);
        assert!(v.distance(Vec3::new(-1.0, 0.0, 0.0)) < 1e-5, "{v:?}");
    }

    #[test]
    fn mat3_round_trip() {
        let q = Quat::from_euler(0.4, 1.1, -0.7);
        let q2 = Quat::from_mat3(q.to_mat3());
        assert!(q.dot(q2).abs() > 0.9999);
    }

    #[test]
    fn slerp_hits_endpoints() {
        let a = Quat::from_axis_angle(Vec3::Y, 0.0);
        let b = Quat::from_axis_angle(Vec3::Y, 1.0);
        assert!(a.slerp(b, 0.0).dot(a).abs() > 0.999);
        assert!(a.slerp(b, 1.0).dot(b).abs() > 0.999);
    }
}
