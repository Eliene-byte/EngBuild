//! Column-major 3x3 / 4x4 matrices laid out exactly like WGSL `mat3x4<f32>`.

use crate::float::{FRAC_PI_2, PI, wrap_pi};
use crate::quat::Quat;
use crate::vec::{Vec3, Vec4};
use core::ops::{Add, Mul, Sub};

// ---------------------------------------------------------------- Mat3

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat3 {
    /// Column-major: `c0 = [m00, m10, m20]`.
    pub c0: Vec3,
    pub c1: Vec3,
    pub c2: Vec3,
}

impl Default for Mat3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mat3 {
    pub const IDENTITY: Self = Self {
        c0: Vec3::X,
        c1: Vec3::Y,
        c2: Vec3::Z,
    };

    #[inline(always)]
    pub const fn from_cols(c0: Vec3, c1: Vec3, c2: Vec3) -> Self {
        Self { c0, c1, c2 }
    }
    #[inline(always)]
    pub const fn to_cols_array_3d(self) -> [[f32; 3]; 3] {
        [self.c0.to_array(), self.c1.to_array(), self.c2.to_array()]
    }
    #[inline(always)]
    pub const fn col(&self, i: usize) -> Vec3 {
        match i {
            0 => self.c0,
            1 => self.c1,
            _ => self.c2,
        }
    }
    #[inline(always)]
    pub fn row(&self, i: usize) -> Vec3 {
        Vec3::new(
            self.c0.to_array()[i],
            self.c1.to_array()[i],
            self.c2.to_array()[i],
        )
    }

    #[inline(always)]
    pub fn from_scale(s: Vec3) -> Self {
        Self::from_cols(
            Vec3::new(s.x, 0.0, 0.0),
            Vec3::new(0.0, s.y, 0.0),
            Vec3::new(0.0, 0.0, s.z),
        )
    }
    #[inline(always)]
    pub fn from_quat(q: Quat) -> Self {
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx, xy, xz) = (x * x2, x * y2, x * z2);
        let (yy, yz, zz) = (y * y2, y * z2, z * z2);
        let (wx, wy, wz) = (w * x2, w * y2, w * z2);
        Self::from_cols(
            Vec3::new(1.0 - (yy + zz), xy + wz, xz - wy),
            Vec3::new(xy - wz, 1.0 - (xx + zz), yz + wx),
            Vec3::new(xz + wy, yz - wx, 1.0 - (xx + yy)),
        )
    }

    /// Columns are the transformed basis vectors of the basis matrix.
    #[inline(always)]
    pub fn transpose(self) -> Self {
        Self::from_cols(
            Vec3::new(self.c0.x, self.c1.x, self.c2.x),
            Vec3::new(self.c0.y, self.c1.y, self.c2.y),
            Vec3::new(self.c0.z, self.c1.z, self.c2.z),
        )
    }

    /// Transform a **direction** (ignores translation, assumes no projective part).
    #[inline(always)]
    pub fn mul_vec3(self, v: Vec3) -> Vec3 {
        self.c0 * v.x + self.c1 * v.y + self.c2 * v.z
    }

    #[inline(always)]
    pub fn determinant(self) -> f32 {
        self.c0.dot(self.c1.cross(self.c2))
    }

    /// General inverse. Falls back to identity for singular input.
    #[inline]
    pub fn inverse(self) -> Self {
        let a = self.c1.cross(self.c2);
        let b = self.c2.cross(self.c0);
        let c = self.c0.cross(self.c1);
        let det = self.c0.dot(a);
        if det.abs() < 1e-12 {
            return Self::IDENTITY;
        }
        let inv = 1.0 / det;
        Self::from_cols(
            Vec3::new(a.x, b.x, c.x) * inv,
            Vec3::new(a.y, b.y, c.y) * inv,
            Vec3::new(a.z, b.z, c.z) * inv,
        )
    }
}

impl Mul for Mat3 {
    type Output = Mat3;
    #[inline(always)]
    fn mul(self, o: Mat3) -> Mat3 {
        Mat3::from_cols(
            self.mul_vec3(o.c0),
            self.mul_vec3(o.c1),
            self.mul_vec3(o.c2),
        )
    }
}

// ---------------------------------------------------------------- Mat4

/// Column-major 4x4. Matches WGSL's `mat4x4<f32>` memory layout byte-for-byte,
/// so it can be written straight into a uniform buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat4 {
    pub c0: Vec4,
    pub c1: Vec4,
    pub c2: Vec4,
    pub c3: Vec4,
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        c0: Vec4::new(1.0, 0.0, 0.0, 0.0),
        c1: Vec4::new(0.0, 1.0, 0.0, 0.0),
        c2: Vec4::new(0.0, 0.0, 1.0, 0.0),
        c3: Vec4::new(0.0, 0.0, 0.0, 1.0),
    };

    #[inline(always)]
    pub const fn from_cols(c0: Vec4, c1: Vec4, c2: Vec4, c3: Vec4) -> Self {
        Self { c0, c1, c2, c3 }
    }
    #[inline(always)]
    pub const fn to_cols_array(self) -> [Vec4; 4] {
        [self.c0, self.c1, self.c2, self.c3]
    }
    /// Flat column-major `[f32; 16]` — the exact bytes WGSL expects.
    #[inline(always)]
    pub fn to_cols_array_2d(self) -> [[f32; 4]; 4] {
        [
            self.c0.to_array(),
            self.c1.to_array(),
            self.c2.to_array(),
            self.c3.to_array(),
        ]
    }
    #[inline(always)]
    pub fn col(&self, i: usize) -> Vec4 {
        match i {
            0 => self.c0,
            1 => self.c1,
            2 => self.c2,
            _ => self.c3,
        }
    }

    #[inline(always)]
    pub fn translation(t: Vec3) -> Self {
        let mut m = Self::IDENTITY;
        m.c3 = Vec4::new(t.x, t.y, t.z, 1.0);
        m
    }
    #[inline(always)]
    pub fn scale(s: Vec3) -> Self {
        Self::from_cols(
            Vec4::new(s.x, 0.0, 0.0, 0.0),
            Vec4::new(0.0, s.y, 0.0, 0.0),
            Vec4::new(0.0, 0.0, s.z, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }
    #[inline(always)]
    pub fn from_quat(q: Quat) -> Self {
        let m = Mat3::from_quat(q);
        Self::from_cols(
            Vec4::new(m.c0.x, m.c0.y, m.c0.z, 0.0),
            Vec4::new(m.c1.x, m.c1.y, m.c1.z, 0.0),
            Vec4::new(m.c2.x, m.c2.y, m.c2.z, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }
    #[inline(always)]
    pub fn from_scale_rotation_translation(s: Vec3, r: Quat, t: Vec3) -> Self {
        let m = Mat3::from_quat(r);
        Self::from_cols(
            Vec4::new(m.c0.x * s.x, m.c0.y * s.x, m.c0.z * s.x, 0.0),
            Vec4::new(m.c1.x * s.y, m.c1.y * s.y, m.c1.z * s.y, 0.0),
            Vec4::new(m.c2.x * s.z, m.c2.y * s.z, m.c2.z * s.z, 0.0),
            Vec4::new(t.x, t.y, t.z, 1.0),
        )
    }
    /// Non-uniform-safe decomposition used by the 3D gizmo.
    #[inline]
    pub fn decompose(self) -> (Vec3, Quat, Vec3) {
        let m = Mat3::from_cols(
            Vec3::new(self.c0.x, self.c0.y, self.c0.z),
            Vec3::new(self.c1.x, self.c1.y, self.c1.z),
            Vec3::new(self.c2.x, self.c2.y, self.c2.z),
        );
        let scale = Vec3::new(m.c0.length(), m.c1.length(), m.c2.length());
        let r = if scale.x.abs() < 1e-9 || scale.y.abs() < 1e-9 || scale.z.abs() < 1e-9 {
            Quat::IDENTITY
        } else {
            Quat::from_mat3(Mat3::from_cols(
                m.c0 / scale.x,
                m.c1 / scale.y,
                m.c2 / scale.z,
            ))
        };
        let t = self.c3.xyz();
        (scale, r, t)
    }

    /// Transform a point (w = 1).
    #[inline(always)]
    pub fn mul_point(&self, p: Vec3) -> Vec3 {
        let x = self.c0.x * p.x + self.c1.x * p.y + self.c2.x * p.z + self.c3.x;
        let y = self.c0.y * p.x + self.c1.y * p.y + self.c2.y * p.z + self.c3.y;
        let z = self.c0.z * p.x + self.c1.z * p.y + self.c2.z * p.z + self.c3.z;
        let w = self.c0.w * p.x + self.c1.w * p.y + self.c2.w * p.z + self.c3.w;
        if (w - 1.0).abs() < 1e-9 || w.abs() < 1e-12 {
            Vec3::new(x, y, z)
        } else {
            Vec3::new(x / w, y / w, z / w)
        }
    }
    /// Transform a direction (w = 0).
    #[inline(always)]
    pub fn mul_dir(&self, p: Vec3) -> Vec3 {
        Vec3::new(
            self.c0.x * p.x + self.c1.x * p.y + self.c2.x * p.z,
            self.c0.y * p.x + self.c1.y * p.y + self.c2.y * p.z,
            self.c0.z * p.x + self.c1.z * p.y + self.c2.z * p.z,
        )
    }
    /// Project a point and return homogeneous coordinates.
    #[inline(always)]
    pub fn mul_vec4(&self, v: Vec4) -> Vec4 {
        self.c0 * v.x + self.c1 * v.y + self.c2 * v.z + self.c3 * v.w
    }

    #[inline(always)]
    pub fn transpose(self) -> Self {
        Self::from_cols(
            Vec4::new(self.c0.x, self.c1.x, self.c2.x, self.c3.x),
            Vec4::new(self.c0.y, self.c1.y, self.c2.y, self.c3.y),
            Vec4::new(self.c0.z, self.c1.z, self.c2.z, self.c3.z),
            Vec4::new(self.c0.w, self.c1.w, self.c2.w, self.c3.w),
        )
    }

    /// Upper-left 3x3, inverse-transposed — the correct normal matrix.
    #[inline]
    pub fn normal_matrix(self) -> Mat3 {
        Mat3::from_cols(
            Vec3::new(self.c0.x, self.c0.y, self.c0.z),
            Vec3::new(self.c1.x, self.c1.y, self.c1.z),
            Vec3::new(self.c2.x, self.c2.y, self.c2.z),
        )
        .inverse()
        .transpose()
    }

    /// Full 4x4 inverse via cofactor expansion. Falls back to identity if singular.
    #[inline]
    pub fn inverse(self) -> Self {
        let det = det4(self);
        if det.abs() < 1e-12 {
            return Self::IDENTITY;
        }
        let id = 1.0 / det;
        let m = |r: usize, c: usize| self.col(c).to_array()[r];
        // A^-1 = adj(A)/det, adj = C^T, so inv[r][c] = C[c][r] / det
        let mut inv = [[0.0f32; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                let mut sub = [[0.0f32; 3]; 3];
                let mut i = 0;
                for rr in 0..4 {
                    if rr == c {
                        continue;
                    }
                    let mut j = 0;
                    for cc in 0..4 {
                        if cc == r {
                            continue;
                        }
                        sub[i][j] = m(rr, cc);
                        j += 1;
                    }
                    i += 1;
                }
                let cof = det3(sub);
                inv[r][c] = if (r + c) % 2 == 0 {
                    cof * id
                } else {
                    -cof * id
                };
            }
        }
        Self::from_cols(
            Vec4::new(inv[0][0], inv[1][0], inv[2][0], inv[3][0]),
            Vec4::new(inv[0][1], inv[1][1], inv[2][1], inv[3][1]),
            Vec4::new(inv[0][2], inv[1][2], inv[2][2], inv[3][2]),
            Vec4::new(inv[0][3], inv[1][3], inv[2][3], inv[3][3]),
        )
    }

    /// Right-handed look-at view matrix (camera looks down `-Z`).
    #[inline]
    pub fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = (center - eye).normalize_or(Vec3::splat(-1.0));
        let mut s = f.cross(up).normalize_or(Vec3::X);
        if s.length_squared() < 1e-12 {
            s = f.cross(Vec3::Y).normalize_or(Vec3::X);
        }
        let u = s.cross(f);
        Self::from_cols(
            Vec4::new(s.x, u.x, -f.x, 0.0),
            Vec4::new(s.y, u.y, -f.y, 0.0),
            Vec4::new(s.z, u.z, -f.z, 0.0),
            Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
        )
    }

    /// Right-handed orthographic projection with **0..1** depth (WebGPU/wgpu NDC).
    #[inline]
    pub fn orthographic_rh_gl(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let rl = right - left;
        let tb = top - bottom;
        let nf = near - far;
        Self::from_cols(
            Vec4::new(2.0 / rl, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 / tb, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0 / nf, 0.0),
            Vec4::new(-(right + left) / rl, -(top + bottom) / tb, near / nf, 1.0),
        )
    }

    /// Right-handed perspective with **0..1** depth, `fov_y` in radians.
    #[inline]
    pub fn perspective_rh_gl(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        let nf = near - far;
        Self::from_cols(
            Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            Vec4::new(0.0, f, 0.0, 0.0),
            Vec4::new(0.0, 0.0, far / nf, -1.0),
            Vec4::new(0.0, 0.0, near * far / nf, 0.0),
        )
    }

    /// Infinite-far perspective — best for CAD when the extents are unknown.
    #[inline]
    pub fn perspective_rh_infinite(fov_y: f32, aspect: f32, near: f32) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        Self::from_cols(
            Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            Vec4::new(0.0, f, 0.0, 0.0),
            Vec4::new(0.0, 0.0, -1.0, -1.0),
            Vec4::new(0.0, 0.0, -near, 0.0),
        )
    }

    /// Rotation about the view/pan axes used by 2D CAD navigation.
    #[inline]
    pub fn rotate_z(r: f32) -> Self {
        let (s, c) = r.sin_cos();
        Self::from_cols(
            Vec4::new(c, s, 0.0, 0.0),
            Vec4::new(-s, c, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }

    /// 2D affine transform (rotation, uniform-ish scale, translation) as a Mat4.
    #[inline]
    pub fn from_affine_2d(rot: f32, scale: f32, translate: crate::vec::Vec2) -> Self {
        let (s, c) = rot.sin_cos();
        Self::from_cols(
            Vec4::new(c * scale, s * scale, 0.0, 0.0),
            Vec4::new(-s * scale, c * scale, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(translate.x, translate.y, 0.0, 1.0),
        )
    }
}

#[inline]
fn det3(m: [[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Determinant via Laplace expansion along the first row - hard to get wrong.
#[inline]
fn det4(m: Mat4) -> f32 {
    let at = |r: usize, c: usize| m.col(c).to_array()[r];
    let mut det = 0.0;
    for c in 0..4 {
        let mut sub = [[0.0f32; 3]; 3];
        let mut i = 0;
        for cc in 0..4 {
            if cc == c {
                continue;
            }
            let mut j = 0;
            for r in 1..4 {
                sub[i][j] = at(r, cc);
                j += 1;
            }
            i += 1;
        }
        let sign = if c % 2 == 0 { 1.0 } else { -1.0 };
        det += sign * at(0, c) * det3(sub);
    }
    det
}

impl Mul for Mat4 {
    type Output = Mat4;
    #[inline(always)]
    fn mul(self, o: Mat4) -> Mat4 {
        Mat4::from_cols(
            self.mul_vec4(o.c0),
            self.mul_vec4(o.c1),
            self.mul_vec4(o.c2),
            self.mul_vec4(o.c3),
        )
    }
}
impl Add for Mat4 {
    type Output = Mat4;
    #[inline(always)]
    fn add(self, o: Mat4) -> Mat4 {
        Mat4::from_cols(
            self.c0 + o.c0,
            self.c1 + o.c1,
            self.c2 + o.c2,
            self.c3 + o.c3,
        )
    }
}
impl Sub for Mat4 {
    type Output = Mat4;
    #[inline(always)]
    fn sub(self, o: Mat4) -> Mat4 {
        Mat4::from_cols(
            self.c0 - o.c0,
            self.c1 - o.c1,
            self.c2 - o.c2,
            self.c3 - o.c3,
        )
    }
}
impl Mul<f32> for Mat4 {
    type Output = Mat4;
    #[inline(always)]
    fn mul(self, s: f32) -> Mat4 {
        Mat4::from_cols(self.c0 * s, self.c1 * s, self.c2 * s, self.c3 * s)
    }
}

/// Euler XYZ rotation helper (degrees -> radians) used by the UI.
#[inline]
pub fn quat_from_euler_deg(x: f32, y: f32, z: f32) -> Quat {
    Quat::from_euler(
        x * core::f32::consts::PI / 180.0,
        y * PI / 180.0,
        z * PI / 180.0,
    )
}

#[inline]
pub fn wrap_angle(a: f32) -> f32 {
    wrap_pi(a)
}

#[inline]
pub fn half_turn() -> f32 {
    FRAC_PI_2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Mat4, b: Mat4, eps: f32) -> bool {
        for i in 0..4 {
            let d = (a.col(i) - b.col(i)).xyz().length();
            if d > eps {
                return false;
            }
        }
        true
    }

    #[test]
    fn identity_mul_is_identity() {
        let m = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 3.0, 4.0),
            Quat::from_axis_angle(Vec3::Z, 0.7),
            Vec3::new(5.0, 6.0, 7.0),
        );
        assert!(approx(m * Mat4::IDENTITY, m, 1e-5));
        assert!(approx(Mat4::IDENTITY * m, m, 1e-5));
    }

    #[test]
    fn inverse_round_trips() {
        let m = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 0.5, 3.0),
            Quat::from_euler(0.3, -1.1, 0.9),
            Vec3::new(-4.0, 2.0, 8.0),
        );
        assert!(approx(m * m.inverse(), Mat4::IDENTITY, 1e-4));
    }

    #[test]
    fn look_at_places_target_on_negative_z() {
        let v = Mat4::look_at_rh(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0), Vec3::Y);
        let p = v.mul_point(Vec3::new(0.0, 0.0, -10.0));
        assert!((p.x).abs() < 1e-5 && (p.y).abs() < 1e-5 && (p.z + 10.0).abs() < 1e-5);
    }

    #[test]
    fn perspective_depth_maps_to_zero_one() {
        let near = 0.1f32;
        let far = 1000.0f32;
        let p = Mat4::perspective_rh_gl(1.0, 1.0, near, far);
        let zn = p.mul_point(Vec3::new(0.0, 0.0, -near));
        let zf = p.mul_point(Vec3::new(0.0, 0.0, -far));
        assert!(zn.z.abs() < 1e-5, "near -> {zn:?}");
        assert!((zf.z - 1.0).abs() < 1e-5, "far -> {zf:?}");
    }

    #[test]
    fn orthographic_depth_maps_to_zero_one() {
        let p = Mat4::orthographic_rh_gl(-2.0, 2.0, -1.0, 1.0, 0.5, 50.0);
        let zn = p.mul_point(Vec3::new(0.0, 0.0, -0.5));
        let zf = p.mul_point(Vec3::new(0.0, 0.0, -50.0));
        assert!(zn.z.abs() < 1e-5, "near -> {zn:?}");
        assert!((zf.z - 1.0).abs() < 1e-5, "far -> {zf:?}");
        // x range maps to NDC
        let xl = p.mul_point(Vec3::new(-2.0, 0.0, -1.0));
        let xr = p.mul_point(Vec3::new(2.0, 0.0, -1.0));
        assert!((xl.x + 1.0).abs() < 1e-5);
        assert!((xr.x - 1.0).abs() < 1e-5);
    }

    #[test]
    fn decompose_round_trips_trs() {
        let (s, r, t) = (
            Vec3::new(2.0, 3.0, 4.0),
            Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize(), 1.234),
            Vec3::new(1.0, 2.0, 3.0),
        );
        let m = Mat4::from_scale_rotation_translation(s, r, t);
        let (s2, r2, t2) = m.decompose();
        assert!((s - s2).length() < 1e-4);
        assert!(t.distance(t2) < 1e-4);
        assert!(r.dot(r2).abs() > 0.9999, "q {:?} vs {:?}", r, r2);
    }

    #[test]
    fn normal_matrix_handles_non_uniform_scale() {
        let m = Mat4::scale(Vec3::new(2.0, 1.0, 1.0));
        let n = m.normal_matrix();
        assert!((n.mul_vec3(Vec3::Y) - Vec3::Y).length() < 1e-5);
        assert!((n.mul_vec3(Vec3::X) - Vec3::new(0.5, 0.0, 0.0)).length() < 1e-5);
    }
}
