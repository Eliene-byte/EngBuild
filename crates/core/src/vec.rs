//! SIMD-friendly vector types. `repr(C)` + scalar layout means they can be
//! uploaded to GPU buffers verbatim via `bytemuck` with no repacking.

use crate::float::{EPS, lerp};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ---------------------------------------------------------------- Vec2

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0 };
    pub const X: Self = Self { x: 1.0, y: 0.0 };
    pub const Y: Self = Self { x: 0.0, y: 1.0 };

    #[inline(always)]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    #[inline(always)]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v }
    }
    #[inline(always)]
    pub const fn to_array(self) -> [f32; 2] {
        [self.x, self.y]
    }
    #[inline(always)]
    pub const fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y
    }
    /// 2D scalar cross product (z component of the 3D cross).
    #[inline(always)]
    pub const fn cross(self, o: Self) -> f32 {
        self.x * o.y - self.y * o.x
    }
    #[inline(always)]
    pub const fn length_squared(self) -> f32 {
        self.dot(self)
    }
    #[inline(always)]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }
    #[inline(always)]
    pub fn normalize_or(self, fallback: Self) -> Self {
        let l2 = self.length_squared();
        if l2 > EPS * EPS {
            self * (1.0 / l2.sqrt())
        } else {
            fallback
        }
    }
    #[inline(always)]
    pub fn normalize(self) -> Self {
        self.normalize_or(Self::X)
    }
    #[inline(always)]
    pub fn distance(self, o: Self) -> f32 {
        (self - o).length()
    }
    #[inline(always)]
    pub fn distance_squared(self, o: Self) -> f32 {
        (self - o).length_squared()
    }
    #[inline(always)]
    pub const fn min(self, o: Self) -> Self {
        Self::new(self.x.min(o.x), self.y.min(o.y))
    }
    #[inline(always)]
    pub const fn max(self, o: Self) -> Self {
        Self::new(self.x.max(o.x), self.y.max(o.y))
    }
    #[inline(always)]
    pub const fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs())
    }
    #[inline(always)]
    pub const fn floor(self) -> Self {
        Self::new(self.x.floor(), self.y.floor())
    }
    #[inline(always)]
    pub const fn ceil(self) -> Self {
        Self::new(self.x.ceil(), self.y.ceil())
    }
    #[inline(always)]
    pub const fn round(self) -> Self {
        Self::new(self.x.round(), self.y.round())
    }
    #[inline(always)]
    pub const fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
    #[inline(always)]
    pub const fn lerp(self, o: Self, t: f32) -> Self {
        Self::new(lerp(self.x, o.x, t), lerp(self.y, o.y, t))
    }
    /// Rotate 90° counter-clockwise.
    #[inline(always)]
    pub const fn perp(self) -> Self {
        Self::new(-self.y, self.x)
    }
    /// Rotate by `r` radians.
    #[inline(always)]
    pub fn rotate(self, r: f32) -> Self {
        let (s, c) = r.sin_cos();
        Self::new(self.x * c - self.y * s, self.x * s + self.y * c)
    }
}

impl From<(f32, f32)> for Vec2 {
    #[inline(always)]
    fn from((x, y): (f32, f32)) -> Self {
        Self::new(x, y)
    }
}
impl From<[f32; 2]> for Vec2 {
    #[inline(always)]
    fn from(a: [f32; 2]) -> Self {
        Self::new(a[0], a[1])
    }
}
impl From<Vec3> for Vec2 {
    #[inline(always)]
    fn from(v: Vec3) -> Self {
        Self::new(v.x, v.y)
    }
}

macro_rules! binop {
    ($tr:path, $m:ident, $f:ident, $op:tt) => {
        impl $tr for Vec2 {
            type Output = Vec2;
            #[inline(always)]
            fn $m(self, o: Vec2) -> Vec2 {
                Vec2::$f(self.x $op o.x, self.y $op o.y)
            }
        }
    };
}
binop!(Add, add, new, +);
binop!(Sub, sub, new, -);

impl Mul<f32> for Vec2 {
    type Output = Vec2;
    #[inline(always)]
    fn mul(self, s: f32) -> Vec2 {
        Vec2::new(self.x * s, self.y * s)
    }
}
impl Mul<Vec2> for f32 {
    type Output = Vec2;
    #[inline(always)]
    fn mul(self, v: Vec2) -> Vec2 {
        Vec2::new(v.x * self, v.y * self)
    }
}
impl Div<f32> for Vec2 {
    type Output = Vec2;
    #[inline(always)]
    fn div(self, s: f32) -> Vec2 {
        Vec2::new(self.x / s, self.y / s)
    }
}
impl Neg for Vec2 {
    type Output = Vec2;
    #[inline(always)]
    fn neg(self) -> Vec2 {
        Vec2::new(-self.x, -self.y)
    }
}
impl AddAssign for Vec2 {
    #[inline(always)]
    fn add_assign(&mut self, o: Vec2) {
        *self = *self + o;
    }
}
impl SubAssign for Vec2 {
    #[inline(always)]
    fn sub_assign(&mut self, o: Vec2) {
        *self = *self - o;
    }
}
impl MulAssign<f32> for Vec2 {
    #[inline(always)]
    fn mul_assign(&mut self, s: f32) {
        *self = *self * s;
    }
}
impl DivAssign<f32> for Vec2 {
    #[inline(always)]
    fn div_assign(&mut self, s: f32) {
        *self = *self / s;
    }
}

// ---------------------------------------------------------------- Vec3

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };
    pub const X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    pub const Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    pub const Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    #[inline(always)]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }
    #[inline(always)]
    pub const fn from_vec2(v: Vec2, z: f32) -> Self {
        Self { x: v.x, y: v.y, z }
    }
    #[inline(always)]
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
    #[inline(always)]
    pub const fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }
    #[inline(always)]
    pub const fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    #[inline(always)]
    pub const fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    #[inline(always)]
    pub const fn length_squared(self) -> f32 {
        self.dot(self)
    }
    #[inline(always)]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }
    #[inline(always)]
    pub fn normalize_or(self, fallback: Self) -> Self {
        let l2 = self.length_squared();
        if l2 > EPS * EPS {
            self * (1.0 / l2.sqrt())
        } else {
            fallback
        }
    }
    #[inline(always)]
    pub fn normalize(self) -> Self {
        self.normalize_or(Self::X)
    }
    #[inline(always)]
    pub fn distance(self, o: Self) -> f32 {
        (self - o).length()
    }
    #[inline(always)]
    pub fn distance_squared(self, o: Self) -> f32 {
        (self - o).length_squared()
    }
    #[inline(always)]
    pub const fn min(self, o: Self) -> Self {
        Self::new(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }
    #[inline(always)]
    pub const fn max(self, o: Self) -> Self {
        Self::new(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }
    #[inline(always)]
    pub const fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }
    #[inline(always)]
    pub const fn floor(self) -> Self {
        Self::new(self.x.floor(), self.y.floor(), self.z.floor())
    }
    #[inline(always)]
    pub const fn ceil(self) -> Self {
        Self::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }
    #[inline(always)]
    pub const fn round(self) -> Self {
        Self::new(self.x.round(), self.y.round(), self.z.round())
    }
    #[inline(always)]
    pub const fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
    #[inline(always)]
    pub const fn lerp(self, o: Self, t: f32) -> Self {
        Self::new(
            lerp(self.x, o.x, t),
            lerp(self.y, o.y, t),
            lerp(self.z, o.z, t),
        )
    }
    /// Rotate 90° counter-clockwise (in the XY plane), keeping Z.
    #[inline(always)]
    pub fn perp(self) -> Self {
        Self::new(-self.y, self.x, self.z)
    }
    /// Rotate about the Z axis by `r` radians (Z is preserved).
    #[inline]
    pub fn rotate_z(self, r: f32) -> Self {
        let (s, c) = r.sin_cos();
        Self::new(self.x * c - self.y * s, self.x * s + self.y * c, self.z)
    }
    /// Any unit vector orthogonal to `self`.
    #[inline]
    pub fn any_orthonormal(self) -> Self {
        let a = if self.x.abs() > 0.9 { Vec3::Y } else { Vec3::X };
        self.cross(a).normalize()
    }
    /// Builds an orthonormal basis with `self` as the third axis.
    #[inline]
    pub fn orthonormal_basis(&self) -> (Vec3, Vec3) {
        let t = self.any_orthonormal();
        let b = self.cross(t).normalize();
        (t, b)
    }
}

impl From<[f32; 3]> for Vec3 {
    #[inline(always)]
    fn from(a: [f32; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }
}
impl From<(f32, f32, f32)> for Vec3 {
    #[inline(always)]
    fn from((x, y, z): (f32, f32, f32)) -> Self {
        Self::new(x, y, z)
    }
}
impl From<Vec2> for Vec3 {
    #[inline(always)]
    fn from(v: Vec2) -> Self {
        Self::new(v.x, v.y, 0.0)
    }
}
impl From<(Vec3, f32)> for Vec4 {
    #[inline(always)]
    fn from((v, w): (Vec3, f32)) -> Self {
        Self::new(v.x, v.y, v.z, w)
    }
}

macro_rules! binop3 {
    ($tr:path, $m:ident, $op:tt) => {
        impl $tr for Vec3 {
            type Output = Vec3;
            #[inline(always)]
            fn $m(self, o: Vec3) -> Vec3 {
                Vec3::new(self.x $op o.x, self.y $op o.y, self.z $op o.z)
            }
        }
    };
}
binop3!(Add, add, +);
binop3!(Sub, sub, -);

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}
impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}
impl Div<f32> for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn div(self, s: f32) -> Vec3 {
        Vec3::new(self.x / s, self.y / s, self.z / s)
    }
}
impl Neg for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}
impl AddAssign for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, o: Vec3) {
        *self = *self + o;
    }
}
impl SubAssign for Vec3 {
    #[inline(always)]
    fn sub_assign(&mut self, o: Vec3) {
        *self = *self - o;
    }
}
impl MulAssign<f32> for Vec3 {
    #[inline(always)]
    fn mul_assign(&mut self, s: f32) {
        *self = *self * s;
    }
}
impl DivAssign<f32> for Vec3 {
    #[inline(always)]
    fn div_assign(&mut self, s: f32) {
        *self = *self / s;
    }
}

// ---------------------------------------------------------------- Vec4

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };

    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
    #[inline(always)]
    pub const fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }
    #[inline(always)]
    pub const fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
    #[inline(always)]
    pub const fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w
    }
    #[inline(always)]
    pub const fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
}

impl From<[f32; 4]> for Vec4 {
    #[inline(always)]
    fn from(a: [f32; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }
}

macro_rules! binop4 {
    ($tr:path, $m:ident, $op:tt) => {
        impl $tr for Vec4 {
            type Output = Vec4;
            #[inline(always)]
            fn $m(self, o: Vec4) -> Vec4 {
                Vec4::new(self.x $op o.x, self.y $op o.y, self.z $op o.z, self.w $op o.w)
            }
        }
    };
}
binop4!(Add, add, +);
binop4!(Sub, sub, -);

impl Mul<f32> for Vec4 {
    type Output = Vec4;
    #[inline(always)]
    fn mul(self, s: f32) -> Vec4 {
        Vec4::new(self.x * s, self.y * s, self.z * s, self.w * s)
    }
}
impl Div<f32> for Vec4 {
    type Output = Vec4;
    #[inline(always)]
    fn div(self, s: f32) -> Vec4 {
        Vec4::new(self.x / s, self.y / s, self.z / s, self.w / s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_and_length() {
        let x = Vec3::X;
        let y = Vec3::Y;
        assert!((x.cross(y) - Vec3::Z).length() < 1e-6);
        assert!((x.length() - 1.0).abs() < 1e-6);
        assert!((Vec3::new(3.0, 4.0, 0.0).length() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_degenerate_falls_back() {
        assert_eq!(Vec3::ZERO.normalize(), Vec3::X);
        assert_eq!(Vec2::ZERO.normalize(), Vec2::X);
    }

    #[test]
    fn orthonormal_basis_is_orthonormal() {
        let n = Vec3::new(0.3, -0.9, 0.31).normalize();
        let (t, b) = n.orthonormal_basis();
        assert!(t.dot(n).abs() < 1e-5);
        assert!(b.dot(n).abs() < 1e-5);
        assert!((t.dot(b)).abs() < 1e-5);
        assert!((t.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn vec2_ops() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(4.0, 6.0);
        assert_eq!(a + b, Vec2::new(5.0, 8.0));
        assert_eq!(b - a, Vec2::new(3.0, 4.0));
        assert_eq!(a * 2.0, Vec2::new(2.0, 4.0));
        assert_eq!(a.perp(), Vec2::new(-2.0, 1.0));
        assert_eq!(a.rotate(core::f32::consts::FRAC_PI_2).x, -2.0);
    }
}
