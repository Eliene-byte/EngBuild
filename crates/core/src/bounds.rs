//! Axis-aligned bounds: 2D screen/world rect and 3D bounding box.

use crate::vec::{Vec2, Vec3};

// ---------------------------------------------------------------- Rect2

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Rect2 {
    pub min: Vec2,
    pub max: Vec2,
}

impl Rect2 {
    pub const ZERO: Self = Self {
        min: Vec2::ZERO,
        max: Vec2::ZERO,
    };

    /// A rect containing nothing, for accumulating a bounding box.
    ///
    /// `ZERO` is the rect *at the origin*, not an identity for `expand_point`:
    /// growing it can only ever grow away from `(0, 0)`, so any geometry that
    /// does not contain the origin reports bounds that include it. Every
    /// "bounds of these points" loop must start here, or from the first point.
    pub const EMPTY: Self = Self {
        min: Vec2::splat(f32::INFINITY),
        max: Vec2::splat(f32::NEG_INFINITY),
    };

    /// Bounding box of `pts`, or [`Rect2::EMPTY`] when there are none.
    pub fn of_points(pts: &[Vec2]) -> Self {
        let mut r = Self::EMPTY;
        for p in pts {
            r = r.expand_point(*p);
        }
        r
    }

    #[inline(always)]
    pub const fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }
    #[inline(always)]
    pub const fn from_xywh(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self::new(Vec2::new(x, y), Vec2::new(x + w, y + h))
    }
    #[inline(always)]
    pub const fn size(self) -> Vec2 {
        Vec2::new(self.max.x - self.min.x, self.max.y - self.min.y)
    }
    #[inline(always)]
    pub const fn width(self) -> f32 {
        self.max.x - self.min.x
    }
    #[inline(always)]
    pub const fn height(self) -> f32 {
        self.max.y - self.min.y
    }
    #[inline(always)]
    pub fn center(self) -> Vec2 {
        Vec2::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
        )
    }
    #[inline(always)]
    pub const fn area(self) -> f32 {
        self.width() * self.height()
    }
    /// True only when the rect has no extent on **both** axes.
    ///
    /// Deliberately *not* "has zero area": a horizontal line has zero height but
    /// is a perfectly valid, pickable rectangle. Use [`Rect2::has_area`] when you
    /// really do mean the area.
    #[inline(always)]
    pub fn is_empty(self) -> bool {
        self.max.x <= self.min.x && self.max.y <= self.min.y
    }
    /// True when the rect encloses no area (it is a line or a point).
    #[inline(always)]
    pub fn has_area(self) -> bool {
        self.width() > 0.0 && self.height() > 0.0
    }
    /// Inclusive axis-aligned overlap test.
    #[inline(always)]
    pub fn overlaps(self, o: Rect2) -> bool {
        self.min.x <= o.max.x
            && o.min.x <= self.max.x
            && self.min.y <= o.max.y
            && o.min.y <= self.max.y
    }
    #[inline(always)]
    pub const fn contains(self, p: Vec2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }
    #[inline(always)]
    pub const fn contains_rect(self, o: Rect2) -> bool {
        self.contains(o.min) && self.contains(o.max)
    }
    #[inline(always)]
    pub fn intersect(self, o: Rect2) -> Rect2 {
        Rect2::new(self.min.max(o.min), self.max.min(o.max))
    }
    #[inline(always)]
    pub fn union(self, o: Rect2) -> Rect2 {
        Rect2::new(self.min.min(o.min), self.max.max(o.max))
    }
    #[inline(always)]
    pub fn expand_point(self, p: Vec2) -> Rect2 {
        Rect2::new(self.min.min(p), self.max.max(p))
    }
    #[inline(always)]
    pub fn expand(self, v: Vec2) -> Rect2 {
        Rect2::new(self.min - v, self.max + v)
    }
    #[inline(always)]
    pub fn translated(self, v: Vec2) -> Rect2 {
        Rect2::new(self.min + v, self.max + v)
    }
    #[inline(always)]
    pub fn inset(self, v: Vec2) -> Rect2 {
        Rect2::new(self.min + v, self.max - v)
    }
    /// Grows the rect in *pixels* regardless of the current zoom, i.e. the
    /// pixels-to-world scale factor must be applied by the caller.
    #[inline]
    pub fn grow_px(self, px: f32, px_per_world: f32) -> Rect2 {
        let d = Vec2::splat(px / px_per_world.max(1e-9));
        self.expand(d)
    }

    /// Four corners in CCW order starting from bottom-left (Y-down screen space).
    #[inline(always)]
    pub fn corners(self) -> [Vec2; 4] {
        [
            self.min,
            Vec2::new(self.max.x, self.min.y),
            self.max,
            Vec2::new(self.min.x, self.max.y),
        ]
    }
    /// Uniform scale that fits `self` inside `target`, preserving aspect.
    #[inline]
    pub fn fit_uniform(self, target: Rect2) -> (f32, Vec2) {
        let s = self.size();
        let t = target.size();
        if s.x <= 0.0 || s.y <= 0.0 || t.x <= 0.0 || t.y <= 0.0 {
            return (1.0, target.center());
        }
        let scale = (t.x / s.x).min(t.y / s.y);
        (scale, target.center())
    }

    /// Shortest distance from `p` to the rect (0 when inside).
    #[inline]
    pub fn distance_to(self, p: Vec2) -> f32 {
        let d = Vec2::new(
            (self.min.x - p.x).max(0.0).max(p.x - self.max.x),
            (self.min.y - p.y).max(0.0).max(p.y - self.max.y),
        );
        d.length()
    }

    /// Liang–Barsky clip of a segment against this rect.
    /// Returns the `[t0, t1]` parameter interval of the part inside.
    #[inline]
    pub fn clip_segment(&self, a: Vec2, b: Vec2) -> Option<(f32, f32)> {
        let d = b - a;
        let mut t0 = 0.0f32;
        let mut t1 = 1.0f32;
        let p = [-d.x, d.x, -d.y, d.y];
        let q = [
            a.x - self.min.x,
            self.max.x - a.x,
            a.y - self.min.y,
            self.max.y - a.y,
        ];
        for i in 0..4 {
            if p[i].abs() < 1e-12 {
                if q[i] < 0.0 {
                    return None;
                }
                continue;
            }
            let r = q[i] / p[i];
            if p[i] < 0.0 {
                if r > t1 {
                    return None;
                }
                t0 = t0.max(r);
            } else {
                if r < t0 {
                    return None;
                }
                t1 = t1.min(r);
            }
        }
        Some((t0, t1))
    }
}

// ---------------------------------------------------------------- Aabb3

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Aabb3 {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb3 {
    pub const ZERO: Self = Self {
        min: Vec3::ZERO,
        max: Vec3::ZERO,
    };
    pub const INFINITE: Self = Self {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    #[inline(always)]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }
    #[inline(always)]
    pub const fn size(self) -> Vec3 {
        Vec3::new(
            self.max.x - self.min.x,
            self.max.y - self.min.y,
            self.max.z - self.min.z,
        )
    }
    #[inline(always)]
    pub fn center(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    #[inline(always)]
    pub fn extents_half(self) -> Vec3 {
        self.size() * 0.5
    }
    #[inline(always)]
    pub const fn contains(self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }
    #[inline(always)]
    pub fn union(self, o: Self) -> Self {
        Self::new(self.min.min(o.min), self.max.max(o.max))
    }
    #[inline(always)]
    pub fn expand_point(self, p: Vec3) -> Self {
        Self::new(self.min.min(p), self.max.max(p))
    }
    #[inline(always)]
    pub fn corners(self) -> [Vec3; 8] {
        let (a, b) = (self.min, self.max);
        [
            Vec3::new(a.x, a.y, a.z),
            Vec3::new(b.x, a.y, a.z),
            Vec3::new(b.x, b.y, a.z),
            Vec3::new(a.x, b.y, a.z),
            Vec3::new(a.x, a.y, b.z),
            Vec3::new(b.x, a.y, b.z),
            Vec3::new(b.x, b.y, b.z),
            Vec3::new(a.x, b.y, b.z),
        ]
    }
    /// Squared distance from a point to the box (0 when inside).
    #[inline]
    pub fn distance_squared_to(self, p: Vec3) -> f32 {
        let d = Vec3::new(
            (self.min.x - p.x).max(0.0).max(p.x - self.max.x),
            (self.min.y - p.y).max(0.0).max(p.y - self.max.y),
            (self.min.z - p.z).max(0.0).max(p.z - self.max.z),
        );
        d.length_squared()
    }
    /// Slab test. `inv_dir` may contain infinities.
    #[inline]
    pub fn intersect_ray(&self, origin: Vec3, inv_dir: Vec3, t_max: f32) -> Option<f32> {
        let t0 = Vec3::new(
            (self.min.x - origin.x) * inv_dir.x,
            (self.min.y - origin.y) * inv_dir.y,
            (self.min.z - origin.z) * inv_dir.z,
        );
        let t1 = Vec3::new(
            (self.max.x - origin.x) * inv_dir.x,
            (self.max.y - origin.y) * inv_dir.y,
            (self.max.z - origin.z) * inv_dir.z,
        );
        let lo = t0.min(t1);
        let hi = t0.max(t1);
        let tn = lo.x.max(lo.y).max(lo.z).max(0.0);
        let tf = hi.x.min(hi.y).min(hi.z).min(t_max);
        if tf >= tn { Some(tn) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_basics() {
        let r = Rect2::from_xywh(10.0, 20.0, 100.0, 50.0);
        assert_eq!(r.size(), Vec2::new(100.0, 50.0));
        assert_eq!(r.center(), Vec2::new(60.0, 45.0));
        assert!(r.contains(Vec2::new(11.0, 21.0)));
        assert!(!r.contains(Vec2::new(9.0, 21.0)));
        assert!((r.distance_to(Vec2::ZERO) - 22.36).abs() < 0.01);
    }

    #[test]
    fn rect_union_intersect() {
        let a = Rect2::from_xywh(0.0, 0.0, 10.0, 10.0);
        let b = Rect2::from_xywh(5.0, 5.0, 10.0, 10.0);
        assert_eq!(a.union(b), Rect2::from_xywh(0.0, 0.0, 15.0, 15.0));
        assert_eq!(a.intersect(b), Rect2::from_xywh(5.0, 5.0, 5.0, 5.0));
        assert!(
            a.intersect(Rect2::from_xywh(50.0, 50.0, 1.0, 1.0))
                .is_empty()
        );
    }

    #[test]
    fn empty_starts_empty_and_absorbs_any_point() {
        // The failure this prevents: starting from `ZERO` silently pins the
        // lower corner at the origin.
        assert!(Rect2::EMPTY.is_empty());
        let r = Rect2::EMPTY.expand_point(Vec2::new(10.0, 20.0));
        assert_eq!(r.min, Vec2::new(10.0, 20.0), "{r:?}");
        assert_eq!(r.max, Vec2::new(10.0, 20.0));
        // A single point has no extent on either axis, which `is_empty` counts
        // as empty; that is the documented meaning, not a bug.
        assert!(r.is_empty(), "a single point has no extent: {r:?}");
        let r = r.expand_point(Vec2::new(14.0, 26.0));
        assert_eq!(r.min, Vec2::new(10.0, 20.0), "{r:?}");
        assert_eq!(r.max, Vec2::new(14.0, 26.0));
        assert!(!r.is_empty());
        // A point below the origin moves min, which `ZERO` could never do.
        let r = Rect2::EMPTY.expand_point(Vec2::new(-10.0, -20.0));
        assert_eq!(r.min, Vec2::new(-10.0, -20.0), "{r:?}");
    }

    #[test]
    fn of_points_is_the_true_bounding_box() {
        let r = Rect2::of_points(&[
            Vec2::new(3.0, 9.0),
            Vec2::new(-4.0, 1.0),
            Vec2::new(2.0, -8.0),
        ]);
        assert_eq!(r.min, Vec2::new(-4.0, -8.0), "{r:?}");
        assert_eq!(r.max, Vec2::new(3.0, 9.0), "{r:?}");
        assert!(Rect2::of_points(&[]).is_empty());
    }

    #[test]
    fn segment_clipping() {
        let r = Rect2::from_xywh(0.0, 0.0, 10.0, 10.0);
        let (t0, t1) = r
            .clip_segment(Vec2::new(-5.0, 5.0), Vec2::new(15.0, 5.0))
            .unwrap();
        assert!((t0 - 0.25).abs() < 1e-5);
        assert!((t1 - 0.75).abs() < 1e-5);
        assert!(
            r.clip_segment(Vec2::new(-5.0, 50.0), Vec2::new(-1.0, 50.0))
                .is_none()
        );
    }

    #[test]
    fn ray_box_hit() {
        let b = Aabb3::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        // dir = (1, 0, 0)  =>  inv_dir = (1, inf, inf)
        let inv = Vec3::new(1.0, f32::INFINITY, f32::INFINITY);
        let t = b.intersect_ray(Vec3::new(-10.0, 0.0, 0.0), inv, f32::INFINITY);
        assert_eq!(t, Some(9.0));
        let miss = b.intersect_ray(Vec3::new(-10.0, 5.0, 0.0), inv, f32::INFINITY);
        assert!(miss.is_none());
        // clipped by t_max
        assert_eq!(b.intersect_ray(Vec3::new(-10.0, 0.0, 0.0), inv, 5.0), None);
    }
}
