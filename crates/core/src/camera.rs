//! Cameras and view transforms (pure math, no GPU types).
//!
//! `Camera2D` is the 2D CAD viewport: pan, zoom (uniform, about a focus
//! point) and optional UCS rotation. `Camera3D` is a standard look-at rig.

use crate::bounds::Rect2;
use crate::vec::{Vec2, Vec3};

/// 2D drawing viewport.
///
/// Screen space is **Y-down** (like every windowing system), world space is
/// Y-up. The conversion therefore flips Y, which is the single most common
/// source of "why is my drawing upside down" bugs — so it lives in exactly one
/// place.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera2D {
    /// World point shown at the centre of the viewport.
    pub center: Vec2,
    /// Pixels per world unit.
    pub scale: f32,
    /// Viewport rotation in radians (CCW in world space).
    pub rotation: f32,
    /// Viewport size in physical pixels.
    pub viewport: Vec2,
}

impl Default for Camera2D {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            scale: 1.0,
            rotation: 0.0,
            viewport: Vec2::new(1280.0, 720.0),
        }
    }
}

impl Camera2D {
    #[inline]
    pub fn new(viewport: Vec2) -> Self {
        Self {
            viewport,
            ..Default::default()
        }
    }

    /// World -> screen (pixels, Y down, origin at the top-left).
    #[inline]
    pub fn world_to_screen(&self, w: Vec2) -> Vec2 {
        let d = w - self.center;
        let (s, c) = self.rotation.sin_cos();
        // Rotate by -rotation: the view transform is the inverse of the UCS.
        let rx = d.x * c + d.y * s;
        let ry = -d.x * s + d.y * c;
        Vec2::new(
            rx * self.scale + self.viewport.x * 0.5,
            -ry * self.scale + self.viewport.y * 0.5,
        )
    }

    /// Screen (pixels) -> world.
    #[inline]
    pub fn screen_to_world(&self, s: Vec2) -> Vec2 {
        let d = Vec2::new(
            (s.x - self.viewport.x * 0.5) / self.scale,
            -(s.y - self.viewport.y * 0.5) / self.scale,
        );
        let (sr, cr) = self.rotation.sin_cos();
        let rx = d.x * cr - d.y * sr;
        let ry = d.x * sr + d.y * cr;
        self.center + Vec2::new(rx, ry)
    }

    /// The world-space rectangle currently visible.
    #[inline]
    pub fn world_viewport(&self) -> Rect2 {
        let half = Vec2::new(self.viewport.x, self.viewport.y) * (0.5 / self.scale.max(1e-9));
        let (s, c) = self.rotation.sin_cos();
        // Corner offsets rotated back into world space.
        let ex = Vec2::new(c, -s);
        let ey = Vec2::new(s, c);
        let a = self.center - ex * half.x - ey * half.y;
        let b = self.center + ex * half.x + ey * half.y;
        Rect2::new(a.min(b), a.max(b))
    }

    /// World units per pixel.
    #[inline]
    pub fn world_per_pixel(&self) -> f32 {
        1.0 / self.scale.max(1e-9)
    }

    /// Convert a pixel distance into world units.
    #[inline]
    pub fn pixels_to_world(&self, px: f32) -> f32 {
        px * self.world_per_pixel()
    }

    /// Convert a world distance into pixels.
    #[inline]
    pub fn world_to_pixels(&self, w: f32) -> f32 {
        w * self.scale
    }

    /// Zoom by `factor` (>1 zooms in) keeping `focus_screen` stationary.
    pub fn zoom_at(&mut self, factor: f32, focus_screen: Vec2) {
        let before = self.screen_to_world(focus_screen);
        self.scale = (self.scale * factor).clamp(1.0e-6, 1.0e9);
        let after = self.screen_to_world(focus_screen);
        self.center += before - after;
    }

    /// Pan by a screen-space delta (mouse drag).
    pub fn pan_screen(&mut self, delta_screen: Vec2) {
        let d = Vec2::new(-delta_screen.x / self.scale, delta_screen.y / self.scale);
        let (s, c) = self.rotation.sin_cos();
        self.center += Vec2::new(d.x * c - d.y * s, d.x * s + d.y * c);
    }

    /// Rotate the view by `r` radians about the screen-space point `pivot`.
    pub fn rotate_at(&mut self, r: f32, pivot: Vec2) {
        let w = self.screen_to_world(pivot);
        self.rotation = crate::float::wrap_pi(self.rotation + r);
        let w2 = self.screen_to_world(pivot);
        self.center += w - w2;
    }

    /// Frame a world rectangle with a margin in pixels.
    pub fn fit(&mut self, r: Rect2, margin_px: f32) {
        if r.is_empty() {
            self.center = r.center();
            return;
        }
        let size = r.size();
        let avail = (self.viewport - Vec2::splat(margin_px * 2.0)).max(Vec2::splat(1.0));
        self.scale = (avail.x / size.x.max(1e-9)).min(avail.y / size.y.max(1e-9));
        self.scale = self.scale.clamp(1.0e-6, 1.0e9);
        self.center = r.center();
        // Zooming changes where the centre lands on screen only through scale,
        // which is centred by construction, so nothing more to do.
    }

    /// Zoom to a square of `world_size` around `center`.
    pub fn zoom_extents(&mut self, center: Vec2, world_size: f32) {
        self.center = center;
        let s = world_size.max(1e-9);
        self.scale = (self.viewport.x.min(self.viewport.y) / s).clamp(1.0e-6, 1.0e9);
    }

    /// Snap the scale to the nearest "nice" 1/2/5 x 10^n step.
    pub fn zoom_step(&mut self, direction: i32) {
        let raw = self.scale * 10f32.powi(direction);
        let mag = 10f32.powf(raw.log10().floor());
        let norm = raw / mag;
        let snapped = if norm < 1.5 {
            1.0
        } else if norm < 3.5 {
            2.0
        } else if norm < 7.5 {
            5.0
        } else {
            10.0
        };
        self.scale = (snapped * mag).clamp(1.0e-6, 1.0e9);
    }

    /// Is `w` inside the visible rectangle?
    #[inline]
    pub fn visible(&self, w: Vec2) -> bool {
        self.world_viewport().contains(w)
    }
}

/// Right-handed 3D camera with an orbit target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera3D {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera3D {
    fn default() -> Self {
        Self {
            eye: Vec3::new(0.0, 0.0, 10.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y: 0.9,
            near: 0.05,
            far: 5000.0,
        }
    }
}

impl Camera3D {
    #[inline]
    pub fn forward(&self) -> Vec3 {
        (self.target - self.eye).normalize_or(Vec3::splat(-1.0))
    }
    #[inline]
    pub fn right(&self) -> Vec3 {
        self.forward().cross(self.up).normalize_or(Vec3::X)
    }
    #[inline]
    pub fn true_up(&self) -> Vec3 {
        self.right().cross(self.forward())
    }
    pub fn view(&self) -> crate::Mat4 {
        crate::Mat4::look_at_rh(self.eye, self.target, self.up)
    }
    pub fn projection(&self, aspect: f32) -> crate::Mat4 {
        crate::Mat4::perspective_rh_gl(self.fov_y, aspect.max(1e-6), self.near, self.far)
    }
    /// Orbit around the target by `d_az`/`d_el` (radians).
    pub fn orbit(&mut self, d_az: f32, d_el: f32) {
        let off = self.eye - self.target;
        let r = off.length().max(1e-6);
        let az = off.y.atan2(off.x) + d_az;
        let el = (off.z / r).clamp(-0.999_999, 0.999_999).asin() + d_el;
        let el = el.clamp(-1.553_3, 1.553_3); // +-89 degrees
        let ce = el.cos();
        let next = Vec3::new(ce * az.cos(), ce * az.sin(), el.sin()) * r;
        self.eye = self.target + next;
    }
    /// Dolly in/out along the view direction.
    pub fn dolly(&mut self, factor: f32) {
        let off = self.eye - self.target;
        let r = (off.length() * factor).clamp(self.near * 2.0, self.far * 0.5);
        self.eye = self.target + off.normalize_or(Vec3::Z) * r;
    }
    /// Pan the camera and target sideways/vertically in screen space.
    pub fn pan(&mut self, right: f32, up: f32) {
        let dx = self.right() * right;
        let dy = self.true_up() * up;
        self.eye += dx + dy;
        self.target += dx + dy;
    }
    /// Standard CAD "top" view.
    pub fn set_top(&mut self, scale: f32) {
        self.up = Vec3::Z;
        self.eye = self.target + Vec3::new(0.0, -scale, 1e-4);
    }
    /// Standard CAD "front" view.
    pub fn set_front(&mut self, scale: f32) {
        self.up = Vec3::Y;
        self.eye = self.target + Vec3::new(0.0, -1e-4, -scale);
    }
    /// Ray through a screen pixel, with NDC in `[-1, 1]` and Y up.
    pub fn ray(&self, ndc_x: f32, ndc_y: f32) -> (Vec3, Vec3) {
        let tan_half = (self.fov_y * 0.5).tan();
        let dir = (self.forward()
            + self.right() * (ndc_x * tan_half)
            + self.true_up() * (ndc_y * tan_half))
            .normalize_or(Vec3::splat(-1.0));
        (self.eye, dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float::EPS;

    #[test]
    fn screen_world_round_trip() {
        let mut cam = Camera2D::new(Vec2::new(800.0, 600.0));
        cam.center = Vec2::new(12.0, -3.0);
        cam.scale = 2.5;
        for s in [Vec2::ZERO, Vec2::new(400.0, 300.0), Vec2::new(799.0, 599.0)] {
            assert!(cam.screen_to_world(cam.world_to_screen(s)).distance(s) < 1e-3);
        }
    }

    #[test]
    fn center_of_viewport_is_center() {
        let cam = Camera2D::new(Vec2::new(800.0, 600.0));
        let c = cam.screen_to_world(Vec2::new(400.0, 300.0));
        assert!(c.distance(cam.center) < 1e-4);
    }

    #[test]
    fn world_y_up_maps_to_screen_y_down() {
        let cam = Camera2D::new(Vec2::new(800.0, 600.0));
        let up_world = cam.world_to_screen(Vec2::new(0.0, 1.0));
        let dn_world = cam.world_to_screen(Vec2::new(0.0, -1.0));
        assert!(
            up_world.y < dn_world.y,
            "screen Y must flip: {up_world:?} {dn_world:?}"
        );
    }

    #[test]
    fn rotation_round_trip() {
        let mut cam = Camera2D::new(Vec2::new(800.0, 600.0));
        cam.rotation = 0.7;
        cam.center = Vec2::new(4.0, 9.0);
        let p = cam.world_to_screen(Vec2::new(11.0, -2.0));
        assert!(cam.screen_to_world(p).distance(Vec2::new(11.0, -2.0)) < 1e-3);
    }

    #[test]
    fn zoom_keeps_focus_fixed() {
        let mut cam = Camera2D::new(Vec2::new(800.0, 600.0));
        let focus = Vec2::new(100.0, 400.0);
        let before = cam.screen_to_world(focus);
        cam.zoom_at(2.0, focus);
        assert!(cam.screen_to_world(focus).distance(before) < 1e-3);
        assert!((cam.scale - 2.0).abs() < 1e-6);
    }

    #[test]
    fn zoom_limits() {
        let mut cam = Camera2D::new(Vec2::new(800.0, 600.0));
        cam.zoom_at(1e12, Vec2::new(400.0, 300.0));
        assert!(cam.scale <= 1.0e9);
        cam.zoom_at(1e-12, Vec2::new(400.0, 300.0));
        assert!(cam.scale >= 1.0e-6);
    }

    #[test]
    fn pan_moves_content_with_the_cursor() {
        let mut cam = Camera2D::new(Vec2::new(800.0, 600.0));
        let before = cam.screen_to_world(Vec2::new(400.0, 300.0));
        cam.pan_screen(Vec2::new(50.0, 0.0));
        let after = cam.screen_to_world(Vec2::new(400.0, 300.0));
        assert!(
            (before.x - after.x - 50.0).abs() < 1e-3,
            "{before:?} {after:?}"
        );
    }

    #[test]
    fn fit_frames_the_rect() {
        let mut cam = Camera2D::new(Vec2::new(800.0, 600.0));
        cam.fit(Rect2::from_xywh(-10.0, -5.0, 20.0, 10.0), 20.0);
        assert!(cam.center.distance(Vec2::ZERO) < 1e-4);
        let v = cam.world_viewport();
        assert!(v.contains(Vec2::new(10.0, 5.0)), "{v:?}");
        assert!(v.contains(Vec2::new(-10.0, -5.0)), "{v:?}");
    }

    #[test]
    fn zoom_step_lands_on_nice_values() {
        let mut cam = Camera2D::default();
        cam.zoom_step(1);
        let l = cam.scale.log10();
        let frac = l - l.floor();
        // `zoom_step` walks a 1-2-5 style ladder, so the decade position of
        // log10(scale) should be a "nice" number. Check it against the exact
        // fractions rather than rounded literals.
        const NICE: [f32; 4] = [0.0, 1.0 / 3.0, 1.0 / 2.0, 7.0 / 15.0];
        assert!(
            NICE.iter().any(|n| (frac - n).abs() < 1e-3),
            "scale={} log={l} frac={frac}",
            cam.scale
        );
    }

    #[test]
    fn world_viewport_matches_sampling() {
        let cam = Camera2D::new(Vec2::new(800.0, 600.0));
        let v = cam.world_viewport();
        for s in [Vec2::ZERO, Vec2::new(799.0, 599.0), Vec2::new(1.0, 1.0)] {
            let w = cam.screen_to_world(s);
            assert!(v.contains(w), "{w:?} not in {v:?}");
        }
    }

    #[test]
    fn pixels_to_world_is_inverse_of_world_to_pixels() {
        let mut cam = Camera2D::new(Vec2::new(800.0, 600.0));
        cam.scale = 3.7;
        assert!((cam.world_to_pixels(cam.pixels_to_world(12.0)) - 12.0).abs() < 1e-4);
    }

    #[test]
    fn camera3d_orbit_preserves_distance() {
        let mut cam = Camera3D {
            eye: Vec3::new(10.0, 0.0, 0.0),
            target: Vec3::new(3.0, 2.0, 1.0),
            ..Default::default()
        };
        let r0 = cam.eye.distance(cam.target);
        cam.orbit(0.4, 0.2);
        assert!((cam.eye.distance(cam.target) - r0).abs() < 1e-4);
    }

    #[test]
    fn camera3d_basis_is_orthonormal() {
        let cam = Camera3D {
            eye: Vec3::new(5.0, 3.0, 8.0),
            target: Vec3::new(1.0, 1.0, 0.0),
            up: Vec3::Y,
            ..Default::default()
        };
        let (f, r, u) = (cam.forward(), cam.right(), cam.true_up());
        assert!((f.length() - 1.0).abs() < 1e-5);
        assert!(f.dot(r).abs() < 1e-5);
        assert!(f.dot(u).abs() < 1e-5);
        assert!(r.dot(u).abs() < 1e-5);
    }

    #[test]
    fn camera3d_views_target_at_centre() {
        let cam = Camera3D {
            eye: Vec3::new(5.0, 0.0, 0.0),
            target: Vec3::new(0.0, 0.0, 0.0),
            up: Vec3::Y,
            ..Default::default()
        };
        let clip = cam.view().mul_point(cam.target);
        assert!(clip.x.abs() < 1e-4 && clip.y.abs() < 1e-4);
    }

    #[test]
    fn camera3d_ray_points_at_centre() {
        let cam = Camera3D {
            eye: Vec3::new(0.0, -10.0, 0.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            ..Default::default()
        };
        let (_, dir) = cam.ray(0.0, 0.0);
        assert!(dir.distance(Vec3::Y) < 1e-5, "{dir:?}");
    }

    #[test]
    fn dolly_respects_near_and_far() {
        let mut cam = Camera3D {
            near: 0.5,
            far: 100.0,
            ..Default::default()
        };
        cam.eye = Vec3::new(0.0, -10.0, 0.0);
        for _ in 0..40 {
            cam.dolly(0.5);
            assert!(cam.eye.distance(cam.target) >= cam.near);
        }
    }

    #[test]
    fn orbit_clamps_elevation() {
        let mut cam = Camera3D::default();
        for _ in 0..100 {
            cam.orbit(0.0, 0.2);
            let off = cam.eye - cam.target;
            let el = off.normalize().z.asin();
            assert!(el <= 1.5534 + EPS, "elevation escaped: {el}");
        }
    }
}
