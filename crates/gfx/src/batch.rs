//! CPU-side geometry batching.
//!
//! The whole point of this module is that a frame with 500k line segments costs
//! one `write_buffer`, not 500k. Everything is `repr(C)` + `Pod` so the vectors
//! upload verbatim.

use bytemuck::{Pod, Zeroable};
use cad_core::{Rgba, Vec2, Vec3};

/// A 2D line segment in **screen pixels**, expanded on the CPU.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct LineVertex {
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub color: [f32; 4],
    /// Dashed patterns: `pattern = (on_px, off_px, phase_px, 0)`, `0` = solid.
    pub pattern: [f32; 4],
    /// Pixels; 0 means "hairline" (1 device pixel, GPU-independent).
    pub width: f32,
}

impl LineVertex {
    pub fn new(a: Vec2, b: Vec2, color: Rgba, width: f32) -> Self {
        Self {
            a: [a.x, a.y],
            b: [b.x, b.y],
            color: color.to_array(),
            pattern: [0.0; 4],
            width,
        }
    }
    pub fn dashed(mut self, on: f32, off: f32, phase: f32) -> Self {
        self.pattern = [on, off, phase, 0.0];
        self
    }
}

/// A UI rectangle/triangle in screen pixels with a solid colour.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct UiVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    /// `0` = flat fill, `1` = rounded-rect SDF, `2` = border-only.
    pub shape: f32,
    pub radius: f32,
    pub border: f32,
    /// Half-extents of the rectangle the SDF is evaluated over. Only meaningful
    /// when `shape != 0`; the fragment shader needs them because a screen-space
    /// quad carries no size of its own.
    pub half_ext: [f32; 2],
}

impl UiVertex {
    pub fn rect(x: f32, y: f32, w: f32, h: f32, color: Rgba) -> Self {
        Self {
            pos: [x, y],
            uv: [0.0, 0.0],
            color: color.to_array(),
            shape: 0.0,
            radius: 0.0,
            border: 0.0,
            half_ext: [0.0, 0.0],
        }
    }
    pub fn rounded(mut self, radius: f32) -> Self {
        self.shape = 1.0;
        self.radius = radius;
        self
    }
    pub fn outline(mut self, width: f32) -> Self {
        self.shape = 2.0;
        self.border = width;
        self
    }
    /// Set the SDF half-extents for a rounded rect or outline.
    pub fn sized(mut self, w: f32, h: f32) -> Self {
        self.half_ext = [w * 0.5, h * 0.5];
        self
    }
}

/// Push a rectangle as two triangles into `out`.
///
/// `uv` runs from `(0,0)` at the rect's min corner to `(w,h)` at its max corner,
/// so the fragment shader gets a local coordinate without needing `half_ext`.
pub fn push_rect(out: &mut Vec<UiVertex>, x: f32, y: f32, w: f32, h: f32, color: Rgba) {
    let base = UiVertex::rect(0.0, 0.0, 0.0, 0.0, color);
    let corners: [(f32, f32, f32, f32); 6] = [
        (x, y, 0.0, 0.0),
        (x + w, y, w, 0.0),
        (x, y + h, 0.0, h),
        (x + w, y, w, 0.0),
        (x + w, y + h, w, h),
        (x, y + h, 0.0, h),
    ];
    for (px, py, u, v) in corners {
        out.push(UiVertex {
            pos: [px, py],
            uv: [u, v],
            ..base
        });
    }
}

/// Push a rounded rectangle as two triangles.
///
/// The interpolated `uv` is the local coordinate in pixels from the rect's min
/// corner, so `sd_rounded_rect` can be evaluated without a second uniform.
pub fn push_rounded_rect(out: &mut Vec<UiVertex>, r: Rect, color: Rgba, radius: f32) {
    let radius = radius.min(r.width() * 0.5).min(r.height() * 0.5).max(0.0);
    let base = UiVertex::rect(0.0, 0.0, 0.0, 0.0, color).rounded(radius);
    let (x0, y0) = (r.min.x, r.min.y);
    let (w, h) = (r.width(), r.height());
    let corners: [(f32, f32, f32, f32); 6] = [
        (x0, y0, 0.0, 0.0),
        (x0 + w, y0, w, 0.0),
        (x0, y0 + h, 0.0, h),
        (x0 + w, y0, w, 0.0),
        (x0 + w, y0 + h, w, h),
        (x0, y0 + h, 0.0, h),
    ];
    for (px, py, u, v) in corners {
        out.push(UiVertex {
            pos: [px, py],
            uv: [u, v],
            ..base
        });
    }
}

/// Push a rectangle outline as four thin rectangles.
///
/// Cheaper and sharper than an SDF outline: no per-fragment distance function
/// and no alpha blending along the edge.
pub fn stroke_rect(out: &mut Vec<UiVertex>, r: Rect, color: Rgba, width: f32) {
    let h = width * 0.5;
    push_rect(out, r.min.x, r.min.y, r.width(), h, color);
    push_rect(out, r.min.x, r.max.y - h, r.width(), h, color);
    push_rect(out, r.min.x, r.min.y + h, h, r.height() - h * 2.0, color);
    push_rect(
        out,
        r.max.x - h,
        r.min.y + h,
        h,
        r.height() - h * 2.0,
        color,
    );
}

/// Rectangle alias so this module does not need the core import everywhere.
pub type Rect = cad_core::Rect2;

/// Accumulates 2D line geometry for one frame.
#[derive(Debug, Default, Clone)]
pub struct Batch2d {
    pub lines: Vec<LineVertex>,
    /// Estimated GPU bytes, for the stats overlay.
    pub bytes: usize,
}

impl Batch2d {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn clear(&mut self) {
        self.lines.clear();
        self.bytes = 0;
    }
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
    pub fn len(&self) -> usize {
        self.lines.len()
    }
    /// Add a screen-space segment, honouring a dash pattern in pixels.
    pub fn segment(&mut self, a: Vec2, b: Vec2, color: Rgba, width: f32) {
        self.lines.push(LineVertex::new(a, b, color, width));
        self.bytes += std::mem::size_of::<LineVertex>();
    }
    /// Add a dashed segment. `pattern` is `(on_px, off_px, phase_px)`.
    pub fn dashed(
        &mut self,
        a: Vec2,
        b: Vec2,
        color: Rgba,
        width: f32,
        on: f32,
        off: f32,
        phase: f32,
    ) {
        self.lines
            .push(LineVertex::new(a, b, color, width).dashed(on, off, phase));
        self.bytes += std::mem::size_of::<LineVertex>();
    }
    /// Add a closed polyline in screen space.
    pub fn polyline(&mut self, pts: &[Vec2], color: Rgba, width: f32) {
        for w in pts.windows(2) {
            self.segment(w[0], w[1], color, width);
        }
        if pts.len() > 2 && (pts[0] - pts[pts.len() - 1]).length_squared() < 1e-9 {
            self.segment(pts[pts.len() - 1], pts[0], color, width);
        }
    }
    /// Add a circle approximated by `segments` chords.
    pub fn circle(&mut self, center: Vec2, radius: f32, color: Rgba, width: f32, segments: u32) {
        let n = segments.max(8);
        let mut prev = center + Vec2::new(radius, 0.0);
        for i in 1..=n {
            let a = cad_core::TAU * i as f32 / n as f32;
            let p = center + Vec2::new(a.cos(), a.sin()) * radius;
            self.segment(prev, p, color, width);
            prev = p;
        }
    }
    /// Triangle fan fill (used for grid dots, pie wedges, arrow heads).
    pub fn triangle(&mut self, a: Vec2, b: Vec2, c: Vec2, color: Rgba, width: f32) {
        self.segment(a, b, color, width);
        self.segment(b, c, color, width);
        self.segment(c, a, color, width);
    }
}

/// A 3D vertex with position, normal and colour.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct SolidVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    /// Packed id: bit 0..7 layer index, bit 8..15 entity id (for picking).
    pub flags: f32,
}

/// A 3D line vertex: one corner of a screen-space quad.
///
/// The layout matches `pipeline::line3d_layout` byte for byte:
/// `pos`(3) + `color`(4) + `width`(1) + `across`(1) + `dir`(3) + `pad`(1)
/// = 13 floats = 52 bytes.
///
/// `pos` is the world-space endpoint this corner is attached to, `dir` the
/// world-space unit direction of the segment, and `across` is ±1 for the side of
/// the line. The vertex shader projects both endpoints, works out the screen
/// perpendicular itself, and offsets by `width * across`, which is how a 3D line
/// keeps a constant pixel width at any depth.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct LineVertex3d {
    pub pos: [f32; 3],
    pub color: [f32; 4],
    pub width: f32,
    pub across: f32,
    pub dir: [f32; 3],
    pub _pad: f32,
}

/// Accumulates 3D geometry for one frame.
#[derive(Debug, Default, Clone)]
pub struct Batch3d {
    pub solids: Vec<SolidVertex>,
    pub lines: Vec<LineVertex3d>,
}

impl Batch3d {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn clear(&mut self) {
        self.solids.clear();
        self.lines.clear();
    }
    pub fn is_empty(&self) -> bool {
        self.solids.is_empty() && self.lines.is_empty()
    }
    /// Add a triangle with a flat normal.
    pub fn triangle(&mut self, a: Vec3, b: Vec3, c: Vec3, color: Rgba) {
        let n = (b - a).cross(c - a).normalize_or(Vec3::Z);
        for p in [a, b, c] {
            self.solids.push(SolidVertex {
                pos: p.to_array(),
                normal: n.to_array(),
                color: color.to_array(),
                flags: 0.0,
            });
        }
    }
    /// Add a quad (a, b, c, d) as two triangles, CCW.
    pub fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, color: Rgba) {
        self.triangle(a, b, c, color);
        self.triangle(a, c, d, color);
    }
    /// Add one segment as six vertices, already expanded into a quad.
    ///
    /// Keeping the expansion on the CPU means the vertex shader never needs
    /// the neighbouring vertex to compute the screen-space offset.
    pub fn segment(&mut self, a: Vec3, b: Vec3, color: Rgba, width: f32) {
        let d = b - a;
        let len = d.length();
        // A zero-length segment has no direction; pick an arbitrary one so the
        // shader's normalize stays finite instead of producing NaN positions.
        let dir = if len > 1e-6 { d * (1.0 / len) } else { Vec3::X };
        let c = color.to_array();
        let dd = dir.to_array();
        // Triangles (a+, b+, b-) and (a+, b-, a-).
        let corners: [(Vec3, f32); 6] = [
            (a, 1.0),
            (b, 1.0),
            (b, -1.0),
            (a, 1.0),
            (b, -1.0),
            (a, -1.0),
        ];
        for (p, across) in corners {
            self.lines.push(LineVertex3d {
                pos: p.to_array(),
                color: c,
                width,
                across,
                dir: dd,
                _pad: 0.0,
            });
        }
    }
    pub fn polyline(&mut self, pts: &[Vec3], color: Rgba, width: f32) {
        for w in pts.windows(2) {
            self.segment(w[0], w[1], color, width);
        }
    }
    /// Add the 12 edges of an axis-aligned box.
    pub fn wire_box(&mut self, min: Vec3, max: Vec3, color: Rgba, width: f32) {
        let [a, b, c, d, e, f, g, h] = cad_core::Aabb3::new(min, max).corners();
        for (p, q) in [
            (a, b),
            (b, c),
            (c, d),
            (d, a),
            (e, f),
            (f, g),
            (g, h),
            (h, e),
            (a, e),
            (b, f),
            (c, g),
            (d, h),
        ] {
            self.segment(p, q, color, width);
        }
    }
    /// Add the six faces of a box.
    pub fn solid_box(&mut self, min: Vec3, max: Vec3, color: Rgba) {
        let [a, b, c, d, e, f, g, h] = cad_core::Aabb3::new(min, max).corners();
        self.quad(a, b, c, d, color);
        self.quad(e, f, g, h, color);
        self.quad(a, b, f, e, color);
        self.quad(b, c, g, f, color);
        self.quad(c, d, h, g, color);
        self.quad(d, a, e, h, color);
    }
    /// Add a mesh from the document model.
    pub fn mesh(&mut self, m: &cad_doc::entity::Mesh3d, color: Rgba) {
        for tri in m.indices.chunks_exact(3) {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let (a, b, c) = (m.positions[i0], m.positions[i1], m.positions[i2]);
            let n = if m.normals.len() == m.positions.len() {
                m.normals[i0]
            } else {
                (b - a).cross(c - a).normalize_or(Vec3::Z)
            };
            for p in [a, b, c] {
                self.solids.push(SolidVertex {
                    pos: p.to_array(),
                    normal: n.to_array(),
                    color: color.to_array(),
                    flags: 0.0,
                });
            }
        }
    }
    pub fn triangle_count(&self) -> usize {
        self.solids.len() / 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::Rgba;

    const C: Rgba = Rgba::new(1.0, 1.0, 1.0, 1.0);

    #[test]
    fn vertices_are_pod() {
        // `bytemuck` refuses non-POD types at compile time, so reaching here
        // means the layouts are GPU-safe.
        let v = LineVertex::new(Vec2::ZERO, Vec2::X, C, 1.0);
        let bytes: &[u8] = bytemuck::bytes_of(&v);
        assert_eq!(bytes.len(), std::mem::size_of::<LineVertex>());
        // The struct must have no implicit padding: `bytemuck` enforces that at
        // compile time via the `Pod` derive, and this pins the actual size.
        assert_eq!(bytes.len(), 52);
        let u = UiVertex::rect(0.0, 0.0, 1.0, 1.0, C);
        assert_eq!(
            bytemuck::bytes_of(&u).len(),
            std::mem::size_of::<UiVertex>()
        );
        let s = SolidVertex {
            pos: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            color: [1.0; 4],
            flags: 0.0,
        };
        assert_eq!(
            bytemuck::bytes_of(&s).len(),
            std::mem::size_of::<SolidVertex>()
        );
    }

    #[test]
    fn batch2d_counts_and_bytes() {
        let mut b = Batch2d::new();
        assert!(b.is_empty());
        b.segment(Vec2::ZERO, Vec2::new(1.0, 1.0), C, 1.0);
        assert_eq!(b.len(), 1);
        assert_eq!(b.bytes, std::mem::size_of::<LineVertex>());
        b.clear();
        assert!(b.is_empty());
        assert_eq!(b.bytes, 0);
    }

    #[test]
    fn polyline_does_not_double_close_an_already_closed_ring() {
        let mut b = Batch2d::new();
        // Three points where the last equals the first: windows(2) yields two
        // segments and the closure check must add nothing, or the closing edge
        // would be drawn twice.
        let pts = [Vec2::ZERO, Vec2::new(10.0, 0.0), Vec2::ZERO];
        b.polyline(&pts, C, 1.0);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn polyline_skips_the_degenerate_closing_segment() {
        // Same ring without repeating the first point: closure adds exactly one.
        let mut b = Batch2d::new();
        let pts = [Vec2::ZERO, Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0)];
        b.polyline(&pts, C, 1.0);
        assert_eq!(b.len(), 3);
    }

    #[test]
    fn polyline_closes_implicitly() {
        let mut b = Batch2d::new();
        let pts = [Vec2::ZERO, Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0)];
        b.polyline(&pts, C, 1.0);
        assert_eq!(b.len(), 3, "two edges plus the closing edge");
    }

    #[test]
    fn circle_segment_count() {
        let mut b = Batch2d::new();
        b.circle(Vec2::ZERO, 10.0, C, 1.0, 24);
        assert_eq!(b.len(), 24);
    }

    #[test]
    fn push_rect_makes_two_triangles() {
        let mut out = Vec::new();
        push_rect(&mut out, 0.0, 0.0, 10.0, 20.0, C);
        assert_eq!(out.len(), 6);
        // Bottom-left, bottom-right, top-left, bottom-right, top-right, top-left
        assert_eq!(out[0].pos, [0.0, 0.0]);
        assert_eq!(out[1].pos, [10.0, 0.0]);
        assert_eq!(out[2].pos, [0.0, 20.0]);
        assert_eq!(out[5].pos, [0.0, 20.0]);
    }

    #[test]
    fn push_rect_uvs_cover_the_area() {
        let mut out = Vec::new();
        push_rect(&mut out, 5.0, 5.0, 30.0, 40.0, C);
        let us: Vec<f32> = out.iter().map(|v| v.uv[0]).collect();
        let vs: Vec<f32> = out.iter().map(|v| v.uv[1]).collect();
        assert!(us.contains(&0.0) && us.contains(&30.0));
        assert!(vs.contains(&0.0) && vs.contains(&40.0));
    }

    #[test]
    fn rounded_rect_clamped_to_half_size() {
        let mut out = Vec::new();
        push_rounded_rect(&mut out, Rect::from_xywh(0.0, 0.0, 10.0, 10.0), C, 100.0);
        assert_eq!(out.len(), 6);
        assert!(
            out[0].radius <= 5.0,
            "radius must be clamped: {}",
            out[0].radius
        );
        assert_eq!(out[0].shape, 1.0);
    }

    #[test]
    fn rounded_rect_uv_spans_the_rect() {
        // The SDF is evaluated from `uv`, so it must cover 0..w and 0..h or the
        // corner rounding lands in the wrong place.
        let mut out = Vec::new();
        push_rounded_rect(&mut out, Rect::from_xywh(4.0, 7.0, 20.0, 30.0), C, 4.0);
        let us: Vec<f32> = out.iter().map(|v| v.uv[0]).collect();
        let vs: Vec<f32> = out.iter().map(|v| v.uv[1]).collect();
        assert!(us.contains(&0.0) && us.contains(&20.0), "{us:?}");
        assert!(vs.contains(&0.0) && vs.contains(&30.0), "{vs:?}");
        // Positions are absolute, not offsets.
        assert_eq!(out[0].pos, [4.0, 7.0]);
        assert_eq!(out[2].pos, [4.0, 37.0]);
    }

    #[test]
    fn stroke_rect_emits_four_bars() {
        let mut out = Vec::new();
        stroke_rect(&mut out, Rect::from_xywh(0.0, 0.0, 10.0, 20.0), C, 2.0);
        assert_eq!(out.len(), 24, "four bars of two triangles each");
        // Top and bottom bars span the full width.
        assert_eq!(out[0].pos, [0.0, 0.0]);
        assert_eq!(out[5].pos, [10.0, 1.0]);
        // Side bars are inset so corners do not overlap.
        assert_eq!(out[12].pos, [0.0, 1.0]);
        assert_eq!(out[18].pos, [9.0, 1.0]);
    }

    #[test]
    fn line3d_segment_is_six_finite_vertices() {
        let mut b = Batch3d::new();
        b.segment(Vec3::ZERO, Vec3::new(3.0, 4.0, 0.0), C, 1.0);
        assert_eq!(b.lines.len(), 6);
        // `dir` is the unit world direction; a 3-4-5 segment normalises to
        // (0.6, 0.8, 0).
        for v in &b.lines {
            assert!(v.dir.iter().all(|f| f.is_finite()), "{v:?}");
            assert!((v.dir[0] - 0.6).abs() < 1e-5, "{v:?}");
            assert!((v.dir[1] - 0.8).abs() < 1e-5, "{v:?}");
            assert_eq!(v.dir[2], 0.0);
            assert!(v.across == 1.0 || v.across == -1.0, "{v:?}");
        }
        // Three corners on each side of the line.
        assert_eq!(b.lines.iter().filter(|v| v.across > 0.0).count(), 3);
        assert_eq!(b.lines.iter().filter(|v| v.across < 0.0).count(), 3);
    }

    #[test]
    fn zero_length_3d_segment_has_a_finite_direction() {
        // normalize() of a zero vector is NaN, which would poison every vertex
        // downstream; the batch must substitute a safe axis.
        let mut b = Batch3d::new();
        b.segment(Vec3::ZERO, Vec3::ZERO, C, 1.0);
        assert_eq!(b.lines.len(), 6);
        for v in &b.lines {
            assert!(v.dir.iter().all(|f| f.is_finite()), "{v:?}");
            assert_eq!(v.dir, [1.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn solid_triangle_has_consistent_normals() {
        let mut b = Batch3d::new();
        b.triangle(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            C,
        );
        assert_eq!(b.solids.len(), 3);
        for v in &b.solids {
            assert!((v.normal[2] - 1.0).abs() < 1e-5, "{:?}", v.normal);
        }
        assert_eq!(b.triangle_count(), 1);
    }

    #[test]
    fn wire_box_has_twelve_edges() {
        let mut b = Batch3d::new();
        b.wire_box(Vec3::ZERO, Vec3::splat(1.0), C, 1.0);
        assert_eq!(b.lines.len(), 24);
    }

    #[test]
    fn solid_box_has_six_quads() {
        let mut b = Batch3d::new();
        b.solid_box(Vec3::ZERO, Vec3::splat(1.0), C);
        assert_eq!(b.triangle_count(), 12);
    }

    #[test]
    fn mesh_upload_uses_indices() {
        let mut m = cad_doc::entity::Mesh3d {
            positions: vec![
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            normals: Vec::new(),
            indices: vec![0, 1, 2],
        };
        m.recompute_normals();
        let mut b = Batch3d::new();
        b.mesh(&m, C);
        assert_eq!(b.solids.len(), 3);
        for v in &b.solids {
            assert!((v.normal[2] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn degenerate_triangle_does_not_produce_nan() {
        let mut b = Batch3d::new();
        b.triangle(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, C);
        for v in &b.solids {
            assert!(v.normal.iter().all(|n| n.is_finite()));
        }
    }
}
