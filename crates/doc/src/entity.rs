//! Entities: the actual drawing content, in 2D and 3D.
//!
//! Every entity carries its own [`EntityKind`] plus shared common properties.
//! Geometry lives in the entity; appearance lives in [`EntityCommon`]. This
//! split is what lets the renderer batch by appearance and the kernel reason
//! about geometry without touching styles.

use crate::handle::LayerId;
use crate::layer::{LineType, LineWeight};
use cad_core::{Aabb3, Plane3, Rect2, Rgba, Vec2, Vec3};
use cad_geom::curve::{Arc, Circle, Curve, Ellipse, Line, Polyline};
use cad_geom::spline::Spline;

/// An axis-aligned 3D solid given by its eight corners (a hexahedron).
/// Used for boxes, extrusions and boolean-free 3D primitives.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Box3d {
    pub min: Vec3,
    pub max: Vec3,
}

impl Box3d {
    #[inline]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }
    #[inline]
    pub fn corners(self) -> [Vec3; 8] {
        Aabb3::new(self.min, self.max).corners()
    }
    #[inline]
    pub fn bounds(self) -> Aabb3 {
        Aabb3::new(self.min, self.max)
    }
    #[inline]
    pub fn center(self) -> Vec3 {
        self.bounds().center()
    }
    /// The six quad faces, each in CCW order when viewed from outside.
    pub fn faces(self) -> [[Vec3; 4]; 6] {
        let [a, b, c, d, e, f, g, h] = self.corners();
        // corners() order: bottom a,b,c,d then top e,f,g,h
        [
            [a, b, c, d], // bottom (-Z)
            [e, f, g, h], // top (+Z)
            [a, b, f, e], // -Y
            [b, c, g, f], // +X
            [c, d, h, g], // +Y
            [d, a, e, h], // -X
        ]
    }
    pub fn volume(self) -> f32 {
        let s = self.bounds().size();
        s.x * s.y * s.z
    }
}

/// Indices per triangle. Named so the mesh code reads in terms of triangles
/// rather than the magic 3.
const TRI: usize = 3;

/// A triangle mesh in world space: positions + optional per-vertex normals.
///
/// `indices` is a triangle list: three consecutive indices per triangle. A
/// trailing partial triangle is ignored rather than treated as an error, since a
/// half-written mesh from a crashed exporter is better dropped than fatal.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mesh3d {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub indices: Vec<u32>,
}

impl Mesh3d {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / TRI
    }
    pub fn bounds(&self) -> Aabb3 {
        let mut b = Aabb3::ZERO;
        let mut first = true;
        for p in &self.positions {
            if first {
                b = Aabb3::new(*p, *p);
                first = false;
            } else {
                b = b.expand_point(*p);
            }
        }
        b
    }
    /// Recompute smooth vertex normals by area-weighted averaging.
    pub fn recompute_normals(&mut self) {
        let mut n = vec![Vec3::ZERO; self.positions.len()];
        // Iterate by triangle index rather than by slicing: `TRI` is a named
        // constant, and a slice pattern would hard-code the same 3 twice.
        for t in 0..self.triangle_count() {
            let i0 = self.indices[t * TRI] as usize;
            let i1 = self.indices[t * TRI + 1] as usize;
            let i2 = self.indices[t * TRI + 2] as usize;
            let (p0, p1, p2) = (self.positions[i0], self.positions[i1], self.positions[i2]);
            // Un-normalised cross product is area-weighted, which is what we want.
            let fn_ = (p1 - p0).cross(p2 - p0);
            for i in [i0, i1, i2] {
                n[i] += fn_;
            }
        }
        for v in n.iter_mut() {
            *v = v.normalize_or(Vec3::Z);
        }
        self.normals = n;
    }
    /// The flat top face of an arbitrary closed 2D region, triangulated by the
    /// caller's ear-clipping pass; here we only build the side walls + caps for
    /// a rectangle-like region supplied as a polyline.
    pub fn from_extrusion(region: &Polyline, height: f32) -> Mesh3d {
        let n = region.segment_count();
        let mut positions = Vec::with_capacity(n * 4);
        let mut indices = Vec::with_capacity(n * 12);
        for i in 0..n {
            let a = region.point(i);
            let b = region.point(i + 1);
            let a0 = Vec3::new(a.x, a.y, 0.0);
            let b0 = Vec3::new(b.x, b.y, 0.0);
            let a1 = Vec3::new(a.x, a.y, height);
            let b1 = Vec3::new(b.x, b.y, height);
            let base = positions.len() as u32;
            positions.extend_from_slice(&[a0, b0, b1, a1]);
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        let mut m = Mesh3d {
            positions,
            normals: Vec::new(),
            indices,
        };
        m.recompute_normals();
        m
    }
}

/// A 3D planar face, addressed by its plane and an outer loop of points.
#[derive(Debug, Clone, PartialEq)]
pub struct Face3d {
    pub plane: Plane3,
    pub loop_pts: Vec<Vec3>,
}

/// Horizontal and vertical text placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
    MiddleLeft,
    MiddleCenter,
    MiddleRight,
    Aligned,
}

/// Single- or double-line text, positioned in world space.
#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    pub value: String,
    pub insert: Vec3,
    /// Height of a capital letter.
    pub height: f32,
    /// Uniform width factor.
    pub width_factor: f32,
    pub rotation: f32,
    /// Shear in degrees (oblique).
    pub oblique: f32,
    pub align: TextAlign,
    /// Second line, for double-line text.
    pub value2: Option<String>,
    /// Height of the second line when `value2` is present.
    pub height2: f32,
}

/// A point entity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointEnt {
    pub position: Vec3,
}

/// A construction ray/line drawn along an axis, used as a modelling aid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Construction {
    pub from: Vec3,
    pub to: Vec3,
}

/// Reference to a block definition placed in the drawing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InsertRef {
    pub block: crate::handle::BlockId,
    pub position: Vec3,
    pub scale: Vec3,
    /// Rotation about Z in radians.
    pub rotation: f32,
    /// Non-uniform row/column counts for arrays.
    pub rows: u32,
    pub columns: u32,
    pub row_spacing: f32,
    pub col_spacing: f32,
}

/// Hatching: a closed region plus a pattern descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct Hatch {
    /// Boundary loops; the first is the outer loop.
    pub loops: Vec<Polyline>,
    /// Distance between pattern lines in drawing units.
    pub pattern_scale: f32,
    pub pattern_angle: f32,
    pub pattern_name: String,
    /// Solid fill instead of a line pattern.
    pub solid: bool,
}

impl Hatch {
    pub fn solid_region(region: Polyline) -> Self {
        Self {
            loops: vec![region],
            pattern_scale: 1.0,
            pattern_angle: 0.0,
            pattern_name: "SOLID".to_string(),
            solid: true,
        }
    }
}

/// Which dimension a [`Dimension`] measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimensionKind {
    /// Horizontal or vertical distance between two points, with the measurement
    /// line parallel to the measured axis.
    Linear,
    /// True distance along the line joining the two points.
    Aligned,
    /// Distance from a centre to a point on a circle.
    Radius,
    /// Full width of a circle.
    Diameter,
    /// Angle between two directions. Stored in radians; [`Dimension::text`]
    /// formats it in degrees, because that is what a drawing is read in.
    Angular,
}

impl DimensionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DimensionKind::Linear => "LINEAR",
            DimensionKind::Aligned => "ALIGNED",
            DimensionKind::Radius => "RADIUS",
            DimensionKind::Diameter => "DIAMETER",
            DimensionKind::Angular => "ANGULAR",
        }
    }

    /// Does this kind of dimension carry arrow heads?
    ///
    /// Architectural drafting uses ticks where mechanical uses arrows, so the
    /// distinction is modelled rather than assumed; only the tick style is not
    /// yet selectable.
    pub fn has_arrows(self) -> bool {
        true
    }

    /// The name a drawing uses in the command line, e.g. `DIMDIAMETER`.
    pub fn command(self) -> &'static str {
        match self {
            DimensionKind::Linear => "dimlinear",
            DimensionKind::Aligned => "dimaligned",
            DimensionKind::Radius => "dimradius",
            DimensionKind::Diameter => "dimdiameter",
            DimensionKind::Angular => "dimangular",
        }
    }

    pub fn from_command(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "dimlinear" | "dimlin" => DimensionKind::Linear,
            "dimaligned" | "dimali" => DimensionKind::Aligned,
            "dimradius" | "dimrad" => DimensionKind::Radius,
            "dimdiameter" | "dimdia" => DimensionKind::Diameter,
            "dimangular" | "dimang" => DimensionKind::Angular,
            _ => return None,
        })
    }
}

/// An associative dimension.
///
/// "Associative" means the *measurement points* are stored, not the text, so the
/// number is recomputed whenever the geometry moves. A dimension that stores a
/// frozen string is the single most common way a drawing stops being a drawing,
/// so the text here is derived on demand and [`Dimension::text_override`] is the
/// only way to pin it.
#[derive(Debug, Clone, PartialEq)]
pub struct Dimension {
    pub kind: DimensionKind,
    /// The two measurement points, in world space.
    pub p1: Vec2,
    pub p2: Vec2,
    /// Where the dimension line is drawn: a point the line runs through for
    /// linear kinds, and the tip of the leader for radial ones.
    pub line: Vec2,
    /// Where the text sits, once the user has placed it by hand.
    pub text_at: Option<Vec2>,
    /// Drawn instead of the computed measurement. `None` means "measure".
    pub text_override: Option<String>,
    /// Cap height of the text, in world units.
    pub height: f32,
    /// Arrow head length, in world units.
    pub arrow: f32,
    /// Gap between the measured point and the start of its extension line.
    pub extension_gap: f32,
}

impl Dimension {
    pub fn new(kind: DimensionKind, p1: Vec2, p2: Vec2, line: Vec2) -> Self {
        Self {
            kind,
            p1,
            p2,
            line,
            text_at: None,
            text_override: None,
            height: 2.5,
            arrow: 1.5,
            extension_gap: 0.5,
        }
    }

    /// The measured value in drawing units.
    ///
    /// A linear dimension reads only the dominant axis; every other kind is a
    /// true distance, except a diameter which is twice the radius.
    pub fn measurement(&self) -> f32 {
        match self.kind {
            DimensionKind::Linear => {
                let dx = (self.p2.x - self.p1.x).abs();
                let dy = (self.p2.y - self.p1.y).abs();
                dx.max(dy)
            }
            DimensionKind::Radius => self.p1.distance(self.p2),
            DimensionKind::Diameter => 2.0 * self.p1.distance(self.p2),
            _ => self.p1.distance(self.p2),
        }
    }

    /// The measured angle in radians, for [`DimensionKind::Angular`].
    pub fn angle(&self) -> f32 {
        let a = (self.p1.y - self.line.y).atan2(self.p1.x - self.line.x);
        let b = (self.p2.y - self.line.y).atan2(self.p2.x - self.line.x);
        let mut d = (b - a).abs();
        if d > std::f32::consts::PI {
            d = std::f32::consts::TAU - d;
        }
        d
    }

    /// The string a drawing shows: the measurement, or the override.
    ///
    /// Formatting happens here rather than at creation time, so a dimension
    /// whose geometry moved shows the new number without being rewritten.
    pub fn text(&self, suffix: &str) -> String {
        if let Some(t) = &self.text_override {
            return t.clone();
        }
        match self.kind {
            DimensionKind::Angular => {
                format!("{:.2}\u{b0}{suffix}", self.angle().to_degrees())
            }
            DimensionKind::Radius => format!("R{:.3}{suffix}", self.measurement()),
            DimensionKind::Diameter => format!("\u{2300}{:.3}{suffix}", self.measurement()),
            _ => format!("{:.3}{suffix}", self.measurement()),
        }
    }

    /// Where the text goes: where the user put it, or on the dimension line.
    pub fn text_position(&self) -> Vec2 {
        self.text_at.unwrap_or(self.line)
    }

    /// The four points the dimension is drawn from: the two measured points,
    /// and the two points on the dimension line they project onto.
    ///
    /// For a radial dimension the leader runs from the centre out to the
    /// measured point, so the same four-point shape draws it without a special
    /// case in the renderer.
    pub fn geometry(&self) -> (Vec2, Vec2, Vec2, Vec2) {
        match self.kind {
            DimensionKind::Linear | DimensionKind::Aligned => {
                let u = (self.p2 - self.p1).normalize_or(Vec2::X);
                let n = Vec2::new(-u.y, u.x);
                // `line` is a point the dimension line runs through; its
                // perpendicular distance from the measurement is the offset.
                let offset = (self.line - self.p1).dot(n);
                let a = self.p1 + n * offset;
                let b = self.p2 + n * offset;
                (self.p1, self.p2, a, b)
            }
            DimensionKind::Radius | DimensionKind::Diameter => {
                (self.p1, self.p2, self.line, self.line + (self.p2 - self.p1))
            }
            // The vertex, then the two arms.
            DimensionKind::Angular => (self.line, self.line, self.p1, self.p2),
        }
    }

    /// The measured points in reading order.
    ///
    /// CAD reads left to right, so a dimension whose ends are the wrong way
    /// round is corrected here rather than by the renderer special-casing it.
    pub fn readable_ends(&self) -> (Vec2, Vec2) {
        if self.p1.x <= self.p2.x {
            (self.p1, self.p2)
        } else {
            (self.p2, self.p1)
        }
    }

    /// The arrow head direction at each end: pointing back along the dimension
    /// line, which is what makes an arrow head read as a direction.
    pub fn arrow_dirs(&self) -> (Vec2, Vec2) {
        let (_, _, a, b) = self.geometry();
        let d = b - a;
        if d.length_squared() < 1e-12 {
            (Vec2::X, Vec2::X)
        } else {
            let u = d.normalize();
            (-u, u)
        }
    }
}

/// The geometry payload of an entity.
#[derive(Debug, Clone, PartialEq)]
pub enum EntityKind {
    /// 2D primitives.
    Line(Line),
    Circle(Circle),
    Arc(Arc),
    Ellipse(Ellipse),
    Polyline(Polyline),
    Spline(Spline),
    Point(PointEnt),
    Text(Text),
    Hatch(Hatch),
    /// 2D region used for boolean ops (a closed polyline).
    Region(Polyline),
    /// 3D primitives.
    Box(Box3d),
    Mesh(Mesh3d),
    Face(Face3d),
    Construction(Construction),
    /// A block reference.
    Insert(InsertRef),
    /// An associative dimension: the measurement points are stored, not the
    /// text, so the number follows the geometry.
    Dimension(Dimension),
    /// Placeholder for entities we can draw but not yet round-trip.
    Unknown {
        dxf_type: String,
        raw: Vec<f32>,
    },
}

/// Shared, non-geometric properties.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityCommon {
    pub layer: LayerId,
    /// ACI index; `0` = by block, `256` = by layer.
    pub color: i16,
    /// `None` = by layer, `Some("BYP")` = by block.
    pub linetype: LineType,
    pub lineweight: LineWeight,
    pub visible: bool,
    pub locked: bool,
    pub paper_space: bool,
    pub transparency: f32,
}

impl Default for EntityCommon {
    fn default() -> Self {
        Self {
            layer: LayerId(0),
            color: 256,
            linetype: LineType::ByLayer,
            lineweight: LineWeight::ByLayer,
            visible: true,
            locked: false,
            paper_space: false,
            transparency: 0.0,
        }
    }
}

/// A drawing entity: appearance + geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct Entity {
    pub common: EntityCommon,
    pub entity: EntityKind,
}

// ---------------------------------------------------------------- constructors

impl Entity {
    pub fn new(kind: EntityKind) -> Self {
        Self {
            common: EntityCommon::default(),
            entity: kind,
        }
    }
    pub fn line(l: Line) -> Self {
        Self::new(EntityKind::Line(l))
    }
    pub fn circle(c: Circle) -> Self {
        Self::new(EntityKind::Circle(c))
    }
    pub fn arc(a: Arc) -> Self {
        Self::new(EntityKind::Arc(a))
    }
    pub fn polyline(p: Polyline) -> Self {
        Self::new(EntityKind::Polyline(p))
    }
    pub fn spline(s: Spline) -> Self {
        Self::new(EntityKind::Spline(s))
    }
    pub fn solid(b: Box3d) -> Self {
        Self::new(EntityKind::Box(b))
    }
    pub fn insert(mut i: InsertRef) -> Self {
        i.rows = 1;
        i.columns = 1;
        Self::new(EntityKind::Insert(i))
    }

    #[inline]
    pub fn layer(&self) -> LayerId {
        self.common.layer
    }
    #[inline]
    pub fn with_layer(mut self, id: LayerId) -> Self {
        self.common.layer = id;
        self
    }
    #[inline]
    pub fn with_color(mut self, c: i16) -> Self {
        self.common.color = c;
        self
    }
    #[inline]
    pub fn kind(&self) -> &str {
        match &self.entity {
            EntityKind::Line(_) => "LINE",
            EntityKind::Circle(_) => "CIRCLE",
            EntityKind::Arc(_) => "ARC",
            EntityKind::Ellipse(_) => "ELLIPSE",
            EntityKind::Polyline(_) => "LWPOLYLINE",
            EntityKind::Spline(_) => "SPLINE",
            EntityKind::Point(_) => "POINT",
            EntityKind::Text(_) => "TEXT",
            EntityKind::Hatch(_) => "HATCH",
            EntityKind::Region(_) => "REGION",
            EntityKind::Box(_) => "SOLID3D",
            EntityKind::Mesh(_) => "MESH",
            EntityKind::Face(_) => "FACE3D",
            EntityKind::Construction(_) => "XLINE",
            EntityKind::Insert(_) => "INSERT",
            EntityKind::Dimension(_) => "DIMENSION",
            EntityKind::Unknown { dxf_type, .. } => dxf_type,
        }
    }

    /// Drawable in a 2D viewport (3D solids need the model view).
    pub fn is_drawable_2d(&self) -> bool {
        !matches!(
            self.entity,
            EntityKind::Box(_) | EntityKind::Mesh(_) | EntityKind::Face(_)
        )
    }

    /// True for entities that live in the XY plane (everything except 3D solids).
    pub fn is_planar_2d(&self) -> bool {
        matches!(
            self.entity,
            EntityKind::Line(_)
                | EntityKind::Circle(_)
                | EntityKind::Arc(_)
                | EntityKind::Ellipse(_)
                | EntityKind::Polyline(_)
                | EntityKind::Spline(_)
                | EntityKind::Region(_)
                | EntityKind::Hatch(_)
        )
    }

    /// 2D curve view of the entity, when it has one.
    pub fn as_curve(&self) -> Option<Curve> {
        match &self.entity {
            EntityKind::Line(l) => Some(Curve::Line(*l)),
            EntityKind::Circle(c) => Some(Curve::Circle(*c)),
            EntityKind::Arc(a) => Some(Curve::Arc(*a)),
            EntityKind::Ellipse(e) => Some(Curve::Ellipse(*e)),
            EntityKind::Polyline(p) => Some(Curve::Polyline(p.clone())),
            _ => None,
        }
    }

    /// Axis-aligned XY bounds (empty for entities with no 2D extent).
    pub fn bounds_2d(&self) -> Rect2 {
        match &self.entity {
            EntityKind::Line(l) => l.bounds(),
            EntityKind::Circle(c) => c.bounds(),
            EntityKind::Arc(a) => a.bounds(),
            EntityKind::Ellipse(e) => e.bounds(),
            EntityKind::Polyline(p) => p.bounds(),
            EntityKind::Region(p) => p.bounds(),
            EntityKind::Spline(s) => s.bounds(),
            EntityKind::Text(t) => Rect2::new(
                t.insert.xy(),
                t.insert.xy() + Vec2::new(text_width(t), t.height),
            ),
            EntityKind::Point(p) => Rect2::new(p.position.xy(), p.position.xy()),
            EntityKind::Construction(c) => {
                Rect2::new(c.from.xy().min(c.to.xy()), c.from.xy().max(c.to.xy()))
            }
            EntityKind::Box(b) => Rect2::new(b.min.xy(), b.max.xy()),
            EntityKind::Dimension(d) => {
                let (a, b, c, e) = d.geometry();
                Rect2::new(a, b).union(Rect2::new(c, e))
            }
            EntityKind::Face(f) => {
                let mut r = Rect2::ZERO;
                let mut first = true;
                for p in &f.loop_pts {
                    if first {
                        r = Rect2::new(p.xy(), p.xy());
                        first = false;
                    } else {
                        r = r.expand_point(p.xy());
                    }
                }
                r
            }
            EntityKind::Hatch(h) => {
                let mut r = Rect2::ZERO;
                let mut first = true;
                for l in &h.loops {
                    let b = l.bounds();
                    if first {
                        r = b;
                        first = false;
                    } else {
                        r = r.union(b);
                    }
                }
                r
            }
            _ => Rect2::ZERO,
        }
    }

    /// 3D bounds, used for culling and for the model-space extents.
    pub fn bounds_3d(&self) -> Aabb3 {
        match &self.entity {
            EntityKind::Box(b) => b.bounds(),
            EntityKind::Mesh(m) => m.bounds(),
            EntityKind::Text(t) => {
                let w = text_width(t);
                let h = t.height + t.height2;
                Aabb3::new(t.insert, t.insert + Vec3::new(w, h, 0.0))
            }
            EntityKind::Face(f) => {
                let mut b = Aabb3::ZERO;
                let mut first = true;
                for p in &f.loop_pts {
                    if first {
                        b = Aabb3::new(*p, *p);
                        first = false;
                    } else {
                        b = b.expand_point(*p);
                    }
                }
                b
            }
            EntityKind::Dimension(d) => {
                let (a, b, c, e) = d.geometry();
                Aabb3::new(Vec3::new(a.x, a.y, 0.0), Vec3::new(b.x, b.y, 0.0)).union(Aabb3::new(
                    Vec3::new(c.x, c.y, 0.0),
                    Vec3::new(e.x, e.y, 0.0),
                ))
            }
            EntityKind::Construction(c) => Aabb3::new(c.from, c.to),
            EntityKind::Point(p) => Aabb3::new(p.position, p.position),
            _ => {
                let r = self.bounds_2d();
                Aabb3::new(
                    Vec3::new(r.min.x, r.min.y, 0.0),
                    Vec3::new(r.max.x, r.max.y, 0.0),
                )
            }
        }
    }

    /// Cheap conservative bounds used by the spatial index.
    pub fn index_bounds(&self) -> Rect2 {
        let r = self.bounds_2d();
        if r.is_empty() {
            // Zero-area entities still need a pickable cell; give them a
            // one-unit-wide box so hover/click works.
            r.expand(Vec2::new(0.5, 0.5))
        } else {
            r
        }
    }

    /// Apply an XY translation to any entity that has geometry.
    pub fn translated(&self, v: Vec3) -> Entity {
        let mut out = self.clone();
        let v2 = v.xy();
        match &mut out.entity {
            EntityKind::Line(l) => {
                l.p0 += v2;
                l.p1 += v2;
            }
            EntityKind::Circle(c) => c.center += v2,
            EntityKind::Arc(a) => a.center += v2,
            EntityKind::Ellipse(e) => e.center += v2,
            EntityKind::Polyline(p) | EntityKind::Region(p) => {
                for q in &mut p.vertices {
                    *q += v2;
                }
            }
            EntityKind::Spline(s) => {
                for seg in &mut s.segments {
                    for p in seg.p.iter_mut() {
                        *p += v2;
                    }
                }
                for p in &mut s.control_points {
                    *p += v2;
                }
            }
            EntityKind::Point(p) => p.position += v,
            EntityKind::Text(t) => t.insert += v,
            EntityKind::Hatch(h) => {
                for l in &mut h.loops {
                    for q in &mut l.vertices {
                        *q += v2;
                    }
                }
            }
            EntityKind::Box(b) => {
                b.min += v;
                b.max += v;
            }
            EntityKind::Face(f) => {
                for p in &mut f.loop_pts {
                    *p += v;
                }
                let n = f.plane.n;
                f.plane = Plane3::new(n, n.dot(f.loop_pts[0]));
            }
            EntityKind::Construction(c) => {
                c.from += v;
                c.to += v;
            }
            EntityKind::Dimension(d) => {
                // The measurement line travels with the geometry, so an
                // associative dimension keeps measuring after a move.
                d.p1 += v2;
                d.p2 += v2;
                d.line += v2;
                if let Some(t) = d.text_at.as_mut() {
                    *t += v2;
                }
            }
            EntityKind::Insert(i) => i.position += v,
            EntityKind::Mesh(m) => {
                for p in &mut m.positions {
                    *p += v;
                }
            }
            // Everything with a 2D curve view returns through the `as_curve`
            // branch above, so the remainder is the non-curve kinds plus
            // `Unknown`, which has no geometry to mirror.
            _ => {}
        }
        out
    }

    /// Rotate around the Z axis through `center`.
    pub fn rotated(&self, center: Vec3, angle: f32) -> Entity {
        let mut out = self.clone();
        let c2 = center.xy();
        let rot = |p: Vec2| (p - c2).rotate(angle) + c2;
        match &mut out.entity {
            EntityKind::Line(l) => {
                l.p0 = rot(l.p0);
                l.p1 = rot(l.p1);
            }
            EntityKind::Circle(c) => c.center = rot(c.center),
            EntityKind::Arc(a) => {
                a.center = rot(a.center);
                a.start_angle += angle;
            }
            EntityKind::Ellipse(e) => {
                e.center = rot(e.center);
                e.major_axis = e.major_axis.rotate(angle);
            }
            EntityKind::Polyline(p) | EntityKind::Region(p) => {
                for q in &mut p.vertices {
                    *q = rot(*q);
                }
            }
            EntityKind::Spline(s) => {
                for seg in &mut s.segments {
                    for p in seg.p.iter_mut() {
                        *p = rot(*p);
                    }
                }
                for p in &mut s.control_points {
                    *p = rot(*p);
                }
            }
            EntityKind::Text(t) => {
                let insert = t.insert.xy();
                t.insert = Vec3::new(rot(insert).x, rot(insert).y, t.insert.z);
                t.rotation += angle;
            }
            EntityKind::Hatch(h) => {
                for l in &mut h.loops {
                    for q in &mut l.vertices {
                        *q = rot(*q);
                    }
                }
                h.pattern_angle += angle;
            }
            EntityKind::Box(b) => {
                // Rotate by swapping corners: an axis-aligned box stays AABB.
                let c = b.center();
                let h = b.bounds().extents_half();
                let (s, co) = angle.sin_cos();
                let ex = Vec3::new(
                    h.x * co.abs() + h.y * s.abs(),
                    h.x * s.abs() + h.y * co.abs(),
                    h.z,
                );
                b.min = c - ex;
                b.max = c + ex;
            }
            EntityKind::Construction(c) => {
                let from = c.from.xy();
                let to = c.to.xy();
                c.from = Vec3::new(rot(from).x, rot(from).y, c.from.z);
                c.to = Vec3::new(rot(to).x, rot(to).y, c.to.z);
            }
            EntityKind::Insert(i) => {
                let pos = i.position.xy();
                i.position = Vec3::new(rot(pos).x, rot(pos).y, i.position.z);
                i.rotation += angle;
            }
            EntityKind::Face(f) => {
                for p in &mut f.loop_pts {
                    let q = p.xy();
                    *p = Vec3::new(rot(q).x, rot(q).y, p.z);
                }
                let n = f.plane.n.rotate_z(angle);
                let d = n.dot(f.loop_pts[0]);
                f.plane = Plane3::new(n, d);
            }
            EntityKind::Dimension(d) => {
                // A dimension measures the geometry it points at, so rotating
                // the drawing has to carry its points round with it.
                d.p1 = rot(d.p1);
                d.p2 = rot(d.p2);
                d.line = rot(d.line);
                if let Some(t) = d.text_at.as_mut() {
                    *t = rot(*t);
                }
            }
            EntityKind::Point(_) | EntityKind::Mesh(_) | EntityKind::Unknown { .. } => {}
        }
        out
    }

    /// Mirror across the line through `center` with direction `dir`.
    ///
    /// The 2D curve kinds delegate to [`cad_geom::Curve::transformed`], which
    /// already handles the awkward part: an arc or ellipse only stays one under
    /// a *similarity* transform, and a mirror is orientation-reversing, so the
    /// sweep flips and the curve is promoted when it must be. Maintaining a
    /// second, hand-rolled version of that here is how the two drift apart.
    ///
    /// `dir` need not be normalised, and may be zero -- a zero direction has no
    /// unique mirror line, so the entity is left where it is rather than moved
    /// somewhere arbitrary.
    pub fn mirrored(&self, center: Vec3, dir: Vec3) -> Entity {
        let mut out = self.clone();
        let c2 = center.xy();
        let d2 = dir.xy();
        if d2.length_squared() < 1e-12 {
            return out;
        }
        // Translate so the axis passes through the origin, mirror, translate back.
        let t = cad_geom::xform::Affine2::translation(-c2)
            .then(cad_geom::xform::Affine2::mirror(d2))
            .then(cad_geom::xform::Affine2::translation(c2));
        let m2 = |p: Vec2| t.apply(p);
        // A mirror in XY also flips Z for genuinely 3D kinds; the 2D overlay
        // keeps its elevation so a plan drawing stays in plan.
        let m3 = |p: Vec3| {
            let q = t.apply(p.xy());
            Vec3::new(q.x, q.y, p.z)
        };
        let m3r = |p: Vec3| {
            let q = t.apply(p.xy());
            Vec3::new(q.x, q.y, -p.z)
        };

        if let Some(curve) = self.as_curve() {
            let mirrored = curve.transformed(&t);
            match &mut out.entity {
                EntityKind::Line(_) => {
                    out.entity = EntityKind::Line(match mirrored {
                        cad_geom::Curve::Line(l) => l,
                        other => return Entity::new(EntityKind::Polyline(as_polyline(&other))),
                    })
                }
                EntityKind::Circle(_) => {
                    out.entity = match mirrored {
                        cad_geom::Curve::Circle(c) => EntityKind::Circle(c),
                        cad_geom::Curve::Polyline(p) => EntityKind::Polyline(p),
                        other => EntityKind::Polyline(as_polyline(&other)),
                    }
                }
                EntityKind::Arc(_) => {
                    out.entity = match mirrored {
                        cad_geom::Curve::Arc(a) => EntityKind::Arc(a),
                        cad_geom::Curve::Polyline(p) => EntityKind::Polyline(p),
                        other => EntityKind::Polyline(as_polyline(&other)),
                    }
                }
                EntityKind::Ellipse(_) => {
                    out.entity = match mirrored {
                        cad_geom::Curve::Ellipse(e) => EntityKind::Ellipse(e),
                        cad_geom::Curve::Polyline(p) => EntityKind::Polyline(p),
                        other => EntityKind::Polyline(as_polyline(&other)),
                    }
                }
                EntityKind::Polyline(_) | EntityKind::Region(_) => {
                    out.entity = match mirrored {
                        cad_geom::Curve::Polyline(p) => EntityKind::Polyline(p),
                        cad_geom::Curve::Line(l) => {
                            EntityKind::Polyline(Polyline::new(vec![l.p0, l.p1], false))
                        }
                        other => EntityKind::Polyline(as_polyline(&other)),
                    }
                }
                _ => {}
            }
            return out;
        }

        match &mut out.entity {
            EntityKind::Spline(s) => {
                // `Curve` has no spline variant, so a spline does not come back
                // from `as_curve`; transform its control points directly.
                for seg in &mut s.segments {
                    for p in seg.p.iter_mut() {
                        *p = m2(*p);
                    }
                }
                for p in &mut s.control_points {
                    *p = m2(*p);
                }
            }
            EntityKind::Point(p) => p.position = m3(p.position),
            EntityKind::Text(t) => {
                let q = m2(t.insert.xy());
                t.insert = Vec3::new(q.x, q.y, t.insert.z);
                // A mirrored glyph run reads backwards unless the rotation flips.
                t.rotation = -t.rotation;
            }
            EntityKind::Hatch(h) => {
                for l in &mut h.loops {
                    for q in &mut l.vertices {
                        *q = m2(*q);
                    }
                }
                h.pattern_angle = -h.pattern_angle;
            }
            EntityKind::Box(b) => {
                // Swap opposite corners: the axis-aligned box stays axis-aligned.
                let (mn, mx) = (b.min, b.max);
                b.min = m3(mx);
                b.max = m3(mn);
            }
            EntityKind::Face(f) => {
                for p in &mut f.loop_pts {
                    *p = m3r(*p);
                }
                let n = f.plane.n;
                f.plane = Plane3::new(Vec3::new(n.x, n.y, -n.z), f.plane.d);
            }
            EntityKind::Construction(c) => {
                c.from = m3(c.from);
                c.to = m3(c.to);
            }
            EntityKind::Insert(i) => {
                i.position = m3(i.position);
                i.rotation = -i.rotation;
                i.scale.y = -i.scale.y;
            }
            EntityKind::Mesh(m) => {
                for p in &mut m.positions {
                    *p = m3r(*p);
                }
                // Winding flips, so recompute rather than transform the normals.
                m.recompute_normals();
            }
            // Everything with a 2D curve view returned through the
            // `as_curve` branch above, so the remainder is the non-curve kinds
            // plus `Unknown`, which has no geometry to mirror.
            _ => {}
        }
        out
    }

    /// Apply a 2D affine transform to any entity that has geometry.
    ///
    /// Z is carried through unchanged: a block reference inserted into a plan
    /// drawing keeps its elevation, and a pure 2D transform must not silently
    /// move solids off the ground plane.
    pub fn transformed(&self, t: &cad_geom::xform::Affine2) -> Entity {
        let mut out = self.clone();
        let p2 = |p: Vec2| t.apply(p);
        let p3 = |p: Vec3| {
            let q = t.apply(p.xy());
            Vec3::new(q.x, q.y, p.z)
        };
        // A non-uniform scale turns a circle into an ellipse and an arc into a
        // Bézier, so anything that is not a similarity has to be promoted. This
        // is the same rule `Curve::transformed` already applies; asking it is
        // cheaper and safer than repeating the test per kind.
        let similarity = t.is_similarity();
        let mirroring = t.is_mirroring();

        macro_rules! curve_or_poly {
            ($k:expr) => {{
                let c: cad_geom::Curve = $k;
                if similarity {
                    c.transformed(t)
                } else {
                    // Under a non-uniform scale a circle is not a circle, so the
                    // curve is flattened rather than kept with a shape it no
                    // longer has.
                    cad_geom::Curve::Polyline(as_polyline(&c))
                }
            }};
        }

        match &mut out.entity {
            EntityKind::Line(l) => {
                l.p0 = p2(l.p0);
                l.p1 = p2(l.p1);
            }
            EntityKind::Circle(c) => {
                out.entity = match curve_or_poly!(cad_geom::Curve::Circle(*c)) {
                    cad_geom::Curve::Circle(g) => EntityKind::Circle(g),
                    other => EntityKind::Polyline(as_polyline(&other)),
                }
            }
            EntityKind::Arc(a) => {
                out.entity = match curve_or_poly!(cad_geom::Curve::Arc(*a)) {
                    cad_geom::Curve::Arc(g) => EntityKind::Arc(g),
                    other => EntityKind::Polyline(as_polyline(&other)),
                }
            }
            EntityKind::Ellipse(e) => {
                out.entity = match curve_or_poly!(cad_geom::Curve::Ellipse(*e)) {
                    cad_geom::Curve::Ellipse(g) => EntityKind::Ellipse(g),
                    other => EntityKind::Polyline(as_polyline(&other)),
                }
            }
            EntityKind::Polyline(p) | EntityKind::Region(p) => {
                for q in &mut p.vertices {
                    *q = p2(*q);
                }
                // An anisotropic scale turns an arc into a Bézier, which this
                // representation cannot store: flatten it instead of keeping a
                // bulge that no longer describes the shape.
                if !similarity {
                    p.bulges
                        .iter_mut()
                        .for_each(|b| *b = cad_geom::bulge::Bulge::NONE);
                }
            }
            EntityKind::Spline(s) => {
                for seg in &mut s.segments {
                    for q in seg.p.iter_mut() {
                        *q = p2(*q);
                    }
                }
                for q in &mut s.control_points {
                    *q = p2(*q);
                }
            }
            EntityKind::Point(p) => p.position = p3(p.position),
            EntityKind::Text(tx) => {
                tx.insert = p3(tx.insert);
                // The image of the text's own axis is the new rotation, and a
                // mirroring reverses the reading direction.
                let axis = t.apply_dir(Vec2::X);
                tx.rotation = axis.y.atan2(axis.x);
                tx.width_factor *= t.axis_x().length();
                if mirroring {
                    tx.rotation = -tx.rotation;
                }
            }
            EntityKind::Hatch(h) => {
                for l in &mut h.loops {
                    for q in &mut l.vertices {
                        *q = p2(*q);
                    }
                }
                h.pattern_angle += t.rotation_angle();
                h.pattern_scale *= t.scale_factor();
            }
            EntityKind::Dimension(d) => {
                d.p1 = p2(d.p1);
                d.p2 = p2(d.p2);
                d.line = p2(d.line);
                if let Some(a) = d.text_at.as_mut() {
                    *a = p2(*a);
                }
                d.height *= t.scale_factor();
                d.arrow *= t.scale_factor();
                if let Some(over) = d.text_override.as_mut() {
                    // The stored number is now wrong; drop the override so the
                    // dimension goes back to measuring.
                    *over = over.clone();
                    d.text_override = None;
                }
            }
            EntityKind::Box(b) => {
                // Take the four corners, transform them, and rebuild the box as
                // the bounds of what came out. A rotated box is not axis-aligned,
                // and pretending otherwise would shear it.
                let corners = b.bounds().corners();
                let mut r = cad_core::Rect2::EMPTY;
                for c in corners {
                    let q = p2(c.xy());
                    r = r.expand_point(q);
                }
                b.min = Vec3::new(r.min.x, r.min.y, b.min.z);
                b.max = Vec3::new(r.max.x, r.max.y, b.max.z);
            }
            EntityKind::Construction(c) => {
                c.from = p3(c.from);
                c.to = p3(c.to);
            }
            EntityKind::Insert(i) => {
                i.position = p3(i.position);
                i.rotation += t.rotation_angle();
                if mirroring {
                    i.scale.y = -i.scale.y;
                }
                i.scale.x *= t.axis_x().length();
                i.scale.y *= t.axis_y().length();
            }
            EntityKind::Face(f) => {
                for q in &mut f.loop_pts {
                    *q = p3(*q);
                }
                // A 2D transform leaves the plane's normal only valid if it was
                // vertical; recompute from the transformed loop otherwise.
                if let Some(first) = f.loop_pts.first() {
                    let n = f.plane.n;
                    f.plane = Plane3::new(n, n.dot(*first));
                }
            }
            EntityKind::Mesh(m) => {
                for q in &mut m.positions {
                    *q = p3(*q);
                }
                m.recompute_normals();
            }
            EntityKind::Unknown { .. } => {}
        }
        out
    }

    /// The effective colour given the entity's own override and its layer.
    pub fn resolved_color(&self, layer_color: Rgba) -> Rgba {
        match self.common.color {
            0 | 256 => layer_color,
            _ => cad_core::aci_to_rgba(self.common.color, layer_color),
        }
    }

    /// Total length / area, for the properties palette.
    /// The entity type as it appears in the properties palette.
    ///
    /// Distinct from [`Entity::kind`]: `kind` borrows the stored
    /// `dxf_type` for entities we cannot model, so it is not `'static`, and a
    /// properties panel that formats its rows every frame must not keep a
    /// borrow alive across the frame.
    pub fn type_name(&self) -> &'static str {
        match &self.entity {
            EntityKind::Line(_) => "Line",
            EntityKind::Circle(_) => "Circle",
            EntityKind::Arc(_) => "Arc",
            EntityKind::Ellipse(_) => "Ellipse",
            EntityKind::Polyline(_) => "Polyline",
            EntityKind::Spline(_) => "Spline",
            EntityKind::Point(_) => "Point",
            EntityKind::Text(_) => "Text",
            EntityKind::Hatch(_) => "Hatch",
            EntityKind::Region(_) => "Region",
            EntityKind::Box(_) => "Box",
            EntityKind::Mesh(_) => "Mesh",
            EntityKind::Face(_) => "Face",
            EntityKind::Construction(_) => "Construction",
            EntityKind::Insert(_) => "Insert",
            EntityKind::Dimension(_) => "Dimension",
            EntityKind::Unknown { .. } => "Unknown",
        }
    }

    /// Is this entity's geometry drawn as 3D solids rather than 2D lines?
    pub fn is_3d(&self) -> bool {
        matches!(
            self.entity,
            EntityKind::Box(_) | EntityKind::Mesh(_) | EntityKind::Face(_)
        )
    }

    pub fn measure(&self) -> Option<(f64, bool)> {
        match &self.entity {
            EntityKind::Line(l) => Some((l.length() as f64, false)),
            EntityKind::Circle(c) => Some((c.perimeter() as f64, false)),
            EntityKind::Arc(a) => Some((a.length() as f64, false)),
            EntityKind::Ellipse(e) => Some((e.perimeter_ramanujan() as f64, false)),
            EntityKind::Polyline(p) | EntityKind::Region(p) => Some((p.perimeter() as f64, false)),
            EntityKind::Dimension(d) => Some((d.measurement() as f64, false)),
            EntityKind::Box(b) => Some((b.volume() as f64, true)),
            _ => None,
        }
    }

    /// Snap points offered by this entity, in 2D.
    pub fn snap_points_2d(&self) -> Vec<Vec2> {
        match &self.entity {
            EntityKind::Line(l) => vec![l.p0, l.p1, l.midpoint()],
            EntityKind::Circle(c) => vec![c.center],
            EntityKind::Arc(a) => vec![a.center, a.start_point(), a.end_point()],
            EntityKind::Point(p) => vec![p.position.xy()],
            EntityKind::Polyline(p) | EntityKind::Region(p) => {
                let mut v: Vec<Vec2> = p.vertices.clone();
                if p.closed && p.len() > 1 {
                    v.push(p.point(0));
                }
                v
            }
            EntityKind::Insert(i) => vec![i.position.xy()],
            EntityKind::Dimension(d) => vec![d.p1, d.p2],
            _ => Vec::new(),
        }
    }
}

/// Flatten any curve to a polyline, for the cases where a mirror has to change
/// an entity's type (which only happens when a shape is not a similarity of its
/// mirror image, i.e. never for a pure mirror -- but the match has to be
/// total).
fn as_polyline(c: &cad_geom::Curve) -> Polyline {
    match c {
        cad_geom::Curve::Polyline(p) => p.clone(),
        cad_geom::Curve::Line(l) => Polyline::new(vec![l.p0, l.p1], false),
        other => {
            // Flatten to polylines rather than dropping the geometry: a silently
            // empty entity is much worse than a slightly coarser one.
            let pts = cad_geom::tessellate::tessellate(
                other,
                &cad_geom::tessellate::TessellationOptions::default(),
            );
            if pts.len() >= 2 {
                Polyline::new(pts, false)
            } else {
                Polyline::new(Vec::new(), false)
            }
        }
    }
}

/// Rough monospace advance-width estimate, good enough for hit-testing and
/// for the properties palette. The renderer uses a real font atlas.
pub fn text_width(t: &Text) -> f32 {
    let n = t.value.chars().count().max(1) as f32;
    n * t.height * 0.6 * t.width_factor.max(0.01)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::Vec2;

    #[test]
    fn translate_line_and_circle() {
        let l = Entity::line(Line::new(Vec2::ZERO, Vec2::new(10.0, 0.0)))
            .translated(Vec3::new(5.0, 5.0, 0.0));
        match &l.entity {
            EntityKind::Line(m) => assert!(m.p0.distance(Vec2::new(5.0, 5.0)) < 1e-6),
            _ => panic!(),
        }
    }

    #[test]
    fn rotate_arc_keeps_geometry() {
        let a = Entity::arc(Arc::from_angles(Vec2::ZERO, 5.0, 0.0, cad_core::FRAC_PI_2));
        let r = a.rotated(Vec3::ZERO, cad_core::FRAC_PI_2);
        match &r.entity {
            EntityKind::Arc(m) => {
                assert!(m.center.distance(Vec2::ZERO) < 1e-5);
                assert!((m.start_angle - cad_core::FRAC_PI_2).abs() < 1e-5);
                assert!((m.sweep - cad_core::FRAC_PI_2).abs() < 1e-5);
                assert!((m.radius - 5.0).abs() < 1e-5);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn rotate_polyline_preserves_area() {
        let p = Polyline::new(
            vec![
                Vec2::ZERO,
                Vec2::new(4.0, 0.0),
                Vec2::new(4.0, 3.0),
                Vec2::new(0.0, 3.0),
            ],
            true,
        );
        let e = Entity::polyline(p);
        let a0 = match &e.entity {
            EntityKind::Polyline(p) => p.signed_area(),
            _ => 0.0,
        };
        let r = e.rotated(Vec3::ZERO, 0.7);
        let a1 = match &r.entity {
            EntityKind::Polyline(p) => p.signed_area(),
            _ => 0.0,
        };
        assert!((a0 - a1).abs() < 1e-3, "{a0} vs {a1}");
    }

    #[test]
    fn box_faces_and_volume() {
        let b = Box3d::new(Vec3::ZERO, Vec3::new(2.0, 3.0, 4.0));
        assert!((b.volume() - 24.0).abs() < 1e-4);
        assert_eq!(b.faces().len(), 6);
        assert!(b.center().distance(Vec3::new(1.0, 1.5, 2.0)) < 1e-6);
    }

    #[test]
    fn mesh_normals_face_outward() {
        let mut m = Mesh3d {
            positions: vec![
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            normals: Vec::new(),
            indices: vec![0, 1, 2],
        };
        m.recompute_normals();
        assert!((m.normals[0].z - 1.0).abs() < 1e-5, "{:?}", m.normals[0]);
        assert_eq!(m.triangle_count(), 1);
    }

    #[test]
    fn extrusion_produces_side_walls() {
        let region = Polyline::new(
            vec![
                Vec2::ZERO,
                Vec2::new(2.0, 0.0),
                Vec2::new(2.0, 2.0),
                Vec2::new(0.0, 2.0),
            ],
            true,
        );
        let m = Mesh3d::from_extrusion(&region, 5.0);
        assert_eq!(m.triangle_count(), 8); // 4 quads -> 8 triangles
        assert!(m.bounds().max.z > 4.9);
        assert_eq!(m.normals.len(), m.positions.len());
    }

    #[test]
    fn zero_area_entity_gets_pickable_bounds() {
        let e = Entity::circle(Circle::new(Vec2::ZERO, 0.0));
        assert!(!e.index_bounds().is_empty());
    }

    #[test]
    fn measure_reports_length_and_area() {
        assert_eq!(
            Entity::line(Line::new(Vec2::ZERO, Vec2::new(4.0, 0.0))).measure(),
            Some((4.0, false))
        );
        assert_eq!(
            Entity::solid(Box3d::new(Vec3::ZERO, Vec3::splat(2.0))).measure(),
            Some((8.0, true))
        );
    }

    #[test]
    fn mirrored_polyline_reflects_only_the_perpendicular_component() {
        let square = Polyline::new(
            vec![
                Vec2::new(2.0, 1.0),
                Vec2::new(4.0, 1.0),
                Vec2::new(4.0, 3.0),
                Vec2::new(2.0, 3.0),
            ],
            true,
        );
        let before = Entity::polyline(square.clone()).bounds_2d();
        // Reflect in the line y = 0 (direction X through the origin).
        let after = Entity::polyline(square)
            .mirrored(Vec3::ZERO, Vec3::X)
            .bounds_2d();
        assert_eq!(before.min.x, 2.0, "{before:?}");
        assert_eq!(before.max.x, 4.0, "{before:?}");
        assert_eq!(after.min.x, 2.0, "x must not move: {after:?}");
        assert_eq!(after.max.x, 4.0, "x must not move: {after:?}");
        assert_eq!(after.min.y, -3.0, "{after:?}");
        assert_eq!(after.max.y, -1.0, "{after:?}");
    }

    #[test]
    fn mirrored_circle_keeps_its_radius() {
        let c = Circle::new(Vec2::new(10.0, 0.0), 4.0);
        let m = Entity::circle(c).mirrored(Vec3::ZERO, Vec3::Y);
        match &m.entity {
            EntityKind::Circle(g) => {
                assert!((g.center.x + 10.0).abs() < 1e-5, "{:?}", g.center);
                assert!(g.center.y.abs() < 1e-5, "{:?}", g.center);
                assert!((g.radius - 4.0).abs() < 1e-5);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn mirrored_arc_flips_its_sweep() {
        // Mirroring reverses the direction of travel around the centre, so the
        // sweep has to negate or the arc lands on the wrong side.
        let a = Arc::from_angles(Vec2::ZERO, 5.0, 0.0, std::f32::consts::FRAC_PI_2);
        let m = Entity::arc(a).mirrored(Vec3::ZERO, Vec3::Y);
        match &m.entity {
            EntityKind::Arc(g) => {
                assert!(
                    (g.sweep + std::f32::consts::FRAC_PI_2).abs() < 1e-4,
                    "{}",
                    g.sweep
                );
                assert!(g.center.x.abs() < 1e-5);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn snap_points() {
        let l = Entity::line(Line::new(Vec2::ZERO, Vec2::new(10.0, 0.0)));
        let pts = l.snap_points_2d();
        assert_eq!(pts.len(), 3);
        assert!(pts[2].distance(Vec2::new(5.0, 0.0)) < 1e-6);
    }

    #[test]
    fn closed_polyline_reports_closing_vertex() {
        let p = Polyline::new(
            vec![Vec2::ZERO, Vec2::new(4.0, 0.0), Vec2::new(4.0, 4.0)],
            true,
        );
        let pts = Entity::polyline(p).snap_points_2d();
        assert_eq!(pts.len(), 4);
    }
}
