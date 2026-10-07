//! `cad-snap` — the object-snap and tracking engine.
//!
//! Snaps are resolved in a fixed priority order, exactly like a real CAD: the
//! cursor position is converted into **screen pixels**, every candidate is
//! projected to pixels, and the closest one within the aperture wins. That
//! makes the behaviour identical at every zoom level, which is the whole point.

use cad_core::{Camera2D, Rect2, Vec2, Vec3, wrap_pi};
use cad_doc::{Document, Entity, EntityId, EntityKind};
use cad_geom::curve::{Arc, Circle, Curve, Line, Polyline};
use cad_geom::intersect::HitKind;
use cad_geom::tessellate::{TessellationOptions, tessellate};

/// Individual snap modes. The bitset is a plain `u32` so no dependency on a
/// bitflags crate is needed (we are chasing bytes).
///
/// **Bit order is priority order**: lower bit == stronger snap. The sequence
/// mirrors AutoCAD's osnap ordering (endpoint, intersection, apparent
/// intersection, midpoint, center, quadrant, extension, insert, perpendicular,
/// tangent, nearest, node), because that ordering is what users already have in
/// their fingers.
pub mod mode {
    pub const ENDPOINT: u32 = 1 << 0;
    pub const INTERSECTION: u32 = 1 << 1;
    pub const APP_INT: u32 = 1 << 2;
    pub const MIDPOINT: u32 = 1 << 3;
    pub const CENTER: u32 = 1 << 4;
    pub const QUADRANT: u32 = 1 << 5;
    pub const EXTENSION: u32 = 1 << 6;
    pub const INSERT: u32 = 1 << 7;
    pub const PERPENDICULAR: u32 = 1 << 8;
    pub const TANGENT: u32 = 1 << 9;
    pub const NEAREST: u32 = 1 << 10;
    pub const NODE: u32 = 1 << 11;
    pub const CLOSEST: u32 = 1 << 12;
    /// Polar / ortho tracking (not point snaps, but same mechanism).
    pub const ORTHO: u32 = 1 << 13;
    pub const POLAR: u32 = 1 << 14;
    pub const GRID: u32 = 1 << 15;
    /// Everything AutoCAD enables by default.
    pub const DEFAULT: u32 = ENDPOINT
        | MIDPOINT
        | CENTER
        | QUADRANT
        | INTERSECTION
        | PERPENDICULAR
        | TANGENT
        | NEAREST
        | NODE
        | ORTHO
        | POLAR;
    /// Cheap modes for large drawings (no intersection or curve queries).
    pub const FAST: u32 = ENDPOINT | MIDPOINT | CENTER | QUADRANT | ORTHO | POLAR;
}

/// Which snap produced a point — the renderer draws a different glyph for each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapKind {
    Endpoint,
    Midpoint,
    Center,
    Quadrant,
    Intersection,
    Perpendicular,
    Tangent,
    Nearest,
    Node,
    Insert,
    ApparentIntersection,
    Closest,
    Extension,
    Ortho,
    Polar,
    Grid,
}

impl SnapKind {
    /// Single-character marker used by the command line, like AutoCAD's.
    pub fn marker(self) -> &'static str {
        match self {
            SnapKind::Endpoint => "END",
            SnapKind::Midpoint => "MID",
            SnapKind::Center => "CEN",
            SnapKind::Quadrant => "QUA",
            SnapKind::Intersection => "INT",
            SnapKind::Perpendicular => "PER",
            SnapKind::Tangent => "TAN",
            SnapKind::Nearest => "NEA",
            SnapKind::Node => "NOD",
            SnapKind::Insert => "INS",
            SnapKind::ApparentIntersection => "APP",
            SnapKind::Closest => "CLO",
            SnapKind::Extension => "EXT",
            SnapKind::Ortho => "ORG",
            SnapKind::Polar => "POL",
            SnapKind::Grid => "GRD",
        }
    }

    /// Glyph id in the renderer's marker atlas.
    pub fn glyph(self) -> u32 {
        match self {
            SnapKind::Endpoint => 0,
            SnapKind::Midpoint => 1,
            SnapKind::Center => 2,
            SnapKind::Quadrant => 3,
            SnapKind::Intersection => 4,
            SnapKind::Perpendicular => 5,
            SnapKind::Tangent => 6,
            SnapKind::Nearest => 7,
            SnapKind::Node => 8,
            SnapKind::Insert => 9,
            SnapKind::ApparentIntersection => 10,
            SnapKind::Closest => 11,
            SnapKind::Extension => 12,
            SnapKind::Ortho => 13,
            SnapKind::Polar => 14,
            SnapKind::Grid => 15,
        }
    }

    fn bit(self) -> u32 {
        match self {
            SnapKind::Endpoint => mode::ENDPOINT,
            SnapKind::Midpoint => mode::MIDPOINT,
            SnapKind::Center => mode::CENTER,
            SnapKind::Quadrant => mode::QUADRANT,
            SnapKind::Intersection => mode::INTERSECTION,
            SnapKind::Perpendicular => mode::PERPENDICULAR,
            SnapKind::Tangent => mode::TANGENT,
            SnapKind::Nearest => mode::NEAREST,
            SnapKind::Node => mode::NODE,
            SnapKind::Insert => mode::INSERT,
            SnapKind::ApparentIntersection => mode::APP_INT,
            SnapKind::Closest => mode::CLOSEST,
            SnapKind::Extension => mode::EXTENSION,
            SnapKind::Ortho => mode::ORTHO,
            SnapKind::Polar => mode::POLAR,
            SnapKind::Grid => mode::GRID,
        }
    }
}

/// A resolved snap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Snap {
    pub point: Vec2,
    pub kind: SnapKind,
    /// The entity the snap came from, when it came from one.
    pub source: Option<EntityId>,
    /// For perpendicular/tangent: the entity we are perpendicular to.
    pub reference: Option<EntityId>,
}

impl Snap {
    fn new(point: Vec2, kind: SnapKind) -> Self {
        Self {
            point,
            kind,
            source: None,
            reference: None,
        }
    }
}

/// User-tunable snap behaviour.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapSettings {
    pub enabled: bool,
    pub modes: u32,
    /// Pick radius in **pixels**.
    pub aperture_px: f32,
    /// Marker size in pixels.
    pub marker_px: f32,
    /// Grid spacing in world units (0 disables the grid snap).
    pub grid_spacing: f32,
    /// Resolution increment for polar tracking, in degrees.
    pub polar_increment_deg: f32,
    /// Ortho tracking angles to hold (multiples of 90 by default).
    pub ortho_angles: Vec<f32>,
    /// Show the dynamic measurement while tracking.
    pub show_tracking: bool,
    /// World-space flatness used when tessellating for intersections.
    pub curve_tolerance: f32,
}

impl Default for SnapSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            modes: mode::DEFAULT,
            aperture_px: 12.0,
            marker_px: 9.0,
            grid_spacing: 0.0,
            polar_increment_deg: 0.0,
            ortho_angles: vec![
                0.0,
                std::f32::consts::FRAC_PI_2,
                std::f32::consts::PI,
                -std::f32::consts::FRAC_PI_2,
            ],
            show_tracking: true,
            curve_tolerance: 0.01,
        }
    }
}

impl SnapSettings {
    pub fn has(&self, k: SnapKind) -> bool {
        self.modes & k.bit() != 0
    }
    pub fn set(&mut self, k: SnapKind, on: bool) {
        if on {
            self.modes |= k.bit();
        } else {
            self.modes &= !k.bit();
        }
    }
    pub fn toggle(&mut self, k: SnapKind) {
        let on = self.has(k);
        self.set(k, !on);
    }
    fn tess(&self) -> TessellationOptions {
        TessellationOptions::with_tolerance(self.curve_tolerance)
    }
}

/// A candidate point produced by one of the point-snap extractors.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    point: Vec2,
    kind: SnapKind,
    entity: Option<EntityId>,
    /// Tracking constraints stay valid no matter how far the cursor is, so
    /// they bypass the aperture check.
    sticky: bool,
}

impl Candidate {
    fn new(point: Vec2, kind: SnapKind, entity: Option<EntityId>) -> Self {
        Self {
            point,
            kind,
            entity,
            sticky: false,
        }
    }
}

/// The snap engine. Stateless apart from the settings.
#[derive(Debug, Clone, Default)]
pub struct SnapEngine {
    pub settings: SnapSettings,
    /// Last resolved snap, so the UI can keep showing the marker.
    pub last: Option<Snap>,
    /// Debug list of everything considered in the last query.
    pub candidates_seen: usize,
}

impl SnapEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_settings(s: SnapSettings) -> Self {
        Self {
            settings: s,
            ..Default::default()
        }
    }

    #[inline]
    pub fn toggle(&mut self) {
        self.settings.enabled = !self.settings.enabled;
    }

    /// Resolve the cursor position to a snap point.
    ///
    /// `base` is the previous point the user fixed (for polar/ortho tracking);
    /// pass `None` when there is none.
    pub fn resolve(
        &mut self,
        doc: &Document,
        cam: &Camera2D,
        cursor_world: Vec2,
        base: Option<Vec2>,
    ) -> Option<Snap> {
        if !self.settings.enabled {
            self.last = None;
            return None;
        }
        let world_r = cam.pixels_to_world(self.settings.aperture_px);
        let screen_cursor = cam.world_to_screen(cursor_world);
        let search = Rect2::new(cursor_world, cursor_world).expand(Vec2::splat(world_r));

        let mut cands: Vec<Candidate> = Vec::new();

        // Tracking is collected alongside the point snaps rather than first:
        // priority arbitration below decides, and a real snap (endpoint) must
        // beat a tracking constraint (ortho), exactly as in AutoCAD.
        if let Some(t) = base.and_then(|b| self.tracking(b, cursor_world)) {
            cands.push(Candidate {
                sticky: true,
                ..Candidate::new(t.point, t.kind, None)
            });
        }

        self.collect_point_snaps(doc, cursor_world, search, &mut cands);
        self.collect_intersections(doc, cam, search, &mut cands);
        if self.settings.has(SnapKind::Nearest) {
            self.collect_nearest(doc, cam, cursor_world, search, &mut cands);
        }
        if self.settings.has(SnapKind::Perpendicular) {
            self.collect_perpendicular(doc, cam, cursor_world, base, &mut cands);
        }
        if self.settings.has(SnapKind::Tangent) {
            self.collect_tangent(doc, cam, cursor_world, base, &mut cands);
        }
        if self.settings.has(SnapKind::Closest) {
            self.collect_closest(doc, cam, cursor_world, search, &mut cands);
        }
        if self.settings.has(SnapKind::Grid) && self.settings.grid_spacing > 0.0 {
            let g = self.settings.grid_spacing;
            let snapped = Vec2::new(
                (cursor_world.x / g).round() * g,
                (cursor_world.y / g).round() * g,
            );
            if snapped.distance(cursor_world) <= world_r {
                cands.push(Candidate::new(snapped, SnapKind::Grid, None));
            }
        }

        self.candidates_seen = cands.len();
        let best = self.pick_best(cam, screen_cursor, &cands)?;
        self.last = Some(best);
        Some(best)
    }

    /// Choose the candidate to snap to.
    ///
    /// Priority dominates: the highest-priority mode with a candidate inside the
    /// aperture wins, and only then is distance used to break ties. This is why
    /// an endpoint marker beats a "nearest point on curve" that is a hair
    /// closer — the same rule AutoCAD uses to make endpoints feel magnetic.
    fn pick_best(&self, cam: &Camera2D, screen_cursor: Vec2, cands: &[Candidate]) -> Option<Snap> {
        let mut best: Option<(u32, f32, Snap)> = None;
        for c in cands {
            if !self.settings.has(c.kind) {
                continue;
            }
            let d_px = cam.world_to_screen(c.point).distance(screen_cursor);
            if !c.sticky && d_px > self.settings.aperture_px {
                continue;
            }
            let prio = c.kind.bit().trailing_zeros();
            let better = match &best {
                None => true,
                Some((bp, bd, _)) => prio < *bp || (prio == *bp && d_px < *bd),
            };
            if better {
                best = Some((
                    prio,
                    d_px,
                    Snap {
                        point: c.point,
                        kind: c.kind,
                        source: c.entity,
                        reference: None,
                    },
                ));
            }
        }
        best.map(|(_, _, s)| s)
    }

    /// Ortho / polar tracking from `base` toward the cursor.
    pub fn tracking(&self, base: Vec2, cursor: Vec2) -> Option<Snap> {
        let d = cursor - base;
        if d.length_squared() < 1e-12 {
            return None;
        }
        let mut angle = d.y.atan2(d.x);

        if self.settings.has(SnapKind::Ortho) {
            for a in &self.settings.ortho_angles {
                // ~5 degrees of slack, as AutoCAD uses.
                if let Some(delta) = angle_delta_to(angle, *a)
                    && delta.abs() < 0.0873
                {
                    angle = *a;
                    break;
                }
            }
        }
        let polar_on = self.settings.has(SnapKind::Polar);
        if polar_on && self.settings.polar_increment_deg > 0.0 {
            let inc = self.settings.polar_increment_deg * cad_core::RAD;
            if inc > 1e-9 {
                angle = (angle / inc).round() * inc;
            }
        }

        let kind = if polar_on && self.settings.polar_increment_deg > 0.0 {
            SnapKind::Polar
        } else if (angle - d.y.atan2(d.x)).abs() > 1e-6 {
            SnapKind::Ortho
        } else {
            return None;
        };
        Some(Snap::new(
            base + Vec2::new(angle.cos(), angle.sin()) * d.length(),
            kind,
        ))
    }

    // -------------------------------------------------------------- collectors

    fn collect_point_snaps(
        &self,
        doc: &Document,
        cursor: Vec2,
        search: Rect2,
        out: &mut Vec<Candidate>,
    ) {
        for (id, e) in iter_entities(doc) {
            if !self.entity_pickable(doc, e) {
                continue;
            }
            if !search.overlaps(e.bounds_2d()) {
                continue;
            }
            push_entity_points(e, self, out, id);
        }
        let _ = cursor;
    }

    fn entity_pickable(&self, doc: &Document, e: &Entity) -> bool {
        if !e.common.visible || e.common.locked {
            return false;
        }
        doc.layers.visible(e.layer()) && !doc.layers.locked(e.layer())
    }

    fn collect_intersections(
        &self,
        doc: &Document,
        cam: &Camera2D,
        search: Rect2,
        out: &mut Vec<Candidate>,
    ) {
        let want_real = self.settings.has(SnapKind::Intersection);
        let want_app = self.settings.has(SnapKind::ApparentIntersection);
        if !want_real && !want_app {
            return;
        }
        let ids: Vec<EntityId> = doc
            .entities
            .candidates_in(search)
            .into_iter()
            .filter(|id| {
                doc.entities
                    .get(*id)
                    .map(|e| self.entity_pickable(doc, e))
                    .unwrap_or(false)
            })
            .collect();
        if ids.len() < 2 {
            return;
        }
        let opts = self.settings.tess();
        let screen_c = cam.world_to_screen(search.center());
        let max_px = self.settings.aperture_px;
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let (Some(a), Some(b)) = (doc.entities.get(ids[i]), doc.entities.get(ids[j]))
                else {
                    continue;
                };
                let (Some(ca), Some(cb)) = (a.as_curve(), b.as_curve()) else {
                    continue;
                };
                let hits = cad_geom::intersect::intersect(&ca, &cb, &opts);
                for h in hits {
                    let kind = match h.kind {
                        HitKind::Crossing | HitKind::ArcOnLine | HitKind::ArcOnArc => {
                            SnapKind::Intersection
                        }
                        HitKind::VertexOnCurve | HitKind::Overlap => SnapKind::ApparentIntersection,
                    };
                    if !self.settings.has(kind) {
                        continue;
                    }
                    if cam.world_to_screen(h.point).distance(screen_c) <= max_px {
                        out.push(Candidate::new(h.point, kind, None));
                    }
                }
            }
        }
    }

    fn collect_nearest(
        &self,
        doc: &Document,
        cam: &Camera2D,
        cursor: Vec2,
        search: Rect2,
        out: &mut Vec<Candidate>,
    ) {
        let screen_c = cam.world_to_screen(cursor);
        let max_px = self.settings.aperture_px;
        for id in doc.entities.candidates_in(search) {
            let Some(e) = doc.entities.get(id) else {
                continue;
            };
            if !self.entity_pickable(doc, e) {
                continue;
            }
            let Some(curve) = e.as_curve() else { continue };
            let q = curve.closest_point(cursor);
            if cam.world_to_screen(q).distance(screen_c) <= max_px {
                out.push(Candidate::new(q, SnapKind::Nearest, Some(id)));
            }
        }
    }

    fn collect_closest(
        &self,
        doc: &Document,
        cam: &Camera2D,
        cursor: Vec2,
        search: Rect2,
        out: &mut Vec<Candidate>,
    ) {
        // "Closest" ignores the aperture and always wins on distance, which is
        // what makes it useful for picking in dense drawings.
        let mut best: Option<(f32, Vec2, EntityId)> = None;
        for id in doc.entities.candidates_in(search) {
            let Some(e) = doc.entities.get(id) else {
                continue;
            };
            if !self.entity_pickable(doc, e) {
                continue;
            }
            let Some(curve) = e.as_curve() else { continue };
            let q = curve.closest_point(cursor);
            let d = cam.world_to_screen(q).distance(cam.world_to_screen(cursor));
            if best.is_none_or(|(bd, _, _)| d < bd) {
                best = Some((d, q, id));
            }
        }
        if let Some((_, q, id)) = best {
            out.push(Candidate::new(q, SnapKind::Closest, Some(id)));
        }
    }

    fn collect_perpendicular(
        &self,
        doc: &Document,
        cam: &Camera2D,
        cursor: Vec2,
        base: Option<Vec2>,
        out: &mut Vec<Candidate>,
    ) {
        // With a base point, perpendicular means "along the line from base to
        // cursor, foot of the perpendicular onto the target entity".
        let Some(base) = base else { return };
        let d = cursor - base;
        if d.length_squared() < 1e-12 {
            return;
        }
        let screen_c = cam.world_to_screen(cursor);
        let max_px = self.settings.aperture_px;
        let search = Rect2::new(cursor, cursor).expand(Vec2::splat(cam.pixels_to_world(max_px)));
        for id in doc.entities.candidates_in(search) {
            let Some(e) = doc.entities.get(id) else {
                continue;
            };
            if !self.entity_pickable(doc, e) {
                continue;
            }
            let Some(curve) = e.as_curve() else { continue };
            if let Some(foot) = perpendicular_foot(&curve, base, d) {
                let p = foot;
                if cam.world_to_screen(p).distance(screen_c) <= max_px {
                    out.push(Candidate::new(p, SnapKind::Perpendicular, Some(id)));
                }
            }
        }
    }

    fn collect_tangent(
        &self,
        doc: &Document,
        cam: &Camera2D,
        cursor: Vec2,
        base: Option<Vec2>,
        out: &mut Vec<Candidate>,
    ) {
        let Some(base) = base else { return };
        let search = Rect2::new(cursor, cursor)
            .expand(Vec2::splat(cam.pixels_to_world(self.settings.aperture_px)));
        let screen_c = cam.world_to_screen(cursor);
        let max_px = self.settings.aperture_px;
        for id in doc.entities.candidates_in(search) {
            let Some(e) = doc.entities.get(id) else {
                continue;
            };
            if !self.entity_pickable(doc, e) {
                continue;
            }
            let Some(p) = tangent_point(&e.entity, base, cursor) else {
                continue;
            };
            if cam.world_to_screen(p).distance(screen_c) <= max_px {
                out.push(Candidate::new(p, SnapKind::Tangent, Some(id)));
            }
        }
    }
}

fn angle_delta_to(a: f32, target: f32) -> Option<f32> {
    Some(wrap_pi(a - target))
}

/// Iterate live entities with their handles, without allocating.
fn iter_entities(doc: &Document) -> impl Iterator<Item = (EntityId, &Entity)> {
    doc.entities
        .iter()
        .enumerate()
        .map(|(i, e)| (EntityId(i as u32), e))
}

fn push_entity_points(e: &Entity, engine: &SnapEngine, out: &mut Vec<Candidate>, id: EntityId) {
    let s = &engine.settings;
    match &e.entity {
        EntityKind::Line(l) => {
            if s.has(SnapKind::Endpoint) {
                out.push(Candidate::new(l.p0, SnapKind::Endpoint, Some(id)));
                out.push(Candidate::new(l.p1, SnapKind::Endpoint, Some(id)));
            }
            if s.has(SnapKind::Midpoint) {
                out.push(Candidate::new(l.midpoint(), SnapKind::Midpoint, Some(id)));
            }
            if s.has(SnapKind::Node) {
                out.push(Candidate::new(l.p0, SnapKind::Node, Some(id)));
            }
        }
        EntityKind::Circle(c) => {
            if s.has(SnapKind::Center) {
                out.push(Candidate::new(c.center, SnapKind::Center, Some(id)));
            }
            if s.has(SnapKind::Quadrant) {
                for i in 0..4 {
                    let a = cad_core::FRAC_PI_2 * i as f32;
                    out.push(Candidate::new(
                        c.center + Vec2::new(a.cos(), a.sin()) * c.radius,
                        SnapKind::Quadrant,
                        Some(id),
                    ));
                }
            }
        }
        EntityKind::Arc(a) => {
            if s.has(SnapKind::Center) {
                out.push(Candidate::new(a.center, SnapKind::Center, Some(id)));
            }
            if s.has(SnapKind::Endpoint) {
                out.push(Candidate::new(
                    a.start_point(),
                    SnapKind::Endpoint,
                    Some(id),
                ));
                out.push(Candidate::new(a.end_point(), SnapKind::Endpoint, Some(id)));
            }
            if s.has(SnapKind::Midpoint) {
                out.push(Candidate::new(
                    a.point_at(0.5),
                    SnapKind::Midpoint,
                    Some(id),
                ));
            }
            if s.has(SnapKind::Quadrant) {
                for i in -4..=4 {
                    let ang = cad_core::FRAC_PI_2 * i as f32;
                    if a.contains_angle(ang) {
                        out.push(Candidate::new(
                            a.center + Vec2::new(ang.cos(), ang.sin()) * a.radius,
                            SnapKind::Quadrant,
                            Some(id),
                        ));
                    }
                }
            }
        }
        EntityKind::Ellipse(e) => {
            if s.has(SnapKind::Center) {
                out.push(Candidate::new(e.center, SnapKind::Center, Some(id)));
            }
            if s.has(SnapKind::Quadrant) {
                for i in 0..4 {
                    out.push(Candidate::new(
                        e.point_at(cad_core::FRAC_PI_2 * i as f32),
                        SnapKind::Quadrant,
                        Some(id),
                    ));
                }
            }
            if s.has(SnapKind::Endpoint) {
                out.push(Candidate::new(
                    e.point_at(0.0),
                    SnapKind::Endpoint,
                    Some(id),
                ));
                out.push(Candidate::new(
                    e.point_at(cad_core::PI),
                    SnapKind::Endpoint,
                    Some(id),
                ));
            }
        }
        EntityKind::Polyline(p) | EntityKind::Region(p) => {
            // Every vertex snaps, including the last one of an open polyline.
            for v in &p.vertices {
                if s.has(SnapKind::Endpoint) {
                    out.push(Candidate::new(*v, SnapKind::Endpoint, Some(id)));
                }
            }
            if s.has(SnapKind::Midpoint) {
                for i in 0..p.segment_count() {
                    let a = p.point(i);
                    let b = p.point(i + 1);
                    if p.bulge_at(i).is_arc() {
                        out.push(Candidate::new(
                            p.segment_arc(i).point_at(0.5),
                            SnapKind::Midpoint,
                            Some(id),
                        ));
                    } else {
                        out.push(Candidate::new((a + b) * 0.5, SnapKind::Midpoint, Some(id)));
                    }
                }
            }
        }
        EntityKind::Point(p) => {
            if s.has(SnapKind::Node) {
                out.push(Candidate::new(p.position.xy(), SnapKind::Node, Some(id)));
            }
        }
        EntityKind::Insert(i) if s.has(SnapKind::Insert) => {
            out.push(Candidate::new(i.position.xy(), SnapKind::Insert, Some(id)));
        }
        _ => {}
    }
}

/// Perpendicular from a point `p` along direction `dir` onto a curve.
fn perpendicular_foot(curve: &Curve, base: Vec2, dir: Vec2) -> Option<Vec2> {
    let d = dir.normalize();
    match curve {
        Curve::Line(l) => {
            let l = l.extended(1e6);
            let n = l.normal();
            let denom = n.dot(d);
            if denom.abs() < 1e-9 {
                return None; // parallel
            }
            let t = (l.p0 - base).dot(n) / denom;
            let p = base + d * t;
            // Must land on the (extended) line, ahead of the base point.
            if (p - base).dot(d) >= 0.0 {
                Some(p)
            } else {
                None
            }
        }
        Curve::Circle(c) => {
            // Foot of the perpendicular from the centre onto the ray.
            let oc = base - c.center;
            let t = oc.dot(d);
            if t <= 0.0 {
                return None;
            }
            let foot = c.center + d * t;
            let dist = foot.distance(base);
            if (dist - c.radius).abs() < 1e-3 * c.radius.max(1.0) {
                Some(foot)
            } else {
                None
            }
        }
        Curve::Arc(_) | Curve::Ellipse(_) | Curve::Polyline(_) => {
            // Solve on the tessellated approximation; accurate enough for a snap
            // and always inside the aperture.
            let opts = TessellationOptions::with_tolerance(0.005);
            let pts = tessellate(curve, &opts);
            let mut best: Option<f32> = None;
            for w in pts.windows(2) {
                let l = Line::new(w[0], w[1]);
                let n = l.normal();
                let denom = n.dot(d);
                if denom.abs() < 1e-9 {
                    continue;
                }
                let t = (l.p0 - base).dot(n) / denom;
                if t < 0.0 {
                    continue;
                }
                let p = base + d * t;
                let along = (p - l.p0).dot(l.dir());
                let seg = l.delta().length();
                if along < -1e-3 || along > seg + 1e-3 {
                    continue;
                }
                if best.is_none_or(|b| t < b) {
                    best = Some(t);
                }
            }
            best.map(|t| base + d * t)
        }
    }
}

/// Tangent point construction: from `base`, find where the line to the circle
/// `center/radius` is tangent, on the side closest to `cursor`.
fn tangent_point(kind: &EntityKind, base: Vec2, cursor: Vec2) -> Option<Vec2> {
    let (center, radius) = match kind {
        EntityKind::Circle(c) => (c.center, c.radius),
        EntityKind::Arc(a) => (a.center, a.radius),
        _ => return None,
    };
    let oc = base - center;
    let d2 = oc.length_squared();
    if d2 <= radius * radius {
        return None; // base is inside the circle
    }
    // Standard tangent-point construction.
    let d = d2.sqrt();
    let a = radius * radius / d;
    let h = (radius * radius - a * a).max(0.0).sqrt();
    let dir = (oc / d).normalize_or(Vec2::X);
    let mid = center + dir * a;
    let perp = dir.perp();
    let p0 = mid + perp * h;
    let p1 = mid - perp * h;
    // Choose the tangent point nearest the cursor.
    if p0.distance(cursor) <= p1.distance(cursor) {
        Some(p0)
    } else {
        Some(p1)
    }
}

/// Distance from a point to a polyline, used by the "closest" test in tests.
pub fn distance_to_polyline(pts: &[Vec2], p: Vec2) -> f32 {
    pts.windows(2)
        .map(|w| Line::new(w[0], w[1]).distance_to(p))
        .fold(f32::INFINITY, f32::min)
}

/// Convenience: the quadrant angles of a circle, for tests and the UI.
pub fn quadrants() -> [Vec2; 4] {
    [Vec2::X, Vec2::Y, -Vec2::X, -Vec2::Y]
}

/// Silences the unused-import warning for `Polyline` when features change.
const _: Option<Polyline> = None;
const _: Option<Circle> = None;
const _: Option<Arc> = None;
const _: Option<Vec3> = None;

#[cfg(test)]
mod tests {
    use super::*;
    use cad_geom::curve::Line;

    fn doc_with_line() -> (Document, EntityId) {
        let mut doc = Document::new();
        let layer = doc.layers.ensure_default();
        let id = doc.add(
            Entity::line(Line::new(Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0))).with_layer(layer),
        );
        (doc, id)
    }

    fn doc_with_circle() -> (Document, EntityId) {
        let mut doc = Document::new();
        let layer = doc.layers.ensure_default();
        let id = doc.add(Entity::circle(Circle::new(Vec2::ZERO, 10.0)).with_layer(layer));
        (doc, id)
    }

    fn cam() -> Camera2D {
        let mut c = Camera2D::new(Vec2::new(800.0, 600.0));
        c.center = Vec2::new(50.0, 0.0);
        c.scale = 2.0;
        c
    }

    #[test]
    fn endpoint_snap_wins() {
        let (doc, _) = doc_with_line();
        let mut e = SnapEngine::new();
        // Cursor 3px (1.5 world units) from the left endpoint.
        let s = e.resolve(&doc, &cam(), Vec2::new(1.5, 0.4), None).unwrap();
        assert_eq!(s.kind, SnapKind::Endpoint);
        assert!(s.point.distance(Vec2::ZERO) < 1e-4);
    }

    #[test]
    fn snap_outside_the_aperture_returns_none() {
        let (doc, _) = doc_with_line();
        let mut e = SnapEngine::new();
        assert!(
            e.resolve(&doc, &cam(), Vec2::new(50.0, 40.0), None)
                .is_none()
        );
    }

    #[test]
    fn midpoint_snap() {
        let (doc, _) = doc_with_line();
        let mut e = SnapEngine::new();
        e.settings.set(SnapKind::Endpoint, false);
        let s = e.resolve(&doc, &cam(), Vec2::new(50.0, 1.0), None).unwrap();
        assert_eq!(s.kind, SnapKind::Midpoint);
        assert!(s.point.distance(Vec2::new(50.0, 0.0)) < 1e-4);
    }

    #[test]
    fn center_and_quadrant_snap() {
        let (doc, _) = doc_with_circle();
        let mut e = SnapEngine::new();
        e.settings.set(SnapKind::Quadrant, false);
        let s = e.resolve(&doc, &cam(), Vec2::new(1.0, 1.0), None).unwrap();
        assert_eq!(s.kind, SnapKind::Center);

        let mut e2 = SnapEngine::new();
        e2.settings.set(SnapKind::Center, false);
        let s2 = e2
            .resolve(&doc, &cam(), Vec2::new(10.0, 1.0), None)
            .unwrap();
        assert_eq!(s2.kind, SnapKind::Quadrant);
        assert!(s2.point.distance(Vec2::new(10.0, 0.0)) < 1e-4);
    }

    #[test]
    fn intersection_snap() {
        let mut doc = Document::new();
        let layer = doc.layers.ensure_default();
        doc.add(
            Entity::line(Line::new(Vec2::new(0.0, -20.0), Vec2::new(0.0, 20.0))).with_layer(layer),
        );
        doc.add(
            Entity::line(Line::new(Vec2::new(-20.0, 0.0), Vec2::new(20.0, 0.0))).with_layer(layer),
        );
        let mut e = SnapEngine::new();
        let s = e.resolve(&doc, &cam(), Vec2::new(0.6, 0.6), None).unwrap();
        assert_eq!(s.kind, SnapKind::Intersection);
        assert!(s.point.distance(Vec2::ZERO) < 1e-3, "{:?}", s.point);
    }

    #[test]
    fn ortho_tracking_locks_to_the_axes() {
        let e = SnapEngine::new();
        let s = e.tracking(Vec2::ZERO, Vec2::new(10.0, 0.6)).unwrap();
        assert_eq!(s.kind, SnapKind::Ortho);
        // 3.4 degrees off axis is inside the 5 degree slack, so Y snaps to 0.
        assert!(
            s.point.distance(Vec2::new(10.0087, 0.0)) < 0.02,
            "{:?}",
            s.point
        );
        assert!(s.point.y.abs() < 1e-5);
    }

    #[test]
    fn polar_tracking_increments() {
        let mut e = SnapEngine::new();
        e.settings.polar_increment_deg = 15.0;
        let s = e.tracking(Vec2::ZERO, Vec2::new(10.0, 3.0)).unwrap();
        assert_eq!(s.kind, SnapKind::Polar);
        let a = (s.point - Vec2::ZERO)
            .y
            .atan2((s.point - Vec2::ZERO).x)
            .to_degrees();
        assert!((a / 15.0 - (a / 15.0).round()).abs() < 1e-3, "angle={a}");
    }

    #[test]
    fn tracking_returns_none_without_a_lock() {
        let mut e = SnapEngine::new();
        e.settings.polar_increment_deg = 0.0;
        assert!(e.tracking(Vec2::ZERO, Vec2::new(10.0, 3.0)).is_none());
    }

    #[test]
    fn perpendicular_snap_from_a_base_point() {
        let (doc, _) = doc_with_line();
        let mut e = SnapEngine::new();
        e.settings.modes = mode::PERPENDICULAR;
        // Base above the line, cursor 4 world units (8 px) from the foot.
        let s = e
            .resolve(
                &doc,
                &cam(),
                Vec2::new(50.0, 4.0),
                Some(Vec2::new(50.0, 40.0)),
            )
            .unwrap();
        assert_eq!(s.kind, SnapKind::Perpendicular);
        assert!(
            s.point.distance(Vec2::new(50.0, 0.0)) < 1e-3,
            "{:?}",
            s.point
        );
    }

    #[test]
    fn real_snap_beats_ortho_tracking() {
        let (doc, _) = doc_with_line();
        let mut e = SnapEngine::new();
        e.settings.modes = mode::MIDPOINT | mode::ORTHO;
        // The base point is directly above the midpoint, so the ortho constraint
        // and the midpoint candidate are 4 units apart. The stronger snap wins
        // even though the tracked point is closer to the cursor.
        let s = e
            .resolve(
                &doc,
                &cam(),
                Vec2::new(50.0, 4.0),
                Some(Vec2::new(50.0, 40.0)),
            )
            .unwrap();
        assert_eq!(s.kind, SnapKind::Midpoint);
        assert!(s.point.distance(Vec2::new(50.0, 0.0)) < 1e-3);
    }

    #[test]
    fn tangent_point_from_outside_a_circle() {
        let (doc, id) = doc_with_circle();
        let mut e = SnapEngine::new();
        e.settings.modes = mode::TANGENT;
        // Cursor close to the upper tangent point seen from (30, 0).
        let s = e
            .resolve(
                &doc,
                &cam(),
                Vec2::new(4.0, 8.5),
                Some(Vec2::new(30.0, 0.0)),
            )
            .unwrap();
        assert_eq!(s.kind, SnapKind::Tangent);
        assert!((s.point.length() - 10.0).abs() < 1e-2, "{:?}", s.point);
        let _ = id;
    }

    #[test]
    fn tangent_from_inside_a_circle_is_rejected() {
        let (doc, _) = doc_with_circle();
        let mut e = SnapEngine::new();
        e.settings.modes = mode::TANGENT;
        let s = e.resolve(&doc, &cam(), Vec2::new(9.2, 4.3), Some(Vec2::ZERO));
        assert!(s.is_none());
    }

    #[test]
    fn nearest_snap_on_a_curve() {
        let (doc, _) = doc_with_line();
        let mut e = SnapEngine::new();
        e.settings.modes = mode::NEAREST;
        let s = e.resolve(&doc, &cam(), Vec2::new(30.0, 1.5), None).unwrap();
        assert_eq!(s.kind, SnapKind::Nearest);
        assert!(
            s.point.distance(Vec2::new(30.0, 0.0)) < 1e-3,
            "{:?}",
            s.point
        );
    }

    #[test]
    fn grid_snap_is_optional() {
        let (doc, _) = doc_with_line();
        let mut e = SnapEngine::new();
        e.settings.modes = mode::GRID;
        e.settings.grid_spacing = 5.0;
        let s = e.resolve(&doc, &cam(), Vec2::new(52.0, 7.0), None).unwrap();
        assert_eq!(s.kind, SnapKind::Grid);
        assert!(s.point.distance(Vec2::new(50.0, 5.0)) < 1e-4);
    }

    #[test]
    fn disabled_engine_never_snaps() {
        let (doc, _) = doc_with_line();
        let mut e = SnapEngine::new();
        e.settings.enabled = false;
        assert!(e.resolve(&doc, &cam(), Vec2::ZERO, None).is_none());
        e.toggle();
        assert!(e.resolve(&doc, &cam(), Vec2::ZERO, None).is_some());
    }

    #[test]
    fn locked_and_hidden_entities_are_ignored() {
        let (mut doc, id) = doc_with_line();
        let layer = doc.entities.get(id).unwrap().layer();
        doc.layers.set_locked(layer, true);
        let mut e = SnapEngine::new();
        e.settings.modes = mode::ENDPOINT;
        assert!(e.resolve(&doc, &cam(), Vec2::ZERO, None).is_none());
        doc.layers.set_locked(layer, false);
        doc.layers.set_visible(layer, false);
        assert!(e.resolve(&doc, &cam(), Vec2::ZERO, None).is_none());
    }

    #[test]
    fn settings_bit_toggling() {
        let mut s = SnapSettings::default();
        assert!(s.has(SnapKind::Endpoint));
        s.toggle(SnapKind::Endpoint);
        assert!(!s.has(SnapKind::Endpoint));
        s.toggle(SnapKind::Endpoint);
        assert!(s.has(SnapKind::Endpoint));
        s.modes = mode::FAST;
        assert!(!s.has(SnapKind::Intersection));
        assert!(s.has(SnapKind::Endpoint));
    }

    #[test]
    fn snapping_is_resolution_independent() {
        let (doc, _) = doc_with_line();
        let mut e = SnapEngine::new();
        let mut results = Vec::new();
        for scale in [0.5f32, 2.0, 20.0] {
            let mut c = cam();
            c.scale = scale;
            // Always 3 pixels away.
            let world_off = c.pixels_to_world(3.0);
            let s = e.resolve(
                &doc,
                &c,
                Vec2::ZERO + Vec2::new(world_off * 0.3, world_off * 0.9),
                None,
            );
            results.push(s.map(|s| s.kind));
        }
        assert!(
            results.iter().all(|r| *r == Some(SnapKind::Endpoint)),
            "{results:?}"
        );
    }

    #[test]
    fn polyline_vertex_snaps() {
        let mut doc = Document::new();
        let layer = doc.layers.ensure_default();
        doc.entities
            .rebuild_index(cad_core::Rect2::from_xywh(-10.0, -10.0, 200.0, 200.0));
        doc.add(
            Entity::polyline(Polyline::new(
                vec![
                    Vec2::new(0.0, 0.0),
                    Vec2::new(10.0, 0.0),
                    Vec2::new(10.0, 10.0),
                ],
                false,
            ))
            .with_layer(layer),
        );
        doc.entities
            .rebuild_index(cad_core::Rect2::from_xywh(-10.0, -10.0, 200.0, 200.0));
        let mut e = SnapEngine::new();
        e.settings.modes = mode::ENDPOINT;
        let s = e
            .resolve(&doc, &cam(), Vec2::new(10.0, 10.5), None)
            .unwrap();
        assert_eq!(s.kind, SnapKind::Endpoint);
        assert!(s.point.distance(Vec2::new(10.0, 10.0)) < 1e-4);
    }
}
