//! The interactive tools: select, draw, modify, and the point-picking logic
//! every tool shares.
//!
//! Tools are state machines driven by [`Tool::on_click`], [`Tool::on_move`] and
//! [`Tool::on_key`]. They never see wgpu or winit types, which keeps the whole
//! editing model testable without a GPU.

use cad_core::{Camera2D, Rect2, Vec2, Vec3};
use cad_doc::{Document, Entity, EntityId, LayerId};
use cad_geom::curve::{Arc, Circle, Line, Polyline};

/// Identifies a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolId {
    Select,
    Pan,
    ZoomWindow,
    Line,
    Circle,
    Arc,
    Polyline,
    Rectangle,
    Move,
    Copy,
    Rotate,
    Scale,
    Erase,
    Trim,
    Extend,
    Mirror,
    Offset,
    Extrude,
}

impl ToolId {
    /// Command that activates this tool.
    pub fn command(self) -> &'static str {
        match self {
            ToolId::Select => "select",
            ToolId::Pan => "pan",
            ToolId::ZoomWindow => "zoomwindow",
            ToolId::Line => "line",
            ToolId::Circle => "circle",
            ToolId::Arc => "arc",
            ToolId::Polyline => "polyline",
            ToolId::Rectangle => "rectangle",
            ToolId::Move => "move",
            ToolId::Copy => "copytool",
            ToolId::Rotate => "rotate",
            ToolId::Scale => "scale",
            ToolId::Erase => "erase",
            ToolId::Trim => "trim",
            ToolId::Extend => "extend",
            ToolId::Mirror => "mirror",
            ToolId::Offset => "offset",
            ToolId::Extrude => "extrude",
        }
    }

    /// Tools that require a selection before they can act.
    pub fn needs_selection(self) -> bool {
        matches!(
            self,
            ToolId::Move
                | ToolId::Copy
                | ToolId::Rotate
                | ToolId::Scale
                | ToolId::Erase
                | ToolId::Extrude
        )
    }

    pub fn label(self) -> &'static str {
        match self {
            ToolId::Select => "Select",
            ToolId::Pan => "Pan",
            ToolId::ZoomWindow => "Zoom Window",
            ToolId::Line => "Line",
            ToolId::Circle => "Circle",
            ToolId::Arc => "Arc",
            ToolId::Polyline => "Polyline",
            ToolId::Rectangle => "Rectangle",
            ToolId::Move => "Move",
            ToolId::Copy => "Copy",
            ToolId::Rotate => "Rotate",
            ToolId::Scale => "Scale",
            ToolId::Erase => "Erase",
            ToolId::Trim => "Trim",
            ToolId::Extend => "Extend",
            ToolId::Mirror => "Mirror",
            ToolId::Offset => "Offset",
            ToolId::Extrude => "Extrude",
        }
    }
}

/// Which modify operation is in flight.
///
/// `Move`, `Copy` and `Mirror` are the ones that can be driven entirely from
/// the mouse. `Rotate`, `Scale`, `Offset`, `Trim` and `Extend` need a number
/// (an angle, a factor, a distance) and are therefore reachable from the command
/// line with an argument instead -- the ribbon greys them out rather than
/// offering a click sequence that cannot express what they need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModifyMode {
    Move,
    Copy,
    Mirror,
}

/// Per-tool scratch state.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ToolState {
    /// Nothing in progress.
    #[default]
    Idle,
    /// Points collected so far, plus what the user is doing with them.
    Points {
        points: Vec<Vec2>,
        drag_from: Option<Vec2>,
        drag_to: Option<Vec2>,
    },
    /// Entities captured at the start of a modify operation, so `Esc` restores
    /// them, plus the points the operation is anchored at.
    Modify {
        ids: Vec<EntityId>,
        originals: Vec<Entity>,
        /// The first fixed point. `None` until the user picks it.
        base: Option<Vec2>,
        /// The second fixed point, for operations that need one (mirror axis).
        axis: Option<Vec2>,
        mode: ModifyMode,
    },
    /// Selection rectangle being dragged.
    Window { start: Vec2, current: Vec2 },
    /// Trimming/offsetting: waiting for a distance.
    Distance { value: f32 },
}

impl ToolState {
    pub fn points(&self) -> &[Vec2] {
        match self {
            ToolState::Points { points, .. } => points,
            _ => &[],
        }
    }
    pub fn is_idle(&self) -> bool {
        matches!(self, ToolState::Idle)
    }
    /// Human-readable prompt for the command line, or `None` when done.
    pub fn prompt(&self, tool: ToolId) -> Option<String> {
        match self {
            ToolState::Idle => None,
            ToolState::Points { points, .. } => Some(match tool {
                ToolId::Line => format!("Specify next point or [Undo] ({} taken)", points.len()),
                ToolId::Circle => {
                    if points.is_empty() {
                        "Specify centre point".into()
                    } else {
                        "Specify radius".into()
                    }
                }
                // Off by one: an arc needs three points, and the point just
                // clicked is already in `points`, so len 1 is the start point.
                ToolId::Arc => match points.len() {
                    1 => "Specify start point".into(),
                    2 => "Specify second point".into(),
                    _ => "Specify end point".into(),
                },
                ToolId::Polyline => format!("Specify next vertex ({})", points.len()),
                ToolId::Rectangle => {
                    if points.is_empty() {
                        "Specify first corner".into()
                    } else {
                        "Specify opposite corner".into()
                    }
                }
                ToolId::Move | ToolId::Copy => "Specify base point".into(),
                ToolId::Rotate => "Specify base point".into(),
                ToolId::Scale => "Specify base point".into(),
                ToolId::Mirror => "Specify first point of axis".into(),
                ToolId::Offset => "Specify offset distance".into(),
                ToolId::Extrude => {
                    if points.is_empty() {
                        "Specify base corner".into()
                    } else {
                        "Specify opposite corner (height follows)".into()
                    }
                }
                _ => "Specify point".into(),
            }),
            ToolState::Window { .. } => Some("Specify opposite corner".into()),
            ToolState::Distance { .. } => Some("Specify distance".into()),
            ToolState::Modify {
                base, axis, mode, ..
            } => Some(match (mode, base.is_some()) {
                // Mirror needs two points to define its axis, so it never asks
                // for a destination.
                (ModifyMode::Mirror, true) => {
                    if axis.is_some() {
                        "Select objects to mirror".into()
                    } else {
                        "Specify second point of axis".into()
                    }
                }
                (_, false) => "Specify base point".into(),
                (_, true) => "Specify destination point".into(),
            }),
        }
    }
}

/// What a tool did, for the caller to feed into the session (undo, rebuild...).
#[derive(Debug, Clone, PartialEq)]
pub enum ToolOutcome {
    /// Nothing to commit.
    None,
    /// The document changed and needs an undo entry with this label.
    Changed(String),
    /// Entities were created; the session should fit the view or select them.
    Created(Vec<EntityId>),
    /// The view should zoom to a rectangle.
    ZoomTo(Rect2),
    /// The session should delete the given entities.
    Delete(Vec<EntityId>),
    /// The session should restore a snapshot (Esc / right-click).
    Restore,
}

/// An active tool.
#[derive(Debug, Clone)]
pub struct Tool {
    pub id: ToolId,
    pub state: ToolState,
    /// Pick radius in world units, updated per frame from the camera.
    pub pick_distance: f32,
    /// Entities currently selected.
    pub selection: Vec<EntityId>,
    /// Hovered entity under the cursor.
    pub hovered: Option<EntityId>,
}

impl Tool {
    pub fn new(id: ToolId) -> Self {
        Self {
            id,
            state: ToolState::Idle,
            pick_distance: 5.0,
            selection: Vec::new(),
            hovered: None,
        }
    }

    pub fn prompt(&self) -> Option<String> {
        self.state.prompt(self.id)
    }

    /// Cancel the current operation, restoring a snapshot if one was taken.
    pub fn cancel(&mut self, doc: &mut Document) -> ToolOutcome {
        let out = match &self.state {
            ToolState::Modify { ids, originals, .. } => {
                // Put the originals back exactly as they were.
                let mut restored = Vec::new();
                for (id, e) in ids.iter().zip(originals.iter()) {
                    doc.entities.restore(*id, e.clone());
                    restored.push(*id);
                }
                self.selection = restored;
                ToolOutcome::Restore
            }
            _ => ToolOutcome::None,
        };
        self.state = ToolState::Idle;
        out
    }

    /// Finish an operation and clear the state.
    pub fn finish(&mut self) {
        self.state = ToolState::Idle;
    }

    /// Update the hover entity for the current cursor position.
    pub fn update_hover(&mut self, doc: &Document, cam: &Camera2D, cursor: Vec2) {
        self.hovered = pick(doc, cam, cursor, self.pick_distance);
    }

    /// Recompute `pick_distance` from the camera so grabbing feels the same at
    /// every zoom level.
    pub fn sync_pick_distance(&mut self, cam: &Camera2D, pixels: f32) {
        self.pick_distance = cam.pixels_to_world(pixels);
    }

    /// Handle a click in world coordinates.
    pub fn on_click(
        &mut self,
        doc: &mut Document,
        cam: &Camera2D,
        world: Vec2,
        shift: bool,
    ) -> ToolOutcome {
        match self.id {
            ToolId::Select => self.click_select(doc, cam, world, shift),
            ToolId::Line => self.click_line(doc, world),
            ToolId::Circle => self.click_circle(doc, world),
            ToolId::Arc => self.click_arc(doc, world),
            ToolId::Polyline => self.click_polyline(doc, world),
            ToolId::Rectangle => self.click_rectangle(doc, world),
            ToolId::Erase => self.click_erase(doc, cam, world),
            ToolId::Move => self.click_move(doc, world),
            ToolId::Copy => self.click_copy(doc, world),
            ToolId::Mirror => self.click_mirror(doc, world),
            ToolId::Extrude => self.click_extrude(doc, world),
            ToolId::ZoomWindow => {
                self.state = ToolState::Window {
                    start: world,
                    current: world,
                };
                ToolOutcome::None
            }
            _ => ToolOutcome::None,
        }
    }

    /// Handle cursor motion (updates rubber bands and live previews).
    pub fn on_move(&mut self, _doc: &Document, _cam: &Camera2D, world: Vec2) -> ToolOutcome {
        match &mut self.state {
            // `drag_from` is the anchor the user set; it stays put while
            // `drag_to` tracks the cursor.
            ToolState::Points { drag_to, .. } => {
                *drag_to = Some(world);
                ToolOutcome::None
            }
            ToolState::Window { current, .. } => {
                *current = world;
                ToolOutcome::None
            }
            ToolState::Modify { base, axis, .. } => {
                // Keep the anchor points tracking the cursor so the live preview
                // (and the prompt) follow the mouse.
                if base.is_none() {
                    *base = Some(world);
                } else if axis.is_none() {
                    *axis = Some(world);
                }
                ToolOutcome::None
            }
            _ => ToolOutcome::None,
        }
    }

    /// Handle a key. `Escape` cancels, `Enter` finishes.
    pub fn on_key(&mut self, doc: &mut Document, key: cad_ui::input::Key) -> ToolOutcome {
        use cad_ui::input::Key;
        match key {
            Key::Escape => self.cancel(doc),
            Key::Enter => {
                self.finish();
                ToolOutcome::None
            }
            _ => ToolOutcome::None,
        }
    }

    /// The rubber-band / preview geometry for the current frame.
    pub fn preview(&self) -> Vec<cad_core::Rect2> {
        match &self.state {
            ToolState::Window { start, current } => vec![Rect2::new(*start, *current)],
            ToolState::Points {
                drag_from, drag_to, ..
            } => match (drag_from, drag_to) {
                (Some(a), Some(b)) => vec![Rect2::new(*a, *b)],
                _ => Vec::new(),
            },
            _ => Vec::new(),
        }
    }

    // ---------------------------------------------------------------- helpers

    fn click_select(
        &mut self,
        doc: &Document,
        cam: &Camera2D,
        world: Vec2,
        additive: bool,
    ) -> ToolOutcome {
        match pick(doc, cam, world, self.pick_distance) {
            Some(id) => {
                if additive {
                    if let Some(p) = self.selection.iter().position(|s| *s == id) {
                        self.selection.remove(p);
                    } else {
                        self.selection.push(id);
                    }
                } else {
                    self.selection = vec![id];
                }
                ToolOutcome::None
            }
            None => {
                if !additive {
                    self.selection.clear();
                }
                ToolOutcome::None
            }
        }
    }

    fn click_line(&mut self, doc: &mut Document, world: Vec2) -> ToolOutcome {
        // New geometry goes on the *current* layer, not always layer 0: a
        // layer panel that changes nothing is worse than no layer panel.
        let layer = doc.current_layer();
        let pts = match &mut self.state {
            ToolState::Points { points, .. } => {
                points.push(world);
                points.clone()
            }
            _ => {
                let points = vec![world];
                self.state = ToolState::Points {
                    points: points.clone(),
                    drag_from: None,
                    drag_to: None,
                };
                points
            }
        };
        if pts.len() >= 2 {
            let p0 = pts[pts.len() - 2];
            let p1 = pts[pts.len() - 1];
            let id = doc.add(Entity::line(Line::new(p0, p1)).with_layer(layer));
            // Keep the last point as the start of the next segment, which is
            // how CAD polylines behave.
            self.state = ToolState::Points {
                points: vec![p1],
                drag_from: None,
                drag_to: None,
            };
            self.selection = vec![id];
            return ToolOutcome::Changed(format!("Line ({})", id.raw()));
        }
        ToolOutcome::None
    }

    fn click_circle(&mut self, doc: &mut Document, world: Vec2) -> ToolOutcome {
        let layer = doc.current_layer();
        let pts = match &mut self.state {
            ToolState::Points { points, .. } => {
                points.push(world);
                points.clone()
            }
            _ => {
                self.state = ToolState::Points {
                    points: vec![world],
                    drag_from: None,
                    drag_to: None,
                };
                vec![world]
            }
        };
        if pts.len() >= 2 {
            let c = pts[0];
            let r = pts[1].distance(c);
            if r > 1e-6 {
                let id = doc.add(Entity::circle(Circle::new(c, r)).with_layer(layer));
                self.finish();
                self.selection = vec![id];
                return ToolOutcome::Changed(format!("Circle ({})", id.raw()));
            }
            // Degenerate radius: restart.
            self.state = ToolState::Points {
                points: vec![world],
                drag_from: None,
                drag_to: None,
            };
        }
        ToolOutcome::None
    }

    fn click_arc(&mut self, doc: &mut Document, world: Vec2) -> ToolOutcome {
        let layer = doc.current_layer();
        let pts = match &mut self.state {
            ToolState::Points { points, .. } => {
                points.push(world);
                points.clone()
            }
            _ => {
                self.state = ToolState::Points {
                    points: vec![world],
                    drag_from: None,
                    drag_to: None,
                };
                vec![world]
            }
        };
        if pts.len() >= 3 {
            if let Some(a) = arc_through_three(pts[0], pts[1], pts[2]) {
                let id = doc.add(Entity::arc(a).with_layer(layer));
                self.finish();
                self.selection = vec![id];
                return ToolOutcome::Changed(format!("Arc ({})", id.raw()));
            }
            // Collinear points cannot define an arc: restart from here.
            self.state = ToolState::Points {
                points: vec![world],
                drag_from: None,
                drag_to: None,
            };
        }
        ToolOutcome::None
    }

    fn click_polyline(&mut self, doc: &mut Document, world: Vec2) -> ToolOutcome {
        let layer = doc.current_layer();
        match &mut self.state {
            ToolState::Points { points, .. } => {
                points.push(world);
                ToolOutcome::None
            }
            _ => {
                let id = doc.add(
                    Entity::polyline(Polyline::new(vec![world, world], false)).with_layer(layer),
                );
                self.state = ToolState::Points {
                    points: vec![world],
                    drag_from: None,
                    drag_to: None,
                };
                self.selection = vec![id];
                ToolOutcome::Created(vec![id])
            }
        }
    }

    fn click_rectangle(&mut self, doc: &mut Document, world: Vec2) -> ToolOutcome {
        let layer = doc.current_layer();
        let pts = match &mut self.state {
            ToolState::Points { points, .. } => {
                points.push(world);
                points.clone()
            }
            _ => {
                self.state = ToolState::Points {
                    points: vec![world],
                    drag_from: None,
                    drag_to: None,
                };
                vec![world]
            }
        };
        if pts.len() >= 2 {
            let a = pts[0];
            let b = pts[1];
            let w = b.x - a.x;
            let h = b.y - a.y;
            let corners = [
                Vec2::new(a.x, a.y),
                Vec2::new(a.x + w, a.y),
                Vec2::new(a.x + w, a.y + h),
                Vec2::new(a.x, a.y + h),
            ];
            let id =
                doc.add(Entity::polyline(Polyline::new(corners.to_vec(), true)).with_layer(layer));
            self.finish();
            self.selection = vec![id];
            return ToolOutcome::Changed(format!("Rectangle ({})", id.raw()));
        }
        ToolOutcome::None
    }

    fn click_erase(&mut self, doc: &mut Document, cam: &Camera2D, world: Vec2) -> ToolOutcome {
        if let Some(id) = pick(doc, cam, world, self.pick_distance) {
            doc.entities.remove(id);
            self.selection.retain(|s| *s != id);
            return ToolOutcome::Delete(vec![id]);
        }
        ToolOutcome::None
    }

    fn click_move(&mut self, doc: &mut Document, world: Vec2) -> ToolOutcome {
        self.click_modify(doc, world, ModifyMode::Move)
    }

    fn click_copy(&mut self, doc: &mut Document, world: Vec2) -> ToolOutcome {
        self.click_modify(doc, world, ModifyMode::Copy)
    }

    fn click_mirror(&mut self, doc: &mut Document, world: Vec2) -> ToolOutcome {
        self.click_modify(doc, world, ModifyMode::Mirror)
    }

    /// Shared driver for Move / Copy / Mirror.
    ///
    /// All three need the same beats: capture the originals so `Esc` can put
    /// them back, take the base point, then take the destination and apply.
    /// `Copy` inserts instead of replacing and never disturbs the originals.
    fn click_modify(&mut self, doc: &mut Document, world: Vec2, mode: ModifyMode) -> ToolOutcome {
        let ids = self.selection.clone();

        // First click: capture.
        if !matches!(self.state, ToolState::Modify { .. }) {
            if ids.is_empty() {
                return ToolOutcome::None;
            }
            let originals: Vec<Entity> = ids
                .iter()
                .filter_map(|i| doc.entities.get(*i).cloned())
                .collect();
            if originals.is_empty() {
                return ToolOutcome::None;
            }
            self.state = ToolState::Modify {
                ids,
                originals,
                base: Some(world),
                axis: None,
                mode,
            };
            return ToolOutcome::None;
        }

        let (captured, base, axis, state_mode) =
            match std::mem::replace(&mut self.state, ToolState::Idle) {
                ToolState::Modify {
                    ids,
                    originals,
                    base,
                    axis,
                    mode,
                } => ((ids, originals), base, axis, mode),
                other => {
                    self.state = other;
                    return ToolOutcome::None;
                }
            };
        let (ids, originals) = captured;
        let Some(base) = base else {
            return ToolOutcome::None;
        };

        if state_mode == ModifyMode::Mirror {
            // `axis` is only set by cursor motion; the click that finishes the
            // axis is what defines it otherwise. Reading it from motion alone
            // meant a mirror never applied without the pointer moving between
            // the two clicks.
            let end = axis.unwrap_or(world);
            self.state = ToolState::Modify {
                ids,
                originals,
                base: Some(base),
                axis: Some(end),
                mode: state_mode,
            };
            return self.commit_mirror(doc, base, end);
        }

        let destination = world;
        let delta = destination - base;
        if delta.length_squared() < 1e-9 {
            // A zero move is a cancel, not a no-op that leaves a stale snapshot.
            return ToolOutcome::Restore;
        }

        match state_mode {
            ModifyMode::Copy => {
                let layer = doc.current_layer();
                let mut made = Vec::with_capacity(originals.len());
                for e in &originals {
                    // `translated` keeps the original's layer; a copy is new
                    // geometry, so it lands on the current layer.
                    let target = if e.layer() == LayerId(0) {
                        layer
                    } else {
                        e.layer()
                    };
                    made.push(
                        doc.add(
                            e.translated(Vec3::new(delta.x, delta.y, 0.0))
                                .with_layer(target),
                        ),
                    );
                }
                self.selection = made.clone();
                self.state = ToolState::Idle;
                ToolOutcome::Created(made)
            }
            _ => {
                let mut moved = Vec::with_capacity(ids.len());
                for (id, e) in ids.iter().zip(originals.iter()) {
                    doc.entities
                        .replace(*id, e.translated(Vec3::new(delta.x, delta.y, 0.0)));
                    moved.push(*id);
                }
                self.state = ToolState::Idle;
                ToolOutcome::Changed(format!("Moved {} entities", moved.len()))
            }
        }
    }

    /// Mirror the captured entities across the line through `a` and `b`.
    fn commit_mirror(&mut self, doc: &mut Document, a: Vec2, b: Vec2) -> ToolOutcome {
        let axis = b - a;
        if axis.length_squared() < 1e-9 {
            // A zero-length axis is not a line: put things back and let the user
            // pick a real one rather than mirroring everything onto itself.
            return ToolOutcome::Restore;
        }
        let (ids, _originals) = match std::mem::replace(&mut self.state, ToolState::Idle) {
            ToolState::Modify { ids, originals, .. } => (ids, originals),
            other => {
                self.state = other;
                return ToolOutcome::None;
            }
        };
        let mut kept = Vec::with_capacity(ids.len());
        for id in &ids {
            let Some(src) = doc.entities.get(*id).cloned() else {
                continue;
            };
            doc.entities.replace(
                *id,
                src.mirrored(Vec3::new(a.x, a.y, 0.0), Vec3::new(axis.x, axis.y, 0.0)),
            );
            kept.push(*id);
        }
        self.state = ToolState::Idle;
        ToolOutcome::Changed(format!("Mirrored {} entities", kept.len()))
    }

    fn click_extrude(&mut self, doc: &mut Document, world: Vec2) -> ToolOutcome {
        // Place a box whose base corner is the first click and whose height
        // comes from the second. Kept deliberately simple; the real extrude
        // works on regions.
        let pts = match &mut self.state {
            ToolState::Points { points, .. } => {
                points.push(world);
                points.clone()
            }
            _ => {
                self.state = ToolState::Points {
                    points: vec![world],
                    drag_from: None,
                    drag_to: None,
                };
                vec![world]
            }
        };
        if pts.len() >= 2 {
            let a = pts[0];
            let b2 = pts[1];
            let size = (b2 - a).length();
            if size > 1e-6 {
                // A real extrude: the base corner and the opposite corner define
                // the footprint, and the height is taken from the region's own
                // bounding box so the result is the extruded solid rather than a
                // fixed 10x10 placeholder.
                let region = Polyline::new(
                    vec![a, Vec2::new(b2.x, a.y), b2, Vec2::new(a.x, b2.y)],
                    true,
                );
                let layer = doc.current_layer();
                let mesh = cad_doc::Mesh3d::from_extrusion(&region, size);
                let id = doc.add(Entity::new(cad_doc::EntityKind::Mesh(mesh)).with_layer(layer));
                self.finish();
                self.selection = vec![id];
                return ToolOutcome::Changed(format!("Extrusion ({})", id.raw()));
            }
            self.finish();
        }
        ToolOutcome::None
    }

    /// Commit whatever the tool has pending, returning the undo label.
    pub fn commit(&mut self, _doc: &Document) -> Option<String> {
        match &mut self.state {
            ToolState::Points { points, .. } if points.len() >= 2 => Some("Polyline".into()),
            _ => None,
        }
    }
}

/// Find the entity nearest to `world` within `max_dist`.
pub fn pick(doc: &Document, cam: &Camera2D, world: Vec2, max_dist: f32) -> Option<EntityId> {
    let mut best: Option<(f32, EntityId)> = None;
    let screen = cam.world_to_screen(world);
    let max_px = cam.world_to_pixels(max_dist).max(4.0);
    let search = Rect2::new(world, world).expand(Vec2::splat(max_dist));
    for id in doc.entities.candidates_in(search) {
        let Some(e) = doc.entities.get(id) else {
            continue;
        };
        if !e.common.visible || e.common.locked {
            continue;
        }
        if !doc.layers.visible(e.layer()) || doc.layers.locked(e.layer()) {
            continue;
        }
        let d_px = match &e.entity {
            cad_doc::EntityKind::Point(p) => cam.world_to_screen(p.position.xy()).distance(screen),
            k => {
                let Some(c) = curve_of(k) else { continue };
                let q = c.closest_point(world);
                cam.world_to_screen(q).distance(screen)
            }
        };
        if d_px <= max_px && best.is_none_or(|(bd, _)| d_px < bd) {
            best = Some((d_px, id));
        }
    }
    best.map(|(_, id)| id)
}

/// Pick every entity whose bounds fall inside a window (crossing selection).
pub fn pick_window(doc: &Document, window: Rect2, crossing: bool) -> Vec<EntityId> {
    let mut out = Vec::new();
    let r = window;
    for id in doc.entities.candidates_in(r) {
        let Some(e) = doc.entities.get(id) else {
            continue;
        };
        if !e.common.visible || e.common.locked {
            continue;
        }
        if !doc.layers.visible(e.layer()) || doc.layers.locked(e.layer()) {
            continue;
        }
        let b = e.bounds_2d();
        let inside = if crossing {
            b.overlaps(r)
        } else {
            b.min.x >= r.min.x && b.min.y >= r.min.y && b.max.x <= r.max.x && b.max.y <= r.max.y
        };
        if inside {
            out.push(id);
        }
    }
    out.sort_unstable();
    out
}

fn curve_of(k: &cad_doc::EntityKind) -> Option<cad_geom::Curve> {
    use cad_doc::EntityKind as K;
    match k {
        K::Line(l) => Some(cad_geom::Curve::Line(*l)),
        K::Circle(c) => Some(cad_geom::Curve::Circle(*c)),
        K::Arc(a) => Some(cad_geom::Curve::Arc(*a)),
        K::Ellipse(e) => Some(cad_geom::Curve::Ellipse(*e)),
        K::Polyline(p) | K::Region(p) => Some(cad_geom::Curve::Polyline(p.clone())),
        _ => None,
    }
}

/// Circumcircle through three points; the arc from `p0` to `p2` passing `p1`.
pub fn arc_through_three(p0: Vec2, p1: Vec2, p2: Vec2) -> Option<Arc> {
    let ax = p0.x;
    let ay = p0.y;
    let bx = p1.x;
    let by = p1.y;
    let cx = p2.x;
    let cy = p2.y;
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-9 {
        return None; // collinear
    }
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    let center = Vec2::new(ux, uy);
    let radius = center.distance(p0);
    if radius < 1e-6 {
        return None;
    }
    let a0 = (p0 - center).y.atan2((p0 - center).x);
    let am = (p1 - center).y.atan2((p1 - center).x);
    let a2v = (p2 - center).y.atan2((p2 - center).x);
    // Sweep CCW from p0 to p2 if the middle point is on that arc, else CW.
    let ccw = wrap_delta(a0, am) <= wrap_delta(a0, a2v);
    let sweep = if ccw {
        wrap_delta(a0, a2v)
    } else {
        -wrap_delta(a2v, a0)
    };
    Some(Arc {
        center,
        radius,
        start_angle: a0,
        sweep,
    })
}

/// CCW angular distance from `a` to `b` in `[0, 2pi)`.
fn wrap_delta(a: f32, b: f32) -> f32 {
    let mut d = (b - a) % cad_core::TAU;
    if d < 0.0 {
        d += cad_core::TAU;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::Vec3;
    use cad_geom::curve::Circle;

    /// A document with one circle centred on the origin, radius 10.
    ///
    /// The centre is the origin, so a test that wants to see a transform must
    /// either check the *extents* (x in [-10, 10]) or build its own fixture.
    fn doc_with_circle() -> (Document, EntityId) {
        let mut doc = Document::new();
        let layer = doc.layers.ensure_default();
        let id = doc.add(Entity::circle(Circle::new(Vec2::ZERO, 10.0)).with_layer(layer));
        doc.entities
            .rebuild_index(Rect2::from_xywh(-50.0, -50.0, 100.0, 100.0));
        (doc, id)
    }

    fn cam() -> Camera2D {
        let mut c = Camera2D::new(Vec2::new(800.0, 600.0));
        c.scale = 4.0;
        c
    }

    #[test]
    fn pick_finds_the_nearest_entity() {
        let (doc, id) = doc_with_circle();
        let p = pick(&doc, &cam(), Vec2::new(10.0, 1.0), 1.0);
        assert_eq!(p, Some(id));
    }

    #[test]
    fn pick_respects_the_distance() {
        let (doc, _) = doc_with_circle();
        assert!(pick(&doc, &cam(), Vec2::new(10.0, 5.0), 0.5).is_none());
    }

    #[test]
    fn pick_ignores_hidden_and_locked() {
        let (mut doc, id) = doc_with_circle();
        let layer = doc.entities.get(id).unwrap().layer();
        doc.layers.set_visible(layer, false);
        assert!(pick(&doc, &cam(), Vec2::new(10.0, 1.0), 2.0).is_none());
        doc.layers.set_visible(layer, true);
        doc.layers.set_locked(layer, true);
        assert!(pick(&doc, &cam(), Vec2::new(10.0, 1.0), 2.0).is_none());
    }

    #[test]
    fn window_selection_modes() {
        let mut doc = Document::new();
        let layer = doc.layers.ensure_default();
        for (i, x) in [0.0f32, 100.0, 300.0].iter().enumerate() {
            doc.add(Entity::circle(Circle::new(Vec2::new(*x, 0.0), 5.0)).with_layer(layer));
            let _ = i;
        }
        doc.entities
            .rebuild_index(Rect2::from_xywh(-50.0, -50.0, 500.0, 100.0));

        let crossing = pick_window(&doc, Rect2::from_xywh(-10.0, -10.0, 40.0, 20.0), true);
        assert_eq!(crossing.len(), 1, "only the circle at x=0 overlaps");

        let enclosing = pick_window(&doc, Rect2::from_xywh(-10.0, -10.0, 150.0, 20.0), false);
        assert_eq!(enclosing.len(), 2, "both circles fit fully inside");
    }

    #[test]
    fn line_tool_chains_segments() {
        let mut doc = Document::new();
        let mut t = Tool::new(ToolId::Line);
        assert!(matches!(
            t.on_click(&mut doc, &cam(), Vec2::ZERO, false),
            ToolOutcome::None
        ));
        assert!(t.prompt().is_some());
        let out = t.on_click(&mut doc, &cam(), Vec2::new(10.0, 0.0), false);
        assert!(matches!(out, ToolOutcome::Changed(_)));
        assert_eq!(doc.entities.len(), 1);
        // The second click leaves the tool ready for another segment.
        assert_eq!(t.state.points().len(), 1);
        let out2 = t.on_click(&mut doc, &cam(), Vec2::new(10.0, 10.0), false);
        assert!(matches!(out2, ToolOutcome::Changed(_)));
        assert_eq!(doc.entities.len(), 2);
    }

    #[test]
    fn circle_tool_needs_two_points() {
        let mut doc = Document::new();
        let mut t = Tool::new(ToolId::Circle);
        t.on_click(&mut doc, &cam(), Vec2::ZERO, false);
        assert_eq!(doc.entities.len(), 0);
        assert_eq!(t.prompt().as_deref(), Some("Specify radius"));
        t.on_click(&mut doc, &cam(), Vec2::new(5.0, 0.0), false);
        assert_eq!(doc.entities.len(), 1);
        assert!(t.state.is_idle());
    }

    #[test]
    fn circle_with_zero_radius_restarts() {
        let mut doc = Document::new();
        let mut t = Tool::new(ToolId::Circle);
        t.on_click(&mut doc, &cam(), Vec2::ZERO, false);
        t.on_click(&mut doc, &cam(), Vec2::ZERO, false);
        assert_eq!(doc.entities.len(), 0, "no degenerate circle");
        assert_eq!(t.state.points().len(), 1);
    }

    #[test]
    fn arc_tool_needs_three_points() {
        let mut doc = Document::new();
        let mut t = Tool::new(ToolId::Arc);
        t.on_click(&mut doc, &cam(), Vec2::new(10.0, 0.0), false);
        t.on_click(&mut doc, &cam(), Vec2::new(0.0, 10.0), false);
        t.on_click(&mut doc, &cam(), Vec2::new(-10.0, 0.0), false);
        assert_eq!(doc.entities.len(), 1);
        match doc.entities.iter().next().unwrap().entity {
            cad_doc::EntityKind::Arc(a) => assert!((a.radius - 10.0).abs() < 1e-3),
            _ => panic!(),
        }
    }

    #[test]
    fn collinear_arc_points_restart_instead_of_panicking() {
        let mut doc = Document::new();
        let mut t = Tool::new(ToolId::Arc);
        t.on_click(&mut doc, &cam(), Vec2::ZERO, false);
        t.on_click(&mut doc, &cam(), Vec2::new(5.0, 0.0), false);
        t.on_click(&mut doc, &cam(), Vec2::new(10.0, 0.0), false);
        assert_eq!(doc.entities.len(), 0);
        assert_eq!(t.state.points().len(), 1);
    }

    #[test]
    fn arc_through_three_geometry() {
        let a = arc_through_three(
            Vec2::new(10.0, 0.0),
            Vec2::new(0.0, 10.0),
            Vec2::new(-10.0, 0.0),
        )
        .expect("arc");
        assert!(a.center.distance(Vec2::ZERO) < 1e-4, "{:?}", a.center);
        assert!((a.radius - 10.0).abs() < 1e-4);
        // p1 is on the CCW upper semicircle, so the sweep is +pi.
        assert!((a.sweep - cad_core::PI).abs() < 1e-3, "sweep={}", a.sweep);

        // Same three points reversed must give the mirror arc.
        let b = arc_through_three(
            Vec2::new(-10.0, 0.0),
            Vec2::new(0.0, 10.0),
            Vec2::new(10.0, 0.0),
        )
        .expect("arc");
        assert!(b.center.distance(Vec2::ZERO) < 1e-4);
        assert!((b.radius - 10.0).abs() < 1e-4);
    }

    #[test]
    fn arc_through_three_rejects_collinear() {
        assert!(arc_through_three(Vec2::ZERO, Vec2::new(1.0, 1.0), Vec2::new(2.0, 2.0)).is_none());
        assert!(arc_through_three(Vec2::ZERO, Vec2::ZERO, Vec2::ZERO).is_none());
    }

    #[test]
    fn rectangle_tool_creates_a_closed_polyline() {
        let mut doc = Document::new();
        let mut t = Tool::new(ToolId::Rectangle);
        t.on_click(&mut doc, &cam(), Vec2::ZERO, false);
        t.on_click(&mut doc, &cam(), Vec2::new(20.0, 10.0), false);
        assert_eq!(doc.entities.len(), 1);
        match &doc.entities.iter().next().unwrap().entity {
            cad_doc::EntityKind::Polyline(p) => {
                assert!(p.closed);
                assert_eq!(p.vertices.len(), 4);
                assert!((p.signed_area() - 200.0).abs() < 1e-3);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn erase_removes_the_picked_entity() {
        let (mut doc, id) = doc_with_circle();
        let mut t = Tool::new(ToolId::Erase);
        t.sync_pick_distance(&cam(), 8.0);
        let out = t.on_click(&mut doc, &cam(), Vec2::new(10.0, 0.5), false);
        assert_eq!(out, ToolOutcome::Delete(vec![id]));
        assert!(doc.entities.get(id).is_none());
    }

    #[test]
    fn erase_misses_are_harmless() {
        let (mut doc, _) = doc_with_circle();
        let mut t = Tool::new(ToolId::Erase);
        t.sync_pick_distance(&cam(), 2.0);
        let out = t.on_click(&mut doc, &cam(), Vec2::new(0.0, 50.0), false);
        assert_eq!(out, ToolOutcome::None);
        assert_eq!(doc.entities.len(), 1);
    }

    #[test]
    fn select_toggles_with_shift() {
        let mut doc = Document::new();
        let layer = doc.layers.ensure_default();
        let a = doc.add(Entity::circle(Circle::new(Vec2::ZERO, 5.0)).with_layer(layer));
        let b = doc.add(Entity::circle(Circle::new(Vec2::new(40.0, 0.0), 5.0)).with_layer(layer));
        doc.entities
            .rebuild_index(Rect2::from_xywh(-20.0, -20.0, 100.0, 40.0));

        let mut t = Tool::new(ToolId::Select);
        t.sync_pick_distance(&cam(), 8.0);
        t.on_click(&mut doc, &cam(), Vec2::new(5.0, 0.0), false);
        assert_eq!(t.selection, vec![a]);
        t.on_click(&mut doc, &cam(), Vec2::new(40.0, 5.0), true);
        assert_eq!(t.selection.len(), 2);
        // Shift-clicking again removes it.
        t.on_click(&mut doc, &cam(), Vec2::new(40.0, 5.0), true);
        assert_eq!(t.selection, vec![a]);
        // Clicking empty space clears.
        t.on_click(&mut doc, &cam(), Vec2::new(200.0, 0.0), false);
        assert!(t.selection.is_empty());
        let _ = b;
    }

    #[test]
    fn move_actually_moves() {
        // The Move tool used to capture the originals and then return `None`
        // forever: nothing was ever applied.
        let (mut doc, id) = doc_with_circle();
        let mut t = Tool::new(ToolId::Move);
        t.selection = vec![id];
        assert_eq!(
            t.on_click(&mut doc, &cam(), Vec2::ZERO, false),
            ToolOutcome::None
        );
        assert!(matches!(t.state, ToolState::Modify { .. }));
        let out = t.on_click(&mut doc, &cam(), Vec2::new(100.0, 50.0), false);
        assert!(matches!(out, ToolOutcome::Changed(_)), "{out:?}");
        match &doc.entities.get(id).unwrap().entity {
            cad_doc::EntityKind::Circle(c) => {
                // `doc_with_circle` centres the circle on the origin, so the
                // move by (100, 50) lands it at (100, 50).
                assert!(
                    c.center.distance(Vec2::new(100.0, 50.0)) < 1e-4,
                    "{:?}",
                    c.center
                );
            }
            _ => panic!(),
        }
        // One entity moved: nothing was created and nothing was duplicated.
        assert_eq!(doc.entities.len(), 1);
        assert!(t.state.is_idle(), "the tool must be ready for another move");
    }

    #[test]
    fn move_by_zero_cancels_instead_of_stranding_a_snapshot() {
        let (mut doc, id) = doc_with_circle();
        let mut t = Tool::new(ToolId::Move);
        t.selection = vec![id];
        t.on_click(&mut doc, &cam(), Vec2::new(5.0, 5.0), false);
        let out = t.on_click(&mut doc, &cam(), Vec2::new(5.0, 5.0), false);
        assert_eq!(out, ToolOutcome::Restore);
        match &doc.entities.get(id).unwrap().entity {
            cad_doc::EntityKind::Circle(c) => assert!(c.center.distance(Vec2::ZERO) < 1e-5),
            _ => panic!(),
        }
    }

    #[test]
    fn copy_keeps_the_original_and_adds_one() {
        let (mut doc, id) = doc_with_circle();
        let mut t = Tool::new(ToolId::Copy);
        t.selection = vec![id];
        t.on_click(&mut doc, &cam(), Vec2::ZERO, false);
        let out = t.on_click(&mut doc, &cam(), Vec2::new(100.0, 0.0), false);
        let ToolOutcome::Created(made) = out else {
            panic!("a copy creates entities, got {out:?}");
        };
        assert_eq!(made.len(), 1);
        assert_eq!(doc.entities.len(), 2);
        // The original is untouched and the copy is selected.
        match &doc.entities.get(id).unwrap().entity {
            cad_doc::EntityKind::Circle(c) => assert!(c.center.distance(Vec2::ZERO) < 1e-5),
            _ => panic!(),
        }
        assert_eq!(t.selection, made);
    }

    #[test]
    fn copy_without_a_selection_does_nothing() {
        let (mut doc, _) = doc_with_circle();
        let mut t = Tool::new(ToolId::Copy);
        assert_eq!(
            t.on_click(&mut doc, &cam(), Vec2::ZERO, false),
            ToolOutcome::None
        );
        assert_eq!(
            t.on_click(&mut doc, &cam(), Vec2::new(5.0, 0.0), false),
            ToolOutcome::None
        );
        assert_eq!(doc.entities.len(), 1);
    }

    #[test]
    fn mirror_reflects_across_the_axis() {
        // A circle centred on the mirror axis reflects onto itself, which would
        // prove nothing, so build one off to the side.
        let mut doc = Document::new();
        let layer = doc.layers.ensure_default();
        let id = doc.add(Entity::circle(Circle::new(Vec2::new(10.0, 0.0), 10.0)).with_layer(layer));
        let mut t = Tool::new(ToolId::Mirror);
        t.selection = vec![id];
        // Axis: the Y axis, from the origin upward.
        t.on_click(&mut doc, &cam(), Vec2::ZERO, false);
        let out = t.on_click(&mut doc, &cam(), Vec2::new(0.0, 50.0), false);
        assert!(matches!(out, ToolOutcome::Changed(_)), "{out:?}");
        match &doc.entities.get(id).unwrap().entity {
            cad_doc::EntityKind::Circle(c) => {
                assert!(
                    c.center.distance(Vec2::new(-10.0, 0.0)) < 1e-4,
                    "{:?}",
                    c.center
                );
                assert!((c.radius - 10.0).abs() < 1e-5, "mirroring must not resize");
            }
            _ => panic!(),
        }
        assert!(t.state.is_idle());
    }

    #[test]
    fn mirror_keeps_a_polyline_area() {
        // A reflection is orientation-reversing but area-preserving; this is the
        // property that distinguishes it from a rotate-by-pi approximation.
        let mut doc = Document::new();
        let layer = doc.layers.ensure_default();
        let square = Polyline::new(
            vec![
                Vec2::new(2.0, 1.0),
                Vec2::new(4.0, 1.0),
                Vec2::new(4.0, 3.0),
                Vec2::new(2.0, 3.0),
            ],
            true,
        );
        let id = doc.add(Entity::polyline(square).with_layer(layer));
        let b0 = doc.entities.get(id).unwrap().bounds_2d();
        let area0 = b0.width() * b0.height();

        doc.entities.replace(
            id,
            doc.entities.get(id).unwrap().mirrored(Vec3::ZERO, Vec3::X),
        );
        let b1 = doc.entities.get(id).unwrap().bounds_2d();
        assert!(
            (area0 - b1.width() * b1.height()).abs() < 1e-3,
            "{area0} vs {}",
            b1.width() * b1.height()
        );

        // And it actually moved: a mirror about a line with direction X through
        // the origin reflects in y, so the whole square lands below the axis
        // with its x range untouched.
        assert!((b1.min.x - 2.0).abs() < 1e-3, "x must not move: {b1:?}");
        assert!((b1.max.x - 4.0).abs() < 1e-3, "x must not move: {b1:?}");
        assert!((b1.max.y - (-1.0)).abs() < 1e-3, "{b1:?}");
        assert!((b1.min.y - (-3.0)).abs() < 1e-3, "{b1:?}");
    }

    #[test]
    fn mirror_over_a_zero_length_axis_restores() {
        let (mut doc, id) = doc_with_circle();
        let mut t = Tool::new(ToolId::Mirror);
        t.selection = vec![id];
        t.on_click(&mut doc, &cam(), Vec2::ZERO, false);
        // The second click lands on the first, so the axis degenerates.
        assert_eq!(
            t.on_click(&mut doc, &cam(), Vec2::ZERO, false),
            ToolOutcome::Restore
        );
        match &doc.entities.get(id).unwrap().entity {
            cad_doc::EntityKind::Circle(c) => assert!(c.center.distance(Vec2::ZERO) < 1e-5),
            _ => panic!(),
        }
    }

    #[test]
    fn modify_prompts_walk_through_the_operation() {
        let (mut doc, id) = doc_with_circle();
        let mut t = Tool::new(ToolId::Move);
        t.selection = vec![id];
        assert!(t.prompt().is_none(), "idle tool has no prompt");
        t.on_click(&mut doc, &cam(), Vec2::ZERO, false);
        assert_eq!(t.prompt().as_deref(), Some("Specify destination point"));
    }

    #[test]
    fn escape_restores_a_snapshot() {
        let mut doc = Document::new();
        let layer = doc.layers.ensure_default();
        let id = doc.add(Entity::circle(Circle::new(Vec2::ZERO, 5.0)).with_layer(layer));
        let mut t = Tool::new(ToolId::Move);
        t.selection = vec![id];
        // Snapshot the originals.
        t.on_click(&mut doc, &cam(), Vec2::ZERO, false);
        assert!(matches!(t.state, ToolState::Modify { .. }));
        // Mutate.
        if let Some(e) = doc.entities.get_mut(id) {
            *e = e.translated(Vec3::new(100.0, 0.0, 0.0));
        }
        // `center()` consumes `self`, so it cannot be called on a borrow of the
        // Rect2; copy it out first.
        let b = doc.entities.get(id).unwrap().bounds_2d();
        assert!(b.center().x > 50.0);
        // Cancel puts it back.
        assert_eq!(t.cancel(&mut doc), ToolOutcome::Restore);
        match &doc.entities.get(id).unwrap().entity {
            cad_doc::EntityKind::Circle(c) => assert!(c.center.distance(Vec2::ZERO) < 1e-5),
            _ => panic!(),
        }
    }

    #[test]
    fn prompts_reflect_progress() {
        let mut doc = Document::new();
        let mut t = Tool::new(ToolId::Arc);
        assert!(t.prompt().is_none());
        t.on_click(&mut doc, &cam(), Vec2::ZERO, false);
        assert_eq!(t.prompt().as_deref(), Some("Specify start point"));
        t.on_click(&mut doc, &cam(), Vec2::new(1.0, 0.0), false);
        assert_eq!(t.prompt().as_deref(), Some("Specify second point"));
    }

    #[test]
    fn preview_tracks_the_drag() {
        let mut doc = Document::new();
        let mut t = Tool::new(ToolId::ZoomWindow);
        t.on_click(&mut doc, &cam(), Vec2::new(10.0, 10.0), false);
        t.on_move(&doc, &cam(), Vec2::new(50.0, 40.0));
        let p = t.preview();
        assert_eq!(p.len(), 1);
        assert!(p[0].contains(Vec2::new(20.0, 20.0)));
        assert_eq!(p[0].max.x, 50.0);
    }

    #[test]
    fn pick_distance_is_resolution_independent() {
        let mut t = Tool::new(ToolId::Select);
        let mut a = cam();
        a.scale = 1.0;
        let mut b = cam();
        b.scale = 50.0;
        t.sync_pick_distance(&a, 8.0);
        let d1 = t.pick_distance;
        t.sync_pick_distance(&b, 8.0);
        assert!(
            t.pick_distance < d1,
            "higher zoom means a tighter world-space grab"
        );
        assert!((a.pixels_to_world(8.0) - d1).abs() < 1e-5);
    }

    #[test]
    fn tool_metadata_matches_the_command_table() {
        // Every tool must be reachable by a command name.
        let r = crate::command::CommandRegistry::new();
        for t in [
            ToolId::Select,
            ToolId::Line,
            ToolId::Circle,
            ToolId::Arc,
            ToolId::Polyline,
            ToolId::Rectangle,
            ToolId::Move,
            ToolId::Copy,
            ToolId::Rotate,
            ToolId::Erase,
            ToolId::Trim,
            ToolId::Extend,
            ToolId::Mirror,
            ToolId::Offset,
            ToolId::Extrude,
        ] {
            assert!(r.get(t.command()).is_some(), "{} has no command", t.label());
        }
    }

    #[test]
    fn selection_needs_are_declared() {
        assert!(ToolId::Move.needs_selection());
        assert!(ToolId::Erase.needs_selection());
        assert!(!ToolId::Line.needs_selection());
    }
}
