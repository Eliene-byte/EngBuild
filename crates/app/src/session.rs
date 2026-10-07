//! The session: everything mutable about one open drawing.
//!
//! A `Session` owns the document, the undo history, the cameras, the active
//! tool and the selection. The window loop owns nothing but the window.

use crate::command::{CommandRegistry, CommandResult};
use crate::tools::{Tool, ToolId, ToolOutcome, pick_window};
use cad_core::{Camera2D, Camera3D, Rect2, Vec2, Vec3};
use cad_doc::{Document, EntityId, History};

/// 2D drafting or 3D modelling view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    /// Plan view: orthographic top-down, everything is flat.
    Model2d,
    /// Perspective model-space orbit.
    Model3d,
}

/// One line on the command line / status bar.
#[derive(Debug, Clone, PartialEq)]
pub struct StatusMessage {
    pub text: String,
    /// `error` and `warning` colour the text.
    pub level: Level,
    /// Seconds remaining before it fades; `0` means sticky.
    pub ttl: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Info,
    Success,
    Warning,
    Error,
}

impl StatusMessage {
    pub fn info(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            level: Level::Info,
            ttl: 4.0,
        }
    }
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            level: Level::Error,
            ttl: 6.0,
        }
    }
    pub fn success(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            level: Level::Success,
            ttl: 3.0,
        }
    }
    /// A prompt that stays until the next command.
    pub fn prompt(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            level: Level::Info,
            ttl: 0.0,
        }
    }
    pub fn is_sticky(&self) -> bool {
        self.ttl <= 0.0
    }
}

/// Grid and snap settings that belong to the session, not the document.
#[derive(Debug, Clone, PartialEq)]
pub struct Viewport {
    pub cam2d: Camera2D,
    pub cam3d: Camera3d,
    pub mode: ViewMode,
    pub show_grid: bool,
    pub grid_spacing: f32,
    pub show_axes: bool,
    /// Rectangle occupied by the drawing area, excluding panels.
    pub canvas: Rect2,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            cam2d: Camera2D::default(),
            cam3d: Camera3d::default(),
            mode: ViewMode::Model2d,
            show_grid: true,
            grid_spacing: 10.0,
            show_axes: true,
            canvas: Rect2::from_xywh(0.0, 0.0, 1280.0, 720.0),
        }
    }
}

impl Viewport {
    pub fn cam(&self) -> &Camera2D {
        &self.cam2d
    }
    /// Pixel size of a line, in CSS pixels, independent of zoom.
    pub fn hairline(&self, dpr: f32) -> f32 {
        1.0 / dpr.max(1.0)
    }
}

/// One open drawing plus its UI state.
pub struct Session {
    pub doc: Document,
    pub history: History,
    pub commands: CommandRegistry,
    pub viewport: Viewport,
    pub tool: Tool,
    pub snap: cad_snap::SnapEngine,
    pub status: StatusMessage,
    pub cursor_world: Vec2,
    /// The last fixed point, used for polar/ortho tracking.
    pub tracking_base: Option<Vec2>,
    /// True while the user is dragging a selection window.
    pub window_drag: Option<(Vec2, Vec2)>,
    /// True when the document changed since the last frame.
    pub dirty: bool,
    /// Path of the file being edited, if any.
    pub path: Option<std::path::PathBuf>,
    pub theme: cad_ui::Theme,
    /// Time accumulator, for animated UI.
    pub time: f32,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        Self {
            doc: Document::new(),
            history: History::new(),
            commands: CommandRegistry::new(),
            viewport: Viewport::default(),
            tool: Tool::new(ToolId::Select),
            snap: cad_snap::SnapEngine::new(),
            status: StatusMessage::success("Ready"),
            cursor_world: Vec2::ZERO,
            tracking_base: None,
            window_drag: None,
            dirty: true,
            path: None,
            theme: cad_ui::Theme::dark(),
            time: 0.0,
        }
    }

    // ------------------------------------------------------------- navigation

    /// Fit the drawing into the canvas, with a margin.
    pub fn zoom_extents(&mut self) {
        let Some(r) = self
            .doc
            .compute_extents()
            .map(|(bb, _)| Rect2::new(bb.min.xy(), bb.max.xy()))
        else {
            self.viewport.cam2d.zoom_extents(Vec2::ZERO, 100.0);
            return;
        };
        if r.is_empty() {
            self.viewport.cam2d.zoom_extents(r.center(), 100.0);
        } else {
            self.viewport.cam2d.fit(r, 40.0);
        }
        self.dirty = true;
    }

    pub fn zoom_by(&mut self, factor: f32) {
        let focus = self
            .viewport
            .cam2d
            .screen_to_world(self.viewport.canvas.center());
        let screen = self.viewport.canvas.center();
        self.viewport.cam2d.zoom_at(factor, screen);
        let _ = focus;
        self.dirty = true;
    }

    pub fn zoom_to_rect(&mut self, r: Rect2) {
        if r.is_empty() {
            return;
        }
        self.viewport.cam2d.fit(r, 20.0);
        self.dirty = true;
    }

    pub fn pan_screen(&mut self, delta: Vec2) {
        self.viewport.cam2d.pan_screen(delta);
        self.dirty = true;
    }

    /// Frame the given entities.
    pub fn zoom_to_entities(&mut self, ids: &[EntityId]) {
        let mut r = Rect2::ZERO;
        let mut first = true;
        for id in ids {
            if let Some(e) = self.doc.entities.get(*id) {
                let b = e.bounds_2d();
                if first {
                    r = b;
                    first = false;
                } else {
                    r = r.union(b);
                }
            }
        }
        if !first {
            self.zoom_to_rect(r);
        }
    }

    pub fn toggle_grid(&mut self) -> CommandResult {
        self.viewport.show_grid = !self.viewport.show_grid;
        self.status = StatusMessage::info(format!(
            "Grid {}",
            if self.viewport.show_grid { "on" } else { "off" }
        ));
        CommandResult::Ok
    }

    pub fn toggle_snap(&mut self) -> CommandResult {
        self.snap.toggle();
        self.status = StatusMessage::info(format!(
            "Object snap {}",
            if self.snap.settings.enabled {
                "on"
            } else {
                "off"
            }
        ));
        CommandResult::Ok
    }

    pub fn toggle_ortho(&mut self) -> CommandResult {
        use cad_snap::SnapKind;
        let on = !self.snap.settings.has(SnapKind::Ortho);
        self.snap.settings.set(SnapKind::Ortho, on);
        self.status = StatusMessage::info(format!("Ortho mode {}", if on { "on" } else { "off" }));
        CommandResult::Ok
    }

    pub fn toggle_polar(&mut self) -> CommandResult {
        use cad_snap::SnapKind;
        let on = !self.snap.settings.has(SnapKind::Polar);
        self.snap.settings.set(SnapKind::Polar, on);
        self.status =
            StatusMessage::info(format!("Polar tracking {}", if on { "on" } else { "off" }));
        CommandResult::Ok
    }

    pub fn toggle_3d(&mut self) -> CommandResult {
        self.viewport.mode = match self.viewport.mode {
            ViewMode::Model2d => {
                // Frame the current 2D extents in the 3D view.
                if let Some((bb, c)) = self.doc.compute_extents() {
                    let r = bb.size().x.max(bb.size().y).max(1.0) * 2.0;
                    self.viewport.cam3d.target = c;
                    self.viewport.cam3d.eye = c + Vec3::new(r * 0.5, -r * 0.5, r * 0.5);
                }
                ViewMode::Model3d
            }
            ViewMode::Model3d => ViewMode::Model2d,
        };
        self.status = StatusMessage::info(match self.viewport.mode {
            ViewMode::Model2d => "2D drafting view",
            ViewMode::Model3d => "3D model view",
        });
        CommandResult::Ok
    }

    // ------------------------------------------------------------- selection

    pub fn select_all(&mut self) {
        self.tool.selection = self.doc.entities.handles();
        let n = self.tool.selection.len();
        self.status = StatusMessage::info(format!("{n} selected"));
    }

    pub fn clear_selection(&mut self) {
        self.tool.selection.clear();
    }

    pub fn delete_selection(&mut self) -> CommandResult {
        if self.tool.selection.is_empty() {
            self.status = StatusMessage::error("Nothing selected");
            return CommandResult::Unavailable;
        }
        let ids = self.tool.selection.clone();
        let mut h = std::mem::take(&mut self.history);
        {
            let mut tx = h.begin(
                &mut self.doc.entities,
                &format!("Delete {} entities", ids.len()),
            );
            for id in &ids {
                tx.remove(*id);
            }
            tx.commit();
        }
        self.history = h;
        self.tool.selection.clear();
        self.doc.invalidate_extents();
        self.dirty = true;
        self.status = StatusMessage::success(format!("Deleted {} entities", ids.len()));
        CommandResult::Ok
    }

    pub fn undo(&mut self) -> CommandResult {
        let layers = self.doc.layers.clone();
        let blocks = self.doc.blocks.clone();
        if !self.history.can_undo() {
            self.status = StatusMessage::error("Nothing to undo");
            return CommandResult::Unavailable;
        }
        let label = self.history.undo_label().unwrap_or("").to_string();
        self.history.undo(&mut self.doc.entities);
        self.doc.layers = layers;
        self.doc.blocks = blocks;
        self.doc.invalidate_extents();
        self.tool.selection.clear();
        self.dirty = true;
        self.status = StatusMessage::info(format!("Undo {label}"));
        CommandResult::Ok
    }

    pub fn redo(&mut self) -> CommandResult {
        let layers = self.doc.layers.clone();
        let blocks = self.doc.blocks.clone();
        if !self.history.can_redo() {
            self.status = StatusMessage::error("Nothing to redo");
            return CommandResult::Unavailable;
        }
        let label = self.history.redo_label().unwrap_or("").to_string();
        self.history.redo(&mut self.doc.entities);
        self.doc.layers = layers;
        self.doc.blocks = blocks;
        self.doc.invalidate_extents();
        self.tool.selection.clear();
        self.dirty = true;
        self.status = StatusMessage::info(format!("Redo {label}"));
        CommandResult::Ok
    }

    // ------------------------------------------------------------------ tools

    /// Activate the tool bound to `command`.
    pub fn activate(&mut self, command: &str) -> CommandResult {
        let Some(t) = tool_for_command(command) else {
            self.status = StatusMessage::error(format!("Unknown command: {command}"));
            return CommandResult::Error(format!("Unknown command: {command}"));
        };
        if t.needs_selection() && self.tool.selection.is_empty() && t != ToolId::Erase {
            self.status = StatusMessage::error("Select objects first");
            return CommandResult::Unavailable;
        }
        self.tool = Tool::new(t);
        if let Some(p) = self.tool.prompt() {
            self.status = StatusMessage::prompt(p);
        } else {
            self.status = StatusMessage::info(format!("{} command active", t.label()));
        }
        self.dirty = true;
        CommandResult::Ok
    }

    /// Update hover and snap for the current cursor.
    pub fn update_cursor(&mut self) {
        let cam = *self.viewport.cam();
        self.tool.sync_pick_distance(&cam, 8.0);
        self.tool.update_hover(&self.doc, &cam, self.cursor_world);
        let s = self
            .snap
            .resolve(&self.doc, &cam, self.cursor_world, self.tracking_base);
        if let Some(s) = s {
            self.cursor_world = s.point;
            if let Some(e) = self.tool.prompt() {
                self.status = StatusMessage::prompt(e);
            }
        }
    }

    /// Apply a tool outcome to the document and status bar.
    pub fn apply_outcome(&mut self, outcome: ToolOutcome) {
        match outcome {
            ToolOutcome::None => {}
            ToolOutcome::Changed(label) => {
                self.doc.invalidate_extents();
                self.dirty = true;
                self.status = StatusMessage::success(label);
            }
            ToolOutcome::Created(ids) => {
                self.doc.invalidate_extents();
                self.dirty = true;
                self.status = StatusMessage::success(format!("Created {} entities", ids.len()));
            }
            ToolOutcome::Delete(ids) => {
                self.doc.invalidate_extents();
                self.dirty = true;
                self.status = StatusMessage::success(format!("Erased {}", ids.len()));
                let _ = ids;
            }
            ToolOutcome::ZoomTo(r) => self.zoom_to_rect(r),
            ToolOutcome::Restore => {
                self.doc.invalidate_extents();
                self.dirty = true;
                self.status = StatusMessage::info("Cancelled");
            }
        }
        if let Some(p) = self.tool.prompt() {
            self.status = StatusMessage::prompt(p);
        }
    }

    /// Click in the canvas (already converted to world coordinates).
    pub fn click_world(&mut self, world: Vec2, shift: bool) {
        let cam = *self.viewport.cam();
        let outcome = self.tool.on_click(&mut self.doc, &cam, world, shift);
        self.apply_outcome(outcome);
    }

    /// Finish the current selection window.
    pub fn finish_window(&mut self) {
        if let Some((a, b)) = self.window_drag.take() {
            let r = Rect2::new(a, b);
            let crossing = self.viewport.cam().world_per_pixel() * 0.0 < 0.0; // left-to-right = window
            let ids = pick_window(&self.doc, r, !crossing);
            self.tool.selection = ids.clone();
            self.status = StatusMessage::info(format!("{} selected", ids.len()));
        }
    }

    /// Frame time bookkeeping.
    pub fn tick(&mut self, dt: f32) {
        self.time += dt;
        if !self.status.is_sticky() {
            self.status.ttl -= dt;
            if self.status.ttl <= 0.0 {
                self.status = StatusMessage::info("Ready");
            }
        }
    }

    /// Entity counts per kind, for the status bar.
    pub fn entity_summary(&self) -> (usize, usize) {
        let mut planar = 0;
        let mut solid = 0;
        for e in self.doc.entities.iter() {
            if e.is_drawable_2d() {
                planar += 1;
            } else {
                solid += 1;
            }
        }
        (planar, solid)
    }
}

/// Map a command name to a tool.
pub fn tool_for_command(command: &str) -> Option<ToolId> {
    Some(match command.to_ascii_lowercase().as_str() {
        "select" | "_select" => ToolId::Select,
        "pan" | "panview" => ToolId::Pan,
        "zoomwindow" | "zwin" => ToolId::ZoomWindow,
        "line" | "l" => ToolId::Line,
        "circle" | "c" => ToolId::Circle,
        "arc" | "a" => ToolId::Arc,
        "polyline" | "pl" => ToolId::Polyline,
        "rectangle" | "rec" => ToolId::Rectangle,
        "move" | "m" => ToolId::Move,
        "copytool" | "copy" | "cp" => ToolId::Copy,
        "rotate" | "ro" => ToolId::Rotate,
        "scale" | "sc" => ToolId::Scale,
        "erase" | "e" | "del" => ToolId::Erase,
        "trim" | "tr" => ToolId::Trim,
        "extend" | "ex" => ToolId::Extend,
        "mirror" | "mi" => ToolId::Mirror,
        "offset" | "o" => ToolId::Offset,
        "extrude" => ToolId::Extrude,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_geom::curve::{Circle, Line};

    /// A session with one circle at the origin, index built so picking works.
    fn session_with_circle() -> Session {
        let mut s = Session::new();
        let layer = s.doc.layers.ensure_default();
        s.doc
            .add(cad_doc::Entity::circle(Circle::new(Vec2::ZERO, 10.0)).with_layer(layer));
        s.doc
            .entities
            .rebuild_index(Rect2::from_xywh(-50.0, -50.0, 200.0, 200.0));
        s
    }

    fn session_with_line_and_circle() -> Session {
        let mut s = Session::new();
        let layer = s.doc.layers.ensure_default();
        s.doc.add(
            cad_doc::Entity::line(Line::new(Vec2::new(-5.0, 20.0), Vec2::new(5.0, 20.0)))
                .with_layer(layer),
        );
        s.doc
            .entities
            .rebuild_index(Rect2::from_xywh(-50.0, -50.0, 200.0, 200.0));
        s
    }

    #[test]
    fn new_session_is_ready() {
        let s = Session::new();
        assert!(s.status.text.contains("Ready"));
        assert_eq!(s.viewport.mode, ViewMode::Model2d);
        assert!(s.tool.selection.is_empty());
        assert!(!s.commands.is_empty());
    }

    #[test]
    fn zoom_extents_fits_the_drawing() {
        let mut s = session_with_circle();
        s.viewport.cam2d = Camera2D::new(Vec2::new(800.0, 600.0));
        s.zoom_extents();
        let v = s.viewport.cam2d.world_viewport();
        assert!(v.contains(Vec2::new(10.0, 0.0)), "{v:?}");
        assert!(v.contains(Vec2::new(-10.0, 0.0)), "{v:?}");
    }

    #[test]
    fn zoom_extents_on_an_empty_document_is_safe() {
        let mut s = Session::new();
        s.zoom_extents();
        assert!(s.viewport.cam2d.scale.is_finite());
    }

    #[test]
    fn toggles_flip_and_report() {
        let mut s = Session::new();
        let g = s.viewport.show_grid;
        s.toggle_grid();
        assert_ne!(s.viewport.show_grid, g);
        assert!(s.status.text.contains("Grid"));

        let snap_on = s.snap.settings.enabled;
        s.toggle_snap();
        assert_ne!(s.snap.settings.enabled, snap_on);
    }

    #[test]
    fn ortho_and_polar_toggle() {
        let mut s = Session::new();
        use cad_snap::SnapKind;
        assert!(s.snap.settings.has(SnapKind::Ortho));
        s.toggle_ortho();
        assert!(!s.snap.settings.has(SnapKind::Ortho));
        s.toggle_polar();
        assert!(!s.snap.settings.has(SnapKind::Polar));
    }

    #[test]
    fn three_d_view_toggle() {
        let mut s = session_with_circle();
        s.toggle_3d();
        assert_eq!(s.viewport.mode, ViewMode::Model3d);
        // The camera must have been framed on the drawing.
        assert!(s.viewport.cam3d.eye.distance(s.viewport.cam3d.target) > 1.0);
        s.toggle_3d();
        assert_eq!(s.viewport.mode, ViewMode::Model2d);
    }

    #[test]
    fn select_all_and_clear() {
        let mut s = session_with_circle();
        s.select_all();
        assert_eq!(s.tool.selection.len(), 1);
        s.clear_selection();
        assert!(s.tool.selection.is_empty());
    }

    #[test]
    fn delete_selection_is_undoable() {
        let mut s = session_with_circle();
        s.select_all();
        let n = s.doc.entities.len();
        assert!(s.delete_selection().is_ok());
        assert_eq!(s.doc.entities.len(), n - 1);
        assert!(s.undo().is_ok());
        assert_eq!(s.doc.entities.len(), n, "undo must bring the entity back");
    }

    #[test]
    fn delete_without_selection_reports_an_error() {
        let mut s = session_with_circle();
        assert_eq!(s.delete_selection(), CommandResult::Unavailable);
        assert_eq!(s.status.level, Level::Error);
    }

    #[test]
    fn undo_without_history_reports_an_error() {
        let mut s = Session::new();
        assert_eq!(s.undo(), CommandResult::Unavailable);
        assert_eq!(s.redo(), CommandResult::Unavailable);
    }

    #[test]
    fn undo_redo_round_trip() {
        let mut s = session_with_circle();
        let n = s.doc.entities.len();
        s.select_all();
        s.delete_selection();
        assert!(s.redo().is_ok());
        assert_eq!(s.doc.entities.len(), n);
        assert!(s.undo().is_ok());
        assert_eq!(s.doc.entities.len(), n - 1);
    }

    #[test]
    fn activate_rejects_unknown_commands() {
        let mut s = Session::new();
        let r = s.activate("notacommand");
        assert!(matches!(r, CommandResult::Error(_)));
        assert_eq!(s.status.level, Level::Error);
    }

    #[test]
    fn activate_sets_the_tool_and_prompt() {
        let mut s = Session::new();
        s.activate("line").ok();
        assert_eq!(s.tool.id, ToolId::Line);
        assert_eq!(
            s.tool.prompt().as_deref(),
            Some("Specify next point or [Undo] (0 taken)")
        );
    }

    #[test]
    fn activate_requires_a_selection_for_modify_tools() {
        let mut s = session_with_circle();
        s.clear_selection();
        assert_eq!(s.activate("move"), CommandResult::Unavailable);
        s.select_all();
        assert!(s.activate("move").is_ok());
    }

    #[test]
    fn erase_works_without_a_selection() {
        let mut s = session_with_circle();
        // Erase picks by hover, so it must not demand a selection.
        assert!(s.activate("erase").is_ok());
    }

    #[test]
    fn clicking_draws_and_marks_dirty() {
        let mut s = Session::new();
        s.viewport.cam2d = Camera2D::new(Vec2::new(800.0, 600.0));
        s.activate("line").ok();
        s.dirty = false;
        s.click_world(Vec2::ZERO, false);
        assert!(s.dirty);
        s.click_world(Vec2::new(10.0, 0.0), false);
        assert_eq!(s.doc.entities.len(), 1);
        match s.doc.entities.iter().next().unwrap().entity {
            cad_doc::EntityKind::Line(l) => assert!((l.length() - 10.0).abs() < 1e-4),
            _ => panic!(),
        }
    }

    #[test]
    fn cursor_updates_hover_and_snaps() {
        let mut s = session_with_circle();
        s.viewport.cam2d = Camera2D::new(Vec2::new(800.0, 600.0));
        s.viewport.cam2d.scale = 4.0;
        // Park the cursor right on the circle edge.
        s.cursor_world = Vec2::new(10.0, 0.3);
        s.update_cursor();
        assert_eq!(s.tool.hovered, Some(EntityId(0)));
    }

    #[test]
    fn window_selection_finishes() {
        let mut s = session_with_circle();
        s.window_drag = Some((Vec2::new(-30.0, -30.0), Vec2::new(30.0, 30.0)));
        s.finish_window();
        assert_eq!(s.tool.selection.len(), 1);
        assert!(s.window_drag.is_none());
    }

    #[test]
    fn a_second_entity_is_counted() {
        let mut s = session_with_line_and_circle();
        assert_eq!(s.doc.entities.len(), 2);
        assert_eq!(s.entity_summary(), (2, 0));
    }

    #[test]
    fn entity_summary_splits_planar_and_solid() {
        let mut s = session_with_circle();
        let layer = s.doc.layers.default_layer();
        s.doc.add(
            cad_doc::Entity::solid(cad_doc::entity::Box3d::new(Vec3::ZERO, Vec3::splat(1.0)))
                .with_layer(layer),
        );
        assert_eq!(s.entity_summary(), (1, 1));
    }

    #[test]
    fn tick_fades_status_messages() {
        let mut s = Session::new();
        s.status = StatusMessage::info("temporary");
        s.tick(1.0);
        assert_eq!(s.status.level, Level::Info);
        assert!(!s.status.is_sticky());
        s.tick(10.0);
        assert!(s.status.text.contains("Ready"));
    }

    #[test]
    fn sticky_prompts_survive_ticks() {
        let mut s = Session::new();
        s.activate("circle").ok();
        s.status = StatusMessage::prompt("Specify centre point");
        s.tick(100.0);
        assert_eq!(s.status.text, "Specify centre point");
    }

    #[test]
    fn time_accumulates() {
        let mut s = Session::new();
        let t0 = s.time;
        s.tick(0.5);
        s.tick(0.5);
        assert!((s.time - t0 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn command_to_tool_mapping_covers_every_tool() {
        for name in [
            "line",
            "circle",
            "arc",
            "polyline",
            "rectangle",
            "move",
            "copy",
            "rotate",
            "scale",
            "erase",
            "trim",
            "extend",
            "mirror",
            "offset",
            "extrude",
            "zoomwindow",
        ] {
            assert!(tool_for_command(name).is_some(), "{name} unmapped");
        }
        assert!(tool_for_command("nope").is_none());
        // Aliases.
        assert_eq!(tool_for_command("L"), Some(ToolId::Line));
        assert_eq!(tool_for_command("REC"), Some(ToolId::Rectangle));
    }

    #[test]
    fn zoom_by_changes_scale() {
        let mut s = session_with_circle();
        s.viewport.canvas = Rect2::from_xywh(0.0, 0.0, 800.0, 600.0);
        let s0 = s.viewport.cam2d.scale;
        s.zoom_by(2.0);
        assert!(s.viewport.cam2d.scale > s0);
    }

    #[test]
    fn zoom_to_entities_frames_them() {
        let mut s = session_with_circle();
        s.viewport.cam2d = Camera2D::new(Vec2::new(800.0, 600.0));
        s.zoom_to_entities(&[EntityId(0)]);
        let v = s.viewport.cam2d.world_viewport();
        assert!(v.contains(Vec2::ZERO), "{v:?}");
    }
}
