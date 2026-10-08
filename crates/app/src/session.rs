//! The session: everything mutable about one open drawing.
//!
//! A `Session` owns the document, the undo history, the cameras, the active
//! tool and the selection. The window loop owns nothing but the window.

use crate::command::{CommandRegistry, CommandResult};
use crate::tools::{Tool, ToolId, ToolOutcome, pick_window};
use cad_core::{Camera2D, Camera3D, Rect2, Vec2, Vec3};
use cad_doc::{Document, EntityId, History};
use cad_gfx::renderer::RenderTarget;

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
    pub cam3d: Camera3D,
    pub mode: ViewMode,
    pub show_grid: bool,
    pub grid_spacing: f32,
    pub show_axes: bool,
    /// Rectangle occupied by the drawing area, excluding panels.
    pub canvas: Rect2,
    /// 3D view target, refreshed by [`Viewport::refresh_target`]. Rebuilding it
    /// every frame would allocate a `RenderTarget` per frame for no reason; the
    /// shaders only need it when a 3D entity is actually drawn.
    target: RenderTarget,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            cam2d: Camera2D::default(),
            cam3d: Camera3D::default(),
            mode: ViewMode::Model2d,
            show_grid: true,
            grid_spacing: 10.0,
            show_axes: true,
            canvas: Rect2::from_xywh(0.0, 0.0, 1280.0, 720.0),
            target: RenderTarget::default(),
        }
    }
}

impl Viewport {
    /// The cached 3D view/projection pair the renderer uploads as globals.
    pub fn target(&self) -> RenderTarget {
        self.target
    }

    /// Rebuild the cached target after the camera or canvas changed.
    pub fn refresh_target(&mut self) {
        let aspect = if self.canvas.height() > 0.0 {
            self.canvas.width() / self.canvas.height()
        } else {
            1.0
        };
        self.target = RenderTarget::from_camera(&self.cam3d, aspect);
    }

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
    /// Which theme the user picked, so the choice survives a `New`.
    pub dark: bool,
    /// Time accumulator, for animated UI.
    pub time: f32,
    /// The sketch constraints, in the order they were added.
    ///
    /// Stored on the session rather than the document because they are a
    /// working state, not drawing content: saving a file saves the geometry,
    /// not the relationships that produced it.
    pub constraints: cad_geom::constraint::Problem,
    /// The last commands run, oldest first, capped.
    ///
    /// This is the model's context. It is deliberately capped: a CAD session can
    /// run thousands of commands and a longer context buys nothing here, while
    /// making every prediction proportionally slower.
    pub command_log: Vec<String>,
    /// The command-line suggestion model, trained once on first use.
    predictor: Option<cad_ai::suggest::NextCommand>,
}

/// How many commands of context the model is given.
const COMMAND_LOG_CAP: usize = 8;

/// Which constraint the user asked for.
///
/// Separate from the solver's `Constraint` because the solver needs point
/// indices and the user has a selection: translating one to the other is the
/// session's job, and conflating them is how a UI concept leaks into the math.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    Coincident,
    Horizontal,
    Vertical,
    Distance,
    Fix,
}

impl ConstraintKind {
    pub fn command(self) -> &'static str {
        match self {
            ConstraintKind::Coincident => "coincident",
            ConstraintKind::Horizontal => "horizontal",
            ConstraintKind::Vertical => "vertical",
            ConstraintKind::Distance => "distance",
            ConstraintKind::Fix => "fix",
        }
    }

    pub fn from_command(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "coincident" | "coinc" => ConstraintKind::Coincident,
            "horizontal" | "horiz" => ConstraintKind::Horizontal,
            "vertical" | "vert" => ConstraintKind::Vertical,
            "distance" | "dist" => ConstraintKind::Distance,
            "fix" => ConstraintKind::Fix,
            _ => return None,
        })
    }
}

/// A residual formatted the way the drawing reads it: three decimals and the
/// drawing's own unit suffix, so "0.004mm" rather than "0.00428571".
fn format_residual(r: f32, units: &cad_doc::Units) -> String {
    format!("{:.3}{}", r, units.suffix())
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
            dark: true,
            time: 0.0,
            command_log: Vec::new(),
            predictor: None,
            constraints: cad_geom::constraint::Problem::new(),
        }
    }

    // ------------------------------------------------------- constraints

    // ----------------------------------------------------------- suggestions

    /// Run an already-parsed intent.
    ///
    /// Split from [`Session::run_intent`] so a caller that parsed the line itself
    /// -- as the unknown-command fallback does -- does not parse it twice.
    pub fn apply_intent(&mut self, intent: cad_ai::Intent) -> CommandResult {
        match &intent {
            cad_ai::Intent::Command { name, args } => {
                if name == "print" {
                    self.status = StatusMessage::info(args.clone());
                    return CommandResult::Ok;
                }
                let result = self.run_command_line_inner(name, args, &[]);
                if result.is_ok() {
                    self.log_command(&intent_command(&intent));
                }
                result
            }
            cad_ai::Intent::Geometry { kind, a, b } => {
                let cmd = kind.command();
                // Geometry needs the tool active before its points mean
                // anything, so a failure here must not leave the points behind.
                if !self.run_command_line_inner(cmd, "", &[]).is_ok() {
                    return CommandResult::Unavailable;
                }
                self.click_world(cad_core::Vec2::new(a[0], a[1]), false);
                self.click_world(cad_core::Vec2::new(b[0], b[1]), false);
                self.log_command(cmd);
                CommandResult::Ok
            }
            cad_ai::Intent::Unknown => CommandResult::Unavailable,
        }
    }

    /// Run a natural-language request.
    ///
    /// This is the path that makes the app usable without knowing its aliases:
    /// "draw a line from 0,0 to 10,10" runs the line tool, "circle at 5,5 radius
    /// 3" runs the circle tool, and "= 5 + 3" prints 8. Anything the parser does
    /// not recognise falls through to the command line, so a mistyped command is
    /// still a mistyped command and not a silent no-op.
    pub fn run_intent(&mut self, text: &str) -> CommandResult {
        self.apply_intent(cad_ai::parse(text))
    }

    /// Record a command so the model can use it as context.
    fn log_command(&mut self, name: &str) {
        // Log the *canonical* name, not what was typed: `L`, `line` and `LINE`
        // are the same context for the model.
        let canonical = match tool_for_command(name) {
            Some(t) => t.command().to_string(),
            None => name.to_ascii_lowercase(),
        };
        self.command_log.push(canonical);
        if self.command_log.len() > COMMAND_LOG_CAP {
            let excess = self.command_log.len() - COMMAND_LOG_CAP;
            self.command_log.drain(..excess);
        }
    }

    /// The model, trained on first use.
    ///
    /// Training is a few thousand full-batch steps over a few hundred rows. It
    /// happens once per process and costs milliseconds, which is cheaper than
    /// shipping a weight file and having to version it.
    pub fn predictor(&mut self) -> &cad_ai::suggest::NextCommand {
        self.predictor
            .get_or_insert_with(cad_ai::suggest::NextCommand::trained)
    }

    /// What the user most likely wants to type next.
    pub fn suggestion(&mut self) -> Option<&'static str> {
        // Leave it untrained until something has already asked for it: the
        // cold-start distribution is nearly uniform, so a session that never
        // types a command never pays for it.
        let net = self.predictor.as_ref()?;
        let has_selection = !self.tool.selection.is_empty();
        let tool_active = !self.tool.state.is_idle();
        let log: Vec<&str> = self.command_log.iter().map(|s| s.as_str()).collect();
        net.suggest(&log, has_selection, tool_active)
            .first()
            .map(|(n, _)| *n)
    }

    /// Completions for a partially typed prefix, best first.
    ///
    /// This is the path that always trains: the user typed something, so they
    /// are using the command line, so the model is worth having.
    pub fn completions(&mut self, prefix: &str) -> Vec<String> {
        let known: Vec<&str> = cad_ai::suggest::VOCAB.to_vec();
        let has_selection = !self.tool.selection.is_empty();
        let tool_active = !self.tool.state.is_idle();
        // Own the names rather than borrowing them from `self.command_log`: the
        // model has to be trained through `&mut self`, and the two borrows would
        // otherwise overlap.
        let log: Vec<&str> = self.command_log.iter().map(|s| s.as_str()).collect();
        let net = self
            .predictor
            .get_or_insert_with(cad_ai::suggest::NextCommand::trained);
        let prior = net.distribution(&log, has_selection, tool_active);
        cad_ai::suggest::rank(prefix, &known, &prior)
            .into_iter()
            .map(|(n, _)| n.to_string())
            .collect()
    }

    /// The last `n` commands, oldest first, for display.
    pub fn recent_commands(&self, n: usize) -> Vec<&str> {
        self.command_log
            .iter()
            .rev()
            .take(n)
            .map(|s| s.as_str())
            .collect()
    }

    // ------------------------------------------------------------ constraints

    /// Add a constraint between the selection's anchor points and solve.
    ///
    /// The two entities contribute their anchor points, the constraint is built
    /// between the named grips, and the whole sketch is solved as one undo step.
    /// A solve that does not converge still applies: a partially-solved sketch
    /// is closer than an unsolved one, and the status bar says how far off it
    /// is rather than pretending it worked.
    pub fn constrain(&mut self, kind: ConstraintKind, value: Option<f32>) -> CommandResult {
        // Fix pins whatever is selected, so it needs one object, not two. Every
        // other kind relates two objects to each other.
        if kind == ConstraintKind::Fix {
            if self.tool.selection.is_empty() {
                self.status = StatusMessage::error("Select objects to fix");
                return CommandResult::Unavailable;
            }
        } else if self.tool.selection.len() < 2 {
            self.status = StatusMessage::error("Select two objects to constrain");
            return CommandResult::Unavailable;
        }
        let ids: Vec<EntityId> = self.tool.selection.iter().take(2).cloned().collect();
        let anchors: Vec<Vec<cad_core::Vec2>> = ids
            .iter()
            .filter_map(|id| self.doc.entities.get(*id))
            .map(|e| e.anchor_points())
            .collect();
        if anchors.len() < 2 || anchors[0].is_empty() || anchors[1].is_empty() {
            self.status = StatusMessage::error("Those objects have no points to constrain");
            return CommandResult::Unavailable;
        }
        // The grips are the closest pair of anchor points: constraining "these
        // two objects" means the points the user can see touching, not an
        // arbitrary first vertex.
        let (mut ai, mut bi) = (0usize, 0usize);
        let mut best = f32::INFINITY;
        for (i, a) in anchors[0].iter().enumerate() {
            for (j, b) in anchors[1].iter().enumerate() {
                let d = a.distance(*b);
                if d < best {
                    best = d;
                    ai = i;
                    bi = j;
                }
            }
        }
        // World indices: entity 0 owns 0..n0, entity 1 owns n0...
        let n0 = anchors[0].len();
        let mut points = anchors[0].clone();
        points.extend_from_slice(&anchors[1]);
        let mut problem = self.constraints.clone();
        let constraint = match kind {
            ConstraintKind::Coincident => {
                cad_geom::constraint::Constraint::Coincident { a: ai, b: n0 + bi }
            }
            ConstraintKind::Horizontal => {
                cad_geom::constraint::Constraint::Horizontal { a: ai, b: n0 + bi }
            }
            ConstraintKind::Vertical => {
                cad_geom::constraint::Constraint::Vertical { a: ai, b: n0 + bi }
            }
            ConstraintKind::Distance => {
                let Some(d) = value.filter(|d| *d > 0.0) else {
                    self.status = StatusMessage::error("DISTANCE needs a positive length");
                    return CommandResult::Error("bad distance".into());
                };
                cad_geom::constraint::Constraint::Distance {
                    a: ai,
                    b: n0 + bi,
                    distance: d,
                }
            }
            ConstraintKind::Fix => {
                // Fix pins every anchor of the selection, not just two points.
                let mut problem = self.constraints.clone();
                for id in &self.tool.selection {
                    if let Some(e) = self.doc.entities.get(*id) {
                        let base = points.len();
                        let anchors = e.anchor_points();
                        points.extend_from_slice(&anchors);
                        for k in 0..anchors.len() {
                            problem.add(cad_geom::constraint::Constraint::Fix { point: base + k });
                        }
                    }
                }
                return self.apply_constraints(problem, points, vec![ids.clone()]);
            }
        };
        problem.add(constraint);
        self.apply_constraints(problem, points, vec![ids])
    }

    /// Solve `problem` over `points` and write the result back as one undo step.
    ///
    /// `groups` maps point ranges back to entities: each entry is the entity ids
    /// whose anchors occupy one contiguous run of `points`, in order.
    fn apply_constraints(
        &mut self,
        problem: cad_geom::constraint::Problem,
        mut points: Vec<cad_core::Vec2>,
        groups: Vec<Vec<EntityId>>,
    ) -> CommandResult {
        let fixed = vec![false; points.len()];
        let report = cad_geom::constraint::solve(&problem, &mut points, &fixed, 1e-3, 200);
        // Split the solved points back across the entities they came from.
        let mut offset = 0usize;
        let mut rebuilt: Vec<(EntityId, cad_doc::Entity)> = Vec::new();
        for ids in &groups {
            for id in ids {
                let Some(e) = self.doc.entities.get(*id) else {
                    continue;
                };
                let n = e.anchor_points().len();
                if points.len() < offset + n {
                    continue;
                }
                if let Some(next) = e.with_anchor_points(&points[offset..offset + n]) {
                    rebuilt.push((*id, next));
                }
                offset += n;
            }
        }
        if rebuilt.is_empty() {
            self.status = StatusMessage::error("Nothing to solve");
            return CommandResult::Unavailable;
        }
        let label = format!(
            "Constraint ({} left)",
            format_residual(report.residual, &self.doc.units)
        );
        let mut history = std::mem::take(&mut self.history);
        {
            let mut tx = history.begin(&mut self.doc.entities, &label);
            for (id, e) in &rebuilt {
                tx.replace(*id, e.clone());
            }
            tx.commit();
        }
        self.history = history;
        self.constraints = problem;
        self.doc.invalidate_extents();
        self.dirty = true;
        self.status = if report.converged {
            StatusMessage::success(format!("Constrained ({})", rebuilt.len()))
        } else {
            StatusMessage::info(format!(
                "Constrained, off by {}",
                format_residual(report.residual, &self.doc.units)
            ))
        };
        CommandResult::Ok
    }

    /// Drop every constraint. The geometry stays where the solver left it;
    /// constraints are relationships, not history.
    pub fn clear_constraints(&mut self) -> CommandResult {
        let n = self.constraints.len();
        self.constraints = cad_geom::constraint::Problem::new();
        self.status = StatusMessage::info(format!("{n} constraints cleared"));
        CommandResult::Ok
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

    /// Multiply the zoom by `factor`, keeping `focus` (screen pixels) fixed.
    pub fn zoom_by_at(&mut self, factor: f32, focus: Vec2) {
        self.viewport.cam2d.zoom_at(factor, focus);
        self.dirty = true;
    }

    /// Multiply the zoom by `factor` about the canvas centre.
    pub fn zoom_by(&mut self, factor: f32) {
        let focus = self.viewport.canvas.center();
        self.viewport.cam2d.zoom_at(factor, focus);
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
            // No outcome still means the tool moved: the first click of a line
            // starts a rubber band and needs a redraw even though nothing in the
            // document changed.
            ToolOutcome::None => self.dirty = true,
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
            // Direction decides the rule, as in every CAD app: left-to-right is a
            // window selection (fully enclosed), right-to-left is a crossing
            // selection (anything the rubber band touches).
            let crossing = b.x < a.x;
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

    // ----------------------------------------------------------------- files

    /// Load `path` into this session, replacing the drawing.
    ///
    /// The format is chosen by extension: `.dxf` goes through the DXF reader,
    /// everything else through the native container. Doing it by extension
    /// rather than by sniffing means a `.dxf` that is actually a project file
    /// fails loudly instead of importing as an empty drawing.
    pub fn open(&mut self, path: &std::path::Path) -> CommandResult {
        let is_dxf = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("dxf"));
        let doc = if is_dxf {
            let mut doc = Document::new();
            match cad_io::import(&mut doc, path) {
                Ok(report) => {
                    self.status = StatusMessage::success(format!(
                        "Opened {} ({} entities, {} layers)",
                        display_name(path),
                        report.entities,
                        report.layers
                    ));
                }
                Err(e) => {
                    self.status =
                        StatusMessage::error(format!("Cannot open {}: {e}", path.display()));
                    return CommandResult::Error(e.to_string());
                }
            }
            doc
        } else {
            match cad_io::load(path) {
                Ok(doc) => {
                    self.status = StatusMessage::success(format!("Opened {}", display_name(path)));
                    doc
                }
                Err(e) => {
                    self.status =
                        StatusMessage::error(format!("Cannot open {}: {e}", path.display()));
                    return CommandResult::Error(e.to_string());
                }
            }
        };

        // Swap the drawing in. Layer and block tables come from the file, so they
        // replace the session's rather than merging with it.
        self.doc = doc;
        self.doc.entities.rebuild_index(index_bounds(&self.doc));
        self.history = History::new();
        self.tool.selection.clear();
        self.tool.hovered = None;
        self.tool.finish();
        self.path = Some(path.to_path_buf());
        self.dirty = false;
        self.viewport.cam3d = cam3d_framing(&self.doc);
        self.viewport.refresh_target();
        self.zoom_extents();
        CommandResult::Ok
    }

    /// Save to `path`, or to the current path if none is given.
    pub fn save(&mut self, path: Option<&std::path::Path>) -> CommandResult {
        let target = match path.or(self.path.as_deref()) {
            Some(p) => p.to_path_buf(),
            None => {
                self.status = StatusMessage::error("No file name - use SAVEAS <path>");
                return CommandResult::Unavailable;
            }
        };
        let result = if target
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("dxf"))
        {
            cad_io::dxf::save(&self.doc, &target).map_err(|e| e.to_string())
        } else {
            cad_io::save(&self.doc, &target).map_err(|e| e.to_string())
        };
        match result {
            Ok(()) => {
                self.path = Some(target.clone());
                self.dirty = false;
                self.status = StatusMessage::success(format!("Saved {}", display_name(&target)));
                CommandResult::Ok
            }
            Err(e) => {
                self.status =
                    StatusMessage::error(format!("Cannot save {}: {e}", target.display()));
                CommandResult::Error(e)
            }
        }
    }

    /// Write a DXF of the current drawing without changing the session path.
    pub fn export_dxf(&mut self, path: &std::path::Path) -> CommandResult {
        match cad_io::dxf::save(&self.doc, path) {
            Ok(()) => {
                self.status = StatusMessage::success(format!("Exported {}", display_name(path)));
                CommandResult::Ok
            }
            Err(e) => {
                self.status =
                    StatusMessage::error(format!("Cannot export {}: {e}", path.display()));
                CommandResult::Error(e.to_string())
            }
        }
    }

    /// Parse and run one command line, with arguments.
    ///
    /// `line` is a command name followed by optional arguments, e.g.
    /// `open plan.dxf` or `undo 3`. File commands take a path argument because
    /// there is no native file dialog: the user types it, and the error from a
    /// bad path comes back through the status bar rather than a modal.
    ///
    /// A line that is not a command is tried as a sentence: "zoom all" and
    /// "draw a line from 0,0 to 10,10" both run, because the intent parser sees
    /// the whole line. Known commands are still tried first, so a real command
    /// always wins over an interpretation.
    pub fn run_command_line(&mut self, line: &str) -> CommandResult {
        let line = line.trim();
        if line.is_empty() {
            return CommandResult::Ok;
        }
        let mut parts = line.split_whitespace();
        let head = parts.next().unwrap_or("").to_string();
        let name = head.to_ascii_lowercase();
        let arg = parts.next().unwrap_or("");
        let extra: Vec<&str> = parts.collect();

        // Only structured commands get the argument-count check. An unknown
        // first word means the line might be a sentence, and a sentence with
        // more than two words is the normal case, not a typo.
        let structured = self.commands.get(&name).is_some()
            || tool_for_command(&name).is_some()
            || &name == "open"
            || &name == "saveas"
            || &name == "export"
            || &name == "array"
            || &name == "block"
            || &name == "insert"
            || &name == "explode"
            || &name == "layer"
            || &name == "help"
            || &name == "about"
            || &name == "selectall"
            || &name == "all"
            || &name == "new"
            || &name == "undo"
            || &name == "redo";
        if structured
            && !extra.is_empty()
            && !matches!(
                name.as_str(),
                "undo" | "redo" | "array" | "insert" | "block" | "layer"
            )
        {
            // `open a.dxf b.dxf` is a typo, not a request to ignore an argument.
            self.status =
                StatusMessage::error(format!("{name} takes at most one argument: {line}"));
            return CommandResult::Error(format!("unexpected argument in `{line}`"));
        }
        if !structured {
            // Not a command at all: let the intent parser see the whole line.
            // `apply_intent` logs the canonical command when it runs.
            return match self.apply_intent(cad_ai::parse(line)) {
                CommandResult::Unavailable => {
                    // Nothing understood it either. A mistyped command is an
                    // error, not a quiet no-op: the user has to know it failed.
                    let msg = format!("Unknown command: {line}");
                    self.status = StatusMessage::error(msg.clone());
                    CommandResult::Error(msg)
                }
                other => other,
            };
        }

        // Only a command that actually ran becomes context. Logging a typo would
        // teach the model that the user runs commands that do not exist.
        let result = self.run_command_line_inner(&name, arg, &extra);
        if result.is_ok() {
            self.log_command(&name);
        }
        result
    }

    fn run_command_line_inner(&mut self, name: &str, arg: &str, extra: &[&str]) -> CommandResult {
        match name {
            "open" | "o" => {
                if arg.is_empty() {
                    self.status = StatusMessage::prompt("Specify a file to open");
                    return CommandResult::Unavailable;
                }
                self.open(std::path::Path::new(arg))
            }
            "save" => self.save(None),
            "saveas" => {
                if arg.is_empty() {
                    self.status = StatusMessage::prompt("Specify a file name");
                    return CommandResult::Unavailable;
                }
                self.save(Some(std::path::Path::new(arg)))
            }
            "export" => {
                if arg.is_empty() {
                    self.status = StatusMessage::prompt("Specify a DXF to write");
                    return CommandResult::Unavailable;
                }
                self.export_dxf(std::path::Path::new(arg))
            }
            "new" | "_new" => {
                self.new_document();
                CommandResult::Ok
            }
            "undo" | "u" => {
                let n: usize = arg.parse().unwrap_or(1);
                for _ in 0..n.max(1) {
                    if !self.undo().is_ok() {
                        break;
                    }
                }
                CommandResult::Ok
            }
            "redo" => {
                let n: usize = arg.parse().unwrap_or(1);
                for _ in 0..n.max(1) {
                    if !self.redo().is_ok() {
                        break;
                    }
                }
                CommandResult::Ok
            }
            "zoomall" | "z" => {
                self.zoom_extents();
                CommandResult::Ok
            }
            "zoomin" => {
                self.zoom_by(1.25);
                CommandResult::Ok
            }
            "zoomout" => {
                self.zoom_by(1.0 / 1.25);
                CommandResult::Ok
            }
            "view3d" => self.toggle_3d(),
            "grid" => self.toggle_grid(),
            "snap" => self.toggle_snap(),
            "ortho" => self.toggle_ortho(),
            "polar" => self.toggle_polar(),
            "layer" | "la" => {
                if arg.is_empty() {
                    let c = self.doc.current_layer();
                    self.status =
                        StatusMessage::info(format!("Current layer: {}", self.doc.layers.name(c)));
                    return CommandResult::Ok;
                }
                match self.doc.layers.by_name(arg) {
                    Some(id) => {
                        self.set_current_layer(id);
                        CommandResult::Ok
                    }
                    None => {
                        self.add_layer(arg);
                        CommandResult::Ok
                    }
                }
            }
            "selectall" | "all" => {
                self.select_all();
                CommandResult::Ok
            }
            "array" => {
                // `array rows cols [dx dy] [angle]`. The counts default to 2x2 and
                // the spacing to the selection's own size, so a bare `array`
                // does something visible instead of complaining.
                let (rows, cols) = match (arg, extra.first()) {
                    ("", _) => (2, 2),
                    (a, None) => match a.parse::<u32>() {
                        Ok(n) => (n, n),
                        Err(_) => {
                            self.status = StatusMessage::error("ARRAY needs row and column counts");
                            return CommandResult::Error("bad array".into());
                        }
                    },
                    (a, Some(b)) => match (a.parse::<u32>(), b.parse::<u32>()) {
                        (Ok(r), Ok(c)) => (r, c),
                        _ => {
                            self.status =
                                StatusMessage::error("ARRAY counts must be whole numbers");
                            return CommandResult::Error("bad array".into());
                        }
                    },
                };
                let span = self.selection_span();
                let polar = extra
                    .get(1)
                    .and_then(|v| v.parse::<f32>().ok())
                    .map(|d| d.to_radians());
                let spacing =
                    Vec2::new(span.x * (cols.max(2) as f32), span.y * (rows.max(2) as f32));
                self.array_selection(rows, cols, spacing, polar)
            }
            "block" | "b" => {
                if arg.is_empty() {
                    self.status = StatusMessage::prompt("Type BLOCK <name>");
                    return CommandResult::Unavailable;
                }
                self.make_block(arg)
            }
            "insert" | "i" => {
                if arg.is_empty() {
                    self.status = StatusMessage::prompt("Type INSERT <block>");
                    return CommandResult::Unavailable;
                }
                let Some(id) = self.doc.blocks.by_name(&arg.to_ascii_uppercase()) else {
                    self.status = StatusMessage::error(format!("No block named {arg}"));
                    return CommandResult::Error(format!("no block {arg}"));
                };
                // Without a point on the command line, insert at the selection's
                // centre if there is one and at the origin otherwise.
                let at = self.selection_span_centre();
                self.insert_block(id, at, 0.0, 1.0)
            }
            "explode" | "x" => self.explode_selection(),
            "coincident" | "coinc" => self.constrain(ConstraintKind::Coincident, None),
            "horizontal" | "horiz" => self.constrain(ConstraintKind::Horizontal, None),
            "vertical" | "vert" => self.constrain(ConstraintKind::Vertical, None),
            "distance" | "dist" => match arg.parse::<f32>() {
                Ok(d) if d > 0.0 => self.constrain(ConstraintKind::Distance, Some(d)),
                _ => {
                    self.status = StatusMessage::prompt("Type DISTANCE <length>");
                    CommandResult::Unavailable
                }
            },
            "fix" => self.constrain(ConstraintKind::Fix, None),
            "unconstrain" | "clearconstraints" => self.clear_constraints(),
            "dimlinear" | "dimlin" | "dimaligned" | "dimali" | "dimradius" | "dimrad"
            | "dimdiameter" | "dimdia" | "dimangular" | "dimang" => {
                // Every dimension kind shares the same two-point sequence; only
                // what the number means differs.
                let Some(kind) = cad_doc::DimensionKind::from_command(name) else {
                    return CommandResult::Error(format!("unknown dimension {name}"));
                };
                if self.tool.selection.len() < 2 {
                    self.status = StatusMessage::error(format!(
                        "{} needs two points",
                        kind.command().to_ascii_uppercase()
                    ));
                    return CommandResult::Unavailable;
                }
                // Two entities, each contributing one measurement point.
                let pts: Vec<cad_core::Vec2> = self
                    .tool
                    .selection
                    .iter()
                    .take(2)
                    .filter_map(|id| self.doc.entities.get(*id))
                    .map(|e| e.bounds_2d().center())
                    .collect();
                if pts.len() < 2 {
                    self.status = StatusMessage::error("Select two objects");
                    return CommandResult::Unavailable;
                }
                self.add_dimension(kind, pts[0], pts[1], self.selection_span_centre())
            }
            "erase" | "e" | "del" | "delete" => self.delete_selection(),
            "help" | "?" | "??" => {
                self.status = StatusMessage::info(format!(
                    "{} commands. Try: line, circle, arc, polyline, rectangle, move, copy, \
                     rotate, mirror, trim, offset, extrude, zoomall, view3d, layer <name>, \
                     open <file>, save [file], export <file.dxf>, undo [n]",
                    self.commands.len()
                ));
                CommandResult::Ok
            }
            "about" => {
                self.status = StatusMessage::info(concat!(
                    "CADKit - native 2D/3D CAD on wgpu. ",
                    "Analytic geometry, transactional undo, DXF interchange."
                ));
                CommandResult::Ok
            }
            other => {
                if self.commands.get(other).is_some() || tool_for_command(other).is_some() {
                    self.activate(other)
                } else {
                    // Unreachable in practice: `run_command_line` only calls this
                    // for structured commands, and anything else goes straight to
                    // the intent parser. Kept as a guard so an inner caller that
                    // passes a bare word still gets a real error.
                    let msg = format!("Unknown command: {other}");
                    self.status = StatusMessage::error(msg.clone());
                    CommandResult::Error(msg)
                }
            }
        }
    }

    /// Start an empty drawing.
    pub fn new_document(&mut self) {
        let theme = self.theme;
        let dark = self.dark;
        let cam2d = self.viewport.cam2d;
        let canvas = self.viewport.canvas;
        // The trained model survives: it cost milliseconds to build and it knows
        // nothing about this drawing. The command log does not: it is this
        // drawing's history, not the next one's.
        let predictor = self.predictor.take();
        *self = Session::new();
        self.theme = theme;
        self.dark = dark;
        self.predictor = predictor;
        self.viewport.cam2d = cam2d;
        self.viewport.canvas = canvas;
        self.viewport.refresh_target();
        self.path = None;
        self.status = StatusMessage::success("New drawing");
    }

    // ------------------------------------------------------- array and blocks

    /// Array the selection: `rows` x `cols` copies, or `count` copies on a
    /// circle when `polar` is set.
    ///
    /// The original is replaced by the whole array, which is AutoCAD's behaviour
    /// and the reason an array is a single undo step rather than `n-1` of them.
    pub fn array_selection(
        &mut self,
        rows: u32,
        cols: u32,
        spacing: Vec2,
        polar: Option<f32>,
    ) -> CommandResult {
        if self.tool.selection.is_empty() {
            self.status = StatusMessage::error("Select objects, then Array");
            return CommandResult::Unavailable;
        }
        let (rows, cols) = (rows.max(1), cols.max(1));
        let total = if let Some(_angle) = polar {
            rows.max(cols)
        } else {
            rows.saturating_mul(cols)
        };
        if total <= 1 {
            self.status = StatusMessage::error("An array needs more than one copy");
            return CommandResult::Unavailable;
        }
        // Refuse a grid that would run off the end of a float. A user typing
        // `array 1000 1000` should get a message, not a hang or a NaN.
        if (rows as u64) * (cols as u64) > 100_000 {
            self.status =
                StatusMessage::error(format!("That would create {total} copies; pick fewer"));
            return CommandResult::Error("array too large".into());
        }

        let originals: Vec<(cad_doc::EntityId, cad_doc::Entity)> = self
            .tool
            .selection
            .iter()
            .filter_map(|id| self.doc.entities.get(*id).map(|e| (*id, e.clone())))
            .collect();
        if originals.is_empty() {
            return CommandResult::Unavailable;
        }
        let total = total as usize;
        let mut made = Vec::with_capacity(originals.len() * total);
        let mut history = std::mem::take(&mut self.history);
        {
            let mut tx = history.begin(&mut self.doc.entities, &format!("Array ({total} copies)"));
            for id in originals.iter().map(|(id, _)| *id) {
                tx.remove(id);
            }
            for (_, src) in originals.iter() {
                for copy in 0..total {
                    let placed = match polar {
                        Some(total_angle) => {
                            let step = total_angle / total as f32;
                            let about = array_centre(&originals);
                            src.rotated(about, step * copy as f32)
                        }
                        None => {
                            let (dx, dy) = array_offset(copy as u32, rows, cols, spacing);
                            src.translated(Vec3::new(dx, dy, 0.0))
                        }
                    };
                    // Copy 0 lands on the original's own slot, so the array
                    // starts where the user drew it.
                    made.push(tx.insert(placed));
                }
            }
            tx.commit();
        }
        self.history = history;
        self.doc.invalidate_extents();
        self.tool.selection = made;
        self.dirty = true;
        self.status = StatusMessage::success(format!("Arrayed {} copies", total));
        CommandResult::Ok
    }

    /// The selection's own size, for array spacing defaults.
    pub fn selection_span(&self) -> Vec2 {
        let mut r = Rect2::EMPTY;
        let mut any = false;
        for id in &self.tool.selection {
            if let Some(e) = self.doc.entities.get(*id) {
                let b = e.bounds_2d();
                r = if any { r.union(b) } else { b };
                any = true;
            }
        }
        if any { r.size() } else { Vec2::splat(10.0) }
    }

    /// Where an insert lands with no point on the command line.
    pub fn selection_span_centre(&self) -> Vec2 {
        let mut r = Rect2::EMPTY;
        let mut any = false;
        for id in &self.tool.selection {
            if let Some(e) = self.doc.entities.get(*id) {
                let b = e.bounds_2d();
                r = if any { r.union(b) } else { b };
                any = true;
            }
        }
        if any { r.center() } else { Vec2::ZERO }
    }

    /// Add an associative dimension between two points, with `text` placed at
    /// `at`.
    pub fn add_dimension(
        &mut self,
        kind: cad_doc::DimensionKind,
        p1: Vec2,
        p2: Vec2,
        at: Vec2,
    ) -> CommandResult {
        let suffix = self.doc.units.suffix().to_string();
        let layer = self.doc.current_layer();
        let d = cad_doc::Dimension::new(kind, p1, p2, at);
        let text = d.text(&suffix);
        // A dimension whose measured value rounds to zero is a mistake, not a
        // measurement, and drawing it hides the mistake.
        if kind != cad_doc::DimensionKind::Angular && d.measurement() <= 1e-6 {
            self.status = StatusMessage::error("Those points coincide");
            return CommandResult::Error("zero-length dimension".into());
        }
        let id = self
            .doc
            .add(cad_doc::Entity::new(cad_doc::EntityKind::Dimension(d)).with_layer(layer));
        self.tool.selection = vec![id];
        self.doc.invalidate_extents();
        self.dirty = true;
        self.status = StatusMessage::success(format!("Dimension {text}"));
        CommandResult::Ok
    }

    /// Turn the selection into a block definition named `name`.
    pub fn make_block(&mut self, name: &str) -> CommandResult {
        if self.tool.selection.is_empty() {
            self.status = StatusMessage::error("Select objects, then Block");
            return CommandResult::Unavailable;
        }
        let mut name = name.trim().to_ascii_uppercase();
        if name.is_empty() {
            name = "BLOCK1".to_string();
        }
        // The base point is the selection's own lower-left corner, which is what
        // makes the insert land predictably without asking for a point first.
        let mut base = Rect2::ZERO;
        let mut first = true;
        for id in &self.tool.selection {
            if let Some(e) = self.doc.entities.get(*id) {
                let b = e.bounds_2d();
                base = if first { b } else { base.union(b) };
                first = false;
            }
        }
        if first {
            self.status = StatusMessage::error("Select objects, then Block");
            return CommandResult::Unavailable;
        }
        let origin = base.min;
        let entities: Vec<cad_doc::Entity> = self
            .tool
            .selection
            .iter()
            .filter_map(|id| self.doc.entities.get(*id).cloned())
            .map(|e| e.translated(Vec3::new(-origin.x, -origin.y, 0.0)))
            .collect();
        let n = entities.len();
        let mut block = cad_doc::Block::new(&name);
        block.base_point = Vec3::new(origin.x, origin.y, 0.0);
        block.entities = entities;
        let id = self.doc.blocks.insert(block);
        // Keep the original on screen: AutoCAD leaves it and reports the block.
        self.status = StatusMessage::success(format!(
            "Block {} created from {n} entities",
            self.doc.blocks.name(id)
        ));
        self.dirty = true;
        CommandResult::Ok
    }

    /// Place `block` at `pos`, `rotation` radians, scaled by `factor`.
    pub fn insert_block(
        &mut self,
        block: cad_doc::BlockId,
        pos: Vec2,
        rotation: f32,
        factor: f32,
    ) -> CommandResult {
        let Some(b) = self.doc.blocks.by_id(block) else {
            self.status = StatusMessage::error("No such block");
            return CommandResult::Unavailable;
        };
        if b.is_empty() {
            self.status =
                StatusMessage::error(format!("Block {} is empty", self.doc.blocks.name(block)));
            return CommandResult::Unavailable;
        }
        let name = self.doc.blocks.name(block).to_string();
        let ins = cad_doc::entity::InsertRef {
            block,
            position: Vec3::new(pos.x, pos.y, 0.0),
            scale: Vec3::splat(factor.abs().max(1e-6)),
            rotation,
            rows: 1,
            columns: 1,
            row_spacing: 0.0,
            col_spacing: 0.0,
        };
        let layer = self.doc.current_layer();
        let id = self.doc.add(cad_doc::Entity::insert(ins).with_layer(layer));
        self.tool.selection = vec![id];
        self.dirty = true;
        self.status = StatusMessage::success(format!("Inserted {name}"));
        CommandResult::Ok
    }

    /// Replace the selection with real geometry from its blocks.
    ///
    /// Explode is the escape hatch from a block that cannot be edited, so it has
    /// to be lossless in what it keeps: every entity becomes real geometry and
    /// nothing is left behind.
    pub fn explode_selection(&mut self) -> CommandResult {
        let inserts: Vec<(cad_doc::EntityId, cad_doc::Entity)> = self
            .tool
            .selection
            .iter()
            .filter_map(|id| self.doc.entities.get(*id).map(|e| (*id, e.clone())))
            .filter(|(_, e)| matches!(e.entity, cad_doc::EntityKind::Insert(_)))
            .collect();
        if inserts.is_empty() {
            self.status = StatusMessage::error("Select a block reference to explode");
            return CommandResult::Unavailable;
        }
        let mut made = Vec::new();
        let mut history = std::mem::take(&mut self.history);
        {
            let mut tx = history.begin(&mut self.doc.entities, "Explode");
            for (id, e) in &inserts {
                tx.remove(*id);
                for child in explode_entity(&self.doc.blocks, e, 0) {
                    made.push(tx.insert(child.with_layer(e.layer())));
                }
            }
            tx.commit();
        }
        self.history = history;
        self.doc.invalidate_extents();
        self.tool.selection = made.clone();
        self.dirty = true;
        self.status = StatusMessage::success(format!("Exploded into {} entities", made.len()));
        CommandResult::Ok
    }

    // ---------------------------------------------------------------- layers

    /// Flip a layer's visibility and report it.
    pub fn toggle_layer_visibility(&mut self, id: cad_doc::LayerId) {
        let now = self.doc.layers.by_id(id).is_some_and(|l| l.visible);
        self.doc.layers.set_visible(id, !now);
        self.doc.invalidate_extents();
        self.dirty = true;
        self.status = StatusMessage::info(format!(
            "Layer {} {}",
            self.doc.layers.name(id),
            if now { "hidden" } else { "shown" }
        ));
    }

    pub fn toggle_layer_lock(&mut self, id: cad_doc::LayerId) {
        let now = self.doc.layers.by_id(id).is_some_and(|l| l.locked);
        self.doc.layers.set_locked(id, !now);
        self.dirty = true;
        self.status = StatusMessage::info(format!(
            "Layer {} {}",
            self.doc.layers.name(id),
            if now { "unlocked" } else { "locked" }
        ));
    }

    /// Add a layer with a fresh name and make it current.
    pub fn add_layer(&mut self, name: &str) {
        let mut n = name.trim().to_string();
        if n.is_empty() {
            n = format!("Layer{}", self.doc.layers.len() + 1);
        }
        // `insert` de-duplicates by name, so a collision silently reuses the
        // existing layer. Append a suffix until it is genuinely new.
        let mut candidate = n.clone();
        let mut n_suffix = 1;
        while self.doc.layers.by_name(&candidate).is_some() {
            candidate = format!("{n}{n_suffix}");
            n_suffix += 1;
        }
        let id = self.doc.layers.insert(cad_doc::Layer::new(&candidate));
        self.doc.set_current_layer(id);
        self.dirty = true;
        self.status = StatusMessage::success(format!("Layer {candidate} created"));
    }

    pub fn set_current_layer(&mut self, id: cad_doc::LayerId) {
        self.doc.set_current_layer(id);
        self.dirty = true;
        self.status = StatusMessage::info(format!("Current layer: {}", self.doc.layers.name(id)));
    }
}

/// Where the `copy`-th member of a rectangular array goes.
fn array_offset(copy: u32, _rows: u32, cols: u32, spacing: Vec2) -> (f32, f32) {
    let r = copy / cols.max(1);
    let c = copy % cols.max(1);
    (c as f32 * spacing.x, r as f32 * spacing.y)
}

/// Extents of a set of captured entities, for reporting an array's span.
fn originals_bounds(originals: &[(cad_doc::EntityId, cad_doc::Entity)]) -> Rect2 {
    let mut r = Rect2::EMPTY;
    for (_, e) in originals {
        r = r.expand_point(e.bounds_2d().min);
        r = r.expand_point(e.bounds_2d().max);
    }
    r
}

/// The centre a polar array rotates about: the selection's own centre, which
/// is what makes a polar array read as a rotation rather than a scatter.
fn array_centre(originals: &[(cad_doc::EntityId, cad_doc::Entity)]) -> Vec3 {
    let b = originals_bounds(originals);
    if b.is_empty() {
        Vec3::ZERO
    } else {
        let c = b.center();
        Vec3::new(c.x, c.y, 0.0)
    }
}

/// Expand one INSERT into real geometry, bounded by `depth`.
///
/// A block that contains itself would otherwise expand forever, so the depth
/// limit *is* the cycle guard: at the limit the reference is kept rather than
/// resolved, which loses the nesting but terminates.
fn explode_entity(
    blocks: &cad_doc::BlockTable,
    e: &cad_doc::Entity,
    depth: usize,
) -> Vec<cad_doc::Entity> {
    /// How many levels of nesting an explode will follow.
    const MAX_DEPTH: usize = 8;
    let cad_doc::EntityKind::Insert(ins) = &e.entity else {
        return vec![e.clone()];
    };
    if depth >= MAX_DEPTH {
        return vec![e.clone()];
    }
    let Some(b) = blocks.by_id(ins.block) else {
        // A dangling reference has no geometry to become. Dropping it is the
        // only honest option, and the caller reports the count.
        return Vec::new();
    };
    // Block definition space -> world: undo the base point, scale, rotate, place.
    let xf = cad_geom::xform::Affine2::translation(-b.base_point.xy())
        .then(cad_geom::xform::Affine2::scaling(cad_core::Vec2::new(
            ins.scale.x,
            ins.scale.y,
        )))
        .then(cad_geom::xform::Affine2::rotation(ins.rotation))
        .then(cad_geom::xform::Affine2::translation(ins.position.xy()));
    let mut out = Vec::with_capacity(b.entities.len());
    for child in &b.entities {
        let placed = child.transformed(&xf);
        if placed.is_3d() {
            out.push(placed);
        } else {
            out.extend(explode_entity(blocks, &placed, depth + 1));
        }
    }
    out
}

/// The canonical command name an intent runs, for the command log.
///
/// `apply_intent` logs what actually ran rather than what was typed, so the
/// model's context is "line" whether the user typed `L`, `line` or
/// "draw a line".
fn intent_command(intent: &cad_ai::Intent) -> String {
    match intent {
        cad_ai::Intent::Command { name, .. } => match tool_for_command(name) {
            Some(t) => t.command().to_string(),
            None => name.clone(),
        },
        cad_ai::Intent::Geometry { kind, .. } => kind.command().to_string(),
        cad_ai::Intent::Unknown => String::new(),
    }
}

/// File name for a status message: the last path component, so the message stays
/// readable in a narrow status bar.
fn display_name(p: &std::path::Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}

/// A world rectangle big enough to hold every entity, for the spatial index.
///
/// An empty document has no extents, so this falls back to a fixed window around
/// the origin rather than an empty rect, which would make every pick miss.
fn index_bounds(doc: &Document) -> Rect2 {
    match doc.compute_extents() {
        Some((bb, _)) => {
            let r = Rect2::new(bb.min.xy(), bb.max.xy());
            r.expand(Vec2::splat(1.0).max(r.size() * 0.05))
        }
        None => Rect2::from_xywh(-1000.0, -1000.0, 2000.0, 2000.0),
    }
}

/// A 3D camera framing the whole drawing, used when entering the model view.
fn cam3d_framing(doc: &Document) -> Camera3D {
    let mut cam = Camera3D::default();
    match doc.compute_extents() {
        Some((bb, c)) => {
            let r = bb.size().length().max(1.0);
            cam.target = c;
            cam.eye = c + Vec3::new(r * 0.5, -r * 0.5, r * 0.4);
            cam.near = (r * 0.01).max(0.01);
            cam.far = r * 20.0;
        }
        None => {
            cam.eye = Vec3::new(100.0, -100.0, 80.0);
            cam.far = 5000.0;
        }
    }
    cam
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

    /// A session holding two entities: a line and a circle, as the name says.
    fn session_with_line_and_circle() -> Session {
        let mut s = Session::new();
        let layer = s.doc.layers.ensure_default();
        s.doc.add(
            cad_doc::Entity::line(Line::new(Vec2::new(-5.0, 20.0), Vec2::new(5.0, 20.0)))
                .with_layer(layer),
        );
        s.doc.add(
            cad_doc::Entity::circle(Circle::new(Vec2::new(30.0, 30.0), 12.0)).with_layer(layer),
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
        assert_eq!(s.doc.entities.len(), n - 1, "delete removed one");
        // Undo first: a fresh delete leaves the redo stack empty, so calling
        // redo() here would correctly report "Nothing to redo".
        assert!(s.undo().is_ok());
        assert_eq!(s.doc.entities.len(), n, "undo restored it");
        assert!(s.redo().is_ok());
        assert_eq!(s.doc.entities.len(), n - 1, "redo removed it again");
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
        assert!(s.activate("line").is_ok());
        assert_eq!(s.tool.id, ToolId::Line);
        // A fresh tool has collected no points, so `prompt()` is None and the
        // status bar announces the command instead. The point prompt appears
        // after the first click.
        assert_eq!(s.tool.prompt(), None);
        assert!(
            s.status.text.contains("Line"),
            "status was {:?}",
            s.status.text
        );
        s.click_world(Vec2::ZERO, false);
        assert_eq!(
            s.tool.prompt().as_deref(),
            Some("Specify next point or [Undo] (1 taken)")
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
        assert!(s.activate("line").is_ok());
        s.dirty = false;
        s.click_world(Vec2::ZERO, false);
        assert!(s.dirty);
        s.click_world(Vec2::new(10.0, 0.0), false);
        assert_eq!(s.doc.entities.len(), 1);
        match &s.doc.entities.iter().next().unwrap().entity {
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
        let s = session_with_line_and_circle();
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
        assert!(s.activate("circle").is_ok());
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

    // -------------------------------------------------------------- layers

    #[test]
    fn a_new_layer_becomes_current() {
        let mut s = Session::new();
        let before = s.doc.current_layer();
        s.add_layer("walls");
        let after = s.doc.current_layer();
        assert_ne!(before, after);
        assert_eq!(s.doc.layers.name(after), "walls");
        assert_eq!(s.doc.layers.len(), 2);
    }

    #[test]
    fn adding_a_duplicate_layer_name_gets_a_suffix() {
        let mut s = Session::new();
        s.add_layer("walls");
        let first = s.doc.current_layer();
        s.add_layer("walls");
        let second = s.doc.current_layer();
        assert_ne!(
            first, second,
            "insert() de-duplicates, so the second must not alias"
        );
        assert_eq!(s.doc.layers.name(second), "walls1");
    }

    #[test]
    fn an_empty_layer_name_gets_a_default() {
        let mut s = Session::new();
        s.add_layer("");
        assert!(
            s.doc
                .layers
                .name(s.doc.current_layer())
                .starts_with("Layer")
        );
    }

    #[test]
    fn layer_visibility_toggles_and_reports() {
        let mut s = Session::new();
        s.add_layer("hidden-me");
        let id = s.doc.current_layer();
        assert!(s.doc.layers.visible(id));
        s.toggle_layer_visibility(id);
        assert!(!s.doc.layers.visible(id));
        assert!(s.status.text.contains("hidden-me"), "{:?}", s.status.text);
        s.toggle_layer_visibility(id);
        assert!(s.doc.layers.visible(id));
    }

    #[test]
    fn layer_lock_toggles_and_reports() {
        let mut s = Session::new();
        s.add_layer("locked");
        let id = s.doc.current_layer();
        assert!(!s.doc.layers.locked(id));
        s.toggle_layer_lock(id);
        assert!(s.doc.layers.locked(id));
        assert!(s.status.text.contains("locked"), "{:?}", s.status.text);
    }

    #[test]
    fn new_geometry_lands_on_the_current_layer() {
        // The layer panel is only worth anything if drawing respects it.
        let mut s = Session::new();
        s.add_layer("wires");
        let layer = s.doc.current_layer();
        assert!(s.activate("line").is_ok());
        s.click_world(Vec2::ZERO, false);
        s.click_world(Vec2::new(10.0, 0.0), false);
        let e = s.doc.entities.iter().next().expect("an entity");
        assert_eq!(e.layer(), layer);
    }

    #[test]
    fn a_stale_current_layer_falls_back_to_layer_zero() {
        let mut s = Session::new();
        // Point the handle at a layer that does not exist.
        s.doc.set_current_layer(cad_doc::LayerId(999));
        assert_eq!(
            s.doc.current_layer(),
            s.doc.layers.default_layer(),
            "a stale handle must not strand new geometry"
        );
    }

    // --------------------------------------------------------------- files

    #[test]
    fn save_then_open_round_trips_the_drawing() {
        let dir = std::env::temp_dir().join(format!("cadkit-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("round.cad");

        let mut s = session_with_circle();
        s.add_layer("round");
        s.set_current_layer(s.doc.current_layer());
        s.doc.set_name("round.cad");
        assert!(s.save(Some(&path)).is_ok());
        assert_eq!(s.path.as_deref(), Some(path.as_path()));
        assert!(!s.dirty, "a successful save clears the dirty flag");

        let mut other = Session::new();
        assert!(other.open(&path).is_ok());
        assert_eq!(other.doc.entities.len(), 1);
        assert!(
            other.doc.layers.by_name("round").is_some(),
            "layers must survive the round trip"
        );
        // A successful open reports success, not an error.
        assert_eq!(other.status.level, Level::Success);
        assert!(
            other.status.text.contains("round.cad"),
            "the status should name the file: {:?}",
            other.status.text
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn opening_a_missing_file_reports_and_keeps_the_document() {
        let mut s = session_with_circle();
        let r = s.open(std::path::Path::new("definitely-not-here.cad"));
        assert!(!r.is_ok());
        assert_eq!(s.status.level, Level::Error);
        assert_eq!(
            s.doc.entities.len(),
            1,
            "a failed open must not clear the drawing"
        );
        assert!(s.path.is_none());
    }

    #[test]
    fn saving_without_a_name_asks_for_one() {
        let mut s = Session::new();
        assert_eq!(s.save(None), CommandResult::Unavailable);
        assert!(s.status.text.contains("SAVEAS"), "{:?}", s.status.text);
    }

    #[test]
    fn export_writes_a_dxf() {
        let dir = std::env::temp_dir().join(format!("cadkit-dxf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.dxf");
        let mut s = session_with_circle();
        assert!(s.export_dxf(&path).is_ok());
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("CIRCLE"),
            "the export must contain the entity"
        );
        // Export must not repoint the session at the DXF: it is not the file
        // being edited.
        assert!(s.path.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_rebuilds_the_spatial_index() {
        let dir = std::env::temp_dir().join(format!("cadkit-idx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("idx.cad");
        let mut s = session_with_circle();
        assert!(s.save(Some(&path)).is_ok());

        let mut other = Session::new();
        assert!(other.open(&path).is_ok());
        // Without a rebuilt index, hover and window selection would find nothing.
        let hits = other
            .doc
            .entities
            .candidates_in(Rect2::from_xywh(-50.0, -50.0, 200.0, 200.0));
        assert_eq!(hits.len(), 1, "{hits:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn new_document_clears_everything_but_the_view() {
        let mut s = session_with_circle();
        s.path = Some("x.cad".into());
        s.select_all();
        s.zoom_extents();
        let cam = s.viewport.cam2d;
        s.new_document();
        assert!(s.doc.entities.is_empty());
        assert!(s.path.is_none());
        assert!(s.tool.selection.is_empty());
        assert!(!s.history.can_undo());
        // The camera survives: a new drawing should not throw the user back to
        // the origin.
        assert_eq!(s.viewport.cam2d.center, cam.center);
        assert!(s.status.text.contains("New"), "{:?}", s.status.text);
    }

    // ------------------------------------------------------- command line

    #[test]
    fn the_command_line_runs_named_commands() {
        let mut s = session_with_circle();
        assert!(s.run_command_line("line").is_ok());
        assert_eq!(s.tool.id, ToolId::Line);
    }

    #[test]
    fn the_command_line_reports_unknown_commands() {
        let mut s = Session::new();
        let r = s.run_command_line("wibble");
        assert!(matches!(r, CommandResult::Error(_)));
        assert_eq!(s.status.level, Level::Error);
    }

    #[test]
    fn undo_takes_an_argument() {
        let mut s = session_with_circle();
        s.select_all();
        s.delete_selection();
        assert!(s.history.can_undo());
        s.run_command_line("undo");
        assert_eq!(s.doc.entities.len(), 1);
    }

    #[test]
    fn too_many_arguments_is_an_error_not_a_silent_ignore() {
        let mut s = Session::new();
        let r = s.run_command_line("open a.cad b.cad");
        assert!(
            !r.is_ok(),
            "a trailing argument must not be silently dropped"
        );
        assert!(s.status.text.contains("at most one"), "{:?}", s.status.text);
    }

    #[test]
    fn layer_command_selects_or_creates() {
        let mut s = Session::new();
        assert!(s.run_command_line("layer walls").is_ok());
        assert!(s.doc.layers.by_name("walls").is_some());
        assert_eq!(
            s.doc.current_layer(),
            s.doc.layers.by_name("walls").unwrap()
        );

        // Querying reports the current layer.
        assert!(s.run_command_line("layer").is_ok());
        assert!(s.status.text.contains("walls"), "{:?}", s.status.text);
    }

    #[test]
    fn an_empty_command_line_is_a_no_op() {
        let mut s = Session::new();
        assert!(s.run_command_line("   ").is_ok());
        assert_eq!(s.tool.id, ToolId::Select);
    }

    // ---------------------------------------------------------- suggestions

    #[test]
    fn commands_are_logged_in_canonical_form() {
        let mut s = Session::new();
        // `L` is an alias of `line`; the model must not see two different
        // contexts for the same command.
        assert!(s.run_command_line("L").is_ok());
        assert_eq!(s.command_log, vec!["line".to_string()]);
        assert!(s.run_command_line("circle").is_ok());
        assert_eq!(s.recent_commands(2), vec!["circle", "line"]);
    }

    #[test]
    fn a_failed_command_is_not_logged() {
        let mut s = Session::new();
        assert!(!s.run_command_line("wibble").is_ok());
        assert!(s.command_log.is_empty(), "a typo must not become context");
    }

    #[test]
    fn the_command_log_is_capped() {
        let mut s = Session::new();
        for _ in 0..30 {
            let _ = s.run_command_line("zoomall");
        }
        assert!(
            s.command_log.len() <= COMMAND_LOG_CAP,
            "{} entries",
            s.command_log.len()
        );
    }

    #[test]
    fn completions_find_a_prefix() {
        let mut s = Session::new();
        let c = s.completions("cir");
        assert_eq!(c.first().map(|s| s.as_str()), Some("circle"), "{c:?}");
    }

    #[test]
    fn completions_tolerate_a_typo() {
        let mut s = Session::new();
        let c = s.completions("mov");
        assert!(
            c.iter().any(|x| x == "move"),
            "a mistyped prefix must still find its command: {c:?}"
        );
    }

    #[test]
    fn completions_for_nonsense_are_empty() {
        let mut s = Session::new();
        assert!(s.completions("qqqqqqqqqqqq").is_empty());
    }

    #[test]
    fn the_suggestion_appears_once_the_model_exists() {
        let mut s = Session::new();
        // Nothing typed yet: deliberately untrained, so the UI shows nothing
        // rather than a confident guess out of a cold start.
        assert!(s.suggestion().is_none());
        // One command typed, and `completions` trained the model.
        let _ = s.completions("li");
        let _ = s.run_command_line("line");
        assert!(
            s.suggestion().is_some(),
            "the model should now have a top guess"
        );
    }

    #[test]
    fn the_model_is_small() {
        let mut s = Session::new();
        let _ = s.completions("l");
        assert!(
            s.predictor().parameters() < 6000,
            "{} parameters",
            s.predictor().parameters()
        );
    }

    #[test]
    fn help_lists_the_command_count() {
        let mut s = Session::new();
        assert!(s.run_command_line("help").is_ok());
        assert!(s.status.text.contains("commands"), "{:?}", s.status.text);
    }

    #[test]
    fn a_sentence_runs_through_the_command_line() {
        // "zoom all" is not a command name; it is a sentence the intent parser
        // resolves. The command line must hand it over, not reject it.
        let mut s = Session::new();
        assert!(s.run_command_line("zoom all").is_ok());
        assert_eq!(s.recent_commands(1), vec!["zoomall"]);
    }

    #[test]
    fn a_sentence_with_many_words_is_not_an_argument_error() {
        // The argument-count check used to fire before the parser ever saw the
        // line, so every sentence with more than two words was an error.
        let mut s = Session::new();
        s.viewport.cam2d = Camera2D::new(Vec2::new(800.0, 600.0));
        assert!(s.run_command_line("draw a line from 0,0 to 10,10").is_ok());
        assert_eq!(s.doc.entities.len(), 1);
        assert_eq!(s.recent_commands(1), vec!["line"]);
    }

    #[test]
    fn run_intent_draws_geometry() {
        let mut s = Session::new();
        s.viewport.cam2d = Camera2D::new(Vec2::new(800.0, 600.0));
        assert!(s.run_intent("circle at 5,5 radius 3").is_ok());
        assert_eq!(s.doc.entities.len(), 1);
        match &s.doc.entities.iter().next().unwrap().entity {
            cad_doc::EntityKind::Circle(c) => {
                assert!(
                    c.center.distance(Vec2::new(5.0, 5.0)) < 1e-3,
                    "{:?}",
                    c.center
                );
                assert!((c.radius - 3.0).abs() < 1e-3, "{}", c.radius);
            }
            other => panic!("expected a circle, got {other:?}"),
        }
    }

    #[test]
    fn run_intent_reports_nonsense_as_unavailable() {
        let mut s = Session::new();
        assert_eq!(
            s.run_intent("frobnicate the gizmo"),
            CommandResult::Unavailable
        );
        assert!(s.command_log.is_empty());
    }

    // ------------------------------------------------------------ constraints

    fn line_doc(a: Vec2, b: Vec2) -> (Document, EntityId) {
        let mut doc = Document::new();
        let layer = doc.layers.ensure_default();
        let id = doc.add(cad_doc::Entity::line(cad_geom::curve::Line::new(a, b)).with_layer(layer));
        (doc, id)
    }

    #[test]
    fn coincident_joins_two_lines() {
        let (mut doc_a, _) = line_doc(Vec2::ZERO, Vec2::new(10.0, 0.0));
        let (doc_b, _) = line_doc(Vec2::new(12.0, 0.5), Vec2::new(22.0, 0.5));
        for e in doc_b.entities.iter() {
            doc_a.add(e.clone());
        }
        let mut s = Session::new();
        s.doc = doc_a;
        s.tool.selection = s.doc.entities.handles();
        assert!(s.constrain(ConstraintKind::Coincident, None).is_ok());
        // The nearest pair was (10,0) and (12,0.5): they now coincide.
        let ends: Vec<Vec2> = s
            .doc
            .entities
            .iter()
            .flat_map(|e| e.anchor_points())
            .collect();
        let mut gap = f32::INFINITY;
        for (i, a) in ends.iter().enumerate() {
            for b in ends.iter().skip(i + 1) {
                gap = gap.min(a.distance(*b));
            }
        }
        assert!(gap < 1e-2, "nearest points did not meet: {ends:?}");
        assert!(s.history.can_undo(), "a constraint must be one undo step");
        assert_eq!(s.constraints.len(), 1);
    }

    #[test]
    fn distance_sets_the_gap_between_two_points() {
        let (mut doc_a, _) = line_doc(Vec2::ZERO, Vec2::new(2.0, 0.0));
        let (doc_b, _) = line_doc(Vec2::new(10.0, 0.0), Vec2::new(12.0, 0.0));
        for e in doc_b.entities.iter() {
            doc_a.add(e.clone());
        }
        let mut s = Session::new();
        s.doc = doc_a;
        s.tool.selection = s.doc.entities.handles();
        assert!(s.constrain(ConstraintKind::Distance, Some(5.0)).is_ok());
        let ends: Vec<Vec2> = s
            .doc
            .entities
            .iter()
            .flat_map(|e| e.anchor_points())
            .collect();
        // The two lines' nearest ends were 8 apart; now the constrained pair is 5.
        let mut found = false;
        for (i, a) in ends.iter().enumerate() {
            for b in ends.iter().skip(i + 1) {
                if (a.distance(*b) - 5.0).abs() < 5e-2 {
                    found = true;
                }
            }
        }
        assert!(found, "no pair at distance 5: {ends:?}");
    }

    #[test]
    fn distance_without_a_length_is_rejected() {
        let (doc_a, _) = line_doc(Vec2::ZERO, Vec2::new(2.0, 0.0));
        let mut s = Session::new();
        s.doc = doc_a;
        s.tool.selection = s.doc.entities.handles();
        assert!(!s.constrain(ConstraintKind::Distance, None).is_ok());
        assert!(!s.constrain(ConstraintKind::Distance, Some(-1.0)).is_ok());
        assert!(s.constraints.is_empty());
    }

    #[test]
    fn constrain_needs_two_objects() {
        let mut s = Session::new();
        assert_eq!(
            s.constrain(ConstraintKind::Coincident, None),
            CommandResult::Unavailable
        );
    }

    #[test]
    fn constrain_is_undoable() {
        let (mut doc_a, _) = line_doc(Vec2::ZERO, Vec2::new(10.0, 0.0));
        let (doc_b, _) = line_doc(Vec2::new(30.0, 0.0), Vec2::new(40.0, 0.0));
        for e in doc_b.entities.iter() {
            doc_a.add(e.clone());
        }
        let mut s = Session::new();
        s.doc = doc_a;
        s.tool.selection = s.doc.entities.handles();
        let before: Vec<Vec2> = s
            .doc
            .entities
            .iter()
            .flat_map(|e| e.anchor_points())
            .collect();
        assert!(s.constrain(ConstraintKind::Coincident, None).is_ok());
        assert!(s.undo().is_ok());
        let after: Vec<Vec2> = s
            .doc
            .entities
            .iter()
            .flat_map(|e| e.anchor_points())
            .collect();
        assert_eq!(before, after, "undo must restore the pre-solve geometry");
    }

    #[test]
    fn fix_pins_the_selection() {
        let (doc_a, _) = line_doc(Vec2::ZERO, Vec2::new(10.0, 0.0));
        let mut s = Session::new();
        s.doc = doc_a;
        s.tool.selection = s.doc.entities.handles();
        assert!(s.constrain(ConstraintKind::Fix, None).is_ok());
        assert_eq!(s.constraints.len(), 2, "a line has two anchors");
    }

    #[test]
    fn the_command_line_reaches_constraints() {
        let (mut doc_a, _) = line_doc(Vec2::ZERO, Vec2::new(10.0, 0.0));
        let (doc_b, _) = line_doc(Vec2::new(30.0, 0.0), Vec2::new(40.0, 0.0));
        for e in doc_b.entities.iter() {
            doc_a.add(e.clone());
        }
        let mut s = Session::new();
        s.doc = doc_a;
        s.tool.selection = s.doc.entities.handles();
        assert!(s.run_command_line("coincident").is_ok());
        assert!(s.run_command_line("distance 7").is_ok());
        assert_eq!(s.constraints.len(), 2);
        assert!(!s.run_command_line("distance").is_ok());
    }

    #[test]
    fn run_intent_calculates() {
        let mut s = Session::new();
        assert!(s.run_intent("= 5 + 3").is_ok());
        assert_eq!(s.status.text, "8");
    }
}
