//! CADKit — window setup, the winit event loop, and the per-frame pipeline.
//!
//! This is the only crate that knows about the window system. Everything below
//! it is testable without a display: `cad-app` owns the state, `cad-gfx` owns
//! the GPU, `cad-ui` owns the widgets, `cad-io` owns the file formats, and this
//! crate wires them together.
//!
//! The per-frame split is deliberate:
//!
//! ```text
//!   winit event  ->  InputState  ->  chrome (widgets propose Actions)
//!                                        |
//!                        Session (document, camera, tools) <- Actions applied
//!                             |                       |
//!                    canvas geometry            3D geometry
//!                             \                   /
//!                              -> Gpu::render -> present
//! ```

mod chrome;

use std::sync::Arc;
use std::time::Instant;

use cad_app::command::CommandResult;
use cad_app::session::{Session, ViewMode};
use cad_core::{Camera2D, Rect2, Vec2, Vec3};
use cad_doc::EntityId;
use cad_gfx::batch::{Batch2d, Batch3d, UiVertex, push_rect};
use cad_gfx::renderer::{FrameError, Gpu, SurfaceConfig};
use cad_ui::input::{Event, InputState, Key, Modifiers, MouseButton, ScrollDelta};
use cad_ui::theme::Theme;
use cad_ui::widgets::{TextEditState, Ui};
use winit::window::Window;

use crate::chrome::{Action, Chrome, StatusFacts, Suggestions};

pub use chrome::{Action as ChromeAction, Panels};

/// The application: session, input, GPU, window and chrome state.
pub struct App {
    pub session: Session,
    pub input: InputState,
    pub gpu: Gpu,
    /// Held as an `Arc` because `wgpu::Surface` takes ownership of the window
    /// handle and the event loop still needs to drive it.
    pub window: Arc<Window>,
    pub panels: Panels,
    pub chrome: Chrome,
    pub should_close: bool,
    pub frame: u64,
    /// Where a left-drag started, in world coordinates.
    pub selection_anchor: Option<Vec2>,
    /// Current keyboard modifiers, tracked from winit's `ModifiersChanged`.
    mods: Modifiers,
    /// Smoothed frame rate, for the status bar.
    fps: f32,
    start: Instant,
    last: Instant,
}

impl App {
    /// Build the app for `window`, creating the GPU device.
    pub fn new(window: Arc<Window>) -> Result<Self, Box<dyn std::error::Error>> {
        let size = window.inner_size();
        let dpr = window.scale_factor() as f32;
        let w = size.width as f32 / dpr;
        let h = size.height as f32 / dpr;

        let mut session = Session::new();
        session.theme = Theme::dark();
        session.viewport.cam2d = Camera2D::new(Vec2::new(w.max(1.0), h.max(1.0)));

        let gpu = Gpu::new(
            window.clone(),
            SurfaceConfig {
                width: size.width.max(1),
                height: size.height.max(1),
                msaa_samples: 4,
            },
        )?;

        let mut input = InputState::new();
        input.viewport = Rect2::from_xywh(0.0, 0.0, w, h);

        let now = Instant::now();
        let mut app = Self {
            session,
            input,
            gpu,
            window,
            panels: Panels::layout(w, h),
            chrome: Chrome::new(),
            should_close: false,
            frame: 0,
            selection_anchor: None,
            mods: Modifiers::NONE,
            fps: 0.0,
            start: now,
            last: now,
        };
        app.relayout();
        Ok(app)
    }

    /// Recompute the panel geometry and push it into the session.
    ///
    /// Every path that changes the window size or scale factor must go through
    /// here: panel rects, the camera's viewport and the 3D aspect ratio all
    /// derive from them, and letting any two disagree is what produces a canvas
    /// that is offset from where the crosshair actually lands.
    pub fn relayout(&mut self) {
        let dpr = self.window.scale_factor() as f32;
        let size = self.window.inner_size();
        let (w, h) = (size.width as f32 / dpr, size.height as f32 / dpr);
        self.panels = Panels::layout(w, h);
        self.session.viewport.cam2d.viewport = Vec2::new(w.max(1.0), h.max(1.0));
        self.session.viewport.canvas = self.panels.canvas;
        self.session.viewport.refresh_target();
        self.input.viewport = Rect2::from_xywh(0.0, 0.0, w, h);
        self.gpu.resize(size.width.max(1), size.height.max(1));
        self.session.dirty = true;
    }

    fn now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Convert a winit physical position into CSS pixels (Y down).
    fn to_css(&self, p: &winit::dpi::PhysicalPosition<f64>) -> Vec2 {
        let dpr = self.window.scale_factor() as f32;
        Vec2::new((p.x / dpr as f64) as f32, (p.y / dpr as f64) as f32)
    }

    #[inline]
    fn in_canvas(&self, p: Vec2) -> bool {
        self.panels.canvas.contains(p)
    }

    /// Handle one winit event.
    pub fn on_window_event(&mut self, event: &winit::event::WindowEvent) {
        match event {
            winit::event::WindowEvent::CloseRequested => self.should_close = true,
            winit::event::WindowEvent::Resized(size) => {
                self.relayout();
                self.input.push(
                    &Event::Resized {
                        width: size.width,
                        height: size.height,
                    },
                    self.now(),
                );
            }
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                let p = self.to_css(position);
                self.input.push(&Event::MouseMoved { pos: p }, self.now());
                self.on_mouse_move(p);
            }
            winit::event::WindowEvent::MouseInput { state, button, .. } => {
                let p = self.input.mouse;
                let b = map_button(*button);
                let down = *state == winit::event::ElementState::Pressed;
                let ev = if down {
                    Event::MouseDown { pos: p, button: b }
                } else {
                    Event::MouseUp { pos: p, button: b }
                };
                self.input.push(&ev, self.now());
                if down {
                    self.on_mouse_down(b);
                } else if b == MouseButton::Left && self.in_canvas(p) {
                    self.on_mouse_up();
                }
            }
            winit::event::WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => ScrollDelta { x: *x, y: *y },
                    winit::event::MouseScrollDelta::PixelDelta(p) => ScrollDelta {
                        x: p.x as f32 / 50.0,
                        y: p.y as f32 / 50.0,
                    },
                };
                self.input.push(&Event::MouseWheel { delta: d }, self.now());
                self.on_scroll(d);
            }
            // winit delivers modifier state as its own `ModifiersChanged` event,
            // not as a field on the key event, so `self.mods` is the
            // authoritative current state. Reading it here is what stops Ctrl+Z
            // firing when the modifier event has not arrived yet.
            winit::event::WindowEvent::ModifiersChanged(m) => {
                // winit 0.30 replaced the per-modifier `bool` accessors with a
                // `ModifiersState` bitflag; the predicates still exist, but on
                // the state rather than on `Modifiers` itself.
                let s = m.state();
                self.mods = Modifiers {
                    shift: s.shift_key(),
                    ctrl: s.control_key(),
                    alt: s.alt_key(),
                    super_key: s.super_key(),
                };
            }
            winit::event::WindowEvent::KeyboardInput { event, .. } => {
                let mods = self.mods;
                if let Some(k) = map_key(&event.logical_key) {
                    let down = event.state.is_pressed();
                    let ev = if down {
                        Event::KeyDown { key: k, mods }
                    } else {
                        Event::KeyUp { key: k, mods }
                    };
                    self.input.push(&ev, self.now());
                    // `repeat` is the OS re-sending the same press while the key
                    // is held. It must not re-trigger a shortcut or append a
                    // second character to the command line.
                    if down && !event.repeat {
                        self.on_key(k, mods);
                    }
                }
                // Text arrives on the same event as the key. Control
                // characters (Enter, Backspace) are dropped here because they
                // have their own Key paths and would otherwise be typed twice.
                if event.state.is_pressed()
                    && !event.repeat
                    && let Some(text) = event.text.as_ref()
                {
                    let typed: String = text.chars().filter(|c| !c.is_control()).collect();
                    if !typed.is_empty() {
                        self.input.push(&Event::Text(typed), self.now());
                    }
                }
            }
            winit::event::WindowEvent::Focused(f) => {
                self.input.push(&Event::Focused(*f), self.now());
            }
            winit::event::WindowEvent::CursorLeft { .. } => {
                self.input.push(&Event::MouseLeave, self.now());
            }
            _ => {}
        }
    }

    // ------------------------------------------------------------ interaction

    fn on_mouse_move(&mut self, p: Vec2) {
        if !self.in_canvas(p) {
            return;
        }
        let world = self.session.viewport.cam2d.screen_to_world(p);
        self.session.cursor_world = world;

        if self.input.is_down(MouseButton::Middle) {
            self.session.pan_screen(self.input.mouse_delta);
            return;
        }
        // A held left button drags the selection rectangle from its anchor.
        if self.input.is_down(MouseButton::Left)
            && let Some(anchor) = self.selection_anchor
        {
            self.session.window_drag = Some((anchor, world));
        }
        self.session.update_cursor();
    }

    fn on_mouse_down(&mut self, b: MouseButton) {
        match b {
            MouseButton::Middle => {}
            MouseButton::Right => {
                // Right-click cancels, as in every CAD app.
                let outcome = self.session.tool.cancel(&mut self.session.doc);
                self.session.apply_outcome(outcome);
                self.session.window_drag = None;
                self.selection_anchor = None;
                self.session.tracking_base = None;
            }
            MouseButton::Left => {
                if !self.in_canvas(self.input.mouse) {
                    return;
                }
                let world = self.session.cursor_world;
                self.selection_anchor = Some(world);
                self.session.tracking_base = Some(world);
                self.session.click_world(world, self.input.modifiers.shift);
            }
            MouseButton::Other(_) => {}
        }
    }

    fn on_mouse_up(&mut self) {
        if self.session.window_drag.is_some() {
            self.session.finish_window();
            self.selection_anchor = None;
        }
    }

    fn on_scroll(&mut self, d: ScrollDelta) {
        if d.y == 0.0 {
            return;
        }
        // In the model-space view the wheel dollies the 3D camera; in the 2D
        // drafting view it zooms the orthographic camera. Doing both at once
        // would move the model twice as fast as the eye expects.
        if self.session.viewport.mode == ViewMode::Model3d {
            let f = if d.y > 0.0 { 1.0 / 1.1 } else { 1.1 };
            self.session.viewport.cam3d.dolly(f);
            self.session.viewport.refresh_target();
            self.session.dirty = true;
            return;
        }
        let factor = if d.y > 0.0 { 1.1 } else { 1.0 / 1.1 };
        // Zoom about the cursor, in the *device* pixels the camera works in.
        let dpr = self.window.scale_factor() as f32;
        let cursor = self.input.mouse * dpr;
        self.session.zoom_by_at(factor, cursor);
    }

    fn on_key(&mut self, key: Key, mods: Modifiers) {
        match key {
            Key::F1 => {
                self.session.status = cad_app::session::StatusMessage::prompt(format!(
                    "{} commands - type help",
                    self.session.commands.len()
                ));
                return;
            }
            Key::F2 => {
                self.session.zoom_by(0.5);
                return;
            }
            Key::F3 => {
                self.session.toggle_snap();
                return;
            }
            Key::F7 => {
                self.session.toggle_grid();
                return;
            }
            Key::F8 => {
                self.session.toggle_ortho();
                return;
            }
            Key::F10 => {
                self.session.toggle_polar();
                return;
            }
            Key::Escape => {
                // Escape closes the command line first, then cancels the tool.
                if !self.chrome.command.text.is_empty() {
                    self.chrome.command = TextEditState::new("");
                    return;
                }
                if self.chrome.open_menu.take().is_some() {
                    return;
                }
                let outcome = self.session.tool.cancel(&mut self.session.doc);
                self.session.apply_outcome(outcome);
                self.selection_anchor = None;
                self.session.window_drag = None;
                self.session.tracking_base = None;
                return;
            }
            _ => {}
        }

        if mods.command() {
            match key {
                Key::Z => {
                    let _ = self.session.undo();
                    return;
                }
                Key::Y => {
                    let _ = self.session.redo();
                    return;
                }
                Key::A => {
                    self.session.select_all();
                    return;
                }
                Key::S => {
                    if mods.shift {
                        self.chrome_open_save_as();
                    } else {
                        let _ = self.session.save(None);
                    }
                    return;
                }
                Key::O => {
                    self.session.status =
                        cad_app::session::StatusMessage::prompt("Type OPEN <path>");
                    return;
                }
                Key::N => {
                    self.session.new_document();
                    return;
                }
                _ => {}
            }
        }

        // While the command line has focus, every key belongs to it: the widget
        // reads `input.text` and `keys_pressed` itself. Anything handled here
        // would be a second, competing consumer of the same keystroke.
        if self.chrome.command.focused {
            return;
        }

        // Plain letters go to the command line unless a tool is mid-prompt.
        if self.session.tool.prompt().is_none()
            && let Some(c) = key.as_letter()
        {
            self.chrome.command.text.push(c);
            self.chrome.command.cursor = self.chrome.command.text.chars().count();
            self.chrome.command.focused = true;
            self.session.dirty = true;
            return;
        }

        let outcome = self.session.tool.on_key(&mut self.session.doc, key);
        self.session.apply_outcome(outcome);
    }

    fn chrome_open_save_as(&mut self) {
        self.session.status =
            cad_app::session::StatusMessage::prompt("Type SAVEAS <path> at the command line");
        self.chrome.command.focused = true;
    }

    /// Run a command line such as `line` or `undo 3`.
    pub fn run_command(&mut self, line: &str) -> CommandResult {
        let r = self.session.run_command_line(line);
        self.session.dirty = true;
        r
    }

    /// Apply a batch of chrome actions, in order.
    fn apply(&mut self, actions: Vec<Action>) {
        for a in actions {
            chrome::apply_action(&a, &mut self.session, &mut self.chrome);
        }
    }

    /// Advance time-dependent state and render.
    ///
    /// A `FrameError::Skip` is not a failure: the window was occluded or the
    /// frame timed out, and the next one should just be attempted.
    pub fn tick_and_draw(&mut self) -> Result<(), FrameError> {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32().clamp(0.0, 0.1);
        self.last = now;
        // Exponential smoothing: a raw 1/dt jumps wildly on a single slow frame
        // and the status bar number becomes unreadable.
        if dt > 1e-4 {
            let inst = 1.0 / dt;
            self.fps = if self.fps <= 0.0 {
                inst
            } else {
                self.fps * 0.9 + inst * 0.1
            };
        }
        self.session.tick(dt);

        let dpr = self.window.scale_factor() as f32;
        let size = self.window.inner_size();
        let (cw, ch) = (size.width as f32, size.height as f32);
        let (lines, solids, ui) = self.build_geometry(dpr);

        // In the 3D model view the shaders need the real camera; in 2D drafting
        // the 2D pipelines derive NDC from `resolution` alone, so the matrix is
        // unused there and passing the camera's is harmless.
        let target = self.session.viewport.target();
        let view_proj = if self.session.viewport.mode == ViewMode::Model3d {
            target.view_proj()
        } else {
            identity_mat4()
        };
        let eye = self.session.viewport.cam3d.eye;
        self.gpu.set_globals(
            view_proj,
            [cw, ch],
            self.session.viewport.cam2d.scale,
            dpr,
            self.session.time,
            [eye.x, eye.y],
        );
        self.gpu.render(&lines, &ui, &solids)?;
        self.frame += 1;
        // Clear the per-frame input queues now that they have all been consumed.
        self.input.end_frame();
        Ok(())
    }

    /// Status-bar read-outs for the chrome.
    fn status_facts(&self) -> StatusFacts {
        StatusFacts {
            cursor: self.session.cursor_world,
            scale: self.session.viewport.cam2d.scale,
            entities: self.session.doc.entities.len(),
            selection: self.session.tool.selection.len(),
            fps: self.fps,
            adapter: adapter_label(&self.gpu),
        }
    }

    /// Build the frame's geometry: canvas, world, chrome.
    fn build_geometry(&mut self, dpr: f32) -> (Batch2d, Batch3d, Vec<UiVertex>) {
        let mut lines = Batch2d::new();
        let mut solids = Batch3d::new();
        let mut ui: Vec<UiVertex> = Vec::new();

        let panels = self.panels;
        let cam = self.session.viewport.cam2d;
        let theme = self.session.theme;
        let facts = self.status_facts();

        // --- canvas backdrop --------------------------------------------------
        let c = panels.canvas;
        push_rect(
            &mut ui,
            c.min.x * dpr,
            c.min.y * dpr,
            c.width() * dpr,
            c.height() * dpr,
            theme.canvas,
        );

        // --- world ------------------------------------------------------------
        if self.session.viewport.show_grid {
            draw_grid(&mut lines, &cam, c, dpr, &self.session.viewport, theme.grid);
        }
        if self.session.viewport.show_axes {
            draw_axes(
                &mut lines,
                cam.world_to_screen(Vec2::ZERO) * dpr,
                c,
                dpr,
                theme,
            );
        }
        draw_entities(&mut lines, &mut solids, &self.session, &cam, c, dpr, theme);

        // --- 3D ---------------------------------------------------------------
        if self.session.viewport.mode == ViewMode::Model3d && self.session.viewport.show_grid {
            // One big quad centred on the orbit target. The shader computes the
            // lines per fragment and fades them by distance, so a CPU-generated
            // line grid would alias badly at this scale.
            let t = self.session.viewport.cam3d.target;
            let r = 4000.0f32;
            solids.ground_plane(
                Vec3::new(t.x - r, t.y - r, 0.0),
                Vec3::new(t.x + r, t.y + r, 0.0),
            );
        }

        // --- previews and crosshair ------------------------------------------
        let mut rects = self.session.tool.preview();
        if let Some((a, b)) = self.session.window_drag {
            rects.push(Rect2::new(a, b));
        }
        for r in rects {
            stroke_world_rect(&mut lines, &cam, r, dpr, theme.rubber_band);
        }

        // Snap marker: AutoCAD shows which snap fired and where. Without it the
        // user cannot tell whether a point was snapped at all.
        if let Some(s) = self
            .session
            .snap
            .last
            .filter(|_| self.session.snap.settings.enabled)
            && c.contains(self.input.mouse)
        {
            draw_snap_marker(&mut lines, &cam, s.point, dpr, theme.rubber_band);
            let label = s.kind.marker().to_string();
            let p = cam.world_to_screen(s.point) * dpr;
            let w = cad_ui::text_width(&label, theme.font_size) * dpr;
            lines.segment(
                Vec2::new(p.x + 8.0 * dpr, p.y + 10.0 * dpr),
                Vec2::new(p.x + 8.0 * dpr + w, p.y + 10.0 * dpr),
                theme.rubber_band,
                dpr,
            );
        }

        if c.contains(self.input.mouse) {
            let m = self.input.mouse * dpr;
            let (x0, y0) = (c.min.x * dpr, c.min.y * dpr);
            let (x1, y1) = (c.max.x * dpr, c.max.y * dpr);
            lines.segment(Vec2::new(x0, m.y), Vec2::new(x1, m.y), theme.grid, dpr);
            lines.segment(Vec2::new(m.x, y0), Vec2::new(m.x, y1), theme.grid, dpr);
            if let Some(base) = self.session.tracking_base {
                lines.segment(cam.world_to_screen(base) * dpr, m, theme.rubber_band, dpr);
            }
        }

        // --- chrome -----------------------------------------------------------
        // The widget pass borrows the input state and both batches mutably, so
        // it has to come after every geometry write and cannot touch the
        // session. It reports `Action`s, which are applied below.
        // Predict before the widget pass: the widget pass only has `&Session`,
        // and the model trains lazily through `&mut Session`.
        let prefix = self.chrome.command.text.clone();
        let completions = if prefix.is_empty() {
            Vec::new()
        } else {
            self.session.completions(&prefix)
        };
        let suggestions = Suggestions::build(&prefix, completions, self.session.suggestion());

        let actions = {
            let mut u = Ui::new(&mut self.input, theme, &mut lines, &mut ui).with_scale(dpr);
            chrome::draw(
                &mut u,
                &mut self.chrome,
                &self.session,
                panels,
                facts,
                &suggestions,
            )
        };
        self.apply(actions);

        (lines, solids, ui)
    }
}

/// Draw every entity: 2D curves through the tessellator, 3D solids and meshes
/// straight into the 3D batch.
fn draw_entities(
    lines: &mut Batch2d,
    solids: &mut Batch3d,
    session: &Session,
    cam: &Camera2D,
    canvas: Rect2,
    dpr: f32,
    theme: Theme,
) {
    let hover = session.tool.hovered;
    let selection = &session.tool.selection;
    let view = cam.world_viewport();
    let tolerance = cam.world_per_pixel() * 0.25;
    let dpr = if dpr > 0.0 { dpr } else { 1.0 };

    for (i, e) in session.doc.entities.iter().enumerate() {
        let id = EntityId(i as u32);
        if !e.common.visible || !session.doc.layers.visible(e.layer()) {
            continue;
        }
        let selected = selection.contains(&id);
        let hovered = hover == Some(id);
        let lc = session.doc.layers.color_of(e.layer(), theme.text);
        let base = if selected {
            theme.selection
        } else {
            e.resolved_color(lc)
        };
        let color = if hovered { theme.highlight } else { base };
        let width = if hovered || selected { 2.5 } else { 1.5 };

        // 3D geometry goes to its own batch, with the real camera.
        if e.is_3d() {
            match &e.entity {
                cad_doc::EntityKind::Box(b) => {
                    solids.solid_box(b.min, b.max, color);
                    solids.wire_box(b.min, b.max, theme.border_focused, 1.0);
                }
                cad_doc::EntityKind::Mesh(m) => solids.mesh(m, color),
                cad_doc::EntityKind::Face(f) => {
                    for p in f.loop_pts.windows(2) {
                        solids.segment(p[0], p[1], color, 1.5);
                    }
                }
                _ => {}
            }
            continue;
        }

        // Points and hatches have no curve; draw them as marks.
        match &e.entity {
            cad_doc::EntityKind::Point(p) => {
                let s = cam.world_to_screen(p.position.xy()) * dpr;
                draw_cross_mark(lines, s, 4.0 * dpr, color, dpr);
            }
            cad_doc::EntityKind::Construction(cons) => {
                let a = cam.world_to_screen(cons.from.xy()) * dpr;
                let b = cam.world_to_screen(cons.to.xy()) * dpr;
                lines.dashed(a, b, color, dpr, cad_gfx::batch::Dash::new(8.0, 6.0, 0.0));
            }
            cad_doc::EntityKind::Text(t) => {
                draw_text_entity(lines, cam, t, dpr, color, theme);
            }
            cad_doc::EntityKind::Hatch(h) => {
                if h.solid {
                    // A solid hatch is its boundary, filled in the boundary colour:
                    // the UI pipeline has no polygon fill, so this is honest
                    // rather than pretending.
                    for l in &h.loops {
                        let pts: Vec<Vec2> = l
                            .vertices
                            .iter()
                            .map(|v| cam.world_to_screen(*v) * dpr)
                            .collect();
                        lines.polyline(&pts, color, dpr);
                    }
                } else {
                    for l in &h.loops {
                        let pts: Vec<Vec2> = l
                            .vertices
                            .iter()
                            .map(|v| cam.world_to_screen(*v) * dpr)
                            .collect();
                        lines.polyline(&pts, color, dpr);
                    }
                }
            }
            _ => {}
        }

        let Some(curve) = e.as_curve() else { continue };
        if !curve.bounds().overlaps(view) {
            continue;
        }
        let pts = cad_geom::tessellate::tessellate(
            &curve,
            &cad_geom::tessellate::TessellationOptions::with_tolerance(tolerance),
        );
        let screen: Vec<Vec2> = pts.iter().map(|w| cam.world_to_screen(*w) * dpr).collect();
        // Off-canvas segments still cost vertices, so clip the polyline to the
        // canvas rect before pushing it.
        // `Rect2` has no `Mul<f32>`, and the canvas is in CSS pixels while the
        // screen-space points are already in device pixels, so scale the rect
        // explicitly.
        let clip_rect = Rect2::from_xywh(
            canvas.min.x * dpr,
            canvas.min.y * dpr,
            canvas.width() * dpr,
            canvas.height() * dpr,
        );
        clip_polyline(&screen, clip_rect, |a, b| {
            lines.segment(a, b, color, width * dpr)
        });
    }
}

/// Sutherland-Hodgman clip of a polyline to `rect`, emitting each kept run.
///
/// A drawing can extend far outside the viewport; uploading every tessellated
/// point would make the frame cost scale with the document rather than with what
/// is visible.
fn clip_polyline<F: FnMut(Vec2, Vec2)>(pts: &[Vec2], rect: Rect2, mut emit: F) {
    // Clip each segment independently and emit the visible run. Tracking
    // continuous runs across segments would avoid a duplicate vertex at each
    // boundary, but it costs a state machine per point and the shared vertex
    // costs one segment's worth of overdraw at most.
    for w in pts.windows(2) {
        if let Some((p, q)) = clip_segment_to_rect(w[0], w[1], rect)
            && p.distance_squared(q) > 1e-9
        {
            emit(p, q);
        }
    }
}

/// Liang-Barsky clip of one segment against `rect`.
///
/// Returns the clipped run's endpoints, or `None` when the segment misses the
/// rect entirely. A fully inside segment comes back unchanged, so the caller
/// does not need a separate case for it.
fn clip_segment_to_rect(a: Vec2, b: Vec2, rect: Rect2) -> Option<(Vec2, Vec2)> {
    let d = b - a;
    let mut t0 = 0.0f32;
    let mut t1 = 1.0f32;
    for (p, q) in [
        (-d.x, a.x - rect.min.x),
        (d.x, rect.max.x - a.x),
        (-d.y, a.y - rect.min.y),
        (d.y, rect.max.y - a.y),
    ] {
        if p.abs() < 1e-9 {
            // Parallel to this edge: either always inside it or always outside.
            if q < 0.0 {
                return None;
            }
            continue;
        }
        let t = q / p;
        if p < 0.0 {
            // Leaving: the entry parameter must stay below the exit.
            if t > t1 {
                return None;
            }
            t0 = t0.max(t);
        } else {
            if t < t0 {
                return None;
            }
            t1 = t1.min(t);
        }
    }
    Some((a + d * t0, a + d * t1))
}

/// An X mark, the conventional symbol for a point entity.
fn draw_cross_mark(lines: &mut Batch2d, p: Vec2, r: f32, color: cad_core::Rgba, w: f32) {
    lines.segment(p - Vec2::new(r, r), p + Vec2::new(r, r), color, w);
    lines.segment(p - Vec2::new(r, -r), p + Vec2::new(r, -r), color, w);
}

/// A square snap marker, the AutoCAD convention for object snaps.
fn draw_snap_marker(
    lines: &mut Batch2d,
    cam: &Camera2D,
    world: Vec2,
    dpr: f32,
    color: cad_core::Rgba,
) {
    let p = cam.world_to_screen(world) * dpr;
    let r = 6.0 * dpr;
    lines.segment(p - Vec2::new(r, 0.0), p + Vec2::new(r, 0.0), color, dpr);
    lines.segment(p - Vec2::new(0.0, r), p + Vec2::new(0.0, r), color, dpr);
}

/// Text as stroked glyphs through the built-in font, positioned and scaled in world space.
fn draw_text_entity(
    lines: &mut Batch2d,
    cam: &Camera2D,
    t: &cad_doc::Text,
    dpr: f32,
    color: cad_core::Rgba,
    _theme: Theme,
) {
    // Glyph strokes are in font units; project the unit box through the camera so
    // the text tracks zoom and rotation exactly like any other geometry.
    let scale = cam.scale * t.height.max(1e-3) / cad_ui::font::UNITS_H;
    let (s, c) = t.rotation.sin_cos();
    let ox = t.insert.x;
    let oy = t.insert.y;
    for ch in t.value.chars() {
        let g = cad_ui::font::glyph(ch);
        for stroke in g.strokes {
            for pair in stroke.windows(2) {
                let pts: Vec<Vec2> = [pair[0], pair[1]]
                    .iter()
                    .map(|p| {
                        let wx = ox + (p.0 * c - p.1 * s) * scale;
                        let wy = oy + (p.0 * s + p.1 * c) * scale;
                        cam.world_to_screen(Vec2::new(wx, wy)) * dpr
                    })
                    .collect();
                lines.segment(pts[0], pts[1], color, dpr);
            }
        }
    }
}

fn stroke_world_rect(
    lines: &mut Batch2d,
    cam: &Camera2D,
    r: Rect2,
    dpr: f32,
    color: cad_core::Rgba,
) {
    let a = cam.world_to_screen(r.min) * dpr;
    let b = cam.world_to_screen(r.max) * dpr;
    lines.segment(a, Vec2::new(b.x, a.y), color, dpr);
    lines.segment(Vec2::new(b.x, a.y), b, color, dpr);
    lines.segment(b, Vec2::new(a.x, b.y), color, dpr);
    lines.segment(Vec2::new(a.x, b.y), a, color, dpr);
}

fn draw_grid(
    lines: &mut Batch2d,
    cam: &Camera2D,
    canvas: Rect2,
    dpr: f32,
    vp: &cad_app::session::Viewport,
    color: cad_core::Rgba,
) {
    let step = vp.grid_spacing;
    if step <= 0.0 {
        return;
    }
    if cam.world_to_pixels(step) * dpr < 4.0 {
        return;
    }
    let view = cam.world_viewport();
    // The 512 cap bounds a pathological zoom-out; past that the grid is denser
    // than a pixel anyway and drawing it costs more than it shows.
    const MAX: usize = 512;
    let mut x = (view.min.x / step).floor() * step;
    let mut n = 0;
    while x <= view.max.x && n < MAX {
        let sx = cam.world_to_screen(Vec2::new(x, 0.0)).x * dpr;
        lines.segment(
            Vec2::new(sx, canvas.min.y * dpr),
            Vec2::new(sx, canvas.max.y * dpr),
            color,
            dpr,
        );
        x += step;
        n += 1;
    }
    let mut y = (view.min.y / step).floor() * step;
    let mut n = 0;
    while y <= view.max.y && n < MAX {
        let sy = cam.world_to_screen(Vec2::new(0.0, y)).y * dpr;
        lines.segment(
            Vec2::new(canvas.min.x * dpr, sy),
            Vec2::new(canvas.max.x * dpr, sy),
            color,
            dpr,
        );
        y += step;
        n += 1;
    }
}

fn draw_axes(lines: &mut Batch2d, origin: Vec2, canvas: Rect2, dpr: f32, theme: Theme) {
    lines.segment(
        Vec2::new(canvas.min.x * dpr, origin.y),
        Vec2::new(canvas.max.x * dpr, origin.y),
        theme.axis_x,
        dpr,
    );
    lines.segment(
        Vec2::new(origin.x, canvas.min.y * dpr),
        Vec2::new(origin.x, canvas.max.y * dpr),
        theme.axis_y,
        dpr,
    );
}

const fn identity_mat4() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// A short, fixed label for the status bar.
///
/// `AdapterInfo` is not `'static`, so the name is copied once and interned as a
/// leaked string: it is at most a few dozen bytes and it must outlive the frame
/// that formats it.
fn adapter_label(gpu: &Gpu) -> &'static str {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static CACHE: Mutex<Option<HashMap<String, &'static str>>> = Mutex::new(None);
    let key = format!("{:?}", gpu.adapter_info.name);
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let map = cache.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&key) {
        return s;
    }
    let leaked: &'static str = Box::leak(key.clone().into_boxed_str());
    map.insert(key, leaked);
    leaked
}

fn map_button(b: winit::event::MouseButton) -> MouseButton {
    match b {
        winit::event::MouseButton::Left => MouseButton::Left,
        winit::event::MouseButton::Right => MouseButton::Right,
        winit::event::MouseButton::Middle => MouseButton::Middle,
        winit::event::MouseButton::Other(n) => MouseButton::Other(n as u8),
        // Added in winit 0.30. Treat them as "other" rather than dropping the
        // event, so a stray click still closes menus.
        winit::event::MouseButton::Back | winit::event::MouseButton::Forward => {
            MouseButton::Other(0)
        }
    }
}

fn map_key(k: &winit::keyboard::Key) -> Option<Key> {
    match k {
        // winit reports printable keys as a `SmolStr`, which on some layouts
        // holds more than one character.
        winit::keyboard::Key::Character(c) => c.chars().next().and_then(char_to_key),
        winit::keyboard::Key::Named(n) => named_key(n),
        _ => None,
    }
}

/// Map a winit [`NamedKey`] to the app's [`Key`].
///
/// Punctuation is deliberately absent: winit reports those as
/// `Key::Character`, not as named keys, so they arrive through
/// [`char_to_key`] instead. Getting this backwards would make "-" unmappable
/// on every layout.
fn named_key(n: &winit::keyboard::NamedKey) -> Option<Key> {
    use winit::keyboard::NamedKey as N;
    Some(match n {
        N::Escape => Key::Escape,
        // winit 0.30 reports punctuation and the numpad's non-digit keys as
        // `Key::Character`, not as `NamedKey` variants -- there is no
        // `NumpadEnter`, `Minus` or `Slash` in the enum. They arrive through
        // `char_to_key` instead.
        N::Enter => Key::Enter,
        N::Tab => Key::Tab,
        N::Backspace => Key::Backspace,
        N::Delete => Key::Delete,
        N::Insert => Key::Insert,
        N::Home => Key::Home,
        N::End => Key::End,
        N::PageUp => Key::PageUp,
        N::PageDown => Key::PageDown,
        N::ArrowLeft => Key::Left,
        N::ArrowRight => Key::Right,
        N::ArrowUp => Key::Up,
        N::ArrowDown => Key::Down,
        N::Space => Key::Space,
        N::F1 => Key::F1,
        N::F2 => Key::F2,
        N::F3 => Key::F3,
        N::F4 => Key::F4,
        N::F5 => Key::F5,
        N::F6 => Key::F6,
        N::F7 => Key::F7,
        N::F8 => Key::F8,
        N::F9 => Key::F9,
        N::F10 => Key::F10,
        N::F11 => Key::F11,
        N::F12 => Key::F12,
        N::Shift => Key::Shift,
        N::Control => Key::Control,
        N::Alt => Key::Alt,
        N::Super => Key::Super,
        N::Meta => Key::Super,
        _ => return None,
    })
}

fn char_to_key(c: char) -> Option<Key> {
    use Key::*;
    Some(match c.to_ascii_lowercase() {
        'a' => A,
        'b' => B,
        'c' => C,
        'd' => D,
        'e' => E,
        'f' => F,
        'g' => G,
        'h' => H,
        'i' => I,
        'j' => J,
        'k' => K,
        'l' => L,
        'm' => M,
        'n' => N,
        'o' => O,
        'p' => P,
        'q' => Q,
        'r' => R,
        's' => S,
        't' => T,
        'u' => U,
        'v' => V,
        'w' => W,
        'x' => X,
        'y' => Y,
        'z' => Z,
        '0' => Num0,
        '1' => Num1,
        '2' => Num2,
        '3' => Num3,
        '4' => Num4,
        '5' => Num5,
        '6' => Num6,
        '7' => Num7,
        '8' => Num8,
        '9' => Num9,
        _ => return None,
    })
}

/// The winit application handler.
///
/// winit 0.30 deprecated the `FnMut(Event, &ActiveEventLoop)` form of `run` in
/// favour of `ApplicationHandler`; using the old one would fail CI's
/// `-D warnings` clippy gate on the deprecation attribute alone.
#[derive(Default)]
struct Handler {
    /// `None` until the first `Resumed`, because winit only guarantees that a
    /// surface can be created after that event.
    app: Option<App>,
}

impl winit::application::ApplicationHandler for Handler {
    fn resumed(&mut self, elwt: &winit::event_loop::ActiveEventLoop) {
        if self.app.is_none() {
            create_window(elwt, &mut self.app);
        }
    }

    fn window_event(
        &mut self,
        elwt: &winit::event_loop::ActiveEventLoop,
        _id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(app) = self.app.as_mut() else {
            return;
        };
        app.on_window_event(&event);
        match event {
            winit::event::WindowEvent::CloseRequested => elwt.exit(),
            winit::event::WindowEvent::ScaleFactorChanged { .. } => {
                // Panel geometry is in CSS pixels, so a scale change means a
                // re-layout even though the pixel size may be unchanged.
                app.relayout();
            }
            winit::event::WindowEvent::Occluded(true) => app.session.dirty = true,
            _ => {}
        }
    }

    fn about_to_wait(&mut self, elwt: &winit::event_loop::ActiveEventLoop) {
        let Some(app) = self.app.as_mut() else {
            return;
        };
        // Drive redraws from the input stream rather than a timer, which keeps
        // the CPU idle when nothing is happening.
        let busy = !app.input.released.is_empty()
            || !app.input.keys_pressed.is_empty()
            || !app.input.text.is_empty()
            || app.input.mouse_delta.length_squared() > 0.0
            || !app.input.pressed.is_empty()
            || app.session.dirty
            || app.input.scroll.y != 0.0;
        if !busy {
            return;
        }
        app.session.dirty = false;
        match app.tick_and_draw() {
            Ok(()) | Err(FrameError::Skip) => {}
            Err(FrameError::Outdated) | Err(FrameError::Lost) => {
                let size = app.window.inner_size();
                app.gpu.resize(size.width.max(1), size.height.max(1));
            }
            Err(e @ FrameError::Invalid) => {
                eprintln!("render error: {e}");
                elwt.exit();
            }
        }
        app.window.request_redraw();
    }
}

/// Create the window and run the event loop until it is closed.
pub fn main_loop() -> Result<Option<String>, Box<dyn std::error::Error>> {
    let event_loop = winit::event_loop::EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut handler = Handler::default();
    event_loop.run_app(&mut handler)?;
    Ok(handler.app.map(|a| format!("{:?}", a.gpu.adapter_info)))
}

/// Create the window and build the app around it.
///
/// Returns `false` if either step failed, having already asked the event loop to
/// exit, so the caller does not have to.
fn create_window(elwt: &winit::event_loop::ActiveEventLoop, slot: &mut Option<App>) -> bool {
    // A file named on the command line is opened before the first frame, so
    // double-clicking a .cad or .dxf works.
    let argv: Vec<String> = std::env::args().skip(1).collect();

    let attrs = winit::window::WindowAttributes::default()
        .with_title("CADKit")
        .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 900.0))
        .with_min_inner_size(winit::dpi::LogicalSize::new(900.0, 600.0));
    let window = match elwt.create_window(attrs) {
        Ok(w) => Arc::new(w),
        Err(e) => {
            eprintln!("could not create the window: {e}");
            elwt.exit();
            return false;
        }
    };
    match App::new(window) {
        Ok(mut app) => {
            // Frame the (empty) document so the grid and axes are visible.
            app.session.viewport.cam2d.scale = 4.0;
            app.relayout();
            // Announce a successful start on stderr. Without this, "the app did
            // not open" and "the app opened and then failed silently" are
            // indistinguishable from outside the machine; the CI smoke test
            // greps for this line to prove the GPU came up.
            eprintln!(
                "cadkit: started on {:?} ({:?}, {:?})",
                app.gpu.adapter_info.backend, app.gpu.adapter_info.name, app.gpu.format
            );
            if let Some(path) = argv.first() {
                let r = app.session.open(std::path::Path::new(path));
                match r {
                    CommandResult::Ok => {
                        eprintln!("cadkit: opened {path}");
                    }
                    CommandResult::Error(m) => {
                        eprintln!("cadkit: could not open {path}: {m}");
                    }
                    _ => {}
                }
                app.relayout();
            }
            *slot = Some(app);
            true
        }
        Err(e) => {
            eprintln!("cadkit: could not initialise the GPU: {e}");
            elwt.exit();
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_leaves_a_fully_inside_segment_alone() {
        let r = Rect2::from_xywh(0.0, 0.0, 100.0, 100.0);
        let (p, q) = clip_segment_to_rect(Vec2::new(10.0, 10.0), Vec2::new(20.0, 20.0), r).unwrap();
        assert!(p.distance(Vec2::new(10.0, 10.0)) < 1e-4, "{p:?}");
        assert!(q.distance(Vec2::new(20.0, 20.0)) < 1e-4, "{q:?}");
    }

    #[test]
    fn clip_truncates_a_segment_entering_from_the_left() {
        let r = Rect2::from_xywh(0.0, 0.0, 100.0, 100.0);
        // Starts outside on the left, ends inside: the entry point is on the left
        // edge and the exit point is the original end.
        let (p, q) =
            clip_segment_to_rect(Vec2::new(-100.0, 50.0), Vec2::new(50.0, 50.0), r).unwrap();
        assert!((p.x - 0.0).abs() < 1e-3, "{p:?}");
        assert!((p.y - 50.0).abs() < 1e-3, "{p:?}");
        assert!((q.x - 50.0).abs() < 1e-3, "{q:?}");
    }

    #[test]
    fn clip_truncates_a_segment_leaving_to_the_right() {
        let r = Rect2::from_xywh(0.0, 0.0, 100.0, 100.0);
        let (p, q) =
            clip_segment_to_rect(Vec2::new(50.0, 50.0), Vec2::new(150.0, 50.0), r).unwrap();
        assert!((p.x - 50.0).abs() < 1e-3, "{p:?}");
        assert!((q.x - 100.0).abs() < 1e-3, "{q:?}");
    }

    #[test]
    fn clip_rejects_a_fully_outside_segment() {
        let r = Rect2::from_xywh(0.0, 0.0, 10.0, 10.0);
        assert!(
            clip_segment_to_rect(Vec2::new(100.0, 100.0), Vec2::new(200.0, 200.0), r).is_none()
        );
        // Passing entirely beside the rect on one axis is also a miss, and takes
        // the `p.abs() < 1e-9` branch.
        assert!(clip_segment_to_rect(Vec2::new(-5.0, 100.0), Vec2::new(20.0, 100.0), r).is_none());
    }

    #[test]
    fn clipping_emits_the_visible_runs() {
        // One segment enters, one leaves: both have a visible part.
        let r = Rect2::from_xywh(0.0, 0.0, 100.0, 100.0);
        let pts = [
            Vec2::new(-50.0, 50.0),
            Vec2::new(50.0, 50.0),
            Vec2::new(150.0, 50.0),
        ];
        let mut n = 0;
        clip_polyline(&pts, r, |_, _| n += 1);
        assert_eq!(n, 2);
    }

    #[test]
    fn clipping_an_entirely_outside_polyline_emits_nothing() {
        let r = Rect2::from_xywh(0.0, 0.0, 10.0, 10.0);
        let pts = [Vec2::new(100.0, 100.0), Vec2::new(200.0, 200.0)];
        let mut n = 0;
        clip_polyline(&pts, r, |_, _| n += 1);
        assert_eq!(n, 0);
    }

    #[test]
    fn clipping_a_closed_inside_polyline_keeps_every_edge() {
        let r = Rect2::from_xywh(0.0, 0.0, 100.0, 100.0);
        let pts = [
            Vec2::new(10.0, 10.0),
            Vec2::new(90.0, 10.0),
            Vec2::new(90.0, 90.0),
            Vec2::new(10.0, 90.0),
        ];
        let mut n = 0;
        clip_polyline(&pts, r, |_, _| n += 1);
        assert_eq!(n, 3);
    }

    #[test]
    fn a_zero_length_segment_inside_the_rect_is_dropped() {
        // Clipping collapses a degenerate segment to zero length; emitting it
        // would cost a quad per tessellation artefact.
        let r = Rect2::from_xywh(0.0, 0.0, 100.0, 100.0);
        let pts = [Vec2::new(50.0, 50.0), Vec2::new(50.0, 50.0)];
        let mut n = 0;
        clip_polyline(&pts, r, |_, _| n += 1);
        assert_eq!(n, 0);
    }

    #[test]
    fn key_mapping_covers_the_cad_shortcuts() {
        use winit::keyboard::Key as W;
        assert_eq!(
            map_key(&W::Named(winit::keyboard::NamedKey::F3)),
            Some(Key::F3)
        );
        assert_eq!(
            map_key(&W::Named(winit::keyboard::NamedKey::Escape)),
            Some(Key::Escape)
        );
        // winit reports printable keys as a SmolStr, so these are not `char`.
        assert_eq!(map_key(&W::Character("l".into())), Some(Key::L));
        assert_eq!(map_key(&W::Character("7".into())), Some(Key::Num7));
        // Uppercase arrives as one character and must fold to the same key, or
        // Shift+letter would stop working.
        assert_eq!(map_key(&W::Character("L".into())), Some(Key::L));
        // Punctuation has no printable-Key path other than the named ones.
        assert_eq!(map_key(&W::Character("%".into())), None);
    }

    #[test]
    fn modifier_keys_map_from_named_keys() {
        use winit::keyboard::Key as W;
        use winit::keyboard::NamedKey as N;
        // Modifiers are NamedKey::Shift/Control/..., not per-side variants.
        assert_eq!(map_key(&W::Named(N::Shift)), Some(Key::Shift));
        assert_eq!(map_key(&W::Named(N::Control)), Some(Key::Control));
        assert_eq!(map_key(&W::Named(N::Super)), Some(Key::Super));
        // Left and right both report the same logical key, which is what makes
        // Ctrl+Z work from either side.
        assert_eq!(map_key(&W::Named(N::Alt)), Some(Key::Alt));
    }

    #[test]
    fn digits_map_from_characters_and_punctuation_does_not() {
        // winit 0.30 dropped every punctuation `NamedKey` and the numpad's
        // non-digit keys, so `-`, `.`, `/` and numpad-enter all arrive as
        // `Key::Character`. `char_to_key` has no arms for them -- there is no
        // `Key::Minus` producer -- so they are dropped rather than guessed at.
        // Punctuation reaches the command line as `Event::Text` instead, which
        // is what makes paths like `open a.dxf` typeable at all.
        use winit::keyboard::Key as W;
        assert_eq!(map_key(&W::Character("7".into())), Some(Key::Num7));
        for s in ["-", ".", ",", "/"] {
            assert_eq!(map_key(&W::Character(s.into())), None, "{s}");
        }
        assert_eq!(map_key(&W::Character("%".into())), None);
        assert_eq!(map_key(&W::Character("é".into())), None);
    }
}
