//! CADKit — window setup, the winit event loop, and the per-frame pipeline.
//!
//! This is the only crate that knows about the window system. Everything below
//! it is testable without a display: `cad-app` owns the state, `cad-gfx` owns
//! the GPU, `cad-ui` owns the widgets, and this crate wires them together.

use std::sync::Arc;
use std::time::Instant;

use cad_app::command::CommandResult;
use cad_app::session::{Session, StatusMessage};
use cad_core::{Camera2D, Rect2, Rgba, Vec2};
use cad_doc::EntityId;
use cad_gfx::batch::{Batch2d, Batch3d, UiVertex, push_rect};
use cad_gfx::renderer::{FrameError, Gpu, SurfaceConfig};
use cad_ui::input::{Event, InputState, Key, Modifiers, MouseButton, ScrollDelta};
use cad_ui::theme::Theme;
use winit::window::Window;

/// The 2D viewport is orthographic top-down, so screen pixels map to clip space
/// with no transform. Every 2D shader derives its position from `resolution`
/// alone; this is only here so the 3D pipelines have a sane matrix.
const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Panel geometry in CSS pixels, derived from the window size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Panels {
    pub canvas: Rect2,
    pub ribbon: Rect2,
    pub command_line: Rect2,
    pub layer_panel: Rect2,
    pub properties: Rect2,
    pub status_bar: Rect2,
}

impl Panels {
    /// Split the window the way a CAD app is laid out: ribbon on top, command
    /// line and status bar at the bottom, layer and properties panels on the
    /// sides.
    pub fn layout(w: f32, h: f32) -> Self {
        let w = w.max(0.0);
        let h = h.max(0.0);
        let ribbon_h = (96.0f32).min(h * 0.2);
        let cmd_h = (28.0f32).min(h * 0.1);
        let status_h = (22.0f32).min(h * 0.1);
        let side_w = (240.0f32).min(w * 0.22);
        let body_top = ribbon_h;
        let body_bottom = (h - cmd_h - status_h).max(body_top);
        let body_h = body_bottom - body_top;
        Self {
            ribbon: Rect2::from_xywh(0.0, 0.0, w, ribbon_h),
            canvas: Rect2::from_xywh(side_w, body_top, (w - side_w * 2.0).max(0.0), body_h),
            layer_panel: Rect2::from_xywh(0.0, body_top, side_w, body_h),
            properties: Rect2::from_xywh((w - side_w).max(0.0), body_top, side_w, body_h),
            command_line: Rect2::from_xywh(0.0, body_bottom, w, cmd_h),
            status_bar: Rect2::from_xywh(0.0, (body_bottom + cmd_h).min(h), w, status_h),
        }
    }
}

/// The application: session, input, GPU and window.
pub struct App {
    pub session: Session,
    pub input: InputState,
    pub gpu: Gpu,
    /// Held as an `Arc` because `wgpu::Surface` takes ownership of the window
    /// handle and the event loop still needs to drive it.
    pub window: Arc<Window>,
    pub panels: Panels,
    pub should_close: bool,
    pub frame: u64,
    /// Command line buffer.
    pub command_line: String,
    /// Where a left-drag started, in world coordinates.
    pub selection_anchor: Option<Vec2>,
    /// Current keyboard modifiers, tracked from winit's `ModifiersChanged`.
    mods: Modifiers,
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
            should_close: false,
            frame: 0,
            command_line: String::new(),
            selection_anchor: None,
            mods: Modifiers::NONE,
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
                        self.on_text();
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
        let factor = if d.y > 0.0 { 1.1 } else { 1.0 / 1.1 };
        // Zoom about the cursor, in the *device* pixels the camera works in.
        let dpr = self.window.scale_factor() as f32;
        let cursor = self.input.mouse * dpr;
        self.session.zoom_by_at(factor, cursor);
    }

    fn on_key(&mut self, key: Key, mods: Modifiers) {
        match key {
            Key::F1 => {
                let n = self.session.commands.len();
                self.session.status =
                    StatusMessage::info(format!("{n} commands - type help for the common ones"));
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
                let outcome = self.session.tool.cancel(&mut self.session.doc);
                self.session.apply_outcome(outcome);
                self.selection_anchor = None;
                self.session.window_drag = None;
                self.session.tracking_base = None;
                return;
            }
            Key::Enter => {
                if !self.command_line.is_empty() {
                    let cmd = std::mem::take(&mut self.command_line);
                    self.run_command(&cmd);
                } else {
                    self.session.tool.finish();
                }
                self.session.tracking_base = None;
                return;
            }
            Key::Backspace => {
                self.command_line.pop();
                return;
            }
            _ => {}
        }

        if mods.command() {
            match key {
                Key::Z => {
                    self.session.undo();
                    return;
                }
                Key::Y => {
                    self.session.redo();
                    return;
                }
                Key::A => {
                    self.session.select_all();
                    return;
                }
                Key::S => {
                    self.session.status = StatusMessage::success("Saved");
                    return;
                }
                Key::N => {
                    let theme = self.session.theme;
                    let cam2d = self.session.viewport.cam2d;
                    self.session = Session::new();
                    self.session.theme = theme;
                    self.session.viewport.cam2d = cam2d;
                    self.session.viewport.canvas = self.panels.canvas;
                    return;
                }
                _ => {}
            }
        }

        // Plain letters go to the command line unless a tool is mid-prompt.
        // Plain letters type into the command line, but only when no tool is
        // waiting for a specific key.
        if self.session.tool.prompt().is_none()
            && let Some(c) = key.as_letter()
        {
            self.command_line.push(c);
            return;
        }

        let outcome = self.session.tool.on_key(&mut self.session.doc, key);
        self.session.apply_outcome(outcome);
    }

    fn on_text(&mut self) {
        if self.session.tool.prompt().is_none() {
            self.command_line.push_str(&self.input.text);
        }
    }

    /// Run a command line such as `line` or `undo 3`.
    ///
    /// Returns the command's own result so a caller (or a test) can tell whether
    /// the command ran, was unavailable, or failed. Unknown names come back as
    /// `Error` after having been reported on the status bar.
    pub fn run_command(&mut self, line: &str) -> CommandResult {
        let line = line.trim();
        if line.is_empty() {
            // Nothing typed: not an error, just no-op.
            return CommandResult::Ok;
        }
        let name = line
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        // Every arm yields a CommandResult so the caller can report success;
        // the old version mixed `()` and `CommandResult`, which did not compile.
        let result = match name.as_str() {
            "undo" | "u" => self.session.undo(),
            "redo" => self.session.redo(),
            "zoomall" | "z" => {
                self.session.zoom_extents();
                self.session.status = StatusMessage::info("Zoom extents");
                CommandResult::Ok
            }
            "zoomin" => {
                self.session.zoom_by(1.25);
                CommandResult::Ok
            }
            "zoomout" => {
                self.session.zoom_by(1.0 / 1.25);
                CommandResult::Ok
            }
            "view3d" => self.session.toggle_3d(),
            "grid" => self.session.toggle_grid(),
            "snap" => self.session.toggle_snap(),
            "ortho" => self.session.toggle_ortho(),
            "polar" => self.session.toggle_polar(),
            "erase" | "e" => {
                if self.session.tool.selection.is_empty() {
                    self.session.status = StatusMessage::info("Select objects, then Erase");
                    CommandResult::Unavailable
                } else {
                    self.session.delete_selection()
                }
            }
            // `select_all` mutates in place and reports nothing; wrap it so
            // every arm of this match yields a CommandResult.
            "selectall" | "all" => {
                self.session.select_all();
                CommandResult::Ok
            }
            "delete" | "del" => self.session.delete_selection(),
            "help" | "?" => {
                self.session.status = StatusMessage::info(
                    "Commands: line, circle, arc, polyline, rect, erase, move, undo, zoomall, view3d",
                );
                CommandResult::Ok
            }
            other => {
                let known = self.session.commands.get(other).is_some();
                if known || cad_app::session::tool_for_command(other).is_some() {
                    self.session.activate(other)
                } else {
                    self.session.status = StatusMessage::error(format!("Unknown command: {other}"));
                    CommandResult::Error(format!("Unknown command: {other}"))
                }
            }
        };
        self.session.dirty = true;
        result
    }

    /// Advance time-dependent state and render.
    ///
    /// A `FrameError::Skip` is not a failure: the window was occluded or the
    /// frame timed out, and the next one should just be attempted.
    pub fn tick_and_draw(&mut self) -> Result<(), FrameError> {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32().min(0.1);
        self.last = now;
        self.session.tick(dt);

        let dpr = self.window.scale_factor() as f32;
        let size = self.window.inner_size();
        let (cw, ch) = (size.width as f32, size.height as f32);
        let (lines, solids, ui) = build_geometry(self, dpr);

        // Resolution must be in *device* pixels: the shaders divide by it to get
        // NDC, so CSS pixels would make every quad `dpr` times too small.
        // In 2D drafting the "camera" is the orthographic top-down identity, so
        // 2D geometry needs no matrix at all and the 3D shaders simply see
        // nothing. A real 3D session would use `viewport.cam3d` here.
        self.gpu.set_globals(
            IDENTITY,
            [cw, ch],
            self.session.viewport.cam2d.scale,
            dpr,
            self.session.time,
            [
                self.session.viewport.cam3d.eye.x,
                self.session.viewport.cam3d.eye.y,
            ],
        );
        self.gpu.render(&lines, &ui, &solids)?;
        self.frame += 1;
        Ok(())
    }
}

/// Build the frame's geometry from the session.
pub fn build_geometry(app: &App, dpr: f32) -> (Batch2d, Batch3d, Vec<UiVertex>) {
    let cam = app.session.viewport.cam2d;
    let theme = app.session.theme;
    let panels = app.panels;
    let mut lines = Batch2d::new();
    let solids = Batch3d::new();
    let mut ui: Vec<UiVertex> = Vec::new();

    let c = panels.canvas;
    push_rect(
        &mut ui,
        c.min.x * dpr,
        c.min.y * dpr,
        c.width() * dpr,
        c.height() * dpr,
        theme.canvas,
    );

    if app.session.viewport.show_grid {
        draw_grid(&mut lines, &cam, c, dpr, &app.session.viewport, theme.grid);
    }
    if app.session.viewport.show_axes {
        draw_axes(
            &mut lines,
            cam.world_to_screen(Vec2::ZERO) * dpr,
            c,
            dpr,
            theme,
        );
    }

    let hover = app.session.tool.hovered;
    let selection = &app.session.tool.selection;
    let view = cam.world_viewport();
    let tolerance = cam.world_per_pixel() * 0.25;
    for (i, e) in app.session.doc.entities.iter().enumerate() {
        let id = EntityId(i as u32);
        if !e.common.visible || !app.session.doc.layers.visible(e.layer()) {
            continue;
        }
        let Some(curve) = e.as_curve() else { continue };
        if !curve.bounds().overlaps(view) {
            continue;
        }
        let pts = cad_geom::tessellate::tessellate(
            &curve,
            &cad_geom::tessellate::TessellationOptions::with_tolerance(tolerance),
        );
        let selected = selection.contains(&id);
        let hovered = hover == Some(id);
        let color = if selected {
            theme.selection
        } else {
            let lc = app.session.doc.layers.color_of(e.layer(), theme.text);
            e.resolved_color(lc)
        };
        let col = if hovered { theme.highlight } else { color };
        let width = if hovered || selected { 2.5 } else { 1.5 };
        for w in pts.windows(2) {
            lines.segment(
                cam.world_to_screen(w[0]) * dpr,
                cam.world_to_screen(w[1]) * dpr,
                col,
                width * dpr,
            );
        }
    }

    let mut rects: Vec<Rect2> = app.session.tool.preview();
    if let Some((a, b)) = app.session.window_drag {
        rects.push(Rect2::new(a, b));
    }
    for r in rects {
        stroke_world_rect(&mut lines, &cam, r, dpr, theme.rubber_band);
    }

    // Panels.
    let full_w = panels.properties.max.x;
    let full_h = (panels.status_bar.max.y).max(panels.command_line.max.y);
    push_rect(
        &mut ui,
        0.0,
        0.0,
        full_w * dpr,
        panels.ribbon.height() * dpr,
        theme.background,
    );
    push_rect(
        &mut ui,
        0.0,
        panels.layer_panel.min.y * dpr,
        panels.layer_panel.width() * dpr,
        panels.layer_panel.height() * dpr,
        theme.background,
    );
    push_rect(
        &mut ui,
        panels.properties.min.x * dpr,
        panels.properties.min.y * dpr,
        panels.properties.width() * dpr,
        panels.properties.height() * dpr,
        theme.background,
    );
    push_rect(
        &mut ui,
        0.0,
        panels.command_line.min.y * dpr,
        full_w * dpr,
        (full_h - panels.command_line.min.y) * dpr,
        theme.background,
    );

    // Separators.
    let seps = [
        (
            Vec2::new(panels.canvas.min.x * dpr, 0.0),
            Vec2::new(panels.canvas.min.x * dpr, full_h * dpr),
        ),
        (
            Vec2::new(panels.properties.min.x * dpr, 0.0),
            Vec2::new(panels.properties.min.x * dpr, full_h * dpr),
        ),
        (
            Vec2::new(0.0, panels.ribbon.max.y * dpr),
            Vec2::new(full_w * dpr, panels.ribbon.max.y * dpr),
        ),
        (
            Vec2::new(0.0, panels.command_line.min.y * dpr),
            Vec2::new(full_w * dpr, panels.command_line.min.y * dpr),
        ),
    ];
    for (a, b) in seps {
        lines.segment(a, b, theme.border, dpr);
    }

    // Crosshair.
    if c.contains(app.input.mouse) {
        let m = app.input.mouse * dpr;
        let (x0, y0) = (c.min.x * dpr, c.min.y * dpr);
        let (x1, y1) = (c.max.x * dpr, c.max.y * dpr);
        lines.segment(Vec2::new(x0, m.y), Vec2::new(x1, m.y), theme.grid, dpr);
        lines.segment(Vec2::new(m.x, y0), Vec2::new(m.x, y1), theme.grid, dpr);
        if let Some(base) = app.session.tracking_base {
            lines.segment(cam.world_to_screen(base) * dpr, m, theme.rubber_band, dpr);
        }
    }

    (lines, solids, ui)
}

fn stroke_world_rect(lines: &mut Batch2d, cam: &Camera2D, r: Rect2, dpr: f32, color: Rgba) {
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
    color: Rgba,
) {
    let step = vp.grid_spacing;
    if step <= 0.0 {
        return;
    }
    if cam.world_to_pixels(step) * dpr < 4.0 {
        return;
    }
    let view = cam.world_viewport();
    let mut x = (view.min.x / step).floor() * step;
    let mut n = 0;
    while x <= view.max.x && n < 512 {
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
    while y <= view.max.y && n < 512 {
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
        's' => T,
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
            *slot = Some(app);
            true
        }
        Err(e) => {
            eprintln!("could not initialise the GPU: {e}");
            elwt.exit();
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panels_tile_the_window_without_gaps() {
        let p = Panels::layout(1600.0, 900.0);
        assert_eq!(p.ribbon.max.y, p.canvas.min.y);
        assert_eq!(p.canvas.min.x, p.layer_panel.max.x);
        assert_eq!(p.canvas.max.x, p.properties.min.x);
        assert_eq!(p.canvas.max.y, p.command_line.min.y);
        assert_eq!(p.command_line.max.y, p.status_bar.min.y);
    }

    #[test]
    fn canvas_never_overlaps_a_panel() {
        let p = Panels::layout(1200.0, 800.0);
        assert!(p.canvas.contains(Vec2::new(600.0, 400.0)));
        assert!(!p.canvas.overlaps(p.layer_panel));
        assert!(!p.canvas.overlaps(p.properties));
        assert!(!p.canvas.overlaps(p.ribbon));
        assert!(!p.canvas.overlaps(p.command_line));
        assert!(!p.canvas.overlaps(p.status_bar));
    }

    #[test]
    fn panels_survive_a_tiny_window() {
        for (w, h) in [(100.0f32, 50.0f32), (10.0, 10.0), (0.0, 0.0)] {
            let p = Panels::layout(w, h);
            assert!(p.canvas.width() >= 0.0, "{w}x{h}");
            assert!(p.canvas.height() >= 0.0, "{w}x{h}");
            assert!(p.layer_panel.width() >= 0.0, "{w}x{h}");
            assert!(p.properties.min.x >= 0.0, "{w}x{h}");
        }
    }

    #[test]
    fn panels_handle_a_very_wide_window() {
        let p = Panels::layout(5000.0, 400.0);
        assert!(p.canvas.width() > p.layer_panel.width());
        assert!((p.layer_panel.width() - 240.0).abs() < 1e-3);
    }

    #[test]
    fn status_bar_ends_at_the_window_bottom() {
        let p = Panels::layout(1600.0, 900.0);
        assert!((p.status_bar.max.y - 900.0).abs() < 1e-3);
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
        // Punctuation has no Key variant, so it is dropped rather than guessed at.
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
}
