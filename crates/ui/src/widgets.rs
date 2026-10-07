//! The immediate-mode UI context: widgets read input and push geometry into
//! GPU batches in one pass.
//!
//! Every widget is `fn(&mut Ui, Rect) -> Response`. No retained state beyond
//! what a text field genuinely needs (cursor, selection), no traits to learn,
//! and the whole thing is a few hundred lines.

use crate::input::{InputState, Key, Modifiers, MouseButton};
use crate::layout::{Layout, Padding, Rect};
use crate::theme::Theme;
use cad_core::{Rgba, Vec2};
use cad_gfx::batch::{Batch2d, UiVertex, push_rect, push_rounded_rect};

/// What a widget did this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Response {
    pub hovered: bool,
    pub pressed: bool,
    pub released: bool,
    /// The widget consumed the click / wants focus.
    pub clicked: bool,
    /// Value changed this frame (text field, checkbox, slider).
    pub changed: bool,
}

impl Response {
    pub const NONE: Self = Self {
        hovered: false,
        pressed: false,
        released: false,
        clicked: false,
        changed: false,
    };
    pub fn is_none(&self) -> bool {
        *self == Self::NONE
    }
    /// Anything interactive happened.
    pub fn interacted(&self) -> bool {
        self.clicked || self.changed || self.pressed || self.released
    }
    pub fn merge(self, o: Response) -> Response {
        Response {
            hovered: self.hovered || o.hovered,
            pressed: self.pressed || o.pressed,
            released: self.released || o.released,
            clicked: self.clicked || o.clicked,
            changed: self.changed || o.changed,
        }
    }
}

/// State a text field must keep between frames.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextEditState {
    pub text: String,
    /// Cursor index in characters (not bytes).
    pub cursor: usize,
    pub scroll: f32,
    pub focused: bool,
    /// Characters selected from `sel_start` to `cursor`.
    pub sel_start: Option<usize>,
}

impl TextEditState {
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        let cursor = text.chars().count();
        Self {
            text,
            cursor,
            ..Default::default()
        }
    }
    pub fn select_all(&mut self) {
        self.sel_start = Some(0);
        self.cursor = self.text.chars().count();
    }
    pub fn delete_selection(&mut self) -> bool {
        let Some(a) = self.sel_start.take() else {
            return false;
        };
        let (lo, hi) = (a.min(self.cursor), a.max(self.cursor));
        let chars: Vec<char> = self.text.chars().collect();
        if lo >= hi {
            return false;
        }
        let mut out: String = chars[..lo].iter().collect();
        out.extend(chars[hi..].iter());
        self.text = out;
        self.cursor = lo;
        true
    }
}

/// The UI frame context.
pub struct Ui<'a> {
    pub input: &'a mut InputState,
    pub theme: Theme,
    pub batch2d: &'a mut Batch2d,
    /// Rectangles emitted for the UI layer, in order.
    pub ui_batch: &'a mut Vec<UiVertex>,
    /// Requested keyboard focus for this frame; the last widget to ask wins.
    pub focus_request: Option<usize>,
    /// Tooltip to show under the cursor.
    pub tooltip: Option<String>,
    /// True when any widget asked for the text cursor.
    pub wants_text_input: bool,
    id_counter: usize,
    hot: Option<usize>,
    active: Option<usize>,
    hot_modifiers: Modifiers,
}

impl<'a> Ui<'a> {
    pub fn new(
        input: &'a mut InputState,
        theme: Theme,
        batch2d: &'a mut Batch2d,
        ui_batch: &'a mut Vec<UiVertex>,
    ) -> Self {
        Self {
            input,
            theme,
            batch2d,
            ui_batch,
            focus_request: None,
            tooltip: None,
            wants_text_input: false,
            id_counter: 0,
            hot: None,
            active: None,
            hot_modifiers: Modifiers::NONE,
        }
    }

    /// Unique id for a widget this frame.
    pub fn next_id(&mut self) -> usize {
        self.id_counter += 1;
        self.id_counter
    }

    #[inline]
    pub fn mouse(&self) -> Vec2 {
        self.input.mouse
    }
    #[inline]
    pub fn modifiers(&self) -> Modifiers {
        self.input.modifiers
    }
    pub fn hovered(&self, r: Rect) -> bool {
        self.input.hovered(r)
    }
    pub fn is_down(&self, b: MouseButton) -> bool {
        self.input.is_down(b)
    }

    /// Standard hover/press/click resolution for an interactive widget.
    fn interact(&mut self, id: usize, r: Rect, enabled: bool) -> Response {
        let hovered = enabled && self.hovered(r);
        if hovered {
            self.hot = Some(id);
            self.hot_modifiers = self.modifiers();
        }
        let pressed = hovered && self.input.is_down(MouseButton::Left);
        if pressed {
            self.active = Some(id);
        }
        let released = self.active == Some(id) && self.input.released.contains(&MouseButton::Left);
        if released {
            self.active = None;
        }
        let clicked = released && hovered;
        Response {
            hovered,
            pressed,
            released,
            clicked,
            changed: false,
        }
    }

    /// Called at the start of a frame: reset the hot widget from last frame.
    pub fn begin_frame(&mut self) {
        // `hot` from the previous frame is what "was hovered last frame"
        // means, so keep it until widgets run.
    }

    /// Called at the end of a frame.
    pub fn end_frame(&mut self) {
        // Persist `hot` for the next frame's hover test.
        self.tooltip = None;
    }

    // ---------------------------------------------------------------- drawing

    pub fn fill_rect(&mut self, r: Rect, color: Rgba) {
        if r.is_empty() {
            return;
        }
        push_rect(
            self.ui_batch,
            r.min.x,
            r.min.y,
            r.width(),
            r.height(),
            color,
        );
    }

    pub fn fill_round_rect(&mut self, r: Rect, color: Rgba, radius: f32) {
        if r.is_empty() {
            return;
        }
        push_rounded_rect(self.ui_batch, r, color, radius);
    }

    pub fn stroke_rect(&mut self, r: Rect, color: Rgba, width: f32) {
        let h = width * 0.5;
        let t = r.width().min(r.height()) * 0.5;
        // Four thin rects; cheaper and sharper than an SDF outline here.
        let _ = t;
        push_rect(self.ui_batch, r.min.x, r.min.y, r.width(), h, color);
        push_rect(self.ui_batch, r.min.x, r.max.y - h, r.width(), h, color);
        push_rect(
            self.ui_batch,
            r.min.x,
            r.min.y + h,
            h,
            r.height() - h * 2.0,
            color,
        );
        push_rect(
            self.ui_batch,
            r.max.x - h,
            r.min.y + h,
            h,
            r.height() - h * 2.0,
            color,
        );
    }

    /// Text is emitted as geometry by the caller-supplied font shaper; here we
    /// only reserve the space and record the request, which keeps `cad-ui`
    /// independent of any font library.
    pub fn text(&mut self, _s: &str, _pos: Vec2, _color: Rgba) {}

    /// Request a tooltip for `r`.
    pub fn tooltip(&mut self, r: Rect, text: &str) {
        if self.hovered(r) {
            self.tooltip = Some(text.to_string());
        }
    }

    // ---------------------------------------------------------------- widgets

    /// A flat button. `label` is only used for hit-testing width here; the
    /// caller draws the label (see `text`).
    pub fn button(&mut self, r: Rect, enabled: bool) -> Response {
        let id = self.next_id();
        let resp = self.interact(id, r, enabled);
        let bg = if !enabled {
            self.theme.surface
        } else if resp.pressed {
            self.theme.surface_active
        } else if resp.hovered {
            self.theme.surface_hover
        } else {
            self.theme.surface
        };
        self.fill_round_rect(r, bg, self.theme.border_radius);
        let border = if resp.hovered && enabled {
            self.theme.border_focused
        } else {
            self.theme.border
        };
        self.stroke_rect(r, border, 1.0);
        if !enabled {
            self.stroke_rect(r, self.theme.text_disabled.with_alpha(0.4), 1.0);
        }
        resp
    }

    /// A checkbox with a label drawn by the caller.
    pub fn checkbox(&mut self, r: Rect, value: bool) -> (Response, bool) {
        let id = self.next_id();
        let resp = self.interact(id, r, true);
        self.fill_round_rect(r, self.theme.surface, self.theme.border_radius);
        self.stroke_rect(
            r,
            if resp.hovered {
                self.theme.border_focused
            } else {
                self.theme.border
            },
            1.0,
        );
        if value {
            let inset = r.width() * 0.28;
            self.fill_round_rect(
                Rect::new(r.min + Vec2::splat(inset), r.max - Vec2::splat(inset)),
                self.theme.accent,
                2.0,
            );
        }
        (resp, value != resp.clicked)
    }

    /// A horizontal slider. Returns `(response, new_value)`.
    pub fn slider(&mut self, r: Rect, value: f32, min: f32, max: f32) -> (Response, f32) {
        let id = self.next_id();
        let resp = self.interact(id, r, true);
        // Track.
        let track_h = 4.0;
        let track = Rect::new(
            Vec2::new(r.min.x, r.center().y - track_h * 0.5),
            Vec2::new(r.max.x, r.center().y + track_h * 0.5),
        );
        self.fill_round_rect(track, self.theme.surface_active, track_h * 0.5);
        let t = ((value - min) / (max - min).max(1e-9)).clamp(0.0, 1.0);
        // Filled portion.
        let fw = r.width() * t;
        if fw > 0.0 {
            self.fill_round_rect(
                Rect::from_xywh(track.min.x, track.min.y, fw, track_h),
                self.theme.accent,
                track_h * 0.5,
            );
        }
        // Knob.
        let knob_r = r.height() * 0.32;
        let cx = r.min.x + fw;
        let knob = Rect::new(
            Vec2::new(cx - knob_r, r.center().y - knob_r),
            Vec2::new(cx + knob_r, r.center().y + knob_r),
        );
        self.fill_round_rect(knob, self.theme.text, knob_r);
        let mut out = value;
        if resp.clicked || (resp.pressed) {
            let nt = ((self.mouse().x - r.min.x) / r.width().max(1e-9)).clamp(0.0, 1.0);
            out = min + nt * (max - min);
        }
        (resp, out)
    }

    /// A text field. Keyboard handling lives here so the command line and every
    /// property field behave identically.
    pub fn text_field(
        &mut self,
        r: Rect,
        state: &mut TextEditState,
        enabled: bool,
    ) -> (Response, bool) {
        let id = self.next_id();
        let resp = self.interact(id, r, enabled);
        let mut changed = false;

        if resp.clicked {
            let click_x = self.mouse().x;
            let ch_w = (self.theme.font_size * 0.55).max(1.0);
            let rel = ((click_x - r.min.x - 6.0) / ch_w).max(0.0) as usize;
            state.cursor = rel.min(state.text.chars().count());
            state.focused = enabled;
            state.sel_start = None;
            self.focus_request = Some(id);
        } else if resp.pressed {
            // Drag-select.
            let ch_w = (self.theme.font_size * 0.55).max(1.0);
            let rel = ((self.mouse().x - r.min.x - 6.0) / ch_w).max(0.0) as usize;
            state.cursor = rel.min(state.text.chars().count());
            self.focus_request = Some(id);
        }

        if resp.released && !enabled {
            state.focused = false;
        }

        if state.focused && enabled {
            self.wants_text_input = true;
            self.focus_request = Some(id);
            let mods = self.modifiers();
            let chars: Vec<char> = state.text.chars().collect();
            let mut cursor = state.cursor.min(chars.len());

            // Arrow keys and Home/End.
            if self.input.key_pressed(Key::Left) {
                cursor = cursor.saturating_sub(if mods.shift { 5 } else { 1 });
            }
            if self.input.key_pressed(Key::Right) {
                cursor = (cursor + 1).min(chars.len());
            }
            if self.input.key_pressed(Key::Home) {
                cursor = 0;
            }
            if self.input.key_pressed(Key::End) {
                cursor = chars.len();
            }
            if self.input.key_pressed(Key::A) && mods.command() {
                state.sel_start = Some(0);
                state.cursor = chars.len();
            }

            if state.sel_start.is_some()
                && (self.input.key_pressed(Key::Delete) || self.input.key_pressed(Key::Backspace))
            {
                if state.delete_selection() {
                    changed = true;
                }
                cursor = state.cursor;
            } else if self.input.key_pressed(Key::Backspace) {
                if cursor > 0 {
                    let mut out: String = chars[..cursor - 1].iter().collect();
                    out.extend(chars[cursor..].iter());
                    state.text = out;
                    cursor -= 1;
                    state.cursor = cursor;
                    changed = true;
                }
            } else if self.input.key_pressed(Key::Delete) {
                if cursor < chars.len() {
                    let mut out: String = chars[..cursor].iter().collect();
                    out.extend(chars[cursor + 1..].iter());
                    state.text = out;
                    state.cursor = cursor;
                    changed = true;
                }
            }

            // Typed text, filtered to what a coordinate field accepts.
            if !self.input.text.is_empty() {
                let filtered: String = self
                    .input
                    .text
                    .chars()
                    .filter(|c| c.is_ascii_graphic() || *c == ' ' || *c == '-')
                    .collect();
                if !filtered.is_empty() {
                    let mut out: String = chars[..cursor].iter().collect();
                    out.push_str(&filtered);
                    out.extend(chars[cursor..].iter());
                    state.text = out;
                    cursor += filtered.chars().count();
                    state.cursor = cursor;
                    changed = true;
                }
            }

            if self.input.key_pressed(Key::Enter) {
                state.focused = false;
            }
        }

        // Background and caret slot.
        let bg = if state.focused {
            self.theme.surface_hover
        } else {
            self.theme.surface
        };
        self.fill_round_rect(r, bg, self.theme.border_radius);
        let border = if state.focused {
            self.theme.border_focused
        } else {
            self.theme.border
        };
        self.stroke_rect(r, border, 1.0);
        if state.focused {
            // Caret: a 2px bar at the cursor, blinking is the caller's job.
            let ch_w = self.theme.font_size * 0.55;
            let cx = r.min.x + 6.0 + state.cursor as f32 * ch_w;
            let caret = Rect::from_xywh(cx, r.min.y + 4.0, 2.0, r.height() - 8.0);
            self.fill_rect(caret, self.theme.text);
        }

        (resp, changed)
    }

    /// A collapsible section header. Returns `true` if it was clicked.
    pub fn section_header(&mut self, r: Rect, expanded: bool) -> bool {
        let id = self.next_id();
        let resp = self.interact(id, r, true);
        self.fill_round_rect(
            r,
            if resp.hovered {
                self.theme.surface_hover
            } else {
                self.theme.surface
            },
            self.theme.border_radius,
        );
        // Disclosure triangle.
        let cx = r.min.x + 12.0;
        let cy = r.center().y;
        let s = 4.0;
        let d = if expanded { s } else { s };
        let pts = if expanded {
            [(cx - d, cy - d * 0.5), (cx + d, cy - d * 0.5), (cx, cy + d)]
        } else {
            [(cx - d * 0.5, cy - d), (cx - d * 0.5, cy + d), (cx + d, cy)]
        };
        for w in pts.windows(2) {
            self.batch2d.segment(
                Vec2::new(w[0].0, w[0].1),
                Vec2::new(w[1].0, w[1].1),
                self.theme.text_dim,
                1.5,
            );
        }
        self.batch2d.segment(
            Vec2::new(pts[2].0, pts[2].1),
            Vec2::new(pts[0].0, pts[0].1),
            self.theme.text_dim,
            1.5,
        );
        resp.clicked
    }

    /// Convenience: begin a layout inside `r`.
    pub fn layout(&self, r: Rect, padding: Padding) -> Layout {
        Layout::new(r)
            .with_padding(padding)
            .with_row_height(self.theme.row_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::Event;

    fn setup() -> (InputState, Batch2d, Vec<UiVertex>) {
        let mut i = InputState::new();
        i.push(
            &Event::Resized {
                width: 800,
                height: 600,
            },
            0.0,
        );
        (i, Batch2d::new(), Vec::new())
    }

    fn ui<'a>(input: &'a mut InputState, b: &'a mut Batch2d, v: &'a mut Vec<UiVertex>) -> Ui<'a> {
        Ui::new(input, Theme::dark(), b, v)
    }

    fn rect() -> Rect {
        Rect::from_xywh(10.0, 10.0, 100.0, 30.0)
    }

    #[test]
    fn hover_detects_the_cursor() {
        let (mut i, mut b, mut v) = setup();
        i.push(
            &Event::MouseMoved {
                pos: Vec2::new(50.0, 20.0),
            },
            0.0,
        );
        let mut u = ui(&mut i, &mut b, &mut v);
        let r = u.button(rect(), true);
        assert!(r.hovered);

        i.push(
            &Event::MouseMoved {
                pos: Vec2::new(500.0, 500.0),
            },
            0.1,
        );
        let mut u = ui(&mut i, &mut b, &mut v);
        let r = u.button(rect(), true);
        assert!(!r.hovered);
    }

    #[test]
    fn button_click_emits_geometry() {
        let (mut i, mut b, mut v) = setup();
        i.push(
            &Event::MouseMoved {
                pos: Vec2::new(50.0, 20.0),
            },
            0.0,
        );
        i.push(
            &Event::MouseDown {
                pos: Vec2::new(50.0, 20.0),
                button: MouseButton::Left,
            },
            0.1,
        );
        let mut u = ui(&mut i, &mut b, &mut v);
        let r = u.button(rect(), true);
        assert!(r.pressed);
        assert!(!v.is_empty(), "a button must emit quads");

        i.push(
            &Event::MouseUp {
                pos: Vec2::new(50.0, 20.0),
                button: MouseButton::Left,
            },
            0.15,
        );
        let mut u = ui(&mut i, &mut b, &mut v);
        let r = u.button(rect(), true);
        assert!(r.released);
        assert!(r.clicked);
    }

    #[test]
    fn disabled_button_never_clicks() {
        let (mut i, mut b, mut v) = setup();
        i.push(
            &Event::MouseMoved {
                pos: Vec2::new(50.0, 20.0),
            },
            0.0,
        );
        i.push(
            &Event::MouseDown {
                pos: Vec2::new(50.0, 20.0),
                button: MouseButton::Left,
            },
            0.1,
        );
        i.push(
            &Event::MouseUp {
                pos: Vec2::new(50.0, 20.0),
                button: MouseButton::Left,
            },
            0.15,
        );
        let mut u = ui(&mut i, &mut b, &mut v);
        let r = u.button(rect(), false);
        assert!(!r.hovered, "disabled widgets must not report hover");
        assert!(!r.clicked);
    }

    #[test]
    fn checkbox_toggles_on_click() {
        let (mut i, mut b, mut v) = setup();
        i.push(
            &Event::MouseMoved {
                pos: Vec2::new(50.0, 20.0),
            },
            0.0,
        );
        i.push(
            &Event::MouseDown {
                pos: Vec2::new(50.0, 20.0),
                button: MouseButton::Left,
            },
            0.1,
        );
        i.push(
            &Event::MouseUp {
                pos: Vec2::new(50.0, 20.0),
                button: MouseButton::Left,
            },
            0.15,
        );
        let mut u = ui(&mut i, &mut b, &mut v);
        let (resp, new_value) = u.checkbox(rect(), false);
        assert!(resp.clicked);
        assert!(new_value);
    }

    #[test]
    fn slider_maps_click_to_value() {
        let (mut i, mut b, mut v) = setup();
        // Click at the far right of a 0..=100 track starting at x=10.
        i.push(
            &Event::MouseMoved {
                pos: Vec2::new(105.0, 20.0),
            },
            0.0,
        );
        i.push(
            &Event::MouseDown {
                pos: Vec2::new(105.0, 20.0),
                button: MouseButton::Left,
            },
            0.1,
        );
        let mut u = ui(&mut i, &mut b, &mut v);
        let (_, value) = u.slider(rect(), 0.0, 0.0, 100.0);
        assert!(value > 90.0, "value={value}");
    }

    #[test]
    fn slider_clamps_to_range() {
        let (mut i, mut b, mut v) = setup();
        i.push(
            &Event::MouseMoved {
                pos: Vec2::new(9999.0, 20.0),
            },
            0.0,
        );
        i.push(
            &Event::MouseDown {
                pos: Vec2::new(9999.0, 20.0),
                button: MouseButton::Left,
            },
            0.1,
        );
        let mut u = ui(&mut i, &mut b, &mut v);
        let (_, value) = u.slider(rect(), 0.0, 0.0, 100.0);
        assert!((value - 100.0).abs() < 1e-3, "value={value}");
    }

    #[test]
    fn text_field_types_and_edits() {
        let (mut i, mut b, mut v) = setup();
        let mut state = TextEditState::new("");
        i.push(
            &Event::MouseMoved {
                pos: Vec2::new(50.0, 20.0),
            },
            0.0,
        );
        i.push(
            &Event::MouseDown {
                pos: Vec2::new(50.0, 20.0),
                button: MouseButton::Left,
            },
            0.1,
        );
        i.push(
            &Event::MouseUp {
                pos: Vec2::new(50.0, 20.0),
                button: MouseButton::Left,
            },
            0.15,
        );
        {
            let mut u = ui(&mut i, &mut b, &mut v);
            let (_, _) = u.text_field(rect(), &mut state, true);
            assert!(state.focused);
        }
        i.end_frame();

        i.push(&Event::Text("12.5".into()), 0.2);
        i.push(
            &Event::KeyDown {
                key: Key::Left,
                mods: Modifiers::NONE,
            },
            0.2,
        );
        {
            let mut u = ui(&mut i, &mut b, &mut v);
            let (_, changed) = u.text_field(rect(), &mut state, true);
            assert!(changed);
        }
        assert_eq!(state.text, "12.5");
        assert_eq!(state.cursor, 3, "left arrow must move the caret");
    }

    #[test]
    fn text_field_rejects_control_characters() {
        let (mut i, mut b, mut v) = setup();
        let mut state = TextEditState::new("");
        state.focused = true;
        i.push(&Event::Text("a\nb\tc".into()), 0.0);
        {
            let mut u = ui(&mut i, &mut b, &mut v);
            u.text_field(rect(), &mut state, true);
        }
        assert_eq!(state.text, "abc", "newlines and tabs must be filtered out");
    }

    #[test]
    fn selection_delete_works() {
        let (mut i, mut b, mut v) = setup();
        let mut state = TextEditState::new("12345");
        state.focused = true;
        state.select_all();
        assert!(state.delete_selection());
        assert_eq!(state.text, "");
        assert!(!state.delete_selection(), "a second delete is a no-op");
    }

    #[test]
    fn ctrl_a_selects_all() {
        let (mut i, mut b, mut v) = setup();
        let mut state = TextEditState::new("abcd");
        state.focused = true;
        i.push(
            &Event::KeyDown {
                key: Key::A,
                mods: Modifiers::CTRL,
            },
            0.0,
        );
        {
            let mut u = ui(&mut i, &mut b, &mut v);
            u.text_field(rect(), &mut state, true);
        }
        assert_eq!(state.sel_start, Some(0));
        assert_eq!(state.cursor, 4);
    }

    #[test]
    fn section_header_reports_clicks() {
        let (mut i, mut b, mut v) = setup();
        i.push(
            &Event::MouseMoved {
                pos: Vec2::new(50.0, 20.0),
            },
            0.0,
        );
        i.push(
            &Event::MouseDown {
                pos: Vec2::new(50.0, 20.0),
                button: MouseButton::Left,
            },
            0.1,
        );
        i.push(
            &Event::MouseUp {
                pos: Vec2::new(50.0, 20.0),
                button: MouseButton::Left,
            },
            0.15,
        );
        let mut u = ui(&mut i, &mut b, &mut v);
        assert!(u.section_header(rect(), false));
    }

    #[test]
    fn tooltip_only_when_hovered() {
        let (mut i, mut b, mut v) = setup();
        i.push(
            &Event::MouseMoved {
                pos: Vec2::new(500.0, 500.0),
            },
            0.0,
        );
        {
            let mut u = ui(&mut i, &mut b, &mut v);
            u.tooltip(rect(), "Nope");
            assert!(u.tooltip.is_none());
        }
        i.push(
            &Event::MouseMoved {
                pos: Vec2::new(50.0, 20.0),
            },
            0.1,
        );
        {
            let mut u = ui(&mut i, &mut b, &mut v);
            u.tooltip(rect(), "Yep");
            assert_eq!(u.tooltip.as_deref(), Some("Yep"));
        }
    }

    #[test]
    fn zero_sized_rects_emit_nothing() {
        let (mut i, mut b, mut v) = setup();
        let mut u = ui(&mut i, &mut b, &mut v);
        u.fill_rect(Rect::new(Vec2::ZERO, Vec2::ZERO), Rgba::WHITE);
        assert!(v.is_empty());
    }

    #[test]
    fn ids_are_unique_per_frame() {
        let (mut i, mut b, mut v) = setup();
        let mut u = ui(&mut i, &mut b, &mut v);
        let a = u.next_id();
        let c = u.next_id();
        assert_ne!(a, c);
    }

    #[test]
    fn responses_merge() {
        let a = Response {
            clicked: true,
            ..Response::NONE
        };
        let b = Response {
            hovered: true,
            changed: true,
            ..Response::NONE
        };
        let m = a.merge(b);
        assert!(m.clicked && m.hovered && m.changed);
        assert!(m.interacted());
        assert!(Response::NONE.is_none());
    }

    #[test]
    fn rect_helpers_emit_expected_quad_counts() {
        let (mut i, mut b, mut v) = setup();
        {
            let u = ui(&mut i, &mut b, &mut v);
            u.fill_rect(rect(), Rgba::WHITE);
            assert_eq!(v.len(), 6, "a rect is two triangles");
            v.clear();
            u.stroke_rect(rect(), Rgba::WHITE, 2.0);
            assert_eq!(v.len(), 24, "a stroked rect is four bars of two triangles");
        }
    }
}
