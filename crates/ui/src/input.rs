//! Input events, normalized away from the windowing backend.
//!
//! `winit` speaks in `WindowEvent`; the UI and tools speak in this. Keeping the
//! translation in one place means a tool never has to match on winit variants.

use cad_core::Vec2;

/// Physical keys we care about, named after the US layout like AutoCAD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Num0,
    Num1,
    Num2,
    Num3,
    Num4,
    Num5,
    Num6,
    Num7,
    Num8,
    Num9,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Escape,
    Enter,
    Tab,
    Backspace,
    Delete,
    Insert,
    Home,
    End,
    PageUp,
    PageDown,
    Left,
    Right,
    Up,
    Down,
    Minus,
    Equals,
    Comma,
    Period,
    Slash,
    Backslash,
    Semicolon,
    Apostrophe,
    LBracket,
    RBracket,
    Backtick,
    Space,
    Shift,
    Control,
    Alt,
    Super,
}

impl Key {
    /// Lowercase letter for `A`..`Z`, else `None`.
    pub fn as_letter(self) -> Option<char> {
        use Key::*;
        Some(match self {
            A => 'a',
            B => 'b',
            C => 'c',
            D => 'd',
            E => 'e',
            F => 'f',
            G => 'g',
            H => 'h',
            I => 'i',
            J => 'j',
            K => 'k',
            L => 'l',
            M => 'm',
            N => 'n',
            O => 'o',
            P => 'p',
            Q => 'q',
            R => 'r',
            S => 's',
            T => 't',
            U => 'u',
            V => 'v',
            W => 'w',
            X => 'x',
            Y => 'y',
            Z => 'z',
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    /// Scroll wheel pressed.
    Other(u8),
}

/// Modifier state at the time of an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub super_key: bool,
}

impl Modifiers {
    pub const NONE: Self = Self {
        shift: false,
        ctrl: false,
        alt: false,
        super_key: false,
    };
    pub const SHIFT: Self = Self {
        shift: true,
        ..Self::NONE
    };
    pub const CTRL: Self = Self {
        ctrl: true,
        ..Self::NONE
    };

    pub fn any(&self) -> bool {
        self.shift || self.ctrl || self.alt || self.super_key
    }

    /// True when Ctrl (or Cmd on macOS) is held — the CAD convention for
    /// "temporary" overrides.
    pub fn command(&self) -> bool {
        self.ctrl || self.super_key
    }
}

/// Scroll wheel delta in lines (positive = away from the user).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScrollDelta {
    pub x: f32,
    pub y: f32,
}

/// Normalized UI input for one frame.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Pointer moved. `pos` is in physical pixels with Y down.
    MouseMoved {
        pos: Vec2,
    },
    MouseDragged {
        pos: Vec2,
        button: MouseButton,
    },
    MouseWheel {
        delta: ScrollDelta,
    },
    MouseDown {
        pos: Vec2,
        button: MouseButton,
    },
    MouseUp {
        pos: Vec2,
        button: MouseButton,
    },
    /// A click that started and ended within `click_radius` and `click_time`.
    Click {
        pos: Vec2,
        button: MouseButton,
    },
    MouseLeave,
    KeyDown {
        key: Key,
        mods: Modifiers,
    },
    KeyUp {
        key: Key,
        mods: Modifiers,
    },
    /// Text input (already composed, respects keyboard layout).
    Text(String),
    /// DPI / framebuffer resize.
    Resized {
        width: u32,
        height: u32,
    },
    /// Window gained focus (AutoCAD stops cancelling commands on refocus).
    Focused(bool),
    /// Close requested.
    CloseRequested,
}

/// Accumulates raw events for a frame and provides derived queries
/// (double-click, drag threshold) so widgets do not re-implement them.
#[derive(Debug, Default)]
pub struct InputState {
    pub mouse: Vec2,
    pub mouse_delta: Vec2,
    pub scroll: ScrollDelta,
    pub pressed: Vec<MouseButton>,
    pub released: Vec<MouseButton>,
    pub keys_down: Vec<Key>,
    pub keys_pressed: Vec<Key>,
    pub text: String,
    pub modifiers: Modifiers,
    pub focused: bool,
    /// Rect of the viewport in pixels, so clicks outside it can be rejected.
    pub viewport: cad_core::Rect2,
    pub click_slop: f32,
    pub now: f64,
    last_down_pos: Vec2,
    last_down_time: f64,
    drag_distance: f32,
    /// Set by `MouseUp`, consumed by [`InputState::take_click`].
    pending_click: Option<(Vec2, MouseButton)>,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            click_slop: 4.0,
            ..Default::default()
        }
    }

    /// Fold one event in. Returns `true` when the event was consumed by the
    /// focus/hover bookkeeping (the caller still decides about hotkeys).
    pub fn push(&mut self, e: &Event, now: f64) {
        self.now = now;
        match e {
            Event::MouseMoved { pos } => {
                self.mouse_delta = *pos - self.mouse;
                self.mouse = *pos;
            }
            Event::MouseDown { pos, button } => {
                self.mouse = *pos;
                self.pressed.push(*button);
                self.last_down_pos = *pos;
                self.last_down_time = now;
                self.drag_distance = 0.0;
            }
            Event::MouseDragged { pos, button } => {
                self.mouse_delta = *pos - self.mouse;
                self.mouse = *pos;
                if !self.pressed.contains(button) {
                    self.pressed.push(*button);
                }
                self.drag_distance = self.drag_distance.max((*pos - self.last_down_pos).length());
            }
            Event::MouseUp { pos, button } => {
                self.mouse = *pos;
                self.released.push(*button);
                // A release close to where the press happened, within
                // `click_time`, is a click.
                if self.last_down_time > 0.0
                    && (now - self.last_down_time) < 0.5
                    && pos.distance(self.last_down_pos) <= self.click_slop
                    && self.drag_distance <= self.click_slop
                {
                    self.pending_click = Some((*pos, *button));
                }
            }
            Event::MouseWheel { delta } => {
                self.scroll = ScrollDelta {
                    x: self.scroll.x + delta.x,
                    y: self.scroll.y + delta.y,
                };
            }
            Event::KeyDown { key, mods } => {
                self.modifiers = *mods;
                if !self.keys_down.contains(key) {
                    self.keys_pressed.push(*key);
                }
                self.keys_down.push(*key);
            }
            Event::KeyUp { key, mods } => {
                self.modifiers = *mods;
                self.keys_down.retain(|k| k != key);
            }
            Event::Text(s) => self.text.push_str(s),
            Event::Resized { width, height } => {
                self.viewport = cad_core::Rect2::from_xywh(0.0, 0.0, *width as f32, *height as f32);
            }
            Event::Focused(f) => {
                self.focused = *f;
                if !*f {
                    // Losing focus must not leave buttons stuck down.
                    self.pressed.clear();
                    self.keys_down.clear();
                }
            }
            Event::CloseRequested => {}
            Event::MouseLeave => {
                self.pressed.clear();
            }
            Event::Click { .. } => {}
        }
    }

    /// Take the click detected by the last `MouseUp`, if any.
    pub fn take_click(&mut self) -> Option<(Vec2, MouseButton)> {
        self.pending_click.take()
    }

    pub fn is_down(&self, b: MouseButton) -> bool {
        self.pressed.contains(&b)
    }
    pub fn just_pressed(&self, b: MouseButton) -> bool {
        self.pressed.contains(&b) && self.released.is_empty()
    }
    pub fn key_down(&self, k: Key) -> bool {
        self.keys_down.contains(&k)
    }
    pub fn key_pressed(&self, k: Key) -> bool {
        self.keys_pressed.contains(&k)
    }
    /// Is the cursor over `r`?
    pub fn hovered(&self, r: cad_core::Rect2) -> bool {
        r.contains(self.mouse)
    }
    /// Was this frame a drag with `b` held past the slop threshold?
    pub fn dragging(&self, b: MouseButton) -> bool {
        self.is_down(b) && self.drag_distance > self.click_slop
    }

    /// Call at the end of each frame.
    pub fn end_frame(&mut self) {
        self.released.clear();
        self.keys_pressed.clear();
        self.text.clear();
        self.scroll = ScrollDelta::default();
        self.mouse_delta = Vec2::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mods(shift: bool, ctrl: bool) -> Modifiers {
        Modifiers {
            shift,
            ctrl,
            ..Modifiers::NONE
        }
    }

    #[test]
    fn modifiers_commands() {
        assert!(!Modifiers::NONE.any());
        assert!(mods(true, false).any());
        assert!(mods(false, true).command());
        assert!(!mods(true, false).command());
        let m = Modifiers {
            super_key: true,
            ..Modifiers::NONE
        };
        assert!(m.command(), "cmd counts as command on macOS");
    }

    #[test]
    fn key_letters() {
        assert_eq!(Key::L.as_letter(), Some('l'));
        assert_eq!(Key::Z.as_letter(), Some('z'));
        assert_eq!(Key::F1.as_letter(), None);
        assert_eq!(Key::Space.as_letter(), None);
    }

    #[test]
    fn hover_and_click() {
        let mut i = InputState::new();
        i.push(
            &Event::MouseMoved {
                pos: Vec2::new(10.0, 10.0),
            },
            0.0,
        );
        assert!(i.hovered(cad_core::Rect2::from_xywh(0.0, 0.0, 20.0, 20.0)));
        assert!(!i.hovered(cad_core::Rect2::from_xywh(50.0, 50.0, 20.0, 20.0)));

        i.push(
            &Event::MouseDown {
                pos: Vec2::new(10.0, 10.0),
                button: MouseButton::Left,
            },
            0.0,
        );
        i.push(
            &Event::MouseUp {
                pos: Vec2::new(11.0, 10.0),
                button: MouseButton::Left,
            },
            0.1,
        );
        let (pos, btn) = i.take_click().unwrap();
        assert_eq!(btn, MouseButton::Left);
        assert!(pos.distance(Vec2::new(11.0, 10.0)) < 1e-6);
        assert!(i.take_click().is_none(), "click must be consumable once");
    }

    #[test]
    fn a_drag_is_not_a_click() {
        let mut i = InputState::new();
        i.push(
            &Event::MouseDown {
                pos: Vec2::new(10.0, 10.0),
                button: MouseButton::Left,
            },
            0.0,
        );
        for k in 1..6 {
            i.push(
                &Event::MouseDragged {
                    pos: Vec2::new(10.0 + k as f32 * 20.0, 10.0),
                    button: MouseButton::Left,
                },
                k as f64 * 0.05,
            );
        }
        assert!(i.dragging(MouseButton::Left));
        i.push(
            &Event::MouseUp {
                pos: Vec2::new(200.0, 10.0),
                button: MouseButton::Left,
            },
            0.4,
        );
        assert!(i.take_click().is_none(), "a 200px drag must not click");
    }

    #[test]
    fn slow_release_is_not_a_click() {
        let mut i = InputState::new();
        i.push(
            &Event::MouseDown {
                pos: Vec2::new(10.0, 10.0),
                button: MouseButton::Left,
            },
            0.0,
        );
        // Held for two seconds, released in place.
        i.push(
            &Event::MouseUp {
                pos: Vec2::new(10.0, 10.0),
                button: MouseButton::Left,
            },
            2.0,
        );
        assert!(i.take_click().is_none(), "a long hold is not a click");
    }

    #[test]
    fn keys_edge_trigger() {
        let mut i = InputState::new();
        i.push(
            &Event::KeyDown {
                key: Key::D,
                mods: Modifiers::NONE,
            },
            0.0,
        );
        assert!(i.key_down(Key::D));
        assert!(i.key_pressed(Key::D));
        i.end_frame();
        assert!(i.key_down(Key::D));
        assert!(!i.key_pressed(Key::D), "key_pressed must be edge-triggered");

        i.push(
            &Event::KeyDown {
                key: Key::D,
                mods: Modifiers::NONE,
            },
            0.1,
        );
        assert!(!i.key_pressed(Key::D), "auto-repeat must not re-trigger");

        i.push(
            &Event::KeyUp {
                key: Key::D,
                mods: Modifiers::NONE,
            },
            0.2,
        );
        assert!(!i.key_down(Key::D));
    }

    #[test]
    fn modifiers_are_tracked() {
        let mut i = InputState::new();
        i.push(
            &Event::KeyDown {
                key: Key::Z,
                mods: mods(true, false),
            },
            0.0,
        );
        assert!(i.modifiers.shift);
        i.push(
            &Event::KeyDown {
                key: Key::Y,
                mods: mods(true, true),
            },
            0.1,
        );
        assert!(i.modifiers.shift && i.modifiers.ctrl);
    }

    #[test]
    fn losing_focus_clears_stuck_buttons() {
        let mut i = InputState::new();
        i.push(
            &Event::MouseDown {
                pos: Vec2::ZERO,
                button: MouseButton::Left,
            },
            0.0,
        );
        i.push(
            &Event::KeyDown {
                key: Key::D,
                mods: Modifiers::NONE,
            },
            0.0,
        );
        assert!(i.is_down(MouseButton::Left));
        i.push(&Event::Focused(false), 0.1);
        assert!(!i.is_down(MouseButton::Left), "stuck button after alt-tab");
        assert!(!i.key_down(Key::D));
    }

    #[test]
    fn scroll_accumulates_and_resets() {
        let mut i = InputState::new();
        i.push(
            &Event::MouseWheel {
                delta: ScrollDelta { x: 0.0, y: 1.0 },
            },
            0.0,
        );
        i.push(
            &Event::MouseWheel {
                delta: ScrollDelta { x: 0.0, y: 2.0 },
            },
            0.0,
        );
        assert!((i.scroll.y - 3.0).abs() < 1e-6);
        i.end_frame();
        assert_eq!(i.scroll, ScrollDelta::default());
    }

    #[test]
    fn text_input_accumulates() {
        let mut i = InputState::new();
        i.push(&Event::Text("12.5".into()), 0.0);
        i.push(&Event::Text(",".into()), 0.0);
        assert_eq!(i.text, "12.5,");
        i.end_frame();
        assert!(i.text.is_empty());
    }

    #[test]
    fn mouse_delta_tracks_motion() {
        let mut i = InputState::new();
        i.push(
            &Event::MouseMoved {
                pos: Vec2::new(5.0, 5.0),
            },
            0.0,
        );
        i.end_frame();
        i.push(
            &Event::MouseMoved {
                pos: Vec2::new(8.0, 7.0),
            },
            0.1,
        );
        assert!((i.mouse_delta - Vec2::new(3.0, 2.0)).length() < 1e-6);
    }

    #[test]
    fn resize_updates_the_viewport() {
        let mut i = InputState::new();
        i.push(
            &Event::Resized {
                width: 1920,
                height: 1080,
            },
            0.0,
        );
        assert_eq!(i.viewport.size(), Vec2::new(1920.0, 1080.0));
        assert!(i.hovered(cad_core::Rect2::from_xywh(1900.0, 1000.0, 20.0, 80.0)));
    }
}
