//! `cad-ui` — an immediate-mode widget layer drawn on the `cad-gfx` quads.
//!
//! No retained widget tree, no layout engine dependency: every frame the UI
//! declares what it wants, the batch is filled, and input is returned. That is
//! ~2k lines instead of a 2 MB framework, which is the whole point of this
//! project.

pub mod font;
pub mod input;
pub mod layout;
pub mod theme;
pub mod widgets;

pub use font::measure as text_width;
pub use input::{Event, Key, Modifiers, MouseButton};
pub use layout::{Align, Layout, Padding, Rect, Size};
pub use theme::Theme;
pub use widgets::Ui;
