//! `cad-app` — the application shell: document, view state, tools and commands.
//!
//! This crate owns *what* the user is doing. It does not own pixels (that is
//! `cad-gfx`) or widgets (that is `cad-ui`); it produces the geometry those
//! crates draw and consumes normalized input.

pub mod command;
pub mod session;
pub mod tools;

pub use command::{Command, CommandRegistry, CommandResult};
pub use session::{Session, StatusMessage, ViewMode};
pub use tools::{Tool, ToolId, ToolState};
