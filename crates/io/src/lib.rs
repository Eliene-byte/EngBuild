//! `cad-io` — DXF import/export plus a compact native project format.
//!
//! The DXF reader is a real group-code tokenizer (DXF is just pairs of lines:
//! a code line and a value line), not a regex hack, because group codes carry
//! the *type* of the value and getting that wrong is how broken importers are
//! born.

pub mod dxf;
pub mod native;

pub use dxf::{DxfError, DxfReadOptions, import, read, write};
pub use native::{NativeError, load, save};
