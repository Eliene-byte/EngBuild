//! `cad-core` — the numeric foundation of the whole CAD engine.
//!
//! Deliberately **zero external dependencies**: every type here is `repr(C)`,
//! `Pod`-compatible and written for `memcpy`-speed bulk transfer to the GPU.
//! This is what keeps the final binary small and the hot loops fast.

pub mod bounds;
pub mod camera;
pub mod color;
pub mod float;
pub mod mat;
pub mod plane;
pub mod quat;
pub mod vec;

pub use bounds::{Aabb3, Rect2};
pub use camera::{Camera2D, Camera3D};
pub use color::{ACI_PALETTE, Rgba, Srgba, aci_to_rgba};
pub use float::*;
pub use mat::{Mat3, Mat4};
pub use plane::Plane3;
pub use quat::Quat;
pub use vec::{Vec2, Vec3, Vec4};
