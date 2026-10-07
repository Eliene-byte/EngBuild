//! `cad-geom` — exact 2D curve primitives, adaptive tessellation and
//! analytic intersections.
//!
//! Everything is float32 for GPU friendliness, but the *algorithms* are the
//! analytic ones (quadratics solved in closed form, not sampled), because a CAD
//! kernel that "almost" intersects is a CAD kernel you cannot trust.

pub mod bulge;
pub mod curve;
pub mod intersect;
pub mod spline;
pub mod tessellate;
pub mod xform;

pub use bulge::Bulge;
pub use curve::{Arc, Circle, Curve, Ellipse, Line, Polyline, Shape};
pub use intersect::{Hit, HitKind};
pub use spline::{Bezier, Spline};
pub use tessellate::TessellationOptions;
