//! `cad-gfx` — the GPU layer.
//!
//! Everything the renderer needs from wgpu lives behind three modules:
//!
//! * [`batch`]   — CPU-side vertex assembly. Entities are tessellated once and
//!                 written into flat, `bytemuck`-castable arrays so a draw call
//!                 is a pointer bump, not a per-entity upload.
//! * [`pipeline`]— the actual wgpu pipelines (2D overlay, 3D solid, 3D line).
//! * [`renderer`]— frame orchestration: surface, depth, MSAA, pass order.
//!
//! Shaders are WGSL, inlined as strings so there is no asset-loading path at
//! runtime and no `include_bytes!` juggling in the build script.

pub mod batch;
pub mod pipeline;
pub mod renderer;

pub use batch::{Batch2d, Batch3d, LineVertex, SolidVertex, UiVertex};
pub use pipeline::{PipelineSet, ShaderError};
pub use renderer::{FrameStats, Gpu, GpuError, RenderTarget, SurfaceConfig};
