//! `cad-gfx` is the GPU layer.
//!
//! Everything the renderer needs from wgpu lives behind three modules:
//!
//! - [`batch`]: CPU-side vertex assembly. Entities are tessellated once and
//!   written into flat, `bytemuck`-castable arrays, so a draw call is a pointer
//!   bump rather than a per-entity upload.
//! - [`pipeline`]: the actual wgpu pipelines (2D overlay, UI, 3D solid, 3D line,
//!   3D grid).
//! - [`renderer`]: frame orchestration: surface, depth, MSAA, pass order.
//!
//! Shaders are WGSL, inlined as strings so there is no asset-loading path at
//! runtime and no `include_bytes!` juggling in the build script.

//! Shaders are WGSL, inlined as strings so there is no asset-loading path at
//! runtime and no `include_bytes!` juggling in the build script.

pub mod batch;
pub mod pipeline;
pub mod renderer;

pub use batch::{
    Batch2d, Batch3d, LineVertex, LineVertex3d, SolidVertex, UiVertex, push_rect,
    push_rounded_rect, stroke_rect,
};
pub use pipeline::{Globals, PipelineSet, ShaderError};
pub use renderer::{FrameError, FrameStats, Gpu, GpuError, RenderTarget, SurfaceConfig};
