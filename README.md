# CADKit

A native 2D/3D CAD application written in Rust with a [wgpu] renderer.

No Electron, no web view, no bundled browser. The window comes from `winit`,
the GPU work goes through `wgpu`, and everything in between is our own code —
which is why the binary stays in the tens of megabytes instead of the hundreds.

```
crates/core    math, bounds, cameras, colour          zero dependencies
crates/geom    curves, adaptive tessellation, intersections
crates/doc     document model, layers, blocks, undo, spatial index
crates/io      DXF import/export, native binary format
crates/snap    object snaps and polar/ortho tracking
crates/gfx     wgpu pipelines, batches, WGSL shaders
crates/ui      immediate-mode widgets drawn on the GPU
crates/app     tools, commands, session state
src            the window loop and frame assembly
```

## Building

Everything is built by GitHub Actions (`.github/workflows/build.yml`):

- **CI** on every push and pull request — format check, clippy with
  `-D warnings`, the full test suite and documentation.
- **Artifacts** for Windows x64 and arm64, Linux x64, and a macOS universal
  binary (Apple Silicon + Intel in one file).
- **Releases** are cut automatically when a `v*` tag is pushed.

Download a build from the [Actions page][actions] or from the
[releases page][releases].

To build locally you only need a stable toolchain and the usual platform
dependencies. On Windows, install the Visual Studio Build Tools with the
*Desktop development with C++* workload; on Linux, `libwayland-dev`,
`libxkbcommon-dev` and `pkg-config`.

```sh
cargo run --release
```

## What it does today

**Drawing.** Line, circle, arc (three-point), polyline and rectangle, with
chains that continue from the last point the way CAD users expect. Object snaps
resolve in screen pixels, so endpoint markers feel identical at every zoom level.

**Viewing.** Pan with the middle button, zoom with the wheel, `Z` for extents.
A grid and axes are drawn under the geometry, and the grid fades out instead of
turning into moire when you zoom out.

**Editing.** Selection by click or window, undo/redo with `Ctrl+Z` / `Ctrl+Y`,
a command line that takes the same names as the classic commands (`line`,
`circle`, `arc`, `erase`, `zoomall`, `view3d`, ...).

**Interoperability.** DXF in and out — LINE, CIRCLE, ARC, ELLIPSE, LWPOLYLINE
(including bulges), SPLINE, TEXT, POINT, INSERT, layers with visibility, colour,
linetype and lineweight. Projects also save to a compact native format that
round-trips everything the document model holds.

## Design notes

A few decisions that shaped the code:

**The geometry kernel is analytic.** Lines and circles intersect by solving the
quadratic in closed form, not by sampling and hoping. Arcs preserve their sweep
through an affine transform and are only promoted to polylines when the transform
would actually distort them. Adaptive flattening is driven by chord sagitta, so a
1000 x 10 ellipse comes out smooth without bloating the vertex buffer.

**Undo records images, not inverses.** Every mutation runs inside a transaction
that snapshots the *before* image of each touched entity, then derives the after
image by diffing on commit. It is correct by construction, and the cost is
proportional to what actually changed.

**Bullets, not tree walks.** A frame is three `write_buffer` calls and four draw
calls. Entity geometry is tessellated into flat `bytemuck`-castable arrays, and a
uniform grid keeps picking proportional to what is visible.

## Testing

```sh
cargo test --workspace
```

The suite covers the parts where being subtly wrong is worse than not shipping:
matrix inversion, arc sweeps, affine transforms of curves, tessellation
tolerance, DXF round-trips against hand-written fixtures, snap priority
arbitration, and undo/redo over 20 consecutive operations.

[wgpu]: https://wgpu.rs
[actions]: https://github.com/Eliene-byte/EngBuild/actions
[releases]: https://github.com/Eliene-byte/EngBuild/releases
