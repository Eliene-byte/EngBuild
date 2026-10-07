//! Frame orchestration: surface, depth/MSAA targets, pass order, and the
//! buffers that back the pipelines.

use crate::batch::{Batch2d, Batch3d, LineVertex, SolidVertex, UiVertex};
use crate::pipeline::{Globals, LineQuadVertex, PipelineSet, expand_lines};
use std::fmt;

/// Everything that can go wrong during initialisation.
#[derive(Debug)]
pub enum GpuError {
    NoAdapter,
    RequestDevice(wgpu::RequestDeviceError),
    CreateSurface(String),
    NoFormat,
    Buffer(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "no suitable GPU adapter was found"),
            GpuError::RequestDevice(e) => write!(f, "could not create the GPU device: {e}"),
            GpuError::CreateSurface(m) => write!(f, "could not create the surface: {m}"),
            GpuError::NoFormat => write!(f, "the surface exposes no preferred texture format"),
            GpuError::Buffer(m) => write!(f, "buffer allocation failed: {m}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Surface configuration, kept separate from the device so it can be recreated
/// without leaking the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceConfig {
    pub width: u32,
    pub height: u32,
    /// 0 disables MSAA.
    pub msaa_samples: u32,
}

impl Default for SurfaceConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            msaa_samples: 4,
        }
    }
}

/// The 3D view the renderer is drawing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderTarget {
    pub view: [[f32; 4]; 4],
    pub projection: [[f32; 4]; 4],
    pub eye: [f32; 3],
    /// Screen position of the view centre, used by the 3D line shader for its
    /// rim light. Keeping it here avoids a second uniform buffer.
    pub eye_screen: [f32; 2],
}

impl Default for RenderTarget {
    fn default() -> Self {
        Self {
            view: Mat_identity(),
            projection: Mat_identity(),
            eye: [0.0, 0.0, 10.0],
            eye_screen: [0.0, 0.0],
        }
    }
}

const fn Mat_identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

impl RenderTarget {
    /// Build from `cad_core` cameras.
    pub fn from_camera(cam: &cad_core::Camera3D, aspect: f32) -> Self {
        let view = cam.view();
        let proj = cam.projection(aspect);
        Self {
            view: view.to_cols_array_2d(),
            projection: proj.to_cols_array_2d(),
            eye: cam.eye.to_array(),
            eye_screen: [0.0, 0.0],
        }
    }

    fn view_proj(&self) -> [[f32; 4]; 4] {
        let vp = mul4(self.projection, self.view);
        vp
    }
}

/// Column-major 4x4 multiply, matching `Mat4`'s memory order.
fn mul4(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += b[k][row] * a[col][k];
            }
            out[col][row] = sum;
        }
    }
    out
}

/// A growable vertex buffer that reallocates only when it must.
struct VertexBuffer {
    buffer: wgpu::Buffer,
    capacity: u64,
    label: &'static str,
}

impl VertexBuffer {
    fn new(device: &wgpu::Device, capacity: u64, label: &'static str) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity.max(16),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            capacity,
            label,
        }
    }
    /// Upload `data`, reallocating if needed. Returns the draw vertex count.
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> u64 {
        let need = data.len().max(16) as u64;
        if need > self.capacity {
            // Over-allocate so a growing frame does not realloc every frame.
            let next = need.next_power_of_two().max(1024);
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: next,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.capacity = next;
        }
        if !data.is_empty() {
            queue.write_buffer(&self.buffer, 0, data);
        }
        data.len() as u64
    }
}

/// Per-frame counters for the stats overlay.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameStats {
    pub frame: u64,
    pub draw_calls: u32,
    pub line_vertices: u64,
    pub solid_vertices: u64,
    pub ui_vertices: u64,
    pub bytes_uploaded: u64,
    pub msaa_samples: u32,
}

/// The renderer.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub config: SurfaceConfig,
    pub surface: wgpu::Surface<'static>,
    pub format: wgpu::TextureFormat,
    pub pipelines: PipelineSet,
    color_msaa: Option<wgpu::TextureView>,
    depth: Option<wgpu::TextureView>,
    line_buf: VertexBuffer,
    solid_buf: VertexBuffer,
    ui_buf: VertexBuffer,
    line3d_buf: VertexBuffer,
    grid_buf: VertexBuffer,
    scratch_lines: Vec<LineQuadVertex>,
    globals: Globals,
    pub stats: FrameStats,
    /// Set when the last frame found no adapter surface; the loop should idle.
    pub occluded: bool,
}

impl Gpu {
    /// Create a renderer for `window`.
    pub fn new(
        window: std::sync::Arc<wgpu::Window>,
        config: SurfaceConfig,
    ) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Window(Box::new(window)))
            .map_err(|e| GpuError::CreateSurface(e.to_string()))?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("cadkit-device"),
            required_features: wgpu::Features::empty(),
            // No features: we only need core WebGPU, which keeps the binary small
            // and runs on every backend.
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .map_err(GpuError::RequestDevice)?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or_else(|| caps.formats[0]);

        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: config.width.max(1),
                height: config.height.max(1),
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        let pipelines = PipelineSet::new(&device, format, config.msaa_samples);

        let mut gpu = Self {
            line_buf: VertexBuffer::new(&device, 0, "lines"),
            solid_buf: VertexBuffer::new(&device, 0, "solids"),
            ui_buf: VertexBuffer::new(&device, 0, "ui"),
            line3d_buf: VertexBuffer::new(&device, 0, "lines3d"),
            grid_buf: VertexBuffer::new(&device, 0, "grid"),
            device,
            queue,
            adapter_info,
            config,
            surface,
            format,
            pipelines,
            color_msaa: None,
            depth: None,
            scratch_lines: Vec::new(),
            globals: Globals::default(),
            stats: FrameStats::default(),
            occluded: false,
        };
        gpu.resize(config.width, config.height);
        Ok(gpu)
    }

    /// Recreate size-dependent targets.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.format,
                width: self.config.width,
                height: self.config.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );
        self.color_msaa = None;
        self.depth = None;
        self.ensure_targets();
    }

    fn ensure_targets(&mut self) {
        let (w, h) = (self.config.width, self.config.height);
        if self.depth.is_none() {
            let d = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("depth"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth = Some(d.create_view(&wgpu::TextureViewDescriptor::default()));
        }
        if self.pipelines.msaa_samples > 1 && self.color_msaa.is_none() {
            let c = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("color-msaa"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: self.pipelines.msaa_samples,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.color_msaa = Some(c.create_view(&wgpu::TextureViewDescriptor::default()));
        }
    }

    /// How many samples actually get rendered (may be 1 on low-end GPUs).
    pub fn effective_msaa(&self) -> u32 {
        if self.pipelines.msaa_samples > 1 && self.color_msaa.is_some() {
            self.pipelines.msaa_samples
        } else {
            1
        }
    }

    /// Update the per-frame uniform block.
    pub fn set_globals(
        &mut self,
        view_proj: [[f32; 4]; 4],
        resolution: [f32; 2],
        zoom: f32,
        dpr: f32,
        time: f32,
        eye_screen: [f32; 2],
    ) {
        self.globals = Globals {
            view_proj,
            resolution,
            zoom,
            dpr,
            time,
            frame: self.stats.frame as f32,
            pad: eye_screen,
        };
        self.pipelines.update_globals(&self.queue, &self.globals);
    }

    /// Render one frame.
    ///
    /// Pass order (2D viewport on top of the 3D model view):
    /// 1. clear
    /// 2. grid (depth-tested, no write)
    /// 3. 3D solids (depth write)
    /// 4. 3D lines (depth-tested, no write)
    /// 5. 2D lines (screen space, no depth)
    /// 6. UI quads (screen space, no depth)
    pub fn render(
        &mut self,
        lines: &Batch2d,
        ui: &[UiVertex],
        solids: &Batch3d,
    ) -> Result<(), wgpu::SurfaceError> {
        self.ensure_targets();
        self.stats.draw_calls = 0;
        self.stats.bytes_uploaded = 0;

        let frame = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                // The window was resized or the surface was lost: reconfigure
                // and skip this frame instead of crashing.
                self.resize(self.config.width, self.config.height);
                self.occluded = true;
                return Ok(());
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                self.occluded = true;
                return Err(wgpu::SurfaceError::OutOfMemory);
            }
            Err(e) => return Err(e),
        };
        self.occluded = false;

        // --- upload ---
        let line_bytes = if lines.is_empty() {
            0
        } else {
            expand_lines(&lines.lines, &mut self.scratch_lines);
            let b = bytemuck::cast_slice(&self.scratch_lines);
            let n = self.line_buf.upload(&self.device, &self.queue, b);
            self.stats.line_vertices = n / std::mem::size_of::<LineQuadVertex>() as u64;
            self.stats.bytes_uploaded += n;
            n
        };

        let ui_bytes = bytemuck::cast_slice(ui);
        let ui_count = if ui.is_empty() {
            0
        } else {
            let n = self.ui_buf.upload(&self.device, &self.queue, ui_bytes);
            self.stats.ui_vertices = n / std::mem::size_of::<UiVertex>() as u64;
            self.stats.bytes_uploaded += n;
            n
        };

        let solid_count = if solids.solids.is_empty() {
            0
        } else {
            let b = bytemuck::cast_slice(&solids.solids);
            let n = self.solid_buf.upload(&self.device, &self.queue, b);
            self.stats.solid_vertices = n / std::mem::size_of::<SolidVertex>() as u64;
            self.stats.bytes_uploaded += n;
            n
        };

        let line3d_count = if solids.lines.is_empty() {
            0
        } else {
            let b = bytemuck::cast_slice(&solids.lines);
            let n = self.line3d_buf.upload(&self.device, &self.queue, b);
            self.stats.bytes_uploaded += n;
            n
        };

        // --- record ---
        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let msaa = self.effective_msaa();
        let color_attachment = if msaa > 1 {
            self.color_msaa.as_ref().unwrap()
        } else {
            &surface_view
        };

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        {
            let color_attachment = color_attachment.clone();
            let depth_attachment = self
                .depth
                .as_ref()
                .map(|d| wgpu::RenderPassColorAttachment {
                    view: d,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                });
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_attachment,
                    resolve_target: if msaa > 1 { Some(&surface_view) } else { None },
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.078,
                            g: 0.086,
                            b: 0.102,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: depth_attachment,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            pass.set_bind_group(0, &self.pipelines.globals_bind_group, &[]);

            if solid_count > 0 {
                pass.set_pipeline(&self.pipelines.solid);
                pass.set_vertex_buffer(0, self.solid_buf.buffer.slice(..));
                pass.draw(0..solid_count, 0..1);
                self.stats.draw_calls += 1;
            }
            if line3d_count > 0 {
                pass.set_pipeline(&self.pipelines.line3d);
                pass.set_vertex_buffer(0, self.line3d_buf.buffer.slice(..));
                pass.draw(0..line3d_count, 0..1);
                self.stats.draw_calls += 1;
            }
            if line_bytes > 0 {
                pass.set_pipeline(&self.pipelines.line_quad);
                pass.set_vertex_buffer(0, self.line_buf.buffer.slice(..));
                pass.draw(0..line_bytes, 0..1);
                self.stats.draw_calls += 1;
            }
            if ui_count > 0 {
                pass.set_pipeline(&self.pipelines.ui);
                pass.set_vertex_buffer(0, self.ui_buf.buffer.slice(..));
                pass.draw(0..ui_count, 0..1);
                self.stats.draw_calls += 1;
            }
        }

        self.queue.submit(Some(enc.finish()));
        frame.present();
        self.stats.frame += 1;
        self.stats.msaa_samples = msaa;
        Ok(())
    }

    /// Bytes currently resident in the vertex buffers (for the debug overlay).
    pub fn vram_bytes(&self) -> u64 {
        self.line_buf.capacity
            + self.solid_buf.capacity
            + self.ui_buf.capacity
            + self.line3d_buf.capacity
            + self.grid_buf.capacity
    }

    /// Project a world point into screen pixels with the current globals.
    pub fn project(&self, p: cad_core::Vec3) -> Option<cad_core::Vec2> {
        let vp = self.globals.view_proj;
        let clip = [
            vp[0][0] * p.x + vp[1][0] * p.y + vp[2][0] * p.z + vp[3][0],
            vp[0][1] * p.x + vp[1][1] * p.y + vp[2][1] * p.z + vp[3][1],
            vp[0][2] * p.x + vp[1][2] * p.y + vp[2][2] * p.z + vp[3][2],
            vp[0][3] * p.x + vp[1][3] * p.y + vp[2][3] * p.z + vp[3][3],
        ];
        if clip[3].abs() < 1e-9 {
            return None;
        }
        let ndc_x = clip[0] / clip[3];
        let ndc_y = clip[1] / clip[3];
        let (rx, ry) = (self.globals.resolution[0], self.globals.resolution[1]);
        Some(cad_core::Vec2::new(
            (ndc_x * 0.5 + 0.5) * rx,
            (0.5 - ndc_y * 0.5) * ry,
        ))
    }

    /// Unproject a screen pixel onto the world plane `z = plane_z`.
    pub fn unproject_plane(&self, screen: cad_core::Vec2, plane_z: f32) -> Option<cad_core::Vec3> {
        let (rx, ry) = (self.globals.resolution[0], self.globals.resolution[1]);
        let ndc_x = (screen.x / rx) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen.y / ry) * 2.0;
        // Invert the view-projection matrix.
        let inv = invert4(self.globals.view_proj);
        let un = |z: f32| -> Option<cad_core::Vec3> {
            let x = inv[0][0] * ndc_x + inv[1][0] * ndc_y + inv[2][0] * z + inv[3][0];
            let y = inv[0][1] * ndc_x + inv[1][1] * ndc_y + inv[2][1] * z + inv[3][1];
            let w = inv[0][3] * ndc_x + inv[1][3] * ndc_y + inv[2][3] * z + inv[3][3];
            let zz = inv[0][2] * ndc_x + inv[1][2] * ndc_y + inv[2][2] * z + inv[3][2];
            if w.abs() < 1e-9 {
                return None;
            }
            Some(cad_core::Vec3::new(x / w, y / w, zz / w))
        };
        let near = un(-1.0)?;
        let far = un(1.0)?;
        let d = far - near;
        if d.z.abs() < 1e-9 {
            return None;
        }
        let t = (plane_z - near.z) / d.z;
        Some(near + d * t)
    }
}

/// Gauss-Jordan inverse of a column-major 4x4. Returns the input unchanged if
/// it is singular, which keeps picking usable even under a degenerate camera.
fn invert4(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // Work in row-major for the solve, then transpose back.
    let mut a = [[0.0f32; 8]; 4];
    for r in 0..4 {
        for c in 0..4 {
            a[r][c] = m[c][r];
        }
        a[r][4 + r] = 1.0;
    }
    for col in 0..4 {
        let mut piv = col;
        for r in (col + 1)..4 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return m;
        }
        a.swap(col, piv);
        let d = a[col][col];
        for c in 0..8 {
            a[col][c] /= d;
        }
        for r in 0..4 {
            if r == col {
                continue;
            }
            let f = a[r][col];
            if f == 0.0 {
                continue;
            }
            for c in 0..8 {
                a[r][c] -= f * a[col][c];
            }
        }
    }
    let mut out = [[0.0f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[c][r] = a[r][4 + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::{Camera3D, Vec2, Vec3};

    #[test]
    fn mat4_multiply_is_column_major() {
        let id = Mat_identity();
        assert_eq!(mul4(id, id), id);
        let scale = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let r = mul4(scale, id);
        assert_eq!(r[0][0], 2.0);
        assert_eq!(r[1][1], 3.0);
        assert_eq!(r[2][2], 4.0);
    }

    #[test]
    fn mat4_inverse_round_trips() {
        let m = [
            [2.0, 0.0, 0.0, 5.0],
            [0.0, 4.0, 0.0, -2.0],
            [0.0, 0.0, 8.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let i = invert4(m);
        let r = mul4(m, i);
        for c in 0..4 {
            for row in 0..4 {
                let expect = if c == row { 1.0 } else { 0.0 };
                assert!(
                    (r[c][row] - expect).abs() < 1e-5,
                    "col {c} row {row} = {}",
                    r[c][row]
                );
            }
        }
    }

    #[test]
    fn singular_inverse_returns_input() {
        let zero = [[0.0f32; 4]; 4];
        assert_eq!(invert4(zero), zero);
    }

    #[test]
    fn render_target_from_camera_is_consistent() {
        let cam = Camera3D {
            eye: Vec3::new(0.0, -10.0, 0.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            ..Default::default()
        };
        let t = RenderTarget::from_camera(&cam, 16.0 / 9.0);
        // The target must be at the centre of the frustum: z negative, x/y zero.
        let vp = t.view_proj();
        let p = Vec3::ZERO;
        let w = vp[0][3] * p.x + vp[1][3] * p.y + vp[2][3] * p.z + vp[3][3];
        let x = vp[0][0] * p.x + vp[1][0] * p.y + vp[2][0] * p.z + vp[3][0];
        let y = vp[0][1] * p.x + vp[1][1] * p.y + vp[2][1] * p.z + vp[3][1];
        assert!((x / w).abs() < 1e-4, "x={}", x / w);
        assert!((y / w).abs() < 1e-4, "y={}", y / w);
        assert!(w > 0.0, "w must be positive in front of the camera");
    }

    #[test]
    fn surface_config_defaults_are_sane() {
        let c = SurfaceConfig::default();
        assert!(c.width > 0 && c.height > 0);
        assert_eq!(c.msaa_samples, 4);
        // Zero dimensions would break texture creation.
        assert_eq!(
            SurfaceConfig {
                width: 0,
                height: 0,
                msaa_samples: 1
            }
            .width
            .max(1),
            1
        );
    }

    #[test]
    fn errors_have_messages() {
        let msgs = [
            GpuError::NoAdapter.to_string(),
            GpuError::NoFormat.to_string(),
            GpuError::Buffer("x".into()).to_string(),
        ];
        assert!(msgs.iter().all(|m| !m.is_empty()));
        assert!(msgs[0].contains("adapter"));
    }
}
