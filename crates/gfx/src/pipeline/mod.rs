//! wgpu pipelines: bind group layouts, vertex layouts, and the pipeline set.
//!
//! Pipelines are created once and reused for the life of the surface. Nothing
//! here is recreated per frame — that is what keeps a CAD viewport at 120 fps on
//! a laptop.

pub mod shaders;

use crate::batch::{LineVertex, LineVertex3d, SolidVertex, UiVertex};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Mirrors `Globals` in `shaders.rs`. Must stay 96 bytes and 16-byte aligned.
///
/// `#[repr(C)]` plus explicit padding is what keeps this byte-compatible with
/// the WGSL `struct Globals`: `bytemuck` refuses the derive outright if the two
/// ever disagree, which is the failure mode we want.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C, align(16))]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub resolution: [f32; 2],
    pub zoom: f32,
    pub dpr: f32,
    pub time: f32,
    pub frame: f32,
    /// World-space eye position (x, y). The 3D shaders use it for the rim light
    /// and the grid's distance fade.
    pub eye: [f32; 2],
    pub _pad: [f32; 2],
}

impl Default for Globals {
    fn default() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            resolution: [1.0, 1.0],
            zoom: 1.0,
            dpr: 1.0,
            time: 0.0,
            frame: 0.0,
            eye: [0.0, 0.0],
            _pad: [0.0, 0.0],
        }
    }
}

/// Shader compilation failed.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderError {
    pub name: &'static str,
    pub message: String,
}

impl std::fmt::Display for ShaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "shader `{}` failed to compile: {}",
            self.name, self.message
        )
    }
}

impl std::error::Error for ShaderError {}

/// Vertex buffer layouts, one per pipeline. Keeping them as functions (rather
/// than statics built with `once_cell` or `lazy_static`) means no extra crates.
fn line_quad_layout() -> wgpu::VertexBufferLayout<'static> {
    // pos(2) local(2) color(4) pattern(4) width(1)
    const ATTR: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32x2,
        2 => Float32x4,
        3 => Float32x4,
        4 => Float32,
    ];
    wgpu::VertexBufferLayout {
        array_stride: 52,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTR,
    }
}

/// Expanded 2D line quad: position, across-axis offset, along distance.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct LineQuadVertex {
    pub pos: [f32; 2],
    pub local: [f32; 2],
    pub color: [f32; 4],
    pub pattern: [f32; 4],
    pub width: f32,
    /// Explicit padding. `LineVertex` and `LineQuadVertex` are deliberately the
    /// same size so `stride_helpers_are_stable` and the layout test agree; the
    /// `Pod` derive rejects implicit padding, so it has to be written out.
    pub _pad: [f32; 3],
}

fn ui_layout() -> wgpu::VertexBufferLayout<'static> {
    // pos(2) uv(2) color(4) shape radius border half_ext(2)
    const ATTR: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32x2,
        2 => Float32x4,
        3 => Float32,
        4 => Float32,
        5 => Float32,
        6 => Float32x2,
    ];
    wgpu::VertexBufferLayout {
        array_stride: 56,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTR,
    }
}

fn solid_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTR: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x4,
        3 => Float32,
    ];
    wgpu::VertexBufferLayout {
        array_stride: 44,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTR,
    }
}

fn line3d_layout() -> wgpu::VertexBufferLayout<'static> {
    // pos(3) color(4) width(1) across(1) dir(3) pad(1)
    const ATTR: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x4,
        2 => Float32,
        3 => Float32,
        4 => Float32x3,
        5 => Float32,
    ];
    wgpu::VertexBufferLayout {
        array_stride: 52,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTR,
    }
}

fn grid_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTR: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];
    wgpu::VertexBufferLayout {
        array_stride: 12,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTR,
    }
}

/// The GPU color target format + alpha mode we render everything into.
pub fn color_target(format: wgpu::TextureFormat) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    }
}

/// Blend state for opaque 3D solids: no blending, but alpha must still write.
fn opaque_target(format: wgpu::TextureFormat) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    }
}

/// All pipelines plus the bind group they share.
pub struct PipelineSet {
    pub globals_layout: wgpu::BindGroupLayout,
    pub globals_buffer: wgpu::Buffer,
    pub globals_bind_group: wgpu::BindGroup,
    pub line_quad: wgpu::RenderPipeline,
    pub ui: wgpu::RenderPipeline,
    pub solid: wgpu::RenderPipeline,
    pub line3d: wgpu::RenderPipeline,
    pub grid: wgpu::RenderPipeline,
    pub msaa_samples: u32,
}

impl PipelineSet {
    /// Compile everything against `format`.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, msaa_samples: u32) -> Self {
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let globals = Globals::default();
        let globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("globals"),
            contents: bytemuck::bytes_of(&globals),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals-bind-group"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });
        // One layout shared by every pipeline: the bind group is then set once
        // per frame instead of once per draw.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shared-pipeline-layout"),
            bind_group_layouts: &[&globals_layout],
            immediate_size: 0,
        });

        // `depth_write_enabled` / `depth_compare` are `Option` in wgpu 30. The
        // `Some`/`None` split is load-bearing, not decoration: a depth-tested
        // but non-writing pass must leave the buffer alone.
        let depth_write = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });
        let depth_test_only = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        let target = color_target(format);
        let msaa = wgpu::MultisampleState {
            count: msaa_samples,
            ..Default::default()
        };
        let triangle_list = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        };

        // Build one pipeline per (module, vertex layout, depth policy) pair.
        // The closure owns the `ShaderModule` so both stages can borrow it.
        let build = |name: &'static str,
                     source: &str,
                     layout: wgpu::VertexBufferLayout<'static>,
                     depth_stencil: Option<wgpu::DepthStencilState>,
                     cull: Option<wgpu::Face>| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[layout],
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: cull,
                    ..triangle_list
                },
                depth_stencil,
                multisample: msaa,
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(target)],
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        let line_quad = build(
            "line-quad-2d",
            &concat!(shaders::WGSL_GLOBALS, shaders::WGSL_LINE_QUAD_2D),
            line_quad_layout(),
            None,
            None,
        );
        let ui = build(
            "ui",
            &concat!(shaders::WGSL_GLOBALS, shaders::WGSL_UI),
            ui_layout(),
            None,
            None,
        );
        let solid = build(
            "solid-3d",
            &concat!(shaders::WGSL_GLOBALS, shaders::WGSL_SOLID_3D),
            solid_layout(),
            depth_write,
            // Solids are back-face culled; lines and UI quads are not, because
            // their winding depends on the drag direction.
            Some(wgpu::Face::Back),
        );
        let line3d = build(
            "line-3d",
            &concat!(shaders::WGSL_GLOBALS, shaders::WGSL_LINE_3D),
            line3d_layout(),
            depth_test_only,
            None,
        );
        let grid = build(
            "grid-3d",
            &concat!(shaders::WGSL_GLOBALS, shaders::WGSL_GRID_3D),
            grid_layout(),
            depth_test_only,
            None,
        );

        Self {
            globals_layout,
            globals_buffer,
            globals_bind_group,
            line_quad,
            ui,
            solid,
            line3d,
            grid,
            msaa_samples,
        }
    }

    /// Upload new globals for this frame.
    pub fn update_globals(&self, queue: &wgpu::Queue, g: &Globals) {
        queue.write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(g));
    }
}

fn module(device: &wgpu::Device, name: &'static str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

/// Vertex + fragment states for a pipeline whose two stages share one module.
///
/// Returning a tuple lets the caller bind the same `ShaderModule` to both
/// stages, which halves the module count. `buffers` must outlive the pipeline
/// descriptor, so it is passed by reference straight from a `const`.
#[allow(clippy::type_complexity)]
fn states<'a>(
    device: &wgpu::Device,
    name: &'static str,
    source: &str,
    buffers: &'a [wgpu::VertexBufferLayout<'a>],
    targets: &'a [Option<wgpu::ColorTargetState>],
) -> (
    wgpu::ShaderModule,
    wgpu::VertexState<'a>,
    wgpu::FragmentState<'a>,
) {
    let m = module(device, name, source);
    let vertex = wgpu::VertexState {
        module: &m,
        entry_point: Some("vs"),
        compilation_options: Default::default(),
        buffers,
    };
    let fragment = wgpu::FragmentState {
        module: &m,
        entry_point: Some("fs"),
        compilation_options: Default::default(),
        targets,
    };
    (m, vertex, fragment)
}

/// Expand CPU line segments into GPU triangles.
///
/// Each segment becomes 6 vertices; this keeps the shader trivial and, unlike the
/// line-list topology, gives correct `width` semantics on every backend.
pub fn expand_lines(lines: &[LineVertex], out: &mut Vec<LineQuadVertex>) {
    out.clear();
    out.reserve(lines.len() * 6);
    for l in lines {
        let ax = l.a[0];
        let ay = l.a[1];
        let bx = l.b[0];
        let by = l.b[1];
        let dx = bx - ax;
        let dy = by - ay;
        let len = (dx * dx + dy * dy).sqrt().max(1e-5);
        let nx = -dy / len;
        let ny = dx / len;
        // Corner offsets: (-1, 0) start-left, (+1, 0) start-right, etc.
        let corners: [(f32, f32, f32, f32); 4] = [
            (-1.0, 0.0, ax, ay),
            (1.0, 0.0, ax, ay),
            (1.0, len, bx, by),
            (-1.0, 0.0, ax, ay),
            (1.0, len, bx, by),
            (-1.0, len, bx, by),
        ];
        for (across, along, px, py) in corners {
            let hw = (if l.width > 0.0 { l.width } else { 1.0 }) * 0.5;
            out.push(LineQuadVertex {
                pos: [px + nx * across * hw, py + ny * across * hw],
                local: [across, along],
                color: l.color,
                pattern: l.pattern,
                width: l.width,
                _pad: [0.0; 3],
            });
        }
    }
}

/// Byte size the 3D line vertex buffer needs.
pub const fn line3d_stride() -> u64 {
    std::mem::size_of::<LineVertex3d>() as u64
}
/// Byte size the UI vertex buffer needs.
pub const fn ui_stride() -> u64 {
    std::mem::size_of::<UiVertex>() as u64
}
/// Byte size the 3D solid vertex buffer needs.
pub const fn solid_stride() -> u64 {
    std::mem::size_of::<SolidVertex>() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::{Rgba, Vec2};

    const C: Rgba = Rgba::new(1.0, 0.0, 0.0, 1.0);

    #[test]
    fn globals_are_pod_and_sized() {
        let g = Globals::default();
        let bytes = bytemuck::bytes_of(&g);
        assert_eq!(bytes.len(), shaders::GLOBALS_SIZE as usize);
        assert_eq!(
            bytes.len() % 16,
            0,
            "uniform buffers need 16-byte alignment"
        );
    }

    #[test]
    fn vertex_layouts_match_their_structs() {
        assert_eq!(
            line_quad_layout().array_stride,
            std::mem::size_of::<LineQuadVertex>() as u64
        );
        assert_eq!(
            ui_layout().array_stride,
            std::mem::size_of::<UiVertex>() as u64
        );
        assert_eq!(
            solid_layout().array_stride,
            std::mem::size_of::<SolidVertex>() as u64
        );
        assert_eq!(line3d_layout().array_stride, line3d_stride());
        assert_eq!(grid_layout().array_stride, 12);
    }

    #[test]
    fn expand_lines_makes_six_vertices_per_segment() {
        let lines = vec![
            LineVertex::new(Vec2::ZERO, Vec2::new(10.0, 0.0), C, 2.0),
            LineVertex::new(Vec2::ZERO, Vec2::new(0.0, 10.0), C, 2.0),
        ];
        let mut out = Vec::new();
        expand_lines(&lines, &mut out);
        assert_eq!(out.len(), 12);
        // `out` is reused across calls: capacity is kept, length is reset.
        expand_lines(&lines, &mut out);
        assert_eq!(out.len(), 12);
    }

    #[test]
    fn expanded_quad_is_offset_perpendicular() {
        let lines = vec![LineVertex::new(
            Vec2::new(5.0, 5.0),
            Vec2::new(15.0, 5.0),
            C,
            2.0,
        )];
        let mut out = Vec::new();
        expand_lines(&lines, &mut out);
        // Horizontal segment, width 2 -> the across offsets are +-1 in Y.
        assert!((out[0].pos[1] - 4.0).abs() < 1e-5, "{:?}", out[0].pos);
        assert!((out[1].pos[1] - 6.0).abs() < 1e-5, "{:?}", out[1].pos);
        // `local.x` carries the sign so the shader can compute the AA ramp.
        assert_eq!(out[0].local[0], -1.0);
        assert_eq!(out[1].local[0], 1.0);
        // `local.y` carries the along-axis distance in pixels.
        assert_eq!(out[0].local[1], 0.0);
        assert!((out[2].local[1] - 10.0).abs() < 1e-4);
    }

    #[test]
    fn hairline_segments_get_one_pixel_width() {
        let lines = vec![LineVertex::new(Vec2::ZERO, Vec2::new(10.0, 0.0), C, 0.0)];
        let mut out = Vec::new();
        expand_lines(&lines, &mut out);
        assert!((out[1].pos[1] - 0.5).abs() < 1e-5, "{:?}", out[1].pos);
    }

    #[test]
    fn zero_length_segment_does_not_produce_nan() {
        let lines = vec![LineVertex::new(Vec2::ZERO, Vec2::ZERO, C, 2.0)];
        let mut out = Vec::new();
        expand_lines(&lines, &mut out);
        for v in &out {
            assert!(v.pos[0].is_finite() && v.pos[1].is_finite(), "{v:?}");
        }
    }

    #[test]
    fn dashed_pattern_is_carried_through() {
        let lines =
            vec![LineVertex::new(Vec2::ZERO, Vec2::new(10.0, 0.0), C, 1.0).dashed(4.0, 2.0, 1.0)];
        let mut out = Vec::new();
        expand_lines(&lines, &mut out);
        assert_eq!(out[0].pattern, [4.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn expand_handles_empty_input() {
        let mut out = vec![LineQuadVertex {
            pos: [0.0; 2],
            local: [0.0; 2],
            color: [0.0; 4],
            pattern: [0.0; 4],
            width: 0.0,
            _pad: [0.0; 3],
        }];
        expand_lines(&[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn stride_helpers_are_stable() {
        // These feed buffer allocations, so a silent change would corrupt
        // rendering. Pin them.
        assert_eq!(ui_stride(), 56);
        assert_eq!(solid_stride(), 44);
        assert_eq!(line3d_stride(), 52);
    }

    #[test]
    fn line_quad_stride_matches_the_expansion() {
        // `expand_lines` writes LineQuadVertex, which is what gets uploaded under
        // the line_quad layout; if the two drift, every segment renders garbage.
        assert_eq!(
            line_quad_layout().array_stride,
            std::mem::size_of::<LineQuadVertex>() as u64
        );
        assert_eq!(
            std::mem::size_of::<LineVertex>(),
            std::mem::size_of::<LineQuadVertex>()
        );
    }

    #[test]
    fn strides_are_four_byte_aligned() {
        // wgpu requires array_stride to be a multiple of 4 for every backend;
        // a stray `f32` in a struct is the usual way to break that.
        for s in [
            line_quad_layout(),
            ui_layout(),
            solid_layout(),
            line3d_layout(),
            grid_layout(),
        ] {
            assert_eq!(
                s.array_stride % 4,
                0,
                "stride {} is not 4-byte aligned",
                s.array_stride
            );
        }
    }
}
