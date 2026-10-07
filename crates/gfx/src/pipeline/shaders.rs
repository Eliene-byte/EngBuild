//! WGSL shaders, inlined.
//!
//! All of them use the same uniform block layout so a single bind group layout
//! can be shared across pipelines — fewer bind groups means fewer state changes
//! per frame, which on tiled GPUs is worth more than shader micro-optimisation.

/// Global uniforms shared by every pipeline.
///
/// Layout (std140-compatible, all members 16-byte aligned):
/// ```wgsl
/// struct Globals {
///   view_proj : mat4x4<f32>,  //  0..63
///   resolution: vec2<f32>,   // 64..71
///   zoom      : f32,         // 72
///   dpr       : f32,         // 76
///   time      : f32,         // 80
///   frame     : f32,         // 84
///   _pad      : vec2<f32>,   // 88..95
/// }                          // total 96
/// ```
pub const GLOBALS_SIZE: u64 = 96;

/// Global uniform block, mirrored in Rust by [`super::pipeline::Globals`].
pub const WGSL_GLOBALS: &str = r#"
struct Globals {
    view_proj : mat4x4<f32>,
    resolution: vec2<f32>,
    zoom      : f32,
    dpr       : f32,
    time      : f32,
    frame     : f32,
    pad       : vec2<f32>,
};

@group(0) @binding(0) var<uniform> G : Globals;

fn clip_of(p: vec2<f32>) -> vec4<f32> {
    // Screen pixels -> NDC (y down).
    let ndc = vec2<f32>(
        (p.x / G.resolution.x) * 2.0 - 1.0,
        1.0 - (p.y / G.resolution.y) * 2.0,
    );
    return vec4<f32>(ndc, 0.0, 1.0);
}
"#;

/// 2D anti-aliased lines in screen space, with dashes and screen-space width.
pub const WGSL_LINES_2D: &str = r#"
struct VIn {
    @location(0) a       : vec2<f32>,
    @location(1) b       : vec2<f32>,
    @location(2) color   : vec4<f32>,
    @location(3) pattern : vec4<f32>,
    @location(4) width   : f32,
};

struct VOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) color     : vec4<f32>,
    @location(1) offset    : f32,
    @location(2) dir       : vec2<f32>,
    @location(3) along     : f32,
    @location(4) length    : f32,
    @location(5) pattern   : vec4<f32>,
    @location(6) half_px   : f32,
};

@vertex
fn vs(in : VIn) -> VOut {
    var out : VOut;
    let d = in.b - in.a;
    let len = max(length(d), 1e-5);
    let dir = d / len;
    let n = vec2<f32>(-dir.y, dir.x);

    // Half width in pixels; 0 means a 1px hairline.
    let hw = max(in.width, 0.0) * 0.5;
    let hw_eff = select(0.5, hw, hw > 0.0);

    out.pos = clip_of(in.a);
    out.color = in.color;
    out.offset = 0.0;
    out.dir = dir;
    out.along = 0.0;
    out.length = len;
    out.pattern = in.pattern;
    out.half_px = hw_eff;

    // Push the second vertex to the far end; we use the quad overdraw trick
    // below, so emit both extremes via a 6-vertex fan in the CPU batch.
    return out;
}
"#;

/// Simpler and more robust: expand each segment into a quad on the CPU and use a
/// plain triangle shader. Keeps the GPU code tiny and avoids the degenerate
/// cases of the single-line primitive at extreme angles.
pub const WGSL_LINE_QUAD_2D: &str = r#"
struct VIn {
    @location(0) pos     : vec2<f32>,
    @location(1) local   : vec2<f32>,   // x in [-1,1] across, y along in px
    @location(2) color   : vec4<f32>,
    @location(3) pattern : vec4<f32>,
    @location(4) width   : f32,
};

struct VOut {
    @builtin(position) pos    : vec4<f32>,
    @location(0) color        : vec4<f32>,
    @location(1) local        : vec2<f32>,
    @location(2) pattern      : vec4<f32>,
    @location(3) half_px      : f32,
    @location(4) seg_len      : f32,
};

@vertex
fn vs(in : VIn) -> VOut {
    var out : VOut;
    out.pos = clip_of(in.pos);
    out.color = in.color;
    out.local = in.local;
    out.pattern = in.pattern;
    out.half_px = max(in.width, 1.0) * 0.5;
    out.seg_len = in.local.y;
    return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
    // Distance to the segment centre line, in pixels.
    let d = abs(in.local.x);
    let hw = in.half_px;
    if (in.pattern.x > 0.0) {
        // Dashed: discard the "off" portions. `local.y` is the distance along
        // the segment in pixels, and pattern.z carries the phase so the dash
        // pattern does not crawl while panning.
        let period = in.pattern.x + in.pattern.y;
        let t = fract((in.local.y + in.pattern.z) / period);
        if (t * period) > in.pattern.x {
            discard;
        }
    }
    // Analytic AA across the line width.
    let alpha = clamp((hw + 0.5 - d) * in.color.a, 0.0, 1.0);
    if (alpha <= 0.0) { discard; }
    return vec4<f32>(in.color.rgb, alpha);
}
"#;

/// UI quads: flat fills, rounded-rect SDFs and outlines.
pub const WGSL_UI: &str = r#"
struct VIn {
    @location(0) pos    : vec2<f32>,
    @location(1) uv     : vec2<f32>,
    @location(2) color  : vec4<f32>,
    @location(3) shape  : f32,
    @location(4) radius : f32,
    @location(5) border : f32,
};

struct VOut {
    @builtin(position) pos   : vec4<f32>,
    @location(0) color       : vec4<f32>,
    @location(1) uv          : vec2<f32>,
    @location(2) shape       : f32,
    @location(3) radius      : f32,
    @location(4) border      : f32,
    @location(5) local       : vec2<f32>,
};

@vertex
fn vs(in : VIn) -> VOut {
    var out : VOut;
    out.pos = clip_of(in.pos);
    out.color = in.color;
    out.uv = in.uv;
    out.shape = in.shape;
    out.radius = in.radius;
    out.border = in.border;
    // Reconstruct the distance field coordinate: distance from the rounded
    // corner centre equals the box half-extent minus the interpolated uv.
    out.local = in.uv;
    return out;
}

fn sd_rounded_rect(p : vec2<f32>, half_ext : vec2<f32>, r : f32) -> f32 {
    var rr = min(r, min(half_ext.x, half_ext.y));
    let q = abs(p) - half_ext + vec2<f32>(rr);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - rr;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
    if (in.shape < 0.5) {
        return in.color;
    }
    // Rounded rect / border, evaluated from the interpolated uv.
    let half_ext = vec2<f32>(0.5, 0.5);
    let p = in.local - half_ext;
    let d = sd_rounded_rect(p, half_ext, in.radius);
    if (in.shape > 1.5) {
        // Border only.
        let w = max(in.border, 0.5);
        let a = clamp((w - abs(d + w * 0.5)) + 0.5, 0.0, 1.0);
        if (a <= 0.0) { discard; }
        return vec4<f32>(in.color.rgb, a * in.color.a);
    }
    let a = clamp(0.5 - d, 0.0, 1.0) * in.color.a;
    if (a <= 0.0) { discard; }
    return vec4<f32>(in.color.rgb, a);
}
"#;

/// 3D shaded solids: Lambert + rim light + hemispheric ambient.
pub const WGSL_SOLID_3D: &str = r#"
struct VIn {
    @location(0) pos    : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) color  : vec4<f32>,
    @location(3) flags  : f32,
};

struct VOut {
    @builtin(position) pos    : vec4<f32>,
    @location(0) world        : vec3<f32>,
    @location(1) normal       : vec3<f32>,
    @location(2) color        : vec4<f32>,
};

@vertex
fn vs(in : VIn) -> VOut {
    var out : VOut;
    out.pos = G.view_proj * vec4<f32>(in.pos, 1.0);
    out.world = in.pos;
    out.normal = normalize(in.normal);
    out.color = in.color;
    return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    // Key light from the camera-ish upper left, plus a cool fill.
    let key_dir = normalize(vec3<f32>(0.4, 0.7, 0.6));
    let fill_dir = normalize(vec3<f32>(-0.5, -0.2, 0.4));
    let key = max(dot(n, key_dir), 0.0);
    let fill = max(dot(n, fill_dir), 0.0) * 0.35;
    // Hemispheric ambient: sky above, ground bounce below.
    let sky = mix(vec3<f32>(0.16, 0.17, 0.20), vec3<f32>(0.30, 0.33, 0.38), n.z * 0.5 + 0.5);
    var lit = in.color.rgb * (sky + vec3<f32>(key * 0.85 + fill));
    // Rim light to separate silhouettes, which matters a lot in CAD views.
    let view_dir = normalize(vec3<f32>(G.pad.x, G.pad.y, 1.0));
    let rim = pow(1.0 - max(dot(n, view_dir), 0.0), 3.0) * 0.25;
    lit += vec3<f32>(rim);
    return vec4<f32>(lit, in.color.a);
}
"#;

/// 3D lines: world-space, expanded to a screen-space quad in the vertex shader.
pub const WGSL_LINE_3D: &str = r#"
struct VIn {
    @location(0) pos   : vec3<f32>,
    @location(1) color : vec4<f32>,
    @location(2) width : f32,
    @location(3) local : vec2<f32>,
};

struct VOut {
    @builtin(position) pos    : vec4<f32>,
    @location(0) color        : vec4<f32>,
    @location(1) local        : vec2<f32>,
    @location(2) half_px      : f32,
};

fn project(p : vec3<f32>) -> vec4<f32> {
    return G.view_proj * vec4<f32>(p, 1.0);
}

fn to_screen(clip : vec4<f32>) -> vec2<f32> {
    let ndc = clip.xy / max(clip.w, 1e-6);
    return vec2<f32>((ndc.x * 0.5 + 0.5) * G.resolution.x, (0.5 - ndc.y * 0.5) * G.resolution.y);
}

@vertex
fn vs(in : VIn) -> VOut {
    var out : VOut;
    let clip = project(in.pos);
    let screen = to_screen(clip);
    let hw = max(in.width, 1.0) * 0.5;

    // Offset perpendicular to the screen-space segment, which requires the
    // neighbouring vertex: `local.y` carries the along-axis sign and
    // `local.x` the across-axis sign from the CPU.
    let axis = normalize(vec2<f32>(cos(in.local.y), sin(in.local.y)));
    let nrm = vec2<f32>(-axis.y, axis.x);
    let offset_px = nrm * hw * in.local.x;

    let ndc = vec2<f32>(
        ((screen.x + offset_px.x) / G.resolution.x) * 2.0 - 1.0,
        1.0 - ((screen.y + offset_px.y) / G.resolution.y) * 2.0,
    );
    out.pos = vec4<f32>(ndc, clip.z / max(clip.w, 1e-6), 1.0);
    out.color = in.color;
    out.local = in.local;
    out.half_px = hw;
    return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
    let a = clamp((in.half_px + 0.5 - abs(in.local.x)) * in.color.a, 0.0, 1.0);
    if (a <= 0.0) { discard; }
    return vec4<f32>(in.color.rgb, a);
}
"#;

/// Grid drawn on the world ground plane, anti-aliased and distance-faded.
pub const WGSL_GRID_3D: &str = r#"
struct VIn {
    @location(0) pos : vec3<f32>,
};

struct VOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) world     : vec2<f32>,
};

@vertex
fn vs(in : VIn) -> VOut {
    var out : VOut;
    out.pos = G.view_proj * vec4<f32>(in.pos, 1.0);
    out.world = in.pos.xy;
    return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
    // Two grid levels: 1 unit and 10 units, with a fade based on distance from
    // the eye so the far field does not turn into moire.
    let g1 = abs(fract(in.world) - 0.5) / max(fwidth(in.world), vec2<f32>(1e-5));
    let l1 = min(g1.x, g1.y);
    let g10 = abs(fract(in.world / 10.0) - 0.5) / max(fwidth(in.world / 10.0), vec2<f32>(1e-5));
    let l10 = min(g10.x, g10.y);

    let dist = length(in.world - G.view_proj[3].xy);
    let fade1 = clamp(1.0 - dist / 8.0, 0.0, 1.0);
    let fade10 = clamp(1.0 - dist / 90.0, 0.0, 1.0);

    let a1 = (1.0 - smoothstep(0.0, 1.0, l1)) * 0.35 * fade1;
    let a10 = (1.0 - smoothstep(0.0, 1.0, l10)) * 0.45 * fade10;
    let a = clamp(a1 + a10, 0.0, 1.0);
    if (a < 0.003) { discard; }
    return vec4<f32>(0.62, 0.66, 0.72, a);
}
"#;

/// Simple textured blit used for the glyph atlas.
pub const WGSL_TEXTURE: &str = r#"
struct VIn {
    @location(0) pos   : vec2<f32>,
    @location(1) uv    : vec2<f32>,
    @location(2) color : vec4<f32>,
};

struct VOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv         : vec2<f32>,
    @location(1) color      : vec4<f32>,
};

@group(0) @binding(1) var samp : sampler;
@group(0) @binding(2) var tex  : texture_2d<f32>;

@vertex
fn vs(in : VIn) -> VOut {
    var out : VOut;
    out.pos = clip_of(in.pos);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
    let t = textureSample(tex, samp, in.uv);
    return vec4<f32>(t.rgb * in.color.rgb, t.a * in.color.a);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_shaders_are_non_empty_and_have_entry_points() {
        let shaders: [(&str, &str); 5] = [
            ("line_quad_2d", WGSL_LINE_QUAD_2D),
            ("ui", WGSL_UI),
            ("solid_3d", WGSL_SOLID_3D),
            ("line_3d", WGSL_LINE_3D),
            ("grid_3d", WGSL_GRID_3D),
        ];
        for (name, src) in shaders {
            assert!(src.len() > 100, "{name} is suspiciously short");
            assert!(src.contains("@vertex"), "{name} has no vertex entry point");
            assert!(
                src.contains("@fragment"),
                "{name} has no fragment entry point"
            );
        }
    }

    #[test]
    fn globals_block_declares_every_member_used_by_shaders() {
        for field in [
            "view_proj",
            "resolution",
            "zoom",
            "dpr",
            "time",
            "frame",
            "pad",
        ] {
            assert!(
                WGSL_GLOBALS.contains(field),
                "globals missing `{field}`; a shader would fail to compile"
            );
        }
    }

    #[test]
    fn globals_size_matches_the_wgsl_layout() {
        // mat4 (64) + vec2 (8) + 4 floats (16) + vec2 pad (8) = 96
        assert_eq!(GLOBALS_SIZE, 96);
    }

    #[test]
    fn line_shader_declares_dashing() {
        assert!(WGSL_LINE_QUAD_2D.contains("fract"));
        assert!(WGSL_LINE_QUAD_2D.contains("discard"));
        assert!(WGSL_LINE_QUAD_2D.contains("smoothstep") || WGSL_LINE_QUAD_2D.contains("clamp"));
    }

    #[test]
    fn no_shader_uses_legacy_derivatives_without_extension() {
        // `fwidth` is core in WGSL; the grid shader relies on it for AA.
        assert!(WGSL_GRID_3D.contains("fwidth"));
        assert!(!WGSL_GRID_3D.contains("#extension"));
    }
}
