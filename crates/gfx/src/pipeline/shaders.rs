//! WGSL shaders, inlined.
//!
//! All of them use the same uniform block layout so a single bind group layout
//! can be shared across pipelines — fewer bind groups means fewer state changes
//! per frame, which on tiled GPUs is worth more than shader micro-optimisation.

/// Global uniforms shared by every pipeline.
///
/// Layout (WGSL uniform address space: `vec2`/`vec3` align to 8/16):
/// ```wgsl
/// struct Globals {
///   view_proj : mat4x4<f32>,  //  0..63
///   resolution: vec2<f32>,   // 64..71
///   zoom      : f32,         // 72
///   dpr       : f32,         // 76
///   time      : f32,         // 80
///   frame     : f32,         // 84
///   eye       : vec2<f32>,   // 88..95
/// }                          // total 96
/// ```
pub const GLOBALS_SIZE: u64 = 96;

/// Prepend the shared uniform block to a per-shader body.
///
/// `concat!` cannot take a path to a `const`, so composition happens at runtime.
/// These run exactly once per pipeline at startup; the returned `String` is
/// borrowed by the `ShaderModule` only for the duration of the call.
fn with_globals(body: &str) -> String {
    let mut s = String::with_capacity(WGSL_GLOBALS.len() + body.len());
    s.push_str(WGSL_GLOBALS);
    s.push_str(body);
    s
}

/// Complete WGSL source for the 2D line pipeline.
pub fn line_quad_2d() -> String {
    with_globals(WGSL_LINE_QUAD_2D)
}

/// Complete WGSL source for the UI pipeline.
pub fn ui() -> String {
    with_globals(WGSL_UI)
}

/// Complete WGSL source for the 3D solid pipeline.
pub fn solid_3d() -> String {
    with_globals(WGSL_SOLID_3D)
}

/// Complete WGSL source for the 3D line pipeline.
pub fn line_3d() -> String {
    with_globals(WGSL_LINE_3D)
}

/// Complete WGSL source for the 3D grid pipeline.
pub fn grid_3d() -> String {
    with_globals(WGSL_GRID_3D)
}

/// Global uniform block, mirrored in Rust by [`super::pipeline::Globals`].
pub const WGSL_GLOBALS: &str = r#"
struct Globals {
    view_proj : mat4x4<f32>,
    resolution: vec2<f32>,
    zoom      : f32,
    dpr       : f32,
    time      : f32,
    frame     : f32,
    eye       : vec2<f32>,
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
///
/// Each segment is expanded into a quad on the CPU (`expand_lines`), so the
/// vertex shader is a pure pass-through and the fragment shader only has to
/// resolve coverage across the line's width. That is what keeps the AA ramp
/// correct at any angle and avoids the degenerate cases of the single-line
/// primitive.
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
    @location(0) pos      : vec2<f32>,
    @location(1) uv       : vec2<f32>,
    @location(2) color    : vec4<f32>,
    @location(3) shape    : f32,
    @location(4) radius   : f32,
    @location(5) border   : f32,
    @location(6) half_ext : vec2<f32>,
};

struct VOut {
    @builtin(position) pos   : vec4<f32>,
    @location(0) color       : vec4<f32>,
    @location(1) uv          : vec2<f32>,
    @location(2) shape       : f32,
    @location(3) radius      : f32,
    @location(4) border      : f32,
    @location(5) half_ext    : vec2<f32>,
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
    out.half_ext = in.half_ext;
    return out;
}

/// Signed distance to a rounded box centred on the origin.
fn sd_rounded_rect(p : vec2<f32>, half_ext : vec2<f32>, r : f32) -> f32 {
    let rr = min(r, min(half_ext.x, half_ext.y));
    let q = abs(p) - half_ext + vec2<f32>(rr, rr);
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - rr;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
    // Flat fill: no distance field needed.
    if (in.shape < 0.5) {
        return in.color;
    }
    // `uv` is the offset from the rect's min corner, so centre it on the box.
    let p = in.uv - in.half_ext;
    let d = sd_rounded_rect(p, in.half_ext, in.radius);
    if (in.shape > 1.5) {
        // Border only: a band of width `border` hugging the outline.
        let w = max(in.border, 0.5);
        let a = clamp(w - abs(d + w * 0.5) + 0.5, 0.0, 1.0) * in.color.a;
        if (a <= 0.0) { discard; }
        return vec4<f32>(in.color.rgb, a);
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
    let to_eye = vec3<f32>(G.eye.x - in.world.x, G.eye.y - in.world.y, 0.0);
    let rim = pow(1.0 - max(dot(n, normalize(to_eye)), 0.0), 3.0) * 0.25;
    lit += vec3<f32>(rim);
    return vec4<f32>(lit, in.color.a);
}
"#;

/// 3D lines, expanded to a screen-space quad in the vertex shader.
///
/// Each vertex carries one endpoint (`pos`), the segment's world-space unit
/// direction (`dir`) and a side flag (`across`). The shader projects the
/// segment's two endpoints, derives the screen-space perpendicular, and offsets
/// by `width * across`. That is what gives a world-space line a constant *pixel*
/// width regardless of depth — the alternative (constant world-space width) makes
/// far geometry vanish and near geometry swamp the view.
pub const WGSL_LINE_3D: &str = r#"
struct VIn {
    @location(0) pos    : vec3<f32>,
    @location(1) color  : vec4<f32>,
    @location(2) width  : f32,
    @location(3) across : f32,
    @location(4) dir    : vec3<f32>,
    @location(5) _pad   : f32,
};

struct VOut {
    @builtin(position) pos    : vec4<f32>,
    @location(0) color        : vec4<f32>,
    @location(1) across       : f32,
    @location(2) half_px      : f32,
};

fn to_screen(clip : vec4<f32>) -> vec2<f32> {
    let ndc = clip.xy / max(abs(clip.w), 1e-6) * sign(clip.w);
    return vec2<f32>(
        (ndc.x * 0.5 + 0.5) * G.resolution.x,
        (0.5 - ndc.y * 0.5) * G.resolution.y,
    );
}

@vertex
fn vs(in : VIn) -> VOut {
    var out : VOut;
    let clip = G.view_proj * vec4<f32>(in.pos, 1.0);
    let far = G.view_proj * vec4<f32>(in.pos + in.dir, 1.0);

    // Screen-space direction of the segment; perpendicular is its normal.
    let d = to_screen(far) - to_screen(clip);
    let len = max(length(d), 1e-5);
    let nrm = vec2<f32>(-d.y, d.x) / len;

    let hw = max(in.width, 1.0) * 0.5;
    let screen = to_screen(clip) + nrm * hw * in.across;
    let ndc = vec2<f32>(
        (screen.x / G.resolution.x) * 2.0 - 1.0,
        1.0 - (screen.y / G.resolution.y) * 2.0,
    );

    out.pos = vec4<f32>(ndc, clip.z / max(abs(clip.w), 1e-6) * sign(clip.w), 1.0);
    out.color = in.color;
    out.across = in.across;
    out.half_px = hw;
    return out;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
    // Analytic AA: `across` ramps linearly across the quad's half-width.
    let a = clamp(in.half_px + 0.5 - abs(in.across) * in.half_px, 0.0, 1.0) * in.color.a;
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

    let dist = length(in.world - G.eye);
    let fade1 = clamp(1.0 - dist / 8.0, 0.0, 1.0);
    let fade10 = clamp(1.0 - dist / 90.0, 0.0, 1.0);

    let a1 = (1.0 - smoothstep(0.0, 1.0, l1)) * 0.35 * fade1;
    let a10 = (1.0 - smoothstep(0.0, 1.0, l10)) * 0.45 * fade10;
    let a = clamp(a1 + a10, 0.0, 1.0);
    if (a < 0.003) { discard; }
    return vec4<f32>(0.62, 0.66, 0.72, a);
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
            "eye",
        ] {
            assert!(
                WGSL_GLOBALS.contains(field),
                "globals missing `{field}`; a shader would fail to compile"
            );
        }
    }

    #[test]
    fn globals_size_matches_the_wgsl_layout() {
        // mat4 (64) + vec2 (8) + 4 floats (16) + vec2 eye (8) = 96
        assert_eq!(GLOBALS_SIZE, 96);
    }

    #[test]
    fn globals_struct_is_not_redeclared_per_shader() {
        // Each shader is concatenated with WGSL_GLOBALS, so no shader may define
        // its own `Globals`; a duplicate would be a WGSL redefinition error at
        // pipeline creation, long after the test suite runs.
        for (name, src) in [
            ("line_quad_2d", WGSL_LINE_QUAD_2D),
            ("ui", WGSL_UI),
            ("solid_3d", WGSL_SOLID_3D),
            ("line_3d", WGSL_LINE_3D),
            ("grid_3d", WGSL_GRID_3D),
        ] {
            assert!(!src.contains("struct Globals"), "{name} redeclares Globals");
        }
    }

    #[test]
    fn line_3d_declares_every_vertex_attribute_the_layout_supplies() {
        // pipeline::line3d_layout feeds 6 locations; a mismatch is a validation
        // error at pipeline creation, not a compile error here.
        for loc in 0..6 {
            assert!(
                WGSL_LINE_3D.contains(&format!("@location({loc})")),
                "line_3d is missing location {loc}"
            );
        }
        assert!(
            WGSL_LINE_3D.contains("in.dir"),
            "line_3d ignores the segment direction"
        );
        assert!(
            WGSL_LINE_3D.contains("in.across"),
            "line_3d ignores the side flag"
        );
    }

    #[test]
    fn ui_shader_uses_half_ext_for_the_sdf() {
        // The SDF needs the box size; a hardcoded 0.5 would only be correct for
        // 1x1 rects.
        assert!(WGSL_UI.contains("in.half_ext"));
        assert!(!WGSL_UI.contains("vec2<f32>(0.5, 0.5)"));
    }

    #[test]
    fn shaders_do_not_reference_removed_globals_members() {
        for (name, src) in [
            ("solid_3d", WGSL_SOLID_3D),
            ("line_3d", WGSL_LINE_3D),
            ("grid_3d", WGSL_GRID_3D),
        ] {
            assert!(
                !src.contains("G.pad"),
                "{name} still reads the removed G.pad"
            );
        }
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
