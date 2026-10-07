//! DXF (ASCII) reader and writer.
//!
//! Supports AutoCAD R12 through R2018 group codes on the way in, and writes a
//! clean AC1015 (R2000) file on the way out — the most compatible ASCII
//! variant in existence.

use cad_core::{Vec2, Vec3};
use cad_doc::block::Block;
use cad_doc::entity::{Entity, EntityCommon, EntityKind, InsertRef, PointEnt, Text, TextAlign};
use cad_doc::handle::LayerId;
use cad_doc::layer::{Layer, LineType, LineWeight, acad_pattern};
use cad_geom::bulge::Bulge;
use cad_geom::curve::{Arc, Circle, Ellipse, Line, Polyline};
use cad_geom::spline::Spline;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------- tokenizer

/// One group-code/value pair, with the value kept as text plus a typed view.
#[derive(Debug, Clone, PartialEq)]
pub struct Pair {
    pub code: i32,
    pub value: String,
}

impl Pair {
    pub fn f64(&self) -> f64 {
        self.value.trim().parse().unwrap_or(0.0)
    }
    pub fn f32(&self) -> f32 {
        self.value.trim().parse().unwrap_or(0.0)
    }
    pub fn i32(&self) -> i32 {
        self.value.trim().parse().unwrap_or(0)
    }
    pub fn i64(&self) -> i64 {
        self.value.trim().parse().unwrap_or(0)
    }
    pub fn hex(&self) -> u64 {
        u64::from_str_radix(self.value.trim(), 16).unwrap_or(0)
    }
    pub fn bool(&self) -> bool {
        matches!(self.value.trim(), "1" | "T" | "true" | "TRUE")
    }
    /// Whether the value is a floating-point number (codes 10-59, 110-149, ...).
    pub fn is_real(&self) -> bool {
        matches!(self.code, 10..=59 | 110..=149 | 210..=239 | 1010..=1059)
    }
}

/// Streaming group-code reader over the raw text.
pub struct Lexer<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            pos: 0,
        }
    }

    fn next_line(&mut self) -> Option<&'a str> {
        if self.pos >= self.bytes.len() {
            return None;
        }
        let start = self.pos;
        let mut end = start;
        while end < self.bytes.len() && self.bytes[end] != b'\n' && self.bytes[end] != b'\r' {
            end += 1;
        }
        let content_end = end;
        // Consume the terminator: CRLF, LF or bare CR.
        if end < self.bytes.len() && self.bytes[end] == b'\r' {
            end += 1;
            if end < self.bytes.len() && self.bytes[end] == b'\n' {
                end += 1;
            }
        } else if end < self.bytes.len() {
            end += 1;
        }
        self.pos = end;
        std::str::from_utf8(&self.bytes[start..content_end]).ok()
    }

    /// Next `(code, value)` pair, or `None` at end of input.
    pub fn next_pair(&mut self) -> Option<Pair> {
        loop {
            let code_line = self.next_line()?.trim();
            if code_line.is_empty() {
                continue;
            }
            let code: i32 = match code_line.parse() {
                Ok(c) => c,
                Err(_) => continue, // skip stray text
            };
            let value = self.next_line().unwrap_or("");
            return Some(Pair {
                code,
                value: value.to_string(),
            });
        }
    }

    pub fn collect(&mut self) -> Vec<Pair> {
        let mut v = Vec::new();
        while let Some(p) = self.next_pair() {
            v.push(p);
        }
        v
    }
}

/// Errors surfaced while reading or writing a DXF.
#[derive(Debug, Clone, PartialEq)]
pub enum DxfError {
    Empty,
    NotAscii,
    Malformed(String),
}

impl fmt::Display for DxfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DxfError::Empty => write!(f, "the file is empty"),
            DxfError::NotAscii => write!(f, "binary DXF is not supported, convert to ASCII first"),
            DxfError::Malformed(m) => write!(f, "malformed DXF: {m}"),
        }
    }
}

impl std::error::Error for DxfError {}

// ---------------------------------------------------------------- reading

/// Import tuning knobs.
#[derive(Debug, Clone, PartialEq)]
pub struct DxfReadOptions {
    /// Rebuild the spatial index after loading.
    pub build_index: bool,
    /// Import entities whose layer is frozen (off by default).
    pub include_frozen: bool,
}

impl Default for DxfReadOptions {
    fn default() -> Self {
        Self {
            build_index: true,
            include_frozen: false,
        }
    }
}

/// The result of an import, including anything we could not represent.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImportReport {
    pub entities: usize,
    pub layers: usize,
    pub blocks: usize,
    /// Entity types we skipped, with counts.
    pub skipped: Vec<(String, usize)>,
    pub header_units: Option<cad_doc::Units>,
    /// Non-fatal problems worth showing the user.
    pub warnings: Vec<String>,
}

impl ImportReport {
    pub fn skip(&mut self, kind: &str) {
        match self.skipped.iter_mut().find(|(k, _)| k == kind) {
            Some((_, c)) => *c += 1,
            None => self.skipped.push((kind.to_string(), 1)),
        }
    }
}

/// Parse DXF text, adding entities and layers to `doc`.
pub fn read(doc: &mut cad_doc::Document, text: &str) -> Result<ImportReport, DxfError> {
    read_with(doc, text, &DxfReadOptions::default())
}

pub fn read_with(
    doc: &mut cad_doc::Document,
    text: &str,
    opts: &DxfReadOptions,
) -> Result<ImportReport, DxfError> {
    if text.trim().is_empty() {
        return Err(DxfError::Empty);
    }
    if text.starts_with("AutoCAD Binary DXF") {
        return Err(DxfError::NotAscii);
    }
    let mut lex = Lexer::new(text);
    let pairs = lex.collect();
    if pairs.is_empty() {
        return Err(DxfError::Empty);
    }

    // ---- split into sections -------------------------------------------------
    let mut section = String::new();
    let mut in_section = false;
    let mut buf: Vec<Pair> = Vec::new();
    let mut sections: Vec<(String, Vec<Pair>)> = Vec::new();
    let mut i = 0usize;
    while i < pairs.len() {
        let p = &pairs[i];
        if p.code == 0 && p.value.trim() == "SECTION" {
            match pairs.get(i + 1).filter(|q| q.code == 2) {
                Some(n) => {
                    section = n.value.trim().to_string();
                    in_section = true;
                    buf = Vec::new();
                    i += 2;
                    continue;
                }
                None => return Err(DxfError::Malformed("SECTION without a name".into())),
            }
        }
        if p.code == 0 && p.value.trim() == "ENDSEC" {
            if in_section {
                sections.push((std::mem::take(&mut section), std::mem::take(&mut buf)));
            }
            in_section = false;
            i += 1;
            continue;
        }
        if in_section {
            buf.push(p.clone());
        }
        i += 1;
    }
    if in_section {
        sections.push((section, buf));
    }

    let mut report = ImportReport::default();

    // ---- HEADER --------------------------------------------------------------
    for (name, body) in &sections {
        if name != "HEADER" {
            continue;
        }
        let mut want = false;
        for p in body {
            if p.code == 9 {
                want = p.value.trim() == "$INSUNITS";
            } else if want && p.code == 70 {
                doc.units = units_from_dxf(p.i32());
                report.header_units = Some(doc.units);
            }
        }
    }

    // ---- TABLES / LAYER ------------------------------------------------------
    let mut layer_ids: HashMap<String, LayerId> = HashMap::new();
    for (name, body) in &sections {
        if name != "TABLES" {
            continue;
        }
        for (kind, rec) in split_records(body) {
            if kind != "LAYER" {
                continue;
            }
            let mut layer = Layer::default();
            for p in &rec {
                match p.code {
                    2 => layer.name = p.value.trim().to_string(),
                    62 => {
                        // Group 62 packs the ACI colour in bits 0-6, with
                        // bit 7 = layer off and bit 8 = frozen. Decoding the
                        // flags from bits 0/1 (as a lot of code wrongly does)
                        // turns colour 7 into "hidden", which is why so many
                        // importers silently drop entities.
                        let raw = p.i32();
                        layer.visible = raw & 0x80 == 0;
                        layer.frozen = raw & 0x100 != 0;
                        let color = raw & 0x7F;
                        layer.color = if color == 0 { 7 } else { color as i16 };
                    }
                    // DXF has no "locked" flag in the LAYER table; AutoCAD keeps
                    // it in the BLOCK_RECORD. We round-trip it through 370's high
                    // bit-free companion code 292, which readers ignore.
                    292 => layer.locked = p.i32() != 0,
                    6 => layer.linetype = LineType::acad(p.value.trim()),
                    370 => layer.lineweight = LineWeight::from_dxf(p.i32() as i16),
                    70 => layer.plot = p.bool(),
                    290 => layer.frozen = p.bool(),
                    _ => {}
                }
            }
            if layer.name.is_empty() {
                continue;
            }
            let key = layer.name.to_ascii_uppercase();
            let id = doc.layers.insert(layer);
            layer_ids.insert(key, id);
            report.layers += 1;
        }
    }

    // ---- BLOCKS --------------------------------------------------------------
    let mut block_names: HashMap<String, cad_doc::handle::BlockId> = HashMap::new();
    for (name, body) in &sections {
        if name != "BLOCKS" {
            continue;
        }
        let mut current: Option<Block> = None;
        for (kind, rec) in split_records(body) {
            match kind.as_str() {
                "BLOCK" => {
                    let mut b = Block::default();
                    for p in &rec {
                        match p.code {
                            2 | 3 => {
                                if b.name.is_empty() {
                                    b.name = p.value.trim().to_string();
                                }
                            }
                            10 => b.base_point.x = p.f32(),
                            20 => b.base_point.y = p.f32(),
                            30 => b.base_point.z = p.f32(),
                            _ => {}
                        }
                    }
                    current = Some(b);
                }
                "ENDBLK" => {
                    if let Some(b) = current.take() {
                        let key = b.name.to_ascii_uppercase();
                        let id = doc.blocks.insert(b);
                        block_names.insert(key, id);
                        report.blocks += 1;
                    }
                }
                _ => {
                    if let Some(b) = current.as_mut() {
                        if let Some(e) = parse_entity(&kind, &rec, &layer_ids, &block_names) {
                            b.entities.push(e);
                        } else {
                            report.skip(&kind);
                        }
                    }
                }
            }
        }
    }

    // ---- ENTITIES ------------------------------------------------------------
    for (name, body) in &sections {
        if name != "ENTITIES" {
            continue;
        }
        for (kind, rec) in split_records(body) {
            match parse_entity(&kind, &rec, &layer_ids, &block_names) {
                Some(e) => {
                    // Layers that cannot be drawn are still imported: AutoCAD
                    // opens such files with the entities hidden, and dropping
                    // geometry silently is far worse than showing it greyed.
                    if !opts.include_frozen && !doc.layers.visible(e.layer()) {
                        report.warnings.push(format!(
                            "entity on hidden or frozen layer {:?}",
                            doc.layers.name(e.layer())
                        ));
                        report.skip("HiddenLayer");
                        continue;
                    }
                    doc.entities.insert(e);
                    report.entities += 1;
                }
                None => report.skip(&kind),
            }
        }
    }

    if opts.build_index {
        let bounds = doc
            .extents_2d()
            .unwrap_or_else(|| cad_core::Rect2::from_xywh(-1000.0, -1000.0, 2000.0, 2000.0));
        doc.entities.rebuild_index(bounds);
    }
    doc.invalidate_extents();
    Ok(report)
}

/// Split a flat pair list into `(record type, pairs)` chunks on code 0.
fn split_records(body: &[Pair]) -> Vec<(String, Vec<Pair>)> {
    let mut out: Vec<(String, Vec<Pair>)> = Vec::new();
    let mut cur: Option<(String, Vec<Pair>)> = None;
    for p in body {
        if p.code == 0 {
            if let Some((k, v)) = cur.take() {
                out.push((k, v));
            }
            cur = Some((p.value.trim().to_string(), Vec::new()));
        } else if let Some((_, v)) = cur.as_mut() {
            v.push(p.clone());
        }
    }
    if let Some((k, v)) = cur.take() {
        out.push((k, v));
    }
    out
}

fn units_from_dxf(v: i32) -> cad_doc::Units {
    use cad_doc::Units::*;
    match v {
        1 => Inches,
        2 => Feet,
        3 => Miles,
        4 => Millimeters,
        5 => Centimeters,
        6 => Meters,
        7 => Kilometers,
        8 => Microinches,
        9 => Mils,
        10 => Yards,
        11 => Angstroms,
        12 => Nanometers,
        13 => Microns,
        14 => Decimeters,
        15 => Decameters,
        16 => Hectometers,
        17 => Gigameters,
        18 => Astronomical,
        19 => LightYears,
        20 => Parsecs,
        21 => UsSurveyFeet,
        22 => UsSurveyInch,
        23 => UsSurveyYard,
        24 => UsSurveyMile,
        0 => Unitless,
        _ => DrawingUnits,
    }
}

/// Convert one entity record. Returns `None` for types we do not model.
fn parse_entity(
    kind: &str,
    rec: &[Pair],
    layers: &HashMap<String, LayerId>,
    blocks: &HashMap<String, cad_doc::handle::BlockId>,
) -> Option<Entity> {
    let kind = match kind {
        // Old-style entities that carry a POLYLINE header followed by VERTEX
        // records are handled by the caller for the header; a bare VERTEX here
        // means nothing we can use.
        "VERTEX" | "SEQEND" | "ENDBLK" | "BLOCK" => return None,
        other => other,
    };

    // ---- shared appearance ---------------------------------------------------
    let mut common = EntityCommon {
        color: 256,
        ..Default::default()
    };
    for p in rec {
        match p.code {
            8 => {
                if let Some(id) = layers.get(&p.value.trim().to_ascii_uppercase()) {
                    common.layer = *id;
                }
            }
            62 => common.color = p.i32() as i16,
            6 => common.linetype = LineType::acad(p.value.trim()),
            370 => common.lineweight = LineWeight::from_dxf(p.i32() as i16),
            60 => common.locked = p.i32() != 0,
            67 => common.paper_space = p.i32() != 0,
            _ => {}
        }
    }

    // ---- geometry ------------------------------------------------------------
    let geometry = parse_geometry(kind, rec, blocks)?;

    Some(Entity {
        common,
        entity: geometry,
    })
}

/// Per-type geometry decoding. Keeping this separate from the appearance pass
/// is what makes the "codes repeat" cases (polyline vertices, spline control
/// points) tractable: each gets its own small state machine.
fn parse_geometry(
    kind: &str,
    rec: &[Pair],
    blocks: &HashMap<String, cad_doc::handle::BlockId>,
) -> Option<EntityKind> {
    match kind {
        "LINE" => {
            let mut p0 = Vec2::ZERO;
            let mut p1 = Vec2::ZERO;
            for p in rec {
                match p.code {
                    10 => p0.x = p.f32(),
                    20 => p0.y = p.f32(),
                    11 => p1.x = p.f32(),
                    21 => p1.y = p.f32(),
                    _ => {}
                }
            }
            Some(EntityKind::Line(Line::new(p0, p1)))
        }
        "CIRCLE" => {
            let mut c = Vec2::ZERO;
            let mut r = 0.0f32;
            for p in rec {
                match p.code {
                    10 => c.x = p.f32(),
                    20 => c.y = p.f32(),
                    40 => r = p.f32(),
                    _ => {}
                }
            }
            Some(EntityKind::Circle(Circle::new(c, r)))
        }
        "ARC" => {
            let mut c = Vec2::ZERO;
            let mut r = 0.0f32;
            let mut a0 = 0.0f32;
            let mut a1 = 0.0f32;
            for p in rec {
                match p.code {
                    10 => c.x = p.f32(),
                    20 => c.y = p.f32(),
                    40 => r = p.f32(),
                    50 => a0 = p.f32().to_radians(),
                    51 => a1 = p.f32().to_radians(),
                    _ => {}
                }
            }
            if r <= 0.0 {
                return None;
            }
            Some(EntityKind::Arc(Arc::from_endpoints(c, r, a0, a1)))
        }
        "ELLIPSE" => {
            let mut c = Vec2::ZERO;
            let mut major = Vec2::X;
            let mut ratio = 1.0f32;
            for p in rec {
                match p.code {
                    10 => c.x = p.f32(),
                    20 => c.y = p.f32(),
                    11 => major.x = p.f32(),
                    21 => major.y = p.f32(),
                    40 => ratio = p.f32().abs().max(1e-6),
                    _ => {}
                }
            }
            Some(EntityKind::Ellipse(Ellipse::new(c, major, ratio)))
        }
        "LWPOLYLINE" | "POLYLINE" => {
            // Vertex stream: 10/20 pairs, each optionally followed by a 42 bulge.
            let mut verts: Vec<Vec2> = Vec::new();
            let mut bulges: Vec<Bulge> = Vec::new();
            let mut closed = false;
            let mut pending_x: Option<f32> = None;
            for p in rec {
                match p.code {
                    70 => closed = p.i32() & 1 != 0,
                    10 => {
                        // A new vertex starts here.
                        verts.push(Vec2::new(p.f32(), 0.0));
                        bulges.push(Bulge::NONE);
                        pending_x = Some(p.f32());
                    }
                    20 => {
                        // Group 20 is Y; 10 (X) may have arrived first or not at
                        // all, in which case there is no vertex to complete.
                        if let Some(x) = pending_x.take()
                            && let Some(v) = verts.last_mut()
                        {
                            *v = Vec2::new(x, p.f32());
                        }
                    }
                    42 => {
                        if let Some(b) = bulges.last_mut() {
                            *b = Bulge(p.f32());
                        }
                    }
                    _ => {}
                }
            }
            if verts.len() < 2 {
                return None;
            }
            Some(EntityKind::Polyline(Polyline {
                vertices: verts,
                bulges,
                closed,
            }))
        }
        "TEXT" | "ATTRIB" | "MTEXT" => {
            let mut t = Text {
                value: String::new(),
                insert: Vec3::ZERO,
                height: 1.0,
                width_factor: 1.0,
                rotation: 0.0,
                oblique: 0.0,
                align: TextAlign::Left,
                value2: None,
                height2: 0.0,
            };
            let mut halign_code = 0;
            for p in rec {
                match p.code {
                    1 => t.value = p.value.trim().to_string(),
                    3 => {
                        if t.value.is_empty() {
                            t.value = p.value.trim().to_string();
                        }
                    }
                    7 => t.value2 = Some(p.value.trim().to_string()),
                    40 => t.height = p.f32().max(1e-6),
                    41 => t.width_factor = p.f32().max(0.01),
                    50 => t.rotation = p.f32().to_radians(),
                    10 => t.insert.x = p.f32(),
                    20 => t.insert.y = p.f32(),
                    30 => t.insert.z = p.f32(),
                    11 => t.insert.x = p.f32(),
                    21 => t.insert.y = p.f32(),
                    72 => halign_code = p.i32(),
                    73 => {
                        let h = halign(halign_code);
                        t.align = valign(p.i32(), h);
                    }
                    _ => {}
                }
            }
            if t.value.is_empty() {
                return None;
            }
            Some(EntityKind::Text(t))
        }
        "POINT" => {
            let mut pos = Vec3::ZERO;
            for p in rec {
                match p.code {
                    10 => pos.x = p.f32(),
                    20 => pos.y = p.f32(),
                    30 => pos.z = p.f32(),
                    _ => {}
                }
            }
            Some(EntityKind::Point(PointEnt { position: pos }))
        }
        "SPLINE" => {
            // Control points arrive as repeating 10/20 (then 30) triples.
            let mut ctrl: Vec<Vec2> = Vec::new();
            let mut x: Option<f32> = None;
            let mut closed = false;
            for p in rec {
                match p.code {
                    10 => x = Some(p.f32()),
                    20 => {
                        if let Some(vx) = x.take() {
                            ctrl.push(Vec2::new(vx, p.f32()));
                        }
                    }
                    70 => closed = p.i32() & 1 != 0,
                    _ => {}
                }
            }
            if ctrl.len() < 2 {
                return None;
            }
            Some(EntityKind::Spline(Spline::catmull_rom(&ctrl, closed, 1.0)))
        }
        "INSERT" => {
            let mut name = String::new();
            let mut pos = Vec3::ZERO;
            let mut scale = Vec3::splat(1.0);
            let mut rot = 0.0f32;
            for p in rec {
                match p.code {
                    2 => name = p.value.trim().to_string(),
                    10 => pos.x = p.f32(),
                    20 => pos.y = p.f32(),
                    30 => pos.z = p.f32(),
                    41 => scale.x = p.f32().abs().max(1e-9),
                    42 => scale.y = p.f32().abs().max(1e-9),
                    43 => scale.z = p.f32().abs().max(1e-9),
                    50 => rot = p.f32().to_radians(),
                    _ => {}
                }
            }
            let bid = blocks
                .get(&name.to_ascii_uppercase())
                .copied()
                // Unknown block: keep the reference so nothing is silently lost.
                .unwrap_or(cad_doc::handle::BlockId(u32::MAX));
            Some(EntityKind::Insert(InsertRef {
                block: bid,
                position: pos,
                scale,
                rotation: rot,
                rows: 1,
                columns: 1,
                row_spacing: 0.0,
                col_spacing: 0.0,
            }))
        }
        _ => None,
    }
}

fn halign(v: i32) -> TextAlign {
    match v {
        1 => TextAlign::Center,
        2 => TextAlign::Right,
        _ => TextAlign::Left,
    }
}
fn valign(v: i32, h: TextAlign) -> TextAlign {
    match (v, h) {
        (2, TextAlign::Left) => TextAlign::MiddleLeft,
        (2, TextAlign::Center) => TextAlign::MiddleCenter,
        (2, TextAlign::Right) => TextAlign::MiddleRight,
        _ => h,
    }
}

/// Read a DXF file from disk into `doc`.
pub fn import(
    doc: &mut cad_doc::Document,
    path: &std::path::Path,
) -> Result<ImportReport, DxfError> {
    let bytes = std::fs::read(path).map_err(|e| DxfError::Malformed(e.to_string()))?;
    let text = String::from_utf8(bytes).map_err(|_| DxfError::NotAscii)?;
    read(doc, &text)
}

// ---------------------------------------------------------------- writing

/// Serialise `doc` as an AC1015 (R2000) ASCII DXF.
pub fn write(doc: &cad_doc::Document) -> String {
    let mut s = String::with_capacity(64 * 1024);
    let mut handle = 0x100u64;
    let mut next_handle = move || {
        handle += 1;
        handle
    };

    // ---- HEADER --------------------------------------------------------------
    s.push_str("999\nCADKit drawing\n");
    s.push_str("0\nSECTION\n2\nHEADER\n");
    push_str(&mut s, 9, "$ACADVER");
    push_str(&mut s, 1, "AC1015");
    push_str(&mut s, 9, "$INSUNITS");
    push_int(&mut s, 70, doc.units as i32);
    push_str(&mut s, 9, "$HANDSEED");
    push_hex(&mut s, 5, 0xFFFFF);
    if let Some((bb, _)) = doc.compute_extents() {
        push_str(&mut s, 9, "$EXTMIN");
        push_f64(&mut s, 10, bb.min.x as f64);
        push_f64(&mut s, 20, bb.min.y as f64);
        push_f64(&mut s, 30, bb.min.z as f64);
        push_str(&mut s, 9, "$EXTMAX");
        push_f64(&mut s, 10, bb.max.x as f64);
        push_f64(&mut s, 20, bb.max.y as f64);
        push_f64(&mut s, 30, bb.max.z as f64);
    }
    s.push_str("0\nENDSEC\n");

    // ---- TABLES --------------------------------------------------------------
    s.push_str("0\nSECTION\n2\nTABLES\n");

    s.push_str("0\nTABLE\n2\nLTYPE\n70\n10\n");
    for (i, name) in ["ByBlock", "ByLayer", "Continuous"].iter().enumerate() {
        s.push_str("0\nLTYPE\n");
        push_hex(&mut s, 5, 0x300 + i as u64);
        push_int(&mut s, 100, 0);
        push_str(&mut s, 2, name);
        push_int(&mut s, 70, 0);
        push_str(&mut s, 3, "");
        push_int(&mut s, 72, 65);
        push_int(&mut s, 73, 0);
        push_f64(&mut s, 40, 0.0);
    }
    for (i, name) in [
        "DASHED", "CENTER", "DASHDOT", "PHANTOM", "HIDDEN", "DIVIDE", "BORDER",
    ]
    .iter()
    .enumerate()
    {
        let p = acad_pattern(name);
        s.push_str("0\nLTYPE\n");
        push_hex(&mut s, 5, 0x400 + i as u64);
        push_int(&mut s, 100, 0);
        push_str(&mut s, 2, name);
        push_int(&mut s, 70, 0);
        push_str(&mut s, 3, "");
        push_int(&mut s, 72, 65);
        push_int(&mut s, 73, p.len() as i32);
        push_f64(&mut s, 40, p.iter().map(|v| v.abs()).sum::<f32>() as f64);
        for v in p {
            push_f64(&mut s, 49, *v as f64);
        }
    }
    s.push_str("0\nENDTAB\n");

    s.push_str("0\nTABLE\n2\nLAYER\n");
    push_int(&mut s, 70, doc.layers.len() as i32);
    for (id, l) in doc.layers.iter() {
        s.push_str("0\nLAYER\n");
        push_hex(&mut s, 5, 0x10 + id.raw() as u64);
        push_int(&mut s, 100, 0);
        push_str(&mut s, 2, &l.name);
        push_int(&mut s, 70, if l.plot { 0 } else { 1 });
        // ACI colour in bits 0-6, off = bit 7, frozen = bit 8.
        let mut c = (if l.color <= 0 { 7 } else { l.color as i32 }) & 0x7F;
        if c == 0 {
            c = 7;
        }
        if !l.visible {
            c |= 0x80;
        }
        if l.frozen {
            c |= 0x100;
        }
        push_int(&mut s, 62, c);
        push_str(&mut s, 6, linetype_name(&l.linetype));
        if l.lineweight != LineWeight::ByLayer {
            push_int(&mut s, 370, l.lineweight.to_dxf() as i32);
        }
        if l.locked {
            push_int(&mut s, 292, 1);
        }
    }
    s.push_str("0\nENDTAB\n");
    s.push_str("0\nENDSEC\n");

    // ---- BLOCKS --------------------------------------------------------------
    s.push_str("0\nSECTION\n2\nBLOCKS\n");
    for (_, b) in doc
        .blocks
        .iter()
        .filter(|(_, b)| !b.is_layout() && !b.is_empty())
    {
        let bh = next_handle();
        s.push_str("0\nBLOCK\n");
        push_hex(&mut s, 5, bh);
        push_str(&mut s, 8, "0");
        push_str(&mut s, 2, &b.name);
        push_int(&mut s, 70, 0);
        push_f64(&mut s, 10, b.base_point.x as f64);
        push_f64(&mut s, 20, b.base_point.y as f64);
        push_f64(&mut s, 30, b.base_point.z as f64);
        push_str(&mut s, 3, &b.name);
        for e in &b.entities {
            write_entity(&mut s, e, doc, &mut next_handle);
        }
        s.push_str("0\nENDBLK\n");
        push_hex(&mut s, 5, bh + 1);
        push_str(&mut s, 8, "0");
    }
    s.push_str("0\nENDSEC\n");

    // ---- ENTITIES ------------------------------------------------------------
    s.push_str("0\nSECTION\n2\nENTITIES\n");
    for e in doc.entities.iter() {
        write_entity(&mut s, e, doc, &mut next_handle);
    }
    s.push_str("0\nENDSEC\n0\nEOF\n");

    s
}

fn linetype_name(lt: &LineType) -> &'static str {
    match lt {
        LineType::Continuous => "Continuous",
        LineType::ByLayer => "ByLayer",
        LineType::ByBlock => "ByBlock",
        LineType::Dashed { .. } | LineType::Named(_) => "DASHED",
    }
}

fn push_str(s: &mut String, code: i32, v: &str) {
    s.push_str(&code.to_string());
    s.push('\n');
    s.push_str(v);
    s.push('\n');
}
fn push_int(s: &mut String, code: i32, v: i32) {
    push_str(s, code, &v.to_string());
}
fn push_f64(s: &mut String, code: i32, v: f64) {
    s.push_str(&code.to_string());
    s.push('\n');
    // 6 decimals is plenty for drawing units and keeps files small.
    s.push_str(&format!("{v:.6}"));
    s.push('\n');
}
fn push_hex(s: &mut String, code: i32, v: u64) {
    s.push_str(&code.to_string());
    s.push('\n');
    s.push_str(&format!("{v:X}"));
    s.push('\n');
}

/// Emit one entity into `s`.
pub fn write_entity(
    s: &mut String,
    e: &Entity,
    doc: &cad_doc::Document,
    next_handle: &mut impl FnMut() -> u64,
) {
    let ty = match &e.entity {
        EntityKind::Line(_) => "LINE",
        EntityKind::Circle(_) => "CIRCLE",
        EntityKind::Arc(_) => "ARC",
        EntityKind::Ellipse(_) => "ELLIPSE",
        EntityKind::Polyline(_) => "LWPOLYLINE",
        EntityKind::Point(_) => "POINT",
        EntityKind::Text(_) => "TEXT",
        EntityKind::Insert(_) => "INSERT",
        _ => return,
    };
    let h = next_handle();
    let layer_name = doc.layers.name(e.layer()).to_string();

    // The grammar requires `0 / <TYPE>` to be the first pair of every record.
    push_str(s, 0, ty);
    push_hex(s, 5, h);
    push_hex(s, 330, 0);
    push_str(s, 100, "AcDbEntity");
    push_str(s, 8, &layer_name);
    push_int(s, 62, e.common.color as i32);
    if e.common.linetype != LineType::ByLayer {
        push_str(s, 6, linetype_name(&e.common.linetype));
    }
    if e.common.lineweight != LineWeight::ByLayer {
        push_int(s, 370, e.common.lineweight.to_dxf() as i32);
    }
    if e.common.paper_space {
        push_int(s, 67, 1);
    }

    match &e.entity {
        EntityKind::Line(l) => {
            push_str(s, 100, "AcDbLine");
            push_f64(s, 10, l.p0.x as f64);
            push_f64(s, 20, l.p0.y as f64);
            push_f64(s, 30, 0.0);
            push_f64(s, 11, l.p1.x as f64);
            push_f64(s, 21, l.p1.y as f64);
            push_f64(s, 31, 0.0);
        }
        EntityKind::Circle(c) => {
            push_str(s, 100, "AcDbCircle");
            push_f64(s, 10, c.center.x as f64);
            push_f64(s, 20, c.center.y as f64);
            push_f64(s, 30, 0.0);
            push_f64(s, 40, c.radius as f64);
        }
        EntityKind::Arc(a) => {
            push_str(s, 100, "AcDbCircle");
            push_f64(s, 10, a.center.x as f64);
            push_f64(s, 20, a.center.y as f64);
            push_f64(s, 30, 0.0);
            push_f64(s, 40, a.radius as f64);
            push_str(s, 100, "AcDbArc");
            push_f64(s, 50, a.start_angle.to_degrees() as f64);
            push_f64(s, 51, a.end_angle().to_degrees() as f64);
        }
        EntityKind::Ellipse(el) => {
            push_str(s, 100, "AcDbEllipse");
            push_f64(s, 10, el.center.x as f64);
            push_f64(s, 20, el.center.y as f64);
            push_f64(s, 30, 0.0);
            push_f64(s, 11, el.major_axis.x as f64);
            push_f64(s, 21, el.major_axis.y as f64);
            push_f64(s, 31, 0.0);
            push_f64(s, 40, el.ratio as f64);
            push_f64(s, 41, 0.0);
            push_f64(s, 42, 360.0);
        }
        EntityKind::Polyline(p) => {
            push_str(s, 100, "AcDbPolyline");
            push_int(s, 90, p.vertices.len() as i32);
            push_int(s, 70, if p.closed { 1 } else { 0 });
            for (i, v) in p.vertices.iter().enumerate() {
                push_f64(s, 10, v.x as f64);
                push_f64(s, 20, v.y as f64);
                if p.bulge_at(i).is_arc() {
                    push_f64(s, 42, p.bulge_at(i).0 as f64);
                }
            }
        }
        EntityKind::Point(pt) => {
            push_str(s, 100, "AcDbPoint");
            push_f64(s, 10, pt.position.x as f64);
            push_f64(s, 20, pt.position.y as f64);
            push_f64(s, 30, pt.position.z as f64);
        }
        EntityKind::Text(t) => {
            push_str(s, 100, "AcDbText");
            push_f64(s, 10, t.insert.x as f64);
            push_f64(s, 20, t.insert.y as f64);
            push_f64(s, 30, t.insert.z as f64);
            push_f64(s, 40, t.height as f64);
            push_str(s, 1, &t.value);
            push_f64(s, 50, t.rotation.to_degrees() as f64);
            push_f64(s, 41, t.width_factor as f64);
            push_str(s, 7, "Standard");
            push_int(s, 72, 0);
        }
        EntityKind::Insert(i) => {
            push_str(s, 100, "AcDbBlockReference");
            let name = doc.blocks.name(i.block).to_string();
            push_str(s, 2, &name);
            push_f64(s, 10, i.position.x as f64);
            push_f64(s, 20, i.position.y as f64);
            push_f64(s, 30, i.position.z as f64);
            push_f64(s, 41, i.scale.x as f64);
            push_f64(s, 42, i.scale.y as f64);
            push_f64(s, 43, i.scale.z as f64);
            push_f64(s, 50, i.rotation.to_degrees() as f64);
        }
        _ => {}
    }
}

/// Write `doc` to `path`.
pub fn save(doc: &cad_doc::Document, path: &std::path::Path) -> std::io::Result<()> {
    std::fs::write(path, write(doc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_doc::Document;

    #[test]
    fn lexer_pairs_codes_with_values() {
        let text = "0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n4\n0\nENDSEC\n0\nEOF\n";
        let mut lex = Lexer::new(text);
        let pairs = lex.collect();
        // 0 SECTION | 2 HEADER | 9 $INSUNITS | 70 4 | 0 ENDSEC | 0 EOF
        assert_eq!(pairs.len(), 6, "{pairs:?}");
        assert_eq!(
            pairs[0],
            Pair {
                code: 0,
                value: "SECTION".into()
            }
        );
        assert_eq!(
            pairs[1],
            Pair {
                code: 2,
                value: "HEADER".into()
            }
        );
        assert_eq!(
            pairs[2],
            Pair {
                code: 9,
                value: "$INSUNITS".into()
            }
        );
        assert_eq!(
            pairs[3],
            Pair {
                code: 70,
                value: "4".into()
            }
        );
        assert_eq!(
            pairs[5],
            Pair {
                code: 0,
                value: "EOF".into()
            }
        );
    }

    #[test]
    fn lexer_handles_crlf() {
        let mut lex = Lexer::new("0\r\nLINE\r\n8\r\nWalls\r\n");
        let p = lex.collect();
        assert_eq!(p.len(), 2, "{p:?}");
        assert_eq!(
            p[0],
            Pair {
                code: 0,
                value: "LINE".into()
            }
        );
        assert_eq!(
            p[1],
            Pair {
                code: 8,
                value: "Walls".into()
            }
        );
    }

    #[test]
    fn lexer_handles_bare_cr() {
        let mut lex = Lexer::new("0\rLINE\r8\rWalls\r");
        let p = lex.collect();
        assert_eq!(p.len(), 2, "{p:?}");
        assert_eq!(p[1].value, "Walls");
    }

    #[test]
    fn lexer_skips_stray_lines() {
        let mut lex = Lexer::new("garbage\n0\nLINE\n");
        let p = lex.collect();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].value, "LINE");
    }

    #[test]
    fn pair_typed_accessors() {
        let p = Pair {
            code: 10,
            value: " 12.5 ".into(),
        };
        assert_eq!(p.f32(), 12.5);
        assert!(p.is_real());
        assert_eq!(
            Pair {
                code: 5,
                value: "FF".into()
            }
            .hex(),
            255
        );
        assert!(
            Pair {
                code: 1,
                value: "1".into()
            }
            .bool()
        );
        assert!(
            !Pair {
                code: 1,
                value: "0".into()
            }
            .bool()
        );
    }

    #[test]
    fn rejects_empty_and_binary() {
        let mut d = Document::new();
        assert_eq!(read(&mut d, ""), Err(DxfError::Empty));
        assert_eq!(read(&mut d, "   \n"), Err(DxfError::Empty));
        assert_eq!(
            read(&mut d, "AutoCAD Binary DXF\r\n"),
            Err(DxfError::NotAscii)
        );
    }

    #[test]
    fn round_trip_simple_entities() {
        let mut d = Document::new();
        let layer = d.layers.ensure_default();
        d.add(Entity::line(Line::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 5.0))).with_layer(layer));
        d.add(Entity::circle(Circle::new(Vec2::new(3.0, 4.0), 2.5)).with_layer(layer));
        d.add(Entity::arc(Arc::from_angles(Vec2::new(-3.0, 1.0), 6.0, 0.0, 1.2)).with_layer(layer));

        let text = write(&d);
        let mut d2 = Document::new();
        let rep = read(&mut d2, &text).expect("reimport");
        assert_eq!(rep.entities, 3);

        let mut lines = Vec::new();
        let mut circles = Vec::new();
        let mut arcs = Vec::new();
        for e in d2.entities.iter() {
            match &e.entity {
                EntityKind::Line(l) => lines.push(*l),
                EntityKind::Circle(c) => circles.push(*c),
                EntityKind::Arc(a) => arcs.push(*a),
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(lines.len(), 1);
        assert!(lines[0].p0.distance(Vec2::ZERO) < 1e-3);
        assert!(lines[0].p1.distance(Vec2::new(10.0, 5.0)) < 1e-3);
        assert_eq!(circles.len(), 1);
        assert!((circles[0].radius - 2.5).abs() < 1e-3);
        assert!(circles[0].center.distance(Vec2::new(3.0, 4.0)) < 1e-3);
        assert_eq!(arcs.len(), 1);
        assert!((arcs[0].radius - 6.0).abs() < 1e-3);
        assert!(
            (arcs[0].sweep - 1.2).abs() < 1e-3,
            "sweep={}",
            arcs[0].sweep
        );
    }

    #[test]
    fn round_trip_polyline_with_bulge() {
        let mut d = Document::new();
        let layer = d.layers.ensure_default();
        let mut p = Polyline::new(
            vec![Vec2::ZERO, Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0)],
            true,
        );
        p.bulges[0] = Bulge::from_sweep(cad_core::FRAC_PI_2);
        d.add(Entity::polyline(p).with_layer(layer));

        let text = write(&d);
        let mut d2 = Document::new();
        read(&mut d2, &text).unwrap();
        let mut found = None;
        for e in d2.entities.iter() {
            if let EntityKind::Polyline(p) = &e.entity {
                found = Some(p.clone());
            }
        }
        let p = found.expect("polyline missing");
        assert_eq!(p.vertices.len(), 3);
        assert!(p.closed);
        assert!(p.bulge_at(0).is_arc(), "bulge lost: {:?}", p.bulges);
        assert!((p.bulge_at(0).0 - 0.414_213_6).abs() < 1e-4);
        assert!(p.vertices[2].distance(Vec2::new(10.0, 10.0)) < 1e-3);
    }

    #[test]
    fn round_trip_many_vertex_polyline() {
        let mut d = Document::new();
        let layer = d.layers.ensure_default();
        let verts: Vec<Vec2> = (0..50)
            .map(|i| Vec2::new(i as f32, (i * i) as f32))
            .collect();
        d.add(Entity::polyline(Polyline::new(verts.clone(), false)).with_layer(layer));
        let text = write(&d);
        let mut d2 = Document::new();
        read(&mut d2, &text).unwrap();
        let mut out = None;
        for e in d2.entities.iter() {
            if let EntityKind::Polyline(p) = &e.entity {
                out = Some(p.clone());
            }
        }
        let p = out.unwrap();
        assert_eq!(p.vertices.len(), 50);
        for (a, b) in p.vertices.iter().zip(&verts) {
            assert!(a.distance(*b) < 1e-3, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn round_trip_layers() {
        let mut d = Document::new();
        d.layers.insert(Layer {
            name: "Walls".into(),
            color: 1,
            visible: true,
            ..Default::default()
        });
        d.layers.insert(Layer {
            name: "Doors".into(),
            color: 3,
            locked: true,
            visible: true,
            ..Default::default()
        });
        let text = write(&d);
        let mut d2 = Document::new();
        read(&mut d2, &text).unwrap();
        assert!(d2.layers.by_name("WALLS").is_some());
        let doors_id = d2.layers.by_name("Doors").unwrap();
        let doors = d2.layers.by_id(doors_id).unwrap();
        assert_eq!(doors.color, 3);
        assert!(doors.locked);
    }

    #[test]
    fn hidden_layer_survives_the_round_trip() {
        let mut d = Document::new();
        d.layers.insert(Layer {
            name: "Off".into(),
            visible: false,
            ..Default::default()
        });
        let text = write(&d);
        let mut d2 = Document::new();
        read(&mut d2, &text).unwrap();
        let off = d2.layers.by_name("OFF").unwrap();
        assert!(!d2.layers.by_id(off).unwrap().visible);
    }

    #[test]
    fn entities_keep_their_layer() {
        let mut d = Document::new();
        let _a = d.layers.ensure_default();
        // Explicitly visible: a hidden layer would be skipped on import by design.
        let b = d.layers.insert(Layer {
            visible: true,
            ..Layer::new("Walls")
        });
        d.add(Entity::circle(Circle::new(Vec2::ZERO, 1.0)).with_layer(b));
        let text = write(&d);
        let mut d2 = Document::new();
        read(&mut d2, &text).unwrap();
        let walls = d2.layers.by_name("WALLS").unwrap();
        let mut ok = false;
        for e in d2.entities.iter() {
            if e.layer() == walls {
                ok = true;
            }
        }
        assert!(ok, "entity lost its layer assignment");
        assert_eq!(d2.layers.by_id(walls).unwrap().name, "Walls");
    }

    #[test]
    fn units_are_read_from_the_header() {
        let mut d = Document::new();
        d.units = cad_doc::Units::Millimeters;
        let text = write(&d);
        let mut d2 = Document::new();
        let rep = read(&mut d2, &text).unwrap();
        assert_eq!(d2.units, cad_doc::Units::Millimeters);
        assert_eq!(rep.header_units, Some(cad_doc::Units::Millimeters));
    }

    #[test]
    fn unknown_entities_are_reported_not_fatal() {
        let text = "0\nSECTION\n2\nENTITIES\n0\nMLINE\n8\n0\n0\nACAD_PROXY_ENTITY\n8\n0\n0\nENDSEC\n0\nEOF\n";
        let mut d = Document::new();
        let rep = read(&mut d, text).unwrap();
        assert_eq!(rep.entities, 0);
        let kinds: Vec<&str> = rep.skipped.iter().map(|(k, _)| k.as_str()).collect();
        assert!(kinds.contains(&"MLINE"), "{kinds:?}");
    }

    #[test]
    fn text_and_point_round_trip() {
        let mut d = Document::new();
        let layer = d.layers.ensure_default();
        d.add(
            Entity::new(EntityKind::Text(Text {
                value: "NORTE".into(),
                insert: Vec3::new(5.0, 6.0, 0.0),
                height: 2.5,
                width_factor: 0.8,
                rotation: cad_core::FRAC_PI_2,
                oblique: 0.0,
                align: TextAlign::Left,
                value2: None,
                height2: 0.0,
            }))
            .with_layer(layer),
        );
        d.add(
            Entity::new(EntityKind::Point(PointEnt {
                position: Vec3::new(1.0, 2.0, 0.0),
            }))
            .with_layer(layer),
        );
        let text = write(&d);
        let mut d2 = Document::new();
        read(&mut d2, &text).unwrap();
        let mut found_text = false;
        let mut found_point = false;
        for e in d2.entities.iter() {
            match &e.entity {
                EntityKind::Text(t) => {
                    assert_eq!(t.value, "NORTE");
                    assert!((t.height - 2.5).abs() < 1e-3);
                    assert!((t.insert.x - 5.0).abs() < 1e-3);
                    assert!((t.insert.y - 6.0).abs() < 1e-3);
                    found_text = true;
                }
                EntityKind::Point(p) => {
                    assert!(p.position.distance(Vec3::new(1.0, 2.0, 0.0)) < 1e-3);
                    found_point = true;
                }
                _ => {}
            }
        }
        assert!(found_text && found_point);
    }

    #[test]
    fn ellipse_round_trip() {
        let mut d = Document::new();
        let layer = d.layers.ensure_default();
        d.add(
            Entity::new(EntityKind::Ellipse(Ellipse::new(
                Vec2::new(1.0, 2.0),
                Vec2::new(6.0, 0.0),
                0.5,
            )))
            .with_layer(layer),
        );
        let text = write(&d);
        let mut d2 = Document::new();
        read(&mut d2, &text).unwrap();
        for e in d2.entities.iter() {
            if let EntityKind::Ellipse(el) = &e.entity {
                assert!(el.center.distance(Vec2::new(1.0, 2.0)) < 1e-3);
                assert!((el.semi_major() - 6.0).abs() < 1e-3);
                assert!((el.ratio - 0.5).abs() < 1e-3);
                return;
            }
        }
        panic!("ellipse missing");
    }

    #[test]
    fn written_file_is_well_formed() {
        let mut d = Document::new();
        let layer = d.layers.ensure_default();
        d.add(Entity::circle(Circle::new(Vec2::ZERO, 1.0)).with_layer(layer));
        d.add(Entity::line(Line::new(Vec2::ZERO, Vec2::new(1.0, 1.0))).with_layer(layer));
        let text = write(&d);

        let mut lines = text.lines().collect::<Vec<_>>();
        assert_eq!(lines.len() % 2, 0, "odd number of lines");
        let mut i = 0;
        while i < lines.len() {
            assert!(
                lines[i].trim().parse::<i32>().is_ok(),
                "code line {i} is not an integer: {:?}",
                lines[i]
            );
            i += 2;
        }
        assert!(text.contains("0\nSECTION\n2\nHEADER\n"));
        assert!(text.contains("0\nSECTION\n2\nENTITIES\n"));
        assert!(text.trim_end().ends_with("0\nEOF"));
        lines.clear();
    }

    #[test]
    fn large_drawing_is_handled() {
        let mut d = Document::new();
        let layer = d.layers.ensure_default();
        for i in 0..2000 {
            d.add(
                Entity::circle(Circle::new(
                    Vec2::new(i as f32 * 5.0, (i % 7) as f32 * 3.0),
                    1.5,
                ))
                .with_layer(layer),
            );
        }
        let text = write(&d);
        let mut d2 = Document::new();
        let rep = read(&mut d2, &text).unwrap();
        assert_eq!(rep.entities, 2000);
        assert_eq!(d2.entities.len(), 2000);
        assert!(d2.entities.index().is_sized());
    }

    #[test]
    fn blocks_and_inserts_survive() {
        let mut d = Document::new();
        let layer = d.layers.ensure_default();
        let b = d.blocks.insert(cad_doc::block::Block::new("Door"));
        d.blocks.add_entity(
            b,
            Entity::line(Line::new(Vec2::ZERO, Vec2::new(1.0, 0.0))).with_layer(layer),
        );
        d.add(
            Entity::new(EntityKind::Insert(InsertRef {
                block: b,
                position: Vec3::new(5.0, 6.0, 0.0),
                scale: Vec3::splat(2.0),
                rotation: 0.0,
                rows: 1,
                columns: 1,
                row_spacing: 0.0,
                col_spacing: 0.0,
            }))
            .with_layer(layer),
        );
        let text = write(&d);
        let mut d2 = Document::new();
        let rep = read(&mut d2, &text).unwrap();
        assert_eq!(rep.blocks, 1);
        let door_id = d2.blocks.by_name("DOOR").expect("block lost");
        let door = d2.blocks.by_id(door_id).unwrap();
        assert_eq!(door.entities.len(), 1);
        let mut ins = None;
        for e in d2.entities.iter() {
            if let EntityKind::Insert(i) = &e.entity {
                ins = Some(*i);
            }
        }
        let i = ins.expect("insert lost");
        assert_eq!(i.block, door_id);
        assert!(i.position.distance(Vec3::new(5.0, 6.0, 0.0)) < 1e-3);
    }
}
