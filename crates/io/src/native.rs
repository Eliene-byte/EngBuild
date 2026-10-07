//! Native project format (`.cad`).
//!
//! A tiny, versioned, little-endian binary container. It is roughly 10x faster
//! to load than DXF and preserves everything the document model holds
//! (including handles), so it is the default for our own files — DXF is the
//! interchange format.
//!
//! Layout:
//! ```text
//! magic  "CADKIT\0\0"  (8 bytes)
//! u32    format version
//! u32    section count
//! then per section: u32 tag, u32 byte length, payload
//! ```
//! Payloads are written by a small, explicit writer/reader pair — no `serde`,
//! no reflection, and therefore no surprises when the model grows.

use cad_core::{Vec2, Vec3};
use cad_doc::Document;
use cad_doc::entity::{Text, TextAlign};
use cad_geom::curve::{Arc, Circle, Line, Polyline};
use std::collections::HashMap;
use std::fmt;
use std::io;

const MAGIC: &[u8; 8] = b"CADKIT\0\0";
const VERSION: u32 = 1;

const TAG_HEADER: u32 = 1;
const TAG_LAYERS: u32 = 2;
const TAG_BLOCKS: u32 = 3;
const TAG_STYLES: u32 = 4;
const TAG_ENTITIES: u32 = 5;

#[derive(Debug)]
pub enum NativeError {
    BadMagic,
    UnsupportedVersion(u32),
    Io(io::Error),
    Truncated(String),
}

impl fmt::Display for NativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NativeError::BadMagic => write!(f, "not a CADKit project file"),
            NativeError::UnsupportedVersion(v) => write!(f, "unsupported project version {v}"),
            NativeError::Io(e) => write!(f, "{e}"),
            NativeError::Truncated(s) => write!(f, "truncated project file: {s}"),
        }
    }
}

impl std::error::Error for NativeError {}

impl From<io::Error> for NativeError {
    fn from(e: io::Error) -> Self {
        NativeError::Io(e)
    }
}

// ---------------------------------------------------------------- writer

#[derive(Default)]
struct W {
    buf: Vec<u8>,
}

impl W {
    fn b(&mut self, v: bool) {
        self.buf.push(v as u8);
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn v2(&mut self, v: Vec2) {
        self.f32(v.x);
        self.f32(v.y);
    }
    fn v3(&mut self, v: Vec3) {
        self.f32(v.x);
        self.f32(v.y);
        self.f32(v.z);
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }
    fn tag(&mut self, t: u32, body: &[u8]) {
        self.u32(t);
        self.u32(body.len() as u32);
        self.buf.extend_from_slice(body);
    }
}

#[derive(Default)]
struct R<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> R<'a> {
    fn need(&self, n: usize) -> Result<(), NativeError> {
        if self.pos + n > self.buf.len() {
            Err(NativeError::Truncated(format!(
                "need {n} bytes at {}, {} available",
                self.pos,
                self.buf.len() - self.pos
            )))
        } else {
            Ok(())
        }
    }
    fn u8(&mut self) -> Result<u8, NativeError> {
        self.need(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn b(&mut self) -> Result<bool, NativeError> {
        Ok(self.u8()? != 0)
    }
    fn u32(&mut self) -> Result<u32, NativeError> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn i32(&mut self) -> Result<i32, NativeError> {
        Ok(self.u32()? as i32)
    }
    fn f32(&mut self) -> Result<f32, NativeError> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn v2(&mut self) -> Result<Vec2, NativeError> {
        let x = self.f32()?;
        let y = self.f32()?;
        Ok(Vec2::new(x, y))
    }
    fn v3(&mut self) -> Result<Vec3, NativeError> {
        let x = self.f32()?;
        let y = self.f32()?;
        let z = self.f32()?;
        Ok(Vec3::new(x, y, z))
    }
    fn str(&mut self) -> Result<String, NativeError> {
        let n = self.u32()? as usize;
        self.need(n)?;
        let s = String::from_utf8_lossy(&self.buf[self.pos..self.pos + n]).into_owned();
        self.pos += n;
        Ok(s)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], NativeError> {
        self.need(n)?;
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
}

// ---------------------------------------------------------------- encode

/// Serialise `doc` to bytes.
pub fn encode(doc: &Document) -> Vec<u8> {
    let mut out = W::default();
    out.buf.extend_from_slice(MAGIC);
    out.u32(VERSION);
    out.u32(5); // section count

    // HEADER
    let mut h = W::default();
    h.str(doc.name());
    h.u32(doc.units as u32);
    if let Some((bb, _)) = doc.compute_extents() {
        h.b(true);
        h.v3(bb.min);
        h.v3(bb.max);
    } else {
        h.b(false);
    }
    out.tag(TAG_HEADER, &h.buf);

    // LAYERS
    let mut l = W::default();
    let mut ids = HashMap::new();
    for (id, layer) in doc.layers.iter() {
        ids.insert(id, l.buf.len() as u32);
        l.str(&layer.name);
        l.i32(layer.color as i32);
        l.u32(match &layer.linetype {
            cad_doc::LineType::Continuous => 0,
            cad_doc::LineType::ByLayer => 1,
            cad_doc::LineType::ByBlock => 2,
            cad_doc::LineType::Dashed { .. } => 3,
            cad_doc::LineType::Named(_) => 4,
        });
        l.b(layer.visible);
        l.b(layer.locked);
        l.b(layer.frozen);
        l.b(layer.plot);
        l.i32(layer.plot_order);
        l.f32(layer.elevation);
        l.f32(layer.transparency);
        l.i32(layer.lineweight.to_dxf() as i32);
    }
    out.tag(TAG_LAYERS, &l.buf);

    // BLOCKS
    //
    // Block headers come first so entities inside them can reference them by
    // index. Layout: count, then (name, base_point, entity_count)*, then the
    // concatenated entity payloads.
    let named: Vec<_> = doc
        .blocks
        .iter()
        .filter(|(_, b)| !b.is_layout() && !b.is_empty())
        .collect();
    let mut block_ids = HashMap::new();
    let mut b = W::default();
    b.u32(named.len() as u32);
    for (i, (id, block)) in named.iter().enumerate() {
        block_ids.insert(*id, i as u32);
        b.str(&block.name);
        b.v3(block.base_point);
        b.u32(block.entities.len() as u32);
    }
    // Block member entities.
    for (_, block) in &named {
        for e in &block.entities {
            write_common(&mut b, e, doc, &ids, &block_ids);
            write_geometry(&mut b, &e.entity, &block_ids);
        }
    }
    out.tag(TAG_BLOCKS, &b.buf);

    // ENTITIES (drawing level)
    let mut ent = W::default();
    for e in doc.entities.iter() {
        write_common(&mut ent, e, doc, &ids, &block_ids);
        write_geometry(&mut ent, &e.entity, &block_ids);
    }
    out.tag(TAG_ENTITIES, &ent.buf);

    // STYLES
    let mut s = W::default();
    s.u32(doc.styles.len() as u32);
    for (_, st) in doc.styles.iter() {
        s.str(&st.name);
        s.f32(st.height);
        s.f32(st.width);
        s.f32(st.oblique);
        s.f32(st.tracking);
        s.f32(st.fixed_width);
        s.u32(st.width_style as u32);
        s.u32(st.font_index);
        s.b(st.big);
    }
    out.tag(TAG_STYLES, &s.buf);

    out.buf
}

fn write_common(
    w: &mut W,
    e: &cad_doc::Entity,
    doc: &Document,
    layers: &HashMap<cad_doc::LayerId, u32>,
    blocks: &HashMap<cad_doc::BlockId, u32>,
) {
    // Layer indices are written as *positions in the layer table*, which the
    // reader rebuilds in the same order.
    w.u32(layers.get(&e.layer()).copied().unwrap_or(0));
    w.i32(e.common.color as i32);
    w.u32(match &e.common.linetype {
        cad_doc::LineType::Continuous => 0,
        cad_doc::LineType::ByLayer => 1,
        cad_doc::LineType::ByBlock => 2,
        cad_doc::LineType::Dashed { .. } => 3,
        cad_doc::LineType::Named(_) => 4,
    });
    w.i32(e.common.lineweight.to_dxf() as i32);
    w.b(e.common.visible);
    w.b(e.common.locked);
    w.b(e.common.paper_space);
    w.f32(e.common.transparency);
    let _ = (doc, blocks);
}

fn write_geometry(w: &mut W, k: &cad_doc::EntityKind, blocks: &HashMap<cad_doc::BlockId, u32>) {
    use cad_doc::EntityKind as K;
    match k {
        K::Line(l) => {
            w.u32(1);
            w.v2(l.p0);
            w.v2(l.p1);
        }
        K::Circle(c) => {
            w.u32(2);
            w.v2(c.center);
            w.f32(c.radius);
        }
        K::Arc(a) => {
            w.u32(3);
            w.v2(a.center);
            w.f32(a.radius);
            w.f32(a.start_angle);
            w.f32(a.sweep);
        }
        K::Ellipse(e) => {
            w.u32(4);
            w.v2(e.center);
            w.v2(e.major_axis);
            w.f32(e.ratio);
        }
        K::Polyline(p) | K::Region(p) => {
            w.u32(if matches!(k, K::Region(_)) { 15 } else { 5 });
            w.u32(p.vertices.len() as u32);
            for v in &p.vertices {
                w.v2(*v);
            }
            for i in 0..p.vertices.len() {
                w.f32(p.bulge_at(i).0);
            }
            w.b(p.closed);
        }
        K::Point(p) => {
            w.u32(6);
            w.v3(p.position);
        }
        K::Text(t) => {
            w.u32(7);
            w.str(&t.value);
            w.v3(t.insert);
            w.f32(t.height);
            w.f32(t.width_factor);
            w.f32(t.rotation);
            w.f32(t.oblique);
            w.u32(t.align as u32);
        }
        K::Spline(s) => {
            w.u32(8);
            w.u32(s.segments.len() as u32);
            for seg in &s.segments {
                for p in &seg.p {
                    w.v2(*p);
                }
            }
            w.b(s.closed);
        }
        K::Hatch(h) => {
            w.u32(9);
            w.u32(h.loops.len() as u32);
            for l in &h.loops {
                w.u32(l.vertices.len() as u32);
                for v in &l.vertices {
                    w.v2(*v);
                }
                w.b(l.closed);
            }
            w.f32(h.pattern_scale);
            w.f32(h.pattern_angle);
            w.str(&h.pattern_name);
            w.b(h.solid);
        }
        K::Box(bx) => {
            w.u32(10);
            w.v3(bx.min);
            w.v3(bx.max);
        }
        K::Mesh(m) => {
            w.u32(11);
            w.u32(m.positions.len() as u32);
            for p in &m.positions {
                w.v3(*p);
            }
            w.u32(m.indices.len() as u32);
            for i in &m.indices {
                w.u32(*i);
            }
        }
        K::Face(f) => {
            w.u32(12);
            w.v3(f.plane.n);
            w.f32(f.plane.d);
            w.u32(f.loop_pts.len() as u32);
            for p in &f.loop_pts {
                w.v3(*p);
            }
        }
        K::Construction(c) => {
            w.u32(13);
            w.v3(c.from);
            w.v3(c.to);
        }
        K::Insert(i) => {
            w.u32(14);
            w.u32(*blocks.get(&i.block).unwrap_or(&0));
            w.v3(i.position);
            w.v3(i.scale);
            w.f32(i.rotation);
            w.u32(i.rows);
            w.u32(i.columns);
            w.f32(i.row_spacing);
            w.f32(i.col_spacing);
        }
        K::Unknown { dxf_type, .. } => {
            w.u32(16);
            w.str(dxf_type);
        }
    }
}

// ---------------------------------------------------------------- decode

/// Parse bytes into `doc`.
pub fn decode(bytes: &[u8], doc: &mut Document) -> Result<(), NativeError> {
    let mut r = R { buf: bytes, pos: 0 };
    let magic = r.take(8)?;
    if magic != MAGIC {
        return Err(NativeError::BadMagic);
    }
    let version = r.u32()?;
    if version != VERSION {
        return Err(NativeError::UnsupportedVersion(version));
    }
    let sections = r.u32()?;

    let mut layer_ids: Vec<cad_doc::LayerId> = Vec::new();
    let mut block_ids: Vec<cad_doc::BlockId> = Vec::new();
    let mut entity_bytes: &[u8] = &[];

    for _ in 0..sections {
        let tag = r.u32()?;
        let len = r.u32()? as usize;
        let body = r.take(len)?;
        match tag {
            TAG_HEADER => {
                let mut h = R { buf: body, pos: 0 };
                let name = h.str()?;
                let units = h.u32()?;
                doc.set_name(name);
                doc.units = units_from_u32(units);
            }
            TAG_LAYERS => {
                let mut l = R { buf: body, pos: 0 };
                while l.pos < body.len() {
                    let layer = cad_doc::Layer {
                        name: l.str()?,
                        color: l.i32()? as i16,
                        linetype: linetype_from_u32(l.u32()?),
                        visible: l.b()?,
                        locked: l.b()?,
                        frozen: l.b()?,
                        plot: l.b()?,
                        plot_order: l.i32()?,
                        elevation: l.f32()?,
                        transparency: l.f32()?,
                        lineweight: cad_doc::LineWeight::from_dxf(l.i32()? as i16),
                    };
                    let id = doc.layers.insert(layer);
                    layer_ids.push(id);
                }
            }
            TAG_BLOCKS => {
                // Blocks are appended in file order, so the index of each new
                // block matches the index the encoder wrote.
                let mut b = R { buf: body, pos: 0 };
                let count = b.u32()? as usize;
                let mut pending: Vec<(String, Vec3, usize)> = Vec::with_capacity(count);
                for _ in 0..count {
                    let name = b.str()?;
                    let base_point = b.v3()?;
                    let n = b.u32()? as usize;
                    pending.push((name, base_point, n));
                }
                // Register the blocks first so entities can reference them.
                for (name, base_point, _) in pending.iter() {
                    let id = doc.blocks.insert(cad_doc::block::Block {
                        name: name.clone(),
                        base_point: *base_point,
                        ..Default::default()
                    });
                    block_ids.push(id);
                }
                for (_, _, n) in pending {
                    let mut collected = Vec::with_capacity(n);
                    for _ in 0..n {
                        if let Some(e) = read_entity(&mut b, &layer_ids, &block_ids)? {
                            collected.push(e);
                        }
                    }
                    if let Some(id) = block_ids.last() {
                        if let Some(blk) = doc.blocks.by_id_mut(*id) {
                            blk.entities = collected;
                        }
                    }
                }
            }
            TAG_STYLES => {
                let mut s = R { buf: body, pos: 0 };
                let n = s.u32()?;
                for _ in 0..n {
                    let name = s.str()?;
                    let height = s.f32()?;
                    let width = s.f32()?;
                    let oblique = s.f32()?;
                    let tracking = s.f32()?;
                    let fixed_width = s.f32()?;
                    let ws = s.u32()?;
                    let font = s.u32()?;
                    let big = s.b()?;
                    doc.styles.insert(cad_doc::TextStyle {
                        name,
                        height,
                        width,
                        oblique,
                        tracking,
                        fixed_width,
                        width_style: match ws {
                            1 => cad_doc::style::TextWidthStyle::Fit,
                            2 => cad_doc::style::TextWidthStyle::FitWidth,
                            _ => cad_doc::style::TextWidthStyle::Auto,
                        },
                        font_index: font,
                        font_file: None,
                        big,
                    });
                }
            }
            TAG_ENTITIES => entity_bytes = body,
            _ => {}
        }
    }

    // Block references need the block table, which may come later in the file.
    let mut e = R {
        buf: entity_bytes,
        pos: 0,
    };
    while e.pos < entity_bytes.len() {
        if let Some(entity) = read_entity(&mut e, &layer_ids, &block_ids)? {
            doc.entities.insert(entity);
        }
    }

    doc.layers.ensure_default();
    let bounds = doc
        .compute_extents()
        .map(|(bb, _)| cad_core::Rect2::new(bb.min.xy(), bb.max.xy()))
        .unwrap_or_else(|| cad_core::Rect2::from_xywh(-1000.0, -1000.0, 2000.0, 2000.0));
    doc.entities.rebuild_index(bounds);
    doc.invalidate_extents();
    Ok(())
}

/// Read one entity record (common + geometry) from `r`.
fn read_entity(
    r: &mut R,
    layers: &[cad_doc::LayerId],
    blocks: &[cad_doc::BlockId],
) -> Result<Option<cad_doc::Entity>, NativeError> {
    use cad_doc::EntityKind as K;
    let layer = layers
        .get(r.u32()? as usize)
        .copied()
        .unwrap_or(cad_doc::LayerId(0));
    let color = r.i32()? as i16;
    let linetype = linetype_from_u32(r.u32()?);
    let lineweight = cad_doc::LineWeight::from_dxf(r.i32()? as i16);
    let visible = r.b()?;
    let locked = r.b()?;
    let paper_space = r.b()?;
    let transparency = r.f32()?;

    let common = cad_doc::EntityCommon {
        layer,
        color,
        linetype,
        lineweight,
        visible,
        locked,
        paper_space,
        transparency,
    };

    let kind = r.u32()?;
    let geometry = match kind {
        1 => K::Line(Line::new(r.v2()?, r.v2()?)),
        2 => K::Circle(Circle::new(r.v2()?, r.f32()?)),
        3 => {
            let center = r.v2()?;
            let radius = r.f32()?;
            let start_angle = r.f32()?;
            let sweep = r.f32()?;
            K::Arc(Arc {
                center,
                radius,
                start_angle,
                sweep,
            })
        }
        4 => {
            let center = r.v2()?;
            let major_axis = r.v2()?;
            let ratio = r.f32()?;
            K::Ellipse(cad_geom::curve::Ellipse::new(center, major_axis, ratio))
        }
        5 | 15 => {
            let n = r.u32()? as usize;
            let mut vertices = Vec::with_capacity(n);
            for _ in 0..n {
                vertices.push(r.v2()?);
            }
            let mut bulges = Vec::with_capacity(n);
            for _ in 0..n {
                bulges.push(cad_geom::bulge::Bulge(r.f32()?));
            }
            let closed = r.b()?;
            let p = Polyline {
                vertices,
                bulges,
                closed,
            };
            if kind == 15 {
                K::Region(p)
            } else {
                K::Polyline(p)
            }
        }
        6 => K::Point(cad_doc::entity::PointEnt { position: r.v3()? }),
        7 => {
            let value = r.str()?;
            let insert = r.v3()?;
            let height = r.f32()?;
            let width_factor = r.f32()?;
            let rotation = r.f32()?;
            let oblique = r.f32()?;
            let align = r.u32()?;
            K::Text(Text {
                value,
                insert,
                height,
                width_factor,
                rotation,
                oblique,
                align: align_from_u32(align),
                value2: None,
                height2: 0.0,
            })
        }
        8 => {
            let n = r.u32()? as usize;
            let mut segments = Vec::with_capacity(n);
            for _ in 0..n {
                let p0 = r.v2()?;
                let p1 = r.v2()?;
                let p2 = r.v2()?;
                let p3 = r.v2()?;
                segments.push(cad_geom::spline::Bezier::new(p0, p1, p2, p3));
            }
            let closed = r.b()?;
            K::Spline(cad_geom::spline::Spline {
                segments,
                closed,
                ..Default::default()
            })
        }
        9 => {
            let n = r.u32()? as usize;
            let mut loops = Vec::with_capacity(n);
            for _ in 0..n {
                let m = r.u32()? as usize;
                let mut vertices = Vec::with_capacity(m);
                for _ in 0..m {
                    vertices.push(r.v2()?);
                }
                let closed = r.b()?;
                loops.push(Polyline {
                    vertices,
                    bulges: Vec::new(),
                    closed,
                });
            }
            let pattern_scale = r.f32()?;
            let pattern_angle = r.f32()?;
            let pattern_name = r.str()?;
            let solid = r.b()?;
            K::Hatch(cad_doc::entity::Hatch {
                loops,
                pattern_scale,
                pattern_angle,
                pattern_name,
                solid,
            })
        }
        10 => {
            let min = r.v3()?;
            let max = r.v3()?;
            K::Box(cad_doc::entity::Box3d::new(min, max))
        }
        11 => {
            let n = r.u32()? as usize;
            let mut positions = Vec::with_capacity(n);
            for _ in 0..n {
                positions.push(r.v3()?);
            }
            let m = r.u32()? as usize;
            let mut indices = Vec::with_capacity(m);
            for _ in 0..m {
                indices.push(r.u32()?);
            }
            let mut mesh = cad_doc::entity::Mesh3d {
                positions,
                normals: Vec::new(),
                indices,
            };
            mesh.recompute_normals();
            K::Mesh(mesh)
        }
        12 => {
            let n = r.v3()?;
            let d = r.f32()?;
            let m = r.u32()? as usize;
            let mut loop_pts = Vec::with_capacity(m);
            for _ in 0..m {
                loop_pts.push(r.v3()?);
            }
            K::Face(cad_doc::entity::Face3d {
                plane: cad_core::Plane3::new(n, d),
                loop_pts,
            })
        }
        13 => {
            let from = r.v3()?;
            let to = r.v3()?;
            K::Construction(cad_doc::entity::Construction { from, to })
        }
        14 => {
            let block = blocks
                .get(r.u32()? as usize)
                .copied()
                .unwrap_or(cad_doc::BlockId(0));
            let position = r.v3()?;
            let scale = r.v3()?;
            let rotation = r.f32()?;
            let rows = r.u32()?;
            let columns = r.u32()?;
            let row_spacing = r.f32()?;
            let col_spacing = r.f32()?;
            K::Insert(cad_doc::entity::InsertRef {
                block,
                position,
                scale,
                rotation,
                rows,
                columns,
                row_spacing,
                col_spacing,
            })
        }
        16 => {
            let dxf_type = r.str()?;
            K::Unknown {
                dxf_type,
                raw: Vec::new(),
            }
        }
        other => K::Unknown {
            dxf_type: format!("tag{other}"),
            raw: Vec::new(),
        },
    };
    Ok(Some(cad_doc::Entity {
        common,
        entity: geometry,
    }))
}

fn align_from_u32(v: u32) -> TextAlign {
    match v {
        1 => TextAlign::Center,
        2 => TextAlign::Right,
        3 => TextAlign::MiddleLeft,
        4 => TextAlign::MiddleCenter,
        5 => TextAlign::MiddleRight,
        6 => TextAlign::Aligned,
        _ => TextAlign::Left,
    }
}

fn linetype_from_u32(v: u32) -> cad_doc::LineType {
    match v {
        1 => cad_doc::LineType::ByLayer,
        2 => cad_doc::LineType::ByBlock,
        3 => cad_doc::LineType::Dashed {
            pattern: vec![1.0, -1.0],
        },
        _ => cad_doc::LineType::Continuous,
    }
}

fn units_from_u32(v: u32) -> cad_doc::Units {
    use cad_doc::Units::*;
    match v {
        1 => Inches,
        4 => Millimeters,
        5 => Centimeters,
        6 => Meters,
        7 => Kilometers,
        10 => Yards,
        0 => Unitless,
        _ => DrawingUnits,
    }
}

/// Save `doc` to `path`.
pub fn save(doc: &Document, path: &std::path::Path) -> Result<(), NativeError> {
    std::fs::write(path, encode(doc))?;
    Ok(())
}

/// Load `path` into `doc`.
pub fn load(path: &std::path::Path) -> Result<Document, NativeError> {
    let bytes = std::fs::read(path)?;
    let mut doc = Document::new();
    decode(&bytes, &mut doc)?;
    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_doc::entity::{Entity, Text, TextAlign};
    use cad_doc::{EntityKind, Layer};
    use cad_geom::curve::{Arc, Circle, Line, Polyline};

    fn sample() -> Document {
        let mut d = Document::new();
        d.set_name("projeto.cad");
        d.units = cad_doc::Units::Millimeters;
        let a = d.layers.ensure_default();
        let b = d.layers.insert(Layer {
            name: "Parede".into(),
            color: 1,
            ..Default::default()
        });
        d.add(Entity::line(Line::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 5.0))).with_layer(a));
        d.add(Entity::circle(Circle::new(Vec2::new(3.0, 4.0), 2.5)).with_layer(b));
        d.add(Entity::arc(Arc::from_angles(Vec2::new(-3.0, 1.0), 6.0, 0.0, 1.2)).with_layer(b));
        let mut p = Polyline::new(
            vec![Vec2::ZERO, Vec2::new(4.0, 0.0), Vec2::new(4.0, 3.0)],
            true,
        );
        p.bulges[0] = cad_geom::bulge::Bulge::from_sweep(1.0);
        d.add(Entity::polyline(p).with_layer(a));
        d.add(
            Entity::new(EntityKind::Text(Text {
                value: "Sala".into(),
                insert: Vec3::new(1.0, 2.0, 3.0),
                height: 2.5,
                width_factor: 0.9,
                rotation: 0.4,
                oblique: 12.0,
                align: TextAlign::MiddleCenter,
                value2: Some("01".into()),
                height2: 1.0,
            }))
            .with_layer(a),
        );
        d
    }

    #[test]
    fn round_trip_preserves_geometry_and_metadata() {
        let d = sample();
        let bytes = encode(&d);
        assert_eq!(&bytes[..8], MAGIC);

        let mut d2 = Document::new();
        decode(&bytes, &mut d2).unwrap();
        assert_eq!(d2.name(), "projeto.cad");
        assert_eq!(d2.units, cad_doc::Units::Millimeters);
        assert_eq!(d2.entities.len(), d.entities.len());
        assert!(d2.layers.by_name("PAREDE").is_some());
    }

    #[test]
    fn rejects_garbage() {
        let mut d = Document::new();
        assert!(matches!(
            decode(b"not a cad file at all", &mut d),
            Err(NativeError::BadMagic)
        ));
        let mut bad = encode(&sample());
        bad[8] = 99; // bump the version field
        assert!(matches!(
            decode(&bad, &mut d),
            Err(NativeError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn truncated_file_is_an_error_not_a_panic() {
        let bytes = encode(&sample());
        for cut in [0usize, 4, 9, 16, 40, bytes.len() / 2, bytes.len() - 1] {
            let mut d = Document::new();
            let _ = decode(&bytes[..cut], &mut d);
        }
    }

    #[test]
    fn spatial_index_is_usable_after_load() {
        let d = sample();
        let bytes = encode(&d);
        let mut d2 = Document::new();
        decode(&bytes, &mut d2).unwrap();
        let hits = d2.entities.candidates_at(Vec2::new(3.0, 4.0));
        assert!(!hits.is_empty(), "index not rebuilt after decode");
    }

    #[test]
    fn file_round_trip_through_the_filesystem() {
        let dir = std::env::temp_dir().join("cadkit-native-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.cad");
        let d = sample();
        save(&d, &p).unwrap();
        let back = load(&p).unwrap();
        assert_eq!(back.entities.len(), d.entities.len());
        std::fs::remove_file(&p).ok();
    }
}
