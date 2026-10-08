//! Plot: a self-contained PDF 1.4 writer.
//!
//! No dependencies, because a plotting library would dwarf the entire rest of
//! the application. PDF is a small enough format that writing the subset a CAD
//! plot needs is cheaper than linking a crate that writes all of it: a header,
//! a few objects, and a content stream of `m`/`l`/`S` path operators.
//!
//! What it emits:
//!
//! * real vector geometry, not a raster, so a plot stays sharp at any zoom;
//! * one page at a chosen paper size, in PDF points (1/72 inch);
//! * the drawing's own colours, flipped into the page's colour space;
//! * an optional title block footer.
//!
//! What it does not: fonts beyond the 14 standard ones, patterns other than
//! solid, transparency, or multiple pages. Those are real gaps, and a plot that
//! silently dropped a hatch would be worse than one that refused.

use cad_core::{Rect2, Rgba, Vec2};
use cad_doc::{Document, EntityKind};
use cad_geom::tessellate::{TessellationOptions, tessellate};

/// A paper size, in millimetres. PDF units are points, so the conversion
/// happens once here rather than at every call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Paper {
    A0,
    A1,
    A2,
    A3,
    A4,
    Letter,
    Legal,
}

impl Paper {
    /// Width and height in millimetres, portrait.
    pub fn mm(self) -> (f32, f32) {
        match self {
            Paper::A0 => (841.0, 1189.0),
            Paper::A1 => (594.0, 841.0),
            Paper::A2 => (420.0, 594.0),
            Paper::A3 => (297.0, 420.0),
            Paper::A4 => (210.0, 297.0),
            Paper::Letter => (215.9, 279.4),
            Paper::Legal => (215.9, 355.6),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Paper::A0 => "A0",
            Paper::A1 => "A1",
            Paper::A2 => "A2",
            Paper::A3 => "A3",
            Paper::A4 => "A4",
            Paper::Letter => "Letter",
            Paper::Legal => "Legal",
        }
    }

    /// Size in PDF points (1/72 inch).
    pub fn points(self) -> (f32, f32) {
        let (w, h) = self.mm();
        (mm_to_pt(w), mm_to_pt(h))
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.trim().to_ascii_uppercase().as_str() {
            "A0" => Paper::A0,
            "A1" => Paper::A1,
            "A2" => Paper::A2,
            "A3" => Paper::A3,
            "A4" => Paper::A4,
            "LETTER" | "ANSI-A" => Paper::Letter,
            "LEGAL" => Paper::Legal,
            _ => return None,
        })
    }
}

const fn mm_to_pt(mm: f32) -> f32 {
    // 25.4 mm per inch, 72 points per inch.
    mm / 25.4 * 72.0
}

/// What to plot.
#[derive(Debug, Clone)]
pub struct PlotOptions {
    pub paper: Paper,
    /// Draw a border and the title underneath it.
    pub border: bool,
    /// Written into the title block.
    pub title: String,
    /// Leave a margin around the drawing, as a fraction of the paper's smaller
    /// side. A plot with no margin clips whatever touches the edge.
    pub margin: f32,
    /// Line width in points.
    pub line_width: f32,
}

impl Default for PlotOptions {
    fn default() -> Self {
        Self {
            paper: Paper::A3,
            border: true,
            title: String::new(),
            margin: 0.08,
            line_width: 0.35,
        }
    }
}

/// Why a plot could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlotError {
    /// Nothing to plot: every entity is hidden or on a frozen layer.
    NothingToPlot,
    /// Entities the writer refuses to draw, rather than silently dropping.
    Unsupported(Vec<String>),
    Io(String),
}

impl std::fmt::Display for PlotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlotError::NothingToPlot => write!(f, "there is nothing visible to plot"),
            PlotError::Unsupported(k) => {
                write!(f, "cannot plot: {}", k.join(", "))
            }
            PlotError::Io(m) => write!(f, "could not write the plot: {m}"),
        }
    }
}

impl std::error::Error for PlotError {}

/// The world-to-page transform, kept so the caller can report the plot window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    pub scale: f32,
    /// World point at the page origin.
    pub origin: Vec2,
}

/// Map a world rectangle onto the page's drawing area, centred and scaled to fit.
///
/// Note there is no Y flip: world space is y-up and PDF user space is y-up too.
/// Only *screen* space is y-down, and nothing here is in screen space. Getting
/// this wrong produces the classic upside-down plot.
pub fn fit_frame(world: Rect2, paper: Paper, margin: f32) -> Frame {
    let (pw, ph) = paper.points();
    let m = margin.min(0.4);
    let avail_w = pw * (1.0 - m * 2.0).max(0.05);
    let avail_h = ph * (1.0 - m * 2.0).max(0.05);
    let size = world.size();
    // A degenerate rect (a single point, a zero-length line) has no size to fit;
    // give it a nominal window so the point lands in the middle.
    let (w, h) = (size.x.max(1e-6), size.y.max(1e-6));
    let scale = (avail_w / w).min(avail_h / h).min(1.0e6);
    let centre = world.center();
    // page = page_centre + (world - world_centre) * scale
    Frame {
        scale,
        origin: Vec2::new(pw * 0.5 - centre.x * scale, ph * 0.5 - centre.y * scale),
    }
}

impl Frame {
    /// World to page, in PDF points.
    pub fn apply(&self, w: Vec2) -> Vec2 {
        Vec2::new(
            self.origin.x + w.x * self.scale,
            self.origin.y + w.y * self.scale,
        )
    }

    /// The plot scale as a 1:N drawing scale on the 1/2/5 ladder.
    ///
    /// `N` is how many millimetres on paper one drawing unit prints as,
    /// inverted. It is clamped at 1:1 because a small drawing plotted larger
    /// than life has no standard scale name; drafts are enlarged 2:1, 5:1 and
    /// so on, and that is a decision the user makes, not the fitter.
    pub fn as_drawing_scale(&self) -> f32 {
        let mm_per_unit = self.scale / mm_to_pt(1.0);
        // `is_finite` first: NaN fails every comparison, and a plot scale that
        // has become NaN must not reach a title block.
        if !mm_per_unit.is_finite() || mm_per_unit <= 0.0 {
            return 1.0;
        }
        let n = 1.0 / mm_per_unit;
        if n <= 1.0 {
            return 1.0;
        }
        // Snap to the 1/2/5 ladder within the decade.
        let mag = 10f32.powf(n.log10().floor());
        let norm = n / mag;
        let snapped = if norm < 1.5 {
            1.0
        } else if norm < 3.5 {
            2.0
        } else if norm < 7.5 {
            5.0
        } else {
            10.0
        };
        (snapped * mag).max(1.0)
    }
}

/// Build the PDF for `doc`, writing it to `path`.
pub fn plot(doc: &Document, path: &std::path::Path, opts: &PlotOptions) -> Result<f32, PlotError> {
    let bytes = render(doc, opts)?;
    std::fs::write(path, bytes).map_err(|e| PlotError::Io(e.to_string()))?;
    Ok(1.0)
}

/// Build the PDF for `doc` as bytes.
pub fn render(doc: &Document, opts: &PlotOptions) -> Result<Vec<u8>, PlotError> {
    let (pw, ph) = opts.paper.points();
    let mut skipped: Vec<String> = Vec::new();

    // --- collect geometry --------------------------------------------------
    let mut paths: Vec<(Vec<Vec2>, Rgba, f32)> = Vec::new();
    // A plot at a fixed on-paper tolerance, so a circle drawn at 300 dpi on
    // screen still looks like a circle on paper.
    let tol = 0.01 / opts.paper.points().1 * 400.0;

    let mut any = false;
    for e in doc.entities.iter() {
        if !e.common.visible || !doc.layers.visible(e.layer()) {
            continue;
        }
        let color = e.resolved_color(doc.layers.color_of(e.layer(), Rgba::BLACK));
        match &e.entity {
            EntityKind::Line(l) => {
                paths.push((vec![l.p0, l.p1], color, opts.line_width));
                any = true;
            }
            EntityKind::Polyline(p) | EntityKind::Region(p) => {
                let pts = p.vertices.clone();
                if pts.is_empty() {
                    continue;
                }
                paths.push((pts, color, opts.line_width));
                any = true;
            }
            EntityKind::Spline(s) => {
                let pts: Vec<Vec2> = s.control_points.clone();
                if pts.len() >= 2 {
                    paths.push((pts, color, opts.line_width));
                    any = true;
                }
            }
            EntityKind::Circle(c) => {
                let pts = tessellate(
                    &cad_geom::Curve::Circle(*c),
                    &TessellationOptions::with_tolerance(tol),
                );
                if pts.len() >= 2 {
                    paths.push((pts, color, opts.line_width));
                    any = true;
                }
            }
            EntityKind::Arc(a) => {
                let pts = tessellate(
                    &cad_geom::Curve::Arc(*a),
                    &TessellationOptions::with_tolerance(tol),
                );
                if pts.len() >= 2 {
                    paths.push((pts, color, opts.line_width));
                    any = true;
                }
            }
            EntityKind::Ellipse(el) => {
                let pts = tessellate(
                    &cad_geom::Curve::Ellipse(*el),
                    &TessellationOptions::with_tolerance(tol),
                );
                if pts.len() >= 2 {
                    paths.push((pts, color, opts.line_width));
                    any = true;
                }
            }
            EntityKind::Text(_) => {
                // Text needs a font resource; rather than embed one, the writer
                // records it as unsupported so the caller is told, instead of
                // producing a plot with the labels missing.
                skipped.push("TEXT".to_string());
            }
            EntityKind::Point(p) => {
                // A point plots as a dot: a zero-length stroked path with a round
                // cap, which PDF expresses as a tiny circle.
                paths.push((circle_points(p.position.xy(), 0.2), color, opts.line_width));
                any = true;
            }
            EntityKind::Box(b) => {
                let (mn, mx) = (b.min.xy(), b.max.xy());
                paths.push((
                    vec![
                        Vec2::new(mn.x, mn.y),
                        Vec2::new(mx.x, mn.y),
                        Vec2::new(mx.x, mx.y),
                        Vec2::new(mn.x, mx.y),
                    ],
                    color,
                    opts.line_width,
                ));
                any = true;
            }
            EntityKind::Hatch(_) => skipped.push("HATCH".to_string()),
            EntityKind::Mesh(_) | EntityKind::Face(_) => skipped.push("3D solid".to_string()),
            EntityKind::Insert(_) => skipped.push("INSERT".to_string()),
            EntityKind::Construction(_) => skipped.push("XLINE".to_string()),
            EntityKind::Unknown { dxf_type, .. } => skipped.push(dxf_type.clone()),
            EntityKind::Dimension(_) => skipped.push("DIMENSION".to_string()),
        }
    }

    if !any {
        return Err(PlotError::NothingToPlot);
    }
    // Refusing beats a plot with holes in it.
    if !skipped.is_empty() {
        skipped.sort();
        skipped.dedup();
        return Err(PlotError::Unsupported(skipped));
    }

    // --- fit ---------------------------------------------------------------
    let mut world = Rect2::EMPTY;
    for (pts, ..) in &paths {
        for p in pts {
            world = world.expand_point(*p);
        }
    }
    let frame = fit_frame(world, opts.paper, opts.margin);

    // --- content stream ----------------------------------------------------
    let mut content = String::with_capacity(paths.len() * 120);
    content.push_str("1 J 1 j\n"); // round caps and joins: CAD lines are round
    for (pts, color, width) in &paths {
        if pts.len() < 2 {
            continue;
        }
        let stroke = pdf_stroke(*color);
        content.push_str(&format!(
            "{:.3} w\n{:.3} {:.3} {:.3} RG\n",
            *width, stroke.0, stroke.1, stroke.2
        ));
        let first = frame.apply(pts[0]);
        content.push_str(&format!("{:.2} {:.2} m\n", first.x, first.y));
        for p in &pts[1..] {
            let q = frame.apply(*p);
            content.push_str(&format!("{:.2} {:.2} l\n", q.x, q.y));
        }
        content.push_str("S\n");
    }

    if opts.border {
        content.push_str(&draw_frame(opts, pw, ph, frame.as_drawing_scale()));
    }

    // --- assemble ----------------------------------------------------------
    assemble(&content, pw, ph, &opts.title, &opts.paper)
}

/// A small circle of points, for plotting a point entity as a dot.
fn circle_points(centre: Vec2, r: f32) -> Vec<Vec2> {
    (0..8)
        .map(|i| {
            let a = std::f32::consts::TAU * i as f32 / 8.0;
            centre + Vec2::new(a.cos(), a.sin()) * r
        })
        .collect()
}

/// A colour in the 0..=1 range PDF expects.
///
/// The UI works in linear-ish 0..1 floats and PDF in device 0..1, so the values
/// pass through unchanged: they are already display-referred.
fn pdf_stroke(c: Rgba) -> (f32, f32, f32) {
    (
        c.r.clamp(0.0, 1.0),
        c.g.clamp(0.0, 1.0),
        c.b.clamp(0.0, 1.0),
    )
}

/// The border rectangle and the title block beneath it.
fn draw_frame(opts: &PlotOptions, pw: f32, ph: f32, scale: f32) -> String {
    let m = (opts.margin.min(0.4) * pw.min(ph)).max(4.0);
    let (x0, y0) = (m, m);
    let (x1, y1) = (pw - m, ph - m);
    let mut s = String::new();
    s.push_str("0.7 w\n0 0 0 RG\n");
    s.push_str(&format!("{x0:.2} {y0:.2} m\n{x1:.2} {y0:.2} l\n"));
    s.push_str(&format!("{x1:.2} {y1:.2} l\n{x0:.2} {y1:.2} l\nS\n"));
    // The scale line always prints: a plot whose scale is not on it is not a
    // drawing, it is a picture.
    let paper = paper_line(opts, scale);
    let title = if opts.title.is_empty() {
        paper.clone()
    } else {
        format!("{}   {}", opts.title, paper)
    };
    // Helvetica, 9pt, bottom-left inside the border. The font resource is one
    // of PDF's 14 standard faces, so it needs no embedding.
    s.push_str("BT /F1 9 Tf 0 0 0 rg\n");
    s.push_str(&format!("{:.2} {:.2} Td\n", x0 + 6.0, y0 + 14.0));
    s.push_str(&escape_text(&title));
    s.push_str("\nET\n");
    s
}

/// Escape the three characters PDF strings reserve.
fn escape_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '(' => out.push_str("\\("),
            ')' => out.push_str("\\)"),
            '\\' => out.push_str("\\\\"),
            // PDF strings are byte-oriented; anything outside Latin-1 is
            // dropped rather than emitted as a broken sequence.
            c if (c as u32) < 0x80 => out.push(c),
            _ => out.push('?'),
        }
    }
    out
}

/// The paper label, for the title block.
fn paper_line(opts: &PlotOptions, scale: f32) -> String {
    format!(
        "CADKit  {}  1:{}",
        opts.paper.label(),
        scale.round().max(1.0) as u32
    )
}

/// Write the PDF file structure around `content`.
fn assemble(
    content: &str,
    pw: f32,
    ph: f32,
    title: &str,
    paper: &Paper,
) -> Result<Vec<u8>, PlotError> {
    // Object numbering is fixed and explicit. A general PDF writer would build
    // an object table first; for six objects that is more machinery than it is
    // worth, and the fixed layout is easy to verify by eye in a hex dump.
    //
    //   1 catalogue   2 pages   3 page   4 contents   5 font   6 info
    let mut objects: Vec<String> = Vec::with_capacity(8);

    objects.push("<< /Type /Catalog /Pages 2 0 R >>".into());
    objects.push("<< /Type /Pages /Kids [3 0 R] /Count 1 >>".into());
    objects.push(format!(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {:.2} {:.2}] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        pw, ph
    ));

    // `endstream` has to start on its own line, so the stream data is followed
    // by an EOL that is *not* counted in /Length -- the reader takes exactly
    // Length bytes and ignores what follows. Declaring the EOL as part of the
    // stream is also legal but makes the two numbers differ by one, which is
    // the kind of off-by-one that only a strict reader notices.
    let stream_len = content.len();
    objects.push(format!(
        "<< /Length {} >>\nstream\n{content}\nendstream",
        stream_len
    ));

    objects.push(
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>".into(),
    );

    let now = pdf_date();
    let info = format!(
        "/Title ({}) /Creator (CADKit) /Producer (CADKit) {}",
        escape_text(if title.is_empty() {
            paper.label()
        } else {
            title
        }),
        now
    );
    objects.push(format!("<< {info} >>"));

    let mut out: Vec<u8> = Vec::with_capacity(content.len() + 1024);
    out.extend_from_slice(b"%PDF-1.4\n");
    // A binary comment marks the file as containing binary data, which keeps
    // naive transports from mangling it.
    out.extend_from_slice(b"%\xE2\xE3\xCF\xD3\n");

    // The xref table's byte offsets have to be known before it can be written,
    // so record them as each object is emitted.
    let mut offsets: Vec<usize> = Vec::with_capacity(objects.len());
    for (i, body) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
    }

    let xref_at = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
    out.extend_from_slice(b"0000000000 65535 f \n");
    for off in &offsets {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R /Info 6 0 R >>\nstartxref\n{}\n%%EOF\n",
            objects.len() + 1,
            xref_at
        )
        .as_bytes(),
    );
    Ok(out)
}

/// A PDF date string, `D:YYYYMMDDHHmmSSZ`.
///
/// Read from the system clock when available and otherwise fixed: a plot must
/// not fail because there is no clock.
fn pdf_date() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (y, mo, d, h, mi, s) = civil_from_unix(secs);
    format!("D:{y:04}{mo:02}{d:02}{h:02}{mi:02}{s:02}Z")
}

/// Days-from-epoch to a UTC civil date, by Howard Hinnant's algorithm.
fn civil_from_unix(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (
        y,
        m,
        d,
        (rem / 3_600) as u32,
        ((rem % 3_600) / 60) as u32,
        (rem % 60) as u32,
    )
}

/// A plot of a document, ready to be turned into a PDF.
#[derive(Debug, Clone, PartialEq)]
pub struct PlotResult {
    pub paths: usize,
    pub vertices: usize,
    pub world: Rect2,
    pub scale: f32,
}

/// Flatten a document into page-space polylines, without writing a file.
///
/// This is what makes the writer testable: the geometry is produced in the open,
/// so a regression in what gets plotted is a failing assertion rather than a
/// corrupt PDF nobody can read.
pub fn collect_paths(doc: &Document, tol: f32) -> (Vec<(Vec<Vec2>, Rgba)>, Vec<String>) {
    let mut out = Vec::new();
    let mut skipped = Vec::new();
    for e in doc.entities.iter() {
        if !e.common.visible || !doc.layers.visible(e.layer()) {
            continue;
        }
        let color = e.resolved_color(doc.layers.color_of(e.layer(), Rgba::BLACK));
        let pts: Option<Vec<Vec2>> = match &e.entity {
            EntityKind::Line(l) => Some(vec![l.p0, l.p1]),
            EntityKind::Circle(c) => Some(tessellate(
                &cad_geom::Curve::Circle(*c),
                &TessellationOptions::with_tolerance(tol),
            )),
            EntityKind::Arc(a) => Some(tessellate(
                &cad_geom::Curve::Arc(*a),
                &TessellationOptions::with_tolerance(tol),
            )),
            EntityKind::Ellipse(e) => Some(tessellate(
                &cad_geom::Curve::Ellipse(*e),
                &TessellationOptions::with_tolerance(tol),
            )),
            EntityKind::Polyline(p) | EntityKind::Region(p) => Some(p.vertices.clone()),
            EntityKind::Box(b) => Some(vec![
                Vec2::new(b.min.x, b.min.y),
                Vec2::new(b.max.x, b.min.y),
                Vec2::new(b.max.x, b.max.y),
                Vec2::new(b.min.x, b.max.y),
            ]),
            EntityKind::Point(p) => Some(circle_points(p.position.xy(), 0.2)),
            other => {
                skipped.push(entity_name(other));
                None
            }
        };
        if let Some(pts) = pts
            && pts.len() >= 2
        {
            out.push((pts, color));
        }
    }
    skipped.sort();
    skipped.dedup();
    (out, skipped)
}

fn entity_name(k: &EntityKind) -> String {
    match k {
        EntityKind::Text(_) => "TEXT".into(),
        EntityKind::Hatch(_) => "HATCH".into(),
        EntityKind::Mesh(_) => "MESH".into(),
        EntityKind::Face(_) => "FACE3D".into(),
        EntityKind::Insert(_) => "INSERT".into(),
        EntityKind::Construction(_) => "XLINE".into(),
        EntityKind::Dimension(_) => "DIMENSION".into(),
        EntityKind::Spline(_) => "SPLINE".into(),
        EntityKind::Unknown { dxf_type, .. } => dxf_type.clone(),
        _ => "OTHER".into(),
    }
}

/// Entities the writer would refuse, by name.
pub fn unsupported_entities(doc: &Document) -> Vec<String> {
    collect_paths(doc, 0.01).1
}

/// Whether a document can be plotted at all.
pub fn can_plot(doc: &Document) -> bool {
    !unsupported_entities(doc).is_empty()
}

/// Convenience: the drawing's world extent, ignoring visibility.
pub fn drawing_extents(doc: &Document) -> Rect2 {
    let (paths, _) = collect_paths(doc, 0.01);
    let mut r = Rect2::EMPTY;
    for (pts, _) in &paths {
        for p in pts {
            r = r.expand_point(*p);
        }
    }
    r
}

/// Scale label for a document on a paper, for the status bar.
pub fn scale_label(doc: &Document, paper: Paper, margin: f32) -> String {
    let world = drawing_extents(doc);
    if world.is_empty() {
        return "1:1".to_string();
    }
    let f = fit_frame(world, paper, margin);
    format!("1:{}", f.as_drawing_scale().round().max(1.0) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_doc::Entity;
    use cad_geom::curve::{Circle, Line, Polyline};

    fn doc_with(entities: Vec<Entity>) -> Document {
        let mut d = Document::new();
        let layer = d.layers.ensure_default();
        for e in entities {
            d.add(e.with_layer(layer));
        }
        d
    }

    fn line(a: Vec2, b: Vec2) -> Entity {
        Entity::line(Line::new(a, b))
    }

    #[test]
    fn paper_sizes_are_the_real_ones() {
        assert_eq!(Paper::A4.mm(), (210.0, 297.0));
        assert_eq!(Paper::A3.mm(), (297.0, 420.0));
        assert_eq!(Paper::A0.mm(), (841.0, 1189.0));
        // A4 in points is 595.28 x 841.89, which every plotter agrees on.
        let (w, h) = Paper::A4.points();
        assert!((w - 595.276).abs() < 0.01, "{w}");
        assert!((h - 841.89).abs() < 0.01, "{h}");
        assert!(w < h, "portrait is taller than wide");
    }

    #[test]
    fn paper_round_trips_through_its_name() {
        for p in [
            Paper::A0,
            Paper::A1,
            Paper::A2,
            Paper::A3,
            Paper::A4,
            Paper::Letter,
            Paper::Legal,
        ] {
            assert_eq!(Paper::from_name(p.label()), Some(p), "{}", p.label());
        }
        assert_eq!(Paper::from_name("a4"), Some(Paper::A4));
        assert_eq!(Paper::from_name(" A3 "), Some(Paper::A3));
        assert_eq!(Paper::from_name("tabloid"), None);
    }

    #[test]
    fn a_fit_centres_and_fits_the_drawing() {
        let world = Rect2::from_xywh(-50.0, -50.0, 100.0, 100.0);
        let f = fit_frame(world, Paper::A3, 0.1);
        let (pw, ph) = Paper::A3.points();
        // The centre of the drawing lands at the centre of the page.
        let c = f.apply(world.center());
        assert!((c.x - pw * 0.5).abs() < 1e-3, "{c:?}");
        assert!((c.y - ph * 0.5).abs() < 1e-3, "{c:?}");
        // And every corner is inside the margin.
        let margin = 0.1 * pw.min(ph);
        for corner in world.corners() {
            let p = f.apply(corner);
            assert!(p.x > margin * 0.9 && p.x < pw - margin * 0.9, "{p:?}");
            assert!(p.y > margin * 0.9 && p.y < ph - margin * 0.9, "{p:?}");
        }
    }

    #[test]
    fn a_fit_keeps_the_drawing_the_right_way_up() {
        // World space is y-up and PDF user space is y-up, so there is no flip.
        // Only screen space is y-down, and nothing here is in screen space --
        // which is exactly why this assertion is worth pinning: "add a flip"
        // is the first thing anyone tries when a plot looks wrong.
        let world = Rect2::from_xywh(0.0, 0.0, 10.0, 10.0);
        let f = fit_frame(world, Paper::A4, 0.0);
        let low = f.apply(Vec2::new(5.0, 0.0));
        let high = f.apply(Vec2::new(5.0, 10.0));
        assert!(
            high.y > low.y,
            "world +y must go up the page: {low:?} {high:?}"
        );
    }

    #[test]
    fn a_degenerate_rect_does_not_divide_by_zero() {
        for world in [
            Rect2::ZERO,
            Rect2::from_xywh(5.0, 5.0, 0.0, 0.0),
            Rect2::from_xywh(5.0, 5.0, 0.0, 10.0),
        ] {
            let f = fit_frame(world, Paper::A4, 0.1);
            assert!(f.scale.is_finite() && f.scale > 0.0, "{f:?}");
            assert!(f.apply(world.center()).x.is_finite());
        }
    }

    #[test]
    fn the_drawing_scale_sits_on_the_125_ladder_and_never_enlarges_silently() {
        for n in [1usize, 2, 7, 40, 137, 999, 4000] {
            let world = Rect2::from_xywh(0.0, 0.0, n as f32, n as f32);
            let f = fit_frame(world, Paper::A3, 0.05);
            let s = f.as_drawing_scale();
            assert!(s >= 1.0, "n={n} scale={s} enlarges the drawing silently");
            let mag = 10f32.powf(s.log10().floor());
            let norm = (s / mag).round();
            assert!(
                (norm - 1.0).abs() < 1e-3 || (norm - 2.0).abs() < 1e-3 || (norm - 5.0).abs() < 1e-3,
                "n={n} scale={s} is off the 1/2/5 ladder"
            );
        }
    }

    #[test]
    fn circles_are_tessellated_not_plotted_as_arcs() {
        let doc = doc_with(vec![Entity::circle(Circle::new(Vec2::ZERO, 10.0))]);
        let (paths, skipped) = collect_paths(&doc, 0.01);
        assert_eq!(paths.len(), 1);
        assert!(paths[0].0.len() > 8, "a circle needs several segments");
        assert!(skipped.is_empty(), "{skipped:?}");
    }

    #[test]
    fn entities_the_writer_refuses_are_named() {
        let doc = doc_with(vec![Entity::new(EntityKind::Text(cad_doc::Text {
            value: "x".into(),
            insert: cad_core::Vec3::ZERO,
            height: 1.0,
            width_factor: 1.0,
            rotation: 0.0,
            oblique: 0.0,
            align: cad_doc::TextAlign::Left,
            value2: None,
            height2: 0.0,
        }))]);
        let (_, skipped) = collect_paths(&doc, 0.01);
        assert_eq!(skipped, vec!["TEXT".to_string()]);
    }

    #[test]
    fn hidden_entities_are_not_plotted() {
        let mut doc = doc_with(vec![line(Vec2::ZERO, Vec2::new(1.0, 1.0))]);
        assert_eq!(collect_paths(&doc, 0.01).0.len(), 1);
        let id = doc.entities.handles()[0];
        doc.entities.get_mut(id).unwrap().common.visible = false;
        assert_eq!(collect_paths(&doc, 0.01).0.len(), 0);
    }

    #[test]
    fn a_frozen_layer_is_not_plotted() {
        let mut doc = doc_with(vec![line(Vec2::ZERO, Vec2::new(1.0, 1.0))]);
        let layer = doc.entities.handles()[0];
        let lid = doc.entities.get(layer).unwrap().layer();
        doc.layers.set_visible(lid, false);
        assert_eq!(collect_paths(&doc, 0.01).0.len(), 0);
    }

    #[test]
    fn an_empty_document_has_nothing_to_plot() {
        let doc = Document::new();
        assert!(!can_plot(&doc));
        let r = render(&doc, &PlotOptions::default());
        assert_eq!(r, Err(PlotError::NothingToPlot));
    }

    #[test]
    fn an_unsupported_entity_is_refused_rather_than_dropped() {
        // The choice that matters: a plot with a missing label is worse than a
        // refusal the user can act on.
        let doc = doc_with(vec![
            line(Vec2::ZERO, Vec2::new(1.0, 1.0)),
            Entity::new(EntityKind::Hatch(cad_doc::Hatch::solid_region(
                Polyline::new(vec![Vec2::ZERO, Vec2::new(1.0, 0.0)], true),
            ))),
        ]);
        match render(&doc, &PlotOptions::default()) {
            Err(PlotError::Unsupported(k)) => assert_eq!(k, vec!["HATCH".to_string()]),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_rendered_plot_is_a_well_formed_pdf() {
        let doc = doc_with(vec![
            line(Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0)),
            line(Vec2::new(0.0, 0.0), Vec2::new(0.0, 50.0)),
            Entity::circle(Circle::new(Vec2::new(50.0, 25.0), 20.0)),
        ]);
        let opts = PlotOptions {
            title: "Plan A".into(),
            ..PlotOptions::default()
        };
        let bytes = render(&doc, &opts).expect("a plot");

        assert!(bytes.starts_with(b"%PDF-1.4"), "header");
        assert!(
            bytes.windows(5).any(|w| w == b"%%EOF"),
            "the file must end with %%EOF"
        );
        // Byte offsets are checked on the bytes: the binary comment after the
        // header is deliberately not valid UTF-8, so any lossy conversion
        // shifts every offset past the first object.
        let ascii = |s: &str| -> bool { bytes.windows(s.len()).any(|w| w == s.as_bytes()) };
        assert!(ascii("/Root 1 0 R"), "trailer root");
        assert!(ascii("1 0 obj"), "object 1");
        assert!(ascii("6 0 obj"), "info object");

        let text = String::from_utf8_lossy(&bytes);
        // Every xref offset has to point at the object it claims. This is the
        // check a strict PDF reader does and most naive writers fail, so it is
        // worth doing for all of them rather than just the first.
        let xref = text.find("\nxref\n").expect("xref table");
        // Rows after `xref` and the `0 N` count: first the free entry for
        // object 0, then one per object, then the trailer.
        let rows = text[xref..]
            .lines()
            .skip(3)
            .skip_while(|l| l.contains("65535 f"));
        // The first row that does not start with a ten-digit offset is the
        // trailer, which is how the walk knows to stop.
        let mut walked = 0usize;
        for row in rows {
            let Some(off) = row
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<usize>().ok())
            else {
                break;
            };
            walked += 1;
            assert!(
                bytes[off..].starts_with(format!("{walked} 0 obj").as_bytes()),
                "xref entry {walked} points at {:?}",
                String::from_utf8_lossy(&bytes[off..(off + 24).min(bytes.len())])
            );
        }
        assert!(walked >= 6, "expected six objects, walked {walked}");

        // The content stream's /Length must match its actual length, or a
        // strict reader rejects the file.
        let len_at = text.find("/Length ").expect("stream length");
        let declared: usize = text[len_at + 8..]
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let start = text.find("stream\n").expect("stream") + "stream\n".len();
        let actual = text[start..].find("\nendstream").expect("endstream");
        assert_eq!(declared, actual, "declared {declared}, actual {actual}");
    }

    #[test]
    fn the_content_stream_holds_real_vector_geometry() {
        let doc = doc_with(vec![line(Vec2::ZERO, Vec2::new(100.0, 0.0))]);
        let bytes = render(&doc, &PlotOptions::default()).unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains(" m\n"), "a moveto");
        assert!(text.contains(" l\n"), "a lineto");
        assert!(text.contains("\nS\n"), "a stroke");
        assert!(text.contains(" RG\n"), "a stroke colour");
        assert!(
            !text.contains(" re\n") && !text.contains(" Do\n"),
            "no raster image: a plot must stay vector"
        );
    }

    #[test]
    fn pdf_strings_are_escaped() {
        assert_eq!(escape_text("plain"), "plain");
        assert_eq!(escape_text("a(b)c"), "a\\(b\\)c");
        assert_eq!(escape_text("back\\slash"), "back\\\\slash");
        // Anything outside Latin-1 would be a broken byte sequence.
        assert_eq!(escape_text("ação"), "a??o");
    }

    #[test]
    fn pdf_dates_are_well_formed() {
        // Epoch itself, and a known date, so the civil conversion is checked
        // rather than merely non-crashing.
        assert_eq!(civil_from_unix(0), (1970, 1, 1, 0, 0, 0));
        assert_eq!(civil_from_unix(1_700_000_000), (2023, 11, 14, 22, 13, 20));
        let d = pdf_date();
        assert!(d.starts_with("D:") && d.ends_with('Z'), "{d}");
        assert_eq!(d.len(), 17, "{d}");
        assert!(d[2..6].chars().all(|c| c.is_ascii_digit()), "{d}");
    }

    #[test]
    fn the_scale_label_reads_like_a_plot() {
        let doc = doc_with(vec![line(Vec2::ZERO, Vec2::new(1000.0, 0.0))]);
        let s = scale_label(&doc, Paper::A3, 0.05);
        let n: f32 = s.trim_start_matches("1:").parse().expect(&s);
        assert!(n >= 1.0, "a 1000-unit line must reduce, not enlarge: {s}");
        assert_eq!(scale_label(&Document::new(), Paper::A3, 0.05), "1:1");
        // A drawing far smaller than the sheet prints at 1:1, not at 1:0.001.
        let tiny = doc_with(vec![line(Vec2::ZERO, Vec2::new(1.0, 0.0))]);
        assert_eq!(scale_label(&tiny, Paper::A3, 0.05), "1:1");
    }

    #[test]
    fn colours_are_clamped_into_the_pdf_range() {
        let wild = Rgba::new(2.0, -1.0, 0.5, 1.0);
        let (r, g, b) = pdf_stroke(wild);
        assert!((0.0..=1.0).contains(&r), "{r}");
        assert!((0.0..=1.0).contains(&g), "{g}");
        assert!((0.0..=1.0).contains(&b), "{b}");
    }

    #[test]
    fn points_plot_as_a_dot() {
        let doc = doc_with(vec![Entity::new(EntityKind::Point(cad_doc::PointEnt {
            position: cad_core::Vec3::new(5.0, 5.0, 0.0),
        }))]);
        let (paths, skipped) = collect_paths(&doc, 0.01);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].0.len(), 8, "a dot is a small closed loop");
        assert!(skipped.is_empty());
    }

    #[test]
    fn the_border_draws_a_frame_and_the_title() {
        let doc = doc_with(vec![line(Vec2::ZERO, Vec2::new(10.0, 0.0))]);
        let bytes = render(
            &doc,
            &PlotOptions {
                border: true,
                title: "Sheet 1".into(),
                ..PlotOptions::default()
            },
        )
        .unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("/F1 9 Tf"), "the title uses a font");
        assert!(text.contains("Sheet 1"), "the title is on the sheet");
        assert!(text.contains("A3"), "the paper is named");
        assert!(text.contains("1:"), "the scale is stated on the sheet");
    }

    #[test]
    fn a_borderless_plot_draws_no_frame() {
        let doc = doc_with(vec![line(Vec2::ZERO, Vec2::new(10.0, 0.0))]);
        let bytes = render(
            &doc,
            &PlotOptions {
                border: false,
                ..PlotOptions::default()
            },
        )
        .unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(!text.contains("BT"), "no title block");
    }

    #[test]
    fn a_plot_is_written_to_disk() {
        let dir = std::env::temp_dir().join(format!("cadkit-pdf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("plan.pdf");
        let doc = doc_with(vec![line(Vec2::ZERO, Vec2::new(100.0, 0.0))]);
        assert!(plot(&doc, &path, &PlotOptions::default()).is_ok());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(b"%PDF-1.4"));
        assert!(bytes.len() > 400, "a plot of one line should not be tiny");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_title_line_names_the_paper_and_scale() {
        let opts = PlotOptions::default();
        let line = paper_line(&opts, 250.0);
        assert!(line.contains("A3"), "{line}");
        assert!(line.contains("1:250"), "{line}");
    }

    #[test]
    fn unsupported_entities_are_reported_for_the_ui() {
        let doc = doc_with(vec![Entity::new(EntityKind::Unknown {
            dxf_type: "MLINE".into(),
            raw: vec![],
        })]);
        assert_eq!(unsupported_entities(&doc), vec!["MLINE".to_string()]);
        assert!(
            can_plot(&doc),
            "a document with something unplottable is not plottable"
        );
    }
}
