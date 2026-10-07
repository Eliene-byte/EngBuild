//! Layers, linetypes and lineweights — plus their tables.

use crate::handle::LayerId;
use cad_core::Rgba;

/// DXF lineweight codes, in hundredths of a millimetre (`-4` = "by layer").
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(i16)]
pub enum LineWeight {
    #[default]
    ByLayer,
    ByBlock,
    Default,
    W0_05mm,
    W0_09mm,
    W0_13mm,
    W0_15mm,
    W0_18mm,
    W0_20mm,
    W0_25mm,
    W0_30mm,
    W0_35mm,
    W0_40mm,
    W0_50mm,
    W0_53mm,
    W0_60mm,
    W0_70mm,
    W0_80mm,
    W0_90mm,
    W1_00mm,
    W1_06mm,
    W1_20mm,
    W1_40mm,
    W1_58mm,
    W2_00mm,
    W2_11mm,
}

impl LineWeight {
    /// Width in millimetres, or `None` when the entity inherits.
    pub fn mm(self) -> Option<f32> {
        use LineWeight::*;
        let v = match self {
            ByLayer | ByBlock | Default => return None,
            W0_05mm => 0.05,
            W0_09mm => 0.09,
            W0_13mm => 0.13,
            W0_15mm => 0.15,
            W0_18mm => 0.18,
            W0_20mm => 0.20,
            W0_25mm => 0.25,
            W0_30mm => 0.30,
            W0_35mm => 0.35,
            W0_40mm => 0.40,
            W0_50mm => 0.50,
            W0_53mm => 0.53,
            W0_60mm => 0.60,
            W0_70mm => 0.70,
            W0_80mm => 0.80,
            W0_90mm => 0.90,
            W1_00mm => 1.00,
            W1_06mm => 1.06,
            W1_20mm => 1.20,
            W1_40mm => 1.40,
            W1_58mm => 1.58,
            W2_00mm => 2.00,
            W2_11mm => 2.11,
        };
        Some(v)
    }

    pub fn from_dxf(code: i16) -> Self {
        use LineWeight::*;
        match code {
            -3 => ByBlock,
            -2 => ByLayer,
            -1 => Default,
            5 => W0_05mm,
            9 => W0_09mm,
            13 => W0_13mm,
            15 => W0_15mm,
            18 => W0_18mm,
            20 => W0_20mm,
            25 => W0_25mm,
            30 => W0_30mm,
            35 => W0_35mm,
            40 => W0_40mm,
            50 => W0_50mm,
            53 => W0_53mm,
            60 => W0_60mm,
            70 => W0_70mm,
            80 => W0_80mm,
            90 => W0_90mm,
            100 => W1_00mm,
            106 => W1_06mm,
            120 => W1_20mm,
            140 => W1_40mm,
            158 => W1_58mm,
            200 => W2_00mm,
            211 => W2_11mm,
            _ => Default,
        }
    }

    pub fn to_dxf(self) -> i16 {
        use LineWeight::*;
        match self {
            ByBlock => -3,
            ByLayer => -2,
            Default => -1,
            W0_05mm => 5,
            W0_09mm => 9,
            W0_13mm => 13,
            W0_15mm => 15,
            W0_18mm => 18,
            W0_20mm => 20,
            W0_25mm => 25,
            W0_30mm => 30,
            W0_35mm => 35,
            W0_40mm => 40,
            W0_50mm => 50,
            W0_53mm => 53,
            W0_60mm => 60,
            W0_70mm => 70,
            W0_80mm => 80,
            W0_90mm => 90,
            W1_00mm => 100,
            W1_06mm => 106,
            W1_20mm => 120,
            W1_40mm => 140,
            W1_58mm => 158,
            W2_00mm => 200,
            W2_11mm => 211,
        }
    }

    /// Display label for the UI.
    pub fn label(self) -> &'static str {
        match self.mm() {
            None => "ByLayer",
            Some(v) => {
                // Static strings for the common set; fall back to a generic label.
                match v {
                    0.05 => "0.05 mm",
                    0.09 => "0.09 mm",
                    0.13 => "0.13 mm",
                    0.15 => "0.15 mm",
                    0.18 => "0.18 mm",
                    0.20 => "0.20 mm",
                    0.25 => "0.25 mm",
                    0.30 => "0.30 mm",
                    0.35 => "0.35 mm",
                    0.40 => "0.40 mm",
                    0.50 => "0.50 mm",
                    0.53 => "0.53 mm",
                    0.60 => "0.60 mm",
                    0.70 => "0.70 mm",
                    0.80 => "0.80 mm",
                    0.90 => "0.90 mm",
                    1.00 => "1.00 mm",
                    1.06 => "1.06 mm",
                    1.20 => "1.20 mm",
                    1.40 => "1.40 mm",
                    1.58 => "1.58 mm",
                    2.00 => "2.00 mm",
                    2.11 => "2.11 mm",
                    _ => "Custom",
                }
            }
        }
    }
}

/// A linetype: a repeating dash pattern plus an optional text/shape element.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum LineType {
    /// The default: BYLAYER entities inherit their layer's linetype, and an
    /// unresolvable linetype falls back to this rather than to nothing.
    #[default]
    Continuous,
    ByLayer,
    ByBlock,
    /// Dash lengths in drawing units, alternating on/off.
    Dashed {
        pattern: Vec<f32>,
    },
    /// Dot / centre / dash-dot style, resolved into a pattern.
    Named(&'static str),
}

impl LineType {
    /// The dash pattern to feed the GPU; `None` means a solid line.
    pub fn pattern(&self) -> Option<&[f32]> {
        match self {
            LineType::Dashed { pattern } => Some(pattern),
            _ => None,
        }
    }

    /// Total repeat length, used to keep the pattern stable while panning.
    /// DXF stores "off" segments as negative numbers, so we take the magnitudes.
    pub fn period(&self) -> f32 {
        match self {
            LineType::Dashed { pattern } => pattern.iter().map(|v| v.abs()).sum(),
            _ => 0.0,
        }
    }

    /// Resolve the inheritance chain against a layer.
    pub fn resolve<'a>(self: &'a LineType, layer: &'a LineType) -> &'a LineType {
        match self {
            LineType::ByLayer => layer,
            LineType::ByBlock => &LineType::Continuous,
            other => other,
        }
    }

    /// The 6 standard DXF linetypes, built lazily as owned patterns.
    pub fn acad(name: &str) -> LineType {
        match name.to_ascii_uppercase().as_str() {
            "CONTINUOUS" => LineType::Continuous,
            "BYLAYER" => LineType::ByLayer,
            "BYBLOCK" => LineType::ByBlock,
            "DASHED" | "DASHDOT" | "DASHDOTDOT" | "CENTER" | "PHANTOM" | "HIDDEN" | "DIVIDE"
            | "DOT" | "BORDER" | "DASHDOT2" | "DASHDOTALT" => LineType::Dashed {
                pattern: acad_pattern(name).to_vec(),
            },
            _ => LineType::Continuous,
        }
    }
}

/// The canonical DXF dash patterns, expressed in millimetres.
pub fn acad_pattern(name: &str) -> &'static [f32] {
    match name.to_ascii_uppercase().as_str() {
        "DASHED" => &[12.7, -6.35],
        "DASHDOT" => &[12.7, -6.35, 0.0, -6.35],
        "DASHDOTDOT" => &[12.7, -6.35, 0.0, -6.35, 0.0, -6.35],
        "CENTER" => &[31.75, -6.35, 6.35, -6.35],
        "PHANTOM" => &[31.75, -6.35, 6.35, -6.35, 0.0, -6.35],
        "HIDDEN" => &[6.35, -2.54],
        "DIVIDE" => &[6.35, -2.54, 0.0, -2.54],
        "DOT" => &[0.0, -2.54],
        "BORDER" => &[6.35, -2.54, 6.35, -2.54, 0.0, -2.54, 6.35, -2.54],
        _ => &[12.7, -6.35],
    }
}

/// A single layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub name: String,
    /// ACI colour index; `256` means "by block".
    pub color: i16,
    pub linetype: LineType,
    pub lineweight: LineWeight,
    pub visible: bool,
    pub locked: bool,
    pub frozen: bool,
    pub plot: bool,
    /// Print/plot order.
    pub plot_order: i32,
    /// Working plane elevation for 2D layouts.
    pub elevation: f32,
    /// Optional per-layer transparency in `[0, 1]`.
    pub transparency: f32,
}

impl Default for Layer {
    fn default() -> Self {
        Self {
            name: "0".to_string(),
            color: 7,
            linetype: LineType::Continuous,
            lineweight: LineWeight::Default,
            visible: true,
            locked: false,
            frozen: false,
            plot: true,
            plot_order: 0,
            elevation: 0.0,
            transparency: 0.0,
        }
    }
}

impl Layer {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Effective colour given the UI theme (ACI 7 is "white/black").
    pub fn resolved_color(&self, by_block: Rgba) -> Rgba {
        cad_core::aci_to_rgba(self.color, by_block)
    }

    /// True when the layer can be picked or edited.
    pub fn is_editable(&self) -> bool {
        self.visible && !self.locked && !self.frozen
    }
}

/// All layers of a drawing, keyed by handle and by (case-insensitive) name.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LayerTable {
    items: Vec<Option<Layer>>,
    names: std::collections::HashMap<String, LayerId>,
}

impl LayerTable {
    /// Insert, or return the existing layer with this name.
    pub fn insert(&mut self, layer: Layer) -> LayerId {
        if let Some(id) = self.names.get(&layer.name.to_ascii_uppercase()) {
            return *id;
        }
        let id = LayerId(self.items.len() as u32);
        self.names.insert(layer.name.to_ascii_uppercase(), id);
        self.items.push(Some(layer));
        id
    }

    /// Insert at a specific id, keeping handle/name maps consistent.
    pub fn insert_at(&mut self, id: LayerId, layer: Layer) {
        let key = layer.name.to_ascii_uppercase();
        while self.items.len() <= id.index() {
            self.items.push(None);
        }
        self.items[id.index()] = Some(layer);
        self.names.insert(key, id);
    }

    pub fn by_id(&self, id: LayerId) -> Option<&Layer> {
        self.items.get(id.index()).and_then(|o| o.as_ref())
    }
    pub fn by_id_mut(&mut self, id: LayerId) -> Option<&mut Layer> {
        self.items.get_mut(id.index()).and_then(|o| o.as_mut())
    }
    pub fn by_name(&self, name: &str) -> Option<LayerId> {
        self.names.get(&name.to_ascii_uppercase()).copied()
    }
    pub fn name(&self, id: LayerId) -> &str {
        self.by_id(id).map(|l| l.name.as_str()).unwrap_or("<none>")
    }
    pub fn len(&self) -> usize {
        self.items.iter().filter(|o| o.is_some()).count()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Layer 0 always exists; it is the fallback for every new entity.
    pub fn ensure_default(&mut self) -> LayerId {
        match self.by_name("0") {
            Some(id) => id,
            None => self.insert(Layer::new("0")),
        }
    }
    pub fn default_layer(&self) -> LayerId {
        self.by_name("0").unwrap_or(LayerId(0))
    }
    pub fn iter(&self) -> impl Iterator<Item = (LayerId, &Layer)> {
        self.items
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.as_ref().map(|l| (LayerId(i as u32), l)))
    }
    pub fn visible(&self, id: LayerId) -> bool {
        self.by_id(id)
            .map(|l| l.visible && !l.frozen)
            .unwrap_or(false)
    }
    pub fn set_visible(&mut self, id: LayerId, v: bool) {
        if let Some(l) = self.by_id_mut(id) {
            l.visible = v;
        }
    }
    pub fn locked(&self, id: LayerId) -> bool {
        self.by_id(id).map(|l| l.locked || l.frozen).unwrap_or(true)
    }
    pub fn set_locked(&mut self, id: LayerId, v: bool) {
        if let Some(l) = self.by_id_mut(id) {
            l.locked = v;
        }
    }
    /// Resolved colour of a layer, honouring "by block".
    pub fn color_of(&self, id: LayerId, by_block: Rgba) -> Rgba {
        self.by_id(id)
            .map(|l| l.resolved_color(by_block))
            .unwrap_or(by_block)
    }
    /// Snapshot for undo.
    pub fn snapshot(&self) -> Vec<Option<Layer>> {
        self.items.clone()
    }
    pub fn restore(&mut self, snap: Vec<Option<Layer>>) {
        self.names.clear();
        for (i, o) in snap.iter().enumerate() {
            if let Some(l) = o {
                self.names
                    .insert(l.name.to_ascii_uppercase(), LayerId(i as u32));
            }
        }
        self.items = snap;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_table_is_case_insensitive() {
        let mut t = LayerTable::default();
        t.ensure_default();
        let a = t.insert(Layer::new("Walls"));
        let b = t.insert(Layer::new("walls"));
        assert_eq!(a, b);
        assert_eq!(t.len(), 2);
        assert_eq!(t.by_name("WALLS"), Some(a));
    }

    #[test]
    fn remove_and_restore() {
        let mut t = LayerTable::default();
        t.ensure_default();
        let a = t.insert(Layer::new("x"));
        let snap = t.snapshot();
        t.restore(snap);
        assert_eq!(t.by_name("x"), Some(a));
    }

    #[test]
    fn linetype_inheritance() {
        let layer = LineType::Dashed {
            pattern: vec![1.0, -1.0],
        };
        assert!(LineType::ByLayer.resolve(&layer).pattern().is_some());
        assert!(LineType::Continuous.resolve(&layer).pattern().is_none());
        assert_eq!(layer.period(), 2.0);
    }

    #[test]
    fn lineweight_round_trip() {
        for w in [
            LineWeight::ByLayer,
            LineWeight::W0_20mm,
            LineWeight::W2_11mm,
        ] {
            assert_eq!(LineWeight::from_dxf(w.to_dxf()), w);
        }
        assert_eq!(LineWeight::W0_20mm.mm(), Some(0.2));
        assert_eq!(LineWeight::ByLayer.mm(), None);
        assert_eq!(LineWeight::W1_00mm.label(), "1.00 mm");
    }

    #[test]
    fn linetype_pattern_lookup() {
        let d = LineType::acad("CENTER");
        let p = d.pattern().unwrap();
        assert_eq!(p.len(), 4);
        assert!((p.iter().sum::<f32>() - 25.4).abs() < 1e-4);
    }

    #[test]
    fn editable_requires_visible_unlocked() {
        let mut l = Layer::new("a");
        assert!(l.is_editable());
        l.locked = true;
        assert!(!l.is_editable());
        l.locked = false;
        l.frozen = true;
        assert!(!l.is_editable());
    }
}
