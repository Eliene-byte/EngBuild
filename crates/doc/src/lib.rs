//! `cad-doc` — the in-memory drawing database.
//!
//! Design goals, in order:
//! 1. **Stable identity.** Every entity has a 64-bit handle that never changes,
//!    even if the entity moves in the backing store.
//! 2. **Cheap undo.** Every mutation runs inside a [`Txn`], which records
//!    before/after images of only the touched entities.
//! 3. **Fast picking.** A uniform-grid spatial index keeps hover and window
//!    selection O(visible) instead of O(document).

pub mod block;
pub mod entity;
pub mod handle;
pub mod layer;
pub mod store;
pub mod style;
pub mod undo;

use cad_core::{Rect2, Vec3};

pub use block::{Block, BlockRecord, BlockTable};
pub use entity::{
    Box3d, Construction, Entity, EntityCommon, EntityKind, Face3d, Hatch, InsertRef, Mesh3d,
    PointEnt, Text, TextAlign,
};
pub use handle::{BlockId, EntityId, HandleTable, LayerId};
pub use layer::{Layer, LayerTable, LineType, LineWeight};
pub use store::EntityStore;
pub use style::{TextStyle, TextStyleTable};
pub use undo::{History, Txn};

/// Unit system of a drawing, mirroring the DXF `$INSUNITS` group code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u16)]
pub enum Units {
    Unitless = 0,
    Inches = 1,
    Feet = 2,
    Miles = 3,
    Millimeters = 4,
    Centimeters = 5,
    Meters = 6,
    Kilometers = 7,
    Microinches = 8,
    Mils = 9,
    Yards = 10,
    Angstroms = 11,
    Nanometers = 12,
    Microns = 13,
    Decimeters = 14,
    Decameters = 15,
    Hectometers = 16,
    Gigameters = 17,
    Astronomical = 18,
    LightYears = 19,
    Parsecs = 20,
    UsSurveyFeet = 21,
    UsSurveyInch = 22,
    UsSurveyYard = 23,
    UsSurveyMile = 24,
    #[default]
    DrawingUnits = 25,
}

impl Units {
    /// Metres per one of these units — used for the status bar read-out.
    pub fn to_metres(self) -> f32 {
        match self {
            Units::Inches | Units::UsSurveyInch => 0.0254,
            Units::Feet | Units::UsSurveyFeet => 0.3048,
            Units::Yards | Units::UsSurveyYard => 0.9144,
            Units::Miles | Units::UsSurveyMile => 1609.344,
            Units::Millimeters => 0.001,
            Units::Centimeters => 0.01,
            Units::Decimeters => 0.1,
            Units::Meters => 1.0,
            Units::Decameters => 10.0,
            Units::Hectometers => 100.0,
            Units::Kilometers => 1000.0,
            Units::Gigameters => 1.0e9,
            Units::Astronomical => 9.4607e15,
            Units::LightYears => 9.4607e15,
            Units::Parsecs => 3.0857e16,
            Units::Mils => 2.54e-5,
            Units::Microinches => 2.54e-8,
            Units::Microns => 1.0e-6,
            Units::Nanometers => 1.0e-9,
            Units::Angstroms => 1.0e-10,
            _ => 1.0,
        }
    }

    /// Canonical DXF spelling.
    pub fn label(self) -> &'static str {
        match self {
            Units::Unitless => "Unitless",
            Units::Inches => "Inches",
            Units::Feet => "Feet",
            Units::Miles => "Miles",
            Units::Millimeters => "Millimeters",
            Units::Centimeters => "Centimeters",
            Units::Meters => "Meters",
            Units::Kilometers => "Kilometers",
            Units::Microinches => "Microinches",
            Units::Mils => "Mils",
            Units::Yards => "Yards",
            Units::Angstroms => "Angstroms",
            Units::Nanometers => "Nanometers",
            Units::Microns => "Microns",
            Units::Decimeters => "Decimeters",
            Units::Decameters => "Decameters",
            Units::Hectometers => "Hectometers",
            Units::Gigameters => "Gigameters",
            Units::Astronomical => "Astronomical",
            Units::LightYears => "Light Years",
            Units::Parsecs => "Parsecs",
            Units::UsSurveyFeet => "US Survey Feet",
            Units::UsSurveyInch => "US Survey Inches",
            Units::UsSurveyYard => "US Survey Yards",
            Units::UsSurveyMile => "US Survey Miles",
            Units::DrawingUnits => "Drawing Units",
        }
    }

    pub fn suffix(self) -> &'static str {
        match self {
            Units::Inches | Units::UsSurveyInch => "\"",
            Units::Feet | Units::UsSurveyFeet => "'",
            Units::Millimeters | Units::Centimeters | Units::Meters => "m",
            Units::DrawingUnits => "u",
            _ => "",
        }
    }
}

/// Everything the document owns. Cheap to clone via `Arc` for undo snapshots.
#[derive(Debug, Clone)]
pub struct Document {
    pub entities: EntityStore,
    pub layers: LayerTable,
    pub blocks: BlockTable,
    pub styles: TextStyleTable,
    pub units: Units,
    /// Header variable: `EXTMIN`/`EXTMAX` cached from the last full scan.
    cached_extents: Option<(cad_core::Aabb3, Vec3)>,
    /// Bumped on every mutation so the renderer can rebuild caches lazily.
    pub revision: u64,
    name: String,
}

impl Default for Document {
    fn default() -> Self {
        let mut layers = LayerTable::default();
        layers.ensure_default();
        Self {
            entities: EntityStore::default(),
            layers,
            blocks: BlockTable::default(),
            styles: TextStyleTable::default(),
            units: Units::DrawingUnits,
            cached_extents: None,
            revision: 1,
            name: "Drawing1.dwg".to_string(),
        }
    }
}

impl Document {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn set_name(&mut self, n: impl Into<String>) {
        self.name = n.into();
    }

    /// Number of live entities.
    pub fn len(&self) -> usize {
        self.entities.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Cached 3D extents of every visible entity plus the centre point.
    /// Returns `None` for an empty document.
    pub fn extents(&mut self) -> Option<(cad_core::Aabb3, Vec3)> {
        if self.cached_extents.is_none() {
            let mut bb = cad_core::Aabb3::ZERO;
            let mut any = false;
            for e in self.entities.iter() {
                if !self.layers.visible(e.layer()) {
                    continue;
                }
                bb = bb.union(e.bounds_3d());
                any = true;
            }
            if any {
                self.cached_extents = Some((bb, bb.center()));
            }
        }
        self.cached_extents
    }

    /// Extents computed from scratch (no caching), so callers holding `&self`
    /// — exporters, for instance — do not have to clone the document.
    pub fn compute_extents(&self) -> Option<(cad_core::Aabb3, Vec3)> {
        let mut bb = cad_core::Aabb3::ZERO;
        let mut any = false;
        for e in self.entities.iter() {
            if !self.layers.visible(e.layer()) {
                continue;
            }
            bb = bb.union(e.bounds_3d());
            any = true;
        }
        if any { Some((bb, bb.center())) } else { None }
    }

    pub fn extents_2d(&mut self) -> Option<Rect2> {
        let (bb, _) = self.extents()?;
        Some(cad_core::Rect2::new(
            cad_core::Vec2::new(bb.min.x, bb.min.y),
            cad_core::Vec2::new(bb.max.x, bb.max.y),
        ))
    }

    /// Forces the extents cache to be recomputed on the next query.
    pub fn invalidate_extents(&mut self) {
        self.cached_extents = None;
        self.revision = self.revision.wrapping_add(1);
    }

    /// Convenience: add an entity, honouring the current layer.
    pub fn add(&mut self, entity: Entity) -> EntityId {
        let id = self.entities.insert(entity);
        self.invalidate_extents();
        id
    }

    /// Remove and return an entity.
    pub fn remove(&mut self, id: EntityId) -> Option<Entity> {
        let r = self.entities.remove(id);
        if r.is_some() {
            self.invalidate_extents();
        }
        r
    }

    /// Iterate handles of all entities, in insertion order.
    pub fn handles(&self) -> Vec<EntityId> {
        self.entities.handles()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_geom::curve::{Circle, Line};

    #[test]
    fn default_document_has_layer_zero() {
        let d = Document::new();
        assert_eq!(d.layers.len(), 1);
        assert!(d.layers.by_id(LayerId(0)).is_some());
        assert_eq!(d.units, Units::DrawingUnits);
    }

    #[test]
    fn add_updates_extents() {
        let mut d = Document::new();
        assert!(d.extents().is_none());
        d.add(Entity::line(Line::new(
            cad_core::Vec2::ZERO,
            cad_core::Vec2::new(10.0, 0.0),
        )));
        let (bb, c) = d.extents().unwrap();
        assert!(
            (c - cad_core::Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5,
            "{c:?}"
        );
        assert!((bb.size().x - 10.0).abs() < 1e-5);
    }

    #[test]
    fn hidden_layer_excluded_from_extents() {
        let mut d = Document::new();
        let lid = d.layers.insert(Layer::new("hidden"));
        d.layers.set_visible(lid, false);
        let mut e = Entity::circle(Circle::new(cad_core::Vec2::ZERO, 5.0));
        e.common.layer = lid;
        d.add(e);
        assert!(d.extents().is_none());
    }

    #[test]
    fn units_conversion() {
        assert!((Units::Feet.to_metres() - 0.3048).abs() < 1e-6);
        assert!((Units::Millimeters.to_metres() - 0.001).abs() < 1e-9);
        assert_eq!(Units::Inches.suffix(), "\"");
    }
}
