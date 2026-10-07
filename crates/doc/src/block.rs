//! Block definitions and their placement table.

use crate::entity::Entity;
use crate::handle::{BlockId, EntityId};
use cad_core::{Aabb3, Vec3};

/// A named collection of entities that can be inserted many times.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Block {
    pub name: String,
    /// Entities in **block definition space** (the base point is the origin).
    pub entities: Vec<Entity>,
    /// Insertion base point.
    pub base_point: Vec3,
    /// Dynamic blocks carry parameter records; we keep the raw payload so the
    /// DXF round-trip does not lose data.
    pub parameters: Vec<(String, Vec<f32>)>,
}

impl Block {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
    pub fn len(&self) -> usize {
        self.entities.len()
    }
    pub fn bounds(&self) -> Aabb3 {
        let mut b = Aabb3::ZERO;
        let mut first = true;
        for e in &self.entities {
            let eb = e.bounds_3d();
            if first {
                b = eb;
                first = false;
            } else {
                b = b.union(eb);
            }
        }
        b
    }
    /// `*Model_Space`, `*Paper_Space` and the user block for a sheet.
    pub fn is_layout(&self) -> bool {
        self.name.starts_with('*')
    }
}

/// `Block` plus its handle.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockRecord {
    pub id: BlockId,
    pub block: Block,
}

/// All block definitions in the drawing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BlockTable {
    items: Vec<Option<BlockRecord>>,
    names: std::collections::HashMap<String, BlockId>,
    pub model_space: BlockId,
    pub paper_space: BlockId,
}

impl BlockTable {
    /// Creates a table seeded with the two mandatory layouts.
    pub fn new() -> Self {
        let mut t = Self::default();
        let model = t.insert(Block::new("*Model_Space"));
        let paper = t.insert(Block::new("*Paper_Space"));
        t.model_space = model;
        t.paper_space = paper;
        t
    }

    pub fn insert(&mut self, b: Block) -> BlockId {
        if let Some(id) = self.by_name(&b.name) {
            if let Some(slot) = self.items.get_mut(id.index()).and_then(|o| o.as_mut()) {
                slot.block = b;
            }
            return id;
        }
        let id = BlockId(self.items.len() as u32);
        self.names.insert(b.name.to_ascii_uppercase(), id);
        self.items.push(Some(BlockRecord { id, block: b }));
        id
    }

    pub fn by_id(&self, id: BlockId) -> Option<&Block> {
        self.items
            .get(id.index())
            .and_then(|o| o.as_ref())
            .map(|r| &r.block)
    }
    pub fn by_id_mut(&mut self, id: BlockId) -> Option<&mut Block> {
        self.items
            .get_mut(id.index())
            .and_then(|o| o.as_mut())
            .map(|r| &mut r.block)
    }
    pub fn by_name(&self, name: &str) -> Option<BlockId> {
        self.names.get(&name.to_ascii_uppercase()).copied()
    }
    pub fn name(&self, id: BlockId) -> &str {
        self.by_id(id).map(|b| b.name.as_str()).unwrap_or("<none>")
    }
    pub fn len(&self) -> usize {
        self.items.iter().filter(|o| o.is_some()).count()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn iter(&self) -> impl Iterator<Item = (BlockId, &Block)> {
        // Slots are sparse (an id is reused after an erase), so skip the empties
        // rather than reporting their indices as block ids.
        self.items
            .iter()
            .filter_map(|o| o.as_ref().map(|r| (r.id, &r.block)))
    }

    /// Add an entity to a block definition.
    pub fn add_entity(&mut self, block: BlockId, e: Entity) -> Option<EntityId> {
        let b = self.by_id_mut(block)?;
        b.entities.push(e);
        Some(EntityId(b.entities.len() as u32 - 1))
    }

    /// Explode an INSERT: returns the referenced entities already transformed
    /// into world space, ready to be added to the drawing.
    ///
    /// `resolve` looks up nested block references; it is passed in so the caller
    /// controls recursion depth and cycle detection.
    pub fn explode(
        &self,
        ins: &crate::entity::InsertRef,
        resolve: &mut dyn FnMut(BlockId, Vec3, Vec3, f32, usize) -> Vec<Entity>,
        depth: usize,
    ) -> Vec<Entity> {
        resolve(ins.block, ins.position, ins.scale, ins.rotation, depth)
    }

    pub fn snapshot(&self) -> Vec<Option<BlockRecord>> {
        self.items.clone()
    }
    pub fn restore(&mut self, snap: Vec<Option<BlockRecord>>) {
        self.names.clear();
        for r in snap.iter().flatten() {
            self.names.insert(r.block.name.to_ascii_uppercase(), r.id);
        }
        self.items = snap;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::Vec2;
    use cad_geom::curve::Line;

    #[test]
    fn new_seeds_layouts() {
        let t = BlockTable::new();
        assert_eq!(t.len(), 2);
        assert_eq!(t.name(t.model_space), "*Model_Space");
        assert_eq!(t.name(t.paper_space), "*Paper_Space");
        assert!(t.by_id(t.model_space).unwrap().is_layout());
    }

    #[test]
    fn insert_is_idempotent_by_name() {
        let mut t = BlockTable::new();
        let a = t.insert(Block::new("Door"));
        let b = t.insert(Block::new("door"));
        assert_eq!(a, b);
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn block_bounds_union() {
        let mut t = BlockTable::new();
        let id = t.insert(Block::new("two"));
        t.add_entity(id, Entity::line(Line::new(Vec2::ZERO, Vec2::new(3.0, 0.0))));
        t.add_entity(
            id,
            Entity::line(Line::new(Vec2::new(0.0, 0.0), Vec2::new(0.0, 4.0))),
        );
        let b = t.by_id(id).unwrap().bounds();
        assert!((b.size().x - 3.0).abs() < 1e-5, "{b:?}");
        assert!((b.size().y - 4.0).abs() < 1e-5, "{b:?}");
    }

    #[test]
    fn explode_delegates_to_resolver() {
        let t = BlockTable::new();
        let ins = crate::entity::InsertRef {
            block: BlockId(0),
            position: Vec3::new(5.0, 0.0, 0.0),
            scale: Vec3::splat(2.0),
            rotation: 0.0,
            rows: 1,
            columns: 1,
            row_spacing: 0.0,
            col_spacing: 0.0,
        };
        let out = t.explode(
            &ins,
            &mut |_b, pos, scale, _r, depth| {
                vec![
                    Entity::line(Line::new(Vec2::ZERO, Vec2::splat(scale.x)))
                        .translated(pos)
                        .with_layer(crate::handle::LayerId(depth as u32)),
                ]
            },
            0,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].layer(), crate::handle::LayerId(0));
    }

    #[test]
    fn snapshot_restore_keeps_names() {
        let mut t = BlockTable::new();
        let id = t.insert(Block::new("Keep"));
        let snap = t.snapshot();
        t.restore(snap);
        assert_eq!(t.by_name("KEEP"), Some(id));
    }
}
