//! Transactional undo/redo.
//!
//! Instead of implementing an inverse operation per command (which is where
//! most CAD undo stacks rot), we snapshot the *before* image of every touched
//! entity inside a [`Txn`], then derive the after image by diffing on commit.
//! Correct by construction, and the cost is proportional to what actually
//! changed — dragging one entity out of a million does not copy a million.
//!
//! Layer and block tables are small enough to snapshot wholesale, so a
//! transaction that edits them is undone just as reliably.

use crate::block::BlockRecord;
use crate::entity::Entity;
use crate::handle::EntityId;
use crate::layer::Layer;
use crate::store::EntityStore;
use std::collections::BTreeMap;

/// Snapshot of a whole table, used for the small side-tables.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TableSnapshot<T> {
    pub items: Vec<Option<T>>,
}

type LayerSnap = TableSnapshot<Layer>;
type BlockSnap = TableSnapshot<BlockRecord>;

impl LayerSnap {
    fn of(t: &crate::layer::LayerTable) -> Self {
        Self {
            items: t.snapshot(),
        }
    }
}
impl BlockSnap {
    fn of(t: &crate::block::BlockTable) -> Self {
        Self {
            items: t.snapshot(),
        }
    }
}

/// One undoable step: the before/after images of everything it touched.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Change {
    before: BTreeMap<EntityId, Option<Entity>>,
    after: BTreeMap<EntityId, Option<Entity>>,
    layers_before: Option<LayerSnap>,
    layers_after: Option<LayerSnap>,
    blocks_before: Option<BlockSnap>,
    blocks_after: Option<BlockSnap>,
    pub label: String,
}

impl Change {
    /// True when nothing actually changed.
    pub fn is_empty(&self) -> bool {
        self.before.is_empty() && self.layers_before.is_none() && self.blocks_before.is_none()
    }

    /// Handles whose entity was created, deleted or modified.
    pub fn touched(&self) -> impl Iterator<Item = (EntityId, &Option<Entity>)> {
        self.before.iter().map(|(k, v)| (*k, v))
    }

    /// `(created, deleted, modified)`.
    pub fn stats(&self) -> (usize, usize, usize) {
        let mut created = 0;
        let mut deleted = 0;
        let mut modified = 0;
        for (id, b) in &self.before {
            match (b, self.after.get(id)) {
                // No prior image and one now: the entity was created.
                (None, Some(Some(_))) => created += 1,
                // Prior image and none now: the entity was deleted.
                (Some(_), Some(None)) => deleted += 1,
                (Some(x), Some(Some(y))) => {
                    if x != y {
                        modified += 1;
                    }
                }
                _ => {}
            }
        }
        (created, deleted, modified)
    }

    fn rewind(&self, store: &mut EntityStore) {
        for (id, old) in &self.before {
            match old {
                Some(e) => {
                    store.restore(*id, e.clone());
                }
                None => {
                    store.remove(*id);
                }
            }
        }
    }
    fn replay(&self, store: &mut EntityStore) {
        for (id, new) in &self.after {
            match new {
                Some(e) => {
                    store.restore(*id, e.clone());
                }
                None => {
                    store.remove(*id);
                }
            }
        }
    }
}

/// The undo/redo stack.
#[derive(Debug, Default)]
pub struct History {
    undo: Vec<Change>,
    redo: Vec<Change>,
    pub limit: usize,
    /// When false, transactions are applied but not recorded (macro playback).
    pub enabled: bool,
}

impl History {
    pub fn new() -> Self {
        Self {
            limit: 512,
            enabled: true,
            ..Default::default()
        }
    }

    #[inline]
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    #[inline]
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
    #[inline]
    pub fn undo_depth(&self) -> usize {
        self.undo.len()
    }
    #[inline]
    pub fn redo_depth(&self) -> usize {
        self.redo.len()
    }
    #[inline]
    pub fn undo_label(&self) -> Option<&str> {
        self.undo.last().map(|c| c.label.as_str())
    }
    #[inline]
    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().map(|c| c.label.as_str())
    }
    pub fn next_undo_label(&self) -> Option<String> {
        self.undo_label().map(|l| format!("Undo {l}"))
    }
    pub fn next_redo_label(&self) -> Option<String> {
        self.redo_label().map(|l| format!("Redo {l}"))
    }

    /// Start a transaction. Dropping the `Txn` without `commit` rolls back.
    pub fn begin<'a>(&'a mut self, store: &'a mut EntityStore, label: &str) -> Txn<'a> {
        Txn {
            history: self,
            store,
            before: BTreeMap::new(),
            layers_before: None,
            layers_after: None,
            blocks_before: None,
            blocks_after: None,
            label: label.to_string(),
            done: false,
        }
    }

    fn push(&mut self, change: Change) {
        if !self.enabled || change.is_empty() {
            return;
        }
        self.redo.clear();
        self.undo.push(change);
        if self.undo.len() > self.limit {
            self.undo.remove(0);
        }
    }

    /// Roll the store back one step. The caller restores the tables from
    /// [`Change`] if the UI cares (the document wrapper does it for us).
    pub fn undo(&mut self, store: &mut EntityStore) -> Option<&Change> {
        let c = self.undo.pop()?;
        c.rewind(store);
        self.redo.push(c);
        self.redo.last()
    }

    pub fn redo(&mut self, store: &mut EntityStore) -> Option<&Change> {
        let c = self.redo.pop()?;
        c.replay(store);
        self.undo.push(c);
        self.undo.last()
    }

    /// Pop the raw change without applying it (the document applies the tables).
    pub fn pop_undo(&mut self) -> Option<Change> {
        let c = self.undo.pop()?;
        self.redo.push(c.clone());
        Some(c)
    }
    pub fn pop_redo(&mut self) -> Option<Change> {
        let c = self.redo.pop()?;
        self.undo.push(c.clone());
        Some(c)
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }
}

/// A live transaction. While it exists, every touched entity's original value
/// is remembered; `commit` turns it into an undoable [`Change`].
pub struct Txn<'a> {
    history: &'a mut History,
    store: &'a mut EntityStore,
    before: BTreeMap<EntityId, Option<Entity>>,
    layers_before: Option<LayerSnap>,
    layers_after: Option<LayerSnap>,
    blocks_before: Option<BlockSnap>,
    blocks_after: Option<BlockSnap>,
    label: String,
    done: bool,
}

impl<'a> Txn<'a> {
    /// Capture the current value of `id` before the first mutation.
    pub fn touch(&mut self, id: EntityId) {
        if !self.before.contains_key(&id) {
            let cur = self.store.get(id).cloned();
            self.before.insert(id, cur);
        }
    }

    /// Snapshot the layer table so layer edits become undoable too.
    pub fn touch_layers(&mut self, layers: &crate::layer::LayerTable) {
        if self.layers_before.is_none() {
            self.layers_before = Some(LayerSnap::of(layers));
        }
    }

    /// Snapshot the block table.
    pub fn touch_blocks(&mut self, blocks: &crate::block::BlockTable) {
        if self.blocks_before.is_none() {
            self.blocks_before = Some(BlockSnap::of(blocks));
        }
    }

    pub fn insert(&mut self, e: Entity) -> EntityId {
        let id = self.store.insert(e);
        // Record the (absent) prior image so undo removes it again.
        self.before.entry(id).or_insert(None);
        id
    }
    pub fn remove(&mut self, id: EntityId) -> Option<Entity> {
        self.touch(id);
        self.store.remove(id)
    }
    pub fn replace(&mut self, id: EntityId, e: Entity) -> bool {
        self.touch(id);
        self.store.replace(id, e)
    }
    /// Read-only access while holding the store borrow.
    pub fn get(&self, id: EntityId) -> Option<&Entity> {
        self.store.get(id)
    }
    pub fn set_label(&mut self, l: impl Into<String>) {
        self.label = l.into();
    }
    pub fn touched_count(&self) -> usize {
        self.before.len()
    }

    /// Finalise the transaction into an undoable step.
    pub fn commit(mut self) -> bool {
        self.done = true;
        let after: BTreeMap<_, _> = self
            .before
            .keys()
            .map(|id| (*id, self.store.get(*id).cloned()))
            .collect();

        // Drop no-op touches (touched then restored the same value).
        let before: BTreeMap<_, _> = self
            .before
            .iter()
            .filter(|(id, b)| after.get(id) != Some(b))
            .map(|(k, v)| (*k, v.clone()))
            .collect();

        let layers_before = self.layers_before.clone();
        let layers_after = self.layers_after.clone();
        let blocks_before = self.blocks_before.clone();
        let blocks_after = self.blocks_after.clone();
        if before.is_empty() && layers_before.is_none() && blocks_before.is_none() {
            return false;
        }
        let change = Change {
            before,
            after,
            layers_before,
            layers_after,
            blocks_before,
            blocks_after,
            label: std::mem::take(&mut self.label),
        };
        self.history.push(change);
        true
    }

    /// Attach the post-edit table snapshots so undo/redo can restore them.
    /// Call this just before `commit_with_tables`.
    pub fn capture_tables_after(
        &mut self,
        layers: Option<&crate::layer::LayerTable>,
        blocks: Option<&crate::block::BlockTable>,
    ) {
        if self.layers_before.is_some() {
            if let Some(l) = layers {
                self.layers_after = Some(LayerSnap::of(l));
            }
        }
        if self.blocks_before.is_some() {
            if let Some(b) = blocks {
                self.blocks_after = Some(BlockSnap::of(b));
            }
        }
    }

    pub fn layers_before(&self) -> Option<&LayerSnap> {
        self.layers_before.as_ref()
    }
    pub fn blocks_before(&self) -> Option<&BlockSnap> {
        self.blocks_before.as_ref()
    }

    /// Abort: roll every touched entity back to its original value.
    pub fn rollback(mut self) {
        self.done = true;
        self.restore_before();
    }

    fn restore_before(&mut self) {
        for (id, old) in &self.before {
            match old {
                Some(e) => {
                    self.store.restore(*id, e.clone());
                }
                None => {
                    self.store.remove(*id);
                }
            }
        }
    }
}

impl Drop for Txn<'_> {
    fn drop(&mut self) {
        if !self.done {
            // Implicit rollback: nothing observed, so nothing should persist.
            self.restore_before();
        }
    }
}

/// Apply a `Change` in reverse (used by the document-level undo).
pub fn apply_rewind(c: &Change, store: &mut EntityStore) {
    c.rewind(store);
}
/// Apply a `Change` forwards.
pub fn apply_replay(c: &Change, store: &mut EntityStore) {
    c.replay(store);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockTable;
    use crate::entity::{Entity, EntityKind};
    use crate::layer::{Layer, LayerTable};
    use cad_core::Vec2;
    use cad_geom::curve::Circle;

    fn store_with_circle() -> (EntityStore, EntityId) {
        let mut s = EntityStore::new();
        let id = s.insert(Entity::circle(Circle::new(Vec2::ZERO, 1.0)));
        (s, id)
    }

    fn circle_x(e: &Entity) -> f32 {
        match &e.entity {
            EntityKind::Circle(c) => c.center.x,
            _ => f32::NAN,
        }
    }

    #[test]
    fn insert_then_undo_removes() {
        let (mut store, _) = store_with_circle();
        let mut h = History::new();
        {
            let mut tx = h.begin(&mut store, "Add circle");
            tx.insert(Entity::circle(Circle::new(Vec2::new(10.0, 0.0), 2.0)));
            assert!(tx.commit());
        }
        assert_eq!(store.len(), 2);
        assert!(h.can_undo());
        assert!(h.undo(&mut store).is_some());
        assert_eq!(store.len(), 1);
        assert!(h.can_redo());
        assert!(h.redo(&mut store).is_some());
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn modify_then_undo_restores_value() {
        let (mut store, id) = store_with_circle();
        let mut h = History::new();
        {
            let mut tx = h.begin(&mut store, "Move");
            let e = tx.get(id).unwrap().clone();
            tx.replace(id, e.translated(cad_core::Vec3::new(5.0, 0.0, 0.0)));
            tx.commit();
        }
        assert!((circle_x(store.get(id).unwrap()) - 5.0).abs() < 1e-6);
        h.undo(&mut store);
        assert!(circle_x(store.get(id).unwrap()).abs() < 1e-6);
    }

    #[test]
    fn delete_then_undo_restores_entity() {
        let (mut store, id) = store_with_circle();
        let mut h = History::new();
        {
            let mut tx = h.begin(&mut store, "Erase");
            tx.remove(id);
            tx.commit();
        }
        assert!(store.get(id).is_none());
        h.undo(&mut store);
        assert!(store.get(id).is_some());
    }

    #[test]
    fn dropped_transaction_rolls_back() {
        let (mut store, id) = store_with_circle();
        let mut h = History::new();
        {
            let mut tx = h.begin(&mut store, "Abandoned");
            let e = tx.get(id).unwrap().clone();
            tx.replace(id, e.translated(cad_core::Vec3::new(9.0, 9.0, 0.0)));
            tx.insert(Entity::circle(Circle::new(Vec2::ZERO, 7.0)));
            // no commit: dropping rolls everything back
        }
        assert_eq!(store.len(), 1);
        assert!(circle_x(store.get(id).unwrap()).abs() < 1e-6);
        assert!(!h.can_undo());
    }

    #[test]
    fn explicit_rollback() {
        let (mut store, id) = store_with_circle();
        let mut h = History::new();
        let mut tx = h.begin(&mut store, "Abandoned");
        let e = tx.get(id).unwrap().clone();
        tx.replace(id, e.translated(cad_core::Vec3::new(9.0, 9.0, 0.0)));
        tx.rollback();
        assert!(circle_x(store.get(id).unwrap()).abs() < 1e-6);
        assert!(!h.can_undo());
    }

    #[test]
    fn no_op_transaction_records_nothing() {
        let (mut store, id) = store_with_circle();
        let mut h = History::new();
        let mut tx = h.begin(&mut store, "Nothing");
        let e = tx.get(id).unwrap().clone();
        tx.replace(id, e);
        assert!(!tx.commit());
        assert!(!h.can_undo());
    }

    #[test]
    fn undo_limit_is_enforced() {
        let (mut store, _) = store_with_circle();
        let mut h = History::new();
        h.limit = 4;
        for i in 0..10 {
            let mut tx = h.begin(&mut store, "Add");
            tx.insert(Entity::circle(Circle::new(Vec2::new(i as f32, 0.0), 0.5)));
            tx.commit();
        }
        assert_eq!(h.undo_depth(), 4);
        let mut n = 0;
        while h.undo(&mut store).is_some() {
            n += 1;
        }
        assert_eq!(n, 4);
    }

    #[test]
    fn redo_stack_clears_on_new_edit() {
        let (mut store, _) = store_with_circle();
        let mut h = History::new();
        {
            let mut tx = h.begin(&mut store, "A");
            tx.insert(Entity::circle(Circle::new(Vec2::new(1.0, 0.0), 1.0)));
            tx.commit();
        }
        h.undo(&mut store);
        assert!(h.can_redo());
        {
            let mut tx = h.begin(&mut store, "B");
            tx.insert(Entity::circle(Circle::new(Vec2::new(2.0, 0.0), 1.0)));
            tx.commit();
        }
        assert!(!h.can_redo());
    }

    #[test]
    fn multi_entity_change_is_atomic() {
        let (mut store, _) = store_with_circle();
        let mut h = History::new();
        let before = store.len();
        {
            let mut tx = h.begin(&mut store, "Explode");
            for i in 0..10 {
                tx.insert(Entity::circle(Circle::new(
                    Vec2::new(3.0 + i as f32, 3.0),
                    1.0,
                )));
            }
            tx.commit();
        }
        assert_eq!(store.len(), before + 10);
        h.undo(&mut store);
        assert_eq!(store.len(), before);
    }

    #[test]
    fn layer_edits_are_undoable() {
        let (mut store, _) = store_with_circle();
        let mut layers = LayerTable::default();
        layers.ensure_default();
        let mut h = History::new();
        let new_id = {
            let mut tx = h.begin(&mut store, "New layer");
            tx.touch_layers(&layers);
            let id = layers.insert(Layer::new("Walls"));
            tx.capture_tables_after(Some(&layers), None);
            tx.commit();
            id
        };
        assert_eq!(layers.by_name("Walls"), Some(new_id));
        let c = h.pop_undo().unwrap();
        layers.restore(c.layers_before.unwrap().items);
        assert_eq!(layers.by_name("Walls"), None);
    }

    #[test]
    fn blocks_touched() {
        let (mut store, _) = store_with_circle();
        let blocks = BlockTable::new();
        let mut h = History::new();
        let mut tx = h.begin(&mut store, "Blocks");
        tx.touch_blocks(&blocks);
        assert!(tx.blocks_before().is_some());
        tx.commit();
    }

    #[test]
    fn change_stats() {
        let mut c = Change::default();
        let mut s = EntityStore::new();
        let id = s.insert(Entity::circle(Circle::new(Vec2::ZERO, 1.0)));
        let id2 = s.insert(Entity::circle(Circle::new(Vec2::new(5.0, 0.0), 1.0)));
        let id3 = s.insert(Entity::circle(Circle::new(Vec2::new(9.0, 0.0), 1.0)));
        c.before.insert(id, None);
        c.after
            .insert(id, Some(Entity::circle(Circle::new(Vec2::ZERO, 1.0))));
        c.before.insert(
            id2,
            Some(Entity::circle(Circle::new(Vec2::new(5.0, 0.0), 1.0))),
        );
        c.after.insert(id2, None);
        let old = Entity::circle(Circle::new(Vec2::new(9.0, 0.0), 1.0));
        c.before.insert(id3, Some(old.clone()));
        c.after.insert(
            id3,
            Some(old.translated(cad_core::Vec3::new(1.0, 0.0, 0.0))),
        );
        assert_eq!(c.stats(), (1, 1, 1));
    }

    #[test]
    fn disabled_history_records_nothing() {
        let (mut store, _) = store_with_circle();
        let mut h = History::new();
        h.enabled = false;
        {
            let mut tx = h.begin(&mut store, "Silent");
            tx.insert(Entity::circle(Circle::new(Vec2::ZERO, 1.0)));
            tx.commit();
        }
        assert!(!h.can_undo());
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn labels_flow_to_the_menu() {
        let (mut store, _) = store_with_circle();
        let mut h = History::new();
        {
            let mut tx = h.begin(&mut store, "Erase 3 entities");
            tx.insert(Entity::circle(Circle::new(Vec2::ZERO, 1.0)));
            tx.commit();
        }
        assert_eq!(h.next_undo_label().unwrap(), "Undo Erase 3 entities");
    }

    #[test]
    fn deep_undo_redo_cycles_stay_consistent() {
        let (mut store, id) = store_with_circle();
        let mut h = History::new();
        let mut positions = Vec::new();
        for i in 1..=20 {
            let mut tx = h.begin(&mut store, "Move");
            let e = tx.get(id).unwrap().clone();
            tx.replace(id, e.translated(cad_core::Vec3::new(1.0, 0.0, 0.0)));
            tx.commit();
            positions.push(i as f32);
        }
        assert!((circle_x(store.get(id).unwrap()) - 20.0).abs() < 1e-4);
        for expected in (0..20).rev() {
            h.undo(&mut store);
            assert!(
                (circle_x(store.get(id).unwrap()) - expected as f32).abs() < 1e-4,
                "expected {expected}"
            );
        }
        for expected in 1..=20 {
            h.redo(&mut store);
            assert!(
                (circle_x(store.get(id).unwrap()) - expected as f32).abs() < 1e-4,
                "expected {expected}"
            );
        }
    }
}
