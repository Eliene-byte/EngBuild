//! The entity slab plus the uniform-grid spatial index.
//!
//! Storage is a slab (`Vec<Option<Entity>>`) so handles are stable, but free
//! slots are recycled through a FIFO queue: repeated draw/erase cycles do not
//! grow memory without bound, which matters for long interactive sessions.

use crate::entity::{Entity, EntityKind};
use crate::handle::{EntityId, HandleTable};
use cad_core::{Rect2, Vec2};

/// A uniform grid bucketed by the entity's 2D bounds.
#[derive(Debug, Default, Clone)]
pub struct SpatialIndex {
    cell: f32,
    origin: Vec2,
    cols: usize,
    rows: usize,
    buckets: Vec<Vec<EntityId>>,
    /// Entities with degenerate bounds are kept in a linear list.
    overflow: Vec<EntityId>,
    /// Stored bounds per entity, so remove/move stays O(cells touched).
    placed: Vec<Rect2>,
}

impl SpatialIndex {
    /// Build an index covering `bounds` with roughly `target_cells` buckets.
    pub fn new(bounds: Rect2, target_cells: usize) -> Self {
        let target = target_cells.max(64);
        let size = bounds.size();
        // Aim for square cells.
        let aspect = if size.y > 1e-6 { size.x / size.y } else { 1.0 };
        let cols = ((target as f32 * aspect.max(1e-3)).sqrt().max(1.0)) as usize;
        let rows = (target / cols.max(1)).max(1);
        let cell = (size.x.max(1e-6) / cols as f32).max(size.y.max(1e-6) / rows as f32);
        Self {
            cell,
            origin: bounds.min,
            cols,
            rows,
            buckets: vec![Vec::new(); cols * rows],
            overflow: Vec::new(),
            placed: Vec::new(),
        }
    }

    #[inline]
    fn cell_range(&self, r: Rect2) -> Option<(usize, usize, usize, usize)> {
        if self.cols == 0 || self.rows == 0 {
            return None;
        }
        let cx0 = ((r.min.x - self.origin.x) / self.cell).floor();
        let cy0 = ((r.min.y - self.origin.y) / self.cell).floor();
        let cx1 = ((r.max.x - self.origin.x) / self.cell).floor();
        let cy1 = ((r.max.y - self.origin.y) / self.cell).floor();
        // Bail only when the rect misses the grid entirely. A partially
        // overlapping rect must be clamped, otherwise large queries silently
        // lose every candidate.
        if cx1 < 0.0 || cy1 < 0.0 || cx0 >= self.cols as f32 || cy0 >= self.rows as f32 {
            return None;
        }
        Some((
            cx0.max(0.0) as usize,
            cy0.max(0.0) as usize,
            (cx1.max(0.0) as usize).min(self.cols - 1),
            (cy1.max(0.0) as usize).min(self.rows - 1),
        ))
    }

    /// True when the index has not been sized yet.
    #[inline]
    pub fn is_empty_grid(&self) -> bool {
        self.cols == 0 || self.rows == 0
    }

    /// Public alias: has the index been sized?
    #[inline]
    pub fn is_sized(&self) -> bool {
        !self.is_empty_grid()
    }

    fn insert_id(&mut self, id: EntityId, r: Rect2) {
        self.placed.resize(id.index() + 1, Rect2::ZERO);
        self.placed[id.index()] = r;
        match self.cell_range(r) {
            Some((x0, y0, x1, y1)) => {
                for cy in y0..=y1 {
                    for cx in x0..=x1 {
                        self.buckets[cy * self.cols + cx].push(id);
                    }
                }
            }
            None => self.overflow.push(id),
        }
    }

    fn remove_id(&mut self, id: EntityId, r: Rect2) {
        let mut found = false;
        if let Some((x0, y0, x1, y1)) = self.cell_range(r) {
            for cy in y0..=y1 {
                for cx in x0..=x1 {
                    let b = &mut self.buckets[cy * self.cols + cx];
                    if let Some(p) = b.iter().position(|v| *v == id) {
                        b.swap_remove(p);
                        found = true;
                    }
                }
            }
        }
        // `found` is false when the id was only in the overflow list, so this is
        // the one place that list is pruned.
        if !found && let Some(p) = self.overflow.iter().position(|v| *v == id) {
            self.overflow.swap_remove(p);
        }
    }

    /// Candidate entities whose bounds overlap `r`. The caller must still
    /// do the exact test - this is a broad phase, not a hit list.
    pub fn query(&self, r: Rect2) -> Vec<EntityId> {
        let mut out: Vec<EntityId> = Vec::new();
        if let Some((x0, y0, x1, y1)) = self.cell_range(r) {
            for cy in y0..=y1 {
                for cx in x0..=x1 {
                    out.extend_from_slice(&self.buckets[cy * self.cols + cx]);
                }
            }
        }
        // Entities parked outside the grid are still candidates, but only when
        // their real bounds overlap the query.
        for id in &self.overflow {
            if let Some(b) = self.placed.get(id.index())
                && b.overlaps(r)
            {
                out.push(*id);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Entities whose bounds contain `p`.
    pub fn query_point(&self, p: Vec2) -> Vec<EntityId> {
        self.query(Rect2::new(p, p))
    }

    pub fn len(&self) -> usize {
        self.placed.iter().filter(|r| !r.is_empty()).count()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Drop every entry (used when the index is rebuilt from scratch).
    pub fn clear(&mut self) {
        for b in &mut self.buckets {
            b.clear();
        }
        self.overflow.clear();
        self.placed.clear();
    }
    pub fn stats(&self) -> (usize, usize, f32) {
        let occupied = self.buckets.iter().filter(|b| !b.is_empty()).count();
        let total: usize =
            self.buckets.iter().map(|b| b.len()).sum::<usize>() + self.overflow.len();
        (occupied, total, self.cell)
    }
}

/// Everything that owns entities.
#[derive(Debug, Default, Clone)]
pub struct EntityStore {
    slab: Vec<Option<Entity>>,
    free: Vec<EntityId>,
    next_handle: HandleTable,
    index: SpatialIndex,
    /// Bumped on every change; the renderer compares it to skip rebuilds.
    pub revision: u64,
}

impl EntityStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.slab.iter().filter(|o| o.is_some()).count()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Highest slot ever used + 1.
    pub fn capacity(&self) -> usize {
        self.slab.len()
    }

    /// Insert, reusing a freed slot when one exists.
    pub fn insert(&mut self, e: Entity) -> EntityId {
        let bounds = e.index_bounds();
        // Size the index on first use so picking works without an explicit
        // rebuild pass.
        if self.index.is_empty_grid() {
            self.index = SpatialIndex::new(bounds.expand(Vec2::splat(1.0)), 1024);
        }
        let id = match self.free.pop() {
            Some(id) => {
                self.slab[id.index()] = Some(e);
                id
            }
            None => {
                let id = self.next_handle.alloc();
                self.slab.push(Some(e));
                id
            }
        };
        self.index.insert_id(id, bounds);
        self.revision = self.revision.wrapping_add(1);
        id
    }

    pub fn get(&self, id: EntityId) -> Option<&Entity> {
        self.slab.get(id.index()).and_then(|o| o.as_ref())
    }
    pub fn get_mut(&mut self, id: EntityId) -> Option<&mut Entity> {
        self.slab.get_mut(id.index()).and_then(|o| o.as_mut())
    }
    pub fn contains(&self, id: EntityId) -> bool {
        self.get(id).is_some()
    }

    pub fn remove(&mut self, id: EntityId) -> Option<Entity> {
        let old = self.slab.get_mut(id.index())?.take()?;
        // An empty old rect means the entity was never indexed, so there is
        // nothing to un-bucket.
        if let Some(r) = self.index.placed.get(id.index()).copied()
            && !r.is_empty()
        {
            self.index.remove_id(id, r);
        }
        self.free.push(id);
        self.revision = self.revision.wrapping_add(1);
        Some(old)
    }

    /// Replace an entity's payload, keeping its handle.
    pub fn replace(&mut self, id: EntityId, e: Entity) -> bool {
        if self.slab.get(id.index()).and_then(|o| o.as_ref()).is_none() {
            return false;
        }
        if let Some(old) = self.index.placed.get(id.index()).copied()
            && !old.is_empty()
        {
            self.index.remove_id(id, old);
        }
        let bounds = e.index_bounds();
        self.slab[id.index()] = Some(e);
        self.index.insert_id(id, bounds);
        self.revision = self.revision.wrapping_add(1);
        true
    }

    /// Put an entity back at a specific handle, recreating the slot if needed.
    /// This is what undo uses; it never reuses the handle for something else.
    pub fn restore(&mut self, id: EntityId, e: Entity) {
        let bounds = e.index_bounds();
        if self.slab.len() <= id.index() {
            self.slab.resize(id.index() + 1, None);
            // Do not let the allocator hand this slot out to a new entity.
            self.next_handle.reserve(id.index() + 1);
        }
        if self.slab[id.index()].is_some() {
            self.slab[id.index()] = Some(e);
            if let Some(old) = self.index.placed.get(id.index()).copied()
                && !old.is_empty()
            {
                self.index.remove_id(id, old);
            }
            self.index.insert_id(id, bounds);
        } else {
            self.slab[id.index()] = Some(e);
            self.index.insert_id(id, bounds);
        }
        self.free.retain(|f| *f != id);
        self.revision = self.revision.wrapping_add(1);
    }

    /// Replace many entities in one pass (drag operations).
    pub fn replace_many(&mut self, updates: &[(EntityId, Entity)]) {
        for (id, e) in updates {
            self.replace(*id, e.clone());
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Entity> {
        self.slab.iter().filter_map(|o| o.as_ref())
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Entity> {
        self.slab.iter_mut().filter_map(|o| o.as_mut())
    }
    /// Live handles in slot order.
    pub fn handles(&self) -> Vec<EntityId> {
        self.slab
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.as_ref().map(|_| EntityId(i as u32)))
            .collect()
    }

    pub fn index(&self) -> &SpatialIndex {
        &self.index
    }

    /// Rebuild the spatial index around the current drawing extents.
    pub fn rebuild_index(&mut self, bounds: Rect2) {
        self.index = SpatialIndex::new(bounds, 4096);
        for (i, o) in self.slab.iter().enumerate() {
            if let Some(e) = o {
                let r = e.index_bounds();
                self.index.insert_id(EntityId(i as u32), r);
            }
        }
    }

    /// Broad-phase candidates for a point.
    pub fn candidates_at(&self, p: Vec2) -> Vec<EntityId> {
        self.index.query_point(p)
    }
    /// Broad-phase candidates for a box (window selection).
    pub fn candidates_in(&self, r: Rect2) -> Vec<EntityId> {
        self.index.query(r)
    }

    /// Number of entities of a given kind (used by the UI for statistics).
    pub fn count_of(&self, pred: impl Fn(&EntityKind) -> bool) -> usize {
        self.iter().filter(|e| pred(&e.entity)).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::Vec2;
    use cad_geom::curve::{Circle, Line, Polyline};

    fn store_with_grid() -> EntityStore {
        let mut s = EntityStore::new();
        for i in 0..200 {
            let e = Entity::circle(Circle::new(Vec2::new(i as f32 * 10.0, 0.0), 2.0));
            s.insert(e);
        }
        s.rebuild_index(Rect2::from_xywh(-50.0, -50.0, 2100.0, 100.0));
        s
    }

    #[test]
    fn insert_get_remove() {
        let mut s = EntityStore::new();
        let id = s.insert(Entity::circle(Circle::new(Vec2::ZERO, 5.0)));
        assert!(s.contains(id));
        assert_eq!(s.len(), 1);
        assert!(s.remove(id).is_some());
        assert!(!s.contains(id));
        assert_eq!(s.len(), 0);
        assert!(s.get(id).is_none());
    }

    #[test]
    fn handles_are_recycled() {
        let mut s = EntityStore::new();
        let a = s.insert(Entity::circle(Circle::new(Vec2::ZERO, 1.0)));
        s.remove(a);
        let b = s.insert(Entity::circle(Circle::new(Vec2::ZERO, 1.0)));
        assert_eq!(a, b);
        assert_eq!(s.capacity(), 1);
    }

    #[test]
    fn replace_preserves_handle() {
        let mut s = EntityStore::new();
        let id = s.insert(Entity::circle(Circle::new(Vec2::ZERO, 1.0)));
        assert!(s.replace(id, Entity::circle(Circle::new(Vec2::new(10.0, 0.0), 1.0))));
        match s.get(id).unwrap().entity {
            EntityKind::Circle(c) => assert!(c.center.distance(Vec2::new(10.0, 0.0)) < 1e-6),
            _ => panic!(),
        }
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn replace_nonexistent_fails() {
        let mut s = EntityStore::new();
        assert!(!s.replace(EntityId(9), Entity::circle(Circle::new(Vec2::ZERO, 1.0))));
    }

    #[test]
    fn spatial_query_finds_the_right_circle() {
        let s = store_with_grid();
        let hits = s.candidates_at(Vec2::new(500.0, 0.0));
        assert_eq!(hits.len(), 1, "{hits:?}");
        match s.get(hits[0]).unwrap().entity {
            EntityKind::Circle(c) => assert!((c.center.x - 500.0).abs() < 1e-6),
            _ => panic!(),
        }
    }

    #[test]
    fn spatial_query_returns_candidates_not_hits() {
        let s = store_with_grid();
        // A box between two circles still returns the neighbours as candidates;
        // the exact test is the caller's job.
        let hits = s.candidates_in(Rect2::from_xywh(20.0, -1.0, 20.0, 2.0));
        assert!(hits.len() >= 2);
    }

    #[test]
    fn query_outside_the_grid_returns_overflow() {
        let mut s = EntityStore::new();
        let far = s.insert(Entity::circle(Circle::new(Vec2::new(1e6, 1e6), 1.0)));
        s.rebuild_index(Rect2::from_xywh(-10.0, -10.0, 20.0, 20.0));
        assert!(s.candidates_at(Vec2::new(1e6, 1e6)).contains(&far));
    }

    #[test]
    fn moving_an_entity_updates_the_index() {
        let mut s = store_with_grid();
        let id = EntityId(50);
        // It lives at x = 500, y = 0.
        assert!(s.candidates_at(Vec2::new(500.0, 0.0)).contains(&id));
        s.replace(
            id,
            Entity::circle(Circle::new(Vec2::new(500.0, 500.0), 2.0)),
        );
        assert!(s.candidates_at(Vec2::new(500.0, 500.0)).contains(&id));
        assert!(!s.candidates_at(Vec2::new(500.0, 0.0)).contains(&id));
    }

    #[test]
    fn stats_are_sane() {
        let s = store_with_grid();
        let (occupied, total, cell) = s.index.stats();
        assert!(occupied > 0 && occupied <= s.index.buckets.len());
        assert!(total >= 200);
        assert!(cell > 0.0);
    }

    #[test]
    fn count_of_by_kind() {
        let mut s = EntityStore::new();
        s.insert(Entity::circle(Circle::new(Vec2::ZERO, 1.0)));
        s.insert(Entity::line(Line::new(Vec2::ZERO, Vec2::X)));
        s.insert(Entity::polyline(Polyline::new(
            vec![Vec2::ZERO, Vec2::X],
            false,
        )));
        assert_eq!(s.count_of(|k| matches!(k, EntityKind::Circle(_))), 1);
        assert_eq!(
            s.count_of(|k| matches!(k, EntityKind::Line(_) | EntityKind::Polyline(_))),
            2
        );
    }

    #[test]
    fn revision_changes_on_mutation() {
        let mut s = EntityStore::new();
        let r0 = s.revision;
        let id = s.insert(Entity::circle(Circle::new(Vec2::ZERO, 1.0)));
        assert_ne!(s.revision, r0);
        let r1 = s.revision;
        s.replace(id, Entity::circle(Circle::new(Vec2::ZERO, 2.0)));
        assert_ne!(s.revision, r1);
        let r2 = s.revision;
        s.remove(id);
        assert_ne!(s.revision, r2);
    }

    #[test]
    fn many_cycles_do_not_grow_capacity() {
        let mut s = EntityStore::new();
        for _ in 0..1000 {
            let id = s.insert(Entity::circle(Circle::new(Vec2::ZERO, 1.0)));
            s.remove(id);
        }
        assert_eq!(s.capacity(), 1);
    }
}
