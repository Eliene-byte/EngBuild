//! Stable, densely-packed identifiers.
//!
//! Handles are indices into a slab that never shrinks during an edit session,
//! so a handle stays valid across undo/redo and file round-trips.

use std::fmt;

macro_rules! id_type {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        #[repr(transparent)]
        pub struct $name(pub u32);

        impl $name {
            #[inline]
            pub const fn index(self) -> usize {
                self.0 as usize
            }
            #[inline]
            pub const fn raw(self) -> u32 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

id_type!(EntityId, "Identifier of a drawing entity.");
id_type!(LayerId, "Identifier of a layer.");
id_type!(BlockId, "Identifier of a block definition.");

/// Sentinel meaning "no entity selected".
pub const NO_ENTITY: EntityId = EntityId(u32::MAX);

/// A per-document allocator of handles that can be snapshotted and restored.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HandleTable {
    next: u32,
}

impl HandleTable {
    #[inline]
    pub fn alloc(&mut self) -> EntityId {
        let id = EntityId(self.next);
        self.next = self.next.wrapping_add(1);
        id
    }
    #[inline]
    pub fn peek(&self) -> EntityId {
        EntityId(self.next)
    }
    #[inline]
    pub fn set_next(&mut self, n: u32) {
        self.next = n;
    }
    /// Handle the document would hand out after `used` live entities.
    #[inline]
    pub fn reserve(&mut self, n: usize) {
        self.next = self.next.max(n as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_dense_and_ordered() {
        let mut t = HandleTable::default();
        let a = t.alloc();
        let b = t.alloc();
        assert_eq!(a, EntityId(0));
        assert_eq!(b, EntityId(1));
        assert!(a < b);
        assert_eq!(a.index(), 0);
    }

    #[test]
    fn snapshot_and_restore() {
        let mut t = HandleTable::default();
        t.alloc();
        t.alloc();
        let saved = t.clone();
        t.alloc();
        assert_eq!(t.peek(), EntityId(3));
        t = saved;
        assert_eq!(t.peek(), EntityId(2));
        t.set_next(2);
        assert_eq!(t.peek(), EntityId(2));
        t.reserve(5);
        assert_eq!(t.peek(), EntityId(5));
    }

    #[test]
    fn no_entity_sentinel_is_last() {
        assert!(EntityId(5) < NO_ENTITY);
    }
}
