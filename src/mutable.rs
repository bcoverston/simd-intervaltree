//! Mutable interval collection with stable identifiers.
//!
//! This module provides a dynamic interval collection that supports
//! insertion and removal while returning stable identifiers that can
//! be used to map to external resources.

use alloc::vec::Vec;

use crate::builder::IntervalTreeBuilder;
use crate::tree::IntervalTree;
use crate::Interval;

/// A stable identifier for an interval in the collection.
///
/// These identifiers remain valid across insertions and removals,
/// making them suitable for mapping to external resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntervalId(u64);

impl IntervalId {
    /// Returns the raw identifier value.
    #[inline]
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Entry in the mutable collection.
#[derive(Debug, Clone)]
struct Entry<T, V> {
    interval: Interval<T>,
    value: V,
    generation: u32,
}

/// A mutable interval collection with stable identifiers.
///
/// Unlike [`IntervalTree`], this collection supports dynamic insertion
/// and removal. Each inserted interval receives a stable [`IntervalId`]
/// that remains valid until the interval is removed.
///
/// Internally maintains a pending buffer that gets merged into an
/// optimized tree structure on query.
///
/// # Example
///
/// ```
/// use simd_intervaltree::IntervalSet;
///
/// let mut set = IntervalSet::new();
///
/// // Insert returns stable IDs
/// let id1 = set.insert(0..10, "first");
/// let id2 = set.insert(5..15, "second");
///
/// // Query overlapping intervals with IDs
/// for (id, interval, value) in set.query(3..12) {
///     println!("{id:?}: {interval:?} => {value}");
/// }
///
/// // Remove by ID
/// set.remove(id1);
/// ```
#[derive(Debug, Clone)]
pub struct IntervalSet<T, V> {
    /// Active entries indexed by slot.
    entries: Vec<Option<Entry<T, V>>>,
    /// Free slot indices for reuse.
    free_slots: Vec<usize>,
    /// Next generation counter for ID uniqueness.
    next_generation: u32,
    /// Count of active intervals.
    count: usize,
    /// Cached tree storing slot indices (invalidated on mutation).
    cached_tree: Option<IntervalTree<T, usize>>,
}

impl<T, V> Default for IntervalSet<T, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, V> IntervalSet<T, V> {
    /// Creates a new empty interval set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            free_slots: Vec::new(),
            next_generation: 0,
            count: 0,
            cached_tree: None,
        }
    }

    /// Creates a new interval set with the specified capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            free_slots: Vec::new(),
            next_generation: 0,
            count: 0,
            cached_tree: None,
        }
    }

    /// Returns the number of intervals in the set.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the set is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Clears all intervals from the set.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.free_slots.clear();
        self.count = 0;
        self.cached_tree = None;
    }
}

impl<T: Ord + Copy, V> IntervalSet<T, V> {
    /// Inserts an interval with its associated value.
    ///
    /// Returns a stable [`IntervalId`] that can be used for removal
    /// or mapping to external resources.
    pub fn insert<R: Into<Interval<T>>>(&mut self, range: R, value: V) -> IntervalId {
        let interval = range.into();
        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);

        let entry = Entry {
            interval,
            value,
            generation,
        };

        let slot = if let Some(slot) = self.free_slots.pop() {
            self.entries[slot] = Some(entry);
            slot
        } else {
            let slot = self.entries.len();
            self.entries.push(Some(entry));
            slot
        };

        self.count += 1;
        self.cached_tree = None; // Invalidate cache

        // Encode slot and generation into ID
        IntervalId(((generation as u64) << 32) | (slot as u64))
    }

    /// Removes an interval by its ID.
    ///
    /// Returns `true` if the interval was found and removed.
    pub fn remove(&mut self, id: IntervalId) -> bool {
        let slot = (id.0 & 0xFFFF_FFFF) as usize;
        let generation = (id.0 >> 32) as u32;

        if slot >= self.entries.len() {
            return false;
        }

        if let Some(entry) = &self.entries[slot] {
            if entry.generation == generation {
                self.entries[slot] = None;
                self.free_slots.push(slot);
                self.count -= 1;
                self.cached_tree = None; // Invalidate cache
                return true;
            }
        }

        false
    }

    /// Returns the value associated with an interval ID, if it exists.
    #[must_use]
    pub fn get(&self, id: IntervalId) -> Option<&V> {
        let slot = (id.0 & 0xFFFF_FFFF) as usize;
        let generation = (id.0 >> 32) as u32;

        self.entries.get(slot).and_then(|e| {
            e.as_ref()
                .filter(|entry| entry.generation == generation)
                .map(|entry| &entry.value)
        })
    }

    /// Returns the interval associated with an ID, if it exists.
    #[must_use]
    pub fn get_interval(&self, id: IntervalId) -> Option<Interval<T>> {
        let slot = (id.0 & 0xFFFF_FFFF) as usize;
        let generation = (id.0 >> 32) as u32;

        self.entries.get(slot).and_then(|e| {
            e.as_ref()
                .filter(|entry| entry.generation == generation)
                .map(|entry| entry.interval)
        })
    }

    /// Rebuilds the internal tree if needed.
    fn ensure_tree(&mut self) {
        if self.cached_tree.is_some() {
            return;
        }

        let mut builder = IntervalTreeBuilder::with_capacity(self.count);

        for (slot, entry) in self.entries.iter().enumerate() {
            if let Some(e) = entry {
                builder = builder.insert(e.interval, slot);
            }
        }

        self.cached_tree = Some(builder.build());
    }

    /// Queries for all intervals overlapping the given range.
    ///
    /// Returns an iterator yielding `(IntervalId, Interval<T>, &V)` tuples.
    /// This may trigger an internal rebuild if the collection has been modified.
    pub fn query<R: Into<Interval<T>>>(
        &mut self,
        range: R,
    ) -> impl Iterator<Item = (IntervalId, Interval<T>, &V)> {
        self.ensure_tree();
        let entries = &self.entries;
        self.cached_tree
            .as_ref()
            .unwrap()
            .query(range)
            .filter_map(move |entry| {
                let slot = *entry.value;
                entries.get(slot).and_then(|e| {
                    e.as_ref().map(|ent| {
                        let id = IntervalId(((ent.generation as u64) << 32) | (slot as u64));
                        (id, ent.interval, &ent.value)
                    })
                })
            })
    }

    /// Returns an iterator over all intervals and their IDs.
    pub fn iter(&self) -> impl Iterator<Item = (IntervalId, Interval<T>, &V)> {
        self.entries.iter().enumerate().filter_map(|(slot, entry)| {
            entry.as_ref().map(|e| {
                let id = IntervalId(((e.generation as u64) << 32) | (slot as u64));
                (id, e.interval, &e.value)
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_query() {
        let mut set = IntervalSet::new();

        let id1 = set.insert(0..10, "first");
        let id2 = set.insert(5..15, "second");
        let _id3 = set.insert(20..30, "third");

        assert_eq!(set.len(), 3);

        let results: Vec<_> = set.query(3..12).collect();
        assert_eq!(results.len(), 2);

        // Verify IDs are returned with query results
        let ids: Vec<_> = results.iter().map(|(id, _, _)| *id).collect();
        assert!(ids.contains(&id1) || ids.contains(&id2));

        assert_eq!(set.get(id1), Some(&"first"));
        assert_eq!(set.get(id2), Some(&"second"));
    }

    #[test]
    fn remove_by_id() {
        let mut set = IntervalSet::new();

        let id1 = set.insert(0..10, "first");
        let id2 = set.insert(5..15, "second");

        assert!(set.remove(id1));
        assert_eq!(set.len(), 1);
        assert_eq!(set.get(id1), None);
        assert_eq!(set.get(id2), Some(&"second"));

        // Double remove returns false
        assert!(!set.remove(id1));
    }

    #[test]
    fn slot_reuse() {
        let mut set = IntervalSet::new();

        let id1 = set.insert(0..10, "first");
        set.remove(id1);

        let id2 = set.insert(20..30, "second");

        // Same slot, different generation - old ID invalid
        assert_eq!(set.get(id1), None);
        assert_eq!(set.get(id2), Some(&"second"));
    }

    #[test]
    fn iter_all() {
        let mut set: IntervalSet<i32, &str> = IntervalSet::new();

        set.insert(0..10, "a");
        set.insert(5..15, "b");
        set.insert(20..30, "c");

        let items: Vec<_> = set.iter().collect();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn query_returns_ids() {
        let mut set = IntervalSet::new();

        let id1 = set.insert(0..10, "first");
        let _id2 = set.insert(100..200, "second");

        let results: Vec<_> = set.query(5..8).collect();
        assert_eq!(results.len(), 1);

        let (returned_id, interval, value) = &results[0];
        assert_eq!(*returned_id, id1);
        assert_eq!(interval.start, 0);
        assert_eq!(interval.end, 10);
        assert_eq!(*value, &"first");
    }
}
