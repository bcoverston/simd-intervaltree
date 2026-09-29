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
/// An ID packs a slot index and a 32-bit generation. Slots are reused after
/// removal; the generation tells a stale ID from the slot's new occupant.
/// The generation counter wraps after 2³² insertions, so an ID held across
/// that many insertions could match a later interval in the same slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntervalId(u64);

impl IntervalId {
    const fn new(slot: usize, generation: u32) -> Self {
        Self(((generation as u64) << 32) | slot as u64)
    }

    const fn slot(self) -> usize {
        (self.0 & 0xFFFF_FFFF) as usize
    }

    const fn generation(self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Returns the raw identifier value.
    #[inline]
    #[must_use]
    pub const fn as_u64(self) -> u64 {
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

/// Rebuild once this many pending inserts and removals have accumulated.
///
/// Queries scan the pending inserts linearly, and each rebuild costs
/// O(n log n), so √n balances the two; the floor keeps small sets from
/// rebuilding constantly.
fn rebuild_threshold(tree_len: usize) -> usize {
    tree_len.isqrt().max(64)
}

/// A mutable interval collection with stable identifiers.
///
/// Unlike [`IntervalTree`], this collection supports dynamic insertion
/// and removal. Each inserted interval receives a stable [`IntervalId`]
/// that remains valid until the interval is removed.
///
/// Internally it keeps an [`IntervalTree`] of the intervals present at the
/// last rebuild, plus a pending list of later insertions that queries scan
/// linearly. Removed intervals stay in the tree and are filtered out. Once
/// pending insertions and removals together exceed max(64, √n), the next
/// insert or remove rebuilds the tree in O(n log n); queries never rebuild.
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
    /// Intervals present at the last rebuild. Values are full IDs rather
    /// than slots, so a slot reused since then is not mistaken for its old
    /// occupant.
    tree: IntervalTree<T, IntervalId>,
    /// Insertions since the last rebuild.
    pending: Vec<IntervalId>,
    /// Removals since the last rebuild; their IDs linger in `tree` or
    /// `pending` until then.
    removed_since_rebuild: usize,
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
        Self::with_capacity(0)
    }

    /// Creates a new interval set with the specified capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            free_slots: Vec::new(),
            next_generation: 0,
            count: 0,
            tree: IntervalTree::empty(),
            pending: Vec::new(),
            removed_since_rebuild: 0,
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
    ///
    /// The generation counter keeps running, so IDs issued before the clear
    /// stay invalid.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.free_slots.clear();
        self.count = 0;
        self.tree = IntervalTree::empty();
        self.pending.clear();
        self.removed_since_rebuild = 0;
    }

    /// The live entry for `id`, or `None` if it was removed.
    fn entry(&self, id: IntervalId) -> Option<&Entry<T, V>> {
        self.entries
            .get(id.slot())?
            .as_ref()
            .filter(|entry| entry.generation == id.generation())
    }

    /// Returns the value associated with an interval ID, if it exists.
    #[must_use]
    pub fn get(&self, id: IntervalId) -> Option<&V> {
        self.entry(id).map(|entry| &entry.value)
    }

    /// Returns an iterator over all intervals and their IDs.
    pub fn iter(&self) -> impl Iterator<Item = (IntervalId, Interval<T>, &V)>
    where
        T: Copy,
    {
        self.entries.iter().enumerate().filter_map(|(slot, entry)| {
            entry.as_ref().map(|e| {
                let id = IntervalId::new(slot, e.generation);
                (id, e.interval, &e.value)
            })
        })
    }
}

impl<T: Ord + Copy, V> IntervalSet<T, V> {
    /// Inserts an interval with its associated value.
    ///
    /// Returns a stable [`IntervalId`] that can be used for removal
    /// or mapping to external resources.
    pub fn insert<R: Into<Interval<T>>>(&mut self, range: R, value: V) -> IntervalId {
        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);

        let entry = Entry {
            interval: range.into(),
            value,
            generation,
        };

        let slot = if let Some(slot) = self.free_slots.pop() {
            self.entries[slot] = Some(entry);
            slot
        } else {
            self.entries.push(Some(entry));
            self.entries.len() - 1
        };
        self.count += 1;

        let id = IntervalId::new(slot, generation);
        self.pending.push(id);
        self.rebuild_if_due();
        id
    }

    /// Removes an interval by its ID.
    ///
    /// Returns `true` if the interval was found and removed.
    pub fn remove(&mut self, id: IntervalId) -> bool {
        if self.entry(id).is_none() {
            return false;
        }
        self.entries[id.slot()] = None;
        self.free_slots.push(id.slot());
        self.count -= 1;
        self.removed_since_rebuild += 1;
        self.rebuild_if_due();
        true
    }

    /// Returns the interval associated with an ID, if it exists.
    #[must_use]
    pub fn get_interval(&self, id: IntervalId) -> Option<Interval<T>> {
        self.entry(id).map(|entry| entry.interval)
    }

    /// Queries for all intervals overlapping the given range.
    ///
    /// Returns an iterator yielding `(IntervalId, Interval<T>, &V)` tuples.
    /// Intervals inserted since the last internal rebuild come after those
    /// from the tree.
    pub fn query<R: Into<Interval<T>>>(
        &self,
        range: R,
    ) -> impl Iterator<Item = (IntervalId, Interval<T>, &V)> {
        let query = range.into();
        let from_tree = self.tree.query(query).map(|hit| *hit.value);
        let from_pending = self.pending.iter().copied().filter(move |&id| {
            self.entry(id)
                .is_some_and(|entry| entry.interval.overlaps(&query))
        });
        from_tree
            .chain(from_pending)
            .filter_map(move |id| self.entry(id).map(|e| (id, e.interval, &e.value)))
    }

    fn rebuild_if_due(&mut self) {
        let churn = self.pending.len() + self.removed_since_rebuild;
        if churn <= rebuild_threshold(self.tree.len()) {
            return;
        }

        let mut builder = IntervalTreeBuilder::with_capacity(self.count);
        for (slot, entry) in self.entries.iter().enumerate() {
            if let Some(e) = entry {
                builder = builder.insert(e.interval, IntervalId::new(slot, e.generation));
            }
        }
        self.tree = builder.build();
        self.pending.clear();
        self.removed_since_rebuild = 0;
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
