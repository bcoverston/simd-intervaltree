//! Core interval tree data structure.

use alloc::vec::Vec;
use core::ops::ControlFlow;

use crate::builder::IntervalTreeBuilder;
use crate::query::QueryIter;
use crate::Interval;

/// An immutable interval tree optimized for overlap queries.
///
/// The tree is constructed via [`IntervalTreeBuilder`] and is immutable after
/// construction, making it `Send + Sync` by default.
///
/// # Data Layout
///
/// Data is laid out contiguously per node for SIMD-friendly scanning:
/// - Each node's intervals are stored contiguously in `starts`, `ends`, `values`
/// - Within a node, intervals are sorted by start (ascending)
/// - `by_end_indices` provides descending-by-end ordering via indirection
#[derive(Debug, Clone)]
pub struct IntervalTree<T, V> {
    /// Start bounds for all intervals (contiguous per node, sorted by start).
    pub(crate) starts: Vec<T>,
    /// End bounds for all intervals.
    pub(crate) ends: Vec<T>,
    /// Values associated with each interval.
    pub(crate) values: Vec<V>,
    /// Node structures.
    pub(crate) nodes: Vec<Node>,
    /// Indices sorted by end (descending) for each node.
    pub(crate) by_end_indices: Vec<usize>,
    /// End values sorted descending (contiguous per node, for SIMD scanning).
    pub(crate) ends_desc: Vec<T>,
}

/// A node in the interval tree.
#[derive(Debug, Clone)]
pub(crate) struct Node {
    /// Index of pivot interval in the data arrays.
    pub pivot_idx: usize,
    /// Start index of this node's intervals in data arrays (contiguous, sorted by start).
    pub data_begin: usize,
    /// End index (exclusive) of this node's intervals.
    pub data_end: usize,
    /// Start index in by_end_indices for descending-by-end ordering.
    pub by_end_begin: usize,
    /// End index (exclusive) in by_end_indices.
    pub by_end_end: usize,
    /// Index of left child node, or `usize::MAX` if none.
    pub left: usize,
    /// Index of right child node, or `usize::MAX` if none.
    pub right: usize,
}

impl Node {
    pub const NULL: usize = usize::MAX;

    #[inline]
    pub fn has_left(&self) -> bool {
        self.left != Self::NULL
    }

    #[inline]
    pub fn has_right(&self) -> bool {
        self.right != Self::NULL
    }
}

impl<T, V> IntervalTree<T, V> {
    /// Creates a new builder for constructing an interval tree.
    #[must_use]
    pub fn builder() -> IntervalTreeBuilder<T, V> {
        IntervalTreeBuilder::new()
    }

    /// Returns the number of intervals in the tree.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if the tree is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the number of nodes in the tree.
    #[inline]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl<T: Ord + Copy, V> IntervalTree<T, V> {
    /// Queries for all intervals overlapping the given range.
    ///
    /// Returns an iterator that yields entries without allocation.
    #[inline]
    pub fn query<R: Into<Interval<T>>>(&self, range: R) -> QueryIter<'_, T, V> {
        QueryIter::new(self, range.into())
    }

    /// Queries with a callback for early termination.
    ///
    /// The callback receives each overlapping interval and can return
    /// `ControlFlow::Break(result)` to stop iteration early.
    pub fn query_with<R, F, B>(&self, range: R, mut callback: F) -> ControlFlow<B>
    where
        R: Into<Interval<T>>,
        F: FnMut(&Interval<T>, &V) -> ControlFlow<B>,
    {
        let query = range.into();
        self.query_node(0, &query, &mut callback)
    }

    fn query_node<F, B>(
        &self,
        node_idx: usize,
        query: &Interval<T>,
        callback: &mut F,
    ) -> ControlFlow<B>
    where
        F: FnMut(&Interval<T>, &V) -> ControlFlow<B>,
    {
        if node_idx >= self.nodes.len() {
            return ControlFlow::Continue(());
        }

        let node = &self.nodes[node_idx];
        let pivot = self.starts[node.pivot_idx];

        // Case 1: Query contains pivot - all intervals at node overlap
        if query.start <= pivot && pivot < query.end {
            // Yield all intervals at this node (contiguous in data arrays)
            for i in node.data_begin..node.data_end {
                let interval = Interval {
                    start: self.starts[i],
                    end: self.ends[i],
                };
                callback(&interval, &self.values[i])?;
            }

            // Search both subtrees
            if node.has_left() {
                self.query_node(node.left, query, callback)?;
            }
            if node.has_right() {
                self.query_node(node.right, query, callback)?;
            }
        }
        // Case 2: Pivot is left of query - scan by end (descending), go right
        else if pivot < query.start {
            // Scan intervals sorted by end (descending) until end <= query.start
            for pos in node.by_end_begin..node.by_end_end {
                let i = self.by_end_indices[pos];
                let end = self.ends[i];
                if end <= query.start {
                    break; // No more overlaps possible
                }
                let interval = Interval {
                    start: self.starts[i],
                    end,
                };
                callback(&interval, &self.values[i])?;
            }

            if node.has_right() {
                self.query_node(node.right, query, callback)?;
            }
        }
        // Case 3: Pivot is right of query - scan by start (ascending), go left
        else {
            // Scan intervals sorted by start (ascending) until start >= query.end
            // Data is contiguous and sorted - can use SIMD here for i64
            for i in node.data_begin..node.data_end {
                let start = self.starts[i];
                if start >= query.end {
                    break; // No more overlaps possible
                }
                let interval = Interval {
                    start,
                    end: self.ends[i],
                };
                callback(&interval, &self.values[i])?;
            }

            if node.has_left() {
                self.query_node(node.left, query, callback)?;
            }
        }

        ControlFlow::Continue(())
    }
}

// SIMD-optimized query for i64 intervals
impl<V> IntervalTree<i64, V> {
    /// Queries with SIMD acceleration for i64 intervals.
    pub fn query_simd<R, F, B>(&self, range: R, mut callback: F) -> ControlFlow<B>
    where
        R: Into<Interval<i64>>,
        F: FnMut(&Interval<i64>, &V) -> ControlFlow<B>,
    {
        let query = range.into();
        self.query_node_simd(0, &query, &mut callback)
    }

    fn query_node_simd<F, B>(
        &self,
        node_idx: usize,
        query: &Interval<i64>,
        callback: &mut F,
    ) -> ControlFlow<B>
    where
        F: FnMut(&Interval<i64>, &V) -> ControlFlow<B>,
    {
        if node_idx >= self.nodes.len() {
            return ControlFlow::Continue(());
        }

        let node = &self.nodes[node_idx];
        let pivot = self.starts[node.pivot_idx];

        if query.start <= pivot && pivot < query.end {
            // Case 1: Query contains pivot - yield all
            for i in node.data_begin..node.data_end {
                let interval = Interval {
                    start: self.starts[i],
                    end: self.ends[i],
                };
                callback(&interval, &self.values[i])?;
            }

            if node.has_left() {
                self.query_node_simd(node.left, query, callback)?;
            }
            if node.has_right() {
                self.query_node_simd(node.right, query, callback)?;
            }
        } else if pivot < query.start {
            // Case 2: Scan by end descending - use SIMD to find cutoff
            let node_ends = &self.ends_desc[node.by_end_begin..node.by_end_end];
            let cutoff = crate::simd::find_le_threshold_i64(node_ends, query.start);

            for pos in node.by_end_begin..(node.by_end_begin + cutoff) {
                let i = self.by_end_indices[pos];
                let interval = Interval {
                    start: self.starts[i],
                    end: self.ends[i],
                };
                callback(&interval, &self.values[i])?;
            }

            if node.has_right() {
                self.query_node_simd(node.right, query, callback)?;
            }
        } else {
            // Case 3: Scan by start ascending - use SIMD to find cutoff
            let node_starts = &self.starts[node.data_begin..node.data_end];
            let cutoff = crate::simd::find_ge_threshold_i64(node_starts, query.end);

            for i in node.data_begin..(node.data_begin + cutoff) {
                let interval = Interval {
                    start: self.starts[i],
                    end: self.ends[i],
                };
                callback(&interval, &self.values[i])?;
            }

            if node.has_left() {
                self.query_node_simd(node.left, query, callback)?;
            }
        }

        ControlFlow::Continue(())
    }
}

// Safety: Tree is immutable after construction
unsafe impl<T: Send, V: Send> Send for IntervalTree<T, V> {}
unsafe impl<T: Sync, V: Sync> Sync for IntervalTree<T, V> {}
