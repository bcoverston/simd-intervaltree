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
/// - `ends_desc` provides a separate copy sorted by end (descending) for fast queries
/// - Each node stores `max_end` for efficient subtree pruning
#[derive(Debug, Clone)]
pub struct IntervalTree<T, V> {
    /// Start bounds for all intervals (contiguous per node, sorted by start).
    pub(crate) starts: Vec<T>,
    /// End bounds for all intervals.
    pub(crate) ends: Vec<T>,
    /// Values associated with each interval.
    pub(crate) values: Vec<V>,
    /// Node structures.
    pub(crate) nodes: Vec<Node<T>>,
    /// End values sorted descending (contiguous per node, for SIMD scanning).
    pub(crate) ends_desc: Vec<T>,
    /// Indices into starts/ends/values for by-end ordering.
    pub(crate) by_end_indices: Vec<u32>,
}

/// A node in the interval tree.
///
/// Indices are `u32` rather than `usize`: a tree holds at most `u32::MAX - 1`
/// intervals (enforced by the builder), and the narrower fields keep nodes
/// small so more of the traversal metadata stays in cache.
#[derive(Debug, Clone)]
pub(crate) struct Node<T> {
    /// The pivot value used for partitioning.
    pub pivot: T,
    /// Maximum end value in this subtree (for pruning).
    pub max_end: T,
    /// Start index of this node's intervals in data arrays.
    pub data_begin: u32,
    /// End index (exclusive) of this node's intervals.
    pub data_end: u32,
    /// Start index in by_end arrays.
    pub by_end_begin: u32,
    /// End index (exclusive) in by_end arrays.
    pub by_end_end: u32,
    /// Index of left child node, or `u32::MAX` if none.
    pub left: u32,
    /// Index of right child node, or `u32::MAX` if none.
    pub right: u32,
}

impl<T> Node<T> {
    pub const NULL: u32 = u32::MAX;

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
        node_idx: u32,
        query: &Interval<T>,
        callback: &mut F,
    ) -> ControlFlow<B>
    where
        F: FnMut(&Interval<T>, &V) -> ControlFlow<B>,
    {
        if node_idx as usize >= self.nodes.len() {
            return ControlFlow::Continue(());
        }

        let node = &self.nodes[node_idx as usize];

        // Early pruning: if max_end <= query.start, no overlaps possible in this subtree
        if node.max_end <= query.start {
            return ControlFlow::Continue(());
        }

        let pivot = node.pivot;

        // Case 1: Query contains pivot - all intervals at node overlap
        if query.start <= pivot && pivot < query.end {
            for i in node.data_begin as usize..node.data_end as usize {
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
        // Case 2: Pivot is left of query - scan by end descending, go right
        else if pivot < query.start {
            // Use ends_desc for early termination
            for pos in node.by_end_begin as usize..node.by_end_end as usize {
                let end = self.ends_desc[pos];
                if end <= query.start {
                    break;
                }
                let i = self.by_end_indices[pos] as usize;
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
            for i in node.data_begin as usize..node.data_end as usize {
                let start = self.starts[i];
                if start >= query.end {
                    break;
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
    /// Counts overlapping intervals using SIMD acceleration.
    ///
    /// This is significantly faster than `.query().count()` as it uses SIMD
    /// to find cutoff points and counts without yielding individual intervals.
    #[inline]
    pub fn count_overlaps<R: Into<Interval<i64>>>(&self, range: R) -> usize {
        let query = range.into();
        self.count_node_simd(0, &query)
    }

    fn count_node_simd(&self, node_idx: u32, query: &Interval<i64>) -> usize {
        if node_idx as usize >= self.nodes.len() {
            return 0;
        }

        let node = &self.nodes[node_idx as usize];

        // Early pruning with max_end
        if node.max_end <= query.start {
            return 0;
        }

        let pivot = node.pivot;

        if query.start <= pivot && pivot < query.end {
            // Case 1: Query contains pivot - all intervals at node overlap
            let count = (node.data_end - node.data_begin) as usize;
            let left_count = if node.has_left() {
                self.count_node_simd(node.left, query)
            } else {
                0
            };
            let right_count = if node.has_right() {
                self.count_node_simd(node.right, query)
            } else {
                0
            };
            count + left_count + right_count
        } else if pivot < query.start {
            // Case 2: Pivot left of query - use SIMD on ends_desc
            let node_ends = &self.ends_desc[node.by_end_begin as usize..node.by_end_end as usize];
            let cutoff = crate::simd::find_le_threshold_i64(node_ends, query.start);
            let count = cutoff;

            let right_count = if node.has_right() {
                self.count_node_simd(node.right, query)
            } else {
                0
            };
            count + right_count
        } else {
            // Case 3: Pivot right of query - use SIMD on starts
            let node_starts = &self.starts[node.data_begin as usize..node.data_end as usize];
            let cutoff = crate::simd::find_ge_threshold_i64(node_starts, query.end);
            let count = cutoff;

            let left_count = if node.has_left() {
                self.count_node_simd(node.left, query)
            } else {
                0
            };
            count + left_count
        }
    }

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
        node_idx: u32,
        query: &Interval<i64>,
        callback: &mut F,
    ) -> ControlFlow<B>
    where
        F: FnMut(&Interval<i64>, &V) -> ControlFlow<B>,
    {
        if node_idx as usize >= self.nodes.len() {
            return ControlFlow::Continue(());
        }

        let node = &self.nodes[node_idx as usize];

        // Early pruning with max_end
        if node.max_end <= query.start {
            return ControlFlow::Continue(());
        }

        let pivot = node.pivot;

        if query.start <= pivot && pivot < query.end {
            // Case 1: Query contains pivot - yield all
            for i in node.data_begin as usize..node.data_end as usize {
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
            // Case 2: Use SIMD to find cutoff in ends_desc
            let by_end_begin = node.by_end_begin as usize;
            let node_ends = &self.ends_desc[by_end_begin..node.by_end_end as usize];
            let cutoff = crate::simd::find_le_threshold_i64(node_ends, query.start);

            for pos in by_end_begin..(by_end_begin + cutoff) {
                let i = self.by_end_indices[pos] as usize;
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
            // Case 3: Use SIMD to find cutoff by start
            let data_begin = node.data_begin as usize;
            let node_starts = &self.starts[data_begin..node.data_end as usize];
            let cutoff = crate::simd::find_ge_threshold_i64(node_starts, query.end);

            for i in data_begin..(data_begin + cutoff) {
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
