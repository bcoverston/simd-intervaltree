//! Core interval tree data structure.

use alloc::vec::Vec;
use core::ops::{ControlFlow, Range};

use crate::builder::IntervalTreeBuilder;
use crate::query::QueryIter;
use crate::simd::{Cutoffs, Scalar, Simd};
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
}

impl<T, V> IntervalTree<T, V> {
    /// A tree holding no intervals.
    pub(crate) const fn empty() -> Self {
        Self {
            starts: Vec::new(),
            ends: Vec::new(),
            values: Vec::new(),
            nodes: Vec::new(),
            ends_desc: Vec::new(),
            by_end_indices: Vec::new(),
        }
    }

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

/// Which of a node's intervals overlap a query: a run of positions in one of
/// the tree's two orderings, sorted so the overlaps come first. The variant
/// names the condition that ends the overlapping prefix; its bound is always
/// the query's own start or end.
#[derive(Debug, Clone)]
pub(crate) enum Scan {
    /// Positions in `starts`/`ends`/`values`; every one overlaps.
    All(Range<usize>),
    /// Positions in `ends_desc`/`by_end_indices`; overlaps while the end is
    /// above `query.start`.
    EndsAbove(Range<usize>),
    /// Positions in `starts`/`ends`/`values`; overlaps while the start is
    /// below `query.end`.
    StartsBelow(Range<usize>),
}

impl Scan {
    pub const NONE: Self = Self::All(0..0);

    /// The positions still to scan, whichever ordering they index.
    const fn run(&self) -> &Range<usize> {
        match self {
            Self::All(run) | Self::EndsAbove(run) | Self::StartsBelow(run) => run,
        }
    }
}

/// What one node contributes to a query: the scan over its intervals and the
/// children that can still hold overlaps (`Node::NULL` when ruled out).
pub(crate) struct Visit {
    pub scan: Scan,
    pub left: u32,
    pub right: u32,
}

impl<T: Ord + Copy, V> IntervalTree<T, V> {
    /// Classifies one node against `query`, or returns `None` when its whole
    /// subtree ends before the query starts.
    ///
    /// Every interval stored at a node contains the node's pivot, so:
    /// - pivot inside the query: all of them overlap, and both subtrees may;
    /// - pivot left of the query: the ones ending after `query.start` overlap,
    ///   and only the right subtree may (the left one ends at or before pivot);
    /// - pivot right of the query: the ones starting before `query.end`
    ///   overlap, and only the left subtree may.
    #[inline]
    pub(crate) fn visit<C: Cutoffs<T>>(&self, node_idx: u32, query: &Interval<T>) -> Option<Visit> {
        let node = &self.nodes[node_idx as usize];
        if node.max_end <= query.start {
            return None;
        }

        let by_start = node.data_begin as usize..node.data_end as usize;
        let by_end = node.by_end_begin as usize..node.by_end_end as usize;

        let visit = if query.start <= node.pivot && node.pivot < query.end {
            Visit {
                scan: Scan::All(by_start),
                left: node.left,
                right: node.right,
            }
        } else if node.pivot < query.start {
            let len = C::first_le(&self.ends_desc[by_end.clone()], query.start);
            Visit {
                scan: Scan::EndsAbove(by_end.start..by_end.start + len),
                left: Node::<T>::NULL,
                right: node.right,
            }
        } else {
            let len = C::first_ge(&self.starts[by_start.clone()], query.end);
            Visit {
                scan: Scan::StartsBelow(by_start.start..by_start.start + len),
                left: node.left,
                right: Node::<T>::NULL,
            }
        };
        Some(visit)
    }

    /// The next overlapping interval from `scan`, advancing it; `None` once
    /// the run is exhausted or reaches the non-overlapping suffix.
    #[inline]
    pub(crate) fn next_hit(
        &self,
        scan: &mut Scan,
        query: &Interval<T>,
    ) -> Option<(Interval<T>, &V)> {
        match scan {
            Scan::All(run) => run.next().map(|pos| self.by_start(pos)),
            Scan::EndsAbove(run) => {
                let pos = run.next()?;
                if self.ends_desc[pos] <= query.start {
                    *run = pos..pos; // ends descend: nothing after overlaps
                    return None;
                }
                Some(self.by_end(pos))
            }
            Scan::StartsBelow(run) => {
                let pos = run.next()?;
                if self.starts[pos] >= query.end {
                    *run = pos..pos; // starts ascend: nothing after overlaps
                    return None;
                }
                Some(self.by_start(pos))
            }
        }
    }

    #[inline]
    fn by_start(&self, pos: usize) -> (Interval<T>, &V) {
        let interval = Interval {
            start: self.starts[pos],
            end: self.ends[pos],
        };
        (interval, &self.values[pos])
    }

    /// Reads `end` from `ends_desc[pos]`, which the scan is already streaming
    /// through; going via the index to `ends[i]` would add a random load per
    /// hit (measured ~20% on large trees).
    #[inline]
    fn by_end(&self, pos: usize) -> (Interval<T>, &V) {
        let i = self.by_end_indices[pos] as usize;
        let interval = Interval {
            start: self.starts[i],
            end: self.ends_desc[pos],
        };
        (interval, &self.values[i])
    }

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
        // An empty query range overlaps nothing.
        if query.start >= query.end || self.nodes.is_empty() {
            return ControlFlow::Continue(());
        }
        self.walk::<Scalar, _, _>(0, &query, &mut callback)
    }

    fn walk<C, F, B>(&self, node_idx: u32, query: &Interval<T>, callback: &mut F) -> ControlFlow<B>
    where
        C: Cutoffs<T>,
        F: FnMut(&Interval<T>, &V) -> ControlFlow<B>,
    {
        let Some(mut visit) = self.visit::<C>(node_idx, query) else {
            return ControlFlow::Continue(());
        };
        while let Some((interval, value)) = self.next_hit(&mut visit.scan, query) {
            callback(&interval, value)?;
        }
        for child in [visit.left, visit.right] {
            if child != Node::<T>::NULL {
                self.walk::<C, _, _>(child, query, callback)?;
            }
        }
        ControlFlow::Continue(())
    }
}

// SIMD cutoffs for i64 intervals. Rust has no specialization, so these are
// separate entry points rather than a faster path behind `query_with`.
impl<V> IntervalTree<i64, V> {
    /// Counts overlapping intervals using SIMD acceleration.
    ///
    /// Faster than `.query().count()`: each node contributes a count from its
    /// cutoff index, so no interval is visited individually.
    #[inline]
    pub fn count_overlaps<R: Into<Interval<i64>>>(&self, range: R) -> usize {
        let query = range.into();
        // An empty query range overlaps nothing.
        if query.start >= query.end || self.nodes.is_empty() {
            return 0;
        }
        self.count(0, &query)
    }

    /// `Simd` cutoffs are exact, so each trimmed run's length is its count.
    fn count(&self, node_idx: u32, query: &Interval<i64>) -> usize {
        let Some(visit) = self.visit::<Simd>(node_idx, query) else {
            return 0;
        };
        let mut count = visit.scan.run().len();
        for child in [visit.left, visit.right] {
            if child != Node::<i64>::NULL {
                count += self.count(child, query);
            }
        }
        count
    }

    /// Queries with SIMD acceleration for i64 intervals.
    ///
    /// Same results and order as [`query_with`](Self::query_with); each
    /// node's cutoff comes from the SIMD kernels rather than a per-interval
    /// comparison.
    pub fn query_simd<R, F, B>(&self, range: R, mut callback: F) -> ControlFlow<B>
    where
        R: Into<Interval<i64>>,
        F: FnMut(&Interval<i64>, &V) -> ControlFlow<B>,
    {
        let query = range.into();
        // An empty query range overlaps nothing.
        if query.start >= query.end || self.nodes.is_empty() {
            return ControlFlow::Continue(());
        }
        self.walk::<Simd, _, _>(0, &query, &mut callback)
    }
}
