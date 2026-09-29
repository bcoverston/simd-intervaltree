//! Query iterators for zero-allocation traversal.

use crate::simd::Scalar;
use crate::tree::{IntervalTree, Node, Scan};
use crate::Interval;

/// An entry returned by query iteration.
#[derive(Debug, Clone, Copy)]
pub struct QueryEntry<'a, T, V> {
    /// The interval.
    pub interval: Interval<T>,
    /// Reference to the associated value.
    pub value: &'a V,
}

/// Iterator over intervals overlapping a query range.
///
/// This iterator does not allocate. It uses a fixed-size inline array
/// for tree traversal (bounded by tree depth).
pub struct QueryIter<'a, T, V> {
    tree: &'a IntervalTree<T, V>,
    query: Interval<T>,
    /// Nodes still to visit.
    stack: [u32; 64],
    stack_len: usize,
    /// The current node's intervals not yet yielded.
    scan: Scan,
}

impl<'a, T: Ord + Copy, V> QueryIter<'a, T, V> {
    pub(crate) fn new(tree: &'a IntervalTree<T, V>, query: Interval<T>) -> Self {
        let mut iter = Self {
            tree,
            query,
            stack: [0; 64],
            stack_len: 0,
            scan: Scan::NONE,
        };

        // An empty query range overlaps nothing; leave the stack empty.
        if !tree.nodes.is_empty() && query.start < query.end {
            iter.push(0);
        }

        iter
    }

    /// Every level's partitions hold at most half the parent's intervals, so
    /// depth <= log2(n) + 1, and the stack holds at most one pending sibling
    /// per level: far below 64 for any tree the builder accepts.
    fn push(&mut self, node_idx: u32) {
        debug_assert!(self.stack_len < self.stack.len());
        self.stack[self.stack_len] = node_idx;
        self.stack_len += 1;
    }
}

impl<'a, T: Ord + Copy, V> Iterator for QueryIter<'a, T, V> {
    type Item = QueryEntry<'a, T, V>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((interval, value)) = self.tree.next_hit(&mut self.scan, &self.query) {
                return Some(QueryEntry { interval, value });
            }

            if self.stack_len == 0 {
                return None;
            }
            self.stack_len -= 1;
            let node_idx = self.stack[self.stack_len];

            if let Some(visit) = self.tree.visit::<Scalar>(node_idx, &self.query) {
                self.scan = visit.scan;
                // Right first so the left subtree is visited first, matching
                // `query_with`.
                for child in [visit.right, visit.left] {
                    if child != Node::<T>::NULL {
                        self.push(child);
                    }
                }
            }
        }
    }
}
