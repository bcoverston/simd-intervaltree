//! Query iterators for zero-allocation traversal.

use crate::tree::IntervalTree;
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
    /// Stack of node indices for traversal.
    stack: [usize; 64], // Max depth of 64 should be plenty
    stack_len: usize,
    /// Current position within a node's interval list.
    current_pos: usize,
    current_end: usize,
    /// Which case we're in: 0 = all intervals, 1 = by-end desc, 2 = check start
    current_case: u8,
}

impl<'a, T: Ord + Copy, V> QueryIter<'a, T, V> {
    pub(crate) fn new(tree: &'a IntervalTree<T, V>, query: Interval<T>) -> Self {
        let mut iter = Self {
            tree,
            query,
            stack: [0; 64],
            stack_len: 0,
            current_pos: 0,
            current_end: 0,
            current_case: 0,
        };

        if !tree.nodes.is_empty() {
            iter.stack[0] = 0;
            iter.stack_len = 1;
        }

        iter
    }

    fn advance_to_next(&mut self) -> Option<QueryEntry<'a, T, V>> {
        loop {
            // Try to yield from current node's interval list
            while self.current_pos < self.current_end {
                let pos = self.current_pos;
                self.current_pos += 1;

                match self.current_case {
                    0 => {
                        // Case 1: All intervals overlap (iterate by start)
                        let start = self.tree.starts[pos];
                        let end = self.tree.ends[pos];
                        return Some(QueryEntry {
                            interval: Interval { start, end },
                            value: &self.tree.values[pos],
                        });
                    }
                    1 => {
                        // Case 2: By-end descending - early terminate when end <= query.start
                        let end = self.tree.ends_desc[pos];
                        if end <= self.query.start {
                            self.current_pos = self.current_end; // Skip rest
                            break;
                        }
                        let i = self.tree.by_end_indices[pos];
                        let start = self.tree.starts[i];
                        return Some(QueryEntry {
                            interval: Interval { start, end },
                            value: &self.tree.values[i],
                        });
                    }
                    2 => {
                        // Case 3: By start - early terminate when start >= query.end
                        let start = self.tree.starts[pos];
                        if start >= self.query.end {
                            self.current_pos = self.current_end; // Skip rest
                            break;
                        }
                        let end = self.tree.ends[pos];
                        return Some(QueryEntry {
                            interval: Interval { start, end },
                            value: &self.tree.values[pos],
                        });
                    }
                    _ => unreachable!(),
                }
            }

            // Pop from stack
            if self.stack_len == 0 {
                return None;
            }

            self.stack_len -= 1;
            let node_idx = self.stack[self.stack_len];

            if node_idx >= self.tree.nodes.len() {
                continue;
            }

            let node = &self.tree.nodes[node_idx];

            // Early pruning with max_end
            if node.max_end <= self.query.start {
                continue;
            }

            let pivot = node.pivot;

            if self.query.start <= pivot && pivot < self.query.end {
                // Case 1: Query contains pivot - yield all, search both
                self.current_pos = node.data_begin;
                self.current_end = node.data_end;
                self.current_case = 0;

                // Push children for later
                if node.has_right() {
                    self.stack[self.stack_len] = node.right;
                    self.stack_len += 1;
                }
                if node.has_left() {
                    self.stack[self.stack_len] = node.left;
                    self.stack_len += 1;
                }
            } else if pivot < self.query.start {
                // Case 2: Pivot left of query - use by_end_desc for early termination
                self.current_pos = node.by_end_begin;
                self.current_end = node.by_end_end;
                self.current_case = 1;

                if node.has_right() {
                    self.stack[self.stack_len] = node.right;
                    self.stack_len += 1;
                }
            } else {
                // Case 3: Pivot right of query - check start, go left
                self.current_pos = node.data_begin;
                self.current_end = node.data_end;
                self.current_case = 2;

                if node.has_left() {
                    self.stack[self.stack_len] = node.left;
                    self.stack_len += 1;
                }
            }
        }
    }
}

impl<'a, T: Ord + Copy, V> Iterator for QueryIter<'a, T, V> {
    type Item = QueryEntry<'a, T, V>;

    fn next(&mut self) -> Option<Self::Item> {
        self.advance_to_next()
    }
}
