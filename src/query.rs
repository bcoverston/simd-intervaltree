//! Query iterators for zero-allocation traversal.

use crate::tree::{IntervalTree, Node};
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
/// This iterator does not allocate. It maintains a stack on the stack
/// for tree traversal (bounded by tree depth).
pub struct QueryIter<'a, T, V> {
    tree: &'a IntervalTree<T, V>,
    query: Interval<T>,
    /// Stack of (node_index, phase) for traversal.
    /// Phase: 0 = process node, 1 = process right child
    stack: [(usize, u8); 64], // Max depth of 64 should be plenty
    stack_len: usize,
    /// Current position within a node's interval list.
    current_node: usize,
    current_pos: usize,
    current_end: usize,
    /// Which list we're iterating: 0 = by_start (data array), 1 = by_end (indirect)
    current_list: u8,
}

impl<'a, T: Ord + Copy, V> QueryIter<'a, T, V> {
    pub(crate) fn new(tree: &'a IntervalTree<T, V>, query: Interval<T>) -> Self {
        let mut iter = Self {
            tree,
            query,
            stack: [(0, 0); 64],
            stack_len: 0,
            current_node: Node::NULL,
            current_pos: 0,
            current_end: 0,
            current_list: 0,
        };

        if !tree.nodes.is_empty() {
            iter.stack[0] = (0, 0);
            iter.stack_len = 1;
        }

        iter
    }

    fn advance_to_next(&mut self) -> Option<QueryEntry<'a, T, V>> {
        loop {
            // First, try to yield from current node's interval list
            while self.current_pos < self.current_end {
                let pos = self.current_pos;
                self.current_pos += 1;

                // Get the actual index
                let idx = if self.current_list == 0 {
                    // Direct access - data is contiguous per node
                    pos
                } else {
                    // Indirect access through by_end_indices
                    self.tree.by_end_indices[pos]
                };

                let start = self.tree.starts[idx];
                let end = self.tree.ends[idx];

                // Check early termination based on which list we're in
                if self.current_list == 0 {
                    // Sorted by start ascending
                    if start >= self.query.end {
                        self.current_pos = self.current_end; // Skip rest
                        break;
                    }
                } else {
                    // Sorted by end descending
                    if end <= self.query.start {
                        self.current_pos = self.current_end; // Skip rest
                        break;
                    }
                }

                // This interval overlaps
                return Some(QueryEntry {
                    interval: Interval { start, end },
                    value: &self.tree.values[idx],
                });
            }

            // Pop from stack
            if self.stack_len == 0 {
                return None;
            }

            self.stack_len -= 1;
            let (node_idx, phase) = self.stack[self.stack_len];

            if node_idx >= self.tree.nodes.len() {
                continue;
            }

            let node = &self.tree.nodes[node_idx];
            let pivot = self.tree.starts[node.pivot_idx];

            match phase {
                0 => {
                    // Determine which case we're in
                    if self.query.start <= pivot && pivot < self.query.end {
                        // Case 1: Query contains pivot - yield all, search both
                        self.current_node = node_idx;
                        self.current_pos = node.data_begin;
                        self.current_end = node.data_end;
                        self.current_list = 0;

                        // Push children for later
                        if node.has_right() {
                            self.stack[self.stack_len] = (node.right, 0);
                            self.stack_len += 1;
                        }
                        if node.has_left() {
                            self.stack[self.stack_len] = (node.left, 0);
                            self.stack_len += 1;
                        }
                    } else if pivot < self.query.start {
                        // Case 2: Pivot left of query - scan by end, go right
                        self.current_node = node_idx;
                        self.current_pos = node.by_end_begin;
                        self.current_end = node.by_end_end;
                        self.current_list = 1;

                        if node.has_right() {
                            self.stack[self.stack_len] = (node.right, 0);
                            self.stack_len += 1;
                        }
                    } else {
                        // Case 3: Pivot right of query - scan by start, go left
                        self.current_node = node_idx;
                        self.current_pos = node.data_begin;
                        self.current_end = node.data_end;
                        self.current_list = 0;

                        if node.has_left() {
                            self.stack[self.stack_len] = (node.left, 0);
                            self.stack_len += 1;
                        }
                    }
                }
                _ => continue,
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
