//! Builder for constructing interval trees.

use alloc::vec;
use alloc::vec::Vec;

use crate::tree::{IntervalTree, Node};
use crate::Interval;

/// Builder for constructing an [`IntervalTree`].
///
/// # Example
///
/// ```
/// use simd_intervaltree::IntervalTree;
///
/// let tree = IntervalTree::builder()
///     .insert(0..10, "first")
///     .insert(5..15, "second")
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct IntervalTreeBuilder<T, V> {
    intervals: Vec<(Interval<T>, V)>,
}

impl<T, V> Default for IntervalTreeBuilder<T, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, V> IntervalTreeBuilder<T, V> {
    /// Creates a new empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            intervals: Vec::new(),
        }
    }

    /// Creates a new builder with the specified capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            intervals: Vec::with_capacity(capacity),
        }
    }

    /// Inserts an interval with its associated value.
    #[must_use]
    pub fn insert<R: Into<Interval<T>>>(mut self, range: R, value: V) -> Self {
        self.intervals.push((range.into(), value));
        self
    }

    /// Returns the number of intervals added.
    #[must_use]
    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    /// Returns true if no intervals have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }
}

impl<T: Ord + Copy, V> IntervalTreeBuilder<T, V> {
    /// Builds the interval tree.
    ///
    /// Data is laid out contiguously per node for SIMD-friendly scanning.
    /// Each node's intervals are sorted by start (ascending) in the main arrays,
    /// with a separate index array for by-end (descending) ordering.
    #[must_use]
    pub fn build(self) -> IntervalTree<T, V> {
        let n = self.intervals.len();

        if n == 0 {
            return IntervalTree {
                starts: Vec::new(),
                ends: Vec::new(),
                values: Vec::new(),
                nodes: Vec::new(),
                by_end_indices: Vec::new(),
                ends_desc: Vec::new(),
            };
        }

        // Phase 1: Build tree structure and collect intervals per node
        let mut input_starts: Vec<T> = Vec::with_capacity(n);
        let mut input_ends: Vec<T> = Vec::with_capacity(n);
        let mut input_values: Vec<V> = Vec::with_capacity(n);

        for (interval, value) in self.intervals {
            input_starts.push(interval.start);
            input_ends.push(interval.end);
            input_values.push(value);
        }

        // Build tree and collect intervals per node
        let indices: Vec<usize> = (0..n).collect();
        let mut planner = TreePlanner::new(&input_starts, &input_ends);
        planner.plan_node(&indices);

        // Phase 2: Output data in node order (contiguous per node, sorted by start)
        let mut starts: Vec<T> = Vec::with_capacity(n);
        let mut ends: Vec<T> = Vec::with_capacity(n);
        let mut values: Vec<V> = Vec::with_capacity(n);
        let mut by_end_indices: Vec<usize> = Vec::with_capacity(n);
        let mut ends_desc: Vec<T> = Vec::with_capacity(n);
        let mut nodes: Vec<Node> = Vec::with_capacity(planner.node_intervals.len());

        // We need to map old indices to new positions
        let mut index_map: Vec<usize> = vec![0; n];

        for node_data in &planner.node_intervals {
            let node_start_pos = starts.len();

            // Sort by start ascending and output
            let mut sorted_indices = node_data.containing.clone();
            sorted_indices.sort_by_key(|&i| input_starts[i]);

            for &old_idx in &sorted_indices {
                let new_idx = starts.len();
                index_map[old_idx] = new_idx;
                starts.push(input_starts[old_idx]);
                ends.push(input_ends[old_idx]);
            }

            let node_end_pos = starts.len();

            // Build by_end indices (sorted by end descending, pointing to new positions)
            // Also build ends_desc for SIMD scanning
            let mut by_end = sorted_indices.clone();
            by_end.sort_by(|&a, &b| input_ends[b].cmp(&input_ends[a]));

            let by_end_begin = by_end_indices.len();
            for &old_idx in &by_end {
                by_end_indices.push(index_map[old_idx]);
                ends_desc.push(input_ends[old_idx]);
            }
            let by_end_end = by_end_indices.len();

            // Find pivot in new positions
            let pivot_new_idx = index_map[node_data.pivot_idx];

            nodes.push(Node {
                pivot_idx: pivot_new_idx,
                data_begin: node_start_pos,
                data_end: node_end_pos,
                by_end_begin,
                by_end_end,
                left: node_data.left,
                right: node_data.right,
            });
        }

        // Move values in correct order
        // We need to iterate in the order we added to starts/ends
        let mut value_order: Vec<(usize, usize)> = index_map
            .iter()
            .enumerate()
            .map(|(old, &new)| (new, old))
            .collect();
        value_order.sort_by_key(|(new, _)| *new);

        // Use a temporary to allow moving values
        let mut temp_values: Vec<Option<V>> = input_values.into_iter().map(Some).collect();
        for (_, old_idx) in value_order {
            let val: Option<V> = temp_values[old_idx].take();
            values.push(val.unwrap());
        }

        IntervalTree {
            starts,
            ends,
            values,
            nodes,
            by_end_indices,
            ends_desc,
        }
    }
}

/// Intermediate node data during planning phase.
struct NodeData {
    pivot_idx: usize,
    containing: Vec<usize>,
    left: usize,
    right: usize,
}

/// First pass: plan tree structure without moving data.
struct TreePlanner<'a, T> {
    starts: &'a [T],
    ends: &'a [T],
    node_intervals: Vec<NodeData>,
}

impl<'a, T: Ord + Copy> TreePlanner<'a, T> {
    fn new(starts: &'a [T], ends: &'a [T]) -> Self {
        Self {
            starts,
            ends,
            node_intervals: Vec::new(),
        }
    }

    fn plan_node(&mut self, indices: &[usize]) -> usize {
        if indices.is_empty() {
            return Node::NULL;
        }

        // Find median endpoint as pivot
        let pivot_idx = self.find_median_endpoint(indices);
        let pivot = self.starts[pivot_idx];

        // Partition intervals
        let mut containing = Vec::new();
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in indices {
            let start = self.starts[idx];
            let end = self.ends[idx];

            if end <= pivot {
                left_indices.push(idx);
            } else if start > pivot {
                right_indices.push(idx);
            } else {
                containing.push(idx);
            }
        }

        // Allocate node
        let node_idx = self.node_intervals.len();
        self.node_intervals.push(NodeData {
            pivot_idx,
            containing,
            left: Node::NULL,
            right: Node::NULL,
        });

        // Build children
        let left_child = self.plan_node(&left_indices);
        let right_child = self.plan_node(&right_indices);

        self.node_intervals[node_idx].left = left_child;
        self.node_intervals[node_idx].right = right_child;

        node_idx
    }

    fn find_median_endpoint(&self, indices: &[usize]) -> usize {
        let mut endpoints: Vec<T> = Vec::with_capacity(indices.len() * 2);
        for &idx in indices {
            endpoints.push(self.starts[idx]);
            endpoints.push(self.ends[idx]);
        }
        endpoints.sort();

        let median = endpoints[endpoints.len() / 2];

        for &idx in indices {
            if self.starts[idx] <= median && median < self.ends[idx] {
                return idx;
            }
        }

        indices[0]
    }
}
