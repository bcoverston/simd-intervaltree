//! Builder for constructing interval trees.

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
    /// Uses O(n log n) construction:
    /// 1. Sort intervals by start once
    /// 2. Build tree with in-place partitioning
    /// 3. Compute max_end bottom-up
    /// 4. Layout data contiguously per node (values placed in the same pass)
    /// 5. Build by-end sorted arrays for efficient queries
    ///
    /// # Panics
    ///
    /// Panics if more than `u32::MAX - 1` intervals were inserted (node
    /// indices are stored as `u32` to keep the tree compact).
    #[must_use]
    pub fn build(mut self) -> IntervalTree<T, V> {
        let n = self.intervals.len();
        assert!(
            n < u32::MAX as usize,
            "IntervalTree supports at most u32::MAX - 1 intervals"
        );

        if n == 0 {
            return IntervalTree {
                starts: Vec::new(),
                ends: Vec::new(),
                values: Vec::new(),
                nodes: Vec::new(),
                ends_desc: Vec::new(),
                by_end_indices: Vec::new(),
            };
        }

        // Phase 1: Sort by start once - O(n log n)
        self.intervals
            .sort_unstable_by(|(a, _), (b, _)| a.start.cmp(&b.start));

        // Extract into SOA format
        let mut input_starts: Vec<T> = Vec::with_capacity(n);
        let mut input_ends: Vec<T> = Vec::with_capacity(n);
        let mut input_values: Vec<V> = Vec::with_capacity(n);

        for (interval, value) in self.intervals {
            input_starts.push(interval.start);
            input_ends.push(interval.end);
            input_values.push(value);
        }

        // Phase 2: Build tree structure
        let mut indices: Vec<usize> = (0..n).collect();
        let mut node_data: Vec<NodeBuildData<T>> = Vec::new();
        let mut scratch = PartitionScratch::with_capacity(n);

        build_node_recursive(
            &input_starts,
            &input_ends,
            &mut indices,
            0,
            n,
            &mut node_data,
            &mut scratch,
        );

        // Phase 3: Compute max_end bottom-up
        compute_max_end(&input_ends, &indices, &mut node_data);

        // Phase 4: Layout data contiguously per node.
        // Values are placed in the same pass, in the same order as
        // starts/ends, so no separate permutation (or its extra sort) is
        // needed.
        let mut starts: Vec<T> = Vec::with_capacity(n);
        let mut ends: Vec<T> = Vec::with_capacity(n);
        let mut values: Vec<V> = Vec::with_capacity(n);
        let mut nodes: Vec<Node<T>> = Vec::with_capacity(node_data.len());

        let mut temp_values: Vec<Option<V>> = input_values.into_iter().map(Some).collect();

        for data in &node_data {
            let node_start_pos = starts.len() as u32;

            for &old_idx in &indices[data.indices_begin..data.indices_end] {
                starts.push(input_starts[old_idx]);
                ends.push(input_ends[old_idx]);
                values.push(temp_values[old_idx].take().unwrap());
            }

            let node_end_pos = starts.len() as u32;

            nodes.push(Node {
                pivot: data.pivot,
                max_end: data.max_end,
                data_begin: node_start_pos,
                data_end: node_end_pos,
                by_end_begin: 0, // Will be filled in phase 5
                by_end_end: 0,
                left: data.left,
                right: data.right,
            });
        }

        // Phase 5: Build by-end sorted arrays for each node
        // Pre-allocate all output space, then sort indices in-place per node
        let mut ends_desc: Vec<T> = Vec::with_capacity(n);
        let mut by_end_indices: Vec<u32> = Vec::with_capacity(n);

        // Single scratch buffer for sorting indices
        let mut sort_indices: Vec<u32> = Vec::with_capacity(n);

        for node in &mut nodes {
            let by_end_begin = ends_desc.len() as u32;

            // Collect indices into scratch buffer
            sort_indices.clear();
            sort_indices.extend(node.data_begin..node.data_end);

            // Sort indices by their end value (descending)
            sort_indices.sort_unstable_by(|&a, &b| ends[b as usize].cmp(&ends[a as usize]));

            // Write sorted data to output arrays
            for &idx in &sort_indices[..] {
                by_end_indices.push(idx);
                ends_desc.push(ends[idx as usize]);
            }

            node.by_end_begin = by_end_begin;
            node.by_end_end = ends_desc.len() as u32;
        }

        IntervalTree {
            starts,
            ends,
            values,
            nodes,
            ends_desc,
            by_end_indices,
        }
    }
}

/// Scratch space for partitioning
struct PartitionScratch {
    left: Vec<usize>,
    containing: Vec<usize>,
    right: Vec<usize>,
}

impl PartitionScratch {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            left: Vec::with_capacity(capacity),
            containing: Vec::with_capacity(capacity),
            right: Vec::with_capacity(capacity),
        }
    }

    fn clear(&mut self) {
        self.left.clear();
        self.containing.clear();
        self.right.clear();
    }
}

/// Intermediate node data during build
struct NodeBuildData<T> {
    pivot: T,
    max_end: T, // Will be computed in phase 3
    indices_begin: usize,
    indices_end: usize,
    left: u32,
    right: u32,
}

/// Recursively build tree nodes
fn build_node_recursive<T: Ord + Copy>(
    starts: &[T],
    ends: &[T],
    indices: &mut [usize],
    lo: usize,
    hi: usize,
    nodes: &mut Vec<NodeBuildData<T>>,
    scratch: &mut PartitionScratch,
) -> u32 {
    if lo >= hi {
        return Node::<T>::NULL;
    }

    let count = hi - lo;

    // Select pivot: median start value (O(1) since sorted)
    let mid = lo + count / 2;
    let pivot_idx = indices[mid];
    let pivot = starts[pivot_idx];

    // Partition: left (end <= pivot), containing (spans pivot), right (start > pivot)
    scratch.clear();

    for &idx in &indices[lo..hi] {
        let start = starts[idx];
        let end = ends[idx];

        if end <= pivot {
            scratch.left.push(idx);
        } else if start > pivot {
            scratch.right.push(idx);
        } else {
            scratch.containing.push(idx);
        }
    }

    // Write partitioned indices back
    let left_end = lo + scratch.left.len();
    let containing_end = left_end + scratch.containing.len();

    indices[lo..left_end].copy_from_slice(&scratch.left);
    indices[left_end..containing_end].copy_from_slice(&scratch.containing);
    indices[containing_end..hi].copy_from_slice(&scratch.right);

    // Allocate node (max_end will be filled in later)
    let node_idx = nodes.len();
    nodes.push(NodeBuildData {
        pivot,
        max_end: pivot, // Placeholder, computed in phase 3
        indices_begin: left_end,
        indices_end: containing_end,
        left: Node::<T>::NULL,
        right: Node::<T>::NULL,
    });

    // Recurse on children
    let left_child = build_node_recursive(starts, ends, indices, lo, left_end, nodes, scratch);
    let right_child =
        build_node_recursive(starts, ends, indices, containing_end, hi, nodes, scratch);

    nodes[node_idx].left = left_child;
    nodes[node_idx].right = right_child;

    node_idx as u32
}

/// Compute max_end for each node bottom-up
fn compute_max_end<T: Ord + Copy>(ends: &[T], indices: &[usize], nodes: &mut [NodeBuildData<T>]) {
    // Process nodes in reverse order (children before parents due to DFS construction)
    for i in (0..nodes.len()).rev() {
        let node = &nodes[i];

        // Find max end among this node's intervals
        let mut max_end = ends[indices[node.indices_begin]]; // At least one interval
        for &idx in &indices[node.indices_begin..node.indices_end] {
            if ends[idx] > max_end {
                max_end = ends[idx];
            }
        }

        // Include children's max_end
        let left = node.left;
        let right = node.right;

        if left != Node::<T>::NULL && nodes[left as usize].max_end > max_end {
            max_end = nodes[left as usize].max_end;
        }
        if right != Node::<T>::NULL && nodes[right as usize].max_end > max_end {
            max_end = nodes[right as usize].max_end;
        }

        nodes[i].max_end = max_end;
    }
}
