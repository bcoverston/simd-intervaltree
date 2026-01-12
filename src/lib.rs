//! # simd-intervaltree
//!
//! A SIMD-accelerated interval tree with zero-allocation queries.
//!
//! ## Features
//!
//! - **Zero-allocation queries**: Iterator and callback-based APIs that don't allocate
//! - **SIMD acceleration**: Uses AVX2/AVX-512 on x86_64, NEON on ARM for fast scans
//! - **Generic bounds**: Works with any `Ord` type, with fast paths for primitives
//! - **Immutable after construction**: `Send + Sync` by default
//! - **`no_std` compatible**: Only requires `alloc`
//!
//! ## Example
//!
//! ```
//! use simd_intervaltree::IntervalTree;
//!
//! let tree = IntervalTree::builder()
//!     .insert(0..10, "first")
//!     .insert(5..15, "second")
//!     .insert(20..30, "third")
//!     .build();
//!
//! // Zero-allocation iteration
//! for entry in tree.query(3..12) {
//!     println!("{:?} => {}", entry.interval, entry.value);
//! }
//!
//! // Early termination with callback
//! use std::ops::ControlFlow;
//! tree.query_with(3..12, |interval, value| {
//!     println!("{interval:?} => {value}");
//!     ControlFlow::<()>::Continue(())
//! });
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

extern crate alloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod builder;
mod mutable;
mod query;
mod simd;
mod tree;

pub use builder::IntervalTreeBuilder;
pub use mutable::{IntervalId, IntervalSet};
pub use query::{QueryEntry, QueryIter};
pub use tree::IntervalTree;

/// A half-open interval `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Interval<T> {
    /// Start bound (inclusive).
    pub start: T,
    /// End bound (exclusive).
    pub end: T,
}

impl<T: Ord> Interval<T> {
    /// Creates a new interval.
    ///
    /// # Panics
    ///
    /// Panics if `start > end`.
    #[must_use]
    pub fn new(start: T, end: T) -> Self {
        assert!(start <= end, "interval start must be <= end");
        Self { start, end }
    }

    /// Returns true if this interval overlaps with `other`.
    ///
    /// Two intervals `[a, b)` and `[c, d)` overlap iff `a < d && c < b`.
    #[inline]
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Returns true if this interval contains the point.
    #[inline]
    #[must_use]
    pub fn contains_point(&self, point: &T) -> bool {
        self.start <= *point && *point < self.end
    }
}

impl<T: Ord> From<core::ops::Range<T>> for Interval<T> {
    fn from(range: core::ops::Range<T>) -> Self {
        Self::new(range.start, range.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_overlaps_matches_iter() {
        let tree = IntervalTree::builder()
            .insert(0i64..10, "a")
            .insert(5..15, "b")
            .insert(10..20, "c")
            .insert(20..30, "d")
            .insert(25..35, "e")
            .build();

        // Test various query ranges
        let queries = [(3, 12), (0, 5), (15, 25), (0, 100), (50, 60)];
        for (start, end) in queries {
            let iter_count = tree.query(start..end).count();
            let simd_count = tree.count_overlaps(start..end);
            assert_eq!(
                iter_count, simd_count,
                "mismatch for query {}..{}: iter={}, simd={}",
                start, end, iter_count, simd_count
            );
        }
    }

    #[test]
    fn interval_overlap() {
        let a = Interval::new(0, 10);
        let b = Interval::new(5, 15);
        let c = Interval::new(10, 20);
        let d = Interval::new(20, 30);

        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
        assert!(!a.overlaps(&c)); // [0,10) and [10,20) don't overlap (half-open)
        assert!(!a.overlaps(&d));
    }

    #[test]
    fn interval_contains_point() {
        let a = Interval::new(0, 10);
        assert!(a.contains_point(&0));
        assert!(a.contains_point(&5));
        assert!(!a.contains_point(&10)); // half-open, 10 is excluded
        assert!(!a.contains_point(&-1));
    }

    #[test]
    fn interval_from_range() {
        let interval: Interval<i32> = (0..10).into();
        assert_eq!(interval.start, 0);
        assert_eq!(interval.end, 10);
    }
}
