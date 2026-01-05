//! Property-based tests for interval tree correctness.

use proptest::prelude::*;
use simd_intervaltree::{Interval, IntervalSet, IntervalTree};
use std::collections::HashSet;

/// Generate a valid interval (start < end)
fn interval_strategy() -> impl Strategy<Value = (i64, i64)> {
    (0i64..1_000_000).prop_flat_map(|start| (Just(start), (start + 1)..=(start + 10_000)))
}

/// Generate a list of intervals
fn intervals_strategy(max_count: usize) -> impl Strategy<Value = Vec<(i64, i64)>> {
    prop::collection::vec(interval_strategy(), 1..=max_count)
}

/// Naive brute-force query for comparison
fn naive_query(intervals: &[(i64, i64)], query_start: i64, query_end: i64) -> Vec<(i64, i64)> {
    intervals
        .iter()
        .filter(|(s, e)| *s < query_end && query_start < *e)
        .copied()
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn tree_query_matches_naive(
        intervals in intervals_strategy(100),
        query_start in 0i64..1_000_000,
        query_len in 1i64..10_000,
    ) {
        let query_end = query_start + query_len;

        // Build tree
        let tree = {
            let mut builder = IntervalTree::builder();
            for (i, (start, end)) in intervals.iter().enumerate() {
                builder = builder.insert(*start..*end, i);
            }
            builder.build()
        };

        // Query tree
        let tree_results: HashSet<(i64, i64)> = tree
            .query(query_start..query_end)
            .map(|e| (e.interval.start, e.interval.end))
            .collect();

        // Naive query
        let naive_results: HashSet<(i64, i64)> = naive_query(&intervals, query_start, query_end)
            .into_iter()
            .collect();

        prop_assert_eq!(tree_results, naive_results);
    }

    #[test]
    fn set_query_matches_naive(
        intervals in intervals_strategy(50),
        query_start in 0i64..1_000_000,
        query_len in 1i64..10_000,
    ) {
        let query_end = query_start + query_len;

        // Build set
        let mut set = IntervalSet::new();
        for (start, end) in &intervals {
            set.insert(*start..*end, ());
        }

        // Query set
        let set_results: HashSet<(i64, i64)> = set
            .query(query_start..query_end)
            .map(|(_, interval, _)| (interval.start, interval.end))
            .collect();

        // Naive query
        let naive_results: HashSet<(i64, i64)> = naive_query(&intervals, query_start, query_end)
            .into_iter()
            .collect();

        prop_assert_eq!(set_results, naive_results);
    }

    #[test]
    fn set_insert_remove_consistency(
        intervals in intervals_strategy(50),
        to_remove in prop::collection::vec(0usize..50, 0..25),
    ) {
        let mut set = IntervalSet::new();
        let mut ids = Vec::new();

        // Insert all
        for (start, end) in &intervals {
            let id = set.insert(*start..*end, ());
            ids.push(id);
        }

        prop_assert_eq!(set.len(), intervals.len());

        // Remove some (deduplicated indices within bounds)
        let mut removed = HashSet::new();
        for idx in to_remove {
            if idx < ids.len() && !removed.contains(&idx) {
                let success = set.remove(ids[idx]);
                prop_assert!(success);
                removed.insert(idx);
            }
        }

        prop_assert_eq!(set.len(), intervals.len() - removed.len());

        // Verify removed IDs return None
        for &idx in &removed {
            prop_assert!(set.get(ids[idx]).is_none());
        }

        // Verify remaining IDs still work
        for (idx, id) in ids.iter().enumerate() {
            if !removed.contains(&idx) {
                prop_assert!(set.get(*id).is_some());
            }
        }
    }

    #[test]
    fn simd_query_matches_generic(
        intervals in intervals_strategy(100),
        query_start in 0i64..1_000_000,
        query_len in 1i64..10_000,
    ) {
        use std::ops::ControlFlow;

        let query_end = query_start + query_len;

        // Build tree
        let tree = {
            let mut builder = IntervalTree::<i64, usize>::builder();
            for (i, (start, end)) in intervals.iter().enumerate() {
                builder = builder.insert(*start..*end, i);
            }
            builder.build()
        };

        // Generic query
        let mut generic_results = Vec::new();
        tree.query_with(query_start..query_end, |interval, _| {
            generic_results.push((interval.start, interval.end));
            ControlFlow::<()>::Continue(())
        });

        // SIMD query
        let mut simd_results = Vec::new();
        tree.query_simd(query_start..query_end, |interval, _| {
            simd_results.push((interval.start, interval.end));
            ControlFlow::<()>::Continue(())
        });

        // Results should match (order may differ)
        let generic_set: HashSet<_> = generic_results.into_iter().collect();
        let simd_set: HashSet<_> = simd_results.into_iter().collect();

        prop_assert_eq!(generic_set, simd_set);
    }

    #[test]
    fn empty_tree_returns_empty(query_start in 0i64..1_000_000, query_len in 1i64..10_000) {
        let tree: IntervalTree<i64, ()> = IntervalTree::builder().build();
        let results: Vec<_> = tree.query(query_start..(query_start + query_len)).collect();
        prop_assert!(results.is_empty());
    }

    #[test]
    fn single_interval_tree(
        start in 0i64..1_000_000,
        len in 1i64..10_000,
        query_start in 0i64..1_000_000,
        query_len in 1i64..10_000,
    ) {
        let end = start + len;
        let query_end = query_start + query_len;

        let tree = IntervalTree::builder()
            .insert(start..end, ())
            .build();

        let results: Vec<_> = tree.query(query_start..query_end).collect();

        let should_overlap = start < query_end && query_start < end;

        if should_overlap {
            prop_assert_eq!(results.len(), 1);
        } else {
            prop_assert!(results.is_empty());
        }
    }
}
