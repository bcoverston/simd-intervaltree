//! Property-based tests for interval tree correctness.

use proptest::prelude::*;
use simd_intervaltree::{IntervalSet, IntervalTree};
use std::collections::HashSet;

/// Generate a valid interval (start < end)
fn interval_strategy() -> impl Strategy<Value = (i64, i64)> {
    (0i64..1_000_000).prop_flat_map(|start| (Just(start), (start + 1)..=(start + 10_000)))
}

/// Generate a list of intervals
fn intervals_strategy(max_count: usize) -> impl Strategy<Value = Vec<(i64, i64)>> {
    prop::collection::vec(interval_strategy(), 1..=max_count)
}

/// Naive brute-force query for comparison.
///
/// Uses mathematical half-open overlap: empty intervals and empty queries
/// overlap nothing (matching the library's documented semantics).
fn naive_query(intervals: &[(i64, i64)], query_start: i64, query_end: i64) -> Vec<(i64, i64)> {
    if query_start >= query_end {
        return Vec::new();
    }
    intervals
        .iter()
        .filter(|(s, e)| *s < *e && *s < query_end && query_start < *e)
        .copied()
        .collect()
}

/// Coordinates spanning the full i64 domain, biased toward the extremes,
/// bounded so `start + 10_000` cannot overflow.
fn extreme_coord() -> impl Strategy<Value = i64> {
    prop_oneof![
        Just(i64::MIN),
        i64::MIN..i64::MIN + 10_000,
        -10_000i64..10_000,
        (i64::MAX - 20_000)..(i64::MAX - 10_000),
        any::<i64>().prop_map(|v| v.min(i64::MAX - 10_001)),
    ]
}

/// Intervals over the full domain, including zero-width ones (len 0).
fn extreme_intervals(max_count: usize) -> impl Strategy<Value = Vec<(i64, i64)>> {
    let iv = (extreme_coord(), 0i64..=10_000).prop_map(|(start, len)| (start, start + len));
    prop::collection::vec(iv, 1..=max_count)
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
        let _ = tree.query_with(query_start..query_end, |interval, _| {
            generic_results.push((interval.start, interval.end));
            ControlFlow::<()>::Continue(())
        });

        // SIMD query
        let mut simd_results = Vec::new();
        let _ = tree.query_simd(query_start..query_end, |interval, _| {
            simd_results.push((interval.start, interval.end));
            ControlFlow::<()>::Continue(())
        });

        // Results should match (order may differ)
        let generic_set: HashSet<_> = generic_results.into_iter().collect();
        let simd_set: HashSet<_> = simd_results.into_iter().collect();

        prop_assert_eq!(generic_set, simd_set);
    }

    #[test]
    fn count_overlaps_matches_naive(
        intervals in intervals_strategy(100),
        query_start in 0i64..1_000_000,
        query_len in 1i64..10_000,
    ) {
        let query_end = query_start + query_len;

        let tree = {
            let mut builder = IntervalTree::<i64, usize>::builder();
            for (i, (start, end)) in intervals.iter().enumerate() {
                builder = builder.insert(*start..*end, i);
            }
            builder.build()
        };

        let count = tree.count_overlaps(query_start..query_end);
        let naive = naive_query(&intervals, query_start, query_end).len();

        prop_assert_eq!(count, naive);
    }

    /// Full-i64-domain property: negative coordinates, values at the extremes,
    /// zero-width intervals, and zero-width queries, checked against the
    /// mathematical overlap definition on every query path.
    #[test]
    fn extreme_domain_matches_naive(
        intervals in extreme_intervals(100),
        query_start in extreme_coord(),
        query_len in 0i64..=10_000,
    ) {
        let query_end = query_start + query_len;

        let tree = {
            let mut builder = IntervalTree::<i64, usize>::builder();
            for (i, (start, end)) in intervals.iter().enumerate() {
                builder = builder.insert(*start..*end, i);
            }
            builder.build()
        };

        let naive: HashSet<(i64, i64)> = naive_query(&intervals, query_start, query_end)
            .into_iter()
            .collect();

        // Iterator path
        let iter_results: HashSet<(i64, i64)> = tree
            .query(query_start..query_end)
            .map(|e| (e.interval.start, e.interval.end))
            .collect();
        prop_assert_eq!(&iter_results, &naive);

        // SIMD callback path
        use std::ops::ControlFlow;
        let mut simd_results = HashSet::new();
        let _ = tree.query_simd(query_start..query_end, |interval, _| {
            simd_results.insert((interval.start, interval.end));
            ControlFlow::<()>::Continue(())
        });
        prop_assert_eq!(&simd_results, &naive);

        // SIMD count path (counts duplicates, so compare against the
        // unsorted naive Vec length)
        prop_assert_eq!(
            tree.count_overlaps(query_start..query_end),
            naive_query(&intervals, query_start, query_end).len()
        );
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

/// A single node holding thousands of intervals: all intervals contain the
/// center point, so they land in one node and queries must scan large sorted
/// arrays. This exercises the binary-narrow + SIMD-window hybrid path, which
/// randomized small trees never reach.
#[test]
fn large_single_node_scans() {
    const N: i64 = 5000;

    // Nested intervals [i, 2N - i) for i in 0..N; every one contains N.
    let mut builder = IntervalTree::<i64, i64>::builder();
    for i in 0..N {
        builder = builder.insert(i..(2 * N - i), i);
    }
    let tree = builder.build();

    // Case 2 (query right of pivot): [i, 2N - i) overlaps [q, q+100) iff
    // 2N - i > q, i.e. i < 2N - q.
    for q in [N + 1, 2 * N - 1000, 2 * N - 64, 2 * N - 1] {
        let expected = (2 * N - q).min(N) as usize;
        assert_eq!(
            tree.count_overlaps(q..(q + 100)),
            expected,
            "case 2 count at q={q}"
        );
        assert_eq!(
            tree.query(q..(q + 100)).count(),
            expected,
            "case 2 iter at q={q}"
        );
    }

    // Case 3 (query left of pivot): overlaps [q-100, q) iff i < q.
    for q in [1, 63, 64, 65, 1000, N - 1] {
        let expected = q.min(N) as usize;
        assert_eq!(
            tree.count_overlaps((q - 100)..q),
            expected,
            "case 3 count at q={q}"
        );
        assert_eq!(
            tree.query((q - 100)..q).count(),
            expected,
            "case 3 iter at q={q}"
        );
    }

    // Query containing the center matches everything.
    assert_eq!(tree.count_overlaps((N - 1)..(N + 1)), N as usize);
}

/// Regression: a zero-width interval used to send the builder into infinite
/// recursion (stack overflow), because [x, x) can never straddle a pivot and
/// re-entered its own partition forever. Empty intervals are now dropped at
/// build time.
#[test]
fn zero_width_interval_does_not_hang_builder() {
    let tree = IntervalTree::builder()
        .insert(5..5, "empty")
        .insert(0..10, "real")
        .insert(7..7, "also empty")
        .build();

    // Empty intervals are dropped and never match.
    assert_eq!(tree.len(), 1);
    let hits: Vec<_> = tree.query(0..100).collect();
    assert_eq!(hits.len(), 1);
    assert_eq!(*hits[0].value, "real");

    // The all-empty tree builds and matches nothing.
    let tree: IntervalTree<i64, ()> = IntervalTree::builder().insert(5..5, ()).build();
    assert!(tree.is_empty());
    assert_eq!(tree.query(0..100).count(), 0);
}

/// Empty query ranges ([x, x)) overlap nothing, on every query path.
#[test]
fn empty_query_returns_nothing_on_all_paths() {
    use std::ops::ControlFlow;

    let tree = IntervalTree::builder()
        .insert(0i64..10, "a")
        .insert(3..7, "b")
        .build();

    // Query point 5 is strictly inside both intervals; the raw formula
    // `s < qe && qs < e` would claim overlap, but an empty set overlaps
    // nothing.
    assert_eq!(tree.query(5..5).count(), 0);
    assert_eq!(tree.count_overlaps(5..5), 0);

    let mut with_hits = 0;
    let _ = tree.query_with(5..5, |_, _| {
        with_hits += 1;
        ControlFlow::<()>::Continue(())
    });
    assert_eq!(with_hits, 0);

    let mut simd_hits = 0;
    let _ = tree.query_simd(5..5, |_, _| {
        simd_hits += 1;
        ControlFlow::<()>::Continue(())
    });
    assert_eq!(simd_hits, 0);
}

/// IntervalSet with zero-width intervals: IDs stay valid, queries skip them.
#[test]
fn interval_set_zero_width() {
    let mut set = IntervalSet::new();
    let empty_id = set.insert(5..5, "empty");
    let real_id = set.insert(0..10, "real");

    assert_eq!(set.len(), 2); // both stored
    assert_eq!(set.get(empty_id), Some(&"empty")); // ID lookup still works

    let hits: Vec<_> = set.query(0..100).map(|(id, _, _)| id).collect();
    assert_eq!(hits, vec![real_id]); // but queries never return it
}

/// Boundary coordinates at the edges of i64.
#[test]
fn extreme_coordinate_boundaries() {
    let tree = IntervalTree::builder()
        .insert(i64::MIN..i64::MIN + 1, "low")
        .insert(-5..5, "mid")
        .insert(i64::MAX - 1..i64::MAX, "high")
        .build();

    assert_eq!(tree.query(i64::MIN..i64::MIN + 1).count(), 1);
    assert_eq!(tree.query(i64::MAX - 1..i64::MAX).count(), 1);
    assert_eq!(tree.query(i64::MIN..i64::MAX).count(), 3);
    assert_eq!(tree.count_overlaps(i64::MIN..i64::MAX), 3);
    assert_eq!(tree.count_overlaps(i64::MIN + 1..i64::MAX - 1), 1);
}
