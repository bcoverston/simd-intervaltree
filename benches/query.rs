use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use simd_intervaltree::{IntervalSet, IntervalTree};

fn generate_intervals(n: usize, seed: u64) -> Vec<(i64, i64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let start = rng.gen_range(0..1_000_000);
            let end = start + rng.gen_range(1..10_000);
            (start, end)
        })
        .collect()
}

fn bench_immutable_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("IntervalTree");

    for size in [100, 1000, 10_000, 100_000] {
        let intervals = generate_intervals(size, 42);

        // Build tree
        let tree = {
            let mut builder = IntervalTree::builder();
            for (start, end) in &intervals {
                builder = builder.insert(*start..*end, ());
            }
            builder.build()
        };

        group.bench_with_input(
            BenchmarkId::new("query", size),
            &size,
            |b, _| {
                let mut rng = StdRng::seed_from_u64(123);
                b.iter(|| {
                    let start = rng.gen_range(0i64..1_000_000);
                    let end = start + rng.gen_range(1..1000);
                    let count = tree.query(start..end).count();
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

fn bench_mutable_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("IntervalSet");

    for size in [100, 1000, 10_000] {
        let intervals = generate_intervals(size, 42);

        group.bench_with_input(
            BenchmarkId::new("insert", size),
            &intervals,
            |b, intervals| {
                b.iter(|| {
                    let mut set = IntervalSet::new();
                    for (start, end) in intervals {
                        set.insert(*start..*end, ());
                    }
                    black_box(set.len())
                });
            },
        );

        // Build set for query benchmark
        let mut set = IntervalSet::new();
        for (start, end) in &intervals {
            set.insert(*start..*end, ());
        }
        // Trigger initial build
        let _ = set.query(0..1).count();

        group.bench_with_input(
            BenchmarkId::new("query", size),
            &size,
            |b, _| {
                let mut rng = StdRng::seed_from_u64(123);
                b.iter(|| {
                    let start = rng.gen_range(0i64..1_000_000);
                    let end = start + rng.gen_range(1..1000);
                    let count = set.query(start..end).count();
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

fn bench_vs_intervaltree_crate(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs_intervaltree_crate");

    for size in [1000, 10_000] {
        let intervals = generate_intervals(size, 42);

        // Our tree
        let our_tree = {
            let mut builder = IntervalTree::builder();
            for (start, end) in &intervals {
                builder = builder.insert(*start..*end, ());
            }
            builder.build()
        };

        // intervaltree crate
        let their_tree: intervaltree::IntervalTree<i64, ()> = intervals
            .iter()
            .map(|(s, e)| ((*s..*e), ()))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("ours", size),
            &size,
            |b, _| {
                let mut rng = StdRng::seed_from_u64(123);
                b.iter(|| {
                    let start = rng.gen_range(0i64..1_000_000);
                    let end = start + rng.gen_range(1..1000);
                    let count = our_tree.query(start..end).count();
                    black_box(count)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("intervaltree_crate", size),
            &size,
            |b, _| {
                let mut rng = StdRng::seed_from_u64(123);
                b.iter(|| {
                    let start = rng.gen_range(0i64..1_000_000);
                    let end = start + rng.gen_range(1..1000);
                    let count = their_tree.query(start..end).count();
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_immutable_tree,
    bench_mutable_set,
    bench_vs_intervaltree_crate,
);
criterion_main!(benches);
