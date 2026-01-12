use coitrees::IntervalTree as _;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use simd_intervaltree::IntervalTree;

fn generate_intervals(n: usize, seed: u64) -> Vec<(u64, u64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let start: u64 = rng.gen_range(0..1_000_000);
            let end = start + rng.gen_range(1..10_000);
            (start, end)
        })
        .collect()
}

fn generate_queries(n: usize, seed: u64) -> Vec<(u64, u64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let start: u64 = rng.gen_range(0..1_000_000);
            let end = start + rng.gen_range(1..1000);
            (start, end)
        })
        .collect()
}

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");

    for size in [1_000, 10_000, 100_000] {
        let intervals = generate_intervals(size, 42);

        // simd-intervaltree (uses i64)
        group.bench_with_input(BenchmarkId::new("simd-intervaltree", size), &intervals, |b, ivs| {
            b.iter(|| {
                let mut builder = IntervalTree::<i64, ()>::builder();
                for (start, end) in ivs {
                    builder = builder.insert((*start as i64)..(*end as i64), ());
                }
                black_box(builder.build())
            });
        });

        // intervaltree (uses i64)
        group.bench_with_input(BenchmarkId::new("intervaltree", size), &intervals, |b, ivs| {
            b.iter(|| {
                let tree: intervaltree::IntervalTree<i64, ()> = ivs
                    .iter()
                    .map(|(s, e)| ((*s as i64..*e as i64), ()))
                    .collect();
                black_box(tree)
            });
        });

        // rust-lapper (requires unsigned)
        group.bench_with_input(BenchmarkId::new("rust-lapper", size), &intervals, |b, ivs| {
            b.iter(|| {
                let data: Vec<rust_lapper::Interval<u64, ()>> = ivs
                    .iter()
                    .map(|(s, e)| rust_lapper::Interval {
                        start: *s,
                        stop: *e,
                        val: (),
                    })
                    .collect();
                black_box(rust_lapper::Lapper::new(data))
            });
        });

        // coitrees (uses i32, end-inclusive)
        group.bench_with_input(BenchmarkId::new("coitrees", size), &intervals, |b, ivs| {
            b.iter(|| {
                let nodes: Vec<_> = ivs
                    .iter()
                    .map(|(s, e)| coitrees::Interval::new(*s as i32, (*e - 1) as i32, ()))
                    .collect();
                black_box(coitrees::COITree::<(), usize>::new(&nodes))
            });
        });

        // superintervals (uses i32)
        group.bench_with_input(BenchmarkId::new("superintervals", size), &intervals, |b, ivs| {
            b.iter(|| {
                let mut imap = superintervals::IntervalMap::new();
                for (s, e) in ivs {
                    imap.add(*s as i32, *e as i32, ());
                }
                imap.build();
                black_box(imap)
            });
        });
    }

    group.finish();
}

fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");

    for size in [1_000, 10_000, 100_000] {
        let intervals = generate_intervals(size, 42);
        let queries = generate_queries(1000, 123);

        // Build all trees upfront
        let simd_tree = {
            let mut builder = IntervalTree::<i64, ()>::builder();
            for (start, end) in &intervals {
                builder = builder.insert((*start as i64)..(*end as i64), ());
            }
            builder.build()
        };

        let intervaltree_tree: intervaltree::IntervalTree<i64, ()> = intervals
            .iter()
            .map(|(s, e)| ((*s as i64..*e as i64), ()))
            .collect();

        let lapper = {
            let data: Vec<rust_lapper::Interval<u64, ()>> = intervals
                .iter()
                .map(|(s, e)| rust_lapper::Interval {
                    start: *s,
                    stop: *e,
                    val: (),
                })
                .collect();
            rust_lapper::Lapper::new(data)
        };

        let coitree: coitrees::COITree<(), usize> = {
            let nodes: Vec<_> = intervals
                .iter()
                .map(|(s, e)| coitrees::Interval::new(*s as i32, (*e - 1) as i32, ()))
                .collect();
            coitrees::COITree::new(&nodes)
        };

        let superintervals_map = {
            let mut imap = superintervals::IntervalMap::new();
            for (s, e) in &intervals {
                imap.add(*s as i32, *e as i32, ());
            }
            imap.build();
            imap
        };

        // simd-intervaltree (SIMD count)
        group.bench_with_input(BenchmarkId::new("simd-intervaltree", size), &queries, |b, qs| {
            let mut i = 0;
            b.iter(|| {
                let (start, end) = qs[i % qs.len()];
                i += 1;
                black_box(simd_tree.count_overlaps((start as i64)..(end as i64)))
            });
        });

        // intervaltree
        group.bench_with_input(BenchmarkId::new("intervaltree", size), &queries, |b, qs| {
            let mut i = 0;
            b.iter(|| {
                let (start, end) = qs[i % qs.len()];
                i += 1;
                black_box(intervaltree_tree.query((start as i64)..(end as i64)).count())
            });
        });

        // rust-lapper
        group.bench_with_input(BenchmarkId::new("rust-lapper", size), &queries, |b, qs| {
            let mut i = 0;
            b.iter(|| {
                let (start, end) = qs[i % qs.len()];
                i += 1;
                black_box(lapper.find(start, end).count())
            });
        });

        // coitrees (end-inclusive)
        group.bench_with_input(BenchmarkId::new("coitrees", size), &queries, |b, qs| {
            let mut i = 0;
            b.iter(|| {
                let (start, end) = qs[i % qs.len()];
                i += 1;
                black_box(coitree.query_count(start as i32, (end - 1) as i32))
            });
        });

        // superintervals
        group.bench_with_input(BenchmarkId::new("superintervals", size), &queries, |b, qs| {
            let mut i = 0;
            let mut results = Vec::new();
            b.iter(|| {
                let (start, end) = qs[i % qs.len()];
                i += 1;
                results.clear();
                superintervals_map.search_values(start as i32, end as i32, &mut results);
                black_box(results.len())
            });
        });
    }

    group.finish();
}

fn bench_query_collect(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_collect");

    for size in [1_000, 10_000] {
        let intervals = generate_intervals(size, 42);
        let queries = generate_queries(1000, 123);

        // Build all trees upfront
        let simd_tree = {
            let mut builder = IntervalTree::<i64, ()>::builder();
            for (start, end) in &intervals {
                builder = builder.insert((*start as i64)..(*end as i64), ());
            }
            builder.build()
        };

        let intervaltree_tree: intervaltree::IntervalTree<i64, ()> = intervals
            .iter()
            .map(|(s, e)| ((*s as i64..*e as i64), ()))
            .collect();

        let lapper = {
            let data: Vec<rust_lapper::Interval<u64, ()>> = intervals
                .iter()
                .map(|(s, e)| rust_lapper::Interval {
                    start: *s,
                    stop: *e,
                    val: (),
                })
                .collect();
            rust_lapper::Lapper::new(data)
        };

        // simd-intervaltree (collecting results)
        group.bench_with_input(
            BenchmarkId::new("simd-intervaltree", size),
            &queries,
            |b, qs| {
                let mut i = 0;
                b.iter(|| {
                    let (start, end) = qs[i % qs.len()];
                    i += 1;
                    black_box(simd_tree.query((start as i64)..(end as i64)).collect::<Vec<_>>())
                });
            },
        );

        // intervaltree (collecting results)
        group.bench_with_input(BenchmarkId::new("intervaltree", size), &queries, |b, qs| {
            let mut i = 0;
            b.iter(|| {
                let (start, end) = qs[i % qs.len()];
                i += 1;
                black_box(
                    intervaltree_tree
                        .query((start as i64)..(end as i64))
                        .collect::<Vec<_>>(),
                )
            });
        });

        // rust-lapper (collecting results)
        group.bench_with_input(BenchmarkId::new("rust-lapper", size), &queries, |b, qs| {
            let mut i = 0;
            b.iter(|| {
                let (start, end) = qs[i % qs.len()];
                i += 1;
                black_box(lapper.find(start, end).collect::<Vec<_>>())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_construction, bench_query, bench_query_collect);
criterion_main!(benches);
