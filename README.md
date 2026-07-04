# simd-intervaltree

A SIMD-accelerated interval tree with zero-allocation queries.

## Features

- **SIMD acceleration**: AVX2 and NEON kernels with runtime dispatch; AVX-512 opt-in
- **Hybrid scans**: large sorted runs are binary-narrowed, only the final window is SIMD-scanned
- **Zero-allocation queries**: Iterator-based API with stack-based traversal
- **Mutable collections**: `IntervalSet` with stable IDs for insert/remove
- **`no_std` compatible**: Only requires `alloc`
- **Zero dependencies**

## Usage

### Immutable Tree

```rust
use simd_intervaltree::IntervalTree;
use std::ops::ControlFlow;

let tree = IntervalTree::builder()
    .insert(0..10, "first")
    .insert(5..15, "second")
    .insert(20..30, "third")
    .build();

// Iterator-based query (zero allocation)
for entry in tree.query(3..12) {
    println!("{:?} => {}", entry.interval, entry.value);
}

// Callback with early termination
tree.query_with(3..12, |interval, value| {
    println!("{interval:?} => {value}");
    ControlFlow::<()>::Continue(())
});

// SIMD-accelerated query for i64 intervals
tree.query_simd(3..12, |interval, value| {
    ControlFlow::<()>::Continue(())
});

// SIMD-accelerated overlap counting (no per-interval yielding)
let n = tree.count_overlaps(3..12);
```

### Mutable Set with Stable IDs

Useful for tracking resources like SSTables in an LSM tree:

```rust
use simd_intervaltree::IntervalSet;

let mut sstables = IntervalSet::new();

// Insert returns stable ID
let id1 = sstables.insert(100..500, "sst_001.db");
let id2 = sstables.insert(300..700, "sst_002.db");

// Query returns (ID, Interval, &Value)
for (id, range, filename) in sstables.query(250..350) {
    println!("{id:?}: {range:?} => {filename}");
}

// Remove by ID after compaction
sstables.remove(id1);

// Lookup by ID
if let Some(filename) = sstables.get(id2) {
    println!("Found: {filename}");
}
```

## Performance

Run `cargo bench` for numbers on your hardware. The `query` benchmark group
compares against `intervaltree`, `rust-lapper`, `coitrees`, and
`superintervals`. Note the two simd-intervaltree entries: the count-only path
(`count_overlaps`, comparable to coitrees' `query_count`) and the iterator
path (comparable to crates that must enumerate results).

## Architecture

### Data Layout

Data is laid out contiguously per node for cache efficiency:

- Each node's intervals are stored contiguously, sorted by start
- A separate `ends_desc` array enables early-terminating descending scans
- Node metadata uses `u32` indices to keep traversal state compact
  (a tree holds at most `u32::MAX - 1` intervals)

### Query Scans

Per-node cutoff searches are hybrid:

- Runs of ≤ 64 elements are scanned linearly with SIMD — the common case,
  and the case SIMD is best at
- Longer runs are first narrowed by binary search, so large nodes cost
  O(log n) probes instead of an O(n) sweep

### SIMD Support

| Architecture | Instruction Set | Elements/Op | Availability |
|--------------|-----------------|-------------|--------------|
| x86_64       | AVX-512         | 8 × i64     | `avx512` feature (Rust 1.89+) |
| x86_64       | AVX2            | 4 × i64     | default |
| aarch64      | NEON            | 2 × i64     | default |
| fallback     | scalar          | —           | always |

On x86_64 the CPU feature level is detected once and cached; per-query
dispatch is a single relaxed atomic load. Without `std`, detection is
compile-time only (`-C target-feature=+avx2`).

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `std` | yes | Standard library support (enables runtime CPU feature detection) |
| `avx512` | no | AVX-512 kernels; requires Rust 1.89+ |

Disable defaults for `no_std` (requires only `alloc`):

```toml
[dependencies]
simd-intervaltree = { version = "0.2", default-features = false }
```

## MSRV

Rust 1.86 (1.89+ with the `avx512` feature).

## License

MIT
