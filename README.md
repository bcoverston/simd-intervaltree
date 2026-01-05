# simd-intervaltree

A SIMD-accelerated interval tree with zero-allocation queries.

## Features

- **SIMD acceleration**: AVX-512, AVX2, and NEON for fast sorted-array scans
- **Zero-allocation queries**: Iterator-based API with stack-based traversal
- **Mutable collections**: `IntervalSet` with stable IDs for insert/remove
- **`no_std` compatible**: Only requires `alloc`
- **jemalloc**: Optional (default on) for allocation performance

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
    ControlFlow::Continue(())
});

// SIMD-accelerated query for i64 intervals
tree.query_simd(3..12, |interval, value| {
    ControlFlow::Continue(())
});
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

Benchmarks vs `intervaltree` crate (Apple M-series):

| Size | simd-intervaltree | intervaltree | Speedup |
|------|-------------------|--------------|---------|
| 1K   | 164ns            | 216ns        | 24%     |
| 10K  | 470ns            | 1052ns       | 55%     |

## Architecture

### Data Layout

Data is laid out contiguously per node for cache efficiency:

- Each node's intervals are stored contiguously, sorted by start
- SIMD scans find early-termination cutoffs in O(n/lanes) time
- Separate `ends_desc` array enables SIMD for descending scans

### SIMD Support

| Architecture | Instruction Set | Elements/Op |
|--------------|-----------------|-------------|
| x86_64       | AVX-512         | 8 × i64     |
| x86_64       | AVX2            | 4 × i64     |
| aarch64      | NEON            | 2 × i64     |
| fallback     | scalar          | 1 × i64     |

Runtime detection selects the best available.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `std` | yes | Standard library support |
| `jemalloc` | yes | Use jemalloc allocator |

Disable defaults for `no_std`:

```toml
[dependencies]
simd-intervaltree = { version = "0.1", default-features = false }
```

## License

MIT
