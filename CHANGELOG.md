# Changelog

## 0.2.0

### Changed

- `IntervalSet` no longer rebuilds its tree on the first query after every
  mutation. Inserts go to a pending list that queries scan, removals are
  filtered out, and the tree is rebuilt once pending changes exceed
  max(64, √n). `IntervalSet::query` now takes `&self` instead of `&mut self`,
  and `get` and `iter` no longer require `T: Ord`.
- Empty intervals (`start == end`) are dropped at build time. They overlap
  nothing and are not counted by `IntervalTree::len`. `IntervalSet` still
  stores them, so their IDs stay valid.
- Queries share one per-node traversal instead of four copies. Results and
  their order are unchanged.

### Fixed

- The builder hung on zero-width intervals.
- x86_64, Windows, and `no_std` builds.
- AVX-512 type casts across stdarch versions.

### Added

- CI across Linux, Windows, and macOS, plus AVX-512 under Intel SDE, `no_std`
  targets, MSRV, and Miri.
- Benchmarks at 1M intervals, compared against coitrees, superintervals,
  rust-lapper, and intervaltree.

## 0.1.0

- Initial release.
