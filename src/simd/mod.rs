//! SIMD-accelerated operations for interval scanning.
//!
//! This module provides optimized scanning operations using:
//! - AVX-512 on x86_64 (512-bit, 8 × i64) — opt-in via the `avx512` feature (Rust 1.89+)
//! - AVX2 on x86_64 (256-bit, 4 × i64)
//! - NEON on aarch64 (128-bit, 2 × i64)
//! - Scalar fallback for other architectures
//!
//! The primary operation is finding the cutoff index in a sorted array
//! where elements stop satisfying a comparison condition.
//!
//! # Strategy
//!
//! Both entry points are hybrid: arrays larger than [`LINEAR_SCAN_MAX`] are
//! first narrowed with a branchy-but-cheap binary search (O(log n) probes),
//! and only the final small window is scanned linearly with SIMD. This keeps
//! the fast single-vector path for the small per-node arrays that dominate
//! real trees, without degrading to O(n) on large nodes.
//!
//! On x86_64 the CPU feature level is detected once and cached in an atomic,
//! so per-call dispatch is a single relaxed load. Without `std` there is no
//! runtime detection; the level is fixed at compile time from
//! `cfg(target_feature)`.

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "aarch64")]
mod arm;

// On aarch64 the NEON kernels are unconditional, so the scalar fallbacks are
// only reachable from tests there.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
mod scalar;

/// Window size at or below which we scan linearly (with SIMD where available)
/// instead of binary-searching. Chosen to cover a few cache lines of i64s.
const LINEAR_SCAN_MAX: usize = 64;

/// Cached CPU feature level for x86_64 dispatch.
#[cfg(target_arch = "x86_64")]
mod level {
    use core::sync::atomic::{AtomicU8, Ordering};

    pub const UNKNOWN: u8 = 0;
    pub const SCALAR: u8 = 1;
    pub const AVX2: u8 = 2;
    #[cfg(feature = "avx512")]
    pub const AVX512: u8 = 3;

    static LEVEL: AtomicU8 = AtomicU8::new(UNKNOWN);

    /// Returns the cached feature level, detecting it on first use.
    ///
    /// A racing first call may detect twice; both writers store the same
    /// value, so `Relaxed` ordering is sufficient.
    #[inline]
    pub fn get() -> u8 {
        let cached = LEVEL.load(Ordering::Relaxed);
        if cached != UNKNOWN {
            return cached;
        }
        let detected = detect();
        LEVEL.store(detected, Ordering::Relaxed);
        detected
    }

    #[cfg(feature = "std")]
    fn detect() -> u8 {
        #[cfg(feature = "avx512")]
        if is_x86_feature_detected!("avx512f") {
            return AVX512;
        }
        if is_x86_feature_detected!("avx2") {
            return AVX2;
        }
        SCALAR
    }

    /// Without `std` there is no runtime CPUID; trust compile-time target
    /// features only.
    #[cfg(not(feature = "std"))]
    fn detect() -> u8 {
        #[cfg(all(feature = "avx512", target_feature = "avx512f"))]
        {
            AVX512
        }
        #[cfg(all(
            not(all(feature = "avx512", target_feature = "avx512f")),
            target_feature = "avx2"
        ))]
        {
            AVX2
        }
        #[cfg(all(
            not(all(feature = "avx512", target_feature = "avx512f")),
            not(target_feature = "avx2")
        ))]
        {
            SCALAR
        }
    }
}

/// Finds the first index where `arr[i] >= threshold` in a sorted (ascending) array.
///
/// Returns `arr.len()` if no such element exists.
#[inline]
pub fn find_ge_threshold_i64(arr: &[i64], threshold: i64) -> usize {
    // Binary-narrow large arrays down to a small window first.
    // Invariant: the boundary index lies in `lo..=lo + len`.
    let mut lo = 0usize;
    let mut len = arr.len();
    while len > LINEAR_SCAN_MAX {
        let half = len / 2;
        // If the last element of the lower half is still below the
        // threshold, the boundary is in the upper half.
        if arr[lo + half - 1] < threshold {
            lo += half;
        }
        len -= half;
    }
    lo + scan_ge(&arr[lo..lo + len], threshold)
}

/// Finds the first index where `arr[i] <= threshold` in a sorted (descending) array.
///
/// Returns `arr.len()` if no such element exists.
#[inline]
pub fn find_le_threshold_i64(arr: &[i64], threshold: i64) -> usize {
    let mut lo = 0usize;
    let mut len = arr.len();
    while len > LINEAR_SCAN_MAX {
        let half = len / 2;
        // Descending order: if the last element of the lower half is still
        // above the threshold, the boundary is in the upper half.
        if arr[lo + half - 1] > threshold {
            lo += half;
        }
        len -= half;
    }
    lo + scan_le(&arr[lo..lo + len], threshold)
}

/// Linear scan of a small window for the first element `>= threshold`.
#[allow(unsafe_code)]
#[inline]
fn scan_ge(arr: &[i64], threshold: i64) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        match level::get() {
            #[cfg(feature = "avx512")]
            // SAFETY: level::AVX512 is only ever cached after runtime
            // detection of avx512f (or compile-time target_feature).
            level::AVX512 => unsafe { x86::find_ge_threshold_avx512(arr, threshold) },
            // SAFETY: as above, for avx2.
            level::AVX2 => unsafe { x86::find_ge_threshold_avx2(arr, threshold) },
            _ => scalar::find_ge_threshold(arr, threshold),
        }
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is mandatory on aarch64.
    unsafe {
        arm::find_ge_threshold_neon(arr, threshold)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::find_ge_threshold(arr, threshold)
}

/// Linear scan of a small window for the first element `<= threshold`
/// (descending order).
#[allow(unsafe_code)]
#[inline]
fn scan_le(arr: &[i64], threshold: i64) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        match level::get() {
            #[cfg(feature = "avx512")]
            // SAFETY: level::AVX512 is only ever cached after runtime
            // detection of avx512f (or compile-time target_feature).
            level::AVX512 => unsafe { x86::find_le_threshold_avx512(arr, threshold) },
            // SAFETY: as above, for avx2.
            level::AVX2 => unsafe { x86::find_le_threshold_avx2(arr, threshold) },
            _ => scalar::find_le_threshold(arr, threshold),
        }
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is mandatory on aarch64.
    unsafe {
        arm::find_le_threshold_neon(arr, threshold)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::find_le_threshold(arr, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn test_find_ge_threshold() {
        let arr = [1i64, 3, 5, 7, 9, 11, 13, 15];

        assert_eq!(find_ge_threshold_i64(&arr, 0), 0);
        assert_eq!(find_ge_threshold_i64(&arr, 1), 0);
        assert_eq!(find_ge_threshold_i64(&arr, 2), 1);
        assert_eq!(find_ge_threshold_i64(&arr, 5), 2);
        assert_eq!(find_ge_threshold_i64(&arr, 6), 3);
        assert_eq!(find_ge_threshold_i64(&arr, 15), 7);
        assert_eq!(find_ge_threshold_i64(&arr, 16), 8);
    }

    #[test]
    fn test_find_le_threshold() {
        let arr = [15i64, 13, 11, 9, 7, 5, 3, 1]; // Descending

        assert_eq!(find_le_threshold_i64(&arr, 16), 0);
        assert_eq!(find_le_threshold_i64(&arr, 15), 0);
        assert_eq!(find_le_threshold_i64(&arr, 14), 1);
        assert_eq!(find_le_threshold_i64(&arr, 9), 3);
        assert_eq!(find_le_threshold_i64(&arr, 1), 7);
        assert_eq!(find_le_threshold_i64(&arr, 0), 8);
    }

    #[test]
    fn empty_arrays() {
        assert_eq!(find_ge_threshold_i64(&[], 42), 0);
        assert_eq!(find_le_threshold_i64(&[], 42), 0);
    }

    /// Reference implementations, straight from the definition.
    fn ref_ge(arr: &[i64], t: i64) -> usize {
        arr.iter().position(|&x| x >= t).unwrap_or(arr.len())
    }

    fn ref_le(arr: &[i64], t: i64) -> usize {
        arr.iter().position(|&x| x <= t).unwrap_or(arr.len())
    }

    /// Deterministic LCG so this test needs no external RNG.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }
    }

    /// Cross-check the hybrid scans against the reference on sizes that
    /// straddle `LINEAR_SCAN_MAX`, with duplicates and boundary thresholds.
    #[test]
    fn hybrid_matches_reference_across_window_boundary() {
        let mut rng = Lcg(0xC0FF_EE00_1234_5678);

        for size in [
            0usize,
            1,
            2,
            3,
            7,
            8,
            63,
            LINEAR_SCAN_MAX,
            LINEAR_SCAN_MAX + 1,
            100,
            127,
            128,
            129,
            255,
            1000,
        ] {
            // Sorted ascending with duplicates.
            let mut asc: Vec<i64> = (0..size).map(|_| (rng.next() % 500) as i64).collect();
            asc.sort_unstable();
            let mut desc = asc.clone();
            desc.reverse();

            // Probe every distinct value plus off-by-one and extreme thresholds.
            let mut thresholds: Vec<i64> = asc.clone();
            thresholds.extend(asc.iter().map(|v| v - 1));
            thresholds.extend(asc.iter().map(|v| v + 1));
            thresholds.extend([i64::MIN, i64::MAX, 0, -1]);

            for &t in &thresholds {
                assert_eq!(
                    find_ge_threshold_i64(&asc, t),
                    ref_ge(&asc, t),
                    "ge mismatch: size={size}, t={t}"
                );
                assert_eq!(
                    find_le_threshold_i64(&desc, t),
                    ref_le(&desc, t),
                    "le mismatch: size={size}, t={t}"
                );
            }
        }
    }
}
