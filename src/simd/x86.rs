//! x86_64 SIMD implementations using AVX2 and (optionally) AVX-512.
//!
//! The AVX-512 kernels are gated behind the `avx512` cargo feature because
//! the underlying intrinsics require Rust 1.89+.

#![allow(unsafe_code)]

use core::arch::x86_64::*;

// ============================================================================
// AVX2 implementations (256-bit, 4 × i64)
// ============================================================================

/// Finds the first index where `arr[i] >= threshold` using AVX2.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub unsafe fn find_ge_threshold_avx2(arr: &[i64], threshold: i64) -> usize {
    let len = arr.len();
    if len == 0 {
        return 0;
    }

    let threshold_vec = _mm256_set1_epi64x(threshold);
    let ptr = arr.as_ptr();

    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= len {
        let values = _mm256_loadu_si256(ptr.add(i).cast());

        // Compare: values >= threshold
        // _mm256_cmpgt_epi64 gives us values > threshold
        // We need >= so we check if NOT(threshold > values)
        let gt = _mm256_cmpgt_epi64(threshold_vec, values);
        let mask = _mm256_movemask_pd(_mm256_castsi256_pd(gt));

        // If any comparison is false (value >= threshold), we found our boundary
        if mask != 0xF {
            // Find first position where value >= threshold
            // mask bit is 0 where value >= threshold
            let inverted = (!mask) & 0xF;
            let first_ge = inverted.trailing_zeros() as usize;
            return i + first_ge;
        }

        i += 4;
    }

    // Handle remaining elements
    while i < len {
        if arr[i] >= threshold {
            return i;
        }
        i += 1;
    }

    len
}

/// Finds the first index where `arr[i] <= threshold` using AVX2.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub unsafe fn find_le_threshold_avx2(arr: &[i64], threshold: i64) -> usize {
    let len = arr.len();
    if len == 0 {
        return 0;
    }

    let threshold_vec = _mm256_set1_epi64x(threshold);
    let ptr = arr.as_ptr();

    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= len {
        let values = _mm256_loadu_si256(ptr.add(i).cast());

        // Compare: values > threshold (we want first where value <= threshold)
        let gt = _mm256_cmpgt_epi64(values, threshold_vec);
        let mask = _mm256_movemask_pd(_mm256_castsi256_pd(gt));

        // If any comparison is false (value <= threshold), we found our boundary
        if mask != 0xF {
            // Find first position where value <= threshold
            let inverted = (!mask) & 0xF;
            let first_le = inverted.trailing_zeros() as usize;
            return i + first_le;
        }

        i += 4;
    }

    // Handle remaining elements
    while i < len {
        if arr[i] <= threshold {
            return i;
        }
        i += 1;
    }

    len
}

// ============================================================================
// AVX-512 implementations (512-bit, 8 × i64) — `avx512` feature only
// ============================================================================

/// Finds the first index where `arr[i] >= threshold` using AVX-512.
///
/// # Safety
///
/// Caller must ensure AVX-512F is available.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f")]
pub unsafe fn find_ge_threshold_avx512(arr: &[i64], threshold: i64) -> usize {
    let len = arr.len();
    if len == 0 {
        return 0;
    }

    let threshold_vec = _mm512_set1_epi64(threshold);
    let ptr = arr.as_ptr();

    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= len {
        // `as *const _` lets inference pick the parameter type, which differs
        // across stdarch versions (*const __m512i since stabilization).
        let values = _mm512_loadu_si512(ptr.add(i) as *const _);

        // Compare: values >= threshold
        // _mm512_cmpge_epi64_mask returns a mask where bit is 1 if value >= threshold
        let mask = _mm512_cmpge_epi64_mask(values, threshold_vec);

        if mask != 0 {
            // Find first set bit
            let first_ge = mask.trailing_zeros() as usize;
            return i + first_ge;
        }

        i += 8;
    }

    // Handle remaining elements
    while i < len {
        if arr[i] >= threshold {
            return i;
        }
        i += 1;
    }

    len
}

/// Finds the first index where `arr[i] <= threshold` using AVX-512.
///
/// # Safety
///
/// Caller must ensure AVX-512F is available.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f")]
pub unsafe fn find_le_threshold_avx512(arr: &[i64], threshold: i64) -> usize {
    let len = arr.len();
    if len == 0 {
        return 0;
    }

    let threshold_vec = _mm512_set1_epi64(threshold);
    let ptr = arr.as_ptr();

    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= len {
        let values = _mm512_loadu_si512(ptr.add(i) as *const _);

        // Compare: values <= threshold
        let mask = _mm512_cmple_epi64_mask(values, threshold_vec);

        if mask != 0 {
            // Find first set bit
            let first_le = mask.trailing_zeros() as usize;
            return i + first_le;
        }

        i += 8;
    }

    // Handle remaining elements
    while i < len {
        if arr[i] <= threshold {
            return i;
        }
        i += 1;
    }

    len
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "std")]
    use super::*;

    #[test]
    #[cfg(feature = "std")]
    fn test_avx2_find_ge() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let arr: Vec<i64> = (0..100).map(|i| i * 2).collect();

        unsafe {
            assert_eq!(find_ge_threshold_avx2(&arr, 0), 0);
            assert_eq!(find_ge_threshold_avx2(&arr, 1), 1);
            assert_eq!(find_ge_threshold_avx2(&arr, 10), 5);
            assert_eq!(find_ge_threshold_avx2(&arr, 11), 6);
            assert_eq!(find_ge_threshold_avx2(&arr, 198), 99);
            assert_eq!(find_ge_threshold_avx2(&arr, 199), 100);
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_avx2_find_le() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let arr: Vec<i64> = (0..100).rev().map(|i| i * 2).collect();

        unsafe {
            assert_eq!(find_le_threshold_avx2(&arr, 200), 0);
            assert_eq!(find_le_threshold_avx2(&arr, 198), 0);
            assert_eq!(find_le_threshold_avx2(&arr, 197), 1);
            assert_eq!(find_le_threshold_avx2(&arr, 10), 94);
            assert_eq!(find_le_threshold_avx2(&arr, 0), 99);
            assert_eq!(find_le_threshold_avx2(&arr, -1), 100);
        }
    }

    #[test]
    #[cfg(all(feature = "std", feature = "avx512"))]
    fn test_avx512_find_ge() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        let arr: Vec<i64> = (0..100).map(|i| i * 2).collect();

        unsafe {
            assert_eq!(find_ge_threshold_avx512(&arr, 0), 0);
            assert_eq!(find_ge_threshold_avx512(&arr, 1), 1);
            assert_eq!(find_ge_threshold_avx512(&arr, 10), 5);
            assert_eq!(find_ge_threshold_avx512(&arr, 11), 6);
            assert_eq!(find_ge_threshold_avx512(&arr, 198), 99);
            assert_eq!(find_ge_threshold_avx512(&arr, 199), 100);
        }
    }

    #[test]
    #[cfg(all(feature = "std", feature = "avx512"))]
    fn test_avx512_find_le() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        let arr: Vec<i64> = (0..100).rev().map(|i| i * 2).collect();

        unsafe {
            assert_eq!(find_le_threshold_avx512(&arr, 200), 0);
            assert_eq!(find_le_threshold_avx512(&arr, 198), 0);
            assert_eq!(find_le_threshold_avx512(&arr, 197), 1);
            assert_eq!(find_le_threshold_avx512(&arr, 10), 94);
            assert_eq!(find_le_threshold_avx512(&arr, 0), 99);
            assert_eq!(find_le_threshold_avx512(&arr, -1), 100);
        }
    }
}
