//! SIMD-accelerated operations for interval scanning.
//!
//! This module provides optimized scanning operations using:
//! - AVX-512 on x86_64 (512-bit, 8 × i64) - preferred when available
//! - AVX2 on x86_64 (256-bit, 4 × i64)
//! - NEON on ARM (128-bit, 2 × i64)
//! - Scalar fallback for other architectures
//!
//! The primary operation is finding the cutoff index in a sorted array
//! where elements stop satisfying a comparison condition.

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "aarch64")]
mod arm;

mod scalar;

/// Finds the first index where `arr[i] >= threshold` in a sorted (ascending) array.
///
/// Returns `arr.len()` if no such element exists.
#[inline]
pub fn find_ge_threshold_i64(arr: &[i64], threshold: i64) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86::find_ge_threshold_avx512(arr, threshold) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { x86::find_ge_threshold_avx2(arr, threshold) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { arm::find_ge_threshold_neon(arr, threshold) };
    }

    #[allow(unreachable_code)]
    scalar::find_ge_threshold(arr, threshold)
}

/// Counts elements where `arr[i] > threshold` (unsorted array).
///
/// Uses SIMD to count in parallel.
#[inline]
pub fn count_gt_threshold_i64(arr: &[i64], threshold: i64) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86::count_gt_threshold_avx512(arr, threshold) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { x86::count_gt_threshold_avx2(arr, threshold) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { arm::count_gt_threshold_neon(arr, threshold) };
    }

    #[allow(unreachable_code)]
    scalar::count_gt_threshold(arr, threshold)
}

/// Finds the first index where `arr[i] <= threshold` in a sorted (descending) array.
///
/// Returns `arr.len()` if no such element exists.
#[inline]
pub fn find_le_threshold_i64(arr: &[i64], threshold: i64) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86::find_le_threshold_avx512(arr, threshold) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { x86::find_le_threshold_avx2(arr, threshold) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { arm::find_le_threshold_neon(arr, threshold) };
    }

    #[allow(unreachable_code)]
    scalar::find_le_threshold(arr, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
