//! ARM NEON SIMD implementations.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Finds the first index where `arr[i] >= threshold` using NEON.
///
/// # Safety
///
/// Caller must ensure NEON is available (always on aarch64).
#[cfg(target_arch = "aarch64")]
pub unsafe fn find_ge_threshold_neon(arr: &[i64], threshold: i64) -> usize {
    let len = arr.len();
    if len == 0 {
        return 0;
    }

    let threshold_vec = vdupq_n_s64(threshold);
    let ptr = arr.as_ptr();

    let mut i = 0;

    // Process 2 elements at a time
    while i + 2 <= len {
        let values = vld1q_s64(ptr.add(i));

        // Compare: values >= threshold (returns uint64x2_t mask)
        let ge = vcgeq_s64(values, threshold_vec);

        // Extract mask - ge is already uint64x2_t
        let mask = vgetq_lane_u64(ge, 0) != 0
            || vgetq_lane_u64(ge, 1) != 0;

        if mask {
            // Check individual elements
            if vgetq_lane_u64(ge, 0) != 0 {
                return i;
            }
            return i + 1;
        }

        i += 2;
    }

    // Handle remaining element
    if i < len && arr[i] >= threshold {
        return i;
    }

    len
}

/// Finds the first index where `arr[i] <= threshold` using NEON.
///
/// # Safety
///
/// Caller must ensure NEON is available (always on aarch64).
#[cfg(target_arch = "aarch64")]
pub unsafe fn find_le_threshold_neon(arr: &[i64], threshold: i64) -> usize {
    let len = arr.len();
    if len == 0 {
        return 0;
    }

    let threshold_vec = vdupq_n_s64(threshold);
    let ptr = arr.as_ptr();

    let mut i = 0;

    // Process 2 elements at a time
    while i + 2 <= len {
        let values = vld1q_s64(ptr.add(i));

        // Compare: values <= threshold (returns uint64x2_t mask)
        let le = vcleq_s64(values, threshold_vec);

        // Extract mask - le is already uint64x2_t
        let mask = vgetq_lane_u64(le, 0) != 0
            || vgetq_lane_u64(le, 1) != 0;

        if mask {
            // Check individual elements
            if vgetq_lane_u64(le, 0) != 0 {
                return i;
            }
            return i + 1;
        }

        i += 2;
    }

    // Handle remaining element
    if i < len && arr[i] <= threshold {
        return i;
    }

    len
}

/// Counts elements where `arr[i] > threshold` using NEON.
///
/// # Safety
///
/// Caller must ensure NEON is available (always on aarch64).
#[cfg(target_arch = "aarch64")]
pub unsafe fn count_gt_threshold_neon(arr: &[i64], threshold: i64) -> usize {
    let len = arr.len();
    if len == 0 {
        return 0;
    }

    let threshold_vec = vdupq_n_s64(threshold);
    let ptr = arr.as_ptr();
    let mut count: usize = 0;
    let mut i = 0;

    // Process 2 elements at a time
    while i + 2 <= len {
        let values = vld1q_s64(ptr.add(i));

        // Compare: values > threshold (returns uint64x2_t mask)
        let gt = vcgtq_s64(values, threshold_vec);

        // Count set bits in mask (each lane is all 1s or all 0s)
        // -1 in two's complement is all 1s, so we can use the sign bit
        if vgetq_lane_u64(gt, 0) != 0 {
            count += 1;
        }
        if vgetq_lane_u64(gt, 1) != 0 {
            count += 1;
        }

        i += 2;
    }

    // Handle remaining element
    if i < len && arr[i] > threshold {
        count += 1;
    }

    count
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    #[test]
    fn test_neon_find_ge() {
        let arr: Vec<i64> = (0..100).map(|i| i * 2).collect();

        unsafe {
            assert_eq!(find_ge_threshold_neon(&arr, 0), 0);
            assert_eq!(find_ge_threshold_neon(&arr, 1), 1);
            assert_eq!(find_ge_threshold_neon(&arr, 10), 5);
            assert_eq!(find_ge_threshold_neon(&arr, 198), 99);
            assert_eq!(find_ge_threshold_neon(&arr, 199), 100);
        }
    }

    #[test]
    fn test_neon_find_le() {
        let arr: Vec<i64> = (0..100).rev().map(|i| i * 2).collect();

        unsafe {
            assert_eq!(find_le_threshold_neon(&arr, 198), 0);
            assert_eq!(find_le_threshold_neon(&arr, 197), 1);
            assert_eq!(find_le_threshold_neon(&arr, 0), 99);
            assert_eq!(find_le_threshold_neon(&arr, -1), 100);
        }
    }
}
