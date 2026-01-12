//! Scalar fallback implementations.

/// Finds the first index where `arr[i] >= threshold` in a sorted (ascending) array.
#[inline]
pub fn find_ge_threshold<T: Ord>(arr: &[T], threshold: T) -> usize {
    // Binary search for the partition point
    arr.partition_point(|x| *x < threshold)
}

/// Finds the first index where `arr[i] <= threshold` in a sorted (descending) array.
#[inline]
pub fn find_le_threshold<T: Ord>(arr: &[T], threshold: T) -> usize {
    // Binary search for the partition point
    arr.partition_point(|x| *x > threshold)
}

/// Counts elements where `arr[i] > threshold` (unsorted array).
#[inline]
pub fn count_gt_threshold<T: Ord>(arr: &[T], threshold: T) -> usize {
    arr.iter().filter(|x| **x > threshold).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ge_threshold() {
        let arr = [1, 3, 5, 7, 9];
        assert_eq!(find_ge_threshold(&arr, 0), 0);
        assert_eq!(find_ge_threshold(&arr, 1), 0);
        assert_eq!(find_ge_threshold(&arr, 2), 1);
        assert_eq!(find_ge_threshold(&arr, 5), 2);
        assert_eq!(find_ge_threshold(&arr, 10), 5);
    }

    #[test]
    fn test_le_threshold() {
        let arr = [9, 7, 5, 3, 1]; // Descending
        assert_eq!(find_le_threshold(&arr, 10), 0);
        assert_eq!(find_le_threshold(&arr, 9), 0);
        assert_eq!(find_le_threshold(&arr, 8), 1);
        assert_eq!(find_le_threshold(&arr, 5), 2);
        assert_eq!(find_le_threshold(&arr, 0), 5);
    }
}
