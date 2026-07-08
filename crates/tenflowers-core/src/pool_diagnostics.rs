/// GPU Memory Pool Diagnostics — per-size-class fragmentation analysis.
///
/// This module models a memory pool subdivided into power-of-two (or custom) size
/// classes.  Each allocation is mapped to the smallest bucket whose size is ≥ the
/// requested size.  Fragmentation metrics are derived from the ratio of used vs.
/// free capacity within and across buckets.

use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// PoolBucketStats
// ---------------------------------------------------------------------------

/// Statistics for a single size-class bucket in the memory pool.
#[derive(Debug, Clone)]
pub struct PoolBucketStats {
    /// Allocation size class (upper boundary in bytes, e.g. 64, 128, 256 …).
    pub size_class: usize,
    /// Total number of blocks managed by this bucket.
    pub total_blocks: usize,
    /// Number of blocks currently in use.
    pub used_blocks: usize,
    /// Number of blocks currently free (`total_blocks - used_blocks`).
    pub free_blocks: usize,
    /// Bytes occupied by used blocks (`used_blocks × size_class`).
    pub bytes_used: usize,
    /// Bytes held in free blocks (`free_blocks × size_class`).
    pub bytes_free: usize,
}

// ---------------------------------------------------------------------------
// FragmentationReport
// ---------------------------------------------------------------------------

/// Comprehensive fragmentation report derived from pool bucket statistics.
#[derive(Debug, Clone)]
pub struct FragmentationReport {
    /// Overall fragmentation score in [0.0, 1.0].  0.0 = perfectly packed.
    pub overall_fragmentation: f32,
    /// Fraction of free space that cannot satisfy a new max-size request.
    pub external_fragmentation: f32,
    /// Fraction of allocated space wasted due to size-class rounding.
    pub internal_fragmentation: f32,
    /// Size of the largest contiguous free block in bytes.
    pub largest_free_block_bytes: usize,
    /// Total number of free blocks across all buckets.
    pub num_free_blocks: usize,
    /// Total free bytes across all buckets.
    pub total_free_bytes: usize,
    /// Per-bucket breakdown.
    pub bucket_stats: Vec<PoolBucketStats>,
}

// ---------------------------------------------------------------------------
// PoolDiagnostics
// ---------------------------------------------------------------------------

/// Default size-class boundaries (bytes).
const DEFAULT_BUCKETS: &[usize] = &[
    64, 128, 256, 512, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576,
];

/// Per-pool fragmentation tracker using size-class buckets.
///
/// Each call to [`record_alloc`] / [`record_dealloc`] atomically increments or
/// decrements the relevant bucket's used-block counter.  Fragmentation metrics
/// are computed on demand via [`generate_fragmentation_report`].
pub struct PoolDiagnostics {
    /// Upper-boundary sizes for each bucket.
    bucket_sizes: Vec<usize>,
    /// Number of used blocks per bucket (atomic for lock-free updates).
    bucket_used: Vec<AtomicU64>,
    /// Total blocks per bucket — each alloc that succeeds increments this once
    /// (blocks are never physically removed, mirroring pool semantics).
    bucket_total: Vec<AtomicU64>,
}

impl PoolDiagnostics {
    /// Construct a new diagnostics tracker with custom bucket boundaries.
    ///
    /// `bucket_sizes` must be non-empty and in ascending order.  Each bucket
    /// represents allocations whose size is in the range
    /// `(prev_boundary, size_class]`.
    pub fn new(bucket_sizes: Vec<usize>) -> Self {
        let n = bucket_sizes.len();
        let mut bucket_used = Vec::with_capacity(n);
        let mut bucket_total = Vec::with_capacity(n);
        for _ in 0..n {
            bucket_used.push(AtomicU64::new(0));
            bucket_total.push(AtomicU64::new(0));
        }
        Self {
            bucket_sizes,
            bucket_used,
            bucket_total,
        }
    }

    /// Construct a diagnostics tracker with the default bucket sizes.
    pub fn with_default_buckets() -> Self {
        Self::new(DEFAULT_BUCKETS.to_vec())
    }

    /// Return the bucket index for a given allocation size.
    ///
    /// Returns the index of the smallest bucket whose `size_class >= size`.
    /// If the size exceeds all defined buckets the last (largest) bucket is used.
    pub fn get_bucket_for_size(&self, size: usize) -> usize {
        for (i, &bucket_size) in self.bucket_sizes.iter().enumerate() {
            if size <= bucket_size {
                return i;
            }
        }
        self.bucket_sizes.len() - 1
    }

    /// Record an allocation of `size` bytes.
    pub fn record_alloc(&self, size: usize) {
        let idx = self.get_bucket_for_size(size);
        self.bucket_used[idx].fetch_add(1, Ordering::Relaxed);
        self.bucket_total[idx].fetch_add(1, Ordering::Relaxed);
    }

    /// Record a deallocation of `size` bytes.
    ///
    /// If the used-block count would underflow it is clamped to zero.
    pub fn record_dealloc(&self, size: usize) {
        let idx = self.get_bucket_for_size(size);
        let prev = self.bucket_used[idx].load(Ordering::Relaxed);
        if prev > 0 {
            self.bucket_used[idx].fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Generate a [`FragmentationReport`] from the current bucket counters.
    pub fn generate_fragmentation_report(&self) -> FragmentationReport {
        let mut bucket_stats = Vec::with_capacity(self.bucket_sizes.len());
        let mut total_free_bytes: usize = 0;
        let mut total_allocated_bytes: usize = 0;
        let mut largest_free_block_bytes: usize = 0;
        let mut num_free_blocks: usize = 0;
        let mut total_wasted_bytes: usize = 0;

        for (i, &size_class) in self.bucket_sizes.iter().enumerate() {
            let total_blocks = self.bucket_total[i].load(Ordering::Relaxed) as usize;
            let used_blocks = (self.bucket_used[i].load(Ordering::Relaxed) as usize)
                .min(total_blocks);
            let free_blocks = total_blocks.saturating_sub(used_blocks);

            let bytes_used = used_blocks.saturating_mul(size_class);
            let bytes_free = free_blocks.saturating_mul(size_class);

            total_free_bytes = total_free_bytes.saturating_add(bytes_free);
            total_allocated_bytes = total_allocated_bytes.saturating_add(bytes_used);
            num_free_blocks = num_free_blocks.saturating_add(free_blocks);

            if free_blocks > 0 && size_class > largest_free_block_bytes {
                largest_free_block_bytes = size_class;
            }

            // Internal fragmentation: wasted padding inside blocks.
            // Modelled as half of the size_class minus the previous boundary
            // for all used blocks in this bucket (worst-case average).
            let prev_boundary = if i == 0 { 0 } else { self.bucket_sizes[i - 1] };
            let padding_per_block = size_class.saturating_sub(prev_boundary) / 2;
            total_wasted_bytes =
                total_wasted_bytes.saturating_add(used_blocks.saturating_mul(padding_per_block));

            bucket_stats.push(PoolBucketStats {
                size_class,
                total_blocks,
                used_blocks,
                free_blocks,
                bytes_used,
                bytes_free,
            });
        }

        // External fragmentation: fraction of free bytes that can't be used for
        // a request of size `largest_free_block_bytes`.
        let external_fragmentation = if total_free_bytes > 0 {
            let unusable = total_free_bytes.saturating_sub(largest_free_block_bytes);
            (unusable as f32 / total_free_bytes as f32).clamp(0.0, 1.0)
        } else {
            0.0_f32
        };

        // Internal fragmentation: wasted bytes / total allocated bytes.
        let internal_fragmentation = if total_allocated_bytes > 0 {
            (total_wasted_bytes as f32 / total_allocated_bytes as f32).clamp(0.0, 1.0)
        } else {
            0.0_f32
        };

        let overall_fragmentation =
            ((external_fragmentation + internal_fragmentation) / 2.0_f32).clamp(0.0, 1.0);

        FragmentationReport {
            overall_fragmentation,
            external_fragmentation,
            internal_fragmentation,
            largest_free_block_bytes,
            num_free_blocks,
            total_free_bytes,
            bucket_stats,
        }
    }

    /// Reset all bucket counters to zero.
    pub fn reset(&self) {
        for i in 0..self.bucket_sizes.len() {
            self.bucket_used[i].store(0, Ordering::Relaxed);
            self.bucket_total[i].store(0, Ordering::Relaxed);
        }
    }
}

// ---------------------------------------------------------------------------
// Global instance
// ---------------------------------------------------------------------------

/// Process-wide pool diagnostics using the default size-class boundaries.
pub static GLOBAL_POOL_DIAGNOSTICS: Lazy<PoolDiagnostics> =
    Lazy::new(PoolDiagnostics::with_default_buckets);

/// Obtain a reference to the global [`PoolDiagnostics`].
pub fn get_pool_diagnostics() -> &'static PoolDiagnostics {
    &GLOBAL_POOL_DIAGNOSTICS
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> PoolDiagnostics {
        PoolDiagnostics::new(DEFAULT_BUCKETS.to_vec())
    }

    #[test]
    fn test_pool_diagnostics_bucket_allocation() {
        let pd = fresh();
        // 200 bytes should land in the 256-byte bucket (index 2)
        let idx = pd.get_bucket_for_size(200);
        assert_eq!(pd.bucket_sizes[idx], 256);
        pd.record_alloc(200);
        assert_eq!(pd.bucket_used[idx].load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_pool_diagnostics_fragmentation_report() {
        let pd = fresh();
        pd.record_alloc(100);
        pd.record_alloc(500);
        pd.record_dealloc(100);
        let report = pd.generate_fragmentation_report();
        assert!(report.overall_fragmentation >= 0.0);
        assert!(report.overall_fragmentation <= 1.0);
        assert!(report.external_fragmentation >= 0.0);
        assert!(report.internal_fragmentation >= 0.0);
    }

    #[test]
    fn test_pool_diagnostics_fragmentation_zero_when_empty() {
        let pd = fresh();
        let report = pd.generate_fragmentation_report();
        assert_eq!(report.overall_fragmentation, 0.0);
        assert_eq!(report.external_fragmentation, 0.0);
        assert_eq!(report.internal_fragmentation, 0.0);
        assert_eq!(report.total_free_bytes, 0);
    }

    #[test]
    fn test_pool_diagnostics_top_bucket() {
        let pd = fresh();
        // 2MB — larger than all defined buckets → last bucket
        pd.record_alloc(2 * 1024 * 1024);
        let last_idx = pd.bucket_sizes.len() - 1;
        assert_eq!(pd.bucket_used[last_idx].load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_pool_diagnostics_reset() {
        let pd = fresh();
        pd.record_alloc(64);
        pd.record_alloc(1024);
        pd.reset();
        for i in 0..pd.bucket_sizes.len() {
            assert_eq!(pd.bucket_used[i].load(Ordering::Relaxed), 0);
            assert_eq!(pd.bucket_total[i].load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn test_pool_diagnostics_dealloc_clamped() {
        let pd = fresh();
        // Dealloc without prior alloc should not underflow
        pd.record_dealloc(64);
        assert_eq!(pd.bucket_used[0].load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_pool_diagnostics_bucket_stats_len() {
        let pd = fresh();
        let report = pd.generate_fragmentation_report();
        assert_eq!(report.bucket_stats.len(), DEFAULT_BUCKETS.len());
    }
}
