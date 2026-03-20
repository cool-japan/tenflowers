//! Cache-friendly memory layout optimizations for improved performance
//!
//! This module provides cache optimization utilities including cache line alignment,
//! blocked matrix layouts, and memory access pattern optimization.

/// Cache line size detection and alignment utilities
pub struct CacheOptimizer {
    cache_line_size: usize,
    l1_cache_size: usize,
    l2_cache_size: usize,
    tlb_page_size: usize,
}

impl CacheOptimizer {
    /// Create new cache optimizer with detected parameters
    pub fn new() -> Self {
        Self {
            cache_line_size: Self::detect_cache_line_size(),
            l1_cache_size: Self::detect_l1_cache_size(),
            l2_cache_size: Self::detect_l2_cache_size(),
            tlb_page_size: Self::detect_page_size(),
        }
    }

    /// Detect cache line size (typically 64 bytes on modern processors)
    fn detect_cache_line_size() -> usize {
        // Most modern processors use 64-byte cache lines
        // ARM and x86-64 typically use 64 bytes
        64
    }

    /// Detect L1 cache size (typically 32KB)
    fn detect_l1_cache_size() -> usize {
        32 * 1024 // 32KB typical L1 data cache
    }

    /// Detect L2 cache size (typically 256KB-512KB)
    fn detect_l2_cache_size() -> usize {
        256 * 1024 // 256KB typical L2 cache
    }

    /// Detect TLB page size (typically 4KB)
    fn detect_page_size() -> usize {
        4096 // 4KB pages on most systems
    }

    /// Get optimal alignment for data structures
    pub fn get_optimal_alignment(&self, data_size: usize) -> usize {
        if data_size <= self.cache_line_size {
            // Align to cache line for small structures
            self.cache_line_size
        } else if data_size <= self.tlb_page_size {
            // Align to page boundary for medium structures
            self.tlb_page_size
        } else {
            // For large structures, align to multiple of page size
            ((data_size + self.tlb_page_size - 1) / self.tlb_page_size) * self.tlb_page_size
        }
    }

    /// Calculate optimal block size for cache-friendly iteration
    pub fn get_optimal_block_size(&self, element_size: usize, total_elements: usize) -> usize {
        let target_bytes = self.l1_cache_size / 2; // Use half of L1 for working set
        let elements_per_block = target_bytes / element_size;

        // Ensure block size is reasonable (not too small, not larger than total)
        elements_per_block
            .max(64) // Minimum block size
            .min(total_elements) // Don't exceed total elements
            .min(8192) // Maximum practical block size
    }

    /// Suggest memory access pattern optimization
    pub fn optimize_access_pattern(&self, dims: &[usize], element_size: usize) -> AccessPattern {
        let total_size = dims.iter().product::<usize>() * element_size;

        if total_size <= self.l1_cache_size {
            AccessPattern::Sequential // Fits in L1, sequential is best
        } else if total_size <= self.l2_cache_size {
            AccessPattern::Blocked {
                block_size: self.get_optimal_block_size(element_size, dims[dims.len() - 1]),
            }
        } else {
            AccessPattern::Tiled {
                tile_dims: self.calculate_optimal_tile_size(dims, element_size),
            }
        }
    }

    /// Calculate optimal tile dimensions for large matrices
    fn calculate_optimal_tile_size(&self, dims: &[usize], element_size: usize) -> Vec<usize> {
        let target_bytes = self.l2_cache_size / 3; // Use 1/3 of L2 for each tile
        let mut tile_dims = Vec::with_capacity(dims.len());

        for &dim_size in dims {
            let max_tile_elements = target_bytes / element_size;
            let optimal_tile_size =
                (max_tile_elements as f64).powf(1.0 / dims.len() as f64) as usize;
            tile_dims.push(optimal_tile_size.min(dim_size).max(8)); // Min 8, max original dim
        }

        tile_dims
    }

    /// Get cache line size
    pub fn cache_line_size(&self) -> usize {
        self.cache_line_size
    }

    /// Get L1 cache size
    pub fn l1_cache_size(&self) -> usize {
        self.l1_cache_size
    }

    /// Get L2 cache size
    pub fn l2_cache_size(&self) -> usize {
        self.l2_cache_size
    }

    /// Get page size
    pub fn page_size(&self) -> usize {
        self.tlb_page_size
    }
}

impl Default for CacheOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory access pattern recommendations
#[derive(Debug, Clone)]
pub enum AccessPattern {
    Sequential,
    Blocked { block_size: usize },
    Tiled { tile_dims: Vec<usize> },
}

/// Cache-friendly matrix layout transformer
pub struct MatrixLayoutOptimizer {
    optimizer: CacheOptimizer,
}

impl MatrixLayoutOptimizer {
    /// Create new matrix layout optimizer
    pub fn new() -> Self {
        Self {
            optimizer: CacheOptimizer::new(),
        }
    }

    /// Transform row-major matrix to cache-friendly blocked layout
    pub fn to_blocked_layout<T: Copy>(
        &self,
        data: &[T],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) -> Vec<T> {
        let mut blocked_data = Vec::with_capacity(data.len());

        // Iterate over blocks
        for block_row in (0..rows).step_by(block_size) {
            for block_col in (0..cols).step_by(block_size) {
                // Process each block
                for row in block_row..(block_row + block_size).min(rows) {
                    for col in block_col..(block_col + block_size).min(cols) {
                        let index = row * cols + col;
                        blocked_data.push(data[index]);
                    }
                }
            }
        }

        blocked_data
    }

    /// Transform blocked layout back to row-major
    pub fn from_blocked_layout<T: Copy>(
        &self,
        blocked_data: &[T],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) -> Vec<T> {
        let mut data = vec![blocked_data[0]; blocked_data.len()]; // Initialize with first element
        let mut blocked_idx = 0;

        // Iterate over blocks in the same order as to_blocked_layout
        for block_row in (0..rows).step_by(block_size) {
            for block_col in (0..cols).step_by(block_size) {
                // Process each element in the block
                for row in block_row..(block_row + block_size).min(rows) {
                    for col in block_col..(block_col + block_size).min(cols) {
                        let index = row * cols + col;
                        data[index] = blocked_data[blocked_idx];
                        blocked_idx += 1;
                    }
                }
            }
        }

        data
    }

    /// Suggest optimal matrix multiplication blocking parameters
    pub fn suggest_gemm_blocking(
        &self,
        m: usize,
        n: usize,
        k: usize,
        element_size: usize,
    ) -> (usize, usize, usize) {
        let target_bytes = self.optimizer.l2_cache_size / 3; // Divide L2 among A, B, C blocks
        let target_elements = target_bytes / element_size;

        // Calculate block sizes that fit in cache
        let mk_block = (target_elements / n).max(8).min(m);
        let nk_block = (target_elements / m).max(8).min(n);
        let k_block = (target_elements / (mk_block + nk_block)).max(8).min(k);

        (mk_block, nk_block, k_block)
    }
}

impl Default for MatrixLayoutOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory prefetching hints for improved cache performance
pub struct PrefetchOptimizer {
    prefetch_distance: usize,
    stride_threshold: usize,
}

impl PrefetchOptimizer {
    /// Create new prefetch optimizer
    pub fn new() -> Self {
        Self {
            prefetch_distance: 64, // Prefetch 64 elements ahead
            stride_threshold: 4,   // Use prefetching for strides >= 4
        }
    }

    /// Add software prefetch hints for sequential access
    pub fn prefetch_sequential<T>(&self, data: &[T], current_index: usize) {
        let prefetch_index = current_index + self.prefetch_distance;
        if prefetch_index < data.len() {
            // Hint to prefetch data into cache
            // In real implementation, this would use platform-specific intrinsics
            // For now, we just touch the memory to simulate prefetching
            let _prefetch_ptr = &data[prefetch_index] as *const T;
        }
    }

    /// Add software prefetch hints for strided access
    pub fn prefetch_strided<T>(&self, data: &[T], current_index: usize, stride: usize) {
        if stride >= self.stride_threshold {
            let prefetch_index = current_index + stride * self.prefetch_distance;
            if prefetch_index < data.len() {
                // Prefetch next element in strided pattern
                let _prefetch_ptr = &data[prefetch_index] as *const T;
            }
        }
    }

    /// Suggest whether to use prefetching for given access pattern
    pub fn should_prefetch(&self, access_stride: usize, data_size: usize) -> bool {
        // Use prefetching for:
        // 1. Large datasets (> 1MB)
        // 2. Strided access patterns
        // 3. When stride is large enough to benefit from prefetching
        data_size > 1024 * 1024 || access_stride >= self.stride_threshold
    }
}

impl Default for PrefetchOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Global cache optimizer instance
static GLOBAL_CACHE_OPTIMIZER: std::sync::OnceLock<CacheOptimizer> = std::sync::OnceLock::new();

/// Get the global cache optimizer
pub fn global_cache_optimizer() -> &'static CacheOptimizer {
    GLOBAL_CACHE_OPTIMIZER.get_or_init(CacheOptimizer::new)
}

/// Utility function to align size to cache line boundary
pub fn align_to_cache_line(size: usize) -> usize {
    let cache_line_size = global_cache_optimizer().cache_line_size;
    (size + cache_line_size - 1) & !(cache_line_size - 1)
}

/// Utility function to check if memory is cache-line aligned
pub fn is_cache_aligned(ptr: *const u8) -> bool {
    let cache_line_size = global_cache_optimizer().cache_line_size;
    (ptr as usize) & (cache_line_size - 1) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_optimizer_creation() {
        let optimizer = CacheOptimizer::new();
        assert_eq!(optimizer.cache_line_size(), 64);
        assert_eq!(optimizer.l1_cache_size(), 32 * 1024);
        assert_eq!(optimizer.l2_cache_size(), 256 * 1024);
        assert_eq!(optimizer.page_size(), 4096);
    }

    #[test]
    fn test_optimal_alignment() {
        let optimizer = CacheOptimizer::new();

        // Small data should align to cache line
        assert_eq!(optimizer.get_optimal_alignment(32), 64);

        // Medium data should align to page
        assert_eq!(optimizer.get_optimal_alignment(1024), 4096);

        // Large data should align to page multiple
        let large_size = 8192;
        let alignment = optimizer.get_optimal_alignment(large_size);
        assert!(alignment >= large_size);
        assert_eq!(alignment % 4096, 0);
    }

    #[test]
    fn test_optimal_block_size() {
        let optimizer = CacheOptimizer::new();

        let block_size = optimizer.get_optimal_block_size(4, 1000);
        assert!(block_size >= 64); // Minimum block size
        assert!(block_size <= 1000); // Not larger than total
        assert!(block_size <= 8192); // Maximum practical block size
    }

    #[test]
    fn test_access_pattern_optimization() {
        let optimizer = CacheOptimizer::new();

        // Small data should use sequential access
        let pattern = optimizer.optimize_access_pattern(&[100, 100], 4);
        matches!(pattern, AccessPattern::Sequential);

        // Large data should use tiled access
        let pattern = optimizer.optimize_access_pattern(&[1000, 1000], 4);
        matches!(pattern, AccessPattern::Tiled { .. });
    }

    #[test]
    fn test_tile_size_calculation() {
        let optimizer = CacheOptimizer::new();
        let dims = vec![1000, 1000];
        let tile_dims = optimizer.calculate_optimal_tile_size(&dims, 4);

        assert_eq!(tile_dims.len(), 2);
        for &tile_dim in &tile_dims {
            assert!(tile_dim >= 8); // Minimum tile size
            assert!(tile_dim <= 1000); // Not larger than original dimension
        }
    }

    #[test]
    fn test_matrix_layout_optimizer() {
        let optimizer = MatrixLayoutOptimizer::new();

        // Test blocking transformation
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; // 3x4 matrix
        let blocked = optimizer.to_blocked_layout(&data, 3, 4, 2);

        // Should have same number of elements
        assert_eq!(blocked.len(), data.len());

        // Test reverse transformation
        let restored = optimizer.from_blocked_layout(&blocked, 3, 4, 2);
        assert_eq!(restored, data);
    }

    #[test]
    fn test_gemm_blocking_suggestion() {
        let optimizer = MatrixLayoutOptimizer::new();
        let (mb, nb, kb) = optimizer.suggest_gemm_blocking(1000, 1000, 1000, 4);

        // All block sizes should be reasonable
        assert!(mb >= 8 && mb <= 1000);
        assert!(nb >= 8 && nb <= 1000);
        assert!(kb >= 8 && kb <= 1000);
    }

    #[test]
    fn test_prefetch_optimizer() {
        let optimizer = PrefetchOptimizer::new();

        // Test prefetch decision
        assert!(optimizer.should_prefetch(8, 2 * 1024 * 1024)); // Large data, large stride
        assert!(!optimizer.should_prefetch(2, 1024)); // Small data, small stride
    }

    #[test]
    fn test_cache_alignment_utilities() {
        // Test alignment function
        assert_eq!(align_to_cache_line(50), 64);
        assert_eq!(align_to_cache_line(64), 64);
        assert_eq!(align_to_cache_line(65), 128);

        // Test alignment checking
        let aligned_ptr = 0x1000 as *const u8; // 64-byte aligned
        let unaligned_ptr = 0x1001 as *const u8; // Not aligned

        assert!(is_cache_aligned(aligned_ptr));
        assert!(!is_cache_aligned(unaligned_ptr));
    }

    #[test]
    fn test_global_cache_optimizer() {
        let optimizer1 = global_cache_optimizer();
        let optimizer2 = global_cache_optimizer();

        // Should be the same instance
        assert!(std::ptr::eq(optimizer1, optimizer2));

        // Should have expected values
        assert_eq!(optimizer1.cache_line_size(), 64);
    }

    #[test]
    fn test_access_patterns() {
        let sequential = AccessPattern::Sequential;
        matches!(sequential, AccessPattern::Sequential);

        let blocked = AccessPattern::Blocked { block_size: 128 };
        if let AccessPattern::Blocked { block_size } = blocked {
            assert_eq!(block_size, 128);
        }

        let tiled = AccessPattern::Tiled {
            tile_dims: vec![64, 64],
        };
        if let AccessPattern::Tiled { tile_dims } = tiled {
            assert_eq!(tile_dims, vec![64, 64]);
        }
    }

    #[test]
    fn test_prefetch_methods() {
        let optimizer = PrefetchOptimizer::new();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // These methods don't return values, just test they don't panic
        optimizer.prefetch_sequential(&data, 0);
        optimizer.prefetch_strided(&data, 0, 2);

        // Test edge cases
        optimizer.prefetch_sequential(&data, data.len() - 1); // Near end
        optimizer.prefetch_strided(&data, data.len() - 1, 1); // Near end with stride
    }
}
