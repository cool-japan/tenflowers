//! SIMD Capabilities Detection and Utilities
//!
//! This module provides platform capabilities detection and optimization utilities
//! for making runtime decisions about which SIMD algorithms to use.

/// Platform capabilities for optimization decisions
#[derive(Debug, Clone, Copy)]
pub struct SimdCapabilities {
    pub has_auto_vectorization: bool,
    pub has_target_features: bool,
    pub recommended_unroll_factor: usize,
    pub cache_line_size: usize,
}

/// SIMD capabilities detection and utilities
pub struct Capabilities;

impl Capabilities {
    /// Platform-specific capabilities detection (stable approach)
    pub fn detect_capabilities() -> SimdCapabilities {
        SimdCapabilities {
            has_auto_vectorization: true, // Compiler can auto-vectorize
            has_target_features: Self::has_target_feature_support(),
            recommended_unroll_factor: 8,
            cache_line_size: Self::estimate_cache_line_size(),
        }
    }

    /// Detect if target-specific features are available
    fn has_target_feature_support() -> bool {
        // Use stable feature detection when possible
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // On x86/x86_64, we can detect some features
            Self::detect_x86_features()
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            // For other architectures, assume basic vectorization support
            true
        }
    }

    /// Detect x86-specific SIMD features
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn detect_x86_features() -> bool {
        // Use is_x86_feature_detected! macro for runtime detection
        #[cfg(target_feature = "sse2")]
        {
            true // SSE2 is baseline for x86_64
        }

        #[cfg(not(target_feature = "sse2"))]
        {
            // Runtime detection for SSE2
            std::arch::is_x86_feature_detected!("sse2")
        }
    }

    /// Estimate cache line size for optimization
    pub fn estimate_cache_line_size() -> usize {
        // Most modern processors use 64-byte cache lines
        64
    }

    /// Get optimal block size for cache-friendly algorithms
    pub fn optimal_block_size(data_size: usize, element_size: usize) -> usize {
        let cache_line_size = Self::estimate_cache_line_size();
        let l1_cache_size = 32 * 1024; // Typical L1 cache size (32KB)

        // Calculate block size that fits well in L1 cache
        let max_block_elements = l1_cache_size / element_size / 4; // Use 1/4 of L1 cache
        let cache_line_elements = cache_line_size / element_size;

        // Round to cache line boundaries
        let optimal_size = (max_block_elements / cache_line_elements) * cache_line_elements;

        // Ensure it's not larger than the data itself and not too small
        optimal_size.min(data_size).max(cache_line_elements * 4)
    }

    /// Determine optimal unroll factor based on data size
    pub fn optimal_unroll_factor(data_size: usize) -> usize {
        match data_size {
            0..=32 => 1,     // Very small: no unrolling
            33..=128 => 2,   // Small: minimal unrolling
            129..=512 => 4,  // Medium: moderate unrolling
            513..=2048 => 8, // Large: standard unrolling
            _ => 16,         // Very large: aggressive unrolling
        }
    }

    /// Check if SIMD optimizations are beneficial for given size
    pub fn should_use_simd(data_size: usize) -> bool {
        // SIMD optimizations are typically beneficial for arrays larger than 8 elements
        data_size >= 8
    }

    /// Get memory alignment requirement for optimal SIMD performance
    pub fn memory_alignment_requirement() -> usize {
        // Most SIMD operations benefit from 16-byte or 32-byte alignment
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            32 // AVX requires 32-byte alignment for optimal performance
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            16 // Default to 16-byte alignment
        }
    }

    /// Check if data is properly aligned for SIMD operations
    pub fn is_aligned(ptr: *const f32) -> bool {
        let alignment = Self::memory_alignment_requirement();
        (ptr as usize) % alignment == 0
    }

    /// Calculate the number of SIMD lanes for f32 operations
    pub fn simd_lanes_f32() -> usize {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if Self::has_avx2_support() {
                8 // AVX2 can process 8 f32 values at once
            } else if Self::has_sse2_support() {
                4 // SSE2 can process 4 f32 values at once
            } else {
                1 // Fallback to scalar
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            4 // NEON can process 4 f32 values at once
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            1 // Fallback to scalar for other architectures
        }
    }

    /// Check for AVX2 support (x86/x86_64 only)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn has_avx2_support() -> bool {
        std::arch::is_x86_feature_detected!("avx2")
    }

    /// Check for SSE2 support (x86/x86_64 only)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn has_sse2_support() -> bool {
        std::arch::is_x86_feature_detected!("sse2")
    }

    /// Get platform-specific performance hints
    pub fn get_performance_hints() -> PerformanceHints {
        PerformanceHints {
            prefer_unrolling: Self::should_prefer_unrolling(),
            prefer_blocking: Self::should_prefer_blocking(),
            optimal_chunk_size: Self::get_optimal_chunk_size(),
            memory_prefetch_distance: Self::get_prefetch_distance(),
        }
    }

    /// Determine if loop unrolling is beneficial on this platform
    fn should_prefer_unrolling() -> bool {
        // Modern CPUs generally benefit from loop unrolling
        #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
        {
            true
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false // Conservative default for unknown architectures
        }
    }

    /// Determine if cache blocking is beneficial on this platform
    fn should_prefer_blocking() -> bool {
        // Blocking is beneficial for most architectures with hierarchical caches
        true
    }

    /// Get optimal chunk size for processing
    fn get_optimal_chunk_size() -> usize {
        let simd_lanes = Self::simd_lanes_f32();
        let cache_line_elements = Self::estimate_cache_line_size() / std::mem::size_of::<f32>();

        // Choose chunk size that's a multiple of SIMD lanes and fits cache line
        let base_chunk = simd_lanes.max(cache_line_elements);

        // Round up to next power of 2 for better performance
        base_chunk.next_power_of_two().min(64) // Cap at reasonable size
    }

    /// Get memory prefetch distance
    fn get_prefetch_distance() -> usize {
        // Prefetch a few cache lines ahead
        Self::estimate_cache_line_size() * 4
    }
}

/// Performance optimization hints for the current platform
#[derive(Debug, Clone)]
pub struct PerformanceHints {
    pub prefer_unrolling: bool,
    pub prefer_blocking: bool,
    pub optimal_chunk_size: usize,
    pub memory_prefetch_distance: usize,
}

impl Default for PerformanceHints {
    fn default() -> Self {
        Capabilities::get_performance_hints()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_capabilities() {
        let capabilities = Capabilities::detect_capabilities();

        // Basic sanity checks
        assert!(capabilities.recommended_unroll_factor > 0);
        assert!(capabilities.cache_line_size > 0);
        assert!(capabilities.cache_line_size.is_power_of_two());
    }

    #[test]
    fn test_optimal_block_size() {
        let data_size = 1024;
        let element_size = std::mem::size_of::<f32>();

        let block_size = Capabilities::optimal_block_size(data_size, element_size);

        // Block size should be reasonable
        assert!(block_size > 0);
        assert!(block_size <= data_size);

        // Should be a multiple of cache line elements for efficiency
        let cache_line_elements = Capabilities::estimate_cache_line_size() / element_size;
        assert_eq!(block_size % cache_line_elements, 0);
    }

    #[test]
    fn test_optimal_unroll_factor() {
        // Test different data sizes
        assert_eq!(Capabilities::optimal_unroll_factor(16), 1);
        assert_eq!(Capabilities::optimal_unroll_factor(64), 2);
        assert_eq!(Capabilities::optimal_unroll_factor(256), 4);
        assert_eq!(Capabilities::optimal_unroll_factor(1024), 8);
        assert_eq!(Capabilities::optimal_unroll_factor(4096), 16);
    }

    #[test]
    fn test_should_use_simd() {
        // Small arrays shouldn't use SIMD
        assert!(!Capabilities::should_use_simd(4));
        assert!(!Capabilities::should_use_simd(7));

        // Larger arrays should use SIMD
        assert!(Capabilities::should_use_simd(8));
        assert!(Capabilities::should_use_simd(32));
        assert!(Capabilities::should_use_simd(1024));
    }

    #[test]
    fn test_memory_alignment_requirement() {
        let alignment = Capabilities::memory_alignment_requirement();

        // Should be a power of 2 and at least 8 bytes
        assert!(alignment.is_power_of_two());
        assert!(alignment >= 8);
    }

    #[test]
    fn test_simd_lanes_f32() {
        let lanes = Capabilities::simd_lanes_f32();

        // Should be at least 1 and a power of 2
        assert!(lanes >= 1);
        assert!(lanes.is_power_of_two());
    }

    #[test]
    fn test_performance_hints() {
        let hints = Capabilities::get_performance_hints();

        // Basic sanity checks
        assert!(hints.optimal_chunk_size > 0);
        assert!(hints.optimal_chunk_size.is_power_of_two());
        assert!(hints.memory_prefetch_distance > 0);
    }

    #[test]
    fn test_cache_line_size() {
        let cache_line_size = Capabilities::estimate_cache_line_size();

        // Should be a reasonable power of 2
        assert!(cache_line_size.is_power_of_two());
        assert!(cache_line_size >= 32);
        assert!(cache_line_size <= 256);
    }
}
