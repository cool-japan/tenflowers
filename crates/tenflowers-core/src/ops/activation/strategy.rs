pub use super::core::ActivationStrategy;

/// Ultra-performance thresholds for different optimization strategies
pub const SIMD_THRESHOLD: usize = 1024; // Use SIMD for arrays >= 1K elements
pub const PARALLEL_THRESHOLD: usize = 10000; // Use parallel for arrays >= 10K elements
#[allow(dead_code)]
pub const GPU_THRESHOLD: usize = 100000; // Use GPU for arrays >= 100K elements
pub const APPROX_THRESHOLD: usize = 1000000; // Use approximations for arrays >= 1M elements

/// Select optimal strategy based on array size and function complexity
pub fn select_activation_strategy(elements: usize, is_transcendental: bool) -> ActivationStrategy {
    // For very large arrays, consider approximations for transcendental functions
    if elements >= APPROX_THRESHOLD && is_transcendental {
        return ActivationStrategy::Approximation;
    }

    // GPU acceleration for very large arrays
    #[cfg(feature = "gpu")]
    if elements >= GPU_THRESHOLD {
        return ActivationStrategy::Gpu;
    }

    // SIMD + Parallel for large arrays
    if elements >= PARALLEL_THRESHOLD {
        return ActivationStrategy::SimdParallel;
    }

    // Pure SIMD for medium arrays
    if elements >= SIMD_THRESHOLD {
        return ActivationStrategy::Simd;
    }

    // Sequential for small arrays
    ActivationStrategy::Sequential
}

/// Strategy configuration for specific activation functions
#[derive(Debug, Clone)]
pub struct ActivationConfig {
    pub prefer_simd: bool,
    pub prefer_parallel: bool,
    pub prefer_gpu: bool,
    pub allow_approximation: bool,
    pub transcendental: bool,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            prefer_simd: true,
            prefer_parallel: true,
            prefer_gpu: true,
            allow_approximation: false,
            transcendental: false,
        }
    }
}

impl ActivationConfig {
    pub fn new_relu() -> Self {
        Self {
            prefer_simd: true,
            prefer_parallel: true,
            prefer_gpu: true,
            allow_approximation: false,
            transcendental: false,
        }
    }

    pub fn new_sigmoid() -> Self {
        Self {
            prefer_simd: true,
            prefer_parallel: true,
            prefer_gpu: true,
            allow_approximation: true,
            transcendental: true,
        }
    }

    pub fn new_gelu() -> Self {
        Self {
            prefer_simd: true,
            prefer_parallel: true,
            prefer_gpu: true,
            allow_approximation: true,
            transcendental: true,
        }
    }

    pub fn select_strategy(&self, elements: usize) -> ActivationStrategy {
        // Custom strategy based on activation-specific preferences
        if elements >= APPROX_THRESHOLD && self.allow_approximation && self.transcendental {
            return ActivationStrategy::Approximation;
        }

        #[cfg(feature = "gpu")]
        if elements >= GPU_THRESHOLD && self.prefer_gpu {
            return ActivationStrategy::Gpu;
        }

        if elements >= PARALLEL_THRESHOLD && self.prefer_parallel {
            if self.prefer_simd {
                ActivationStrategy::SimdParallel
            } else {
                ActivationStrategy::Parallel
            }
        } else if elements >= SIMD_THRESHOLD && self.prefer_simd {
            ActivationStrategy::Simd
        } else {
            ActivationStrategy::Sequential
        }
    }
}
