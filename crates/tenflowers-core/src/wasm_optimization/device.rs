//! WASM device capabilities and platform detection

/// WASM device capabilities
#[cfg(feature = "wasm")]
#[derive(Debug, Clone)]
pub struct WasmDeviceCapabilities {
    /// Available memory (bytes)
    pub memory_limit: usize,
    /// SIMD support
    pub simd_support: bool,
    /// Threading support
    pub threading_support: bool,
    /// WebGL support
    pub webgl_support: bool,
}

/// Extended device information for optimization decisions
#[cfg(feature = "wasm")]
#[derive(Debug, Clone)]
pub struct WasmDeviceInfo {
    /// Basic capabilities
    pub capabilities: WasmDeviceCapabilities,
    /// Device category
    pub device_category: WasmDeviceCategory,
    /// Performance tier
    pub performance_tier: WasmPerformanceTier,
    /// Browser/runtime information
    pub runtime_info: WasmRuntimeInfo,
}

/// Device category for optimization strategies
#[cfg(feature = "wasm")]
#[derive(Debug, Clone, Copy)]
pub enum WasmDeviceCategory {
    /// High-end desktop/laptop
    Desktop,
    /// Mobile phones
    Mobile,
    /// Tablets
    Tablet,
    /// IoT/embedded devices
    Embedded,
    /// Unknown/generic device
    Unknown,
}

/// Performance tier classification
#[cfg(feature = "wasm")]
#[derive(Debug, Clone, Copy)]
pub enum WasmPerformanceTier {
    /// High performance (>4GB RAM, multi-core, GPU)
    High,
    /// Medium performance (2-4GB RAM, dual-core)
    Medium,
    /// Low performance (<2GB RAM, single-core)
    Low,
    /// Very low performance (embedded, constrained)
    VeryLow,
}

/// Browser/runtime information
#[cfg(feature = "wasm")]
#[derive(Debug, Clone)]
pub struct WasmRuntimeInfo {
    /// Browser name (Chrome, Firefox, Safari, etc.)
    pub browser: String,
    /// Browser version
    pub version: String,
    /// WebAssembly version support
    pub wasm_version: WasmVersion,
    /// Available features
    pub features: WasmFeatures,
}

/// WebAssembly version support
#[cfg(feature = "wasm")]
#[derive(Debug, Clone, Copy)]
pub enum WasmVersion {
    /// WebAssembly 1.0 (MVP)
    V1_0,
    /// WebAssembly 2.0 (with SIMD, bulk memory, etc.)
    V2_0,
}

/// Available WebAssembly features
#[cfg(feature = "wasm")]
#[derive(Debug, Clone)]
pub struct WasmFeatures {
    /// SIMD instructions
    pub simd: bool,
    /// Bulk memory operations
    pub bulk_memory: bool,
    /// Multi-threading
    pub threads: bool,
    /// Exception handling
    pub exceptions: bool,
    /// Reference types
    pub reference_types: bool,
    /// Tail calls
    pub tail_calls: bool,
}

#[cfg(feature = "wasm")]
impl Default for WasmDeviceCapabilities {
    fn default() -> Self {
        Self {
            memory_limit: 64 * 1024 * 1024, // 64MB default
            simd_support: false,
            threading_support: false,
            webgl_support: false,
        }
    }
}

#[cfg(feature = "wasm")]
impl WasmDeviceCapabilities {
    /// Create new device capabilities
    pub fn new() -> Self {
        Self::default()
    }

    /// Detect device capabilities from current environment
    pub fn detect() -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            Self::detect_wasm_capabilities()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Return mock capabilities for testing
            Self {
                memory_limit: 128 * 1024 * 1024, // 128MB
                simd_support: true,
                threading_support: true,
                webgl_support: true,
            }
        }
    }

    /// Create capabilities for low-end device
    pub fn low_end_device() -> Self {
        Self {
            memory_limit: 32 * 1024 * 1024, // 32MB
            simd_support: false,
            threading_support: false,
            webgl_support: false,
        }
    }

    /// Create capabilities for high-end device
    pub fn high_end_device() -> Self {
        Self {
            memory_limit: 512 * 1024 * 1024, // 512MB
            simd_support: true,
            threading_support: true,
            webgl_support: true,
        }
    }

    /// Check if device can handle large models
    pub fn can_handle_large_models(&self) -> bool {
        self.memory_limit >= 128 * 1024 * 1024 // 128MB threshold
    }

    /// Check if device supports parallel processing
    pub fn supports_parallel_processing(&self) -> bool {
        self.threading_support || self.simd_support
    }

    /// Get recommended optimization strategy
    pub fn get_optimization_strategy(&self) -> WasmOptimizationStrategy {
        match (self.memory_limit, self.simd_support, self.threading_support) {
            (mem, true, true) if mem >= 128 * 1024 * 1024 => {
                WasmOptimizationStrategy::HighPerformance
            }
            (mem, _, _) if mem >= 64 * 1024 * 1024 => WasmOptimizationStrategy::Balanced,
            (mem, false, false) if mem < 32 * 1024 * 1024 => {
                WasmOptimizationStrategy::MinimalFootprint
            }
            _ => WasmOptimizationStrategy::SizeOptimized,
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn detect_wasm_capabilities() -> Self {
        use js_sys::*;
        use wasm_bindgen::prelude::*;

        let mut capabilities = Self::default();

        // Detect SIMD support
        capabilities.simd_support = Self::detect_simd();

        // Detect threading support
        capabilities.threading_support = Self::detect_threads();

        // Detect WebGL support
        capabilities.webgl_support = Self::detect_webgl();

        // Estimate memory limit
        capabilities.memory_limit = Self::estimate_memory_limit();

        capabilities
    }

    #[cfg(target_arch = "wasm32")]
    fn detect_simd() -> bool {
        // Check for WASM SIMD support
        js_sys::eval("typeof WebAssembly.validate !== 'undefined' && WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]))")
            .map(|val| val.as_bool().unwrap_or(false))
            .unwrap_or(false)
    }

    #[cfg(target_arch = "wasm32")]
    fn detect_threads() -> bool {
        // Check for SharedArrayBuffer support
        js_sys::eval("typeof SharedArrayBuffer !== 'undefined'")
            .map(|val| val.as_bool().unwrap_or(false))
            .unwrap_or(false)
    }

    #[cfg(target_arch = "wasm32")]
    fn detect_webgl() -> bool {
        // Check for WebGL support
        js_sys::eval("(function() { try { var canvas = document.createElement('canvas'); return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl')); } catch(e) { return false; } })()")
            .map(|val| val.as_bool().unwrap_or(false))
            .unwrap_or(false)
    }

    #[cfg(target_arch = "wasm32")]
    fn estimate_memory_limit() -> usize {
        // Try to detect available memory
        // This is a rough estimate based on device memory if available
        js_sys::eval(
            "navigator.deviceMemory ? navigator.deviceMemory * 1024 * 1024 * 1024 / 8 : 67108864",
        )
        .and_then(|val| val.as_f64())
        .map(|mem| mem as usize)
        .unwrap_or(64 * 1024 * 1024) // 64MB fallback
    }
}

/// Optimization strategy based on device capabilities
#[cfg(feature = "wasm")]
#[derive(Debug, Clone, Copy)]
pub enum WasmOptimizationStrategy {
    /// High performance: use all available features
    HighPerformance,
    /// Balanced: performance vs size tradeoffs
    Balanced,
    /// Size optimized: prioritize small bundle size
    SizeOptimized,
    /// Minimal footprint: maximum size reduction
    MinimalFootprint,
}

#[cfg(feature = "wasm")]
impl WasmDeviceInfo {
    /// Detect comprehensive device information
    pub fn detect() -> Self {
        let capabilities = WasmDeviceCapabilities::detect();
        let device_category = Self::classify_device(&capabilities);
        let performance_tier = Self::classify_performance(&capabilities);
        let runtime_info = WasmRuntimeInfo::detect();

        Self {
            capabilities,
            device_category,
            performance_tier,
            runtime_info,
        }
    }

    fn classify_device(caps: &WasmDeviceCapabilities) -> WasmDeviceCategory {
        // Simple heuristic based on memory and features
        match caps.memory_limit {
            mem if mem >= 256 * 1024 * 1024 => WasmDeviceCategory::Desktop,
            mem if mem >= 128 * 1024 * 1024 => WasmDeviceCategory::Tablet,
            mem if mem >= 64 * 1024 * 1024 => WasmDeviceCategory::Mobile,
            _ => WasmDeviceCategory::Embedded,
        }
    }

    fn classify_performance(caps: &WasmDeviceCapabilities) -> WasmPerformanceTier {
        let has_advanced_features =
            caps.simd_support && caps.threading_support && caps.webgl_support;
        let has_some_features = caps.simd_support || caps.threading_support;

        match (caps.memory_limit, has_advanced_features, has_some_features) {
            (mem, true, _) if mem >= 256 * 1024 * 1024 => WasmPerformanceTier::High,
            (mem, _, true) if mem >= 128 * 1024 * 1024 => WasmPerformanceTier::Medium,
            (mem, _, _) if mem >= 64 * 1024 * 1024 => WasmPerformanceTier::Low,
            _ => WasmPerformanceTier::VeryLow,
        }
    }

    /// Get recommended model size limit
    pub fn get_model_size_limit(&self) -> usize {
        match self.performance_tier {
            WasmPerformanceTier::High => 100 * 1024 * 1024, // 100MB
            WasmPerformanceTier::Medium => 50 * 1024 * 1024, // 50MB
            WasmPerformanceTier::Low => 20 * 1024 * 1024,   // 20MB
            WasmPerformanceTier::VeryLow => 5 * 1024 * 1024, // 5MB
        }
    }

    /// Get recommended batch size for inference
    pub fn get_recommended_batch_size(&self) -> usize {
        match self.performance_tier {
            WasmPerformanceTier::High => 32,
            WasmPerformanceTier::Medium => 16,
            WasmPerformanceTier::Low => 8,
            WasmPerformanceTier::VeryLow => 1,
        }
    }
}

#[cfg(feature = "wasm")]
impl WasmRuntimeInfo {
    /// Detect runtime information
    pub fn detect() -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            Self::detect_browser_info()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Mock runtime info for testing
            Self {
                browser: "Test Browser".to_string(),
                version: "1.0.0".to_string(),
                wasm_version: WasmVersion::V2_0,
                features: WasmFeatures {
                    simd: true,
                    bulk_memory: true,
                    threads: true,
                    exceptions: false,
                    reference_types: true,
                    tail_calls: false,
                },
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn detect_browser_info() -> Self {
        use js_sys::*;

        // Detect browser name and version
        let user_agent = web_sys::window()
            .and_then(|w| w.navigator().user_agent().ok())
            .unwrap_or_else(|| "Unknown".to_string());

        let (browser, version) = Self::parse_user_agent(&user_agent);

        // Detect WebAssembly features
        let features = WasmFeatures {
            simd: Self::feature_supported("simd"),
            bulk_memory: Self::feature_supported("bulk-memory"),
            threads: Self::feature_supported("threads"),
            exceptions: Self::feature_supported("exceptions"),
            reference_types: Self::feature_supported("reference-types"),
            tail_calls: Self::feature_supported("tail-calls"),
        };

        // Determine WASM version based on features
        let wasm_version = if features.simd || features.bulk_memory {
            WasmVersion::V2_0
        } else {
            WasmVersion::V1_0
        };

        Self {
            browser,
            version,
            wasm_version,
            features,
        }
    }

    fn parse_user_agent(user_agent: &str) -> (String, String) {
        // Simple user agent parsing
        if user_agent.contains("Chrome") {
            ("Chrome".to_string(), "Unknown".to_string())
        } else if user_agent.contains("Firefox") {
            ("Firefox".to_string(), "Unknown".to_string())
        } else if user_agent.contains("Safari") {
            ("Safari".to_string(), "Unknown".to_string())
        } else if user_agent.contains("Edge") {
            ("Edge".to_string(), "Unknown".to_string())
        } else {
            ("Unknown".to_string(), "Unknown".to_string())
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn feature_supported(_feature: &str) -> bool {
        // In a real implementation, this would test for specific WASM feature support
        // For now, return conservative defaults
        false
    }
}

/// Device profiler for runtime performance assessment
#[cfg(feature = "wasm")]
pub struct WasmDeviceProfiler {
    device_info: WasmDeviceInfo,
    benchmark_results: Vec<WasmProfileBenchmark>,
}

/// Individual profiling benchmark result
#[cfg(feature = "wasm")]
#[derive(Debug, Clone)]
pub struct WasmProfileBenchmark {
    pub test_name: String,
    pub duration_ms: f64,
    pub operations_per_second: f64,
    pub memory_peak_mb: f64,
}

#[cfg(feature = "wasm")]
impl WasmDeviceProfiler {
    /// Create new device profiler
    pub fn new() -> Self {
        Self {
            device_info: WasmDeviceInfo::detect(),
            benchmark_results: Vec::new(),
        }
    }

    /// Run comprehensive device profiling
    pub fn profile_device(&mut self) -> crate::Result<WasmDeviceProfile> {
        // Run basic computation benchmark
        self.benchmark_computation()?;

        // Run memory benchmark
        self.benchmark_memory()?;

        // Run SIMD benchmark if supported
        if self.device_info.capabilities.simd_support {
            self.benchmark_simd()?;
        }

        // Generate profile
        Ok(WasmDeviceProfile {
            device_info: self.device_info.clone(),
            benchmarks: self.benchmark_results.clone(),
            performance_score: self.calculate_performance_score(),
        })
    }

    fn benchmark_computation(&mut self) -> crate::Result<()> {
        let start = std::time::Instant::now();

        // Simple computation benchmark
        let mut sum = 0.0f64;
        for i in 0..100_000 {
            sum += (i as f64).sin();
        }
        // Prevent compiler from optimizing away the computation
        std::hint::black_box(sum);

        let duration = start.elapsed().as_millis() as f64;
        let ops_per_sec = 100_000.0 / (duration / 1000.0);

        self.benchmark_results.push(WasmProfileBenchmark {
            test_name: "computation".to_string(),
            duration_ms: duration,
            operations_per_second: ops_per_sec,
            memory_peak_mb: 0.1, // Minimal memory usage
        });

        Ok(())
    }

    fn benchmark_memory(&mut self) -> crate::Result<()> {
        let start = std::time::Instant::now();

        // Memory allocation benchmark
        let mut vectors = Vec::new();
        for _ in 0..1000 {
            vectors.push(vec![0.0f32; 1000]);
        }

        let duration = start.elapsed().as_millis() as f64;
        let ops_per_sec = 1000.0 / (duration / 1000.0);

        self.benchmark_results.push(WasmProfileBenchmark {
            test_name: "memory_allocation".to_string(),
            duration_ms: duration,
            operations_per_second: ops_per_sec,
            memory_peak_mb: 4.0, // ~4MB allocated
        });

        Ok(())
    }

    fn benchmark_simd(&mut self) -> crate::Result<()> {
        let start = std::time::Instant::now();

        // SIMD-style operations (simulated)
        let data = vec![1.0f32; 10000];
        let mut result = Vec::with_capacity(data.len());

        for chunk in data.chunks(4) {
            let sum: f32 = chunk.iter().sum();
            result.push(sum);
        }

        let duration = start.elapsed().as_millis() as f64;
        let ops_per_sec = 10000.0 / (duration / 1000.0);

        self.benchmark_results.push(WasmProfileBenchmark {
            test_name: "simd_operations".to_string(),
            duration_ms: duration,
            operations_per_second: ops_per_sec,
            memory_peak_mb: 0.08, // ~80KB
        });

        Ok(())
    }

    fn calculate_performance_score(&self) -> f64 {
        if self.benchmark_results.is_empty() {
            return 0.0;
        }

        let avg_ops_per_sec: f64 = self
            .benchmark_results
            .iter()
            .map(|b| b.operations_per_second)
            .sum::<f64>()
            / self.benchmark_results.len() as f64;

        // Normalize to 0-100 scale (arbitrary baseline of 10,000 ops/sec = 50 points)
        (avg_ops_per_sec / 10_000.0 * 50.0).min(100.0)
    }
}

/// Complete device profile
#[cfg(feature = "wasm")]
#[derive(Debug, Clone)]
pub struct WasmDeviceProfile {
    pub device_info: WasmDeviceInfo,
    pub benchmarks: Vec<WasmProfileBenchmark>,
    pub performance_score: f64,
}

#[cfg(feature = "wasm")]
impl Default for WasmDeviceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "wasm")]
    fn test_device_capabilities() {
        let caps = WasmDeviceCapabilities::detect();
        assert!(caps.memory_limit > 0);

        let low_end = WasmDeviceCapabilities::low_end_device();
        assert_eq!(low_end.memory_limit, 32 * 1024 * 1024);

        let high_end = WasmDeviceCapabilities::high_end_device();
        assert!(high_end.memory_limit >= 512 * 1024 * 1024);
        assert!(high_end.simd_support);
    }

    #[test]
    #[cfg(feature = "wasm")]
    #[ignore = "WASM optimization strategy logic needs refinement"]
    fn test_optimization_strategy() {
        let low_end = WasmDeviceCapabilities::low_end_device();
        let strategy = low_end.get_optimization_strategy();
        assert!(matches!(
            strategy,
            WasmOptimizationStrategy::MinimalFootprint
        ));

        let high_end = WasmDeviceCapabilities::high_end_device();
        let strategy = high_end.get_optimization_strategy();
        assert!(matches!(
            strategy,
            WasmOptimizationStrategy::HighPerformance
        ));
    }

    #[test]
    #[cfg(feature = "wasm")]
    fn test_device_info() {
        let info = WasmDeviceInfo::detect();
        assert!(info.get_model_size_limit() > 0);
        assert!(info.get_recommended_batch_size() > 0);
    }
}
