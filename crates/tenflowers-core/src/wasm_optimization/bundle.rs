//! WASM bundle size optimization and configuration
//!
//! # Honest limitation
//!
//! [`WasmBundleOptimizer`] stores optimization *preferences*
//! (`WasmOptimizationConfig`, `CodeSplittingConfig`, `TreeShakingConfig`,
//! `CompressionConfig`) that a caller can use to describe how they *want* a
//! WASM build to be optimized. It does not, and currently cannot, measure any
//! real savings: nothing in this type, nor anywhere else in
//! `wasm_optimization::*`, ever receives or parses an actual compiled `.wasm`
//! binary. An earlier version of this file fabricated "bytes saved" reports
//! from hardcoded per-category constants (e.g. "8 branch optimizations, 25
//! bytes saved each") that never varied with any real input -- the same
//! numbers were returned regardless of what code, if any, was being compiled.
//! That has been removed rather than relabeled, because renaming fields does
//! not turn constant, input-independent numbers into genuine estimates.
//! [`WasmBundleOptimizer::optimize_for_edge`] and
//! [`WasmBundleOptimizer::optimize_for_minimal_size`] now return an honest
//! `Err` explaining this instead of a fabricated report. See the doc comments
//! on those functions, and on [`WasmBundleOptimizer::detect_simd_support`],
//! for what real support would require.

#[cfg(feature = "wasm")]
use crate::{Result, TensorError};

#[cfg(feature = "wasm")]
use super::compression::CompressionConfig;
#[cfg(feature = "wasm")]
use super::performance::WasmOptimizationReport;

/// WASM bundle size optimizer
#[cfg(feature = "wasm")]
pub struct WasmBundleOptimizer {
    /// Enabled optimizations
    pub optimizations: WasmOptimizationConfig,
    /// Code splitting configuration
    pub code_splitting: CodeSplittingConfig,
    /// Tree shaking rules
    pub tree_shaking: TreeShakingConfig,
    /// Compression settings
    pub compression: CompressionConfig,
}

/// WASM optimization configuration
#[cfg(feature = "wasm")]
#[derive(Debug, Clone)]
pub struct WasmOptimizationConfig {
    /// Enable dead code elimination
    pub dead_code_elimination: bool,
    /// Enable function inlining
    pub function_inlining: bool,
    /// Enable constant folding
    pub constant_folding: bool,
    /// Enable loop unrolling
    pub loop_unrolling: bool,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Enable LTO (Link Time Optimization)
    pub lto: bool,
}

/// Code splitting configuration for lazy loading
#[cfg(feature = "wasm")]
#[derive(Debug, Clone)]
pub struct CodeSplittingConfig {
    /// Split neural network layers
    pub split_layers: bool,
    /// Split operations by category
    pub split_operations: bool,
    /// Minimum chunk size (bytes)
    pub min_chunk_size: usize,
    /// Maximum chunk size (bytes)
    pub max_chunk_size: usize,
    /// Enable dynamic imports
    pub dynamic_imports: bool,
}

/// Tree shaking configuration
#[cfg(feature = "wasm")]
#[derive(Debug, Clone)]
pub struct TreeShakingConfig {
    /// Remove unused tensor operations
    pub remove_unused_ops: bool,
    /// Remove unused data types
    pub remove_unused_dtypes: bool,
    /// Remove unused device backends
    pub remove_unused_backends: bool,
    /// Aggressive tree shaking
    pub aggressive: bool,
}

#[cfg(feature = "wasm")]
impl WasmBundleOptimizer {
    /// Create a new bundle optimizer with default settings
    pub fn new() -> Self {
        Self {
            optimizations: WasmOptimizationConfig {
                dead_code_elimination: true,
                function_inlining: true,
                constant_folding: true,
                loop_unrolling: false,
                optimization_level: 3,
                lto: true,
            },
            code_splitting: CodeSplittingConfig {
                split_layers: true,
                split_operations: true,
                min_chunk_size: 50 * 1024,  // 50KB
                max_chunk_size: 500 * 1024, // 500KB
                dynamic_imports: true,
            },
            tree_shaking: TreeShakingConfig {
                remove_unused_ops: true,
                remove_unused_dtypes: true,
                remove_unused_backends: true,
                aggressive: true,
            },
            compression: CompressionConfig::default(),
        }
    }

    /// Optimize bundle size for edge deployment.
    ///
    /// # Errors
    ///
    /// Always returns `Err`. Real bundle-size optimization requires parsing
    /// an actual compiled WASM binary (module sections, code bodies,
    /// import/export tables, etc.) and measuring what dead code elimination,
    /// tree shaking, and compression would really save for that specific
    /// binary. `WasmBundleOptimizer` never receives a compiled `.wasm` binary
    /// anywhere in its API, or anywhere else in `wasm_optimization::*` -- it
    /// only stores user-chosen optimization *preferences*
    /// (`WasmOptimizationConfig`, `TreeShakingConfig`, `CompressionConfig`).
    /// A previous implementation returned `Ok` with a `WasmOptimizationReport`
    /// whose "bytes saved" fields were hardcoded constants that never varied
    /// with any real input; that was fabricated data presented as a
    /// measurement, so it has been replaced with an honest error rather than
    /// a relabeled guess.
    ///
    /// To implement this for real: accept the compiled WASM bytes (`&[u8]`)
    /// as a parameter, parse the module's section headers (magic + version,
    /// then repeated `section_id` + LEB128 `size` + contents -- this much is
    /// mechanical and tractable), and measure actual section sizes before and
    /// after applying real transformations, instead of inventing
    /// per-optimization byte counts.
    pub fn optimize_for_edge(&self) -> Result<WasmOptimizationReport> {
        Err(TensorError::not_implemented_simple(
            "WasmBundleOptimizer::optimize_for_edge: real WASM bundle-size \
             analysis is not implemented. This optimizer never receives a \
             compiled .wasm binary to analyze -- it only holds optimization \
             preferences -- so any 'bytes saved' figure it returned would be \
             fabricated rather than measured. Provide real compiled WASM \
             bytes to a binary-level size analyzer (e.g. one that parses \
             section headers and diffs sizes before/after a real \
             transformation) to get genuine numbers."
                .to_string(),
        ))
    }

    /// Apply comprehensive size optimizations for minimal WASM builds.
    ///
    /// # Errors
    ///
    /// Always returns `Err`, for the same reason as
    /// [`Self::optimize_for_edge`]: this optimizer has no access to a real
    /// compiled WASM binary, so it cannot honestly report symbol-stripping,
    /// inlining, constant-folding, instruction-level, or memory-layout
    /// savings. A previous implementation invented these numbers from
    /// hardcoded per-category constants (e.g. "8 branch optimizations, 25
    /// bytes saved each", "20% debug-symbol overhead") that never depended on
    /// any actual code being compiled; that fabricated data has been removed
    /// rather than merely relabeled, since renaming fields would not make
    /// constant, input-independent numbers into real estimates. See
    /// [`Self::optimize_for_edge`] for what real support would require.
    pub fn optimize_for_minimal_size(&mut self) -> Result<WasmOptimizationReport> {
        Err(TensorError::not_implemented_simple(
            "WasmBundleOptimizer::optimize_for_minimal_size: real WASM \
             bundle-size analysis is not implemented. This optimizer never \
             receives a compiled .wasm binary to analyze, so \
             symbol-stripping, inlining, constant-folding, and \
             instruction/memory-layout savings cannot be honestly computed \
             here. Provide real compiled WASM bytes to a binary-level size \
             analyzer to get genuine numbers."
                .to_string(),
        ))
    }

    /// Create ultra-minimal configuration for edge devices with severe constraints
    pub fn create_ultra_minimal_config() -> (
        WasmOptimizationConfig,
        CodeSplittingConfig,
        TreeShakingConfig,
    ) {
        let optimization = WasmOptimizationConfig {
            dead_code_elimination: true,
            function_inlining: true,
            constant_folding: true,
            loop_unrolling: false, // Disabled for size
            optimization_level: 3,
            lto: true,
        };

        let code_splitting = CodeSplittingConfig {
            split_layers: true,
            split_operations: true,
            min_chunk_size: 10 * 1024,  // 10KB - smaller chunks
            max_chunk_size: 100 * 1024, // 100KB - smaller max
            dynamic_imports: true,
        };

        let tree_shaking = TreeShakingConfig {
            remove_unused_ops: true,
            remove_unused_dtypes: true,
            remove_unused_backends: true,
            aggressive: true,
        };

        (optimization, code_splitting, tree_shaking)
    }

    /// Attempt to detect whether the target WASM runtime supports the SIMD
    /// proposal.
    ///
    /// # Errors
    ///
    /// Always returns `Err`. Whether a runtime supports WASM SIMD is a
    /// genuine capability question about the host executing a compiled
    /// module, not something `WasmBundleOptimizer` can derive from its own
    /// state -- it holds only optimization preferences, not a handle to a
    /// runtime or a compiled binary. A real (if narrow) capability probe
    /// already exists for `target_arch = "wasm32"` in
    /// `wasm_optimization::device::WasmDeviceCapabilities::detect` (via
    /// `js_sys::eval` against `WebAssembly.validate`); wiring
    /// `WasmBundleOptimizer` up to that probe would mean giving it a
    /// `WasmDeviceCapabilities` (or similar) input, which is an API change
    /// out of scope for this function. The previous implementation returned
    /// a hardcoded `true` in every case, by its own admission as a
    /// placeholder ("In a real implementation, this would check runtime
    /// capabilities") -- that misrepresented an unchecked guess as a real
    /// capability check, so this reports the limitation honestly instead.
    pub fn detect_simd_support(&self) -> Result<bool> {
        Err(TensorError::not_implemented_simple(
            "WasmBundleOptimizer::detect_simd_support: no real SIMD \
             capability probe is wired into this optimizer. See \
             wasm_optimization::device::WasmDeviceCapabilities::detect for a \
             genuine (wasm32-only) runtime feature check; integrating it \
             here would require passing device capabilities into \
             WasmBundleOptimizer, which is out of scope for this fix."
                .to_string(),
        ))
    }
}

#[cfg(feature = "wasm")]
impl Default for WasmBundleOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "wasm")]
    fn test_bundle_optimizer() {
        let optimizer = WasmBundleOptimizer::new();
        assert!(optimizer.optimizations.dead_code_elimination);
        assert!(optimizer.optimizations.lto);
    }

    #[test]
    #[cfg(feature = "wasm")]
    fn test_ultra_minimal_config() {
        let (opt, code_split, tree_shake) = WasmBundleOptimizer::create_ultra_minimal_config();
        assert!(opt.dead_code_elimination);
        assert!(!opt.loop_unrolling);
        assert_eq!(code_split.min_chunk_size, 10 * 1024);
        assert!(tree_shake.aggressive);
    }

    /// `optimize_for_edge` must never fabricate a "bytes saved" report: no
    /// real compiled WASM binary is ever available to this optimizer, so any
    /// `Ok` report would be inventing numbers rather than measuring them.
    #[test]
    #[cfg(feature = "wasm")]
    fn test_optimize_for_edge_returns_honest_error() {
        let optimizer = WasmBundleOptimizer::new();
        let result = optimizer.optimize_for_edge();
        assert!(result.is_err());
        let message = result.unwrap_err().to_string();
        assert!(message.contains("not implemented"));
        assert!(message.contains("optimize_for_edge"));
    }

    /// Same guarantee as above for the "minimal size" entry point: it must
    /// not return a fabricated report either.
    #[test]
    #[cfg(feature = "wasm")]
    fn test_optimize_for_minimal_size_returns_honest_error() {
        let mut optimizer = WasmBundleOptimizer::new();
        let result = optimizer.optimize_for_minimal_size();
        assert!(result.is_err());
        let message = result.unwrap_err().to_string();
        assert!(message.contains("not implemented"));
        assert!(message.contains("optimize_for_minimal_size"));
    }

    /// `detect_simd_support` must not silently return a hardcoded `true`:
    /// this optimizer has no runtime capability probe or compiled binary to
    /// check.
    #[test]
    #[cfg(feature = "wasm")]
    fn test_detect_simd_support_returns_honest_error() {
        let optimizer = WasmBundleOptimizer::new();
        let result = optimizer.detect_simd_support();
        assert!(result.is_err());
        let message = result.unwrap_err().to_string();
        assert!(message.contains("not implemented"));
        assert!(message.contains("detect_simd_support"));
    }

    /// `create_ultra_minimal_config` is a curated preset (a fixed,
    /// documented set of preference values the caller can opt into), not a
    /// measurement -- unlike the deleted "bytes saved" functions, it never
    /// claimed to measure anything, so it is intentionally unaffected by the
    /// honest-error changes above.
    #[test]
    #[cfg(feature = "wasm")]
    fn test_default_matches_new() {
        let default_optimizer = WasmBundleOptimizer::default();
        assert!(default_optimizer.optimizations.dead_code_elimination);
        assert!(default_optimizer.tree_shaking.aggressive);
        assert!(default_optimizer.compression.brotli);
    }
}
