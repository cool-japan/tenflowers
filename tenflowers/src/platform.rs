//! Platform detection and SIMD capability introspection.
//!
//! This module provides compile-time (via `cfg` attributes) and logical
//! detection of the current target architecture and SIMD instruction-set
//! extensions.  No runtime CPU-ID queries are performed; the results are
//! determined entirely from the compiler's view of the target.
//!
//! # Example
//!
//! ```rust
//! use tenflowers::platform::{current_platform, detect_simd_capabilities, Platform};
//!
//! let platform = current_platform();
//! let simd     = detect_simd_capabilities();
//!
//! println!("Platform : {platform}");
//! println!("AVX2     : {}", simd.has_avx2);
//! println!("SSE4.1   : {}", simd.has_sse4);
//! println!("NEON     : {}", simd.has_neon);
//! ```

// ─── Platform ─────────────────────────────────────────────────────────────────

/// Target architecture of the current build.
///
/// Detected at compile time via `#[cfg(target_arch = …)]` — the variants are
/// mutually exclusive and the correct one is chosen by the compiler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Platform {
    /// 64-bit x86 (e.g., Intel/AMD desktop/server CPUs).
    X86_64,
    /// 64-bit ARM (e.g., Apple Silicon, AWS Graviton, Raspberry Pi 4).
    Aarch64,
    /// WebAssembly (browser and WASI targets).
    Wasm32,
    /// Any other target not listed above.
    Other(String),
}

impl Platform {
    /// Returns `true` for any x86 / x86_64 target.
    pub fn is_x86_family(&self) -> bool {
        matches!(self, Self::X86_64)
    }

    /// Returns `true` for any ARM / AArch64 target.
    pub fn is_arm_family(&self) -> bool {
        matches!(self, Self::Aarch64)
    }

    /// Returns `true` for WebAssembly targets.
    pub fn is_wasm(&self) -> bool {
        matches!(self, Self::Wasm32)
    }
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::X86_64 => f.write_str("x86_64"),
            Self::Aarch64 => f.write_str("aarch64"),
            Self::Wasm32 => f.write_str("wasm32"),
            Self::Other(s) => write!(f, "other({s})"),
        }
    }
}

/// Returns the [`Platform`] corresponding to the current compilation target.
///
/// This function has no runtime cost — the branch is resolved at compile time
/// and the unused variants are optimised away.
pub fn current_platform() -> Platform {
    #[cfg(target_arch = "x86_64")]
    {
        return Platform::X86_64;
    }

    #[cfg(target_arch = "aarch64")]
    {
        return Platform::Aarch64;
    }

    #[cfg(target_arch = "wasm32")]
    {
        return Platform::Wasm32;
    }

    #[allow(unreachable_code)]
    Platform::Other(std::env::consts::ARCH.to_string())
}

// ─── SimdCapabilities ─────────────────────────────────────────────────────────

/// SIMD instruction-set capabilities of the compilation target.
///
/// All fields are set at compile time from Cargo's `target_feature` cfg flags.
/// Applications that need runtime detection (e.g., for JIT dispatch) should
/// use the `std::arch::is_x86_feature_detected!` macro instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SimdCapabilities {
    /// AVX2 (256-bit integer / float SIMD, Intel Haswell+ / AMD Ryzen+).
    pub has_avx2: bool,
    /// SSE 4.1 (128-bit SIMD with dot-product and round instructions).
    pub has_sse4: bool,
    /// NEON (ARM Advanced SIMD, available on all AArch64 targets).
    pub has_neon: bool,
    /// WebAssembly SIMD (128-bit vectors in WASM targets).
    pub has_wasm_simd: bool,
}

impl SimdCapabilities {
    /// Returns `true` if any SIMD extension is available.
    pub fn has_any(&self) -> bool {
        self.has_avx2 || self.has_sse4 || self.has_neon || self.has_wasm_simd
    }

    /// Preferred SIMD "tier" name for logging / diagnostics.
    pub fn best_available(&self) -> &'static str {
        if self.has_avx2 {
            "avx2"
        } else if self.has_sse4 {
            "sse4.1"
        } else if self.has_neon {
            "neon"
        } else if self.has_wasm_simd {
            "wasm-simd"
        } else {
            "none"
        }
    }
}

impl std::fmt::Display for SimdCapabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SimdCapabilities {{ avx2={}, sse4={}, neon={}, wasm_simd={} }}",
            self.has_avx2, self.has_sse4, self.has_neon, self.has_wasm_simd
        )
    }
}

/// Detect SIMD capabilities from compiler `target_feature` flags.
///
/// The result is a pure compile-time constant — calling this function
/// at runtime has no overhead beyond loading a few boolean literals.
pub fn detect_simd_capabilities() -> SimdCapabilities {
    SimdCapabilities {
        has_avx2: cfg!(target_feature = "avx2"),
        has_sse4: cfg!(target_feature = "sse4.1"),
        has_neon: cfg!(target_feature = "neon"),
        has_wasm_simd: cfg!(target_feature = "simd128"),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_platform_returns_valid_variant() {
        let p = current_platform();
        // Whatever variant we get, it must be representable as a non-empty string.
        assert!(!p.to_string().is_empty());
    }

    #[test]
    fn current_platform_is_one_of_known_variants() {
        // On any host the platform must be one of the known variants or Other.
        let p = current_platform();
        match &p {
            Platform::X86_64 | Platform::Aarch64 | Platform::Wasm32 => {}
            Platform::Other(s) => {
                assert!(!s.is_empty(), "Other(…) must contain a non-empty string");
            }
        }
    }

    #[test]
    fn platform_family_helpers_are_consistent() {
        let x = Platform::X86_64;
        assert!(x.is_x86_family());
        assert!(!x.is_arm_family());
        assert!(!x.is_wasm());

        let a = Platform::Aarch64;
        assert!(!a.is_x86_family());
        assert!(a.is_arm_family());

        let w = Platform::Wasm32;
        assert!(w.is_wasm());
    }

    #[test]
    fn platform_display_is_non_empty() {
        for p in [
            Platform::X86_64,
            Platform::Aarch64,
            Platform::Wasm32,
            Platform::Other("riscv64".to_string()),
        ] {
            assert!(!p.to_string().is_empty());
        }
    }

    #[test]
    fn simd_capabilities_has_any_consistent_with_fields() {
        let caps = SimdCapabilities {
            has_avx2: false,
            has_sse4: false,
            has_neon: false,
            has_wasm_simd: false,
        };
        assert!(!caps.has_any());

        let caps_avx2 = SimdCapabilities {
            has_avx2: true,
            ..caps
        };
        assert!(caps_avx2.has_any());
    }

    #[test]
    fn simd_best_available_respects_priority() {
        let full = SimdCapabilities {
            has_avx2: true,
            has_sse4: true,
            has_neon: true,
            has_wasm_simd: true,
        };
        assert_eq!(full.best_available(), "avx2");

        let sse_only = SimdCapabilities {
            has_sse4: true,
            ..SimdCapabilities::default()
        };
        assert_eq!(sse_only.best_available(), "sse4.1");

        let neon_only = SimdCapabilities {
            has_neon: true,
            ..SimdCapabilities::default()
        };
        assert_eq!(neon_only.best_available(), "neon");

        let none = SimdCapabilities::default();
        assert_eq!(none.best_available(), "none");
    }

    #[test]
    fn detect_simd_capabilities_returns_simd_capabilities() {
        // Just ensure the function is callable and returns a valid struct.
        let caps = detect_simd_capabilities();
        // best_available() must always return a non-empty str.
        assert!(!caps.best_available().is_empty());
    }

    #[test]
    fn simd_capabilities_display_contains_fields() {
        let caps = SimdCapabilities {
            has_avx2: true,
            ..SimdCapabilities::default()
        };
        let s = caps.to_string();
        assert!(s.contains("avx2=true"));
        assert!(s.contains("sse4=false"));
    }
}
