//! API stabilization module — curated list of types and functions considered stable.
//!
//! This module defines the public API surface of the TenfloweRS FFI bindings,
//! allowing downstream consumers to query which symbols are stable, in beta,
//! experimental, or deprecated.  The information is fully static (no I/O, no
//! reflection) so it adds zero runtime cost.
//!
//! # Python Usage
//!
//! ```python
//! import tenflowers as tf
//!
//! ver = tf.stable_api_version()
//! print(f"API version: {ver.major}.{ver.minor}.{ver.patch} ({ver.stability})")
//!
//! surface = tf.stable_api_surface()
//! for entry in surface.entries:
//!     print(f"  {entry.name:40s} [{entry.stability}] since {entry.since_version}")
//! ```

use pyo3::prelude::*;

// ─── ApiStability ─────────────────────────────────────────────────────────────

/// Stability classification for a public API symbol.
#[pyclass(name = "ApiStability", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiStability {
    /// Covered by semantic-versioning guarantees; breaking changes require a
    /// major version bump.
    Stable,
    /// Functional and tested, but the interface may change in a minor release.
    Beta,
    /// Preview API subject to change at any time; not covered by semver.
    Experimental,
    /// Retained for backward compatibility; scheduled for removal in a future
    /// major release.
    Deprecated,
}

#[pymethods]
impl ApiStability {
    fn __repr__(&self) -> &'static str {
        match self {
            Self::Stable => "ApiStability.Stable",
            Self::Beta => "ApiStability.Beta",
            Self::Experimental => "ApiStability.Experimental",
            Self::Deprecated => "ApiStability.Deprecated",
        }
    }

    fn __str__(&self) -> &'static str {
        match self {
            Self::Stable => "Stable",
            Self::Beta => "Beta",
            Self::Experimental => "Experimental",
            Self::Deprecated => "Deprecated",
        }
    }
}

impl std::fmt::Display for ApiStability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.__str__())
    }
}

// ─── StableApiVersion ─────────────────────────────────────────────────────────

/// Semantic version and stability of the current FFI API.
#[pyclass(name = "StableApiVersion")]
#[derive(Debug, Clone)]
pub struct StableApiVersion {
    /// Major version component.
    #[pyo3(get)]
    pub major: u32,
    /// Minor version component.
    #[pyo3(get)]
    pub minor: u32,
    /// Patch version component.
    #[pyo3(get)]
    pub patch: u32,
    /// Stability classification of this API version.
    #[pyo3(get)]
    pub stability: ApiStability,
}

/// Parses a `major.minor.patch[-prerelease][+build]` string at compile time.
///
/// This is a `const fn` (no heap, no `unwrap`) so it can be evaluated while
/// building [`STABLE_API_VERSION`] directly from `env!("CARGO_PKG_VERSION")`,
/// which keeps the two permanently in sync — there is no literal version
/// number left to drift when the workspace version is bumped.
///
/// Any pre-release/build metadata suffix (e.g. `-alpha`, `+exp.sha`) is
/// ignored for the patch component's parsing, matching how downstream
/// tooling treats `CARGO_PKG_VERSION_{MAJOR,MINOR,PATCH}` env vars.
const fn const_parse_semver(version: &str) -> (u32, u32, u32) {
    let bytes = version.as_bytes();
    let mut i = 0;
    let mut major = 0u32;
    while i < bytes.len() && bytes[i] != b'.' {
        major = major * 10 + (bytes[i] - b'0') as u32;
        i += 1;
    }
    i += 1; // skip '.'

    let mut minor = 0u32;
    while i < bytes.len() && bytes[i] != b'.' {
        minor = minor * 10 + (bytes[i] - b'0') as u32;
        i += 1;
    }
    i += 1; // skip '.'

    let mut patch = 0u32;
    while i < bytes.len() && bytes[i] != b'-' && bytes[i] != b'+' {
        patch = patch * 10 + (bytes[i] - b'0') as u32;
        i += 1;
    }

    (major, minor, patch)
}

/// `(major, minor, patch)` parsed at compile time from `CARGO_PKG_VERSION`,
/// i.e. from `version.workspace = true` in `Cargo.toml`.
const PARSED_CRATE_VERSION: (u32, u32, u32) = const_parse_semver(env!("CARGO_PKG_VERSION"));

/// The canonical stable-API version for this build.
///
/// Derived at compile time from `env!("CARGO_PKG_VERSION")` (see
/// [`const_parse_semver`]), which itself comes from `version.workspace =
/// true` in `Cargo.toml`. This can never drift from the crate's real
/// version, unlike a hand-maintained literal.
///
/// The stability is `Beta` for v0.x releases; it will move to `Stable`
/// when the crate reaches v1.0.
pub const STABLE_API_VERSION: StableApiVersion = StableApiVersion {
    major: PARSED_CRATE_VERSION.0,
    minor: PARSED_CRATE_VERSION.1,
    patch: PARSED_CRATE_VERSION.2,
    stability: ApiStability::Beta,
};

#[pymethods]
impl StableApiVersion {
    /// Dot-separated version string, e.g. `"0.1.2"`.
    pub fn version_string(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }

    fn __repr__(&self) -> String {
        format!(
            "StableApiVersion({}.{}.{}, stability={})",
            self.major, self.minor, self.patch, self.stability
        )
    }
}

// ─── ApiEntry ─────────────────────────────────────────────────────────────────

/// A single entry in the public API surface catalogue.
#[pyclass(name = "ApiEntry")]
#[derive(Debug, Clone)]
pub struct ApiEntry {
    /// Python-visible symbol name (e.g. `"PyTensor"`, `"zeros"`).
    #[pyo3(get)]
    pub name: String,
    /// Version string in which this symbol was first introduced.
    #[pyo3(get)]
    pub since_version: String,
    /// Current stability classification.
    #[pyo3(get)]
    pub stability: ApiStability,
}

#[pymethods]
impl ApiEntry {
    fn __repr__(&self) -> String {
        format!(
            "ApiEntry(name='{}', since='{}', stability={})",
            self.name, self.since_version, self.stability
        )
    }
}

// ─── ApiSurface ───────────────────────────────────────────────────────────────

/// The complete public API surface of the TenfloweRS FFI bindings.
#[pyclass(name = "ApiSurface")]
#[derive(Debug, Clone)]
pub struct ApiSurface {
    /// All API entries in the surface catalogue.
    #[pyo3(get)]
    pub entries: Vec<ApiEntry>,
}

#[pymethods]
impl ApiSurface {
    /// Return only the entries with [`ApiStability::Stable`].
    pub fn stable_entries(&self) -> Vec<ApiEntry> {
        self.entries
            .iter()
            .filter(|e| e.stability == ApiStability::Stable)
            .cloned()
            .collect()
    }

    /// Return only the entries with [`ApiStability::Deprecated`].
    pub fn deprecated_entries(&self) -> Vec<ApiEntry> {
        self.entries
            .iter()
            .filter(|e| e.stability == ApiStability::Deprecated)
            .cloned()
            .collect()
    }

    /// Return entries introduced in or after the given semver version string.
    ///
    /// The comparison is lexicographic on the `since_version` field.  For
    /// well-formed semver strings this is correct within a single major version.
    pub fn entries_since(&self, version: &str) -> Vec<ApiEntry> {
        self.entries
            .iter()
            .filter(|e| e.since_version.as_str() >= version)
            .cloned()
            .collect()
    }

    /// Total number of entries in the surface catalogue.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    fn __repr__(&self) -> String {
        format!("ApiSurface(count={})", self.entries.len())
    }
}

// ─── Catalogue ────────────────────────────────────────────────────────────────

/// Helper for constructing [`ApiEntry`] values concisely.
fn entry(name: &str, since: &str, stability: ApiStability) -> ApiEntry {
    ApiEntry {
        name: name.to_string(),
        since_version: since.to_string(),
        stability,
    }
}

/// Returns the full [`ApiSurface`] for the current build.
///
/// This is the authoritative source of truth for which symbols are considered
/// stable, beta, experimental, or deprecated.
pub fn stable_api_surface() -> ApiSurface {
    ApiSurface {
        entries: vec![
            // ── Core tensor ──────────────────────────────────────────────────
            entry("PyTensor", "0.1.0", ApiStability::Beta),
            entry("PyTensorIter", "0.1.0", ApiStability::Beta),
            // ── Tensor creation ──────────────────────────────────────────────
            entry("zeros", "0.1.0", ApiStability::Stable),
            entry("ones", "0.1.0", ApiStability::Stable),
            entry("rand", "0.1.0", ApiStability::Stable),
            entry("randn", "0.1.0", ApiStability::Stable),
            entry("zeros_pinned", "0.1.1", ApiStability::Beta),
            entry("ones_pinned", "0.1.1", ApiStability::Beta),
            entry("rand_pinned", "0.1.1", ApiStability::Beta),
            // ── Arithmetic ───────────────────────────────────────────────────
            entry("add", "0.1.0", ApiStability::Stable),
            entry("sub", "0.1.0", ApiStability::Stable),
            entry("mul", "0.1.0", ApiStability::Stable),
            entry("div", "0.1.0", ApiStability::Stable),
            entry("matmul", "0.1.0", ApiStability::Stable),
            // ── Math ops ─────────────────────────────────────────────────────
            entry("exp", "0.1.0", ApiStability::Stable),
            entry("log", "0.1.0", ApiStability::Stable),
            entry("sqrt", "0.1.0", ApiStability::Stable),
            entry("abs", "0.1.0", ApiStability::Stable),
            entry("sum", "0.1.0", ApiStability::Stable),
            entry("mean", "0.1.0", ApiStability::Stable),
            entry("max", "0.1.0", ApiStability::Stable),
            entry("min", "0.1.0", ApiStability::Stable),
            entry("var", "0.1.0", ApiStability::Stable),
            entry("std", "0.1.0", ApiStability::Stable),
            entry("clamp", "0.1.0", ApiStability::Stable),
            // ── Reduction / shape ────────────────────────────────────────────
            entry("reshape", "0.1.0", ApiStability::Stable),
            entry("transpose", "0.1.0", ApiStability::Stable),
            entry("cat", "0.1.0", ApiStability::Stable),
            entry("stack", "0.1.0", ApiStability::Stable),
            entry("squeeze", "0.1.0", ApiStability::Stable),
            entry("unsqueeze", "0.1.0", ApiStability::Stable),
            entry("flatten", "0.1.0", ApiStability::Stable),
            // ── Device management ────────────────────────────────────────────
            entry("PyDevice", "0.1.0", ApiStability::Beta),
            entry("PyDeviceKind", "0.1.0", ApiStability::Beta),
            entry("get_default_device", "0.1.0", ApiStability::Stable),
            entry("set_default_device", "0.1.0", ApiStability::Stable),
            // ── DType system ─────────────────────────────────────────────────
            entry("PyDType", "0.1.0", ApiStability::Beta),
            entry("is_safe_cast", "0.1.0", ApiStability::Beta),
            entry("result_type", "0.1.0", ApiStability::Beta),
            entry("promote_types", "0.1.0", ApiStability::Beta),
            // ── Neural network ───────────────────────────────────────────────
            entry("relu", "0.1.0", ApiStability::Stable),
            entry("sigmoid", "0.1.0", ApiStability::Stable),
            entry("tanh", "0.1.0", ApiStability::Stable),
            entry("gelu", "0.1.0", ApiStability::Beta),
            entry("softmax", "0.1.0", ApiStability::Stable),
            // ── Serialization ────────────────────────────────────────────────
            entry("PyCheckpointManager", "0.1.0", ApiStability::Beta),
            entry("save_tensor", "0.1.0", ApiStability::Beta),
            entry("load_tensor", "0.1.0", ApiStability::Beta),
            entry("save_state_dict", "0.1.0", ApiStability::Beta),
            entry("load_state_dict", "0.1.0", ApiStability::Beta),
            // ── Metrics ──────────────────────────────────────────────────────
            entry("accuracy", "0.1.0", ApiStability::Stable),
            entry("mean_squared_error", "0.1.0", ApiStability::Stable),
            entry("mean_absolute_error", "0.1.0", ApiStability::Stable),
            entry("r2_score", "0.1.0", ApiStability::Stable),
            entry("auc_roc", "0.1.0", ApiStability::Beta),
            entry("confusion_matrix", "0.1.0", ApiStability::Beta),
            // ── Memory profiling ─────────────────────────────────────────────
            entry("enable_memory_profiling", "0.1.0", ApiStability::Beta),
            entry("disable_memory_profiling", "0.1.0", ApiStability::Beta),
            entry("get_memory_info", "0.1.0", ApiStability::Beta),
            // ── Profiling (new in 0.1.2) ─────────────────────────────────────
            entry("PyProfiler", "0.1.2", ApiStability::Beta),
            entry("PyProfileReport", "0.1.2", ApiStability::Beta),
            entry("PyProfileRecord", "0.1.2", ApiStability::Beta),
            // ── Stable API metadata (new in 0.1.2) ───────────────────────────
            entry("stable_api_version", "0.1.2", ApiStability::Beta),
            entry("stable_api_surface", "0.1.2", ApiStability::Beta),
            // ── Gradient management ──────────────────────────────────────────
            entry("is_grad_enabled", "0.1.0", ApiStability::Stable),
            entry("set_grad_enabled", "0.1.0", ApiStability::Stable),
            // ── NumPy interop ────────────────────────────────────────────────
            entry("tensor_from_numpy", "0.1.0", ApiStability::Beta),
            entry("tensor_to_numpy", "0.1.0", ApiStability::Beta),
        ],
    }
}

// ─── Python-facing free functions ────────────────────────────────────────────

/// Return the current FFI API version.
#[pyfunction]
#[pyo3(name = "stable_api_version")]
pub fn stable_api_version_py() -> StableApiVersion {
    STABLE_API_VERSION.clone()
}

/// Return the full API surface catalogue.
#[pyfunction]
#[pyo3(name = "stable_api_surface")]
pub fn stable_api_surface_py() -> ApiSurface {
    stable_api_surface()
}

// ─── Registration helper ──────────────────────────────────────────────────────

/// Register stable-API types and functions into the given Python module.
pub fn register_stable_api(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ApiStability>()?;
    m.add_class::<StableApiVersion>()?;
    m.add_class::<ApiEntry>()?;
    m.add_class::<ApiSurface>()?;
    m.add_function(wrap_pyfunction!(stable_api_version_py, py)?)?;
    m.add_function(wrap_pyfunction!(stable_api_surface_py, py)?)?;
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_api_version_constant() {
        // Derive expectations from CARGO_PKG_VERSION rather than hardcoding
        // literals, so this test can't silently drift from the real crate
        // version on the next release bump.
        let (major, minor, patch) = PARSED_CRATE_VERSION;
        assert_eq!(STABLE_API_VERSION.major, major);
        assert_eq!(STABLE_API_VERSION.minor, minor);
        assert_eq!(STABLE_API_VERSION.patch, patch);
        assert_eq!(STABLE_API_VERSION.stability, ApiStability::Beta);
    }

    #[test]
    fn test_version_string() {
        let v = STABLE_API_VERSION.clone();
        assert_eq!(v.version_string(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_const_parse_semver_basic() {
        assert_eq!(const_parse_semver("0.1.2"), (0, 1, 2));
        assert_eq!(const_parse_semver("1.2.3"), (1, 2, 3));
        assert_eq!(const_parse_semver("10.20.30"), (10, 20, 30));
    }

    #[test]
    fn test_const_parse_semver_with_suffix() {
        assert_eq!(const_parse_semver("0.1.2-alpha"), (0, 1, 2));
        assert_eq!(const_parse_semver("0.1.2+build.5"), (0, 1, 2));
    }

    #[test]
    fn test_stable_api_surface_non_empty() {
        let surface = stable_api_surface();
        assert!(!surface.entries.is_empty());
    }

    #[test]
    fn test_surface_contains_pytensor() {
        let surface = stable_api_surface();
        assert!(surface.entries.iter().any(|e| e.name == "PyTensor"));
    }

    #[test]
    fn test_stable_entries_subset() {
        let surface = stable_api_surface();
        let stable = surface.stable_entries();
        assert!(!stable.is_empty());
        for e in &stable {
            assert_eq!(e.stability, ApiStability::Stable);
        }
    }

    #[test]
    fn test_deprecated_entries_all_deprecated() {
        let surface = stable_api_surface();
        for e in surface.deprecated_entries() {
            assert_eq!(e.stability, ApiStability::Deprecated);
        }
    }

    #[test]
    fn test_entries_since_filters() {
        let surface = stable_api_surface();
        let new = surface.entries_since("0.1.2");
        // All 0.1.2+ entries should appear.
        assert!(new.iter().all(|e| e.since_version.as_str() >= "0.1.2"));
    }

    #[test]
    fn test_count_matches_entries_len() {
        let surface = stable_api_surface();
        assert_eq!(surface.count(), surface.entries.len());
    }

    #[test]
    fn test_api_stability_display() {
        assert_eq!(ApiStability::Stable.to_string(), "Stable");
        assert_eq!(ApiStability::Deprecated.to_string(), "Deprecated");
    }
}
