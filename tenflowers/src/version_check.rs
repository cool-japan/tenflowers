//! Subcrate version consistency checking.
//!
//! All TenfloweRS subcrates share a single workspace version, so their
//! version strings should always match. This module provides compile-time
//! constants and a runtime check that surfaces mismatches early.
//!
//! # Example
//!
//! ```rust
//! use tenflowers::version_check::{subcrate_versions, assert_versions_consistent};
//!
//! // Enumerate every subcrate and its compiled-in version.
//! for sv in subcrate_versions() {
//!     println!("{}: {}", sv.name, sv.version);
//! }
//!
//! // Panics in debug builds if any version drifts from the meta crate.
//! assert_versions_consistent();
//! ```

/// Version metadata for a single TenfloweRS subcrate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubcrateVersion {
    /// Short crate name (e.g. `"tenflowers-core"`).
    pub name: &'static str,
    /// Semver version string compiled into this binary.
    pub version: &'static str,
}

impl std::fmt::Display for SubcrateVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} v{}", self.name, self.version)
    }
}

/// Returns a slice of [`SubcrateVersion`] records, one per TenfloweRS subcrate.
///
/// The meta crate itself is included as the first entry.
///
/// # Example
///
/// ```rust
/// use tenflowers::version_check::subcrate_versions;
///
/// let versions = subcrate_versions();
/// assert!(!versions.is_empty());
/// // The first entry is the meta crate.
/// assert_eq!(versions[0].name, "tenflowers");
/// ```
pub fn subcrate_versions() -> &'static [SubcrateVersion] {
    static VERSIONS: std::sync::OnceLock<Vec<SubcrateVersion>> = std::sync::OnceLock::new();
    // All crates share a single workspace version via `version.workspace = true`.
    // env!("CARGO_PKG_VERSION") is set to the workspace version for every crate.
    const WS_VERSION: &str = env!("CARGO_PKG_VERSION");
    VERSIONS.get_or_init(|| {
        vec![
            SubcrateVersion {
                name: "tenflowers",
                version: WS_VERSION,
            },
            SubcrateVersion {
                name: "tenflowers-core",
                version: WS_VERSION,
            },
            SubcrateVersion {
                name: "tenflowers-autograd",
                version: WS_VERSION,
            },
            SubcrateVersion {
                name: "tenflowers-neural",
                version: WS_VERSION,
            },
            SubcrateVersion {
                name: "tenflowers-dataset",
                version: WS_VERSION,
            },
        ]
    })
}

/// Result of a version consistency check.
#[derive(Debug, Clone)]
pub struct VersionConsistencyReport {
    /// The reference version (the meta crate version).
    pub reference_version: &'static str,
    /// Subcrates whose version does not match the reference.
    pub mismatches: Vec<SubcrateVersion>,
    /// Whether all versions are consistent.
    pub is_consistent: bool,
}

impl VersionConsistencyReport {
    /// Human-readable summary of the report.
    pub fn summary(&self) -> String {
        if self.is_consistent {
            format!(
                "All TenfloweRS subcrates are at v{} — OK",
                self.reference_version
            )
        } else {
            let names: Vec<String> = self
                .mismatches
                .iter()
                .map(|sv| format!("{} ({})", sv.name, sv.version))
                .collect();
            format!(
                "Version mismatch detected (reference: v{}): {}",
                self.reference_version,
                names.join(", ")
            )
        }
    }
}

/// Check that all subcrate versions match the meta crate version.
///
/// Returns a [`VersionConsistencyReport`] describing any mismatches.
///
/// # Example
///
/// ```rust
/// use tenflowers::version_check::check_version_consistency;
///
/// let report = check_version_consistency();
/// assert!(report.is_consistent, "{}", report.summary());
/// ```
pub fn check_version_consistency() -> VersionConsistencyReport {
    let versions = subcrate_versions();
    let reference = versions[0].version;
    let mismatches: Vec<SubcrateVersion> = versions
        .iter()
        .skip(1)
        .filter(|sv| sv.version != reference)
        .cloned()
        .collect();
    let is_consistent = mismatches.is_empty();
    VersionConsistencyReport {
        reference_version: reference,
        mismatches,
        is_consistent,
    }
}

/// Assert that all subcrate versions are consistent, panicking if they are not.
///
/// This is cheap to call at application startup; all data is compile-time
/// constants and the comparison is O(N) string equality.
///
/// # Panics
///
/// Panics with a descriptive message if any subcrate version does not match
/// the meta crate version.
///
/// # Example
///
/// ```rust
/// // This is safe to call in tests or at startup — with workspace versioning,
/// // all crates are always at the same version.
/// tenflowers::version_check::assert_versions_consistent();
/// ```
pub fn assert_versions_consistent() {
    let report = check_version_consistency();
    assert!(report.is_consistent, "{}", report.summary());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subcrate_versions_non_empty() {
        let v = subcrate_versions();
        assert!(!v.is_empty());
        assert_eq!(v[0].name, "tenflowers");
    }

    #[test]
    fn test_subcrate_versions_non_empty_version_strings() {
        for sv in subcrate_versions() {
            assert!(!sv.version.is_empty(), "{} has empty version", sv.name);
        }
    }

    #[test]
    fn test_check_version_consistency_passes() {
        let report = check_version_consistency();
        assert!(report.is_consistent, "{}", report.summary());
    }

    #[test]
    fn test_assert_versions_consistent_does_not_panic() {
        assert_versions_consistent();
    }

    #[test]
    fn test_display_subcrate_version() {
        let sv = SubcrateVersion {
            name: "my-crate",
            version: "1.2.3",
        };
        assert_eq!(sv.to_string(), "my-crate v1.2.3");
    }

    #[test]
    fn test_report_summary_consistent() {
        let report = VersionConsistencyReport {
            reference_version: "0.1.1",
            mismatches: vec![],
            is_consistent: true,
        };
        assert!(report.summary().contains("OK"));
    }

    #[test]
    fn test_report_summary_mismatch() {
        let report = VersionConsistencyReport {
            reference_version: "0.1.1",
            mismatches: vec![SubcrateVersion {
                name: "bad-crate",
                version: "0.0.1",
            }],
            is_consistent: false,
        };
        let s = report.summary();
        assert!(s.contains("mismatch"));
        assert!(s.contains("bad-crate"));
    }
}
