//! Versioned serialization utilities
//!
//! This module provides semantic versioning support for model serialization,
//! ensuring proper version compatibility and migration handling.

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

/// Current serialization format version
pub const CURRENT_VERSION: SemanticVersion = SemanticVersion {
    major: 0,
    minor: 1,
    patch: 0,
};

/// Semantic version for compatibility checking
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemanticVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl SemanticVersion {
    /// Create a new semantic version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Check if this version is compatible with another version
    pub fn is_compatible_with(&self, other: &SemanticVersion) -> bool {
        // Same major version is compatible
        self.major == other.major
    }

    /// Check if this version needs migration from another version
    pub fn needs_migration_from(&self, other: &SemanticVersion) -> bool {
        self > other
    }

    /// Get the next major version
    pub fn next_major(&self) -> Self {
        Self::new(self.major + 1, 0, 0)
    }

    /// Get the next minor version
    pub fn next_minor(&self) -> Self {
        Self::new(self.major, self.minor + 1, 0)
    }

    /// Get the next patch version
    pub fn next_patch(&self) -> Self {
        Self::new(self.major, self.minor, self.patch + 1)
    }

    /// Check if this is a breaking change from another version
    pub fn is_breaking_change_from(&self, other: &SemanticVersion) -> bool {
        self.major != other.major
    }

    /// Check if this has new features compared to another version
    pub fn has_new_features_from(&self, other: &SemanticVersion) -> bool {
        self.major == other.major && self.minor > other.minor
    }

    /// Check if this is a bug fix from another version
    pub fn is_bug_fix_from(&self, other: &SemanticVersion) -> bool {
        self.major == other.major && self.minor == other.minor && self.patch > other.patch
    }

    /// Parse version from string
    pub fn parse(version_str: &str) -> Result<Self> {
        let parts: Vec<&str> = version_str.split('.').collect();
        if parts.len() != 3 {
            return Err(TensorError::serialization_error_simple(
                "Invalid version format. Expected major.minor.patch".to_string(),
            ));
        }

        let major = parts[0].parse::<u32>().map_err(|_| {
            TensorError::serialization_error_simple("Invalid major version".to_string())
        })?;
        let minor = parts[1].parse::<u32>().map_err(|_| {
            TensorError::serialization_error_simple("Invalid minor version".to_string())
        })?;
        let patch = parts[2].parse::<u32>().map_err(|_| {
            TensorError::serialization_error_simple("Invalid patch version".to_string())
        })?;

        Ok(Self::new(major, minor, patch))
    }
}

impl std::fmt::Display for SemanticVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl std::str::FromStr for SemanticVersion {
    type Err = TensorError;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

/// Version range for compatibility checking
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct VersionRange {
    /// Minimum supported version (inclusive)
    pub min_version: SemanticVersion,
    /// Maximum supported version (inclusive)
    pub max_version: SemanticVersion,
}

impl VersionRange {
    /// Create a new version range
    pub fn new(min_version: SemanticVersion, max_version: SemanticVersion) -> Self {
        Self {
            min_version,
            max_version,
        }
    }

    /// Check if a version is within this range
    pub fn contains(&self, version: &SemanticVersion) -> bool {
        version >= &self.min_version && version <= &self.max_version
    }

    /// Check if this range overlaps with another range
    pub fn overlaps_with(&self, other: &VersionRange) -> bool {
        self.min_version <= other.max_version && self.max_version >= other.min_version
    }

    /// Create a range for the same major version
    pub fn same_major(version: &SemanticVersion) -> Self {
        Self::new(
            SemanticVersion::new(version.major, 0, 0),
            SemanticVersion::new(version.major, u32::MAX, u32::MAX),
        )
    }

    /// Create a range for compatible versions
    pub fn compatible_with(version: &SemanticVersion) -> Self {
        Self::same_major(version)
    }
}

/// Version compatibility information
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct VersionCompatibility {
    /// Current version
    pub current_version: SemanticVersion,
    /// Supported version range
    pub supported_range: VersionRange,
    /// Deprecated versions
    pub deprecated_versions: Vec<SemanticVersion>,
    /// Migration paths available
    pub migration_paths: HashMap<SemanticVersion, SemanticVersion>,
}

impl VersionCompatibility {
    /// Create new version compatibility info
    pub fn new(current_version: SemanticVersion) -> Self {
        Self {
            supported_range: VersionRange::compatible_with(&current_version),
            current_version,
            deprecated_versions: Vec::new(),
            migration_paths: HashMap::new(),
        }
    }

    /// Check if a version is supported
    pub fn is_supported(&self, version: &SemanticVersion) -> bool {
        self.supported_range.contains(version) && !self.deprecated_versions.contains(version)
    }

    /// Check if a version is deprecated
    pub fn is_deprecated(&self, version: &SemanticVersion) -> bool {
        self.deprecated_versions.contains(version)
    }

    /// Add a deprecated version
    pub fn add_deprecated_version(&mut self, version: SemanticVersion) {
        if !self.deprecated_versions.contains(&version) {
            self.deprecated_versions.push(version);
        }
    }

    /// Add a migration path
    pub fn add_migration_path(&mut self, from: SemanticVersion, to: SemanticVersion) {
        self.migration_paths.insert(from, to);
    }

    /// Get migration target for a version
    pub fn get_migration_target(&self, version: &SemanticVersion) -> Option<&SemanticVersion> {
        self.migration_paths.get(version)
    }

    /// Check if migration is available for a version
    pub fn has_migration_path(&self, version: &SemanticVersion) -> bool {
        self.migration_paths.contains_key(version)
    }
}

/// Versioned data wrapper
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct VersionedData<T> {
    /// Version of the data
    pub version: SemanticVersion,
    /// The actual data
    pub data: T,
    /// Metadata about the version
    pub metadata: VersionMetadata,
}

impl<T> VersionedData<T> {
    /// Create new versioned data
    pub fn new(version: SemanticVersion, data: T) -> Self {
        Self {
            version,
            data,
            metadata: VersionMetadata::new(),
        }
    }

    /// Create versioned data with current version
    pub fn current(data: T) -> Self {
        Self::new(CURRENT_VERSION, data)
    }

    /// Check if this data is compatible with a version
    pub fn is_compatible_with(&self, version: &SemanticVersion) -> bool {
        self.version.is_compatible_with(version)
    }

    /// Check if this data needs migration
    pub fn needs_migration(&self) -> bool {
        CURRENT_VERSION.needs_migration_from(&self.version)
    }

    /// Get the data if version is compatible
    pub fn get_data_if_compatible(&self, version: &SemanticVersion) -> Option<&T> {
        if self.is_compatible_with(version) {
            Some(&self.data)
        } else {
            None
        }
    }

    /// Update version metadata
    pub fn update_metadata(&mut self, key: &str, value: &str) {
        self.metadata.add_metadata(key, value);
    }
}

/// Version metadata
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct VersionMetadata {
    /// Creation timestamp
    pub created_at: String,
    /// Creator information
    pub created_by: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl VersionMetadata {
    /// Create new version metadata
    #[cfg(feature = "serialize")]
    pub fn new() -> Self {
        Self {
            created_at: chrono::Utc::now().to_rfc3339(),
            created_by: "TenfloweRS".to_string(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

impl Default for VersionMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Version comparison result
#[derive(Debug, Clone, PartialEq)]
pub enum VersionComparison {
    /// Versions are identical
    Identical,
    /// Current version is newer (patch version)
    PatchUpdate,
    /// Current version is newer (minor version)
    MinorUpdate,
    /// Current version is newer (major version)
    MajorUpdate,
    /// Current version is older
    Downgrade,
    /// Versions are incompatible
    Incompatible,
}

/// Version utilities
pub mod utils {
    use super::*;

    /// Compare two versions
    pub fn compare_versions(
        current: &SemanticVersion,
        other: &SemanticVersion,
    ) -> VersionComparison {
        if current == other {
            return VersionComparison::Identical;
        }

        if current.major != other.major {
            if current.major > other.major {
                return VersionComparison::MajorUpdate;
            } else {
                return VersionComparison::Incompatible;
            }
        }

        if current.minor != other.minor {
            if current.minor > other.minor {
                return VersionComparison::MinorUpdate;
            } else {
                return VersionComparison::Downgrade;
            }
        }

        if current.patch > other.patch {
            VersionComparison::PatchUpdate
        } else {
            VersionComparison::Downgrade
        }
    }

    /// Get version recommendations
    pub fn get_version_recommendations(
        current: &SemanticVersion,
        target: &SemanticVersion,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        match compare_versions(current, target) {
            VersionComparison::Identical => {
                recommendations.push("Versions are identical. No action needed.".to_string());
            }
            VersionComparison::PatchUpdate => {
                recommendations
                    .push("Patch update available. Consider updating for bug fixes.".to_string());
            }
            VersionComparison::MinorUpdate => {
                recommendations
                    .push("Minor update available. New features may be available.".to_string());
            }
            VersionComparison::MajorUpdate => {
                recommendations.push(
                    "Major update available. Review breaking changes before updating.".to_string(),
                );
                recommendations.push("Consider creating a backup before updating.".to_string());
            }
            VersionComparison::Downgrade => {
                recommendations.push(
                    "Target version is older. Consider if downgrade is necessary.".to_string(),
                );
            }
            VersionComparison::Incompatible => {
                recommendations
                    .push("Versions are incompatible. Migration may be required.".to_string());
                recommendations.push("Check migration documentation for guidance.".to_string());
            }
        }

        recommendations
    }

    /// Validate version format
    pub fn validate_version_format(version_str: &str) -> Result<()> {
        SemanticVersion::parse(version_str)?;
        Ok(())
    }

    /// Get next version based on change type
    pub fn get_next_version(current: &SemanticVersion, change_type: ChangeType) -> SemanticVersion {
        match change_type {
            ChangeType::Major => current.next_major(),
            ChangeType::Minor => current.next_minor(),
            ChangeType::Patch => current.next_patch(),
        }
    }

    /// Determine change type between versions
    pub fn determine_change_type(
        from: &SemanticVersion,
        to: &SemanticVersion,
    ) -> Option<ChangeType> {
        if from.major != to.major {
            Some(ChangeType::Major)
        } else if from.minor != to.minor {
            Some(ChangeType::Minor)
        } else if from.patch != to.patch {
            Some(ChangeType::Patch)
        } else {
            None
        }
    }

    /// Create version range from string
    pub fn parse_version_range(range_str: &str) -> Result<VersionRange> {
        let parts: Vec<&str> = range_str.split("..").collect();
        if parts.len() != 2 {
            return Err(TensorError::serialization_error_simple(
                "Invalid version range format. Expected 'min..max'".to_string(),
            ));
        }

        let min_version = SemanticVersion::parse(parts[0])?;
        let max_version = SemanticVersion::parse(parts[1])?;

        Ok(VersionRange::new(min_version, max_version))
    }
}

/// Change type for version bumping
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChangeType {
    /// Major version change (breaking changes)
    Major,
    /// Minor version change (new features)
    Minor,
    /// Patch version change (bug fixes)
    Patch,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_version_creation() {
        let version = SemanticVersion::new(1, 2, 3);
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
    }

    #[test]
    fn test_version_display() {
        let version = SemanticVersion::new(1, 2, 3);
        assert_eq!(format!("{}", version), "1.2.3");
    }

    #[test]
    fn test_version_parsing() {
        let version = SemanticVersion::parse("1.2.3").unwrap();
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);

        assert!(SemanticVersion::parse("1.2").is_err());
        assert!(SemanticVersion::parse("1.2.3.4").is_err());
        assert!(SemanticVersion::parse("a.b.c").is_err());
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = SemanticVersion::new(1, 0, 0);
        let v2 = SemanticVersion::new(1, 1, 0);
        let v3 = SemanticVersion::new(2, 0, 0);

        assert!(v1.is_compatible_with(&v2));
        assert!(v2.is_compatible_with(&v1));
        assert!(!v1.is_compatible_with(&v3));
        assert!(!v3.is_compatible_with(&v1));
    }

    #[test]
    fn test_version_migration() {
        let v1 = SemanticVersion::new(1, 0, 0);
        let v2 = SemanticVersion::new(1, 1, 0);

        assert!(v2.needs_migration_from(&v1));
        assert!(!v1.needs_migration_from(&v2));
    }

    #[test]
    fn test_version_ordering() {
        let v1 = SemanticVersion::new(1, 0, 0);
        let v2 = SemanticVersion::new(1, 1, 0);
        let v3 = SemanticVersion::new(2, 0, 0);

        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v1 < v3);
    }

    #[test]
    fn test_version_range() {
        let range = VersionRange::new(SemanticVersion::new(1, 0, 0), SemanticVersion::new(1, 5, 0));

        assert!(range.contains(&SemanticVersion::new(1, 2, 0)));
        assert!(range.contains(&SemanticVersion::new(1, 0, 0)));
        assert!(range.contains(&SemanticVersion::new(1, 5, 0)));
        assert!(!range.contains(&SemanticVersion::new(0, 9, 0)));
        assert!(!range.contains(&SemanticVersion::new(1, 6, 0)));
        assert!(!range.contains(&SemanticVersion::new(2, 0, 0)));
    }

    #[test]
    fn test_version_comparison() {
        let v1 = SemanticVersion::new(1, 0, 0);
        let v2 = SemanticVersion::new(1, 0, 1);
        let v3 = SemanticVersion::new(1, 1, 0);
        let v4 = SemanticVersion::new(2, 0, 0);

        assert_eq!(
            utils::compare_versions(&v1, &v1),
            VersionComparison::Identical
        );
        assert_eq!(
            utils::compare_versions(&v2, &v1),
            VersionComparison::PatchUpdate
        );
        assert_eq!(
            utils::compare_versions(&v3, &v1),
            VersionComparison::MinorUpdate
        );
        assert_eq!(
            utils::compare_versions(&v4, &v1),
            VersionComparison::MajorUpdate
        );
        assert_eq!(
            utils::compare_versions(&v1, &v2),
            VersionComparison::Downgrade
        );
    }

    #[test]
    fn test_version_compatibility_info() {
        let mut compat = VersionCompatibility::new(SemanticVersion::new(1, 0, 0));

        assert!(compat.is_supported(&SemanticVersion::new(1, 0, 0)));
        assert!(compat.is_supported(&SemanticVersion::new(1, 1, 0)));
        assert!(!compat.is_supported(&SemanticVersion::new(2, 0, 0)));

        compat.add_deprecated_version(SemanticVersion::new(1, 0, 0));
        assert!(!compat.is_supported(&SemanticVersion::new(1, 0, 0)));
        assert!(compat.is_deprecated(&SemanticVersion::new(1, 0, 0)));
    }

    #[test]
    fn test_versioned_data() {
        let data = VersionedData::new(SemanticVersion::new(1, 0, 0), "test_data".to_string());

        assert_eq!(data.version, SemanticVersion::new(1, 0, 0));
        assert_eq!(data.data, "test_data");
        assert!(data.is_compatible_with(&SemanticVersion::new(1, 1, 0)));
        assert!(!data.is_compatible_with(&SemanticVersion::new(2, 0, 0)));
    }

    #[test]
    fn test_change_type_determination() {
        let v1 = SemanticVersion::new(1, 0, 0);
        let v2 = SemanticVersion::new(1, 0, 1);
        let v3 = SemanticVersion::new(1, 1, 0);
        let v4 = SemanticVersion::new(2, 0, 0);

        assert_eq!(
            utils::determine_change_type(&v1, &v2),
            Some(ChangeType::Patch)
        );
        assert_eq!(
            utils::determine_change_type(&v1, &v3),
            Some(ChangeType::Minor)
        );
        assert_eq!(
            utils::determine_change_type(&v1, &v4),
            Some(ChangeType::Major)
        );
        assert_eq!(utils::determine_change_type(&v1, &v1), None);
    }

    #[test]
    fn test_version_range_parsing() {
        let range = utils::parse_version_range("1.0.0..1.5.0").unwrap();
        assert_eq!(range.min_version, SemanticVersion::new(1, 0, 0));
        assert_eq!(range.max_version, SemanticVersion::new(1, 5, 0));

        assert!(utils::parse_version_range("1.0.0").is_err());
        assert!(utils::parse_version_range("1.0.0..1.5.0..2.0.0").is_err());
    }
}
