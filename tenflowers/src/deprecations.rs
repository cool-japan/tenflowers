//! Deprecation helpers for the TenfloweRS meta-crate.
//!
//! This module provides the `deprecated_use` macro, which wraps a `pub use`
//! with a `#[deprecated]` attribute and a structured notice.
//!
//! # Policy
//! - Items are soft-deprecated in minor versions (`#[deprecated(since, note)]`).
//! - Items are removed in the next major version.
//! - Experimental items have no deprecation notice requirement.
//! - For 0.x releases: stability guarantees are best-effort; breaking changes
//!   may occur in minor bumps with documented notice in CHANGELOG.

/// Emit a deprecated re-export with structured notice.
///
/// # Example
/// ```rust
/// # use tenflowers::deprecated_use;
/// // deprecated_use!(
/// //     #[deprecated(since = "0.2.0", note = "Use `NewType` instead")]
/// //     pub use OldType;
/// // );
/// ```
#[macro_export]
macro_rules! deprecated_use {
    (
        #[deprecated(since = $since:literal, note = $note:literal)]
        pub use $path:path as $alias:ident;
    ) => {
        #[deprecated(since = $since, note = $note)]
        pub use $path as $alias;
    };
    (
        #[deprecated(since = $since:literal, note = $note:literal)]
        pub use $path:path;
    ) => {
        #[deprecated(since = $since, note = $note)]
        pub use $path;
    };
}
