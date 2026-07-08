//! Unified error types for the TenfloweRS framework.
//!
//! This module re-exports the primary error type from `tenflowers-core`
//! and provides a top-level [`crate::error::FrameworkError`] that can represent errors
//! from any TenfloweRS subcrate. Use [`crate::error::Result`] as a convenient alias.
//!
//! # Quick Example
//!
//! ```rust
//! use tenflowers::error::{FrameworkError, Result};
//!
//! fn try_something() -> Result<i32> {
//!     Ok(42)
//! }
//! assert_eq!(try_something().unwrap(), 42);
//! ```

use thiserror::Error;

/// Primary error type for the TenfloweRS framework.
///
/// Wraps errors from every subcrate and provides a single `?`-compatible
/// error type for applications built on `tenflowers`.
#[derive(Debug, Error)]
pub enum FrameworkError {
    /// Error originating in the tensor / core compute engine.
    #[error("Core tensor error: {0}")]
    Core(#[from] tenflowers_core::TensorError),

    /// Error originating in the dataset / data-loading layer.
    #[error("Dataset error: {0}")]
    Dataset(String),

    /// A general-purpose wrapper for errors from external crates or
    /// code that does not map to a specific variant.
    #[error("Framework error: {0}")]
    Other(String),
}

impl FrameworkError {
    /// Wrap an arbitrary error message as a [`FrameworkError::Other`].
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }
}

impl From<Box<dyn std::error::Error>> for FrameworkError {
    fn from(e: Box<dyn std::error::Error>) -> Self {
        Self::Other(e.to_string())
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for FrameworkError {
    fn from(e: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Self::Other(e.to_string())
    }
}

/// Convenience `Result` alias that uses [`FrameworkError`] as the error type.
///
/// # Example
///
/// ```rust
/// use tenflowers::error::Result;
///
/// fn add_one(x: i32) -> Result<i32> {
///     Ok(x + 1)
/// }
///
/// assert_eq!(add_one(5).unwrap(), 6);
/// ```
pub type Result<T> = std::result::Result<T, FrameworkError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_other_construction() {
        let e = FrameworkError::other("something went wrong");
        assert!(e.to_string().contains("something went wrong"));
    }

    #[test]
    fn test_result_ok() {
        let r: Result<u32> = Ok(7);
        assert!(matches!(r, Ok(7)));
    }

    #[test]
    fn test_result_err_other() {
        let r: Result<u32> = Err(FrameworkError::other("fail"));
        assert!(r.is_err());
    }

    #[test]
    fn test_from_boxed_error() {
        let boxed: Box<dyn std::error::Error> = "boxed".into();
        let e = FrameworkError::from(boxed);
        assert!(e.to_string().contains("boxed"));
    }
}
