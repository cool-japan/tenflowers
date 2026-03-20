//! Types, error types, and configuration for segment reduction operations

/// Segment reduction error (no special errors needed, uses TensorError from crate)
/// This module exists to hold future error types and configuration structs.

/// Configuration for segmented operations
#[derive(Debug, Clone)]
pub struct SegmentConfig {
    /// Number of output segments
    pub num_segments: usize,
    /// Whether to use parallel processing for large inputs
    pub use_parallel: bool,
    /// Threshold for switching to parallel processing
    pub parallel_threshold: usize,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            num_segments: 1,
            use_parallel: true,
            parallel_threshold: 1000,
        }
    }
}
