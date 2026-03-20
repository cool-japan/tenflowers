//! Segment-based reduction operations for ragged tensor support
//!
//! This module provides segmented reduction operations that can handle ragged tensors
//! by performing reductions within segments defined by segment IDs. These operations
//! are essential for handling variable-length sequences and ragged data structures.

pub mod minmax;
pub mod prod_any_all;
pub mod sum_mean;
pub mod types;
pub mod unsorted;

#[cfg(test)]
mod tests;

// Re-export all public functions for backward compatibility
pub use minmax::{segment_max, segment_min};
pub use prod_any_all::{segment_all, segment_any, segment_prod};
pub use sum_mean::{segment_mean, segment_sum};
pub use types::SegmentConfig;
pub use unsorted::{
    unsorted_segment_max, unsorted_segment_mean, unsorted_segment_min, unsorted_segment_prod,
    unsorted_segment_sum,
};
