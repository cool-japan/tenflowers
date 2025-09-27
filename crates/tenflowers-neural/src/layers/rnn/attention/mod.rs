//! Attention mechanisms for RNN layers

// TODO: Move attention implementations from rnn.rs (lines 4263-5012)
// This includes: BahdanauAttention, LuongAttention, HierarchicalAttention

pub mod bahdanau;
pub mod hierarchical;
pub mod luong;

// Re-export commonly used types
pub use bahdanau::BahdanauAttention;
pub use hierarchical::HierarchicalAttention;
pub use luong::{LuongAttention, LuongAttentionType};
