//! LLM Serving & Fine-tuning Infrastructure
//!
//! Production-grade components for deploying and fine-tuning large language models:
//!
//! - [`LsKvCacheManager`] — Block-based KV cache with LRU eviction
//! - [`LsPagedKvCache`] — Virtual-memory-style paged attention cache
//! - [`LsSpeculativeDecoder`] — Speculative decoding with adaptive K
//! - [`LsInstructionTuner`] — Instruction fine-tuning with loss masking
//! - [`LsRlhfRewardModel`] — Bradley-Terry reward model for RLHF
//! - [`LsDynamicBatcher`] — Continuous batching with priority scheduling
//! - [`LsModelWarmup`] — Model profiling and calibration
//! - [`LsTokenClassifier`] — BIO/BILOU sequence labeling with CRF
//! - [`LsServingMetrics`] — TTFT/TPS/latency tracking
//! - [`LsContinuousBatchProcessor`] — Iteration-level batch engine
//!
//! References: Leviathan et al. 2023 (speculative decoding), vLLM (PagedAttention),
//! Ouyang et al. 2022 (InstructGPT/RLHF), Yu et al. 2022 (continuous batching).

pub mod core;
pub mod serving;

// Re-export all public types for backwards compatibility.
pub use self::core::*;
pub use self::serving::*;

#[cfg(test)]
mod tests;
