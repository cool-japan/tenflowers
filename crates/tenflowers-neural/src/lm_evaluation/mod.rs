//! Language Model Evaluation Framework
//!
//! Comprehensive metrics and evaluation harnesses for language models:
//!
//! - [`LmePerplexity`]       — Per-token NLL / sliding-window / byte-pair perplexity
//! - [`LmeBLEU`]             — Corpus/sentence BLEU (Papineni 2002, n=1..4)
//! - [`LmeROUGE`]            — ROUGE-N, ROUGE-L, ROUGE-W (Lin 2004)
//! - [`LmeBERTScore`]        — Token-level cosine similarity (Zhang 2020 proxy)
//! - [`LmeMathEval`]         — Mathematical reasoning: exact/approx match, CoT
//! - [`LmeFewShotEval`]      — N-shot MC evaluation (ARC/HellaSwag/MMLU-style)
//! - [`LmeHarmEval`]         — Toxicity / refusal-rate / format conformance
//! - [`LmeTruthfulnessEval`] — TruthfulQA-style MC1/MC2 accuracy
//! - [`LmeCodeEval`]         — pass@k (unbiased), syntax validity, execution match
//! - [`LmeReport`]           — Aggregate report with bootstrap CI and A/B comparison
//!
//! References: Papineni et al. 2002, Lin 2004, Zhang et al. 2020, Chen et al. 2021 (HumanEval).

pub mod core;
pub mod metrics;

pub use self::core::*;
pub use self::metrics::*;

#[cfg(test)]
mod tests;
