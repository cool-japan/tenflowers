//! Contrastive learning primitives for self-supervised and supervised
//! representation learning.
//!
//! # Modules
//!
//! * [`losses`] — NT-Xent, Supervised Contrastive, InfoNCE, cosine similarity,
//!   L2-normalisation.
//! * [`moco`]   — MoCo momentum encoder queue and loss.
//! * [`augmentation`] — Embedding-space augmentation helpers.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::contrastive::losses::{nt_xent_loss, l2_normalize_batch};
//! use tenflowers_neural::contrastive::moco::{MoCoQueue, moco_loss, momentum_update};
//! use tenflowers_neural::contrastive::augmentation::create_augmented_pair;
//! ```

pub mod augmentation;
pub mod losses;
pub mod moco;

// Convenient flat re-exports.
pub use augmentation::{add_gaussian_noise, create_augmented_pair, feature_dropout};
pub use losses::{
    cosine_similarity, info_nce_loss, l2_normalize_batch, nt_xent_loss, supervised_contrastive_loss,
};
pub use moco::{moco_loss, momentum_update, MoCoQueue};
