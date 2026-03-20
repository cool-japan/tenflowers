//! Meta-learning algorithms for few-shot and fast adaptation.
//!
//! Implements three families of meta-learning approaches:
//!
//! - **MAML** (Model-Agnostic Meta-Learning, Finn et al. 2017) and its
//!   first-order variant FOMAML, plus **Reptile** (Nichol et al. 2018).
//! - **Prototypical Networks** (Snell et al. 2017) — embedding-based metric
//!   learning with Euclidean, cosine, and dot-product distance.
//! - **Few-shot utilities**: episodic sampling and Matching Networks helpers
//!   (Vinyals et al. 2016).
//!
//! # Design notes
//!
//! All components use plain `Vec<f32>` for parameters and embeddings, keeping
//! the module free from any autograd dependency and easy to compose with external
//! neural-network implementations.  Random number generation is done exclusively
//! via `scirs2_core::random` (no `rand` crate).

pub mod few_shot;
pub mod maml;
pub mod prototypical;
pub mod reptile;
pub mod types;

mod tests;

// Re-export public API
pub use few_shot::{attention_kernel, matching_networks_predict, EpisodeSampler};
pub use maml::{maml_inner_update, maml_meta_gradient, MamlConfig, MamlGradient};
pub use prototypical::{
    cosine_similarity, dot_product, euclidean_distance, DistanceMetric, PrototypicalNetwork,
};
pub use reptile::{reptile_meta_update, ReptileConfig};
pub use types::{Episode, MetaLearningError};
