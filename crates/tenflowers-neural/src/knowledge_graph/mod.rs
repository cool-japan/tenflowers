//! Knowledge Graph Embeddings (KGE) — comprehensive implementation.
//!
//! Implements TransE, RotatE, and ComplEx embedding models for knowledge graph
//! link prediction.  All models implement the [`KgeModel`] trait and are
//! trained via the [`KgeTrainer`] with negative sampling.
//!
//! # Mathematical background
//!
//! **TransE** (Bordes et al., 2013): models relations as translations in ℝ^d.
//! Score: `-(‖h + r − t‖_p)`  — higher is better.
//!
//! **RotatE** (Sun et al., 2019): models relations as rotations in ℂ^(d/2).
//! Score: `-(‖h ∘ r − t‖)` where ∘ is element-wise complex multiplication.
//!
//! **ComplEx** (Trouillon et al., 2016): uses complex embeddings with a
//! bilinear interaction `Re(<h, r, t̄>)`.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::knowledge_graph::{
//!     KgDataset, KgeTrainer, KgeModelType, TransEConfig,
//! };
//! use scirs2_core::random::{rngs::StdRng, SeedableRng};
//!
//! let entities = vec!["A".into(), "B".into(), "C".into()];
//! let relations = vec!["r".into()];
//! let triples = vec![(0, 0, 1), (1, 0, 2)];
//! let dataset = KgDataset::from_triples(entities, relations, triples);
//!
//! let trainer = KgeTrainer {
//!     model_type: KgeModelType::TransE(TransEConfig { dim: 16, margin: 1.0, norm: 2 }),
//!     lr: 0.01,
//!     n_epochs: 50,
//!     batch_size: 8,
//!     n_negatives: 2,
//!     seed: 42,
//! };
//! let result = trainer.train(&dataset)?;
//! println!("Final loss: {}", result.final_loss);
//! # Ok::<(), tenflowers_core::TensorError>(())
//! ```

pub mod extensions;
pub use extensions::*;

pub mod advanced;
pub use advanced::*;

#[cfg(test)]
mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::f64::consts::PI;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Xavier (Glorot) uniform initialisation for a 1-D weight vector.
///
/// Samples uniformly from `[-limit, limit]` where
/// `limit = sqrt(6 / (fan_in + fan_out))`.
pub fn xavier_init_1d(n: usize, fan_in: usize, fan_out: usize, rng: &mut StdRng) -> Vec<f64> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    (0..n)
        .map(|_| {
            let u: f64 = rng.random::<f64>();
            u * 2.0 * limit - limit
        })
        .collect()
}

/// L2 norm of a slice.
#[inline]
pub fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Normalise a mutable slice to unit L2 norm in-place.
/// If the norm is effectively zero the vector is left unchanged.
pub fn normalize_vec(v: &mut [f64]) {
    let norm = l2_norm(v);
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Element-wise complex multiplication: `(a_r + i·a_i) * (b_r + i·b_i)`.
/// Returns `(real, imag)`.
#[inline]
pub fn complex_multiply(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> (f64, f64) {
    (a_r * b_r - a_i * b_i, a_r * b_i + a_i * b_r)
}

/// Numerically stable softplus: `ln(1 + exp(x))`.
///
/// Uses 3-branch approximation:
/// - `x > 30`  → `x`  (difference < 1e-13)
/// - `x < -30` → `exp(x)` (positive but tiny)
/// - otherwise  → `ln(1 + exp(x))`
#[inline]
pub fn softplus(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        x.exp()
    } else {
        (1.0_f64 + x.exp()).ln()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core data structures
// ─────────────────────────────────────────────────────────────────────────────

/// A single knowledge-graph triple `(head, relation, tail)` stored as entity /
/// relation indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KgTriple {
    pub head: usize,
    pub relation: usize,
    pub tail: usize,
}

impl KgTriple {
    /// Convenience constructor.
    pub fn new(head: usize, relation: usize, tail: usize) -> Self {
        Self {
            head,
            relation,
            tail,
        }
    }
}

/// In-memory knowledge graph dataset.
///
/// Stores entity / relation string labels and the set of triples as
/// `KgTriple` values.  Provides negative sampling and basic statistics.
#[derive(Debug, Clone)]
pub struct KgDataset {
    /// Entity string labels (indices are positions in this Vec).
    pub entities: Vec<String>,
    /// Relation string labels (indices are positions in this Vec).
    pub relations: Vec<String>,
    /// All positive triples.
    pub triples: Vec<KgTriple>,
    /// Set of positive triples for O(1) filtered-setting lookup.
    positive_set: std::collections::HashSet<(usize, usize, usize)>,
}

impl KgDataset {
    /// Construct a dataset from string labels and raw index triples.
    ///
    /// `raw_triples` is a `Vec<(head, relation, tail)>` tuple slice.
    pub fn from_triples(
        entities: Vec<String>,
        relations: Vec<String>,
        raw_triples: Vec<(usize, usize, usize)>,
    ) -> Self {
        let triples: Vec<KgTriple> = raw_triples
            .iter()
            .map(|&(h, r, t)| KgTriple::new(h, r, t))
            .collect();
        let positive_set = triples
            .iter()
            .map(|t| (t.head, t.relation, t.tail))
            .collect();
        Self {
            entities,
            relations,
            triples,
            positive_set,
        }
    }

    /// Number of unique entities.
    pub fn n_entities(&self) -> usize {
        self.entities.len()
    }

    /// Number of unique relations.
    pub fn n_relations(&self) -> usize {
        self.relations.len()
    }

    /// Total number of positive triples.
    pub fn n_triples(&self) -> usize {
        self.triples.len()
    }

    /// Returns `true` if the given `(h, r, t)` key is a known positive triple.
    pub fn is_positive(&self, h: usize, r: usize, t: usize) -> bool {
        self.positive_set.contains(&(h, r, t))
    }

    /// Generate one negative sample by uniformly corrupting either the head or
    /// the tail entity of `triple`.
    ///
    /// The sampled entity is chosen uniformly from `[0, n_entities)` and re-
    /// drawn if it accidentally reproduces the original positive triple (up to
    /// `max_attempts` retries; after that the last candidate is returned).
    pub fn negative_sample(
        &self,
        triple: &KgTriple,
        rng: &mut StdRng,
        corrupt_head: bool,
    ) -> KgTriple {
        let n = self.n_entities();
        let max_attempts = 64usize;
        let mut candidate = *triple;

        for _ in 0..max_attempts {
            // Safety: n > 0 guaranteed by construction (caller must ensure).
            let idx: usize = (rng.random::<f64>() * n as f64) as usize % n.max(1);
            if corrupt_head {
                candidate.head = idx;
            } else {
                candidate.tail = idx;
            }
            if !self.is_positive(candidate.head, candidate.relation, candidate.tail) {
                break;
            }
        }
        candidate
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KgeModel trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait implemented by all KGE models.
pub trait KgeModel {
    /// Score a triple.  Higher means more plausible.
    fn score(&self, head: usize, relation: usize, tail: usize) -> f64;

    /// Raw entity embedding (full stored vector).
    fn entity_embedding(&self, entity: usize) -> &[f64];

    /// Raw relation embedding (full stored vector).
    fn relation_embedding(&self, relation: usize) -> &[f64];

    /// Logical embedding dimensionality (may differ from storage length).
    fn embedding_dim(&self) -> usize;
}

// ─────────────────────────────────────────────────────────────────────────────
// TransE
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the TransE model.
#[derive(Debug, Clone)]
pub struct TransEConfig {
    /// Embedding dimension.
    pub dim: usize,
    /// Margin γ for the ranking loss.
    pub margin: f64,
    /// Which Lp norm to use (1 or 2).
    pub norm: u8,
}

impl Default for TransEConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            margin: 1.0,
            norm: 2,
        }
    }
}

/// TransE knowledge graph embedding model.
///
/// Each entity `e` is represented as a vector `h_e ∈ ℝ^d` normalised to the
/// unit sphere.  Each relation `r` is a vector `r_r ∈ ℝ^d`.  The score of
/// a triple `(h, r, t)` is `-(‖h + r − t‖_p)`.
#[derive(Debug, Clone)]
pub struct TransEModel {
    /// Entity embeddings: `entity_emb[id]` has length `dim`.
    pub entity_emb: Vec<Vec<f64>>,
    /// Relation embeddings: `relation_emb[id]` has length `dim`.
    pub relation_emb: Vec<Vec<f64>>,
    /// Configuration.
    pub config: TransEConfig,
}

impl TransEModel {
    /// Construct a randomly-initialised TransE model.
    pub fn new(
        n_entities: usize,
        n_relations: usize,
        config: TransEConfig,
        rng: &mut StdRng,
    ) -> Result<Self> {
        if config.dim == 0 {
            return Err(TensorError::invalid_argument(
                "TransE dim must be > 0".into(),
            ));
        }
        if n_entities == 0 {
            return Err(TensorError::invalid_argument(
                "n_entities must be > 0".into(),
            ));
        }

        let mut entity_emb: Vec<Vec<f64>> = (0..n_entities)
            .map(|_| {
                let mut v = xavier_init_1d(config.dim, config.dim, config.dim, rng);
                normalize_vec(&mut v);
                v
            })
            .collect();

        // Ensure entity embeddings are normalised.
        for e in entity_emb.iter_mut() {
            normalize_vec(e);
        }

        let relation_emb: Vec<Vec<f64>> = (0..n_relations)
            .map(|_| xavier_init_1d(config.dim, config.dim, config.dim, rng))
            .collect();

        Ok(Self {
            entity_emb,
            relation_emb,
            config,
        })
    }

    /// Compute `‖h + r − t‖_p`.
    fn distance(&self, h: usize, r: usize, t: usize) -> f64 {
        let he = &self.entity_emb[h];
        let re = &self.relation_emb[r];
        let te = &self.entity_emb[t];
        let diff: Vec<f64> = (0..self.config.dim)
            .map(|i| he[i] + re[i] - te[i])
            .collect();
        if self.config.norm == 1 {
            diff.iter().map(|x| x.abs()).sum::<f64>()
        } else {
            l2_norm(&diff)
        }
    }
}

impl KgeModel for TransEModel {
    fn score(&self, head: usize, relation: usize, tail: usize) -> f64 {
        -self.distance(head, relation, tail)
    }

    fn entity_embedding(&self, entity: usize) -> &[f64] {
        &self.entity_emb[entity]
    }

    fn relation_embedding(&self, relation: usize) -> &[f64] {
        &self.relation_emb[relation]
    }

    fn embedding_dim(&self) -> usize {
        self.config.dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RotatE
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the RotatE model.
#[derive(Debug, Clone)]
pub struct RotatEConfig {
    /// Embedding dimension (must be even; uses dim/2 complex numbers).
    pub dim: usize,
    /// Margin γ for the self-adversarial negative sampling loss.
    pub margin: f64,
}

impl Default for RotatEConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            margin: 2.0,
        }
    }
}

/// RotatE knowledge graph embedding model.
///
/// Entities are modelled as complex vectors in ℂ^(dim/2).
/// Relations are unit-modulus rotations parameterised by phases in `[-π, π]`.
///
/// Storage layout:
/// - `entity_real[id]` — real parts, length `dim/2`
/// - `entity_imag[id]` — imaginary parts, length `dim/2`
/// - `relation_phase[id]` — phase angles, length `dim/2`
///
/// The full "flat" entity embedding returned by [`KgeModel::entity_embedding`]
/// is `[real..., imag...]` (interleaved would also work, but flat is simpler).
#[derive(Debug, Clone)]
pub struct RotatEModel {
    /// Real parts of entity embeddings.
    pub entity_real: Vec<Vec<f64>>,
    /// Imaginary parts of entity embeddings.
    pub entity_imag: Vec<Vec<f64>>,
    /// Relation phase angles (θ), constrained to [-π, π].
    pub relation_phase: Vec<Vec<f64>>,
    /// Flat entity embedding cache (real ++ imag).
    flat_entity_emb: Vec<Vec<f64>>,
    /// Flat relation embedding cache (just phases).
    flat_relation_emb: Vec<Vec<f64>>,
    /// Configuration.
    pub config: RotatEConfig,
    /// Complex dimension = dim/2.
    pub(crate) complex_dim: usize,
}

impl RotatEModel {
    /// Construct a randomly-initialised RotatE model.
    pub fn new(
        n_entities: usize,
        n_relations: usize,
        config: RotatEConfig,
        rng: &mut StdRng,
    ) -> Result<Self> {
        if config.dim < 2 || config.dim % 2 != 0 {
            return Err(TensorError::invalid_argument(
                "RotatE dim must be even and >= 2".into(),
            ));
        }
        let complex_dim = config.dim / 2;

        let entity_real: Vec<Vec<f64>> = (0..n_entities)
            .map(|_| {
                (0..complex_dim)
                    .map(|_| rng.random::<f64>() * 2.0 - 1.0)
                    .collect()
            })
            .collect();
        let entity_imag: Vec<Vec<f64>> = (0..n_entities)
            .map(|_| {
                (0..complex_dim)
                    .map(|_| rng.random::<f64>() * 2.0 - 1.0)
                    .collect()
            })
            .collect();
        let relation_phase: Vec<Vec<f64>> = (0..n_relations)
            .map(|_| {
                (0..complex_dim)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * PI)
                    .collect()
            })
            .collect();

        let flat_entity_emb: Vec<Vec<f64>> = entity_real
            .iter()
            .zip(entity_imag.iter())
            .map(|(r, im)| r.iter().chain(im.iter()).cloned().collect())
            .collect();
        let flat_relation_emb: Vec<Vec<f64>> = relation_phase.clone();

        Ok(Self {
            entity_real,
            entity_imag,
            relation_phase,
            flat_entity_emb,
            flat_relation_emb,
            config,
            complex_dim,
        })
    }

    /// Recompute the flat embedding cache after parameter updates.
    pub(crate) fn rebuild_flat_cache(&mut self) {
        for i in 0..self.entity_real.len() {
            let flat: Vec<f64> = self.entity_real[i]
                .iter()
                .chain(self.entity_imag[i].iter())
                .cloned()
                .collect();
            self.flat_entity_emb[i] = flat;
        }
        for i in 0..self.relation_phase.len() {
            self.flat_relation_emb[i] = self.relation_phase[i].clone();
        }
    }

    /// Compute `‖h ∘ r − t‖` where ∘ is element-wise complex multiplication
    /// and r = (cos θ, sin θ) for each dimension.
    pub fn rotate_distance(&self, h: usize, r: usize, t: usize) -> f64 {
        let hr = &self.entity_real[h];
        let hi = &self.entity_imag[h];
        let rp = &self.relation_phase[r];
        let tr = &self.entity_real[t];
        let ti = &self.entity_imag[t];

        let mut sum_sq = 0.0_f64;
        for k in 0..self.complex_dim {
            let (rot_r, rot_i) = complex_multiply(hr[k], hi[k], rp[k].cos(), rp[k].sin());
            let diff_r = rot_r - tr[k];
            let diff_i = rot_i - ti[k];
            sum_sq += diff_r * diff_r + diff_i * diff_i;
        }
        sum_sq.sqrt()
    }
}

impl KgeModel for RotatEModel {
    fn score(&self, head: usize, relation: usize, tail: usize) -> f64 {
        -self.rotate_distance(head, relation, tail)
    }

    fn entity_embedding(&self, entity: usize) -> &[f64] {
        &self.flat_entity_emb[entity]
    }

    fn relation_embedding(&self, relation: usize) -> &[f64] {
        &self.flat_relation_emb[relation]
    }

    fn embedding_dim(&self) -> usize {
        self.config.dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ComplEx
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the ComplEx model.
#[derive(Debug, Clone)]
pub struct ComplExConfig {
    /// Embedding dimension (number of complex coordinates = dim; total floats = 2*dim).
    pub dim: usize,
    /// L2 regularisation coefficient λ.
    pub lambda: f64,
}

impl Default for ComplExConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            lambda: 1e-3,
        }
    }
}

/// ComplEx knowledge graph embedding model.
///
/// Each entity/relation is a complex vector in ℂ^d.  The score is:
/// `Re(<h, r, t̄>) = Σ_k (h_r·r_r·t_r + h_r·r_i·t_i + h_i·r_r·t_i − h_i·r_i·t_r)`
///
/// Storage: real parts first, then imaginary parts in flat vectors.
#[derive(Debug, Clone)]
pub struct ComplExModel {
    /// Real parts of entity embeddings, shape `[n_entities][dim]`.
    pub entity_real: Vec<Vec<f64>>,
    /// Imaginary parts of entity embeddings, shape `[n_entities][dim]`.
    pub entity_imag: Vec<Vec<f64>>,
    /// Real parts of relation embeddings, shape `[n_relations][dim]`.
    pub relation_real: Vec<Vec<f64>>,
    /// Imaginary parts of relation embeddings, shape `[n_relations][dim]`.
    pub relation_imag: Vec<Vec<f64>>,
    /// Flat entity embedding cache: `[real..., imag...]`.
    flat_entity_emb: Vec<Vec<f64>>,
    /// Flat relation embedding cache: `[real..., imag...]`.
    flat_relation_emb: Vec<Vec<f64>>,
    /// Configuration.
    pub config: ComplExConfig,
}

impl ComplExModel {
    /// Construct a randomly-initialised ComplEx model.
    pub fn new(
        n_entities: usize,
        n_relations: usize,
        config: ComplExConfig,
        rng: &mut StdRng,
    ) -> Result<Self> {
        if config.dim == 0 {
            return Err(TensorError::invalid_argument(
                "ComplEx dim must be > 0".into(),
            ));
        }

        let init = |rng: &mut StdRng| -> Vec<f64> {
            xavier_init_1d(config.dim, config.dim, config.dim, rng)
        };

        let entity_real: Vec<Vec<f64>> = (0..n_entities).map(|_| init(rng)).collect();
        let entity_imag: Vec<Vec<f64>> = (0..n_entities).map(|_| init(rng)).collect();
        let relation_real: Vec<Vec<f64>> = (0..n_relations).map(|_| init(rng)).collect();
        let relation_imag: Vec<Vec<f64>> = (0..n_relations).map(|_| init(rng)).collect();

        let flat_entity_emb: Vec<Vec<f64>> = entity_real
            .iter()
            .zip(entity_imag.iter())
            .map(|(r, im)| r.iter().chain(im.iter()).cloned().collect())
            .collect();
        let flat_relation_emb: Vec<Vec<f64>> = relation_real
            .iter()
            .zip(relation_imag.iter())
            .map(|(r, im)| r.iter().chain(im.iter()).cloned().collect())
            .collect();

        Ok(Self {
            entity_real,
            entity_imag,
            relation_real,
            relation_imag,
            flat_entity_emb,
            flat_relation_emb,
            config,
        })
    }

    /// `Re(<h, r, t̄>)` — ComplEx bilinear score.
    pub fn complex_score(&self, h: usize, r: usize, t: usize) -> f64 {
        let hr = &self.entity_real[h];
        let hi = &self.entity_imag[h];
        let rr = &self.relation_real[r];
        let ri = &self.relation_imag[r];
        let tr = &self.entity_real[t];
        let ti = &self.entity_imag[t];

        (0..self.config.dim)
            .map(|k| {
                hr[k] * rr[k] * tr[k] + hr[k] * ri[k] * ti[k] + hi[k] * rr[k] * ti[k]
                    - hi[k] * ri[k] * tr[k]
            })
            .sum::<f64>()
    }

    /// Recompute flat cache after gradient step.
    pub(crate) fn rebuild_flat_cache(&mut self) {
        for i in 0..self.entity_real.len() {
            let flat: Vec<f64> = self.entity_real[i]
                .iter()
                .chain(self.entity_imag[i].iter())
                .cloned()
                .collect();
            self.flat_entity_emb[i] = flat;
        }
        for i in 0..self.relation_real.len() {
            let flat: Vec<f64> = self.relation_real[i]
                .iter()
                .chain(self.relation_imag[i].iter())
                .cloned()
                .collect();
            self.flat_relation_emb[i] = flat;
        }
    }
}

impl KgeModel for ComplExModel {
    fn score(&self, head: usize, relation: usize, tail: usize) -> f64 {
        self.complex_score(head, relation, tail)
    }

    fn entity_embedding(&self, entity: usize) -> &[f64] {
        &self.flat_entity_emb[entity]
    }

    fn relation_embedding(&self, relation: usize) -> &[f64] {
        &self.flat_relation_emb[relation]
    }

    fn embedding_dim(&self) -> usize {
        self.config.dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model type enum
// ─────────────────────────────────────────────────────────────────────────────

/// Selects the KGE model variant used by [`KgeTrainer`].
#[derive(Debug, Clone)]
pub enum KgeModelType {
    TransE(TransEConfig),
    RotatE(RotatEConfig),
    ComplEx(ComplExConfig),
}

// ─────────────────────────────────────────────────────────────────────────────
// Trained model enum (returned after training)
// ─────────────────────────────────────────────────────────────────────────────

/// Boxed trained model returned by [`KgeTrainer::train`].
pub enum TrainedKgeModel {
    TransE(TransEModel),
    RotatE(RotatEModel),
    ComplEx(ComplExModel),
}

impl KgeModel for TrainedKgeModel {
    fn score(&self, h: usize, r: usize, t: usize) -> f64 {
        match self {
            TrainedKgeModel::TransE(m) => m.score(h, r, t),
            TrainedKgeModel::RotatE(m) => m.score(h, r, t),
            TrainedKgeModel::ComplEx(m) => m.score(h, r, t),
        }
    }

    fn entity_embedding(&self, entity: usize) -> &[f64] {
        match self {
            TrainedKgeModel::TransE(m) => m.entity_embedding(entity),
            TrainedKgeModel::RotatE(m) => m.entity_embedding(entity),
            TrainedKgeModel::ComplEx(m) => m.entity_embedding(entity),
        }
    }

    fn relation_embedding(&self, relation: usize) -> &[f64] {
        match self {
            TrainedKgeModel::TransE(m) => m.relation_embedding(relation),
            TrainedKgeModel::RotatE(m) => m.relation_embedding(relation),
            TrainedKgeModel::ComplEx(m) => m.relation_embedding(relation),
        }
    }

    fn embedding_dim(&self) -> usize {
        match self {
            TrainedKgeModel::TransE(m) => m.embedding_dim(),
            TrainedKgeModel::RotatE(m) => m.embedding_dim(),
            TrainedKgeModel::ComplEx(m) => m.embedding_dim(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Training result
// ─────────────────────────────────────────────────────────────────────────────

/// Results returned by [`KgeTrainer::train`].
pub struct KgeTrainingResult {
    /// Per-epoch average loss values.
    pub loss_history: Vec<f64>,
    /// Loss recorded during the final epoch.
    pub final_loss: f64,
    /// The trained model.
    pub model: TrainedKgeModel,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal gradient helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Clip a scalar to `[-clip, clip]`.
#[inline]
pub(crate) fn clip(x: f64, clip: f64) -> f64 {
    x.clamp(-clip, clip)
}

/// Analytical gradient of the TransE margin loss w.r.t. entity/relation vectors.
pub(crate) fn transe_grad_update(
    model: &mut TransEModel,
    pos: &KgTriple,
    neg: &KgTriple,
    lr: f64,
) {
    let clip_val = 1.0_f64;
    let dim = model.config.dim;

    let diff_pos: Vec<f64> = (0..dim)
        .map(|i| {
            model.entity_emb[pos.head][i] + model.relation_emb[pos.relation][i]
                - model.entity_emb[pos.tail][i]
        })
        .collect();
    let diff_neg: Vec<f64> = (0..dim)
        .map(|i| {
            model.entity_emb[neg.head][i] + model.relation_emb[neg.relation][i]
                - model.entity_emb[neg.tail][i]
        })
        .collect();

    let d_pos = if model.config.norm == 1 {
        diff_pos.iter().map(|x| x.abs()).sum::<f64>()
    } else {
        l2_norm(&diff_pos)
    };
    let d_neg = if model.config.norm == 1 {
        diff_neg.iter().map(|x| x.abs()).sum::<f64>()
    } else {
        l2_norm(&diff_neg)
    };

    let margin = model.config.margin;
    if margin - d_pos + d_neg <= 0.0 {
        return;
    }

    let grad_pos: Vec<f64> = (0..dim)
        .map(|i| {
            if model.config.norm == 1 {
                diff_pos[i].signum()
            } else if d_pos > 1e-12 {
                diff_pos[i] / d_pos
            } else {
                0.0
            }
        })
        .collect();
    let grad_neg: Vec<f64> = (0..dim)
        .map(|i| {
            if model.config.norm == 1 {
                diff_neg[i].signum()
            } else if d_neg > 1e-12 {
                diff_neg[i] / d_neg
            } else {
                0.0
            }
        })
        .collect();

    for i in 0..dim {
        let g = clip(grad_pos[i], clip_val);
        model.entity_emb[pos.head][i] -= lr * g;
        model.relation_emb[pos.relation][i] -= lr * g;
        model.entity_emb[pos.tail][i] += lr * g;
    }

    for i in 0..dim {
        let g = clip(grad_neg[i], clip_val);
        model.entity_emb[neg.head][i] += lr * g;
        model.relation_emb[neg.relation][i] += lr * g;
        model.entity_emb[neg.tail][i] -= lr * g;
    }

    let all_entities = vec![pos.head, neg.head, pos.tail, neg.tail];
    let mut seen = std::collections::HashSet::new();
    for &e in &all_entities {
        if seen.insert(e) {
            normalize_vec(&mut model.entity_emb[e]);
        }
    }
}

/// Finite-difference gradient update for RotatE.
pub(crate) fn rotate_grad_update(
    model: &mut RotatEModel,
    pos: &KgTriple,
    neg: &KgTriple,
    lr: f64,
) {
    let score_pos = model.score(pos.head, pos.relation, pos.tail);
    let score_neg = model.score(neg.head, neg.relation, neg.tail);

    let margin = model.config.margin;
    if margin - score_pos + score_neg <= 0.0 {
        return;
    }

    let h = 1e-5_f64;
    let clip_val = 1.0_f64;
    let complex_dim = model.complex_dim;

    // --- Update entity_real for pos.head ---
    for k in 0..complex_dim {
        let orig = model.entity_real[pos.head][k];
        model.entity_real[pos.head][k] = orig + h;
        let sp = model.score(pos.head, pos.relation, pos.tail);
        model.entity_real[pos.head][k] = orig - h;
        let sm = model.score(pos.head, pos.relation, pos.tail);
        model.entity_real[pos.head][k] = orig;
        let g = clip((sp - sm) / (2.0 * h), clip_val);
        model.entity_real[pos.head][k] -= lr * g;
    }
    for k in 0..complex_dim {
        let orig = model.entity_imag[pos.head][k];
        model.entity_imag[pos.head][k] = orig + h;
        let sp = model.score(pos.head, pos.relation, pos.tail);
        model.entity_imag[pos.head][k] = orig - h;
        let sm = model.score(pos.head, pos.relation, pos.tail);
        model.entity_imag[pos.head][k] = orig;
        let g = clip((sp - sm) / (2.0 * h), clip_val);
        model.entity_imag[pos.head][k] -= lr * g;
    }
    for k in 0..complex_dim {
        let orig = model.entity_real[pos.tail][k];
        model.entity_real[pos.tail][k] = orig + h;
        let sp = model.score(pos.head, pos.relation, pos.tail);
        model.entity_real[pos.tail][k] = orig - h;
        let sm = model.score(pos.head, pos.relation, pos.tail);
        model.entity_real[pos.tail][k] = orig;
        let g = clip((sp - sm) / (2.0 * h), clip_val);
        model.entity_real[pos.tail][k] -= lr * g;
    }
    for k in 0..complex_dim {
        let orig = model.entity_imag[pos.tail][k];
        model.entity_imag[pos.tail][k] = orig + h;
        let sp = model.score(pos.head, pos.relation, pos.tail);
        model.entity_imag[pos.tail][k] = orig - h;
        let sm = model.score(pos.head, pos.relation, pos.tail);
        model.entity_imag[pos.tail][k] = orig;
        let g = clip((sp - sm) / (2.0 * h), clip_val);
        model.entity_imag[pos.tail][k] -= lr * g;
    }
    for k in 0..complex_dim {
        let orig = model.relation_phase[pos.relation][k];
        model.relation_phase[pos.relation][k] = orig + h;
        let sp = model.score(pos.head, pos.relation, pos.tail);
        model.relation_phase[pos.relation][k] = orig - h;
        let sm = model.score(pos.head, pos.relation, pos.tail);
        model.relation_phase[pos.relation][k] = orig;
        let g = clip((sp - sm) / (2.0 * h), clip_val);
        model.relation_phase[pos.relation][k] -= lr * g;
        model.relation_phase[pos.relation][k] =
            model.relation_phase[pos.relation][k].clamp(-PI, PI);
    }

    // --- Negative triple ---
    for k in 0..complex_dim {
        let orig = model.entity_real[neg.head][k];
        model.entity_real[neg.head][k] = orig + h;
        let sp = model.score(neg.head, neg.relation, neg.tail);
        model.entity_real[neg.head][k] = orig - h;
        let sm = model.score(neg.head, neg.relation, neg.tail);
        model.entity_real[neg.head][k] = orig;
        let g = clip((sp - sm) / (2.0 * h), clip_val);
        model.entity_real[neg.head][k] += lr * g;
    }
    for k in 0..complex_dim {
        let orig = model.entity_imag[neg.head][k];
        model.entity_imag[neg.head][k] = orig + h;
        let sp = model.score(neg.head, neg.relation, neg.tail);
        model.entity_imag[neg.head][k] = orig - h;
        let sm = model.score(neg.head, neg.relation, neg.tail);
        model.entity_imag[neg.head][k] = orig;
        let g = clip((sp - sm) / (2.0 * h), clip_val);
        model.entity_imag[neg.head][k] += lr * g;
    }
    for k in 0..complex_dim {
        let orig = model.entity_real[neg.tail][k];
        model.entity_real[neg.tail][k] = orig + h;
        let sp = model.score(neg.head, neg.relation, neg.tail);
        model.entity_real[neg.tail][k] = orig - h;
        let sm = model.score(neg.head, neg.relation, neg.tail);
        model.entity_real[neg.tail][k] = orig;
        let g = clip((sp - sm) / (2.0 * h), clip_val);
        model.entity_real[neg.tail][k] += lr * g;
    }
    for k in 0..complex_dim {
        let orig = model.entity_imag[neg.tail][k];
        model.entity_imag[neg.tail][k] = orig + h;
        let sp = model.score(neg.head, neg.relation, neg.tail);
        model.entity_imag[neg.tail][k] = orig - h;
        let sm = model.score(neg.head, neg.relation, neg.tail);
        model.entity_imag[neg.tail][k] = orig;
        let g = clip((sp - sm) / (2.0 * h), clip_val);
        model.entity_imag[neg.tail][k] += lr * g;
    }

    model.rebuild_flat_cache();
}

/// Analytical gradient update for ComplEx using softplus loss + L2 regularisation.
pub(crate) fn complex_grad_update(
    model: &mut ComplExModel,
    pos: &KgTriple,
    neg: &KgTriple,
    lr: f64,
) {
    let dim = model.config.dim;
    let lambda = model.config.lambda;
    let clip_val = 1.0_f64;

    let sigmoid = |x: f64| -> f64 { 1.0 / (1.0 + (-x).exp()) };

    let sp = model.complex_score(pos.head, pos.relation, pos.tail);
    let sn = model.complex_score(neg.head, neg.relation, neg.tail);

    let coeff_p = -sigmoid(-sp);
    let coeff_n = sigmoid(sn);

    {
        let ph = pos.head;
        let pr = pos.relation;
        let pt = pos.tail;

        for k in 0..dim {
            let hr = model.entity_real[ph][k];
            let hi = model.entity_imag[ph][k];
            let rr = model.relation_real[pr][k];
            let ri = model.relation_imag[pr][k];
            let tr = model.entity_real[pt][k];
            let ti = model.entity_imag[pt][k];

            let g_hr = clip(coeff_p * (rr * tr + ri * ti) + lambda * hr, clip_val);
            let g_hi = clip(coeff_p * (rr * ti - ri * tr) + lambda * hi, clip_val);
            let g_rr = clip(coeff_p * (hr * tr + hi * ti) + lambda * rr, clip_val);
            let g_ri = clip(coeff_p * (hr * ti - hi * tr) + lambda * ri, clip_val);
            let g_tr = clip(coeff_p * (hr * rr + hi * ri) + lambda * tr, clip_val);
            let g_ti = clip(coeff_p * (hr * ri - hi * rr) + lambda * ti, clip_val);

            model.entity_real[ph][k] -= lr * g_hr;
            model.entity_imag[ph][k] -= lr * g_hi;
            model.relation_real[pr][k] -= lr * g_rr;
            model.relation_imag[pr][k] -= lr * g_ri;
            model.entity_real[pt][k] -= lr * g_tr;
            model.entity_imag[pt][k] -= lr * g_ti;
        }
    }

    {
        let nh = neg.head;
        let nr = neg.relation;
        let nt = neg.tail;

        for k in 0..dim {
            let hr = model.entity_real[nh][k];
            let hi = model.entity_imag[nh][k];
            let rr = model.relation_real[nr][k];
            let ri = model.relation_imag[nr][k];
            let tr = model.entity_real[nt][k];
            let ti = model.entity_imag[nt][k];

            let g_hr = clip(coeff_n * (rr * tr + ri * ti) + lambda * hr, clip_val);
            let g_hi = clip(coeff_n * (rr * ti - ri * tr) + lambda * hi, clip_val);
            let g_rr = clip(coeff_n * (hr * tr + hi * ti) + lambda * rr, clip_val);
            let g_ri = clip(coeff_n * (hr * ti - hi * tr) + lambda * ri, clip_val);
            let g_tr = clip(coeff_n * (hr * rr + hi * ri) + lambda * tr, clip_val);
            let g_ti = clip(coeff_n * (hr * ri - hi * rr) + lambda * ti, clip_val);

            model.entity_real[nh][k] -= lr * g_hr;
            model.entity_imag[nh][k] -= lr * g_hi;
            model.relation_real[nr][k] -= lr * g_rr;
            model.relation_imag[nr][k] -= lr * g_ri;
            model.entity_real[nt][k] -= lr * g_tr;
            model.entity_imag[nt][k] -= lr * g_ti;
        }
    }

    model.rebuild_flat_cache();
}

// ─────────────────────────────────────────────────────────────────────────────
// Trainer
// ─────────────────────────────────────────────────────────────────────────────

/// Training configuration and entry-point for KGE models.
pub struct KgeTrainer {
    /// Which model architecture to train.
    pub model_type: KgeModelType,
    /// Learning rate for SGD.
    pub lr: f64,
    /// Number of training epochs.
    pub n_epochs: usize,
    /// Mini-batch size (number of positive triples per step).
    pub batch_size: usize,
    /// Number of negative samples generated per positive triple.
    pub n_negatives: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl KgeTrainer {
    /// Train a KGE model on `dataset` and return the training history and
    /// the trained model.
    pub fn train(&self, dataset: &KgDataset) -> Result<KgeTrainingResult> {
        if dataset.n_triples() == 0 {
            return Err(TensorError::invalid_argument(
                "dataset must contain at least one triple".into(),
            ));
        }
        if dataset.n_entities() == 0 {
            return Err(TensorError::invalid_argument(
                "dataset must contain at least one entity".into(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(self.seed);

        match &self.model_type {
            KgeModelType::TransE(cfg) => {
                let mut model = TransEModel::new(
                    dataset.n_entities(),
                    dataset.n_relations(),
                    cfg.clone(),
                    &mut rng,
                )?;
                let (loss_history, final_loss) =
                    self.train_transe(&mut model, dataset, &mut rng)?;
                Ok(KgeTrainingResult {
                    loss_history,
                    final_loss,
                    model: TrainedKgeModel::TransE(model),
                })
            }
            KgeModelType::RotatE(cfg) => {
                let mut model = RotatEModel::new(
                    dataset.n_entities(),
                    dataset.n_relations(),
                    cfg.clone(),
                    &mut rng,
                )?;
                let (loss_history, final_loss) =
                    self.train_rotate(&mut model, dataset, &mut rng)?;
                Ok(KgeTrainingResult {
                    loss_history,
                    final_loss,
                    model: TrainedKgeModel::RotatE(model),
                })
            }
            KgeModelType::ComplEx(cfg) => {
                let mut model = ComplExModel::new(
                    dataset.n_entities(),
                    dataset.n_relations(),
                    cfg.clone(),
                    &mut rng,
                )?;
                let (loss_history, final_loss) =
                    self.train_complex(&mut model, dataset, &mut rng)?;
                Ok(KgeTrainingResult {
                    loss_history,
                    final_loss,
                    model: TrainedKgeModel::ComplEx(model),
                })
            }
        }
    }

    fn train_transe(
        &self,
        model: &mut TransEModel,
        dataset: &KgDataset,
        rng: &mut StdRng,
    ) -> Result<(Vec<f64>, f64)> {
        let mut loss_history = Vec::with_capacity(self.n_epochs);
        let triples = &dataset.triples;
        let n_triples = triples.len();

        for _epoch in 0..self.n_epochs {
            let mut epoch_loss = 0.0_f64;
            let mut n_batches = 0usize;

            let mut idx: Vec<usize> = (0..n_triples).collect();
            for i in (1..n_triples).rev() {
                let j = (rng.random::<f64>() * (i + 1) as f64) as usize % (i + 1);
                idx.swap(i, j);
            }

            let mut start = 0usize;
            while start < n_triples {
                let end = (start + self.batch_size).min(n_triples);
                let mut batch_loss = 0.0_f64;

                for &ti in &idx[start..end] {
                    let pos = &triples[ti];

                    for neg_k in 0..self.n_negatives {
                        let corrupt_head = neg_k % 2 == 0;
                        let neg = dataset.negative_sample(pos, rng, corrupt_head);

                        let score_pos = model.score(pos.head, pos.relation, pos.tail);
                        let score_neg = model.score(neg.head, neg.relation, neg.tail);
                        let loss = (model.config.margin - score_pos + score_neg).max(0.0);
                        batch_loss += loss;

                        transe_grad_update(model, pos, &neg, self.lr);
                    }
                }

                let n_pairs = ((end - start) * self.n_negatives) as f64;
                if n_pairs > 0.0 {
                    epoch_loss += batch_loss / n_pairs;
                    n_batches += 1;
                }
                start = end;
            }

            let avg_loss = if n_batches > 0 {
                epoch_loss / n_batches as f64
            } else {
                0.0
            };
            loss_history.push(avg_loss);
        }

        let final_loss = loss_history.last().cloned().unwrap_or(0.0);
        Ok((loss_history, final_loss))
    }

    fn train_rotate(
        &self,
        model: &mut RotatEModel,
        dataset: &KgDataset,
        rng: &mut StdRng,
    ) -> Result<(Vec<f64>, f64)> {
        let mut loss_history = Vec::with_capacity(self.n_epochs);
        let triples = &dataset.triples;
        let n_triples = triples.len();

        for _epoch in 0..self.n_epochs {
            let mut epoch_loss = 0.0_f64;
            let mut n_batches = 0usize;

            let mut idx: Vec<usize> = (0..n_triples).collect();
            for i in (1..n_triples).rev() {
                let j = (rng.random::<f64>() * (i + 1) as f64) as usize % (i + 1);
                idx.swap(i, j);
            }

            let mut start = 0usize;
            while start < n_triples {
                let end = (start + self.batch_size).min(n_triples);
                let mut batch_loss = 0.0_f64;

                for &ti in &idx[start..end] {
                    let pos = triples[ti];

                    for neg_k in 0..self.n_negatives {
                        let corrupt_head = neg_k % 2 == 0;
                        let neg = dataset.negative_sample(&pos, rng, corrupt_head);

                        let score_pos = model.score(pos.head, pos.relation, pos.tail);
                        let score_neg = model.score(neg.head, neg.relation, neg.tail);
                        let loss = (model.config.margin - score_pos + score_neg).max(0.0);
                        batch_loss += loss;

                        rotate_grad_update(model, &pos, &neg, self.lr);
                    }
                }

                let n_pairs = ((end - start) * self.n_negatives) as f64;
                if n_pairs > 0.0 {
                    epoch_loss += batch_loss / n_pairs;
                    n_batches += 1;
                }
                start = end;
            }

            let avg_loss = if n_batches > 0 {
                epoch_loss / n_batches as f64
            } else {
                0.0
            };
            loss_history.push(avg_loss);
        }

        let final_loss = loss_history.last().cloned().unwrap_or(0.0);
        Ok((loss_history, final_loss))
    }

    fn train_complex(
        &self,
        model: &mut ComplExModel,
        dataset: &KgDataset,
        rng: &mut StdRng,
    ) -> Result<(Vec<f64>, f64)> {
        let mut loss_history = Vec::with_capacity(self.n_epochs);
        let triples = &dataset.triples;
        let n_triples = triples.len();

        for _epoch in 0..self.n_epochs {
            let mut epoch_loss = 0.0_f64;
            let mut n_batches = 0usize;

            let mut idx: Vec<usize> = (0..n_triples).collect();
            for i in (1..n_triples).rev() {
                let j = (rng.random::<f64>() * (i + 1) as f64) as usize % (i + 1);
                idx.swap(i, j);
            }

            let mut start = 0usize;
            while start < n_triples {
                let end = (start + self.batch_size).min(n_triples);
                let mut batch_loss = 0.0_f64;

                for &ti in &idx[start..end] {
                    let pos = triples[ti];

                    for neg_k in 0..self.n_negatives {
                        let corrupt_head = neg_k % 2 == 0;
                        let neg = dataset.negative_sample(&pos, rng, corrupt_head);

                        let sp = model.complex_score(pos.head, pos.relation, pos.tail);
                        let sn = model.complex_score(neg.head, neg.relation, neg.tail);
                        let loss = softplus(-sp) + softplus(sn);
                        batch_loss += loss;

                        complex_grad_update(model, &pos, &neg, self.lr);
                    }
                }

                let n_pairs = ((end - start) * self.n_negatives) as f64;
                if n_pairs > 0.0 {
                    epoch_loss += batch_loss / n_pairs;
                    n_batches += 1;
                }
                start = end;
            }

            let avg_loss = if n_batches > 0 {
                epoch_loss / n_batches as f64
            } else {
                0.0
            };
            loss_history.push(avg_loss);
        }

        let final_loss = loss_history.last().cloned().unwrap_or(0.0);
        Ok((loss_history, final_loss))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluator
// ─────────────────────────────────────────────────────────────────────────────

/// Link-prediction evaluation in the filtered setting.
///
/// For each test triple `(h, r, t)`, the evaluator corrupts either head or
/// tail, scores all candidates, and ranks the true entity — excluding other
/// known positives from the denominator.
pub struct KgeEvaluator;

impl KgeEvaluator {
    /// Compute Hits@k (proportion of test triples where the true entity is
    /// ranked in the top k) in the *filtered* setting.
    pub fn hits_at_k(model: &dyn KgeModel, dataset: &KgDataset, k: usize) -> f64 {
        let ranks = Self::compute_filtered_ranks(model, dataset);
        if ranks.is_empty() {
            return 0.0;
        }
        let hits = ranks.iter().filter(|&&r| r <= k).count();
        hits as f64 / ranks.len() as f64
    }

    /// Mean rank of the true entity in the filtered ranking.
    pub fn mean_rank(model: &dyn KgeModel, dataset: &KgDataset) -> f64 {
        let ranks = Self::compute_filtered_ranks(model, dataset);
        if ranks.is_empty() {
            return 0.0;
        }
        ranks.iter().sum::<usize>() as f64 / ranks.len() as f64
    }

    /// Mean reciprocal rank (MRR) in the filtered setting.
    pub fn mean_reciprocal_rank(model: &dyn KgeModel, dataset: &KgDataset) -> f64 {
        let ranks = Self::compute_filtered_ranks(model, dataset);
        if ranks.is_empty() {
            return 0.0;
        }
        ranks.iter().map(|&r| 1.0 / r as f64).sum::<f64>() / ranks.len() as f64
    }

    fn compute_filtered_ranks(model: &dyn KgeModel, dataset: &KgDataset) -> Vec<usize> {
        let n_entities = dataset.n_entities();
        let mut ranks = Vec::with_capacity(dataset.n_triples() * 2);

        for triple in &dataset.triples {
            {
                let true_score = model.score(triple.head, triple.relation, triple.tail);
                let rank = (0..n_entities)
                    .filter(|&c| {
                        c != triple.tail && !dataset.is_positive(triple.head, triple.relation, c)
                    })
                    .filter(|&c| model.score(triple.head, triple.relation, c) >= true_score)
                    .count()
                    + 1;
                ranks.push(rank);
            }

            {
                let true_score = model.score(triple.head, triple.relation, triple.tail);
                let rank = (0..n_entities)
                    .filter(|&c| {
                        c != triple.head && !dataset.is_positive(c, triple.relation, triple.tail)
                    })
                    .filter(|&c| model.score(c, triple.relation, triple.tail) >= true_score)
                    .count()
                    + 1;
                ranks.push(rank);
            }
        }

        ranks
    }
}
