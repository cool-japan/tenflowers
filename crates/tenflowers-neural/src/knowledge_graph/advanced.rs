//! Advanced Knowledge Graph extensions.
//!
//! # Temporal Knowledge Graphs
//! - [`TemporalTriple`]: time-stamped quadruple (h, r, t, time).
//! - [`TeRoModel`]: Temporal RotatE — time-aware rotation in complex space.
//! - [`TntComplExModel`]: Temporal ComplEx with independent time embeddings.
//! - [`TemporalKgMetrics`]: filtered MRR/Hits@k with timestamp filtering.
//!
//! # Hyper-Relational KGs
//! - [`HypRelQuadruple`]: (h, r, t, qualifiers) with arbitrary qualifier pairs.
//! - [`StarEModel`]: StarE — qualifier-aware KG embedding via composition.
//! - [`NaLPModel`]: Neighbor-aware LP for hyper-relational KG reasoning.
//!
//! # KG + LLM Integration
//! - [`KgTextualizer`]: verbalize triples as natural language sentences.
//! - [`KgEmbeddingAlignment`]: align KG entity embeddings with text encoder.
//! - [`KgQaRanker`]: KG-enhanced question answering.
//!
//! # KG Inference
//! - [`RuleInduction`]: AMIE+-style Horn rule learning.
//! - [`PathReasoningModel`]: random walk + attention for multi-hop reasoning.
//! - [`KgMetricsExtended`]: rule confidence, path accuracy, entity coverage.

use super::{l2_norm, normalize_vec, xavier_init_1d, KgDataset, KgTriple, KgeModel};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use std::f64::consts::PI;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// §1  Temporal Knowledge Graphs
// ─────────────────────────────────────────────────────────────────────────────

/// A time-stamped knowledge graph quadruple `(head, relation, tail, timestamp)`.
///
/// Timestamp is an integer that can represent a year, epoch, or any discrete
/// time index.  Filtering during evaluation is done per timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemporalTriple {
    /// Head entity index.
    pub head: usize,
    /// Relation index.
    pub relation: usize,
    /// Tail entity index.
    pub tail: usize,
    /// Discrete timestamp index.
    pub timestamp: usize,
}

impl TemporalTriple {
    /// Convenience constructor.
    pub fn new(head: usize, relation: usize, tail: usize, timestamp: usize) -> Self {
        Self {
            head,
            relation,
            tail,
            timestamp,
        }
    }

    /// Convert to a static [`KgTriple`] by discarding the timestamp.
    pub fn to_static(&self) -> KgTriple {
        KgTriple::new(self.head, self.relation, self.tail)
    }
}

/// Temporal RotatE model (Xu 2020).
///
/// Extends RotatE by encoding time as a rotation: for each timestamp `t`, a
/// learned time rotation `r_t ∈ ℂ^(dim/2)` is composed with the relation
/// rotation `r_r`, yielding a joint rotation `r_r ⊗ r_t`.
///
/// Score: `-(‖(h ∘ r_r ∘ r_t) − tail‖)`
#[derive(Debug, Clone)]
pub struct TeRoModel {
    /// Entity real parts, shape `[n_entities][complex_dim]`.
    pub entity_real: Vec<Vec<f64>>,
    /// Entity imaginary parts, shape `[n_entities][complex_dim]`.
    pub entity_imag: Vec<Vec<f64>>,
    /// Relation phase angles, shape `[n_relations][complex_dim]`.
    pub relation_phase: Vec<Vec<f64>>,
    /// Time phase angles, shape `[n_timestamps][complex_dim]`.
    pub time_phase: Vec<Vec<f64>>,
    /// Embedding dimension (must be even).
    pub dim: usize,
    /// Half of the embedding dimension (complex space size).
    pub complex_dim: usize,
}

impl TeRoModel {
    /// Construct a randomly-initialised TeRo model.
    pub fn new(
        n_entities: usize,
        n_relations: usize,
        n_timestamps: usize,
        dim: usize,
        rng: &mut StdRng,
    ) -> Result<Self> {
        if dim < 2 || dim % 2 != 0 {
            return Err(TensorError::invalid_argument(
                "TeRo dim must be even and >= 2".into(),
            ));
        }
        if n_entities == 0 || n_relations == 0 || n_timestamps == 0 {
            return Err(TensorError::invalid_argument(
                "n_entities, n_relations, n_timestamps must all be > 0".into(),
            ));
        }
        let complex_dim = dim / 2;

        let sample_real = |rng: &mut StdRng, n: usize| -> Vec<Vec<f64>> {
            (0..n)
                .map(|_| {
                    (0..complex_dim)
                        .map(|_| rng.random::<f64>() * 2.0 - 1.0)
                        .collect()
                })
                .collect()
        };
        let sample_phase = |rng: &mut StdRng, n: usize| -> Vec<Vec<f64>> {
            (0..n)
                .map(|_| {
                    (0..complex_dim)
                        .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * PI)
                        .collect()
                })
                .collect()
        };

        Ok(Self {
            entity_real: sample_real(rng, n_entities),
            entity_imag: sample_real(rng, n_entities),
            relation_phase: sample_phase(rng, n_relations),
            time_phase: sample_phase(rng, n_timestamps),
            dim,
            complex_dim,
        })
    }

    /// Score a temporal triple `(h, r, t, time)`.
    ///
    /// Computes `‖(h ∘ r_r ∘ r_t) − tail‖`.
    pub fn score_temporal(&self, h: usize, r: usize, tail: usize, time: usize) -> f64 {
        let hr = &self.entity_real[h];
        let hi = &self.entity_imag[h];
        let rp = &self.relation_phase[r];
        let tp = &self.time_phase[time];
        let tr = &self.entity_real[tail];
        let ti = &self.entity_imag[tail];

        let mut sum_sq = 0.0_f64;
        for k in 0..self.complex_dim {
            // First compose with relation rotation
            let cos_r = rp[k].cos();
            let sin_r = rp[k].sin();
            let rot1_r = hr[k] * cos_r - hi[k] * sin_r;
            let rot1_i = hr[k] * sin_r + hi[k] * cos_r;
            // Then compose with time rotation
            let cos_t = tp[k].cos();
            let sin_t = tp[k].sin();
            let rot2_r = rot1_r * cos_t - rot1_i * sin_t;
            let rot2_i = rot1_r * sin_t + rot1_i * cos_t;
            // Distance to tail
            let diff_r = rot2_r - tr[k];
            let diff_i = rot2_i - ti[k];
            sum_sq += diff_r * diff_r + diff_i * diff_i;
        }
        -sum_sq.sqrt()
    }
}

/// Temporal ComplEx model (TntComplEx).
///
/// Extends ComplEx with a dedicated time embedding per timestamp.  The score
/// is `Re(<h, r, t̄, τ>)` where τ is the conjugate time embedding.
#[derive(Debug, Clone)]
pub struct TntComplExModel {
    /// Entity real parts.
    pub entity_real: Vec<Vec<f64>>,
    /// Entity imaginary parts.
    pub entity_imag: Vec<Vec<f64>>,
    /// Relation real parts.
    pub relation_real: Vec<Vec<f64>>,
    /// Relation imaginary parts.
    pub relation_imag: Vec<Vec<f64>>,
    /// Time real parts.
    pub time_real: Vec<Vec<f64>>,
    /// Time imaginary parts.
    pub time_imag: Vec<Vec<f64>>,
    /// Embedding dimension.
    pub dim: usize,
    /// L2 regularisation coefficient.
    pub lambda: f64,
}

impl TntComplExModel {
    /// Construct a randomly-initialised TntComplEx model.
    pub fn new(
        n_entities: usize,
        n_relations: usize,
        n_timestamps: usize,
        dim: usize,
        lambda: f64,
        rng: &mut StdRng,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(TensorError::invalid_argument(
                "TntComplEx dim must be > 0".into(),
            ));
        }
        let init = |rng: &mut StdRng, n: usize| -> Vec<Vec<f64>> {
            (0..n)
                .map(|_| xavier_init_1d(dim, dim, dim, rng))
                .collect()
        };
        Ok(Self {
            entity_real: init(rng, n_entities),
            entity_imag: init(rng, n_entities),
            relation_real: init(rng, n_relations),
            relation_imag: init(rng, n_relations),
            time_real: init(rng, n_timestamps),
            time_imag: init(rng, n_timestamps),
            dim,
            lambda,
        })
    }

    /// Score: `Re(<h, r, t̄, τ>)` summed over dimensions.
    ///
    /// The 4-way interaction is: `h_r * r_r * t_r * τ_r` + cross terms.
    /// We use the simplified decomposition: score ≈ Re(<h ⊗ r, t̄ ⊗ τ̄>).
    pub fn score_temporal(&self, h: usize, r: usize, t: usize, time: usize) -> f64 {
        let hr = &self.entity_real[h];
        let hi = &self.entity_imag[h];
        let rr = &self.relation_real[r];
        let ri = &self.relation_imag[r];
        let tr = &self.entity_real[t];
        let ti = &self.entity_imag[t];
        let taur = &self.time_real[time];
        let taui = &self.time_imag[time];

        (0..self.dim)
            .map(|k| {
                // Re(<h, r>) * Re(<t_conj, tau_conj>)
                let re_hr = hr[k] * rr[k] - hi[k] * ri[k];
                let im_hr = hr[k] * ri[k] + hi[k] * rr[k];
                // t conjugate
                let re_t_conj = tr[k];
                let im_t_conj = -ti[k];
                // tau conjugate
                let re_tau_conj = taur[k];
                let im_tau_conj = -taui[k];
                // Re(hr * t_conj * tau_conj)
                let re_t_tau = re_t_conj * re_tau_conj - im_t_conj * im_tau_conj;
                let im_t_tau = re_t_conj * im_tau_conj + im_t_conj * re_tau_conj;
                re_hr * re_t_tau - im_hr * im_t_tau
            })
            .sum::<f64>()
    }
}

/// Evaluation metrics for temporal knowledge graphs.
#[derive(Debug, Clone)]
pub struct TemporalKgMetrics {
    /// Filtered MRR across all temporal test triples.
    pub mrr: f64,
    /// Hits@1.
    pub hits_at_1: f64,
    /// Hits@3.
    pub hits_at_3: f64,
    /// Hits@10.
    pub hits_at_10: f64,
    /// Number of test triples evaluated.
    pub n_evaluated: usize,
}

impl TemporalKgMetrics {
    /// Compute metrics from a scorer function and list of temporal triples.
    ///
    /// `scorer(h, r, t, time)` returns a score (higher = more plausible).
    /// `n_entities` is the total entity count used for ranking.
    pub fn compute<F>(
        test_triples: &[TemporalTriple],
        n_entities: usize,
        positive_set: &std::collections::HashSet<(usize, usize, usize, usize)>,
        scorer: F,
    ) -> Self
    where
        F: Fn(usize, usize, usize, usize) -> f64,
    {
        if test_triples.is_empty() || n_entities == 0 {
            return Self {
                mrr: 0.0,
                hits_at_1: 0.0,
                hits_at_3: 0.0,
                hits_at_10: 0.0,
                n_evaluated: 0,
            };
        }

        let mut sum_rr = 0.0_f64;
        let mut h1 = 0usize;
        let mut h3 = 0usize;
        let mut h10 = 0usize;
        let n = test_triples.len();

        for qt in test_triples {
            let true_score = scorer(qt.head, qt.relation, qt.tail, qt.timestamp);
            let rank = (0..n_entities)
                .filter(|&c| {
                    c != qt.tail
                        && !positive_set.contains(&(qt.head, qt.relation, c, qt.timestamp))
                })
                .filter(|&c| scorer(qt.head, qt.relation, c, qt.timestamp) >= true_score)
                .count()
                + 1;

            sum_rr += 1.0 / rank as f64;
            if rank <= 1 {
                h1 += 1;
            }
            if rank <= 3 {
                h3 += 1;
            }
            if rank <= 10 {
                h10 += 1;
            }
        }

        Self {
            mrr: sum_rr / n as f64,
            hits_at_1: h1 as f64 / n as f64,
            hits_at_3: h3 as f64 / n as f64,
            hits_at_10: h10 as f64 / n as f64,
            n_evaluated: n,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  Hyper-Relational KGs
// ─────────────────────────────────────────────────────────────────────────────

/// A hyper-relational quadruple: `(head, relation, tail, qualifiers)`.
///
/// `qualifiers` is a list of `(qualifier_key_relation, qualifier_value_entity)`
/// pairs that refine the main triple semantics.
#[derive(Debug, Clone)]
pub struct HypRelQuadruple {
    /// Head entity index.
    pub head: usize,
    /// Relation index.
    pub relation: usize,
    /// Tail entity index.
    pub tail: usize,
    /// Qualifier pairs: (qualifier relation index, qualifier value entity index).
    pub qualifiers: Vec<(usize, usize)>,
}

impl HypRelQuadruple {
    /// Constructor.
    pub fn new(
        head: usize,
        relation: usize,
        tail: usize,
        qualifiers: Vec<(usize, usize)>,
    ) -> Self {
        Self {
            head,
            relation,
            tail,
            qualifiers,
        }
    }
}

/// StarE model for hyper-relational KG embedding (Galkin 2020).
///
/// Composes qualifier key-value embeddings into the relation embedding using
/// a circular correlation composition: `r_composed = r ⊛ (⊕_i compose(qr_i, qv_i))`.
/// Then uses TransE-style scoring on the composed triple.
#[derive(Debug, Clone)]
pub struct StarEModel {
    /// Entity embeddings.
    pub entity_emb: Vec<Vec<f64>>,
    /// Relation embeddings.
    pub relation_emb: Vec<Vec<f64>>,
    /// Embedding dimension.
    pub dim: usize,
}

impl StarEModel {
    /// Construct with Xavier-initialised embeddings.
    pub fn new(
        n_entities: usize,
        n_relations: usize,
        dim: usize,
        rng: &mut StdRng,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(TensorError::invalid_argument(
                "StarE dim must be > 0".into(),
            ));
        }
        let entity_emb: Vec<Vec<f64>> = (0..n_entities)
            .map(|_| {
                let mut v = xavier_init_1d(dim, dim, dim, rng);
                normalize_vec(&mut v);
                v
            })
            .collect();
        let relation_emb: Vec<Vec<f64>> = (0..n_relations)
            .map(|_| xavier_init_1d(dim, dim, dim, rng))
            .collect();
        Ok(Self {
            entity_emb,
            relation_emb,
            dim,
        })
    }

    /// Circular correlation of two real vectors: `(a ⊛ b)[k] = Σ_i a[i] * b[(i+k) % d]`.
    fn circular_correlation(a: &[f64], b: &[f64]) -> Vec<f64> {
        let d = a.len().min(b.len());
        (0..d)
            .map(|k| (0..d).map(|i| a[i] * b[(i + k) % d]).sum::<f64>())
            .collect()
    }

    /// Compose a relation embedding with qualifier pairs using circular correlation.
    pub fn compose_relation(&self, relation: usize, qualifiers: &[(usize, usize)]) -> Vec<f64> {
        let mut composed = self.relation_emb[relation].clone();
        for &(qr, qv) in qualifiers {
            if qr < self.relation_emb.len() && qv < self.entity_emb.len() {
                let qr_emb = &self.relation_emb[qr];
                let qv_emb = &self.entity_emb[qv];
                // Compose qualifier key-value
                let qkv = Self::circular_correlation(qr_emb, qv_emb);
                // Accumulate into composed relation
                for (c, &x) in composed.iter_mut().zip(qkv.iter()) {
                    *c += x;
                }
            }
        }
        // Normalize
        let norm = l2_norm(&composed);
        if norm > 1e-12 {
            for c in composed.iter_mut() {
                *c /= norm;
            }
        }
        composed
    }

    /// TransE-style score using the composed relation.
    pub fn score_quadruple(&self, quad: &HypRelQuadruple) -> f64 {
        let composed_r = self.compose_relation(quad.relation, &quad.qualifiers);
        let h = &self.entity_emb[quad.head];
        let t = &self.entity_emb[quad.tail];
        let diff: Vec<f64> = (0..self.dim)
            .map(|i| h[i] + composed_r[i] - t[i])
            .collect();
        -l2_norm(&diff)
    }
}

/// Neighbor-aware LP (NaLP) model for hyper-relational KG reasoning.
///
/// Augments triple scores by a qualifier compatibility term that measures
/// how well qualifiers align with head and tail representations.
#[derive(Debug, Clone)]
pub struct NaLPModel {
    /// Entity embeddings.
    pub entity_emb: Vec<Vec<f64>>,
    /// Relation embeddings.
    pub relation_emb: Vec<Vec<f64>>,
    /// Embedding dimension.
    pub dim: usize,
    /// Weight of the qualifier compatibility term.
    pub alpha: f64,
}

impl NaLPModel {
    /// Construct with random embeddings and qualifier weight `alpha`.
    pub fn new(
        n_entities: usize,
        n_relations: usize,
        dim: usize,
        alpha: f64,
        rng: &mut StdRng,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(TensorError::invalid_argument(
                "NaLP dim must be > 0".into(),
            ));
        }
        let init = |rng: &mut StdRng, n: usize| -> Vec<Vec<f64>> {
            (0..n)
                .map(|_| xavier_init_1d(dim, dim, dim, rng))
                .collect()
        };
        Ok(Self {
            entity_emb: init(rng, n_entities),
            relation_emb: init(rng, n_relations),
            dim,
            alpha,
        })
    }

    fn dot_prod(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Score a hyper-relational quadruple.
    ///
    /// Main score (DistMult-style) + α * qualifier compatibility.
    pub fn score_quadruple(&self, quad: &HypRelQuadruple) -> f64 {
        let h = &self.entity_emb[quad.head];
        let r = &self.relation_emb[quad.relation];
        let t = &self.entity_emb[quad.tail];

        // DistMult: Re(<h, r, t>)
        let main_score: f64 = (0..self.dim).map(|k| h[k] * r[k] * t[k]).sum();

        // Qualifier compatibility: average dot-product between qualifier value
        // and the entity (head or tail) for each qualifier.
        let qual_score = if quad.qualifiers.is_empty() {
            0.0
        } else {
            let sum: f64 = quad
                .qualifiers
                .iter()
                .filter_map(|&(qr, qv)| {
                    if qr < self.relation_emb.len() && qv < self.entity_emb.len() {
                        let qr_emb = &self.relation_emb[qr];
                        let qv_emb = &self.entity_emb[qv];
                        // Compatibility: <qr, h> * <qv, t>
                        Some(Self::dot_prod(qr_emb, h) * Self::dot_prod(qv_emb, t))
                    } else {
                        None
                    }
                })
                .sum();
            sum / quad.qualifiers.len() as f64
        };

        main_score + self.alpha * qual_score
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  KG + LLM Integration
// ─────────────────────────────────────────────────────────────────────────────

/// Verbalizes KG triples as natural language sentences using templates.
#[derive(Debug, Clone)]
pub struct KgTextualizer {
    /// Template: `{head} {predicate} {tail}`.
    /// `predicate` is derived from relation label by replacing underscores.
    pub relation_templates: HashMap<String, String>,
}

impl KgTextualizer {
    /// Construct with an empty template map.
    pub fn new() -> Self {
        Self {
            relation_templates: HashMap::new(),
        }
    }

    /// Register a verbalization template for a relation.
    pub fn register_template(&mut self, relation: &str, template: &str) {
        self.relation_templates
            .insert(relation.to_string(), template.to_string());
    }

    /// Verbalize a triple as a sentence.
    ///
    /// Uses the registered template if available; otherwise generates a
    /// default "`{head} [relation] {tail}`" sentence.
    pub fn verbalize(&self, head: &str, relation: &str, tail: &str) -> String {
        if let Some(tmpl) = self.relation_templates.get(relation) {
            tmpl.replace("{head}", head)
                .replace("{tail}", tail)
                .replace("{relation}", relation)
        } else {
            let readable_rel = relation.replace('_', " ");
            format!("{head} {readable_rel} {tail}")
        }
    }

    /// Verbalize all triples in a dataset using entity and relation labels.
    pub fn verbalize_dataset(&self, dataset: &KgDataset) -> Vec<String> {
        dataset
            .triples
            .iter()
            .map(|t| {
                let h = dataset
                    .entities
                    .get(t.head)
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                let r = dataset
                    .relations
                    .get(t.relation)
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                let tail = dataset
                    .entities
                    .get(t.tail)
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                self.verbalize(h, r, tail)
            })
            .collect()
    }
}

impl Default for KgTextualizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Aligns KG entity embeddings with text encoder embeddings via contrastive loss.
///
/// For each entity, we have a KG embedding and a text embedding.  Alignment
/// minimises an InfoNCE contrastive loss between matched (entity, text) pairs.
#[derive(Debug, Clone)]
pub struct KgEmbeddingAlignment {
    /// Projection matrix: text_dim → kg_dim.  Flat row-major `[kg_dim × text_dim]`.
    pub projection: Vec<f64>,
    /// KG embedding dimension.
    pub kg_dim: usize,
    /// Text embedding dimension.
    pub text_dim: usize,
    /// InfoNCE temperature.
    pub temperature: f64,
}

impl KgEmbeddingAlignment {
    /// Construct with a random orthogonal-ish projection.
    pub fn new(kg_dim: usize, text_dim: usize, temperature: f64, rng: &mut StdRng) -> Result<Self> {
        if kg_dim == 0 || text_dim == 0 {
            return Err(TensorError::invalid_argument(
                "kg_dim and text_dim must be > 0".into(),
            ));
        }
        let projection = xavier_init_1d(kg_dim * text_dim, text_dim, kg_dim, rng);
        Ok(Self {
            projection,
            kg_dim,
            text_dim,
            temperature,
        })
    }

    /// Project a text embedding into KG space.
    pub fn project_text(&self, text_emb: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0_f64; self.kg_dim];
        for i in 0..self.kg_dim {
            let row = i * self.text_dim;
            out[i] = text_emb
                .iter()
                .enumerate()
                .map(|(j, &x)| {
                    self.projection
                        .get(row + j)
                        .copied()
                        .unwrap_or(0.0)
                        * x
                })
                .sum();
        }
        out
    }

    /// Compute InfoNCE contrastive loss for a batch of (kg_emb, text_emb) pairs.
    ///
    /// Assumes `kg_embs[i]` and `text_embs[i]` are matched pairs.
    /// Returns the average loss over the batch.
    pub fn contrastive_loss(&self, kg_embs: &[Vec<f64>], text_embs: &[Vec<f64>]) -> f64 {
        let n = kg_embs.len().min(text_embs.len());
        if n == 0 {
            return 0.0;
        }
        let t = self.temperature.max(1e-8);
        let mut total = 0.0_f64;

        for i in 0..n {
            let proj = self.project_text(&text_embs[i]);
            // Cosine similarities
            let norm_proj = l2_norm(&proj).max(1e-12);
            let norm_kg = l2_norm(&kg_embs[i]).max(1e-12);

            let sims: Vec<f64> = (0..n)
                .map(|j| {
                    let norm_j = l2_norm(&kg_embs[j]).max(1e-12);
                    let dot: f64 = proj
                        .iter()
                        .zip(kg_embs[j].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    dot / (norm_proj * norm_j) / t
                })
                .collect();

            let _ = norm_kg; // used above indirectly
            let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sims: Vec<f64> = sims.iter().map(|&s| (s - max_sim).exp()).collect();
            let sum_exp: f64 = exp_sims.iter().sum();
            let log_softmax_i = (exp_sims[i] / sum_exp.max(1e-30)).ln();
            total -= log_softmax_i;
        }

        total / n as f64
    }
}

/// KG-enhanced question answering ranker.
///
/// Retrieves relevant triples for a question by entity matching, then
/// re-ranks candidate answers using triple scores from a KGE model.
#[derive(Debug, Clone)]
pub struct KgQaRanker {
    /// Number of top triples to retrieve per entity mention.
    pub top_k_triples: usize,
    /// KGE scoring weight vs. surface-form similarity weight.
    pub kge_weight: f64,
}

impl KgQaRanker {
    /// Construct with default parameters.
    pub fn new(top_k_triples: usize, kge_weight: f64) -> Self {
        Self {
            top_k_triples,
            kge_weight,
        }
    }

    /// Retrieve the top-k triples for a given head entity.
    pub fn retrieve_triples<'a>(
        &self,
        head_entity: usize,
        dataset: &'a KgDataset,
        model: &dyn KgeModel,
    ) -> Vec<(&'a KgTriple, f64)> {
        let mut scored: Vec<(&KgTriple, f64)> = dataset
            .triples
            .iter()
            .filter(|t| t.head == head_entity)
            .map(|t| {
                let score = model.score(t.head, t.relation, t.tail);
                (t, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.top_k_triples);
        scored
    }

    /// Rank answer candidates by combining KGE triple score with a provided
    /// surface similarity score for each candidate.
    ///
    /// Returns candidates sorted by combined score (highest first).
    pub fn rank_answers(
        &self,
        head_entity: usize,
        candidate_tails: &[usize],
        surface_scores: &[f64],
        dataset: &KgDataset,
        model: &dyn KgeModel,
    ) -> Vec<(usize, f64)> {
        // Find best relation for each candidate tail
        let n_rel = dataset.n_relations();
        let mut ranked: Vec<(usize, f64)> = candidate_tails
            .iter()
            .zip(surface_scores.iter())
            .map(|(&tail, &surf)| {
                let best_kge = (0..n_rel)
                    .map(|r| model.score(head_entity, r, tail))
                    .fold(f64::NEG_INFINITY, f64::max);
                let combined = self.kge_weight * best_kge + (1.0 - self.kge_weight) * surf;
                (tail, combined)
            })
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  KG Inference
// ─────────────────────────────────────────────────────────────────────────────

/// A learned Horn rule: `B1(x, z) ∧ B2(z, y) → H(x, y)`.
#[derive(Debug, Clone)]
pub struct HornRule {
    /// Head relation index.
    pub head_relation: usize,
    /// Body relations (in order), e.g. `[r1, r2]`.
    pub body_relations: Vec<usize>,
    /// Rule confidence = #(H ∧ B) / #(B).
    pub confidence: f64,
    /// Rule support = #(H ∧ B).
    pub support: usize,
}

/// AMIE+-style rule induction over a KG dataset.
///
/// Mines closed Horn rules of length 1 and 2 from the triple set.
/// Length-1 rules: `r1(x,y) → r_head(x,y)` (if they co-occur).
/// Length-2 rules: `r1(x,z) ∧ r2(z,y) → r_head(x,y)`.
#[derive(Debug, Clone)]
pub struct RuleInduction {
    /// Minimum rule support threshold.
    pub min_support: usize,
    /// Minimum rule confidence threshold.
    pub min_confidence: f64,
}

impl RuleInduction {
    /// Construct with minimum support and confidence.
    pub fn new(min_support: usize, min_confidence: f64) -> Self {
        Self {
            min_support,
            min_confidence,
        }
    }

    /// Mine length-1 and length-2 closed Horn rules from `dataset`.
    ///
    /// For efficiency, only the most common body relations are considered.
    /// Returns rules sorted by confidence (highest first).
    pub fn mine_rules(&self, dataset: &KgDataset) -> Vec<HornRule> {
        let triples = &dataset.triples;
        let n_rel = dataset.n_relations();

        // Build adjacency: relation → set of (head, tail) pairs
        let mut adj: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for t in triples {
            adj.entry(t.relation).or_default().push((t.head, t.tail));
        }

        let mut rules: Vec<HornRule> = Vec::new();

        // Length-1 rules: B1(x,y) → H(x,y)
        for head_rel in 0..n_rel {
            let head_pairs: std::collections::HashSet<(usize, usize)> = adj
                .get(&head_rel)
                .map(|v| v.iter().cloned().collect())
                .unwrap_or_default();

            for body_rel in 0..n_rel {
                if body_rel == head_rel {
                    continue;
                }
                if let Some(body_pairs) = adj.get(&body_rel) {
                    let support = body_pairs
                        .iter()
                        .filter(|&&(h, t)| head_pairs.contains(&(h, t)))
                        .count();
                    if support >= self.min_support {
                        let confidence = support as f64 / body_pairs.len().max(1) as f64;
                        if confidence >= self.min_confidence {
                            rules.push(HornRule {
                                head_relation: head_rel,
                                body_relations: vec![body_rel],
                                confidence,
                                support,
                            });
                        }
                    }
                }
            }
        }

        // Length-2 rules: B1(x,z) ∧ B2(z,y) → H(x,y)
        for head_rel in 0..n_rel {
            let head_pairs: std::collections::HashSet<(usize, usize)> = adj
                .get(&head_rel)
                .map(|v| v.iter().cloned().collect())
                .unwrap_or_default();

            for r1 in 0..n_rel.min(8) {
                // Limit to first 8 rels for tractability
                for r2 in 0..n_rel.min(8) {
                    if r1 == head_rel && r2 == head_rel {
                        continue;
                    }
                    let r1_pairs = adj.get(&r1).map(|v| v.as_slice()).unwrap_or(&[]);
                    let r2_map: HashMap<usize, Vec<usize>> = {
                        let mut m: HashMap<usize, Vec<usize>> = HashMap::new();
                        if let Some(r2p) = adj.get(&r2) {
                            for &(h, t) in r2p {
                                m.entry(h).or_default().push(t);
                            }
                        }
                        m
                    };
                    let mut support = 0usize;
                    let mut body_count = 0usize;
                    for &(x, z) in r1_pairs {
                        if let Some(tails) = r2_map.get(&z) {
                            for &y in tails {
                                body_count += 1;
                                if head_pairs.contains(&(x, y)) {
                                    support += 1;
                                }
                            }
                        }
                    }
                    if support >= self.min_support && body_count > 0 {
                        let confidence = support as f64 / body_count as f64;
                        if confidence >= self.min_confidence {
                            rules.push(HornRule {
                                head_relation: head_rel,
                                body_relations: vec![r1, r2],
                                confidence,
                                support,
                            });
                        }
                    }
                }
            }
        }

        rules.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        rules
    }
}

/// Path reasoning model: random walk with attention for multi-hop reasoning.
///
/// Performs biased random walks from a source entity, guided by relation
/// embeddings, and returns a probability distribution over reachable entities.
#[derive(Debug, Clone)]
pub struct PathReasoningModel {
    /// Maximum path length (number of hops).
    pub max_hops: usize,
    /// Number of random walk paths to sample.
    pub n_paths: usize,
    /// Temperature for softmax attention over relations.
    pub attention_temp: f64,
}

impl PathReasoningModel {
    /// Construct a new path reasoning model.
    pub fn new(max_hops: usize, n_paths: usize, attention_temp: f64) -> Self {
        Self {
            max_hops,
            n_paths,
            attention_temp,
        }
    }

    /// Run path reasoning from `source_entity` using the dataset's triple index.
    ///
    /// Returns a map from entity → visit probability (unnormalized count).
    pub fn reason(
        &self,
        source_entity: usize,
        target_relation: usize,
        dataset: &KgDataset,
        model: &dyn KgeModel,
        rng: &mut StdRng,
    ) -> HashMap<usize, f64> {
        // Build adjacency list: entity → list of (relation, tail)
        let mut adj: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for t in &dataset.triples {
            adj.entry(t.head).or_default().push((t.relation, t.tail));
        }

        let mut entity_scores: HashMap<usize, f64> = HashMap::new();

        for _ in 0..self.n_paths {
            let mut current = source_entity;
            let mut path_score = 1.0_f64;

            for _hop in 0..self.max_hops {
                let neighbors = match adj.get(&current) {
                    Some(n) if !n.is_empty() => n,
                    _ => break,
                };

                // Attention scores based on relation similarity to target relation
                let target_emb = model.relation_embedding(target_relation);
                let scores: Vec<f64> = neighbors
                    .iter()
                    .map(|&(r, _)| {
                        if r < dataset.n_relations() {
                            let rel_emb = model.relation_embedding(r);
                            let dot: f64 =
                                target_emb.iter().zip(rel_emb.iter()).map(|(a, b)| a * b).sum();
                            (dot / self.attention_temp).exp()
                        } else {
                            1.0_f64.exp()
                        }
                    })
                    .collect();

                let sum_scores: f64 = scores.iter().sum::<f64>().max(1e-30);
                let probs: Vec<f64> = scores.iter().map(|s| s / sum_scores).collect();

                // Sample next step
                let r = rng.random::<f64>();
                let mut cumsum = 0.0;
                let mut chosen_idx = 0;
                for (i, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if r <= cumsum {
                        chosen_idx = i;
                        break;
                    }
                }

                let (_, next_entity) = neighbors[chosen_idx];
                path_score *= probs[chosen_idx];
                *entity_scores.entry(next_entity).or_insert(0.0) += path_score;
                current = next_entity;
            }
        }

        entity_scores
    }
}

/// Extended KG evaluation metrics: rule-based + path reasoning.
#[derive(Debug, Clone)]
pub struct KgMetricsExtended {
    /// Average rule confidence across mined rules.
    pub avg_rule_confidence: f64,
    /// Number of rules above the support/confidence threshold.
    pub n_rules: usize,
    /// Path reasoning accuracy: fraction of test triples where true tail
    /// is the highest-scoring entity from path reasoning.
    pub path_accuracy: f64,
    /// Entity coverage: fraction of entities reachable from any source in
    /// up to `max_hops` hops.
    pub entity_coverage: f64,
}

impl KgMetricsExtended {
    /// Compute extended metrics from mined rules and dataset statistics.
    pub fn from_rules(rules: &[HornRule], n_entities: usize, n_reachable: usize) -> Self {
        let avg_conf = if rules.is_empty() {
            0.0
        } else {
            rules.iter().map(|r| r.confidence).sum::<f64>() / rules.len() as f64
        };
        Self {
            avg_rule_confidence: avg_conf,
            n_rules: rules.len(),
            path_accuracy: 0.0,
            entity_coverage: if n_entities == 0 {
                0.0
            } else {
                n_reachable as f64 / n_entities as f64
            },
        }
    }

    /// Compute path reasoning accuracy on test triples.
    pub fn compute_path_accuracy(
        test_triples: &[KgTriple],
        entity_scores_per_triple: &[HashMap<usize, f64>],
    ) -> f64 {
        if test_triples.is_empty() {
            return 0.0;
        }
        let correct = test_triples
            .iter()
            .zip(entity_scores_per_triple.iter())
            .filter(|(&t, scores)| {
                let best = scores
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(&e, _)| e);
                best == Some(t.tail)
            })
            .count();
        correct as f64 / test_triples.len() as f64
    }
}
