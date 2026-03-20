//! Neural Collaborative Filtering (GMF + MLP).

use scirs2_core::random::{rngs::StdRng, SeedableRng};

use super::{he_fill, matvec, relu_f32, xavier_fill, RecResult, RecSysError};

// ─────────────────────────────────────────────────────────────────────────────
// BprLoss
// ─────────────────────────────────────────────────────────────────────────────

/// Bayesian Personalized Ranking loss (Rendle et al., 2009).
///
/// Maximises the probability that the positive item is ranked above the
/// negative item: `L = -log(σ(s+ − s−)) + λ · ‖θ‖²`.
#[derive(Debug, Clone)]
pub struct BprLoss {
    /// L2 regularisation strength applied to raw scores.
    pub reg_lambda: f32,
}

impl BprLoss {
    /// Create a new BPR loss with the given regularisation weight.
    pub fn new(reg_lambda: f32) -> Self {
        Self { reg_lambda }
    }

    /// Compute BPR loss for a batch of (positive score, negative score) pairs.
    ///
    /// `L = mean[-log(σ(s_pos - s_neg))]` plus optional L2 term.
    pub fn bpr_loss(&self, pos_scores: &[f32], neg_scores: &[f32]) -> RecResult<f32> {
        if pos_scores.len() != neg_scores.len() {
            return Err(RecSysError::DimensionMismatch {
                expected: pos_scores.len(),
                got: neg_scores.len(),
            });
        }
        if pos_scores.is_empty() {
            return Err(RecSysError::EmptySequence);
        }
        let n = pos_scores.len() as f32;
        let loss: f32 = pos_scores
            .iter()
            .zip(neg_scores.iter())
            .map(|(s_p, s_n)| {
                let diff = (s_p - s_n).clamp(-30.0, 30.0);
                -(super::sigmoid_f32(diff)).ln()
            })
            .sum::<f32>()
            / n;
        Ok(loss)
    }

    /// Convenience scalar version: one (pos, neg) score pair.
    #[inline]
    pub fn bpr_loss_scalar(&self, pos_score: f32, neg_score: f32) -> f32 {
        let diff = (pos_score - neg_score).clamp(-30.0, 30.0);
        -(super::sigmoid_f32(diff)).ln()
    }

    /// Gradient of BPR loss w.r.t. `s_pos - s_neg` for a scalar pair.
    #[inline]
    pub fn bpr_grad(&self, pos_score: f32, neg_score: f32) -> f32 {
        super::sigmoid_f32(neg_score - pos_score) - 1.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NcfConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`NeuralCF`].
#[derive(Debug, Clone)]
pub struct NcfConfig {
    /// Total number of users.
    pub n_users: usize,
    /// Total number of items.
    pub n_items: usize,
    /// Embedding dimension for the GMF branch.
    pub gmf_dim: usize,
    /// Embedding dimension for the MLP branch (input to first MLP layer).
    pub mlp_emb_dim: usize,
    /// Hidden layer sizes for the MLP branch (e.g. `[128, 64, 32]`).
    pub mlp_layers: Vec<usize>,
    /// Output dimension (1 = rating / binary click prediction).
    pub output_dim: usize,
}

impl Default for NcfConfig {
    fn default() -> Self {
        Self {
            n_users: 1000,
            n_items: 1000,
            gmf_dim: 32,
            mlp_emb_dim: 32,
            mlp_layers: vec![64, 32, 16],
            output_dim: 1,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NeuralCF
// ─────────────────────────────────────────────────────────────────────────────

/// Neural Collaborative Filtering (He et al., 2017).
///
/// Fuses a Generalised Matrix Factorisation (GMF) path (element-wise product
/// of embeddings) with a Multi-Layer Perceptron (MLP) path (concatenated
/// embeddings passed through a deep network). The final logit is a linear
/// combination of both paths.
#[derive(Debug, Clone)]
pub struct NeuralCF {
    cfg: NcfConfig,
    // GMF embeddings
    gmf_user_emb: Vec<f32>,
    gmf_item_emb: Vec<f32>,
    // MLP embeddings
    mlp_user_emb: Vec<f32>,
    mlp_item_emb: Vec<f32>,
    // MLP layers: weights row-major, biases; layers[k] = (W, b, in, out)
    pub(crate) mlp_weights: Vec<Vec<f32>>,
    mlp_biases: Vec<Vec<f32>>,
    // Final prediction layer
    out_weight: Vec<f32>, // [output_dim × (gmf_dim + last_mlp_dim)]
    out_bias: Vec<f32>,
}

impl NeuralCF {
    /// Create and randomly initialise a NeuralCF model.
    pub fn new(cfg: NcfConfig, seed: u64) -> RecResult<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let gmf_d = cfg.gmf_dim;
        let mlp_d = cfg.mlp_emb_dim;
        let nu = cfg.n_users;
        let ni = cfg.n_items;

        let mut gmf_ue = vec![0.0_f32; nu * gmf_d];
        let mut gmf_ie = vec![0.0_f32; ni * gmf_d];
        xavier_fill(&mut gmf_ue, gmf_d, gmf_d, &mut rng);
        xavier_fill(&mut gmf_ie, gmf_d, gmf_d, &mut rng);

        let mut mlp_ue = vec![0.0_f32; nu * mlp_d];
        let mut mlp_ie = vec![0.0_f32; ni * mlp_d];
        xavier_fill(&mut mlp_ue, mlp_d, mlp_d, &mut rng);
        xavier_fill(&mut mlp_ie, mlp_d, mlp_d, &mut rng);

        let mut mlp_weights = Vec::new();
        let mut mlp_biases = Vec::new();
        let mut in_dim = mlp_d * 2;
        for &out_dim in &cfg.mlp_layers {
            let mut w = vec![0.0_f32; out_dim * in_dim];
            he_fill(&mut w, in_dim, &mut rng);
            mlp_weights.push(w);
            mlp_biases.push(vec![0.0_f32; out_dim]);
            in_dim = out_dim;
        }

        let last_mlp = cfg.mlp_layers.last().copied().unwrap_or(mlp_d * 2);
        let out_in = gmf_d + last_mlp;
        let out_d = cfg.output_dim;
        let mut ow = vec![0.0_f32; out_d * out_in];
        xavier_fill(&mut ow, out_in, out_d, &mut rng);

        Ok(Self {
            cfg,
            gmf_user_emb: gmf_ue,
            gmf_item_emb: gmf_ie,
            mlp_user_emb: mlp_ue,
            mlp_item_emb: mlp_ie,
            mlp_weights,
            mlp_biases,
            out_weight: ow,
            out_bias: vec![0.0_f32; out_d],
        })
    }

    /// Forward pass: predict score for (user_id, item_id).
    ///
    /// Returns a vector of length `cfg.output_dim` (typically 1).
    pub fn forward(&self, user_id: usize, item_id: usize) -> RecResult<Vec<f32>> {
        let nu = self.cfg.n_users;
        let ni = self.cfg.n_items;
        if user_id >= nu {
            return Err(RecSysError::IndexOutOfBounds {
                what: "user",
                idx: user_id,
                max: nu,
            });
        }
        if item_id >= ni {
            return Err(RecSysError::IndexOutOfBounds {
                what: "item",
                idx: item_id,
                max: ni,
            });
        }

        // GMF branch: element-wise product
        let gmf_d = self.cfg.gmf_dim;
        let pu_gmf = &self.gmf_user_emb[user_id * gmf_d..(user_id + 1) * gmf_d];
        let qi_gmf = &self.gmf_item_emb[item_id * gmf_d..(item_id + 1) * gmf_d];
        let gmf_out: Vec<f32> = pu_gmf
            .iter()
            .zip(qi_gmf.iter())
            .map(|(a, b)| a * b)
            .collect();

        // MLP branch: concat embeddings, forward through MLP
        let mlp_d = self.cfg.mlp_emb_dim;
        let pu_mlp = &self.mlp_user_emb[user_id * mlp_d..(user_id + 1) * mlp_d];
        let qi_mlp = &self.mlp_item_emb[item_id * mlp_d..(item_id + 1) * mlp_d];
        let mut h: Vec<f32> = pu_mlp.iter().chain(qi_mlp.iter()).cloned().collect();
        let mut in_dim = mlp_d * 2;
        for (k, w) in self.mlp_weights.iter().enumerate() {
            let out_dim = self.mlp_biases[k].len();
            let mut next = matvec(w, &h, out_dim, in_dim);
            for (j, b) in self.mlp_biases[k].iter().enumerate() {
                next[j] += b;
                next[j] = relu_f32(next[j]);
            }
            in_dim = out_dim;
            h = next;
        }

        // Concat GMF and MLP outputs → final prediction
        let concat: Vec<f32> = gmf_out.iter().chain(h.iter()).cloned().collect();
        let out_d = self.cfg.output_dim;
        let out_in = concat.len();
        let mut logit = matvec(&self.out_weight, &concat, out_d, out_in);
        for (j, b) in self.out_bias.iter().enumerate() {
            logit[j] += b;
        }
        Ok(logit)
    }
}
