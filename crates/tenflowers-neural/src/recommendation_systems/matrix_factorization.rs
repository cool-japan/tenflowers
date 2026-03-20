//! Matrix Factorization (ALS + SGD variants).

use scirs2_core::random::{rngs::StdRng, SeedableRng};

use super::{dot, solve_linear_system, xavier_fill, RecResult, RecSysError};

// ─────────────────────────────────────────────────────────────────────────────
// MfConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`MatrixFactorization`].
#[derive(Debug, Clone)]
pub struct MfConfig {
    /// Total number of users.
    pub n_users: usize,
    /// Total number of items.
    pub n_items: usize,
    /// Embedding dimension (latent factor count).
    pub emb_dim: usize,
    /// L2 regularisation coefficient.
    pub reg_lambda: f32,
    /// Whether to include per-user and per-item bias terms.
    pub use_bias: bool,
}

impl Default for MfConfig {
    fn default() -> Self {
        Self {
            n_users: 1000,
            n_items: 1000,
            emb_dim: 64,
            reg_lambda: 1e-4,
            use_bias: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MatrixFactorization
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix Factorization with both SGD and ALS training.
///
/// The rating model is:
/// `r̂(u, i) = μ + b_u + b_i + p_u · q_i`
///
/// where `p_u ∈ ℝ^d` is the user embedding, `q_i ∈ ℝ^d` is the item
/// embedding, and the biases are optional.
#[derive(Debug, Clone)]
pub struct MatrixFactorization {
    pub(crate) cfg: MfConfig,
    /// User embedding matrix, row-major `[n_users × emb_dim]`.
    pub(crate) user_emb: Vec<f32>,
    /// Item embedding matrix, row-major `[n_items × emb_dim]`.
    pub(crate) item_emb: Vec<f32>,
    /// Per-user bias (len = n_users).
    pub(crate) user_bias: Vec<f32>,
    /// Per-item bias (len = n_items).
    pub(crate) item_bias: Vec<f32>,
    /// Global mean (set externally or defaults to 0).
    pub global_mean: f32,
}

impl MatrixFactorization {
    /// Create a new randomly-initialised model.
    pub fn new(cfg: MfConfig, seed: u64) -> RecResult<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let d = cfg.emb_dim;
        let nu = cfg.n_users;
        let ni = cfg.n_items;
        let mut user_emb = vec![0.0_f32; nu * d];
        let mut item_emb = vec![0.0_f32; ni * d];
        xavier_fill(&mut user_emb, d, d, &mut rng);
        xavier_fill(&mut item_emb, d, d, &mut rng);
        Ok(Self {
            cfg,
            user_emb,
            item_emb,
            user_bias: vec![0.0_f32; nu],
            item_bias: vec![0.0_f32; ni],
            global_mean: 0.0,
        })
    }

    /// Return the user embedding slice for user `u`.
    #[inline]
    pub fn user_vec(&self, u: usize) -> &[f32] {
        let d = self.cfg.emb_dim;
        &self.user_emb[u * d..(u + 1) * d]
    }

    /// Return the item embedding slice for item `i`.
    #[inline]
    pub fn item_vec(&self, i: usize) -> &[f32] {
        let d = self.cfg.emb_dim;
        &self.item_emb[i * d..(i + 1) * d]
    }

    /// Predict the rating for (user `u`, item `i`).
    pub fn predict(&self, u: usize, i: usize) -> f32 {
        let pu = self.user_vec(u);
        let qi = self.item_vec(i);
        let dot_val = dot(pu, qi);
        let mut score = self.global_mean + dot_val;
        if self.cfg.use_bias {
            score += self.user_bias[u] + self.item_bias[i];
        }
        score
    }

    /// SGD update for a single observed (user, item, rating) triple.
    ///
    /// Returns the squared error before the update.
    pub fn train_step(&mut self, user_id: usize, item_id: usize, rating: f32, lr: f32) -> f32 {
        let d = self.cfg.emb_dim;
        let lam = self.cfg.reg_lambda;
        let pred = self.predict(user_id, item_id);
        let err = rating - pred;
        let sq_err = err * err;

        if self.cfg.use_bias {
            self.user_bias[user_id] += lr * (err - lam * self.user_bias[user_id]);
            self.item_bias[item_id] += lr * (err - lam * self.item_bias[item_id]);
        }

        let qi_snap: Vec<f32> = self.item_vec(item_id).to_vec();
        let pu_snap: Vec<f32> = self.user_vec(user_id).to_vec();

        let pu_start = user_id * d;
        for k in 0..d {
            let grad_u = err * qi_snap[k] - lam * self.user_emb[pu_start + k];
            self.user_emb[pu_start + k] += lr * grad_u;
        }
        let qi_start = item_id * d;
        for k in 0..d {
            let grad_i = err * pu_snap[k] - lam * self.item_emb[qi_start + k];
            self.item_emb[qi_start + k] += lr * grad_i;
        }
        sq_err
    }

    /// ALS update for a single user given a list of `(item_id, rating)` pairs.
    ///
    /// Solves the least-squares problem:
    /// `(Q^T Q + λI) p_u = Q^T r_u`
    /// using Cholesky-style Gaussian elimination (without external LAPACK).
    pub fn als_update_user(
        &mut self,
        user_id: usize,
        observations: &[(usize, f32)],
    ) -> RecResult<()> {
        let d = self.cfg.emb_dim;
        let lam = self.cfg.reg_lambda;
        let mut a = vec![0.0_f32; d * d];
        let mut b = vec![0.0_f32; d];
        for &(item_id, rating) in observations {
            let qi = self.item_vec(item_id);
            let r_adj = rating
                - self.global_mean
                - if self.cfg.use_bias {
                    self.item_bias[item_id]
                } else {
                    0.0
                };
            for row in 0..d {
                b[row] += qi[row] * r_adj;
                for col in 0..d {
                    a[row * d + col] += qi[row] * qi[col];
                }
            }
        }
        for k in 0..d {
            a[k * d + k] += lam;
        }
        let p = solve_linear_system(&mut a, &mut b, d)?;
        let start = user_id * d;
        self.user_emb[start..(d + start)].copy_from_slice(&p[..d]);
        Ok(())
    }

    /// ALS update for a single item given a list of `(user_id, rating)` pairs.
    pub fn als_update_item(
        &mut self,
        item_id: usize,
        observations: &[(usize, f32)],
    ) -> RecResult<()> {
        let d = self.cfg.emb_dim;
        let lam = self.cfg.reg_lambda;
        let mut a = vec![0.0_f32; d * d];
        let mut b = vec![0.0_f32; d];
        for &(user_id, rating) in observations {
            let pu = self.user_vec(user_id);
            let r_adj = rating
                - self.global_mean
                - if self.cfg.use_bias {
                    self.user_bias[user_id]
                } else {
                    0.0
                };
            for row in 0..d {
                b[row] += pu[row] * r_adj;
                for col in 0..d {
                    a[row * d + col] += pu[row] * pu[col];
                }
            }
        }
        for k in 0..d {
            a[k * d + k] += lam;
        }
        let q = solve_linear_system(&mut a, &mut b, d)?;
        let start = item_id * d;
        self.item_emb[start..(d + start)].copy_from_slice(&q[..d]);
        Ok(())
    }

    /// Score all items for a given user; returns `(item_id, score)` sorted descending.
    pub fn rank_items_for_user(&self, user_id: usize) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = (0..self.cfg.n_items)
            .map(|i| (i, self.predict(user_id, i)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }
}
