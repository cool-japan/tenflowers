//! Graph-based collaborative filtering: LightGCN.

use scirs2_core::random::{rngs::StdRng, SeedableRng};

use super::{dot, sigmoid_f32, xavier_fill, RecResult, RecSysError};

// ─────────────────────────────────────────────────────────────────────────────
// LightGcnConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`LightGCN`].
#[derive(Debug, Clone)]
pub struct LightGcnConfig {
    /// Number of users.
    pub n_users: usize,
    /// Number of items.
    pub n_items: usize,
    /// Embedding dimension.
    pub emb_dim: usize,
    /// L2 regularisation coefficient.
    pub reg_lambda: f32,
}

impl Default for LightGcnConfig {
    fn default() -> Self {
        Self {
            n_users: 500,
            n_items: 500,
            emb_dim: 64,
            reg_lambda: 1e-4,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LightGCN
// ─────────────────────────────────────────────────────────────────────────────

/// LightGCN (He et al., 2020) — simplifies NGCF by removing feature
/// transformation and non-linear activation.
///
/// Propagation rule:
/// `e^(k+1)_u = Σ_{i ∈ N_u} (1 / √|N_u| √|N_i|) e^k_i`
///
/// The final embedding is the mean over all layers.
#[derive(Debug, Clone)]
pub struct LightGCN {
    cfg: LightGcnConfig,
    /// User embeddings (layer 0) `[n_users × emb_dim]`.
    pub user_emb: Vec<f32>,
    /// Item embeddings (layer 0) `[n_items × emb_dim]`.
    pub item_emb: Vec<f32>,
}

impl LightGCN {
    /// Create a randomly-initialised LightGCN model.
    pub fn new(cfg: LightGcnConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let d = cfg.emb_dim;
        let mut ue = vec![0.0_f32; cfg.n_users * d];
        let mut ie = vec![0.0_f32; cfg.n_items * d];
        xavier_fill(&mut ue, d, d, &mut rng);
        xavier_fill(&mut ie, d, d, &mut rng);
        Self {
            cfg,
            user_emb: ue,
            item_emb: ie,
        }
    }

    /// Run `n_layers` rounds of symmetric normalised graph propagation.
    ///
    /// `user_items` is the observed interaction graph as (user_id, item_id) pairs.
    ///
    /// Returns `(final_user_emb, final_item_emb)` where each embedding is the
    /// mean of layer-0 through layer-K representations.
    pub fn propagate(
        &self,
        user_items: &[(usize, usize)],
        n_layers: usize,
    ) -> RecResult<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let nu = self.cfg.n_users;
        let ni = self.cfg.n_items;
        let d = self.cfg.emb_dim;

        // Compute node degrees
        let mut user_deg = vec![0usize; nu];
        let mut item_deg = vec![0usize; ni];
        for &(u, i) in user_items {
            if u >= nu || i >= ni {
                return Err(RecSysError::IndexOutOfBounds {
                    what: "user or item",
                    idx: u.max(i),
                    max: nu.max(ni),
                });
            }
            user_deg[u] += 1;
            item_deg[i] += 1;
        }

        // Running sum accumulators (layer-0 contribution already in)
        let mut sum_user: Vec<f32> = self.user_emb.clone();
        let mut sum_item: Vec<f32> = self.item_emb.clone();

        let mut cur_user: Vec<f32> = self.user_emb.clone();
        let mut cur_item: Vec<f32> = self.item_emb.clone();

        for _layer in 0..n_layers {
            let mut next_user = vec![0.0_f32; nu * d];
            let mut next_item = vec![0.0_f32; ni * d];

            for &(u, i) in user_items {
                let coeff =
                    1.0 / ((user_deg[u] as f32).sqrt() * (item_deg[i] as f32).sqrt() + 1e-8);

                let i_emb = &cur_item[i * d..(i + 1) * d];
                let u_emb = &cur_user[u * d..(u + 1) * d];
                for k in 0..d {
                    next_user[u * d + k] += coeff * i_emb[k];
                    next_item[i * d + k] += coeff * u_emb[k];
                }
            }

            for k in 0..nu * d {
                sum_user[k] += next_user[k];
            }
            for k in 0..ni * d {
                sum_item[k] += next_item[k];
            }

            cur_user = next_user;
            cur_item = next_item;
        }

        // Average over n_layers + 1 contributions (layers 0..=K)
        let scale = 1.0 / (n_layers + 1) as f32;
        let final_user: Vec<Vec<f32>> = (0..nu)
            .map(|u| (0..d).map(|k| sum_user[u * d + k] * scale).collect())
            .collect();
        let final_item: Vec<Vec<f32>> = (0..ni)
            .map(|i| (0..d).map(|k| sum_item[i * d + k] * scale).collect())
            .collect();

        Ok((final_user, final_item))
    }

    /// Predict a rating for (user, item) using the propagated embeddings.
    pub fn predict_with_propagated(&self, user_emb_row: &[f32], item_emb_row: &[f32]) -> f32 {
        dot(user_emb_row, item_emb_row)
    }

    /// SGD update for a BPR triplet (user, positive item, negative item).
    pub fn bpr_update(
        &mut self,
        user_id: usize,
        pos_item: usize,
        neg_item: usize,
        lr: f32,
    ) -> RecResult<f32> {
        let d = self.cfg.emb_dim;
        let nu = self.cfg.n_users;
        let ni = self.cfg.n_items;
        if user_id >= nu {
            return Err(RecSysError::IndexOutOfBounds {
                what: "user",
                idx: user_id,
                max: nu,
            });
        }
        if pos_item >= ni || neg_item >= ni {
            return Err(RecSysError::IndexOutOfBounds {
                what: "item",
                idx: pos_item.max(neg_item),
                max: ni,
            });
        }

        let pu: Vec<f32> = self.user_emb[user_id * d..(user_id + 1) * d].to_vec();
        let qi_pos: Vec<f32> = self.item_emb[pos_item * d..(pos_item + 1) * d].to_vec();
        let qi_neg: Vec<f32> = self.item_emb[neg_item * d..(neg_item + 1) * d].to_vec();

        let s_pos = dot(&pu, &qi_pos);
        let s_neg = dot(&pu, &qi_neg);
        let diff = (s_pos - s_neg).clamp(-30.0, 30.0);
        let sig = sigmoid_f32(-diff); // gradient coefficient = σ(s_neg - s_pos)
        let lam = self.cfg.reg_lambda;

        for k in 0..d {
            self.user_emb[user_id * d + k] += lr * (sig * (qi_pos[k] - qi_neg[k]) - lam * pu[k]);
            self.item_emb[pos_item * d + k] += lr * (sig * pu[k] - lam * qi_pos[k]);
            self.item_emb[neg_item * d + k] += lr * (-sig * pu[k] - lam * qi_neg[k]);
        }

        Ok(-(sigmoid_f32(diff)).ln())
    }
}
