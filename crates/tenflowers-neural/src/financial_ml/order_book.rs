//! Financial deep learning: order book encoders, alpha factor nets, portfolio optimisation.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Common helpers (shared across all financial_ml submodules via pub(super))
// ─────────────────────────────────────────────────────────────────────────────

pub(super) type FinResult<T> = Result<T, String>;

#[inline]
pub(super) fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
pub(super) fn sigmoid_f32(x: f32) -> f32 {
    let c = x.clamp(-88.0, 88.0);
    1.0 / (1.0 + (-c).exp())
}

#[inline]
pub(super) fn tanh_f32(x: f32) -> f32 {
    x.tanh()
}

pub(super) fn softmax_vec(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        return Vec::new();
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / v.len() as f32; v.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

pub(super) fn layer_norm_vec(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    x.iter().map(|&v| (v - mean) / (var + eps).sqrt()).collect()
}

/// Simple linear layer: y = W x + b
#[derive(Debug, Clone)]
pub(super) struct Linear {
    pub(super) w: Vec<f32>, // [out x in]
    pub(super) b: Vec<f32>, // [out]
    pub(super) in_dim: usize,
    pub(super) out_dim: usize,
}

impl Linear {
    pub(super) fn new(in_dim: usize, out_dim: usize, rng: &mut StdRng) -> Self {
        let scale = (2.0_f32 / in_dim as f32).sqrt();
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        let b = vec![0.0_f32; out_dim];
        Self {
            w,
            b,
            in_dim,
            out_dim,
        }
    }

    pub(super) fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = self.b.clone();
        for o in 0..self.out_dim {
            for i in 0..self.in_dim {
                out[o] += self.w[o * self.in_dim + i] * x[i];
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Order Book
// ─────────────────────────────────────────────────────────────────────────────

/// Encodes a Level-2 order book snapshot into feature vectors.
///
/// Features: mid-price, spread, volume imbalance, weighted mid-price.
#[derive(Debug, Clone)]
pub struct OrderBookEncoder {
    /// Number of price levels on each side (bid / ask).
    pub n_levels: usize,
}

impl OrderBookEncoder {
    pub fn new(n_levels: usize) -> Self {
        Self { n_levels }
    }

    /// Encode one snapshot.
    ///
    /// * `bid_prices`  – best bid at index 0, deepest at index n_levels-1
    /// * `bid_vols`    – corresponding bid volumes
    /// * `ask_prices`  – best ask at index 0
    /// * `ask_vols`    – corresponding ask volumes
    ///
    /// Returns `[mid_price, spread, volume_imbalance, weighted_mid]`.
    pub fn encode(
        &self,
        bid_prices: &[f32],
        bid_vols: &[f32],
        ask_prices: &[f32],
        ask_vols: &[f32],
    ) -> FinResult<Vec<f32>> {
        let n = self.n_levels;
        if bid_prices.len() < n || bid_vols.len() < n || ask_prices.len() < n || ask_vols.len() < n
        {
            return Err("OrderBookEncoder: slice length < n_levels".to_string());
        }
        let best_bid = bid_prices[0];
        let best_ask = ask_prices[0];
        let mid = (best_bid + best_ask) / 2.0;
        let spread = best_ask - best_bid;

        let total_bid_vol: f32 = bid_vols[..n].iter().sum();
        let total_ask_vol: f32 = ask_vols[..n].iter().sum();
        let total_vol = total_bid_vol + total_ask_vol;
        let vol_imbalance = if total_vol > 0.0 {
            (total_bid_vol - total_ask_vol) / total_vol
        } else {
            0.0
        };

        // Volume-weighted mid price using best level only
        let wm = if total_bid_vol + total_ask_vol > 0.0 {
            (best_bid * total_ask_vol + best_ask * total_bid_vol) / (total_bid_vol + total_ask_vol)
        } else {
            mid
        };

        Ok(vec![mid, spread, vol_imbalance, wm])
    }

    /// Encode multiple snapshots into a feature matrix (n_snapshots × 4).
    pub fn encode_batch(
        &self,
        snapshots: &[(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)],
    ) -> FinResult<Vec<Vec<f32>>> {
        snapshots
            .iter()
            .map(|(bp, bv, ap, av)| self.encode(bp, bv, ap, av))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Limit Order Book Benchmark encoder: CNN on order book snapshots → latent.
#[derive(Debug, Clone)]
pub struct LobbEncoder {
    conv1: Linear,
    conv2: Linear,
    pub latent_dim: usize,
}

impl LobbEncoder {
    pub fn new(n_levels: usize, latent_dim: usize, seed: u64) -> FinResult<Self> {
        if n_levels == 0 || latent_dim == 0 {
            return Err("LobbEncoder: n_levels and latent_dim must be > 0".to_string());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let input_dim = n_levels * 4; // bid_p, bid_v, ask_p, ask_v per level
        let hidden = latent_dim * 2;
        Ok(Self {
            conv1: Linear::new(input_dim, hidden, &mut rng),
            conv2: Linear::new(hidden, latent_dim, &mut rng),
            latent_dim,
        })
    }

    /// Forward: flatten snapshot → conv1 (relu) → conv2 → latent.
    pub fn forward(&self, snapshot: &[f32]) -> FinResult<Vec<f32>> {
        if snapshot.len() < self.conv1.in_dim {
            return Err(format!(
                "LobbEncoder: expected {} inputs, got {}",
                self.conv1.in_dim,
                snapshot.len()
            ));
        }
        let h: Vec<f32> = self.conv1.forward(snapshot).into_iter().map(relu).collect();
        Ok(self.conv2.forward(&h))
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Simple LSTM cell for 1-step forward pass (stateful).
#[derive(Debug, Clone)]
pub(super) struct LstmCell {
    // Gates: input, forget, cell, output  (each out_dim × in_dim)
    pub(super) wi: Vec<f32>,
    pub(super) wf: Vec<f32>,
    pub(super) wg: Vec<f32>,
    pub(super) wo: Vec<f32>,
    pub(super) ui: Vec<f32>,
    pub(super) uf: Vec<f32>,
    pub(super) ug: Vec<f32>,
    pub(super) uo: Vec<f32>,
    pub(super) bi: Vec<f32>,
    pub(super) bf: Vec<f32>,
    pub(super) bg: Vec<f32>,
    pub(super) bo: Vec<f32>,
    pub(super) in_dim: usize,
    pub(super) out_dim: usize,
}

impl LstmCell {
    pub(super) fn new(in_dim: usize, out_dim: usize, rng: &mut StdRng) -> Self {
        let scale = (1.0_f32 / (in_dim + out_dim) as f32).sqrt();
        let mut rand_mat = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                .collect()
        };
        Self {
            wi: rand_mat(out_dim * in_dim),
            wf: rand_mat(out_dim * in_dim),
            wg: rand_mat(out_dim * in_dim),
            wo: rand_mat(out_dim * in_dim),
            ui: rand_mat(out_dim * out_dim),
            uf: rand_mat(out_dim * out_dim),
            ug: rand_mat(out_dim * out_dim),
            uo: rand_mat(out_dim * out_dim),
            bi: vec![0.0; out_dim],
            bf: vec![1.0; out_dim], // forget bias = 1
            bg: vec![0.0; out_dim],
            bo: vec![0.0; out_dim],
            in_dim,
            out_dim,
        }
    }

    pub(super) fn step(&self, x: &[f32], h: &[f32], c: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let d = self.out_dim;
        let mut i_gate = self.bi.clone();
        let mut f_gate = self.bf.clone();
        let mut g_gate = self.bg.clone();
        let mut o_gate = self.bo.clone();

        for o in 0..d {
            for j in 0..self.in_dim {
                i_gate[o] += self.wi[o * self.in_dim + j] * x[j];
                f_gate[o] += self.wf[o * self.in_dim + j] * x[j];
                g_gate[o] += self.wg[o * self.in_dim + j] * x[j];
                o_gate[o] += self.wo[o * self.in_dim + j] * x[j];
            }
            for j in 0..d {
                i_gate[o] += self.ui[o * d + j] * h[j];
                f_gate[o] += self.uf[o * d + j] * h[j];
                g_gate[o] += self.ug[o * d + j] * h[j];
                o_gate[o] += self.uo[o * d + j] * h[j];
            }
        }
        let i: Vec<f32> = i_gate.iter().map(|&v| sigmoid_f32(v)).collect();
        let f: Vec<f32> = f_gate.iter().map(|&v| sigmoid_f32(v)).collect();
        let g: Vec<f32> = g_gate.iter().map(|&v| tanh_f32(v)).collect();
        let o: Vec<f32> = o_gate.iter().map(|&v| sigmoid_f32(v)).collect();
        let new_c: Vec<f32> = (0..d).map(|j| f[j] * c[j] + i[j] * g[j]).collect();
        let new_h: Vec<f32> = (0..d).map(|j| o[j] * tanh_f32(new_c[j])).collect();
        (new_h, new_c)
    }
}

/// Deep LOB: CNN feature extractor + LSTM → 3-class price movement prediction.
///
/// Classes: 0 = down, 1 = neutral, 2 = up.
#[derive(Debug, Clone)]
pub struct DeepLobModel {
    cnn1: Linear,
    cnn2: Linear,
    lstm: LstmCell,
    classifier: Linear,
    hidden_dim: usize,
}

impl DeepLobModel {
    pub fn new(
        input_dim: usize,
        cnn_hidden: usize,
        lstm_hidden: usize,
        seed: u64,
    ) -> FinResult<Self> {
        if input_dim == 0 || cnn_hidden == 0 || lstm_hidden == 0 {
            return Err("DeepLobModel: dimensions must be > 0".to_string());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            cnn1: Linear::new(input_dim, cnn_hidden, &mut rng),
            cnn2: Linear::new(cnn_hidden, cnn_hidden, &mut rng),
            lstm: LstmCell::new(cnn_hidden, lstm_hidden, &mut rng),
            classifier: Linear::new(lstm_hidden, 3, &mut rng),
            hidden_dim: lstm_hidden,
        })
    }

    /// Forward over a sequence of order book feature vectors.
    ///
    /// Returns softmax probabilities for [down, neutral, up].
    pub fn forward(&self, sequence: &[Vec<f32>]) -> FinResult<Vec<f32>> {
        if sequence.is_empty() {
            return Err("DeepLobModel: empty sequence".to_string());
        }
        let mut h = vec![0.0_f32; self.hidden_dim];
        let mut c = vec![0.0_f32; self.hidden_dim];

        for snap in sequence {
            let c1: Vec<f32> = self.cnn1.forward(snap).into_iter().map(relu).collect();
            let c2: Vec<f32> = self.cnn2.forward(&c1).into_iter().map(relu).collect();
            let (nh, nc) = self.lstm.step(&c2, &h, &c);
            h = nh;
            c = nc;
        }
        let logits = self.classifier.forward(&h);
        Ok(softmax_vec(&logits))
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Learns alpha factors from raw price/volume data via a convolutional
/// feature extractor followed by a per-asset scoring head.
#[derive(Debug, Clone)]
pub struct AlphaFactorNet {
    feat1: Linear,
    feat2: Linear,
    score: Linear,
    pub output_dim: usize,
}

impl AlphaFactorNet {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        n_alpha_factors: usize,
        seed: u64,
    ) -> FinResult<Self> {
        if input_dim == 0 || hidden_dim == 0 || n_alpha_factors == 0 {
            return Err("AlphaFactorNet: dimensions must be > 0".to_string());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            feat1: Linear::new(input_dim, hidden_dim, &mut rng),
            feat2: Linear::new(hidden_dim, hidden_dim, &mut rng),
            score: Linear::new(hidden_dim, n_alpha_factors, &mut rng),
            output_dim: n_alpha_factors,
        })
    }

    /// Forward: raw features → alpha scores.
    pub fn forward(&self, x: &[f32]) -> FinResult<Vec<f32>> {
        if x.len() < self.feat1.in_dim {
            return Err("AlphaFactorNet: input too short".to_string());
        }
        let h1: Vec<f32> = self.feat1.forward(x).into_iter().map(relu).collect();
        let h2: Vec<f32> = self.feat2.forward(&h1).into_iter().map(relu).collect();
        Ok(self.score.forward(&h2))
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Mean-variance (Markowitz) portfolio optimisation.
///
/// Solves `argmin_w  ½ λ wᵀΣw − μᵀw`  subject to `Σw_i = 1`, `w_i ≥ 0`
/// using projected gradient descent.
#[derive(Debug, Clone)]
pub struct PortfolioOptimizer {
    pub n_assets: usize,
    pub max_iter: usize,
    pub lr: f64,
}

impl PortfolioOptimizer {
    pub fn new(n_assets: usize) -> Self {
        Self {
            n_assets,
            max_iter: 500,
            lr: 0.01,
        }
    }

    /// Markowitz mean-variance optimisation.
    ///
    /// * `mu`             – expected returns (length n_assets)
    /// * `sigma`          – covariance matrix (n_assets × n_assets, row-major)
    /// * `risk_aversion`  – λ (larger = more risk-averse)
    ///
    /// Returns portfolio weights that sum to 1, all ≥ 0.
    pub fn markowitz(&self, mu: &[f64], sigma: &[f64], risk_aversion: f64) -> FinResult<Vec<f64>> {
        let n = self.n_assets;
        if mu.len() != n || sigma.len() != n * n {
            return Err("PortfolioOptimizer: dimension mismatch".to_string());
        }
        let lam = risk_aversion.max(1e-8);
        // Initialise with equal weights.
        let mut w: Vec<f64> = vec![1.0 / n as f64; n];

        for _ in 0..self.max_iter {
            // Gradient: λ Σ w − μ
            let mut grad = vec![0.0_f64; n];
            for i in 0..n {
                for j in 0..n {
                    grad[i] += lam * sigma[i * n + j] * w[j];
                }
                grad[i] -= mu[i];
            }
            // Gradient step
            for i in 0..n {
                w[i] -= self.lr * grad[i];
            }
            // Project onto simplex (non-negative, sum = 1)
            w = project_simplex(&w);
        }
        Ok(w)
    }

    /// Black-Litterman blending of market equilibrium returns with investor views.
    ///
    /// * `pi`   – equilibrium returns (market-implied)
    /// * `q`    – view vector
    /// * `p`    – pick matrix (n_views × n_assets)
    /// * `tau`  – uncertainty scalar
    /// * `omega_diag` – view uncertainty diagonal (length n_views)
    pub fn black_litterman(
        &self,
        pi: &[f64],
        q: &[f64],
        p: &[f64],
        tau: f64,
        sigma: &[f64],
        omega_diag: &[f64],
    ) -> FinResult<Vec<f64>> {
        let n = self.n_assets;
        let k = q.len();
        if pi.len() != n || p.len() != k * n || omega_diag.len() != k || sigma.len() != n * n {
            return Err("BlackLitterman: dimension mismatch".to_string());
        }
        // BL posterior mean: μ_BL = [(τΣ)^{-1} + Pᵀ Ω^{-1} P]^{-1} [(τΣ)^{-1}π + Pᵀ Ω^{-1} q]
        // Simplified: use iterative blending with small tau
        let t = tau.max(1e-8);
        // Compute Pᵀ Ω^{-1} (q − P π)
        let mut mu_bl = pi.to_vec();
        for kk in 0..k {
            // view error
            let mut ppi = 0.0_f64;
            for j in 0..n {
                ppi += p[kk * n + j] * pi[j];
            }
            let err = q[kk] - ppi;
            let omega_inv = 1.0 / omega_diag[kk].max(1e-12);
            // Add contribution to posterior mean
            for j in 0..n {
                let mut sp = 0.0_f64;
                for i in 0..n {
                    sp += t * sigma[j * n + i] * p[kk * n + i];
                }
                mu_bl[j] += sp * omega_inv * err;
            }
        }
        Ok(mu_bl)
    }
}

/// Project vector onto probability simplex (Duchi et al., 2008).
pub(super) fn project_simplex(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut cssv = 0.0_f64;
    let mut rho = 0usize;
    for j in 0..n {
        cssv += u[j];
        if u[j] - (cssv - 1.0) / (j as f64 + 1.0) > 0.0 {
            rho = j;
        }
    }
    let cssv2: f64 = u[..=rho].iter().sum();
    let theta = (cssv2 - 1.0) / (rho as f64 + 1.0);
    v.iter().map(|&x| (x - theta).max(0.0)).collect()
}
