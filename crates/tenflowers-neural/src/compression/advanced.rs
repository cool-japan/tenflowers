//! Compression — Advanced: learned image compression, advanced NN compression, and knowledge compression.
//!
//! ## Components
//!
//! - **HyperpriorModel** — Balle 2018 hyperprior-based learned image compression
//! - **ChannelConditionalModel** — channel-conditional Gaussian entropy model
//! - **RdOptimizer** — Lagrangian R-D optimization
//! - **MagnitudePruner** — one-shot magnitude-based weight pruning
//! - **MovementPrunerV2** — movement-based pruning (Sanh 2020)
//! - **MixedPrecisionSearch** — sensitivity analysis + evolutionary bit-width search
//! - **LayerDropper** — structured layer dropping with performance-aware selection
//! - **VocabPruner** — vocabulary embedding pruning by frequency + importance
//! - **CompMetrics** — compression evaluation metrics

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};
use std::cmp::Ordering;

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

fn relu_f64(x: f64) -> f64 {
    x.max(0.0)
}

fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus_f64(x: f64) -> f64 {
    (1.0 + x.exp()).ln()
}

fn linear_f64(x: &[f64], w: &[f64], b: &[f64], in_d: usize, out_d: usize) -> Vec<f64> {
    (0..out_d)
        .map(|o| {
            w[o * in_d..(o + 1) * in_d]
                .iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f64>()
                + b[o]
        })
        .collect()
}

fn rand_normal_f64(rng: &mut StdRng, std_dev: f64) -> f64 {
    let u1: f64 = (rng.random::<f64>()).max(1e-15);
    let u2: f64 = rng.random::<f64>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos() * std_dev
}

fn rand_vec_f64(n: usize, std_dev: f64, rng: &mut StdRng) -> Vec<f64> {
    (0..n).map(|_| rand_normal_f64(rng, std_dev)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Learned Image Compression
// ─────────────────────────────────────────────────────────────────────────────

/// Hyperprior-based learned image compression (Balle et al. 2018).
///
/// Architecture: Encoder → Quantizer → Hyperencoder → Hyperprior → Rate estimate.
/// Lagrangian optimization: total_loss = λ·D + R where R is estimated entropy.
pub struct HyperpriorModel {
    /// Number of latent channels.
    pub n_latent: usize,
    /// Number of hyperlatent channels.
    pub n_hyper: usize,
    /// R-D trade-off weight (lambda).
    pub lambda: f64,
    // Encoder: input_dim → n_latent
    enc_w: Vec<f64>,
    enc_b: Vec<f64>,
    // Hyperencoder: n_latent → n_hyper
    hyper_enc_w: Vec<f64>,
    hyper_enc_b: Vec<f64>,
    // Hyperdecoder: n_hyper → n_latent*2 (mean + log_scale)
    hyper_dec_w: Vec<f64>,
    hyper_dec_b: Vec<f64>,
    // Decoder: n_latent → output_dim
    dec_w: Vec<f64>,
    dec_b: Vec<f64>,
    /// Input/output dimension.
    pub input_dim: usize,
}

impl HyperpriorModel {
    /// Create a new HyperpriorModel.
    pub fn new(input_dim: usize, n_latent: usize, n_hyper: usize, lambda: f64, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let std_enc = (2.0 / input_dim as f64).sqrt();
        let std_hyp = (2.0 / n_latent as f64).sqrt();
        let std_dec = (2.0 / n_hyper as f64).sqrt();
        Self {
            n_latent,
            n_hyper,
            lambda,
            input_dim,
            enc_w: rand_vec_f64(input_dim * n_latent, std_enc, &mut rng),
            enc_b: vec![0.0; n_latent],
            hyper_enc_w: rand_vec_f64(n_latent * n_hyper, std_hyp, &mut rng),
            hyper_enc_b: vec![0.0; n_hyper],
            hyper_dec_w: rand_vec_f64(n_hyper * n_latent * 2, std_dec, &mut rng),
            hyper_dec_b: vec![0.0; n_latent * 2],
            dec_w: rand_vec_f64(n_latent * input_dim, std_hyp, &mut rng),
            dec_b: vec![0.0; input_dim],
        }
    }

    /// Encode input to latent: \[input_dim\] → \[n_latent\].
    pub fn encode(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.input_dim);
        let raw = linear_f64(x, &self.enc_w, &self.enc_b, self.input_dim, self.n_latent);
        raw.iter().map(|&v| relu_f64(v)).collect()
    }

    /// Straight-through quantization: round to nearest integer during forward.
    pub fn quantize(latent: &[f64]) -> Vec<f64> {
        latent.iter().map(|&v| v.round()).collect()
    }

    /// Hyper-encode: latent → hyper-latent.
    pub fn hyper_encode(&self, y: &[f64]) -> Vec<f64> {
        assert_eq!(y.len(), self.n_latent);
        let raw = linear_f64(y, &self.hyper_enc_w, &self.hyper_enc_b, self.n_latent, self.n_hyper);
        raw.iter().map(|&v| relu_f64(v)).collect()
    }

    /// Hyper-decode: hyper-latent → (mean, log_scale) for each latent channel.
    pub fn hyper_decode(&self, z: &[f64]) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(z.len(), self.n_hyper);
        let out = linear_f64(z, &self.hyper_dec_w, &self.hyper_dec_b, self.n_hyper, self.n_latent * 2);
        let mean = out[..self.n_latent].to_vec();
        let log_scale: Vec<f64> = out[self.n_latent..].iter().map(|&v| v.clamp(-10.0, 10.0)).collect();
        (mean, log_scale)
    }

    /// Estimate entropy rate for quantized latent given hyperprior (mean, log_scale).
    ///
    /// Uses Gaussian entropy: 0.5 * log(2πe * σ²) per latent dimension.
    pub fn rate_estimate(&self, y_hat: &[f64], mean: &[f64], log_scale: &[f64]) -> f64 {
        y_hat.iter().zip(mean.iter()).zip(log_scale.iter())
            .map(|((&y, &m), &ls)| {
                let scale = ls.exp().max(1e-9);
                // Gaussian bits: log_2(scale * sqrt(2πe))
                let normalized = (y - m) / scale;
                0.5 * normalized * normalized + ls + 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln()
            })
            .sum::<f64>() / std::f64::consts::LN_2
    }

    /// Decode quantized latent to reconstruction: \[n_latent\] → \[input_dim\].
    pub fn decode(&self, y_hat: &[f64]) -> Vec<f64> {
        assert_eq!(y_hat.len(), self.n_latent);
        linear_f64(y_hat, &self.dec_w, &self.dec_b, self.n_latent, self.input_dim)
    }

    /// Compute R-D loss: λ·MSE_distortion + bits_rate.
    pub fn rd_loss(&self, x: &[f64], x_hat: &[f64], bits_rate: f64) -> f64 {
        let distortion = x.iter().zip(x_hat.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>() / x.len() as f64;
        self.lambda * distortion + bits_rate
    }

    /// Full forward pass: encode → quantize → hyperprior → decode.
    ///
    /// Returns (reconstruction, rate_estimate, rd_loss).
    pub fn forward(&self, x: &[f64]) -> (Vec<f64>, f64, f64) {
        let y = self.encode(x);
        let y_hat = Self::quantize(&y);
        let z = self.hyper_encode(&y);
        let (mean, log_scale) = self.hyper_decode(&z);
        let rate = self.rate_estimate(&y_hat, &mean, &log_scale);
        let x_hat = self.decode(&y_hat);
        let loss = self.rd_loss(x, &x_hat, rate);
        (x_hat, rate, loss)
    }
}

/// Channel-conditional Gaussian entropy model for multi-scale compression.
///
/// Models the probability of each latent channel conditioned on nearby channels
/// using a learned context model.
pub struct ChannelConditionalModel {
    /// Number of latent channels.
    pub n_channels: usize,
    /// Context window size (number of previous channels used).
    pub ctx_size: usize,
    ctx_w: Vec<f64>,
    ctx_b: Vec<f64>,
    // Predicts (mean, log_scale) from context
    pred_w: Vec<f64>,
    pred_b: Vec<f64>,
}

impl ChannelConditionalModel {
    /// Create a new ChannelConditionalModel.
    pub fn new(n_channels: usize, ctx_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let hidden = ctx_size * 4;
        let std = (2.0 / ctx_size as f64).sqrt();
        Self {
            n_channels,
            ctx_size,
            ctx_w: rand_vec_f64(ctx_size * hidden, std, &mut rng),
            ctx_b: vec![0.0; hidden],
            pred_w: rand_vec_f64(hidden * 2, std, &mut rng),
            pred_b: vec![0.0; 2],
        }
    }

    /// Predict (mean, log_scale) for channel `c` given previous channels.
    ///
    /// `latents` is \[n_channels\] flat; uses channels 0..c as context.
    pub fn predict_params(&self, latents: &[f64], c: usize) -> (f64, f64) {
        assert!(c < self.n_channels);
        let ctx_start = c.saturating_sub(self.ctx_size);
        let ctx: Vec<f64> = {
            let mut v = vec![0.0_f64; self.ctx_size];
            for (i, j) in (ctx_start..c).enumerate().take(self.ctx_size) {
                v[self.ctx_size - (c - ctx_start) + i] = latents[j];
            }
            v
        };
        let hidden = self.ctx_size * 4;
        let h1: Vec<f64> = (0..hidden)
            .map(|oi| relu_f64(
                self.ctx_w[oi * self.ctx_size..(oi + 1) * self.ctx_size]
                    .iter()
                    .zip(ctx.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f64>() + self.ctx_b[oi]
            ))
            .collect();
        let mean_logscale: Vec<f64> = (0..2)
            .map(|oi| {
                self.pred_w[oi * hidden..(oi + 1) * hidden]
                    .iter()
                    .zip(h1.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f64>() + self.pred_b[oi]
            })
            .collect();
        (mean_logscale[0], mean_logscale[1].clamp(-10.0, 10.0))
    }

    /// Compute total rate estimate for all latent channels.
    pub fn total_rate(&self, latents: &[f64]) -> f64 {
        assert_eq!(latents.len(), self.n_channels);
        (0..self.n_channels).map(|c| {
            let (mean, log_scale) = self.predict_params(latents, c);
            let scale = log_scale.exp().max(1e-9);
            let normalized = (latents[c] - mean) / scale;
            (0.5 * normalized * normalized + log_scale + 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln())
                / std::f64::consts::LN_2
        }).sum()
    }
}

/// Lagrangian Rate-Distortion optimizer for learned compression.
///
/// Minimizes total_loss = λ·D + R using gradient-free sensitivity analysis.
pub struct RdOptimizer {
    /// R-D trade-off lambda values to evaluate.
    pub lambdas: Vec<f64>,
    /// Current best lambda index.
    pub best_lambda_idx: usize,
}

impl RdOptimizer {
    /// Create a new RdOptimizer with a set of lambda values.
    pub fn new(lambdas: Vec<f64>) -> Self {
        let n = lambdas.len();
        Self { lambdas, best_lambda_idx: n / 2 }
    }

    /// Compute R-D curve points given distortions and rates at each lambda.
    ///
    /// Returns Vec of (bits_per_dim, psnr_db) operating points.
    pub fn rd_curve(&self, distortions: &[f64], rates: &[f64]) -> Vec<(f64, f64)> {
        assert_eq!(distortions.len(), self.lambdas.len());
        assert_eq!(rates.len(), self.lambdas.len());
        distortions.iter().zip(rates.iter()).map(|(&d, &r)| {
            let psnr = if d > 0.0 { 10.0 * (1.0 / d).log10() } else { 100.0 };
            (r, psnr)
        }).collect()
    }

    /// Select optimal lambda index for a given bitrate budget.
    pub fn select_lambda(&mut self, target_rate: f64, rates: &[f64]) -> usize {
        assert_eq!(rates.len(), self.lambdas.len());
        let best = rates.iter().enumerate().min_by(|a, b| {
            (a.1 - target_rate).abs().partial_cmp(&(b.1 - target_rate).abs())
                .unwrap_or(Ordering::Equal)
        }).map(|(i, _)| i).unwrap_or(0);
        self.best_lambda_idx = best;
        best
    }

    /// Compute Bjontegaard delta rate (BD-Rate) between two R-D curves.
    ///
    /// Uses trapezoidal integration over overlapping PSNR range.
    /// Returns percentage rate saving (negative = anchor is better).
    pub fn bd_rate(&self, anchor: &[(f64, f64)], test: &[(f64, f64)]) -> f64 {
        if anchor.len() < 2 || test.len() < 2 { return 0.0; }
        // Find overlapping PSNR range
        let min_psnr = anchor.iter().map(|p| p.1).chain(test.iter().map(|p| p.1))
            .fold(f64::NEG_INFINITY, f64::max).min(
                anchor.iter().map(|p| p.1).fold(f64::INFINITY, f64::min).max(
                    test.iter().map(|p| p.1).fold(f64::INFINITY, f64::min)
                )
            );
        let max_psnr = anchor.iter().map(|p| p.1).chain(test.iter().map(|p| p.1))
            .fold(f64::INFINITY, f64::min).max(
                anchor.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max).min(
                    test.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max)
                )
            );
        if max_psnr <= min_psnr { return 0.0; }

        // Trapezoidal integration of log(rate) vs PSNR
        let integrate = |curve: &[(f64, f64)]| -> f64 {
            let n_eval = 100usize;
            let mut integral = 0.0_f64;
            for i in 0..n_eval {
                let psnr = min_psnr + (max_psnr - min_psnr) * i as f64 / (n_eval - 1) as f64;
                // Linear interpolation of log(rate)
                let log_rate = curve.windows(2).find_map(|w| {
                    if w[0].1 <= psnr && psnr <= w[1].1 {
                        let t = (psnr - w[0].1) / (w[1].1 - w[0].1 + 1e-10);
                        Some(w[0].0.max(1e-10).ln() * (1.0 - t) + w[1].0.max(1e-10).ln() * t)
                    } else { None }
                }).unwrap_or(curve[0].0.max(1e-10).ln());
                integral += log_rate / n_eval as f64;
            }
            integral
        };

        let int_anchor = integrate(anchor);
        let int_test = integrate(test);
        (int_test.exp() / int_anchor.exp() - 1.0) * 100.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Neural Network Compression
// ─────────────────────────────────────────────────────────────────────────────

/// One-shot magnitude-based weight pruning.
///
/// Prunes weights globally by magnitude to reach a target sparsity level.
pub struct MagnitudePruner {
    /// Target sparsity (0.0 = dense, 1.0 = fully pruned).
    pub target_sparsity: f64,
    /// Computed threshold after pruning.
    pub threshold: f64,
}

impl MagnitudePruner {
    /// Create a new MagnitudePruner.
    pub fn new(target_sparsity: f64) -> Self {
        Self { target_sparsity: target_sparsity.clamp(0.0, 1.0), threshold: 0.0 }
    }

    /// Prune all weights globally: set those below threshold to zero.
    ///
    /// Returns pruned weight matrices.
    pub fn prune(&mut self, weights: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut all_abs: Vec<f64> = weights.iter()
            .flat_map(|r| r.iter().map(|&w| w.abs()))
            .collect();
        all_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let n = all_abs.len();
        let prune_n = ((n as f64 * self.target_sparsity) as usize).min(n);
        self.threshold = if prune_n < n { all_abs[prune_n] } else { f64::MAX };
        let thr = self.threshold;
        weights.into_iter().map(|row| {
            row.into_iter().map(|w| if w.abs() >= thr { w } else { 0.0 }).collect()
        }).collect()
    }

    /// Compute achieved sparsity for a set of weight matrices.
    pub fn achieved_sparsity(weights: &[Vec<f64>]) -> f64 {
        let total: usize = weights.iter().map(|r| r.len()).sum();
        let zeros: usize = weights.iter()
            .flat_map(|r| r.iter())
            .filter(|&&v| v == 0.0)
            .count();
        if total == 0 { 0.0 } else { zeros as f64 / total as f64 }
    }

    /// Generate binary mask from pruned weights.
    pub fn mask_from_weights(weights: &[Vec<f64>]) -> Vec<Vec<f64>> {
        weights.iter().map(|r| {
            r.iter().map(|&w| if w != 0.0 { 1.0 } else { 0.0 }).collect()
        }).collect()
    }
}

/// Movement-based pruning (Sanh et al. 2020).
///
/// Score = weight * gradient (accumulated over training).
/// Weights with small movement score are pruned as they contribute little.
pub struct MovementPrunerV2 {
    /// Target sparsity.
    pub target_sparsity: f64,
    /// Accumulated movement scores.
    pub scores: Vec<Vec<f64>>,
    /// Number of update steps accumulated.
    pub n_steps: usize,
}

impl MovementPrunerV2 {
    /// Create a new MovementPrunerV2.
    pub fn new(target_sparsity: f64, n_layers: usize, layer_sizes: &[(usize, usize)]) -> Self {
        let scores = layer_sizes.iter().take(n_layers)
            .map(|&(rows, cols)| vec![0.0; rows * cols])
            .collect();
        Self { target_sparsity, scores, n_steps: 0 }
    }

    /// Accumulate movement scores from weight * gradient for each layer.
    pub fn accumulate(&mut self, weights: &[Vec<f64>], gradients: &[Vec<f64>]) -> Result<()> {
        if weights.len() != self.scores.len() || gradients.len() != self.scores.len() {
            return Err(TensorError::invalid_argument(
                "weights/gradients/scores must have the same number of layers".to_string()
            ));
        }
        for (layer_idx, ((w_row, g_row), scores)) in weights.iter()
            .zip(gradients.iter())
            .zip(self.scores.iter_mut())
            .enumerate()
        {
            if w_row.len() != g_row.len() {
                return Err(TensorError::invalid_argument(format!(
                    "layer {}: weight len {} != gradient len {}", layer_idx, w_row.len(), g_row.len()
                )));
            }
            for (s, (w, g)) in scores.iter_mut().zip(w_row.iter().zip(g_row.iter())) {
                *s += w * g;
            }
        }
        self.n_steps += 1;
        Ok(())
    }

    /// Apply pruning based on accumulated movement scores.
    ///
    /// Returns binary masks for each layer.
    pub fn compute_mask(&self) -> Vec<Vec<f64>> {
        let all_scores: Vec<f64> = self.scores.iter()
            .flat_map(|r| r.iter().cloned())
            .collect();
        let n = all_scores.len();
        let prune_n = ((n as f64 * self.target_sparsity) as usize).min(n);

        let mut sorted = all_scores.clone();
        sorted.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap_or(Ordering::Equal));
        let threshold = if prune_n < n { sorted[prune_n].abs() } else { f64::MAX };

        self.scores.iter().map(|row| {
            row.iter().map(|&s| if s.abs() >= threshold { 1.0 } else { 0.0 }).collect()
        }).collect()
    }
}

/// Mixed-precision search via sensitivity analysis and evolutionary optimization.
///
/// Assigns per-layer bit-widths (from `allowed_bits`) to minimize accuracy loss
/// under a memory budget constraint.
pub struct MixedPrecisionSearch {
    /// Allowed bit-widths (e.g., [2, 4, 8]).
    pub allowed_bits: Vec<usize>,
    /// Memory budget in bits.
    pub memory_budget_bits: usize,
    /// Sensitivity scores per layer (higher = more sensitive, needs higher precision).
    pub sensitivity: Vec<f64>,
}

impl MixedPrecisionSearch {
    /// Create a new MixedPrecisionSearch.
    pub fn new(allowed_bits: Vec<usize>, memory_budget_bits: usize, n_layers: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let sensitivity = (0..n_layers).map(|_| rng.random::<f64>()).collect();
        Self { allowed_bits, memory_budget_bits, sensitivity }
    }

    /// Compute sensitivity from weight L2 norms (Hessian proxy).
    pub fn compute_sensitivity(&mut self, weights: &[Vec<f64>]) {
        self.sensitivity = weights.iter().map(|row| {
            let l2_sq: f64 = row.iter().map(|&w| w * w).sum();
            l2_sq.sqrt() / (row.len() as f64).sqrt()
        }).collect();
    }

    /// Evolutionary search for bit-width assignment.
    ///
    /// Returns per-layer bit-widths satisfying the memory budget.
    pub fn search(&self, layer_sizes: &[usize], n_generations: usize, rng: &mut StdRng) -> Vec<usize> {
        let n = layer_sizes.len().min(self.sensitivity.len());
        if n == 0 || self.allowed_bits.is_empty() {
            return vec![];
        }

        // Initialize population with random assignments
        let pop_size = 20usize;
        let mut population: Vec<Vec<usize>> = (0..pop_size)
            .map(|_| (0..n).map(|_| {
                self.allowed_bits[(rng.random::<f64>() * self.allowed_bits.len() as f64) as usize
                    % self.allowed_bits.len()]
            }).collect())
            .collect();

        let fitness = |assignment: &[usize]| -> f64 {
            let total_bits: usize = assignment.iter().zip(layer_sizes.iter())
                .map(|(&b, &s)| b * s)
                .sum();
            let budget_penalty = if total_bits > self.memory_budget_bits {
                (total_bits - self.memory_budget_bits) as f64 * 1e-3
            } else { 0.0 };
            // Accuracy proxy: sensitive layers penalized for low bit-width
            let acc_loss: f64 = assignment.iter().zip(self.sensitivity.iter().take(n))
                .map(|(&b, &s)| {
                    let b_max = *self.allowed_bits.iter().max().unwrap_or(&8) as f64;
                    s * (1.0 - b as f64 / b_max)
                })
                .sum();
            -(acc_loss + budget_penalty)
        };

        for _gen in 0..n_generations {
            // Tournament selection + mutation
            let mut new_pop = Vec::with_capacity(pop_size);
            for _ in 0..pop_size {
                let i = (rng.random::<f64>() * pop_size as f64) as usize % pop_size;
                let j = (rng.random::<f64>() * pop_size as f64) as usize % pop_size;
                let parent = if fitness(&population[i]) > fitness(&population[j]) { &population[i] } else { &population[j] };
                let mut child = parent.clone();
                // Mutate one random layer
                let mut_layer = (rng.random::<f64>() * n as f64) as usize % n;
                child[mut_layer] = self.allowed_bits[
                    (rng.random::<f64>() * self.allowed_bits.len() as f64) as usize % self.allowed_bits.len()
                ];
                new_pop.push(child);
            }
            population = new_pop;
        }

        // Return best individual
        population.into_iter().max_by(|a, b| {
            fitness(a).partial_cmp(&fitness(b)).unwrap_or(Ordering::Equal)
        }).unwrap_or_else(|| vec![*self.allowed_bits.last().unwrap_or(&8); n])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Knowledge Compression
// ─────────────────────────────────────────────────────────────────────────────

/// Structured layer dropping with performance-aware selection.
///
/// Removes entire Transformer layers based on their importance score,
/// as in the L-th Transformer layer removal strategy.
pub struct LayerDropper {
    /// Number of layers to keep after dropping.
    pub n_keep: usize,
    /// Importance scores per layer (higher = more important, keep).
    pub scores: Vec<f64>,
}

impl LayerDropper {
    /// Create a new LayerDropper.
    pub fn new(n_layers: usize, n_keep: usize) -> Self {
        assert!(n_keep <= n_layers, "n_keep must be <= n_layers");
        Self { n_keep, scores: vec![1.0; n_layers] }
    }

    /// Compute layer importance using cosine similarity between adjacent layer outputs.
    ///
    /// Layers with outputs nearly identical to their input are deemed less important.
    /// `layer_outputs` is [n_layers, n_tokens, d_model] flattened by layer.
    pub fn compute_importance(&mut self, layer_outputs: &[Vec<f64>]) {
        self.scores = layer_outputs.windows(2).map(|w| {
            let a = &w[0];
            let b = &w[1];
            if a.is_empty() || b.is_empty() { return 1.0; }
            let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
            let na: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
            let cos_sim = dot / (na * nb + 1e-10);
            // Importance = 1 - cos_sim: layers that change the representation more are important
            1.0 - cos_sim
        }).collect();
        // Pad if needed
        while self.scores.len() < layer_outputs.len() {
            self.scores.push(1.0);
        }
    }

    /// Select which layers to keep based on importance scores.
    ///
    /// Always keeps the first and last layer; fills remaining slots with most important layers.
    /// Returns indices of layers to keep (sorted).
    pub fn select_layers(&self) -> Vec<usize> {
        let n = self.scores.len();
        if n == 0 || self.n_keep >= n { return (0..n).collect(); }

        let mut indexed: Vec<(usize, f64)> = self.scores.iter().cloned().enumerate().collect();
        // Always keep first and last
        let first = 0usize;
        let last = n - 1;
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let mut keep: Vec<usize> = vec![first, last];
        for &(idx, _) in &indexed {
            if keep.len() >= self.n_keep { break; }
            if !keep.contains(&idx) {
                keep.push(idx);
            }
        }
        keep.sort_unstable();
        keep.truncate(self.n_keep);
        keep
    }

    /// Apply layer dropping: extract only the kept layer outputs.
    pub fn drop_layers<'a>(&self, layer_outputs: &'a [Vec<f64>]) -> Vec<&'a Vec<f64>> {
        let keep = self.select_layers();
        keep.into_iter().filter_map(|i| layer_outputs.get(i)).collect()
    }
}

/// Vocabulary embedding pruner.
///
/// Prunes infrequently-used token embeddings from a vocabulary embedding matrix,
/// also considering downstream task importance scores.
pub struct VocabPruner {
    /// Frequency threshold (tokens below this frequency are candidates for pruning).
    pub freq_threshold: f64,
    /// Kept vocabulary size after pruning.
    pub kept_vocab_size: usize,
}

impl VocabPruner {
    /// Create a new VocabPruner.
    pub fn new(freq_threshold: f64) -> Self {
        Self { freq_threshold, kept_vocab_size: 0 }
    }

    /// Prune vocabulary embeddings.
    ///
    /// `embeddings` is [vocab_size, embed_dim] flattened.
    /// `frequencies` is \[vocab_size\] token frequency counts.
    /// `importance` is \[vocab_size\] optional downstream importance scores.
    ///
    /// Returns (pruned_embeddings, kept_indices).
    pub fn prune(
        &mut self,
        embeddings: &[f64],
        vocab_size: usize,
        embed_dim: usize,
        frequencies: &[f64],
        importance: Option<&[f64]>,
    ) -> Result<(Vec<f64>, Vec<usize>)> {
        if embeddings.len() != vocab_size * embed_dim {
            return Err(TensorError::invalid_argument(
                "embeddings size mismatch".to_string()
            ));
        }
        if frequencies.len() != vocab_size {
            return Err(TensorError::invalid_argument(
                "frequencies length must equal vocab_size".to_string()
            ));
        }

        let kept: Vec<usize> = (0..vocab_size).filter(|&i| {
            let freq_ok = frequencies[i] >= self.freq_threshold;
            let imp_ok = importance.map(|imp| imp[i] > 0.0).unwrap_or(true);
            freq_ok || imp_ok
        }).collect();

        self.kept_vocab_size = kept.len();
        let pruned: Vec<f64> = kept.iter().flat_map(|&i| {
            embeddings[i * embed_dim..(i + 1) * embed_dim].iter().cloned()
        }).collect();

        Ok((pruned, kept))
    }

    /// Compute vocabulary coverage (fraction of tokens in a corpus covered by kept vocab).
    pub fn coverage(kept_indices: &[usize], token_counts: &[f64]) -> f64 {
        let total: f64 = token_counts.iter().sum();
        if total == 0.0 { return 0.0; }
        let covered: f64 = kept_indices.iter()
            .filter_map(|&i| token_counts.get(i))
            .sum();
        covered / total
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Compression Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Comprehensive compression evaluation metrics.
pub struct CompMetrics;

impl CompMetrics {
    /// Compression ratio: original_size / compressed_size.
    pub fn compression_ratio(original_params: usize, compressed_params: usize, bits_per_param: f64) -> f64 {
        if compressed_params == 0 { return 0.0; }
        let original_bits = original_params as f64 * 32.0; // FP32
        let compressed_bits = compressed_params as f64 * bits_per_param;
        original_bits / compressed_bits
    }

    /// Accuracy degradation (percentage drop).
    pub fn accuracy_degradation(original_acc: f64, compressed_acc: f64) -> f64 {
        (original_acc - compressed_acc) * 100.0
    }

    /// Latency reduction percentage.
    pub fn latency_reduction(original_ms: f64, compressed_ms: f64) -> f64 {
        if original_ms == 0.0 { return 0.0; }
        (1.0 - compressed_ms / original_ms) * 100.0
    }

    /// Bits-per-dim (bpd) for generative model evaluation.
    pub fn bits_per_dim(nll_nats: f64, n_dims: usize) -> f64 {
        nll_nats / (n_dims as f64 * std::f64::consts::LN_2)
    }

    /// PSNR (Peak Signal-to-Noise Ratio) in dB.
    pub fn psnr(original: &[f64], reconstructed: &[f64]) -> f64 {
        let mse: f64 = original.iter().zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>() / original.len() as f64;
        if mse < 1e-12 { return 100.0; }
        10.0 * (1.0 / mse).log10()
    }

    /// Model size in MB given parameter count and bit-width.
    pub fn model_size_mb(n_params: usize, bits: usize) -> f64 {
        n_params as f64 * bits as f64 / (8.0 * 1024.0 * 1024.0)
    }

    /// Sparsity-aware FLOPs estimate (sparse matmul savings).
    pub fn sparse_flops(dense_flops: u64, sparsity: f64) -> u64 {
        let reduction = 1.0 - sparsity * sparsity; // CSR-style savings
        (dense_flops as f64 * reduction) as u64
    }

    /// Summary report as formatted string.
    pub fn report(
        n_params_orig: usize,
        n_params_compressed: usize,
        bits: usize,
        original_acc: f64,
        compressed_acc: f64,
        original_ms: f64,
        compressed_ms: f64,
    ) -> String {
        let ratio = Self::compression_ratio(n_params_orig, n_params_compressed, bits as f64);
        let acc_drop = Self::accuracy_degradation(original_acc, compressed_acc);
        let lat_red = Self::latency_reduction(original_ms, compressed_ms);
        let size_orig = Self::model_size_mb(n_params_orig, 32);
        let size_comp = Self::model_size_mb(n_params_compressed, bits);
        format!(
            "CompMetrics: ratio={:.2}x | acc_drop={:.2}% | lat_reduction={:.1}% | size {:.1}MB→{:.1}MB",
            ratio, acc_drop, lat_red, size_orig, size_comp
        )
    }
}
