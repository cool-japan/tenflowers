//! Neural SDE Advanced — Rough Path Theory, Controlled SDEs, Score-Based Diffusion via SDE.
//!
//! Implements:
//! - §A Rough Path Theory: RoughPath, SignatureTransform, LogSignatureLayer
//! - §B Controlled SDEs: ControlledSde, NeuralRde
//! - §C Score-Based Diffusion via SDE: VpSde, VeSde, ScoreMatchingSde, AncestralSampler
//! - §D Extended SDE Metrics: SdeMetricsExtended

use super::{sample_standard_normal, NsdeMlp, PathSignature};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─── §A Rough Path Theory ─────────────────────────────────────────────────────

/// A rough path of Hölder regularity 1/p, represented as a sequence of
/// iterated integrals up to level M = floor(p).
///
/// Reference: Lyons (1998), Terry Lyons rough path theory.
#[derive(Debug, Clone)]
pub struct RoughPath {
    /// Path dimension.
    pub dim: usize,
    /// Truncation level M (iterated integrals up to level M).
    pub level: usize,
    /// Signature tensor for each time step pair stored as flat vec.
    /// `signatures[t]` is the truncated signature over `path[0..=t]`.
    pub signatures: Vec<Vec<f64>>,
}

impl RoughPath {
    /// Construct a rough path from a sequence of observations.
    ///
    /// Computes the truncated signature of the path up to the given level.
    pub fn new(path: &[Vec<f64>], level: usize) -> Self {
        let dim = path.first().map(|v| v.len()).unwrap_or(1);
        let ps = PathSignature::new(dim, level);
        // Compute cumulative signatures path[0..=t]
        let mut signatures = Vec::with_capacity(path.len());
        for t in 1..=path.len() {
            let sub = &path[..t];
            signatures.push(ps.compute(sub));
        }
        Self {
            dim,
            level,
            signatures,
        }
    }

    /// Retrieve the signature at time step t (0-indexed).
    pub fn signature_at(&self, t: usize) -> &[f64] {
        let idx = t.min(self.signatures.len().saturating_sub(1));
        self.signatures
            .get(idx)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Compute the Lévy area (antisymmetric part of level-2 signature) between two dimensions.
    ///
    /// Lévy(i,j) = (S^{ij} - S^{ji}) / 2, approximates the stochastic area.
    pub fn levy_area(&self, t: usize, i: usize, j: usize) -> f64 {
        let sig = self.signature_at(t);
        let d = self.dim;
        let level2_offset = 1 + d; // skip level-0 (1 elem) and level-1 (d elems)
        let s_ij = sig.get(level2_offset + i * d + j).copied().unwrap_or(0.0);
        let s_ji = sig.get(level2_offset + j * d + i).copied().unwrap_or(0.0);
        (s_ij - s_ji) * 0.5
    }

    /// Compute the Hölder p-variation estimate: max over dyadic intervals.
    ///
    /// Returns p-variation^{1/p} as an approximation.
    pub fn holder_variation(&self, p: f64) -> f64 {
        let n = self.signatures.len();
        if n < 2 {
            return 0.0;
        }
        let mut total = 0.0_f64;
        for t in 0..(n - 1) {
            let s_t = self.signature_at(t);
            let s_t1 = self.signature_at(t + 1);
            let diff_sq: f64 = s_t
                .iter()
                .zip(s_t1.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum();
            total += diff_sq.sqrt().powf(p);
        }
        total.powf(1.0 / p)
    }
}

/// Signature transform layer: maps an input path to its (log-)signature features.
///
/// Suitable for use as a feature extractor in path-based models.
#[derive(Debug, Clone)]
pub struct SignatureTransform {
    /// Input path dimension.
    pub x_dim: usize,
    /// Truncation order.
    pub level: usize,
    /// Whether to compute log-signature (more compact) or full signature.
    pub use_log_signature: bool,
}

impl SignatureTransform {
    /// Construct a SignatureTransform.
    pub fn new(x_dim: usize, level: usize, use_log_signature: bool) -> Self {
        Self {
            x_dim,
            level,
            use_log_signature,
        }
    }

    /// Output dimension of the transform.
    pub fn output_dim(&self) -> usize {
        let ps = PathSignature::new(self.x_dim, self.level);
        if self.use_log_signature && self.level >= 2 {
            // Level 1: d; level 2 brackets: d*(d-1)/2
            self.x_dim + self.x_dim * (self.x_dim - 1) / 2
        } else if self.use_log_signature {
            self.x_dim // level 1 only
        } else {
            ps.signature_dim()
        }
    }

    /// Apply the transform to a path, returning the (log-)signature vector.
    pub fn forward(&self, path: &[Vec<f64>]) -> Vec<f64> {
        let ps = PathSignature::new(self.x_dim, self.level);
        if self.use_log_signature {
            ps.compute_log_signature(path)
        } else {
            ps.compute(path)
        }
    }
}

/// Log-signature feature layer that includes a learnable linear projection.
///
/// Maps path → log-signature → Linear(log_sig_dim, out_dim) → ReLU.
#[derive(Debug, Clone)]
pub struct LogSignatureLayer {
    /// The signature transform.
    pub transform: SignatureTransform,
    /// Linear projection from log-sig space to output space.
    pub projection: NsdeMlp,
    /// Output dimension.
    pub out_dim: usize,
}

impl LogSignatureLayer {
    /// Construct a LogSignatureLayer with a single-layer linear projection.
    pub fn new(x_dim: usize, level: usize, out_dim: usize, seed: u64) -> Self {
        let transform = SignatureTransform::new(x_dim, level, true);
        let in_dim = transform.output_dim().max(1);
        // Build MLP: in_dim → out_dim (linear projection with ReLU = out_dim)
        // We use NsdeMlp with two layers to allow non-linearity.
        let hidden = (in_dim + out_dim) / 2 + 1;
        let sizes = [in_dim, hidden, out_dim];
        let _ = seed; // NsdeMlp uses internal seed; seed parameter reserved for future use
        let projection = NsdeMlp::new(&sizes);
        Self {
            transform,
            projection,
            out_dim,
        }
    }

    /// Forward: path → log-signature → projected features.
    pub fn forward(&self, path: &[Vec<f64>]) -> Vec<f64> {
        let log_sig = self.transform.forward(path);
        self.projection.forward(&log_sig)
    }
}

// ─── §B Controlled SDEs ───────────────────────────────────────────────────────

/// A controlled SDE driven by a rough path X:
///
/// dZ(t) = f(Z(t)) dX(t) + g(Z(t)) dt
///
/// Solved via Euler-Maruyama on the rough path grid.
#[derive(Debug, Clone)]
pub struct ControlledSde {
    /// Dimension of the rough path X.
    pub x_dim: usize,
    /// Dimension of the latent state Z.
    pub z_dim: usize,
    /// f: R^{z_dim} → R^{z_dim × x_dim} (controlled vector field).
    pub controlled_field: NsdeMlp,
    /// g: R^{z_dim} → R^{z_dim} (drift).
    pub drift_net: NsdeMlp,
}

impl ControlledSde {
    /// Construct a ControlledSde with given network sizes.
    pub fn new(x_dim: usize, z_dim: usize, hidden_dim: usize) -> Self {
        // controlled_field: z_dim → z_dim * x_dim
        let cf_out = z_dim * x_dim;
        let controlled_field = NsdeMlp::new(&[z_dim, hidden_dim, cf_out]);
        // drift_net: z_dim → z_dim
        let drift_net = NsdeMlp::new(&[z_dim, hidden_dim, z_dim]);
        Self {
            x_dim,
            z_dim,
            controlled_field,
            drift_net,
        }
    }

    /// Solve the controlled SDE given an initial state z0 and a rough path.
    ///
    /// Returns trajectory `[n_steps+1][z_dim]`.
    pub fn integrate(&self, z0: &[f64], rough_path: &RoughPath, dt: f64) -> Vec<Vec<f64>> {
        let n = rough_path.signatures.len();
        let mut traj = Vec::with_capacity(n + 1);
        let mut z = z0.to_vec();
        traj.push(z.clone());

        for t in 0..n {
            let sig_t = rough_path.signature_at(t);
            // Approximate dX as difference of level-1 signature components
            let d = self.x_dim;
            let dx: Vec<f64> = if t == 0 {
                // First step: level-1 from sig (offset 1)
                (0..d)
                    .map(|i| sig_t.get(1 + i).copied().unwrap_or(0.0))
                    .collect()
            } else {
                let sig_prev = rough_path.signature_at(t - 1);
                (0..d)
                    .map(|i| {
                        sig_t.get(1 + i).copied().unwrap_or(0.0)
                            - sig_prev.get(1 + i).copied().unwrap_or(0.0)
                    })
                    .collect()
            };

            // f(z) * dx: [z_dim×x_dim] × [x_dim] → [z_dim]
            let f_mat = self.controlled_field.forward(&z);
            let mut controlled_term = vec![0.0; self.z_dim];
            for zi in 0..self.z_dim {
                for xi in 0..self.x_dim {
                    let m = f_mat.get(zi * self.x_dim + xi).copied().unwrap_or(0.0);
                    let dxi = dx.get(xi).copied().unwrap_or(0.0);
                    controlled_term[zi] += m * dxi;
                }
            }

            // g(z) * dt
            let drift = self.drift_net.forward(&z);

            let z_new: Vec<f64> = z
                .iter()
                .enumerate()
                .map(|(i, &zi)| {
                    zi + controlled_term.get(i).copied().unwrap_or(0.0)
                        + drift.get(i).copied().unwrap_or(0.0) * dt
                })
                .collect();
            z = z_new;
            traj.push(z.clone());
        }
        traj
    }
}

/// Neural Rough Differential Equation (NeuralRDE / Neural RDE).
///
/// Extends Neural CDE to handle rough paths (higher-order controlled equations).
/// dZ = f(Z) dX^{≤M}, where X^{≤M} is the rough path lift of order M.
#[derive(Debug, Clone)]
pub struct NeuralRde {
    /// Controlled SDE for integration.
    pub csde: ControlledSde,
    /// Initial net: x_0 → z_0.
    pub initial_net: NsdeMlp,
    /// Readout: z_T → output.
    pub readout_net: NsdeMlp,
    /// Path signature transform for feature extraction.
    pub sig_transform: SignatureTransform,
    /// Output dimension.
    pub out_dim: usize,
}

impl NeuralRde {
    /// Construct a NeuralRDE with default architecture.
    pub fn new(x_dim: usize, z_dim: usize, out_dim: usize, level: usize, hidden: usize) -> Self {
        let csde = ControlledSde::new(x_dim, z_dim, hidden);
        let initial_net = NsdeMlp::new(&[x_dim, hidden, z_dim]);
        let readout_net = NsdeMlp::new(&[z_dim, hidden, out_dim]);
        let sig_transform = SignatureTransform::new(x_dim, level, false);
        Self {
            csde,
            initial_net,
            readout_net,
            sig_transform,
            out_dim,
        }
    }

    /// Forward pass: path → rough path → integrate → readout.
    pub fn forward(&self, path: &[Vec<f64>], dt: f64) -> Vec<f64> {
        let rough = RoughPath::new(path, self.sig_transform.level);
        let x0 = path.first().map(|v| v.as_slice()).unwrap_or(&[]);
        let z0 = self.initial_net.forward(x0);
        let traj = self.csde.integrate(&z0, &rough, dt);
        let z_final = traj
            .last()
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.csde.z_dim]);
        self.readout_net.forward(&z_final)
    }
}

// ─── §C Score-Based Diffusion via SDE ────────────────────────────────────────

/// Variance Preserving SDE (Ho et al. 2020 / Song et al. 2021).
///
/// Forward: dX = -0.5 * β(t) * X dt + sqrt(β(t)) dW
/// where β(t) = β_min + t*(β_max - β_min) is the noise schedule.
#[derive(Debug, Clone)]
pub struct VpSde {
    /// Minimum noise level.
    pub beta_min: f64,
    /// Maximum noise level.
    pub beta_max: f64,
    /// Total diffusion time [0, T].
    pub t_max: f64,
}

impl VpSde {
    /// Construct a VpSde with standard DDPM schedule (β_min=0.1, β_max=20).
    pub fn new(beta_min: f64, beta_max: f64, t_max: f64) -> Self {
        Self {
            beta_min,
            beta_max,
            t_max,
        }
    }

    /// Linear noise schedule β(t).
    pub fn beta(&self, t: f64) -> f64 {
        let t_norm = (t / self.t_max).clamp(0.0, 1.0);
        self.beta_min + t_norm * (self.beta_max - self.beta_min)
    }

    /// Log-signal-to-noise ratio λ(t) = log(α̅(t) / (1 - α̅(t))).
    ///
    /// α̅(t) = exp(-0.5 * integral_0^t β(s) ds)
    pub fn log_alpha_bar(&self, t: f64) -> f64 {
        let t_norm = (t / self.t_max).clamp(0.0, 1.0);
        // Integral of β: β_min*t + 0.5*(β_max - β_min)*t^2 (over [0,1])
        let integral =
            self.beta_min * t_norm + 0.5 * (self.beta_max - self.beta_min) * t_norm * t_norm;
        -0.5 * integral * self.t_max
    }

    /// Marginal standard deviation σ(t) = sqrt(1 - exp(2 * log_alpha_bar(t))).
    pub fn sigma(&self, t: f64) -> f64 {
        let log_ab = self.log_alpha_bar(t);
        let alpha_bar_sq = (2.0 * log_ab).exp();
        (1.0 - alpha_bar_sq).max(0.0).sqrt()
    }

    /// Forward diffusion: add noise to x0 at time t.
    ///
    /// Returns (x_t, noise) where x_t = alpha_bar(t)*x0 + sigma(t)*noise.
    pub fn forward_diffuse(&self, x0: &[f64], t: f64, rng: &mut StdRng) -> (Vec<f64>, Vec<f64>) {
        let log_ab = self.log_alpha_bar(t);
        let alpha_bar = log_ab.exp();
        let sigma = self.sigma(t);
        let noise: Vec<f64> = x0.iter().map(|_| sample_standard_normal(rng)).collect();
        let x_t: Vec<f64> = x0
            .iter()
            .zip(noise.iter())
            .map(|(&x, &eps)| alpha_bar * x + sigma * eps)
            .collect();
        (x_t, noise)
    }
}

/// Variance Exploding SDE (Song et al. 2021).
///
/// Forward: dX = sigma(t) * sqrt(d/dt[sigma^2(t)]) dW
/// where sigma(t) = sigma_min * (sigma_max/sigma_min)^t.
#[derive(Debug, Clone)]
pub struct VeSde {
    /// Minimum sigma level.
    pub sigma_min: f64,
    /// Maximum sigma level.
    pub sigma_max: f64,
}

impl VeSde {
    /// Construct a VeSde with NCSN-style schedule.
    pub fn new(sigma_min: f64, sigma_max: f64) -> Self {
        Self {
            sigma_min,
            sigma_max,
        }
    }

    /// Sigma schedule σ(t) = σ_min * (σ_max/σ_min)^t, t ∈ [0, 1].
    pub fn sigma(&self, t: f64) -> f64 {
        let t_norm = t.clamp(0.0, 1.0);
        self.sigma_min * (self.sigma_max / self.sigma_min).powf(t_norm)
    }

    /// Forward diffusion: x_t = x0 + sigma(t) * noise.
    pub fn forward_diffuse(&self, x0: &[f64], t: f64, rng: &mut StdRng) -> (Vec<f64>, Vec<f64>) {
        let sigma = self.sigma(t);
        let noise: Vec<f64> = x0.iter().map(|_| sample_standard_normal(rng)).collect();
        let x_t: Vec<f64> = x0
            .iter()
            .zip(noise.iter())
            .map(|(&x, &eps)| x + sigma * eps)
            .collect();
        (x_t, noise)
    }

    /// Score of the marginal: ∇_x log p_t(x | x0) = -(x_t - x0) / sigma^2(t).
    pub fn score(&self, x_t: &[f64], x0: &[f64], t: f64) -> Vec<f64> {
        let sigma_sq = self.sigma(t).powi(2).max(1e-12);
        x_t.iter()
            .zip(x0.iter())
            .map(|(&xt, &x0i)| -(xt - x0i) / sigma_sq)
            .collect()
    }
}

/// Score-matching SDE training wrapper.
///
/// Trains a score network s_θ(x, t) ≈ ∇_x log p_t(x) using denoising score matching.
#[derive(Debug, Clone)]
pub struct ScoreMatchingSde {
    /// Score network: [x, t] → score.
    pub score_net: NsdeMlp,
    /// Data dimension.
    pub x_dim: usize,
    /// VP SDE for forward diffusion.
    pub vp_sde: VpSde,
    /// Number of diffusion time steps.
    pub n_timesteps: usize,
}

impl ScoreMatchingSde {
    /// Construct a ScoreMatchingSde with default VP SDE.
    pub fn new(x_dim: usize, hidden_dim: usize, n_timesteps: usize) -> Self {
        // Score network input: [x (dim) | t (1)] → score (x_dim)
        let score_net = NsdeMlp::new(&[x_dim + 1, hidden_dim, hidden_dim, x_dim]);
        let vp_sde = VpSde::new(0.1, 20.0, 1.0);
        Self {
            score_net,
            x_dim,
            vp_sde,
            n_timesteps,
        }
    }

    /// Evaluate score s_θ(x_t, t).
    pub fn score(&self, x_t: &[f64], t: f64) -> Vec<f64> {
        let mut inp = x_t.to_vec();
        inp.push(t);
        self.score_net.forward(&inp)
    }

    /// Compute denoising score matching loss for a batch of clean samples.
    ///
    /// Loss = E_{t, x0, ε} [||s_θ(x_t, t) + ε/σ(t)||^2]
    pub fn dsm_loss(&self, x0_batch: &[Vec<f64>], rng: &mut StdRng) -> f64 {
        let n = x0_batch.len();
        if n == 0 {
            return 0.0;
        }
        let mut total_loss = 0.0;
        for x0 in x0_batch {
            // Sample random t ∈ (0, 1]
            let t = (rng.random::<f64>() * 0.99 + 0.01).clamp(1e-3, 1.0);
            let (x_t, noise) = self.vp_sde.forward_diffuse(x0, t, rng);
            let sigma = self.vp_sde.sigma(t).max(1e-8);
            let score_pred = self.score(&x_t, t);
            // Target score: -noise / sigma
            let loss: f64 = score_pred
                .iter()
                .zip(noise.iter())
                .map(|(&sp, &eps)| {
                    let target = -eps / sigma;
                    (sp - target).powi(2)
                })
                .sum::<f64>()
                / self.x_dim as f64;
            total_loss += loss;
        }
        total_loss / n as f64
    }
}

/// Ancestral sampling (reverse SDE solver) for score-based generative models.
///
/// Implements the reverse-time SDE: dX = [-f(X,t) + g²(t)*s_θ(X,t)] dt + g(t) dW̄
/// using the Euler-Maruyama discretization.
#[derive(Debug, Clone)]
pub struct AncestralSampler {
    /// The trained score-matching SDE.
    pub score_sde: ScoreMatchingSde,
    /// Number of reverse diffusion steps.
    pub n_steps: usize,
}

impl AncestralSampler {
    /// Construct an AncestralSampler.
    pub fn new(score_sde: ScoreMatchingSde, n_steps: usize) -> Self {
        Self { score_sde, n_steps }
    }

    /// Generate samples from the model by running the reverse SDE.
    ///
    /// Starts from x_T ~ N(0, I) and integrates backward to t=0.
    pub fn sample(&self, rng: &mut StdRng) -> Vec<f64> {
        let x_dim = self.score_sde.x_dim;
        // Initialize from x_T ~ N(0, I)
        let mut x: Vec<f64> = (0..x_dim).map(|_| sample_standard_normal(rng)).collect();
        let dt = 1.0 / self.n_steps as f64;

        for step in 0..self.n_steps {
            // t decreases from 1 to 0 (reverse time)
            let t = 1.0 - step as f64 * dt;
            let t_clamped = t.clamp(1e-3, 1.0);
            let beta = self.score_sde.vp_sde.beta(t_clamped);
            // VP SDE: f(x,t) = -0.5*beta(t)*x, g(t) = sqrt(beta(t))
            let g_sq = beta;
            let score = self.score_sde.score(&x, t_clamped);
            // Reverse drift: -f(x,t) + g^2 * score = 0.5*beta*x + beta*score
            let noise: Vec<f64> = (0..x_dim).map(|_| sample_standard_normal(rng)).collect();
            x = x
                .iter()
                .enumerate()
                .map(|(i, &xi)| {
                    let drift = 0.5 * beta * xi + g_sq * score.get(i).copied().unwrap_or(0.0);
                    let diffusion = g_sq.sqrt() * noise.get(i).copied().unwrap_or(0.0);
                    xi + drift * dt + diffusion * dt.sqrt()
                })
                .collect();
        }
        x
    }

    /// Generate a batch of samples.
    pub fn sample_batch(&self, n: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.sample(rng)).collect()
    }
}

// ─── §D Extended SDE Metrics ──────────────────────────────────────────────────

/// Extended metrics for evaluating SDE-based generative models.
#[derive(Debug, Clone)]
pub struct SdeMetricsExtended {
    /// Fréchet distance estimate (mean + covariance mismatch).
    pub frechet_distance: f64,
    /// Maximum mean discrepancy (polynomial kernel, degree 2).
    pub polynomial_mmd: f64,
    /// Energy distance between generated and real samples.
    pub energy_distance: f64,
    /// Sample autocorrelation at lag 1 (for time series quality).
    pub autocorr_lag1: f64,
}

impl SdeMetricsExtended {
    /// Compute extended metrics given real and generated sample sets.
    ///
    /// `real`: real data samples `[n_real][x_dim]`
    /// `generated`: generated samples `[n_gen][x_dim]`
    pub fn compute(real: &[Vec<f64>], generated: &[Vec<f64>]) -> Self {
        Self {
            frechet_distance: Self::frechet_distance(real, generated),
            polynomial_mmd: Self::polynomial_mmd(real, generated),
            energy_distance: Self::energy_distance(real, generated),
            autocorr_lag1: Self::autocorr_lag1(generated),
        }
    }

    /// Estimate Fréchet distance via mean and diagonal covariance mismatch.
    ///
    /// FD ≈ ||μ_r - μ_g||^2 + Tr(Σ_r + Σ_g - 2*sqrt(Σ_r Σ_g))
    /// With diagonal Σ: last term = 2*Σ_r^{1/2}*Σ_g^{1/2} component-wise.
    fn frechet_distance(real: &[Vec<f64>], generated: &[Vec<f64>]) -> f64 {
        if real.is_empty() || generated.is_empty() {
            return 0.0;
        }
        let d = real[0].len();
        let mean_r = Self::mean(real, d);
        let mean_g = Self::mean(generated, d);
        let var_r = Self::variance(real, &mean_r);
        let var_g = Self::variance(generated, &mean_g);

        let mean_diff_sq: f64 = mean_r
            .iter()
            .zip(mean_g.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();

        let trace_term: f64 = var_r
            .iter()
            .zip(var_g.iter())
            .map(|(&vr, &vg)| vr + vg - 2.0 * (vr * vg).max(0.0).sqrt())
            .sum();

        (mean_diff_sq + trace_term).max(0.0)
    }

    /// Polynomial MMD with degree-2 kernel: k(x,y) = (1 + x·y/d)^2.
    fn polynomial_mmd(real: &[Vec<f64>], generated: &[Vec<f64>]) -> f64 {
        let n_r = real.len();
        let n_g = generated.len();
        if n_r == 0 || n_g == 0 {
            return 0.0;
        }
        let d = real[0].len().max(1) as f64;
        let kernel = |a: &[f64], b: &[f64]| -> f64 {
            let dot: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
            (1.0 + dot / d).powi(2)
        };

        let k_rr: f64 = if n_r >= 2 {
            let sum: f64 = real
                .iter()
                .enumerate()
                .flat_map(|(i, ri)| real.iter().enumerate().map(move |(j, rj)| (i, j, ri, rj)))
                .filter(|(i, j, _, _)| i != j)
                .map(|(_, _, ri, rj)| kernel(ri, rj))
                .sum();
            sum / (n_r * (n_r - 1)) as f64
        } else {
            kernel(&real[0], &real[0])
        };

        let k_gg: f64 = if n_g >= 2 {
            let sum: f64 = generated
                .iter()
                .enumerate()
                .flat_map(|(i, gi)| {
                    generated
                        .iter()
                        .enumerate()
                        .map(move |(j, gj)| (i, j, gi, gj))
                })
                .filter(|(i, j, _, _)| i != j)
                .map(|(_, _, gi, gj)| kernel(gi, gj))
                .sum();
            sum / (n_g * (n_g - 1)) as f64
        } else {
            kernel(&generated[0], &generated[0])
        };

        let k_rg: f64 = real
            .iter()
            .flat_map(|ri| generated.iter().map(move |gj| kernel(ri, gj)))
            .sum::<f64>()
            / (n_r * n_g) as f64;

        (k_rr + k_gg - 2.0 * k_rg).max(0.0)
    }

    /// Energy distance: E[||X - Y||] - 0.5*E[||X - X'||] - 0.5*E[||Y - Y'||].
    fn energy_distance(real: &[Vec<f64>], generated: &[Vec<f64>]) -> f64 {
        let n_r = real.len();
        let n_g = generated.len();
        if n_r == 0 || n_g == 0 {
            return 0.0;
        }
        let l2 = |a: &[f64], b: &[f64]| -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(&ai, &bi)| (ai - bi).powi(2))
                .sum::<f64>()
                .sqrt()
        };

        let e_rg: f64 = real
            .iter()
            .flat_map(|ri| generated.iter().map(move |gj| l2(ri, gj)))
            .sum::<f64>()
            / (n_r * n_g) as f64;

        let e_rr: f64 = if n_r >= 2 {
            let sum: f64 = real
                .iter()
                .enumerate()
                .flat_map(|(i, ri)| real.iter().enumerate().map(move |(j, rj)| (i, j, ri, rj)))
                .filter(|(i, j, _, _)| i != j)
                .map(|(_, _, ri, rj)| l2(ri, rj))
                .sum();
            sum / (n_r * (n_r - 1)) as f64
        } else {
            0.0
        };

        let e_gg: f64 = if n_g >= 2 {
            let sum: f64 = generated
                .iter()
                .enumerate()
                .flat_map(|(i, gi)| {
                    generated
                        .iter()
                        .enumerate()
                        .map(move |(j, gj)| (i, j, gi, gj))
                })
                .filter(|(i, j, _, _)| i != j)
                .map(|(_, _, gi, gj)| l2(gi, gj))
                .sum();
            sum / (n_g * (n_g - 1)) as f64
        } else {
            0.0
        };

        (2.0 * e_rg - e_rr - e_gg).max(0.0)
    }

    /// Lag-1 autocorrelation of a batch of samples (treats each sample as scalar via first dim).
    fn autocorr_lag1(samples: &[Vec<f64>]) -> f64 {
        let n = samples.len();
        if n < 2 {
            return 0.0;
        }
        // Use first dimension as scalar series
        let series: Vec<f64> = samples
            .iter()
            .map(|v| v.first().copied().unwrap_or(0.0))
            .collect();
        let mean = series.iter().sum::<f64>() / n as f64;
        let variance = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        if variance < 1e-12 {
            return 0.0;
        }
        let cov: f64 = series[..n - 1]
            .iter()
            .zip(series[1..].iter())
            .map(|(&a, &b)| (a - mean) * (b - mean))
            .sum::<f64>()
            / (n - 1) as f64;
        (cov / variance).clamp(-1.0, 1.0)
    }

    fn mean(samples: &[Vec<f64>], d: usize) -> Vec<f64> {
        let n = samples.len() as f64;
        let mut m = vec![0.0; d];
        for s in samples {
            for (i, &v) in s.iter().enumerate() {
                if i < d {
                    m[i] += v;
                }
            }
        }
        m.iter_mut().for_each(|v| *v /= n);
        m
    }

    fn variance(samples: &[Vec<f64>], mean: &[f64]) -> Vec<f64> {
        let n = samples.len() as f64;
        let d = mean.len();
        let mut var = vec![0.0; d];
        for s in samples {
            for (i, &v) in s.iter().enumerate() {
                if i < d {
                    let diff = v - mean[i];
                    var[i] += diff * diff;
                }
            }
        }
        var.iter_mut().for_each(|v| *v /= n.max(1.0));
        var
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    // super = advanced; super::super = neural_sde re-exporting everything
    use super::super::{sample_standard_normal, PathSignature};
    use super::*;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(99)
    }

    fn make_path(n: usize, d: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|t| (0..d).map(|i| (t + i) as f64 * 0.1).collect())
            .collect()
    }

    // ─── RoughPath tests ───────────────────────────────────────────────────────

    #[test]
    fn test_rough_path_new_signatures_length() {
        let path = make_path(5, 2);
        let rp = RoughPath::new(&path, 2);
        assert_eq!(rp.signatures.len(), 5, "one sig per time step");
    }

    #[test]
    fn test_rough_path_signature_at_finite() {
        let path = make_path(4, 2);
        let rp = RoughPath::new(&path, 2);
        for t in 0..4 {
            let sig = rp.signature_at(t);
            assert!(sig.iter().all(|v| v.is_finite()), "sig at t={} finite", t);
        }
    }

    #[test]
    fn test_rough_path_levy_area_antisymmetric() {
        let path = make_path(5, 3);
        let rp = RoughPath::new(&path, 2);
        let la_01 = rp.levy_area(4, 0, 1);
        let la_10 = rp.levy_area(4, 1, 0);
        // levy_area(i,j) = (S^{ij} - S^{ji})/2 = -levy_area(j,i)
        assert!((la_01 + la_10).abs() < 1e-8, "Lévy area antisymmetry");
    }

    #[test]
    fn test_rough_path_holder_variation_nonneg() {
        let path = make_path(6, 2);
        let rp = RoughPath::new(&path, 2);
        let hv = rp.holder_variation(2.0);
        assert!(hv >= 0.0, "Hölder variation non-negative");
    }

    #[test]
    fn test_rough_path_holder_variation_finite() {
        let path = make_path(6, 2);
        let rp = RoughPath::new(&path, 2);
        let hv = rp.holder_variation(1.5);
        assert!(hv.is_finite(), "Hölder variation finite");
    }

    // ─── SignatureTransform tests ──────────────────────────────────────────────

    #[test]
    fn test_signature_transform_output_dim_full() {
        let st = SignatureTransform::new(2, 2, false);
        // PathSignature::new(2, 2).signature_dim() = 1 + 2 + 4 = 7
        let ps = PathSignature::new(2, 2);
        assert_eq!(st.output_dim(), ps.signature_dim());
    }

    #[test]
    fn test_signature_transform_output_dim_log() {
        let d = 3;
        let st = SignatureTransform::new(d, 2, true);
        let expected = d + d * (d - 1) / 2;
        assert_eq!(st.output_dim(), expected);
    }

    #[test]
    fn test_signature_transform_forward_shape() {
        let st = SignatureTransform::new(2, 2, false);
        let path = make_path(4, 2);
        let out = st.forward(&path);
        assert_eq!(out.len(), st.output_dim());
    }

    #[test]
    fn test_signature_transform_log_forward_finite() {
        let st = SignatureTransform::new(3, 2, true);
        let path = make_path(5, 3);
        let out = st.forward(&path);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ─── LogSignatureLayer tests ───────────────────────────────────────────────

    #[test]
    fn test_log_signature_layer_output_shape() {
        let layer = LogSignatureLayer::new(2, 2, 8, 42);
        let path = make_path(4, 2);
        let out = layer.forward(&path);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_log_signature_layer_output_finite() {
        let layer = LogSignatureLayer::new(3, 2, 6, 42);
        let path = make_path(5, 3);
        let out = layer.forward(&path);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ─── ControlledSde tests ───────────────────────────────────────────────────

    #[test]
    fn test_controlled_sde_integrate_shape() {
        let csde = ControlledSde::new(2, 4, 8);
        let path = make_path(6, 2);
        let rp = RoughPath::new(&path, 2);
        let z0 = vec![0.0; 4];
        let traj = csde.integrate(&z0, &rp, 0.1);
        assert_eq!(traj.len(), rp.signatures.len() + 1);
        assert_eq!(traj[0].len(), 4);
    }

    #[test]
    fn test_controlled_sde_integrate_finite() {
        let csde = ControlledSde::new(2, 4, 8);
        let path = make_path(5, 2);
        let rp = RoughPath::new(&path, 2);
        let z0 = vec![0.1; 4];
        let traj = csde.integrate(&z0, &rp, 0.1);
        assert!(traj.iter().all(|z| z.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn test_controlled_sde_initial_state_preserved() {
        let csde = ControlledSde::new(2, 3, 6);
        let path = make_path(4, 2);
        let rp = RoughPath::new(&path, 2);
        let z0 = vec![1.0, -1.0, 0.5];
        let traj = csde.integrate(&z0, &rp, 0.05);
        assert_eq!(traj[0], z0, "initial state must be z0");
    }

    // ─── NeuralRde tests ──────────────────────────────────────────────────────

    #[test]
    fn test_neural_rde_forward_shape() {
        let rde = NeuralRde::new(2, 4, 3, 2, 8);
        let path = make_path(5, 2);
        let out = rde.forward(&path, 0.1);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_neural_rde_forward_finite() {
        let rde = NeuralRde::new(2, 4, 3, 2, 8);
        let path = make_path(5, 2);
        let out = rde.forward(&path, 0.1);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ─── VpSde tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_vp_sde_beta_range() {
        let sde = VpSde::new(0.1, 20.0, 1.0);
        assert!((sde.beta(0.0) - 0.1).abs() < 1e-8);
        assert!((sde.beta(1.0) - 20.0).abs() < 1e-8);
    }

    #[test]
    fn test_vp_sde_sigma_positive() {
        let sde = VpSde::new(0.1, 20.0, 1.0);
        for t in [0.1, 0.3, 0.5, 0.7, 0.9] {
            assert!(sde.sigma(t) > 0.0, "sigma({}) must be positive", t);
        }
    }

    #[test]
    fn test_vp_sde_forward_diffuse_shape() {
        let mut rng = make_rng();
        let sde = VpSde::new(0.1, 20.0, 1.0);
        let x0 = vec![1.0, 2.0, 3.0];
        let (x_t, noise) = sde.forward_diffuse(&x0, 0.5, &mut rng);
        assert_eq!(x_t.len(), 3);
        assert_eq!(noise.len(), 3);
    }

    #[test]
    fn test_vp_sde_forward_diffuse_finite() {
        let mut rng = make_rng();
        let sde = VpSde::new(0.1, 20.0, 1.0);
        let x0 = vec![0.5, -0.5];
        let (x_t, noise) = sde.forward_diffuse(&x0, 0.7, &mut rng);
        assert!(x_t.iter().all(|v| v.is_finite()));
        assert!(noise.iter().all(|v| v.is_finite()));
    }

    // ─── VeSde tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_ve_sde_sigma_monotone() {
        let sde = VeSde::new(0.01, 50.0);
        assert!(sde.sigma(0.0) < sde.sigma(0.5));
        assert!(sde.sigma(0.5) < sde.sigma(1.0));
    }

    #[test]
    fn test_ve_sde_score_direction() {
        let sde = VeSde::new(0.01, 50.0);
        let x0 = vec![1.0, 0.0];
        let x_t = vec![2.0, 1.0]; // noisy version
        let score = sde.score(&x_t, &x0, 0.5);
        // Score should point from x_t toward x0
        let expected_sign_0 = (x0[0] - x_t[0]).signum();
        assert_eq!(score[0].signum(), expected_sign_0);
    }

    // ─── ScoreMatchingSde tests ────────────────────────────────────────────────

    #[test]
    fn test_score_matching_sde_score_shape() {
        let sm = ScoreMatchingSde::new(4, 16, 10);
        let x_t = vec![0.1, -0.2, 0.3, -0.4];
        let score = sm.score(&x_t, 0.5);
        assert_eq!(score.len(), 4);
    }

    #[test]
    fn test_score_matching_sde_dsm_loss_positive() {
        let mut rng = make_rng();
        let sm = ScoreMatchingSde::new(3, 12, 10);
        let batch: Vec<Vec<f64>> = (0..5)
            .map(|i| (0..3).map(|j| (i + j) as f64 * 0.1).collect())
            .collect();
        let loss = sm.dsm_loss(&batch, &mut rng);
        assert!(loss >= 0.0, "DSM loss must be non-negative");
        assert!(loss.is_finite(), "DSM loss must be finite");
    }

    // ─── AncestralSampler tests ────────────────────────────────────────────────

    #[test]
    fn test_ancestral_sampler_sample_shape() {
        let mut rng = make_rng();
        let sm = ScoreMatchingSde::new(3, 8, 10);
        let sampler = AncestralSampler::new(sm, 10);
        let sample = sampler.sample(&mut rng);
        assert_eq!(sample.len(), 3);
    }

    #[test]
    fn test_ancestral_sampler_sample_finite() {
        let mut rng = make_rng();
        let sm = ScoreMatchingSde::new(2, 8, 10);
        let sampler = AncestralSampler::new(sm, 5);
        let sample = sampler.sample(&mut rng);
        assert!(sample.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ancestral_sampler_batch_size() {
        let mut rng = make_rng();
        let sm = ScoreMatchingSde::new(2, 8, 5);
        let sampler = AncestralSampler::new(sm, 5);
        let batch = sampler.sample_batch(4, &mut rng);
        assert_eq!(batch.len(), 4);
        assert!(batch.iter().all(|s| s.len() == 2));
    }

    // ─── SdeMetricsExtended tests ──────────────────────────────────────────────

    #[test]
    fn test_sde_metrics_extended_frechet_same_dist() {
        let samples: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64 * 0.1, i as f64 * 0.2])
            .collect();
        let m = SdeMetricsExtended::compute(&samples, &samples);
        // Same distribution: Fréchet distance should be 0
        assert!(
            m.frechet_distance.abs() < 1e-8,
            "FD must be 0 for same dist, got {}",
            m.frechet_distance
        );
    }

    #[test]
    fn test_sde_metrics_extended_polynomial_mmd_nonneg() {
        let real: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64 * 0.1]).collect();
        let gen: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64 * 0.2 + 1.0]).collect();
        let m = SdeMetricsExtended::compute(&real, &gen);
        assert!(m.polynomial_mmd >= 0.0);
    }

    #[test]
    fn test_sde_metrics_extended_energy_distance_nonneg() {
        let real: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64]).collect();
        let gen: Vec<Vec<f64>> = (0..6).map(|i| vec![-(i as f64)]).collect();
        let m = SdeMetricsExtended::compute(&real, &gen);
        assert!(m.energy_distance >= 0.0);
    }

    #[test]
    fn test_sde_metrics_extended_autocorr_range() {
        let samples: Vec<Vec<f64>> = (0..20).map(|i| vec![(i as f64 * 0.5).sin()]).collect();
        let m = SdeMetricsExtended::compute(&samples, &samples);
        assert!(m.autocorr_lag1 >= -1.0 && m.autocorr_lag1 <= 1.0);
    }

    #[test]
    fn test_sde_metrics_extended_all_finite() {
        let real: Vec<Vec<f64>> = (0..8)
            .map(|i| vec![i as f64 * 0.1, i as f64 * 0.2])
            .collect();
        let gen: Vec<Vec<f64>> = (0..6)
            .map(|i| vec![i as f64 * 0.15, -i as f64 * 0.1])
            .collect();
        let m = SdeMetricsExtended::compute(&real, &gen);
        assert!(m.frechet_distance.is_finite());
        assert!(m.polynomial_mmd.is_finite());
        assert!(m.energy_distance.is_finite());
        assert!(m.autocorr_lag1.is_finite());
    }
}
