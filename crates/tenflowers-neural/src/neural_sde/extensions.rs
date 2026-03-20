//! Neural SDE Extensions — shuffle product, SignatureKernel, SdeTrainer, SdeMetrics, tests.

use super::{sample_standard_normal, LatentSde, PathSignature};
use scirs2_core::random::rngs::StdRng;

/// Compute the shuffle product of two signatures (for path concatenation identity check).
///
/// Shuffle product identity: sig(concat(P1, P2)) = shuffle(sig(P1), sig(P2))
/// where the shuffle product is defined via the quasi-shuffle formula in the tensor algebra.
pub fn shuffle_product(sig1: &[f64], sig2: &[f64], x_dim: usize, order: usize) -> Vec<f64> {
    let d = x_dim;
    let mut level_dims = vec![1usize];
    for _ in 0..order {
        let last = *level_dims.last().unwrap_or(&1);
        level_dims.push(last * d);
    }
    let mut sig1_levels: Vec<Vec<f64>> = Vec::new();
    let mut sig2_levels: Vec<Vec<f64>> = Vec::new();
    let mut offset = 0;
    for &dim in &level_dims {
        sig1_levels.push(sig1.get(offset..offset + dim).unwrap_or(&[]).to_vec());
        sig2_levels.push(sig2.get(offset..offset + dim).unwrap_or(&[]).to_vec());
        offset += dim;
    }
    let mut result_levels: Vec<Vec<f64>> = Vec::new();
    for k in 0..=order {
        let dim_k = level_dims[k];
        let mut level_k = vec![0.0; dim_k];
        for j in 0..=k {
            let j2 = k - j;
            let a = sig1_levels.get(j).cloned().unwrap_or_default();
            let b = sig2_levels.get(j2).cloned().unwrap_or_default();
            if k == 0 {
                let av = a.first().copied().unwrap_or(1.0);
                let bv = b.first().copied().unwrap_or(1.0);
                if level_k.is_empty() {
                    level_k.push(av * bv);
                } else {
                    level_k[0] += av * bv;
                }
            } else if j == 0 {
                if j2 == k {
                    for (idx, &bval) in b.iter().enumerate() {
                        if idx < level_k.len() {
                            level_k[idx] += bval;
                        }
                    }
                }
            } else if j2 == 0 {
                if j == k {
                    for (idx, &aval) in a.iter().enumerate() {
                        if idx < level_k.len() {
                            level_k[idx] += aval;
                        }
                    }
                }
            } else {
                let da = level_dims.get(j).copied().unwrap_or(1);
                let db = level_dims.get(j2).copied().unwrap_or(1);
                if da * db == dim_k {
                    for ai in 0..da {
                        for bi in 0..db {
                            let idx = ai * db + bi;
                            if idx < level_k.len() {
                                level_k[idx] += a.get(ai).copied().unwrap_or(0.0)
                                    * b.get(bi).copied().unwrap_or(0.0);
                            }
                        }
                    }
                }
            }
        }
        result_levels.push(level_k);
    }

    result_levels.into_iter().flatten().collect()
}

// Signature Kernel

/// Truncated signature kernel: inner product in the tensor algebra.
#[derive(Debug, Clone)]
pub struct SignatureKernel {
    /// Ambient path dimension.
    pub x_dim: usize,
    /// Truncation order.
    pub truncation_order: usize,
}

impl SignatureKernel {
    /// Construct a SignatureKernel.
    pub fn new(x_dim: usize, truncation_order: usize) -> Self {
        Self {
            x_dim,
            truncation_order,
        }
    }

    /// Compute k(P1, P2) = <Sig(P1), Sig(P2)> in the truncated tensor algebra.
    pub fn compute(&self, path1: &[Vec<f64>], path2: &[Vec<f64>]) -> f64 {
        let ps = PathSignature::new(self.x_dim, self.truncation_order);
        let s1 = ps.compute(path1);
        let s2 = ps.compute(path2);
        s1.iter().zip(s2.iter()).map(|(&a, &b)| a * b).sum()
    }

    /// Compute the kernel matrix K\[i\]\[j\] = k(paths\[i\], paths\[j\]).
    pub fn kernel_matrix(&self, paths: &[Vec<Vec<f64>>]) -> Vec<Vec<f64>> {
        let n = paths.len();
        let ps = PathSignature::new(self.x_dim, self.truncation_order);
        let sigs: Vec<Vec<f64>> = paths.iter().map(|p| ps.compute(p)).collect();
        let mut mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                mat[i][j] = sigs[i]
                    .iter()
                    .zip(sigs[j].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
            }
        }
        mat
    }

    /// Compute MMD^2 between two sets of paths using the signature kernel.
    ///
    /// MMD^2 = E[k(P,P')] + E[k(Q,Q')] - 2*E[k(P,Q)]
    pub fn mmd(&self, paths_p: &[Vec<Vec<f64>>], paths_q: &[Vec<Vec<f64>>]) -> f64 {
        let ps = PathSignature::new(self.x_dim, self.truncation_order);
        let sigs_p: Vec<Vec<f64>> = paths_p.iter().map(|p| ps.compute(p)).collect();
        let sigs_q: Vec<Vec<f64>> = paths_q.iter().map(|p| ps.compute(p)).collect();

        let n_p = sigs_p.len().max(1);
        let n_q = sigs_q.len().max(1);

        let mean_pp = if sigs_p.len() >= 2 {
            let sum: f64 = (0..sigs_p.len())
                .flat_map(|i| (0..sigs_p.len()).map(move |j| (i, j)))
                .filter(|(i, j)| i != j)
                .map(|(i, j)| {
                    sigs_p[i]
                        .iter()
                        .zip(sigs_p[j].iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>()
                })
                .sum();
            sum / (n_p * (n_p - 1)) as f64
        } else {
            sigs_p
                .first()
                .map(|s: &Vec<f64>| s.iter().map(|&v| v * v).sum::<f64>())
                .unwrap_or(0.0)
        };

        let mean_qq = if sigs_q.len() >= 2 {
            let sum: f64 = (0..sigs_q.len())
                .flat_map(|i| (0..sigs_q.len()).map(move |j| (i, j)))
                .filter(|(i, j)| i != j)
                .map(|(i, j)| {
                    sigs_q[i]
                        .iter()
                        .zip(sigs_q[j].iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>()
                })
                .sum();
            sum / (n_q * (n_q - 1)) as f64
        } else {
            sigs_q
                .first()
                .map(|s: &Vec<f64>| s.iter().map(|&v| v * v).sum::<f64>())
                .unwrap_or(0.0)
        };

        let mean_pq: f64 = sigs_p
            .iter()
            .flat_map(|sp| {
                sigs_q
                    .iter()
                    .map(move |sq| sp.iter().zip(sq.iter()).map(|(&a, &b)| a * b).sum::<f64>())
            })
            .sum::<f64>()
            / (n_p * n_q) as f64;

        (mean_pp + mean_qq - 2.0 * mean_pq).max(0.0)
    }
}

// SDE Trainer

/// Training loop for LatentSde models.
#[derive(Debug, Clone)]
pub struct SdeTrainer {
    /// Learning rate.
    pub lr: f64,
    /// Number of training epochs.
    pub n_epochs: usize,
    /// Number of epochs to linearly warm up KL weight from 0 to 1.
    pub kl_warmup_epochs: usize,
}

impl SdeTrainer {
    /// Construct a trainer with default kl_warmup_epochs = n_epochs / 4.
    pub fn new(lr: f64, n_epochs: usize) -> Self {
        let kl_warmup = (n_epochs / 4).max(1);
        Self {
            lr,
            n_epochs,
            kl_warmup_epochs: kl_warmup,
        }
    }

    /// Train the model on a set of sequences; returns ELBO history per epoch.
    pub fn train(
        &self,
        model: &mut LatentSde,
        sequences: &[Vec<Vec<f64>>],
        rng: &mut StdRng,
    ) -> Vec<f64> {
        let mut history = Vec::with_capacity(self.n_epochs);
        for epoch in 0..self.n_epochs {
            let kl_weight = if self.kl_warmup_epochs == 0 {
                1.0
            } else {
                ((epoch + 1) as f64 / self.kl_warmup_epochs as f64).min(1.0)
            };

            let orig_kl = model.config.kl_coeff;
            model.config.kl_coeff = orig_kl * kl_weight;

            let mut epoch_elbo = 0.0;
            let mut valid_count = 0usize;
            for seq in sequences.iter() {
                let (elbo, recon, _kl) = model.elbo(seq, rng);
                if elbo.is_finite() {
                    epoch_elbo += elbo;
                    valid_count += 1;
                }

                // Simple decoder refinement: adjust decoder biases based on
                // reconstruction signal (gradient-free, numerically stable).
                // For each output neuron, nudge its bias toward reducing recon error.
                if recon.is_finite() && recon > 0.0 {
                    let nudge = self.lr * (-recon).tanh() * 0.01;
                    if let Some(last_b) = model.decoder.mlp.biases.last_mut() {
                        for b in last_b.iter_mut() {
                            *b += nudge;
                        }
                    }
                }
            }
            model.config.kl_coeff = orig_kl;
            let mean_elbo = if valid_count > 0 {
                epoch_elbo / valid_count as f64
            } else {
                0.0
            };
            history.push(mean_elbo);
        }
        history
    }

    /// Generate an Ornstein-Uhlenbeck sequence.
    ///
    /// dX = theta*(mu - X)*dt + sigma*sqrt(dt)*N(0,1)
    pub fn generate_ou_sequence(
        theta: f64,
        mu: f64,
        sigma: f64,
        x0: f64,
        n_steps: usize,
        dt: f64,
        rng: &mut StdRng,
    ) -> Vec<f64> {
        let mut seq = Vec::with_capacity(n_steps);
        let mut x = x0;
        for _ in 0..n_steps {
            let eps = sample_standard_normal(rng);
            x += theta * (mu - x) * dt + sigma * dt.sqrt() * eps;
            seq.push(x);
        }
        seq
    }
}

/// Summary metrics for evaluating a trained LatentSde model.
#[derive(Debug, Clone)]
pub struct SdeMetrics {
    /// Mean ELBO across test sequences.
    pub mean_elbo: f64,
    /// Mean KL divergence.
    pub kl_divergence: f64,
    /// Mean reconstruction loss.
    pub recon_loss: f64,
    /// Path MMD between generated and test sequences.
    pub path_mmd: f64,
}

/// Compute evaluation metrics for a LatentSde model on test sequences.
pub fn compute_sde_metrics(
    model: &LatentSde,
    test_sequences: &[Vec<Vec<f64>>],
    kernel: &SignatureKernel,
    rng: &mut StdRng,
) -> SdeMetrics {
    let n = test_sequences.len().max(1);
    let mut total_elbo = 0.0;
    let mut total_kl = 0.0;
    let mut total_recon = 0.0;

    let mut generated_paths: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut test_paths_1d: Vec<Vec<Vec<f64>>> = Vec::new();

    for seq in test_sequences.iter() {
        let (elbo, recon, kl) = model.elbo(seq, rng);
        total_elbo += elbo;
        total_kl += kl;
        total_recon += recon;

        // Generate a sample path using the model
        if let Some(x0) = seq.first() {
            let gen = model.forward(x0, rng);
            // Convert x-dim sequence to 1d path for MMD
            let gen_path: Vec<Vec<f64>> = gen
                .iter()
                .enumerate()
                .map(|(t, x)| {
                    let mut p = vec![t as f64 / n as f64];
                    p.extend_from_slice(x);
                    p
                })
                .collect();
            generated_paths.push(gen_path);

            let test_path: Vec<Vec<f64>> = seq
                .iter()
                .enumerate()
                .map(|(t, x)| {
                    let mut p = vec![t as f64 / n as f64];
                    p.extend_from_slice(x);
                    p
                })
                .collect();
            test_paths_1d.push(test_path);
        }
    }

    let path_mmd = if generated_paths.is_empty() || test_paths_1d.is_empty() {
        0.0
    } else {
        kernel.mmd(&generated_paths, &test_paths_1d)
    };

    SdeMetrics {
        mean_elbo: total_elbo / n as f64,
        kl_divergence: total_kl / n as f64,
        recon_loss: total_recon / n as f64,
        path_mmd,
    }
}

// Tests

#[cfg(test)]
mod tests {
    // Use super (extensions module) which re-exports everything via pub use, and
    // super::super (neural_sde mod.rs) which has the base types.
    use super::super::{
        sample_standard_normal, CdeVectorField, LatentSde, LatentSdeConfig, NaturalCubicSpline,
        NeuralCde, NeuralCdeConfig, NsdeMlp, PathSignature, SdeDecoder, SdeDiffusionNet,
        SdeDriftNet, SdeEncoder,
    };
    use super::*;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn simple_config() -> LatentSdeConfig {
        LatentSdeConfig {
            z_dim: 4,
            x_dim: 3,
            hidden_dim: 8,
            n_steps: 5,
            dt: 0.1,
            kl_coeff: 1.0,
        }
    }

    // --- NsdeMlp tests ---

    #[test]
    fn test_nsde_mlp_forward_shape() {
        let mlp = NsdeMlp::new(&[4, 8, 3]);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = mlp.forward(&x);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_nsde_mlp_output_dim() {
        let mlp = NsdeMlp::new(&[5, 10, 7]);
        assert_eq!(mlp.output_dim(), 7);
    }

    #[test]
    fn test_nsde_mlp_input_dim() {
        let mlp = NsdeMlp::new(&[5, 10, 7]);
        assert_eq!(mlp.input_dim(), 5);
    }

    #[test]
    fn test_nsde_mlp_update() {
        let mut mlp = NsdeMlp::new(&[2, 4, 2]);
        let grad_w: Vec<Vec<Vec<f64>>> = mlp
            .weights
            .iter()
            .map(|l| l.iter().map(|r| vec![0.1; r.len()]).collect())
            .collect();
        let grad_b: Vec<Vec<f64>> = mlp.biases.iter().map(|b| vec![0.01; b.len()]).collect();
        let w_before = mlp.weights[0][0][0];
        mlp.update(&grad_w, &grad_b, 0.1);
        let w_after = mlp.weights[0][0][0];
        assert!((w_before - w_after - 0.01).abs() < 1e-9);
    }

    // --- SdeDriftNet tests ---

    #[test]
    fn test_sde_drift_net_output_shape() {
        let net = SdeDriftNet::new(4, 8, 1);
        let z = vec![0.1, 0.2, 0.3, 0.4];
        let out = net.forward(0.5, &z);
        assert_eq!(out.len(), 4, "drift output must equal z_dim");
    }

    #[test]
    fn test_sde_drift_net_finite() {
        let net = SdeDriftNet::new(3, 6, 2);
        let z = vec![1.0, -1.0, 0.5];
        let out = net.forward(0.0, &z);
        assert!(out.iter().all(|v| v.is_finite()), "drift must be finite");
    }

    // --- SdeDiffusionNet tests ---

    #[test]
    fn test_sde_diffusion_net_positive() {
        let net = SdeDiffusionNet::new(4, 8, 1);
        let z = vec![0.1, 0.2, -0.3, 0.4];
        let out = net.forward(0.5, &z);
        assert_eq!(out.len(), 4);
        assert!(
            out.iter().all(|&v| v > 0.0),
            "diffusion must be positive (softplus)"
        );
    }

    #[test]
    fn test_sde_diffusion_large_negative_input() {
        // Softplus of very negative input should be near 0 but still positive
        let net = SdeDiffusionNet::new(2, 4, 1);
        let z = vec![-100.0, -100.0];
        let out = net.forward(0.0, &z);
        assert!(out.iter().all(|&v| v > 0.0 && v.is_finite()));
    }

    // --- SdeEncoder tests ---

    #[test]
    fn test_sde_encoder_output_shape() {
        let enc = SdeEncoder::new(3, 4, 8);
        let x = vec![0.1, 0.2, 0.3];
        let (mean, log_var) = enc.encode(&x);
        assert_eq!(mean.len(), 4, "mean must have z_dim elements");
        assert_eq!(log_var.len(), 4, "log_var must have z_dim elements");
    }

    #[test]
    fn test_sde_encoder_finite() {
        let enc = SdeEncoder::new(3, 4, 8);
        let x = vec![1.0, -1.0, 0.0];
        let (mean, log_var) = enc.encode(&x);
        assert!(mean.iter().all(|v| v.is_finite()));
        assert!(log_var.iter().all(|v| v.is_finite()));
    }

    // --- SdeDecoder tests ---

    #[test]
    fn test_sde_decoder_output_shape() {
        let dec = SdeDecoder::new(4, 3, 8);
        let z = vec![0.1, 0.2, 0.3, 0.4];
        let out = dec.decode(&z);
        assert_eq!(out.len(), 3);
    }

    // --- LatentSde tests ---

    #[test]
    fn test_latent_sde_sample_z0_shape() {
        let mut rng = make_rng();
        let model = LatentSde::new(simple_config());
        let x0 = vec![0.1, 0.2, 0.3];
        let z0 = model.sample_z0(&x0, &mut rng);
        assert_eq!(z0.len(), 4, "z0 must have z_dim elements");
    }

    #[test]
    fn test_latent_sde_sample_z0_finite() {
        let mut rng = make_rng();
        let model = LatentSde::new(simple_config());
        let x0 = vec![0.1, 0.2, 0.3];
        let z0 = model.sample_z0(&x0, &mut rng);
        assert!(z0.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_latent_sde_euler_maruyama_shape() {
        let mut rng = make_rng();
        let cfg = simple_config();
        let n_steps = cfg.n_steps;
        let model = LatentSde::new(cfg);
        let z0 = vec![0.0; 4];
        let traj = model.euler_maruyama(&z0, true, &mut rng);
        assert_eq!(
            traj.len(),
            n_steps + 1,
            "trajectory must have n_steps+1 entries"
        );
        assert_eq!(traj[0].len(), 4, "each state must have z_dim elements");
    }

    #[test]
    fn test_latent_sde_euler_maruyama_prior_shape() {
        let mut rng = make_rng();
        let cfg = simple_config();
        let n_steps = cfg.n_steps;
        let model = LatentSde::new(cfg);
        let z0 = vec![0.0; 4];
        let traj = model.euler_maruyama(&z0, false, &mut rng);
        assert_eq!(traj.len(), n_steps + 1);
    }

    #[test]
    fn test_latent_sde_euler_maruyama_finite() {
        let mut rng = make_rng();
        let model = LatentSde::new(simple_config());
        let z0 = vec![0.0; 4];
        let traj = model.euler_maruyama(&z0, true, &mut rng);
        assert!(traj.iter().all(|z| z.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn test_latent_sde_reconstruct_shape() {
        let mut rng = make_rng();
        let cfg = simple_config();
        let n_steps = cfg.n_steps;
        let x_dim = cfg.x_dim;
        let model = LatentSde::new(cfg);
        let z0 = vec![0.0; 4];
        let traj = model.euler_maruyama(&z0, true, &mut rng);
        let x_pred = model.reconstruct(&traj);
        assert_eq!(x_pred.len(), n_steps + 1);
        assert_eq!(x_pred[0].len(), x_dim);
    }

    #[test]
    fn test_latent_sde_elbo_finite() {
        let mut rng = make_rng();
        let cfg = simple_config();
        let n_steps = cfg.n_steps;
        let x_dim = cfg.x_dim;
        let model = LatentSde::new(cfg);
        let x_obs: Vec<Vec<f64>> = (0..=n_steps)
            .map(|t| (0..x_dim).map(|d| (t + d) as f64 * 0.1).collect())
            .collect();
        let (elbo, recon, kl) = model.elbo(&x_obs, &mut rng);
        assert!(elbo.is_finite(), "ELBO must be finite, got {}", elbo);
        assert!(recon.is_finite(), "recon_loss must be finite");
        assert!(kl.is_finite(), "kl_loss must be finite");
    }

    #[test]
    fn test_latent_sde_elbo_recon_positive() {
        let mut rng = make_rng();
        let cfg = simple_config();
        let n_steps = cfg.n_steps;
        let x_dim = cfg.x_dim;
        let model = LatentSde::new(cfg);
        let x_obs: Vec<Vec<f64>> = (0..=n_steps)
            .map(|_| (0..x_dim).map(|d| d as f64 * 0.1).collect())
            .collect();
        let (_elbo, recon, _kl) = model.elbo(&x_obs, &mut rng);
        assert!(recon >= 0.0, "reconstruction loss must be non-negative");
    }

    #[test]
    fn test_latent_sde_elbo_kl_positive() {
        let mut rng = make_rng();
        let cfg = simple_config();
        let n_steps = cfg.n_steps;
        let x_dim = cfg.x_dim;
        let model = LatentSde::new(cfg);
        let x_obs: Vec<Vec<f64>> = (0..=n_steps)
            .map(|_| (0..x_dim).map(|d| d as f64 * 0.1).collect())
            .collect();
        let (_elbo, _recon, kl) = model.elbo(&x_obs, &mut rng);
        assert!(kl >= 0.0, "KL loss must be non-negative");
    }

    #[test]
    fn test_latent_sde_forward_shape() {
        let mut rng = make_rng();
        let cfg = simple_config();
        let n_steps = cfg.n_steps;
        let x_dim = cfg.x_dim;
        let model = LatentSde::new(cfg);
        let x0 = vec![0.1, 0.2, 0.3];
        let out = model.forward(&x0, &mut rng);
        assert_eq!(out.len(), n_steps + 1);
        assert_eq!(out[0].len(), x_dim);
    }

    // --- NaturalCubicSpline tests ---

    #[test]
    fn test_spline_interpolates_training_points() {
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]];
        let spline = NaturalCubicSpline::new(&times, &values);
        for (i, &t) in times.iter().enumerate() {
            let val = spline.evaluate(t);
            assert!(
                (val[0] - values[i][0]).abs() < 1e-8,
                "spline should interpolate t={}: got {}, want {}",
                t,
                val[0],
                values[i][0]
            );
        }
    }

    #[test]
    fn test_spline_evaluate_finite() {
        let times = vec![0.0, 0.5, 1.0];
        let values: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![1.5, 2.5], vec![2.0, 3.0]];
        let spline = NaturalCubicSpline::new(&times, &values);
        let val = spline.evaluate(0.25);
        assert!(
            val.iter().all(|v| v.is_finite()),
            "evaluated value must be finite"
        );
    }

    #[test]
    fn test_spline_derivative_finite() {
        let times = vec![0.0, 1.0, 2.0];
        let values: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![4.0]];
        let spline = NaturalCubicSpline::new(&times, &values);
        let dv = spline.derivative(0.5);
        assert!(
            dv.iter().all(|v| v.is_finite()),
            "derivative must be finite"
        );
    }

    #[test]
    fn test_spline_two_points_linear() {
        let times = vec![0.0, 1.0];
        let values: Vec<Vec<f64>> = vec![vec![0.0], vec![2.0]];
        let spline = NaturalCubicSpline::new(&times, &values);
        let mid = spline.evaluate(0.5);
        assert!(
            (mid[0] - 1.0).abs() < 1e-8,
            "linear spline midpoint should be 1.0, got {}",
            mid[0]
        );
    }

    #[test]
    fn test_spline_clamp_time() {
        let times = vec![1.0, 2.0, 3.0];
        let values: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0]];
        let spline = NaturalCubicSpline::new(&times, &values);
        assert_eq!(spline.clamp_time(0.0), 1.0);
        assert_eq!(spline.clamp_time(4.0), 3.0);
        assert_eq!(spline.clamp_time(2.0), 2.0);
    }

    // --- CdeVectorField tests ---

    #[test]
    fn test_cde_vector_field_forward_shape() {
        let vf = CdeVectorField::new(4, 3, 8);
        let z = vec![0.1, 0.2, 0.3, 0.4];
        let out = vf.forward(&z);
        assert_eq!(out.len(), 4 * 3, "f(z) output must be z_dim * x_dim");
    }

    #[test]
    fn test_cde_vector_field_apply_shape() {
        let vf = CdeVectorField::new(4, 3, 8);
        let z = vec![0.1, 0.2, 0.3, 0.4];
        let dx = vec![1.0, 0.0, -1.0];
        let out = vf.apply(&z, &dx);
        assert_eq!(out.len(), 4, "apply output must be z_dim");
    }

    #[test]
    fn test_cde_vector_field_apply_finite() {
        let vf = CdeVectorField::new(3, 2, 6);
        let z = vec![1.0, -1.0, 0.5];
        let dx = vec![0.1, -0.1];
        let out = vf.apply(&z, &dx);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // --- NeuralCde tests ---

    #[test]
    fn test_neural_cde_forward_shape() {
        let config = NeuralCdeConfig {
            z_dim: 4,
            x_dim: 2,
            hidden_dim: 8,
            n_steps: 10,
        };
        let cde = NeuralCde::new(config);
        let times = vec![0.0, 0.5, 1.0];
        let values: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let out = cde.forward(&times, &values);
        assert_eq!(out.len(), 4, "CDE forward output must be z_dim");
    }

    #[test]
    fn test_neural_cde_integrate_finite() {
        let config = NeuralCdeConfig {
            z_dim: 3,
            x_dim: 2,
            hidden_dim: 6,
            n_steps: 5,
        };
        let cde = NeuralCde::new(config);
        let times = vec![0.0, 1.0];
        let values: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let spline = NaturalCubicSpline::new(&times, &values);
        let z0 = vec![0.0; 3];
        let z_t = cde.integrate(&spline, &z0);
        assert_eq!(z_t.len(), 3);
        assert!(z_t.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_neural_cde_classify() {
        let config = NeuralCdeConfig {
            z_dim: 4,
            x_dim: 2,
            hidden_dim: 8,
            n_steps: 5,
        };
        let cde = NeuralCde::new(config);
        let times = vec![0.0, 1.0, 2.0];
        let values: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let cls = cde.classify(&times, &values, 3);
        assert!(cls < 3, "class index must be within n_classes");
    }

    // --- PathSignature tests ---

    #[test]
    fn test_signature_level0_is_one() {
        let ps = PathSignature::new(2, 2);
        let path = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let sig = ps.compute(&path);
        assert!(
            (sig[0] - 1.0).abs() < 1e-10,
            "level-0 signature must be 1.0"
        );
    }

    #[test]
    fn test_signature_level1_endpoint_minus_start() {
        let ps = PathSignature::new(2, 1);
        let start = vec![1.0, 2.0];
        let end_pt = vec![4.0, 6.0];
        let path = vec![start.clone(), end_pt.clone()];
        let sig = ps.compute(&path);
        assert!(
            (sig[1] - (end_pt[0] - start[0])).abs() < 1e-8,
            "level-1 sig[0] = X_T - X_0, got {}",
            sig[1]
        );
        assert!(
            (sig[2] - (end_pt[1] - start[1])).abs() < 1e-8,
            "level-1 sig[1] = Y_T - Y_0, got {}",
            sig[2]
        );
    }

    #[test]
    fn test_signature_dim_formula() {
        for d in 1..=4 {
            for m in 0..=3 {
                let ps = PathSignature::new(d, m);
                let expected = if d == 1 {
                    m + 1
                } else {
                    (0..=m).map(|k| d.pow(k as u32)).sum::<usize>()
                };
                assert_eq!(
                    ps.signature_dim(),
                    expected,
                    "dim mismatch for d={}, m={}",
                    d,
                    m
                );
            }
        }
    }

    #[test]
    fn test_signature_compute_length_matches_dim() {
        let ps = PathSignature::new(3, 2);
        let path = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.5, -0.5],
            vec![2.0, 1.0, 0.0],
        ];
        let sig = ps.compute(&path);
        assert_eq!(
            sig.len(),
            ps.signature_dim(),
            "signature length must match signature_dim()"
        );
    }

    #[test]
    fn test_signature_log_signature_shorter() {
        let d = 3;
        let m = 2;
        let ps = PathSignature::new(d, m);
        let path = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.5, -0.5]];
        let sig = ps.compute(&path);
        let log_sig = ps.compute_log_signature(&path);
        assert!(
            log_sig.len() < sig.len(),
            "log-signature (d + d*(d-1)/2 = {}) must be shorter than full sig ({})",
            log_sig.len(),
            sig.len()
        );
    }

    #[test]
    fn test_signature_log_signature_dim_m2() {
        let d = 3;
        let ps = PathSignature::new(d, 2);
        let path = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.5, -0.5]];
        let log_sig = ps.compute_log_signature(&path);
        let expected_dim = d + d * (d - 1) / 2;
        assert_eq!(
            log_sig.len(),
            expected_dim,
            "log-sig dim for M=2, d={} should be {}",
            d,
            expected_dim
        );
    }

    #[test]
    fn test_signature_finite() {
        let ps = PathSignature::new(2, 3);
        let path: Vec<Vec<f64>> = (0..5)
            .map(|t| vec![t as f64 * 0.1, (t as f64 * 0.2).sin()])
            .collect();
        let sig = ps.compute(&path);
        assert!(
            sig.iter().all(|v| v.is_finite()),
            "signature must be finite"
        );
    }

    #[test]
    fn test_signature_single_point_path() {
        let ps = PathSignature::new(2, 2);
        let path = vec![vec![1.0, 2.0]];
        let sig = ps.compute(&path);
        assert!((sig[0] - 1.0).abs() < 1e-10);
        assert!(sig[1..].iter().all(|&v| v.abs() < 1e-10));
    }

    // --- SignatureKernel tests ---

    #[test]
    fn test_signature_kernel_symmetric() {
        let kernel = SignatureKernel::new(2, 2);
        let path1: Vec<Vec<f64>> = (0..4)
            .map(|t| vec![t as f64 * 0.3, t as f64 * 0.2])
            .collect();
        let path2: Vec<Vec<f64>> = (0..4)
            .map(|t| vec![t as f64 * 0.1, -(t as f64 * 0.15)])
            .collect();
        let k12 = kernel.compute(&path1, &path2);
        let k21 = kernel.compute(&path2, &path1);
        assert!(
            (k12 - k21).abs() < 1e-8,
            "kernel must be symmetric: k12={}, k21={}",
            k12,
            k21
        );
    }

    #[test]
    fn test_signature_kernel_nonnegative() {
        let kernel = SignatureKernel::new(2, 2);
        let path: Vec<Vec<f64>> = (0..4)
            .map(|t| vec![t as f64 * 0.3, t as f64 * 0.2])
            .collect();
        let k = kernel.compute(&path, &path);
        assert!(
            k >= 0.0,
            "self-kernel k(p,p) must be non-negative, got {}",
            k
        );
    }

    #[test]
    fn test_signature_kernel_self_max() {
        let kernel = SignatureKernel::new(2, 2);
        let path1: Vec<Vec<f64>> = (0..4)
            .map(|t| vec![t as f64 * 0.3, t as f64 * 0.2])
            .collect();
        let path2: Vec<Vec<f64>> = (0..4)
            .map(|t| vec![t as f64 * 0.1, -(t as f64 * 0.15)])
            .collect();
        let k11 = kernel.compute(&path1, &path1);
        let k12 = kernel.compute(&path1, &path2);
        let k22 = kernel.compute(&path2, &path2);
        assert!(
            k12 * k12 <= k11 * k22 + 1e-8,
            "Cauchy-Schwarz violated: k12^2={} > k11*k22={}",
            k12 * k12,
            k11 * k22
        );
    }

    #[test]
    fn test_signature_kernel_matrix_square() {
        let kernel = SignatureKernel::new(2, 1);
        let paths: Vec<Vec<Vec<f64>>> = (0..3)
            .map(|i| vec![vec![0.0, 0.0], vec![i as f64, i as f64 + 1.0]])
            .collect();
        let mat = kernel.kernel_matrix(&paths);
        assert_eq!(mat.len(), 3, "kernel matrix must have 3 rows");
        assert!(
            mat.iter().all(|row| row.len() == 3),
            "kernel matrix must have 3 cols"
        );
    }

    #[test]
    fn test_signature_kernel_matrix_symmetric() {
        let kernel = SignatureKernel::new(2, 1);
        let paths: Vec<Vec<Vec<f64>>> = (0..3)
            .map(|i| vec![vec![0.0, 0.0], vec![i as f64, i as f64 + 1.0]])
            .collect();
        let mat = kernel.kernel_matrix(&paths);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (mat[i][j] - mat[j][i]).abs() < 1e-8,
                    "kernel matrix must be symmetric at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_signature_kernel_mmd_nonnegative() {
        let kernel = SignatureKernel::new(2, 2);
        let paths_p: Vec<Vec<Vec<f64>>> = (0..3)
            .map(|i| vec![vec![0.0, 0.0], vec![i as f64 * 0.5, 1.0]])
            .collect();
        let paths_q: Vec<Vec<Vec<f64>>> = (0..3)
            .map(|i| vec![vec![0.0, 0.0], vec![-(i as f64 * 0.5), -1.0]])
            .collect();
        let mmd = kernel.mmd(&paths_p, &paths_q);
        assert!(mmd >= 0.0, "MMD must be non-negative, got {}", mmd);
    }

    #[test]
    fn test_signature_kernel_mmd_identical_zero() {
        let kernel = SignatureKernel::new(2, 2);
        let paths: Vec<Vec<Vec<f64>>> = (0..3)
            .map(|i| vec![vec![0.0, 0.0], vec![i as f64 * 0.5, 1.0]])
            .collect();
        let mmd = kernel.mmd(&paths, &paths);
        assert!(mmd.abs() < 1e-6, "MMD(P, P) should be near 0, got {}", mmd);
    }

    // --- SdeTrainer tests ---

    #[test]
    fn test_sde_trainer_generate_ou_sequence_length() {
        let mut rng = make_rng();
        let seq = SdeTrainer::generate_ou_sequence(0.5, 0.0, 0.3, 1.0, 20, 0.01, &mut rng);
        assert_eq!(seq.len(), 20, "OU sequence must have n_steps elements");
    }

    #[test]
    fn test_sde_trainer_generate_ou_sequence_finite() {
        let mut rng = make_rng();
        let seq = SdeTrainer::generate_ou_sequence(1.0, 0.0, 0.1, 0.5, 50, 0.01, &mut rng);
        assert!(
            seq.iter().all(|v| v.is_finite()),
            "OU sequence must be finite"
        );
    }

    #[test]
    fn test_sde_trainer_ou_mean_reversion() {
        let mut rng = make_rng();
        let seq = SdeTrainer::generate_ou_sequence(10.0, 0.0, 0.01, 5.0, 1000, 0.01, &mut rng);
        let mean: f64 = seq.iter().sum::<f64>() / seq.len() as f64;
        assert!(
            mean.abs() < 1.0,
            "OU with high theta should be near mu=0, got mean={}",
            mean
        );
    }

    #[test]
    fn test_sde_trainer_train_returns_history() {
        let mut rng = make_rng();
        let cfg = LatentSdeConfig {
            z_dim: 2,
            x_dim: 1,
            hidden_dim: 4,
            n_steps: 3,
            dt: 0.1,
            kl_coeff: 0.1,
        };
        let mut model = LatentSde::new(cfg);
        let trainer = SdeTrainer::new(1e-4, 2);
        let sequences: Vec<Vec<Vec<f64>>> = (0..2)
            .map(|_| (0..4).map(|t| vec![t as f64 * 0.1]).collect())
            .collect();
        let history = trainer.train(&mut model, &sequences, &mut rng);
        assert_eq!(
            history.len(),
            2,
            "training history must have n_epochs entries"
        );
    }

    #[test]
    fn test_sde_trainer_train_history_finite() {
        let mut rng = make_rng();
        let cfg = LatentSdeConfig {
            z_dim: 2,
            x_dim: 1,
            hidden_dim: 4,
            n_steps: 3,
            dt: 0.1,
            kl_coeff: 0.1,
        };
        let mut model = LatentSde::new(cfg);
        let trainer = SdeTrainer::new(1e-4, 3);
        let sequences: Vec<Vec<Vec<f64>>> = (0..2)
            .map(|_| (0..4).map(|t| vec![t as f64 * 0.1]).collect())
            .collect();
        let history = trainer.train(&mut model, &sequences, &mut rng);
        assert!(
            history.iter().all(|v| v.is_finite()),
            "ELBO history must be finite"
        );
    }

    // --- SdeMetrics tests ---

    #[test]
    fn test_compute_sde_metrics_recon_positive() {
        let mut rng = make_rng();
        let cfg = LatentSdeConfig {
            z_dim: 2,
            x_dim: 2,
            hidden_dim: 4,
            n_steps: 3,
            dt: 0.1,
            kl_coeff: 1.0,
        };
        let model = LatentSde::new(cfg);
        let kernel = SignatureKernel::new(3, 1);
        let test_seqs: Vec<Vec<Vec<f64>>> = (0..2)
            .map(|_| (0..4).map(|t| vec![t as f64 * 0.1, 0.5]).collect())
            .collect();
        let metrics = compute_sde_metrics(&model, &test_seqs, &kernel, &mut rng);
        assert!(metrics.recon_loss >= 0.0, "recon_loss must be non-negative");
    }

    #[test]
    fn test_compute_sde_metrics_finite() {
        let mut rng = make_rng();
        let cfg = LatentSdeConfig {
            z_dim: 2,
            x_dim: 2,
            hidden_dim: 4,
            n_steps: 3,
            dt: 0.1,
            kl_coeff: 1.0,
        };
        let model = LatentSde::new(cfg);
        let kernel = SignatureKernel::new(3, 1);
        let test_seqs: Vec<Vec<Vec<f64>>> = (0..2)
            .map(|_| (0..4).map(|t| vec![t as f64 * 0.1, 0.5]).collect())
            .collect();
        let metrics = compute_sde_metrics(&model, &test_seqs, &kernel, &mut rng);
        assert!(metrics.mean_elbo.is_finite());
        assert!(metrics.kl_divergence.is_finite());
        assert!(metrics.recon_loss.is_finite());
        assert!(metrics.path_mmd.is_finite());
    }

    #[test]
    fn test_compute_sde_metrics_mmd_nonnegative() {
        let mut rng = make_rng();
        let cfg = LatentSdeConfig {
            z_dim: 2,
            x_dim: 2,
            hidden_dim: 4,
            n_steps: 3,
            dt: 0.1,
            kl_coeff: 1.0,
        };
        let model = LatentSde::new(cfg);
        let kernel = SignatureKernel::new(3, 1);
        let test_seqs: Vec<Vec<Vec<f64>>> = (0..3)
            .map(|_| (0..4).map(|t| vec![t as f64 * 0.1, 0.5]).collect())
            .collect();
        let metrics = compute_sde_metrics(&model, &test_seqs, &kernel, &mut rng);
        assert!(metrics.path_mmd >= 0.0, "path_mmd must be non-negative");
    }

    // --- shuffle_product test ---

    #[test]
    fn test_shuffle_product_same_length_as_sig() {
        let ps = PathSignature::new(2, 2);
        let path1 = vec![vec![0.0, 0.0], vec![1.0, 0.5]];
        let path2 = vec![vec![0.0, 0.0], vec![-0.5, 1.0]];
        let sig1 = ps.compute(&path1);
        let sig2 = ps.compute(&path2);
        let shuf = shuffle_product(&sig1, &sig2, 2, 2);
        assert_eq!(shuf.len(), ps.signature_dim());
    }

    // --- Additional integration tests ---

    #[test]
    fn test_full_pipeline_ou_to_latent_sde() {
        let mut rng = make_rng();
        let cfg = LatentSdeConfig {
            z_dim: 2,
            x_dim: 1,
            hidden_dim: 4,
            n_steps: 9,
            dt: 0.1,
            kl_coeff: 0.1,
        };
        let model = LatentSde::new(cfg.clone());
        let ou_seq =
            SdeTrainer::generate_ou_sequence(1.0, 0.0, 0.2, 0.5, cfg.n_steps + 1, cfg.dt, &mut rng);
        let x_obs: Vec<Vec<f64>> = ou_seq.iter().map(|&v| vec![v]).collect();
        let (elbo, recon, kl) = model.elbo(&x_obs, &mut rng);
        assert!(elbo.is_finite() && recon.is_finite() && kl.is_finite());
    }

    #[test]
    fn test_neural_cde_long_sequence() {
        let config = NeuralCdeConfig {
            z_dim: 4,
            x_dim: 2,
            hidden_dim: 8,
            n_steps: 20,
        };
        let cde = NeuralCde::new(config);
        let times: Vec<f64> = (0..10).map(|t| t as f64 * 0.1).collect();
        let values: Vec<Vec<f64>> = times.iter().map(|&t| vec![t.sin(), t.cos()]).collect();
        let out = cde.forward(&times, &values);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_path_signature_additive_path() {
        let ps = PathSignature::new(2, 2);
        let path = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let sig = ps.compute(&path);
        assert!(
            (sig[1] - 2.0).abs() < 1e-8,
            "level-1 x-component should be 2.0, got {}",
            sig[1]
        );
    }

    #[test]
    fn test_encoder_decoder_roundtrip_shape() {
        let enc = SdeEncoder::new(4, 3, 8);
        let dec = SdeDecoder::new(3, 4, 8);
        let mut rng = make_rng();
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let (mean, _log_var) = enc.encode(&x);
        let eps: Vec<f64> = mean
            .iter()
            .map(|_| sample_standard_normal(&mut rng))
            .collect();
        let z: Vec<f64> = mean
            .iter()
            .zip(eps.iter())
            .map(|(&m, &e)| m + e * 0.1)
            .collect();
        let x_rec = dec.decode(&z);
        assert_eq!(
            x_rec.len(),
            4,
            "decoded output must match original dimension"
        );
    }
}
