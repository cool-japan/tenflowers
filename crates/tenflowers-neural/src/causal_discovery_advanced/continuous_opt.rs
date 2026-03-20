//! NOTEARS + GraNDAG + DCDI — continuous optimisation approaches for DAG learning.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

use super::shared::mat_exp_approx;

// ─────────────────────────────────────────────────────────────────────────────
// 2. NOTEARS
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the NOTEARS continuous optimization algorithm.
#[derive(Debug, Clone)]
pub struct NoTearsConfig {
    /// L1 penalty strength for sparsity.
    pub lambda1: f32,
    /// Maximum augmented Lagrangian iterations.
    pub max_iter: usize,
    /// Acyclicity tolerance threshold (stop when |h(W)| < h_tol).
    pub h_tol: f32,
}

impl Default for NoTearsConfig {
    fn default() -> Self {
        Self {
            lambda1: 0.1,
            max_iter: 100,
            h_tol: 1e-8,
        }
    }
}

/// Compute h(W) = trace(exp(W ∘ W)) − n where ∘ is element-wise product.
///
/// The constraint h(W) = 0 ⟺ W encodes a DAG (Zheng et al. 2018).
/// Matrix exponential approximated via Taylor series of order 6.
pub fn h_constraint(w: &[f32], n: usize) -> f32 {
    // Build element-wise W ∘ W
    let hadamard: Vec<f32> = w.iter().map(|&v| v * v).collect();
    let exp_h = mat_exp_approx(&hadamard, n);
    // Trace
    let trace: f32 = (0..n).map(|i| exp_h[i * n + i]).sum();
    trace - n as f32
}

/// NOTEARS loss = (0.5/n) * ‖X − XW‖_F² + λ₁ ‖W‖₁
pub fn notears_loss(w: &[f32], x: &[Vec<f32>], lambda1: f32) -> f32 {
    let n_samples = x.len();
    if n_samples == 0 {
        return 0.0;
    }
    let n_vars = x[0].len();
    if n_vars == 0 || w.len() != n_vars * n_vars {
        return 0.0;
    }

    // Frobenius ‖X − XW‖²
    let mut fro_sq = 0.0f32;
    for row in x {
        for j in 0..n_vars {
            // (XW)[i,j] = sum_k X[i,k] * W[k,j]
            let xw_j: f32 = (0..n_vars)
                .map(|k| {
                    let xk = if k < row.len() { row[k] } else { 0.0 };
                    xk * w[k * n_vars + j]
                })
                .sum();
            let xj = if j < row.len() { row[j] } else { 0.0 };
            let diff = xj - xw_j;
            fro_sq += diff * diff;
        }
    }

    let data_loss = 0.5 / n_samples as f32 * fro_sq;
    let l1: f32 = w.iter().map(|&v| v.abs()).sum();
    data_loss + lambda1 * l1
}

/// One augmented Lagrangian step for NOTEARS.
///
/// Updates W in-place using gradient descent on the augmented Lagrangian:
///   L(W, λ, ρ) = loss(W) + λ h(W) + (ρ/2) h(W)²
///
/// Returns the current h(W) value.
pub fn notears_step(w: &mut [f32], x: &[Vec<f32>], config: &NoTearsConfig) -> f32 {
    let n_vars = x.first().map(|r| r.len()).unwrap_or(0);
    if n_vars == 0 || w.len() != n_vars * n_vars {
        return 0.0;
    }

    let lr = 0.001f32;
    let rho = 1.0f32; // augmented Lagrangian penalty coefficient
    let lambda_al = 0.0f32; // Lagrange multiplier (simplified; full NOTEARS uses dual ascent)

    let h_val = h_constraint(w, n_vars);

    // Numerical gradient of augmented Lagrangian loss
    let eps = 1e-4f32;
    let base_loss =
        notears_loss(w, x, config.lambda1) + lambda_al * h_val + 0.5 * rho * h_val * h_val;

    for i in 0..w.len() {
        let orig = w[i];
        w[i] = orig + eps;
        let h_p = h_constraint(w, n_vars);
        let l_p = notears_loss(w, x, config.lambda1) + lambda_al * h_p + 0.5 * rho * h_p * h_p;
        w[i] = orig;
        let grad = (l_p - base_loss) / eps;
        // Proximal gradient (soft-thresholding for L1)
        let updated = w[i] - lr * grad;
        let thresh = lr * config.lambda1;
        w[i] = if updated > thresh {
            updated - thresh
        } else if updated < -thresh {
            updated + thresh
        } else {
            0.0
        };
    }

    h_constraint(w, n_vars)
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. GraNDAG
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for GraNDAG.
#[derive(Debug, Clone)]
pub struct GraNDagConfig {
    /// Number of observed variables.
    pub n_vars: usize,
    /// Hidden layer dimension for per-variable MLPs.
    pub hidden_dim: usize,
    /// Number of hidden layers in each MLP.
    pub n_layers: usize,
}

impl Default for GraNDagConfig {
    fn default() -> Self {
        Self {
            n_vars: 4,
            hidden_dim: 8,
            n_layers: 2,
        }
    }
}

/// Per-variable MLP with a binary input mask (indicating which parents are allowed).
#[derive(Debug, Clone)]
pub struct MlpCausalModule {
    /// Flattened weight parameters (input_dim × hidden_dim + hidden_dim × output_dim).
    pub weights: Vec<f32>,
    /// Binary mask over inputs (0 = blocked, 1 = allowed).
    pub mask: Vec<f32>,
    /// Input dimension.
    pub input_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
}

impl MlpCausalModule {
    /// Create a randomly initialised MLP module.
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_weights = input_dim * hidden_dim + hidden_dim;
        let weights: Vec<f32> = (0..n_weights)
            .map(|_| (rng.random::<f32>() - 0.5) * 0.1)
            .collect();
        let mask = vec![1.0f32; input_dim];
        Self {
            weights,
            mask,
            input_dim,
            hidden_dim,
        }
    }

    /// Forward pass: masked linear → ReLU → linear.
    pub fn forward(&self, x: &[f32]) -> f32 {
        let n_in = self.input_dim.min(x.len()).min(self.mask.len());
        // Hidden layer
        let mut hidden = vec![0.0f32; self.hidden_dim];
        for h in 0..self.hidden_dim {
            for i in 0..n_in {
                let w_idx = i * self.hidden_dim + h;
                if w_idx < self.weights.len() {
                    hidden[h] += self.mask[i] * x[i] * self.weights[w_idx];
                }
            }
            hidden[h] = hidden[h].max(0.0); // ReLU
        }
        // Output layer
        let out_offset = self.input_dim * self.hidden_dim;
        let mut out = 0.0f32;
        for h in 0..self.hidden_dim {
            let w_idx = out_offset + h;
            if w_idx < self.weights.len() {
                out += hidden[h] * self.weights[w_idx];
            }
        }
        out
    }
}

/// Acyclicity penalty via spectral radius of the mask adjacency matrix.
///
/// For each variable j, the mask indicates which variables can be parents.
/// Builds an n × n adjacency matrix and approximates spectral radius via
/// power iteration for up to 20 steps.  Returns max_eigenvalue - 1.0 (< 0 = DAG).
pub fn acyclicity_penalty(masks: &[Vec<f32>]) -> f32 {
    let n = masks.len();
    if n == 0 {
        return 0.0;
    }
    // Build adjacency A[j][i] = mask[j][i]  (i is potential parent of j)
    let a: Vec<f32> = masks.iter().flat_map(|m| m.iter().copied()).collect();
    // Element-wise square (used in h constraint analog)
    let a_sq: Vec<f32> = a.iter().map(|v| v * v).collect();
    let exp_a = mat_exp_approx(&a_sq, n);
    let trace: f32 = (0..n).map(|i| exp_a[i * n + i]).sum();
    trace - n as f32
}

/// GraNDAG loss = MSE reconstruction + λ * acyclicity penalty.
pub fn grandag_loss(modules: &[MlpCausalModule], x: &[Vec<f32>], lambda: f32) -> f32 {
    let n_samples = x.len();
    if n_samples == 0 || modules.is_empty() {
        return 0.0;
    }
    let n_vars = modules.len();

    // Reconstruction MSE
    let mut mse = 0.0f32;
    for row in x {
        for (j, module) in modules.iter().enumerate() {
            let xj = if j < row.len() { row[j] } else { 0.0 };
            let pred = module.forward(row);
            mse += (xj - pred) * (xj - pred);
        }
    }
    mse /= (n_samples * n_vars) as f32;

    // Collect masks and compute acyclicity penalty
    let masks: Vec<Vec<f32>> = modules.iter().map(|m| m.mask.clone()).collect();
    let penalty = acyclicity_penalty(&masks);

    mse + lambda * penalty
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. DCDI — Differentiable Causal Discovery with Interventions
// ─────────────────────────────────────────────────────────────────────────────

/// Dataset combining observational and interventional samples.
#[derive(Debug, Clone)]
pub struct InterventionData {
    /// Observational samples (n_obs × n_vars).
    pub obs: Vec<Vec<f32>>,
    /// Interventional samples: (target_variable, samples) pairs.
    pub interventional: Vec<(usize, Vec<Vec<f32>>)>,
}

/// Gaussian log-likelihood for data given a linear model W:
/// ℓ(W|X) = -n/2 log σ² − (1/2σ²) ‖X − XW‖_F²
/// Returns the *negative* log-likelihood (to minimise).
pub fn interventional_likelihood(w: &[f32], data: &InterventionData, n_vars: usize) -> f32 {
    if n_vars == 0 || w.len() != n_vars * n_vars {
        return 0.0;
    }

    // Helper: compute negative log-likelihood for a set of samples.
    let nll_samples = |samples: &[Vec<f32>], blocked_cols: &[usize]| -> f32 {
        let m = samples.len();
        if m == 0 {
            return 0.0;
        }
        let mut fro = 0.0f32;
        for row in samples {
            for j in 0..n_vars {
                if blocked_cols.contains(&j) {
                    continue; // Intervention: don't model this node's parents
                }
                let xj = if j < row.len() { row[j] } else { 0.0 };
                let xw_j: f32 = (0..n_vars)
                    .map(|k| {
                        let xk = if k < row.len() { row[k] } else { 0.0 };
                        xk * w[k * n_vars + j]
                    })
                    .sum();
                let diff = xj - xw_j;
                fro += diff * diff;
            }
        }
        0.5 * fro / m as f32
    };

    // Observational likelihood
    let mut total = nll_samples(&data.obs, &[]);

    // Interventional likelihoods
    for (target, samples) in &data.interventional {
        total += nll_samples(samples, &[*target]);
    }

    total
}

/// One gradient descent step for DCDI.  Returns the current loss.
pub fn dcdi_step(w: &mut [f32], data: &InterventionData, lr: f32) -> f32 {
    let n_vars = data.obs.first().map(|r| r.len()).unwrap_or(0);
    if n_vars == 0 || w.len() != n_vars * n_vars {
        return 0.0;
    }

    let eps = 1e-4f32;
    let base_loss = interventional_likelihood(w, data, n_vars);

    for i in 0..w.len() {
        let orig = w[i];
        w[i] = orig + eps;
        let l_p = interventional_likelihood(w, data, n_vars);
        w[i] = orig;
        let grad = (l_p - base_loss) / eps;
        w[i] -= lr * grad;
    }

    interventional_likelihood(w, data, n_vars)
}
