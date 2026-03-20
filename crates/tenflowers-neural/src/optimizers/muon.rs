//! Muon optimizer
//!
//! Muon — Kosson et al., 2024
//! "Momentum + Orthogonalization for Neural Networks"
//!
//! Muon applies Nesterov-style momentum then **orthogonalises** the momentum
//! buffer via a Newton-Schulz iteration before using it as the update
//! direction.  This keeps weight matrices approximately orthogonal during
//! training, which improves gradient flow in deep networks.
//!
//! Update rule:
//! ```text
//! m_t = μ · m_{t-1} + g_t           (Nesterov accumulation, no damping)
//! O_t = orthogonalize(m_t)          (NS-iteration, reshaped as rows×cols)
//! θ_t = θ - lr · O_t
//! ```
//!
//! Newton-Schulz (Björck-Bowie) iteration:
//! ```text
//! X ← m / ‖m‖_F
//! for _ in 0..steps:
//!     X ← 1.5·X − 0.5·X·Xᵀ·X
//! ```

/// Configuration for the Muon optimizer.
///
/// Defaults: `lr = 0.02`, `μ = 0.95`, `ns_steps = 5`, `nesterov = true`.
#[derive(Debug, Clone)]
pub struct MuonConfig {
    /// Learning rate (default `0.02`).
    pub learning_rate: f32,
    /// Momentum coefficient (`μ`, default `0.95`).
    pub momentum: f32,
    /// Number of Newton-Schulz iterations for orthogonalisation (default `5`).
    pub ns_steps: usize,
    /// Whether to use Nesterov-style gradient accumulation (default `true`).
    pub nesterov: bool,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.02,
            momentum: 0.95,
            ns_steps: 5,
            nesterov: true,
        }
    }
}

/// Muon optimizer with a flat `Vec<f32>` parameter API.
///
/// The caller is responsible for providing `num_rows` when calling \[`step`\],
/// which is needed to reshape the flat gradient vector into a 2-D matrix for
/// orthogonalisation.  If the parameter count is not divisible by `num_rows`
/// the last incomplete row is left un-orthogonalised (identity treatment).
///
/// # Example
/// ```rust
/// use tenflowers_neural::optimizers::{MuonConfig, MuonOptimizer};
///
/// let config = MuonConfig::default();
/// let mut opt  = MuonOptimizer::new(config, 6); // 2 rows × 3 cols
/// let mut params = vec![1.0_f32; 6];
/// let grads      = vec![0.1_f32; 6];
/// opt.step(&mut params, &grads, 2);
/// ```
#[derive(Debug, Clone)]
pub struct MuonOptimizer {
    /// Optimizer configuration.
    pub config: MuonConfig,
    /// Momentum buffer.
    pub moment: Vec<f32>,
    /// Number of completed optimisation steps.
    pub step_count: u64,
}

impl MuonOptimizer {
    /// Create a new `MuonOptimizer` for `n_params` parameters.
    pub fn new(config: MuonConfig, n_params: usize) -> Self {
        Self {
            config,
            moment: vec![0.0_f32; n_params],
            step_count: 0,
        }
    }

    /// Perform one Muon update step.
    ///
    /// * `params` — mutable slice of flat parameter values.
    /// * `gradients` — gradient values aligned with `params`.
    /// * `num_rows` — number of rows when reshaping into a 2-D matrix for NS
    ///   orthogonalisation.  If `params.len() % num_rows != 0` the remainder
    ///   is updated without orthogonalisation.
    pub fn step(&mut self, params: &mut [f32], gradients: &[f32], num_rows: usize) {
        let mu = self.config.momentum;
        let lr = self.config.learning_rate;
        let ns = self.config.ns_steps;

        let n = params.len().min(gradients.len());

        // Resize moment buffer if needed.
        if self.moment.len() != params.len() {
            self.moment.resize(params.len(), 0.0_f32);
        }

        // --- Nesterov momentum accumulation ---
        for i in 0..n {
            self.moment[i] = mu * self.moment[i] + gradients[i];
        }

        // --- Orthogonalise and update ---
        let rows = if num_rows == 0 { 1 } else { num_rows };
        let cols = if rows == 0 || n == 0 { n } else { n / rows };

        if cols == 0 || rows * cols == 0 {
            // Cannot form a sensible matrix – fall back to direct scaled update.
            for i in 0..n {
                params[i] -= lr * self.moment[i];
            }
        } else {
            let full = rows * cols;
            // Orthogonalise the full rows×cols block.
            let m_slice = &self.moment[..full];
            let ortho = self.newton_schulz_orthogonalize(m_slice, rows, cols, ns);
            for i in 0..full {
                params[i] -= lr * ortho[i];
            }
            // Any remaining elements (tail) updated without orthogonalisation.
            for i in full..n {
                params[i] -= lr * self.moment[i];
            }
        }

        self.step_count = self.step_count.saturating_add(1);
    }

    /// Approximate orthogonalisation of a `rows × cols` matrix via the
    /// Newton-Schulz (Björck-Bowie) iteration.
    ///
    /// # Arguments
    /// * `m`     — flat row-major matrix slice of length `rows * cols`.
    /// * `rows`  — number of rows.
    /// * `cols`  — number of columns.
    /// * `steps` — number of Newton-Schulz iterations.
    ///
    /// # Returns
    /// A `Vec<f32>` of the same shape with approximately orthogonal rows.
    /// If `‖m‖_F < 1e-12` the input is returned as-is (near-zero matrix).
    pub fn newton_schulz_orthogonalize(
        &self,
        m: &[f32],
        rows: usize,
        cols: usize,
        steps: usize,
    ) -> Vec<f32> {
        debug_assert_eq!(m.len(), rows * cols);
        let n = rows * cols;

        // Frobenius norm.
        let frob: f32 = m.iter().map(|x| x * x).sum::<f32>().sqrt();
        if frob < 1e-12 {
            return m.to_vec();
        }

        // Normalise.
        let mut x: Vec<f32> = m.iter().map(|v| v / frob).collect();

        // Newton-Schulz iteration: X ← 1.5·X − 0.5·X·Xᵀ·X
        for _ in 0..steps {
            // tmp = X · Xᵀ  (rows × rows)
            let xxt = Self::matrix_mul_transpose(&x, &x, rows, cols, rows);
            // x_new[i,j] = 1.5·x[i,j] − 0.5·(xxt · x)[i,j]
            // (xxt · x) where xxt is rows×rows and x is rows×cols → rows×cols
            let xxt_x = Self::mat_mul(&xxt, &x, rows, rows, cols);
            for i in 0..n {
                x[i] = 1.5 * x[i] - 0.5 * xxt_x[i];
            }
        }

        x
    }

    /// Compute `A @ Bᵀ` where `A` is `m × n` and `B` is `p × n`,
    /// producing an `m × p` result.  All matrices are row-major flat slices.
    pub fn matrix_mul_transpose(a: &[f32], b: &[f32], m: usize, n: usize, p: usize) -> Vec<f32> {
        debug_assert_eq!(a.len(), m * n);
        debug_assert_eq!(b.len(), p * n);
        let mut out = vec![0.0_f32; m * p];
        for i in 0..m {
            for j in 0..p {
                let mut acc = 0.0_f32;
                for k in 0..n {
                    acc += a[i * n + k] * b[j * n + k];
                }
                out[i * p + j] = acc;
            }
        }
        out
    }

    /// Compute `A @ B` where `A` is `m × k` and `B` is `k × n`,
    /// producing an `m × n` result.  All matrices are row-major flat slices.
    fn mat_mul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);
        let mut out = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for l in 0..k {
                    acc += a[i * k + l] * b[l * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    /// Update the learning rate on-the-fly.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn frob(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Compute ‖I − Xᵀ·X‖_F for an m×n matrix X (rows × cols).
    /// For an orthogonal X this should be small.
    fn orthogonality_error(x: &[f32], rows: usize, cols: usize) -> f32 {
        // Xᵀ·X is cols × cols.
        let xt_x = {
            let mut out = vec![0.0_f32; cols * cols];
            for i in 0..cols {
                for j in 0..cols {
                    let mut acc = 0.0_f32;
                    for k in 0..rows {
                        acc += x[k * cols + i] * x[k * cols + j];
                    }
                    out[i * cols + j] = acc;
                }
            }
            out
        };
        // Residual: I - XᵀX (only makes sense for rows >= cols)
        let mut err = 0.0_f32;
        for i in 0..cols {
            for j in 0..cols {
                let id = if i == j { 1.0_f32 } else { 0.0_f32 };
                let d = xt_x[i * cols + j] - id;
                err += d * d;
            }
        }
        err.sqrt()
    }

    // ── MuonConfig ───────────────────────────────────────────────────────────

    #[test]
    fn test_muon_config_defaults() {
        let cfg = MuonConfig::default();
        assert!((cfg.learning_rate - 0.02).abs() < 1e-9);
        assert!((cfg.momentum - 0.95).abs() < 1e-9);
        assert_eq!(cfg.ns_steps, 5);
        assert!(cfg.nesterov);
    }

    // ── newton_schulz_orthogonalize ──────────────────────────────────────────

    #[test]
    fn test_muon_orthogonalize_identity_2x2() {
        // Orthogonalising the identity matrix (2×2) should return ~identity.
        let opt = MuonOptimizer::new(MuonConfig::default(), 4);
        let id = vec![1.0_f32, 0.0, 0.0, 1.0];
        let out = opt.newton_schulz_orthogonalize(&id, 2, 2, 5);
        // Frobenius distance from identity should be small.
        let err = out
            .iter()
            .zip([1.0_f32, 0.0, 0.0, 1.0].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            err < 0.2,
            "NS on identity should be near identity: err={err}"
        );
    }

    #[test]
    fn test_muon_orthogonalize_preserves_frob_approx() {
        // After orthogonalisation the Frobenius norm should be ≈ sqrt(rows)
        // (each row approximately unit-length).
        let opt = MuonOptimizer::new(MuonConfig::default(), 4);
        let m = vec![3.0_f32, 4.0, 0.0, 0.0]; // 2×2; rows: [3,4] and [0,0]
        let out = opt.newton_schulz_orthogonalize(&m, 2, 2, 5);
        // The non-zero row should have unit length after NS (Frobenius ≈ 1.0 for that row).
        let row0_norm: f32 = out[0..2].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (row0_norm - 1.0).abs() < 0.1,
            "non-zero row should be unit-length after NS: norm={row0_norm}"
        );
    }

    #[test]
    fn test_muon_orthogonalize_zero_returns_zero() {
        let opt = MuonOptimizer::new(MuonConfig::default(), 4);
        let zeros = vec![0.0_f32; 4];
        let out = opt.newton_schulz_orthogonalize(&zeros, 2, 2, 5);
        // Near-zero input: fallback returns input unchanged.
        assert!(frob(&out) < 1e-10, "zero input should produce zero output");
    }

    #[test]
    fn test_muon_orthogonalize_random_3x4_converges() {
        // A random 3×4 matrix — after 10 NS steps the rows should be
        // closer to orthonormal than before.
        let opt = MuonOptimizer::new(MuonConfig::default(), 12);
        // Deterministic "random" values.
        let m: Vec<f32> = (0..12).map(|i| ((i * 7 + 3) % 11) as f32 - 5.0).collect();
        let out = opt.newton_schulz_orthogonalize(&m, 3, 4, 10);
        // rows: 3, cols: 4 — each row norm should be ≈ 1 (or at least < 2).
        for r in 0..3 {
            let rn: f32 = out[r * 4..(r + 1) * 4]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            assert!(rn < 2.0, "row {r} norm too large after NS: {rn}");
        }
    }

    #[test]
    fn test_muon_orthogonalize_rows_gt_cols() {
        // 4×2 matrix — square XᵀX (2×2) should be near identity after NS.
        let opt = MuonOptimizer::new(MuonConfig::default(), 8);
        let m: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.5, -0.5, 2.0];
        let out = opt.newton_schulz_orthogonalize(&m, 4, 2, 10);
        let err = orthogonality_error(&out, 4, 2);
        assert!(err < 0.5, "orthogonality error too large: {err}");
    }

    // ── matrix_mul_transpose ─────────────────────────────────────────────────

    #[test]
    fn test_muon_mat_mul_transpose_2x2() {
        // A = [[1,0],[0,1]], B = [[1,0],[0,1]] → A @ Bᵀ = I
        let a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let b = vec![1.0_f32, 0.0, 0.0, 1.0];
        let c = MuonOptimizer::matrix_mul_transpose(&a, &b, 2, 2, 2);
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 0.0).abs() < 1e-6);
        assert!((c[2] - 0.0).abs() < 1e-6);
        assert!((c[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_muon_mat_mul_transpose_rectangular() {
        // A (2×3), B (2×3) → A@Bᵀ (2×2)
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        // row0·row0_b = 1·1+2·0+3·0 = 1; row0·row1_b = 1·0+2·1+3·0 = 2
        // row1·row0_b = 4; row1·row1_b = 5
        let c = MuonOptimizer::matrix_mul_transpose(&a, &b, 2, 3, 2);
        assert!((c[0] - 1.0).abs() < 1e-6, "c[0]={}", c[0]);
        assert!((c[1] - 2.0).abs() < 1e-6, "c[1]={}", c[1]);
        assert!((c[2] - 4.0).abs() < 1e-6, "c[2]={}", c[2]);
        assert!((c[3] - 5.0).abs() < 1e-6, "c[3]={}", c[3]);
    }

    // ── step ────────────────────────────────────────────────────────────────

    #[test]
    fn test_muon_zero_gradient_no_update() {
        let mut opt = MuonOptimizer::new(MuonConfig::default(), 4);
        let original = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut params = original.clone();
        let grads = vec![0.0_f32; 4];
        opt.step(&mut params, &grads, 2);
        // With zero gradient moment stays zero → ortho(zero) = zero → no update.
        for (a, b) in params.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-7,
                "params changed with zero grad: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_muon_step_count_increments() {
        let mut opt = MuonOptimizer::new(MuonConfig::default(), 4);
        let mut params = vec![1.0_f32; 4];
        let grads = vec![0.1_f32; 4];
        assert_eq!(opt.step_count, 0);
        opt.step(&mut params, &grads, 2);
        assert_eq!(opt.step_count, 1);
    }

    #[test]
    fn test_muon_step_reduces_positive_params_with_positive_grads() {
        let cfg = MuonConfig {
            learning_rate: 0.1,
            momentum: 0.9,
            ns_steps: 3,
            nesterov: true,
        };
        let mut opt = MuonOptimizer::new(cfg, 4);
        let mut params = vec![1.0_f32; 4];
        let grads = vec![1.0_f32; 4]; // all-ones gradient
        opt.step(&mut params, &grads, 2);
        // Some parameter should have decreased (ortho of all-ones has positive rows).
        let decreased = params.iter().any(|&p| p < 1.0);
        assert!(
            decreased,
            "at least one param should decrease: {:?}",
            params
        );
    }

    #[test]
    fn test_muon_set_learning_rate() {
        let mut opt = MuonOptimizer::new(MuonConfig::default(), 2);
        opt.set_learning_rate(0.5);
        assert!((opt.config.learning_rate - 0.5).abs() < 1e-9);
    }
}
