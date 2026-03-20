//! LAMB optimizer (vec-based flat API)
//!
//! LAMB — You et al., 2020
//! "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
//!
//! LAMB extends Adam with a **layer-wise trust ratio** that scales each
//! parameter's effective learning rate by `‖θ‖ / ‖update‖`, preventing
//! individual layers from receiving unreasonably large or small steps at large
//! batch sizes.
//!
//! Update rule per step (step index *t* starts at 1):
//! ```text
//! m_t  = β₁·m_{t-1} + (1-β₁)·g_t
//! v_t  = β₂·v_{t-1} + (1-β₂)·g_t²
//! m̂   = m_t / (1 - β₁ᵗ)
//! v̂   = v_t / (1 - β₂ᵗ)
//! upd  = m̂ / (√v̂ + ε) + λ·θ
//! r    = trust_ratio(‖θ‖, ‖upd‖)
//! θ_t  = θ - lr · r · upd
//! ```

/// Configuration for the LAMB optimizer.
///
/// Default values follow the original BERT fine-tuning configuration:
/// `lr=1e-3`, `β₁=0.9`, `β₂=0.999`, `ε=1e-6`, `λ=0.01`,
/// `clamp_value=10.0`.
#[derive(Debug, Clone)]
pub struct LambConfig {
    /// Learning rate (default `1e-3`).
    pub learning_rate: f32,
    /// Exponential decay for the first moment (`β₁`, default `0.9`).
    pub beta1: f32,
    /// Exponential decay for the second moment (`β₂`, default `0.999`).
    pub beta2: f32,
    /// Numerical stability constant (`ε`, default `1e-6`).
    pub epsilon: f32,
    /// Decoupled weight-decay coefficient (`λ`, default `0.01`).
    pub weight_decay: f32,
    /// Upper bound on the trust ratio (default `10.0`).
    pub clamp_value: f32,
}

impl Default for LambConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-6,
            weight_decay: 0.01,
            clamp_value: 10.0,
        }
    }
}

/// LAMB optimizer with a flat `Vec<f32>` parameter API.
///
/// # Example
/// ```rust
/// use tenflowers_neural::optimizers::{LambConfig, LambOptimizer};
///
/// let config = LambConfig { weight_decay: 0.0, ..LambConfig::default() };
/// let mut opt = LambOptimizer::new(config, 3);
/// let mut params = vec![1.0_f32, -1.0, 0.5];
/// let grads  = vec![0.1_f32,  0.1, 0.1];
/// opt.step(&mut params, &grads);
/// ```
#[derive(Debug, Clone)]
pub struct LambOptimizer {
    /// Optimizer configuration.
    pub config: LambConfig,
    /// First-moment buffer.
    pub m: Vec<f32>,
    /// Second-moment buffer.
    pub v: Vec<f32>,
    /// Number of completed optimisation steps (1-indexed for bias correction).
    pub step_count: u64,
}

impl LambOptimizer {
    /// Create a new `LambOptimizer` for `n_params` parameters.
    pub fn new(config: LambConfig, n_params: usize) -> Self {
        Self {
            config,
            m: vec![0.0_f32; n_params],
            v: vec![0.0_f32; n_params],
            step_count: 0,
        }
    }

    /// Compute the LAMB trust ratio.
    ///
    /// Returns `min(clamp_value, ‖θ‖ / ‖update‖)`.
    /// Returns `1.0` when either norm is zero to avoid division by zero.
    pub fn trust_ratio(&self, param_norm: f32, update_norm: f32) -> f32 {
        if param_norm == 0.0 || update_norm == 0.0 {
            1.0
        } else {
            let ratio = param_norm / update_norm;
            ratio.min(self.config.clamp_value)
        }
    }

    /// Perform one LAMB update step.
    pub fn step(&mut self, params: &mut [f32], gradients: &[f32]) {
        self.step_count = self.step_count.saturating_add(1);
        let t = self.step_count as i32;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.epsilon;
        let wd = self.config.weight_decay;
        let lr = self.config.learning_rate;

        let bc1 = 1.0 - beta1.powi(t);
        let bc2 = 1.0 - beta2.powi(t);

        // Ensure buffers are sized correctly.
        if self.m.len() != params.len() {
            self.m.resize(params.len(), 0.0_f32);
            self.v.resize(params.len(), 0.0_f32);
        }

        // --- First pass: compute per-element updates ---
        let mut updates = Vec::with_capacity(params.len().min(gradients.len()));

        for i in 0..params.len().min(gradients.len()) {
            let g = gradients[i];
            let p = params[i];

            // Moment updates.
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;

            // Bias-corrected estimates.
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;

            // Adam-style update direction + weight decay.
            let upd = m_hat / (v_hat.sqrt() + eps) + wd * p;
            updates.push((i, p, upd));
        }

        // --- Compute global norms for trust ratio ---
        let param_norm = {
            let ss: f32 = updates.iter().map(|(_, p, _)| p * p).sum();
            ss.sqrt()
        };
        let update_norm = {
            let ss: f32 = updates.iter().map(|(_, _, u)| u * u).sum();
            ss.sqrt()
        };

        let r = self.trust_ratio(param_norm, update_norm);

        // --- Apply update ---
        for (i, _, upd) in updates {
            params[i] -= lr * r * upd;
        }
    }

    /// Update the learning rate on-the-fly.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_opt(wd: f32) -> LambOptimizer {
        let cfg = LambConfig {
            learning_rate: 0.01,
            weight_decay: wd,
            ..LambConfig::default()
        };
        LambOptimizer::new(cfg, 3)
    }

    // ── LambConfig ───────────────────────────────────────────────────────────

    #[test]
    fn test_lamb_config_defaults() {
        let cfg = LambConfig::default();
        assert!((cfg.learning_rate - 1e-3).abs() < 1e-9);
        assert!((cfg.beta1 - 0.9).abs() < 1e-9);
        assert!((cfg.beta2 - 0.999).abs() < 1e-9);
        assert!((cfg.epsilon - 1e-6).abs() < 1e-10);
        assert!((cfg.weight_decay - 0.01).abs() < 1e-9);
        assert!((cfg.clamp_value - 10.0).abs() < 1e-9);
    }

    // ── trust_ratio ──────────────────────────────────────────────────────────

    #[test]
    fn test_lamb_trust_ratio_equal_norms() {
        let opt = make_opt(0.0);
        // When both norms are equal the ratio is 1.0 (min(10, 1) = 1).
        let r = opt.trust_ratio(2.0, 2.0);
        assert!((r - 1.0).abs() < 1e-6, "trust_ratio={r}");
    }

    #[test]
    fn test_lamb_trust_ratio_clamps() {
        let opt = make_opt(0.0);
        // Large param_norm relative to update_norm: ratio = 100/5 = 20, clamped to 10.
        let r = opt.trust_ratio(100.0, 5.0);
        assert!(
            (r - 10.0).abs() < 1e-6,
            "trust_ratio should be clamped: {r}"
        );
    }

    #[test]
    fn test_lamb_trust_ratio_zero_param_norm() {
        let opt = make_opt(0.0);
        let r = opt.trust_ratio(0.0, 5.0);
        assert!(
            (r - 1.0).abs() < 1e-6,
            "trust_ratio with zero param_norm: {r}"
        );
    }

    #[test]
    fn test_lamb_trust_ratio_zero_update_norm() {
        let opt = make_opt(0.0);
        let r = opt.trust_ratio(5.0, 0.0);
        assert!(
            (r - 1.0).abs() < 1e-6,
            "trust_ratio with zero update_norm: {r}"
        );
    }

    #[test]
    fn test_lamb_trust_ratio_custom_clamp() {
        let cfg = LambConfig {
            clamp_value: 3.0,
            ..LambConfig::default()
        };
        let opt = LambOptimizer::new(cfg, 2);
        let r = opt.trust_ratio(100.0, 5.0); // raw ratio = 20, clamp = 3
        assert!(
            (r - 3.0).abs() < 1e-6,
            "trust_ratio should respect custom clamp: {r}"
        );
    }

    // ── step ────────────────────────────────────────────────────────────────

    #[test]
    fn test_lamb_step_count_increments() {
        let mut opt = make_opt(0.0);
        let mut params = vec![1.0_f32, 2.0, 3.0];
        let grads = vec![0.1_f32, 0.1, 0.1];
        assert_eq!(opt.step_count, 0);
        opt.step(&mut params, &grads);
        assert_eq!(opt.step_count, 1);
        opt.step(&mut params, &grads);
        assert_eq!(opt.step_count, 2);
    }

    #[test]
    fn test_lamb_zero_weight_decay_moves_toward_negative_gradient() {
        // With wd=0 and a positive gradient params should decrease.
        let mut opt = make_opt(0.0);
        let original = vec![1.0_f32, 2.0, 3.0];
        let mut params = original.clone();
        let grads = vec![1.0_f32, 1.0, 1.0];
        opt.step(&mut params, &grads);
        for (a, b) in params.iter().zip(original.iter()) {
            assert!(
                a < b,
                "param should decrease with positive grad: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_lamb_bias_correction_first_step() {
        // On the first step bias corrections should be applied.
        // We just verify the step doesn't panic and step_count becomes 1.
        let cfg = LambConfig {
            learning_rate: 0.001,
            weight_decay: 0.0,
            ..LambConfig::default()
        };
        let mut opt = LambOptimizer::new(cfg, 2);
        let mut params = vec![0.5_f32, -0.5];
        let grads = vec![0.1_f32, -0.1];
        opt.step(&mut params, &grads);
        assert_eq!(opt.step_count, 1);
    }

    #[test]
    fn test_lamb_weight_decay_shrinks_param() {
        // Weight decay should pull parameters toward zero.
        let cfg = LambConfig {
            learning_rate: 0.1,
            weight_decay: 1.0, // large wd for visible effect
            ..LambConfig::default()
        };
        let mut opt = LambOptimizer::new(cfg, 1);
        let mut params = vec![5.0_f32];
        let grads = vec![0.0_f32];
        let before = params[0];
        opt.step(&mut params, &grads);
        assert!(
            params[0] < before,
            "weight_decay should shrink param: {}",
            params[0]
        );
    }

    #[test]
    fn test_lamb_set_learning_rate() {
        let mut opt = make_opt(0.0);
        opt.set_learning_rate(0.5);
        assert!((opt.config.learning_rate - 0.5).abs() < 1e-9);
    }
}
