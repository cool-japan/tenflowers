//! Lion optimizer (vec-based flat API)
//!
//! Lion (EvoLved Sign Momentum) — Chen et al., 2023
//! "Symbolic Discovery of Optimization Algorithms"
//!
//! Compared to Adam, Lion uses only one momentum buffer (not two) and
//! applies the sign of the interpolated momentum, producing sparser,
//! magnitude-decoupled updates.
//!
//! Update rule per step:
//! 1. `c   = β₁ * m_{t-1} + (1-β₁) * g_t`   (interpolation before moment update)
//! 2. `θ_t = θ - lr * (sign(c) + λ * θ)`     (update with decoupled weight decay)
//! 3. `m_t = β₂ * m_{t-1} + (1-β₂) * g_t`   (moment update for next step)

/// Configuration for the Lion optimizer.
///
/// Default values follow the original paper (Chen et al., 2023):
/// `lr = 1e-4`, `β₁ = 0.9`, `β₂ = 0.99`, `λ = 0.0`.
#[derive(Debug, Clone)]
pub struct LionConfig {
    /// Learning rate (typically smaller than Adam — default `1e-4`).
    pub learning_rate: f32,
    /// Momentum coefficient for the interpolation used to compute the sign (`β₁`, default `0.9`).
    pub beta1: f32,
    /// Momentum coefficient for the moment update (`β₂`, default `0.99`).
    pub beta2: f32,
    /// Decoupled weight-decay coefficient (`λ`, default `0.0`).
    pub weight_decay: f32,
}

impl Default for LionConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        }
    }
}

/// Lion optimizer with a flat `Vec<f32>` parameter API.
///
/// Unlike the model-based [`crate::optimizers::Lion`], this struct operates
/// directly on parameter and gradient slices, making it easy to use outside
/// the layer/model abstraction.
///
/// # Example
/// ```rust
/// use tenflowers_neural::optimizers::{LionConfig, LionOptimizer};
///
/// let config = LionConfig::default();
/// let mut opt = LionOptimizer::new(config, 4);
/// let mut params = vec![1.0_f32, -1.0, 0.5, -0.5];
/// let grads  = vec![0.1_f32,  0.1, 0.1,  0.1];
/// opt.step(&mut params, &grads);
/// ```
#[derive(Debug, Clone)]
pub struct LionOptimizer {
    /// Optimizer configuration.
    pub config: LionConfig,
    /// First-moment buffer (single buffer, unlike Adam's two).
    pub moment: Vec<f32>,
    /// Number of completed optimisation steps.
    pub step_count: u64,
}

impl LionOptimizer {
    /// Create a new `LionOptimizer` with `n_params` parameters.
    ///
    /// The moment buffer is zero-initialised.
    pub fn new(config: LionConfig, n_params: usize) -> Self {
        Self {
            config,
            moment: vec![0.0_f32; n_params],
            step_count: 0,
        }
    }

    /// Perform one Lion update step.
    ///
    /// # Panics (never — uses saturating arithmetic)
    /// The function will not panic; if `params` and `gradients` have different
    /// lengths it silently iterates over the shorter one (zip behaviour).
    pub fn step(&mut self, params: &mut [f32], gradients: &[f32]) {
        let lr = self.config.learning_rate;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let wd = self.config.weight_decay;

        // Ensure moment buffer matches parameter count.
        if self.moment.len() != params.len() {
            self.moment.resize(params.len(), 0.0_f32);
        }

        for i in 0..params.len().min(gradients.len()) {
            let g = gradients[i];
            let m = self.moment[i];
            let p = params[i];

            // Step 1 – interpolated value whose *sign* drives the update.
            let c = beta1 * m + (1.0 - beta1) * g;

            // Step 2 – parameter update: θ ← θ - lr * (sign(c) + λ·θ)
            let sign_c = sign_f32(c);
            params[i] = p - lr * (sign_c + wd * p);

            // Step 3 – moment update for the *next* step.
            self.moment[i] = beta2 * m + (1.0 - beta2) * g;
        }

        self.step_count = self.step_count.saturating_add(1);
    }

    /// Return a zero-filled gradient buffer with the same size as the current
    /// parameter vector.
    pub fn zero_grad(&self) -> Vec<f32> {
        vec![0.0_f32; self.moment.len()]
    }

    /// Update the learning rate on-the-fly (e.g. from a scheduler).
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

/// Element-wise sign: +1 if x > 0, -1 if x < 0, 0 if x == 0.
#[inline]
fn sign_f32(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ─────────────────────────────────────────────────────────────

    fn make_opt(lr: f32) -> LionOptimizer {
        let cfg = LionConfig {
            learning_rate: lr,
            ..LionConfig::default()
        };
        LionOptimizer::new(cfg, 4)
    }

    // ── LionConfig ───────────────────────────────────────────────────────────

    #[test]
    fn test_lion_config_defaults() {
        let cfg = LionConfig::default();
        assert!((cfg.learning_rate - 1e-4).abs() < 1e-9);
        assert!((cfg.beta1 - 0.9).abs() < 1e-9);
        assert!((cfg.beta2 - 0.99).abs() < 1e-9);
        assert!((cfg.weight_decay).abs() < 1e-9);
    }

    // ── LionOptimizer ────────────────────────────────────────────────────────

    #[test]
    fn test_lion_lr_zero_no_change() {
        // With lr=0 parameters must not change regardless of gradients.
        let mut opt = make_opt(0.0);
        let original = vec![1.0_f32, -1.0, 0.5, -0.5];
        let mut params = original.clone();
        let grads = vec![0.3_f32, -0.7, 1.0, -1.0];
        opt.step(&mut params, &grads);
        for (a, b) in params.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-9, "param changed with lr=0: {a} vs {b}");
        }
    }

    #[test]
    fn test_lion_sign_based_update_direction() {
        // With a positive gradient and positive momentum the update should push
        // the parameter in the negative direction (since sign > 0 and lr > 0).
        let cfg = LionConfig {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        };
        let mut opt = LionOptimizer::new(cfg, 1);
        // Pre-warm momentum in the positive direction.
        let mut params = vec![2.0_f32];
        let grads_pos = vec![1.0_f32];
        opt.step(&mut params, &grads_pos);
        // After one step with positive grad, param should decrease.
        assert!(
            params[0] < 2.0,
            "param should have decreased: {}",
            params[0]
        );
    }

    #[test]
    fn test_lion_negative_gradient_increases_param() {
        let cfg = LionConfig {
            learning_rate: 0.1,
            ..LionConfig::default()
        };
        let mut opt = LionOptimizer::new(cfg, 1);
        let mut params = vec![0.0_f32];
        let grads = vec![-1.0_f32]; // negative gradient → sign = -1 → update = +lr → param increases? No: p - lr*sign(-1) = 0 - 0.1*(-1) = 0.1
        opt.step(&mut params, &grads);
        assert!(
            params[0] > 0.0,
            "param should have increased: {}",
            params[0]
        );
    }

    #[test]
    fn test_lion_weight_decay_shrinks_positive_param() {
        // With weight_decay > 0 a positive parameter should be pushed toward 0.
        let cfg = LionConfig {
            learning_rate: 0.01,
            beta1: 0.0, // disable momentum to isolate weight_decay effect
            beta2: 0.0,
            weight_decay: 0.5,
        };
        let mut opt = LionOptimizer::new(cfg, 1);
        let mut params = vec![10.0_f32];
        let grads = vec![0.0_f32]; // zero grad — only wd term contributes
                                   // c = 0*m + 1*0 = 0 → sign(c) = 0
                                   // update = lr * (0 + 0.5 * 10) = 0.01 * 5 = 0.05
        opt.step(&mut params, &grads);
        assert!(
            params[0] < 10.0,
            "weight_decay should shrink param: {}",
            params[0]
        );
    }

    #[test]
    fn test_lion_weight_decay_shrinks_negative_param() {
        let cfg = LionConfig {
            learning_rate: 0.01,
            beta1: 0.0,
            beta2: 0.0,
            weight_decay: 0.5,
        };
        let mut opt = LionOptimizer::new(cfg, 1);
        let mut params = vec![-10.0_f32];
        let grads = vec![0.0_f32];
        opt.step(&mut params, &grads);
        assert!(
            params[0] > -10.0,
            "weight_decay should shrink negative param: {}",
            params[0]
        );
    }

    #[test]
    fn test_lion_step_count_increments() {
        let mut opt = make_opt(0.01);
        let mut params = vec![1.0_f32, 2.0];
        let grads = vec![0.1_f32, 0.2];
        assert_eq!(opt.step_count, 0);
        opt.step(&mut params, &grads);
        assert_eq!(opt.step_count, 1);
        opt.step(&mut params, &grads);
        assert_eq!(opt.step_count, 2);
    }

    #[test]
    fn test_lion_zero_grad_size() {
        let opt = make_opt(0.01);
        let zg = opt.zero_grad();
        assert_eq!(zg.len(), 4);
        assert!(zg.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn test_lion_set_learning_rate() {
        let mut opt = make_opt(0.01);
        opt.set_learning_rate(0.001);
        assert!((opt.config.learning_rate - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_lion_moment_update() {
        // After one step with grad=1 and β₂=0.99 the moment should be
        // 0*0.99 + 0.01*1 = 0.01.
        let cfg = LionConfig {
            learning_rate: 0.0,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        };
        let mut opt = LionOptimizer::new(cfg, 1);
        let mut params = vec![0.0_f32];
        let grads = vec![1.0_f32];
        opt.step(&mut params, &grads);
        assert!(
            (opt.moment[0] - 0.01).abs() < 1e-6,
            "moment[0]={}",
            opt.moment[0]
        );
    }

    #[test]
    fn test_lion_mismatched_sizes_no_panic() {
        // If gradients are shorter than params, the function should not panic.
        let mut opt = make_opt(0.01);
        let mut params = vec![1.0_f32, 2.0, 3.0, 4.0];
        let grads = vec![0.1_f32, 0.2]; // only 2 elements
        opt.step(&mut params, &grads); // must not panic
                                       // params[0] and [1] were updated, [2] and [3] unchanged.
        assert_eq!(params[2], 3.0);
        assert_eq!(params[3], 4.0);
    }

    #[test]
    fn test_sign_f32_values() {
        assert_eq!(sign_f32(5.0), 1.0);
        assert_eq!(sign_f32(-3.0), -1.0);
        assert_eq!(sign_f32(0.0), 0.0);
    }
}
