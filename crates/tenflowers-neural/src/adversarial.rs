//! Adversarial Training Utilities — Track V.
//!
//! Implements adversarial example generation and robustness training methods:
//!
//! - **FGSM** (Fast Gradient Sign Method): Goodfellow et al., 2014.
//! - **PGD** (Projected Gradient Descent): Madry et al., 2018.
//! - **TRADES**: Zhang et al., 2019 — a KL-regularised robust training objective.
//! - **RandomizedSmoothing**: Certified L2 robustness (Cohen et al., 2019).
//!
//! All randomness is sourced from `scirs2_core::random`.  No `unwrap()` is used.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::adversarial::{Fgsm, Pgd, AdversarialTrainer, RandomizedSmoothing};
//!
//! // FGSM single-step attack
//! let fgsm = Fgsm::new(0.03);
//! let x_adv = fgsm.perturb(&input, &gradient)?;
//!
//! // PGD multi-step attack
//! let pgd = Pgd::new(0.03, 0.007, 40);
//! let x_adv = pgd.attack(&input, |x| compute_gradient(x));
//!
//! // TRADES loss
//! let trainer = AdversarialTrainer::new_pgd(0.03, 0.007, 10);
//! let (nat_loss, rob_loss, trades_loss) = trainer.trades_loss(&nat_out, &adv_out, label)?;
//!
//! // Certified robustness
//! let smoother = RandomizedSmoothing::new(0.25, 1000);
//! let radius = smoother.certified_radius(p_a);
//! ```

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by adversarial training utilities.
#[derive(Debug, Clone)]
pub enum AdversarialError {
    /// Input slice is empty.
    EmptyInput,
    /// Input and gradient have different lengths.
    DimensionMismatch {
        /// Input length.
        input: usize,
        /// Gradient length.
        gradient: usize,
    },
    /// Non-positive or non-finite epsilon.
    InvalidEpsilon {
        /// The epsilon value that was invalid.
        epsilon: f32,
    },
    /// Zero steps requested.
    InvalidNumSteps {
        /// The number of steps that was invalid.
        steps: usize,
    },
    /// Attack could not find a perturbation (e.g. gradient is all-zero).
    NoPerturbationFound,
}

impl std::fmt::Display for AdversarialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdversarialError::EmptyInput => write!(f, "input slice must not be empty"),
            AdversarialError::DimensionMismatch { input, gradient } => write!(
                f,
                "dimension mismatch: input length {input} != gradient length {gradient}"
            ),
            AdversarialError::InvalidEpsilon { epsilon } => {
                write!(f, "epsilon must be finite and positive, got {epsilon}")
            }
            AdversarialError::InvalidNumSteps { steps } => {
                write!(f, "num_steps must be > 0, got {steps}")
            }
            AdversarialError::NoPerturbationFound => {
                write!(f, "no valid perturbation found (gradient may be all-zero)")
            }
        }
    }
}

impl std::error::Error for AdversarialError {}

// ─────────────────────────────────────────────────────────────────────────────
// Perturbation norm
// ─────────────────────────────────────────────────────────────────────────────

/// Which norm ball constrains the adversarial perturbation.
#[derive(Debug, Clone)]
pub enum PerturbationNorm {
    /// L∞ ball: each component is bounded by `|δ_i| ≤ epsilon`.
    Linf {
        /// Maximum per-element perturbation.
        epsilon: f32,
    },
    /// L2 ball: `||δ||_2 ≤ epsilon`.
    L2 {
        /// L2 radius.
        epsilon: f32,
    },
    /// L1 ball: `||δ||_1 ≤ epsilon` (promotes sparsity).
    L1 {
        /// L1 radius.
        epsilon: f32,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// FGSM
// ─────────────────────────────────────────────────────────────────────────────

/// Fast Gradient Sign Method — Goodfellow et al. (2014).
///
/// Produces a single-step adversarial perturbation:
/// ```text
/// x_adv = x + ε · sign(∇_x L)
/// ```
#[derive(Debug, Clone)]
pub struct Fgsm {
    /// Perturbation budget.
    pub epsilon: f32,
    /// Norm constraint.
    pub norm: PerturbationNorm,
}

impl Fgsm {
    /// Create an FGSM attacker with L∞ norm and the given epsilon.
    pub fn new(epsilon: f32) -> Self {
        Self {
            epsilon,
            norm: PerturbationNorm::Linf { epsilon },
        }
    }

    /// Create an FGSM attacker with a custom norm.
    pub fn with_norm(epsilon: f32, norm: PerturbationNorm) -> Self {
        Self { epsilon, norm }
    }

    /// Perturb `input` by the gradient sign scaled by epsilon.
    ///
    /// # Errors
    /// Returns `AdversarialError` when input is empty or lengths mismatch.
    pub fn perturb(
        &self,
        input: &[f32],
        gradient: &[f32],
    ) -> std::result::Result<Vec<f32>, AdversarialError> {
        if input.is_empty() {
            return Err(AdversarialError::EmptyInput);
        }
        if input.len() != gradient.len() {
            return Err(AdversarialError::DimensionMismatch {
                input: input.len(),
                gradient: gradient.len(),
            });
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(AdversarialError::InvalidEpsilon {
                epsilon: self.epsilon,
            });
        }

        let eps = self.epsilon;
        let x_adv: Vec<f32> = match &self.norm {
            PerturbationNorm::Linf { .. } => {
                // δ = ε · sign(g)
                input
                    .iter()
                    .zip(gradient.iter())
                    .map(|(xi, gi)| xi + eps * gi.signum())
                    .collect()
            }
            PerturbationNorm::L2 { .. } => {
                // δ = ε · g / ||g||_2
                let norm = l2_norm(gradient).max(1e-12);
                input
                    .iter()
                    .zip(gradient.iter())
                    .map(|(xi, gi)| xi + eps * gi / norm)
                    .collect()
            }
            PerturbationNorm::L1 { .. } => {
                // L1 FGSM: put full budget on the max-gradient component
                let max_idx = gradient
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.abs()
                            .partial_cmp(&b.abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let mut out = input.to_vec();
                out[max_idx] += eps * gradient[max_idx].signum();
                out
            }
        };
        Ok(x_adv)
    }

    /// Clip adversarial example so each element stays within `[x_min, x_max]`
    /// and within `epsilon` of the original input per-element.
    pub fn clip_to_range(&self, x_adv: &[f32], x_orig: &[f32], x_min: f32, x_max: f32) -> Vec<f32> {
        let eps = self.epsilon;
        x_adv
            .iter()
            .zip(x_orig.iter())
            .map(|(xa, xo)| {
                // Project to epsilon ball around original, then to valid range
                let lo = (xo - eps).max(x_min);
                let hi = (xo + eps).min(x_max);
                xa.clamp(lo, hi)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PGD
// ─────────────────────────────────────────────────────────────────────────────

/// Projected Gradient Descent attack — Madry et al. (2018).
///
/// Iterates:
/// ```text
/// x_{t+1} = Π_{B_ε(x)} ( x_t + α · sign(∇_{x_t} L) )
/// ```
#[derive(Debug, Clone)]
pub struct Pgd {
    /// Total perturbation budget.
    pub epsilon: f32,
    /// Step size per PGD iteration.
    pub alpha: f32,
    /// Number of PGD steps.
    pub num_steps: usize,
    /// Whether to initialise with random noise inside the ε-ball.
    pub random_start: bool,
    /// Norm constraint.
    pub norm: PerturbationNorm,
}

impl Pgd {
    /// Construct PGD with L∞ norm, random start enabled.
    pub fn new(epsilon: f32, alpha: f32, num_steps: usize) -> Self {
        Self {
            epsilon,
            alpha,
            num_steps,
            random_start: true,
            norm: PerturbationNorm::Linf { epsilon },
        }
    }

    /// Construct PGD with a custom norm.
    pub fn with_norm(epsilon: f32, alpha: f32, num_steps: usize, norm: PerturbationNorm) -> Self {
        Self {
            epsilon,
            alpha,
            num_steps,
            random_start: true,
            norm,
        }
    }

    /// Run the full PGD attack.
    ///
    /// `grad_fn` receives the current adversarial candidate and returns the
    /// gradient of the loss w.r.t. that input.
    pub fn attack(&self, input: &[f32], grad_fn: impl Fn(&[f32]) -> Vec<f32>) -> Vec<f32> {
        let n = input.len();
        if n == 0 {
            return input.to_vec();
        }

        // Optional random start inside ε-ball
        let mut x_adv: Vec<f32> = if self.random_start {
            let mut rng = StdRng::seed_from_u64(12345);
            let eps = self.epsilon;
            match &self.norm {
                PerturbationNorm::Linf { .. } => input
                    .iter()
                    .map(|xi| {
                        let u: f32 = rng.random();
                        xi + (u * 2.0 - 1.0) * eps
                    })
                    .collect(),
                PerturbationNorm::L2 { .. } | PerturbationNorm::L1 { .. } => {
                    // Random direction uniformly sampled, scale to epsilon
                    let raw: Vec<f32> = (0..n)
                        .map(|_| {
                            // Box-Muller for Gaussian direction
                            let u1: f32 = rng.random::<f32>().max(1e-10);
                            let u2: f32 = rng.random();
                            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
                        })
                        .collect();
                    let norm = l2_norm(&raw).max(1e-12);
                    let scale = eps / norm;
                    input
                        .iter()
                        .zip(raw.iter())
                        .map(|(xi, ri)| xi + ri * scale)
                        .collect()
                }
            }
        } else {
            input.to_vec()
        };

        for _ in 0..self.num_steps {
            let gradient = grad_fn(&x_adv);
            x_adv = self.step(&x_adv, input, &gradient);
        }
        x_adv
    }

    /// Project a perturbation vector onto the epsilon ball.
    pub fn project(&self, delta: &[f32], epsilon: f32) -> Vec<f32> {
        match &self.norm {
            PerturbationNorm::Linf { .. } => project_linf(delta, epsilon),
            PerturbationNorm::L2 { .. } => project_l2(delta, epsilon),
            PerturbationNorm::L1 { .. } => project_l1(delta, epsilon),
        }
    }

    /// Perform one PGD update step.
    ///
    /// `x_adv` is the current adversarial candidate, `input` is the original
    /// example, and `gradient` is `∇_{x_adv} L`.
    pub fn step(&self, x_adv: &[f32], input: &[f32], gradient: &[f32]) -> Vec<f32> {
        let n = x_adv.len().min(input.len()).min(gradient.len());
        let alpha = self.alpha;
        let eps = self.epsilon;

        // Take a gradient step (sign for L∞, normalised for L2)
        let stepped: Vec<f32> = match &self.norm {
            PerturbationNorm::Linf { .. } => (0..n)
                .map(|i| x_adv[i] + alpha * gradient[i].signum())
                .collect(),
            PerturbationNorm::L2 { .. } => {
                let gnorm = l2_norm(gradient).max(1e-12);
                (0..n)
                    .map(|i| x_adv[i] + alpha * gradient[i] / gnorm)
                    .collect()
            }
            PerturbationNorm::L1 { .. } => {
                // Frank-Wolfe step for L1
                let max_idx = gradient
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.abs()
                            .partial_cmp(&b.abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let mut out = x_adv.to_vec();
                if max_idx < out.len() {
                    out[max_idx] += alpha * gradient[max_idx].signum();
                }
                out
            }
        };

        // Project perturbation back onto ε-ball around original input
        let delta: Vec<f32> = stepped
            .iter()
            .zip(input.iter())
            .map(|(xs, xi)| xs - xi)
            .collect();
        let projected_delta = self.project(&delta, eps);
        projected_delta
            .iter()
            .zip(input.iter())
            .map(|(d, xi)| xi + d)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adversarial Trainer
// ─────────────────────────────────────────────────────────────────────────────

/// Type of inner-maximisation attack used during adversarial training.
#[derive(Debug, Clone)]
pub enum AttackType {
    /// FGSM single-step attack.
    Fgsm(Fgsm),
    /// PGD multi-step attack.
    Pgd(Pgd),
    /// Simplified Carlini-Wagner attack (confidence-margin-based).
    CW {
        /// Confidence margin for the attack.
        confidence: f32,
        /// Learning rate for the CW optimiser.
        lr: f32,
        /// Number of optimisation steps.
        num_steps: usize,
    },
}

/// High-level wrapper that combines an inner attack with training-loss utilities.
#[derive(Debug, Clone)]
pub struct AdversarialTrainer {
    /// The adversarial attack to use.
    pub attack: AttackType,
    /// β for TRADES loss (Eq. 1 in Zhang et al. 2019).
    pub trades_beta: f32,
}

impl AdversarialTrainer {
    /// Create a trainer backed by FGSM.
    pub fn new_fgsm(epsilon: f32) -> Self {
        Self {
            attack: AttackType::Fgsm(Fgsm::new(epsilon)),
            trades_beta: 6.0,
        }
    }

    /// Create a trainer backed by PGD.
    pub fn new_pgd(epsilon: f32, alpha: f32, steps: usize) -> Self {
        Self {
            attack: AttackType::Pgd(Pgd::new(epsilon, alpha, steps)),
            trades_beta: 6.0,
        }
    }

    /// Generate an adversarial example using the configured attack.
    ///
    /// `grad_fn` computes `∇_x L` for any given input `x`.
    pub fn adversarial_example(
        &self,
        input: &[f32],
        grad_fn: impl Fn(&[f32]) -> Vec<f32>,
    ) -> Vec<f32> {
        match &self.attack {
            AttackType::Fgsm(fgsm) => {
                let gradient = grad_fn(input);
                fgsm.perturb(input, &gradient)
                    .unwrap_or_else(|_| input.to_vec())
            }
            AttackType::Pgd(pgd) => pgd.attack(input, grad_fn),
            AttackType::CW {
                confidence,
                lr,
                num_steps,
            } => cw_attack_simplified(input, *confidence, *lr, *num_steps, grad_fn),
        }
    }

    /// Compute the TRADES loss (Zhang et al. 2019).
    ///
    /// ```text
    /// L_TRADES = CE(f(x), y) + β · KL( f(x_adv) ∥ f(x) )
    /// ```
    ///
    /// Returns `(natural_loss, robust_loss, trades_loss)`.
    ///
    /// # Errors
    /// Returns `TensorError` if any input slice is empty or lengths mismatch.
    pub fn trades_loss(
        &self,
        natural_output: &[f32],
        robust_output: &[f32],
        label_idx: usize,
    ) -> Result<(f32, f32, f32)> {
        if natural_output.is_empty() {
            return Err(TensorError::invalid_argument(
                "natural_output must not be empty".to_string(),
            ));
        }
        if robust_output.len() != natural_output.len() {
            return Err(TensorError::invalid_argument(format!(
                "natural_output length {} != robust_output length {}",
                natural_output.len(),
                robust_output.len()
            )));
        }
        if label_idx >= natural_output.len() {
            return Err(TensorError::invalid_argument(format!(
                "label_idx {label_idx} out of range for output length {}",
                natural_output.len()
            )));
        }

        let natural_loss = cross_entropy_loss(natural_output, label_idx);
        let robust_loss = kl_divergence(robust_output, natural_output);
        let trades_loss = natural_loss + self.trades_beta * robust_loss;

        Ok((natural_loss, robust_loss, trades_loss))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Simplified Carlini-Wagner attack
// ─────────────────────────────────────────────────────────────────────────────

/// Simplified CW attack: gradient-descent in input space with confidence margin.
///
/// This is an Adam-like optimisation loop in the *tanh* change-of-variables.
fn cw_attack_simplified(
    input: &[f32],
    confidence: f32,
    lr: f32,
    num_steps: usize,
    grad_fn: impl Fn(&[f32]) -> Vec<f32>,
) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return input.to_vec();
    }
    // Clamp to avoid atanh domain issues
    let w: Vec<f32> = input
        .iter()
        .map(|xi| {
            let clamped = xi.clamp(-0.999, 0.999);
            // atanh(x) = 0.5 * ln((1+x)/(1-x))
            0.5 * ((1.0 + clamped) / (1.0 - clamped)).ln()
        })
        .collect();

    // Adam moment estimates
    let mut m = vec![0.0_f32; n];
    let mut v = vec![0.0_f32; n];
    let (beta1, beta2) = (0.9_f32, 0.999_f32);
    let epsilon_adam = 1e-8_f32;

    let mut w_cur = w;
    for step in 1..=num_steps {
        let x_cur: Vec<f32> = w_cur.iter().map(|wi| wi.tanh()).collect();
        let grad_x = grad_fn(&x_cur);
        // Chain-rule through tanh: ∂x/∂w = 1 - tanh²(w)
        let grad_w: Vec<f32> = grad_x
            .iter()
            .zip(w_cur.iter())
            .map(|(gx, wi)| {
                let t = wi.tanh();
                gx * (1.0 - t * t) + confidence
            })
            .collect();

        let t = step as f32;
        for i in 0..n {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad_w[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad_w[i] * grad_w[i];
            let m_hat = m[i] / (1.0 - beta1.powf(t));
            let v_hat = v[i] / (1.0 - beta2.powf(t));
            w_cur[i] += lr * m_hat / (v_hat.sqrt() + epsilon_adam);
        }
    }
    w_cur.iter().map(|wi| wi.tanh()).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Randomized Smoothing
// ─────────────────────────────────────────────────────────────────────────────

/// Certified L2 robustness via randomized smoothing (Cohen et al., 2019).
///
/// The certify bound is `r = σ · Φ^{-1}(p_A)` where `Φ` is the standard
/// normal CDF.  Certification requires `p_A > 0.5`.
#[derive(Debug, Clone)]
pub struct RandomizedSmoothing {
    /// Standard deviation of the isotropic Gaussian noise added to inputs.
    pub noise_std: f32,
    /// Number of Monte-Carlo samples used for probability estimation.
    pub num_samples: usize,
    /// Confidence level (1 − α), e.g. 0.999.
    pub confidence: f32,
}

impl RandomizedSmoothing {
    /// Construct with default confidence 0.999.
    pub fn new(noise_std: f32, num_samples: usize) -> Self {
        Self {
            noise_std,
            num_samples,
            confidence: 0.999,
        }
    }

    /// Compute the certified L2 radius for the majority class.
    ///
    /// Returns `Some(r)` if `p_a > 0.5`, `None` otherwise.
    ///
    /// Formula: `r = σ · Φ^{-1}(p_A)`.
    pub fn certify_radius(&self, p_a: f32) -> Option<f32> {
        if p_a <= 0.5 {
            return None;
        }
        Some(self.noise_std * normal_quantile(p_a))
    }

    /// Certified radius (returns 0.0 when `p_a ≤ 0.5`).
    pub fn certified_radius(&self, p_a: f32) -> f32 {
        if p_a <= 0.5 {
            return 0.0;
        }
        self.noise_std * normal_quantile(p_a)
    }

    /// Return `true` when the point is certifiably robust.
    ///
    /// Requires `p_A − p_B > 0` after accounting for the confidence-level
    /// threshold (simplified: checks `p_a > p_b`).
    pub fn is_certifiable(&self, p_a: f32, p_b: f32) -> bool {
        p_a > p_b && p_a > 0.5
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Normal quantile (Φ^{-1}) — rational approximation (Beasley & Springer, 1977)
// ─────────────────────────────────────────────────────────────────────────────

/// Inverse standard normal CDF Φ^{-1}(p) for `p ∈ (0, 1)`.
///
/// Uses Beasley-Springer-Moro rational approximation with max error ~4.5e-4.
fn normal_quantile(p: f32) -> f32 {
    debug_assert!(p > 0.0 && p < 1.0, "p must be in (0,1)");
    let p = p.clamp(1e-7, 1.0 - 1e-7);

    // Rational approximation constants
    let a = [
        -3.969683028665376e+01_f64,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01_f64,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03_f64,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03_f64,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_lo = 0.02425_f64;
    let p_hi = 1.0 - p_lo;
    let p = p as f64;

    let q = if p < p_lo {
        // Lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_hi {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        // Upper tail by symmetry
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    };
    q as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Norm utilities
// ─────────────────────────────────────────────────────────────────────────────

/// L∞ norm: `max |v_i|`.
pub fn linf_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max)
}

/// L2 norm: `sqrt(Σ v_i²)`.
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// L1 norm: `Σ |v_i|`.
pub fn l1_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).sum()
}

/// Normalise to L∞ unit: `sign(v)`.
pub fn normalize_linf(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| x.signum()).collect()
}

/// Normalise to L2 unit: `v / ||v||_2`.  Returns zeros for zero vector.
pub fn normalize_l2(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm < 1e-12 {
        vec![0.0; v.len()]
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Project `delta` onto L∞ ball: clamp each element to `[-ε, ε]`.
pub fn project_linf(delta: &[f32], epsilon: f32) -> Vec<f32> {
    delta.iter().map(|d| d.clamp(-epsilon, epsilon)).collect()
}

/// Project `delta` onto L2 ball of radius `epsilon`.
///
/// If `||delta||_2 ≤ epsilon`, returns `delta` unchanged.
/// Otherwise scales so `||delta||_2 = epsilon`.
pub fn project_l2(delta: &[f32], epsilon: f32) -> Vec<f32> {
    let norm = l2_norm(delta);
    if norm <= epsilon {
        delta.to_vec()
    } else {
        delta.iter().map(|d| d * epsilon / norm).collect()
    }
}

/// Project `delta` onto L1 ball of radius `epsilon` (Duchi et al. 2008).
///
/// Uses the soft-thresholding / simplex projection algorithm.
pub fn project_l1(delta: &[f32], epsilon: f32) -> Vec<f32> {
    let n = delta.len();
    if n == 0 {
        return Vec::new();
    }
    // Check if already feasible
    if l1_norm(delta) <= epsilon {
        return delta.to_vec();
    }
    // Simplex projection on absolute values
    let u: Vec<f32> = {
        let mut abs_vals: Vec<f32> = delta.iter().map(|d| d.abs()).collect();
        abs_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        abs_vals
    };
    let mut cssv = 0.0_f32;
    let mut rho = 0_usize;
    for (j, uj) in u.iter().enumerate() {
        cssv += uj;
        let cond = uj - (cssv - epsilon) / (j as f32 + 1.0);
        if cond > 0.0 {
            rho = j;
        }
    }
    let cssv_rho: f32 = u[..=rho].iter().sum();
    let theta = (cssv_rho - epsilon) / (rho as f32 + 1.0);
    delta
        .iter()
        .map(|d| {
            let abs_d = d.abs();
            let proj = abs_d - theta;
            if proj <= 0.0 {
                0.0
            } else {
                proj * d.signum()
            }
        })
        .collect()
}

/// Cross-entropy loss: `-log(p[target])`.
///
/// `probs` must be a probability distribution (should sum to 1).
pub fn cross_entropy_loss(probs: &[f32], target_idx: usize) -> f32 {
    let p = probs.get(target_idx).copied().unwrap_or(0.0).max(1e-12);
    -p.ln()
}

/// KL divergence: `Σ p_i log(p_i / q_i)`.
///
/// Treats `0 · log(0)` as 0.  `q_i` is clamped to avoid log(0).
pub fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q.iter())
        .map(|(pi, qi)| {
            if *pi <= 0.0 {
                0.0
            } else {
                let qi_safe = qi.max(1e-12);
                pi * (pi / qi_safe).ln()
            }
        })
        .sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: softmax
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    // ── norm utilities ───────────────────────────────────────────────────────

    #[test]
    fn test_l2_norm_correct() {
        let v = vec![3.0_f32, 4.0];
        let norm = l2_norm(&v);
        assert!(
            (norm - 5.0).abs() < 1e-5,
            "l2_norm should be 5.0, got {norm}"
        );
    }

    #[test]
    fn test_l1_norm_correct() {
        let v = vec![-1.0_f32, 2.0, -3.0];
        let norm = l1_norm(&v);
        assert!(
            (norm - 6.0).abs() < 1e-5,
            "l1_norm should be 6.0, got {norm}"
        );
    }

    #[test]
    fn test_linf_norm_correct() {
        let v = vec![1.0_f32, -5.0, 3.0];
        let norm = linf_norm(&v);
        assert!(
            (norm - 5.0).abs() < 1e-5,
            "linf_norm should be 5.0, got {norm}"
        );
    }

    #[test]
    fn test_normalize_l2_unit_norm() {
        let v = vec![3.0_f32, 4.0];
        let u = normalize_l2(&v);
        let norm = l2_norm(&u);
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "normalized L2 norm should be 1.0, got {norm}"
        );
    }

    #[test]
    fn test_normalize_l2_zero_input() {
        let v = vec![0.0_f32, 0.0];
        let u = normalize_l2(&v);
        assert_eq!(u, vec![0.0, 0.0]);
    }

    // ── project_linf ─────────────────────────────────────────────────────────

    #[test]
    fn test_project_linf_clips_to_epsilon() {
        let delta = vec![0.1_f32, -0.2, 0.05, -0.15];
        let eps = 0.1;
        let proj = project_linf(&delta, eps);
        for d in &proj {
            assert!(*d >= -eps - 1e-6 && *d <= eps + 1e-6);
        }
        // Elements within range unchanged
        assert!((proj[0] - 0.1).abs() < 1e-6);
        assert!((proj[2] - 0.05).abs() < 1e-6);
    }

    // ── project_l2 ───────────────────────────────────────────────────────────

    #[test]
    fn test_project_l2_respects_ball() {
        let delta = vec![3.0_f32, 4.0]; // L2 norm = 5
        let eps = 2.0;
        let proj = project_l2(&delta, eps);
        let norm = l2_norm(&proj);
        assert!(
            (norm - eps).abs() < 1e-5,
            "projected L2 norm should be {eps}, got {norm}"
        );
    }

    #[test]
    fn test_project_l2_inside_ball_unchanged() {
        let delta = vec![0.5_f32, 0.5]; // L2 norm < 1.0
        let eps = 2.0;
        let proj = project_l2(&delta, eps);
        assert!((proj[0] - 0.5).abs() < 1e-6 && (proj[1] - 0.5).abs() < 1e-6);
    }

    // ── cross_entropy / kl ───────────────────────────────────────────────────

    #[test]
    fn test_cross_entropy_loss_correct() {
        let probs = softmax(&[2.0_f32, 1.0, 0.1]);
        let ce = cross_entropy_loss(&probs, 0);
        // p[0] = softmax([2,1,0.1])[0] ≈ 0.659
        assert!(ce > 0.0, "CE loss should be positive");
        assert!(
            ce < 1.0,
            "CE loss for high-confidence correct class should be < 1"
        );
    }

    #[test]
    fn test_kl_divergence_zero_for_identical() {
        let p = vec![0.5_f32, 0.3, 0.2];
        let kl = kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-5, "KL(p||p) should be 0.0, got {kl}");
    }

    #[test]
    fn test_kl_divergence_non_negative() {
        let p = vec![0.5_f32, 0.3, 0.2];
        let q = vec![0.1_f32, 0.6, 0.3];
        let kl = kl_divergence(&p, &q);
        assert!(kl >= 0.0, "KL divergence should be non-negative, got {kl}");
    }

    // ── FGSM ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_fgsm_sign_matches_gradient_sign() {
        let input = vec![0.0_f32, 0.0, 0.0, 0.0];
        let gradient = vec![0.5_f32, -1.0, 2.0, -0.3];
        let fgsm = Fgsm::new(0.1);
        let x_adv = fgsm
            .perturb(&input, &gradient)
            .expect("perturb should succeed");

        // Each adversarial element should move in the direction of the gradient sign.
        // x_adv[i] - input[i] should equal epsilon * sign(gradient[i])
        for (i, (xa, g)) in x_adv.iter().zip(gradient.iter()).enumerate() {
            let delta = xa - input[i];
            let expected = 0.1 * g.signum();
            assert!(
                (delta - expected).abs() < 1e-6,
                "x_adv[{i}]: expected delta {expected}, got {delta}"
            );
        }
    }

    #[test]
    fn test_fgsm_clip_to_range_respects_bounds() {
        let input = vec![0.5_f32, 0.5, 0.5];
        let x_adv = vec![0.7_f32, 0.3, 0.9]; // some out of epsilon, some out of range
        let fgsm = Fgsm::new(0.1);
        let clipped = fgsm.clip_to_range(&x_adv, &input, 0.0, 1.0);

        // Each element should be in [x_orig - eps, x_orig + eps] AND [0.0, 1.0]
        for (c, o) in clipped.iter().zip(input.iter()) {
            assert!(*c >= (o - 0.1).max(0.0) - 1e-6);
            assert!(*c <= (o + 0.1).min(1.0) + 1e-6);
        }
    }

    #[test]
    fn test_fgsm_empty_input_error() {
        let fgsm = Fgsm::new(0.1);
        let result = fgsm.perturb(&[], &[]);
        assert!(matches!(result, Err(AdversarialError::EmptyInput)));
    }

    #[test]
    fn test_fgsm_dimension_mismatch_error() {
        let fgsm = Fgsm::new(0.1);
        let result = fgsm.perturb(&[1.0, 2.0], &[1.0]);
        assert!(matches!(
            result,
            Err(AdversarialError::DimensionMismatch {
                input: 2,
                gradient: 1
            })
        ));
    }

    #[test]
    fn test_fgsm_l2_norm_variant() {
        let input = vec![0.0_f32; 4];
        let gradient = vec![3.0_f32, 4.0, 0.0, 0.0]; // L2 norm = 5
        let fgsm = Fgsm::with_norm(1.0, PerturbationNorm::L2 { epsilon: 1.0 });
        let x_adv = fgsm
            .perturb(&input, &gradient)
            .expect("perturb should succeed");
        // perturbation should have L2 norm ≈ 1.0
        let delta: Vec<f32> = x_adv.iter().zip(input.iter()).map(|(a, b)| a - b).collect();
        let delta_norm = l2_norm(&delta);
        assert!(
            (delta_norm - 1.0).abs() < 1e-5,
            "L2 FGSM delta norm should be 1.0, got {delta_norm}"
        );
    }

    // ── PGD ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_pgd_step_count() {
        // Dummy gradient function that counts calls
        let count = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        let count_clone = count.clone();
        let grad_fn = move |x: &[f32]| {
            let mut c = count_clone.lock().unwrap_or_else(|e| e.into_inner());
            *c += 1;
            vec![1.0_f32; x.len()]
        };

        let pgd = Pgd::new(0.1, 0.01, 5);
        let _ = pgd.attack(&[0.0_f32; 3], grad_fn);
        let final_count = *count.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            final_count, 5,
            "PGD should call grad_fn exactly num_steps times"
        );
    }

    #[test]
    fn test_pgd_project_linf_clips() {
        let pgd = Pgd::new(0.1, 0.01, 10);
        let delta = vec![0.2_f32, -0.05, 0.15];
        let proj = pgd.project(&delta, 0.1);
        for d in &proj {
            assert!(*d >= -0.1 - 1e-6 && *d <= 0.1 + 1e-6);
        }
    }

    #[test]
    fn test_pgd_project_l2_respects_ball() {
        let pgd = Pgd::with_norm(1.0, 0.1, 10, PerturbationNorm::L2 { epsilon: 1.0 });
        let delta = vec![3.0_f32, 4.0]; // L2 norm = 5 > 1
        let proj = pgd.project(&delta, 1.0);
        let norm = l2_norm(&proj);
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "projected L2 norm should be 1.0, got {norm}"
        );
    }

    #[test]
    fn test_pgd_step_stays_in_epsilon_ball() {
        let pgd = Pgd::new(0.1, 0.01, 1);
        let input = vec![0.5_f32; 4];
        let gradient = vec![1.0_f32; 4];
        let x_adv = pgd.step(&input, &input, &gradient);
        let delta: Vec<f32> = x_adv.iter().zip(input.iter()).map(|(a, b)| a - b).collect();
        let linf = linf_norm(&delta);
        assert!(
            linf <= 0.1 + 1e-6,
            "PGD step should stay in epsilon ball, got linf={linf}"
        );
    }

    // ── AdversarialTrainer ────────────────────────────────────────────────────

    #[test]
    fn test_adversarial_example_differs_from_input() {
        let trainer = AdversarialTrainer::new_fgsm(0.1);
        let input = vec![0.5_f32; 5];
        // Gradient function that always returns a constant non-zero gradient
        let grad_fn = |_x: &[f32]| vec![1.0_f32; 5];
        let x_adv = trainer.adversarial_example(&input, grad_fn);
        assert_ne!(x_adv, input, "adversarial example should differ from input");
    }

    #[test]
    fn test_adversarial_trainer_pgd_differs_from_input() {
        let trainer = AdversarialTrainer::new_pgd(0.1, 0.01, 5);
        let input = vec![0.0_f32; 4];
        let grad_fn = |_x: &[f32]| vec![1.0_f32, -1.0, 0.5, -0.5];
        let x_adv = trainer.adversarial_example(&input, grad_fn);
        assert_ne!(
            x_adv, input,
            "PGD adversarial example should differ from input"
        );
    }

    #[test]
    fn test_trades_loss_components() {
        let nat_logits = vec![2.0_f32, 1.0, 0.1];
        let adv_logits = vec![1.5_f32, 1.2, 0.3];
        let nat_probs = softmax(&nat_logits);
        let adv_probs = softmax(&adv_logits);

        let trainer = AdversarialTrainer::new_pgd(0.03, 0.007, 10);
        let (nat_loss, rob_loss, trades_loss) = trainer
            .trades_loss(&nat_probs, &adv_probs, 0)
            .expect("TRADES loss should succeed");

        assert!(nat_loss > 0.0, "natural loss should be positive");
        assert!(rob_loss >= 0.0, "robust loss (KL) should be non-negative");
        let expected = nat_loss + trainer.trades_beta * rob_loss;
        assert!(
            (trades_loss - expected).abs() < 1e-5,
            "TRADES loss should be nat + beta*rob"
        );
    }

    #[test]
    fn test_trades_loss_out_of_range_label() {
        let probs = vec![0.5_f32, 0.3, 0.2];
        let trainer = AdversarialTrainer::new_fgsm(0.03);
        let result = trainer.trades_loss(&probs, &probs, 10);
        assert!(
            result.is_err(),
            "out-of-range label should produce an error"
        );
    }

    // ── RandomizedSmoothing ───────────────────────────────────────────────────

    #[test]
    fn test_certify_radius_returns_none_below_half() {
        let smoother = RandomizedSmoothing::new(0.25, 1000);
        assert!(
            smoother.certify_radius(0.5).is_none(),
            "certify_radius should be None when p_a == 0.5"
        );
        assert!(
            smoother.certify_radius(0.3).is_none(),
            "certify_radius should be None when p_a < 0.5"
        );
    }

    #[test]
    fn test_certify_radius_positive_for_high_p_a() {
        let smoother = RandomizedSmoothing::new(0.25, 1000);
        let r = smoother.certify_radius(0.9);
        assert!(
            matches!(r, Some(x) if x > 0.0),
            "certify_radius should be positive for p_a = 0.9"
        );
    }

    #[test]
    fn test_certified_radius_zero_at_half() {
        let smoother = RandomizedSmoothing::new(0.25, 1000);
        assert_eq!(smoother.certified_radius(0.5), 0.0);
    }

    #[test]
    fn test_is_certifiable() {
        let smoother = RandomizedSmoothing::new(0.25, 1000);
        assert!(smoother.is_certifiable(0.8, 0.2), "should be certifiable");
        assert!(
            !smoother.is_certifiable(0.4, 0.3),
            "p_a < 0.5 should not be certifiable"
        );
        assert!(
            !smoother.is_certifiable(0.6, 0.7),
            "p_a < p_b should not be certifiable"
        );
    }

    #[test]
    fn test_certified_radius_scales_with_noise_std() {
        let smoother1 = RandomizedSmoothing::new(0.25, 1000);
        let smoother2 = RandomizedSmoothing::new(0.5, 1000);
        let r1 = smoother1.certified_radius(0.9);
        let r2 = smoother2.certified_radius(0.9);
        assert!(
            (r2 - 2.0 * r1).abs() < 1e-4,
            "radius should scale linearly with noise_std"
        );
    }

    #[test]
    fn test_adversarial_error_display() {
        let e = AdversarialError::DimensionMismatch {
            input: 3,
            gradient: 5,
        };
        let s = format!("{e}");
        assert!(s.contains("3") && s.contains("5"));
    }
}
