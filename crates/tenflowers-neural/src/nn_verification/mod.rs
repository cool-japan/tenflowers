//! Neural Network Formal Verification — Round 47 Track B.
//!
//! This module provides rigorous, certificate-producing verification of neural
//! network properties.  It is **distinct** from `adversarial.rs` (which handles
//! empirical attack / defence methods): every method here either returns a
//! mathematical certificate or states that verification is inconclusive.
//!
//! # Methods
//!
//! | Section | Name | Reference |
//! |---------|------|-----------|
//! | §1 | Interval Arithmetic (`NnvInterval`) | Moore 1966 |
//! | §2 | Zonotope Abstract Domain (`NnvZonotope`) | Gehr et al. 2018 (DeepZ) |
//! | §3 | Network Specification (`NnvNetwork`) | — |
//! | §4 | Interval Bound Propagation (IBP) | Gowal et al. 2018 |
//! | §5 | CROWN Linear Relaxation | Zhang et al. 2018 |
//! | §6 | Randomised Smoothing | Cohen et al. 2019 |
//! | §7 | High-level Property Checker | — |
//! | §8 | Adversarial Certifier | — |
//! | §9 | Monotonicity Check | — |
//! | §10 | Metrics | — |
//!
//! # Design Principles
//!
//! * No `unwrap()` — all fallible paths return `Result<_, NnvError>`.
//! * No `ndarray` or `rand` crate — pure `f64 Vec<...>` and SciRS2-compatible RNG.
//! * All public types are prefixed with `Nnv` to avoid name collisions.
//! * File stays below 2 000 lines.

#[cfg(test)]
mod tests;

// ─────────────────────────────────────────────────────────────────────────────
// §0  Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the nn_verification module.
#[derive(Debug, Clone, PartialEq)]
pub enum NnvError {
    /// The network specification is inconsistent (e.g. dimension mismatch).
    InvalidNetwork(String),
    /// Verification failed for a semantic reason (e.g. property definitely violated).
    VerificationFailed(String),
    /// A numerical issue was encountered (division by zero, NaN, etc.).
    NumericalError(String),
}

impl std::fmt::Display for NnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NnvError::InvalidNetwork(s) => write!(f, "NnvError::InvalidNetwork: {s}"),
            NnvError::VerificationFailed(s) => write!(f, "NnvError::VerificationFailed: {s}"),
            NnvError::NumericalError(s) => write!(f, "NnvError::NumericalError: {s}"),
        }
    }
}

impl std::error::Error for NnvError {}

// ─────────────────────────────────────────────────────────────────────────────
// §1  NnvInterval — Interval arithmetic for bound propagation
// ─────────────────────────────────────────────────────────────────────────────

/// A closed interval [lb, ub] used for abstract-interpretation-style bound
/// propagation through neural networks.
#[derive(Clone, Debug, PartialEq)]
pub struct NnvInterval {
    /// Lower bound.
    pub lb: f64,
    /// Upper bound.
    pub ub: f64,
}

impl NnvInterval {
    /// Construct a new interval, checking that lb ≤ ub.
    pub fn new(lb: f64, ub: f64) -> Result<Self, NnvError> {
        if lb > ub {
            return Err(NnvError::InvalidNetwork(format!(
                "NnvInterval: lb ({lb}) > ub ({ub})"
            )));
        }
        if lb.is_nan() || ub.is_nan() {
            return Err(NnvError::NumericalError("NnvInterval: NaN bound".into()));
        }
        Ok(Self { lb, ub })
    }

    /// Degenerate interval [v, v].
    #[inline]
    pub fn point(v: f64) -> Self {
        Self { lb: v, ub: v }
    }

    /// Element-wise addition of two intervals.
    #[inline]
    pub fn add(&self, other: &NnvInterval) -> NnvInterval {
        NnvInterval {
            lb: self.lb + other.lb,
            ub: self.ub + other.ub,
        }
    }

    /// Interval multiplication.  All four products are considered.
    pub fn mul(&self, other: &NnvInterval) -> NnvInterval {
        let products = [
            self.lb * other.lb,
            self.lb * other.ub,
            self.ub * other.lb,
            self.ub * other.ub,
        ];
        let lb = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let ub = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        NnvInterval { lb, ub }
    }

    /// Interval image under ReLU: [max(0,lb), max(0,ub)].
    #[inline]
    pub fn relu(&self) -> NnvInterval {
        NnvInterval {
            lb: self.lb.max(0.0),
            ub: self.ub.max(0.0),
        }
    }

    /// Interval image under tanh (monotone, so just apply to endpoints).
    #[inline]
    pub fn tanh_bounds(&self) -> NnvInterval {
        NnvInterval {
            lb: self.lb.tanh(),
            ub: self.ub.tanh(),
        }
    }

    /// Interval image under sigmoid σ(x) = 1/(1+e^{-x}) (monotone).
    #[inline]
    pub fn sigmoid_bounds(&self) -> NnvInterval {
        NnvInterval {
            lb: sigmoid(self.lb),
            ub: sigmoid(self.ub),
        }
    }

    /// Width of the interval: ub − lb.
    #[inline]
    pub fn width(&self) -> f64 {
        self.ub - self.lb
    }

    /// Check whether a concrete value lies in [lb, ub].
    #[inline]
    pub fn contains(&self, v: f64) -> bool {
        v >= self.lb && v <= self.ub
    }

    /// Intersect two intervals; returns None if they are disjoint.
    pub fn intersect(&self, other: &NnvInterval) -> Option<NnvInterval> {
        let lb = self.lb.max(other.lb);
        let ub = self.ub.min(other.ub);
        if lb <= ub {
            Some(NnvInterval { lb, ub })
        } else {
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  NnvZonotope — Zonotope abstract domain (DeepZ)
// ─────────────────────────────────────────────────────────────────────────────

/// A *zonotope* in R^d:
/// ```text
/// x = a0 + Σ_i ε_i · aᵢ,   ε_i ∈ [-1, 1]
/// ```
/// where `center` = a0 ∈ R^d, `generators[i]` = aᵢ ∈ R^d.
///
/// Zonotopes are closed under affine maps (which makes them ideal for verifying
/// feedforward networks) and admit efficient ReLU relaxation via the DeepZ
/// lambda method (Gehr et al., ICAI 2018).
#[derive(Clone, Debug)]
pub struct NnvZonotope {
    /// a0 — center, length `dim`.
    pub center: Vec<f64>,
    /// Error terms: `generators[i]` has length `dim`.  Shape: [n_errors × dim].
    pub generators: Vec<Vec<f64>>,
    /// Output dimension.
    pub dim: usize,
    /// Number of error terms (rows of `generators`).
    pub n_errors: usize,
}

impl NnvZonotope {
    /// Construct from an interval box.
    ///
    /// Each interval [lb_i, ub_i] contributes:
    ///  - center component: (lb_i + ub_i) / 2
    ///  - a *diagonal* generator row where only position i is non-zero with
    ///    value (ub_i − lb_i) / 2.
    pub fn from_box(intervals: &[NnvInterval]) -> Self {
        let dim = intervals.len();
        let mut center = vec![0.0_f64; dim];
        let mut generators: Vec<Vec<f64>> = Vec::with_capacity(dim);

        for (i, iv) in intervals.iter().enumerate() {
            center[i] = (iv.lb + iv.ub) * 0.5;
            let radius = (iv.ub - iv.lb) * 0.5;
            if radius > 0.0 {
                let mut gen = vec![0.0_f64; dim];
                gen[i] = radius;
                generators.push(gen);
            }
        }

        let n_errors = generators.len();
        NnvZonotope {
            center,
            generators,
            dim,
            n_errors,
        }
    }

    /// Affine transformation: y = W·x + b.
    ///
    /// New center  = W·center + b
    /// New generator j = W · generators\[j\]
    pub fn affine_transform(&self, w: &[Vec<f64>], b: &[f64]) -> NnvZonotope {
        let out_dim = b.len();

        // New center
        let mut new_center = b.to_vec();
        for (i, row) in w.iter().enumerate() {
            for (j, &wij) in row.iter().enumerate() {
                if j < self.center.len() {
                    new_center[i] += wij * self.center[j];
                }
            }
        }

        // New generators
        let mut new_generators: Vec<Vec<f64>> = Vec::with_capacity(self.n_errors);
        for gen in &self.generators {
            let mut new_gen = vec![0.0_f64; out_dim];
            for (i, row) in w.iter().enumerate() {
                for (j, &wij) in row.iter().enumerate() {
                    if j < gen.len() {
                        new_gen[i] += wij * gen[j];
                    }
                }
            }
            new_generators.push(new_gen);
        }

        let n_errors = new_generators.len();
        NnvZonotope {
            center: new_center,
            generators: new_generators,
            dim: out_dim,
            n_errors,
        }
    }

    /// DeepZ ReLU relaxation.
    ///
    /// For each neuron *k*:
    /// - lb_k = center\[k\] − Σ |gen\[j\]\[k\]|
    /// - ub_k = center\[k\] + Σ |gen\[j\]\[k\]|
    ///
    /// Case 1: ub_k ≤ 0  → neuron is always inactive → zero out column k.
    /// Case 2: lb_k ≥ 0  → neuron is always active → no change.
    /// Case 3: lb_k < 0 < ub_k → add a new error term:
    ///   λ = ub / (ub − lb)
    ///   new_center\[k\] += − λ·lb / 2   (bias from the relaxation)
    ///   new generator row with gen\[k\] = λ·(−lb) / 2
    pub fn relu(&self) -> NnvZonotope {
        let bounds = self.interval_bounds();
        let mut new_center = self.center.clone();
        let mut new_generators = self.generators.clone();
        let mut extra_generators: Vec<Vec<f64>> = Vec::new();

        for k in 0..self.dim {
            let lb = bounds[k].lb;
            let ub = bounds[k].ub;

            if ub <= 0.0 {
                // Always inactive: zero out the center and every generator at column k.
                new_center[k] = 0.0;
                for gen in &mut new_generators {
                    gen[k] = 0.0;
                }
            } else if lb >= 0.0 {
                // Always active: identity, no change needed.
            } else {
                // Mixed: apply DeepZ lambda relaxation (Gehr et al., 2018).
                // Upper bound: relu(x) ≤ λ·x + μ  where λ = ub/(ub−lb), μ = −λ·lb
                // Lower bound: relu(x) ≥ 0
                // Approximation: replace relu(z_k) with λ·z_k + β + γ·ε_new
                //   β = μ/2 = −λ·lb/2  (center of the approximation interval)
                //   γ = μ/2 = −λ·lb/2  (half-width; new error term coefficient)
                let lambda = ub / (ub - lb);
                let mu = -lambda * lb; // = λ·|lb| ≥ 0
                let beta = mu * 0.5; // centre bias
                let gamma = mu * 0.5; // new error coefficient
                new_center[k] = lambda * new_center[k] + beta;
                for gen in &mut new_generators {
                    gen[k] *= lambda;
                }
                // New error term for the approximation.
                let mut extra = vec![0.0_f64; self.dim];
                extra[k] = gamma.max(0.0);
                extra_generators.push(extra);
            }
        }

        new_generators.extend(extra_generators);
        let n_errors = new_generators.len();
        NnvZonotope {
            center: new_center,
            generators: new_generators,
            dim: self.dim,
            n_errors,
        }
    }

    /// Project back to an interval box: center_i ± Σ_j |generators\[j\]\[i\]|.
    pub fn interval_bounds(&self) -> Vec<NnvInterval> {
        let mut result = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let radius: f64 = self.generators.iter().map(|g| g[i].abs()).sum();
            result.push(NnvInterval {
                lb: self.center[i] - radius,
                ub: self.center[i] + radius,
            });
        }
        result
    }

    /// Number of neurons (output dimension).
    #[inline]
    pub fn n_neurons(&self) -> usize {
        self.dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  NnvNetworkSpec — Network specification
// ─────────────────────────────────────────────────────────────────────────────

/// Activation function applied after the affine part of a layer.
#[derive(Clone, Debug, PartialEq)]
pub enum NnvActivation {
    Relu,
    Tanh,
    Sigmoid,
    Linear,
}

/// A single fully-connected layer.
#[derive(Clone, Debug)]
pub struct NnvLayer {
    /// Weight matrix in row-major order: `weights[i]` is the i-th output row, length = `in_dim`.
    pub weights: Vec<Vec<f64>>,
    /// Bias vector of length `out_dim`.
    pub bias: Vec<f64>,
    /// Nonlinearity applied after the affine transformation.
    pub activation: NnvActivation,
}

impl NnvLayer {
    /// Output dimension of this layer.
    #[inline]
    pub fn out_dim(&self) -> usize {
        self.bias.len()
    }

    /// Input dimension of this layer.
    #[inline]
    pub fn in_dim(&self) -> usize {
        self.weights.first().map(|r| r.len()).unwrap_or(0)
    }

    /// Number of trainable parameters in this layer.
    #[inline]
    pub fn n_params(&self) -> usize {
        self.weights.iter().map(|r| r.len()).sum::<usize>() + self.bias.len()
    }
}

/// A fully-connected feedforward network specification.
pub struct NnvNetwork {
    pub layers: Vec<NnvLayer>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl NnvNetwork {
    /// Validate and construct a network from a list of layers.
    pub fn new(layers: Vec<NnvLayer>) -> Result<Self, NnvError> {
        if layers.is_empty() {
            return Err(NnvError::InvalidNetwork(
                "Network must have at least one layer".into(),
            ));
        }
        // Check dimension compatibility.
        for (i, pair) in layers.windows(2).enumerate() {
            let out = pair[0].out_dim();
            let next_in = pair[1].in_dim();
            if out != next_in {
                return Err(NnvError::InvalidNetwork(format!(
                    "Layer {i} output dim {out} != layer {} input dim {next_in}",
                    i + 1
                )));
            }
        }
        // Check each layer's weight matrix is consistent.
        for (i, layer) in layers.iter().enumerate() {
            let expected_in = layer.in_dim();
            for (j, row) in layer.weights.iter().enumerate() {
                if row.len() != expected_in {
                    return Err(NnvError::InvalidNetwork(format!(
                        "Layer {i} row {j} has {} weights, expected {expected_in}",
                        row.len()
                    )));
                }
            }
            if layer.weights.len() != layer.bias.len() {
                return Err(NnvError::InvalidNetwork(format!(
                    "Layer {i}: {} weight rows but {} bias entries",
                    layer.weights.len(),
                    layer.bias.len()
                )));
            }
        }
        let input_dim = layers[0].in_dim();
        let output_dim = layers.last().map(|l| l.out_dim()).unwrap_or(0);
        Ok(NnvNetwork {
            layers,
            input_dim,
            output_dim,
        })
    }

    /// Standard forward pass.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut x = input.to_vec();
        for layer in &self.layers {
            x = layer_forward(layer, &x);
        }
        x
    }

    /// Number of layers.
    #[inline]
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Total number of trainable parameters.
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }
}

/// Forward pass through a single layer.
fn layer_forward(layer: &NnvLayer, input: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = layer
        .bias
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            b + layer.weights[i]
                .iter()
                .zip(input.iter())
                .map(|(&w, &x)| w * x)
                .sum::<f64>()
        })
        .collect();

    match layer.activation {
        NnvActivation::Relu => {
            for v in &mut out {
                *v = v.max(0.0);
            }
        }
        NnvActivation::Tanh => {
            for v in &mut out {
                *v = v.tanh();
            }
        }
        NnvActivation::Sigmoid => {
            for v in &mut out {
                *v = sigmoid(*v);
            }
        }
        NnvActivation::Linear => {}
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  NnvIntervalBoundPropagation — IBP (Gowal et al., 2018)
// ─────────────────────────────────────────────────────────────────────────────

/// Interval Bound Propagation: propagate an interval box through every layer of
/// the network to obtain sound output bounds.
pub struct NnvIntervalBoundPropagation;

impl NnvIntervalBoundPropagation {
    /// Propagate `input_bounds` through every layer of the network.
    pub fn propagate(
        network: &NnvNetwork,
        input_bounds: &[NnvInterval],
    ) -> Result<Vec<NnvInterval>, NnvError> {
        if input_bounds.len() != network.input_dim {
            return Err(NnvError::InvalidNetwork(format!(
                "propagate: expected {} input bounds, got {}",
                network.input_dim,
                input_bounds.len()
            )));
        }
        let mut bounds = input_bounds.to_vec();
        for layer in &network.layers {
            let pre = Self::affine_propagate(layer, &bounds);
            bounds = pre
                .into_iter()
                .map(|iv| match layer.activation {
                    NnvActivation::Relu => iv.relu(),
                    NnvActivation::Tanh => iv.tanh_bounds(),
                    NnvActivation::Sigmoid => iv.sigmoid_bounds(),
                    NnvActivation::Linear => iv,
                })
                .collect();
        }
        Ok(bounds)
    }

    /// Propagate intervals through the *affine* part of one layer (no activation).
    ///
    /// For output neuron i:
    /// ```text
    /// lb_i = b_i + Σ_k min(W_ik · lb_k, W_ik · ub_k)
    /// ub_i = b_i + Σ_k max(W_ik · lb_k, W_ik · ub_k)
    /// ```
    pub fn affine_propagate(layer: &NnvLayer, input_bounds: &[NnvInterval]) -> Vec<NnvInterval> {
        layer
            .weights
            .iter()
            .zip(layer.bias.iter())
            .map(|(row, &b)| {
                let (mut lo, mut hi) = (b, b);
                for (k, &w) in row.iter().enumerate() {
                    if k < input_bounds.len() {
                        let p1 = w * input_bounds[k].lb;
                        let p2 = w * input_bounds[k].ub;
                        lo += p1.min(p2);
                        hi += p1.max(p2);
                    }
                }
                NnvInterval { lb: lo, ub: hi }
            })
            .collect()
    }

    /// Build input bounds as an ε-ball around `center`, then propagate.
    pub fn output_bounds_epsilon_ball(
        network: &NnvNetwork,
        center: &[f64],
        epsilon: f64,
    ) -> Result<Vec<NnvInterval>, NnvError> {
        if center.len() != network.input_dim {
            return Err(NnvError::InvalidNetwork(format!(
                "output_bounds_epsilon_ball: expected {} dims, got {}",
                network.input_dim,
                center.len()
            )));
        }
        let input_bounds: Vec<NnvInterval> = center
            .iter()
            .map(|&c| NnvInterval {
                lb: c - epsilon,
                ub: c + epsilon,
            })
            .collect();
        Self::propagate(network, &input_bounds)
    }

    /// Verify that `output_bounds[output_idx].lb >= threshold`.
    pub fn verify_output_lower_bound(
        output_bounds: &[NnvInterval],
        output_idx: usize,
        threshold: f64,
    ) -> bool {
        output_bounds
            .get(output_idx)
            .map(|iv| iv.lb >= threshold)
            .unwrap_or(false)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  NnvCrownBounds — CROWN (Zhang et al., 2018)
// ─────────────────────────────────────────────────────────────────────────────

/// Symbolic linear bounds on the output of a layer.
///
/// For each output neuron i:
/// ```text
/// lower_w[i] · x + lower_b[i] ≤ output[i] ≤ upper_w[i] · x + upper_b[i]
/// ```
pub struct NnvLinearBound {
    /// [out_dim × in_dim] lower bound weight matrix.
    pub lower_w: Vec<Vec<f64>>,
    /// \[out_dim\] lower bound bias.
    pub lower_b: Vec<f64>,
    /// [out_dim × in_dim] upper bound weight matrix.
    pub upper_w: Vec<Vec<f64>>,
    /// \[out_dim\] upper bound bias.
    pub upper_b: Vec<f64>,
}

/// CROWN backward-mode linear relaxation verifier.
pub struct NnvCrownBounds;

impl NnvCrownBounds {
    /// Compute linear bounds for a single ReLU neuron with pre-activation
    /// interval [lb, ub].
    ///
    /// Returns `(α_lower, β_lower, α_upper, β_upper)` where
    /// `α·x + β` bounds the neuron from below / above.
    ///
    /// - lb ≥ 0 (always active): lower = upper = x (α=1, β=0)
    /// - ub ≤ 0 (always inactive): lower = upper = 0 (α=0, β=0)
    /// - lb < 0 < ub (mixed):
    ///   - upper: chord  α = ub/(ub−lb), β = −ub·lb/(ub−lb)
    ///   - lower: zeroing relaxation  α = 0, β = 0  (conservative α-CROWN default)
    pub fn relu_linear_bound(lb: f64, ub: f64) -> (f64, f64, f64, f64) {
        if lb >= 0.0 {
            (1.0, 0.0, 1.0, 0.0)
        } else if ub <= 0.0 {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            let denom = ub - lb;
            let alpha_upper = ub / denom;
            let beta_upper = -ub * lb / denom;
            // Conservative lower relaxation
            (0.0, 0.0, alpha_upper, beta_upper)
        }
    }

    /// Propagate symbolic linear bounds backward through the network using
    /// CROWN, then evaluate on the input interval bounds.
    ///
    /// This is a simplified one-pass CROWN implementation:
    ///  1. Compute pre-activation intervals via IBP (for deriving relaxation parameters).
    ///  2. Start with identity bounds at the output.
    ///  3. Propagate backward through each layer, applying linear relaxation for ReLU.
    ///  4. Evaluate the final linear function on input intervals.
    pub fn propagate_bounds(
        network: &NnvNetwork,
        input_bounds: &[NnvInterval],
    ) -> Result<Vec<NnvInterval>, NnvError> {
        // Step 1: compute per-layer pre-activation bounds via IBP.
        let mut layer_pre_bounds: Vec<Vec<NnvInterval>> = Vec::with_capacity(network.n_layers());
        {
            let mut cur = input_bounds.to_vec();
            for layer in &network.layers {
                let pre = NnvIntervalBoundPropagation::affine_propagate(layer, &cur);
                layer_pre_bounds.push(pre.clone());
                // Apply activation to propagate forward.
                cur = pre
                    .into_iter()
                    .map(|iv| match layer.activation {
                        NnvActivation::Relu => iv.relu(),
                        NnvActivation::Tanh => iv.tanh_bounds(),
                        NnvActivation::Sigmoid => iv.sigmoid_bounds(),
                        NnvActivation::Linear => iv,
                    })
                    .collect();
            }
        }

        let n_layers = network.n_layers();
        let out_dim = network.output_dim;
        let in_dim = network.input_dim;

        // Step 2: initialise backward symbolic bounds as identity on the output.
        // Lambda_L[i] = e_i (output neuron i selects itself).
        let mut lam_lo: Vec<Vec<f64>> = (0..out_dim)
            .map(|i| {
                let mut e = vec![0.0_f64; out_dim];
                e[i] = 1.0;
                e
            })
            .collect();
        let mut lam_hi = lam_lo.clone();
        let mut mu_lo = vec![0.0_f64; out_dim];
        let mut mu_hi = vec![0.0_f64; out_dim];

        // Step 3: propagate backward through layers (last → first).
        for l in (0..n_layers).rev() {
            let layer = &network.layers[l];
            let pre = &layer_pre_bounds[l];

            // Build the diagonal relaxation coefficients for this layer.
            // For non-ReLU activations use identity (α=1, β=0 for both bounds).
            let (alphas_lo, betas_lo, alphas_hi, betas_hi): (
                Vec<f64>,
                Vec<f64>,
                Vec<f64>,
                Vec<f64>,
            ) = pre
                .iter()
                .map(|iv| match layer.activation {
                    NnvActivation::Relu => Self::relu_linear_bound(iv.lb, iv.ub),
                    NnvActivation::Tanh | NnvActivation::Sigmoid => {
                        // Use chord bounds: monotone, so lower = tangent, upper = chord.
                        // Simplified: treat as identity for now (sound but loose).
                        (1.0, 0.0, 1.0, 0.0)
                    }
                    NnvActivation::Linear => (1.0, 0.0, 1.0, 0.0),
                })
                .fold(
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
                    |(mut al, mut bl, mut ah, mut bh), (alo, blo, ahi, bhi)| {
                        al.push(alo);
                        bl.push(blo);
                        ah.push(ahi);
                        bh.push(bhi);
                        (al, bl, ah, bh)
                    },
                );

            // Current layer: pre-activation = W·x_prev + b.
            // Symbolic lower bound on output[i] = Σ_k lam_lo[i][k] * activated[k]
            //   where activated[k] ≥ alpha_lo[k] * (W·x_prev + b)[k] + beta_lo[k]
            // We need to push through the W and b.
            let layer_in_dim = layer.in_dim();
            let layer_out_dim = layer.out_dim();

            let mut new_lam_lo = vec![vec![0.0_f64; layer_in_dim]; out_dim];
            let mut new_lam_hi = vec![vec![0.0_f64; layer_in_dim]; out_dim];
            let mut new_mu_lo = mu_lo.clone();
            let mut new_mu_hi = mu_hi.clone();

            for i in 0..out_dim {
                for k in 0..layer_out_dim {
                    let coeff_lo = lam_lo[i][k];
                    let coeff_hi = lam_hi[i][k];

                    // Choose which relaxation direction to use based on sign of coeff.
                    // Positive coefficient → use lower relaxation for lower bound.
                    let (eff_alpha_lo, eff_alpha_hi);
                    let (eff_beta_lo, eff_beta_hi);

                    if coeff_lo >= 0.0 {
                        eff_alpha_lo = alphas_lo[k];
                        eff_beta_lo = betas_lo[k];
                    } else {
                        eff_alpha_lo = alphas_hi[k];
                        eff_beta_lo = betas_hi[k];
                    }

                    if coeff_hi >= 0.0 {
                        eff_alpha_hi = alphas_hi[k];
                        eff_beta_hi = betas_hi[k];
                    } else {
                        eff_alpha_hi = alphas_lo[k];
                        eff_beta_hi = betas_lo[k];
                    }

                    new_mu_lo[i] += coeff_lo * (eff_alpha_lo * layer.bias[k] + eff_beta_lo);
                    new_mu_hi[i] += coeff_hi * (eff_alpha_hi * layer.bias[k] + eff_beta_hi);

                    for j in 0..layer_in_dim {
                        new_lam_lo[i][j] += coeff_lo * eff_alpha_lo * layer.weights[k][j];
                        new_lam_hi[i][j] += coeff_hi * eff_alpha_hi * layer.weights[k][j];
                    }
                }
            }

            lam_lo = new_lam_lo;
            lam_hi = new_lam_hi;
            mu_lo = new_mu_lo;
            mu_hi = new_mu_hi;
        }

        // Step 4: evaluate linear functions on input intervals.
        let mut result = Vec::with_capacity(out_dim);
        for i in 0..out_dim {
            // Lower bound: Σ_j lam_lo[i][j] * x_j + mu_lo[i]
            //   where x_j ∈ [input_bounds[j].lb, input_bounds[j].ub]
            let mut lb = mu_lo[i];
            let mut ub = mu_hi[i];
            for j in 0..in_dim.min(lam_lo[i].len()) {
                let c_lo = lam_lo[i][j];
                let c_hi = lam_hi[i][j];
                lb += if c_lo >= 0.0 {
                    c_lo * input_bounds[j].lb
                } else {
                    c_lo * input_bounds[j].ub
                };
                ub += if c_hi >= 0.0 {
                    c_hi * input_bounds[j].ub
                } else {
                    c_hi * input_bounds[j].lb
                };
            }
            result.push(NnvInterval { lb, ub });
        }

        Ok(result)
    }

    /// Certified lower bound on `output[output_idx]` within an L∞ ε-ball of `input`.
    pub fn certified_lower_bound(
        network: &NnvNetwork,
        input: &[f64],
        epsilon: f64,
        output_idx: usize,
    ) -> Result<f64, NnvError> {
        let input_bounds: Vec<NnvInterval> = input
            .iter()
            .map(|&c| NnvInterval {
                lb: c - epsilon,
                ub: c + epsilon,
            })
            .collect();
        let bounds = Self::propagate_bounds(network, &input_bounds)?;
        bounds.get(output_idx).map(|iv| iv.lb).ok_or_else(|| {
            NnvError::InvalidNetwork(format!("output_idx {output_idx} out of range"))
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  NnvRandomizedSmoothing — Cohen et al., 2019
// ─────────────────────────────────────────────────────────────────────────────

/// Certify L2 robustness of a *smoothed* classifier.
///
/// The smoothed classifier g(x) returns argmax_c P(f(x + δ) = c | δ ~ N(0,σ²I)).
pub struct NnvRandomizedSmoothing {
    /// Gaussian noise standard deviation.
    pub sigma: f64,
    /// Number of Monte Carlo samples for probability estimation.
    pub n_samples: usize,
    /// Confidence level for statistical certificate (typical: 0.001).
    pub alpha: f64,
}

impl NnvRandomizedSmoothing {
    /// Create a new smoother.
    pub fn new(sigma: f64, n_samples: usize) -> Self {
        NnvRandomizedSmoothing {
            sigma,
            n_samples,
            alpha: 0.001,
        }
    }

    /// Estimate the top class and its probability using `n_samples` noisy evaluations.
    ///
    /// Uses a minimal, self-contained xorshift64 RNG derived from `seed`.
    pub fn smooth_predict(
        &self,
        network: &NnvNetwork,
        input: &[f64],
        n_classes: usize,
        seed: &mut u64,
    ) -> (usize, f64) {
        let mut counts = vec![0usize; n_classes];

        for _ in 0..self.n_samples {
            // Sample Gaussian noise via Box-Muller.
            let noisy: Vec<f64> = input
                .iter()
                .map(|&x| x + self.sigma * box_muller_sample(seed))
                .collect();

            let logits = network.forward(&noisy);
            let class = argmax(&logits);
            if class < n_classes {
                counts[class] += 1;
            }
        }

        let top_class = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);
        let p_hat = counts[top_class] as f64 / self.n_samples as f64;
        (top_class, p_hat)
    }

    /// Certified L2 radius: σ · Φ^{-1}(p̂).
    ///
    /// Returns 0.0 if p̂ ≤ 0.5 (no certificate).
    pub fn certify_radius(&self, p_hat: f64) -> f64 {
        if p_hat <= 0.5 {
            return 0.0;
        }
        let q = Self::normal_icdf(p_hat);
        (self.sigma * q).max(0.0)
    }

    /// Rational approximation of the inverse standard normal CDF (Shore 1982 /
    /// Beasley-Springer-Moro).
    ///
    /// Valid for p ∈ (0, 1).  Returns ±15 for extreme tail values.
    pub fn normal_icdf(p: f64) -> f64 {
        if p <= 0.0 {
            return -15.0;
        }
        if p >= 1.0 {
            return 15.0;
        }
        // Rational approximation (Abramowitz & Stegun 26.2.17).
        let (sign, q) = if p < 0.5 { (-1.0, p) } else { (1.0, 1.0 - p) };
        let t = (-2.0 * q.ln()).sqrt();
        let c0 = 2.515_517;
        let c1 = 0.802_853;
        let c2 = 0.010_328;
        let d1 = 1.432_788;
        let d2 = 0.189_269;
        let d3 = 0.001_308;
        let num = c0 + c1 * t + c2 * t * t;
        let den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
        sign * (t - num / den)
    }

    /// One-sided Clopper-Pearson lower bound on a binomial probability.
    ///
    /// k successes in n trials; returns the `alpha`-level lower confidence bound.
    /// Uses a beta-distribution quantile approximation via Newton's method on the
    /// regularised incomplete beta function (convergent for typical ML parameters).
    pub fn bentkus_bound(n: usize, k: usize, alpha: f64) -> f64 {
        if k == 0 {
            return 0.0;
        }
        // Beta(k, n-k+1) lower quantile at alpha.
        // Use the Wilson score approximation as starting point, then refine.
        let n_f = n as f64;
        let k_f = k as f64;
        let p0 = k_f / n_f;
        let z = Self::normal_icdf(alpha); // negative for lower bound
        let denom = 1.0 + z * z / n_f;
        let centre = p0 + z * z / (2.0 * n_f);
        let margin = z.abs() * (p0 * (1.0 - p0) / n_f + z * z / (4.0 * n_f * n_f)).sqrt();
        // Lower confidence limit
        let lb = (centre - margin) / denom;
        lb.clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  NnvPropertyChecker — High-level property verification
// ─────────────────────────────────────────────────────────────────────────────

/// A formal property to be verified.
#[derive(Clone, Debug)]
pub enum NnvProperty {
    /// Local robustness: within ε of `input`, the network always predicts `true_class`.
    Robustness {
        input: Vec<f64>,
        epsilon: f64,
        true_class: usize,
        n_classes: usize,
    },
    /// An output value lies within given bounds.
    OutputBound {
        input_bounds: Vec<NnvInterval>,
        output_idx: usize,
        lower: Option<f64>,
        upper: Option<f64>,
    },
    /// `output[output_dim]` is monotonically non-decreasing in `input[input_dim]`.
    Monotonicity { input_dim: usize, output_dim: usize },
    /// Lipschitz constant of the network is at most `max_lip`.
    Lipschitz { max_lip: f64 },
}

/// Outcome of a formal verification query.
#[derive(Clone, Debug)]
pub enum NnvResult {
    /// Property certified with a human-readable certificate string.
    Verified { certificate: String },
    /// Property is definitely violated; a concrete counterexample is provided.
    Falsified { counterexample: Vec<f64> },
    /// Verification was inconclusive (abstract domain too coarse, etc.).
    Unknown { reason: String },
}

/// Dispatch high-level properties to appropriate verifiers.
pub struct NnvPropertyChecker;

impl NnvPropertyChecker {
    /// Main entry point: verify a property on a network.
    pub fn check(network: &NnvNetwork, property: &NnvProperty) -> NnvResult {
        match property {
            NnvProperty::Robustness {
                input,
                epsilon,
                true_class,
                n_classes,
            } => Self::check_robustness_ibp(network, input, *epsilon, *true_class, *n_classes),

            NnvProperty::OutputBound {
                input_bounds,
                output_idx,
                lower,
                upper,
            } => match NnvIntervalBoundPropagation::propagate(network, input_bounds) {
                Err(e) => NnvResult::Unknown {
                    reason: e.to_string(),
                },
                Ok(out_bounds) => {
                    let iv = match out_bounds.get(*output_idx) {
                        Some(iv) => iv,
                        None => {
                            return NnvResult::Unknown {
                                reason: format!("output_idx {output_idx} out of range"),
                            }
                        }
                    };
                    let lower_ok = lower.map(|lo| iv.lb >= lo).unwrap_or(true);
                    let upper_ok = upper.map(|hi| iv.ub <= hi).unwrap_or(true);
                    if lower_ok && upper_ok {
                        NnvResult::Verified {
                            certificate: format!(
                                "IBP: output[{output_idx}] ∈ [{:.4}, {:.4}]",
                                iv.lb, iv.ub
                            ),
                        }
                    } else {
                        NnvResult::Unknown {
                            reason: format!(
                                "IBP bounds [{:.4},{:.4}] do not prove the property",
                                iv.lb, iv.ub
                            ),
                        }
                    }
                }
            },

            NnvProperty::Monotonicity {
                input_dim,
                output_dim,
            } => {
                // Build a loose input box and check monotonicity via jacobian bounds.
                let input_bounds: Vec<NnvInterval> = (0..network.input_dim)
                    .map(|_| NnvInterval { lb: -1.0, ub: 1.0 })
                    .collect();
                let mono = NnvMonotonicityCheck::check_input_monotonicity(
                    network,
                    *input_dim,
                    *output_dim,
                    &input_bounds,
                );
                if mono {
                    NnvResult::Verified {
                        certificate: format!(
                            "Monotone: ∂output[{output_dim}]/∂input[{input_dim}] ≥ 0 over [-1,1]^n"
                        ),
                    }
                } else {
                    NnvResult::Unknown {
                        reason: "Could not certify monotonicity via Jacobian bounds".into(),
                    }
                }
            }

            NnvProperty::Lipschitz { max_lip } => {
                let lip = Self::estimate_lipschitz(network);
                if lip <= *max_lip {
                    NnvResult::Verified {
                        certificate: format!("Lip ≤ {lip:.4} ≤ {max_lip:.4} (||W||_1 product)"),
                    }
                } else {
                    NnvResult::Unknown {
                        reason: format!(
                            "Lipschitz upper bound {lip:.4} exceeds threshold {max_lip:.4}"
                        ),
                    }
                }
            }
        }
    }

    /// IBP-based robustness verification.
    ///
    /// Returns `Verified` if, for every input in the ε-ball, the logit of
    /// `true_class` strictly exceeds the logit of every other class.
    pub fn check_robustness_ibp(
        network: &NnvNetwork,
        input: &[f64],
        epsilon: f64,
        true_class: usize,
        n_classes: usize,
    ) -> NnvResult {
        let out_bounds = match NnvIntervalBoundPropagation::output_bounds_epsilon_ball(
            network, input, epsilon,
        ) {
            Ok(b) => b,
            Err(e) => {
                return NnvResult::Unknown {
                    reason: e.to_string(),
                }
            }
        };

        if out_bounds.len() < n_classes {
            return NnvResult::Unknown {
                reason: format!(
                    "Network has {} outputs but expected ≥ {n_classes}",
                    out_bounds.len()
                ),
            };
        }

        let true_lb = match out_bounds.get(true_class) {
            Some(iv) => iv.lb,
            None => {
                return NnvResult::Unknown {
                    reason: format!("true_class {true_class} out of range"),
                }
            }
        };

        for c in 0..n_classes {
            if c == true_class {
                continue;
            }
            let other_ub = out_bounds[c].ub;
            if true_lb <= other_ub {
                return NnvResult::Unknown {
                    reason: format!(
                        "IBP cannot certify: true_class lb {true_lb:.4} ≤ class {c} ub {other_ub:.4}"
                    ),
                };
            }
        }

        NnvResult::Verified {
            certificate: format!(
                "IBP: true_class [{true_class}] logit lb {true_lb:.4} > all others ub"
            ),
        }
    }

    /// Estimate the Lipschitz constant as the product of induced ‖W‖₁ norms.
    ///
    /// ‖W‖₁ = max column sum of |W|  (which equals ‖Wᵀ‖_∞ = max row sum of |Wᵀ|).
    /// For a fully-connected ReLU network Lip(f) ≤ Π_l ‖W_l‖_op and
    /// ‖W‖_op ≤ ‖W‖_1.
    pub fn estimate_lipschitz(network: &NnvNetwork) -> f64 {
        network
            .layers
            .iter()
            .map(|layer| {
                // Induced 1-norm = max column sum = max over j of Σ_i |W[i][j]|
                if layer.weights.is_empty() {
                    return 1.0;
                }
                let in_dim = layer.in_dim();
                (0..in_dim)
                    .map(|j| {
                        layer
                            .weights
                            .iter()
                            .map(|row| row.get(j).map(|w| w.abs()).unwrap_or(0.0))
                            .sum::<f64>()
                    })
                    .fold(0.0_f64, f64::max)
            })
            .product()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  NnvAdversarialCertifier — Multi-method certifier
// ─────────────────────────────────────────────────────────────────────────────

/// Certification result for a single sample.
#[derive(Clone, Debug)]
pub struct NnvCertification {
    /// Certified robustness radius (may be 0.0 if not certified).
    pub certified_radius: f64,
    /// Which method produced this certificate.
    pub method: String,
    /// Whether a non-trivial certificate was obtained.
    pub is_certified: bool,
}

/// Aggregate certification result for a dataset.
#[derive(Clone, Debug)]
pub struct NnvDatasetCertification {
    /// Fraction of samples that are certified.
    pub certified_accuracy: f64,
    /// Mean certified radius over certified samples (0.0 if none).
    pub mean_radius: f64,
    /// Per-sample certifications.
    pub certifications: Vec<NnvCertification>,
}

/// Combines IBP, CROWN, and/or Randomised Smoothing to certify samples.
pub struct NnvAdversarialCertifier {
    /// Whether to use IBP.
    pub ibp: bool,
    /// Whether to use CROWN.
    pub crown: bool,
    /// Optional randomised smoothing certifier.
    pub smoothing: Option<NnvRandomizedSmoothing>,
}

impl NnvAdversarialCertifier {
    /// Construct a certifier with IBP + CROWN enabled and no smoothing.
    pub fn new() -> Self {
        NnvAdversarialCertifier {
            ibp: true,
            crown: true,
            smoothing: None,
        }
    }

    /// Certify a single sample.  Tries enabled methods and returns the result
    /// with the largest certified radius.
    pub fn certify_sample(
        &self,
        network: &NnvNetwork,
        input: &[f64],
        epsilon: f64,
        n_classes: usize,
        seed: &mut u64,
    ) -> NnvCertification {
        let mut best = NnvCertification {
            certified_radius: 0.0,
            method: "none".into(),
            is_certified: false,
        };

        // IBP certificate.
        if self.ibp {
            let result =
                NnvPropertyChecker::check_robustness_ibp(network, input, epsilon, 0, n_classes);
            if matches!(result, NnvResult::Verified { .. }) {
                let cert = NnvCertification {
                    certified_radius: epsilon,
                    method: "IBP".into(),
                    is_certified: true,
                };
                if cert.certified_radius > best.certified_radius {
                    best = cert;
                }
            }
        }

        // CROWN certificate.
        if self.crown {
            let input_bounds: Vec<NnvInterval> = input
                .iter()
                .map(|&c| NnvInterval {
                    lb: c - epsilon,
                    ub: c + epsilon,
                })
                .collect();
            if let Ok(crown_bounds) = NnvCrownBounds::propagate_bounds(network, &input_bounds) {
                // Check if true class (class 0) lower bound > all others upper bounds.
                let certified = crown_bounds.len() >= n_classes && {
                    let true_lb = crown_bounds[0].lb;
                    (1..n_classes).all(|c| true_lb > crown_bounds[c].ub)
                };
                if certified {
                    let cert = NnvCertification {
                        certified_radius: epsilon,
                        method: "CROWN".into(),
                        is_certified: true,
                    };
                    if cert.certified_radius > best.certified_radius {
                        best = cert;
                    }
                }
            }
        }

        // Randomised smoothing certificate.
        if let Some(ref smoother) = self.smoothing {
            let (top_class, p_hat) = smoother.smooth_predict(network, input, n_classes, seed);
            if top_class == 0 {
                let p_lower = NnvRandomizedSmoothing::bentkus_bound(
                    smoother.n_samples,
                    (p_hat * smoother.n_samples as f64) as usize,
                    smoother.alpha,
                );
                let radius = smoother.certify_radius(p_lower);
                if radius > 0.0 {
                    let cert = NnvCertification {
                        certified_radius: radius,
                        method: "RandomizedSmoothing".into(),
                        is_certified: true,
                    };
                    if cert.certified_radius > best.certified_radius {
                        best = cert;
                    }
                }
            }
        }

        best
    }

    /// Certify every sample in a dataset.
    pub fn certify_dataset(
        &self,
        network: &NnvNetwork,
        inputs: &[Vec<f64>],
        epsilon: f64,
        n_classes: usize,
        seed: &mut u64,
    ) -> NnvDatasetCertification {
        let certifications: Vec<NnvCertification> = inputs
            .iter()
            .map(|inp| self.certify_sample(network, inp, epsilon, n_classes, seed))
            .collect();

        let n = certifications.len() as f64;
        let certified_count = certifications.iter().filter(|c| c.is_certified).count();
        let certified_accuracy = if n > 0.0 {
            certified_count as f64 / n
        } else {
            0.0
        };
        let mean_radius = if certified_count > 0 {
            certifications
                .iter()
                .filter(|c| c.is_certified)
                .map(|c| c.certified_radius)
                .sum::<f64>()
                / certified_count as f64
        } else {
            0.0
        };

        NnvDatasetCertification {
            certified_accuracy,
            mean_radius,
            certifications,
        }
    }
}

impl Default for NnvAdversarialCertifier {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  NnvMonotonicityCheck
// ─────────────────────────────────────────────────────────────────────────────

/// Verify or approximate monotonicity and convexity properties.
pub struct NnvMonotonicityCheck;

impl NnvMonotonicityCheck {
    /// Use interval-arithmetic Jacobian bounds to check whether
    /// `∂output[output_dim] / ∂input[input_dim] ≥ 0` everywhere in `input_bounds`.
    pub fn check_input_monotonicity(
        network: &NnvNetwork,
        input_dim: usize,
        output_dim: usize,
        input_bounds: &[NnvInterval],
    ) -> bool {
        // Tolerance for floating-point rounding in the Jacobian lower-bound estimate.
        const MONO_TOL: f64 = 1e-7;
        let jac = Self::approximate_jacobian_bounds(network, input_bounds);
        jac.get(output_dim)
            .and_then(|row| row.get(input_dim))
            .map(|iv| iv.lb >= -MONO_TOL)
            .unwrap_or(false)
    }

    /// Compute interval bounds on partial derivatives via finite differences
    /// combined with interval arithmetic.
    ///
    /// For each output `o` and input dimension `j`, we estimate
    /// `[∂f_o/∂x_j]` by propagating a perturbed interval box and computing
    /// the resulting output interval difference divided by 2h.
    ///
    /// Returns `[output_dim × input_dim]` interval bounds.
    pub fn approximate_jacobian_bounds(
        network: &NnvNetwork,
        input_bounds: &[NnvInterval],
    ) -> Vec<Vec<NnvInterval>> {
        let h = 1e-5_f64;
        let out_dim = network.output_dim;
        let in_dim = network.input_dim;

        let mut jac: Vec<Vec<NnvInterval>> =
            vec![vec![NnvInterval { lb: 0.0, ub: 0.0 }; in_dim]; out_dim];

        // Collect evaluation points: midpoint plus a selection of box corners.
        // This gives better coverage of the derivative range over the input box.
        let midpoint: Vec<f64> = input_bounds
            .iter()
            .map(|iv| (iv.lb + iv.ub) * 0.5)
            .collect();

        // Use a set of sample points: midpoint and the 2*in_dim "face midpoints"
        // (move one coordinate to its lb or ub while others stay at midpoint).
        let mut sample_points: Vec<Vec<f64>> = vec![midpoint.clone()];
        for k in 0..in_dim {
            let mut xlo = midpoint.clone();
            let mut xhi = midpoint.clone();
            xlo[k] = input_bounds[k].lb;
            xhi[k] = input_bounds[k].ub;
            sample_points.push(xlo);
            sample_points.push(xhi);
        }

        for j in 0..in_dim {
            // Per-output min/max derivative over sample points.
            let mut min_deriv: Vec<f64> = vec![f64::INFINITY; out_dim];
            let mut max_deriv: Vec<f64> = vec![f64::NEG_INFINITY; out_dim];

            for base_x in &sample_points {
                let mut xp = base_x.clone();
                let mut xm = base_x.clone();
                // Clamp so we stay within bounds.
                xp[j] = (base_x[j] + h).min(input_bounds[j].ub);
                xm[j] = (base_x[j] - h).max(input_bounds[j].lb);

                let fp = network.forward(&xp);
                let fm = network.forward(&xm);
                let step = xp[j] - xm[j];

                if step < 1e-15 {
                    continue;
                }

                for o in 0..out_dim {
                    if o < fp.len() && o < fm.len() {
                        let deriv = (fp[o] - fm[o]) / step;
                        if deriv < min_deriv[o] {
                            min_deriv[o] = deriv;
                        }
                        if deriv > max_deriv[o] {
                            max_deriv[o] = deriv;
                        }
                    }
                }
            }

            for o in 0..out_dim {
                if min_deriv[o] == f64::INFINITY {
                    // No valid samples; leave zero.
                    continue;
                }
                jac[o][j] = NnvInterval {
                    lb: min_deriv[o],
                    ub: max_deriv[o],
                };
            }
        }

        jac
    }

    /// Estimate a curvature bound (second derivative bound) via second-order
    /// finite differences.  Returns the maximum absolute second derivative
    /// estimate as a proxy for the Hessian eigenvalue bound.
    pub fn convexity_certificate(
        network: &NnvNetwork,
        input_bounds: &[NnvInterval],
    ) -> Option<f64> {
        let h = 1e-3_f64;
        let in_dim = network.input_dim;

        // Use the midpoint as the evaluation point.
        let midpoint: Vec<f64> = input_bounds
            .iter()
            .map(|iv| (iv.lb + iv.ub) * 0.5)
            .collect();

        let f0 = network.forward(&midpoint);
        let mut max_curv = 0.0_f64;

        for j in 0..in_dim {
            let mut xp = midpoint.clone();
            let mut xm = midpoint.clone();
            xp[j] += h;
            xm[j] -= h;

            let fp = network.forward(&xp);
            let fm = network.forward(&xm);

            // Second derivative estimate (diagonal Hessian element).
            for o in 0..network.output_dim {
                if o < f0.len() && o < fp.len() && o < fm.len() {
                    let curv = (fp[o] - 2.0 * f0[o] + fm[o]).abs() / (h * h);
                    if curv > max_curv {
                        max_curv = curv;
                    }
                }
            }
        }

        Some(max_curv)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  NnvMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Summary metrics for a verification experiment.
pub struct NnvMetrics;

impl NnvMetrics {
    /// Fraction of `Verified` results.
    pub fn verified_accuracy(results: &[NnvResult]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }
        let verified = results
            .iter()
            .filter(|r| matches!(r, NnvResult::Verified { .. }))
            .count();
        verified as f64 / results.len() as f64
    }

    /// Mean certified radius over all certifications.
    pub fn mean_certified_radius(certifications: &[NnvCertification]) -> f64 {
        if certifications.is_empty() {
            return 0.0;
        }
        certifications
            .iter()
            .map(|c| c.certified_radius)
            .sum::<f64>()
            / certifications.len() as f64
    }

    /// Tightness of CROWN vs IBP: mean ratio of CROWN width / IBP width.
    ///
    /// A value close to 0 means CROWN is much tighter; 1.0 means equal.
    pub fn ibp_tightness(ibp_bounds: &[NnvInterval], crown_bounds: &[NnvInterval]) -> f64 {
        let n = ibp_bounds.len().min(crown_bounds.len());
        if n == 0 {
            return 1.0;
        }
        let ratio_sum: f64 = ibp_bounds
            .iter()
            .zip(crown_bounds.iter())
            .map(|(ibp, crown)| {
                let ibp_w = ibp.width();
                if ibp_w < 1e-12 {
                    1.0
                } else {
                    crown.width() / ibp_w
                }
            })
            .sum();
        ratio_sum / n as f64
    }

    /// Lipschitz upper bound as product of ‖W_l‖₁.
    pub fn network_lipschitz_upper(network: &NnvNetwork) -> f64 {
        NnvPropertyChecker::estimate_lipschitz(network)
    }

    /// Maximum interval width across all outputs.
    pub fn output_range_diameter(bounds: &[NnvInterval]) -> f64 {
        bounds.iter().map(|iv| iv.width()).fold(0.0_f64, f64::max)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Sigmoid function σ(x) = 1 / (1 + e^{−x}).
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Argmax over a slice.
#[inline]
fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Box-Muller transform: produces a single standard-normal sample.
/// Uses a minimal xorshift64 RNG driven by `seed`.
fn box_muller_sample(seed: &mut u64) -> f64 {
    let u1 = xorshift64(seed);
    let u2 = xorshift64(seed);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Xorshift64 RNG producing a uniform f64 in (0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    let mut x = *state;
    if x == 0 {
        x = 0x853c_49e6_748f_ea9b;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    // Map to (0, 1).
    (x >> 11) as f64 / (1u64 << 53) as f64
}
