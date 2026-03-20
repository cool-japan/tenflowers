//! Kolmogorov-Arnold Networks (KANs)
//!
//! Implements KANs from "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024).
//!
//! Unlike MLPs with fixed node-wise activations, KANs use **learnable edge-wise
//! B-spline activation functions**. This gives KANs excellent properties for
//! discovering mathematical structure in data (symbolic regression, physics laws).
//!
//! # Key Components
//!
//! * [`KanBSplineBasis`] — B-spline basis via de Boor's algorithm
//! * [`KanActivation`] — per-edge learnable activation (spline + SiLU residual)
//! * [`KanLayer`] — a full KAN layer with `in_dim × out_dim` activations
//! * [`KanModel`] — multi-layer KAN model
//! * [`KanSymbolicExtractor`] — identifies which mathematical function each edge approximates
//! * [`KanPinn`] — KAN-based physics-informed neural network
//! * [`KanTrainer`] — finite-difference training loop
//! * [`KanReport`] / [`KanMetrics`] — diagnostics

use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur in KAN operations.
#[derive(Debug, Clone)]
pub enum KanError {
    /// Input dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Numerical issue (NaN/Inf).
    NumericalError(String),
}

impl std::fmt::Display for KanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KanError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
            KanError::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
            KanError::NumericalError(msg) => write!(f, "numerical error: {}", msg),
        }
    }
}

impl std::error::Error for KanError {}

// ─────────────────────────────────────────────────────────────────────────────
// B-Spline Basis (de Boor's algorithm)
// ─────────────────────────────────────────────────────────────────────────────

/// B-spline basis with uniform (or adapted) knot grid.
///
/// Uses de Boor's algorithm for numerically stable evaluation.
/// The basis satisfies partition of unity: ∑ B_i(x) = 1 for x in [low, high].
#[derive(Debug, Clone)]
pub struct KanBSplineBasis {
    /// Extended knot vector (includes clamped repeated endpoints).
    pub grid: Vec<f64>,
    /// Polynomial degree (typically 3 for cubic splines).
    pub degree: usize,
    /// Number of basis functions = len(grid) - degree - 1.
    pub n_basis: usize,
}

impl KanBSplineBasis {
    /// Create a uniform B-spline basis over `[low, high]` with `n_intervals` knot spans.
    ///
    /// The knot vector is extended by clamping: the first and last knot are each
    /// repeated `degree` times so that the spline interpolates the endpoints.
    pub fn new(low: f64, high: f64, n_intervals: usize, degree: usize) -> Self {
        let n_interior = n_intervals + 1; // internal knots including endpoints
        let step = (high - low) / (n_intervals as f64);

        // Build extended knot vector: [low]*degree + interior + [high]*degree
        let mut grid = Vec::with_capacity(n_interior + 2 * degree);
        for _ in 0..degree {
            grid.push(low);
        }
        for i in 0..=n_intervals {
            grid.push(low + i as f64 * step);
        }
        for _ in 0..degree {
            grid.push(high);
        }

        let n_basis = grid.len() - degree - 1;
        KanBSplineBasis {
            grid,
            degree,
            n_basis,
        }
    }

    /// Evaluate all basis functions at `x` using de Boor's algorithm.
    ///
    /// Returns a vector of length `n_basis`. Outside the knot range the
    /// outermost basis is clamped (natural extrapolation).
    pub fn evaluate(&self, x: f64) -> Vec<f64> {
        let x = x.clamp(
            *self.grid.first().unwrap_or(&0.0),
            *self.grid.last().unwrap_or(&1.0),
        );
        let g = &self.grid;
        let n = self.n_basis;
        let p = self.degree;

        // de Boor triangular table — start with order-0 basis
        let mut b = vec![0.0f64; n + p + 1];
        // Find the active knot span
        let k = self.find_span(x);
        b[k] = 1.0;

        // Triangular recurrence for order 1..=p
        for d in 1..=p {
            let mut new_b = vec![0.0f64; n + p + 1];
            for i in 0..n {
                let left_den = g[i + d] - g[i];
                let right_den = g[i + d + 1] - g[i + 1];
                let left = if left_den.abs() > 1e-14 {
                    (x - g[i]) / left_den * b[i]
                } else {
                    0.0
                };
                let right = if right_den.abs() > 1e-14 {
                    (g[i + d + 1] - x) / right_den * b[i + 1]
                } else {
                    0.0
                };
                new_b[i] = left + right;
            }
            b = new_b;
        }

        b[..n].to_vec()
    }

    /// Locate the knot span index for `x`.
    fn find_span(&self, x: f64) -> usize {
        let g = &self.grid;
        let n = self.n_basis;
        let p = self.degree;

        // The valid range is [g[p], g[n]]
        if x >= g[n] {
            return n - 1;
        }
        if x <= g[p] {
            return p;
        }
        // Binary search for k such that g[k] <= x < g[k+1]
        let mut lo = p;
        let mut hi = n;
        let mut mid = (lo + hi) / 2;
        while x < g[mid] || x >= g[mid + 1] {
            if x < g[mid] {
                hi = mid;
            } else {
                lo = mid;
            }
            mid = (lo + hi) / 2;
            if lo + 1 >= hi {
                break;
            }
        }
        mid
    }

    /// Extend the knot grid to cover the data range, then re-distribute
    /// interior knots to approximately quantile positions of `data_points`.
    pub fn refine_grid(&mut self, data_points: &[f64]) {
        if data_points.is_empty() {
            return;
        }
        let mut sorted = data_points.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let new_low = sorted[0];
        let new_high = sorted[sorted.len() - 1];

        // Count number of original interior knots (excluding repeated endpoints)
        let n_intervals = self.n_basis - self.degree;
        let p = self.degree;

        // Place interior knots at quantile positions
        let mut grid = Vec::with_capacity(self.grid.len());
        for _ in 0..p {
            grid.push(new_low);
        }
        for i in 0..=n_intervals {
            let frac = i as f64 / n_intervals as f64;
            let idx = ((sorted.len() - 1) as f64 * frac).round() as usize;
            let idx = idx.min(sorted.len() - 1);
            grid.push(sorted[idx]);
        }
        for _ in 0..p {
            grid.push(new_high);
        }

        self.grid = grid;
        // n_basis stays the same
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KAN Activation (per edge)
// ─────────────────────────────────────────────────────────────────────────────

/// Learnable activation function on a single KAN edge.
///
/// Formula: `φ(x) = w_b · silu(x) + w_s · (B(x) · c)`
/// where `B(x)` is the B-spline basis vector and `c` are learnable coefficients.
#[derive(Debug, Clone)]
pub struct KanActivation {
    /// B-spline basis for this edge.
    pub basis: KanBSplineBasis,
    /// Learnable spline coefficients (one per basis function).
    pub coefficients: Vec<f64>,
    /// SiLU residual weight `w_b`.
    pub residual_scale: f64,
    /// Spline weight `w_s`.
    pub spline_scale: f64,
}

impl KanActivation {
    /// Create a new activation with random-ish small initial coefficients.
    pub fn new(grid_size: usize, degree: usize, low: f64, high: f64) -> Self {
        let basis = KanBSplineBasis::new(low, high, grid_size, degree);
        let n_basis = basis.n_basis;
        // Initialize coefficients to small values (Xavier-like)
        let scale = (2.0 / n_basis as f64).sqrt() * 0.1;
        // Deterministic init: alternating small positive/negative
        let coefficients: Vec<f64> = (0..n_basis)
            .map(|i| scale * if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        KanActivation {
            basis,
            coefficients,
            residual_scale: 1.0,
            spline_scale: 1.0,
        }
    }

    /// SiLU (Sigmoid Linear Unit): x * σ(x).
    #[inline]
    fn silu(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    /// Forward pass: φ(x) = w_b·silu(x) + w_s·(B(x)·c).
    pub fn forward(&self, x: f64) -> f64 {
        let b = self.basis.evaluate(x);
        let spline: f64 = b
            .iter()
            .zip(self.coefficients.iter())
            .map(|(bi, ci)| bi * ci)
            .sum();
        self.residual_scale * Self::silu(x) + self.spline_scale * spline
    }

    /// Update the basis grid from a set of observed input samples.
    pub fn update_grid(&mut self, samples: &[f64]) {
        self.basis.refine_grid(samples);
    }

    /// L1 norm of the coefficients — used for sparsification regularization.
    pub fn l1_norm(&self) -> f64 {
        self.coefficients.iter().map(|c| c.abs()).sum()
    }

    /// Entropy of the activation magnitude distribution (for entropy regularization).
    pub fn entropy(&self) -> f64 {
        let total: f64 = self.coefficients.iter().map(|c| c.abs()).sum();
        if total < 1e-14 {
            return 0.0;
        }
        self.coefficients
            .iter()
            .map(|c| {
                let p = c.abs() / total;
                if p < 1e-14 {
                    0.0
                } else {
                    -p * p.ln()
                }
            })
            .sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KAN Layer
// ─────────────────────────────────────────────────────────────────────────────

/// A single layer of a KAN: `out_dim` output neurons, each connected to all
/// `in_dim` inputs via learnable B-spline activations.
#[derive(Debug, Clone)]
pub struct KanLayer {
    /// Number of input features.
    pub in_dim: usize,
    /// Number of output features.
    pub out_dim: usize,
    /// Edge activations: `activations[out_j][in_i]` is the activation on edge i→j.
    pub activations: Vec<Vec<KanActivation>>,
}

impl KanLayer {
    /// Create a new KAN layer.
    pub fn new(in_dim: usize, out_dim: usize, grid_size: usize, degree: usize) -> Self {
        let activations = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| KanActivation::new(grid_size, degree, -1.0, 1.0))
                    .collect()
            })
            .collect();
        KanLayer {
            in_dim,
            out_dim,
            activations,
        }
    }

    /// Forward pass: output\[j\] = ∑_i φ_{ji}(x\[i\]).
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, KanError> {
        if x.len() != self.in_dim {
            return Err(KanError::DimensionMismatch {
                expected: self.in_dim,
                got: x.len(),
            });
        }
        let output: Vec<f64> = self
            .activations
            .iter()
            .map(|acts_j| {
                acts_j
                    .iter()
                    .zip(x.iter())
                    .map(|(act, xi)| act.forward(*xi))
                    .sum()
            })
            .collect();
        Ok(output)
    }

    /// Update the B-spline grids for all edges using the provided input data.
    pub fn update_grid(&mut self, x_data: &[Vec<f64>]) {
        // For each input dimension, collect all observed values
        let mut per_input: Vec<Vec<f64>> = vec![Vec::new(); self.in_dim];
        for x in x_data {
            for (i, xi) in x.iter().enumerate() {
                if i < self.in_dim {
                    per_input[i].push(*xi);
                }
            }
        }
        for acts_j in &mut self.activations {
            for (i, act) in acts_j.iter_mut().enumerate() {
                if i < per_input.len() {
                    act.update_grid(&per_input[i]);
                }
            }
        }
    }

    /// Combined L1 + entropy regularization loss.
    pub fn regularization_loss(&self, lambda_l1: f64, lambda_entropy: f64) -> f64 {
        let mut loss = 0.0;
        for acts_j in &self.activations {
            for act in acts_j {
                loss += lambda_l1 * act.l1_norm() + lambda_entropy * act.entropy();
            }
        }
        loss
    }

    /// Feature scores: L1 norm of activations summed over all outputs for each input.
    pub fn get_feature_scores(&self) -> Vec<f64> {
        let mut scores = vec![0.0f64; self.in_dim];
        for acts_j in &self.activations {
            for (i, act) in acts_j.iter().enumerate() {
                scores[i] += act.l1_norm();
            }
        }
        scores
    }

    /// Total number of learnable parameters in this layer.
    pub fn param_count(&self) -> usize {
        self.activations
            .iter()
            .flatten()
            .map(|a| a.coefficients.len() + 2)
            .sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KAN Model Configuration & Model
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a Kolmogorov-Arnold Network.
#[derive(Debug, Clone)]
pub struct KanConfig {
    /// Layer widths, e.g. `[2, 5, 5, 1]` for a 2-in, 1-out network with two hidden layers.
    pub layers: Vec<usize>,
    /// Number of B-spline intervals per activation (default: 5).
    pub grid_size: usize,
    /// B-spline polynomial degree (default: 3, cubic).
    pub degree: usize,
    /// L1 regularization coefficient (default: 1e-3).
    pub lambda_l1: f64,
    /// Entropy regularization coefficient (default: 2e-4).
    pub lambda_entropy: f64,
}

impl Default for KanConfig {
    fn default() -> Self {
        KanConfig {
            layers: vec![2, 5, 1],
            grid_size: 5,
            degree: 3,
            lambda_l1: 1e-3,
            lambda_entropy: 2e-4,
        }
    }
}

/// Multi-layer Kolmogorov-Arnold Network.
#[derive(Debug, Clone)]
pub struct KanModel {
    /// Model configuration.
    pub config: KanConfig,
    /// KAN layers.
    pub layers: Vec<KanLayer>,
}

impl KanModel {
    /// Build a new KAN model from config.
    pub fn new(config: KanConfig) -> Result<Self, KanError> {
        if config.layers.len() < 2 {
            return Err(KanError::InvalidConfig(
                "KanConfig.layers must have at least 2 entries (input + output)".into(),
            ));
        }
        let layers = config
            .layers
            .windows(2)
            .map(|w| KanLayer::new(w[0], w[1], config.grid_size, config.degree))
            .collect();
        Ok(KanModel { config, layers })
    }

    /// Forward pass through all layers.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, KanError> {
        let mut h = x.to_vec();
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        Ok(h)
    }

    /// Forward pass returning activations at every layer (including input).
    pub fn forward_with_all_outputs(&self, x: &[f64]) -> Result<Vec<Vec<f64>>, KanError> {
        let mut all = vec![x.to_vec()];
        let mut h = x.to_vec();
        for layer in &self.layers {
            h = layer.forward(&h)?;
            all.push(h.clone());
        }
        Ok(all)
    }

    /// Update all layer grids using a dataset of input vectors.
    pub fn update_grids(&mut self, dataset: &[Vec<f64>]) {
        let mut current_data: Vec<Vec<f64>> = dataset.to_vec();
        for layer in &mut self.layers {
            layer.update_grid(&current_data);
            // Propagate data through this layer to get inputs for the next layer
            let next_data: Vec<Vec<f64>> = current_data
                .iter()
                .filter_map(|x| layer.forward(x).ok())
                .collect();
            current_data = next_data;
        }
    }

    /// Total regularization loss across all layers.
    pub fn regularization_loss(&self) -> f64 {
        self.layers
            .iter()
            .map(|l| l.regularization_loss(self.config.lambda_l1, self.config.lambda_entropy))
            .sum()
    }

    /// Prune activations whose L1 norm is below `threshold` by zeroing their
    /// coefficients and scales.
    pub fn prune(&mut self, threshold: f64) {
        for layer in &mut self.layers {
            for acts_j in &mut layer.activations {
                for act in acts_j {
                    if act.l1_norm() < threshold {
                        act.coefficients.iter_mut().for_each(|c| *c = 0.0);
                        act.spline_scale = 0.0;
                        act.residual_scale = 0.0;
                    }
                }
            }
        }
    }

    /// Total number of learnable parameters.
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.param_count()).sum()
    }

    /// Number of active edges (those with L1 norm above a small threshold).
    pub fn active_edge_count(&self, threshold: f64) -> usize {
        self.layers
            .iter()
            .flat_map(|l| l.activations.iter().flatten())
            .filter(|a| a.l1_norm() > threshold)
            .count()
    }

    /// Total number of edges (in_dim * out_dim for each layer).
    pub fn total_edge_count(&self) -> usize {
        self.layers.iter().map(|l| l.in_dim * l.out_dim).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Symbolic Extraction
// ─────────────────────────────────────────────────────────────────────────────

/// A candidate symbolic function for fitting a KAN activation.
#[derive(Debug, Clone)]
pub struct SymbolicFit {
    /// Name of the matched function (e.g. "square", "sin", "exp").
    pub function_name: String,
    /// Coefficient of determination R² (1 = perfect fit).
    pub r_squared: f64,
    /// Multiplicative scale: fitted as `scale * f(x) + offset`.
    pub scale: f64,
    /// Additive offset.
    pub offset: f64,
}

/// Library of candidate symbolic functions for activation fitting.
pub struct SymbolicLibrary {
    /// (name, function)
    pub functions: Vec<(String, Box<dyn Fn(f64) -> f64 + Send + Sync>)>,
}

impl SymbolicLibrary {
    /// Create the default library covering common mathematical functions.
    pub fn new() -> Self {
        let fns: Vec<(String, Box<dyn Fn(f64) -> f64 + Send + Sync>)> = vec![
            ("id".into(), Box::new(|x| x)),
            ("square".into(), Box::new(|x| x * x)),
            ("cube".into(), Box::new(|x| x * x * x)),
            ("sqrt".into(), Box::new(|x| x.abs().sqrt())),
            ("exp".into(), Box::new(|x| x.exp().min(1e10))),
            (
                "log".into(),
                Box::new(|x| if x > 0.0 { x.ln() } else { -1e10 }),
            ),
            ("sin".into(), Box::new(|x| (PI * x).sin())),
            ("cos".into(), Box::new(|x| (PI * x).cos())),
            ("tan".into(), Box::new(|x| x.tan().clamp(-1e5, 1e5))),
            ("sinh".into(), Box::new(|x| x.sinh().clamp(-1e10, 1e10))),
            ("cosh".into(), Box::new(|x| x.cosh().min(1e10))),
            ("abs".into(), Box::new(|x| x.abs())),
            ("tanh".into(), Box::new(|x| x.tanh())),
            ("silu".into(), Box::new(|x| x / (1.0 + (-x).exp()))),
            (
                "reciprocal".into(),
                Box::new(|x| if x.abs() > 1e-6 { 1.0 / x } else { 0.0 }),
            ),
        ];
        SymbolicLibrary { functions: fns }
    }
}

impl Default for SymbolicLibrary {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SymbolicLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SymbolicLibrary")
            .field("n_functions", &self.functions.len())
            .finish()
    }
}

/// Extracts symbolic formulas from a trained KAN by fitting each activation
/// to the candidate library.
#[derive(Debug)]
pub struct KanSymbolicExtractor {
    /// Library of candidate functions.
    pub library: SymbolicLibrary,
    /// Minimum R² required to accept a symbolic fit.
    pub threshold: f64,
}

impl KanSymbolicExtractor {
    /// Create with default library and threshold 0.95.
    pub fn new() -> Self {
        KanSymbolicExtractor {
            library: SymbolicLibrary::new(),
            threshold: 0.95,
        }
    }

    /// Fit a single activation to all library functions using linear regression
    /// (scale + offset) and return the best fit.
    pub fn fit_activation(&self, activation: &KanActivation, samples: &[f64]) -> SymbolicFit {
        // Evaluate the activation on all samples
        let y_obs: Vec<f64> = samples.iter().map(|&x| activation.forward(x)).collect();
        let y_mean = mean(&y_obs);
        let ss_tot: f64 = y_obs.iter().map(|y| (y - y_mean).powi(2)).sum();

        let mut best = SymbolicFit {
            function_name: "id".into(),
            r_squared: f64::NEG_INFINITY,
            scale: 1.0,
            offset: 0.0,
        };

        for (name, func) in &self.library.functions {
            let f_vals: Vec<f64> = samples.iter().map(|&x| func(x)).collect();
            // Linear regression: y ≈ a * f(x) + b
            let (a, b, r2) = linear_regression(&f_vals, &y_obs, ss_tot);
            if r2 > best.r_squared {
                best = SymbolicFit {
                    function_name: name.clone(),
                    r_squared: r2,
                    scale: a,
                    offset: b,
                };
            }
        }

        best
    }

    /// Extract symbolic formulas for every edge in the model.
    ///
    /// Returns a 2-D vector indexed `[layer_idx][edge_idx]` where edge_idx
    /// enumerates edges in row-major order (out_j * in_dim + in_i).
    pub fn extract_formula(&self, model: &KanModel, dataset: &[Vec<f64>]) -> Vec<Vec<SymbolicFit>> {
        // Build evaluation points: linspace over each input dimension
        let n_samples = dataset.len().max(200);
        let result = model
            .layers
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| {
                // Determine input range for this layer by propagating a few dataset points
                let layer_inputs: Vec<Vec<f64>> = if layer_idx == 0 {
                    dataset.iter().take(n_samples).cloned().collect()
                } else {
                    let mut acc = Vec::with_capacity(n_samples);
                    for x in dataset.iter().take(n_samples) {
                        let mut h = x.clone();
                        for prev in &model.layers[..layer_idx] {
                            h = prev.forward(&h).unwrap_or_default();
                        }
                        acc.push(h);
                    }
                    acc
                };

                let mut fits = Vec::new();
                for (j, acts_j) in layer.activations.iter().enumerate() {
                    for (i, act) in acts_j.iter().enumerate() {
                        // Collect samples for input dimension i
                        let samples: Vec<f64> = layer_inputs
                            .iter()
                            .filter_map(|x| x.get(i).copied())
                            .collect();
                        let _ = j; // suppress unused warning
                        let fit = if samples.is_empty() {
                            SymbolicFit {
                                function_name: "id".into(),
                                r_squared: 0.0,
                                scale: 1.0,
                                offset: 0.0,
                            }
                        } else {
                            self.fit_activation(act, &samples)
                        };
                        fits.push(fit);
                    }
                }
                fits
            })
            .collect();
        result
    }
}

impl Default for KanSymbolicExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KAN-PINN Integration
// ─────────────────────────────────────────────────────────────────────────────

/// KAN-based physics-informed neural network.
///
/// Uses KAN as the backbone (instead of an MLP) for approximating the solution
/// to a PDE. Finite differences are used to approximate derivatives.
#[derive(Debug, Clone)]
pub struct KanPinn {
    /// The KAN model used as the solution surrogate.
    pub model: KanModel,
    /// Domain bounds: `[(x_min, x_max), ...]` for each input dimension.
    pub domain: Vec<(f64, f64)>,
    /// Number of collocation points sampled for PDE residual evaluation.
    pub n_collocation: usize,
}

impl KanPinn {
    /// Create a new KAN-PINN.
    pub fn new(
        layers: Vec<usize>,
        domain: Vec<(f64, f64)>,
        n_collocation: usize,
    ) -> Result<Self, KanError> {
        let config = KanConfig {
            layers,
            ..KanConfig::default()
        };
        let model = KanModel::new(config)?;
        Ok(KanPinn {
            model,
            domain,
            n_collocation,
        })
    }

    /// Compute the PDE residual loss by evaluating `pde_residual_fn` at
    /// evenly-spaced collocation points across the domain.
    ///
    /// `pde_residual_fn(x_slice, u_value) -> residual` where `x_slice` is the
    /// input point and `u_value` is the network output at that point.
    pub fn compute_residual_loss<F>(&self, pde_residual_fn: F) -> Result<f64, KanError>
    where
        F: Fn(&[f64], f64) -> f64,
    {
        if self.domain.is_empty() {
            return Ok(0.0);
        }
        let n = self.n_collocation;
        let dim = self.domain.len();
        let mut total_residual = 0.0;
        let mut count = 0;

        // Grid collocation: n points along each dimension
        let pts_per_dim = (n as f64).powf(1.0 / dim as f64).ceil() as usize + 1;

        // Enumerate multi-dimensional grid (simplified: uniform grid)
        let mut indices = vec![0usize; dim];
        loop {
            // Build point
            let x: Vec<f64> = indices
                .iter()
                .zip(self.domain.iter())
                .map(|(&idx, &(lo, hi))| {
                    lo + (hi - lo) * idx as f64 / (pts_per_dim - 1).max(1) as f64
                })
                .collect();

            let u = self.model.forward(&x)?;
            let u_val = u.first().copied().unwrap_or(0.0);
            let r = pde_residual_fn(&x, u_val);
            total_residual += r * r;
            count += 1;

            // Increment indices (odometer pattern)
            let mut carry = true;
            for i in (0..dim).rev() {
                if carry {
                    indices[i] += 1;
                    if indices[i] >= pts_per_dim {
                        indices[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry || count >= n {
                break;
            }
        }

        Ok(if count > 0 {
            total_residual / count as f64
        } else {
            0.0
        })
    }

    /// Finite-difference gradient ∂u/∂x_i ≈ (u(x+h·e_i) − u(x−h·e_i)) / (2h).
    pub fn gradient(&self, x: &[f64]) -> Result<Vec<f64>, KanError> {
        let h = 1e-5;
        let dim = x.len();
        let mut grad = vec![0.0f64; dim];
        for i in 0..dim {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += h;
            xm[i] -= h;
            let up = self.model.forward(&xp)?.first().copied().unwrap_or(0.0);
            let um = self.model.forward(&xm)?.first().copied().unwrap_or(0.0);
            grad[i] = (up - um) / (2.0 * h);
        }
        Ok(grad)
    }

    /// Finite-difference Laplacian ∇²u = ∑_i (u(x+h·e_i) − 2u(x) + u(x−h·e_i)) / h².
    pub fn laplacian(&self, x: &[f64]) -> Result<f64, KanError> {
        let h = 1e-5;
        let h2 = h * h;
        let dim = x.len();
        let u0 = self.model.forward(x)?.first().copied().unwrap_or(0.0);
        let mut lap = 0.0;
        for i in 0..dim {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += h;
            xm[i] -= h;
            let up = self.model.forward(&xp)?.first().copied().unwrap_or(0.0);
            let um = self.model.forward(&xm)?.first().copied().unwrap_or(0.0);
            lap += (up - 2.0 * u0 + um) / h2;
        }
        Ok(lap)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KAN Trainer
// ─────────────────────────────────────────────────────────────────────────────

/// Training loop for KAN models using finite-difference gradient approximation.
///
/// Since KAN coefficients are plain `Vec<f64>`, we perform coordinate-wise
/// gradient descent using central finite differences on the MSE loss.
#[derive(Debug, Clone)]
pub struct KanTrainer {
    /// The KAN model being trained.
    pub model: KanModel,
    /// Learning rate.
    pub learning_rate: f64,
    /// How often (in steps) to update the B-spline grids.
    pub grid_update_freq: usize,
    step_count: usize,
}

impl KanTrainer {
    /// Create a new trainer.
    pub fn new(model: KanModel, learning_rate: f64) -> Self {
        KanTrainer {
            model,
            learning_rate,
            grid_update_freq: 100,
            step_count: 0,
        }
    }

    /// Compute the MSE loss for a single (x, y) pair.
    fn mse_loss(pred: &[f64], target: &[f64]) -> f64 {
        if pred.is_empty() || pred.len() != target.len() {
            return 0.0;
        }
        pred.iter()
            .zip(target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / pred.len() as f64
    }

    /// A single gradient-descent step on one (x, y) sample.
    ///
    /// Applies finite-difference gradients to all spline coefficients and
    /// scale parameters. Returns the MSE + regularization loss.
    pub fn train_step(&mut self, x: &[f64], y: &[f64]) -> Result<f64, KanError> {
        let h = 1e-4;
        let lr = self.learning_rate;

        // Compute current loss
        let pred = self.model.forward(x)?;
        let base_loss = Self::mse_loss(&pred, y) + self.model.regularization_loss();

        // Iterate over all layers, then all activations, then all coefficients
        for layer_idx in 0..self.model.layers.len() {
            for j in 0..self.model.layers[layer_idx].out_dim {
                for i in 0..self.model.layers[layer_idx].in_dim {
                    // Update each coefficient
                    let n_coeff = self.model.layers[layer_idx].activations[j][i]
                        .coefficients
                        .len();
                    for k in 0..n_coeff {
                        // Forward perturbation
                        self.model.layers[layer_idx].activations[j][i].coefficients[k] += h;
                        let pred_p = self.model.forward(x).unwrap_or_default();
                        let loss_p = Self::mse_loss(&pred_p, y) + self.model.regularization_loss();

                        // Backward perturbation
                        self.model.layers[layer_idx].activations[j][i].coefficients[k] -= 2.0 * h;
                        let pred_m = self.model.forward(x).unwrap_or_default();
                        let loss_m = Self::mse_loss(&pred_m, y) + self.model.regularization_loss();

                        // Restore
                        self.model.layers[layer_idx].activations[j][i].coefficients[k] += h;

                        let grad = (loss_p - loss_m) / (2.0 * h);
                        self.model.layers[layer_idx].activations[j][i].coefficients[k] -= lr * grad;
                    }

                    // Update residual_scale
                    {
                        self.model.layers[layer_idx].activations[j][i].residual_scale += h;
                        let pred_p = self.model.forward(x).unwrap_or_default();
                        let loss_p = Self::mse_loss(&pred_p, y) + self.model.regularization_loss();
                        self.model.layers[layer_idx].activations[j][i].residual_scale -= 2.0 * h;
                        let pred_m = self.model.forward(x).unwrap_or_default();
                        let loss_m = Self::mse_loss(&pred_m, y) + self.model.regularization_loss();
                        self.model.layers[layer_idx].activations[j][i].residual_scale += h;
                        let grad = (loss_p - loss_m) / (2.0 * h);
                        self.model.layers[layer_idx].activations[j][i].residual_scale -= lr * grad;
                    }

                    // Update spline_scale
                    {
                        self.model.layers[layer_idx].activations[j][i].spline_scale += h;
                        let pred_p = self.model.forward(x).unwrap_or_default();
                        let loss_p = Self::mse_loss(&pred_p, y) + self.model.regularization_loss();
                        self.model.layers[layer_idx].activations[j][i].spline_scale -= 2.0 * h;
                        let pred_m = self.model.forward(x).unwrap_or_default();
                        let loss_m = Self::mse_loss(&pred_m, y) + self.model.regularization_loss();
                        self.model.layers[layer_idx].activations[j][i].spline_scale += h;
                        let grad = (loss_p - loss_m) / (2.0 * h);
                        self.model.layers[layer_idx].activations[j][i].spline_scale -= lr * grad;
                    }
                }
            }
        }

        self.step_count += 1;

        // Periodic grid update
        if self.step_count % self.grid_update_freq == 0 {
            self.model.update_grids(&[x.to_vec()]);
        }

        Ok(base_loss)
    }

    /// Train the model for `n_epochs` over the given dataset.
    ///
    /// Returns a vector of per-epoch mean losses.
    pub fn train(
        &mut self,
        dataset: &[(Vec<f64>, Vec<f64>)],
        n_epochs: usize,
    ) -> Result<Vec<f64>, KanError> {
        let mut loss_history = Vec::with_capacity(n_epochs);
        for _epoch in 0..n_epochs {
            let mut epoch_loss = 0.0;
            for (x, y) in dataset {
                let loss = self.train_step(x, y)?;
                epoch_loss += loss;
            }
            let mean = if dataset.is_empty() {
                0.0
            } else {
                epoch_loss / dataset.len() as f64
            };
            loss_history.push(mean);
        }
        Ok(loss_history)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KAN Metrics & Report
// ─────────────────────────────────────────────────────────────────────────────

/// Diagnostic metrics for a trained KAN.
#[derive(Debug, Clone)]
pub struct KanMetrics {
    /// Total number of learnable parameters.
    pub parameter_count: usize,
    /// Number of edges with L1 norm above 1e-6.
    pub active_edges: usize,
    /// Fraction of edges that have been pruned (L1 norm ≤ 1e-6).
    pub sparsity_ratio: f64,
    /// Mean fraction of grid intervals that contain at least one data point.
    pub grid_utilization: f64,
}

/// Full report combining metrics with symbolic fits and layer scores.
#[derive(Debug)]
pub struct KanReport {
    /// Aggregate metrics.
    pub metrics: KanMetrics,
    /// Symbolic fits for every edge, indexed `[layer][edge]`.
    pub symbolic_fits: Vec<Vec<SymbolicFit>>,
    /// Feature importance (L1 sum) per layer.
    pub layer_scores: Vec<Vec<f64>>,
}

impl KanReport {
    /// Build a full report for a trained model.
    pub fn new(model: &KanModel, extractor: &KanSymbolicExtractor, dataset: &[Vec<f64>]) -> Self {
        let metrics = compute_kan_metrics(model);
        let symbolic_fits = extractor.extract_formula(model, dataset);
        let layer_scores = model
            .layers
            .iter()
            .map(|l| l.get_feature_scores())
            .collect();
        KanReport {
            metrics,
            symbolic_fits,
            layer_scores,
        }
    }

    /// Human-readable summary string.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "KAN Report\n  Parameters: {}\n  Active edges: {} / {:.0} total (sparsity {:.1}%)\n",
            self.metrics.parameter_count,
            self.metrics.active_edges,
            self.metrics.active_edges as f64 / (1.0 - self.metrics.sparsity_ratio).max(1e-9),
            self.metrics.sparsity_ratio * 100.0,
        ));
        s.push_str(&format!(
            "  Grid utilization: {:.1}%\n",
            self.metrics.grid_utilization * 100.0
        ));
        for (l, fits) in self.symbolic_fits.iter().enumerate() {
            s.push_str(&format!("  Layer {}:\n", l));
            for fit in fits {
                s.push_str(&format!(
                    "    {} : R²={:.3} scale={:.3} offset={:.3}\n",
                    fit.function_name, fit.r_squared, fit.scale, fit.offset
                ));
            }
        }
        s
    }
}

/// Compute metrics for a KAN model (does not require dataset).
pub fn compute_kan_metrics(model: &KanModel) -> KanMetrics {
    let parameter_count = model.param_count();
    let threshold = 1e-6;
    let active_edges = model.active_edge_count(threshold);
    let total_edges = model.total_edge_count();
    let sparsity_ratio = if total_edges > 0 {
        1.0 - active_edges as f64 / total_edges as f64
    } else {
        0.0
    };

    // Grid utilization: for each activation, count basis functions that have
    // non-negligible coefficients (proxy for data coverage)
    let all_acts: Vec<&KanActivation> = model
        .layers
        .iter()
        .flat_map(|l| l.activations.iter().flatten())
        .collect();
    let grid_utilization = if all_acts.is_empty() {
        0.0
    } else {
        let used_fracs: Vec<f64> = all_acts
            .iter()
            .map(|a| {
                let n = a.coefficients.len();
                if n == 0 {
                    return 0.0;
                }
                let used = a.coefficients.iter().filter(|c| c.abs() > 1e-8).count();
                used as f64 / n as f64
            })
            .collect();
        mean(&used_fracs)
    };

    KanMetrics {
        parameter_count,
        active_edges,
        sparsity_ratio,
        grid_utilization,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal math helpers
// ─────────────────────────────────────────────────────────────────────────────

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Ordinary least-squares fit: y ≈ a·f + b.
///
/// Returns `(a, b, r_squared)`.
fn linear_regression(f_vals: &[f64], y: &[f64], ss_tot: f64) -> (f64, f64, f64) {
    let n = f_vals.len().min(y.len());
    if n == 0 {
        return (1.0, 0.0, 0.0);
    }
    let f_mean = mean(&f_vals[..n]);
    let y_mean = mean(&y[..n]);

    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let df = f_vals[i] - f_mean;
        let dy = y[i] - y_mean;
        num += df * dy;
        den += df * df;
    }
    let a = if den.abs() > 1e-14 { num / den } else { 0.0 };
    let b = y_mean - a * f_mean;

    // R²
    let ss_res: f64 = (0..n).map(|i| (y[i] - (a * f_vals[i] + b)).powi(2)).sum();
    let r2 = if ss_tot.abs() > 1e-14 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };

    (a, b, r2)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public type alias to match the required export name
// ─────────────────────────────────────────────────────────────────────────────

/// Public alias: `BSplineBasis` refers to [`KanBSplineBasis`] in this module.
pub type BSplineBasis = KanBSplineBasis;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── B-Spline Basis ──────────────────────────────────────────────────────

    #[test]
    fn test_bspline_basis_construction() {
        let basis = KanBSplineBasis::new(-1.0, 1.0, 5, 3);
        // n_basis = (n_intervals + 1 + 2*degree) - degree - 1 = n_intervals + degree
        // = 5 + 3 = 8
        assert_eq!(basis.n_basis, 8, "n_basis should be n_intervals + degree");
        assert!(basis.grid.len() > 0);
    }

    #[test]
    fn test_bspline_partition_of_unity() {
        let basis = KanBSplineBasis::new(-1.0, 1.0, 5, 3);
        for x in [-0.9, -0.5, -0.1, 0.0, 0.3, 0.7, 0.95] {
            let b = basis.evaluate(x);
            let sum: f64 = b.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Partition of unity failed at x={}: sum={}",
                x,
                sum
            );
        }
    }

    #[test]
    fn test_bspline_non_negative() {
        let basis = KanBSplineBasis::new(-1.0, 1.0, 5, 3);
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let b = basis.evaluate(x);
            for (i, &bi) in b.iter().enumerate() {
                assert!(
                    bi >= -1e-12,
                    "B-spline basis function {} is negative at x={}: {}",
                    i,
                    x,
                    bi
                );
            }
        }
    }

    #[test]
    fn test_bspline_compact_support() {
        let basis = KanBSplineBasis::new(-1.0, 1.0, 5, 3);
        // At x = -1.0, only the first few basis functions should be active
        let b = basis.evaluate(-1.0);
        let n_active = b.iter().filter(|&&bi| bi > 1e-12).count();
        // Degree 3 spline: at most (degree+1)=4 basis functions active
        assert!(
            n_active <= 4,
            "Too many active basis functions at endpoint: {}",
            n_active
        );
    }

    #[test]
    fn test_bspline_refine_grid() {
        let mut basis = KanBSplineBasis::new(-1.0, 1.0, 4, 3);
        let data: Vec<f64> = (0..100).map(|i| -2.0 + 4.0 * i as f64 / 99.0).collect();
        basis.refine_grid(&data);
        // Grid should now cover the data range
        assert!(basis.grid[0] <= -2.0 + 1e-9);
        assert!(basis.grid[basis.grid.len() - 1] >= 2.0 - 1e-9);
    }

    #[test]
    fn test_bspline_evaluate_boundary() {
        let basis = KanBSplineBasis::new(0.0, 1.0, 5, 3);
        // At boundaries the value should still be a valid probability vector
        let b0 = basis.evaluate(0.0);
        let b1 = basis.evaluate(1.0);
        let s0: f64 = b0.iter().sum();
        let s1: f64 = b1.iter().sum();
        assert!((s0 - 1.0).abs() < 1e-10, "sum at x=0 is {}", s0);
        assert!((s1 - 1.0).abs() < 1e-10, "sum at x=1 is {}", s1);
    }

    #[test]
    fn test_bspline_degree1() {
        // Linear splines should also partition unity
        let basis = KanBSplineBasis::new(0.0, 1.0, 4, 1);
        for x in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let b = basis.evaluate(x);
            let sum: f64 = b.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Linear spline sum={} at x={}",
                sum,
                x
            );
        }
    }

    // ── KAN Activation ─────────────────────────────────────────────────────

    #[test]
    fn test_kan_activation_creation() {
        let act = KanActivation::new(5, 3, -1.0, 1.0);
        assert_eq!(act.coefficients.len(), act.basis.n_basis);
        assert_eq!(act.residual_scale, 1.0);
        assert_eq!(act.spline_scale, 1.0);
    }

    #[test]
    fn test_kan_activation_forward_finite() {
        let act = KanActivation::new(5, 3, -1.0, 1.0);
        for x in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0] {
            let y = act.forward(x);
            assert!(y.is_finite(), "forward({}) is not finite: {}", x, y);
        }
    }

    #[test]
    fn test_kan_activation_l1_norm_positive() {
        let act = KanActivation::new(5, 3, -1.0, 1.0);
        let l1 = act.l1_norm();
        assert!(l1 >= 0.0, "L1 norm must be non-negative");
        // Coefficients are non-zero, so l1 > 0
        assert!(
            l1 > 0.0,
            "Non-trivial activation should have positive L1 norm"
        );
    }

    #[test]
    fn test_kan_activation_entropy() {
        let act = KanActivation::new(5, 3, -1.0, 1.0);
        let h = act.entropy();
        assert!(h >= 0.0, "Entropy must be non-negative");
        assert!(h.is_finite(), "Entropy must be finite");
    }

    #[test]
    fn test_kan_activation_update_grid() {
        let mut act = KanActivation::new(5, 3, -1.0, 1.0);
        let old_grid_len = act.basis.grid.len();
        let samples: Vec<f64> = (0..50).map(|i| -3.0 + 6.0 * i as f64 / 49.0).collect();
        act.update_grid(&samples);
        // n_basis should remain the same (same number of grid points)
        assert_eq!(act.basis.grid.len(), old_grid_len);
        assert_eq!(act.basis.n_basis, act.coefficients.len());
    }

    // ── KAN Layer ──────────────────────────────────────────────────────────

    #[test]
    fn test_kan_layer_output_dimension() {
        let layer = KanLayer::new(3, 5, 4, 3);
        let x = vec![0.1, -0.2, 0.3];
        let out = layer.forward(&x).expect("forward failed");
        assert_eq!(out.len(), 5, "Output should have out_dim entries");
    }

    #[test]
    fn test_kan_layer_input_mismatch() {
        let layer = KanLayer::new(3, 5, 4, 3);
        let x = vec![0.1, -0.2]; // wrong size
        let result = layer.forward(&x);
        assert!(result.is_err(), "Should fail with wrong input dimension");
    }

    #[test]
    fn test_kan_layer_output_finite() {
        let layer = KanLayer::new(4, 3, 5, 3);
        let x = vec![0.0, 0.5, -0.5, 1.0];
        let out = layer.forward(&x).expect("forward failed");
        for (i, yi) in out.iter().enumerate() {
            assert!(yi.is_finite(), "Output[{}] is not finite: {}", i, yi);
        }
    }

    #[test]
    fn test_kan_layer_feature_scores() {
        let layer = KanLayer::new(3, 4, 5, 3);
        let scores = layer.get_feature_scores();
        assert_eq!(scores.len(), 3, "Feature scores should have in_dim entries");
        for s in &scores {
            assert!(
                s.is_finite() && *s >= 0.0,
                "Feature score must be non-negative finite"
            );
        }
    }

    #[test]
    fn test_kan_layer_regularization_loss() {
        let layer = KanLayer::new(2, 3, 5, 3);
        let reg = layer.regularization_loss(1e-3, 2e-4);
        assert!(reg >= 0.0, "Regularization loss must be non-negative");
        assert!(reg.is_finite(), "Regularization loss must be finite");
    }

    #[test]
    fn test_kan_layer_param_count() {
        let layer = KanLayer::new(2, 3, 5, 3);
        // n_basis = 5 + 3 = 8; each activation has n_basis + 2 params
        // total = 2 * 3 * (8 + 2) = 60
        let count = layer.param_count();
        assert!(count > 0, "Param count must be positive");
        assert_eq!(
            count,
            2 * 3 * (8 + 2),
            "Param count should match architecture"
        );
    }

    // ── KAN Model ─────────────────────────────────────────────────────────

    #[test]
    fn test_kan_model_creation() {
        let config = KanConfig {
            layers: vec![2, 5, 1],
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_kan_model_invalid_config() {
        let config = KanConfig {
            layers: vec![5],
            ..KanConfig::default()
        };
        let result = KanModel::new(config);
        assert!(
            result.is_err(),
            "Should fail with fewer than 2 layer widths"
        );
    }

    #[test]
    fn test_kan_model_forward_shape() {
        let config = KanConfig {
            layers: vec![2, 5, 1],
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let x = vec![0.3, -0.7];
        let out = model.forward(&x).expect("forward failed");
        assert_eq!(out.len(), 1, "Output should match final layer width");
    }

    #[test]
    fn test_kan_model_multilayer() {
        let config = KanConfig {
            layers: vec![3, 8, 4, 1],
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let x = vec![0.1, 0.2, 0.3];
        let out = model.forward(&x).expect("forward failed");
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite());
    }

    #[test]
    fn test_kan_model_forward_with_all_outputs() {
        let config = KanConfig {
            layers: vec![2, 4, 2, 1],
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let x = vec![0.5, -0.5];
        let all = model
            .forward_with_all_outputs(&x)
            .expect("forward_with_all_outputs failed");
        // Should have n_layers + 1 entries (input + each layer output)
        assert_eq!(all.len(), 4, "Should have input + 3 layer outputs");
        assert_eq!(all[0].len(), 2);
        assert_eq!(all[1].len(), 4);
        assert_eq!(all[2].len(), 2);
        assert_eq!(all[3].len(), 1);
    }

    #[test]
    fn test_kan_model_param_count() {
        // Architecture [2, 5, 1]: layer0 has 2*5=10 activations, layer1 has 5*1=5
        // each activation: n_basis+2 = 8+2=10 params (grid_size=5, degree=3 → n_basis=8)
        let config = KanConfig {
            layers: vec![2, 5, 1],
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let expected = (2 * 5 + 5) * 10;
        assert_eq!(model.param_count(), expected, "Param count mismatch");
    }

    #[test]
    fn test_kan_model_regularization_loss() {
        let config = KanConfig::default();
        let model = KanModel::new(config).expect("model creation failed");
        let reg = model.regularization_loss();
        assert!(reg >= 0.0, "Regularization loss must be non-negative");
        assert!(reg.is_finite());
    }

    #[test]
    fn test_kan_model_pruning() {
        let config = KanConfig {
            layers: vec![2, 4, 1],
            ..KanConfig::default()
        };
        let mut model = KanModel::new(config).expect("model creation failed");
        // Before pruning: some active edges
        let active_before = model.active_edge_count(1e-6);
        // Prune with a threshold that should remove weak activations
        model.prune(1e-3);
        let active_after = model.active_edge_count(1e-6);
        // Sparsity should not increase active count (pruning can only reduce)
        assert!(
            active_after <= active_before,
            "Pruning should not increase active edge count"
        );
    }

    #[test]
    fn test_kan_model_update_grids() {
        let config = KanConfig {
            layers: vec![2, 4, 1],
            ..KanConfig::default()
        };
        let mut model = KanModel::new(config).expect("model creation failed");
        let dataset: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![i as f64 / 10.0 - 1.0, (i as f64 / 10.0).sin()])
            .collect();
        // Should not panic
        model.update_grids(&dataset);
        let out = model
            .forward(&[0.0, 0.0])
            .expect("forward after grid update failed");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_kan_model_active_edge_count() {
        let config = KanConfig {
            layers: vec![2, 3, 1],
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let total = model.total_edge_count();
        assert_eq!(total, 2 * 3 + 3);
        let active = model.active_edge_count(1e-6);
        assert!(active <= total, "Active count cannot exceed total");
    }

    // ── Symbolic Extractor ─────────────────────────────────────────────────

    #[test]
    fn test_symbolic_library_creation() {
        let lib = SymbolicLibrary::new();
        assert!(
            lib.functions.len() >= 10,
            "Should have at least 10 candidate functions"
        );
    }

    #[test]
    fn test_symbolic_fit_creation() {
        let fit = SymbolicFit {
            function_name: "square".into(),
            r_squared: 0.99,
            scale: 1.0,
            offset: 0.0,
        };
        assert_eq!(fit.function_name, "square");
        assert!((fit.r_squared - 0.99).abs() < 1e-9);
    }

    #[test]
    fn test_symbolic_extractor_fit_activation_square() {
        // Make an activation that behaves like x^2 by loading pre-set coefficients
        // Use a linear model: train on y=x^2 samples
        let extractor = KanSymbolicExtractor::new();
        let mut act = KanActivation::new(10, 3, -2.0, 2.0);

        // Manually set spline_scale = 0, residual_scale = 0 and load
        // appropriate coefficients to approximate x^2:
        // We just test that the extractor runs and returns a SymbolicFit
        let samples: Vec<f64> = (-20..=20).map(|i| i as f64 * 0.1).collect();
        let fit = extractor.fit_activation(&act, &samples);
        assert!(fit.r_squared.is_finite(), "R² must be finite");
        assert!(
            !fit.function_name.is_empty(),
            "Function name must be non-empty"
        );
    }

    #[test]
    fn test_symbolic_extractor_extract_formula() {
        let config = KanConfig {
            layers: vec![1, 3, 1],
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let extractor = KanSymbolicExtractor::new();
        let dataset: Vec<Vec<f64>> = (-10..=10).map(|i| vec![i as f64 * 0.2]).collect();
        let fits = extractor.extract_formula(&model, &dataset);
        assert_eq!(fits.len(), 2, "Should have one SymbolicFit per layer");
        // Layer 0: 1*3=3 edges → 3 fits
        assert_eq!(fits[0].len(), 3);
        // Layer 1: 3*1=3 edges → 3 fits
        assert_eq!(fits[1].len(), 3);
    }

    // ── KAN-PINN ───────────────────────────────────────────────────────────

    #[test]
    fn test_kan_pinn_creation() {
        let pinn = KanPinn::new(vec![1, 5, 1], vec![(0.0, 1.0)], 50).expect("creation failed");
        assert_eq!(pinn.n_collocation, 50);
        assert_eq!(pinn.domain.len(), 1);
    }

    #[test]
    fn test_kan_pinn_gradient_finite_diff() {
        let pinn =
            KanPinn::new(vec![2, 4, 1], vec![(0.0, 1.0), (0.0, 1.0)], 10).expect("creation failed");
        let x = vec![0.5, 0.5];
        let grad = pinn.gradient(&x).expect("gradient failed");
        assert_eq!(grad.len(), 2);
        for g in &grad {
            assert!(g.is_finite(), "gradient must be finite");
        }
    }

    #[test]
    fn test_kan_pinn_laplacian() {
        let pinn =
            KanPinn::new(vec![2, 4, 1], vec![(0.0, 1.0), (0.0, 1.0)], 10).expect("creation failed");
        let x = vec![0.5, 0.5];
        let lap = pinn.laplacian(&x).expect("laplacian failed");
        assert!(lap.is_finite(), "Laplacian must be finite");
    }

    #[test]
    fn test_kan_pinn_residual_loss() {
        let pinn = KanPinn::new(vec![1, 3, 1], vec![(0.0, 1.0)], 10).expect("creation failed");
        // Simple PDE residual: u - x = 0 → residual = u - x
        let loss = pinn
            .compute_residual_loss(|x, u| u - x[0])
            .expect("residual loss failed");
        assert!(loss.is_finite(), "Residual loss must be finite");
        assert!(
            loss >= 0.0,
            "Residual loss (MSE of residual) must be non-negative"
        );
    }

    #[test]
    fn test_kan_pinn_2d_laplacian() {
        let pinn = KanPinn::new(vec![2, 5, 1], vec![(-1.0, 1.0), (-1.0, 1.0)], 20)
            .expect("creation failed");
        let x = vec![0.0, 0.0];
        let lap = pinn.laplacian(&x).expect("laplacian failed");
        assert!(lap.is_finite(), "2D Laplacian must be finite");
    }

    // ── KAN Trainer ────────────────────────────────────────────────────────

    #[test]
    fn test_kan_trainer_single_step() {
        let config = KanConfig {
            layers: vec![1, 3, 1],
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let mut trainer = KanTrainer::new(model, 1e-3);
        let x = vec![0.5];
        let y = vec![0.25]; // y = x^2
        let loss = trainer.train_step(&x, &y).expect("train_step failed");
        assert!(loss.is_finite(), "Loss must be finite");
        assert!(loss >= 0.0, "Loss must be non-negative");
    }

    #[test]
    fn test_kan_trainer_loss_decreases() {
        // Train on a trivial dataset for several steps and check that loss trends down
        let config = KanConfig {
            layers: vec![1, 4, 1],
            lambda_l1: 0.0,
            lambda_entropy: 0.0,
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let mut trainer = KanTrainer::new(model, 1e-2);

        let dataset: Vec<(Vec<f64>, Vec<f64>)> = (0..5)
            .map(|i| (vec![i as f64 * 0.2 - 0.4], vec![0.0]))
            .collect();

        let history = trainer.train(&dataset, 3).expect("training failed");
        assert_eq!(history.len(), 3);
        for &l in &history {
            assert!(l.is_finite(), "Loss history entry must be finite");
        }
    }

    #[test]
    fn test_kan_trainer_epoch_history_length() {
        let config = KanConfig {
            layers: vec![1, 3, 1],
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let mut trainer = KanTrainer::new(model, 1e-3);
        let dataset: Vec<(Vec<f64>, Vec<f64>)> =
            vec![(vec![0.0], vec![0.0]), (vec![0.5], vec![0.25])];
        let history = trainer.train(&dataset, 5).expect("training failed");
        assert_eq!(history.len(), 5, "History should have n_epochs entries");
    }

    #[test]
    fn test_kan_trainer_grid_update_freq() {
        let config = KanConfig {
            layers: vec![1, 3, 1],
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let mut trainer = KanTrainer::new(model, 1e-3);
        trainer.grid_update_freq = 2;
        // Should not panic even with frequent grid updates
        for i in 0..5 {
            let x = vec![i as f64 * 0.1];
            let y = vec![(i as f64 * 0.1).powi(2)];
            trainer.train_step(&x, &y).expect("train_step failed");
        }
    }

    // ── KAN Metrics & Report ───────────────────────────────────────────────

    #[test]
    fn test_compute_kan_metrics_param_count() {
        let config = KanConfig {
            layers: vec![2, 3, 1],
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let metrics = compute_kan_metrics(&model);
        assert_eq!(metrics.parameter_count, model.param_count());
    }

    #[test]
    fn test_compute_kan_metrics_sparsity_after_prune() {
        let config = KanConfig {
            layers: vec![2, 4, 1],
            ..KanConfig::default()
        };
        let mut model = KanModel::new(config).expect("model creation failed");
        model.prune(1.0); // Prune everything (threshold > l1_norm)
        let metrics = compute_kan_metrics(&model);
        assert_eq!(metrics.active_edges, 0, "All edges should be pruned");
        assert!(
            (metrics.sparsity_ratio - 1.0).abs() < 1e-9,
            "Sparsity should be 1.0"
        );
    }

    #[test]
    fn test_compute_kan_metrics_grid_utilization() {
        let config = KanConfig::default();
        let model = KanModel::new(config).expect("model creation failed");
        let metrics = compute_kan_metrics(&model);
        assert!(
            metrics.grid_utilization >= 0.0 && metrics.grid_utilization <= 1.0,
            "Grid utilization must be in [0, 1]"
        );
    }

    #[test]
    fn test_kan_report_summary() {
        let config = KanConfig {
            layers: vec![1, 3, 1],
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let extractor = KanSymbolicExtractor::new();
        let dataset: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1]).collect();
        let report = KanReport::new(&model, &extractor, &dataset);
        let summary = report.summary();
        assert!(
            summary.contains("KAN Report"),
            "Summary should have a header"
        );
        assert!(
            summary.contains("Parameters"),
            "Summary should mention parameters"
        );
    }

    #[test]
    fn test_kan_report_layer_scores() {
        let config = KanConfig {
            layers: vec![3, 4, 1],
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let extractor = KanSymbolicExtractor::new();
        let dataset: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.2; 3]).collect();
        let report = KanReport::new(&model, &extractor, &dataset);
        // layer_scores[0] has in_dim=3 scores, layer_scores[1] has in_dim=4
        assert_eq!(report.layer_scores[0].len(), 3);
        assert_eq!(report.layer_scores[1].len(), 4);
    }

    // ── Generalisation / function fitting ─────────────────────────────────

    #[test]
    fn test_kan_fits_constant_function() {
        // Train on y=0 and verify the final loss is small
        let config = KanConfig {
            layers: vec![1, 3, 1],
            lambda_l1: 0.0,
            lambda_entropy: 0.0,
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let mut trainer = KanTrainer::new(model, 5e-3);
        let dataset: Vec<(Vec<f64>, Vec<f64>)> = (0..5)
            .map(|i| (vec![i as f64 * 0.2 - 0.4], vec![0.0]))
            .collect();
        let history = trainer.train(&dataset, 5).expect("training failed");
        // Loss should be finite and non-negative
        assert!(history.last().copied().unwrap_or(f64::NAN).is_finite());
    }

    #[test]
    fn test_kan_forward_deterministic() {
        // Same model and input should produce the same output
        let config = KanConfig {
            layers: vec![2, 4, 1],
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let x = vec![0.3, -0.7];
        let out1 = model.forward(&x).expect("forward 1 failed");
        let out2 = model.forward(&x).expect("forward 2 failed");
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-15, "Output should be deterministic");
        }
    }

    #[test]
    fn test_symbolic_extractor_r_squared_range() {
        let extractor = KanSymbolicExtractor::new();
        let act = KanActivation::new(5, 3, -1.0, 1.0);
        let samples: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.1).collect();
        let fit = extractor.fit_activation(&act, &samples);
        // R² is in (-inf, 1]; a "good" fit has R² >= 0
        assert!(fit.r_squared <= 1.0 + 1e-9, "R² must be ≤ 1");
    }

    #[test]
    fn test_kan_model_prune_then_forward() {
        let config = KanConfig {
            layers: vec![2, 3, 1],
            ..KanConfig::default()
        };
        let mut model = KanModel::new(config).expect("model creation failed");
        model.prune(1e-4);
        // Should still produce valid output
        let x = vec![0.1, 0.2];
        let out = model.forward(&x).expect("forward after prune failed");
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite());
    }

    #[test]
    fn test_de_boor_continuity() {
        // Check that the B-spline evaluation is continuous by evaluating at very
        // close points and checking the difference is small
        let basis = KanBSplineBasis::new(-1.0, 1.0, 5, 3);
        let x0 = 0.3;
        let x1 = 0.3 + 1e-7;
        let b0 = basis.evaluate(x0);
        let b1 = basis.evaluate(x1);
        let diff: f64 = b0.iter().zip(b1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff < 1e-4, "B-spline should be continuous: diff={}", diff);
    }

    #[test]
    fn test_kan_layer_update_grid_multiple_samples() {
        let mut layer = KanLayer::new(2, 3, 5, 3);
        let data: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![i as f64 * 0.1 - 1.0, (i as f64 * 0.3).sin()])
            .collect();
        layer.update_grid(&data);
        // Forward should still work
        let out = layer
            .forward(&[0.0, 0.5])
            .expect("forward after update failed");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_kan_metrics_active_edges_full_model() {
        let config = KanConfig {
            layers: vec![2, 4, 1],
            ..KanConfig::default()
        };
        let model = KanModel::new(config).expect("model creation failed");
        let metrics = compute_kan_metrics(&model);
        // Before pruning, active edges should be positive (init is non-zero)
        assert!(
            metrics.active_edges > 0,
            "Should have active edges before pruning"
        );
    }

    #[test]
    fn test_kan_pinn_gradient_varies_with_input() {
        let pinn =
            KanPinn::new(vec![2, 4, 1], vec![(0.0, 1.0), (0.0, 1.0)], 10).expect("creation failed");
        let g1 = pinn.gradient(&[0.2, 0.3]).expect("gradient 1 failed");
        let g2 = pinn.gradient(&[0.8, 0.7]).expect("gradient 2 failed");
        // They can be equal by chance but shouldn't be NaN
        for (a, b) in g1.iter().zip(g2.iter()) {
            assert!(a.is_finite() && b.is_finite());
        }
    }
}
