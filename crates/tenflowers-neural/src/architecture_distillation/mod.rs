//! # Neural Architecture Distillation & Efficient Architecture Search
//!
//! Production-grade module for jointly searching and distilling neural architectures.
//! Combines architecture search (NAS) with knowledge distillation to find compact,
//! high-performance student networks guided by teacher supervision.
//!
//! ## Components
//!
//! - [`AdLinear`] / [`AdMlp`] — Utility layers with Xavier initialization
//! - [`AdArchEncoding`] — Architecture encoding with 7 operation types
//! - [`NeuralPredictor`] — MLP accuracy predictor for cheap architecture ranking
//! - [`CompoundScaling`] — EfficientNet-style compound scaling (depth/width/resolution)
//! - [`InvertedResidualBlock`] — MobileNet-style inverted residual with SE
//! - [`ArchMorphism`] — Function-preserving architecture transformations
//! - [`ProgressiveShrinking`] — OFA-style elastic supernet training
//! - [`ProxyTask`] — Cheap proxy evaluation with correlation analysis
//! - [`ArchDistiller`] — Joint architecture search + knowledge distillation
//! - [`AdMetrics`] / [`AdReport`] — Architecture evaluation metrics

#[cfg(test)]
mod tests;

mod helpers;
pub use helpers::*;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Softmax with temperature scaling.
fn ad_softmax_temp(logits: &[f32], temp: f32) -> Vec<f32> {
    let t = temp.max(1e-8);
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| ((x - max_l) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum < 1e-30 {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}

/// KL divergence KL(p || q).
fn ad_kl_div(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi < 1e-30 {
                0.0
            } else {
                pi * ((pi / (qi + 1e-30)).ln())
            }
        })
        .sum()
}

/// Cross-entropy loss -sum(y_true * log(y_pred)).
fn ad_cross_entropy(y_true: &[f32], y_pred: &[f32]) -> f32 {
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| {
            if t < 1e-30 {
                0.0
            } else {
                -t * (p + 1e-30).ln()
            }
        })
        .sum()
}

/// MSE between two slices.
fn ad_mse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum / n as f32
}

/// Dot product.
fn ad_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm.
fn ad_l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// ReLU activation.
fn ad_relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Sigmoid activation.
fn ad_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Xavier uniform initialization bound: sqrt(6 / (fan_in + fan_out)).
fn ad_xavier_bound(fan_in: usize, fan_out: usize) -> f32 {
    (6.0 / (fan_in + fan_out) as f32).sqrt()
}

/// Matrix-vector multiply: out[i] = sum_j(w[i*cols + j] * x[j]) + b[i].
fn ad_matvec(w: &[f32], b: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows];
    for i in 0..rows {
        let mut sum = b[i];
        for j in 0..cols {
            sum += w[i * cols + j] * x[j];
        }
        out[i] = sum;
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// S1: AdLinear — Single linear layer with Xavier init
// ─────────────────────────────────────────────────────────────────────────────

/// A single linear (fully-connected) layer with Xavier-uniform initialization.
///
/// Computes `y = Wx + b` where W is `[out_features x in_features]`.
#[derive(Debug, Clone)]
pub struct AdLinear {
    /// Weight matrix stored row-major: `[out_features x in_features]`.
    pub weights: Vec<f32>,
    /// Bias vector of length `out_features`.
    pub bias: Vec<f32>,
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
}

impl AdLinear {
    /// Create a new linear layer with Xavier-uniform initialization.
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(TensorError::invalid_argument(
                "AdLinear: in_features and out_features must be > 0".into(),
            ));
        }
        let bound = ad_xavier_bound(in_features, out_features);
        let mut rng = StdRng::seed_from_u64(seed);
        let n_weights = in_features * out_features;
        let weights: Vec<f32> = (0..n_weights)
            .map(|_| rng.random_range(-bound..bound))
            .collect();
        let bias = vec![0.0; out_features];
        Ok(Self {
            weights,
            bias,
            in_features,
            out_features,
        })
    }

    /// Forward pass: y = Wx + b.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(TensorError::invalid_argument(format!(
                "AdLinear: expected input len {}, got {}",
                self.in_features,
                x.len()
            )));
        }
        Ok(ad_matvec(
            &self.weights,
            &self.bias,
            x,
            self.out_features,
            self.in_features,
        ))
    }

    /// Number of trainable parameters.
    pub fn param_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// S2: AdMlp — Multi-layer perceptron with ReLU hidden layers
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-layer perceptron with ReLU activations on hidden layers.
///
/// The output layer has no activation (raw logits).
#[derive(Debug, Clone)]
pub struct AdMlp {
    /// Sequence of linear layers.
    pub layers: Vec<AdLinear>,
}

impl AdMlp {
    /// Create an MLP with the specified layer sizes.
    ///
    /// `sizes` must have at least 2 elements (input, output).
    /// Hidden layers use ReLU; the final layer has no activation.
    pub fn new(sizes: &[usize], seed: u64) -> Result<Self> {
        if sizes.len() < 2 {
            return Err(TensorError::invalid_argument(
                "AdMlp: need at least 2 layer sizes (input, output)".into(),
            ));
        }
        let mut layers = Vec::with_capacity(sizes.len() - 1);
        for (i, pair) in sizes.windows(2).enumerate() {
            layers.push(AdLinear::new(
                pair[0],
                pair[1],
                seed.wrapping_add(i as u64),
            )?);
        }
        Ok(Self { layers })
    }

    /// Forward pass through all layers.
    /// Hidden layers use ReLU; the output layer is linear (no activation).
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let n_layers = self.layers.len();
        let mut h = x.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h)?;
            // Apply ReLU to all hidden layers (not the last one)
            if i < n_layers - 1 {
                h.iter_mut().for_each(|v| *v = ad_relu(*v));
            }
        }
        Ok(h)
    }

    /// Total trainable parameter count.
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.param_count()).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// S3: AdArchEncoding — Architecture encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Operation types available in a NAS cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdOpType {
    /// Standard 3x3 convolution.
    Conv3x3,
    /// Standard 5x5 convolution.
    Conv5x5,
    /// Depthwise separable convolution.
    DWConv,
    /// Max pooling 3x3.
    MaxPool,
    /// Average pooling 3x3.
    AvgPool,
    /// Skip (identity) connection.
    Skip,
    /// Zero (no connection).
    Zero,
}

impl AdOpType {
    /// Total number of operation types.
    pub const fn count() -> usize {
        7
    }

    /// Convert an integer index (0..6) to an operation type.
    pub fn from_index(idx: usize) -> Result<Self> {
        match idx {
            0 => Ok(AdOpType::Conv3x3),
            1 => Ok(AdOpType::Conv5x5),
            2 => Ok(AdOpType::DWConv),
            3 => Ok(AdOpType::MaxPool),
            4 => Ok(AdOpType::AvgPool),
            5 => Ok(AdOpType::Skip),
            6 => Ok(AdOpType::Zero),
            _ => Err(TensorError::invalid_argument(format!(
                "AdOpType: index {} out of range [0, 6]",
                idx
            ))),
        }
    }

    /// Convert to integer index.
    pub fn to_index(self) -> usize {
        match self {
            AdOpType::Conv3x3 => 0,
            AdOpType::Conv5x5 => 1,
            AdOpType::DWConv => 2,
            AdOpType::MaxPool => 3,
            AdOpType::AvgPool => 4,
            AdOpType::Skip => 5,
            AdOpType::Zero => 6,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &str {
        match self {
            AdOpType::Conv3x3 => "conv3x3",
            AdOpType::Conv5x5 => "conv5x5",
            AdOpType::DWConv => "dw_conv",
            AdOpType::MaxPool => "max_pool",
            AdOpType::AvgPool => "avg_pool",
            AdOpType::Skip => "skip",
            AdOpType::Zero => "zero",
        }
    }

    /// FLOPs estimate per spatial position for given channels.
    pub fn flops_per_position(&self, channels: usize) -> usize {
        let c2 = channels * channels;
        match self {
            AdOpType::Conv3x3 => 2 * 9 * c2,
            AdOpType::Conv5x5 => 2 * 25 * c2,
            AdOpType::DWConv => 2 * 9 * channels + 2 * c2,
            AdOpType::MaxPool => 9 * channels,
            AdOpType::AvgPool => 9 * channels,
            AdOpType::Skip => channels,
            AdOpType::Zero => 0,
        }
    }

    /// Parameter count estimate for given channels.
    pub fn params_estimate(&self, channels: usize) -> usize {
        let c2 = channels * channels;
        match self {
            AdOpType::Conv3x3 => 9 * c2 + channels,
            AdOpType::Conv5x5 => 25 * c2 + channels,
            AdOpType::DWConv => 9 * channels + c2 + 2 * channels,
            AdOpType::MaxPool | AdOpType::AvgPool | AdOpType::Skip | AdOpType::Zero => 0,
        }
    }

    /// All operation types.
    pub fn all() -> [AdOpType; 7] {
        [
            AdOpType::Conv3x3,
            AdOpType::Conv5x5,
            AdOpType::DWConv,
            AdOpType::MaxPool,
            AdOpType::AvgPool,
            AdOpType::Skip,
            AdOpType::Zero,
        ]
    }
}

/// A single edge in a NAS cell: two operations and a connection source node.
#[derive(Debug, Clone)]
pub struct AdEdge {
    /// First operation.
    pub op1: AdOpType,
    /// Second operation.
    pub op2: AdOpType,
    /// Source node index for this edge pair.
    pub connection: usize,
}

/// Architecture encoding as a sequence of cell edges.
///
/// A cell consists of `n_nodes` intermediate nodes. Each node `i` has edges
/// from previous nodes (0..i+2, including 2 input nodes).
#[derive(Debug, Clone)]
pub struct AdArchEncoding {
    /// Number of intermediate nodes in the cell.
    pub n_nodes: usize,
    /// Edges defining the cell topology.
    pub edges: Vec<AdEdge>,
    /// Number of channels (width) for this architecture.
    pub channels: usize,
    /// Number of stacked cells (depth).
    pub n_cells: usize,
}

impl AdArchEncoding {
    /// Create a new architecture encoding.
    pub fn new(
        n_nodes: usize,
        edges: Vec<AdEdge>,
        channels: usize,
        n_cells: usize,
    ) -> Result<Self> {
        if n_nodes == 0 {
            return Err(TensorError::invalid_argument(
                "AdArchEncoding: n_nodes must be > 0".into(),
            ));
        }
        if channels == 0 {
            return Err(TensorError::invalid_argument(
                "AdArchEncoding: channels must be > 0".into(),
            ));
        }
        if n_cells == 0 {
            return Err(TensorError::invalid_argument(
                "AdArchEncoding: n_cells must be > 0".into(),
            ));
        }
        Ok(Self {
            n_nodes,
            edges,
            channels,
            n_cells,
        })
    }

    /// Encode the architecture into a fixed-length float vector.
    ///
    /// Format: for each edge, emit [op1_onehot(7), op2_onehot(7), connection_norm].
    /// Then append [channels_norm, n_cells_norm, n_nodes_norm].
    pub fn encode(&self) -> Vec<f32> {
        let edge_dim = 2 * AdOpType::count() + 1; // 15 per edge
        let global_dim = 3;
        let total = self.edges.len() * edge_dim + global_dim;
        let mut vec = Vec::with_capacity(total);

        for edge in &self.edges {
            // One-hot for op1
            let mut oh1 = vec![0.0f32; AdOpType::count()];
            oh1[edge.op1.to_index()] = 1.0;
            vec.extend_from_slice(&oh1);
            // One-hot for op2
            let mut oh2 = vec![0.0f32; AdOpType::count()];
            oh2[edge.op2.to_index()] = 1.0;
            vec.extend_from_slice(&oh2);
            // Normalized connection index
            let max_conn = (self.n_nodes + 1).max(1);
            vec.push(edge.connection as f32 / max_conn as f32);
        }

        // Global features
        vec.push(self.channels as f32 / 512.0); // normalize to typical max
        vec.push(self.n_cells as f32 / 20.0);
        vec.push(self.n_nodes as f32 / 8.0);

        vec
    }

    /// Decode a description string from the architecture encoding.
    pub fn decode(&self) -> String {
        let mut desc = format!(
            "Arch(nodes={}, cells={}, channels={}): ",
            self.n_nodes, self.n_cells, self.channels
        );
        for (i, edge) in self.edges.iter().enumerate() {
            if i > 0 {
                desc.push_str(", ");
            }
            desc.push_str(&format!(
                "E{}[{}-{}<-{}]",
                i,
                edge.op1.name(),
                edge.op2.name(),
                edge.connection
            ));
        }
        desc
    }

    /// Estimate total FLOPs for this architecture on a given spatial resolution.
    pub fn estimate_flops(&self, resolution: usize) -> usize {
        let spatial = resolution * resolution;
        let mut flops = 0usize;
        for edge in &self.edges {
            flops += edge.op1.flops_per_position(self.channels) * spatial;
            flops += edge.op2.flops_per_position(self.channels) * spatial;
        }
        flops * self.n_cells
    }

    /// Estimate total parameter count.
    pub fn estimate_params(&self) -> usize {
        let mut params = 0usize;
        for edge in &self.edges {
            params += edge.op1.params_estimate(self.channels);
            params += edge.op2.params_estimate(self.channels);
        }
        params * self.n_cells
    }

    /// Generate a random architecture encoding.
    pub fn random(n_nodes: usize, channels: usize, n_cells: usize, seed: u64) -> Result<Self> {
        if n_nodes == 0 || channels == 0 || n_cells == 0 {
            return Err(TensorError::invalid_argument(
                "AdArchEncoding::random: all dimensions must be > 0".into(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut edges = Vec::new();
        for node_idx in 0..n_nodes {
            let max_conn = node_idx + 2; // 2 input nodes + previous intermediate nodes
            let op1 = AdOpType::from_index(rng.random_range(0..AdOpType::count()))?;
            let op2 = AdOpType::from_index(rng.random_range(0..AdOpType::count()))?;
            let connection = rng.random_range(0..max_conn);
            edges.push(AdEdge {
                op1,
                op2,
                connection,
            });
        }
        Self::new(n_nodes, edges, channels, n_cells)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// S4: NeuralPredictor — Accuracy predictor from architecture encoding
// ─────────────────────────────────────────────────────────────────────────────

/// MLP-based accuracy predictor that maps architecture encodings to predicted accuracy.
///
/// Used for cheap ranking of candidate architectures during NAS without actually
/// training each one. The predictor is trained on a small set of (architecture, accuracy)
/// pairs and generalizes to unseen architectures.
#[derive(Debug, Clone)]
pub struct NeuralPredictor {
    /// The underlying MLP.
    pub mlp: AdMlp,
    /// Expected encoding dimension.
    pub encoding_dim: usize,
    /// Learning rate for training.
    pub lr: f32,
}

impl NeuralPredictor {
    /// Create a new neural predictor.
    ///
    /// `encoding_dim` is the expected length of architecture encoding vectors.
    /// `hidden_dim` is the hidden layer size.
    pub fn new(encoding_dim: usize, hidden_dim: usize, seed: u64) -> Result<Self> {
        if encoding_dim == 0 || hidden_dim == 0 {
            return Err(TensorError::invalid_argument(
                "NeuralPredictor: dimensions must be > 0".into(),
            ));
        }
        let mlp = AdMlp::new(&[encoding_dim, hidden_dim, hidden_dim, 1], seed)?;
        Ok(Self {
            mlp,
            encoding_dim,
            lr: 0.001,
        })
    }

    /// Predict accuracy for a given architecture encoding.
    pub fn predict(&self, encoding: &[f32]) -> Result<f32> {
        if encoding.len() != self.encoding_dim {
            return Err(TensorError::invalid_argument(format!(
                "NeuralPredictor: expected encoding len {}, got {}",
                self.encoding_dim,
                encoding.len()
            )));
        }
        let out = self.mlp.forward(encoding)?;
        Ok(ad_sigmoid(out.first().copied().unwrap_or(0.0)))
    }

    /// Train the predictor on (encoding, accuracy) pairs.
    ///
    /// Uses MSE loss with simple SGD. Returns the final training loss.
    pub fn train(
        &mut self,
        encodings: &[Vec<f32>],
        accuracies: &[f32],
        epochs: usize,
    ) -> Result<f32> {
        if encodings.len() != accuracies.len() {
            return Err(TensorError::invalid_argument(
                "NeuralPredictor::train: encodings and accuracies must have same length".into(),
            ));
        }
        if encodings.is_empty() {
            return Err(TensorError::invalid_argument(
                "NeuralPredictor::train: need at least one training sample".into(),
            ));
        }

        let mut final_loss = 0.0;
        let lr = self.lr;
        let n = encodings.len() as f32;

        for _epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for (enc, &target) in encodings.iter().zip(accuracies.iter()) {
                let pred = self.predict(enc)?;
                let error = pred - target;
                epoch_loss += error * error;

                // Backprop through sigmoid: d_sigmoid = pred * (1 - pred)
                let d_sigmoid = pred * (1.0 - pred);
                let grad_output = 2.0 * error * d_sigmoid / n;

                // Simple gradient descent on last layer only (lightweight training)
                self.update_last_layer(enc, grad_output, lr)?;
            }

            final_loss = epoch_loss / n;
        }

        Ok(final_loss)
    }

    /// Update only the last layer weights via gradient descent.
    fn update_last_layer(&mut self, input: &[f32], grad_output: f32, lr: f32) -> Result<()> {
        // Forward to get hidden activations
        let n_layers = self.mlp.layers.len();
        let mut activations = vec![input.to_vec()];
        for (i, layer) in self.mlp.layers.iter().enumerate() {
            let mut h = layer.forward(&activations[i])?;
            if i < n_layers - 1 {
                h.iter_mut().for_each(|v| *v = ad_relu(*v));
            }
            activations.push(h);
        }

        // Update last layer: dW = grad_output * h_prev, db = grad_output
        if let Some(last) = self.mlp.layers.last_mut() {
            let h_prev = &activations[n_layers - 1];
            for j in 0..last.in_features {
                last.weights[j] -= lr * grad_output * h_prev[j];
            }
            last.bias[0] -= lr * grad_output;
        }

        Ok(())
    }

    /// Rank architectures by predicted accuracy (descending).
    /// Returns indices sorted by predicted accuracy.
    pub fn rank(&self, encodings: &[Vec<f32>]) -> Result<Vec<(usize, f32)>> {
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(encodings.len());
        for (i, enc) in encodings.iter().enumerate() {
            let acc = self.predict(enc)?;
            scored.push((i, acc));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// S5: CompoundScaling — EfficientNet-style compound scaling
// ─────────────────────────────────────────────────────────────────────────────

/// Base architecture description for compound scaling.
#[derive(Debug, Clone)]
pub struct AdBaseArch {
    /// Base depth (number of layers/blocks).
    pub depth: usize,
    /// Base width (number of channels).
    pub width: usize,
    /// Base input resolution.
    pub resolution: usize,
    /// Base FLOPs estimate.
    pub base_flops: f64,
}

/// Scaled architecture after compound scaling.
#[derive(Debug, Clone)]
pub struct AdScaledArch {
    /// Scaled depth.
    pub depth: usize,
    /// Scaled width.
    pub width: usize,
    /// Scaled resolution.
    pub resolution: usize,
    /// Estimated FLOPs for scaled arch.
    pub estimated_flops: f64,
    /// Compound coefficient used.
    pub phi: f64,
}

/// EfficientNet-style compound scaling.
///
/// Scales depth, width, and resolution simultaneously under the constraint
/// `alpha * beta^2 * gamma^2 ~ 2`, ensuring roughly 2x compute per phi step.
///
/// - depth = ceil(d_base * alpha^phi)
/// - width = round(w_base * beta^phi) aligned to 8
/// - resolution = round(r_base * gamma^phi)
#[derive(Debug, Clone)]
pub struct CompoundScaling {
    /// Depth scaling coefficient.
    pub alpha: f64,
    /// Width scaling coefficient.
    pub beta: f64,
    /// Resolution scaling coefficient.
    pub gamma: f64,
}

impl CompoundScaling {
    /// Create with explicit scaling coefficients.
    ///
    /// Validates that alpha * beta^2 * gamma^2 is approximately 2.
    pub fn new(alpha: f64, beta: f64, gamma: f64) -> Result<Self> {
        if alpha <= 0.0 || beta <= 0.0 || gamma <= 0.0 {
            return Err(TensorError::invalid_argument(
                "CompoundScaling: all coefficients must be > 0".into(),
            ));
        }
        let constraint = alpha * beta * beta * gamma * gamma;
        if !(1.5..=2.5).contains(&constraint) {
            return Err(TensorError::invalid_argument(format!(
                "CompoundScaling: alpha*beta^2*gamma^2 = {:.3}, should be ~2.0",
                constraint
            )));
        }
        Ok(Self { alpha, beta, gamma })
    }

    /// Search for optimal (alpha, beta, gamma) at phi=1 via grid search.
    ///
    /// Searches over a grid and returns the combination closest to the constraint.
    pub fn grid_search(
        alpha_range: (f64, f64),
        beta_range: (f64, f64),
        gamma_range: (f64, f64),
        n_steps: usize,
    ) -> Result<Self> {
        if n_steps == 0 {
            return Err(TensorError::invalid_argument(
                "CompoundScaling::grid_search: n_steps must be > 0".into(),
            ));
        }

        let mut best_alpha = 1.2;
        let mut best_beta = 1.1;
        let mut best_gamma = 1.15;
        let mut best_err = f64::MAX;

        let a_step = (alpha_range.1 - alpha_range.0) / n_steps as f64;
        let b_step = (beta_range.1 - beta_range.0) / n_steps as f64;
        let g_step = (gamma_range.1 - gamma_range.0) / n_steps as f64;

        let mut a = alpha_range.0;
        while a <= alpha_range.1 {
            let mut b = beta_range.0;
            while b <= beta_range.1 {
                let mut g = gamma_range.0;
                while g <= gamma_range.1 {
                    let constraint = a * b * b * g * g;
                    let err = (constraint - 2.0).abs();
                    if err < best_err {
                        best_err = err;
                        best_alpha = a;
                        best_beta = b;
                        best_gamma = g;
                    }
                    g += g_step;
                }
                b += b_step;
            }
            a += a_step;
        }

        // Relax constraint check for grid search results
        Ok(Self {
            alpha: best_alpha,
            beta: best_beta,
            gamma: best_gamma,
        })
    }

    /// Scale a base architecture by compound coefficient phi.
    pub fn scale(&self, base: &AdBaseArch, phi: f64) -> AdScaledArch {
        let depth = ((base.depth as f64) * self.alpha.powf(phi)).ceil() as usize;
        let width_raw = (base.width as f64) * self.beta.powf(phi);
        // Align width to nearest multiple of 8
        let width = ((width_raw / 8.0).round() * 8.0).max(8.0) as usize;
        let resolution = ((base.resolution as f64) * self.gamma.powf(phi)).round() as usize;
        let resolution = resolution.max(1);

        // FLOPs scale roughly as depth * width^2 * resolution^2
        let flops_ratio = (depth as f64 / base.depth as f64)
            * (width as f64 / base.width as f64).powi(2)
            * (resolution as f64 / base.resolution as f64).powi(2);
        let estimated_flops = base.base_flops * flops_ratio;

        AdScaledArch {
            depth,
            width,
            resolution,
            estimated_flops,
            phi,
        }
    }

    /// Compute the constraint value alpha * beta^2 * gamma^2.
    pub fn constraint_value(&self) -> f64 {
        self.alpha * self.beta * self.beta * self.gamma * self.gamma
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// S6: InvertedResidualBlock — MobileNet-style blocks
// ─────────────────────────────────────────────────────────────────────────────

/// Squeeze-and-Excitation block.
///
/// Performs global average pooling, then FC→ReLU→FC→Sigmoid to produce
/// per-channel scaling factors.
#[derive(Debug, Clone)]
pub struct AdSeBlock {
    /// Reduction ratio for the bottleneck.
    pub reduction: usize,
    /// FC1 weights: [reduced x channels].
    pub fc1_weights: Vec<f32>,
    /// FC1 bias.
    pub fc1_bias: Vec<f32>,
    /// FC2 weights: [channels x reduced].
    pub fc2_weights: Vec<f32>,
    /// FC2 bias.
    pub fc2_bias: Vec<f32>,
    /// Number of channels.
    pub channels: usize,
    /// Reduced dimension.
    pub reduced_dim: usize,
}

impl AdSeBlock {
    /// Create a new SE block.
    pub fn new(channels: usize, reduction: usize, seed: u64) -> Result<Self> {
        if channels == 0 || reduction == 0 {
            return Err(TensorError::invalid_argument(
                "AdSeBlock: channels and reduction must be > 0".into(),
            ));
        }
        let reduced_dim = (channels / reduction).max(1);
        let mut rng = StdRng::seed_from_u64(seed);

        let b1 = ad_xavier_bound(channels, reduced_dim);
        let fc1_weights: Vec<f32> = (0..reduced_dim * channels)
            .map(|_| rng.random_range(-b1..b1))
            .collect();
        let fc1_bias = vec![0.0; reduced_dim];

        let b2 = ad_xavier_bound(reduced_dim, channels);
        let fc2_weights: Vec<f32> = (0..channels * reduced_dim)
            .map(|_| rng.random_range(-b2..b2))
            .collect();
        let fc2_bias = vec![0.0; channels];

        Ok(Self {
            reduction,
            fc1_weights,
            fc1_bias,
            fc2_weights,
            fc2_bias,
            channels,
            reduced_dim,
        })
    }

    /// Forward pass: global avg pool → FC1 → ReLU → FC2 → Sigmoid → scale.
    ///
    /// `input` is a flat feature map of length `channels * spatial_size`.
    /// Returns the scaled feature map.
    pub fn forward(&self, input: &[f32], spatial_size: usize) -> Result<Vec<f32>> {
        if spatial_size == 0 {
            return Err(TensorError::invalid_argument(
                "AdSeBlock: spatial_size must be > 0".into(),
            ));
        }
        let expected = self.channels * spatial_size;
        if input.len() != expected {
            return Err(TensorError::invalid_argument(format!(
                "AdSeBlock: expected input len {}, got {}",
                expected,
                input.len()
            )));
        }

        // Global average pooling per channel
        let mut avg = vec![0.0f32; self.channels];
        for c in 0..self.channels {
            let mut sum = 0.0f32;
            for s in 0..spatial_size {
                sum += input[c * spatial_size + s];
            }
            avg[c] = sum / spatial_size as f32;
        }

        // FC1 + ReLU
        let h = ad_matvec(
            &self.fc1_weights,
            &self.fc1_bias,
            &avg,
            self.reduced_dim,
            self.channels,
        );
        let h: Vec<f32> = h.into_iter().map(ad_relu).collect();

        // FC2 + Sigmoid
        let scale = ad_matvec(
            &self.fc2_weights,
            &self.fc2_bias,
            &h,
            self.channels,
            self.reduced_dim,
        );
        let scale: Vec<f32> = scale.into_iter().map(ad_sigmoid).collect();

        // Scale input channels
        let mut output = input.to_vec();
        for c in 0..self.channels {
            for s in 0..spatial_size {
                output[c * spatial_size + s] *= scale[c];
            }
        }

        Ok(output)
    }
}

/// MobileNet-style inverted residual block.
///
/// Pipeline: Expansion (1x1) → Depthwise conv → SE → Projection (1x1).
/// Skip connection when `in_channels == out_channels` and `stride == 1`.
#[derive(Debug, Clone)]
pub struct InvertedResidualBlock {
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Expansion ratio.
    pub expand_ratio: usize,
    /// Kernel size for depthwise convolution.
    pub kernel_size: usize,
    /// Stride for depthwise convolution.
    pub stride: usize,
    /// Expansion projection weights [expanded x in_channels].
    pub expand_weights: Vec<f32>,
    /// Expansion bias.
    pub expand_bias: Vec<f32>,
    /// Depthwise conv weights [expanded x kernel_size].
    pub dw_weights: Vec<f32>,
    /// Depthwise bias.
    pub dw_bias: Vec<f32>,
    /// Squeeze-and-Excite block.
    pub se: AdSeBlock,
    /// Projection weights [out_channels x expanded].
    pub project_weights: Vec<f32>,
    /// Projection bias.
    pub project_bias: Vec<f32>,
    /// Expanded channel count.
    pub expanded_channels: usize,
    /// Whether to use skip connection.
    pub use_skip: bool,
}

impl InvertedResidualBlock {
    /// Create a new inverted residual block.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        expand_ratio: usize,
        kernel_size: usize,
        stride: usize,
        seed: u64,
    ) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(TensorError::invalid_argument(
                "InvertedResidualBlock: channels must be > 0".into(),
            ));
        }
        if expand_ratio == 0 {
            return Err(TensorError::invalid_argument(
                "InvertedResidualBlock: expand_ratio must be > 0".into(),
            ));
        }
        if kernel_size == 0 {
            return Err(TensorError::invalid_argument(
                "InvertedResidualBlock: kernel_size must be > 0".into(),
            ));
        }
        if stride == 0 {
            return Err(TensorError::invalid_argument(
                "InvertedResidualBlock: stride must be > 0".into(),
            ));
        }

        let expanded_channels = in_channels * expand_ratio;
        let mut rng = StdRng::seed_from_u64(seed);

        // Expansion 1x1 conv
        let b_exp = ad_xavier_bound(in_channels, expanded_channels);
        let expand_weights: Vec<f32> = (0..expanded_channels * in_channels)
            .map(|_| rng.random_range(-b_exp..b_exp))
            .collect();
        let expand_bias = vec![0.0; expanded_channels];

        // Depthwise conv
        let b_dw = ad_xavier_bound(kernel_size, 1);
        let dw_weights: Vec<f32> = (0..expanded_channels * kernel_size)
            .map(|_| rng.random_range(-b_dw..b_dw))
            .collect();
        let dw_bias = vec![0.0; expanded_channels];

        // SE block
        let se = AdSeBlock::new(expanded_channels, 4, seed.wrapping_add(100))?;

        // Projection 1x1 conv
        let b_proj = ad_xavier_bound(expanded_channels, out_channels);
        let project_weights: Vec<f32> = (0..out_channels * expanded_channels)
            .map(|_| rng.random_range(-b_proj..b_proj))
            .collect();
        let project_bias = vec![0.0; out_channels];

        let use_skip = in_channels == out_channels && stride == 1;

        Ok(Self {
            in_channels,
            out_channels,
            expand_ratio,
            kernel_size,
            stride,
            expand_weights,
            expand_bias,
            dw_weights,
            dw_bias,
            se,
            project_weights,
            project_bias,
            expanded_channels,
            use_skip,
        })
    }

    /// Forward pass through the inverted residual block.
    ///
    /// `input` is a flat feature vector of length `in_channels * spatial_size`.
    /// For simplicity, spatial dimensions are abstracted as a single `spatial_size`.
    pub fn forward(&self, input: &[f32], spatial_size: usize) -> Result<Vec<f32>> {
        if spatial_size == 0 {
            return Err(TensorError::invalid_argument(
                "InvertedResidualBlock::forward: spatial_size must be > 0".into(),
            ));
        }
        let expected = self.in_channels * spatial_size;
        if input.len() != expected {
            return Err(TensorError::invalid_argument(format!(
                "InvertedResidualBlock: expected input len {}, got {}",
                expected,
                input.len()
            )));
        }

        // 1. Expansion: apply 1x1 conv (linear map per spatial position)
        let out_spatial = spatial_size / self.stride;
        let out_spatial = out_spatial.max(1);
        let mut expanded = vec![0.0f32; self.expanded_channels * out_spatial];
        for s in 0..out_spatial {
            let s_in = s * self.stride;
            for ec in 0..self.expanded_channels {
                let mut val = self.expand_bias[ec];
                for ic in 0..self.in_channels {
                    val += self.expand_weights[ec * self.in_channels + ic]
                        * input[ic * spatial_size + s_in.min(spatial_size - 1)];
                }
                expanded[ec * out_spatial + s] = ad_relu(val);
            }
        }

        // 2. Depthwise conv (simplified 1D with kernel_size)
        let mut dw_out = vec![0.0f32; self.expanded_channels * out_spatial];
        let half_k = self.kernel_size / 2;
        for c in 0..self.expanded_channels {
            for s in 0..out_spatial {
                let mut val = self.dw_bias[c];
                for k in 0..self.kernel_size {
                    let idx = (s + k).saturating_sub(half_k);
                    let idx = idx.min(out_spatial - 1);
                    val +=
                        self.dw_weights[c * self.kernel_size + k] * expanded[c * out_spatial + idx];
                }
                dw_out[c * out_spatial + s] = ad_relu(val);
            }
        }

        // 3. SE block
        let se_out = self.se.forward(&dw_out, out_spatial)?;

        // 4. Projection: 1x1 conv (no activation — linear bottleneck)
        let mut output = vec![0.0f32; self.out_channels * out_spatial];
        for s in 0..out_spatial {
            for oc in 0..self.out_channels {
                let mut val = self.project_bias[oc];
                for ec in 0..self.expanded_channels {
                    val += self.project_weights[oc * self.expanded_channels + ec]
                        * se_out[ec * out_spatial + s];
                }
                output[oc * out_spatial + s] = val; // No activation (linear)
            }
        }

        // 5. Skip connection
        if self.use_skip {
            for oc in 0..self.out_channels {
                for s in 0..out_spatial {
                    output[oc * out_spatial + s] += input[oc * spatial_size + s];
                }
            }
        }

        Ok(output)
    }

    /// Estimate FLOPs for this block at given spatial size.
    pub fn estimate_flops(&self, spatial_size: usize) -> usize {
        let out_spatial = (spatial_size / self.stride).max(1);
        let mut flops = 0usize;
        // Expansion 1x1
        flops += 2 * self.in_channels * self.expanded_channels * out_spatial;
        // Depthwise conv
        flops += 2 * self.expanded_channels * self.kernel_size * out_spatial;
        // SE
        flops += 2 * self.expanded_channels * self.se.reduced_dim * 2;
        // Projection 1x1
        flops += 2 * self.expanded_channels * self.out_channels * out_spatial;
        flops
    }

    /// Parameter count.
    pub fn param_count(&self) -> usize {
        self.expand_weights.len()
            + self.expand_bias.len()
            + self.dw_weights.len()
            + self.dw_bias.len()
            + self.se.fc1_weights.len()
            + self.se.fc1_bias.len()
            + self.se.fc2_weights.len()
            + self.se.fc2_bias.len()
            + self.project_weights.len()
            + self.project_bias.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// S7: ArchMorphism — Function-preserving architecture transformations
// ─────────────────────────────────────────────────────────────────────────────

/// Description of a layer for architecture morphism.
#[derive(Debug, Clone)]
pub struct AdLayerDesc {
    /// Layer index.
    pub index: usize,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
    /// Weight matrix (row-major, out_dim x in_dim).
    pub weights: Vec<f32>,
    /// Bias vector.
    pub bias: Vec<f32>,
}

/// Architecture morphism operations that preserve network function.
///
/// Implements Net2WiderNet (Chen et al., 2015) and Net2DeeperNet for
/// function-preserving transformations of neural networks.
#[derive(Debug, Clone)]
pub struct ArchMorphism {
    /// Current layer descriptions.
    pub layers: Vec<AdLayerDesc>,
}

impl ArchMorphism {
    /// Create from a list of layer descriptions.
    pub fn new(layers: Vec<AdLayerDesc>) -> Result<Self> {
        if layers.is_empty() {
            return Err(TensorError::invalid_argument(
                "ArchMorphism: need at least one layer".into(),
            ));
        }
        // Validate layer connectivity
        for i in 1..layers.len() {
            if layers[i].in_dim != layers[i - 1].out_dim {
                return Err(TensorError::invalid_argument(format!(
                    "ArchMorphism: layer {} in_dim ({}) != layer {} out_dim ({})",
                    i,
                    layers[i].in_dim,
                    i - 1,
                    layers[i - 1].out_dim
                )));
            }
        }
        Ok(Self { layers })
    }

    /// Net2WiderNet: widen a layer by adding neurons/channels.
    ///
    /// Widens layer at `layer_idx` to `new_width` by randomly replicating
    /// existing neurons. The next layer's input weights are adjusted so
    /// the output function is preserved.
    pub fn widen(&mut self, layer_idx: usize, new_width: usize, seed: u64) -> Result<()> {
        if layer_idx >= self.layers.len() {
            return Err(TensorError::invalid_argument(format!(
                "ArchMorphism::widen: layer_idx {} out of range [0, {})",
                layer_idx,
                self.layers.len()
            )));
        }
        let old_width = self.layers[layer_idx].out_dim;
        if new_width <= old_width {
            return Err(TensorError::invalid_argument(format!(
                "ArchMorphism::widen: new_width {} must be > old_width {}",
                new_width, old_width
            )));
        }
        if layer_idx + 1 >= self.layers.len() {
            return Err(TensorError::invalid_argument(
                "ArchMorphism::widen: cannot widen last layer (no next layer to adjust)".into(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let in_dim = self.layers[layer_idx].in_dim;

        // Build mapping: for new neurons, randomly pick which old neuron to copy
        let mut mapping = Vec::with_capacity(new_width);
        for i in 0..old_width {
            mapping.push(i);
        }
        for _ in old_width..new_width {
            mapping.push(rng.random_range(0..old_width));
        }

        // Count how many times each old neuron is replicated
        let mut counts = vec![0usize; old_width];
        for &m in &mapping {
            counts[m] += 1;
        }

        // Widen current layer: new weights [new_width x in_dim]
        let mut new_weights = Vec::with_capacity(new_width * in_dim);
        let mut new_bias = Vec::with_capacity(new_width);
        for &src in &mapping {
            for j in 0..in_dim {
                new_weights.push(self.layers[layer_idx].weights[src * in_dim + j]);
            }
            new_bias.push(self.layers[layer_idx].bias[src]);
        }

        self.layers[layer_idx].weights = new_weights;
        self.layers[layer_idx].bias = new_bias;
        self.layers[layer_idx].out_dim = new_width;

        // Adjust next layer: scale incoming weights by 1/count
        let next = &mut self.layers[layer_idx + 1];
        let next_out = next.out_dim;
        let mut next_weights = Vec::with_capacity(next_out * new_width);
        for i in 0..next_out {
            for &src in &mapping {
                let scale = 1.0 / counts[src] as f32;
                next_weights.push(next.weights[i * old_width + src] * scale);
            }
        }
        next.weights = next_weights;
        next.in_dim = new_width;

        Ok(())
    }

    /// Net2DeeperNet: insert an identity layer after `layer_idx`.
    ///
    /// The new layer is initialized as an identity transformation,
    /// preserving the network function exactly.
    pub fn deepen(&mut self, layer_idx: usize) -> Result<()> {
        if layer_idx >= self.layers.len() {
            return Err(TensorError::invalid_argument(format!(
                "ArchMorphism::deepen: layer_idx {} out of range",
                layer_idx
            )));
        }

        let dim = self.layers[layer_idx].out_dim;

        // Create identity weights [dim x dim]
        let mut id_weights = vec![0.0f32; dim * dim];
        for i in 0..dim {
            id_weights[i * dim + i] = 1.0;
        }
        let id_bias = vec![0.0; dim];

        let new_layer = AdLayerDesc {
            index: layer_idx + 1,
            in_dim: dim,
            out_dim: dim,
            weights: id_weights,
            bias: id_bias,
        };

        // Insert after layer_idx
        self.layers.insert(layer_idx + 1, new_layer);

        // Reindex
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.index = i;
        }

        Ok(())
    }

    /// Add a skip connection by initializing a residual path with small weight.
    ///
    /// Creates an additional connection from `from_idx` to `to_idx` with
    /// initial weight `epsilon` (small, so the function is approximately preserved).
    pub fn add_skip(
        &self,
        from_idx: usize,
        to_idx: usize,
        epsilon: f32,
    ) -> Result<AdSkipConnection> {
        if from_idx >= self.layers.len() || to_idx >= self.layers.len() {
            return Err(TensorError::invalid_argument(format!(
                "ArchMorphism::add_skip: indices ({}, {}) out of range [0, {})",
                from_idx,
                to_idx,
                self.layers.len()
            )));
        }
        if from_idx >= to_idx {
            return Err(TensorError::invalid_argument(
                "ArchMorphism::add_skip: from_idx must be < to_idx".into(),
            ));
        }

        let from_dim = self.layers[from_idx].out_dim;
        let to_dim = self.layers[to_idx].in_dim;

        // Projection weights if dimensions differ, otherwise identity * epsilon
        let proj_dim = from_dim.min(to_dim);
        let mut proj_weights = vec![0.0f32; to_dim * from_dim];
        for i in 0..proj_dim {
            proj_weights[i * from_dim + i] = epsilon;
        }

        Ok(AdSkipConnection {
            from_layer: from_idx,
            to_layer: to_idx,
            projection: proj_weights,
            from_dim,
            to_dim,
            weight: epsilon,
        })
    }

    /// Forward pass through morphed architecture.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if self.layers.is_empty() {
            return Err(TensorError::invalid_argument(
                "ArchMorphism::forward: no layers".into(),
            ));
        }
        if input.len() != self.layers[0].in_dim {
            return Err(TensorError::invalid_argument(format!(
                "ArchMorphism::forward: expected input len {}, got {}",
                self.layers[0].in_dim,
                input.len()
            )));
        }

        let n_layers = self.layers.len();
        let mut h = input.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            h = ad_matvec(&layer.weights, &layer.bias, &h, layer.out_dim, layer.in_dim);
            // ReLU on hidden layers
            if i < n_layers - 1 {
                h.iter_mut().for_each(|v| *v = ad_relu(*v));
            }
        }
        Ok(h)
    }

    /// Total parameter count across all layers.
    pub fn total_params(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.len() + l.bias.len())
            .sum()
    }

    /// Number of layers.
    pub fn depth(&self) -> usize {
        self.layers.len()
    }
}

/// A skip connection between two layers.
#[derive(Debug, Clone)]
pub struct AdSkipConnection {
    /// Source layer index.
    pub from_layer: usize,
    /// Target layer index.
    pub to_layer: usize,
    /// Projection weights [to_dim x from_dim].
    pub projection: Vec<f32>,
    /// Source dimension.
    pub from_dim: usize,
    /// Target dimension.
    pub to_dim: usize,
    /// Initial weight of the skip connection.
    pub weight: f32,
}

impl AdSkipConnection {
    /// Apply the skip connection: project `source_activation` and add to `target`.
    pub fn apply(&self, source: &[f32], target: &mut [f32]) -> Result<()> {
        if source.len() != self.from_dim {
            return Err(TensorError::invalid_argument(format!(
                "AdSkipConnection: source len {} != from_dim {}",
                source.len(),
                self.from_dim
            )));
        }
        if target.len() != self.to_dim {
            return Err(TensorError::invalid_argument(format!(
                "AdSkipConnection: target len {} != to_dim {}",
                target.len(),
                self.to_dim
            )));
        }

        // target += projection @ source
        for i in 0..self.to_dim {
            let mut val = 0.0f32;
            for j in 0..self.from_dim {
                val += self.projection[i * self.from_dim + j] * source[j];
            }
            target[i] += val;
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// S8: ProgressiveShrinking — OFA-style elastic supernet
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a sub-network sampled from the supernet.
#[derive(Debug, Clone)]
pub struct AdSubnetConfig {
    /// Depth for each stage (number of blocks per stage).
    pub depths: Vec<usize>,
    /// Width multiplier for each stage.
    pub widths: Vec<usize>,
    /// Kernel sizes for each block.
    pub kernel_sizes: Vec<usize>,
    /// Estimated FLOPs.
    pub estimated_flops: usize,
}

/// Progressive shrinking (OFA — Once-For-All) style elastic supernet training.
///
/// Trains a single supernet that can be sliced into many sub-networks of varying
/// depth, width, and kernel size. Sub-networks inherit weights from the supernet.
#[derive(Debug, Clone)]
pub struct ProgressiveShrinking {
    /// Maximum depth per stage.
    pub max_depths: Vec<usize>,
    /// Maximum width per stage.
    pub max_widths: Vec<usize>,
    /// Allowed kernel sizes (e.g., [3, 5, 7]).
    pub kernel_choices: Vec<usize>,
    /// Number of stages.
    pub n_stages: usize,
    /// Supernet weights: one per stage, each is [max_width x max_width x max_kernel].
    pub supernet_weights: Vec<Vec<f32>>,
    /// Current training phase (0=elastic kernel, 1=elastic depth, 2=elastic width).
    pub phase: usize,
}

impl ProgressiveShrinking {
    /// Create a new progressive shrinking supernet.
    pub fn new(
        max_depths: Vec<usize>,
        max_widths: Vec<usize>,
        kernel_choices: Vec<usize>,
        seed: u64,
    ) -> Result<Self> {
        if max_depths.is_empty() || max_widths.is_empty() || kernel_choices.is_empty() {
            return Err(TensorError::invalid_argument(
                "ProgressiveShrinking: all config vectors must be non-empty".into(),
            ));
        }
        if max_depths.len() != max_widths.len() {
            return Err(TensorError::invalid_argument(
                "ProgressiveShrinking: max_depths and max_widths must have same length".into(),
            ));
        }

        let n_stages = max_depths.len();
        let mut rng = StdRng::seed_from_u64(seed);
        let max_k = kernel_choices.iter().copied().max().unwrap_or(7);

        let mut supernet_weights = Vec::with_capacity(n_stages);
        for i in 0..n_stages {
            let w = max_widths[i];
            let total = max_depths[i] * w * w * max_k;
            let bound = ad_xavier_bound(w, w);
            let weights: Vec<f32> = (0..total)
                .map(|_| rng.random_range(-bound..bound))
                .collect();
            let _ = i; // stage index used for seed variation
            supernet_weights.push(weights);
        }

        Ok(Self {
            max_depths,
            max_widths,
            kernel_choices,
            n_stages,
            supernet_weights,
            phase: 0,
        })
    }

    /// Sample a random sub-network configuration.
    pub fn sample_subnet(&self, seed: u64) -> Result<AdSubnetConfig> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut depths = Vec::with_capacity(self.n_stages);
        let mut widths = Vec::with_capacity(self.n_stages);
        let mut kernel_sizes = Vec::new();

        for i in 0..self.n_stages {
            // Sample depth: [max_depth/2, max_depth] based on phase
            let min_d = match self.phase {
                0 | 1 => self.max_depths[i], // Full depth initially
                _ => (self.max_depths[i] + 1) / 2,
            };
            let d = if min_d >= self.max_depths[i] {
                self.max_depths[i]
            } else {
                rng.random_range(min_d..=self.max_depths[i])
            };
            depths.push(d);

            // Sample width
            let min_w = match self.phase {
                0..=2 => self.max_widths[i],           // Full width initially
                _ => (self.max_widths[i] * 3 + 3) / 4, // 75% of max
            };
            let w = if min_w >= self.max_widths[i] {
                self.max_widths[i]
            } else {
                rng.random_range(min_w..=self.max_widths[i])
            };
            widths.push(w);

            // Sample kernel sizes for each block
            for _ in 0..d {
                let k_idx = rng.random_range(0..self.kernel_choices.len());
                kernel_sizes.push(self.kernel_choices[k_idx]);
            }
        }

        // Estimate FLOPs
        let mut flops = 0usize;
        for i in 0..self.n_stages {
            let w = widths[i];
            flops += depths[i] * 2 * w * w; // Simplified FLOPs estimate
        }

        Ok(AdSubnetConfig {
            depths,
            widths,
            kernel_sizes,
            estimated_flops: flops,
        })
    }

    /// Extract subnet weights from the supernet via slicing.
    ///
    /// Returns sliced weights for each stage of the subnet.
    pub fn extract_weights(&self, config: &AdSubnetConfig) -> Result<Vec<Vec<f32>>> {
        if config.depths.len() != self.n_stages || config.widths.len() != self.n_stages {
            return Err(TensorError::invalid_argument(
                "ProgressiveShrinking::extract_weights: config dimensions mismatch".into(),
            ));
        }

        let max_k = self.kernel_choices.iter().copied().max().unwrap_or(7);
        let mut result = Vec::with_capacity(self.n_stages);

        for i in 0..self.n_stages {
            let mw = self.max_widths[i];
            let sw = config.widths[i];
            let sd = config.depths[i];
            let stage_weights = &self.supernet_weights[i];

            // Slice: take first sw channels out of mw for each block
            let block_size = mw * mw * max_k;
            let mut sliced = Vec::new();
            for b in 0..sd {
                let block_start = b * block_size;
                // Extract [sw x sw x max_k] from [mw x mw x max_k]
                for row in 0..sw {
                    let row_start = block_start + row * mw * max_k;
                    for col in 0..sw {
                        let col_start = row_start + col * max_k;
                        for k in 0..max_k {
                            if col_start + k < stage_weights.len() {
                                sliced.push(stage_weights[col_start + k]);
                            } else {
                                sliced.push(0.0);
                            }
                        }
                    }
                }
            }
            result.push(sliced);
        }

        Ok(result)
    }

    /// Advance to the next training phase.
    pub fn advance_phase(&mut self) {
        if self.phase < 3 {
            self.phase += 1;
        }
    }

    /// Get the current training phase name.
    pub fn phase_name(&self) -> &str {
        match self.phase {
            0 => "elastic_kernel",
            1 => "elastic_depth",
            2 => "elastic_width",
            _ => "full_elastic",
        }
    }

    /// Total supernet parameter count.
    pub fn supernet_params(&self) -> usize {
        self.supernet_weights.iter().map(|w| w.len()).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// S10: ArchDistiller — Joint architecture search + knowledge distillation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for architecture distillation search.
#[derive(Debug, Clone)]
pub struct AdSearchConfig {
    /// Number of candidate architectures to evaluate per round.
    pub population_size: usize,
    /// Number of search rounds.
    pub n_rounds: usize,
    /// FLOPs budget constraint.
    pub flops_budget: usize,
    /// Temperature for knowledge distillation.
    pub temperature: f32,
    /// Alpha: weight of hard label loss (1-alpha = soft label weight).
    pub alpha: f32,
    /// Learning rate for student training.
    pub lr: f32,
    /// Architecture encoding parameters.
    pub n_nodes: usize,
    /// Number of cells.
    pub n_cells: usize,
}

impl AdSearchConfig {
    /// Create a default search configuration.
    pub fn default_config() -> Self {
        Self {
            population_size: 20,
            n_rounds: 10,
            flops_budget: 100_000_000,
            temperature: 4.0,
            alpha: 0.5,
            lr: 0.01,
            n_nodes: 4,
            n_cells: 8,
        }
    }
}

/// Result of architecture distillation search.
#[derive(Debug, Clone)]
pub struct AdSearchResult {
    /// Best architecture found.
    pub best_arch: AdArchEncoding,
    /// Predicted accuracy of best architecture.
    pub best_accuracy: f32,
    /// FLOPs of best architecture.
    pub best_flops: usize,
    /// Number of architectures evaluated.
    pub total_evaluated: usize,
    /// Search history: (round, best_accuracy).
    pub history: Vec<(usize, f32)>,
}

/// Joint architecture search + knowledge distillation.
///
/// Searches for a student architecture that maximizes accuracy under a FLOPs
/// constraint, using teacher soft targets for guidance.
///
/// The combined loss is: `alpha * CE(student, hard) + (1-alpha) * KL(student, teacher)`.
#[derive(Debug, Clone)]
pub struct ArchDistiller {
    /// Search configuration.
    pub config: AdSearchConfig,
    /// Neural predictor for cheap ranking.
    pub predictor: NeuralPredictor,
    /// Proxy task for evaluation.
    pub proxy: ProxyTask,
}

impl ArchDistiller {
    /// Create a new architecture distiller.
    pub fn new(config: AdSearchConfig, predictor_seed: u64) -> Result<Self> {
        if config.population_size == 0 {
            return Err(TensorError::invalid_argument(
                "ArchDistiller: population_size must be > 0".into(),
            ));
        }
        if config.n_rounds == 0 {
            return Err(TensorError::invalid_argument(
                "ArchDistiller: n_rounds must be > 0".into(),
            ));
        }
        if config.temperature <= 0.0 {
            return Err(TensorError::invalid_argument(
                "ArchDistiller: temperature must be > 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&config.alpha) {
            return Err(TensorError::invalid_argument(
                "ArchDistiller: alpha must be in [0, 1]".into(),
            ));
        }

        // Encoding dim for 4-node architecture: 4 edges * 15 + 3 global = 63
        let encoding_dim = config.n_nodes * (2 * AdOpType::count() + 1) + 3;
        let predictor = NeuralPredictor::new(encoding_dim, 64, predictor_seed)?;
        let proxy = ProxyTask::new(5, 0.1, 0.5, 0.5)?;

        Ok(Self {
            config,
            predictor,
            proxy,
        })
    }

    /// Compute the combined distillation loss.
    ///
    /// `student_logits`: raw student output logits.
    /// `teacher_logits`: raw teacher output logits.
    /// `hard_labels`: one-hot ground truth labels.
    pub fn distillation_loss(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
        hard_labels: &[f32],
    ) -> Result<f32> {
        let n = student_logits.len();
        if teacher_logits.len() != n || hard_labels.len() != n {
            return Err(TensorError::invalid_argument(
                "ArchDistiller::distillation_loss: all inputs must have same length".into(),
            ));
        }

        // Hard label loss: CE(softmax(student), hard_labels)
        let student_probs = ad_softmax_temp(student_logits, 1.0);
        let hard_loss = ad_cross_entropy(hard_labels, &student_probs);

        // Soft label loss: KL(softmax(teacher/T), softmax(student/T)) * T^2
        let teacher_soft = ad_softmax_temp(teacher_logits, self.config.temperature);
        let student_soft = ad_softmax_temp(student_logits, self.config.temperature);
        let soft_loss = ad_kl_div(&teacher_soft, &student_soft)
            * self.config.temperature
            * self.config.temperature;

        let alpha = self.config.alpha;
        Ok(alpha * hard_loss + (1.0 - alpha) * soft_loss)
    }

    /// Search for the best student architecture.
    ///
    /// `teacher_logits`: teacher model outputs for training samples.
    /// `train_data`: (input, hard_label) pairs.
    pub fn search(
        &mut self,
        teacher_logits: &[Vec<f32>],
        train_data: &[(Vec<f32>, Vec<f32>)],
        seed: u64,
    ) -> Result<AdSearchResult> {
        if teacher_logits.is_empty() || train_data.is_empty() {
            return Err(TensorError::invalid_argument(
                "ArchDistiller::search: need non-empty teacher_logits and train_data".into(),
            ));
        }
        if teacher_logits.len() != train_data.len() {
            return Err(TensorError::invalid_argument(
                "ArchDistiller::search: teacher_logits and train_data must have same length".into(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let mut best_arch: Option<AdArchEncoding> = None;
        let mut best_accuracy = f32::NEG_INFINITY;
        let mut best_flops = 0usize;
        let mut total_evaluated = 0usize;
        let mut history = Vec::new();

        // Collect training data for predictor
        let mut predictor_encs = Vec::new();
        let mut predictor_accs = Vec::new();

        for round in 0..self.config.n_rounds {
            let mut round_best = f32::NEG_INFINITY;

            for _ in 0..self.config.population_size {
                let arch_seed = rng.random_range(0..u64::MAX);
                // Sample candidate widths in [16, 128]
                let channels = (rng.random_range(2..16_u32) * 8) as usize;

                let arch = AdArchEncoding::random(
                    self.config.n_nodes,
                    channels,
                    self.config.n_cells,
                    arch_seed,
                )?;

                // Check FLOPs constraint
                let flops = arch.estimate_flops(32);
                if flops > self.config.flops_budget {
                    total_evaluated += 1;
                    continue;
                }

                // Evaluate via proxy
                let proxy_result =
                    self.proxy
                        .evaluate_proxy(&arch, train_data, Some(3), Some(0.1))?;

                // Compute distillation-aware score
                let n_cls = teacher_logits.first().map(|v| v.len()).unwrap_or(2);
                let n_eval = train_data.len().min(10);
                let mut distil_score = 0.0f32;
                for i in 0..n_eval {
                    // Simple student "forward": project input to class logits
                    let student_logits: Vec<f32> = (0..n_cls)
                        .map(|c| {
                            let mut val = 0.0f32;
                            for (j, &x) in train_data[i]
                                .0
                                .iter()
                                .enumerate()
                                .take(channels.min(train_data[i].0.len()))
                            {
                                val += x * ((c * channels + j) as f32 * 0.01);
                            }
                            val
                        })
                        .collect();

                    let loss = self.distillation_loss(
                        &student_logits,
                        &teacher_logits[i],
                        &train_data[i].1,
                    )?;
                    distil_score += 1.0 / (1.0 + loss); // Convert loss to score
                }
                let distil_score = distil_score / n_eval as f32;

                // Combined score
                let combined = 0.6 * proxy_result.score + 0.4 * distil_score;

                predictor_encs.push(arch.encode());
                predictor_accs.push(combined);

                if combined > best_accuracy {
                    best_accuracy = combined;
                    best_arch = Some(arch);
                    best_flops = flops;
                }
                if combined > round_best {
                    round_best = combined;
                }

                total_evaluated += 1;
            }

            // Train predictor periodically
            if !predictor_encs.is_empty() && round % 3 == 2 {
                let _ = self.predictor.train(&predictor_encs, &predictor_accs, 10);
            }

            history.push((round, best_accuracy));
        }

        let final_arch = best_arch.ok_or_else(|| {
            TensorError::compute_error_simple(
                "ArchDistiller::search: no valid architecture found within FLOPs budget".into(),
            )
        })?;

        Ok(AdSearchResult {
            best_arch: final_arch,
            best_accuracy,
            best_flops,
            total_evaluated,
            history,
        })
    }
}
