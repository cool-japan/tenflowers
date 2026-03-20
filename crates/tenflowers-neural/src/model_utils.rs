//! Model statistics and utility functions for analyzing neural network architectures.
//!
//! This module provides:
//! - [`LayerStats`]: Statistics about a single layer's parameters and FLOPs
//! - [`ModelStats`]: Aggregated statistics over all layers in a model
//! - FLOPs estimation functions for dense, conv1d, and attention layers
//! - [`dense_model_stats`]: Convenience builder for dense-only models

use tenflowers_core::TensorError;

/// Error type for model utility operations
#[derive(Debug, thiserror::Error)]
#[allow(clippy::large_enum_variant)]
pub enum ModelUtilsError {
    /// Returned when a layer spec is invalid (e.g. zero dimensions)
    #[error("invalid layer spec: {0}")]
    InvalidLayerSpec(String),
    /// Propagated from tensor operations
    #[error("tensor error: {0}")]
    TensorError(#[from] TensorError),
}

/// Result alias for model utility operations
pub type Result<T> = std::result::Result<T, ModelUtilsError>;

// ---------------------------------------------------------------------------
// LayerStats
// ---------------------------------------------------------------------------

/// Statistics about a single layer's parameters and compute cost.
///
/// `weight_shape` stores the primary weight tensor dimensions.
/// `flops_per_sample` is `None` when the FLOPs cannot be estimated statically
/// (e.g. recurrent layers whose cost depends on sequence length).
#[derive(Debug, Clone)]
pub struct LayerStats {
    /// Human-readable layer name (e.g. `"fc1"`, `"attn"`)
    pub name: String,
    /// Total number of scalar parameters in this layer
    pub param_count: usize,
    /// Shape of the primary weight tensor
    pub weight_shape: Vec<usize>,
    /// Estimated multiply-add operations per sample (optional)
    pub flops_per_sample: Option<u64>,
}

impl LayerStats {
    /// Create a new [`LayerStats`] with all fields specified.
    pub fn new(
        name: impl Into<String>,
        param_count: usize,
        weight_shape: Vec<usize>,
        flops_per_sample: Option<u64>,
    ) -> Self {
        Self {
            name: name.into(),
            param_count,
            weight_shape,
            flops_per_sample,
        }
    }
}

// ---------------------------------------------------------------------------
// ModelStats
// ---------------------------------------------------------------------------

/// Aggregated statistics over an entire model.
///
/// `size_bytes` assumes every parameter is stored as a 32-bit float (4 bytes).
#[derive(Debug, Clone)]
pub struct ModelStats {
    /// Sum of all layer parameter counts
    pub total_params: usize,
    /// Number of trainable parameters (equals `total_params` unless manually
    /// adjusted; this field exists so callers can record frozen layers)
    pub trainable_params: usize,
    /// Sum of per-layer `flops_per_sample`; `None` if any layer is unknown
    pub total_flops_per_sample: Option<u64>,
    /// Per-layer statistics in insertion order
    pub layers: Vec<LayerStats>,
    /// Estimated storage size: `total_params * 4` bytes (f32)
    pub size_bytes: usize,
}

impl ModelStats {
    /// Create an empty [`ModelStats`].
    pub fn new() -> Self {
        Self {
            total_params: 0,
            trainable_params: 0,
            total_flops_per_sample: Some(0),
            layers: Vec::new(),
            size_bytes: 0,
        }
    }

    /// Append a layer and update the running totals.
    ///
    /// If `stats.flops_per_sample` is `None`, `total_flops_per_sample` becomes
    /// `None` for the entire model (conservative: unknown FLOPs propagate).
    pub fn add_layer(&mut self, stats: LayerStats) {
        self.total_params += stats.param_count;
        self.trainable_params += stats.param_count;
        self.size_bytes += stats.param_count * 4;

        self.total_flops_per_sample = match (self.total_flops_per_sample, stats.flops_per_sample) {
            (Some(acc), Some(f)) => Some(acc + f),
            _ => None,
        };

        self.layers.push(stats);
    }

    /// Return `size_bytes` expressed in mebibytes (MiB).
    ///
    /// ```
    /// use tenflowers_neural::model_utils::ModelStats;
    /// let mut s = ModelStats::new();
    /// // 1 MiB = 1024 * 1024 bytes = 262144 f32 params
    /// // so size_mb of 262144 params ≈ 1.0
    /// ```
    pub fn size_mb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Build a multi-line summary string resembling PyTorch's `model.summary()`.
    ///
    /// Each row shows: layer name, weight shape, parameter count, and optional
    /// FLOPs. Footer lines report totals.
    pub fn summary(&self) -> String {
        let sep = "=".repeat(72);
        let thin = "-".repeat(72);
        let mut out = String::new();

        out.push_str(&sep);
        out.push('\n');
        out.push_str(&format!(
            "{:<24} {:<20} {:>10}  {:>12}\n",
            "Layer", "Weight Shape", "Params", "FLOPs/sample"
        ));
        out.push_str(&thin);
        out.push('\n');

        for layer in &self.layers {
            let shape_str = format!("{:?}", layer.weight_shape);
            let flops_str = match layer.flops_per_sample {
                Some(f) => format_number_u64(f),
                None => "?".to_string(),
            };
            out.push_str(&format!(
                "{:<24} {:<20} {:>10}  {:>12}\n",
                layer.name,
                shape_str,
                format_number(layer.param_count),
                flops_str,
            ));
        }

        out.push_str(&sep);
        out.push('\n');
        out.push_str(&format!(
            "Total params       : {}\n",
            format_number(self.total_params)
        ));
        out.push_str(&format!(
            "Trainable params   : {}\n",
            format_number(self.trainable_params)
        ));
        let flops_total = match self.total_flops_per_sample {
            Some(f) => format_number_u64(f),
            None => "?".to_string(),
        };
        out.push_str(&format!("FLOPs/sample       : {}\n", flops_total));
        out.push_str(&format!("Model size         : {:.3} MB\n", self.size_mb()));

        out
    }
}

impl Default for ModelStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Parameter-count helpers
// ---------------------------------------------------------------------------

/// Count parameters in a dense (fully-connected) layer.
///
/// `weight` contributes `input_dim * output_dim` scalars; a bias vector
/// contributes `output_dim` additional scalars when `has_bias` is `true`.
///
/// # Example
/// ```
/// use tenflowers_neural::model_utils::dense_param_count;
/// assert_eq!(dense_param_count(784, 128, true), 784 * 128 + 128);
/// assert_eq!(dense_param_count(128,  10, false), 128 * 10);
/// ```
pub fn dense_param_count(input_dim: usize, output_dim: usize, has_bias: bool) -> usize {
    let weight = input_dim * output_dim;
    if has_bias {
        weight + output_dim
    } else {
        weight
    }
}

// ---------------------------------------------------------------------------
// FLOPs estimation helpers
// ---------------------------------------------------------------------------

/// Estimate multiply-add operations for a single dense-layer forward pass.
///
/// The dominant cost is a matrix-vector product: `input_dim` multiply-adds per
/// output neuron, giving `input_dim * output_dim` MACs in total.  Each bias
/// addition counts as one additional FLOPs per output neuron.
///
/// Following the widely-used convention where one MAC = 2 FLOPs:
///   - MACs = `input_dim * output_dim`
///   - FLOPs = `2 * MACs + (output_dim if has_bias else 0)`
///
/// # Example
/// ```
/// use tenflowers_neural::model_utils::dense_flops;
/// // 3 in, 2 out, no bias: 2 * 3 * 2 = 12
/// assert_eq!(dense_flops(3, 2, false), 12);
/// // 3 in, 2 out, with bias: 12 + 2 = 14
/// assert_eq!(dense_flops(3, 2, true), 14);
/// ```
pub fn dense_flops(input_dim: usize, output_dim: usize, has_bias: bool) -> u64 {
    let macs = (input_dim * output_dim) as u64;
    let bias_ops = if has_bias { output_dim as u64 } else { 0 };
    2 * macs + bias_ops
}

/// Estimate multiply-add operations for a 1-D convolution forward pass.
///
/// For each output position (there are `input_len` positions assuming same
/// padding / unit stride) and each output channel, the kernel sweeps
/// `in_channels * kernel_size` multiply-adds, giving:
///
///   `FLOPs = 2 * input_len * out_channels * in_channels * kernel_size`
///
/// # Example
/// ```
/// use tenflowers_neural::model_utils::conv1d_flops;
/// // length=10, in_ch=3, out_ch=8, kernel=5
/// assert_eq!(conv1d_flops(10, 3, 8, 5), 2 * 10 * 3 * 8 * 5);
/// ```
pub fn conv1d_flops(
    input_len: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
) -> u64 {
    2 * (input_len as u64) * (in_channels as u64) * (out_channels as u64) * (kernel_size as u64)
}

/// Estimate multiply-add operations for a multi-head self-attention layer.
///
/// The standard self-attention complexity is O(n² · d) where:
/// - n = `seq_len`
/// - d = `model_dim`
///
/// Concretely, for each of `num_heads` heads with head-dim `d_h = model_dim /
/// num_heads`:
/// - Q, K, V projections : 3 × 2 × n × d × d_h  MACs
/// - QKᵀ (score matrix)  : 2 × n² × d_h           MACs per head
/// - softmax × V          : 2 × n² × d_h           MACs per head
/// - output projection    : 2 × n × d × d           MACs
///
/// This function returns an approximation that captures the dominant n² term:
///
///   `FLOPs ≈ 4 * 2 * seq_len² * model_dim + 2 * 3 * seq_len * model_dim²`
///
/// The `num_heads` argument affects head dimension but cancels in the final
/// formula when d is divisible by h, so this function keeps the closed-form
/// expression without per-head loops.
///
/// # Example
/// ```
/// use tenflowers_neural::model_utils::attention_flops;
/// // When seq_len dominates model_dim, doubling seq_len more than doubles cost.
/// let f1 = attention_flops(128, 32, 4);
/// let f2 = attention_flops(256, 32, 4);
/// assert!(f2 > 3 * f1, "attention FLOPs should grow super-linearly with seq_len");
/// ```
pub fn attention_flops(seq_len: usize, model_dim: usize, num_heads: usize) -> u64 {
    let n = seq_len as u64;
    let d = model_dim as u64;
    // head dimension
    let d_h = if num_heads == 0 {
        d
    } else {
        d / (num_heads as u64).max(1)
    };

    // Q, K, V projections: 3 heads * 2 * n * d * d_h (but d_h * h = d, so 3 * 2 * n * d^2 / h * h = 3 * 2 * n * d * d_h * h = 3*2*n*d*d)
    // Actually: each projection is (n, d) x (d, d_h) = 2 * n * d * d_h MACs per head, times h heads = 2 * n * d * d
    let qkv_proj = 3 * 2 * n * d * d_h * (num_heads as u64).max(1);

    // score matrix QKᵀ: per head (n, d_h) x (d_h, n) = 2 * n^2 * d_h, times h heads = 2 * n^2 * d
    let scores = 2 * n * n * d_h * (num_heads as u64).max(1);

    // softmax_scores x V: per head (n, n) x (n, d_h) = 2 * n^2 * d_h, times h heads = 2 * n^2 * d
    let sv = 2 * n * n * d_h * (num_heads as u64).max(1);

    // output projection: (n, d) x (d, d) = 2 * n * d^2
    let out_proj = 2 * n * d * d;

    qkv_proj + scores + sv + out_proj
}

// ---------------------------------------------------------------------------
// High-level builder
// ---------------------------------------------------------------------------

/// Build a [`ModelStats`] from a slice of `(name, input_dim, output_dim)` dense
/// layer specifications.
///
/// Bias is assumed to be present for every layer.  FLOPs are estimated with
/// [`dense_flops`].
///
/// # Example
/// ```
/// use tenflowers_neural::model_utils::dense_model_stats;
/// let stats = dense_model_stats(&[("fc1", 784, 128), ("fc2", 128, 10)]);
/// assert_eq!(stats.total_params, 784 * 128 + 128 + 128 * 10 + 10);
/// assert_eq!(stats.layers.len(), 2);
/// ```
pub fn dense_model_stats(layers: &[(&str, usize, usize)]) -> ModelStats {
    let mut ms = ModelStats::new();
    for &(name, in_dim, out_dim) in layers {
        let param_count = dense_param_count(in_dim, out_dim, true);
        let flops = dense_flops(in_dim, out_dim, true);
        let stats = LayerStats::new(name, param_count, vec![in_dim, out_dim], Some(flops));
        ms.add_layer(stats);
    }
    ms
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn format_number_u64(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- dense_param_count ---------------------------------------------------

    #[test]
    fn test_dense_param_count_with_bias() {
        // weight: 784*128, bias: 128
        assert_eq!(dense_param_count(784, 128, true), 784 * 128 + 128);
    }

    #[test]
    fn test_dense_param_count_without_bias() {
        assert_eq!(dense_param_count(128, 10, false), 128 * 10);
    }

    #[test]
    fn test_dense_param_count_single_neuron() {
        // 1 input, 1 output, with bias => 2 params
        assert_eq!(dense_param_count(1, 1, true), 2);
    }

    #[test]
    fn test_dense_param_count_zero_bias() {
        // no bias: 3 * 5 = 15
        assert_eq!(dense_param_count(3, 5, false), 15);
    }

    // --- dense_flops ---------------------------------------------------------

    #[test]
    fn test_dense_flops_no_bias() {
        // 2 * 3 * 2 = 12
        assert_eq!(dense_flops(3, 2, false), 12);
    }

    #[test]
    fn test_dense_flops_with_bias() {
        // 2 * 3 * 2 + 2 = 14
        assert_eq!(dense_flops(3, 2, true), 14);
    }

    #[test]
    fn test_dense_flops_scales_linearly_with_output() {
        let f1 = dense_flops(100, 10, false);
        let f2 = dense_flops(100, 20, false);
        assert_eq!(f2, 2 * f1);
    }

    #[test]
    fn test_dense_flops_large_layer() {
        // 768 * 3072 * 2 = 4718592
        assert_eq!(dense_flops(768, 3072, false), 2 * 768 * 3072);
    }

    // --- attention_flops -----------------------------------------------------

    #[test]
    fn test_attention_flops_quadratic_in_seq_len() {
        // Use large seq_len relative to model_dim so the O(n^2·d) term dominates.
        // With seq_len=512 vs 1024 (doubling) and model_dim=64 / heads=8,
        // the quadratic term (2*n^2*d per scores+sv) dominates linear terms,
        // so cost should grow more than 3× when seq_len doubles.
        let f1 = attention_flops(128, 32, 4);
        let f2 = attention_flops(256, 32, 4);
        // doubling seq_len should more than double FLOPs due to n^2 term
        assert!(f2 > 3 * f1, "Expected f2 ({}) > 3*f1 ({})", f2, 3 * f1);
    }

    #[test]
    fn test_attention_flops_scales_with_model_dim() {
        let f1 = attention_flops(16, 64, 8);
        let f2 = attention_flops(16, 128, 8);
        // doubling model_dim increases FLOPs
        assert!(f2 > f1, "Expected f2 ({}) > f1 ({})", f2, f1);
    }

    #[test]
    fn test_attention_flops_nonzero() {
        let f = attention_flops(32, 128, 4);
        assert!(f > 0);
    }

    // --- conv1d_flops --------------------------------------------------------

    #[test]
    fn test_conv1d_flops_formula() {
        // 2 * 10 * 3 * 8 * 5 = 2400
        assert_eq!(conv1d_flops(10, 3, 8, 5), 2400);
    }

    #[test]
    fn test_conv1d_flops_scales_with_length() {
        let f1 = conv1d_flops(10, 4, 4, 3);
        let f2 = conv1d_flops(20, 4, 4, 3);
        assert_eq!(f2, 2 * f1);
    }

    // --- ModelStats ----------------------------------------------------------

    #[test]
    fn test_model_stats_new_empty() {
        let ms = ModelStats::new();
        assert_eq!(ms.total_params, 0);
        assert_eq!(ms.trainable_params, 0);
        assert_eq!(ms.size_bytes, 0);
        assert_eq!(ms.total_flops_per_sample, Some(0));
        assert!(ms.layers.is_empty());
    }

    #[test]
    fn test_model_stats_add_layer_accumulates() {
        let mut ms = ModelStats::new();
        ms.add_layer(LayerStats::new("l1", 100, vec![10, 10], Some(200)));
        ms.add_layer(LayerStats::new("l2", 50, vec![10, 5], Some(100)));

        assert_eq!(ms.total_params, 150);
        assert_eq!(ms.trainable_params, 150);
        assert_eq!(ms.size_bytes, 150 * 4);
        assert_eq!(ms.total_flops_per_sample, Some(300));
        assert_eq!(ms.layers.len(), 2);
    }

    #[test]
    fn test_model_stats_unknown_flops_propagate() {
        let mut ms = ModelStats::new();
        ms.add_layer(LayerStats::new("l1", 100, vec![10, 10], Some(200)));
        ms.add_layer(LayerStats::new("rnn", 64, vec![8, 8], None));
        // Once None appears, total becomes None
        assert_eq!(ms.total_flops_per_sample, None);
    }

    #[test]
    fn test_model_stats_size_mb() {
        let mut ms = ModelStats::new();
        // 1 MiB = 1024 * 1024 bytes; at 4 bytes/param => 262144 params
        let params_for_1mb = 1024 * 1024 / 4;
        ms.add_layer(LayerStats::new(
            "big",
            params_for_1mb,
            vec![params_for_1mb],
            None,
        ));
        let mb = ms.size_mb();
        assert!((mb - 1.0).abs() < 1e-6, "Expected ~1.0 MB, got {}", mb);
    }

    #[test]
    fn test_model_stats_summary_contains_layer_names() {
        let mut ms = ModelStats::new();
        ms.add_layer(LayerStats::new("encoder", 1000, vec![32, 32], Some(2000)));
        ms.add_layer(LayerStats::new("decoder", 500, vec![32, 16], Some(1000)));
        let s = ms.summary();
        assert!(s.contains("encoder"), "summary missing 'encoder'");
        assert!(s.contains("decoder"), "summary missing 'decoder'");
    }

    #[test]
    fn test_model_stats_summary_contains_param_counts() {
        let mut ms = ModelStats::new();
        ms.add_layer(LayerStats::new("layer1", 99999, vec![333, 300], Some(0)));
        let s = ms.summary();
        // Total params line should be present
        assert!(s.contains("Total params"), "summary missing 'Total params'");
    }

    #[test]
    fn test_model_stats_summary_contains_size() {
        let ms = ModelStats::new();
        let s = ms.summary();
        assert!(s.contains("Model size"), "summary missing 'Model size'");
    }

    // --- dense_model_stats ---------------------------------------------------

    #[test]
    fn test_dense_model_stats_single_layer() {
        let ms = dense_model_stats(&[("fc", 10, 5)]);
        assert_eq!(ms.layers.len(), 1);
        // with bias: 10*5 + 5 = 55
        assert_eq!(ms.total_params, dense_param_count(10, 5, true));
        assert_eq!(ms.layers[0].name, "fc");
    }

    #[test]
    fn test_dense_model_stats_two_layers() {
        let ms = dense_model_stats(&[("fc1", 784, 128), ("fc2", 128, 10)]);
        let expected = dense_param_count(784, 128, true) + dense_param_count(128, 10, true);
        assert_eq!(ms.total_params, expected);
        assert_eq!(ms.layers.len(), 2);
        assert_eq!(ms.layers[1].name, "fc2");
    }

    #[test]
    fn test_dense_model_stats_flops_populated() {
        let ms = dense_model_stats(&[("fc1", 100, 50)]);
        assert_eq!(ms.total_flops_per_sample, Some(dense_flops(100, 50, true)));
    }

    #[test]
    fn test_dense_model_stats_weight_shape() {
        let ms = dense_model_stats(&[("fc", 16, 8)]);
        assert_eq!(ms.layers[0].weight_shape, vec![16, 8]);
    }
}
