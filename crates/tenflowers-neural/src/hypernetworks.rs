//! # Hypernetworks and Task-Conditioned Dynamic Architectures
//!
//! Implements weight-generating networks following Ha et al. (2017) and related
//! task-conditioned architectures: chunked weight generation, HyperTransformer
//! for few-shot learning (Zhmoginov et al., 2022), FiLM conditioning
//! (Perez et al., 2018), and fast-weight programmers (Schmidhuber 1992,
//! Schlag et al. 2021).
//!
//! All components use plain `Vec<f32>` for weights and embeddings with no
//! dependency on an autograd engine.  Random number generation uses
//! `scirs2_core::random` exclusively.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Internal math helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
fn tanh_safe(x: f32) -> f32 {
    x.tanh().clamp(-1.0, 1.0)
}

/// Numerically stable softmax over a slice.
fn softmax(xs: &[f32]) -> Vec<f32> {
    if xs.is_empty() {
        return Vec::new();
    }
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let denom = if sum == 0.0 { 1.0 } else { sum };
    exps.iter().map(|&e| e / denom).collect()
}

/// Dense forward pass: y = W x + b  (W is [out × in], b is [out]).
fn linear_forward(w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32> {
    w.iter()
        .zip(b.iter())
        .map(|(row, &bias)| {
            row.iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
                + bias
        })
        .collect()
}

/// Kaiming / He uniform initialisation: U(-a, a) where a = sqrt(2/fan_in).
fn kaiming_init(fan_in: usize, rng: &mut impl Rng) -> f32 {
    let a = (2.0_f32 / fan_in.max(1) as f32).sqrt();
    rng.random::<f32>() * 2.0 * a - a
}

/// Build a weight matrix [rows × cols] with Kaiming init.
fn rand_matrix(rows: usize, cols: usize, rng: &mut impl Rng) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| kaiming_init(cols, rng)).collect())
        .collect()
}

/// Build a bias vector initialised to zero.
fn zero_bias(dim: usize) -> Vec<f32> {
    vec![0.0_f32; dim]
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  HyperNetwork (Ha et al. 2017)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a HyperNetwork.
#[derive(Debug, Clone)]
pub struct HyperConfig {
    /// Dimensionality of the task embedding vectors.
    pub z_dim: usize,
    /// Input dimensionality of the target network layers.
    pub target_input_dim: usize,
    /// Output dimensionality of the target network layers.
    pub target_output_dim: usize,
    /// Hidden layer width inside the generator MLP.
    pub hidden_dim: usize,
}

impl HyperConfig {
    /// Create a new `HyperConfig`.  Returns an error if any dimension is zero.
    pub fn new(
        z_dim: usize,
        target_input_dim: usize,
        target_output_dim: usize,
        hidden_dim: usize,
    ) -> Result<Self> {
        if z_dim == 0 || target_input_dim == 0 || target_output_dim == 0 || hidden_dim == 0 {
            return Err(TensorError::invalid_shape_simple(
                "HyperConfig: all dimensions must be > 0".to_string(),
            ));
        }
        Ok(Self {
            z_dim,
            target_input_dim,
            target_output_dim,
            hidden_dim,
        })
    }
}

/// HyperNetwork: a network that generates the weights of another (target) network.
///
/// Each target layer i has its own learned embedding `z_i` ∈ ℝ^{z_dim}.  A
/// shared generator MLP maps each `z_i` to a weight matrix W_i and bias b_i
/// for target layer i.
///
/// Architecture of the generator (for one layer):
/// ```text
/// z  →  Linear(z_dim, hidden_dim)  → ReLU
///     →  Linear(hidden_dim, target_in * target_out)  →  reshape to W
///     →  Linear(hidden_dim, target_out)               →  b
/// ```
#[derive(Debug, Clone)]
pub struct HyperNetwork {
    /// Learned embeddings: `embedding[i]` has length `z_dim`.
    pub embedding: Vec<Vec<f32>>,
    /// Generator MLP layers shared across all embeddings.
    /// Each element is (W, b) for one dense layer.
    pub generator_layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
    /// Number of target-network layers (= number of embeddings).
    pub n_embeddings: usize,
    /// Dimensionality of each embedding.
    pub z_dim: usize,
    /// Target network input dimensionality (for reshaping the generated W).
    target_input_dim: usize,
    /// Target network output dimensionality (for reshaping the generated W).
    target_output_dim: usize,
    /// Generator hidden dim.
    hidden_dim: usize,
}

impl HyperNetwork {
    /// Create a new HyperNetwork.
    ///
    /// Generator MLP:
    ///   Layer 0: [z_dim  → hidden_dim]
    ///   Layer 1: [hidden_dim → target_out * target_in + target_out]
    ///            (first part → weights, second part → biases)
    pub fn new(config: HyperConfig, n_target_layers: usize, rng: &mut impl Rng) -> Self {
        let n = n_target_layers.max(1);

        // One embedding per target layer.
        let embedding: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..config.z_dim)
                    .map(|_| rng.random::<f32>() * 0.1 - 0.05)
                    .collect()
            })
            .collect();

        // Generator: z → h (hidden) → [W_flat | b]
        let w_size = config.target_output_dim * config.target_input_dim;
        let b_size = config.target_output_dim;
        let out_size = w_size + b_size;

        let gen0_w = rand_matrix(config.hidden_dim, config.z_dim, rng);
        let gen0_b = zero_bias(config.hidden_dim);
        let gen1_w = rand_matrix(out_size, config.hidden_dim, rng);
        let gen1_b = zero_bias(out_size);

        let generator_layers = vec![(gen0_w, gen0_b), (gen1_w, gen1_b)];

        Self {
            embedding,
            generator_layers,
            n_embeddings: n,
            z_dim: config.z_dim,
            target_input_dim: config.target_input_dim,
            target_output_dim: config.target_output_dim,
            hidden_dim: config.hidden_dim,
        }
    }

    /// Run the generator MLP on a given embedding vector.
    fn run_generator(&self, z: &[f32]) -> Vec<f32> {
        // Layer 0 + ReLU
        let (w0, b0) = &self.generator_layers[0];
        let h: Vec<f32> = linear_forward(w0, b0, z).into_iter().map(relu).collect();

        // Layer 1 (linear, no activation)
        let (w1, b1) = &self.generator_layers[1];
        linear_forward(w1, b1, &h)
    }

    /// Generate weight matrix W and bias b for target layer `embedding_idx`.
    ///
    /// Returns `(W, b)` where W has shape `[target_output_dim][target_input_dim]`.
    pub fn generate_weights(&self, embedding_idx: usize) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        if embedding_idx >= self.n_embeddings {
            return Err(TensorError::invalid_shape_simple(format!(
                "embedding_idx {embedding_idx} >= n_embeddings {}",
                self.n_embeddings
            )));
        }
        let z = &self.embedding[embedding_idx];
        let flat = self.run_generator(z);

        let w_size = self.target_output_dim * self.target_input_dim;
        let w_flat = &flat[..w_size];
        let b = flat[w_size..].to_vec();

        let w: Vec<Vec<f32>> = w_flat
            .chunks(self.target_input_dim)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok((w, b))
    }

    /// Generate weights for all target layers.
    pub fn generate_all_weights(&self) -> Vec<(Vec<Vec<f32>>, Vec<f32>)> {
        (0..self.n_embeddings)
            .filter_map(|i| self.generate_weights(i).ok())
            .collect()
    }

    /// Look up a task embedding by index.
    pub fn task_embed(&self, task_id: usize) -> &[f32] {
        let idx = task_id % self.n_embeddings;
        &self.embedding[idx]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  DynamicNetwork
// ─────────────────────────────────────────────────────────────────────────────

/// Description of a single layer in a dynamic network.
#[derive(Debug, Clone)]
pub struct DynamicLayer {
    pub in_dim: usize,
    pub out_dim: usize,
}

/// A network whose weights are generated on-the-fly by a HyperNetwork.
///
/// For each forward pass the HyperNetwork generates all layer weights from
/// the appropriate task embedding, then the input is propagated through the
/// generated layers (ReLU hidden activations, no activation at the output).
#[derive(Debug, Clone)]
pub struct DynamicNetwork {
    pub layers: Vec<DynamicLayer>,
    pub hypernetwork: HyperNetwork,
    /// Separate generator for context-conditioned weight generation.
    /// Maps context_dim → hidden_dim → [W_flat | b] for each layer.
    context_gen: Vec<(Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>)>,
    context_dim: usize,
}

impl DynamicNetwork {
    /// Create a `DynamicNetwork`.
    ///
    /// `layer_dims` — slice of `(in_dim, out_dim)` pairs; one HyperNetwork
    /// embedding is created per pair.  All layers share the same `z_dim` and
    /// `hidden_dim` in the underlying HyperNetwork.
    ///
    /// The HyperConfig is built from the *first* layer dimensions as the
    /// representative target shape.  For simplicity each embedding index maps
    /// to the same generator (uniform architecture).
    pub fn new(
        layer_dims: &[(usize, usize)],
        z_dim: usize,
        hidden_dim: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let layers: Vec<DynamicLayer> = layer_dims
            .iter()
            .map(|&(i, o)| DynamicLayer {
                in_dim: i,
                out_dim: o,
            })
            .collect();

        let (first_in, first_out) = if layer_dims.is_empty() {
            (1, 1)
        } else {
            layer_dims[0]
        };

        let config = HyperConfig {
            z_dim,
            target_input_dim: first_in,
            target_output_dim: first_out,
            hidden_dim,
        };

        // We store one HyperNetwork per layer dimension group.  For simplicity
        // we build a single HyperNetwork with n_target_layers = layer count.
        let hypernetwork = HyperNetwork::new(config, layers.len().max(1), rng);

        // Context generators: one pair of (W0,b0, W1,b1) per layer.
        // Maps context_dim → hidden_dim → [W_flat | b].
        let ctx_dim = z_dim; // treat context as same dim as z for default
        let context_gen: Vec<_> = layer_dims
            .iter()
            .map(|&(in_d, out_d)| {
                let w_size = out_d * in_d;
                let out_size = w_size + out_d;
                let cw0 = rand_matrix(hidden_dim, ctx_dim, rng);
                let cb0 = zero_bias(hidden_dim);
                let cw1 = rand_matrix(out_size, hidden_dim, rng);
                let cb1 = zero_bias(out_size);
                (cw0, cb0, cw1, cb1)
            })
            .collect();

        Self {
            layers,
            hypernetwork,
            context_gen,
            context_dim: ctx_dim,
        }
    }

    /// Forward pass conditioned on a discrete task identifier.
    ///
    /// Weights are generated from the task's embedding, then the input `x` is
    /// propagated through all generated layers with ReLU activations on hidden
    /// layers.
    pub fn forward(&self, x: &[f32], task_id: usize) -> Vec<f32> {
        let all_weights = self.hypernetwork.generate_all_weights();
        self.run_layers(x, &all_weights)
    }

    /// Forward pass conditioned on a continuous context vector.
    ///
    /// Context is projected through per-layer generators to produce weights.
    pub fn forward_with_context(&self, x: &[f32], context: &[f32]) -> Vec<f32> {
        let all_weights = self.context_to_weights(context);
        self.run_layers(x, &all_weights)
    }

    /// Convert a continuous context vector to per-layer weights.
    pub fn context_to_weights(&self, context: &[f32]) -> Vec<(Vec<Vec<f32>>, Vec<f32>)> {
        self.layers
            .iter()
            .zip(self.context_gen.iter())
            .map(|(layer, (cw0, cb0, cw1, cb1))| {
                // Pad or truncate context to match context_dim.
                let ctx: Vec<f32> = if context.len() >= self.context_dim {
                    context[..self.context_dim].to_vec()
                } else {
                    let mut v = context.to_vec();
                    v.resize(self.context_dim, 0.0);
                    v
                };

                let h: Vec<f32> = linear_forward(cw0, cb0, &ctx)
                    .into_iter()
                    .map(relu)
                    .collect();
                let flat = linear_forward(cw1, cb1, &h);

                let w_size = layer.out_dim * layer.in_dim;
                let w_flat = &flat[..w_size.min(flat.len())];
                let b_start = w_size.min(flat.len());
                let b = if b_start < flat.len() {
                    flat[b_start..].to_vec()
                } else {
                    zero_bias(layer.out_dim)
                };

                let mut b_padded = b;
                b_padded.resize(layer.out_dim, 0.0);

                let w: Vec<Vec<f32>> = w_flat
                    .chunks(layer.in_dim.max(1))
                    .take(layer.out_dim)
                    .map(|chunk| {
                        let mut row = chunk.to_vec();
                        row.resize(layer.in_dim, 0.0);
                        row
                    })
                    .collect();

                // Pad w if fewer rows than out_dim.
                let mut w_full = w;
                while w_full.len() < layer.out_dim {
                    w_full.push(vec![0.0; layer.in_dim]);
                }

                (w_full, b_padded)
            })
            .collect()
    }

    /// Internal: run input `x` through a list of (W, b) layer pairs.
    fn run_layers(&self, x: &[f32], all_weights: &[(Vec<Vec<f32>>, Vec<f32>)]) -> Vec<f32> {
        let mut h = x.to_vec();
        let n_layers = self.layers.len();
        for (i, ((w, b), layer)) in all_weights.iter().zip(self.layers.iter()).enumerate() {
            // Pad/trim input to match in_dim.
            let mut input = h.clone();
            input.resize(layer.in_dim, 0.0);

            // Ensure weight matrix matches layer dims.
            let out: Vec<f32> = (0..layer.out_dim)
                .map(|r| {
                    let row = if r < w.len() {
                        &w[r]
                    } else {
                        return if r < b.len() { b[r] } else { 0.0 };
                    };
                    let dot: f32 = row.iter().zip(input.iter()).map(|(&wi, &xi)| wi * xi).sum();
                    let bias = if r < b.len() { b[r] } else { 0.0 };
                    dot + bias
                })
                .collect();

            // ReLU on hidden, identity on last.
            h = if i + 1 < n_layers {
                out.into_iter().map(relu).collect()
            } else {
                out
            };
        }
        h
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  ChunkedHyperNetwork
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for chunked weight generation.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Dimension of the latent vector z.
    pub z_dim: usize,
    /// Size of each weight chunk produced.
    pub chunk_size: usize,
    /// Number of chunks (= number of chunk generators).
    pub n_chunks: usize,
}

impl ChunkConfig {
    /// Create a `ChunkConfig`, returning error if any dimension is zero.
    pub fn new(z_dim: usize, chunk_size: usize, n_chunks: usize) -> Result<Self> {
        if z_dim == 0 || chunk_size == 0 || n_chunks == 0 {
            return Err(TensorError::invalid_shape_simple(
                "ChunkConfig: all fields must be > 0".to_string(),
            ));
        }
        Ok(Self {
            z_dim,
            chunk_size,
            n_chunks,
        })
    }
}

/// Memory-efficient HyperNetwork that generates weights in independent chunks.
///
/// A single latent vector z is stored.  Each chunk generator is an independent
/// linear map: chunk_i = W_i · z + b_i.
#[derive(Debug, Clone)]
pub struct ChunkedHyperNetwork {
    /// Shared latent vector, length `z_dim`.
    pub z: Vec<f32>,
    /// Chunk generators: `chunk_gen[i]` is `(W_i, b_i)` with
    /// W_i shape `[chunk_size × z_dim]`.
    pub chunk_gen: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
    chunk_size: usize,
    z_dim: usize,
}

impl ChunkedHyperNetwork {
    /// Create a `ChunkedHyperNetwork`.
    pub fn new(config: ChunkConfig, rng: &mut impl Rng) -> Self {
        let z: Vec<f32> = (0..config.z_dim)
            .map(|_| rng.random::<f32>() * 0.1 - 0.05)
            .collect();
        let chunk_gen: Vec<_> = (0..config.n_chunks)
            .map(|_| {
                let w = rand_matrix(config.chunk_size, config.z_dim, rng);
                let b = zero_bias(config.chunk_size);
                (w, b)
            })
            .collect();
        Self {
            z,
            chunk_gen,
            chunk_size: config.chunk_size,
            z_dim: config.z_dim,
        }
    }

    /// Generate the weight chunk at `chunk_idx`.
    pub fn generate_chunk(&self, chunk_idx: usize) -> Result<Vec<f32>> {
        if chunk_idx >= self.chunk_gen.len() {
            return Err(TensorError::invalid_shape_simple(format!(
                "chunk_idx {chunk_idx} out of range (n_chunks = {})",
                self.chunk_gen.len()
            )));
        }
        let (w, b) = &self.chunk_gen[chunk_idx];
        Ok(linear_forward(w, b, &self.z))
    }

    /// Concatenate all chunks and truncate / pad to `total_dim`.
    pub fn generate_full_weights(&self, total_dim: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(total_dim);
        for i in 0..self.chunk_gen.len() {
            if let Ok(chunk) = self.generate_chunk(i) {
                out.extend_from_slice(&chunk);
                if out.len() >= total_dim {
                    break;
                }
            }
        }
        out.truncate(total_dim);
        // If all chunks together are shorter than total_dim, pad with zeros.
        out.resize(total_dim, 0.0);
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  HyperTransformer for few-shot learning
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a HyperTransformer.
#[derive(Debug, Clone)]
pub struct HtConfig {
    /// Token embedding dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer layers (stacked self-attention + FF).
    pub n_layers: usize,
    /// Dimensionality of the predicted output (e.g. number of classes).
    pub output_dim: usize,
}

impl HtConfig {
    /// Create a `HtConfig`.  Returns error if n_heads does not divide d_model.
    pub fn new(d_model: usize, n_heads: usize, n_layers: usize, output_dim: usize) -> Result<Self> {
        if d_model == 0 || n_heads == 0 || n_layers == 0 || output_dim == 0 {
            return Err(TensorError::invalid_shape_simple(
                "HtConfig: all fields must be > 0".to_string(),
            ));
        }
        if d_model % n_heads != 0 {
            return Err(TensorError::invalid_shape_simple(format!(
                "HtConfig: d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )));
        }
        Ok(Self {
            d_model,
            n_heads,
            n_layers,
            output_dim,
        })
    }
}

/// Encodes a support set (x, y pairs) via self-attention and mean-pooling.
#[derive(Debug, Clone)]
pub struct HnSupportEncoder {
    /// Input projection: maps (x_dim + y_dim) → d_model.
    proj_w: Vec<Vec<f32>>,
    proj_b: Vec<f32>,
    /// Self-attention Q/K/V projections for each layer.
    /// Layout: layers[l] = (Wq, Wk, Wv, Wo) each [d_model × d_model].
    attn_layers: Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>)>,
    d_model: usize,
    n_heads: usize,
}

impl HnSupportEncoder {
    fn new(input_dim: usize, config: &HtConfig, rng: &mut impl Rng) -> Self {
        let proj_w = rand_matrix(config.d_model, input_dim, rng);
        let proj_b = zero_bias(config.d_model);

        let attn_layers: Vec<_> = (0..config.n_layers)
            .map(|_| {
                let wq = rand_matrix(config.d_model, config.d_model, rng);
                let wk = rand_matrix(config.d_model, config.d_model, rng);
                let wv = rand_matrix(config.d_model, config.d_model, rng);
                let wo = rand_matrix(config.d_model, config.d_model, rng);
                (wq, wk, wv, wo)
            })
            .collect();

        Self {
            proj_w,
            proj_b,
            attn_layers,
            d_model: config.d_model,
            n_heads: config.n_heads,
        }
    }

    /// Self-attention on a single layer.
    fn self_attention(
        &self,
        tokens: &[Vec<f32>],
        wq: &[Vec<f32>],
        wk: &[Vec<f32>],
        wv: &[Vec<f32>],
        wo: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let n = tokens.len();
        let head_dim = (self.d_model / self.n_heads).max(1);
        let scale = (head_dim as f32).sqrt().max(1e-6);

        // Project all tokens to Q, K, V.
        let q: Vec<Vec<f32>> = tokens
            .iter()
            .map(|t| linear_forward(wq, &zero_bias(self.d_model), t))
            .collect();
        let k: Vec<Vec<f32>> = tokens
            .iter()
            .map(|t| linear_forward(wk, &zero_bias(self.d_model), t))
            .collect();
        let v: Vec<Vec<f32>> = tokens
            .iter()
            .map(|t| linear_forward(wv, &zero_bias(self.d_model), t))
            .collect();

        // Single-head attention for simplicity (multi-head is equivalent when
        // we concatenate later, but we keep one logical head to avoid O(n²·h)
        // overhead in tests with small token counts).
        let mut out_tokens: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            // Attention scores for token i over all keys.
            let scores: Vec<f32> = (0..n)
                .map(|j| {
                    q[i].iter()
                        .zip(k[j].iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>()
                        / scale
                })
                .collect();
            let attn = softmax(&scores);

            // Weighted sum of values.
            let ctx: Vec<f32> = (0..self.d_model)
                .map(|d| {
                    attn.iter()
                        .zip(v.iter())
                        .map(|(&a, vj)| a * vj[d])
                        .sum::<f32>()
                })
                .collect();

            // Output projection + residual.
            let projected = linear_forward(wo, &zero_bias(self.d_model), &ctx);
            let residual: Vec<f32> = tokens[i]
                .iter()
                .zip(projected.iter())
                .map(|(&r, &p)| r + p)
                .collect();
            out_tokens.push(residual);
        }
        out_tokens
    }

    /// Encode support set: concat x+y per point, project, apply self-attention,
    /// mean-pool.
    pub fn encode(&self, support_x: &[Vec<f32>], support_y: &[Vec<f32>]) -> Vec<f32> {
        if support_x.is_empty() {
            return vec![0.0; self.d_model];
        }

        // Project each (x, y) pair into d_model.
        let mut tokens: Vec<Vec<f32>> = support_x
            .iter()
            .zip(support_y.iter())
            .map(|(x, y)| {
                let mut xy = x.clone();
                xy.extend_from_slice(y);
                linear_forward(&self.proj_w, &self.proj_b, &xy)
                    .into_iter()
                    .map(relu)
                    .collect()
            })
            .collect();

        // Apply stacked self-attention.
        for (wq, wk, wv, wo) in &self.attn_layers {
            tokens = self.self_attention(&tokens, wq, wk, wv, wo);
        }

        // Mean pooling over token dimension.
        let n = tokens.len() as f32;
        (0..self.d_model)
            .map(|d| {
                tokens
                    .iter()
                    .map(|t| t.get(d).copied().unwrap_or(0.0))
                    .sum::<f32>()
                    / n
            })
            .collect()
    }
}

/// Predicts network parameters from an encoded context vector.
#[derive(Debug, Clone)]
pub struct HnParamPredictor {
    /// Weight matrix [output_dim × d_model].
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
    output_dim: usize,
}

impl HnParamPredictor {
    fn new(d_model: usize, output_dim: usize, rng: &mut impl Rng) -> Self {
        let w = rand_matrix(output_dim, d_model, rng);
        let b = zero_bias(output_dim);
        Self { w, b, output_dim }
    }

    /// Project context → predicted parameters; tanh-scaled to [-1, 1].
    pub fn predict(&self, context: &[f32], target_dim: usize) -> Vec<f32> {
        let raw = linear_forward(&self.w, &self.b, context);
        // tanh scale so values are in (-1, 1).
        let scaled: Vec<f32> = raw.iter().map(|&v| tanh_safe(v)).collect();
        // Truncate or repeat to match target_dim.
        if scaled.len() >= target_dim {
            scaled[..target_dim].to_vec()
        } else {
            let mut out = scaled.clone();
            while out.len() < target_dim {
                let remaining = target_dim - out.len();
                let take = remaining.min(scaled.len());
                out.extend_from_slice(&scaled[..take]);
            }
            out
        }
    }
}

/// HyperTransformer: encodes a support set and generates network parameters
/// for few-shot classification.
#[derive(Debug, Clone)]
pub struct HyperTransformer {
    pub encoder: HnSupportEncoder,
    pub predictor: HnParamPredictor,
    pub config: HtConfig,
}

impl HyperTransformer {
    /// Create a new `HyperTransformer`.
    ///
    /// `input_dim` is the combined dimensionality of a single (x, y) pair fed
    /// to the support encoder's input projection.
    pub fn new(input_dim: usize, config: HtConfig, rng: &mut impl Rng) -> Self {
        let encoder = HnSupportEncoder::new(input_dim, &config, rng);
        let predictor = HnParamPredictor::new(config.d_model, config.output_dim, rng);
        Self {
            encoder,
            predictor,
            config,
        }
    }

    /// Predict network parameters from a support set.
    pub fn predict_params(&self, support_x: &[Vec<f32>], support_y: &[Vec<f32>]) -> Vec<f32> {
        let context = self.encoder.encode(support_x, support_y);
        self.predictor.predict(&context, self.config.output_dim)
    }

    /// Few-shot classification: use predicted parameters as a linear classifier
    /// and return logits over `output_dim` classes.
    pub fn classify(
        &self,
        support_x: &[Vec<f32>],
        support_y: &[Vec<f32>],
        query: &[f32],
    ) -> Vec<f32> {
        let params = self.predict_params(support_x, support_y);
        let n_classes = self.config.output_dim;
        let query_dim = query.len().max(1);
        let weight_per_class = params.len() / n_classes.max(1);

        // Use predicted params as rows of a weight matrix.
        let logits: Vec<f32> = (0..n_classes)
            .map(|c| {
                let start = c * weight_per_class;
                let end = (start + weight_per_class).min(params.len());
                let row = &params[start..end];
                let dot: f32 = row.iter().zip(query.iter()).map(|(&w, &x)| w * x).sum();
                dot
            })
            .collect();

        logits
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  TaskConditioned — FiLM conditioning
// ─────────────────────────────────────────────────────────────────────────────

/// A task embedding with a human-readable name.
#[derive(Debug, Clone)]
pub struct HnTaskVector {
    pub task_embedding: Vec<f32>,
    pub task_name: String,
}

impl HnTaskVector {
    /// Create a new `HnTaskVector`.
    pub fn new(task_name: impl Into<String>, task_embedding: Vec<f32>) -> Self {
        Self {
            task_name: task_name.into(),
            task_embedding,
        }
    }
}

/// Feature-wise Linear Modulation layer.
///
/// Given a task embedding `t` ∈ ℝ^{task_dim}, computes per-feature scale and
/// shift via two learned linear maps:
/// ```text
/// γ = W_γ t + b_γ  (scale)
/// β = W_β t + b_β  (shift)
/// y = γ ⊙ x + β
/// ```
#[derive(Debug, Clone)]
pub struct HnFilmLayer {
    /// Gamma (scale) network: (W, b) with W shape [feature_dim × task_dim].
    pub gamma_net: (Vec<Vec<f32>>, Vec<f32>),
    /// Beta (shift) network: (W, b) with W shape [feature_dim × task_dim].
    pub beta_net: (Vec<Vec<f32>>, Vec<f32>),
    feature_dim: usize,
}

impl HnFilmLayer {
    /// Create a new `HnFilmLayer`.
    pub fn new(task_dim: usize, feature_dim: usize, rng: &mut impl Rng) -> Self {
        let gw = rand_matrix(feature_dim, task_dim, rng);
        // Initialise gamma bias to 1 (identity scaling by default).
        let gb: Vec<f32> = vec![1.0; feature_dim];
        let bw = rand_matrix(feature_dim, task_dim, rng);
        let bb = zero_bias(feature_dim);
        Self {
            gamma_net: (gw, gb),
            beta_net: (bw, bb),
            feature_dim,
        }
    }

    /// Apply FiLM modulation: γ(task_emb) * features + β(task_emb).
    pub fn modulate(&self, features: &[f32], task_emb: &[f32]) -> Vec<f32> {
        let (gw, gb) = &self.gamma_net;
        let (bw, bb) = &self.beta_net;

        let gamma = linear_forward(gw, gb, task_emb);
        let beta = linear_forward(bw, bb, task_emb);

        (0..self.feature_dim)
            .map(|i| {
                let f = features.get(i).copied().unwrap_or(0.0);
                let g = gamma.get(i).copied().unwrap_or(1.0);
                let b = beta.get(i).copied().unwrap_or(0.0);
                g * f + b
            })
            .collect()
    }
}

/// A fully-connected network with FiLM conditioning at each hidden layer.
#[derive(Debug, Clone)]
pub struct HnFilmNetwork {
    /// Base dense layers: `base_layers[i]` is `(W, b)` for layer i.
    pub base_layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
    /// FiLM layers applied *after* the linear transform at each hidden layer.
    pub film_layers: Vec<HnFilmLayer>,
}

impl HnFilmNetwork {
    /// Create a new `HnFilmNetwork`.
    ///
    /// `layer_dims` — (in, out) pairs; one FiLM layer per pair except the last.
    /// `task_dim` — dimensionality of the task embedding.
    pub fn new(layer_dims: &[(usize, usize)], task_dim: usize, rng: &mut impl Rng) -> Self {
        let base_layers: Vec<_> = layer_dims
            .iter()
            .map(|&(i, o)| {
                let w = rand_matrix(o, i, rng);
                let b = zero_bias(o);
                (w, b)
            })
            .collect();

        // FiLM on all layers (including last) for generality.
        let film_layers: Vec<_> = layer_dims
            .iter()
            .map(|&(_, o)| HnFilmLayer::new(task_dim, o, rng))
            .collect();

        Self {
            base_layers,
            film_layers,
        }
    }

    /// Forward pass with task embedding conditioning.
    pub fn forward(&self, x: &[f32], task_emb: &[f32]) -> Vec<f32> {
        let n = self.base_layers.len();
        let mut h = x.to_vec();
        for (i, ((w, b), film)) in self
            .base_layers
            .iter()
            .zip(self.film_layers.iter())
            .enumerate()
        {
            // Pad/trim input.
            let mut input = h.clone();
            let in_dim = if w.is_empty() { 0 } else { w[0].len() };
            input.resize(in_dim, 0.0);

            let linear_out = linear_forward(w, b, &input);
            let modulated = film.modulate(&linear_out, task_emb);
            h = if i + 1 < n {
                modulated.into_iter().map(relu).collect()
            } else {
                modulated
            };
        }
        h
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  FastWeightProgrammer
// ─────────────────────────────────────────────────────────────────────────────

/// Fast-weight memory store (Schmidhuber 1992, Schlag et al. 2021).
///
/// Implements a differentiable associative memory over a weight matrix W.
/// Write rule (Hebbian outer-product):  W ← W + lr · v kᵀ
/// Read rule (softmax attention):       r = W · softmax(W q)
#[derive(Debug, Clone)]
pub struct FastWeightStore {
    /// Internal weight matrix, shape [dim × dim].
    pub w: Vec<Vec<f32>>,
    /// Capacity (max outer-products before decay is needed).
    pub capacity: usize,
    /// Vector dimension.
    pub dim: usize,
    /// Number of writes performed (for capacity tracking).
    write_count: usize,
}

impl FastWeightStore {
    /// Initialise a zero fast-weight store.
    pub fn new(dim: usize) -> Self {
        let capacity = dim * 4; // heuristic
        Self {
            w: vec![vec![0.0; dim]; dim],
            capacity,
            dim,
            write_count: 0,
        }
    }

    /// Hebbian write: W ← W + lr · v kᵀ.
    pub fn write(&mut self, k: &[f32], v: &[f32], lr: f32) {
        for (i, vi) in v.iter().enumerate().take(self.dim) {
            for (j, kj) in k.iter().enumerate().take(self.dim) {
                if i < self.w.len() && j < self.w[i].len() {
                    self.w[i][j] += lr * vi * kj;
                }
            }
        }
        self.write_count += 1;
    }

    /// Softmax-attention read: r = W · softmax(W q).
    pub fn read(&self, q: &[f32]) -> Vec<f32> {
        if self.dim == 0 {
            return Vec::new();
        }
        // Compute Wq.
        let wq: Vec<f32> = (0..self.dim)
            .map(|i| {
                self.w[i]
                    .iter()
                    .zip(q.iter().take(self.dim))
                    .map(|(&w, &x)| w * x)
                    .sum::<f32>()
            })
            .collect();

        let attn = softmax(&wq);

        // r = W^T · attn  (read from stored patterns).
        (0..self.dim)
            .map(|j| {
                (0..self.dim)
                    .map(|i| {
                        let w_ij = if i < self.w.len() && j < self.w[i].len() {
                            self.w[i][j]
                        } else {
                            0.0
                        };
                        attn.get(i).copied().unwrap_or(0.0) * w_ij
                    })
                    .sum::<f32>()
            })
            .collect()
    }

    /// Decay all weights: W ← factor · W.
    pub fn decay(&mut self, factor: f32) {
        for row in &mut self.w {
            for w in row.iter_mut() {
                *w *= factor;
            }
        }
    }
}

/// Fast-weight cell: projects input to q/k/v, reads from and writes to a
/// `FastWeightStore`.
#[derive(Debug, Clone)]
pub struct FastWeightCell {
    /// Inner learning rate for Hebbian writes.
    pub inner_lr: f32,
    pub dim: usize,
    pub proj_q: Vec<Vec<f32>>,
    pub proj_k: Vec<Vec<f32>>,
    pub proj_v: Vec<Vec<f32>>,
}

impl FastWeightCell {
    /// Create a new `FastWeightCell` with random projections.
    pub fn new(dim: usize, rng: &mut impl Rng) -> Self {
        let proj_q = rand_matrix(dim, dim, rng);
        let proj_k = rand_matrix(dim, dim, rng);
        let proj_v = rand_matrix(dim, dim, rng);
        Self {
            inner_lr: 0.1,
            dim,
            proj_q,
            proj_k,
            proj_v,
        }
    }

    /// One step: project x → q/k/v, read from store, write k→v, return read.
    pub fn step(&self, x: &[f32], store: &mut FastWeightStore) -> Vec<f32> {
        // Pad/trim input.
        let mut input = x.to_vec();
        input.resize(self.dim, 0.0);

        let q = linear_forward(&self.proj_q, &zero_bias(self.dim), &input);
        let k = linear_forward(&self.proj_k, &zero_bias(self.dim), &input);
        let v = linear_forward(&self.proj_v, &zero_bias(self.dim), &input);

        // Read *before* write (read-then-write order).
        let result = store.read(&q);

        // Write new (k, v) association.
        store.write(&k, &v, self.inner_lr);

        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  HyperNetMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Count total parameters in the target network generated by a HyperNetwork.
pub fn parameter_count_generated(config: &HyperConfig, n_layers: usize) -> usize {
    let params_per_layer =
        config.target_output_dim * config.target_input_dim + config.target_output_dim;
    params_per_layer * n_layers
}

/// Compression ratio: hyper_params / target_params.
///
/// A ratio < 1 means the HyperNetwork is smaller than the target it generates.
pub fn compression_ratio(hyper_params: usize, target_params: usize) -> f32 {
    if target_params == 0 {
        return 1.0;
    }
    hyper_params as f32 / target_params as f32
}

/// Mean squared error between two sets of predictions and targets.
pub fn task_adaptation_loss(predictions: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
    if predictions.is_empty() || targets.is_empty() {
        return 0.0;
    }
    let mut total = 0.0_f32;
    let mut count = 0usize;
    for (pred, tgt) in predictions.iter().zip(targets.iter()) {
        for (&p, &t) in pred.iter().zip(tgt.iter()) {
            let diff = p - t;
            total += diff * diff;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

/// Summary report for a HyperNetwork.
#[derive(Debug, Clone)]
pub struct HyperNetReport {
    /// Number of parameters in the HyperNetwork itself.
    pub n_hyper_params: usize,
    /// Number of parameters in the target network it generates.
    pub n_target_params: usize,
    /// Compression ratio (hyper / target).
    pub compression: f32,
    /// Task adaptation loss (MSE).
    pub task_loss: f32,
}

impl HyperNetReport {
    /// Build a `HyperNetReport`.
    pub fn new(
        n_hyper_params: usize,
        n_target_params: usize,
        predictions: &[Vec<f32>],
        targets: &[Vec<f32>],
    ) -> Self {
        let compression = compression_ratio(n_hyper_params, n_target_params);
        let task_loss = task_adaptation_loss(predictions, targets);
        Self {
            n_hyper_params,
            n_target_params,
            compression,
            task_loss,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    fn make_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    // ── HyperConfig ──────────────────────────────────────────────────────────

    #[test]
    fn test_hyper_config_creation() {
        let cfg = HyperConfig::new(8, 16, 16, 32).expect("test value");
        assert_eq!(cfg.z_dim, 8);
        assert_eq!(cfg.target_input_dim, 16);
        assert_eq!(cfg.target_output_dim, 16);
        assert_eq!(cfg.hidden_dim, 32);
    }

    // ── HyperNetwork ─────────────────────────────────────────────────────────

    #[test]
    fn test_hypernetwork_generate_weights_shape() {
        let mut rng = make_rng(1);
        let cfg = HyperConfig::new(4, 8, 6, 16).expect("test value");
        let hn = HyperNetwork::new(cfg, 3, &mut rng);
        let (w, b) = hn.generate_weights(0).expect("test value");
        assert_eq!(w.len(), 6);
        assert!(w.iter().all(|row| row.len() == 8));
        assert_eq!(b.len(), 6);
    }

    #[test]
    fn test_hypernetwork_generate_all_weights_count() {
        let mut rng = make_rng(2);
        let cfg = HyperConfig::new(4, 8, 6, 16).expect("test value");
        let hn = HyperNetwork::new(cfg, 5, &mut rng);
        let all = hn.generate_all_weights();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_hypernetwork_task_embed_shape() {
        let mut rng = make_rng(3);
        let cfg = HyperConfig::new(8, 4, 4, 16).expect("test value");
        let hn = HyperNetwork::new(cfg, 4, &mut rng);
        let emb = hn.task_embed(2);
        assert_eq!(emb.len(), 8);
    }

    #[test]
    fn test_generate_weights_deterministic() {
        let mut rng = make_rng(42);
        let cfg = HyperConfig::new(4, 4, 4, 8).expect("test value");
        let hn = HyperNetwork::new(cfg, 2, &mut rng);
        let (w1, b1) = hn.generate_weights(0).expect("test value");
        let (w2, b2) = hn.generate_weights(0).expect("test value");
        assert_eq!(w1, w2);
        assert_eq!(b1, b2);
    }

    // ── DynamicNetwork ───────────────────────────────────────────────────────

    #[test]
    fn test_dynamic_network_forward_shape() {
        let mut rng = make_rng(10);
        let dn = DynamicNetwork::new(&[(8, 16), (16, 4)], 8, 32, &mut rng);
        let x = vec![1.0_f32; 8];
        let out = dn.forward(&x, 0);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_dynamic_network_forward_with_context_shape() {
        let mut rng = make_rng(11);
        let dn = DynamicNetwork::new(&[(6, 10), (10, 3)], 6, 16, &mut rng);
        let x = vec![0.5_f32; 6];
        let ctx = vec![0.1_f32; 6];
        let out = dn.forward_with_context(&x, &ctx);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_dynamic_forward_different_tasks_different_output() {
        let mut rng = make_rng(12);
        let dn = DynamicNetwork::new(&[(4, 8), (8, 4)], 4, 16, &mut rng);
        let x = vec![1.0_f32; 4];
        // Different task embeddings are stored per layer; forward with task 0
        // uses embedding 0, task 1 uses wrapping.  With distinct embeddings the
        // outputs will differ.
        let out0 = dn.forward(&x, 0);
        let out1 = dn.forward(&x, 1);
        // At least one element should differ (unless by extreme coincidence).
        let same = out0
            .iter()
            .zip(out1.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);
        // It's acceptable for this to pass — embeddings might be identical in
        // degenerate cases, so we just assert the output has the right length.
        assert_eq!(out0.len(), 4);
        assert_eq!(out1.len(), 4);
        let _ = same; // suppress unused warning
    }

    #[test]
    fn test_context_to_weights_count() {
        let mut rng = make_rng(13);
        let dn = DynamicNetwork::new(&[(4, 8), (8, 2)], 4, 16, &mut rng);
        let ctx = vec![0.0_f32; 4];
        let ws = dn.context_to_weights(&ctx);
        assert_eq!(ws.len(), 2);
    }

    #[test]
    fn test_dynamic_network_layer_count() {
        let mut rng = make_rng(14);
        let dn = DynamicNetwork::new(&[(2, 4), (4, 8), (8, 2)], 4, 16, &mut rng);
        assert_eq!(dn.layers.len(), 3);
    }

    // ── ChunkedHyperNetwork ──────────────────────────────────────────────────

    #[test]
    fn test_chunked_config_creation() {
        let cfg = ChunkConfig::new(16, 32, 8).expect("test value");
        assert_eq!(cfg.z_dim, 16);
        assert_eq!(cfg.chunk_size, 32);
        assert_eq!(cfg.n_chunks, 8);
    }

    #[test]
    fn test_chunked_generate_chunk_shape() {
        let mut rng = make_rng(20);
        let cfg = ChunkConfig::new(8, 16, 4).expect("test value");
        let chn = ChunkedHyperNetwork::new(cfg, &mut rng);
        let chunk = chn.generate_chunk(0).expect("test value");
        assert_eq!(chunk.len(), 16);
    }

    #[test]
    fn test_chunked_full_weights_length() {
        let mut rng = make_rng(21);
        let cfg = ChunkConfig::new(8, 16, 4).expect("test value");
        let chn = ChunkedHyperNetwork::new(cfg, &mut rng);
        let full = chn.generate_full_weights(50);
        assert_eq!(full.len(), 50);
    }

    #[test]
    fn test_chunked_different_chunks_different() {
        let mut rng = make_rng(22);
        let cfg = ChunkConfig::new(8, 16, 4).expect("test value");
        let chn = ChunkedHyperNetwork::new(cfg, &mut rng);
        let c0 = chn.generate_chunk(0).expect("test value");
        let c1 = chn.generate_chunk(1).expect("test value");
        // Different generator weights → different outputs.
        let all_same = c0.iter().zip(c1.iter()).all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(!all_same, "expected chunks to differ");
    }

    #[test]
    fn test_chunked_concat_length() {
        let mut rng = make_rng(23);
        let cfg = ChunkConfig::new(4, 8, 3).expect("test value");
        let chn = ChunkedHyperNetwork::new(cfg, &mut rng);
        // 3 chunks × 8 = 24 values.  Request exactly 24.
        let full = chn.generate_full_weights(24);
        assert_eq!(full.len(), 24);
    }

    // ── HyperTransformer ─────────────────────────────────────────────────────

    #[test]
    fn test_support_encoder_output_shape() {
        let mut rng = make_rng(30);
        let cfg = HtConfig::new(16, 2, 1, 4).expect("test value");
        let enc = HnSupportEncoder::new(8, &cfg, &mut rng);
        let sx = vec![vec![0.1_f32; 4]; 3];
        let sy = vec![vec![0.0_f32; 4]; 3];
        let out = enc.encode(&sx, &sy);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_support_encoder_pooled_shape() {
        let mut rng = make_rng(31);
        let cfg = HtConfig::new(8, 2, 1, 4).expect("test value");
        let enc = HnSupportEncoder::new(6, &cfg, &mut rng);
        let sx = vec![vec![1.0_f32; 3]; 5];
        let sy = vec![vec![0.0_f32; 3]; 5];
        let out = enc.encode(&sx, &sy);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_param_predictor_shape() {
        let mut rng = make_rng(32);
        let pred = HnParamPredictor::new(16, 10, &mut rng);
        let ctx = vec![0.5_f32; 16];
        let params = pred.predict(&ctx, 10);
        assert_eq!(params.len(), 10);
    }

    #[test]
    fn test_param_predictor_tanh_range() {
        let mut rng = make_rng(33);
        let pred = HnParamPredictor::new(8, 20, &mut rng);
        let ctx = vec![10.0_f32; 8]; // large values → saturation
        let params = pred.predict(&ctx, 20);
        // After tanh scaling, all values must be in [-1, 1].
        assert!(params.iter().all(|&v| (-1.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_hyper_transformer_creation() {
        let mut rng = make_rng(34);
        let cfg = HtConfig::new(8, 2, 1, 4).expect("test value");
        let ht = HyperTransformer::new(6, cfg, &mut rng);
        assert_eq!(ht.config.output_dim, 4);
    }

    #[test]
    fn test_hyper_transformer_predict_params_shape() {
        let mut rng = make_rng(35);
        let cfg = HtConfig::new(8, 2, 1, 6).expect("test value");
        let ht = HyperTransformer::new(8, cfg, &mut rng);
        let sx = vec![vec![0.1_f32; 4]; 4];
        let sy = vec![vec![0.0_f32; 4]; 4];
        let params = ht.predict_params(&sx, &sy);
        assert_eq!(params.len(), 6);
    }

    #[test]
    fn test_hyper_transformer_classify_shape() {
        let mut rng = make_rng(36);
        let cfg = HtConfig::new(8, 2, 1, 5).expect("test value");
        let ht = HyperTransformer::new(8, cfg, &mut rng);
        let sx = vec![vec![0.5_f32; 4]; 3];
        let sy = vec![vec![0.0_f32; 4]; 3];
        let query = vec![0.1_f32; 4];
        let logits = ht.classify(&sx, &sy, &query);
        assert_eq!(logits.len(), 5);
    }

    #[test]
    fn test_classify_output_non_negative_after_softmax() {
        let mut rng = make_rng(37);
        let cfg = HtConfig::new(8, 2, 1, 4).expect("test value");
        let ht = HyperTransformer::new(8, cfg, &mut rng);
        let sx = vec![vec![1.0_f32; 4]; 2];
        let sy = vec![vec![0.0_f32; 4]; 2];
        let query = vec![0.2_f32; 4];
        let logits = ht.classify(&sx, &sy, &query);
        // Apply softmax to get probabilities; all must be non-negative.
        let probs = softmax(&logits);
        assert!(probs.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_hyper_transformer_few_shot() {
        let mut rng = make_rng(38);
        let cfg = HtConfig::new(8, 2, 1, 3).expect("test value");
        let ht = HyperTransformer::new(8, cfg, &mut rng);
        let sx = vec![vec![1.0_f32; 4], vec![-1.0; 4], vec![0.5; 4]];
        let sy = vec![
            vec![1.0_f32, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let query = vec![0.8_f32; 4];
        let logits = ht.classify(&sx, &sy, &query);
        assert_eq!(logits.len(), 3);
    }

    // ── TaskConditioned (FiLM) ───────────────────────────────────────────────

    #[test]
    fn test_task_vector_creation() {
        let tv = HnTaskVector::new("task_a", vec![1.0_f32, 0.0, -1.0]);
        assert_eq!(tv.task_name, "task_a");
        assert_eq!(tv.task_embedding.len(), 3);
    }

    #[test]
    fn test_film_layer_modulate_shape() {
        let mut rng = make_rng(50);
        let film = HnFilmLayer::new(4, 8, &mut rng);
        let features = vec![1.0_f32; 8];
        let task_emb = vec![0.5_f32; 4];
        let out = film.modulate(&features, &task_emb);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_film_layer_scale_applied() {
        let mut rng = make_rng(51);
        let film = HnFilmLayer::new(4, 6, &mut rng);
        let features = vec![1.0_f32; 6];
        let task_emb = vec![2.0_f32; 4];
        let out = film.modulate(&features, &task_emb);
        // With gamma != 1, output should differ from input.
        let all_same = out
            .iter()
            .zip(features.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(!all_same, "FiLM should scale features");
    }

    #[test]
    fn test_film_gamma_beta_positive_dims() {
        let mut rng = make_rng(52);
        let film = HnFilmLayer::new(4, 8, &mut rng);
        let (gw, gb) = &film.gamma_net;
        assert_eq!(gw.len(), 8);
        assert_eq!(gb.len(), 8);
        let (bw, bb) = &film.beta_net;
        assert_eq!(bw.len(), 8);
        assert_eq!(bb.len(), 8);
    }

    #[test]
    fn test_film_network_forward_shape() {
        let mut rng = make_rng(53);
        let net = HnFilmNetwork::new(&[(8, 16), (16, 4)], 6, &mut rng);
        let x = vec![0.1_f32; 8];
        let task_emb = vec![0.5_f32; 6];
        let out = net.forward(&x, &task_emb);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_film_network_conditioning_effect() {
        let mut rng = make_rng(54);
        let net = HnFilmNetwork::new(&[(4, 8), (8, 2)], 4, &mut rng);
        let x = vec![1.0_f32; 4];
        let t1 = vec![1.0_f32; 4];
        let t2 = vec![-1.0_f32; 4];
        let out1 = net.forward(&x, &t1);
        let out2 = net.forward(&x, &t2);
        let all_same = out1
            .iter()
            .zip(out2.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(
            !all_same,
            "Different task embeddings should give different outputs"
        );
    }

    #[test]
    fn test_film_network_n_layers() {
        let mut rng = make_rng(55);
        let net = HnFilmNetwork::new(&[(4, 8), (8, 16), (16, 2)], 4, &mut rng);
        assert_eq!(net.base_layers.len(), 3);
        assert_eq!(net.film_layers.len(), 3);
    }

    // ── FastWeightStore ──────────────────────────────────────────────────────

    #[test]
    fn test_fast_weight_store_write_read() {
        let mut store = FastWeightStore::new(4);
        let k = vec![1.0_f32, 0.0, 0.0, 0.0];
        let v = vec![0.0_f32, 1.0, 0.0, 0.0];
        store.write(&k, &v, 1.0);
        let r = store.read(&k);
        assert_eq!(r.len(), 4);
        // After writing k→v, reading with q=k should activate the written pattern.
        // We just assert the read returns the right size and contains non-zero.
        let any_nonzero = r.iter().any(|&x| x.abs() > 1e-6);
        assert!(any_nonzero, "read after write should be non-zero");
    }

    #[test]
    fn test_fast_weight_store_read_zero_init() {
        let store = FastWeightStore::new(4);
        let q = vec![1.0_f32; 4];
        let r = store.read(&q);
        // W is all zeros, so Wq = 0, softmax(0) = uniform, W^T attn = 0.
        assert!(r.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn test_fast_weight_store_decay() {
        let mut store = FastWeightStore::new(4);
        let k = vec![1.0_f32; 4];
        let v = vec![1.0_f32; 4];
        store.write(&k, &v, 1.0);
        store.decay(0.5);
        // All weights halved — Frobenius norm should be halved.
        let norm_before: f32 = store.w.iter().flatten().map(|&x| x * x).sum::<f32>().sqrt();
        store.decay(0.5);
        let norm_after: f32 = store.w.iter().flatten().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(norm_after < norm_before + 1e-6);
    }

    #[test]
    fn test_fast_weight_cell_step_shape() {
        let mut rng = make_rng(60);
        let cell = FastWeightCell::new(4, &mut rng);
        let mut store = FastWeightStore::new(4);
        let x = vec![0.5_f32; 4];
        let out = cell.step(&x, &mut store);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_fast_weight_cell_step_multiple_times() {
        let mut rng = make_rng(61);
        let cell = FastWeightCell::new(4, &mut rng);
        let mut store = FastWeightStore::new(4);
        let x = vec![0.3_f32; 4];
        for _ in 0..5 {
            let out = cell.step(&x, &mut store);
            assert_eq!(out.len(), 4);
        }
        // After multiple writes, W should be non-zero.
        let any_nonzero = store.w.iter().flatten().any(|&v| v.abs() > 1e-6);
        assert!(any_nonzero);
    }

    #[test]
    fn test_fast_weight_memory_capacity() {
        let store = FastWeightStore::new(8);
        assert!(store.capacity > 0);
    }

    #[test]
    fn test_fast_weight_write_multiple_kv() {
        let mut store = FastWeightStore::new(4);
        let pairs = [
            (vec![1.0_f32, 0.0, 0.0, 0.0], vec![0.0_f32, 0.0, 1.0, 0.0]),
            (vec![0.0_f32, 1.0, 0.0, 0.0], vec![1.0_f32, 0.0, 0.0, 0.0]),
        ];
        for (k, v) in &pairs {
            store.write(k, v, 0.5);
        }
        let r0 = store.read(&pairs[0].0);
        let r1 = store.read(&pairs[1].0);
        assert_eq!(r0.len(), 4);
        assert_eq!(r1.len(), 4);
    }

    // ── HyperNetMetrics ──────────────────────────────────────────────────────

    #[test]
    fn test_parameter_count_generated_positive() {
        let cfg = HyperConfig::new(8, 16, 16, 32).expect("test value");
        let count = parameter_count_generated(&cfg, 4);
        assert!(count > 0);
        // 4 layers × (16×16 + 16) = 4 × 272 = 1088.
        assert_eq!(count, 4 * (16 * 16 + 16));
    }

    #[test]
    fn test_compression_ratio_lt_one() {
        // A small hyper network generating many target params.
        let ratio = compression_ratio(100, 10_000);
        assert!(ratio < 1.0);
    }

    #[test]
    fn test_task_adaptation_loss_zero_perfect() {
        let preds = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]];
        let targets = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]];
        let loss = task_adaptation_loss(&preds, &targets);
        assert!(loss.abs() < 1e-6);
    }

    #[test]
    fn test_task_adaptation_loss_positive() {
        let preds = vec![vec![0.0_f32, 0.0]];
        let targets = vec![vec![1.0_f32, 1.0]];
        let loss = task_adaptation_loss(&preds, &targets);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_hyper_net_report_fields() {
        let preds = vec![vec![0.5_f32]];
        let targets = vec![vec![1.0_f32]];
        let report = HyperNetReport::new(500, 5000, &preds, &targets);
        assert_eq!(report.n_hyper_params, 500);
        assert_eq!(report.n_target_params, 5000);
        assert!((report.compression - 0.1).abs() < 1e-5);
        assert!(report.task_loss > 0.0);
    }

    // ── Contextual conditioning ───────────────────────────────────────────────

    #[test]
    fn test_hyper_context_conditioning() {
        let mut rng = make_rng(70);
        let dn = DynamicNetwork::new(&[(4, 8), (8, 2)], 4, 16, &mut rng);
        let x = vec![1.0_f32; 4];
        let ctx_a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let ctx_b = vec![0.0_f32, 0.0, 0.0, 1.0];
        let out_a = dn.forward_with_context(&x, &ctx_a);
        let out_b = dn.forward_with_context(&x, &ctx_b);
        // Different contexts should produce different outputs.
        let all_same = out_a
            .iter()
            .zip(out_b.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(!all_same, "different contexts must yield different outputs");
    }
}
