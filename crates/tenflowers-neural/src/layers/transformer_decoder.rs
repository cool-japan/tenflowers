//! Seq2Seq Transformer Decoder components (pre-norm style).
//!
//! Implements a production-grade Transformer decoder stack that mirrors the
//! pre-norm design used in `transformer.rs` for the encoder.  The three-sublayer
//! decoder block follows:
//!
//! ```text
//! x = x + MaskedSelfAttn(Norm(x))          // causal self-attention
//! x = x + CrossAttn(Norm(x), enc_out)      // cross-attention to encoder
//! x = x + FFN(Norm(x))                     // position-wise feed-forward
//! ```
//!
//! All types use a flat `Vec<f32>` buffer API so they compose naturally with
//! the rest of the `tenflowers-neural` layer zoo.
//!
//! # Shape convention
//!
//! Flat buffers represent `[seq_len, model_dim]` in row-major order.
//! Batch dimensions are *not* supported in this module (operate per-sample).

use tenflowers_core::TensorError;

use super::transformer::{
    DenseF32, FeedForward, FfnActivation, LayerNormLayer, MultiHeadAttentionLayer, TransformerEncoder,
};

/// Shorthand `Result` type re-using the project-wide error type.
type Result<T> = tenflowers_core::Result<T>;

// ─────────────────────────────────────────────────────────────────────────────
// Private math helpers (local copies kept here to avoid reaching into parent
// module private items)
// ─────────────────────────────────────────────────────────────────────────────

/// Dense linear transform: `y[i] = sum_j x[j] * W[j, i] + b[i]`.
/// `weight` layout is row-major `[in_dim, out_dim]`.
fn linear(x: &[f32], weight: &[f32], bias: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(weight.len(), in_dim * out_dim);
    debug_assert_eq!(bias.len(), out_dim);

    let mut y = vec![0.0_f32; out_dim];
    for j in 0..out_dim {
        let mut acc = 0.0_f32;
        for i in 0..in_dim {
            acc += x[i] * weight[i * out_dim + j];
        }
        acc += bias[j];
        y[j] = acc;
    }
    y
}

/// Apply `linear` to every token in a flat `[n_tokens, in_dim]` buffer.
fn linear_batch(
    tokens: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let n_tokens = tokens.len() / in_dim;
    let mut out = Vec::with_capacity(n_tokens * out_dim);
    for t in 0..n_tokens {
        let row = &tokens[t * in_dim..(t + 1) * in_dim];
        out.extend_from_slice(&linear(row, weight, bias, in_dim, out_dim));
    }
    out
}

/// Softmax over a mutable slice in-place (numerically stable).
fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Scaled dot-product attention for a single head.
///
/// Query shape: `[q_len, head_dim]`, Key/Value shapes: `[kv_len, head_dim]`.
/// When `causal == true` a lower-triangular mask is applied to the `[q_len, kv_len]`
/// score matrix.  For cross-attention `causal` should be `false`.
fn scaled_dot_product_attn_cross(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Scores: [q_len, kv_len]
    let mut scores = vec![0.0_f32; q_len * kv_len];
    for i in 0..q_len {
        for j in 0..kv_len {
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * kv_len + j] = dot * scale;
        }
    }

    // Causal mask: only valid when q_len == kv_len (self-attention).
    if causal {
        for i in 0..q_len {
            for j in (i + 1)..kv_len {
                scores[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    // Row-wise softmax.
    for i in 0..q_len {
        softmax_inplace(&mut scores[i * kv_len..(i + 1) * kv_len]);
    }

    // Output: [q_len, head_dim]
    let mut out = vec![0.0_f32; q_len * head_dim];
    for i in 0..q_len {
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for j in 0..kv_len {
                acc += scores[i * kv_len + j] * v[j * head_dim + d];
            }
            out[i * head_dim + d] = acc;
        }
    }
    out
}

/// Elementwise addition of two equal-length flat buffers.
fn elementwise_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(TensorError::invalid_argument(format!(
            "elementwise_add: length mismatch {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// CrossAttentionLayer
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-attention layer: attends from decoder queries to encoder key/value pairs.
///
/// Query projections are computed from the decoder hidden state; key and value
/// projections are computed from the encoder output.  All four weight matrices
/// (`Wq`, `Wk`, `Wv`, `Wo`) have shape `[model_dim, model_dim]`.
///
/// # Shape convention
///
/// * `decoder_x`: flat `[tgt_seq * model_dim]`
/// * `encoder_out`: flat `[src_seq * model_dim]`
/// * Output: flat `[tgt_seq * model_dim]`
#[derive(Debug, Clone)]
pub struct CrossAttentionLayer {
    /// Query projection (from decoder hidden state).
    pub wq: DenseF32,
    /// Key projection (from encoder output).
    pub wk: DenseF32,
    /// Value projection (from encoder output).
    pub wv: DenseF32,
    /// Output projection.
    pub wo: DenseF32,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head (`model_dim / num_heads`).
    pub head_dim: usize,
    /// Total model dimension.
    pub model_dim: usize,
}

impl CrossAttentionLayer {
    /// Construct a new `CrossAttentionLayer`.
    ///
    /// Requires `model_dim % num_heads == 0`.
    pub fn new(model_dim: usize, num_heads: usize) -> Result<Self> {
        if num_heads == 0 {
            return Err(TensorError::invalid_argument(
                "CrossAttentionLayer: num_heads must be > 0".to_string(),
            ));
        }
        if model_dim % num_heads != 0 {
            return Err(TensorError::invalid_argument(format!(
                "CrossAttentionLayer: model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
            )));
        }
        let head_dim = model_dim / num_heads;
        Ok(Self {
            wq: DenseF32::new(model_dim, model_dim)?,
            wk: DenseF32::new(model_dim, model_dim)?,
            wv: DenseF32::new(model_dim, model_dim)?,
            wo: DenseF32::new(model_dim, model_dim)?,
            num_heads,
            head_dim,
            model_dim,
        })
    }

    /// Forward pass.
    ///
    /// * `decoder_x` – flat `[tgt_seq, model_dim]` decoder hidden states.
    /// * `encoder_out` – flat `[src_seq, model_dim]` encoder outputs.
    ///
    /// Returns `(output, shape)` where `shape = [tgt_seq, model_dim]`.
    pub fn forward(
        &self,
        decoder_x: &[f32],
        tgt_seq: usize,
        encoder_out: &[f32],
        src_seq: usize,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        self.validate_inputs(decoder_x, tgt_seq, encoder_out, src_seq)?;

        // Project decoder queries.
        let q_flat = linear_batch(
            decoder_x,
            &self.wq.weight,
            &self.wq.bias,
            self.model_dim,
            self.model_dim,
        );
        // Project encoder keys and values.
        let k_flat = linear_batch(
            encoder_out,
            &self.wk.weight,
            &self.wk.bias,
            self.model_dim,
            self.model_dim,
        );
        let v_flat = linear_batch(
            encoder_out,
            &self.wv.weight,
            &self.wv.bias,
            self.model_dim,
            self.model_dim,
        );

        // Multi-head cross-attention.
        let mut attn_out = vec![0.0_f32; tgt_seq * self.model_dim];

        for h in 0..self.num_heads {
            // Extract per-head slices.
            let q_h = extract_head_slice(&q_flat, tgt_seq, self.model_dim, h, self.head_dim);
            let k_h = extract_head_slice(&k_flat, src_seq, self.model_dim, h, self.head_dim);
            let v_h = extract_head_slice(&v_flat, src_seq, self.model_dim, h, self.head_dim);

            // Attend (no causal mask for cross-attention).
            let head_out = scaled_dot_product_attn_cross(
                &q_h,
                &k_h,
                &v_h,
                tgt_seq,
                src_seq,
                self.head_dim,
                false,
            );

            // Write head output back into the concatenated buffer.
            for t in 0..tgt_seq {
                let dst = t * self.model_dim + h * self.head_dim;
                attn_out[dst..dst + self.head_dim]
                    .copy_from_slice(&head_out[t * self.head_dim..(t + 1) * self.head_dim]);
            }
        }

        // Output projection.
        let out = linear_batch(
            &attn_out,
            &self.wo.weight,
            &self.wo.bias,
            self.model_dim,
            self.model_dim,
        );

        Ok((out, vec![tgt_seq, self.model_dim]))
    }

    // ── validation ────────────────────────────────────────────────────────

    fn validate_inputs(
        &self,
        decoder_x: &[f32],
        tgt_seq: usize,
        encoder_out: &[f32],
        src_seq: usize,
    ) -> Result<()> {
        if tgt_seq == 0 {
            return Err(TensorError::invalid_argument(
                "CrossAttentionLayer: tgt_seq must be > 0".to_string(),
            ));
        }
        if src_seq == 0 {
            return Err(TensorError::invalid_argument(
                "CrossAttentionLayer: src_seq must be > 0".to_string(),
            ));
        }
        let expected_decoder = tgt_seq * self.model_dim;
        if decoder_x.len() != expected_decoder {
            return Err(TensorError::invalid_argument(format!(
                "CrossAttentionLayer: decoder_x length {} != tgt_seq*model_dim {}",
                decoder_x.len(),
                expected_decoder
            )));
        }
        let expected_encoder = src_seq * self.model_dim;
        if encoder_out.len() != expected_encoder {
            return Err(TensorError::invalid_argument(format!(
                "CrossAttentionLayer: encoder_out length {} != src_seq*model_dim {}",
                encoder_out.len(),
                expected_encoder
            )));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TransformerDecoderBlock
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-norm Transformer decoder block.
///
/// Applies three sublayers with residual connections:
///
/// ```text
/// x = x + self_attn(norm1(x),  causal=true)
/// x = x + cross_attn(norm2(x), encoder_output)
/// x = x + ffn(norm3(x))
/// ```
///
/// Input and output share shape `[tgt_seq, model_dim]`.
#[derive(Debug, Clone)]
pub struct TransformerDecoderBlock {
    /// Causal self-attention sublayer.
    pub self_attn: MultiHeadAttentionLayer,
    /// Cross-attention sublayer (decoder→encoder).
    pub cross_attn: CrossAttentionLayer,
    /// Position-wise feed-forward sublayer.
    pub ffn: FeedForward,
    /// Layer norm applied before self-attention.
    pub norm1: LayerNormLayer,
    /// Layer norm applied before cross-attention.
    pub norm2: LayerNormLayer,
    /// Layer norm applied before FFN.
    pub norm3: LayerNormLayer,
    /// Dropout probability for sublayer outputs.
    pub dropout_prob: f32,
}

impl TransformerDecoderBlock {
    /// Construct a new decoder block.
    ///
    /// `ff_dim` is typically `4 * model_dim`.
    pub fn new(model_dim: usize, num_heads: usize, ff_dim: usize) -> Result<Self> {
        Self::new_with_dropout(model_dim, num_heads, ff_dim, 0.0)
    }

    /// Construct a new decoder block with explicit dropout probability.
    pub fn new_with_dropout(
        model_dim: usize,
        num_heads: usize,
        ff_dim: usize,
        dropout_prob: f32,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&dropout_prob) {
            return Err(TensorError::invalid_argument(format!(
                "TransformerDecoderBlock: dropout_prob must be in [0, 1], got {dropout_prob}"
            )));
        }
        Ok(Self {
            self_attn: MultiHeadAttentionLayer::new(model_dim, num_heads)?,
            cross_attn: CrossAttentionLayer::new(model_dim, num_heads)?,
            ffn: FeedForward::new(model_dim, ff_dim, FfnActivation::Gelu, dropout_prob)?,
            norm1: LayerNormLayer::new(model_dim),
            norm2: LayerNormLayer::new(model_dim),
            norm3: LayerNormLayer::new(model_dim),
            dropout_prob,
        })
    }

    /// Forward pass of the decoder block.
    ///
    /// * `decoder_input` – flat `[tgt_seq, model_dim]`.
    /// * `encoder_output` – flat `[src_seq, model_dim]`.
    /// * `training` – enables dropout.
    ///
    /// Returns `(output, shape)` where `output` is `[tgt_seq * model_dim]`
    /// and `shape = [tgt_seq, model_dim]`.
    pub fn forward(
        &self,
        decoder_input: &[f32],
        tgt_seq: usize,
        encoder_output: &[f32],
        src_seq: usize,
        training: bool,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let shape = [tgt_seq, self.self_attn.model_dim];
        let model_dim = self.self_attn.model_dim;

        // Validate dimensions.
        if tgt_seq == 0 {
            return Err(TensorError::invalid_argument(
                "TransformerDecoderBlock: tgt_seq must be > 0".to_string(),
            ));
        }
        if src_seq == 0 {
            return Err(TensorError::invalid_argument(
                "TransformerDecoderBlock: src_seq must be > 0".to_string(),
            ));
        }
        let expected_decoder = tgt_seq * model_dim;
        if decoder_input.len() != expected_decoder {
            return Err(TensorError::invalid_argument(format!(
                "TransformerDecoderBlock: decoder_input length {} != tgt_seq*model_dim {}",
                decoder_input.len(),
                expected_decoder
            )));
        }
        let expected_encoder = src_seq * model_dim;
        if encoder_output.len() != expected_encoder {
            return Err(TensorError::invalid_argument(format!(
                "TransformerDecoderBlock: encoder_output length {} != src_seq*model_dim {}",
                encoder_output.len(),
                expected_encoder
            )));
        }

        // ── Sublayer 1: causal self-attention ─────────────────────────────
        let normed1 = self.norm1.forward(decoder_input, &shape)?;
        let (sa_out, _) = self.self_attn.forward(&normed1, &shape, true)?;
        let sa_out = apply_dropout_simple(&sa_out, training, self.dropout_prob, 0);
        let res1 = elementwise_add(decoder_input, &sa_out)?;

        // ── Sublayer 2: cross-attention ───────────────────────────────────
        let normed2 = self.norm2.forward(&res1, &shape)?;
        let (ca_out, _) = self.cross_attn.forward(&normed2, tgt_seq, encoder_output, src_seq)?;
        let ca_out = apply_dropout_simple(&ca_out, training, self.dropout_prob, 1);
        let res2 = elementwise_add(&res1, &ca_out)?;

        // ── Sublayer 3: feed-forward ──────────────────────────────────────
        let normed3 = self.norm3.forward(&res2, &shape)?;
        let (ffn_out, _) = self.ffn.forward(&normed3, &shape, training)?;
        let ffn_out = apply_dropout_simple(&ffn_out, training, self.dropout_prob, 2);
        let out = elementwise_add(&res2, &ffn_out)?;

        Ok((out, vec![tgt_seq, model_dim]))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TransformerDecoder (stack)
// ─────────────────────────────────────────────────────────────────────────────

/// A stack of `N` [`TransformerDecoderBlock`]s.
///
/// Passes the same encoder output to every layer in the stack, as is standard
/// for Seq2Seq transformer decoders.
#[derive(Debug, Clone)]
pub struct TransformerDecoderStack {
    blocks: Vec<TransformerDecoderBlock>,
    /// Number of decoder layers.
    pub num_layers: usize,
}

impl TransformerDecoderStack {
    /// Construct a new decoder stack.
    pub fn new(
        num_layers: usize,
        model_dim: usize,
        num_heads: usize,
        ff_dim: usize,
    ) -> Result<Self> {
        Self::new_with_dropout(num_layers, model_dim, num_heads, ff_dim, 0.0)
    }

    /// Construct a new decoder stack with explicit dropout probability.
    pub fn new_with_dropout(
        num_layers: usize,
        model_dim: usize,
        num_heads: usize,
        ff_dim: usize,
        dropout_prob: f32,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(TensorError::invalid_argument(
                "TransformerDecoderStack: num_layers must be > 0".to_string(),
            ));
        }
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(TransformerDecoderBlock::new_with_dropout(
                model_dim,
                num_heads,
                ff_dim,
                dropout_prob,
            )?);
        }
        Ok(Self { blocks, num_layers })
    }

    /// Pass decoder input through all blocks sequentially.
    ///
    /// `encoder_output` is shared across all layers (not updated between layers).
    pub fn forward(
        &self,
        decoder_input: &[f32],
        tgt_seq: usize,
        encoder_output: &[f32],
        src_seq: usize,
        training: bool,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let mut current = decoder_input.to_vec();
        let mut current_shape = vec![tgt_seq, decoder_input.len() / tgt_seq];
        for block in &self.blocks {
            let (out, shape) = block.forward(&current, tgt_seq, encoder_output, src_seq, training)?;
            current = out;
            current_shape = shape;
        }
        Ok((current, current_shape))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Seq2SeqTransformer
// ─────────────────────────────────────────────────────────────────────────────

/// Full encoder-decoder Seq2Seq Transformer.
///
/// Combines a [`TransformerEncoder`] stack with a [`TransformerDecoderStack`].
/// The encoder processes the source sequence once; the decoder output is
/// conditioned on the encoder's final hidden states via cross-attention.
///
/// # Shape convention
///
/// * `src`: flat `[src_seq * model_dim]`
/// * `tgt`: flat `[tgt_seq * model_dim]`
/// * Output: flat `[tgt_seq * model_dim]`
#[derive(Debug, Clone)]
pub struct Seq2SeqTransformer {
    /// Encoder tower.
    pub encoder: TransformerEncoder,
    /// Decoder tower.
    pub decoder: TransformerDecoderStack,
    /// Model (embedding) dimension.
    pub model_dim: usize,
}

impl Seq2SeqTransformer {
    /// Construct a new `Seq2SeqTransformer`.
    ///
    /// | param | meaning |
    /// |---|---|
    /// | `num_encoder_layers` | depth of encoder stack (≥ 1) |
    /// | `num_decoder_layers` | depth of decoder stack (≥ 1) |
    /// | `model_dim` | token embedding / hidden dimension |
    /// | `num_heads` | number of attention heads (must divide `model_dim`) |
    /// | `ff_dim` | inner dimension of the FFN sublayer |
    pub fn new(
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        model_dim: usize,
        num_heads: usize,
        ff_dim: usize,
    ) -> Result<Self> {
        Self::new_with_dropout(
            num_encoder_layers,
            num_decoder_layers,
            model_dim,
            num_heads,
            ff_dim,
            0.0,
        )
    }

    /// Construct a `Seq2SeqTransformer` with explicit dropout probability.
    pub fn new_with_dropout(
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        model_dim: usize,
        num_heads: usize,
        ff_dim: usize,
        dropout_prob: f32,
    ) -> Result<Self> {
        Ok(Self {
            encoder: TransformerEncoder::new(
                num_encoder_layers,
                model_dim,
                num_heads,
                ff_dim,
                dropout_prob,
            )?,
            decoder: TransformerDecoderStack::new_with_dropout(
                num_decoder_layers,
                model_dim,
                num_heads,
                ff_dim,
                dropout_prob,
            )?,
            model_dim,
        })
    }

    /// Full encoder-decoder forward pass.
    ///
    /// 1. Encode `src` → encoder hidden states `[src_seq, model_dim]`.
    /// 2. Decode `tgt` conditioned on encoder outputs → `[tgt_seq, model_dim]`.
    ///
    /// Returns `(output, shape)` where `shape = [tgt_seq, model_dim]`.
    pub fn forward(
        &self,
        src: &[f32],
        src_seq: usize,
        tgt: &[f32],
        tgt_seq: usize,
        training: bool,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        if src_seq == 0 {
            return Err(TensorError::invalid_argument(
                "Seq2SeqTransformer: src_seq must be > 0".to_string(),
            ));
        }
        if tgt_seq == 0 {
            return Err(TensorError::invalid_argument(
                "Seq2SeqTransformer: tgt_seq must be > 0".to_string(),
            ));
        }
        let expected_src = src_seq * self.model_dim;
        if src.len() != expected_src {
            return Err(TensorError::invalid_argument(format!(
                "Seq2SeqTransformer: src length {} != src_seq*model_dim {}",
                src.len(),
                expected_src
            )));
        }
        let expected_tgt = tgt_seq * self.model_dim;
        if tgt.len() != expected_tgt {
            return Err(TensorError::invalid_argument(format!(
                "Seq2SeqTransformer: tgt length {} != tgt_seq*model_dim {}",
                tgt.len(),
                expected_tgt
            )));
        }

        // Encode source sequence.
        let (enc_out, _) = self
            .encoder
            .forward(src, &[src_seq, self.model_dim], training, false)?;

        // Decode target sequence conditioned on encoder output.
        self.decoder
            .forward(tgt, tgt_seq, &enc_out, src_seq, training)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract a per-head slice `[seq_len, head_dim]` from a `[seq_len, model_dim]` buffer.
fn extract_head_slice(
    flat: &[f32],
    seq_len: usize,
    model_dim: usize,
    head_idx: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(seq_len * head_dim);
    for t in 0..seq_len {
        let start = t * model_dim + head_idx * head_dim;
        out.extend_from_slice(&flat[start..start + head_dim]);
    }
    out
}

/// Inverted dropout applied to a flat buffer.
/// `seed_offset` shifts the quasi-random sequence per sublayer to diversify masks.
fn apply_dropout_simple(v: &[f32], training: bool, dropout_prob: f32, seed_offset: usize) -> Vec<f32> {
    if !training || dropout_prob <= 0.0 {
        return v.to_vec();
    }
    let keep_prob = 1.0 - dropout_prob;
    let scale = 1.0 / keep_prob;
    v.iter()
        .enumerate()
        .map(|(idx, &x)| {
            let frac =
                (((idx + seed_offset * 65537) as f64 * 1.618033988749_f64 + 0.5) % 1.0) as f32;
            if frac < dropout_prob {
                0.0
            } else {
                x * scale
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CrossAttentionLayer ───────────────────────────────────────────────

    #[test]
    fn test_cross_attn_output_shape_basic() {
        let model_dim = 8;
        let num_heads = 2;
        let tgt_seq = 4;
        let src_seq = 6;

        let layer = CrossAttentionLayer::new(model_dim, num_heads)
            .expect("CrossAttentionLayer::new");
        let decoder_x = vec![0.1_f32; tgt_seq * model_dim];
        let encoder_out = vec![0.2_f32; src_seq * model_dim];

        let (out, shape) = layer
            .forward(&decoder_x, tgt_seq, &encoder_out, src_seq)
            .expect("CrossAttentionLayer::forward");

        assert_eq!(
            out.len(),
            tgt_seq * model_dim,
            "output length must equal tgt_seq * model_dim"
        );
        assert_eq!(shape, vec![tgt_seq, model_dim], "shape must be [tgt_seq, model_dim]");
    }

    #[test]
    fn test_cross_attn_different_src_tgt_lengths() {
        // tgt_seq ≠ src_seq — cross-attention must handle arbitrary lengths.
        let model_dim = 16;
        let num_heads = 4;
        let tgt_seq = 3;
        let src_seq = 10;

        let layer = CrossAttentionLayer::new(model_dim, num_heads).expect("new");
        let decoder_x: Vec<f32> = (0..tgt_seq * model_dim).map(|i| i as f32 * 0.01).collect();
        let encoder_out: Vec<f32> = (0..src_seq * model_dim).map(|i| i as f32 * 0.02).collect();

        let (out, shape) = layer
            .forward(&decoder_x, tgt_seq, &encoder_out, src_seq)
            .expect("forward");

        assert_eq!(out.len(), tgt_seq * model_dim);
        assert_eq!(shape[0], tgt_seq);
        assert_eq!(shape[1], model_dim);
    }

    #[test]
    fn test_cross_attn_tgt_longer_than_src() {
        // tgt_seq > src_seq — also valid for cross-attention.
        let model_dim = 8;
        let num_heads = 2;
        let tgt_seq = 8;
        let src_seq = 2;

        let layer = CrossAttentionLayer::new(model_dim, num_heads).expect("new");
        let decoder_x = vec![0.5_f32; tgt_seq * model_dim];
        let encoder_out = vec![0.5_f32; src_seq * model_dim];

        let (out, _) = layer
            .forward(&decoder_x, tgt_seq, &encoder_out, src_seq)
            .expect("forward");

        assert_eq!(out.len(), tgt_seq * model_dim);
    }

    #[test]
    fn test_cross_attn_equal_seq_lengths() {
        let model_dim = 8;
        let num_heads = 2;
        let seq = 5;

        let layer = CrossAttentionLayer::new(model_dim, num_heads).expect("new");
        let decoder_x = vec![1.0_f32; seq * model_dim];
        let encoder_out = vec![1.0_f32; seq * model_dim];

        let (out, shape) = layer
            .forward(&decoder_x, seq, &encoder_out, seq)
            .expect("forward");

        assert_eq!(out.len(), seq * model_dim);
        assert_eq!(shape, vec![seq, model_dim]);
    }

    #[test]
    fn test_cross_attn_invalid_num_heads_error() {
        // model_dim=8 not divisible by num_heads=3.
        let result = CrossAttentionLayer::new(8, 3);
        assert!(result.is_err(), "invalid num_heads should return an error");
    }

    #[test]
    fn test_cross_attn_zero_num_heads_error() {
        let result = CrossAttentionLayer::new(8, 0);
        assert!(result.is_err(), "zero num_heads should return an error");
    }

    // ── TransformerDecoderBlock ───────────────────────────────────────────

    #[test]
    fn test_decoder_block_residual_shape_preservation() {
        let model_dim = 8;
        let num_heads = 2;
        let ff_dim = 32;
        let tgt_seq = 5;
        let src_seq = 7;

        let block = TransformerDecoderBlock::new(model_dim, num_heads, ff_dim)
            .expect("TransformerDecoderBlock::new");
        let decoder_input: Vec<f32> = (0..tgt_seq * model_dim).map(|i| i as f32 * 0.01).collect();
        let encoder_output = vec![0.1_f32; src_seq * model_dim];

        let (out, shape) = block
            .forward(&decoder_input, tgt_seq, &encoder_output, src_seq, false)
            .expect("forward");

        assert_eq!(
            out.len(),
            decoder_input.len(),
            "residual: output size must equal input size"
        );
        assert_eq!(shape, vec![tgt_seq, model_dim]);
    }

    #[test]
    fn test_decoder_block_training_vs_eval_differ_with_dropout() {
        let model_dim = 8;
        let num_heads = 2;
        let ff_dim = 16;
        let tgt_seq = 4;
        let src_seq = 4;

        let block = TransformerDecoderBlock::new_with_dropout(model_dim, num_heads, ff_dim, 0.5)
            .expect("new with dropout");
        let decoder_input = vec![1.0_f32; tgt_seq * model_dim];
        let encoder_output = vec![0.5_f32; src_seq * model_dim];

        let (out_train, _) = block
            .forward(&decoder_input, tgt_seq, &encoder_output, src_seq, true)
            .expect("train forward");
        let (out_eval, _) = block
            .forward(&decoder_input, tgt_seq, &encoder_output, src_seq, false)
            .expect("eval forward");

        let different = out_train
            .iter()
            .zip(out_eval.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-7);
        assert!(different, "train/eval outputs should differ with dropout");
    }

    // ── TransformerDecoderStack ───────────────────────────────────────────

    #[test]
    fn test_decoder_stack_two_layers_output_shape() {
        let model_dim = 8;
        let num_heads = 2;
        let ff_dim = 32;
        let num_layers = 2;
        let tgt_seq = 6;
        let src_seq = 5;

        let stack = TransformerDecoderStack::new(num_layers, model_dim, num_heads, ff_dim)
            .expect("TransformerDecoderStack::new");
        let decoder_input = vec![0.1_f32; tgt_seq * model_dim];
        let encoder_output = vec![0.2_f32; src_seq * model_dim];

        let (out, shape) = stack
            .forward(&decoder_input, tgt_seq, &encoder_output, src_seq, false)
            .expect("forward");

        assert_eq!(out.len(), tgt_seq * model_dim);
        assert_eq!(shape, vec![tgt_seq, model_dim]);
    }

    #[test]
    fn test_decoder_stack_zero_layers_error() {
        let result = TransformerDecoderStack::new(0, 8, 2, 32);
        assert!(result.is_err(), "zero layers should error");
    }

    #[test]
    fn test_decoder_stack_single_layer() {
        let stack = TransformerDecoderStack::new(1, 16, 4, 64).expect("new");
        let tgt_seq = 3;
        let src_seq = 5;
        let decoder_input = vec![0.0_f32; tgt_seq * 16];
        let encoder_output = vec![0.0_f32; src_seq * 16];

        let (out, shape) = stack
            .forward(&decoder_input, tgt_seq, &encoder_output, src_seq, false)
            .expect("forward");

        assert_eq!(out.len(), tgt_seq * 16);
        assert_eq!(shape[0], tgt_seq);
    }

    // ── Seq2SeqTransformer ────────────────────────────────────────────────

    #[test]
    fn test_seq2seq_output_shape() {
        let model_dim = 8;
        let num_heads = 2;
        let ff_dim = 32;
        let src_seq = 6;
        let tgt_seq = 4;

        let model = Seq2SeqTransformer::new(2, 2, model_dim, num_heads, ff_dim)
            .expect("Seq2SeqTransformer::new");
        let src = vec![0.1_f32; src_seq * model_dim];
        let tgt = vec![0.2_f32; tgt_seq * model_dim];

        let (out, shape) = model
            .forward(&src, src_seq, &tgt, tgt_seq, false)
            .expect("forward");

        assert_eq!(out.len(), tgt_seq * model_dim);
        assert_eq!(shape, vec![tgt_seq, model_dim]);
    }

    #[test]
    fn test_seq2seq_different_src_tgt_seq_lengths() {
        let model_dim = 16;
        let num_heads = 4;
        let ff_dim = 64;
        let src_seq = 10;
        let tgt_seq = 3;

        let model = Seq2SeqTransformer::new(1, 1, model_dim, num_heads, ff_dim).expect("new");
        let src: Vec<f32> = (0..src_seq * model_dim).map(|i| i as f32 * 0.01).collect();
        let tgt: Vec<f32> = (0..tgt_seq * model_dim).map(|i| i as f32 * 0.02).collect();

        let (out, shape) = model.forward(&src, src_seq, &tgt, tgt_seq, false).expect("forward");

        assert_eq!(out.len(), tgt_seq * model_dim);
        assert_eq!(shape[0], tgt_seq);
        assert_eq!(shape[1], model_dim);
    }

    #[test]
    fn test_seq2seq_tgt_longer_than_src() {
        // Decoder can generate longer sequences than the source.
        let model_dim = 8;
        let num_heads = 2;
        let ff_dim = 16;
        let src_seq = 2;
        let tgt_seq = 8;

        let model = Seq2SeqTransformer::new(1, 1, model_dim, num_heads, ff_dim).expect("new");
        let src = vec![0.5_f32; src_seq * model_dim];
        let tgt = vec![0.5_f32; tgt_seq * model_dim];

        let (out, _) = model.forward(&src, src_seq, &tgt, tgt_seq, false).expect("forward");

        assert_eq!(out.len(), tgt_seq * model_dim);
    }

    #[test]
    fn test_seq2seq_invalid_num_heads_error() {
        // model_dim=8 not divisible by num_heads=3.
        let result = Seq2SeqTransformer::new(1, 1, 8, 3, 32);
        assert!(result.is_err(), "invalid num_heads should error");
    }

    #[test]
    fn test_seq2seq_values_are_finite() {
        // With default initialisation all outputs should be finite.
        let model_dim = 8;
        let model = Seq2SeqTransformer::new(2, 2, model_dim, 2, 32).expect("new");
        let src: Vec<f32> = (0..5 * model_dim).map(|i| (i as f32 - 20.0) * 0.05).collect();
        let tgt: Vec<f32> = (0..4 * model_dim).map(|i| (i as f32 - 10.0) * 0.05).collect();

        let (out, _) = model.forward(&src, 5, &tgt, 4, false).expect("forward");

        for &v in &out {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
    }

    #[test]
    fn test_seq2seq_training_vs_eval_no_dropout_identical() {
        // With dropout_prob=0 training and eval modes should produce identical results.
        let model_dim = 8;
        let model = Seq2SeqTransformer::new(1, 1, model_dim, 2, 16).expect("new");
        let src = vec![0.3_f32; 4 * model_dim];
        let tgt = vec![0.3_f32; 3 * model_dim];

        let (out_train, _) = model.forward(&src, 4, &tgt, 3, true).expect("train");
        let (out_eval, _) = model.forward(&src, 4, &tgt, 3, false).expect("eval");

        for (&a, &b) in out_train.iter().zip(out_eval.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "train/eval should be identical with dropout_prob=0: {a} vs {b}"
            );
        }
    }
}
