//! §4 — Conditional Generation.
//!
//! - [`AdaptiveLayerNorm`]          — AdaLN: γ = Wγ·c + 1, β = Wβ·c
//! - [`CrossAttentionConditioning`] — scaled dot-product cross-attention
//! - [`ControlNetAdapter`]          — side network with zero-conv injection
//! - [`InpaintingMask`]             — mask-conditional generation
//! - [`GuidedDiffusionStep`]        — classifier guidance

use tenflowers_core::TensorError;

use super::helpers::{dot, linear_fwd, make_err, softmax};

// ─────────────────────────────────────────────────────────────────────────────

/// Adaptive Layer Normalisation (AdaLN) for conditional diffusion.
///
/// `y = (x − mean) / (std + ε) · (1 + γ) + β`
/// where `γ = W_γ · c`,  `β = W_β · c`.
#[derive(Debug, Clone)]
pub struct AdaptiveLayerNorm {
    pub feature_dim: usize,
    pub cond_dim: usize,
    pub eps: f64,
    w_gamma: Vec<f64>,
    b_gamma: Vec<f64>,
    w_beta: Vec<f64>,
    b_beta: Vec<f64>,
}

impl AdaptiveLayerNorm {
    /// Create AdaLN with zero-initialised projections.
    pub fn new(feature_dim: usize, cond_dim: usize) -> Self {
        Self {
            feature_dim,
            cond_dim,
            eps: 1e-5,
            w_gamma: vec![0.0; cond_dim * feature_dim],
            b_gamma: vec![0.0; feature_dim],
            w_beta: vec![0.0; cond_dim * feature_dim],
            b_beta: vec![0.0; feature_dim],
        }
    }

    /// Apply AdaLN: normalise x then scale-shift using conditioning c.
    pub fn forward(&self, x: &[f64], c: &[f64]) -> Result<Vec<f64>, TensorError> {
        if x.len() != self.feature_dim {
            return Err(make_err("AdaptiveLayerNorm: x dimension mismatch"));
        }
        if c.len() != self.cond_dim {
            return Err(make_err("AdaptiveLayerNorm: c dimension mismatch"));
        }

        let mean = x.iter().sum::<f64>() / self.feature_dim as f64;
        let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / self.feature_dim as f64;
        let std = (var + self.eps).sqrt();
        let x_norm: Vec<f64> = x.iter().map(|&v| (v - mean) / std).collect();

        let gamma = linear_fwd(c, &self.w_gamma, &self.b_gamma, self.feature_dim);
        let beta = linear_fwd(c, &self.w_beta, &self.b_beta, self.feature_dim);

        let out: Vec<f64> = x_norm
            .iter()
            .zip(gamma.iter())
            .zip(beta.iter())
            .map(|((&xn, &g), &b)| xn * (1.0 + g) + b)
            .collect();

        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Cross-attention conditioning between latent tokens and conditioning tokens.
///
/// `Attn(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V`
#[derive(Debug, Clone)]
pub struct CrossAttentionConditioning {
    pub query_dim: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
}

impl CrossAttentionConditioning {
    /// Create a new cross-attention conditioning module.
    pub fn new(query_dim: usize, kv_dim: usize, head_dim: usize) -> Self {
        Self {
            query_dim,
            kv_dim,
            head_dim,
        }
    }

    /// Apply single-head cross-attention.
    ///
    /// `q`: query vector (query_dim)
    /// `keys`: [n_keys, kv_dim] flat row-major
    /// `values`: [n_keys, kv_dim] flat row-major
    pub fn attend(
        &self,
        q: &[f64],
        keys: &[f64],
        values: &[f64],
        n_keys: usize,
    ) -> Result<Vec<f64>, TensorError> {
        if q.len() != self.query_dim {
            return Err(make_err(
                "CrossAttentionConditioning: query dimension mismatch",
            ));
        }
        if keys.len() != n_keys * self.kv_dim {
            return Err(make_err(
                "CrossAttentionConditioning: keys dimension mismatch",
            ));
        }
        if values.len() != n_keys * self.kv_dim {
            return Err(make_err(
                "CrossAttentionConditioning: values dimension mismatch",
            ));
        }

        let q_hd: Vec<f64> = q.iter().take(self.head_dim).cloned().collect();
        let scale = (self.head_dim as f64).sqrt().max(1e-8);

        let scores: Vec<f64> = (0..n_keys)
            .map(|i| {
                let key_i: Vec<f64> = keys[i * self.kv_dim..]
                    .iter()
                    .take(self.head_dim)
                    .cloned()
                    .collect();
                dot(&q_hd, &key_i) / scale
            })
            .collect();

        let weights = softmax(&scores);

        let out_dim = self.kv_dim;
        let mut out = vec![0.0_f64; out_dim];
        for (i, &w) in weights.iter().enumerate() {
            for j in 0..out_dim {
                out[j] += w * values[i * out_dim + j];
            }
        }

        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// ControlNet adapter: side network with zero-conv injection.
///
/// Injects encoder features via a zero-initialised gate per block.
#[derive(Debug, Clone)]
pub struct ControlNetAdapter {
    pub feature_dim: usize,
    pub num_blocks: usize,
    pub gates: Vec<f64>,
}

impl ControlNetAdapter {
    /// Create a new ControlNet adapter with zero gates.
    pub fn new(feature_dim: usize, num_blocks: usize) -> Self {
        Self {
            feature_dim,
            num_blocks,
            gates: vec![0.0; num_blocks],
        }
    }

    /// Inject encoder block output: `main + gate * encoder_out`.
    pub fn inject(
        &self,
        encoder_out: &[f64],
        block_idx: usize,
        main_features: &[f64],
    ) -> Result<Vec<f64>, TensorError> {
        if block_idx >= self.num_blocks {
            return Err(make_err("ControlNetAdapter: block_idx out of range"));
        }
        if encoder_out.len() != self.feature_dim || main_features.len() != self.feature_dim {
            return Err(make_err("ControlNetAdapter: feature dimension mismatch"));
        }
        let gate = self.gates[block_idx];
        let out = main_features
            .iter()
            .zip(encoder_out.iter())
            .map(|(&m, &e)| m + gate * e)
            .collect();
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Inpainting mask for mask-conditional generation.
///
/// Known pixels guide the noisy region: at each step, known pixel values are
/// reinserted after the denoising step.
#[derive(Debug, Clone)]
pub struct InpaintingMask {
    pub mask: Vec<bool>,
    pub known: Vec<f64>,
}

impl InpaintingMask {
    /// Create an inpainting mask.
    pub fn new(mask: Vec<bool>, known: Vec<f64>) -> Result<Self, TensorError> {
        if mask.len() != known.len() {
            return Err(make_err(
                "InpaintingMask: mask and known must have equal length",
            ));
        }
        Ok(Self { mask, known })
    }

    /// Apply mask: replace generated values with known values where mask=true.
    pub fn apply(&self, x_t: &[f64], x_known_noisy: &[f64]) -> Result<Vec<f64>, TensorError> {
        if x_t.len() != self.mask.len() || x_known_noisy.len() != self.mask.len() {
            return Err(make_err("InpaintingMask: dimension mismatch"));
        }
        let out: Vec<f64> = x_t
            .iter()
            .zip(x_known_noisy.iter())
            .zip(self.mask.iter())
            .map(|((&xt, &xk), &m)| if m { xk } else { xt })
            .collect();
        Ok(out)
    }

    /// Number of known pixels.
    pub fn num_known(&self) -> usize {
        self.mask.iter().filter(|&&m| m).count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Guided diffusion step with classifier guidance.
///
/// `ε_guided = ε − √(1 − ᾱ_t) · w · ∇_x log p(y | x_t)`
#[derive(Debug, Clone)]
pub struct GuidedDiffusionStep {
    pub guidance_weight: f64,
}

impl GuidedDiffusionStep {
    /// Create a new guided diffusion step.
    pub fn new(guidance_weight: f64) -> Self {
        Self { guidance_weight }
    }

    /// Apply classifier guidance.
    pub fn apply(
        &self,
        eps: &[f64],
        grad_log_p: &[f64],
        alpha_bar_t: f64,
    ) -> Result<Vec<f64>, TensorError> {
        if eps.len() != grad_log_p.len() {
            return Err(make_err("GuidedDiffusionStep: dimension mismatch"));
        }
        let scale = (1.0 - alpha_bar_t).sqrt() * self.guidance_weight;
        let out: Vec<f64> = eps
            .iter()
            .zip(grad_log_p.iter())
            .map(|(&e, &g)| e - scale * g)
            .collect();
        Ok(out)
    }
}
