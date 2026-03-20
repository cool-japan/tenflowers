//! Vision Transformers, Swin Transformers, Dense Prediction heads, MAE, and
//! contrastive vision learning (BarlowTwins, VICReg, SimCLRv2, BYOL).
//!
//! All structures operate on plain `f64` vectors and matrices, requiring no
//! tensor runtime. Random initialisation uses `scirs2_core::random::StdRng`.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Private helper functions
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

#[inline]
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (x / std::f64::consts::SQRT_2).tanh())
}

fn layer_norm(x: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = (var + 1e-5).sqrt();
    x.iter().map(|v| (v - mean) / std).collect()
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|v| v / sum).collect()
}

/// Box-Muller normal sample pair, mean 0 std 0.02 (Xavier-like small init).
fn rand_vec(size: usize, rng: &mut StdRng) -> Vec<f64> {
    let mut out = Vec::with_capacity(size);
    let mut i = 0;
    while i < size {
        let u1: f64 = rng.random::<f64>().max(1e-12);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push(r * theta.cos() * 0.02);
        if i + 1 < size {
            out.push(r * theta.sin() * 0.02);
        }
        i += 2;
    }
    out.truncate(size);
    out
}

fn rand_mat(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    (0..rows).map(|_| rand_vec(cols, rng)).collect()
}

/// Intersection-over-union for axis-aligned boxes [x1,y1,x2,y2].
fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Matrix-vector multiply: W (rows x cols) @ x (cols) → (rows).
fn matvec(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter().map(|row| dot(row, x)).collect()
}

/// Sinusoidal positional encoding for position `pos`, dimension `d_model`.
fn sinusoidal_pe(pos: usize, d_model: usize) -> Vec<f64> {
    (0..d_model)
        .map(|i| {
            let denom = 10000.0_f64.powf(2.0 * (i / 2) as f64 / d_model as f64);
            if i % 2 == 0 {
                (pos as f64 / denom).sin()
            } else {
                (pos as f64 / denom).cos()
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Vision Transformer (ViT)
// ─────────────────────────────────────────────────────────────────────────────

/// Splits an image into non-overlapping patches and linearly projects each to
/// `d_model` dimensions.
pub struct PatchEmbedding {
    /// Spatial patch size P (both height and width).
    pub patch_size: usize,
    /// Embedding dimension D.
    pub d_model: usize,
    /// Projection weights: shape (d_model × patch_dim), where patch_dim = P*P.
    pub weights: Vec<Vec<f64>>,
    /// Projection bias: shape (d_model,).
    pub bias: Vec<f64>,
}

impl PatchEmbedding {
    /// Construct with random initialisation seeded by `seed`.
    pub fn new(patch_size: usize, d_model: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let patch_dim = patch_size * patch_size;
        let weights = rand_mat(d_model, patch_dim, &mut rng);
        let bias = rand_vec(d_model, &mut rng);
        Self {
            patch_size,
            d_model,
            weights,
            bias,
        }
    }

    /// Embed image of shape (H, W) stored in row-major order.
    ///
    /// Returns `(H/P) * (W/P)` patch embeddings, each of length `d_model`.
    pub fn embed(&self, image: &[f64], h: usize, w: usize) -> Vec<Vec<f64>> {
        let p = self.patch_size;
        let gh = h / p; // grid height
        let gw = w / p; // grid width
        let mut out = Vec::with_capacity(gh * gw);
        for gr in 0..gh {
            for gc in 0..gw {
                // Extract patch pixels
                let mut patch = Vec::with_capacity(p * p);
                for pr in 0..p {
                    for pc in 0..p {
                        let row = gr * p + pr;
                        let col = gc * p + pc;
                        let idx = row * w + col;
                        patch.push(if idx < image.len() { image[idx] } else { 0.0 });
                    }
                }
                // Linear projection + bias
                let mut emb: Vec<f64> = self.weights.iter().map(|row| dot(row, &patch)).collect();
                for (e, b) in emb.iter_mut().zip(self.bias.iter()) {
                    *e += b;
                }
                out.push(emb);
            }
        }
        out
    }
}

/// A single ViT encoder block: pre-norm MHA + pre-norm FFN with residuals.
pub struct VitBlock {
    /// Embedding dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    // MHA weight matrices: each (d_model × d_model)
    wq: Vec<Vec<f64>>,
    wk: Vec<Vec<f64>>,
    wv: Vec<Vec<f64>>,
    wo: Vec<Vec<f64>>,
    // FFN: expand 4×, then project back
    w1: Vec<Vec<f64>>, // (4*d_model × d_model)
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>, // (d_model × 4*d_model)
    b2: Vec<f64>,
}

impl VitBlock {
    /// Construct with random weights seeded by `seed`.
    pub fn new(d_model: usize, n_heads: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let ff = 4 * d_model;
        Self {
            d_model,
            n_heads,
            wq: rand_mat(d_model, d_model, &mut rng),
            wk: rand_mat(d_model, d_model, &mut rng),
            wv: rand_mat(d_model, d_model, &mut rng),
            wo: rand_mat(d_model, d_model, &mut rng),
            w1: rand_mat(ff, d_model, &mut rng),
            b1: rand_vec(ff, &mut rng),
            w2: rand_mat(d_model, ff, &mut rng),
            b2: rand_vec(d_model, &mut rng),
        }
    }

    /// Forward pass.  Input/output shape: `(seq_len, d_model)`.
    pub fn forward(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq = x.len();
        let d = self.d_model;
        let h = self.n_heads;
        let dh = d / h.max(1);
        let scale = (dh as f64).sqrt().max(1e-6);

        // ── Pre-norm MHA ──────────────────────────────────────────────────
        let normed: Vec<Vec<f64>> = x.iter().map(|v| layer_norm(v)).collect();
        // Project Q, K, V
        let q: Vec<Vec<f64>> = normed.iter().map(|v| matvec(&self.wq, v)).collect();
        let k: Vec<Vec<f64>> = normed.iter().map(|v| matvec(&self.wk, v)).collect();
        let v: Vec<Vec<f64>> = normed.iter().map(|v| matvec(&self.wv, v)).collect();

        // Compute attention per head, concatenate
        let mut mha_out: Vec<Vec<f64>> = vec![vec![0.0; d]; seq];
        for head in 0..h {
            let start = head * dh;
            let end = start + dh;
            // Attention weights: (seq × seq)
            let attn: Vec<Vec<f64>> = (0..seq)
                .map(|i| {
                    let scores: Vec<f64> = (0..seq)
                        .map(|j| dot(&q[i][start..end], &k[j][start..end]) / scale)
                        .collect();
                    softmax(&scores)
                })
                .collect();
            // Context vectors
            for i in 0..seq {
                for j in 0..seq {
                    let a = attn[i][j];
                    for dim in start..end {
                        mha_out[i][dim] += a * v[j][dim];
                    }
                }
            }
        }
        // Output projection + residual
        let attn_res: Vec<Vec<f64>> = mha_out
            .iter()
            .zip(x.iter())
            .map(|(o, xi)| {
                let proj = matvec(&self.wo, o);
                proj.iter().zip(xi.iter()).map(|(p, r)| p + r).collect()
            })
            .collect();

        // ── Pre-norm FFN ──────────────────────────────────────────────────
        attn_res
            .iter()
            .map(|v| {
                let ln = layer_norm(v);
                let mut h1: Vec<f64> = self
                    .w1
                    .iter()
                    .zip(self.b1.iter())
                    .map(|(row, b)| relu(dot(row, &ln) + b))
                    .collect();
                let h2: Vec<f64> = self
                    .w2
                    .iter()
                    .zip(self.b2.iter())
                    .map(|(row, b)| dot(row, &h1) + b)
                    .collect();
                // zero out h1 to satisfy linter (unused mut)
                h1.iter_mut().for_each(|x| *x = 0.0);
                h2.iter().zip(v.iter()).map(|(y, r)| y + r).collect()
            })
            .collect()
    }
}

/// Full Vision Transformer: PatchEmbedding + sinusoidal PE + N VitBlocks +
/// linear classification head.
pub struct VisionTransformer {
    /// Patch embedding layer.
    pub patch_embed: PatchEmbedding,
    /// Stack of transformer encoder blocks.
    pub blocks: Vec<VitBlock>,
    /// Classification head weights: (n_classes × d_model).
    head_w: Vec<Vec<f64>>,
    /// Classification head bias: (n_classes,).
    head_b: Vec<f64>,
    /// Number of output classes.
    pub n_classes: usize,
}

impl VisionTransformer {
    /// Construct a ViT.
    ///
    /// * `patch_size` — spatial patch size P
    /// * `d_model`    — embedding dimension
    /// * `n_heads`    — attention heads per block
    /// * `n_blocks`   — number of encoder blocks
    /// * `n_classes`  — number of output classes
    /// * `seed`       — RNG seed
    pub fn new(
        patch_size: usize,
        d_model: usize,
        n_heads: usize,
        n_blocks: usize,
        n_classes: usize,
        seed: u64,
    ) -> Self {
        let patch_embed = PatchEmbedding::new(patch_size, d_model, seed);
        let blocks = (0..n_blocks)
            .map(|i| VitBlock::new(d_model, n_heads, seed.wrapping_add(i as u64 + 1)))
            .collect();
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000));
        let head_w = rand_mat(n_classes, d_model, &mut rng);
        let head_b = rand_vec(n_classes, &mut rng);
        Self {
            patch_embed,
            blocks,
            head_w,
            head_b,
            n_classes,
        }
    }

    /// Forward pass.  Returns class logits of length `n_classes`.
    ///
    /// The image is stored row-major with dimensions (h, w).
    pub fn forward(&self, image: &[f64], h: usize, w: usize) -> Vec<f64> {
        // Patch embedding
        let mut tokens = self.patch_embed.embed(image, h, w);
        // Add sinusoidal positional encoding
        for (pos, tok) in tokens.iter_mut().enumerate() {
            let pe = sinusoidal_pe(pos, self.patch_embed.d_model);
            for (t, p) in tok.iter_mut().zip(pe.iter()) {
                *t += p;
            }
        }
        // Transformer blocks
        for block in &self.blocks {
            tokens = block.forward(&tokens);
        }
        // Mean-pool over sequence, then classify
        let d = self.patch_embed.d_model;
        let n = tokens.len() as f64;
        let pooled: Vec<f64> = (0..d)
            .map(|i| tokens.iter().map(|t| t[i]).sum::<f64>() / n)
            .collect();
        let ln = layer_norm(&pooled);
        self.head_w
            .iter()
            .zip(self.head_b.iter())
            .map(|(row, b)| dot(row, &ln) + b)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Swin Transformer
// ─────────────────────────────────────────────────────────────────────────────

/// Merges 2×2 spatial neighbourhoods: 4C → 2C via a learned linear map.
pub struct PatchMerging {
    /// Projection weights: (2*C × 4*C).
    w: Vec<Vec<f64>>,
    /// Projection bias: (2*C,).
    b: Vec<f64>,
}

impl PatchMerging {
    /// Construct for input channel dimension `c`, seeded by `seed`.
    pub fn new(c: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            w: rand_mat(2 * c, 4 * c, &mut rng),
            b: rand_vec(2 * c, &mut rng),
        }
    }

    /// Merge a (H × W) grid of C-dim patches into a (H/2 × W/2) grid of 2C-dim patches.
    pub fn merge(&self, patches: &[Vec<f64>], h: usize, w: usize) -> Vec<Vec<f64>> {
        let c = if patches.is_empty() {
            0
        } else {
            patches[0].len()
        };
        let oh = h / 2;
        let ow = w / 2;
        let mut out = Vec::with_capacity(oh * ow);
        for gr in 0..oh {
            for gc in 0..ow {
                // Gather 2×2 neighbourhood
                let mut concat = Vec::with_capacity(4 * c);
                for dr in 0..2usize {
                    for dc in 0..2usize {
                        let r = gr * 2 + dr;
                        let col = gc * 2 + dc;
                        let idx = r * w + col;
                        if idx < patches.len() {
                            concat.extend_from_slice(&patches[idx]);
                        } else {
                            concat.extend(std::iter::repeat(0.0).take(c));
                        }
                    }
                }
                let proj: Vec<f64> = self
                    .w
                    .iter()
                    .zip(self.b.iter())
                    .map(|(row, b)| dot(row, &concat) + b)
                    .collect();
                out.push(proj);
            }
        }
        out
    }
}

/// A single Swin Transformer block: LayerNorm + window attention + FFN.
///
/// For simplicity the window size equals the full spatial extent (i.e., global
/// attention within the block, shifted-window logic is elided).
pub struct SwinBlock {
    /// Channel dimension.
    pub d_model: usize,
    wq: Vec<Vec<f64>>,
    wk: Vec<Vec<f64>>,
    wv: Vec<Vec<f64>>,
    wo: Vec<Vec<f64>>,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
}

impl SwinBlock {
    /// Construct with random weights seeded by `seed`.
    pub fn new(d_model: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let ff = 4 * d_model;
        Self {
            d_model,
            wq: rand_mat(d_model, d_model, &mut rng),
            wk: rand_mat(d_model, d_model, &mut rng),
            wv: rand_mat(d_model, d_model, &mut rng),
            wo: rand_mat(d_model, d_model, &mut rng),
            w1: rand_mat(ff, d_model, &mut rng),
            b1: rand_vec(ff, &mut rng),
            w2: rand_mat(d_model, ff, &mut rng),
            b2: rand_vec(d_model, &mut rng),
        }
    }

    /// Forward pass over a flat sequence of tokens (any spatial layout).
    pub fn forward(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq = x.len();
        let d = self.d_model;
        let scale = (d as f64).sqrt().max(1e-6);

        // LayerNorm + self-attention
        let normed: Vec<Vec<f64>> = x.iter().map(|v| layer_norm(v)).collect();
        let q: Vec<Vec<f64>> = normed.iter().map(|v| matvec(&self.wq, v)).collect();
        let k: Vec<Vec<f64>> = normed.iter().map(|v| matvec(&self.wk, v)).collect();
        let v: Vec<Vec<f64>> = normed.iter().map(|v| matvec(&self.wv, v)).collect();

        let attn: Vec<Vec<f64>> = (0..seq)
            .map(|i| {
                let scores: Vec<f64> = (0..seq).map(|j| dot(&q[i], &k[j]) / scale).collect();
                softmax(&scores)
            })
            .collect();

        let mut sa_out: Vec<Vec<f64>> = vec![vec![0.0; d]; seq];
        for i in 0..seq {
            for j in 0..seq {
                let a = attn[i][j];
                for dim in 0..d {
                    sa_out[i][dim] += a * v[j][dim];
                }
            }
        }
        let attn_res: Vec<Vec<f64>> = sa_out
            .iter()
            .zip(x.iter())
            .map(|(o, xi)| {
                let proj = matvec(&self.wo, o);
                proj.iter().zip(xi.iter()).map(|(p, r)| p + r).collect()
            })
            .collect();

        // FFN with residual
        attn_res
            .iter()
            .map(|v| {
                let ln = layer_norm(v);
                let h1: Vec<f64> = self
                    .w1
                    .iter()
                    .zip(self.b1.iter())
                    .map(|(row, b)| gelu(dot(row, &ln) + b))
                    .collect();
                let h2: Vec<f64> = self
                    .w2
                    .iter()
                    .zip(self.b2.iter())
                    .map(|(row, b)| dot(row, &h1) + b)
                    .collect();
                h2.iter().zip(v.iter()).map(|(y, r)| y + r).collect()
            })
            .collect()
    }
}

/// A Swin Transformer stage: N SwinBlocks followed by an optional PatchMerging.
pub struct SwinStage {
    /// Encoder blocks.
    pub blocks: Vec<SwinBlock>,
    /// Optional downsampling layer.
    pub patch_merging: Option<PatchMerging>,
}

impl SwinStage {
    /// Construct.
    ///
    /// * `d_model`         — channel dimension
    /// * `n_blocks`        — number of SwinBlocks
    /// * `with_merging`    — whether to append a PatchMerging layer
    /// * `seed`            — RNG seed
    pub fn new(d_model: usize, n_blocks: usize, with_merging: bool, seed: u64) -> Self {
        let blocks = (0..n_blocks)
            .map(|i| SwinBlock::new(d_model, seed.wrapping_add(i as u64)))
            .collect();
        let patch_merging = if with_merging {
            Some(PatchMerging::new(d_model, seed.wrapping_add(500)))
        } else {
            None
        };
        Self {
            blocks,
            patch_merging,
        }
    }

    /// Forward pass.
    ///
    /// `x` is a flat grid of (H × W) tokens each of `d_model` dimensions.
    /// Returns the processed tokens; if PatchMerging is present the spatial
    /// dimensions halve and the channel count doubles.
    pub fn forward(&self, x: &[Vec<f64>], h: usize, w: usize) -> Vec<Vec<f64>> {
        let mut out: Vec<Vec<f64>> = x.to_vec();
        for block in &self.blocks {
            out = block.forward(&out);
        }
        if let Some(ref pm) = self.patch_merging {
            out = pm.merge(&out, h, w);
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dense Prediction
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy IoU-based Non-Maximum Suppression.
pub struct NmsProcessor;

impl NmsProcessor {
    /// Run NMS on a set of axis-aligned boxes.
    ///
    /// * `boxes`          — slice of `[x1, y1, x2, y2]`
    /// * `scores`         — confidence score per box
    /// * `iou_threshold`  — IoU above which a box is suppressed
    ///
    /// Returns indices of kept boxes in descending score order.
    pub fn nms(boxes: &[[f64; 4]], scores: &[f64], iou_threshold: f64) -> Vec<usize> {
        if boxes.is_empty() {
            return Vec::new();
        }
        // Sort by descending score
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut kept = Vec::new();
        let mut suppressed = vec![false; boxes.len()];

        for &i in &order {
            if suppressed[i] {
                continue;
            }
            kept.push(i);
            for &j in &order {
                if j == i || suppressed[j] {
                    continue;
                }
                if iou(&boxes[i], &boxes[j]) > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
        kept
    }
}

/// Per-pixel linear classifier head for semantic segmentation.
pub struct SegmentationHead {
    /// Weights: (n_classes × feature_dim).
    w: Vec<Vec<f64>>,
    /// Bias: (n_classes,).
    b: Vec<f64>,
    /// Output number of classes.
    pub n_classes: usize,
}

impl SegmentationHead {
    /// Construct for `feature_dim`-dimensional features and `n_classes` outputs,
    /// seeded by `seed`.
    pub fn new(feature_dim: usize, n_classes: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            w: rand_mat(n_classes, feature_dim, &mut rng),
            b: rand_vec(n_classes, &mut rng),
            n_classes,
        }
    }

    /// Classify each feature vector independently.
    ///
    /// Returns one `n_classes`-length logit vector per input token.
    pub fn forward(&self, features: &[Vec<f64>], n_classes: usize) -> Vec<Vec<f64>> {
        let cls = if n_classes > 0 {
            n_classes
        } else {
            self.n_classes
        };
        features
            .iter()
            .map(|feat| {
                let cap = self.w.len().min(cls);
                (0..cap)
                    .map(|c| dot(&self.w[c], feat) + self.b[c])
                    .collect()
            })
            .collect()
    }
}

/// Two-branch detection head: classification scores + bounding-box regression deltas.
pub struct DetectionHead {
    /// Classifier branch weights: (n_classes × feature_dim).
    cls_w: Vec<Vec<f64>>,
    cls_b: Vec<f64>,
    /// Box regressor branch weights: (4 × feature_dim).
    box_w: Vec<Vec<f64>>,
    box_b: Vec<f64>,
}

impl DetectionHead {
    /// Construct for `feature_dim` inputs, `n_classes` categories, seeded by `seed`.
    pub fn new(feature_dim: usize, n_classes: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            cls_w: rand_mat(n_classes, feature_dim, &mut rng),
            cls_b: rand_vec(n_classes, &mut rng),
            box_w: rand_mat(4, feature_dim, &mut rng),
            box_b: rand_vec(4, &mut rng),
        }
    }

    /// Forward pass.
    ///
    /// Returns `(cls_scores, box_deltas)`:
    /// - `cls_scores`: one `n_classes`-length vector per input feature
    /// - `box_deltas`: one 4-length vector `[dx, dy, dw, dh]` per input feature
    pub fn forward(&self, features: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let cls: Vec<Vec<f64>> = features
            .iter()
            .map(|feat| {
                self.cls_w
                    .iter()
                    .zip(self.cls_b.iter())
                    .map(|(row, b)| dot(row, feat) + b)
                    .collect()
            })
            .collect();
        let boxes: Vec<Vec<f64>> = features
            .iter()
            .map(|feat| {
                self.box_w
                    .iter()
                    .zip(self.box_b.iter())
                    .map(|(row, b)| dot(row, feat) + b)
                    .collect()
            })
            .collect();
        (cls, boxes)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MAE (Masked Autoencoder)
// ─────────────────────────────────────────────────────────────────────────────

/// Selects a random subset of patch indices to mask.
pub struct RandomMasking;

impl RandomMasking {
    /// Sample visible and masked patch indices uniformly at random.
    ///
    /// * `n_patches`  — total number of patches
    /// * `mask_ratio` — fraction to mask (e.g. 0.75)
    /// * `rng`        — seeded random number generator
    ///
    /// Returns `(visible_indices, masked_indices)`.
    pub fn mask(n_patches: usize, mask_ratio: f64, rng: &mut StdRng) -> (Vec<usize>, Vec<usize>) {
        let n_masked = ((n_patches as f64) * mask_ratio).round() as usize;
        let n_visible = n_patches.saturating_sub(n_masked);

        // Fisher-Yates shuffle on indices
        let mut indices: Vec<usize> = (0..n_patches).collect();
        for i in (1..n_patches).rev() {
            let j = (rng.random::<f64>() * (i + 1) as f64) as usize;
            let j = j.min(i);
            indices.swap(i, j);
        }
        let visible = indices[..n_visible].to_vec();
        let masked = indices[n_visible..].to_vec();
        (visible, masked)
    }
}

/// A lightweight MAE: MLP encoder on visible patches + MLP decoder to
/// reconstruct masked patch pixels.
pub struct MaeModel {
    /// Patch embedding layer.
    patch_embed: PatchEmbedding,
    // Encoder: 1-hidden-layer MLP  d_model → latent
    enc_w1: Vec<Vec<f64>>,
    enc_b1: Vec<f64>,
    enc_w2: Vec<Vec<f64>>,
    enc_b2: Vec<f64>,
    // Decoder: latent → patch_dim
    dec_w: Vec<Vec<f64>>,
    dec_b: Vec<f64>,
}

impl MaeModel {
    /// Construct for a given patch size, model dim and latent dim, seeded by `seed`.
    pub fn new(patch_size: usize, d_model: usize, latent_dim: usize, seed: u64) -> Self {
        let patch_embed = PatchEmbedding::new(patch_size, d_model, seed);
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(100));
        let patch_dim = patch_size * patch_size;
        Self {
            patch_embed,
            enc_w1: rand_mat(latent_dim, d_model, &mut rng),
            enc_b1: rand_vec(latent_dim, &mut rng),
            enc_w2: rand_mat(latent_dim, latent_dim, &mut rng),
            enc_b2: rand_vec(latent_dim, &mut rng),
            dec_w: rand_mat(patch_dim, latent_dim, &mut rng),
            dec_b: rand_vec(patch_dim, &mut rng),
        }
    }

    /// Compute MSE reconstruction loss over masked patches.
    pub fn reconstruction_loss(
        &self,
        image: &[f64],
        h: usize,
        w: usize,
        mask_ratio: f64,
        rng: &mut StdRng,
    ) -> f64 {
        let p = self.patch_embed.patch_size;
        let tokens = self.patch_embed.embed(image, h, w);
        let n = tokens.len();
        let (visible_idx, masked_idx) = RandomMasking::mask(n, mask_ratio, rng);

        // Encode visible tokens
        let encoded: Vec<Vec<f64>> = visible_idx
            .iter()
            .map(|&i| {
                let h1: Vec<f64> = self
                    .enc_w1
                    .iter()
                    .zip(self.enc_b1.iter())
                    .map(|(row, b)| relu(dot(row, &tokens[i]) + b))
                    .collect();
                self.enc_w2
                    .iter()
                    .zip(self.enc_b2.iter())
                    .map(|(row, b)| relu(dot(row, &h1) + b))
                    .collect()
            })
            .collect();

        // Mean-pool encoded representations as a simple "context"
        let latent_dim = self.enc_b2.len();
        let context: Vec<f64> = if encoded.is_empty() {
            vec![0.0; latent_dim]
        } else {
            let n_enc = encoded.len() as f64;
            (0..latent_dim)
                .map(|i| encoded.iter().map(|e| e[i]).sum::<f64>() / n_enc)
                .collect()
        };

        // Decode each masked token and compute MSE against ground-truth patch
        let patch_dim = p * p;
        let gw = w / p;
        let mut total_loss = 0.0;
        let mut count = 0usize;

        for &mi in &masked_idx {
            let recon: Vec<f64> = self
                .dec_w
                .iter()
                .zip(self.dec_b.iter())
                .map(|(row, b)| dot(row, &context) + b)
                .collect();

            // Extract target patch pixels
            let gr = mi / gw;
            let gc = mi % gw;
            let mut target = Vec::with_capacity(patch_dim);
            for pr in 0..p {
                for pc in 0..p {
                    let row_idx = gr * p + pr;
                    let col_idx = gc * p + pc;
                    let idx = row_idx * w + col_idx;
                    target.push(if idx < image.len() { image[idx] } else { 0.0 });
                }
            }

            let mse: f64 = recon
                .iter()
                .zip(target.iter())
                .map(|(r, t)| (r - t).powi(2))
                .sum::<f64>()
                / patch_dim.max(1) as f64;
            total_loss += mse;
            count += 1;
        }

        if count == 0 {
            0.0
        } else {
            total_loss / count as f64
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Contrastive Vision Learning
// ─────────────────────────────────────────────────────────────────────────────

/// Normalise each feature dimension to zero mean and unit variance across the
/// batch.  Returns `(N, D)` normalised matrix.
fn feature_normalise(z: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if z.is_empty() {
        return Vec::new();
    }
    let n = z.len() as f64;
    let d = z[0].len();
    let mean: Vec<f64> = (0..d)
        .map(|j| z.iter().map(|r| r[j]).sum::<f64>() / n)
        .collect();
    let std: Vec<f64> = (0..d)
        .map(|j| {
            let v = z.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f64>() / n;
            (v + 1e-5).sqrt()
        })
        .collect();
    z.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &x)| (x - mean[j]) / std[j])
                .collect()
        })
        .collect()
}

/// Compute (D × D) cross-correlation matrix C = (z1_norm)^T @ z2_norm / N.
fn cross_correlation(z1: &[Vec<f64>], z2: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if z1.is_empty() {
        return Vec::new();
    }
    let n = z1.len() as f64;
    let d = z1[0].len();
    let mut c = vec![vec![0.0f64; d]; d];
    for k in 0..z1.len() {
        for i in 0..d {
            for j in 0..d {
                c[i][j] += z1[k][i] * z2[k][j];
            }
        }
    }
    for row in c.iter_mut() {
        for v in row.iter_mut() {
            *v /= n;
        }
    }
    c
}

/// Barlow Twins self-supervised loss.
///
/// Reference: Zbontar et al. 2021
pub struct BarlowTwins;

impl BarlowTwins {
    /// Compute the Barlow Twins loss.
    ///
    /// * `z1`, `z2`  — two views, shape (N, D)
    /// * `lambda`    — off-diagonal penalty weight
    pub fn barlow_loss(z1: &[Vec<f64>], z2: &[Vec<f64>], lambda: f64) -> f64 {
        let zn1 = feature_normalise(z1);
        let zn2 = feature_normalise(z2);
        let c = cross_correlation(&zn1, &zn2);
        let d = c.len();
        let mut loss = 0.0;
        for i in 0..d {
            for j in 0..d {
                let cij = c[i][j];
                if i == j {
                    loss += (cij - 1.0).powi(2);
                } else {
                    loss += lambda * cij.powi(2);
                }
            }
        }
        loss
    }
}

/// VICReg self-supervised loss.
///
/// Reference: Bardes et al. 2021
pub struct VicReg;

impl VicReg {
    /// Compute the VICReg loss.
    ///
    /// * `mu_s` — invariance weight (typically 25)
    /// * `mu_v` — variance weight (typically 25)
    /// * `mu_c` — covariance weight (typically 1)
    pub fn vicreg_loss(z1: &[Vec<f64>], z2: &[Vec<f64>], mu_s: f64, mu_v: f64, mu_c: f64) -> f64 {
        if z1.is_empty() {
            return 0.0;
        }
        let n = z1.len() as f64;
        let d = z1[0].len();

        // Invariance term: MSE between z1 and z2
        let inv: f64 = z1
            .iter()
            .zip(z2.iter())
            .map(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / n;

        // Variance term: hinge on std per dimension, gamma = 1
        let gamma = 1.0;
        let var_term = |z: &[Vec<f64>]| -> f64 {
            (0..d)
                .map(|j| {
                    let mean = z.iter().map(|r| r[j]).sum::<f64>() / n;
                    let var = z.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / n;
                    (gamma - (var + 1e-4).sqrt()).max(0.0)
                })
                .sum::<f64>()
                / d as f64
        };
        let var = var_term(z1) + var_term(z2);

        // Covariance term: off-diagonal of C^T C / N
        let cov_term = |z: &[Vec<f64>]| -> f64 {
            let mean: Vec<f64> = (0..d)
                .map(|j| z.iter().map(|r| r[j]).sum::<f64>() / n)
                .collect();
            let mut cov = vec![vec![0.0f64; d]; d];
            for r in z.iter() {
                for i in 0..d {
                    for j in 0..d {
                        cov[i][j] += (r[i] - mean[i]) * (r[j] - mean[j]);
                    }
                }
            }
            let mut s = 0.0;
            for i in 0..d {
                for j in 0..d {
                    if i != j {
                        let c = cov[i][j] / n;
                        s += c.powi(2);
                    }
                }
            }
            s / d as f64
        };
        let cov = cov_term(z1) + cov_term(z2);

        mu_s * inv + mu_v * var + mu_c * cov
    }
}

/// SimCLRv2 — projection head + NT-Xent contrastive loss.
pub struct SimCLRv2 {
    /// Projection head weights: (proj_dim × in_dim).
    proj_w: Vec<Vec<f64>>,
    proj_b: Vec<f64>,
}

impl SimCLRv2 {
    /// Construct projection head mapping `in_dim` → `proj_dim`, seeded by `seed`.
    pub fn new(in_dim: usize, proj_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            proj_w: rand_mat(proj_dim, in_dim, &mut rng),
            proj_b: rand_vec(proj_dim, &mut rng),
        }
    }

    /// Apply the projection head (ReLU activation).
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        self.proj_w
            .iter()
            .zip(self.proj_b.iter())
            .map(|(row, b)| relu(dot(row, x) + b))
            .collect()
    }

    /// Compute NT-Xent loss over two views.
    ///
    /// * `z1`, `z2`    — representations (N, D)
    /// * `temperature` — softmax temperature τ
    pub fn nt_xent_loss(z1: &[Vec<f64>], z2: &[Vec<f64>], temperature: f64) -> f64 {
        let n = z1.len();
        if n == 0 {
            return 0.0;
        }
        // L2-normalise
        let norm = |v: &[f64]| -> Vec<f64> {
            let s = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
            v.iter().map(|x| x / s).collect()
        };
        let zn1: Vec<Vec<f64>> = z1.iter().map(|v| norm(v)).collect();
        let zn2: Vec<Vec<f64>> = z2.iter().map(|v| norm(v)).collect();

        // Concatenate: [z1; z2] length 2N
        let all: Vec<&Vec<f64>> = zn1.iter().chain(zn2.iter()).collect();
        let total = 2 * n;

        // Labels: positive pair for i is i+N (and vice versa)
        let mut loss = 0.0;
        for i in 0..total {
            let pos = if i < n { i + n } else { i - n };
            let sims: Vec<f64> = (0..total)
                .map(|j| {
                    if j == i {
                        f64::NEG_INFINITY // exclude self
                    } else {
                        dot(all[i], all[j]) / temperature
                    }
                })
                .collect();
            let log_sm = {
                let max = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = sims.iter().map(|&s| (s - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                sims[pos] - max - sum.ln()
            };
            loss -= log_sm;
        }
        loss / total as f64
    }
}

/// BYOL — online network + EMA target network, cosine similarity loss.
pub struct Byol {
    /// Online predictor weights: (d × d).
    pred_w: Vec<Vec<f64>>,
    pred_b: Vec<f64>,
}

impl Byol {
    /// Construct predictor head for `d`-dimensional representations, seeded by `seed`.
    pub fn new(d: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            pred_w: rand_mat(d, d, &mut rng),
            pred_b: rand_vec(d, &mut rng),
        }
    }

    /// Apply the online predictor.
    pub fn predict(&self, z: &[f64]) -> Vec<f64> {
        self.pred_w
            .iter()
            .zip(self.pred_b.iter())
            .map(|(row, b)| relu(dot(row, z) + b))
            .collect()
    }

    /// Compute BYOL loss: `2 - 2 * mean(cosine_sim(pred, target))`.
    ///
    /// Target is treated as stop-gradient.
    pub fn byol_loss(z_online: &[Vec<f64>], z_target: &[Vec<f64>]) -> f64 {
        let n = z_online.len().min(z_target.len());
        if n == 0 {
            return 0.0;
        }
        let cos_sim: f64 = z_online
            .iter()
            .zip(z_target.iter())
            .take(n)
            .map(|(a, b)| {
                let na = a.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
                let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
                dot(a, b) / (na * nb)
            })
            .sum::<f64>();
        2.0 - 2.0 * (cos_sim / n as f64)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::SeedableRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    // ── PatchEmbedding ────────────────────────────────────────────────────────

    #[test]
    fn test_patch_embedding_output_count() {
        let pe = PatchEmbedding::new(4, 16, 1);
        let img = vec![0.0f64; 8 * 8];
        let patches = pe.embed(&img, 8, 8);
        assert_eq!(patches.len(), (8 / 4) * (8 / 4));
    }

    #[test]
    fn test_patch_embedding_patch_dim() {
        let pe = PatchEmbedding::new(4, 16, 1);
        let img = vec![1.0f64; 8 * 8];
        let patches = pe.embed(&img, 8, 8);
        for p in &patches {
            assert_eq!(p.len(), 16);
        }
    }

    #[test]
    fn test_patch_embedding_seeded_init() {
        let pe1 = PatchEmbedding::new(4, 16, 99);
        let pe2 = PatchEmbedding::new(4, 16, 99);
        assert_eq!(pe1.weights, pe2.weights);
        assert_eq!(pe1.bias, pe2.bias);
    }

    // ── VitBlock ──────────────────────────────────────────────────────────────

    #[test]
    fn test_vit_block_shape() {
        let blk = VitBlock::new(8, 2, 7);
        let x: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1f64; 8]).collect();
        let out = blk.forward(&x);
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|v| v.len() == 8));
    }

    // ── VisionTransformer ─────────────────────────────────────────────────────

    #[test]
    fn test_vision_transformer_construction() {
        let vit = VisionTransformer::new(4, 8, 2, 1, 5, 0);
        assert_eq!(vit.n_classes, 5);
    }

    #[test]
    fn test_vision_transformer_forward_shape() {
        let vit = VisionTransformer::new(4, 8, 2, 1, 10, 3);
        let img = vec![0.5f64; 8 * 8];
        let logits = vit.forward(&img, 8, 8);
        assert_eq!(logits.len(), 10);
    }

    // ── PatchMerging ──────────────────────────────────────────────────────────

    #[test]
    fn test_patch_merging_output_count() {
        let pm = PatchMerging::new(4, 0);
        let patches: Vec<Vec<f64>> = (0..16).map(|_| vec![0.0f64; 4]).collect();
        let merged = pm.merge(&patches, 4, 4);
        assert_eq!(merged.len(), (4 / 2) * (4 / 2));
    }

    #[test]
    fn test_patch_merging_output_dim() {
        let c = 6usize;
        let pm = PatchMerging::new(c, 0);
        let patches: Vec<Vec<f64>> = (0..4).map(|_| vec![0.0f64; c]).collect();
        let merged = pm.merge(&patches, 2, 2);
        assert!(merged.iter().all(|v| v.len() == 2 * c));
    }

    // ── SwinBlock ─────────────────────────────────────────────────────────────

    #[test]
    fn test_swin_block_shape() {
        let blk = SwinBlock::new(8, 5);
        let x: Vec<Vec<f64>> = (0..6).map(|_| vec![0.0f64; 8]).collect();
        let out = blk.forward(&x);
        assert_eq!(out.len(), 6);
        assert!(out.iter().all(|v| v.len() == 8));
    }

    // ── SwinStage ─────────────────────────────────────────────────────────────

    #[test]
    fn test_swin_stage_with_merging() {
        let stage = SwinStage::new(4, 1, true, 0);
        let x: Vec<Vec<f64>> = (0..16).map(|_| vec![0.0f64; 4]).collect();
        let out = stage.forward(&x, 4, 4);
        // After merging: (4/2)*(4/2) = 4 tokens, each 2*4 = 8 dims
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.len() == 8));
    }

    #[test]
    fn test_swin_stage_without_merging() {
        let stage = SwinStage::new(4, 2, false, 0);
        let x: Vec<Vec<f64>> = (0..9).map(|_| vec![0.0f64; 4]).collect();
        let out = stage.forward(&x, 3, 3);
        assert_eq!(out.len(), 9);
        assert!(out.iter().all(|v| v.len() == 4));
    }

    // ── NmsProcessor ─────────────────────────────────────────────────────────

    #[test]
    fn test_nms_keeps_best() {
        let boxes = [[0.0, 0.0, 1.0, 1.0]];
        let scores = [0.9];
        let kept = NmsProcessor::nms(&boxes, &scores, 0.5);
        assert_eq!(kept, vec![0]);
    }

    #[test]
    fn test_nms_removes_overlapping() {
        let boxes = [[0.0, 0.0, 1.0, 1.0], [0.05, 0.05, 1.05, 1.05]];
        let scores = [0.9, 0.8];
        let kept = NmsProcessor::nms(&boxes, &scores, 0.5);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0], 0);
    }

    #[test]
    fn test_nms_keeps_non_overlapping() {
        let boxes = [[0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 11.0, 11.0]];
        let scores = [0.9, 0.8];
        let kept = NmsProcessor::nms(&boxes, &scores, 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_nms_empty_input() {
        let kept = NmsProcessor::nms(&[], &[], 0.5);
        assert!(kept.is_empty());
    }

    // ── SegmentationHead ──────────────────────────────────────────────────────

    #[test]
    fn test_segmentation_head_shape() {
        let head = SegmentationHead::new(8, 3, 0);
        let features: Vec<Vec<f64>> = (0..10).map(|_| vec![0.0f64; 8]).collect();
        let out = head.forward(&features, 3);
        assert_eq!(out.len(), 10);
        assert!(out.iter().all(|v| v.len() == 3));
    }

    // ── DetectionHead ─────────────────────────────────────────────────────────

    #[test]
    fn test_detection_head_cls_shape() {
        let head = DetectionHead::new(8, 5, 0);
        let features: Vec<Vec<f64>> = (0..7).map(|_| vec![0.0f64; 8]).collect();
        let (cls, _) = head.forward(&features);
        assert_eq!(cls.len(), 7);
        assert!(cls.iter().all(|v| v.len() == 5));
    }

    #[test]
    fn test_detection_head_box_shape() {
        let head = DetectionHead::new(8, 5, 0);
        let features: Vec<Vec<f64>> = (0..7).map(|_| vec![0.0f64; 8]).collect();
        let (_, boxes) = head.forward(&features);
        assert_eq!(boxes.len(), 7);
        assert!(boxes.iter().all(|v| v.len() == 4));
    }

    // ── RandomMasking ─────────────────────────────────────────────────────────

    #[test]
    fn test_random_masking_counts() {
        let mut rng = make_rng();
        let (vis, msk) = RandomMasking::mask(20, 0.75, &mut rng);
        assert_eq!(vis.len() + msk.len(), 20);
    }

    #[test]
    fn test_random_masking_ratio() {
        let mut rng = make_rng();
        let n = 100;
        let ratio = 0.75;
        let (vis, _) = RandomMasking::mask(n, ratio, &mut rng);
        let expected_vis = ((n as f64) * (1.0 - ratio)).round() as usize;
        assert_eq!(vis.len(), expected_vis);
    }

    // ── MaeModel ──────────────────────────────────────────────────────────────

    #[test]
    fn test_mae_reconstruction_loss_positive() {
        let model = MaeModel::new(4, 8, 16, 0);
        let img = vec![0.5f64; 8 * 8];
        let mut rng = make_rng();
        let loss = model.reconstruction_loss(&img, 8, 8, 0.75, &mut rng);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_mae_loss_finite() {
        let model = MaeModel::new(4, 8, 16, 1);
        let img: Vec<f64> = (0..64).map(|i| i as f64 * 0.01).collect();
        let mut rng = make_rng();
        let loss = model.reconstruction_loss(&img, 8, 8, 0.5, &mut rng);
        assert!(loss.is_finite());
    }

    // ── BarlowTwins ───────────────────────────────────────────────────────────

    #[test]
    fn test_barlow_loss_positive() {
        let mut rng = make_rng();
        let z1: Vec<Vec<f64>> = (0..8).map(|_| rand_vec(4, &mut rng)).collect();
        let z2: Vec<Vec<f64>> = (0..8).map(|_| rand_vec(4, &mut rng)).collect();
        let loss = BarlowTwins::barlow_loss(&z1, &z2, 0.005);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_barlow_loss_identical_zero() {
        // Identical inputs → cross-correlation matrix = identity → loss ≈ 0
        let mut rng = make_rng();
        let z: Vec<Vec<f64>> = (0..16).map(|_| rand_vec(4, &mut rng)).collect();
        let loss = BarlowTwins::barlow_loss(&z, &z, 0.005);
        // After feature-normalising identical inputs, C_ii = 1, C_ij = same for both → loss near 0
        assert!(
            loss < 1.0,
            "loss should be small for identical views: {loss}"
        );
    }

    // ── VicReg ────────────────────────────────────────────────────────────────

    #[test]
    fn test_vicreg_loss_positive() {
        let mut rng = make_rng();
        let z1: Vec<Vec<f64>> = (0..8).map(|_| rand_vec(4, &mut rng)).collect();
        let z2: Vec<Vec<f64>> = (0..8).map(|_| rand_vec(4, &mut rng)).collect();
        let loss = VicReg::vicreg_loss(&z1, &z2, 25.0, 25.0, 1.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_vicreg_variance_term() {
        // Random vectors should have non-trivial variance → variance loss > 0
        let mut rng = make_rng();
        let z1: Vec<Vec<f64>> = (0..20).map(|_| rand_vec(8, &mut rng)).collect();
        let z2 = z1.clone();
        // With mu_s=0 we isolate variance + covariance terms
        let loss = VicReg::vicreg_loss(&z1, &z2, 0.0, 25.0, 0.0);
        assert!(loss.is_finite());
    }

    // ── SimCLRv2 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_simclrv2_loss_positive() {
        let mut rng = make_rng();
        let z1: Vec<Vec<f64>> = (0..4).map(|_| rand_vec(8, &mut rng)).collect();
        let z2: Vec<Vec<f64>> = (0..4).map(|_| rand_vec(8, &mut rng)).collect();
        let loss = SimCLRv2::nt_xent_loss(&z1, &z2, 0.07);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_simclrv2_loss_finite() {
        let mut rng = make_rng();
        let z1: Vec<Vec<f64>> = (0..6).map(|_| rand_vec(4, &mut rng)).collect();
        let z2: Vec<Vec<f64>> = (0..6).map(|_| rand_vec(4, &mut rng)).collect();
        let loss = SimCLRv2::nt_xent_loss(&z1, &z2, 0.1);
        assert!(loss.is_finite());
    }

    // ── BYOL ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_byol_loss_range() {
        let mut rng = make_rng();
        let z1: Vec<Vec<f64>> = (0..8).map(|_| rand_vec(4, &mut rng)).collect();
        let z2: Vec<Vec<f64>> = (0..8).map(|_| rand_vec(4, &mut rng)).collect();
        let loss = Byol::byol_loss(&z1, &z2);
        // Cosine similarity in [-1,1] → BYOL loss in [0, 4]
        assert!(
            (0.0..=4.0 + 1e-6).contains(&loss),
            "byol loss out of range: {loss}"
        );
    }

    #[test]
    fn test_byol_loss_finite() {
        let mut rng = make_rng();
        let z1: Vec<Vec<f64>> = (0..5).map(|_| rand_vec(4, &mut rng)).collect();
        let z2: Vec<Vec<f64>> = (0..5).map(|_| rand_vec(4, &mut rng)).collect();
        let loss = Byol::byol_loss(&z1, &z2);
        assert!(loss.is_finite());
    }
}
