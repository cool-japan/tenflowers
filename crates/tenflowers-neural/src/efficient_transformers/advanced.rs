//! Advanced efficient transformer architectures:
//! RetNet (Microsoft 2023), Mamba-2 / SSD, enhanced GQA/MQA, KV-cache optimisations.

use super::{dot, layer_norm, matmul, rand_matrix, relu, softmax_rows, softplus_f64, transpose};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

type EtResult<T> = Result<T, String>;

// ─────────────────────────────────────────────────────────────────────────────
// 1.  RetNet (Microsoft 2023)
// ─────────────────────────────────────────────────────────────────────────────

/// Single retention head in parallel mode.
///
/// Parallel form: `Ret(Q,K,V) = (Q·K^T ⊙ D) · V`
/// where `D[i,j] = γ^(i-j)` for `i ≥ j`, else `0` (causal decay matrix).
pub struct RetentionLayer {
    /// Number of query/key/value dimensions.
    pub d_head: usize,
    /// Per-head decay rate γ ∈ (0,1).
    pub gamma: f64,
    /// Query projection: (d_model × d_head).
    w_q: Vec<Vec<f64>>,
    /// Key projection: (d_model × d_head).
    w_k: Vec<Vec<f64>>,
    /// Value projection: (d_model × d_head).
    w_v: Vec<Vec<f64>>,
}

impl RetentionLayer {
    /// Construct a retention layer.
    ///
    /// `d_model`: input dimension; `d_head`: head dimension; `gamma`: decay rate;
    /// `seed`: RNG seed.
    pub fn new(d_model: usize, d_head: usize, gamma: f64, seed: u64) -> EtResult<Self> {
        if d_head == 0 {
            return Err("RetentionLayer: d_head must be > 0".into());
        }
        let gamma = gamma.clamp(0.0, 1.0 - 1e-9);
        let scale = (1.0 / d_model as f64).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            d_head,
            gamma,
            w_q: rand_matrix(&mut rng, d_head, d_model, scale),
            w_k: rand_matrix(&mut rng, d_head, d_model, scale),
            w_v: rand_matrix(&mut rng, d_head, d_model, scale),
        })
    }

    /// Parallel forward pass: `(N × d_model)` → `(N × d_head)`.
    ///
    /// Complexity O(N² · d_head).
    pub fn forward_parallel(&self, x: &[Vec<f64>]) -> EtResult<Vec<Vec<f64>>> {
        let n = x.len();
        if n == 0 {
            return Ok(vec![]);
        }
        // Project to Q, K, V: each (N × d_head)
        let q = project_rows(x, &self.w_q);
        let k = project_rows(x, &self.w_k);
        let v = project_rows(x, &self.w_v);

        // Score matrix S = Q·K^T / sqrt(d_head) ⊙ D (causal decay)
        let k_t = transpose(&k);
        let mut s = matmul(&q, &k_t)?;
        let scale = 1.0 / (self.d_head as f64).sqrt();
        for i in 0..n {
            for j in 0..n {
                if j <= i {
                    let decay = self.gamma.powi((i - j) as i32);
                    s[i][j] *= scale * decay;
                } else {
                    s[i][j] = 0.0;
                }
            }
        }
        matmul(&s, &v)
    }

    /// Recurrent forward pass: single step for inference.
    ///
    /// `x_t`: current token `(d_model,)`.
    /// `state`: recurrent state `(d_head × d_head)`.
    /// Returns `(output: d_head, new_state)`.
    pub fn forward_recurrent(
        &self,
        x_t: &[f64],
        state: &[Vec<f64>],
    ) -> EtResult<(Vec<f64>, Vec<Vec<f64>>)> {
        let q_t: Vec<f64> = self.w_q.iter().map(|row| dot(row, x_t)).collect();
        let k_t: Vec<f64> = self.w_k.iter().map(|row| dot(row, x_t)).collect();
        let v_t: Vec<f64> = self.w_v.iter().map(|row| dot(row, x_t)).collect();

        // New state: S' = γ · S + k_t ⊗ v_t
        let dh = self.d_head;
        let mut new_state = vec![vec![0.0_f64; dh]; dh];
        for i in 0..dh {
            for j in 0..dh {
                let prev = state.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                new_state[i][j] = self.gamma * prev
                    + k_t.get(i).copied().unwrap_or(0.0)
                        * v_t.get(j).copied().unwrap_or(0.0);
            }
        }
        // Output: q_t · S'
        let out: Vec<f64> = (0..dh)
            .map(|j| {
                (0..dh)
                    .map(|i| q_t.get(i).copied().unwrap_or(0.0) * new_state[i][j])
                    .sum::<f64>()
            })
            .collect();
        Ok((out, new_state))
    }

    /// Chunk-wise forward: split sequence into chunks of `chunk_size` for efficiency.
    ///
    /// Combines parallel within-chunk and recurrent across-chunk computation.
    pub fn forward_chunkwise(
        &self,
        x: &[Vec<f64>],
        chunk_size: usize,
    ) -> EtResult<Vec<Vec<f64>>> {
        let n = x.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let chunk_size = chunk_size.max(1);
        let mut out = Vec::with_capacity(n);
        let dh = self.d_head;
        let mut cross_state = vec![vec![0.0_f64; dh]; dh];

        let mut t = 0;
        while t < n {
            let end = (t + chunk_size).min(n);
            let chunk = &x[t..end];
            let m = chunk.len();

            let q_chunk = project_rows(chunk, &self.w_q);
            let k_chunk = project_rows(chunk, &self.w_k);
            let v_chunk = project_rows(chunk, &self.w_v);

            // Within-chunk parallel retention
            let k_t = transpose(&k_chunk);
            let mut s_intra = matmul(&q_chunk, &k_t)?;
            let scale = 1.0 / (dh as f64).sqrt();
            for i in 0..m {
                for j in 0..m {
                    if j <= i {
                        let decay = self.gamma.powi((i - j) as i32);
                        s_intra[i][j] *= scale * decay;
                    } else {
                        s_intra[i][j] = 0.0;
                    }
                }
            }
            let intra_out = matmul(&s_intra, &v_chunk)?;

            // Cross-chunk contribution: q_chunk · cross_state
            for i in 0..m {
                let cross_contrib: Vec<f64> = (0..dh)
                    .map(|j| {
                        (0..dh)
                            .map(|ii| {
                                q_chunk[i].get(ii).copied().unwrap_or(0.0) * cross_state[ii][j]
                            })
                            .sum::<f64>()
                    })
                    .collect();
                let combined: Vec<f64> = intra_out[i]
                    .iter()
                    .zip(cross_contrib.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                out.push(combined);
            }

            // Update cross_state for next chunk
            for i in 0..m {
                let decay_pow = self.gamma.powi(m as i32);
                for ii in 0..dh {
                    for jj in 0..dh {
                        cross_state[ii][jj] = decay_pow * cross_state[ii][jj]
                            + k_chunk[i].get(ii).copied().unwrap_or(0.0)
                                * v_chunk[i].get(jj).copied().unwrap_or(0.0);
                    }
                }
            }
            t = end;
        }
        Ok(out)
    }
}

/// RetNet block: multi-head retention + FFN + group norm.
pub struct RetNetBlock {
    heads: Vec<RetentionLayer>,
    /// Output projection: (d_model × d_model).
    w_out: Vec<Vec<f64>>,
    /// FFN intermediate projection: (ffn_dim × d_model).
    ffn_w1: Vec<Vec<f64>>,
    /// FFN output projection: (d_model × ffn_dim).
    ffn_w2: Vec<Vec<f64>>,
    d_model: usize,
}

impl RetNetBlock {
    /// Build a RetNet block with `n_heads` retention heads.
    ///
    /// `ffn_mul`: FFN intermediate dimension = `ffn_mul × d_model`.
    pub fn new(
        d_model: usize,
        n_heads: usize,
        ffn_mul: usize,
        gammas: Option<&[f64]>,
        seed: u64,
    ) -> EtResult<Self> {
        if n_heads == 0 || d_model % n_heads != 0 {
            return Err("RetNetBlock: d_model must be divisible by n_heads".into());
        }
        let d_head = d_model / n_heads;
        let ffn_dim = (ffn_mul * d_model).max(1);
        let scale = (1.0 / d_model as f64).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);

        // Default gammas: 1 - 2^{-(5 + head_idx)}
        let heads: Vec<RetentionLayer> = (0..n_heads)
            .map(|h| {
                let g = gammas
                    .and_then(|gs| gs.get(h))
                    .copied()
                    .unwrap_or(1.0 - (-(5.0 + h as f64)).exp2());
                RetentionLayer::new(d_model, d_head, g, seed + h as u64 + 1)
                    .expect("RetentionLayer construction should succeed")
            })
            .collect();

        let w_out = rand_matrix(&mut rng, d_model, d_model, scale);
        let ffn_w1 = rand_matrix(&mut rng, ffn_dim, d_model, scale);
        let ffn_w2 = rand_matrix(&mut rng, d_model, ffn_dim, scale);

        Ok(Self {
            heads,
            w_out,
            ffn_w1,
            ffn_w2,
            d_model,
        })
    }

    /// Forward: `(N × d_model)` → `(N × d_model)`.
    pub fn forward(&self, x: &[Vec<f64>]) -> EtResult<Vec<Vec<f64>>> {
        let n = x.len();
        if n == 0 {
            return Ok(vec![]);
        }
        // Multi-head retention (parallel form)
        let mut head_outs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(self.heads.len());
        for head in &self.heads {
            head_outs.push(head.forward_parallel(x)?);
        }
        // Concatenate heads: (N × d_model)
        let concat: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                head_outs
                    .iter()
                    .flat_map(|ho| ho[i].iter().cloned())
                    .collect()
            })
            .collect();
        // Output projection
        let ret_out = project_rows(&concat, &self.w_out);
        // Residual + layer norm
        let ret_normed: Vec<Vec<f64>> = x
            .iter()
            .zip(ret_out.iter())
            .map(|(xi, ri)| {
                let added: Vec<f64> = xi.iter().zip(ri.iter()).map(|(a, b)| a + b).collect();
                layer_norm(&added, None, None)
            })
            .collect();
        // FFN: GeLU-MLP
        let ffn_out: Vec<Vec<f64>> = ret_normed
            .iter()
            .map(|row| {
                let h: Vec<f64> = self
                    .ffn_w1
                    .iter()
                    .map(|w| gelu(dot(w, row)))
                    .collect();
                self.ffn_w2.iter().map(|w| dot(w, &h)).collect()
            })
            .collect();
        // Residual + layer norm
        let out: Vec<Vec<f64>> = ret_normed
            .iter()
            .zip(ffn_out.iter())
            .map(|(rn, ff)| {
                let added: Vec<f64> = rn.iter().zip(ff.iter()).map(|(a, b)| a + b).collect();
                layer_norm(&added, None, None)
            })
            .collect();
        Ok(out)
    }
}

/// Stacked RetNet model.
pub struct RetNetModel {
    blocks: Vec<RetNetBlock>,
    d_model: usize,
}

impl RetNetModel {
    /// Build a RetNet model with `n_layers` blocks.
    pub fn new(d_model: usize, n_heads: usize, n_layers: usize, ffn_mul: usize, seed: u64) -> EtResult<Self> {
        let blocks: Result<Vec<_>, _> = (0..n_layers)
            .map(|l| RetNetBlock::new(d_model, n_heads, ffn_mul, None, seed + l as u64 * 100))
            .collect();
        Ok(Self {
            blocks: blocks?,
            d_model,
        })
    }

    /// Forward: `(N × d_model)` → `(N × d_model)`.
    pub fn forward(&self, x: &[Vec<f64>]) -> EtResult<Vec<Vec<f64>>> {
        let mut h = x.to_vec();
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        Ok(h)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  Mamba-2 / State Space Duality (SSD)
// ─────────────────────────────────────────────────────────────────────────────

/// Structured State Space Duality (SSD) kernel.
///
/// SSD unifies SSMs and semi-separable matrices. The SSD kernel computes:
/// `Y = (1 ⊗ C)(L ⊗ 1)(1 ⊗ B)u` where `L` is a lower-triangular matrix
/// with `L[i,j] = α[i] · α[i-1] · … · α[j+1]` for `i > j`.
pub struct Ssd {
    d_state: usize,
    d_inner: usize,
    /// Input projection B: (T, d_state) — computed per-token.
    w_b: Vec<Vec<f64>>,
    /// Output projection C: (T, d_state) — computed per-token.
    w_c: Vec<Vec<f64>>,
    /// State transition projection: scalar α per token from input.
    w_alpha: Vec<f64>,
}

impl Ssd {
    /// Construct an SSD layer.
    pub fn new(d_inner: usize, d_state: usize, seed: u64) -> EtResult<Self> {
        if d_inner == 0 || d_state == 0 {
            return Err("Ssd: d_inner and d_state must be > 0".into());
        }
        let scale = (1.0 / d_inner as f64).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            d_state,
            d_inner,
            w_b: rand_matrix(&mut rng, d_state, d_inner, scale),
            w_c: rand_matrix(&mut rng, d_state, d_inner, scale),
            w_alpha: (0..d_inner).map(|_| rng.random::<f64>() * scale).collect(),
        })
    }

    /// Parallel SSD scan: `(N × d_inner)` → `(N × d_inner)`.
    ///
    /// Implements the semi-separable matrix-vector product in O(N · d_state²).
    pub fn forward(&self, x: &[Vec<f64>]) -> EtResult<Vec<Vec<f64>>> {
        let n = x.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let ds = self.d_state;
        let di = self.d_inner;

        // Compute per-token α (gating via sigmoid of dot(w_alpha, x))
        let alphas: Vec<f64> = x
            .iter()
            .map(|xi| {
                let v: f64 = self
                    .w_alpha
                    .iter()
                    .zip(xi.iter())
                    .map(|(w, x)| w * x)
                    .sum();
                sigmoid_f64(v)
            })
            .collect();

        // Compute B[t] ∈ R^{d_state} and C[t] ∈ R^{d_state}
        let b_seq: Vec<Vec<f64>> = x.iter().map(|xi| project_vec(xi, &self.w_b)).collect();
        let c_seq: Vec<Vec<f64>> = x.iter().map(|xi| project_vec(xi, &self.w_c)).collect();

        // SSM scan: h[t] = α[t] · h[t-1] + B[t] · x[t]_mean, y[t] = C[t] · h[t]
        // x[t]_mean is the mean of x[t] as a scalar input
        let x_scalars: Vec<f64> = x
            .iter()
            .map(|xi| xi.iter().sum::<f64>() / di as f64)
            .collect();

        let mut h = vec![0.0_f64; ds];
        let mut out = Vec::with_capacity(n);
        for t in 0..n {
            let a = alphas[t];
            h = h
                .iter()
                .zip(b_seq[t].iter())
                .map(|(&hi, &bi)| a * hi + bi * x_scalars[t])
                .collect();
            // y_t scalar = C[t] · h (repeated as d_inner output)
            let y_scalar: f64 = c_seq[t].iter().zip(h.iter()).map(|(c, hh)| c * hh).sum();
            // Broadcast to d_inner, adding residual
            let row: Vec<f64> = x[t].iter().map(|xi| xi + y_scalar).collect();
            out.push(row);
        }
        Ok(out)
    }
}

/// SSD layer: wraps `Ssd` with input/output projections and layer norm.
pub struct SsdLayer {
    /// Input expansion: (d_inner × d_model).
    w_in: Vec<Vec<f64>>,
    /// Output contraction: (d_model × d_inner).
    w_out: Vec<Vec<f64>>,
    ssd: Ssd,
    d_model: usize,
}

impl SsdLayer {
    /// Construct an SSD layer.
    pub fn new(d_model: usize, d_inner: usize, d_state: usize, seed: u64) -> EtResult<Self> {
        let scale = (1.0 / d_model as f64).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            w_in: rand_matrix(&mut rng, d_inner, d_model, scale),
            w_out: rand_matrix(&mut rng, d_model, d_inner, scale),
            ssd: Ssd::new(d_inner, d_state, seed + 1)?,
            d_model,
        })
    }

    /// Forward: `(N × d_model)` → `(N × d_model)`.
    pub fn forward(&self, x: &[Vec<f64>]) -> EtResult<Vec<Vec<f64>>> {
        let n = x.len();
        if n == 0 {
            return Ok(vec![]);
        }
        // Expand
        let expanded: Vec<Vec<f64>> = x.iter().map(|xi| project_vec(xi, &self.w_in)).collect();
        // SSD scan
        let scanned = self.ssd.forward(&expanded)?;
        // Contract + residual + layer-norm
        let out: Vec<Vec<f64>> = x
            .iter()
            .zip(scanned.iter())
            .map(|(xi, si)| {
                let contracted: Vec<f64> = project_vec(si, &self.w_out);
                let added: Vec<f64> = xi.iter().zip(contracted.iter()).map(|(a, b)| a + b).collect();
                layer_norm(&added, None, None)
            })
            .collect();
        Ok(out)
    }
}

/// Mamba-2 block: SSD + normalisation + MLP gate.
pub struct Mamba2Block {
    ssd_layer: SsdLayer,
    /// Gate projection: (d_model × d_model).
    w_gate: Vec<Vec<f64>>,
    /// FFN projection: (d_model × d_model).
    w_ffn: Vec<Vec<f64>>,
    d_model: usize,
}

impl Mamba2Block {
    /// Construct a Mamba-2 block.
    pub fn new(d_model: usize, d_inner: usize, d_state: usize, seed: u64) -> EtResult<Self> {
        let scale = (1.0 / d_model as f64).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            ssd_layer: SsdLayer::new(d_model, d_inner, d_state, seed + 10)?,
            w_gate: rand_matrix(&mut rng, d_model, d_model, scale),
            w_ffn: rand_matrix(&mut rng, d_model, d_model, scale),
            d_model,
        })
    }

    /// Forward: `(N × d_model)` → `(N × d_model)`.
    pub fn forward(&self, x: &[Vec<f64>]) -> EtResult<Vec<Vec<f64>>> {
        let n = x.len();
        if n == 0 {
            return Ok(vec![]);
        }
        // SSD (main branch)
        let ssd_out = self.ssd_layer.forward(x)?;
        // Gate branch: SiLU(W_gate · x) ⊙ ssd_out
        let out: Vec<Vec<f64>> = x
            .iter()
            .zip(ssd_out.iter())
            .map(|(xi, si)| {
                let gate: Vec<f64> = self
                    .w_gate
                    .iter()
                    .map(|w| silu_f64(dot(w, xi)))
                    .collect();
                let gated: Vec<f64> = gate
                    .iter()
                    .zip(si.iter())
                    .map(|(g, s)| g * s)
                    .collect();
                // FFN residual
                let ffn: Vec<f64> = self
                    .w_ffn
                    .iter()
                    .map(|w| relu(dot(w, &gated)))
                    .collect();
                let added: Vec<f64> = xi.iter().zip(ffn.iter()).map(|(a, b)| a + b).collect();
                layer_norm(&added, None, None)
            })
            .collect();
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  GQA/MQA  (standalone enhanced versions)
// ─────────────────────────────────────────────────────────────────────────────

/// Enhanced Grouped-Query Attention (Ainslie 2023) with learnable projections.
///
/// `n_kv_heads` key/value heads are shared across `n_heads / n_kv_heads` query heads each.
pub struct EnhancedGroupedQueryAttention {
    n_heads: usize,
    n_kv_heads: usize,
    d_head: usize,
    /// Query projections: (n_heads × d_head × d_model).
    w_q: Vec<Vec<Vec<f64>>>,
    /// Key projections: (n_kv_heads × d_head × d_model).
    w_k: Vec<Vec<Vec<f64>>>,
    /// Value projections: (n_kv_heads × d_head × d_model).
    w_v: Vec<Vec<Vec<f64>>>,
    /// Output projection: (d_model × d_model).
    w_o: Vec<Vec<f64>>,
    d_model: usize,
}

impl EnhancedGroupedQueryAttention {
    /// Construct a GQA module.
    ///
    /// `n_kv_heads` must divide `n_heads`. `d_model` must be divisible by `n_heads`.
    pub fn new(d_model: usize, n_heads: usize, n_kv_heads: usize, seed: u64) -> EtResult<Self> {
        if n_kv_heads == 0 || n_heads == 0 {
            return Err("EnhancedGQA: n_heads and n_kv_heads must be > 0".into());
        }
        if n_heads % n_kv_heads != 0 {
            return Err("EnhancedGQA: n_heads must be divisible by n_kv_heads".into());
        }
        if d_model % n_heads != 0 {
            return Err("EnhancedGQA: d_model must be divisible by n_heads".into());
        }
        let d_head = d_model / n_heads;
        let scale = (1.0 / d_model as f64).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);
        let w_q: Vec<Vec<Vec<f64>>> = (0..n_heads)
            .map(|_| rand_matrix(&mut rng, d_head, d_model, scale))
            .collect();
        let w_k: Vec<Vec<Vec<f64>>> = (0..n_kv_heads)
            .map(|_| rand_matrix(&mut rng, d_head, d_model, scale))
            .collect();
        let w_v: Vec<Vec<Vec<f64>>> = (0..n_kv_heads)
            .map(|_| rand_matrix(&mut rng, d_head, d_model, scale))
            .collect();
        let w_o = rand_matrix(&mut rng, d_model, d_model, scale);
        Ok(Self {
            n_heads,
            n_kv_heads,
            d_head,
            w_q,
            w_k,
            w_v,
            w_o,
            d_model,
        })
    }

    /// Forward: `(N × d_model)` → `(N × d_model)`.
    pub fn forward(&self, x: &[Vec<f64>]) -> EtResult<Vec<Vec<f64>>> {
        let n = x.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let heads_per_group = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (self.d_head as f64).sqrt();

        // Project K, V per KV-head: each (N × d_head)
        let k_heads: Vec<Vec<Vec<f64>>> = self
            .w_k
            .iter()
            .map(|wk| x.iter().map(|xi| project_vec(xi, wk)).collect())
            .collect();
        let v_heads: Vec<Vec<Vec<f64>>> = self
            .w_v
            .iter()
            .map(|wv| x.iter().map(|xi| project_vec(xi, wv)).collect())
            .collect();

        // Per query head attention
        let mut all_head_outs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(self.n_heads);
        for h in 0..self.n_heads {
            let kv_group = h / heads_per_group;
            let q_h: Vec<Vec<f64>> = x
                .iter()
                .map(|xi| project_vec(xi, &self.w_q[h]))
                .collect();
            let k_h = &k_heads[kv_group];
            let v_h = &v_heads[kv_group];
            let k_ht = transpose(k_h);
            let mut scores = matmul(&q_h, &k_ht)?;
            for row in scores.iter_mut() {
                for s in row.iter_mut() {
                    *s *= scale;
                }
            }
            softmax_rows(&mut scores);
            all_head_outs.push(matmul(&scores, v_h)?);
        }

        // Concatenate + output projection
        let concat: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                all_head_outs
                    .iter()
                    .flat_map(|ho| ho[i].iter().cloned())
                    .collect()
            })
            .collect();
        let projected: Vec<Vec<f64>> = concat.iter().map(|c| project_vec(c, &self.w_o)).collect();
        Ok(projected)
    }
}

/// Multi-Query Attention (Shazeer 2019) — single KV head, full Q heads.
///
/// Special case of GQA with `n_kv_heads = 1`.
pub struct MultiQuerySingleHeadAttention {
    inner: EnhancedGroupedQueryAttention,
}

impl MultiQuerySingleHeadAttention {
    /// Construct an MQA module with `n_heads` query heads but only 1 KV head.
    pub fn new(d_model: usize, n_heads: usize, seed: u64) -> EtResult<Self> {
        Ok(Self {
            inner: EnhancedGroupedQueryAttention::new(d_model, n_heads, 1, seed)?,
        })
    }

    /// Forward: `(N × d_model)` → `(N × d_model)`.
    pub fn forward(&self, x: &[Vec<f64>]) -> EtResult<Vec<Vec<f64>>> {
        self.inner.forward(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  KV Cache Optimisations
// ─────────────────────────────────────────────────────────────────────────────

/// Sliding-window KV cache (Mistral-style).
///
/// Evicts the oldest KV pair once the window is full.
pub struct SlidingWindowKvCache {
    /// Maximum number of KV pairs to retain.
    pub window_size: usize,
    /// Stored keys: ring buffer of `(d_head,)` vectors.
    keys: std::collections::VecDeque<Vec<f64>>,
    /// Stored values: ring buffer of `(d_head,)` vectors.
    values: std::collections::VecDeque<Vec<f64>>,
}

impl SlidingWindowKvCache {
    /// Create a new sliding-window KV cache.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(1),
            keys: std::collections::VecDeque::new(),
            values: std::collections::VecDeque::new(),
        }
    }

    /// Insert a new key-value pair. Evicts oldest if beyond window.
    pub fn insert(&mut self, key: Vec<f64>, value: Vec<f64>) {
        if self.keys.len() >= self.window_size {
            self.keys.pop_front();
            self.values.pop_front();
        }
        self.keys.push_back(key);
        self.values.push_back(value);
    }

    /// Current number of cached tokens.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Retrieve all cached keys as a slice of vectors.
    pub fn keys(&self) -> Vec<&Vec<f64>> {
        self.keys.iter().collect()
    }

    /// Retrieve all cached values as a slice of vectors.
    pub fn values(&self) -> Vec<&Vec<f64>> {
        self.values.iter().collect()
    }

    /// Compute attention for a query `q` against the cached KV pairs.
    ///
    /// Returns a weighted sum of values via softmax attention.
    pub fn attend(&self, q: &[f64]) -> Vec<f64> {
        if self.keys.is_empty() {
            return vec![0.0; q.len()];
        }
        let d = q.len() as f64;
        let scale = 1.0 / d.sqrt();
        let mut scores: Vec<f64> = self
            .keys
            .iter()
            .map(|k| dot(q, k) * scale)
            .collect();
        // Softmax
        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = scores.iter().map(|s| (s - max_s).exp()).sum();
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp() / exp_sum.max(1e-9);
        }
        // Weighted sum of values
        let d_v = self.values.front().map(|v| v.len()).unwrap_or(0);
        let mut out = vec![0.0_f64; d_v];
        for (w, v) in scores.iter().zip(self.values.iter()) {
            for (o, vi) in out.iter_mut().zip(v.iter()) {
                *o += w * vi;
            }
        }
        out
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }
}

/// Sink Token KV Cache (StreamingLLM, Xiao et al. 2023).
///
/// Retains the first `n_sink` "attention sink" tokens plus the most recent `window_size` tokens.
pub struct SinkTokenCache {
    /// Number of sink tokens to keep (from the beginning of the sequence).
    pub n_sink: usize,
    /// Number of recent tokens to keep (sliding window).
    pub window_size: usize,
    /// Sink KV pairs (fixed, never evicted).
    sink_keys: Vec<Vec<f64>>,
    sink_values: Vec<Vec<f64>>,
    /// Recent KV pairs (ring buffer).
    recent_keys: std::collections::VecDeque<Vec<f64>>,
    recent_values: std::collections::VecDeque<Vec<f64>>,
    /// Total number of tokens processed so far.
    total_tokens: usize,
}

impl SinkTokenCache {
    /// Create a new sink token cache.
    pub fn new(n_sink: usize, window_size: usize) -> Self {
        Self {
            n_sink: n_sink.max(1),
            window_size: window_size.max(1),
            sink_keys: Vec::new(),
            sink_values: Vec::new(),
            recent_keys: std::collections::VecDeque::new(),
            recent_values: std::collections::VecDeque::new(),
            total_tokens: 0,
        }
    }

    /// Insert a new key-value pair.
    ///
    /// The first `n_sink` tokens are stored as sink tokens.
    /// Subsequent tokens go into the sliding window.
    pub fn insert(&mut self, key: Vec<f64>, value: Vec<f64>) {
        if self.total_tokens < self.n_sink {
            self.sink_keys.push(key);
            self.sink_values.push(value);
        } else {
            if self.recent_keys.len() >= self.window_size {
                self.recent_keys.pop_front();
                self.recent_values.pop_front();
            }
            self.recent_keys.push_back(key);
            self.recent_values.push_back(value);
        }
        self.total_tokens += 1;
    }

    /// Total cached tokens (sink + recent).
    pub fn len(&self) -> usize {
        self.sink_keys.len() + self.recent_keys.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.sink_keys.is_empty() && self.recent_keys.is_empty()
    }

    /// Number of sink tokens currently stored.
    pub fn n_sink_stored(&self) -> usize {
        self.sink_keys.len()
    }

    /// Number of recent tokens currently stored.
    pub fn n_recent_stored(&self) -> usize {
        self.recent_keys.len()
    }

    /// Compute attention for query `q` against all cached KV pairs.
    pub fn attend(&self, q: &[f64]) -> Vec<f64> {
        let all_keys: Vec<&Vec<f64>> = self
            .sink_keys
            .iter()
            .chain(self.recent_keys.iter())
            .collect();
        let all_values: Vec<&Vec<f64>> = self
            .sink_values
            .iter()
            .chain(self.recent_values.iter())
            .collect();
        if all_keys.is_empty() {
            return vec![0.0; q.len()];
        }
        let scale = 1.0 / (q.len() as f64).sqrt();
        let mut scores: Vec<f64> = all_keys.iter().map(|k| dot(q, k) * scale).collect();
        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = scores.iter().map(|s| (s - max_s).exp()).sum();
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp() / exp_sum.max(1e-9);
        }
        let d_v = all_values.first().map(|v| v.len()).unwrap_or(0);
        let mut out = vec![0.0_f64; d_v];
        for (w, v) in scores.iter().zip(all_values.iter()) {
            for (o, vi) in out.iter_mut().zip(v.iter()) {
                *o += w * vi;
            }
        }
        out
    }

    /// Clear the recent window (but keep sink tokens).
    pub fn clear_recent(&mut self) {
        self.recent_keys.clear();
        self.recent_values.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Efficiency metrics for transformer components.
#[derive(Debug, Clone)]
pub struct EtMetrics {
    /// Estimated FLOP count for a forward pass (multiply-add operations).
    pub flops: f64,
    /// Estimated memory usage in bytes (assuming f64 = 8 bytes).
    pub memory_bytes: f64,
    /// Throughput proxy: tokens/GFLOP.
    pub throughput_proxy: f64,
}

impl EtMetrics {
    /// Compute metrics for a standard dense attention layer.
    ///
    /// `seq_len`: sequence length; `d_model`: model dimension; `n_heads`: head count;
    /// `batch_size`: batch size.
    pub fn dense_attention(seq_len: usize, d_model: usize, n_heads: usize, batch_size: usize) -> Self {
        let n = seq_len as f64;
        let d = d_model as f64;
        let h = n_heads as f64;
        let b = batch_size as f64;
        // QKV projections: 3 × (N × d × d)
        let qkv_flops = 3.0 * b * n * d * d;
        // Attention scores + softmax: N × N × d/h × h
        let attn_flops = b * h * n * n * (d / h);
        // Context + output: N × N × d/h + N × d × d
        let ctx_flops = b * n * n * (d / h) * h + b * n * d * d;
        let flops = qkv_flops + attn_flops + ctx_flops;
        // Memory: Q,K,V tensors + attention matrix
        let memory = b * n * d * 3.0 * 8.0 + b * h * n * n * 8.0;
        let throughput_proxy = (b * n) / (flops / 1e9).max(1e-12);
        Self {
            flops,
            memory_bytes: memory,
            throughput_proxy,
        }
    }

    /// Compute metrics for linear attention (O(N·d) instead of O(N²·d)).
    pub fn linear_attention(seq_len: usize, d_model: usize, batch_size: usize) -> Self {
        let n = seq_len as f64;
        let d = d_model as f64;
        let b = batch_size as f64;
        let flops = 3.0 * b * n * d * d + b * n * d * d;
        let memory = b * n * d * 4.0 * 8.0 + b * d * d * 8.0;
        let throughput_proxy = (b * n) / (flops / 1e9).max(1e-12);
        Self {
            flops,
            memory_bytes: memory,
            throughput_proxy,
        }
    }

    /// Compute metrics for GQA with `n_kv_heads` KV heads.
    pub fn grouped_query_attention(
        seq_len: usize,
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        batch_size: usize,
    ) -> Self {
        let n = seq_len as f64;
        let d = d_model as f64;
        let b = batch_size as f64;
        let h = n_heads as f64;
        let hkv = n_kv_heads as f64;
        // Q projection: N × d × d; KV projections: N × d × (d/h × hkv) each
        let d_kv = d / h * hkv;
        let proj_flops = b * n * d * d + 2.0 * b * n * d * d_kv;
        let attn_flops = b * h * n * n * (d / h);
        let flops = proj_flops + attn_flops + b * n * d * d;
        let memory = b * n * (d + 2.0 * d_kv) * 8.0 + b * h * n * n * 8.0;
        let throughput_proxy = (b * n) / (flops / 1e9).max(1e-12);
        Self {
            flops,
            memory_bytes: memory,
            throughput_proxy,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Project each row of `x` through weight matrix `w`: returns (N × out_dim).
fn project_rows(x: &[Vec<f64>], w: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter().map(|xi| project_vec(xi, w)).collect()
}

/// Matrix-vector: `w` is (out_dim × in_dim), returns out_dim vector.
fn project_vec(x: &[f64], w: &[Vec<f64>]) -> Vec<f64> {
    w.iter().map(|row| dot(row, x)).collect()
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh())
}

fn sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn silu_f64(x: f64) -> f64 {
    x * sigmoid_f64(x)
}
