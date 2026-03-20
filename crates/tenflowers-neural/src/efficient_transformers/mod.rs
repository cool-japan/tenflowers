//! Advanced Efficient Transformer Architectures — low-rank, sub-quadratic,
//! position encodings, mixture, and training-efficiency utilities.
//! All computations use plain `Vec<f64>` buffers (no Tensor dependency).

pub mod extensions;
pub use extensions::*;

pub mod advanced;
pub use advanced::*;

#[cfg(test)]
pub mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

type EtResult<T> = Result<T, String>;

// ─────────────────────────────────────────────────────────────────────────────
// Internal math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix multiply: (m×k) · (k×n) → (m×n), row-major.
pub(crate) fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> EtResult<Vec<Vec<f64>>> {
    let m = a.len();
    if m == 0 {
        return Ok(vec![]);
    }
    let k = a[0].len();
    let n = b[0].len();
    if b.len() != k {
        return Err(format!("matmul dim mismatch: k={k} vs b.rows={}", b.len()));
    }
    let mut out = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f64;
            for l in 0..k {
                s += a[i][l] * b[l][j];
            }
            out[i][j] = s;
        }
    }
    Ok(out)
}

/// Transpose a 2-D matrix.
pub(crate) fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return vec![];
    }
    let rows = a.len();
    let cols = a[0].len();
    let mut t = vec![vec![0.0_f64; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Row-wise softmax (in-place on a mutable slice per row).
pub(crate) fn softmax_rows(m: &mut [Vec<f64>]) {
    for row in m.iter_mut() {
        let max_v = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row.iter().map(|x| (x - max_v).exp()).sum();
        for v in row.iter_mut() {
            *v = (*v - max_v).exp() / sum.max(1e-9);
        }
    }
}

/// Dot product of two slices.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Layer norm over a row: `(x - μ) / (σ + ε) * γ + β` (γ=1, β=0 when `None`).
pub(crate) fn layer_norm(x: &[f64], gamma: Option<&[f64]>, beta: Option<&[f64]>) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = (var + 1e-6).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let g = gamma.map(|g| g[i]).unwrap_or(1.0);
            let b = beta.map(|b| b[i]).unwrap_or(0.0);
            (v - mean) / std * g + b
        })
        .collect()
}

/// Element-wise ReLU.
pub(crate) fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Random weight matrix seeded deterministically.
pub(crate) fn rand_matrix(rng: &mut StdRng, rows: usize, cols: usize, scale: f64) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Low-Rank & Sparse Attention
// ─────────────────────────────────────────────────────────────────────────────

/// Low-rank attention: Q′ = Q·A, K′ = K·B (d→r), complexity O(N·r).
///
/// `A` : (d_model × rank), `B` : (d_model × rank), `W_V` : (d_model × d_v).
pub struct LowRankAttention {
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    w_v: Vec<Vec<f64>>,
    d_model: usize,
    rank: usize,
    d_v: usize,
}

impl LowRankAttention {
    /// Create a new low-rank attention with the given dimensions and a fixed seed.
    pub fn new(d_model: usize, rank: usize, d_v: usize, seed: u64) -> EtResult<Self> {
        if rank == 0 || rank > d_model {
            return Err(format!("rank must be in [1, {d_model}]"));
        }
        let scale = (1.0 / d_model as f64).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            a: rand_matrix(&mut rng, d_model, rank, scale),
            b: rand_matrix(&mut rng, d_model, rank, scale),
            w_v: rand_matrix(&mut rng, d_model, d_v, scale),
            d_model,
            rank,
            d_v,
        })
    }

    /// Forward pass: `(Q, K, V)` with shapes `(N×d_model)` → `(N×d_v)`.
    pub fn forward(
        &self,
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
    ) -> EtResult<Vec<Vec<f64>>> {
        // Q' = Q·A  (N × rank)
        let q_prime = matmul(q, &self.a)?;
        // K' = K·B  (N × rank)
        let k_prime = matmul(k, &self.b)?;
        // Scores = Q' · K'^T / sqrt(rank)   (N × N)
        let k_t = transpose(&k_prime);
        let mut scores = matmul(&q_prime, &k_t)?;
        let scale = 1.0 / (self.rank as f64).sqrt();
        for row in scores.iter_mut() {
            for s in row.iter_mut() {
                *s *= scale;
            }
        }
        softmax_rows(&mut scores);
        // V_proj = V·W_V  (N × d_v)
        let v_proj = matmul(v, &self.w_v)?;
        // Out = Attn · V_proj  (N × d_v)
        matmul(&scores, &v_proj)
    }
}

/// Randomized SVD attention: compress attention matrix via rank-r SVD approximation.
pub struct RandomizedSvdAttention {
    seed: u64,
}

impl RandomizedSvdAttention {
    /// Create a new randomized SVD attention.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Rank-approximated attention via randomized SVD (sketch + thin SVD).
    pub fn forward(
        &self,
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
        rank: usize,
    ) -> EtResult<Vec<Vec<f64>>> {
        let n = q.len();
        let d = q[0].len();
        let rank = rank.min(n);
        let scale = 1.0 / (d as f64).sqrt();
        let k_t = transpose(k);
        let mut s = matmul(q, &k_t)?;
        for r in s.iter_mut() {
            for x in r.iter_mut() {
                *x *= scale;
            }
        }
        let mut rng = StdRng::seed_from_u64(self.seed);
        let omega = rand_matrix(&mut rng, n, rank, 1.0);
        let y = matmul(&s, &omega)?;
        let q_hat = gram_schmidt(&y, rank)?;
        let b_mat = matmul(&transpose(&q_hat), &s)?;
        let (u_b, sigma, vt) = thin_svd(&b_mat)?;
        let u_full = matmul(&q_hat, &u_b)?;
        let rk = sigma.len();
        let mut s_approx = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                for r in 0..rk {
                    s_approx[i][j] += u_full[i][r] * sigma[r] * vt[r][j];
                }
            }
        }
        softmax_rows(&mut s_approx);
        matmul(&s_approx, v)
    }
}

/// Gram-Schmidt orthonormalization: returns Q of shape (rows × k).
pub(crate) fn gram_schmidt(a: &[Vec<f64>], k: usize) -> EtResult<Vec<Vec<f64>>> {
    let rows = a.len();
    let k = k.min(a[0].len());
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(k);
    for j in 0..k {
        let mut v: Vec<f64> = (0..rows).map(|i| a[i][j]).collect();
        for qi in cols.iter() {
            let d = dot(qi, &v);
            for (vi, qi_i) in v.iter_mut().zip(qi.iter()) {
                *vi -= d * qi_i;
            }
        }
        let norm = dot(&v, &v).sqrt().max(1e-12);
        cols.push(v.iter().map(|x| x / norm).collect());
    }
    let mut out = vec![vec![0.0_f64; k]; rows];
    for j in 0..k {
        for i in 0..rows {
            out[i][j] = cols[j][i];
        }
    }
    Ok(out)
}

/// Rank-k SVD via normal equations + Jacobi eigen (compact form).
/// Returns (U: N×k, sigma: k, Vt: k×N).
pub(crate) fn thin_svd(a: &[Vec<f64>]) -> EtResult<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>)> {
    let m = a.len();
    let n = a[0].len();
    let k = m.min(n);
    let at = transpose(a);
    let aat = matmul(a, &at)?;
    // Jacobi eigen on A·A^T
    let mut w = aat.clone();
    let mut ev: Vec<Vec<f64>> = (0..m)
        .map(|i| (0..m).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    for _ in 0..30 * m * m {
        let (mut pi, mut pj, mut mx) = (0, 1, 0.0_f64);
        for i in 0..m {
            for j in (i + 1)..m {
                let v = w[i][j].abs();
                if v > mx {
                    mx = v;
                    pi = i;
                    pj = j;
                }
            }
        }
        if mx < 1e-12 {
            break;
        }
        let th = if (w[pi][pi] - w[pj][pj]).abs() < 1e-12 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * w[pi][pj]) / (w[pj][pj] - w[pi][pi])).atan()
        };
        let (c, s) = (th.cos(), th.sin());
        let mut wn = w.clone();
        for r in 0..m {
            if r != pi && r != pj {
                wn[r][pi] = c * w[r][pi] - s * w[r][pj];
                wn[pi][r] = wn[r][pi];
                wn[r][pj] = s * w[r][pi] + c * w[r][pj];
                wn[pj][r] = wn[r][pj];
            }
        }
        wn[pi][pi] = c * c * w[pi][pi] - 2.0 * s * c * w[pi][pj] + s * s * w[pj][pj];
        wn[pj][pj] = s * s * w[pi][pi] + 2.0 * s * c * w[pi][pj] + c * c * w[pj][pj];
        wn[pi][pj] = 0.0;
        wn[pj][pi] = 0.0;
        w = wn;
        for r in 0..m {
            let (tp, tq) = (c * ev[r][pi] - s * ev[r][pj], s * ev[r][pi] + c * ev[r][pj]);
            ev[r][pi] = tp;
            ev[r][pj] = tq;
        }
    }
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..m)
        .map(|i| (w[i][i], (0..m).map(|r| ev[r][i]).collect()))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let sigma: Vec<f64> = pairs[..k].iter().map(|(e, _)| e.max(0.0).sqrt()).collect();
    let mut u = vec![vec![0.0_f64; k]; m];
    for (col, (_, ev_c)) in pairs[..k].iter().enumerate() {
        for row in 0..m {
            u[row][col] = ev_c[row];
        }
    }
    let mut vt = vec![vec![0.0_f64; n]; k];
    for r in 0..k {
        if sigma[r] < 1e-14 {
            continue;
        }
        let inv_s = 1.0 / sigma[r];
        for j in 0..n {
            vt[r][j] = at[j]
                .iter()
                .enumerate()
                .map(|(i, x)| x * u[i][r])
                .sum::<f64>()
                * inv_s;
        }
    }
    Ok((u, sigma, vt))
}

/// RevNet-style reversible layer: `(x1,x2)→(y1=x1+F(x2), y2=x2+G(y1))`.
///
/// F and G are simple single-layer MLPs stored as weight matrices.
pub struct ReversibleLayer {
    w_f: Vec<Vec<f64>>, // (d × d)
    w_g: Vec<Vec<f64>>,
    d: usize,
}

impl ReversibleLayer {
    /// Create a new reversible layer with dimension `d`.
    pub fn new(d: usize, seed: u64) -> EtResult<Self> {
        let scale = (1.0 / d as f64).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            w_f: rand_matrix(&mut rng, d, d, scale),
            w_g: rand_matrix(&mut rng, d, d, scale),
            d,
        })
    }

    fn apply_mlp(w: &[Vec<f64>], x: &[f64]) -> EtResult<Vec<f64>> {
        if w.is_empty() || w[0].len() != x.len() {
            return Err("ReversibleLayer: dimension mismatch in MLP".into());
        }
        let out: Vec<f64> = w.iter().map(|row| relu(dot(row, x))).collect();
        Ok(out)
    }

    /// Forward: `(x1, x2) → (y1, y2)`.
    pub fn forward(&self, x1: &[f64], x2: &[f64]) -> EtResult<(Vec<f64>, Vec<f64>)> {
        let f_x2 = Self::apply_mlp(&self.w_f, x2)?;
        let y1: Vec<f64> = x1.iter().zip(f_x2.iter()).map(|(a, b)| a + b).collect();
        let g_y1 = Self::apply_mlp(&self.w_g, &y1)?;
        let y2: Vec<f64> = x2.iter().zip(g_y1.iter()).map(|(a, b)| a + b).collect();
        Ok((y1, y2))
    }

    /// Backward (reconstruction): `(y1, y2) → (x1, x2)`.
    pub fn backward(&self, y1: &[f64], y2: &[f64]) -> EtResult<(Vec<f64>, Vec<f64>)> {
        let g_y1 = Self::apply_mlp(&self.w_g, y1)?;
        let x2: Vec<f64> = y2.iter().zip(g_y1.iter()).map(|(a, b)| a - b).collect();
        let f_x2 = Self::apply_mlp(&self.w_f, &x2)?;
        let x1: Vec<f64> = y1.iter().zip(f_x2.iter()).map(|(a, b)| a - b).collect();
        Ok((x1, x2))
    }
}

/// MLP-Mixer: token-mixing MLP + channel-mixing MLP.
///
/// Input shape: (N_tokens × d_channels).
pub struct MixerLayer {
    /// Token-mixing weight (N_tokens × N_tokens) – transposed application.
    w_token: Vec<Vec<f64>>,
    /// Channel-mixing weight (d_channels × d_channels).
    w_channel: Vec<Vec<f64>>,
    n_tokens: usize,
    d_channels: usize,
}

impl MixerLayer {
    /// Create a new MLP-Mixer layer.
    pub fn new(n_tokens: usize, d_channels: usize, seed: u64) -> EtResult<Self> {
        let scale = 0.02;
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            w_token: rand_matrix(&mut rng, n_tokens, n_tokens, scale),
            w_channel: rand_matrix(&mut rng, d_channels, d_channels, scale),
            n_tokens,
            d_channels,
        })
    }

    /// Forward pass for MLP-Mixer.
    pub fn forward(&self, x: &[Vec<f64>]) -> EtResult<Vec<Vec<f64>>> {
        let n = x.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let d = x[0].len();
        // Layer-norm over each token row
        let x_ln: Vec<Vec<f64>> = x.iter().map(|r| layer_norm(r, None, None)).collect();
        // Token mixing: operate on columns (across tokens)
        let x_t = transpose(&x_ln);
        let mut mixed_t = vec![vec![0.0_f64; n]; d];
        let w_t_use = &self.w_token;
        for c in 0..d {
            // Apply w_token (n_tokens × n_tokens, clamp to actual n)
            let w_rows = w_t_use.len().min(n);
            for i in 0..n {
                let row_i = i.min(w_rows.saturating_sub(1));
                let s: f64 = (0..n)
                    .map(|j| {
                        let col_j = j.min(w_t_use[row_i].len().saturating_sub(1));
                        w_t_use[row_i][col_j] * x_t[c][j]
                    })
                    .sum();
                mixed_t[c][i] = relu(s);
            }
        }
        let x_token_mixed = transpose(&mixed_t);
        // Residual connection
        let x2: Vec<Vec<f64>> = x
            .iter()
            .zip(x_token_mixed.iter())
            .map(|(xi, mi)| xi.iter().zip(mi.iter()).map(|(a, b)| a + b).collect())
            .collect();
        // Channel mixing: layer-norm then MLP per token
        let x2_ln: Vec<Vec<f64>> = x2.iter().map(|r| layer_norm(r, None, None)).collect();
        let w_c = &self.w_channel;
        let channel_mixed: Vec<Vec<f64>> = x2_ln
            .iter()
            .map(|row| {
                w_c.iter()
                    .map(|w_row| {
                        let len = w_row.len().min(d);
                        relu(
                            w_row[..len]
                                .iter()
                                .zip(row[..len].iter())
                                .map(|(a, b)| a * b)
                                .sum(),
                        )
                    })
                    .collect()
            })
            .collect();
        // Residual
        let out: Vec<Vec<f64>> = x2
            .iter()
            .zip(channel_mixed.iter())
            .map(|(xi, ci)| {
                let ci_len = ci.len().min(xi.len());
                let out_len = xi.len();
                (0..out_len)
                    .map(|j| xi[j] + if j < ci_len { ci[j] } else { 0.0 })
                    .collect()
            })
            .collect();
        Ok(out)
    }
}

/// FNet: replace self-attention with 2-D FFT over (seq_len × d_model).
///
/// Implements: `FFT2D(x).real + x` as the token-mixing step.
pub struct FNetLayer {
    d_model: usize,
}

impl FNetLayer {
    /// Create a new FNet layer.
    pub fn new(d_model: usize) -> Self {
        Self { d_model }
    }

    /// Apply DFT along seq dimension only (1-D DFT per feature dim) using the real part.
    pub fn forward(&self, x: &[Vec<f64>]) -> EtResult<Vec<Vec<f64>>> {
        let n = x.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let d = x[0].len();
        // DFT along sequence dimension for each feature channel
        let mut fft_out = vec![vec![0.0_f64; d]; n];
        for c in 0..d {
            let col: Vec<f64> = x.iter().map(|r| r[c]).collect();
            let dft_real = dft_real(&col);
            for i in 0..n {
                fft_out[i][c] = dft_real[i];
            }
        }
        // Add residual and layer-norm
        let out: Vec<Vec<f64>> = x
            .iter()
            .zip(fft_out.iter())
            .map(|(xi, fi)| {
                let mixed: Vec<f64> = xi.iter().zip(fi.iter()).map(|(a, b)| a + b).collect();
                layer_norm(&mixed, None, None)
            })
            .collect();
        Ok(out)
    }
}

/// Naive DFT real part: `Re[X[k]] = Σ_n x[n] cos(2π k n / N)`.
pub(crate) fn dft_real(x: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let len = x.len();
    (0..len)
        .map(|k| {
            x.iter()
                .enumerate()
                .map(|(j, xj)| {
                    let angle = 2.0 * std::f64::consts::PI * k as f64 * j as f64 / n;
                    xj * angle.cos()
                })
                .sum()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  Sub-Quadratic Attention
// ─────────────────────────────────────────────────────────────────────────────

/// Diagonal SSM (S4-inspired): `y[t] = C·(sI-A)^{-1}·B·u` simplified as
/// `y[t] = Σ_j C[j] * B[j] / (s - A[j]) * u[t]`, approximated by a
/// recurrent scan: `h[t] = A⊙h[t-1] + B⊙u[t]`,  `y[t] = C·h[t]`.
pub struct StateSpaceAttention;

impl StateSpaceAttention {
    /// Create a new state-space attention.
    pub fn new() -> Self {
        Self
    }

    /// Forward scan: `h[t] = A⊙h[t-1] + B·u[t]`, `y[t] = C·h[t]`.
    ///
    /// - `u`: input sequence (length T)
    /// - `A`, `B`, `C`: diagonal SSM parameters (length N_state)
    pub fn forward(u: &[f64], a: &[f64], b: &[f64], c: &[f64]) -> EtResult<Vec<f64>> {
        let n = a.len();
        if b.len() != n || c.len() != n {
            return Err("StateSpaceAttention: A, B, C must have the same length".into());
        }
        let t_len = u.len();
        let mut h = vec![0.0_f64; n];
        let mut y = Vec::with_capacity(t_len);
        for t in 0..t_len {
            h = h
                .iter()
                .zip(a.iter())
                .zip(b.iter())
                .map(|((hi, ai), bi)| ai * hi + bi * u[t])
                .collect();
            y.push(dot(c, &h));
        }
        Ok(y)
    }
}

impl Default for StateSpaceAttention {
    fn default() -> Self {
        Self::new()
    }
}

/// RetNet parallel form: `Ret(Q,K,V) = (Q·K^T ⊙ Γ) · V`
/// where `Γ[i,j] = γ^{i-j}` for `i≥j` else 0.
pub struct RetentiveNetworkLayer {
    d_model: usize,
}

impl RetentiveNetworkLayer {
    /// Create a new retentive network layer.
    pub fn new(d_model: usize) -> Self {
        Self { d_model }
    }

    /// Parallel retention: `(N × d_model)` → `(N × d_v)`.
    pub fn forward_parallel(
        &self,
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
        gamma: f64,
    ) -> EtResult<Vec<Vec<f64>>> {
        let n = q.len();
        let d = q[0].len().max(1);
        let k_t = transpose(k);
        let mut s = matmul(q, &k_t)?;
        let scale = 1.0 / (d as f64).sqrt();
        // Apply causal decay mask Γ[i,j] = γ^{i-j} for i≥j, else 0
        for i in 0..n {
            for j in 0..n {
                if j <= i {
                    let decay = gamma.powi((i - j) as i32);
                    s[i][j] *= scale * decay;
                } else {
                    s[i][j] = 0.0;
                }
            }
        }
        matmul(&s, v)
    }
}

/// MEGA layer: exponential moving average + single-head gated attention.
pub struct MegaLayer {
    d_model: usize,
    d_hidden: usize,
    w_gate: Vec<Vec<f64>>,
    w_out: Vec<Vec<f64>>,
}

impl MegaLayer {
    /// Create a new MEGA layer.
    pub fn new(d_model: usize, d_hidden: usize, seed: u64) -> EtResult<Self> {
        let scale = 0.02;
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            d_model,
            d_hidden,
            w_gate: rand_matrix(&mut rng, d_model, d_model, scale),
            w_out: rand_matrix(&mut rng, d_model, d_model, scale),
        })
    }

    /// Forward: EMA + gated attention over `(N × d_model)`.
    ///
    /// - `alpha`: per-dim EMA decay (length d_model)
    /// - `beta`: per-dim EMA scale  (length d_model)
    pub fn forward(&self, x: &[Vec<f64>], alpha: &[f64], beta: &[f64]) -> EtResult<Vec<Vec<f64>>> {
        let n = x.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let d = x[0].len();
        if alpha.len() != d || beta.len() != d {
            return Err("MegaLayer: alpha/beta must match d_model".into());
        }
        // EMA over sequence: h[t] = α⊙h[t-1] + (1-α)⊙(β⊙x[t])
        let mut h = vec![0.0_f64; d];
        let mut ema_out = Vec::with_capacity(n);
        for t in 0..n {
            let new_h: Vec<f64> = (0..d)
                .map(|j| {
                    let a_j = alpha[j].clamp(0.0, 1.0);
                    a_j * h[j] + (1.0 - a_j) * beta[j] * x[t][j]
                })
                .collect();
            h = new_h.clone();
            ema_out.push(new_h);
        }
        // Single-head attention gate: gate = sigmoid(W_gate · x)
        // out = gate ⊙ ema + (1-gate) ⊙ x, then project
        let w_g = &self.w_gate;
        let w_o = &self.w_out;
        let out: Vec<Vec<f64>> = (0..n)
            .map(|t| {
                let gate: Vec<f64> = w_g
                    .iter()
                    .map(|row| {
                        let len = row.len().min(x[t].len());
                        let s: f64 = row[..len]
                            .iter()
                            .zip(x[t][..len].iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        1.0 / (1.0 + (-s).exp()) // sigmoid
                    })
                    .collect();
                let gated: Vec<f64> = (0..gate.len().min(d))
                    .map(|j| gate[j] * ema_out[t][j] + (1.0 - gate[j]) * x[t][j])
                    .collect();
                w_o.iter()
                    .map(|row| {
                        let len = row.len().min(gated.len());
                        relu(
                            row[..len]
                                .iter()
                                .zip(gated[..len].iter())
                                .map(|(a, b)| a * b)
                                .sum(),
                        )
                    })
                    .collect()
            })
            .collect();
        Ok(out)
    }
}

/// Gated Linear Attention: `O = (Q · (K^T · V)) / Z ⊙ gate`.
pub struct GatedLinearAttention;

impl GatedLinearAttention {
    /// Create a new gated linear attention.
    pub fn new() -> Self {
        Self
    }

    /// Forward: `(N×d)` Q, K, V + `(N×d)` gate → `(N×d_v)`.
    pub fn forward(
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
        gate: &[Vec<f64>],
    ) -> EtResult<Vec<Vec<f64>>> {
        let n = q.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let d_v = v[0].len();
        let d_k = k[0].len();
        // K^T · V  (d_k × d_v)
        let k_t = transpose(k);
        let kv = matmul(&k_t, v)?;
        // Q · kv  (N × d_v)
        let qkv = matmul(q, &kv)?;
        // Normalizer Z[i] = Σ_j Q[i,j] * Σ_k K[k,j]
        let k_sum: Vec<f64> = (0..d_k)
            .map(|j| k.iter().map(|r| r[j]).sum::<f64>())
            .collect();
        let z: Vec<f64> = q
            .iter()
            .map(|row| dot(row, &k_sum[..row.len().min(d_k)]).abs().max(1e-6))
            .collect();
        let out: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let g_i = &gate[i];
                qkv[i]
                    .iter()
                    .enumerate()
                    .map(|(j, val)| {
                        let g = if j < g_i.len() {
                            1.0 / (1.0 + (-g_i[j]).exp())
                        } else {
                            0.5
                        };
                        (val / z[i]) * g
                    })
                    .collect()
            })
            .collect();
        Ok(out)
    }
}

impl Default for GatedLinearAttention {
    fn default() -> Self {
        Self::new()
    }
}

/// Mamba-2 simplified layer with selective scan.
pub struct Mamba2Layer {
    d_model: usize,
    d_state: usize,
    w_proj: Vec<Vec<f64>>,
}

impl Mamba2Layer {
    /// Create a new Mamba-2 layer.
    pub fn new(d_model: usize, d_state: usize, seed: u64) -> EtResult<Self> {
        let scale = 0.02;
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            d_model,
            d_state,
            w_proj: rand_matrix(&mut rng, d_model, d_model, scale),
        })
    }

    /// Selective scan: `h[t] = diag(A^delta[t]) · h[t-1] + delta[t]⊙B[t]⊙u[t]`,
    /// `y[t] = C[t] · h[t]`.
    ///
    /// - `u`: (T,) input
    /// - `delta`: (T,) step sizes (softplus-activated internally)
    /// - `A`: (d_state,) state decay (negative log scale)
    /// - `B`: (T, d_state) input projection
    /// - `C`: (T, d_state) output projection
    pub fn selective_scan(
        u: &[f64],
        delta: &[f64],
        a: &[f64],
        b: &[Vec<f64>],
        c: &[Vec<f64>],
    ) -> EtResult<Vec<f64>> {
        let t_len = u.len();
        let d_state = a.len();
        if delta.len() != t_len || b.len() != t_len || c.len() != t_len {
            return Err("Mamba2Layer::selective_scan: length mismatch".into());
        }
        let mut h = vec![0.0_f64; d_state];
        let mut y = Vec::with_capacity(t_len);
        for t in 0..t_len {
            // delta_t = softplus(delta[t])
            let dt = softplus_f64(delta[t]);
            // A_bar[j] = exp(delta_t * A[j])
            let a_bar: Vec<f64> = a.iter().map(|aj| (dt * aj).exp()).collect();
            // B_bar[j] = delta_t * B[t][j]
            let b_t = if b[t].len() >= d_state {
                b[t][..d_state].to_vec()
            } else {
                let mut v = b[t].to_vec();
                v.resize(d_state, 0.0);
                v
            };
            h = (0..d_state)
                .map(|j| a_bar[j] * h[j] + dt * b_t[j] * u[t])
                .collect();
            let c_t = if c[t].len() >= d_state {
                &c[t][..d_state]
            } else {
                &c[t][..]
            };
            y.push(dot(c_t, &h[..c_t.len()]));
        }
        Ok(y)
    }
}

pub(crate) fn softplus_f64(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0_f64 + x.exp()).ln()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  Efficient Position Encodings
// ─────────────────────────────────────────────────────────────────────────────

/// ALiBi: additive linear bias `−m · |i−j|` per head.
pub struct AliBiPositionBias;

impl AliBiPositionBias {
    /// Create a new ALiBi position bias.
    pub fn new() -> Self {
        Self
    }

    /// Return bias matrix `(n_heads × seq_len × seq_len)` flattened as
    /// `Vec<Vec<f64>>` of shape `(n_heads × seq_len², seq_len)`.
    /// Actually returns `Vec<Vec<f64>>` where `[h][i*seq_len + j]` is bias for head h, pos i,j.
    pub fn compute_bias(seq_len: usize, n_heads: usize) -> Vec<Vec<f64>> {
        // Slopes: m_h = 2^{-8h/n_heads}
        let slopes: Vec<f64> = (1..=n_heads)
            .map(|h| (-(8.0 * h as f64 / n_heads as f64)).exp2())
            .collect();
        slopes
            .iter()
            .map(|&m| {
                (0..seq_len)
                    .flat_map(|i| (0..seq_len).map(move |j| -m * (i as f64 - j as f64).abs()))
                    .collect()
            })
            .collect()
    }
}

impl Default for AliBiPositionBias {
    fn default() -> Self {
        Self::new()
    }
}

/// Rotary Position Embedding (RoPE).
pub struct RopeEncoding {
    theta_base: f64,
}

impl RopeEncoding {
    /// Create a new RoPE encoding with the given theta base.
    pub fn new(theta_base: f64) -> Self {
        Self { theta_base }
    }

    /// Apply RoPE to Q and K.
    ///
    /// Each token at position `t` gets its `d`-dimensional vector rotated
    /// by `θ_i = t / θ_base^{2i/d}`.
    pub fn apply_rope(
        &self,
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        theta_base: f64,
    ) -> EtResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let q_out = Self::rotate_embeddings(q, theta_base)?;
        let k_out = Self::rotate_embeddings(k, theta_base)?;
        Ok((q_out, k_out))
    }

    fn rotate_embeddings(x: &[Vec<f64>], theta_base: f64) -> EtResult<Vec<Vec<f64>>> {
        x.iter()
            .enumerate()
            .map(|(pos, row)| {
                let d = row.len();
                if d % 2 != 0 {
                    return Err("RoPE requires even embedding dimension".into());
                }
                let mut out = row.clone();
                for i in 0..(d / 2) {
                    let theta = pos as f64 / theta_base.powf(2.0 * i as f64 / d as f64);
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();
                    let x0 = row[2 * i];
                    let x1 = row[2 * i + 1];
                    out[2 * i] = x0 * cos_t - x1 * sin_t;
                    out[2 * i + 1] = x0 * sin_t + x1 * cos_t;
                }
                Ok(out)
            })
            .collect()
    }
}

/// YaRN: interpolated RoPE for longer sequences.
pub struct YarnRope;

impl YarnRope {
    /// Create a new YaRN RoPE.
    pub fn new() -> Self {
        Self
    }

    /// Interpolate position encoding for position `pos`, dim index `dim`,
    /// with `scale_factor` (sequence length extension ratio) and base `base`.
    ///
    /// Returns `(cos_theta, sin_theta)`.
    pub fn interpolate(
        pos: usize,
        dim: usize,
        total_dim: usize,
        scale_factor: f64,
        base: f64,
    ) -> (f64, f64) {
        // YaRN splits dimensions into NTK-by-parts and linear interpolation regions
        // Simplified: theta = pos / (scale_factor * base^{2*dim/total_dim})
        let alpha = 1.0; // NTK alpha factor
        let beta = 32.0; // NTK beta factor
        let r = base.powf(2.0 * dim as f64 / total_dim as f64);
        // Interpolation weighting
        let w = if r < alpha {
            1.0 / scale_factor
        } else if r > beta {
            1.0
        } else {
            (1.0 - 1.0 / scale_factor) * (r - alpha) / (beta - alpha) + 1.0 / scale_factor
        };
        let theta = pos as f64 * w / r;
        (theta.cos(), theta.sin())
    }
}

impl Default for YarnRope {
    fn default() -> Self {
        Self::new()
    }
}

/// xPos: position-dependent exponential decay for better length generalization.
pub struct Xpos {
    scale_base: f64,
}

impl Xpos {
    /// Create a new xPos encoding with the given scale base.
    pub fn new(scale_base: f64) -> Self {
        Self { scale_base }
    }

    /// Apply xPos to Q and K with decay.
    ///
    /// `scale(t, i) = ((t + 0.4 * d) / (1.4 * d))^{-2i/d}` approximated as
    /// `exp(scale_base^{-2i/d} * t / scale_base)`.
    pub fn apply(
        &self,
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        scale: f64,
        base: f64,
    ) -> EtResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let q_out = self.xpos_rotate(q, scale, base, true)?;
        let k_out = self.xpos_rotate(k, scale, base, false)?;
        Ok((q_out, k_out))
    }

    fn xpos_rotate(
        &self,
        x: &[Vec<f64>],
        scale: f64,
        base: f64,
        forward: bool,
    ) -> EtResult<Vec<Vec<f64>>> {
        x.iter()
            .enumerate()
            .map(|(pos, row)| {
                let d = row.len();
                if d % 2 != 0 {
                    return Err("xPos requires even dimension".into());
                }
                let mut out = row.clone();
                for i in 0..(d / 2) {
                    let theta = pos as f64 / base.powf(2.0 * i as f64 / d as f64);
                    let decay_exp = (scale * pos as f64) / self.scale_base;
                    let decay = if forward {
                        base.powf(-decay_exp / d as f64)
                    } else {
                        base.powf(decay_exp / d as f64)
                    };
                    let cos_t = theta.cos() * decay;
                    let sin_t = theta.sin() * decay;
                    let x0 = row[2 * i];
                    let x1 = row[2 * i + 1];
                    out[2 * i] = x0 * cos_t - x1 * sin_t;
                    out[2 * i + 1] = x0 * sin_t + x1 * cos_t;
                }
                Ok(out)
            })
            .collect()
    }
}

/// Random Fourier Feature position encoding.
pub struct RandomFourierFeaturePos {
    seed: u64,
}

impl RandomFourierFeaturePos {
    /// Create a new random Fourier feature position encoding.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Encode scalar position `pos` into a `dim`-dimensional vector using
    /// random Fourier features: `φ(pos) = [cos(ω_1·pos), sin(ω_1·pos), …]`.
    pub fn encode(&self, pos: usize, dim: usize, rng: &mut StdRng) -> Vec<f64> {
        let half = dim / 2;
        let omega: Vec<f64> = (0..half).map(|_| rng.random::<f64>() * 10.0).collect();
        let mut feat = Vec::with_capacity(dim);
        let scale = (1.0 / half as f64).sqrt();
        for &w in omega.iter() {
            feat.push(scale * (w * pos as f64).cos());
            feat.push(scale * (w * pos as f64).sin());
        }
        feat.truncate(dim);
        feat
    }
}
