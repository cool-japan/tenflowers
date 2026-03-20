//! Neural Rendering & 3D Vision.
//!
//! Implements cutting-edge neural rendering techniques:
//!
//! - **NeRF**: [`PositionalEncoding`], [`NeRFMlp`], [`VolumeRenderer`], [`RayMarcher`], [`NeRFLoss`]
//! - **3D Gaussian Splatting**: [`Gaussian3D`], [`GaussianSplatRenderer`], [`Gaussian2D`], [`GaussianOptimizer`], [`QuaternionOps`]
//! - **Implicit Neural Representations**: [`SirenLayer`], [`SirenNetwork`], [`NeuralSdf`], [`InstantNgp`], [`OccupancyNetwork`]
//! - **Scene Understanding**: [`DepthEstimationNet`], [`SurfaceNormalEstimator`], [`SemanticNerfDecoder`], [`PanopticLiftingHead`], [`SceneFlowEstimator`]
//! - **Camera & View Synthesis**: [`CameraModel`], [`MultiViewConsistencyLoss`], [`PoseEstimator`], [`ViewInterpolator`], [`CameraOptimizer`]

pub mod extensions;
pub use extensions::*;

pub mod advanced;
// Explicit re-exports from advanced to avoid name conflicts with mod-level types.
// Note: advanced::Gaussian3D is a different type (with SH coefficients) and is
// accessed as neural_rendering::advanced::Gaussian3D to avoid collision.
pub use advanced::{
    DeformationField, DynamicNerf, GaussianDensification, GaussianSplatter,
    NrcCache, NrMetrics, Reservoir, ReSTIR, ShCoefficients,
};

#[cfg(test)]
pub mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Shared utilities
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub(crate) fn relu(x: f64) -> f64 {
    x.max(0.0)
}

#[inline]
pub(crate) fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
pub(crate) fn softplus(x: f64) -> f64 {
    (1.0 + x.exp()).ln()
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn matvec(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter().map(|row| dot(row, v)).collect()
}

pub(crate) fn rand_weight(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let scale = (2.0 / cols as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                .collect()
        })
        .collect()
}

pub(crate) fn linear(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    let out = matvec(w, x);
    out.iter().zip(b).map(|(o, bi)| o + bi).collect()
}

pub(crate) fn linear_relu(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    linear(w, b, x).into_iter().map(relu).collect()
}

pub(crate) fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < 1e-300 {
        v
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

pub(crate) fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  Neural Radiance Fields (NeRF)
// ─────────────────────────────────────────────────────────────────────────────

/// Fourier positional encoding used by NeRF.
///
/// For a scalar `x` and `n_freqs` frequencies the encoding is:
/// `[sin(2^0 x), cos(2^0 x), sin(2^1 x), cos(2^1 x), …, sin(2^{L-1} x), cos(2^{L-1} x)]`
/// giving a vector of length `2 * n_freqs`.
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    /// Number of frequency bands L.
    pub n_freqs: usize,
    /// Whether to include the raw input (identity) in the encoding.
    pub include_input: bool,
}

impl PositionalEncoding {
    /// Create a new encoding with `n_freqs` frequency bands.
    pub fn new(n_freqs: usize) -> Self {
        Self {
            n_freqs,
            include_input: false,
        }
    }

    /// Create a new encoding including the raw input.
    pub fn with_identity(n_freqs: usize) -> Self {
        Self {
            n_freqs,
            include_input: true,
        }
    }

    /// Encode a scalar `x`.  Returns a vector of length `2 * n_freqs` (or
    /// `2 * n_freqs + 1` when `include_input` is true).
    pub fn encode_scalar(&self, x: f64) -> Vec<f64> {
        let mut out = if self.include_input { vec![x] } else { vec![] };
        for i in 0..self.n_freqs {
            let freq = (2.0_f64).powi(i as i32);
            out.push((freq * x).sin());
            out.push((freq * x).cos());
        }
        out
    }

    /// Encode an arbitrary-length input vector by encoding each element
    /// independently and concatenating the results.
    pub fn encode(&self, x: &[f64]) -> Vec<f64> {
        x.iter().flat_map(|&xi| self.encode_scalar(xi)).collect()
    }

    /// Output dimension for a single scalar input.
    pub fn output_dim_scalar(&self) -> usize {
        2 * self.n_freqs + if self.include_input { 1 } else { 0 }
    }

    /// Output dimension for an input of length `input_len`.
    pub fn output_dim(&self, input_len: usize) -> usize {
        self.output_dim_scalar() * input_len
    }
}

/// NeRF MLP: (point, direction) → (RGB, density σ).
/// 4-layer backbone with skip connection + density head + direction-conditioned colour head.
#[derive(Debug, Clone)]
pub struct NeRFMlp {
    pub point_enc: PositionalEncoding,
    pub dir_enc: PositionalEncoding,
    backbone_w: Vec<Vec<Vec<f64>>>,
    backbone_b: Vec<Vec<f64>>,
    density_w: Vec<Vec<f64>>,
    density_b: Vec<f64>,
    feature_w: Vec<Vec<f64>>,
    feature_b: Vec<f64>,
    colour_w: Vec<Vec<f64>>,
    colour_b: Vec<f64>,
    hidden_dim: usize,
}

impl NeRFMlp {
    /// Build NeRF MLP. `point_n_freqs`/`dir_n_freqs`: encoding bands; `hidden_dim`: layer width.
    pub fn new(point_n_freqs: usize, dir_n_freqs: usize, hidden_dim: usize) -> Self {
        let point_enc = PositionalEncoding::new(point_n_freqs);
        let dir_enc = PositionalEncoding::new(dir_n_freqs);
        let pt_dim = point_enc.output_dim(3);
        let dir_dim = dir_enc.output_dim(3);

        let mut seed = 42_u64;
        let mut mk = |r: usize, c: usize| {
            let w = rand_weight(r, c, seed);
            seed += 1;
            (w, vec![0.0_f64; r])
        };

        // 4 backbone layers; layer 0 takes pt_dim, layers 1–3 take hidden_dim
        // layer 2 (index 2) gets pt_dim concatenated → input = hidden_dim + pt_dim
        let mut backbone_w = Vec::new();
        let mut backbone_b = Vec::new();
        for i in 0..4 {
            let in_dim = if i == 0 {
                pt_dim
            } else if i == 2 {
                hidden_dim + pt_dim
            } else {
                hidden_dim
            };
            let (w, b) = mk(hidden_dim, in_dim);
            backbone_w.push(w);
            backbone_b.push(b);
        }

        let (density_w, density_b) = mk(1, hidden_dim);
        let (feature_w, feature_b) = mk(hidden_dim, hidden_dim);
        let (colour_w, colour_b) = mk(3, hidden_dim + dir_dim);

        Self {
            point_enc,
            dir_enc,
            backbone_w,
            backbone_b,
            density_w,
            density_b,
            feature_w,
            feature_b,
            colour_w,
            colour_b,
            hidden_dim,
        }
    }

    /// Forward pass.
    ///
    /// Returns `(rgb, sigma)` where `rgb` is `[r,g,b]` ∈ \[0,1\]³ and
    /// `sigma` ≥ 0 is the volumetric density.
    pub fn forward(&self, point: &[f64], direction: &[f64]) -> Result<([f64; 3], f64)> {
        if point.len() < 3 {
            return Err(TensorError::invalid_argument_op(
                "nerf_mlp",
                "point must have at least 3 elements",
            ));
        }
        if direction.len() < 3 {
            return Err(TensorError::invalid_argument_op(
                "nerf_mlp",
                "direction must have at least 3 elements",
            ));
        }

        let pt_enc = self.point_enc.encode(&point[..3]);
        let dir_enc = self.dir_enc.encode(&direction[..3]);

        // backbone with skip connection
        let mut h = pt_enc.clone();
        for i in 0..4 {
            if i == 2 {
                // skip: concatenate original point encoding
                let mut inp = h.clone();
                inp.extend_from_slice(&pt_enc);
                h = linear_relu(&self.backbone_w[i], &self.backbone_b[i], &inp);
            } else {
                h = linear_relu(&self.backbone_w[i], &self.backbone_b[i], &h);
            }
        }

        // density head (softplus for non-negativity)
        let raw_sigma = linear(&self.density_w, &self.density_b, &h)[0];
        let sigma = softplus(raw_sigma);

        // feature → colour head conditioned on direction
        let feat = linear_relu(&self.feature_w, &self.feature_b, &h);
        let mut feat_dir = feat;
        feat_dir.extend_from_slice(&dir_enc);
        let rgb_raw = linear(&self.colour_w, &self.colour_b, &feat_dir);

        let rgb = [
            sigmoid(rgb_raw[0]),
            sigmoid(rgb_raw.get(1).copied().unwrap_or(0.0)),
            sigmoid(rgb_raw.get(2).copied().unwrap_or(0.0)),
        ];

        Ok((rgb, sigma))
    }
}

/// Alpha compositing volume renderer (NeRF discrete rendering equation).
/// `T_i = exp(-Σ σ_j δ_j)`, `w_i = T_i(1-exp(-σ_i δ_i))`, `C = Σ w_i c_i`.
#[derive(Debug, Clone)]
pub struct VolumeRenderer {
    /// Background colour blended with `(1 - accumulated_alpha)`.
    pub background: [f64; 3],
}

impl VolumeRenderer {
    /// Create a new volume renderer with black background.
    pub fn new() -> Self {
        Self {
            background: [0.0; 3],
        }
    }
    /// Create a volume renderer with a specific background colour.
    pub fn with_background(background: [f64; 3]) -> Self {
        Self { background }
    }

    /// Render a ray. `samples`: `(sigma, delta, rgb)` ordered near→far.
    /// Returns `(colour, per-sample weights)`.
    pub fn render(&self, samples: &[(f64, f64, [f64; 3])]) -> ([f64; 3], Vec<f64>) {
        let mut transmittance = 1.0_f64;
        let mut colour = [0.0_f64; 3];
        let mut weights = Vec::with_capacity(samples.len());

        for &(sigma, delta, rgb) in samples {
            let alpha = 1.0 - (-sigma * delta).exp();
            let weight = transmittance * alpha;
            weights.push(weight);

            for ch in 0..3 {
                colour[ch] += weight * rgb[ch];
            }

            transmittance *= 1.0 - alpha;
            if transmittance < 1e-10 {
                // fill remaining weights with zero
                weights.resize(samples.len(), 0.0);
                break;
            }
        }

        // blend background
        for ch in 0..3 {
            colour[ch] += transmittance * self.background[ch];
        }

        // pad weights to match samples length
        weights.resize(samples.len(), 0.0);

        (colour, weights)
    }

    /// Convenience overload matching the task signature:
    /// `samples: &[(sigma: f64, rgb: [f64;3])]` with uniform step δ = 1/(n-1).
    pub fn render_uniform(&self, samples: &[(f64, [f64; 3])]) -> [f64; 3] {
        let n = samples.len();
        if n == 0 {
            return self.background;
        }
        let delta = if n > 1 { 1.0 / (n as f64 - 1.0) } else { 1.0 };
        let full: Vec<_> = samples.iter().map(|&(s, rgb)| (s, delta, rgb)).collect();
        self.render(&full).0
    }
}

impl Default for VolumeRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Stratified ray marcher: samples `n_samples` points near→far along a ray.
#[derive(Debug, Clone)]
pub struct RayMarcher {
    pub near: f64,
    pub far: f64,
    pub n_samples: usize,
    pub stratified: bool,
}

impl RayMarcher {
    /// Create a new stratified ray marcher.
    pub fn new(near: f64, far: f64, n_samples: usize) -> Self {
        Self {
            near,
            far,
            n_samples,
            stratified: true,
        }
    }

    /// Returns `Vec<(point, t)>` sampled stratified along the ray.
    pub fn sample_points(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        rng: &mut StdRng,
    ) -> Vec<([f64; 3], f64)> {
        let n = self.n_samples.max(1);
        let step = (self.far - self.near) / n as f64;
        (0..n)
            .map(|i| {
                let t_low = self.near + i as f64 * step;
                let jitter: f64 = if self.stratified {
                    rng.random::<f64>() * step
                } else {
                    step * 0.5
                };
                let t = (t_low + jitter).min(self.far);
                let pt = [
                    origin[0] + t * direction[0],
                    origin[1] + t * direction[1],
                    origin[2] + t * direction[2],
                ];
                (pt, t)
            })
            .collect()
    }
}

/// Photometric NeRF training loss.
///
/// L = (1/N) Σ_i ‖ĉ_i − c_i‖² + λ_depth · depth_consistency_term
#[derive(Debug, Clone)]
pub struct NeRFLoss {
    /// Weight for the optional depth consistency term.
    pub lambda_depth: f64,
}

impl NeRFLoss {
    /// Create a new NeRF loss with no depth regularisation.
    pub fn new() -> Self {
        Self { lambda_depth: 0.0 }
    }
    /// Create a new NeRF loss with depth regularisation weight `lambda_depth`.
    pub fn with_depth(lambda_depth: f64) -> Self {
        Self { lambda_depth }
    }

    /// Photometric MSE: (1/N) Σ ‖ĉ − c‖².
    pub fn photometric_mse(&self, rendered: &[[f64; 3]], target: &[[f64; 3]]) -> Result<f64> {
        if rendered.len() != target.len() {
            return Err(TensorError::invalid_argument_op(
                "nerf_loss",
                "rendered and target must have the same length",
            ));
        }
        if rendered.is_empty() {
            return Ok(0.0);
        }
        let n = rendered.len() as f64;
        let mse = rendered
            .iter()
            .zip(target)
            .map(|(r, t)| (r[0] - t[0]).powi(2) + (r[1] - t[1]).powi(2) + (r[2] - t[2]).powi(2))
            .sum::<f64>()
            / n;
        Ok(mse)
    }

    /// Full loss = MSE + λ_depth · depth_consistency.
    pub fn compute(
        &self,
        rendered: &[[f64; 3]],
        target: &[[f64; 3]],
        depth_rendered: Option<&[f64]>,
        depth_target: Option<&[f64]>,
    ) -> Result<f64> {
        let photo = self.photometric_mse(rendered, target)?;
        let depth_term = match (depth_rendered, depth_target) {
            (Some(dr), Some(dt)) if self.lambda_depth > 0.0 && dr.len() == dt.len() => {
                let n = dr.len().max(1) as f64;
                let d = dr.iter().zip(dt).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n;
                self.lambda_depth * d
            }
            _ => 0.0,
        };
        Ok(photo + depth_term)
    }
}

impl Default for NeRFLoss {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  3D Gaussian Splatting
// ─────────────────────────────────────────────────────────────────────────────

/// Quaternion `[w,x,y,z]` helper utilities.
#[derive(Debug, Clone)]
pub struct QuaternionOps;

impl QuaternionOps {
    /// Normalise to unit quaternion.
    pub fn normalize(q: [f64; 4]) -> [f64; 4] {
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        if n < 1e-300 {
            return [1.0, 0.0, 0.0, 0.0];
        }
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }

    /// Hamilton product.
    pub fn multiply(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        let (aw, ax, ay, az) = (a[0], a[1], a[2], a[3]);
        let (bw, bx, by, bz) = (b[0], b[1], b[2], b[3]);
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    }

    /// Convert unit quaternion to 3×3 rotation matrix (row-major).
    pub fn to_rotation_matrix(q: [f64; 4]) -> [[f64; 3]; 3] {
        let q = Self::normalize(q);
        let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }
}

/// A single 3D Gaussian primitive for 3DGS.
#[derive(Debug, Clone)]
pub struct Gaussian3D {
    /// Centre position `[x, y, z]`.
    pub center: [f64; 3],
    /// Log-scale parameters `[sx, sy, sz]` (exponentiated to give actual scale).
    pub log_scale: [f64; 3],
    /// Unit quaternion `[w, x, y, z]` representing orientation.
    pub rotation: [f64; 4],
    /// Logit-opacity (sigmoid → opacity in \[0,1\]).
    pub logit_opacity: f64,
    /// Spherical-harmonic DC coefficient as RGB colour.
    pub color: [f64; 3],
}

impl Gaussian3D {
    /// Create a new 3D Gaussian at the given center with uniform scale and color.
    pub fn new(center: [f64; 3], scale: f64, color: [f64; 3]) -> Self {
        let log_s = scale.abs().max(1e-10).ln();
        Self {
            center,
            log_scale: [log_s; 3],
            rotation: [1.0, 0.0, 0.0, 0.0],
            logit_opacity: 0.0,
            color,
        }
    }

    /// Get the scale of this Gaussian.
    pub fn scale(&self) -> [f64; 3] {
        [
            self.log_scale[0].exp(),
            self.log_scale[1].exp(),
            self.log_scale[2].exp(),
        ]
    }
    /// Get the opacity (sigmoid of logit).
    pub fn opacity(&self) -> f64 {
        sigmoid(self.logit_opacity)
    }

    /// Compute 3D covariance: Σ = R diag(s²) Rᵀ.
    pub fn covariance3d(&self) -> [[f64; 3]; 3] {
        let r = QuaternionOps::to_rotation_matrix(self.rotation);
        let s = self.scale();
        // R * diag(s^2) * R^T
        let mut cov = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] = (0..3).map(|k| r[i][k] * s[k] * s[k] * r[j][k]).sum();
            }
        }
        cov
    }
}

/// 2D Gaussian projected onto the image plane.
#[derive(Debug, Clone)]
pub struct Gaussian2D {
    pub center: [f64; 2],
    pub cov2d: [[f64; 2]; 2],
    pub opacity: f64,
    pub color: [f64; 3],
    /// Camera-space depth (for back-to-front sorting).
    pub depth: f64,
}

impl Gaussian2D {
    pub(crate) fn eval(&self, px: f64, py: f64) -> f64 {
        let dx = px - self.center[0];
        let dy = py - self.center[1];
        let det = self.cov2d[0][0] * self.cov2d[1][1] - self.cov2d[0][1] * self.cov2d[1][0];
        if det.abs() < 1e-12 {
            return 0.0;
        }
        let inv = [
            [self.cov2d[1][1] / det, -self.cov2d[0][1] / det],
            [-self.cov2d[1][0] / det, self.cov2d[0][0] / det],
        ];
        let q = dx * (inv[0][0] * dx + inv[0][1] * dy) + dy * (inv[1][0] * dx + inv[1][1] * dy);
        (-0.5 * q).exp()
    }
}

/// Tile-based 3DGS renderer.
#[derive(Debug, Clone)]
pub struct GaussianSplatRenderer {
    pub focal_length: f64,
    pub width: usize,
    pub height: usize,
}

impl GaussianSplatRenderer {
    /// Create a new Gaussian splat renderer.
    pub fn new(focal_length: f64, width: usize, height: usize) -> Self {
        Self {
            focal_length,
            width,
            height,
        }
    }

    /// Project a 3D Gaussian to 2D (pinhole model, simple translation).
    pub fn project(&self, g: &Gaussian3D, camera_pos: [f64; 3]) -> Option<Gaussian2D> {
        let cx = g.center[0] - camera_pos[0];
        let cy = g.center[1] - camera_pos[1];
        let cz = g.center[2] - camera_pos[2];
        if cz <= 0.001 {
            return None;
        }

        let f = self.focal_length;
        let px = f * cx / cz + self.width as f64 * 0.5;
        let py = f * cy / cz + self.height as f64 * 0.5;

        // Jacobian of perspective projection (2×3)
        let j = [
            [f / cz, 0.0, -f * cx / (cz * cz)],
            [0.0, f / cz, -f * cy / (cz * cz)],
        ];

        // 3D covariance
        let sigma3 = g.covariance3d();

        // Σ2D = J Σ3D Jᵀ
        // tmp[2×3] = J * Σ3D
        let mut tmp = [[0.0_f64; 3]; 2];
        for i in 0..2 {
            for k in 0..3 {
                tmp[i][k] = j[i]
                    .iter()
                    .enumerate()
                    .map(|(l, &jil)| jil * sigma3[l][k])
                    .sum();
            }
        }
        // cov2d[2×2] = tmp * Jᵀ
        let mut cov2d = [[0.0_f64; 2]; 2];
        for i in 0..2 {
            for k in 0..2 {
                cov2d[i][k] = (0..3).map(|l| tmp[i][l] * j[k][l]).sum::<f64>();
            }
        }
        // Add small regularisation to diagonal
        cov2d[0][0] += 0.3;
        cov2d[1][1] += 0.3;

        Some(Gaussian2D {
            center: [px, py],
            cov2d,
            opacity: g.opacity(),
            color: g.color,
            depth: cz,
        })
    }

    /// Render a single pixel `(px, py)` by alpha-blending sorted 2D Gaussians.
    ///
    /// Gaussians must be sorted back-to-front (largest depth first).
    pub fn render_pixel(&self, gaussians_2d: &[Gaussian2D], px: f64, py: f64) -> [f64; 3] {
        let mut colour = [0.0_f64; 3];
        let mut transmittance = 1.0_f64;
        for g in gaussians_2d {
            let alpha = g.opacity * g.eval(px, py);
            let w = transmittance * alpha;
            for ch in 0..3 {
                colour[ch] += w * g.color[ch];
            }
            transmittance *= 1.0 - alpha;
            if transmittance < 1e-4 {
                break;
            }
        }
        colour
    }

    /// Render a full image given a set of 3D Gaussians and camera position.
    pub fn render_image(
        &self,
        gaussians: &[Gaussian3D],
        camera_pos: [f64; 3],
    ) -> Vec<Vec<[f64; 3]>> {
        let mut g2d: Vec<Gaussian2D> = gaussians
            .iter()
            .filter_map(|g| self.project(g, camera_pos))
            .collect();
        // Sort back-to-front
        g2d.sort_by(|a, b| {
            b.depth
                .partial_cmp(&a.depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        (0..self.height)
            .map(|y| {
                (0..self.width)
                    .map(|x| self.render_pixel(&g2d, x as f64, y as f64))
                    .collect()
            })
            .collect()
    }
}

/// Finite-difference gradient optimiser for 3DGS: update centres, densify, prune.
#[derive(Debug, Clone)]
pub struct GaussianOptimizer {
    pub lr: f64,
    pub eps: f64,
    pub grad_accum: Vec<f64>,
    pub step: usize,
}

impl GaussianOptimizer {
    /// Create a new Gaussian optimizer with the given learning rate.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            eps: 1e-4,
            grad_accum: vec![],
            step: 0,
        }
    }

    /// Gradient descent on Gaussian centres. Returns loss before step.
    #[allow(clippy::ptr_arg)]
    pub fn step_once(
        &mut self,
        renderer: &GaussianSplatRenderer,
        gaussians: &mut Vec<Gaussian3D>,
        camera_pos: [f64; 3],
        target: &[[f64; 3]],
    ) -> f64 {
        let w = renderer.width;
        let h = renderer.height;
        self.grad_accum.resize(gaussians.len(), 0.0);

        let image0 = renderer.render_image(gaussians, camera_pos);
        let loss0 = mse_image(&image0, target, w, h);

        for gi in 0..gaussians.len() {
            let mut grad_norm = 0.0_f64;
            for ax in 0..3 {
                let orig = gaussians[gi].center[ax];
                gaussians[gi].center[ax] = orig + self.eps;
                let img_p = renderer.render_image(gaussians, camera_pos);
                let loss_p = mse_image(&img_p, target, w, h);
                gaussians[gi].center[ax] = orig;

                let grad = (loss_p - loss0) / self.eps;
                grad_norm += grad.abs();
                gaussians[gi].center[ax] -= self.lr * grad;
            }
            self.grad_accum[gi] += grad_norm;
        }
        self.step += 1;
        loss0
    }

    /// Split Gaussians with accumulated gradient > `threshold` into two.
    pub fn densify(&mut self, gaussians: &mut Vec<Gaussian3D>, threshold: f64) {
        let mut new_gaussians = Vec::new();
        let mut to_keep = Vec::new();
        for (gi, g) in gaussians.iter().enumerate() {
            let acc = self.grad_accum.get(gi).copied().unwrap_or(0.0);
            if acc > threshold {
                let s = g.scale()[0] * 0.5;
                let log_s = s.max(1e-10).ln();
                // Copy 1: slightly positive offset
                let mut g1 = g.clone();
                g1.center[0] += s;
                g1.log_scale = [log_s; 3];
                // Copy 2: slightly negative offset
                let mut g2 = g.clone();
                g2.center[0] -= s;
                g2.log_scale = [log_s; 3];
                new_gaussians.push(g1);
                new_gaussians.push(g2);
            } else {
                to_keep.push(gi);
            }
        }
        let kept: Vec<Gaussian3D> = to_keep.iter().map(|&i| gaussians[i].clone()).collect();
        *gaussians = kept;
        gaussians.extend(new_gaussians);
        self.grad_accum = vec![0.0; gaussians.len()];
    }

    /// Remove Gaussians with opacity below `opacity_threshold`.
    pub fn prune(&mut self, gaussians: &mut Vec<Gaussian3D>, opacity_threshold: f64) {
        gaussians.retain(|g| g.opacity() >= opacity_threshold);
        self.grad_accum.resize(gaussians.len(), 0.0);
    }
}

pub(crate) fn mse_image(image: &[Vec<[f64; 3]>], target: &[[f64; 3]], w: usize, h: usize) -> f64 {
    let n = (w * h).max(1) as f64;
    let mut sum = 0.0;
    for (row_idx, row) in image.iter().enumerate() {
        for (col_idx, &px) in row.iter().enumerate() {
            let flat = row_idx * w + col_idx;
            if let Some(&tgt) = target.get(flat) {
                sum +=
                    (px[0] - tgt[0]).powi(2) + (px[1] - tgt[1]).powi(2) + (px[2] - tgt[2]).powi(2);
            }
        }
    }
    sum / n
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  Implicit Neural Representations
// ─────────────────────────────────────────────────────────────────────────────

/// SIREN layer: `sin(ω₀ · (Wx + b))` (Sitzmann et al., 2020).
#[derive(Debug, Clone)]
pub struct SirenLayer {
    pub weight: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl SirenLayer {
    /// `is_first`: use `U(-1/in, 1/in)` init (first layer); else SIREN-scaled Xavier.
    pub fn new(in_dim: usize, out_dim: usize, omega_0: f64, seed: u64, is_first: bool) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let weight = if is_first {
            let bound = 1.0 / in_dim as f64;
            (0..out_dim)
                .map(|_| {
                    (0..in_dim)
                        .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * bound)
                        .collect()
                })
                .collect()
        } else {
            let bound = (6.0_f64 / in_dim as f64).sqrt() / omega_0;
            (0..out_dim)
                .map(|_| {
                    (0..in_dim)
                        .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * bound)
                        .collect()
                })
                .collect()
        };
        let bias = vec![0.0; out_dim];
        Self {
            weight,
            bias,
            in_dim,
            out_dim,
        }
    }

    /// Forward pass: `sin(ω₀ · (W x + b))`.
    pub fn forward(&self, x: &[f64], omega_0: f64) -> Vec<f64> {
        let pre = linear(&self.weight, &self.bias, x);
        pre.into_iter().map(|v| (omega_0 * v).sin()).collect()
    }
}

/// Stack of SIREN layers (all layers use sin; last layer is linear).
#[derive(Debug, Clone)]
pub struct SirenNetwork {
    pub layers: Vec<SirenLayer>,
    pub omega_0: f64,
}

impl SirenNetwork {
    /// Create a new SIREN network.
    pub fn new(in_dim: usize, hidden_dims: &[usize], out_dim: usize, omega_0: f64) -> Self {
        let mut layers = Vec::new();
        let mut seed = 100_u64;
        let mut prev = in_dim;
        for (i, &h) in hidden_dims.iter().enumerate() {
            layers.push(SirenLayer::new(prev, h, omega_0, seed, i == 0));
            seed += 1;
            prev = h;
        }
        layers.push(SirenLayer::new(prev, out_dim, omega_0, seed, false));
        Self { layers, omega_0 }
    }

    /// Forward pass through the SIREN network.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let n = self.layers.len();
        let mut h = x.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            h = if i < n - 1 {
                layer.forward(&h, self.omega_0)
            } else {
                linear(&layer.weight, &layer.bias, &h)
            };
        }
        h
    }
}

/// Neural SDF: SIREN predicting signed distance. Negative inside, positive outside.
#[derive(Debug, Clone)]
pub struct NeuralSdf {
    network: SirenNetwork,
}

impl NeuralSdf {
    /// Create a new neural SDF.
    pub fn new(hidden_dim: usize, n_layers: usize) -> Self {
        let hidden_dims = vec![hidden_dim; n_layers.max(1) - 1];
        Self {
            network: SirenNetwork::new(3, &hidden_dims, 1, 30.0),
        }
    }
    /// Forward pass: returns the signed distance at the given 3D point.
    pub fn forward(&self, point: &[f64]) -> f64 {
        self.network.forward(point).first().copied().unwrap_or(0.0)
    }
}

/// Instant NGP multi-resolution hash encoding (Müller et al., 2022).
/// Trilinear interpolation over a spatial hash table per level.
#[derive(Debug, Clone)]
pub struct InstantNgp {
    pub feature_dim: usize,
    tables: Vec<Vec<Vec<f64>>>,
    resolutions: Vec<usize>,
}

impl InstantNgp {
    /// `resolutions`: grid sizes per level; `table_size`: hash entries; `feature_dim`: features/entry.
    pub fn new(resolutions: &[usize], table_size: usize, feature_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(999);
        let tables: Vec<Vec<Vec<f64>>> = resolutions
            .iter()
            .map(|_| {
                (0..table_size)
                    .map(|_| {
                        (0..feature_dim)
                            .map(|_| rng.random::<f64>() * 0.001)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Self {
            feature_dim,
            tables,
            resolutions: resolutions.to_vec(),
        }
    }

    fn hash(ix: usize, iy: usize, iz: usize, table_size: usize) -> usize {
        (ix ^ iy.wrapping_mul(2654435761) ^ iz.wrapping_mul(805459861)) % table_size
    }

    /// Encode 3D point in `[0,1]³` → concatenated multi-level features.
    pub fn encode(&self, point: &[f64]) -> Vec<f64> {
        let px = point.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let py = point.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let pz = point.get(2).copied().unwrap_or(0.0).clamp(0.0, 1.0);

        let mut out = Vec::with_capacity(self.feature_dim * self.tables.len());
        let table_size = self.tables.first().map(|t| t.len()).unwrap_or(1);

        for (level_idx, &res_u) in self.resolutions.iter().enumerate() {
            let res = res_u as f64;
            let (fx, fy, fz) = (px * res, py * res, pz * res);
            let (ix, iy, iz) = (
                fx.floor() as usize,
                fy.floor() as usize,
                fz.floor() as usize,
            );
            let (tx, ty, tz) = (fx.fract(), fy.fract(), fz.fract());
            let mut feat = vec![0.0_f64; self.feature_dim];
            for dz in 0..2usize {
                for dy in 0..2usize {
                    for dx in 0..2usize {
                        let h = Self::hash(ix + dx, iy + dy, iz + dz, table_size);
                        let w = (if dx == 0 { 1.0 - tx } else { tx })
                            * (if dy == 0 { 1.0 - ty } else { ty })
                            * (if dz == 0 { 1.0 - tz } else { tz });
                        for (k, &v) in self.tables[level_idx][h].iter().enumerate() {
                            feat[k] += w * v;
                        }
                    }
                }
            }
            out.extend_from_slice(&feat);
        }
        out
    }

    /// Output dimension of the encoding.
    pub fn output_dim(&self) -> usize {
        self.tables.len() * self.feature_dim
    }
}

/// MLP predicting occupancy probability ∈ \[0,1\] at arbitrary 3D query points.
#[derive(Debug, Clone)]
pub struct OccupancyNetwork {
    layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
}

impl OccupancyNetwork {
    /// Create a new occupancy network.
    pub fn new(in_dim: usize, hidden_dims: &[usize], seed: u64) -> Self {
        let mut s = seed;
        let mut layers = Vec::new();
        let mut prev = in_dim;
        for &h in hidden_dims {
            layers.push((rand_weight(h, prev, s), vec![0.0; h]));
            s += 1;
            prev = h;
        }
        layers.push((rand_weight(1, prev, s), vec![0.0]));
        Self { layers }
    }

    /// Forward pass: returns occupancy probability in \[0,1\].
    pub fn forward(&self, point: &[f64]) -> f64 {
        let n = self.layers.len();
        let mut h = point.to_vec();
        for (i, (w, b)) in self.layers.iter().enumerate() {
            if i < n - 1 {
                h = linear_relu(w, b, &h);
            } else {
                h = linear(w, b, &h);
            }
        }
        sigmoid(h.first().copied().unwrap_or(0.0))
    }
}
