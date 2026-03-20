//! Neural Rendering Advanced — 3DGS enhancements, Neural Radiance Caching,
//! Deformable NeRF, and rendering metrics.
//!
//! References:
//! - Kerbl et al. 2023 — 3D Gaussian Splatting for Real-Time Novel View Synthesis
//! - Müller et al. 2021 — Real-time Neural Radiance Caching for Path Tracing
//! - Park et al. 2021 — Nerfies: Deformable Neural Radiance Fields

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// Re-use helpers from parent module
use super::{dot, linear, linear_relu, rand_weight, relu, sigmoid, softplus};

// ============================================================
// §1 Enhanced 3D Gaussian Splatting
// ============================================================

/// Spherical harmonics degree-1 color coefficients (DC + 3 directional).
/// Layout: [r_dc, g_dc, b_dc, r_1, g_1, b_1, r_2, g_2, b_2, r_3, g_3, b_3]
pub type ShCoefficients = Vec<f64>;

/// A 3D Gaussian with full covariance, opacity and SH color representation.
#[derive(Debug, Clone)]
pub struct Gaussian3D {
    /// World-space mean position [x, y, z].
    pub mean: [f64; 3],
    /// Quaternion rotation [w, x, y, z] (unit).
    pub rotation: [f64; 4],
    /// Log-scale in each axis [sx, sy, sz] — exponentiated before use.
    pub log_scale: [f64; 3],
    /// Opacity logit — sigmoid(opacity_logit) gives α ∈ (0,1).
    pub opacity_logit: f64,
    /// Spherical harmonics colour coefficients (degree-0..1, 12 values).
    pub sh: ShCoefficients,
}

impl Gaussian3D {
    /// Construct a Gaussian at `mean` with isotropic scale, full opacity and
    /// white DC SH colour.
    pub fn new(mean: [f64; 3], log_scale: f64) -> Self {
        Self {
            mean,
            rotation: [1.0, 0.0, 0.0, 0.0],
            log_scale: [log_scale; 3],
            opacity_logit: 2.0, // sigmoid(2)≈0.88
            sh: vec![
                0.5, 0.5, 0.5, // DC r,g,b ≈ 0.5 white
                0.0, 0.0, 0.0, // band-1
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ],
        }
    }

    /// Effective scale in world space (exp of log_scale).
    pub fn scale(&self) -> [f64; 3] {
        [
            self.log_scale[0].exp(),
            self.log_scale[1].exp(),
            self.log_scale[2].exp(),
        ]
    }

    /// Effective opacity α.
    pub fn alpha(&self) -> f64 {
        sigmoid_f64(self.opacity_logit)
    }

    /// Evaluate degree-0+1 SH colour toward view direction `dir` (unit vec).
    /// Returns (r, g, b) ∈ \[0,1\].
    pub fn sh_color(&self, dir: [f64; 3]) -> [f64; 3] {
        let sh = &self.sh;
        // SH band-0 (DC)
        let c0 = 0.2820947918; // 1/(2*sqrt(π))
        let r = c0 * sh[0];
        let g = c0 * sh[1];
        let b = c0 * sh[2];
        // SH band-1 coefficients
        let c1 = 0.4886025119; // sqrt(3/(4π))
        let r = r + c1 * (sh[3] * dir[1] + sh[6] * dir[2] + sh[9] * dir[0]);
        let g = g + c1 * (sh[4] * dir[1] + sh[7] * dir[2] + sh[10] * dir[0]);
        let b = b + c1 * (sh[5] * dir[1] + sh[8] * dir[2] + sh[11] * dir[0]);
        // Clamp to [0,1]
        [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
    }

    /// Build the 3×3 covariance matrix Σ = R S S^T R^T.
    pub fn covariance_3d(&self) -> [[f64; 3]; 3] {
        let s = self.scale();
        let q = normalise_quat(self.rotation);
        // Rotation matrix from quaternion
        let r = quat_to_rotation(q);
        // Σ = R * diag(s²) * R^T
        // First compute R * diag(s)
        let mut rs = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rs[i][j] = r[i][j] * s[j];
            }
        }
        // Then Σ = (RS)(RS)^T
        let mut cov = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    cov[i][j] += rs[i][k] * rs[j][k];
                }
            }
        }
        cov
    }
}

fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn normalise_quat(q: [f64; 4]) -> [f64; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm < 1e-12 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
}

fn quat_to_rotation(q: [f64; 4]) -> [[f64; 3]; 3] {
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

/// Projects a 3D Gaussian to 2D screen space via EWA splatting approximation.
///
/// Returns (mean_2d, cov_2d_2x2, alpha_effective) or None if behind camera.
pub struct GaussianSplatter {
    /// Camera focal lengths (fx, fy).
    pub focal: (f64, f64),
    /// Image centre (cx, cy).
    pub principal: (f64, f64),
    /// Near-clip distance.
    pub near: f64,
}

impl GaussianSplatter {
    pub fn new(focal: (f64, f64), principal: (f64, f64)) -> Self {
        Self { focal, principal, near: 0.1 }
    }

    /// Project one Gaussian.  `view_mat` is a row-major 4×4 view matrix
    /// (world → camera).  Returns screen-space (u, v, Σ_screen_2x2, α).
    pub fn project(
        &self,
        g: &Gaussian3D,
        view_mat: &[[f64; 4]; 4],
    ) -> Option<([f64; 2], [[f64; 2]; 2], f64)> {
        // Transform mean to camera space
        let m = g.mean;
        let cam = [
            view_mat[0][0] * m[0] + view_mat[0][1] * m[1] + view_mat[0][2] * m[2] + view_mat[0][3],
            view_mat[1][0] * m[0] + view_mat[1][1] * m[1] + view_mat[1][2] * m[2] + view_mat[1][3],
            view_mat[2][0] * m[0] + view_mat[2][1] * m[1] + view_mat[2][2] * m[2] + view_mat[2][3],
        ];
        if cam[2] < self.near {
            return None;
        }
        let inv_z = 1.0 / cam[2];
        let u = cam[0] * inv_z * self.focal.0 + self.principal.0;
        let v = cam[1] * inv_z * self.focal.1 + self.principal.1;

        // Jacobian of perspective projection (2×3)
        let (fx, fy) = self.focal;
        let j = [
            [fx * inv_z, 0.0, -fx * cam[0] * inv_z * inv_z],
            [0.0, fy * inv_z, -fy * cam[1] * inv_z * inv_z],
        ];

        // World-space covariance
        let cov3 = g.covariance_3d();

        // Extract upper-left 3×3 of view_mat as rotation part W
        let w = [
            [view_mat[0][0], view_mat[0][1], view_mat[0][2]],
            [view_mat[1][0], view_mat[1][1], view_mat[1][2]],
            [view_mat[2][0], view_mat[2][1], view_mat[2][2]],
        ];

        // T = J * W  (2×3)
        let mut t = [[0.0_f64; 3]; 2];
        for i in 0..2 {
            for k in 0..3 {
                for l in 0..3 {
                    t[i][k] += j[i][l] * w[l][k];
                }
            }
        }

        // Σ_screen = T * Σ_world * T^T  (2×2)
        // Intermediate: tmp = T * Σ_world  (2×3)
        let mut tmp = [[0.0_f64; 3]; 2];
        for i in 0..2 {
            for k in 0..3 {
                for l in 0..3 {
                    tmp[i][k] += t[i][l] * cov3[l][k];
                }
            }
        }
        // Σ_screen = tmp * T^T  (2×2)
        let mut cov2 = [[0.0_f64; 2]; 2];
        for i in 0..2 {
            for k in 0..2 {
                for l in 0..3 {
                    cov2[i][k] += tmp[i][l] * t[k][l];
                }
            }
        }
        // Add low-pass filter (EWA)
        cov2[0][0] += 0.3;
        cov2[1][1] += 0.3;

        Some(([u, v], cov2, g.alpha()))
    }

    /// Alpha-composite a list of Gaussians front-to-back for a single pixel.
    /// `gaussians_2d`: list of (mean_2d, cov_2d, alpha, color_rgb).
    /// Returns accumulated (r, g, b, transmittance).
    pub fn alpha_composite(
        px: f64,
        py: f64,
        gaussians_2d: &[([f64; 2], [[f64; 2]; 2], f64, [f64; 3])],
    ) -> [f64; 4] {
        let mut color = [0.0_f64; 3];
        let mut transmittance = 1.0_f64;

        for (mean, cov, alpha_gauss, rgb) in gaussians_2d {
            let dx = px - mean[0];
            let dy = py - mean[1];
            // Evaluate 2D Gaussian at pixel
            let power = gaussian_2d_power(dx, dy, cov);
            if power < -8.0 {
                continue;
            }
            let g_val = power.exp();
            let alpha = (alpha_gauss * g_val).min(0.99);
            if alpha < 1e-5 {
                continue;
            }
            for c in 0..3 {
                color[c] += transmittance * alpha * rgb[c];
            }
            transmittance *= 1.0 - alpha;
            if transmittance < 1e-5 {
                break;
            }
        }
        [color[0], color[1], color[2], transmittance]
    }
}

/// Evaluate –0.5 * [dx,dy] * Σ⁻¹ * [dx,dy]^T for a 2×2 SPD Σ.
fn gaussian_2d_power(dx: f64, dy: f64, cov: &[[f64; 2]; 2]) -> f64 {
    let det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    if det.abs() < 1e-12 {
        return -1e9;
    }
    let inv_det = 1.0 / det;
    let inv = [
        [cov[1][1] * inv_det, -cov[0][1] * inv_det],
        [-cov[1][0] * inv_det, cov[0][0] * inv_det],
    ];
    -0.5 * (dx * (inv[0][0] * dx + inv[0][1] * dy) + dy * (inv[1][0] * dx + inv[1][1] * dy))
}

/// Adaptive densification controller for 3DGS training.
///
/// Clones Gaussians with small scale (under-reconstructed regions) and splits
/// large Gaussians (over-reconstructed) into two smaller ones.
pub struct GaussianDensification {
    /// Scale threshold below which a Gaussian is a candidate for cloning.
    pub clone_scale_threshold: f64,
    /// Scale threshold above which a Gaussian is split.
    pub split_scale_threshold: f64,
    /// Accumulated 2D gradient magnitude per Gaussian (proxy for visibility).
    pub grad_accumulator: Vec<f64>,
    /// Gradient magnitude threshold for densification candidates.
    pub grad_threshold: f64,
}

impl GaussianDensification {
    pub fn new(n: usize) -> Self {
        Self {
            clone_scale_threshold: 0.01,
            split_scale_threshold: 0.1,
            grad_accumulator: vec![0.0; n],
            grad_threshold: 0.0002,
        }
    }

    /// Record per-Gaussian gradient magnitude for current step.
    pub fn accumulate(&mut self, idx: usize, grad_magnitude: f64) {
        if idx < self.grad_accumulator.len() {
            self.grad_accumulator[idx] += grad_magnitude;
        }
    }

    /// Densify gaussians: returns (cloned, split_replacements) index sets.
    pub fn densify(
        &mut self,
        gaussians: &mut Vec<Gaussian3D>,
        n_steps: usize,
    ) {
        if n_steps == 0 {
            return;
        }
        let threshold = self.grad_threshold * n_steps as f64;
        let mut to_clone: Vec<usize> = Vec::new();
        let mut to_split: Vec<usize> = Vec::new();

        for (i, g) in gaussians.iter().enumerate() {
            if i >= self.grad_accumulator.len() {
                break;
            }
            if self.grad_accumulator[i] < threshold {
                continue;
            }
            let s = g.scale();
            let max_s = s[0].max(s[1]).max(s[2]);
            if max_s < self.clone_scale_threshold {
                to_clone.push(i);
            } else if max_s > self.split_scale_threshold {
                to_split.push(i);
            }
        }

        // Clone: duplicate with small random offset
        let mut rng = StdRng::seed_from_u64(42);
        for &idx in &to_clone {
            let mut g = gaussians[idx].clone();
            for v in g.mean.iter_mut() {
                *v += (rng.random::<f64>() - 0.5) * 0.001;
            }
            gaussians.push(g);
        }

        // Split: replace original with two smaller Gaussians
        // Process in reverse to preserve indices
        for &idx in to_split.iter().rev() {
            if idx >= gaussians.len() {
                continue;
            }
            let orig = gaussians[idx].clone();
            let new_log_scale: [f64; 3] = [
                orig.log_scale[0] - std::f64::consts::LN_2 / 2.0,
                orig.log_scale[1] - std::f64::consts::LN_2 / 2.0,
                orig.log_scale[2] - std::f64::consts::LN_2 / 2.0,
            ];
            // Two children offset along principal scale axis
            let axis_idx = orig.log_scale.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let offset = orig.scale()[axis_idx] * 0.5;
            let r = quat_to_rotation(normalise_quat(orig.rotation));
            let axis = [r[0][axis_idx], r[1][axis_idx], r[2][axis_idx]];

            let mut child_a = orig.clone();
            child_a.log_scale = new_log_scale;
            child_a.mean[0] += axis[0] * offset;
            child_a.mean[1] += axis[1] * offset;
            child_a.mean[2] += axis[2] * offset;

            let mut child_b = orig.clone();
            child_b.log_scale = new_log_scale;
            child_b.mean[0] -= axis[0] * offset;
            child_b.mean[1] -= axis[1] * offset;
            child_b.mean[2] -= axis[2] * offset;

            gaussians[idx] = child_a;
            gaussians.push(child_b);
        }

        // Reset accumulator for new size
        self.grad_accumulator = vec![0.0; gaussians.len()];
    }
}

// ============================================================
// §2 Neural Radiance Caching
// ============================================================

/// Hash-grid entry: stores a small feature vector.
#[derive(Debug, Clone)]
pub struct HashEntry {
    pub features: Vec<f64>,
}

/// Neural Radiance Cache — hash-grid feature lookup + tiny MLP decoder.
///
/// Inspired by Müller et al. 2021.  The hash grid maps 3D positions to
/// feature vectors that a small MLP decodes into cached radiance.
pub struct NrcCache {
    /// Number of hash grid levels.
    pub n_levels: usize,
    /// Feature dimensions per level.
    pub feature_dim: usize,
    /// Hash table size (per level).
    pub table_size: usize,
    /// Hash tables: [level][hash_idx] → feature_dim values.
    hash_tables: Vec<Vec<Vec<f64>>>,
    /// Tiny MLP weights: list of (W, b) layers.
    mlp_weights: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    /// MLP hidden dim.
    hidden_dim: usize,
}

impl NrcCache {
    /// Construct with `n_levels` hash grid levels and a 2-layer MLP.
    pub fn new(n_levels: usize, feature_dim: usize, table_size: usize, hidden_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(77);
        // Initialise hash tables with small random features
        let hash_tables: Vec<Vec<Vec<f64>>> = (0..n_levels)
            .map(|_| {
                (0..table_size)
                    .map(|_| {
                        (0..feature_dim)
                            .map(|_| (rng.random::<f64>() - 0.5) * 0.1)
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let input_dim = n_levels * feature_dim;
        let scale0 = (2.0 / input_dim as f64).sqrt();
        let scale1 = (2.0 / hidden_dim as f64).sqrt();

        let w0: Vec<Vec<f64>> = (0..hidden_dim)
            .map(|_| (0..input_dim).map(|_| (rng.random::<f64>() - 0.5) * scale0).collect())
            .collect();
        let b0 = vec![0.0; hidden_dim];
        let w1: Vec<Vec<f64>> = (0..3)
            .map(|_| (0..hidden_dim).map(|_| (rng.random::<f64>() - 0.5) * scale1).collect())
            .collect();
        let b1 = vec![0.0; 3];

        Self {
            n_levels,
            feature_dim,
            table_size,
            hash_tables,
            mlp_weights: vec![(w0, b0), (w1, b1)],
            hidden_dim,
        }
    }

    /// Hash a 3D voxel coordinate at a given level to a table index.
    fn hash(ix: i64, iy: i64, iz: i64, level: usize, table_size: usize) -> usize {
        // π-hashing as in Müller 2022 instant NGP
        const PI1: i64 = 2_654_435_761_i64;
        const PI2: i64 = 805_459_861_i64;
        let h = (ix ^ (iy.wrapping_mul(PI1)) ^ (iz.wrapping_mul(PI2)))
            .wrapping_add(level as i64);
        (h.unsigned_abs() as usize) % table_size
    }

    /// Look up and tri-linearly interpolate feature at position `pos` for all levels.
    pub fn lookup(&self, pos: [f64; 3]) -> Vec<f64> {
        let mut all_features = Vec::with_capacity(self.n_levels * self.feature_dim);

        for level in 0..self.n_levels {
            let scale = (1 << level) as f64;
            let p = [pos[0] * scale, pos[1] * scale, pos[2] * scale];
            let ix = p[0].floor() as i64;
            let iy = p[1].floor() as i64;
            let iz = p[2].floor() as i64;
            let fx = p[0] - ix as f64;
            let fy = p[1] - iy as f64;
            let fz = p[2] - iz as f64;

            // 8 corners
            let corners = [
                (ix, iy, iz),
                (ix + 1, iy, iz),
                (ix, iy + 1, iz),
                (ix + 1, iy + 1, iz),
                (ix, iy, iz + 1),
                (ix + 1, iy, iz + 1),
                (ix, iy + 1, iz + 1),
                (ix + 1, iy + 1, iz + 1),
            ];
            let weights = [
                (1.0 - fx) * (1.0 - fy) * (1.0 - fz),
                fx * (1.0 - fy) * (1.0 - fz),
                (1.0 - fx) * fy * (1.0 - fz),
                fx * fy * (1.0 - fz),
                (1.0 - fx) * (1.0 - fy) * fz,
                fx * (1.0 - fy) * fz,
                (1.0 - fx) * fy * fz,
                fx * fy * fz,
            ];

            let table = &self.hash_tables[level];
            let mut feat = vec![0.0_f64; self.feature_dim];
            for (c, &(cx, cy, cz)) in corners.iter().enumerate() {
                let hidx = Self::hash(cx, cy, cz, level, self.table_size);
                let entry = &table[hidx];
                for f in 0..self.feature_dim {
                    feat[f] += weights[c] * entry[f];
                }
            }
            all_features.extend(feat);
        }
        all_features
    }

    /// Decode feature vector to RGB radiance via tiny MLP.
    pub fn decode(&self, features: &[f64]) -> [f64; 3] {
        let (w0, b0) = &self.mlp_weights[0];
        let (w1, b1) = &self.mlp_weights[1];

        // Layer 0: hidden_dim
        let mut h = vec![0.0_f64; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut s = b0[i];
            for (j, &x) in features.iter().enumerate() {
                if j < w0[i].len() {
                    s += w0[i][j] * x;
                }
            }
            h[i] = relu(s);
        }
        // Layer 1: 3 outputs
        let mut out = [0.0_f64; 3];
        for i in 0..3 {
            let mut s = b1[i];
            for (j, &hj) in h.iter().enumerate() {
                if j < w1[i].len() {
                    s += w1[i][j] * hj;
                }
            }
            out[i] = sigmoid(s);
        }
        out
    }

    /// Forward: lookup + decode.
    pub fn query(&self, pos: [f64; 3]) -> [f64; 3] {
        let features = self.lookup(pos);
        self.decode(&features)
    }
}

/// Simplified ReSTIR reservoir for spatiotemporal importance resampling.
///
/// Reservoir holds a single selected sample and its weight sum for
/// weighted reservoir sampling (Talbot 2005 / Bitterli 2020 ReSTIR GI).
#[derive(Debug, Clone)]
pub struct Reservoir {
    /// Current selected sample (position + radiance).
    pub sample: Option<([f64; 3], [f64; 3])>,
    /// Accumulated weight.
    pub w_sum: f64,
    /// Number of candidates streamed so far.
    pub m: usize,
    /// Unbiased contribution weight W.
    pub capital_w: f64,
}

impl Reservoir {
    pub fn new() -> Self {
        Self { sample: None, w_sum: 0.0, m: 0, capital_w: 0.0 }
    }

    /// Stream one candidate with weight `w`.
    pub fn update(
        &mut self,
        candidate: ([f64; 3], [f64; 3]),
        w: f64,
        rng: &mut StdRng,
    ) {
        self.w_sum += w;
        self.m += 1;
        if rng.random::<f64>() < w / self.w_sum {
            self.sample = Some(candidate);
        }
    }

    /// Compute unbiased weight given target pdf `p_hat` of current sample.
    pub fn compute_weight(&mut self, p_hat: f64) {
        if p_hat < 1e-12 || self.m == 0 {
            self.capital_w = 0.0;
        } else {
            self.capital_w = (1.0 / p_hat) * (self.w_sum / self.m as f64);
        }
    }

    /// Combine another reservoir into this one (spatial reuse).
    pub fn merge(&mut self, other: &Reservoir, p_hat: f64, rng: &mut StdRng) {
        let w = p_hat * other.capital_w * other.m as f64;
        if let Some(s) = other.sample {
            self.update(s, w, rng);
        }
        self.m += other.m;
    }
}

impl Default for Reservoir {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplified ReSTIR global illumination pass.
pub struct ReSTIR {
    pub n_spatial_neighbors: usize,
    pub n_temporal_history: usize,
}

impl ReSTIR {
    pub fn new(n_spatial: usize, n_temporal: usize) -> Self {
        Self {
            n_spatial_neighbors: n_spatial,
            n_temporal_history: n_temporal,
        }
    }

    /// Perform one resampling step: combine current reservoir with neighbors.
    pub fn resample(
        &self,
        current: &mut Reservoir,
        neighbors: &[Reservoir],
        p_hat_fn: impl Fn(&([f64; 3], [f64; 3])) -> f64,
        rng: &mut StdRng,
    ) {
        let limit = self.n_spatial_neighbors.min(neighbors.len());
        for neighbor in neighbors.iter().take(limit) {
            if let Some(s) = &neighbor.sample {
                let p_hat = p_hat_fn(s);
                current.merge(neighbor, p_hat, rng);
            }
        }
        // Recompute weight for current sample
        if let Some(s) = &current.sample.clone() {
            let p_hat = p_hat_fn(s);
            current.compute_weight(p_hat);
        }
    }
}

// ============================================================
// §3 Deformable NeRF
// ============================================================

/// MLP-based deformation field for dynamic scenes.
///
/// Maps (x, y, z, t) → (Δx, Δy, Δz) displacement.
pub struct DeformationField {
    /// Positional encoding frequencies for space.
    pub n_freq_space: usize,
    /// Positional encoding frequencies for time.
    pub n_freq_time: usize,
    /// MLP layer weights (W, b).
    layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
}

impl DeformationField {
    /// Build a deformation field MLP.
    /// Input: positional-encoded (x,t), output: 3D displacement.
    pub fn new(n_freq_space: usize, n_freq_time: usize, hidden_dim: usize, n_layers: usize) -> Self {
        let input_dim = 3 * n_freq_space * 2 + n_freq_time * 2 + 3 + 1; // pos_enc + raw
        let mut rng = StdRng::seed_from_u64(13);
        let mut layers = Vec::new();

        let mut in_d = input_dim;
        for i in 0..n_layers {
            let out_d = if i == n_layers - 1 { 3 } else { hidden_dim };
            let scale = (2.0 / in_d as f64).sqrt();
            let w: Vec<Vec<f64>> = (0..out_d)
                .map(|_| (0..in_d).map(|_| (rng.random::<f64>() - 0.5) * scale).collect())
                .collect();
            let b = vec![0.0; out_d];
            layers.push((w, b));
            in_d = out_d;
        }
        Self { n_freq_space, n_freq_time, layers }
    }

    fn positional_encode(x: f64, n_freq: usize) -> Vec<f64> {
        let mut out = vec![x];
        for k in 0..n_freq {
            let freq = std::f64::consts::PI * 2.0_f64.powi(k as i32);
            out.push((freq * x).sin());
            out.push((freq * x).cos());
        }
        out
    }

    /// Encode 3D position + scalar time into the full input vector.
    fn encode(&self, pos: [f64; 3], t: f64) -> Vec<f64> {
        let mut input = Vec::new();
        for &xi in &pos {
            input.extend(Self::positional_encode(xi, self.n_freq_space));
        }
        input.extend(Self::positional_encode(t, self.n_freq_time));
        input
    }

    /// Forward pass: return displacement Δx for position and time.
    pub fn forward(&self, pos: [f64; 3], t: f64) -> [f64; 3] {
        let mut x = self.encode(pos, t);
        let n = self.layers.len();
        for (i, (w, b)) in self.layers.iter().enumerate() {
            let out_dim = w.len();
            let mut y = vec![0.0_f64; out_dim];
            for j in 0..out_dim {
                let mut s = b[j];
                for (k, &xk) in x.iter().enumerate() {
                    if k < w[j].len() {
                        s += w[j][k] * xk;
                    }
                }
                y[j] = if i < n - 1 { relu(s) } else { s };
            }
            x = y;
        }
        if x.len() >= 3 {
            [x[0], x[1], x[2]]
        } else {
            [0.0, 0.0, 0.0]
        }
    }
}

/// Dynamic NeRF: canonical NeRF combined with a deformation field.
///
/// At time t, query point x is deformed to x' = x + Δx(x, t),
/// then the canonical NeRF is evaluated at x'.
pub struct DynamicNerf {
    pub deformation: DeformationField,
    /// Canonical NeRF MLP layers (W, b).
    nerf_layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    pub n_freq: usize,
}

impl DynamicNerf {
    pub fn new(n_freq_space: usize, n_freq_time: usize, hidden_dim: usize) -> Self {
        let deformation = DeformationField::new(n_freq_space, n_freq_time, hidden_dim, 4);
        // Canonical NeRF: 3D positional encoding → density + RGB
        let input_dim = 3 * n_freq_space * 2 + 3; // pos_enc of deformed x
        let mut rng = StdRng::seed_from_u64(99);
        let mut layers = Vec::new();
        let dims = [input_dim, hidden_dim, hidden_dim, hidden_dim, 4usize]; // last 4 = [r,g,b,σ]
        for i in 0..dims.len() - 1 {
            let in_d = dims[i];
            let out_d = dims[i + 1];
            let scale = (2.0 / in_d as f64).sqrt();
            let w: Vec<Vec<f64>> = (0..out_d)
                .map(|_| (0..in_d).map(|_| (rng.random::<f64>() - 0.5) * scale).collect())
                .collect();
            let b = vec![0.0; out_d];
            layers.push((w, b));
        }
        Self { deformation, nerf_layers: layers, n_freq: n_freq_space }
    }

    fn pos_encode(x: f64, n_freq: usize) -> Vec<f64> {
        let mut out = vec![x];
        for k in 0..n_freq {
            let freq = std::f64::consts::PI * 2.0_f64.powi(k as i32);
            out.push((freq * x).sin());
            out.push((freq * x).cos());
        }
        out
    }

    /// Query the dynamic NeRF: returns (r, g, b, density σ).
    pub fn query(&self, pos: [f64; 3], t: f64) -> [f64; 4] {
        let delta = self.deformation.forward(pos, t);
        let deformed = [pos[0] + delta[0], pos[1] + delta[1], pos[2] + delta[2]];

        // Encode deformed position
        let mut input = Vec::new();
        for &xi in &deformed {
            input.extend(Self::pos_encode(xi, self.n_freq));
        }

        let n = self.nerf_layers.len();
        let mut x = input;
        for (i, (w, b)) in self.nerf_layers.iter().enumerate() {
            let out_dim = w.len();
            let mut y = vec![0.0_f64; out_dim];
            for j in 0..out_dim {
                let mut s = b[j];
                for (k, &xk) in x.iter().enumerate() {
                    if k < w[j].len() {
                        s += w[j][k] * xk;
                    }
                }
                y[j] = if i < n - 1 { relu(s) } else { s };
            }
            x = y;
        }
        if x.len() >= 4 {
            [
                sigmoid(x[0]),
                sigmoid(x[1]),
                sigmoid(x[2]),
                softplus(x[3]),
            ]
        } else {
            [0.0, 0.0, 0.0, 0.0]
        }
    }
}

// ============================================================
// §4 Rendering Metrics
// ============================================================

/// Comprehensive neural rendering evaluation metrics.
#[derive(Debug, Clone)]
pub struct NrMetrics {
    /// Peak signal-to-noise ratio in dB.
    pub psnr: f64,
    /// Structural similarity index (simplified luminance + contrast).
    pub ssim: f64,
    /// Depth-map RMSE.
    pub depth_rmse: f64,
    /// Geometry accuracy (chamfer-like proxy, lower is better).
    pub geometry_accuracy: f64,
    /// FID proxy — mean squared feature difference (lower is better).
    pub fid_proxy: f64,
}

impl NrMetrics {
    /// Compute all metrics given rendered and reference images + depth maps.
    ///
    /// Images are flat `Vec<f64>` of length `w*h*3` (RGB in \[0,1\]).
    /// Depth maps are flat `Vec<f64>` of length `w*h`.
    pub fn compute(
        rendered: &[f64],
        reference: &[f64],
        rendered_depth: Option<&[f64]>,
        reference_depth: Option<&[f64]>,
    ) -> Self {
        let psnr = compute_psnr(rendered, reference);
        let ssim = compute_ssim(rendered, reference);
        let depth_rmse = match (rendered_depth, reference_depth) {
            (Some(rd), Some(ref_d)) => compute_depth_rmse(rd, ref_d),
            _ => f64::NAN,
        };
        let geometry_accuracy = compute_geometry_accuracy(rendered, reference);
        let fid_proxy = compute_fid_proxy(rendered, reference);
        Self { psnr, ssim, depth_rmse, geometry_accuracy, fid_proxy }
    }

    /// Returns true if PSNR ≥ threshold and SSIM ≥ ssim_threshold.
    pub fn passes_quality_gate(&self, psnr_threshold: f64, ssim_threshold: f64) -> bool {
        self.psnr >= psnr_threshold && self.ssim >= ssim_threshold
    }
}

fn compute_psnr(rendered: &[f64], reference: &[f64]) -> f64 {
    if rendered.is_empty() || rendered.len() != reference.len() {
        return 0.0;
    }
    let mse: f64 = rendered
        .iter()
        .zip(reference.iter())
        .map(|(r, g)| (r - g).powi(2))
        .sum::<f64>()
        / rendered.len() as f64;
    if mse < 1e-12 {
        return 100.0;
    }
    10.0 * (1.0_f64 / mse).log10()
}

/// Simplified SSIM: luminance × contrast term over whole image.
fn compute_ssim(rendered: &[f64], reference: &[f64]) -> f64 {
    if rendered.len() != reference.len() || rendered.is_empty() {
        return 0.0;
    }
    let n = rendered.len() as f64;
    let mu_r = rendered.iter().sum::<f64>() / n;
    let mu_g = reference.iter().sum::<f64>() / n;
    let var_r = rendered.iter().map(|&x| (x - mu_r).powi(2)).sum::<f64>() / n;
    let var_g = reference.iter().map(|&x| (x - mu_g).powi(2)).sum::<f64>() / n;
    let cov = rendered
        .iter()
        .zip(reference.iter())
        .map(|(&r, &g)| (r - mu_r) * (g - mu_g))
        .sum::<f64>()
        / n;
    let c1 = 0.01_f64.powi(2);
    let c2 = 0.03_f64.powi(2);
    let lum = (2.0 * mu_r * mu_g + c1) / (mu_r * mu_r + mu_g * mu_g + c1);
    let con = (2.0 * cov + c2) / (var_r + var_g + c2);
    (lum * con).clamp(-1.0, 1.0)
}

fn compute_depth_rmse(rendered_depth: &[f64], reference_depth: &[f64]) -> f64 {
    if rendered_depth.len() != reference_depth.len() || rendered_depth.is_empty() {
        return f64::NAN;
    }
    let mse: f64 = rendered_depth
        .iter()
        .zip(reference_depth.iter())
        .map(|(r, g)| (r - g).powi(2))
        .sum::<f64>()
        / rendered_depth.len() as f64;
    mse.sqrt()
}

/// Geometry accuracy proxy: mean absolute gradient-magnitude difference.
/// Approximates structure preservation in the rendered image.
fn compute_geometry_accuracy(rendered: &[f64], reference: &[f64]) -> f64 {
    if rendered.len() != reference.len() || rendered.is_empty() {
        return f64::NAN;
    }
    // Use simple finite-difference magnitude as geometry proxy
    let n = rendered.len();
    let mut diff_sum = 0.0;
    for i in 1..n {
        let grad_r = (rendered[i] - rendered[i - 1]).abs();
        let grad_g = (reference[i] - reference[i - 1]).abs();
        diff_sum += (grad_r - grad_g).abs();
    }
    diff_sum / (n - 1) as f64
}

/// FID proxy: mean squared difference of channel statistics (mean + std).
fn compute_fid_proxy(rendered: &[f64], reference: &[f64]) -> f64 {
    if rendered.len() != reference.len() || rendered.is_empty() {
        return f64::NAN;
    }
    let n = rendered.len() as f64;
    let mu_r = rendered.iter().sum::<f64>() / n;
    let mu_g = reference.iter().sum::<f64>() / n;
    let std_r = (rendered.iter().map(|&x| (x - mu_r).powi(2)).sum::<f64>() / n).sqrt();
    let std_g = (reference.iter().map(|&x| (x - mu_g).powi(2)).sum::<f64>() / n).sqrt();
    (mu_r - mu_g).powi(2) + (std_r - std_g).powi(2)
}
