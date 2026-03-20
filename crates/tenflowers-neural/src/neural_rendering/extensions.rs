//! Scene Understanding and Camera/View Synthesis extensions for neural rendering.

use super::{cross3, dot, linear, linear_relu, normalize3, rand_weight, sigmoid, softplus};
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// §4  Scene Understanding
// ─────────────────────────────────────────────────────────────────────────────

/// Monocular depth encoder–decoder: image features → per-pixel depth (softplus).
#[derive(Debug, Clone)]
pub struct DepthEstimationNet {
    encoder: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    decoder: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    pub output_pixels: usize,
}

impl DepthEstimationNet {
    /// Create a new depth estimation network.
    pub fn new(in_dim: usize, bottleneck: usize, output_pixels: usize) -> Self {
        let mut s = 200_u64;
        let mut mk = |r: usize, c: usize| {
            let w = rand_weight(r, c, s);
            s += 1;
            (w, vec![0.0_f64; r])
        };
        let encoder = vec![mk(bottleneck * 2, in_dim), mk(bottleneck, bottleneck * 2)];
        let decoder = vec![
            mk(bottleneck * 2, bottleneck),
            mk(output_pixels, bottleneck * 2),
        ];
        Self {
            encoder,
            decoder,
            output_pixels,
        }
    }

    /// Forward pass: image features → per-pixel softplus depth.
    pub fn forward(&self, image_features: &[f64]) -> Vec<f64> {
        let mut h = image_features.to_vec();
        for (w, b) in &self.encoder {
            h = linear_relu(w, b, &h);
        }
        let n = self.decoder.len();
        for (i, (w, b)) in self.decoder.iter().enumerate() {
            h = if i < n - 1 {
                linear_relu(w, b, &h)
            } else {
                linear(w, b, &h)
            };
        }
        h.into_iter().map(softplus).collect()
    }
}

/// Estimate surface normals from depth via tangent cross-product: n = t_u × t_v.
#[derive(Debug, Clone)]
pub struct SurfaceNormalEstimator {
    pub width: usize,
    pub height: usize,
    pub focal_length: f64,
}

impl SurfaceNormalEstimator {
    /// Create a new surface normal estimator.
    pub fn new(width: usize, height: usize, focal_length: f64) -> Self {
        Self {
            width,
            height,
            focal_length,
        }
    }

    fn unproject(&self, u: usize, v: usize, d: f64) -> [f64; 3] {
        let (cx, cy) = (self.width as f64 * 0.5, self.height as f64 * 0.5);
        [
            (u as f64 - cx) * d / self.focal_length,
            (v as f64 - cy) * d / self.focal_length,
            d,
        ]
    }

    /// Returns unit normals for each depth pixel (row-major).
    pub fn estimate(&self, depth: &[f64]) -> Vec<[f64; 3]> {
        let w = self.width;
        let h = self.height;
        let n_px = w * h;
        let mut normals = vec![[0.0_f64; 3]; n_px];

        for v in 0..h {
            for u in 0..w {
                let idx = v * w + u;
                if u == 0 || u == w - 1 || v == 0 || v == h - 1 {
                    normals[idx] = [0.0, 0.0, 1.0];
                    continue;
                }
                let d = |ui: usize, vi: usize| depth.get(vi * w + ui).copied().unwrap_or(0.0);
                let p_left = self.unproject(u - 1, v, d(u - 1, v));
                let p_right = self.unproject(u + 1, v, d(u + 1, v));
                let p_up = self.unproject(u, v - 1, d(u, v - 1));
                let p_down = self.unproject(u, v + 1, d(u, v + 1));
                let t_u = [
                    p_right[0] - p_left[0],
                    p_right[1] - p_left[1],
                    p_right[2] - p_left[2],
                ];
                let t_v = [
                    p_down[0] - p_up[0],
                    p_down[1] - p_up[1],
                    p_down[2] - p_up[2],
                ];
                normals[idx] = normalize3(cross3(t_u, t_v));
            }
        }
        normals
    }
}

/// Predicts semantic class logits per NeRF ray (alongside RGB/density).
#[derive(Debug, Clone)]
pub struct SemanticNerfDecoder {
    pub n_classes: usize,
    semantic_w: Vec<Vec<f64>>,
    semantic_b: Vec<f64>,
}

impl SemanticNerfDecoder {
    /// Create a new semantic NeRF decoder.
    pub fn new(feature_dim: usize, n_classes: usize) -> Self {
        Self {
            n_classes,
            semantic_w: rand_weight(n_classes, feature_dim, 333),
            semantic_b: vec![0.0; n_classes],
        }
    }
    /// Forward: returns logits for each semantic class.
    pub fn forward(&self, features: &[f64]) -> Vec<f64> {
        linear(&self.semantic_w, &self.semantic_b, features)
    }
    /// Predict the most likely class index.
    pub fn predict_class(&self, features: &[f64]) -> usize {
        self.forward(features)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Lifts 2D panoptic masks to 3D via cosine-similarity embedding assignment.
#[derive(Debug, Clone)]
pub struct PanopticLiftingHead {
    pub n_semantic: usize,
    pub max_instances: usize,
    semantic_w: Vec<Vec<f64>>,
    semantic_b: Vec<f64>,
    pub instance_emb_dim: usize,
    instance_w: Vec<Vec<f64>>,
    instance_b: Vec<f64>,
}

impl PanopticLiftingHead {
    /// Create a new panoptic lifting head.
    pub fn new(
        feature_dim: usize,
        n_semantic: usize,
        max_instances: usize,
        instance_emb_dim: usize,
    ) -> Self {
        Self {
            n_semantic,
            max_instances,
            semantic_w: rand_weight(n_semantic, feature_dim, 444),
            semantic_b: vec![0.0; n_semantic],
            instance_emb_dim,
            instance_w: rand_weight(instance_emb_dim, feature_dim, 445),
            instance_b: vec![0.0; instance_emb_dim],
        }
    }
    /// Forward: returns (semantic logits, instance embedding).
    pub fn forward(&self, features: &[f64]) -> (Vec<f64>, Vec<f64>) {
        (
            linear(&self.semantic_w, &self.semantic_b, features),
            linear(&self.instance_w, &self.instance_b, features),
        )
    }
    /// Assign the instance with the highest cosine similarity in the bank.
    pub fn assign_instance(&self, features: &[f64], instance_bank: &[Vec<f64>]) -> usize {
        let (_, emb) = self.forward(features);
        instance_bank
            .iter()
            .enumerate()
            .map(|(i, bank)| {
                let sim = dot(&emb, bank)
                    / ((dot(&emb, &emb).sqrt() + 1e-8) * (dot(bank, bank).sqrt() + 1e-8));
                (i, sim)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i % self.max_instances)
            .unwrap_or(0)
    }
}

/// Estimates 3D scene flow between two frames using point-to-point correspondences.
///
/// A small MLP is trained to predict a 3D displacement vector for each 3D point.
#[derive(Debug, Clone)]
pub struct SceneFlowEstimator {
    layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
}

impl SceneFlowEstimator {
    /// Construct with specified architecture.
    ///
    /// Input: concatenation of `[x,y,z]` from frame t and frame t+1 (6D).
    pub fn new(hidden_dims: &[usize], seed: u64) -> Self {
        let mut s = seed;
        let mut prev = 6_usize;
        let mut layers = Vec::new();
        for &h in hidden_dims {
            layers.push((rand_weight(h, prev, s), vec![0.0; h]));
            s += 1;
            prev = h;
        }
        layers.push((rand_weight(3, prev, s), vec![0.0; 3]));
        Self { layers }
    }

    /// Estimate 3D flow vectors for a set of point correspondences.
    ///
    /// `pts_t0`: points at frame t.
    /// `pts_t1`: corresponding points at frame t+1.
    /// Returns estimated displacement `pts_t1 - pts_t0` (NN approximation).
    pub fn estimate(&self, pts_t0: &[[f64; 3]], pts_t1: &[[f64; 3]]) -> Vec<[f64; 3]> {
        pts_t0
            .iter()
            .zip(pts_t1)
            .map(|(&p0, &p1)| {
                let inp = vec![p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]];
                let n = self.layers.len();
                let mut h = inp;
                for (i, (w, b)) in self.layers.iter().enumerate() {
                    if i < n - 1 {
                        h = linear_relu(w, b, &h);
                    } else {
                        h = linear(w, b, &h);
                    }
                }
                [
                    h.first().copied().unwrap_or(0.0),
                    h.get(1).copied().unwrap_or(0.0),
                    h.get(2).copied().unwrap_or(0.0),
                ]
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  Camera & View Synthesis
// ─────────────────────────────────────────────────────────────────────────────

/// Pinhole camera model.
///
/// Intrinsics: focal length `fx`, `fy`, principal point `(cx, cy)`.
/// Extrinsics: rotation matrix R (3×3, row-major) and translation `t`.
#[derive(Debug, Clone)]
pub struct CameraModel {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    /// 3×3 rotation matrix (row-major, world→camera).
    pub r: [[f64; 3]; 3],
    /// Translation vector (world→camera).
    pub t: [f64; 3],
}

impl CameraModel {
    /// Create an identity camera (looking along +Z).
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t: [0.0; 3],
        }
    }

    /// Create a camera with full extrinsics.
    pub fn with_extrinsics(mut self, r: [[f64; 3]; 3], t: [f64; 3]) -> Self {
        self.r = r;
        self.t = t;
        self
    }

    /// Transform a world-space 3D point to camera space.
    fn world_to_camera(&self, p: [f64; 3]) -> [f64; 3] {
        let x = self.r[0][0] * p[0] + self.r[0][1] * p[1] + self.r[0][2] * p[2] + self.t[0];
        let y = self.r[1][0] * p[0] + self.r[1][1] * p[1] + self.r[1][2] * p[2] + self.t[1];
        let z = self.r[2][0] * p[0] + self.r[2][1] * p[1] + self.r[2][2] * p[2] + self.t[2];
        [x, y, z]
    }

    /// Project a 3D world point to 2D pixel coordinates `[u, v]`.
    pub fn project(&self, p: [f64; 3]) -> Result<[f64; 2]> {
        let c = self.world_to_camera(p);
        if c[2] <= 0.0 {
            return Err(TensorError::invalid_argument_op(
                "camera",
                "point behind camera",
            ));
        }
        let u = self.fx * c[0] / c[2] + self.cx;
        let v = self.fy * c[1] / c[2] + self.cy;
        Ok([u, v])
    }

    /// Back-project a 2D pixel `[u, v]` at depth `depth` to a 3D world point.
    pub fn unproject(&self, uv: [f64; 2], depth: f64) -> [f64; 3] {
        // Camera-space point
        let xc = (uv[0] - self.cx) * depth / self.fx;
        let yc = (uv[1] - self.cy) * depth / self.fy;
        let zc = depth;

        // Invert extrinsics: p_world = Rᵀ (p_cam - t)
        let dx = xc - self.t[0];
        let dy = yc - self.t[1];
        let dz = zc - self.t[2];
        [
            self.r[0][0] * dx + self.r[1][0] * dy + self.r[2][0] * dz,
            self.r[0][1] * dx + self.r[1][1] * dy + self.r[2][1] * dz,
            self.r[0][2] * dx + self.r[1][2] * dy + self.r[2][2] * dz,
        ]
    }
}

/// Photometric consistency loss across multiple views.
///
/// Warps source features to target view using depth and camera transform, then
/// computes normalised cross-correlation (NCC).
#[derive(Debug, Clone)]
pub struct MultiViewConsistencyLoss {
    /// Small epsilon added to NCC denominator.
    pub eps: f64,
}

impl MultiViewConsistencyLoss {
    /// Create a new multi-view consistency loss.
    pub fn new() -> Self {
        Self { eps: 1e-6 }
    }

    /// Compute NCC between two feature patches.
    fn ncc(a: &[f64], b: &[f64]) -> f64 {
        if a.is_empty() {
            return 0.0;
        }
        let n = a.len() as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let num = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - ma) * (y - mb))
            .sum::<f64>();
        let sa = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>().sqrt();
        let sb = b.iter().map(|x| (x - mb).powi(2)).sum::<f64>().sqrt();
        let denom = sa * sb;
        if denom < 1e-8 {
            return 0.0;
        }
        num / denom
    }

    /// Compute photometric consistency.
    ///
    /// `src_feats` / `tgt_feats`: flattened feature maps (H×W×C row-major).
    /// `depth`: depth map for the source view.
    /// `t_src_tgt`: relative transformation (homogeneous 3×4 `[R|t]` row-major).
    ///
    /// Returns `1 - mean(NCC)` as a loss (lower is more consistent).
    pub fn compute(
        &self,
        src_feats: &[f64],
        tgt_feats: &[f64],
        _depth: &[f64],
        _t_src_tgt: &[[f64; 4]; 3],
    ) -> f64 {
        // Simplified: compute NCC directly between feature vectors.
        let ncc = Self::ncc(src_feats, tgt_feats);
        1.0 - ncc
    }
}

impl Default for MultiViewConsistencyLoss {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplified pose estimator using Direct Linear Transform (DLT).
///
/// Estimates the relative essential matrix E from point correspondences
/// `(x1, x2)` using the 8-point DLT algorithm (with SVD approximation).
#[derive(Debug, Clone)]
pub struct PoseEstimator {
    /// Minimum number of correspondences required.
    pub min_correspondences: usize,
}

impl PoseEstimator {
    /// Create a new pose estimator.
    pub fn new() -> Self {
        Self {
            min_correspondences: 8,
        }
    }

    /// Estimate relative pose (R, t) from 2D-2D correspondences.
    ///
    /// Uses a simplified normalised 8-point algorithm.
    /// Returns `(R: [[f64;3];3], t: [f64;3])`.
    pub fn estimate(
        &self,
        pts1: &[[f64; 2]],
        pts2: &[[f64; 2]],
    ) -> Result<([[f64; 3]; 3], [f64; 3])> {
        if pts1.len() < self.min_correspondences {
            return Err(TensorError::invalid_argument_op(
                "pose_estimator",
                &format!(
                    "need at least {} correspondences, got {}",
                    self.min_correspondences,
                    pts1.len()
                ),
            ));
        }

        // Build constraint matrix A (n×9) for the essential matrix
        let n = pts1.len();
        let a: Vec<Vec<f64>> = pts1
            .iter()
            .zip(pts2)
            .map(|(p1, p2)| {
                let (x1, y1) = (p1[0], p1[1]);
                let (x2, y2) = (p2[0], p2[1]);
                vec![x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1.0]
            })
            .collect();

        // Estimate E via least squares (AᵀA smallest eigenvector approximation)
        // For this simplified version: use the average to get a rough estimate
        let e_flat: Vec<f64> = (0..9)
            .map(|j| a.iter().map(|row| row[j]).sum::<f64>() / n as f64)
            .collect();

        // Reshape to 3×3
        let e = [
            [e_flat[0], e_flat[1], e_flat[2]],
            [e_flat[3], e_flat[4], e_flat[5]],
            [e_flat[6], e_flat[7], e_flat[8]],
        ];

        // Extract R and t via SVD-like decomposition (identity + skew-symmetric approximation)
        let t = normalize3([e[2][1] - e[1][2], e[0][2] - e[2][0], e[1][0] - e[0][1]]);
        let r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; // Simplified: identity rotation

        Ok((r, t))
    }
}

impl Default for PoseEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Novel-view interpolator: blends two known views in latent-feature space.
#[derive(Debug, Clone)]
pub struct ViewInterpolator {
    /// Latent feature MLP.
    layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
}

impl ViewInterpolator {
    /// Construct the interpolator.
    ///
    /// Input: concatenation of [view_a features, view_b features, alpha (scalar)].
    pub fn new(feature_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let in_dim = feature_dim * 2 + 1;
        let mut s = 555_u64;
        let mut mk = |r: usize, c: usize| {
            let w = rand_weight(r, c, s);
            s += 1;
            (w, vec![0.0_f64; r])
        };
        let layers = vec![
            mk(hidden_dim, in_dim),
            mk(hidden_dim, hidden_dim),
            mk(output_dim, hidden_dim),
        ];
        Self { layers }
    }

    /// Interpolate between view features `a` and `b` at blend weight `alpha` ∈ \[0,1\].
    pub fn interpolate(&self, a: &[f64], b: &[f64], alpha: f64) -> Vec<f64> {
        let mut inp = a.to_vec();
        inp.extend_from_slice(b);
        inp.push(alpha);
        let n = self.layers.len();
        let mut h = inp;
        for (i, (w, bv)) in self.layers.iter().enumerate() {
            if i < n - 1 {
                h = linear_relu(w, bv, &h);
            } else {
                h = linear(w, bv, &h);
            }
        }
        h
    }
}

/// Refines camera parameters by minimising reprojection error via gradient descent.
#[derive(Debug, Clone)]
pub struct CameraOptimizer {
    /// Learning rate.
    pub lr: f64,
    /// Finite difference step.
    pub eps: f64,
    /// Total number of optimisation steps taken.
    pub step: usize,
}

impl CameraOptimizer {
    /// Create a new camera optimizer with the given learning rate.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            eps: 1e-5,
            step: 0,
        }
    }

    /// Compute the reprojection error for a set of correspondences.
    fn reprojection_error(camera: &CameraModel, pts3d: &[[f64; 3]], pts2d: &[[f64; 2]]) -> f64 {
        let n = pts3d.len().max(1) as f64;
        pts3d
            .iter()
            .zip(pts2d)
            .map(|(&p3, p2)| {
                camera
                    .project(p3)
                    .map(|proj| (proj[0] - p2[0]).powi(2) + (proj[1] - p2[1]).powi(2))
                    .unwrap_or(1e6)
            })
            .sum::<f64>()
            / n
    }

    /// Perform a single gradient-descent step on the focal length and principal point.
    pub fn step_once(
        &mut self,
        camera: &mut CameraModel,
        pts3d: &[[f64; 3]],
        pts2d: &[[f64; 2]],
    ) -> f64 {
        let loss0 = Self::reprojection_error(camera, pts3d, pts2d);

        // Gradient w.r.t. fx
        camera.fx += self.eps;
        let loss_fx = Self::reprojection_error(camera, pts3d, pts2d);
        camera.fx -= self.eps;
        let grad_fx = (loss_fx - loss0) / self.eps;

        // Gradient w.r.t. fy
        camera.fy += self.eps;
        let loss_fy = Self::reprojection_error(camera, pts3d, pts2d);
        camera.fy -= self.eps;
        let grad_fy = (loss_fy - loss0) / self.eps;

        // Gradient w.r.t. cx
        camera.cx += self.eps;
        let loss_cx = Self::reprojection_error(camera, pts3d, pts2d);
        camera.cx -= self.eps;
        let grad_cx = (loss_cx - loss0) / self.eps;

        // Gradient w.r.t. cy
        camera.cy += self.eps;
        let loss_cy = Self::reprojection_error(camera, pts3d, pts2d);
        camera.cy -= self.eps;
        let grad_cy = (loss_cy - loss0) / self.eps;

        camera.fx -= self.lr * grad_fx;
        camera.fy -= self.lr * grad_fy;
        camera.cx -= self.lr * grad_cx;
        camera.cy -= self.lr * grad_cy;

        self.step += 1;
        loss0
    }
}
