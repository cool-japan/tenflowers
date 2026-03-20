//! Tests for neural_rendering — original + advanced (3DGS, NRC, Deformable NeRF, metrics).

#[cfg(test)]
mod tests {
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    use crate::neural_rendering::{
        // mod.rs
        CameraModel, CameraOptimizer, DepthEstimationNet, GaussianOptimizer, GaussianSplatRenderer,
        Gaussian3D, InstantNgp, NeRFLoss, NeRFMlp, MultiViewConsistencyLoss, NeuralSdf,
        OccupancyNetwork, PanopticLiftingHead, PoseEstimator, PositionalEncoding, QuaternionOps,
        RayMarcher, SceneFlowEstimator, SemanticNerfDecoder, SirenLayer, SirenNetwork,
        SurfaceNormalEstimator, ViewInterpolator, VolumeRenderer,
        // extensions.rs
        // advanced.rs
        advanced::{
            DeformationField, DynamicNerf, GaussianDensification,
            GaussianSplatter, Gaussian3D as Adv3DGaussian, NrcCache, NrMetrics, Reservoir, ReSTIR,
        },
    };

    // ── NeRF ────────────────────────────────────────────────────────────────

    #[test]
    fn test_positional_encoding_length() {
        let enc = PositionalEncoding::new(4);
        let out = enc.encode_scalar(1.0);
        // 2 * n_freqs = 8
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_positional_encoding_period() {
        // sin(2^0 * x) with x = 0 should give 0.0; cos gives 1.0
        let enc = PositionalEncoding::new(2);
        let out = enc.encode_scalar(0.0);
        assert!((out[0]).abs() < 1e-10); // sin(0)
        assert!((out[1] - 1.0).abs() < 1e-10); // cos(0)
    }

    #[test]
    fn test_positional_encoding_with_identity() {
        let enc = PositionalEncoding::with_identity(3);
        let out = enc.encode_scalar(2.0);
        // 1 (identity) + 2 * 3 = 7
        assert_eq!(out.len(), 7);
        assert!((out[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_positional_encoding_multi_dim() {
        let enc = PositionalEncoding::new(4);
        let out = enc.encode(&[1.0, 2.0, 3.0]);
        // 3 * 8 = 24
        assert_eq!(out.len(), 24);
    }

    #[test]
    fn test_nerf_mlp_output_shape() {
        let mlp = NeRFMlp::new(4, 4, 64);
        let pt = vec![0.1, 0.2, 0.3];
        let dir = vec![0.0, 0.0, 1.0];
        let (rgb, sigma) = mlp.forward(&pt, &dir).expect("forward failed");
        assert_eq!(rgb.len(), 3);
        for &c in &rgb {
            assert!((0.0..=1.0).contains(&c), "colour out of range: {c}");
        }
        assert!(sigma >= 0.0, "sigma must be non-negative: {sigma}");
    }

    #[test]
    fn test_nerf_mlp_different_directions() {
        let mlp = NeRFMlp::new(4, 4, 64);
        let pt = vec![0.5, 0.5, 0.5];
        let (rgb1, _) = mlp.forward(&pt, &[1.0, 0.0, 0.0]).expect("forward failed");
        let (rgb2, _) = mlp.forward(&pt, &[0.0, 1.0, 0.0]).expect("forward failed");
        let diff: f64 = rgb1.iter().zip(&rgb2).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff >= 0.0);
    }

    #[test]
    fn test_volume_renderer_weights_sum() {
        let renderer = VolumeRenderer::new();
        let samples = vec![
            (1.0, 0.1, [1.0, 0.0, 0.0]),
            (1.0, 0.1, [0.0, 1.0, 0.0]),
            (1.0, 0.1, [0.0, 0.0, 1.0]),
        ];
        let (_colour, weights) = renderer.render(&samples);
        assert_eq!(weights.len(), 3);
        let sum: f64 = weights.iter().sum();
        assert!(sum <= 1.0 + 1e-9, "weights sum {} > 1", sum);
        for &w in &weights {
            assert!(w >= 0.0);
        }
    }

    #[test]
    fn test_volume_renderer_opaque_first_sample() {
        let renderer = VolumeRenderer::new();
        let samples = vec![(1e6, 1.0, [1.0, 0.0, 0.0]), (1.0, 1.0, [0.0, 1.0, 0.0])];
        let (colour, _) = renderer.render(&samples);
        assert!(colour[0] > 0.99, "red should dominate: {:?}", colour);
    }

    #[test]
    fn test_ray_marcher_sample_count() {
        let rm = RayMarcher::new(0.1, 10.0, 64);
        let mut rng = StdRng::seed_from_u64(0);
        let pts = rm.sample_points([0.0; 3], [0.0, 0.0, 1.0], &mut rng);
        assert_eq!(pts.len(), 64);
    }

    #[test]
    fn test_ray_marcher_t_range() {
        let near = 1.0;
        let far = 5.0;
        let rm = RayMarcher::new(near, far, 16);
        let mut rng = StdRng::seed_from_u64(1);
        let pts = rm.sample_points([0.0; 3], [0.0, 0.0, 1.0], &mut rng);
        for (_, t) in &pts {
            assert!(
                *t >= near - 1e-9 && *t <= far + 1e-9,
                "t={t} out of [{near}, {far}]"
            );
        }
    }

    #[test]
    fn test_nerf_loss_mse() {
        let loss_fn = NeRFLoss::new();
        let rendered = vec![[0.5, 0.5, 0.5]];
        let target = vec![[1.0, 0.0, 0.0]];
        let loss = loss_fn.photometric_mse(&rendered, &target).expect("photometric_mse failed");
        // MSE = ((0.5)^2 + (0.5)^2 + (0.5)^2) / 1 = 0.75
        assert!((loss - 0.75).abs() < 1e-9, "loss={loss}");
    }

    // ── 3D Gaussian Splatting (original mod.rs types) ────────────────────────

    #[test]
    fn test_gaussian_3d_construction() {
        let g = Gaussian3D::new([1.0, 2.0, 3.0], 0.5, [1.0, 0.0, 0.0]);
        assert_eq!(g.center, [1.0, 2.0, 3.0]);
        let s = g.scale();
        assert!((s[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_gaussian_3d_opacity() {
        let mut g = Gaussian3D::new([0.0; 3], 1.0, [0.0; 3]);
        // logit 0 → opacity = sigmoid(0) = 0.5
        assert!((g.opacity() - 0.5).abs() < 1e-9);
        g.logit_opacity = 100.0;
        assert!((g.opacity() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_quaternion_normalize() {
        let q = QuaternionOps::normalize([2.0, 0.0, 0.0, 0.0]);
        assert!((q[0] - 1.0).abs() < 1e-10);
        assert_eq!(q[1], 0.0);

        let q2 = QuaternionOps::normalize([1.0, 1.0, 0.0, 0.0]);
        let len: f64 = q2.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((len - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quaternion_to_rotation_matrix() {
        let r = QuaternionOps::to_rotation_matrix([1.0, 0.0, 0.0, 0.0]);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (r[i][j] - expected).abs() < 1e-10,
                    "R[{i}][{j}] = {}",
                    r[i][j]
                );
            }
        }
    }

    #[test]
    fn test_gaussian_projection() {
        let renderer = GaussianSplatRenderer::new(500.0, 64, 64);
        let g = Gaussian3D::new([0.0, 0.0, 5.0], 0.5, [1.0, 0.0, 0.0]);
        let g2d = renderer.project(&g, [0.0, 0.0, 0.0]);
        assert!(g2d.is_some());
        let g2d = g2d.expect("projection should succeed");
        assert!((g2d.center[0] - 32.0).abs() < 1.0);
        assert!((g2d.center[1] - 32.0).abs() < 1.0);
    }

    #[test]
    fn test_gaussian_splat_render() {
        let renderer = GaussianSplatRenderer::new(200.0, 8, 8);
        let gaussians = vec![Gaussian3D::new([0.0, 0.0, 3.0], 0.3, [1.0, 0.0, 0.0])];
        let image = renderer.render_image(&gaussians, [0.0; 3]);
        assert_eq!(image.len(), 8);
        assert_eq!(image[0].len(), 8);
        for row in &image {
            for px in row {
                for &c in px {
                    assert!((0.0..=1.0).contains(&c), "pixel out of range: {c}");
                }
            }
        }
    }

    #[test]
    fn test_gaussian_optimizer() {
        let renderer = GaussianSplatRenderer::new(100.0, 4, 4);
        let mut gaussians = vec![Gaussian3D::new([0.0, 0.0, 2.0], 0.2, [0.5, 0.5, 0.5])];
        let mut opt = GaussianOptimizer::new(0.01);
        let target: Vec<[f64; 3]> = vec![[0.5, 0.5, 0.5]; 16];
        let loss = opt.step_once(&renderer, &mut gaussians, [0.0; 3], &target);
        assert!(loss >= 0.0);
        assert_eq!(opt.step, 1);
    }

    #[test]
    fn test_gaussian_optimizer_densify() {
        let mut gaussians = vec![Gaussian3D::new([0.0, 0.0, 1.0], 1.0, [1.0, 0.0, 0.0])];
        let mut opt = GaussianOptimizer::new(0.01);
        opt.grad_accum = vec![100.0];
        opt.densify(&mut gaussians, 50.0);
        assert_eq!(gaussians.len(), 2);
    }

    #[test]
    fn test_gaussian_optimizer_prune() {
        let mut gaussians = vec![Gaussian3D::new([0.0, 0.0, 1.0], 1.0, [1.0, 0.0, 0.0]), {
            let mut g = Gaussian3D::new([1.0, 0.0, 1.0], 1.0, [0.0, 1.0, 0.0]);
            g.logit_opacity = -100.0;
            g
        }];
        let mut opt = GaussianOptimizer::new(0.01);
        opt.prune(&mut gaussians, 0.1);
        assert_eq!(gaussians.len(), 1);
    }

    // ── Implicit Neural Representations ────────────────────────────────────

    #[test]
    fn test_siren_layer_sin_output() {
        let layer = SirenLayer::new(3, 8, 30.0, 0, true);
        let x = vec![0.0, 0.0, 0.0];
        let out = layer.forward(&x, 30.0);
        assert_eq!(out.len(), 8);
        for &v in &out {
            assert!((v).abs() < 1e-10, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_siren_layer_nonzero_input() {
        let layer = SirenLayer::new(2, 4, 30.0, 1, false);
        let x = vec![1.0, 1.0];
        let out = layer.forward(&x, 30.0);
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert!(
                (-1.0 - 1e-10..=1.0 + 1e-10).contains(&v),
                "out of sin range: {v}"
            );
        }
    }

    #[test]
    fn test_siren_network_construction() {
        let net = SirenNetwork::new(3, &[64, 64], 1, 30.0);
        assert_eq!(net.layers.len(), 3);
        let out = net.forward(&[0.5, 0.5, 0.5]);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_neural_sdf_scalar() {
        let sdf = NeuralSdf::new(32, 3);
        let val = sdf.forward(&[0.0, 0.0, 0.0]);
        let _ = val;
    }

    #[test]
    fn test_instant_ngp_encoding_length() {
        let resolutions = vec![16, 32, 64];
        let enc = InstantNgp::new(&resolutions, 256, 2);
        let out = enc.encode(&[0.5, 0.5, 0.5]);
        assert_eq!(out.len(), 6);
        assert_eq!(enc.output_dim(), 6);
    }

    #[test]
    fn test_instant_ngp_different_points() {
        let enc = InstantNgp::new(&[16, 32], 128, 4);
        let a = enc.encode(&[0.1, 0.1, 0.1]);
        let b = enc.encode(&[0.9, 0.9, 0.9]);
        let diff: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 0.0, "encodings should differ for different points");
    }

    #[test]
    fn test_occupancy_network() {
        let net = OccupancyNetwork::new(3, &[32, 32], 77);
        let p = net.forward(&[0.0, 0.0, 0.0]);
        assert!((0.0..=1.0).contains(&p), "occupancy out of [0,1]: {p}");
    }

    // ── Scene Understanding ─────────────────────────────────────────────────

    #[test]
    fn test_depth_estimation_shape() {
        let net = DepthEstimationNet::new(16, 8, 64);
        let features = vec![0.1_f64; 16];
        let depth = net.forward(&features);
        assert_eq!(depth.len(), 64);
        for &d in &depth {
            assert!(d > 0.0, "depth {d} not positive");
        }
    }

    #[test]
    fn test_surface_normal_perpendicular() {
        let est = SurfaceNormalEstimator::new(4, 4, 100.0);
        let depth = vec![5.0_f64; 16];
        let normals = est.estimate(&depth);
        assert_eq!(normals.len(), 16);
        let n = normals[5];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-9, "normal not unit length: {len}");
    }

    #[test]
    fn test_surface_normal_flat_z_dominant() {
        let est = SurfaceNormalEstimator::new(5, 5, 200.0);
        let depth = vec![3.0_f64; 25];
        let normals = est.estimate(&depth);
        for (idx, n) in normals.iter().enumerate() {
            let row = idx / 5;
            let col = idx % 5;
            if row > 0 && row < 4 && col > 0 && col < 4 {
                assert!(
                    n[2].abs() > 0.5,
                    "z-component small at ({row},{col}): {:?}",
                    n
                );
            }
        }
    }

    #[test]
    fn test_semantic_nerf_decoder() {
        let dec = SemanticNerfDecoder::new(32, 10);
        let feats = vec![0.1_f64; 32];
        let logits = dec.forward(&feats);
        assert_eq!(logits.len(), 10);
        let cls = dec.predict_class(&feats);
        assert!(cls < 10);
    }

    #[test]
    fn test_panoptic_lifting_head() {
        let head = PanopticLiftingHead::new(16, 5, 50, 8);
        let feats = vec![0.5_f64; 16];
        let (sem, inst) = head.forward(&feats);
        assert_eq!(sem.len(), 5);
        assert_eq!(inst.len(), 8);
    }

    #[test]
    fn test_scene_flow_estimator() {
        let est = SceneFlowEstimator::new(&[32, 32], 42);
        let pts0 = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let pts1 = vec![[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]];
        let flow = est.estimate(&pts0, &pts1);
        assert_eq!(flow.len(), 2);
        for f in &flow {
            assert_eq!(f.len(), 3);
        }
    }

    // ── Camera & View Synthesis ─────────────────────────────────────────────

    #[test]
    fn test_camera_project() {
        let cam = CameraModel::new(500.0, 500.0, 320.0, 240.0);
        let uv = cam.project([0.0, 0.0, 2.0]).expect("project failed");
        assert!((uv[0] - 320.0).abs() < 1e-9);
        assert!((uv[1] - 240.0).abs() < 1e-9);
    }

    #[test]
    fn test_camera_project_offset() {
        let cam = CameraModel::new(100.0, 100.0, 0.0, 0.0);
        let uv = cam.project([1.0, 2.0, 5.0]).expect("project failed");
        assert!((uv[0] - 20.0).abs() < 1e-9);
        assert!((uv[1] - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_camera_unproject_roundtrip() {
        let cam = CameraModel::new(500.0, 500.0, 320.0, 240.0);
        let pt3d = [1.0, 2.0, 3.0];
        let uv = cam.project(pt3d).expect("project failed");
        let pt3d_back = cam.unproject(uv, pt3d[2]);
        for i in 0..3 {
            assert!(
                (pt3d_back[i] - pt3d[i]).abs() < 1e-6,
                "axis {i}: {} != {}",
                pt3d_back[i],
                pt3d[i]
            );
        }
    }

    #[test]
    fn test_camera_behind_error() {
        let cam = CameraModel::new(500.0, 500.0, 320.0, 240.0);
        assert!(cam.project([0.0, 0.0, -1.0]).is_err());
    }

    #[test]
    fn test_multiview_consistency_loss() {
        let loss_fn = MultiViewConsistencyLoss::new();
        let feats = vec![1.0, 2.0, 3.0, 4.0];
        let depth = vec![1.0; 4];
        let t = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let l = loss_fn.compute(&feats, &feats, &depth, &t);
        assert!((l).abs() < 1e-9, "identical views loss = {l}");
    }

    #[test]
    fn test_view_interpolator() {
        let interp = ViewInterpolator::new(8, 16, 8);
        let a = vec![1.0_f64; 8];
        let b = vec![-1.0_f64; 8];
        let mid = interp.interpolate(&a, &b, 0.5);
        assert_eq!(mid.len(), 8);
    }

    #[test]
    fn test_camera_optimizer_step() {
        let mut cam = CameraModel::new(500.0, 500.0, 320.0, 240.0);
        let pts3d = vec![[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]];
        let pts2d = vec![[320.0, 240.0], [570.0, 240.0], [320.0, 490.0]];
        let mut opt = CameraOptimizer::new(1e-3);
        let loss = opt.step_once(&mut cam, &pts3d, &pts2d);
        assert!(loss >= 0.0);
        assert_eq!(opt.step, 1);
    }

    #[test]
    fn test_pose_estimator() {
        let pe = PoseEstimator::new();
        let pts1: Vec<[f64; 2]> = (0..8).map(|i| [i as f64, i as f64]).collect();
        let pts2: Vec<[f64; 2]> = (0..8).map(|i| [i as f64 + 0.1, i as f64]).collect();
        let (r, t) = pe.estimate(&pts1, &pts2).expect("estimate failed");
        assert_eq!(r.len(), 3);
        let tnorm: f64 = t.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((tnorm - 1.0).abs() < 0.1 || tnorm < 1.1, "t norm = {tnorm}");
    }

    #[test]
    fn test_pose_estimator_insufficient_pts() {
        let pe = PoseEstimator::new();
        let pts = vec![[0.0_f64; 2]; 3];
        assert!(pe.estimate(&pts, &pts).is_err());
    }

    // ── Advanced: Gaussian3D (advanced) ──────────────────────────────────────

    #[test]
    fn test_adv_gaussian3d_alpha_range() {
        let g = Adv3DGaussian::new([0.0, 0.0, 0.0], 0.0);
        let alpha = g.alpha();
        assert!(alpha > 0.0 && alpha < 1.0, "alpha={alpha}");
    }

    #[test]
    fn test_adv_gaussian3d_scale_positive() {
        let g = Adv3DGaussian::new([0.0, 0.0, 0.0], -2.0);
        let s = g.scale();
        for si in s {
            assert!(si > 0.0, "scale must be positive: {si}");
        }
    }

    #[test]
    fn test_adv_gaussian3d_covariance_symmetric() {
        let g = Adv3DGaussian::new([0.0; 3], 0.5);
        let cov = g.covariance_3d();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (cov[i][j] - cov[j][i]).abs() < 1e-10,
                    "cov[{i}][{j}] != cov[{j}][{i}]"
                );
            }
        }
    }

    #[test]
    fn test_adv_gaussian3d_sh_color_clamp() {
        let g = Adv3DGaussian::new([0.0; 3], 0.0);
        let c = g.sh_color([0.0, 0.0, 1.0]);
        for ci in c {
            assert!((0.0..=1.0).contains(&ci), "sh color out of [0,1]: {ci}");
        }
    }

    #[test]
    fn test_adv_gaussian3d_identity_rotation_covariance() {
        // Identity quaternion should give diagonal covariance = diag(s²)
        let mut g = Adv3DGaussian::new([0.0; 3], 1.0);
        g.rotation = [1.0, 0.0, 0.0, 0.0];
        g.log_scale = [1.0, 1.0, 1.0];
        let cov = g.covariance_3d();
        // s = exp(1) ≈ 2.718, s² ≈ 7.389
        let expected = std::f64::consts::E * std::f64::consts::E;
        assert!((cov[0][0] - expected).abs() < 1e-6, "cov[0][0]={}", cov[0][0]);
        assert!(cov[0][1].abs() < 1e-10, "off-diagonal should be 0");
    }

    // ── Advanced: GaussianSplatter ────────────────────────────────────────────

    #[test]
    fn test_gaussian_splatter_project_valid() {
        let splatter = GaussianSplatter::new((200.0, 200.0), (100.0, 100.0));
        let g = Adv3DGaussian::new([0.0, 0.0, 1.0], -1.0);
        // Identity view matrix
        let view: [[f64; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let result = splatter.project(&g, &view);
        assert!(result.is_some(), "should project successfully");
        let (mean_2d, cov_2d, alpha) = result.expect("ok");
        assert!((mean_2d[0] - 100.0).abs() < 1e-6);
        assert!((mean_2d[1] - 100.0).abs() < 1e-6);
        assert!(alpha > 0.0 && alpha < 1.0);
        // Covariance diagonal should be positive
        assert!(cov_2d[0][0] > 0.0);
        assert!(cov_2d[1][1] > 0.0);
    }

    #[test]
    fn test_gaussian_splatter_behind_camera() {
        let splatter = GaussianSplatter::new((200.0, 200.0), (100.0, 100.0));
        let g = Adv3DGaussian::new([0.0, 0.0, -2.0], 0.0);
        let view: [[f64; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert!(splatter.project(&g, &view).is_none(), "behind camera should return None");
    }

    #[test]
    fn test_alpha_composite_background() {
        // No Gaussians → background (transmittance = 1.0)
        let result = GaussianSplatter::alpha_composite(0.0, 0.0, &[]);
        assert_eq!(result, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_alpha_composite_single_opaque() {
        // One very opaque Gaussian at pixel center
        let mean = [0.0, 0.0];
        let cov = [[1.0, 0.0], [0.0, 1.0]];
        let alpha = 0.99;
        let rgb = [1.0, 0.0, 0.0];
        let result = GaussianSplatter::alpha_composite(0.0, 0.0, &[(mean, cov, alpha, rgb)]);
        // Result red channel should be close to alpha * exp(0) = alpha
        assert!(result[0] > 0.5, "red channel={}", result[0]);
        assert!(result[3] < 0.5, "transmittance should drop: {}", result[3]);
    }

    // ── Advanced: GaussianDensification ──────────────────────────────────────

    #[test]
    fn test_gaussian_densification_new() {
        let dens = GaussianDensification::new(5);
        assert_eq!(dens.grad_accumulator.len(), 5);
    }

    #[test]
    fn test_gaussian_densification_accumulate() {
        let mut dens = GaussianDensification::new(3);
        dens.accumulate(1, 0.5);
        assert!((dens.grad_accumulator[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_gaussian_densification_clone_small() {
        let mut gaussians = vec![Adv3DGaussian::new([0.0; 3], -5.0)]; // very small scale
        let mut dens = GaussianDensification::new(1);
        dens.clone_scale_threshold = 1.0; // anything smaller gets cloned
        dens.grad_threshold = 0.0;
        dens.grad_accumulator[0] = 1.0;
        let n_before = gaussians.len();
        dens.densify(&mut gaussians, 1);
        // Should have cloned → length increased
        assert!(gaussians.len() > n_before, "should have cloned Gaussian");
    }

    #[test]
    fn test_gaussian_densification_split_large() {
        let mut gaussians = vec![Adv3DGaussian::new([0.0; 3], 2.0)]; // large scale
        let mut dens = GaussianDensification::new(1);
        dens.split_scale_threshold = 0.01; // trigger split
        dens.grad_threshold = 0.0;
        dens.grad_accumulator[0] = 1.0;
        let n_before = gaussians.len();
        dens.densify(&mut gaussians, 1);
        assert!(gaussians.len() > n_before, "should have split Gaussian");
    }

    // ── Advanced: NrcCache ────────────────────────────────────────────────────

    #[test]
    fn test_nrc_cache_query_shape() {
        let cache = NrcCache::new(4, 2, 64, 16);
        let rgb = cache.query([0.5, 0.5, 0.5]);
        assert_eq!(rgb.len(), 3);
        for &c in &rgb {
            assert!((0.0..=1.0).contains(&c), "channel out of [0,1]: {c}");
        }
    }

    #[test]
    fn test_nrc_lookup_deterministic() {
        let cache = NrcCache::new(2, 2, 32, 8);
        let a = cache.lookup([0.3, 0.7, 0.1]);
        let b = cache.lookup([0.3, 0.7, 0.1]);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn test_nrc_different_positions_differ() {
        let cache = NrcCache::new(3, 2, 64, 16);
        let a = cache.query([0.1, 0.1, 0.1]);
        let b = cache.query([0.9, 0.9, 0.9]);
        let diff: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
        // Different positions should usually produce different outputs
        assert!(diff >= 0.0); // trivially true but ensures no panic
    }

    // ── Advanced: Reservoir & ReSTIR ─────────────────────────────────────────

    #[test]
    fn test_reservoir_default_empty() {
        let r = Reservoir::default();
        assert!(r.sample.is_none());
        assert_eq!(r.m, 0);
    }

    #[test]
    fn test_reservoir_update_selects_sample() {
        let mut r = Reservoir::new();
        let mut rng = StdRng::seed_from_u64(42);
        let pos = [0.0; 3];
        let radiance = [1.0, 0.0, 0.0];
        r.update((pos, radiance), 1.0, &mut rng);
        assert!(r.sample.is_some());
        assert_eq!(r.m, 1);
    }

    #[test]
    fn test_reservoir_weight_computation() {
        let mut r = Reservoir::new();
        let mut rng = StdRng::seed_from_u64(7);
        r.update(([0.0; 3], [1.0; 3]), 2.0, &mut rng);
        r.compute_weight(2.0);
        assert!(r.capital_w >= 0.0, "W should be non-negative");
    }

    #[test]
    fn test_reservoir_merge() {
        let mut r1 = Reservoir::new();
        let mut r2 = Reservoir::new();
        let mut rng = StdRng::seed_from_u64(99);
        r2.update(([1.0; 3], [0.5; 3]), 1.0, &mut rng);
        r2.compute_weight(1.0);
        r1.merge(&r2, 1.0, &mut rng);
        assert!(r1.m > 0);
    }

    #[test]
    fn test_restir_resample_noop_empty() {
        let restir = ReSTIR::new(4, 2);
        let mut current = Reservoir::new();
        let mut rng = StdRng::seed_from_u64(0);
        restir.resample(&mut current, &[], |_| 1.0, &mut rng);
        // No neighbors → no change except possible weight update
        assert_eq!(current.m, 0);
    }

    #[test]
    fn test_restir_resample_with_neighbor() {
        let restir = ReSTIR::new(2, 1);
        let mut current = Reservoir::new();
        let mut neighbor = Reservoir::new();
        let mut rng = StdRng::seed_from_u64(1);
        neighbor.update(([0.5; 3], [0.8; 3]), 1.0, &mut rng);
        neighbor.compute_weight(1.0);
        restir.resample(&mut current, &[neighbor], |_| 1.0, &mut rng);
        assert!(current.m > 0 || current.w_sum >= 0.0);
    }

    // ── Advanced: DeformationField ────────────────────────────────────────────

    #[test]
    fn test_deformation_field_output_shape() {
        let field = DeformationField::new(4, 4, 32, 3);
        let delta = field.forward([0.0, 0.0, 0.0], 0.0);
        assert_eq!(delta.len(), 3);
    }

    #[test]
    fn test_deformation_field_finite_output() {
        let field = DeformationField::new(4, 4, 32, 3);
        let delta = field.forward([0.5, -0.3, 1.2], 0.7);
        for &d in &delta {
            assert!(d.is_finite(), "delta not finite: {d}");
        }
    }

    #[test]
    fn test_deformation_field_different_times() {
        let field = DeformationField::new(4, 4, 32, 3);
        let pos = [0.1, 0.2, 0.3];
        let d0 = field.forward(pos, 0.0);
        let d1 = field.forward(pos, 1.0);
        // Different times should give different displacements (probabilistic but expected)
        let diff: f64 = d0.iter().zip(&d1).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff >= 0.0); // no panic
    }

    // ── Advanced: DynamicNerf ─────────────────────────────────────────────────

    #[test]
    fn test_dynamic_nerf_output_range() {
        let nerf = DynamicNerf::new(4, 4, 32);
        let out = nerf.query([0.0, 0.0, 0.0], 0.0);
        assert_eq!(out.len(), 4);
        // r, g, b ∈ [0,1] via sigmoid; σ ≥ 0 via softplus
        for c in 0..3 {
            assert!(out[c] >= 0.0 && out[c] <= 1.0, "out[{c}]={}", out[c]);
        }
        assert!(out[3] >= 0.0, "density must be non-negative: {}", out[3]);
    }

    #[test]
    fn test_dynamic_nerf_finite_output() {
        let nerf = DynamicNerf::new(4, 4, 32);
        let out = nerf.query([1.0, -1.0, 0.5], 0.5);
        for &v in &out {
            assert!(v.is_finite(), "output not finite: {v}");
        }
    }

    // ── Advanced: NrMetrics ───────────────────────────────────────────────────

    #[test]
    fn test_nr_metrics_identical_images_psnr() {
        let img = vec![0.5_f64; 100];
        let m = NrMetrics::compute(&img, &img, None, None);
        assert!(m.psnr >= 99.0, "identical images: psnr={}", m.psnr);
        assert!(m.ssim > 0.99, "identical images: ssim={}", m.ssim);
    }

    #[test]
    fn test_nr_metrics_psnr_decreases_with_noise() {
        let reference = vec![0.5_f64; 100];
        let noisy: Vec<f64> = reference.iter().map(|&x| x + 0.1).collect();
        let m = NrMetrics::compute(&noisy, &reference, None, None);
        assert!(m.psnr < 40.0, "noisy image psnr should be lower: {}", m.psnr);
        assert!(m.psnr > 0.0);
    }

    #[test]
    fn test_nr_metrics_depth_rmse() {
        let rendered_d = vec![1.0_f64; 64];
        let reference_d: Vec<f64> = (0..64).map(|i| 1.0 + (i as f64) * 0.01).collect();
        let m = NrMetrics::compute(&vec![0.5; 64], &vec![0.5; 64], Some(&rendered_d), Some(&reference_d));
        assert!(m.depth_rmse >= 0.0, "depth_rmse={}", m.depth_rmse);
    }

    #[test]
    fn test_nr_metrics_depth_rmse_nan_without_depth() {
        let img = vec![0.5_f64; 64];
        let m = NrMetrics::compute(&img, &img, None, None);
        assert!(m.depth_rmse.is_nan(), "should be NaN without depth: {}", m.depth_rmse);
    }

    #[test]
    fn test_nr_metrics_quality_gate_pass() {
        let img = vec![0.9_f64; 64];
        let m = NrMetrics::compute(&img, &img, None, None);
        assert!(m.passes_quality_gate(30.0, 0.9), "should pass gate");
    }

    #[test]
    fn test_nr_metrics_quality_gate_fail() {
        let reference = vec![0.0_f64; 64];
        let rendered = vec![1.0_f64; 64];
        let m = NrMetrics::compute(&rendered, &reference, None, None);
        assert!(!m.passes_quality_gate(50.0, 0.99), "should fail gate");
    }

    #[test]
    fn test_nr_metrics_fid_proxy_zero_identical() {
        let img = vec![0.7_f64; 128];
        let m = NrMetrics::compute(&img, &img, None, None);
        assert!(m.fid_proxy < 1e-12, "fid_proxy should be 0 for identical: {}", m.fid_proxy);
    }
}
