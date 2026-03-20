use super::*;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

// §1 SirenLayer tests

#[test]
fn test_siren_layer_creation() {
    let mut rng = StdRng::seed_from_u64(42);
    let cfg = InrSirenLayerConfig {
        fan_in: 2,
        fan_out: 64,
        omega_0: 30.0,
        is_first: true,
    };
    let layer = InrSirenLayer::new(&cfg, &mut rng);
    assert_eq!(layer.weights.len(), 64);
    assert_eq!(layer.weights[0].len(), 2);
    assert_eq!(layer.biases.len(), 64);
}

#[test]
fn test_siren_layer_forward_bounded() {
    let mut rng = StdRng::seed_from_u64(42);
    let cfg = InrSirenLayerConfig {
        fan_in: 2,
        fan_out: 32,
        omega_0: 30.0,
        is_first: true,
    };
    let layer = InrSirenLayer::new(&cfg, &mut rng);
    let out = layer.forward(&[0.5, 0.3]);
    // Output of sin() is bounded in [-1, 1]
    for &v in &out {
        assert!((-1.0..=1.0).contains(&v), "SIREN output {} out of bounds", v);
    }
}

#[test]
fn test_siren_layer_hidden_init() {
    let mut rng = StdRng::seed_from_u64(42);
    let cfg = InrSirenLayerConfig {
        fan_in: 64,
        fan_out: 64,
        omega_0: 30.0,
        is_first: false,
    };
    let layer = InrSirenLayer::new(&cfg, &mut rng);
    let limit = (6.0_f64 / 64.0).sqrt() / 30.0;
    for row in &layer.weights {
        for &w in row {
            assert!(
                w.abs() <= limit + 1e-10,
                "Weight {} exceeds init bound {}",
                w,
                limit
            );
        }
    }
}

// §2 SirenNetwork tests

#[test]
fn test_siren_network_creation() {
    let cfg = InrSirenConfig {
        input_dim: 2,
        hidden_dims: vec![64, 64],
        output_dim: 3,
        omega_0_first: 30.0,
        omega_0_hidden: 30.0,
    };
    let net = InrSirenNetwork::new(cfg, 42).expect("Failed to create SIREN");
    assert_eq!(net.layers.len(), 2);
    assert_eq!(net.out_weights.len(), 3);
}

#[test]
fn test_siren_network_forward() {
    let cfg = InrSirenConfig {
        input_dim: 2,
        hidden_dims: vec![32, 32],
        output_dim: 1,
        omega_0_first: 30.0,
        omega_0_hidden: 30.0,
    };
    let net = InrSirenNetwork::new(cfg, 42).expect("Failed to create SIREN");
    let out = net.forward(&[0.5, 0.5]);
    assert_eq!(out.len(), 1);
    assert!(out[0].is_finite());
}

#[test]
fn test_siren_network_gradient_fd() {
    let cfg = InrSirenConfig {
        input_dim: 2,
        hidden_dims: vec![32, 32],
        output_dim: 1,
        omega_0_first: 30.0,
        omega_0_hidden: 30.0,
    };
    let net = InrSirenNetwork::new(cfg, 42).expect("Failed to create SIREN");
    let grad = net.gradient_fd(&[0.3, 0.7], 1e-4);
    assert_eq!(grad.len(), 1); // one output
    assert_eq!(grad[0].len(), 2); // two inputs
    for &g in &grad[0] {
        assert!(g.is_finite());
    }
}

#[test]
fn test_siren_eikonal_loss() {
    let cfg = InrSirenConfig {
        input_dim: 3,
        hidden_dims: vec![32, 32],
        output_dim: 1,
        omega_0_first: 30.0,
        omega_0_hidden: 30.0,
    };
    let net = InrSirenNetwork::new(cfg, 42).expect("Failed to create SIREN");
    let pts = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.5, 0.5, 0.5],
        vec![1.0, 1.0, 1.0],
    ];
    let loss = net.eikonal_loss(&pts, 1e-4);
    assert!(loss >= 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_siren_empty_hidden_error() {
    let cfg = InrSirenConfig {
        input_dim: 2,
        hidden_dims: vec![],
        output_dim: 1,
        omega_0_first: 30.0,
        omega_0_hidden: 30.0,
    };
    assert!(InrSirenNetwork::new(cfg, 42).is_err());
}

#[test]
fn test_siren_1d_audio() {
    let cfg = InrSirenConfig {
        input_dim: 1,
        hidden_dims: vec![32, 32],
        output_dim: 1,
        omega_0_first: 30.0,
        omega_0_hidden: 30.0,
    };
    let net = InrSirenNetwork::new(cfg, 42).expect("Failed to create SIREN");
    let out = net.forward(&[0.5]);
    assert_eq!(out.len(), 1);
    assert!(out[0].is_finite());
}

// §3 FourierFeatureNetwork tests

#[test]
fn test_fourier_encoding() {
    let cfg = InrFourierConfig {
        input_dim: 2,
        n_frequencies: 16,
        sigma: 10.0,
        hidden_dims: vec![32],
        output_dim: 3,
    };
    let net = InrFourierFeatureNetwork::new(cfg, 42).expect("Failed to create FFN");
    let encoded = net.encode(&[0.5, 0.5]);
    assert_eq!(encoded.len(), 32); // 2 * 16
    for &v in &encoded {
        assert!((-1.0..=1.0).contains(&v));
    }
}

#[test]
fn test_fourier_forward() {
    let cfg = InrFourierConfig {
        input_dim: 2,
        n_frequencies: 16,
        sigma: 10.0,
        hidden_dims: vec![32],
        output_dim: 3,
    };
    let net = InrFourierFeatureNetwork::new(cfg, 42).expect("Failed to create FFN");
    let out = net.forward(&[0.5, 0.5]);
    assert_eq!(out.len(), 3);
    for &v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_fourier_different_sigma() {
    let cfg1 = InrFourierConfig {
        input_dim: 2,
        n_frequencies: 8,
        sigma: 1.0,
        hidden_dims: vec![16],
        output_dim: 1,
    };
    let cfg2 = InrFourierConfig {
        sigma: 100.0,
        ..cfg1.clone()
    };
    let net1 = InrFourierFeatureNetwork::new(cfg1, 42).expect("FFN1");
    let net2 = InrFourierFeatureNetwork::new(cfg2, 42).expect("FFN2");
    // Different sigma -> different B matrices -> different encodings
    let enc1 = net1.encode(&[0.3, 0.7]);
    let enc2 = net2.encode(&[0.3, 0.7]);
    let diff: f64 = enc1
        .iter()
        .zip(enc2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 0.0,
        "Different sigma should give different encodings"
    );
}

#[test]
fn test_fourier_b_matrix_shape() {
    let cfg = InrFourierConfig {
        input_dim: 3,
        n_frequencies: 64,
        sigma: 5.0,
        hidden_dims: vec![32],
        output_dim: 1,
    };
    let net = InrFourierFeatureNetwork::new(cfg, 42).expect("FFN");
    assert_eq!(net.b_matrix.len(), 64);
    assert_eq!(net.b_matrix[0].len(), 3);
}

// §4 HashGridEncoding tests

#[test]
fn test_hash_grid_creation() {
    let cfg = InrHashGridConfig {
        n_levels: 4,
        n_features_per_level: 2,
        log2_hashmap_size: 10,
        base_resolution: 16,
        per_level_scale: 2.0,
        input_dim: 3,
    };
    let hg = InrHashGridEncoding::new(cfg, 42).expect("Hash grid");
    assert_eq!(hg.tables.len(), 4);
    assert_eq!(hg.tables[0].len(), 1024); // 2^10
    assert_eq!(hg.output_dim(), 8); // 4 * 2
}

#[test]
fn test_hash_grid_encode() {
    let cfg = InrHashGridConfig {
        n_levels: 4,
        n_features_per_level: 2,
        log2_hashmap_size: 10,
        base_resolution: 16,
        per_level_scale: 2.0,
        input_dim: 3,
    };
    let hg = InrHashGridEncoding::new(cfg, 42).expect("Hash grid");
    let encoded = hg.encode(&[0.5, 0.5, 0.5]);
    assert_eq!(encoded.len(), 8);
    for &v in &encoded {
        assert!(v.is_finite());
    }
}

#[test]
fn test_hash_grid_boundary() {
    let cfg = InrHashGridConfig {
        n_levels: 4,
        n_features_per_level: 2,
        log2_hashmap_size: 10,
        base_resolution: 16,
        per_level_scale: 2.0,
        input_dim: 3,
    };
    let hg = InrHashGridEncoding::new(cfg, 42).expect("Hash grid");
    // Encode at corners
    let e0 = hg.encode(&[0.0, 0.0, 0.0]);
    let e1 = hg.encode(&[1.0, 1.0, 1.0]);
    assert_eq!(e0.len(), e1.len());
    // Different corners should give different encodings
    let diff: f64 = e0.iter().zip(e1.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 0.0 || e0.is_empty());
}

#[test]
fn test_hash_grid_2d() {
    let cfg = InrHashGridConfig {
        n_levels: 4,
        n_features_per_level: 2,
        log2_hashmap_size: 8,
        base_resolution: 8,
        per_level_scale: 2.0,
        input_dim: 2,
    };
    let hg = InrHashGridEncoding::new(cfg, 42).expect("Hash grid 2D");
    let encoded = hg.encode(&[0.3, 0.7]);
    assert_eq!(encoded.len(), 8);
}

#[test]
fn test_hash_grid_resolution_growth() {
    let cfg = InrHashGridConfig {
        n_levels: 4,
        n_features_per_level: 2,
        log2_hashmap_size: 8,
        base_resolution: 16,
        per_level_scale: 2.0,
        input_dim: 3,
    };
    let hg = InrHashGridEncoding::new(cfg, 42).expect("Hash grid");
    assert_eq!(hg.resolutions[0], 16);
    assert_eq!(hg.resolutions[1], 32);
    assert_eq!(hg.resolutions[2], 64);
    assert_eq!(hg.resolutions[3], 128);
}

#[test]
fn test_hash_grid_invalid_config() {
    let cfg = InrHashGridConfig {
        n_levels: 0,
        ..Default::default()
    };
    assert!(InrHashGridEncoding::new(cfg, 42).is_err());
}

// §5 NeuralSdf tests

#[test]
fn test_neural_sdf_creation() {
    let cfg = InrNeuralSdfConfig::default();
    let sdf = InrNeuralSdf::new(cfg, 42).expect("NeuralSdf");
    let val = sdf.compute_sdf(&[0.0, 0.0, 0.0]);
    assert!(val.is_finite());
}

#[test]
fn test_neural_sdf_gradient() {
    let cfg = InrNeuralSdfConfig {
        siren_config: InrSirenConfig {
            input_dim: 3,
            hidden_dims: vec![32, 32],
            output_dim: 1,
            omega_0_first: 30.0,
            omega_0_hidden: 30.0,
        },
        ..Default::default()
    };
    let sdf = InrNeuralSdf::new(cfg, 42).expect("NeuralSdf");
    let grad = sdf.sdf_gradient(&[0.5, 0.5, 0.5]);
    assert_eq!(grad.len(), 3);
    for &g in &grad {
        assert!(g.is_finite());
    }
}

#[test]
fn test_neural_sdf_sphere_trace() {
    let cfg = InrNeuralSdfConfig {
        siren_config: InrSirenConfig {
            input_dim: 3,
            hidden_dims: vec![16, 16],
            output_dim: 1,
            omega_0_first: 30.0,
            omega_0_hidden: 30.0,
        },
        max_trace_steps: 32,
        ..Default::default()
    };
    let sdf = InrNeuralSdf::new(cfg, 42).expect("NeuralSdf");
    let result = sdf.sphere_trace(&[0.0, 0.0, -5.0], &[0.0, 0.0, 1.0]);
    assert!(result.steps <= 32);
    assert!(result.distance.is_finite());
}

#[test]
fn test_neural_sdf_eikonal() {
    let cfg = InrNeuralSdfConfig {
        siren_config: InrSirenConfig {
            input_dim: 3,
            hidden_dims: vec![32, 32],
            output_dim: 1,
            omega_0_first: 30.0,
            omega_0_hidden: 30.0,
        },
        ..Default::default()
    };
    let sdf = InrNeuralSdf::new(cfg, 42).expect("NeuralSdf");
    let pts = vec![vec![0.1, 0.2, 0.3], vec![0.5, 0.5, 0.5]];
    let loss = sdf.eikonal_loss(&pts);
    assert!(loss >= 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_neural_sdf_train() {
    let cfg = InrNeuralSdfConfig {
        siren_config: InrSirenConfig {
            input_dim: 3,
            hidden_dims: vec![16, 16],
            output_dim: 1,
            omega_0_first: 30.0,
            omega_0_hidden: 30.0,
        },
        ..Default::default()
    };
    let mut sdf = InrNeuralSdf::new(cfg, 42).expect("NeuralSdf");
    let pts = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
    ];
    let vals = vec![0.0, 1.0, 1.0]; // on surface, far, far
    let eik_pts = vec![vec![0.5, 0.5, 0.5]];
    let loss = sdf
        .train(&pts, &vals, &eik_pts, 2, 0.01, 0.1)
        .expect("train");
    assert!(loss.is_finite());
}

// §6 OccupancyNetwork tests

#[test]
fn test_occupancy_creation() {
    let cfg = InrOccupancyConfig::default();
    let net = InrOccupancyNetwork::new(cfg, 42).expect("OccNet");
    let z = vec![0.0; 128];
    let occ = net.decode(&[0.5, 0.5, 0.5], &z);
    assert!((0.0..=1.0).contains(&occ));
}

#[test]
fn test_occupancy_encode() {
    let cfg = InrOccupancyConfig {
        point_dim: 3,
        latent_dim: 32,
        encoder_hidden: vec![64],
        decoder_hidden: vec![64],
    };
    let net = InrOccupancyNetwork::new(cfg, 42).expect("OccNet");
    let pc = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
    ];
    let z = net.encode(&pc).expect("encode");
    assert_eq!(z.len(), 32);
}

#[test]
fn test_occupancy_bce_loss() {
    let cfg = InrOccupancyConfig {
        point_dim: 3,
        latent_dim: 16,
        encoder_hidden: vec![32],
        decoder_hidden: vec![32],
    };
    let net = InrOccupancyNetwork::new(cfg, 42).expect("OccNet");
    let z = vec![0.0; 16];
    let pts = vec![vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]];
    let labels = vec![1.0, 0.0];
    let loss = net.bce_loss(&pts, &labels, &z);
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

#[test]
fn test_occupancy_marching_cubes() {
    let cfg = InrOccupancyConfig {
        point_dim: 3,
        latent_dim: 8,
        encoder_hidden: vec![16],
        decoder_hidden: vec![16],
    };
    let net = InrOccupancyNetwork::new(cfg, 42).expect("OccNet");
    let z = vec![0.0; 8];
    let surface = net.marching_cubes_simple(&z, 4, &[-1.0, -1.0, -1.0], &[1.0, 1.0, 1.0]);
    // Just verify it returns without error
    for pt in &surface {
        assert!(pt[0].is_finite());
    }
}

#[test]
fn test_occupancy_empty_pointcloud_error() {
    let cfg = InrOccupancyConfig::default();
    let net = InrOccupancyNetwork::new(cfg, 42).expect("OccNet");
    assert!(net.encode(&[]).is_err());
}

// §7 ImageFitting tests

#[test]
fn test_image_fitting_siren() {
    let cfg = InrImageFitConfig {
        width: 4,
        height: 4,
        channels: 3,
        use_siren: true,
        hidden_dims: vec![16, 16],
        omega_0: 30.0,
        ..Default::default()
    };
    let fitter = InrImageFitting::new(&cfg, 42).expect("ImgFit");
    let pred = fitter.predict(0.5, 0.5);
    assert_eq!(pred.len(), 3);
}

#[test]
fn test_image_fitting_fourier() {
    let cfg = InrImageFitConfig {
        width: 4,
        height: 4,
        channels: 3,
        use_siren: false,
        hidden_dims: vec![16, 16],
        n_frequencies: 8,
        fourier_sigma: 10.0,
        ..Default::default()
    };
    let fitter = InrImageFitting::new(&cfg, 42).expect("ImgFit");
    let pred = fitter.predict(0.5, 0.5);
    assert_eq!(pred.len(), 3);
}

#[test]
fn test_image_fitting_make_coords() {
    let coords = InrImageFitting::make_coords(4, 4);
    assert_eq!(coords.len(), 16);
    for c in &coords {
        assert!(c[0] > 0.0 && c[0] < 1.0);
        assert!(c[1] > 0.0 && c[1] < 1.0);
    }
}

#[test]
fn test_image_fitting_fit() {
    let cfg = InrImageFitConfig {
        width: 2,
        height: 2,
        channels: 1,
        use_siren: true,
        hidden_dims: vec![8, 8],
        omega_0: 30.0,
        ..Default::default()
    };
    let mut fitter = InrImageFitting::new(&cfg, 42).expect("ImgFit");
    let coords = InrImageFitting::make_coords(2, 2);
    let pixels = vec![0.5, 0.3, 0.7, 0.1]; // 4 pixels, 1 channel each
    let mse = fitter.fit_image(&pixels, &coords, 2, 0.01).expect("fit");
    assert!(mse.is_finite());
}

#[test]
fn test_image_fitting_dimension_mismatch() {
    let cfg = InrImageFitConfig {
        width: 2,
        height: 2,
        channels: 3,
        use_siren: true,
        hidden_dims: vec![8],
        omega_0: 30.0,
        ..Default::default()
    };
    let mut fitter = InrImageFitting::new(&cfg, 42).expect("ImgFit");
    let coords = InrImageFitting::make_coords(2, 2);
    let pixels = vec![0.5; 4]; // wrong: should be 12
    assert!(fitter.fit_image(&pixels, &coords, 1, 0.01).is_err());
}

// §8 SuperResolution tests

#[test]
fn test_super_resolution_creation() {
    let cfg = InrImageFitConfig {
        width: 4,
        height: 4,
        channels: 1,
        use_siren: true,
        hidden_dims: vec![16, 16],
        omega_0: 30.0,
        ..Default::default()
    };
    let sr = InrSuperResolution::new(&cfg, 2.0, 42).expect("SR");
    assert!((sr.scale_factor - 2.0).abs() < 1e-10);
}

#[test]
fn test_super_resolution_upscale() {
    let cfg = InrImageFitConfig {
        width: 2,
        height: 2,
        channels: 1,
        use_siren: true,
        hidden_dims: vec![8, 8],
        omega_0: 30.0,
        ..Default::default()
    };
    let sr = InrSuperResolution::new(&cfg, 2.0, 42).expect("SR");
    let (coords, pixels) = sr.upscale(2, 2);
    // 2x upscale -> 4x4 = 16 pixels
    assert_eq!(coords.len(), 16);
    assert_eq!(pixels.len(), 16);
}

#[test]
fn test_super_resolution_train_upscale() {
    let cfg = InrImageFitConfig {
        width: 2,
        height: 2,
        channels: 1,
        use_siren: true,
        hidden_dims: vec![8, 8],
        omega_0: 30.0,
        ..Default::default()
    };
    let mut sr = InrSuperResolution::new(&cfg, 2.0, 42).expect("SR");
    let lr_pixels = vec![0.5, 0.3, 0.7, 0.1];
    let mse = sr.train_low_res(&lr_pixels, 2, 2, 1, 0.01).expect("train");
    assert!(mse.is_finite());
    let (_, hr_pixels) = sr.upscale(2, 2);
    assert_eq!(hr_pixels.len(), 16);
}

// §9 MetaSdf tests

#[test]
fn test_meta_sdf_creation() {
    let cfg = InrMetaSdfConfig {
        sdf_config: InrNeuralSdfConfig {
            siren_config: InrSirenConfig {
                input_dim: 3,
                hidden_dims: vec![16, 16],
                output_dim: 1,
                omega_0_first: 30.0,
                omega_0_hidden: 30.0,
            },
            ..Default::default()
        },
        inner_steps: 3,
        inner_lr: 0.01,
        meta_lr: 0.001,
    };
    let msdf = InrMetaSdf::new(cfg, 42).expect("MetaSdf");
    let val = msdf.base_sdf.compute_sdf(&[0.0, 0.0, 0.0]);
    assert!(val.is_finite());
}

#[test]
fn test_meta_sdf_meta_train() {
    let cfg = InrMetaSdfConfig {
        sdf_config: InrNeuralSdfConfig {
            siren_config: InrSirenConfig {
                input_dim: 3,
                hidden_dims: vec![8, 8],
                output_dim: 1,
                omega_0_first: 30.0,
                omega_0_hidden: 30.0,
            },
            ..Default::default()
        },
        inner_steps: 2,
        inner_lr: 0.01,
        meta_lr: 0.001,
    };
    let mut msdf = InrMetaSdf::new(cfg, 42).expect("MetaSdf");
    let tasks = vec![
        InrSdfTask {
            points: vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]],
            sdf_values: vec![0.0, 1.0],
        },
        InrSdfTask {
            points: vec![vec![0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            sdf_values: vec![0.0, 1.0],
        },
    ];
    let loss = msdf.meta_train(&tasks, 1).expect("meta_train");
    assert!(loss.is_finite());
}

#[test]
fn test_meta_sdf_adapt() {
    let cfg = InrMetaSdfConfig {
        sdf_config: InrNeuralSdfConfig {
            siren_config: InrSirenConfig {
                input_dim: 3,
                hidden_dims: vec![8, 8],
                output_dim: 1,
                omega_0_first: 30.0,
                omega_0_hidden: 30.0,
            },
            ..Default::default()
        },
        inner_steps: 2,
        inner_lr: 0.01,
        meta_lr: 0.001,
    };
    let msdf = InrMetaSdf::new(cfg, 42).expect("MetaSdf");
    let task = InrSdfTask {
        points: vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]],
        sdf_values: vec![0.0, 1.0],
    };
    let adapted = msdf.adapt(&task, 3).expect("adapt");
    let val = adapted.compute_sdf(&[0.5, 0.0, 0.0]);
    assert!(val.is_finite());
}

#[test]
fn test_meta_sdf_empty_tasks_error() {
    let cfg = InrMetaSdfConfig::default();
    let mut msdf = InrMetaSdf::new(cfg, 42).expect("MetaSdf");
    assert!(msdf.meta_train(&[], 1).is_err());
}

// §10 InrMetrics tests

#[test]
fn test_psnr_identical() {
    let a = vec![0.5, 0.3, 0.7, 0.1];
    let psnr = InrMetrics::psnr(&a, &a, 1.0).expect("psnr");
    assert!(psnr >= 90.0, "Identical signals should have very high PSNR");
}

#[test]
fn test_psnr_different() {
    let a = vec![0.5, 0.3, 0.7, 0.1];
    let b = vec![0.6, 0.4, 0.8, 0.2];
    let psnr = InrMetrics::psnr(&a, &b, 1.0).expect("psnr");
    assert!(psnr > 0.0 && psnr < 100.0);
}

#[test]
fn test_psnr_empty_error() {
    assert!(InrMetrics::psnr(&[], &[], 1.0).is_err());
}

#[test]
fn test_ssim_identical() {
    let a = vec![0.5; 4 * 4 ]; // 4x4, 1 channel
    let ssim = InrMetrics::ssim(&a, &a, 4, 4, 1).expect("ssim");
    assert!(
        (ssim - 1.0).abs() < 0.01,
        "Identical images should have SSIM near 1.0"
    );
}

#[test]
fn test_ssim_different() {
    let mut rng = StdRng::seed_from_u64(123);
    let a: Vec<f64> = (0..8 * 8).map(|_| rng.random::<f64>()).collect();
    let b: Vec<f64> = (0..8 * 8).map(|_| rng.random::<f64>()).collect();
    let ssim = InrMetrics::ssim(&a, &b, 8, 8, 1).expect("ssim");
    assert!(ssim < 1.0);
}

#[test]
fn test_chamfer_distance() {
    let a = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
    let b = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
    let cd = InrMetrics::chamfer_distance(&a, &b).expect("chamfer");
    assert!(cd < 1e-10, "Identical point sets should have CD near 0");
}

#[test]
fn test_chamfer_distance_nonzero() {
    let a = vec![vec![0.0, 0.0, 0.0]];
    let b = vec![vec![1.0, 0.0, 0.0]];
    let cd = InrMetrics::chamfer_distance(&a, &b).expect("chamfer");
    assert!((cd - 2.0).abs() < 1e-10, "CD should be 1+1 = 2");
}

#[test]
fn test_hausdorff_distance() {
    let a = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
    let b = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
    let hd = InrMetrics::hausdorff_distance(&a, &b).expect("hausdorff");
    assert!(hd < 1e-10);
}

#[test]
fn test_hausdorff_nonzero() {
    let a = vec![vec![0.0, 0.0, 0.0]];
    let b = vec![vec![3.0, 4.0, 0.0]]; // distance 5
    let hd = InrMetrics::hausdorff_distance(&a, &b).expect("hausdorff");
    assert!((hd - 5.0).abs() < 1e-10);
}

#[test]
fn test_hausdorff_empty_error() {
    assert!(InrMetrics::hausdorff_distance(&[], &[vec![0.0]]).is_err());
}

#[test]
fn test_iou_identical() {
    let a = vec![true, false, true, true];
    let iou = InrMetrics::iou(&a, &a).expect("iou");
    assert!((iou - 1.0).abs() < 1e-10);
}

#[test]
fn test_iou_no_overlap() {
    let a = vec![true, true, false, false];
    let b = vec![false, false, true, true];
    let iou = InrMetrics::iou(&a, &b).expect("iou");
    assert!(iou < 1e-10);
}

#[test]
fn test_iou_partial() {
    let a = vec![true, true, false, false];
    let b = vec![true, false, true, false];
    let iou = InrMetrics::iou(&a, &b).expect("iou");
    // intersection=1, union=3 -> IoU=1/3
    assert!((iou - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_iou_empty_error() {
    assert!(InrMetrics::iou(&[], &[]).is_err());
}

#[test]
fn test_inr_report_default() {
    let report = InrReport::default();
    assert!(report.psnr.is_none());
    assert!(report.ssim.is_none());
    assert!(report.chamfer.is_none());
    assert!(report.hausdorff.is_none());
    assert!(report.iou.is_none());
}
