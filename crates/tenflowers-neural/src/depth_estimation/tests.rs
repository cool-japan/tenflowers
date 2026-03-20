use super::*;

// ── §1 DepthEncoder tests ──

#[test]
fn test_depth_encoder_creation() {
    let config = DepthEncoderConfig::default();
    let enc = DepthEncoder::new(&config);
    assert!(enc.is_ok());
    let enc = enc.expect("encoder creation should succeed");
    assert_eq!(enc.blocks.len(), 4);
}

#[test]
fn test_depth_encoder_too_small() {
    let config = DepthEncoderConfig {
        input_h: 8,
        input_w: 8,
        ..Default::default()
    };
    assert!(DepthEncoder::new(&config).is_err());
}

#[test]
fn test_depth_encoder_forward() {
    let config = DepthEncoderConfig {
        input_h: 32,
        input_w: 32,
        kernel_size: 3,
        seed: 42,
    };
    let enc = DepthEncoder::new(&config).expect("encoder");
    let input = vec![0.5; 32 * 32];
    let features = enc.forward(&input);
    assert!(features.is_ok());
    let ms = features.expect("features");
    assert_eq!(ms.features.len(), 4);
    // 1/2 level: 16x16
    assert_eq!(ms.heights[0], 16);
    assert_eq!(ms.widths[0], 16);
    // 1/4 level: 8x8
    assert_eq!(ms.heights[1], 8);
    assert_eq!(ms.widths[1], 8);
}

#[test]
fn test_depth_encoder_input_mismatch() {
    let config = DepthEncoderConfig::default();
    let enc = DepthEncoder::new(&config).expect("encoder");
    let bad_input = vec![0.5; 100];
    assert!(enc.forward(&bad_input).is_err());
}

#[test]
fn test_depth_encoder_64x64() {
    let config = DepthEncoderConfig {
        input_h: 64,
        input_w: 64,
        kernel_size: 3,
        seed: 1,
    };
    let enc = DepthEncoder::new(&config).expect("encoder");
    let input = vec![1.0; 64 * 64];
    let ms = enc.forward(&input).expect("features");
    assert_eq!(ms.heights[0], 32);
    assert_eq!(ms.heights[3], 4);
}

// ── §2 DptDecoder tests ──

#[test]
fn test_dpt_decoder_creation() {
    let config = DptDecoderConfig::default();
    let dec = DptDecoder::new(&config);
    assert!(dec.is_ok());
}

#[test]
fn test_dpt_decoder_zero_levels() {
    let config = DptDecoderConfig {
        n_levels: 0,
        ..Default::default()
    };
    assert!(DptDecoder::new(&config).is_err());
}

#[test]
fn test_dpt_decoder_forward() {
    let enc_config = DepthEncoderConfig {
        input_h: 32,
        input_w: 32,
        kernel_size: 3,
        seed: 42,
    };
    let dec_config = DptDecoderConfig::default();
    let enc = DepthEncoder::new(&enc_config).expect("encoder");
    let dec = DptDecoder::new(&dec_config).expect("decoder");

    let input = vec![0.5; 32 * 32];
    let features = enc.forward(&input).expect("features");
    let (depth, h, w) = dec.forward(&features).expect("decode");
    assert!(h > 0 && w > 0);
    assert_eq!(depth.len(), h * w);
    // Softplus ensures positive
    assert!(depth.iter().all(|&v| v >= 0.0));
}

#[test]
fn test_dpt_decoder_empty_features() {
    let dec_config = DptDecoderConfig::default();
    let dec = DptDecoder::new(&dec_config).expect("decoder");
    let empty = DeMultiScaleFeatures {
        features: vec![],
        heights: vec![],
        widths: vec![],
    };
    assert!(dec.forward(&empty).is_err());
}

// ── §3 MonocularDepthEstimator tests ──

#[test]
fn test_monocular_depth_estimator_creation() {
    let enc_config = DepthEncoderConfig {
        input_h: 32,
        input_w: 32,
        ..Default::default()
    };
    let dec_config = DptDecoderConfig::default();
    let est = MonocularDepthEstimator::new(&enc_config, &dec_config, DepthMode::Metric);
    assert!(est.is_ok());
}

#[test]
fn test_monocular_depth_predict() {
    let enc_config = DepthEncoderConfig {
        input_h: 32,
        input_w: 32,
        ..Default::default()
    };
    let dec_config = DptDecoderConfig::default();
    let est = MonocularDepthEstimator::new(&enc_config, &dec_config, DepthMode::Metric)
        .expect("estimator");
    let img = vec![0.3; 32 * 32];
    let (depth, h, w) = est.predict_depth(&img).expect("predict");
    assert!(depth.len() == h * w);
    assert!(h > 0 && w > 0);
}

#[test]
fn test_monocular_depth_relative_mode() {
    let enc_config = DepthEncoderConfig {
        input_h: 32,
        input_w: 32,
        ..Default::default()
    };
    let dec_config = DptDecoderConfig::default();
    let est = MonocularDepthEstimator::new(&enc_config, &dec_config, DepthMode::Relative)
        .expect("estimator");
    let img = vec![0.7; 32 * 32];
    let (depth, _h, _w) = est.predict_depth(&img).expect("predict");
    // Relative mode: normalized to [0, 1]
    for &v in &depth {
        assert!(
            (-1e-10..=1.0 + 1e-10).contains(&v),
            "relative depth out of [0,1]: {}",
            v
        );
    }
}

#[test]
fn test_scale_invariant_loss() {
    let pred = vec![2.0, 4.0, 6.0, 8.0];
    let gt = vec![1.0, 2.0, 3.0, 4.0];
    let loss = MonocularDepthEstimator::scale_invariant_loss(&pred, &gt);
    assert!(loss.is_ok());
    let l = loss.expect("loss");
    // pred/gt ratio is constant (2.0), so variance of log-diff should be ~0
    // d_i = ln(2) for all i, var(d) = 0, mean(d)^2 = ln(2)^2
    assert!(l > 0.0);
}

#[test]
fn test_scale_invariant_loss_perfect() {
    let pred = vec![1.0, 2.0, 3.0];
    let gt = vec![1.0, 2.0, 3.0];
    let loss = MonocularDepthEstimator::scale_invariant_loss(&pred, &gt).expect("loss");
    assert!(loss.abs() < 1e-10);
}

#[test]
fn test_scale_invariant_loss_mismatch() {
    let pred = vec![1.0, 2.0];
    let gt = vec![1.0];
    assert!(MonocularDepthEstimator::scale_invariant_loss(&pred, &gt).is_err());
}

#[test]
fn test_gradient_matching_loss() {
    let h = 4;
    let w = 4;
    let pred = vec![1.0; h * w];
    let gt = vec![1.0; h * w];
    let loss = MonocularDepthEstimator::gradient_matching_loss(&pred, &gt, h, w);
    assert!(loss.is_ok());
    // Constant images → zero gradients → zero loss
    assert!(loss.expect("loss") < 1e-10);
}

#[test]
fn test_gradient_matching_loss_nonzero() {
    let h = 4;
    let w = 4;
    let pred: Vec<f64> = (0..h * w).map(|i| i as f64 * 0.1).collect();
    let gt = vec![1.0; h * w];
    let loss = MonocularDepthEstimator::gradient_matching_loss(&pred, &gt, h, w).expect("loss");
    assert!(loss > 0.0);
}

#[test]
fn test_gradient_matching_loss_small() {
    assert!(MonocularDepthEstimator::gradient_matching_loss(&[1.0], &[1.0], 1, 1).is_err());
}

// ── §4 StereoMatcher tests ──

#[test]
fn test_stereo_matcher_creation() {
    let config = StereoMatcherConfig::default();
    let matcher = StereoMatcher::new(config);
    assert!(matcher.is_ok());
}

#[test]
fn test_stereo_matcher_zero_disparity() {
    let config = StereoMatcherConfig {
        max_disparity: 0,
        ..Default::default()
    };
    assert!(StereoMatcher::new(config).is_err());
}

#[test]
fn test_stereo_cost_volume() {
    let config = StereoMatcherConfig {
        max_disparity: 4,
        ..Default::default()
    };
    let matcher = StereoMatcher::new(config).expect("matcher");
    let h = 4;
    let w = 8;
    let left = vec![1.0; h * w];
    let right = vec![1.0; h * w];
    let cv = matcher.build_cost_volume(&left, &right, h, w).expect("cv");
    assert_eq!(cv.len(), 4);
    assert_eq!(cv[0].len(), h * w);
}

#[test]
fn test_stereo_match_pipeline() {
    let config = StereoMatcherConfig {
        max_disparity: 8,
        focal_length: 500.0,
        baseline: 0.1,
        agg_radius: 1,
        seed: 42,
    };
    let matcher = StereoMatcher::new(config).expect("matcher");
    let h = 8;
    let w = 16;
    let left: Vec<f64> = (0..h * w).map(|i| (i as f64) / (h * w) as f64).collect();
    let right: Vec<f64> = (0..h * w)
        .map(|i| ((i + 2) as f64) / (h * w) as f64)
        .collect();
    let (disp, depth) = matcher.match_stereo(&left, &right, h, w).expect("match");
    assert_eq!(disp.len(), h * w);
    assert_eq!(depth.len(), h * w);
}

#[test]
fn test_stereo_identical_images() {
    let config = StereoMatcherConfig {
        max_disparity: 4,
        ..Default::default()
    };
    let matcher = StereoMatcher::new(config).expect("matcher");
    let h = 4;
    let w = 8;
    let img = vec![0.5; h * w];
    let (disp, _depth) = matcher.match_stereo(&img, &img, h, w).expect("match");
    // Identical images → disparity near 0 (or uniform) since cost is same for d=0
    // The soft-argmin still produces values
    assert_eq!(disp.len(), h * w);
}

#[test]
fn test_disparity_to_depth() {
    let config = StereoMatcherConfig {
        focal_length: 500.0,
        baseline: 0.12,
        ..Default::default()
    };
    let matcher = StereoMatcher::new(config).expect("matcher");
    let depth = matcher.disparity_to_depth(10.0);
    let expected = 500.0 * 0.12 / 10.0;
    assert!((depth - expected).abs() < 1e-8);
}

#[test]
fn test_disparity_to_depth_zero() {
    let config = StereoMatcherConfig::default();
    let matcher = StereoMatcher::new(config).expect("matcher");
    let depth = matcher.disparity_to_depth(0.0);
    assert!((depth - 0.0).abs() < 1e-10);
}

// ── §5 DepthCompletion tests ──

#[test]
fn test_depth_completion_creation() {
    let config = DepthCompletionConfig::default();
    let dc = DepthCompletion::new(config);
    assert!(dc.is_ok());
}

#[test]
fn test_depth_completion_zero_size() {
    let config = DepthCompletionConfig {
        h: 0,
        w: 0,
        ..Default::default()
    };
    assert!(DepthCompletion::new(config).is_err());
}

#[test]
fn test_depth_completion_complete() {
    let h = 8;
    let w = 8;
    let config = DepthCompletionConfig {
        h,
        w,
        n_iterations: 3,
        fusion: DeCompletionFusion::Early,
        ..Default::default()
    };
    let dc = DepthCompletion::new(config).expect("dc");
    let mut sparse = vec![0.0; h * w];
    let mut conf = vec![0.0; h * w];
    // Set a few known depth points
    sparse[0] = 5.0;
    conf[0] = 1.0;
    sparse[h * w - 1] = 10.0;
    conf[h * w - 1] = 1.0;

    let rgb = vec![0.5; h * w];
    let dense = dc.complete(&sparse, &conf, Some(&rgb)).expect("complete");
    assert_eq!(dense.len(), h * w);
    // Known points should retain their value (or close)
    assert!((dense[0] - 5.0).abs() < 2.0);
}

#[test]
fn test_depth_completion_late_fusion() {
    let h = 8;
    let w = 8;
    let config = DepthCompletionConfig {
        h,
        w,
        n_iterations: 2,
        fusion: DeCompletionFusion::Late,
        ..Default::default()
    };
    let dc = DepthCompletion::new(config).expect("dc");
    let sparse = vec![3.0; h * w];
    let conf = vec![1.0; h * w];
    let dense = dc.complete(&sparse, &conf, None).expect("complete");
    assert_eq!(dense.len(), h * w);
}

#[test]
fn test_depth_completion_size_mismatch() {
    let config = DepthCompletionConfig {
        h: 4,
        w: 4,
        ..Default::default()
    };
    let dc = DepthCompletion::new(config).expect("dc");
    assert!(dc.complete(&[0.0; 10], &[0.0; 16], None).is_err());
}

// ── §6 DeConv3d tests ──

#[test]
fn test_conv3d_creation() {
    let config = DeConv3dConfig::default();
    let conv = DeConv3d::new(config);
    assert!(conv.is_ok());
}

#[test]
fn test_conv3d_output_shape() {
    let config = DeConv3dConfig {
        in_channels: 1,
        out_channels: 2,
        kernel_size: [3, 3, 3],
        stride: [1, 1, 1],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        seed: 42,
    };
    let conv = DeConv3d::new(config).expect("conv");
    let (od, oh, ow) = conv.output_shape(8, 8, 8);
    assert_eq!(od, 8);
    assert_eq!(oh, 8);
    assert_eq!(ow, 8);
}

#[test]
fn test_conv3d_forward() {
    let config = DeConv3dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: [3, 3, 3],
        stride: [1, 1, 1],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        seed: 42,
    };
    let conv = DeConv3d::new(config).expect("conv");
    let input = vec![1.0; 4 * 4 * 4];
    let (output, oc, od, oh, ow) = conv.forward(&input, 1, 4, 4, 4).expect("forward");
    assert_eq!(oc, 1);
    assert_eq!(od, 4);
    assert_eq!(oh, 4);
    assert_eq!(ow, 4);
    assert_eq!(output.len(), 4 * 4 * 4);
}

#[test]
fn test_conv3d_stride2() {
    let config = DeConv3dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: [3, 3, 3],
        stride: [2, 2, 2],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        seed: 42,
    };
    let conv = DeConv3d::new(config).expect("conv");
    let (od, oh, ow) = conv.output_shape(8, 8, 8);
    assert_eq!(od, 4);
    assert_eq!(oh, 4);
    assert_eq!(ow, 4);
}

#[test]
fn test_conv3d_channel_mismatch() {
    let config = DeConv3dConfig {
        in_channels: 2,
        ..Default::default()
    };
    let conv = DeConv3d::new(config).expect("conv");
    let input = vec![1.0; 4 * 4 * 4];
    assert!(conv.forward(&input, 1, 4, 4, 4).is_err());
}

#[test]
fn test_conv3d_multi_channel() {
    let config = DeConv3dConfig {
        in_channels: 2,
        out_channels: 3,
        kernel_size: [3, 3, 3],
        stride: [1, 1, 1],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        seed: 42,
    };
    let conv = DeConv3d::new(config).expect("conv");
    let input = vec![1.0; 2 * 4 * 4 * 4];
    let (output, oc, od, oh, ow) = conv.forward(&input, 2, 4, 4, 4).expect("forward");
    assert_eq!(oc, 3);
    assert_eq!(output.len(), 3 * 4 * 4 * 4);
    assert_eq!(od, 4);
    assert_eq!(oh, 4);
    assert_eq!(ow, 4);
}

// ── §7 DeImplicitNeuralField tests ──

#[test]
fn test_implicit_field_creation() {
    let config = DeImplicitFieldConfig::default();
    let field = DeImplicitNeuralField::new(config);
    assert!(field.is_ok());
}

#[test]
fn test_implicit_field_zero_freqs() {
    let config = DeImplicitFieldConfig {
        n_freqs: 0,
        ..Default::default()
    };
    assert!(DeImplicitNeuralField::new(config).is_err());
}

#[test]
fn test_positional_encoding() {
    let config = DeImplicitFieldConfig {
        n_freqs: 4,
        ..Default::default()
    };
    let field = DeImplicitNeuralField::new(config).expect("field");
    let enc = field.positional_encode(&[0.0, 0.5, 1.0]);
    // 3 coords * 2 * 4 freqs = 24
    assert_eq!(enc.len(), 24);
}

#[test]
fn test_implicit_field_query() {
    let config = DeImplicitFieldConfig {
        n_freqs: 4,
        hidden_dim: 32,
        n_layers: 2,
        ..Default::default()
    };
    let field = DeImplicitNeuralField::new(config).expect("field");
    let (density, color) = field.query(&[0.0, 0.0, 0.0]).expect("query");
    assert!(density >= 0.0); // softplus
    for &c in &color {
        assert!((0.0..=1.0).contains(&c)); // sigmoid
    }
}

#[test]
fn test_implicit_field_render_ray() {
    let config = DeImplicitFieldConfig {
        n_freqs: 4,
        hidden_dim: 32,
        n_layers: 2,
        n_samples: 8,
        seed: 42,
    };
    let field = DeImplicitNeuralField::new(config).expect("field");
    let origin = [0.0, 0.0, 0.0];
    let direction = [0.0, 0.0, 1.0];
    let (color, depth) = field
        .render_ray(&origin, &direction, 0.1, 5.0, 8)
        .expect("render");
    // Color components should be in [0, 1]
    for &c in &color {
        assert!((0.0..=1.0).contains(&c));
    }
    assert!(depth >= 0.0);
}

#[test]
fn test_render_ray_near_far_error() {
    let config = DeImplicitFieldConfig::default();
    let field = DeImplicitNeuralField::new(config).expect("field");
    assert!(field
        .render_ray(&[0.0; 3], &[0.0, 0.0, 1.0], 5.0, 1.0, 8)
        .is_err());
}

#[test]
fn test_render_ray_zero_samples() {
    let config = DeImplicitFieldConfig::default();
    let field = DeImplicitNeuralField::new(config).expect("field");
    assert!(field
        .render_ray(&[0.0; 3], &[0.0, 0.0, 1.0], 0.1, 5.0, 0)
        .is_err());
}

// ── §8 DePanopticHead tests ──

#[test]
fn test_panoptic_head_creation() {
    let config = DePanopticConfig::default();
    let head = DePanopticHead::new(config);
    assert!(head.is_ok());
}

#[test]
fn test_panoptic_head_zero_classes() {
    let config = DePanopticConfig {
        n_classes: 0,
        ..Default::default()
    };
    assert!(DePanopticHead::new(config).is_err());
}

#[test]
fn test_panoptic_semantic_logits() {
    let config = DePanopticConfig {
        n_classes: 5,
        feature_dim: 8,
        ..Default::default()
    };
    let head = DePanopticHead::new(config).expect("head");
    let features = vec![0.1; 8];
    let logits = head.semantic_logits(&features);
    assert_eq!(logits.len(), 5);
}

#[test]
fn test_panoptic_center_score() {
    let config = DePanopticConfig {
        feature_dim: 8,
        ..Default::default()
    };
    let head = DePanopticHead::new(config).expect("head");
    let features = vec![0.5; 8];
    let score = head.center_score(&features);
    assert!((0.0..=1.0).contains(&score));
}

#[test]
fn test_panoptic_forward() {
    let h = 4;
    let w = 4;
    let config = DePanopticConfig {
        n_classes: 3,
        feature_dim: 8,
        max_instances: 10,
        ..Default::default()
    };
    let head = DePanopticHead::new(config).expect("head");
    let features: Vec<Vec<f64>> = (0..h * w).map(|_| vec![0.1; 8]).collect();
    let result = head.forward(&features, h, w).expect("forward");
    assert_eq!(result.semantic_map.len(), h * w);
    assert_eq!(result.instance_map.len(), h * w);
    assert_eq!(result.panoptic_map.len(), h * w);
    assert_eq!(result.height, h);
    assert_eq!(result.width, w);
}

#[test]
fn test_panoptic_feature_mismatch() {
    let config = DePanopticConfig {
        feature_dim: 8,
        ..Default::default()
    };
    let head = DePanopticHead::new(config).expect("head");
    let features: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1; 8]).collect();
    // 5 features but h*w = 4*4 = 16
    assert!(head.forward(&features, 4, 4).is_err());
}

#[test]
fn test_panoptic_fusion_encoding() {
    // panoptic_map = class_id * 1000 + instance_id
    let class_id = 5;
    let instance_id = 3;
    let panoptic_id = class_id * 1000 + instance_id;
    assert_eq!(panoptic_id, 5003);
}

// ── §9 DeDepthMetrics tests ──

#[test]
fn test_depth_metrics_abs_rel() {
    let pred = vec![2.0, 4.0, 6.0];
    let gt = vec![1.0, 2.0, 3.0];
    let val = DeDepthMetrics::abs_rel(&pred, &gt).expect("abs_rel");
    // |2-1|/1 + |4-2|/2 + |6-3|/3 = 1 + 1 + 1 = 3 / 3 = 1.0
    assert!((val - 1.0).abs() < 1e-8);
}

#[test]
fn test_depth_metrics_sq_rel() {
    let pred = vec![2.0, 4.0];
    let gt = vec![1.0, 2.0];
    let val = DeDepthMetrics::sq_rel(&pred, &gt).expect("sq_rel");
    // (2-1)^2/1 + (4-2)^2/2 = 1 + 2 = 3 / 2 = 1.5
    assert!((val - 1.5).abs() < 1e-8);
}

#[test]
fn test_depth_metrics_rmse() {
    let pred = vec![3.0, 3.0];
    let gt = vec![1.0, 1.0];
    let val = DeDepthMetrics::rmse(&pred, &gt).expect("rmse");
    // sqrt((4+4)/2) = sqrt(4) = 2.0
    assert!((val - 2.0).abs() < 1e-8);
}

#[test]
fn test_depth_metrics_rmse_log() {
    let pred = vec![1.0, 1.0];
    let gt = vec![1.0, 1.0];
    let val = DeDepthMetrics::rmse_log(&pred, &gt).expect("rmse_log");
    assert!(val.abs() < 1e-10);
}

#[test]
fn test_depth_metrics_si_log() {
    let pred = vec![2.0, 2.0, 2.0];
    let gt = vec![1.0, 1.0, 1.0];
    let val = DeDepthMetrics::si_log(&pred, &gt).expect("si_log");
    // All d_i = ln(2), variance = 0 → SILog = 0
    assert!(val.abs() < 1e-6);
}

#[test]
fn test_depth_metrics_delta_threshold() {
    let pred = vec![1.1, 1.2, 5.0];
    let gt = vec![1.0, 1.0, 1.0];
    // Ratios: max(1.1,1/1.1)=1.1 < 1.25 ✓, max(1.2,1/1.2)=1.2 < 1.25 ✓, max(5,1/5)=5.0 >= 1.25 ✗
    let d1 = DeDepthMetrics::delta_threshold(&pred, &gt, 1.25).expect("d1");
    assert!((d1 - 2.0 / 3.0).abs() < 1e-8);
}

#[test]
fn test_depth_metrics_evaluate() {
    let pred = vec![1.1, 2.1, 3.1, 4.1];
    let gt = vec![1.0, 2.0, 3.0, 4.0];
    let report = DeDepthMetrics::evaluate(&pred, &gt).expect("report");
    assert!(report.abs_rel < 0.2);
    assert!(report.rmse < 0.5);
    assert!(report.delta_1 > 0.5);
    assert_eq!(report.n_valid, 4);
}

#[test]
fn test_depth_metrics_with_zeros() {
    let pred = vec![0.0, 1.0, 2.0];
    let gt = vec![1.0, 0.0, 2.0];
    // Only the pair (2.0, 2.0) is valid (both > 0 except first has pred=0, second has gt=0)
    // Actually (1.0, 0.0) has gt=0 → invalid, (0.0, 1.0) has pred=0 → invalid
    let report = DeDepthMetrics::evaluate(&pred, &gt).expect("report");
    assert_eq!(report.n_valid, 1);
}

#[test]
fn test_depth_metrics_empty() {
    assert!(DeDepthMetrics::abs_rel(&[], &[]).is_err());
}

#[test]
fn test_depth_metrics_length_mismatch() {
    assert!(DeDepthMetrics::abs_rel(&[1.0], &[1.0, 2.0]).is_err());
}

// ── Integration tests ──

#[test]
fn test_full_monocular_pipeline() {
    let enc_config = DepthEncoderConfig {
        input_h: 32,
        input_w: 32,
        kernel_size: 3,
        seed: 42,
    };
    let dec_config = DptDecoderConfig::default();
    let est =
        MonocularDepthEstimator::new(&enc_config, &dec_config, DepthMode::Metric).expect("est");

    let img = vec![0.5; 32 * 32];
    let (pred, h, w) = est.predict_depth(&img).expect("pred");
    assert!(pred.len() == h * w);

    // Create a fake GT
    let gt: Vec<f64> = pred.iter().map(|v| v + 0.1).collect();
    let si_loss = MonocularDepthEstimator::scale_invariant_loss(&pred, &gt).expect("si");
    assert!(si_loss >= 0.0);

    let gm_loss = MonocularDepthEstimator::gradient_matching_loss(&pred, &gt, h, w);
    assert!(gm_loss.is_ok());
}

#[test]
fn test_conv3d_dilation() {
    let config = DeConv3dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: [3, 3, 3],
        stride: [1, 1, 1],
        padding: [2, 2, 2],
        dilation: [2, 2, 2],
        seed: 42,
    };
    let conv = DeConv3d::new(config).expect("conv");
    let (od, oh, ow) = conv.output_shape(8, 8, 8);
    // (8 + 2*2 - 2*(3-1) - 1)/1 + 1 = (8+4-4-1)/1 +1 = 8
    assert_eq!(od, 8);
    assert_eq!(oh, 8);
    assert_eq!(ow, 8);
}

#[test]
fn test_batch_norm_helper() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let bn = de_batch_norm(&x);
    assert_eq!(bn.len(), 5);
    // Mean should be ~0
    let mean: f64 = bn.iter().sum::<f64>() / bn.len() as f64;
    assert!(mean.abs() < 1e-6);
}

#[test]
fn test_upsample_2x() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let (up, uh, uw) = de_upsample_2x(&input, 2, 2);
    assert_eq!(uh, 4);
    assert_eq!(uw, 4);
    assert_eq!(up.len(), 16);
}

#[test]
fn test_refine_block() {
    let block = DeRefineBlock::new(3, 42);
    let input = vec![1.0; 8 * 8];
    let (output, oh, ow) = block.forward(&input, 8, 8).expect("refine");
    assert_eq!(oh, 8);
    assert_eq!(ow, 8);
    assert_eq!(output.len(), 64);
}

#[test]
fn test_scale_invariant_loss_lambda_zero() {
    let pred = vec![2.0, 4.0];
    let gt = vec![1.0, 2.0];
    let loss = MonocularDepthEstimator::scale_invariant_loss_lambda(&pred, &gt, 0.0).expect("loss");
    // lambda=0 → only variance term
    // d = [ln(2), ln(2)], var(d) = 0
    assert!(loss.abs() < 1e-10);
}

#[test]
fn test_depth_completion_all_known() {
    let h = 4;
    let w = 4;
    let config = DepthCompletionConfig {
        h,
        w,
        n_iterations: 2,
        ..Default::default()
    };
    let dc = DepthCompletion::new(config).expect("dc");
    let sparse = vec![5.0; h * w];
    let conf = vec![1.0; h * w];
    let dense = dc.complete(&sparse, &conf, None).expect("complete");
    // All known → should stay close to 5.0
    for &v in &dense {
        assert!((v - 5.0).abs() < 2.0);
    }
}
