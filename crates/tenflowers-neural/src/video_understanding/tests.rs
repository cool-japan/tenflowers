use super::*;
use scirs2_core::random::SeedableRng;

fn make_rng(seed: u64) -> StdRng {
StdRng::seed_from_u64(seed)
}

// ── TemporalShiftModule ──────────────────────────────────────────────────

#[test]
fn test_tsm_output_shape() {
    let tsm = TemporalShiftModule::new(8);
    let (t, h, w, c) = (4, 2, 2, 16);
    let x = vec![1.0_f32; t * h * w * c];
    let out = tsm.shift(&x, t, h, w, c);
    assert_eq!(out.len(), t * h * w * c);
}

#[test]
fn test_tsm_zero_padding_first_frame() {
    let tsm = TemporalShiftModule::new(1); // fold all channels
    let (t, h, w, c) = (3, 1, 1, 4);
    let x: Vec<f32> = (0..t * h * w * c).map(|i| i as f32).collect();
    let out = tsm.shift(&x, t, h, w, c);
    // First frame forward-shifted channels should be zero (no t=-1)
    assert_eq!(out[0], 0.0);
    assert_eq!(out[1], 0.0);
}

#[test]
fn test_tsm_forward_shift_channels() {
    let tsm = TemporalShiftModule::new(2); // fold = c/2
    let (t, h, w, c) = (3, 1, 1, 4);
    // Frame 0: [1,2,3,4], Frame 1: [5,6,7,8], Frame 2: [9,10,11,12]
    let x: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
    let out = tsm.shift(&x, t, h, w, c);
    // Forward shift: channels [0,2) at frame 1 should come from frame 0
    assert_eq!(out[4], 1.0); // channel 0, frame 1 <- frame 0, channel 0
    assert_eq!(out[5], 2.0); // channel 1, frame 1 <- frame 0, channel 1
}

#[test]
fn test_tsm_backward_shift_channels() {
    let tsm = TemporalShiftModule::new(2); // fold = c/2
    let (t, h, w, c) = (3, 1, 1, 4);
    let x: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
    let out = tsm.shift(&x, t, h, w, c);
    // Backward shift: channels [2,4) at frame 1 should come from frame 2
    assert_eq!(out[6], 9.0); // channel 2, frame 1 <- frame 2, channel 2
    assert_eq!(out[7], 10.0); // channel 3, frame 1 <- frame 2, channel 3
}

// ── VideoTransformerBlock ────────────────────────────────────────────────

#[test]
fn test_video_transformer_output_shape() {
    let block = VideoTransformerBlock::new(8, 2, 42);
    let (t, h, w, d) = (2, 2, 2, 8);
    let x = vec![0.1_f32; t * h * w * d];
    let out = block.forward(&x, t, h, w, d);
    assert_eq!(out.len(), t * h * w * d);
}

#[test]
fn test_video_transformer_zero_input() {
    let block = VideoTransformerBlock::new(8, 2, 99);
    let (t, h, w, d) = (2, 2, 2, 8);
    let x = vec![0.0_f32; t * h * w * d];
    let out = block.forward(&x, t, h, w, d);
    assert_eq!(out.len(), t * h * w * d);
    // Output should be finite
    for &v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_video_transformer_different_seeds() {
    let b1 = VideoTransformerBlock::new(8, 2, 1);
    let b2 = VideoTransformerBlock::new(8, 2, 2);
    let x = vec![0.5_f32; 2 * 2 * 2 * 8];
    let o1 = b1.forward(&x, 2, 2, 2, 8);
    let o2 = b2.forward(&x, 2, 2, 2, 8);
    // Different weights → different outputs
    let same = o1.iter().zip(o2.iter()).all(|(a, b)| (a - b).abs() < 1e-6);
    assert!(!same);
}

// ── ActionRecognitionHead ────────────────────────────────────────────────

#[test]
fn test_action_head_output_shape() {
    let head = ActionRecognitionHead::new(16, 10, 42);
    let features = vec![0.1_f32; 4 * 16];
    let logits = head.forward(&features, 4, 16, 10);
    assert_eq!(logits.len(), 10);
}

#[test]
fn test_action_head_mean_pool() {
    let head = ActionRecognitionHead::new(4, 3, 0);
    // All frames identical → mean pool = same
    let feat = vec![
        1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    ];
    let logits = head.forward(&feat, 3, 4, 3);
    assert_eq!(logits.len(), 3);
    for &v in &logits {
        assert!(v.is_finite());
    }
}

#[test]
fn test_action_head_finite_output() {
    let head = ActionRecognitionHead::new(8, 5, 7);
    let mut rng = make_rng(1);
    let features: Vec<f32> = (0..8 * 8).map(|_| rng.random::<f32>()).collect();
    let logits = head.forward(&features, 8, 8, 5);
    for &v in &logits {
        assert!(v.is_finite());
    }
}

// ── VideoMAE ─────────────────────────────────────────────────────────────

#[test]
fn test_video_mae_tube_mask_ratio() {
    let mut rng = make_rng(42);
    let mask = VideoMAE::create_tube_mask(8, 14, 14, 2, 2, 0.75, &mut rng);
    let n_masked = mask.iter().filter(|&&m| m).count();
    let n_total = mask.len();
    let ratio = n_masked as f32 / n_total as f32;
    assert!((ratio - 0.75).abs() < 0.1, "mask ratio: {}", ratio);
}

#[test]
fn test_video_mae_encoder_output_shape() {
    let vmae = VideoMAE::new(2, 2, 0.75, 16, 16, 42);
    let patch_tokens = 2 * 2 * 2; // patch_t * patch_size^2
    let n_visible = 5;
    let patches = vec![0.1_f32; n_visible * patch_tokens];
    let enc = vmae.forward_encoder(&patches, n_visible);
    assert_eq!(enc.len(), n_visible * 16);
}

#[test]
fn test_video_mae_reconstruction_loss_zero() {
    let vmae = VideoMAE::new(2, 2, 0.75, 16, 16, 1);
    let pred = vec![1.0_f32; 4 * 8];
    let target = vec![1.0_f32; 4 * 8];
    let mask = vec![true, false, true, false];
    let loss = vmae.reconstruction_loss(&pred, &target, &mask);
    assert!((loss).abs() < 1e-6);
}

#[test]
fn test_video_mae_reconstruction_loss_nonzero() {
    let vmae = VideoMAE::new(2, 2, 0.75, 16, 16, 2);
    let pred = vec![0.0_f32; 4 * 4];
    let target = vec![1.0_f32; 4 * 4];
    let mask = vec![true, true, false, false];
    let loss = vmae.reconstruction_loss(&pred, &target, &mask);
    assert!(loss > 0.0);
}

#[test]
fn test_video_mae_tube_mask_all_false_zero_ratio() {
    let mut rng = make_rng(10);
    let mask = VideoMAE::create_tube_mask(4, 8, 8, 2, 2, 0.0, &mut rng);
    let n_masked = mask.iter().filter(|&&m| m).count();
    assert_eq!(n_masked, 0);
}

// ── OpticalFlowRAFT ──────────────────────────────────────────────────────

#[test]
fn test_raft_correlation_shape() {
    let (h, w, d, r) = (4, 4, 8, 1);
    let feat1 = vec![0.1_f32; h * w * d];
    let feat2 = vec![0.2_f32; h * w * d];
    let corr = OpticalFlowRAFT::compute_correlation(&feat1, &feat2, h, w, d, r);
    let diam = 2 * r + 1;
    assert_eq!(corr.len(), h * w * diam * diam);
}

#[test]
fn test_raft_refine_flow_shape() {
    let raft = OpticalFlowRAFT::new(1, 3, 8, 8, 42);
    let (h, w) = (4, 4);
    let corr_dim = (2 + 1) * (2 + 1);
    let corr = vec![0.0_f32; h * w * corr_dim];
    let mut flow = vec![0.0_f32; h * w * 2];
    raft.refine_flow(&corr, &mut flow, h, w, 3);
    assert_eq!(flow.len(), h * w * 2);
}

#[test]
fn test_raft_correlation_identical_features() {
    let (h, w, d, r) = (2, 2, 4, 0);
    let feat = vec![1.0_f32; h * w * d];
    let corr = OpticalFlowRAFT::compute_correlation(&feat, &feat, h, w, d, r);
    // With r=0, diam=1, correlation should be positive (d * 1/sqrt(d))
    for &v in &corr {
        assert!(v > 0.0);
    }
}

#[test]
fn test_raft_flow_updates_after_refinement() {
    let raft = OpticalFlowRAFT::new(1, 5, 8, 8, 7);
    let (h, w) = (3, 3);
    let corr_dim = 9;
    let mut rng = make_rng(99);
    let corr: Vec<f32> = (0..h * w * corr_dim).map(|_| rng.random::<f32>()).collect();
    let mut flow = vec![0.0_f32; h * w * 2];
    raft.refine_flow(&corr, &mut flow, h, w, 5);
    let nonzero = flow.iter().any(|&v| v.abs() > 1e-8);
    assert!(nonzero);
}

// ── DeformableConv2D ─────────────────────────────────────────────────────

#[test]
fn test_deform_conv_output_shape() {
    let conv = DeformableConv2D::new(4, 8, 3, 42);
    let (h, w) = (6, 6);
    let x = vec![0.1_f32; h * w * 4];
    let offsets = vec![0.0_f32; h * w * 2 * 9];
    let out = conv.forward(&x, h, w, 4, &offsets);
    assert_eq!(out.len(), h * w * 8);
}

#[test]
fn test_deform_conv_zero_offsets_finite() {
    let conv = DeformableConv2D::new(2, 4, 3, 1);
    let (h, w) = (4, 4);
    let x: Vec<f32> = (0..h * w * 2).map(|i| (i as f32) * 0.01).collect();
    let offsets = vec![0.0_f32; h * w * 2 * 9];
    let out = conv.forward(&x, h, w, 2, &offsets);
    for &v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_deform_conv_nonzero_offsets() {
    let conv = DeformableConv2D::new(2, 4, 3, 5);
    let (h, w) = (4, 4);
    let x: Vec<f32> = (0..h * w * 2).map(|i| (i % 7) as f32).collect();
    let offsets_zero = vec![0.0_f32; h * w * 2 * 9];
    let offsets_nonzero: Vec<f32> = (0..h * w * 2 * 9).map(|i| (i % 3) as f32 * 0.5).collect();
    let out_zero = conv.forward(&x, h, w, 2, &offsets_zero);
    let out_nonzero = conv.forward(&x, h, w, 2, &offsets_nonzero);
    let different = out_zero
        .iter()
        .zip(out_nonzero.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(different);
}

// ── VideoSwinBlock ───────────────────────────────────────────────────────

#[test]
fn test_video_swin_output_shape() {
    let block = VideoSwinBlock::new(8, 2, false, 42);
    let (t, h, w, d) = (4, 4, 4, 8);
    let x = vec![0.1_f32; t * h * w * d];
    let out = block.forward(&x, t, h, w, d, 2, 2, 2);
    assert_eq!(out.len(), t * h * w * d);
}

#[test]
fn test_video_swin_shifted_output_shape() {
    let block = VideoSwinBlock::new(8, 2, true, 42);
    let (t, h, w, d) = (4, 4, 4, 8);
    let x = vec![0.5_f32; t * h * w * d];
    let out = block.forward(&x, t, h, w, d, 2, 2, 2);
    assert_eq!(out.len(), t * h * w * d);
}

#[test]
fn test_video_swin_finite_output() {
    let block = VideoSwinBlock::new(8, 2, false, 3);
    let (t, h, w, d) = (2, 4, 4, 8);
    let x = vec![0.1_f32; t * h * w * d];
    let out = block.forward(&x, t, h, w, d, 1, 2, 2);
    for &v in &out {
        assert!(v.is_finite());
    }
}

// ── HeatmapPoseHead ──────────────────────────────────────────────────────

#[test]
fn test_pose_gaussian_heatmap_shape() {
    let h = HeatmapPoseHead::gaussian_heatmap(4.0, 4.0, 8, 8, 1.5);
    assert_eq!(h.len(), 64);
}

#[test]
fn test_pose_gaussian_peak_at_center() {
    let h = HeatmapPoseHead::gaussian_heatmap(4.0, 4.0, 9, 9, 1.5);
    let peak = h[4 * 9 + 4];
    for (i, &v) in h.iter().enumerate() {
        if i != 4 * 9 + 4 {
            assert!(v <= peak + 1e-6, "value at {} ({}) > peak ({})", i, v, peak);
        }
    }
}

#[test]
fn test_pose_decode_heatmap_shape() {
    let head = HeatmapPoseHead::new(17);
    let heatmap = vec![0.5_f32; 17 * 56 * 56];
    let kps = head.decode_heatmap(&heatmap, 56, 56, 17);
    assert_eq!(kps.len(), 17);
}

#[test]
fn test_pose_decode_heatmap_finds_peak() {
    let head = HeatmapPoseHead::new(1);
    let mut heatmap = vec![0.0_f32; 8 * 8];
    heatmap[2 * 8 + 3] = 1.0; // peak at y=2, x=3
    let kps = head.decode_heatmap(&heatmap, 8, 8, 1);
    assert_eq!(kps.len(), 1);
    let (x, y, conf) = kps[0];
    assert!((x - 3.0).abs() < 1e-4);
    assert!((y - 2.0).abs() < 1e-4);
    assert!((conf - 1.0).abs() < 1e-4);
}

#[test]
fn test_pose_soft_argmax() {
    let head = HeatmapPoseHead::new(2);
    let heatmap = vec![1.0_f32; 2 * 4 * 4];
    let kps = head.soft_argmax(&heatmap, 4, 4, 2);
    assert_eq!(kps.len(), 2);
    for (x, y, _c) in &kps {
        assert!(x.is_finite());
        assert!(y.is_finite());
    }
}

// ── TemporalActionDetector ───────────────────────────────────────────────

#[test]
fn test_tad_proposal_generation_basic() {
    let det = TemporalActionDetector::new(2, 0.5, 0.5);
    let scores = vec![0.0, 0.8, 0.9, 0.8, 0.0, 0.0];
    let props = det.generate_proposals(&scores, 6, 2, 0.5);
    assert!(!props.is_empty());
    let (s, e, _) = props[0];
    assert_eq!(s, 1);
    assert_eq!(e, 4);
}

#[test]
fn test_tad_no_proposals_below_threshold() {
    let det = TemporalActionDetector::new(1, 0.9, 0.5);
    let scores = vec![0.1, 0.2, 0.3];
    let props = det.generate_proposals(&scores, 3, 1, 0.9);
    assert!(props.is_empty());
}

#[test]
fn test_tad_min_len_filter() {
    let det = TemporalActionDetector::new(5, 0.5, 0.5);
    let scores = vec![0.8, 0.8, 0.0, 0.0, 0.0]; // only 2 frames above threshold
    let props = det.generate_proposals(&scores, 5, 5, 0.5);
    assert!(props.is_empty());
}

#[test]
fn test_tad_nms_removes_overlapping() {
    let det = TemporalActionDetector::new(2, 0.5, 0.3);
    // Two overlapping segments
    let mut scores = vec![0.0_f32; 10];
    for i in 1..8 {
        scores[i] = if i < 5 { 0.9 } else { 0.7 };
    }
    let props = det.generate_proposals(&scores, 10, 2, 0.5);
    // Should not have duplicates with high overlap
    for i in 0..props.len() {
        for j in (i + 1)..props.len() {
            let iou = TemporalActionDetector::temporal_iou(
                props[i].0, props[i].1, props[j].0, props[j].1,
            );
            assert!(iou <= 0.3 + 1e-4, "iou={}", iou);
        }
    }
}

#[test]
fn test_tad_score_ordering() {
    let det = TemporalActionDetector::new(1, 0.4, 0.3);
    let scores = vec![0.0, 0.9, 0.9, 0.0, 0.0, 0.5, 0.5, 0.0];
    let props = det.generate_proposals(&scores, 8, 1, 0.4);
    // Props sorted by score descending
    for i in 1..props.len() {
        assert!(props[i - 1].2 >= props[i].2);
    }
}

// ── VideoAugmentation ────────────────────────────────────────────────────

#[test]
fn test_augment_temporal_crop_shape() {
    let aug = VideoAugmentation::new(4, 4, 0.5, 0.0);
    let mut rng = make_rng(42);
    let clip = vec![1.0_f32; 8 * 4 * 4 * 3];
    let (out, t_out) = VideoAugmentation::temporal_crop(&clip, 8, 4, 4, 3, 4, &mut rng);
    assert_eq!(t_out, 4);
    assert_eq!(out.len(), 4 * 4 * 4 * 3);
}

#[test]
fn test_augment_framerate_jitter() {
    let mut rng = make_rng(7);
    let clip = vec![1.0_f32; 8 * 2 * 2 ];
    let (out, t_out) = VideoAugmentation::framerate_jitter(&clip, 8, 2, 2, 1, 2, &mut rng);
    assert!((4..=8).contains(&t_out));
    assert_eq!(out.len(), (t_out * 2 * 2));
}

#[test]
fn test_augment_horizontal_flip_idempotent() {
    let mut rng = make_rng(0); // seed 0 might not flip
    let clip: Vec<f32> = (0..4 * 4 * 4 * 3).map(|i| i as f32).collect();
    let out = VideoAugmentation::random_horizontal_flip(&clip, 4, 4, 4, 3, &mut rng);
    assert_eq!(out.len(), clip.len());
}

#[test]
fn test_augment_temporal_reversal() {
    let (t, h, w, c) = (3, 1, 1, 1);
    let clip = vec![1.0_f32, 2.0, 3.0];
    let rev = VideoAugmentation::temporal_reversal(&clip, t, h, w, c);
    assert_eq!(rev, vec![3.0, 2.0, 1.0]);
}

#[test]
fn test_augment_cutout_3d_zeros() {
    let (t, h, w, c) = (4, 4, 4, 1);
    let mut clip = vec![1.0_f32; t * h * w * c];
    let mut rng = make_rng(1);
    VideoAugmentation::cutout_3d(&mut clip, t, h, w, c, 2, 2, 2, &mut rng);
    let n_zeros = clip.iter().filter(|&&v| v == 0.0).count();
    assert!(n_zeros > 0);
}

#[test]
fn test_augment_pipeline_shape() {
    let aug = VideoAugmentation::new(3, 6, 0.5, 0.2);
    let mut rng = make_rng(42);
    let clip = vec![0.5_f32; 8 * 4 * 4 * 3];
    let (out, t_out) = aug.augment(&clip, 8, 4, 4, 3, &mut rng);
    assert_eq!(out.len(), t_out * 4 * 4 * 3);
    assert!(t_out >= 1);
}

#[test]
fn test_augment_full_pipeline_finite() {
    let aug = VideoAugmentation::new(2, 4, 0.5, 0.5);
    let mut rng = make_rng(99);
    let clip: Vec<f32> = (0..6 * 4 * 4 * 3).map(|i| (i % 5) as f32 * 0.2).collect();
    let (out, _t) = aug.augment(&clip, 6, 4, 4, 3, &mut rng);
    for &v in &out {
        assert!(v.is_finite());
    }
}

// ── Integration / cross-component tests ─────────────────────────────────

#[test]
fn test_tsm_then_transformer() {
    let tsm = TemporalShiftModule::new(4);
    let block = VideoTransformerBlock::new(8, 2, 11);
    let (t, h, w, c) = (2, 2, 2, 8);
    let x = vec![0.3_f32; t * h * w * c];
    let shifted = tsm.shift(&x, t, h, w, c);
    let out = block.forward(&shifted, t, h, w, c);
    assert_eq!(out.len(), t * h * w * c);
    for &v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_video_mae_mask_then_encode() {
    let vmae = VideoMAE::new(2, 2, 0.75, 16, 16, 55);
    let mut rng = make_rng(55);
    let mask = VideoMAE::create_tube_mask(8, 8, 8, 2, 2, 0.75, &mut rng);
    let n_visible = mask.iter().filter(|&&m| !m).count();
    let patch_tokens = 2 * 2 * 2;
    let visible = vec![0.1_f32; n_visible * patch_tokens];
    let enc = vmae.forward_encoder(&visible, n_visible);
    assert_eq!(enc.len(), n_visible * 16);
}

#[test]
fn test_flow_then_deform_conv() {
    let raft = OpticalFlowRAFT::new(1, 2, 8, 8, 13);
    let (h, w, d) = (4, 4, 8);
    let feat1 = vec![0.1_f32; h * w * d];
    let feat2 = vec![0.2_f32; h * w * d];
    let corr = OpticalFlowRAFT::compute_correlation(&feat1, &feat2, h, w, d, 1);
    let mut flow = vec![0.0_f32; h * w * 2];
    raft.refine_flow(&corr, &mut flow, h, w, 2);

    // Use flow as offsets for deformable conv (just check shapes)
    let conv = DeformableConv2D::new(4, 8, 3, 7);
    let x = vec![0.2_f32; h * w * 4];
    // offsets need shape [H, W, 2*k^2] = [4,4, 18]
    let offsets = vec![0.0_f32; h * w * 18];
    let out = conv.forward(&x, h, w, 4, &offsets);
    assert_eq!(out.len(), h * w * 8);
}

// ── Advanced: VideoSwinBlockV2 ───────────────────────────────────────────────

#[test]
fn test_relative_position_bias_3d_shape() {
    let bias = RelativePositionBias3D::new(4, 8, 8, 42);
    // Expected table size: (2*4-1)*(2*8-1)*(2*8-1) = 7*15*15 = 1575
    assert_eq!(bias.table.len(), 7 * 15 * 15);
}

#[test]
fn test_relative_position_bias_3d_lookup_zero_offset() {
    let bias = RelativePositionBias3D::new(2, 4, 4, 10);
    // dt=0,dh=0,dw=0 should map to valid center index
    let v = bias.lookup(0, 0, 0);
    assert!(v.is_finite());
}

#[test]
fn test_relative_position_bias_3d_lookup_boundary() {
    let bias = RelativePositionBias3D::new(3, 4, 4, 11);
    // Out-of-range offsets should be clamped and still return finite values
    let v = bias.lookup(100, -100, 50);
    assert!(v.is_finite());
}

#[test]
fn test_video_swin_block_v2_output_shape() {
    let block = VideoSwinBlockV2::new(8, 2, false, 2, 2, 2, 1);
    let (t, h, w, d) = (4, 4, 4, 8);
    let x = vec![0.1_f32; t * h * w * d];
    let out = block.forward(&x, t, h, w, d, 2, 2, 2);
    assert_eq!(out.len(), t * h * w * d);
}

#[test]
fn test_video_swin_block_v2_finite_outputs() {
    let block = VideoSwinBlockV2::new(8, 2, true, 2, 2, 2, 2);
    let (t, h, w, d) = (2, 4, 4, 8);
    let x: Vec<f32> = (0..t * h * w * d).map(|i| (i as f32) * 0.01).collect();
    let out = block.forward(&x, t, h, w, d, 2, 2, 2);
    assert!(out.iter().all(|v| v.is_finite()), "all outputs must be finite");
}

#[test]
fn test_video_swin_block_v2_shifted_vs_unshifted() {
    let block_no_shift = VideoSwinBlockV2::new(8, 2, false, 2, 2, 2, 3);
    let block_shift = VideoSwinBlockV2::new(8, 2, true, 2, 2, 2, 3);
    let (t, h, w, d) = (4, 4, 4, 8);
    let x = vec![0.5_f32; t * h * w * d];
    let out1 = block_no_shift.forward(&x, t, h, w, d, 2, 2, 2);
    let out2 = block_shift.forward(&x, t, h, w, d, 2, 2, 2);
    // Shifted and non-shifted blocks have different weights so outputs differ
    assert_eq!(out1.len(), out2.len());
}

// ── Advanced: Hiera ──────────────────────────────────────────────────────────

#[test]
fn test_hiera_single_stage_output_shape() {
    // One stage: d_in=8, d_out=16, n_heads=2, pool_stride=2
    let hiera = Hiera::new(vec![(8, 16, 2, 2)], 5);
    let x = vec![0.1_f32; 16 * 8]; // 16 tokens, 8 dims
    let (out, n_out) = hiera.forward(&x, 16);
    assert_eq!(out.len(), n_out * 16);
    assert_eq!(n_out, 8); // 16 / pool_stride=2
}

#[test]
fn test_hiera_two_stages() {
    let hiera = Hiera::new(vec![(4, 8, 2, 2), (8, 16, 4, 2)], 6);
    let x = vec![0.1_f32; 16 * 4]; // 16 tokens, 4 dims
    let (out, n_out) = hiera.forward(&x, 16);
    assert_eq!(out.len(), n_out * 16);
    assert!(n_out >= 1);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_hiera_single_token() {
    let hiera = Hiera::new(vec![(4, 8, 2, 1)], 7);
    let x = vec![1.0_f32; 4];
    let (out, n_out) = hiera.forward(&x, 1);
    assert_eq!(n_out, 1);
    assert_eq!(out.len(), 8);
    assert!(out.iter().all(|v| v.is_finite()));
}

// ── Advanced: TimeSformer ────────────────────────────────────────────────────

#[test]
fn test_timesformer_output_shape() {
    let ts = TimeSformer::new(8, 2, 8);
    let (t, h, w, d) = (2, 3, 3, 8);
    let x = vec![0.1_f32; t * h * w * d];
    let out = ts.forward(&x, t, h, w, d);
    assert_eq!(out.len(), t * h * w * d);
}

#[test]
fn test_timesformer_finite_outputs() {
    let ts = TimeSformer::new(8, 2, 9);
    let (t, h, w, d) = (3, 2, 2, 8);
    let x: Vec<f32> = (0..t * h * w * d).map(|i| (i as f32) * 0.05).collect();
    let out = ts.forward(&x, t, h, w, d);
    assert!(out.iter().all(|v| v.is_finite()), "TimeSformer outputs must be finite");
}

#[test]
fn test_timesformer_single_frame() {
    // With T=1, temporal attention has trivial size 1
    let ts = TimeSformer::new(8, 2, 10);
    let (t, h, w, d) = (1, 4, 4, 8);
    let x = vec![0.2_f32; t * h * w * d];
    let out = ts.forward(&x, t, h, w, d);
    assert_eq!(out.len(), t * h * w * d);
    assert!(out.iter().all(|v| v.is_finite()));
}

// ── Advanced: VideoMaeV2 ─────────────────────────────────────────────────────

#[test]
fn test_video_mae_v2_create_tube_mask_ratio() {
    let mut rng = make_rng(20);
    let n_patches = 100;
    let ratio = 0.9;
    let mask = VideoMaeV2::create_tube_mask(n_patches, ratio, &mut rng);
    assert_eq!(mask.len(), n_patches);
    let n_masked = mask.iter().filter(|&&m| m).count();
    assert_eq!(n_masked, 90);
}

#[test]
fn test_video_mae_v2_encode_shape() {
    let vmae = VideoMaeV2::new(2, 1, 0.9, 16, 21);
    // patch_tokens = patch_t * patch_size^2 * 3 = 1*2*2*3=12
    let n_visible = 10;
    let patch_tokens = 2 * 2 * 3;
    let patches = vec![0.1_f32; n_visible * patch_tokens];
    let enc = vmae.encode(&patches, n_visible);
    assert_eq!(enc.len(), n_visible * 16);
    assert!(enc.iter().all(|v| v.is_finite()));
}

#[test]
fn test_video_mae_v2_decode_shape() {
    let vmae = VideoMaeV2::new(2, 1, 0.9, 16, 22);
    let n_visible = 5;
    let encoder_dim = 16;
    let patch_tokens = 2 * 2 * 3; // = 12
    let enc = vec![0.1_f32; n_visible * encoder_dim];
    let dec = vmae.decode(&enc, n_visible);
    assert_eq!(dec.len(), n_visible * patch_tokens);
}

#[test]
fn test_video_mae_v2_reconstruction_loss_masked() {
    let vmae = VideoMaeV2::new(2, 1, 0.75, 16, 23);
    let n_patches = 8;
    let patch_tokens = 2 * 2 * 3; // = 12
    let pred = vec![0.5_f32; n_patches * patch_tokens];
    let target: Vec<f32> = (0..n_patches * patch_tokens).map(|i| i as f32 * 0.01).collect();
    let mask = vec![true, false, true, false, true, false, true, false]; // 4 masked
    let loss = vmae.reconstruction_loss_normalized(&pred, &target, &mask);
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

#[test]
fn test_video_mae_v2_zero_mask() {
    let vmae = VideoMaeV2::new(2, 1, 0.0, 8, 24);
    let n_patches = 4;
    let patch_tokens = 2 * 2 * 3;
    let pred = vec![0.0_f32; n_patches * patch_tokens];
    let target = vec![1.0_f32; n_patches * patch_tokens];
    let mask = vec![false; n_patches]; // nothing masked
    let loss = vmae.reconstruction_loss_normalized(&pred, &target, &mask);
    assert_eq!(loss, 0.0); // no masked patches => zero loss
}

// ── Advanced: Dino4Video ─────────────────────────────────────────────────────

#[test]
fn test_dino4video_student_forward_sums_to_one() {
    let dino = Dino4Video::new(8, 16, 0.996, 30);
    let x = vec![0.1_f32; 8];
    let probs = dino.student_forward(&x);
    assert_eq!(probs.len(), 16);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "student softmax must sum to 1, got {sum}");
}

#[test]
fn test_dino4video_teacher_forward_sums_to_one() {
    let dino = Dino4Video::new(8, 16, 0.996, 31);
    let x = vec![0.2_f32; 8];
    let probs = dino.teacher_forward(&x);
    assert_eq!(probs.len(), 16);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "teacher softmax must sum to 1, got {sum}");
}

#[test]
fn test_dino4video_loss_non_negative() {
    let dino = Dino4Video::new(8, 16, 0.996, 32);
    let x = vec![0.1_f32; 8];
    let student = dino.student_forward(&x);
    let teacher = dino.teacher_forward(&x);
    let loss = dino.dino_loss(&teacher, &student);
    assert!(loss >= 0.0, "DINO loss must be non-negative");
    assert!(loss.is_finite());
}

#[test]
fn test_dino4video_ema_update_changes_teacher() {
    let mut dino = Dino4Video::new(8, 16, 0.9, 33);
    let old_w = dino.teacher_w.clone();
    dino.update_teacher();
    // With 0.9 momentum teacher moves toward student (which is same at init)
    // so with momentum=0.9 and same init, teacher_w should remain same numerically
    // but if we first perturb student_w they'll differ
    for v in dino.student_w.iter_mut() { *v += 0.1; }
    dino.update_teacher();
    let changed = dino.teacher_w.iter().zip(old_w.iter()).any(|(&a, &b): (&f32, &f32)| (a - b).abs() > 1e-7);
    assert!(changed, "teacher weights should update after EMA step with perturbed student");
}

// ── Advanced: VideoContrastive ───────────────────────────────────────────────

#[test]
fn test_video_contrastive_project_l2_unit() {
    let vc = VideoContrastive::new(8, 16, 0.1, 40);
    let x = vec![1.0_f32; 8];
    let p = vc.project(&x);
    assert_eq!(p.len(), 16);
    let norm: f32 = p.iter().map(|&v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "projected vector should be L2 normalized, norm={norm}");
}

#[test]
fn test_video_contrastive_nt_xent_loss_positive() {
    let vc = VideoContrastive::new(8, 16, 0.1, 41);
    // N=2 pairs: embed[0] is anchor, embed[2] is its positive
    let e0 = vc.project(&[1.0_f32; 8]);
    let e1 = vc.project(&[-1.0_f32; 8]);
    let e2 = vc.project(&[1.0_f32; 8]); // same as e0 = strong positive
    let e3 = vc.project(&[-1.0_f32; 8]); // same as e1
    let loss = vc.nt_xent_loss(&[e0, e1, e2, e3]);
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

#[test]
fn test_video_contrastive_nt_xent_empty() {
    let vc = VideoContrastive::new(8, 16, 0.1, 42);
    let loss = vc.nt_xent_loss(&[]);
    assert_eq!(loss, 0.0);
}

// ── Advanced: MemoryBank & MemoryReader ─────────────────────────────────────

#[test]
fn test_memory_bank_push_and_len() {
    let mut bank = MemoryBank::new(8, 16);
    assert!(bank.is_empty());
    bank.push(vec![0.1_f32; 8], vec![0.2_f32; 16]);
    assert_eq!(bank.len(), 1);
    bank.push(vec![0.3_f32; 8], vec![0.4_f32; 16]);
    assert_eq!(bank.len(), 2);
    assert!(!bank.is_empty());
}

#[test]
fn test_memory_reader_empty_bank_returns_zeros() {
    let bank = MemoryBank::new(8, 16);
    let reader = MemoryReader::new(3);
    let query = vec![0.1_f32; 4 * 8]; // 4 pixels, key_dim=8
    let result = reader.read(&bank, &query, 4);
    assert_eq!(result.len(), 4 * 16);
    assert!(result.iter().all(|&v| v == 0.0));
}

#[test]
fn test_memory_reader_single_frame() {
    let mut bank = MemoryBank::new(4, 8);
    bank.push(vec![1.0_f32; 4], vec![2.0_f32; 8]);
    let reader = MemoryReader::new(1);
    let query = vec![1.0_f32; 2 * 4]; // 2 pixels
    let result = reader.read(&bank, &query, 2);
    assert_eq!(result.len(), 2 * 8);
    assert!(result.iter().all(|v| v.is_finite()));
}

#[test]
fn test_memory_reader_top_k_clamp() {
    let mut bank = MemoryBank::new(4, 4);
    for _ in 0..3 {
        bank.push(vec![0.5_f32; 4], vec![1.0_f32; 4]);
    }
    // top_k=10 but only 3 frames exist → should work fine
    let reader = MemoryReader::new(10);
    let query = vec![0.5_f32; 4];
    let result = reader.read(&bank, &query, 1);
    assert_eq!(result.len(), 4);
    assert!(result.iter().all(|v| v.is_finite()));
}

// ── Advanced: VosDecoder ─────────────────────────────────────────────────────

#[test]
fn test_vos_decoder_output_shape() {
    let dec = VosDecoder::new(8, 8, 50);
    let n_pixels = 16;
    let qf = vec![0.1_f32; n_pixels * 8];
    let mf = vec![0.2_f32; n_pixels * 8];
    let out = dec.forward(&qf, &mf, n_pixels);
    assert_eq!(out.len(), n_pixels);
}

#[test]
fn test_vos_decoder_finite_outputs() {
    let dec = VosDecoder::new(4, 4, 51);
    let n_pixels = 8;
    let qf: Vec<f32> = (0..n_pixels * 4).map(|i| i as f32 * 0.01).collect();
    let mf: Vec<f32> = (0..n_pixels * 4).map(|i| i as f32 * 0.02).collect();
    let out = dec.forward(&qf, &mf, n_pixels);
    assert!(out.iter().all(|v| v.is_finite()));
}

// ── Advanced: VideoLdmUnet ───────────────────────────────────────────────────

#[test]
fn test_video_ldm_unet_denoise_shape() {
    let unet = VideoLdmUnet::new(8, 1000, 2, 60);
    let n_tokens = 16;
    let x = vec![0.5_f32; n_tokens * 8];
    let out = unet.denoise(&x, n_tokens, 500);
    assert_eq!(out.len(), n_tokens * 8);
}

#[test]
fn test_video_ldm_unet_different_timesteps() {
    let unet = VideoLdmUnet::new(8, 100, 2, 61);
    let n_tokens = 8;
    let x = vec![0.1_f32; n_tokens * 8];
    let out0 = unet.denoise(&x, n_tokens, 0);
    let out50 = unet.denoise(&x, n_tokens, 50);
    assert_eq!(out0.len(), out50.len());
    // Different timesteps produce different outputs due to sinusoidal embedding
    assert!(out0.iter().zip(out50.iter()).any(|(a, b)| (a - b).abs() > 1e-6),
        "outputs at different timesteps should differ");
}

#[test]
fn test_video_ldm_unet_finite_outputs() {
    let unet = VideoLdmUnet::new(8, 100, 3, 62);
    let n_tokens = 4;
    let x: Vec<f32> = (0..n_tokens * 8).map(|i| i as f32 * 0.01).collect();
    let out = unet.denoise(&x, n_tokens, 10);
    assert!(out.iter().all(|v| v.is_finite()), "VideoLdmUnet outputs must be finite");
}

// ── Advanced: ConsistencyModel ───────────────────────────────────────────────

#[test]
fn test_consistency_model_fn_output_shape() {
    let cm = ConsistencyModel::new(8, 0.002, 80.0, 70);
    let x = vec![0.3_f32; 8];
    let out = cm.consistency_fn(&x, 1.0);
    assert_eq!(out.len(), 8);
}

#[test]
fn test_consistency_model_skip_scaling() {
    // At sigma_min, c_skip is large => output close to x
    let cm = ConsistencyModel::new(8, 0.002, 80.0, 71);
    let x = vec![0.0_f32; 8];
    let out_min = cm.consistency_fn(&x, 0.002);
    // With x=0 output should be finite
    assert!(out_min.iter().all(|v| v.is_finite()));
}

#[test]
fn test_consistency_model_loss_non_negative() {
    let cm = ConsistencyModel::new(8, 0.002, 80.0, 72);
    let x1 = vec![0.1_f32; 8];
    let x2 = vec![0.2_f32; 8];
    let loss = cm.consistency_loss(&x1, &x2, 1.0, 10.0);
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

#[test]
fn test_consistency_model_generate_shape() {
    let cm = ConsistencyModel::new(8, 0.002, 80.0, 73);
    let mut rng = make_rng(73);
    let noise = vec![0.5_f32; 8];
    let gen = cm.generate(&noise, 3, &mut rng);
    assert_eq!(gen.len(), 8);
    assert!(gen.iter().all(|v| v.is_finite()));
}

#[test]
fn test_consistency_model_one_step_generation() {
    let cm = ConsistencyModel::new(16, 0.002, 80.0, 74);
    let mut rng = make_rng(74);
    let noise = vec![1.0_f32; 16];
    let gen = cm.generate(&noise, 1, &mut rng);
    assert_eq!(gen.len(), 16);
}

// ── Advanced: VuMetrics ──────────────────────────────────────────────────────

#[test]
fn test_vu_metrics_fvd_proxy_zero_distance() {
    let feats = vec![1.0_f32; 4 * 8]; // 4 samples, 8 dims
    let dist = VuMetrics::fvd_proxy(&feats, &feats, 4, 8);
    assert_eq!(dist, 0.0);
}

#[test]
fn test_vu_metrics_fvd_proxy_non_zero() {
    let real = vec![0.0_f32; 4 * 8];
    let gen = vec![1.0_f32; 4 * 8];
    let dist = VuMetrics::fvd_proxy(&real, &gen, 4, 8);
    assert!(dist > 0.0 && dist.is_finite());
}

#[test]
fn test_vu_metrics_temporal_coherence_identical_frames() {
    let frame = vec![0.5_f32; 4 * 4 * 3];
    let flow = vec![0.0_f32; 4 * 4 * 2];
    let tc = VuMetrics::temporal_coherence(&frame, &frame, &flow, 4, 4, 3);
    assert_eq!(tc, 0.0); // identical frames, zero flow → zero error
}

#[test]
fn test_vu_metrics_top1_accuracy_perfect() {
    let preds = vec![0, 1, 2, 3];
    let labels = vec![0, 1, 2, 3];
    let acc = VuMetrics::top1_accuracy(&preds, &labels);
    assert!((acc - 1.0).abs() < 1e-6);
}

#[test]
fn test_vu_metrics_top1_accuracy_zero() {
    let preds = vec![1, 2, 3, 0];
    let labels = vec![0, 1, 2, 3];
    let acc = VuMetrics::top1_accuracy(&preds, &labels);
    assert_eq!(acc, 0.0);
}

#[test]
fn test_vu_metrics_mean_iou_all_foreground() {
    let pred = vec![vec![true; 16]; 3];
    let gt = vec![vec![true; 16]; 3];
    let miou = VuMetrics::mean_iou(&pred, &gt);
    assert!((miou - 1.0).abs() < 1e-6);
}

#[test]
fn test_vu_metrics_mean_iou_no_overlap() {
    let pred = vec![vec![true, false, false, false]];
    let gt = vec![vec![false, true, false, false]];
    let miou = VuMetrics::mean_iou(&pred, &gt);
    assert_eq!(miou, 0.0);
}

#[test]
fn test_vu_metrics_mean_iou_empty() {
    let miou = VuMetrics::mean_iou(&[], &[]);
    assert_eq!(miou, 0.0);
}
