//! Tests for the data_augmentation module.
//!
//! Covers DaSample, all augmentation operators, DaMixup / DaCutmix / DaMixupBatch,
//! DaAutoAugment, DaRandAugment, DaTrivialAugment, DaAugmentationPipeline,
//! DaDiffAugment, and DaMetrics.

#![allow(clippy::float_cmp)]

use super::*;

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_seed() -> u64 {
    0xDEAD_BEEF_CAFE_1234
}

/// Create a simple HxW ramp sample.
fn ramp_sample(h: usize, w: usize) -> DaSample {
    let data: Vec<f64> = (0..h * w).map(|i| i as f64).collect();
    DaSample::new(data, vec![h, w]).expect("ramp_sample")
}

/// Create a uniform 1D sample.
fn seq_sample(len: usize, val: f64) -> DaSample {
    DaSample::new(vec![val; len], vec![len]).expect("seq_sample")
}

// ── DaSample ─────────────────────────────────────────────────────────────────

#[test]
fn test_da_sample_new_valid() {
    let s = DaSample::new(vec![1.0; 6], vec![2, 3]).expect("valid");
    assert_eq!(s.flat_len(), 6);
    assert_eq!(s.ndim(), 2);
}

#[test]
fn test_da_sample_new_invalid_length() {
    let r = DaSample::new(vec![1.0; 5], vec![2, 3]);
    assert!(r.is_err(), "mismatched length should error");
}

#[test]
fn test_da_sample_new_empty_shape() {
    let r = DaSample::new(vec![], vec![]);
    assert!(r.is_err(), "empty shape should error");
}

#[test]
fn test_da_sample_get_set() {
    let mut s = DaSample::new(vec![0.0; 6], vec![2, 3]).expect("new");
    s.set(&[1, 2], 42.0).expect("set");
    assert!((s.get(&[1, 2]).expect("get") - 42.0).abs() < 1e-12);
}

#[test]
fn test_da_sample_get_out_of_bounds() {
    let s = DaSample::new(vec![0.0; 6], vec![2, 3]).expect("new");
    assert!(s.get(&[5, 0]).is_err());
}

#[test]
fn test_da_sample_flat_len() {
    let s = ramp_sample(4, 5);
    assert_eq!(s.flat_len(), 20);
}

#[test]
fn test_da_sample_clone_sample() {
    let s = ramp_sample(3, 3);
    let c = s.clone_sample();
    assert_eq!(s.data, c.data);
    assert_eq!(s.shape, c.shape);
}

#[test]
fn test_da_sample_ndim() {
    let s = DaSample::new(vec![0.0; 24], vec![2, 3, 4]).expect("new");
    assert_eq!(s.ndim(), 3);
}

// ── DaHorizontalFlip ─────────────────────────────────────────────────────────

#[test]
fn test_hflip_shape_preserved() {
    let s = ramp_sample(4, 6);
    let mut seed = make_seed();
    let aug = DaHorizontalFlip { p: 1.0 };
    let out = aug.apply(&s, &mut seed).expect("hflip");
    assert_eq!(out.shape, s.shape);
    assert_eq!(out.flat_len(), s.flat_len());
}

#[test]
fn test_hflip_data_is_flipped() {
    // 1×4 row: [0, 1, 2, 3] → [3, 2, 1, 0]
    let s = DaSample::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]).expect("new");
    let mut seed = make_seed();
    let aug = DaHorizontalFlip { p: 1.0 };
    let out = aug.apply(&s, &mut seed).expect("apply");
    assert!((out.data[0] - 3.0).abs() < 1e-12);
    assert!((out.data[3] - 0.0).abs() < 1e-12);
}

#[test]
fn test_hflip_p0_never_flips() {
    let s = ramp_sample(4, 4);
    let aug = DaHorizontalFlip { p: 0.0 };
    for i in 0..20 {
        let mut seed = make_seed().wrapping_add(i);
        let out = aug.apply(&s, &mut seed).expect("no flip");
        assert_eq!(out.data, s.data);
    }
}

#[test]
fn test_hflip_p1_always_flips() {
    let s = DaSample::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("new");
    let aug = DaHorizontalFlip { p: 1.0 };
    for i in 0..20 {
        let mut seed = make_seed().wrapping_add(i);
        let out = aug.apply(&s, &mut seed).expect("flip");
        // First row should be [2, 1].
        assert!((out.data[0] - 2.0).abs() < 1e-12);
        assert!((out.data[1] - 1.0).abs() < 1e-12);
    }
}

// ── DaVerticalFlip ───────────────────────────────────────────────────────────

#[test]
fn test_vflip_shape_preserved() {
    let s = ramp_sample(5, 4);
    let aug = DaVerticalFlip { p: 1.0 };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("vflip");
    assert_eq!(out.shape, s.shape);
}

#[test]
fn test_vflip_data_is_flipped() {
    // shape [2, 2]: [[0,1],[2,3]] → [[2,3],[0,1]]
    let s = DaSample::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).expect("new");
    let aug = DaVerticalFlip { p: 1.0 };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("apply");
    assert!((out.data[0] - 2.0).abs() < 1e-12);
    assert!((out.data[2] - 0.0).abs() < 1e-12);
}

// ── DaRandomCrop ─────────────────────────────────────────────────────────────

#[test]
fn test_random_crop_output_size() {
    let s = ramp_sample(16, 16);
    let aug = DaRandomCrop {
        p: 1.0,
        crop_h: 12,
        crop_w: 12,
        pad: 4,
    };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("crop");
    assert_eq!(out.shape, vec![12, 12]);
    assert_eq!(out.flat_len(), 144);
}

#[test]
fn test_random_crop_1d() {
    let s = DaSample::new(vec![1.0; 20], vec![20]).expect("new");
    let aug = DaRandomCrop {
        p: 1.0,
        crop_h: 1,
        crop_w: 14,
        pad: 2,
    };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("crop 1d");
    assert_eq!(out.flat_len(), 14);
}

#[test]
fn test_random_crop_p0() {
    // When p=0, center crop is returned at same shape as crop_h × crop_w.
    let s = ramp_sample(8, 8);
    let aug = DaRandomCrop {
        p: 0.0,
        crop_h: 6,
        crop_w: 6,
        pad: 1,
    };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("p0");
    assert_eq!(out.shape, vec![6, 6]);
}

// ── DaGaussianNoise ──────────────────────────────────────────────────────────

#[test]
fn test_gaussian_noise_data_changes() {
    let s = seq_sample(1000, 5.0);
    let aug = DaGaussianNoise {
        p: 1.0,
        std_range: (0.5, 0.5),
    };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("noise");
    let changed = out
        .data
        .iter()
        .zip(s.data.iter())
        .filter(|(a, b)| (*a - *b).abs() > 1e-12)
        .count();
    assert!(changed > 900, "expected most elements to change");
}

#[test]
fn test_gaussian_noise_variance() {
    let n = 5000;
    let s = seq_sample(n, 0.0);
    let std_target = 0.3;
    let aug = DaGaussianNoise {
        p: 1.0,
        std_range: (std_target, std_target),
    };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("noise");
    let var: f64 = out.data.iter().map(|x| x * x).sum::<f64>() / n as f64;
    let std_est = var.sqrt();
    // Allow ±30% tolerance.
    assert!(
        (std_est - std_target).abs() < 0.3 * std_target + 0.05,
        "std estimate {} differs from target {}",
        std_est,
        std_target
    );
}

#[test]
fn test_gaussian_noise_shape_preserved() {
    let s = ramp_sample(6, 6);
    let aug = DaGaussianNoise {
        p: 1.0,
        std_range: (0.01, 0.1),
    };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("noise");
    assert_eq!(out.shape, s.shape);
}

// ── DaRandomErasing ──────────────────────────────────────────────────────────

#[test]
fn test_random_erasing_some_zeroed() {
    let s = DaSample::new(vec![1.0; 100], vec![10, 10]).expect("new");
    let aug = DaRandomErasing {
        p: 1.0,
        scale: (0.1, 0.3),
        ratio: (0.5, 2.0),
    };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("erasing");
    let zeroed = out.data.iter().filter(|&&v| v.abs() < 1e-12).count();
    assert!(zeroed > 0, "expected some pixels to be zeroed");
}

#[test]
fn test_random_erasing_shape_preserved() {
    let s = ramp_sample(8, 8);
    let aug = DaRandomErasing {
        p: 1.0,
        scale: (0.1, 0.4),
        ratio: (0.3, 3.0),
    };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("erasing");
    assert_eq!(out.shape, s.shape);
}

// ── DaMixup ──────────────────────────────────────────────────────────────────

#[test]
fn test_beta_sample_range() {
    let mut seed = make_seed();
    for _ in 0..200 {
        let lam = DaMixup::beta_sample(0.4, &mut seed);
        assert!((0.0..=1.0).contains(&lam), "beta out of range: {}", lam);
    }
}

#[test]
fn test_mixup_shape_preserved() {
    let s1 = ramp_sample(4, 4);
    let s2 = ramp_sample(4, 4);
    let mix = DaMixup::new(0.4);
    let mut seed = make_seed();
    let out = mix.mix(&s1, &s2, &mut seed).expect("mix");
    assert_eq!(out.shape, s1.shape);
}

#[test]
fn test_mixup_soft_label_sums_to_one() {
    let mut s1 = ramp_sample(4, 4);
    s1.label = Some(0);
    s1.soft_label = Some(vec![1.0, 0.0, 0.0]);
    let mut s2 = ramp_sample(4, 4);
    s2.label = Some(1);
    s2.soft_label = Some(vec![0.0, 1.0, 0.0]);
    let mix = DaMixup::new(0.4);
    let mut seed = make_seed();
    let out = mix.mix(&s1, &s2, &mut seed).expect("mix");
    let sum: f64 = out.soft_label.as_ref().expect("soft label").iter().sum();
    assert!((sum - 1.0).abs() < 1e-10, "soft labels sum to {}", sum);
}

#[test]
fn test_mixup_values_in_convex_range() {
    let s1 = DaSample::new(vec![0.0; 4], vec![4]).expect("new");
    let s2 = DaSample::new(vec![1.0; 4], vec![4]).expect("new");
    let mix = DaMixup::new(0.4);
    let mut seed = make_seed();
    let out = mix.mix(&s1, &s2, &mut seed).expect("mix");
    for v in &out.data {
        assert!(*v >= 0.0 && *v <= 1.0, "value {} not in [0,1]", v);
    }
}

#[test]
fn test_mixup_shape_mismatch_error() {
    let s1 = ramp_sample(4, 4);
    let s2 = ramp_sample(4, 5);
    let mix = DaMixup::new(0.4);
    let mut seed = make_seed();
    assert!(mix.mix(&s1, &s2, &mut seed).is_err());
}

// ── DaCutmix ─────────────────────────────────────────────────────────────────

#[test]
fn test_cutmix_shape_preserved() {
    let s1 = ramp_sample(8, 8);
    let s2 = DaSample::new(vec![99.0; 64], vec![8, 8]).expect("new");
    let cm = DaCutmix::new(1.0);
    let mut seed = make_seed();
    let out = cm.mix(&s1, &s2, &mut seed).expect("cutmix");
    assert_eq!(out.shape, s1.shape);
}

#[test]
fn test_cutmix_some_values_from_s2() {
    // Use alpha=4.0 so Beta(4,4) concentrates lam around 0.5,
    // ensuring cut_ratio = sqrt(1-lam) is well in (0, 1) and neither
    // all of s1 nor all of s2 fills the output.
    let s1 = DaSample::new(vec![0.0; 64], vec![8, 8]).expect("new");
    let s2 = DaSample::new(vec![1.0; 64], vec![8, 8]).expect("new");
    let cm = DaCutmix::new(4.0);
    let mut seed = make_seed();
    let out = cm.mix(&s1, &s2, &mut seed).expect("cutmix");
    let ones = out
        .data
        .iter()
        .filter(|&&v| (v - 1.0).abs() < 1e-12)
        .count();
    assert!(ones > 0, "expected some values from s2");
    let zeros = out.data.iter().filter(|&&v| v.abs() < 1e-12).count();
    assert!(
        zeros > 0,
        "expected some values from s1, ones={} zeros=0",
        ones
    );
}

#[test]
fn test_cutmix_soft_label_sums_to_one() {
    let mut s1 = DaSample::new(vec![0.0; 64], vec![8, 8]).expect("new");
    s1.label = Some(0);
    s1.soft_label = Some(vec![1.0, 0.0]);
    let mut s2 = DaSample::new(vec![1.0; 64], vec![8, 8]).expect("new");
    s2.label = Some(1);
    s2.soft_label = Some(vec![0.0, 1.0]);
    let cm = DaCutmix::new(1.0);
    let mut seed = make_seed();
    let out = cm.mix(&s1, &s2, &mut seed).expect("cutmix");
    let sum: f64 = out.soft_label.as_ref().expect("soft").iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "soft label sum = {}", sum);
}

// ── DaMixupBatch ─────────────────────────────────────────────────────────────

#[test]
fn test_mixup_batch_size() {
    let batch: Vec<DaSample> = (0..4).map(|_| ramp_sample(4, 4)).collect();
    let mb = DaMixupBatch::new(0.4, false);
    let mut seed = make_seed();
    let out = mb.augment_batch(&batch, &mut seed).expect("batch");
    assert_eq!(out.len(), 4);
}

#[test]
fn test_mixup_batch_with_cutmix() {
    let batch: Vec<DaSample> = (0..6).map(|_| ramp_sample(4, 4)).collect();
    let mb = DaMixupBatch::new(1.0, true);
    let mut seed = make_seed();
    let out = mb
        .augment_batch(&batch, &mut seed)
        .expect("batch with cutmix");
    assert_eq!(out.len(), 6);
}

#[test]
fn test_mixup_batch_empty() {
    let mb = DaMixupBatch::new(0.4, false);
    let mut seed = make_seed();
    let out = mb.augment_batch(&[], &mut seed).expect("empty batch");
    assert!(out.is_empty());
}

// ── DaAutoAugment ────────────────────────────────────────────────────────────

#[test]
fn test_autoaugment_policy_count() {
    let aa = DaAutoAugment::imagenet_policy();
    assert_eq!(aa.subpolicies.len(), 25);
}

#[test]
fn test_autoaugment_same_shape() {
    let s = ramp_sample(8, 8);
    let aa = DaAutoAugment::imagenet_policy();
    let mut seed = make_seed();
    let out = aa.apply_policy(&s, &mut seed).expect("autoaugment");
    assert_eq!(out.shape, s.shape);
    assert_eq!(out.flat_len(), s.flat_len());
}

#[test]
fn test_autoaugment_multiple_seeds() {
    let s = ramp_sample(6, 6);
    let aa = DaAutoAugment::imagenet_policy();
    for i in 0..20 {
        let mut seed = make_seed().wrapping_add(i * 97);
        let out = aa.apply_policy(&s, &mut seed).expect("aa");
        assert_eq!(out.shape, s.shape);
    }
}

// ── DaRandAugment ────────────────────────────────────────────────────────────

#[test]
fn test_randaugment_same_shape() {
    let s = ramp_sample(8, 8);
    let ra = DaRandAugment::new(3, 9.0);
    let mut seed = make_seed();
    let out = ra.apply(&s, &mut seed).expect("randaugment");
    assert_eq!(out.shape, s.shape);
}

#[test]
fn test_randaugment_n0() {
    let s = ramp_sample(4, 4);
    let ra = DaRandAugment::new(0, 15.0);
    let mut seed = make_seed();
    let out = ra.apply(&s, &mut seed).expect("n=0");
    assert_eq!(out.shape, s.shape);
}

#[test]
fn test_randaugment_operations_nonempty() {
    let ops = DaRandAugment::operations();
    assert!(!ops.is_empty());
}

// ── DaTrivialAugment ─────────────────────────────────────────────────────────

#[test]
fn test_trivialaugment_same_shape() {
    let s = ramp_sample(8, 8);
    let ta = DaTrivialAugment::new();
    let mut seed = make_seed();
    let out = ta.apply(&s, &mut seed).expect("trivial");
    assert_eq!(out.shape, s.shape);
}

#[test]
fn test_trivialaugment_multiple_runs() {
    let s = ramp_sample(6, 6);
    let ta = DaTrivialAugment::new();
    for i in 0..20 {
        let mut seed = make_seed().wrapping_add(i * 137);
        let out = ta.apply(&s, &mut seed).expect("trivial run");
        assert_eq!(out.shape, s.shape);
    }
}

// ── DaAugmentationPipeline ───────────────────────────────────────────────────

#[test]
fn test_pipeline_empty_preserves_sample() {
    let s = ramp_sample(4, 4);
    let pipe = DaAugmentationPipeline::new();
    let mut seed = make_seed();
    let out = pipe.apply(&s, &mut seed).expect("empty pipeline");
    assert_eq!(out.data, s.data);
}

#[test]
fn test_pipeline_add_apply() {
    let s = ramp_sample(8, 8);
    let mut pipe = DaAugmentationPipeline::new();
    pipe.add(DaHorizontalFlip { p: 1.0 }, 1.0);
    pipe.add(
        DaGaussianNoise {
            p: 1.0,
            std_range: (0.01, 0.05),
        },
        1.0,
    );
    let mut seed = make_seed();
    let out = pipe.apply(&s, &mut seed).expect("pipeline");
    assert_eq!(out.shape, s.shape);
}

#[test]
fn test_pipeline_batch_correct_size() {
    let batch: Vec<DaSample> = (0..8).map(|_| ramp_sample(4, 4)).collect();
    let pipe = DaAugmentationPipeline::from_randaugment(2, 10.0);
    let mut seed = make_seed();
    let out = pipe.augment_batch(&batch, &mut seed).expect("batch");
    assert_eq!(out.len(), 8);
}

#[test]
fn test_pipeline_from_autoaugment() {
    let s = ramp_sample(8, 8);
    let pipe = DaAugmentationPipeline::from_autoaugment();
    let mut seed = make_seed();
    let out = pipe.apply(&s, &mut seed).expect("autoaugment pipeline");
    assert_eq!(out.shape, s.shape);
}

#[test]
fn test_pipeline_p0_passthrough() {
    let s = ramp_sample(4, 4);
    let mut pipe = DaAugmentationPipeline::new();
    pipe.add(DaHorizontalFlip { p: 1.0 }, 0.0); // p_each = 0 → never applied
    let mut seed = make_seed();
    let out = pipe.apply(&s, &mut seed).expect("p0 pipeline");
    assert_eq!(out.data, s.data);
}

// ── DaDiffAugment ─────────────────────────────────────────────────────────────

#[test]
fn test_diffaugment_color_same_shape() {
    let s = ramp_sample(8, 8);
    let da = DaDiffAugment::new(DaDiffPolicy::Color);
    let mut seed = make_seed();
    let out = da.apply(&s, &mut seed).expect("color");
    assert_eq!(out.shape, s.shape);
}

#[test]
fn test_diffaugment_translation_same_shape() {
    let s = ramp_sample(8, 8);
    let da = DaDiffAugment::new(DaDiffPolicy::Translation);
    let mut seed = make_seed();
    let out = da.apply(&s, &mut seed).expect("translation");
    assert_eq!(out.shape, s.shape);
}

#[test]
fn test_diffaugment_cutout_same_shape() {
    let s = ramp_sample(8, 8);
    let da = DaDiffAugment::new(DaDiffPolicy::Cutout);
    let mut seed = make_seed();
    let out = da.apply(&s, &mut seed).expect("cutout");
    assert_eq!(out.shape, s.shape);
}

#[test]
fn test_diffaugment_all_same_shape() {
    let s = ramp_sample(8, 8);
    let da = DaDiffAugment::new(DaDiffPolicy::ColorTranslationCutout);
    let mut seed = make_seed();
    let out = da.apply(&s, &mut seed).expect("all");
    assert_eq!(out.shape, s.shape);
}

#[test]
fn test_diffaugment_color_modifies_data() {
    let s = DaSample::new(vec![0.5; 64], vec![8, 8]).expect("new");
    let da = DaDiffAugment::new(DaDiffPolicy::Color);
    let mut seed = make_seed();
    let out = da.color_augment(&s, &mut seed).expect("color");
    let changed = out
        .data
        .iter()
        .zip(s.data.iter())
        .filter(|(a, b)| (*a - *b).abs() > 1e-12)
        .count();
    assert!(changed > 0, "color augment should change values");
}

// ── DaMetrics ────────────────────────────────────────────────────────────────

#[test]
fn test_metrics_diversity_nonnegative() {
    let orig = ramp_sample(4, 4);
    let mut seed = make_seed();
    let aug = DaGaussianNoise {
        p: 1.0,
        std_range: (0.1, 0.5),
    };
    let augmented: Vec<DaSample> = (0..5)
        .map(|i| {
            let mut s = make_seed().wrapping_add(i * 31);
            aug.apply(&orig, &mut s).expect("noise")
        })
        .collect();
    let div = DaMetrics::augmentation_diversity(&orig, &augmented);
    assert!(div >= 0.0, "diversity must be >= 0, got {}", div);
}

#[test]
fn test_metrics_diversity_zero_for_identity() {
    let orig = ramp_sample(4, 4);
    let same = vec![orig.clone_sample(), orig.clone_sample()];
    let div = DaMetrics::augmentation_diversity(&orig, &same);
    assert!(
        div.abs() < 1e-12,
        "diversity should be 0 for identical samples, got {}",
        div
    );
}

#[test]
fn test_metrics_label_preserving_ratio_all_same() {
    let orig = vec![0usize, 1, 2];
    let aug = vec![0usize, 1, 2];
    let ratio = DaMetrics::label_preserving_ratio(&orig, &aug);
    assert!((ratio - 1.0).abs() < 1e-12);
}

#[test]
fn test_metrics_label_preserving_ratio_none_same() {
    let orig = vec![0usize, 1, 2];
    let aug = vec![1usize, 2, 0];
    let ratio = DaMetrics::label_preserving_ratio(&orig, &aug);
    assert!(ratio.abs() < 1e-12);
}

#[test]
fn test_metrics_strength_nonnegative() {
    let orig = ramp_sample(4, 4);
    let mut seed = make_seed();
    let aug = DaGaussianNoise {
        p: 1.0,
        std_range: (0.05, 0.1),
    };
    let augmented = aug.apply(&orig, &mut seed).expect("noise");
    let strength = DaMetrics::augmentation_strength(&orig, &augmented);
    assert!(strength >= 0.0, "strength = {}", strength);
}

#[test]
fn test_metrics_strength_zero_for_identity() {
    let orig = ramp_sample(4, 4);
    let same = orig.clone_sample();
    let strength = DaMetrics::augmentation_strength(&orig, &same);
    assert!(strength.abs() < 1e-12);
}

#[test]
fn test_metrics_coverage_range() {
    let mut seed = make_seed();
    let aug = DaGaussianNoise {
        p: 1.0,
        std_range: (0.1, 0.5),
    };
    let samples: Vec<DaSample> = (0..20)
        .map(|i| {
            let mut s = make_seed().wrapping_add(i * 17);
            let base = DaSample::new(vec![(i as f64) * 0.1; 16], vec![4, 4]).expect("new");
            aug.apply(&base, &mut s).expect("noise")
        })
        .collect();
    let cov = DaMetrics::coverage_of_space(&samples, 8);
    assert!((0.0..=1.0).contains(&cov), "coverage = {}", cov);
}

#[test]
fn test_metrics_coverage_empty() {
    let cov = DaMetrics::coverage_of_space(&[], 8);
    assert!(cov.abs() < 1e-12);
}

// ── Additional edge-case tests ────────────────────────────────────────────────

#[test]
fn test_brightness_factor_range() {
    let s = DaSample::new(vec![1.0; 16], vec![4, 4]).expect("new");
    let aug = DaBrightness {
        p: 1.0,
        factor_range: (2.0, 2.0),
    };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("brightness");
    // All values should be doubled.
    assert!(out.data.iter().all(|&v| (v - 2.0).abs() < 1e-10));
}

#[test]
fn test_contrast_mean_preserved() {
    let s = DaSample::new((0..16).map(|i| i as f64).collect(), vec![4, 4]).expect("new");
    let mean_orig: f64 = s.data.iter().sum::<f64>() / 16.0;
    let aug = DaContrast {
        p: 1.0,
        factor_range: (0.5, 2.0),
    };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("contrast");
    let mean_out: f64 = out.data.iter().sum::<f64>() / 16.0;
    // (x - mean) * f + mean → mean is preserved.
    assert!(
        (mean_out - mean_orig).abs() < 1e-9,
        "mean changed: {} vs {}",
        mean_orig,
        mean_out
    );
}

#[test]
fn test_gaussian_blur_output_shape() {
    let s = ramp_sample(10, 10);
    let aug = DaGaussianBlur {
        p: 1.0,
        sigma_range: (1.0, 2.0),
        kernel_size: 5,
    };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("blur");
    assert_eq!(out.shape, s.shape);
}

#[test]
fn test_rotation_shape_preserved() {
    let s = ramp_sample(8, 8);
    let aug = DaRandomRotation {
        p: 1.0,
        max_degrees: 45.0,
    };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("rotate");
    assert_eq!(out.shape, s.shape);
}

#[test]
fn test_shear_shape_preserved() {
    let s = ramp_sample(8, 8);
    let aug = DaShear {
        p: 1.0,
        max_shear: 0.3,
    };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("shear");
    assert_eq!(out.shape, s.shape);
}

#[test]
fn test_da_rand01_range() {
    let mut seed = make_seed();
    for _ in 0..1000 {
        let v = da_rand01(&mut seed);
        assert!(v > 0.0 && v < 1.0, "out of (0,1): {}", v);
    }
}

#[test]
fn test_da_randint_range() {
    let mut seed = make_seed();
    for _ in 0..200 {
        let v = da_randint(&mut seed, 3, 10);
        assert!((3..10).contains(&v), "out of [3,10): {}", v);
    }
}

#[test]
fn test_da_randn_reasonable() {
    let mut seed = make_seed();
    let n = 4000usize;
    let vals: Vec<f64> = (0..n).map(|_| da_randn(&mut seed)).collect();
    let mean: f64 = vals.iter().sum::<f64>() / n as f64;
    let variance: f64 = vals.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let std: f64 = variance.sqrt();
    // Mean should be close to 0, std close to 1.
    // Use generous tolerances appropriate for a seeded pseudo-RNG test.
    assert!(mean.abs() < 0.2, "mean = {}", mean);
    assert!((std - 1.0).abs() < 0.25, "std = {}", std);
}

#[test]
fn test_diffaugment_cutout_has_zeros() {
    let s = DaSample::new(vec![1.0; 64], vec![8, 8]).expect("new");
    let da = DaDiffAugment::new(DaDiffPolicy::Cutout);
    let mut seed = make_seed();
    let out = da.cutout_augment(&s, &mut seed).expect("cutout");
    let zeroed = out.data.iter().filter(|&&v| v.abs() < 1e-12).count();
    assert!(zeroed > 0, "cutout should zero some pixels");
}

#[test]
fn test_autoaugment_empty_subpolicies() {
    let aa = DaAutoAugment {
        subpolicies: vec![],
    };
    let s = ramp_sample(4, 4);
    let mut seed = make_seed();
    let out = aa.apply_policy(&s, &mut seed).expect("empty policy");
    assert_eq!(out.data, s.data);
}

#[test]
fn test_hflip_chw_shape() {
    // CHW: [2, 3, 4] → data length = 24
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let s = DaSample::new(data, vec![2, 3, 4]).expect("chw");
    let aug = DaHorizontalFlip { p: 1.0 };
    let mut seed = make_seed();
    let out = aug.apply(&s, &mut seed).expect("hflip chw");
    assert_eq!(out.shape, s.shape);
}
