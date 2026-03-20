//! Tests for augmentation module.

use super::image_ops::*;
use super::sequence_ops::*;
use super::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};
use tenflowers_core::Tensor;

#[test]
fn test_augmentation_config_default() {
    let config = AugmentationConfig::default();
    assert!(config.enabled);
    assert_eq!(config.probability, 0.5);
    assert!(config.seed.is_none());
}

#[test]
fn test_augmentation_config_builder() {
    let config = AugmentationConfig::new()
        .with_enabled(false)
        .with_probability(0.8)
        .with_seed(42)
        .with_parameter("test".to_string(), 1.5);

    assert!(!config.enabled);
    assert_eq!(config.probability, 0.8);
    assert_eq!(config.seed, Some(42));
    assert_eq!(config.get_parameter("test"), Some(1.5));
}

#[test]
fn test_augmentation_config_probability_clamping() {
    let config1 = AugmentationConfig::new().with_probability(1.5);
    assert_eq!(config1.probability, 1.0);

    let config2 = AugmentationConfig::new().with_probability(-0.5);
    assert_eq!(config2.probability, 0.0);
}

#[test]
fn test_image_augmentation_names() {
    assert_eq!(ImageAugmentation::HorizontalFlip.name(), "horizontal_flip");
    assert_eq!(ImageAugmentation::Rotation.name(), "rotation");
    assert_eq!(ImageAugmentation::Mixup.name(), "mixup");
}

#[test]
fn test_sequence_augmentation_names() {
    assert_eq!(SequenceAugmentation::WordDeletion.name(), "word_deletion");
    assert_eq!(SequenceAugmentation::TimeWarping.name(), "time_warping");
}

#[test]
fn test_augmentation_pipeline_creation() {
    let pipeline = AugmentationPipeline::new();
    assert_eq!(pipeline.num_image_augmentations(), 0);
    assert_eq!(pipeline.num_sequence_augmentations(), 0);
    assert_eq!(pipeline.total_augmentations(), 0);
}

#[test]
fn test_augmentation_pipeline_add_image() {
    let pipeline = AugmentationPipeline::new().add_image_augmentation(
        ImageAugmentation::HorizontalFlip,
        AugmentationConfig::default(),
    );

    assert_eq!(pipeline.num_image_augmentations(), 1);
    assert_eq!(pipeline.total_augmentations(), 1);
}

#[test]
fn test_augmentation_pipeline_add_sequence() {
    let pipeline = AugmentationPipeline::new().add_sequence_augmentation(
        SequenceAugmentation::WordSwap,
        AugmentationConfig::default(),
    );

    assert_eq!(pipeline.num_sequence_augmentations(), 1);
    assert_eq!(pipeline.total_augmentations(), 1);
}

#[test]
fn test_augmentation_pipeline_multiple() {
    let pipeline = AugmentationPipeline::new()
        .add_image_augmentation(
            ImageAugmentation::HorizontalFlip,
            AugmentationConfig::default(),
        )
        .add_image_augmentation(ImageAugmentation::Rotation, AugmentationConfig::default())
        .add_sequence_augmentation(
            SequenceAugmentation::WordSwap,
            AugmentationConfig::default(),
        );

    assert_eq!(pipeline.num_image_augmentations(), 2);
    assert_eq!(pipeline.num_sequence_augmentations(), 1);
    assert_eq!(pipeline.total_augmentations(), 3);
}

#[test]
fn test_preset_standard_image_classification() {
    let pipeline = presets::standard_image_classification();
    assert!(pipeline.num_image_augmentations() > 0);
}

#[test]
fn test_preset_aggressive_image_augmentation() {
    let pipeline = presets::aggressive_image_augmentation();
    assert!(pipeline.num_image_augmentations() > 4);
}

#[test]
fn test_preset_modern_image_augmentation() {
    let pipeline = presets::modern_image_augmentation();
    assert!(pipeline.num_image_augmentations() > 0);
}

#[test]
fn test_preset_standard_text_augmentation() {
    let pipeline = presets::standard_text_augmentation();
    assert!(pipeline.num_sequence_augmentations() > 0);
}

#[test]
fn test_preset_standard_audio_augmentation() {
    let pipeline = presets::standard_audio_augmentation();
    assert!(pipeline.num_sequence_augmentations() > 0);
}

#[test]
fn test_augmentation_stats_creation() {
    let stats = AugmentationStats::new();
    assert_eq!(stats.total_applied, 0);
    assert_eq!(stats.avg_time_ms, 0.0);
}

#[test]
fn test_augmentation_stats_record() {
    let mut stats = AugmentationStats::new();

    stats.record("flip", 1.0);
    stats.record("rotation", 2.0);
    stats.record("flip", 3.0);

    assert_eq!(stats.total_applied, 3);
    assert_eq!(stats.get_count("flip"), 2);
    assert_eq!(stats.get_count("rotation"), 1);
    assert_eq!(stats.avg_time_ms, 2.0);
}

#[test]
fn test_augmentation_stats_most_used() {
    let mut stats = AugmentationStats::new();

    stats.record("flip", 1.0);
    stats.record("flip", 1.0);
    stats.record("flip", 1.0);
    stats.record("rotation", 1.0);

    let most_used = stats.most_used();
    assert!(most_used.is_some());
    let (name, count) = most_used.expect("test: operation should succeed");
    assert_eq!(name, "flip");
    assert_eq!(count, 3);
}

#[test]
fn test_augmentation_stats_least_used() {
    let mut stats = AugmentationStats::new();

    stats.record("flip", 1.0);
    stats.record("flip", 1.0);
    stats.record("flip", 1.0);
    stats.record("rotation", 1.0);

    let least_used = stats.least_used();
    assert!(least_used.is_some());
    let (name, count) = least_used.expect("test: operation should succeed");
    assert_eq!(name, "rotation");
    assert_eq!(count, 1);
}

// ----- Tests for actual augmentation logic -----

fn make_test_image() -> Tensor<f64> {
    let data: Vec<f64> = (0..16).map(|i| i as f64 / 15.0).collect();
    Tensor::from_vec(data, &[1, 4, 4]).expect("test: create test image")
}

fn make_test_batch() -> Tensor<f64> {
    let data: Vec<f64> = (0..32).map(|i| i as f64 / 31.0).collect();
    Tensor::from_vec(data, &[2, 1, 4, 4]).expect("test: create test batch")
}

fn make_test_rgb_image() -> Tensor<f64> {
    let data: Vec<f64> = (0..48).map(|i| (i as f64) / 47.0).collect();
    Tensor::from_vec(data, &[3, 4, 4]).expect("test: create test rgb image")
}

fn make_test_sequence() -> Tensor<f64> {
    let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
    Tensor::from_vec(data, &[8]).expect("test: create test sequence")
}

fn make_test_spectrogram() -> Tensor<f64> {
    let data: Vec<f64> = (0..200).map(|i| i as f64 / 199.0).collect();
    Tensor::from_vec(data, &[10, 20]).expect("test: create test spectrogram")
}

#[test]
fn test_horizontal_flip_identity() {
    let img = make_test_image();
    let dims = img.shape().dims();
    let flipped = apply_horizontal_flip(&img, dims, 4, 4).expect("test: horizontal flip");
    let double = apply_horizontal_flip(&flipped, flipped.shape().dims(), 4, 4)
        .expect("test: double horizontal flip");
    let orig = img.data();
    let restored = double.data();
    for i in 0..orig.len() {
        assert!(
            (orig[i] - restored[i]).abs() < 1e-10,
            "Mismatch at {i}: {} vs {}",
            orig[i],
            restored[i]
        );
    }
}

#[test]
fn test_vertical_flip_identity() {
    let img = make_test_image();
    let dims = img.shape().dims();
    let flipped = apply_vertical_flip(&img, dims, 4, 4).expect("test: vertical flip");
    let double = apply_vertical_flip(&flipped, flipped.shape().dims(), 4, 4)
        .expect("test: double vertical flip");
    let orig = img.data();
    let restored = double.data();
    for i in 0..orig.len() {
        assert!((orig[i] - restored[i]).abs() < 1e-10, "Mismatch at {i}");
    }
}

#[test]
fn test_horizontal_flip_values() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let img = Tensor::<f64>::from_vec(data, &[1, 2, 3]).expect("test: create");
    let flipped = apply_horizontal_flip(&img, img.shape().dims(), 2, 3).expect("test: flip");
    let out = flipped.data();
    assert_eq!(out[0], 3.0);
    assert_eq!(out[1], 2.0);
    assert_eq!(out[2], 1.0);
    assert_eq!(out[3], 6.0);
    assert_eq!(out[4], 5.0);
    assert_eq!(out[5], 4.0);
}

#[test]
fn test_rotation_zero_angle() {
    let img = make_test_image();
    let dims = img.shape().dims();
    let rotated = apply_rotation(&img, dims, 4, 4, 0.0).expect("test: zero rotation");
    let orig = img.data();
    let out = rotated.data();
    for i in 0..orig.len() {
        assert!(
            (orig[i] - out[i]).abs() < 1e-6,
            "Zero rotation changed value at {i}"
        );
    }
}

#[test]
fn test_brightness_offset() {
    let img = make_test_image();
    let bright = apply_brightness(&img, 0.1).expect("test: brightness");
    let orig = img.data();
    let out = bright.data();
    for i in 0..orig.len() {
        assert!(
            (out[i] - orig[i] - 0.1).abs() < 1e-10,
            "Brightness not applied correctly at {i}"
        );
    }
}

#[test]
fn test_gaussian_noise_changes_values() {
    let img = make_test_image();
    let mut rng = StdRng::seed_from_u64(42);
    let noisy = apply_gaussian_noise(&img, 0.1, &mut rng).expect("test: gaussian noise");
    let orig = img.data();
    let out = noisy.data();
    let mut any_different = false;
    for i in 0..orig.len() {
        if (out[i] - orig[i]).abs() > 1e-12 {
            any_different = true;
            break;
        }
    }
    assert!(
        any_different,
        "Gaussian noise should change at least one value"
    );
}

#[test]
fn test_salt_pepper_noise() {
    let img = make_test_image();
    let mut rng = StdRng::seed_from_u64(99);
    let noisy = apply_salt_pepper_noise(&img, 0.5, &mut rng).expect("test: salt pepper");
    let out = noisy.data();
    let salt_pepper_count = out.iter().filter(|&&v| v == 0.0 || v == 1.0).count();
    assert!(salt_pepper_count > 0, "Expected some salt/pepper values");
}

#[test]
fn test_cutout_zeros_region() {
    let img = make_test_image();
    let dims = img.shape().dims();
    let mut rng = StdRng::seed_from_u64(7);
    let cut = apply_cutout(&img, dims, 4, 4, 2, &mut rng).expect("test: cutout");
    let out = cut.data();
    let zero_count = out.iter().filter(|&&v| v == 0.0).count();
    assert!(zero_count > 0, "Cutout should zero at least one pixel");
}

#[test]
fn test_random_erasing() {
    let img = make_test_image();
    let dims = img.shape().dims();
    let mut rng = StdRng::seed_from_u64(3);
    let erased =
        apply_random_erasing(&img, dims, 4, 4, 0.3, &mut rng).expect("test: random erasing");
    let out = erased.data();
    let zero_count = out.iter().filter(|&&v| v == 0.0).count();
    assert!(zero_count >= 1, "Random erasing should zero some pixels");
}

#[test]
fn test_contrast_adjustment() {
    let img = make_test_image();
    let dims = img.shape().dims();
    let out_ident = apply_contrast(&img, dims, 4, 4, 1.0).expect("test: contrast identity");
    let orig = img.data();
    let out = out_ident.data();
    for i in 0..orig.len() {
        assert!(
            (orig[i] - out[i]).abs() < 1e-10,
            "Contrast with alpha=1 should be identity"
        );
    }
}

#[test]
fn test_saturation_rgb() {
    let img = make_test_rgb_image();
    let dims = img.shape().dims();
    let out_ident = apply_saturation(&img, dims, 1.0).expect("test: saturation identity");
    let orig = img.data();
    let out = out_ident.data();
    for i in 0..orig.len() {
        assert!(
            (orig[i] - out[i]).abs() < 1e-10,
            "Saturation with alpha=1 should be identity at {i}"
        );
    }
}

#[test]
fn test_scaling_identity() {
    let img = make_test_image();
    let dims = img.shape().dims();
    let scaled = apply_scaling(&img, dims, 4, 4, 1.0).expect("test: scale=1 identity");
    let orig = img.data();
    let out = scaled.data();
    for i in 0..orig.len() {
        assert!(
            (orig[i] - out[i]).abs() < 1e-4,
            "Scale=1 should be near-identity"
        );
    }
}

#[test]
fn test_self_mixup_preserves_shape() {
    let img = make_test_image();
    let mut rng = StdRng::seed_from_u64(1);
    let mixed = apply_self_mixup(&img, 0.2, &mut rng).expect("test: self-mixup");
    assert_eq!(mixed.shape().dims(), img.shape().dims());
}

#[test]
fn test_self_cutmix_preserves_shape() {
    let batch = make_test_batch();
    let dims = batch.shape().dims();
    let mut rng = StdRng::seed_from_u64(2);
    let mixed = apply_self_cutmix(&batch, dims, 4, 4, 1.0, &mut rng).expect("test: self-cutmix");
    assert_eq!(mixed.shape().dims(), batch.shape().dims());
}

#[test]
fn test_batch_horizontal_flip() {
    let batch = make_test_batch();
    let dims = batch.shape().dims();
    let flipped = apply_horizontal_flip(&batch, dims, 4, 4).expect("test: batch flip");
    assert_eq!(flipped.shape().dims(), batch.shape().dims());
}

#[test]
fn test_sequence_reversal() {
    let seq = make_test_sequence();
    let rev = apply_sequence_reversal(&seq).expect("test: reversal");
    let data = rev.data();
    assert_eq!(data[0], 7.0);
    assert_eq!(data[7], 0.0);
}

#[test]
fn test_token_swap() {
    let seq = make_test_sequence();
    let mut rng = StdRng::seed_from_u64(5);
    let swapped = apply_token_swap(&seq, 2, &mut rng).expect("test: token swap");
    assert_eq!(swapped.shape().dims(), seq.shape().dims());
    let orig_sum: f64 = seq.data().iter().sum();
    let swap_sum: f64 = swapped.data().iter().sum();
    assert!((orig_sum - swap_sum).abs() < 1e-10);
}

#[test]
fn test_token_deletion() {
    let seq = make_test_sequence();
    let mut rng = StdRng::seed_from_u64(10);
    let deleted = apply_token_deletion(&seq, 0.5, &mut rng).expect("test: token deletion");
    let zero_count = deleted.data().iter().filter(|&&v| v == 0.0).count();
    assert!(zero_count >= 1, "Deletion should produce at least 1 zero");
}

#[test]
fn test_time_masking() {
    let spec = make_test_spectrogram();
    let mut rng = StdRng::seed_from_u64(8);
    let masked = apply_time_masking(&spec, 5, &mut rng).expect("test: time masking");
    assert_eq!(masked.shape().dims(), spec.shape().dims());
}

#[test]
fn test_frequency_masking() {
    let spec = make_test_spectrogram();
    let mut rng = StdRng::seed_from_u64(9);
    let masked = apply_frequency_masking(&spec, 3, &mut rng).expect("test: frequency masking");
    assert_eq!(masked.shape().dims(), spec.shape().dims());
    let zero_count = masked.data().iter().filter(|&&v| v == 0.0).count();
    assert!(zero_count > 0, "Frequency masking should zero some values");
}

#[test]
fn test_time_warping() {
    let spec = make_test_spectrogram();
    let mut rng = StdRng::seed_from_u64(11);
    let warped = apply_time_warping(&spec, 0.2, &mut rng).expect("test: time warping");
    assert_eq!(warped.shape().dims(), spec.shape().dims());
}

#[test]
fn test_pipeline_apply_image_with_seed() {
    let img = make_test_image();
    let pipeline = AugmentationPipeline::new()
        .add_image_augmentation(
            ImageAugmentation::HorizontalFlip,
            AugmentationConfig::new()
                .with_probability(1.0)
                .with_seed(42),
        )
        .add_image_augmentation(
            ImageAugmentation::GaussianNoise,
            AugmentationConfig::new()
                .with_probability(1.0)
                .with_seed(43)
                .with_parameter("std".to_string(), 0.01),
        );
    let result = pipeline.apply_image(&img).expect("test: pipeline apply");
    assert_eq!(result.shape().dims(), img.shape().dims());
}

#[test]
fn test_pipeline_apply_sequence_with_seed() {
    let seq = make_test_sequence();
    let pipeline = AugmentationPipeline::new()
        .add_sequence_augmentation(
            SequenceAugmentation::WordSwap,
            AugmentationConfig::new()
                .with_probability(1.0)
                .with_seed(44)
                .with_parameter("n_swaps".to_string(), 1.0),
        )
        .add_sequence_augmentation(
            SequenceAugmentation::NoiseInsertion,
            AugmentationConfig::new()
                .with_probability(1.0)
                .with_seed(45)
                .with_parameter("std".to_string(), 0.01),
        );
    let result = pipeline
        .apply_sequence(&seq)
        .expect("test: pipeline sequence");
    assert_eq!(result.shape().dims(), seq.shape().dims());
}

#[test]
fn test_disabled_augmentation_is_skipped() {
    let img = make_test_image();
    let pipeline = AugmentationPipeline::new().add_image_augmentation(
        ImageAugmentation::HorizontalFlip,
        AugmentationConfig::new().with_enabled(false),
    );
    let result = pipeline.apply_image(&img).expect("test: disabled skip");
    let orig = img.data();
    let out = result.data();
    for i in 0..orig.len() {
        assert_eq!(orig[i], out[i], "Disabled augmentation should be a no-op");
    }
}

#[test]
fn test_hue_shift_preserves_shape() {
    let img = make_test_rgb_image();
    let shifted = apply_hue_shift(&img, 0.1).expect("test: hue shift");
    assert_eq!(shifted.shape().dims(), img.shape().dims());
}

#[test]
fn test_shear_preserves_shape() {
    let img = make_test_image();
    let dims = img.shape().dims();
    let sheared = apply_shear(&img, dims, 4, 4, 0.1, 0.1).expect("test: shear");
    assert_eq!(sheared.shape().dims(), img.shape().dims());
}

#[test]
fn test_translation_preserves_shape() {
    let img = make_test_image();
    let dims = img.shape().dims();
    let translated = apply_translation(&img, dims, 4, 4, 1.0, 1.0).expect("test: translation");
    assert_eq!(translated.shape().dims(), img.shape().dims());
}

#[test]
fn test_sample_beta_range() {
    let mut rng = StdRng::seed_from_u64(100);
    for _ in 0..100 {
        let v = sample_beta(0.5, 0.5, &mut rng);
        assert!((0.0..=1.0).contains(&v), "Beta sample {v} out of [0,1]");
    }
}

#[test]
fn test_box_muller_produces_values() {
    let mut rng = StdRng::seed_from_u64(200);
    let (z0, z1) = box_muller(&mut rng);
    assert!(z0.is_finite());
    assert!(z1.is_finite());
}

#[test]
fn test_random_crop_and_resize() {
    let img = make_test_image();
    let dims = img.shape().dims();
    let mut rng = StdRng::seed_from_u64(12);
    let cropped = apply_random_crop_and_resize(&img, dims, 4, 4, 0.5, &mut rng)
        .expect("test: crop and resize");
    assert_eq!(cropped.shape().dims(), img.shape().dims());
}

#[test]
fn test_double_sequence_reversal_identity() {
    let seq = make_test_sequence();
    let rev1 = apply_sequence_reversal(&seq).expect("test: rev1");
    let rev2 = apply_sequence_reversal(&rev1).expect("test: rev2");
    let orig = seq.data();
    let restored = rev2.data();
    for i in 0..orig.len() {
        assert!(
            (orig[i] - restored[i]).abs() < 1e-10,
            "Double reversal should be identity"
        );
    }
}

#[test]
fn test_invalid_shape_rejected() {
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<f64>::from_vec(data, &[2, 2]).expect("test: create 2d");
    let pipeline = AugmentationPipeline::new().add_image_augmentation(
        ImageAugmentation::HorizontalFlip,
        AugmentationConfig::new().with_probability(1.0).with_seed(1),
    );
    let result = pipeline.apply_image(&tensor);
    assert!(result.is_err());
}
