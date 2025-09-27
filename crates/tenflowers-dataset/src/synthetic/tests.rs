//! Tests for Synthetic Dataset Generation
//!
//! This module contains unit tests for all synthetic dataset generation functions
//! to ensure correctness and reliability.

#[cfg(test)]
mod tests {
    use super::super::core::{DatasetGenerator, SyntheticConfig};
    use super::super::image::{ImagePatternConfig, ImagePatternType};
    use super::super::text::{TextCorpusConfig, TextSynthesisTask};
    use super::super::timeseries::TimeSeriesPattern;
    use crate::Dataset;

    #[test]
    fn test_synthetic_config() {
        let config = SyntheticConfig::new(100)
            .with_seed(42)
            .with_noise(0.05)
            .with_shuffle(false);

        assert_eq!(config.n_samples, 100);
        assert_eq!(config.random_seed, Some(42));
        assert_eq!(config.noise_level, 0.05);
        assert!(!config.shuffle);
    }

    #[test]
    fn test_make_moons() {
        let config = SyntheticConfig::new(100).with_seed(42);
        let dataset = DatasetGenerator::make_moons(config).unwrap();

        assert_eq!(dataset.len(), 100);
        let (features, labels) = dataset.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[2]);
        assert_eq!(labels.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_make_circles() {
        let config = SyntheticConfig::new(50).with_seed(42);
        let dataset = DatasetGenerator::make_circles(config, 0.5).unwrap();

        assert_eq!(dataset.len(), 50);
        let (features, labels) = dataset.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[2]);
        assert_eq!(labels.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_make_blobs() {
        let config = SyntheticConfig::new(150).with_seed(42);
        let dataset = DatasetGenerator::make_blobs(
            config,
            4,           // n_features
            Some(3),     // centers
            1.0,         // cluster_std
            (-5.0, 5.0), // center_box
        )
        .unwrap();

        assert_eq!(dataset.len(), 150);
        let (features, labels) = dataset.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[4]);
        assert_eq!(labels.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_make_classification() {
        let config = SyntheticConfig::new(100).with_seed(42);
        let dataset = DatasetGenerator::make_classification(
            config, 10,   // n_features
            5,    // n_informative
            2,    // n_redundant
            3,    // n_classes
            0.01, // flip_y
        )
        .unwrap();

        assert_eq!(dataset.len(), 100);
        let (features, labels) = dataset.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[10]);
        assert_eq!(labels.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_make_regression() {
        let config = SyntheticConfig::new(100).with_seed(42);
        let dataset = DatasetGenerator::make_regression(
            config,
            5,       // n_features
            3,       // n_informative
            Some(2), // effective_rank
            0.01,    // tail_strength
            0.0,     // bias
        )
        .unwrap();

        assert_eq!(dataset.len(), 100);
        let (features, labels) = dataset.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[5]);
        assert_eq!(labels.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_make_s_curve() {
        let config = SyntheticConfig::new(100).with_seed(42);
        let dataset = DatasetGenerator::make_s_curve(config, 0.1).unwrap();

        assert_eq!(dataset.len(), 100);
        let (features, labels) = dataset.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[3]);
        assert_eq!(labels.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_make_swiss_roll() {
        let config = SyntheticConfig::new(100).with_seed(42);
        let dataset = DatasetGenerator::make_swiss_roll(config, 0.1).unwrap();

        assert_eq!(dataset.len(), 100);
        let (features, labels) = dataset.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[3]);
        assert_eq!(labels.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_time_series_patterns() {
        let config = SyntheticConfig::new(50).with_seed(42);
        let pattern = TimeSeriesPattern::Sine { frequency: 2.0 };
        let dataset = DatasetGenerator::make_time_series(config, pattern, 20).unwrap();

        assert_eq!(dataset.len(), 50);
        let (features, labels) = dataset.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[20]);
        assert_eq!(labels.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_deterministic_generation() {
        let config1 = SyntheticConfig::new(50).with_seed(42);
        let config2 = SyntheticConfig::new(50).with_seed(42);

        let dataset1 = DatasetGenerator::make_moons(config1).unwrap();
        let dataset2 = DatasetGenerator::make_moons(config2).unwrap();

        // With same seed, should generate identical datasets
        let (features1, _) = dataset1.get(0).unwrap();
        let (features2, _) = dataset2.get(0).unwrap();

        let data1 = features1.to_vec().unwrap();
        let data2 = features2.to_vec().unwrap();

        // Check first few values are equal (within floating point precision)
        for (a, b) in data1.iter().zip(data2.iter()).take(4) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_text_corpus_generation() {
        let config = TextCorpusConfig::new(100)
            .with_sequence_length(5, 15)
            .with_task(TextSynthesisTask::Classification)
            .with_seed(42);

        let dataset = DatasetGenerator::make_text_corpus(config).unwrap();
        assert!(dataset.len() > 0);

        let (features, labels) = dataset.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[15]); // max_sequence_length
        assert_eq!(labels.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_image_pattern_generation() {
        use super::super::image::{ImagePatternGenerator, StripeOrientation};

        let config = ImagePatternConfig::new(32, 32)
            .with_pattern(ImagePatternType::Stripes {
                width: 4,
                orientation: StripeOrientation::Horizontal,
            })
            .with_channels(3);

        let mut rng = scirs2_core::random::rng();
        let image = ImagePatternGenerator::generate_image(&config, &mut rng).unwrap();

        assert_eq!(image.shape().dims(), &[3, 32, 32]);
    }
}
