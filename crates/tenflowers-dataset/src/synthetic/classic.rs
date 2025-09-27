//! Classical Machine Learning Dataset Generation
//!
//! This module contains implementations for generating standard synthetic datasets
//! commonly used in machine learning research and benchmarking.

use super::core::{DatasetGenerator, SyntheticConfig, SyntheticDataset};
// use crate::Dataset; // Unused import removed
use scirs2_core::random::Random;
use std::f64::consts::PI;
use tenflowers_core::{Result, Tensor, TensorError};

impl DatasetGenerator {
    /// Generate two interleaving half circles (moons)
    pub fn make_moons(config: SyntheticConfig) -> Result<SyntheticDataset<f64>> {
        let mut rng = if let Some(seed) = config.random_seed {
            Random::seed(seed)
        } else {
            Random::seed(0)
        };

        let n_samples_out = config.n_samples / 2;
        let n_samples_in = config.n_samples - n_samples_out;

        let mut features = Vec::new();
        let mut labels = Vec::new();

        // Generate outer semicircle
        for i in 0..n_samples_out {
            let angle = PI * (i as f64) / (n_samples_out as f64 - 1.0);
            let x = angle.cos();
            let y = angle.sin();

            // Add noise
            let noise_x = rng.gen_range(-config.noise_level..config.noise_level);
            let noise_y = rng.gen_range(-config.noise_level..config.noise_level);

            features.push(x + noise_x);
            features.push(y + noise_y);
            labels.push(0.0);
        }

        // Generate inner semicircle
        for i in 0..n_samples_in {
            let angle = PI * (i as f64) / (n_samples_in as f64 - 1.0);
            let x = 1.0 - angle.cos();
            let y = 1.0 - angle.sin() - 0.5;

            // Add noise
            let noise_x = rng.gen_range(-config.noise_level..config.noise_level);
            let noise_y = rng.gen_range(-config.noise_level..config.noise_level);

            features.push(x + noise_x);
            features.push(y + noise_y);
            labels.push(1.0);
        }

        // Shuffle if requested
        if config.shuffle {
            let mut combined: Vec<(f64, f64, f64)> = features
                .chunks_exact(2)
                .zip(labels.iter())
                .map(|(chunk, &label)| (chunk[0], chunk[1], label))
                .collect();

            rng.shuffle(&mut combined);

            features.clear();
            labels.clear();
            for (x, y, label) in combined {
                features.push(x);
                features.push(y);
                labels.push(label);
            }
        }

        let feature_tensor = Tensor::from_vec(features, &[config.n_samples, 2])?;
        let label_tensor = Tensor::from_vec(labels, &[config.n_samples])?;

        Ok(SyntheticDataset::new(feature_tensor, label_tensor))
    }

    /// Generate concentric circles
    pub fn make_circles(config: SyntheticConfig, factor: f64) -> Result<SyntheticDataset<f64>> {
        let mut rng = if let Some(seed) = config.random_seed {
            Random::seed(seed)
        } else {
            Random::seed(0)
        };

        let n_samples_out = config.n_samples / 2;
        let n_samples_in = config.n_samples - n_samples_out;

        let mut features = Vec::new();
        let mut labels = Vec::new();

        // Generate outer circle
        for _ in 0..n_samples_out {
            let angle = rng.gen_range(0.0..2.0 * PI);
            let radius = 1.0;
            let x = radius * angle.cos();
            let y = radius * angle.sin();

            // Add noise
            let noise_x = rng.gen_range(-config.noise_level..config.noise_level);
            let noise_y = rng.gen_range(-config.noise_level..config.noise_level);

            features.push(x + noise_x);
            features.push(y + noise_y);
            labels.push(0.0);
        }

        // Generate inner circle
        for _ in 0..n_samples_in {
            let angle = rng.gen_range(0.0..2.0 * PI);
            let radius = factor;
            let x = radius * angle.cos();
            let y = radius * angle.sin();

            // Add noise
            let noise_x = rng.gen_range(-config.noise_level..config.noise_level);
            let noise_y = rng.gen_range(-config.noise_level..config.noise_level);

            features.push(x + noise_x);
            features.push(y + noise_y);
            labels.push(1.0);
        }

        // Shuffle if requested
        if config.shuffle {
            let mut combined: Vec<(f64, f64, f64)> = features
                .chunks_exact(2)
                .zip(labels.iter())
                .map(|(chunk, &label)| (chunk[0], chunk[1], label))
                .collect();

            rng.shuffle(&mut combined);

            features.clear();
            labels.clear();
            for (x, y, label) in combined {
                features.push(x);
                features.push(y);
                labels.push(label);
            }
        }

        let feature_tensor = Tensor::from_vec(features, &[config.n_samples, 2])?;
        let label_tensor = Tensor::from_vec(labels, &[config.n_samples])?;

        Ok(SyntheticDataset::new(feature_tensor, label_tensor))
    }

    /// Generate Gaussian blobs
    pub fn make_blobs(
        config: SyntheticConfig,
        n_features: usize,
        centers: Option<usize>,
        cluster_std: f64,
        center_box: (f64, f64),
    ) -> Result<SyntheticDataset<f64>> {
        let mut rng = if let Some(seed) = config.random_seed {
            Random::seed(seed)
        } else {
            Random::seed(0)
        };

        let n_centers = centers.unwrap_or(3);

        // Generate random cluster centers
        let mut cluster_centers = Vec::new();
        for _ in 0..n_centers {
            let mut center = Vec::new();
            for _ in 0..n_features {
                center.push(rng.gen_range(center_box.0..center_box.1));
            }
            cluster_centers.push(center);
        }

        let mut features = Vec::new();
        let mut labels = Vec::new();

        // Generate samples for each cluster
        let samples_per_cluster = config.n_samples / n_centers;
        let remaining_samples = config.n_samples % n_centers;

        for (cluster_id, center) in cluster_centers.iter().enumerate() {
            let cluster_samples = if cluster_id < remaining_samples {
                samples_per_cluster + 1
            } else {
                samples_per_cluster
            };

            for _ in 0..cluster_samples {
                for &center_val in center.iter().take(n_features) {
                    let noise = rng.random_range(-cluster_std, cluster_std);
                    let value = center_val + noise;
                    features.push(value);
                }
                labels.push(cluster_id as f64);
            }
        }

        // Shuffle if requested
        if config.shuffle {
            let mut combined: Vec<(Vec<f64>, f64)> = features
                .chunks_exact(n_features)
                .zip(labels.iter())
                .map(|(chunk, &label)| (chunk.to_vec(), label))
                .collect();

            rng.shuffle(&mut combined);

            features.clear();
            labels.clear();
            for (feat_vec, label) in combined {
                features.extend(feat_vec);
                labels.push(label);
            }
        }

        let feature_tensor = Tensor::from_vec(features, &[config.n_samples, n_features])?;
        let label_tensor = Tensor::from_vec(labels, &[config.n_samples])?;

        Ok(SyntheticDataset::new(feature_tensor, label_tensor))
    }

    /// Generate linearly separable classification data
    pub fn make_classification(
        config: SyntheticConfig,
        n_features: usize,
        n_informative: usize,
        n_redundant: usize,
        n_classes: usize,
        flip_y: f64,
    ) -> Result<SyntheticDataset<f64>> {
        let mut rng = if let Some(seed) = config.random_seed {
            Random::seed(seed)
        } else {
            Random::seed(0)
        };

        if n_informative + n_redundant > n_features {
            return Err(TensorError::invalid_argument(
                "n_informative + n_redundant cannot exceed n_features".to_string(),
            ));
        }

        // Generate informative features
        let mut features = vec![0.0; config.n_samples * n_features];
        let mut labels = Vec::new();

        // Generate random weights for each class
        let mut class_weights = Vec::new();
        for _ in 0..n_classes {
            let mut weights = Vec::new();
            for _ in 0..n_informative {
                weights.push(rng.gen_range(-1.0..1.0));
            }
            class_weights.push(weights);
        }

        // Generate samples
        for sample_idx in 0..config.n_samples {
            // Choose random class
            let class_id = rng.random_range(0, n_classes);

            // Generate informative features
            for feat_idx in 0..n_informative {
                let base_value = rng.gen_range(-1.0..1.0);
                let class_bias = class_weights[class_id][feat_idx];
                let feature_value = base_value
                    + class_bias
                    + rng.gen_range(-config.noise_level..config.noise_level);

                features[sample_idx * n_features + feat_idx] = feature_value;
            }

            // Generate redundant features (linear combinations of informative features)
            for redundant_idx in 0..n_redundant {
                let feat_idx = n_informative + redundant_idx;
                let mut redundant_value = 0.0;

                for info_idx in 0..n_informative {
                    let weight = rng.gen_range(-0.5..0.5);
                    redundant_value += weight * features[sample_idx * n_features + info_idx];
                }

                redundant_value += rng.gen_range(-config.noise_level..config.noise_level);
                features[sample_idx * n_features + feat_idx] = redundant_value;
            }

            // Generate noise features
            for noise_idx in (n_informative + n_redundant)..n_features {
                features[sample_idx * n_features + noise_idx] = rng.gen_range(-1.0..1.0);
            }

            // Assign label with possible flip
            let final_label = if rng.gen_range(0.0..1.0) < flip_y {
                rng.random_range(0, n_classes)
            } else {
                class_id
            };

            labels.push(final_label as f64);
        }

        // Shuffle if requested
        if config.shuffle {
            let mut combined: Vec<(Vec<f64>, f64)> = features
                .chunks_exact(n_features)
                .zip(labels.iter())
                .map(|(chunk, &label)| (chunk.to_vec(), label))
                .collect();

            rng.shuffle(&mut combined);

            features.clear();
            labels.clear();
            for (feat_vec, label) in combined {
                features.extend(feat_vec);
                labels.push(label);
            }
        }

        let feature_tensor = Tensor::from_vec(features, &[config.n_samples, n_features])?;
        let label_tensor = Tensor::from_vec(labels, &[config.n_samples])?;

        Ok(SyntheticDataset::new(feature_tensor, label_tensor))
    }

    /// Generate regression data
    pub fn make_regression(
        config: SyntheticConfig,
        n_features: usize,
        n_informative: usize,
        effective_rank: Option<usize>,
        tail_strength: f64,
        bias: f64,
    ) -> Result<SyntheticDataset<f64>> {
        let mut rng = if let Some(seed) = config.random_seed {
            Random::seed(seed)
        } else {
            Random::seed(0)
        };

        // Generate random X matrix
        let mut features = Vec::new();
        for _ in 0..(config.n_samples * n_features) {
            features.push(rng.gen_range(-1.0..1.0));
        }

        // Generate random ground truth weights
        let mut true_weights = Vec::new();
        for i in 0..n_informative {
            let weight = if let Some(rank) = effective_rank {
                if i < rank {
                    100.0 * rng.gen_range(-1.0..1.0)
                } else {
                    tail_strength * rng.gen_range(-1.0..1.0)
                }
            } else {
                rng.gen_range(-1.0..1.0)
            };
            true_weights.push(weight);
        }

        // Extend weights with zeros for non-informative features
        while true_weights.len() < n_features {
            true_weights.push(0.0);
        }

        // Generate targets
        let mut labels = Vec::new();
        for sample_idx in 0..config.n_samples {
            let mut target = bias;

            for feat_idx in 0..n_features {
                let feature_value = features[sample_idx * n_features + feat_idx];
                target += feature_value * true_weights[feat_idx];
            }

            // Add noise
            target += rng.gen_range(-config.noise_level..config.noise_level);
            labels.push(target);
        }

        let feature_tensor = Tensor::from_vec(features, &[config.n_samples, n_features])?;
        let label_tensor = Tensor::from_vec(labels, &[config.n_samples])?;

        Ok(SyntheticDataset::new(feature_tensor, label_tensor))
    }

    /// Generate S-curve manifold
    pub fn make_s_curve(config: SyntheticConfig, noise: f64) -> Result<SyntheticDataset<f64>> {
        let mut rng = if let Some(seed) = config.random_seed {
            Random::seed(seed)
        } else {
            Random::seed(0)
        };

        let mut features = Vec::new();
        let mut labels = Vec::new(); // Will contain the parameter t for color coding

        for _ in 0..config.n_samples {
            let t = rng.gen_range(0.0..1.0);

            // S-curve parametric equations
            let arg: f64 = 1.5 * (1.5 * t - 1.0);
            let x = arg.sin();
            let y = 2.0 * rng.gen_range(-1.0..1.0); // Random y coordinate
            let z = arg.signum() * arg.cos();

            // Add noise
            features.push(x + noise * rng.gen_range(-1.0..1.0));
            features.push(y + noise * rng.gen_range(-1.0..1.0));
            features.push(z + noise * rng.gen_range(-1.0..1.0));

            labels.push(t);
        }

        let feature_tensor = Tensor::from_vec(features, &[config.n_samples, 3])?;
        let label_tensor = Tensor::from_vec(labels, &[config.n_samples])?;

        Ok(SyntheticDataset::new(feature_tensor, label_tensor))
    }

    /// Generate Swiss roll manifold
    pub fn make_swiss_roll(config: SyntheticConfig, noise: f64) -> Result<SyntheticDataset<f64>> {
        let mut rng = if let Some(seed) = config.random_seed {
            Random::seed(seed)
        } else {
            Random::seed(0)
        };

        let mut features = Vec::new();
        let mut labels = Vec::new(); // Will contain the parameter t for color coding

        for _ in 0..config.n_samples {
            let t = rng.gen_range(1.5 * PI..4.5 * PI);
            let height = rng.gen_range(0.0..21.0);

            // Swiss roll parametric equations
            let x = t * t.cos();
            let y = height;
            let z = t * t.sin();

            // Add noise
            features.push(x + noise * rng.gen_range(-1.0..1.0));
            features.push(y + noise * rng.gen_range(-1.0..1.0));
            features.push(z + noise * rng.gen_range(-1.0..1.0));

            labels.push(t);
        }

        let feature_tensor = Tensor::from_vec(features, &[config.n_samples, 3])?;
        let label_tensor = Tensor::from_vec(labels, &[config.n_samples])?;

        Ok(SyntheticDataset::new(feature_tensor, label_tensor))
    }
}
