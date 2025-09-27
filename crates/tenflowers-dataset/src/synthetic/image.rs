//! Image Pattern Generation
//!
//! This module contains functionality for generating synthetic images with
//! various patterns for computer vision tasks and dataset creation.

use super::core::{SyntheticConfig, SyntheticDataset};
use scirs2_core::random::{Rng, RngCore, SeedableRng};
use tenflowers_core::{Result, Tensor};

/// Configuration for synthetic image pattern generation
#[derive(Debug, Clone)]
pub struct ImagePatternConfig {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub pattern_type: ImagePatternType,
    pub noise_level: f64,
    pub background_color: [f32; 3],
    pub foreground_color: [f32; 3],
}

#[derive(Debug, Clone)]
pub enum ImagePatternType {
    Checkerboard {
        size: usize,
    },
    Stripes {
        width: usize,
        orientation: StripeOrientation,
    },
    Circles {
        radius: f32,
        num_circles: usize,
    },
    Gradient {
        direction: GradientDirection,
    },
    Noise {
        distribution: NoiseDistribution,
    },
    Geometric {
        shape: GeometricShape,
        size: f32,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum StripeOrientation {
    Horizontal,
    Vertical,
    Diagonal,
}

#[derive(Debug, Clone, Copy)]
pub enum GradientDirection {
    Horizontal,
    Vertical,
    Radial,
}

#[derive(Debug, Clone, Copy)]
pub enum NoiseDistribution {
    Uniform,
    Gaussian,
    Salt,
    Pepper,
    SaltAndPepper,
}

#[derive(Debug, Clone, Copy)]
pub enum GeometricShape {
    Rectangle,
    Circle,
    Triangle,
    Star,
}

impl Default for ImagePatternConfig {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            channels: 3,
            pattern_type: ImagePatternType::Checkerboard { size: 8 },
            noise_level: 0.0,
            background_color: [0.0, 0.0, 0.0], // Black
            foreground_color: [1.0, 1.0, 1.0], // White
        }
    }
}

impl ImagePatternConfig {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    pub fn with_pattern(mut self, pattern_type: ImagePatternType) -> Self {
        self.pattern_type = pattern_type;
        self
    }

    pub fn with_colors(mut self, background: [f32; 3], foreground: [f32; 3]) -> Self {
        self.background_color = background;
        self.foreground_color = foreground;
        self
    }

    pub fn with_noise(mut self, noise_level: f64) -> Self {
        self.noise_level = noise_level;
        self
    }

    pub fn with_channels(mut self, channels: usize) -> Self {
        self.channels = channels;
        self
    }
}

/// Image pattern generator for creating synthetic image datasets
pub struct ImagePatternGenerator;

impl ImagePatternGenerator {
    /// Generate a single image with the specified pattern
    pub fn generate_image(
        config: &ImagePatternConfig,
        rng: &mut dyn RngCore,
    ) -> Result<Tensor<f32>> {
        let total_pixels = config.width * config.height * config.channels;
        let mut image_data = vec![0.0f32; total_pixels];

        // Generate base pattern
        match &config.pattern_type {
            ImagePatternType::Checkerboard { size } => {
                Self::generate_checkerboard(&mut image_data, config, *size);
            }
            ImagePatternType::Stripes { width, orientation } => {
                Self::generate_stripes(&mut image_data, config, *width, *orientation);
            }
            ImagePatternType::Circles {
                radius,
                num_circles,
            } => {
                Self::generate_circles(&mut image_data, config, *radius, *num_circles, rng);
            }
            ImagePatternType::Gradient { direction } => {
                Self::generate_gradient(&mut image_data, config, *direction);
            }
            ImagePatternType::Noise { distribution } => {
                Self::generate_noise(&mut image_data, config, *distribution, rng);
            }
            ImagePatternType::Geometric { shape, size } => {
                Self::generate_geometric(&mut image_data, config, *shape, *size);
            }
        }

        // Add noise if specified
        if config.noise_level > 0.0 {
            for pixel in image_data.iter_mut() {
                let noise = rng.random_range(-config.noise_level..config.noise_level) as f32;
                *pixel = (*pixel + noise).clamp(0.0, 1.0);
            }
        }

        let shape = vec![config.channels, config.height, config.width];
        Tensor::from_vec(image_data, &shape)
    }

    fn generate_checkerboard(data: &mut [f32], config: &ImagePatternConfig, size: usize) {
        for y in 0..config.height {
            for x in 0..config.width {
                let checker_x = (x / size) % 2;
                let checker_y = (y / size) % 2;
                let is_foreground = (checker_x + checker_y) % 2 == 0;

                let color = if is_foreground {
                    config.foreground_color
                } else {
                    config.background_color
                };

                for c in 0..config.channels {
                    let idx = c * config.height * config.width + y * config.width + x;
                    data[idx] = color[c.min(2)];
                }
            }
        }
    }

    fn generate_stripes(
        data: &mut [f32],
        config: &ImagePatternConfig,
        width: usize,
        orientation: StripeOrientation,
    ) {
        for y in 0..config.height {
            for x in 0..config.width {
                let stripe_pos = match orientation {
                    StripeOrientation::Horizontal => y,
                    StripeOrientation::Vertical => x,
                    StripeOrientation::Diagonal => x + y,
                };

                let is_foreground = (stripe_pos / width) % 2 == 0;
                let color = if is_foreground {
                    config.foreground_color
                } else {
                    config.background_color
                };

                for c in 0..config.channels {
                    let idx = c * config.height * config.width + y * config.width + x;
                    data[idx] = color[c.min(2)];
                }
            }
        }
    }

    fn generate_circles(
        data: &mut [f32],
        config: &ImagePatternConfig,
        radius: f32,
        num_circles: usize,
        rng: &mut dyn RngCore,
    ) {
        // Initialize with background
        for pixel in data.iter_mut() {
            *pixel = config.background_color[0];
        }

        for _ in 0..num_circles {
            let center_x = rng.random_range(0.0..config.width as f32);
            let center_y = rng.random_range(0.0..config.height as f32);

            for y in 0..config.height {
                for x in 0..config.width {
                    let dx = x as f32 - center_x;
                    let dy = y as f32 - center_y;
                    let distance = (dx * dx + dy * dy).sqrt();

                    if distance <= radius {
                        for c in 0..config.channels {
                            let idx = c * config.height * config.width + y * config.width + x;
                            data[idx] = config.foreground_color[c.min(2)];
                        }
                    }
                }
            }
        }
    }

    fn generate_gradient(
        data: &mut [f32],
        config: &ImagePatternConfig,
        direction: GradientDirection,
    ) {
        for y in 0..config.height {
            for x in 0..config.width {
                let gradient_value = match direction {
                    GradientDirection::Horizontal => x as f32 / config.width as f32,
                    GradientDirection::Vertical => y as f32 / config.height as f32,
                    GradientDirection::Radial => {
                        let center_x = config.width as f32 / 2.0;
                        let center_y = config.height as f32 / 2.0;
                        let dx = x as f32 - center_x;
                        let dy = y as f32 - center_y;
                        let max_distance = (center_x * center_x + center_y * center_y).sqrt();
                        let distance = (dx * dx + dy * dy).sqrt();
                        distance / max_distance
                    }
                };

                for c in 0..config.channels {
                    let idx = c * config.height * config.width + y * config.width + x;
                    data[idx] = gradient_value.clamp(0.0, 1.0);
                }
            }
        }
    }

    fn generate_noise(
        data: &mut [f32],
        _config: &ImagePatternConfig,
        distribution: NoiseDistribution,
        rng: &mut dyn RngCore,
    ) {
        for pixel in data.iter_mut() {
            *pixel = match distribution {
                NoiseDistribution::Uniform => rng.random::<f32>(),
                NoiseDistribution::Gaussian => {
                    // Box-Muller transform for Gaussian noise
                    let u1 = rng.random::<f32>();
                    let u2 = rng.random::<f32>();
                    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                    (z0 * 0.5 + 0.5).clamp(0.0, 1.0)
                }
                NoiseDistribution::Salt => {
                    if rng.random::<f32>() < 0.1 {
                        1.0
                    } else {
                        0.0
                    }
                }
                NoiseDistribution::Pepper => {
                    if rng.random::<f32>() < 0.1 {
                        0.0
                    } else {
                        1.0
                    }
                }
                NoiseDistribution::SaltAndPepper => {
                    let rand_val = rng.random::<f32>();
                    if rand_val < 0.05 {
                        0.0
                    } else if rand_val < 0.1 {
                        1.0
                    } else {
                        0.5
                    }
                }
            };
        }
    }

    fn generate_geometric(
        data: &mut [f32],
        config: &ImagePatternConfig,
        shape: GeometricShape,
        size: f32,
    ) {
        // Initialize with background
        for pixel in data.iter_mut() {
            *pixel = config.background_color[0];
        }

        let center_x = config.width as f32 / 2.0;
        let center_y = config.height as f32 / 2.0;

        for y in 0..config.height {
            for x in 0..config.width {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;

                let is_inside = match shape {
                    GeometricShape::Rectangle => dx.abs() <= size / 2.0 && dy.abs() <= size / 2.0,
                    GeometricShape::Circle => (dx * dx + dy * dy).sqrt() <= size / 2.0,
                    GeometricShape::Triangle => {
                        // Simple triangle approximation
                        dy >= -size / 2.0 && dy <= size / 2.0 && dx.abs() <= size / 2.0 - dy.abs()
                    }
                    GeometricShape::Star => {
                        // Simple star approximation
                        let angle = dy.atan2(dx);
                        let distance = (dx * dx + dy * dy).sqrt();
                        let star_radius = size / 2.0 * (1.0 + 0.3 * (5.0 * angle).sin());
                        distance <= star_radius
                    }
                };

                if is_inside {
                    for c in 0..config.channels {
                        let idx = c * config.height * config.width + y * config.width + x;
                        data[idx] = config.foreground_color[c.min(2)];
                    }
                }
            }
        }
    }

    /// Generate a dataset of synthetic images
    pub fn generate_dataset(
        config: &ImagePatternConfig,
        synthetic_config: SyntheticConfig,
    ) -> Result<SyntheticDataset<f32>> {
        let mut rng: Box<dyn RngCore> = if let Some(seed) = synthetic_config.random_seed {
            Box::new(scirs2_core::random::rngs::StdRng::seed_from_u64(seed))
        } else {
            Box::new(scirs2_core::random::rng())
        };

        let image_size = config.width * config.height * config.channels;
        let mut all_images = Vec::with_capacity(synthetic_config.n_samples * image_size);
        let mut labels = Vec::with_capacity(synthetic_config.n_samples);

        for i in 0..synthetic_config.n_samples {
            let image = Self::generate_image(config, &mut *rng)?;
            let image_data = image.to_vec().map_err(|_| {
                tenflowers_core::TensorError::invalid_argument(
                    "Failed to convert image tensor to vector".to_string(),
                )
            })?;

            all_images.extend(image_data);
            labels.push(i as f32); // Simple label scheme
        }

        let features = Tensor::from_vec(
            all_images,
            &[
                synthetic_config.n_samples,
                config.channels,
                config.height,
                config.width,
            ],
        )?;
        let labels_tensor = Tensor::from_vec(labels, &[synthetic_config.n_samples])?;

        Ok(SyntheticDataset::new(features, labels_tensor))
    }
}
