use crate::Dataset;
use tenflowers_core::{Result, Tensor, TensorError};

/// Common benchmark datasets for machine learning
pub struct BenchmarkDatasets;

impl BenchmarkDatasets {
    /// Create a simple synthetic MNIST-like dataset for testing
    /// This generates synthetic data with MNIST-like characteristics
    pub fn synthetic_mnist(num_samples: usize, seed: Option<u64>) -> Result<MnistDataset<f32>> {
        use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

        let mut rng = if let Some(seed) = seed {
            StdRng::seed_from_u64(seed)
        } else {
            // Use a default seed if none provided
            StdRng::seed_from_u64(12345)
        };

        let image_size = 28 * 28;
        let num_classes = 10;

        // Generate synthetic images (28x28 = 784 features)
        let mut image_data = Vec::with_capacity(num_samples * image_size);
        let mut labels = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            // Generate a synthetic "digit" pattern
            let class = rng.random_range(0..num_classes);
            labels.push(class as f32);

            // Create a simple pattern based on the class
            for i in 0..image_size {
                let row = i / 28;
                let col = i % 28;

                // Create simple patterns for each digit class
                let pixel_value = match class {
                    0 => create_circle_pattern(row, col, 28),
                    1 => create_vertical_line_pattern(row, col, 28),
                    2 => create_horizontal_line_pattern(row, col, 28),
                    3 => create_diagonal_pattern(row, col, 28),
                    4 => create_cross_pattern(row, col, 28),
                    5 => create_square_pattern(row, col, 28),
                    _ => rng.random::<f32>() * 0.3, // Random noise for other classes
                };

                // Add some noise
                let noise = rng.random::<f32>() * 0.1;
                let final_value = (pixel_value + noise).clamp(0.0, 1.0);
                image_data.push(final_value);
            }
        }

        let images = Tensor::from_vec(image_data, &[num_samples, image_size])?;
        let labels_tensor = Tensor::from_vec(labels, &[num_samples])?;

        Ok(MnistDataset {
            images,
            labels: labels_tensor,
            num_samples,
        })
    }

    /// Create a simple synthetic CIFAR-like dataset for testing
    /// This generates synthetic RGB images with CIFAR-like characteristics
    pub fn synthetic_cifar10(num_samples: usize, seed: Option<u64>) -> Result<CifarDataset<f32>> {
        use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

        let mut rng = if let Some(seed) = seed {
            StdRng::seed_from_u64(seed)
        } else {
            // Use a default seed if none provided
            StdRng::seed_from_u64(12345)
        };

        let image_size = 32 * 32 * 3; // CIFAR-10: 32x32 RGB
        let num_classes = 10;

        // Generate synthetic RGB images
        let mut image_data = Vec::with_capacity(num_samples * image_size);
        let mut labels = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let class = rng.random_range(0..num_classes);
            labels.push(class as f32);

            // Create RGB patterns for each class
            for i in 0..(32 * 32) {
                let row = i / 32;
                let col = i % 32;

                // Generate RGB values based on class
                let (r, g, b) = match class {
                    0 => (
                        rng.random::<f32>() * 0.5 + 0.5,
                        rng.random::<f32>() * 0.3,
                        rng.random::<f32>() * 0.3,
                    ), // Red-ish
                    1 => (
                        rng.random::<f32>() * 0.3,
                        rng.random::<f32>() * 0.5 + 0.5,
                        rng.random::<f32>() * 0.3,
                    ), // Green-ish
                    2 => (
                        rng.random::<f32>() * 0.3,
                        rng.random::<f32>() * 0.3,
                        rng.random::<f32>() * 0.5 + 0.5,
                    ), // Blue-ish
                    3 => create_rgb_gradient(row, col, 32), // Gradient pattern
                    4 => create_rgb_checkerboard(row, col), // Checkerboard
                    _ => (
                        rng.random::<f32>(),
                        rng.random::<f32>(),
                        rng.random::<f32>(),
                    ), // Random
                };

                image_data.push(r);
                image_data.push(g);
                image_data.push(b);
            }
        }

        let images = Tensor::from_vec(image_data, &[num_samples, 3, 32, 32])?;
        let labels_tensor = Tensor::from_vec(labels, &[num_samples])?;

        Ok(CifarDataset {
            images,
            labels: labels_tensor,
            num_samples,
        })
    }

    /// Create a synthetic iris-like dataset for classification
    pub fn synthetic_iris(num_samples: usize, seed: Option<u64>) -> Result<IrisDataset<f32>> {
        use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

        let mut rng = if let Some(seed) = seed {
            StdRng::seed_from_u64(seed)
        } else {
            // Use a default seed if none provided
            StdRng::seed_from_u64(12345)
        };

        let num_features = 4; // sepal_length, sepal_width, petal_length, petal_width
        let num_classes = 3; // setosa, versicolor, virginica

        let mut features = Vec::with_capacity(num_samples * num_features);
        let mut labels = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let class = rng.random_range(0..num_classes);
            labels.push(class as f32);

            // Generate features based on typical iris characteristics
            let (sepal_length, sepal_width, petal_length, petal_width) = match class {
                0 => {
                    // Setosa
                    (
                        rng.random::<f32>() * 1.0 + 4.5, // sepal_length: 4.5-5.5
                        rng.random::<f32>() * 1.0 + 3.0, // sepal_width: 3.0-4.0
                        rng.random::<f32>() * 0.5 + 1.0, // petal_length: 1.0-1.5
                        rng.random::<f32>() * 0.3 + 0.1, // petal_width: 0.1-0.4
                    )
                }
                1 => {
                    // Versicolor
                    (
                        rng.random::<f32>() * 1.0 + 5.5, // sepal_length: 5.5-6.5
                        rng.random::<f32>() * 0.8 + 2.2, // sepal_width: 2.2-3.0
                        rng.random::<f32>() * 1.0 + 3.5, // petal_length: 3.5-4.5
                        rng.random::<f32>() * 0.5 + 1.0, // petal_width: 1.0-1.5
                    )
                }
                2 => {
                    // Virginica
                    (
                        rng.random::<f32>() * 1.0 + 6.0, // sepal_length: 6.0-7.0
                        rng.random::<f32>() * 0.8 + 2.5, // sepal_width: 2.5-3.3
                        rng.random::<f32>() * 1.5 + 4.5, // petal_length: 4.5-6.0
                        rng.random::<f32>() * 0.8 + 1.5, // petal_width: 1.5-2.3
                    )
                }
                _ => unreachable!(),
            };

            features.push(sepal_length);
            features.push(sepal_width);
            features.push(petal_length);
            features.push(petal_width);
        }

        let features_tensor = Tensor::from_vec(features, &[num_samples, num_features])?;
        let labels_tensor = Tensor::from_vec(labels, &[num_samples])?;

        Ok(IrisDataset {
            features: features_tensor,
            labels: labels_tensor,
            num_samples,
        })
    }
}

// Helper functions for pattern generation
fn create_circle_pattern(row: usize, col: usize, size: usize) -> f32 {
    let center = size as f32 / 2.0;
    let distance = ((row as f32 - center).powi(2) + (col as f32 - center).powi(2)).sqrt();
    let radius = size as f32 / 3.0;
    if distance < radius && distance > radius - 3.0 {
        0.8
    } else {
        0.1
    }
}

fn create_vertical_line_pattern(_row: usize, col: usize, size: usize) -> f32 {
    let center_col = size / 2;
    if col.abs_diff(center_col) < 2 {
        0.8
    } else {
        0.1
    }
}

fn create_horizontal_line_pattern(row: usize, _col: usize, size: usize) -> f32 {
    let center_row = size / 2;
    if row.abs_diff(center_row) < 2 {
        0.8
    } else {
        0.1
    }
}

fn create_diagonal_pattern(row: usize, col: usize, _size: usize) -> f32 {
    if row.abs_diff(col) < 2 {
        0.8
    } else {
        0.1
    }
}

fn create_cross_pattern(row: usize, col: usize, size: usize) -> f32 {
    let center_row = size / 2;
    let center_col = size / 2;
    if row.abs_diff(center_row) < 2 || col.abs_diff(center_col) < 2 {
        0.8
    } else {
        0.1
    }
}

fn create_square_pattern(row: usize, col: usize, size: usize) -> f32 {
    let margin = size / 4;
    if row >= margin && row < size - margin && col >= margin && col < size - margin {
        if row < margin + 2
            || row >= size - margin - 2
            || col < margin + 2
            || col >= size - margin - 2
        {
            0.8
        } else {
            0.1
        }
    } else {
        0.1
    }
}

fn create_rgb_gradient(row: usize, col: usize, size: usize) -> (f32, f32, f32) {
    let r = row as f32 / size as f32;
    let g = col as f32 / size as f32;
    let b = 1.0 - r;
    (r, g, b)
}

fn create_rgb_checkerboard(row: usize, col: usize) -> (f32, f32, f32) {
    if (row / 4 + col / 4) % 2 == 0 {
        (0.8, 0.8, 0.8)
    } else {
        (0.2, 0.2, 0.2)
    }
}

/// MNIST-like dataset
#[derive(Debug, Clone)]
pub struct MnistDataset<T> {
    images: Tensor<T>,
    labels: Tensor<T>,
    num_samples: usize,
}

impl<T: Clone + Default + num_traits::Zero + Send + Sync + 'static> Dataset<T> for MnistDataset<T> {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.num_samples {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index, self.num_samples
            )));
        }

        // Extract single image and label using slice operations from the imported slice function
        use tenflowers_core::ops::slice;

        let image_size = 28 * 28;

        // Create slice ranges for single sample
        let image_ranges = vec![index..index + 1, 0..image_size];
        #[allow(clippy::single_range_in_vec_init)]
        let label_ranges = vec![index..index + 1];

        let image_slice = slice(&self.images, &image_ranges)?;
        let label_slice = slice(&self.labels, &label_ranges)?;

        // Reshape to remove batch dimension
        let image = tenflowers_core::ops::reshape(&image_slice, &[image_size])?;
        let label = tenflowers_core::ops::reshape(&label_slice, &[])?;

        Ok((image, label))
    }
}

/// CIFAR-like dataset
#[derive(Debug, Clone)]
pub struct CifarDataset<T> {
    images: Tensor<T>,
    labels: Tensor<T>,
    num_samples: usize,
}

impl<T: Clone + Default + num_traits::Zero + Send + Sync + 'static> Dataset<T> for CifarDataset<T> {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.num_samples {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index, self.num_samples
            )));
        }

        // Extract single image and label using slice operations
        use tenflowers_core::ops::slice;

        // Create slice ranges for single sample (channels, height, width)
        let image_ranges = vec![index..index + 1, 0..3, 0..32, 0..32];
        #[allow(clippy::single_range_in_vec_init)]
        let label_ranges = vec![index..index + 1];

        let image_slice = slice(&self.images, &image_ranges)?;
        let label_slice = slice(&self.labels, &label_ranges)?;

        // Reshape to remove batch dimension
        let image = tenflowers_core::ops::reshape(&image_slice, &[3, 32, 32])?;
        let label = tenflowers_core::ops::reshape(&label_slice, &[])?;

        Ok((image, label))
    }
}

/// Iris-like dataset
#[derive(Debug, Clone)]
pub struct IrisDataset<T> {
    features: Tensor<T>,
    labels: Tensor<T>,
    num_samples: usize,
}

impl<T: Clone + Default + num_traits::Zero + Send + Sync + 'static> Dataset<T> for IrisDataset<T> {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.num_samples {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index, self.num_samples
            )));
        }

        // Extract single sample using slice operations
        use tenflowers_core::ops::slice;

        // Create slice ranges for single sample (4 features)
        let features_ranges = vec![index..index + 1, 0..4];
        #[allow(clippy::single_range_in_vec_init)]
        let label_ranges = vec![index..index + 1];

        let features_slice = slice(&self.features, &features_ranges)?;
        let label_slice = slice(&self.labels, &label_ranges)?;

        // Reshape to remove batch dimension
        let features = tenflowers_core::ops::reshape(&features_slice, &[4])?;
        let label = tenflowers_core::ops::reshape(&label_slice, &[])?;

        Ok((features, label))
    }
}

impl<T> MnistDataset<T> {
    /// Get the raw images tensor
    pub fn images(&self) -> &Tensor<T> {
        &self.images
    }

    /// Get the raw labels tensor
    pub fn labels(&self) -> &Tensor<T> {
        &self.labels
    }

    /// Get dataset information
    pub fn info(&self) -> DatasetInfo {
        DatasetInfo {
            name: "Synthetic MNIST".to_string(),
            num_samples: self.num_samples,
            num_classes: 10,
            image_shape: vec![28, 28],
            num_channels: 1,
        }
    }
}

impl<T> CifarDataset<T> {
    /// Get the raw images tensor
    pub fn images(&self) -> &Tensor<T> {
        &self.images
    }

    /// Get the raw labels tensor
    pub fn labels(&self) -> &Tensor<T> {
        &self.labels
    }

    /// Get dataset information
    pub fn info(&self) -> DatasetInfo {
        DatasetInfo {
            name: "Synthetic CIFAR-10".to_string(),
            num_samples: self.num_samples,
            num_classes: 10,
            image_shape: vec![32, 32],
            num_channels: 3,
        }
    }
}

impl<T> IrisDataset<T> {
    /// Get the raw features tensor
    pub fn features(&self) -> &Tensor<T> {
        &self.features
    }

    /// Get the raw labels tensor
    pub fn labels(&self) -> &Tensor<T> {
        &self.labels
    }

    /// Get dataset information
    pub fn info(&self) -> DatasetInfo {
        DatasetInfo {
            name: "Synthetic Iris".to_string(),
            num_samples: self.num_samples,
            num_classes: 3,
            image_shape: vec![4], // 4 features
            num_channels: 1,
        }
    }

    /// Get feature names
    pub fn feature_names(&self) -> Vec<String> {
        vec![
            "sepal_length".to_string(),
            "sepal_width".to_string(),
            "petal_length".to_string(),
            "petal_width".to_string(),
        ]
    }

    /// Get class names
    pub fn class_names(&self) -> Vec<String> {
        vec![
            "setosa".to_string(),
            "versicolor".to_string(),
            "virginica".to_string(),
        ]
    }
}

/// Information about a benchmark dataset
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub name: String,
    pub num_samples: usize,
    pub num_classes: usize,
    pub image_shape: Vec<usize>,
    pub num_channels: usize,
}

impl DatasetInfo {
    pub fn display(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("Dataset: {}\n", self.name));
        output.push_str(&format!("Samples: {}\n", self.num_samples));
        output.push_str(&format!("Classes: {}\n", self.num_classes));
        output.push_str(&format!("Shape: {:?}\n", self.image_shape));
        output.push_str(&format!("Channels: {}\n", self.num_channels));
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_mnist() {
        let dataset = BenchmarkDatasets::synthetic_mnist(100, Some(42)).unwrap();
        assert_eq!(dataset.len(), 100);

        let (image, label) = dataset.get(0).unwrap();
        assert_eq!(image.shape().dims(), &[784]); // 28*28
        assert_eq!(label.shape().dims(), &[] as &[usize]); // scalar

        let info = dataset.info();
        assert_eq!(info.name, "Synthetic MNIST");
        assert_eq!(info.num_classes, 10);
    }

    #[test]
    fn test_synthetic_cifar10() {
        let dataset = BenchmarkDatasets::synthetic_cifar10(50, Some(42)).unwrap();
        assert_eq!(dataset.len(), 50);

        let (image, label) = dataset.get(0).unwrap();
        assert_eq!(image.shape().dims(), &[3, 32, 32]); // RGB 32x32
        assert_eq!(label.shape().dims(), &[] as &[usize]); // scalar

        let info = dataset.info();
        assert_eq!(info.name, "Synthetic CIFAR-10");
        assert_eq!(info.num_classes, 10);
        assert_eq!(info.num_channels, 3);
    }

    #[test]
    fn test_synthetic_iris() {
        let dataset = BenchmarkDatasets::synthetic_iris(150, Some(42)).unwrap();
        assert_eq!(dataset.len(), 150);

        let (features, label) = dataset.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[4]); // 4 features
        assert_eq!(label.shape().dims(), &[] as &[usize]); // scalar

        let info = dataset.info();
        assert_eq!(info.name, "Synthetic Iris");
        assert_eq!(info.num_classes, 3);

        let feature_names = dataset.feature_names();
        assert_eq!(feature_names.len(), 4);
        assert!(feature_names.contains(&"sepal_length".to_string()));

        let class_names = dataset.class_names();
        assert_eq!(class_names.len(), 3);
        assert!(class_names.contains(&"setosa".to_string()));
    }

    #[test]
    fn test_dataset_reproducibility() {
        let dataset1 = BenchmarkDatasets::synthetic_mnist(10, Some(123)).unwrap();
        let dataset2 = BenchmarkDatasets::synthetic_mnist(10, Some(123)).unwrap();

        // With same seed, datasets should be identical
        let (img1, label1) = dataset1.get(0).unwrap();
        let (img2, label2) = dataset2.get(0).unwrap();

        assert_eq!(img1.shape(), img2.shape());
        assert_eq!(label1.shape(), label2.shape());

        // Check that the actual data is the same (this is a basic check)
        if let (Some(data1), Some(data2)) = (img1.as_slice(), img2.as_slice()) {
            assert_eq!(data1[0], data2[0]); // First pixel should be identical
        }
    }

    #[test]
    fn test_pattern_generation() {
        // Test that pattern functions don't panic and return valid values
        let circle = create_circle_pattern(14, 14, 28);
        assert!(circle >= 0.0 && circle <= 1.0);

        let line = create_vertical_line_pattern(10, 14, 28);
        assert!(line >= 0.0 && line <= 1.0);

        let (r, g, b) = create_rgb_gradient(16, 16, 32);
        assert!(r >= 0.0 && r <= 1.0);
        assert!(g >= 0.0 && g <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }
}
