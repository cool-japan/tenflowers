//! Normalization transformations for datasets

use crate::{transforms::Transform, Dataset};
use tenflowers_core::{Result, Tensor, TensorError};

/// Normalize features by subtracting mean and dividing by standard deviation
pub struct Normalize<T> {
    mean: Vec<T>,
    std: Vec<T>,
}

impl<T> Normalize<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(mean: Vec<T>, std: Vec<T>) -> Result<Self> {
        if mean.len() != std.len() {
            return Err(TensorError::invalid_argument(
                "Mean and std vectors must have the same length".to_string(),
            ));
        }
        Ok(Self { mean, std })
    }

    /// Compute normalization parameters from a dataset
    pub fn from_dataset<D: Dataset<T>>(dataset: &D) -> Result<Self> {
        if dataset.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot compute normalization from empty dataset".to_string(),
            ));
        }

        // Get first sample to determine feature dimension
        let (first_features, _) = dataset.get(0)?;
        let feature_dim = first_features.shape().size();

        let mut feature_sums = vec![T::zero(); feature_dim];
        let mut feature_sq_sums = vec![T::zero(); feature_dim];
        let n = T::from(dataset.len()).unwrap();

        // Compute means and variances
        for i in 0..dataset.len() {
            let (features, _) = dataset.get(i)?;

            // Flatten features to 1D for computation
            let flat_features = tenflowers_core::ops::reshape(&features, &[feature_dim])?;

            for j in 0..feature_dim {
                if let Some(val) = flat_features.get(&[j]) {
                    feature_sums[j] = feature_sums[j] + val;
                    feature_sq_sums[j] = feature_sq_sums[j] + val * val;
                }
            }
        }

        let mut means = Vec::new();
        let mut stds = Vec::new();

        for i in 0..feature_dim {
            let mean = feature_sums[i] / n;
            let variance = (feature_sq_sums[i] / n) - (mean * mean);
            let std = variance.sqrt();

            means.push(mean);
            stds.push(std);
        }

        Self::new(means, stds)
    }

    /// Create mean tensor for given feature dimension
    fn create_mean_tensor(&self, feature_dim: usize) -> Result<Tensor<T>> {
        // Extend or truncate mean vector to match feature dimension
        let mut mean_vec = self.mean.clone();
        match mean_vec.len().cmp(&feature_dim) {
            std::cmp::Ordering::Less => {
                // Repeat last value if we need more elements
                if let Some(last_val) = mean_vec.last() {
                    mean_vec.resize(feature_dim, *last_val);
                } else {
                    mean_vec.resize(feature_dim, T::zero());
                }
            }
            std::cmp::Ordering::Greater => {
                // Truncate if we have too many elements
                mean_vec.truncate(feature_dim);
            }
            std::cmp::Ordering::Equal => {
                // Perfect match, no changes needed
            }
        }
        Tensor::from_vec(mean_vec, &[feature_dim])
    }

    /// Create std tensor for given feature dimension
    fn create_std_tensor(&self, feature_dim: usize) -> Result<Tensor<T>> {
        // Extend or truncate std vector to match feature dimension
        let mut std_vec = self.std.clone();
        match std_vec.len().cmp(&feature_dim) {
            std::cmp::Ordering::Less => {
                // Repeat last value if we need more elements
                if let Some(last_val) = std_vec.last() {
                    std_vec.resize(feature_dim, *last_val);
                } else {
                    std_vec.resize(feature_dim, T::one());
                }
            }
            std::cmp::Ordering::Greater => {
                // Truncate if we have too many elements
                std_vec.truncate(feature_dim);
            }
            std::cmp::Ordering::Equal => {
                // Perfect match, no changes needed
            }
        }
        Tensor::from_vec(std_vec, &[feature_dim])
    }
}

impl<T> Transform<T> for Normalize<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let (features, labels) = sample;
        let original_shape = features.shape().dims().to_vec();
        let feature_dim = features.shape().size();

        // Flatten features for normalization
        let flat_features = tenflowers_core::ops::reshape(&features, &[feature_dim])?;

        // Create mean and std tensors using optimized helper methods
        let mean_tensor = self.create_mean_tensor(feature_dim)?;
        let std_tensor = self.create_std_tensor(feature_dim)?;

        // Normalize: (x - mean) / std
        let centered = flat_features.sub(&mean_tensor)?;
        let normalized = centered.div(&std_tensor)?;

        // Reshape back to original shape
        let normalized_features = tenflowers_core::ops::reshape(&normalized, &original_shape)?;

        Ok((normalized_features, labels))
    }
}

/// Scale features to a specific range [min_val, max_val]
pub struct MinMaxScale<T> {
    data_min: Vec<T>,
    data_max: Vec<T>,
    feature_range: (T, T),
}

impl<T> MinMaxScale<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(data_min: Vec<T>, data_max: Vec<T>, feature_range: (T, T)) -> Result<Self> {
        if data_min.len() != data_max.len() {
            return Err(TensorError::invalid_argument(
                "Data min and max vectors must have the same length".to_string(),
            ));
        }
        Ok(Self {
            data_min,
            data_max,
            feature_range,
        })
    }

    /// Compute scaling parameters from a dataset
    pub fn from_dataset<D: Dataset<T>>(dataset: &D, feature_range: (T, T)) -> Result<Self> {
        if dataset.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot compute scaling from empty dataset".to_string(),
            ));
        }

        // Get first sample to determine feature dimension
        let (first_features, _) = dataset.get(0)?;
        let feature_dim = first_features.shape().size();

        let mut data_min = vec![T::infinity(); feature_dim];
        let mut data_max = vec![T::neg_infinity(); feature_dim];

        // Find min and max values for each feature
        for i in 0..dataset.len() {
            let (features, _) = dataset.get(i)?;
            let flat_features = tenflowers_core::ops::reshape(&features, &[feature_dim])?;

            for j in 0..feature_dim {
                if let Some(val) = flat_features.get(&[j]) {
                    if val < data_min[j] {
                        data_min[j] = val;
                    }
                    if val > data_max[j] {
                        data_max[j] = val;
                    }
                }
            }
        }

        Self::new(data_min, data_max, feature_range)
    }
}

impl<T> Transform<T> for MinMaxScale<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let (features, labels) = sample;
        let original_shape = features.shape().dims().to_vec();
        let feature_dim = features.shape().size();

        // Flatten features for scaling
        let flat_features = tenflowers_core::ops::reshape(&features, &[feature_dim])?;

        let mut scaled_data = Vec::with_capacity(feature_dim);
        let (min_range, max_range) = self.feature_range;
        let range_scale = max_range - min_range;

        for i in 0..feature_dim {
            if let Some(val) = flat_features.get(&[i]) {
                let data_range = self.data_max[i] - self.data_min[i];
                let scaled = if data_range == T::zero() {
                    min_range // If no variance, set to min of range
                } else {
                    min_range + (val - self.data_min[i]) / data_range * range_scale
                };
                scaled_data.push(scaled);
            } else {
                return Err(TensorError::invalid_argument(
                    "Failed to get feature value".to_string(),
                ));
            }
        }

        let scaled_tensor = Tensor::from_vec(scaled_data, &[feature_dim])?;
        let scaled_features = tenflowers_core::ops::reshape(&scaled_tensor, &original_shape)?;

        Ok((scaled_features, labels))
    }
}

/// Robust scaler using median and IQR instead of mean and std
pub struct RobustScaler<T> {
    medians: Vec<T>,
    iqrs: Vec<T>,
}

impl<T> RobustScaler<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(medians: Vec<T>, iqrs: Vec<T>) -> Result<Self> {
        if medians.len() != iqrs.len() {
            return Err(TensorError::invalid_argument(
                "Medians and IQRs vectors must have the same length".to_string(),
            ));
        }
        Ok(Self { medians, iqrs })
    }

    /// Compute robust scaling parameters from a dataset
    pub fn from_dataset<D: Dataset<T>>(dataset: &D) -> Result<Self> {
        if dataset.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot compute robust scaling from empty dataset".to_string(),
            ));
        }

        // Get first sample to determine feature dimension
        let (first_features, _) = dataset.get(0)?;
        let feature_dim = first_features.shape().size();

        // Collect all feature values by dimension
        let mut feature_values: Vec<Vec<T>> = vec![Vec::new(); feature_dim];

        for i in 0..dataset.len() {
            let (features, _) = dataset.get(i)?;
            let flat_features = tenflowers_core::ops::reshape(&features, &[feature_dim])?;

            for (j, feature_value) in feature_values.iter_mut().enumerate().take(feature_dim) {
                if let Some(val) = flat_features.get(&[j]) {
                    feature_value.push(val);
                }
            }
        }

        // Compute medians and IQRs for each dimension
        let mut medians = Vec::new();
        let mut iqrs = Vec::new();

        for values in feature_values {
            if values.is_empty() {
                medians.push(T::zero());
                iqrs.push(T::one());
                continue;
            }

            let mut sorted_values = values;
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let n = sorted_values.len();
            let median = if n % 2 == 0 {
                (sorted_values[n / 2 - 1] + sorted_values[n / 2]) / T::from(2.0).unwrap()
            } else {
                sorted_values[n / 2]
            };

            let q1_idx = n / 4;
            let q3_idx = (3 * n) / 4;
            let q1 = sorted_values[q1_idx];
            let q3 = sorted_values[q3_idx];
            let iqr = q3 - q1;

            medians.push(median);
            iqrs.push(if iqr > T::zero() { iqr } else { T::one() });
        }

        Ok(Self { medians, iqrs })
    }
}

impl<T> Transform<T> for RobustScaler<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let (features, labels) = sample;
        let original_shape = features.shape().dims().to_vec();
        let feature_dim = features.shape().size();

        // Flatten features for normalization
        let flat_features = tenflowers_core::ops::reshape(&features, &[feature_dim])?;

        // Apply robust scaling: (x - median) / IQR
        let mut scaled_data = Vec::new();
        for i in 0..feature_dim {
            let idx = i % self.medians.len();
            if let Some(val) = flat_features.get(&[i]) {
                let scaled = (val - self.medians[idx]) / self.iqrs[idx];
                scaled_data.push(scaled);
            }
        }

        let scaled_tensor = Tensor::from_vec(scaled_data, &[feature_dim])?;
        let reshaped_features = tenflowers_core::ops::reshape(&scaled_tensor, &original_shape)?;

        Ok((reshaped_features, labels))
    }
}

/// Per-channel normalization for multi-channel data (e.g., RGB images)
pub struct PerChannelNormalize<T> {
    channel_means: Vec<T>,
    channel_stds: Vec<T>,
}

impl<T> PerChannelNormalize<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(channel_means: Vec<T>, channel_stds: Vec<T>) -> Result<Self> {
        if channel_means.len() != channel_stds.len() {
            return Err(TensorError::invalid_argument(
                "Channel means and stds must have the same length".to_string(),
            ));
        }
        Ok(Self {
            channel_means,
            channel_stds,
        })
    }

    /// Common ImageNet normalization values
    pub fn imagenet() -> Self {
        Self {
            channel_means: vec![
                T::from(0.485).unwrap(),
                T::from(0.456).unwrap(),
                T::from(0.406).unwrap(),
            ],
            channel_stds: vec![
                T::from(0.229).unwrap(),
                T::from(0.224).unwrap(),
                T::from(0.225).unwrap(),
            ],
        }
    }
}

impl<T> Transform<T> for PerChannelNormalize<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let (features, labels) = sample;
        let shape = features.shape().dims();

        // Assume features are in format [channels, height, width] or [channels, ...]
        if shape.is_empty() {
            return Ok((features, labels));
        }

        let channels = shape[0];
        if channels != self.channel_means.len() {
            return Err(TensorError::invalid_argument(format!(
                "Expected {} channels, got {}",
                self.channel_means.len(),
                channels
            )));
        }

        let data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;
        let mut normalized_data = Vec::new();

        let channel_size = data.len() / channels;

        for c in 0..channels {
            let start = c * channel_size;
            let end = start + channel_size;

            for value in data.iter().skip(start).take(end - start) {
                let normalized = (*value - self.channel_means[c]) / self.channel_stds[c];
                normalized_data.push(normalized);
            }
        }

        let normalized_tensor = Tensor::from_vec(normalized_data, shape)?;
        Ok((normalized_tensor, labels))
    }
}

/// Global normalization across all samples in the dataset
pub struct GlobalNormalize<T> {
    global_mean: T,
    global_std: T,
}

impl<T> GlobalNormalize<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(global_mean: T, global_std: T) -> Self {
        Self {
            global_mean,
            global_std,
        }
    }

    /// Compute global normalization parameters from a dataset
    pub fn from_dataset<D: Dataset<T>>(dataset: &D) -> Result<Self> {
        if dataset.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot compute global normalization from empty dataset".to_string(),
            ));
        }

        let mut total_sum = T::zero();
        let mut total_sq_sum = T::zero();
        let mut total_count = 0;

        for i in 0..dataset.len() {
            let (features, _) = dataset.get(i)?;
            let data = features.as_slice().ok_or_else(|| {
                TensorError::invalid_argument(
                    "Cannot access tensor data (GPU tensor not supported)".to_string(),
                )
            })?;

            for &val in data {
                total_sum = total_sum + val;
                total_sq_sum = total_sq_sum + val * val;
                total_count += 1;
            }
        }

        let n = T::from(total_count).unwrap();
        let mean = total_sum / n;
        let variance = (total_sq_sum / n) - (mean * mean);
        let std = variance.sqrt();

        Ok(Self {
            global_mean: mean,
            global_std: if std > T::zero() { std } else { T::one() },
        })
    }
}

impl<T> Transform<T> for GlobalNormalize<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let (features, labels) = sample;
        let shape = features.shape().dims();
        let data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let normalized_data: Vec<T> = data
            .iter()
            .map(|&val| (val - self.global_mean) / self.global_std)
            .collect();

        let normalized_tensor = Tensor::from_vec(normalized_data, shape)?;
        Ok((normalized_tensor, labels))
    }
}
