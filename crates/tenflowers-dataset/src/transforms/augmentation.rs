//! Data augmentation transformations
//!
//! This module provides advanced data augmentation techniques commonly used
//! to improve model generalization and robustness in machine learning.

use crate::transforms::Transform;
use scirs2_core::random::Rng;
use std::marker::PhantomData;
use tenflowers_core::{Result, Tensor, TensorError};

/// CutMix augmentation - combines two samples by cutting and pasting regions
pub struct CutMix<T> {
    alpha: f32,
    probability: f32,
    _phantom: PhantomData<T>,
}

impl<T> CutMix<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(alpha: f32, probability: f32) -> Self {
        Self {
            alpha: alpha.max(0.0),
            probability: probability.clamp(0.0, 1.0),
            _phantom: PhantomData,
        }
    }

    /// Generate a random bounding box for CutMix
    fn generate_cutmix_bbox(
        &self,
        width: usize,
        height: usize,
        lambda: f32,
    ) -> (usize, usize, usize, usize) {
        let mut rng = scirs2_core::random::rng();

        let cut_ratio = (1.0 - lambda).sqrt();
        let cut_w = (width as f32 * cut_ratio) as usize;
        let cut_h = (height as f32 * cut_ratio) as usize;

        let cx = rng.gen_range(0..width);
        let cy = rng.gen_range(0..height);

        let x1 = (cx.saturating_sub(cut_w / 2)).min(width);
        let y1 = (cy.saturating_sub(cut_h / 2)).min(height);
        let x2 = (cx + cut_w / 2).min(width);
        let y2 = (cy + cut_h / 2).min(height);

        (x1, y1, x2, y2)
    }

    /// Apply CutMix to two samples
    pub fn apply_cutmix(
        &self,
        sample1: (Tensor<T>, Tensor<T>),
        sample2: (Tensor<T>, Tensor<T>),
    ) -> Result<(Tensor<T>, Tensor<T>)> {
        let mut rng = scirs2_core::random::rng();

        if rng.random::<f32>() >= self.probability {
            return Ok(sample1);
        }

        let (features1, labels1) = sample1;
        let (features2, labels2) = sample2;

        let shape = features1.shape().dims();

        // Assume image format [channels, height, width] or [height, width, channels]
        if shape.len() < 2 {
            return Ok((features1, labels1)); // Skip non-image data
        }

        let (channels, height, width) = if shape.len() == 3 {
            if shape[0] <= 4 {
                (shape[0], shape[1], shape[2])
            } else {
                (shape[2], shape[0], shape[1])
            }
        } else {
            (1, shape[0], shape[1])
        };

        // Generate lambda from Beta distribution
        let lambda = if self.alpha > 0.0 {
            // Simplified beta distribution sampling
            let u1: f32 = rng.random();
            let u2: f32 = rng.random();
            let x = u1.powf(1.0 / self.alpha);
            let y = u2.powf(1.0 / self.alpha);
            x / (x + y)
        } else {
            0.5
        };

        let (x1, y1, x2, y2) = self.generate_cutmix_bbox(width, height, lambda);

        let data1 = features1.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;
        let data2 = features2.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let mut mixed_data = data1.to_vec();

        // Copy region from second image
        for h in y1..y2 {
            for w in x1..x2 {
                for c in 0..channels {
                    let idx = if shape.len() == 3 && shape[0] <= 4 {
                        // [channels, height, width]
                        c * height * width + h * width + w
                    } else {
                        // [height, width, channels] or [height, width]
                        h * width * channels + w * channels + c
                    };

                    if idx < mixed_data.len() && idx < data2.len() {
                        mixed_data[idx] = data2[idx];
                    }
                }
            }
        }

        let mixed_features = Tensor::from_vec(mixed_data, shape)?;

        // Mix labels based on area ratio
        let total_area = (width * height) as f32;
        let cut_area = ((x2 - x1) * (y2 - y1)) as f32;
        let actual_lambda = 1.0 - (cut_area / total_area);

        let label1_data = labels1.as_slice().unwrap_or(&[]);
        let label2_data = labels2.as_slice().unwrap_or(&[]);

        let mixed_label_data: Vec<T> = label1_data
            .iter()
            .zip(label2_data.iter())
            .map(|(&l1, &l2)| {
                l1 * T::from(actual_lambda).unwrap() + l2 * T::from(1.0 - actual_lambda).unwrap()
            })
            .collect();

        let mixed_labels = Tensor::from_vec(mixed_label_data, labels1.shape().dims())?;

        Ok((mixed_features, mixed_labels))
    }
}

impl<T> Transform<T> for CutMix<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        // For single sample, CutMix needs another sample
        // In a real implementation, this would get another random sample from the dataset
        // For now, we'll just return the original sample
        Ok(sample)
    }
}

/// MixUp augmentation - linear interpolation between two samples
pub struct MixUp<T> {
    alpha: f32,
    probability: f32,
    _phantom: PhantomData<T>,
}

impl<T> MixUp<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(alpha: f32, probability: f32) -> Self {
        Self {
            alpha: alpha.max(0.0),
            probability: probability.clamp(0.0, 1.0),
            _phantom: PhantomData,
        }
    }

    /// Apply MixUp to two samples
    pub fn apply_mixup(
        &self,
        sample1: (Tensor<T>, Tensor<T>),
        sample2: (Tensor<T>, Tensor<T>),
    ) -> Result<(Tensor<T>, Tensor<T>)> {
        let mut rng = scirs2_core::random::rng();

        if rng.random::<f32>() >= self.probability {
            return Ok(sample1);
        }

        let (features1, labels1) = sample1;
        let (features2, labels2) = sample2;

        // Generate lambda from Beta distribution
        let lambda = if self.alpha > 0.0 {
            // Simplified beta distribution sampling
            let u1: f32 = rng.random();
            let u2: f32 = rng.random();
            let x = u1.powf(1.0 / self.alpha);
            let y = u2.powf(1.0 / self.alpha);
            x / (x + y)
        } else {
            0.5
        };

        let lambda_t = T::from(lambda).unwrap();
        let one_minus_lambda = T::from(1.0 - lambda).unwrap();

        // Mix features
        let data1 = features1.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;
        let data2 = features2.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let mixed_feature_data: Vec<T> = data1
            .iter()
            .zip(data2.iter())
            .map(|(&f1, &f2)| f1 * lambda_t + f2 * one_minus_lambda)
            .collect();

        let mixed_features = Tensor::from_vec(mixed_feature_data, features1.shape().dims())?;

        // Mix labels
        let label1_data = labels1.as_slice().unwrap_or(&[]);
        let label2_data = labels2.as_slice().unwrap_or(&[]);

        let mixed_label_data: Vec<T> = label1_data
            .iter()
            .zip(label2_data.iter())
            .map(|(&l1, &l2)| l1 * lambda_t + l2 * one_minus_lambda)
            .collect();

        let mixed_labels = Tensor::from_vec(mixed_label_data, labels1.shape().dims())?;

        Ok((mixed_features, mixed_labels))
    }
}

impl<T> Transform<T> for MixUp<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        // For single sample, MixUp needs another sample
        // In a real implementation, this would get another random sample from the dataset
        // For now, we'll just return the original sample
        Ok(sample)
    }
}

/// Cutout augmentation - randomly mask rectangular regions
pub struct Cutout<T> {
    cutout_size: usize,
    probability: f32,
    fill_value: T,
    _phantom: PhantomData<T>,
}

impl<T> Cutout<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(cutout_size: usize, probability: f32, fill_value: T) -> Self {
        Self {
            cutout_size,
            probability: probability.clamp(0.0, 1.0),
            fill_value,
            _phantom: PhantomData,
        }
    }

    pub fn default_cutout(cutout_size: usize) -> Self {
        Self::new(cutout_size, 0.5, T::zero())
    }
}

impl<T> Transform<T> for Cutout<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let mut rng = scirs2_core::random::rng();

        if rng.random::<f32>() >= self.probability {
            return Ok(sample);
        }

        let (features, labels) = sample;
        let shape = features.shape().dims();

        if shape.len() < 2 {
            return Ok((features, labels));
        }

        let (channels, height, width) = if shape.len() == 3 {
            if shape[0] <= 4 {
                (shape[0], shape[1], shape[2])
            } else {
                (shape[2], shape[0], shape[1])
            }
        } else {
            (1, shape[0], shape[1])
        };

        let data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;
        let mut cutout_data = data.to_vec();

        // Generate random cutout position
        let cutout_x = rng.gen_range(0..width.saturating_sub(self.cutout_size));
        let cutout_y = rng.gen_range(0..height.saturating_sub(self.cutout_size));

        // Apply cutout
        for h in cutout_y..(cutout_y + self.cutout_size).min(height) {
            for w in cutout_x..(cutout_x + self.cutout_size).min(width) {
                for c in 0..channels {
                    let idx = if shape.len() == 3 && shape[0] <= 4 {
                        c * height * width + h * width + w
                    } else {
                        h * width * channels + w * channels + c
                    };

                    if idx < cutout_data.len() {
                        cutout_data[idx] = self.fill_value;
                    }
                }
            }
        }

        let cutout_features = Tensor::from_vec(cutout_data, shape)?;
        Ok((cutout_features, labels))
    }
}

/// Random erasing augmentation - randomly mask rectangular regions with random values
pub struct RandomErasing<T> {
    probability: f32,
    area_ratio_range: (f32, f32),
    aspect_ratio_range: (f32, f32),
    fill_mode: FillMode<T>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub enum FillMode<T> {
    /// Fill with constant value
    Constant(T),
    /// Fill with random values in range
    Random(T, T),
    /// Fill with mean pixel value
    Mean,
}

impl<T> RandomErasing<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(
        probability: f32,
        area_ratio_range: (f32, f32),
        aspect_ratio_range: (f32, f32),
        fill_mode: FillMode<T>,
    ) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
            area_ratio_range: (area_ratio_range.0.max(0.0), area_ratio_range.1.min(1.0)),
            aspect_ratio_range,
            fill_mode,
            _phantom: PhantomData,
        }
    }

    pub fn default_random_erasing() -> Self {
        Self::new(
            0.5,
            (0.02, 0.33),
            (0.3, 3.3),
            FillMode::Random(T::zero(), T::one()),
        )
    }

    fn get_fill_value(&self, data: &[T]) -> T {
        let mut rng = scirs2_core::random::rng();

        match &self.fill_mode {
            FillMode::Constant(value) => *value,
            FillMode::Random(min_val, max_val) => {
                let random = rng.random::<f32>();
                let min_f = min_val.to_f32().unwrap_or(0.0);
                let max_f = max_val.to_f32().unwrap_or(1.0);
                let val = min_f + random * (max_f - min_f);
                T::from(val).unwrap_or(T::zero())
            }
            FillMode::Mean => {
                if data.is_empty() {
                    T::zero()
                } else {
                    let sum = data.iter().fold(T::zero(), |acc, &x| acc + x);
                    sum / T::from(data.len()).unwrap()
                }
            }
        }
    }
}

impl<T> Transform<T> for RandomErasing<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let mut rng = scirs2_core::random::rng();

        if rng.random::<f32>() >= self.probability {
            return Ok(sample);
        }

        let (features, labels) = sample;
        let shape = features.shape().dims();

        if shape.len() < 2 {
            return Ok((features, labels));
        }

        let (channels, height, width) = if shape.len() == 3 {
            if shape[0] <= 4 {
                (shape[0], shape[1], shape[2])
            } else {
                (shape[2], shape[0], shape[1])
            }
        } else {
            (1, shape[0], shape[1])
        };

        let data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;
        let mut erased_data = data.to_vec();

        let total_area = (height * width) as f32;

        // Try multiple times to find valid rectangle
        for _ in 0..10 {
            let area_ratio = rng.gen_range(self.area_ratio_range.0..=self.area_ratio_range.1);
            let erase_area = (total_area * area_ratio) as usize;

            let aspect_ratio = rng.gen_range(self.aspect_ratio_range.0..=self.aspect_ratio_range.1);
            let h = ((erase_area as f32 * aspect_ratio).sqrt()) as usize;
            let w = (erase_area / h.max(1)) as usize;

            if h < height && w < width {
                let y = rng.gen_range(0..=(height - h));
                let x = rng.gen_range(0..=(width - w));

                let fill_value = self.get_fill_value(data);

                // Apply erasing
                for cur_h in y..(y + h) {
                    for cur_w in x..(x + w) {
                        for c in 0..channels {
                            let idx = if shape.len() == 3 && shape[0] <= 4 {
                                c * height * width + cur_h * width + cur_w
                            } else {
                                cur_h * width * channels + cur_w * channels + c
                            };

                            if idx < erased_data.len() {
                                erased_data[idx] = fill_value;
                            }
                        }
                    }
                }
                break;
            }
        }

        let erased_features = Tensor::from_vec(erased_data, shape)?;
        Ok((erased_features, labels))
    }
}

/// AutoAugment - a collection of learned augmentation policies
pub struct AutoAugment<T> {
    policies: Vec<AutoAugmentPolicy<T>>,
    _phantom: PhantomData<T>,
}

#[derive(Clone)]
pub struct AutoAugmentPolicy<T> {
    operations: Vec<(AutoAugmentOp, f32, f32)>, // (operation, probability, magnitude)
    _phantom: PhantomData<T>,
}

#[derive(Clone)]
pub enum AutoAugmentOp {
    /// Rotate image
    Rotate,
    /// Shear X axis
    ShearX,
    /// Shear Y axis
    ShearY,
    /// Translate X axis
    TranslateX,
    /// Translate Y axis
    TranslateY,
    /// Adjust brightness
    Brightness,
    /// Adjust contrast
    Contrast,
    /// Adjust saturation
    Saturation,
    /// Adjust hue
    Hue,
}

impl<T> AutoAugment<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(policies: Vec<AutoAugmentPolicy<T>>) -> Self {
        Self {
            policies,
            _phantom: PhantomData,
        }
    }

    /// Create ImageNet AutoAugment policies
    pub fn imagenet_policies() -> Self {
        let policies = vec![
            AutoAugmentPolicy {
                operations: vec![
                    (AutoAugmentOp::Rotate, 0.9, 0.4),
                    (AutoAugmentOp::Brightness, 0.8, 0.6),
                ],
                _phantom: PhantomData,
            },
            AutoAugmentPolicy {
                operations: vec![
                    (AutoAugmentOp::ShearX, 0.5, 0.3),
                    (AutoAugmentOp::Contrast, 0.7, 0.5),
                ],
                _phantom: PhantomData,
            },
            // Add more policies as needed...
        ];

        Self::new(policies)
    }

    fn select_policy(&self) -> &AutoAugmentPolicy<T> {
        let mut rng = scirs2_core::random::rng();
        let idx = rng.gen_range(0..self.policies.len());
        &self.policies[idx]
    }

    fn apply_operation(
        &self,
        sample: (Tensor<T>, Tensor<T>),
        op: &AutoAugmentOp,
        magnitude: f32,
    ) -> Result<(Tensor<T>, Tensor<T>)> {
        // This would implement the actual augmentation operations
        // For now, we'll just return the sample unchanged
        match op {
            AutoAugmentOp::Rotate => Ok(sample),
            AutoAugmentOp::ShearX => Ok(sample),
            AutoAugmentOp::ShearY => Ok(sample),
            AutoAugmentOp::TranslateX => Ok(sample),
            AutoAugmentOp::TranslateY => Ok(sample),
            AutoAugmentOp::Brightness => Ok(sample),
            AutoAugmentOp::Contrast => Ok(sample),
            AutoAugmentOp::Saturation => Ok(sample),
            AutoAugmentOp::Hue => Ok(sample),
        }
    }
}

impl<T> Transform<T> for AutoAugment<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn apply(&self, mut sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        if self.policies.is_empty() {
            return Ok(sample);
        }

        let policy = self.select_policy();
        let mut rng = scirs2_core::random::rng();

        for (op, probability, magnitude) in &policy.operations {
            if rng.random::<f32>() < *probability {
                sample = self.apply_operation(sample, op, *magnitude)?;
            }
        }

        Ok(sample)
    }
}

/// RandAugment - randomized augmentation with uniform sampling
pub struct RandAugment<T> {
    n_ops: usize,
    magnitude: f32,
    operations: Vec<AutoAugmentOp>,
    _phantom: PhantomData<T>,
}

impl<T> RandAugment<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(n_ops: usize, magnitude: f32) -> Self {
        let operations = vec![
            AutoAugmentOp::Rotate,
            AutoAugmentOp::ShearX,
            AutoAugmentOp::ShearY,
            AutoAugmentOp::TranslateX,
            AutoAugmentOp::TranslateY,
            AutoAugmentOp::Brightness,
            AutoAugmentOp::Contrast,
            AutoAugmentOp::Saturation,
            AutoAugmentOp::Hue,
        ];

        Self {
            n_ops,
            magnitude: magnitude.clamp(0.0, 30.0),
            operations,
            _phantom: PhantomData,
        }
    }
}

impl<T> Transform<T> for RandAugment<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn apply(&self, mut sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let mut rng = scirs2_core::random::rng();

        for _ in 0..self.n_ops {
            let op_idx = rng.gen_range(0..self.operations.len());
            let op = &self.operations[op_idx];

            // Apply operation with fixed magnitude
            // In a full implementation, this would apply the actual transformation
            // For now, we just pass through the sample unchanged
            let _op = op; // Suppress unused warning
        }

        Ok(sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cutout() {
        let cutout = Cutout::default_cutout(16);
        let features = Tensor::<f32>::zeros(&[3, 32, 32]);
        let labels = Tensor::<f32>::zeros(&[1]);

        let result = cutout.apply((features, labels));
        assert!(result.is_ok());
    }

    #[test]
    fn test_random_erasing() {
        let erasing = RandomErasing::default_random_erasing();
        let features = Tensor::<f32>::zeros(&[3, 32, 32]);
        let labels = Tensor::<f32>::zeros(&[1]);

        let result = erasing.apply((features, labels));
        assert!(result.is_ok());
    }

    #[test]
    fn test_mixup() {
        let mixup = MixUp::new(1.0, 1.0);
        let features1 = Tensor::<f32>::zeros(&[10]);
        let labels1 = Tensor::<f32>::zeros(&[1]);
        let features2 = Tensor::<f32>::ones(&[10]);
        let labels2 = Tensor::<f32>::ones(&[1]);

        let result = mixup.apply_mixup((features1, labels1), (features2, labels2));
        assert!(result.is_ok());
    }

    #[test]
    fn test_cutmix() {
        let cutmix = CutMix::new(1.0, 1.0);
        let features1 = Tensor::<f32>::zeros(&[3, 32, 32]);
        let labels1 = Tensor::<f32>::zeros(&[1]);
        let features2 = Tensor::<f32>::ones(&[3, 32, 32]);
        let labels2 = Tensor::<f32>::ones(&[1]);

        let result = cutmix.apply_cutmix((features1, labels1), (features2, labels2));
        assert!(result.is_ok());
    }

    #[test]
    fn test_autoaugment() {
        let autoaugment = AutoAugment::imagenet_policies();
        let features = Tensor::<f32>::zeros(&[3, 32, 32]);
        let labels = Tensor::<f32>::zeros(&[1]);

        let result = autoaugment.apply((features, labels));
        assert!(result.is_ok());
    }
}
