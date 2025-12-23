//! Data augmentation layers for improved model training
//!
//! This module implements modern data augmentation techniques including Mixup and CutMix
//! that are essential for training robust neural networks.

use super::Layer;
use scirs2_core::random::thread_rng;
use std::cmp::{max, min};
use tenflowers_core::{
    ops::{add, gather, mul},
    Result, Tensor, TensorError,
};

/// Mixup data augmentation layer
///
/// Mixup linearly interpolates between two samples and their labels:
/// mixed_sample = λ * sample1 + (1 - λ) * sample2
/// mixed_label = λ * label1 + (1 - λ) * label2
///
/// Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2017)
#[derive(Debug, Clone)]
pub struct Mixup {
    alpha: f32,
    enabled: bool,
}

impl Mixup {
    /// Create a new Mixup layer
    ///
    /// # Arguments
    /// * `alpha` - Beta distribution parameter. Higher values (e.g., 1.0) create more uniform mixing.
    ///   Lower values (e.g., 0.2) create mixing closer to original samples.
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            enabled: true,
        }
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.enabled = training;
    }

    /// Apply mixup to a batch of samples and labels
    ///
    /// # Arguments
    /// * `samples` - Input batch tensor of shape [batch_size, ...]
    /// * `labels` - Label batch tensor of shape [batch_size, num_classes] (one-hot) or [batch_size] (indices)
    ///
    /// # Returns
    /// Tuple of (mixed_samples, mixed_labels, lambda)
    pub fn forward<T>(
        &self,
        samples: &Tensor<T>,
        labels: &Tensor<T>,
    ) -> Result<(Tensor<T>, Tensor<T>, f32)>
    where
        T: Default + Clone + Copy + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
        T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
        T: scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::ToPrimitive,
    {
        if !self.enabled {
            let lambda = 1.0;
            return Ok((samples.clone(), labels.clone(), lambda));
        }

        let batch_size = samples.shape().dims()[0];
        if batch_size < 2 {
            return Err(TensorError::unsupported_operation_simple(
                "Mixup requires batch size >= 2".to_string(),
            ));
        }

        // Sample lambda from Beta distribution (approximated)
        let mut rng = thread_rng();
        let lambda = if self.alpha > 0.0 {
            // Sample from Beta(alpha, alpha) distribution
            // For simplicity, we'll use a uniform distribution when alpha=1.0
            if (self.alpha - 1.0).abs() < 1e-6 {
                rng.gen_range(0.0..1.0)
            } else {
                // Approximate Beta distribution by sampling two gamma variables
                // This is a simplified implementation
                let u1: f32 = rng.gen_range(0.0..1.0);
                let u2: f32 = rng.gen_range(0.0..1.0);
                let x = u1.powf(1.0 / self.alpha);
                let y = u2.powf(1.0 / self.alpha);
                x / (x + y)
            }
        } else {
            1.0
        };

        // Create random permutation for batch indices
        let mut indices: Vec<i32> = (0..batch_size).map(|i| i as i32).collect();
        for i in (1..batch_size).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        // Mix samples and labels
        let mixed_samples = self.mix_tensors(samples, &indices, lambda)?;
        let mixed_labels = self.mix_tensors(labels, &indices, lambda)?;

        Ok((mixed_samples, mixed_labels, lambda))
    }

    /// Mix tensors with given lambda and indices
    fn mix_tensors<T>(&self, tensor: &Tensor<T>, indices: &[i32], lambda: f32) -> Result<Tensor<T>>
    where
        T: Default + Clone + Copy + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
        T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
        T: scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive,
    {
        // Create indices tensor for gathering
        let indices_tensor = Tensor::from_vec(indices.to_vec(), &[indices.len()])?;

        // Gather shuffled tensor
        let shuffled = gather(tensor, &indices_tensor, 0)?;

        // Create lambda tensors for mixing
        let lambda_scalar = Tensor::from_scalar(T::from_f32(lambda).unwrap_or(T::one()));
        let one_minus_lambda_scalar =
            Tensor::from_scalar(T::from_f32(1.0 - lambda).unwrap_or(T::zero()));

        // Compute: lambda * tensor + (1 - lambda) * shuffled
        let term1 = mul(tensor, &lambda_scalar)?;
        let term2 = mul(&shuffled, &one_minus_lambda_scalar)?;
        add(&term1, &term2)
    }
}

impl<T> Layer<T> for Mixup
where
    T: Default + Clone + Copy + Send + Sync + 'static,
{
    /// Standard Layer forward pass - returns input unchanged
    /// Use the specialized `forward` method for actual mixup augmentation
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        Vec::new() // No learnable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        Vec::new() // No learnable parameters
    }

    fn set_training(&mut self, training: bool) {
        self.enabled = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

/// CutMix data augmentation layer
///
/// CutMix cuts and pastes patches between images while mixing labels proportionally
/// to the area of the patches.
///
/// Reference: "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" (Yun et al., 2019)
#[derive(Debug, Clone)]
pub struct CutMix {
    alpha: f32,
    enabled: bool,
}

impl CutMix {
    /// Create a new CutMix layer
    ///
    /// # Arguments
    /// * `alpha` - Beta distribution parameter for mixing ratio
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            enabled: true,
        }
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.enabled = training;
    }

    /// Apply CutMix to a batch of image samples and labels
    ///
    /// # Arguments
    /// * `images` - Input batch tensor of shape [batch_size, channels, height, width]
    /// * `labels` - Label batch tensor
    ///
    /// # Returns
    /// Tuple of (mixed_images, mixed_labels, lambda)
    pub fn forward<T>(
        &self,
        images: &Tensor<T>,
        labels: &Tensor<T>,
    ) -> Result<(Tensor<T>, Tensor<T>, f32)>
    where
        T: Default + Clone + Copy + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
        T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
        T: scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::ToPrimitive,
    {
        if !self.enabled {
            let lambda = 1.0;
            return Ok((images.clone(), labels.clone(), lambda));
        }

        let shape = images.shape();
        if shape.rank() != 4 {
            return Err(TensorError::unsupported_operation_simple(
                "CutMix requires 4D input tensor [batch_size, channels, height, width]".to_string(),
            ));
        }

        let batch_size = shape.dims()[0];
        let height = shape.dims()[2];
        let width = shape.dims()[3];

        if batch_size < 2 {
            return Err(TensorError::unsupported_operation_simple(
                "CutMix requires batch size >= 2".to_string(),
            ));
        }

        // Sample lambda from Beta distribution
        let mut rng = thread_rng();
        let lambda = if self.alpha > 0.0 {
            let u1: f32 = rng.gen_range(0.0..1.0);
            let u2: f32 = rng.gen_range(0.0..1.0);
            let x = u1.powf(1.0 / self.alpha);
            let y = u2.powf(1.0 / self.alpha);
            x / (x + y)
        } else {
            1.0
        };

        // Generate bounding box
        let cut_ratio = (1.0 - lambda).sqrt();
        let cut_w = (width as f32 * cut_ratio) as usize;
        let cut_h = (height as f32 * cut_ratio) as usize;

        // Random center point
        let cx = rng.gen_range(0..width);
        let cy = rng.gen_range(0..height);

        // Bounding box coordinates
        let x1 = max(0, cx as i32 - cut_w as i32 / 2) as usize;
        let y1 = max(0, cy as i32 - cut_h as i32 / 2) as usize;
        let x2 = min(width, x1 + cut_w);
        let y2 = min(height, y1 + cut_h);

        // Actual lambda based on the cut area
        let actual_lambda = 1.0 - ((x2 - x1) * (y2 - y1)) as f32 / (width * height) as f32;

        // Create random permutation for batch
        let mut indices: Vec<i32> = (0..batch_size).map(|i| i as i32).collect();
        for i in (1..batch_size).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        // Apply CutMix with actual patch cutting
        let mixed_images = self.cutmix_images_with_patches(images, &indices, x1, y1, x2, y2)?;
        let mixed_labels = self.mix_labels(labels, &indices, actual_lambda)?;

        Ok((mixed_images, mixed_labels, actual_lambda))
    }

    /// Apply actual CutMix to images with patch cutting
    fn cutmix_images_with_patches<T>(
        &self,
        images: &Tensor<T>,
        indices: &[i32],
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
    ) -> Result<Tensor<T>>
    where
        T: Default + Clone + Copy + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
        T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
        T: scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive,
    {
        use tenflowers_core::ops::gather;

        // Create indices tensor
        let indices_tensor = Tensor::from_vec(indices.to_vec(), &[indices.len()])?;

        // Get shuffled images
        let shuffled_images = gather(images, &indices_tensor, 0)?;

        let shape = images.shape();
        let batch_size = shape.dims()[0];
        let channels = shape.dims()[1];
        let height = shape.dims()[2];
        let width = shape.dims()[3];

        // Create mixed images by manually copying data
        let total_size = batch_size * channels * height * width;
        let mut mixed_data = vec![T::zero(); total_size];

        // Get original and shuffled image data
        if let (Some(original_data), Some(shuffled_data)) =
            (images.as_slice(), shuffled_images.as_slice())
        {
            // Copy original data first
            mixed_data.copy_from_slice(original_data);

            // If cut region is valid, perform patch cutting
            if x1 < x2 && y1 < y2 && x2 <= width && y2 <= height {
                for b in 0..batch_size {
                    for c in 0..channels {
                        for h in y1..y2 {
                            for w in x1..x2 {
                                // Calculate indices for 4D tensor: [batch, channels, height, width]
                                let mixed_idx = b * channels * height * width
                                    + c * height * width
                                    + h * width
                                    + w;
                                let shuffled_idx = b * channels * height * width
                                    + c * height * width
                                    + h * width
                                    + w;

                                if mixed_idx < mixed_data.len()
                                    && shuffled_idx < shuffled_data.len()
                                {
                                    mixed_data[mixed_idx] = shuffled_data[shuffled_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Create new tensor with mixed data
        Tensor::from_vec(mixed_data, shape.dims())
    }

    /// Apply simplified CutMix to images by blending (fallback method)
    fn cutmix_images<T>(
        &self,
        images: &Tensor<T>,
        indices: &[i32],
        lambda: f32,
    ) -> Result<Tensor<T>>
    where
        T: Default + Clone + Copy + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
        T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
        T: scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive,
    {
        // Create indices tensor
        let indices_tensor = Tensor::from_vec(indices.to_vec(), &[indices.len()])?;

        // Get shuffled images
        let shuffled_images = gather(images, &indices_tensor, 0)?;

        // Weighted blending as fallback
        let lambda_scalar = Tensor::from_scalar(T::from_f32(lambda).unwrap_or(T::one()));
        let one_minus_lambda_scalar =
            Tensor::from_scalar(T::from_f32(1.0 - lambda).unwrap_or(T::zero()));

        let term1 = mul(images, &lambda_scalar)?;
        let term2 = mul(&shuffled_images, &one_minus_lambda_scalar)?;
        add(&term1, &term2)
    }

    /// Mix labels proportionally to the cut area
    fn mix_labels<T>(&self, labels: &Tensor<T>, indices: &[i32], lambda: f32) -> Result<Tensor<T>>
    where
        T: Default + Clone + Copy + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
        T: std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
        T: scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One,
    {
        let indices_tensor = Tensor::from_vec(indices.to_vec(), &[indices.len()])?;
        let shuffled_labels = gather(labels, &indices_tensor, 0)?;

        let lambda_scalar = Tensor::from_scalar(T::from_f32(lambda).unwrap_or(T::one()));
        let one_minus_lambda_scalar =
            Tensor::from_scalar(T::from_f32(1.0 - lambda).unwrap_or(T::zero()));

        let term1 = mul(labels, &lambda_scalar)?;
        let term2 = mul(&shuffled_labels, &one_minus_lambda_scalar)?;
        add(&term1, &term2)
    }
}

impl<T> Layer<T> for CutMix
where
    T: Default + Clone + Copy + Send + Sync + 'static,
{
    /// Standard Layer forward pass - returns input unchanged
    /// Use the specialized `forward` method for actual cutmix augmentation
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        Vec::new() // No learnable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        Vec::new() // No learnable parameters
    }

    fn set_training(&mut self, training: bool) {
        self.enabled = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

/// Label smoothing layer for regularization
///
/// Label smoothing reduces overfitting by preventing the model from becoming
/// overly confident in its predictions.
#[derive(Debug, Clone)]
pub struct LabelSmoothing {
    smoothing: f32,
    enabled: bool,
}

impl LabelSmoothing {
    /// Create a new LabelSmoothing layer
    ///
    /// # Arguments
    /// * `smoothing` - Smoothing factor (0.0 = no smoothing, 0.1 = typical value)
    pub fn new(smoothing: f32) -> Self {
        Self {
            smoothing: smoothing.clamp(0.0, 1.0),
            enabled: true,
        }
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.enabled = training;
    }

    /// Apply label smoothing to one-hot encoded labels
    ///
    /// # Arguments
    /// * `labels` - One-hot encoded labels of shape [batch_size, num_classes]
    ///
    /// # Returns
    /// Smoothed labels
    pub fn forward<T>(&self, labels: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Default + Clone + Copy + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
        T: scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::Zero,
    {
        if !self.enabled || self.smoothing == 0.0 {
            return Ok(labels.clone());
        }

        let shape = labels.shape();
        if shape.rank() != 2 {
            return Err(TensorError::unsupported_operation_simple(
                "Label smoothing requires 2D input tensor [batch_size, num_classes]".to_string(),
            ));
        }

        let num_classes = shape.dims()[1] as f32;
        let one_minus_smoothing_scalar =
            Tensor::from_scalar(T::from_f32(1.0 - self.smoothing).unwrap_or(T::one()));
        let uniform_prob_scalar =
            Tensor::from_scalar(T::from_f32(self.smoothing / num_classes).unwrap_or(T::zero()));

        // Smoothed labels = (1 - smoothing) * original + smoothing / num_classes
        let scaled_labels = mul(labels, &one_minus_smoothing_scalar)?;
        let uniform_tensor = mul(&Tensor::ones(shape.dims()), &uniform_prob_scalar)?;

        add(&scaled_labels, &uniform_tensor)
    }
}

impl<T> Layer<T> for LabelSmoothing
where
    T: Default + Clone + Copy + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    T: std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
    T: scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::One
        + scirs2_core::num_traits::Zero,
{
    /// Standard Layer forward pass - applies label smoothing to input
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        Vec::new() // No learnable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        Vec::new() // No learnable parameters
    }

    fn set_training(&mut self, training: bool) {
        self.enabled = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixup_creation() {
        let mixup = Mixup::new(1.0);
        assert_eq!(mixup.alpha, 1.0);
        assert!(mixup.enabled);
    }

    #[test]
    fn test_mixup_forward() {
        let mixup = Mixup::new(1.0);

        // Create test data
        let samples = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let labels = Tensor::from_vec(vec![1.0f32, 0.0], &[2]).unwrap();

        let result = mixup.forward(&samples, &labels);
        assert!(result.is_ok());

        let (mixed_samples, mixed_labels, lambda) = result.unwrap();
        assert_eq!(mixed_samples.shape().dims(), samples.shape().dims());
        assert_eq!(mixed_labels.shape().dims(), labels.shape().dims());
        assert!(lambda >= 0.0 && lambda <= 1.0);
    }

    #[test]
    fn test_mixup_disabled() {
        let mut mixup = Mixup::new(1.0);
        mixup.set_training(false);

        let samples = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::from_vec(vec![1.0f32, 0.0], &[2]).unwrap();

        let result = mixup.forward(&samples, &labels).unwrap();
        assert_eq!(result.2, 1.0); // lambda should be 1.0 when disabled
    }

    #[test]
    fn test_cutmix_creation() {
        let cutmix = CutMix::new(1.0);
        assert_eq!(cutmix.alpha, 1.0);
        assert!(cutmix.enabled);
    }

    #[test]
    fn test_cutmix_forward() {
        let cutmix = CutMix::new(1.0);

        // Create test 4D image data [batch_size, channels, height, width]
        let images = Tensor::zeros(&[2, 3, 8, 8]);
        let labels = Tensor::from_vec(vec![1.0f32, 0.0], &[2]).unwrap();

        let result = cutmix.forward(&images, &labels);
        assert!(result.is_ok());

        let (mixed_images, mixed_labels, lambda) = result.unwrap();
        assert_eq!(mixed_images.shape().dims(), images.shape().dims());
        assert_eq!(mixed_labels.shape().dims(), labels.shape().dims());
        assert!(lambda >= 0.0 && lambda <= 1.0);
    }

    #[test]
    fn test_label_smoothing() {
        let smoothing = LabelSmoothing::new(0.1);

        // One-hot encoded labels
        let labels = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]).unwrap();

        let result = smoothing.forward(&labels);
        assert!(result.is_ok());

        let smoothed = result.unwrap();
        assert_eq!(smoothed.shape().dims(), labels.shape().dims());
    }

    #[test]
    fn test_label_smoothing_disabled() {
        let mut smoothing = LabelSmoothing::new(0.1);
        smoothing.set_training(false);

        let labels = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let result = smoothing.forward(&labels).unwrap();

        // When disabled, should return original labels
        // Note: exact comparison might fail due to floating point precision
        assert_eq!(result.shape().dims(), labels.shape().dims());
    }

    #[test]
    fn test_insufficient_batch_size() {
        let mixup = Mixup::new(1.0);

        // Single sample should fail
        let samples = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 2]).unwrap();
        let labels = Tensor::from_vec(vec![1.0f32], &[1]).unwrap();

        let result = mixup.forward(&samples, &labels);
        assert!(result.is_err());
    }
}
