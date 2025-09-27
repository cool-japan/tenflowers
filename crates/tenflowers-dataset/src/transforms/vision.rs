//! Vision-specific transformations for image datasets
//!
//! This module provides computer vision transformations commonly used for
//! image preprocessing and data augmentation.

use crate::transforms::Transform;
use scirs2_core::random::Rng;
use std::marker::PhantomData;
use tenflowers_core::{Result, Tensor, TensorError};

/// Resize transformation for images
/// Resizes images to a target size using bilinear interpolation
pub struct Resize<T> {
    target_height: usize,
    target_width: usize,
    _phantom: PhantomData<T>,
}

impl<T> Resize<T>
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
    pub fn new(target_height: usize, target_width: usize) -> Self {
        Self {
            target_height,
            target_width,
            _phantom: PhantomData,
        }
    }

    /// Bilinear interpolation for image resizing
    fn bilinear_interpolate(
        &self,
        image: &[T],
        orig_height: usize,
        orig_width: usize,
        channels: usize,
        x: f32,
        y: f32,
        c: usize,
    ) -> T {
        let x1 = x.floor() as usize;
        let y1 = y.floor() as usize;
        let x2 = (x1 + 1).min(orig_width - 1);
        let y2 = (y1 + 1).min(orig_height - 1);

        let dx = x - x1 as f32;
        let dy = y - y1 as f32;

        let get_pixel = |h: usize, w: usize, ch: usize| -> T {
            let index = h * orig_width * channels + w * channels + ch;
            if index < image.len() {
                image[index]
            } else {
                T::zero()
            }
        };

        let p11 = get_pixel(y1, x1, c);
        let p12 = get_pixel(y1, x2, c);
        let p21 = get_pixel(y2, x1, c);
        let p22 = get_pixel(y2, x2, c);

        let dx_t = T::from(dx).unwrap_or(T::zero());
        let dy_t = T::from(dy).unwrap_or(T::zero());
        let one_minus_dx = T::one() - dx_t;
        let one_minus_dy = T::one() - dy_t;

        let interpolated = p11 * one_minus_dx * one_minus_dy
            + p12 * dx_t * one_minus_dy
            + p21 * one_minus_dx * dy_t
            + p22 * dx_t * dy_t;

        interpolated
    }
}

impl<T> Transform<T> for Resize<T>
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
        let (features, labels) = sample;
        let shape = features.shape().dims();

        // Assume image format [channels, height, width] or [height, width, channels]
        if shape.len() < 2 {
            return Err(TensorError::invalid_argument(
                "Image tensor must have at least 2 dimensions".to_string(),
            ));
        }

        let (channels, orig_height, orig_width) = if shape.len() == 3 {
            if shape[0] <= 4 {
                // Assume [channels, height, width]
                (shape[0], shape[1], shape[2])
            } else {
                // Assume [height, width, channels]
                (shape[2], shape[0], shape[1])
            }
        } else {
            // Grayscale [height, width]
            (1, shape[0], shape[1])
        };

        let image_data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let mut resized_data =
            Vec::with_capacity(self.target_height * self.target_width * channels);

        let scale_y = orig_height as f32 / self.target_height as f32;
        let scale_x = orig_width as f32 / self.target_width as f32;

        for new_y in 0..self.target_height {
            for new_x in 0..self.target_width {
                let src_y = new_y as f32 * scale_y;
                let src_x = new_x as f32 * scale_x;

                for c in 0..channels {
                    let interpolated = self.bilinear_interpolate(
                        image_data,
                        orig_height,
                        orig_width,
                        channels,
                        src_x,
                        src_y,
                        c,
                    );
                    resized_data.push(interpolated);
                }
            }
        }

        let new_shape = if shape.len() == 3 {
            if shape[0] <= 4 {
                vec![channels, self.target_height, self.target_width]
            } else {
                vec![self.target_height, self.target_width, channels]
            }
        } else {
            vec![self.target_height, self.target_width]
        };

        let resized_features = Tensor::from_vec(resized_data, &new_shape)?;
        Ok((resized_features, labels))
    }
}

/// Random crop with optional padding
pub struct RandomCropWithPadding<T> {
    crop_height: usize,
    crop_width: usize,
    padding: Option<usize>,
    fill_value: T,
    _phantom: PhantomData<T>,
}

impl<T> RandomCropWithPadding<T>
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
        crop_height: usize,
        crop_width: usize,
        padding: Option<usize>,
        fill_value: T,
    ) -> Self {
        Self {
            crop_height,
            crop_width,
            padding,
            fill_value,
            _phantom: PhantomData,
        }
    }

    pub fn without_padding(crop_height: usize, crop_width: usize) -> Self {
        Self::new(crop_height, crop_width, None, T::zero())
    }

    pub fn with_padding(
        crop_height: usize,
        crop_width: usize,
        padding: usize,
        fill_value: T,
    ) -> Self {
        Self::new(crop_height, crop_width, Some(padding), fill_value)
    }
}

impl<T> Transform<T> for RandomCropWithPadding<T>
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
        let (features, labels) = sample;
        let shape = features.shape().dims();

        if shape.len() < 2 {
            return Err(TensorError::invalid_argument(
                "Image tensor must have at least 2 dimensions".to_string(),
            ));
        }

        let (channels, orig_height, orig_width) = if shape.len() == 3 {
            if shape[0] <= 4 {
                (shape[0], shape[1], shape[2])
            } else {
                (shape[2], shape[0], shape[1])
            }
        } else {
            (1, shape[0], shape[1])
        };

        let image_data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let mut rng = scirs2_core::random::rng();

        // Apply padding if specified
        let (padded_data, padded_height, padded_width) = if let Some(pad) = self.padding {
            let new_height = orig_height + 2 * pad;
            let new_width = orig_width + 2 * pad;
            let mut padded = vec![self.fill_value; new_height * new_width * channels];

            // Copy original image to center of padded image
            for h in 0..orig_height {
                for w in 0..orig_width {
                    for c in 0..channels {
                        let src_idx = h * orig_width * channels + w * channels + c;
                        let dst_idx = (h + pad) * new_width * channels + (w + pad) * channels + c;
                        if src_idx < image_data.len() && dst_idx < padded.len() {
                            padded[dst_idx] = image_data[src_idx];
                        }
                    }
                }
            }

            (padded, new_height, new_width)
        } else {
            (image_data.to_vec(), orig_height, orig_width)
        };

        // Determine crop position
        let max_crop_y = padded_height.saturating_sub(self.crop_height);
        let max_crop_x = padded_width.saturating_sub(self.crop_width);

        let crop_y = if max_crop_y > 0 {
            rng.gen_range(0..=max_crop_y)
        } else {
            0
        };
        let crop_x = if max_crop_x > 0 {
            rng.gen_range(0..=max_crop_x)
        } else {
            0
        };

        // Extract crop
        let mut cropped_data = Vec::with_capacity(self.crop_height * self.crop_width * channels);

        for y in 0..self.crop_height {
            for x in 0..self.crop_width {
                for c in 0..channels {
                    let src_y = crop_y + y;
                    let src_x = crop_x + x;

                    if src_y < padded_height && src_x < padded_width {
                        let idx = src_y * padded_width * channels + src_x * channels + c;
                        if idx < padded_data.len() {
                            cropped_data.push(padded_data[idx]);
                        } else {
                            cropped_data.push(self.fill_value);
                        }
                    } else {
                        cropped_data.push(self.fill_value);
                    }
                }
            }
        }

        let new_shape = if shape.len() == 3 {
            if shape[0] <= 4 {
                vec![channels, self.crop_height, self.crop_width]
            } else {
                vec![self.crop_height, self.crop_width, channels]
            }
        } else {
            vec![self.crop_height, self.crop_width]
        };

        let cropped_features = Tensor::from_vec(cropped_data, &new_shape)?;
        Ok((cropped_features, labels))
    }
}

/// Center crop transformation
pub struct CenterCrop<T> {
    crop_height: usize,
    crop_width: usize,
    _phantom: PhantomData<T>,
}

impl<T> CenterCrop<T> {
    pub fn new(crop_height: usize, crop_width: usize) -> Self {
        Self {
            crop_height,
            crop_width,
            _phantom: PhantomData,
        }
    }

    pub fn square(size: usize) -> Self {
        Self::new(size, size)
    }
}

impl<T> Transform<T> for CenterCrop<T>
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
        let (features, labels) = sample;
        let shape = features.shape().dims();

        if shape.len() < 2 {
            return Err(TensorError::invalid_argument(
                "Image tensor must have at least 2 dimensions".to_string(),
            ));
        }

        let (channels, orig_height, orig_width) = if shape.len() == 3 {
            if shape[0] <= 4 {
                (shape[0], shape[1], shape[2])
            } else {
                (shape[2], shape[0], shape[1])
            }
        } else {
            (1, shape[0], shape[1])
        };

        let image_data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        // Calculate center crop position
        let crop_y = if orig_height >= self.crop_height {
            (orig_height - self.crop_height) / 2
        } else {
            0
        };
        let crop_x = if orig_width >= self.crop_width {
            (orig_width - self.crop_width) / 2
        } else {
            0
        };

        let actual_height = self.crop_height.min(orig_height);
        let actual_width = self.crop_width.min(orig_width);

        let mut cropped_data = Vec::with_capacity(actual_height * actual_width * channels);

        for y in 0..actual_height {
            for x in 0..actual_width {
                for c in 0..channels {
                    let src_y = crop_y + y;
                    let src_x = crop_x + x;
                    let idx = src_y * orig_width * channels + src_x * channels + c;

                    if idx < image_data.len() {
                        cropped_data.push(image_data[idx]);
                    } else {
                        cropped_data.push(T::zero());
                    }
                }
            }
        }

        let new_shape = if shape.len() == 3 {
            if shape[0] <= 4 {
                vec![channels, actual_height, actual_width]
            } else {
                vec![actual_height, actual_width, channels]
            }
        } else {
            vec![actual_height, actual_width]
        };

        let cropped_features = Tensor::from_vec(cropped_data, &new_shape)?;
        Ok((cropped_features, labels))
    }
}

/// Random horizontal flip transformation
pub struct RandomHorizontalFlip {
    probability: f32,
}

impl RandomHorizontalFlip {
    pub fn new(probability: f32) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
        }
    }

    pub fn default() -> Self {
        Self::new(0.5)
    }
}

impl<T> Transform<T> for RandomHorizontalFlip
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
        let (features, labels) = sample;

        let mut rng = scirs2_core::random::rng();
        if rng.random::<f32>() >= self.probability {
            return Ok((features, labels));
        }

        let shape = features.shape().dims();

        if shape.len() < 2 {
            return Err(TensorError::invalid_argument(
                "Image tensor must have at least 2 dimensions".to_string(),
            ));
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

        let image_data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let mut flipped_data = Vec::with_capacity(image_data.len());

        for h in 0..height {
            for w in 0..width {
                let flipped_w = width - 1 - w;
                for c in 0..channels {
                    let src_idx = h * width * channels + flipped_w * channels + c;
                    if src_idx < image_data.len() {
                        flipped_data.push(image_data[src_idx]);
                    } else {
                        flipped_data.push(T::zero());
                    }
                }
            }
        }

        let flipped_features = Tensor::from_vec(flipped_data, shape)?;
        Ok((flipped_features, labels))
    }
}

/// Random vertical flip transformation
pub struct RandomVerticalFlip {
    probability: f32,
}

impl RandomVerticalFlip {
    pub fn new(probability: f32) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
        }
    }

    pub fn default() -> Self {
        Self::new(0.5)
    }
}

impl<T> Transform<T> for RandomVerticalFlip
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
        let (features, labels) = sample;

        let mut rng = scirs2_core::random::rng();
        if rng.random::<f32>() >= self.probability {
            return Ok((features, labels));
        }

        let shape = features.shape().dims();

        if shape.len() < 2 {
            return Err(TensorError::invalid_argument(
                "Image tensor must have at least 2 dimensions".to_string(),
            ));
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

        let image_data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let mut flipped_data = Vec::with_capacity(image_data.len());

        for h in 0..height {
            let flipped_h = height - 1 - h;
            for w in 0..width {
                for c in 0..channels {
                    let src_idx = flipped_h * width * channels + w * channels + c;
                    if src_idx < image_data.len() {
                        flipped_data.push(image_data[src_idx]);
                    } else {
                        flipped_data.push(T::zero());
                    }
                }
            }
        }

        let flipped_features = Tensor::from_vec(flipped_data, shape)?;
        Ok((flipped_features, labels))
    }
}

/// Color jitter transformation for brightness, contrast, saturation, and hue adjustments
pub struct ColorJitter<T> {
    brightness: Option<(T, T)>, // (min_factor, max_factor)
    contrast: Option<(T, T)>,   // (min_factor, max_factor)
    saturation: Option<(T, T)>, // (min_factor, max_factor)
    hue: Option<(T, T)>,        // (min_offset, max_offset)
    _phantom: PhantomData<T>,
}

impl<T> ColorJitter<T>
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
    pub fn new() -> Self {
        Self {
            brightness: None,
            contrast: None,
            saturation: None,
            hue: None,
            _phantom: PhantomData,
        }
    }

    pub fn with_brightness(mut self, min_factor: T, max_factor: T) -> Self {
        self.brightness = Some((min_factor, max_factor));
        self
    }

    pub fn with_contrast(mut self, min_factor: T, max_factor: T) -> Self {
        self.contrast = Some((min_factor, max_factor));
        self
    }

    pub fn with_saturation(mut self, min_factor: T, max_factor: T) -> Self {
        self.saturation = Some((min_factor, max_factor));
        self
    }

    pub fn with_hue(mut self, min_offset: T, max_offset: T) -> Self {
        self.hue = Some((min_offset, max_offset));
        self
    }

    /// Common preset for moderate color jittering
    pub fn moderate() -> Self {
        Self::new()
            .with_brightness(T::from(0.8).unwrap(), T::from(1.2).unwrap())
            .with_contrast(T::from(0.8).unwrap(), T::from(1.2).unwrap())
            .with_saturation(T::from(0.8).unwrap(), T::from(1.2).unwrap())
            .with_hue(T::from(-0.1).unwrap(), T::from(0.1).unwrap())
    }

    /// Apply brightness adjustment
    fn adjust_brightness(&self, pixel: (T, T, T), factor: T) -> (T, T, T) {
        let (r, g, b) = pixel;
        (
            (r * factor).min(T::one()).max(T::zero()),
            (g * factor).min(T::one()).max(T::zero()),
            (b * factor).min(T::one()).max(T::zero()),
        )
    }

    /// Apply contrast adjustment
    fn adjust_contrast(&self, pixel: (T, T, T), factor: T) -> (T, T, T) {
        let gray = T::from(0.299).unwrap();
        let (r, g, b) = pixel;
        let luminance =
            r * T::from(0.299).unwrap() + g * T::from(0.587).unwrap() + b * T::from(0.114).unwrap();

        let new_r = (luminance + (r - luminance) * factor)
            .min(T::one())
            .max(T::zero());
        let new_g = (luminance + (g - luminance) * factor)
            .min(T::one())
            .max(T::zero());
        let new_b = (luminance + (b - luminance) * factor)
            .min(T::one())
            .max(T::zero());

        (new_r, new_g, new_b)
    }
}

impl<T> Default for ColorJitter<T>
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
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Transform<T> for ColorJitter<T>
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
        let (features, labels) = sample;
        let shape = features.shape().dims();

        // Only process RGB images (3 channels)
        if shape.len() != 3 {
            return Ok((features, labels));
        }

        let (channels, height, width) = if shape[0] == 3 {
            (shape[0], shape[1], shape[2])
        } else if shape[2] == 3 {
            (shape[2], shape[0], shape[1])
        } else {
            return Ok((features, labels)); // Not RGB
        };

        if channels != 3 {
            return Ok((features, labels));
        }

        let image_data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let mut rng = scirs2_core::random::rng();
        let mut adjusted_data = image_data.to_vec();

        // Generate random factors for this sample
        let brightness_factor = if let Some((min, max)) = self.brightness {
            Some(rng.gen_range(min.to_f32().unwrap()..=max.to_f32().unwrap()))
        } else {
            None
        };

        let contrast_factor = if let Some((min, max)) = self.contrast {
            Some(rng.gen_range(min.to_f32().unwrap()..=max.to_f32().unwrap()))
        } else {
            None
        };

        // Process each pixel
        for h in 0..height {
            for w in 0..width {
                let r_idx = if shape[0] == 3 {
                    h * width + w
                } else {
                    h * width * 3 + w * 3
                };
                let g_idx = if shape[0] == 3 {
                    height * width + h * width + w
                } else {
                    h * width * 3 + w * 3 + 1
                };
                let b_idx = if shape[0] == 3 {
                    2 * height * width + h * width + w
                } else {
                    h * width * 3 + w * 3 + 2
                };

                if r_idx < adjusted_data.len()
                    && g_idx < adjusted_data.len()
                    && b_idx < adjusted_data.len()
                {
                    let mut pixel = (
                        adjusted_data[r_idx],
                        adjusted_data[g_idx],
                        adjusted_data[b_idx],
                    );

                    // Apply brightness adjustment
                    if let Some(factor) = brightness_factor {
                        pixel = self.adjust_brightness(pixel, T::from(factor).unwrap());
                    }

                    // Apply contrast adjustment
                    if let Some(factor) = contrast_factor {
                        pixel = self.adjust_contrast(pixel, T::from(factor).unwrap());
                    }

                    adjusted_data[r_idx] = pixel.0;
                    adjusted_data[g_idx] = pixel.1;
                    adjusted_data[b_idx] = pixel.2;
                }
            }
        }

        let adjusted_features = Tensor::from_vec(adjusted_data, shape)?;
        Ok((adjusted_features, labels))
    }
}

/// Grid distortion transformation for geometric augmentation
pub struct GridDistortion<T> {
    distortion_strength: f32,
    grid_size: usize,
    _phantom: PhantomData<T>,
}

impl<T> GridDistortion<T>
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
    pub fn new(distortion_strength: f32, grid_size: usize) -> Self {
        Self {
            distortion_strength: distortion_strength.clamp(0.0, 1.0),
            grid_size: grid_size.max(2),
            _phantom: PhantomData,
        }
    }

    pub fn mild(grid_size: usize) -> Self {
        Self::new(0.1, grid_size)
    }

    pub fn moderate(grid_size: usize) -> Self {
        Self::new(0.3, grid_size)
    }

    pub fn strong(grid_size: usize) -> Self {
        Self::new(0.5, grid_size)
    }
}

impl<T> Transform<T> for GridDistortion<T>
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
        let (features, labels) = sample;
        let shape = features.shape().dims();

        if shape.len() < 2 {
            return Err(TensorError::invalid_argument(
                "Image tensor must have at least 2 dimensions".to_string(),
            ));
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

        let image_data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let mut rng = scirs2_core::random::rng();
        let mut distorted_data = vec![T::zero(); image_data.len()];

        // Create a random displacement grid
        let mut dx_grid = vec![vec![0.0; self.grid_size + 1]; self.grid_size + 1];
        let mut dy_grid = vec![vec![0.0; self.grid_size + 1]; self.grid_size + 1];

        for i in 0..=self.grid_size {
            for j in 0..=self.grid_size {
                dx_grid[i][j] = (rng.random::<f32>() - 0.5) * 2.0 * self.distortion_strength;
                dy_grid[i][j] = (rng.random::<f32>() - 0.5) * 2.0 * self.distortion_strength;
            }
        }

        let grid_step_x = width as f32 / self.grid_size as f32;
        let grid_step_y = height as f32 / self.grid_size as f32;

        // Apply grid distortion
        for y in 0..height {
            for x in 0..width {
                // Find grid cell
                let grid_x = (x as f32 / grid_step_x).min(self.grid_size as f32 - 1.0);
                let grid_y = (y as f32 / grid_step_y).min(self.grid_size as f32 - 1.0);

                let gx0 = grid_x.floor() as usize;
                let gy0 = grid_y.floor() as usize;
                let gx1 = (gx0 + 1).min(self.grid_size);
                let gy1 = (gy0 + 1).min(self.grid_size);

                // Bilinear interpolation of displacement
                let fx = grid_x - gx0 as f32;
                let fy = grid_y - gy0 as f32;

                let dx = (1.0 - fx) * (1.0 - fy) * dx_grid[gy0][gx0]
                    + fx * (1.0 - fy) * dx_grid[gy0][gx1]
                    + (1.0 - fx) * fy * dx_grid[gy1][gx0]
                    + fx * fy * dx_grid[gy1][gx1];

                let dy = (1.0 - fx) * (1.0 - fy) * dy_grid[gy0][gx0]
                    + fx * (1.0 - fy) * dy_grid[gy0][gx1]
                    + (1.0 - fx) * fy * dy_grid[gy1][gx0]
                    + fx * fy * dy_grid[gy1][gx1];

                // Apply displacement
                let src_x = (x as f32 + dx * width as f32).round() as i32;
                let src_y = (y as f32 + dy * height as f32).round() as i32;

                if src_x >= 0 && src_x < width as i32 && src_y >= 0 && src_y < height as i32 {
                    for c in 0..channels {
                        let src_idx =
                            (src_y as usize) * width * channels + (src_x as usize) * channels + c;
                        let dst_idx = y * width * channels + x * channels + c;

                        if src_idx < image_data.len() && dst_idx < distorted_data.len() {
                            distorted_data[dst_idx] = image_data[src_idx];
                        }
                    }
                }
            }
        }

        let distorted_features = Tensor::from_vec(distorted_data, shape)?;
        Ok((distorted_features, labels))
    }
}
