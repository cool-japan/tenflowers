//! ROI (Region of Interest) pooling operations for object detection
//!
//! This module provides ROI pooling and ROI align operations commonly used
//! in object detection models like R-CNN, Fast R-CNN, and Mask R-CNN.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};

/// ROI (Region of Interest) pooling operation
/// Used in object detection models like R-CNN
/// Input: feature_maps [batch, channels, height, width], rois [num_rois, 5]
/// ROI format: [batch_idx, x1, y1, x2, y2] where coordinates are normalized to [0, 1]
/// Output: [num_rois, channels, pooled_height, pooled_width]
pub fn roi_pool2d<T>(
    feature_maps: &Tensor<T>,
    rois: &Tensor<T>,
    pooled_size: (usize, usize),
    spatial_scale: f32,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &feature_maps.storage {
        TensorStorage::Cpu(_input_arr) => {
            roi_pool2d_cpu(feature_maps, rois, pooled_size, spatial_scale)
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => {
            roi_pool2d_gpu(feature_maps, rois, pooled_size, spatial_scale)
        }
    }
}

/// ROI Align pooling operation (improved version of ROI pooling)
/// Uses bilinear interpolation for better gradient flow
pub fn roi_align2d<T>(
    feature_maps: &Tensor<T>,
    rois: &Tensor<T>,
    pooled_size: (usize, usize),
    spatial_scale: f32,
    sampling_ratio: i32,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &feature_maps.storage {
        TensorStorage::Cpu(_input_arr) => roi_align2d_cpu(
            feature_maps,
            rois,
            pooled_size,
            spatial_scale,
            sampling_ratio,
        ),
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => roi_align2d_gpu(
            feature_maps,
            rois,
            pooled_size,
            spatial_scale,
            sampling_ratio,
        ),
    }
}

// CPU implementations

fn roi_pool2d_cpu<T>(
    feature_maps: &Tensor<T>,
    rois: &Tensor<T>,
    pooled_size: (usize, usize),
    spatial_scale: f32,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + PartialOrd + Float + FromPrimitive + Send + Sync + 'static,
{
    let feature_shape = feature_maps.shape();
    let roi_shape = rois.shape();

    if feature_shape.rank() != 4 {
        return Err(TensorError::invalid_shape_simple(format!(
            "Feature maps must be 4D, got {}D",
            feature_shape.rank()
        )));
    }

    if roi_shape.rank() != 2 || roi_shape.dims()[1] != 5 {
        return Err(TensorError::invalid_shape_simple(format!(
            "ROIs must be [num_rois, 5], got {:?}",
            roi_shape.dims()
        )));
    }

    let batch_size = feature_shape.dims()[0];
    let channels = feature_shape.dims()[1];
    let feature_height = feature_shape.dims()[2];
    let feature_width = feature_shape.dims()[3];
    let num_rois = roi_shape.dims()[0];
    let (pooled_height, pooled_width) = pooled_size;

    let rois_data = rois.as_slice().ok_or_else(|| {
        TensorError::device_error_simple("Cannot access ROIs tensor data".to_string())
    })?;

    let mut output_data = vec![T::zero(); num_rois * channels * pooled_height * pooled_width];

    for roi_idx in 0..num_rois {
        let roi_start_idx = roi_idx * 5;
        let batch_idx = rois_data[roi_start_idx].to_usize().unwrap_or(0);
        let roi_x1 = rois_data[roi_start_idx + 1].to_f32().unwrap_or(0.0);
        let roi_y1 = rois_data[roi_start_idx + 2].to_f32().unwrap_or(0.0);
        let roi_x2 = rois_data[roi_start_idx + 3].to_f32().unwrap_or(1.0);
        let roi_y2 = rois_data[roi_start_idx + 4].to_f32().unwrap_or(1.0);

        if batch_idx >= batch_size {
            continue;
        }

        // Convert normalized coordinates to feature map coordinates
        let x1 = (roi_x1 * spatial_scale * feature_width as f32) as i32;
        let y1 = (roi_y1 * spatial_scale * feature_height as f32) as i32;
        let x2 = (roi_x2 * spatial_scale * feature_width as f32) as i32;
        let y2 = (roi_y2 * spatial_scale * feature_height as f32) as i32;

        let roi_width = std::cmp::max(x2 - x1, 1);
        let roi_height = std::cmp::max(y2 - y1, 1);

        let bin_size_w = roi_width as f32 / pooled_width as f32;
        let bin_size_h = roi_height as f32 / pooled_height as f32;

        for c in 0..channels {
            for ph in 0..pooled_height {
                for pw in 0..pooled_width {
                    let hstart = y1 + ((ph as f32 * bin_size_h) as i32);
                    let wstart = x1 + ((pw as f32 * bin_size_w) as i32);
                    let hend = y1 + (((ph + 1) as f32 * bin_size_h) as i32);
                    let wend = x1 + (((pw + 1) as f32 * bin_size_w) as i32);

                    let hstart = std::cmp::max(hstart, 0) as usize;
                    let wstart = std::cmp::max(wstart, 0) as usize;
                    let hend = std::cmp::min(hend, feature_height as i32) as usize;
                    let wend = std::cmp::min(wend, feature_width as i32) as usize;

                    let mut max_val: Option<T> = None;
                    for h in hstart..hend {
                        for w in wstart..wend {
                            if let Some(val) = feature_maps.get(&[batch_idx, c, h, w]) {
                                max_val = match max_val {
                                    None => Some(val),
                                    Some(current_max) => {
                                        if val > current_max {
                                            Some(val)
                                        } else {
                                            Some(current_max)
                                        }
                                    }
                                };
                            }
                        }
                    }

                    let out_idx = roi_idx * channels * pooled_height * pooled_width
                        + c * pooled_height * pooled_width
                        + ph * pooled_width
                        + pw;
                    output_data[out_idx] = max_val.unwrap_or_else(T::zero);
                }
            }
        }
    }

    Tensor::from_vec(
        output_data,
        &[num_rois, channels, pooled_height, pooled_width],
    )
}

fn roi_align2d_cpu<T>(
    feature_maps: &Tensor<T>,
    rois: &Tensor<T>,
    pooled_size: (usize, usize),
    spatial_scale: f32,
    sampling_ratio: i32,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + PartialOrd + Float + FromPrimitive + Send + Sync + 'static,
{
    let feature_shape = feature_maps.shape();
    let roi_shape = rois.shape();

    if feature_shape.rank() != 4 {
        return Err(TensorError::invalid_shape_simple(format!(
            "Feature maps must be 4D, got {}D",
            feature_shape.rank()
        )));
    }

    if roi_shape.rank() != 2 || roi_shape.dims()[1] != 5 {
        return Err(TensorError::invalid_shape_simple(format!(
            "ROIs must be [num_rois, 5], got {:?}",
            roi_shape.dims()
        )));
    }

    let batch_size = feature_shape.dims()[0];
    let channels = feature_shape.dims()[1];
    let feature_height = feature_shape.dims()[2];
    let feature_width = feature_shape.dims()[3];
    let num_rois = roi_shape.dims()[0];
    let (pooled_height, pooled_width) = pooled_size;

    let rois_data = rois.as_slice().ok_or_else(|| {
        TensorError::device_error_simple("Cannot access ROIs tensor data".to_string())
    })?;

    let mut output_data = vec![T::zero(); num_rois * channels * pooled_height * pooled_width];

    for roi_idx in 0..num_rois {
        let roi_start_idx = roi_idx * 5;
        let batch_idx = rois_data[roi_start_idx].to_usize().unwrap_or(0);
        let roi_x1 = rois_data[roi_start_idx + 1].to_f32().unwrap_or(0.0);
        let roi_y1 = rois_data[roi_start_idx + 2].to_f32().unwrap_or(0.0);
        let roi_x2 = rois_data[roi_start_idx + 3].to_f32().unwrap_or(1.0);
        let roi_y2 = rois_data[roi_start_idx + 4].to_f32().unwrap_or(1.0);

        if batch_idx >= batch_size {
            continue;
        }

        // Convert normalized coordinates to feature map coordinates
        let x1 = roi_x1 * spatial_scale * feature_width as f32;
        let y1 = roi_y1 * spatial_scale * feature_height as f32;
        let x2 = roi_x2 * spatial_scale * feature_width as f32;
        let y2 = roi_y2 * spatial_scale * feature_height as f32;

        let roi_width = x2 - x1;
        let roi_height = y2 - y1;

        let bin_size_w = roi_width / pooled_width as f32;
        let bin_size_h = roi_height / pooled_height as f32;

        // Determine sampling ratio
        let roi_bin_grid_h = if sampling_ratio > 0 {
            sampling_ratio
        } else {
            (roi_height / pooled_height as f32).ceil() as i32
        };
        let roi_bin_grid_w = if sampling_ratio > 0 {
            sampling_ratio
        } else {
            (roi_width / pooled_width as f32).ceil() as i32
        };

        for c in 0..channels {
            for ph in 0..pooled_height {
                for pw in 0..pooled_width {
                    let mut output_val = T::zero();
                    let count = roi_bin_grid_h * roi_bin_grid_w;

                    for iy in 0..roi_bin_grid_h {
                        let y = y1
                            + (ph as f32 + (iy as f32 + 0.5) / roi_bin_grid_h as f32) * bin_size_h;
                        for ix in 0..roi_bin_grid_w {
                            let x = x1
                                + (pw as f32 + (ix as f32 + 0.5) / roi_bin_grid_w as f32)
                                    * bin_size_w;

                            // Bilinear interpolation
                            let val = bilinear_interpolate(
                                feature_maps,
                                batch_idx,
                                c,
                                y,
                                x,
                                feature_height,
                                feature_width,
                            )?;
                            output_val = output_val + val;
                        }
                    }

                    output_val = output_val / T::from_i32(count).unwrap_or(T::one());

                    let out_idx = roi_idx * channels * pooled_height * pooled_width
                        + c * pooled_height * pooled_width
                        + ph * pooled_width
                        + pw;
                    output_data[out_idx] = output_val;
                }
            }
        }
    }

    Tensor::from_vec(
        output_data,
        &[num_rois, channels, pooled_height, pooled_width],
    )
}

/// Bilinear interpolation for ROI Align
fn bilinear_interpolate<T>(
    feature_maps: &Tensor<T>,
    batch_idx: usize,
    channel: usize,
    y: f32,
    x: f32,
    height: usize,
    width: usize,
) -> Result<T>
where
    T: Clone + Default + Zero + Float + FromPrimitive,
{
    if y < -1.0 || y > height as f32 || x < -1.0 || x > width as f32 {
        return Ok(T::zero());
    }

    let y = y.max(0.0);
    let x = x.max(0.0);

    let y_low = y.floor() as i32;
    let x_low = x.floor() as i32;
    let y_high = y_low + 1;
    let x_high = x_low + 1;

    let ly = y - y_low as f32;
    let lx = x - x_low as f32;
    let hy = 1.0 - ly;
    let hx = 1.0 - lx;

    let mut v1 = T::zero();
    let mut v2 = T::zero();
    let mut v3 = T::zero();
    let mut v4 = T::zero();

    if y_low >= 0 && y_low < height as i32 && x_low >= 0 && x_low < width as i32 {
        v1 = feature_maps
            .get(&[batch_idx, channel, y_low as usize, x_low as usize])
            .unwrap_or(T::zero());
    }
    if y_low >= 0 && y_low < height as i32 && x_high >= 0 && x_high < width as i32 {
        v2 = feature_maps
            .get(&[batch_idx, channel, y_low as usize, x_high as usize])
            .unwrap_or(T::zero());
    }
    if y_high >= 0 && y_high < height as i32 && x_low >= 0 && x_low < width as i32 {
        v3 = feature_maps
            .get(&[batch_idx, channel, y_high as usize, x_low as usize])
            .unwrap_or(T::zero());
    }
    if y_high >= 0 && y_high < height as i32 && x_high >= 0 && x_high < width as i32 {
        v4 = feature_maps
            .get(&[batch_idx, channel, y_high as usize, x_high as usize])
            .unwrap_or(T::zero());
    }

    let w1 = T::from_f32(hy * hx).unwrap_or(T::zero());
    let w2 = T::from_f32(hy * lx).unwrap_or(T::zero());
    let w3 = T::from_f32(ly * hx).unwrap_or(T::zero());
    let w4 = T::from_f32(ly * lx).unwrap_or(T::zero());

    Ok(w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4)
}

// GPU implementations

#[cfg(feature = "gpu")]
fn roi_pool2d_gpu<T>(
    feature_maps: &Tensor<T>,
    rois: &Tensor<T>,
    pooled_size: (usize, usize),
    spatial_scale: f32,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // For now, fallback to CPU implementation
    // GPU ROI pooling would require specialized kernels
    roi_pool2d_cpu(feature_maps, rois, pooled_size, spatial_scale)
}

#[cfg(feature = "gpu")]
fn roi_align2d_gpu<T>(
    feature_maps: &Tensor<T>,
    rois: &Tensor<T>,
    pooled_size: (usize, usize),
    spatial_scale: f32,
    sampling_ratio: i32,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // For now, fallback to CPU implementation
    // GPU ROI align would require specialized kernels
    roi_align2d_cpu(
        feature_maps,
        rois,
        pooled_size,
        spatial_scale,
        sampling_ratio,
    )
}
