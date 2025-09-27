use num_traits::{Float, FromPrimitive, Zero};
use tenflowers_core::{Result, Tensor};

/// ROI (Region of Interest) Pooling Layer
/// Used in object detection models like R-CNN
pub struct ROIPool2D {
    pooled_size: (usize, usize),
    spatial_scale: f32,
}

impl ROIPool2D {
    pub fn new(pooled_size: (usize, usize), spatial_scale: f32) -> Self {
        Self {
            pooled_size,
            spatial_scale,
        }
    }

    /// Square pooled size constructor
    pub fn square(size: usize, spatial_scale: f32) -> Self {
        Self::new((size, size), spatial_scale)
    }
}

impl ROIPool2D {
    pub fn forward<T>(&self, feature_maps: &Tensor<T>, rois: &Tensor<T>) -> Result<Tensor<T>>
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
        tenflowers_core::ops::roi_pool2d(feature_maps, rois, self.pooled_size, self.spatial_scale)
    }
}

/// ROI Align Layer (improved version of ROI pooling)
/// Uses bilinear interpolation for better gradient flow
pub struct ROIAlign2D {
    pooled_size: (usize, usize),
    spatial_scale: f32,
    sampling_ratio: i32,
}

impl ROIAlign2D {
    pub fn new(pooled_size: (usize, usize), spatial_scale: f32, sampling_ratio: i32) -> Self {
        Self {
            pooled_size,
            spatial_scale,
            sampling_ratio,
        }
    }

    /// Square pooled size constructor
    pub fn square(size: usize, spatial_scale: f32, sampling_ratio: i32) -> Self {
        Self::new((size, size), spatial_scale, sampling_ratio)
    }

    /// Auto sampling ratio constructor (sampling_ratio = -1 means adaptive)
    pub fn auto_sampling(pooled_size: (usize, usize), spatial_scale: f32) -> Self {
        Self::new(pooled_size, spatial_scale, -1)
    }
}

impl ROIAlign2D {
    pub fn forward<T>(&self, feature_maps: &Tensor<T>, rois: &Tensor<T>) -> Result<Tensor<T>>
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
        tenflowers_core::ops::roi_align2d(
            feature_maps,
            rois,
            self.pooled_size,
            self.spatial_scale,
            self.sampling_ratio,
        )
    }
}
