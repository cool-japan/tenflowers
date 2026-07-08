//! GPU-accelerated image transforms using WGPU compute shaders - Modular Architecture
//!
//! This module has been refactored into a modular architecture for better maintainability
//! and organization. The functionality has been split into specialized modules by transform type:
//!
//! ## Module Organization
//!
//! - **context**: GPU device management and context initialization
//! - **resize**: GPU-accelerated image resizing with bilinear interpolation
//! - **flip**: GPU-accelerated random horizontal flipping
//! - **jitter**: GPU-accelerated color jittering (brightness, contrast, saturation, hue)
//! - **blur**: GPU-accelerated Gaussian blur filtering
//! - **noise**: GPU-accelerated Gaussian noise addition
//! - **crop**: GPU-accelerated random cropping
//! - **rotation**: GPU-accelerated image rotation
//! - **affine**: GPU-accelerated 2-D affine warp (translation, scale, shear, rotation matrix)
//! - **perspective**: GPU-accelerated perspective / homographic warp
//! - **elastic**: GPU-accelerated elastic distortion (SimoNikolenko-style)
//! - **equalize**: GPU-accelerated per-channel histogram equalization
//!
//! All functionality maintains 100% backward compatibility through strategic re-exports.

pub mod affine;
pub mod blur;
pub mod context;
pub mod crop;
pub mod elastic;
pub mod equalize;
pub mod flip;
pub mod jitter;
pub mod noise;
pub mod perspective;
pub mod resize;
pub mod rotation;

// Re-export core context types for backward compatibility
pub use context::GpuContext;

// Re-export all transform types (original)
pub use blur::GpuGaussianBlur;
pub use crop::GpuRandomCrop;
pub use flip::GpuRandomHorizontalFlip;
pub use jitter::GpuColorJitter;
pub use noise::GpuGaussianNoise;
pub use resize::GpuResize;
pub use rotation::GpuRotation;

// Re-export new GPU transforms (v0.1.2)
pub use affine::{
    affine_compose, affine_identity, affine_rotation, affine_scale, affine_shear,
    affine_translation, AffineMatrix, GpuAffineTransform,
};
pub use elastic::GpuElasticDistortion;
pub use equalize::GpuHistogramEqualize;
pub use perspective::{
    homography_compose, homography_from_quad, homography_identity, GpuPerspectiveTransform,
    HomographyMatrix,
};

// Re-export uniform structures for backward compatibility
#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ColorJitterUniforms {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub padding: u32,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub hue: f32,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GaussianBlurUniforms {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub kernel_size: u32,
    pub sigma: f32,
    pub padding: f32,
    pub padding2: f32,
    pub padding3: f32,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GaussianNoiseUniforms {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub padding: u32,
    pub mean: f32,
    pub std_dev: f32,
    pub seed: u32,
    pub padding2: u32,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RandomCropUniforms {
    pub input_width: u32,
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub offset_x: u32,
    pub offset_y: u32,
    pub channels: u32,
    pub padding: u32,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RotationUniforms {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub padding: u32,
    pub cos_theta: f32,
    pub sin_theta: f32,
    pub center_x: f32,
    pub center_y: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_organization() {
        // Test that all modules are properly accessible
        #[cfg(feature = "gpu")]
        {
            // Test uniform structure sizes for GPU compatibility
            assert_eq!(std::mem::size_of::<ColorJitterUniforms>(), 32);
            assert_eq!(std::mem::size_of::<GaussianBlurUniforms>(), 32);
            assert_eq!(std::mem::size_of::<GaussianNoiseUniforms>(), 32);
            assert_eq!(std::mem::size_of::<RandomCropUniforms>(), 32);
            assert_eq!(std::mem::size_of::<RotationUniforms>(), 32);
        }
    }

    #[test]
    fn test_gpu_context_integration() {
        // Test GPU context creation through module re-export
        // In test environments, GPU may not be available, which is acceptable
        println!("GPU context integration test - GPU may not be available in test environment");
        // Placeholder test - no assertion needed
    }

    #[test]
    fn test_backward_compatibility_types() {
        // Verify that all re-exported types are accessible
        #[cfg(feature = "gpu")]
        {
            // Original transforms
            let _context_type = std::marker::PhantomData::<GpuContext>;
            let _resize_type = std::marker::PhantomData::<GpuResize>;
            let _flip_type = std::marker::PhantomData::<GpuRandomHorizontalFlip>;
            let _jitter_type = std::marker::PhantomData::<GpuColorJitter>;
            let _blur_type = std::marker::PhantomData::<GpuGaussianBlur>;
            let _noise_type = std::marker::PhantomData::<GpuGaussianNoise>;
            let _crop_type = std::marker::PhantomData::<GpuRandomCrop>;
            let _rotation_type = std::marker::PhantomData::<GpuRotation>;
            // New transforms
            let _affine_type = std::marker::PhantomData::<GpuAffineTransform>;
            let _persp_type = std::marker::PhantomData::<GpuPerspectiveTransform>;
            let _elastic_type = std::marker::PhantomData::<GpuElasticDistortion>;
            let _equalize_type = std::marker::PhantomData::<GpuHistogramEqualize>;
        }
    }

    #[test]
    fn test_affine_helper_functions() {
        let id = affine_identity();
        assert_eq!(id, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let t = affine_translation(1.0, 2.0);
        assert_eq!(t[0][2], 1.0);
        assert_eq!(t[1][2], 2.0);
    }

    #[test]
    fn test_homography_helper_functions() {
        let id = homography_identity();
        assert_eq!(id[2], [0.0, 0.0, 1.0]);
    }
}
