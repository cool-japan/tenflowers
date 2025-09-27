//! Half precision floating point support
//!
//! This module provides support for IEEE 754-2008 half precision (f16) and
//! Google's brain floating point (bf16) data types for mixed precision training.

pub use half::{bf16, f16};

/// Trait for half precision floating point types
pub trait HalfPrecision: Copy + Clone + Send + Sync + 'static {
    type FullPrecision: num_traits::Float;

    /// Convert to full precision (f32)
    fn to_f32(self) -> f32;

    /// Convert from full precision (f32)
    fn from_f32(value: f32) -> Self;

    /// Get the data type
    fn dtype() -> crate::DType;
}

impl HalfPrecision for f16 {
    type FullPrecision = f32;

    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }

    fn dtype() -> crate::DType {
        crate::DType::Float16
    }
}

impl HalfPrecision for bf16 {
    type FullPrecision = f32;

    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        bf16::from_f32(value)
    }

    fn dtype() -> crate::DType {
        crate::DType::BFloat16
    }
}

/// Mixed precision configuration for automatic mixed precision (AMP) training
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Whether to enable automatic mixed precision
    pub enabled: bool,
    /// Loss scaling factor to prevent gradient underflow
    pub loss_scale: f32,
    /// Growth factor for loss scaling
    pub growth_factor: f32,
    /// Backoff factor for loss scaling
    pub backoff_factor: f32,
    /// Growth interval (number of steps without overflow)
    pub growth_interval: u32,
    /// Counter for steps without overflow
    pub(crate) steps_without_overflow: u32,
    /// Whether to use bf16 instead of f16
    pub use_bfloat16: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            loss_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_without_overflow: 0,
            use_bfloat16: false,
        }
    }
}

impl MixedPrecisionConfig {
    /// Create a new mixed precision configuration with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable mixed precision training
    pub fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Set the initial loss scale
    pub fn with_loss_scale(mut self, scale: f32) -> Self {
        self.loss_scale = scale;
        self
    }

    /// Use bfloat16 instead of float16
    pub fn with_bfloat16(mut self) -> Self {
        self.use_bfloat16 = true;
        self
    }

    /// Check for gradient overflow and update loss scaling
    pub fn update_loss_scale(&mut self, has_overflow: bool) {
        if has_overflow {
            // Decrease loss scale and reset counter
            self.loss_scale *= self.backoff_factor;
            self.steps_without_overflow = 0;
        } else {
            // Increment counter
            self.steps_without_overflow += 1;

            // Increase loss scale if no overflow for growth_interval steps
            if self.steps_without_overflow >= self.growth_interval {
                self.loss_scale *= self.growth_factor;
                self.steps_without_overflow = 0;
            }
        }

        // Ensure loss scale doesn't become too small or too large
        self.loss_scale = self.loss_scale.clamp(1.0, f32::MAX / 1000.0);
    }

    /// Get the target half precision type
    pub fn target_dtype(&self) -> crate::DType {
        if self.use_bfloat16 {
            crate::DType::BFloat16
        } else {
            crate::DType::Float16
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        let value = 3.14159f32;
        let f16_val = f16::from_f32(value);
        let converted_back = f16_val.to_f32();

        // f16 has limited precision, so allow some error
        assert!((converted_back - value).abs() < 0.01);
    }

    #[test]
    fn test_bf16_conversion() {
        let value = 3.14159f32;
        let bf16_val = bf16::from_f32(value);
        let converted_back = bf16_val.to_f32();

        // bf16 has better precision than f16 for this range
        assert!((converted_back - value).abs() < 0.001);
    }

    #[test]
    fn test_mixed_precision_config() {
        let mut config = MixedPrecisionConfig::new()
            .enable()
            .with_loss_scale(1024.0)
            .with_bfloat16();

        assert!(config.enabled);
        assert_eq!(config.loss_scale, 1024.0);
        assert!(config.use_bfloat16);
        assert_eq!(config.target_dtype(), crate::DType::BFloat16);

        // Test overflow handling
        config.update_loss_scale(true);
        assert_eq!(config.loss_scale, 512.0); // 1024 * 0.5
        assert_eq!(config.steps_without_overflow, 0);

        // Test growth
        for _ in 0..config.growth_interval {
            config.update_loss_scale(false);
        }
        assert_eq!(config.loss_scale, 1024.0); // 512 * 2.0
    }

    #[test]
    fn test_dtype_mapping() {
        assert_eq!(f16::dtype(), crate::DType::Float16);
        assert_eq!(bf16::dtype(), crate::DType::BFloat16);
    }
}
