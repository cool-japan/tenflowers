#[cfg(feature = "gpu")]
use crate::tensor::TensorStorage;
use crate::{DType, Result, Tensor, TensorError};
use std::collections::HashMap;

/// Mixed precision training configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Enable automatic mixed precision
    pub enabled: bool,
    /// Loss scaling factor for gradient stability
    pub loss_scale: f32,
    /// Dynamic loss scaling configuration
    pub dynamic_loss_scaling: bool,
    /// Minimum loss scale value
    pub min_loss_scale: f32,
    /// Maximum loss scale value  
    pub max_loss_scale: f32,
    /// Number of steps without overflow before increasing scale
    pub scale_growth_interval: usize,
    /// Operations that should use FP32 precision (whitelist)
    pub fp32_operations: Vec<String>,
    /// Operations that can use FP16 precision (blacklist exceptions)
    pub fp16_blacklist: Vec<String>,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            loss_scale: 65536.0, // 2^16
            dynamic_loss_scaling: true,
            min_loss_scale: 1.0,
            max_loss_scale: 65536.0 * 65536.0, // 2^32
            scale_growth_interval: 2000,
            fp32_operations: vec![
                "softmax".to_string(),
                "log_softmax".to_string(),
                "cross_entropy".to_string(),
                "batch_norm".to_string(),
                "layer_norm".to_string(),
            ],
            fp16_blacklist: vec![
                "exp".to_string(),
                "log".to_string(),
                "sqrt".to_string(),
                "pow".to_string(),
            ],
        }
    }
}

/// Mixed precision state tracking
#[derive(Debug)]
pub struct MixedPrecisionState {
    pub config: MixedPrecisionConfig,
    pub current_loss_scale: f32,
    pub steps_since_overflow: usize,
    pub overflow_detected: bool,
    pub autocast_stack: Vec<DType>,
}

impl MixedPrecisionState {
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let loss_scale = config.loss_scale;
        Self {
            config,
            current_loss_scale: loss_scale,
            steps_since_overflow: 0,
            overflow_detected: false,
            autocast_stack: Vec::new(),
        }
    }

    /// Update loss scale based on gradient overflow detection
    pub fn update_loss_scale(&mut self, has_overflow: bool) {
        if !self.config.dynamic_loss_scaling {
            return;
        }

        if has_overflow {
            // Reduce loss scale on overflow
            self.current_loss_scale =
                (self.current_loss_scale / 2.0).max(self.config.min_loss_scale);
            self.steps_since_overflow = 0;
            self.overflow_detected = true;
        } else {
            self.steps_since_overflow += 1;
            self.overflow_detected = false;

            // Increase loss scale after stable period
            if self.steps_since_overflow >= self.config.scale_growth_interval {
                self.current_loss_scale =
                    (self.current_loss_scale * 2.0).min(self.config.max_loss_scale);
                self.steps_since_overflow = 0;
            }
        }
    }

    /// Check if gradient overflow occurred
    pub fn check_gradient_overflow<T>(&self, gradients: &[Tensor<T>]) -> bool
    where
        T: num_traits::Float + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        for grad in gradients {
            if has_inf_or_nan(grad) {
                return true;
            }
        }
        false
    }

    /// Scale loss for mixed precision training
    pub fn scale_loss<T>(&self, loss: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + num_traits::Float
            + num_traits::Zero
            + num_traits::One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        if !self.config.enabled {
            return Ok(loss.clone());
        }

        let scale_value = T::from(self.current_loss_scale).unwrap();
        let scale_tensor = Tensor::from_scalar(scale_value);
        loss.mul(&scale_tensor)
    }

    /// Unscale gradients after backpropagation
    pub fn unscale_gradients<T>(&self, gradients: &mut [Tensor<T>]) -> Result<()>
    where
        T: Clone
            + Default
            + num_traits::Float
            + num_traits::Zero
            + num_traits::One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        if !self.config.enabled {
            return Ok(());
        }

        let scale_factor = T::from(1.0 / self.current_loss_scale).unwrap();
        let scale_tensor = Tensor::from_scalar(scale_factor);

        for grad in gradients.iter_mut() {
            *grad = grad.mul(&scale_tensor)?;
        }

        Ok(())
    }
}

/// Automatic mixed precision context manager
pub struct AutocastContext {
    enabled: bool,
    target_dtype: DType,
    operation_overrides: HashMap<String, DType>,
}

impl AutocastContext {
    pub fn new(enabled: bool, target_dtype: DType) -> Self {
        let mut operation_overrides = HashMap::new();

        // Operations that should stay in FP32 for numerical stability
        let fp32_ops = vec![
            "softmax",
            "log_softmax",
            "cross_entropy",
            "batch_norm",
            "layer_norm",
            "group_norm",
            "exp",
            "log",
            "sqrt",
        ];

        for op in fp32_ops {
            operation_overrides.insert(op.to_string(), DType::Float32);
        }

        Self {
            enabled,
            target_dtype,
            operation_overrides,
        }
    }

    /// Get the appropriate dtype for an operation
    pub fn get_operation_dtype(&self, operation_name: &str, default_dtype: DType) -> DType {
        if !self.enabled {
            return default_dtype;
        }

        // Check for specific operation overrides
        if let Some(&override_dtype) = self.operation_overrides.get(operation_name) {
            return override_dtype;
        }

        // Use target dtype for compatible operations
        match default_dtype {
            DType::Float32 => self.target_dtype,
            DType::Float64 => DType::Float32, // Downcast F64 to F32 in autocast
            _ => default_dtype,               // Keep integer types unchanged
        }
    }

    /// Check if operation should use mixed precision
    pub fn should_autocast(&self, operation_name: &str, input_dtype: DType) -> bool {
        self.enabled
            && matches!(input_dtype, DType::Float32 | DType::Float64)
            && !self.operation_overrides.contains_key(operation_name)
    }

    /// Check if autocast is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Convert f32 tensor to half precision (f16)
pub fn to_half_f32(input: &Tensor<f32>) -> Result<Tensor<crate::half_precision::f16>> {
    use crate::half_precision::f16;

    let data = input.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple(
            "Cannot access tensor data for conversion".to_string(),
        )
    })?;

    let f16_data: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
    Tensor::from_vec(f16_data, input.shape().dims())
}

/// Convert f64 tensor to half precision (f16)
pub fn to_half_f64(input: &Tensor<f64>) -> Result<Tensor<crate::half_precision::f16>> {
    use crate::half_precision::f16;

    let data = input.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple(
            "Cannot access tensor data for conversion".to_string(),
        )
    })?;

    let f16_data: Vec<f16> = data.iter().map(|&v| f16::from_f32(v as f32)).collect();
    Tensor::from_vec(f16_data, input.shape().dims())
}

/// Convert f32 tensor to bfloat16
pub fn to_bfloat16_f32(input: &Tensor<f32>) -> Result<Tensor<crate::half_precision::bf16>> {
    use crate::half_precision::bf16;

    let data = input.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple(
            "Cannot access tensor data for conversion".to_string(),
        )
    })?;

    let bf16_data: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
    Tensor::from_vec(bf16_data, input.shape().dims())
}

/// Convert f64 tensor to bfloat16
pub fn to_bfloat16_f64(input: &Tensor<f64>) -> Result<Tensor<crate::half_precision::bf16>> {
    use crate::half_precision::bf16;

    let data = input.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple(
            "Cannot access tensor data for conversion".to_string(),
        )
    })?;

    let bf16_data: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    Tensor::from_vec(bf16_data, input.shape().dims())
}

/// Generic half precision conversion - dispatches to specific types
pub fn to_half<T>(_input: &Tensor<T>) -> Result<Tensor<crate::half_precision::f16>>
where
    T: Clone + Send + Sync + 'static,
{
    Err(TensorError::unsupported_operation_simple(
        "Generic half precision conversion not implemented - use to_half_f32 or to_half_f64"
            .to_string(),
    ))
}

/// Convert from f16 to f32
pub fn from_half_f32(input: &Tensor<crate::half_precision::f16>) -> Result<Tensor<f32>> {
    let data = input.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple(
            "Cannot access tensor data for conversion".to_string(),
        )
    })?;

    let f32_data: Vec<f32> = data.iter().map(|&v| v.to_f32()).collect();
    Tensor::from_vec(f32_data, input.shape().dims())
}

/// Convert from f16 to f64
pub fn from_half_f64(input: &Tensor<crate::half_precision::f16>) -> Result<Tensor<f64>> {
    let data = input.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple(
            "Cannot access tensor data for conversion".to_string(),
        )
    })?;

    let f64_data: Vec<f64> = data.iter().map(|&v| v.to_f32() as f64).collect();
    Tensor::from_vec(f64_data, input.shape().dims())
}

/// Convert from bf16 to f32
pub fn from_bfloat16_f32(input: &Tensor<crate::half_precision::bf16>) -> Result<Tensor<f32>> {
    let data = input.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple(
            "Cannot access tensor data for conversion".to_string(),
        )
    })?;

    let f32_data: Vec<f32> = data.iter().map(|&v| v.to_f32()).collect();
    Tensor::from_vec(f32_data, input.shape().dims())
}

/// Convert from bf16 to f64
pub fn from_bfloat16_f64(input: &Tensor<crate::half_precision::bf16>) -> Result<Tensor<f64>> {
    let data = input.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple(
            "Cannot access tensor data for conversion".to_string(),
        )
    })?;

    let f64_data: Vec<f64> = data.iter().map(|&v| v.to_f32() as f64).collect();
    Tensor::from_vec(f64_data, input.shape().dims())
}

/// Generic from half precision conversion - dispatches to specific types  
pub fn from_half<T>(_input: &Tensor<crate::half_precision::f16>) -> Result<Tensor<T>>
where
    T: Clone + Send + Sync + 'static,
{
    Err(TensorError::unsupported_operation_simple(
        "Generic from half precision conversion not implemented - use from_half_f32 or from_half_f64".to_string()
    ))
}

/// Gradient scaler for mixed precision training
#[derive(Debug)]
pub struct GradientScaler {
    state: MixedPrecisionState,
}

impl GradientScaler {
    pub fn new(config: MixedPrecisionConfig) -> Self {
        Self {
            state: MixedPrecisionState::new(config),
        }
    }

    /// Scale loss before backpropagation
    pub fn scale<T>(&self, loss: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + num_traits::Float
            + num_traits::Zero
            + num_traits::One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        self.state.scale_loss(loss)
    }

    /// Check gradients and unscale them, returning whether step should be skipped
    pub fn unscale_gradients_and_check<T>(&mut self, gradients: &mut [Tensor<T>]) -> Result<bool>
    where
        T: Clone
            + Default
            + num_traits::Float
            + num_traits::Zero
            + num_traits::One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Check for gradient overflow
        let has_overflow = self.state.check_gradient_overflow(gradients);

        if has_overflow {
            // Skip step on overflow
            self.state.update_loss_scale(true);
            return Ok(false);
        }

        // Unscale gradients
        self.state.unscale_gradients(gradients)?;

        // Update loss scale for successful step
        self.state.update_loss_scale(false);

        Ok(true)
    }

    /// Get current loss scale
    pub fn get_scale(&self) -> f32 {
        self.state.current_loss_scale
    }

    /// Update configuration
    pub fn update_config(&mut self, config: MixedPrecisionConfig) {
        // Update current loss scale if it has changed
        if config.loss_scale != self.state.config.loss_scale {
            self.state.current_loss_scale = config.loss_scale;
        }
        self.state.config = config;
    }

    /// Check if mixed precision is enabled
    pub fn is_enabled(&self) -> bool {
        self.state.config.enabled
    }

    /// Get current configuration
    pub fn get_config(&self) -> MixedPrecisionConfig {
        self.state.config.clone()
    }
}

/// Check if tensor contains infinite or NaN values
fn has_inf_or_nan<T>(tensor: &Tensor<T>) -> bool
where
    T: num_traits::Float + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // Check CPU tensor data
    if let Some(data) = tensor.as_slice() {
        return data
            .iter()
            .any(|&value| value.is_infinite() || value.is_nan());
    }

    // For GPU tensors, use GPU kernel for inf/nan detection
    #[cfg(feature = "gpu")]
    {
        if tensor.device().is_gpu() {
            // Use GPU kernel for inf/nan detection
            use crate::gpu::{buffer::GpuBuffer, ReductionOp};

            // Get GPU buffer from tensor storage
            if let TensorStorage::Gpu(ref gpu_buffer) = tensor.storage {
                // Execute inf/nan detection kernel
                match crate::gpu::ops::execute_reduction_op(
                    gpu_buffer,
                    ReductionOp::InfNanDetection,
                    None, // Reduce over all axes to get single value
                ) {
                    Ok(result_buffer) => {
                        // Read result back from GPU
                        match result_buffer.to_cpu() {
                            Ok(result_data) => {
                                // Check if any inf/nan was found
                                if !result_data.is_empty() {
                                    return !result_data[0].is_zero();
                                }
                            }
                            Err(_) => {
                                // Fall back to conservative approach on error
                                return false;
                            }
                        }
                    }
                    Err(_) => {
                        // Fall back to conservative approach on error
                        return false;
                    }
                }
            }
        }
        false
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Enable automatic mixed precision for operations (uses FP16 by default)
pub fn enable_autocast() -> AutocastContext {
    AutocastContext::new(true, DType::Float16)
}

/// Enable automatic mixed precision with bfloat16
pub fn enable_autocast_bfloat16() -> AutocastContext {
    AutocastContext::new(true, DType::BFloat16)
}

/// Disable automatic mixed precision
pub fn disable_autocast() -> AutocastContext {
    AutocastContext::new(false, DType::Float32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_precision_config() {
        let config = MixedPrecisionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.loss_scale, 65536.0);
        assert!(config.dynamic_loss_scaling);
    }

    #[test]
    fn test_autocast_context() {
        let ctx = enable_autocast();
        assert!(ctx.enabled);

        // Should use Float32 for softmax (numerical stability)
        assert_eq!(
            ctx.get_operation_dtype("softmax", DType::Float32),
            DType::Float32
        );

        // Should use Float16 for regular operations
        assert_eq!(
            ctx.get_operation_dtype("conv2d", DType::Float32),
            DType::Float16
        );

        // Test bfloat16 autocast
        let ctx_bf16 = enable_autocast_bfloat16();
        assert_eq!(
            ctx_bf16.get_operation_dtype("conv2d", DType::Float32),
            DType::BFloat16
        );
    }

    #[test]
    fn test_gradient_scaler() {
        let config = MixedPrecisionConfig {
            enabled: true,
            ..Default::default()
        };
        let scaler = GradientScaler::new(config);
        assert_eq!(scaler.get_scale(), 65536.0);
    }

    #[test]
    fn test_half_precision_conversions() {
        use crate::Tensor;

        // Test f32 to f16 conversion
        let f32_tensor = Tensor::<f32>::from_vec(vec![1.0, 2.5, -3.14159], &[3]).unwrap();
        let f16_tensor = to_half_f32(&f32_tensor).unwrap();
        let f32_back = from_half_f32(&f16_tensor).unwrap();

        // Check that conversion is reasonably close (f16 has limited precision)
        let original_data = f32_tensor.as_slice().unwrap();
        let converted_data = f32_back.as_slice().unwrap();
        for (orig, conv) in original_data.iter().zip(converted_data.iter()) {
            assert!(
                (orig - conv).abs() < 0.01,
                "f16 conversion error too large: {} vs {}",
                orig,
                conv
            );
        }

        // Test f32 to bf16 conversion
        let bf16_tensor = to_bfloat16_f32(&f32_tensor).unwrap();
        let f32_back_bf16 = from_bfloat16_f32(&bf16_tensor).unwrap();

        // bf16 should have better precision than f16 in this range
        let bf16_data = f32_back_bf16.as_slice().unwrap();
        for (orig, conv) in original_data.iter().zip(bf16_data.iter()) {
            assert!(
                (orig - conv).abs() < 0.001,
                "bf16 conversion error too large: {} vs {}",
                orig,
                conv
            );
        }
    }

    #[test]
    fn test_mixed_precision_dtype_mapping() {
        use crate::DType;

        // Test dtype sizes
        assert_eq!(DType::Float16.size(), 2);
        assert_eq!(DType::BFloat16.size(), 2);
        assert_eq!(DType::Float32.size(), 4);

        // Test dtype names
        assert_eq!(DType::Float16.name(), "float16");
        assert_eq!(DType::BFloat16.name(), "bfloat16");
    }
}
