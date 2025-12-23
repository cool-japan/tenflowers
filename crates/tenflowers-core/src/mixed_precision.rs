#[cfg(feature = "gpu")]
use crate::tensor::TensorStorage;
use crate::{DType, Result, Tensor, TensorError};
use std::collections::HashMap;

/// Mixed precision optimization level (NVIDIA Apex-style)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    /// O0: Pure FP32 training (no mixed precision)
    O0,
    /// O1: Conservative mixed precision (recommended)
    /// - Compute-intensive ops in FP16
    /// - Memory-intensive ops in FP32
    /// - Dynamic loss scaling
    #[default]
    O1,
    /// O2: Fast mixed precision
    /// - Most ops in FP16
    /// - Batch norm in FP32
    /// - Master weights in FP32
    O2,
    /// O3: Pure FP16 training (experimental)
    /// - All ops in FP16 except loss scaling
    /// - Maximum performance, may affect accuracy
    O3,
}

/// Mixed precision training configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Enable automatic mixed precision
    pub enabled: bool,
    /// Optimization level
    pub opt_level: OptimizationLevel,
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
    /// Factor to multiply loss scale by when growing
    pub scale_growth_factor: f32,
    /// Factor to multiply loss scale by when shrinking (on overflow)
    pub scale_backoff_factor: f32,
    /// Operations that should use FP32 precision (whitelist)
    pub fp32_operations: Vec<String>,
    /// Operations that can use FP16 precision (blacklist exceptions)
    pub fp16_blacklist: Vec<String>,
    /// Keep master weights in FP32
    pub keep_master_weights: bool,
    /// Enable gradient clipping to prevent overflow
    pub enable_gradient_clipping: bool,
    /// Gradient clipping threshold
    pub gradient_clip_norm: f32,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            opt_level: OptimizationLevel::O1,
            loss_scale: 65536.0, // 2^16
            dynamic_loss_scaling: true,
            min_loss_scale: 1.0,
            max_loss_scale: 65536.0 * 65536.0, // 2^32
            scale_growth_interval: 2000,
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
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
            keep_master_weights: true,
            enable_gradient_clipping: false,
            gradient_clip_norm: 1.0,
        }
    }
}

impl MixedPrecisionConfig {
    /// Create O0 (pure FP32) configuration
    pub fn o0() -> Self {
        Self {
            enabled: false,
            opt_level: OptimizationLevel::O0,
            ..Default::default()
        }
    }

    /// Create O1 (conservative mixed precision) configuration
    pub fn o1() -> Self {
        Self {
            enabled: true,
            opt_level: OptimizationLevel::O1,
            loss_scale: 65536.0,
            keep_master_weights: true,
            ..Default::default()
        }
    }

    /// Create O2 (fast mixed precision) configuration
    pub fn o2() -> Self {
        Self {
            enabled: true,
            opt_level: OptimizationLevel::O2,
            loss_scale: 32768.0,
            keep_master_weights: true,
            fp32_operations: vec!["batch_norm".to_string(), "layer_norm".to_string()],
            ..Default::default()
        }
    }

    /// Create O3 (pure FP16) configuration
    pub fn o3() -> Self {
        Self {
            enabled: true,
            opt_level: OptimizationLevel::O3,
            loss_scale: 16384.0,
            keep_master_weights: false,
            fp32_operations: vec![],
            enable_gradient_clipping: true,
            gradient_clip_norm: 1.0,
            ..Default::default()
        }
    }
}

/// Statistics for mixed precision training
#[derive(Debug, Clone, Default)]
pub struct MixedPrecisionStatistics {
    /// Total training steps
    pub total_steps: usize,
    /// Steps with gradient overflow
    pub overflow_steps: usize,
    /// Number of scale increases
    pub scale_increases: usize,
    /// Number of scale decreases
    pub scale_decreases: usize,
    /// Cumulative scale value (for averaging)
    pub cumulative_scale: f64,
    /// Minimum scale reached
    pub min_scale_reached: f32,
    /// Maximum scale reached
    pub max_scale_reached: f32,
}

impl MixedPrecisionStatistics {
    /// Get overflow rate
    pub fn overflow_rate(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.overflow_steps as f64 / self.total_steps as f64
        }
    }

    /// Get average loss scale
    pub fn average_scale(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.cumulative_scale / self.total_steps as f64
        }
    }

    /// Get scale stability (lower is more stable)
    pub fn scale_stability(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            (self.scale_increases + self.scale_decreases) as f64 / self.total_steps as f64
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
    pub statistics: MixedPrecisionStatistics,
}

impl MixedPrecisionState {
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let loss_scale = config.loss_scale;
        let stats = MixedPrecisionStatistics {
            min_scale_reached: loss_scale,
            max_scale_reached: loss_scale,
            ..Default::default()
        };

        Self {
            config,
            current_loss_scale: loss_scale,
            steps_since_overflow: 0,
            overflow_detected: false,
            autocast_stack: Vec::new(),
            statistics: stats,
        }
    }

    /// Update loss scale based on gradient overflow detection
    pub fn update_loss_scale(&mut self, has_overflow: bool) {
        // Update statistics
        self.statistics.total_steps += 1;
        self.statistics.cumulative_scale += self.current_loss_scale as f64;

        if !self.config.dynamic_loss_scaling {
            return;
        }

        if has_overflow {
            // Reduce loss scale on overflow
            let new_scale = (self.current_loss_scale * self.config.scale_backoff_factor)
                .max(self.config.min_loss_scale);
            self.current_loss_scale = new_scale;
            self.steps_since_overflow = 0;
            self.overflow_detected = true;

            // Update statistics
            self.statistics.overflow_steps += 1;
            self.statistics.scale_decreases += 1;
            self.statistics.min_scale_reached = self.statistics.min_scale_reached.min(new_scale);
        } else {
            self.steps_since_overflow += 1;
            self.overflow_detected = false;

            // Increase loss scale after stable period
            if self.steps_since_overflow >= self.config.scale_growth_interval {
                let new_scale = (self.current_loss_scale * self.config.scale_growth_factor)
                    .min(self.config.max_loss_scale);
                self.current_loss_scale = new_scale;
                self.steps_since_overflow = 0;

                // Update statistics
                self.statistics.scale_increases += 1;
                self.statistics.max_scale_reached =
                    self.statistics.max_scale_reached.max(new_scale);
            }
        }
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> MixedPrecisionStatistics {
        self.statistics.clone()
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = MixedPrecisionStatistics::default();
        self.statistics.min_scale_reached = self.current_loss_scale;
        self.statistics.max_scale_reached = self.current_loss_scale;
    }

    /// Check if gradient overflow occurred
    pub fn check_gradient_overflow<T>(&self, gradients: &[Tensor<T>]) -> bool
    where
        T: scirs2_core::num_traits::Float
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
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
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
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
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
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
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
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
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
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

    /// Get current statistics
    pub fn get_statistics(&self) -> MixedPrecisionStatistics {
        self.state.get_statistics()
    }

    /// Reset scaler state and statistics
    pub fn reset(&mut self) {
        self.state.current_loss_scale = self.state.config.loss_scale;
        self.state.steps_since_overflow = 0;
        self.state.overflow_detected = false;
        self.state.reset_statistics();
    }

    /// Clip gradients by norm if enabled
    pub fn clip_gradients(&self, gradients: &mut [f32]) -> f32 {
        if !self.state.config.enable_gradient_clipping {
            return 0.0;
        }

        // Compute global norm
        let global_norm = gradients.iter().map(|&g| g * g).sum::<f32>().sqrt();

        if global_norm > self.state.config.gradient_clip_norm {
            let clip_coef = self.state.config.gradient_clip_norm / global_norm;
            for grad in gradients.iter_mut() {
                *grad *= clip_coef;
            }
        }

        global_norm
    }
}

/// Master weights manager for mixed precision training
///
/// Maintains FP32 copies of model weights while using FP16 for forward/backward passes.
/// This ensures optimizer updates maintain full precision accumulation.
#[derive(Debug, Clone)]
pub struct MasterWeightsManager {
    master_weights: HashMap<String, Vec<f32>>,
    enabled: bool,
}

impl MasterWeightsManager {
    /// Create a new master weights manager
    pub fn new(enabled: bool) -> Self {
        Self {
            master_weights: HashMap::new(),
            enabled,
        }
    }

    /// Store master weight for a parameter
    pub fn store(&mut self, name: String, weights: Vec<f32>) {
        if !self.enabled {
            return;
        }
        self.master_weights.insert(name, weights);
    }

    /// Retrieve master weight for a parameter
    pub fn retrieve(&self, name: &str) -> Option<&Vec<f32>> {
        self.master_weights.get(name)
    }

    /// Update master weight from FP16 gradients
    pub fn update_from_gradients(&mut self, name: &str, gradients: &[f32], learning_rate: f32) {
        if !self.enabled {
            return;
        }

        if let Some(master) = self.master_weights.get_mut(name) {
            for (weight, &grad) in master.iter_mut().zip(gradients.iter()) {
                *weight -= learning_rate * grad;
            }
        }
    }

    /// Copy master weights to FP16 working copy
    pub fn copy_to_working(&self, name: &str) -> Option<Vec<f32>> {
        self.master_weights.get(name).cloned()
    }

    /// Clear all master weights
    pub fn clear(&mut self) {
        self.master_weights.clear();
    }

    /// Get number of parameter groups stored
    pub fn count(&self) -> usize {
        self.master_weights.len()
    }

    /// Check if manager is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Check if tensor contains infinite or NaN values
fn has_inf_or_nan<T>(tensor: &Tensor<T>) -> bool
where
    T: scirs2_core::num_traits::Float + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
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

    #[test]
    fn test_optimization_levels() {
        let o0 = MixedPrecisionConfig::o0();
        assert_eq!(o0.opt_level, OptimizationLevel::O0);
        assert!(!o0.enabled);

        let o1 = MixedPrecisionConfig::o1();
        assert_eq!(o1.opt_level, OptimizationLevel::O1);
        assert!(o1.enabled);
        assert!(o1.keep_master_weights);

        let o2 = MixedPrecisionConfig::o2();
        assert_eq!(o2.opt_level, OptimizationLevel::O2);
        assert!(o2.enabled);
        assert_eq!(o2.fp32_operations.len(), 2); // Only batch_norm and layer_norm

        let o3 = MixedPrecisionConfig::o3();
        assert_eq!(o3.opt_level, OptimizationLevel::O3);
        assert!(o3.enabled);
        assert!(!o3.keep_master_weights);
        assert!(o3.enable_gradient_clipping);
        assert_eq!(o3.fp32_operations.len(), 0); // Pure FP16
    }

    #[test]
    fn test_mixed_precision_statistics() {
        let mut stats = MixedPrecisionStatistics::default();
        stats.total_steps = 100;
        stats.overflow_steps = 5;
        stats.scale_increases = 10;
        stats.scale_decreases = 5;
        stats.cumulative_scale = 6553600.0; // 100 steps * 65536 avg

        assert_eq!(stats.overflow_rate(), 0.05);
        assert_eq!(stats.average_scale(), 65536.0);
        assert_eq!(stats.scale_stability(), 0.15);
    }

    #[test]
    fn test_loss_scale_update_with_statistics() {
        let config = MixedPrecisionConfig {
            enabled: true,
            dynamic_loss_scaling: true,
            loss_scale: 1024.0,
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
            scale_growth_interval: 2,
            ..Default::default()
        };

        let mut state = MixedPrecisionState::new(config);

        // Test overflow
        let initial_scale = state.current_loss_scale;
        state.update_loss_scale(true);

        assert_eq!(state.current_loss_scale, initial_scale * 0.5);
        assert_eq!(state.statistics.overflow_steps, 1);
        assert_eq!(state.statistics.scale_decreases, 1);
        assert_eq!(state.statistics.min_scale_reached, initial_scale * 0.5);

        // Test scale growth
        state.update_loss_scale(false);
        state.update_loss_scale(false);

        assert_eq!(state.current_loss_scale, initial_scale); // Back to initial after growth
        assert_eq!(state.statistics.scale_increases, 1);
    }

    #[test]
    fn test_gradient_clipping() {
        let config = MixedPrecisionConfig {
            enabled: true,
            enable_gradient_clipping: true,
            gradient_clip_norm: 1.0,
            ..Default::default()
        };

        let scaler = GradientScaler::new(config);
        let mut gradients = vec![0.6, 0.8]; // Norm = 1.0, should not clip

        let norm = scaler.clip_gradients(&mut gradients);
        assert!((norm - 1.0).abs() < 1e-5);
        assert!((gradients[0] - 0.6).abs() < 1e-5);

        // Test clipping with large gradients
        let mut large_gradients = vec![3.0, 4.0]; // Norm = 5.0, should clip to 1.0

        let large_norm = scaler.clip_gradients(&mut large_gradients);
        assert!((large_norm - 5.0).abs() < 1e-5);

        let clipped_norm: f32 = large_gradients.iter().map(|&g| g * g).sum::<f32>().sqrt();
        assert!((clipped_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_master_weights_manager() {
        let mut manager = MasterWeightsManager::new(true);

        // Store weights
        manager.store("layer1".to_string(), vec![1.0, 2.0, 3.0]);
        manager.store("layer2".to_string(), vec![4.0, 5.0]);

        assert_eq!(manager.count(), 2);
        assert!(manager.is_enabled());

        // Retrieve weights
        let weights = manager.retrieve("layer1");
        assert_eq!(weights, Some(&vec![1.0, 2.0, 3.0]));

        // Update from gradients (SGD step)
        manager.update_from_gradients("layer1", &[0.1, 0.2, 0.3], 0.1);

        let updated = manager.retrieve("layer1").unwrap();
        assert!((updated[0] - 0.99).abs() < 1e-5); // 1.0 - 0.1 * 0.1
        assert!((updated[1] - 1.98).abs() < 1e-5); // 2.0 - 0.1 * 0.2
        assert!((updated[2] - 2.97).abs() < 1e-5); // 3.0 - 0.1 * 0.3

        // Copy to working
        let working_copy = manager.copy_to_working("layer1");
        assert!(working_copy.is_some());

        // Clear
        manager.clear();
        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn test_scaler_reset() {
        let config = MixedPrecisionConfig {
            enabled: true,
            loss_scale: 1024.0,
            ..Default::default()
        };

        let mut scaler = GradientScaler::new(config);

        // Modify state
        scaler.state.current_loss_scale = 512.0;
        scaler.state.steps_since_overflow = 10;
        scaler.state.statistics.total_steps = 100;

        // Reset
        scaler.reset();

        assert_eq!(scaler.get_scale(), 1024.0);
        assert_eq!(scaler.state.steps_since_overflow, 0);
        assert_eq!(scaler.state.statistics.total_steps, 0);
    }

    #[test]
    fn test_get_scaler_statistics() {
        let config = MixedPrecisionConfig {
            enabled: true,
            ..Default::default()
        };

        let mut scaler = GradientScaler::new(config);

        // Simulate some steps
        scaler.state.update_loss_scale(false);
        scaler.state.update_loss_scale(true);

        let stats = scaler.get_statistics();
        assert_eq!(stats.total_steps, 2);
        assert_eq!(stats.overflow_steps, 1);
    }

    #[test]
    fn test_disabled_master_weights() {
        let mut manager = MasterWeightsManager::new(false);

        // Should not store when disabled
        manager.store("layer1".to_string(), vec![1.0, 2.0]);
        assert_eq!(manager.count(), 0);
        assert!(!manager.is_enabled());
    }
}
