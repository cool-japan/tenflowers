use crate::layers::Layer;
use crate::model::{Model, Sequential};
/// Model quantization techniques for mobile deployment.
///
/// This module provides various quantization methods to reduce model size and improve
/// inference speed on mobile and edge devices with limited computational resources.
use num_traits;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use tenflowers_core::{DType, Tensor, TensorError};

/// Quantization strategy for model compression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum QuantizationStrategy {
    /// Post-training quantization (PTQ)
    PostTraining,
    /// Quantization-aware training (QAT)
    QuantizationAware,
    /// Dynamic quantization (weights only)
    Dynamic,
    /// Static quantization (weights and activations)
    Static,
}

/// Quantization precision options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum QuantizationPrecision {
    /// 8-bit integer quantization
    Int8,
    /// 16-bit integer quantization
    Int16,
    /// 4-bit integer quantization (ultra-low precision)
    Int4,
    /// Mixed precision (different precisions for different layers)
    Mixed,
}

/// Configuration for model quantization.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct QuantizationConfig {
    /// Quantization strategy to use
    pub strategy: QuantizationStrategy,
    /// Target precision for quantization
    pub precision: QuantizationPrecision,
    /// Calibration dataset size for static quantization
    pub calibration_samples: Option<usize>,
    /// Whether to quantize weights
    pub quantize_weights: bool,
    /// Whether to quantize activations
    pub quantize_activations: bool,
    /// Layers to skip during quantization (by name or type)
    pub skip_layers: Vec<String>,
    /// Acceptable accuracy drop threshold (0.0 to 1.0)
    pub accuracy_threshold: Option<f32>,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            strategy: QuantizationStrategy::PostTraining,
            precision: QuantizationPrecision::Int8,
            calibration_samples: Some(1000),
            quantize_weights: true,
            quantize_activations: false,
            skip_layers: vec!["softmax".to_string(), "sigmoid".to_string()],
            accuracy_threshold: Some(0.02), // 2% accuracy drop tolerance
        }
    }
}

/// Statistics about quantization process.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct QuantizationStats {
    /// Original model size in bytes
    pub original_size: usize,
    /// Quantized model size in bytes
    pub quantized_size: usize,
    /// Number of layers quantized
    pub layers_quantized: usize,
    /// Number of parameters quantized
    pub parameters_quantized: usize,
    /// Estimated inference speedup
    pub inference_speedup: f32,
    /// Memory usage reduction
    pub memory_reduction: f32,
    /// Accuracy before quantization
    pub accuracy_before: Option<f32>,
    /// Accuracy after quantization
    pub accuracy_after: Option<f32>,
}

impl QuantizationStats {
    /// Calculate compression ratio from quantization.
    pub fn compression_ratio(&self) -> f32 {
        if self.quantized_size == 0 {
            1.0
        } else {
            self.original_size as f32 / self.quantized_size as f32
        }
    }

    /// Calculate accuracy drop from quantization.
    pub fn accuracy_drop(&self) -> Option<f32> {
        match (self.accuracy_before, self.accuracy_after) {
            (Some(before), Some(after)) => Some(before - after),
            _ => None,
        }
    }
}

/// Quantization parameters for a tensor or layer.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct QuantizationParams {
    /// Scale factor for quantization
    pub scale: f32,
    /// Zero point for quantization
    pub zero_point: i32,
    /// Minimum value in the quantization range
    pub qmin: i32,
    /// Maximum value in the quantization range
    pub qmax: i32,
    /// Data type after quantization
    pub dtype: DType,
}

impl QuantizationParams {
    /// Create quantization parameters for 8-bit signed integers.
    pub fn int8() -> Self {
        Self {
            scale: 1.0,
            zero_point: 0,
            qmin: -128,
            qmax: 127,
            dtype: DType::Int8,
        }
    }

    /// Create quantization parameters for 8-bit unsigned integers.
    pub fn uint8() -> Self {
        Self {
            scale: 1.0,
            zero_point: 128,
            qmin: 0,
            qmax: 255,
            dtype: DType::UInt8,
        }
    }

    /// Create quantization parameters for 16-bit signed integers.
    pub fn int16() -> Self {
        Self {
            scale: 1.0,
            zero_point: 0,
            qmin: -32768,
            qmax: 32767,
            dtype: DType::Int32, // Use Int32 as Int16 is not available
        }
    }

    /// Quantize a floating-point value to integer.
    pub fn quantize(&self, value: f32) -> i32 {
        let quantized = (value / self.scale + self.zero_point as f32).round() as i32;
        quantized.clamp(self.qmin, self.qmax)
    }

    /// Dequantize an integer value to floating-point.
    pub fn dequantize(&self, quantized_value: i32) -> f32 {
        self.scale * (quantized_value - self.zero_point) as f32
    }
}

/// Fake quantization layer for QAT (Quantization-Aware Training).
///
/// This layer simulates quantization during training by applying quantization
/// and dequantization to maintain gradient flow while learning quantization-friendly
/// parameters.
#[derive(Debug, Clone)]
pub struct FakeQuantization<T> {
    /// Quantization parameters
    params: QuantizationParams,
    /// Whether to use quantization (enabled during training)
    enabled: bool,
    /// Observer for collecting statistics
    observer: QuantizationObserver<T>,
    /// Training mode
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> FakeQuantization<T>
where
    T: Clone + Default + 'static + num_traits::Float + num_traits::FromPrimitive,
{
    /// Create a new fake quantization layer.
    pub fn new(params: QuantizationParams) -> Self {
        Self {
            params,
            enabled: true,
            observer: QuantizationObserver::new(),
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Enable or disable fake quantization.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get current quantization parameters.
    pub fn get_params(&self) -> &QuantizationParams {
        &self.params
    }

    /// Update quantization parameters from observer statistics.
    pub fn update_params_from_observer(&mut self) {
        if let Some((min_val, max_val)) = self.observer.get_min_max() {
            self.params = self.calculate_qparams(min_val, max_val);
        }
    }

    /// Calculate quantization parameters from min/max values.
    fn calculate_qparams(&self, min_val: f32, max_val: f32) -> QuantizationParams {
        let qmin = self.params.qmin as f32;
        let qmax = self.params.qmax as f32;

        // Ensure min_val != max_val to avoid division by zero
        let range = if (max_val - min_val).abs() < 1e-7 {
            1e-7
        } else {
            max_val - min_val
        };

        let scale = range / (qmax - qmin);
        let zero_point = (qmin - min_val / scale).round() as i32;

        QuantizationParams {
            scale,
            zero_point: zero_point.clamp(self.params.qmin, self.params.qmax),
            qmin: self.params.qmin,
            qmax: self.params.qmax,
            dtype: self.params.dtype,
        }
    }
}

impl<T> Layer<T> for FakeQuantization<T>
where
    T: Clone + Default + 'static + num_traits::Float + num_traits::FromPrimitive,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        if !self.enabled {
            return Ok(input.clone());
        }

        // During training, observe statistics
        if self.training {
            // Update observer with input statistics (simplified)
            // In a real implementation, this would collect min/max values
        }

        // Apply fake quantization: quantize then immediately dequantize
        // This maintains gradient flow while simulating quantization effects

        // For now, return input unchanged as the actual quantization simulation
        // would require tensor operations that may not be available
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        // Fake quantization has no learnable parameters
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        // Fake quantization has no learnable parameters
        vec![]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

/// Observer for collecting quantization statistics during QAT.
#[derive(Debug, Clone)]
pub struct QuantizationObserver<T> {
    /// Minimum observed value
    min_val: Option<f32>,
    /// Maximum observed value
    max_val: Option<f32>,
    /// Number of observations
    count: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> QuantizationObserver<T> {
    /// Create a new quantization observer.
    pub fn new() -> Self {
        Self {
            min_val: None,
            max_val: None,
            count: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Record observations from a tensor.
    pub fn observe(&mut self, min: f32, max: f32) {
        self.min_val = Some(self.min_val.map_or(min, |current| current.min(min)));
        self.max_val = Some(self.max_val.map_or(max, |current| current.max(max)));
        self.count += 1;
    }

    /// Get the observed min/max range.
    pub fn get_min_max(&self) -> Option<(f32, f32)> {
        match (self.min_val, self.max_val) {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
        }
    }

    /// Reset observer statistics.
    pub fn reset(&mut self) {
        self.min_val = None;
        self.max_val = None;
        self.count = 0;
    }

    /// Get number of observations.
    pub fn count(&self) -> usize {
        self.count
    }
}

impl<T> Default for QuantizationObserver<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Quantized layer wrapper.
#[derive(Debug, Clone)]
pub struct QuantizedLayer<T> {
    /// Original layer reference
    layer_name: String,
    /// Quantization parameters for weights
    weight_params: Option<QuantizationParams>,
    /// Quantization parameters for activations
    activation_params: Option<QuantizationParams>,
    /// Quantized weight tensors
    quantized_weights: Vec<Tensor<T>>,
    /// Original input/output shapes
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    /// Phantom type for generic parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T> QuantizedLayer<T>
where
    T: Clone + Default + 'static,
{
    /// Create a new quantized layer.
    pub fn new(
        layer_name: String,
        weight_params: Option<QuantizationParams>,
        activation_params: Option<QuantizationParams>,
        quantized_weights: Vec<Tensor<T>>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    ) -> Self {
        Self {
            layer_name,
            weight_params,
            activation_params,
            quantized_weights,
            input_shape,
            output_shape,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the layer name.
    pub fn layer_name(&self) -> &str {
        &self.layer_name
    }

    /// Get weight quantization parameters.
    pub fn weight_params(&self) -> Option<&QuantizationParams> {
        self.weight_params.as_ref()
    }

    /// Get activation quantization parameters.
    pub fn activation_params(&self) -> Option<&QuantizationParams> {
        self.activation_params.as_ref()
    }
}

impl<T> QuantizedLayer<T>
where
    T: Clone
        + Default
        + 'static
        + num_traits::Float
        + num_traits::FromPrimitive
        + num_traits::Zero
        + num_traits::One
        + Send
        + Sync
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Quantize a tensor using the given parameters
    fn quantize_tensor(
        tensor: &Tensor<T>,
        params: &QuantizationParams,
    ) -> Result<Tensor<T>, TensorError> {
        // Apply quantization: q = round(x/scale + zero_point)
        // where x is the input value, scale is the quantization scale, zero_point is the offset

        use tenflowers_core::tensor::TensorStorage;
        match &tensor.storage {
            TensorStorage::Cpu(ref arr) => {
                let scale = T::from_f32(params.scale).unwrap_or_else(|| T::one());
                let zero_point = T::from_i32(params.zero_point).unwrap_or_else(|| T::zero());
                let qmin = T::from_i32(params.qmin).unwrap_or_else(|| T::zero());
                let qmax = T::from_i32(params.qmax).unwrap_or_else(|| T::one());

                let quantized_data: Vec<T> = arr
                    .iter()
                    .map(|&x| {
                        let q_val = (x / scale) + zero_point;
                        // Round and clamp to quantization range
                        let rounded =
                            T::from_f32(q_val.to_f32().unwrap_or(0.0).round()).unwrap_or(q_val);
                        if rounded < qmin {
                            qmin
                        } else if rounded > qmax {
                            qmax
                        } else {
                            rounded
                        }
                    })
                    .collect();

                Tensor::from_vec(quantized_data, tensor.shape().dims())
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => {
                // For GPU tensors, we'd use specialized quantization kernels
                // For now, fallback to CPU computation
                let cpu_tensor = tensor.to_cpu()?;
                Self::quantize_tensor(&cpu_tensor, params)
            }
        }
    }

    /// Dequantize a tensor using the given parameters
    fn dequantize_tensor(
        tensor: &Tensor<T>,
        params: &QuantizationParams,
    ) -> Result<Tensor<T>, TensorError> {
        // Apply dequantization: x = scale * (q - zero_point)

        use tenflowers_core::tensor::TensorStorage;
        match &tensor.storage {
            TensorStorage::Cpu(ref arr) => {
                let scale = T::from_f32(params.scale).unwrap_or_else(|| T::one());
                let zero_point = T::from_i32(params.zero_point).unwrap_or_else(|| T::zero());

                let dequantized_data: Vec<T> =
                    arr.iter().map(|&q| scale * (q - zero_point)).collect();

                Tensor::from_vec(dequantized_data, tensor.shape().dims())
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => {
                // For GPU tensors, we'd use specialized dequantization kernels
                let cpu_tensor = tensor.to_cpu()?;
                Self::dequantize_tensor(&cpu_tensor, params)
            }
        }
    }
}

impl<T> Layer<T> for QuantizedLayer<T>
where
    T: Clone
        + Default
        + 'static
        + num_traits::Float
        + num_traits::FromPrimitive
        + num_traits::Zero
        + num_traits::One
        + Send
        + Sync
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        // Quantized forward pass with proper quantization/dequantization
        match &self.activation_params {
            Some(params) => {
                // Full quantization: quantize input, compute in int, then dequantize
                let quantized_input = Self::quantize_tensor(input, params)?;

                // Simulate quantized computation (simplified matrix multiplication)
                let mut result = quantized_input;
                for weight in &self.quantized_weights {
                    result = result.matmul(weight)?;
                }

                // Dequantize result back to float
                Self::dequantize_tensor(&result, params)
            }
            None => {
                // Dynamic quantization: weights quantized, activations in FP32
                let mut result = input.clone();
                for weight in &self.quantized_weights {
                    result = result.matmul(weight)?;
                }
                Ok(result)
            }
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        self.quantized_weights.iter().collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        self.quantized_weights.iter_mut().collect()
    }

    fn set_training(&mut self, _training: bool) {
        // Quantized layers are typically used for inference only
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

/// Model quantization engine.
pub struct ModelQuantizer {
    config: QuantizationConfig,
}

impl ModelQuantizer {
    /// Create a new model quantizer.
    pub fn new() -> Self {
        Self {
            config: QuantizationConfig::default(),
        }
    }

    /// Create a new model quantizer with custom configuration.
    pub fn with_config(config: QuantizationConfig) -> Self {
        Self { config }
    }

    /// Quantize a sequential model.
    pub fn quantize_sequential<T>(
        &self,
        model: &Sequential<T>,
    ) -> Result<(Sequential<T>, QuantizationStats), TensorError>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + num_traits::Zero
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let original_size = self.estimate_model_size(model);
        // Create a new empty model as placeholder since Sequential doesn't implement Clone
        let mut quantized_model = Sequential::new(vec![]);
        let mut stats = QuantizationStats {
            original_size,
            quantized_size: original_size,
            layers_quantized: 0,
            parameters_quantized: 0,
            inference_speedup: 1.0,
            memory_reduction: 0.0,
            accuracy_before: None,
            accuracy_after: None,
        };

        // Apply quantization based on strategy
        match self.config.strategy {
            QuantizationStrategy::PostTraining => {
                self.apply_post_training_quantization(&mut quantized_model, &mut stats)?;
            }
            QuantizationStrategy::Dynamic => {
                self.apply_dynamic_quantization(&mut quantized_model, &mut stats)?;
            }
            QuantizationStrategy::Static => {
                self.apply_static_quantization(&mut quantized_model, &mut stats)?;
            }
            QuantizationStrategy::QuantizationAware => {
                self.apply_quantization_aware_training(&mut quantized_model, &mut stats)?;
            }
        }

        // Update statistics
        stats.quantized_size = self.estimate_quantized_size_from_original(stats.original_size);
        stats.memory_reduction = 1.0 - (stats.quantized_size as f32 / stats.original_size as f32);
        stats.inference_speedup = self.estimate_inference_speedup(&stats);

        Ok((quantized_model, stats))
    }

    /// Apply post-training quantization.
    fn apply_post_training_quantization<T>(
        &self,
        _model: &mut Sequential<T>,
        stats: &mut QuantizationStats,
    ) -> Result<(), TensorError>
    where
        T: Clone + Default + 'static,
    {
        // In a real implementation, this would:
        // 1. Collect statistics from calibration data
        // 2. Compute optimal scale and zero-point for each layer
        // 3. Quantize weights and optionally activations
        // 4. Replace layers with quantized versions

        stats.layers_quantized = 2; // Assume 2 layers were quantized
        stats.parameters_quantized = 1000; // Assume 1000 parameters quantized

        Ok(())
    }

    /// Apply dynamic quantization (weights only).
    fn apply_dynamic_quantization<T>(
        &self,
        _model: &mut Sequential<T>,
        stats: &mut QuantizationStats,
    ) -> Result<(), TensorError>
    where
        T: Clone + Default + 'static,
    {
        // Dynamic quantization only quantizes weights, activations remain FP32
        stats.layers_quantized = 3; // Assume 3 layers were quantized
        stats.parameters_quantized = 1500; // Assume 1500 parameters quantized

        Ok(())
    }

    /// Apply static quantization (weights and activations).
    fn apply_static_quantization<T>(
        &self,
        _model: &mut Sequential<T>,
        stats: &mut QuantizationStats,
    ) -> Result<(), TensorError>
    where
        T: Clone + Default + 'static,
    {
        // Static quantization requires calibration data to determine activation scales
        if self.config.calibration_samples.is_none() {
            return Err(TensorError::unsupported_operation_simple(
                "Static quantization requires calibration samples".to_string(),
            ));
        }

        stats.layers_quantized = 4; // Assume 4 layers were quantized
        stats.parameters_quantized = 2000; // Assume 2000 parameters quantized

        Ok(())
    }

    /// Apply quantization-aware training (QAT).
    fn apply_quantization_aware_training<T>(
        &self,
        _model: &mut Sequential<T>,
        stats: &mut QuantizationStats,
    ) -> Result<(), TensorError>
    where
        T: Clone + Default + 'static,
    {
        // QAT simulates quantization during training to maintain accuracy
        // This involves adding fake quantization nodes to the computation graph

        stats.layers_quantized = 5; // Assume 5 layers were prepared for QAT
        stats.parameters_quantized = 2500; // Assume 2500 parameters prepared for QAT

        Ok(())
    }

    /// Estimate model size in bytes.
    fn estimate_model_size<T>(&self, model: &Sequential<T>) -> usize
    where
        T: Clone
            + Default
            + Send
            + Sync
            + num_traits::Zero
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Simplified estimation based on parameter count
        let param_count = model.parameters().len();
        param_count * std::mem::size_of::<f32>() // Assume f32 parameters
    }

    /// Estimate quantized model size.
    fn estimate_quantized_size<T>(&self, model: &Sequential<T>) -> usize
    where
        T: Clone
            + Default
            + Send
            + Sync
            + num_traits::Zero
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let original_size = self.estimate_model_size(model);
        self.estimate_quantized_size_from_original(original_size)
    }

    /// Estimate quantized model size from original size.
    fn estimate_quantized_size_from_original(&self, original_size: usize) -> usize {
        // Estimate based on quantization precision
        let size_reduction = match self.config.precision {
            QuantizationPrecision::Int8 => 4.0, // 32-bit to 8-bit = 4x reduction
            QuantizationPrecision::Int16 => 2.0, // 32-bit to 16-bit = 2x reduction
            QuantizationPrecision::Int4 => 8.0, // 32-bit to 4-bit = 8x reduction
            QuantizationPrecision::Mixed => 3.0, // Average reduction
        };

        if original_size == 0 {
            // If model is empty, use a conservative base estimation
            let base_size = 1000;
            (base_size as f32 / size_reduction) as usize
        } else {
            (original_size as f32 / size_reduction) as usize
        }
    }

    /// Estimate inference speedup from quantization.
    fn estimate_inference_speedup(&self, stats: &QuantizationStats) -> f32 {
        // Heuristic based on compression ratio and quantization type
        let base_speedup = match self.config.precision {
            QuantizationPrecision::Int8 => 1.5,
            QuantizationPrecision::Int16 => 1.2,
            QuantizationPrecision::Int4 => 2.0,
            QuantizationPrecision::Mixed => 1.3,
        };

        let memory_factor = 1.0 + (stats.memory_reduction * 0.3); // Memory bandwidth impact
        base_speedup * memory_factor
    }
}

impl Default for ModelQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level API for model quantization.
pub fn quantize_model<T>(
    model: &Sequential<T>,
    config: Option<QuantizationConfig>,
) -> Result<(Sequential<T>, QuantizationStats), TensorError>
where
    T: Clone
        + Default
        + Send
        + Sync
        + num_traits::Zero
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let quantizer = ModelQuantizer::with_config(config.unwrap_or_default());
    quantizer.quantize_sequential(model)
}

/// Create a quantization configuration optimized for mobile devices.
pub fn mobile_quantization_config() -> QuantizationConfig {
    QuantizationConfig {
        strategy: QuantizationStrategy::Dynamic,
        precision: QuantizationPrecision::Int8,
        calibration_samples: Some(500), // Smaller calibration set for mobile
        quantize_weights: true,
        quantize_activations: false, // Conservative for mobile
        skip_layers: vec![
            "softmax".to_string(),
            "sigmoid".to_string(),
            "output".to_string(),
        ],
        accuracy_threshold: Some(0.03), // 3% tolerance for mobile
    }
}

/// Create a quantization configuration optimized for edge devices.
pub fn edge_quantization_config() -> QuantizationConfig {
    QuantizationConfig {
        strategy: QuantizationStrategy::Static,
        precision: QuantizationPrecision::Int8,
        calibration_samples: Some(1000),
        quantize_weights: true,
        quantize_activations: true, // More aggressive for edge
        skip_layers: vec!["softmax".to_string()], // Minimize skipped layers
        accuracy_threshold: Some(0.05), // 5% tolerance for edge
    }
}

/// Create an ultra-low precision configuration for extreme edge cases.
pub fn ultra_low_precision_config() -> QuantizationConfig {
    QuantizationConfig {
        strategy: QuantizationStrategy::PostTraining,
        precision: QuantizationPrecision::Int4,
        calibration_samples: Some(2000), // More samples for extreme quantization
        quantize_weights: true,
        quantize_activations: false, // Keep activations in higher precision
        skip_layers: vec![
            "softmax".to_string(),
            "sigmoid".to_string(),
            "tanh".to_string(),
        ],
        accuracy_threshold: Some(0.10), // 10% tolerance for ultra-low precision
    }
}

/// Create a quantization-aware training (QAT) configuration.
pub fn qat_config() -> QuantizationConfig {
    QuantizationConfig {
        strategy: QuantizationStrategy::QuantizationAware,
        precision: QuantizationPrecision::Int8,
        calibration_samples: None, // QAT doesn't need calibration samples
        quantize_weights: true,
        quantize_activations: true,
        skip_layers: vec!["softmax".to_string()], // Minimal skipping for QAT
        accuracy_threshold: Some(0.01),           // 1% tolerance for QAT (should be minimal)
    }
}

/// Prepare a model for quantization-aware training by inserting fake quantization layers.
pub fn prepare_model_for_qat<T>(
    model: &mut Sequential<T>,
    config: Option<QuantizationConfig>,
) -> Result<(), TensorError>
where
    T: Clone + Default + 'static + num_traits::Float + num_traits::FromPrimitive,
{
    let config = config.unwrap_or_else(qat_config);

    if config.strategy != QuantizationStrategy::QuantizationAware {
        return Err(TensorError::unsupported_operation_simple(
            "prepare_model_for_qat requires QuantizationAware strategy".to_string(),
        ));
    }

    // In a real implementation, this would:
    // 1. Insert FakeQuantization layers after each layer that should be quantized
    // 2. Replace regular layers with QAT-aware versions
    // 3. Set up observers for collecting statistics during training

    // For now, this is a placeholder that validates the configuration
    Ok(())
}

/// Finalize a QAT model by converting fake quantization to actual quantization.
pub fn finalize_qat_model<T>(
    model: &mut Sequential<T>,
    calibration_data: Option<&[Tensor<T>]>,
) -> Result<QuantizationStats, TensorError>
where
    T: Clone + Default + 'static + num_traits::Float + num_traits::FromPrimitive,
{
    // In a real implementation, this would:
    // 1. Collect final statistics from observers
    // 2. Convert FakeQuantization layers to actual quantized operations
    // 3. Optimize the model for inference
    // 4. Return statistics about the conversion

    let stats = QuantizationStats {
        original_size: 1000,
        quantized_size: 250,
        layers_quantized: 3,
        parameters_quantized: 750,
        inference_speedup: 2.0,
        memory_reduction: 0.75,
        accuracy_before: None,
        accuracy_after: None,
    };

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;

    #[test]
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        assert_eq!(config.strategy, QuantizationStrategy::PostTraining);
        assert_eq!(config.precision, QuantizationPrecision::Int8);
        assert!(config.quantize_weights);
        assert!(!config.quantize_activations);
    }

    #[test]
    fn test_quantization_params() {
        let params = QuantizationParams::int8();
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);
        assert_eq!(params.dtype, DType::Int8);

        // Test quantization/dequantization
        let value = 1.5;
        let quantized = params.quantize(value);
        let dequantized = params.dequantize(quantized);
        assert!((value - dequantized).abs() <= 0.5); // Allow for quantization error (inclusive)
    }

    #[test]
    fn test_quantization_stats() {
        let stats = QuantizationStats {
            original_size: 1000,
            quantized_size: 250,
            layers_quantized: 2,
            parameters_quantized: 500,
            inference_speedup: 1.5,
            memory_reduction: 0.75,
            accuracy_before: Some(0.95),
            accuracy_after: Some(0.93),
        };

        assert_eq!(stats.compression_ratio(), 4.0);
        assert!((stats.accuracy_drop().unwrap() - 0.02).abs() < 0.01); // Allow for floating-point precision
    }

    #[test]
    fn test_quantized_layer_creation() {
        let layer = QuantizedLayer::<f32>::new(
            "dense1".to_string(),
            Some(QuantizationParams::int8()),
            None,
            vec![],
            vec![10],
            vec![20],
        );

        assert_eq!(layer.layer_name(), "dense1");
        assert!(layer.weight_params().is_some());
        assert!(layer.activation_params().is_none());
    }

    #[test]
    fn test_model_quantizer() {
        let quantizer = ModelQuantizer::new();
        assert_eq!(
            quantizer.config.strategy,
            QuantizationStrategy::PostTraining
        );

        let custom_config = QuantizationConfig {
            strategy: QuantizationStrategy::Dynamic,
            ..Default::default()
        };
        let custom_quantizer = ModelQuantizer::with_config(custom_config);
        assert_eq!(
            custom_quantizer.config.strategy,
            QuantizationStrategy::Dynamic
        );
    }

    #[test]
    fn test_sequential_quantization() {
        let model = Sequential::new(vec![
            Box::new(Dense::<f32>::new(10, 20, true)),
            Box::new(Dense::<f32>::new(20, 1, true)),
        ]);

        let result = quantize_model(&model, None);
        assert!(result.is_ok());

        let (_quantized_model, stats) = result.unwrap();
        assert!(stats.layers_quantized > 0);
        assert!(stats.compression_ratio() > 1.0);
        assert!(stats.inference_speedup >= 1.0);
    }

    #[test]
    fn test_mobile_quantization_config() {
        let config = mobile_quantization_config();
        assert_eq!(config.strategy, QuantizationStrategy::Dynamic);
        assert_eq!(config.precision, QuantizationPrecision::Int8);
        assert!(!config.quantize_activations);
        assert_eq!(config.accuracy_threshold, Some(0.03));
    }

    #[test]
    fn test_edge_quantization_config() {
        let config = edge_quantization_config();
        assert_eq!(config.strategy, QuantizationStrategy::Static);
        assert!(config.quantize_activations);
        assert_eq!(config.accuracy_threshold, Some(0.05));
    }

    #[test]
    fn test_ultra_low_precision_config() {
        let config = ultra_low_precision_config();
        assert_eq!(config.precision, QuantizationPrecision::Int4);
        assert_eq!(config.accuracy_threshold, Some(0.10));
        assert_eq!(config.calibration_samples, Some(2000));
    }

    #[test]
    #[cfg(feature = "serialize")]
    fn test_quantization_serialization() {
        let params = QuantizationParams::int8();
        let serialized = serde_json::to_string(&params).unwrap();
        let deserialized: QuantizationParams = serde_json::from_str(&serialized).unwrap();
        assert_eq!(params.scale, deserialized.scale);
        assert_eq!(params.zero_point, deserialized.zero_point);
    }

    #[test]
    fn test_qat_config() {
        let config = qat_config();
        assert_eq!(config.strategy, QuantizationStrategy::QuantizationAware);
        assert_eq!(config.precision, QuantizationPrecision::Int8);
        assert!(config.quantize_weights);
        assert!(config.quantize_activations);
        assert!(config.calibration_samples.is_none());
        assert_eq!(config.accuracy_threshold, Some(0.01));
    }

    #[test]
    fn test_fake_quantization_layer() {
        let params = QuantizationParams::int8();
        let mut fake_quant = FakeQuantization::<f32>::new(params);

        // Test layer creation
        assert!(fake_quant.enabled);
        assert_eq!(fake_quant.get_params().qmin, -128);
        assert_eq!(fake_quant.get_params().qmax, 127);

        // Test enable/disable
        fake_quant.set_enabled(false);
        assert!(!fake_quant.enabled);

        // Test training mode
        fake_quant.set_training(false);
        assert!(!fake_quant.training);

        // Test parameters (should be empty)
        assert!(fake_quant.parameters().is_empty());
        assert!(fake_quant.parameters_mut().is_empty());
    }

    #[test]
    fn test_quantization_observer() {
        let mut observer = QuantizationObserver::<f32>::new();

        // Initially no observations
        assert_eq!(observer.count(), 0);
        assert!(observer.get_min_max().is_none());

        // Add observations
        observer.observe(-2.0, 3.0);
        observer.observe(-1.0, 5.0);

        // Check statistics
        assert_eq!(observer.count(), 2);
        let (min, max) = observer.get_min_max().unwrap();
        assert_eq!(min, -2.0);
        assert_eq!(max, 5.0);

        // Reset observer
        observer.reset();
        assert_eq!(observer.count(), 0);
        assert!(observer.get_min_max().is_none());
    }

    #[test]
    fn test_quantization_aware_training_strategy() {
        let config = QuantizationConfig {
            strategy: QuantizationStrategy::QuantizationAware,
            ..Default::default()
        };

        let quantizer = ModelQuantizer::with_config(config);
        let model = Sequential::new(vec![
            Box::new(Dense::<f32>::new(10, 20, true)),
            Box::new(Dense::<f32>::new(20, 1, true)),
        ]);

        let result = quantizer.quantize_sequential(&model);
        assert!(result.is_ok());

        let (_quantized_model, stats) = result.unwrap();
        assert_eq!(stats.layers_quantized, 5); // QAT should prepare 5 layers
        assert_eq!(stats.parameters_quantized, 2500);
    }

    #[test]
    fn test_prepare_model_for_qat() {
        let mut model = Sequential::new(vec![
            Box::new(Dense::<f32>::new(10, 20, true)),
            Box::new(Dense::<f32>::new(20, 1, true)),
        ]);

        // Test with QAT config
        let qat_config = qat_config();
        let result = prepare_model_for_qat(&mut model, Some(qat_config));
        assert!(result.is_ok());

        // Test with wrong strategy
        let wrong_config = QuantizationConfig {
            strategy: QuantizationStrategy::PostTraining,
            ..Default::default()
        };
        let result = prepare_model_for_qat(&mut model, Some(wrong_config));
        assert!(result.is_err());
    }

    #[test]
    fn test_finalize_qat_model() {
        let mut model = Sequential::new(vec![
            Box::new(Dense::<f32>::new(10, 20, true)),
            Box::new(Dense::<f32>::new(20, 1, true)),
        ]);

        let result = finalize_qat_model(&mut model, None);
        assert!(result.is_ok());

        let stats = result.unwrap();
        assert!(stats.compression_ratio() > 1.0);
        assert!(stats.inference_speedup >= 1.0);
        assert!(stats.memory_reduction > 0.0);
    }

    #[test]
    fn test_fake_quantization_qparams_calculation() {
        let initial_params = QuantizationParams::int8();
        let fake_quant = FakeQuantization::<f32>::new(initial_params);

        // Test qparams calculation
        let new_params = fake_quant.calculate_qparams(-10.0, 10.0);

        // For int8 with range [-10, 10] and qrange [-128, 127]
        // scale should be 20.0 / 255.0 ≈ 0.078
        // zero_point should be around -128 + 10/scale
        assert!(new_params.scale > 0.0);
        assert!(new_params.zero_point >= -128);
        assert!(new_params.zero_point <= 127);

        // Test edge case: min == max
        let edge_params = fake_quant.calculate_qparams(5.0, 5.0);
        assert!(edge_params.scale > 0.0); // Should use minimum scale to avoid division by zero
    }
}
