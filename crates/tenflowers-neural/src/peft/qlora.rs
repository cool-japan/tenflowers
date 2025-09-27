//! QLoRA (Quantized LoRA) implementation for memory-efficient fine-tuning
//!
//! QLoRA combines LoRA with 4-bit quantization to significantly reduce memory usage
//! while maintaining competitive performance. The base model weights are quantized
//! to 4-bit precision while LoRA adapters remain in full precision.
//!
//! Reference: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)

use super::{lora::LoRAConfig, PEFTAdapter, PEFTMethod};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use scirs2_core::random::distributions::{Distribution, Normal};
use scirs2_core::random::rng;
use std::marker::PhantomData;
use tenflowers_core::{Result, Tensor, TensorError};

/// Configuration for QLoRA adapters
#[derive(Debug, Clone)]
pub struct QLoRAConfig {
    /// Base LoRA configuration
    pub lora_config: LoRAConfig,
    /// Number of bits for quantization (typically 4)
    pub bits: u8,
    /// Whether to use double quantization for constants
    pub use_double_quant: bool,
    /// Quantization data type (NF4, INT4, etc.)
    pub quant_type: QuantizationType,
    /// Block size for quantization
    pub block_size: usize,
    /// Whether to use gradient checkpointing
    pub use_gradient_checkpointing: bool,
}

/// Supported quantization types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizationType {
    /// Normal Float 4-bit (NF4) - optimized for normally distributed weights
    NF4,
    /// Standard 4-bit integer quantization
    INT4,
    /// 8-bit integer fallback
    INT8,
}

impl QLoRAConfig {
    /// Create a new QLoRA configuration
    pub fn new(lora_config: LoRAConfig) -> Self {
        Self {
            lora_config,
            bits: 4,
            use_double_quant: true,
            quant_type: QuantizationType::NF4,
            block_size: 64,
            use_gradient_checkpointing: false,
        }
    }

    /// Set quantization bits
    pub fn with_bits(mut self, bits: u8) -> Self {
        self.bits = bits;
        self
    }

    /// Enable double quantization
    pub fn with_double_quant(mut self) -> Self {
        self.use_double_quant = true;
        self
    }

    /// Set quantization type
    pub fn with_quant_type(mut self, quant_type: QuantizationType) -> Self {
        self.quant_type = quant_type;
        self
    }

    /// Set block size for quantization
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Enable gradient checkpointing
    pub fn with_gradient_checkpointing(mut self) -> Self {
        self.use_gradient_checkpointing = true;
        self
    }
}

impl Default for QLoRAConfig {
    fn default() -> Self {
        Self::new(LoRAConfig::default())
    }
}

/// Quantized weight storage for memory efficiency
#[derive(Clone)]
struct QuantizedWeight<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Quantized weight data (packed)
    quantized_data: Vec<u8>,
    /// Scaling factors for dequantization
    scales: Tensor<T>,
    /// Zero points for asymmetric quantization
    zero_points: Option<Tensor<T>>,
    /// Original weight shape
    original_shape: Vec<usize>,
    /// Quantization configuration
    config: QLoRAConfig,
}

impl<T> QuantizedWeight<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Quantize a weight tensor
    fn quantize(weight: &Tensor<T>, config: &QLoRAConfig) -> Result<Self> {
        let shape = weight.shape().dims().to_vec();
        let data = weight.to_vec()?;

        match config.quant_type {
            QuantizationType::NF4 => Self::quantize_nf4(data, &shape, config),
            QuantizationType::INT4 => Self::quantize_int4(data, &shape, config),
            QuantizationType::INT8 => Self::quantize_int8(data, &shape, config),
        }
    }

    /// NF4 quantization optimized for normally distributed weights
    fn quantize_nf4(data: Vec<T>, shape: &[usize], config: &QLoRAConfig) -> Result<Self> {
        // NF4 quantization levels optimized for normal distribution
        #[allow(clippy::excessive_precision)]
        const NF4_LEVELS: [f32; 16] = [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ];

        let total_elements = data.len();
        let block_size = config.block_size.min(total_elements);
        let num_blocks = (total_elements + block_size - 1) / block_size;

        let mut quantized_data = Vec::with_capacity((total_elements + 1) / 2); // 4-bit packing
        let mut scales = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(total_elements);
            let block = &data[start..end];

            // Compute scale factor (max absolute value in block)
            let max_val = block
                .iter()
                .map(|x| x.to_f32().unwrap_or(0.0).abs())
                .fold(0.0f32, f32::max);

            let scale = if max_val > 0.0 { max_val } else { 1.0 };
            scales.push(T::from_f32(scale).unwrap_or_else(|| T::one()));

            // Quantize block to NF4 levels
            for chunk in block.chunks(2) {
                let mut packed_byte = 0u8;

                for (i, &value) in chunk.iter().enumerate() {
                    let normalized = value.to_f32().unwrap_or(0.0) / scale;

                    // Find closest NF4 level
                    let mut best_idx = 0;
                    let mut best_dist = (normalized - NF4_LEVELS[0]).abs();

                    for (idx, &level) in NF4_LEVELS.iter().enumerate().skip(1) {
                        let dist = (normalized - level).abs();
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx = idx;
                        }
                    }

                    if i == 0 {
                        packed_byte |= (best_idx as u8) & 0x0F;
                    } else {
                        packed_byte |= ((best_idx as u8) & 0x0F) << 4;
                    }
                }

                quantized_data.push(packed_byte);
            }
        }

        let scales_tensor = Tensor::from_vec(scales, &[num_blocks])?;

        Ok(Self {
            quantized_data,
            scales: scales_tensor,
            zero_points: None,
            original_shape: shape.to_vec(),
            config: config.clone(),
        })
    }

    /// Standard INT4 quantization
    fn quantize_int4(data: Vec<T>, shape: &[usize], config: &QLoRAConfig) -> Result<Self> {
        let total_elements = data.len();
        let block_size = config.block_size.min(total_elements);
        let num_blocks = (total_elements + block_size - 1) / block_size;

        let mut quantized_data = Vec::with_capacity((total_elements + 1) / 2);
        let mut scales = Vec::with_capacity(num_blocks);
        let mut zero_points = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(total_elements);
            let block = &data[start..end];

            // Compute min/max for symmetric quantization
            let min_val = block
                .iter()
                .map(|x| x.to_f32().unwrap_or(0.0))
                .fold(f32::INFINITY, f32::min);
            let max_val = block
                .iter()
                .map(|x| x.to_f32().unwrap_or(0.0))
                .fold(f32::NEG_INFINITY, f32::max);

            let scale = (max_val - min_val) / 15.0; // 4-bit: 0-15 range
            let zero_point = -min_val / scale;

            scales.push(T::from_f32(scale).unwrap_or_else(|| T::one()));
            zero_points.push(T::from_f32(zero_point).unwrap_or_else(|| T::zero()));

            // Quantize and pack
            for chunk in block.chunks(2) {
                let mut packed_byte = 0u8;

                for (i, &value) in chunk.iter().enumerate() {
                    let normalized = value.to_f32().unwrap_or(0.0);
                    let quantized = ((normalized / scale + zero_point).round() as i8).clamp(0, 15);

                    if i == 0 {
                        packed_byte |= (quantized as u8) & 0x0F;
                    } else {
                        packed_byte |= ((quantized as u8) & 0x0F) << 4;
                    }
                }

                quantized_data.push(packed_byte);
            }
        }

        let scales_tensor = Tensor::from_vec(scales, &[num_blocks])?;
        let zero_points_tensor = Tensor::from_vec(zero_points, &[num_blocks])?;

        Ok(Self {
            quantized_data,
            scales: scales_tensor,
            zero_points: Some(zero_points_tensor),
            original_shape: shape.to_vec(),
            config: config.clone(),
        })
    }

    /// INT8 quantization fallback
    fn quantize_int8(data: Vec<T>, shape: &[usize], config: &QLoRAConfig) -> Result<Self> {
        // Simplified INT8 quantization
        let min_val = data
            .iter()
            .map(|x| x.to_f32().unwrap_or(0.0))
            .fold(f32::INFINITY, f32::min);
        let max_val = data
            .iter()
            .map(|x| x.to_f32().unwrap_or(0.0))
            .fold(f32::NEG_INFINITY, f32::max);

        let scale = (max_val - min_val) / 255.0;
        let zero_point = -min_val / scale;

        let quantized_data: Vec<u8> = data
            .iter()
            .map(|&x| {
                let normalized = x.to_f32().unwrap_or(0.0);
                ((normalized / scale + zero_point).round() as i16).clamp(0, 255) as u8
            })
            .collect();

        let scales_tensor =
            Tensor::from_vec(vec![T::from_f32(scale).unwrap_or_else(|| T::one())], &[1])?;
        let zero_points_tensor = Tensor::from_vec(
            vec![T::from_f32(zero_point).unwrap_or_else(|| T::zero())],
            &[1],
        )?;

        Ok(Self {
            quantized_data,
            scales: scales_tensor,
            zero_points: Some(zero_points_tensor),
            original_shape: shape.to_vec(),
            config: config.clone(),
        })
    }

    /// Dequantize the weight tensor for computation
    fn dequantize(&self) -> Result<Tensor<T>> {
        match self.config.quant_type {
            QuantizationType::NF4 => self.dequantize_nf4(),
            QuantizationType::INT4 => self.dequantize_int4(),
            QuantizationType::INT8 => self.dequantize_int8(),
        }
    }

    /// Dequantize NF4 data
    fn dequantize_nf4(&self) -> Result<Tensor<T>> {
        #[allow(clippy::excessive_precision)]
        const NF4_LEVELS: [f32; 16] = [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124625,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ];

        let total_elements = self.original_shape.iter().product::<usize>();
        let block_size = self.config.block_size.min(total_elements);
        let scales_data = self.scales.to_vec()?;

        let mut dequantized = Vec::with_capacity(total_elements);
        let mut data_idx = 0;
        let mut element_idx = 0;

        for block_idx in 0..scales_data.len() {
            let scale = scales_data[block_idx].to_f32().unwrap_or(1.0);
            let elements_in_block = block_size.min(total_elements - element_idx);

            for _ in 0..(elements_in_block + 1) / 2 {
                if data_idx >= self.quantized_data.len() {
                    break;
                }

                let packed_byte = self.quantized_data[data_idx];
                data_idx += 1;

                // Extract two 4-bit values
                let idx1 = (packed_byte & 0x0F) as usize;
                let idx2 = ((packed_byte >> 4) & 0x0F) as usize;

                if element_idx < total_elements {
                    let value1 = NF4_LEVELS[idx1] * scale;
                    dequantized.push(T::from_f32(value1).unwrap_or_else(|| T::zero()));
                    element_idx += 1;
                }

                if element_idx < total_elements {
                    let value2 = NF4_LEVELS[idx2] * scale;
                    dequantized.push(T::from_f32(value2).unwrap_or_else(|| T::zero()));
                    element_idx += 1;
                }
            }
        }

        // Pad if necessary
        while dequantized.len() < total_elements {
            dequantized.push(T::zero());
        }

        Tensor::from_vec(dequantized, &self.original_shape)
    }

    /// Dequantize INT4 data
    fn dequantize_int4(&self) -> Result<Tensor<T>> {
        let total_elements = self.original_shape.iter().product::<usize>();
        let block_size = self.config.block_size.min(total_elements);
        let scales_data = self.scales.to_vec()?;
        let zero_points_data = self
            .zero_points
            .as_ref()
            .ok_or_else(|| {
                TensorError::invalid_argument("Missing zero points for INT4".to_string())
            })?
            .to_vec()?;

        let mut dequantized = Vec::with_capacity(total_elements);
        let mut data_idx = 0;
        let mut element_idx = 0;

        for block_idx in 0..scales_data.len() {
            let scale = scales_data[block_idx].to_f32().unwrap_or(1.0);
            let zero_point = zero_points_data[block_idx].to_f32().unwrap_or(0.0);
            let elements_in_block = block_size.min(total_elements - element_idx);

            for _ in 0..(elements_in_block + 1) / 2 {
                if data_idx >= self.quantized_data.len() {
                    break;
                }

                let packed_byte = self.quantized_data[data_idx];
                data_idx += 1;

                let q1 = (packed_byte & 0x0F) as f32;
                let q2 = ((packed_byte >> 4) & 0x0F) as f32;

                if element_idx < total_elements {
                    let value1 = (q1 - zero_point) * scale;
                    dequantized.push(T::from_f32(value1).unwrap_or_else(|| T::zero()));
                    element_idx += 1;
                }

                if element_idx < total_elements {
                    let value2 = (q2 - zero_point) * scale;
                    dequantized.push(T::from_f32(value2).unwrap_or_else(|| T::zero()));
                    element_idx += 1;
                }
            }
        }

        while dequantized.len() < total_elements {
            dequantized.push(T::zero());
        }

        Tensor::from_vec(dequantized, &self.original_shape)
    }

    /// Dequantize INT8 data
    fn dequantize_int8(&self) -> Result<Tensor<T>> {
        let scales_data = self.scales.to_vec()?;
        let scale = scales_data[0].to_f32().unwrap_or(1.0);
        let zero_point = self
            .zero_points
            .as_ref()
            .and_then(|zp| zp.to_vec().ok().and_then(|data| data.first().copied()))
            .map(|x| x.to_f32().unwrap_or(0.0))
            .unwrap_or(0.0);

        let dequantized: Vec<T> = self
            .quantized_data
            .iter()
            .map(|&q| {
                let value = (q as f32 - zero_point) * scale;
                T::from_f32(value).unwrap_or_else(|| T::zero())
            })
            .collect();

        Tensor::from_vec(dequantized, &self.original_shape)
    }
}

/// QLoRA adapter that combines quantization with LoRA
#[derive(Clone)]
pub struct QLoRAAdapter<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Quantized base weight (optional - for when base layer weights are quantized)
    quantized_base_weight: Option<QuantizedWeight<T>>,
    /// LoRA A matrix (kept in full precision)
    lora_a: Tensor<T>,
    /// LoRA B matrix (kept in full precision)
    lora_b: Tensor<T>,
    /// Optional bias for LoRA path
    bias: Option<Tensor<T>>,
    /// Configuration
    config: QLoRAConfig,
    /// Training mode
    training: bool,
    /// Phantom data
    _phantom: PhantomData<T>,
}

impl<T> QLoRAAdapter<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new QLoRA adapter
    pub fn new(input_dim: usize, output_dim: usize, config: QLoRAConfig) -> Result<Self> {
        let rank = config.lora_config.rank;

        // Initialize LoRA matrices (kept in full precision)
        let lora_a = Self::create_random_matrix(&[input_dim, rank], rank)?;
        let lora_b = Tensor::zeros(&[rank, output_dim]);

        let bias = if config.lora_config.use_bias {
            Some(Tensor::zeros(&[output_dim]))
        } else {
            None
        };

        Ok(Self {
            quantized_base_weight: None,
            lora_a,
            lora_b,
            bias,
            config,
            training: false,
            _phantom: PhantomData,
        })
    }

    /// Create QLoRA adapter from existing weight (quantizing it)
    pub fn from_weight(weight: &Tensor<T>, config: QLoRAConfig) -> Result<Self> {
        let weight_shape = weight.shape().dims();
        let input_dim = weight_shape[0];
        let output_dim = weight_shape[1];
        let rank = config.lora_config.rank;

        // Quantize the base weight
        let quantized_weight = QuantizedWeight::quantize(weight, &config)?;

        // Initialize LoRA matrices
        let lora_a = Self::create_random_matrix(&[input_dim, rank], rank)?;
        let lora_b = Tensor::zeros(&[rank, output_dim]);

        let bias = if config.lora_config.use_bias {
            Some(Tensor::zeros(&[output_dim]))
        } else {
            None
        };

        Ok(Self {
            quantized_base_weight: Some(quantized_weight),
            lora_a,
            lora_b,
            bias,
            config,
            training: false,
            _phantom: PhantomData,
        })
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> QLoRAMemoryStats {
        let lora_a_bytes =
            self.lora_a.shape().dims().iter().product::<usize>() * std::mem::size_of::<T>();
        let lora_b_bytes =
            self.lora_b.shape().dims().iter().product::<usize>() * std::mem::size_of::<T>();
        let bias_bytes = self
            .bias
            .as_ref()
            .map(|b| b.shape().dims().iter().product::<usize>() * std::mem::size_of::<T>())
            .unwrap_or(0);

        let quantized_bytes = self
            .quantized_base_weight
            .as_ref()
            .map(|qw| {
                qw.quantized_data.len()
                    + qw.scales.shape().dims().iter().product::<usize>() * std::mem::size_of::<T>()
                    + qw.zero_points
                        .as_ref()
                        .map(|zp| {
                            zp.shape().dims().iter().product::<usize>() * std::mem::size_of::<T>()
                        })
                        .unwrap_or(0)
            })
            .unwrap_or(0);

        let original_weight_bytes = self
            .quantized_base_weight
            .as_ref()
            .map(|qw| qw.original_shape.iter().product::<usize>() * std::mem::size_of::<T>())
            .unwrap_or(0);

        QLoRAMemoryStats {
            lora_parameters_bytes: lora_a_bytes + lora_b_bytes + bias_bytes,
            quantized_base_bytes: quantized_bytes,
            original_base_bytes: original_weight_bytes,
            total_bytes: lora_a_bytes + lora_b_bytes + bias_bytes + quantized_bytes,
            memory_reduction_ratio: if original_weight_bytes > 0 {
                1.0 - (quantized_bytes as f64 / original_weight_bytes as f64)
            } else {
                0.0
            },
        }
    }

    /// Helper to create random LoRA matrix
    fn create_random_matrix(shape: &[usize], rank: usize) -> Result<Tensor<T>> {
        let total_elements = shape.iter().product::<usize>();
        let std_dev = 1.0 / (rank as f64).sqrt();

        let mut rng = rng();
        let normal = Normal::new(0.0, std_dev).map_err(|_| {
            TensorError::invalid_argument("Invalid normal distribution parameters".to_string())
        })?;

        let mut data = Vec::with_capacity(total_elements);
        for _ in 0..total_elements {
            let random_val = normal.sample(&mut rng);
            let tensor_val = T::from_f64(random_val).unwrap_or_else(|| T::zero());
            data.push(tensor_val);
        }

        Tensor::from_vec(data, shape)
    }
}

impl<T> PEFTAdapter<T> for QLoRAAdapter<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>, base_output: &Tensor<T>) -> Result<Tensor<T>> {
        // Compute LoRA adaptation: input @ A @ B
        let temp = tenflowers_core::ops::matmul(input, &self.lora_a)?;
        let lora_output = tenflowers_core::ops::matmul(&temp, &self.lora_b)?;

        // Apply scaling factor
        let scaling = T::from(self.config.lora_config.scaling_factor()).unwrap_or_else(|| T::one());
        let scaled_output = lora_output.scalar_mul(scaling)?;

        // Add bias if present
        let final_lora = if let Some(ref bias) = self.bias {
            scaled_output.add(bias)?
        } else {
            scaled_output
        };

        // Add to base output (which may come from quantized weights)
        base_output.add(&final_lora)
    }

    fn trainable_parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![&self.lora_a, &self.lora_b];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn trainable_parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = vec![&mut self.lora_a, &mut self.lora_b];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn num_trainable_parameters(&self) -> usize {
        let lora_a_params = self.lora_a.shape().dims().iter().product::<usize>();
        let lora_b_params = self.lora_b.shape().dims().iter().product::<usize>();
        let bias_params = self
            .bias
            .as_ref()
            .map(|b| b.shape().dims().iter().product::<usize>())
            .unwrap_or(0);

        lora_a_params + lora_b_params + bias_params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn method_type(&self) -> PEFTMethod {
        PEFTMethod::QLoRA
    }
}

/// Memory usage statistics for QLoRA
#[derive(Debug, Clone)]
pub struct QLoRAMemoryStats {
    pub lora_parameters_bytes: usize,
    pub quantized_base_bytes: usize,
    pub original_base_bytes: usize,
    pub total_bytes: usize,
    pub memory_reduction_ratio: f64,
}

impl QLoRAMemoryStats {
    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "QLoRA Memory: {:.1}MB total ({:.1}MB LoRA + {:.1}MB quantized base), {:.1}% memory reduction",
            self.total_bytes as f64 / (1024.0 * 1024.0),
            self.lora_parameters_bytes as f64 / (1024.0 * 1024.0),
            self.quantized_base_bytes as f64 / (1024.0 * 1024.0),
            self.memory_reduction_ratio * 100.0
        )
    }
}

/// Predefined QLoRA configurations
impl QLoRAConfig {
    /// Configuration for large language models with aggressive quantization
    pub fn for_llm() -> Self {
        Self::new(LoRAConfig::for_llm())
            .with_bits(4)
            .with_quant_type(QuantizationType::NF4)
            .with_double_quant()
            .with_gradient_checkpointing()
    }

    /// Configuration for vision models with conservative quantization
    pub fn for_vision() -> Self {
        Self::new(LoRAConfig::for_vision())
            .with_bits(8)
            .with_quant_type(QuantizationType::INT8)
            .with_block_size(128)
    }

    /// Configuration for maximum memory efficiency
    pub fn for_efficiency() -> Self {
        Self::new(LoRAConfig::for_efficiency())
            .with_bits(4)
            .with_quant_type(QuantizationType::NF4)
            .with_double_quant()
            .with_block_size(32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Dense, Layer};

    #[test]
    fn test_qlora_config_creation() {
        let config = QLoRAConfig::for_llm();
        assert_eq!(config.bits, 4);
        assert_eq!(config.quant_type, QuantizationType::NF4);
        assert!(config.use_double_quant);
        assert!(config.use_gradient_checkpointing);
    }

    #[test]
    fn test_quantization_types() {
        let nf4_config = QLoRAConfig::default().with_quant_type(QuantizationType::NF4);
        let int4_config = QLoRAConfig::default().with_quant_type(QuantizationType::INT4);
        let int8_config = QLoRAConfig::default().with_quant_type(QuantizationType::INT8);

        assert_eq!(nf4_config.quant_type, QuantizationType::NF4);
        assert_eq!(int4_config.quant_type, QuantizationType::INT4);
        assert_eq!(int8_config.quant_type, QuantizationType::INT8);
    }

    #[test]
    fn test_qlora_adapter_creation() {
        let config = QLoRAConfig::for_efficiency();
        let adapter: QLoRAAdapter<f32> = QLoRAAdapter::new(100, 50, config).unwrap();

        // Check LoRA matrix shapes
        assert_eq!(adapter.lora_a.shape().dims(), &[100, 4]); // rank=4 from efficiency config
        assert_eq!(adapter.lora_b.shape().dims(), &[4, 50]);

        // Check parameter count (only LoRA parameters are trainable)
        assert_eq!(adapter.num_trainable_parameters(), 100 * 4 + 4 * 50);
    }

    #[test]
    fn test_weight_quantization() {
        let weight: Tensor<f32> = Tensor::ones(&[10, 8]); // Small test weight
        let config = QLoRAConfig::for_efficiency();

        let quantized = QuantizedWeight::quantize(&weight, &config).unwrap();
        let dequantized = quantized.dequantize().unwrap();

        // Check shape preservation
        assert_eq!(dequantized.shape().dims(), &[10, 8]);

        // Check that dequantized values are close to original (allowing for quantization error)
        let original_data = weight.to_vec().unwrap();
        let deq_data = dequantized.to_vec().unwrap();

        for (orig, deq) in original_data.iter().zip(deq_data.iter()) {
            let diff = (orig.to_f32().unwrap() - deq.to_f32().unwrap()).abs();
            assert!(
                diff < 0.2,
                "Quantization error too large: {} vs {}",
                orig.to_f32().unwrap(),
                deq.to_f32().unwrap()
            );
        }
    }

    #[test]
    fn test_memory_efficiency() {
        let weight: Tensor<f32> = Tensor::ones(&[1000, 1000]); // Large weight matrix
        let config = QLoRAConfig::for_efficiency();

        let adapter = QLoRAAdapter::from_weight(&weight, config).unwrap();
        let stats = adapter.memory_stats();

        // Should achieve significant memory reduction
        assert!(
            stats.memory_reduction_ratio > 0.5,
            "Expected >50% memory reduction, got {:.1}%",
            stats.memory_reduction_ratio * 100.0
        );
        assert!(stats.quantized_base_bytes < stats.original_base_bytes);

        println!("{}", stats.summary());
    }

    #[test]
    fn test_qlora_forward_pass() {
        let config = QLoRAConfig::for_efficiency();
        let adapter: QLoRAAdapter<f32> = QLoRAAdapter::new(10, 5, config).unwrap();

        let input = Tensor::ones(&[2, 10]);
        let base_output = Tensor::zeros(&[2, 5]);

        let result = adapter.forward(&input, &base_output);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape().dims(), &[2, 5]);
    }
}
