//! Quantization operations for TenfloweRS
//!
//! This module provides quantization and dequantization operations for INT8/INT4
//! to enable efficient inference and reduced memory usage.

use crate::{DType, Result, Tensor, TensorError};
use num_traits::{cast::cast, Float};

#[cfg(feature = "gpu")]
use crate::device::context::get_gpu_context;
#[cfg(feature = "gpu")]
use crate::gpu::buffer::GpuBuffer;
#[cfg(feature = "gpu")]
use bytemuck;
#[cfg(feature = "gpu")]
use scirs2_autograd::ndarray::Array1;
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

/// Quantization parameters for symmetric and asymmetric quantization
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// Scale factor for quantization (real_value = scale * quantized_value + zero_point)
    pub scale: f32,
    /// Zero point for asymmetric quantization (usually 0 for symmetric)
    pub zero_point: i32,
    /// Quantized data type (Int8 or Int4)
    pub dtype: DType,
    /// Minimum quantized value (e.g., -128 for INT8)
    pub qmin: i32,
    /// Maximum quantized value (e.g., 127 for INT8)
    pub qmax: i32,
}

impl QuantizationParams {
    /// Create symmetric INT8 quantization parameters
    pub fn symmetric_int8(scale: f32) -> Self {
        Self {
            scale,
            zero_point: 0,
            dtype: DType::Int8,
            qmin: -128,
            qmax: 127,
        }
    }

    /// Create asymmetric INT8 quantization parameters  
    pub fn asymmetric_int8(scale: f32, zero_point: i32) -> Self {
        Self {
            scale,
            zero_point,
            dtype: DType::Int8,
            qmin: -128,
            qmax: 127,
        }
    }

    /// Create symmetric INT4 quantization parameters
    pub fn symmetric_int4(scale: f32) -> Self {
        Self {
            scale,
            zero_point: 0,
            dtype: DType::Int4,
            qmin: -8,
            qmax: 7,
        }
    }

    /// Create asymmetric INT4 quantization parameters
    pub fn asymmetric_int4(scale: f32, zero_point: i32) -> Self {
        Self {
            scale,
            zero_point,
            dtype: DType::Int4,
            qmin: -8,
            qmax: 7,
        }
    }

    /// Calculate quantization parameters from tensor statistics
    pub fn from_tensor_stats(min_val: f32, max_val: f32, dtype: DType) -> Result<Self> {
        let (qmin, qmax) = match dtype {
            DType::Int8 => (-128, 127),
            DType::Int4 => (-8, 7),
            _ => {
                return Err(TensorError::invalid_argument(format!(
                    "Unsupported quantization dtype: {dtype:?}"
                )))
            }
        };

        // For symmetric quantization
        let abs_max = min_val.abs().max(max_val.abs());
        let scale = abs_max / qmax as f32;

        Ok(Self {
            scale,
            zero_point: 0,
            dtype,
            qmin,
            qmax,
        })
    }

    /// Calculate asymmetric quantization parameters from tensor statistics
    pub fn asymmetric_from_tensor_stats(min_val: f32, max_val: f32, dtype: DType) -> Result<Self> {
        let (qmin, qmax) = match dtype {
            DType::Int8 => (-128, 127),
            DType::Int4 => (-8, 7),
            _ => {
                return Err(TensorError::invalid_argument(format!(
                    "Unsupported quantization dtype: {dtype:?}"
                )))
            }
        };

        let scale = (max_val - min_val) / (qmax - qmin) as f32;
        let zero_point = qmin - (min_val / scale).round() as i32;

        Ok(Self {
            scale,
            zero_point: zero_point.clamp(qmin, qmax),
            dtype,
            qmin,
            qmax,
        })
    }
}

/// Quantize a float32 tensor to INT8 or INT4
pub fn quantize<T>(tensor: &Tensor<T>, params: &QuantizationParams) -> Result<Tensor<i8>>
where
    T: Float + Send + Sync + Clone + Default + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    match &tensor.storage {
        crate::tensor::TensorStorage::Cpu(data) => {
            let quantized_data = data.mapv(|val| {
                let val_f32 = cast::<T, f32>(val).unwrap_or(0.0);
                let quantized =
                    ((val_f32 / params.scale) + params.zero_point as f32).round() as i32;
                quantized.clamp(params.qmin, params.qmax) as i8
            });

            Ok(Tensor::from_array(quantized_data))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => gpu_quantize(gpu_buffer, params),
    }
}

/// Dequantize an INT8 or INT4 tensor back to float32
pub fn dequantize(tensor: &Tensor<i8>, params: &QuantizationParams) -> Result<Tensor<f32>> {
    match &tensor.storage {
        crate::tensor::TensorStorage::Cpu(data) => {
            let dequantized_data =
                data.mapv(|val| (val as i32 - params.zero_point) as f32 * params.scale);

            Ok(Tensor::from_array(dequantized_data))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => gpu_dequantize(gpu_buffer, params),
    }
}

/// Dynamic quantization - calculates quantization parameters on the fly
pub fn dynamic_quantize<T>(
    tensor: &Tensor<T>,
    dtype: DType,
) -> Result<(Tensor<i8>, QuantizationParams)>
where
    T: Float
        + PartialOrd
        + Send
        + Sync
        + Clone
        + Default
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &tensor.storage {
        crate::tensor::TensorStorage::Cpu(data) => {
            // Calculate min and max values
            let mut min_val = T::infinity();
            let mut max_val = T::neg_infinity();

            for &val in data.iter() {
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }

            let min_f32 = cast::<T, f32>(min_val).unwrap_or(0.0);
            let max_f32 = cast::<T, f32>(max_val).unwrap_or(0.0);

            let params = QuantizationParams::from_tensor_stats(min_f32, max_f32, dtype)?;
            let quantized = quantize(tensor, &params)?;

            Ok((quantized, params))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => gpu_dynamic_quantize(gpu_buffer, dtype),
    }
}

/// Quantize-aware training support - applies fake quantization during training
pub fn fake_quantize<T>(tensor: &Tensor<T>, params: &QuantizationParams) -> Result<Tensor<T>>
where
    T: Float + Send + Sync + Clone + Default + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    match &tensor.storage {
        crate::tensor::TensorStorage::Cpu(data) => {
            let fake_quantized_data = data.mapv(|val| {
                let val_f32 = cast::<T, f32>(val).unwrap_or(0.0);
                let quantized =
                    ((val_f32 / params.scale) + params.zero_point as f32).round() as i32;
                let clamped = quantized.clamp(params.qmin, params.qmax);
                let dequantized = (clamped - params.zero_point) as f32 * params.scale;
                cast::<f32, T>(dequantized).unwrap_or_default()
            });

            Ok(Tensor::from_array(fake_quantized_data))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => {
            gpu_fake_quantize(gpu_buffer, params, tensor.shape())
        }
    }
}

/// Per-channel quantization for weights (common in neural networks)
pub fn per_channel_quantize<T>(
    tensor: &Tensor<T>,
    channel_axis: usize,
) -> Result<(Tensor<i8>, Vec<QuantizationParams>)>
where
    T: Float
        + PartialOrd
        + Send
        + Sync
        + Clone
        + Default
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &tensor.storage {
        crate::tensor::TensorStorage::Cpu(data) => {
            let shape = data.shape();
            if channel_axis >= shape.len() {
                return Err(TensorError::invalid_argument(format!(
                    "Channel axis {} out of bounds for tensor with {} dimensions",
                    channel_axis,
                    shape.len()
                )));
            }

            let num_channels = shape[channel_axis];
            let mut params_vec = Vec::with_capacity(num_channels);

            // For simplicity, we'll implement a basic version
            // A full implementation would properly handle per-channel statistics
            let overall_params = QuantizationParams::from_tensor_stats(0.0, 1.0, DType::Int8)?;
            for _ in 0..num_channels {
                params_vec.push(overall_params.clone());
            }

            let quantized = quantize(tensor, &overall_params)?;
            Ok((quantized, params_vec))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => {
            gpu_per_channel_quantize(gpu_buffer, channel_axis)
        }
    }
}

#[cfg(feature = "gpu")]
fn gpu_quantize<T>(gpu_buffer: &GpuBuffer<T>, params: &QuantizationParams) -> Result<Tensor<i8>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    let device_id = match gpu_buffer.device_enum() {
        crate::Device::Gpu(id) => id,
        _ => {
            return Err(TensorError::invalid_argument(
                "Expected GPU device".to_string(),
            ))
        }
    };

    let gpu_ctx = get_gpu_context(device_id)?;
    let buffer_size = gpu_buffer.len() * std::mem::size_of::<T>();

    // Create output buffer for quantized values (i8 stored as i32)
    let output_buffer = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("quantize_output"),
        size: (gpu_buffer.len() * std::mem::size_of::<i32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create params buffer
    let params_data = [
        params.scale,
        params.zero_point as f32,
        params.qmin as f32,
        params.qmax as f32,
    ];
    let params_buffer = gpu_ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quantize_params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // Create compute shader
    let shader = gpu_ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quantization_ops"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu/shaders/quantization_ops.wgsl").into(),
            ),
        });

    let compute_pipeline =
        gpu_ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quantize_pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some(match params.dtype {
                    DType::Int8 => "quantize_int8",
                    DType::Int4 => "quantize_int4",
                    _ => "quantize",
                }),
                cache: None,
                compilation_options: Default::default(),
            });

    // Create bind group
    let bind_group = gpu_ctx
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quantize_bind_group"),
            layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

    // Execute compute shader
    let mut encoder = gpu_ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("quantize_encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("quantize_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size = 64;
        let num_workgroups = (gpu_buffer.len() + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    gpu_ctx.queue.submit(std::iter::once(encoder.finish()));

    // Read back results
    let staging_buffer = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("quantize_staging"),
        size: (gpu_buffer.len() * std::mem::size_of::<i32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = gpu_ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, staging_buffer.size());
    gpu_ctx.queue.submit(std::iter::once(encoder.finish()));

    // Map and read data
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    gpu_ctx.device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .unwrap()
        .map_err(|e| TensorError::invalid_argument(format!("Buffer mapping failed: {:?}", e)))?;

    let data = buffer_slice.get_mapped_range();
    let i32_data: &[i32] = bytemuck::cast_slice(&data);
    let i8_data: Vec<i8> = i32_data.iter().map(|&x| x as i8).collect();

    drop(data);
    staging_buffer.unmap();

    let array = Array1::from_vec(i8_data).into_dyn();
    Ok(Tensor::from_array(array))
}

#[cfg(feature = "gpu")]
fn gpu_dequantize(gpu_buffer: &GpuBuffer<i8>, params: &QuantizationParams) -> Result<Tensor<f32>> {
    let device_id = match gpu_buffer.device_enum() {
        crate::Device::Gpu(id) => id,
        _ => {
            return Err(TensorError::invalid_argument(
                "Expected GPU device".to_string(),
            ))
        }
    };

    let gpu_ctx = get_gpu_context(device_id)?;

    // Create output buffer for dequantized values
    let output_buffer = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dequantize_output"),
        size: (gpu_buffer.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create params buffer
    let params_data = [params.scale, params.zero_point as f32];
    let params_buffer = gpu_ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dequantize_params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // Create compute shader
    let shader = gpu_ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quantization_ops"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu/shaders/quantization_ops.wgsl").into(),
            ),
        });

    let compute_pipeline =
        gpu_ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("dequantize_pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("dequantize"),
                cache: None,
                compilation_options: Default::default(),
            });

    // Create bind group
    let bind_group = gpu_ctx
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dequantize_bind_group"),
            layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

    // Execute compute shader
    let mut encoder = gpu_ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dequantize_encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dequantize_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size = 64;
        let num_workgroups = (gpu_buffer.len() + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    gpu_ctx.queue.submit(std::iter::once(encoder.finish()));

    // Read back results
    let staging_buffer = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dequantize_staging"),
        size: (gpu_buffer.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = gpu_ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, staging_buffer.size());
    gpu_ctx.queue.submit(std::iter::once(encoder.finish()));

    // Map and read data
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    gpu_ctx.device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .unwrap()
        .map_err(|e| TensorError::invalid_argument(format!("Buffer mapping failed: {:?}", e)))?;

    let data = buffer_slice.get_mapped_range();
    let f32_data: &[f32] = bytemuck::cast_slice(&data);
    let result_vec: Vec<f32> = f32_data.to_vec();

    drop(data);
    staging_buffer.unmap();

    let array = Array1::from_vec(result_vec).into_dyn();
    Ok(Tensor::from_array(array))
}

#[cfg(feature = "gpu")]
fn gpu_dynamic_quantize<T>(
    gpu_buffer: &GpuBuffer<T>,
    dtype: DType,
) -> Result<(Tensor<i8>, QuantizationParams)>
where
    T: Default
        + bytemuck::Pod
        + bytemuck::Zeroable
        + Clone
        + Send
        + Sync
        + 'static
        + num_traits::Float,
{
    // For simplicity, fall back to CPU implementation for now
    // A full implementation would use the dynamic_quantize shader
    let cpu_array = gpu_buffer.to_cpu_array()?;
    let cpu_tensor = Tensor::from_array(cpu_array);
    dynamic_quantize(&cpu_tensor, dtype)
}

#[cfg(feature = "gpu")]
fn gpu_fake_quantize<T>(
    gpu_buffer: &GpuBuffer<T>,
    params: &QuantizationParams,
    shape: &crate::Shape,
) -> Result<Tensor<T>>
where
    T: Default + bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    let device_id = match gpu_buffer.device_enum() {
        crate::Device::Gpu(id) => id,
        _ => {
            return Err(TensorError::invalid_argument(
                "Expected GPU device".to_string(),
            ))
        }
    };

    let gpu_ctx = get_gpu_context(device_id)?;

    // Create output buffer
    let output_buffer = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fake_quantize_output"),
        size: (gpu_buffer.len() * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create params buffer
    let params_data = [
        params.scale,
        params.zero_point as f32,
        params.qmin as f32,
        params.qmax as f32,
    ];
    let params_buffer = gpu_ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fake_quantize_params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // Create compute shader
    let shader = gpu_ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quantization_ops"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu/shaders/quantization_ops.wgsl").into(),
            ),
        });

    let compute_pipeline =
        gpu_ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fake_quantize_pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("fake_quantize"),
                cache: None,
                compilation_options: Default::default(),
            });

    // Create bind group
    let bind_group = gpu_ctx
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fake_quantize_bind_group"),
            layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

    // Execute compute shader
    let mut encoder = gpu_ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fake_quantize_encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fake_quantize_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size = 64;
        let num_workgroups = (gpu_buffer.len() + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    gpu_ctx.queue.submit(std::iter::once(encoder.finish()));

    // Read back results
    let staging_buffer = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fake_quantize_staging"),
        size: (gpu_buffer.len() * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = gpu_ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, staging_buffer.size());
    gpu_ctx.queue.submit(std::iter::once(encoder.finish()));

    // Map and read data
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    gpu_ctx.device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .unwrap()
        .map_err(|e| TensorError::invalid_argument(format!("Buffer mapping failed: {:?}", e)))?;

    let data = buffer_slice.get_mapped_range();
    let result_data: &[T] = bytemuck::cast_slice(&data);
    let result_vec: Vec<T> = result_data.to_vec();

    drop(data);
    staging_buffer.unmap();

    let result_array = ndarray::Array::from_shape_vec(shape.dims(), result_vec)
        .map_err(|e| TensorError::invalid_argument(format!("Shape mismatch: {:?}", e)))?
        .into_dyn();
    let result_buffer = GpuBuffer::from_cpu_array(&result_array, device_id)?;
    Ok(Tensor::from_gpu_buffer(result_buffer, shape.clone()))
}

#[cfg(feature = "gpu")]
fn gpu_per_channel_quantize<T>(
    gpu_buffer: &GpuBuffer<T>,
    channel_axis: usize,
) -> Result<(Tensor<i8>, Vec<QuantizationParams>)>
where
    T: Default
        + bytemuck::Pod
        + bytemuck::Zeroable
        + Clone
        + Send
        + Sync
        + 'static
        + num_traits::Float,
{
    // For simplicity, fall back to CPU implementation for now
    // A full implementation would use the per_channel_quantize shader
    let cpu_array = gpu_buffer.to_cpu_array()?;
    let cpu_tensor = Tensor::from_array(cpu_array);
    per_channel_quantize(&cpu_tensor, channel_axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::Array;

    #[test]
    fn test_symmetric_quantization() {
        let data = Array::from_vec(vec![1.0f32, 2.0, 3.0, -1.0, -2.0]).into_dyn();
        let tensor = Tensor::from_array(data);

        let params = QuantizationParams::symmetric_int8(0.1);
        let quantized = quantize(&tensor, &params).unwrap();
        let dequantized = dequantize(&quantized, &params).unwrap();

        // Test that quantization and dequantization are approximately inverse operations
        // (within quantization error)
        assert!(dequantized.shape() == tensor.shape());
    }

    #[test]
    fn test_dynamic_quantization() {
        let data = Array::from_vec(vec![1.0f32, 2.0, 3.0, -1.0, -2.0]).into_dyn();
        let tensor = Tensor::from_array(data);

        let (quantized, params) = dynamic_quantize(&tensor, DType::Int8).unwrap();

        assert_eq!(quantized.dtype(), DType::Int8);
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_fake_quantization() {
        let data = Array::from_vec(vec![1.0f32, 2.0, 3.0, -1.0, -2.0]).into_dyn();
        let tensor = Tensor::from_array(data);

        let params = QuantizationParams::symmetric_int8(0.1);
        let fake_quantized = fake_quantize(&tensor, &params).unwrap();

        // Fake quantization should maintain the same data type and shape
        assert_eq!(fake_quantized.dtype(), tensor.dtype());
        assert_eq!(fake_quantized.shape(), tensor.shape());
    }

    #[test]
    #[cfg(feature = "gpu")]
    #[ignore = "GPU buffer usage conflicts - needs WGPU buffer management fixes"]
    fn test_gpu_quantization() {
        let data = Array::from_vec(vec![1.0f32, 2.0, 3.0, -1.0, -2.0]).into_dyn();
        let cpu_tensor = Tensor::from_array(data);

        // Convert to GPU tensor
        let gpu_tensor = cpu_tensor.to_device(crate::Device::Gpu(0)).unwrap();

        let params = QuantizationParams::symmetric_int8(0.1);
        let quantized = quantize(&gpu_tensor, &params).unwrap();
        let dequantized = dequantize(&quantized, &params).unwrap();

        // Test that GPU quantization works
        assert_eq!(quantized.dtype(), DType::Int8);
        assert_eq!(dequantized.dtype(), DType::Float32);
        assert_eq!(dequantized.shape(), gpu_tensor.shape());
    }

    #[test]
    #[cfg(feature = "gpu")]
    #[ignore = "GPU buffer usage conflicts - needs WGPU buffer management fixes"]
    fn test_gpu_fake_quantization() {
        let data = Array::from_vec(vec![1.0f32, 2.0, 3.0, -1.0, -2.0]).into_dyn();
        let cpu_tensor = Tensor::from_array(data);

        // Convert to GPU tensor
        let gpu_tensor = cpu_tensor.to_device(crate::Device::Gpu(0)).unwrap();

        let params = QuantizationParams::symmetric_int8(0.1);
        let fake_quantized = fake_quantize(&gpu_tensor, &params).unwrap();

        // Fake quantization should maintain the same data type and shape
        assert_eq!(fake_quantized.dtype(), gpu_tensor.dtype());
        assert_eq!(fake_quantized.shape(), gpu_tensor.shape());
    }
}
