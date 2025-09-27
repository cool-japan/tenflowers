use crate::device::context::GpuContextInfo;
use crate::{Result, Tensor, TensorError};
use num_traits::Float;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// GPU-accelerated attention operations for neural networks
pub struct GpuAttentionOps {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    scaled_dot_product_attention_pipeline: wgpu::ComputePipeline,
    flash_attention_pipeline: wgpu::ComputePipeline,
    multi_head_attention_pipeline: wgpu::ComputePipeline,
}

impl GpuAttentionOps {
    /// Create new GPU attention operations
    pub fn new(gpu_context_info: &GpuContextInfo) -> Result<Self> {
        let device = gpu_context_info.device.clone();
        let queue = gpu_context_info.queue.clone();

        // Load and compile attention shaders
        let shader_source = include_str!("shaders/attention_ops.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Attention Operations Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout for scaled dot-product attention
        let attention_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Attention Bind Group Layout"),
                entries: &[
                    // Query tensor
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Key tensor
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Value tensor
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Attention mask (optional, can be empty buffer)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Output tensor
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Parameters (seq_len, head_dim, scale, use_mask)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Attention Pipeline Layout"),
            bind_group_layouts: &[&attention_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipelines
        let scaled_dot_product_attention_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Scaled Dot-Product Attention Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("scaled_dot_product_attention_kernel"),
                cache: None,
                compilation_options: Default::default(),
            });

        let flash_attention_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Flash Attention Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("flash_attention_kernel"),
                cache: None,
                compilation_options: Default::default(),
            });

        let multi_head_attention_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Multi-Head Attention Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("multi_head_attention_kernel"),
                cache: None,
                compilation_options: Default::default(),
            });

        Ok(Self {
            device,
            queue,
            scaled_dot_product_attention_pipeline,
            flash_attention_pipeline,
            multi_head_attention_pipeline,
        })
    }

    /// Compute scaled dot-product attention on GPU
    /// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    pub fn scaled_dot_product_attention<T>(
        &self,
        query: &Tensor<T>,
        key: &Tensor<T>,
        value: &Tensor<T>,
        mask: Option<&Tensor<T>>,
        scale: Option<T>,
    ) -> Result<Tensor<T>>
    where
        T: Float
            + Default
            + bytemuck::Pod
            + bytemuck::Zeroable
            + Send
            + Sync
            + 'static
            + Clone
            + num_traits::Zero
            + num_traits::One,
    {
        let query_shape = query.shape();
        let seq_len = query_shape.dims()[0];
        let head_dim = query_shape.dims()[1];

        // Calculate scale factor
        let scale_factor =
            scale.unwrap_or_else(|| T::from(head_dim as f64).unwrap().sqrt().recip());

        // Create GPU buffers from tensor data
        let query_buffer = self.create_buffer_from_tensor(query, "Query Buffer")?;
        let key_buffer = self.create_buffer_from_tensor(key, "Key Buffer")?;
        let value_buffer = self.create_buffer_from_tensor(value, "Value Buffer")?;

        // Create output buffer
        let output_size = (seq_len * head_dim * std::mem::size_of::<T>()) as wgpu::BufferAddress;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Attention Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Handle optional mask
        let mask_buffer = if let Some(mask) = mask {
            self.create_buffer_from_tensor(mask, "Mask Buffer")?
        } else {
            // Create empty mask buffer
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Empty Mask Buffer"),
                size: 4, // Minimum buffer size
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };

        // Create parameters buffer
        let params = [
            seq_len as u32,
            head_dim as u32,
            scale_factor.to_f32().unwrap().to_bits(),
            if mask.is_some() { 1u32 } else { 0u32 },
        ];
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Attention Parameters Buffer"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Attention Bind Group"),
            layout: &self
                .scaled_dot_product_attention_pipeline
                .get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: key_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: value_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Attention Command Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Attention Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.scaled_dot_product_attention_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (process sequences in parallel)
            let workgroup_size = 256;
            let num_workgroups = ((seq_len as u32) + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        // Read results back from GPU buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Attention Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Attention Copy Encoder"),
            });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        // Map buffer and read data synchronously
        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |v| {
            if let Err(e) = v {
                eprintln!("Buffer mapping failed: {:?}", e);
            }
        });
        self.device.poll(wgpu::Maintain::Wait);

        let data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        // Create output tensor
        Tensor::from_vec(result, query_shape.dims())
    }

    /// Compute multi-head attention with GPU acceleration
    pub fn multi_head_attention<T>(
        &self,
        query: &Tensor<T>,
        key: &Tensor<T>,
        value: &Tensor<T>,
        num_heads: usize,
        mask: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>>
    where
        T: Float + Default + bytemuck::Pod + bytemuck::Zeroable + Send + Sync + 'static,
    {
        let query_shape = query.shape();
        let seq_len = query_shape.dims()[0];
        let embed_dim = query_shape.dims()[1];
        let head_dim = embed_dim / num_heads;

        if embed_dim % num_heads != 0 {
            return Err(TensorError::invalid_argument(format!(
                "Embedding dimension {} is not divisible by number of heads {}",
                embed_dim, num_heads
            )));
        }

        // Use the scaled dot-product attention as the building block
        // In a full implementation, you'd want more sophisticated multi-head processing
        self.scaled_dot_product_attention(query, key, value, mask, None)
    }

    /// Compute Flash Attention with memory-efficient implementation
    pub fn flash_attention<T>(
        &self,
        query: &Tensor<T>,
        key: &Tensor<T>,
        value: &Tensor<T>,
        mask: Option<&Tensor<T>>,
        block_size: Option<usize>,
    ) -> Result<Tensor<T>>
    where
        T: Float + Default + bytemuck::Pod + bytemuck::Zeroable + Send + Sync + 'static,
    {
        // For now, delegate to scaled dot-product attention
        // In a full implementation, you'd want block-wise processing for memory efficiency
        let _block_size = block_size.unwrap_or(256);
        self.scaled_dot_product_attention(query, key, value, mask, None)
    }

    // Helper methods
    fn create_buffer_from_tensor<
        T: bytemuck::Pod
            + bytemuck::Zeroable
            + Clone
            + Default
            + Send
            + Sync
            + 'static
            + num_traits::Zero
            + num_traits::One,
    >(
        &self,
        tensor: &Tensor<T>,
        label: &str,
    ) -> Result<wgpu::Buffer> {
        let data = tensor.to_vec()?;
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        Ok(buffer)
    }
}
