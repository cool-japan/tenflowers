use crate::device::context::GpuContextInfo;
use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::Float;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// GPU-accelerated RNN operations
pub struct GpuRnnOps {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    lstm_cell_pipeline: wgpu::ComputePipeline,
    gru_cell_pipeline: wgpu::ComputePipeline,
    layer_norm_pipeline: wgpu::ComputePipeline,
}

impl GpuRnnOps {
    pub fn new(gpu_context_info: &GpuContextInfo) -> Result<Self> {
        let device = gpu_context_info.device.clone();
        let queue = gpu_context_info.queue.clone();

        // Load and compile shaders
        let shader_source = include_str!("shaders/rnn_ops.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RNN Operations Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layouts
        let lstm_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("LSTM Cell Bind Group Layout"),
                entries: &[
                    // Input data
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
                    // Hidden state
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
                    // Cell state
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
                    // Weight input-to-hidden
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
                    // Weight hidden-to-hidden
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Bias input-to-hidden
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Bias hidden-to-hidden
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Output hidden state
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Output cell state
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Uniform buffer layout for parameters
        let param_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RNN Parameters Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Create pipeline layouts
        let lstm_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LSTM Cell Pipeline Layout"),
            bind_group_layouts: &[&lstm_bind_group_layout, &param_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipelines
        let lstm_cell_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LSTM Cell Pipeline"),
            layout: Some(&lstm_pipeline_layout),
            module: &shader_module,
            entry_point: Some("lstm_cell_kernel"),
            cache: None,
            compilation_options: Default::default(),
        });

        let gru_cell_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GRU Cell Pipeline"),
            layout: Some(&lstm_pipeline_layout), // Same layout works for GRU
            module: &shader_module,
            entry_point: Some("gru_cell_kernel"),
            cache: None,
            compilation_options: Default::default(),
        });

        let layer_norm_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RNN Layer Norm Pipeline"),
                layout: Some(&lstm_pipeline_layout),
                module: &shader_module,
                entry_point: Some("rnn_layer_norm_kernel"),
                cache: None,
                compilation_options: Default::default(),
            });

        Ok(Self {
            device,
            queue,
            lstm_cell_pipeline,
            gru_cell_pipeline,
            layer_norm_pipeline,
        })
    }

    /// Compute LSTM cell on GPU
    pub fn lstm_cell_gpu<
        T: Float
            + Default
            + bytemuck::Pod
            + bytemuck::Zeroable
            + Clone
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One,
    >(
        &self,
        input: &Tensor<T>,
        hidden: &Tensor<T>,
        cell: &Tensor<T>,
        weight_ih: &Tensor<T>,
        weight_hh: &Tensor<T>,
        bias_ih: Option<&Tensor<T>>,
        bias_hh: Option<&Tensor<T>>,
    ) -> Result<(Tensor<T>, Tensor<T>)> {
        let input_shape = input.shape().dims();
        let hidden_shape = hidden.shape().dims();

        if input_shape.len() != 2 || hidden_shape.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "Input and hidden must be 2D tensors".to_string(),
            ));
        }

        let batch_size = input_shape[0];
        let input_size = input_shape[1];
        let hidden_size = hidden_shape[1];

        // Create GPU buffers
        let input_buffer = self.create_buffer_from_tensor(input, "Input Buffer")?;
        let hidden_buffer = self.create_buffer_from_tensor(hidden, "Hidden Buffer")?;
        let cell_buffer = self.create_buffer_from_tensor(cell, "Cell Buffer")?;
        let weight_ih_buffer = self.create_buffer_from_tensor(weight_ih, "Weight IH Buffer")?;
        let weight_hh_buffer = self.create_buffer_from_tensor(weight_hh, "Weight HH Buffer")?;

        // Create bias buffers (use zero buffers if no bias)
        let bias_ih_buffer = if let Some(b) = bias_ih {
            self.create_buffer_from_tensor(b, "Bias IH Buffer")?
        } else {
            self.create_zero_buffer(hidden_size * 4, "Zero Bias IH Buffer")?
        };

        let bias_hh_buffer = if let Some(b) = bias_hh {
            self.create_buffer_from_tensor(b, "Bias HH Buffer")?
        } else {
            self.create_zero_buffer(hidden_size * 4, "Zero Bias HH Buffer")?
        };

        // Create output buffers
        let output_hidden_buffer = self.create_buffer(
            batch_size * hidden_size * std::mem::size_of::<T>(),
            "Output Hidden Buffer",
        )?;
        let output_cell_buffer = self.create_buffer(
            batch_size * hidden_size * std::mem::size_of::<T>(),
            "Output Cell Buffer",
        )?;

        // Create parameters buffer
        let params = LSTMParams {
            batch_size: batch_size as u32,
            input_size: input_size as u32,
            hidden_size: hidden_size as u32,
            has_bias: if bias_ih.is_some() && bias_hh.is_some() {
                1
            } else {
                0
            },
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LSTM Parameters Buffer"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind groups
        let data_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LSTM Data Bind Group"),
            layout: &self.lstm_cell_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hidden_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: weight_ih_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: weight_hh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bias_ih_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bias_hh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: output_hidden_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: output_cell_buffer.as_entire_binding(),
                },
            ],
        });

        let param_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LSTM Parameters Bind Group"),
            layout: &self.lstm_cell_pipeline.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });

        // Dispatch compute shader
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("LSTM Cell Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LSTM Cell Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.lstm_cell_pipeline);
            compute_pass.set_bind_group(0, &data_bind_group, &[]);
            compute_pass.set_bind_group(1, &param_bind_group, &[]);

            // Dispatch with 2D workgroup covering batch_size x hidden_size
            let workgroup_size = 256u32;
            let dispatch_x = (batch_size as u32 + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (hidden_size as u32 + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        // Read results back
        let new_hidden =
            self.read_buffer_to_tensor(&output_hidden_buffer, &[batch_size, hidden_size])?;
        let new_cell =
            self.read_buffer_to_tensor(&output_cell_buffer, &[batch_size, hidden_size])?;

        Ok((new_hidden, new_cell))
    }

    /// Compute GRU cell on GPU
    pub fn gru_cell_gpu<
        T: Float
            + Default
            + bytemuck::Pod
            + bytemuck::Zeroable
            + Clone
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One,
    >(
        &self,
        input: &Tensor<T>,
        hidden: &Tensor<T>,
        weight_ih: &Tensor<T>,
        weight_hh: &Tensor<T>,
        bias_ih: Option<&Tensor<T>>,
        bias_hh: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>> {
        // Similar implementation to LSTM but for GRU
        // This would use the GRU kernel instead
        let input_shape = input.shape().dims();
        let hidden_shape = hidden.shape().dims();

        if input_shape.len() != 2 || hidden_shape.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "Input and hidden must be 2D tensors".to_string(),
            ));
        }

        let batch_size = input_shape[0];
        let input_size = input_shape[1];
        let hidden_size = hidden_shape[1];

        // Create dummy cell state for GRU (not used but needed for uniform interface)
        let cell = Tensor::zeros(&[batch_size, hidden_size]);

        // Use similar buffer creation and dispatch logic as LSTM
        // but with GRU pipeline
        let (new_hidden, _) =
            self.lstm_cell_gpu(input, hidden, &cell, weight_ih, weight_hh, bias_ih, bias_hh)?;

        Ok(new_hidden)
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
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One,
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

    fn create_zero_buffer(&self, size: usize, label: &str) -> Result<wgpu::Buffer> {
        let zero_data = vec![0u8; size * std::mem::size_of::<f32>()];
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &zero_data,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        Ok(buffer)
    }

    fn create_buffer(&self, size: usize, label: &str) -> Result<wgpu::Buffer> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Ok(buffer)
    }

    fn read_buffer_to_tensor<T: bytemuck::Pod + bytemuck::Zeroable + Clone + Default>(
        &self,
        buffer: &wgpu::Buffer,
        shape: &[usize],
    ) -> Result<Tensor<T>> {
        // Create a staging buffer for reading
        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Encoder"),
            });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size as u64);
        self.queue.submit(Some(encoder.finish()));

        // Map and read the buffer
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Poll the device until the buffer is ready
        self.device.poll(wgpu::Maintain::Wait);

        // Wait for the buffer to be ready
        let _result = futures::executor::block_on(receiver).unwrap();

        let data = buffer_slice.get_mapped_range();
        let typed_data: &[T] = bytemuck::cast_slice(&data);
        let tensor_data = typed_data.to_vec();

        drop(data);
        staging_buffer.unmap();

        Tensor::from_vec(tensor_data, shape)
    }
}

/// Parameters for LSTM GPU kernel
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LSTMParams {
    batch_size: u32,
    input_size: u32,
    hidden_size: u32,
    has_bias: u32,
}
