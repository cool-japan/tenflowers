/// Async GPU kernel execution support
use super::*;
use crate::gpu::{ops::BinaryOp, GpuBuffer};
use crate::{Device, Result, TensorError};
use futures::channel::oneshot;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// Represents a pending GPU computation that can be awaited
pub struct GpuKernelFuture<T> {
    receiver: Option<oneshot::Receiver<Result<GpuBuffer<T>>>>,
    device: Arc<wgpu::Device>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Future for GpuKernelFuture<T> {
    type Output = Result<GpuBuffer<T>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        if let Some(receiver) = &mut this.receiver {
            match Pin::new(receiver).poll(cx) {
                Poll::Ready(Ok(result)) => {
                    this.receiver = None;
                    Poll::Ready(result)
                }
                Poll::Ready(Err(_)) => {
                    this.receiver = None;
                    Poll::Ready(Err(TensorError::compute_error_simple(
                        "GPU kernel execution failed".to_string(),
                    )))
                }
                Poll::Pending => Poll::Pending,
            }
        } else {
            Poll::Ready(Err(TensorError::compute_error_simple(
                "GPU future already completed".to_string(),
            )))
        }
    }
}

/// Async GPU execution manager
pub struct AsyncGpuExecutor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pending_kernels: std::sync::Mutex<Vec<oneshot::Sender<Result<GpuBuffer<f32>>>>>,
}

impl AsyncGpuExecutor {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            pending_kernels: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Execute a binary operation asynchronously
    pub fn execute_binary_op_async<T>(
        &self,
        input_a: &GpuBuffer<T>,
        input_b: &GpuBuffer<T>,
        operation: BinaryOp,
        output_len: usize,
    ) -> GpuKernelFuture<T>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
    {
        let (sender, receiver) = oneshot::channel();
        let device = Arc::clone(&self.device);
        let queue = Arc::clone(&self.queue);

        // Clone input buffer references for async execution
        let input_a_buffer = input_a.buffer_arc();
        let input_b_buffer = input_b.buffer_arc();
        let input_a_device = Arc::clone(&input_a.device);
        let input_a_queue = Arc::clone(&input_a.queue);
        let device_enum = input_a.device_enum();

        // Start async computation
        let computation_task = async move {
            let result = execute_binary_op_internal(
                &input_a_buffer,
                &input_b_buffer,
                operation,
                output_len,
                &device,
                &queue,
                &input_a_device,
                &input_a_queue,
                device_enum,
            )
            .await;

            let _ = sender.send(result);
        };

        // Spawn the task (in a real implementation, you'd use a proper async executor)
        std::thread::spawn(move || {
            pollster::block_on(computation_task);
        });

        GpuKernelFuture {
            receiver: Some(receiver),
            device: Arc::clone(&self.device),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Execute a reduction operation asynchronously
    pub fn execute_reduction_op_async<T>(
        &self,
        input_buffer: &GpuBuffer<T>,
        operation: ReductionOp,
        output_len: usize,
    ) -> GpuKernelFuture<T>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
    {
        let (sender, receiver) = oneshot::channel();
        let device = Arc::clone(&self.device);
        let queue = Arc::clone(&self.queue);

        let input_buffer_clone = input_buffer.buffer_arc();
        let input_device = Arc::clone(&input_buffer.device);
        let input_queue = Arc::clone(&input_buffer.queue);
        let device_enum = input_buffer.device_enum();

        let computation_task = async move {
            let result = execute_reduction_op_internal(
                &input_buffer_clone,
                operation,
                output_len,
                &device,
                &queue,
                &input_device,
                &input_queue,
                device_enum,
            )
            .await;

            let _ = sender.send(result);
        };

        std::thread::spawn(move || {
            pollster::block_on(computation_task);
        });

        GpuKernelFuture {
            receiver: Some(receiver),
            device: Arc::clone(&self.device),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Execute a matrix multiplication asynchronously
    pub fn execute_matmul_async<T>(
        &self,
        input_a: &GpuBuffer<T>,
        input_b: &GpuBuffer<T>,
        m: usize,
        k: usize,
        n: usize,
        batch_size: usize,
    ) -> GpuKernelFuture<T>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
    {
        let (sender, receiver) = oneshot::channel();
        let device = Arc::clone(&self.device);
        let queue = Arc::clone(&self.queue);

        let input_a_buffer = input_a.buffer_arc();
        let input_b_buffer = input_b.buffer_arc();
        let input_device = Arc::clone(&input_a.device);
        let input_queue = Arc::clone(&input_a.queue);
        let device_enum = input_a.device_enum();

        let computation_task = async move {
            let result = execute_matmul_internal(
                &input_a_buffer,
                &input_b_buffer,
                m,
                k,
                n,
                batch_size,
                &device,
                &queue,
                &input_device,
                &input_queue,
                device_enum,
            )
            .await;

            let _ = sender.send(result);
        };

        std::thread::spawn(move || {
            pollster::block_on(computation_task);
        });

        GpuKernelFuture {
            receiver: Some(receiver),
            device: Arc::clone(&self.device),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Wait for all pending kernels to complete
    pub async fn wait_for_completion(&self) -> Result<()> {
        // In a real implementation, this would wait for all pending GPU operations
        // For now, just poll the device to ensure completion
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    /// Check if any kernels are pending
    pub fn has_pending_kernels(&self) -> bool {
        let pending = self.pending_kernels.lock().unwrap();
        !pending.is_empty()
    }
}

/// Internal async binary operation implementation
async fn execute_binary_op_internal<T>(
    input_a: &wgpu::Buffer,
    input_b: &wgpu::Buffer,
    operation: BinaryOp,
    output_len: usize,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_device: &Arc<wgpu::Device>,
    input_queue: &Arc<wgpu::Queue>,
    device_enum: Device,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("async_binary_op_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Load shader based on operation
    let shader_source = include_str!("shaders/binary_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("async_binary_ops_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("async_binary_op_bind_group_layout"),
        entries: &[
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("async_binary_op_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("async_binary_op_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let entry_point = match operation {
        BinaryOp::Add => "add_op",
        BinaryOp::Sub => "sub_op",
        BinaryOp::Mul => "mul_op",
        BinaryOp::Div => "div_op",
        BinaryOp::Pow => "pow_op",
        BinaryOp::PReLU => "prelu_op",
        BinaryOp::Min => "min_op",
        BinaryOp::Max => "max_op",
        BinaryOp::MatMul => "matmul_op",
    };

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("async_binary_op_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(entry_point),
        cache: None,
        compilation_options: Default::default(),
    });

    // Submit work without blocking (key difference from sync version)
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("async_binary_op_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("async_binary_op_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size = 64;
        let num_workgroups = (output_len + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    // Submit without waiting (key difference from sync version)
    queue.submit(std::iter::once(encoder.finish()));

    // Return buffer immediately without waiting for completion
    let device_id = match &device_enum {
        Device::Gpu(id) => *id,
        _ => {
            return Err(TensorError::device_error_simple(
                "Expected GPU device".to_string(),
            ))
        }
    };

    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        Arc::clone(input_device),
        Arc::clone(input_queue),
        Device::Gpu(device_id),
        output_len,
    ))
}

/// Internal async reduction operation implementation  
async fn execute_reduction_op_internal<T>(
    input_buffer: &wgpu::Buffer,
    operation: ReductionOp,
    output_len: usize,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_device: &Arc<wgpu::Device>,
    input_queue: &Arc<wgpu::Queue>,
    device_enum: Device,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("async_reduction_op_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create metadata buffer
    use wgpu::util::DeviceExt;
    let input_len = input_buffer.size() as u32 / std::mem::size_of::<T>() as u32;
    let metadata = [input_len, output_len as u32, 0u32];
    let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("async_reduction_metadata"),
        contents: bytemuck::cast_slice(&metadata),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Load shader
    let shader_source = include_str!("shaders/reduction_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("async_reduction_ops_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("async_reduction_bind_group_layout"),
        entries: &[
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
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
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
        ],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("async_reduction_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: metadata_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("async_reduction_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let entry_point = match operation {
        ReductionOp::Sum => "sum_reduce",
        ReductionOp::Mean => "mean_reduce",
        ReductionOp::Max => "max_reduce",
        ReductionOp::Min => "min_reduce",
        ReductionOp::Product => "product_reduce",
        ReductionOp::InfNanDetection => "inf_nan_reduce",
        ReductionOp::ArgMax => "argmax_reduce",
        ReductionOp::ArgMin => "argmin_reduce",
        ReductionOp::All => "all_reduce",
        ReductionOp::Any => "any_reduce",
        ReductionOp::TopK => "topk_reduce",
        ReductionOp::Prod => "prod_reduce",
        ReductionOp::Variance => "variance_reduce",
    };

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("async_reduction_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(entry_point),
        cache: None,
        compilation_options: Default::default(),
    });

    // Submit work without blocking
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("async_reduction_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("async_reduction_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size = 64;
        let num_workgroups = (input_len + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));

    // Return buffer immediately
    let device_id = match &device_enum {
        Device::Gpu(id) => *id,
        _ => {
            return Err(TensorError::device_error_simple(
                "Expected GPU device".to_string(),
            ))
        }
    };

    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        Arc::clone(input_device),
        Arc::clone(input_queue),
        Device::Gpu(device_id),
        output_len,
    ))
}

/// Internal async matrix multiplication implementation
async fn execute_matmul_internal<T>(
    input_a: &wgpu::Buffer,
    input_b: &wgpu::Buffer,
    m: usize,
    k: usize,
    n: usize,
    batch_size: usize,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_device: &Arc<wgpu::Device>,
    input_queue: &Arc<wgpu::Queue>,
    device_enum: Device,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    let output_len = batch_size * m * n;

    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("async_matmul_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create metadata buffer with matrix dimensions
    use wgpu::util::DeviceExt;
    let metadata = [m as u32, k as u32, n as u32, batch_size as u32];
    let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("async_matmul_metadata"),
        contents: bytemuck::cast_slice(&metadata),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Load shader
    let shader_source = include_str!("shaders/matmul_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("async_matmul_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("async_matmul_bind_group_layout"),
        entries: &[
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
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
        ],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("async_matmul_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: metadata_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("async_matmul_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("async_matmul_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("matmul"),
        cache: None,
        compilation_options: Default::default(),
    });

    // Submit work without blocking
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("async_matmul_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("async_matmul_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch with appropriate workgroup dimensions for matrix multiplication
        let tile_size = 16;
        let workgroups_x = (n + tile_size - 1) / tile_size;
        let workgroups_y = (m + tile_size - 1) / tile_size;

        compute_pass.dispatch_workgroups(
            workgroups_x as u32,
            workgroups_y as u32,
            batch_size as u32,
        );
    }

    queue.submit(std::iter::once(encoder.finish()));

    // Return buffer immediately
    let device_id = match &device_enum {
        Device::Gpu(id) => *id,
        _ => {
            return Err(TensorError::device_error_simple(
                "Expected GPU device".to_string(),
            ))
        }
    };

    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        Arc::clone(input_device),
        Arc::clone(input_queue),
        Device::Gpu(device_id),
        output_len,
    ))
}
