// Enhanced tensor with zero-copy view support
#[cfg(feature = "gpu")]
use crate::memory::PooledBuffer;
use crate::memory::{MemoryAliasDetector, StridedView};
use crate::tensor::TensorStorage;
use crate::{Device, Result, Shape, Tensor, TensorError};
use std::sync::Arc;

/// Enhanced tensor that supports zero-copy views and strided operations
#[derive(Debug)]
pub struct TensorView<T> {
    /// Reference to the underlying tensor data
    pub storage: ViewStorage<T>,
    /// Strided view information
    pub view: StridedView,
    /// Device where the tensor is stored
    device: Device,
    /// Whether this tensor requires gradient computation
    requires_grad: bool,
    /// Gradient tensor (if any)
    grad: Option<Arc<TensorView<T>>>,
    /// Memory alias detector for safety
    alias_detector: Arc<MemoryAliasDetector>,
}

/// Storage for tensor views
#[derive(Debug, Clone)]
pub enum ViewStorage<T> {
    /// Reference to CPU tensor storage
    CpuRef(Arc<ndarray::ArrayD<T>>),
    /// Reference to GPU pooled buffer
    #[cfg(feature = "gpu")]
    GpuPooled(Arc<PooledBuffer<'static>>),
    /// Reference to regular GPU buffer
    #[cfg(feature = "gpu")]
    GpuRef(Arc<crate::gpu::buffer::GpuBuffer<T>>),
}

impl<T> TensorView<T> {
    /// Get a unique buffer ID for alias detection
    fn get_buffer_id(&self) -> usize {
        match &self.storage {
            ViewStorage::CpuRef(arr) => Arc::as_ptr(arr) as usize,
            #[cfg(feature = "gpu")]
            ViewStorage::GpuRef(gpu_buffer) => Arc::as_ptr(gpu_buffer) as usize,
            #[cfg(feature = "gpu")]
            ViewStorage::GpuPooled(pooled_buffer) => Arc::as_ptr(pooled_buffer) as usize,
        }
    }

    /// Get the number of elements in the tensor
    pub fn numel(&self) -> usize {
        self.view.shape.iter().product()
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.view.size_bytes()
    }

    /// Check if the tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        self.view.is_contiguous()
    }
}

impl<T: Clone + Default> TensorView<T> {
    /// Create a new tensor view from an existing tensor
    pub fn from_tensor(tensor: &Tensor<T>) -> Result<Self>
    where
        T: Clone + Send + Sync + 'static,
    {
        let element_size = std::mem::size_of::<T>();
        let shape = tensor.shape().dims().to_vec();
        let strides = compute_default_strides(&shape, element_size);

        let view = StridedView::new(0, shape, strides, element_size);
        let alias_detector = Arc::new(MemoryAliasDetector::new());

        let storage = match &tensor.storage {
            TensorStorage::Cpu(arr) => ViewStorage::CpuRef(Arc::new(arr.clone())),
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(gpu_buffer) => ViewStorage::GpuRef(Arc::new(gpu_buffer.clone())),
        };

        Ok(Self {
            storage,
            view,
            device: *tensor.device(),
            requires_grad: tensor.requires_grad(),
            grad: None,
            alias_detector,
        })
    }

    /// Create a zero-copy transpose view
    pub fn transpose(&self, axes: &[usize]) -> Result<TensorView<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        let new_view = self.view.transpose(axes)?;

        // Check for memory aliasing
        let buffer_id = self.get_buffer_id();
        if self
            .alias_detector
            .check_alias(buffer_id, new_view.offset, new_view.size_bytes())
        {
            return Err(TensorError::invalid_argument(
                "Transpose would create memory alias".to_string(),
            ));
        }

        // Register the new view
        self.alias_detector
            .register_view(buffer_id, new_view.offset, new_view.size_bytes());

        Ok(Self {
            storage: self.storage.clone(),
            view: new_view,
            device: self.device,
            requires_grad: self.requires_grad,
            grad: None,
            alias_detector: Arc::clone(&self.alias_detector),
        })
    }

    /// Create a zero-copy reshape view (when possible)
    pub fn reshape(&self, new_shape: &[usize]) -> Result<TensorView<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        let new_view = self.view.reshape(new_shape)?;

        // Check for memory aliasing
        let buffer_id = self.get_buffer_id();
        if self
            .alias_detector
            .check_alias(buffer_id, new_view.offset, new_view.size_bytes())
        {
            return Err(TensorError::invalid_argument(
                "Reshape would create memory alias".to_string(),
            ));
        }

        // Register the new view
        self.alias_detector
            .register_view(buffer_id, new_view.offset, new_view.size_bytes());

        Ok(Self {
            storage: self.storage.clone(),
            view: new_view,
            device: self.device,
            requires_grad: self.requires_grad,
            grad: None,
            alias_detector: Arc::clone(&self.alias_detector),
        })
    }

    /// Create a zero-copy slice view
    pub fn slice(&self, ranges: &[(usize, usize)]) -> Result<TensorView<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        let new_view = self.view.slice(ranges)?;

        // Check for memory aliasing
        let buffer_id = self.get_buffer_id();
        if self
            .alias_detector
            .check_alias(buffer_id, new_view.offset, new_view.size_bytes())
        {
            return Err(TensorError::invalid_argument(
                "Slice would create memory alias".to_string(),
            ));
        }

        // Register the new view
        self.alias_detector
            .register_view(buffer_id, new_view.offset, new_view.size_bytes());

        Ok(Self {
            storage: self.storage.clone(),
            view: new_view,
            device: self.device,
            requires_grad: self.requires_grad,
            grad: None,
            alias_detector: Arc::clone(&self.alias_detector),
        })
    }

    /// Convert back to a regular tensor (may require data copy)
    pub fn to_tensor(&self) -> Result<Tensor<T>>
    where
        T: Clone + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        if self.view.is_contiguous() {
            // Zero-copy conversion for contiguous views
            match &self.storage {
                ViewStorage::CpuRef(arr) => {
                    let _shape = Shape::new(self.view.shape.clone());
                    Ok(Tensor::from_array((**arr).clone()))
                }
                #[cfg(feature = "gpu")]
                ViewStorage::GpuRef(gpu_buffer) => {
                    let shape = Shape::new(self.view.shape.clone());
                    let mut result = Tensor::from_gpu_buffer((**gpu_buffer).clone(), shape);
                    result.set_requires_grad(self.requires_grad);
                    Ok(result)
                }
                #[cfg(feature = "gpu")]
                ViewStorage::GpuPooled(pooled_buffer) => {
                    // For pooled buffers, we need to extract the data from the pool
                    use wgpu::util::DeviceExt;

                    let pool_buffer = pooled_buffer.buffer();
                    let device = &crate::device::context::get_gpu_context(match self.device {
                        Device::Gpu(id) => id,
                        _ => {
                            return Err(TensorError::device_error_simple(
                                "Expected GPU device".to_string(),
                            ))
                        }
                    })?
                    .device;

                    // Create a new buffer with the exact size needed
                    let data_size =
                        self.view.shape.iter().product::<usize>() * std::mem::size_of::<T>();
                    let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("pooled_to_tensor_buffer"),
                        size: data_size as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    // Copy data from pool buffer to new buffer
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("pooled_buffer_copy"),
                        });

                    encoder.copy_buffer_to_buffer(
                        pool_buffer,
                        pooled_buffer.offset() as u64,
                        &new_buffer,
                        0,
                        data_size as u64,
                    );

                    let queue = &crate::device::context::get_gpu_context(match self.device {
                        Device::Gpu(id) => id,
                        _ => {
                            return Err(TensorError::device_error_simple(
                                "Expected GPU device".to_string(),
                            ))
                        }
                    })?
                    .queue;

                    queue.submit(std::iter::once(encoder.finish()));
                    device.poll(wgpu::Maintain::Wait);

                    // Create GPU buffer wrapper
                    let device_id = match self.device {
                        Device::Gpu(id) => id,
                        _ => {
                            return Err(TensorError::device_error_simple(
                                "Expected GPU device".to_string(),
                            ))
                        }
                    };

                    let ctx = crate::device::context::get_gpu_context(device_id)?;
                    let gpu_buffer = crate::gpu::buffer::GpuBuffer::from_wgpu_buffer(
                        new_buffer,
                        ctx.device.clone(),
                        ctx.queue.clone(),
                        Device::Gpu(device_id),
                        self.view.shape.iter().product::<usize>(),
                    );

                    let shape = Shape::new(self.view.shape.clone());
                    let mut result = Tensor::from_gpu_buffer(gpu_buffer, shape);
                    result.set_requires_grad(self.requires_grad);
                    Ok(result)
                }
            }
        } else {
            // Non-contiguous view requires data copy
            self.materialize()
        }
    }

    /// Materialize the view into a new contiguous tensor
    fn materialize(&self) -> Result<Tensor<T>>
    where
        T: Clone + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        match &self.storage {
            ViewStorage::CpuRef(arr) => {
                // Create new contiguous array
                let total_elements: usize = self.view.shape.iter().product();
                let mut new_data = Vec::with_capacity(total_elements);

                // Copy data using strided indexing
                for flat_index in 0..total_elements {
                    let multi_index = flat_to_multi_index(flat_index, &self.view.shape);
                    let strided_index =
                        multi_to_strided_index(&multi_index, &self.view.strides, self.view.offset);

                    let byte_index = strided_index / self.view.element_size;
                    if let Some(slice) = arr.as_slice() {
                        if byte_index < slice.len() {
                            new_data.push(slice[byte_index]);
                        }
                    }
                }

                let _shape = Shape::new(self.view.shape.clone());
                let new_array =
                    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&self.view.shape), new_data)
                        .map_err(|e| TensorError::invalid_argument(e.to_string()))?;

                Ok(Tensor::from_array(new_array))
            }
            #[cfg(feature = "gpu")]
            ViewStorage::GpuRef(gpu_buffer) => {
                // For GPU, we need to implement a kernel to gather strided data
                use wgpu::util::DeviceExt;

                let device_id = match self.device {
                    Device::Gpu(id) => id,
                    _ => {
                        return Err(TensorError::device_error_simple(
                            "Expected GPU device".to_string(),
                        ))
                    }
                };

                let ctx = crate::device::context::get_gpu_context(device_id)?;
                let device = &ctx.device;
                let queue = &ctx.queue;

                let total_elements: usize = self.view.shape.iter().product();
                let output_size = total_elements * std::mem::size_of::<T>();

                // Create output buffer
                let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("strided_materialize_output"),
                    size: output_size as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                // Create info buffer with view parameters
                #[repr(C)]
                #[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
                struct StridedInfo {
                    ndim: u32,
                    total_elements: u32,
                    offset: u32,
                    element_size: u32,
                }

                let info = StridedInfo {
                    ndim: self.view.shape.len() as u32,
                    total_elements: total_elements as u32,
                    offset: self.view.offset as u32,
                    element_size: self.view.element_size as u32,
                };

                let info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("strided_info"),
                    contents: bytemuck::cast_slice(&[info]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                // Create shape and strides buffers
                let shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("strided_shape"),
                    contents: bytemuck::cast_slice(
                        &self
                            .view
                            .shape
                            .iter()
                            .map(|&x| x as u32)
                            .collect::<Vec<u32>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let strides_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("strided_strides"),
                    contents: bytemuck::cast_slice(
                        &self
                            .view
                            .strides
                            .iter()
                            .map(|&x| x as u32)
                            .collect::<Vec<u32>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                // Create compute shader for strided materialization
                let shader_source = include_str!("gpu/shaders/strided_ops.wgsl");
                let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("strided_materialize_shader"),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                });

                // Create bind group layout
                let bind_group_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("strided_materialize_bind_group_layout"),
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
                                    ty: wgpu::BufferBindingType::Uniform,
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
                        ],
                    });

                // Create bind group
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("strided_materialize_bind_group"),
                    layout: &bind_group_layout,
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
                            resource: info_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: shape_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: strides_buffer.as_entire_binding(),
                        },
                    ],
                });

                // Create pipeline
                let pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("strided_materialize_pipeline_layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

                let compute_pipeline =
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("strided_materialize_pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &shader_module,
                        entry_point: Some("strided_materialize"),
                        cache: None,
                        compilation_options: Default::default(),
                    });

                // Execute compute shader
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("strided_materialize_encoder"),
                });

                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("strided_materialize_pass"),
                            timestamp_writes: None,
                        });

                    compute_pass.set_pipeline(&compute_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);

                    let workgroup_size = 64;
                    let num_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;
                    compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
                }

                queue.submit(std::iter::once(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);

                // Create result GPU buffer
                let gpu_buffer_result = crate::gpu::buffer::GpuBuffer::from_wgpu_buffer(
                    output_buffer,
                    ctx.device.clone(),
                    ctx.queue.clone(),
                    Device::Gpu(device_id),
                    total_elements,
                );

                let shape = Shape::new(self.view.shape.clone());
                let mut result = Tensor::from_gpu_buffer(gpu_buffer_result, shape);
                result.set_requires_grad(self.requires_grad);
                Ok(result)
            }
            #[cfg(feature = "gpu")]
            ViewStorage::GpuPooled(pooled_buffer) => {
                // For pooled buffers, we need to implement strided materialization from pool
                use wgpu::util::DeviceExt;

                let device_id = match self.device {
                    Device::Gpu(id) => id,
                    _ => {
                        return Err(TensorError::device_error_simple(
                            "Expected GPU device".to_string(),
                        ))
                    }
                };

                let ctx = crate::device::context::get_gpu_context(device_id)?;
                let device = &ctx.device;
                let queue = &ctx.queue;

                let total_elements: usize = self.view.shape.iter().product();
                let output_size = total_elements * std::mem::size_of::<T>();

                // Create output buffer
                let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pooled_strided_materialize_output"),
                    size: output_size as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                // Create info buffer with view parameters including pool offset
                #[repr(C)]
                #[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
                struct PooledStridedInfo {
                    ndim: u32,
                    total_elements: u32,
                    offset: u32,
                    element_size: u32,
                    pool_offset: u32, // Additional offset from pooled buffer
                    pad: [u32; 3],    // Padding for alignment
                }

                let info = PooledStridedInfo {
                    ndim: self.view.shape.len() as u32,
                    total_elements: total_elements as u32,
                    offset: self.view.offset as u32,
                    element_size: self.view.element_size as u32,
                    pool_offset: (pooled_buffer.offset() / std::mem::size_of::<T>()) as u32,
                    pad: [0; 3],
                };

                let info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pooled_strided_info"),
                    contents: bytemuck::cast_slice(&[info]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                // Create shape and strides buffers
                let shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pooled_strided_shape"),
                    contents: bytemuck::cast_slice(
                        &self
                            .view
                            .shape
                            .iter()
                            .map(|&x| x as u32)
                            .collect::<Vec<u32>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let strides_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pooled_strided_strides"),
                    contents: bytemuck::cast_slice(
                        &self
                            .view
                            .strides
                            .iter()
                            .map(|&x| x as u32)
                            .collect::<Vec<u32>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                // Create compute shader for pooled strided materialization
                let shader_source = include_str!("gpu/shaders/strided_ops.wgsl");
                let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("pooled_strided_materialize_shader"),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                });

                // Create bind group layout (same as regular strided)
                let bind_group_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("pooled_strided_materialize_bind_group_layout"),
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
                                    ty: wgpu::BufferBindingType::Uniform,
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
                        ],
                    });

                // Create bind group using pool buffer
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pooled_strided_materialize_bind_group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: pooled_buffer.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: info_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: shape_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: strides_buffer.as_entire_binding(),
                        },
                    ],
                });

                // Create pipeline
                let pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("pooled_strided_materialize_pipeline_layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

                let compute_pipeline =
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("pooled_strided_materialize_pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &shader_module,
                        entry_point: Some("strided_materialize"),
                        cache: None,
                        compilation_options: Default::default(),
                    });

                // Execute compute shader
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pooled_strided_materialize_encoder"),
                });

                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("pooled_strided_materialize_pass"),
                            timestamp_writes: None,
                        });

                    compute_pass.set_pipeline(&compute_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);

                    let workgroup_size = 64;
                    let num_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;
                    compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
                }

                queue.submit(std::iter::once(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);

                // Create result GPU buffer
                let gpu_buffer_result = crate::gpu::buffer::GpuBuffer::from_wgpu_buffer(
                    output_buffer,
                    ctx.device.clone(),
                    ctx.queue.clone(),
                    Device::Gpu(device_id),
                    total_elements,
                );

                let shape = Shape::new(self.view.shape.clone());
                let mut result = Tensor::from_gpu_buffer(gpu_buffer_result, shape);
                result.set_requires_grad(self.requires_grad);
                Ok(result)
            }
        }
    }

    /// Get the shape of the tensor view
    pub fn shape(&self) -> &[usize] {
        &self.view.shape
    }

    /// Get the strides of the tensor view
    pub fn strides(&self) -> &[usize] {
        &self.view.strides
    }

    /// Get the device where the tensor is stored
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if the tensor requires gradient computation
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set whether the tensor requires gradient computation
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }
}

impl<T> Clone for TensorView<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        // Register the cloned view in the alias detector
        let buffer_id = self.get_buffer_id();
        self.alias_detector
            .register_view(buffer_id, self.view.offset, self.view.size_bytes());

        Self {
            storage: self.storage.clone(),
            view: self.view.clone(),
            device: self.device,
            requires_grad: self.requires_grad,
            grad: self.grad.clone(),
            alias_detector: Arc::clone(&self.alias_detector),
        }
    }
}

impl<T> Drop for TensorView<T> {
    fn drop(&mut self) {
        // Unregister the view from the alias detector
        let buffer_id = self.get_buffer_id();
        self.alias_detector
            .unregister_view(buffer_id, self.view.offset, self.view.size_bytes());
    }
}

/// Memory-efficient tensor operations using views
pub struct TensorViewOps;

impl TensorViewOps {
    /// Perform zero-copy transpose if possible
    pub fn transpose_zero_copy<T>(tensor: &TensorView<T>, axes: &[usize]) -> Result<TensorView<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        tensor.transpose(axes)
    }

    /// Perform zero-copy reshape if possible
    pub fn reshape_zero_copy<T>(
        tensor: &TensorView<T>,
        new_shape: &[usize],
    ) -> Result<TensorView<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        tensor.reshape(new_shape)
    }

    /// Create a zero-copy slice view
    pub fn slice_zero_copy<T>(
        tensor: &TensorView<T>,
        ranges: &[(usize, usize)],
    ) -> Result<TensorView<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        tensor.slice(ranges)
    }

    /// Check if two tensor views share memory
    pub fn shares_memory<T>(tensor1: &TensorView<T>, tensor2: &TensorView<T>) -> bool
    where
        T: Clone + Default,
    {
        tensor1.get_buffer_id() == tensor2.get_buffer_id()
    }

    /// Get memory usage statistics for a tensor view
    pub fn memory_stats<T>(tensor: &TensorView<T>) -> MemoryStats
    where
        T: Clone + Default,
    {
        MemoryStats {
            total_elements: tensor.numel(),
            size_bytes: tensor.size_bytes(),
            is_contiguous: tensor.is_contiguous(),
            has_aliases: Self::has_memory_aliases(tensor),
        }
    }

    /// Check if a tensor view has memory aliases
    ///
    /// Returns true if the tensor's memory region overlaps with any other tracked tensors.
    /// This helps detect potentially unsafe operations on aliased memory.
    pub fn has_memory_aliases<T>(tensor: &TensorView<T>) -> bool
    where
        T: Clone + Default,
    {
        // Use the alias detector to check for overlapping memory regions
        let buffer_id = tensor.get_buffer_id();
        let start_offset = tensor.view.offset;
        let size_bytes = tensor.size_bytes();

        // Check with the memory alias detector
        tensor
            .alias_detector
            .check_alias(buffer_id, start_offset, size_bytes)
    }
}

/// Memory usage statistics for tensor views
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_elements: usize,
    pub size_bytes: usize,
    pub is_contiguous: bool,
    pub has_aliases: bool,
}

/// Utility functions for index calculations
fn flat_to_multi_index(flat_index: usize, shape: &[usize]) -> Vec<usize> {
    let mut multi_index = Vec::with_capacity(shape.len());
    let mut remaining = flat_index;

    for &dim in shape.iter().rev() {
        multi_index.push(remaining % dim);
        remaining /= dim;
    }

    multi_index.reverse();
    multi_index
}

fn multi_to_strided_index(multi_index: &[usize], strides: &[usize], offset: usize) -> usize {
    let mut strided_index = offset;
    for (idx, &stride) in multi_index.iter().zip(strides.iter()) {
        strided_index += idx * stride;
    }
    strided_index
}

fn compute_default_strides(shape: &[usize], element_size: usize) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = element_size;

    for &dim in shape.iter().rev() {
        strides.push(stride);
        stride *= dim;
    }

    strides.reverse();
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_tensor_view_creation() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let view = TensorView::from_tensor(&tensor).unwrap();

        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.numel(), 4);
        assert!(view.is_contiguous());
    }

    #[test]
    fn test_zero_copy_transpose() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let view = TensorView::from_tensor(&tensor).unwrap();

        let transposed = view.transpose(&[1, 0]).unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.strides(), &[4, 12]); // Strides change for transpose
    }

    #[test]
    fn test_zero_copy_reshape() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let view = TensorView::from_tensor(&tensor).unwrap();

        let reshaped = view.reshape(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert!(reshaped.is_contiguous());
    }

    #[test]
    fn test_zero_copy_slice() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let view = TensorView::from_tensor(&tensor).unwrap();

        let sliced = view.slice(&[(0, 1), (1, 3)]).unwrap();
        assert_eq!(sliced.shape(), &[1, 2]);
    }

    #[test]
    fn test_memory_stats() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let view = TensorView::from_tensor(&tensor).unwrap();

        let stats = TensorViewOps::memory_stats(&view);
        assert_eq!(stats.total_elements, 4);
        assert_eq!(stats.size_bytes, 16); // 4 * 4 bytes
        assert!(stats.is_contiguous);
    }

    #[test]
    fn test_shares_memory() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let view1 = TensorView::from_tensor(&tensor).unwrap();
        let view2 = view1.transpose(&[1, 0]).unwrap();

        assert!(TensorViewOps::shares_memory(&view1, &view2));
    }

    #[test]
    fn test_flat_to_multi_index() {
        let shape = vec![2, 3];
        assert_eq!(flat_to_multi_index(0, &shape), vec![0, 0]);
        assert_eq!(flat_to_multi_index(1, &shape), vec![0, 1]);
        assert_eq!(flat_to_multi_index(3, &shape), vec![1, 0]);
        assert_eq!(flat_to_multi_index(5, &shape), vec![1, 2]);
    }

    #[test]
    fn test_multi_to_strided_index() {
        let strides = vec![12, 4];
        let offset = 0;
        assert_eq!(multi_to_strided_index(&[0, 0], &strides, offset), 0);
        assert_eq!(multi_to_strided_index(&[0, 1], &strides, offset), 4);
        assert_eq!(multi_to_strided_index(&[1, 0], &strides, offset), 12);
        assert_eq!(multi_to_strided_index(&[1, 2], &strides, offset), 20);
    }
}
