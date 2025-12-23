//! Basic GPU Linear Algebra Operations
//!
//! This module provides fundamental GPU operations like matrix transpose and
//! multiplication that form the building blocks for more complex linear algebra
//! algorithms. These operations are optimized for GPU execution patterns.

use super::context::{GpuLinalgContext, LinalgMetadata};
use crate::gpu::buffer::GpuBuffer;
use crate::{Result, Shape, TensorError};
use bytemuck::{Pod, Zeroable};
use scirs2_core::numeric::Float;

/// Matrix transpose operation
pub fn transpose<T>(
    context: &mut GpuLinalgContext,
    input: &GpuBuffer<T>,
    output: &GpuBuffer<T>,
    shape: &Shape,
) -> Result<()>
where
    T: Float + Pod + Zeroable + Send + Sync + 'static,
{
    context.transpose(input, output, shape)
}

/// Matrix multiplication optimized for linear algebra operations
pub fn matmul_linalg<T>(
    context: &mut GpuLinalgContext,
    a: &GpuBuffer<T>,
    b: &GpuBuffer<T>,
    c: &GpuBuffer<T>,
    shape_a: &Shape,
    shape_b: &Shape,
) -> Result<()>
where
    T: Float + Pod + Zeroable + Send + Sync + 'static,
{
    context.matmul_linalg(a, b, c, shape_a, shape_b)
}

/// Adaptive matrix multiplication that selects optimal parameters based on matrix dimensions
pub fn adaptive_matmul_linalg<T>(
    context: &mut GpuLinalgContext,
    a: &GpuBuffer<T>,
    b: &GpuBuffer<T>,
    c: &GpuBuffer<T>,
    shape_a: &Shape,
    shape_b: &Shape,
) -> Result<()>
where
    T: Float + Pod + Zeroable + Send + Sync + 'static,
{
    context.adaptive_matmul_linalg(a, b, c, shape_a, shape_b)
}

impl GpuLinalgContext {
    /// Matrix transpose operation
    pub fn transpose<T>(
        &mut self,
        input: &GpuBuffer<T>,
        output: &GpuBuffer<T>,
        shape: &Shape,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        // Ensure pipeline is initialized
        if self.transpose_pipeline.is_none() {
            self.initialize_pipelines()?;
        }

        let pipeline = self.transpose_pipeline.as_ref().unwrap();

        // Create metadata
        let metadata = LinalgMetadata::new(shape[0], shape[1]);
        let metadata_buffer = self.create_metadata_buffer(&metadata)?;

        // Create bind group
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transpose_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: metadata_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("transpose_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("transpose_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with workgroup size optimization
            let workgroup_size = 16; // 16x16 workgroups for 2D operations
            let workgroups_x = (shape[1] + workgroup_size - 1) / workgroup_size;
            let workgroups_y = (shape[0] + workgroup_size - 1) / workgroup_size;

            compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Matrix multiplication optimized for linear algebra operations
    pub fn matmul_linalg<T>(
        &mut self,
        a: &GpuBuffer<T>,
        b: &GpuBuffer<T>,
        c: &GpuBuffer<T>,
        shape_a: &Shape,
        shape_b: &Shape,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        // Ensure pipeline is initialized
        if self.matmul_linalg_pipeline.is_none() {
            self.initialize_pipelines()?;
        }

        let pipeline = self.matmul_linalg_pipeline.as_ref().unwrap();

        // Validate matrix dimensions
        if shape_a.len() != 2 || shape_b.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        if shape_a[1] != shape_b[0] {
            return Err(TensorError::ShapeMismatch {
                operation: "matmul_linalg".to_string(),
                expected: format!(
                    "inner dimensions to match (got {} vs {})",
                    shape_a[1], shape_b[0]
                ),
                got: format!("shapes {:?} and {:?}", shape_a, shape_b),
                context: None,
            });
        }

        // Create metadata
        let metadata =
            LinalgMetadata::new_two_matrices(shape_a[0], shape_a[1], shape_b[0], shape_b[1]);
        let metadata_buffer = self.create_metadata_buffer(&metadata)?;

        // Create bind group
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_linalg_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: metadata_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_linalg_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_linalg_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with tiled workgroups for optimal memory access
            let tile_size = 16;
            let workgroups_x = (shape_b[1] + tile_size - 1) / tile_size;
            let workgroups_y = (shape_a[0] + tile_size - 1) / tile_size;

            compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Adaptive matrix multiplication that selects optimal parameters based on matrix dimensions
    pub fn adaptive_matmul_linalg<T>(
        &mut self,
        a: &GpuBuffer<T>,
        b: &GpuBuffer<T>,
        c: &GpuBuffer<T>,
        shape_a: &Shape,
        shape_b: &Shape,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        // Validate matrix dimensions
        if shape_a.len() != 2 || shape_b.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        if shape_a[1] != shape_b[0] {
            return Err(TensorError::ShapeMismatch {
                operation: "adaptive_matmul_linalg".to_string(),
                expected: format!(
                    "inner dimensions to match (got {} vs {})",
                    shape_a[1], shape_b[0]
                ),
                got: format!("shapes {:?} and {:?}", shape_a, shape_b),
                context: None,
            });
        }

        let m = shape_a[0];
        let k = shape_a[1];
        let n = shape_b[1];

        // Select optimal parameters using adaptive configuration
        let tile_size = self.adaptive_gemm_config().select_tile_size(m, n, k);
        let (workgroup_x, workgroup_y) = self.adaptive_gemm_config().select_workgroup_size(m, n, k);

        // Estimate performance characteristics
        let bandwidth_utilization = self
            .adaptive_gemm_config()
            .estimate_bandwidth_utilization(m, n, k);

        // For very high bandwidth utilization, prefer shared memory optimization
        if bandwidth_utilization > 0.8 && self.adaptive_gemm_config().prefer_shared_memory {
            self.matmul_with_shared_memory_optimization(a, b, c, shape_a, shape_b, tile_size)
        } else {
            self.matmul_with_memory_coalescing_optimization(
                a,
                b,
                c,
                shape_a,
                shape_b,
                workgroup_x,
                workgroup_y,
            )
        }
    }

    /// Matrix multiplication with shared memory optimization for high data reuse scenarios
    fn matmul_with_shared_memory_optimization<T>(
        &mut self,
        a: &GpuBuffer<T>,
        b: &GpuBuffer<T>,
        c: &GpuBuffer<T>,
        shape_a: &Shape,
        shape_b: &Shape,
        tile_size: u32,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        // Ensure pipeline is initialized
        if self.matmul_linalg_pipeline.is_none() {
            self.initialize_pipelines()?;
        }

        let pipeline = self.matmul_linalg_pipeline.as_ref().unwrap();

        // Create metadata with tile size hint
        let mut metadata =
            LinalgMetadata::new_two_matrices(shape_a[0], shape_a[1], shape_b[0], shape_b[1]);
        // Store tile size in max_iterations field as a hint to the shader
        metadata.max_iterations = tile_size;
        let metadata_buffer = self.create_metadata_buffer(&metadata)?;

        // Create bind group
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adaptive_matmul_shared_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: metadata_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute with optimal tile size
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("adaptive_matmul_shared_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("adaptive_matmul_shared_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Use the adaptive tile size for workgroup dispatch
            let workgroups_x = (shape_b[1] + tile_size as usize - 1) / tile_size as usize;
            let workgroups_y = (shape_a[0] + tile_size as usize - 1) / tile_size as usize;

            compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Matrix multiplication with memory coalescing optimization for bandwidth-bound scenarios
    fn matmul_with_memory_coalescing_optimization<T>(
        &mut self,
        a: &GpuBuffer<T>,
        b: &GpuBuffer<T>,
        c: &GpuBuffer<T>,
        shape_a: &Shape,
        shape_b: &Shape,
        workgroup_x: u32,
        workgroup_y: u32,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        // Ensure pipeline is initialized
        if self.matmul_linalg_pipeline.is_none() {
            self.initialize_pipelines()?;
        }

        let pipeline = self.matmul_linalg_pipeline.as_ref().unwrap();

        // Create metadata
        let metadata =
            LinalgMetadata::new_two_matrices(shape_a[0], shape_a[1], shape_b[0], shape_b[1]);
        let metadata_buffer = self.create_metadata_buffer(&metadata)?;

        // Create bind group
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adaptive_matmul_coalescing_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: metadata_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute with optimal workgroup sizes for memory coalescing
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("adaptive_matmul_coalescing_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("adaptive_matmul_coalescing_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Use adaptive workgroup sizes optimized for memory access patterns
            let workgroups_x = (shape_b[1] + workgroup_x as usize - 1) / workgroup_x as usize;
            let workgroups_y = (shape_a[0] + workgroup_y as usize - 1) / workgroup_y as usize;

            compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}
