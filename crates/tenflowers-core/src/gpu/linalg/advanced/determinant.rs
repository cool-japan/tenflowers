//! GPU Matrix Determinant Computation
//!
//! This module provides GPU implementations for computing matrix determinants using
//! Gaussian elimination with partial pivoting, optimized for parallel execution.

use super::super::context::{GpuLinalgContext, LinalgMetadata};
use crate::gpu::buffer::GpuBuffer;
use crate::{Result, Shape, TensorError};
use bytemuck::{Pod, Zeroable};
use scirs2_core::numeric::{Float, One};
use wgpu::{BufferDescriptor, BufferUsages};

/// Matrix determinant computation using Gaussian elimination
pub fn determinant<T>(
    context: &mut GpuLinalgContext,
    input: &GpuBuffer<T>,
    shape: &Shape,
) -> Result<T>
where
    T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
{
    context.determinant(input, shape)
}

impl GpuLinalgContext {
    /// Matrix determinant computation
    ///
    /// Computes the determinant of a square matrix using Gaussian elimination
    /// with partial pivoting. This implementation uses multiple kernel dispatches
    /// to handle the sequential nature of the elimination process.
    pub fn determinant<T>(&mut self, input: &GpuBuffer<T>, shape: &Shape) -> Result<T>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        // Validate input
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(TensorError::invalid_shape_simple(
                "Determinant requires a square matrix".to_string(),
            ));
        }

        let n = shape[0];
        if n == 0 {
            return Ok(T::one());
        }

        let buffers = self.create_determinant_buffers::<T>(n)?;
        self.initialize_determinant_computation(input, &buffers, n)?;

        let pipelines = self.create_determinant_pipelines()?;
        let metadata = self.create_determinant_metadata(n);
        let metadata_buffer = self.create_metadata_buffer(&metadata)?;

        // Perform Gaussian elimination for each column
        self.perform_gaussian_elimination(
            &pipelines,
            &buffers,
            &metadata_buffer,
            n,
        )?;

        // Compute final determinant from diagonal elements
        self.compute_final_determinant(&pipelines.compute_det, &buffers, &metadata_buffer)?;

        // Read back the result
        self.read_determinant_result::<T>(&buffers.determinant_buffer)
    }

    /// Create working buffers for determinant computation
    fn create_determinant_buffers<T>(&self, n: usize) -> Result<DeterminantBuffers>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        let matrix_size = n * n;

        let working_matrix = self.device().create_buffer(&BufferDescriptor {
            label: Some("det_working_matrix"),
            size: (matrix_size * std::mem::size_of::<T>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let determinant_buffer = self.device().create_buffer(&BufferDescriptor {
            label: Some("det_result"),
            size: std::mem::size_of::<T>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pivot_info_buffer = self.device().create_buffer(&BufferDescriptor {
            label: Some("pivot_info"),
            size: (2 * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(DeterminantBuffers {
            working_matrix,
            determinant_buffer,
            pivot_info_buffer,
        })
    }

    /// Initialize determinant computation by copying input and setting initial determinant
    fn initialize_determinant_computation<T>(
        &mut self,
        input: &GpuBuffer<T>,
        buffers: &DeterminantBuffers,
        n: usize,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        let matrix_size = n * n;

        // Copy input matrix to working matrix
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("det_copy_encoder"),
            });

        encoder.copy_buffer_to_buffer(
            input.buffer(),
            0,
            &buffers.working_matrix,
            0,
            (matrix_size * std::mem::size_of::<T>()) as u64,
        );

        self.queue().submit(std::iter::once(encoder.finish()));

        // Initialize determinant to 1.0
        let initial_det = T::one();
        let det_bytes = bytemuck::bytes_of(&initial_det);
        self.queue().write_buffer(&buffers.determinant_buffer, 0, det_bytes);

        Ok(())
    }

    /// Create compute pipelines for determinant computation
    fn create_determinant_pipelines(&self) -> Result<DeterminantPipelines> {
        let shader_source = include_str!("../../shaders/linalg_determinant.wgsl");
        let shader_module = self
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("determinant_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let find_pivot = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("find_pivot_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("find_pivot"),
                cache: None,
                compilation_options: Default::default(),
            });

        let swap_rows = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("swap_rows_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("swap_rows"),
                cache: None,
                compilation_options: Default::default(),
            });

        let elimination = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("elimination_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("elimination_step"),
                cache: None,
                compilation_options: Default::default(),
            });

        let compute_det = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("compute_det_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("compute_determinant"),
                cache: None,
                compilation_options: Default::default(),
            });

        Ok(DeterminantPipelines {
            find_pivot,
            swap_rows,
            elimination,
            compute_det,
        })
    }

    /// Create metadata for determinant computation
    fn create_determinant_metadata(&self, n: usize) -> LinalgMetadata {
        LinalgMetadata::new(n, n).with_tolerance(1e-10)
    }

    /// Perform Gaussian elimination for each column
    fn perform_gaussian_elimination(
        &mut self,
        pipelines: &DeterminantPipelines,
        buffers: &DeterminantBuffers,
        metadata_buffer: &wgpu::Buffer,
        n: usize,
    ) -> Result<()> {
        for k in 0..n {
            // Set current column index
            let k_value = k as u32;
            let k_bytes = bytemuck::bytes_of(&k_value);
            self.queue().write_buffer(&buffers.pivot_info_buffer, 0, k_bytes);

            // Create bind group for this iteration
            let bind_group = self.create_elimination_bind_group(
                &pipelines.find_pivot,
                buffers,
                metadata_buffer,
            )?;

            // Execute elimination steps
            self.execute_elimination_iteration(pipelines, &bind_group, n)?;
        }

        Ok(())
    }

    /// Create bind group for elimination operations
    fn create_elimination_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &DeterminantBuffers,
        metadata_buffer: &wgpu::Buffer,
    ) -> Result<wgpu::BindGroup> {
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("det_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.working_matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.determinant_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.pivot_info_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: metadata_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(bind_group)
    }

    /// Execute one iteration of the elimination process
    fn execute_elimination_iteration(
        &mut self,
        pipelines: &DeterminantPipelines,
        bind_group: &wgpu::BindGroup,
        n: usize,
    ) -> Result<()> {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("det_elimination_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("det_elimination_pass"),
                timestamp_writes: None,
            });

            // Find pivot
            compute_pass.set_pipeline(&pipelines.find_pivot);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);

            // Swap rows if needed
            compute_pass.set_pipeline(&pipelines.swap_rows);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups((n as u32 + 255) / 256, 1, 1);

            // Perform elimination
            compute_pass.set_pipeline(&pipelines.elimination);
            compute_pass.set_bind_group(0, bind_group, &[]);
            let workgroups_x = (n as u32 + 15) / 16;
            let workgroups_y = (n as u32 + 15) / 16;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Compute final determinant from diagonal elements
    fn compute_final_determinant(
        &mut self,
        compute_det_pipeline: &wgpu::ComputePipeline,
        buffers: &DeterminantBuffers,
        metadata_buffer: &wgpu::Buffer,
    ) -> Result<()> {
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("det_final_bind_group"),
            layout: &compute_det_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.working_matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.determinant_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.pivot_info_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: metadata_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("det_final_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("det_final_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(compute_det_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Read back the determinant result from GPU
    fn read_determinant_result<T>(&mut self, determinant_buffer: &wgpu::Buffer) -> Result<T>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        // Create readback buffer
        let result_buffer = self.device().create_buffer(&BufferDescriptor {
            label: Some("det_result_readback"),
            size: std::mem::size_of::<T>() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy determinant result to readback buffer
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("det_readback_encoder"),
            });

        encoder.copy_buffer_to_buffer(
            determinant_buffer,
            0,
            &result_buffer,
            0,
            std::mem::size_of::<T>() as u64,
        );

        self.queue().submit(std::iter::once(encoder.finish()));

        // Map and read the result
        let buffer_slice = result_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device().poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().map_err(|e| TensorError::ComputeError {
            operation: "gpu_read_determinant".to_string(),
            details: format!("Failed to read determinant result: {:?}", e),
            retry_possible: true,
            context: None,
        })?;

        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::from_bytes::<T>(&data[..std::mem::size_of::<T>()]);
        let determinant_value = *result;

        drop(data);
        result_buffer.unmap();

        Ok(determinant_value)
    }
}

/// Working buffers for determinant computation
struct DeterminantBuffers {
    working_matrix: wgpu::Buffer,
    determinant_buffer: wgpu::Buffer,
    pivot_info_buffer: wgpu::Buffer,
}

/// Collection of compute pipelines for determinant computation
struct DeterminantPipelines {
    find_pivot: wgpu::ComputePipeline,
    swap_rows: wgpu::ComputePipeline,
    elimination: wgpu::ComputePipeline,
    compute_det: wgpu::ComputePipeline,
}