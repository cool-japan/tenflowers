//! GPU Matrix Inversion
//!
//! This module provides GPU implementations for matrix inversion using Gauss-Jordan elimination
//! with partial pivoting, optimized for parallel execution on GPU hardware.

use super::super::context::{GpuLinalgContext, LinalgMetadata};
use crate::gpu::buffer::GpuBuffer;
use crate::{Result, Shape, TensorError};
use bytemuck::{Pod, Zeroable};
use scirs2_core::numeric::{Float, One};
use wgpu::{BufferDescriptor, BufferUsages};

/// Matrix inversion using Gauss-Jordan elimination
pub fn inverse<T>(
    context: &mut GpuLinalgContext,
    input: &GpuBuffer<T>,
    output: &GpuBuffer<T>,
    shape: &Shape,
) -> Result<()>
where
    T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
{
    context.inverse(input, output, shape)
}

impl GpuLinalgContext {
    /// Matrix inversion
    ///
    /// Computes the inverse of a square matrix using Gauss-Jordan elimination
    /// with partial pivoting. Uses an augmented matrix [A|I] approach.
    pub fn inverse<T>(
        &mut self,
        input: &GpuBuffer<T>,
        output: &GpuBuffer<T>,
        shape: &Shape,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        // Validate input
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(TensorError::invalid_shape_simple(
                "Matrix inverse requires a square matrix".to_string(),
            ));
        }

        let n = shape[0];
        if n == 0 {
            return Err(TensorError::invalid_shape_simple(
                "Cannot invert empty matrix".to_string(),
            ));
        }

        let buffers = self.create_inverse_buffers::<T>(n)?;
        self.copy_input_to_augmented_matrix(input, &buffers.augmented_matrix, n)?;

        let pipelines = self.create_inverse_pipelines()?;
        let metadata = self.create_inverse_metadata(n);
        let metadata_buffer = self.create_metadata_buffer(&metadata)?;

        let bind_group = self.create_inverse_bind_group(
            &pipelines.init,
            &buffers,
            &metadata_buffer,
        )?;

        // Initialize augmented matrix [A|I]
        self.initialize_augmented_matrix(&pipelines.init, &bind_group, n)?;

        // Perform Gauss-Jordan elimination
        self.perform_gauss_jordan_elimination(&pipelines, &bind_group, &buffers.pivot_info_buffer, n)?;

        // Extract inverse matrix from right half
        self.extract_inverse_matrix(&pipelines.extract, &bind_group, n)?;

        // Copy result to output buffer
        self.copy_inverse_to_output(&buffers.inverse_matrix, output, n)?;

        // Verify that matrix is not singular
        self.verify_inverse_validity(&buffers.status_buffer)?;

        Ok(())
    }

    /// Create working buffers for matrix inversion
    fn create_inverse_buffers<T>(&self, n: usize) -> Result<InverseBuffers>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        let matrix_size = n * n;
        let augmented_size = n * 2 * n; // [A|I] matrix

        let augmented_matrix = self.device().create_buffer(&BufferDescriptor {
            label: Some("inv_augmented_matrix"),
            size: (augmented_size * std::mem::size_of::<T>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let inverse_matrix = self.device().create_buffer(&BufferDescriptor {
            label: Some("inv_result"),
            size: (matrix_size * std::mem::size_of::<T>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pivot_info_buffer = self.device().create_buffer(&BufferDescriptor {
            label: Some("inv_pivot_info"),
            size: (2 * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let status_buffer = self.device().create_buffer(&BufferDescriptor {
            label: Some("inv_status"),
            size: std::mem::size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Ok(InverseBuffers {
            augmented_matrix,
            inverse_matrix,
            pivot_info_buffer,
            status_buffer,
        })
    }

    /// Copy input matrix to the left half of augmented matrix
    fn copy_input_to_augmented_matrix<T>(
        &mut self,
        input: &GpuBuffer<T>,
        augmented_matrix: &wgpu::Buffer,
        n: usize,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        let matrix_size = n * n;
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("inv_copy_encoder"),
            });

        encoder.copy_buffer_to_buffer(
            input.buffer(),
            0,
            augmented_matrix,
            0,
            (matrix_size * std::mem::size_of::<T>()) as u64,
        );

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Create compute pipelines for matrix inversion
    fn create_inverse_pipelines(&self) -> Result<InversePipelines> {
        let shader_source = include_str!("../../shaders/linalg_inverse.wgsl");
        let shader_module = self
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("inverse_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let init = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("init_augmented_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("initialize_augmented"),
                cache: None,
                compilation_options: Default::default(),
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

        let scale_pivot = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("scale_pivot_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("scale_pivot_row"),
                cache: None,
                compilation_options: Default::default(),
            });

        let eliminate = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("eliminate_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("eliminate_column"),
                cache: None,
                compilation_options: Default::default(),
            });

        let extract = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("extract_inverse_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("extract_inverse"),
                cache: None,
                compilation_options: Default::default(),
            });

        Ok(InversePipelines {
            init,
            find_pivot,
            swap_rows,
            scale_pivot,
            eliminate,
            extract,
        })
    }

    /// Create metadata for matrix inversion
    fn create_inverse_metadata(&self, n: usize) -> LinalgMetadata {
        LinalgMetadata::new(n, n).with_tolerance(1e-10)
    }

    /// Create bind group for inverse operations
    fn create_inverse_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &InverseBuffers,
        metadata_buffer: &wgpu::Buffer,
    ) -> Result<wgpu::BindGroup> {
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inv_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.augmented_matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.inverse_matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.pivot_info_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.status_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: metadata_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(bind_group)
    }

    /// Initialize augmented matrix [A|I]
    fn initialize_augmented_matrix(
        &mut self,
        init_pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        n: usize,
    ) -> Result<()> {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("inv_init_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inv_init_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(init_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            let workgroups_x = (2 * n as u32 + 15) / 16;
            let workgroups_y = (n as u32 + 15) / 16;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Perform Gauss-Jordan elimination for each column
    fn perform_gauss_jordan_elimination(
        &mut self,
        pipelines: &InversePipelines,
        bind_group: &wgpu::BindGroup,
        pivot_info_buffer: &wgpu::Buffer,
        n: usize,
    ) -> Result<()> {
        for k in 0..n {
            // Set current column index
            let k_value = k as u32;
            let k_bytes = bytemuck::bytes_of(&k_value);
            self.queue().write_buffer(pivot_info_buffer, 0, k_bytes);

            // Execute the elimination steps
            self.execute_elimination_step(pipelines, bind_group, n)?;
        }

        Ok(())
    }

    /// Execute one step of elimination
    fn execute_elimination_step(
        &mut self,
        pipelines: &InversePipelines,
        bind_group: &wgpu::BindGroup,
        n: usize,
    ) -> Result<()> {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("inv_elimination_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inv_elimination_pass"),
                timestamp_writes: None,
            });

            // Find pivot
            compute_pass.set_pipeline(&pipelines.find_pivot);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);

            // Swap rows if needed
            compute_pass.set_pipeline(&pipelines.swap_rows);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups((2 * n as u32 + 255) / 256, 1, 1);

            // Scale pivot row
            compute_pass.set_pipeline(&pipelines.scale_pivot);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups((2 * n as u32 + 255) / 256, 1, 1);

            // Eliminate column
            compute_pass.set_pipeline(&pipelines.eliminate);
            compute_pass.set_bind_group(0, bind_group, &[]);
            let workgroups_x = (2 * n as u32 + 15) / 16;
            let workgroups_y = (n as u32 + 15) / 16;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Extract inverse matrix from right half of augmented matrix
    fn extract_inverse_matrix(
        &mut self,
        extract_pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        n: usize,
    ) -> Result<()> {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("inv_extract_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inv_extract_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(extract_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            let workgroups_x = (n as u32 + 15) / 16;
            let workgroups_y = (n as u32 + 15) / 16;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Copy inverse result to output buffer
    fn copy_inverse_to_output<T>(
        &mut self,
        inverse_matrix: &wgpu::Buffer,
        output: &GpuBuffer<T>,
        n: usize,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        let matrix_size = n * n;
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("inv_copy_result_encoder"),
            });

        encoder.copy_buffer_to_buffer(
            inverse_matrix,
            0,
            output.buffer(),
            0,
            (matrix_size * std::mem::size_of::<T>()) as u64,
        );

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Verify that the matrix is invertible (not singular)
    fn verify_inverse_validity(&mut self, status_buffer: &wgpu::Buffer) -> Result<()> {
        let status_readback = self.device().create_buffer(&BufferDescriptor {
            label: Some("inv_status_readback"),
            size: std::mem::size_of::<u32>() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("inv_status_encoder"),
            });

        encoder.copy_buffer_to_buffer(
            status_buffer,
            0,
            &status_readback,
            0,
            std::mem::size_of::<u32>() as u64,
        );

        self.queue().submit(std::iter::once(encoder.finish()));

        // Map and check status
        let buffer_slice = status_readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device().poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().map_err(|e| TensorError::ComputeError {
            operation: "gpu_read_status".to_string(),
            details: format!("Failed to read status: {:?}", e),
            retry_possible: true,
            context: None,
        })?;

        let data = buffer_slice.get_mapped_range();
        let status = bytemuck::from_bytes::<u32>(&data[..std::mem::size_of::<u32>()]);

        if *status != 0 {
            drop(data);
            status_readback.unmap();
            return Err(TensorError::unsupported_operation_simple(
                "Matrix is singular and cannot be inverted".to_string(),
            ));
        }

        drop(data);
        status_readback.unmap();
        Ok(())
    }
}

/// Working buffers for matrix inversion
struct InverseBuffers {
    augmented_matrix: wgpu::Buffer,
    inverse_matrix: wgpu::Buffer,
    pivot_info_buffer: wgpu::Buffer,
    status_buffer: wgpu::Buffer,
}

/// Collection of compute pipelines for matrix inversion
struct InversePipelines {
    init: wgpu::ComputePipeline,
    find_pivot: wgpu::ComputePipeline,
    swap_rows: wgpu::ComputePipeline,
    scale_pivot: wgpu::ComputePipeline,
    eliminate: wgpu::ComputePipeline,
    extract: wgpu::ComputePipeline,
}