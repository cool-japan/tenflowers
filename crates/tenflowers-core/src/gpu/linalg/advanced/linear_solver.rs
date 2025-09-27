//! GPU Linear System Solver
//!
//! This module provides GPU implementations for solving linear systems of equations
//! using LU factorization with partial pivoting, optimized for parallel execution.

use super::super::context::{GpuLinalgContext, LinalgMetadata};
use crate::gpu::buffer::GpuBuffer;
use crate::{Result, Shape, TensorError};
use bytemuck::{Pod, Zeroable};
use num_traits::{Float, One};
use wgpu::{BufferDescriptor, BufferUsages};

/// Linear system solver using LU factorization
pub fn solve<T>(
    context: &mut GpuLinalgContext,
    a: &GpuBuffer<T>,
    b: &GpuBuffer<T>,
    x: &GpuBuffer<T>,
    shape_a: &Shape,
    shape_b: &Shape,
) -> Result<()>
where
    T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
{
    context.solve(a, b, x, shape_a, shape_b)
}

impl GpuLinalgContext {
    /// Linear system solver
    ///
    /// Solves the linear system Ax = b using LU factorization with partial pivoting.
    /// For a matrix A [n, n] and vector b [n, 1], computes x such that A * x = b.
    /// Uses forward and backward substitution after LU decomposition.
    pub fn solve<T>(
        &mut self,
        a: &GpuBuffer<T>,
        b: &GpuBuffer<T>,
        x: &GpuBuffer<T>,
        shape_a: &Shape,
        shape_b: &Shape,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        // Validate input dimensions
        self.validate_solve_dimensions(shape_a, shape_b)?;

        let n = shape_a[0];
        let nrhs = shape_b[1]; // Number of right-hand sides

        if n == 0 {
            return Err(TensorError::invalid_shape_simple(
                "Cannot solve empty linear system".to_string(),
            ));
        }

        // For very small systems, suggest CPU fallback
        if n < 4 {
            return Err(TensorError::ComputeError {
                operation: "gpu_linear_solve".to_string(),
                details: "GPU linear solver requires matrices >= 4x4 - use CPU fallback for smaller systems".to_string(),
                retry_possible: false,
                context: None,
            });
        }

        // Initialize linear solve pipeline if needed
        if self.linear_solve_pipeline.is_none() {
            self.linear_solve_pipeline = Some(self.create_linear_solve_pipeline()?);
        }

        self.execute_linear_solve(a, b, x, n, nrhs)
    }

    /// Validate dimensions for linear system solving
    fn validate_solve_dimensions(&self, shape_a: &Shape, shape_b: &Shape) -> Result<()> {
        if shape_a.len() != 2 || shape_a[0] != shape_a[1] {
            return Err(TensorError::invalid_shape_simple(
                "Linear solver requires a square coefficient matrix".to_string(),
            ));
        }

        if shape_b.len() != 2 || shape_b[0] != shape_a[0] {
            return Err(TensorError::ShapeMismatch {
                operation: "linear_solve".to_string(),
                expected: format!(
                    "Right-hand side vector dimensions {} to match matrix size {}",
                    shape_b[0], shape_a[0]
                ),
                got: format!("shapes {:?} and {:?}", shape_a, shape_b),
                context: None,
            });
        }

        Ok(())
    }

    /// Execute the complete linear solve pipeline
    fn execute_linear_solve<T>(
        &mut self,
        a: &GpuBuffer<T>,
        b: &GpuBuffer<T>,
        x: &GpuBuffer<T>,
        n: usize,
        nrhs: usize,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        let buffers = self.create_solve_buffers::<T>(n, nrhs)?;

        // Copy right-hand side to working buffer
        self.copy_rhs_to_working_buffer(b, &buffers.working_b, n, nrhs)?;

        // Step 1: Perform LU decomposition
        let shape_a_obj = Shape::new(vec![n, n]);
        self.lu_decomposition(a, &buffers.l_matrix, &buffers.u_matrix, &buffers.p_matrix, &shape_a_obj)?;

        // Create metadata and pipelines
        let metadata = self.create_solve_metadata::<T>(n, nrhs);
        let metadata_buffer = self.create_metadata_buffer(&metadata)?;
        let pipelines = self.create_solve_pipelines()?;

        // Create bind group
        let bind_group = self.create_solve_bind_group(
            &pipelines.permute,
            &buffers,
            x,
            &metadata_buffer,
        )?;

        // Execute solve pipeline
        self.execute_solve_stages(&pipelines, &bind_group, &metadata, n, nrhs)?;

        // Verify solution (check for singularity)
        self.verify_solution(&buffers.status_buffer)?;

        Ok(())
    }

    /// Create working buffers for linear solve
    fn create_solve_buffers<T>(&self, n: usize, nrhs: usize) -> Result<SolveBuffers>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        let matrix_size = n * n;
        let rhs_size = n * nrhs;

        let l_matrix = self.device().create_buffer(&BufferDescriptor {
            label: Some("solve_l_matrix"),
            size: (matrix_size * std::mem::size_of::<T>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let u_matrix = self.device().create_buffer(&BufferDescriptor {
            label: Some("solve_u_matrix"),
            size: (matrix_size * std::mem::size_of::<T>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let p_matrix = self.device().create_buffer(&BufferDescriptor {
            label: Some("solve_p_matrix"),
            size: (matrix_size * std::mem::size_of::<T>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let working_b = self.device().create_buffer(&BufferDescriptor {
            label: Some("solve_working_b"),
            size: (rhs_size * std::mem::size_of::<T>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let intermediate_y = self.device().create_buffer(&BufferDescriptor {
            label: Some("solve_intermediate_y"),
            size: (rhs_size * std::mem::size_of::<T>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let status_buffer = self.device().create_buffer(&BufferDescriptor {
            label: Some("solve_status"),
            size: std::mem::size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Ok(SolveBuffers {
            l_matrix,
            u_matrix,
            p_matrix,
            working_b,
            intermediate_y,
            status_buffer,
        })
    }

    /// Copy right-hand side to working buffer
    fn copy_rhs_to_working_buffer<T>(
        &mut self,
        b: &GpuBuffer<T>,
        working_b: &wgpu::Buffer,
        n: usize,
        nrhs: usize,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        let rhs_size = n * nrhs;
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("solve_copy_encoder"),
            });

        encoder.copy_buffer_to_buffer(
            b.buffer(),
            0,
            working_b,
            0,
            (rhs_size * std::mem::size_of::<T>()) as u64,
        );

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Create metadata for linear solve
    fn create_solve_metadata<T>(&self, n: usize, nrhs: usize) -> LinalgMetadata
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        LinalgMetadata::new_two_matrices(n, n, n, nrhs).with_tolerance(
            T::from(1e-12)
                .unwrap_or_else(|| T::from(0.0).unwrap())
                .to_f64() as f32,
        )
    }

    /// Create solve pipeline stages
    fn create_solve_pipelines(&self) -> Result<SolvePipelines> {
        let shader_source = include_str!("../../shaders/linalg_solve.wgsl");
        let shader_module = self
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("solve_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let permute = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("solve_permute_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("apply_permutation"),
                cache: None,
                compilation_options: Default::default(),
            });

        let forward_subst = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("solve_forward_subst_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("forward_substitution"),
                cache: None,
                compilation_options: Default::default(),
            });

        let backward_subst = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("solve_backward_subst_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("backward_substitution"),
                cache: None,
                compilation_options: Default::default(),
            });

        let singularity_check = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("solve_singularity_check_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("check_singularity"),
                cache: None,
                compilation_options: Default::default(),
            });

        Ok(SolvePipelines {
            permute,
            forward_subst,
            backward_subst,
            singularity_check,
        })
    }

    /// Create bind group for solve operations
    fn create_solve_bind_group<T>(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &SolveBuffers,
        x: &GpuBuffer<T>,
        metadata_buffer: &wgpu::Buffer,
    ) -> Result<wgpu::BindGroup>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("solve_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.l_matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.u_matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.p_matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.working_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.intermediate_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: x.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.status_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: metadata_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(bind_group)
    }

    /// Execute all solve stages
    fn execute_solve_stages(
        &mut self,
        pipelines: &SolvePipelines,
        bind_group: &wgpu::BindGroup,
        metadata: &LinalgMetadata,
        n: usize,
        nrhs: usize,
    ) -> Result<()> {
        // Step 1: Check for singularity
        self.check_matrix_singularity(&pipelines.singularity_check, bind_group)?;

        // Step 2: Apply permutation to right-hand side (Pb)
        self.apply_permutation(&pipelines.permute, bind_group, n, nrhs)?;

        // Step 3: Forward substitution (solve Ly = Pb)
        self.forward_substitution(&pipelines.forward_subst, metadata, n, nrhs)?;

        // Step 4: Backward substitution (solve Ux = y)
        self.backward_substitution(&pipelines.backward_subst, metadata, n, nrhs)?;

        Ok(())
    }

    /// Check matrix for singularity
    fn check_matrix_singularity(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
    ) -> Result<()> {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("solve_singularity_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("solve_singularity_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Apply permutation matrix to right-hand side
    fn apply_permutation(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        n: usize,
        nrhs: usize,
    ) -> Result<()> {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("solve_permute_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("solve_permute_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups((nrhs as u32 + 255) / 256, (n as u32 + 255) / 256, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Perform forward substitution
    fn forward_substitution(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        metadata: &LinalgMetadata,
        n: usize,
        nrhs: usize,
    ) -> Result<()> {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("solve_forward_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("solve_forward_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);

            // Sequential solving for forward substitution
            for i in 0..n {
                let row_metadata = self.create_row_metadata(metadata, i, n, nrhs);
                let row_metadata_buffer = self.create_metadata_buffer(&row_metadata)?;
                let row_bind_group = self.create_row_bind_group(pipeline, &row_metadata_buffer)?;

                compute_pass.set_bind_group(0, &row_bind_group, &[]);
                compute_pass.dispatch_workgroups((nrhs as u32 + 255) / 256, 1, 1);
            }
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Perform backward substitution
    fn backward_substitution(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        metadata: &LinalgMetadata,
        n: usize,
        nrhs: usize,
    ) -> Result<()> {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("solve_backward_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("solve_backward_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);

            // Sequential solving for backward substitution (reverse order)
            for i in (0..n).rev() {
                let row_metadata = self.create_row_metadata(metadata, i, n, nrhs);
                let row_metadata_buffer = self.create_metadata_buffer(&row_metadata)?;
                let row_bind_group = self.create_row_bind_group(pipeline, &row_metadata_buffer)?;

                compute_pass.set_bind_group(0, &row_bind_group, &[]);
                compute_pass.dispatch_workgroups((nrhs as u32 + 255) / 256, 1, 1);
            }
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Create metadata for a specific row in substitution
    fn create_row_metadata(
        &self,
        base_metadata: &LinalgMetadata,
        row: usize,
        n: usize,
        nrhs: usize,
    ) -> LinalgMetadata {
        LinalgMetadata {
            rows_a: n as u32,
            cols_a: n as u32,
            rows_b: row as u32, // Current row
            cols_b: nrhs as u32,
            batch_size: 1,
            tolerance: base_metadata.tolerance,
            max_iterations: 1,
            _padding: 0,
        }
    }

    /// Verify that the solution is valid (check for singularity)
    fn verify_solution(&mut self, status_buffer: &wgpu::Buffer) -> Result<()> {
        let status_readback = self.device().create_buffer(&BufferDescriptor {
            label: Some("solve_status_readback"),
            size: std::mem::size_of::<u32>() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("solve_status_encoder"),
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
                "Matrix is singular - linear system has no unique solution".to_string(),
            ));
        }

        drop(data);
        status_readback.unmap();
        Ok(())
    }
}

/// Working buffers for linear system solving
struct SolveBuffers {
    l_matrix: wgpu::Buffer,
    u_matrix: wgpu::Buffer,
    p_matrix: wgpu::Buffer,
    working_b: wgpu::Buffer,
    intermediate_y: wgpu::Buffer,
    status_buffer: wgpu::Buffer,
}

/// Collection of compute pipelines for linear solving
struct SolvePipelines {
    permute: wgpu::ComputePipeline,
    forward_subst: wgpu::ComputePipeline,
    backward_subst: wgpu::ComputePipeline,
    singularity_check: wgpu::ComputePipeline,
}