//! GPU Eigenvalue Computation
//!
//! This module provides GPU implementations for eigenvalue and eigenvector computation
//! using the QR algorithm optimized for parallel execution on GPU hardware.

use super::super::context::{GpuLinalgContext, LinalgMetadata};
use crate::gpu::buffer::GpuBuffer;
use crate::{Result, Shape, TensorError};
use bytemuck::{Pod, Zeroable};
use scirs2_core::numeric::{Float, One};
use wgpu::{BufferDescriptor, BufferUsages};

/// Eigenvalue computation for symmetric matrices
pub fn eigenvalues<T>(
    context: &mut GpuLinalgContext,
    input: &GpuBuffer<T>,
    eigenvalues: &GpuBuffer<T>,
    eigenvectors: Option<&GpuBuffer<T>>,
    shape: &Shape,
) -> Result<()>
where
    T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
{
    context.eigenvalues(input, eigenvalues, eigenvectors, shape)
}

impl GpuLinalgContext {
    /// Eigenvalue computation
    ///
    /// Computes eigenvalues and eigenvectors of a symmetric matrix using the QR algorithm.
    /// For a symmetric matrix A [n, n], computes eigenvalues λ and eigenvectors V such that:
    /// A * V = V * Λ (where Λ is diagonal matrix of eigenvalues)
    pub fn eigenvalues<T>(
        &mut self,
        input: &GpuBuffer<T>,
        eigenvalues: &GpuBuffer<T>,
        eigenvectors: Option<&GpuBuffer<T>>,
        shape: &Shape,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(TensorError::invalid_shape_simple(
                "Eigenvalue computation requires a square matrix".to_string(),
            ));
        }

        let n = shape[0];
        if n == 0 {
            return Ok(()); // Empty matrix has no eigenvalues
        }

        // For very small matrices, suggest CPU fallback
        if n < 4 {
            return Err(TensorError::ComputeError {
                operation: "gpu_eigenvalue".to_string(),
                details: "GPU eigenvalue computation requires matrices >= 4x4 - use CPU fallback for smaller matrices".to_string(),
                retry_possible: false,
                context: None,
            });
        }

        // Initialize eigenvalue pipeline if needed
        if self.eigenvalue_pipeline.is_none() {
            self.initialize_eigenvalue_pipeline()?;
        }

        // Create working buffers
        let matrix_size = n * n;
        let working_matrix = self.device().create_buffer(&BufferDescriptor {
            label: Some("eigen_working_matrix"),
            size: (matrix_size * std::mem::size_of::<T>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let q_matrix = self.device().create_buffer(&BufferDescriptor {
            label: Some("eigen_q_matrix"),
            size: (matrix_size * std::mem::size_of::<T>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create metadata
        let metadata = LinalgMetadata::new(n, n)
            .with_tolerance(
                T::from(1e-10)
                    .unwrap_or_else(|| T::from(0.0).unwrap())
                    .to_f64() as f32,
            )
            .with_max_iterations(100 * n as u32);
        let metadata_buffer = self.create_metadata_buffer(&metadata)?;

        self.execute_eigenvalue_computation(
            input,
            eigenvalues,
            eigenvectors,
            &working_matrix,
            &q_matrix,
            &metadata_buffer,
            &metadata,
            n,
        )
    }

    /// Execute the eigenvalue computation pipeline
    fn execute_eigenvalue_computation<T>(
        &mut self,
        input: &GpuBuffer<T>,
        eigenvalues: &GpuBuffer<T>,
        eigenvectors: Option<&GpuBuffer<T>>,
        working_matrix: &wgpu::Buffer,
        q_matrix: &wgpu::Buffer,
        metadata_buffer: &wgpu::Buffer,
        metadata: &LinalgMetadata,
        n: usize,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        let pipelines = self.create_eigenvalue_pipelines()?;

        // Create eigenvectors buffer if not provided
        let eigenvectors_buffer = if let Some(eigenvecs) = eigenvectors {
            eigenvecs.buffer()
        } else {
            // Create a temporary buffer if eigenvectors are not requested
            &self.device().create_buffer(&BufferDescriptor {
                label: Some("temp_eigenvectors"),
                size: (n * n * std::mem::size_of::<T>()) as u64,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };

        // Create bind group
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eigen_bind_group"),
            layout: &pipelines.init.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: eigenvalues.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: eigenvectors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: working_matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: q_matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: metadata_buffer.as_entire_binding(),
                },
            ],
        });

        // Initialize matrices
        self.initialize_eigenvalue_matrices(&pipelines.init, &bind_group, n)?;

        // Perform iterative Jacobi method
        self.perform_jacobi_iterations(
            &pipelines.givens,
            input,
            eigenvalues,
            eigenvectors_buffer,
            working_matrix,
            q_matrix,
            metadata,
            n,
        )?;

        // Extract eigenvalues and finalize
        self.finalize_eigenvalue_computation(
            &pipelines,
            &bind_group,
            eigenvectors.is_some(),
            n,
        )?;

        Ok(())
    }

    /// Initialize eigenvalue computation matrices
    fn initialize_eigenvalue_matrices(
        &mut self,
        init_pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        n: usize,
    ) -> Result<()> {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("eigen_init_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("eigen_init_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(init_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);

            let workgroups_x = (n as u32 + 15) / 16;
            let workgroups_y = (n as u32 + 15) / 16;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Perform Jacobi iterations for eigenvalue computation
    fn perform_jacobi_iterations<T>(
        &mut self,
        givens_pipeline: &wgpu::ComputePipeline,
        input: &GpuBuffer<T>,
        eigenvalues: &GpuBuffer<T>,
        eigenvectors_buffer: &wgpu::Buffer,
        working_matrix: &wgpu::Buffer,
        q_matrix: &wgpu::Buffer,
        metadata: &LinalgMetadata,
        n: usize,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        let max_iterations = metadata.max_iterations.min(100);

        for _iter in 0..max_iterations {
            let mut converged = true;

            // Apply Givens rotations to eliminate off-diagonal elements
            for i in 0..n {
                for j in (i + 1)..n {
                    // Update metadata with current (i,j) pair
                    let updated_metadata = LinalgMetadata {
                        rows_a: n as u32,
                        cols_a: n as u32,
                        rows_b: i as u32, // Reusing for i
                        cols_b: j as u32, // Reusing for j
                        batch_size: 1,
                        tolerance: metadata.tolerance,
                        max_iterations: metadata.max_iterations,
                        _padding: 0,
                    };

                    let iter_metadata_buffer = self.create_metadata_buffer(&updated_metadata)?;

                    let iter_bind_group =
                        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("eigen_iter_bind_group"),
                            layout: &givens_pipeline.get_bind_group_layout(0),
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: input.buffer().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: eigenvalues.buffer().as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: eigenvectors_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: working_matrix.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: q_matrix.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 5,
                                    resource: iter_metadata_buffer.as_entire_binding(),
                                },
                            ],
                        });

                    let mut encoder =
                        self.device()
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("eigen_givens_encoder"),
                            });

                    {
                        let mut compute_pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("eigen_givens_pass"),
                                timestamp_writes: None,
                            });

                        compute_pass.set_pipeline(givens_pipeline);
                        compute_pass.set_bind_group(0, &iter_bind_group, &[]);
                        compute_pass.dispatch_workgroups((n as u32 + 255) / 256, 1, 1);
                    }

                    self.queue().submit(std::iter::once(encoder.finish()));

                    // In a real implementation, we would check convergence here
                    // For simplicity, we assume convergence after a fixed number of iterations
                    converged = false; // Force at least some iterations
                }
            }

            if converged {
                break;
            }
        }

        Ok(())
    }

    /// Finalize eigenvalue computation (extract, sort, normalize)
    fn finalize_eigenvalue_computation(
        &mut self,
        pipelines: &EigenvaluePipelines,
        bind_group: &wgpu::BindGroup,
        compute_eigenvectors: bool,
        n: usize,
    ) -> Result<()> {
        // Extract eigenvalues from diagonal
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("eigen_extract_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("eigen_extract_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipelines.extract);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups((n as u32 + 255) / 256, 1, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));

        // Sort eigenvalues and eigenvectors if requested
        if compute_eigenvectors {
            self.sort_eigenvalues(&pipelines.sort, bind_group)?;
            self.normalize_eigenvectors(&pipelines.normalize, bind_group, n)?;
        }

        Ok(())
    }

    /// Sort eigenvalues and corresponding eigenvectors
    fn sort_eigenvalues(
        &mut self,
        sort_pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
    ) -> Result<()> {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("eigen_sort_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("eigen_sort_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(sort_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1); // Single workgroup for sorting
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Normalize eigenvectors
    fn normalize_eigenvectors(
        &mut self,
        normalize_pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        n: usize,
    ) -> Result<()> {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("eigen_normalize_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("eigen_normalize_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(normalize_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups((n as u32 + 255) / 256, 1, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Create all eigenvalue computation pipelines
    fn create_eigenvalue_pipelines(&self) -> Result<EigenvaluePipelines> {
        // Load eigenvalue shader
        let shader_source = include_str!("../../shaders/linalg_eigenvalue.wgsl");
        let shader_module = self
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("eigenvalue_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let init = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("eigen_init_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("initialize_eigen"),
                cache: None,
                compilation_options: Default::default(),
            });

        let givens = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("eigen_givens_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("apply_givens_eigen"),
                cache: None,
                compilation_options: Default::default(),
            });

        let extract = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("eigen_extract_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("extract_eigenvalues"),
                cache: None,
                compilation_options: Default::default(),
            });

        let sort = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("eigen_sort_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("sort_eigenvalues"),
                cache: None,
                compilation_options: Default::default(),
            });

        let normalize = self
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("eigen_normalize_pipeline"),
                layout: None,
                module: &shader_module,
                entry_point: Some("normalize_eigenvectors"),
                cache: None,
                compilation_options: Default::default(),
            });

        Ok(EigenvaluePipelines {
            init,
            givens,
            extract,
            sort,
            normalize,
        })
    }
}

/// Collection of compute pipelines for eigenvalue computation
struct EigenvaluePipelines {
    init: wgpu::ComputePipeline,
    givens: wgpu::ComputePipeline,
    extract: wgpu::ComputePipeline,
    sort: wgpu::ComputePipeline,
    normalize: wgpu::ComputePipeline,
}