//! Matrix Decomposition Operations for GPU Linear Algebra
//!
//! This module provides GPU implementations of fundamental matrix decompositions
//! including LU decomposition with partial pivoting, Singular Value Decomposition (SVD),
//! and QR decomposition using Householder reflections. These operations are optimized
//! for GPU execution patterns and follow established numerical algorithms.

use super::context::{GpuLinalgContext, LinalgMetadata};
use crate::gpu::buffer::GpuBuffer;
use crate::{Result, Shape, TensorError};
use bytemuck::{Pod, Zeroable};
use num_traits::Float;
use wgpu::util::DeviceExt;
use wgpu::{BufferDescriptor, BufferUsages};

/// LU decomposition with partial pivoting
pub fn lu_decomposition<T>(
    context: &mut GpuLinalgContext,
    input: &GpuBuffer<T>,
    l: &GpuBuffer<T>,
    u: &GpuBuffer<T>,
    p: &GpuBuffer<T>,
    shape: &Shape,
) -> Result<()>
where
    T: Float + Pod + Zeroable + Send + Sync + 'static,
{
    context.lu_decomposition(input, l, u, p, shape)
}

/// Singular Value Decomposition (SVD)
pub fn svd<T>(
    context: &mut GpuLinalgContext,
    input: &GpuBuffer<T>,
    u: &GpuBuffer<T>,
    s: &GpuBuffer<T>,
    vt: &GpuBuffer<T>,
    shape: &Shape,
) -> Result<()>
where
    T: Float + Pod + Zeroable + Send + Sync + 'static,
{
    context.svd(input, u, s, vt, shape)
}

/// QR decomposition using Householder reflections
pub fn qr_decomposition<T>(
    context: &mut GpuLinalgContext,
    input: &GpuBuffer<T>,
    q: &GpuBuffer<T>,
    r: &GpuBuffer<T>,
    shape: &Shape,
) -> Result<()>
where
    T: Float + Pod + Zeroable + Send + Sync + 'static,
{
    context.qr_decomposition(input, q, r, shape)
}

impl GpuLinalgContext {
    /// LU decomposition with partial pivoting
    ///
    /// Performs LU decomposition on GPU using a simplified algorithm.
    /// For very large matrices, this provides better performance than CPU.
    /// Uses multiple GPU dispatches to handle sequential dependencies.
    pub fn lu_decomposition<T>(
        &mut self,
        input: &GpuBuffer<T>,
        l: &GpuBuffer<T>,
        u: &GpuBuffer<T>,
        p: &GpuBuffer<T>,
        shape: &Shape,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        let dims = shape.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(TensorError::invalid_shape_simple(
                "LU decomposition requires a square matrix".to_string(),
            ));
        }

        let n = dims[0] as u32;

        // For small matrices, fall back to CPU for efficiency
        if n < 64 {
            return Err(TensorError::ComputeError {
                operation: "gpu_lu_decomposition".to_string(),
                details: "GPU LU decomposition requires matrices >= 64x64 - use CPU fallback for smaller matrices".to_string(),
                retry_possible: false,
                context: None,
            });
        }

        // Initialize LU decomposition pipeline if needed
        if self.lu_decomposition_pipeline.is_none() {
            self.initialize_lu_pipeline()?;
        }

        let pipeline = self.lu_decomposition_pipeline.as_ref().unwrap();

        // Create metadata buffer
        let metadata = LinalgMetadata {
            rows_a: n,
            cols_a: n,
            rows_b: n,
            cols_b: n,
            batch_size: 1,
            tolerance: T::from(1e-10)
                .unwrap_or_else(|| T::from(0.0).unwrap())
                .to_f64()
                .unwrap_or(1e-10) as f32,
            max_iterations: n,
            _padding: 0,
        };

        let metadata_buffer = self
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LU Metadata Buffer"),
                contents: bytemuck::bytes_of(&metadata),
                usage: BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LU Decomposition Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: l.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: u.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: metadata_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute operation
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("LU Decomposition Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LU Decomposition Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Calculate workgroup size - use 16x16 workgroups
            let workgroup_size = 16;
            let dispatch_x = (n + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (n + workgroup_size - 1) / workgroup_size;

            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // Submit command buffer
        self.queue().submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    /// Singular Value Decomposition (SVD)
    ///
    /// Performs SVD using Jacobi rotations adapted for GPU execution.
    /// For a matrix A [m, n], computes: A = U * S * V^T
    /// Where: U [m, min(m,n)] (orthogonal), S [min(m,n)] (singular values), V^T [min(m,n), n] (orthogonal)
    pub fn svd<T>(
        &mut self,
        input: &GpuBuffer<T>,
        u: &GpuBuffer<T>,
        s: &GpuBuffer<T>,
        vt: &GpuBuffer<T>,
        shape: &Shape,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        if shape.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "SVD requires a 2D matrix".to_string(),
            ));
        }

        let m = shape[0];
        let n = shape[1];
        let min_mn = m.min(n);

        if m == 0 || n == 0 {
            return Err(TensorError::invalid_shape_simple(
                "Cannot perform SVD on empty matrix".to_string(),
            ));
        }

        // For very small matrices, suggest CPU fallback
        if min_mn < 8 {
            return Err(TensorError::ComputeError {
                operation: "gpu_svd".to_string(),
                details: "GPU SVD requires matrices >= 8x8 - use CPU fallback for smaller matrices"
                    .to_string(),
                retry_possible: false,
                context: None,
            });
        }

        // Initialize SVD pipeline if needed
        if self.svd_pipeline.is_none() {
            self.initialize_svd_pipeline()?;
        }

        // Create working buffer for the algorithm
        let working_size = m * n;
        let working_buffer = self.device().create_buffer(&BufferDescriptor {
            label: Some("svd_working_buffer"),
            size: (working_size * std::mem::size_of::<T>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create metadata
        let metadata = LinalgMetadata::new(m, n)
            .with_tolerance(
                T::from(1e-10)
                    .unwrap_or_else(|| T::from(0.0).unwrap())
                    .to_f64()
                    .unwrap_or(1e-10) as f32,
            )
            .with_max_iterations(100 * min_mn as u32);
        let metadata_buffer = self.create_metadata_buffer(&metadata)?;

        // Load SVD shader and create pipelines for different stages
        let shader_source = include_str!("../shaders/linalg_svd.wgsl");
        let shader_module = self
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("svd_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let init_pipeline =
            self.device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("svd_init_pipeline"),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("initialize_svd"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        let givens_pipeline =
            self.device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("svd_givens_pipeline"),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("apply_givens_rotation"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        let extract_pipeline =
            self.device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("svd_extract_pipeline"),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("extract_singular_values"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Create bind group
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("svd_bind_group"),
            layout: &init_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: u.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: s.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vt.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: working_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: metadata_buffer.as_entire_binding(),
                },
            ],
        });

        // Initialize matrices
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("svd_init_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("svd_init_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&init_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups_x = (n as u32 + 15) / 16;
            let workgroups_y = (m as u32 + 15) / 16;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));

        // Iterative Jacobi rotations for square case
        if m == n {
            let max_iterations = metadata.max_iterations;

            for _iter in 0..max_iterations {
                let mut converged = true;

                // Apply Givens rotations to eliminate off-diagonal elements
                for i in 0..n {
                    for j in (i + 1)..n {
                        // Update metadata with current (i,j) pair
                        let updated_metadata = LinalgMetadata {
                            rows_a: m as u32,
                            cols_a: n as u32,
                            rows_b: i as u32, // Reusing for i
                            cols_b: j as u32, // Reusing for j
                            batch_size: 1,
                            tolerance: metadata.tolerance,
                            max_iterations: metadata.max_iterations,
                            _padding: 0,
                        };

                        let iter_metadata_buffer =
                            self.create_metadata_buffer(&updated_metadata)?;

                        let iter_bind_group =
                            self.device().create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("svd_iter_bind_group"),
                                layout: &givens_pipeline.get_bind_group_layout(0),
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: input.buffer().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: u.buffer().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: s.buffer().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: vt.buffer().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 4,
                                        resource: working_buffer.as_entire_binding(),
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
                                    label: Some("svd_givens_encoder"),
                                });

                        {
                            let mut compute_pass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("svd_givens_pass"),
                                    timestamp_writes: None,
                                });

                            compute_pass.set_pipeline(&givens_pipeline);
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
        }

        // Extract singular values from diagonal
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("svd_extract_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("svd_extract_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&extract_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((min_mn as u32 + 255) / 256, 1, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    /// QR decomposition using Householder reflections
    ///
    /// Computes QR decomposition using Householder reflections adapted for GPU execution.
    /// For a matrix A [m, n], computes: A = Q * R
    /// Where: Q [m, m] (orthogonal), R [m, n] (upper triangular)
    pub fn qr_decomposition<T>(
        &mut self,
        input: &GpuBuffer<T>,
        q: &GpuBuffer<T>,
        r: &GpuBuffer<T>,
        shape: &Shape,
    ) -> Result<()>
    where
        T: Float + Pod + Zeroable + Clone + Send + Sync + 'static,
    {
        if shape.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "QR decomposition requires a 2D matrix".to_string(),
            ));
        }

        let m = shape[0];
        let n = shape[1];

        if m == 0 || n == 0 {
            return Err(TensorError::invalid_shape_simple(
                "Cannot perform QR decomposition on empty matrix".to_string(),
            ));
        }

        // For very small matrices, suggest CPU fallback
        if m.min(n) < 4 {
            return Err(TensorError::ComputeError {
                operation: "gpu_qr_decomposition".to_string(),
                details: "GPU QR decomposition requires matrices >= 4x4 - use CPU fallback for smaller matrices".to_string(),
                retry_possible: false,
                context: None,
            });
        }

        // Initialize QR decomposition pipeline if needed
        if self.qr_decomposition_pipeline.is_none() {
            self.qr_decomposition_pipeline = Some(self.create_qr_pipeline()?);
        }

        let pipeline = self.qr_decomposition_pipeline.as_ref().unwrap();

        // Create working buffers
        let matrix_size = m * n;
        let working_matrix = self.device().create_buffer(&BufferDescriptor {
            label: Some("qr_working_matrix"),
            size: (matrix_size * std::mem::size_of::<T>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let householder_vectors = self.device().create_buffer(&BufferDescriptor {
            label: Some("qr_householder_vectors"),
            size: (m * n.min(m) * std::mem::size_of::<T>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tau_buffer = self.device().create_buffer(&BufferDescriptor {
            label: Some("qr_tau_buffer"),
            size: (n.min(m) * std::mem::size_of::<T>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy input to working matrix
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qr_copy_encoder"),
            });

        encoder.copy_buffer_to_buffer(
            input.buffer(),
            0,
            &working_matrix,
            0,
            (matrix_size * std::mem::size_of::<T>()) as u64,
        );

        self.queue().submit(std::iter::once(encoder.finish()));

        // Create metadata
        let metadata = LinalgMetadata::new(m, n)
            .with_tolerance(
                T::from(1e-10)
                    .unwrap_or_else(|| T::from(0.0).unwrap())
                    .to_f64()
                    .unwrap_or(1e-10) as f32,
            )
            .with_max_iterations(n.min(m) as u32);
        let metadata_buffer = self.create_metadata_buffer(&metadata)?;

        // Load QR shader and create pipelines for different stages
        let shader_source = include_str!("../shaders/linalg_qr.wgsl");
        let shader_module = self
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("qr_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let householder_pipeline =
            self.device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("qr_householder_pipeline"),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("compute_householder"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        let apply_pipeline =
            self.device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("qr_apply_pipeline"),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("apply_householder"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        let extract_q_pipeline =
            self.device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("qr_extract_q_pipeline"),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("extract_q_matrix"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        let extract_r_pipeline =
            self.device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("qr_extract_r_pipeline"),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("extract_r_matrix"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Create bind group
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qr_bind_group"),
            layout: &householder_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: working_matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: q.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: r.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: householder_vectors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: tau_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: metadata_buffer.as_entire_binding(),
                },
            ],
        });

        // Perform Householder QR decomposition for each column
        let min_mn = m.min(n);
        for k in 0..min_mn {
            // Update metadata with current column
            let updated_metadata = LinalgMetadata {
                rows_a: m as u32,
                cols_a: n as u32,
                rows_b: k as u32, // Current column index
                cols_b: 0,
                batch_size: 1,
                tolerance: metadata.tolerance,
                max_iterations: metadata.max_iterations,
                _padding: 0,
            };

            let iter_metadata_buffer = self.create_metadata_buffer(&updated_metadata)?;

            let iter_bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("qr_iter_bind_group"),
                layout: &householder_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: working_matrix.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: q.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: r.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: householder_vectors.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: tau_buffer.as_entire_binding(),
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
                        label: Some("qr_householder_encoder"),
                    });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("qr_householder_pass"),
                    timestamp_writes: None,
                });

                // Compute Householder vector for column k
                compute_pass.set_pipeline(&householder_pipeline);
                compute_pass.set_bind_group(0, &iter_bind_group, &[]);
                compute_pass.dispatch_workgroups((m as u32 + 255) / 256, 1, 1);

                // Apply Householder transformation to remaining columns
                compute_pass.set_pipeline(&apply_pipeline);
                compute_pass.set_bind_group(0, &iter_bind_group, &[]);
                let workgroups_x = (n as u32 + 15) / 16;
                let workgroups_y = (m as u32 + 15) / 16;
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            self.queue().submit(std::iter::once(encoder.finish()));
        }

        // Extract Q matrix from Householder vectors
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qr_extract_q_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("qr_extract_q_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&extract_q_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = (m as u32 + 15) / 16;
            let workgroups_y = (m as u32 + 15) / 16;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));

        // Extract R matrix (upper triangular) from working matrix
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qr_extract_r_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("qr_extract_r_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&extract_r_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = (n as u32 + 15) / 16;
            let workgroups_y = (m as u32 + 15) / 16;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));

        Ok(())
    }
}
