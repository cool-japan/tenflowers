//! GPU Linear Algebra Context and Configuration
//!
//! This module provides the core infrastructure for GPU linear algebra operations,
//! including context management, metadata structures, and adaptive configuration
//! for optimal performance across different matrix sizes and hardware.

use crate::gpu::{buffer::GpuBuffer, GpuContext};
use crate::{Result, Shape, TensorError};
use bytemuck::{Pod, Zeroable};
use num_traits::Float;
use std::sync::Arc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, ComputePipeline, Device, Queue};

/// GPU Linear Algebra Context
///
/// Manages compute pipelines and resources for GPU linear algebra operations.
/// Follows the existing tenflowers-core pattern for GPU contexts.
pub struct GpuLinalgContext {
    /// WGPU device reference
    device: Arc<Device>,
    /// WGPU command queue
    queue: Arc<Queue>,

    // Compute pipelines for different operations
    /// LU decomposition with partial pivoting
    pub lu_decomposition_pipeline: Option<ComputePipeline>,
    /// SVD computation using Jacobi rotations
    pub svd_pipeline: Option<ComputePipeline>,
    /// QR decomposition using Householder reflections
    pub qr_decomposition_pipeline: Option<ComputePipeline>,
    /// Eigenvalue computation using QR algorithm
    eigenvalue_pipeline: Option<ComputePipeline>,
    /// Linear system solver (using LU factorization)
    linear_solve_pipeline: Option<ComputePipeline>,
    /// Matrix inversion
    matrix_inverse_pipeline: Option<ComputePipeline>,
    /// Matrix determinant
    determinant_pipeline: Option<ComputePipeline>,

    // Utility pipelines
    /// Matrix transpose
    pub transpose_pipeline: Option<ComputePipeline>,
    /// Matrix multiplication (optimized for linalg operations)
    pub matmul_linalg_pipeline: Option<ComputePipeline>,

    // Optimization configuration
    /// Adaptive GEMM configuration for matrix size optimization
    adaptive_gemm_config: AdaptiveGemmConfig,
}

/// Metadata for linear algebra operations
///
/// This structure is passed to GPU kernels to provide matrix dimensions
/// and computation parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LinalgMetadata {
    /// Number of rows in matrix A
    pub rows_a: u32,
    /// Number of columns in matrix A
    pub cols_a: u32,
    /// Number of rows in matrix B (if applicable)
    pub rows_b: u32,
    /// Number of columns in matrix B (if applicable)
    pub cols_b: u32,
    /// Batch size for batched operations
    pub batch_size: u32,
    /// Tolerance for iterative algorithms
    pub tolerance: f32,
    /// Maximum iterations for iterative algorithms
    pub max_iterations: u32,
    /// Padding for alignment
    pub _padding: u32,
}

/// Adaptive GEMM configuration for optimizing GPU matrix multiplication
/// based on matrix dimensions and hardware characteristics
#[derive(Debug, Clone)]
pub struct AdaptiveGemmConfig {
    /// Tile size for small matrices (< 256x256)
    pub small_tile_size: u32,
    /// Tile size for medium matrices (256x256 to 2048x2048)
    pub medium_tile_size: u32,
    /// Tile size for large matrices (> 2048x2048)
    pub large_tile_size: u32,
    /// Workgroup size for different scenarios
    pub workgroup_sizes: Vec<(u32, u32)>,
    /// Memory coalescing threshold
    pub coalescing_threshold: u32,
    /// Shared memory usage preference
    pub prefer_shared_memory: bool,
}

impl Default for AdaptiveGemmConfig {
    fn default() -> Self {
        Self {
            small_tile_size: 8,
            medium_tile_size: 16,
            large_tile_size: 32,
            workgroup_sizes: vec![(8, 8), (16, 16), (32, 32)],
            coalescing_threshold: 128,
            prefer_shared_memory: true,
        }
    }
}

impl AdaptiveGemmConfig {
    /// Select optimal tile size based on matrix dimensions
    pub fn select_tile_size(&self, m: usize, n: usize, k: usize) -> u32 {
        let matrix_size = (m * n).max(m * k).max(n * k);

        if matrix_size < 256 * 256 {
            self.small_tile_size
        } else if matrix_size < 2048 * 2048 {
            self.medium_tile_size
        } else {
            self.large_tile_size
        }
    }

    /// Select optimal workgroup size based on matrix characteristics
    pub fn select_workgroup_size(&self, m: usize, n: usize, _k: usize) -> (u32, u32) {
        // Prefer square workgroups for balanced memory access
        let total_ops = m * n;

        if total_ops < 64 * 64 {
            self.workgroup_sizes[0]
        } else if total_ops < 512 * 512 {
            self.workgroup_sizes[1]
        } else {
            self.workgroup_sizes[2]
        }
    }

    /// Estimate memory bandwidth utilization
    pub fn estimate_bandwidth_utilization(&self, m: usize, n: usize, k: usize) -> f32 {
        let tile_size = self.select_tile_size(m, n, k) as usize;
        let tiles_m = (m + tile_size - 1) / tile_size;
        let tiles_n = (n + tile_size - 1) / tile_size;
        let tiles_k = (k + tile_size - 1) / tile_size;

        // Estimate based on data reuse in tiled computation
        let total_elements = m * n + m * k + n * k;
        let reused_elements = tiles_m * tiles_n * tiles_k * tile_size * tile_size;

        // Higher reuse means better bandwidth utilization
        (reused_elements as f32 / total_elements as f32).min(1.0)
    }
}

impl LinalgMetadata {
    /// Create metadata for matrix operations
    pub fn new(rows_a: usize, cols_a: usize) -> Self {
        Self {
            rows_a: rows_a as u32,
            cols_a: cols_a as u32,
            rows_b: 0,
            cols_b: 0,
            batch_size: 1,
            tolerance: 1e-6,
            max_iterations: 1000,
            _padding: 0,
        }
    }

    /// Create metadata for two-matrix operations
    pub fn new_two_matrices(rows_a: usize, cols_a: usize, rows_b: usize, cols_b: usize) -> Self {
        Self {
            rows_a: rows_a as u32,
            cols_a: cols_a as u32,
            rows_b: rows_b as u32,
            cols_b: cols_b as u32,
            batch_size: 1,
            tolerance: 1e-6,
            max_iterations: 1000,
            _padding: 0,
        }
    }

    /// Set tolerance for iterative algorithms
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set maximum iterations for iterative algorithms
    pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set batch size for batched operations
    pub fn with_batch_size(mut self, batch_size: u32) -> Self {
        self.batch_size = batch_size;
        self
    }
}

impl GpuLinalgContext {
    /// Create a new GPU linear algebra context
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            device,
            queue,
            lu_decomposition_pipeline: None,
            svd_pipeline: None,
            qr_decomposition_pipeline: None,
            eigenvalue_pipeline: None,
            linear_solve_pipeline: None,
            matrix_inverse_pipeline: None,
            determinant_pipeline: None,
            transpose_pipeline: None,
            matmul_linalg_pipeline: None,
            adaptive_gemm_config: AdaptiveGemmConfig::default(),
        }
    }

    /// Create a new GPU linear algebra context with custom GEMM configuration
    pub fn with_adaptive_gemm_config(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: AdaptiveGemmConfig,
    ) -> Self {
        Self {
            device,
            queue,
            lu_decomposition_pipeline: None,
            svd_pipeline: None,
            qr_decomposition_pipeline: None,
            eigenvalue_pipeline: None,
            linear_solve_pipeline: None,
            matrix_inverse_pipeline: None,
            determinant_pipeline: None,
            transpose_pipeline: None,
            matmul_linalg_pipeline: None,
            adaptive_gemm_config: config,
        }
    }

    /// Initialize compute pipelines
    ///
    /// This method loads and compiles the WGSL shaders for linear algebra operations.
    /// It's called lazily when operations are first requested.
    pub fn initialize_pipelines(&mut self) -> Result<()> {
        // Initialize transpose pipeline (used by many operations)
        if self.transpose_pipeline.is_none() {
            self.transpose_pipeline = Some(self.create_transpose_pipeline()?);
        }

        // Initialize matrix multiplication pipeline optimized for linalg
        if self.matmul_linalg_pipeline.is_none() {
            self.matmul_linalg_pipeline = Some(self.create_matmul_linalg_pipeline()?);
        }

        Ok(())
    }

    /// Initialize SVD pipeline
    pub fn initialize_svd_pipeline(&mut self) -> Result<()> {
        if self.svd_pipeline.is_none() {
            self.svd_pipeline = Some(self.create_svd_pipeline()?);
        }
        Ok(())
    }

    /// Create SVD compute pipeline
    fn create_svd_pipeline(&self) -> Result<ComputePipeline> {
        let shader_source = include_str!("../shaders/linalg_svd.wgsl");
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("linalg_svd_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("linalg_svd_pipeline"),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("initialize_svd"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        Ok(compute_pipeline)
    }

    /// Initialize eigenvalue pipeline
    pub fn initialize_eigenvalue_pipeline(&mut self) -> Result<()> {
        if self.eigenvalue_pipeline.is_none() {
            self.eigenvalue_pipeline = Some(self.create_eigenvalue_pipeline()?);
        }
        Ok(())
    }

    /// Create eigenvalue compute pipeline
    fn create_eigenvalue_pipeline(&self) -> Result<ComputePipeline> {
        let shader_source = include_str!("../shaders/linalg_eigenvalue.wgsl");
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("linalg_eigenvalue_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("linalg_eigenvalue_pipeline"),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("initialize_eigen"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        Ok(compute_pipeline)
    }

    /// Create transpose compute pipeline
    fn create_transpose_pipeline(&self) -> Result<ComputePipeline> {
        let shader_source = include_str!("../shaders/linalg_transpose.wgsl");
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("linalg_transpose_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("linalg_transpose_pipeline"),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        Ok(compute_pipeline)
    }

    /// Create matrix multiplication pipeline optimized for linear algebra
    fn create_matmul_linalg_pipeline(&self) -> Result<ComputePipeline> {
        let shader_source = include_str!("../shaders/linalg_matmul.wgsl");
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("linalg_matmul_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("linalg_matmul_pipeline"),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        Ok(compute_pipeline)
    }

    /// Initialize LU decomposition pipeline
    pub fn initialize_lu_pipeline(&mut self) -> Result<()> {
        if self.lu_decomposition_pipeline.is_none() {
            self.lu_decomposition_pipeline = Some(self.create_lu_pipeline()?);
        }
        Ok(())
    }

    /// Create LU decomposition compute pipeline
    fn create_lu_pipeline(&self) -> Result<ComputePipeline> {
        let shader_source = include_str!("../shaders/linalg_lu.wgsl");
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("linalg_lu_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("linalg_lu_pipeline"),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        Ok(compute_pipeline)
    }

    /// Create QR decomposition compute pipeline
    pub fn create_qr_pipeline(&self) -> Result<ComputePipeline> {
        let shader_source = include_str!("../shaders/linalg_qr.wgsl");
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("linalg_qr_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("linalg_qr_pipeline"),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("compute_householder"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        Ok(compute_pipeline)
    }

    /// Create linear solver compute pipeline
    fn create_linear_solve_pipeline(&self) -> Result<ComputePipeline> {
        let shader_source = include_str!("../shaders/linalg_solve.wgsl");
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("linalg_solve_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("linalg_solve_pipeline"),
                    layout: None,
                    module: &shader_module,
                    entry_point: Some("apply_permutation"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        Ok(compute_pipeline)
    }

    /// Get adaptive GEMM configuration
    pub fn adaptive_gemm_config(&self) -> &AdaptiveGemmConfig {
        &self.adaptive_gemm_config
    }

    /// Update adaptive GEMM configuration
    pub fn set_adaptive_gemm_config(&mut self, config: AdaptiveGemmConfig) {
        self.adaptive_gemm_config = config;
    }

    /// Create metadata buffer for passing parameters to GPU kernels
    pub fn create_metadata_buffer(&self, metadata: &LinalgMetadata) -> Result<Buffer> {
        let metadata_bytes = bytemuck::bytes_of(metadata);

        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("linalg_metadata_buffer"),
            size: metadata_bytes.len() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.queue.write_buffer(&buffer, 0, metadata_bytes);
        Ok(buffer)
    }

    /// Get device reference
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get queue reference
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}
