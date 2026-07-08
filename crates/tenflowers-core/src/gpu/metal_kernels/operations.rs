//! Metal Performance Shaders (MPS) Operations
//!
//! This module provides optimized Metal operations using both MPS and custom compute kernels
//! for maximum performance on Apple devices.

use super::device::MetalDevice;
use super::types::{ActivationType, ElementwiseOp, MetalKernelConfig, ReductionOp};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::{Result, Tensor, TensorError};
#[cfg(all(target_os = "macos", feature = "metal"))]
use metal;

/// Core MPS-based tensor operations
#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalDevice {
    /// Execute optimized matrix multiplication using Metal Performance Shaders
    pub fn matmul_mps<T>(&mut self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: bytemuck::Pod + Clone + Default + Send + Sync + 'static,
    {
        // Use MPS for optimized GEMM operations
        self.execute_mps_gemm(a, b)
    }

    /// Execute optimized convolution using Metal Performance Shaders
    pub fn conv2d_mps<T>(
        &mut self,
        input: &Tensor<T>,
        weights: &Tensor<T>,
        bias: Option<&Tensor<T>>,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        self.execute_mps_conv2d(input, weights, bias, stride, padding)
    }

    /// Execute optimized reduction operations (sum, mean, max, min)
    pub fn reduce_optimized<T>(
        &mut self,
        tensor: &Tensor<T>,
        operation: ReductionOp,
        axes: Option<&[usize]>,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        match operation {
            ReductionOp::Sum => self.execute_optimized_sum(tensor, axes),
            ReductionOp::Mean => self.execute_optimized_mean(tensor, axes),
            ReductionOp::Max => self.execute_optimized_max(tensor, axes),
            ReductionOp::Min => self.execute_optimized_min(tensor, axes),
        }
    }

    /// Execute fused activation functions for maximum performance
    pub fn fused_activation<T>(
        &mut self,
        tensor: &Tensor<T>,
        activation: ActivationType,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let kernel_name = match activation {
            ActivationType::ReLU => "fused_relu",
            ActivationType::GELU => "fused_gelu",
            ActivationType::Swish => "fused_swish",
            ActivationType::Tanh => "fused_tanh",
            ActivationType::Sigmoid => "fused_sigmoid",
        };

        let tensor_data = tensor
            .as_slice()
            .ok_or_else(|| TensorError::InvalidOperation {
                operation: "metal_kernel".to_string(),
                reason: "Failed to access tensor data".to_string(),
                context: None,
            })?;
        self.execute_kernel(kernel_name, &[tensor_data], tensor.shape().dims())
    }

    /// Execute memory-coalesced element-wise operations
    pub fn elementwise_coalesced<T>(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        operation: ElementwiseOp,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let kernel_name = match operation {
            ElementwiseOp::Add => "coalesced_add",
            ElementwiseOp::Mul => "coalesced_mul",
            ElementwiseOp::Sub => "coalesced_sub",
            ElementwiseOp::Div => "coalesced_div",
        };

        // Create optimal Metal kernel config for elementwise operations
        let total_elements = a.shape().dims().iter().product::<usize>();
        let config = MetalKernelConfig {
            threads_per_group: metal::MTLSize::new(64, 1, 1),
            thread_groups: metal::MTLSize::new(((total_elements + 63) / 64) as u64, 1, 1),
        };
        let a_data = a.as_slice().ok_or_else(|| TensorError::InvalidOperation {
            operation: "metal_matmul".to_string(),
            reason: "Failed to access tensor data".to_string(),
            context: None,
        })?;
        let b_data = b.as_slice().ok_or_else(|| TensorError::InvalidOperation {
            operation: "metal_matmul".to_string(),
            reason: "Failed to access tensor data".to_string(),
            context: None,
        })?;
        self.execute_kernel_with_config(kernel_name, &[a_data, b_data], &config)
    }

    /// Execute optimized layer normalization for transformer models
    pub fn layer_norm_optimized<T>(
        &mut self,
        input: &Tensor<T>,
        gamma: &Tensor<T>,
        beta: &Tensor<T>,
        eps: f32,
    ) -> Result<Tensor<T>>
    where
        T: bytemuck::Pod + Clone + Default + Send + Sync + 'static,
    {
        let input_shape = input.shape();
        if input_shape.len() < 2 {
            return Err(TensorError::invalid_operation_simple(
                "LayerNorm requires at least 2D input".to_string(),
            ));
        }

        let batch_size = input_shape[0];
        let feature_size = input_shape.dims()[1..].iter().product::<usize>();

        // Validate gamma and beta shapes
        if gamma.numel() != feature_size || beta.numel() != feature_size {
            return Err(TensorError::invalid_operation_simple(
                "Gamma and beta must match feature dimensions".to_string(),
            ));
        }

        let output_shape = input_shape.to_vec();
        self.execute_layer_norm_kernel(
            input,
            gamma,
            beta,
            eps,
            batch_size,
            feature_size,
            output_shape,
        )
    }

    /// Execute optimized group normalization
    pub fn group_norm_optimized<T>(
        &mut self,
        input: &Tensor<T>,
        gamma: &Tensor<T>,
        beta: &Tensor<T>,
        groups: usize,
        eps: f32,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(TensorError::invalid_operation_simple(
                "GroupNorm requires 4D input [batch, channels, height, width]".to_string(),
            ));
        }

        let (batch_size, channels, height, width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let spatial_size = height * width;

        if channels % groups != 0 {
            return Err(TensorError::invalid_operation_simple(format!(
                "Channels {} must be divisible by groups {}",
                channels, groups
            )));
        }

        let output_shape = input_shape.to_vec();
        self.execute_group_norm_kernel(
            input,
            gamma,
            beta,
            groups,
            eps,
            batch_size,
            channels,
            spatial_size,
            output_shape,
        )
    }

    /// Execute Flash Attention for transformer models
    pub fn flash_attention<T>(
        &mut self,
        query: &Tensor<T>,
        key: &Tensor<T>,
        value: &Tensor<T>,
        scale: f32,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();

        // Validate shapes: [batch_size, num_heads, seq_len, head_dim]
        if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
            return Err(TensorError::invalid_operation_simple(
                "Flash attention requires 4D tensors [batch, heads, seq_len, head_dim]".to_string(),
            ));
        }

        if q_shape != k_shape || q_shape != v_shape {
            return Err(TensorError::invalid_operation_simple(
                "Query, key, and value must have the same shape".to_string(),
            ));
        }

        let (batch_size, num_heads, seq_len, head_dim) =
            (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);

        let output_shape = q_shape.to_vec();
        self.execute_flash_attention_kernel(
            query,
            key,
            value,
            scale,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            output_shape,
        )
    }

    /// Execute Apple Silicon SIMD optimized operations
    pub fn apple_silicon_simd_add<T>(&mut self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Check if tensors can be processed with SIMD (4-element alignment)
        let numel = a.numel();
        if numel % 4 != 0 {
            return Err(TensorError::invalid_operation_simple(
                "SIMD operations require 4-element alignment".to_string(),
            ));
        }

        let kernel_name = "apple_silicon_simd_add";
        let config = MetalKernelConfig {
            threads_per_group: metal::MTLSize::new(64, 1, 1),
            thread_groups: metal::MTLSize::new(((numel / 4 + 63) / 64) as u64, 1, 1),
        };

        self.execute_kernel_with_config(kernel_name, &[a.data(), b.data()], &config)
    }

    /// Execute performance bandwidth testing
    pub fn measure_memory_bandwidth<T>(&mut self, data_size: usize) -> Result<(f64, Vec<u64>)>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Create test buffers
        let input_data = vec![T::default(); data_size];
        let mut output_data = vec![T::default(); data_size];
        let mut stats = vec![0u64; 4]; // [operations, bandwidth_mbps, latency_ns, throughput]

        let kernel_name = "performance_bandwidth_test";

        // Record start time
        let start_time = std::time::Instant::now();

        // Execute bandwidth test kernel
        let command_queue = self.command_queue().clone();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let pipeline = self.get_or_create_pipeline(kernel_name)?;

        encoder.set_compute_pipeline_state(pipeline);

        // Set buffers (simplified - actual implementation would use Metal buffers)
        let config = MetalKernelConfig {
            threads_per_group: metal::MTLSize::new(256, 1, 1),
            thread_groups: metal::MTLSize::new(((data_size + 255) / 256) as u64, 1, 1),
        };

        encoder.dispatch_thread_groups(config.thread_groups, config.threads_per_group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let elapsed = start_time.elapsed();
        let bandwidth_mbps = (data_size * std::mem::size_of::<T>() * 2) as f64
            / (elapsed.as_secs_f64() * 1_000_000.0); // Read + Write

        Ok((bandwidth_mbps, stats))
    }

    // Private implementation methods for MPS operations

    fn execute_mps_gemm<T>(&mut self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: bytemuck::Pod + Clone + Default + Send + Sync + 'static,
    {
        // Implementation using Metal Performance Shaders GEMM
        // Use optimized matrix multiplication kernel for maximum performance

        let a_shape = a.shape();
        let b_shape = b.shape();

        // Validate matrix multiplication dimensions
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(TensorError::invalid_operation_simple(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        let (m, k) = (a_shape[0], a_shape[1]);
        let (k2, n) = (b_shape[0], b_shape[1]);

        if k != k2 {
            return Err(TensorError::invalid_operation_simple(format!(
                "Matrix dimension mismatch: {} vs {}",
                k, k2
            )));
        }

        // Create output tensor
        let output_shape = vec![m, n];
        let output_data = vec![T::default(); m * n];

        // For now, use our optimized Metal kernel instead of MPS
        // In a full implementation, this would use MPSMatrixMultiplication
        //
        // The real kernel entry point in shaders/metal_kernels.metal is named
        // `matrix_multiply_naive` -- there is no `optimized_matmul` function in
        // that file, so requesting that name made `get_or_create_pipeline` fail
        // before this kernel ever ran.
        let kernel_name = "matrix_multiply_naive";

        let command_queue = self.command_queue().clone();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Get pipeline
        let pipeline = self.get_or_create_pipeline(kernel_name)?;
        encoder.set_compute_pipeline_state(pipeline);

        // Create Metal buffers
        let buffer_a = self.create_metal_buffer(a.data())?;
        let buffer_b = self.create_metal_buffer(b.data())?;
        let buffer_c = self.create_metal_buffer(&output_data)?;

        // Set buffers
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_c), 0);

        // Set dimensions
        let m_bytes = std::mem::size_of::<u32>();
        let n_bytes = std::mem::size_of::<u32>();
        let k_bytes = std::mem::size_of::<u32>();

        encoder.set_bytes(
            3,
            m_bytes as u64,
            &(m as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            n_bytes as u64,
            &(n as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            k_bytes as u64,
            &(k as u32) as *const u32 as *const std::ffi::c_void,
        );

        // Calculate optimal dispatch configuration. The shader's own bounds check
        // is `if (gid.x >= M || gid.y >= N) return;` -- `gid.x` ranges over M
        // (rows) and `gid.y` ranges over N (columns) -- so `thread_groups` must be
        // sized (ceil(M/32), ceil(N/32)); the previous (N, M) ordering here was
        // transposed and would silently under-dispatch (or over-dispatch) rows vs.
        // columns whenever M != N.
        let threads_per_group = metal::MTLSize::new(32, 32, 1);
        let thread_groups = metal::MTLSize::new(((m + 31) / 32) as u64, ((n + 31) / 32) as u64, 1);

        encoder.dispatch_thread_groups(thread_groups, threads_per_group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read the GPU-written result back into host memory. `buffer_c` was
        // allocated with `StorageModeShared` (see `create_metal_buffer`), which on
        // Apple Silicon's unified memory architecture is synchronously visible to
        // the CPU as soon as the command buffer finishes -- unlike the wgpu
        // readback path, no `map_async`-style completion handshake is needed.
        //
        // SAFETY: `wait_until_completed()` above guarantees the GPU has finished
        // writing `m * n` elements of `T` into `buffer_c`, which was allocated
        // with exactly `m * n * size_of::<T>()` bytes (from `output_data`).
        // `T: bytemuck::Pod` (this function's tightened bound) guarantees every
        // bit pattern is a valid `T`, so reinterpreting the buffer's raw bytes as
        // `&[T]` here is sound.
        let result_data = unsafe {
            let ptr = buffer_c.contents() as *const T;
            std::slice::from_raw_parts(ptr, m * n).to_vec()
        };

        Tensor::from_vec(result_data, &output_shape)
    }

    fn execute_mps_conv2d<T>(
        &mut self,
        input: &Tensor<T>,
        weights: &Tensor<T>,
        bias: Option<&Tensor<T>>,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Implementation using Metal Performance Shaders convolution
        // For now, use our optimized Metal kernel instead of MPSCNNConvolution
        // In a full implementation, this would use MPSCNNConvolution for maximum performance

        let input_shape = input.shape();
        let weight_shape = weights.shape();

        // Validate convolution shapes: input[batch, channels, height, width], weights[out_channels, in_channels, kernel_h, kernel_w]
        if input_shape.len() != 4 || weight_shape.len() != 4 {
            return Err(TensorError::invalid_operation_simple(
                "Convolution requires 4D tensors (NCHW format)".to_string(),
            ));
        }

        if input_shape[1] != weight_shape[1] {
            return Err(TensorError::invalid_operation_simple(format!(
                "Input channels ({}) must match weight input channels ({})",
                input_shape[1], weight_shape[1]
            )));
        }

        let (batch_size, _in_channels, input_height, input_width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (out_channels, _in_channels, kernel_height, kernel_width) = (
            weight_shape[0],
            weight_shape[1],
            weight_shape[2],
            weight_shape[3],
        );

        // Calculate output dimensions
        let output_height = (input_height + 2 * padding[0] - kernel_height) / stride[0] + 1;
        let output_width = (input_width + 2 * padding[1] - kernel_width) / stride[1] + 1;

        let output_shape = vec![batch_size, out_channels, output_height, output_width];
        let output_size = output_shape.iter().product::<usize>();
        let output_data = vec![T::default(); output_size];

        let command_queue = self.command_queue().clone();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Get the optimized convolution pipeline
        let pipeline = self.get_or_create_pipeline("optimized_conv2d")?;
        encoder.set_compute_pipeline_state(pipeline);

        // Create Metal buffers
        let input_buffer = self.create_metal_buffer(input.data())?;
        let weight_buffer = self.create_metal_buffer(weights.data())?;
        let output_buffer = self.create_metal_buffer(&output_data)?;

        // Set buffers
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);

        // Set shader parameters
        let input_shape_metal = [
            input_shape[0] as u32,
            input_shape[1] as u32,
            input_shape[2] as u32,
            input_shape[3] as u32,
        ];
        let weight_shape_metal = [
            weight_shape[0] as u32,
            weight_shape[1] as u32,
            weight_shape[2] as u32,
            weight_shape[3] as u32,
        ];
        let stride_metal = [stride[0] as u32, stride[1] as u32];
        let padding_metal = [padding[0] as u32, padding[1] as u32];

        encoder.set_bytes(3, 16, input_shape_metal.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(
            4,
            16,
            weight_shape_metal.as_ptr() as *const std::ffi::c_void,
        );
        encoder.set_bytes(5, 8, stride_metal.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(6, 8, padding_metal.as_ptr() as *const std::ffi::c_void);

        // Calculate optimal dispatch configuration for convolution
        let threads_per_group = metal::MTLSize::new(8, 8, 1);
        let thread_groups = metal::MTLSize::new(
            ((output_height * output_width + 63) / 64) as u64,
            ((out_channels + 7) / 8) as u64,
            batch_size as u64,
        );

        encoder.dispatch_thread_groups(thread_groups, threads_per_group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Validate bias shape if provided (this check is still meaningful).
        if let Some(bias_tensor) = bias {
            if bias_tensor.shape().len() != 1 || bias_tensor.shape()[0] != out_channels {
                return Err(TensorError::invalid_operation_simple(
                    "Bias must be 1D with size equal to output channels".to_string(),
                ));
            }
        }

        // HONEST ERROR: the GPU kernel above writes its result into `output_buffer`,
        // but we have NOT implemented the GPU->host readback. Returning `output_data`
        // here would hand back the un-read-back host zeros (a silent fabrication),
        // and any bias addition would only be applied to those zeros. Fail loudly.
        let _ = (
            input_buffer,
            weight_buffer,
            output_buffer,
            output_data,
            output_shape,
            batch_size,
            out_channels,
            output_height,
            output_width,
        );
        Err(TensorError::unsupported_operation_simple(
            "Metal MPS conv2d: GPU->host readback not implemented; result would be fabricated"
                .to_string(),
        ))
    }

    // Specialized reduction operations

    fn execute_optimized_sum<T>(
        &mut self,
        tensor: &Tensor<T>,
        axes: Option<&[usize]>,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Optimized parallel reduction using hierarchical reduction
        self.execute_kernel("hierarchical_sum", &[tensor.data()], tensor.shape().dims())
    }

    fn execute_optimized_mean<T>(
        &mut self,
        tensor: &Tensor<T>,
        axes: Option<&[usize]>,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Optimized mean with single-pass algorithm
        self.execute_kernel("optimized_mean", &[tensor.data()], tensor.shape().dims())
    }

    fn execute_optimized_max<T>(
        &mut self,
        tensor: &Tensor<T>,
        axes: Option<&[usize]>,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Optimized max reduction with early termination
        self.execute_kernel("optimized_max", &[tensor.data()], tensor.shape().dims())
    }

    fn execute_optimized_min<T>(
        &mut self,
        tensor: &Tensor<T>,
        axes: Option<&[usize]>,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Optimized min reduction with early termination
        self.execute_kernel("optimized_min", &[tensor.data()], tensor.shape().dims())
    }

    // Helper methods for neural network operations

    fn execute_layer_norm_kernel<T>(
        &mut self,
        input: &Tensor<T>,
        gamma: &Tensor<T>,
        beta: &Tensor<T>,
        eps: f32,
        batch_size: usize,
        feature_size: usize,
        output_shape: Vec<usize>,
    ) -> Result<Tensor<T>>
    where
        T: bytemuck::Pod + Clone + Default + Send + Sync + 'static,
    {
        // The `layer_norm` shader (shaders/metal_kernels.metal) dedicates exactly
        // one threadgroup per batch row and one thread per feature, reducing
        // mean/variance across the whole row purely within threadgroup shared
        // memory (`shared_sum` / `shared_sum_sq`). That single-threadgroup-per-row
        // design caps `feature_size` at this device's real max threads-per-
        // threadgroup (1024 on Apple Silicon); no multi-pass variant exists yet to
        // normalize wider rows, so fail honestly here instead of silently
        // truncating the row or over-subscribing the threadgroup.
        let max_threads = self.device().max_threads_per_threadgroup();
        if feature_size as u64 > max_threads.width {
            return Err(TensorError::unsupported_operation_simple(format!(
                "Metal layer norm: hidden_size {} exceeds this device's max threads \
                 per threadgroup ({}); the single-pass shared-memory kernel needs one \
                 thread per feature within a single threadgroup, and no multi-pass \
                 variant exists yet for larger hidden sizes",
                feature_size, max_threads.width
            )));
        }

        let input_data = input
            .as_slice()
            .ok_or_else(|| TensorError::InvalidOperation {
                operation: "metal_layer_norm".to_string(),
                reason: "Failed to access input tensor data".to_string(),
                context: None,
            })?;
        let gamma_data = gamma
            .as_slice()
            .ok_or_else(|| TensorError::InvalidOperation {
                operation: "metal_layer_norm".to_string(),
                reason: "Failed to access gamma tensor data".to_string(),
                context: None,
            })?;
        let beta_data = beta
            .as_slice()
            .ok_or_else(|| TensorError::InvalidOperation {
                operation: "metal_layer_norm".to_string(),
                reason: "Failed to access beta tensor data".to_string(),
                context: None,
            })?;

        let output_data = vec![T::default(); batch_size * feature_size];

        let command_queue = self.command_queue().clone();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let pipeline = self.get_or_create_pipeline("layer_norm")?;
        encoder.set_compute_pipeline_state(pipeline);

        let input_buffer = self.create_metal_buffer(input_data)?;
        let gamma_buffer = self.create_metal_buffer(gamma_data)?;
        let beta_buffer = self.create_metal_buffer(beta_data)?;
        let output_buffer = self.create_metal_buffer(&output_data)?;

        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&gamma_buffer), 0);
        encoder.set_buffer(2, Some(&beta_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        let hidden_size_u32 = feature_size as u32;
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &hidden_size_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<f32>() as u64,
            &eps as *const f32 as *const std::ffi::c_void,
        );

        // Threadgroup shared memory backing the shader's `shared_sum` /
        // `shared_sum_sq` accumulators (`threadgroup(0)` / `threadgroup(1)`): one
        // f32 slot per feature, matching `threads_per_threadgroup` below.
        let shared_mem_bytes = (feature_size * std::mem::size_of::<f32>()) as u64;
        encoder.set_threadgroup_memory_length(0, shared_mem_bytes);
        encoder.set_threadgroup_memory_length(1, shared_mem_bytes);

        // One threadgroup per batch row, one thread per feature -- matches the
        // shader's `gid / hidden_size` / `gid % hidden_size` indexing.
        let threadgroups = metal::MTLSize::new(batch_size as u64, 1, 1);
        let threads_per_threadgroup = metal::MTLSize::new(feature_size as u64, 1, 1);
        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // SAFETY: see the equivalent readback comment in `execute_mps_gemm` --
        // `StorageModeShared` + `wait_until_completed()` guarantee the GPU-written
        // bytes are synchronously CPU-visible, `output_buffer` was allocated with
        // exactly `batch_size * feature_size * size_of::<T>()` bytes (from
        // `output_data`), and `T: bytemuck::Pod` guarantees any bit pattern is a
        // valid `T`.
        let result_data = unsafe {
            let ptr = output_buffer.contents() as *const T;
            std::slice::from_raw_parts(ptr, batch_size * feature_size).to_vec()
        };

        Tensor::from_vec(result_data, &output_shape)
    }

    fn execute_group_norm_kernel<T>(
        &mut self,
        input: &Tensor<T>,
        gamma: &Tensor<T>,
        beta: &Tensor<T>,
        groups: usize,
        eps: f32,
        batch_size: usize,
        channels: usize,
        spatial_size: usize,
        output_shape: Vec<usize>,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // HONEST ERROR: this never ran a kernel and simply returned
        // `vec![T::default(); ...]` (host zeros) shaped like the output - a silent
        // fabrication. Fail loudly until a real Metal group-norm + readback exists.
        let _ = (
            input,
            gamma,
            beta,
            groups,
            eps,
            batch_size,
            channels,
            spatial_size,
            output_shape,
        );
        Err(TensorError::unsupported_operation_simple(
            "Metal group norm: not implemented; result would be fabricated".to_string(),
        ))
    }

    fn execute_flash_attention_kernel<T>(
        &mut self,
        query: &Tensor<T>,
        key: &Tensor<T>,
        value: &Tensor<T>,
        scale: f32,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        output_shape: Vec<usize>,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // HONEST ERROR: this never ran a kernel and simply returned
        // `vec![T::default(); ...]` (host zeros) shaped like the output - a silent
        // fabrication. Fail loudly until a real Metal flash-attention + readback exists.
        let _ = (
            query,
            key,
            value,
            scale,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            output_shape,
        );
        Err(TensorError::unsupported_operation_simple(
            "Metal flash attention: not implemented; result would be fabricated".to_string(),
        ))
    }

    // Kernel execution infrastructure

    fn execute_kernel<T>(
        &mut self,
        kernel_name: &str,
        buffers: &[&[T]],
        shape: &[usize],
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let total_elements = shape.iter().product::<usize>();
        let config = MetalKernelConfig {
            threads_per_group: metal::MTLSize::new(256, 1, 1),
            thread_groups: metal::MTLSize::new(((total_elements + 255) / 256) as u64, 1, 1),
        };
        self.execute_kernel_with_config(kernel_name, buffers, &config)
    }

    fn execute_kernel_with_config<T>(
        &mut self,
        kernel_name: &str,
        buffers: &[&[T]],
        config: &MetalKernelConfig,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Pre-allocate all Metal buffers to avoid borrow conflicts
        let mut metal_buffers = Vec::new();
        for buffer in buffers.iter() {
            let metal_buffer = self.create_metal_buffer(buffer)?;
            metal_buffers.push(metal_buffer);
        }

        let command_queue = self.command_queue().clone();
        let pipeline = self.get_or_create_pipeline(kernel_name)?;
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);

        // Set buffer arguments
        for (index, metal_buffer) in metal_buffers.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(metal_buffer), 0);
        }

        // Dispatch threads
        encoder.dispatch_thread_groups(config.thread_groups, config.threads_per_group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        if buffers.is_empty() {
            return Err(TensorError::invalid_operation_simple(
                "No input buffers provided".to_string(),
            ));
        }

        // HONEST ERROR: the kernel `{kernel_name}` ran on the GPU and wrote its
        // result into one of `metal_buffers`, but we have NOT implemented the
        // GPU->host readback. Previously this returned `vec![T::default(); ...]`
        // (host zeros) with the correct shape, which is a silent fabrication that
        // looks functional. Fail loudly so callers fall back / surface the gap.
        let _ = metal_buffers;
        Err(TensorError::unsupported_operation_simple(format!(
            "Metal kernel '{}': GPU->host readback not implemented; result would be fabricated",
            kernel_name
        )))
    }

    fn create_metal_buffer<T>(&self, data: &[T]) -> Result<metal::Buffer>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let size = std::mem::size_of_val(data);
        let buffer = self.device().new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real Metal device, or a graceful skip. Mirrors the skip-gracefully
    /// convention used by this module's sibling test files (e.g.
    /// `device.rs::test_device_creation`), so this test suite behaves
    /// identically whether or not a physical Metal-capable GPU is present.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn metal_device_or_skip(test_name: &str) -> Option<MetalDevice> {
        match MetalDevice::new() {
            Ok(device) => Some(device),
            Err(e) => {
                eprintln!("No Metal device available ({e}), skipping {test_name}");
                None
            }
        }
    }

    /// `matmul_mps` must now genuinely execute `matrix_multiply_naive` on the GPU
    /// and read the result back, instead of failing with the old "GPU->host
    /// readback not implemented" honest error. Verified against a plain CPU
    /// reference implementation of row-major matmul, using a non-square (M != N)
    /// shape so a regression to the old transposed dispatch-dimension bug (which
    /// silently mis-sized the thread grid whenever M != N) would be caught.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_matmul_mps_matches_cpu_reference() {
        let Some(mut device) = metal_device_or_skip("test_matmul_mps_matches_cpu_reference") else {
            return;
        };

        // M=2, K=3, N=4 (deliberately M != N to exercise the fixed dispatch dims).
        let (m, k, n) = (2usize, 3usize, 4usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| i as f32 + 1.0).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5 + 1.0).collect();

        let a = Tensor::from_vec(a_data.clone(), &[m, k]).expect("test: create a");
        let b = Tensor::from_vec(b_data.clone(), &[k, n]).expect("test: create b");

        let result = device
            .matmul_mps(&a, &b)
            .expect("test: matmul_mps must succeed on real Metal hardware");
        let result_data = result
            .as_slice()
            .expect("test: matmul_mps result must be CPU-resident");

        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a_data[i * k + kk] * b_data[kk * n + j];
                }
                expected[i * n + j] = sum;
            }
        }

        assert_eq!(result.shape().dims(), &[m, n]);
        assert_eq!(result_data.len(), expected.len());
        for (idx, (&got, &want)) in result_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "matmul_mps mismatch at index {idx}: got {got}, want {want}"
            );
        }
    }

    /// `layer_norm_optimized` must now genuinely dispatch the shader's
    /// single-pass mean/variance reduction and read the result back, instead of
    /// failing with the old "not implemented" honest error. Verified against a
    /// plain CPU reference implementation of layer normalization.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_layer_norm_optimized_matches_cpu_reference() {
        let Some(mut device) =
            metal_device_or_skip("test_layer_norm_optimized_matches_cpu_reference")
        else {
            return;
        };

        let batch_size = 3usize;
        let feature_size = 17usize; // deliberately not a power of two
        let eps = 1e-5f32;

        let input_data: Vec<f32> = (0..batch_size * feature_size)
            .map(|i| (i as f32 * 0.37).sin() * 5.0)
            .collect();
        let gamma_data: Vec<f32> = (0..feature_size).map(|i| 1.0 + i as f32 * 0.1).collect();
        let beta_data: Vec<f32> = (0..feature_size).map(|i| i as f32 * 0.05 - 0.3).collect();

        let input = Tensor::from_vec(input_data.clone(), &[batch_size, feature_size])
            .expect("test: create input");
        let gamma = Tensor::from_vec(gamma_data.clone(), &[feature_size]).expect("test: gamma");
        let beta = Tensor::from_vec(beta_data.clone(), &[feature_size]).expect("test: beta");

        let result = device
            .layer_norm_optimized(&input, &gamma, &beta, eps)
            .expect("test: layer_norm_optimized must succeed on real Metal hardware");
        let result_data = result
            .as_slice()
            .expect("test: layer_norm_optimized result must be CPU-resident");

        let mut expected = vec![0.0f32; batch_size * feature_size];
        for b in 0..batch_size {
            let row = &input_data[b * feature_size..(b + 1) * feature_size];
            let mean = row.iter().sum::<f32>() / feature_size as f32;
            let var =
                row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / feature_size as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for f in 0..feature_size {
                expected[b * feature_size + f] =
                    (row[f] - mean) * inv_std * gamma_data[f] + beta_data[f];
            }
        }

        assert_eq!(result.shape().dims(), &[batch_size, feature_size]);
        assert_eq!(result_data.len(), expected.len());
        for (idx, (&got, &want)) in result_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-2,
                "layer_norm_optimized mismatch at index {idx}: got {got}, want {want}"
            );
        }
    }

    /// The `layer_norm` shader's single-threadgroup-per-row design caps
    /// `feature_size` at the device's max threads-per-threadgroup. Exceeding it
    /// must return an honest `Err`, not silently truncate the row or crash.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_layer_norm_optimized_rejects_oversized_hidden_dim() {
        let Some(mut device) =
            metal_device_or_skip("test_layer_norm_optimized_rejects_oversized_hidden_dim")
        else {
            return;
        };

        // Comfortably larger than any Apple Silicon max-threads-per-threadgroup
        // (1024 on every Apple GPU family to date).
        let batch_size = 1usize;
        let feature_size = 4096usize;

        let input = Tensor::from_vec(
            vec![0.0f32; batch_size * feature_size],
            &[batch_size, feature_size],
        )
        .expect("test: create input");
        let gamma = Tensor::from_vec(vec![1.0f32; feature_size], &[feature_size])
            .expect("test: create gamma");
        let beta = Tensor::from_vec(vec![0.0f32; feature_size], &[feature_size])
            .expect("test: create beta");

        let result = device.layer_norm_optimized(&input, &gamma, &beta, 1e-5);
        assert!(
            result.is_err(),
            "layer_norm_optimized with an oversized hidden dim must return an honest Err, \
             not crash or silently truncate"
        );
    }
}
