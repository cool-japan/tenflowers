use crate::{Result, Tensor, TensorError};
use num_traits::Float;

/// GPU implementation of softmax for f32 tensors
#[cfg(feature = "gpu")]
pub fn softmax_gpu_f32<T>(x: &Tensor<T>, axis: Option<i32>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Float
        + std::ops::Sub<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + std::iter::Sum
        + Send
        + Sync,
{
    // Cast to f32 for GPU operations
    let x_f32 = unsafe { std::mem::transmute::<&Tensor<T>, &Tensor<f32>>(x) };

    // Default to last axis if not specified
    let ndim = x.shape().rank();
    let axis = axis.unwrap_or(-1);
    let axis = if axis < 0 {
        (ndim as i32 + axis) as i32
    } else {
        axis
    };

    if axis < 0 || axis >= ndim as i32 {
        return Err(TensorError::InvalidAxis {
            operation: "activation".to_string(),
            axis,
            ndim,
            context: None,
        });
    }

    // Step 1: Find max along axis for numerical stability
    let max_tensor = x_f32.max(Some(&[axis]), true)?;

    // Step 2: Subtract max: x - max (broadcasting)
    let shifted = crate::ops::binary::sub(x_f32, &max_tensor)?;

    // Step 3: Compute exp(x - max)
    let exp_tensor = shifted.exp()?;

    // Step 4: Sum along axis
    let sum_tensor = exp_tensor.sum(Some(&[axis]), true)?;

    // Step 5: Divide: exp(x - max) / sum
    let softmax_f32 = crate::ops::binary::div(&exp_tensor, &sum_tensor)?;

    // Cast result back to T
    let result = unsafe { std::mem::transmute::<Tensor<f32>, Tensor<T>>(softmax_f32) };

    Ok(result)
}

/// GPU-specific activation operation types
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy)]
pub enum GpuActivationOp {
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    Swish,
    Mish,
    ELU,
    LeakyReLU,
    HardSwish,
}

/// Execute GPU activation operation
#[cfg(feature = "gpu")]
pub fn execute_gpu_activation<T>(
    input: &crate::gpu::buffer::GpuBuffer<T>,
    op: GpuActivationOp,
) -> Result<crate::gpu::buffer::GpuBuffer<T>>
where
    T: Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // Placeholder implementation - actual GPU kernels would be implemented here
    use crate::gpu::ops::execute_activation_op;

    let activation_op = match op {
        GpuActivationOp::ReLU => crate::gpu::ops::ActivationOp::ReLU,
        GpuActivationOp::Sigmoid => crate::gpu::ops::ActivationOp::Sigmoid,
        GpuActivationOp::Tanh => crate::gpu::ops::ActivationOp::Tanh,
        GpuActivationOp::GELU => crate::gpu::ops::ActivationOp::GELU,
        GpuActivationOp::Swish => crate::gpu::ops::ActivationOp::Swish,
        GpuActivationOp::Mish => crate::gpu::ops::ActivationOp::Mish,
        GpuActivationOp::ELU => crate::gpu::ops::ActivationOp::ELU,
        GpuActivationOp::LeakyReLU => crate::gpu::ops::ActivationOp::LeakyReLU,
        GpuActivationOp::HardSwish => crate::gpu::ops::ActivationOp::HardSwish,
    };

    execute_activation_op(input, activation_op)
}

/// GPU activation function dispatcher
#[cfg(feature = "gpu")]
pub fn dispatch_gpu_activation<T>(x: &Tensor<T>, op: GpuActivationOp) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    match &x.storage {
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => {
            let result_gpu = execute_gpu_activation(gpu_buffer, op)?;
            Ok(Tensor::from_gpu_buffer(result_gpu, x.shape().clone()))
        }
        _ => Err(TensorError::device_mismatch(
            "gpu_activation",
            "GPU",
            &x.device().to_string(),
        )),
    }
}

/// GPU-optimized batch activation processing
#[cfg(feature = "gpu")]
pub fn batch_gpu_activations<T>(
    inputs: &[&Tensor<T>],
    ops: &[GpuActivationOp],
) -> Result<Vec<Tensor<T>>>
where
    T: Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    if inputs.len() != ops.len() {
        return Err(TensorError::invalid_argument(
            "Input and operations count mismatch".to_string(),
        ));
    }

    let mut results = Vec::with_capacity(inputs.len());

    for (input, &op) in inputs.iter().zip(ops.iter()) {
        let result = dispatch_gpu_activation(input, op)?;
        results.push(result);
    }

    Ok(results)
}

/// GPU memory-efficient activation streaming
#[cfg(feature = "gpu")]
pub struct GpuActivationStream<T> {
    buffer: crate::gpu::buffer::GpuBuffer<T>,
    operations: Vec<GpuActivationOp>,
    current_index: usize,
}

#[cfg(feature = "gpu")]
impl<T> GpuActivationStream<T>
where
    T: Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    pub fn new(
        initial_buffer: crate::gpu::buffer::GpuBuffer<T>,
        operations: Vec<GpuActivationOp>,
    ) -> Self {
        Self {
            buffer: initial_buffer,
            operations,
            current_index: 0,
        }
    }

    pub fn next(&mut self) -> Result<Option<crate::gpu::buffer::GpuBuffer<T>>> {
        if self.current_index >= self.operations.len() {
            return Ok(None);
        }

        let op = self.operations[self.current_index];
        let result = execute_gpu_activation(&self.buffer, op)?;

        // Update buffer for chained operations
        self.buffer = result.clone();
        self.current_index += 1;

        Ok(Some(result))
    }

    pub fn process_all(mut self) -> Result<crate::gpu::buffer::GpuBuffer<T>> {
        while let Some(result) = self.next()? {
            self.buffer = result;
        }
        Ok(self.buffer)
    }
}
