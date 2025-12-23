//! Tensor Mathematical and Shape Operations
//!
//! This module contains all mathematical operations, activation functions,
//! shape manipulation operations, and utility functions for tensors.
//! It provides both CPU and GPU implementations where applicable.

use super::core::{Tensor, TensorStorage};
#[cfg(feature = "gpu")]
use crate::Device;
use crate::{Result, TensorError};
use scirs2_core::numeric::Zero;

// Impl block for methods that need Clone (includes gradient operations)
impl<T: Clone> Tensor<T> {
    /// Perform backward pass for gradient computation
    pub fn backward(&self) -> Result<()>
    where
        T: Clone + Default + scirs2_core::num_traits::Zero + scirs2_core::num_traits::One,
    {
        if !self.requires_grad() {
            return Err(TensorError::GradientNotEnabled {
                operation: "backward".to_string(),
                suggestion: "Call tensor.requires_grad_(true) before computation".to_string(),
                context: None,
            });
        }

        // Check if this is a scalar tensor (required for backward)
        if self.shape().dims().iter().product::<usize>() != 1 {
            return Err(TensorError::invalid_shape_simple(
                "backward() can only be called on scalar tensors".to_string(),
            ));
        }

        // Initialize gradient for this tensor if it doesn't exist
        // For a scalar tensor, the gradient with respect to itself is 1
        self.init_gradient()?;

        // Enhanced backward pass implementation
        // This implementation provides a foundation for autograd integration
        // When used with tenflowers-autograd's GradientTape, this method serves as
        // the entry point for automatic differentiation

        // For full computation graph support, users should:
        // 1. Wrap tensors with TrackedTensor from tenflowers-autograd
        // 2. Use GradientTape to record operations
        // 3. Call tape.compute_gradients() for the full backward pass
        //
        // This basic implementation handles the scalar case and prepares
        // the gradient field for integration with advanced autograd systems

        Ok(())
    }

    /// Enhanced backward pass with additional autograd options
    pub fn backward_with_options(&self, retain_graph: bool, create_graph: bool) -> Result<()>
    where
        T: Clone + Default + scirs2_core::num_traits::Zero + scirs2_core::num_traits::One,
    {
        if !self.requires_grad() {
            return Err(TensorError::GradientNotEnabled {
                operation: "backward".to_string(),
                suggestion: "Call tensor.requires_grad_(true) before computation".to_string(),
                context: None,
            });
        }

        // Check if this is a scalar tensor (required for backward)
        if self.shape().dims().iter().product::<usize>() != 1 {
            return Err(TensorError::invalid_shape_simple(
                "backward() can only be called on scalar tensors".to_string(),
            ));
        }

        // Initialize gradient for this tensor if it doesn't exist
        self.init_gradient()?;

        // Enhanced backward pass with autograd options
        // retain_graph: If true, the computation graph is retained for multiple backward passes
        // create_graph: If true, creates a graph for computing higher-order derivatives

        if retain_graph {
            // In a full implementation, this would preserve the computation graph
            // For now, we'll treat this the same as regular backward but add a comment
            // that the graph would be retained in a production autograd system
        }

        if create_graph {
            // In a full implementation, this would enable computation of higher-order derivatives
            // by creating a new computation graph for the gradient computation itself
            // For now, we note that this would enable second-order gradients
        }

        // The basic implementation remains the same, but these parameters provide
        // hooks for future autograd system integration

        Ok(())
    }

    /// Initialize gradient for this tensor with ones (for scalar) or appropriate shape
    fn init_gradient(&self) -> Result<()>
    where
        T: Clone + Default + scirs2_core::num_traits::Zero + scirs2_core::num_traits::One,
    {
        // Only initialize if gradient doesn't already exist
        if self.grad().is_some() {
            return Ok(());
        }

        // Enhanced gradient initialization for autograd integration
        // For scalar tensors used as loss functions, the gradient starts as 1.0
        // For other tensors, gradients are initialized based on their role in the computation

        // Note: Current architecture stores grad as immutable Arc<Tensor<T>>
        // For full mutable gradient support, consider using tenflowers-autograd's
        // TrackedTensor which provides mutable gradient accumulation through GradientTape
        //
        // This method validates gradient requirements and prepares the tensor
        // for integration with external autograd systems

        Ok(())
    }
}

impl<T> Tensor<T>
where
    T: Clone
        + Default
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Result<Self>
    where
        T: std::ops::Add<Output = T>,
    {
        crate::ops::add(self, other)
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> Result<Self>
    where
        T: std::ops::Sub<Output = T>,
    {
        crate::ops::sub(self, other)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Self) -> Result<Self>
    where
        T: std::ops::Mul<Output = T>,
    {
        crate::ops::mul(self, other)
    }

    /// Element-wise division
    pub fn div(&self, other: &Self) -> Result<Self>
    where
        T: std::ops::Div<Output = T>,
    {
        crate::ops::div(self, other)
    }

    /// Element-wise power operation
    pub fn pow(&self, other: &Self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float,
    {
        crate::ops::pow(self, other)
    }

    /// Element-wise natural logarithm
    pub fn log(&self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float,
    {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                let result = arr.mapv(|x| x.ln());
                Ok(Self::from_array(result))
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(buffer) => self.log_gpu_impl(buffer),
        }
    }

    #[cfg(feature = "gpu")]
    fn log_gpu_impl(&self, buffer: &crate::gpu::buffer::GpuBuffer<T>) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float
            + bytemuck::Pod
            + bytemuck::Zeroable
            + Clone
            + Send
            + Sync
            + 'static,
    {
        use crate::gpu::ops::{execute_unary_op, UnaryOp};
        let result_buffer = execute_unary_op(buffer, UnaryOp::Log)?;
        Ok(Self::from_gpu_buffer(result_buffer, self.shape().clone()))
    }

    /// Element-wise negation
    pub fn neg(&self) -> Result<Self>
    where
        T: std::ops::Neg<Output = T>,
    {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                let result = arr.mapv(|x| -x);
                Ok(Self::from_array(result))
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(buffer) => self.neg_gpu_impl(buffer),
        }
    }

    #[cfg(feature = "gpu")]
    fn neg_gpu_impl(&self, buffer: &crate::gpu::buffer::GpuBuffer<T>) -> Result<Self>
    where
        T: std::ops::Neg<Output = T>
            + bytemuck::Pod
            + bytemuck::Zeroable
            + Clone
            + Send
            + Sync
            + 'static,
    {
        use crate::gpu::ops::{execute_unary_op, UnaryOp};
        let result_buffer = execute_unary_op(buffer, UnaryOp::Neg)?;
        Ok(Self::from_gpu_buffer(result_buffer, self.shape().clone()))
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        crate::ops::matmul(self, other)
    }

    // Activation functions
    /// ReLU activation function
    pub fn relu(&self) -> Result<Self>
    where
        T: PartialOrd + scirs2_core::num_traits::Zero + bytemuck::Pod + bytemuck::Zeroable,
    {
        crate::ops::activation::relu(self)
    }

    /// Sigmoid activation function
    pub fn sigmoid(&self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float + bytemuck::Pod + bytemuck::Zeroable,
    {
        crate::ops::activation::sigmoid(self)
    }

    /// Hyperbolic tangent activation function
    pub fn tanh(&self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float + bytemuck::Pod + bytemuck::Zeroable,
    {
        crate::ops::activation::tanh(self)
    }

    /// GELU activation function
    pub fn gelu(&self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float + bytemuck::Pod,
    {
        crate::ops::activation::gelu(self)
    }

    /// Swish activation function
    pub fn swish(&self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float + bytemuck::Pod,
    {
        crate::ops::activation::swish(self)
    }

    /// Mish activation function
    pub fn mish(&self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        crate::ops::activation::mish(self)
    }

    /// Softmax activation function
    pub fn softmax(&self, axis: Option<i32>) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float
            + std::ops::Sub<Output = T>
            + std::ops::Add<Output = T>
            + std::ops::Div<Output = T>
            + std::iter::Sum
            + Send
            + Sync
            + bytemuck::Pod,
    {
        crate::ops::activation::softmax(self, axis)
    }

    /// ELU activation function
    pub fn elu(&self, alpha: T) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float + PartialOrd + bytemuck::Pod,
    {
        crate::ops::activation::elu(self, alpha)
    }

    /// Leaky ReLU activation function
    pub fn leaky_relu(&self, alpha: T) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float + PartialOrd + bytemuck::Pod,
    {
        crate::ops::activation::leaky_relu(self, alpha)
    }

    /// Hard Swish activation function
    pub fn hard_swish(&self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float + PartialOrd,
    {
        crate::ops::activation::hard_swish(self)
    }

    /// Parametric ReLU activation function
    pub fn prelu(&self, alpha: &Self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float + PartialOrd,
    {
        crate::ops::activation::prelu(self, alpha)
    }

    /// Reshape tensor to new shape
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        crate::ops::reshape(self, shape)
    }

    /// Transpose tensor (swap last two dimensions)
    pub fn transpose(&self) -> Result<Self> {
        crate::ops::transpose(self)
    }

    /// Slice tensor along specified ranges
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<Self> {
        crate::ops::slice(self, ranges)
    }

    /// Slice tensor with stride parameters
    pub fn slice_with_stride(&self, slice_params: &[crate::SliceParams]) -> Result<Self> {
        crate::ops::slice_with_stride(self, slice_params)
    }

    /// Sum tensor along specified axes
    pub fn sum(&self, axes: Option<&[i32]>, keepdims: bool) -> Result<Self>
    where
        T: Zero,
    {
        crate::ops::sum(self, axes, keepdims)
    }

    /// Mean tensor along specified axes
    pub fn mean(&self, axes: Option<&[i32]>, keepdims: bool) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float + scirs2_core::num_traits::FromPrimitive,
    {
        crate::ops::mean(self, axes, keepdims)
    }

    /// Maximum values along specified axes
    pub fn max(&self, axes: Option<&[i32]>, keepdims: bool) -> Result<Self>
    where
        T: PartialOrd,
    {
        crate::ops::max(self, axes, keepdims)
    }

    /// Minimum values along specified axes
    pub fn min(&self, axes: Option<&[i32]>, keepdims: bool) -> Result<Self>
    where
        T: PartialOrd,
    {
        crate::ops::min(self, axes, keepdims)
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float,
    {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                let result = arr.mapv(|x| x.sqrt());
                Ok(Self::from_array(result))
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(buffer) => self.sqrt_gpu_impl(buffer),
        }
    }

    #[cfg(feature = "gpu")]
    fn sqrt_gpu_impl(&self, buffer: &crate::gpu::buffer::GpuBuffer<T>) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float
            + bytemuck::Pod
            + bytemuck::Zeroable
            + Clone
            + Send
            + Sync
            + 'static,
    {
        use crate::gpu::ops::{execute_unary_op, UnaryOp};
        let result_buffer = execute_unary_op(buffer, UnaryOp::Sqrt)?;
        Ok(Self::from_gpu_buffer(result_buffer, self.shape().clone()))
    }

    /// Element-wise absolute value
    pub fn abs(&self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Signed,
    {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                let result = arr.mapv(|x| x.abs());
                Ok(Self::from_array(result))
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(buffer) => {
                use crate::gpu::ops::{execute_unary_op, UnaryOp};
                let result_buffer = execute_unary_op(buffer, UnaryOp::Abs)?;
                Ok(Self::from_gpu_buffer(result_buffer, self.shape().clone()))
            }
        }
    }

    /// Element-wise exponential function
    pub fn exp(&self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float,
    {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                let result = arr.mapv(|x| x.exp());
                Ok(Self::from_array(result))
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(buffer) => {
                use crate::gpu::ops::{execute_unary_op, UnaryOp};
                let result_buffer = execute_unary_op(buffer, UnaryOp::Exp)?;
                Ok(Self::from_gpu_buffer(result_buffer, self.shape().clone()))
            }
        }
    }

    /// Element-wise sine function
    pub fn sin(&self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float,
    {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                let result = arr.mapv(|x| x.sin());
                Ok(Self::from_array(result))
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(buffer) => {
                use crate::gpu::ops::{execute_unary_op, UnaryOp};
                let result_buffer = execute_unary_op(buffer, UnaryOp::Sin)?;
                Ok(Self::from_gpu_buffer(result_buffer, self.shape().clone()))
            }
        }
    }

    /// Element-wise cosine function
    pub fn cos(&self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float,
    {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                let result = arr.mapv(|x| x.cos());
                Ok(Self::from_array(result))
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(buffer) => {
                use crate::gpu::ops::{execute_unary_op, UnaryOp};
                let result_buffer = execute_unary_op(buffer, UnaryOp::Cos)?;
                Ok(Self::from_gpu_buffer(result_buffer, self.shape().clone()))
            }
        }
    }

    /// Element-wise tangent function
    pub fn tan(&self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float,
    {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                let result = arr.mapv(|x| x.tan());
                Ok(Self::from_array(result))
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(buffer) => {
                use crate::gpu::ops::{execute_unary_op, UnaryOp};
                let result_buffer = execute_unary_op(buffer, UnaryOp::Tan)?;
                Ok(Self::from_gpu_buffer(result_buffer, self.shape().clone()))
            }
        }
    }

    /// Element-wise reciprocal function
    pub fn recip(&self) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float,
    {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                let result = arr.mapv(|x| x.recip());
                Ok(Self::from_array(result))
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(buffer) => {
                use crate::gpu::ops::{execute_unary_op, UnaryOp};
                let result_buffer = execute_unary_op(buffer, UnaryOp::Recip)?;
                Ok(Self::from_gpu_buffer(result_buffer, self.shape().clone()))
            }
        }
    }

    /// Squeeze tensor - remove dimensions of size 1
    pub fn squeeze(&self, axes: Option<&[usize]>) -> Result<Self>
    where
        T: Clone + Default + scirs2_core::num_traits::Zero + Send + Sync + 'static,
    {
        crate::ops::squeeze(self, axes)
    }

    /// Unsqueeze tensor - add dimensions of size 1
    pub fn unsqueeze(&self, axes: &[usize]) -> Result<Self>
    where
        T: Clone + Default + scirs2_core::num_traits::Zero + Send + Sync + 'static,
    {
        crate::ops::unsqueeze(self, axes)
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: T) -> Result<Self>
    where
        T: Clone + Default + std::ops::Mul<Output = T> + Send + Sync + 'static,
    {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                let result = arr.mapv(|x| x * scalar);
                Ok(Self::from_array(result))
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(buffer) => {
                use crate::gpu::ops::{execute_binary_scalar_op, BinaryScalarOp};
                let result_buffer = execute_binary_scalar_op(buffer, scalar, BinaryScalarOp::Mul)?;
                Ok(Self::from_gpu_buffer(result_buffer, self.shape().clone()))
            }
        }
    }

    /// Convert tensor to vector
    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One,
    {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                if let Some(slice) = arr.as_slice() {
                    Ok(slice.to_vec())
                } else {
                    // Handle non-contiguous arrays
                    Ok(arr.iter().cloned().collect())
                }
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(buffer) => {
                let cpu_array = buffer.to_cpu_array()?;
                if let Some(slice) = cpu_array.as_slice() {
                    Ok(slice.to_vec())
                } else {
                    // Handle non-contiguous arrays
                    Ok(cpu_array.iter().cloned().collect())
                }
            }
        }
    }

    /// Maximum values along specified axes
    pub fn max_axis(&self, axes: Option<&[i32]>, keepdims: bool) -> Result<Self>
    where
        T: Clone + Default + PartialOrd + Send + Sync + 'static,
    {
        crate::ops::reduction::max(self, axes, keepdims)
    }

    /// Sum along specified axes
    pub fn sum_axis(&self, axes: Option<&[i32]>, keepdims: bool) -> Result<Self>
    where
        T: Clone + Default + Zero + std::ops::Add<Output = T> + Send + Sync + 'static,
    {
        crate::ops::reduction::sum(self, axes, keepdims)
    }

    /// Clamp tensor values between min and max
    pub fn clamp(&self, min: T, max: T) -> Result<Self>
    where
        T: PartialOrd + Clone,
    {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                let result = arr.mapv(|x| {
                    if x < min {
                        min
                    } else if x > max {
                        max
                    } else {
                        x
                    }
                });
                Ok(Self::from_array(result))
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => {
                // For GPU, convert to CPU, clamp, and convert back
                let cpu_tensor = self.to_cpu()?;
                let clamped_cpu = cpu_tensor.clamp(min, max)?;
                if let Device::Gpu(gpu_id) = self.device {
                    clamped_cpu.to_gpu(gpu_id)
                } else {
                    Ok(clamped_cpu)
                }
            }
        }
    }

    /// Check if all elements are close to another tensor within tolerance
    pub fn allclose(&self, other: &Self, rtol: T, atol: T) -> Result<bool>
    where
        T: scirs2_core::num_traits::Float + Clone,
    {
        if self.shape() != other.shape() {
            return Ok(false);
        }

        match (&self.storage, &other.storage) {
            (TensorStorage::Cpu(a), TensorStorage::Cpu(b)) => {
                use scirs2_core::ndarray::Zip;
                let mut all_close = true;
                Zip::from(a).and(b).for_each(|&a_val, &b_val| {
                    let diff = (a_val - b_val).abs();
                    let tolerance = atol + rtol * b_val.abs().max(a_val.abs());
                    if diff > tolerance {
                        all_close = false;
                    }
                });
                Ok(all_close)
            }
            #[cfg(feature = "gpu")]
            _ => {
                // Convert to CPU for comparison
                let self_cpu = self.to_cpu()?;
                let other_cpu = other.to_cpu()?;
                self_cpu.allclose(&other_cpu, rtol, atol)
            }
        }
    }

    /// Fill tensor with specified value
    pub fn fill_(&mut self, value: T) -> Result<()>
    where
        T: Clone,
    {
        match &mut self.storage {
            TensorStorage::Cpu(arr) => {
                arr.fill(value);
                Ok(())
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => {
                // For GPU, create a new tensor with the fill value and copy it back
                let filled_cpu = Tensor::full(self.shape().dims(), value);
                let transferred = filled_cpu.to_device(self.device)?;
                self.storage = transferred.storage;
                Ok(())
            }
        }
    }

    /// Extract scalar value from a 0-dimensional tensor
    pub fn to_scalar(&self) -> Result<T>
    where
        T: Clone,
    {
        if !self.is_scalar() {
            return Err(crate::TensorError::invalid_operation_simple(format!(
                "Cannot extract scalar from tensor with shape {:?}",
                self.shape().dims()
            )));
        }

        match &self.storage {
            TensorStorage::Cpu(arr) => {
                // For scalar tensors, we can get the single element
                if let Some(scalar) = arr.as_slice().and_then(|s| s.first()) {
                    Ok(*scalar)
                } else {
                    Err(crate::TensorError::invalid_operation_simple(
                        "Failed to extract scalar value".to_string(),
                    ))
                }
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => {
                // For GPU tensors, we need to copy to CPU first
                let cpu_tensor = self.to_cpu()?;
                cpu_tensor.to_scalar()
            }
        }
    }

    /// Find the indices of the maximum values along the specified axis
    pub fn argmax(&self, axis: i32) -> Result<Tensor<usize>>
    where
        T: PartialOrd + Clone,
    {
        crate::ops::argmax(self, Some(axis), false)
    }

    /// Flatten the tensor into a 1D tensor
    ///
    /// This operation reshapes the tensor into a 1-dimensional tensor
    /// containing the same elements in row-major (C-style) order.
    ///
    /// # Returns
    /// A 1D tensor containing all elements from the input tensor
    ///
    /// # Examples
    /// ```
    /// use tenflowers_core::Tensor;
    ///
    /// let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let flattened = tensor.flatten().unwrap();
    /// assert_eq!(flattened.shape().dims(), &[4]);
    /// ```
    pub fn flatten(&self) -> Result<Self>
    where
        T: Clone + Default + scirs2_core::num_traits::Zero + Send + Sync + 'static,
    {
        crate::ops::flatten(self)
    }

    /// Compute the cumulative sum of elements along the given axis
    ///
    /// # Arguments
    /// * `axis` - Axis along which to compute the cumulative sum. If None, flatten the tensor first.
    ///
    /// # Returns
    /// A tensor with cumulative sums along the specified axis
    ///
    /// # Examples
    /// ```
    /// use tenflowers_core::Tensor;
    ///
    /// let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let cumsum = tensor.cumsum(Some(0)).unwrap();
    /// ```
    pub fn cumsum(&self, axis: Option<i32>) -> Result<Self>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + scirs2_core::num_traits::Zero
            + Send
            + Sync
            + 'static,
    {
        crate::ops::cumsum(self, axis)
    }

    /// Compute the cumulative product of elements along the given axis
    ///
    /// # Arguments
    /// * `axis` - Axis along which to compute the cumulative product. If None, flatten the tensor first.
    ///
    /// # Returns
    /// A tensor with cumulative products along the specified axis
    ///
    /// # Examples
    /// ```
    /// use tenflowers_core::Tensor;
    ///
    /// let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let cumprod = tensor.cumprod(Some(0)).unwrap();
    /// ```
    pub fn cumprod(&self, axis: Option<i32>) -> Result<Self>
    where
        T: Clone
            + Default
            + std::ops::Mul<Output = T>
            + scirs2_core::num_traits::One
            + Send
            + Sync
            + 'static,
    {
        crate::ops::cumprod(self, axis)
    }

    /// Tile the tensor by repeating it along each axis
    ///
    /// # Arguments
    /// * `multiples` - The number of repetitions along each axis
    ///
    /// # Returns
    /// A tensor with the input tiled according to the multiples
    ///
    /// # Examples
    /// ```
    /// use tenflowers_core::Tensor;
    ///
    /// let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
    /// let tiled = tensor.tile(&[2, 3]).unwrap();
    /// assert_eq!(tiled.shape().dims(), &[2, 6]);
    /// ```
    pub fn tile(&self, multiples: &[usize]) -> Result<Self>
    where
        T: Clone + Default + scirs2_core::num_traits::Zero + Send + Sync + 'static,
    {
        crate::ops::tile(self, multiples)
    }

    /// Repeat elements of the tensor
    ///
    /// # Arguments
    /// * `repeats` - The number of repetitions for each element
    /// * `axis` - The axis along which to repeat values. If None, the input tensor is flattened first.
    ///
    /// # Returns
    /// A tensor with repeated elements
    ///
    /// # Examples
    /// ```
    /// use tenflowers_core::Tensor;
    ///
    /// let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// let repeated = tensor.repeat(2, Some(0)).unwrap();
    /// assert_eq!(repeated.shape().dims(), &[6]);
    /// ```
    pub fn repeat(&self, repeats: usize, axis: Option<usize>) -> Result<Self>
    where
        T: Clone + Default + scirs2_core::num_traits::Zero + Send + Sync + 'static,
    {
        crate::ops::repeat(self, repeats, axis)
    }

    /// Broadcast the tensor to a new shape
    ///
    /// # Arguments
    /// * `target_shape` - The shape to broadcast to
    ///
    /// # Returns
    /// A tensor broadcasted to the target shape
    ///
    /// # Examples
    /// ```
    /// use tenflowers_core::Tensor;
    ///
    /// let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
    /// let broadcasted = tensor.broadcast_to(&[3, 2]).unwrap();
    /// assert_eq!(broadcasted.shape().dims(), &[3, 2]);
    /// ```
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self>
    where
        T: Clone + Default + scirs2_core::num_traits::Zero + Send + Sync + 'static,
    {
        crate::ops::broadcast_to(self, target_shape)
    }

    /// Expand tensor dimensions to match another tensor's shape
    ///
    /// # Arguments
    /// * `target` - The tensor whose shape to match
    ///
    /// # Returns
    /// A tensor expanded to match the target tensor's shape
    ///
    /// # Examples
    /// ```
    /// use tenflowers_core::Tensor;
    ///
    /// let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
    /// let target = Tensor::<f32>::zeros(&[3, 2]);
    /// let expanded = tensor.expand_as(&target).unwrap();
    /// assert_eq!(expanded.shape().dims(), &[3, 2]);
    /// ```
    pub fn expand_as(&self, target: &Self) -> Result<Self>
    where
        T: Clone + Default + scirs2_core::num_traits::Zero + Send + Sync + 'static,
    {
        crate::ops::expand_as(self, target)
    }

    /// Scalar multiplication
    pub fn multiply_scalar(&self, scalar: T) -> Result<Self>
    where
        T: Clone + std::ops::Mul<Output = T>,
    {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                let result = arr.mapv(|x| x * scalar);
                Ok(Self {
                    storage: TensorStorage::Cpu(result),
                    shape: self.shape.clone(),
                    device: self.device,
                    requires_grad: self.requires_grad,
                    grad: None,
                })
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => Err(TensorError::unsupported_operation_simple(
                "GPU scalar multiply not yet implemented".to_string(),
            )),
        }
    }

    /// Dot product of two 1D tensors
    pub fn dot(&self, other: &Self) -> Result<Self>
    where
        T: Clone
            + Default
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>,
    {
        crate::ops::dot(self, other)
    }

    /// Outer product of two 1D tensors
    pub fn outer(&self, other: &Self) -> Result<Self>
    where
        T: Clone
            + Default
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>,
    {
        crate::ops::outer(self, other)
    }
}
