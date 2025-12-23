//! Core Tensor Structure and Properties
//!
//! This module contains the fundamental tensor structure, storage definition,
//! and basic property access methods. It provides the foundation for all
//! tensor operations while maintaining clean separation of concerns.

use crate::{Device, Result, Shape};
use scirs2_core::ndarray::ArrayD;
use std::sync::Arc;

/// Core tensor structure that holds data and metadata
#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub storage: TensorStorage<T>,
    pub(in crate::tensor) shape: Shape,
    pub(in crate::tensor) device: Device,
    pub(in crate::tensor) requires_grad: bool,
    pub(in crate::tensor) grad: Option<Arc<Tensor<T>>>,
}

/// Storage abstraction for different device types
#[derive(Debug, Clone)]
pub enum TensorStorage<T> {
    Cpu(ArrayD<T>),
    #[cfg(feature = "gpu")]
    Gpu(crate::gpu::buffer::GpuBuffer<T>),
}

// Core implementation block for all tensor types
impl<T> Tensor<T> {
    /// Get the shape of the tensor
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the device where the tensor is located
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the data type of the tensor
    pub fn dtype(&self) -> crate::DType
    where
        T: 'static,
    {
        crate::dtype_from_type::<T>()
    }

    /// Check if tensor requires gradient computation
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set whether tensor requires gradient computation
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Get the gradient tensor if it exists
    pub fn grad(&self) -> Option<&Tensor<T>> {
        self.grad.as_ref().map(|g| g.as_ref())
    }

    /// Set the gradient tensor
    pub fn set_grad(&mut self, grad: Option<Tensor<T>>) {
        self.grad = grad.map(Arc::new);
    }

    /// Get a reference to the underlying data (CPU only)
    pub fn data(&self) -> &[T] {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                arr.as_slice().unwrap_or_else(|| {
                    panic!("Tensor data is not contiguous. Use to_owned() or iter() for non-contiguous access.")
                })
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => {
                panic!("Cannot access GPU tensor data directly. Use to_cpu() first.")
            }
        }
    }

    /// Get the value at a specific index (for CPU tensors only)
    pub fn get(&self, index: &[usize]) -> Option<T>
    where
        T: Clone,
    {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                if index.len() != arr.ndim() {
                    return None;
                }
                arr.get(index).cloned()
            }
            #[cfg(feature = "gpu")]
            _ => None,
        }
    }

    /// Get the underlying data as a slice (CPU tensors only)
    pub fn as_slice(&self) -> Option<&[T]> {
        match &self.storage {
            TensorStorage::Cpu(array) => array.as_slice(),
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => None,
        }
    }

    /// Check if tensor is empty (has no elements)
    pub fn is_empty(&self) -> bool {
        self.shape.elements() == 0
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let element_size = std::mem::size_of::<T>();
        self.shape.elements() * element_size
    }

    /// Check if two tensors have the same shape
    pub fn same_shape(&self, other: &Self) -> bool {
        self.shape == other.shape
    }

    /// Check if tensors are broadcastable
    pub fn is_broadcastable_with(&self, other: &Self) -> bool {
        let dims1 = self.shape.dims();
        let dims2 = other.shape.dims();

        let max_dims = dims1.len().max(dims2.len());

        for i in 0..max_dims {
            let dim1 = dims1
                .get(dims1.len().saturating_sub(i + 1))
                .copied()
                .unwrap_or(1);
            let dim2 = dims2
                .get(dims2.len().saturating_sub(i + 1))
                .copied()
                .unwrap_or(1);

            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return false;
            }
        }

        true
    }

    /// Get tensor summary statistics as a formatted string
    pub fn summary(&self) -> String
    where
        T: std::fmt::Display + Clone,
    {
        format!(
            "Tensor<{}>: shape={:?}, device={:?}, numel={}, memory={}B, requires_grad={}",
            std::any::type_name::<T>(),
            self.shape.dims(),
            self.device,
            self.shape.elements(),
            self.memory_usage(),
            self.requires_grad
        )
    }

    /// Get the total number of elements (alias for size)
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.size()
    }

    /// Get the number of dimensions (rank)
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Get the number of dimensions (alias for rank)
    pub fn ndim(&self) -> usize {
        self.shape.rank()
    }

    /// Check if tensor is a scalar (0-dimensional)
    pub fn is_scalar(&self) -> bool {
        self.shape.rank() == 0
    }

    /// Check if tensor is a vector (1-dimensional)
    pub fn is_vector(&self) -> bool {
        self.shape.rank() == 1
    }

    /// Check if tensor is a matrix (2-dimensional)
    pub fn is_matrix(&self) -> bool {
        self.shape.rank() == 2
    }

    /// Check if tensor data is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        match &self.storage {
            TensorStorage::Cpu(arr) => arr.is_standard_layout(),
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => true, // GPU buffers are always contiguous
        }
    }
}

// Separate impl block for methods requiring Pod bounds
impl<T> Tensor<T>
where
    T: Clone + bytemuck::Pod + bytemuck::Zeroable + Send + Sync + 'static,
{
    /// Apply a function to each element of the tensor
    pub fn map_inplace<F>(&mut self, f: F) -> Result<()>
    where
        F: Fn(&T) -> T,
    {
        match &mut self.storage {
            TensorStorage::Cpu(arr) => {
                arr.mapv_inplace(|x| f(&x));
                Ok(())
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(buffer) => {
                // Handle GPU case manually to avoid double borrow
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                    || std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
                {
                    // For f32/f64, convert to CPU, apply operation, and convert back
                    let mut cpu_array = buffer.to_cpu_array()?;
                    cpu_array.mapv_inplace(|x| f(&x));
                    let device_id = match self.device {
                        crate::Device::Gpu(id) => id,
                        _ => {
                            return Err(crate::TensorError::device_error_simple(
                                "Expected GPU device".to_string(),
                            ))
                        }
                    };
                    let new_gpu_buffer =
                        crate::gpu::buffer::GpuBuffer::from_cpu_array(&cpu_array, device_id)?;
                    *buffer = new_gpu_buffer;
                    Ok(())
                } else {
                    // Fallback: not supported for this type
                    Err(crate::TensorError::unsupported_operation_simple(format!(
                        "GPU map_inplace not supported for type {}",
                        std::any::type_name::<T>()
                    )))
                }
            }
        }
    }
}
