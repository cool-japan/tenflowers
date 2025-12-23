//! Device Management and Transfer Operations
//!
//! This module handles tensor device placement, transfers between devices
//! (CPU/GPU), and device-specific operations. It provides efficient
//! device-to-device data transfer capabilities.

use super::core::{Tensor, TensorStorage};
use crate::{Device, Result};

// Impl block for methods that need Clone
impl<T: Clone> Tensor<T> {
    /// Transfer tensor to specified device
    pub fn to(&self, device: Device) -> Result<Self>
    where
        T: Default + bytemuck::Pod + bytemuck::Zeroable + Send + Sync + 'static,
    {
        if self.device() == &device {
            return Ok(self.clone());
        }

        match (&self.storage, &device) {
            (TensorStorage::Cpu(_array), Device::Cpu) => Ok(self.clone()),
            #[cfg(feature = "gpu")]
            (TensorStorage::Cpu(array), Device::Gpu(id)) => {
                let gpu_buffer = crate::gpu::buffer::GpuBuffer::from_cpu_array(array, *id)?;
                Ok(Self {
                    storage: TensorStorage::Gpu(gpu_buffer),
                    shape: self.shape().clone(),
                    device,
                    requires_grad: self.requires_grad(),
                    grad: None,
                })
            }
            #[cfg(feature = "gpu")]
            (TensorStorage::Gpu(buffer), Device::Cpu) => {
                let array = buffer.to_cpu_array()?;
                Ok(Self {
                    storage: TensorStorage::Cpu(array),
                    shape: self.shape().clone(),
                    device,
                    requires_grad: self.requires_grad(),
                    grad: None,
                })
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Transfer tensor to a different device
    pub fn to_device(&self, target_device: Device) -> Result<Self>
    where
        T: Clone + Default + Send + Sync + 'static + bytemuck::Pod,
    {
        if self.device() == &target_device {
            return Ok(self.clone());
        }

        self.transfer_to_device(target_device)
    }

    /// Move tensor to CPU
    pub fn to_cpu(&self) -> Result<Self>
    where
        T: Clone + Default + Send + Sync + 'static + bytemuck::Pod,
    {
        self.to_device(Device::Cpu)
    }

    /// Get GPU context information from this tensor (if it's on GPU)
    #[cfg(feature = "gpu")]
    pub fn gpu_context_info(&self) -> Option<crate::device::context::GpuContextInfo> {
        match &self.storage {
            TensorStorage::Gpu(buffer) => Some(crate::device::context::GpuContextInfo {
                device: buffer.device.clone(),
                queue: buffer.queue.clone(),
            }),
            _ => None,
        }
    }

    /// Move tensor to GPU with specified ID
    #[cfg(feature = "gpu")]
    pub fn to_gpu(&self, gpu_id: usize) -> Result<Self>
    where
        T: Clone + Default + Send + Sync + 'static + bytemuck::Pod,
    {
        self.to_device(Device::Gpu(gpu_id))
    }

    /// Internal device transfer implementation
    fn transfer_to_device(&self, target_device: Device) -> Result<Self>
    where
        T: Clone + Default + Send + Sync + 'static + bytemuck::Pod,
    {
        use crate::device::context::DEVICE_MANAGER;

        let _src_ctx = DEVICE_MANAGER.get_context(self.device())?;
        let _dst_ctx = DEVICE_MANAGER.get_context(&target_device)?;

        match (&self.storage, &target_device) {
            // CPU to GPU transfer
            #[cfg(feature = "gpu")]
            (TensorStorage::Cpu(cpu_array), Device::Gpu(_)) => {
                #[cfg(feature = "gpu")]
                {
                    let slice = cpu_array.as_slice().ok_or_else(|| {
                        crate::TensorError::invalid_argument(
                            "Cannot convert CPU array to slice".to_string(),
                        )
                    })?;

                    let gpu_buffer =
                        crate::gpu::buffer::GpuBuffer::from_slice(slice, &target_device)?;

                    Ok(Self {
                        storage: TensorStorage::Gpu(gpu_buffer),
                        shape: self.shape().clone(),
                        device: target_device,
                        requires_grad: self.requires_grad(),
                        grad: None, // Gradients are reset on device transfer
                    })
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Err(crate::TensorError::device_error_simple(
                        "GPU support not compiled",
                    ))
                }
            }

            // GPU to CPU transfer
            #[cfg(feature = "gpu")]
            (TensorStorage::Gpu(gpu_buffer), Device::Cpu) => {
                let cpu_data = gpu_buffer.to_cpu()?;
                let array = scirs2_core::ndarray::ArrayD::from_shape_vec(
                    scirs2_core::ndarray::IxDyn(self.shape().dims()),
                    cpu_data,
                )
                .map_err(|e| crate::TensorError::invalid_shape_simple(e.to_string()))?;

                Ok(Self {
                    storage: TensorStorage::Cpu(array),
                    shape: self.shape().clone(),
                    device: target_device,
                    requires_grad: self.requires_grad(),
                    grad: None, // Gradients are reset on device transfer
                })
            }

            // GPU to GPU transfer (device-to-device)
            #[cfg(feature = "gpu")]
            (TensorStorage::Gpu(src_buffer), Device::Gpu(_)) => {
                let dst_buffer = src_buffer.transfer_to_device(&target_device)?;

                Ok(Self {
                    storage: TensorStorage::Gpu(dst_buffer),
                    shape: self.shape().clone(),
                    device: target_device,
                    requires_grad: self.requires_grad(),
                    grad: None, // Gradients are reset on device transfer
                })
            }

            // CPU to CPU (should not happen due to early return)
            (TensorStorage::Cpu(_), Device::Cpu) => Ok(self.clone()),

            // ROCm patterns (not yet implemented)
            #[cfg(feature = "rocm")]
            (TensorStorage::Cpu(_), Device::Rocm(_)) => {
                todo!("CPU to ROCm transfer not yet implemented")
            }
            #[cfg(feature = "rocm")]
            (TensorStorage::Gpu(_), Device::Rocm(_)) => {
                todo!("GPU to ROCm transfer not yet implemented")
            }
        }
    }

    /// Copy tensor data from another device (for collective operations)
    pub fn copy_from_device(&mut self, src: &Self) -> Result<()>
    where
        T: Clone + Default + Send + Sync + 'static + bytemuck::Pod,
    {
        if self.shape() != src.shape() {
            return Err(crate::TensorError::ShapeMismatch {
                operation: "copy_from_device".to_string(),
                expected: self.shape().to_string(),
                got: src.shape().to_string(),
                context: None,
            });
        }

        let transferred = src.transfer_to_device(*self.device())?;
        self.storage = transferred.storage;

        Ok(())
    }

    /// Check if tensor can be transferred to target device
    pub fn can_transfer_to(&self, target_device: Device) -> bool
    where
        T: bytemuck::Pod,
    {
        match (self.device(), &target_device) {
            (Device::Cpu, Device::Cpu) => true,
            #[cfg(feature = "gpu")]
            (Device::Cpu, Device::Gpu(_)) => true,
            #[cfg(feature = "gpu")]
            (Device::Gpu(_), Device::Cpu) => true,
            #[cfg(feature = "gpu")]
            (Device::Gpu(_), Device::Gpu(_)) => true,
            #[cfg(feature = "rocm")]
            (Device::Rocm(_), _) => true,
            #[cfg(feature = "rocm")]
            (_, Device::Rocm(_)) => true,
        }
    }
}
