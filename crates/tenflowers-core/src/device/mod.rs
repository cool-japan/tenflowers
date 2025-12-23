pub mod async_execution;
pub mod context;
pub mod placement;

pub use context::{
    CpuContext, DeviceAllocator, DeviceContext, DeviceKernel, DeviceManager, DeviceProperties,
    DeviceStream, KernelArgs, KernelParam, DEVICE_MANAGER,
};

#[cfg(feature = "gpu")]
pub use context::{get_gpu_context, GpuContext, GpuContextInfo};

#[cfg(any(feature = "gpu", feature = "cudnn"))]
pub use context::{get_enhanced_gpu_context, EnhancedGpuContext, GpuBackend};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum Device {
    #[default]
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu(usize),
    #[cfg(feature = "rocm")]
    Rocm(usize),
}

impl Device {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    #[cfg(feature = "gpu")]
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Gpu(_))
    }

    #[cfg(feature = "rocm")]
    pub fn is_rocm(&self) -> bool {
        matches!(self, Device::Rocm(_))
    }

    pub fn id(&self) -> usize {
        match self {
            Device::Cpu => 0,
            #[cfg(feature = "gpu")]
            Device::Gpu(id) => *id,
            #[cfg(feature = "rocm")]
            Device::Rocm(id) => *id,
        }
    }

    /// Parse a device string (e.g., "cpu", "gpu:0", "gpu:1")
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, String> {
        let s = s.trim().to_lowercase();

        if s == "cpu" {
            return Ok(Device::Cpu);
        }

        #[cfg(feature = "gpu")]
        {
            if s.starts_with("gpu") {
                if s == "gpu" {
                    return Ok(Device::Gpu(0));
                }
                if let Some(id_str) = s.strip_prefix("gpu:") {
                    match id_str.parse::<usize>() {
                        Ok(id) => return Ok(Device::Gpu(id)),
                        Err(_) => return Err(format!("Invalid GPU ID: {}", id_str)),
                    }
                }
            }
        }

        #[cfg(feature = "rocm")]
        {
            if s.starts_with("rocm") || s.starts_with("amd") {
                if s == "rocm" || s == "amd" {
                    return Ok(Device::Rocm(0));
                }
                if let Some(id_str) = s.strip_prefix("rocm:") {
                    match id_str.parse::<usize>() {
                        Ok(id) => return Ok(Device::Rocm(id)),
                        Err(_) => return Err(format!("Invalid ROCm device ID: {}", id_str)),
                    }
                }
                if let Some(id_str) = s.strip_prefix("amd:") {
                    match id_str.parse::<usize>() {
                        Ok(id) => return Ok(Device::Rocm(id)),
                        Err(_) => return Err(format!("Invalid AMD GPU ID: {}", id_str)),
                    }
                }
            }
        }

        Err(format!("Invalid device string: {s}"))
    }

    /// Get the best available GPU device
    #[cfg(feature = "gpu")]
    pub fn best_gpu() -> Result<Self, String> {
        // Try to get GPU 0 as the default best GPU
        Self::try_gpu(0)
    }

    /// Try to create a GPU device with the specified ID
    #[cfg(feature = "gpu")]
    pub fn try_gpu(gpu_id: usize) -> Result<Self, String> {
        // For now, assume GPU is available - in a full implementation,
        // this would check actual GPU availability
        Ok(Device::Gpu(gpu_id))
    }

    /// Get the best available GPU device (CPU fallback when GPU not available)
    #[cfg(not(feature = "gpu"))]
    pub fn best_gpu() -> Result<Self, String> {
        Err("GPU support not compiled".to_string())
    }

    /// Try to create a GPU device (CPU fallback when GPU not available)
    #[cfg(not(feature = "gpu"))]
    pub fn try_gpu(_gpu_id: usize) -> Result<Self, String> {
        Err("GPU support not compiled".to_string())
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            #[cfg(feature = "gpu")]
            Device::Gpu(id) => write!(f, "gpu:{}", id),
            #[cfg(feature = "rocm")]
            Device::Rocm(id) => write!(f, "rocm:{}", id),
        }
    }
}
