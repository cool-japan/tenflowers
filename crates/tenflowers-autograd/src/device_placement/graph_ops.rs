//! Graph operation utilities for device placement.

use tenflowers_core::Device;

/// Simplified graph operation representation
#[derive(Debug, Clone)]
pub struct GraphOperation<T> {
    pub id: String,
    pub operation_name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub tensor_sizes: Vec<usize>,
    pub _phantom: std::marker::PhantomData<T>,
}

impl<T> GraphOperation<T> {
    pub fn new(
        id: String,
        operation_name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> Self {
        Self {
            id,
            operation_name,
            inputs,
            outputs,
            tensor_sizes: vec![],
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn with_tensor_sizes(mut self, tensor_sizes: Vec<usize>) -> Self {
        self.tensor_sizes = tensor_sizes;
        self
    }
}

/// Helper function to get device name for display
pub(crate) fn device_name(device: &Device) -> &str {
    match device {
        Device::Cpu => "CPU",
        #[cfg(feature = "gpu")]
        Device::Gpu(_) => "GPU",
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => "ROCM",
    }
}
