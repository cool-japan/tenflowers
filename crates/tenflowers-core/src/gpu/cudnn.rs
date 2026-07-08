//! cuDNN integration for optimized neural network operations on NVIDIA GPUs
//!
//! This module provides cuDNN bindings and high-level abstractions for GPU-accelerated
//! neural network operations including convolution, pooling, normalization, and activation
//! functions.

#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
use crate::{Device, Result, Tensor, TensorError};
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
use std::collections::HashMap;
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
use std::sync::Arc;

/// cuDNN handle wrapper for managing cuDNN context.
///
/// Note: construction always fails (see `CudnnHandle::new`) because no cuDNN
/// SDK is linked. Fields are preserved for documentation purposes.
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug)]
pub struct CudnnHandle {
    /// cuDNN handle (cudnnHandle_t equivalent — requires libcudnn.so)
    _handle: *mut std::ffi::c_void,
    device_id: usize,
}

// SAFETY: CudnnHandle is safe to send across threads because cuDNN handles
// are thread-safe when proper synchronization is used. We protect access
// to the global context with a Mutex to ensure thread-safe access.
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
unsafe impl Send for CudnnHandle {}

// SAFETY: CudnnHandle is safe to share between threads when synchronized.
// The cuDNN library is thread-safe, and we use a Mutex to synchronize access.
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
unsafe impl Sync for CudnnHandle {}

/// cuDNN tensor descriptor for describing tensor layout
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone)]
pub struct CudnnTensorDescriptor {
    /// Tensor data type (float, half, etc.)
    data_type: CudnnDataType,
    /// Tensor format (NCHW, NHWC, etc.)
    format: CudnnTensorFormat,
    /// Tensor dimensions
    dimensions: Vec<i32>,
    /// Tensor strides
    strides: Vec<i32>,
}

/// cuDNN data types
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudnnDataType {
    Float,
    Double,
    Half,
    Int8,
    Int32,
    Int8x4,
    Uint8,
    Uint8x4,
    Int8x32,
}

/// cuDNN tensor formats
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudnnTensorFormat {
    NCHW,
    NHWC,
    NCHWVectC,
    NHWCVectC,
}

/// cuDNN activation functions
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudnnActivationMode {
    Sigmoid,
    Relu,
    Tanh,
    ClippedRelu,
    Elu,
    Identity,
    Swish,
}

/// cuDNN convolution descriptor
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone)]
pub struct CudnnConvolutionDescriptor {
    /// Padding in each dimension
    padding: Vec<i32>,
    /// Stride in each dimension
    stride: Vec<i32>,
    /// Dilation in each dimension
    dilation: Vec<i32>,
    /// Convolution mode (cross-correlation vs convolution)
    mode: CudnnConvolutionMode,
    /// Data type for computation
    compute_type: CudnnDataType,
}

/// cuDNN convolution modes
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudnnConvolutionMode {
    Convolution,
    CrossCorrelation,
}

/// cuDNN pooling descriptor
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone)]
pub struct CudnnPoolingDescriptor {
    /// Pooling mode (max, average, etc.)
    mode: CudnnPoolingMode,
    /// NaN propagation mode
    nan_opt: CudnnNanPropagation,
    /// Window size
    window_size: Vec<i32>,
    /// Padding
    padding: Vec<i32>,
    /// Stride
    stride: Vec<i32>,
}

/// cuDNN pooling modes
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudnnPoolingMode {
    Max,
    AverageCountIncludePadding,
    AverageCountExcludePadding,
    MaxDeterministic,
}

/// cuDNN NaN propagation options
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudnnNanPropagation {
    NotPropagateNan,
    PropagateNan,
}

/// High-level cuDNN operations manager
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
pub struct CudnnContext {
    /// cuDNN handles per device
    handles: HashMap<usize, Arc<CudnnHandle>>,
    /// Cached descriptors for performance
    tensor_descriptors: HashMap<String, CudnnTensorDescriptor>,
    convolution_descriptors: HashMap<String, CudnnConvolutionDescriptor>,
    pooling_descriptors: HashMap<String, CudnnPoolingDescriptor>,
}

#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
impl CudnnHandle {
    /// Create a new cuDNN handle for the specified device.
    ///
    /// # Errors
    ///
    /// Always returns `Err` because no cuDNN SDK is linked. Build with the
    /// actual CUDA/cuDNN toolkit present for a real implementation.
    pub fn new(_device_id: usize) -> Result<Self> {
        Err(TensorError::not_implemented_simple(
            "cuDNN handle creation requires libcudnn.so; \
             build with the CUDA/cuDNN toolkit installed and link against it"
                .to_string(),
        ))
    }

    /// Get the device ID for this handle
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Set the CUDA stream for this cuDNN handle.
    ///
    /// # Errors
    ///
    /// Always returns `Err` because no cuDNN SDK is linked.
    pub fn set_stream(&mut self, _stream: *mut std::ffi::c_void) -> Result<()> {
        Err(TensorError::not_implemented_simple(
            "cuDNN stream configuration requires libcudnn.so; \
             build with the CUDA/cuDNN toolkit installed"
                .to_string(),
        ))
    }
}

#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
impl Drop for CudnnHandle {
    fn drop(&mut self) {
        // No cuDNN handle to destroy — CudnnHandle::new always fails, so this
        // path is unreachable in practice. If a handle were created via unsafe
        // code, the caller is responsible for cleanup.
    }
}

#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
impl CudnnTensorDescriptor {
    /// Create a new tensor descriptor
    pub fn new(
        data_type: CudnnDataType,
        format: CudnnTensorFormat,
        dimensions: Vec<i32>,
    ) -> Result<Self> {
        // Calculate strides based on format
        let strides = Self::calculate_strides(&dimensions, format);

        Ok(CudnnTensorDescriptor {
            data_type,
            format,
            dimensions,
            strides,
        })
    }

    /// Create tensor descriptor from TenfloweRS tensor
    pub fn from_tensor<T>(tensor: &Tensor<T>) -> Result<Self>
    where
        T: Clone + Send + Sync + 'static,
    {
        let shape = tensor.shape();
        let dimensions: Vec<i32> = shape.iter().map(|&x| x as i32).collect();

        // Determine data type from T
        let data_type = Self::infer_data_type::<T>()?;

        // Default to NCHW format for now
        let format = CudnnTensorFormat::NCHW;

        Self::new(data_type, format, dimensions)
    }

    /// Calculate strides for the given dimensions and format
    fn calculate_strides(dimensions: &[i32], format: CudnnTensorFormat) -> Vec<i32> {
        let mut strides = vec![0; dimensions.len()];

        match format {
            CudnnTensorFormat::NCHW => {
                // NCHW: N×C×H×W
                if dimensions.len() >= 4 {
                    strides[3] = 1; // W stride
                    strides[2] = dimensions[3]; // H stride
                    strides[1] = dimensions[2] * dimensions[3]; // C stride
                    strides[0] = dimensions[1] * dimensions[2] * dimensions[3]; // N stride
                }
            }
            CudnnTensorFormat::NHWC => {
                // NHWC: N×H×W×C
                if dimensions.len() >= 4 {
                    strides[3] = 1; // C stride
                    strides[2] = dimensions[3]; // W stride
                    strides[1] = dimensions[2] * dimensions[3]; // H stride
                    strides[0] = dimensions[1] * dimensions[2] * dimensions[3]; // N stride
                }
            }
            _ => {
                // For other formats, use default row-major ordering
                let mut stride = 1;
                for i in (0..dimensions.len()).rev() {
                    strides[i] = stride;
                    stride *= dimensions[i];
                }
            }
        }

        strides
    }

    /// Infer cuDNN data type from Rust type
    fn infer_data_type<T>() -> Result<CudnnDataType>
    where
        T: 'static,
    {
        let type_id = std::any::TypeId::of::<T>();

        if type_id == std::any::TypeId::of::<f32>() {
            Ok(CudnnDataType::Float)
        } else if type_id == std::any::TypeId::of::<f64>() {
            Ok(CudnnDataType::Double)
        } else if type_id == std::any::TypeId::of::<i32>() {
            Ok(CudnnDataType::Int32)
        } else if type_id == std::any::TypeId::of::<i8>() {
            Ok(CudnnDataType::Int8)
        } else if type_id == std::any::TypeId::of::<u8>() {
            Ok(CudnnDataType::Uint8)
        } else {
            Err(TensorError::unsupported_operation_simple(format!(
                "Unsupported data type for cuDNN: {:?}",
                std::any::type_name::<T>()
            )))
        }
    }

    /// Get tensor element count
    pub fn element_count(&self) -> usize {
        self.dimensions.iter().map(|&x| x as usize).product()
    }

    /// Get tensor size in bytes
    pub fn size_in_bytes(&self) -> usize {
        let element_size = match self.data_type {
            CudnnDataType::Float => 4,
            CudnnDataType::Double => 8,
            CudnnDataType::Half => 2,
            CudnnDataType::Int8 | CudnnDataType::Uint8 => 1,
            CudnnDataType::Int32 => 4,
            _ => 4, // Default to 4 bytes
        };

        self.element_count() * element_size
    }
}

#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
impl CudnnConvolutionDescriptor {
    /// Create a new convolution descriptor
    pub fn new(
        padding: Vec<i32>,
        stride: Vec<i32>,
        dilation: Vec<i32>,
        mode: CudnnConvolutionMode,
        compute_type: CudnnDataType,
    ) -> Result<Self> {
        Ok(CudnnConvolutionDescriptor {
            padding,
            stride,
            dilation,
            mode,
            compute_type,
        })
    }

    /// Create convolution descriptor for 2D convolution
    pub fn conv2d(
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
        dilation_h: i32,
        dilation_w: i32,
        compute_type: CudnnDataType,
    ) -> Result<Self> {
        Self::new(
            vec![pad_h, pad_w],
            vec![stride_h, stride_w],
            vec![dilation_h, dilation_w],
            CudnnConvolutionMode::CrossCorrelation,
            compute_type,
        )
    }
}

#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
impl CudnnPoolingDescriptor {
    /// Create a new pooling descriptor
    pub fn new(
        mode: CudnnPoolingMode,
        nan_opt: CudnnNanPropagation,
        window_size: Vec<i32>,
        padding: Vec<i32>,
        stride: Vec<i32>,
    ) -> Result<Self> {
        Ok(CudnnPoolingDescriptor {
            mode,
            nan_opt,
            window_size,
            padding,
            stride,
        })
    }

    /// Create pooling descriptor for 2D max pooling
    pub fn max_pool2d(
        window_h: i32,
        window_w: i32,
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
    ) -> Result<Self> {
        Self::new(
            CudnnPoolingMode::Max,
            CudnnNanPropagation::NotPropagateNan,
            vec![window_h, window_w],
            vec![pad_h, pad_w],
            vec![stride_h, stride_w],
        )
    }

    /// Create pooling descriptor for 2D average pooling
    pub fn avg_pool2d(
        window_h: i32,
        window_w: i32,
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
        count_include_pad: bool,
    ) -> Result<Self> {
        let mode = if count_include_pad {
            CudnnPoolingMode::AverageCountIncludePadding
        } else {
            CudnnPoolingMode::AverageCountExcludePadding
        };

        Self::new(
            mode,
            CudnnNanPropagation::NotPropagateNan,
            vec![window_h, window_w],
            vec![pad_h, pad_w],
            vec![stride_h, stride_w],
        )
    }
}

#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
impl CudnnContext {
    /// Create a new cuDNN context
    pub fn new() -> Self {
        CudnnContext {
            handles: HashMap::new(),
            tensor_descriptors: HashMap::new(),
            convolution_descriptors: HashMap::new(),
            pooling_descriptors: HashMap::new(),
        }
    }

    /// Get or create cuDNN handle for device
    pub fn get_handle(&mut self, device_id: usize) -> Result<Arc<CudnnHandle>> {
        if let Some(handle) = self.handles.get(&device_id) {
            Ok(Arc::clone(handle))
        } else {
            let handle = Arc::new(CudnnHandle::new(device_id)?);
            self.handles.insert(device_id, Arc::clone(&handle));
            Ok(handle)
        }
    }

    /// Check if cuDNN is available.
    ///
    /// Always returns `false` because no cuDNN SDK is linked in this build.
    pub fn is_available() -> bool {
        false
    }

    /// Get cuDNN version information.
    ///
    /// # Errors
    ///
    /// Always returns `Err` because no cuDNN SDK is linked.
    pub fn version_info() -> Result<String> {
        Err(TensorError::not_implemented_simple(
            "cuDNN version query requires libcudnn.so; \
             build with the CUDA/cuDNN toolkit installed"
                .to_string(),
        ))
    }

    /// Perform cuDNN convolution forward pass
    pub fn convolution_forward<T>(
        &mut self,
        input: &Tensor<T>,
        _weights: &Tensor<T>,
        _bias: Option<&Tensor<T>>,
        _conv_desc: &CudnnConvolutionDescriptor,
        _output_desc: &CudnnTensorDescriptor,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        let device_id = match input.device() {
            Device::Gpu(id) => *id,
            Device::Cpu => {
                return Err(TensorError::device_error_simple(
                    "cuDNN requires GPU device".to_string(),
                ))
            }
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => {
                return Err(TensorError::device_error_simple(
                    "cuDNN not supported on ROCm devices".to_string(),
                ))
            }
        };

        // `get_handle` always fails (CudnnHandle::new returns Err) — propagate.
        let _handle = self.get_handle(device_id)?;

        // Unreachable: get_handle always errors. Kept for type coherence.
        Err(TensorError::not_implemented_simple(
            "cuDNN convolution_forward requires libcudnn.so; \
             use the WGPU backend for GPU convolution"
                .to_string(),
        ))
    }

    /// Perform cuDNN pooling forward pass
    pub fn pooling_forward<T>(
        &mut self,
        input: &Tensor<T>,
        _pooling_desc: &CudnnPoolingDescriptor,
        _output_desc: &CudnnTensorDescriptor,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        let device_id = match input.device() {
            Device::Gpu(id) => *id,
            Device::Cpu => {
                return Err(TensorError::device_error_simple(
                    "cuDNN requires GPU device".to_string(),
                ))
            }
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => {
                return Err(TensorError::device_error_simple(
                    "cuDNN not supported on ROCm devices".to_string(),
                ))
            }
        };

        // `get_handle` always fails (CudnnHandle::new returns Err) — propagate.
        let _handle = self.get_handle(device_id)?;

        // Unreachable: get_handle always errors. Kept for type coherence.
        Err(TensorError::not_implemented_simple(
            "cuDNN pooling_forward requires libcudnn.so; \
             use the WGPU backend for GPU pooling"
                .to_string(),
        ))
    }

    /// Perform cuDNN activation forward pass
    pub fn activation_forward<T>(
        &mut self,
        input: &Tensor<T>,
        _activation_mode: CudnnActivationMode,
        _alpha: f64,
        _beta: f64,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        let device_id = match input.device() {
            Device::Gpu(id) => *id,
            Device::Cpu => {
                return Err(TensorError::device_error_simple(
                    "cuDNN requires GPU device".to_string(),
                ))
            }
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => {
                return Err(TensorError::device_error_simple(
                    "cuDNN not supported on ROCm devices".to_string(),
                ))
            }
        };

        // `get_handle` always fails (CudnnHandle::new returns Err) — propagate.
        let _handle = self.get_handle(device_id)?;

        // Unreachable: get_handle always errors. Kept for type coherence.
        Err(TensorError::not_implemented_simple(
            "cuDNN activation_forward requires libcudnn.so; \
             use the WGPU backend for GPU activations"
                .to_string(),
        ))
    }

    /// Clear cached descriptors
    pub fn clear_cache(&mut self) {
        self.tensor_descriptors.clear();
        self.convolution_descriptors.clear();
        self.pooling_descriptors.clear();
    }
}

#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
impl Default for CudnnContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Global cuDNN context instance
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
static GLOBAL_CUDNN_CONTEXT: std::sync::OnceLock<std::sync::Mutex<CudnnContext>> =
    std::sync::OnceLock::new();

/// Get global cuDNN context.
///
/// # Panics
///
/// Panics if the internal mutex has been poisoned by a panicking thread.
/// This is a hard error (the global state is corrupted) with no recovery path.
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
pub fn global_cudnn_context() -> std::sync::MutexGuard<'static, CudnnContext> {
    GLOBAL_CUDNN_CONTEXT
        .get_or_init(|| std::sync::Mutex::new(CudnnContext::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Utility functions for cuDNN integration
#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
pub mod utils {
    use super::*;

    /// Check if a tensor is compatible with cuDNN
    pub fn is_tensor_cudnn_compatible<T>(tensor: &Tensor<T>) -> bool
    where
        T: 'static,
    {
        // Check if tensor is on GPU
        #[cfg(feature = "gpu")]
        {
            if !tensor.device().is_gpu() {
                return false;
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            return false;
        }

        // Check if data type is supported
        CudnnTensorDescriptor::infer_data_type::<T>().is_ok()
    }

    /// Convert tensor format between NCHW and NHWC.
    ///
    /// # Errors
    ///
    /// Always returns `Err` because no cuDNN SDK is linked. Without cuDNN,
    /// there is no safe GPU-side transpose and silently cloning the tensor
    /// would return wrong data.
    pub fn convert_tensor_format<T>(
        _tensor: &Tensor<T>,
        _target_format: CudnnTensorFormat,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        Err(TensorError::not_implemented_simple(
            "cuDNN tensor format conversion requires libcudnn.so; \
             build with the CUDA/cuDNN toolkit installed"
                .to_string(),
        ))
    }

    /// Get optimal cuDNN algorithm for convolution.
    ///
    /// # Errors
    ///
    /// Always returns `Err` because no cuDNN SDK is linked.
    pub fn find_best_convolution_algorithm(
        _input_desc: &CudnnTensorDescriptor,
        _filter_desc: &CudnnTensorDescriptor,
        _conv_desc: &CudnnConvolutionDescriptor,
        _output_desc: &CudnnTensorDescriptor,
    ) -> Result<i32> {
        Err(TensorError::not_implemented_simple(
            "cuDNN algorithm selection (cudnnFindConvolutionForwardAlgorithm) \
             requires libcudnn.so; build with the CUDA/cuDNN toolkit installed"
                .to_string(),
        ))
    }

    /// Calculate workspace size needed for operation.
    ///
    /// # Errors
    ///
    /// Always returns `Err` because no cuDNN SDK is linked.
    pub fn get_convolution_workspace_size(
        _input_desc: &CudnnTensorDescriptor,
        _filter_desc: &CudnnTensorDescriptor,
        _conv_desc: &CudnnConvolutionDescriptor,
        _output_desc: &CudnnTensorDescriptor,
        _algorithm: i32,
    ) -> Result<usize> {
        Err(TensorError::not_implemented_simple(
            "cuDNN workspace size query (cudnnGetConvolutionForwardWorkspaceSize) \
             requires libcudnn.so; build with the CUDA/cuDNN toolkit installed"
                .to_string(),
        ))
    }
}

#[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cudnn_context_creation() {
        let context = CudnnContext::new();
        assert!(context.handles.is_empty());
    }

    #[test]
    fn test_tensor_descriptor_creation() {
        let desc = CudnnTensorDescriptor::new(
            CudnnDataType::Float,
            CudnnTensorFormat::NCHW,
            vec![1, 3, 224, 224],
        )
        .expect("test: operation should succeed");

        assert_eq!(desc.data_type, CudnnDataType::Float);
        assert_eq!(desc.format, CudnnTensorFormat::NCHW);
        assert_eq!(desc.dimensions, vec![1, 3, 224, 224]);
        assert_eq!(desc.element_count(), 150528);
    }

    #[test]
    fn test_convolution_descriptor_creation() {
        let desc = CudnnConvolutionDescriptor::conv2d(
            1,
            1, // padding
            1,
            1, // stride
            1,
            1, // dilation
            CudnnDataType::Float,
        )
        .expect("test: operation should succeed");

        assert_eq!(desc.padding, vec![1, 1]);
        assert_eq!(desc.stride, vec![1, 1]);
        assert_eq!(desc.dilation, vec![1, 1]);
    }

    #[test]
    fn test_pooling_descriptor_creation() {
        let desc = CudnnPoolingDescriptor::max_pool2d(
            2, 2, // window
            0, 0, // padding
            2, 2, // stride
        )
        .expect("test: operation should succeed");

        assert_eq!(desc.mode, CudnnPoolingMode::Max);
        assert_eq!(desc.window_size, vec![2, 2]);
        assert_eq!(desc.stride, vec![2, 2]);
    }

    #[test]
    fn test_cudnn_availability() {
        // No cuDNN SDK linked — must always return false
        assert!(!CudnnContext::is_available());
    }

    #[test]
    fn test_version_info() {
        // version_info always errors when no cuDNN SDK is linked
        let result = CudnnContext::version_info();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("cuDNN") || err.contains("libcudnn"));
    }

    #[test]
    fn test_stride_calculation() {
        let dims = vec![1, 3, 224, 224];
        let strides = CudnnTensorDescriptor::calculate_strides(&dims, CudnnTensorFormat::NCHW);

        // For NCHW: strides should be [150528, 50176, 224, 1]
        assert_eq!(strides, vec![150528, 50176, 224, 1]);
    }

    #[test]
    fn test_data_type_inference() {
        assert_eq!(
            CudnnTensorDescriptor::infer_data_type::<f32>()
                .expect("test: operation should succeed"),
            CudnnDataType::Float
        );
        assert_eq!(
            CudnnTensorDescriptor::infer_data_type::<f64>()
                .expect("test: operation should succeed"),
            CudnnDataType::Double
        );
        assert_eq!(
            CudnnTensorDescriptor::infer_data_type::<i32>()
                .expect("test: operation should succeed"),
            CudnnDataType::Int32
        );
    }
}
