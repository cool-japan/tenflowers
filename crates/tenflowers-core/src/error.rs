use crate::{DType, Device};
use thiserror::Error;

/// Enhanced error handling with contextual information and recovery strategies
#[derive(Error, Debug, Clone)]
pub enum TensorError {
    #[error("Shape mismatch in operation '{operation}': expected {expected}, got {got}")]
    ShapeMismatch {
        operation: String,
        expected: String,
        got: String,
        context: Option<ErrorContext>,
    },

    #[error("Incompatible devices in operation '{operation}': {device1} and {device2}")]
    DeviceMismatch {
        operation: String,
        device1: String,
        device2: String,
        context: Option<ErrorContext>,
    },

    #[error("Operation '{operation}' not supported on device: {device}")]
    UnsupportedDevice {
        operation: String,
        device: String,
        fallback_available: bool,
        context: Option<ErrorContext>,
    },

    #[error("Invalid shape in operation '{operation}': {reason}")]
    InvalidShape {
        operation: String,
        reason: String,
        shape: Option<Vec<usize>>,
        context: Option<ErrorContext>,
    },

    #[error("Invalid axis {axis} in operation '{operation}' for tensor with {ndim} dimensions")]
    InvalidAxis {
        operation: String,
        axis: i32,
        ndim: usize,
        context: Option<ErrorContext>,
    },

    #[error("Gradient computation not enabled for tensor in operation '{operation}'")]
    GradientNotEnabled {
        operation: String,
        suggestion: String,
        context: Option<ErrorContext>,
    },

    #[error("Invalid argument in operation '{operation}': {reason}")]
    InvalidArgument {
        operation: String,
        reason: String,
        context: Option<ErrorContext>,
    },

    #[error("Memory allocation failed in operation '{operation}': {details}")]
    AllocationError {
        operation: String,
        details: String,
        requested_bytes: Option<usize>,
        available_bytes: Option<usize>,
        context: Option<ErrorContext>,
    },

    #[error("Operation '{operation}' not supported: {reason}")]
    UnsupportedOperation {
        operation: String,
        reason: String,
        alternatives: Vec<String>,
        context: Option<ErrorContext>,
    },

    #[error("GPU error in operation '{operation}': {details}")]
    #[cfg(feature = "gpu")]
    GpuError {
        operation: String,
        details: String,
        gpu_id: Option<usize>,
        fallback_attempted: bool,
        context: Option<ErrorContext>,
    },

    #[error("Device error in operation '{operation}': {details}")]
    DeviceError {
        operation: String,
        details: String,
        device: String,
        context: Option<ErrorContext>,
    },

    #[error("Compute error in operation '{operation}': {details}")]
    ComputeError {
        operation: String,
        details: String,
        retry_possible: bool,
        context: Option<ErrorContext>,
    },

    #[error("BLAS error in operation '{operation}': {details}")]
    #[cfg(feature = "blas")]
    BlasError {
        operation: String,
        details: String,
        context: Option<ErrorContext>,
    },

    #[error("Serialization error in operation '{operation}': {details}")]
    SerializationError {
        operation: String,
        details: String,
        context: Option<ErrorContext>,
    },

    #[error("Operation '{operation}' not implemented: {details}")]
    NotImplemented {
        operation: String,
        details: String,
        planned_version: Option<String>,
        context: Option<ErrorContext>,
    },

    #[error("Invalid operation '{operation}': {reason}")]
    InvalidOperation {
        operation: String,
        reason: String,
        context: Option<ErrorContext>,
    },

    #[error("Benchmark error in '{operation}': {details}")]
    BenchmarkError {
        operation: String,
        details: String,
        context: Option<ErrorContext>,
    },

    #[error("IO error in operation '{operation}': {details}")]
    IoError {
        operation: String,
        details: String,
        path: Option<String>,
        context: Option<ErrorContext>,
    },

    #[error("Numerical error in operation '{operation}': {details}")]
    NumericalError {
        operation: String,
        details: String,
        suggestions: Vec<String>,
        context: Option<ErrorContext>,
    },

    #[error("Resource exhaustion in operation '{operation}': {resource}")]
    ResourceExhausted {
        operation: String,
        resource: String,
        current_usage: Option<usize>,
        limit: Option<usize>,
        context: Option<ErrorContext>,
    },

    #[error("Timeout in operation '{operation}' after {duration_ms}ms")]
    Timeout {
        operation: String,
        duration_ms: u64,
        context: Option<ErrorContext>,
    },

    #[error("Cache operation failed in '{operation}': {details}")]
    CacheError {
        operation: String,
        details: String,
        recoverable: bool,
        context: Option<ErrorContext>,
    },

    #[error("Other error in operation '{operation}': {details}")]
    Other {
        operation: String,
        details: String,
        context: Option<ErrorContext>,
    },
}

/// Additional context information for errors
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Input tensor shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Input tensor devices
    pub input_devices: Vec<Device>,
    /// Input tensor data types
    pub input_dtypes: Vec<DType>,
    /// Output shape (if applicable)
    pub output_shape: Option<Vec<usize>>,
    /// Thread ID where error occurred
    pub thread_id: String,
    /// Stack trace (if available)
    pub stack_trace: Option<String>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// Recovery strategy for handling errors
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// No recovery possible
    None,
    /// Fallback to CPU execution
    FallbackToCpu,
    /// Retry with different parameters
    RetryWithParams(std::collections::HashMap<String, String>),
    /// Use alternative algorithm
    UseAlternative(String),
    /// Reduce precision
    ReducePrecision,
    /// Free memory and retry
    FreeMemoryAndRetry,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new() -> Self {
        Self {
            input_shapes: Vec::new(),
            input_devices: Vec::new(),
            input_dtypes: Vec::new(),
            output_shape: None,
            thread_id: format!("{:?}", std::thread::current().id()),
            stack_trace: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add input tensor information
    pub fn with_input_tensor(mut self, shape: &[usize], device: Device, dtype: DType) -> Self {
        self.input_shapes.push(shape.to_vec());
        self.input_devices.push(device);
        self.input_dtypes.push(dtype);
        self
    }

    /// Add output shape information
    pub fn with_output_shape(mut self, shape: &[usize]) -> Self {
        self.output_shape = Some(shape.to_vec());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorError {
    /// Create a shape mismatch error with context
    pub fn shape_mismatch(operation: &str, expected: &str, got: &str) -> Self {
        Self::ShapeMismatch {
            operation: operation.to_string(),
            expected: expected.to_string(),
            got: got.to_string(),
            context: None,
        }
    }

    /// Create a device mismatch error with context
    pub fn device_mismatch(operation: &str, device1: &str, device2: &str) -> Self {
        Self::DeviceMismatch {
            operation: operation.to_string(),
            device1: device1.to_string(),
            device2: device2.to_string(),
            context: None,
        }
    }

    /// Create an unsupported device error with fallback information
    pub fn unsupported_device(operation: &str, device: &str, fallback_available: bool) -> Self {
        Self::UnsupportedDevice {
            operation: operation.to_string(),
            device: device.to_string(),
            fallback_available,
            context: None,
        }
    }

    /// Create a GPU error with fallback information
    #[cfg(feature = "gpu")]
    pub fn gpu_error(
        operation: &str,
        details: &str,
        gpu_id: Option<usize>,
        fallback_attempted: bool,
    ) -> Self {
        Self::GpuError {
            operation: operation.to_string(),
            details: details.to_string(),
            gpu_id,
            fallback_attempted,
            context: None,
        }
    }

    /// Create an allocation error with memory information
    pub fn allocation_error(
        operation: &str,
        details: &str,
        requested: Option<usize>,
        available: Option<usize>,
    ) -> Self {
        Self::AllocationError {
            operation: operation.to_string(),
            details: details.to_string(),
            requested_bytes: requested,
            available_bytes: available,
            context: None,
        }
    }

    /// Create a numerical error with suggestions
    pub fn numerical_error(operation: &str, details: &str, suggestions: Vec<String>) -> Self {
        Self::NumericalError {
            operation: operation.to_string(),
            details: details.to_string(),
            suggestions,
            context: None,
        }
    }

    /// Create an invalid argument error (for backward compatibility)
    pub fn invalid_argument(reason: String) -> Self {
        Self::InvalidArgument {
            operation: "unknown".to_string(),
            reason,
            context: None,
        }
    }

    /// Create an invalid argument error with operation context
    pub fn invalid_argument_op(operation: &str, reason: &str) -> Self {
        Self::InvalidArgument {
            operation: operation.to_string(),
            reason: reason.to_string(),
            context: None,
        }
    }

    /// Create a generic "other" error (for backward compatibility)
    pub fn other(details: String) -> Self {
        Self::Other {
            operation: "unknown".to_string(),
            details,
            context: None,
        }
    }

    /// Create a generic "other" error with operation context
    pub fn other_op(operation: &str, details: &str) -> Self {
        Self::Other {
            operation: operation.to_string(),
            details: details.to_string(),
            context: None,
        }
    }

    /// Create an allocation error (for backward compatibility)
    pub fn allocation_error_simple(details: String) -> Self {
        Self::AllocationError {
            operation: "unknown".to_string(),
            details,
            requested_bytes: None,
            available_bytes: None,
            context: None,
        }
    }

    /// Create an unsupported operation error (for backward compatibility)
    pub fn unsupported_operation_simple(reason: String) -> Self {
        Self::UnsupportedOperation {
            operation: "unknown".to_string(),
            reason,
            alternatives: Vec::new(),
            context: None,
        }
    }

    /// Create an invalid shape error with operation context
    pub fn invalid_shape(operation: &str, expected: &str, got: &str) -> Self {
        Self::InvalidShape {
            operation: operation.to_string(),
            reason: format!("Expected {}, got {}", expected, got),
            shape: None,
            context: None,
        }
    }

    /// Create an invalid shape error (for backward compatibility)
    pub fn invalid_shape_simple(reason: String) -> Self {
        Self::InvalidShape {
            operation: "unknown".to_string(),
            reason,
            shape: None,
            context: None,
        }
    }

    /// Create a device error (for backward compatibility)
    pub fn device_error_simple(details: String) -> Self {
        Self::DeviceError {
            operation: "unknown".to_string(),
            details,
            device: "unknown".to_string(),
            context: None,
        }
    }

    /// Create a compute error (for backward compatibility)
    pub fn compute_error_simple(details: String) -> Self {
        Self::ComputeError {
            operation: "unknown".to_string(),
            details,
            retry_possible: false,
            context: None,
        }
    }

    /// Create a serialization error (for backward compatibility)
    pub fn serialization_error_simple(details: String) -> Self {
        Self::SerializationError {
            operation: "unknown".to_string(),
            details,
            context: None,
        }
    }

    /// Create a not implemented error (for backward compatibility)
    pub fn not_implemented_simple(details: String) -> Self {
        Self::NotImplemented {
            operation: "unknown".to_string(),
            details,
            planned_version: None,
            context: None,
        }
    }

    /// Create an invalid operation error (for backward compatibility)
    pub fn invalid_operation_simple(reason: String) -> Self {
        Self::InvalidOperation {
            operation: "unknown".to_string(),
            reason,
            context: None,
        }
    }

    /// Create a benchmark error (for backward compatibility)
    pub fn benchmark_error_simple(details: String) -> Self {
        Self::BenchmarkError {
            operation: "unknown".to_string(),
            details,
            context: None,
        }
    }

    /// Create an IO error (for backward compatibility)
    pub fn io_error_simple(details: String) -> Self {
        Self::IoError {
            operation: "unknown".to_string(),
            details,
            path: None,
            context: None,
        }
    }

    /// Create a resource exhausted error (for backward compatibility)
    pub fn resource_exhausted_simple(resource: String) -> Self {
        Self::ResourceExhausted {
            operation: "unknown".to_string(),
            resource,
            current_usage: None,
            limit: None,
            context: None,
        }
    }

    /// Create a timeout error (for backward compatibility)
    pub fn timeout_simple(duration_ms: u64) -> Self {
        Self::Timeout {
            operation: "unknown".to_string(),
            duration_ms,
            context: None,
        }
    }

    /// Add context to an existing error
    pub fn with_context(mut self, context: ErrorContext) -> Self {
        match &mut self {
            Self::ShapeMismatch { context: ctx, .. } => *ctx = Some(context),
            Self::DeviceMismatch { context: ctx, .. } => *ctx = Some(context),
            Self::UnsupportedDevice { context: ctx, .. } => *ctx = Some(context),
            Self::InvalidShape { context: ctx, .. } => *ctx = Some(context),
            Self::InvalidAxis { context: ctx, .. } => *ctx = Some(context),
            Self::GradientNotEnabled { context: ctx, .. } => *ctx = Some(context),
            Self::InvalidArgument { context: ctx, .. } => *ctx = Some(context),
            Self::AllocationError { context: ctx, .. } => *ctx = Some(context),
            Self::UnsupportedOperation { context: ctx, .. } => *ctx = Some(context),
            #[cfg(feature = "gpu")]
            Self::GpuError { context: ctx, .. } => *ctx = Some(context),
            Self::DeviceError { context: ctx, .. } => *ctx = Some(context),
            Self::ComputeError { context: ctx, .. } => *ctx = Some(context),
            #[cfg(feature = "blas")]
            Self::BlasError { context: ctx, .. } => *ctx = Some(context),
            Self::SerializationError { context: ctx, .. } => *ctx = Some(context),
            Self::NotImplemented { context: ctx, .. } => *ctx = Some(context),
            Self::InvalidOperation { context: ctx, .. } => *ctx = Some(context),
            Self::BenchmarkError { context: ctx, .. } => *ctx = Some(context),
            Self::IoError { context: ctx, .. } => *ctx = Some(context),
            Self::NumericalError { context: ctx, .. } => *ctx = Some(context),
            Self::ResourceExhausted { context: ctx, .. } => *ctx = Some(context),
            Self::Timeout { context: ctx, .. } => *ctx = Some(context),
            Self::CacheError { context: ctx, .. } => *ctx = Some(context),
            Self::Other { context: ctx, .. } => *ctx = Some(context),
        }
        self
    }

    /// Get the operation name for this error
    pub fn operation(&self) -> &str {
        match self {
            Self::ShapeMismatch { operation, .. } => operation,
            Self::DeviceMismatch { operation, .. } => operation,
            Self::UnsupportedDevice { operation, .. } => operation,
            Self::InvalidShape { operation, .. } => operation,
            Self::InvalidAxis { operation, .. } => operation,
            Self::GradientNotEnabled { operation, .. } => operation,
            Self::InvalidArgument { operation, .. } => operation,
            Self::AllocationError { operation, .. } => operation,
            Self::UnsupportedOperation { operation, .. } => operation,
            #[cfg(feature = "gpu")]
            Self::GpuError { operation, .. } => operation,
            Self::DeviceError { operation, .. } => operation,
            Self::ComputeError { operation, .. } => operation,
            #[cfg(feature = "blas")]
            Self::BlasError { operation, .. } => operation,
            Self::SerializationError { operation, .. } => operation,
            Self::NotImplemented { operation, .. } => operation,
            Self::InvalidOperation { operation, .. } => operation,
            Self::BenchmarkError { operation, .. } => operation,
            Self::IoError { operation, .. } => operation,
            Self::NumericalError { operation, .. } => operation,
            Self::ResourceExhausted { operation, .. } => operation,
            Self::Timeout { operation, .. } => operation,
            Self::CacheError { operation, .. } => operation,
            Self::Other { operation, .. } => operation,
        }
    }

    /// Check if this error supports fallback recovery
    pub fn supports_fallback(&self) -> bool {
        match self {
            Self::UnsupportedDevice {
                fallback_available, ..
            } => *fallback_available,
            #[cfg(feature = "gpu")]
            Self::GpuError { .. } => true,
            Self::AllocationError { .. } => true,
            Self::ComputeError { retry_possible, .. } => *retry_possible,
            _ => false,
        }
    }

    /// Get suggested recovery strategy
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            Self::UnsupportedDevice {
                fallback_available: true,
                ..
            } => RecoveryStrategy::FallbackToCpu,
            #[cfg(feature = "gpu")]
            Self::GpuError {
                fallback_attempted: false,
                ..
            } => RecoveryStrategy::FallbackToCpu,
            Self::AllocationError { .. } => RecoveryStrategy::FreeMemoryAndRetry,
            Self::ComputeError {
                retry_possible: true,
                ..
            } => {
                let mut params = std::collections::HashMap::new();
                params.insert("reduce_precision".to_string(), "true".to_string());
                RecoveryStrategy::RetryWithParams(params)
            }
            Self::NumericalError { .. } => RecoveryStrategy::ReducePrecision,
            _ => RecoveryStrategy::None,
        }
    }
}

/// Trait for automatic error recovery
pub trait ErrorRecovery<T> {
    /// Attempt to recover from error using suggested strategy
    fn recover_with_strategy(self, strategy: RecoveryStrategy) -> Result<T>;

    /// Attempt automatic recovery if possible
    fn auto_recover(self) -> Result<T>;
}

impl<T> ErrorRecovery<T> for Result<T> {
    fn recover_with_strategy(self, _strategy: RecoveryStrategy) -> Result<T> {
        // For now, just return the original result
        // In a full implementation, this would attempt recovery based on the strategy
        self
    }

    fn auto_recover(self) -> Result<T> {
        match &self {
            Err(error) if error.supports_fallback() => {
                let strategy = error.recovery_strategy();
                self.recover_with_strategy(strategy)
            }
            _ => self,
        }
    }
}

pub type Result<T> = std::result::Result<T, TensorError>;

/// Convert from scirs2_core::ndarray::ShapeError to TensorError
impl From<scirs2_core::ndarray::ShapeError> for TensorError {
    fn from(err: scirs2_core::ndarray::ShapeError) -> Self {
        Self::InvalidShape {
            operation: "tensor_creation".to_string(),
            reason: format!("Shape error: {err}"),
            shape: None,
            context: None,
        }
    }
}

/// Convert from std::fmt::Error to TensorError
impl From<std::fmt::Error> for TensorError {
    fn from(err: std::fmt::Error) -> Self {
        Self::Other {
            operation: "formatting".to_string(),
            details: format!("Formatting error: {err}"),
            context: None,
        }
    }
}
