/// Unified Operation Dispatch Registry for TenfloweRS
///
/// This module provides a centralized system for registering and dispatching
/// tensor operations across different backends (CPU, GPU, BLAS, etc.) with
/// feature gating and capability detection.
use crate::{Device, Result, Shape, Tensor, TensorError};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Backend type for kernel implementations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    /// CPU implementation (always available)
    Cpu,
    /// SIMD-optimized CPU implementation
    #[cfg(feature = "simd")]
    SimdCpu,
    /// BLAS-accelerated implementation
    #[cfg(feature = "blas")]
    Blas,
    /// GPU implementation via WGPU
    #[cfg(feature = "gpu")]
    Gpu,
    /// CUDA implementation
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    Cuda,
    /// ROCm implementation
    #[cfg(feature = "rocm")]
    Rocm,
    /// Metal Performance Shaders
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal,
}

impl BackendType {
    /// Check if this backend is available at runtime
    pub fn is_available(&self) -> bool {
        match self {
            BackendType::Cpu => true,
            #[cfg(feature = "simd")]
            BackendType::SimdCpu => true,
            #[cfg(feature = "blas")]
            BackendType::Blas => crate::ops::lapack::is_lapack_available(),
            #[cfg(feature = "gpu")]
            BackendType::Gpu => true, // WGPU availability checked at context creation
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            BackendType::Cuda => crate::gpu::cuda_kernels::is_cuda_available(),
            #[cfg(feature = "rocm")]
            BackendType::Rocm => false, // TODO: Implement ROCm availability check
            #[cfg(all(feature = "metal", target_os = "macos"))]
            BackendType::Metal => true,
        }
    }

    /// Get priority for backend selection (higher = preferred)
    pub fn priority(&self) -> u8 {
        match self {
            BackendType::Cpu => 0,
            #[cfg(feature = "simd")]
            BackendType::SimdCpu => 10,
            #[cfg(feature = "blas")]
            BackendType::Blas => 20,
            #[cfg(feature = "gpu")]
            BackendType::Gpu => 30,
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            BackendType::Cuda => 40,
            #[cfg(feature = "rocm")]
            BackendType::Rocm => 40,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            BackendType::Metal => 50,
        }
    }

    /// Get backend from device
    pub fn from_device(device: &Device) -> Self {
        match device {
            Device::Cpu => BackendType::Cpu,
            #[cfg(feature = "gpu")]
            Device::Gpu(_) => BackendType::Gpu,
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => BackendType::Rocm,
        }
    }
}

/// Operation metadata and description
#[derive(Debug, Clone)]
pub struct OperationDescriptor {
    /// Unique operation name
    pub name: String,
    /// Operation category (e.g., "binary", "reduction", "matmul")
    pub category: String,
    /// Semantic version
    pub version: String,
    /// Supported data types
    pub supported_dtypes: Vec<crate::DType>,
    /// Minimum supported rank
    pub min_rank: Option<usize>,
    /// Maximum supported rank
    pub max_rank: Option<usize>,
    /// Whether operation supports broadcasting
    pub supports_broadcast: bool,
    /// Whether operation is in-place capable
    pub supports_inplace: bool,
}

impl OperationDescriptor {
    /// Create a new operation descriptor
    pub fn new(name: impl Into<String>, category: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            category: category.into(),
            version: "1.0.0".to_string(),
            supported_dtypes: vec![crate::DType::Float32, crate::DType::Float64],
            min_rank: None,
            max_rank: None,
            supports_broadcast: false,
            supports_inplace: false,
        }
    }

    /// Set supported data types
    pub fn with_dtypes(mut self, dtypes: Vec<crate::DType>) -> Self {
        self.supported_dtypes = dtypes;
        self
    }

    /// Set rank constraints
    pub fn with_rank_range(mut self, min: Option<usize>, max: Option<usize>) -> Self {
        self.min_rank = min;
        self.max_rank = max;
        self
    }

    /// Enable broadcasting support
    pub fn with_broadcast(mut self) -> Self {
        self.supports_broadcast = true;
        self
    }

    /// Enable in-place support
    pub fn with_inplace(mut self) -> Self {
        self.supports_inplace = true;
        self
    }
}

/// Kernel implementation function signature for unary operations
pub type UnaryKernelFn<T> = fn(&Tensor<T>) -> Result<Tensor<T>>;

/// Kernel implementation function signature for binary operations
pub type BinaryKernelFn<T> = fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>;

/// Kernel implementation for a specific backend
#[derive(Clone)]
pub struct KernelImplementation<T> {
    pub backend: BackendType,
    pub unary_fn: Option<UnaryKernelFn<T>>,
    pub binary_fn: Option<BinaryKernelFn<T>>,
}

impl<T> KernelImplementation<T> {
    /// Create a new unary kernel implementation
    pub fn unary(backend: BackendType, func: UnaryKernelFn<T>) -> Self {
        Self {
            backend,
            unary_fn: Some(func),
            binary_fn: None,
        }
    }

    /// Create a new binary kernel implementation
    pub fn binary(backend: BackendType, func: BinaryKernelFn<T>) -> Self {
        Self {
            backend,
            unary_fn: None,
            binary_fn: Some(func),
        }
    }
}

/// Registered operation with all backend implementations
struct RegisteredOperation<T> {
    descriptor: OperationDescriptor,
    kernels: Vec<KernelImplementation<T>>,
}

impl<T> RegisteredOperation<T> {
    fn new(descriptor: OperationDescriptor) -> Self {
        Self {
            descriptor,
            kernels: Vec::new(),
        }
    }

    fn add_kernel(&mut self, kernel: KernelImplementation<T>) {
        self.kernels.push(kernel);
    }

    /// Select the best available kernel for the given device
    fn select_kernel(&self, device: &Device) -> Option<&KernelImplementation<T>> {
        let preferred_backend = BackendType::from_device(device);

        // First, try to find the preferred backend
        if let Some(kernel) = self
            .kernels
            .iter()
            .find(|k| k.backend == preferred_backend && k.backend.is_available())
        {
            return Some(kernel);
        }

        // Fall back to the highest priority available backend
        self.kernels
            .iter()
            .filter(|k| k.backend.is_available())
            .max_by_key(|k| k.backend.priority())
    }
}

/// Global operation dispatch registry
pub struct DispatchRegistry<T> {
    operations: Arc<RwLock<HashMap<String, RegisteredOperation<T>>>>,
}

impl<T> Default for DispatchRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DispatchRegistry<T> {
    /// Create a new dispatch registry
    pub fn new() -> Self {
        Self {
            operations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new operation
    pub fn register_operation(&self, descriptor: OperationDescriptor) -> Result<()> {
        let mut ops = self.operations.write().unwrap();

        if ops.contains_key(&descriptor.name) {
            return Err(TensorError::invalid_argument(format!(
                "Operation '{}' is already registered",
                descriptor.name
            )));
        }

        ops.insert(
            descriptor.name.clone(),
            RegisteredOperation::new(descriptor),
        );
        Ok(())
    }

    /// Register a kernel implementation for an operation
    pub fn register_kernel(
        &self,
        operation_name: &str,
        kernel: KernelImplementation<T>,
    ) -> Result<()> {
        let mut ops = self.operations.write().unwrap();

        let op = ops.get_mut(operation_name).ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "Operation '{}' not found. Register the operation first.",
                operation_name
            ))
        })?;

        op.add_kernel(kernel);
        Ok(())
    }

    /// Dispatch a unary operation
    pub fn dispatch_unary(&self, operation_name: &str, input: &Tensor<T>) -> Result<Tensor<T>> {
        let ops = self.operations.read().unwrap();

        let op = ops.get(operation_name).ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "Operation '{}' not found in registry",
                operation_name
            ))
        })?;

        let kernel = op.select_kernel(input.device()).ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "No available kernel for operation '{}' on device {:?}",
                operation_name,
                input.device()
            ))
        })?;

        let kernel_fn = kernel.unary_fn.ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "Operation '{}' does not support unary execution",
                operation_name
            ))
        })?;

        kernel_fn(input)
    }

    /// Dispatch a binary operation
    pub fn dispatch_binary(
        &self,
        operation_name: &str,
        lhs: &Tensor<T>,
        rhs: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        // Check device compatibility
        if lhs.device() != rhs.device() {
            return Err(TensorError::device_mismatch(
                operation_name,
                &format!("{:?}", lhs.device()),
                &format!("{:?}", rhs.device()),
            ));
        }

        let ops = self.operations.read().unwrap();

        let op = ops.get(operation_name).ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "Operation '{}' not found in registry",
                operation_name
            ))
        })?;

        let kernel = op.select_kernel(lhs.device()).ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "No available kernel for operation '{}' on device {:?}",
                operation_name,
                lhs.device()
            ))
        })?;

        let kernel_fn = kernel.binary_fn.ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "Operation '{}' does not support binary execution",
                operation_name
            ))
        })?;

        kernel_fn(lhs, rhs)
    }

    /// Get operation descriptor
    pub fn get_operation(&self, name: &str) -> Option<OperationDescriptor> {
        let ops = self.operations.read().unwrap();
        ops.get(name).map(|op| op.descriptor.clone())
    }

    /// List all registered operations
    pub fn list_operations(&self) -> Vec<String> {
        let ops = self.operations.read().unwrap();
        ops.keys().cloned().collect()
    }

    /// Get available backends for an operation
    pub fn available_backends(&self, operation_name: &str) -> Vec<BackendType> {
        let ops = self.operations.read().unwrap();

        if let Some(op) = ops.get(operation_name) {
            op.kernels
                .iter()
                .filter(|k| k.backend.is_available())
                .map(|k| k.backend)
                .collect()
        } else {
            Vec::new()
        }
    }
}

/// Macro to simplify operation registration
#[macro_export]
macro_rules! register_operation {
    ($registry:expr, $name:expr, $category:expr) => {
        $registry.register_operation(
            $crate::OperationDescriptor::new($name, $category)
        ).unwrap();
    };
    ($registry:expr, $name:expr, $category:expr, dtypes: [$($dtype:expr),*]) => {
        $registry.register_operation(
            $crate::OperationDescriptor::new($name, $category)
                .with_dtypes(vec![$($dtype),*])
        ).unwrap();
    };
    ($registry:expr, $name:expr, $category:expr, rank: $min:expr, $max:expr) => {
        $registry.register_operation(
            $crate::OperationDescriptor::new($name, $category)
                .with_rank_range(Some($min), Some($max))
        ).unwrap();
    };
}

/// Macro to register a unary kernel
#[macro_export]
macro_rules! register_unary_kernel {
    ($registry:expr, $op_name:expr, $backend:expr, $func:expr) => {
        $registry
            .register_kernel(
                $op_name,
                $crate::KernelImplementation::unary($backend, $func),
            )
            .unwrap();
    };
}

/// Macro to register a binary kernel
#[macro_export]
macro_rules! register_binary_kernel {
    ($registry:expr, $op_name:expr, $backend:expr, $func:expr) => {
        $registry
            .register_kernel(
                $op_name,
                $crate::KernelImplementation::binary($backend, $func),
            )
            .unwrap();
    };
}

/// Global registry instance (lazily initialized)
use lazy_static::lazy_static;

lazy_static! {
    /// Global f32 dispatch registry
    pub static ref F32_REGISTRY: DispatchRegistry<f32> = DispatchRegistry::new();

    /// Global f64 dispatch registry
    pub static ref F64_REGISTRY: DispatchRegistry<f64> = DispatchRegistry::new();

    /// Global i32 dispatch registry
    pub static ref I32_REGISTRY: DispatchRegistry<i32> = DispatchRegistry::new();
}

/// Get the global registry for a specific type
pub fn get_registry<T: 'static>() -> Option<&'static DispatchRegistry<T>> {
    use std::any::TypeId;

    let type_id = TypeId::of::<T>();

    if type_id == TypeId::of::<f32>() {
        // SAFETY: We've checked that T is f32
        Some(unsafe {
            &*(&*F32_REGISTRY as *const DispatchRegistry<f32> as *const DispatchRegistry<T>)
        })
    } else if type_id == TypeId::of::<f64>() {
        // SAFETY: We've checked that T is f64
        Some(unsafe {
            &*(&*F64_REGISTRY as *const DispatchRegistry<f64> as *const DispatchRegistry<T>)
        })
    } else if type_id == TypeId::of::<i32>() {
        // SAFETY: We've checked that T is i32
        Some(unsafe {
            &*(&*I32_REGISTRY as *const DispatchRegistry<i32> as *const DispatchRegistry<T>)
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_backend_type_priority() {
        assert!(BackendType::Cpu.priority() < BackendType::Cpu.priority() + 1);

        #[cfg(feature = "simd")]
        assert!(BackendType::SimdCpu.priority() > BackendType::Cpu.priority());
    }

    #[test]
    fn test_operation_descriptor() {
        let desc = OperationDescriptor::new("test_op", "binary")
            .with_dtypes(vec![crate::DType::Float32])
            .with_broadcast()
            .with_rank_range(Some(1), Some(4));

        assert_eq!(desc.name, "test_op");
        assert_eq!(desc.category, "binary");
        assert!(desc.supports_broadcast);
        assert_eq!(desc.min_rank, Some(1));
        assert_eq!(desc.max_rank, Some(4));
    }

    #[test]
    fn test_registry_creation() {
        let registry: DispatchRegistry<f32> = DispatchRegistry::new();
        assert_eq!(registry.list_operations().len(), 0);
    }

    #[test]
    fn test_operation_registration() {
        let registry: DispatchRegistry<f32> = DispatchRegistry::new();

        let desc = OperationDescriptor::new("add", "binary");
        registry.register_operation(desc).unwrap();

        assert_eq!(registry.list_operations().len(), 1);
        assert!(registry.get_operation("add").is_some());
    }

    #[test]
    fn test_duplicate_registration_fails() {
        let registry: DispatchRegistry<f32> = DispatchRegistry::new();

        let desc1 = OperationDescriptor::new("add", "binary");
        let desc2 = OperationDescriptor::new("add", "binary");

        registry.register_operation(desc1).unwrap();
        assert!(registry.register_operation(desc2).is_err());
    }

    #[test]
    fn test_kernel_registration() {
        let registry: DispatchRegistry<f32> = DispatchRegistry::new();

        // Register operation
        let desc = OperationDescriptor::new("abs", "unary");
        registry.register_operation(desc).unwrap();

        // Register CPU kernel
        fn abs_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
            let data = x.data();
            let abs_data: Vec<f32> = data.iter().map(|v| v.abs()).collect();
            let array =
                scirs2_core::ndarray::ArrayD::from_shape_vec(x.shape().dims(), abs_data).unwrap();
            Ok(Tensor::from_array(array))
        }

        let kernel = KernelImplementation::unary(BackendType::Cpu, abs_cpu);
        registry.register_kernel("abs", kernel).unwrap();

        assert_eq!(registry.available_backends("abs").len(), 1);
    }

    #[test]
    fn test_unary_dispatch() {
        let registry: DispatchRegistry<f32> = DispatchRegistry::new();

        // Register operation
        let desc = OperationDescriptor::new("negate", "unary");
        registry.register_operation(desc).unwrap();

        // Register CPU kernel
        fn negate_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
            let data = x.data();
            let neg_data: Vec<f32> = data.iter().map(|v| -v).collect();
            let array =
                scirs2_core::ndarray::ArrayD::from_shape_vec(x.shape().dims(), neg_data).unwrap();
            Ok(Tensor::from_array(array))
        }

        let kernel = KernelImplementation::unary(BackendType::Cpu, negate_cpu);
        registry.register_kernel("negate", kernel).unwrap();

        // Test dispatch
        let input = Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn());
        let result = registry.dispatch_unary("negate", &input).unwrap();

        assert_eq!(result.data(), &[-1.0f32, -2.0, -3.0]);
    }

    #[test]
    fn test_binary_dispatch() {
        let registry: DispatchRegistry<f32> = DispatchRegistry::new();

        // Register operation
        let desc = OperationDescriptor::new("add", "binary");
        registry.register_operation(desc).unwrap();

        // Register CPU kernel
        fn add_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
            let a_data = a.data();
            let b_data = b.data();
            let sum_data: Vec<f32> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(x, y)| x + y)
                .collect();
            let array =
                scirs2_core::ndarray::ArrayD::from_shape_vec(a.shape().dims(), sum_data).unwrap();
            Ok(Tensor::from_array(array))
        }

        let kernel = KernelImplementation::binary(BackendType::Cpu, add_cpu);
        registry.register_kernel("add", kernel).unwrap();

        // Test dispatch
        let a = Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn());
        let b = Tensor::from_array(array![4.0f32, 5.0, 6.0].into_dyn());
        let result = registry.dispatch_binary("add", &a, &b).unwrap();

        assert_eq!(result.data(), &[5.0f32, 7.0, 9.0]);
    }

    #[test]
    fn test_device_mismatch_error() {
        let registry: DispatchRegistry<f32> = DispatchRegistry::new();

        let desc = OperationDescriptor::new("add", "binary");
        registry.register_operation(desc).unwrap();

        fn add_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
            Ok(a.clone())
        }

        let kernel = KernelImplementation::binary(BackendType::Cpu, add_cpu);
        registry.register_kernel("add", kernel).unwrap();

        let a = Tensor::from_array(array![1.0f32].into_dyn());
        let b = Tensor::from_array(array![2.0f32].into_dyn());

        // Both on CPU, should work
        let result = registry.dispatch_binary("add", &a, &b);
        assert!(result.is_ok());
    }

    #[test]
    fn test_global_registry_access() {
        let registry = get_registry::<f32>();
        assert!(registry.is_some());

        let registry = get_registry::<f64>();
        assert!(registry.is_some());

        let registry = get_registry::<i32>();
        assert!(registry.is_some());
    }
}
