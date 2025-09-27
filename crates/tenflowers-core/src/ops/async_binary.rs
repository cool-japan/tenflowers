use crate::ops::binary::{AddOp, BinaryOp, DivOp, MulOp, PReLUOp, PowOp, SubOp};
#[cfg(feature = "gpu")]
use crate::ops::hybrid_scheduler::{HybridWorkScheduler, WorkItem, WorkType};

// Re-export WorkPriority for public API
#[cfg(feature = "gpu")]
pub use crate::ops::hybrid_scheduler::WorkPriority;
#[cfg(test)]
use crate::tensor::TensorStorage;
/// Async binary operations with CPU-GPU overlap using hybrid scheduler
use crate::{Device, Result, Tensor, TensorError};

#[cfg(not(feature = "gpu"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WorkPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

use crate::device::async_execution::AsyncExecutor;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

#[cfg(feature = "gpu")]
use crate::gpu::multi_stream_executor::MultiStreamGpuExecutor;

/// Async binary operation future
pub struct AsyncBinaryOpFuture<T> {
    inner: Pin<Box<dyn Future<Output = Result<Tensor<T>>> + Send>>,
}

impl<T> Future for AsyncBinaryOpFuture<T> {
    type Output = Result<Tensor<T>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(cx)
    }
}

/// Async binary operation executor with CPU-GPU overlap
pub struct AsyncBinaryOperationExecutor {
    #[cfg(feature = "gpu")]
    hybrid_scheduler: Option<Arc<HybridWorkScheduler>>,
    #[cfg(not(feature = "gpu"))]
    cpu_executor: Arc<AsyncExecutor>,
}

impl AsyncBinaryOperationExecutor {
    /// Create a new async binary operation executor
    #[cfg(feature = "gpu")]
    pub fn new(device_id: usize) -> Result<Self> {
        // Create CPU executor
        let cpu_executor = Arc::new(AsyncExecutor::new(Device::Cpu));

        // Create GPU executor
        let gpu_ctx = crate::device::context::get_gpu_context(device_id)?;
        let gpu_executor = Arc::new(MultiStreamGpuExecutor::new(
            gpu_ctx.device.clone(),
            gpu_ctx.queue.clone(),
        ));

        // Create hybrid scheduler
        let hybrid_scheduler = Arc::new(HybridWorkScheduler::new(cpu_executor, gpu_executor));

        Ok(Self {
            hybrid_scheduler: Some(hybrid_scheduler),
        })
    }

    /// Create a new async binary operation executor (CPU-only)
    #[cfg(not(feature = "gpu"))]
    pub fn new(_device_id: usize) -> Result<Self> {
        let cpu_executor = Arc::new(AsyncExecutor::new(Device::Cpu));
        Ok(Self { cpu_executor })
    }

    /// Execute an async binary operation with CPU-GPU overlap
    pub fn execute_async<T, Op>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        op: Op,
    ) -> AsyncBinaryOpFuture<T>
    where
        T: Clone
            + Default
            + num_traits::Zero
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
        Op: BinaryOp<T> + Send + Sync + 'static,
    {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref scheduler) = self.hybrid_scheduler {
                // Create work item for the operation
                let input_size = a.shape().elements() + b.shape().elements();
                let dtype = std::any::type_name::<T>();

                let work_item = scheduler.create_binary_op_work(
                    op.name(),
                    input_size,
                    dtype,
                    crate::ops::hybrid_scheduler::WorkPriority::Normal,
                );

                // Clone tensors and operation for async execution
                let a_clone = a.clone();
                let b_clone = b.clone();
                let scheduler_clone = Arc::clone(scheduler);

                let future = async move {
                    // Submit work to hybrid scheduler
                    let work_future = scheduler_clone.submit_work(work_item);

                    // Execute the actual operation concurrently
                    let result = std::thread::spawn(move || {
                        crate::ops::binary::binary_op(&a_clone, &b_clone, op)
                    })
                    .join();

                    // Wait for scheduler to complete
                    work_future.await?;

                    // Return the result
                    result.map_err(|e| {
                        TensorError::compute_error_simple(format!(
                            "Async execution failed: {:?}",
                            e
                        ))
                    })?
                };

                return AsyncBinaryOpFuture {
                    inner: Box::pin(future),
                };
            }
        }

        // Fallback to CPU-only execution
        let a_clone = a.clone();
        let b_clone = b.clone();
        let future = async move {
            std::thread::spawn(move || crate::ops::binary::binary_op(&a_clone, &b_clone, op))
                .join()
                .map_err(|e| {
                    TensorError::compute_error_simple(format!("Async execution failed: {e:?}"))
                })?
        };

        AsyncBinaryOpFuture {
            inner: Box::pin(future),
        }
    }

    /// Execute an async binary operation with custom priority
    pub fn execute_async_with_priority<T, Op>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        op: Op,
        _priority: WorkPriority,
    ) -> AsyncBinaryOpFuture<T>
    where
        T: Clone
            + Default
            + num_traits::Zero
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
        Op: BinaryOp<T> + Send + Sync + 'static,
    {
        // Create work item with custom priority
        let _input_size = a.shape().elements() + b.shape().elements();
        let _dtype = std::any::type_name::<T>();

        #[cfg(feature = "gpu")]
        {
            if let Some(ref scheduler) = self.hybrid_scheduler {
                let work_item =
                    scheduler.create_binary_op_work(op.name(), _input_size, _dtype, _priority);

                // Clone tensors and operation for async execution
                let a_clone = a.clone();
                let b_clone = b.clone();
                let scheduler_clone = Arc::clone(scheduler);

                let future = async move {
                    // Submit work to hybrid scheduler with custom priority
                    let work_future = scheduler_clone.submit_work(work_item);

                    // Execute the actual operation concurrently
                    let result = std::thread::spawn(move || {
                        crate::ops::binary::binary_op(&a_clone, &b_clone, op)
                    })
                    .join();

                    // Wait for scheduler to complete
                    work_future.await?;

                    // Return the result
                    result.map_err(|e| {
                        TensorError::compute_error_simple(format!(
                            "Async execution failed: {:?}",
                            e
                        ))
                    })?
                };

                return AsyncBinaryOpFuture {
                    inner: Box::pin(future),
                };
            }
        }

        // Fallback to CPU-only execution
        let a_clone = a.clone();
        let b_clone = b.clone();
        let future = async move {
            std::thread::spawn(move || crate::ops::binary::binary_op(&a_clone, &b_clone, op))
                .join()
                .map_err(|e| {
                    TensorError::compute_error_simple(format!("Async execution failed: {e:?}"))
                })?
        };

        AsyncBinaryOpFuture {
            inner: Box::pin(future),
        }
    }

    /// Execute multiple binary operations concurrently with CPU-GPU overlap
    pub async fn execute_batch_async<T, Op>(
        &self,
        operations: Vec<(&Tensor<T>, &Tensor<T>, Op)>,
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Clone
            + Default
            + num_traits::Zero
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
        Op: BinaryOp<T> + Send + Sync + Clone + 'static,
    {
        let mut futures = Vec::new();

        for (a, b, op) in operations {
            let future = self.execute_async(a, b, op);
            futures.push(future);
        }

        // Execute all operations concurrently
        let mut results = Vec::new();
        for future in futures {
            results.push(future.await?);
        }

        Ok(results)
    }

    /// Synchronize all pending operations
    pub fn synchronize(&self) {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref scheduler) = self.hybrid_scheduler {
                scheduler.synchronize_all();
            }
        }
    }

    /// Check if executor is idle
    pub fn is_idle(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref scheduler) = self.hybrid_scheduler {
                return scheduler.is_idle();
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            self.cpu_executor.queue_length() == 0
        }

        #[cfg(feature = "gpu")]
        true
    }
}

/// Global async binary operation executor
static GLOBAL_ASYNC_EXECUTOR: std::sync::OnceLock<AsyncBinaryOperationExecutor> =
    std::sync::OnceLock::new();

/// Get the global async binary operation executor
pub fn global_async_executor() -> &'static AsyncBinaryOperationExecutor {
    GLOBAL_ASYNC_EXECUTOR.get_or_init(|| {
        AsyncBinaryOperationExecutor::new(0).unwrap_or_else(|_| {
            // Fallback to CPU-only if GPU initialization fails
            #[cfg(not(feature = "gpu"))]
            {
                AsyncBinaryOperationExecutor::new(0).unwrap()
            }
            #[cfg(feature = "gpu")]
            {
                // Create a fallback CPU-only executor
                AsyncBinaryOperationExecutor {
                    hybrid_scheduler: None,
                }
            }
        })
    })
}

/// Async binary operation functions using the global executor
/// Async add operation
pub async fn add_async<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + num_traits::Zero
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    global_async_executor().execute_async(a, b, AddOp).await
}

/// Async sub operation
pub async fn sub_async<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + num_traits::Zero
        + std::ops::Sub<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    global_async_executor().execute_async(a, b, SubOp).await
}

/// Async mul operation
pub async fn mul_async<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + num_traits::Zero
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    global_async_executor().execute_async(a, b, MulOp).await
}

/// Async div operation
pub async fn div_async<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + num_traits::Zero
        + std::ops::Div<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    global_async_executor().execute_async(a, b, DivOp).await
}

/// Async pow operation
pub async fn pow_async<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + num_traits::Zero
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    global_async_executor().execute_async(a, b, PowOp).await
}

/// Async PReLU operation
pub async fn prelu_async<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + num_traits::Zero
        + num_traits::Float
        + PartialOrd
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    global_async_executor().execute_async(a, b, PReLUOp).await
}

/// Async add operation with priority
pub async fn add_async_priority<T>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    priority: WorkPriority,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + num_traits::Zero
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    global_async_executor()
        .execute_async_with_priority(a, b, AddOp, priority)
        .await
}

/// Async mul operation with priority
pub async fn mul_async_priority<T>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    priority: WorkPriority,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + num_traits::Zero
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    global_async_executor()
        .execute_async_with_priority(a, b, MulOp, priority)
        .await
}

/// Batch processing for multiple operations
pub async fn batch_add_async<T>(operations: Vec<(&Tensor<T>, &Tensor<T>)>) -> Result<Vec<Tensor<T>>>
where
    T: Clone
        + Default
        + num_traits::Zero
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let ops: Vec<_> = operations.into_iter().map(|(a, b)| (a, b, AddOp)).collect();
    global_async_executor().execute_batch_async(ops).await
}

/// Batch processing for multiple multiplication operations
pub async fn batch_mul_async<T>(operations: Vec<(&Tensor<T>, &Tensor<T>)>) -> Result<Vec<Tensor<T>>>
where
    T: Clone
        + Default
        + num_traits::Zero
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let ops: Vec<_> = operations.into_iter().map(|(a, b)| (a, b, MulOp)).collect();
    global_async_executor().execute_batch_async(ops).await
}

/// Synchronize all async operations
pub fn synchronize_async_operations() {
    global_async_executor().synchronize();
}

/// Check if async operations are idle
pub fn is_async_operations_idle() -> bool {
    global_async_executor().is_idle()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_async_executor_creation() {
        let executor = AsyncBinaryOperationExecutor::new(0).unwrap();

        // Test that executor starts idle
        assert!(executor.is_idle());
    }

    #[test]
    fn test_sync_fallback() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();

        // Test that sync operations still work
        let result = crate::ops::add(&a, &b).unwrap();
        let expected = vec![5.0, 7.0, 9.0];

        if let TensorStorage::Cpu(arr) = &result.storage {
            assert_eq!(arr.as_slice().unwrap(), &expected);
        }
    }

    #[test]
    fn test_work_priority_ordering() {
        let high = self::WorkPriority::High;
        let normal = self::WorkPriority::Normal;
        let low = self::WorkPriority::Low;

        assert!(high > normal);
        assert!(normal > low);
    }
}
