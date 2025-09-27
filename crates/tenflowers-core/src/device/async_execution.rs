use crate::{Device, Result, TensorError};
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Async operation handle
pub struct AsyncOperation {
    id: u64,
    device: Device,
    state: Arc<Mutex<OperationState>>,
}

/// State of an async operation
#[derive(Debug)]
enum OperationState {
    Pending,
    Running,
    Completed,
    Failed(String),
}

impl AsyncOperation {
    pub fn new(id: u64, device: Device) -> Self {
        Self {
            id,
            device,
            state: Arc::new(Mutex::new(OperationState::Pending)),
        }
    }

    /// Get operation ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get device
    pub fn device(&self) -> Device {
        self.device
    }

    /// Check if operation is complete
    pub fn is_complete(&self) -> bool {
        matches!(*self.state.lock().unwrap(), OperationState::Completed)
    }

    /// Check if operation failed
    pub fn is_failed(&self) -> bool {
        matches!(*self.state.lock().unwrap(), OperationState::Failed(_))
    }

    /// Wait for operation to complete
    pub async fn wait(&self) -> Result<()> {
        AsyncWaitFuture::new(Arc::clone(&self.state)).await
    }

    /// Mark operation as completed
    pub(crate) fn complete(&self) {
        let mut state = self.state.lock().unwrap();
        if matches!(*state, OperationState::Running) {
            *state = OperationState::Completed;
        }
    }

    /// Mark operation as failed
    pub(crate) fn fail(&self, error: String) {
        let mut state = self.state.lock().unwrap();
        if matches!(*state, OperationState::Running) {
            *state = OperationState::Failed(error);
        }
    }

    /// Mark operation as running
    pub(crate) fn start(&self) {
        let mut state = self.state.lock().unwrap();
        if matches!(*state, OperationState::Pending) {
            *state = OperationState::Running;
        }
    }
}

/// Future for waiting on async operations
struct AsyncWaitFuture {
    state: Arc<Mutex<OperationState>>,
    waker: Option<Waker>,
}

impl AsyncWaitFuture {
    fn new(state: Arc<Mutex<OperationState>>) -> Self {
        Self { state, waker: None }
    }
}

impl Future for AsyncWaitFuture {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let state = self.state.lock().unwrap();
        match *state {
            OperationState::Completed => Poll::Ready(Ok(())),
            OperationState::Failed(ref err) => {
                Poll::Ready(Err(TensorError::compute_error_simple(err.clone())))
            }
            OperationState::Pending | OperationState::Running => {
                drop(state); // Release the lock before setting waker
                self.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

/// Async execution engine with enhanced batching and priority support
pub struct AsyncExecutor {
    device: Device,
    queue: Arc<Mutex<VecDeque<PendingOperation>>>,
    high_priority_queue: Arc<Mutex<VecDeque<PendingOperation>>>,
    next_id: Arc<Mutex<u64>>,
    batch_size: usize,
    is_processing: Arc<Mutex<bool>>,
    stats: Arc<Mutex<ExecutorStats>>,
}

/// Execution statistics
#[derive(Debug, Default, Clone)]
pub struct ExecutorStats {
    pub operations_completed: u64,
    pub operations_failed: u64,
    pub total_execution_time_us: u64,
    pub average_queue_length: f64,
    pub batch_efficiency: f64,
}

/// Operation priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    Low,
    Normal,
    High,
    Critical,
}

/// Pending operation in the queue
#[allow(dead_code)]
struct PendingOperation {
    operation: Box<dyn AsyncKernel>,
    handle: AsyncOperation,
    priority: Priority,
    submitted_at: std::time::Instant,
}

/// Trait for async kernels with enhanced capabilities
pub trait AsyncKernel: Send + Sync {
    /// Execute the kernel asynchronously
    fn execute(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send>>;

    /// Get kernel name for debugging
    fn name(&self) -> &str;

    /// Get estimated execution time in microseconds
    fn estimated_time_us(&self) -> u64 {
        1000 // Default 1ms
    }

    /// Check if this kernel can be batched with another
    fn can_batch_with(&self, _other: &dyn AsyncKernel) -> bool {
        false // Default: no batching
    }

    /// Get kernel priority
    fn priority(&self) -> Priority {
        Priority::Normal
    }

    /// Get memory requirements in bytes
    fn memory_requirements(&self) -> usize {
        0 // Default: unknown
    }

    /// Check if kernel supports concurrent execution
    fn supports_concurrency(&self) -> bool {
        false // Default: sequential execution
    }
}

impl AsyncExecutor {
    /// Create a new async executor for a device
    pub fn new(device: Device) -> Self {
        Self::with_batch_size(device, 8) // Default batch size of 8
    }

    /// Create a new async executor with specified batch size
    pub fn with_batch_size(device: Device, batch_size: usize) -> Self {
        Self {
            device,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            high_priority_queue: Arc::new(Mutex::new(VecDeque::new())),
            next_id: Arc::new(Mutex::new(0)),
            batch_size,
            is_processing: Arc::new(Mutex::new(false)),
            stats: Arc::new(Mutex::new(ExecutorStats::default())),
        }
    }

    /// Submit an async operation
    pub fn submit(&self, kernel: Box<dyn AsyncKernel>) -> AsyncOperation {
        self.submit_with_priority(kernel, Priority::Normal)
    }

    /// Submit an async operation with specified priority
    pub fn submit_with_priority(
        &self,
        kernel: Box<dyn AsyncKernel>,
        priority: Priority,
    ) -> AsyncOperation {
        let id = {
            let mut next_id = self.next_id.lock().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let handle = AsyncOperation::new(id, self.device);
        let pending = PendingOperation {
            operation: kernel,
            handle: AsyncOperation::new(id, self.device),
            priority,
            submitted_at: std::time::Instant::now(),
        };

        // Select appropriate queue based on priority
        match priority {
            Priority::Critical | Priority::High => {
                let mut queue = self.high_priority_queue.lock().unwrap();
                queue.push_back(pending);
            }
            Priority::Normal | Priority::Low => {
                let mut queue = self.queue.lock().unwrap();
                queue.push_back(pending);
            }
        }

        handle
    }

    /// Process pending operations with advanced batching and priority handling
    pub async fn process_queue(&self) -> Result<()> {
        // Check if already processing to avoid race conditions
        {
            let mut is_processing = self.is_processing.lock().unwrap();
            if *is_processing {
                return Ok(());
            }
            *is_processing = true;
        }

        let result = self.process_queue_internal().await;

        // Release processing lock
        {
            let mut is_processing = self.is_processing.lock().unwrap();
            *is_processing = false;
        }

        result
    }

    /// Internal queue processing logic
    async fn process_queue_internal(&self) -> Result<()> {
        // Process high priority queue first
        if let Some(batch) = self.get_next_batch(true) {
            self.execute_batch(batch).await?;
        } else if let Some(batch) = self.get_next_batch(false) {
            self.execute_batch(batch).await?;
        }

        Ok(())
    }

    /// Get next batch of operations to execute
    fn get_next_batch(&self, high_priority: bool) -> Option<Vec<PendingOperation>> {
        let queue = if high_priority {
            &self.high_priority_queue
        } else {
            &self.queue
        };

        let mut queue_guard = queue.lock().unwrap();
        if queue_guard.is_empty() {
            return None;
        }

        let mut batch = Vec::new();
        let first_op = queue_guard.pop_front().unwrap();
        batch.push(first_op);

        // Try to batch compatible operations
        while batch.len() < self.batch_size && !queue_guard.is_empty() {
            let next_op = queue_guard.front().unwrap();

            // Check if operations can be batched
            if batch[0]
                .operation
                .can_batch_with(next_op.operation.as_ref())
            {
                batch.push(queue_guard.pop_front().unwrap());
            } else {
                break;
            }
        }

        Some(batch)
    }

    /// Execute a batch of operations
    async fn execute_batch(&self, batch: Vec<PendingOperation>) -> Result<()> {
        let start_time = std::time::Instant::now();
        let batch_size = batch.len();

        // Start all operations
        for op in &batch {
            op.handle.start();
        }

        // Execute operations (potentially in parallel for concurrent kernels)
        let concurrent_ops: Vec<_> = batch
            .iter()
            .filter(|op| op.operation.supports_concurrency())
            .collect();

        let sequential_ops: Vec<_> = batch
            .iter()
            .filter(|op| !op.operation.supports_concurrency())
            .collect();

        // Execute concurrent operations in parallel
        if !concurrent_ops.is_empty() {
            let futures: Vec<_> = concurrent_ops
                .iter()
                .map(|op| op.operation.execute())
                .collect();

            // Wait for all concurrent operations
            for (i, future) in futures.into_iter().enumerate() {
                match future.await {
                    Ok(()) => concurrent_ops[i].handle.complete(),
                    Err(e) => concurrent_ops[i].handle.fail(e.to_string()),
                }
            }
        }

        // Execute sequential operations one by one
        for op in sequential_ops {
            match op.operation.execute().await {
                Ok(()) => op.handle.complete(),
                Err(e) => op.handle.fail(e.to_string()),
            }
        }

        // Update statistics
        let execution_time = start_time.elapsed();
        self.update_stats(batch_size, execution_time, true);

        Ok(())
    }

    /// Update execution statistics
    fn update_stats(&self, batch_size: usize, execution_time: std::time::Duration, success: bool) {
        let mut stats = self.stats.lock().unwrap();

        if success {
            stats.operations_completed += batch_size as u64;
        } else {
            stats.operations_failed += batch_size as u64;
        }

        stats.total_execution_time_us += execution_time.as_micros() as u64;

        // Update batch efficiency (higher is better)
        stats.batch_efficiency = batch_size as f64;

        // Update average queue length
        let current_queue_len = self.queue_length() as f64;
        stats.average_queue_length = (stats.average_queue_length * 0.9) + (current_queue_len * 0.1);
    }

    /// Get total queue length (both normal and high priority)
    pub fn queue_length(&self) -> usize {
        let normal_len = self.queue.lock().unwrap().len();
        let high_len = self.high_priority_queue.lock().unwrap().len();
        normal_len + high_len
    }

    /// Get queue length by priority
    pub fn queue_length_by_priority(&self, high_priority: bool) -> usize {
        if high_priority {
            self.high_priority_queue.lock().unwrap().len()
        } else {
            self.queue.lock().unwrap().len()
        }
    }

    /// Clear all pending operations
    pub fn clear_queue(&self) {
        // Clear normal priority queue
        {
            let mut queue = self.queue.lock().unwrap();
            for op in queue.drain(..) {
                op.handle.fail("Operation cancelled".to_string());
            }
        }

        // Clear high priority queue
        {
            let mut queue = self.high_priority_queue.lock().unwrap();
            for op in queue.drain(..) {
                op.handle.fail("Operation cancelled".to_string());
            }
        }
    }

    /// Get execution statistics
    pub fn get_stats(&self) -> ExecutorStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset execution statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = ExecutorStats::default();
    }

    /// Get device this executor manages
    pub fn device(&self) -> Device {
        self.device
    }

    /// Check if executor is currently processing
    pub fn is_processing(&self) -> bool {
        *self.is_processing.lock().unwrap()
    }
}

/// Global async executor manager
pub struct AsyncExecutorManager {
    executors: Arc<Mutex<std::collections::HashMap<Device, Arc<AsyncExecutor>>>>,
}

impl AsyncExecutorManager {
    /// Create a new executor manager
    pub fn new() -> Self {
        Self {
            executors: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Get or create executor for a device
    pub fn get_executor(&self, device: Device) -> Arc<AsyncExecutor> {
        let mut executors = self.executors.lock().unwrap();

        if let Some(executor) = executors.get(&device) {
            Arc::clone(executor)
        } else {
            let executor = Arc::new(AsyncExecutor::new(device));
            executors.insert(device, Arc::clone(&executor));
            executor
        }
    }

    /// Process all device queues
    pub async fn process_all(&self) -> Result<()> {
        let executors = {
            let executors = self.executors.lock().unwrap();
            executors.values().cloned().collect::<Vec<_>>()
        };

        for executor in executors {
            executor.process_queue().await?;
        }

        Ok(())
    }
}

impl Default for AsyncExecutorManager {
    fn default() -> Self {
        Self::new()
    }
}

lazy_static::lazy_static! {
    /// Global executor manager instance
    pub static ref ASYNC_EXECUTOR_MANAGER: AsyncExecutorManager = AsyncExecutorManager::new();
}

/// Submit an async operation to the appropriate device executor
pub fn submit_async_operation(device: Device, kernel: Box<dyn AsyncKernel>) -> AsyncOperation {
    let executor = ASYNC_EXECUTOR_MANAGER.get_executor(device);
    executor.submit(kernel)
}

/// Example async kernel implementations
pub mod kernels {
    use super::*;
    use std::time::Duration;

    /// Dummy async kernel for testing
    pub struct DummyKernel {
        name: String,
        delay_ms: u64,
    }

    impl DummyKernel {
        pub fn new(name: String, delay_ms: u64) -> Self {
            Self { name, delay_ms }
        }
    }

    impl AsyncKernel for DummyKernel {
        fn execute(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
            let delay = Duration::from_millis(self.delay_ms);
            Box::pin(async move {
                // Simple delay simulation without tokio
                std::thread::sleep(delay);
                Ok(())
            })
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn estimated_time_us(&self) -> u64 {
            self.delay_ms * 1000
        }
    }

    /// Matrix multiplication async kernel
    pub struct MatMulKernel {
        pub m: usize,
        pub k: usize,
        pub n: usize,
        pub batch_size: usize,
    }

    impl AsyncKernel for MatMulKernel {
        fn execute(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
            // In real implementation, this would launch GPU compute shader
            let ops = self.batch_size * self.m * self.k * self.n * 2;
            let estimated_time_us = (ops / 1_000_000).max(100); // At least 100us

            Box::pin(async move {
                // Simulate GPU execution time
                std::thread::sleep(Duration::from_micros(estimated_time_us as u64));
                Ok(())
            })
        }

        fn name(&self) -> &str {
            "matmul"
        }

        fn estimated_time_us(&self) -> u64 {
            let ops = self.batch_size * self.m * self.k * self.n * 2;
            (ops / 1_000_000).max(100) as u64
        }
    }

    /// Convolution async kernel
    pub struct ConvKernel {
        pub batch_size: usize,
        pub in_channels: usize,
        pub out_channels: usize,
        pub input_size: (usize, usize),
        pub kernel_size: (usize, usize),
    }

    impl AsyncKernel for ConvKernel {
        fn execute(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
            // Estimate convolution complexity
            let ops = self.batch_size
                * self.out_channels
                * self.input_size.0
                * self.input_size.1
                * self.in_channels
                * self.kernel_size.0
                * self.kernel_size.1
                * 2;
            let estimated_time_us = (ops / 10_000_000).max(50); // Faster than matmul

            Box::pin(async move {
                std::thread::sleep(Duration::from_micros(estimated_time_us as u64));
                Ok(())
            })
        }

        fn name(&self) -> &str {
            "conv2d"
        }

        fn estimated_time_us(&self) -> u64 {
            let ops = self.batch_size
                * self.out_channels
                * self.input_size.0
                * self.input_size.1
                * self.in_channels
                * self.kernel_size.0
                * self.kernel_size.1
                * 2;
            (ops / 10_000_000).max(50) as u64
        }

        fn supports_concurrency(&self) -> bool {
            true // Conv operations can often run concurrently
        }
    }

    /// GPU Tensor manipulation async kernel
    pub struct TensorManipulationKernel {
        pub operation: String,
        pub input_size: usize,
        pub output_size: usize,
        pub complexity_factor: f64, // Operation-specific complexity multiplier
    }

    impl TensorManipulationKernel {
        pub fn new_gather(input_size: usize, indices_size: usize) -> Self {
            Self {
                operation: "gather".to_string(),
                input_size,
                output_size: indices_size,
                complexity_factor: 1.2, // Gather has some indexing overhead
            }
        }

        pub fn new_scatter(tensor_size: usize, updates_size: usize) -> Self {
            Self {
                operation: "scatter".to_string(),
                input_size: tensor_size + updates_size,
                output_size: tensor_size,
                complexity_factor: 1.5, // Scatter has more complex indexing
            }
        }

        pub fn new_roll(size: usize) -> Self {
            Self {
                operation: "roll".to_string(),
                input_size: size,
                output_size: size,
                complexity_factor: 1.0, // Roll is relatively simple
            }
        }

        pub fn new_where(_condition_size: usize, total_size: usize) -> Self {
            Self {
                operation: "where".to_string(),
                input_size: total_size,
                output_size: total_size,
                complexity_factor: 0.8, // Where is simple conditional logic
            }
        }
    }

    impl AsyncKernel for TensorManipulationKernel {
        fn execute(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
            let base_time =
                (self.input_size as f64 * self.complexity_factor / 1_000_000.0).max(10.0);
            let estimated_time_us = base_time as u64;

            Box::pin(async move {
                // Simulate GPU kernel execution
                std::thread::sleep(Duration::from_micros(estimated_time_us));
                Ok(())
            })
        }

        fn name(&self) -> &str {
            &self.operation
        }

        fn estimated_time_us(&self) -> u64 {
            (self.input_size as f64 * self.complexity_factor / 1_000_000.0).max(10.0) as u64
        }

        fn can_batch_with(&self, other: &dyn AsyncKernel) -> bool {
            // Manipulation operations of the same type can potentially be batched
            other.name() == self.name()
        }

        fn priority(&self) -> Priority {
            match self.operation.as_str() {
                "gather" | "scatter" => Priority::High, // Data movement is high priority
                "where" => Priority::Normal,
                _ => Priority::Low,
            }
        }

        fn supports_concurrency(&self) -> bool {
            match self.operation.as_str() {
                "roll" | "where" => true,      // These can run concurrently
                "gather" | "scatter" => false, // These may have data dependencies
                _ => false,
            }
        }

        fn memory_requirements(&self) -> usize {
            self.input_size + self.output_size
        }
    }

    /// Reduction operations async kernel
    pub struct ReductionKernel {
        pub operation: String,
        pub input_size: usize,
        pub output_size: usize,
        pub num_axes: usize,
    }

    impl ReductionKernel {
        pub fn new_sum(input_size: usize, output_size: usize, num_axes: usize) -> Self {
            Self {
                operation: "sum".to_string(),
                input_size,
                output_size,
                num_axes,
            }
        }

        pub fn new_mean(input_size: usize, output_size: usize, num_axes: usize) -> Self {
            Self {
                operation: "mean".to_string(),
                input_size,
                output_size,
                num_axes,
            }
        }

        pub fn new_max(input_size: usize, output_size: usize, num_axes: usize) -> Self {
            Self {
                operation: "max".to_string(),
                input_size,
                output_size,
                num_axes,
            }
        }
    }

    impl AsyncKernel for ReductionKernel {
        fn execute(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
            // Reduction complexity depends on input size and number of reduction steps
            let complexity = self.input_size as f64 * (self.num_axes as f64).log2();
            let estimated_time_us = (complexity / 5_000_000.0).max(5.0) as u64;

            Box::pin(async move {
                std::thread::sleep(Duration::from_micros(estimated_time_us));
                Ok(())
            })
        }

        fn name(&self) -> &str {
            &self.operation
        }

        fn estimated_time_us(&self) -> u64 {
            let complexity = self.input_size as f64 * (self.num_axes as f64).log2();
            (complexity / 5_000_000.0).max(5.0) as u64
        }

        fn can_batch_with(&self, other: &dyn AsyncKernel) -> bool {
            // Reduction operations can sometimes be fused
            other.name().starts_with("sum")
                || other.name().starts_with("mean")
                || other.name().starts_with("max")
        }

        fn priority(&self) -> Priority {
            Priority::Normal
        }

        fn supports_concurrency(&self) -> bool {
            true // Reductions can often run concurrently
        }

        fn memory_requirements(&self) -> usize {
            self.input_size + self.output_size
        }
    }
}

#[cfg(test)]
mod tests {
    use super::kernels::*;
    use super::*;

    #[test]
    fn test_async_operation() {
        // Simple test without async runtime
        // futures::executor::block_on(async {
        let executor = AsyncExecutor::new(Device::Cpu);
        let kernel = Box::new(DummyKernel::new("test".to_string(), 10));

        let handle = executor.submit(kernel);
        assert!(!handle.is_complete());

        // For now, just test basic functionality without async
        assert_eq!(handle.id(), 0);
        assert_eq!(handle.device(), Device::Cpu);
        // });
    }

    #[test]
    fn test_multiple_operations() {
        let executor = AsyncExecutor::new(Device::Cpu);

        // Submit multiple operations
        let handles = (0..5)
            .map(|i| {
                let kernel = Box::new(DummyKernel::new(format!("test_{i}"), 5));
                executor.submit(kernel)
            })
            .collect::<Vec<_>>();

        // Test that operations have sequential IDs
        for (i, handle) in handles.iter().enumerate() {
            assert_eq!(handle.id(), i as u64);
        }

        assert_eq!(executor.queue_length(), 5);
    }

    #[test]
    fn test_matmul_kernel_estimation() {
        let kernel = MatMulKernel {
            m: 1024,
            k: 1024,
            n: 1024,
            batch_size: 1,
        };

        let estimated_time = kernel.estimated_time_us();
        assert!(estimated_time > 0);

        // Larger operations should take longer
        let large_kernel = MatMulKernel {
            m: 2048,
            k: 2048,
            n: 2048,
            batch_size: 1,
        };

        assert!(large_kernel.estimated_time_us() > estimated_time);
    }

    #[test]
    fn test_tensor_manipulation_kernels() {
        // Test gather kernel
        let gather_kernel = TensorManipulationKernel::new_gather(10000, 1000);
        assert_eq!(gather_kernel.name(), "gather");
        assert_eq!(gather_kernel.priority(), Priority::High);
        assert!(!gather_kernel.supports_concurrency());

        // Test scatter kernel
        let scatter_kernel = TensorManipulationKernel::new_scatter(10000, 2000);
        assert_eq!(scatter_kernel.name(), "scatter");
        assert_eq!(scatter_kernel.priority(), Priority::High);
        assert!(!scatter_kernel.supports_concurrency());

        // Test roll kernel
        let roll_kernel = TensorManipulationKernel::new_roll(5000);
        assert_eq!(roll_kernel.name(), "roll");
        assert_eq!(roll_kernel.priority(), Priority::Low);
        assert!(roll_kernel.supports_concurrency());

        // Test where kernel
        let where_kernel = TensorManipulationKernel::new_where(1000, 5000);
        assert_eq!(where_kernel.name(), "where");
        assert_eq!(where_kernel.priority(), Priority::Normal);
        assert!(where_kernel.supports_concurrency());
    }

    #[test]
    fn test_reduction_kernels() {
        // Test sum reduction
        let sum_kernel = ReductionKernel::new_sum(10000, 100, 2);
        assert_eq!(sum_kernel.name(), "sum");
        assert_eq!(sum_kernel.priority(), Priority::Normal);
        assert!(sum_kernel.supports_concurrency());

        // Test mean reduction
        let mean_kernel = ReductionKernel::new_mean(20000, 200, 3);
        assert_eq!(mean_kernel.name(), "mean");
        assert!(mean_kernel.estimated_time_us() > 0);

        // Test max reduction
        let max_kernel = ReductionKernel::new_max(15000, 150, 2);
        assert_eq!(max_kernel.name(), "max");

        // Test batching between reduction operations
        assert!(sum_kernel.can_batch_with(&mean_kernel));
        assert!(mean_kernel.can_batch_with(&max_kernel));
    }

    #[test]
    fn test_kernel_batching() {
        let gather1 = TensorManipulationKernel::new_gather(1000, 100);
        let gather2 = TensorManipulationKernel::new_gather(2000, 200);
        let roll = TensorManipulationKernel::new_roll(1000);

        // Same operation types should be batchable
        assert!(gather1.can_batch_with(&gather2));

        // Different operation types should not be batchable
        assert!(!gather1.can_batch_with(&roll));
    }

    #[test]
    fn test_conv_kernel_concurrency() {
        let conv_kernel = ConvKernel {
            batch_size: 8,
            in_channels: 64,
            out_channels: 128,
            input_size: (224, 224),
            kernel_size: (3, 3),
        };

        assert!(conv_kernel.supports_concurrency());
        assert_eq!(conv_kernel.name(), "conv2d");
        assert!(conv_kernel.estimated_time_us() > 0);
    }
}
