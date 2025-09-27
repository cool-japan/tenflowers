#[cfg(feature = "gpu")]
use crate::device::async_execution::AsyncExecutor;
#[cfg(feature = "gpu")]
use crate::gpu::multi_stream_executor::{MultiStreamGpuExecutor, StreamPriority};
/// Hybrid CPU-GPU work scheduler for optimal CPU-GPU overlap
/// This module only works when GPU features are enabled
#[cfg(feature = "gpu")]
use crate::{Device, Result, Tensor, TensorError};
#[cfg(feature = "gpu")]
use std::collections::VecDeque;
#[cfg(feature = "gpu")]
use std::future::Future;
#[cfg(feature = "gpu")]
use std::pin::Pin;
#[cfg(feature = "gpu")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "gpu")]
use std::task::{Context, Poll};
#[cfg(feature = "gpu")]
use std::time::{Duration, Instant};

/// Work item priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WorkPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Work item type for scheduling decisions
#[derive(Debug, Clone)]
pub enum WorkType {
    BinaryOp {
        operation: String,
        input_size: usize,
        dtype: String,
    },
    Reduction {
        operation: String,
        input_size: usize,
        axis: Option<Vec<usize>>,
    },
    MatrixMultiplication {
        m: usize,
        n: usize,
        k: usize,
    },
    Convolution {
        input_shape: Vec<usize>,
        kernel_shape: Vec<usize>,
        stride: Vec<usize>,
    },
    DataTransfer {
        size: usize,
        from_device: Device,
        to_device: Device,
    },
}

/// Work item for scheduling
#[derive(Debug)]
pub struct WorkItem {
    pub id: u64,
    pub work_type: WorkType,
    pub priority: WorkPriority,
    pub estimated_duration: Duration,
    pub preferred_device: Option<Device>,
    pub created_at: Instant,
}

/// Execution strategy for a work item
#[derive(Debug, Clone)]
pub enum ExecutionStrategy {
    CpuOnly,
    GpuOnly {
        stream_priority: StreamPriority,
    },
    CpuGpuOverlap {
        cpu_work: Vec<WorkType>,
        gpu_work: Vec<WorkType>,
    },
    Adaptive, // Let scheduler decide based on current conditions
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    pub cpu_utilization: f32,
    pub gpu_utilization: f32,
    pub memory_usage: f32,
    pub pending_cpu_work: usize,
    pub pending_gpu_work: usize,
    pub last_updated: Instant,
}

/// Hybrid work scheduler that optimizes CPU-GPU overlap
pub struct HybridWorkScheduler {
    cpu_executor: Arc<AsyncExecutor>,
    gpu_executor: Arc<MultiStreamGpuExecutor>,
    work_queue: Arc<Mutex<VecDeque<WorkItem>>>,
    metrics: Arc<Mutex<ResourceMetrics>>,
    work_counter: Arc<Mutex<u64>>,
    scheduling_config: SchedulingConfig,
}

/// Configuration for scheduling decisions
#[derive(Debug, Clone)]
pub struct SchedulingConfig {
    pub cpu_threshold: f32,        // CPU utilization threshold
    pub gpu_threshold: f32,        // GPU utilization threshold
    pub small_op_threshold: usize, // Size threshold for CPU vs GPU
    pub overlap_factor: f32,       // How aggressively to use overlap
    pub adaptive_scheduling: bool, // Enable adaptive scheduling
}

impl Default for SchedulingConfig {
    fn default() -> Self {
        Self {
            cpu_threshold: 0.8,
            gpu_threshold: 0.8,
            small_op_threshold: 1024,
            overlap_factor: 0.7,
            adaptive_scheduling: true,
        }
    }
}

impl HybridWorkScheduler {
    pub fn new(
        cpu_executor: Arc<AsyncExecutor>,
        gpu_executor: Arc<MultiStreamGpuExecutor>,
    ) -> Self {
        Self {
            cpu_executor,
            gpu_executor,
            work_queue: Arc::new(Mutex::new(VecDeque::new())),
            metrics: Arc::new(Mutex::new(ResourceMetrics {
                cpu_utilization: 0.0,
                gpu_utilization: 0.0,
                memory_usage: 0.0,
                pending_cpu_work: 0,
                pending_gpu_work: 0,
                last_updated: Instant::now(),
            })),
            work_counter: Arc::new(Mutex::new(0)),
            scheduling_config: SchedulingConfig::default(),
        }
    }

    pub fn with_config(
        cpu_executor: Arc<AsyncExecutor>,
        gpu_executor: Arc<MultiStreamGpuExecutor>,
        config: SchedulingConfig,
    ) -> Self {
        let mut scheduler = Self::new(cpu_executor, gpu_executor);
        scheduler.scheduling_config = config;
        scheduler
    }

    /// Submit work to the scheduler
    pub fn submit_work(&self, work: WorkItem) -> HybridWorkFuture {
        let work_id = work.id;

        // Add work to queue
        {
            let mut queue = self.work_queue.lock().unwrap();
            queue.push_back(work);
        }

        // Try to schedule immediately
        self.schedule_pending_work();

        HybridWorkFuture {
            work_id,
            scheduler: self,
            completed: false,
        }
    }

    /// Schedule pending work items
    fn schedule_pending_work(&self) {
        let mut queue = self.work_queue.lock().unwrap();
        let metrics = self.metrics.lock().unwrap();

        // Sort by priority (higher priority first)
        let mut work_items: Vec<_> = queue.drain(..).collect();
        work_items.sort_by(|a, b| b.priority.cmp(&a.priority));

        for work in work_items {
            let strategy = self.determine_execution_strategy(&work, &metrics);
            self.execute_work(work, strategy);
        }
    }

    /// Determine optimal execution strategy for a work item
    fn determine_execution_strategy(
        &self,
        work: &WorkItem,
        metrics: &ResourceMetrics,
    ) -> ExecutionStrategy {
        if !self.scheduling_config.adaptive_scheduling {
            return ExecutionStrategy::Adaptive;
        }

        match &work.work_type {
            WorkType::BinaryOp { input_size, .. } => {
                if *input_size < self.scheduling_config.small_op_threshold {
                    // Small operations are better on CPU
                    ExecutionStrategy::CpuOnly
                } else if metrics.gpu_utilization < self.scheduling_config.gpu_threshold {
                    // GPU is available, use it
                    ExecutionStrategy::GpuOnly {
                        stream_priority: self.priority_to_stream_priority(work.priority),
                    }
                } else if metrics.cpu_utilization < self.scheduling_config.cpu_threshold {
                    // GPU is busy, use CPU
                    ExecutionStrategy::CpuOnly
                } else {
                    // Both are busy, use overlap strategy
                    ExecutionStrategy::CpuGpuOverlap {
                        cpu_work: vec![WorkType::BinaryOp {
                            operation: "preprocessing".to_string(),
                            input_size: *input_size / 2,
                            dtype: "f32".to_string(),
                        }],
                        gpu_work: vec![work.work_type.clone()],
                    }
                }
            }
            WorkType::MatrixMultiplication { m, n, k } => {
                let total_ops = m * n * k;
                if total_ops < self.scheduling_config.small_op_threshold * 100 {
                    ExecutionStrategy::CpuOnly
                } else {
                    ExecutionStrategy::GpuOnly {
                        stream_priority: StreamPriority::High,
                    }
                }
            }
            WorkType::Convolution { .. } => {
                // Convolutions are typically better on GPU
                ExecutionStrategy::GpuOnly {
                    stream_priority: StreamPriority::Normal,
                }
            }
            WorkType::DataTransfer { .. } => {
                // Data transfers should use transfer stream
                ExecutionStrategy::GpuOnly {
                    stream_priority: StreamPriority::Normal,
                }
            }
            WorkType::Reduction { input_size, .. } => {
                if *input_size < self.scheduling_config.small_op_threshold {
                    ExecutionStrategy::CpuOnly
                } else {
                    ExecutionStrategy::GpuOnly {
                        stream_priority: StreamPriority::Normal,
                    }
                }
            }
        }
    }

    /// Convert work priority to stream priority
    fn priority_to_stream_priority(&self, priority: WorkPriority) -> StreamPriority {
        match priority {
            WorkPriority::Low => StreamPriority::Low,
            WorkPriority::Normal => StreamPriority::Normal,
            WorkPriority::High => StreamPriority::High,
            WorkPriority::Critical => StreamPriority::Critical,
        }
    }

    /// Execute work item with specified strategy
    fn execute_work(&self, work: WorkItem, strategy: ExecutionStrategy) {
        match strategy {
            ExecutionStrategy::CpuOnly => {
                // Execute on CPU
                self.execute_cpu_work(work);
            }
            ExecutionStrategy::GpuOnly { stream_priority } => {
                // Execute on GPU with specified stream priority
                self.execute_gpu_work(work, stream_priority);
            }
            ExecutionStrategy::CpuGpuOverlap { cpu_work, gpu_work } => {
                // Execute with CPU-GPU overlap
                self.execute_overlapped_work(work, cpu_work, gpu_work);
            }
            ExecutionStrategy::Adaptive => {
                // Use adaptive scheduling
                let metrics = self.metrics.lock().unwrap();
                let adaptive_strategy = self.determine_execution_strategy(&work, &metrics);
                drop(metrics);
                self.execute_work(work, adaptive_strategy);
            }
        }
    }

    /// Execute work on CPU
    fn execute_cpu_work(&self, work: WorkItem) {
        // This would integrate with the actual CPU execution
        // For now, we'll just simulate the work
        println!("Executing work {} on CPU: {:?}", work.id, work.work_type);

        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.pending_cpu_work += 1;
        }
    }

    /// Execute work on GPU with specified stream priority
    fn execute_gpu_work(&self, work: WorkItem, stream_priority: StreamPriority) {
        // This would integrate with the actual GPU execution
        // For now, we'll just simulate the work
        println!(
            "Executing work {} on GPU (priority: {:?}): {:?}",
            work.id, stream_priority, work.work_type
        );

        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.pending_gpu_work += 1;
        }
    }

    /// Execute work with CPU-GPU overlap
    fn execute_overlapped_work(
        &self,
        work: WorkItem,
        cpu_work: Vec<WorkType>,
        gpu_work: Vec<WorkType>,
    ) {
        println!("Executing work {} with CPU-GPU overlap", work.id);
        println!("  CPU work: {:?}", cpu_work);
        println!("  GPU work: {:?}", gpu_work);

        // Start CPU work
        for cpu_item in cpu_work {
            // Execute CPU preprocessing/postprocessing work
            self.execute_cpu_preprocessing(cpu_item);
        }

        // Start GPU work concurrently
        for gpu_item in gpu_work {
            // Execute GPU computation work
            self.execute_gpu_computation(gpu_item);
        }

        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.pending_cpu_work += 1;
            metrics.pending_gpu_work += 1;
        }
    }

    /// Execute CPU preprocessing work
    fn execute_cpu_preprocessing(&self, work: WorkType) {
        // This would implement actual CPU preprocessing
        println!("CPU preprocessing: {:?}", work);
    }

    /// Execute GPU computation work
    fn execute_gpu_computation(&self, work: WorkType) {
        // This would implement actual GPU computation
        println!("GPU computation: {:?}", work);
    }

    /// Update resource metrics
    pub fn update_metrics(&self, cpu_util: f32, gpu_util: f32, memory_usage: f32) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.cpu_utilization = cpu_util;
        metrics.gpu_utilization = gpu_util;
        metrics.memory_usage = memory_usage;
        metrics.last_updated = Instant::now();
    }

    /// Get current resource metrics
    pub fn get_metrics(&self) -> ResourceMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Get next work ID
    fn next_work_id(&self) -> u64 {
        let mut counter = self.work_counter.lock().unwrap();
        *counter += 1;
        *counter
    }

    /// Create a work item for binary operation
    pub fn create_binary_op_work(
        &self,
        operation: &str,
        input_size: usize,
        dtype: &str,
        priority: WorkPriority,
    ) -> WorkItem {
        WorkItem {
            id: self.next_work_id(),
            work_type: WorkType::BinaryOp {
                operation: operation.to_string(),
                input_size,
                dtype: dtype.to_string(),
            },
            priority,
            estimated_duration: Duration::from_micros(input_size as u64 / 1000),
            preferred_device: None,
            created_at: Instant::now(),
        }
    }

    /// Synchronize all work
    pub fn synchronize_all(&self) {
        self.gpu_executor.synchronize_all();
        // CPU executor synchronization would go here
    }

    /// Check if scheduler is idle
    pub fn is_idle(&self) -> bool {
        let queue = self.work_queue.lock().unwrap();
        queue.is_empty() && self.gpu_executor.is_idle()
    }
}

/// Future for hybrid work execution
pub struct HybridWorkFuture<'a> {
    work_id: u64,
    scheduler: &'a HybridWorkScheduler,
    completed: bool,
}

impl<'a> Future for HybridWorkFuture<'a> {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.completed {
            return Poll::Ready(Ok(()));
        }

        // Check if work is completed
        // This is a simplified implementation
        if self.scheduler.is_idle() {
            self.completed = true;
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    }
}
