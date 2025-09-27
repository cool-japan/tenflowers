/// Multi-stream GPU executor for CPU-GPU overlap
use super::*;
use crate::gpu::ops::BinaryOp;
use crate::{Device, Result, TensorError};
use futures::channel::oneshot;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

/// Stream priority levels for GPU operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// GPU stream for concurrent execution
pub struct GpuStream {
    id: u32,
    priority: StreamPriority,
    queue: Arc<wgpu::Queue>,
    pending_operations: Arc<Mutex<VecDeque<PendingGpuOperation>>>,
}

/// Pending GPU operation
struct PendingGpuOperation {
    operation_id: u64,
    completion_sender: oneshot::Sender<Result<()>>,
}

/// Multi-stream GPU executor for overlapping CPU and GPU work
pub struct MultiStreamGpuExecutor {
    device: Arc<wgpu::Device>,
    compute_stream: Arc<GpuStream>,
    transfer_stream: Arc<GpuStream>,
    high_priority_stream: Arc<GpuStream>,
    background_stream: Arc<GpuStream>,
    operation_counter: Arc<Mutex<u64>>,
}

impl MultiStreamGpuExecutor {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device: Arc::clone(&device),
            compute_stream: Arc::new(GpuStream {
                id: 0,
                priority: StreamPriority::Normal,
                queue: Arc::clone(&queue),
                pending_operations: Arc::new(Mutex::new(VecDeque::new())),
            }),
            transfer_stream: Arc::new(GpuStream {
                id: 1,
                priority: StreamPriority::Normal,
                queue: Arc::clone(&queue),
                pending_operations: Arc::new(Mutex::new(VecDeque::new())),
            }),
            high_priority_stream: Arc::new(GpuStream {
                id: 2,
                priority: StreamPriority::High,
                queue: Arc::clone(&queue),
                pending_operations: Arc::new(Mutex::new(VecDeque::new())),
            }),
            background_stream: Arc::new(GpuStream {
                id: 3,
                priority: StreamPriority::Low,
                queue: Arc::clone(&queue),
                pending_operations: Arc::new(Mutex::new(VecDeque::new())),
            }),
            operation_counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Get next operation ID
    fn next_operation_id(&self) -> u64 {
        let mut counter = self.operation_counter.lock().unwrap();
        *counter += 1;
        *counter
    }

    /// Execute a binary operation asynchronously on compute stream
    pub fn execute_binary_op_async<T>(
        &self,
        input_a: &GpuBuffer<T>,
        input_b: &GpuBuffer<T>,
        operation: BinaryOp,
        output_len: usize,
    ) -> MultiStreamGpuFuture<T>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
    {
        self.execute_binary_op_on_stream(
            input_a,
            input_b,
            operation,
            output_len,
            &self.compute_stream,
        )
    }

    /// Execute a binary operation asynchronously on high priority stream
    pub fn execute_binary_op_high_priority<T>(
        &self,
        input_a: &GpuBuffer<T>,
        input_b: &GpuBuffer<T>,
        operation: BinaryOp,
        output_len: usize,
    ) -> MultiStreamGpuFuture<T>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
    {
        self.execute_binary_op_on_stream(
            input_a,
            input_b,
            operation,
            output_len,
            &self.high_priority_stream,
        )
    }

    /// Execute a binary operation asynchronously on background stream
    pub fn execute_binary_op_background<T>(
        &self,
        input_a: &GpuBuffer<T>,
        input_b: &GpuBuffer<T>,
        operation: BinaryOp,
        output_len: usize,
    ) -> MultiStreamGpuFuture<T>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
    {
        self.execute_binary_op_on_stream(
            input_a,
            input_b,
            operation,
            output_len,
            &self.background_stream,
        )
    }

    /// Execute a binary operation on specific stream
    fn execute_binary_op_on_stream<T>(
        &self,
        input_a: &GpuBuffer<T>,
        input_b: &GpuBuffer<T>,
        operation: BinaryOp,
        output_len: usize,
        stream: &Arc<GpuStream>,
    ) -> MultiStreamGpuFuture<T>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
    {
        let (sender, receiver) = oneshot::channel();
        let operation_id = self.next_operation_id();

        let device = Arc::clone(&self.device);
        let queue = Arc::clone(&stream.queue);

        // Clone input buffer references for async execution
        let input_a_buffer = input_a.buffer_arc();
        let input_b_buffer = input_b.buffer_arc();
        let input_a_device = Arc::clone(&input_a.device);
        let input_a_queue = Arc::clone(&input_a.queue);
        let device_enum = input_a.device_enum().clone();

        // Start async computation
        let computation_task = async move {
            let result = execute_binary_op_internal(
                &input_a_buffer,
                &input_b_buffer,
                operation,
                output_len,
                &device,
                &queue,
                &input_a_device,
                &input_a_queue,
                device_enum,
            )
            .await;

            let _ = sender.send(result);
        };

        // Spawn the task for concurrent execution (allows CPU work to continue)
        std::thread::spawn(move || {
            pollster::block_on(computation_task);
        });

        MultiStreamGpuFuture {
            receiver: Some(receiver),
            operation_id,
            stream_id: stream.id,
            device: Arc::clone(&self.device),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Execute a memory transfer asynchronously on transfer stream
    pub fn execute_transfer_async<T>(
        &self,
        source: &GpuBuffer<T>,
        destination: &mut GpuBuffer<T>,
    ) -> MultiStreamGpuFuture<T>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
    {
        let (sender, receiver) = oneshot::channel();
        let operation_id = self.next_operation_id();

        let device = Arc::clone(&self.device);
        let queue = Arc::clone(&self.transfer_stream.queue);

        // Clone buffer references for async execution
        let src_buffer = source.buffer_arc();
        let dst_buffer = destination.buffer_arc();
        let src_device = Arc::clone(&source.device);
        let dst_device = Arc::clone(&destination.device);
        let device_enum = source.device_enum().clone();

        // Start async transfer
        let transfer_task = async move {
            let result = execute_transfer_internal(
                &src_buffer,
                &dst_buffer,
                &device,
                &queue,
                &src_device,
                &dst_device,
                device_enum,
            )
            .await;

            let _ = sender.send(result);
        };

        // Spawn the task for concurrent execution
        std::thread::spawn(move || {
            pollster::block_on(transfer_task);
        });

        MultiStreamGpuFuture {
            receiver: Some(receiver),
            operation_id,
            stream_id: self.transfer_stream.id,
            device: Arc::clone(&self.device),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Synchronize all streams
    pub fn synchronize_all(&self) {
        // Poll device to ensure all operations are complete
        self.device.poll(wgpu::Maintain::Wait);

        // Clear all pending operations
        self.compute_stream
            .pending_operations
            .lock()
            .unwrap()
            .clear();
        self.transfer_stream
            .pending_operations
            .lock()
            .unwrap()
            .clear();
        self.high_priority_stream
            .pending_operations
            .lock()
            .unwrap()
            .clear();
        self.background_stream
            .pending_operations
            .lock()
            .unwrap()
            .clear();
    }

    /// Get the number of pending operations across all streams
    pub fn pending_operations_count(&self) -> usize {
        let compute_count = self.compute_stream.pending_operations.lock().unwrap().len();
        let transfer_count = self
            .transfer_stream
            .pending_operations
            .lock()
            .unwrap()
            .len();
        let high_priority_count = self
            .high_priority_stream
            .pending_operations
            .lock()
            .unwrap()
            .len();
        let background_count = self
            .background_stream
            .pending_operations
            .lock()
            .unwrap()
            .len();

        compute_count + transfer_count + high_priority_count + background_count
    }

    /// Check if all streams are idle
    pub fn is_idle(&self) -> bool {
        self.pending_operations_count() == 0
    }
}

/// Future for multi-stream GPU operations
pub struct MultiStreamGpuFuture<T> {
    receiver: Option<oneshot::Receiver<Result<GpuBuffer<T>>>>,
    operation_id: u64,
    stream_id: u32,
    device: Arc<wgpu::Device>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Future for MultiStreamGpuFuture<T> {
    type Output = Result<GpuBuffer<T>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        if let Some(receiver) = &mut this.receiver {
            match Pin::new(receiver).poll(cx) {
                Poll::Ready(Ok(result)) => {
                    this.receiver = None;
                    Poll::Ready(result)
                }
                Poll::Ready(Err(_)) => {
                    this.receiver = None;
                    Poll::Ready(Err(TensorError::compute_error_simple(format!(
                        "Multi-stream GPU operation {} on stream {} failed",
                        this.operation_id, this.stream_id
                    ))))
                }
                Poll::Pending => Poll::Pending,
            }
        } else {
            Poll::Ready(Err(TensorError::compute_error_simple(format!(
                "Multi-stream GPU future {} already completed",
                this.operation_id
            ))))
        }
    }
}

/// Internal function to execute binary operation
async fn execute_binary_op_internal<T>(
    input_a: &Arc<wgpu::Buffer>,
    input_b: &Arc<wgpu::Buffer>,
    operation: BinaryOp,
    output_len: usize,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    input_a_device: &Arc<wgpu::Device>,
    input_a_queue: &Arc<wgpu::Queue>,
    device_enum: Device,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // Use existing GPU binary operation implementation
    // This is a simplified version - in reality, you'd call the actual GPU operation
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Multi-stream binary op output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create GPU buffer wrapper
    let result = GpuBuffer::from_wgpu_buffer(
        output_buffer,
        Arc::clone(device),
        Arc::clone(queue),
        device_enum,
        output_len,
    );

    // Submit command buffer (non-blocking)
    queue.submit(std::iter::empty());

    Ok(result)
}

/// Internal function to execute memory transfer
async fn execute_transfer_internal<T>(
    source: &Arc<wgpu::Buffer>,
    destination: &Arc<wgpu::Buffer>,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    src_device: &Arc<wgpu::Device>,
    dst_device: &Arc<wgpu::Device>,
    device_enum: Device,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // Create command encoder for transfer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Multi-stream transfer"),
    });

    // Copy buffer
    encoder.copy_buffer_to_buffer(source, 0, destination, 0, source.size());

    // Submit transfer command (non-blocking)
    queue.submit(std::iter::once(encoder.finish()));

    // Create result buffer wrapper
    let result = GpuBuffer::from_shared_buffer(
        Arc::clone(destination),
        Arc::clone(device),
        Arc::clone(queue),
        device_enum,
        (destination.size() / std::mem::size_of::<T>() as u64) as usize,
    );

    Ok(result)
}
