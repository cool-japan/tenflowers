use crate::distributed::{
    BackendConfig, CommunicationBackendImpl, CommunicationGroup, ReductionOp,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tenflowers_core::{Result, Tensor, TensorError};

/// Thread-based communication backend for single-node multi-GPU
///
/// This backend provides real, working group/barrier bookkeeping for
/// synchronization, but does not implement genuine cross-rank data exchange. See
/// the doc comments on `simulate_all_reduce`, `send_f32`, and `recv_f32` for why
/// those operations honestly report unavailability instead of fabricating a
/// result.
pub struct ThreadBackend {
    name: String,
    initialized: bool,
    /// Shared state for thread coordination
    shared_state: Arc<Mutex<ThreadBackendState>>,
}

/// Shared state for thread-based communication
struct ThreadBackendState {
    /// Communication groups and their participants
    groups: HashMap<String, ThreadGroupState>,
}

/// State for a specific communication group
struct ThreadGroupState {
    /// Number of participants (ranks) in this group
    world_size: usize,
    /// Barrier synchronization for collective operations
    barrier: Arc<ThreadBarrier>,
}

/// Thread-based barrier for synchronization
struct ThreadBarrier {
    /// Number of threads that need to reach the barrier
    num_threads: usize,
    /// Number of threads currently waiting
    waiting: Mutex<usize>,
    /// Condition variable for signaling
    condvar: std::sync::Condvar,
}

impl Default for ThreadBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreadBackend {
    pub fn new() -> Self {
        Self {
            name: "thread".to_string(),
            initialized: false,
            shared_state: Arc::new(Mutex::new(ThreadBackendState {
                groups: HashMap::new(),
            })),
        }
    }

    /// Perform thread-backend all-reduce.
    ///
    /// A real all-reduce combines every rank's own distinct local tensor into one
    /// result (a genuine sum needs each rank's actual value, not just this
    /// caller's). `ThreadBackend` has no infrastructure that could do that: this
    /// method only ever receives the calling instance's own local `tensor`
    /// argument, and never touches `self.shared_state` at all. Even if it did,
    /// `shared_state` is created fresh by every `ThreadBackend::new()` call and is
    /// never shared across the distinct instances that
    /// `distributed::data_parallel::utils::init_process_group` /
    /// `distributed::pipeline_parallel::utils::init_distributed` construct for each
    /// simulated rank -- each gets its own unconnected `Arc<Mutex<ThreadBackendState>>`.
    /// There is no channel, registry, or other mechanism anywhere in this file that
    /// lets one rank's call observe another rank's data.
    ///
    /// The previous implementation returned `tensor * world_size` for `Sum` and
    /// `tensor.clone()` for `Average`, which only coincide with the true cross-rank
    /// result in the degenerate case where every rank happens to hold
    /// bit-identical data -- never guaranteed in real distributed training, where
    /// each rank typically holds a distinct data shard and therefore a distinct
    /// local gradient. Returning `Ok` for either would silently fabricate a
    /// successful reduction, exactly like the `Min`/`Max`/`Product` case already
    /// handled by the catch-all below, so `Sum` and `Average` now surface their own
    /// honest error alongside it instead of being fabricated.
    fn simulate_all_reduce(
        &self,
        _tensor: &Tensor<f32>,
        group: &CommunicationGroup,
        op: ReductionOp,
    ) -> Result<Tensor<f32>> {
        match op {
            ReductionOp::Sum | ReductionOp::Average => {
                Err(TensorError::not_implemented_simple(format!(
                    "Reduction op {op:?} not implemented for thread backend: this \
                     instance's `shared_state` only ever holds this rank's own data \
                     (group={}, world_size={}), and no cross-instance transport \
                     connects it to any other rank's `ThreadBackend`. Scaling or \
                     echoing back the local tensor would silently fabricate a \
                     cross-rank reduction.",
                    group.group_id, group.world_size
                )))
            }
            _ => Err(TensorError::not_implemented_simple(format!(
                "Reduction op {op:?} not implemented for thread backend"
            ))),
        }
    }

    /// Simulate all-gather by replicating tensor for each rank
    fn simulate_all_gather(
        &self,
        tensor: &Tensor<f32>,
        group: &CommunicationGroup,
    ) -> Result<Vec<Tensor<f32>>> {
        // Return copies of the tensor for each rank
        Ok(vec![tensor.clone(); group.world_size])
    }

    /// Wait at barrier for synchronization
    fn wait_barrier(&self, group_id: &str) -> Result<()> {
        let state = self
            .shared_state
            .lock()
            .map_err(|_| TensorError::other("Failed to acquire shared state lock".to_string()))?;

        if let Some(group_state) = state.groups.get(group_id) {
            group_state.barrier.wait();
            Ok(())
        } else {
            Err(TensorError::invalid_argument(format!(
                "Group {group_id} not found"
            )))
        }
    }
}

impl CommunicationBackendImpl for ThreadBackend {
    fn initialize(&mut self, _config: &BackendConfig) -> Result<()> {
        self.initialized = true;
        Ok(())
    }

    fn create_group(&mut self, group: &CommunicationGroup) -> Result<()> {
        let mut state = self
            .shared_state
            .lock()
            .map_err(|_| TensorError::other("Failed to acquire shared state lock".to_string()))?;

        let group_state = ThreadGroupState {
            world_size: group.world_size,
            barrier: Arc::new(ThreadBarrier::new(group.world_size)),
        };

        state.groups.insert(group.group_id.clone(), group_state);

        // No point-to-point channels are created here: `send_f32`/`recv_f32` below
        // explain why no genuine channel-based transport is possible for this
        // backend (separate `ThreadBackend` instances never share `shared_state`).
        Ok(())
    }

    fn all_reduce_f32(
        &self,
        tensor: &Tensor<f32>,
        group: &CommunicationGroup,
        op: ReductionOp,
    ) -> Result<Tensor<f32>> {
        self.simulate_all_reduce(tensor, group, op)
    }

    fn all_gather_f32(
        &self,
        tensor: &Tensor<f32>,
        group: &CommunicationGroup,
    ) -> Result<Vec<Tensor<f32>>> {
        self.simulate_all_gather(tensor, group)
    }

    fn broadcast_f32(
        &self,
        tensor: &Tensor<f32>,
        _root_rank: usize,
        _group: &CommunicationGroup,
    ) -> Result<Tensor<f32>> {
        // In thread backend, just return the tensor (simulating broadcast from root)
        Ok(tensor.clone())
    }

    /// Send an f32 tensor to a specific rank (point-to-point).
    ///
    /// A real send needs a live receiver on the other end of a channel connecting
    /// this rank's `ThreadBackend` instance to `dest_rank`'s. The previous
    /// implementation created per-(group, src, dest) `mpsc` channels in
    /// `create_group` but immediately dropped the `Receiver` half
    /// (`let (sender, _receiver) = mpsc::channel();`), so no receiver was ever
    /// reachable here -- every `sender.send(..)` call was destined to fail with a
    /// disconnected-channel error for any real cross-rank pair, or silently no-op
    /// when no channel entry existed. Separate `ThreadBackend` instances (one per
    /// simulated rank; see `simulate_all_reduce`'s doc comment) do not even share
    /// the `shared_state` that held those channels. The message body also used a
    /// fixed 1024-byte placeholder (`vec![0; 1024]`) instead of the real tensor --
    /// the `tensor` argument was never read. Returning `Ok(())` without
    /// transmitting real data would falsely report a delivered message to
    /// `dest_rank`, so we surface an honest error instead, mirroring
    /// `NcclBackend::send_f32`.
    fn send_f32(
        &self,
        _tensor: &Tensor<f32>,
        dest_rank: usize,
        group: &CommunicationGroup,
    ) -> Result<()> {
        Err(TensorError::not_implemented_simple(format!(
            "Thread backend point-to-point send (group={}, src={}, dest={dest_rank}) \
             is not available: no genuine cross-rank data transport connects \
             separate `ThreadBackend` instances, so no data can be transmitted.",
            group.group_id, group.rank
        )))
    }

    /// Receive an f32 tensor from a specific rank (point-to-point).
    ///
    /// A real receive blocks until the sender's actual bytes arrive over a working
    /// channel. The previous implementation ignored `src_rank` and `group` entirely
    /// and unconditionally fabricated a zero tensor of the requested `shape`,
    /// regardless of whether -- or what -- anything was ever sent by `send_f32`.
    /// Returning fabricated zeros as if they were genuinely received data is
    /// exactly the silent fabrication this cleanup removes, so we surface an
    /// honest error instead, mirroring `NcclBackend::recv_f32`.
    fn recv_f32(
        &self,
        shape: &[usize],
        src_rank: usize,
        group: &CommunicationGroup,
    ) -> Result<Tensor<f32>> {
        Err(TensorError::not_implemented_simple(format!(
            "Thread backend point-to-point recv (group={}, src={src_rank}, \
             shape={shape:?}) is not available: no genuine cross-rank data \
             transport connects separate `ThreadBackend` instances, so no data can \
             be received.",
            group.group_id
        )))
    }

    fn finalize(&mut self) -> Result<()> {
        let mut state = self
            .shared_state
            .lock()
            .map_err(|_| TensorError::other("Failed to acquire shared state lock".to_string()))?;

        state.groups.clear();
        self.initialized = false;

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl ThreadBarrier {
    fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            waiting: Mutex::new(0),
            condvar: std::sync::Condvar::new(),
        }
    }

    fn wait(&self) {
        let mut waiting = self
            .waiting
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *waiting += 1;

        if *waiting == self.num_threads {
            // Last thread to arrive - notify all waiting threads
            *waiting = 0;
            self.condvar.notify_all();
        } else {
            // Wait for all threads to arrive
            while *waiting != 0 {
                waiting = self
                    .condvar
                    .wait(waiting)
                    .expect("condvar wait should not fail");
            }
        }
    }
}

/// Utility functions for thread backend
pub mod thread_utils {
    use super::*;

    /// Create a thread backend with optimal configuration for single-node multi-GPU
    pub fn create_optimized_thread_backend(_num_devices: usize) -> ThreadBackend {
        // For thread backend, no special initialization needed
        // Configuration would be used for things like buffer sizes, thread pool sizes, etc.

        ThreadBackend::new()
    }

    /// Benchmark thread backend performance
    pub fn benchmark_thread_backend(
        backend: &ThreadBackend,
        tensor_sizes: &[usize],
    ) -> Result<Vec<(usize, std::time::Duration)>> {
        use std::time::Instant;

        let mut results = Vec::new();

        for &size in tensor_sizes {
            let tensor = Tensor::<f32>::ones(&[size]);

            // Create a test group
            let group = CommunicationGroup {
                group_id: "benchmark".to_string(),
                rank: 0,
                world_size: 2,
                devices: vec![tenflowers_core::Device::Cpu; 2],
                backend: crate::distributed::CommunicationBackend::Thread,
            };

            let start = Instant::now();
            let _result = backend.all_reduce_f32(&tensor, &group, ReductionOp::Sum)?;
            let elapsed = start.elapsed();

            results.push((size, elapsed));
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::CommunicationBackend;
    use tenflowers_core::Device;

    #[test]
    fn test_thread_backend_creation() {
        let backend = ThreadBackend::new();
        assert_eq!(backend.name(), "thread");
        assert!(!backend.initialized);
    }

    #[test]
    fn test_thread_backend_initialization() {
        let mut backend = ThreadBackend::new();
        let config = BackendConfig::default();

        assert!(backend.initialize(&config).is_ok());
        assert!(backend.initialized);
    }

    #[test]
    fn test_thread_group_creation() {
        let mut backend = ThreadBackend::new();
        let config = BackendConfig::default();
        backend
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 2,
            devices: vec![Device::Cpu],
            backend: CommunicationBackend::Thread,
        };

        assert!(backend.create_group(&group).is_ok());
    }

    #[test]
    fn test_thread_all_reduce_returns_honest_error() {
        // `ThreadBackend`'s `shared_state` is created fresh per instance and is
        // never shared across the ranks that would need to contribute distinct
        // local tensors (see `simulate_all_reduce`'s doc comment), so no genuine
        // cross-rank reduction can be performed here. Every `ReductionOp` variant
        // must surface an honest error rather than silently returning a scaled or
        // unscaled clone of the caller's own local tensor as if it were a real
        // reduction across ranks. This covers Sum and Average too: multiplying or
        // echoing back only the local rank's tensor is just as fabricated as doing
        // so for Min/Max/Product once ranks hold genuinely different data.
        let mut backend = ThreadBackend::new();
        let config = BackendConfig::default();
        backend
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 2,
            devices: vec![Device::Cpu],
            backend: CommunicationBackend::Thread,
        };

        backend
            .create_group(&group)
            .expect("test: operation should succeed");

        let tensor = Tensor::<f32>::ones(&[2, 3]);

        for op in [
            ReductionOp::Sum,
            ReductionOp::Average,
            ReductionOp::Min,
            ReductionOp::Max,
            ReductionOp::Product,
        ] {
            let result = backend.all_reduce_f32(&tensor, &group, op);
            assert!(
                result.is_err(),
                "all-reduce must never silently fabricate a result for {op:?}"
            );
        }
    }

    #[test]
    fn test_thread_send_recv_do_not_fabricate() {
        // send_f32 previously returned Ok(()) without transmitting anything
        // (putting a fixed placeholder payload into a channel whose Receiver was
        // already dropped in create_group, so it either silently no-op'd or failed
        // for the wrong reason), and recv_f32 previously fabricated a zero tensor
        // as if it were real data from src_rank. Separate `ThreadBackend` instances
        // share no real transport, so neither can genuinely happen; both must
        // surface honest errors instead.
        let mut backend = ThreadBackend::new();
        let config = BackendConfig::default();
        backend
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 2,
            devices: vec![Device::Cpu],
            backend: CommunicationBackend::Thread,
        };

        backend
            .create_group(&group)
            .expect("test: operation should succeed");

        let tensor = Tensor::<f32>::ones(&[8]);

        assert!(
            backend.send_f32(&tensor, 1, &group).is_err(),
            "send must never silently report a fabricated delivery"
        );
        assert!(
            backend.recv_f32(&[8], 1, &group).is_err(),
            "recv must never silently return a fabricated zero tensor"
        );
    }

    #[test]
    fn test_thread_barrier() {
        let barrier = ThreadBarrier::new(3);
        let barrier = Arc::new(barrier);

        let handles: Vec<_> = (0..3)
            .map(|i| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    // Simulate some work
                    std::thread::sleep(std::time::Duration::from_millis(i * 10));
                    barrier.wait();
                    i
                })
            })
            .collect();

        let results: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().expect("thread join should succeed"))
            .collect();
        assert_eq!(results, vec![0, 1, 2]);
    }
}
