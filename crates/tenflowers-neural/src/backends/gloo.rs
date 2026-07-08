use crate::distributed::{
    BackendConfig, CommunicationBackendImpl, CommunicationGroup, ReductionOp,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tenflowers_core::{Result, Tensor, TensorError};

/// Gloo-based communication backend for CPU/GPU communication
/// Gloo is Facebook's collective communication library that provides efficient
/// collective operations across different device types and network topologies
pub struct GlooBackend {
    name: String,
    initialized: bool,
    /// Configuration for Gloo operations
    config: Option<GlooConfig>,
    /// Groups managed by this backend
    groups: Arc<Mutex<HashMap<String, GlooGroupContext>>>,
}

/// Configuration for Gloo backend
#[derive(Clone)]
struct GlooConfig {
    /// Timeout for collective operations (in milliseconds)
    timeout_ms: u64,
    /// Number of threads for parallel operations
    num_threads: usize,
    /// Buffer size for communication
    buffer_size: usize,
    /// Transport protocol (TCP/InfiniBand/etc)
    transport: GlooTransport,
}

/// Transport protocols supported by Gloo
#[derive(Clone, Debug)]
enum GlooTransport {
    /// TCP transport for cross-machine communication
    Tcp,
    /// InfiniBand transport for high-performance networks
    InfiniBand,
    /// Shared memory for same-machine communication
    SharedMemory,
}

/// Context for a specific communication group in Gloo
struct GlooGroupContext {
    /// Group configuration
    world_size: usize,
    /// Context handle (in real implementation would be Gloo context)
    context_id: String,
    /// Device assignments for this group
    device_mapping: HashMap<usize, tenflowers_core::Device>,
}

impl Default for GlooBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GlooBackend {
    pub fn new() -> Self {
        Self {
            name: "gloo".to_string(),
            initialized: false,
            config: None,
            groups: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create default Gloo configuration
    fn default_config() -> GlooConfig {
        GlooConfig {
            timeout_ms: 30000, // 30 seconds timeout
            num_threads: num_cpus::get(),
            buffer_size: 1024 * 1024, // 1MB buffer
            transport: GlooTransport::Tcp,
        }
    }

    /// Perform Gloo all-reduce operation
    ///
    /// A real all-reduce requires the Gloo collective-communications library
    /// (`libgloo`) to be linked, plus a real transport (TCP, InfiniBand, or shared
    /// memory) connecting every rank in `group`. This crate does not link libgloo
    /// and performs no cross-process communication of any kind: this function only
    /// ever sees the calling rank's own local `tensor`, never the other
    /// `group.world_size - 1` ranks' actual values.
    ///
    /// Every `ReductionOp` variant is affected, not just the arithmetically "obvious"
    /// ones. `Sum` (scale the local tensor by `world_size`) and
    /// `Average`/`Min`/`Max` (echo the local tensor back unchanged) only coincide
    /// with the true cross-rank result in the degenerate case where every rank
    /// happens to hold bit-identical data -- which real distributed training does
    /// not guarantee (each rank typically holds a distinct data shard and therefore
    /// a distinct local gradient). `Product` does not even hold in that degenerate
    /// case (it would need to return `tensor^world_size`, not `tensor`). Returning
    /// `Ok` for any of these would silently fabricate a successful reduction across
    /// ranks, so we surface an honest error instead, exactly as
    /// `NcclBackend::nccl_all_reduce` does for the same reason.
    fn gloo_all_reduce(
        &self,
        _tensor: &Tensor<f32>,
        group: &CommunicationGroup,
        op: ReductionOp,
    ) -> Result<Tensor<f32>> {
        Err(TensorError::not_implemented_simple(format!(
            "Gloo all-reduce (op={op:?}, group={}, world_size={}) is not available: \
             the Gloo runtime (libgloo) is not linked and no real transport connects \
             the ranks in this group, so no genuine cross-rank reduction can be \
             performed. Returning the local tensor (scaled or unscaled) would \
             silently fabricate the result.",
            group.group_id, group.world_size
        )))
    }

    /// Perform Gloo all-gather operation
    ///
    /// A real all-gather collects every rank's distinct local tensor into a `Vec`
    /// with one entry per rank. Without libgloo linked and a real transport, this
    /// function has no way to obtain any rank's data but its own. Cloning the local
    /// tensor `group.world_size` times would fabricate the contributions of all
    /// other ranks, so we surface an honest error instead, mirroring
    /// `NcclBackend::all_gather_f32`.
    fn gloo_all_gather(
        &self,
        _tensor: &Tensor<f32>,
        group: &CommunicationGroup,
    ) -> Result<Vec<Tensor<f32>>> {
        Err(TensorError::not_implemented_simple(format!(
            "Gloo all-gather (group={}, world_size={}) is not available: the Gloo \
             runtime (libgloo) is not linked, so the per-rank contributions cannot \
             be collected.",
            group.group_id, group.world_size
        )))
    }

    /// Perform Gloo broadcast operation
    ///
    /// A real broadcast delivers the root rank's data to every other rank. Without
    /// libgloo linked and a real transport, a non-root rank's `tensor` argument does
    /// not hold the root's data, so echoing it back would fabricate the broadcast
    /// result, mirroring `NcclBackend::broadcast_f32`. The root-rank range check
    /// below is still performed for real: it needs no cross-process data, so keeping
    /// it gives callers a genuine argument diagnostic instead of folding it into the
    /// generic "not available" error.
    fn gloo_broadcast(
        &self,
        _tensor: &Tensor<f32>,
        root_rank: usize,
        group: &CommunicationGroup,
    ) -> Result<Tensor<f32>> {
        if root_rank >= group.world_size {
            return Err(TensorError::invalid_argument(format!(
                "Root rank {} exceeds world size {}",
                root_rank, group.world_size
            )));
        }

        Err(TensorError::not_implemented_simple(format!(
            "Gloo broadcast (group={}, root={root_rank}, world_size={}) is not \
             available: the Gloo runtime (libgloo) is not linked, so data cannot be \
             transmitted from the root rank to the others.",
            group.group_id, group.world_size
        )))
    }

    /// Get optimal algorithm for given tensor size and group configuration
    fn select_algorithm(&self, _tensor_size: usize, _group: &CommunicationGroup) -> GlooAlgorithm {
        // Real Gloo has sophisticated algorithm selection based on:
        // - Tensor size
        // - Network topology
        // - Device types
        // - Historical performance data

        // For simulation, always use ring algorithm
        GlooAlgorithm::Ring
    }

    /// Get recommended Gloo algorithm for given configuration
    pub fn recommend_algorithm(
        tensor_size: usize,
        world_size: usize,
        network_bandwidth: f64,
    ) -> GlooAlgorithm {
        // Algorithm selection heuristics based on Gloo paper and empirical results

        if world_size <= 2 {
            // For small groups, direct communication is optimal
            return GlooAlgorithm::RecursiveDoubling;
        }

        // For large tensors or low bandwidth, ring is most efficient
        // Check this before power-of-2 optimization
        let threshold = if network_bandwidth > 10.0 {
            // 10 GB/s
            1024 * 1024 // 1M elements
        } else {
            256 * 1024 // 256K elements
        };

        if tensor_size >= threshold {
            return GlooAlgorithm::Ring;
        }

        if world_size.is_power_of_two() && world_size <= 16 {
            // Butterfly is optimal for power-of-2 groups up to 16 nodes with small tensors
            return GlooAlgorithm::Butterfly;
        }

        // For small tensors with good connectivity
        GlooAlgorithm::Tree
    }
}

/// Gloo algorithms for collective operations
#[derive(Debug, Clone)]
pub enum GlooAlgorithm {
    /// Ring algorithm - good for large tensors
    Ring,
    /// Tree algorithm - good for small tensors
    Tree,
    /// Butterfly algorithm - good for power-of-2 group sizes
    Butterfly,
    /// Recursive doubling - optimal for small groups
    RecursiveDoubling,
}

impl CommunicationBackendImpl for GlooBackend {
    fn initialize(&mut self, config: &BackendConfig) -> Result<()> {
        // Initialize Gloo context
        let gloo_config = GlooConfig {
            timeout_ms: config.timeout.as_millis() as u64,
            num_threads: num_cpus::get(),
            buffer_size: 1024 * 1024,
            transport: if config.compression {
                GlooTransport::InfiniBand // Use IB for compressed communications
            } else {
                GlooTransport::Tcp
            },
        };

        self.config = Some(gloo_config);
        self.initialized = true;

        // In real implementation, would initialize Gloo context here:
        // gloo::initialize(rank, world_size, &transport_config)

        Ok(())
    }

    fn create_group(&mut self, group: &CommunicationGroup) -> Result<()> {
        if !self.initialized {
            return Err(TensorError::other("Backend not initialized".to_string()));
        }

        let context = GlooGroupContext {
            world_size: group.world_size,
            context_id: format!("gloo_context_{}", group.group_id),
            device_mapping: group
                .devices
                .iter()
                .enumerate()
                .map(|(i, device)| (i, device.clone()))
                .collect(),
        };

        let mut groups = self
            .groups
            .lock()
            .map_err(|_| TensorError::other("Failed to acquire groups lock".to_string()))?;

        groups.insert(group.group_id.clone(), context);

        // In real implementation, would create Gloo process group:
        // let process_group = gloo::ProcessGroup::new(group.rank, group.world_size, &context);

        Ok(())
    }

    fn all_reduce_f32(
        &self,
        tensor: &Tensor<f32>,
        group: &CommunicationGroup,
        op: ReductionOp,
    ) -> Result<Tensor<f32>> {
        if !self.initialized {
            return Err(TensorError::other("Backend not initialized".to_string()));
        }

        // Select optimal algorithm based on tensor size and group configuration
        let _algorithm = self.select_algorithm(tensor.size(), group);

        // Perform Gloo all-reduce
        self.gloo_all_reduce(tensor, group, op)
    }

    fn all_gather_f32(
        &self,
        tensor: &Tensor<f32>,
        group: &CommunicationGroup,
    ) -> Result<Vec<Tensor<f32>>> {
        if !self.initialized {
            return Err(TensorError::other("Backend not initialized".to_string()));
        }

        self.gloo_all_gather(tensor, group)
    }

    fn broadcast_f32(
        &self,
        tensor: &Tensor<f32>,
        root_rank: usize,
        group: &CommunicationGroup,
    ) -> Result<Tensor<f32>> {
        if !self.initialized {
            return Err(TensorError::other("Backend not initialized".to_string()));
        }

        self.gloo_broadcast(tensor, root_rank, group)
    }

    /// Send an f32 tensor to a specific rank (point-to-point).
    ///
    /// A real send (Gloo's point-to-point primitives) transmits the tensor's bytes
    /// to `dest_rank` over a real transport. This crate does not link libgloo, so
    /// nothing is actually transmitted anywhere. Returning `Ok(())` without
    /// transmitting anything would falsely report a delivered message to the
    /// destination rank, so we surface an honest error instead, mirroring
    /// `NcclBackend::send_f32`. The destination-rank range check below is still
    /// performed for real: it needs no cross-process data.
    fn send_f32(
        &self,
        _tensor: &Tensor<f32>,
        dest_rank: usize,
        group: &CommunicationGroup,
    ) -> Result<()> {
        if !self.initialized {
            return Err(TensorError::other("Backend not initialized".to_string()));
        }

        if dest_rank >= group.world_size {
            return Err(TensorError::invalid_argument(format!(
                "Destination rank {} exceeds world size {}",
                dest_rank, group.world_size
            )));
        }

        Err(TensorError::not_implemented_simple(format!(
            "Gloo point-to-point send (group={}, dest={dest_rank}) is not available: \
             the Gloo runtime (libgloo) is not linked, so no data can be \
             transmitted.",
            group.group_id
        )))
    }

    /// Receive an f32 tensor from a specific rank (point-to-point).
    ///
    /// A real receive (Gloo's point-to-point primitives) blocks until the sender's
    /// actual bytes arrive over a real transport. This crate does not link libgloo,
    /// so no data can ever arrive here. Returning a zero tensor would fabricate a
    /// received payload, so we surface an honest error instead, mirroring
    /// `NcclBackend::recv_f32`. The source-rank range check below is still
    /// performed for real: it needs no cross-process data.
    fn recv_f32(
        &self,
        shape: &[usize],
        src_rank: usize,
        group: &CommunicationGroup,
    ) -> Result<Tensor<f32>> {
        if !self.initialized {
            return Err(TensorError::other("Backend not initialized".to_string()));
        }

        if src_rank >= group.world_size {
            return Err(TensorError::invalid_argument(format!(
                "Source rank {} exceeds world size {}",
                src_rank, group.world_size
            )));
        }

        Err(TensorError::not_implemented_simple(format!(
            "Gloo point-to-point recv (group={}, src={src_rank}, shape={shape:?}) is \
             not available: the Gloo runtime (libgloo) is not linked, so no data can \
             be received.",
            group.group_id
        )))
    }

    fn finalize(&mut self) -> Result<()> {
        if !self.initialized {
            return Ok(()); // Already finalized
        }

        // Clear all groups
        let mut groups = self
            .groups
            .lock()
            .map_err(|_| TensorError::other("Failed to acquire groups lock".to_string()))?;
        groups.clear();

        // Reset state
        self.config = None;
        self.initialized = false;

        // In real implementation, would finalize Gloo:
        // gloo::finalize()

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Utility functions for Gloo backend optimization
pub mod gloo_utils {
    use super::*;

    /// Create Gloo backend with optimal configuration for given hardware
    pub fn create_optimized_gloo_backend(num_devices: usize, has_infiniband: bool) -> GlooBackend {
        let mut backend = GlooBackend::new();

        let transport = if has_infiniband {
            GlooTransport::InfiniBand
        } else if num_devices > 1 {
            GlooTransport::Tcp
        } else {
            GlooTransport::SharedMemory
        };

        let config = GlooConfig {
            timeout_ms: 60000, // Longer timeout for large-scale training
            num_threads: num_cpus::get().min(num_devices * 2),
            buffer_size: if has_infiniband {
                4 * 1024 * 1024
            } else {
                1024 * 1024
            },
            transport,
        };

        backend.config = Some(config);
        backend
    }

    /// Benchmark Gloo backend performance across different tensor sizes
    pub fn benchmark_gloo_performance(
        backend: &GlooBackend,
        tensor_sizes: &[usize],
    ) -> Result<Vec<(usize, std::time::Duration)>> {
        use std::time::Instant;

        let mut results = Vec::new();

        for &size in tensor_sizes {
            let tensor = Tensor::<f32>::ones(&[size]);

            let group = CommunicationGroup {
                group_id: "benchmark".to_string(),
                rank: 0,
                world_size: 4, // Simulate 4-GPU setup
                devices: vec![tenflowers_core::Device::Cpu; 4],
                backend: crate::distributed::CommunicationBackend::Gloo,
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
    fn test_gloo_backend_creation() {
        let backend = GlooBackend::new();
        assert_eq!(backend.name(), "gloo");
        assert!(!backend.initialized);
    }

    #[test]
    fn test_gloo_backend_initialization() {
        let mut backend = GlooBackend::new();
        let config = BackendConfig::default();

        assert!(backend.initialize(&config).is_ok());
        assert!(backend.initialized);
    }

    #[test]
    fn test_gloo_group_creation() {
        let mut backend = GlooBackend::new();
        let config = BackendConfig::default();
        backend
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 4,
            devices: vec![Device::Cpu; 4],
            backend: CommunicationBackend::Gloo,
        };

        assert!(backend.create_group(&group).is_ok());
    }

    #[test]
    fn test_gloo_all_reduce_returns_honest_error() {
        // Without libgloo linked, no genuine cross-rank reduction can be performed,
        // so every ReductionOp variant must surface an honest error rather than
        // silently returning the (scaled or unscaled) local tensor as if it were a
        // real reduction across ranks. This covers Sum and Average too: multiplying
        // or echoing back only the local rank's tensor is just as fabricated as
        // doing so for Min/Max/Product once ranks hold genuinely different data.
        let mut backend = GlooBackend::new();
        let config = BackendConfig::default();
        backend
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 4,
            devices: vec![Device::Cpu; 4],
            backend: CommunicationBackend::Gloo,
        };

        backend
            .create_group(&group)
            .expect("test: operation should succeed");

        let tensor = Tensor::<f32>::ones(&[100, 50]);

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
    fn test_gloo_collectives_do_not_fabricate() {
        // all-gather would need to fabricate every other rank's contribution, and
        // broadcast (with a valid root) would need to fabricate the root's data on
        // non-root callers. Neither is possible without libgloo linked, so both must
        // surface honest errors instead of cloning the local tensor.
        let mut backend = GlooBackend::new();
        let config = BackendConfig::default();
        backend
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 4,
            devices: vec![Device::Cpu; 4],
            backend: CommunicationBackend::Gloo,
        };

        backend
            .create_group(&group)
            .expect("test: operation should succeed");

        let tensor = Tensor::<f32>::ones(&[25, 25]);

        assert!(backend.all_gather_f32(&tensor, &group).is_err());
        assert!(backend.broadcast_f32(&tensor, 0, &group).is_err());
    }

    #[test]
    fn test_gloo_broadcast_still_validates_root_rank() {
        // The root-rank range check needs no cross-process data, so it remains a
        // genuine (non-fabricated) validation even though the collective itself now
        // always errors. An out-of-range root must still be rejected distinctly.
        let mut backend = GlooBackend::new();
        let config = BackendConfig::default();
        backend
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 4,
            devices: vec![Device::Cpu; 4],
            backend: CommunicationBackend::Gloo,
        };

        backend
            .create_group(&group)
            .expect("test: operation should succeed");

        let tensor = Tensor::<f32>::ones(&[4]);
        assert!(backend.broadcast_f32(&tensor, 10, &group).is_err());
    }

    #[test]
    fn test_gloo_send_recv_do_not_fabricate() {
        // send_f32 previously returned Ok(()) without transmitting anything
        // (falsely reporting delivery), and recv_f32 previously fabricated a zero
        // tensor as if it were real data from src_rank. Without libgloo linked,
        // neither can genuinely happen, so both must surface honest errors.
        let mut backend = GlooBackend::new();
        let config = BackendConfig::default();
        backend
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 4,
            devices: vec![Device::Cpu; 4],
            backend: CommunicationBackend::Gloo,
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
    fn test_gloo_send_recv_still_validate_rank_bounds() {
        // The destination/source rank range checks need no cross-process data, so
        // they remain genuine (non-fabricated) validations even though a
        // genuinely in-range send/recv now always errors afterward.
        let mut backend = GlooBackend::new();
        let config = BackendConfig::default();
        backend
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 4,
            devices: vec![Device::Cpu; 4],
            backend: CommunicationBackend::Gloo,
        };

        backend
            .create_group(&group)
            .expect("test: operation should succeed");

        let tensor = Tensor::<f32>::ones(&[4]);
        assert!(backend.send_f32(&tensor, 10, &group).is_err());
        assert!(backend.recv_f32(&[4], 10, &group).is_err());
    }

    #[test]
    fn test_algorithm_selection() {
        // Small group should use recursive doubling
        assert!(matches!(
            GlooBackend::recommend_algorithm(1000, 2, 1.0),
            GlooAlgorithm::RecursiveDoubling
        ));

        // Power-of-2 group should use butterfly
        assert!(matches!(
            GlooBackend::recommend_algorithm(1000, 8, 1.0),
            GlooAlgorithm::Butterfly
        ));

        // Large tensor should use ring
        assert!(matches!(
            GlooBackend::recommend_algorithm(2_000_000, 16, 1.0),
            GlooAlgorithm::Ring
        ));

        // Small tensor with good bandwidth should use tree (non-power-of-2 world size)
        assert!(matches!(
            GlooBackend::recommend_algorithm(1000, 12, 15.0),
            GlooAlgorithm::Tree
        ));
    }

    #[test]
    fn test_error_handling() {
        let backend = GlooBackend::new(); // Not initialized

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 4,
            devices: vec![Device::Cpu; 4],
            backend: CommunicationBackend::Gloo,
        };

        let tensor = Tensor::<f32>::ones(&[10]);

        // Should fail when not initialized
        assert!(backend
            .all_reduce_f32(&tensor, &group, ReductionOp::Sum)
            .is_err());
        assert!(backend.all_gather_f32(&tensor, &group).is_err());
        assert!(backend.broadcast_f32(&tensor, 0, &group).is_err());
    }
}
