use crate::distributed::{
    BackendConfig, CommunicationBackendImpl, CommunicationGroup, ReductionOp,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tenflowers_core::{Result, Tensor, TensorError};

/// MPI-based communication backend for distributed computing
/// MPI (Message Passing Interface) is the standard for high-performance computing
/// providing efficient communication primitives for cluster computing
pub struct MpiBackend {
    name: String,
    initialized: bool,
    /// Configuration for MPI operations
    config: Option<MpiConfig>,
    /// Groups managed by this backend
    groups: Arc<Mutex<HashMap<String, MpiGroupContext>>>,
}

/// Configuration for MPI backend
#[derive(Clone)]
struct MpiConfig {
    /// Timeout for collective operations (in milliseconds)
    timeout_ms: u64,
    /// Buffer size for communication
    buffer_size: usize,
    /// MPI communicator type
    communicator: MpiCommunicator,
}

/// MPI communicator types
#[derive(Clone, Debug)]
enum MpiCommunicator {
    /// World communicator (all processes)
    World,
    /// Custom communicator for process groups
    Group(String),
    /// Self communicator (single process)
    SelfComm,
}

/// Context for a specific communication group in MPI
struct MpiGroupContext {
    /// Group configuration
    world_size: usize,
    /// Communicator handle (in real implementation would be MPI_Comm)
    comm_id: String,
    /// Device assignments for this group
    device_mapping: HashMap<usize, tenflowers_core::Device>,
}

impl Default for MpiBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MpiBackend {
    pub fn new() -> Self {
        Self {
            name: "mpi".to_string(),
            initialized: false,
            config: None,
            groups: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create default MPI configuration
    fn default_config() -> MpiConfig {
        MpiConfig {
            timeout_ms: 30000,        // 30 seconds timeout
            buffer_size: 1024 * 1024, // 1MB buffer
            communicator: MpiCommunicator::World,
        }
    }

    /// Perform MPI all-reduce operation
    ///
    /// A real all-reduce requires a linked MPI implementation (e.g. OpenMPI, MPICH)
    /// and calls `MPI_Allreduce` across every rank in `group`. This crate does not
    /// link any MPI runtime and performs no cross-process communication of any
    /// kind: this function only ever sees the calling rank's own local `tensor`,
    /// never the other `group.world_size - 1` ranks' actual values.
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
    fn mpi_all_reduce(
        &self,
        _tensor: &Tensor<f32>,
        group: &CommunicationGroup,
        op: ReductionOp,
    ) -> Result<Tensor<f32>> {
        Err(TensorError::not_implemented_simple(format!(
            "MPI all-reduce (op={op:?}, group={}, world_size={}) is not available: \
             no MPI runtime is linked (MPI_Allreduce cannot be called), so no \
             genuine cross-rank reduction can be performed. Returning the local \
             tensor (scaled or unscaled) would silently fabricate the result.",
            group.group_id, group.world_size
        )))
    }

    /// Perform MPI all-gather operation
    ///
    /// A real all-gather (`MPI_Allgather`) collects every rank's distinct local
    /// tensor into a `Vec` with one entry per rank. Without an MPI runtime linked,
    /// this function has no way to obtain any rank's data but its own. Cloning the
    /// local tensor `group.world_size` times would fabricate the contributions of
    /// all other ranks, so we surface an honest error instead, mirroring
    /// `NcclBackend::all_gather_f32`.
    fn mpi_all_gather(
        &self,
        _tensor: &Tensor<f32>,
        group: &CommunicationGroup,
    ) -> Result<Vec<Tensor<f32>>> {
        Err(TensorError::not_implemented_simple(format!(
            "MPI all-gather (group={}, world_size={}) is not available: no MPI \
             runtime is linked (MPI_Allgather cannot be called), so the per-rank \
             contributions cannot be collected.",
            group.group_id, group.world_size
        )))
    }

    /// Perform MPI broadcast
    ///
    /// A real broadcast (`MPI_Bcast`) delivers the root rank's data to every other
    /// rank. Without an MPI runtime linked, a non-root rank's `tensor` argument does
    /// not hold the root's data, so echoing it back would fabricate the broadcast
    /// result, mirroring `NcclBackend::broadcast_f32`. The root-rank range check
    /// below is still performed for real: it needs no cross-process data, so keeping
    /// it gives callers a genuine argument diagnostic instead of folding it into the
    /// generic "not available" error.
    fn mpi_broadcast(
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
            "MPI broadcast (group={}, root={root_rank}, world_size={}) is not \
             available: no MPI runtime is linked (MPI_Bcast cannot be called), so \
             data cannot be transmitted from the root rank to the others.",
            group.group_id, group.world_size
        )))
    }
}

impl CommunicationBackendImpl for MpiBackend {
    fn initialize(&mut self, config: &BackendConfig) -> Result<()> {
        // Initialize MPI
        let mpi_config = MpiConfig {
            timeout_ms: config.timeout.as_millis() as u64,
            buffer_size: 1024 * 1024,
            communicator: MpiCommunicator::World,
        };

        self.config = Some(mpi_config);
        self.initialized = true;

        // In real implementation, would initialize MPI here:
        // MPI_Init(argc, argv)

        Ok(())
    }

    fn create_group(&mut self, group: &CommunicationGroup) -> Result<()> {
        if !self.initialized {
            return Err(TensorError::other("Backend not initialized".to_string()));
        }

        let context = MpiGroupContext {
            world_size: group.world_size,
            comm_id: format!("mpi_comm_{}", group.group_id),
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

        // In real implementation, would create MPI communicator:
        // MPI_Comm_create_group(MPI_COMM_WORLD, group_handle, tag, &new_comm)

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

        self.mpi_all_reduce(tensor, group, op)
    }

    fn all_gather_f32(
        &self,
        tensor: &Tensor<f32>,
        group: &CommunicationGroup,
    ) -> Result<Vec<Tensor<f32>>> {
        if !self.initialized {
            return Err(TensorError::other("Backend not initialized".to_string()));
        }

        self.mpi_all_gather(tensor, group)
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

        self.mpi_broadcast(tensor, root_rank, group)
    }

    /// Send an f32 tensor to a specific rank (point-to-point).
    ///
    /// A real send (`MPI_Send`) transmits the tensor's bytes to `dest_rank` over a
    /// real MPI runtime. This crate does not link any MPI runtime, so nothing is
    /// actually transmitted anywhere. Returning `Ok(())` without transmitting
    /// anything would falsely report a delivered message to the destination rank,
    /// so we surface an honest error instead, mirroring `NcclBackend::send_f32`.
    /// The destination-rank range check below is still performed for real: it
    /// needs no cross-process data.
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
            "MPI point-to-point send (group={}, dest={dest_rank}) is not available: \
             no MPI runtime is linked (MPI_Send cannot be called), so no data can be \
             transmitted.",
            group.group_id
        )))
    }

    /// Receive an f32 tensor from a specific rank (point-to-point).
    ///
    /// A real receive (`MPI_Recv`) blocks until the sender's actual bytes arrive
    /// over a real MPI runtime. This crate does not link any MPI runtime, so no
    /// data can ever arrive here. Returning a zero tensor would fabricate a
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
            "MPI point-to-point recv (group={}, src={src_rank}, shape={shape:?}) is \
             not available: no MPI runtime is linked (MPI_Recv cannot be called), so \
             no data can be received.",
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

        // In real implementation, would finalize MPI:
        // MPI_Finalize()

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::CommunicationBackend;
    use tenflowers_core::Device;

    #[test]
    fn test_mpi_backend_creation() {
        let backend = MpiBackend::new();
        assert_eq!(backend.name(), "mpi");
        assert!(!backend.initialized);
    }

    #[test]
    fn test_mpi_backend_initialization() {
        let mut backend = MpiBackend::new();
        let config = BackendConfig::default();

        assert!(backend.initialize(&config).is_ok());
        assert!(backend.initialized);
    }

    #[test]
    fn test_mpi_group_creation() {
        let mut backend = MpiBackend::new();
        let config = BackendConfig::default();
        backend
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 4,
            devices: vec![Device::Cpu; 4],
            backend: CommunicationBackend::Mpi,
        };

        assert!(backend.create_group(&group).is_ok());
    }

    #[test]
    fn test_mpi_all_reduce_returns_honest_error() {
        // Without an MPI runtime linked, no genuine cross-rank reduction can be
        // performed, so every ReductionOp variant must surface an honest error
        // rather than silently returning the (scaled or unscaled) local tensor as
        // if it were a real reduction across ranks. This covers Sum and Average
        // too: multiplying or echoing back only the local rank's tensor is just as
        // fabricated as doing so for Min/Max/Product once ranks hold genuinely
        // different data.
        let mut backend = MpiBackend::new();
        let config = BackendConfig::default();
        backend
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 4,
            devices: vec![Device::Cpu; 4],
            backend: CommunicationBackend::Mpi,
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
    fn test_mpi_collectives_do_not_fabricate() {
        // all-gather would need to fabricate every other rank's contribution, and
        // broadcast (with a valid root) would need to fabricate the root's data on
        // non-root callers. Neither is possible without an MPI runtime linked, so
        // both must surface honest errors instead of cloning the local tensor.
        let mut backend = MpiBackend::new();
        let config = BackendConfig::default();
        backend
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 4,
            devices: vec![Device::Cpu; 4],
            backend: CommunicationBackend::Mpi,
        };

        backend
            .create_group(&group)
            .expect("test: operation should succeed");

        let tensor = Tensor::<f32>::ones(&[25, 25]);

        assert!(backend.all_gather_f32(&tensor, &group).is_err());
        assert!(backend.broadcast_f32(&tensor, 0, &group).is_err());
    }

    #[test]
    fn test_mpi_broadcast_still_validates_root_rank() {
        // The root-rank range check needs no cross-process data, so it remains a
        // genuine (non-fabricated) validation even though the collective itself now
        // always errors. An out-of-range root must still be rejected distinctly.
        let mut backend = MpiBackend::new();
        let config = BackendConfig::default();
        backend
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 4,
            devices: vec![Device::Cpu; 4],
            backend: CommunicationBackend::Mpi,
        };

        backend
            .create_group(&group)
            .expect("test: operation should succeed");

        let tensor = Tensor::<f32>::ones(&[4]);
        assert!(backend.broadcast_f32(&tensor, 10, &group).is_err());
    }

    #[test]
    fn test_mpi_send_recv_do_not_fabricate() {
        // send_f32 previously returned Ok(()) without transmitting anything
        // (falsely reporting delivery), and recv_f32 previously fabricated a zero
        // tensor as if it were real data from src_rank. Without an MPI runtime
        // linked, neither can genuinely happen, so both must surface honest errors.
        let mut backend = MpiBackend::new();
        let config = BackendConfig::default();
        backend
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 4,
            devices: vec![Device::Cpu; 4],
            backend: CommunicationBackend::Mpi,
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
    fn test_mpi_send_recv_still_validate_rank_bounds() {
        // The destination/source rank range checks need no cross-process data, so
        // they remain genuine (non-fabricated) validations even though a
        // genuinely in-range send/recv now always errors afterward.
        let mut backend = MpiBackend::new();
        let config = BackendConfig::default();
        backend
            .initialize(&config)
            .expect("test: operation should succeed");

        let group = CommunicationGroup {
            group_id: "test".to_string(),
            rank: 0,
            world_size: 4,
            devices: vec![Device::Cpu; 4],
            backend: CommunicationBackend::Mpi,
        };

        backend
            .create_group(&group)
            .expect("test: operation should succeed");

        let tensor = Tensor::<f32>::ones(&[4]);
        assert!(backend.send_f32(&tensor, 10, &group).is_err());
        assert!(backend.recv_f32(&[4], 10, &group).is_err());
    }
}
