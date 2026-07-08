#[cfg(feature = "nccl")]
use crate::distributed::{
    BackendConfig, CommunicationBackendImpl, CommunicationGroup, ReductionOp,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tenflowers_core::{Device, Result, Tensor, TensorError};

/// NCCL communication backend for NVIDIA GPU distributed training
#[cfg(feature = "nccl")]
pub struct NcclBackend {
    name: String,
    initialized: bool,
    /// NCCL communicators for each group
    communicators: HashMap<String, NcclCommunicator>,
    /// Device contexts
    device_contexts: HashMap<Device, NcclDeviceContext>,
}

#[cfg(feature = "nccl")]
struct NcclCommunicator {
    /// NCCL communicator handle (placeholder - would use actual NCCL C API)
    comm_id: usize,
    /// Number of ranks in this communicator
    nranks: usize,
    /// This rank's position
    rank: usize,
}

#[cfg(feature = "nccl")]
struct NcclDeviceContext {
    /// GPU device ID
    device_id: usize,
    /// CUDA stream handle (placeholder)
    stream: usize,
}

#[cfg(feature = "nccl")]
impl NcclBackend {
    pub fn new() -> Self {
        Self {
            name: "nccl".to_string(),
            initialized: false,
            communicators: HashMap::new(),
            device_contexts: HashMap::new(),
        }
    }

    /// Initialize NCCL backend with environment variables
    fn init_from_env(&mut self) -> Result<()> {
        // Read environment variables for distributed setup
        let rank = std::env::var("RANK")
            .map_err(|_| {
                TensorError::invalid_argument("RANK environment variable not set".to_string())
            })?
            .parse::<usize>()
            .map_err(|_| TensorError::invalid_argument("Invalid RANK value".to_string()))?;

        let world_size = std::env::var("WORLD_SIZE")
            .map_err(|_| {
                TensorError::invalid_argument("WORLD_SIZE environment variable not set".to_string())
            })?
            .parse::<usize>()
            .map_err(|_| TensorError::invalid_argument("Invalid WORLD_SIZE value".to_string()))?;

        let master_addr = std::env::var("MASTER_ADDR").map_err(|_| {
            TensorError::invalid_argument("MASTER_ADDR environment variable not set".to_string())
        })?;

        let master_port = std::env::var("MASTER_PORT")
            .map_err(|_| {
                TensorError::invalid_argument(
                    "MASTER_PORT environment variable not set".to_string(),
                )
            })?
            .parse::<u16>()
            .map_err(|_| TensorError::invalid_argument("Invalid MASTER_PORT value".to_string()))?;

        // Initialize NCCL
        // NOTE: This is a placeholder - actual implementation would use NCCL C API
        self.init_nccl(rank, world_size, &master_addr, master_port)?;

        Ok(())
    }

    /// Initialize NCCL library (placeholder implementation)
    fn init_nccl(
        &mut self,
        rank: usize,
        world_size: usize,
        master_addr: &str,
        master_port: u16,
    ) -> Result<()> {
        // Placeholder for actual NCCL initialization
        // In real implementation, this would:
        // 1. Call ncclGetUniqueId() on rank 0
        // 2. Broadcast unique ID to all ranks
        // 3. Call ncclCommInitRank() with the unique ID

        println!(
            "Initializing NCCL: rank={}, world_size={}, master={}:{}",
            rank, world_size, master_addr, master_port
        );

        // Initialize GPU devices
        self.init_gpu_devices()?;

        self.initialized = true;
        Ok(())
    }

    /// Query the number of CUDA-visible GPU devices.
    ///
    /// A real implementation would call `cudaGetDeviceCount`. Since the CUDA runtime
    /// is not linked here, we honestly report the count we can actually establish.
    /// We honour `CUDA_VISIBLE_DEVICES` when it is set (the standard mechanism for
    /// constraining device visibility) and otherwise return `None` to signal that the
    /// count is unknown rather than fabricating a value such as `4`.
    fn query_gpu_count() -> Option<usize> {
        let visible = std::env::var("CUDA_VISIBLE_DEVICES").ok()?;
        let trimmed = visible.trim();
        if trimmed.is_empty() {
            // Explicitly set to empty => no devices visible.
            return Some(0);
        }
        // Count the comma-separated device identifiers that are actually parseable
        // as device indices; ignore blank entries from trailing/double commas.
        let count = trimmed
            .split(',')
            .filter(|entry| {
                let entry = entry.trim();
                !entry.is_empty() && entry.parse::<usize>().is_ok()
            })
            .count();
        Some(count)
    }

    /// Initialize GPU device contexts
    fn init_gpu_devices(&mut self) -> Result<()> {
        // Determine the real device count instead of fabricating one. If we cannot
        // establish the count (no CUDA runtime and no CUDA_VISIBLE_DEVICES hint), we
        // surface an honest error rather than inventing a device topology.
        let num_gpus = Self::query_gpu_count().ok_or_else(|| {
            TensorError::not_implemented_simple(
                "NCCL device initialization requires a real CUDA device count, but the \
                 CUDA runtime is not linked and CUDA_VISIBLE_DEVICES is not set. The GPU \
                 count cannot be determined."
                    .to_string(),
            )
        })?;

        if num_gpus == 0 {
            return Err(TensorError::not_implemented_simple(
                "NCCL device initialization found zero visible CUDA devices \
                 (CUDA_VISIBLE_DEVICES is empty)."
                    .to_string(),
            ));
        }

        for gpu_id in 0..num_gpus {
            let device = Device::Gpu(gpu_id);
            let context = NcclDeviceContext {
                device_id: gpu_id,
                stream: gpu_id, // Stream handle placeholder; real init needs libnccl.
            };
            self.device_contexts.insert(device, context);
        }

        Ok(())
    }

    /// Create NCCL communicator for a group
    fn create_nccl_communicator(&mut self, group: &CommunicationGroup) -> Result<()> {
        // Placeholder for NCCL communicator creation
        // Real implementation would:
        // 1. Extract GPU device IDs from group.devices
        // 2. Create NCCL communicator with ncclCommInitRank
        // 3. Store communicator handle

        let comm = NcclCommunicator {
            comm_id: group.group_id.len(), // Placeholder ID
            nranks: group.world_size,
            rank: group.rank,
        };

        self.communicators.insert(group.group_id.clone(), comm);
        Ok(())
    }

    /// Convert reduction operation to NCCL reduction op
    fn to_nccl_reduce_op(&self, op: ReductionOp) -> Result<u32> {
        // Placeholder for NCCL reduction op mapping
        // Real implementation would use actual NCCL enum values
        match op {
            ReductionOp::Sum => Ok(0),     // ncclSum
            ReductionOp::Product => Ok(1), // ncclProd
            ReductionOp::Max => Ok(2),     // ncclMax
            ReductionOp::Min => Ok(3),     // ncclMin
            ReductionOp::Average => Ok(4), // ncclAvg (if available) or Sum + division
        }
    }

    /// Perform NCCL all-reduce operation
    ///
    /// A real all-reduce requires the NCCL C library (`libnccl`) to be linked and
    /// at least one CUDA-capable device. This crate does not link the NCCL runtime,
    /// so we cannot perform the actual collective. Returning a clone of the input
    /// would silently fabricate a successful reduction across ranks, so instead we
    /// surface an honest error. The reduction op is validated first so callers still
    /// get the usual argument diagnostics.
    fn nccl_all_reduce(
        &self,
        _tensor: &Tensor<f32>,
        group: &CommunicationGroup,
        op: ReductionOp,
    ) -> Result<Tensor<f32>> {
        // Validate that a communicator exists for the requested group so the caller
        // gets the same argument-level diagnostics as a real implementation.
        let _comm = self.communicators.get(&group.group_id).ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "No NCCL communicator for group {}",
                group.group_id
            ))
        })?;

        // Validate the reduction op maps to a known NCCL op (still surfaces bad ops).
        let _nccl_op = self.to_nccl_reduce_op(op)?;

        Err(TensorError::not_implemented_simple(format!(
            "NCCL all-reduce (op={op:?}, group={}) is not available: the NCCL runtime \
             (libnccl) is not linked and no CUDA collective can be performed. Returning \
             the unmodified input would silently fabricate a cross-rank reduction.",
            group.group_id
        )))
    }
}

#[cfg(feature = "nccl")]
impl CommunicationBackendImpl for NcclBackend {
    fn initialize(&mut self, config: &BackendConfig) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        // Initialize from environment variables or config
        if config.options.contains_key("master_addr") {
            // Initialize from explicit configuration
            let rank = config
                .options
                .get("rank")
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| {
                    TensorError::invalid_argument("Missing or invalid rank in config".to_string())
                })?;

            let world_size = config
                .options
                .get("world_size")
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| {
                    TensorError::invalid_argument(
                        "Missing or invalid world_size in config".to_string(),
                    )
                })?;

            let master_addr = config.options.get("master_addr").ok_or_else(|| {
                TensorError::invalid_argument("Missing master_addr in config".to_string())
            })?;

            let master_port = config
                .options
                .get("master_port")
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| {
                    TensorError::invalid_argument(
                        "Missing or invalid master_port in config".to_string(),
                    )
                })?;

            self.init_nccl(rank, world_size, master_addr, master_port)?;
        } else {
            // Initialize from environment variables
            self.init_from_env()?;
        }

        Ok(())
    }

    fn create_group(&mut self, group: &CommunicationGroup) -> Result<()> {
        self.create_nccl_communicator(group)
    }

    fn all_reduce_f32(
        &self,
        tensor: &Tensor<f32>,
        group: &CommunicationGroup,
        op: ReductionOp,
    ) -> Result<Tensor<f32>> {
        self.nccl_all_reduce(tensor, group, op)
    }

    fn all_gather_f32(
        &self,
        _tensor: &Tensor<f32>,
        group: &CommunicationGroup,
    ) -> Result<Vec<Tensor<f32>>> {
        // Cloning the local tensor for every rank would fabricate the contributions
        // of all other ranks; without the NCCL runtime we cannot gather them.
        Err(TensorError::not_implemented_simple(format!(
            "NCCL all-gather (group={}) is not available: the NCCL runtime is not \
             linked, so the per-rank contributions cannot be collected.",
            group.group_id
        )))
    }

    fn broadcast_f32(
        &self,
        _tensor: &Tensor<f32>,
        root_rank: usize,
        group: &CommunicationGroup,
    ) -> Result<Tensor<f32>> {
        // On non-root ranks the input does not hold the root's data, so echoing it
        // back would fabricate the broadcast result.
        Err(TensorError::not_implemented_simple(format!(
            "NCCL broadcast (group={}, root={root_rank}) is not available: the NCCL \
             runtime is not linked, so data cannot be received from the root rank.",
            group.group_id
        )))
    }

    fn send_f32(
        &self,
        _tensor: &Tensor<f32>,
        dest_rank: usize,
        group: &CommunicationGroup,
    ) -> Result<()> {
        // Returning Ok without transmitting anything would falsely report a delivered
        // message to the destination rank.
        Err(TensorError::not_implemented_simple(format!(
            "NCCL point-to-point send (group={}, dest={dest_rank}) is not available: \
             the NCCL runtime is not linked, so no data can be transmitted.",
            group.group_id
        )))
    }

    fn recv_f32(
        &self,
        _shape: &[usize],
        src_rank: usize,
        group: &CommunicationGroup,
    ) -> Result<Tensor<f32>> {
        // Returning a zero tensor would fabricate a received payload.
        Err(TensorError::not_implemented_simple(format!(
            "NCCL point-to-point recv (group={}, src={src_rank}) is not available: the \
             NCCL runtime is not linked, so no data can be received.",
            group.group_id
        )))
    }

    fn finalize(&mut self) -> Result<()> {
        if !self.initialized {
            return Ok(());
        }

        // Cleanup NCCL resources
        // Real implementation would call ncclCommDestroy for all communicators
        for (group_id, _) in &self.communicators {
            println!("Destroying NCCL communicator for group: {}", group_id);
        }

        self.communicators.clear();
        self.device_contexts.clear();
        self.initialized = false;

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Utility functions for NCCL backend
#[cfg(feature = "nccl")]
pub mod nccl_utils {
    use super::*;

    /// Check whether a usable NCCL runtime is available.
    ///
    /// The `nccl` cargo feature only enables this Rust-side backend scaffolding; it
    /// does NOT link the actual NCCL C library (`libnccl`) or the CUDA runtime.
    /// Because no real collective can be performed, this reports `false` rather than
    /// claiming availability. It will only be able to report `true` once a real NCCL
    /// runtime binding is integrated.
    pub fn is_nccl_available() -> bool {
        false
    }

    /// Get recommended NCCL settings for given configuration
    pub fn get_recommended_config(world_size: usize, devices_per_node: usize) -> BackendConfig {
        let mut config = BackendConfig::default();

        // Add NCCL-specific optimizations
        config
            .options
            .insert("nccl_tree_threshold".to_string(), "0".to_string());
        config
            .options
            .insert("nccl_algo".to_string(), "Tree".to_string());

        // Adjust settings based on scale
        if world_size > 8 {
            config
                .options
                .insert("nccl_buffsize".to_string(), "33554432".to_string()); // 32MB
        } else {
            config
                .options
                .insert("nccl_buffsize".to_string(), "16777216".to_string()); // 16MB
        }

        config
    }

    /// Initialize NCCL for multi-node training
    pub fn init_multinode_nccl(
        rank: usize,
        local_rank: usize,
        world_size: usize,
        master_addr: &str,
        master_port: u16,
    ) -> Result<NcclBackend> {
        let mut backend = NcclBackend::new();

        let mut config = BackendConfig::default();
        config.options.insert("rank".to_string(), rank.to_string());
        config
            .options
            .insert("local_rank".to_string(), local_rank.to_string());
        config
            .options
            .insert("world_size".to_string(), world_size.to_string());
        config
            .options
            .insert("master_addr".to_string(), master_addr.to_string());
        config
            .options
            .insert("master_port".to_string(), master_port.to_string());

        backend.initialize(&config)?;
        Ok(backend)
    }
}

#[cfg(test)]
#[cfg(feature = "nccl")]
mod tests {
    use super::*;
    use crate::distributed::CommunicationBackend;

    #[test]
    fn test_nccl_backend_creation() {
        let backend = NcclBackend::new();
        assert_eq!(backend.name(), "nccl");
        assert!(!backend.initialized);
    }

    #[test]
    fn test_nccl_availability_honest() {
        // The NCCL C runtime is not linked, so availability must be reported as false
        // rather than fabricating a positive answer just because the feature compiled.
        assert!(!nccl_utils::is_nccl_available());
    }

    #[test]
    fn test_nccl_config_generation() {
        let config = nccl_utils::get_recommended_config(16, 8);
        assert!(config.options.contains_key("nccl_buffsize"));
        assert!(config.options.contains_key("nccl_algo"));
    }

    #[test]
    fn test_all_reduce_returns_honest_error() {
        // Without a registered communicator, an argument error is expected.
        let backend = NcclBackend::new();
        let group = CommunicationGroup {
            group_id: "missing_group".to_string(),
            rank: 0,
            world_size: 2,
            devices: vec![Device::Gpu(0), Device::Gpu(1)],
            backend: CommunicationBackend::Nccl,
        };
        let tensor = Tensor::<f32>::ones(&[4]);
        let result = backend.nccl_all_reduce(&tensor, &group, ReductionOp::Sum);
        assert!(
            result.is_err(),
            "all-reduce must never silently return a cloned tensor"
        );
    }

    #[test]
    fn test_collectives_do_not_fabricate() {
        let backend = NcclBackend::new();
        let group = CommunicationGroup {
            group_id: "g".to_string(),
            rank: 0,
            world_size: 2,
            devices: vec![Device::Gpu(0), Device::Gpu(1)],
            backend: CommunicationBackend::Nccl,
        };
        let tensor = Tensor::<f32>::ones(&[2]);

        // Every collective must surface an honest error instead of fabricating data.
        assert!(backend.all_gather_f32(&tensor, &group).is_err());
        assert!(backend.broadcast_f32(&tensor, 0, &group).is_err());
        assert!(backend.send_f32(&tensor, 1, &group).is_err());
        assert!(backend.recv_f32(&[2], 1, &group).is_err());
    }

    #[test]
    fn test_query_gpu_count_honours_visible_devices() {
        // This test mutates a process-global env var; it is isolated to a single
        // assertion path and restores the previous value to avoid cross-test leakage.
        let previous = std::env::var("CUDA_VISIBLE_DEVICES").ok();

        std::env::set_var("CUDA_VISIBLE_DEVICES", "0,1,2");
        assert_eq!(NcclBackend::query_gpu_count(), Some(3));

        std::env::set_var("CUDA_VISIBLE_DEVICES", "");
        assert_eq!(NcclBackend::query_gpu_count(), Some(0));

        std::env::remove_var("CUDA_VISIBLE_DEVICES");
        assert_eq!(
            NcclBackend::query_gpu_count(),
            None,
            "unknown count must be None, never a fabricated constant"
        );

        match previous {
            Some(value) => std::env::set_var("CUDA_VISIBLE_DEVICES", value),
            None => std::env::remove_var("CUDA_VISIBLE_DEVICES"),
        }
    }
}
