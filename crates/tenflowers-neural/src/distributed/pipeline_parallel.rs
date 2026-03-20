//! PipelineParallel and stage scheduling for pipeline-based distributed training

use std::collections::HashMap;
use tenflowers_core::{Device, Result, Tensor, TensorError};

use super::types::CommunicationBackend;

/// Auto-detect available devices for distributed training
pub fn auto_detect_available_devices() -> Vec<Device> {
    let mut devices = vec![Device::Cpu];

    #[cfg(feature = "gpu")]
    {
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
            if let Ok(cuda_devices) = std::env::var("CUDA_VISIBLE_DEVICES") {
                for (i, device_id) in cuda_devices.split(',').enumerate() {
                    if let Ok(_id) = device_id.trim().parse::<u32>() {
                        devices.push(Device::Gpu(i));
                    }
                }
            }
        } else {
            devices.push(Device::Gpu(0));
        }
    }

    devices
}

/// Utility functions for distributed communication
pub mod utils {
    use super::*;
    #[cfg(feature = "gloo")]
    use crate::backends::gloo::GlooBackend;
    #[cfg(feature = "nccl")]
    use crate::backends::nccl::NcclBackend;
    use crate::backends::thread::ThreadBackend;

    use super::super::types::{BackendConfig, CommunicationGroup, CommunicationRuntime};

    /// Initialize distributed environment with automatic backend selection
    pub fn init_distributed(
        rank: usize,
        world_size: usize,
        backend: Option<CommunicationBackend>,
    ) -> Result<CommunicationRuntime> {
        let mut runtime = CommunicationRuntime::new();

        let backend = backend.unwrap_or({
            #[cfg(feature = "nccl")]
            {
                CommunicationBackend::Nccl
            }
            #[cfg(not(feature = "nccl"))]
            {
                CommunicationBackend::Thread
            }
        });

        match backend {
            CommunicationBackend::Thread => {
                runtime
                    .register_backend(CommunicationBackend::Thread, Box::new(ThreadBackend::new()));
            }
            #[cfg(feature = "nccl")]
            CommunicationBackend::Nccl => {
                runtime.register_backend(CommunicationBackend::Nccl, Box::new(NcclBackend::new()));
            }
            #[cfg(feature = "gloo")]
            CommunicationBackend::Gloo => {
                runtime.register_backend(CommunicationBackend::Gloo, Box::new(GlooBackend::new()));
            }
            #[cfg(not(feature = "gloo"))]
            CommunicationBackend::Gloo => {
                return Err(TensorError::unsupported_operation_simple(
                    "Gloo backend not compiled in. Enable 'gloo' feature".to_string(),
                ));
            }
            _ => {
                return Err(TensorError::unsupported_operation_simple(format!(
                    "Backend {backend:?} not supported"
                )));
            }
        }

        let config = BackendConfig::default();
        runtime.initialize(&config)?;

        let devices = auto_detect_available_devices();
        let group = CommunicationGroup {
            group_id: "default".to_string(),
            rank,
            world_size,
            devices,
            backend,
        };

        runtime.create_group(group)?;

        Ok(runtime)
    }

    /// Create a distributed data parallel group
    pub fn create_data_parallel_group(
        runtime: &mut CommunicationRuntime,
        devices: Vec<Device>,
        backend: CommunicationBackend,
    ) -> Result<String> {
        let group_id = "data_parallel".to_string();
        let world_size = devices.len();

        let actual_rank = if let Ok(rank_str) = std::env::var("RANK") {
            rank_str.parse::<usize>().unwrap_or(0)
        } else if let Ok(local_rank_str) = std::env::var("LOCAL_RANK") {
            local_rank_str.parse::<usize>().unwrap_or(0)
        } else {
            0
        };

        let group = CommunicationGroup {
            group_id: group_id.clone(),
            rank: actual_rank,
            world_size,
            devices,
            backend,
        };

        runtime.create_group(group)?;
        Ok(group_id)
    }
}
