//! Distributed training module
//!
//! This module provides collective communication operations and model wrappers for distributed
//! and data-parallel neural network training.

#![allow(unexpected_cfgs)]
#![allow(unreachable_patterns)] // GPU/ROCM patterns unreachable when features are disabled

pub mod collective_ops;
pub mod data_parallel;
pub mod pipeline_parallel;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backward compatibility
pub use types::{
    BackendConfig, CollectiveOp, CollectiveResult, CommunicationBackend, CommunicationBackendImpl,
    CommunicationGroup, CommunicationMetrics, CommunicationRuntime, CompressionAlgorithm,
    OperationMetrics, ReductionOp,
};

pub use data_parallel::{DDPConfig, DataParallel, DistributedDataParallel, SynchronizationMode};

pub use pipeline_parallel::auto_detect_available_devices;

/// High-level distributed model wrappers (backward-compat alias)
pub mod models {
    pub use super::data_parallel::{
        DDPConfig, DataParallel, DistributedDataParallel, SynchronizationMode,
    };

    pub mod utils {
        pub use super::super::data_parallel::utils::{
            create_data_parallel, create_distributed_data_parallel, init_process_group,
        };
    }
}

/// Utility functions for distributed communication
pub mod utils {
    pub use super::pipeline_parallel::utils::{create_data_parallel_group, init_distributed};
}
