//! Parameter server for distributed training.
//!
//! This module provides a parameter server implementation for distributed training
//! with asynchronous gradient updates, fault tolerance, and load balancing.

mod client;
mod config;
mod server;
mod types;

// Re-export public types and functions
pub use client::ParameterServerClient;
pub use config::{FaultToleranceMode, LoadBalancingStrategy, ParameterServerConfig};
pub use server::ParameterServer;
pub use types::{GradientUpdate, ParameterEntry, ParameterServerStats, WorkerStatus};
