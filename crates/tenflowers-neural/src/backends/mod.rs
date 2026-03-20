#![allow(unexpected_cfgs)]

pub mod thread;

#[cfg(feature = "nccl")]
pub mod nccl;

#[cfg(feature = "gloo")]
pub mod gloo;

#[cfg(feature = "mpi")]
pub mod mpi;
