//! GPU Tensor Manipulation Operations
//!
//! This module provides GPU-accelerated tensor manipulation operations including
//! reshape, transpose, slice, pad, tile, repeat, roll, and permutation operations.

pub mod advanced_kernel_fusion {
    pub use super::tile_repeat_roll::advanced_kernel_fusion::*;
}
pub mod ultra_simd_manipulation {
    pub use super::tile_repeat_roll::ultra_simd_manipulation::*;
}

mod pad;
mod permute_concat_reshape;
mod tile_repeat_roll;
mod transpose_slice;

#[cfg(test)]
mod tests;

pub use pad::execute_pad;
pub use permute_concat_reshape::{
    execute_concatenate, execute_reshape, execute_tensor_permutation,
};
pub use tile_repeat_roll::{execute_repeat, execute_roll, execute_tile};
pub use transpose_slice::{execute_slice, execute_transpose};
