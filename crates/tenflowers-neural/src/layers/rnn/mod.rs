//! RNN (Recurrent Neural Network) layers and utilities
//!
//! This module provides implementations of various RNN architectures including
//! LSTM, GRU, and vanilla RNN layers, along with attention mechanisms.

use num_traits::{Float, One, Zero};
use tenflowers_core::Tensor;

#[cfg(feature = "gpu")]
use tenflowers_core::{device::context::get_gpu_context, gpu::rnn_ops::GpuRnnOps};

// Re-export specialized modules
pub mod attention;
pub mod gru;
pub mod lstm;
pub mod utils;
pub mod vanilla_rnn;

// Re-export commonly used types
pub use attention::*;
pub use gru::GRU;
pub use lstm::LSTM;
pub use utils::{pack_padded_sequence, pad_packed_sequence};
pub use vanilla_rnn::RNN;

/// Reset gate variations for GRU layers
///
/// Different variants modify how the reset gate is computed and applied,
/// affecting the information flow and computational complexity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResetGateVariation {
    /// Standard GRU reset gate (default)
    /// r_t = sigmoid(W_ir * x_t + W_hr * h_{t-1})
    /// n_t = tanh(W_in * x_t + W_hn * (r_t * h_{t-1}))
    Standard,

    /// Minimal GRU: Combines reset and update gates into a single forget gate
    /// f_t = sigmoid(W_if * x_t + W_hf * h_{t-1})
    /// n_t = tanh(W_in * x_t + W_hn * h_{t-1})
    /// h_t = f_t * h_{t-1} + (1 - f_t) * n_t
    Minimal,

    /// Light GRU: Simplified reset gate computation
    /// r_t = sigmoid(W_ir * x_t) (no hidden state dependency)
    /// n_t = tanh(W_in * x_t + W_hn * (r_t * h_{t-1}))
    Light,

    /// Coupled GRU: Reset gate depends on update gate
    /// z_t = sigmoid(W_iz * x_t + W_hz * h_{t-1})
    /// r_t = sigmoid(W_ir * x_t + W_hr * h_{t-1} + W_zr * z_t)
    /// n_t = tanh(W_in * x_t + W_hn * (r_t * h_{t-1}))
    Coupled,

    /// Reset After: Apply reset gate after linear transformation
    /// r_t = sigmoid(W_ir * x_t + W_hr * h_{t-1})
    /// z_t = sigmoid(W_iz * x_t + W_hz * h_{t-1})
    /// n_t = tanh(W_in * x_t + r_t * (W_hn * h_{t-1}))
    ResetAfter,
}

/// Packed sequence structure for variable length sequences
#[derive(Debug)]
pub struct PackedSequence<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    /// Packed data tensor containing all sequences concatenated
    pub data: Tensor<T>,
    /// Batch sizes for each time step
    pub batch_sizes: Vec<usize>,
    /// Original sequence lengths for each sample in the batch
    pub sorted_lengths: Vec<usize>,
    /// Indices to restore original order after sorting
    pub unsorted_indices: Vec<usize>,
}

impl<T> PackedSequence<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    /// Create a new packed sequence
    pub fn new(
        data: Tensor<T>,
        batch_sizes: Vec<usize>,
        sorted_lengths: Vec<usize>,
        unsorted_indices: Vec<usize>,
    ) -> Self {
        Self {
            data,
            batch_sizes,
            sorted_lengths,
            unsorted_indices,
        }
    }
}

impl<T: Float + Clone + Default + Zero + One + Send + Sync + 'static> Clone for PackedSequence<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            batch_sizes: self.batch_sizes.clone(),
            sorted_lengths: self.sorted_lengths.clone(),
            unsorted_indices: self.unsorted_indices.clone(),
        }
    }
}
