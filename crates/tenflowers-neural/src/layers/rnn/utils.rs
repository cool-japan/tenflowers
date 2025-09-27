//! Utility functions for RNN layers

use super::PackedSequence;
use num_traits::Float;
use tenflowers_core::{Result, Tensor, TensorError};

/// Pack sequences of different lengths into a PackedSequence
///
/// # Arguments
/// * `sequences` - Vector of sequences with different lengths
/// * `lengths` - Vector of actual sequence lengths
/// * `batch_first` - Whether sequences are in batch-first format
/// * `enforce_sorted` - Whether to enforce that sequences are sorted by length
///
/// # Returns
/// PackedSequence containing efficiently packed data
pub fn pack_padded_sequence<T>(
    sequences: &Tensor<T>,
    lengths: &[usize],
    batch_first: bool,
    enforce_sorted: bool,
) -> Result<PackedSequence<T>>
where
    T: Float
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + num_traits::Zero
        + num_traits::One
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let shape = sequences.shape().dims();
    let batch_size = if batch_first { shape[0] } else { shape[1] };
    let max_seq_len = if batch_first { shape[1] } else { shape[0] };
    let feature_size = if batch_first { shape[2] } else { shape[2] };

    if lengths.len() != batch_size {
        return Err(TensorError::invalid_argument(
            "lengths must have same length as batch size".to_string(),
        ));
    }

    // Create indices for sorting by length (descending)
    let mut indices: Vec<usize> = (0..batch_size).collect();
    if !enforce_sorted {
        indices.sort_by(|&a, &b| lengths[b].cmp(&lengths[a]));
    }

    // Create unsorted indices for later restoration
    let mut unsorted_indices = vec![0; batch_size];
    for (new_idx, &orig_idx) in indices.iter().enumerate() {
        unsorted_indices[orig_idx] = new_idx;
    }

    // Sort lengths
    let sorted_lengths: Vec<usize> = indices.iter().map(|&i| lengths[i]).collect();

    // Compute batch sizes for each time step
    let mut batch_sizes = Vec::new();
    for t in 0..max_seq_len {
        let count = sorted_lengths.iter().filter(|&&len| len > t).count();
        if count == 0 {
            break;
        }
        batch_sizes.push(count);
    }

    // Pack sequences
    let mut packed_data = Vec::new();
    let data = sequences.to_vec()?;

    for (t, &batch_size_t) in batch_sizes.iter().enumerate() {
        for idx in 0..batch_size_t {
            let orig_idx = indices[idx];

            // Calculate source index based on batch_first
            let src_idx = if batch_first {
                orig_idx * max_seq_len * feature_size + t * feature_size
            } else {
                t * batch_size * feature_size + orig_idx * feature_size
            };

            // Copy features for this time step
            for f in 0..feature_size {
                packed_data.push(data[src_idx + f].clone());
            }
        }
    }

    let packed_len: usize = packed_data.len() / feature_size;
    let packed_tensor = Tensor::from_vec(packed_data, &[packed_len, feature_size])?;

    Ok(PackedSequence::new(
        packed_tensor,
        batch_sizes,
        sorted_lengths,
        unsorted_indices,
    ))
}

/// Unpack a PackedSequence back to padded sequences
///
/// # Arguments
/// * `packed` - PackedSequence to unpack
/// * `batch_first` - Whether to return sequences in batch-first format
/// * `padding_value` - Value to use for padding shorter sequences
///
/// # Returns
/// Unpacked padded sequences and original lengths
pub fn pad_packed_sequence<T>(
    packed: &PackedSequence<T>,
    batch_first: bool,
    padding_value: T,
) -> Result<(Tensor<T>, Vec<usize>)>
where
    T: Float
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + num_traits::Zero
        + num_traits::One
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let max_seq_len = packed.batch_sizes.len();
    let batch_size = packed.sorted_lengths.len();
    let feature_size = packed.data.shape().dims()[1];

    // Create output tensor with padding
    let output_shape = if batch_first {
        [batch_size, max_seq_len, feature_size]
    } else {
        [max_seq_len, batch_size, feature_size]
    };

    let mut output_data = vec![padding_value; output_shape.iter().product()];
    let packed_data = packed.data.to_vec()?;

    // Unpack data
    let mut packed_idx = 0;
    for (t, &batch_size_t) in packed.batch_sizes.iter().enumerate() {
        for idx in 0..batch_size_t {
            let orig_idx = packed.unsorted_indices[idx];

            // Calculate destination index based on batch_first
            let dst_idx = if batch_first {
                orig_idx * max_seq_len * feature_size + t * feature_size
            } else {
                t * batch_size * feature_size + orig_idx * feature_size
            };

            // Copy features
            for f in 0..feature_size {
                output_data[dst_idx + f] = packed_data[packed_idx * feature_size + f].clone();
            }
            packed_idx += 1;
        }
    }

    let output_tensor = Tensor::from_vec(output_data, &output_shape)?;

    // Restore original lengths order
    let mut original_lengths = vec![0; batch_size];
    for (sorted_idx, &orig_idx) in packed.unsorted_indices.iter().enumerate() {
        original_lengths[orig_idx] = packed.sorted_lengths[sorted_idx];
    }

    Ok((output_tensor, original_lengths))
}
