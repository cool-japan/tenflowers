//! Sequence augmentation operation implementations.
//!
//! Functions in this module apply token-level and spectrogram-level
//! augmentations to 1-D/2-D/3-D tensors.

use scirs2_core::num_traits::{Float, FromPrimitive, Zero};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::RngExt;
use tenflowers_core::{Result, Tensor};

use super::box_muller;

/// Delete tokens (set to zero) with probability `prob` along the last axis.
pub(super) fn apply_token_deletion<T>(
    input: &Tensor<T>,
    prob: f64,
    rng: &mut StdRng,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let out: Vec<T> = data
        .iter()
        .map(|&v| {
            let coin: f64 = rng.random_range(0.0..1.0);
            if coin < prob {
                T::zero()
            } else {
                v
            }
        })
        .collect();
    Tensor::from_vec(out, input.shape().dims())
}

/// Insert additive noise at random positions.
pub(super) fn apply_token_noise_insertion<T>(
    input: &Tensor<T>,
    prob: f64,
    rng: &mut StdRng,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let out: Vec<T> = data
        .iter()
        .map(|&v| {
            let coin: f64 = rng.random_range(0.0..1.0);
            if coin < prob {
                let noise_val = rng.random_range(-0.1_f64..0.1_f64);
                let nv = v.to_f64().unwrap_or(0.0) + noise_val;
                T::from_f64(nv).unwrap_or(v)
            } else {
                v
            }
        })
        .collect();
    Tensor::from_vec(out, input.shape().dims())
}

/// Swap `n_swaps` pairs of adjacent tokens along the last axis.
pub(super) fn apply_token_swap<T>(
    input: &Tensor<T>,
    n_swaps: usize,
    rng: &mut StdRng,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let dims = input.shape().dims();
    let seq_len = *dims.last().unwrap_or(&0);
    if seq_len < 2 {
        return Ok(input.clone());
    }

    let mut out = data.to_vec();
    let n_sequences = data.len() / seq_len;

    for s in 0..n_sequences {
        let offset = s * seq_len;
        for _ in 0..n_swaps {
            let i = rng.random_range(0..seq_len - 1);
            out.swap(offset + i, offset + i + 1);
        }
    }

    Tensor::from_vec(out, dims)
}

/// Replace tokens with small Gaussian noise at random positions.
pub(super) fn apply_token_substitution<T>(
    input: &Tensor<T>,
    prob: f64,
    rng: &mut StdRng,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let out: Vec<T> = data
        .iter()
        .map(|&v| {
            let coin: f64 = rng.random_range(0.0..1.0);
            if coin < prob {
                let (z, _) = box_muller(rng);
                T::from_f64(z * 0.1).unwrap_or(v)
            } else {
                v
            }
        })
        .collect();
    Tensor::from_vec(out, input.shape().dims())
}

/// Reverse the sequence along the last axis.
pub(super) fn apply_sequence_reversal<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let dims = input.shape().dims();
    let seq_len = *dims.last().unwrap_or(&0);
    if seq_len < 2 {
        return Ok(input.clone());
    }

    let mut out = data.to_vec();
    let n_sequences = data.len() / seq_len;

    for s in 0..n_sequences {
        let offset = s * seq_len;
        for i in 0..seq_len / 2 {
            out.swap(offset + i, offset + seq_len - 1 - i);
        }
    }

    Tensor::from_vec(out, dims)
}

/// SpecAugment-style time warping: locally stretch/compress the sequence.
///
/// Applies piece-wise linear warping by choosing a random warp point
/// and shifting it by up to `warp_factor * seq_len` positions.
pub(super) fn apply_time_warping<T>(
    input: &Tensor<T>,
    warp_factor: f64,
    rng: &mut StdRng,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let dims = input.shape().dims();
    let seq_len = *dims.last().unwrap_or(&0);
    if seq_len < 4 {
        return Ok(input.clone());
    }

    let n_sequences = data.len() / seq_len;
    let max_warp = ((seq_len as f64) * warp_factor).max(1.0) as usize;
    let max_warp = max_warp.min(seq_len / 2);

    let warp_point = rng.random_range(max_warp..seq_len - max_warp);
    let warp_dist = rng.random_range(0..max_warp * 2 + 1) as i64 - max_warp as i64;
    let new_warp = (warp_point as i64 + warp_dist).clamp(1, (seq_len - 2) as i64) as usize;

    let mut out = vec![T::zero(); data.len()];

    for s in 0..n_sequences {
        let offset = s * seq_len;
        for t in 0..seq_len {
            let src_t = if t <= new_warp {
                if new_warp > 0 {
                    (t as f64 / new_warp as f64) * warp_point as f64
                } else {
                    0.0
                }
            } else {
                let remaining = seq_len - 1 - new_warp;
                if remaining > 0 {
                    warp_point as f64
                        + ((t - new_warp) as f64 / remaining as f64)
                            * (seq_len - 1 - warp_point) as f64
                } else {
                    (seq_len - 1) as f64
                }
            };

            let src_lo = src_t.floor() as usize;
            let src_hi = (src_lo + 1).min(seq_len - 1);
            let frac = src_t - src_t.floor();
            let v_lo = data[offset + src_lo].to_f64().unwrap_or(0.0);
            let v_hi = data[offset + src_hi].to_f64().unwrap_or(0.0);
            let v = v_lo * (1.0 - frac) + v_hi * frac;
            out[offset + t] = T::from_f64(v).unwrap_or_else(T::zero);
        }
    }

    Tensor::from_vec(out, dims)
}

/// SpecAugment-style time masking: zero-out a contiguous time band.
pub(super) fn apply_time_masking<T>(
    input: &Tensor<T>,
    max_mask: usize,
    rng: &mut StdRng,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let dims = input.shape().dims();

    let seq_len = *dims.last().unwrap_or(&0);
    if seq_len == 0 {
        return Ok(input.clone());
    }

    let mask_len = if max_mask > 0 {
        rng.random_range(0..max_mask.min(seq_len) + 1)
    } else {
        0
    };
    let mask_start = if seq_len > mask_len {
        rng.random_range(0..seq_len - mask_len + 1)
    } else {
        0
    };

    let mut out = data.to_vec();
    let n_sequences = data.len() / seq_len;

    for s in 0..n_sequences {
        let offset = s * seq_len;
        for t in mask_start..mask_start + mask_len {
            if t < seq_len {
                out[offset + t] = T::zero();
            }
        }
    }

    Tensor::from_vec(out, dims)
}

/// SpecAugment-style frequency masking: zero-out a contiguous frequency band.
///
/// Expects tensor of shape `[N, F, T]` or `[F, T]` where F is the frequency axis.
pub(super) fn apply_frequency_masking<T>(
    input: &Tensor<T>,
    max_mask: usize,
    rng: &mut StdRng,
) -> Result<Tensor<T>>
where
    T: Float + FromPrimitive + Clone + Default + Zero,
{
    let data = input.data();
    let dims = input.shape().dims();
    let ndim = dims.len();

    if ndim < 2 {
        return apply_time_masking(input, max_mask, rng);
    }

    let (n, freq_dim, time_dim) = if ndim == 2 {
        (1, dims[0], dims[1])
    } else if ndim == 3 {
        (dims[0], dims[1], dims[2])
    } else {
        let outer: usize = dims[..ndim - 2].iter().product();
        (outer, dims[ndim - 2], dims[ndim - 1])
    };

    let mask_len = if max_mask > 0 && freq_dim > 0 {
        rng.random_range(0..max_mask.min(freq_dim) + 1)
    } else {
        0
    };
    let mask_start = if freq_dim > mask_len {
        rng.random_range(0..freq_dim - mask_len + 1)
    } else {
        0
    };

    let mut out = data.to_vec();
    let ft = freq_dim * time_dim;

    for ni in 0..n {
        let base = ni * ft;
        for f in mask_start..mask_start + mask_len {
            if f < freq_dim {
                let row_offset = base + f * time_dim;
                for t in 0..time_dim {
                    out[row_offset + t] = T::zero();
                }
            }
        }
    }

    Tensor::from_vec(out, dims)
}
