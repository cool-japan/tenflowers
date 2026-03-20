//! Gradient compression for communication-efficient federated learning.

use super::types::ModelParams;

/// Top-k gradient sparsification for communication-efficient federated learning.
///
/// Retains only the `top_k_fraction` largest-magnitude gradient entries and
/// sets all others to zero.  The caller receives both the compressed gradient
/// (sparse, mostly zeros) and the index lists needed to reconstruct the full
/// vector at the server.
#[derive(Debug, Clone)]
pub struct GradientCompressor {
    /// Fraction of entries to keep (0 < top_k_fraction ≤ 1).
    pub top_k_fraction: f32,
}

impl GradientCompressor {
    /// Create a new compressor.
    ///
    /// `top_k_fraction` must be in `(0, 1]`.
    pub fn new(top_k_fraction: f32) -> Self {
        GradientCompressor {
            top_k_fraction: top_k_fraction.clamp(f32::EPSILON, 1.0),
        }
    }

    /// Compress gradients using top-k sparsification.
    ///
    /// Returns `(sparse_grads, kept_indices)` where `kept_indices[l]` contains
    /// the indices that were retained for layer `l`.
    pub fn compress(&self, grads: &ModelParams) -> (ModelParams, Vec<Vec<usize>>) {
        let mut compressed: ModelParams = Vec::with_capacity(grads.len());
        let mut indices: Vec<Vec<usize>> = Vec::with_capacity(grads.len());

        for layer in grads {
            let n = layer.len();
            let k = ((n as f32 * self.top_k_fraction).ceil() as usize)
                .max(1)
                .min(n);

            // Sort indices by magnitude (descending)
            let mut idx_mag: Vec<(usize, f32)> = layer
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v.abs()))
                .collect();
            idx_mag.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Keep top-k indices
            let mut kept: Vec<usize> = idx_mag[..k].iter().map(|(i, _)| *i).collect();
            kept.sort_unstable(); // restore natural order

            // Build sparse layer: keep selected, zero the rest
            let mut sparse = vec![0.0_f32; n];
            for &idx in &kept {
                sparse[idx] = layer[idx];
            }

            compressed.push(sparse);
            indices.push(kept);
        }

        (compressed, indices)
    }

    /// Reconstruct full-sized gradient vectors from compressed representation.
    ///
    /// `compressed[l]` may already be full-sized (all zeros except kept entries)
    /// or may be a dense vector of just the kept values.  We handle both cases:
    /// if `compressed[l].len() == full_dim[l]` it is already expanded; otherwise
    /// the kept values are scatter-filled.
    pub fn decompress(
        &self,
        compressed: &ModelParams,
        indices: &[Vec<usize>],
        full_dim: &[usize],
    ) -> ModelParams {
        compressed
            .iter()
            .zip(indices.iter())
            .zip(full_dim.iter())
            .map(|((layer, kept), &dim)| {
                if layer.len() == dim {
                    // Already expanded — return as-is
                    layer.clone()
                } else {
                    // layer contains only kept values; scatter them
                    let mut out = vec![0.0_f32; dim];
                    for (&idx, &val) in kept.iter().zip(layer.iter()) {
                        if idx < dim {
                            out[idx] = val;
                        }
                    }
                    out
                }
            })
            .collect()
    }

    /// Compute the compression ratio: `kept_elements / total_elements`.
    ///
    /// A value of `0.1` means 90% of gradient entries are zeroed out.
    pub fn compression_ratio(&self, original: &ModelParams) -> f32 {
        let total: usize = original.iter().map(|l| l.len()).sum();
        if total == 0 {
            return 1.0;
        }
        let kept: usize = original
            .iter()
            .map(|l| {
                ((l.len() as f32 * self.top_k_fraction).ceil() as usize)
                    .max(1)
                    .min(l.len())
            })
            .sum();
        kept as f32 / total as f32
    }
}
