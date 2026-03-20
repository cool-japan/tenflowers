//! MoCo (Momentum Contrast) queue and helpers.
//!
//! Implements the key components of MoCo (He et al., 2020):
//!
//! * [`MoCoQueue`] – a circular queue of key embeddings used as negatives.
//! * [`momentum_update`] – exponential moving average update of the target
//!   (momentum) encoder parameters.
//! * [`moco_loss`] – InfoNCE between the query and the (positive key + queue).

use crate::contrastive::losses::{info_nce_loss, l2_normalize_batch};
use tenflowers_core::{error::TensorError, Result};

// ─── MoCoQueue ──────────────────────────────────────────────────────────────

/// Circular queue of key embeddings used as negatives in MoCo.
///
/// The queue maintains up to `capacity` vectors of length `dim`.  New keys
/// are written at `ptr` (modulo `capacity`), overwriting the oldest entry
/// when full.
#[derive(Debug, Clone)]
pub struct MoCoQueue {
    /// Stored key embeddings — `queue[i]` is the `i`-th slot, length `dim`.
    queue: Vec<Vec<f32>>,
    /// Write pointer (index of the *next* slot to be overwritten).
    ptr: usize,
    /// Maximum number of keys that can be stored.
    pub capacity: usize,
    /// Embedding dimension.
    pub dim: usize,
}

impl MoCoQueue {
    /// Create an empty queue with the given capacity and embedding dimension.
    pub fn new(capacity: usize, dim: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "MoCoQueue::new".to_string(),
                reason: "capacity must be > 0".to_string(),
                context: None,
            });
        }
        if dim == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "MoCoQueue::new".to_string(),
                reason: "dim must be > 0".to_string(),
                context: None,
            });
        }
        Ok(Self {
            queue: Vec::new(),
            ptr: 0,
            capacity,
            dim,
        })
    }

    /// Enqueue a batch of `batch_size` keys from the flat `[batch_size, dim]`
    /// buffer `keys`.
    ///
    /// Keys are L2-normalised before storage.  When the queue is full, the
    /// oldest entry (at `ptr`) is overwritten in a circular fashion.
    pub fn enqueue_batch(&mut self, keys: &[f32], batch_size: usize) -> Result<()> {
        if keys.len() != batch_size * self.dim {
            return Err(TensorError::InvalidShape {
                operation: "MoCoQueue::enqueue_batch".to_string(),
                reason: format!(
                    "expected batch_size*dim={}*{}={} values, got {}",
                    batch_size,
                    self.dim,
                    batch_size * self.dim,
                    keys.len()
                ),
                shape: None,
                context: None,
            });
        }

        // Normalise a copy.
        let mut norm_keys = keys.to_vec();
        l2_normalize_batch(&mut norm_keys, batch_size, self.dim)?;

        for i in 0..batch_size {
            let row = norm_keys[i * self.dim..(i + 1) * self.dim].to_vec();
            if self.queue.len() < self.capacity {
                // Still filling up.
                self.queue.push(row);
            } else {
                // Overwrite oldest slot.
                self.queue[self.ptr] = row;
            }
            self.ptr = (self.ptr + 1) % self.capacity;
        }
        Ok(())
    }

    /// Return all queued keys as a flat `[len, dim]` buffer.
    pub fn all_keys(&self) -> Vec<f32> {
        self.queue.iter().flat_map(|v| v.iter().cloned()).collect()
    }

    /// Number of keys currently stored.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` when the queue has reached its full capacity.
    pub fn is_full(&self) -> bool {
        self.queue.len() == self.capacity
    }

    /// Returns `true` when no keys are stored yet.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

// ─── momentum_update ────────────────────────────────────────────────────────

/// Exponential moving average update for the momentum (target) encoder.
///
/// ```text
/// target = m * target + (1 - m) * online
/// ```
///
/// * `m = 1.0` → target is frozen.
/// * `m = 0.0` → target becomes a copy of online.
///
/// `online_params` and `target_params` must have the same length.
pub fn momentum_update(
    online_params: &[f32],
    target_params: &mut [f32],
    momentum: f32,
) -> Result<()> {
    if online_params.len() != target_params.len() {
        return Err(TensorError::InvalidShape {
            operation: "momentum_update".to_string(),
            reason: format!(
                "online_params length {} != target_params length {}",
                online_params.len(),
                target_params.len()
            ),
            shape: None,
            context: None,
        });
    }
    if !(0.0..=1.0).contains(&momentum) {
        return Err(TensorError::InvalidArgument {
            operation: "momentum_update".to_string(),
            reason: format!("momentum must be in [0, 1], got {}", momentum),
            context: None,
        });
    }
    let one_minus_m = 1.0 - momentum;
    for (t, o) in target_params.iter_mut().zip(online_params.iter()) {
        *t = momentum * *t + one_minus_m * o;
    }
    Ok(())
}

// ─── moco_loss ───────────────────────────────────────────────────────────────

/// MoCo InfoNCE loss.
///
/// Computes InfoNCE using the current positive key and all negatives stored in
/// the queue.  The queue keys serve as negatives; the `positive_key` is the
/// separate forward pass through the momentum encoder for the current sample.
///
/// Returns `0.0` when the queue is empty (no negatives available).
///
/// # Arguments
/// * `query`        – query embedding of length `dim`
/// * `positive_key` – positive key embedding of length `dim`
/// * `queue`        – the MoCoQueue holding negative keys
/// * `temperature`  – InfoNCE temperature τ
pub fn moco_loss(
    query: &[f32],
    positive_key: &[f32],
    queue: &MoCoQueue,
    temperature: f32,
) -> Result<f32> {
    if query.len() != queue.dim {
        return Err(TensorError::InvalidArgument {
            operation: "moco_loss".to_string(),
            reason: format!("query length {} != queue.dim {}", query.len(), queue.dim),
            context: None,
        });
    }
    if positive_key.len() != queue.dim {
        return Err(TensorError::InvalidArgument {
            operation: "moco_loss".to_string(),
            reason: format!(
                "positive_key length {} != queue.dim {}",
                positive_key.len(),
                queue.dim
            ),
            context: None,
        });
    }

    let negative_keys = queue.all_keys();
    let n_neg = queue.len();

    info_nce_loss(
        query,
        positive_key,
        &negative_keys,
        n_neg,
        queue.dim,
        temperature,
    )
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    // ── MoCoQueue ──────────────────────────────────────────────────────────

    #[test]
    fn test_moco_queue_new_invalid() {
        assert!(MoCoQueue::new(0, 4).is_err());
        assert!(MoCoQueue::new(4, 0).is_err());
    }

    #[test]
    fn test_moco_queue_enqueue_batch_fills_correctly() {
        let mut q = MoCoQueue::new(4, 2).expect("queue");
        let keys = vec![1.0_f32, 0.0, 0.0, 1.0]; // 2 keys of dim 2
        q.enqueue_batch(&keys, 2).expect("enqueue");
        assert_eq!(q.len(), 2);
        assert!(!q.is_full());

        // Add 2 more to reach capacity.
        let keys2 = vec![1.0_f32, 1.0, -1.0, 0.0];
        q.enqueue_batch(&keys2, 2).expect("enqueue2");
        assert_eq!(q.len(), 4);
        assert!(q.is_full());
    }

    #[test]
    fn test_moco_queue_circular_overwrite() {
        // Capacity = 2, add 3 keys → oldest is overwritten.
        let mut q = MoCoQueue::new(2, 2).expect("queue");
        // Key 0: [1, 0]
        q.enqueue_batch(&[1.0_f32, 0.0], 1).expect("enqueue");
        // Key 1: [0, 1]
        q.enqueue_batch(&[0.0_f32, 1.0], 1).expect("enqueue");
        assert!(q.is_full());

        // Key 2: [-1, 0] should overwrite key 0 at slot ptr=0.
        q.enqueue_batch(&[-1.0_f32, 0.0], 1).expect("enqueue");
        assert_eq!(q.len(), 2, "still at capacity");

        let all = q.all_keys();
        // Slot 0 was overwritten; slot 1 remains [0, 1] (normalised).
        // Slot 0 should now contain [-1, 0] (already unit norm).
        let slot0 = &all[0..2];
        assert!(
            (slot0[0] + 1.0).abs() < EPS && slot0[1].abs() < EPS,
            "slot 0 = {:?}",
            slot0
        );
    }

    #[test]
    fn test_moco_queue_all_keys_shape() {
        let mut q = MoCoQueue::new(10, 3).expect("queue");
        let keys = vec![0.0_f32; 5 * 3];
        q.enqueue_batch(&keys, 5).expect("enqueue");
        assert_eq!(q.all_keys().len(), 5 * 3);
    }

    #[test]
    fn test_moco_queue_enqueue_shape_mismatch() {
        let mut q = MoCoQueue::new(4, 3).expect("queue");
        let bad = vec![1.0_f32; 7]; // 7 != 2*3
        assert!(q.enqueue_batch(&bad, 2).is_err());
    }

    // ── momentum_update ────────────────────────────────────────────────────

    #[test]
    fn test_momentum_update_m1_target_unchanged() {
        let online = vec![1.0_f32, 2.0, 3.0];
        let mut target = vec![5.0_f32, 6.0, 7.0];
        let orig_target = target.clone();
        momentum_update(&online, &mut target, 1.0).expect("momentum");
        assert_eq!(target, orig_target, "m=1 → target should be unchanged");
    }

    #[test]
    fn test_momentum_update_m0_target_equals_online() {
        let online = vec![1.0_f32, 2.0, 3.0];
        let mut target = vec![5.0_f32, 6.0, 7.0];
        momentum_update(&online, &mut target, 0.0).expect("momentum");
        for (t, o) in target.iter().zip(online.iter()) {
            assert!((t - o).abs() < EPS, "m=0 → target should equal online");
        }
    }

    #[test]
    fn test_momentum_update_mid_value() {
        let online = vec![0.0_f32, 0.0];
        let mut target = vec![1.0_f32, 1.0];
        momentum_update(&online, &mut target, 0.9).expect("momentum");
        // target = 0.9 * 1 + 0.1 * 0 = 0.9
        assert!((target[0] - 0.9).abs() < EPS);
    }

    #[test]
    fn test_momentum_update_length_mismatch() {
        let online = vec![1.0_f32, 2.0];
        let mut target = vec![1.0_f32];
        assert!(momentum_update(&online, &mut target, 0.9).is_err());
    }

    #[test]
    fn test_momentum_update_invalid_momentum() {
        let online = vec![1.0_f32];
        let mut target = vec![1.0_f32];
        assert!(momentum_update(&online, &mut target, 1.1).is_err());
        assert!(momentum_update(&online, &mut target, -0.1).is_err());
    }

    // ── moco_loss ──────────────────────────────────────────────────────────

    #[test]
    fn test_moco_loss_is_finite() {
        let dim = 4;
        let mut q = MoCoQueue::new(8, dim).expect("queue");
        let neg_keys: Vec<f32> = (0..4)
            .flat_map(|i| {
                let mut v = vec![0.0_f32; dim];
                v[i % dim] = 1.0;
                v
            })
            .collect();
        q.enqueue_batch(&neg_keys, 4).expect("enqueue");

        let query = vec![1.0_f32, 0.0, 0.0, 0.0];
        let pos_key = vec![0.9_f32, 0.1, 0.0, 0.0];
        let loss = moco_loss(&query, &pos_key, &q, 0.07).expect("moco_loss");
        assert!(loss.is_finite(), "MoCo loss must be finite, got {}", loss);
        assert!(loss >= 0.0, "MoCo loss must be non-negative, got {}", loss);
    }

    #[test]
    fn test_moco_loss_empty_queue_is_zero() {
        let q = MoCoQueue::new(4, 2).expect("queue");
        let query = vec![1.0_f32, 0.0];
        let pos_key = vec![1.0_f32, 0.0];
        let loss = moco_loss(&query, &pos_key, &q, 0.07).expect("moco_loss empty");
        assert!(loss.abs() < 1e-4, "empty queue → loss ≈ 0, got {}", loss);
    }
}
