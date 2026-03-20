//! Simple embedding-space augmentation utilities.
//!
//! Rather than augmenting raw pixel / token inputs, these helpers operate
//! directly on embedding vectors, which is useful for second-stage contrastive
//! learning and for unit testing loss functions without a full data pipeline.
//!
//! All functions that require randomness accept an explicit `seed` so that
//! results are deterministic and reproducible.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─── public API ─────────────────────────────────────────────────────────────

/// Add independent Gaussian noise to every element of `embeddings`.
///
/// Uses a Box-Muller transform for Gaussian sampling from two uniform draws,
/// seeded deterministically from `seed`.
///
/// # Arguments
/// * `embeddings` – input flat embedding buffer (any shape)
/// * `std`        – standard deviation of the noise
/// * `seed`       – RNG seed for reproducibility
///
/// # Returns
/// A new `Vec<f32>` with noise added element-wise.
pub fn add_gaussian_noise(embeddings: &[f32], std: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut result = embeddings.to_vec();

    let n = result.len();
    let mut i = 0;
    while i < n {
        // Box-Muller transform: two uniform samples → two Gaussian samples.
        let u1: f64 = rng.random::<f64>().max(1e-10);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        let z0 = (r * theta.cos()) as f32;
        let z1 = (r * theta.sin()) as f32;

        result[i] += std * z0;
        if i + 1 < n {
            result[i + 1] += std * z1;
        }
        i += 2;
    }
    result
}

/// Zero out each feature with probability `drop_prob` (feature dropout).
///
/// With `drop_prob = 0.0` the function returns an exact copy; with
/// `drop_prob = 1.0` every feature is zeroed.
///
/// # Arguments
/// * `embeddings` – flat `[n_samples, dim]` buffer (or any shape)
/// * `dim`        – embedding dimension (used for documentation only; the
///                  mask is applied element-wise regardless of `dim`)
/// * `drop_prob`  – probability in `[0, 1)` of zeroing each feature
/// * `seed`       – RNG seed
///
/// # Returns
/// A new `Vec<f32>` with features randomly zeroed.
pub fn feature_dropout(embeddings: &[f32], dim: usize, drop_prob: f32, seed: u64) -> Vec<f32> {
    // dim is carried in the signature for API clarity (matches create_augmented_pair).
    let _ = dim;
    let mut rng = StdRng::seed_from_u64(seed);
    embeddings
        .iter()
        .map(|&x| {
            let r: f32 = rng.random();
            if r < drop_prob {
                0.0
            } else {
                x
            }
        })
        .collect()
}

/// Create two independently augmented views of a batch of embeddings.
///
/// View A uses `seed` and view B uses `seed.wrapping_add(1)`.  Each view
/// applies Gaussian noise followed by feature dropout.
///
/// # Arguments
/// * `embeddings` – flat `[n, dim]` embedding buffer
/// * `n`          – number of samples
/// * `dim`        – embedding dimension
/// * `noise_std`  – standard deviation of Gaussian noise
/// * `drop_prob`  – feature dropout probability
/// * `seed`       – base seed (view A uses `seed`, view B uses `seed+1`)
///
/// # Returns
/// `(view_a, view_b)` — each a `Vec<f32>` of length `n * dim`.
pub fn create_augmented_pair(
    embeddings: &[f32],
    n: usize,
    dim: usize,
    noise_std: f32,
    drop_prob: f32,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let _ = (n, dim); // shape tracked by caller; not validated here to stay simple.
    let view_a = {
        let noisy = add_gaussian_noise(embeddings, noise_std, seed);
        feature_dropout(&noisy, dim, drop_prob, seed)
    };
    let view_b = {
        let noisy = add_gaussian_noise(embeddings, noise_std, seed.wrapping_add(1));
        feature_dropout(&noisy, dim, drop_prob, seed.wrapping_add(1))
    };
    (view_a, view_b)
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-7;

    // ── add_gaussian_noise ────────────────────────────────────────────────

    #[test]
    fn test_add_gaussian_noise_changes_embeddings() {
        let emb = vec![1.0_f32; 16];
        let noisy = add_gaussian_noise(&emb, 1.0, 42);
        assert_ne!(emb, noisy, "noise should change at least one element");
    }

    #[test]
    fn test_add_gaussian_noise_zero_std_identity() {
        let emb: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let noisy = add_gaussian_noise(&emb, 0.0, 7);
        // With std=0 every Box-Muller sample is 0 → embeddings unchanged.
        for (a, b) in emb.iter().zip(noisy.iter()) {
            assert!((a - b).abs() < EPS, "expected identity, got diff {}", a - b);
        }
    }

    #[test]
    fn test_add_gaussian_noise_deterministic() {
        let emb = vec![0.5_f32; 8];
        let a = add_gaussian_noise(&emb, 0.1, 99);
        let b = add_gaussian_noise(&emb, 0.1, 99);
        assert_eq!(a, b, "same seed should produce identical output");
    }

    #[test]
    fn test_add_gaussian_noise_different_seeds_differ() {
        let emb = vec![0.5_f32; 8];
        let a = add_gaussian_noise(&emb, 0.1, 1);
        let b = add_gaussian_noise(&emb, 0.1, 2);
        assert_ne!(
            a, b,
            "different seeds should (almost always) produce different noise"
        );
    }

    #[test]
    fn test_add_gaussian_noise_preserves_length() {
        let emb = vec![1.0_f32; 13]; // odd length
        let noisy = add_gaussian_noise(&emb, 0.5, 55);
        assert_eq!(noisy.len(), emb.len());
    }

    // ── feature_dropout ───────────────────────────────────────────────────

    #[test]
    fn test_feature_dropout_zero_prob_is_identity() {
        let emb: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let out = feature_dropout(&emb, 4, 0.0, 0);
        assert_eq!(emb, out, "drop_prob=0 must be identity");
    }

    #[test]
    fn test_feature_dropout_one_prob_zeroes_all() {
        let emb = vec![1.0_f32; 8];
        let out = feature_dropout(&emb, 4, 1.0, 0);
        assert!(
            out.iter().all(|&x| x == 0.0),
            "drop_prob=1 should zero everything"
        );
    }

    #[test]
    fn test_feature_dropout_deterministic() {
        let emb = vec![1.0_f32; 10];
        let a = feature_dropout(&emb, 5, 0.5, 17);
        let b = feature_dropout(&emb, 5, 0.5, 17);
        assert_eq!(a, b, "same seed → same mask");
    }

    #[test]
    fn test_feature_dropout_partial_preserves_length() {
        let emb = vec![2.0_f32; 20];
        let out = feature_dropout(&emb, 4, 0.3, 42);
        assert_eq!(out.len(), emb.len());
        // With prob 0.3 and 20 elements it is astronomically unlikely for
        // *all* elements to be dropped.
        assert!(
            out.iter().any(|&x| x != 0.0),
            "some elements should survive"
        );
    }

    // ── create_augmented_pair ─────────────────────────────────────────────

    #[test]
    fn test_create_augmented_pair_shapes() {
        let n = 4;
        let dim = 8;
        let emb = vec![1.0_f32; n * dim];
        let (a, b) = create_augmented_pair(&emb, n, dim, 0.1, 0.1, 42);
        assert_eq!(a.len(), n * dim, "view_a length mismatch");
        assert_eq!(b.len(), n * dim, "view_b length mismatch");
    }

    #[test]
    fn test_create_augmented_pair_views_differ() {
        let n = 3;
        let dim = 6;
        let emb: Vec<f32> = (0..n * dim).map(|i| i as f32 * 0.1).collect();
        let (a, b) = create_augmented_pair(&emb, n, dim, 0.1, 0.2, 0);
        assert_ne!(a, b, "the two views should differ (different seeds)");
    }

    #[test]
    fn test_create_augmented_pair_deterministic() {
        let emb = vec![0.5_f32; 12];
        let (a1, b1) = create_augmented_pair(&emb, 2, 6, 0.05, 0.1, 7);
        let (a2, b2) = create_augmented_pair(&emb, 2, 6, 0.05, 0.1, 7);
        assert_eq!(a1, a2);
        assert_eq!(b1, b2);
    }
}
