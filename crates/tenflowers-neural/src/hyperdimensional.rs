//! Hyperdimensional Computing (HDC) / Vector Symbolic Architectures (VSA).
//!
//! Implements brain-inspired, high-dimensional vector operations for
//! classification, sequence encoding, and associative memory — all in
//! pure Rust with no `unwrap()`.
//!
//! # Key Concepts
//!
//! - **Hypervector**: A very high-dimensional (≥10 000) binary or real-valued vector.
//! - **Bundle** (superposition): combines multiple hypervectors into one that is similar to all.
//! - **Bind** (binding): associates two hypervectors; result is dissimilar to both.
//! - **Permute**: cyclic shift used to encode position in sequences.
//!
//! # Quick Example
//!
//! ```rust,ignore
//! use tenflowers_neural::hyperdimensional::{BipolarHv, ItemMemory, HD_DIM};
//! use tenflowers_neural::hyperdimensional::{bind_bipolar, bundle_bipolar};
//!
//! let mut mem = ItemMemory::new(HD_DIM);
//! let a = mem.add_random("cat");
//! let b = mem.add_random("dog");
//! let bound = bind_bipolar(&a, &b);
//! if let Some((label, _score)) = mem.lookup(&bound) {
//!     println!("nearest: {}", label);
//! }
//! ```

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use tenflowers_core::TensorError;

// ─────────────────────────────────────────────────────────────────────────────
// Seed helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a deterministic (seeded) RNG from a u64 seed.
#[inline]
fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Build a non-deterministic RNG seeded from wall-clock nanoseconds.
#[inline]
fn nondeterministic_rng() -> StdRng {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 ^ (d.as_secs().wrapping_mul(6_364_136_223_846_793_005)))
        .unwrap_or(42);
    StdRng::seed_from_u64(nanos)
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Default hypervector dimensionality used throughout HDC literature.
pub const HD_DIM: usize = 10_000;

// ─────────────────────────────────────────────────────────────────────────────
// HvType
// ─────────────────────────────────────────────────────────────────────────────

/// The algebraic type of a hypervector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HvType {
    /// Each component is 0 or 1.
    Binary,
    /// Each component is +1 or -1.
    Bipolar,
    /// Each component is a continuous real number.
    Real,
}

// ─────────────────────────────────────────────────────────────────────────────
// BinaryHv
// ─────────────────────────────────────────────────────────────────────────────

/// A binary hypervector (each component is `true`/`false`).
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryHv {
    /// The raw bit data; length equals the dimensionality.
    pub data: Vec<bool>,
}

impl BinaryHv {
    /// Create a random binary hypervector with each bit independently set to
    /// `true` with probability 0.5.
    pub fn random(dim: usize) -> Self {
        let mut rng = nondeterministic_rng();
        let data = (0..dim).map(|_| rng.random::<f64>() < 0.5).collect();
        Self { data }
    }

    /// Create a seeded random binary hypervector (for reproducible tests).
    pub fn random_seeded(dim: usize, seed: u64) -> Self {
        let mut rng = seeded_rng(seed);
        let data = (0..dim).map(|_| rng.random::<f64>() < 0.5).collect();
        Self { data }
    }

    /// All-zeros binary hypervector.
    pub fn zeros(dim: usize) -> Self {
        Self {
            data: vec![false; dim],
        }
    }

    /// All-ones binary hypervector.
    pub fn ones(dim: usize) -> Self {
        Self {
            data: vec![true; dim],
        }
    }

    /// Dimensionality of this hypervector.
    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Normalised Hamming distance in `[0, 1]`.
    ///
    /// Returns `0.0` for identical vectors and `1.0` for complementary vectors.
    pub fn hamming_distance(&self, other: &BinaryHv) -> f64 {
        let d = self.dim().min(other.dim());
        if d == 0 {
            return 0.0;
        }
        let differing = self
            .data
            .iter()
            .zip(other.data.iter())
            .filter(|(a, b)| a != b)
            .count();
        differing as f64 / d as f64
    }

    /// Cosine similarity between the ±1 representations of the two binary HVs.
    ///
    /// Maps each bit `b` to `+1` if `true` and `-1` if `false` before computing
    /// cosine.  Returns a value in `[-1, 1]`.
    pub fn cosine_similarity(&self, other: &BinaryHv) -> f64 {
        let d = self.dim().min(other.dim());
        if d == 0 {
            return 0.0;
        }
        // cos = (sum of products) / d   — because both have unit L2 norm √d after mapping
        let dot: f64 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| {
                let av = if *a { 1.0_f64 } else { -1.0_f64 };
                let bv = if *b { 1.0_f64 } else { -1.0_f64 };
                av * bv
            })
            .sum();
        dot / d as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BipolarHv
// ─────────────────────────────────────────────────────────────────────────────

/// A bipolar hypervector (each component is +1 or -1).
#[derive(Debug, Clone, PartialEq)]
pub struct BipolarHv {
    /// The raw data; each element must be exactly +1 or -1.
    pub data: Vec<f64>,
}

impl BipolarHv {
    /// Create a random bipolar hypervector with each component independently
    /// sampled from {+1, -1} with equal probability.
    pub fn random(dim: usize) -> Self {
        let mut rng = nondeterministic_rng();
        let data = (0..dim)
            .map(|_| {
                if rng.random::<f64>() < 0.5 {
                    1.0_f64
                } else {
                    -1.0_f64
                }
            })
            .collect();
        Self { data }
    }

    /// Seeded variant for reproducible construction.
    pub fn random_seeded(dim: usize, seed: u64) -> Self {
        let mut rng = seeded_rng(seed);
        let data = (0..dim)
            .map(|_| {
                if rng.random::<f64>() < 0.5 {
                    1.0_f64
                } else {
                    -1.0_f64
                }
            })
            .collect();
        Self { data }
    }

    /// Dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Dot product between two bipolar HVs.
    pub fn dot(&self, other: &BipolarHv) -> f64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Cosine similarity.  For unit bipolar HVs the squared norm is `dim`, so
    /// `cos(u, v) = dot(u, v) / dim`.
    pub fn cosine_similarity(&self, other: &BipolarHv) -> f64 {
        let d = self.dim().min(other.dim());
        if d == 0 {
            return 0.0;
        }
        let dp: f64 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f64 = self.data.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = other.data.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dp / (norm_a * norm_b)
    }

    /// Convert to `BinaryHv` by mapping +1 → true, -1 → false.
    pub fn to_binary(&self) -> BinaryHv {
        BinaryHv {
            data: self.data.iter().map(|&x| x > 0.0).collect(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RealHv
// ─────────────────────────────────────────────────────────────────────────────

/// A real-valued hypervector (components are continuous floats).
#[derive(Debug, Clone, PartialEq)]
pub struct RealHv {
    /// The raw continuous-valued data.
    pub data: Vec<f64>,
}

impl RealHv {
    /// Create a random real hypervector with each component i.i.d. N(0, 1).
    pub fn random(dim: usize) -> Self {
        let mut rng = nondeterministic_rng();
        let data = (0..dim)
            .map(|_| {
                // Box-Muller transform for N(0,1)
                let u1: f64 = rng.random::<f64>().max(1e-15);
                let u2: f64 = rng.random::<f64>();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect();
        Self { data }
    }

    /// Dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Cosine similarity.
    pub fn cosine_similarity(&self, other: &RealHv) -> f64 {
        let d = self.dim().min(other.dim());
        if d == 0 {
            return 0.0;
        }
        let dp: f64 = self.data[..d]
            .iter()
            .zip(other.data[..d].iter())
            .map(|(a, b)| a * b)
            .sum();
        let na: f64 = self.data[..d].iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = other.data[..d].iter().map(|x| x * x).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dp / (na * nb)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VSA Operations — Binary
// ─────────────────────────────────────────────────────────────────────────────

/// Bundle (superposition) of binary HVs via majority vote per position.
///
/// Tied votes (equal number of true and false inputs) are resolved to `true`
/// (deterministic tie-break that preserves reproducibility).
pub fn bundle_binary(hvs: &[&BinaryHv]) -> BinaryHv {
    if hvs.is_empty() {
        return BinaryHv::zeros(0);
    }
    let dim = hvs[0].dim();
    let mut data = Vec::with_capacity(dim);
    for pos in 0..dim {
        let votes: i64 = hvs
            .iter()
            .map(|hv| {
                if pos < hv.data.len() && hv.data[pos] {
                    1
                } else {
                    -1
                }
            })
            .sum();
        // votes >= 0 → true (tie-break to true), votes < 0 → false
        data.push(votes >= 0);
    }
    BinaryHv { data }
}

/// Bundle (superposition) of bipolar HVs: element-wise sum, then sign.
///
/// Zero-sum components are resolved to +1 (deterministic tie-break).
/// This ensures that encoding the same inputs always yields the same output.
pub fn bundle_bipolar(hvs: &[&BipolarHv]) -> BipolarHv {
    if hvs.is_empty() {
        return BipolarHv { data: vec![] };
    }
    let dim = hvs[0].dim();
    let mut sums = vec![0.0_f64; dim];
    for hv in hvs {
        for (i, &v) in hv.data.iter().enumerate() {
            if i < dim {
                sums[i] += v;
            }
        }
    }
    let data = sums
        .iter()
        .map(|&s| if s >= 0.0 { 1.0 } else { -1.0 })
        .collect();
    BipolarHv { data }
}

/// Bind two binary HVs via element-wise XOR.
pub fn bind_binary(a: &BinaryHv, b: &BinaryHv) -> BinaryHv {
    let dim = a.dim().min(b.dim());
    let data = a.data[..dim]
        .iter()
        .zip(b.data[..dim].iter())
        .map(|(x, y)| x ^ y)
        .collect();
    BinaryHv { data }
}

/// Bind two bipolar HVs via element-wise product (Hadamard product).
pub fn bind_bipolar(a: &BipolarHv, b: &BipolarHv) -> BipolarHv {
    let dim = a.dim().min(b.dim());
    let data = a.data[..dim]
        .iter()
        .zip(b.data[..dim].iter())
        .map(|(x, y)| x * y)
        .collect();
    BipolarHv { data }
}

/// Permute a binary HV by a cyclic left shift of `k` positions.
///
/// `k` may be negative (right shift) or exceed `dim` (wraps around).
pub fn permute_binary(hv: &BinaryHv, k: i64) -> BinaryHv {
    let dim = hv.dim();
    if dim == 0 {
        return hv.clone();
    }
    let shift = ((k % dim as i64) + dim as i64) as usize % dim;
    let mut data = vec![false; dim];
    for i in 0..dim {
        data[(i + dim - shift) % dim] = hv.data[i];
    }
    BinaryHv { data }
}

/// Permute a bipolar HV by a cyclic left shift of `k` positions.
pub fn permute_bipolar(hv: &BipolarHv, k: i64) -> BipolarHv {
    let dim = hv.dim();
    if dim == 0 {
        return hv.clone();
    }
    let shift = ((k % dim as i64) + dim as i64) as usize % dim;
    let mut data = vec![0.0_f64; dim];
    for i in 0..dim {
        data[(i + dim - shift) % dim] = hv.data[i];
    }
    BipolarHv { data }
}

/// Unbind a binary HV (XOR is self-inverse: `unbind == bind`).
#[inline]
pub fn unbind_binary(bound: &BinaryHv, key: &BinaryHv) -> BinaryHv {
    bind_binary(bound, key)
}

/// Unbind a bipolar HV (Hadamard product is self-inverse for ±1 vectors).
#[inline]
pub fn unbind_bipolar(bound: &BipolarHv, key: &BipolarHv) -> BipolarHv {
    bind_bipolar(bound, key)
}

// ─────────────────────────────────────────────────────────────────────────────
// Item Memory (Associative Memory)
// ─────────────────────────────────────────────────────────────────────────────

/// An associative item memory mapping string labels to bipolar hypervectors.
///
/// Lookup is nearest-neighbour cosine similarity.
#[derive(Debug, Clone)]
pub struct ItemMemory {
    /// Stored label → hypervector pairs.
    pub items: HashMap<String, BipolarHv>,
    /// Dimensionality of stored hypervectors.
    pub dim: usize,
}

impl ItemMemory {
    /// Create an empty item memory with the given dimensionality.
    pub fn new(dim: usize) -> Self {
        Self {
            items: HashMap::new(),
            dim,
        }
    }

    /// Store a hypervector under `label`.
    pub fn add(&mut self, label: &str, hv: BipolarHv) {
        self.items.insert(label.to_string(), hv);
    }

    /// Retrieve the stored hypervector for `label`, if present.
    pub fn get(&self, label: &str) -> Option<&BipolarHv> {
        self.items.get(label)
    }

    /// Find the label whose stored hypervector is most similar (cosine) to
    /// `query`.  Returns `None` if the memory is empty.
    pub fn lookup(&self, query: &BipolarHv) -> Option<(String, f64)> {
        let mut best_label = None;
        let mut best_score = f64::NEG_INFINITY;
        for (label, hv) in &self.items {
            let score = hv.cosine_similarity(query);
            if score > best_score {
                best_score = score;
                best_label = Some(label.clone());
            }
        }
        best_label.map(|l| (l, best_score))
    }

    /// Create a random hypervector, store it under `label`, and return a clone.
    pub fn add_random(&mut self, label: &str) -> BipolarHv {
        let hv = BipolarHv::random(self.dim);
        self.items.insert(label.to_string(), hv.clone());
        hv
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HD Encoding Strategies
// ─────────────────────────────────────────────────────────────────────────────

/// Level encoder: maps a continuous scalar to one of `n_levels` discrete
/// bipolar hypervectors.
///
/// Consecutive level hypervectors differ by exactly `dim / n_levels` bits,
/// giving them a graded similarity profile (thermometer-like in HV space).
#[derive(Debug, Clone)]
pub struct LevelEncoder {
    /// Ordered level hypervectors (index 0 = lowest value).
    pub levels: Vec<BipolarHv>,
    /// Lower bound of the encoded range.
    pub x_min: f64,
    /// Upper bound of the encoded range.
    pub x_max: f64,
}

impl LevelEncoder {
    /// Build a new `LevelEncoder` with `n_levels` equally spaced levels.
    ///
    /// Construction: `levels[0]` is a random HV; each subsequent level flips
    /// `dim / n_levels` randomly chosen positions (without replacement within
    /// each step).
    pub fn new(n_levels: usize, dim: usize, x_min: f64, x_max: f64) -> Self {
        assert!(n_levels >= 1, "n_levels must be at least 1");
        let mut rng = nondeterministic_rng();
        // Build level 0 randomly.
        let base: Vec<f64> = (0..dim)
            .map(|_| if rng.random::<f64>() < 0.5 { 1.0 } else { -1.0 })
            .collect();
        let mut levels = vec![BipolarHv { data: base }];
        // Determine how many bits to flip per step.
        let flips_per_step = (dim / n_levels).max(1);
        // Build a permutation of indices to deterministically choose flip positions.
        let mut indices: Vec<usize> = (0..dim).collect();
        // Shuffle once, then use slices of size `flips_per_step`.
        for i in (1..dim).rev() {
            let j = (rng.random::<f64>() * (i + 1) as f64) as usize;
            indices.swap(i, j);
        }
        let mut current = levels[0].data.clone();
        let mut flip_cursor = 0usize;
        for _level in 1..n_levels {
            // Flip the next `flips_per_step` indices.
            for k in 0..flips_per_step {
                let idx = indices[(flip_cursor + k) % dim];
                current[idx] = -current[idx];
            }
            flip_cursor = (flip_cursor + flips_per_step) % dim;
            levels.push(BipolarHv {
                data: current.clone(),
            });
        }
        Self {
            levels,
            x_min,
            x_max,
        }
    }

    /// Quantise `x` to the nearest level and return a reference to its HV.
    pub fn encode(&self, x: f64) -> &BipolarHv {
        let n = self.levels.len();
        if n == 1 {
            return &self.levels[0];
        }
        let clamped = x.max(self.x_min).min(self.x_max);
        let t = (clamped - self.x_min) / (self.x_max - self.x_min);
        let idx = ((t * (n - 1) as f64).round() as usize).min(n - 1);
        &self.levels[idx]
    }
}

/// Thermometer encoder: encodes a level index by flipping progressively more
/// bits from a fixed base vector.
///
/// Level `k` has the first `k * dim / n_levels` bits flipped relative to the
/// base, giving a monotone similarity ordering.
#[derive(Debug, Clone)]
pub struct ThermometerEncoder {
    /// The base (all-random) hypervector for level 0.
    pub base_hv: BipolarHv,
    /// Ordered flip positions (length == dim).
    pub flip_hvs: Vec<BipolarHv>,
    /// Number of discrete levels.
    pub n_levels: usize,
}

impl ThermometerEncoder {
    /// Build a new `ThermometerEncoder` with `n_levels` levels and given `dim`.
    pub fn new(n_levels: usize, dim: usize) -> Self {
        let mut rng = nondeterministic_rng();
        let base_data: Vec<f64> = (0..dim)
            .map(|_| if rng.random::<f64>() < 0.5 { 1.0 } else { -1.0 })
            .collect();
        let base_hv = BipolarHv { data: base_data };
        // Precompute intermediate flip HVs for each level (unused in encode_level
        // but stored for transparency / inspection).
        let mut flip_hvs = Vec::with_capacity(n_levels);
        let flips_per_level = (dim / n_levels.max(1)).max(1);
        let mut current = base_hv.data.clone();
        for level in 0..n_levels {
            let start = level * flips_per_level;
            let end = ((level + 1) * flips_per_level).min(dim);
            for i in start..end {
                current[i] = -current[i];
            }
            flip_hvs.push(BipolarHv {
                data: current.clone(),
            });
        }
        Self {
            base_hv,
            flip_hvs,
            n_levels,
        }
    }

    /// Encode `level` (0-indexed) as a thermometer HV.
    ///
    /// Returns a new BipolarHv that has the first `level * dim / n_levels` bits
    /// of the base flipped.
    pub fn encode_level(&self, level: usize) -> BipolarHv {
        let dim = self.base_hv.dim();
        let n = self.n_levels.max(1);
        let flips_per_level = (dim / n).max(1);
        let n_flipped = (level * flips_per_level).min(dim);
        let mut data = self.base_hv.data.clone();
        for i in 0..n_flipped {
            data[i] = -data[i];
        }
        BipolarHv { data }
    }
}

/// ID encoder: maps integer IDs to random (maximally orthogonal) bipolar HVs.
///
/// Each new ID receives a freshly sampled random HV, which is then cached.
#[derive(Debug, Clone)]
pub struct IdEncoder {
    /// Cached HV per integer ID.
    pub ids: HashMap<usize, BipolarHv>,
    /// HV dimensionality.
    pub dim: usize,
}

impl IdEncoder {
    /// Create an empty `IdEncoder`.
    pub fn new(dim: usize) -> Self {
        Self {
            ids: HashMap::new(),
            dim,
        }
    }

    /// Return the HV for `id`, creating and caching a new random one if absent.
    pub fn get_or_create(&mut self, id: usize) -> BipolarHv {
        if !self.ids.contains_key(&id) {
            let hv = BipolarHv::random(self.dim);
            self.ids.insert(id, hv);
        }
        // Safe: we just inserted it above if missing.
        match self.ids.get(&id) {
            Some(hv) => hv.clone(),
            None => BipolarHv::random(self.dim), // unreachable
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HD Classifier
// ─────────────────────────────────────────────────────────────────────────────

/// Online HD classifier that accumulates class prototype hypervectors.
///
/// Training bundles each new sample into the running class prototype; inference
/// finds the most similar prototype by cosine similarity.
#[derive(Debug, Clone)]
pub struct HdClassifier {
    /// Accumulated (un-normalised) prototype HV per class label.
    pub class_hvs: HashMap<String, BipolarHv>,
    /// HV dimensionality.
    pub dim: usize,
    /// Number of training samples added per class.
    pub class_counts: HashMap<String, usize>,
}

impl HdClassifier {
    /// Create a new (empty) classifier.
    pub fn new(dim: usize) -> Self {
        Self {
            class_hvs: HashMap::new(),
            dim,
            class_counts: HashMap::new(),
        }
    }

    /// Incorporate one labelled sample into the running class prototype.
    pub fn train_one(&mut self, sample_hv: &BipolarHv, label: &str) {
        let count = self.class_counts.entry(label.to_string()).or_insert(0);
        *count += 1;
        let entry = self
            .class_hvs
            .entry(label.to_string())
            .or_insert_with(|| BipolarHv {
                data: vec![0.0; self.dim],
            });
        // Accumulate raw sum — we binarise lazily at prediction time.
        for (acc, &s) in entry.data.iter_mut().zip(sample_hv.data.iter()) {
            *acc += s;
        }
    }

    /// Predict the class label for `query`.  Returns `None` if no training has
    /// occurred.
    pub fn predict(&self, query: &BipolarHv) -> Option<String> {
        self.predict_with_score(query).map(|(l, _)| l)
    }

    /// Predict class and return the cosine similarity score.
    pub fn predict_with_score(&self, query: &BipolarHv) -> Option<(String, f64)> {
        let mut best_label: Option<String> = None;
        let mut best_score = f64::NEG_INFINITY;
        for (label, acc_hv) in &self.class_hvs {
            // Binarise the accumulated prototype on-the-fly.
            let proto = binarise_accumulator(acc_hv);
            let score = proto.cosine_similarity(query);
            if score > best_score {
                best_score = score;
                best_label = Some(label.clone());
            }
        }
        best_label.map(|l| (l, best_score))
    }

    /// Online error-corrective retraining.
    ///
    /// When a sample is predicted as `predicted` but the true label is
    /// `true_label`, subtract the sample from the wrong prototype and add it
    /// to the correct prototype.
    pub fn retrain_wrong(&mut self, sample_hv: &BipolarHv, predicted: &str, true_label: &str) {
        // Subtract from the wrong class.
        if let Some(wrong_hv) = self.class_hvs.get_mut(predicted) {
            for (acc, &s) in wrong_hv.data.iter_mut().zip(sample_hv.data.iter()) {
                *acc -= s;
            }
        }
        // Add to the correct class.
        self.train_one(sample_hv, true_label);
    }
}

/// Convert an accumulated (real-valued sum) HV to a bipolar HV.
///
/// Zero components are resolved to +1 (arbitrary but deterministic tie-break).
fn binarise_accumulator(acc: &BipolarHv) -> BipolarHv {
    let data = acc
        .data
        .iter()
        .map(|&v| if v >= 0.0 { 1.0 } else { -1.0 })
        .collect();
    BipolarHv { data }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparse Distributed Memory (Kanerva SDM)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a sparse distributed memory.
#[derive(Debug, Clone)]
pub struct SdmConfig {
    /// Number of bits in the address space.
    pub address_dim: usize,
    /// Width of each data word (bits).
    pub data_dim: usize,
    /// Number of randomly placed hard locations.
    pub n_hard_locations: usize,
    /// Maximum Hamming distance for a location to be activated.
    pub hamming_threshold: usize,
}

/// Kanerva Sparse Distributed Memory with integer counters.
///
/// Each hard location has an address (random binary) and a data counter array
/// (one integer per data bit).  Writing increments (if data bit = 1) or
/// decrements (if data bit = 0) activated locations.  Reading sums counters
/// and thresholds at 0.
#[derive(Debug, Clone)]
pub struct SparseSdm {
    /// Random binary addresses: `[n_locations][address_dim]`.
    pub addresses: Vec<Vec<bool>>,
    /// Integer counter arrays: `[n_locations][data_dim]`.
    pub counters: Vec<Vec<i32>>,
    /// Configuration used at construction.
    pub config: SdmConfig,
}

impl SparseSdm {
    /// Create a new SDM with randomly sampled hard-location addresses.
    pub fn new(config: SdmConfig) -> Self {
        let mut rng = nondeterministic_rng();
        let addresses: Vec<Vec<bool>> = (0..config.n_hard_locations)
            .map(|_| {
                (0..config.address_dim)
                    .map(|_| rng.random::<f64>() < 0.5)
                    .collect()
            })
            .collect();
        let counters = vec![vec![0i32; config.data_dim]; config.n_hard_locations];
        Self {
            addresses,
            counters,
            config,
        }
    }

    /// Return the indices of all hard locations activated by `address`.
    pub fn activated_locations(&self, address: &[bool]) -> Vec<usize> {
        self.addresses
            .iter()
            .enumerate()
            .filter_map(|(i, loc_addr)| {
                let dist = hamming_bool(address, loc_addr);
                if dist <= self.config.hamming_threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Write `data` to all activated hard locations.
    pub fn write(&mut self, address: &[bool], data: &[bool]) {
        let activated = self.activated_locations(address);
        for loc_idx in activated {
            for (bit_idx, &data_bit) in data.iter().enumerate() {
                if bit_idx < self.config.data_dim {
                    if data_bit {
                        self.counters[loc_idx][bit_idx] += 1;
                    } else {
                        self.counters[loc_idx][bit_idx] -= 1;
                    }
                }
            }
        }
    }

    /// Read from all activated hard locations, sum counters, threshold at 0.
    pub fn read(&self, address: &[bool]) -> Vec<bool> {
        let activated = self.activated_locations(address);
        let mut sums = vec![0i32; self.config.data_dim];
        for loc_idx in &activated {
            for (bit_idx, &counter) in self.counters[*loc_idx].iter().enumerate() {
                sums[bit_idx] += counter;
            }
        }
        sums.iter().map(|&s| s >= 0).collect()
    }
}

/// Hamming distance between two boolean slices (counts differing positions).
fn hamming_bool(a: &[bool], b: &[bool]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

// ─────────────────────────────────────────────────────────────────────────────
// Online HDC (Incremental Learning)
// ─────────────────────────────────────────────────────────────────────────────

/// End-to-end online HDC pipeline: feature-level encoding → classifier.
///
/// Each feature dimension is represented by an ID hypervector; the scalar value
/// of that feature is encoded by a level encoder.  The per-feature contribution
/// is `bind(feature_id_hv, level_hv)` and contributions are bundled together.
#[derive(Debug, Clone)]
pub struct OnlineHdc {
    /// Associative memory for feature IDs.
    pub item_memory: ItemMemory,
    /// Shared level encoder for all features.
    pub level_encoder: LevelEncoder,
    /// HD classifier accumulating class prototypes.
    pub classifier: HdClassifier,
    /// HV dimensionality.
    pub dim: usize,
    /// Number of features per sample.
    pub n_features: usize,
}

impl OnlineHdc {
    /// Construct a fresh pipeline.
    ///
    /// * `n_features`: number of input features.
    /// * `n_levels`: quantisation resolution for the level encoder.
    /// * `dim`: HV dimensionality.
    ///
    /// The level encoder covers the range `[-1.0, 1.0]` by default; normalise
    /// your data if needed.
    pub fn new(n_features: usize, n_levels: usize, dim: usize) -> Self {
        let item_memory = ItemMemory::new(dim);
        let level_encoder = LevelEncoder::new(n_levels, dim, -1.0, 1.0);
        let classifier = HdClassifier::new(dim);
        Self {
            item_memory,
            level_encoder,
            classifier,
            dim,
            n_features,
        }
    }

    /// Encode a feature vector into a single bipolar hypervector.
    ///
    /// For each feature `i`, retrieves (or creates) a random ID HV, encodes
    /// `features[i]` via the level encoder, binds them, then bundles across
    /// all features.
    pub fn encode_sample(&mut self, features: &[f64]) -> BipolarHv {
        let n = features.len().min(self.n_features);
        let mut component_hvs: Vec<BipolarHv> = Vec::with_capacity(n);
        for i in 0..n {
            let id_label = format!("feature_{i}");
            let id_hv = match self.item_memory.items.get(&id_label) {
                Some(hv) => hv.clone(),
                None => {
                    let hv = BipolarHv::random(self.dim);
                    self.item_memory.add(&id_label, hv.clone());
                    hv
                }
            };
            let level_hv = self.level_encoder.encode(features[i]).clone();
            component_hvs.push(bind_bipolar(&id_hv, &level_hv));
        }
        if component_hvs.is_empty() {
            return BipolarHv::random(self.dim);
        }
        let refs: Vec<&BipolarHv> = component_hvs.iter().collect();
        bundle_bipolar(&refs)
    }

    /// Encode and train on a single labelled sample.
    pub fn train(&mut self, features: &[f64], label: &str) {
        let hv = self.encode_sample(features);
        self.classifier.train_one(&hv, label);
    }

    /// Encode a query and predict its class label.
    pub fn predict(&mut self, features: &[f64]) -> Option<String> {
        let hv = self.encode_sample(features);
        self.classifier.predict(&hv)
    }

    /// Compute accuracy over a slice of (features, label) pairs.
    pub fn accuracy(&mut self, samples: &[(Vec<f64>, String)]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut correct = 0usize;
        for (features, true_label) in samples {
            if let Some(pred) = self.predict(features) {
                if &pred == true_label {
                    correct += 1;
                }
            }
        }
        correct as f64 / samples.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequence Modeling with Permutation
// ─────────────────────────────────────────────────────────────────────────────

/// Sequence encoder using permutation-based positional encoding.
///
/// A sequence `[a, b, c]` is encoded as:
/// `bundle(permute(hv_a, 2), permute(hv_b, 1), hv_c)`
///
/// This preserves order information: the same tokens in a different order
/// produce a dissimilar HV.
#[derive(Debug, Clone)]
pub struct HdSequenceEncoder {
    /// Backing item memory for token HVs.
    pub item_memory: ItemMemory,
    /// HV dimensionality.
    pub dim: usize,
}

impl HdSequenceEncoder {
    /// Create a new sequence encoder.
    pub fn new(dim: usize) -> Self {
        Self {
            item_memory: ItemMemory::new(dim),
            dim,
        }
    }

    /// Retrieve or create the HV for a token string.
    fn get_or_add_token(&mut self, token: &str) -> BipolarHv {
        match self.item_memory.items.get(token) {
            Some(hv) => hv.clone(),
            None => {
                let hv = BipolarHv::random(self.dim);
                self.item_memory.add(token, hv.clone());
                hv
            }
        }
    }

    /// Encode a token sequence into a single bipolar HV.
    ///
    /// Position 0 (last token in the reversed indexing scheme) receives
    /// permutation 0, position 1 receives permutation 1, etc.
    pub fn encode_sequence(&mut self, tokens: &[&str]) -> BipolarHv {
        let n = tokens.len();
        if n == 0 {
            return BipolarHv::random(self.dim);
        }
        let mut components: Vec<BipolarHv> = Vec::with_capacity(n);
        for (i, &token) in tokens.iter().enumerate() {
            let hv = self.get_or_add_token(token);
            // shift amount: last token gets 0, second-to-last gets 1, …, first gets n-1
            let shift = (n - 1 - i) as i64;
            components.push(permute_bipolar(&hv, shift));
        }
        let refs: Vec<&BipolarHv> = components.iter().collect();
        bundle_bipolar(&refs)
    }

    /// Query whether `token` appears at `position` (0-indexed from the start).
    ///
    /// Returns the cosine similarity between the decoded vector and the token
    /// HV.  A high value (>0.5) indicates likely presence at that position.
    pub fn query_position(
        &self,
        sequence_hv: &BipolarHv,
        token: &str,
        position: usize,
        seq_len: usize,
    ) -> f64 {
        let token_hv = match self.item_memory.items.get(token) {
            Some(hv) => hv,
            None => return 0.0,
        };
        // The shift applied at encoding was (seq_len - 1 - position).
        let shift = (seq_len.saturating_sub(1).saturating_sub(position)) as i64;
        // Undo the permutation.
        let unshifted = permute_bipolar(sequence_hv, -shift);
        unshifted.cosine_similarity(token_hv)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HD Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics measuring the separability of class HVs.
#[derive(Debug, Clone)]
pub struct HdcStats {
    /// Mean intra-class cosine similarity (same class pairs).
    pub mean_similarity_same_class: f64,
    /// Mean inter-class cosine similarity (cross-class pairs).
    pub mean_similarity_diff_class: f64,
    /// Separation = `mean_same - mean_diff` (higher is better).
    pub separation: f64,
}

/// Compute HDC statistics from a map of class prototype HVs.
///
/// For classes with a single prototype each, intra-class similarity is
/// trivially 1.0 (a vector with itself).
pub fn compute_hdc_stats(class_hvs: &HashMap<String, BipolarHv>) -> HdcStats {
    let labels: Vec<&String> = class_hvs.keys().collect();
    let n = labels.len();
    if n == 0 {
        return HdcStats {
            mean_similarity_same_class: 0.0,
            mean_similarity_diff_class: 0.0,
            separation: 0.0,
        };
    }
    // Intra-class: each prototype with itself = 1.0.
    let mean_same = 1.0_f64;
    // Inter-class: all distinct pairs.
    let mut cross_sum = 0.0_f64;
    let mut cross_count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let hv_i = &class_hvs[labels[i]];
            let hv_j = &class_hvs[labels[j]];
            cross_sum += hv_i.cosine_similarity(hv_j);
            cross_count += 1;
        }
    }
    let mean_diff = if cross_count > 0 {
        cross_sum / cross_count as f64
    } else {
        0.0
    };
    HdcStats {
        mean_similarity_same_class: mean_same,
        mean_similarity_diff_class: mean_diff,
        separation: mean_same - mean_diff,
    }
}

/// Compute mean pairwise cosine similarity among a set of HVs.
///
/// For a set of truly random bipolar HVs, the expected value is ≈ 0 (orthogonality).
pub fn orthogonality_test(hvs: &[BipolarHv]) -> f64 {
    let n = hvs.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += hvs[i].cosine_similarity(&hvs[j]);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise from HDC operations.
#[derive(Debug, Clone)]
pub enum HdcError {
    /// Dimension mismatch between two hypervectors.
    DimensionMismatch { expected: usize, got: usize },
    /// An empty collection was supplied where at least one element was required.
    EmptyInput,
    /// The item memory was queried but is empty.
    EmptyMemory,
}

impl std::fmt::Display for HdcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HdcError::DimensionMismatch { expected, got } => {
                write!(f, "HDC dimension mismatch: expected {expected}, got {got}")
            }
            HdcError::EmptyInput => write!(f, "HDC operation requires at least one input"),
            HdcError::EmptyMemory => write!(f, "HDC item memory is empty"),
        }
    }
}

impl std::error::Error for HdcError {}

impl From<HdcError> for TensorError {
    fn from(e: HdcError) -> Self {
        TensorError::invalid_argument(e.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 1000; // smaller than HD_DIM for test speed
    const LARGE_DIM: usize = 5000;

    // ── BinaryHv ─────────────────────────────────────────────────────────────

    #[test]
    fn test_binary_random_approximately_half_ones() {
        let hv = BinaryHv::random_seeded(10_000, 0);
        let ones = hv.data.iter().filter(|&&b| b).count();
        let ratio = ones as f64 / 10_000.0;
        // Expect within 5% of 50%.
        assert!((ratio - 0.5).abs() < 0.05, "ratio={ratio}");
    }

    #[test]
    fn test_binary_zeros_all_false() {
        let hv = BinaryHv::zeros(100);
        assert!(hv.data.iter().all(|&b| !b));
    }

    #[test]
    fn test_binary_ones_all_true() {
        let hv = BinaryHv::ones(100);
        assert!(hv.data.iter().all(|&b| b));
    }

    #[test]
    fn test_binary_hamming_identical() {
        let hv = BinaryHv::random_seeded(DIM, 1);
        assert!((hv.hamming_distance(&hv) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_binary_hamming_complement() {
        let hv = BinaryHv::random_seeded(100, 2);
        let comp = BinaryHv {
            data: hv.data.iter().map(|&b| !b).collect(),
        };
        let d = hv.hamming_distance(&comp);
        assert!((d - 1.0).abs() < 1e-9, "complement hamming={d}");
    }

    #[test]
    fn test_binary_cosine_identical() {
        let hv = BinaryHv::random_seeded(DIM, 3);
        let cos = hv.cosine_similarity(&hv);
        assert!((cos - 1.0).abs() < 1e-9, "cos={cos}");
    }

    #[test]
    fn test_binary_cosine_complement_is_minus_one() {
        let hv = BinaryHv::random_seeded(100, 4);
        let comp = BinaryHv {
            data: hv.data.iter().map(|&b| !b).collect(),
        };
        let cos = hv.cosine_similarity(&comp);
        assert!((cos + 1.0).abs() < 1e-9, "cos={cos}");
    }

    // ── BipolarHv ─────────────────────────────────────────────────────────────

    #[test]
    fn test_bipolar_random_approximately_half_positive() {
        let hv = BipolarHv::random_seeded(10_000, 5);
        let pos = hv.data.iter().filter(|&&v| v > 0.0).count();
        let ratio = pos as f64 / 10_000.0;
        assert!((ratio - 0.5).abs() < 0.05, "ratio={ratio}");
    }

    #[test]
    fn test_bipolar_cosine_identical() {
        let hv = BipolarHv::random_seeded(DIM, 6);
        let cos = hv.cosine_similarity(&hv);
        assert!((cos - 1.0).abs() < 1e-9, "cos={cos}");
    }

    #[test]
    fn test_bipolar_dot_product() {
        let hv = BipolarHv {
            data: vec![1.0, -1.0, 1.0, 1.0],
        };
        let other = BipolarHv {
            data: vec![1.0, 1.0, -1.0, 1.0],
        };
        let dot = hv.dot(&other);
        // 1*1 + (-1)*1 + 1*(-1) + 1*1 = 1 - 1 - 1 + 1 = 0
        assert!((dot - 0.0).abs() < 1e-9, "dot={dot}");
    }

    // ── Bundle ────────────────────────────────────────────────────────────────

    #[test]
    fn test_bundle_binary_identical_returns_same() {
        let hv = BinaryHv::random_seeded(DIM, 7);
        let refs: Vec<&BinaryHv> = vec![&hv, &hv, &hv];
        let result = bundle_binary(&refs);
        // Majority of identical = original
        assert_eq!(result.data, hv.data);
    }

    #[test]
    fn test_bundle_bipolar_identical_returns_same() {
        let hv = BipolarHv::random_seeded(DIM, 8);
        let refs: Vec<&BipolarHv> = vec![&hv, &hv, &hv];
        let result = bundle_bipolar(&refs);
        assert_eq!(result.data, hv.data);
    }

    #[test]
    fn test_bundle_bipolar_two_different_is_similar_to_both() {
        let a = BipolarHv::random_seeded(LARGE_DIM, 9);
        let b = BipolarHv::random_seeded(LARGE_DIM, 10);
        let bundled = bundle_bipolar(&[&a, &b]);
        let cos_a = bundled.cosine_similarity(&a);
        let cos_b = bundled.cosine_similarity(&b);
        // Bundle should be roughly equally similar to both (>0).
        assert!(cos_a > 0.0, "cos_a={cos_a}");
        assert!(cos_b > 0.0, "cos_b={cos_b}");
    }

    // ── Bind / Unbind ─────────────────────────────────────────────────────────

    #[test]
    fn test_bind_binary_self_inverse() {
        let a = BinaryHv::random_seeded(DIM, 11);
        let b = BinaryHv::random_seeded(DIM, 12);
        let bound = bind_binary(&a, &b);
        let recovered = unbind_binary(&bound, &b);
        assert_eq!(recovered.data, a.data);
    }

    #[test]
    fn test_bind_bipolar_self_inverse() {
        let a = BipolarHv::random_seeded(DIM, 13);
        let b = BipolarHv::random_seeded(DIM, 14);
        let bound = bind_bipolar(&a, &b);
        let recovered = unbind_bipolar(&bound, &b);
        // After bind + unbind, recovered should be identical to a.
        let cos = recovered.cosine_similarity(&a);
        assert!((cos - 1.0).abs() < 1e-9, "cos={cos}");
    }

    #[test]
    fn test_bind_bipolar_dissimilar_to_inputs() {
        let a = BipolarHv::random_seeded(LARGE_DIM, 15);
        let b = BipolarHv::random_seeded(LARGE_DIM, 16);
        let bound = bind_bipolar(&a, &b);
        let cos_a = bound.cosine_similarity(&a);
        let cos_b = bound.cosine_similarity(&b);
        // Bound should be nearly orthogonal to both inputs.
        assert!(cos_a.abs() < 0.2, "cos_a={cos_a}");
        assert!(cos_b.abs() < 0.2, "cos_b={cos_b}");
    }

    // ── Permute ───────────────────────────────────────────────────────────────

    #[test]
    fn test_permute_binary_zero_is_identity() {
        let hv = BinaryHv::random_seeded(DIM, 17);
        let perm = permute_binary(&hv, 0);
        assert_eq!(perm.data, hv.data);
    }

    #[test]
    fn test_permute_bipolar_zero_is_identity() {
        let hv = BipolarHv::random_seeded(DIM, 18);
        let perm = permute_bipolar(&hv, 0);
        assert_eq!(perm.data, hv.data);
    }

    #[test]
    fn test_permute_then_reverse_bipolar_is_identity() {
        let hv = BipolarHv::random_seeded(DIM, 19);
        let k = 137_i64;
        let perm = permute_bipolar(&hv, k);
        let back = permute_bipolar(&perm, -k);
        assert_eq!(back.data, hv.data);
    }

    #[test]
    fn test_permute_then_reverse_binary_is_identity() {
        let hv = BinaryHv::random_seeded(DIM, 20);
        let k = 73_i64;
        let perm = permute_binary(&hv, k);
        let back = permute_binary(&perm, -k);
        assert_eq!(back.data, hv.data);
    }

    #[test]
    fn test_permute_bipolar_nonzero_dissimilar() {
        let hv = BipolarHv::random_seeded(LARGE_DIM, 21);
        let perm = permute_bipolar(&hv, 1);
        let cos = hv.cosine_similarity(&perm);
        // A shift of 1 in 5000 dims → ~0 expected cosine.
        assert!(cos.abs() < 0.2, "cos={cos}");
    }

    // ── ItemMemory ────────────────────────────────────────────────────────────

    #[test]
    fn test_item_memory_lookup_finds_correct_label() {
        let mut mem = ItemMemory::new(DIM);
        let hv_a = mem.add_random("alpha");
        let _hv_b = mem.add_random("beta");
        let _hv_c = mem.add_random("gamma");
        let result = mem.lookup(&hv_a);
        assert!(result.is_some());
        let (label, _score) = result.expect("lookup must succeed");
        assert_eq!(label, "alpha");
    }

    #[test]
    fn test_item_memory_get() {
        let mut mem = ItemMemory::new(DIM);
        mem.add_random("x");
        assert!(mem.get("x").is_some());
        assert!(mem.get("missing").is_none());
    }

    #[test]
    fn test_item_memory_empty_lookup_returns_none() {
        let mem = ItemMemory::new(DIM);
        let query = BipolarHv::random_seeded(DIM, 22);
        assert!(mem.lookup(&query).is_none());
    }

    // ── LevelEncoder ──────────────────────────────────────────────────────────

    #[test]
    fn test_level_encoder_extreme_values_map_to_first_last_level() {
        let enc = LevelEncoder::new(10, DIM, 0.0, 1.0);
        let first = enc.encode(0.0);
        let last = enc.encode(1.0);
        // Should be the first and last level HVs.
        assert_eq!(first.data, enc.levels[0].data);
        assert_eq!(last.data, enc.levels[9].data);
    }

    #[test]
    fn test_level_encoder_single_level() {
        let enc = LevelEncoder::new(1, DIM, -1.0, 1.0);
        let hv = enc.encode(0.0);
        assert_eq!(hv.data, enc.levels[0].data);
    }

    #[test]
    fn test_level_encoder_monotone_similarity() {
        // Consecutive level HVs should be more similar than non-consecutive.
        let enc = LevelEncoder::new(20, DIM, 0.0, 1.0);
        let cos_adj = enc.levels[0].cosine_similarity(&enc.levels[1]);
        let cos_far = enc.levels[0].cosine_similarity(&enc.levels[10]);
        assert!(cos_adj > cos_far, "adj={cos_adj}, far={cos_far}");
    }

    // ── ThermometerEncoder ────────────────────────────────────────────────────

    #[test]
    fn test_thermometer_encoding_level_zero_near_base() {
        let enc = ThermometerEncoder::new(10, DIM);
        let l0 = enc.encode_level(0);
        // Level 0 should be the base_hv itself (no bits flipped).
        assert_eq!(l0.data, enc.base_hv.data);
    }

    #[test]
    fn test_thermometer_encoding_monotone() {
        let enc = ThermometerEncoder::new(10, LARGE_DIM);
        // Higher levels should have lower similarity to level 0 (more bits differ).
        let cos_1 = enc.encode_level(0).cosine_similarity(&enc.encode_level(1));
        let cos_5 = enc.encode_level(0).cosine_similarity(&enc.encode_level(5));
        let cos_9 = enc.encode_level(0).cosine_similarity(&enc.encode_level(9));
        assert!(cos_1 > cos_5, "cos_1={cos_1}, cos_5={cos_5}");
        assert!(cos_5 > cos_9, "cos_5={cos_5}, cos_9={cos_9}");
    }

    #[test]
    fn test_thermometer_different_levels_distinct() {
        let enc = ThermometerEncoder::new(5, DIM);
        let l0 = enc.encode_level(0);
        let l4 = enc.encode_level(4);
        // Should not be identical.
        assert_ne!(l0.data, l4.data);
    }

    // ── IdEncoder ─────────────────────────────────────────────────────────────

    #[test]
    fn test_id_encoder_same_id_returns_same_hv() {
        let mut enc = IdEncoder::new(DIM);
        let hv1 = enc.get_or_create(42);
        let hv2 = enc.get_or_create(42);
        assert_eq!(hv1.data, hv2.data);
    }

    #[test]
    fn test_id_encoder_different_ids_orthogonal() {
        let mut enc = IdEncoder::new(LARGE_DIM);
        let hv0 = enc.get_or_create(0);
        let hv1 = enc.get_or_create(1);
        let cos = hv0.cosine_similarity(&hv1);
        assert!(cos.abs() < 0.2, "cos={cos}");
    }

    // ── HdClassifier ──────────────────────────────────────────────────────────

    #[test]
    fn test_classifier_learns_two_classes() {
        let mut clf = HdClassifier::new(LARGE_DIM);
        // Create two random class prototype HVs.
        let hv_a = BipolarHv::random_seeded(LARGE_DIM, 30);
        let hv_b = BipolarHv::random_seeded(LARGE_DIM, 31);
        // Train several samples per class.
        for _ in 0..5 {
            clf.train_one(&hv_a, "A");
            clf.train_one(&hv_b, "B");
        }
        assert_eq!(clf.predict(&hv_a).as_deref(), Some("A"));
        assert_eq!(clf.predict(&hv_b).as_deref(), Some("B"));
    }

    #[test]
    fn test_classifier_predict_empty_returns_none() {
        let clf = HdClassifier::new(DIM);
        let query = BipolarHv::random_seeded(DIM, 32);
        assert!(clf.predict(&query).is_none());
    }

    #[test]
    fn test_classifier_retrain_improves_score() {
        let mut clf = HdClassifier::new(LARGE_DIM);
        let hv_a = BipolarHv::random_seeded(LARGE_DIM, 33);
        let hv_b = BipolarHv::random_seeded(LARGE_DIM, 34);
        clf.train_one(&hv_a, "A");
        clf.train_one(&hv_b, "B");
        // Force a "misclassification" scenario: retrain hv_a from "B" to "A".
        clf.retrain_wrong(&hv_a, "B", "A");
        // After retraining, should still correctly identify A.
        let pred = clf.predict(&hv_a);
        assert_eq!(pred.as_deref(), Some("A"));
    }

    #[test]
    fn test_classifier_predict_with_score_returns_valid_cosine() {
        let mut clf = HdClassifier::new(DIM);
        let hv = BipolarHv::random_seeded(DIM, 35);
        clf.train_one(&hv, "X");
        let (_, score) = clf.predict_with_score(&hv).expect("should have result");
        assert!(score > 0.0 && score <= 1.0 + 1e-9, "score={score}");
    }

    // ── SparseSdm ─────────────────────────────────────────────────────────────

    #[test]
    fn test_sdm_write_read_roundtrip() {
        // Use a generous hamming_threshold (45 out of 100 bits) so many hard
        // locations are activated, giving robust read-back signal.
        let config = SdmConfig {
            address_dim: 100,
            data_dim: 50,
            n_hard_locations: 500,
            hamming_threshold: 45,
        };
        let mut sdm = SparseSdm::new(config);
        // Build a simple address and data.
        let address: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        let data: Vec<bool> = (0..50).map(|i| i % 3 == 0).collect();
        // Write multiple times to reinforce.
        for _ in 0..15 {
            sdm.write(&address, &data);
        }
        let recovered = sdm.read(&address);
        // Count agreements.
        let agreement = recovered
            .iter()
            .zip(data.iter())
            .filter(|(r, d)| r == d)
            .count();
        let ratio = agreement as f64 / 50.0;
        assert!(ratio > 0.8, "agreement={ratio}");
    }

    #[test]
    fn test_sdm_activated_locations_non_empty() {
        let config = SdmConfig {
            address_dim: 50,
            data_dim: 10,
            n_hard_locations: 500,
            hamming_threshold: 20,
        };
        let sdm = SparseSdm::new(config);
        let address: Vec<bool> = vec![false; 50];
        let activated = sdm.activated_locations(&address);
        // With threshold=20 out of 50 bits, expect many activations.
        assert!(!activated.is_empty(), "no locations activated");
    }

    #[test]
    fn test_sdm_empty_write_does_not_panic() {
        let config = SdmConfig {
            address_dim: 20,
            data_dim: 10,
            n_hard_locations: 100,
            hamming_threshold: 5,
        };
        let mut sdm = SparseSdm::new(config);
        let addr: Vec<bool> = vec![true; 20];
        let data: Vec<bool> = vec![false; 10];
        // Should not panic.
        sdm.write(&addr, &data);
    }

    // ── OnlineHdc ─────────────────────────────────────────────────────────────

    #[test]
    fn test_online_hdc_same_sample_encodes_similarly() {
        let mut hdc = OnlineHdc::new(4, 10, LARGE_DIM);
        let sample = vec![0.1, -0.3, 0.7, -0.9];
        let hv1 = hdc.encode_sample(&sample);
        let hv2 = hdc.encode_sample(&sample);
        let cos = hv1.cosine_similarity(&hv2);
        assert!((cos - 1.0).abs() < 1e-9, "cos={cos}");
    }

    #[test]
    fn test_online_hdc_train_predict_binary_classes() {
        let mut hdc = OnlineHdc::new(5, 20, LARGE_DIM);
        // Class A: positive features; Class B: negative features.
        for _ in 0..20 {
            hdc.train(&[0.9, 0.8, 0.7, 0.85, 0.75], "A");
            hdc.train(&[-0.9, -0.8, -0.7, -0.85, -0.75], "B");
        }
        let pred_a = hdc.predict(&[0.9, 0.8, 0.7, 0.85, 0.75]);
        let pred_b = hdc.predict(&[-0.9, -0.8, -0.7, -0.85, -0.75]);
        assert_eq!(pred_a.as_deref(), Some("A"), "pred_a={pred_a:?}");
        assert_eq!(pred_b.as_deref(), Some("B"), "pred_b={pred_b:?}");
    }

    #[test]
    fn test_online_hdc_accuracy_on_trivial_dataset() {
        let mut hdc = OnlineHdc::new(3, 10, LARGE_DIM);
        let samples = vec![
            (vec![1.0, 0.9, 0.8], "pos".to_string()),
            (vec![-1.0, -0.9, -0.8], "neg".to_string()),
        ];
        for (f, l) in &samples {
            for _ in 0..30 {
                hdc.train(f, l);
            }
        }
        let acc = hdc.accuracy(&samples);
        assert!(acc > 0.5, "acc={acc}");
    }

    // ── HdSequenceEncoder ─────────────────────────────────────────────────────

    #[test]
    fn test_sequence_encoder_order_sensitive() {
        let mut enc = HdSequenceEncoder::new(LARGE_DIM);
        let ab = enc.encode_sequence(&["a", "b"]);
        let ba = enc.encode_sequence(&["b", "a"]);
        let cos = ab.cosine_similarity(&ba);
        // Different order → different HV (low cosine).
        assert!(cos < 0.8, "cos={cos}");
    }

    #[test]
    fn test_sequence_encoder_same_sequence_same_hv() {
        let mut enc = HdSequenceEncoder::new(LARGE_DIM);
        let hv1 = enc.encode_sequence(&["x", "y", "z"]);
        let hv2 = enc.encode_sequence(&["x", "y", "z"]);
        let cos = hv1.cosine_similarity(&hv2);
        assert!((cos - 1.0).abs() < 1e-9, "cos={cos}");
    }

    #[test]
    fn test_sequence_encoder_query_position_detects_presence() {
        let mut enc = HdSequenceEncoder::new(LARGE_DIM);
        let seq_hv = enc.encode_sequence(&["cat", "sat", "mat"]);
        // Query "cat" at position 0 (first in "cat sat mat").
        let score = enc.query_position(&seq_hv, "cat", 0, 3);
        // Should have positive similarity.
        assert!(score > 0.0, "score={score}");
    }

    #[test]
    fn test_sequence_encoder_empty_sequence_no_panic() {
        let mut enc = HdSequenceEncoder::new(DIM);
        let _hv = enc.encode_sequence(&[]);
    }

    // ── HD Metrics ────────────────────────────────────────────────────────────

    #[test]
    fn test_orthogonality_test_random_hvs_near_zero() {
        let hvs: Vec<BipolarHv> = (0..10)
            .map(|i| BipolarHv::random_seeded(LARGE_DIM, i as u64 + 50))
            .collect();
        let mean_cos = orthogonality_test(&hvs);
        // 10 random HVs of dim 5000 should be close to orthogonal.
        assert!(mean_cos.abs() < 0.1, "mean_cos={mean_cos}");
    }

    #[test]
    fn test_orthogonality_test_single_hv_returns_zero() {
        let hv = BipolarHv::random_seeded(DIM, 60);
        let result = orthogonality_test(&[hv]);
        assert!((result - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_hdc_stats_separation_positive() {
        // Construct two very different class HVs.
        let mut class_hvs = HashMap::new();
        class_hvs.insert("A".to_string(), BipolarHv::random_seeded(LARGE_DIM, 70));
        class_hvs.insert("B".to_string(), BipolarHv::random_seeded(LARGE_DIM, 71));
        let stats = compute_hdc_stats(&class_hvs);
        // With random orthogonal HVs, mean_same=1, mean_diff≈0 → separation≈1.
        assert!(stats.separation > 0.5, "sep={}", stats.separation);
    }

    #[test]
    fn test_compute_hdc_stats_empty_returns_zeros() {
        let class_hvs: HashMap<String, BipolarHv> = HashMap::new();
        let stats = compute_hdc_stats(&class_hvs);
        assert!((stats.separation - 0.0).abs() < 1e-9);
    }

    // ── Miscellaneous / Integration ────────────────────────────────────────────

    #[test]
    fn test_hd_dim_constant() {
        assert_eq!(HD_DIM, 10_000);
    }

    #[test]
    fn test_real_hv_cosine_identical() {
        let hv = RealHv::random(DIM);
        assert!((hv.cosine_similarity(&hv) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_binary_to_bipolar_conversion_via_to_binary() {
        let bip = BipolarHv::random_seeded(DIM, 80);
        let bin = bip.to_binary();
        // +1 → true, -1 → false.
        for (bip_v, bin_v) in bip.data.iter().zip(bin.data.iter()) {
            let expected = *bip_v > 0.0;
            assert_eq!(*bin_v, expected);
        }
    }

    #[test]
    fn test_hdc_error_display() {
        let e = HdcError::DimensionMismatch {
            expected: 100,
            got: 200,
        };
        let s = format!("{e}");
        assert!(s.contains("100") && s.contains("200"), "msg={s}");
    }

    #[test]
    fn test_bundle_binary_empty_returns_empty() {
        let result = bundle_binary(&[]);
        assert_eq!(result.dim(), 0);
    }

    #[test]
    fn test_bundle_bipolar_empty_returns_empty() {
        let result = bundle_bipolar(&[]);
        assert_eq!(result.dim(), 0);
    }

    #[test]
    fn test_permute_binary_full_cycle_is_identity() {
        let hv = BinaryHv::random_seeded(DIM, 90);
        let cycled = permute_binary(&hv, DIM as i64);
        assert_eq!(cycled.data, hv.data);
    }

    #[test]
    fn test_permute_bipolar_full_cycle_is_identity() {
        let hv = BipolarHv::random_seeded(DIM, 91);
        let cycled = permute_bipolar(&hv, DIM as i64);
        assert_eq!(cycled.data, hv.data);
    }

    #[test]
    fn test_item_memory_add_and_lookup_multiple() {
        let mut mem = ItemMemory::new(LARGE_DIM);
        let labels = ["red", "green", "blue", "yellow", "purple"];
        let mut hvs: Vec<BipolarHv> = Vec::new();
        for label in &labels {
            hvs.push(mem.add_random(label));
        }
        for (hv, &label) in hvs.iter().zip(labels.iter()) {
            let (found, _) = mem.lookup(hv).expect("must find");
            assert_eq!(found, label, "expected {label} got {found}");
        }
    }

    #[test]
    fn test_level_encoder_clamping() {
        let enc = LevelEncoder::new(5, DIM, 0.0, 1.0);
        // Values outside range should clamp to boundary levels.
        let below = enc.encode(-5.0);
        let above = enc.encode(5.0);
        assert_eq!(below.data, enc.levels[0].data);
        assert_eq!(above.data, enc.levels[4].data);
    }

    #[test]
    fn test_online_hdc_different_samples_differ() {
        let mut hdc = OnlineHdc::new(3, 10, LARGE_DIM);
        let hv_a = hdc.encode_sample(&[1.0, 1.0, 1.0]);
        let hv_b = hdc.encode_sample(&[-1.0, -1.0, -1.0]);
        let cos = hv_a.cosine_similarity(&hv_b);
        // Opposite features should produce dissimilar HVs.
        assert!(cos < 0.5, "cos={cos}");
    }
}
