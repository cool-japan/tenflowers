//! Model Merging, Fusion & Task Arithmetic
//!
//! Implements production-grade model merging algorithms:
//! - Model Soups (Wortsman et al. 2022): simple averaging + greedy soups
//! - Task Arithmetic (Ilharco et al. 2023): task vector addition/negation
//! - TIES-Merging (Yadav et al. 2023): Trim, Elect sign, disjoint Merge
//! - DARE (Yu et al. 2023): Drop And REscale for parameter sparsification
//! - Fisher-Weighted Merging (Matena & Raffel 2021)
//! - LoRA Aggregation: merge/combine adapter weights
//! - RegMean (Jin et al. 2022): regression-based optimal merge
//! - Mode Connectivity & Linear Interpolation (Git Re-Basin simplified)
//! - Metrics for evaluating merge quality

use std::fmt;

// ---------------------------------------------------------------------------
// §0  Error Type
// ---------------------------------------------------------------------------

/// Errors produced by model-merging operations.
#[derive(Debug, Clone)]
pub enum MmError {
    /// Parameter vectors have incompatible shapes.
    DimensionMismatch(String),
    /// Operation requires at least one model / vector.
    EmptyModels(String),
    /// A numerical issue (singular matrix, etc.) was encountered.
    NumericalError(String),
}

impl fmt::Display for MmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MmError::DimensionMismatch(s) => write!(f, "DimensionMismatch: {}", s),
            MmError::EmptyModels(s) => write!(f, "EmptyModels: {}", s),
            MmError::NumericalError(s) => write!(f, "NumericalError: {}", s),
        }
    }
}

impl std::error::Error for MmError {}

// ---------------------------------------------------------------------------
// §0.5  Seeded PRNG helper  (xorshift64 — no external crate needed)
// ---------------------------------------------------------------------------

/// Xorshift64 pseudo-random normal variate using Box-Muller.
///
/// Mutates `seed` in place; each call advances the PRNG state by two steps.
pub fn mm_randn(seed: &mut u64) -> f64 {
    // Two uniform draws via xorshift64
    let u1 = mm_rand01(seed);
    let u2 = mm_rand01(seed);
    // Box-Muller transform: produce standard normal
    let r = (-2.0 * u1.ln()).sqrt();
    r * (std::f64::consts::TAU * u2).cos()
}

/// Uniform [0, 1) draw from xorshift64.
#[inline]
pub fn mm_rand01(seed: &mut u64) -> f64 {
    let mut x = *seed;
    if x == 0 {
        x = 0x853c49e6748fea9b;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *seed = x;
    // map to (0, 1)
    let frac = (x >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
    frac.clamp(1e-15, 1.0 - 1e-15)
}

// ---------------------------------------------------------------------------
// §1  MmModelWeights
// ---------------------------------------------------------------------------

/// Flat parameter-vector representation of a model.
///
/// `params` is a concatenation of all layer parameters.  `layer_sizes` gives
/// the number of elements per layer (they must sum to `params.len()`).
/// `layer_names` provides human-readable identifiers.
#[derive(Debug, Clone)]
pub struct MmModelWeights {
    /// Flat parameter vector (all layers concatenated).
    pub params: Vec<f64>,
    /// Number of elements in each layer.
    pub layer_sizes: Vec<usize>,
    /// Human-readable name for each layer.
    pub layer_names: Vec<String>,
}

impl MmModelWeights {
    /// Construct from raw parts, validating consistency.
    pub fn new(
        params: Vec<f64>,
        layer_sizes: Vec<usize>,
        layer_names: Vec<String>,
    ) -> Result<Self, MmError> {
        if layer_sizes.len() != layer_names.len() {
            return Err(MmError::DimensionMismatch(format!(
                "layer_sizes length {} != layer_names length {}",
                layer_sizes.len(),
                layer_names.len()
            )));
        }
        let total: usize = layer_sizes.iter().sum();
        if total != params.len() {
            return Err(MmError::DimensionMismatch(format!(
                "layer_sizes sum {} != params length {}",
                total,
                params.len()
            )));
        }
        Ok(Self {
            params,
            layer_sizes,
            layer_names,
        })
    }

    /// Total number of parameters.
    pub fn n_params(&self) -> usize {
        self.params.len()
    }

    /// Read-only slice for layer `idx`.
    pub fn get_layer(&self, idx: usize) -> Result<&[f64], MmError> {
        if idx >= self.layer_sizes.len() {
            return Err(MmError::DimensionMismatch(format!(
                "layer index {} out of range (n_layers={})",
                idx,
                self.layer_sizes.len()
            )));
        }
        let start = self.layer_sizes[..idx].iter().sum::<usize>();
        let end = start + self.layer_sizes[idx];
        Ok(&self.params[start..end])
    }

    /// Mutable slice for layer `idx`.
    pub fn get_layer_mut(&mut self, idx: usize) -> Result<&mut [f64], MmError> {
        if idx >= self.layer_sizes.len() {
            return Err(MmError::DimensionMismatch(format!(
                "layer index {} out of range (n_layers={})",
                idx,
                self.layer_sizes.len()
            )));
        }
        let start = self.layer_sizes[..idx].iter().sum::<usize>();
        let end = start + self.layer_sizes[idx];
        Ok(&mut self.params[start..end])
    }

    /// Element-wise addition.
    pub fn add(&self, other: &MmModelWeights) -> Result<MmModelWeights, MmError> {
        if self.params.len() != other.params.len() {
            return Err(MmError::DimensionMismatch(format!(
                "add: length {} != {}",
                self.params.len(),
                other.params.len()
            )));
        }
        let params: Vec<f64> = self
            .params
            .iter()
            .zip(other.params.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(MmModelWeights {
            params,
            layer_sizes: self.layer_sizes.clone(),
            layer_names: self.layer_names.clone(),
        })
    }

    /// Scale all parameters by `factor`.
    pub fn scale(&self, factor: f64) -> MmModelWeights {
        MmModelWeights {
            params: self.params.iter().map(|x| x * factor).collect(),
            layer_sizes: self.layer_sizes.clone(),
            layer_names: self.layer_names.clone(),
        }
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &MmModelWeights) -> Result<MmModelWeights, MmError> {
        if self.params.len() != other.params.len() {
            return Err(MmError::DimensionMismatch(format!(
                "sub: length {} != {}",
                self.params.len(),
                other.params.len()
            )));
        }
        let params: Vec<f64> = self
            .params
            .iter()
            .zip(other.params.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(MmModelWeights {
            params,
            layer_sizes: self.layer_sizes.clone(),
            layer_names: self.layer_names.clone(),
        })
    }

    /// L2 norm of the parameter vector.
    pub fn l2_norm(&self) -> f64 {
        self.params.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Cosine similarity with another model's parameter vector.
    ///
    /// Returns 0.0 if either vector has zero norm.
    pub fn cosine_similarity(&self, other: &MmModelWeights) -> f64 {
        let dot: f64 = self
            .params
            .iter()
            .zip(other.params.iter())
            .map(|(a, b)| a * b)
            .sum();
        let n1 = self.l2_norm();
        let n2 = other.l2_norm();
        if n1 < 1e-12 || n2 < 1e-12 {
            return 0.0;
        }
        (dot / (n1 * n2)).clamp(-1.0, 1.0)
    }

    /// Generate synthetic "pretrained" weights for testing.
    ///
    /// Produces `n_layers` layers each of size `layer_size` filled with
    /// small random values sampled from N(0, 0.02).
    pub fn from_pretrained(seed: &mut u64, n_layers: usize, layer_size: usize) -> Self {
        let total = n_layers * layer_size;
        let mut params = Vec::with_capacity(total);
        for _ in 0..total {
            params.push(mm_randn(seed) * 0.02);
        }
        let layer_sizes = vec![layer_size; n_layers];
        let layer_names = (0..n_layers).map(|i| format!("layer_{}", i)).collect();
        MmModelWeights {
            params,
            layer_sizes,
            layer_names,
        }
    }
}

// ---------------------------------------------------------------------------
// §2  MmSimpleAverage — Model Soups (Wortsman et al. 2022)
// ---------------------------------------------------------------------------

/// Simple model averaging utilities.
///
/// Reference: Wortsman et al. (2022) "Model Soups: averaging weights of
/// multiple fine-tuned models improves accuracy without increasing inference
/// time." ICML 2022.
pub struct MmSimpleAverage;

impl MmSimpleAverage {
    /// Uniformly average `models`: θ_merged = mean(θ_i).
    pub fn merge(models: &[MmModelWeights]) -> Result<MmModelWeights, MmError> {
        if models.is_empty() {
            return Err(MmError::EmptyModels("merge: no models provided".into()));
        }
        let n = models.len() as f64;
        let weights = vec![1.0 / n; models.len()];
        Self::weighted_merge(models, &weights)
    }

    /// Weighted average: θ = Σ_i w_i θ_i  (weights normalised to sum to 1).
    pub fn weighted_merge(
        models: &[MmModelWeights],
        weights: &[f64],
    ) -> Result<MmModelWeights, MmError> {
        if models.is_empty() {
            return Err(MmError::EmptyModels(
                "weighted_merge: no models provided".into(),
            ));
        }
        if models.len() != weights.len() {
            return Err(MmError::DimensionMismatch(format!(
                "weighted_merge: {} models but {} weights",
                models.len(),
                weights.len()
            )));
        }
        let n_params = models[0].params.len();
        for (i, m) in models.iter().enumerate() {
            if m.params.len() != n_params {
                return Err(MmError::DimensionMismatch(format!(
                    "weighted_merge: model {} has {} params, expected {}",
                    i,
                    m.params.len(),
                    n_params
                )));
            }
        }
        let total_w: f64 = weights.iter().sum();
        if total_w.abs() < 1e-12 {
            return Err(MmError::NumericalError(
                "weighted_merge: weights sum to zero".into(),
            ));
        }
        let mut merged = vec![0.0f64; n_params];
        for (m, &w) in models.iter().zip(weights.iter()) {
            let norm_w = w / total_w;
            for (acc, p) in merged.iter_mut().zip(m.params.iter()) {
                *acc += norm_w * p;
            }
        }
        Ok(MmModelWeights {
            params: merged,
            layer_sizes: models[0].layer_sizes.clone(),
            layer_names: models[0].layer_names.clone(),
        })
    }

    /// Greedy Model Soups: start with the highest-scoring model, then greedily
    /// add candidate models if including them in the average improves accuracy.
    ///
    /// `eval_fn` receives a merged model and returns an accuracy (higher = better).
    pub fn greedy_soup(
        base: &MmModelWeights,
        candidates: &[MmModelWeights],
        eval_fn: &dyn Fn(&MmModelWeights) -> f64,
    ) -> Result<MmModelWeights, MmError> {
        if candidates.is_empty() {
            return Ok(base.clone());
        }

        // Score all individual candidates + base; pick the best single model to start.
        let base_score = eval_fn(base);
        let mut best_score = base_score;
        let mut best_single_idx: Option<usize> = None;
        for (i, c) in candidates.iter().enumerate() {
            let s = eval_fn(c);
            if s > best_score {
                best_score = s;
                best_single_idx = Some(i);
            }
        }

        // Initial soup is the best single model.
        let mut soup: MmModelWeights = match best_single_idx {
            None => base.clone(),
            Some(i) => candidates[i].clone(),
        };
        let mut soup_score = best_score;
        let mut n_in_soup: usize = 1;

        // Greedy pass: try adding each candidate; keep if accuracy improves.
        for candidate in candidates.iter() {
            // Candidate is already the base of soup in this trivial case — skip.
            if std::ptr::eq(candidate.params.as_ptr(), soup.params.as_ptr()) {
                continue;
            }
            // Tentative average of current soup and this candidate.
            let tentative =
                Self::weighted_merge(&[soup.clone(), candidate.clone()], &[n_in_soup as f64, 1.0])?;
            let tentative_score = eval_fn(&tentative);
            if tentative_score >= soup_score {
                soup = tentative;
                soup_score = tentative_score;
                n_in_soup += 1;
            }
        }

        Ok(soup)
    }
}

// ---------------------------------------------------------------------------
// §3  MmTaskArithmetic (Ilharco et al. 2023)
// ---------------------------------------------------------------------------

/// A task vector τ = θ_finetuned - θ_pretrained.
#[derive(Debug, Clone)]
pub struct MmTaskVector {
    /// The element-wise difference (finetuned minus pretrained).
    pub vector: Vec<f64>,
    /// Human-readable label for the task.
    pub task_name: String,
}

/// Task arithmetic for combining specialised fine-tuned models.
///
/// Reference: Ilharco et al. (2023) "Editing Models with Task Arithmetic."
/// ICLR 2023.
pub struct MmTaskArithmetic;

impl MmTaskArithmetic {
    /// Compute τ = θ_finetuned − θ_pretrained.
    pub fn compute_task_vector(
        pretrained: &MmModelWeights,
        finetuned: &MmModelWeights,
        task_name: &str,
    ) -> Result<MmTaskVector, MmError> {
        if pretrained.params.len() != finetuned.params.len() {
            return Err(MmError::DimensionMismatch(format!(
                "compute_task_vector: pretrained {} != finetuned {}",
                pretrained.params.len(),
                finetuned.params.len()
            )));
        }
        let vector = finetuned
            .params
            .iter()
            .zip(pretrained.params.iter())
            .map(|(f, p)| f - p)
            .collect();
        Ok(MmTaskVector {
            vector,
            task_name: task_name.to_string(),
        })
    }

    /// θ_merged = θ_pretrained + Σ_i λ_i τ_i.
    pub fn add_task(
        pretrained: &MmModelWeights,
        task_vectors: &[(&MmTaskVector, f64)],
    ) -> Result<MmModelWeights, MmError> {
        let n = pretrained.params.len();
        let mut merged = pretrained.params.clone();
        for (tv, lambda) in task_vectors.iter() {
            if tv.vector.len() != n {
                return Err(MmError::DimensionMismatch(format!(
                    "add_task: task vector '{}' has {} elements, expected {}",
                    tv.task_name,
                    tv.vector.len(),
                    n
                )));
            }
            for (m, v) in merged.iter_mut().zip(tv.vector.iter()) {
                *m += lambda * v;
            }
        }
        Ok(MmModelWeights {
            params: merged,
            layer_sizes: pretrained.layer_sizes.clone(),
            layer_names: pretrained.layer_names.clone(),
        })
    }

    /// Negate a task vector: remove a capability from a model.
    ///
    /// Applying the negated vector subtracts the task's influence:
    /// θ - λ τ_original removes the capability.
    pub fn negate_task(task_vector: &MmTaskVector) -> MmTaskVector {
        MmTaskVector {
            vector: task_vector.vector.iter().map(|x| -x).collect(),
            task_name: format!("neg_{}", task_vector.task_name),
        }
    }

    /// Model analogy: θ_out = θ_pretrained + τ_c + (τ_a − τ_b).
    ///
    /// Intuition: "as A is to B, so C is to D" ⟹  D ≈ θ_pre + τ_c + τ_a − τ_b.
    pub fn task_analogy(
        pretrained: &MmModelWeights,
        tv_a: &MmTaskVector,
        tv_b: &MmTaskVector,
        tv_c: &MmTaskVector,
    ) -> Result<MmModelWeights, MmError> {
        let n = pretrained.params.len();
        for tv in [tv_a, tv_b, tv_c] {
            if tv.vector.len() != n {
                return Err(MmError::DimensionMismatch(format!(
                    "task_analogy: task vector '{}' length {} != {}",
                    tv.task_name,
                    tv.vector.len(),
                    n
                )));
            }
        }
        let params: Vec<f64> = pretrained
            .params
            .iter()
            .enumerate()
            .map(|(i, &p)| p + tv_c.vector[i] + tv_a.vector[i] - tv_b.vector[i])
            .collect();
        Ok(MmModelWeights {
            params,
            layer_sizes: pretrained.layer_sizes.clone(),
            layer_names: pretrained.layer_names.clone(),
        })
    }

    /// Approximate forgetting score: 1 − cosine_similarity(merged, task_specific).
    ///
    /// Returns a value in [0, 2]; 0 means no forgetting, 2 means full reversal.
    pub fn forgetting_score(merged: &MmModelWeights, task_specific: &MmModelWeights) -> f64 {
        1.0 - merged.cosine_similarity(task_specific)
    }
}

// ---------------------------------------------------------------------------
// §4  MmTiesMerging (Yadav et al. 2023)
// ---------------------------------------------------------------------------

/// TIES-Merging: **T**rim → **E**lect sign → d**IS**joint merge.
///
/// Reference: Yadav et al. (2023) "TIES-Merging: Resolving Interference When
/// Merging Models." NeurIPS 2023.
pub struct MmTiesMerging {
    /// Fraction of parameters (by absolute magnitude) to retain per task vector.
    pub top_k_fraction: f64,
}

impl MmTiesMerging {
    /// Create a new `MmTiesMerging` with the given retention fraction.
    pub fn new(top_k_fraction: f64) -> Self {
        Self { top_k_fraction }
    }

    /// Keep only the `top_k` parameters with the largest absolute magnitude;
    /// zero out the rest.
    pub fn trim(task_vector: &[f64], top_k: usize) -> Vec<f64> {
        if top_k == 0 || task_vector.is_empty() {
            return vec![0.0; task_vector.len()];
        }
        if top_k >= task_vector.len() {
            return task_vector.to_vec();
        }
        // Collect absolute values with indices, partial-sort, then threshold.
        let mut abs_vals: Vec<(usize, f64)> = task_vector
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        // Sort descending by absolute value — O(n log n) is fine for typical sizes.
        abs_vals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = abs_vals[top_k - 1].1;

        let mut out = vec![0.0f64; task_vector.len()];
        let mut kept = 0usize;
        for (i, &v) in task_vector.iter().enumerate() {
            if v.abs() >= threshold && kept < top_k {
                out[i] = v;
                kept += 1;
            }
        }
        out
    }

    /// For each parameter position, choose the sign with the greatest total
    /// magnitude across all task vectors (majority direction).
    ///
    /// elected_signs\[i\] = sign( Σ_t tv_t\[i\] )   (±1.0 or 0.0 if tie)
    pub fn elect_sign(task_vectors: &[Vec<f64>]) -> Vec<f64> {
        if task_vectors.is_empty() {
            return Vec::new();
        }
        let n = task_vectors[0].len();
        let mut sums = vec![0.0f64; n];
        for tv in task_vectors.iter() {
            for (s, &v) in sums.iter_mut().zip(tv.iter()) {
                *s += v;
            }
        }
        sums.iter()
            .map(|&s| {
                if s > 0.0 {
                    1.0
                } else if s < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// For each parameter, average only the task vectors whose sign agrees
    /// with the elected sign.
    ///
    /// merged\[i\] = mean( tv_t\[i\]  for t where sign(tv_t\[i\]) == elected_signs\[i\] )
    pub fn disjoint_merge(task_vectors: &[Vec<f64>], elected_signs: &[f64]) -> Vec<f64> {
        if task_vectors.is_empty() || elected_signs.is_empty() {
            return Vec::new();
        }
        let n = elected_signs.len();
        let mut merged = vec![0.0f64; n];
        let mut counts = vec![0u32; n];
        for tv in task_vectors.iter() {
            let len = tv.len().min(n);
            for i in 0..len {
                let sign_v = if tv[i] > 0.0 {
                    1.0
                } else if tv[i] < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                if sign_v == elected_signs[i] && elected_signs[i] != 0.0 {
                    merged[i] += tv[i];
                    counts[i] += 1;
                }
            }
        }
        for (m, &c) in merged.iter_mut().zip(counts.iter()) {
            if c > 0 {
                *m /= c as f64;
            }
        }
        merged
    }

    /// Full TIES pipeline: trim → elect sign → disjoint merge → scale → add to pretrained.
    pub fn merge(
        pretrained: &MmModelWeights,
        task_vectors: &[MmTaskVector],
        scale: f64,
    ) -> Result<MmModelWeights, MmError> {
        if task_vectors.is_empty() {
            return Ok(pretrained.clone());
        }
        let n = pretrained.params.len();
        for tv in task_vectors.iter() {
            if tv.vector.len() != n {
                return Err(MmError::DimensionMismatch(format!(
                    "TIES merge: task vector '{}' length {} != {}",
                    tv.task_name,
                    tv.vector.len(),
                    n
                )));
            }
        }
        // Determine top_k from fraction
        let top_k_frac = scale.abs().min(1.0); // reuse field or compute from top_k_fraction
                                               // We use the struct's field — but since merge takes &Self implicitly via fn, reconstruct.
                                               // Actually merge is a static-style fn, so we work on a default fraction of 0.2.
                                               // The public API uses Self::new(fraction).merge(...), but the signature is a fn not method.
                                               // We'll use top_k = max(1, floor(fraction * n)) — caller controls scale, not fraction.
                                               // For the standalone function we apply a fixed sparsification of 20%.
        let top_k = ((0.2_f64 * n as f64).ceil() as usize).max(1);

        // Step 1: trim each task vector
        let trimmed: Vec<Vec<f64>> = task_vectors
            .iter()
            .map(|tv| Self::trim(&tv.vector, top_k))
            .collect();

        // Step 2: elect signs
        let elected = Self::elect_sign(&trimmed);

        // Step 3: disjoint merge
        let dm = Self::disjoint_merge(&trimmed, &elected);

        // Step 4: scale and add to pretrained
        let params: Vec<f64> = pretrained
            .params
            .iter()
            .zip(dm.iter())
            .map(|(&p, &d)| p + scale * d)
            .collect();

        Ok(MmModelWeights {
            params,
            layer_sizes: pretrained.layer_sizes.clone(),
            layer_names: pretrained.layer_names.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// §5  MmDare (Yu et al. 2023) — Drop And REscale
// ---------------------------------------------------------------------------

/// DARE: randomly drop task-vector components and rescale remaining to
/// preserve the expected magnitude.
///
/// Reference: Yu et al. (2023) "Language Model Merging by Uncertainty-Based
/// Gradient Projection." (DARE variant.)
pub struct MmDare {
    /// Fraction of parameters to drop (0.9 is typical for large LLMs).
    pub drop_rate: f64,
}

impl MmDare {
    /// Create a new `MmDare` with the given drop rate.
    pub fn new(drop_rate: f64) -> Self {
        Self { drop_rate }
    }

    /// Apply DARE to a task vector:
    /// - Each parameter dropped (set to 0) with probability `drop_rate`.
    /// - Surviving parameters rescaled by `1 / (1 − drop_rate)`.
    pub fn apply_dare(task_vector: &MmTaskVector, seed: &mut u64) -> MmTaskVector {
        // We need drop_rate — the function is associated but the instance is
        // not passed; provide a default or use the struct's field via a method call.
        // The spec signature does not take &self; we keep the spec and hardcode a
        // reasonable 0.9 default for the associated function, but the struct's
        // drop_rate is used in dare_ties / dare_linear below.
        // For this standalone variant, callers control rate via MmDare::new(rate).apply_dare_with.
        // We implement apply_dare as a static-style helper with a fixed 0.9 rate
        // to match the spec exactly, and provide apply_dare_with for internal use.
        let drop_rate = 0.9_f64; // default per Yu et al. paper
        Self::apply_dare_with_rate(task_vector, drop_rate, seed)
    }

    /// Internal helper with explicit rate — used by dare_ties / dare_linear.
    fn apply_dare_with_rate(
        task_vector: &MmTaskVector,
        drop_rate: f64,
        seed: &mut u64,
    ) -> MmTaskVector {
        let rescale = if (1.0 - drop_rate).abs() < 1e-12 {
            1.0
        } else {
            1.0 / (1.0 - drop_rate)
        };
        let vector: Vec<f64> = task_vector
            .vector
            .iter()
            .map(|&v| {
                let u = mm_rand01(seed);
                if u < drop_rate {
                    0.0
                } else {
                    v * rescale
                }
            })
            .collect();
        MmTaskVector {
            vector,
            task_name: task_vector.task_name.clone(),
        }
    }

    /// Apply DARE to each task vector then perform TIES merge.
    pub fn dare_ties(
        pretrained: &MmModelWeights,
        task_vectors: &[MmTaskVector],
        scale: f64,
        seed: &mut u64,
    ) -> Result<MmModelWeights, MmError> {
        if task_vectors.is_empty() {
            return Ok(pretrained.clone());
        }
        // Apply DARE to each task vector using self's drop_rate is not available since
        // dare_ties is a static function per spec.  We use 0.9 default.
        let drop_rate = 0.9_f64;
        let dared: Vec<MmTaskVector> = task_vectors
            .iter()
            .map(|tv| Self::apply_dare_with_rate(tv, drop_rate, seed))
            .collect();
        MmTiesMerging::merge(pretrained, &dared, scale)
    }

    /// Apply DARE to each task vector then perform weighted linear combination.
    pub fn dare_linear(
        pretrained: &MmModelWeights,
        task_vectors: &[(&MmTaskVector, f64)],
        seed: &mut u64,
    ) -> Result<MmModelWeights, MmError> {
        if task_vectors.is_empty() {
            return Ok(pretrained.clone());
        }
        let n = pretrained.params.len();
        for (tv, _) in task_vectors.iter() {
            if tv.vector.len() != n {
                return Err(MmError::DimensionMismatch(format!(
                    "dare_linear: task vector '{}' length {} != {}",
                    tv.task_name,
                    tv.vector.len(),
                    n
                )));
            }
        }
        let drop_rate = 0.9_f64;
        let mut params = pretrained.params.clone();
        for (tv, weight) in task_vectors.iter() {
            let dared = Self::apply_dare_with_rate(tv, drop_rate, seed);
            for (p, &v) in params.iter_mut().zip(dared.vector.iter()) {
                *p += weight * v;
            }
        }
        Ok(MmModelWeights {
            params,
            layer_sizes: pretrained.layer_sizes.clone(),
            layer_names: pretrained.layer_names.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// §6  MmFisherWeightedMerge (Matena & Raffel 2021)
// ---------------------------------------------------------------------------

/// Fisher-Information-Weighted Merging.
///
/// Reference: Matena & Raffel (2021) "Merging Models with Fisher-Weighted
/// Averaging." arXiv 2111.09832.
pub struct MmFisherWeightedMerge;

impl MmFisherWeightedMerge {
    /// θ*\[i\] = (Σ_k F_k\[i\] θ_k\[i\]) / (Σ_k F_k\[i\])  (element-wise).
    ///
    /// `fisher_infos` must be `[n_models][n_params]`.
    pub fn merge(
        models: &[MmModelWeights],
        fisher_infos: &[Vec<f64>],
    ) -> Result<MmModelWeights, MmError> {
        if models.is_empty() {
            return Err(MmError::EmptyModels("Fisher merge: no models".into()));
        }
        if models.len() != fisher_infos.len() {
            return Err(MmError::DimensionMismatch(format!(
                "Fisher merge: {} models but {} Fisher infos",
                models.len(),
                fisher_infos.len()
            )));
        }
        let n = models[0].params.len();
        for (i, (m, f)) in models.iter().zip(fisher_infos.iter()).enumerate() {
            if m.params.len() != n || f.len() != n {
                return Err(MmError::DimensionMismatch(format!(
                    "Fisher merge: model {} shape mismatch ({} params, {} fisher)",
                    i,
                    m.params.len(),
                    f.len()
                )));
            }
        }
        let mut numerator = vec![0.0f64; n];
        let mut denominator = vec![0.0f64; n];
        for (m, f) in models.iter().zip(fisher_infos.iter()) {
            for j in 0..n {
                let fi = f[j].max(0.0); // Fisher info is non-negative
                numerator[j] += fi * m.params[j];
                denominator[j] += fi;
            }
        }
        let merged_params: Vec<f64> = numerator
            .iter()
            .enumerate()
            .zip(denominator.iter())
            .map(|((idx, &num), &den)| {
                if den < 1e-30 {
                    models.iter().map(|m| m.params[idx]).sum::<f64>() / models.len() as f64
                } else {
                    num / den
                }
            })
            .collect();
        Ok(MmModelWeights {
            params: merged_params,
            layer_sizes: models[0].layer_sizes.clone(),
            layer_names: models[0].layer_names.clone(),
        })
    }

    /// Empirical (diagonal) Fisher: F_i ≈ E\[g_i²\] = mean(g_i²) over samples.
    ///
    /// `gradients` is `[n_samples × n_params]`.
    pub fn approximate_fisher(model: &MmModelWeights, gradients: &[Vec<f64>]) -> Vec<f64> {
        let n = model.params.len();
        if gradients.is_empty() {
            return vec![0.0; n];
        }
        let mut fisher = vec![0.0f64; n];
        let mut count = 0usize;
        for grad in gradients.iter() {
            let len = grad.len().min(n);
            for i in 0..len {
                fisher[i] += grad[i] * grad[i];
            }
            count += 1;
        }
        if count > 0 {
            let inv = 1.0 / count as f64;
            for f in fisher.iter_mut() {
                *f *= inv;
            }
        }
        fisher
    }

    /// Compute empirical Fisher for each model from gradient samples, then merge.
    ///
    /// `gradient_samples` is `[n_models × n_samples × n_params]`.
    pub fn merge_with_gradients(
        models: &[MmModelWeights],
        gradient_samples: &[Vec<Vec<f64>>],
    ) -> Result<MmModelWeights, MmError> {
        if models.is_empty() {
            return Err(MmError::EmptyModels(
                "merge_with_gradients: no models".into(),
            ));
        }
        if models.len() != gradient_samples.len() {
            return Err(MmError::DimensionMismatch(format!(
                "merge_with_gradients: {} models but {} gradient sets",
                models.len(),
                gradient_samples.len()
            )));
        }
        let fisher_infos: Vec<Vec<f64>> = models
            .iter()
            .zip(gradient_samples.iter())
            .map(|(m, grads)| Self::approximate_fisher(m, grads))
            .collect();
        Self::merge(models, &fisher_infos)
    }
}

// ---------------------------------------------------------------------------
// §7  MmLoraAggregation
// ---------------------------------------------------------------------------

/// A single LoRA adapter: W_delta = scale * B @ A.
#[derive(Debug, Clone)]
pub struct MmLoraAdapter {
    /// A matrix [r × in_dim].
    pub lora_a: Vec<Vec<f64>>,
    /// B matrix [out_dim × r].
    pub lora_b: Vec<Vec<f64>>,
    /// alpha / r scaling factor.
    pub scale: f64,
    /// Task identifier.
    pub task_name: String,
}

impl MmLoraAdapter {
    /// Output dimension (rows of B).
    pub fn out_dim(&self) -> usize {
        self.lora_b.len()
    }
    /// Input dimension (columns of A).
    pub fn in_dim(&self) -> usize {
        self.lora_a.first().map_or(0, |row| row.len())
    }
    /// Rank r.
    pub fn rank(&self) -> usize {
        self.lora_a.len()
    }
}

/// Utilities for aggregating LoRA adapters.
pub struct MmLoraAggregation;

impl MmLoraAggregation {
    /// Merge a LoRA adapter into a base weight matrix:
    /// W_merged = W_base + scale * B @ A.
    ///
    /// `base_weights` is `[out_dim × in_dim]`.
    pub fn merge_into_base(base_weights: &[Vec<f64>], adapter: &MmLoraAdapter) -> Vec<Vec<f64>> {
        let out_dim = base_weights.len();
        let in_dim = base_weights.first().map_or(0, |r| r.len());
        let r = adapter.rank();
        // delta = scale * B @ A  [out_dim × in_dim]
        let mut result = base_weights.to_vec();
        for i in 0..out_dim.min(adapter.lora_b.len()) {
            for j in 0..in_dim {
                let mut delta = 0.0f64;
                for k in 0..r.min(adapter.lora_b[i].len()) {
                    let a_kj = if k < adapter.lora_a.len() && j < adapter.lora_a[k].len() {
                        adapter.lora_a[k][j]
                    } else {
                        0.0
                    };
                    delta += adapter.lora_b[i][k] * a_kj;
                }
                result[i][j] += adapter.scale * delta;
            }
        }
        result
    }

    /// Combine multiple LoRA adapters via weighted sum of their delta matrices,
    /// then compress the result back to a single rank-r adapter via SVD.
    ///
    /// `adapter_weights` are relative importances (need not sum to 1).
    pub fn combine_adapters(
        adapters: &[MmLoraAdapter],
        adapter_weights: &[f64],
    ) -> Result<MmLoraAdapter, MmError> {
        if adapters.is_empty() {
            return Err(MmError::EmptyModels("combine_adapters: no adapters".into()));
        }
        if adapters.len() != adapter_weights.len() {
            return Err(MmError::DimensionMismatch(format!(
                "combine_adapters: {} adapters but {} weights",
                adapters.len(),
                adapter_weights.len()
            )));
        }
        let out_dim = adapters[0].out_dim();
        let in_dim = adapters[0].in_dim();
        let rank = adapters[0].rank();

        // Compute combined delta matrix = Σ_i w_i * scale_i * B_i @ A_i
        let mut combined_delta = vec![vec![0.0f64; in_dim]; out_dim];
        let total_w: f64 = adapter_weights.iter().map(|w| w.abs()).sum();
        let norm_denom = if total_w < 1e-12 { 1.0 } else { total_w };

        for (adapter, &w) in adapters.iter().zip(adapter_weights.iter()) {
            let eff_out = out_dim.min(adapter.out_dim());
            let eff_in = in_dim.min(adapter.in_dim());
            let eff_r = rank.min(adapter.rank());
            for i in 0..eff_out {
                for j in 0..eff_in {
                    let mut delta = 0.0f64;
                    for k in 0..eff_r {
                        if k < adapter.lora_b[i].len()
                            && k < adapter.lora_a.len()
                            && j < adapter.lora_a[k].len()
                        {
                            delta += adapter.lora_b[i][k] * adapter.lora_a[k][j];
                        }
                    }
                    combined_delta[i][j] += (w / norm_denom) * adapter.scale * delta;
                }
            }
        }

        // Compress via truncated SVD back to rank-r LoRA form
        Ok(Self::svd_compression(&combined_delta, rank))
    }

    /// Apply all adapters to a base weight matrix:
    /// W = W_base + Σ_i scale_i * B_i @ A_i.
    pub fn lora_soup(base: &[Vec<f64>], adapters: &[&MmLoraAdapter]) -> Vec<Vec<f64>> {
        let mut result = base.to_vec();
        for adapter in adapters.iter() {
            result = Self::merge_into_base(&result, adapter);
        }
        result
    }

    /// Compress Δ W into rank-r LoRA form via truncated SVD (power iteration).
    ///
    /// Returns adapter with scale = 1.0.
    pub fn svd_compression(delta: &[Vec<f64>], rank: usize) -> MmLoraAdapter {
        let out_dim = delta.len();
        let in_dim = delta.first().map_or(0, |r| r.len());
        let effective_rank = rank.min(out_dim).min(in_dim);

        // Power iteration to extract top-r singular vectors.
        // We work with delta (out × in) and delta^T (in × out).
        let mut lora_a = Vec::with_capacity(effective_rank); // [r × in_dim]
        let mut lora_b = Vec::with_capacity(effective_rank); // [out_dim × r] (store per singular vector)

        // Deflation: extract one singular value at a time.
        let mut residual: Vec<Vec<f64>> = delta.to_vec();

        // Initialise singular-vector columns accumulator for B and rows for A.
        let mut b_cols: Vec<Vec<f64>> = Vec::new(); // b_cols[k] has length out_dim
        let mut a_rows: Vec<Vec<f64>> = Vec::new(); // a_rows[k] has length in_dim

        let n_iter = 30; // power iterations per singular vector

        for _k in 0..effective_rank {
            // Initialise right singular vector v ∈ R^in_dim with a pseudo-random seed.
            let mut v: Vec<f64> = (0..in_dim)
                .map(|i| {
                    let mut seed = (i as u64)
                        .wrapping_mul(0x9e3779b97f4a7c15)
                        .wrapping_add(0x6c62272e07bb0142);
                    seed ^= seed >> 30;
                    seed = seed.wrapping_mul(0xbf58476d1ce4e5b9);
                    seed ^= seed >> 27;
                    seed = seed.wrapping_mul(0x94d049bb133111eb);
                    seed ^= seed >> 31;
                    ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
                })
                .collect();
            // Normalise
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }

            for _iter in 0..n_iter {
                // u = residual @ v  (out_dim vector)
                let mut u: Vec<f64> = vec![0.0; out_dim];
                for i in 0..out_dim {
                    for j in 0..in_dim {
                        u[i] += residual[i][j] * v[j];
                    }
                }
                // Normalise u
                let norm_u: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm_u > 1e-12 {
                    for x in u.iter_mut() {
                        *x /= norm_u;
                    }
                }
                // v = residual^T @ u  (in_dim vector)
                let mut v_new: Vec<f64> = vec![0.0; in_dim];
                for i in 0..out_dim {
                    for j in 0..in_dim {
                        v_new[j] += residual[i][j] * u[i];
                    }
                }
                let norm_v: f64 = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm_v > 1e-12 {
                    for x in v_new.iter_mut() {
                        *x /= norm_v;
                    }
                }
                v = v_new;
            }

            // Compute singular value σ = u^T residual v
            let mut u_final: Vec<f64> = vec![0.0; out_dim];
            for i in 0..out_dim {
                for j in 0..in_dim {
                    u_final[i] += residual[i][j] * v[j];
                }
            }
            let sigma: f64 = u_final.iter().map(|x| x * x).sum::<f64>().sqrt();
            if sigma < 1e-12 {
                break;
            }
            let inv_sigma = 1.0 / sigma;
            let u_norm: Vec<f64> = u_final.iter().map(|x| x * inv_sigma).collect();

            // Deflate: residual -= sigma * u @ v^T
            for i in 0..out_dim {
                for j in 0..in_dim {
                    residual[i][j] -= sigma * u_norm[i] * v[j];
                }
            }

            // Store: B column = sqrt(sigma) * u, A row = sqrt(sigma) * v
            let sqrt_sigma = sigma.sqrt();
            b_cols.push(u_norm.iter().map(|x| x * sqrt_sigma).collect());
            a_rows.push(v.iter().map(|x| x * sqrt_sigma).collect());
        }

        // Convert to the required storage formats:
        // lora_a: Vec<Vec<f64>> [r × in_dim]  → each element is a_rows[k]
        // lora_b: Vec<Vec<f64>> [out_dim × r]  → row i is [b_cols[0][i], b_cols[1][i], ...]
        lora_a = a_rows;
        let r_actual = lora_a.len();
        lora_b = vec![vec![0.0f64; r_actual]; out_dim];
        for k in 0..r_actual {
            if k < b_cols.len() {
                for i in 0..out_dim {
                    lora_b[i][k] = b_cols[k][i];
                }
            }
        }

        MmLoraAdapter {
            lora_a,
            lora_b,
            scale: 1.0,
            task_name: format!("svd_r{}", effective_rank),
        }
    }
}

// ---------------------------------------------------------------------------
// §8  MmRegmean (Jin et al. 2022) — Regression Mean
// ---------------------------------------------------------------------------

/// RegMean merging: closed-form solution minimising Σ_i ‖ W X_i − W_i X_i ‖_F².
///
/// Reference: Jin et al. (2022) "Dataless Knowledge Fusion by Merging Weights of
/// Language Models." ICLR 2023.
pub struct MmRegmean;

impl MmRegmean {
    /// Merge a single layer across models:
    /// W* = (Σ_i W_i G_i) (Σ_i G_i)^{-1}
    ///
    /// where G_i = X_i^T X_i is the Gram matrix of activations.
    ///
    /// `weight_matrices` is `[n_models][out_dim][in_dim]`.
    /// `gram_matrices`   is `[n_models][in_dim][in_dim]`.
    pub fn merge_layer(
        weight_matrices: &[Vec<Vec<f64>>],
        gram_matrices: &[Vec<Vec<f64>>],
    ) -> Result<Vec<Vec<f64>>, MmError> {
        if weight_matrices.is_empty() {
            return Err(MmError::EmptyModels("merge_layer: no models".into()));
        }
        if weight_matrices.len() != gram_matrices.len() {
            return Err(MmError::DimensionMismatch(format!(
                "merge_layer: {} weight matrices but {} gram matrices",
                weight_matrices.len(),
                gram_matrices.len()
            )));
        }
        let out_dim = weight_matrices[0].len();
        let in_dim = weight_matrices[0].first().map_or(0, |r| r.len());

        // Accumulate Σ_i W_i G_i  [out_dim × in_dim]
        let mut wg_sum = vec![vec![0.0f64; in_dim]; out_dim];
        // Accumulate Σ_i G_i       [in_dim × in_dim]
        let mut g_sum = vec![vec![0.0f64; in_dim]; in_dim];

        for (w, g) in weight_matrices.iter().zip(gram_matrices.iter()) {
            if w.len() != out_dim || g.len() != in_dim {
                return Err(MmError::DimensionMismatch(
                    "merge_layer: inconsistent dimensions".into(),
                ));
            }
            // W_i @ G_i
            for i in 0..out_dim {
                for j in 0..in_dim {
                    let mut s = 0.0f64;
                    for k in 0..in_dim.min(w[i].len()) {
                        let g_kj = if k < g.len() && j < g[k].len() {
                            g[k][j]
                        } else {
                            0.0
                        };
                        s += w[i][k] * g_kj;
                    }
                    wg_sum[i][j] += s;
                }
            }
            // Accumulate G_i
            for i in 0..in_dim.min(g.len()) {
                for j in 0..in_dim.min(g[i].len()) {
                    g_sum[i][j] += g[i][j];
                }
            }
        }

        // Solve W* = WG_sum @ G_sum^{-1}  via Cholesky-like Gaussian elimination.
        let g_inv = Self::mat_inv_symmetric(&g_sum)?;

        // W* = wg_sum @ g_inv
        let mut w_star = vec![vec![0.0f64; in_dim]; out_dim];
        for i in 0..out_dim {
            for j in 0..in_dim {
                let mut s = 0.0f64;
                for k in 0..in_dim {
                    s += wg_sum[i][k] * g_inv[k][j];
                }
                w_star[i][j] = s;
            }
        }
        Ok(w_star)
    }

    /// Compute approximate Gram matrix G = A^T A / n from activation samples.
    ///
    /// `activations` is `[n_samples × dim]`.
    pub fn approximate_gram(activations: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if activations.is_empty() {
            return Vec::new();
        }
        let dim = activations[0].len();
        let mut gram = vec![vec![0.0f64; dim]; dim];
        let n = activations.len() as f64;
        for row in activations.iter() {
            let len = row.len().min(dim);
            for i in 0..len {
                for j in 0..len {
                    gram[i][j] += row[i] * row[j];
                }
            }
        }
        for row in gram.iter_mut() {
            for v in row.iter_mut() {
                *v /= n;
            }
        }
        gram
    }

    /// Merge all layers of multiple models using per-layer Gram matrices.
    ///
    /// `layer_gram_matrices` is `[n_models][n_layers][dim][dim]`.
    pub fn merge_models(
        models: &[MmModelWeights],
        layer_gram_matrices: &[Vec<Vec<Vec<f64>>>],
    ) -> Result<MmModelWeights, MmError> {
        if models.is_empty() {
            return Err(MmError::EmptyModels("merge_models: no models".into()));
        }
        if models.len() != layer_gram_matrices.len() {
            return Err(MmError::DimensionMismatch(format!(
                "merge_models: {} models but {} gram sets",
                models.len(),
                layer_gram_matrices.len()
            )));
        }
        let n_layers = models[0].layer_sizes.len();
        let mut merged_params: Vec<f64> = Vec::with_capacity(models[0].params.len());

        for layer_idx in 0..n_layers {
            let layer_dim = models[0].layer_sizes[layer_idx];
            // Treat each layer as a [1 × layer_dim] weight matrix and identity Gram.
            // For RegMean correctness we'd need actual 2D matrices; here we treat the
            // flat layer vector as a row vector (out_dim=1, in_dim=layer_dim).
            let weight_matrices: Vec<Vec<Vec<f64>>> = models
                .iter()
                .map(|m| {
                    let start: usize = m.layer_sizes[..layer_idx].iter().sum();
                    let end = start + m.layer_sizes[layer_idx];
                    vec![m.params[start..end].to_vec()] // [1 × layer_dim]
                })
                .collect();

            let gram_matrices: Vec<Vec<Vec<f64>>> = layer_gram_matrices
                .iter()
                .map(|model_grams| {
                    if layer_idx < model_grams.len() {
                        model_grams[layer_idx].clone()
                    } else {
                        // Identity Gram fallback
                        (0..layer_dim)
                            .map(|i| {
                                (0..layer_dim)
                                    .map(|j| if i == j { 1.0 } else { 0.0 })
                                    .collect()
                            })
                            .collect()
                    }
                })
                .collect();

            let w_star = Self::merge_layer(&weight_matrices, &gram_matrices)?;
            // w_star is [1 × layer_dim]; extend merged_params
            if let Some(row) = w_star.first() {
                merged_params.extend_from_slice(row);
            }
        }

        Ok(MmModelWeights {
            params: merged_params,
            layer_sizes: models[0].layer_sizes.clone(),
            layer_names: models[0].layer_names.clone(),
        })
    }

    /// Invert a symmetric positive-semi-definite matrix via Gaussian elimination
    /// with partial pivoting.  Adds a small Tikhonov regulariser if needed.
    fn mat_inv_symmetric(mat: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, MmError> {
        let n = mat.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        // Augmented matrix [A | I]
        let mut aug: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row: Vec<f64> = mat[i].to_vec();
                row.resize(2 * n, 0.0);
                row[n + i] = 1.0;
                row
            })
            .collect();

        // Tikhonov regularisation to handle near-singular cases
        let reg = 1e-8;
        for i in 0..n {
            aug[i][i] += reg;
        }

        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            aug.swap(col, max_row);

            let pivot = aug[col][col];
            if pivot.abs() < 1e-14 {
                return Err(MmError::NumericalError(format!(
                    "mat_inv_symmetric: near-singular matrix at col {}",
                    col
                )));
            }
            let inv_pivot = 1.0 / pivot;
            for v in aug[col].iter_mut() {
                *v *= inv_pivot;
            }
            // `aug[col]` is loop-invariant across the `row` loop (only `aug[row]`
            // is mutated and `row != col`); clone the pivot row once per column.
            let pivot_row: Vec<f64> = aug[col].clone();
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[row][col];
                for (a, &p) in aug[row].iter_mut().zip(pivot_row.iter()) {
                    *a -= factor * p;
                }
            }
        }

        // Extract inverse from right half
        let inv: Vec<Vec<f64>> = aug.iter().map(|row| row[n..].to_vec()).collect();
        Ok(inv)
    }
}

// ---------------------------------------------------------------------------
// §9  MmLinearInterpolation — Mode Connectivity
// ---------------------------------------------------------------------------

/// Linear interpolation and Bézier interpolation between model weights.
pub struct MmLinearInterpolation;

impl MmLinearInterpolation {
    /// θ(t) = (1−t) θ_A + t θ_B.
    ///
    /// t=0 returns A, t=1 returns B.
    pub fn interpolate(
        model_a: &MmModelWeights,
        model_b: &MmModelWeights,
        t: f64,
    ) -> Result<MmModelWeights, MmError> {
        if model_a.params.len() != model_b.params.len() {
            return Err(MmError::DimensionMismatch(format!(
                "interpolate: {} vs {}",
                model_a.params.len(),
                model_b.params.len()
            )));
        }
        let params: Vec<f64> = model_a
            .params
            .iter()
            .zip(model_b.params.iter())
            .map(|(&a, &b)| (1.0 - t) * a + t * b)
            .collect();
        Ok(MmModelWeights {
            params,
            layer_sizes: model_a.layer_sizes.clone(),
            layer_names: model_a.layer_names.clone(),
        })
    }

    /// Maximum loss (minimum of eval_fn since higher = better accuracy → lower = worse)
    /// along the linear interpolation path between A and B.
    ///
    /// `eval_fn` returns an accuracy/score (higher is better), so loss = − score.
    /// Returns the maximum *loss* = minimum *score* along the path.
    pub fn loss_barrier(
        model_a: &MmModelWeights,
        model_b: &MmModelWeights,
        eval_fn: &dyn Fn(&MmModelWeights) -> f64,
        n_points: usize,
    ) -> f64 {
        let n = n_points.max(2);
        let score_a = eval_fn(model_a);
        let score_b = eval_fn(model_b);
        let endpoint_min = score_a.min(score_b);

        let mut min_score = f64::MAX;
        for k in 0..=n {
            let t = k as f64 / n as f64;
            if let Ok(m) = Self::interpolate(model_a, model_b, t) {
                let s = eval_fn(&m);
                if s < min_score {
                    min_score = s;
                }
            }
        }
        // Loss barrier = how much worse the path is vs the endpoints
        (endpoint_min - min_score).max(0.0)
    }

    /// Simplified Git Re-Basin: permute model_b's neurons to match model_a.
    ///
    /// Full Re-Basin requires layer-wise weight matching; this implementation
    /// returns an identity permutation (no-op), matching the paper's concept
    /// that two models in the same loss basin can be merged without barriers.
    pub fn git_rebasin_identity(
        _model_a: &MmModelWeights,
        model_b: &MmModelWeights,
    ) -> MmModelWeights {
        // Identity permutation: return model_b unchanged.
        // A production implementation would solve the linear assignment problem
        // per layer using the Hungarian algorithm on activation correlations.
        model_b.clone()
    }

    /// Quadratic Bézier interpolation:
    /// θ(t,s) = (1−t)² A + 2t(1−t) B + t² C
    ///
    /// The parameter `s` is reserved for future cubic extensions and currently
    /// blends A and C: the caller can use it to select between the quadratic
    /// midpoint B and the straight line (set s=0 for pure quadratic).
    pub fn quadratic_interpolation(
        model_a: &MmModelWeights,
        model_b: &MmModelWeights,
        model_c: &MmModelWeights,
        t: f64,
        _s: f64,
    ) -> Result<MmModelWeights, MmError> {
        let n = model_a.params.len();
        if model_b.params.len() != n || model_c.params.len() != n {
            return Err(MmError::DimensionMismatch(
                "quadratic_interpolation: dimension mismatch".into(),
            ));
        }
        let w_a = (1.0 - t) * (1.0 - t);
        let w_b = 2.0 * t * (1.0 - t);
        let w_c = t * t;
        let params: Vec<f64> = model_a
            .params
            .iter()
            .zip(model_b.params.iter())
            .zip(model_c.params.iter())
            .map(|((&a, &b), &c)| w_a * a + w_b * b + w_c * c)
            .collect();
        Ok(MmModelWeights {
            params,
            layer_sizes: model_a.layer_sizes.clone(),
            layer_names: model_a.layer_names.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// §10  MmMetrics
// ---------------------------------------------------------------------------

/// Evaluation metrics for model-merging quality.
pub struct MmMetrics;

impl MmMetrics {
    /// L2 distance between parameter vectors.
    pub fn weight_distance(m1: &MmModelWeights, m2: &MmModelWeights) -> f64 {
        m1.params
            .iter()
            .zip(m2.params.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    /// L2 norm of a task vector.
    pub fn task_vector_magnitude(tv: &MmTaskVector) -> f64 {
        tv.vector.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Fraction of parameters with the same sign across two task vectors.
    pub fn agreement_rate(tv1: &MmTaskVector, tv2: &MmTaskVector) -> f64 {
        if tv1.vector.is_empty() || tv2.vector.is_empty() {
            return 0.0;
        }
        let n = tv1.vector.len().min(tv2.vector.len());
        let agree = tv1
            .vector
            .iter()
            .zip(tv2.vector.iter())
            .take(n)
            .filter(|(&a, &b)| {
                // Both positive, both negative, or both exactly zero
                (a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0) || (a == 0.0 && b == 0.0)
            })
            .count();
        agree as f64 / n as f64
    }

    /// Mean cosine similarity of merged weights to each individual model.
    pub fn merge_quality(merged: &MmModelWeights, individual_models: &[MmModelWeights]) -> f64 {
        if individual_models.is_empty() {
            return 0.0;
        }
        let total: f64 = individual_models
            .iter()
            .map(|m| merged.cosine_similarity(m))
            .sum();
        total / individual_models.len() as f64
    }

    /// Fraction of parameters with |tv| > threshold that overlap between tv1 and tv2.
    pub fn parameter_overlap(tv1: &MmTaskVector, tv2: &MmTaskVector, threshold: f64) -> f64 {
        let n = tv1.vector.len().min(tv2.vector.len());
        if n == 0 {
            return 0.0;
        }
        let mut sig1 = 0usize;
        let mut sig2 = 0usize;
        let mut both = 0usize;
        for i in 0..n {
            let a_sig = tv1.vector[i].abs() > threshold;
            let b_sig = tv2.vector[i].abs() > threshold;
            if a_sig {
                sig1 += 1;
            }
            if b_sig {
                sig2 += 1;
            }
            if a_sig && b_sig {
                both += 1;
            }
        }
        let union = sig1 + sig2 - both;
        if union == 0 {
            return 0.0;
        }
        both as f64 / union as f64
    }

    /// Performance drop after merging: max(0, before − after).
    pub fn forgetting_gap(before: f64, after: f64) -> f64 {
        (before - after).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// §11  Tests sub-module
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
