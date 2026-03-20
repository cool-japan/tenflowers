//! Multi-Objective Optimization & Pareto-Optimal ML.
//!
//! Covers: Pareto optimization, multi-objective neural networks, constrained
//! optimization, evolutionary algorithms, and multi-task learning gradient methods.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::fmt;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors from multi-objective operations.
#[derive(Debug)]
pub enum MooError {
    DimensionMismatch {
        expected: usize,
        found: usize,
    },
    EmptyInput {
        context: String,
    },
    InvalidParameter {
        name: String,
        value: f64,
        reason: String,
    },
    NumericalError {
        context: String,
    },
}

impl fmt::Display for MooError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MooError::DimensionMismatch { expected, found } => write!(
                f,
                "dimension mismatch: expected {}, found {}",
                expected, found
            ),
            MooError::EmptyInput { context } => write!(f, "empty input in: {}", context),
            MooError::InvalidParameter {
                name,
                value,
                reason,
            } => write!(f, "invalid parameter '{}' = {}: {}", name, value, reason),
            MooError::NumericalError { context } => write!(f, "numerical error in: {}", context),
        }
    }
}
impl std::error::Error for MooError {}
type Result<T> = std::result::Result<T, MooError>;

// ─── Section 1: Pareto Optimization ──────────────────────────────────────────

/// Non-dominated solution set (Pareto front), minimisation convention.
#[derive(Debug, Clone)]
pub struct ParetoFront {
    front: Vec<Vec<f64>>,
}

impl ParetoFront {
    pub fn new() -> Self {
        Self { front: Vec::new() }
    }

    /// `true` iff `a` dominates `b`: `a ≤ b` componentwise with at least one strict.
    pub fn dominates(a: &[f64], b: &[f64]) -> bool {
        if a.len() != b.len() || a.is_empty() {
            return false;
        }
        let mut strict = false;
        for (ai, bi) in a.iter().zip(b.iter()) {
            if ai > bi {
                return false;
            }
            if ai < bi {
                strict = true;
            }
        }
        strict
    }

    /// Add a solution; dominated members are removed.
    pub fn add(&mut self, objectives: Vec<f64>) {
        if self.front.iter().any(|e| Self::dominates(e, &objectives)) {
            return;
        }
        self.front.retain(|e| !Self::dominates(&objectives, e));
        self.front.push(objectives);
    }

    pub fn get_front(&self) -> &[Vec<f64>] {
        &self.front
    }
    pub fn len(&self) -> usize {
        self.front.len()
    }
    pub fn is_empty(&self) -> bool {
        self.front.is_empty()
    }
}

impl Default for ParetoFront {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Hypervolume indicator w.r.t. a reference point.
/// Exact sweep for 2-D; Monte-Carlo approximation for higher dimensions.
pub struct HypervolumeIndicator {
    pub mc_samples: usize,
    pub seed: u64,
}

impl HypervolumeIndicator {
    pub fn new() -> Self {
        Self {
            mc_samples: 10_000,
            seed: 42,
        }
    }
    pub fn with_params(mc_samples: usize, seed: u64) -> Self {
        Self { mc_samples, seed }
    }

    pub fn compute(&self, front: &[Vec<f64>], reference: &[f64]) -> Result<f64> {
        if front.is_empty() {
            return Ok(0.0);
        }
        let n_obj = reference.len();
        if n_obj == 0 {
            return Err(MooError::EmptyInput {
                context: "reference point".into(),
            });
        }
        for sol in front {
            if sol.len() != n_obj {
                return Err(MooError::DimensionMismatch {
                    expected: n_obj,
                    found: sol.len(),
                });
            }
        }
        if n_obj == 2 {
            self.hv_2d(front, reference)
        } else {
            self.hv_mc(front, reference, n_obj)
        }
    }

    fn hv_2d(&self, front: &[Vec<f64>], r: &[f64]) -> Result<f64> {
        let mut pts: Vec<&Vec<f64>> = front
            .iter()
            .filter(|p| p[0] < r[0] && p[1] < r[1])
            .collect();
        if pts.is_empty() {
            return Ok(0.0);
        }
        pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
        let mut hv = 0.0_f64;
        let mut prev_y = r[1];
        for p in &pts {
            let h = prev_y - p[1];
            if h > 0.0 {
                hv += (r[0] - p[0]) * h;
            }
            if p[1] < prev_y {
                prev_y = p[1];
            }
        }
        Ok(hv)
    }

    fn hv_mc(&self, front: &[Vec<f64>], r: &[f64], n_obj: usize) -> Result<f64> {
        let mut nadir = r.to_vec();
        for p in front {
            for (j, &pj) in p.iter().enumerate() {
                if pj < nadir[j] {
                    nadir[j] = pj;
                }
            }
        }
        let box_vol: f64 = nadir
            .iter()
            .zip(r.iter())
            .map(|(lo, hi)| (hi - lo).max(0.0))
            .product();
        if box_vol <= 0.0 {
            return Ok(0.0);
        }
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut hits = 0_u64;
        for _ in 0..self.mc_samples {
            let pt: Vec<f64> = (0..n_obj)
                .map(|j| {
                    let u: f64 = rng.random();
                    nadir[j] + u * (r[j] - nadir[j])
                })
                .collect();
            if front.iter().any(|fp| ParetoFront::dominates(fp, &pt)) {
                hits += 1;
            }
        }
        Ok(box_vol * (hits as f64 / self.mc_samples as f64))
    }
}

impl Default for HypervolumeIndicator {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// NSGA-III with reference-point generation and non-dominated sorting.
pub struct NsgaIii {
    pub n_obj: usize,
    pub divisions: usize,
}

impl NsgaIii {
    pub fn new(n_obj: usize, divisions: usize) -> Self {
        Self { n_obj, divisions }
    }

    /// Das & Dennis reference points on the unit simplex.
    pub fn generate_reference_points(n_obj: usize, divisions: usize) -> Vec<Vec<f64>> {
        let mut pts = Vec::new();
        let mut cur = vec![0usize; n_obj];
        Self::enum_pts(n_obj, divisions, 0, divisions, &mut cur, &mut pts);
        pts
    }

    fn enum_pts(
        n_obj: usize,
        total: usize,
        idx: usize,
        rem: usize,
        cur: &mut Vec<usize>,
        pts: &mut Vec<Vec<f64>>,
    ) {
        if idx == n_obj - 1 {
            cur[idx] = rem;
            pts.push(cur.iter().map(|&c| c as f64 / total as f64).collect());
            return;
        }
        for v in 0..=rem {
            cur[idx] = v;
            Self::enum_pts(n_obj, total, idx + 1, rem - v, cur, pts);
        }
    }

    /// Non-dominated sort + surviving population.
    pub fn evolve(
        &self,
        population: &[Vec<f64>],
        _objectives: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>> {
        if population.is_empty() {
            return Err(MooError::EmptyInput {
                context: "population".into(),
            });
        }
        Ok(self
            .non_dominated_sort(population)
            .into_iter()
            .flatten()
            .collect())
    }

    pub fn non_dominated_sort(&self, pop: &[Vec<f64>]) -> Vec<Vec<Vec<f64>>> {
        let n = pop.len();
        let mut dom_cnt = vec![0usize; n];
        let mut dominates: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if ParetoFront::dominates(&pop[i], &pop[j]) {
                    dominates[i].push(j);
                } else if ParetoFront::dominates(&pop[j], &pop[i]) {
                    dom_cnt[i] += 1;
                }
            }
        }
        let mut fronts: Vec<Vec<usize>> = Vec::new();
        let mut cur: Vec<usize> = (0..n).filter(|&i| dom_cnt[i] == 0).collect();
        while !cur.is_empty() {
            fronts.push(cur.clone());
            let mut nxt = Vec::new();
            for &i in &cur {
                for &j in &dominates[i] {
                    dom_cnt[j] -= 1;
                    if dom_cnt[j] == 0 {
                        nxt.push(j);
                    }
                }
            }
            cur = nxt;
        }
        fronts
            .into_iter()
            .map(|f| f.into_iter().map(|i| pop[i].clone()).collect())
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Crowding-distance-based Pareto ranking.
pub struct SpearRanking;

impl SpearRanking {
    /// Rank by crowding distance descending (rank 0 = least crowded = best).
    pub fn rank(objectives: &[Vec<f64>]) -> Result<Vec<usize>> {
        if objectives.is_empty() {
            return Err(MooError::EmptyInput {
                context: "objectives".into(),
            });
        }
        let n_obj = objectives[0].len();
        let n = objectives.len();
        let dists = Self::crowding(objectives, n_obj);
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            dists[b]
                .partial_cmp(&dists[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut ranks = vec![0usize; n];
        for (r, &i) in idx.iter().enumerate() {
            ranks[i] = r;
        }
        Ok(ranks)
    }

    fn crowding(objs: &[Vec<f64>], n_obj: usize) -> Vec<f64> {
        let n = objs.len();
        let mut d = vec![0.0_f64; n];
        if n <= 2 {
            for v in d.iter_mut() {
                *v = f64::INFINITY;
            }
            return d;
        }
        for m in 0..n_obj {
            let mut si: Vec<usize> = (0..n).collect();
            si.sort_by(|&a, &b| {
                objs[a][m]
                    .partial_cmp(&objs[b][m])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let range = objs[si[n - 1]][m] - objs[si[0]][m];
            d[si[0]] = f64::INFINITY;
            d[si[n - 1]] = f64::INFINITY;
            if range < 1e-15 {
                continue;
            }
            for k in 1..n - 1 {
                d[si[k]] += (objs[si[k + 1]][m] - objs[si[k - 1]][m]) / range;
            }
        }
        d
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// R2 quality indicator — weighted achievement scalarization.
pub struct R2Indicator;

impl R2Indicator {
    pub fn compute(
        front: &[Vec<f64>],
        weight_vectors: &[Vec<f64>],
        reference: &[f64],
    ) -> Result<f64> {
        if front.is_empty() {
            return Err(MooError::EmptyInput {
                context: "front".into(),
            });
        }
        if weight_vectors.is_empty() {
            return Err(MooError::EmptyInput {
                context: "weight_vectors".into(),
            });
        }
        let n_obj = reference.len();
        let mut total = 0.0_f64;
        for w in weight_vectors {
            if w.len() != n_obj {
                return Err(MooError::DimensionMismatch {
                    expected: n_obj,
                    found: w.len(),
                });
            }
            let min_scalar = front
                .iter()
                .map(|p| {
                    if p.len() != n_obj {
                        return f64::INFINITY;
                    }
                    p.iter()
                        .zip(w.iter())
                        .zip(reference.iter())
                        .map(|((pi, wi), ri)| wi * (pi - ri).abs())
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .fold(f64::INFINITY, f64::min);
            total += min_scalar;
        }
        Ok(total / weight_vectors.len() as f64)
    }
}

// ─── Section 2: Multi-Objective Neural Networks ───────────────────────────────

fn dense_relu(x: &[f64], w: &[f64], b: &[f64], in_d: usize, out_d: usize) -> Vec<f64> {
    (0..out_d)
        .map(|j| {
            (x.iter()
                .enumerate()
                .map(|(i, xi)| w[j * in_d + i] * xi)
                .sum::<f64>()
                + b[j])
                .max(0.0)
        })
        .collect()
}

fn dense_linear(x: &[f64], w: &[f64], b: &[f64], in_d: usize, out_d: usize) -> Vec<f64> {
    (0..out_d)
        .map(|j| {
            x.iter()
                .enumerate()
                .map(|(i, xi)| w[j * in_d + i] * xi)
                .sum::<f64>()
                + b[j]
        })
        .collect()
}

fn xavier_f64(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    (0..fan_in * fan_out)
        .map(|_| {
            let u: f64 = rng.random();
            u * 2.0 * limit - limit
        })
        .collect()
}

/// Multi-head network: shared encoder + K task heads.
pub struct MooHeads {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub head_dims: Vec<usize>,
    shared_w: Vec<f64>,
    shared_b: Vec<f64>,
    head_weights: Vec<(Vec<f64>, Vec<f64>)>,
}

impl MooHeads {
    pub fn new(in_dim: usize, hidden_dim: usize, head_dims: Vec<usize>) -> Self {
        let shared_w = xavier_f64(in_dim, hidden_dim, 1);
        let shared_b = vec![0.0; hidden_dim];
        let head_weights = head_dims
            .iter()
            .enumerate()
            .map(|(k, &out)| {
                (
                    xavier_f64(hidden_dim, out, (k + 2) as u64 * 31),
                    vec![0.0; out],
                )
            })
            .collect();
        Self {
            in_dim,
            hidden_dim,
            head_dims,
            shared_w,
            shared_b,
            head_weights,
        }
    }

    pub fn forward(&self, x: &[f64]) -> Result<Vec<Vec<f64>>> {
        if x.len() != self.in_dim {
            return Err(MooError::DimensionMismatch {
                expected: self.in_dim,
                found: x.len(),
            });
        }
        let h = dense_relu(
            x,
            &self.shared_w,
            &self.shared_b,
            self.in_dim,
            self.hidden_dim,
        );
        Ok(self
            .head_weights
            .iter()
            .zip(self.head_dims.iter())
            .map(|((w, b), &out)| dense_linear(&h, w, b, self.hidden_dim, out))
            .collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Pareto-optimal mixture of experts; routing uses Tchebycheff scalarization.
pub struct ParetoOptimalMoe {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub out_dim: usize,
    pub n_experts: usize,
    enc_w: Vec<f64>,
    enc_b: Vec<f64>,
    expert_w: Vec<Vec<f64>>,
    expert_b: Vec<Vec<f64>>,
}

impl ParetoOptimalMoe {
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, n_experts: usize) -> Self {
        let enc_w = xavier_f64(in_dim, hidden_dim, 7);
        let enc_b = vec![0.0; hidden_dim];
        let expert_w: Vec<_> = (0..n_experts)
            .map(|k| xavier_f64(hidden_dim, out_dim, (k + 1) as u64 * 97))
            .collect();
        let expert_b: Vec<_> = (0..n_experts).map(|_| vec![0.0; out_dim]).collect();
        Self {
            in_dim,
            hidden_dim,
            out_dim,
            n_experts,
            enc_w,
            enc_b,
            expert_w,
            expert_b,
        }
    }

    pub fn forward(&self, x: &[f64], lambda: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.in_dim {
            return Err(MooError::DimensionMismatch {
                expected: self.in_dim,
                found: x.len(),
            });
        }
        if lambda.len() != self.n_experts {
            return Err(MooError::DimensionMismatch {
                expected: self.n_experts,
                found: lambda.len(),
            });
        }
        let h = dense_relu(x, &self.enc_w, &self.enc_b, self.in_dim, self.hidden_dim);
        let outs: Vec<Vec<f64>> = (0..self.n_experts)
            .map(|k| {
                dense_linear(
                    &h,
                    &self.expert_w[k],
                    &self.expert_b[k],
                    self.hidden_dim,
                    self.out_dim,
                )
            })
            .collect();
        let mut best_k = 0;
        let mut best = f64::INFINITY;
        for (k, out) in outs.iter().enumerate() {
            let s = out
                .iter()
                .map(|v| lambda[k] * v.abs())
                .fold(f64::NEG_INFINITY, f64::max);
            if s < best {
                best = s;
                best_k = k;
            }
        }
        Ok(outs[best_k].clone())
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Linear scalarization: `L = Σ λ_i * L_i`.
pub struct LinearScalarization;

impl LinearScalarization {
    pub fn compute_loss(losses: &[f64], weights: &[f64]) -> Result<f64> {
        if losses.len() != weights.len() {
            return Err(MooError::DimensionMismatch {
                expected: losses.len(),
                found: weights.len(),
            });
        }
        if losses.is_empty() {
            return Err(MooError::EmptyInput {
                context: "losses".into(),
            });
        }
        Ok(losses.iter().zip(weights.iter()).map(|(l, w)| l * w).sum())
    }

    /// Sample a weight vector from the unit simplex via exponential normalisation.
    pub fn sample_weights(n_obj: usize, rng: &mut StdRng) -> Vec<f64> {
        let us: Vec<f64> = (0..n_obj)
            .map(|_| {
                let u: f64 = rng.random::<f64>().max(1e-15);
                -u.ln()
            })
            .collect();
        let tot: f64 = us.iter().sum();
        us.iter().map(|u| u / tot).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Tchebycheff scalarization: `L = max_i( λ_i * |L_i - z_i^*| )`.
pub struct TchebycheffScalarization;

impl TchebycheffScalarization {
    pub fn compute_loss(losses: &[f64], weights: &[f64], utopia: &[f64]) -> Result<f64> {
        let n = losses.len();
        if weights.len() != n || utopia.len() != n {
            return Err(MooError::DimensionMismatch {
                expected: n,
                found: weights.len().max(utopia.len()),
            });
        }
        if n == 0 {
            return Err(MooError::EmptyInput {
                context: "losses".into(),
            });
        }
        Ok(losses
            .iter()
            .zip(weights.iter())
            .zip(utopia.iter())
            .map(|((l, w), z)| w * (l - z).abs())
            .fold(f64::NEG_INFINITY, f64::max))
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Pareto-Front Learning layer: samples preference, scalarizes, maintains Pareto approx.
pub struct PFLLayer {
    pub pareto_approx: ParetoFront,
    pub seed: u64,
    call_count: u64,
}

impl PFLLayer {
    pub fn new(seed: u64) -> Self {
        Self {
            pareto_approx: ParetoFront::new(),
            seed,
            call_count: 0,
        }
    }

    pub fn step(&mut self, objectives: &[f64]) -> Result<(Vec<f64>, f64)> {
        if objectives.is_empty() {
            return Err(MooError::EmptyInput {
                context: "objectives".into(),
            });
        }
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(self.call_count));
        self.call_count += 1;
        let w = LinearScalarization::sample_weights(objectives.len(), &mut rng);
        let s = LinearScalarization::compute_loss(objectives, &w)?;
        self.pareto_approx.add(objectives.to_vec());
        Ok((w, s))
    }
}

// ─── Section 3: Constrained Optimization ─────────────────────────────────────

/// Augmented Lagrangian (method of multipliers).
pub struct AugmentedLagrangian;

impl AugmentedLagrangian {
    /// Returns `(augmented_loss, updated_lambda)`.
    pub fn step(
        loss: f64,
        constraints: &[f64],
        lambda: &[f64],
        mu: f64,
    ) -> Result<(f64, Vec<f64>)> {
        if constraints.len() != lambda.len() {
            return Err(MooError::DimensionMismatch {
                expected: constraints.len(),
                found: lambda.len(),
            });
        }
        let aug = loss
            + constraints
                .iter()
                .zip(lambda.iter())
                .map(|(g, l)| l * g + (mu / 2.0) * g * g)
                .sum::<f64>();
        let new_lambda: Vec<f64> = constraints
            .iter()
            .zip(lambda.iter())
            .map(|(g, l)| l + mu * g)
            .collect();
        Ok((aug, new_lambda))
    }

    pub fn penalty_term(constraints: &[f64], lambda: &[f64], mu: f64) -> Result<f64> {
        if constraints.len() != lambda.len() {
            return Err(MooError::DimensionMismatch {
                expected: constraints.len(),
                found: lambda.len(),
            });
        }
        Ok(constraints
            .iter()
            .zip(lambda.iter())
            .map(|(g, l)| l * g + (mu / 2.0) * g * g)
            .sum())
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Log-barrier interior point method.
pub struct InteriorPointMethod {
    pub mu_reduction: f64,
}

impl InteriorPointMethod {
    pub fn new(mu_reduction: f64) -> Self {
        Self { mu_reduction }
    }

    /// `-Σ log(-g_i) / t`. Returns `+∞` if any constraint is non-negative.
    pub fn barrier(constraints: &[f64], t: f64) -> f64 {
        let s: f64 = constraints
            .iter()
            .map(|&g| if -g <= 0.0 { f64::INFINITY } else { -(-g).ln() })
            .sum();
        if s.is_infinite() {
            f64::INFINITY
        } else {
            s / t
        }
    }

    pub fn update_t(&self, t: f64) -> f64 {
        t / self.mu_reduction
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Gradient projection onto constraint manifolds.
pub struct ProjectedGradient;

impl ProjectedGradient {
    pub fn project_box(x: &[f64], lb: &[f64], ub: &[f64]) -> Result<Vec<f64>> {
        let n = x.len();
        if lb.len() != n || ub.len() != n {
            return Err(MooError::DimensionMismatch {
                expected: n,
                found: lb.len().max(ub.len()),
            });
        }
        Ok(x.iter()
            .zip(lb.iter())
            .zip(ub.iter())
            .map(|((&xi, &li), &ui)| xi.clamp(li, ui))
            .collect())
    }

    /// Euclidean projection onto the probability simplex (Duchi et al. 2008).
    pub fn project_simplex(x: &[f64]) -> Result<Vec<f64>> {
        let n = x.len();
        if n == 0 {
            return Err(MooError::EmptyInput {
                context: "project_simplex".into(),
            });
        }
        let mut sorted = x.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumsum = 0.0_f64;
        let mut rho = 0;
        for (j, &s) in sorted.iter().enumerate() {
            cumsum += s;
            if s - (cumsum - 1.0) / (j as f64 + 1.0) > 0.0 {
                rho = j;
            }
        }
        let theta = {
            let mut s = 0.0;
            for j in 0..=rho {
                s += sorted[j];
            }
            (s - 1.0) / (rho as f64 + 1.0)
        };
        Ok(x.iter().map(|&xi| (xi - theta).max(0.0)).collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Quadratic penalty method: `L = f + (ρ/2) Σ_i max(0, g_i)²`.
pub struct PenaltyMethod;

impl PenaltyMethod {
    pub fn loss(f: f64, constraints: &[f64], rho: f64) -> f64 {
        f + (rho / 2.0) * constraints.iter().map(|&g| g.max(0.0).powi(2)).sum::<f64>()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Feasibility pump: rounds integer-constrained variables.
pub struct FeasibilityPump;

impl FeasibilityPump {
    pub fn pump_step(x: &[f64], integer_mask: &[bool]) -> Result<Vec<f64>> {
        if x.len() != integer_mask.len() {
            return Err(MooError::DimensionMismatch {
                expected: x.len(),
                found: integer_mask.len(),
            });
        }
        Ok(x.iter()
            .zip(integer_mask.iter())
            .map(|(&xi, &is_int)| if is_int { xi.round() } else { xi })
            .collect())
    }
}

// ─── Section 4: Evolutionary Multi-Objective ─────────────────────────────────

/// MOEA/D: decomposition with Tchebycheff + neighbourhood mating.
pub struct Moead {
    pub n_solutions: usize,
    pub n_obj: usize,
    pub t_neighbor: usize,
    pub weights: Vec<Vec<f64>>,
    pub z_star: Vec<f64>,
    pub population: Vec<Vec<f64>>,
}

impl Moead {
    pub fn initialize(n_solutions: usize, n_obj: usize, t_neighbor: usize) -> Self {
        let weights = ReferencePointSampler::sample(n_obj, n_solutions.max(1) - 1)
            .unwrap_or_else(|_| vec![vec![1.0 / n_obj as f64; n_obj]; n_solutions]);
        let z_star = vec![f64::INFINITY; n_obj];
        let mut rng = StdRng::seed_from_u64(13);
        let population: Vec<Vec<f64>> = (0..n_solutions)
            .map(|_| (0..n_obj).map(|_| rng.random::<f64>()).collect())
            .collect();
        Self {
            n_solutions,
            n_obj,
            t_neighbor,
            weights,
            z_star,
            population,
        }
    }

    pub fn evolve_step<F>(&mut self, objectives_fn: F, rng: &mut StdRng) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let t = self.t_neighbor.min(self.n_solutions);
        for i in 0..self.n_solutions {
            let mut di: Vec<(f64, usize)> = self
                .weights
                .iter()
                .enumerate()
                .map(|(j, wj)| {
                    let d: f64 = wj
                        .iter()
                        .zip(self.weights[i].iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    (d, j)
                })
                .collect();
            di.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let nb: Vec<usize> = di.iter().take(t).map(|(_, j)| *j).collect();
            let ia = nb[rng.random_range(0..nb.len())];
            let ib = nb[rng.random_range(0..nb.len())];
            let (c1, _) = GeneticOperators::sbx_crossover(
                &self.population[ia],
                &self.population[ib],
                20.0,
                rng,
            );
            let bnd: Vec<(f64, f64)> = (0..self.n_obj).map(|_| (0.0, 1.0)).collect();
            let child = GeneticOperators::polynomial_mutation(&c1, 20.0, &bnd, rng);
            let co = objectives_fn(&child);
            for (j, &fj) in co.iter().enumerate() {
                if fj < self.z_star[j] {
                    self.z_star[j] = fj;
                }
            }
            for &nb in &nb {
                let ct =
                    TchebycheffScalarization::compute_loss(&co, &self.weights[nb], &self.z_star)
                        .unwrap_or(f64::INFINITY);
                let po = objectives_fn(&self.population[nb]);
                let pt =
                    TchebycheffScalarization::compute_loss(&po, &self.weights[nb], &self.z_star)
                        .unwrap_or(f64::INFINITY);
                if ct <= pt {
                    self.population[nb] = child.clone();
                }
            }
        }
        self.population.clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// SMS-EMOA: removes solution with smallest hypervolume contribution.
pub struct SmsEmoa {
    pub reference: Vec<f64>,
}

impl SmsEmoa {
    pub fn new(reference: Vec<f64>) -> Self {
        Self { reference }
    }

    pub fn select(&self, population: &[Vec<f64>], objectives: &[Vec<f64>]) -> Result<usize> {
        if objectives.is_empty() {
            return Err(MooError::EmptyInput {
                context: "objectives".into(),
            });
        }
        let hv = HypervolumeIndicator::new();
        let hv_full = hv.compute(objectives, &self.reference)?;
        let mut min_contrib = f64::INFINITY;
        let mut worst = 0;
        for i in 0..objectives.len() {
            let red: Vec<Vec<f64>> = objectives
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, v)| v.clone())
                .collect();
            let contrib = hv_full - hv.compute(&red, &self.reference)?;
            if contrib < min_contrib {
                min_contrib = contrib;
                worst = i;
            }
        }
        let _ = population;
        Ok(worst)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Real-valued genetic operators: SBX crossover and polynomial mutation.
pub struct GeneticOperators;

impl GeneticOperators {
    /// Simulated Binary Crossover (SBX). `eta_c`: distribution index.
    pub fn sbx_crossover(
        p1: &[f64],
        p2: &[f64],
        eta_c: f64,
        rng: &mut StdRng,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = p1.len().min(p2.len());
        let mut c1 = vec![0.0; n];
        let mut c2 = vec![0.0; n];
        for i in 0..n {
            let u: f64 = rng.random();
            let beta = if u <= 0.5 {
                (2.0 * u).powf(1.0 / (eta_c + 1.0))
            } else {
                (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta_c + 1.0))
            };
            c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i]);
            c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i]);
        }
        (c1, c2)
    }

    /// Polynomial mutation. `eta_m`: distribution index; `bounds`: `(lb, ub)` per dim.
    pub fn polynomial_mutation(
        x: &[f64],
        eta_m: f64,
        bounds: &[(f64, f64)],
        rng: &mut StdRng,
    ) -> Vec<f64> {
        x.iter()
            .zip(bounds.iter())
            .map(|(&xi, &(lb, ub))| {
                let range = ub - lb;
                if range <= 0.0 {
                    return xi;
                }
                let u: f64 = rng.random();
                let u2: f64 = u;
                let delta = if u2 < 0.5 {
                    let bl = (xi - lb) / range;
                    (2.0 * u2 + (1.0 - 2.0 * u2) * (1.0 - bl).powf(eta_m + 1.0))
                        .powf(1.0 / (eta_m + 1.0))
                        - 1.0
                } else {
                    1.0 - (2.0 * (1.0 - u2)
                        + 2.0 * (u2 - 0.5) * (1.0 - (ub - xi) / range).powf(eta_m + 1.0))
                    .powf(1.0 / (eta_m + 1.0))
                };
                (xi + delta * range).clamp(lb, ub)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Das & Dennis reference-point lattice on the unit simplex.
pub struct ReferencePointSampler;

impl ReferencePointSampler {
    pub fn sample(n_obj: usize, h: usize) -> Result<Vec<Vec<f64>>> {
        if n_obj == 0 {
            return Err(MooError::InvalidParameter {
                name: "n_obj".into(),
                value: 0.0,
                reason: "must be positive".into(),
            });
        }
        Ok(NsgaIii::generate_reference_points(n_obj, h))
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Tournament, roulette wheel, and rank-based selection operators.
pub struct SelectionOperator;

impl SelectionOperator {
    pub fn tournament(
        population: &[Vec<f64>],
        fitness: &[f64],
        k: usize,
        rng: &mut StdRng,
    ) -> Result<usize> {
        if fitness.is_empty() {
            return Err(MooError::EmptyInput {
                context: "fitness".into(),
            });
        }
        let n = fitness.len();
        let k_act = k.max(1).min(n);
        let mut best = rng.random_range(0..n);
        for _ in 1..k_act {
            let c = rng.random_range(0..n);
            if fitness[c] < fitness[best] {
                best = c;
            }
        }
        let _ = population;
        Ok(best)
    }

    pub fn roulette(fitness: &[f64], rng: &mut StdRng) -> Result<usize> {
        if fitness.is_empty() {
            return Err(MooError::EmptyInput {
                context: "fitness".into(),
            });
        }
        let total: f64 = fitness.iter().sum();
        if total <= 0.0 {
            return Ok(rng.random_range(0..fitness.len()));
        }
        let thr: f64 = rng.random::<f64>() * total;
        let mut cum = 0.0;
        for (i, &f) in fitness.iter().enumerate() {
            cum += f;
            if cum >= thr {
                return Ok(i);
            }
        }
        Ok(fitness.len() - 1)
    }

    pub fn rank_select(fitness: &[f64], rng: &mut StdRng) -> Result<usize> {
        if fitness.is_empty() {
            return Err(MooError::EmptyInput {
                context: "fitness".into(),
            });
        }
        let n = fitness.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            fitness[a]
                .partial_cmp(&fitness[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let rf: Vec<f64> = (0..n).map(|i| (n - i) as f64).collect();
        let tot: f64 = rf.iter().sum();
        let thr: f64 = rng.random::<f64>() * tot;
        let mut cum = 0.0;
        for (rank, &i) in idx.iter().enumerate() {
            cum += rf[rank];
            if cum >= thr {
                return Ok(i);
            }
        }
        Ok(idx[n - 1])
    }
}

// ─── Section 5: Multi-Task Learning ──────────────────────────────────────────

/// GradNorm: dynamically balance task loss weights via gradient norm matching.
/// Reference: Chen et al., ICML 2018.
pub struct GradNormOptimizer;

impl GradNormOptimizer {
    pub fn compute_task_weights(
        losses: &[f64],
        shared_grads: &[Vec<f64>],
        target_rate: f64,
        alpha: f64,
    ) -> Result<Vec<f64>> {
        let n = losses.len();
        if shared_grads.len() != n {
            return Err(MooError::DimensionMismatch {
                expected: n,
                found: shared_grads.len(),
            });
        }
        if n == 0 {
            return Err(MooError::EmptyInput {
                context: "losses".into(),
            });
        }
        let gnorms: Vec<f64> = shared_grads
            .iter()
            .map(|g| g.iter().map(|v| v * v).sum::<f64>().sqrt())
            .collect();
        let mean_gn: f64 = gnorms.iter().sum::<f64>() / n as f64;
        let mean_l: f64 = losses.iter().sum::<f64>() / n as f64;
        let ratios: Vec<f64> = losses
            .iter()
            .map(|&l| if mean_l > 0.0 { l / mean_l } else { 1.0 })
            .collect();
        let targets: Vec<f64> = ratios.iter().map(|&r| mean_gn * r.powf(alpha)).collect();
        let mut weights: Vec<f64> = gnorms
            .iter()
            .zip(targets.iter())
            .map(|(&gn, &tgn)| {
                if gn > 1e-15 {
                    (tgn / gn).powf(target_rate)
                } else {
                    1.0
                }
            })
            .collect();
        let sw: f64 = weights.iter().sum();
        if sw > 1e-15 {
            for w in weights.iter_mut() {
                *w = *w * n as f64 / sw;
            }
        }
        Ok(weights)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// PCGrad: Project Conflicting Gradients.
/// Reference: Yu et al., NeurIPS 2020.
pub struct PcGrad;

impl PcGrad {
    pub fn project(task_grads: &[Vec<f64>]) -> Result<Vec<f64>> {
        if task_grads.is_empty() {
            return Err(MooError::EmptyInput {
                context: "task_grads".into(),
            });
        }
        let dim = task_grads[0].len();
        let n = task_grads.len();
        let mut proj: Vec<Vec<f64>> = task_grads.to_vec();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dot: f64 = proj[i]
                    .iter()
                    .zip(task_grads[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                if dot < 0.0 {
                    let nj2: f64 = task_grads[j].iter().map(|v| v * v).sum();
                    if nj2 > 1e-15 {
                        let s = dot / nj2;
                        for k in 0..dim {
                            proj[i][k] -= s * task_grads[j][k];
                        }
                    }
                }
            }
        }
        let mut avg = vec![0.0_f64; dim];
        for g in &proj {
            for (k, &gk) in g.iter().enumerate() {
                avg[k] += gk;
            }
        }
        for v in avg.iter_mut() {
            *v /= n as f64;
        }
        Ok(avg)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// CAGrad: Conflict-Averse Gradient.
/// Reference: Liu et al., NeurIPS 2021.
pub struct CaGrad;

impl CaGrad {
    pub fn compute(task_grads: &[Vec<f64>], c: f64) -> Result<Vec<f64>> {
        if task_grads.is_empty() {
            return Err(MooError::EmptyInput {
                context: "task_grads".into(),
            });
        }
        let dim = task_grads[0].len();
        let n = task_grads.len();
        let mut avg = vec![0.0_f64; dim];
        for g in task_grads {
            for (k, &gk) in g.iter().enumerate() {
                avg[k] += gk / n as f64;
            }
        }
        let an = avg.iter().map(|v| v * v).sum::<f64>().sqrt();
        if an < 1e-15 {
            return Ok(avg);
        }
        let mut result = avg.clone();
        for g in task_grads {
            let gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            if gn < 1e-15 {
                continue;
            }
            let rn = result.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-15);
            let cos: f64 = g.iter().zip(result.iter()).map(|(a, b)| a * b).sum::<f64>() / (gn * rn);
            if cos < c {
                let blend = (c - cos).min(1.0);
                for k in 0..dim {
                    result[k] = (1.0 - blend) * result[k] + blend * g[k];
                }
            }
        }
        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// IMTL-G: unit-normalised balanced gradient for impartial multi-task learning.
/// Reference: Liu et al., ICLR 2021.
pub struct ImtlG;

impl ImtlG {
    pub fn compute(task_grads: &[Vec<f64>]) -> Result<Vec<f64>> {
        if task_grads.is_empty() {
            return Err(MooError::EmptyInput {
                context: "task_grads".into(),
            });
        }
        let dim = task_grads[0].len();
        let n = task_grads.len();
        let units: Vec<Vec<f64>> = task_grads
            .iter()
            .map(|g| {
                let norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm > 1e-15 {
                    g.iter().map(|v| v / norm).collect()
                } else {
                    vec![0.0; dim]
                }
            })
            .collect();
        let mut r = vec![0.0_f64; dim];
        for u in &units {
            for (k, &uk) in u.iter().enumerate() {
                r[k] += uk / n as f64;
            }
        }
        Ok(r)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Uncertainty-based multi-task weighting (Kendall & Gal, 2018).
/// `L = Σ_i (1/(2σ_i²)) * L_i + log(σ_i)`
pub struct UncertaintyWeighting;

impl UncertaintyWeighting {
    pub fn compute_loss(losses: &[f64], log_sigmas: &[f64]) -> Result<f64> {
        let n = losses.len();
        if log_sigmas.len() != n {
            return Err(MooError::DimensionMismatch {
                expected: n,
                found: log_sigmas.len(),
            });
        }
        if n == 0 {
            return Err(MooError::EmptyInput {
                context: "losses".into(),
            });
        }
        Ok(losses
            .iter()
            .zip(log_sigmas.iter())
            .map(|(&l, &ls)| {
                // σ = exp(ls), 1/(2σ²) = exp(-2ls)/2
                (-2.0 * ls).exp() / 2.0 * l + ls
            })
            .sum())
    }

    /// SGD step on `log_sigma` parameters. `∂L/∂ls_i = -exp(-2ls_i) * L_i + 1`.
    pub fn update_sigmas(log_sigmas: &[f64], losses: &[f64], lr: f64) -> Result<Vec<f64>> {
        let n = log_sigmas.len();
        if losses.len() != n {
            return Err(MooError::DimensionMismatch {
                expected: n,
                found: losses.len(),
            });
        }
        Ok(log_sigmas
            .iter()
            .zip(losses.iter())
            .map(|(&ls, &l)| {
                let grad = -(-2.0 * ls).exp() * l + 1.0;
                ls - lr * grad
            })
            .collect())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Pareto ────────────────────────────────────────────────────────────────

    #[test]
    fn test_pareto_front_dominates() {
        assert!(ParetoFront::dominates(&[1.0, 2.0], &[2.0, 3.0]));
        assert!(ParetoFront::dominates(&[1.0, 2.0], &[1.0, 3.0]));
        assert!(!ParetoFront::dominates(&[1.0, 2.0], &[1.0, 2.0]));
        assert!(!ParetoFront::dominates(&[2.0, 1.0], &[1.0, 2.0]));
        assert!(!ParetoFront::dominates(&[], &[]));
    }

    #[test]
    fn test_pareto_front_add() {
        let mut f = ParetoFront::new();
        f.add(vec![1.0, 3.0]);
        f.add(vec![2.0, 2.0]);
        f.add(vec![3.0, 1.0]);
        assert_eq!(f.len(), 3);
        f.add(vec![2.0, 3.0]);
        assert_eq!(f.len(), 3); // dominated
        f.add(vec![0.5, 0.5]);
        assert_eq!(f.len(), 1); // dominates all
    }

    #[test]
    fn test_hypervolume_2d() {
        let hv = HypervolumeIndicator::new();
        let front = vec![vec![1.0, 4.0], vec![2.0, 2.0], vec![3.0, 1.0]];
        let r = hv.compute(&front, &[5.0, 5.0]).expect("test value");
        assert!(r > 0.0 && r < 25.0);
    }

    #[test]
    fn test_hypervolume_empty_front() {
        let hv = HypervolumeIndicator::new();
        assert_eq!(hv.compute(&[], &[5.0, 5.0]).expect("test value"), 0.0);
    }

    #[test]
    fn test_nsga3_reference_points() {
        let pts = NsgaIii::generate_reference_points(2, 4);
        assert_eq!(pts.len(), 5);
        for p in &pts {
            let s: f64 = p.iter().sum();
            assert!((s - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_nsga3_reference_points_3d() {
        let pts = NsgaIii::generate_reference_points(3, 3);
        assert_eq!(pts.len(), 10);
        for p in &pts {
            let s: f64 = p.iter().sum();
            assert!((s - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_r2_indicator() {
        let front = vec![vec![0.1, 0.9], vec![0.5, 0.5], vec![0.9, 0.1]];
        let weights = vec![vec![0.5, 0.5], vec![0.25, 0.75], vec![0.75, 0.25]];
        let r2 = R2Indicator::compute(&front, &weights, &[0.0, 0.0]).expect("test value");
        assert!(r2 >= 0.0);
    }

    #[test]
    fn test_r2_indicator_empty_front() {
        assert!(R2Indicator::compute(&[], &[vec![0.5, 0.5]], &[0.0, 0.0]).is_err());
    }

    #[test]
    fn test_spear_ranking() {
        let objs = vec![
            vec![1.0, 4.0],
            vec![2.0, 2.0],
            vec![3.0, 1.0],
            vec![2.0, 2.0],
        ];
        let ranks = SpearRanking::rank(&objs).expect("test value");
        assert_eq!(ranks.len(), 4);
        for &r in &ranks {
            assert!(r < 4);
        }
    }

    // ─── MO Neural ─────────────────────────────────────────────────────────────

    #[test]
    fn test_moo_heads_shape() {
        let m = MooHeads::new(4, 8, vec![2, 3]);
        let out = m.forward(&[0.5, -0.3, 1.0, 0.2]).expect("test value");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 2);
        assert_eq!(out[1].len(), 3);
    }

    #[test]
    fn test_moo_heads_dim_mismatch() {
        let m = MooHeads::new(4, 8, vec![2]);
        assert!(m.forward(&[0.5, -0.3]).is_err());
    }

    #[test]
    fn test_linear_scalarization() {
        let r = LinearScalarization::compute_loss(&[1.0, 2.0, 3.0], &[0.5, 0.3, 0.2])
            .expect("test value");
        assert!((r - 1.7).abs() < 1e-10);
    }

    #[test]
    fn test_tchebycheff_loss() {
        let r = TchebycheffScalarization::compute_loss(&[0.5, 1.0], &[0.5, 0.5], &[0.0, 0.0])
            .expect("test value");
        assert!((r - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_weight_sampling_sum_to_1() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..10 {
            let w = LinearScalarization::sample_weights(5, &mut rng);
            let s: f64 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "sum = {}", s);
            for &wi in &w {
                assert!(wi >= 0.0);
            }
        }
    }

    #[test]
    fn test_pfl_layer() {
        let mut pfl = PFLLayer::new(77);
        let (w, s) = pfl.step(&[0.8, 0.4, 0.6]).expect("test value");
        assert_eq!(w.len(), 3);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(s.is_finite());
        assert_eq!(pfl.pareto_approx.len(), 1);
    }

    // ─── Constrained Optimization ──────────────────────────────────────────────

    #[test]
    fn test_augmented_lagrangian_step() {
        let (aug, nl) =
            AugmentedLagrangian::step(2.0, &[0.5, -0.1], &[1.0, 1.0], 10.0).expect("test value");
        assert!((aug - 3.70).abs() < 1e-9);
        assert!((nl[0] - 6.0).abs() < 1e-9);
        assert!((nl[1] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_interior_point_barrier() {
        let b = InteriorPointMethod::barrier(&[-1.0, -2.0], 1.0);
        let exp = -(1.0_f64.ln()) - (2.0_f64.ln());
        assert!((b - exp).abs() < 1e-10);
    }

    #[test]
    fn test_interior_point_infeasible() {
        assert!(InteriorPointMethod::barrier(&[-1.0, 0.5], 1.0).is_infinite());
    }

    #[test]
    fn test_project_box() {
        let p = ProjectedGradient::project_box(&[-1.0, 0.5, 2.0], &[0.0; 3], &[1.0; 3])
            .expect("test value");
        assert_eq!(p, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_project_simplex() {
        let p = ProjectedGradient::project_simplex(&[0.4, 0.6, 0.1]).expect("test value");
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-10);
        for &pi in &p {
            assert!(pi >= -1e-12);
        }
    }

    #[test]
    fn test_project_simplex_negative_input() {
        let p = ProjectedGradient::project_simplex(&[-0.5, 1.5, 0.5]).expect("test value");
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-10);
        for &pi in &p {
            assert!(pi >= -1e-12);
        }
    }

    #[test]
    fn test_penalty_method() {
        let l = PenaltyMethod::loss(1.0, &[0.5, -0.2], 4.0);
        assert!((l - 1.5).abs() < 1e-10);
    }

    // ─── Evolutionary ──────────────────────────────────────────────────────────

    #[test]
    fn test_sbx_crossover() {
        let mut rng = StdRng::seed_from_u64(1);
        let (c1, c2) =
            GeneticOperators::sbx_crossover(&[0.0, 1.0, 0.5], &[1.0, 0.0, 0.5], 20.0, &mut rng);
        assert_eq!(c1.len(), 3);
        assert_eq!(c2.len(), 3);
        for (&a, &b) in c1.iter().zip(c2.iter()) {
            assert!(a.is_finite());
            assert!(b.is_finite());
        }
    }

    #[test]
    fn test_polynomial_mutation() {
        let bnd: Vec<(f64, f64)> = vec![(0.0, 1.0); 3];
        let mut rng = StdRng::seed_from_u64(2);
        let m = GeneticOperators::polynomial_mutation(&[0.5, 0.5, 0.5], 20.0, &bnd, &mut rng);
        assert_eq!(m.len(), 3);
        for (&mi, &(lb, ub)) in m.iter().zip(bnd.iter()) {
            assert!(mi >= lb && mi <= ub);
        }
    }

    #[test]
    fn test_reference_point_sampler_count() {
        let pts = ReferencePointSampler::sample(3, 4).expect("test value");
        assert_eq!(pts.len(), 15);
    }

    #[test]
    fn test_tournament_selection() {
        let pop: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let fit: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let mut rng = StdRng::seed_from_u64(3);
        let idx = SelectionOperator::tournament(&pop, &fit, 3, &mut rng).expect("test value");
        assert!(idx < 10);
    }

    #[test]
    fn test_sms_emoa_selection() {
        let emoa = SmsEmoa::new(vec![5.0, 5.0]);
        let objs = vec![vec![1.0, 4.0], vec![2.0, 2.0], vec![3.0, 1.0]];
        let pop = objs.clone();
        let r = emoa.select(&pop, &objs).expect("test value");
        assert!(r < 3);
    }

    // ─── MTL ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_gradnorm_weights() {
        let w = GradNormOptimizer::compute_task_weights(
            &[1.0, 2.0, 3.0],
            &[
                vec![0.1, 0.2, 0.3],
                vec![0.4, 0.5, 0.6],
                vec![0.1, 0.1, 0.1],
            ],
            1.5,
            1.0,
        )
        .expect("test value");
        assert_eq!(w.len(), 3);
        let s: f64 = w.iter().sum();
        assert!((s - 3.0).abs() < 1e-9, "sum = {}", s);
        for &wi in &w {
            assert!(wi >= 0.0);
        }
    }

    #[test]
    fn test_pcgrad_orthogonal() {
        let avg = PcGrad::project(&[vec![1.0, 0.0], vec![-1.0, 0.0]]).expect("test value");
        let norm: f64 = avg.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm < 0.1, "norm = {}", norm);
    }

    #[test]
    fn test_pcgrad_non_conflicting() {
        let avg = PcGrad::project(&[vec![1.0, 0.0], vec![0.5, 0.0]]).expect("test value");
        assert!(avg[0] > 0.0);
    }

    #[test]
    fn test_cagrad_computation() {
        let r = CaGrad::compute(&[vec![1.0, 0.0], vec![0.0, 1.0]], 0.5).expect("test value");
        assert_eq!(r.len(), 2);
        assert!(r.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_imtl_g_unit() {
        let r = ImtlG::compute(&[vec![3.0, 0.0], vec![0.0, 4.0]]).expect("test value");
        assert!((r[0] - 0.5).abs() < 1e-10);
        assert!((r[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_uncertainty_weighting() {
        let r = UncertaintyWeighting::compute_loss(&[1.0, 2.0], &[0.0, 0.0]).expect("test value");
        assert!((r - 1.5).abs() < 1e-9, "got {}", r);
    }

    #[test]
    fn test_uncertainty_weighting_update() {
        let u = UncertaintyWeighting::update_sigmas(&[0.0, 0.0], &[1.0, 2.0], 0.01)
            .expect("test value");
        assert_eq!(u.len(), 2);
        // grad_0 = -exp(0)*1 + 1 = 0 => ls_0 = 0
        assert!((u[0] - 0.0).abs() < 1e-10);
        // grad_1 = -exp(0)*2 + 1 = -1 => ls_1 = 0 - 0.01*(-1) = 0.01
        assert!((u[1] - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_pareto_moe_forward() {
        let m = ParetoOptimalMoe::new(4, 8, 3, 2);
        let out = m
            .forward(&[0.1, 0.2, 0.3, 0.4], &[0.5, 0.5])
            .expect("test value");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_nsga3_evolve() {
        let nsga = NsgaIii::new(2, 4);
        let pop = vec![
            vec![0.1, 0.9],
            vec![0.5, 0.5],
            vec![0.9, 0.1],
            vec![0.3, 0.7],
        ];
        assert!(!nsga.evolve(&pop, &[]).expect("test value").is_empty());
    }

    #[test]
    fn test_moead_initialize() {
        let m = Moead::initialize(10, 2, 3);
        assert_eq!(m.n_solutions, 10);
        assert_eq!(m.population.len(), 10);
    }

    #[test]
    fn test_moead_evolve_step() {
        let mut m = Moead::initialize(6, 2, 2);
        let mut rng = StdRng::seed_from_u64(42);
        let p = m.evolve_step(|x| x.to_vec(), &mut rng);
        assert_eq!(p.len(), 6);
    }

    #[test]
    fn test_feasibility_pump() {
        let r = FeasibilityPump::pump_step(&[1.4, 2.7, -0.2, 3.5], &[true, true, false, true])
            .expect("test value");
        assert_eq!(r, vec![1.0, 3.0, -0.2, 4.0]);
    }

    #[test]
    fn test_roulette_selection() {
        let mut rng = StdRng::seed_from_u64(7);
        let idx = SelectionOperator::roulette(&[1.0, 3.0, 2.0], &mut rng).expect("test value");
        assert!(idx < 3);
    }

    #[test]
    fn test_rank_select() {
        let mut rng = StdRng::seed_from_u64(8);
        let idx = SelectionOperator::rank_select(&[0.5, 0.1, 0.9], &mut rng).expect("test value");
        assert!(idx < 3);
    }
}
