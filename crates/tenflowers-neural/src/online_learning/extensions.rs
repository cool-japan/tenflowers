//! Extensions for online learning: streaming data structures, online evaluation metrics.

use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Streaming Data Structures
// ─────────────────────────────────────────────────────────────────────────────

/// Count-Min Sketch — approximate frequency counting.
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    pub depth: usize,
    pub width: usize,
    pub table: Vec<Vec<u64>>,
    seeds: Vec<u64>,
}

impl CountMinSketch {
    pub fn new(depth: usize, width: usize) -> Self {
        let seeds: Vec<u64> = (0..depth as u64).map(|i| i * 2654435761 + 1).collect();
        Self {
            depth,
            width,
            table: vec![vec![0u64; width]; depth],
            seeds,
        }
    }

    fn hash(&self, key: usize, row: usize) -> usize {
        let h = (key as u64)
            .wrapping_mul(self.seeds[row])
            .wrapping_add(self.seeds[row] >> 32);
        (h as usize) % self.width
    }

    pub fn add(&mut self, key: usize) {
        for row in 0..self.depth {
            let col = self.hash(key, row);
            self.table[row][col] += 1;
        }
    }

    pub fn estimate(&self, key: usize) -> u64 {
        (0..self.depth)
            .map(|row| {
                let col = self.hash(key, row);
                self.table[row][col]
            })
            .min()
            .unwrap_or(0)
    }
}

/// HyperLogLog — approximate cardinality with 256 registers (b=8).
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    pub m: usize,
    registers: Vec<u8>,
    alpha: f64,
}

impl HyperLogLog {
    pub fn new() -> Self {
        let m = 1usize << 8;
        Self {
            m,
            registers: vec![0u8; m],
            alpha: 0.7213 / (1.0 + 1.079 / m as f64),
        }
    }

    /// splitmix64 hash finalizer.
    fn hash64(x: u64) -> u64 {
        let mut h = x.wrapping_add(0x9e3779b97f4a7c15u64);
        h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9u64);
        h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111ebu64);
        h ^ (h >> 31)
    }

    /// ρ(w) = leading_zeros(w) + 1, capped at 57.
    fn rho(w: u64) -> u8 {
        (w.leading_zeros() as u8 + 1).min(57)
    }

    pub fn add(&mut self, x: u64) {
        let h = Self::hash64(x);
        let idx = (h >> 56) as usize;
        let w = h << 8;
        let r = Self::rho(w);
        if r > self.registers[idx] {
            self.registers[idx] = r;
        }
    }

    pub fn count(&self) -> f64 {
        let m = self.m as f64;
        let z: f64 = self
            .registers
            .iter()
            .map(|&r| 2.0_f64.powi(-(r as i32)))
            .sum();
        let raw = self.alpha * m * m / z;
        if raw <= 2.5 * m {
            let zeros = self.registers.iter().filter(|&&r| r == 0).count() as f64;
            if zeros > 0.0 {
                return m * (m / zeros).ln();
            }
        }
        raw
    }
}

impl Default for HyperLogLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Bloom Filter for probabilistic set membership.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    bits: Vec<u64>,
    n_bits: usize,
    k: usize,
}

impl BloomFilter {
    pub fn new(n_bits: usize, k: usize) -> Self {
        Self {
            bits: vec![0u64; (n_bits + 63) / 64],
            n_bits,
            k,
        }
    }

    fn hash_i(&self, x: u64, i: usize) -> usize {
        let h = x
            .wrapping_mul(6364136223846793005u64.wrapping_add(i as u64 * 1442695040888963407u64))
            .wrapping_add(1442695040888963407u64);
        (h as usize) % self.n_bits
    }

    fn set_bit(&mut self, pos: usize) {
        self.bits[pos / 64] |= 1 << (pos % 64);
    }
    fn get_bit(&self, pos: usize) -> bool {
        (self.bits[pos / 64] >> (pos % 64)) & 1 == 1
    }

    pub fn insert(&mut self, x: u64) {
        for i in 0..self.k {
            let pos = self.hash_i(x, i);
            self.set_bit(pos);
        }
    }

    /// Membership test (may return false positives).
    pub fn contains(&self, x: u64) -> bool {
        (0..self.k).all(|i| self.get_bit(self.hash_i(x, i)))
    }
}

/// Reservoir Sampler (Algorithm R — Vitter, 1985).
#[derive(Debug, Clone)]
pub struct ReservoirSampler {
    pub k: usize,
    reservoir: Vec<f64>,
}

impl ReservoirSampler {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            reservoir: Vec::with_capacity(k),
        }
    }

    /// Push item at 0-based `step`; replaces random entry with prob k/(step+1).
    pub fn push(&mut self, item: f64, step: usize, rng: &mut StdRng) {
        if self.reservoir.len() < self.k {
            self.reservoir.push(item);
        } else {
            let j = rng.random_range(0.0..=(step as f64)) as usize;
            if j < self.k {
                self.reservoir[j] = item;
            }
        }
    }

    pub fn sample(&self) -> &[f64] {
        &self.reservoir
    }
}

/// Exponential Histogram — approximate quantiles over a sliding count window.
#[derive(Debug, Clone)]
pub struct ExponentialHistogram {
    pub epsilon: f64,
    pub window: usize,
    buckets: Vec<(f64, usize)>,
    total_count: usize,
}

impl ExponentialHistogram {
    pub fn new(window: usize, epsilon: f64) -> Self {
        Self {
            epsilon: epsilon.max(1e-6),
            window,
            buckets: Vec::new(),
            total_count: 0,
        }
    }

    pub fn insert(&mut self, x: f64) {
        self.buckets.push((x, 1));
        self.total_count += 1;
        while self.total_count > self.window {
            let (_, w) = self.buckets.remove(0);
            self.total_count -= w;
        }
        self.merge_if_needed();
    }

    pub fn count(&self) -> usize {
        self.total_count
    }

    pub fn merge_if_needed(&mut self) {
        let max_buckets = (1.0 / self.epsilon).ceil() as usize + 2;
        while self.buckets.len() > max_buckets {
            let mut min_idx = 0;
            let mut min_w = usize::MAX;
            for i in 0..self.buckets.len() - 1 {
                let combined = self.buckets[i].1 + self.buckets[i + 1].1;
                if combined < min_w {
                    min_w = combined;
                    min_idx = i;
                }
            }
            let (v1, w1) = self.buckets[min_idx];
            let (v2, w2) = self.buckets[min_idx + 1];
            self.buckets[min_idx] = (
                (v1 * w1 as f64 + v2 * w2 as f64) / (w1 + w2) as f64,
                w1 + w2,
            );
            self.buckets.remove(min_idx + 1);
        }
    }

    pub fn quantile(&self, q: f64) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }
        let target = q * self.total_count as f64;
        let mut cumulative = 0.0_f64;
        for (val, w) in &self.buckets {
            cumulative += *w as f64;
            if cumulative >= target {
                return *val;
            }
        }
        self.buckets.last().map(|(v, _)| *v).unwrap_or(0.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Online Evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Incremental accuracy/precision/recall/F1 tracker (binary classification).
#[derive(Debug, Clone, Default)]
pub struct OnlineMetricsTracker {
    tp: usize,
    tn: usize,
    fp: usize,
    fn_: usize,
    pub threshold: f64,
}

impl OnlineMetricsTracker {
    pub fn new() -> Self {
        Self {
            tp: 0,
            tn: 0,
            fp: 0,
            fn_: 0,
            threshold: 0.5,
        }
    }

    pub fn update(&mut self, pred: f64, label: f64) {
        let predicted_pos = pred >= self.threshold;
        let actual_pos = label >= self.threshold;
        match (predicted_pos, actual_pos) {
            (true, true) => self.tp += 1,
            (true, false) => self.fp += 1,
            (false, true) => self.fn_ += 1,
            (false, false) => self.tn += 1,
        }
    }

    pub fn accuracy(&self) -> f64 {
        let total = self.tp + self.tn + self.fp + self.fn_;
        if total == 0 {
            0.0
        } else {
            (self.tp + self.tn) as f64 / total as f64
        }
    }

    pub fn precision(&self) -> f64 {
        let d = self.tp + self.fp;
        if d == 0 {
            0.0
        } else {
            self.tp as f64 / d as f64
        }
    }

    pub fn recall(&self) -> f64 {
        let d = self.tp + self.fn_;
        if d == 0 {
            0.0
        } else {
            self.tp as f64 / d as f64
        }
    }

    pub fn f1(&self) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        if p + r < 1e-12 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    pub fn n(&self) -> usize {
        self.tp + self.tn + self.fp + self.fn_
    }
}

/// Online Cohen's κ for multi-class classification.
#[derive(Debug, Clone)]
pub struct PraquenKappa {
    pub n_classes: usize,
    confusion: Vec<usize>,
    total: usize,
}

impl PraquenKappa {
    pub fn new(n_classes: usize) -> Self {
        Self {
            n_classes,
            confusion: vec![0usize; n_classes * n_classes],
            total: 0,
        }
    }

    pub fn update(&mut self, pred_class: usize, true_class: usize, n_classes: usize) {
        let nc = n_classes.max(self.n_classes);
        if nc > self.n_classes {
            let new_conf = vec![0usize; nc * nc];
            let mut resized = new_conf;
            for i in 0..self.n_classes {
                for j in 0..self.n_classes {
                    resized[i * nc + j] = self.confusion[i * self.n_classes + j];
                }
            }
            self.confusion = resized;
            self.n_classes = nc;
        }
        let tc = true_class.min(self.n_classes - 1);
        let pc = pred_class.min(self.n_classes - 1);
        self.confusion[tc * self.n_classes + pc] += 1;
        self.total += 1;
    }

    pub fn kappa(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let n = self.total as f64;
        let nc = self.n_classes;
        let p_o: f64 = (0..nc)
            .map(|i| self.confusion[i * nc + i] as f64)
            .sum::<f64>()
            / n;
        let p_e: f64 = (0..nc)
            .map(|k| {
                let row_sum: f64 = (0..nc).map(|j| self.confusion[k * nc + j] as f64).sum();
                let col_sum: f64 = (0..nc).map(|i| self.confusion[i * nc + k] as f64).sum();
                (row_sum / n) * (col_sum / n)
            })
            .sum();
        if (1.0 - p_e).abs() < 1e-12 {
            return 1.0;
        }
        (p_o - p_e) / (1.0 - p_e)
    }
}

/// Wilson-score confidence interval tracker.
#[derive(Debug, Clone, Default)]
pub struct ConfidenceIntervalTracker {
    successes: usize,
    trials: usize,
}

impl ConfidenceIntervalTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, correct: bool) {
        self.trials += 1;
        if correct {
            self.successes += 1;
        }
    }

    /// Wilson-score CI at `1 − alpha` confidence. Returns `(lower, upper)`.
    pub fn ci(&self, alpha: f64) -> (f64, f64) {
        if self.trials == 0 {
            return (0.0, 1.0);
        }
        let n = self.trials as f64;
        let p_hat = self.successes as f64 / n;
        let z = normal_quantile(1.0 - alpha / 2.0);
        let z2 = z * z;
        let denom = 1.0 + z2 / n;
        let center = (p_hat + z2 / (2.0 * n)) / denom;
        let half = z / denom * (p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)).sqrt();
        ((center - half).max(0.0), (center + half).min(1.0))
    }

    pub fn proportion(&self) -> f64 {
        if self.trials == 0 {
            0.0
        } else {
            self.successes as f64 / self.trials as f64
        }
    }
}

/// Normal quantile (Abramowitz & Stegun §26.2.17, max error ~4.5e-4).
pub(crate) fn normal_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-10, 1.0 - 1e-10);
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };
    let c = [2.515517, 0.802853, 0.010328];
    let d = [1.432788, 0.189269, 0.001308];
    let num = c[0] + c[1] * t + c[2] * t * t;
    let den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t;
    let z = t - num / den;
    if p < 0.5 {
        -z
    } else {
        z
    }
}

/// Online AUC via Mann-Whitney U statistic.
#[derive(Debug, Clone, Default)]
pub struct OnlineRocAuc {
    pairs: Vec<(f64, f64)>,
}

impl OnlineRocAuc {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, score: f64, label: f64) {
        let pos = self.pairs.binary_search_by(|(s, _)| {
            score
                .partial_cmp(s)
                .unwrap_or(std::cmp::Ordering::Equal)
                .reverse()
        });
        let idx = match pos {
            Ok(i) | Err(i) => i,
        };
        self.pairs.insert(idx, (score, label));
    }

    /// AUC via Mann-Whitney U (exact O(n_pos × n_neg)).
    pub fn auc(&self) -> f64 {
        if self.pairs.is_empty() {
            return 0.5;
        }
        let pos: Vec<f64> = self
            .pairs
            .iter()
            .filter(|(_, l)| *l >= 0.5)
            .map(|(s, _)| *s)
            .collect();
        let neg: Vec<f64> = self
            .pairs
            .iter()
            .filter(|(_, l)| *l < 0.5)
            .map(|(s, _)| *s)
            .collect();
        if pos.is_empty() || neg.is_empty() {
            return 0.5;
        }
        let mut wins = 0.0_f64;
        for &ps in &pos {
            for &ns in &neg {
                if ps > ns {
                    wins += 1.0;
                } else if (ps - ns).abs() < 1e-15 {
                    wins += 0.5;
                }
            }
        }
        (wins / (pos.len() as f64 * neg.len() as f64)).clamp(0.0, 1.0)
    }
}

/// Cumulative and average regret tracker.
#[derive(Debug, Clone, Default)]
pub struct CumulativeRegretTracker {
    cumulative: f64,
    rounds: usize,
}

impl CumulativeRegretTracker {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn update(&mut self, reward: f64, best_reward: f64) {
        self.cumulative += best_reward - reward;
        self.rounds += 1;
    }
    pub fn cumulative_regret(&self) -> f64 {
        self.cumulative
    }
    pub fn average_regret(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            self.cumulative / self.rounds as f64
        }
    }
    pub fn rounds(&self) -> usize {
        self.rounds
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (ported from original online_learning.rs)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::online_learning::{
        Adwin, DdmDetector, DriftDetected, DriftStatus, EpsilonGreedy, FollowTheRegularizedLeader,
        KsTest, LinUcb, LrSchedule, NeuralBandit, OnlineAdaGrad, OnlineAdam, OnlineLbfgs,
        OnlineSgd, PageHinkleyTest, ThompsonSampling, Ucb1, WindowedStatistics,
    };
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_online_sgd_step() {
        let mut sgd = OnlineSgd::new(0.1);
        let updated = sgd.update(&[1.0, 2.0, 3.0], &[0.5, -0.5, 1.0], 1).expect("test value");
        assert!((updated[0] - 0.95).abs() < 1e-9);
        assert!((updated[1] - 2.05).abs() < 1e-9);
        assert!((updated[2] - 2.90).abs() < 1e-9);
    }

    #[test]
    fn test_online_sgd_with_momentum() {
        let mut sgd = OnlineSgd::with_schedule(LrSchedule::Constant(0.1), 0.9);
        let step1 = sgd.update(&[1.0, 1.0], &[1.0, 1.0], 1).expect("test value");
        let step2 = sgd.update(&step1, &[1.0, 1.0], 2).expect("test value");
        assert!(step2[0] < step1[0]);
    }

    #[test]
    fn test_online_adagrad_decay() {
        let adagrad = OnlineAdaGrad::new(0.1);
        let (new_params, new_acc) = adagrad
            .update(&[1.0, 2.0], &[2.0, 4.0], &[0.0, 0.0])
            .expect("test value");
        assert!((new_acc[0] - 4.0).abs() < 1e-9);
        assert!((new_acc[1] - 16.0).abs() < 1e-9);
        assert!((new_params[0] - (1.0 - 0.1 / (4.0_f64 + 1e-8).sqrt() * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_online_adam_bias_correct() {
        let adam = OnlineAdam::new(0.01);
        let (new_params, _, _) = adam
            .update(&[0.0, 0.0], &[1.0, -1.0], &[0.0, 0.0], &[0.0, 0.0], 1)
            .expect("test value");
        assert!((new_params[0] - (-0.01)).abs() < 1e-4);
        assert!((new_params[1] - 0.01).abs() < 1e-4);
    }

    #[test]
    fn test_ftrl_update() {
        let ftrl = FollowTheRegularizedLeader::new();
        let (_, _, new_n) = ftrl
            .update(
                &[1.0, -1.0],
                &[0.5, 0.5],
                &[0.0, 0.0],
                &[0.0, 0.0],
                1.0,
                1.0,
                0.1,
                0.0,
            )
            .expect("test value");
        assert!((new_n[0] - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_ftrl_l1_sparsity() {
        let ftrl = FollowTheRegularizedLeader::new();
        let (new_params, _, _) = ftrl
            .update(&[0.0], &[0.05], &[0.05], &[0.0025], 1.0, 1.0, 0.1, 0.0)
            .expect("test value");
        assert!(new_params[0].abs() < 0.01);
    }

    #[test]
    fn test_online_lbfgs() {
        let mut lbfgs = OnlineLbfgs::new(5, 0.1);
        let params = vec![2.0, 3.0];
        let updated = lbfgs.update(&params, &[2.0, 3.0]).expect("test value");
        assert!(updated[0] < params[0]);
        assert!(updated[1] < params[1]);
    }

    #[test]
    fn test_online_lbfgs_uses_history() {
        let mut lbfgs = OnlineLbfgs::new(3, 0.1);
        let mut p = vec![5.0, 5.0];
        for _ in 0..5 {
            let g = vec![p[0], p[1]];
            p = lbfgs.update(&p, &g).expect("test value");
        }
        let norm = (p[0] * p[0] + p[1] * p[1]).sqrt();
        assert!(norm < 5.0 * std::f64::consts::SQRT_2);
    }

    #[test]
    fn test_epsilon_greedy_explore() {
        let eg = EpsilonGreedy::new();
        let mut rng = StdRng::seed_from_u64(42);
        assert!(eg.select_arm(&[1.0, 2.0, 3.0], 1.0, &mut rng) < 3);
    }

    #[test]
    fn test_epsilon_greedy_exploit() {
        let eg = EpsilonGreedy::new();
        let mut rng = StdRng::seed_from_u64(42);
        assert_eq!(eg.select_arm(&[1.0, 2.0, 10.0], 0.0, &mut rng), 2);
    }

    #[test]
    fn test_epsilon_greedy_update() {
        let eg = EpsilonGreedy::new();
        let (new_q, new_c) = eg.update(1, 1.0, &[0.0; 3], &[0usize; 3]).expect("test value");
        assert_eq!(new_c[1], 1);
        assert!((new_q[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ucb1_optimism() {
        let ucb = Ucb1::new();
        assert_eq!(ucb.select_arm(&[0.5, 0.5, 0.5], &[10, 10, 0], 20), 2);
    }

    #[test]
    fn test_ucb1_compute_ucb() {
        let ucb = Ucb1::new();
        let score = ucb.compute_ucb(0.5, 10, 100);
        assert!((score - (0.5 + (2.0 * 100.0_f64.ln() / 10.0).sqrt())).abs() < 1e-9);
    }

    #[test]
    fn test_thompson_sampling_update() {
        let ts = ThompsonSampling::new();
        let (na, nb) = ts
            .update(1, 1.0, &[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0])
            .expect("test value");
        assert!((na[1] - 2.0).abs() < 1e-9);
        let (_, nb2) = ts.update(0, 0.0, &na, &nb).expect("test value");
        assert!((nb2[0] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_thompson_sampling_explore() {
        let ts = ThompsonSampling::new();
        let mut rng = StdRng::seed_from_u64(7);
        let count = (0..100)
            .filter(|_| ts.sample(&[1.0, 1.0, 100.0], &[1.0, 1.0, 1.0], &mut rng) == 2)
            .count();
        assert!(count > 70);
    }

    #[test]
    fn test_lin_ucb_update() {
        let linucb = LinUcb::new(2, 3, 1.0);
        let (d, n) = (3, 2);
        let mut a_inv = vec![0.0_f64; n * d * d];
        for a in 0..n {
            for i in 0..d {
                a_inv[a * d * d + i * d + i] = 1.0;
            }
        }
        let mut b = vec![0.0_f64; n * d];
        linucb
            .update(&[1.0, 0.0, 0.0], 0, 1.0, &mut a_inv, &mut b)
            .expect("test value");
        assert!((b[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_lin_ucb_select() {
        let linucb = LinUcb::new(2, 2, 1.5);
        let (d, n) = (2, 2);
        let mut a_inv = vec![0.0_f64; n * d * d];
        for a in 0..n {
            for i in 0..d {
                a_inv[a * d * d + i * d + i] = 1.0;
            }
        }
        let mut b = vec![0.0_f64; n * d];
        linucb
            .update(&[0.0, 1.0], 1, 5.0, &mut a_inv, &mut b)
            .expect("test value");
        assert_eq!(
            linucb.select_arm(&[vec![0.0, 1.0], vec![0.0, 1.0]], &a_inv, &b, 0.0),
            1
        );
    }

    #[test]
    fn test_neural_bandit() {
        let mut nb = NeuralBandit::new(3, 4, 8, 0);
        let ctx = vec![1.0, 0.0, -1.0, 0.5];
        let mut rng = StdRng::seed_from_u64(99);
        let arm = nb.select(&ctx, 0.0, &mut rng);
        assert!(arm < 3);
        nb.update(arm, &ctx, 1.0, 0.01).expect("test value");
    }

    #[test]
    fn test_page_hinkley_detect() {
        let mut ph = PageHinkleyTest::new(50.0, 0.005);
        for _ in 0..100 {
            assert_eq!(ph.update(1.0), DriftDetected::None);
        }
        ph.reset();
        for _ in 0..200 {
            ph.update(0.0);
        }
        let mut detected = DriftDetected::None;
        for _ in 0..500 {
            detected = ph.update(5.0);
            if detected != DriftDetected::None {
                break;
            }
        }
        assert_ne!(detected, DriftDetected::None);
    }

    #[test]
    fn test_adwin_mean_update() {
        let mut adwin = Adwin::new(0.002, 200);
        for _ in 0..51 {
            adwin.update(1.0);
        }
        let (_, m) = adwin.update(1.0);
        assert!((m - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_adwin_detects_shift() {
        let mut adwin = Adwin::new(0.005, 300);
        for _ in 0..100 {
            adwin.update(0.0);
        }
        let any_drift = (0..100).any(|_| adwin.update(100.0).0);
        assert!(any_drift);
    }

    #[test]
    fn test_ddm_detector() {
        let mut ddm = DdmDetector::new();
        for _ in 0..100 {
            assert_ne!(ddm.update(false), DriftStatus::Drift);
        }
        let saw_drift = (0..300).any(|_| ddm.update(true) == DriftStatus::Drift);
        assert!(saw_drift);
    }

    #[test]
    fn test_ks_test_same_dist() {
        let ks = KsTest::new();
        let s1: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        assert!(ks.test(&s1, &s1.clone()) < 1e-9);
    }

    #[test]
    fn test_windowed_stats() {
        let mut ws = WindowedStatistics::new(5);
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
            ws.push(x);
        }
        assert!((ws.mean() - 3.0).abs() < 1e-9);
        assert!((ws.variance() - 2.0).abs() < 1e-9);
        ws.push(6.0);
        assert!((ws.mean() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_count_min_sketch_estimate() {
        let mut cms = CountMinSketch::new(4, 1024);
        for _ in 0..100 {
            cms.add(42);
        }
        for _ in 0..50 {
            cms.add(7);
        }
        assert!(cms.estimate(42) >= 100);
        assert!(cms.estimate(7) >= 50);
    }

    #[test]
    fn test_hyperloglog_cardinality() {
        let mut hll = HyperLogLog::new();
        for i in 0u64..1000 {
            hll.add(i);
        }
        let est = hll.count();
        assert!(
            est > 700.0 && est < 1300.0,
            "HLL estimate {est} should be near 1000"
        );
    }

    #[test]
    fn test_bloom_filter_insert() {
        let mut bf = BloomFilter::new(4096, 4);
        bf.insert(42);
        bf.insert(100);
        assert!(bf.contains(42));
        assert!(bf.contains(100));
    }

    #[test]
    fn test_reservoir_sampler() {
        let mut rs = ReservoirSampler::new(10);
        let mut rng = StdRng::seed_from_u64(0);
        for i in 0..100 {
            rs.push(i as f64, i, &mut rng);
        }
        let sample = rs.sample();
        assert_eq!(sample.len(), 10);
        assert!(sample.iter().all(|&v| (0.0..100.0).contains(&v)));
    }

    #[test]
    fn test_exp_histogram() {
        let mut eh = ExponentialHistogram::new(100, 0.1);
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
            eh.insert(x);
        }
        assert_eq!(eh.count(), 5);
        let med = eh.quantile(0.5);
        assert!((1.0..=5.0).contains(&med));
    }

    #[test]
    fn test_online_metrics_accuracy() {
        let mut t = OnlineMetricsTracker::new();
        for _ in 0..8 {
            t.update(0.9, 1.0);
        }
        for _ in 0..2 {
            t.update(0.9, 0.0);
        }
        assert!((t.accuracy() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_online_f1() {
        let mut t = OnlineMetricsTracker::new();
        for _ in 0..4 {
            t.update(0.9, 1.0);
        }
        t.update(0.9, 0.0);
        t.update(0.1, 1.0);
        let expected = 2.0 * (4.0 / 5.0) * (4.0 / 5.0) / (4.0 / 5.0 + 4.0 / 5.0);
        assert!((t.f1() - expected).abs() < 1e-9);
    }

    #[test]
    fn test_confidence_interval() {
        let mut ci = ConfidenceIntervalTracker::new();
        for _ in 0..100 {
            ci.update(true);
        }
        let (lo, hi) = ci.ci(0.05);
        assert!(lo > 0.9 && hi <= 1.0);
    }

    #[test]
    fn test_online_auc() {
        let mut auc = OnlineRocAuc::new();
        for _ in 0..10 {
            auc.update(0.9, 1.0);
            auc.update(0.1, 0.0);
        }
        assert!(
            auc.auc() > 0.95,
            "AUC for perfect ranking should be ~1.0, got {}",
            auc.auc()
        );
    }

    #[test]
    fn test_online_auc_random() {
        let mut auc = OnlineRocAuc::new();
        for i in 0..20 {
            let (label, score) = if i % 2 == 0 { (1.0, 0.1) } else { (0.0, 0.9) };
            auc.update(score, label);
        }
        assert!(
            auc.auc() < 0.2,
            "AUC for inverted ranking should be ~0, got {}",
            auc.auc()
        );
    }

    #[test]
    fn test_regret_tracker() {
        let mut t = CumulativeRegretTracker::new();
        t.update(0.8, 1.0);
        t.update(0.6, 1.0);
        t.update(1.0, 1.0);
        assert!((t.cumulative_regret() - 0.6).abs() < 1e-9);
        assert!((t.average_regret() - 0.2).abs() < 1e-9);
    }
}
