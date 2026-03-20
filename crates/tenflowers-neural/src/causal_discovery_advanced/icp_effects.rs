//! ICP (Invariant Causal Prediction) + CausalEffect estimation.

use super::shared::{mean_f32, normal_cdf_f32, ols_residuals, variance_f32};

// ─────────────────────────────────────────────────────────────────────────────
// 8. Invariant Causal Prediction (ICP)
// ─────────────────────────────────────────────────────────────────────────────

/// A dataset from a single experimental environment.
#[derive(Debug, Clone)]
pub struct CdEnvironment {
    /// Design matrix (n_samples × n_vars).
    pub x: Vec<Vec<f32>>,
    /// Response variable.
    pub y: Vec<f32>,
}

/// Compute OLS residuals for `features` subset.
pub fn regression_residuals(x_mat: &[Vec<f32>], y: &[f32], features: &[usize]) -> Vec<f32> {
    ols_residuals(x_mat, y, features)
}

/// Test for invariance of the conditional distribution across two environments.
///
/// Uses Levene's test for equal variance and Welch's t-test for equal mean
/// on the regression residuals.  Returns `true` if invariant (p > α).
pub fn is_invariant(
    env1: &CdEnvironment,
    env2: &CdEnvironment,
    features: &[usize],
    alpha: f32,
) -> bool {
    let res1 = regression_residuals(&env1.x, &env1.y, features);
    let res2 = regression_residuals(&env2.x, &env2.y, features);
    let n1 = res1.len();
    let n2 = res2.len();
    if n1 < 2 || n2 < 2 {
        return false;
    }

    let m1 = mean_f32(&res1);
    let m2 = mean_f32(&res2);
    let v1 = variance_f32(&res1).max(1e-12);
    let v2 = variance_f32(&res2).max(1e-12);

    // Welch's t-test
    let se = (v1 / n1 as f32 + v2 / n2 as f32).sqrt();
    if se < 1e-12 {
        return true;
    }
    let t_stat = (m1 - m2).abs() / se;
    // Approximate p-value using normal approximation (conservative)
    let p_t = 2.0 * (1.0 - normal_cdf_f32(t_stat));

    // Levene's test (Brown-Forsythe variant using medians)
    let med1 = {
        let mut s = res1.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if n1 % 2 == 0 {
            (s[n1 / 2 - 1] + s[n1 / 2]) / 2.0
        } else {
            s[n1 / 2]
        }
    };
    let med2 = {
        let mut s = res2.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if n2 % 2 == 0 {
            (s[n2 / 2 - 1] + s[n2 / 2]) / 2.0
        } else {
            s[n2 / 2]
        }
    };

    let z1: Vec<f32> = res1.iter().map(|&r| (r - med1).abs()).collect();
    let z2: Vec<f32> = res2.iter().map(|&r| (r - med2).abs()).collect();
    let zm1 = mean_f32(&z1);
    let zm2 = mean_f32(&z2);
    let vz1 = variance_f32(&z1).max(1e-12);
    let vz2 = variance_f32(&z2).max(1e-12);
    let se_lev = (vz1 / n1 as f32 + vz2 / n2 as f32).sqrt();
    let l_stat = if se_lev < 1e-12 {
        0.0
    } else {
        (zm1 - zm2).abs() / se_lev
    };
    let p_lev = 2.0 * (1.0 - normal_cdf_f32(l_stat));

    // Both tests must not reject
    p_t > alpha && p_lev > alpha
}

/// ICP: find the intersection of all parent sets that are invariant across environments.
///
/// Returns the intersection of feature subsets S such that the conditional
/// distribution of y | X_S is invariant across all pairs of environments.
pub fn icp_find_parents(envs: &[CdEnvironment], n_vars: usize) -> Vec<usize> {
    if envs.len() < 2 || n_vars == 0 {
        return vec![];
    }
    let alpha = 0.05f32;

    // Enumerate all subsets of size 0..=min(n_vars, 4) (restrict to keep tractable)
    let max_size = n_vars.min(4);
    let vars: Vec<usize> = (0..n_vars).collect();

    let mut valid_sets: Vec<Vec<usize>> = Vec::new();

    for size in 0..=max_size {
        collect_subsets(&vars, size, |subset| {
            // Check if this subset is invariant across all environment pairs
            let mut inv = true;
            'outer: for i in 0..envs.len() {
                for j in (i + 1)..envs.len() {
                    if !is_invariant(&envs[i], &envs[j], subset, alpha) {
                        inv = false;
                        break 'outer;
                    }
                }
            }
            if inv {
                valid_sets.push(subset.to_vec());
            }
        });
    }

    if valid_sets.is_empty() {
        return vec![];
    }

    // Intersection of all valid sets
    let mut intersection: std::collections::HashSet<usize> =
        valid_sets[0].iter().copied().collect();
    for s in &valid_sets[1..] {
        let s_set: std::collections::HashSet<usize> = s.iter().copied().collect();
        intersection = intersection.intersection(&s_set).copied().collect();
    }

    let mut result: Vec<usize> = intersection.into_iter().collect();
    result.sort_unstable();
    result
}

fn collect_subsets<F>(items: &[usize], size: usize, mut f: F)
where
    F: FnMut(&[usize]),
{
    if size == 0 {
        f(&[]);
        return;
    }
    let n = items.len();
    if n < size {
        return;
    }
    // Simple recursive enumeration
    fn recurse(
        items: &[usize],
        size: usize,
        start: usize,
        current: &mut Vec<usize>,
        f: &mut dyn FnMut(&[usize]),
    ) {
        if current.len() == size {
            f(current);
            return;
        }
        for i in start..items.len() {
            current.push(items[i]);
            recurse(items, size, i + 1, current, f);
            current.pop();
        }
    }
    let mut current = Vec::with_capacity(size);
    recurse(items, size, 0, &mut current, &mut f);
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. CausalEffect estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Logistic regression model for propensity score estimation.
#[derive(Debug, Clone)]
pub struct PropensityScoreModel {
    /// Weight vector (n_vars + 1 with bias).
    pub weights: Vec<f32>,
}

impl PropensityScoreModel {
    /// Initialise with zero weights for `n_vars` features.
    pub fn new(n_vars: usize) -> Self {
        Self {
            weights: vec![0.0f32; n_vars + 1],
        }
    }

    /// Sigmoid probability for a single sample.
    pub fn predict_one(&self, x: &[f32]) -> f32 {
        let n = x.len().min(self.weights.len().saturating_sub(1));
        let mut logit = self.weights.last().copied().unwrap_or(0.0); // bias
        for i in 0..n {
            logit += x[i] * self.weights[i];
        }
        1.0 / (1.0 + (-logit).exp())
    }

    /// Fit via logistic regression (gradient descent, 200 steps).
    pub fn fit(&mut self, x: &[Vec<f32>], treatment: &[bool]) {
        let n = x.len().min(treatment.len());
        if n == 0 {
            return;
        }
        let lr = 0.01f32;
        for _ in 0..200 {
            let mut grads = vec![0.0f32; self.weights.len()];
            for i in 0..n {
                let p = self.predict_one(&x[i]);
                let t = if treatment[i] { 1.0f32 } else { 0.0 };
                let err = p - t;
                let n_feats = x[i].len().min(self.weights.len().saturating_sub(1));
                for k in 0..n_feats {
                    grads[k] += err * x[i][k];
                }
                if let Some(last) = grads.last_mut() {
                    *last += err; // bias
                }
            }
            for k in 0..self.weights.len() {
                self.weights[k] -= lr * grads[k] / n as f32;
            }
        }
    }
}

/// Estimate propensity scores (P(T=1 | X)) via logistic regression.
pub fn estimate_propensity(x: &[Vec<f32>], treatment: &[bool]) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }
    let n_vars = x[0].len();
    let mut model = PropensityScoreModel::new(n_vars);
    model.fit(x, treatment);
    x.iter().map(|row| model.predict_one(row)).collect()
}

/// Inverse probability weighting (IPW) estimator for the Average Treatment Effect.
///
/// ATE_IPW = (1/n) * Σ [ T*Y/e(X) − (1−T)*Y/(1−e(X)) ]
pub fn ipw_ate(treatment: &[bool], outcome: &[f32], propensity: &[f32]) -> f32 {
    let n = treatment.len().min(outcome.len()).min(propensity.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for i in 0..n {
        let e = propensity[i].clamp(0.01, 0.99);
        let y = outcome[i];
        let term = if treatment[i] { y / e } else { -y / (1.0 - e) };
        sum += term;
    }
    sum / n as f32
}

/// Doubly-robust ATE estimator.
///
/// DR-ATE = (1/n) Σ [ T*(Y − μ₁(X))/e(X) − (1−T)*(Y − μ₀(X))/(1−e(X)) + μ₁(X) − μ₀(X) ]
/// where `outcome_model` is a flat parameter vector giving μ(X) ≈ X·β
/// (uses first half as μ₁ coefficients, second half as μ₀ coefficients).
pub fn doubly_robust_ate(
    treatment: &[bool],
    outcome: &[f32],
    propensity: &[f32],
    outcome_model: &[f32],
) -> f32 {
    let n = treatment.len().min(outcome.len()).min(propensity.len());
    if n == 0 {
        return 0.0;
    }
    let half = outcome_model.len() / 2;
    let beta1 = &outcome_model[..half];
    let beta0 = &outcome_model[half..];

    let mut sum = 0.0f32;
    for i in 0..n {
        let e = propensity[i].clamp(0.01, 0.99);
        let y = outcome[i];
        let mu1 = if i < beta1.len() { beta1[i] } else { 0.0 };
        let mu0 = if i < beta0.len() { beta0[i] } else { 0.0 };

        let t_ind = if treatment[i] { 1.0f32 } else { 0.0 };
        let dr = t_ind * (y - mu1) / e - (1.0 - t_ind) * (y - mu0) / (1.0 - e) + mu1 - mu0;
        sum += dr;
    }
    sum / n as f32
}

/// Compute Manski sensitivity bounds for the ATE.
///
/// Returns (lower, upper) bounds given maximum confounding bias `max_confounding`.
pub fn sensitivity_analysis_bounds(ate: f32, max_confounding: f32) -> (f32, f32) {
    (ate - max_confounding, ate + max_confounding)
}
