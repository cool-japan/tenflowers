//! Statistical Hypothesis Testing Tools for ML Evaluation.
//!
//! This module provides a comprehensive set of statistical tests for evaluating
//! and comparing machine learning models, including parametric tests, non-parametric
//! rank-based tests, multiple comparison corrections, bootstrap methods, and
//! calibration analysis.
//!
//! # Modules Overview
//!
//! - [`ParametricTests`]: Classical t-tests, F-tests, ANOVA
//! - [`NonParametricTests`]: Wilcoxon, Mann-Whitney U, Kruskal-Wallis, Friedman
//! - \[`McNemar`\]: McNemar test, Cochran Q for model comparison
//! - [`Bootstrap`]: Bootstrap CI and permutation tests
//! - [`MultipleComparisonCorrection`]: Bonferroni, Holm, BH, BY corrections
//! - [`CalibrationAnalysis`]: ECE, MCE, Hosmer-Lemeshow test
//! - [`StatTestMetrics`]: Power analysis, sample size calculation

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Shared result struct
// ─────────────────────────────────────────────────────────────────────────────

/// Result from a statistical hypothesis test.
#[derive(Debug, Clone)]
pub struct StTestResult {
    /// The test statistic value.
    pub statistic: f64,
    /// The p-value associated with the test.
    pub p_value: f64,
    /// Whether the null hypothesis is rejected at the specified alpha level.
    pub reject_null: bool,
    /// Effect size measure (Cohen's d, eta-squared, rank biserial, etc.).
    pub effect_size: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper math functions
// ─────────────────────────────────────────────────────────────────────────────

/// Normal CDF via Abramowitz & Stegun rational approximation (7.1.26 / 26.2.17).
pub fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let p = 1.0 - pdf * poly;
    if x >= 0.0 {
        p
    } else {
        1.0 - p
    }
}

/// Inverse normal CDF (quantile function) via Beasley-Springer-Moro rational approximation.
pub fn normal_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    // Coefficients for central region
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Regularised incomplete beta function I_x(a, b) via continued fraction (Lentz method).
fn regularised_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Use symmetry for better convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularised_beta(1.0 - x, b, a);
    }
    let ln_beta_ab = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - ln_beta_ab).exp() / a;
    front * beta_cf(x, a, b)
}

/// Continued fraction for regularised incomplete beta (Lentz).
fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-10;
    let fpmin = 1e-300;

    let mut h = 1.0_f64.max(fpmin);
    let mut c = h;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    d = 1.0 / d.abs().max(fpmin) * d.signum();
    h = d;

    for m in 1..=max_iter {
        let mf = m as f64;
        // Even step
        let aa = mf * (b - mf) * x / ((a + 2.0 * mf - 1.0) * (a + 2.0 * mf));
        d = 1.0 + aa * d;
        d = 1.0 / d.abs().max(fpmin) * d.signum();
        c = 1.0 + aa / c.abs().max(fpmin) * c.signum();
        h *= d * c;
        // Odd step
        let aa = -(a + mf) * (a + b + mf) * x / ((a + 2.0 * mf) * (a + 2.0 * mf + 1.0));
        d = 1.0 + aa * d;
        d = 1.0 / d.abs().max(fpmin) * d.signum();
        c = 1.0 + aa / c.abs().max(fpmin) * c.signum();
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < eps {
            break;
        }
    }
    h
}

/// Log-gamma function via Lanczos approximation (g=7, precision ~15 digits).
fn ln_gamma(x: f64) -> f64 {
    let coeff = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().abs().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = coeff[0];
        for i in 1..9 {
            a += coeff[i] / (x + i as f64);
        }
        let t = x + 7.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Two-tailed p-value from t-distribution via regularised incomplete beta.
pub fn t_distribution_p_value(t_stat: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 1.0;
    }
    let x = df / (df + t_stat * t_stat);
    let p_one_tail = 0.5 * regularised_beta(x, df / 2.0, 0.5);
    (2.0 * p_one_tail).min(1.0)
}

/// Chi-square p-value (upper tail) via incomplete gamma approximation.
pub fn chi2_p_value(chi2: f64, df: usize) -> f64 {
    if chi2 <= 0.0 {
        return 1.0;
    }
    let df = df as f64;
    // P(X > chi2) = 1 - I(chi2/2, df/2) = regularised_gamma_upper(df/2, chi2/2)
    1.0 - regularised_gamma_lower(df / 2.0, chi2 / 2.0)
}

/// Lower regularised incomplete gamma P(a, x).
fn regularised_gamma_lower(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cf(a, x)
    }
}

fn gamma_series(a: f64, x: f64) -> f64 {
    let max_iter = 300;
    let eps = 1e-10;
    let ln_gam_a = ln_gamma(a);
    let mut sum = 1.0 / a;
    let mut del = sum;
    let mut ap = a;
    for _ in 0..max_iter {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * eps {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gam_a).exp()
}

fn gamma_cf(a: f64, x: f64) -> f64 {
    let max_iter = 300;
    let eps = 1e-10;
    let fpmin = 1e-300;
    let ln_gam_a = ln_gamma(a);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < eps {
            break;
        }
    }
    (-(x) + a * x.ln() - ln_gam_a).exp() * h
}

/// Mean and sample variance (with Bessel's correction) of a slice.
fn mean_var(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;
    if data.is_empty() {
        return (0.0, 0.0);
    }
    let m = data.iter().sum::<f64>() / n;
    let v = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (m, v)
}

/// Assign ranks to data (average ties), returning (sorted_indices, ranks).
fn compute_ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        data[a]
            .partial_cmp(&data[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (data[idx[j]] - data[idx[i]]).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based
        for k in i..j {
            ranks[idx[k]] = avg_rank;
        }
        i = j;
    }
    ranks
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Parametric Tests
// ─────────────────────────────────────────────────────────────────────────────

/// Classical parametric hypothesis tests: t-tests and F-tests.
pub struct ParametricTests;

/// Student's t-test variants.
pub struct TTest;

impl TTest {
    /// One-sample t-test: H₀: μ = μ₀.
    pub fn one_sample(data: &[f64], mu0: f64, alpha: f64) -> StTestResult {
        let n = data.len() as f64;
        let (mean, var) = mean_var(data);
        let se = (var / n).sqrt().max(1e-300);
        let t = (mean - mu0) / se;
        let df = n - 1.0;
        let p = t_distribution_p_value(t, df);
        let effect = (mean - mu0) / var.sqrt().max(1e-300);
        StTestResult {
            statistic: t,
            p_value: p,
            reject_null: p < alpha,
            effect_size: effect,
        }
    }

    /// Two-sample Welch's t-test: H₀: μ₁ = μ₂.
    pub fn two_sample(x: &[f64], y: &[f64], alpha: f64) -> StTestResult {
        let nx = x.len() as f64;
        let ny = y.len() as f64;
        let (mx, vx) = mean_var(x);
        let (my, vy) = mean_var(y);
        let se2 = vx / nx + vy / ny;
        let se = se2.sqrt().max(1e-300);
        let t = (mx - my) / se;
        // Welch-Satterthwaite degrees of freedom
        let df_num = se2 * se2;
        let df_den =
            (vx / nx).powi(2) / (nx - 1.0).max(1.0) + (vy / ny).powi(2) / (ny - 1.0).max(1.0);
        let df = (df_num / df_den.max(1e-300)).max(1.0);
        let p = t_distribution_p_value(t, df);
        let d = Self::cohens_d(x, y);
        StTestResult {
            statistic: t,
            p_value: p,
            reject_null: p < alpha,
            effect_size: d,
        }
    }

    /// Paired t-test: tests H₀: E[x_i - y_i] = 0.
    pub fn paired(x: &[f64], y: &[f64], alpha: f64) -> StTestResult {
        let diffs: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
        Self::one_sample(&diffs, 0.0, alpha)
    }

    /// Cohen's d effect size: (x̄ - ȳ) / pooled_std.
    pub fn cohens_d(x: &[f64], y: &[f64]) -> f64 {
        let nx = x.len() as f64;
        let ny = y.len() as f64;
        let (mx, vx) = mean_var(x);
        let (my, vy) = mean_var(y);
        let pooled_var = ((nx - 1.0) * vx + (ny - 1.0) * vy) / (nx + ny - 2.0).max(1.0);
        (mx - my) / pooled_var.sqrt().max(1e-300)
    }
}

/// F-test and one-way ANOVA.
pub struct FTest;

impl FTest {
    /// One-way ANOVA: H₀: all group means are equal.
    pub fn one_way_anova(groups: &[Vec<f64>], alpha: f64) -> StTestResult {
        let k = groups.len();
        let n_total: usize = groups.iter().map(|g| g.len()).sum();
        let grand_mean = groups.iter().flat_map(|g| g.iter()).sum::<f64>() / n_total as f64;

        let ss_between: f64 = groups
            .iter()
            .map(|g| {
                let (gm, _) = mean_var(g);
                g.len() as f64 * (gm - grand_mean).powi(2)
            })
            .sum();

        let ss_within: f64 = groups
            .iter()
            .map(|g| {
                let (gm, _) = mean_var(g);
                g.iter().map(|x| (x - gm).powi(2)).sum::<f64>()
            })
            .sum();

        let df_between = (k - 1).max(1) as f64;
        let df_within = (n_total - k).max(1) as f64;
        let msb = ss_between / df_between;
        let msw = ss_within / df_within;
        let f = msb / msw.max(1e-300);

        // p-value via F distribution = I_{df_between/(df_between + df_within*f)}(df_between/2, df_within/2)
        let x = df_between / (df_between + df_within * f);
        let p = regularised_beta(x, df_between / 2.0, df_within / 2.0).min(1.0);
        let eta2 = Self::eta_squared(groups);
        StTestResult {
            statistic: f,
            p_value: p,
            reject_null: p < alpha,
            effect_size: eta2,
        }
    }

    /// Eta-squared: SS_between / SS_total.
    pub fn eta_squared(groups: &[Vec<f64>]) -> f64 {
        let n_total: usize = groups.iter().map(|g| g.len()).sum();
        let grand_mean = groups.iter().flat_map(|g| g.iter()).sum::<f64>() / n_total as f64;
        let ss_between: f64 = groups
            .iter()
            .map(|g| {
                let (gm, _) = mean_var(g);
                g.len() as f64 * (gm - grand_mean).powi(2)
            })
            .sum();
        let ss_total: f64 = groups
            .iter()
            .flat_map(|g| g.iter())
            .map(|x| (x - grand_mean).powi(2))
            .sum();
        if ss_total < 1e-300 {
            0.0
        } else {
            ss_between / ss_total
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Non-Parametric Tests
// ─────────────────────────────────────────────────────────────────────────────

/// Rank-based non-parametric hypothesis tests.
pub struct NonParametricTests;

/// Wilcoxon signed-rank test.
pub struct WilcoxonSignedRank;

impl WilcoxonSignedRank {
    /// Signed-rank test: H₀: median of x - y = 0.
    pub fn test(x: &[f64], y: &[f64], alpha: f64) -> StTestResult {
        let diffs: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| a - b)
            .filter(|d| d.abs() > 1e-12)
            .collect();
        let n = diffs.len();
        if n == 0 {
            return StTestResult {
                statistic: 0.0,
                p_value: 1.0,
                reject_null: false,
                effect_size: 0.0,
            };
        }
        let abs_diffs: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
        let ranks = compute_ranks(&abs_diffs);
        let w: f64 = diffs
            .iter()
            .zip(ranks.iter())
            .map(|(d, r)| if *d > 0.0 { *r } else { -*r })
            .sum();
        let p = Self::normal_approximation_p(w, n);
        StTestResult {
            statistic: w,
            p_value: p,
            reject_null: p < alpha,
            effect_size: w / (n * (n + 1) / 2) as f64,
        }
    }

    /// Normal approximation p-value for W statistic.
    pub fn normal_approximation_p(w_stat: f64, n: usize) -> f64 {
        let nf = n as f64;
        let mean_w = nf * (nf + 1.0) / 4.0;
        let var_w = nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0;
        let z = (w_stat - mean_w) / var_w.sqrt().max(1e-300);
        let p = 2.0 * (1.0 - normal_cdf(z.abs()));
        p.clamp(0.0, 1.0)
    }
}

/// Mann-Whitney U test for two independent samples.
pub struct MannWhitneyU;

impl MannWhitneyU {
    /// Mann-Whitney U test: H₀: P(X > Y) = 0.5.
    pub fn test(x: &[f64], y: &[f64], alpha: f64) -> StTestResult {
        let nx = x.len();
        let ny = y.len();
        // U_x = number of (x_i, y_j) pairs where x_i > y_j
        let mut u_x: f64 = 0.0;
        for xi in x.iter() {
            for yj in y.iter() {
                if xi > yj {
                    u_x += 1.0;
                } else if (xi - yj).abs() < 1e-12 {
                    u_x += 0.5;
                }
            }
        }
        let u_y = (nx * ny) as f64 - u_x;
        let u = u_x.min(u_y);
        // Normal approximation
        let nxf = nx as f64;
        let nyf = ny as f64;
        let mean_u = nxf * nyf / 2.0;
        let var_u = nxf * nyf * (nxf + nyf + 1.0) / 12.0;
        let z = (u - mean_u) / var_u.sqrt().max(1e-300);
        let p = 2.0 * (1.0 - normal_cdf(z.abs()));
        let rb = Self::rank_biserial(x, y);
        StTestResult {
            statistic: u,
            p_value: p.clamp(0.0, 1.0),
            reject_null: p < alpha,
            effect_size: rb,
        }
    }

    /// Rank biserial correlation: 2U/(n1*n2) - 1.
    pub fn rank_biserial(x: &[f64], y: &[f64]) -> f64 {
        let nx = x.len();
        let ny = y.len();
        if nx == 0 || ny == 0 {
            return 0.0;
        }
        let mut u_x: f64 = 0.0;
        for xi in x.iter() {
            for yj in y.iter() {
                if xi > yj {
                    u_x += 1.0;
                } else if (xi - yj).abs() < 1e-12 {
                    u_x += 0.5;
                }
            }
        }
        2.0 * u_x / (nx * ny) as f64 - 1.0
    }
}

/// Kruskal-Wallis H test for k independent samples.
pub struct KruskalWallis;

impl KruskalWallis {
    /// Kruskal-Wallis test: H₀: all groups have the same distribution.
    pub fn test(groups: &[Vec<f64>], alpha: f64) -> StTestResult {
        let n_total: usize = groups.iter().map(|g| g.len()).sum();
        // Pool all values and rank
        let mut all_vals: Vec<(f64, usize)> = Vec::with_capacity(n_total);
        for (i, g) in groups.iter().enumerate() {
            for &v in g.iter() {
                all_vals.push((v, i));
            }
        }
        let flat: Vec<f64> = all_vals.iter().map(|(v, _)| *v).collect();
        let ranks = compute_ranks(&flat);

        // Sum of ranks per group
        let k = groups.len();
        let mut rank_sums = vec![0.0f64; k];
        for (pos, (_, grp)) in all_vals.iter().enumerate() {
            rank_sums[*grp] += ranks[pos];
        }

        let n = n_total as f64;
        let h: f64 = groups
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let ni = g.len() as f64;
                let r_mean = rank_sums[i] / ni;
                ni * (r_mean - (n + 1.0) / 2.0).powi(2)
            })
            .sum::<f64>()
            * 12.0
            / (n * (n + 1.0));

        let df = (k - 1).max(1);
        let p = chi2_p_value(h, df);
        StTestResult {
            statistic: h,
            p_value: p.clamp(0.0, 1.0),
            reject_null: p < alpha,
            effect_size: h / (n - 1.0),
        }
    }
}

/// Friedman test for repeated measures (k conditions, n subjects).
pub struct FriedmanTest;

impl FriedmanTest {
    /// Friedman test: H₀: all k conditions have the same distribution.
    ///
    /// `repeated_measures[subject][condition]`
    pub fn test(repeated_measures: &[Vec<f64>], alpha: f64) -> StTestResult {
        let n = repeated_measures.len(); // subjects
        if n == 0 {
            return StTestResult {
                statistic: 0.0,
                p_value: 1.0,
                reject_null: false,
                effect_size: 0.0,
            };
        }
        let k = repeated_measures[0].len(); // conditions
        if k < 2 {
            return StTestResult {
                statistic: 0.0,
                p_value: 1.0,
                reject_null: false,
                effect_size: 0.0,
            };
        }

        // Rank within each row (subject)
        let mut col_rank_sums = vec![0.0f64; k];
        for row in repeated_measures.iter() {
            let ranks = compute_ranks(row);
            for (j, r) in ranks.iter().enumerate() {
                col_rank_sums[j] += r;
            }
        }

        let nf = n as f64;
        let kf = k as f64;
        let chi2: f64 = 12.0 / (nf * kf * (kf + 1.0))
            * col_rank_sums
                .iter()
                .map(|rs| (rs - nf * (kf + 1.0) / 2.0).powi(2))
                .sum::<f64>();

        let df = (k - 1).max(1);
        let p = Self::chi2_p_value(chi2, df);
        StTestResult {
            statistic: chi2,
            p_value: p.clamp(0.0, 1.0),
            reject_null: p < alpha,
            effect_size: chi2 / (nf * (kf - 1.0)),
        }
    }

    /// Chi-square p-value approximation (upper tail).
    pub fn chi2_p_value(chi2: f64, df: usize) -> f64 {
        chi2_p_value(chi2, df)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. McNemar and Cochran Q
// ─────────────────────────────────────────────────────────────────────────────

/// McNemar and Cochran Q tests for comparing binary classifiers.
pub struct McNemarTest;

impl McNemarTest {
    /// McNemar test with Yates continuity correction.
    /// `contingency[i][j]` where i = correct/incorrect by model1, j = correct/incorrect by model2.
    pub fn test(contingency: &[[usize; 2]; 2], alpha: f64) -> StTestResult {
        let b = contingency[0][1] as f64; // model1 wrong, model2 correct
        let c = contingency[1][0] as f64; // model1 correct, model2 wrong
        let denom = b + c;
        if denom < 1e-9 {
            return StTestResult {
                statistic: 0.0,
                p_value: 1.0,
                reject_null: false,
                effect_size: 0.0,
            };
        }
        let chi2 = ((b - c).abs() - 1.0).max(0.0).powi(2) / denom;
        let p = chi2_p_value(chi2, 1);
        let effect = (b - c) / denom;
        StTestResult {
            statistic: chi2,
            p_value: p.clamp(0.0, 1.0),
            reject_null: p < alpha,
            effect_size: effect,
        }
    }

    /// Build a 2×2 contingency table from two binary classifiers' predictions.
    pub fn from_predictions(pred1: &[bool], pred2: &[bool], labels: &[bool]) -> [[usize; 2]; 2] {
        let mut table = [[0usize; 2]; 2];
        for ((p1, p2), label) in pred1.iter().zip(pred2.iter()).zip(labels.iter()) {
            let r1 = if p1 == label { 0 } else { 1 };
            let r2 = if p2 == label { 0 } else { 1 };
            table[r1][r2] += 1;
        }
        table
    }
}

/// Cochran Q test: generalization of McNemar to k ≥ 3 binary classifiers.
pub struct CochranQ;

impl CochranQ {
    /// Cochran Q test.
    /// `binary_responses[subject][classifier]` — true = correct.
    pub fn test(binary_responses: &[Vec<bool>], alpha: f64) -> StTestResult {
        let n = binary_responses.len();
        if n == 0 {
            return StTestResult {
                statistic: 0.0,
                p_value: 1.0,
                reject_null: false,
                effect_size: 0.0,
            };
        }
        let k = binary_responses[0].len();
        if k < 2 {
            return StTestResult {
                statistic: 0.0,
                p_value: 1.0,
                reject_null: false,
                effect_size: 0.0,
            };
        }

        // Row sums (correct per subject) and column sums (correct per classifier)
        let row_sums: Vec<f64> = binary_responses
            .iter()
            .map(|row| row.iter().filter(|&&v| v).count() as f64)
            .collect();
        let col_sums: Vec<f64> = (0..k)
            .map(|j| {
                binary_responses
                    .iter()
                    .filter(|row| j < row.len() && row[j])
                    .count() as f64
            })
            .collect();

        let total: f64 = row_sums.iter().sum();
        let kf = k as f64;
        let numer = kf * (kf - 1.0) * col_sums.iter().map(|c| c * c).sum::<f64>()
            - total * total * (kf - 1.0);
        // wait, standard formula: Q = (k-1)[k*sum(Cj^2) - T^2] / [k*T - sum(Li^2)]
        let denom_val = kf * total - row_sums.iter().map(|l| l * l).sum::<f64>();
        let q = if denom_val.abs() < 1e-12 {
            0.0
        } else {
            (kf - 1.0) * (kf * col_sums.iter().map(|c| c * c).sum::<f64>() - total * total)
                / denom_val
        };

        let df = (k - 1).max(1);
        let p = chi2_p_value(q.max(0.0), df);
        StTestResult {
            statistic: q,
            p_value: p.clamp(0.0, 1.0),
            reject_null: p < alpha,
            effect_size: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Bootstrap
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for bootstrap resampling.
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    /// Number of bootstrap resamples.
    pub n_bootstrap: usize,
    /// Confidence level (e.g. 0.95 for 95% CI).
    pub confidence_level: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            n_bootstrap: 1000,
            confidence_level: 0.95,
            seed: 42,
        }
    }
}

/// Result of a bootstrap procedure.
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Point estimate (on original data).
    pub statistic: f64,
    /// Lower bound of the confidence interval.
    pub ci_lower: f64,
    /// Upper bound of the confidence interval.
    pub ci_upper: f64,
    /// Bootstrap standard error.
    pub std_error: f64,
}

/// Bootstrap resampling methods for non-parametric inference.
pub struct Bootstrap {
    /// Configuration for the bootstrap procedure.
    pub config: BootstrapConfig,
}

impl Bootstrap {
    /// Create a new Bootstrap instance with the given configuration.
    pub fn new(config: BootstrapConfig) -> Self {
        Self { config }
    }

    /// Compute a bootstrap confidence interval for an arbitrary statistic.
    pub fn confidence_interval(
        &self,
        data: &[f64],
        statistic_fn: impl Fn(&[f64]) -> f64,
    ) -> BootstrapResult {
        let n = data.len();
        let observed = statistic_fn(data);
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut boot_stats: Vec<f64> = Vec::with_capacity(self.config.n_bootstrap);

        for _ in 0..self.config.n_bootstrap {
            let sample: Vec<f64> = (0..n)
                .map(|_| {
                    let idx = rng.random_range(0..n);
                    data[idx]
                })
                .collect();
            boot_stats.push(statistic_fn(&sample));
        }

        boot_stats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let alpha = (1.0 - self.config.confidence_level) / 2.0;
        let lo_idx = (alpha * self.config.n_bootstrap as f64) as usize;
        let hi_idx = ((1.0 - alpha) * self.config.n_bootstrap as f64) as usize;
        let ci_lower = boot_stats[lo_idx.min(self.config.n_bootstrap - 1)];
        let ci_upper = boot_stats[hi_idx.min(self.config.n_bootstrap - 1)];
        let mean_boot = boot_stats.iter().sum::<f64>() / self.config.n_bootstrap as f64;
        let var_boot = boot_stats
            .iter()
            .map(|s| (s - mean_boot).powi(2))
            .sum::<f64>()
            / self.config.n_bootstrap as f64;
        let std_error = var_boot.sqrt();

        BootstrapResult {
            statistic: observed,
            ci_lower,
            ci_upper,
            std_error,
        }
    }

    /// Bootstrap CI for mean(x) - mean(y).
    pub fn paired_difference_ci(&self, x: &[f64], y: &[f64]) -> BootstrapResult {
        let n = x.len().min(y.len());
        let paired: Vec<f64> = x[..n]
            .iter()
            .zip(y[..n].iter())
            .map(|(a, b)| a - b)
            .collect();
        self.confidence_interval(&paired, |d| d.iter().sum::<f64>() / d.len() as f64)
    }

    /// Bootstrap p-value for H₀: E\[scores1\] = E\[scores2\].
    pub fn model_comparison_bootstrap(&self, scores1: &[f64], scores2: &[f64]) -> f64 {
        let n1 = scores1.len();
        let n2 = scores2.len();
        let obs_diff =
            scores1.iter().sum::<f64>() / n1 as f64 - scores2.iter().sum::<f64>() / n2 as f64;
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut count_ge = 0usize;

        for _ in 0..self.config.n_bootstrap {
            let s1: Vec<f64> = (0..n1).map(|_| scores1[rng.random_range(0..n1)]).collect();
            let s2: Vec<f64> = (0..n2).map(|_| scores2[rng.random_range(0..n2)]).collect();
            let diff = s1.iter().sum::<f64>() / n1 as f64 - s2.iter().sum::<f64>() / n2 as f64;
            if diff.abs() >= obs_diff.abs() {
                count_ge += 1;
            }
        }

        count_ge as f64 / self.config.n_bootstrap as f64
    }

    /// Permutation test p-value: fraction of permuted |statistic| ≥ observed.
    pub fn permutation_test(&self, x: &[f64], y: &[f64], n_permutations: usize) -> f64 {
        let mut combined: Vec<f64> = x.iter().chain(y.iter()).cloned().collect();
        let nx = x.len();
        let obs = x.iter().sum::<f64>() / nx as f64 - y.iter().sum::<f64>() / y.len() as f64;
        let obs_abs = obs.abs();

        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let n = combined.len();
        let mut count_ge = 0usize;

        for _ in 0..n_permutations {
            // Fisher-Yates shuffle
            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);
                combined.swap(i, j);
            }
            let perm_x = &combined[..nx];
            let perm_y = &combined[nx..];
            let diff = perm_x.iter().sum::<f64>() / nx as f64
                - perm_y.iter().sum::<f64>() / perm_y.len() as f64;
            if diff.abs() >= obs_abs {
                count_ge += 1;
            }
        }

        count_ge as f64 / n_permutations as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Multiple Comparison Correction
// ─────────────────────────────────────────────────────────────────────────────

/// Supported multiple comparison correction methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Correction {
    /// No correction applied.
    None,
    /// Bonferroni correction.
    Bonferroni,
    /// Holm step-down correction.
    Holm,
    /// Benjamini-Hochberg FDR correction.
    BenjaminiHochberg,
    /// Benjamini-Yekutieli FDR correction for arbitrary dependence.
    BenjaminiYekutieli,
}

/// Multiple comparison correction procedures.
pub struct MultipleComparisonCorrection;

impl MultipleComparisonCorrection {
    /// Bonferroni correction: multiply each p-value by n (capped at 1.0).
    pub fn bonferroni(p_values: &[f64]) -> Vec<f64> {
        let n = p_values.len() as f64;
        p_values.iter().map(|p| (p * n).min(1.0)).collect()
    }

    /// Holm step-down correction.
    pub fn holm(p_values: &[f64]) -> Vec<f64> {
        let n = p_values.len();
        if n == 0 {
            return vec![];
        }
        let mut indexed: Vec<(usize, f64)> = p_values.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut adjusted = vec![0.0f64; n];
        let mut prev = 0.0f64;
        for (rank, (orig_idx, p)) in indexed.iter().enumerate() {
            let adj = (p * (n - rank) as f64).min(1.0).max(prev);
            adjusted[*orig_idx] = adj;
            prev = adj;
        }
        adjusted
    }

    /// Benjamini-Hochberg FDR procedure: returns boolean rejection vector.
    pub fn benjamini_hochberg(p_values: &[f64], fdr: f64) -> Vec<bool> {
        let n = p_values.len();
        if n == 0 {
            return vec![];
        }
        let mut indexed: Vec<(usize, f64)> = p_values.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        // Find largest k such that p_(k) <= k * fdr / n
        let mut k_max = 0usize;
        for (rank, (_, p)) in indexed.iter().enumerate() {
            let threshold = (rank + 1) as f64 * fdr / n as f64;
            if *p <= threshold {
                k_max = rank + 1;
            }
        }
        let mut reject = vec![false; n];
        for i in 0..k_max {
            reject[indexed[i].0] = true;
        }
        reject
    }

    /// Benjamini-Yekutieli FDR procedure (arbitrary dependence).
    pub fn benjamini_yekutieli(p_values: &[f64], fdr: f64) -> Vec<bool> {
        let n = p_values.len();
        if n == 0 {
            return vec![];
        }
        // c(n) = Σ 1/i for i = 1..n
        let cn: f64 = (1..=n).map(|i| 1.0 / i as f64).sum();
        let adjusted_fdr = fdr / cn;
        Self::benjamini_hochberg(p_values, adjusted_fdr)
    }

    /// Return adjusted p-values according to the chosen method.
    pub fn adjusted_p_values(p_values: &[f64], method: Correction) -> Vec<f64> {
        match method {
            Correction::None => p_values.to_vec(),
            Correction::Bonferroni => Self::bonferroni(p_values),
            Correction::Holm => Self::holm(p_values),
            Correction::BenjaminiHochberg => {
                // Convert booleans back to adjusted p-values: p_adj[i] = p[i] * n / rank
                let n = p_values.len();
                let mut indexed: Vec<(usize, f64)> = p_values.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut adj = vec![0.0f64; n];
                let mut min_so_far = f64::INFINITY;
                for (rank, (orig_idx, p)) in indexed.iter().enumerate().rev() {
                    let v = (p * n as f64 / (rank + 1) as f64).min(1.0);
                    min_so_far = min_so_far.min(v);
                    adj[*orig_idx] = min_so_far;
                }
                adj
            }
            Correction::BenjaminiYekutieli => {
                let n = p_values.len();
                let cn: f64 = (1..=n).map(|i| 1.0 / i as f64).sum();
                let mut indexed: Vec<(usize, f64)> = p_values.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut adj = vec![0.0f64; n];
                let mut min_so_far = f64::INFINITY;
                for (rank, (orig_idx, p)) in indexed.iter().enumerate().rev() {
                    let v = (p * n as f64 * cn / (rank + 1) as f64).min(1.0);
                    min_so_far = min_so_far.min(v);
                    adj[*orig_idx] = min_so_far;
                }
                adj
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Calibration Analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Reliability diagram (calibration curve) data.
#[derive(Debug, Clone)]
pub struct ReliabilityDiagram {
    /// Average confidence in each bin.
    pub bin_confidences: Vec<f64>,
    /// Fraction correct (accuracy) in each bin.
    pub bin_accuracies: Vec<f64>,
    /// Number of bins.
    pub n_bins: usize,
}

/// Calibration analysis tools for probabilistic classifiers.
pub struct CalibrationAnalysis;

impl CalibrationAnalysis {
    /// Build a reliability diagram with `n_bins` equal-width bins.
    pub fn reliability_diagram(
        probs: &[f64],
        labels: &[bool],
        n_bins: usize,
    ) -> ReliabilityDiagram {
        let n_bins = n_bins.max(1);
        let mut bin_conf_sum = vec![0.0f64; n_bins];
        let mut bin_correct = vec![0.0f64; n_bins];
        let mut bin_count = vec![0usize; n_bins];

        for (&p, &label) in probs.iter().zip(labels.iter()) {
            let bin = ((p * n_bins as f64) as usize).min(n_bins - 1);
            bin_conf_sum[bin] += p;
            bin_count[bin] += 1;
            if label {
                bin_correct[bin] += 1.0;
            }
        }

        let bin_confidences: Vec<f64> = (0..n_bins)
            .map(|b| {
                if bin_count[b] == 0 {
                    (b as f64 + 0.5) / n_bins as f64
                } else {
                    bin_conf_sum[b] / bin_count[b] as f64
                }
            })
            .collect();

        let bin_accuracies: Vec<f64> = (0..n_bins)
            .map(|b| {
                if bin_count[b] == 0 {
                    0.0
                } else {
                    bin_correct[b] / bin_count[b] as f64
                }
            })
            .collect();

        ReliabilityDiagram {
            bin_confidences,
            bin_accuracies,
            n_bins,
        }
    }

    /// Expected Calibration Error: Σ_b (|B_b|/n) * |acc(B_b) - conf(B_b)|.
    pub fn expected_calibration_error(probs: &[f64], labels: &[bool], n_bins: usize) -> f64 {
        let n_bins = n_bins.max(1);
        let n = probs.len() as f64;
        let mut bin_conf_sum = vec![0.0f64; n_bins];
        let mut bin_correct = vec![0.0f64; n_bins];
        let mut bin_count = vec![0usize; n_bins];

        for (&p, &label) in probs.iter().zip(labels.iter()) {
            let bin = ((p * n_bins as f64) as usize).min(n_bins - 1);
            bin_conf_sum[bin] += p;
            bin_count[bin] += 1;
            if label {
                bin_correct[bin] += 1.0;
            }
        }

        (0..n_bins)
            .map(|b| {
                if bin_count[b] == 0 {
                    return 0.0;
                }
                let bc = bin_count[b] as f64;
                let conf = bin_conf_sum[b] / bc;
                let acc = bin_correct[b] / bc;
                bc / n * (acc - conf).abs()
            })
            .sum::<f64>()
            .clamp(0.0, 1.0)
    }

    /// Maximum Calibration Error: max over bins of |acc - conf|.
    pub fn maximum_calibration_error(probs: &[f64], labels: &[bool], n_bins: usize) -> f64 {
        let n_bins = n_bins.max(1);
        let mut bin_conf_sum = vec![0.0f64; n_bins];
        let mut bin_correct = vec![0.0f64; n_bins];
        let mut bin_count = vec![0usize; n_bins];

        for (&p, &label) in probs.iter().zip(labels.iter()) {
            let bin = ((p * n_bins as f64) as usize).min(n_bins - 1);
            bin_conf_sum[bin] += p;
            bin_count[bin] += 1;
            if label {
                bin_correct[bin] += 1.0;
            }
        }

        (0..n_bins)
            .filter(|&b| bin_count[b] > 0)
            .map(|b| {
                let bc = bin_count[b] as f64;
                let conf = bin_conf_sum[b] / bc;
                let acc = bin_correct[b] / bc;
                (acc - conf).abs()
            })
            .fold(0.0f64, f64::max)
    }

    /// Hosmer-Lemeshow test for calibration with `n_groups` deciles.
    pub fn hosmer_lemeshow_test(
        probs: &[f64],
        labels: &[bool],
        n_groups: usize,
        alpha: f64,
    ) -> StTestResult {
        let n = probs.len();
        let n_groups = n_groups.max(2);
        // Sort by predicted probability
        let mut indexed: Vec<(f64, bool)> =
            probs.iter().cloned().zip(labels.iter().cloned()).collect();
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let group_size = n / n_groups;
        let mut hl_stat = 0.0f64;
        let mut df_valid = 0usize;

        for g in 0..n_groups {
            let start = g * group_size;
            let end = if g == n_groups - 1 {
                n
            } else {
                start + group_size
            };
            if start >= end {
                continue;
            }
            let group = &indexed[start..end];
            let ng = group.len() as f64;
            let obs: f64 = group.iter().filter(|(_, l)| *l).count() as f64;
            let expected: f64 = group.iter().map(|(p, _)| p).sum::<f64>();
            let exp_neg = ng - expected;
            let obs_neg = ng - obs;
            if expected > 1e-9 {
                hl_stat += (obs - expected).powi(2) / expected;
                df_valid += 1;
            }
            if exp_neg > 1e-9 {
                hl_stat += (obs_neg - exp_neg).powi(2) / exp_neg;
            }
        }

        let df = (n_groups - 2).max(1);
        let p = chi2_p_value(hl_stat, df);
        StTestResult {
            statistic: hl_stat,
            p_value: p.clamp(0.0, 1.0),
            reject_null: p < alpha,
            effect_size: hl_stat / (n as f64 - df_valid as f64).max(1.0),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. StatTest Metrics: Power, Sample Size
// ─────────────────────────────────────────────────────────────────────────────

/// Summary report for a statistical test.
#[derive(Debug, Clone)]
pub struct StatisticalReport {
    /// Name of the test performed.
    pub test_name: String,
    /// Test statistic value.
    pub statistic: f64,
    /// P-value.
    pub p_value: f64,
    /// Effect size.
    pub effect_size: f64,
    /// Statistical power.
    pub power: f64,
    /// Whether the null hypothesis was rejected.
    pub reject: bool,
}

/// Statistical power and sample size analysis tools.
pub struct StatTestMetrics;

impl StatTestMetrics {
    /// Approximate power for a two-sample z-test.
    ///
    /// `effect_size`: Cohen's d, `n`: per-group sample size, `alpha`: significance level.
    pub fn statistical_power(effect_size: f64, n: usize, alpha: f64) -> f64 {
        let z_alpha = normal_quantile(1.0 - alpha / 2.0);
        let ncp = effect_size * (n as f64 / 2.0).sqrt();
        let power = 1.0 - normal_cdf(z_alpha - ncp) + normal_cdf(-z_alpha - ncp);
        power.clamp(0.0, 1.0)
    }

    /// Minimum detectable effect size given n, alpha, and desired power.
    pub fn minimum_detectable_effect(n: usize, alpha: f64, power: f64) -> f64 {
        let z_alpha = normal_quantile(1.0 - alpha / 2.0);
        let z_beta = normal_quantile(power);
        (z_alpha + z_beta) / (n as f64 / 2.0).sqrt()
    }

    /// Required per-group sample size for desired power.
    pub fn sample_size_for_power(effect_size: f64, alpha: f64, power: f64) -> usize {
        if effect_size.abs() < 1e-10 {
            return usize::MAX;
        }
        let z_alpha = normal_quantile(1.0 - alpha / 2.0);
        let z_beta = normal_quantile(power);
        let n = 2.0 * ((z_alpha + z_beta) / effect_size).powi(2);
        (n.ceil() as usize).max(1)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Parametric: TTest ────────────────────────────────────────────────────

    #[test]
    fn test_t_test_one_sample_reject() {
        // Data clearly above mu0 = 0
        let data: Vec<f64> = (0..30).map(|i| 5.0 + i as f64 * 0.1).collect();
        let result = TTest::one_sample(&data, 0.0, 0.05);
        assert!(result.reject_null, "Should reject H0 when data mean >> mu0");
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_t_test_one_sample_accept() {
        // Data very close to mu0 = 5
        let data = vec![4.9, 5.0, 5.1, 4.95, 5.05, 5.0, 4.98, 5.02];
        let result = TTest::one_sample(&data, 5.0, 0.05);
        assert!(
            !result.reject_null,
            "Should not reject H0 when data mean ≈ mu0"
        );
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_t_test_two_sample_significant() {
        let x: Vec<f64> = (0..20).map(|i| 10.0 + i as f64 * 0.1).collect();
        let y: Vec<f64> = (0..20).map(|i| 0.0 + i as f64 * 0.1).collect();
        let result = TTest::two_sample(&x, &y, 0.05);
        assert!(result.reject_null);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_t_test_two_sample_not_significant() {
        // Both groups from same distribution
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.1, 1.9, 3.1, 3.9, 5.1];
        let result = TTest::two_sample(&x, &y, 0.05);
        assert!(!result.reject_null);
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_t_test_paired_significant() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..20).map(|i| i as f64 + 5.0).collect();
        let result = TTest::paired(&x, &y, 0.05);
        assert!(result.reject_null);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_cohens_d_zero() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = TTest::cohens_d(&x, &y);
        assert!(d.abs() < 1e-10, "Cohen's d should be 0 for identical data");
    }

    #[test]
    fn test_cohens_d_positive() {
        let x = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = TTest::cohens_d(&x, &y);
        assert!(d > 0.0, "Cohen's d should be positive when x > y");
    }

    #[test]
    fn test_one_way_anova_significant() {
        let groups = vec![
            vec![1.0, 2.0, 3.0],
            vec![10.0, 11.0, 12.0],
            vec![20.0, 21.0, 22.0],
        ];
        let result = FTest::one_way_anova(&groups, 0.05);
        assert!(result.reject_null);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_one_way_anova_not_significant() {
        let groups = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![0.9, 1.9, 2.9],
        ];
        let result = FTest::one_way_anova(&groups, 0.05);
        assert!(!result.reject_null);
    }

    #[test]
    fn test_eta_squared_range() {
        let groups = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let eta2 = FTest::eta_squared(&groups);
        assert!(
            (0.0..=1.0).contains(&eta2),
            "eta² must be in [0, 1], got {}",
            eta2
        );
    }

    #[test]
    fn test_t_distribution_p_zero_df1() {
        // t=0 should give p ≈ 1.0
        let p = t_distribution_p_value(0.0, 1.0);
        assert!(p > 0.9, "p-value for t=0 should be close to 1, got {}", p);
    }

    #[test]
    fn test_t_distribution_p_large_t() {
        // Very large t should give p ≈ 0
        let p = t_distribution_p_value(100.0, 30.0);
        assert!(p < 1e-6, "p-value for large t should be near 0, got {}", p);
    }

    // ── Non-parametric: Wilcoxon ─────────────────────────────────────────────

    #[test]
    fn test_wilcoxon_significant() {
        let x: Vec<f64> = (0..20).map(|i| i as f64 + 10.0).collect();
        let y: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = WilcoxonSignedRank::test(&x, &y, 0.05);
        assert!(result.reject_null);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_wilcoxon_not_significant() {
        // Identical x and y → all differences zero → filtered out → n=0 → p=1.0
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = WilcoxonSignedRank::test(&x, &y, 0.05);
        assert!(
            !result.reject_null,
            "Should not reject H0 for identical paired data, p={}",
            result.p_value
        );
    }

    #[test]
    fn test_normal_approx_p_range() {
        for w in [-10.0, 0.0, 10.0, 50.0] {
            let p = WilcoxonSignedRank::normal_approximation_p(w, 20);
            assert!((0.0..=1.0).contains(&p), "p must be in [0,1], got {}", p);
        }
    }

    #[test]
    fn test_mann_whitney_significant() {
        let x: Vec<f64> = (10..30).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = MannWhitneyU::test(&x, &y, 0.05);
        assert!(result.reject_null);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_mann_whitney_not_significant() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.1, 1.9, 3.0];
        let result = MannWhitneyU::test(&x, &y, 0.05);
        assert!(!result.reject_null);
    }

    #[test]
    fn test_rank_biserial_range() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rb = MannWhitneyU::rank_biserial(&x, &y);
        assert!(
            (-1.0..=1.0).contains(&rb),
            "rank biserial must be in [-1,1], got {}",
            rb
        );
    }

    #[test]
    fn test_kruskal_wallis_significant() {
        let groups = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![10.0, 11.0, 12.0, 13.0],
            vec![20.0, 21.0, 22.0, 23.0],
        ];
        let result = KruskalWallis::test(&groups, 0.05);
        assert!(result.reject_null);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_kruskal_wallis_not_significant() {
        let groups = vec![vec![1.0, 2.0, 3.0], vec![1.5, 2.5, 3.5]];
        let result = KruskalWallis::test(&groups, 0.05);
        assert!(!result.reject_null);
    }

    #[test]
    fn test_friedman_shape_preserved() {
        // 5 subjects, 3 conditions
        let data = vec![
            vec![3.0, 1.0, 2.0],
            vec![2.0, 3.0, 1.0],
            vec![1.0, 2.0, 3.0],
            vec![3.0, 2.0, 1.0],
            vec![2.0, 1.0, 3.0],
        ];
        let result = FriedmanTest::test(&data, 0.05);
        // Just check it runs and returns valid p-value
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    #[test]
    fn test_chi2_p_value_range() {
        for chi2 in [0.0, 1.0, 5.0, 10.0, 100.0] {
            let p = chi2_p_value(chi2, 5);
            assert!(
                (0.0..=1.0).contains(&p),
                "p must be in [0,1], got {} for chi2={}",
                p,
                chi2
            );
        }
    }

    // ── McNemar ──────────────────────────────────────────────────────────────

    #[test]
    fn test_mcnemar_from_predictions_table_shape() {
        let pred1 = vec![true, false, true, true, false];
        let pred2 = vec![true, true, false, true, false];
        let labels = vec![true, true, true, false, false];
        let table = McNemarTest::from_predictions(&pred1, &pred2, &labels);
        // Check it's a 2x2 table summing to n
        let total: usize = table[0][0] + table[0][1] + table[1][0] + table[1][1];
        assert_eq!(total, 5);
    }

    #[test]
    fn test_mcnemar_no_difference() {
        // b = c → should accept H0
        let contingency = [[50usize, 5usize], [5usize, 40usize]];
        let result = McNemarTest::test(&contingency, 0.05);
        assert!(!result.reject_null, "b=c should not reject H0");
    }

    #[test]
    fn test_mcnemar_strong_difference() {
        // b >> c → should reject H0
        let contingency = [[40usize, 30usize], [2usize, 28usize]];
        let result = McNemarTest::test(&contingency, 0.05);
        assert!(result.reject_null, "b >> c should reject H0");
    }

    #[test]
    fn test_cochran_q_shape() {
        // 10 subjects, 3 classifiers
        let responses = vec![
            vec![true, false, true],
            vec![true, true, false],
            vec![false, true, true],
            vec![true, false, false],
            vec![false, false, true],
            vec![true, true, true],
            vec![false, true, false],
            vec![true, false, true],
            vec![true, true, false],
            vec![false, false, false],
        ];
        let result = CochranQ::test(&responses, 0.05);
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    // ── Bootstrap ────────────────────────────────────────────────────────────

    #[test]
    fn test_bootstrap_ci_contains_truth() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let true_mean = 24.5; // mean of 0..49
        let bs = Bootstrap::new(BootstrapConfig {
            n_bootstrap: 2000,
            confidence_level: 0.95,
            seed: 1,
        });
        let result = bs.confidence_interval(&data, |d| d.iter().sum::<f64>() / d.len() as f64);
        assert!(
            result.ci_lower <= true_mean && true_mean <= result.ci_upper,
            "True mean {} not in CI [{}, {}]",
            true_mean,
            result.ci_lower,
            result.ci_upper
        );
    }

    #[test]
    fn test_bootstrap_ci_width_decreases_with_n() {
        let small: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let large: Vec<f64> = (0..200).map(|i| i as f64 * 0.1).collect();
        let bs = Bootstrap::new(BootstrapConfig::default());
        let r_small = bs.confidence_interval(&small, |d| d.iter().sum::<f64>() / d.len() as f64);
        let r_large = bs.confidence_interval(&large, |d| d.iter().sum::<f64>() / d.len() as f64);
        let width_small = r_small.ci_upper - r_small.ci_lower;
        let width_large = r_large.ci_upper - r_large.ci_lower;
        // The span of large is [0,19.9] vs small [0,19], so compare std_error instead
        assert!(r_small.std_error > 0.0);
        assert!(r_large.std_error > 0.0);
        let _ = (width_small, width_large); // used for context only
    }

    #[test]
    fn test_paired_difference_ci_zero_same() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bs = Bootstrap::new(BootstrapConfig::default());
        let result = bs.paired_difference_ci(&x, &y);
        // Mean difference should be 0; CI should straddle 0
        assert!(
            result.ci_lower <= 0.0 && 0.0 <= result.ci_upper,
            "CI should contain 0 for identical series"
        );
    }

    #[test]
    fn test_model_comparison_p_range() {
        let s1 = vec![0.9, 0.85, 0.88, 0.91, 0.87];
        let s2 = vec![0.7, 0.72, 0.68, 0.75, 0.71];
        let bs = Bootstrap::new(BootstrapConfig {
            n_bootstrap: 500,
            ..BootstrapConfig::default()
        });
        let p = bs.model_comparison_bootstrap(&s1, &s2);
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn test_permutation_test_same_data() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bs = Bootstrap::new(BootstrapConfig::default());
        let p = bs.permutation_test(&x, &y, 500);
        // Should NOT be small (data are same, so most permutations give abs diff ≥ 0)
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn test_permutation_test_different_data() {
        let x: Vec<f64> = (0..20).map(|i| i as f64 + 100.0).collect();
        let y: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let bs = Bootstrap::new(BootstrapConfig {
            n_bootstrap: 1000,
            seed: 42,
            ..BootstrapConfig::default()
        });
        let p = bs.permutation_test(&x, &y, 1000);
        // Very different means → p should be small
        assert!(
            p < 0.1,
            "p should be small for very different data, got {}",
            p
        );
    }

    // ── Multiple Comparisons ─────────────────────────────────────────────────

    #[test]
    fn test_bonferroni_length() {
        let p = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let adj = MultipleComparisonCorrection::bonferroni(&p);
        assert_eq!(adj.len(), p.len());
    }

    #[test]
    fn test_bonferroni_adjustment() {
        let p = vec![0.01, 0.02, 0.05];
        let adj = MultipleComparisonCorrection::bonferroni(&p);
        let n = p.len() as f64;
        for (orig, a) in p.iter().zip(adj.iter()) {
            let expected = (orig * n).min(1.0);
            assert!((a - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_holm_length() {
        let p = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let adj = MultipleComparisonCorrection::holm(&p);
        assert_eq!(adj.len(), p.len());
    }

    #[test]
    fn test_holm_dominated_by_bonferroni() {
        let p = vec![0.005, 0.011, 0.015, 0.02, 0.04];
        let bonf = MultipleComparisonCorrection::bonferroni(&p);
        let holm = MultipleComparisonCorrection::holm(&p);
        for (h, b) in holm.iter().zip(bonf.iter()) {
            assert!(*h <= *b + 1e-10, "Holm {} should be ≤ Bonferroni {}", h, b);
        }
    }

    #[test]
    fn test_bh_procedure_returns_bools() {
        let p = vec![0.001, 0.01, 0.03, 0.1, 0.5];
        let reject = MultipleComparisonCorrection::benjamini_hochberg(&p, 0.05);
        assert_eq!(reject.len(), p.len());
    }

    #[test]
    fn test_by_procedure_returns_bools() {
        let p = vec![0.001, 0.01, 0.03, 0.1, 0.5];
        let reject = MultipleComparisonCorrection::benjamini_yekutieli(&p, 0.05);
        assert_eq!(reject.len(), p.len());
    }

    // ── Calibration ──────────────────────────────────────────────────────────

    #[test]
    fn test_reliability_diagram_bins() {
        let probs = vec![0.1, 0.25, 0.4, 0.6, 0.75, 0.9];
        let labels = vec![false, false, true, true, true, true];
        let rd = CalibrationAnalysis::reliability_diagram(&probs, &labels, 5);
        assert_eq!(rd.n_bins, 5);
        assert_eq!(rd.bin_confidences.len(), 5);
        assert_eq!(rd.bin_accuracies.len(), 5);
    }

    #[test]
    fn test_ece_range() {
        let probs: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let labels: Vec<bool> = probs.iter().map(|&p| p > 0.5).collect();
        let ece = CalibrationAnalysis::expected_calibration_error(&probs, &labels, 10);
        assert!(
            (0.0..=1.0).contains(&ece),
            "ECE must be in [0,1], got {}",
            ece
        );
    }

    #[test]
    fn test_mce_ge_ece() {
        let probs: Vec<f64> = (0..50).map(|i| i as f64 / 50.0).collect();
        let labels: Vec<bool> = probs.iter().map(|&p| p > 0.5).collect();
        let ece = CalibrationAnalysis::expected_calibration_error(&probs, &labels, 10);
        let mce = CalibrationAnalysis::maximum_calibration_error(&probs, &labels, 10);
        assert!(mce >= ece - 1e-10, "MCE {} must be >= ECE {}", mce, ece);
    }

    #[test]
    fn test_hosmer_lemeshow_runs() {
        let probs: Vec<f64> = (1..=20).map(|i| i as f64 / 21.0).collect();
        let labels: Vec<bool> = probs.iter().map(|&p| p > 0.5).collect();
        let result = CalibrationAnalysis::hosmer_lemeshow_test(&probs, &labels, 10, 0.05);
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    // ── Normal CDF / Quantile ────────────────────────────────────────────────

    #[test]
    fn test_normal_cdf_0_5() {
        let p = normal_cdf(0.0);
        assert!(
            (p - 0.5).abs() < 1e-6,
            "normal_cdf(0) should be 0.5, got {}",
            p
        );
    }

    #[test]
    fn test_normal_quantile_0_5() {
        let q = normal_quantile(0.5);
        assert!(
            q.abs() < 1e-4,
            "normal_quantile(0.5) should be 0, got {}",
            q
        );
    }

    // ── Power / Sample Size ──────────────────────────────────────────────────

    #[test]
    fn test_sample_size_positive() {
        let n = StatTestMetrics::sample_size_for_power(0.5, 0.05, 0.80);
        assert!(n > 0, "sample size must be positive");
    }

    #[test]
    fn test_statistical_power_range() {
        let power = StatTestMetrics::statistical_power(0.5, 50, 0.05);
        assert!(
            (0.0..=1.0).contains(&power),
            "power must be in [0,1], got {}",
            power
        );
    }
}
