//! Shared math helpers used across causal_discovery_advanced submodules.

// ─────────────────────────────────────────────────────────────────────────────
// Basic statistics
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub fn mean_f32(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f32>() / v.len() as f32
}

#[inline]
pub fn variance_f32(v: &[f32]) -> f32 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean_f32(v);
    v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / (v.len() - 1) as f32
}

#[inline]
pub fn std_f32(v: &[f32]) -> f32 {
    variance_f32(v).sqrt()
}

/// Pearson correlation between two equal-length slices.
pub fn pearson_corr(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mx = mean_f32(&x[..n]);
    let my = mean_f32(&y[..n]);
    let mut cov = 0.0f32;
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        sx += dx * dx;
        sy += dy * dy;
    }
    let denom = (sx * sy).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

/// Simple OLS: returns (slope, intercept) for univariate y ~ x.
pub fn ols_univariate(x: &[f32], y: &[f32]) -> (f32, f32) {
    let n = x.len().min(y.len());
    if n < 2 {
        return (0.0, 0.0);
    }
    let mx = mean_f32(&x[..n]);
    let my = mean_f32(&y[..n]);
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for i in 0..n {
        let dx = x[i] - mx;
        num += dx * (y[i] - my);
        den += dx * dx;
    }
    if den.abs() < 1e-12 {
        return (0.0, my);
    }
    let slope = num / den;
    let intercept = my - slope * mx;
    (slope, intercept)
}

/// Compute OLS residuals: y - X*beta where beta = (X'X)^{-1} X'y via normal equations.
/// `features` selects columns of X; simple univariate path if only one feature.
pub fn ols_residuals(x_mat: &[Vec<f32>], y: &[f32], features: &[usize]) -> Vec<f32> {
    let n = x_mat.len().min(y.len());
    if features.is_empty() || n == 0 {
        return y[..n].to_vec();
    }
    // Build Xf: n × k sub-matrix (with intercept appended as last column)
    let k = features.len() + 1; // +1 for intercept
    let mut xf: Vec<f32> = Vec::with_capacity(n * k);
    for i in 0..n {
        for &f in features {
            let val = if f < x_mat[i].len() { x_mat[i][f] } else { 0.0 };
            xf.push(val);
        }
        xf.push(1.0); // intercept column
    }
    // Normal equations: (X'X) beta = X'y  — solved via Cholesky (small k)
    let mut xtx = vec![0.0f32; k * k];
    let mut xty = vec![0.0f32; k];
    for i in 0..n {
        for a in 0..k {
            xty[a] += xf[i * k + a] * y[i];
            for b in 0..k {
                xtx[a * k + b] += xf[i * k + a] * xf[i * k + b];
            }
        }
    }
    // Ridge regularisation for stability
    for d in 0..k {
        xtx[d * k + d] += 1e-6;
    }
    // Gaussian elimination (in-place on xtx, xty)
    let beta = gauss_solve(&xtx, &xty, k);
    // Compute residuals
    let mut residuals = Vec::with_capacity(n);
    for i in 0..n {
        let mut pred = 0.0f32;
        for a in 0..k {
            pred += xf[i * k + a] * beta[a];
        }
        residuals.push(y[i] - pred);
    }
    residuals
}

/// Naive Gaussian elimination returning x in Ax = b (square system, size k).
pub fn gauss_solve(a: &[f32], b: &[f32], k: usize) -> Vec<f32> {
    let mut mat = vec![0.0f32; k * (k + 1)];
    for i in 0..k {
        for j in 0..k {
            mat[i * (k + 1) + j] = a[i * k + j];
        }
        mat[i * (k + 1) + k] = b[i];
    }
    for col in 0..k {
        // Find pivot
        let mut max_row = col;
        let mut max_val = mat[col * (k + 1) + col].abs();
        for row in (col + 1)..k {
            let v = mat[row * (k + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            continue;
        }
        // Swap rows
        for j in 0..=(k) {
            mat.swap(col * (k + 1) + j, max_row * (k + 1) + j);
        }
        // Eliminate
        let pivot = mat[col * (k + 1) + col];
        for row in (col + 1)..k {
            let factor = mat[row * (k + 1) + col] / pivot;
            for j in col..=(k) {
                let sub = factor * mat[col * (k + 1) + j];
                mat[row * (k + 1) + j] -= sub;
            }
        }
    }
    // Back-substitution
    let mut x = vec![0.0f32; k];
    for i in (0..k).rev() {
        let mut s = mat[i * (k + 1) + k];
        for j in (i + 1)..k {
            s -= mat[i * (k + 1) + j] * x[j];
        }
        let diag = mat[i * (k + 1) + i];
        x[i] = if diag.abs() < 1e-12 { 0.0 } else { s / diag };
    }
    x
}

/// Matrix exponential via Taylor series up to order 6: exp(A) ≈ I + A + A²/2! + …
/// `a` is n×n stored row-major as a flat `Vec<f32>`.
///
/// Values are clamped to [-1e6, 1e6] to prevent overflow in the accumulation.
pub fn mat_exp_approx(a: &[f32], n: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; n * n];
    // Start with identity
    for i in 0..n {
        result[i * n + i] = 1.0;
    }
    // Clamp input to avoid overflow in higher-order terms
    let a_clamped: Vec<f32> = a.iter().map(|&v| v.clamp(-10.0, 10.0)).collect();
    // Accumulate terms A^k / k!
    let mut ak = a_clamped.clone();
    for k in 1usize..=6 {
        // result += ak / k!
        let fact = factorial(k) as f32;
        for j in 0..n * n {
            let term = ak[j] / fact;
            if term.is_finite() {
                result[j] += term;
            }
        }
        if k < 6 {
            // ak_next = ak * a_clamped
            let mut next = vec![0.0f32; n * n];
            for i in 0..n {
                for l in 0..n {
                    for m in 0..n {
                        let v = ak[i * n + m] * a_clamped[m * n + l];
                        if v.is_finite() {
                            next[i * n + l] += v;
                        }
                    }
                }
            }
            ak = next;
        }
    }
    result
}

fn factorial(k: usize) -> u64 {
    match k {
        0 | 1 => 1,
        2 => 2,
        3 => 6,
        4 => 24,
        5 => 120,
        6 => 720,
        _ => {
            let mut r = 1u64;
            for i in 2..=(k as u64) {
                r = r.saturating_mul(i);
            }
            r
        }
    }
}

/// Standard normal CDF approximation (Abramowitz & Stegun).
pub fn normal_cdf_f32(x: f32) -> f32 {
    let x = x as f64;
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let phi = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = if x >= 0.0 {
        1.0 - phi * poly
    } else {
        phi * poly
    };
    cdf as f32
}

/// Invert a symmetric positive-definite matrix via Gaussian elimination (small k).
pub fn invert_sym(a: &[f32], k: usize) -> Vec<f32> {
    let mut inv = vec![0.0f32; k * k];
    let mut rhs = vec![0.0f32; k];
    for col in 0..k {
        rhs.fill(0.0);
        rhs[col] = 1.0;
        let x = gauss_solve(a, &rhs, k);
        for row in 0..k {
            inv[row * k + col] = x[row];
        }
    }
    inv
}
