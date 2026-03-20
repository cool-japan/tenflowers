//! Advanced causal time series algorithms.
//!
//! Includes Structural VAR (SVAR) with Cholesky identification, nonlinear
//! Granger causality (neural + kernel), causal bandits, and time-varying
//! causal discovery with CUSUM change-point detection.

use super::{chi2_p_value, ols, residuals, ssr, CtsResult};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// 1. Causal Structural VAR (SVAR)
// ─────────────────────────────────────────────────────────────────────────────

/// Structural VAR (SVAR) with Cholesky identification (lower-triangular B matrix).
///
/// The reduced-form VAR(p) is estimated via OLS, then the structural shocks
/// are identified via the Cholesky decomposition of the residual covariance
/// matrix Σ_u = B B', where B is lower-triangular (recursive/Cholesky ordering).
pub struct SvarModel {
    /// Number of lags.
    pub p: usize,
    /// Number of variables.
    pub k: usize,
    /// Reduced-form coefficient matrices \[lag\]\[eq\]\[var\].
    pub coefficients: Vec<Vec<Vec<f64>>>,
    /// Intercepts for each equation.
    pub intercepts: Vec<f64>,
    /// Cholesky factor B (lower-triangular), k×k, identifies structural shocks.
    pub b_matrix: Vec<Vec<f64>>,
    /// Residual covariance matrix Σ_u.
    pub sigma_u: Vec<Vec<f64>>,
    /// Last p observations for forecasting.
    pub last_obs: Vec<Vec<f64>>,
}

impl SvarModel {
    /// Fit SVAR(p) via OLS + Cholesky identification.
    ///
    /// `data[t][v]` is variable `v` at time `t`.
    pub fn fit(data: &[Vec<f64>], p: usize) -> CtsResult<Self> {
        let n = data.len();
        if n == 0 {
            return Err("SVAR: empty data".to_string());
        }
        let k = data[0].len();
        if k == 0 {
            return Err("SVAR: zero variables".to_string());
        }
        if n <= p + k {
            return Err(format!("SVAR: too few observations n={} p={} k={}", n, p, k));
        }
        let n_eff = n - p;

        // Build design matrix: [1, y_{t-1}, ..., y_{t-p}]
        let n_cols = 1 + p * k;
        let mut design: Vec<Vec<f64>> = Vec::with_capacity(n_eff);
        for t in p..n {
            let mut row = Vec::with_capacity(n_cols);
            row.push(1.0);
            for lag in 1..=p {
                for v in 0..k {
                    row.push(data[t - lag][v]);
                }
            }
            design.push(row);
        }

        let mut coefficients = vec![vec![vec![0.0_f64; k]; k]; p];
        let mut intercepts = vec![0.0_f64; k];
        // Residuals matrix: n_eff × k
        let mut resid_mat = vec![vec![0.0_f64; k]; n_eff];

        for eq in 0..k {
            let y_eq: Vec<f64> = (p..n).map(|t| data[t][eq]).collect();
            let beta = ols(&design, &y_eq)?;
            intercepts[eq] = beta[0];
            for lag in 0..p {
                for v in 0..k {
                    coefficients[lag][eq][v] = beta[1 + lag * k + v];
                }
            }
            let resid_eq = residuals(&design, &y_eq, &beta);
            for (t, r) in resid_eq.iter().enumerate() {
                resid_mat[t][eq] = *r;
            }
        }

        // Estimate Σ_u = (1/n_eff) * U' U
        let mut sigma_u = vec![vec![0.0_f64; k]; k];
        for t in 0..n_eff {
            for i in 0..k {
                for j in 0..k {
                    sigma_u[i][j] += resid_mat[t][i] * resid_mat[t][j];
                }
            }
        }
        for i in 0..k {
            for j in 0..k {
                sigma_u[i][j] /= n_eff as f64;
            }
        }

        // Cholesky: Σ_u = B B'  (B lower-triangular)
        let b_matrix = cholesky_lower(&sigma_u)?;
        let last_obs: Vec<Vec<f64>> = (n - p..n).map(|t| data[t].clone()).collect();

        Ok(Self {
            p,
            k,
            coefficients,
            intercepts,
            b_matrix,
            sigma_u,
            last_obs,
        })
    }

    /// Compute structural residuals (white-noise structural shocks) from reduced-form residuals.
    ///
    /// ε_t = B^{-1} u_t (via forward substitution since B is lower-triangular).
    pub fn structural_residuals(&self, reduced_resid: &[Vec<f64>]) -> CtsResult<Vec<Vec<f64>>> {
        let k = self.k;
        let b = &self.b_matrix;
        reduced_resid
            .iter()
            .map(|u| {
                // Solve B ε = u via forward substitution
                let mut eps = vec![0.0_f64; k];
                for i in 0..k {
                    let mut s = u.get(i).copied().unwrap_or(0.0);
                    for j in 0..i {
                        s -= b[i][j] * eps[j];
                    }
                    let denom = b[i][i];
                    if denom.abs() < 1e-15 {
                        return Err("SVAR: singular B matrix".to_string());
                    }
                    eps[i] = s / denom;
                }
                Ok(eps)
            })
            .collect()
    }
}

/// Orthogonalized Impulse Response Functions from a fitted SVAR model.
///
/// OIRF at horizon h gives the response of variable `j` to a unit structural
/// shock in variable `i` at time 0.
pub struct SvarImpulseResponse {
    /// IRF values: `irf[h][j][i]` = response of j to shock i at horizon h.
    pub irf: Vec<Vec<Vec<f64>>>,
    /// Number of horizons computed.
    pub horizons: usize,
    /// Number of variables.
    pub k: usize,
}

impl SvarImpulseResponse {
    /// Compute OIRFs for `horizons` periods from a fitted SvarModel.
    pub fn compute(model: &SvarModel, horizons: usize) -> Self {
        let k = model.k;
        let p = model.p;
        let h_max = horizons.max(1);

        // Phi_h: reduced-form impulse response (MA representation)
        // Phi_0 = I_k, Phi_h = sum_{j=1}^{min(h,p)} A_j Phi_{h-j}
        let mut phi: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0_f64; k]; k]; h_max + 1];
        // Phi_0 = identity
        for i in 0..k {
            phi[0][i][i] = 1.0;
        }
        for h in 1..=h_max {
            let mut phi_h = vec![vec![0.0_f64; k]; k];
            for j in 1..=p.min(h) {
                let a_j = &model.coefficients[j - 1]; // A_j is [eq][var]
                let phi_prev = &phi[h - j];
                for r in 0..k {
                    for c in 0..k {
                        for m in 0..k {
                            phi_h[r][c] += a_j[r][m] * phi_prev[m][c];
                        }
                    }
                }
            }
            phi[h] = phi_h;
        }

        // OIRF_h = Phi_h * B  (structural shocks via Cholesky B)
        let b = &model.b_matrix;
        let mut irf = vec![vec![vec![0.0_f64; k]; k]; h_max + 1];
        for h in 0..=h_max {
            for r in 0..k {
                for c in 0..k {
                    // irf[h][r][c] = (Phi_h B)[r][c] = sum_m Phi_h[r][m] * B[m][c]
                    irf[h][r][c] = (0..k).map(|m| phi[h][r][m] * b[m][c]).sum();
                }
            }
        }

        Self {
            irf,
            horizons: h_max,
            k,
        }
    }

    /// Get IRF value: response of variable `resp` to shock `shock` at horizon `h`.
    pub fn get(&self, h: usize, resp: usize, shock: usize) -> f64 {
        self.irf
            .get(h)
            .and_then(|r| r.get(resp))
            .and_then(|c| c.get(shock))
            .copied()
            .unwrap_or(0.0)
    }

    /// Cumulative IRF from horizon 0 to `h` (sum of responses).
    pub fn cumulative(&self, h: usize, resp: usize, shock: usize) -> f64 {
        (0..=h.min(self.horizons))
            .map(|hh| self.get(hh, resp, shock))
            .sum()
    }
}

/// Forecast Error Variance Decomposition (FEVD) from SVAR.
///
/// FEVD\[h\]\[j\]\[i\] = fraction of forecast-error variance of variable j at
/// horizon h explained by structural shock i.
pub struct SvarForecastErrorVarianceDecomp {
    /// FEVD values: `fevd[h][response_var][shock_var]`.
    pub fevd: Vec<Vec<Vec<f64>>>,
    /// Number of horizons.
    pub horizons: usize,
    /// Number of variables.
    pub k: usize,
}

impl SvarForecastErrorVarianceDecomp {
    /// Compute FEVD from OIRFs up to `horizons`.
    pub fn compute(oirf: &SvarImpulseResponse) -> Self {
        let k = oirf.k;
        let h_max = oirf.horizons;
        // MSE_h[j] = sum_{s=0}^{h} sum_{i=0}^{k} (irf[s][j][i])^2
        // FEVD[h][j][i] = (sum_{s=0}^{h} irf[s][j][i]^2) / MSE_h[j]
        let mut cumulative_sq = vec![vec![vec![0.0_f64; k]; k]; h_max + 1];
        // Build cumulative squared OIRF
        let mut running = vec![vec![0.0_f64; k]; k]; // [j][i]
        for h in 0..=h_max {
            for j in 0..k {
                for i in 0..k {
                    running[j][i] += oirf.get(h, j, i).powi(2);
                    cumulative_sq[h][j][i] = running[j][i];
                }
            }
        }

        let mut fevd = vec![vec![vec![0.0_f64; k]; k]; h_max + 1];
        for h in 0..=h_max {
            for j in 0..k {
                let total: f64 = (0..k).map(|i| cumulative_sq[h][j][i]).sum::<f64>().max(1e-30);
                for i in 0..k {
                    fevd[h][j][i] = cumulative_sq[h][j][i] / total;
                }
            }
        }

        Self {
            fevd,
            horizons: h_max,
            k,
        }
    }

    /// Get FEVD value: fraction of variance of `resp` at horizon `h` due to shock `shock`.
    pub fn get(&self, h: usize, resp: usize, shock: usize) -> f64 {
        self.fevd
            .get(h)
            .and_then(|r| r.get(resp))
            .and_then(|c| c.get(shock))
            .copied()
            .unwrap_or(0.0)
    }
}

/// Cholesky decomposition: returns lower-triangular L such that A = L L'.
fn cholesky_lower(a: &[Vec<f64>]) -> CtsResult<Vec<Vec<f64>>> {
    let k = a.len();
    let mut l = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let mut s: f64 = a[i][j];
            for m in 0..j {
                s -= l[i][m] * l[j][m];
            }
            if i == j {
                if s <= 0.0 {
                    s = 1e-12;
                }
                l[i][j] = s.sqrt();
            } else {
                if l[j][j].abs() < 1e-15 {
                    return Err("SVAR Cholesky: near-singular covariance".to_string());
                }
                l[i][j] = s / l[j][j];
            }
        }
    }
    Ok(l)
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Nonlinear Granger Causality
// ─────────────────────────────────────────────────────────────────────────────

/// Simple MLP for nonlinear Granger causality tests.
struct MlpRegressor {
    w1: Vec<f64>, // [hidden x input]
    b1: Vec<f64>, // [hidden]
    w2: Vec<f64>, // [1 x hidden]
    b2: f64,
    input_dim: usize,
    hidden_dim: usize,
}

impl MlpRegressor {
    fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let sc1 = (2.0_f64 / input_dim as f64).sqrt();
        let sc2 = (2.0_f64 / hidden_dim as f64).sqrt();
        Self {
            w1: (0..hidden_dim * input_dim)
                .map(|_| (rng.random::<f64>() - 0.5) * sc1)
                .collect(),
            b1: vec![0.0_f64; hidden_dim],
            w2: (0..hidden_dim)
                .map(|_| (rng.random::<f64>() - 0.5) * sc2)
                .collect(),
            b2: 0.0,
            input_dim,
            hidden_dim,
        }
    }

    fn forward(&self, x: &[f64]) -> f64 {
        let hd = self.hidden_dim;
        let id = self.input_dim;
        let mut h = vec![0.0_f64; hd];
        for i in 0..hd {
            let mut val = self.b1[i];
            for j in 0..id.min(x.len()) {
                val += self.w1[i * id + j] * x[j];
            }
            h[i] = val.max(0.0); // ReLU
        }
        let out: f64 = h.iter().zip(&self.w2).map(|(hi, wi)| hi * wi).sum::<f64>() + self.b2;
        out
    }

    /// Mini-batch gradient descent (MSE loss). Returns final MSE.
    fn fit(&mut self, x_mat: &[Vec<f64>], y: &[f64], lr: f64, epochs: usize) -> f64 {
        let n = y.len().min(x_mat.len());
        if n == 0 {
            return 0.0;
        }
        for _ in 0..epochs {
            let mut dw1 = vec![0.0_f64; self.hidden_dim * self.input_dim];
            let mut db1 = vec![0.0_f64; self.hidden_dim];
            let mut dw2 = vec![0.0_f64; self.hidden_dim];
            let mut db2 = 0.0_f64;

            for i in 0..n {
                let x = &x_mat[i];
                let hd = self.hidden_dim;
                let id = self.input_dim;
                // Forward
                let mut h = vec![0.0_f64; hd];
                let mut h_pre = vec![0.0_f64; hd];
                for j in 0..hd {
                    let mut val = self.b1[j];
                    for l in 0..id.min(x.len()) {
                        val += self.w1[j * id + l] * x[l];
                    }
                    h_pre[j] = val;
                    h[j] = val.max(0.0);
                }
                let out: f64 =
                    h.iter().zip(&self.w2).map(|(hj, wj)| hj * wj).sum::<f64>() + self.b2;
                let err = out - y[i];
                // Backprop
                db2 += err;
                for j in 0..hd {
                    dw2[j] += err * h[j];
                    let d_h = err * self.w2[j] * if h_pre[j] > 0.0 { 1.0 } else { 0.0 };
                    db1[j] += d_h;
                    for l in 0..id.min(x.len()) {
                        dw1[j * id + l] += d_h * x[l];
                    }
                }
            }
            let inv_n = lr / n as f64;
            for j in 0..self.hidden_dim * self.input_dim {
                self.w1[j] -= inv_n * dw1[j];
            }
            for j in 0..self.hidden_dim {
                self.b1[j] -= inv_n * db1[j];
                self.w2[j] -= inv_n * dw2[j];
            }
            self.b2 -= inv_n * db2;
        }
        // Compute MSE
        let mse: f64 = (0..n)
            .map(|i| (self.forward(&x_mat[i]) - y[i]).powi(2))
            .sum::<f64>()
            / n as f64;
        mse
    }
}

/// Neural Granger Test: replace linear AR with MLP and compare restricted vs
/// unrestricted MSE (Montalto et al. 2015 spirit).
///
/// H0: X does not Granger-cause Y.
/// Restricted model: Y ~ MLP(Y_lags).
/// Unrestricted model: Y ~ MLP(Y_lags, X_lags).
/// Test statistic: proportional log-ratio of MSEs.
pub struct NeuralGrangerTest {
    /// Number of lags.
    pub lags: usize,
    /// Hidden units in the MLP.
    pub hidden_dim: usize,
    /// Learning rate.
    pub lr: f64,
    /// Training epochs.
    pub epochs: usize,
}

impl NeuralGrangerTest {
    /// Create NeuralGrangerTest.
    pub fn new(lags: usize, hidden_dim: usize, lr: f64, epochs: usize) -> Self {
        Self {
            lags: lags.max(1),
            hidden_dim: hidden_dim.max(2),
            lr: lr.max(1e-6),
            epochs: epochs.max(10),
        }
    }

    /// Test whether `x` Granger-causes `y` (nonlinear, neural).
    /// Returns (log_ratio_statistic, mse_restricted, mse_unrestricted).
    pub fn test(&self, x: &[f64], y: &[f64], seed: u64) -> CtsResult<(f64, f64, f64)> {
        let n = x.len().min(y.len());
        let p = self.lags;
        if n <= p + 2 {
            return Err(format!("NeuralGranger: too few samples n={} p={}", n, p));
        }
        let n_eff = n - p;

        // Build restricted design (Y lags only)
        let mut x_restr: Vec<Vec<f64>> = Vec::with_capacity(n_eff);
        let mut x_unres: Vec<Vec<f64>> = Vec::with_capacity(n_eff);
        let mut y_vec: Vec<f64> = Vec::with_capacity(n_eff);

        for t in p..n {
            let mut row_r = Vec::with_capacity(p);
            let mut row_u = Vec::with_capacity(2 * p);
            for lag in 1..=p {
                row_r.push(y[t - lag]);
                row_u.push(y[t - lag]);
            }
            for lag in 1..=p {
                row_u.push(x[t - lag]);
            }
            x_restr.push(row_r);
            x_unres.push(row_u);
            y_vec.push(y[t]);
        }

        let mut mlp_r = MlpRegressor::new(p, self.hidden_dim, seed);
        let mse_r = mlp_r.fit(&x_restr, &y_vec, self.lr, self.epochs);

        let mut mlp_u = MlpRegressor::new(2 * p, self.hidden_dim, seed + 1);
        let mse_u = mlp_u.fit(&x_unres, &y_vec, self.lr, self.epochs);

        let ratio = if mse_u > 1e-20 {
            (mse_r / mse_u).ln()
        } else {
            0.0
        };
        Ok((ratio, mse_r, mse_u))
    }
}

/// Kernel-based Granger causality test using HSIC (Hilbert-Schmidt Independence Criterion).
///
/// Tests if X_{t-lag} provides additional predictive information about Y_t
/// given Y_{t-lag}, via kernel independence testing (Marinazzo et al. 2008 spirit).
pub struct KernelGrangerTest {
    /// Number of lags.
    pub lags: usize,
    /// Bandwidth for RBF kernel (auto-selected via median heuristic if 0.0).
    pub bandwidth: f64,
    /// Regularization parameter for HSIC centering.
    pub regularization: f64,
}

impl KernelGrangerTest {
    /// Create KernelGrangerTest.
    pub fn new(lags: usize, bandwidth: f64, regularization: f64) -> Self {
        Self {
            lags: lags.max(1),
            bandwidth: bandwidth.max(0.0),
            regularization: regularization.max(1e-10),
        }
    }

    /// Compute RBF kernel matrix for a 1-D vector.
    fn rbf_kernel(v: &[f64], bw: f64) -> Vec<Vec<f64>> {
        let n = v.len();
        let bw2 = 2.0 * bw * bw;
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| (-(v[i] - v[j]).powi(2) / bw2).exp())
                    .collect()
            })
            .collect()
    }

    /// Center a kernel matrix: K_c = H K H, H = I - 1/n * 1 1'.
    fn center_kernel(k: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = k.len();
        if n == 0 {
            return Vec::new();
        }
        let row_means: Vec<f64> = k.iter().map(|row| row.iter().sum::<f64>() / n as f64).collect();
        let grand_mean: f64 = row_means.iter().sum::<f64>() / n as f64;
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| k[i][j] - row_means[i] - row_means[j] + grand_mean)
                    .collect()
            })
            .collect()
    }

    /// HSIC statistic: Tr(K_c L_c) / (n-1)^2.
    fn hsic(kc: &[Vec<f64>], lc: &[Vec<f64>]) -> f64 {
        let n = kc.len().min(lc.len());
        if n < 2 {
            return 0.0;
        }
        let tr: f64 = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| kc[i][j] * lc[j][i])
                    .sum::<f64>()
            })
            .sum();
        tr / (n - 1).pow(2) as f64
    }

    /// Estimate median pairwise distance for bandwidth selection.
    fn median_bandwidth(v: &[f64]) -> f64 {
        let n = v.len();
        if n < 2 {
            return 1.0;
        }
        let mut dists: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                dists.push((v[i] - v[j]).abs());
            }
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dists[dists.len() / 2].max(1e-6)
    }

    /// Test whether `x` Granger-causes `y` using kernel HSIC.
    /// Returns (hsic_statistic, permutation_p_value) using `n_perm` permutations.
    pub fn test(
        &self,
        x: &[f64],
        y: &[f64],
        n_perm: usize,
        seed: u64,
    ) -> CtsResult<(f64, f64)> {
        let n = x.len().min(y.len());
        let p = self.lags;
        if n <= p + 2 {
            return Err(format!("KernelGranger: too few samples n={} p={}", n, p));
        }
        let n_eff = n - p;

        // Collect (y_t, y_{t-p}, x_{t-p}) triples
        let y_curr: Vec<f64> = (p..n).map(|t| y[t]).collect();
        let y_prev: Vec<f64> = (p..n).map(|t| y[t - p]).collect();
        let x_prev: Vec<f64> = (p..n).map(|t| x[t - p]).collect();

        // Bandwidth via median heuristic
        let bw_y = if self.bandwidth > 0.0 {
            self.bandwidth
        } else {
            Self::median_bandwidth(&y_curr).max(1e-6)
        };
        let bw_x = if self.bandwidth > 0.0 {
            self.bandwidth
        } else {
            Self::median_bandwidth(&x_prev).max(1e-6)
        };

        // Residualize y_curr from y_prev via kernel regression (Nadaraya-Watson)
        let bw_yp = Self::median_bandwidth(&y_prev).max(1e-6);
        let resid_y: Vec<f64> = (0..n_eff)
            .map(|t| {
                let mut num = 0.0_f64;
                let mut den = 0.0_f64;
                for s in 0..n_eff {
                    let w = (-(y_prev[t] - y_prev[s]).powi(2) / (2.0 * bw_yp * bw_yp)).exp();
                    num += w * y_curr[s];
                    den += w;
                }
                let yhat = if den > 1e-20 { num / den } else { y_curr[t] };
                y_curr[t] - yhat
            })
            .collect();

        let resid_x: Vec<f64> = (0..n_eff)
            .map(|t| {
                let mut num = 0.0_f64;
                let mut den = 0.0_f64;
                for s in 0..n_eff {
                    let w = (-(y_prev[t] - y_prev[s]).powi(2) / (2.0 * bw_yp * bw_yp)).exp();
                    num += w * x_prev[s];
                    den += w;
                }
                let xhat = if den > 1e-20 { num / den } else { x_prev[t] };
                x_prev[t] - xhat
            })
            .collect();

        // Compute HSIC between residuals
        let ky = Self::rbf_kernel(&resid_y, bw_y);
        let kx = Self::rbf_kernel(&resid_x, bw_x);
        let kyc = Self::center_kernel(&ky);
        let kxc = Self::center_kernel(&kx);
        let hsic_obs = Self::hsic(&kyc, &kxc);

        // Permutation test
        let n_perm = n_perm.max(99);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut count_above = 0usize;
        let mut perm_x = resid_x.clone();
        for _ in 0..n_perm {
            // Fisher-Yates shuffle
            for i in (1..n_eff).rev() {
                let j = rng.random_range(0..=i);
                perm_x.swap(i, j);
            }
            let kxp = Self::rbf_kernel(&perm_x, bw_x);
            let kxpc = Self::center_kernel(&kxp);
            let hsic_perm = Self::hsic(&kyc, &kxpc);
            if hsic_perm >= hsic_obs {
                count_above += 1;
            }
        }
        let p_value = (count_above + 1) as f64 / (n_perm + 1) as f64;

        Ok((hsic_obs, p_value))
    }
}

/// Neural Transfer Entropy via MINE (Mutual Information Neural Estimator).
///
/// Estimates TE(X→Y) = I(Y_t; X_{t-lag} | Y_{t-lag}) using the Donsker-Varadhan
/// representation of KL divergence with a neural network critic.
pub struct TransferEntropyNeural {
    /// Number of lags.
    pub lags: usize,
    /// Hidden units in the MINE critic network.
    pub hidden_dim: usize,
    /// Training iterations.
    pub iterations: usize,
    /// Learning rate.
    pub lr: f64,
}

impl TransferEntropyNeural {
    /// Create TransferEntropyNeural estimator.
    pub fn new(lags: usize, hidden_dim: usize, iterations: usize, lr: f64) -> Self {
        Self {
            lags: lags.max(1),
            hidden_dim: hidden_dim.max(4),
            iterations: iterations.max(50),
            lr: lr.max(1e-6),
        }
    }

    /// Estimate TE(x → y) using neural MINE.
    /// Returns estimated transfer entropy (lower-bounded, clipped at 0).
    pub fn estimate(&self, x: &[f64], y: &[f64], seed: u64) -> CtsResult<f64> {
        let n = x.len().min(y.len());
        let p = self.lags;
        if n <= p + 2 {
            return Err(format!("TE-Neural: too few samples n={} p={}", n, p));
        }
        let n_eff = n - p;
        // Joint samples: (y_t, y_{t-p}, x_{t-p})
        let y_t: Vec<f64> = (p..n).map(|t| y[t]).collect();
        let y_prev: Vec<f64> = (p..n).map(|t| y[t - p]).collect();
        let x_prev: Vec<f64> = (p..n).map(|t| x[t - p]).collect();

        // MINE critic: T(y_t, y_prev, x_prev) - scalar
        // Joint: (y_t, y_prev, x_prev) as-is
        // Marginal: shuffle x_prev (independent)
        let mut rng = StdRng::seed_from_u64(seed);
        let hd = self.hidden_dim;
        // Weights: 3-input → hd → 1
        let sc1 = (2.0_f64 / 3.0).sqrt();
        let sc2 = (2.0_f64 / hd as f64).sqrt();
        let mut w1: Vec<f64> = (0..hd * 3)
            .map(|_| (rng.random::<f64>() - 0.5) * sc1)
            .collect();
        let mut b1: Vec<f64> = vec![0.0_f64; hd];
        let mut w2: Vec<f64> = (0..hd)
            .map(|_| (rng.random::<f64>() - 0.5) * sc2)
            .collect();
        let mut b2 = 0.0_f64;

        let critic = |inp: &[f64; 3],
                      w1: &[f64],
                      b1: &[f64],
                      w2: &[f64],
                      b2: f64|
         -> f64 {
            let mut h = vec![0.0_f64; hd];
            for i in 0..hd {
                let val = b1[i]
                    + w1[i * 3] * inp[0]
                    + w1[i * 3 + 1] * inp[1]
                    + w1[i * 3 + 2] * inp[2];
                h[i] = val.max(0.0);
            }
            h.iter().zip(w2.iter()).map(|(hi, wi)| hi * wi).sum::<f64>() + b2
        };

        let mut shuffled_x = x_prev.clone();
        for _ in 0..self.iterations {
            // Shuffle x_prev for marginal
            for i in (1..n_eff).rev() {
                let j = rng.random_range(0..=i);
                shuffled_x.swap(i, j);
            }
            // Estimate E_joint[T] and log E_marginal[exp(T)]
            let mut sum_joint = 0.0_f64;
            let mut sum_exp_marg = 0.0_f64;
            // Gradients
            let mut dw1 = vec![0.0_f64; hd * 3];
            let mut db1 = vec![0.0_f64; hd];
            let mut dw2 = vec![0.0_f64; hd];
            let mut db2_g = 0.0_f64;

            for i in 0..n_eff {
                let joint_inp = [y_t[i], y_prev[i], x_prev[i]];
                let marg_inp = [y_t[i], y_prev[i], shuffled_x[i]];
                let tj = critic(&joint_inp, &w1, &b1, &w2, b2);
                let tm = critic(&marg_inp, &w1, &b1, &w2, b2);
                sum_joint += tj;
                sum_exp_marg += tm.exp();
            }
            let mine_obj = sum_joint / n_eff as f64
                - (sum_exp_marg / n_eff as f64).ln();
            let _ = mine_obj;

            // Gradient update: maximize MINE objective
            let z_marg = sum_exp_marg.max(1e-20);
            for i in 0..n_eff {
                let joint_inp = [y_t[i], y_prev[i], x_prev[i]];
                let marg_inp = [y_t[i], y_prev[i], shuffled_x[i]];
                let tm = critic(&marg_inp, &w1, &b1, &w2, b2);
                let grad_j = 1.0 / n_eff as f64;
                let grad_m = -tm.exp() / z_marg;
                // Backprop for joint
                let mut hj = vec![0.0_f64; hd];
                for k in 0..hd {
                    let val = b1[k]
                        + w1[k * 3] * joint_inp[0]
                        + w1[k * 3 + 1] * joint_inp[1]
                        + w1[k * 3 + 2] * joint_inp[2];
                    hj[k] = val.max(0.0);
                }
                for k in 0..hd {
                    dw2[k] += grad_j * hj[k];
                    let dhk = grad_j * w2[k] * if hj[k] > 0.0 { 1.0 } else { 0.0 };
                    db1[k] += dhk;
                    dw1[k * 3] += dhk * joint_inp[0];
                    dw1[k * 3 + 1] += dhk * joint_inp[1];
                    dw1[k * 3 + 2] += dhk * joint_inp[2];
                }
                db2_g += grad_j;
                // Backprop for marginal
                let mut hm = vec![0.0_f64; hd];
                for k in 0..hd {
                    let val = b1[k]
                        + w1[k * 3] * marg_inp[0]
                        + w1[k * 3 + 1] * marg_inp[1]
                        + w1[k * 3 + 2] * marg_inp[2];
                    hm[k] = val.max(0.0);
                }
                for k in 0..hd {
                    dw2[k] += grad_m * hm[k];
                    let dhk = grad_m * w2[k] * if hm[k] > 0.0 { 1.0 } else { 0.0 };
                    db1[k] += dhk;
                    dw1[k * 3] += dhk * marg_inp[0];
                    dw1[k * 3 + 1] += dhk * marg_inp[1];
                    dw1[k * 3 + 2] += dhk * marg_inp[2];
                }
                db2_g += grad_m;
            }
            // Gradient ascent (maximize)
            for k in 0..hd * 3 {
                w1[k] += self.lr * dw1[k];
            }
            for k in 0..hd {
                b1[k] += self.lr * db1[k];
                w2[k] += self.lr * dw2[k];
            }
            b2 += self.lr * db2_g;
        }

        // Final estimate
        let mut sum_j = 0.0_f64;
        let mut sum_em = 0.0_f64;
        for i in (1..n_eff).rev() {
            let j = rng.random_range(0..=i);
            shuffled_x.swap(i, j);
        }
        for i in 0..n_eff {
            let joint_inp = [y_t[i], y_prev[i], x_prev[i]];
            let marg_inp = [y_t[i], y_prev[i], shuffled_x[i]];
            sum_j += critic(&joint_inp, &w1, &b1, &w2, b2);
            sum_em += critic(&marg_inp, &w1, &b1, &w2, b2).exp();
        }
        let te = (sum_j / n_eff as f64 - (sum_em / n_eff as f64).ln()).max(0.0);
        Ok(te)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Causal Bandits in Time Series
// ─────────────────────────────────────────────────────────────────────────────

/// Causal bandit environment: interventions on variables with causal graph structure.
///
/// The reward is a linear function of downstream variables affected by the intervention.
pub struct CausalBanditEnv {
    /// Number of variables (arms).
    pub n_vars: usize,
    /// Causal adjacency matrix: `adj[i][j]` = weight of i → j.
    pub adj: Vec<Vec<f64>>,
    /// Reward weights: linear combination of variable values.
    pub reward_weights: Vec<f64>,
    /// Observation noise std.
    pub noise_std: f64,
}

impl CausalBanditEnv {
    /// Create environment with given causal graph and reward weights.
    pub fn new(
        adj: Vec<Vec<f64>>,
        reward_weights: Vec<f64>,
        noise_std: f64,
    ) -> CtsResult<Self> {
        let k = adj.len();
        if k == 0 {
            return Err("CausalBanditEnv: empty adjacency".to_string());
        }
        for row in &adj {
            if row.len() != k {
                return Err("CausalBanditEnv: adj must be square".to_string());
            }
        }
        Ok(Self {
            n_vars: k,
            adj,
            reward_weights: reward_weights
                .into_iter()
                .chain(std::iter::repeat(0.0))
                .take(k)
                .collect(),
            noise_std: noise_std.max(0.0),
        })
    }

    /// Simulate an intervention: set variable `action` to value `val`,
    /// propagate through the causal graph (one-step forward), return reward.
    pub fn intervene(&self, action: usize, val: f64, rng: &mut StdRng) -> f64 {
        let k = self.n_vars;
        let mut state = vec![0.0_f64; k];
        if action < k {
            state[action] = val;
        }
        // One forward pass through the graph (topological — no cycles assumed)
        for j in 0..k {
            if j == action {
                continue;
            }
            state[j] = (0..k).map(|i| self.adj[i][j] * state[i]).sum::<f64>();
        }
        let reward: f64 = state
            .iter()
            .zip(&self.reward_weights)
            .map(|(s, w)| s * w)
            .sum();
        let noise = if self.noise_std > 1e-15 {
            // Box-Muller
            let u1: f64 = rng.random::<f64>().max(1e-300);
            let u2: f64 = rng.random::<f64>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * self.noise_std
        } else {
            0.0
        };
        reward + noise
    }
}

/// Causal UCB bandit agent: uses causal graph to compute expected intervention values.
///
/// Each arm corresponds to an intervention on one variable. The expected reward
/// is estimated by causal reasoning (do-calculus approximation: E[R|do(X_i=v)]).
pub struct CausalUcbAgent {
    /// Number of arms.
    pub n_arms: usize,
    /// Empirical mean reward per arm.
    pub means: Vec<f64>,
    /// Pull count per arm.
    pub counts: Vec<usize>,
    /// UCB exploration parameter.
    pub c: f64,
    /// Total pulls.
    pub total: usize,
}

impl CausalUcbAgent {
    /// Create CausalUcbAgent with `n_arms` arms.
    pub fn new(n_arms: usize, c: f64) -> Self {
        Self {
            n_arms: n_arms.max(1),
            means: vec![0.0_f64; n_arms.max(1)],
            counts: vec![0usize; n_arms.max(1)],
            c: c.max(0.0),
            total: 0,
        }
    }

    /// Select arm with highest UCB score.
    pub fn select(&self) -> usize {
        let t = self.total.max(1) as f64;
        self.means
            .iter()
            .zip(&self.counts)
            .enumerate()
            .map(|(i, (&mu, &n))| {
                let ucb = if n == 0 {
                    f64::INFINITY
                } else {
                    mu + self.c * (t.ln() / n as f64).sqrt()
                };
                (i, ucb)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Update after observing reward `r` from arm `arm`.
    pub fn update(&mut self, arm: usize, r: f64) {
        if arm >= self.n_arms {
            return;
        }
        self.counts[arm] += 1;
        self.total += 1;
        let n = self.counts[arm] as f64;
        self.means[arm] += (r - self.means[arm]) / n;
    }

    /// Run `n_rounds` of bandit with the given environment.
    pub fn run(
        &mut self,
        env: &CausalBanditEnv,
        n_rounds: usize,
        intervention_val: f64,
        seed: u64,
    ) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rewards = Vec::with_capacity(n_rounds);
        for _ in 0..n_rounds {
            let arm = self.select();
            let r = env.intervene(arm, intervention_val, &mut rng);
            self.update(arm, r);
            rewards.push(r);
        }
        rewards
    }
}

/// Thompson Sampling agent with causal posterior update.
///
/// Maintains a Gaussian posterior (μ, σ²) for each arm, updated via
/// Bayesian update with known noise variance.
pub struct CausalThompsonSampling {
    /// Number of arms.
    pub n_arms: usize,
    /// Posterior mean for each arm.
    pub mu: Vec<f64>,
    /// Posterior variance for each arm.
    pub sigma2: Vec<f64>,
    /// Prior variance.
    pub prior_var: f64,
    /// Likelihood noise variance.
    pub noise_var: f64,
}

impl CausalThompsonSampling {
    /// Create agent with Gaussian prior N(0, prior_var) for each arm.
    pub fn new(n_arms: usize, prior_var: f64, noise_var: f64) -> Self {
        let k = n_arms.max(1);
        Self {
            n_arms: k,
            mu: vec![0.0_f64; k],
            sigma2: vec![prior_var.max(1e-10); k],
            prior_var: prior_var.max(1e-10),
            noise_var: noise_var.max(1e-10),
        }
    }

    /// Sample from posterior and select the best arm.
    pub fn select(&self, rng: &mut StdRng) -> usize {
        self.mu
            .iter()
            .zip(&self.sigma2)
            .enumerate()
            .map(|(i, (&mu, &sig2))| {
                let u1: f64 = rng.random::<f64>().max(1e-300);
                let u2: f64 = rng.random::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                (i, mu + sig2.sqrt() * z)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Bayesian update: posterior conjugate update for Gaussian likelihood.
    pub fn update(&mut self, arm: usize, reward: f64) {
        if arm >= self.n_arms {
            return;
        }
        // Conjugate Gaussian update: σ²_post = 1/(1/σ²_prior + 1/σ²_noise)
        let sigma2_post = 1.0 / (1.0 / self.sigma2[arm] + 1.0 / self.noise_var);
        let mu_post = sigma2_post * (self.mu[arm] / self.sigma2[arm] + reward / self.noise_var);
        self.sigma2[arm] = sigma2_post;
        self.mu[arm] = mu_post;
    }

    /// Run `n_rounds` using the causal bandit environment.
    pub fn run(
        &mut self,
        env: &CausalBanditEnv,
        n_rounds: usize,
        intervention_val: f64,
        seed: u64,
    ) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rewards = Vec::with_capacity(n_rounds);
        for _ in 0..n_rounds {
            let arm = self.select(&mut rng);
            let r = env.intervene(arm, intervention_val, &mut rng);
            self.update(arm, r);
            rewards.push(r);
        }
        rewards
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Time-Varying Causal Discovery
// ─────────────────────────────────────────────────────────────────────────────

/// Time-varying VAR model with kernel-weighted OLS.
///
/// Each coefficient β(t) is estimated locally via kernel-weighted regression
/// (Epanechnikov kernel), smoothing over nearby time points.
pub struct TvVarModel {
    /// Number of variables.
    pub k: usize,
    /// Number of lags.
    pub p: usize,
    /// Kernel bandwidth (in fraction of series length).
    pub bandwidth: f64,
    /// Estimated coefficient matrices at each time: `coeffs[t][lag][eq][var]`.
    pub coeffs: Vec<Vec<Vec<Vec<f64>>>>,
    /// Time indices at which coefficients are estimated.
    pub time_points: Vec<usize>,
}

impl TvVarModel {
    /// Fit TV-VAR(p) on multivariate data at `n_grid` equidistant time points.
    ///
    /// `data[t][v]` is variable `v` at time `t`.
    pub fn fit(data: &[Vec<f64>], p: usize, bandwidth: f64, n_grid: usize) -> CtsResult<Self> {
        let n = data.len();
        if n == 0 {
            return Err("TvVAR: empty data".to_string());
        }
        let k = data[0].len();
        if k == 0 {
            return Err("TvVAR: zero variables".to_string());
        }
        if n <= p {
            return Err(format!("TvVAR: n={} <= p={}", n, p));
        }
        let bw = bandwidth.clamp(0.01, 1.0) * n as f64;
        let n_eff = n - p;
        let n_grid = n_grid.max(1).min(n_eff);

        // Grid points (in original time indices p..n)
        let time_points: Vec<usize> = (0..n_grid)
            .map(|i| p + i * n_eff / n_grid)
            .collect();

        // Epanechnikov kernel: K(u) = 3/4 (1-u^2) for |u| <= 1
        let epan = |u: f64| -> f64 {
            if u.abs() <= 1.0 {
                0.75 * (1.0 - u * u)
            } else {
                0.0
            }
        };

        let mut coeffs: Vec<Vec<Vec<Vec<f64>>>> = Vec::with_capacity(n_grid);

        for &t0 in &time_points {
            // Compute kernel weights for each observation in p..n
            let weights: Vec<f64> = (p..n)
                .map(|t| epan((t as f64 - t0 as f64) / bw))
                .collect();

            // Build weighted design matrix
            let n_cols = 1 + p * k;
            let mut design: Vec<Vec<f64>> = Vec::with_capacity(n_eff);
            let mut w_vec: Vec<f64> = Vec::with_capacity(n_eff);
            for (idx, t) in (p..n).enumerate() {
                let wt = weights[idx];
                if wt <= 1e-15 {
                    // Include with near-zero weight to keep design matrix non-singular
                    let mut row = vec![1e-8_f64];
                    for lag in 1..=p {
                        for v in 0..k {
                            row.push(data[t - lag][v] * 1e-8);
                        }
                    }
                    design.push(row);
                    w_vec.push(1e-8);
                } else {
                    let sqrt_w = wt.sqrt();
                    let mut row = vec![sqrt_w];
                    for lag in 1..=p {
                        for v in 0..k {
                            row.push(data[t - lag][v] * sqrt_w);
                        }
                    }
                    design.push(row);
                    w_vec.push(wt);
                }
            }

            let mut coeff_t = vec![vec![vec![0.0_f64; k]; k]; p];
            for eq in 0..k {
                let y_eq: Vec<f64> = (p..n)
                    .enumerate()
                    .map(|(idx, t)| data[t][eq] * w_vec[idx].sqrt())
                    .collect();
                let beta = match ols(&design, &y_eq) {
                    Ok(b) => b,
                    Err(_) => vec![0.0_f64; n_cols],
                };
                for lag in 0..p {
                    for v in 0..k {
                        coeff_t[lag][eq][v] = beta.get(1 + lag * k + v).copied().unwrap_or(0.0);
                    }
                }
            }
            coeffs.push(coeff_t);
        }

        Ok(Self {
            k,
            p,
            bandwidth,
            coeffs,
            time_points,
        })
    }

    /// Get the Granger causal strength of variable `src` → `dst` at grid point `idx`.
    /// Returns L2 norm of the coefficient vector across lags.
    pub fn causal_strength(&self, idx: usize, src: usize, dst: usize) -> f64 {
        if idx >= self.coeffs.len() || src >= self.k || dst >= self.k {
            return 0.0;
        }
        let coeff = &self.coeffs[idx];
        (0..self.p)
            .map(|lag| coeff.get(lag).and_then(|c| c.get(dst)).and_then(|e| e.get(src)).copied().unwrap_or(0.0).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Time-varying Granger causality test using kernel-OLS residuals.
///
/// Tests significance of x → y at each local time point using F-test on
/// kernel-weighted restricted vs. unrestricted residuals.
pub struct TvGrangerTest {
    /// Number of lags.
    pub lags: usize,
    /// Kernel bandwidth fraction.
    pub bandwidth: f64,
    /// Significance level.
    pub alpha: f64,
}

impl TvGrangerTest {
    /// Create TvGrangerTest.
    pub fn new(lags: usize, bandwidth: f64, alpha: f64) -> Self {
        Self {
            lags: lags.max(1),
            bandwidth: bandwidth.clamp(0.02, 1.0),
            alpha: alpha.clamp(1e-10, 1.0),
        }
    }

    /// Run TV-Granger test at `n_grid` equidistant time points.
    /// Returns Vec of (time_index, f_stat, p_value, is_causal) tuples.
    pub fn test(
        &self,
        x: &[f64],
        y: &[f64],
        n_grid: usize,
    ) -> CtsResult<Vec<(usize, f64, f64, bool)>> {
        let n = x.len().min(y.len());
        let p = self.lags;
        if n <= p + 2 {
            return Err(format!("TvGranger: too few samples n={} p={}", n, p));
        }
        let n_eff = n - p;
        let bw = self.bandwidth * n as f64;
        let n_grid = n_grid.max(1).min(n_eff);
        let time_points: Vec<usize> = (0..n_grid)
            .map(|i| p + i * n_eff / n_grid)
            .collect();

        let epan = |u: f64| -> f64 {
            if u.abs() <= 1.0 { 0.75 * (1.0 - u * u) } else { 0.0 }
        };

        let mut results = Vec::with_capacity(n_grid);

        for &t0 in &time_points {
            let weights: Vec<f64> = (p..n).map(|t| epan((t as f64 - t0 as f64) / bw)).collect();
            // Build weighted restricted and unrestricted designs
            let mut x_r: Vec<Vec<f64>> = Vec::new();
            let mut x_u: Vec<Vec<f64>> = Vec::new();
            let mut y_vec: Vec<f64> = Vec::new();
            for (idx, t) in (p..n).enumerate() {
                let sw = weights[idx].sqrt().max(1e-8);
                let mut row_r = vec![sw];
                let mut row_u = vec![sw];
                for lag in 1..=p {
                    row_r.push(y[t - lag] * sw);
                    row_u.push(y[t - lag] * sw);
                }
                for lag in 1..=p {
                    row_u.push(x[t - lag] * sw);
                }
                x_r.push(row_r);
                x_u.push(row_u);
                y_vec.push(y[t] * sw);
            }

            let n_obs = y_vec.len();
            let beta_r = match ols(&x_r, &y_vec) {
                Ok(b) => b,
                Err(_) => { results.push((t0, 0.0, 1.0, false)); continue; }
            };
            let beta_u = match ols(&x_u, &y_vec) {
                Ok(b) => b,
                Err(_) => { results.push((t0, 0.0, 1.0, false)); continue; }
            };
            let ssr_r = ssr(&residuals(&x_r, &y_vec, &beta_r));
            let ssr_u = ssr(&residuals(&x_u, &y_vec, &beta_u));

            let df1 = p as f64;
            let df2 = n_obs as f64 - 2.0 * p as f64 - 1.0;
            let f_stat = if df2 > 0.0 && ssr_u > 1e-20 {
                ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
            } else {
                0.0
            };
            let f_stat = f_stat.max(0.0);
            let p_val = chi2_p_value(f_stat * df1, df1);
            results.push((t0, f_stat, p_val, p_val < self.alpha));
        }

        Ok(results)
    }
}

/// Change-point detector for causal structure using CUSUM on Granger p-values.
///
/// Maintains a CUSUM statistic over time; when it exceeds a threshold, a
/// structural break (change in causal relationships) is declared.
pub struct ChangePointCausalDetector {
    /// Granger test lag order.
    pub lags: usize,
    /// Rolling window size for local Granger tests.
    pub window: usize,
    /// CUSUM threshold for declaring a change point.
    pub threshold: f64,
    /// Significance level for Granger tests.
    pub alpha: f64,
}

impl ChangePointCausalDetector {
    /// Create detector with given parameters.
    pub fn new(lags: usize, window: usize, threshold: f64, alpha: f64) -> Self {
        Self {
            lags: lags.max(1),
            window: window.max(10),
            threshold: threshold.max(0.1),
            alpha: alpha.clamp(1e-10, 1.0),
        }
    }

    /// Detect change points in Granger causal relationship x → y.
    ///
    /// Returns Vec of detected change point indices in the original series.
    pub fn detect(&self, x: &[f64], y: &[f64]) -> Vec<usize> {
        let n = x.len().min(y.len());
        let w = self.window;
        if n < w + self.lags {
            return Vec::new();
        }

        // Compute rolling p-values
        let mut p_values: Vec<f64> = Vec::new();
        let mut centers: Vec<usize> = Vec::new();
        let step = (w / 4).max(1);
        let mut start = 0;
        while start + w <= n {
            let end = start + w;
            match super::GrangerCausalityTest::test(&x[start..end], &y[start..end], self.lags) {
                Ok(r) => p_values.push(r.p_value_approx),
                Err(_) => p_values.push(0.5),
            }
            centers.push(start + w / 2);
            start += step;
        }

        if p_values.len() < 3 {
            return Vec::new();
        }

        // CUSUM on log(p): E_0[log p] ≈ log(0.5) under no causality
        let baseline = (self.alpha / 2.0).ln();
        let mut cusum = 0.0_f64;
        let mut change_points = Vec::new();
        let mut in_cp = false;

        for (i, &pv) in p_values.iter().enumerate() {
            let log_p = pv.max(1e-15).ln();
            cusum += log_p - baseline;
            // Reset cusum below 0 (only detect negative deviations = more causal)
            if cusum > 0.0 {
                cusum = 0.0;
            }
            let abs_cusum = cusum.abs();
            if abs_cusum > self.threshold && !in_cp {
                change_points.push(centers[i]);
                in_cp = true;
                cusum = 0.0;
            } else if abs_cusum < self.threshold / 2.0 {
                in_cp = false;
            }
        }

        change_points
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Advanced Causal TS Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Extended evaluation metrics for time-varying and advanced causal TS methods.
#[derive(Debug, Clone)]
pub struct CausalTsExtMetrics {
    /// TV-AUROC: area under the ROC curve for time-varying causal detection.
    pub tv_auroc: f64,
    /// Lag-specific F1: F1 score across all lag-specific edges.
    pub lag_f1: f64,
    /// Structural break detection rate: fraction of true breaks detected.
    pub break_detection_rate: f64,
    /// False alarm rate: fraction of non-break windows flagged as breaks.
    pub false_alarm_rate: f64,
    /// Mean SVAR FEVD at horizon h=10 (avg across all var-shock pairs).
    pub mean_fevd_h10: f64,
}

impl CausalTsExtMetrics {
    /// Compute TV-AUROC from predicted causal strengths and binary true labels.
    ///
    /// `scores[t]` = predicted causal strength at time t.
    /// `labels[t]` = true causal presence (1.0 = causal, 0.0 = not).
    pub fn tv_auroc(scores: &[f64], labels: &[f64]) -> f64 {
        let n = scores.len().min(labels.len());
        if n == 0 {
            return 0.5;
        }
        // Sort by score descending
        let mut pairs: Vec<(f64, f64)> = scores[..n]
            .iter()
            .zip(&labels[..n])
            .map(|(&s, &l)| (s, l))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let n_pos: f64 = labels[..n].iter().sum::<f64>().max(1e-15);
        let n_neg = (n as f64 - n_pos).max(1e-15);

        let mut auc = 0.0_f64;
        let mut tp = 0.0_f64;
        let mut fp = 0.0_f64;
        let mut prev_fp = 0.0_f64;
        let mut prev_tp = 0.0_f64;

        for (_, label) in &pairs {
            if *label > 0.5 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            // Trapezoidal rule
            auc += (fp / n_neg - prev_fp / n_neg) * (tp / n_pos + prev_tp / n_pos) / 2.0;
            prev_fp = fp;
            prev_tp = tp;
        }
        auc.clamp(0.0, 1.0)
    }

    /// Compute lag-specific F1 from predicted/true edge tensors.
    ///
    /// `pred[lag][i][j]` = predicted edge, `true[lag][i][j]` = ground truth.
    pub fn lag_specific_f1(pred: &[Vec<Vec<bool>>], truth: &[Vec<Vec<bool>>]) -> f64 {
        let n_lag = pred.len().min(truth.len());
        if n_lag == 0 {
            return 0.0;
        }
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;
        for lag in 0..n_lag {
            let k = pred[lag].len().min(truth[lag].len());
            for i in 0..k {
                let kj = pred[lag][i].len().min(truth[lag][i].len());
                for j in 0..kj {
                    match (pred[lag][i][j], truth[lag][i][j]) {
                        (true, true) => tp += 1,
                        (true, false) => fp += 1,
                        (false, true) => fn_ += 1,
                        (false, false) => {}
                    }
                }
            }
        }
        let prec = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let rec = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
        if prec + rec > 1e-15 {
            2.0 * prec * rec / (prec + rec)
        } else {
            0.0
        }
    }

    /// Compute break detection metrics.
    ///
    /// `detected` = change point indices returned by detector.
    /// `true_breaks` = true change point indices.
    /// `tolerance` = max allowed distance to count as a true detection.
    /// Returns (detection_rate, false_alarm_rate).
    pub fn break_detection_metrics(
        detected: &[usize],
        true_breaks: &[usize],
        tolerance: usize,
        n_windows: usize,
    ) -> (f64, f64) {
        if true_breaks.is_empty() {
            let far = if n_windows > 0 {
                detected.len() as f64 / n_windows as f64
            } else {
                0.0
            };
            return (0.0, far);
        }
        let mut n_detected = 0usize;
        for &tb in true_breaks {
            let found = detected
                .iter()
                .any(|&d| (d as isize - tb as isize).unsigned_abs() <= tolerance);
            if found {
                n_detected += 1;
            }
        }
        let n_fa = detected
            .iter()
            .filter(|&&d| {
                !true_breaks
                    .iter()
                    .any(|&tb| (d as isize - tb as isize).unsigned_abs() <= tolerance)
            })
            .count();
        let detection_rate = n_detected as f64 / true_breaks.len() as f64;
        let n_non_break = n_windows.saturating_sub(true_breaks.len()).max(1);
        let far = (n_fa as f64 / n_non_break as f64).min(1.0);
        (detection_rate, far)
    }
}
