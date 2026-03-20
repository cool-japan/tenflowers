//! Statistical distribution sampling: GMM, Copula, KDE, DRE, Bootstrap.

use super::math::{
    cdf_normal, cholesky, empirical_cdf, empirical_quantile, gauss_kernel, log_sum_exp,
    lower_tri_mv, mean_vec, mvn_log_pdf_diag, probit, sample_normal, var_vec,
};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

/// Gaussian Mixture Model sampler (EM-based fitting, diagonal covariance).
pub struct GaussianMixture {
    pub(super) weights: Vec<f64>,
    means: Vec<Vec<f64>>,
    variances: Vec<Vec<f64>>,
    dim: usize,
}

impl GaussianMixture {
    /// Fit GMM via EM on `data`. `n_components` mixture components, `n_iter` EM steps.
    pub fn fit(data: &[Vec<f64>], n_components: usize, n_iter: usize, seed: u64) -> Result<Self> {
        if data.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "GaussianMixture::fit",
                "data must not be empty",
            ));
        }
        let n = data.len();
        let dim = data[0].len();
        if dim == 0 || n_components == 0 {
            return Err(TensorError::invalid_argument_op(
                "GaussianMixture::fit",
                "dim and n_components must be > 0",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut means: Vec<Vec<f64>> = (0..n_components)
            .map(|_| data[rng.random::<u64>() as usize % n].clone())
            .collect();
        let mut variances: Vec<Vec<f64>> = vec![vec![1.0_f64; dim]; n_components];
        let mut weights: Vec<f64> = vec![1.0 / n_components as f64; n_components];
        let mut resp = vec![vec![0.0_f64; n_components]; n];
        for _iter in 0..n_iter {
            for i in 0..n {
                let log_ws: Vec<f64> = (0..n_components)
                    .map(|k| {
                        weights[k].max(1e-300).ln()
                            + mvn_log_pdf_diag(&data[i], &means[k], &variances[k])
                    })
                    .collect();
                let lse = log_sum_exp(&log_ws);
                for k in 0..n_components {
                    resp[i][k] = (log_ws[k] - lse).exp();
                }
            }
            for k in 0..n_components {
                let nk: f64 = resp.iter().map(|r| r[k]).sum::<f64>().max(1e-12);
                weights[k] = nk / n as f64;
                let mut new_mean = vec![0.0_f64; dim];
                for i in 0..n {
                    for d in 0..dim {
                        new_mean[d] += resp[i][k] * data[i][d];
                    }
                }
                for d in 0..dim {
                    new_mean[d] /= nk;
                }
                let mut new_var = vec![1e-6_f64; dim];
                for i in 0..n {
                    for d in 0..dim {
                        let diff = data[i][d] - new_mean[d];
                        new_var[d] += resp[i][k] * diff * diff;
                    }
                }
                for d in 0..dim {
                    new_var[d] = (new_var[d] / nk).max(1e-6);
                }
                means[k] = new_mean;
                variances[k] = new_var;
            }
        }
        Ok(Self {
            weights,
            means,
            variances,
            dim,
        })
    }

    /// Draw `n` samples from the fitted GMM.
    pub fn sample(&self, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let u: f64 = rng.random();
            let mut cum = 0.0_f64;
            let mut k = self.weights.len() - 1;
            for (i, &w) in self.weights.iter().enumerate() {
                cum += w;
                if u <= cum {
                    k = i;
                    break;
                }
            }
            let x: Vec<f64> = (0..self.dim)
                .map(|d| self.means[k][d] + self.variances[k][d].sqrt() * sample_normal(&mut rng))
                .collect();
            out.push(x);
        }
        out
    }

    /// Log-likelihood of a single observation under the fitted GMM.
    pub fn log_likelihood(&self, x: &[f64]) -> f64 {
        let log_ws: Vec<f64> = (0..self.weights.len())
            .map(|k| {
                self.weights[k].max(1e-300).ln()
                    + mvn_log_pdf_diag(x, &self.means[k], &self.variances[k])
            })
            .collect();
        log_sum_exp(&log_ws)
    }
}

/// Gaussian copula model (empirical marginals + Gaussian dependence structure).
pub struct CopulaModel {
    chol: Vec<f64>,
    sorted_marginals: Vec<Vec<f64>>,
    dim: usize,
}

impl CopulaModel {
    /// Fit copula to `data`.
    pub fn fit(data: &[Vec<f64>]) -> Result<Self> {
        if data.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "CopulaModel::fit",
                "data must not be empty",
            ));
        }
        let dim = data[0].len();
        if dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "CopulaModel::fit",
                "dim must be > 0",
            ));
        }
        let n = data.len();
        let sorted_marginals: Vec<Vec<f64>> = (0..dim)
            .map(|d| {
                let mut col: Vec<f64> = data.iter().map(|row| row[d]).collect();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                col
            })
            .collect();
        let z_scores: Vec<Vec<f64>> = data
            .iter()
            .map(|row| {
                (0..dim)
                    .map(|d| {
                        probit(empirical_cdf(&sorted_marginals[d], row[d]).clamp(1e-6, 1.0 - 1e-6))
                    })
                    .collect()
            })
            .collect();
        let mut corr = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            corr[i * dim + i] = 1.0;
            for j in (i + 1)..dim {
                let zi: Vec<f64> = z_scores.iter().map(|r| r[i]).collect();
                let zj: Vec<f64> = z_scores.iter().map(|r| r[j]).collect();
                let mi = mean_vec(&zi);
                let mj = mean_vec(&zj);
                let cov: f64 = zi
                    .iter()
                    .zip(zj.iter())
                    .map(|(a, b)| (a - mi) * (b - mj))
                    .sum::<f64>()
                    / n as f64;
                let si = var_vec(&zi).sqrt().max(1e-12);
                let sj = var_vec(&zj).sqrt().max(1e-12);
                let r = (cov / (si * sj)).clamp(-0.999, 0.999);
                corr[i * dim + j] = r;
                corr[j * dim + i] = r;
            }
        }
        let chol = cholesky(&corr, dim)?;
        Ok(Self {
            chol,
            sorted_marginals,
            dim,
        })
    }

    /// Draw `n` samples from the fitted copula.
    pub fn sample(&self, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let z_raw: Vec<f64> = (0..self.dim).map(|_| sample_normal(&mut rng)).collect();
            let z_corr = lower_tri_mv(&self.chol, &z_raw, self.dim);
            let x: Vec<f64> = (0..self.dim)
                .map(|d| {
                    let u = cdf_normal(z_corr[d]).clamp(0.0, 1.0);
                    empirical_quantile(&self.sorted_marginals[d], u)
                })
                .collect();
            out.push(x);
        }
        out
    }
}

/// Kernel Density Estimator with Gaussian product kernel.
pub struct KernelDensityEstimator {
    data: Vec<Vec<f64>>,
    bandwidth: f64,
    dim: usize,
}

impl KernelDensityEstimator {
    /// Fit KDE. Uses Scott's rule if `bandwidth <= 0`.
    pub fn fit(data: &[Vec<f64>], bandwidth: f64) -> Result<Self> {
        if data.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "KernelDensityEstimator::fit",
                "data must not be empty",
            ));
        }
        let dim = data[0].len();
        if dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "KernelDensityEstimator::fit",
                "dim must be > 0",
            ));
        }
        let bw = if bandwidth > 0.0 {
            bandwidth
        } else {
            let n = data.len() as f64;
            n.powf(-1.0 / (dim as f64 + 4.0))
        };
        Ok(Self {
            data: data.to_vec(),
            bandwidth: bw,
            dim,
        })
    }

    /// Evaluate the density estimate at `x`.
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        let n = self.data.len() as f64;
        let bw = self.bandwidth;
        let norm =
            (2.0 * std::f64::consts::PI).sqrt().powf(self.dim as f64) * bw.powf(self.dim as f64);
        let sum: f64 = self
            .data
            .iter()
            .map(|xi| {
                let exp_sum: f64 = xi
                    .iter()
                    .zip(x.iter())
                    .map(|(a, b)| {
                        let d = (a - b) / bw;
                        -0.5 * d * d
                    })
                    .sum();
                exp_sum.exp()
            })
            .sum();
        sum / (n * norm)
    }

    /// Draw `n` samples from the KDE.
    pub fn sample(&self, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let m = self.data.len();
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = rng.random::<u64>() as usize % m;
            let x: Vec<f64> = self.data[idx]
                .iter()
                .map(|&xi| xi + self.bandwidth * sample_normal(&mut rng))
                .collect();
            out.push(x);
        }
        out
    }
}

/// KLIEP density ratio estimator `p(x)/q(x)`.
pub struct DensityRatioEstimator {
    centers: Vec<Vec<f64>>,
    alpha: Vec<f64>,
    sigma: f64,
}

impl DensityRatioEstimator {
    /// Fit using P samples (numerator) and Q samples (denominator).
    pub fn fit(
        p_samples: &[Vec<f64>],
        q_samples: &[Vec<f64>],
        sigma: f64,
        seed: u64,
    ) -> Result<Self> {
        if p_samples.is_empty() || q_samples.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "DensityRatioEstimator::fit",
                "both P and Q samples must be non-empty",
            ));
        }
        let n_centers = p_samples.len().min(50);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..p_samples.len()).collect();
        for i in 0..n_centers {
            let j = i + rng.random::<u64>() as usize % (indices.len() - i);
            indices.swap(i, j);
        }
        let centers: Vec<Vec<f64>> = indices[..n_centers]
            .iter()
            .map(|&i| p_samples[i].clone())
            .collect();
        let nq = q_samples.len();
        let kernel_q: Vec<Vec<f64>> = q_samples
            .iter()
            .map(|qi| centers.iter().map(|c| gauss_kernel(qi, c, sigma)).collect())
            .collect();
        let kernel_p: Vec<Vec<f64>> = p_samples
            .iter()
            .map(|pi| centers.iter().map(|c| gauss_kernel(pi, c, sigma)).collect())
            .collect();
        let mut alpha = vec![1.0_f64 / n_centers as f64; n_centers];
        let lr = 0.01_f64;
        let n_iter = 500_usize;
        let np = p_samples.len() as f64;
        for _ in 0..n_iter {
            let norm: f64 = kernel_q
                .iter()
                .map(|k| {
                    let r: f64 = k.iter().zip(alpha.iter()).map(|(a, b)| a * b).sum();
                    r.max(0.0)
                })
                .sum::<f64>()
                / nq as f64;
            if norm > 1e-12 {
                for a in alpha.iter_mut() {
                    *a /= norm;
                }
            }
            for l in 0..n_centers {
                let grad: f64 = kernel_p
                    .iter()
                    .map(|kp| {
                        let r: f64 = kp
                            .iter()
                            .zip(alpha.iter())
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            .max(1e-12);
                        kp[l] / r
                    })
                    .sum::<f64>()
                    / np;
                alpha[l] = (alpha[l] + lr * grad).max(0.0);
            }
        }
        Ok(Self {
            centers,
            alpha,
            sigma,
        })
    }

    /// Estimate the density ratio `p(x)/q(x)` at `x`.
    pub fn ratio(&self, x: &[f64]) -> f64 {
        self.centers
            .iter()
            .zip(self.alpha.iter())
            .map(|(c, &a)| a * gauss_kernel(x, c, self.sigma))
            .sum::<f64>()
            .max(0.0)
    }
}

/// Nonparametric bootstrap resampler.
pub struct BootstrapSampler;

impl BootstrapSampler {
    /// Draw `n` bootstrap samples (with replacement) from `data`.
    pub fn sample(data: &[Vec<f64>], n: usize, seed: u64) -> Vec<Vec<f64>> {
        if data.is_empty() {
            return vec![];
        }
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| data[rng.random::<u64>() as usize % data.len()].clone())
            .collect()
    }

    /// Bootstrap confidence interval for a scalar statistic. Returns `(lower, upper)`.
    pub fn confidence_interval(
        data: &[Vec<f64>],
        stat_fn: &dyn Fn(&[Vec<f64>]) -> f64,
        n_boot: usize,
        alpha: f64,
        seed: u64,
    ) -> (f64, f64) {
        if data.is_empty() {
            return (0.0, 0.0);
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut stats: Vec<f64> = (0..n_boot)
            .map(|_| {
                let boot: Vec<Vec<f64>> = (0..data.len())
                    .map(|_| data[rng.random::<u64>() as usize % data.len()].clone())
                    .collect();
                stat_fn(&boot)
            })
            .collect();
        stats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo_idx = ((alpha / 2.0) * n_boot as f64).floor() as usize;
        let hi_idx = ((1.0 - alpha / 2.0) * n_boot as f64).floor() as usize;
        (
            stats[lo_idx.min(stats.len() - 1)],
            stats[hi_idx.min(stats.len() - 1)],
        )
    }
}
