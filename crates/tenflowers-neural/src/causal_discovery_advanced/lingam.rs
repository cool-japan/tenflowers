//! DirectLiNGAM — Linear Non-Gaussian Acyclic Model via residual independence.

use tenflowers_core::{Result, TensorError};

use super::shared::{mean_f32, ols_univariate, std_f32};

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for DirectLiNGAM.
#[derive(Debug, Clone)]
pub struct LinGamConfig {
    /// Number of observed variables.
    pub n_vars: usize,
    /// Maximum iterations for the causal ordering search.
    pub max_iter: usize,
}

impl Default for LinGamConfig {
    fn default() -> Self {
        Self {
            n_vars: 4,
            max_iter: 100,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public functions
// ─────────────────────────────────────────────────────────────────────────────

/// Bin-based mutual information approximation between two equal-length samples.
///
/// Uses equal-width histogram binning.  Returns bits (log₂ base).
pub fn mutual_information_approx(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    if n < 4 {
        return 0.0;
    }
    let n_bins = ((n as f32).sqrt() as usize).clamp(4, 32);

    let (xmin, xmax) = x[..n]
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let (ymin, ymax) = y[..n]
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));

    let xrange = (xmax - xmin).max(1e-12);
    let yrange = (ymax - ymin).max(1e-12);

    let mut joint = vec![0u32; n_bins * n_bins];
    let mut px = vec![0u32; n_bins];
    let mut py = vec![0u32; n_bins];

    for i in 0..n {
        let xi = (((x[i] - xmin) / xrange) * (n_bins as f32 - 1.0)) as usize;
        let yi = (((y[i] - ymin) / yrange) * (n_bins as f32 - 1.0)) as usize;
        let xi = xi.min(n_bins - 1);
        let yi = yi.min(n_bins - 1);
        joint[xi * n_bins + yi] += 1;
        px[xi] += 1;
        py[yi] += 1;
    }

    let nf = n as f32;
    let mut mi = 0.0f32;
    for i in 0..n_bins {
        for j in 0..n_bins {
            let pxy = joint[i * n_bins + j] as f32 / nf;
            if pxy < 1e-12 {
                continue;
            }
            let pxi = px[i] as f32 / nf;
            let pyj = py[j] as f32 / nf;
            if pxi < 1e-12 || pyj < 1e-12 {
                continue;
            }
            mi += pxy * (pxy / (pxi * pyj)).log2();
        }
    }
    mi.max(0.0)
}

/// Negentropy approximation using excess kurtosis as a non-Gaussianity measure.
///
/// J(x) ≈ (1/12) E\[x³\]² + (1/48) kurt(x)²  (Hyvärinen 1998 approximation).
pub fn entropy_ica(x: &[f32]) -> f32 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mu = mean_f32(x);
    let sigma = std_f32(x).max(1e-12);
    // Standardise
    let zs: Vec<f32> = x.iter().map(|&v| (v - mu) / sigma).collect();
    let nf = n as f32;
    let skew = zs.iter().map(|&z| z * z * z).sum::<f32>() / nf;
    let kurt = zs
        .iter()
        .map(|&z| {
            let z2 = z * z;
            z2 * z2
        })
        .sum::<f32>()
        / nf
        - 3.0;
    // Negentropy approximation
    skew * skew / 12.0 + kurt * kurt / 48.0
}

/// Run DirectLiNGAM on `data` (n_samples × n_vars matrix stored as Vec of rows).
///
/// Returns a weighted adjacency matrix `W` where `W[i][j]` represents the
/// coefficient of the causal effect i → j.  The algorithm:
/// 1. Iteratively identifies the causally earliest variable (most non-Gaussian residual).
/// 2. Removes its effect from all remaining variables.
/// 3. Repeats until all variables are ordered.
pub fn direct_lingam(data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
    let n = data.len();
    if n == 0 {
        return Err(TensorError::invalid_argument_op(
            "direct_lingam",
            "empty data",
        ));
    }
    let n_vars = data[0].len();
    if n_vars == 0 {
        return Err(TensorError::invalid_argument_op(
            "direct_lingam",
            "zero variables",
        ));
    }

    let mut residuals: Vec<Vec<f32>> = (0..n_vars)
        .map(|j| {
            data.iter()
                .map(|row| if j < row.len() { row[j] } else { 0.0 })
                .collect()
        })
        .collect();

    let mut remaining: Vec<usize> = (0..n_vars).collect();
    let mut adj = vec![vec![0.0f32; n_vars]; n_vars];

    while !remaining.is_empty() {
        // Find the variable with highest non-Gaussianity (negentropy) in residuals
        let best = remaining.iter().copied().max_by(|&a, &b| {
            let na = entropy_ica(&residuals[a]);
            let nb = entropy_ica(&residuals[b]);
            na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let root = match best {
            Some(v) => v,
            None => break,
        };

        remaining.retain(|&v| v != root);

        // Regress root's residual out of all remaining variables
        let x_root: Vec<f32> = residuals[root].clone();
        for &j in &remaining {
            let y_j: Vec<f32> = residuals[j].clone();
            let (slope, _) = ols_univariate(&x_root, &y_j);
            // Store weight: root → j
            adj[root][j] = slope;
            // Update residual of j
            let new_res: Vec<f32> = y_j
                .iter()
                .zip(x_root.iter())
                .map(|(&yv, &xv)| yv - slope * xv)
                .collect();
            residuals[j] = new_res;
        }
    }

    Ok(adj)
}
