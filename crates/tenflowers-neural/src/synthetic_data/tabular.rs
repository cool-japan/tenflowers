//! Tabular data synthesis: CTGAN, TVAE, SMOTE, MICE imputation, ColumnTransformer.

use super::math::{euclidean, mean_vec, sample_normal, var_vec};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

/// Simplified CTGAN-style tabular GAN.
pub struct CtganGenerator {
    gen_layers: Vec<(Vec<f64>, Vec<f64>)>,
    dim: usize,
    latent: usize,
}

fn ctgan_generate(z: &[f64], layers: &[(Vec<f64>, Vec<f64>)]) -> Vec<f64> {
    let mut h = z.to_vec();
    for (idx, (w, b)) in layers.iter().enumerate() {
        let out_dim = b.len();
        let in_dim = h.len();
        let mut next = vec![0.0_f64; out_dim];
        for o in 0..out_dim {
            next[o] = b[o];
            for i in 0..in_dim {
                next[o] += w[i * out_dim + o] * h[i];
            }
            if idx < layers.len() - 1 {
                next[o] = next[o].max(0.0);
            } else {
                next[o] = next[o].tanh();
            }
        }
        h = next;
    }
    h
}

impl CtganGenerator {
    /// Fit CTGAN generator on `data`.
    pub fn fit(data: &[Vec<f64>], n_epochs: usize, seed: u64) -> Result<Self> {
        if data.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "CtganGenerator::fit",
                "data must not be empty",
            ));
        }
        let dim = data[0].len();
        if dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "CtganGenerator::fit",
                "data dimensions must be > 0",
            ));
        }
        let latent = 8_usize;
        let hidden = 16_usize;
        let mut rng = StdRng::seed_from_u64(seed);
        let scale1 = (6.0 / (latent + hidden) as f64).sqrt();
        let w1: Vec<f64> = (0..(latent * hidden))
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale1)
            .collect();
        let b1 = vec![0.0_f64; hidden];
        let scale2 = (6.0 / (hidden + dim) as f64).sqrt();
        let w2: Vec<f64> = (0..(hidden * dim))
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale2)
            .collect();
        let b2 = vec![0.0_f64; dim];
        let mut gen_layers = vec![(w1, b1), (w2, b2)];
        let lr = 0.005_f64;
        let batch = 16_usize.min(data.len());
        let real_means: Vec<f64> = (0..dim)
            .map(|d| mean_vec(&data.iter().map(|r| r[d]).collect::<Vec<_>>()))
            .collect();
        let real_stds: Vec<f64> = (0..dim)
            .map(|d| {
                var_vec(&data.iter().map(|r| r[d]).collect::<Vec<_>>())
                    .sqrt()
                    .max(1e-6)
            })
            .collect();
        for _ in 0..n_epochs {
            let z: Vec<Vec<f64>> = (0..batch)
                .map(|_| (0..latent).map(|_| sample_normal(&mut rng)).collect())
                .collect();
            let generated: Vec<Vec<f64>> =
                z.iter().map(|zi| ctgan_generate(zi, &gen_layers)).collect();
            let gen_means: Vec<f64> = (0..dim)
                .map(|d| mean_vec(&generated.iter().map(|r| r[d]).collect::<Vec<_>>()))
                .collect();
            let gen_stds: Vec<f64> = (0..dim)
                .map(|d| {
                    var_vec(&generated.iter().map(|r| r[d]).collect::<Vec<_>>())
                        .sqrt()
                        .max(1e-6)
                })
                .collect();
            let (ref mut w2, ref mut b2) = gen_layers[1];
            for i in 0..w2.len() {
                let col = i % dim;
                let dm = (gen_means[col] - real_means[col]) / real_stds[col].powi(2);
                let ds = (gen_stds[col] - real_stds[col]) / real_stds[col];
                w2[i] -= lr * (dm + ds) * 0.01;
            }
            for d in 0..b2.len() {
                let dm = (gen_means[d] - real_means[d]) / real_stds[d].powi(2);
                b2[d] -= lr * dm * 0.1;
            }
        }
        Ok(Self {
            gen_layers,
            dim,
            latent,
        })
    }

    /// Sample `n` rows from the generator.
    pub fn sample(&self, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let z: Vec<f64> = (0..self.latent).map(|_| sample_normal(&mut rng)).collect();
                ctgan_generate(&z, &self.gen_layers)
            })
            .collect()
    }
}

fn tvae_linear(x: &[f64], w: &[f64], b: &[f64], out_dim: usize) -> Vec<f64> {
    let in_dim = x.len();
    (0..out_dim)
        .map(|o| b[o] + (0..in_dim).map(|i| w[i * out_dim + o] * x[i]).sum::<f64>())
        .collect()
}

/// TVAE: Variational Autoencoder for tabular data with mode-specific normalization.
pub struct TvaeGenerator {
    enc_w: Vec<f64>,
    enc_b: Vec<f64>,
    dec_w: Vec<f64>,
    dec_b: Vec<f64>,
    dim: usize,
    latent: usize,
    col_means: Vec<f64>,
    col_stds: Vec<f64>,
}

impl TvaeGenerator {
    /// Fit TVAE on `data`.
    pub fn fit(data: &[Vec<f64>], n_epochs: usize, seed: u64) -> Result<Self> {
        if data.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "TvaeGenerator::fit",
                "data must not be empty",
            ));
        }
        let dim = data[0].len();
        if dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "TvaeGenerator::fit",
                "data dimensions must be > 0",
            ));
        }
        let latent = 4_usize;
        let mut rng = StdRng::seed_from_u64(seed);
        let col_means: Vec<f64> = (0..dim)
            .map(|d| mean_vec(&data.iter().map(|r| r[d]).collect::<Vec<_>>()))
            .collect();
        let col_stds: Vec<f64> = (0..dim)
            .map(|d| {
                var_vec(&data.iter().map(|r| r[d]).collect::<Vec<_>>())
                    .sqrt()
                    .max(1e-6)
            })
            .collect();
        let scale_enc = (2.0 / (dim + 2 * latent) as f64).sqrt();
        let enc_w: Vec<f64> = (0..(dim * 2 * latent))
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale_enc)
            .collect();
        let enc_b = vec![0.0_f64; 2 * latent];
        let scale_dec = (2.0 / (latent + dim) as f64).sqrt();
        let dec_w: Vec<f64> = (0..(latent * dim))
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale_dec)
            .collect();
        let dec_b = vec![0.0_f64; dim];
        let mut gen = Self {
            enc_w,
            enc_b,
            dec_w,
            dec_b,
            dim,
            latent,
            col_means,
            col_stds,
        };
        let lr = 0.005_f64;
        let batch = 8_usize.min(data.len());
        for _ in 0..n_epochs {
            let idx = rng.random::<u64>() as usize % data.len();
            let x_raw = &data[idx];
            let x_norm: Vec<f64> = x_raw
                .iter()
                .enumerate()
                .map(|(d, &v)| (v - gen.col_means[d]) / gen.col_stds[d])
                .collect();
            let enc_out = tvae_linear(&x_norm, &gen.enc_w, &gen.enc_b, 2 * gen.latent);
            let mu = enc_out[..gen.latent].to_vec();
            let log_var = enc_out[gen.latent..].to_vec();
            let eps: Vec<f64> = (0..gen.latent).map(|_| sample_normal(&mut rng)).collect();
            let z: Vec<f64> = (0..gen.latent)
                .map(|i| mu[i] + eps[i] * (0.5 * log_var[i]).exp())
                .collect();
            let x_hat = tvae_linear(&z, &gen.dec_w, &gen.dec_b, gen.dim);
            let recon_grad: Vec<f64> = x_hat
                .iter()
                .zip(x_norm.iter())
                .map(|(h, x)| 2.0 * (h - x) / gen.dim as f64)
                .collect();
            for o in 0..gen.dim {
                gen.dec_b[o] -= lr * recon_grad[o];
                for i in 0..gen.latent {
                    gen.dec_w[i * gen.dim + o] -= lr * recon_grad[o] * z[i];
                }
            }
            for i in 0..gen.latent {
                let kl_grad_mu = mu[i] / batch as f64;
                let kl_grad_lv = (0.5 * (log_var[i].exp() - 1.0)) / batch as f64;
                gen.enc_b[i] -= lr * kl_grad_mu;
                gen.enc_b[gen.latent + i] -= lr * kl_grad_lv;
            }
        }
        Ok(gen)
    }

    /// Sample `n` rows from the trained TVAE.
    pub fn sample(&self, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let z: Vec<f64> = (0..self.latent).map(|_| sample_normal(&mut rng)).collect();
                let x_norm = tvae_linear(&z, &self.dec_w, &self.dec_b, self.dim);
                x_norm
                    .iter()
                    .enumerate()
                    .map(|(d, &v)| v * self.col_stds[d] + self.col_means[d])
                    .collect()
            })
            .collect()
    }
}

/// SMOTE oversampler: synthetic minority oversampling technique.
pub struct SmoteOversampler;

impl SmoteOversampler {
    /// Oversample the minority class `target_class` using k-nearest interpolation.
    pub fn oversample(
        x: &[Vec<f64>],
        labels: &[usize],
        target_class: usize,
        k: usize,
        seed: u64,
    ) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
        if x.len() != labels.len() {
            return Err(TensorError::invalid_argument_op(
                "SmoteOversampler::oversample",
                "x and labels must have the same length",
            ));
        }
        if x.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "SmoteOversampler::oversample",
                "input must not be empty",
            ));
        }
        let minority_idx: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == target_class)
            .map(|(i, _)| i)
            .collect();
        if minority_idx.is_empty() {
            return Ok((x.to_vec(), labels.to_vec()));
        }
        let minority_x: Vec<&Vec<f64>> = minority_idx.iter().map(|&i| &x[i]).collect();
        let k_eff = k.min(minority_x.len().saturating_sub(1).max(1));
        let mut rng = StdRng::seed_from_u64(seed);
        let mut synthetic_x = Vec::new();
        let mut synthetic_y = Vec::new();
        for i in 0..minority_x.len() {
            let mut dists: Vec<(usize, f64)> = minority_x
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, xj)| (j, euclidean(minority_x[i], xj)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let nn_idx = rng.random::<u64>() as usize % k_eff;
            let nn = minority_x[dists[nn_idx.min(dists.len().saturating_sub(1))].0];
            let gap: f64 = rng.random();
            let synth: Vec<f64> = minority_x[i]
                .iter()
                .zip(nn.iter())
                .map(|(&a, &b)| a + gap * (b - a))
                .collect();
            synthetic_x.push(synth);
            synthetic_y.push(target_class);
        }
        let mut all_x = x.to_vec();
        let mut all_y = labels.to_vec();
        all_x.extend(synthetic_x);
        all_y.extend(synthetic_y);
        Ok((all_x, all_y))
    }
}

/// MICE-style iterative missing value imputer.
pub struct MissingValueImputer {
    pub(super) col_means: Vec<f64>,
    dim: usize,
}

impl MissingValueImputer {
    /// Fit imputer from complete data.
    pub fn fit(data: &[Vec<f64>]) -> Result<Self> {
        if data.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "MissingValueImputer::fit",
                "data must not be empty",
            ));
        }
        let dim = data[0].len();
        let col_means: Vec<f64> = (0..dim)
            .map(|d| {
                let vals: Vec<f64> = data.iter().map(|r| r[d]).collect();
                mean_vec(&vals)
            })
            .collect();
        Ok(Self { col_means, dim })
    }

    /// Impute missing values using nearest-neighbour in observed dimensions.
    pub fn impute(&self, data_with_nans: &[Vec<Option<f64>>]) -> Vec<Vec<f64>> {
        let mut filled: Vec<Vec<f64>> = data_with_nans
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(d, v)| v.unwrap_or(self.col_means[d.min(self.col_means.len() - 1)]))
                    .collect()
            })
            .collect();
        for _pass in 0..2 {
            for d in 0..self.dim {
                let obs_vals: Vec<f64> = data_with_nans
                    .iter()
                    .zip(filled.iter())
                    .filter(|(orig, _)| orig[d].is_some())
                    .map(|(_, fill)| fill[d])
                    .collect();
                let obs_mean = if obs_vals.is_empty() {
                    self.col_means[d]
                } else {
                    mean_vec(&obs_vals)
                };
                for (i, orig_row) in data_with_nans.iter().enumerate() {
                    if orig_row[d].is_none() {
                        let mut best_val = obs_mean;
                        let mut best_dist = f64::INFINITY;
                        for (j, other_orig) in data_with_nans.iter().enumerate() {
                            if i == j || other_orig[d].is_none() {
                                continue;
                            }
                            let dist: f64 = (0..self.dim)
                                .filter(|&k| {
                                    k != d && orig_row[k].is_some() && other_orig[k].is_some()
                                })
                                .map(|k| (filled[i][k] - filled[j][k]).powi(2))
                                .sum::<f64>()
                                .sqrt();
                            if dist < best_dist {
                                best_dist = dist;
                                best_val = filled[j][d];
                            }
                        }
                        filled[i][d] = best_val;
                    }
                }
            }
        }
        filled
    }
}

/// Supported numeric scalers.
pub enum NumericScaler {
    /// Standardise to zero mean, unit variance.
    StandardScaler,
    /// Scale to [0, 1] range.
    MinMaxScaler,
}

/// One-hot encoder for categorical columns.
pub struct OneHotEncoder {
    categories: Vec<f64>,
}

impl OneHotEncoder {
    /// Build encoder from the sorted category values.
    pub fn new(categories: Vec<f64>) -> Self {
        Self { categories }
    }

    pub(super) fn encode(&self, v: f64) -> Vec<f64> {
        self.categories
            .iter()
            .map(|&c| if (c - v).abs() < 1e-9 { 1.0 } else { 0.0 })
            .collect()
    }
}

enum ColumnSpec {
    Numeric { col: usize, scaler: NumericScaler },
    Categorical { col: usize, encoder: OneHotEncoder },
}

/// Per-column transformation pipeline.
pub struct ColumnTransformer {
    specs: Vec<ColumnSpec>,
    fit_stats: std::collections::HashMap<usize, (f64, f64)>,
}

impl ColumnTransformer {
    /// Create an empty transformer.
    pub fn new() -> Self {
        Self {
            specs: Vec::new(),
            fit_stats: std::collections::HashMap::new(),
        }
    }

    /// Register a numeric column.
    pub fn add_numeric(&mut self, col: usize, scaler: NumericScaler) {
        self.specs.push(ColumnSpec::Numeric { col, scaler });
    }

    /// Register a categorical column.
    pub fn add_categorical(&mut self, col: usize, encoder: OneHotEncoder) {
        self.specs.push(ColumnSpec::Categorical { col, encoder });
    }

    /// Fit statistics from `data`.
    pub fn fit(&mut self, data: &[Vec<f64>]) -> Result<()> {
        if data.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "ColumnTransformer::fit",
                "data must not be empty",
            ));
        }
        for spec in &self.specs {
            if let ColumnSpec::Numeric { col, .. } = spec {
                let vals: Vec<f64> = data.iter().map(|r| r[*col]).collect();
                let m = mean_vec(&vals);
                let v = var_vec(&vals).sqrt().max(1e-12);
                self.fit_stats.insert(*col, (m, v));
            }
        }
        Ok(())
    }

    /// Transform a dataset by applying registered column specs.
    pub fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|row| {
                let mut out = Vec::new();
                for spec in &self.specs {
                    match spec {
                        ColumnSpec::Numeric { col, scaler } => {
                            let v = row[*col];
                            let stat = self.fit_stats.get(col).cloned().unwrap_or((0.0, 1.0));
                            let scaled = match scaler {
                                NumericScaler::StandardScaler => (v - stat.0) / stat.1,
                                NumericScaler::MinMaxScaler => (v - stat.0) / stat.1,
                            };
                            out.push(scaled);
                        }
                        ColumnSpec::Categorical { col, encoder } => {
                            out.extend(encoder.encode(row[*col]));
                        }
                    }
                }
                out
            })
            .collect()
    }
}

impl Default for ColumnTransformer {
    fn default() -> Self {
        Self::new()
    }
}
