//! §1 Spectral Graph Convolutions.

use super::utils::{
    degree_vec, dot, identity, mat_add, mat_mul, mat_scale, matvec, normalize_vec, rand_weight,
};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

/// Symmetric normalised Laplacian: L = I - D^{-1/2} A D^{-1/2}.
#[derive(Debug, Clone)]
pub struct GraphLaplacian;

impl GraphLaplacian {
    /// Compute L = I - D^{-1/2} A D^{-1/2} from adjacency matrix.
    pub fn compute(adj: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = adj.len();
        let d = degree_vec(adj);
        let d_inv_sqrt: Vec<f64> = d.iter().map(|di| 1.0 / di.sqrt()).collect();
        let mut lap = identity(n);
        for i in 0..n {
            for j in 0..n {
                lap[i][j] -= d_inv_sqrt[i] * adj[i][j] * d_inv_sqrt[j];
            }
        }
        lap
    }

    /// Rescale Laplacian to [-1, 1] range: L_tilde = (2/lambda_max) * L - I.
    pub fn rescale(lap: &[Vec<f64>], lambda_max: f64) -> Vec<Vec<f64>> {
        let n = lap.len();
        let scale = 2.0 / lambda_max.max(1e-12);
        let mut result = mat_scale(lap, scale);
        for i in 0..n {
            result[i][i] -= 1.0;
        }
        result
    }
}

/// Eigendecomposition-based spectral convolution via power iteration.
#[derive(Debug, Clone)]
pub struct SpectralConv {
    pub filter_weights: Vec<f64>,
    pub seed: u64,
}

impl SpectralConv {
    pub fn new(n_eigs: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let filter_weights = (0..n_eigs).map(|_| rng.random::<f64>()).collect();
        Self {
            filter_weights,
            seed,
        }
    }

    fn power_iteration(mat: &[Vec<f64>], k: usize, iterations: usize, seed: u64) -> Vec<Vec<f64>> {
        let n = mat.len();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut eigvecs: Vec<Vec<f64>> = Vec::with_capacity(k);

        for _ in 0..k {
            let mut v: Vec<f64> = (0..n).map(|_| rng.random::<f64>() - 0.5).collect();
            normalize_vec(&mut v);

            for _ in 0..iterations {
                let mut av = matvec(mat, &v);
                for prev in &eigvecs {
                    let proj = dot(&av, prev);
                    for (x, p) in av.iter_mut().zip(prev) {
                        *x -= proj * p;
                    }
                }
                normalize_vec(&mut av);
                v = av;
            }

            for prev in &eigvecs {
                let proj = dot(&v, prev);
                for (x, p) in v.iter_mut().zip(prev) {
                    *x -= proj * p;
                }
            }
            normalize_vec(&mut v);
            eigvecs.push(v);
        }
        eigvecs
    }

    pub fn forward(&self, x: &[Vec<f64>], adj: &[Vec<f64>], n_eigs: usize) -> Vec<Vec<f64>> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let lap = GraphLaplacian::compute(adj);
        let eigvecs = Self::power_iteration(&lap, n_eigs.min(n), 50, self.seed);

        let f_dim = x.first().map(|r| r.len()).unwrap_or(1);
        let mut out = vec![vec![0.0; f_dim]; n];

        for (idx, ev) in eigvecs.iter().enumerate() {
            let filter_val = self.filter_weights.get(idx).copied().unwrap_or(1.0);
            for f in 0..f_dim {
                let x_hat: f64 = x.iter().zip(ev).map(|(xi, ui)| xi[f] * ui).sum();
                for i in 0..n {
                    out[i][f] += filter_val * x_hat * ev[i];
                }
            }
        }
        out
    }
}

/// Chebyshev polynomial spectral convolution (K-hop approximation).
#[derive(Debug, Clone)]
pub struct ChebyshevConv {
    pub theta: Vec<Vec<f64>>,
    pub in_features: usize,
    pub out_features: usize,
    pub k_order: usize,
}

impl ChebyshevConv {
    pub fn new(in_features: usize, out_features: usize, k_order: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0 / (in_features * k_order) as f64).sqrt();
        let theta = (0..out_features)
            .map(|_| {
                (0..in_features * k_order)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        Self {
            theta,
            in_features,
            out_features,
            k_order,
        }
    }

    pub fn forward(&self, x: &[Vec<f64>], l_tilde: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let k_use = k.min(self.k_order).max(1);
        let f_in = self.in_features;

        let mut cheb_xs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(k_use);
        cheb_xs.push(x.to_vec());

        if k_use > 1 {
            let t1 = mat_mul(l_tilde, x);
            cheb_xs.push(t1);
        }

        for ord in 2..k_use {
            let t_k_minus_2 = &cheb_xs[ord - 2];
            let t_k_minus_1 = &cheb_xs[ord - 1];
            let lt_tkm1 = mat_mul(l_tilde, t_k_minus_1);
            let tk: Vec<Vec<f64>> = lt_tkm1
                .iter()
                .zip(t_k_minus_2)
                .map(|(row_new, row_old)| {
                    row_new
                        .iter()
                        .zip(row_old)
                        .map(|(a, b)| 2.0 * a - b)
                        .collect()
                })
                .collect();
            cheb_xs.push(tk);
        }

        let z: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = Vec::with_capacity(k_use * f_in);
                for t in &cheb_xs {
                    for fval in &t[i] {
                        row.push(*fval);
                    }
                }
                row
            })
            .collect();

        (0..n)
            .map(|i| {
                (0..self.out_features)
                    .map(|o| dot(&z[i], &self.theta[o]))
                    .collect()
            })
            .collect()
    }
}

/// Graph wavelet transform using diffusion wavelet approximation.
#[derive(Debug, Clone)]
pub struct WaveletTransformGraph;

impl WaveletTransformGraph {
    /// Returns [scales x N x N] wavelet coefficient matrices.
    pub fn compute_wavelets(adj: &[Vec<f64>], scales: &[f64]) -> Vec<Vec<Vec<f64>>> {
        let n = adj.len();
        if n == 0 {
            return Vec::new();
        }
        let lap = GraphLaplacian::compute(adj);

        scales
            .iter()
            .map(|&t| {
                let tl = mat_scale(&lap, t);
                let tl2 = mat_mul(&tl, &tl);
                let tl3 = mat_mul(&tl2, &tl);
                let id = identity(n);
                let term2 = mat_scale(&tl2, 0.5);
                let term3 = mat_scale(&tl3, 1.0 / 6.0);
                let mut result = mat_add(&id, &mat_scale(&tl, -1.0));
                result = mat_add(&result, &term2);
                result = mat_add(&result, &mat_scale(&term3, -1.0));
                result
            })
            .collect()
    }
}

/// Graph Fourier Transform: project signal onto graph Laplacian eigenvectors.
#[derive(Debug, Clone)]
pub struct HarmonicAnalysis;

impl HarmonicAnalysis {
    pub fn gft(signal: &[f64], eigenvectors: &[Vec<f64>]) -> Vec<f64> {
        eigenvectors.iter().map(|ev| dot(signal, ev)).collect()
    }

    pub fn inverse_gft(coefficients: &[f64], eigenvectors: &[Vec<f64>]) -> Vec<f64> {
        if eigenvectors.is_empty() {
            return Vec::new();
        }
        let n = eigenvectors[0].len();
        let mut out = vec![0.0; n];
        for (coeff, ev) in coefficients.iter().zip(eigenvectors) {
            for (xi, ui) in out.iter_mut().zip(ev) {
                *xi += coeff * ui;
            }
        }
        out
    }
}

/// Apply polynomial filter H(λ) = Σ_k h_k T_k(L) to a graph signal.
#[derive(Debug, Clone)]
pub struct GraphFilter {
    pub coeffs: Vec<f64>,
}

impl GraphFilter {
    pub fn new(coeffs: Vec<f64>) -> Self {
        Self { coeffs }
    }

    pub fn apply(&self, signal: &[f64], adj: &[Vec<f64>]) -> Vec<f64> {
        let n = signal.len();
        if n == 0 || self.coeffs.is_empty() {
            return signal.to_vec();
        }
        let lap = GraphLaplacian::compute(adj);
        let lambda_max = Self::estimate_lambda_max(&lap);
        let l_tilde = GraphLaplacian::rescale(&lap, lambda_max);

        let mut t0 = signal.to_vec();
        let mut t1 = matvec(&l_tilde, signal);
        let mut out = vec![0.0; n];

        for (i, xi) in out.iter_mut().enumerate() {
            *xi += self.coeffs[0] * t0[i];
        }
        if self.coeffs.len() > 1 {
            for (i, xi) in out.iter_mut().enumerate() {
                *xi += self.coeffs[1] * t1[i];
            }
        }
        for k in 2..self.coeffs.len() {
            let t2 = matvec(&l_tilde, &t1)
                .iter()
                .zip(&t0)
                .map(|(a, b)| 2.0 * a - b)
                .collect::<Vec<_>>();
            for (i, xi) in out.iter_mut().enumerate() {
                *xi += self.coeffs[k] * t2[i];
            }
            t0 = t1;
            t1 = t2;
        }
        out
    }

    fn estimate_lambda_max(lap: &[Vec<f64>]) -> f64 {
        let n = lap.len();
        if n == 0 {
            return 2.0;
        }
        let mut v: Vec<f64> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        normalize_vec(&mut v);
        let mut lambda = 2.0;
        for _ in 0..30 {
            let av = matvec(lap, &v);
            let new_lambda = dot(&v, &av);
            v = av;
            normalize_vec(&mut v);
            lambda = new_lambda;
        }
        lambda.max(1e-6)
    }
}

/// Bandpass filter: pass graph frequencies in [lambda_low, lambda_high].
#[derive(Debug, Clone)]
pub struct BandpassGraphFilter {
    pub lambda_low: f64,
    pub lambda_high: f64,
    pub order: usize,
}

impl BandpassGraphFilter {
    pub fn new(lambda_low: f64, lambda_high: f64, order: usize) -> Self {
        Self {
            lambda_low,
            lambda_high,
            order,
        }
    }

    pub fn apply(&self, signal: &[f64], adj: &[Vec<f64>]) -> Vec<f64> {
        let n = signal.len();
        if n == 0 {
            return Vec::new();
        }
        let order = self.order.max(2);
        let coeffs: Vec<f64> = (0..order)
            .map(|k| {
                let sum: f64 = (0..order)
                    .map(|j| {
                        let node = ((2.0 * j as f64 + 1.0) / (2.0 * order as f64)
                            * std::f64::consts::PI)
                            .cos();
                        let lambda = (node + 1.0) * 1.0;
                        let h = if lambda >= self.lambda_low && lambda <= self.lambda_high {
                            1.0
                        } else {
                            0.0
                        };
                        h * ((k as f64
                            * ((2.0 * j as f64 + 1.0) / (2.0 * order as f64)
                                * std::f64::consts::PI))
                            .cos())
                    })
                    .sum();
                sum * (2.0 / order as f64)
            })
            .collect();
        GraphFilter::new(coeffs).apply(signal, adj)
    }
}

/// Wiener optimal filter: H(λ) = S_xx(λ) / (S_xx(λ) + S_nn(λ)).
#[derive(Debug, Clone)]
pub struct GraphWienerFilter {
    pub s_xx: Vec<f64>,
    pub s_nn: Vec<f64>,
}

impl GraphWienerFilter {
    pub fn new(s_xx: Vec<f64>, s_nn: Vec<f64>) -> Self {
        Self { s_xx, s_nn }
    }

    pub fn apply(&self, noisy_signal: &[f64], eigenvectors: &[Vec<f64>]) -> Vec<f64> {
        let coeffs = HarmonicAnalysis::gft(noisy_signal, eigenvectors);
        let filtered: Vec<f64> = coeffs
            .iter()
            .enumerate()
            .map(|(k, &c)| {
                let sxx = self.s_xx.get(k).copied().unwrap_or(1.0);
                let snn = self.s_nn.get(k).copied().unwrap_or(0.1);
                c * sxx / (sxx + snn + 1e-12)
            })
            .collect();
        HarmonicAnalysis::inverse_gft(&filtered, eigenvectors)
    }
}

/// Heat diffusion process on graph: x_t = exp(-tL) x.
#[derive(Debug, Clone)]
pub struct DiffusionProcess;

impl DiffusionProcess {
    pub fn diffuse(x: &[f64], adj: &[Vec<f64>], t: f64, steps: usize) -> Vec<f64> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let lap = GraphLaplacian::compute(adj);
        let dt = t / steps.max(1) as f64;
        let mut state = x.to_vec();
        for _ in 0..steps {
            let lx = matvec(&lap, &state);
            for (xi, lxi) in state.iter_mut().zip(&lx) {
                *xi -= dt * lxi;
            }
        }
        state
    }

    pub fn step(x: &[f64], adj: &[Vec<f64>], dt: f64) -> Vec<f64> {
        Self::diffuse(x, adj, dt, 1)
    }
}

/// Interpolate signal from sampled nodes via graph Laplacian CG.
#[derive(Debug, Clone)]
pub struct SignalInterpolation {
    pub alpha: f64,
    pub cg_iters: usize,
}

impl SignalInterpolation {
    pub fn new(alpha: f64, cg_iters: usize) -> Self {
        Self { alpha, cg_iters }
    }

    pub fn interpolate(&self, known_values: &[Option<f64>], adj: &[Vec<f64>]) -> Vec<f64> {
        let n = known_values.len();
        if n == 0 {
            return Vec::new();
        }
        let lap = GraphLaplacian::compute(adj);

        let mut b = vec![0.0; n];
        let mut x = vec![0.0; n];
        for (i, val) in known_values.iter().enumerate() {
            if let Some(v) = val {
                b[i] = self.alpha * v;
                x[i] = *v;
            }
        }

        let a_op = |v: &[f64]| -> Vec<f64> {
            let lv = matvec(&lap, v);
            v.iter()
                .zip(&lv)
                .enumerate()
                .map(|(i, (vi, lvi))| {
                    let mask = if known_values[i].is_some() {
                        self.alpha
                    } else {
                        0.0
                    };
                    mask * vi + lvi
                })
                .collect()
        };

        let mut r: Vec<f64> = b.iter().zip(a_op(&x)).map(|(bi, axi)| bi - axi).collect();
        let mut p = r.clone();
        let mut rsold = dot(&r, &r);

        for _ in 0..self.cg_iters {
            let ap = a_op(&p);
            let alpha = rsold / (dot(&p, &ap) + 1e-12);
            for (xi, pi) in x.iter_mut().zip(&p) {
                *xi += alpha * pi;
            }
            for (ri, api) in r.iter_mut().zip(&ap) {
                *ri -= alpha * api;
            }
            let rsnew = dot(&r, &r);
            if rsnew < 1e-20 {
                break;
            }
            let beta = rsnew / (rsold + 1e-12);
            p = r.iter().zip(&p).map(|(ri, pi)| ri + beta * pi).collect();
            rsold = rsnew;
        }
        x
    }
}

// Re-export rand_weight for submodules
pub(super) use super::utils::rand_weight as _rand_weight;
