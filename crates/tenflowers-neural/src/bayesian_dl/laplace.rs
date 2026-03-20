//! Laplace approximation for Bayesian deep learning.

use super::helpers::sample_normal;
use super::shared::BdlMlp;
use scirs2_core::random::rngs::StdRng;
use std::f64::consts::PI;

/// Laplace approximation: MAP estimate + diagonal Hessian posterior.
pub struct LaplaceApproximation {
    pub map_params: Vec<f64>,
    pub hessian_diag: Vec<f64>,
    pub prior_std: f64,
}

impl LaplaceApproximation {
    /// Fit Laplace approximation via gradient descent MAP + diagonal FD Hessian.
    pub fn fit(
        model: &BdlMlp,
        x_data: &[Vec<f64>],
        y_data: &[Vec<f64>],
        n_epochs: usize,
        lr: f64,
        prior_std: f64,
    ) -> Self {
        let mut m = model.clone();
        let prior_prec = 1.0 / (prior_std * prior_std);
        let eps_fd = 1e-5;

        for _ in 0..n_epochs {
            let params = m.params_flat();
            let loss_grad = m.batch_gradient_fd(x_data, y_data, eps_fd);
            let new_params: Vec<f64> = params
                .iter()
                .zip(loss_grad.iter())
                .map(|(&p, &g)| p - lr * (g + prior_prec * p))
                .collect();
            m.set_params(&new_params);
        }

        let map_params = m.params_flat();
        let n = map_params.len();
        let eps_h = 1e-4;
        let mut hessian_diag = vec![0.0; n];
        let loss_0 = m.batch_loss(x_data, y_data);

        for i in 0..n {
            let mut p_plus = map_params.clone();
            let mut p_minus = map_params.clone();
            p_plus[i] += eps_h;
            p_minus[i] -= eps_h;
            let mut tmp = m.clone();
            tmp.set_params(&p_plus);
            let loss_plus = tmp.batch_loss(x_data, y_data);
            tmp.set_params(&p_minus);
            let loss_minus = tmp.batch_loss(x_data, y_data);
            hessian_diag[i] = (loss_plus - 2.0 * loss_0 + loss_minus) / (eps_h * eps_h);
        }

        Self {
            map_params,
            hessian_diag,
            prior_std,
        }
    }

    /// Posterior variance: `1 / (|h_i| + 1/prior_std²)`.
    pub fn posterior_variance(&self) -> Vec<f64> {
        let prior_prec = 1.0 / (self.prior_std * self.prior_std);
        self.hessian_diag
            .iter()
            .map(|h| 1.0 / (h.abs() + prior_prec).max(1e-12))
            .collect()
    }

    /// Sample θ ~ N(map_params, diag(posterior_variance)).
    pub fn sample(&self, rng: &mut StdRng) -> Vec<f64> {
        let var = self.posterior_variance();
        self.map_params
            .iter()
            .zip(var.iter())
            .map(|(&mu, &v)| mu + v.sqrt() * sample_normal(rng))
            .collect()
    }

    /// Predict via Monte Carlo samples from the posterior.
    pub fn predict(
        &self,
        model: &mut BdlMlp,
        x: &[f64],
        n_samples: usize,
        rng: &mut StdRng,
    ) -> (Vec<f64>, Vec<f64>) {
        let ns = n_samples.max(1);
        let mut preds: Vec<Vec<f64>> = Vec::with_capacity(ns);
        for _ in 0..ns {
            let params = self.sample(rng);
            model.set_params(&params);
            preds.push(model.forward(x));
        }
        model.set_params(&self.map_params);
        let out_dim = preds[0].len();
        let n = preds.len() as f64;
        let mean: Vec<f64> = (0..out_dim)
            .map(|d| preds.iter().map(|p| p[d]).sum::<f64>() / n)
            .collect();
        let variance: Vec<f64> = (0..out_dim)
            .map(|d| {
                let m = mean[d];
                preds.iter().map(|p| (p[d] - m).powi(2)).sum::<f64>() / n
            })
            .collect();
        (mean, variance)
    }

    /// Approximate log marginal likelihood (Laplace log Z).
    pub fn log_marginal_likelihood_approx(&self, n_data: usize) -> f64 {
        let k = self.hessian_diag.len() as f64;
        let log_det_term: f64 = self
            .hessian_diag
            .iter()
            .map(|h| (2.0 * PI / h.abs().max(1e-12)).ln())
            .sum::<f64>();
        let n = n_data.max(1) as f64;
        -n * k.ln() + 0.5 * log_det_term
    }
}
