//! FedProx algorithm — Li et al., 2020.

use super::types::{
    check_dimensions, max_layer_delta_norm, weighted_average, weighted_avg_loss, ClientUpdate,
    FederatedConfig, FederatedError, GlobalUpdate, ModelParams,
};

/// Federated Proximal (FedProx) — Li et al., 2020.
///
/// Extends FedAvg by adding a proximal regularisation term to each client's
/// local objective:
///
/// ```text
/// h_i(w; w_global) = F_i(w) + μ/2 · ‖w − w_global‖²
/// ```
///
/// This constrains local updates to stay close to the global model, improving
/// stability in heterogeneous (non-IID) settings.
///
/// # Reference
/// T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Smola, and V. Smith,
/// "Federated Optimization in Heterogeneous Networks," MLSys 2020.
#[derive(Debug, Clone)]
pub struct FedProx {
    /// Shared configuration.
    pub config: FederatedConfig,
    /// Current global model parameters.
    pub global_params: ModelParams,
    /// Proximal regularisation coefficient μ.
    pub mu: f32,
    /// Current round index.
    pub round: usize,
}

impl FedProx {
    /// Create a new FedProx server.
    pub fn new(config: FederatedConfig, initial_params: ModelParams, mu: f32) -> Self {
        FedProx {
            config,
            global_params: initial_params,
            mu,
            round: 0,
        }
    }

    /// Return a reference to the current global parameters.
    pub fn distribute(&self) -> &ModelParams {
        &self.global_params
    }

    /// Aggregate client updates via sample-weighted averaging.
    pub fn aggregate(&mut self, updates: &[ClientUpdate]) -> Result<GlobalUpdate, FederatedError> {
        if updates.is_empty() {
            return Err(FederatedError::NoClientUpdates);
        }
        if updates.len() < self.config.min_clients_available {
            return Err(FederatedError::InsufficientClients {
                required: self.config.min_clients_available,
                available: updates.len(),
            });
        }
        check_dimensions(&self.global_params, updates)?;

        let old_params = self.global_params.clone();
        let new_params = weighted_average(updates);
        let convergence_metric = max_layer_delta_norm(&old_params, &new_params);
        let avg_loss = weighted_avg_loss(updates);

        self.round += 1;
        self.global_params = new_params.clone();

        Ok(GlobalUpdate {
            round: self.round,
            params: new_params,
            participating_clients: updates.len(),
            avg_loss,
            convergence_metric,
        })
    }

    /// Simulate local training with the FedProx proximal term.
    ///
    /// The effective gradient at step `t` is:
    /// ```text
    /// g_eff = ∇F_i(w) + μ · (w − w_global)
    /// ```
    pub fn local_update(
        &self,
        params: &ModelParams,
        global_params: &ModelParams,
        grad_fn: impl Fn(&ModelParams) -> ModelParams,
        lr: f32,
        num_steps: usize,
    ) -> ModelParams {
        let mut w: ModelParams = params.clone();
        for _ in 0..num_steps {
            let grads = grad_fn(&w);
            for (l, (layer, grad_layer)) in w.iter_mut().zip(grads.iter()).enumerate() {
                for (j, (p, &g)) in layer.iter_mut().zip(grad_layer.iter()).enumerate() {
                    // Proximal correction: μ · (w_j - w_global_j)
                    let prox = self.mu * (*p - global_params[l][j]);
                    *p -= lr * (g + prox);
                }
            }
        }
        w
    }

    /// Compute the scalar value of the proximal term:
    /// `μ/2 · ‖w − w_global‖²` summed across all layers.
    pub fn proximal_term(&self, params: &ModelParams, global_params: &ModelParams) -> f32 {
        let sq_norm: f32 = params
            .iter()
            .zip(global_params.iter())
            .flat_map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)))
            .sum();
        0.5 * self.mu * sq_norm
    }
}
