//! FedAvg algorithm — McMahan et al., 2017.

use super::types::{
    check_dimensions, max_layer_delta_norm, weighted_average, weighted_avg_loss, ClientUpdate,
    FederatedConfig, FederatedError, GlobalUpdate, ModelParams,
};

/// Federated Averaging (FedAvg) — McMahan et al., 2017.
///
/// Each round:
/// 1. Server broadcasts `global_params` to a subset of clients.
/// 2. Each client runs `E` local SGD epochs and returns updated parameters.
/// 3. Server aggregates by a weighted average (weighted by `num_samples`).
///
/// # Reference
/// H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas,
/// "Communication-Efficient Learning of Deep Networks from Decentralized Data,"
/// AISTATS 2017.
#[derive(Debug, Clone)]
pub struct FedAvg {
    /// Configuration shared with all sub-algorithms.
    pub config: FederatedConfig,
    /// Current global model parameters.
    pub global_params: ModelParams,
    /// Current round index (0-based internally, 1-based in [`GlobalUpdate`]).
    pub round: usize,
}

impl FedAvg {
    /// Create a new FedAvg server.
    pub fn new(config: FederatedConfig, initial_params: ModelParams) -> Self {
        FedAvg {
            config,
            global_params: initial_params,
            round: 0,
        }
    }

    /// Return a reference to the current global parameters for distribution to
    /// clients.
    pub fn distribute(&self) -> &ModelParams {
        &self.global_params
    }

    /// Aggregate client updates via sample-weighted averaging.
    ///
    /// Validates:
    /// - At least one update is present.
    /// - Enough clients participated.
    /// - All updates have matching layer dimensions.
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

    /// Simulate `num_steps` local SGD steps on a single client.
    ///
    /// `grad_fn` receives the current parameters and returns gradients with the
    /// same shape.  The update rule is plain SGD: `w ← w - lr * g`.
    pub fn local_update(
        &self,
        params: &ModelParams,
        grad_fn: impl Fn(&ModelParams) -> ModelParams,
        lr: f32,
        num_steps: usize,
    ) -> ModelParams {
        let mut w: ModelParams = params.clone();
        for _ in 0..num_steps {
            let grads = grad_fn(&w);
            for (layer, grad_layer) in w.iter_mut().zip(grads.iter()) {
                for (p, &g) in layer.iter_mut().zip(grad_layer.iter()) {
                    *p -= lr * g;
                }
            }
        }
        w
    }

    /// Current communication round (0 before any aggregation).
    pub fn current_round(&self) -> usize {
        self.round
    }

    /// Reference to current global parameters.
    pub fn global_params(&self) -> &ModelParams {
        &self.global_params
    }
}
