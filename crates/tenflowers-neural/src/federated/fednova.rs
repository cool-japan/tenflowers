//! FedNova algorithm — Wang et al., 2020.

use super::types::{
    check_dimensions, max_layer_delta_norm, weighted_avg_loss, ClientUpdate, FederatedConfig,
    FederatedError, GlobalUpdate, ModelParams,
};

/// Federated Normalized Averaging (FedNova) — Wang et al., 2020.
///
/// Corrects objective inconsistency by normalising each client's update by the
/// number of local steps τ_i before aggregation:
///
/// ```text
/// Δ_i = (w_i^T − w_global) / τ_i
/// global_update = Σ_i (n_i / N) · Δ_i
/// ```
///
/// This prevents clients with more local steps from dominating the aggregation.
///
/// # Reference
/// J. Wang, Q. Liu, H. Liang, G. Joshi, and H. V. Poor, "Tackling the Objective
/// Inconsistency Problem in Heterogeneous Federated Optimization," NeurIPS 2020.
#[derive(Debug, Clone)]
pub struct FedNova {
    /// Shared configuration.
    pub config: FederatedConfig,
    /// Current global model parameters.
    pub global_params: ModelParams,
    /// Current round index.
    pub round: usize,
}

impl FedNova {
    /// Create a new FedNova server.
    pub fn new(config: FederatedConfig, initial_params: ModelParams) -> Self {
        FedNova {
            config,
            global_params: initial_params,
            round: 0,
        }
    }

    /// Return a reference to the current global parameters.
    pub fn distribute(&self) -> &ModelParams {
        &self.global_params
    }

    /// Aggregate updates using normalised averaging.
    ///
    /// Each client's update `Δ_i = w_i − w_global` is normalised by
    /// `τ_i = num_local_steps`.  The global update is:
    /// ```text
    /// w_global += Σ_i (n_i / N) · Δ_i / τ_i  ·  τ_eff
    /// ```
    /// where `τ_eff = Σ_i (n_i / N) · τ_i` is the effective number of steps
    /// that re-scales back to the right learning-rate regime.
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

        let total_samples: f32 = updates.iter().map(|u| u.num_samples as f32).sum();

        // Effective τ: sample-weighted mean of local steps
        let tau_eff: f32 = updates
            .iter()
            .map(|u| (u.num_samples as f32 / total_samples) * u.num_local_steps as f32)
            .sum();

        // Accumulate normalised deltas
        let mut delta: ModelParams = self
            .global_params
            .iter()
            .map(|layer| vec![0.0_f32; layer.len()])
            .collect();

        for update in updates {
            let w_i = update.num_samples as f32 / total_samples;
            let tau_i = update.num_local_steps.max(1) as f32;
            for (l, layer) in update.params.iter().enumerate() {
                for (j, &p) in layer.iter().enumerate() {
                    let delta_ij = (p - self.global_params[l][j]) / tau_i;
                    delta[l][j] += w_i * delta_ij;
                }
            }
        }

        let old_params = self.global_params.clone();

        // Apply: w_global += tau_eff * delta
        for (l, layer) in self.global_params.iter_mut().enumerate() {
            for (j, p) in layer.iter_mut().enumerate() {
                *p += tau_eff * delta[l][j];
            }
        }

        let convergence_metric = max_layer_delta_norm(&old_params, &self.global_params);
        let avg_loss = weighted_avg_loss(updates);

        self.round += 1;

        Ok(GlobalUpdate {
            round: self.round,
            params: self.global_params.clone(),
            participating_clients: updates.len(),
            avg_loss,
            convergence_metric,
        })
    }

    /// Current round (0 before any aggregation).
    pub fn current_round(&self) -> usize {
        self.round
    }

    /// Reference to current global parameters.
    pub fn global_params(&self) -> &ModelParams {
        &self.global_params
    }
}
