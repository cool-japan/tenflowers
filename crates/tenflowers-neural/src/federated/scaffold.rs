//! SCAFFOLD algorithm — Karimireddy et al., 2020.

use super::types::{
    max_layer_delta_norm, FederatedConfig, FederatedError, GlobalUpdate, ModelParams,
};

/// Per-client and global state for the SCAFFOLD algorithm.
///
/// `global_control` is the server-side control variate `c`.
/// `client_controls[i]` is the control variate `c_i` of client `i`.
#[derive(Debug, Clone)]
pub struct ScaffoldState {
    /// Global control variate `c` (same shape as model parameters).
    pub global_control: ModelParams,
    /// Per-client control variates `c_i`.
    pub client_controls: Vec<ModelParams>,
}

/// SCAFFOLD — Stochastic Controlled Averaging for Federated Learning.
///
/// Corrects for client drift by maintaining control variates that estimate
/// each client's local gradient bias relative to the global objective.  The
/// local update rule is:
///
/// ```text
/// w ← w − lr · (∇F_i(w) − c_i + c)
/// ```
///
/// where `c_i` is the client's control variate and `c` is the server's.
///
/// After local training, the client control is updated by the Option II rule:
/// ```text
/// c_i^+ = c_i − c + (w_global − w^T) / (T · lr)
/// ```
/// where `T` is the number of local steps.
///
/// # Reference
/// S. P. Karimireddy, S. Kale, M. Mohri, S. Reddi, S. Stich, and A. T. Suresh,
/// "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning," ICML 2020.
#[derive(Debug, Clone)]
pub struct Scaffold {
    /// Shared configuration.
    pub config: FederatedConfig,
    /// Current global model parameters.
    pub global_params: ModelParams,
    /// SCAFFOLD state (control variates).
    pub state: ScaffoldState,
    /// Current round index.
    pub round: usize,
}

impl Scaffold {
    /// Create a new SCAFFOLD server.  All control variates are initialised to
    /// zero vectors matching the shape of `initial_params`.
    pub fn new(config: FederatedConfig, initial_params: ModelParams) -> Self {
        let zero_params: ModelParams = initial_params
            .iter()
            .map(|layer| vec![0.0_f32; layer.len()])
            .collect();

        let client_controls: Vec<ModelParams> = (0..config.num_clients)
            .map(|_| zero_params.clone())
            .collect();

        let state = ScaffoldState {
            global_control: zero_params,
            client_controls,
        };

        Scaffold {
            config,
            global_params: initial_params,
            state,
            round: 0,
        }
    }

    /// Return a reference to the global parameters and the client-specific
    /// control variate for `client_id`.
    ///
    /// Returns `Err` if `client_id` is out of range.
    pub fn distribute(
        &self,
        client_id: usize,
    ) -> Result<(&ModelParams, &ModelParams), FederatedError> {
        let max = self.config.num_clients;
        if client_id >= max {
            return Err(FederatedError::InvalidClientId { id: client_id, max });
        }
        Ok((&self.global_params, &self.state.client_controls[client_id]))
    }

    /// Simulate `num_steps` local SGD steps with control-variate correction.
    ///
    /// The effective gradient is `g_i(w) − c_i + c`.
    ///
    /// Returns `(updated_params, updated_client_control)`.
    /// The client control is updated using Option II of the paper:
    /// ```text
    /// c_i^+ = c_i − c + (w_global − w^T) / (T · lr)
    /// ```
    pub fn local_update_with_control(
        &self,
        params: &ModelParams,
        client_id: usize,
        grad_fn: impl Fn(&ModelParams) -> ModelParams,
        lr: f32,
        num_steps: usize,
    ) -> Result<(ModelParams, ModelParams), FederatedError> {
        let max = self.config.num_clients;
        if client_id >= max {
            return Err(FederatedError::InvalidClientId { id: client_id, max });
        }

        let client_control = &self.state.client_controls[client_id];
        let global_control = &self.state.global_control;
        let w_global = &self.global_params;

        let mut w: ModelParams = params.clone();
        let steps = num_steps.max(1);

        for _ in 0..steps {
            let grads = grad_fn(&w);
            for (l, (layer, grad_layer)) in w.iter_mut().zip(grads.iter()).enumerate() {
                for (j, (p, &g)) in layer.iter_mut().zip(grad_layer.iter()).enumerate() {
                    // Control-variate correction: g - c_i + c
                    let correction = g - client_control[l][j] + global_control[l][j];
                    *p -= lr * correction;
                }
            }
        }

        // Update client control — Option II
        // c_i^+ = c_i − c + (w_global − w^T) / (T · lr)
        let scale = 1.0 / (steps as f32 * lr);
        let new_client_control: ModelParams = client_control
            .iter()
            .zip(global_control.iter())
            .zip(w_global.iter())
            .zip(w.iter())
            .map(|(((ci, c), wg), wt)| {
                ci.iter()
                    .zip(c.iter())
                    .zip(wg.iter())
                    .zip(wt.iter())
                    .map(|(((ci_j, c_j), wg_j), wt_j)| ci_j - c_j + (wg_j - wt_j) * scale)
                    .collect()
            })
            .collect();

        Ok((w, new_client_control))
    }

    /// Aggregate SCAFFOLD updates from clients.
    ///
    /// `updates` is a slice of `(client_id, updated_params, updated_client_control)`.
    ///
    /// The server global parameters are updated by the average of client
    /// parameter changes.  The global control variate is updated as:
    /// ```text
    /// c^+ = c + (|S| / N) · Σ_{i∈S} (c_i^+ − c_i)
    /// ```
    pub fn aggregate(
        &mut self,
        updates: &[(usize, ModelParams, ModelParams)],
    ) -> Result<GlobalUpdate, FederatedError> {
        if updates.is_empty() {
            return Err(FederatedError::NoClientUpdates);
        }
        if updates.len() < self.config.min_clients_available {
            return Err(FederatedError::InsufficientClients {
                required: self.config.min_clients_available,
                available: updates.len(),
            });
        }

        let num_clients = self.config.num_clients as f32;
        let num_selected = updates.len() as f32;

        // Validate dimensions and client ids
        for (client_id, params, _) in updates {
            if *client_id >= self.config.num_clients {
                return Err(FederatedError::InvalidClientId {
                    id: *client_id,
                    max: self.config.num_clients,
                });
            }
            if params.len() != self.global_params.len() {
                return Err(FederatedError::DimensionMismatch {
                    layer: 0,
                    expected: self.global_params.len(),
                    found: params.len(),
                });
            }
            for (l, (ref_layer, client_layer)) in
                self.global_params.iter().zip(params.iter()).enumerate()
            {
                if ref_layer.len() != client_layer.len() {
                    return Err(FederatedError::DimensionMismatch {
                        layer: l,
                        expected: ref_layer.len(),
                        found: client_layer.len(),
                    });
                }
            }
        }

        let old_params = self.global_params.clone();

        // Average of new params
        let avg_new: ModelParams = {
            let mut acc: ModelParams = self
                .global_params
                .iter()
                .map(|layer| vec![0.0_f32; layer.len()])
                .collect();
            let w = 1.0 / num_selected;
            for (_, params, _) in updates {
                for (l, layer) in params.iter().enumerate() {
                    for (j, &p) in layer.iter().enumerate() {
                        acc[l][j] += w * p;
                    }
                }
            }
            acc
        };

        // Update global control: c += (|S|/N) * Σ(c_i^+ - c_i)
        let scale = num_selected / num_clients;
        let old_global_control = self.state.global_control.clone();
        for (client_id, _, new_ci) in updates {
            let old_ci = &self.state.client_controls[*client_id];
            for (l, (gc_layer, (new_ci_layer, old_ci_layer))) in self
                .state
                .global_control
                .iter_mut()
                .zip(new_ci.iter().zip(old_ci.iter()))
                .enumerate()
            {
                let _ = l;
                for (gc_j, (&new_j, &old_j)) in gc_layer
                    .iter_mut()
                    .zip(new_ci_layer.iter().zip(old_ci_layer.iter()))
                {
                    *gc_j += scale * (new_j - old_j);
                }
            }
        }
        let _ = old_global_control;

        // Update client controls in state
        for (client_id, _, new_ci) in updates {
            self.state.client_controls[*client_id] = new_ci.clone();
        }

        let convergence_metric = max_layer_delta_norm(&old_params, &avg_new);

        self.round += 1;
        self.global_params = avg_new.clone();

        Ok(GlobalUpdate {
            round: self.round,
            params: avg_new,
            participating_clients: updates.len(),
            avg_loss: 0.0, // SCAFFOLD aggregation does not receive loss directly
            convergence_metric,
        })
    }

    /// Current round (0 before any aggregation).
    pub fn current_round(&self) -> usize {
        self.round
    }
}
