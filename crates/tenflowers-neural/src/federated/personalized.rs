//! Personalized federated learning algorithms.
//!
//! These algorithms allow each client to maintain a personalized model that
//! trades off between global generalization and local adaptation.
//!
//! | Algorithm | Reference |
//! |-----------|-----------|
//! | [`PFedMeClient`] | Dinh et al., 2020 |
//! | [`ApflClient`] | Cheng et al., 2021 |
//! | [`FedBnClient`] | Li et al., 2021 |
//! | [`HeurFl`] | Fallah et al., 2020 (FedMeta / MAML-style) |

use super::types::{
    max_layer_delta_norm, ClientUpdate, FederatedConfig, FederatedError, GlobalUpdate, ModelParams,
};

// ─────────────────────────────────────────────────────────────────────────────
// pFedMe
// ─────────────────────────────────────────────────────────────────────────────

/// Per-client state for pFedMe.
#[derive(Debug, Clone)]
pub struct PFedMeClientState {
    /// The personalized local model `θ_i`.
    pub personal_params: ModelParams,
    /// Number of personalization steps performed.
    pub personalization_steps: usize,
}

/// pFedMe client — Dinh et al., NeurIPS 2020.
///
/// Each client solves a bi-level problem:
/// ```text
/// min_{w_i} { F_i(θ_i*) + λ/2 · ‖θ_i* − w_i‖² }
/// where θ_i* = argmin_θ { f_i(θ) + λ/2 · ‖θ − w_i‖² }
/// ```
///
/// The inner problem (finding `θ_i*`) is solved by proximal gradient steps.
/// The server aggregates the global models `w_i` via FedAvg.
///
/// # Reference
/// C. T. Dinh, N. H. Tran, T. D. Nguyen,
/// "Personalized Federated Learning with Moreau Envelopes," NeurIPS 2020.
#[derive(Debug, Clone)]
pub struct PFedMeClient {
    /// Shared federated configuration.
    pub config: FederatedConfig,
    /// Current global model `w` received from the server.
    pub global_params: ModelParams,
    /// Proximal regularization strength λ.
    pub lambda: f32,
    /// Learning rate for the inner personalization loop.
    pub inner_lr: f32,
    /// Per-client personalized states.
    pub client_states: Vec<PFedMeClientState>,
    /// Current aggregation round.
    pub round: usize,
}

impl PFedMeClient {
    /// Create a new pFedMe server/coordinator.
    pub fn new(
        config: FederatedConfig,
        initial_params: ModelParams,
        lambda: f32,
        inner_lr: f32,
    ) -> Result<Self, FederatedError> {
        if lambda <= 0.0 {
            return Err(FederatedError::InvalidConfig {
                reason: format!("lambda must be positive; got {lambda}"),
            });
        }
        let num_clients = config.num_clients;
        let client_states = (0..num_clients)
            .map(|_| PFedMeClientState {
                personal_params: initial_params.clone(),
                personalization_steps: 0,
            })
            .collect();
        Ok(PFedMeClient {
            config,
            global_params: initial_params,
            lambda,
            inner_lr,
            client_states,
            round: 0,
        })
    }

    /// Distribute the current global model to a client.
    pub fn distribute(&self, client_id: usize) -> Result<&ModelParams, FederatedError> {
        if client_id >= self.config.num_clients {
            return Err(FederatedError::InvalidClientId {
                id: client_id,
                max: self.config.num_clients,
            });
        }
        Ok(&self.global_params)
    }

    /// Perform inner-loop personalization for a client.
    ///
    /// Solves `K` steps of proximal gradient descent on the Moreau envelope:
    /// ```text
    /// θ ← θ − lr_inner · (∇f_i(θ) + λ · (θ − w))
    /// ```
    ///
    /// Returns the updated personalized parameters.
    pub fn personalize(
        &mut self,
        client_id: usize,
        grad_fn: impl Fn(&ModelParams) -> ModelParams,
        num_inner_steps: usize,
    ) -> Result<ModelParams, FederatedError> {
        if client_id >= self.config.num_clients {
            return Err(FederatedError::InvalidClientId {
                id: client_id,
                max: self.config.num_clients,
            });
        }
        let w = &self.global_params;
        let mut theta = self.client_states[client_id].personal_params.clone();

        for _ in 0..num_inner_steps {
            let grads = grad_fn(&theta);
            for (l, (layer, grad_layer)) in theta.iter_mut().zip(grads.iter()).enumerate() {
                for (j, (p, &g)) in layer.iter_mut().zip(grad_layer.iter()).enumerate() {
                    // proximal gradient: g + λ · (θ_j − w_j)
                    let prox_correction = self.lambda * (*p - w[l][j]);
                    *p -= self.inner_lr * (g + prox_correction);
                }
            }
        }

        self.client_states[client_id].personal_params = theta.clone();
        self.client_states[client_id].personalization_steps += num_inner_steps;
        Ok(theta)
    }

    /// Aggregate global model updates (standard FedAvg on `w_i`).
    ///
    /// Clients should supply `ClientUpdate` with the global-model portion `w_i`
    /// (not the personalized `θ_i`).
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

        let old_params = self.global_params.clone();
        let total: f32 = updates.iter().map(|u| u.num_samples as f32).sum();
        let num_layers = self.global_params.len();
        let mut new_params: ModelParams = self
            .global_params
            .iter()
            .map(|l| vec![0.0_f32; l.len()])
            .collect();

        for update in updates {
            let wt = update.num_samples as f32 / total;
            for l in 0..num_layers {
                for j in 0..new_params[l].len() {
                    new_params[l][j] += wt * update.params[l][j];
                }
            }
        }

        let convergence_metric = max_layer_delta_norm(&old_params, &new_params);
        let avg_loss = updates
            .iter()
            .map(|u| u.loss * u.num_samples as f32 / total)
            .sum();
        self.global_params = new_params;
        self.round += 1;

        Ok(GlobalUpdate {
            round: self.round,
            params: self.global_params.clone(),
            participating_clients: updates.len(),
            avg_loss,
            convergence_metric,
        })
    }

    /// Return a reference to a client's personalized model.
    pub fn personal_params(&self, client_id: usize) -> Option<&ModelParams> {
        self.client_states
            .get(client_id)
            .map(|s| &s.personal_params)
    }

    /// Current communication round.
    pub fn current_round(&self) -> usize {
        self.round
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// APFL
// ─────────────────────────────────────────────────────────────────────────────

/// Per-client APFL state.
#[derive(Debug, Clone)]
pub struct ApflClientState {
    /// Personalized model `v_i = α_i · w_local + (1 − α_i) · w_global`.
    pub personal_params: ModelParams,
    /// Local model `w_local_i` (trained only on local data).
    pub local_params: ModelParams,
    /// Mixing coefficient α_i ∈ [0, 1].
    pub alpha: f32,
    /// Gradient of α from the last update.
    pub alpha_grad: f32,
}

/// APFL — Adaptive Personalized Federated Learning (Cheng et al., 2021).
///
/// Each client maintains both a global model `w` and a local model `w_local`.
/// The personalized model is a convex combination:
/// ```text
/// v_i = α_i · w_local_i + (1 − α_i) · w
/// ```
/// where `α_i` is a per-client scalar learned by gradient descent.
///
/// # Reference
/// Y. Deng, M. M. Kamani, M. Mahdavi,
/// "Adaptive Personalized Federated Learning," arXiv 2020.
#[derive(Debug, Clone)]
pub struct ApflClient {
    /// Shared configuration.
    pub config: FederatedConfig,
    /// Current global model parameters.
    pub global_params: ModelParams,
    /// Per-client adaptive states.
    pub client_states: Vec<ApflClientState>,
    /// Learning rate for α updates.
    pub alpha_lr: f32,
    /// Current round.
    pub round: usize,
}

impl ApflClient {
    /// Create a new APFL coordinator.
    ///
    /// All clients start with `alpha = 0.5` (equal mixture).
    pub fn new(config: FederatedConfig, initial_params: ModelParams, alpha_lr: f32) -> Self {
        let num_clients = config.num_clients;
        let client_states = (0..num_clients)
            .map(|_| ApflClientState {
                personal_params: initial_params.clone(),
                local_params: initial_params.clone(),
                alpha: 0.5,
                alpha_grad: 0.0,
            })
            .collect();
        ApflClient {
            config,
            global_params: initial_params,
            client_states,
            alpha_lr,
            round: 0,
        }
    }

    /// Local update for one client.
    ///
    /// 1. Trains `w_local_i` for `num_steps` steps using `local_grad_fn`.
    /// 2. Recomputes the personalized model `v_i = α · w_local + (1-α) · w`.
    /// 3. Updates `α_i` via gradient descent on the personalized loss.
    pub fn local_update(
        &mut self,
        client_id: usize,
        local_grad_fn: impl Fn(&ModelParams) -> ModelParams,
        personal_grad_fn: impl Fn(&ModelParams) -> ModelParams,
        lr: f32,
        num_steps: usize,
    ) -> Result<ModelParams, FederatedError> {
        if client_id >= self.config.num_clients {
            return Err(FederatedError::InvalidClientId {
                id: client_id,
                max: self.config.num_clients,
            });
        }

        let w_global = self.global_params.clone();
        let state = &mut self.client_states[client_id];

        // Step 1: train local model
        for _ in 0..num_steps {
            let grads = local_grad_fn(&state.local_params);
            for (layer, grad_layer) in state.local_params.iter_mut().zip(grads.iter()) {
                for (p, &g) in layer.iter_mut().zip(grad_layer.iter()) {
                    *p -= lr * g;
                }
            }
        }

        // Step 2: compute personalized model
        let alpha = state.alpha;
        let mut personal = Vec::with_capacity(w_global.len());
        for (l, w_layer) in w_global.iter().enumerate() {
            let p_layer: Vec<f32> = w_layer
                .iter()
                .enumerate()
                .map(|(j, &wg)| alpha * state.local_params[l][j] + (1.0 - alpha) * wg)
                .collect();
            personal.push(p_layer);
        }
        state.personal_params = personal.clone();

        // Step 3: update alpha via gradient on personalized loss
        // ∂L/∂α = <∇L(v), w_local - w_global>
        let pgrads = personal_grad_fn(&personal);
        let mut alpha_grad = 0.0_f32;
        for (l, gl) in pgrads.iter().enumerate() {
            for (j, &pg) in gl.iter().enumerate() {
                if l < state.local_params.len()
                    && j < state.local_params[l].len()
                    && l < w_global.len()
                    && j < w_global[l].len()
                {
                    alpha_grad += pg * (state.local_params[l][j] - w_global[l][j]);
                }
            }
        }
        state.alpha_grad = alpha_grad;
        state.alpha = (state.alpha - self.alpha_lr * alpha_grad).clamp(0.0, 1.0);

        Ok(personal)
    }

    /// Aggregate global model via FedAvg on local models.
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

        let old_params = self.global_params.clone();
        let total: f32 = updates.iter().map(|u| u.num_samples as f32).sum();
        let num_layers = self.global_params.len();
        let mut new_params: ModelParams = self
            .global_params
            .iter()
            .map(|l| vec![0.0_f32; l.len()])
            .collect();

        for update in updates {
            let wt = update.num_samples as f32 / total;
            for l in 0..num_layers {
                for j in 0..new_params[l].len() {
                    new_params[l][j] += wt * update.params[l][j];
                }
            }
        }

        let convergence_metric = max_layer_delta_norm(&old_params, &new_params);
        let avg_loss = updates
            .iter()
            .map(|u| u.loss * u.num_samples as f32 / total)
            .sum();
        self.global_params = new_params;
        self.round += 1;

        Ok(GlobalUpdate {
            round: self.round,
            params: self.global_params.clone(),
            participating_clients: updates.len(),
            avg_loss,
            convergence_metric,
        })
    }

    /// Return the alpha value for a given client.
    pub fn client_alpha(&self, client_id: usize) -> Option<f32> {
        self.client_states.get(client_id).map(|s| s.alpha)
    }

    /// Return the personalized model for a given client.
    pub fn personal_params(&self, client_id: usize) -> Option<&ModelParams> {
        self.client_states
            .get(client_id)
            .map(|s| &s.personal_params)
    }

    /// Current communication round.
    pub fn current_round(&self) -> usize {
        self.round
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FedBN
// ─────────────────────────────────────────────────────────────────────────────

/// FedBN client — Li et al., ICLR 2021.
///
/// Only aggregates non-BatchNorm parameters across clients.  Each client
/// keeps its own local batch normalization statistics (running mean/variance,
/// γ, β), preventing feature shift from heterogeneous data distributions.
///
/// # Parameter convention
/// The model is represented as `ModelParams = Vec<Vec<f32>>`.  Layers are
/// split into two groups identified by index:
/// - **Shared layers** (indices not in `bn_layer_indices`): aggregated globally.
/// - **BN layers** (indices in `bn_layer_indices`): kept local per client.
///
/// # Reference
/// X. Li, M. Jiang, X. Zhang, M. Fang, Q. Dou,
/// "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization,"
/// ICLR 2021.
#[derive(Debug, Clone)]
pub struct FedBnClient {
    /// Shared configuration.
    pub config: FederatedConfig,
    /// Current global (shared-layer) model parameters.
    pub global_params: ModelParams,
    /// Indices of layers treated as batch-normalisation layers (not aggregated).
    pub bn_layer_indices: Vec<usize>,
    /// Per-client BN layer parameters (locally maintained).
    pub client_bn_params: Vec<Vec<Vec<f32>>>,
    /// Current round.
    pub round: usize,
}

impl FedBnClient {
    /// Create a new FedBN coordinator.
    ///
    /// `bn_layer_indices`: layer indices that are NOT aggregated globally.
    pub fn new(
        config: FederatedConfig,
        initial_params: ModelParams,
        bn_layer_indices: Vec<usize>,
    ) -> Self {
        let num_clients = config.num_clients;
        let num_layers = initial_params.len();
        // Initialize per-client BN params from global init
        let client_bn_params: Vec<Vec<Vec<f32>>> = (0..num_clients)
            .map(|_| {
                bn_layer_indices
                    .iter()
                    .map(|&li| {
                        if li < num_layers {
                            initial_params[li].clone()
                        } else {
                            Vec::new()
                        }
                    })
                    .collect()
            })
            .collect();
        FedBnClient {
            config,
            global_params: initial_params,
            bn_layer_indices,
            client_bn_params,
            round: 0,
        }
    }

    /// Extract only the shared (non-BN) parameters from a full parameter set.
    pub fn extract_shared(&self, params: &ModelParams) -> ModelParams {
        params
            .iter()
            .enumerate()
            .filter(|(l, _)| !self.bn_layer_indices.contains(l))
            .map(|(_, layer)| layer.clone())
            .collect()
    }

    /// Aggregate only the shared (non-BN) layers across clients.
    ///
    /// `updates` should carry only the shared layers (use `extract_shared`).
    pub fn aggregate(
        &mut self,
        updates: &[ClientUpdate],
        client_ids: &[usize],
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

        // Determine shared layer indices
        let num_layers = self.global_params.len();
        let shared_indices: Vec<usize> = (0..num_layers)
            .filter(|l| !self.bn_layer_indices.contains(l))
            .collect();

        let total: f32 = updates.iter().map(|u| u.num_samples as f32).sum();

        // Average shared layers
        let mut new_shared: Vec<Vec<f32>> = shared_indices
            .iter()
            .map(|&l| vec![0.0_f32; self.global_params[l].len()])
            .collect();

        for update in updates {
            let wt = update.num_samples as f32 / total;
            for (si, layer) in update.params.iter().enumerate() {
                if si < new_shared.len() {
                    for (j, &p) in layer.iter().enumerate() {
                        if j < new_shared[si].len() {
                            new_shared[si][j] += wt * p;
                        }
                    }
                }
            }
        }

        // Reconstruct full global params (BN layers stay unchanged)
        let old_params = self.global_params.clone();
        let mut new_params = self.global_params.clone();
        for (si, &li) in shared_indices.iter().enumerate() {
            if si < new_shared.len() {
                new_params[li] = new_shared[si].clone();
            }
        }

        // Store updated client BN params (from updates bn portion)
        // Here we just store what was passed in the update's bn-indexed layers
        for (update, &cid) in updates.iter().zip(client_ids.iter()) {
            if cid < self.client_bn_params.len() {
                for (bi, &li) in self.bn_layer_indices.iter().enumerate() {
                    if li < update.params.len() && bi < self.client_bn_params[cid].len() {
                        // Client BN params come from positions after shared
                        // Convention: updates.params has shared layers only
                        let _ = bi; // BN params not in the update — kept local
                    }
                }
            }
        }

        let convergence_metric = max_layer_delta_norm(&old_params, &new_params);
        let avg_loss = updates
            .iter()
            .map(|u| u.loss * u.num_samples as f32 / total)
            .sum();
        self.global_params = new_params;
        self.round += 1;

        Ok(GlobalUpdate {
            round: self.round,
            params: self.global_params.clone(),
            participating_clients: updates.len(),
            avg_loss,
            convergence_metric,
        })
    }

    /// Return the full per-client model (global shared + client BN).
    pub fn client_full_params(&self, client_id: usize) -> Option<ModelParams> {
        if client_id >= self.config.num_clients {
            return None;
        }
        let num_layers = self.global_params.len();
        let mut full = self.global_params.clone();
        let bn_params = &self.client_bn_params[client_id];
        for (bi, &li) in self.bn_layer_indices.iter().enumerate() {
            if li < num_layers && bi < bn_params.len() {
                full[li] = bn_params[bi].clone();
            }
        }
        Some(full)
    }

    /// Current communication round.
    pub fn current_round(&self) -> usize {
        self.round
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HeurFl (FedMeta / MAML-style)
// ─────────────────────────────────────────────────────────────────────────────

/// HeurFl — MAML-style personalized federated learning (Fallah et al., 2020).
///
/// Also known as **Per-FedAvg**.  The server optimizes an initial model `w`
/// such that each client can quickly adapt with one (or few) gradient steps:
/// ```text
/// w* = argmin_w Σ_i F_i(w − α · ∇F_i(w))
/// ```
///
/// The meta-gradient through the adaptation step requires a Hessian-vector
/// product, approximated here by first-order MAML (FOMAML) using finite
/// differences.
///
/// # Reference
/// A. Fallah, A. Mokhtari, A. Ozdaglar,
/// "Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic
/// Meta-Learning Approach," NeurIPS 2020.
#[derive(Debug, Clone)]
pub struct HeurFl {
    /// Shared configuration.
    pub config: FederatedConfig,
    /// Current global meta-model `w`.
    pub global_params: ModelParams,
    /// Inner (adaptation) learning rate α.
    pub inner_lr: f32,
    /// Outer (meta) learning rate β.
    pub outer_lr: f32,
    /// Number of inner gradient steps (default 1 for MAML).
    pub num_inner_steps: usize,
    /// Current round.
    pub round: usize,
}

impl HeurFl {
    /// Create a new HeurFl (Per-FedAvg) coordinator.
    pub fn new(
        config: FederatedConfig,
        initial_params: ModelParams,
        inner_lr: f32,
        outer_lr: f32,
        num_inner_steps: usize,
    ) -> Self {
        HeurFl {
            config,
            global_params: initial_params,
            inner_lr,
            outer_lr,
            num_inner_steps: num_inner_steps.max(1),
            round: 0,
        }
    }

    /// Compute the adapted parameters for a client given its local gradient function.
    ///
    /// Applies `num_inner_steps` gradient steps starting from the global model:
    /// ```text
    /// θ_i = w − α · ∇F_i(w)  [1-step MAML]
    /// ```
    pub fn adapt(&self, grad_fn: impl Fn(&ModelParams) -> ModelParams) -> ModelParams {
        let mut theta = self.global_params.clone();
        for _ in 0..self.num_inner_steps {
            let grads = grad_fn(&theta);
            for (layer, grad_layer) in theta.iter_mut().zip(grads.iter()) {
                for (p, &g) in layer.iter_mut().zip(grad_layer.iter()) {
                    *p -= self.inner_lr * g;
                }
            }
        }
        theta
    }

    /// Aggregate meta-gradients from clients.
    ///
    /// Each `ClientUpdate` should carry the adapted parameters `θ_i` (result of
    /// `adapt()`).  The meta-gradient for client `i` is approximated as:
    /// ```text
    /// meta_grad_i ≈ ∇F_i(θ_i)  [FOMAML approximation]
    /// ```
    /// The global model is updated as:
    /// ```text
    /// w ← w − β · (1/n) Σ_i meta_grad_i
    /// ```
    /// Since `meta_grad_i = (w − θ_i) / (α · K)` in FOMAML, we can recover it
    /// from the difference between the global and adapted parameters.
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

        let n = updates.len() as f32;
        let num_layers = self.global_params.len();

        // Compute meta-gradient: g_meta = (w − θ_i) / (α · K)
        // Then w ← w − β · mean(g_meta)
        // Equivalently: w ← w − β / (α·K) · (w − mean(θ_i))
        // = w · (1 − β/(α·K)) + (β/(α·K)) · mean(θ_i)
        let scale = self.outer_lr / (self.inner_lr * self.num_inner_steps as f32).max(f32::EPSILON);

        // Average adapted params
        let mut theta_avg: ModelParams = self
            .global_params
            .iter()
            .map(|l| vec![0.0_f32; l.len()])
            .collect();
        let total: f32 = updates.iter().map(|u| u.num_samples as f32).sum();
        for update in updates {
            let wt = update.num_samples as f32 / (total * n); // uniform for meta
            for l in 0..num_layers {
                for j in 0..theta_avg[l].len() {
                    theta_avg[l][j] += wt * update.params[l][j] * n;
                }
            }
        }
        // normalize
        for l in 0..num_layers {
            for j in 0..theta_avg[l].len() {
                theta_avg[l][j] /= n;
            }
        }

        let old_params = self.global_params.clone();

        // Update: w ← w − scale · (w − θ_avg)
        for l in 0..num_layers {
            for j in 0..self.global_params[l].len() {
                let diff = self.global_params[l][j] - theta_avg[l][j];
                self.global_params[l][j] -= scale * diff;
            }
        }

        let convergence_metric = max_layer_delta_norm(&old_params, &self.global_params);
        let avg_loss = updates
            .iter()
            .map(|u| u.loss * u.num_samples as f32 / total)
            .sum();
        self.round += 1;

        Ok(GlobalUpdate {
            round: self.round,
            params: self.global_params.clone(),
            participating_clients: updates.len(),
            avg_loss,
            convergence_metric,
        })
    }

    /// Return a reference to the current global (meta) model.
    pub fn distribute(&self) -> &ModelParams {
        &self.global_params
    }

    /// Current communication round.
    pub fn current_round(&self) -> usize {
        self.round
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Personalized Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Metrics for evaluating personalized federated learning.
#[derive(Debug, Clone, Default)]
pub struct PersonalizedMetrics {
    /// Per-client local accuracy after personalization.
    pub per_client_accuracy: Vec<f32>,
    /// Global model accuracy (without personalization).
    pub global_accuracy: f32,
    /// Personalization gap: mean(per_client_accuracy) − global_accuracy.
    pub personalization_gap: f32,
    /// Average number of communication rounds to convergence.
    pub avg_rounds_to_convergence: f32,
    /// Total parameters communicated (in units of f32).
    pub total_params_communicated: usize,
}

impl PersonalizedMetrics {
    /// Create a new zeroed metrics report.
    pub fn new() -> Self {
        PersonalizedMetrics::default()
    }

    /// Compute the personalization gap from accumulated per-client accuracies.
    pub fn compute_gap(&mut self) {
        if self.per_client_accuracy.is_empty() {
            self.personalization_gap = 0.0;
            return;
        }
        let mean_personal: f32 =
            self.per_client_accuracy.iter().sum::<f32>() / self.per_client_accuracy.len() as f32;
        self.personalization_gap = mean_personal - self.global_accuracy;
    }

    /// Record communication cost for one round.
    pub fn record_communication(&mut self, params: &ModelParams) {
        let n: usize = params.iter().map(|l| l.len()).sum();
        self.total_params_communicated += n;
    }
}
