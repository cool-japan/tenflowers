//! Core data types for federated learning.

use tenflowers_core::TensorError;

// ─────────────────────────────────────────────────────────────────────────────
// Public error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during federated learning operations.
#[derive(Debug)]
pub enum FederatedError {
    /// No client updates were supplied to the aggregator.
    NoClientUpdates,
    /// Not enough clients are available to form a round.
    InsufficientClients {
        /// Minimum required clients.
        required: usize,
        /// Clients actually available.
        available: usize,
    },
    /// A client's parameter vector for a layer has the wrong length.
    DimensionMismatch {
        /// Layer index where mismatch occurred.
        layer: usize,
        /// Expected dimension.
        expected: usize,
        /// Dimension found in the client update.
        found: usize,
    },
    /// Configuration values are logically invalid.
    InvalidConfig {
        /// Human-readable description.
        reason: String,
    },
    /// Client id out of range.
    InvalidClientId {
        /// The offending id.
        id: usize,
        /// Maximum valid id (exclusive).
        max: usize,
    },
}

impl std::fmt::Display for FederatedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FederatedError::NoClientUpdates => {
                write!(f, "federated: no client updates provided")
            }
            FederatedError::InsufficientClients {
                required,
                available,
            } => {
                write!(
                    f,
                    "federated: need at least {required} clients, only {available} available"
                )
            }
            FederatedError::DimensionMismatch {
                layer,
                expected,
                found,
            } => {
                write!(
                    f,
                    "federated: layer {layer} dimension mismatch — expected {expected}, found {found}"
                )
            }
            FederatedError::InvalidConfig { reason } => {
                write!(f, "federated: invalid config — {reason}")
            }
            FederatedError::InvalidClientId { id, max } => {
                write!(f, "federated: client id {id} is out of range (max {max})")
            }
        }
    }
}

impl std::error::Error for FederatedError {}

/// Convert a [`FederatedError`] into a [`TensorError`] for compatibility with
/// the rest of the TenfloweRS error hierarchy.
impl From<FederatedError> for TensorError {
    fn from(e: FederatedError) -> Self {
        TensorError::compute_error_simple(e.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core data types
// ─────────────────────────────────────────────────────────────────────────────

/// Model parameters represented as a flat vector per layer.
///
/// `ModelParams[l][i]` is the `i`-th parameter of layer `l`.
pub type ModelParams = Vec<Vec<f32>>;

/// The result of one client's local training step.
#[derive(Debug, Clone)]
pub struct ClientUpdate {
    /// Index of the client that produced this update.
    pub client_id: usize,
    /// Updated model parameters after local training.
    pub params: ModelParams,
    /// Number of local data samples used.
    pub num_samples: usize,
    /// Average loss on the client's local dataset.
    pub loss: f32,
    /// Number of local SGD steps performed.
    pub num_local_steps: usize,
}

/// The result of one global aggregation round.
#[derive(Debug, Clone)]
pub struct GlobalUpdate {
    /// Global round number (1-based).
    pub round: usize,
    /// Aggregated global parameters.
    pub params: ModelParams,
    /// Number of clients that participated.
    pub participating_clients: usize,
    /// Weighted average loss across participating clients.
    pub avg_loss: f32,
    /// Convergence metric: maximum L2 norm of parameter change across layers.
    pub convergence_metric: f32,
}

/// Configuration for a federated learning experiment.
#[derive(Debug, Clone)]
pub struct FederatedConfig {
    /// Total number of clients in the federation.
    pub num_clients: usize,
    /// Number of clients sampled per communication round.
    pub clients_per_round: usize,
    /// Total number of communication rounds.
    pub num_rounds: usize,
    /// Number of local epochs per round.
    pub local_epochs: usize,
    /// Learning rate for local SGD.
    pub local_lr: f32,
    /// Minimum number of clients that must be available to start a round.
    pub min_clients_available: usize,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        FederatedConfig {
            num_clients: 100,
            clients_per_round: 10,
            num_rounds: 100,
            local_epochs: 5,
            local_lr: 0.01,
            min_clients_available: 5,
        }
    }
}

/// Differential-privacy budget specification.
#[derive(Debug, Clone)]
pub struct DpBudget {
    /// Total privacy budget ε (smaller = more private).
    pub epsilon: f32,
    /// Failure probability δ (typically 1e-5).
    pub delta: f32,
    /// L2 clipping threshold for gradients.
    pub max_grad_norm: f32,
    /// Gaussian noise multiplier σ (noise std = σ · max_grad_norm).
    pub noise_multiplier: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers (pub(super) so submodules can use them)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the L2 norm of a flat layer parameter vector.
#[inline]
pub(super) fn layer_l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Verify that all updates have the same layer structure as `reference`.
pub(super) fn check_dimensions(
    reference: &ModelParams,
    updates: &[ClientUpdate],
) -> Result<(), FederatedError> {
    for update in updates {
        if update.params.len() != reference.len() {
            return Err(FederatedError::DimensionMismatch {
                layer: 0,
                expected: reference.len(),
                found: update.params.len(),
            });
        }
        for (l, (ref_layer, client_layer)) in reference.iter().zip(update.params.iter()).enumerate()
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
    Ok(())
}

/// Compute the maximum L2 norm of per-layer parameter differences.
/// Used as the convergence metric in [`GlobalUpdate`].
pub(super) fn max_layer_delta_norm(old: &ModelParams, new_params: &ModelParams) -> f32 {
    old.iter()
        .zip(new_params.iter())
        .map(|(old_layer, new_layer)| {
            let sq_sum: f32 = old_layer
                .iter()
                .zip(new_layer.iter())
                .map(|(a, b)| (b - a) * (b - a))
                .sum();
            sq_sum.sqrt()
        })
        .fold(0.0_f32, f32::max)
}

/// Weighted average aggregation: for each parameter compute
/// `Σ_i weight_i * param_i / Σ_i weight_i`.
pub(super) fn weighted_average(updates: &[ClientUpdate]) -> ModelParams {
    let total_weight: f32 = updates.iter().map(|u| u.num_samples as f32).sum();
    let num_layers = updates[0].params.len();
    let mut result: ModelParams = updates[0]
        .params
        .iter()
        .map(|layer| vec![0.0_f32; layer.len()])
        .collect();

    for update in updates {
        let w = update.num_samples as f32 / total_weight;
        for (l, layer) in update.params.iter().enumerate() {
            for (j, &p) in layer.iter().enumerate() {
                result[l][j] += w * p;
            }
        }
    }
    let _ = num_layers; // used implicitly through result
    result
}

/// Weighted average loss across updates.
pub(super) fn weighted_avg_loss(updates: &[ClientUpdate]) -> f32 {
    let total: f32 = updates.iter().map(|u| u.num_samples as f32).sum();
    if total < f32::EPSILON {
        return 0.0;
    }
    updates
        .iter()
        .map(|u| u.loss * u.num_samples as f32 / total)
        .sum()
}
