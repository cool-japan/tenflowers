//! Reptile meta-learning algorithm (Nichol et al., 2018).

/// Configuration for the Reptile meta-learning algorithm.
#[derive(Debug, Clone)]
pub struct ReptileConfig {
    /// Learning rate used inside each task's SGD loop.
    pub inner_lr: f32,
    /// Step size ε for the meta update: θ ← θ + ε (φ̄ − θ).
    pub meta_lr: f32,
    /// Number of inner gradient steps per task.
    pub num_inner_steps: usize,
    /// Number of tasks sampled per meta-update step.
    pub meta_batch_size: usize,
}

impl ReptileConfig {
    /// Create a Reptile config with sensible defaults.
    pub fn new(inner_lr: f32, meta_lr: f32) -> Self {
        Self {
            inner_lr,
            meta_lr,
            num_inner_steps: 5,
            meta_batch_size: 4,
        }
    }
}

/// Apply the Reptile meta-update in-place.
///
/// Updates `params` toward the mean of the task-adapted parameter sets:
/// ```text
/// θ ← θ + ε · (φ̄ − θ)
/// ```
/// where `φ̄` is the element-wise mean of `task_params`.
///
/// # Arguments
///
/// * `params` – current meta-parameters; updated in place.
/// * `task_params` – per-task adapted parameters returned by the inner loop.
/// * `config` – Reptile hyperparameters (only `meta_lr` is used here).
pub fn reptile_meta_update(
    params: &mut [Vec<f32>],
    task_params: &[Vec<Vec<f32>>],
    config: &ReptileConfig,
) {
    if task_params.is_empty() {
        return;
    }
    let num_tasks = task_params.len() as f32;

    for (layer_idx, layer) in params.iter_mut().enumerate() {
        for (param_idx, p) in layer.iter_mut().enumerate() {
            // Mean of the adapted parameter across tasks.
            let mean_adapted: f32 = task_params
                .iter()
                .filter_map(|tp| tp.get(layer_idx)?.get(param_idx).copied())
                .sum::<f32>()
                / num_tasks;

            *p += config.meta_lr * (mean_adapted - *p);
        }
    }
}
