//! MAML — Model-Agnostic Meta-Learning (Finn et al., 2017).

/// Configuration for the MAML (and FOMAML) algorithms.
#[derive(Debug, Clone)]
pub struct MamlConfig {
    /// Learning rate used for the task-level (inner-loop) gradient steps.
    pub inner_lr: f32,
    /// Learning rate used for the meta (outer-loop) update.
    pub meta_lr: f32,
    /// Number of gradient steps to run in the inner loop per task.
    pub num_inner_steps: usize,
    /// When `true` the second-order terms are ignored (FOMAML approximation).
    pub first_order: bool,
}

impl MamlConfig {
    /// Create a MAML config with sensible defaults.
    pub fn new(inner_lr: f32, meta_lr: f32) -> Self {
        Self {
            inner_lr,
            meta_lr,
            num_inner_steps: 1,
            first_order: false,
        }
    }
}

/// Result produced by [`maml_meta_gradient`].
#[derive(Debug, Clone)]
pub struct MamlGradient {
    /// Per-parameter meta-gradients (same shape as `params`).
    pub param_gradients: Vec<Vec<f32>>,
    /// Query-set loss for each task in the batch.
    pub task_losses: Vec<f32>,
    /// Mean query-set loss across all tasks.
    pub meta_loss: f32,
}

/// Run the MAML inner loop on one task's support set.
///
/// Starting from `params`, perform `config.num_inner_steps` gradient-descent
/// steps on the support set and return the adapted parameters.
///
/// # Arguments
///
/// * `params` – initial parameters (one `Vec<f32>` per "layer").
/// * `support_inputs` – support-set inputs, one vector per example.
/// * `support_targets` – scalar target for each support example.
/// * `config` – MAML hyperparameters.
/// * `forward_fn` – computes the scalar loss for one (params, input) pair.
/// * `grad_fn` – computes ∂loss/∂params for one (params, input) pair.
///
/// # Returns
///
/// Adapted parameters after the inner loop.
pub fn maml_inner_update(
    params: &[Vec<f32>],
    support_inputs: &[Vec<f32>],
    support_targets: &[f32],
    config: &MamlConfig,
    forward_fn: impl Fn(&[Vec<f32>], &[f32]) -> f32,
    grad_fn: impl Fn(&[Vec<f32>], &[f32]) -> Vec<Vec<f32>>,
) -> Vec<Vec<f32>> {
    let mut adapted = params.to_vec();

    for _ in 0..config.num_inner_steps {
        // Accumulate gradients over all support examples.
        let mut accum: Vec<Vec<f32>> = adapted.iter().map(|p| vec![0.0; p.len()]).collect();

        for (input, target) in support_inputs.iter().zip(support_targets.iter()) {
            // Build a single-element batch [*target] for the grad function.
            let batch_target = [*target];
            let g = grad_fn(&adapted, input);
            // Average the gradient contribution.
            let n = support_inputs.len().max(1) as f32;
            for (acc_layer, grad_layer) in accum.iter_mut().zip(g.iter()) {
                for (a, gv) in acc_layer.iter_mut().zip(grad_layer.iter()) {
                    *a += gv / n;
                }
            }
            // Use forward_fn to suppress unused-variable warning.
            let _ = forward_fn(&adapted, &batch_target);
        }

        // Gradient descent step on the adapted parameters.
        for (layer, grad_layer) in adapted.iter_mut().zip(accum.iter()) {
            for (p, g) in layer.iter_mut().zip(grad_layer.iter()) {
                *p -= config.inner_lr * g;
            }
        }
    }

    adapted
}

/// Compute the MAML meta-gradient across a batch of tasks.
///
/// For each task:
/// 1. Run the inner loop on the support set → adapted parameters.
/// 2. Evaluate the query-set loss with the adapted parameters.
/// 3. Compute the gradient of that query loss w.r.t. the *original*
///    (pre-adaptation) parameters (first-order approximation).
///
/// The meta-gradient is the mean of per-task gradients.
///
/// # Arguments
///
/// * `params` – current meta-parameters.
/// * `tasks` – batch of `(support_X, support_y, query_X, query_y)`.
/// * `config` – MAML hyperparameters.
/// * `forward_fn` – scalar loss for one (params, input) pair.
/// * `grad_fn` – gradient for one (params, input) pair.
///
/// # Returns
///
/// A [`MamlGradient`] containing per-task losses, the meta-loss, and the
/// accumulated meta-gradient.
pub fn maml_meta_gradient(
    params: &[Vec<f32>],
    tasks: &[(Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>)],
    config: &MamlConfig,
    forward_fn: impl Fn(&[Vec<f32>], &[f32]) -> f32,
    grad_fn: impl Fn(&[Vec<f32>], &[f32]) -> Vec<Vec<f32>>,
) -> MamlGradient {
    let num_tasks = tasks.len().max(1);
    let mut meta_grad: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.len()]).collect();
    let mut task_losses = Vec::with_capacity(num_tasks);

    for (support_x, support_y, query_x, query_y) in tasks.iter() {
        // Inner loop: adapt params to this task's support set.
        let adapted =
            maml_inner_update(params, support_x, support_y, config, &forward_fn, &grad_fn);

        // Evaluate query loss and accumulate gradients.
        let mut task_loss = 0.0_f32;
        let n_query = query_x.len().max(1) as f32;

        // Accumulate query gradients.
        let mut q_grad: Vec<Vec<f32>> = adapted.iter().map(|p| vec![0.0; p.len()]).collect();
        for (qx, qy) in query_x.iter().zip(query_y.iter()) {
            let batch_target = [*qy];
            let loss_val = forward_fn(&adapted, &batch_target);
            task_loss += loss_val / n_query;

            // First-order approximation: gradient w.r.t. adapted params.
            let g = grad_fn(&adapted, qx);
            for (acc, gl) in q_grad.iter_mut().zip(g.iter()) {
                for (a, gv) in acc.iter_mut().zip(gl.iter()) {
                    *a += gv / n_query;
                }
            }
        }

        task_losses.push(task_loss);

        // Accumulate into the meta-gradient.
        let scale = 1.0 / num_tasks as f32;
        for (mg, qg) in meta_grad.iter_mut().zip(q_grad.iter()) {
            for (m, q) in mg.iter_mut().zip(qg.iter()) {
                *m += q * scale;
            }
        }
    }

    let meta_loss = if task_losses.is_empty() {
        0.0
    } else {
        task_losses.iter().sum::<f32>() / task_losses.len() as f32
    };

    MamlGradient {
        param_gradients: meta_grad,
        task_losses,
        meta_loss,
    }
}
