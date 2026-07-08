//! Model pruning engine: the `ModelPruner` struct and all its methods.
#![allow(unreachable_patterns)] // GPU/ROCM patterns unreachable when features are disabled

use super::types::{PruningConfig, PruningMask, PruningScope, PruningStats, PruningStrategy};
use crate::model::{Model, Sequential};
use scirs2_core::random::RngExt;
use tenflowers_core::{Tensor, TensorError};

/// Model pruning engine.
pub struct ModelPruner {
    pub(super) config: PruningConfig,
}

impl ModelPruner {
    /// Create a new model pruner.
    pub fn new() -> Self {
        Self {
            config: PruningConfig::default(),
        }
    }

    /// Create a new model pruner with custom configuration.
    pub fn with_config(config: PruningConfig) -> Self {
        Self { config }
    }

    /// Prune a sequential model.
    ///
    /// This performs *real* weight pruning: it clones the input model, builds a
    /// binary keep/prune mask for each weight tensor from the actual weight
    /// magnitudes, zeroes the pruned weights in place, and reports statistics
    /// derived from the genuine number of zeroed scalar parameters.
    ///
    /// Only the unstructured strategies that can be expressed as element-wise
    /// masking on the existing layer parameters (`Magnitude`, `Random`) are
    /// supported on a generic [`Sequential`]. Strategies that require rewriting
    /// the network architecture (`Structured`, `Gradual`, `LotteryTicket`)
    /// return an explicit error rather than fabricating results.
    pub fn prune_sequential<T>(
        &self,
        model: &Sequential<T>,
    ) -> Result<(Sequential<T>, PruningStats), TensorError>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + scirs2_core::num_traits::ToPrimitive
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let mut stats = PruningStats::new();
        stats.original_params = self.count_parameters(model);

        // Sequential implements Clone (each layer is cloned via clone_box), so we
        // start from a true copy of the model and mutate its weights in place.
        let mut pruned_model = model.clone();

        // Apply pruning based on strategy. Each routine zeroes weights in
        // `pruned_model` and returns the number of layers it actually touched.
        let layers_pruned = match self.config.strategy {
            PruningStrategy::Magnitude => self.apply_magnitude_pruning(&mut pruned_model)?,
            PruningStrategy::Random => self.apply_random_pruning(&mut pruned_model)?,
            PruningStrategy::Structured => {
                return Err(TensorError::not_implemented_simple(
                    "structured pruning requires rewriting layer dimensions and is not \
                     supported through the generic Sequential interface; use magnitude pruning"
                        .to_string(),
                ));
            }
            PruningStrategy::Gradual => {
                return Err(TensorError::not_implemented_simple(
                    "gradual pruning requires an interleaved fine-tuning loop; apply magnitude \
                     pruning repeatedly between training steps instead"
                        .to_string(),
                ));
            }
            PruningStrategy::LotteryTicket => {
                return Err(TensorError::not_implemented_simple(
                    "lottery-ticket pruning requires storing initial weights and iterative \
                     retraining, which is not available here; use magnitude pruning"
                        .to_string(),
                ));
            }
        };

        stats.layers_pruned = layers_pruned;

        // Update final statistics from the genuinely pruned model. Pruning keeps
        // the tensor shapes intact (weights are zeroed, not removed), so the
        // "remaining" count is the number of non-zero scalar parameters.
        stats.remaining_params = self.count_nonzero_parameters(&pruned_model);
        stats.pruned_params = stats.original_params.saturating_sub(stats.remaining_params);
        stats.achieved_sparsity = if stats.original_params > 0 {
            stats.pruned_params as f32 / stats.original_params as f32
        } else {
            0.0
        };
        stats.memory_reduction = stats.achieved_sparsity;
        stats.flops_reduction = self.estimate_flops_reduction(&stats);
        stats.inference_speedup = self.estimate_inference_speedup(&stats);

        Ok((pruned_model, stats))
    }

    /// Apply magnitude-based pruning to a model in place.
    ///
    /// Builds a keep/prune mask from the real absolute weight values and zeroes
    /// the smallest-magnitude weights. Returns the number of weight tensors that
    /// were modified.
    fn apply_magnitude_pruning<T>(&self, model: &mut Sequential<T>) -> Result<usize, TensorError>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + scirs2_core::num_traits::ToPrimitive
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        match self.config.scope {
            PruningScope::Global => self.apply_global_magnitude_pruning(model),
            PruningScope::LayerWise => self.apply_layerwise_magnitude_pruning(model),
            _ => Err(TensorError::unsupported_operation_simple(
                "Magnitude pruning only supports Global and LayerWise scopes".to_string(),
            )),
        }
    }

    /// Compute the magnitude threshold for a target sparsity from a set of
    /// absolute weight values. Returns the value at the `sparsity` quantile, so
    /// that weights with magnitude strictly below it are pruned.
    fn magnitude_threshold(&self, mut magnitudes: Vec<f32>, sparsity: f32) -> f32 {
        if magnitudes.is_empty() {
            return 0.0;
        }
        magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let clamped = sparsity.clamp(0.0, 1.0);
        let threshold_index = ((magnitudes.len() as f32 * clamped) as usize).min(magnitudes.len());
        if threshold_index >= magnitudes.len() {
            // Pruning everything: use a threshold above the maximum magnitude.
            magnitudes[magnitudes.len() - 1] + 1.0
        } else {
            magnitudes[threshold_index]
        }
    }

    /// Zero every weight whose absolute value is below `threshold`, in place.
    /// Returns the number of scalar weights that were zeroed.
    fn zero_below_threshold<T>(
        &self,
        param: &mut Tensor<T>,
        threshold: f32,
    ) -> Result<usize, TensorError>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::Signed
            + scirs2_core::num_traits::ToPrimitive
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let values = param.to_vec()?;
        let shape = param.shape().dims().to_vec();
        let mut pruned = 0usize;

        let new_values: Vec<T> = values
            .into_iter()
            .map(|value| {
                let magnitude = value.abs().to_f32().unwrap_or(0.0);
                if magnitude < threshold {
                    pruned += 1;
                    T::zero()
                } else {
                    value
                }
            })
            .collect();

        *param = Tensor::from_vec(new_values, &shape)?;
        Ok(pruned)
    }

    /// Apply global magnitude-based pruning across all layers.
    ///
    /// Collects every weight magnitude across all parameter tensors, derives a
    /// single global threshold, and applies it uniformly. Returns the number of
    /// weight tensors that ended up with at least one weight pruned.
    fn apply_global_magnitude_pruning<T>(
        &self,
        model: &mut Sequential<T>,
    ) -> Result<usize, TensorError>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + scirs2_core::num_traits::ToPrimitive
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Step 1: collect all weight magnitudes globally from the real weights.
        let mut all_magnitudes: Vec<f32> = Vec::new();
        for param in model.parameters() {
            for value in param.to_vec()? {
                all_magnitudes.push(value.abs().to_f32().unwrap_or(0.0));
            }
        }

        // Step 2: derive a single global threshold at the target quantile.
        let global_threshold =
            self.magnitude_threshold(all_magnitudes, self.config.target_sparsity);

        // Step 3: apply the threshold uniformly to every weight tensor.
        let mut layers_pruned = 0usize;
        for param in model.parameters_mut() {
            let pruned = self.zero_below_threshold(param, global_threshold)?;
            if pruned > 0 {
                layers_pruned += 1;
            }
        }

        Ok(layers_pruned)
    }

    /// Apply layer-wise magnitude-based pruning.
    ///
    /// Each weight tensor gets an independent threshold so the target sparsity
    /// is reached uniformly per layer. Returns the number of weight tensors that
    /// had at least one weight pruned.
    fn apply_layerwise_magnitude_pruning<T>(
        &self,
        model: &mut Sequential<T>,
    ) -> Result<usize, TensorError>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + scirs2_core::num_traits::ToPrimitive
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let mut layers_pruned = 0usize;

        for param in model.parameters_mut() {
            // Step 1: collect this layer's weight magnitudes.
            let layer_magnitudes: Vec<f32> = param
                .to_vec()?
                .into_iter()
                .map(|value| value.abs().to_f32().unwrap_or(0.0))
                .collect();

            // Step 2: derive a layer-specific threshold.
            let layer_threshold =
                self.magnitude_threshold(layer_magnitudes, self.config.target_sparsity);

            // Step 3: zero this layer's sub-threshold weights.
            let pruned = self.zero_below_threshold(param, layer_threshold)?;
            if pruned > 0 {
                layers_pruned += 1;
            }
        }

        Ok(layers_pruned)
    }

    /// Apply random pruning (baseline method) to a model in place.
    ///
    /// Randomly zeroes a `target_sparsity` fraction of weights in each tensor
    /// using a deterministically seeded RNG from `scirs2_core::random`. Returns
    /// the number of weight tensors that had at least one weight pruned.
    fn apply_random_pruning<T>(&self, model: &mut Sequential<T>) -> Result<usize, TensorError>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

        // Deterministic seed so results are reproducible across runs.
        let mut rng = StdRng::seed_from_u64(0x5072_756E_696E_6701);
        let sparsity = self.config.target_sparsity.clamp(0.0, 1.0);
        let mut layers_pruned = 0usize;

        for param in model.parameters_mut() {
            let values = param.to_vec()?;
            let shape = param.shape().dims().to_vec();
            let mut pruned = 0usize;

            let new_values: Vec<T> = values
                .into_iter()
                .map(|value| {
                    if rng.random::<f32>() < sparsity {
                        pruned += 1;
                        T::zero()
                    } else {
                        value
                    }
                })
                .collect();

            *param = Tensor::from_vec(new_values, &shape)?;
            if pruned > 0 {
                layers_pruned += 1;
            }
        }

        Ok(layers_pruned)
    }

    /// Count total scalar parameters in a model (sum over all weight tensors).
    fn count_parameters<T>(&self, model: &Sequential<T>) -> usize
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        model
            .parameters()
            .iter()
            .map(|param| param.shape().dims().iter().product::<usize>())
            .sum()
    }

    /// Count the number of non-zero scalar parameters in a model.
    fn count_nonzero_parameters<T>(&self, model: &Sequential<T>) -> usize
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        model
            .parameters()
            .iter()
            .map(|param| match param.to_vec() {
                Ok(values) => values.iter().filter(|value| !value.is_zero()).count(),
                // If a parameter cannot be read back, fall back to its full size
                // (treat all as remaining) rather than fabricating a sparsity.
                Err(_) => param.shape().dims().iter().product::<usize>(),
            })
            .sum()
    }

    /// Estimate FLOPS reduction from pruning.
    fn estimate_flops_reduction(&self, stats: &PruningStats) -> f32 {
        // FLOPS reduction is typically proportional to parameter reduction
        // but can be higher for structured pruning
        match self.config.strategy {
            PruningStrategy::Structured => stats.achieved_sparsity * 1.2, // Better FLOPS reduction
            _ => stats.achieved_sparsity * 0.8, // Conservative estimate for unstructured
        }
    }

    /// Estimate inference speedup from pruning.
    fn estimate_inference_speedup(&self, stats: &PruningStats) -> f32 {
        // Speedup depends on sparsity and hardware support for sparse operations
        let base_speedup = match self.config.strategy {
            PruningStrategy::Structured => 1.0 + (stats.achieved_sparsity * 0.8), // Better hardware support
            _ => 1.0 + (stats.achieved_sparsity * 0.4), // Limited sparse support
        };

        // Memory bandwidth can also contribute to speedup
        let memory_factor = 1.0 + (stats.memory_reduction * 0.2);
        base_speedup * memory_factor
    }

    /// Generate pruning masks for layers.
    pub fn generate_masks<T>(&self, model: &Sequential<T>) -> Result<Vec<PruningMask>, TensorError>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + scirs2_core::num_traits::ToPrimitive
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Generate masks based on pruning strategy and scope. Masks are derived
        // from the model's real weight values, not synthetic data.
        match self.config.strategy {
            PruningStrategy::Magnitude => self.generate_magnitude_masks(model),
            PruningStrategy::Random => self.generate_random_masks(model),
            PruningStrategy::Structured => Err(TensorError::not_implemented_simple(
                "structured mask generation requires per-architecture channel/neuron analysis \
                 and is not supported through the generic Sequential interface"
                    .to_string(),
            )),
            // Gradual and LotteryTicket reduce to repeated magnitude masking; the
            // single-shot mask is the magnitude mask at the target sparsity.
            _ => self.generate_magnitude_masks(model),
        }
    }

    /// Build a binary keep/prune mask tensor for a parameter from real weights.
    ///
    /// A mask value of `1.0` keeps the weight and `0.0` prunes it. Weights whose
    /// absolute value is below `threshold` are pruned.
    fn magnitude_mask_for_param<T>(
        &self,
        param: &Tensor<T>,
        threshold: f32,
    ) -> Result<Tensor<f32>, TensorError>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::Signed
            + scirs2_core::num_traits::ToPrimitive
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let shape = param.shape().dims().to_vec();
        let mask_data: Vec<f32> = param
            .to_vec()?
            .into_iter()
            .map(|value| {
                let magnitude = value.abs().to_f32().unwrap_or(0.0);
                if magnitude < threshold {
                    0.0
                } else {
                    1.0
                }
            })
            .collect();

        Tensor::from_vec(mask_data, &shape)
    }

    /// Generate magnitude-based pruning masks from the real model weights.
    fn generate_magnitude_masks<T>(
        &self,
        model: &Sequential<T>,
    ) -> Result<Vec<PruningMask>, TensorError>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + scirs2_core::num_traits::ToPrimitive
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let mut masks = Vec::new();

        match self.config.scope {
            PruningScope::Global => {
                // Global magnitude pruning: a single threshold over all weights.
                let mut all_magnitudes: Vec<f32> = Vec::new();
                for param in model.parameters() {
                    for value in param.to_vec()? {
                        all_magnitudes.push(value.abs().to_f32().unwrap_or(0.0));
                    }
                }
                let global_threshold =
                    self.magnitude_threshold(all_magnitudes, self.config.target_sparsity);

                for (i, param) in model.parameters().iter().enumerate() {
                    let layer_name = format!("layer_{i}");
                    let mask_tensor = self.magnitude_mask_for_param(param, global_threshold)?;
                    let actual_sparsity = Self::mask_sparsity(&mask_tensor);
                    masks.push(PruningMask::new(layer_name, mask_tensor, actual_sparsity));
                }
            }

            PruningScope::LayerWise => {
                // Layer-wise magnitude pruning: an independent threshold per layer.
                for (i, param) in model.parameters().iter().enumerate() {
                    let layer_name = format!("layer_{i}");
                    let layer_magnitudes: Vec<f32> = param
                        .to_vec()?
                        .into_iter()
                        .map(|value| value.abs().to_f32().unwrap_or(0.0))
                        .collect();
                    let layer_threshold =
                        self.magnitude_threshold(layer_magnitudes, self.config.target_sparsity);

                    let mask_tensor = self.magnitude_mask_for_param(param, layer_threshold)?;
                    let actual_sparsity = Self::mask_sparsity(&mask_tensor);
                    masks.push(PruningMask::new(layer_name, mask_tensor, actual_sparsity));
                }
            }

            _ => {
                return Err(TensorError::unsupported_operation_simple(
                    "Magnitude-based pruning only supports Global and LayerWise scopes".to_string(),
                ));
            }
        }

        Ok(masks)
    }

    /// Fraction of pruned (zero) entries in a mask tensor.
    fn mask_sparsity(mask: &Tensor<f32>) -> f32 {
        let total = mask.shape().dims().iter().product::<usize>();
        if total == 0 {
            return 0.0;
        }
        let pruned = match mask.to_vec() {
            Ok(values) => values.iter().filter(|&&value| value == 0.0).count(),
            Err(_) => 0,
        };
        pruned as f32 / total as f32
    }

    /// Generate random pruning masks (for baseline comparison) from real shapes.
    ///
    /// Each mask matches its parameter's shape and zeroes a `target_sparsity`
    /// fraction of entries chosen by a deterministically seeded RNG.
    fn generate_random_masks<T>(
        &self,
        model: &Sequential<T>,
    ) -> Result<Vec<PruningMask>, TensorError>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(0x4D61_736B_5247_4E02);
        let sparsity = self.config.target_sparsity.clamp(0.0, 1.0);
        let mut masks = Vec::new();

        for (i, param) in model.parameters().iter().enumerate() {
            let layer_name = format!("layer_{i}");
            let shape = param.shape().dims().to_vec();
            let total: usize = shape.iter().product();

            let mut pruned = 0usize;
            let mask_data: Vec<f32> = (0..total)
                .map(|_| {
                    if rng.random::<f32>() < sparsity {
                        pruned += 1;
                        0.0
                    } else {
                        1.0
                    }
                })
                .collect();

            let mask_tensor = Tensor::from_vec(mask_data, &shape)?;
            let actual_sparsity = if total > 0 {
                pruned as f32 / total as f32
            } else {
                0.0
            };
            masks.push(PruningMask::new(layer_name, mask_tensor, actual_sparsity));
        }

        Ok(masks)
    }

    /// Compute importance scores for magnitude-based pruning.
    pub fn compute_weight_importance<T>(weights: &Tensor<T>) -> Result<Tensor<T>, TensorError>
    where
        T: Clone
            + Default
            + 'static
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable
            + Send
            + Sync,
    {
        // Compute L1 norm (absolute value) as importance score
        // Higher absolute values are more important and less likely to be pruned
        use tenflowers_core::tensor::TensorStorage;

        match &weights.storage {
            TensorStorage::Cpu(ref arr) => {
                let importance_data: Vec<T> = arr.iter().map(|&w| w.abs()).collect();

                Tensor::from_vec(importance_data, weights.shape().dims())
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => {
                // For GPU tensors, we'd use abs() operation
                // For now, fallback to CPU computation
                let cpu_tensor = weights.to_cpu()?;
                Self::compute_weight_importance(&cpu_tensor)
            }
            #[cfg(not(feature = "gpu"))]
            _ => unreachable!("GPU variant should not exist without gpu feature"),
        }
    }

    /// Compute channel importance scores for structured pruning.
    pub fn compute_channel_importance<T>(weights: &Tensor<T>) -> Result<Vec<f32>, TensorError>
    where
        T: Clone
            + Default
            + 'static
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable
            + Send
            + Sync,
    {
        // Compute L1 norm of filters per channel for structured pruning
        use tenflowers_core::tensor::TensorStorage;

        let shape = weights.shape().dims();
        if shape.len() < 2 {
            return Ok(vec![1.0]); // Single channel case
        }

        let num_channels = shape[1]; // Assuming weights are [out_channels, in_channels, ...]
        let elements_per_channel = shape[2..].iter().product::<usize>() * shape[0];

        match &weights.storage {
            TensorStorage::Cpu(ref arr) => {
                let mut channel_importance = vec![0.0f32; num_channels];

                // For each output channel, compute L1 norm of all weights in that channel
                for out_ch in 0..shape[0] {
                    for in_ch in 0..num_channels {
                        let mut channel_norm = 0.0f32;

                        // Sum absolute values for this channel across all spatial dimensions
                        let base_idx = out_ch * shape[1] * elements_per_channel / shape[0]
                            + in_ch * elements_per_channel / shape[0];
                        for spatial_idx in 0..(elements_per_channel / shape[0]) {
                            let idx = base_idx + spatial_idx;
                            if let Some(&weight) = arr.get(scirs2_core::ndarray::IxDyn(&[idx])) {
                                channel_norm += weight.abs().to_f32().unwrap_or(0.0);
                            }
                        }

                        channel_importance[in_ch] += channel_norm;
                    }
                }

                Ok(channel_importance)
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => {
                // For GPU tensors, fallback to CPU computation
                let cpu_tensor = weights.to_cpu()?;
                Self::compute_channel_importance(&cpu_tensor)
            }
            #[cfg(not(feature = "gpu"))]
            _ => unreachable!("GPU variant should not exist without gpu feature"),
        }
    }

    /// Compute neuron importance scores for dense layer pruning.
    pub fn compute_neuron_importance<T>(weights: &Tensor<T>) -> Result<Vec<f32>, TensorError>
    where
        T: Clone
            + Default
            + 'static
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable
            + Send
            + Sync,
    {
        // Compute L1 norm of weights connected to each neuron
        use tenflowers_core::tensor::TensorStorage;

        let shape = weights.shape().dims();
        if shape.is_empty() {
            return Ok(vec![1.0]);
        }

        let num_neurons = shape[0]; // Output neurons
        let weights_per_neuron = if shape.len() > 1 { shape[1] } else { 1 };

        let neuron_importance = match &weights.storage {
            TensorStorage::Cpu(ref arr) => {
                let mut neuron_importance = vec![0.0f32; num_neurons];

                // For each output neuron, sum absolute values of all connected weights
                for neuron_idx in 0..num_neurons {
                    let mut neuron_norm = 0.0f32;

                    for weight_idx in 0..weights_per_neuron {
                        let linear_idx = neuron_idx * weights_per_neuron + weight_idx;
                        if let Some(&weight) = arr.get(scirs2_core::ndarray::IxDyn(&[linear_idx])) {
                            neuron_norm += weight.abs().to_f32().unwrap_or(0.0);
                        }
                    }

                    neuron_importance[neuron_idx] = neuron_norm;
                }

                neuron_importance
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => {
                // For GPU tensors, fallback to CPU computation
                let cpu_tensor = weights.to_cpu()?;
                return Self::compute_neuron_importance(&cpu_tensor);
            }
            #[cfg(not(feature = "gpu"))]
            _ => unreachable!("GPU variant should not exist without gpu feature"),
        };

        Ok(neuron_importance)
    }
}

impl Default for ModelPruner {
    fn default() -> Self {
        Self::new()
    }
}
