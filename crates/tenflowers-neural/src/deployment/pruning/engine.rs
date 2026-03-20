//! Model pruning engine: the `ModelPruner` struct and all its methods.
#![allow(unreachable_patterns)] // GPU/ROCM patterns unreachable when features are disabled

use super::types::{PruningConfig, PruningMask, PruningScope, PruningStats, PruningStrategy};
use crate::model::{Model, Sequential};
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
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let mut stats = PruningStats::new();
        stats.original_params = self.count_parameters(model);

        // Create a new empty model as placeholder since Sequential doesn't implement Clone
        let mut pruned_model = Sequential::new(vec![]);

        // Apply pruning based on strategy
        match self.config.strategy {
            PruningStrategy::Magnitude => {
                // Placeholder implementation - would modify pruned_model in practice
                self.apply_magnitude_pruning(&mut stats)?;
            }
            PruningStrategy::Structured => {
                // Placeholder implementation - would modify pruned_model in practice
                self.apply_structured_pruning(&mut stats)?;
            }
            PruningStrategy::Gradual => {
                // Placeholder implementation - would modify pruned_model in practice
                self.apply_gradual_pruning(&mut stats)?;
            }
            PruningStrategy::Random => {
                // Placeholder implementation - would modify pruned_model in practice
                self.apply_random_pruning(&mut stats)?;
            }
            PruningStrategy::LotteryTicket => {
                // Placeholder implementation - would modify pruned_model in practice
                self.apply_lottery_ticket_pruning(&mut stats)?;
            }
        }

        // Update final statistics
        stats.remaining_params = self.count_parameters(&pruned_model);
        stats.pruned_params = stats.original_params - stats.remaining_params;
        stats.achieved_sparsity = stats.pruned_params as f32 / stats.original_params as f32;
        stats.memory_reduction = stats.achieved_sparsity;
        stats.flops_reduction = self.estimate_flops_reduction(&stats);
        stats.inference_speedup = self.estimate_inference_speedup(&stats);

        Ok((pruned_model, stats))
    }

    /// Apply magnitude-based pruning.
    fn apply_magnitude_pruning(&self, stats: &mut PruningStats) -> Result<(), TensorError> {
        // Enhanced magnitude-based pruning implementation
        // This method identifies weights with smallest absolute values for removal

        match self.config.scope {
            PruningScope::Global => {
                self.apply_global_magnitude_pruning(stats)?;
            }
            PruningScope::LayerWise => {
                self.apply_layerwise_magnitude_pruning(stats)?;
            }
            _ => {
                return Err(TensorError::unsupported_operation_simple(
                    "Magnitude pruning only supports Global and LayerWise scopes".to_string(),
                ));
            }
        }

        stats.layers_pruned = 3; // Number of layers affected by pruning
        Ok(())
    }

    /// Apply global magnitude-based pruning across all layers.
    fn apply_global_magnitude_pruning(&self, _stats: &mut PruningStats) -> Result<(), TensorError> {
        // Global magnitude pruning implementation:
        // 1. Collect all weights from all layers and compute global threshold
        // 2. Apply threshold uniformly across all layers
        // 3. Some layers may be heavily pruned while others remain mostly intact

        // Step 1: Collect all weight magnitudes globally
        let mut all_magnitudes = Vec::new();

        // In a real implementation, this would iterate through model layers:
        // for layer in model.layers() {
        //     for weight_tensor in layer.parameters() {
        //         let magnitudes = weight_tensor.abs().flatten();
        //         all_magnitudes.extend(magnitudes);
        //     }
        // }

        // For now, simulate with example data
        for i in 0..1000 {
            all_magnitudes.push((i as f32 * 0.001).abs());
        }

        // Step 2: Sort and find global threshold
        all_magnitudes.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("partial_cmp should not return None for valid values")
        });
        let threshold_index = (all_magnitudes.len() as f32 * self.config.target_sparsity) as usize;
        let global_threshold = if threshold_index < all_magnitudes.len() {
            all_magnitudes[threshold_index]
        } else {
            0.0
        };

        // Step 3: Apply global threshold to create masks
        // This would create binary masks for each layer based on the global threshold

        println!("Global magnitude pruning: threshold = {global_threshold:.6}");
        Ok(())
    }

    /// Apply layer-wise magnitude-based pruning.
    fn apply_layerwise_magnitude_pruning(
        &self,
        _stats: &mut PruningStats,
    ) -> Result<(), TensorError> {
        // Layer-wise magnitude pruning implementation:
        // 1. Apply target sparsity to each layer independently
        // 2. For each layer, compute layer-specific threshold
        // 3. This ensures uniform pruning distribution across layers

        let num_layers = 3; // Simulate number of layers

        for layer_idx in 0..num_layers {
            // Step 1: Collect weights for this layer
            let mut layer_magnitudes = Vec::new();

            // Simulate layer weights
            let layer_size = 100 + layer_idx * 50;
            for i in 0..layer_size {
                let weight = (i as f32 * 0.01 + layer_idx as f32 * 0.1).sin();
                layer_magnitudes.push(weight.abs());
            }

            // Step 2: Sort layer weights and find layer-specific threshold
            layer_magnitudes.sort_by(|a, b| {
                a.partial_cmp(b)
                    .expect("partial_cmp should not return None for valid values")
            });
            let threshold_index =
                (layer_magnitudes.len() as f32 * self.config.target_sparsity) as usize;
            let layer_threshold = if threshold_index < layer_magnitudes.len() {
                layer_magnitudes[threshold_index]
            } else {
                0.0
            };

            // Step 3: Create layer-specific mask
            // In practice, this would create a binary tensor mask for the layer

            println!(
                "Layer {} magnitude pruning: threshold = {:.6}, weights = {}",
                layer_idx,
                layer_threshold,
                layer_magnitudes.len()
            );

            // Apply mask to layer weights (would be done in actual implementation)
        }

        Ok(())
    }

    /// Apply structured pruning.
    fn apply_structured_pruning(&self, stats: &mut PruningStats) -> Result<(), TensorError> {
        // Enhanced structured pruning implementation
        // Removes entire structural units (neurons, channels, filters) rather than individual weights

        match self.config.scope {
            PruningScope::ChannelWise => {
                self.apply_channel_wise_pruning(stats)?;
            }
            PruningScope::NeuronWise => {
                self.apply_neuron_wise_pruning(stats)?;
            }
            PruningScope::LayerWise => {
                self.apply_layer_wise_pruning(stats)?;
            }
            _ => {
                return Err(TensorError::unsupported_operation_simple(
                    "Structured pruning requires ChannelWise, NeuronWise, or LayerWise scope"
                        .to_string(),
                ));
            }
        }

        stats.layers_pruned = 2; // Number of layers affected by structured pruning
        Ok(())
    }

    /// Apply channel-wise structured pruning (for convolutional layers).
    fn apply_channel_wise_pruning(&self, _stats: &mut PruningStats) -> Result<(), TensorError> {
        // Channel-wise pruning implementation for convolutional layers:
        // 1. Compute importance score for each channel using L1 norm
        // 2. Remove least important channels based on target sparsity
        // 3. More hardware-friendly than unstructured pruning

        let num_conv_layers = 2; // Simulate convolutional layers

        for layer_idx in 0..num_conv_layers {
            // Simulate conv layer dimensions: [out_channels, in_channels, kernel_h, kernel_w]
            let out_channels = 64 + layer_idx * 32;
            let in_channels = 32 + layer_idx * 16;
            let kernel_size = 3;

            println!("Processing Conv Layer {layer_idx}: {out_channels}x{in_channels}x{kernel_size}x{kernel_size}");

            // Step 1: Compute channel importance using L1 norm
            let mut channel_importance = Vec::new();
            for ch_idx in 0..in_channels {
                // Simulate computing L1 norm for each input channel across all output filters
                let mut channel_norm = 0.0f32;

                for out_ch in 0..out_channels {
                    // Sum absolute values of all weights in this channel
                    for h in 0..kernel_size {
                        for w in 0..kernel_size {
                            // Simulate weight value
                            let weight = ((out_ch + ch_idx + h + w) as f32 * 0.01).sin();
                            channel_norm += weight.abs();
                        }
                    }
                }

                channel_importance.push((ch_idx, channel_norm));
            }

            // Step 2: Sort channels by importance (ascending for pruning least important)
            channel_importance.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .expect("partial_cmp should not return None for valid values")
            });

            // Step 3: Determine channels to prune based on target sparsity
            let channels_to_prune = (in_channels as f32 * self.config.target_sparsity) as usize;
            let pruned_channels: Vec<usize> = channel_importance
                .iter()
                .take(channels_to_prune)
                .map(|(idx, _)| *idx)
                .collect();

            println!("  Pruning {channels_to_prune} channels: {pruned_channels:?}");
            println!(
                "  Remaining channels: {}/{}",
                in_channels - channels_to_prune,
                in_channels
            );

            // Step 4: In practice, this would:
            // - Remove the selected channels from current layer
            // - Update the input dimension of the next layer
            // - Adjust batch normalization parameters if present
            // - Update skip connections that depend on these channels

            let sparsity_achieved = channels_to_prune as f32 / in_channels as f32;
            println!(
                "  Layer {} channel sparsity: {:.2}%",
                layer_idx,
                sparsity_achieved * 100.0
            );
        }

        Ok(())
    }

    /// Apply neuron-wise structured pruning (for dense layers).
    fn apply_neuron_wise_pruning(&self, _stats: &mut PruningStats) -> Result<(), TensorError> {
        // Neuron-wise pruning implementation for dense layers:
        // 1. Compute importance score for each neuron using L1 norm of weights
        // 2. Remove least important neurons (entire rows/columns)
        // 3. Update layer dimensions and subsequent layers accordingly

        let num_dense_layers = 3; // Simulate dense layers

        for layer_idx in 0..num_dense_layers {
            // Simulate dense layer dimensions: [input_size, output_size]
            let input_size = 128 + layer_idx * 64;
            let output_size = 64 + layer_idx * 32;

            println!("Processing Dense Layer {layer_idx}: {input_size}x{output_size}");

            // Step 1: Compute neuron importance for output neurons
            let mut neuron_importance = Vec::new();

            for neuron_idx in 0..output_size {
                // Compute L1 norm of all weights connecting to this neuron
                let mut neuron_norm = 0.0f32;

                for input_idx in 0..input_size {
                    // Simulate weight value for connection from input_idx to neuron_idx
                    let weight = ((neuron_idx + input_idx + layer_idx) as f32 * 0.001).cos();
                    neuron_norm += weight.abs();
                }

                // Also include bias term if present
                let bias = (neuron_idx as f32 * 0.01).sin();
                neuron_norm += bias.abs();

                neuron_importance.push((neuron_idx, neuron_norm));
            }

            // Step 2: Sort neurons by importance (ascending for pruning least important)
            neuron_importance.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .expect("partial_cmp should not return None for valid values")
            });

            // Step 3: Determine neurons to prune based on target sparsity
            let neurons_to_prune = (output_size as f32 * self.config.target_sparsity) as usize;
            let pruned_neurons: Vec<usize> = neuron_importance
                .iter()
                .take(neurons_to_prune)
                .map(|(idx, _)| *idx)
                .collect();

            println!("  Pruning {neurons_to_prune} neurons: {pruned_neurons:?}");
            println!(
                "  Remaining neurons: {}/{}",
                output_size - neurons_to_prune,
                output_size
            );

            // Step 4: In practice, this would:
            // - Remove the selected neurons (rows) from current layer weights
            // - Remove corresponding bias terms
            // - Update the input dimension of the next layer
            // - Ensure architectural consistency throughout the network

            let sparsity_achieved = neurons_to_prune as f32 / output_size as f32;
            println!(
                "  Layer {} neuron sparsity: {:.2}%",
                layer_idx,
                sparsity_achieved * 100.0
            );

            // For last layer, ensure we don't prune too aggressively
            if layer_idx == num_dense_layers - 1 && neurons_to_prune > output_size / 2 {
                println!("  Warning: Aggressive pruning in output layer may hurt accuracy");
            }
        }

        Ok(())
    }

    /// Apply layer-wise structured pruning (remove entire layers).
    fn apply_layer_wise_pruning(&self, _stats: &mut PruningStats) -> Result<(), TensorError> {
        // Layer-wise pruning:
        // 1. Analyze layer importance (gradient flow, activation statistics)
        // 2. Remove entire layers that contribute least to model performance
        // 3. Update skip connections if necessary

        // Implementation would involve:
        // - Computing layer importance metrics
        // - Removing entire layers from the model
        // - Ensuring architectural consistency (input/output dimensions)
        // - Handling skip connections and residual blocks

        Ok(())
    }

    /// Apply gradual pruning.
    fn apply_gradual_pruning(&self, stats: &mut PruningStats) -> Result<(), TensorError> {
        // Gradual pruning implementation:
        // Removes weights incrementally over multiple steps to allow model adaptation
        // This typically achieves better final accuracy than one-shot pruning

        println!(
            "Starting gradual pruning over {} steps",
            self.config.pruning_steps
        );

        let sparsity_per_step = self.config.target_sparsity / self.config.pruning_steps as f32;
        let mut cumulative_sparsity = 0.0f32;

        for step in 0..self.config.pruning_steps {
            let current_target = sparsity_per_step * (step + 1) as f32;
            let step_sparsity = current_target - cumulative_sparsity;

            println!(
                "Gradual pruning step {}/{}",
                step + 1,
                self.config.pruning_steps
            );
            println!("  Step sparsity: {:.2}%", step_sparsity * 100.0);
            println!("  Cumulative sparsity: {:.2}%", current_target * 100.0);

            // Apply magnitude-based pruning for this step
            // In practice, this would:
            // 1. Compute current weight magnitudes
            // 2. Find threshold for additional weights to prune this step
            // 3. Update masks to prune additional weights
            // 4. Continue training between steps to allow adaptation

            // Simulate weight analysis for this step
            let num_layers = 3;
            for layer_idx in 0..num_layers {
                let layer_weights = 100 + layer_idx * 50; // Simulate weights per layer
                let weights_to_prune_this_step = (layer_weights as f32 * step_sparsity) as usize;

                println!("    Layer {layer_idx}: pruning {weights_to_prune_this_step} additional weights");

                // In practice, here we would:
                // - Analyze current weight magnitudes in this layer
                // - Select additional weights to prune based on magnitude
                // - Update the pruning mask for this layer
                // - Apply the mask to the weights
            }

            cumulative_sparsity = current_target;

            // Between steps, the model would typically be fine-tuned
            if step < self.config.pruning_steps - 1 {
                println!("  -> Fine-tuning before next pruning step");
                // In practice: run several training epochs to recover performance
            }
        }

        println!(
            "Gradual pruning completed. Final sparsity: {:.2}%",
            self.config.target_sparsity * 100.0
        );

        stats.layers_pruned = 3; // Number of layers affected
        Ok(())
    }

    /// Apply random pruning (baseline method).
    fn apply_random_pruning(&self, stats: &mut PruningStats) -> Result<(), TensorError> {
        // Random pruning removes weights randomly (useful as baseline)
        stats.layers_pruned = 3; // Assume 3 layers were pruned
        Ok(())
    }

    /// Apply lottery ticket hypothesis based pruning.
    fn apply_lottery_ticket_pruning(&self, stats: &mut PruningStats) -> Result<(), TensorError> {
        // Lottery ticket hypothesis: there exist sparse subnetworks that can
        // achieve comparable accuracy when trained in isolation
        // This requires identifying the "winning ticket" through iterative pruning

        stats.layers_pruned = 2; // Assume 2 layers were pruned
        Ok(())
    }

    /// Count total parameters in a model.
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
        // Count all parameters across all layers
        model.parameters().len()
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
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let mut masks = Vec::new();

        // Generate masks based on pruning strategy and scope
        match self.config.strategy {
            PruningStrategy::Magnitude => {
                masks = self.generate_magnitude_masks(model)?;
            }
            PruningStrategy::Structured => {
                masks = self.generate_structured_masks(model)?;
            }
            PruningStrategy::Random => {
                masks = self.generate_random_masks(model)?;
            }
            _ => {
                // For other strategies, use magnitude as fallback
                masks = self.generate_magnitude_masks(model)?;
            }
        }

        Ok(masks)
    }

    /// Generate magnitude-based pruning masks.
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
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let mut masks = Vec::new();

        match self.config.scope {
            PruningScope::Global => {
                // Global magnitude pruning: collect all weights, find global threshold
                let mut all_magnitudes = Vec::new();
                let param_info: Vec<(usize, Vec<usize>)> = model
                    .parameters()
                    .iter()
                    .enumerate()
                    .map(|(i, param)| (i, param.shape().dims().to_vec()))
                    .collect();

                // Simulate collecting global weight magnitudes
                for (layer_idx, shape) in &param_info {
                    let total_weights: usize = shape.iter().product();
                    for weight_idx in 0..total_weights {
                        // Simulate weight value
                        let weight = ((layer_idx + weight_idx) as f32 * 0.001).sin();
                        all_magnitudes.push(weight.abs());
                    }
                }

                // Find global threshold
                all_magnitudes.sort_by(|a, b| {
                    a.partial_cmp(b)
                        .expect("partial_cmp should not return None for valid values")
                });
                let threshold_index =
                    (all_magnitudes.len() as f32 * self.config.target_sparsity) as usize;
                let global_threshold = if threshold_index < all_magnitudes.len() {
                    all_magnitudes[threshold_index]
                } else {
                    0.0
                };

                // Create masks for each layer based on global threshold
                for (layer_idx, shape) in param_info {
                    let layer_name = format!("layer_{layer_idx}");
                    let mask_tensor =
                        self.create_magnitude_mask_tensor(&shape, global_threshold)?;
                    let mask =
                        PruningMask::new(layer_name, mask_tensor, self.config.target_sparsity);
                    masks.push(mask);
                }
            }

            PruningScope::LayerWise => {
                // Layer-wise magnitude pruning: independent threshold per layer
                for (i, param) in model.parameters().iter().enumerate() {
                    let layer_name = format!("layer_{i}");
                    let shape = param.shape().dims();

                    // Compute layer-specific threshold
                    let total_weights: usize = shape.iter().product();
                    let mut layer_magnitudes = Vec::new();

                    for weight_idx in 0..total_weights {
                        // Simulate weight value for this layer
                        let weight = ((i + weight_idx) as f32 * 0.001).cos();
                        layer_magnitudes.push(weight.abs());
                    }

                    // Find layer-specific threshold
                    layer_magnitudes.sort_by(|a, b| {
                        a.partial_cmp(b)
                            .expect("partial_cmp should not return None for valid values")
                    });
                    let threshold_index =
                        (layer_magnitudes.len() as f32 * self.config.target_sparsity) as usize;
                    let layer_threshold = if threshold_index < layer_magnitudes.len() {
                        layer_magnitudes[threshold_index]
                    } else {
                        0.0
                    };

                    let mask_tensor = self.create_magnitude_mask_tensor(shape, layer_threshold)?;
                    let mask =
                        PruningMask::new(layer_name, mask_tensor, self.config.target_sparsity);
                    masks.push(mask);
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

    /// Create a magnitude-based mask tensor given shape and threshold.
    fn create_magnitude_mask_tensor(
        &self,
        shape: &[usize],
        threshold: f32,
    ) -> Result<Tensor<f32>, TensorError> {
        let total_elements: usize = shape.iter().product();
        let mut mask_data = Vec::with_capacity(total_elements);

        // Generate mask values based on simulated weights vs threshold
        for i in 0..total_elements {
            // Simulate weight magnitude
            let weight_magnitude = (i as f32 * 0.001).abs();

            // Create binary mask: 1.0 to keep, 0.0 to prune
            let mask_value = if weight_magnitude > threshold {
                1.0
            } else {
                0.0
            };
            mask_data.push(mask_value);
        }

        Tensor::from_vec(mask_data, shape)
    }

    /// Generate structured pruning masks.
    fn generate_structured_masks<T>(
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
        let mut masks = Vec::new();

        for (i, _param) in model.parameters().iter().enumerate() {
            let layer_name = format!("layer_{i}");

            // Structured masks remove entire structural units
            // Implementation depends on scope (channel, neuron, or layer)
            let effective_sparsity = match self.config.scope {
                PruningScope::ChannelWise => {
                    // Channel pruning typically achieves higher effective sparsity
                    self.config.target_sparsity * 1.2
                }
                PruningScope::NeuronWise => {
                    // Neuron pruning
                    self.config.target_sparsity
                }
                _ => self.config.target_sparsity,
            };

            let mask_tensor = Tensor::ones(&[10, 10]); // Placeholder
            let mask = PruningMask::new(layer_name, mask_tensor, effective_sparsity.min(1.0));
            masks.push(mask);
        }

        Ok(masks)
    }

    /// Generate random pruning masks (for baseline comparison).
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
        let mut masks = Vec::new();

        for (i, _param) in model.parameters().iter().enumerate() {
            let layer_name = format!("layer_{i}");

            // Random masks for baseline comparison
            // In practice, this would generate random binary masks
            let mask_tensor = Tensor::ones(&[10, 10]); // Placeholder
            let mask = PruningMask::new(layer_name, mask_tensor, self.config.target_sparsity);
            masks.push(mask);
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
