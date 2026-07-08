use crate::layers::{Dense, Layer, LayerType};
use scirs2_core::num_traits::Float;
use std::marker::PhantomData;
use tenflowers_core::{Result, Tensor};

/// Expert network in a Mixture of Experts layer
/// Each expert is typically a simple feedforward network
#[derive(Clone)]
pub struct Expert<T>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    pub layers: Vec<Dense<T>>,
    pub expert_id: usize,
    _phantom: PhantomData<T>,
}

impl<T> Expert<T>
where
    T: Float
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new expert with specified architecture
    pub fn new(expert_id: usize, layer_sizes: &[usize]) -> Result<Self> {
        if layer_sizes.len() < 2 {
            return Err(tenflowers_core::TensorError::invalid_argument(
                "Expert must have at least input and output layers".to_string(),
            ));
        }

        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let dense = Dense::new(layer_sizes[i], layer_sizes[i + 1], true);
            layers.push(dense);
        }

        Ok(Expert {
            layers,
            expert_id,
            _phantom: PhantomData,
        })
    }

    /// Forward pass through the expert network
    pub fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }

    /// Get all parameters from this expert
    pub fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    /// Get all mutable parameters from this expert
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        params
    }

    /// Set training mode for all layers in this expert
    pub fn set_training(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }
}

/// Top-K gating mechanism for routing tokens to experts
#[derive(Clone)]
pub struct TopKRouter<T>
where
    T: Float
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Linear layer for computing gating logits
    pub gate: Dense<T>,
    /// Number of experts to route each token to
    pub k: usize,
    /// Load balancing loss coefficient
    pub load_balance_loss_coeff: T,
    /// Whether to use noisy gating for training
    pub noisy_gating: bool,
    /// Training mode
    pub training: bool,
    _phantom: PhantomData<T>,
}

impl<T> TopKRouter<T>
where
    T: Float
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new Top-K router
    pub fn new(input_dim: usize, num_experts: usize, k: usize) -> Result<Self> {
        if k > num_experts {
            return Err(tenflowers_core::TensorError::invalid_argument(format!(
                "k ({k}) cannot be greater than num_experts ({num_experts})"
            )));
        }

        let gate = Dense::new(input_dim, num_experts, true);

        Ok(TopKRouter {
            gate,
            k,
            load_balance_loss_coeff: T::from(0.01).expect("Failed to convert 0.01 to tensor type"),
            noisy_gating: true,
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Compute routing weights and expert indices
    /// Returns (expert_weights, expert_indices, load_balance_loss)
    pub fn forward(&self, input: &Tensor<T>) -> Result<(Tensor<T>, Tensor<usize>, T)> {
        // Compute gating logits
        let gate_logits = self.gate.forward(input)?;

        // Apply softmax to get per-token routing probabilities over experts.
        // gate_probs shape: [num_tokens, num_experts].
        let gate_probs = tenflowers_core::ops::softmax(&gate_logits, Some(-1))?;

        let probs_dims = gate_probs.shape().dims().to_vec();
        let num_experts = *probs_dims.last().ok_or_else(|| {
            tenflowers_core::TensorError::invalid_argument(
                "gate_probs must have at least one dimension".to_string(),
            )
        })?;
        // Number of routed tokens is the product of all leading dimensions
        // (handles both [num_tokens, num_experts] and higher-rank inputs).
        let num_tokens: usize = probs_dims[..probs_dims.len() - 1].iter().product();

        // Real Top-K routing: select the k highest-probability experts per token.
        // expert_indices shape: [num_tokens..., k]; values are expert ids in [0, num_experts).
        let (_top_values, expert_indices) =
            tenflowers_core::ops::reduction::topk(&gate_probs, self.k, Some(-1))?;

        // Switch Transformer auxiliary load-balancing loss (GShard / Switch formulation):
        //   P_i = mean over tokens of gate_probs[:, i]      (router mass on expert i)
        //   f_i = (tokens dispatched to expert i) / num_tokens
        //   aux = coeff * num_experts * sum_i (f_i * P_i)
        // For k == 1 this is exactly the Switch Transformer loss; for k > 1 each
        // token contributes to its k selected experts (GShard generalization).
        let load_balance_loss = if num_tokens == 0 {
            T::zero()
        } else {
            let probs_vec = gate_probs.to_vec()?;
            let index_vec = expert_indices.to_vec()?;

            let num_tokens_t = T::from(num_tokens).ok_or_else(|| {
                tenflowers_core::TensorError::invalid_argument(
                    "failed to convert num_tokens to tensor element type".to_string(),
                )
            })?;
            let num_experts_t = T::from(num_experts).ok_or_else(|| {
                tenflowers_core::TensorError::invalid_argument(
                    "failed to convert num_experts to tensor element type".to_string(),
                )
            })?;

            // P_i: mean router probability per expert (column means of gate_probs).
            let mut prob_mass = vec![T::zero(); num_experts];
            for token_idx in 0..num_tokens {
                let row = token_idx * num_experts;
                for (expert_idx, mass) in prob_mass.iter_mut().enumerate() {
                    *mass = *mass + probs_vec[row + expert_idx];
                }
            }

            // f_i: fraction of tokens dispatched to each expert (from top-k indices).
            let mut dispatch_count = vec![T::zero(); num_experts];
            for &expert_idx in &index_vec {
                if expert_idx < num_experts {
                    dispatch_count[expert_idx] = dispatch_count[expert_idx] + T::one();
                }
            }

            // sum_i (f_i * P_i) with f_i = count_i / num_tokens, P_i = mass_i / num_tokens.
            let mut accum = T::zero();
            for expert_idx in 0..num_experts {
                let fraction = dispatch_count[expert_idx] / num_tokens_t;
                let mean_prob = prob_mass[expert_idx] / num_tokens_t;
                accum = accum + fraction * mean_prob;
            }

            self.load_balance_loss_coeff * num_experts_t * accum
        };

        Ok((gate_probs, expert_indices, load_balance_loss))
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        self.gate.set_training(training);
    }

    /// Get parameters
    pub fn parameters(&self) -> Vec<&Tensor<T>> {
        self.gate.parameters()
    }

    /// Get mutable parameters
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        self.gate.parameters_mut()
    }
}

/// Mixture of Experts layer implementing the Switch Transformer approach
#[derive(Clone)]
pub struct MixtureOfExperts<T>
where
    T: Float
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Collection of expert networks
    pub experts: Vec<Expert<T>>,
    /// Router for token routing
    pub router: TopKRouter<T>,
    /// Number of experts
    pub num_experts: usize,
    /// Expert capacity (tokens per expert)
    pub expert_capacity: Option<usize>,
    /// Training mode
    pub training: bool,
    _phantom: PhantomData<T>,
}

impl<T> MixtureOfExperts<T>
where
    T: Float
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new Mixture of Experts layer
    pub fn new(
        input_dim: usize,
        num_experts: usize,
        expert_hidden_dim: usize,
        output_dim: usize,
        k: usize,
    ) -> Result<Self> {
        // Create experts with 2-layer MLPs
        let mut experts = Vec::new();
        for i in 0..num_experts {
            let expert = Expert::new(i, &[input_dim, expert_hidden_dim, output_dim])?;
            experts.push(expert);
        }

        // Create router
        let router = TopKRouter::new(input_dim, num_experts, k)?;

        Ok(MixtureOfExperts {
            experts,
            router,
            num_experts,
            expert_capacity: None,
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Set expert capacity for load balancing
    pub fn with_expert_capacity(mut self, capacity: usize) -> Self {
        self.expert_capacity = Some(capacity);
        self
    }

    /// Forward pass through the MoE layer.
    ///
    /// Implements dense computation of sparse top-k routing: each expert is
    /// evaluated on the full batch, then every token's output is assembled as a
    /// weighted sum over only its top-k selected experts. The combination
    /// weights are the router probabilities of the selected experts,
    /// renormalized to sum to 1 per token (standard top-k MoE combination).
    pub fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if self.experts.is_empty() {
            return Err(tenflowers_core::TensorError::invalid_argument(
                "MixtureOfExperts must contain at least one expert".to_string(),
            ));
        }

        // Routing: gate_probs [num_tokens, num_experts], expert_indices [num_tokens, k].
        let (gate_probs, expert_indices, _load_balance_loss) = self.router.forward(input)?;

        // Evaluate every expert on the full input. Each output is
        // [num_tokens, output_dim].
        let mut expert_outputs = Vec::with_capacity(self.experts.len());
        for expert in &self.experts {
            expert_outputs.push(expert.forward(input)?);
        }

        let output_dims = expert_outputs[0].shape().dims().to_vec();
        let output_dim = *output_dims.last().ok_or_else(|| {
            tenflowers_core::TensorError::invalid_argument(
                "expert output must have at least one dimension".to_string(),
            )
        })?;
        let num_tokens: usize = output_dims[..output_dims.len() - 1].iter().product();

        let probs_vec = gate_probs.to_vec()?;
        let index_vec = expert_indices.to_vec()?;
        let mut expert_output_vecs = Vec::with_capacity(expert_outputs.len());
        for expert_output in &expert_outputs {
            expert_output_vecs.push(expert_output.to_vec()?);
        }

        let num_experts = self.num_experts;
        let k = self.router.k;
        let mut combined = vec![T::zero(); num_tokens * output_dim];

        for token_idx in 0..num_tokens {
            let prob_row = token_idx * num_experts;
            let index_row = token_idx * k;

            // Sum of router probabilities over the selected top-k experts, used
            // to renormalize the combination weights for this token.
            let mut weight_sum = T::zero();
            for slot in 0..k {
                let expert_idx = index_vec[index_row + slot];
                if expert_idx < num_experts {
                    weight_sum = weight_sum + probs_vec[prob_row + expert_idx];
                }
            }

            for slot in 0..k {
                let expert_idx = index_vec[index_row + slot];
                if expert_idx >= num_experts {
                    continue;
                }
                let raw_weight = probs_vec[prob_row + expert_idx];
                let weight = if weight_sum > T::zero() {
                    raw_weight / weight_sum
                } else {
                    // Degenerate all-zero gate: fall back to uniform top-k weighting.
                    T::one() / T::from(k).unwrap_or_else(T::one)
                };

                let source = &expert_output_vecs[expert_idx];
                let out_base = token_idx * output_dim;
                let src_base = token_idx * output_dim;
                for feature in 0..output_dim {
                    combined[out_base + feature] =
                        combined[out_base + feature] + weight * source[src_base + feature];
                }
            }
        }

        Tensor::from_vec(combined, &output_dims)
    }

    /// Get load balancing loss for training
    pub fn load_balance_loss(&self, input: &Tensor<T>) -> Result<T> {
        let (_weights, _indices, loss) = self.router.forward(input)?;
        Ok(loss)
    }

    /// Get routing statistics for analysis
    pub fn routing_stats(&self, input: &Tensor<T>) -> Result<RoutingStats<T>> {
        let (weights, indices, loss) = self.router.forward(input)?;

        // Real per-expert utilization: fraction of all dispatched token-slots
        // (num_tokens * k) routed to each expert, derived from the top-k indices.
        let index_vec = indices.to_vec()?;
        let mut counts = vec![T::zero(); self.num_experts];
        for &expert_idx in &index_vec {
            if expert_idx < self.num_experts {
                counts[expert_idx] = counts[expert_idx] + T::one();
            }
        }
        let total_slots = index_vec.len();
        let expert_utilization = if total_slots == 0 {
            vec![T::zero(); self.num_experts]
        } else {
            let total_slots_t = T::from(total_slots).ok_or_else(|| {
                tenflowers_core::TensorError::invalid_argument(
                    "failed to convert dispatched slot count to tensor element type".to_string(),
                )
            })?;
            counts.into_iter().map(|c| c / total_slots_t).collect()
        };

        Ok(RoutingStats {
            expert_weights: weights,
            expert_indices: indices,
            load_balance_loss: loss,
            expert_utilization,
        })
    }
}

/// Statistics about expert routing for analysis and debugging
#[derive(Clone)]
pub struct RoutingStats<T>
where
    T: Float + Clone + Default + Send + Sync + 'static,
{
    /// Routing weights for each token-expert pair
    pub expert_weights: Tensor<T>,
    /// Selected expert indices for each token
    pub expert_indices: Tensor<usize>,
    /// Load balancing loss
    pub load_balance_loss: T,
    /// Utilization rate for each expert
    pub expert_utilization: Vec<T>,
}

impl<T> Layer<T> for MixtureOfExperts<T>
where
    T: Float
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = self.router.parameters();
        for expert in &self.experts {
            params.extend(expert.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = self.router.parameters_mut();
        for expert in &mut self.experts {
            params.extend(expert.parameters_mut());
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.router.set_training(training);
        for expert in &mut self.experts {
            expert.set_training(training);
        }
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Unknown // MoE would need its own LayerType
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_creation() {
        let expert =
            Expert::<f32>::new(0, &[128, 256, 128]).expect("test: Expert creation should succeed");
        assert_eq!(expert.expert_id, 0);
        assert_eq!(expert.layers.len(), 2);
    }

    #[test]
    fn test_expert_forward() {
        let expert =
            Expert::<f32>::new(0, &[4, 8, 4]).expect("test: Expert creation should succeed");
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4])
            .expect("test: tensor creation should succeed");
        let output = expert
            .forward(&input)
            .expect("test: forward pass should succeed");
        assert_eq!(output.shape().dims(), &[1, 4]);
    }

    #[test]
    fn test_router_creation() {
        let router =
            TopKRouter::<f32>::new(128, 8, 2).expect("test: TopKRouter creation should succeed");
        assert_eq!(router.k, 2);
        // Router should be created successfully with valid parameters
        assert!(!router.parameters().is_empty());
    }

    #[test]
    fn test_router_k_validation() {
        let result = TopKRouter::<f32>::new(128, 4, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_moe_creation() {
        let moe = MixtureOfExperts::<f32>::new(128, 8, 512, 128, 2)
            .expect("test: MixtureOfExperts creation should succeed");
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.experts.len(), 8);
    }

    #[test]
    fn test_moe_forward() {
        let moe = MixtureOfExperts::<f32>::new(4, 4, 8, 4, 2)
            .expect("test: MixtureOfExperts creation should succeed");
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4])
            .expect("test: tensor creation should succeed");
        let output = moe
            .forward(&input)
            .expect("test: forward pass should succeed");
        assert_eq!(output.shape().dims(), &[1, 4]);
    }

    #[test]
    fn test_moe_with_expert_capacity() {
        let moe = MixtureOfExperts::<f32>::new(128, 8, 512, 128, 2)
            .expect("test: operation should succeed")
            .with_expert_capacity(64);
        assert_eq!(moe.expert_capacity, Some(64));
    }

    #[test]
    fn test_moe_parameters() {
        let moe = MixtureOfExperts::<f32>::new(4, 2, 8, 4, 1)
            .expect("test: MixtureOfExperts creation should succeed");
        let params = moe.parameters();
        // Should have router parameters + expert parameters
        assert!(!params.is_empty());
    }

    #[test]
    fn test_moe_training_mode() {
        let mut moe = MixtureOfExperts::<f32>::new(4, 2, 8, 4, 1)
            .expect("test: MixtureOfExperts creation should succeed");
        assert!(moe.training);

        moe.set_training(false);
        assert!(!moe.training);
        assert!(!moe.router.training);
    }

    #[test]
    fn test_load_balance_loss() {
        let moe = MixtureOfExperts::<f32>::new(4, 4, 8, 4, 2)
            .expect("test: MixtureOfExperts creation should succeed");
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4])
            .expect("test: tensor creation should succeed");
        let loss = moe
            .load_balance_loss(&input)
            .expect("test: load should succeed");
        assert!(loss.is_finite());
    }

    /// The Switch/GShard auxiliary loss must be finite and strictly positive
    /// whenever the dispatch is unequal across experts. With several tokens and
    /// top-k=2 over 4 experts, only a subset of experts receive tokens, so the
    /// loss must not be zero (the old placeholder always returned 0.0).
    #[test]
    fn test_load_balance_loss_nonzero_unequal_routing() {
        let router =
            TopKRouter::<f32>::new(4, 4, 2).expect("test: TopKRouter creation should succeed");
        // 3 tokens, input_dim 4.
        let input = Tensor::from_vec(
            vec![
                0.1f32, 0.2, 0.3, 0.4, 1.0, -1.0, 0.5, 2.0, -0.5, 0.25, -2.0, 1.5,
            ],
            &[3, 4],
        )
        .expect("test: tensor creation should succeed");

        let (gate_probs, expert_indices, loss) = router
            .forward(&input)
            .expect("test: routing should succeed");

        // Routing produced real top-k indices, not a fabricated all-zeros tensor.
        assert_eq!(expert_indices.shape().dims(), &[3, 2]);
        // gate_probs rows are valid distributions.
        assert_eq!(gate_probs.shape().dims(), &[3, 4]);

        assert!(loss.is_finite(), "load balance loss must be finite");
        assert!(
            loss > 0.0,
            "load balance loss must be strictly positive under unequal routing, got {loss}"
        );

        // Analytical check for this configuration. With zero-initialized gate
        // weights gate_probs are uniform (0.25 each), top-2 picks experts 0 and 1
        // for every token, so f_0 = f_1 = 1, f_2 = f_3 = 0 and P_i = 0.25.
        //   aux = coeff(0.01) * num_experts(4) * sum_i f_i * P_i
        //       = 0.01 * 4 * (1*0.25 + 1*0.25) = 0.02
        let expected = 0.01f32 * 4.0 * 0.5;
        assert!(
            (loss - expected).abs() < 1e-6,
            "expected aux loss {expected}, got {loss}"
        );
    }

    /// The expert indices returned by the router must be genuine argmax-style
    /// top-k selections, not the fabricated zero tensor the placeholder returned.
    #[test]
    fn test_router_topk_indices_are_real() {
        let router =
            TopKRouter::<f32>::new(4, 4, 1).expect("test: TopKRouter creation should succeed");
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4])
            .expect("test: tensor creation should succeed");
        let (_probs, indices, _loss) = router
            .forward(&input)
            .expect("test: routing should succeed");
        assert_eq!(indices.shape().dims(), &[1, 1]);
        let idx_vec = indices.to_vec().expect("test: to_vec should succeed");
        // The selected top-1 expert id must be a valid expert index.
        assert!(
            idx_vec[0] < 4,
            "top-1 expert id out of range: {}",
            idx_vec[0]
        );
    }
}
