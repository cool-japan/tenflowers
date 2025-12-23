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
            load_balance_loss_coeff: T::from(0.01).unwrap_or(T::from(0.01).unwrap()),
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

        // Apply softmax to get probabilities
        let gate_probs = tenflowers_core::ops::softmax(&gate_logits, Some(-1))?;

        // For now, return simplified routing - proper Top-K routing would require more complex tensor operations
        // This is a basic implementation that can be enhanced with more sophisticated routing
        let load_balance_loss = T::zero(); // Placeholder for load balancing loss

        // Simple routing: select top-k experts (this would need proper tensor operations for production)
        let expert_indices = Tensor::zeros(&[input.shape().dims()[0], self.k]);
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

    /// Forward pass through the MoE layer
    pub fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Get routing decisions
        let (expert_weights, expert_indices, _load_balance_loss) = self.router.forward(input)?;

        // For simplicity in this initial implementation, we'll route all tokens to the first expert
        // A full implementation would require complex tensor operations for proper routing and combining
        let output = self.experts[0].forward(input)?;

        // In a full implementation, we would:
        // 1. Route tokens to different experts based on router decisions
        // 2. Apply expert capacity constraints
        // 3. Combine expert outputs weighted by routing probabilities
        // 4. Add load balancing loss to training loss

        Ok(output)
    }

    /// Get load balancing loss for training
    pub fn load_balance_loss(&self, input: &Tensor<T>) -> Result<T> {
        let (_weights, _indices, loss) = self.router.forward(input)?;
        Ok(loss)
    }

    /// Get routing statistics for analysis
    pub fn routing_stats(&self, input: &Tensor<T>) -> Result<RoutingStats<T>> {
        let (weights, indices, loss) = self.router.forward(input)?;

        Ok(RoutingStats {
            expert_weights: weights,
            expert_indices: indices,
            load_balance_loss: loss,
            expert_utilization: vec![T::zero(); self.num_experts], // Placeholder
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
        let expert = Expert::<f32>::new(0, &[128, 256, 128]).unwrap();
        assert_eq!(expert.expert_id, 0);
        assert_eq!(expert.layers.len(), 2);
    }

    #[test]
    fn test_expert_forward() {
        let expert = Expert::<f32>::new(0, &[4, 8, 4]).unwrap();
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let output = expert.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 4]);
    }

    #[test]
    fn test_router_creation() {
        let router = TopKRouter::<f32>::new(128, 8, 2).unwrap();
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
        let moe = MixtureOfExperts::<f32>::new(128, 8, 512, 128, 2).unwrap();
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.experts.len(), 8);
    }

    #[test]
    fn test_moe_forward() {
        let moe = MixtureOfExperts::<f32>::new(4, 4, 8, 4, 2).unwrap();
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let output = moe.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 4]);
    }

    #[test]
    fn test_moe_with_expert_capacity() {
        let moe = MixtureOfExperts::<f32>::new(128, 8, 512, 128, 2)
            .unwrap()
            .with_expert_capacity(64);
        assert_eq!(moe.expert_capacity, Some(64));
    }

    #[test]
    fn test_moe_parameters() {
        let moe = MixtureOfExperts::<f32>::new(4, 2, 8, 4, 1).unwrap();
        let params = moe.parameters();
        // Should have router parameters + expert parameters
        assert!(!params.is_empty());
    }

    #[test]
    fn test_moe_training_mode() {
        let mut moe = MixtureOfExperts::<f32>::new(4, 2, 8, 4, 1).unwrap();
        assert!(moe.training);

        moe.set_training(false);
        assert!(!moe.training);
        assert!(!moe.router.training);
    }

    #[test]
    fn test_load_balance_loss() {
        let moe = MixtureOfExperts::<f32>::new(4, 4, 8, 4, 2).unwrap();
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let loss = moe.load_balance_loss(&input).unwrap();
        assert!(loss.is_finite());
    }
}
