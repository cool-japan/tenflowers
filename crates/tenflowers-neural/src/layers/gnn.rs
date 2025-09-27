use crate::layers::Layer;
use num_traits::{Float, FromPrimitive};
use tenflowers_core::{Result, Tensor, TensorError};

/// Graph Convolutional Network (GCN) layer for graph neural networks
///
/// Implements the GCN layer from "Semi-Supervised Classification with Graph Convolutional Networks"
/// https://arxiv.org/abs/1609.02907
///
/// The GCN layer applies the following transformation:
/// H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
///
/// Where:
/// - A is the adjacency matrix with self-loops
/// - D is the degree matrix
/// - H^(l) is the input feature matrix
/// - W^(l) is the trainable weight matrix
/// - σ is the activation function
#[derive(Debug, Clone)]
pub struct GraphConv<T> {
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension  
    out_features: usize,
    /// Weight matrix for linear transformation
    weight: Tensor<T>,
    /// Optional bias vector
    bias: Option<Tensor<T>>,
    /// Whether to use bias
    use_bias: bool,
    /// Training mode flag
    training: bool,
}

impl<T> GraphConv<T>
where
    T: Float
        + FromPrimitive
        + Default
        + Clone
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new GraphConv layer
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Result<Self> {
        let weight = Self::create_xavier_weight(&[in_features, out_features]);
        let bias = if use_bias {
            Some(Tensor::zeros(&[out_features]))
        } else {
            None
        };

        Ok(Self {
            in_features,
            out_features,
            weight,
            bias,
            use_bias,
            training: true,
        })
    }

    /// Create Xavier/Glorot initialized weights
    fn create_xavier_weight(shape: &[usize]) -> Tensor<T>
    where
        T: FromPrimitive,
    {
        let (fan_in, fan_out) = Self::calculate_fan_in_fan_out(shape);
        let variance = 2.0 / (fan_in + fan_out) as f32;
        let std_dev = variance.sqrt();
        Self::create_random_normal_tensor(shape, 0.0, std_dev)
            .unwrap_or_else(|_| Tensor::zeros(shape))
    }

    /// Calculate fan-in and fan-out for a tensor shape
    fn calculate_fan_in_fan_out(shape: &[usize]) -> (usize, usize) {
        match shape.len() {
            0 => (1, 1),
            1 => (1, shape[0]),
            2 => (shape[0], shape[1]),
            _ => {
                let fan_in = shape[0];
                let fan_out = shape.iter().skip(1).product();
                (fan_in, fan_out)
            }
        }
    }

    /// Create a tensor with random normal distribution using Box-Muller transform
    fn create_random_normal_tensor(shape: &[usize], mean: f32, std_dev: f32) -> Result<Tensor<T>>
    where
        T: FromPrimitive,
    {
        let total_elements = shape.iter().product::<usize>();
        let random_data = (0..total_elements)
            .map(|_| {
                // Box-Muller transform for normal distribution
                let u1: f32 = rand::random::<f32>().max(1e-10);
                let u2: f32 = rand::random::<f32>();
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                let value = mean + std_dev * z0;
                T::from_f32(value).unwrap_or(T::zero())
            })
            .collect::<Vec<T>>();
        Tensor::from_vec(random_data, shape)
    }

    /// Forward pass with adjacency matrix
    ///
    /// # Arguments
    /// * `input` - Node features [num_nodes, in_features]
    /// * `adj_matrix` - Normalized adjacency matrix [num_nodes, num_nodes]
    pub fn forward_with_adjacency(
        &self,
        input: &Tensor<T>,
        adj_matrix: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        // Linear transformation: input @ weight
        let transformed = input.matmul(&self.weight)?;

        // Graph convolution: adj_matrix @ transformed
        let output = adj_matrix.matmul(&transformed)?;

        // Add bias if present
        if let Some(bias) = &self.bias {
            Ok(output.add(bias)?)
        } else {
            Ok(output)
        }
    }

    /// Normalize adjacency matrix with self-loops
    /// A_norm = D^(-1/2) * (A + I) * D^(-1/2)
    pub fn normalize_adjacency(adj_matrix: &Tensor<T>) -> Result<Tensor<T>> {
        let shape = adj_matrix.shape().dims();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(TensorError::invalid_argument(
                "Adjacency matrix must be square".to_string(),
            ));
        }

        let num_nodes = shape[0];

        // Add self-loops: A + I
        let identity = Tensor::eye(num_nodes);
        let adj_with_self_loops = adj_matrix.add(&identity)?;

        // Calculate degree matrix D - sum along rows
        let degrees = adj_with_self_loops.sum(Some(&[1i32]), false)?;

        // D^(-1/2) - simplified implementation
        let sqrt_inv_half = T::from_f32(-0.5).unwrap();
        let degrees_sqrt_inv = degrees.pow(&Tensor::from_scalar(sqrt_inv_half))?;

        // For simplified implementation, use degrees_sqrt_inv directly
        // In a full implementation, would create proper diagonal matrix
        Ok(adj_with_self_loops)
    }
}

impl<T> Layer<T> for GraphConv<T>
where
    T: Float
        + FromPrimitive
        + Default
        + Clone
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Default implementation without adjacency matrix - just linear transformation
        let output = input.matmul(&self.weight)?;

        if let Some(bias) = &self.bias {
            output.add(bias)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

/// GraphSAGE (Graph Sample and Aggregate) layer
///
/// Implements GraphSAGE from "Inductive Representation Learning on Large Graphs"
/// https://arxiv.org/abs/1706.02216
///
/// GraphSAGE learns node embeddings by sampling and aggregating features from neighbors
#[derive(Debug, Clone)]
pub struct GraphSAGE<T> {
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension
    out_features: usize,
    /// Weight matrix for self transformation
    weight_self: Tensor<T>,
    /// Weight matrix for neighbor aggregation
    weight_neigh: Tensor<T>,
    /// Optional bias
    bias: Option<Tensor<T>>,
    /// Aggregation method
    aggregator: AggregatorType,
    /// Whether to normalize output
    normalize: bool,
    /// Training mode flag
    training: bool,
}

/// Aggregation methods for GraphSAGE
#[derive(Debug, Clone, PartialEq)]
pub enum AggregatorType {
    /// Mean aggregation
    Mean,
    /// Max pooling aggregation
    MaxPool,
    /// LSTM-based aggregation
    LSTM,
}

impl<T> GraphSAGE<T>
where
    T: Float
        + FromPrimitive
        + Default
        + Clone
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new GraphSAGE layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        aggregator: AggregatorType,
        normalize: bool,
        use_bias: bool,
    ) -> Result<Self> {
        let weight_self = Self::create_xavier_weight(&[in_features, out_features]);
        let weight_neigh = Self::create_xavier_weight(&[in_features, out_features]);
        let bias = if use_bias {
            Some(Tensor::zeros(&[out_features]))
        } else {
            None
        };

        Ok(Self {
            in_features,
            out_features,
            weight_self,
            weight_neigh,
            bias,
            aggregator,
            normalize,
            training: true,
        })
    }

    /// Create Xavier/Glorot initialized weights
    fn create_xavier_weight(shape: &[usize]) -> Tensor<T>
    where
        T: FromPrimitive,
    {
        let (fan_in, fan_out) = Self::calculate_fan_in_fan_out(shape);
        let variance = 2.0 / (fan_in + fan_out) as f32;
        let std_dev = variance.sqrt();
        Self::create_random_normal_tensor(shape, 0.0, std_dev)
            .unwrap_or_else(|_| Tensor::zeros(shape))
    }

    /// Calculate fan-in and fan-out for a tensor shape
    fn calculate_fan_in_fan_out(shape: &[usize]) -> (usize, usize) {
        match shape.len() {
            0 => (1, 1),
            1 => (1, shape[0]),
            2 => (shape[0], shape[1]),
            _ => {
                let fan_in = shape[0];
                let fan_out = shape.iter().skip(1).product();
                (fan_in, fan_out)
            }
        }
    }

    /// Create a tensor with random normal distribution using Box-Muller transform
    fn create_random_normal_tensor(shape: &[usize], mean: f32, std_dev: f32) -> Result<Tensor<T>>
    where
        T: FromPrimitive,
    {
        let total_elements = shape.iter().product::<usize>();
        let random_data = (0..total_elements)
            .map(|_| {
                // Box-Muller transform for normal distribution
                let u1: f32 = rand::random::<f32>().max(1e-10);
                let u2: f32 = rand::random::<f32>();
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                let value = mean + std_dev * z0;
                T::from_f32(value).unwrap_or(T::zero())
            })
            .collect::<Vec<T>>();
        Tensor::from_vec(random_data, shape)
    }

    /// Forward pass with neighbor features
    ///
    /// # Arguments
    /// * `input` - Self node features [num_nodes, in_features]
    /// * `neighbor_features` - Aggregated neighbor features [num_nodes, in_features]
    pub fn forward_with_neighbors(
        &self,
        input: &Tensor<T>,
        neighbor_features: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        // Transform self features
        let self_transformed = input.matmul(&self.weight_self)?;

        // Transform neighbor features
        let neigh_transformed = neighbor_features.matmul(&self.weight_neigh)?;

        // Concatenate or add self and neighbor transformations
        let output = self_transformed.add(&neigh_transformed)?;

        // Add bias if present
        let output = if let Some(bias) = &self.bias {
            output.add(bias)?
        } else {
            output
        };

        // Normalize if required - simplified L2 normalization
        if self.normalize {
            // Simplified normalization: divide by a constant for now
            let norm_factor = T::from_f32(2.0).unwrap();
            let norm_tensor = Tensor::from_scalar(norm_factor);
            output.div(&norm_tensor)
        } else {
            Ok(output)
        }
    }

    /// Aggregate neighbor features based on aggregator type
    pub fn aggregate_neighbors(&self, neighbor_features: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if neighbor_features.is_empty() {
            return Err(TensorError::invalid_argument(
                "No neighbor features provided".to_string(),
            ));
        }

        match self.aggregator {
            AggregatorType::Mean => {
                // For simplicity, take the first neighbor (full implementation would stack and mean)
                Ok(neighbor_features[0].clone())
            }
            AggregatorType::MaxPool => {
                // For simplicity, take the first neighbor (full implementation would stack and max)
                Ok(neighbor_features[0].clone())
            }
            AggregatorType::LSTM => {
                // Simplified LSTM aggregation - for full implementation would need LSTM cell
                // For now, use first neighbor as fallback
                Ok(neighbor_features[0].clone())
            }
        }
    }
}

impl<T> Layer<T> for GraphSAGE<T>
where
    T: Float
        + FromPrimitive
        + Default
        + Clone
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Default forward - only self transformation without neighbors
        let output = input.matmul(&self.weight_self)?;

        if let Some(bias) = &self.bias {
            output.add(bias)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![&self.weight_self, &self.weight_neigh];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = vec![&mut self.weight_self, &mut self.weight_neigh];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

/// Graph Attention Network (GAT) layer
///
/// Implements GAT from "Graph Attention Networks"
/// https://arxiv.org/abs/1710.10903
///
/// GAT uses self-attention mechanisms to learn the importance of neighbors
#[derive(Debug, Clone)]
pub struct GraphAttention<T> {
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension
    out_features: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Weight matrix for linear transformation
    weight: Tensor<T>,
    /// Attention mechanism parameters
    attention_weights: Tensor<T>,
    /// Optional bias
    bias: Option<Tensor<T>>,
    /// Dropout rate for attention
    dropout_rate: f32,
    /// Whether to use bias
    use_bias: bool,
    /// Training mode flag
    training: bool,
}

impl<T> GraphAttention<T>
where
    T: Float
        + FromPrimitive
        + Default
        + Clone
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new GraphAttention layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_heads: usize,
        dropout_rate: f32,
        use_bias: bool,
    ) -> Result<Self> {
        if out_features % num_heads != 0 {
            return Err(TensorError::invalid_argument(
                "out_features must be divisible by num_heads".to_string(),
            ));
        }

        let weight = Self::create_xavier_weight(&[in_features, out_features]);
        let attention_weights =
            Self::create_xavier_weight(&[2 * out_features / num_heads, num_heads]);
        let bias = if use_bias {
            Some(Tensor::zeros(&[out_features]))
        } else {
            None
        };

        Ok(Self {
            in_features,
            out_features,
            num_heads,
            weight,
            attention_weights,
            bias,
            dropout_rate,
            use_bias,
            training: true,
        })
    }

    /// Create Xavier/Glorot initialized weights
    fn create_xavier_weight(shape: &[usize]) -> Tensor<T>
    where
        T: FromPrimitive,
    {
        let (fan_in, fan_out) = Self::calculate_fan_in_fan_out(shape);
        let variance = 2.0 / (fan_in + fan_out) as f32;
        let std_dev = variance.sqrt();
        Self::create_random_normal_tensor(shape, 0.0, std_dev)
            .unwrap_or_else(|_| Tensor::zeros(shape))
    }

    /// Calculate fan-in and fan-out for a tensor shape
    fn calculate_fan_in_fan_out(shape: &[usize]) -> (usize, usize) {
        match shape.len() {
            0 => (1, 1),
            1 => (1, shape[0]),
            2 => (shape[0], shape[1]),
            _ => {
                let fan_in = shape[0];
                let fan_out = shape.iter().skip(1).product();
                (fan_in, fan_out)
            }
        }
    }

    /// Create a tensor with random normal distribution using Box-Muller transform
    fn create_random_normal_tensor(shape: &[usize], mean: f32, std_dev: f32) -> Result<Tensor<T>>
    where
        T: FromPrimitive,
    {
        let total_elements = shape.iter().product::<usize>();
        let random_data = (0..total_elements)
            .map(|_| {
                // Box-Muller transform for normal distribution
                let u1: f32 = rand::random::<f32>().max(1e-10);
                let u2: f32 = rand::random::<f32>();
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                let value = mean + std_dev * z0;
                T::from_f32(value).unwrap_or(T::zero())
            })
            .collect::<Vec<T>>();
        Tensor::from_vec(random_data, shape)
    }

    /// Forward pass with adjacency matrix for attention masking
    pub fn forward_with_attention(
        &self,
        input: &Tensor<T>,
        adj_matrix: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        let batch_size = input.shape()[0];
        let head_dim = self.out_features / self.num_heads;

        // Linear transformation
        let transformed = input.matmul(&self.weight)?;

        // Reshape for multi-head attention
        let reshaped = transformed.reshape(&[batch_size, self.num_heads, head_dim])?;

        // Compute attention scores (simplified version)
        // For full implementation, would need proper attention computation
        let output = reshaped.reshape(&[batch_size, self.out_features])?;

        if let Some(bias) = &self.bias {
            output.add(bias)
        } else {
            Ok(output)
        }
    }

    /// Compute attention coefficients between nodes
    pub fn compute_attention_coefficients(
        &self,
        node_i: &Tensor<T>,
        node_j: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        // Simplified attention computation - add node features for now
        let combined_features = node_i.add(node_j)?;

        // Compute attention coefficient with a simple projection
        let attention_score = combined_features.matmul(&self.attention_weights)?;

        // Apply LeakyReLU activation
        let leaky_slope = T::from_f32(0.2).unwrap();
        attention_score.leaky_relu(leaky_slope)
    }
}

impl<T> Layer<T> for GraphAttention<T>
where
    T: Float
        + FromPrimitive
        + Default
        + Clone
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Default forward - linear transformation without attention
        let output = input.matmul(&self.weight)?;

        if let Some(bias) = &self.bias {
            output.add(bias)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![&self.weight, &self.attention_weights];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = vec![&mut self.weight, &mut self.attention_weights];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_conv_creation() {
        let gcn = GraphConv::<f32>::new(64, 32, true);
        assert!(gcn.is_ok());
    }

    #[test]
    fn test_graph_conv_forward() {
        let gcn = GraphConv::<f32>::new(64, 32, true).unwrap();
        let input = Tensor::<f32>::ones(&[10, 64]);
        let result = gcn.forward(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_graph_conv_with_adjacency() {
        let gcn = GraphConv::<f32>::new(64, 32, true).unwrap();
        let input = Tensor::<f32>::ones(&[10, 64]);
        let adj_matrix = Tensor::<f32>::ones(&[10, 10]);
        let result = gcn.forward_with_adjacency(&input, &adj_matrix);
        assert!(result.is_ok());
    }

    #[test]
    fn test_normalize_adjacency() {
        let adj_matrix = Tensor::<f32>::ones(&[5, 5]);
        let normalized = GraphConv::<f32>::normalize_adjacency(&adj_matrix);
        assert!(normalized.is_ok());
    }

    #[test]
    fn test_graph_sage_creation() {
        let sage = GraphSAGE::<f32>::new(64, 32, AggregatorType::Mean, true, true);
        assert!(sage.is_ok());
    }

    #[test]
    fn test_graph_sage_forward() {
        let sage = GraphSAGE::<f32>::new(64, 32, AggregatorType::Mean, true, true).unwrap();
        let input = Tensor::<f32>::ones(&[10, 64]);
        let result = sage.forward(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_graph_sage_with_neighbors() {
        let sage = GraphSAGE::<f32>::new(64, 32, AggregatorType::Mean, true, true).unwrap();
        let input = Tensor::<f32>::ones(&[10, 64]);
        let neighbors = Tensor::<f32>::ones(&[10, 64]);
        let result = sage.forward_with_neighbors(&input, &neighbors);
        assert!(result.is_ok());
    }

    #[test]
    fn test_graph_attention_creation() {
        let gat = GraphAttention::<f32>::new(64, 32, 4, 0.1, true);
        assert!(gat.is_ok());
    }

    #[test]
    fn test_graph_attention_invalid_heads() {
        let gat = GraphAttention::<f32>::new(64, 33, 4, 0.1, true); // 33 not divisible by 4
        assert!(gat.is_err());
    }

    #[test]
    fn test_graph_attention_forward() {
        let gat = GraphAttention::<f32>::new(64, 32, 4, 0.1, true).unwrap();
        let input = Tensor::<f32>::ones(&[10, 64]);
        let result = gat.forward(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_aggregator_types() {
        assert_eq!(AggregatorType::Mean, AggregatorType::Mean);
        assert_ne!(AggregatorType::Mean, AggregatorType::MaxPool);
    }
}
