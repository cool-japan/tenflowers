//! GraphSAGE layer implementation.

use tenflowers_core::{Result, TensorError};

use super::types::{
    add_bias_inplace, l2_norm, matvec, xavier_init, AggregationMethod,
};

/// GraphSAGE layer (Hamilton et al., 2017).
///
/// Produces a new `[num_nodes × out_features]` feature matrix from:
///
/// ```text
/// h_v  = W_self · h_v  +  W_neigh · AGG({h_u | u ∈ N(v)})  +  bias
/// ```
///
/// followed by optional per-node L2 normalisation.
///
/// Reference: <https://arxiv.org/abs/1706.02216>
#[derive(Debug, Clone)]
pub struct GraphSageLayer {
    /// Dimensionality of input node features.
    pub in_features: usize,
    /// Dimensionality of output node features.
    pub out_features: usize,
    /// W_self  — row-major, shape [out_features, in_features].
    w_self: Vec<f32>,
    /// W_neigh — row-major, shape [out_features, in_features].
    w_neigh: Vec<f32>,
    /// Optional bias vector, length out_features.
    pub bias: Option<Vec<f32>>,
    /// Which aggregation to apply over neighbours.
    pub aggregation: AggregationMethod,
    /// Whether to L2-normalise each output row.
    pub normalize: bool,
}

impl GraphSageLayer {
    /// Construct a new GraphSAGE layer with Xavier-initialised weights.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::invalid_argument`] when `in_features` or
    /// `out_features` is zero.
    pub fn new(
        in_features: usize,
        out_features: usize,
        aggregation: AggregationMethod,
        normalize: bool,
    ) -> Result<Self> {
        if in_features == 0 {
            return Err(TensorError::invalid_argument(
                "GraphSageLayer: in_features must be > 0".to_string(),
            ));
        }
        if out_features == 0 {
            return Err(TensorError::invalid_argument(
                "GraphSageLayer: out_features must be > 0".to_string(),
            ));
        }
        let w_self = xavier_init(out_features, in_features);
        let w_neigh = xavier_init(out_features, in_features);
        let bias = Some(vec![0.0f32; out_features]);

        Ok(Self {
            in_features,
            out_features,
            w_self,
            w_neigh,
            bias,
            aggregation,
            normalize,
        })
    }

    /// Construct without a bias term.
    pub fn new_no_bias(
        in_features: usize,
        out_features: usize,
        aggregation: AggregationMethod,
        normalize: bool,
    ) -> Result<Self> {
        let mut layer = Self::new(in_features, out_features, aggregation, normalize)?;
        layer.bias = None;
        Ok(layer)
    }

    // ── internal aggregation ──────────────────────────────────────────────────

    /// Aggregate neighbour feature vectors into a single vector of length
    /// `in_features`.  Returns a zero vector when the neighbour list is empty.
    fn aggregate_neighbors(
        &self,
        node_features: &[f32],
        num_nodes: usize,
        neighbor_list: &[usize],
    ) -> Result<Vec<f32>> {
        if neighbor_list.is_empty() {
            return Ok(vec![0.0f32; self.in_features]);
        }

        match self.aggregation {
            AggregationMethod::Mean => {
                let mut acc = vec![0.0f32; self.in_features];
                for &nb in neighbor_list {
                    if nb >= num_nodes {
                        return Err(TensorError::invalid_argument(format!(
                            "GraphSageLayer: neighbour index {nb} out of range (num_nodes = {num_nodes})"
                        )));
                    }
                    let base = nb * self.in_features;
                    for k in 0..self.in_features {
                        acc[k] += node_features[base + k];
                    }
                }
                let n = neighbor_list.len() as f32;
                for v in acc.iter_mut() {
                    *v /= n;
                }
                Ok(acc)
            }

            AggregationMethod::Max => {
                let mut acc = vec![f32::NEG_INFINITY; self.in_features];
                for &nb in neighbor_list {
                    if nb >= num_nodes {
                        return Err(TensorError::invalid_argument(format!(
                            "GraphSageLayer: neighbour index {nb} out of range (num_nodes = {num_nodes})"
                        )));
                    }
                    let base = nb * self.in_features;
                    for k in 0..self.in_features {
                        let v = node_features[base + k];
                        if v > acc[k] {
                            acc[k] = v;
                        }
                    }
                }
                // Replace any remaining -INF with 0 for a well-defined zero signal.
                for v in acc.iter_mut() {
                    if v.is_infinite() {
                        *v = 0.0;
                    }
                }
                Ok(acc)
            }

            AggregationMethod::Sum => {
                let mut acc = vec![0.0f32; self.in_features];
                for &nb in neighbor_list {
                    if nb >= num_nodes {
                        return Err(TensorError::invalid_argument(format!(
                            "GraphSageLayer: neighbour index {nb} out of range (num_nodes = {num_nodes})"
                        )));
                    }
                    let base = nb * self.in_features;
                    for k in 0..self.in_features {
                        acc[k] += node_features[base + k];
                    }
                }
                Ok(acc)
            }
        }
    }

    // ── public forward ────────────────────────────────────────────────────────

    /// Run the GraphSAGE forward pass.
    ///
    /// # Arguments
    ///
    /// * `node_features` — Flat row-major array of shape `[num_nodes, in_features]`.
    /// * `num_nodes`     — Number of nodes in the graph.
    /// * `neighbor_lists` — `neighbor_lists[i]` lists the indices of node `i`'s
    ///   neighbours. Self-loops should **not** be included; the implementation
    ///   handles the self-embedding separately.
    ///
    /// # Returns
    ///
    /// `(output_flat, out_features)` where `output_flat` has length
    /// `num_nodes * out_features` in row-major order.
    ///
    /// # Errors
    ///
    /// * Input slice length does not match `num_nodes * in_features`.
    /// * `neighbor_lists.len()` does not match `num_nodes`.
    /// * Any neighbour index is out of range.
    pub fn forward(
        &self,
        node_features: &[f32],
        num_nodes: usize,
        neighbor_lists: &[Vec<usize>],
    ) -> Result<(Vec<f32>, usize)> {
        // ── validation ──────────────────────────────────────────────────────
        let expected_len = num_nodes * self.in_features;
        if node_features.len() != expected_len {
            return Err(TensorError::invalid_argument(format!(
                "GraphSageLayer::forward: node_features length {} does not match \
                 num_nodes ({}) × in_features ({}) = {}",
                node_features.len(),
                num_nodes,
                self.in_features,
                expected_len,
            )));
        }
        if neighbor_lists.len() != num_nodes {
            return Err(TensorError::invalid_argument(format!(
                "GraphSageLayer::forward: neighbor_lists.len() = {} but num_nodes = {}",
                neighbor_lists.len(),
                num_nodes,
            )));
        }

        // ── per-node computation ────────────────────────────────────────────
        let mut output = Vec::with_capacity(num_nodes * self.out_features);

        for node_idx in 0..num_nodes {
            let node_feat_slice =
                &node_features[node_idx * self.in_features..(node_idx + 1) * self.in_features];

            // 1. Transform self embedding: W_self · h_v
            let h_self = matvec(&self.w_self, node_feat_slice, self.out_features, self.in_features);

            // 2. Aggregate neighbours
            let h_agg = self.aggregate_neighbors(
                node_features,
                num_nodes,
                &neighbor_lists[node_idx],
            )?;

            // 3. Transform neighbourhood aggregate: W_neigh · agg
            let h_neigh = matvec(&self.w_neigh, &h_agg, self.out_features, self.in_features);

            // 4. Combine: element-wise sum
            let mut h_out: Vec<f32> = h_self
                .iter()
                .zip(h_neigh.iter())
                .map(|(a, b)| a + b)
                .collect();

            // 5. Add bias
            if let Some(ref b) = self.bias {
                add_bias_inplace(&mut h_out, b);
            }

            // 6. L2 normalisation (optional)
            if self.normalize {
                let norm = l2_norm(&h_out);
                if norm > 1e-8 {
                    for v in h_out.iter_mut() {
                        *v /= norm;
                    }
                }
            }

            output.extend_from_slice(&h_out);
        }

        Ok((output, self.out_features))
    }

    // ── weight access for testing / manual initialisation ────────────────────

    /// Replace `W_self`.  Length must equal `out_features * in_features`.
    pub fn set_w_self(&mut self, w: Vec<f32>) -> Result<()> {
        let expected = self.out_features * self.in_features;
        if w.len() != expected {
            return Err(TensorError::invalid_argument(format!(
                "set_w_self: expected length {expected}, got {}",
                w.len()
            )));
        }
        self.w_self = w;
        Ok(())
    }

    /// Replace `W_neigh`.  Length must equal `out_features * in_features`.
    pub fn set_w_neigh(&mut self, w: Vec<f32>) -> Result<()> {
        let expected = self.out_features * self.in_features;
        if w.len() != expected {
            return Err(TensorError::invalid_argument(format!(
                "set_w_neigh: expected length {expected}, got {}",
                w.len()
            )));
        }
        self.w_neigh = w;
        Ok(())
    }
}
