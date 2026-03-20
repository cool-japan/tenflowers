//! MessagePassing trait and utilities.

/// Generic message-passing interface.
///
/// Implementors define three steps:
///
/// 1. **`message`** — compute a message from a source node to a target node.
/// 2. **`aggregate`** — combine all incoming messages for a target node.
/// 3. **`update`** — produce the new target-node representation.
///
/// The [`message_passing`] function runs a complete forward pass using any
/// implementation of this trait.
pub trait MessagePassing {
    /// Compute the message sent from `source_feat` to `target_feat`.
    ///
    /// Both slices have length `feature_dim` (the input feature dimension of
    /// the nodes passed to [`message_passing`]).
    fn message(&self, source_feat: &[f32], target_feat: &[f32]) -> Vec<f32>;

    /// Aggregate a (possibly empty) collection of messages for one target node.
    ///
    /// Returns a single aggregated vector.  If `messages` is empty the
    /// implementation should return a sensible zero/identity vector.
    fn aggregate(&self, messages: &[Vec<f32>]) -> Vec<f32>;

    /// Produce the updated representation of a target node given its current
    /// features and the aggregated message.
    fn update(&self, node_feat: &[f32], aggregated: &[f32]) -> Vec<f32>;

    /// Dimension of the feature vectors that [`update`] returns.
    ///
    /// The default implementation returns the length of `node_feat`, assuming
    /// a feature-preserving update.  Override when the update changes
    /// dimensionality.
    fn output_dim(&self, input_dim: usize) -> usize {
        input_dim
    }
}

/// Run a single message-passing step on a graph.
///
/// For every node `v`, the function:
///
/// 1. Collects messages from all neighbours `u ∈ N(v)` via
///    [`MessagePassing::message`].
/// 2. Aggregates them via [`MessagePassing::aggregate`].
/// 3. Produces the new node representation via [`MessagePassing::update`].
///
/// # Arguments
///
/// * `mp`             — Implementation of [`MessagePassing`].
/// * `node_features`  — Flat `[num_nodes × feature_dim]` feature matrix.
/// * `num_nodes`      — Number of nodes.
/// * `feature_dim`    — Number of features per node (input).
/// * `neighbor_lists` — `neighbor_lists[i]` = list of neighbour node indices.
///
/// # Returns
///
/// Flat `[num_nodes × out_dim]` output, where `out_dim` is determined by
/// `mp.output_dim(feature_dim)`.
///
/// # Panics
///
/// Does **not** panic; instead returns a best-effort result even for empty
/// graphs (returns an empty Vec).
pub fn message_passing<M: MessagePassing>(
    mp: &M,
    node_features: &[f32],
    num_nodes: usize,
    feature_dim: usize,
    neighbor_lists: &[Vec<usize>],
) -> Vec<f32> {
    if num_nodes == 0 || feature_dim == 0 {
        return Vec::new();
    }

    let out_dim = mp.output_dim(feature_dim);
    let mut output = Vec::with_capacity(num_nodes * out_dim);

    for v in 0..num_nodes {
        let v_feat = &node_features[v * feature_dim..(v + 1) * feature_dim];

        // Gather messages from all neighbours
        let nb_list = if v < neighbor_lists.len() {
            neighbor_lists[v].as_slice()
        } else {
            &[]
        };

        let messages: Vec<Vec<f32>> = nb_list
            .iter()
            .filter_map(|&u| {
                if u < num_nodes {
                    let u_feat = &node_features[u * feature_dim..(u + 1) * feature_dim];
                    Some(mp.message(u_feat, v_feat))
                } else {
                    None // silently skip out-of-range neighbours
                }
            })
            .collect();

        let aggregated = mp.aggregate(&messages);
        let updated = mp.update(v_feat, &aggregated);
        output.extend_from_slice(&updated);
    }

    output
}
