//! Clustered federated learning algorithms.
//!
//! Groups clients into clusters based on gradient or loss similarity, then
//! trains a separate global model per cluster.
//!
//! | Algorithm | Reference |
//! |-----------|-----------|
//! | [`IfcaAlgorithm`] | Ghosh et al., 2020 |
//! | [`HypCluster`] | hierarchical ward clustering |

use super::types::{
    max_layer_delta_norm, weighted_avg_loss, ClientUpdate, FederatedConfig, FederatedError,
    GlobalUpdate, ModelParams,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Flatten a `ModelParams` into a single `Vec<f32>`.
fn flatten(params: &ModelParams) -> Vec<f32> {
    params.iter().flat_map(|l| l.iter().copied()).collect()
}

/// Cosine distance between two flat vectors (1 − cosine_similarity).
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON {
        return 1.0;
    }
    let cos = (dot / (na * nb)).clamp(-1.0, 1.0);
    1.0 - cos
}

/// Squared Euclidean distance between two flat vectors.
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

/// Sample-weighted average of ModelParams slices.
fn weighted_model_avg(params_list: &[ModelParams], weights: &[f32]) -> ModelParams {
    if params_list.is_empty() {
        return Vec::new();
    }
    let num_layers = params_list[0].len();
    let total: f32 = weights.iter().sum::<f32>().max(f32::EPSILON);
    let mut result: ModelParams = params_list[0]
        .iter()
        .map(|l| vec![0.0_f32; l.len()])
        .collect();
    for (params, &w) in params_list.iter().zip(weights.iter()) {
        let wt = w / total;
        for l in 0..num_layers.min(params.len()) {
            for (j, &v) in params[l].iter().enumerate() {
                if j < result[l].len() {
                    result[l][j] += wt * v;
                }
            }
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// IFCA
// ─────────────────────────────────────────────────────────────────────────────

/// Client assignment result from one IFCA round.
#[derive(Debug, Clone)]
pub struct IfcaAssignment {
    /// Client index.
    pub client_id: usize,
    /// Assigned cluster index.
    pub cluster_id: usize,
    /// Cluster loss for this client.
    pub loss: f32,
}

/// IFCA — Iterative Federated Clustering Algorithm (Ghosh et al., NeurIPS 2020).
///
/// Maintains `k` cluster models.  Each round:
/// 1. **Assignment**: each client is assigned to the cluster whose model
///    yields the lowest local loss.
/// 2. **Aggregation**: within each cluster, aggregate client updates via
///    FedAvg.
///
/// Converges when cluster assignments stabilize.
///
/// # Reference
/// A. Ghosh, J. Chung, D. Yin, K. Ramchandran,
/// "An Efficient Framework for Clustered Federated Learning," NeurIPS 2020.
#[derive(Debug, Clone)]
pub struct IfcaAlgorithm {
    /// Shared configuration.
    pub config: FederatedConfig,
    /// Number of clusters `k`.
    pub k: usize,
    /// Model parameters for each cluster.
    pub cluster_params: Vec<ModelParams>,
    /// Current cluster assignment for each client.
    pub client_assignments: Vec<usize>,
    /// Current round.
    pub round: usize,
}

impl IfcaAlgorithm {
    /// Create a new IFCA algorithm with `k` clusters.
    ///
    /// All clusters are initialized from `initial_params` with a small
    /// perturbation to break symmetry (scaled by cluster index).
    pub fn new(
        config: FederatedConfig,
        initial_params: ModelParams,
        k: usize,
    ) -> Result<Self, FederatedError> {
        if k == 0 {
            return Err(FederatedError::InvalidConfig {
                reason: "k must be at least 1".to_string(),
            });
        }
        // Initialize cluster models with slight perturbation to break symmetry
        let cluster_params: Vec<ModelParams> = (0..k)
            .map(|c| {
                initial_params
                    .iter()
                    .map(|layer| {
                        layer
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| v + 0.01 * (c as f32) * ((i % 7) as f32 - 3.0) / 7.0)
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let num_clients = config.num_clients;
        // Initially assign clients round-robin
        let client_assignments: Vec<usize> = (0..num_clients).map(|i| i % k).collect();

        Ok(IfcaAlgorithm {
            config,
            k,
            cluster_params,
            client_assignments,
            round: 0,
        })
    }

    /// Assign a client to the cluster with the lowest loss.
    ///
    /// `client_losses[c]` is the loss of client `client_id` under cluster
    /// model `c`.  Returns the assigned cluster index.
    pub fn assign_client(
        &mut self,
        client_id: usize,
        client_losses: &[f32],
    ) -> Result<usize, FederatedError> {
        if client_id >= self.config.num_clients {
            return Err(FederatedError::InvalidClientId {
                id: client_id,
                max: self.config.num_clients,
            });
        }
        if client_losses.len() != self.k {
            return Err(FederatedError::InvalidConfig {
                reason: format!(
                    "client_losses length {} != k {}",
                    client_losses.len(),
                    self.k
                ),
            });
        }

        let best = client_losses
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        self.client_assignments[client_id] = best;
        Ok(best)
    }

    /// Aggregate updates for a specific cluster.
    ///
    /// `updates`: updates from clients assigned to `cluster_id`.
    pub fn aggregate_cluster(
        &mut self,
        cluster_id: usize,
        updates: &[ClientUpdate],
    ) -> Result<GlobalUpdate, FederatedError> {
        if cluster_id >= self.k {
            return Err(FederatedError::InvalidConfig {
                reason: format!("cluster_id {cluster_id} >= k {}", self.k),
            });
        }
        if updates.is_empty() {
            // No updates for this cluster — keep current model
            self.round += 1;
            return Ok(GlobalUpdate {
                round: self.round,
                params: self.cluster_params[cluster_id].clone(),
                participating_clients: 0,
                avg_loss: 0.0,
                convergence_metric: 0.0,
            });
        }

        let total: f32 = updates.iter().map(|u| u.num_samples as f32).sum();
        let num_layers = self.cluster_params[cluster_id].len();
        let mut new_params: ModelParams = self.cluster_params[cluster_id]
            .iter()
            .map(|l| vec![0.0_f32; l.len()])
            .collect();

        for update in updates {
            let wt = update.num_samples as f32 / total;
            for l in 0..num_layers.min(update.params.len()) {
                for j in 0..new_params[l].len().min(update.params[l].len()) {
                    new_params[l][j] += wt * update.params[l][j];
                }
            }
        }

        let old_params = self.cluster_params[cluster_id].clone();
        let convergence_metric = max_layer_delta_norm(&old_params, &new_params);
        let avg_loss = weighted_avg_loss(updates);
        self.cluster_params[cluster_id] = new_params;
        self.round += 1;

        Ok(GlobalUpdate {
            round: self.round,
            params: self.cluster_params[cluster_id].clone(),
            participating_clients: updates.len(),
            avg_loss,
            convergence_metric,
        })
    }

    /// Aggregate all clusters in one call.
    ///
    /// `all_updates`: all client updates; each is routed to its assigned cluster
    /// based on `client_assignments`.
    pub fn aggregate_all(
        &mut self,
        all_updates: &[ClientUpdate],
    ) -> Result<Vec<GlobalUpdate>, FederatedError> {
        if all_updates.is_empty() {
            return Err(FederatedError::NoClientUpdates);
        }

        // Group updates by cluster
        let mut cluster_buckets: Vec<Vec<ClientUpdate>> = vec![Vec::new(); self.k];
        for update in all_updates {
            let cid = update.client_id;
            if cid < self.client_assignments.len() {
                let cluster = self.client_assignments[cid];
                cluster_buckets[cluster.min(self.k - 1)].push(update.clone());
            } else {
                cluster_buckets[0].push(update.clone());
            }
        }

        let mut results = Vec::with_capacity(self.k);
        for c in 0..self.k {
            let bucket = cluster_buckets[c].clone();
            let result = self.aggregate_cluster(c, &bucket)?;
            results.push(result);
        }
        Ok(results)
    }

    /// Get the cluster model for a given cluster index.
    pub fn cluster_model(&self, cluster_id: usize) -> Option<&ModelParams> {
        self.cluster_params.get(cluster_id)
    }

    /// Number of clients in each cluster.
    pub fn cluster_sizes(&self) -> Vec<usize> {
        (0..self.k)
            .map(|c| self.client_assignments.iter().filter(|&&a| a == c).count())
            .collect()
    }

    /// Current communication round.
    pub fn current_round(&self) -> usize {
        self.round
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HypCluster
// ─────────────────────────────────────────────────────────────────────────────

/// Hierarchical clustering state node.
#[derive(Debug, Clone)]
pub struct ClusterNode {
    /// Indices of clients belonging to this cluster.
    pub members: Vec<usize>,
    /// Ward linkage distance at which this cluster was formed.
    pub merge_distance: f32,
    /// Optional left child cluster index.
    pub left: Option<usize>,
    /// Optional right child cluster index.
    pub right: Option<usize>,
}

/// HypCluster — hierarchical client clustering by gradient similarity.
///
/// Groups clients into clusters by iteratively merging the pair of clusters
/// with the smallest Ward linkage distance, computed from cosine distances
/// of their gradient (update) vectors.
///
/// Ward's criterion minimizes the total within-cluster variance:
/// ```text
/// d_Ward(A, B) = (|A| · |B|) / (|A| + |B|) · ‖mean_A − mean_B‖²
/// ```
///
/// After building the full dendrogram, clients are split into `k` clusters
/// by cutting the tree at the appropriate level.
#[derive(Debug, Clone)]
pub struct HypCluster {
    /// Shared configuration.
    pub config: FederatedConfig,
    /// Number of target clusters.
    pub k: usize,
    /// The dendrogram (list of cluster nodes).
    pub dendrogram: Vec<ClusterNode>,
    /// Final cluster assignments after cutting the dendrogram.
    pub assignments: Vec<usize>,
}

impl HypCluster {
    /// Create a new HypCluster with `k` target clusters.
    pub fn new(config: FederatedConfig, k: usize) -> Result<Self, FederatedError> {
        if k == 0 {
            return Err(FederatedError::InvalidConfig {
                reason: "k must be at least 1".to_string(),
            });
        }
        let num_clients = config.num_clients;
        Ok(HypCluster {
            config,
            k,
            dendrogram: Vec::new(),
            assignments: vec![0; num_clients],
        })
    }

    /// Perform hierarchical clustering on client gradient vectors.
    ///
    /// `updates`: the client updates from a single FL round.  The gradient
    /// vectors are computed as the delta between client params and a reference
    /// (or simply the flattened params for standalone use).
    ///
    /// Builds the dendrogram using Ward linkage on cosine distances and cuts
    /// it to produce `k` clusters.
    pub fn cluster(&mut self, updates: &[ClientUpdate]) -> Result<(), FederatedError> {
        if updates.is_empty() {
            return Err(FederatedError::NoClientUpdates);
        }
        let n = updates.len();
        if n <= self.k {
            // Each client is its own cluster
            for (i, update) in updates.iter().enumerate() {
                let cid = update.client_id.min(self.assignments.len() - 1);
                self.assignments[cid] = i.min(self.k - 1);
            }
            return Ok(());
        }

        let flat: Vec<Vec<f32>> = updates.iter().map(|u| flatten(&u.params)).collect();
        let samples: Vec<f32> = updates.iter().map(|u| u.num_samples as f32).collect();

        // Initialize: each client is its own cluster
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut cluster_means: Vec<Vec<f32>> = flat.clone();
        let mut cluster_sizes: Vec<f32> = samples.clone();
        self.dendrogram.clear();

        // Add leaf nodes
        for i in 0..n {
            self.dendrogram.push(ClusterNode {
                members: vec![i],
                merge_distance: 0.0,
                left: None,
                right: None,
            });
        }

        let mut active: Vec<usize> = (0..n).collect();

        while active.len() > self.k {
            let na = active.len();
            // Find pair with minimum Ward distance
            let mut best_i = 0;
            let mut best_j = 1;
            let mut best_dist = f32::INFINITY;

            for ii in 0..na {
                for jj in (ii + 1)..na {
                    let ci = active[ii];
                    let cj = active[jj];
                    let si = cluster_sizes[ci];
                    let sj = cluster_sizes[cj];
                    // Ward distance: (si*sj)/(si+sj) * sq_dist(mean_i, mean_j)
                    let ward =
                        (si * sj) / (si + sj) * sq_dist(&cluster_means[ci], &cluster_means[cj]);
                    if ward < best_dist {
                        best_dist = ward;
                        best_i = ii;
                        best_j = jj;
                    }
                }
            }

            let ci = active[best_i];
            let cj = active[best_j];
            let si = cluster_sizes[ci];
            let sj = cluster_sizes[cj];

            // Merge: compute new mean
            let dim = cluster_means[ci].len();
            let new_mean: Vec<f32> = (0..dim)
                .map(|d| (si * cluster_means[ci][d] + sj * cluster_means[cj][d]) / (si + sj))
                .collect();

            let mut new_members = clusters[ci].clone();
            new_members.extend_from_slice(&clusters[cj]);

            let new_node = ClusterNode {
                members: new_members.clone(),
                merge_distance: best_dist.sqrt(),
                left: Some(ci),
                right: Some(cj),
            };
            let new_idx = self.dendrogram.len();
            self.dendrogram.push(new_node);

            // Replace ci with new merged cluster, remove cj
            cluster_means.push(new_mean);
            cluster_sizes.push(si + sj);
            clusters.push(new_members);

            active[best_i] = new_idx;
            active.remove(best_j);
        }

        // Assign cluster labels
        for (label, &cluster_idx) in active.iter().enumerate() {
            for &member in &clusters[cluster_idx] {
                let client_id = updates[member].client_id;
                if client_id < self.assignments.len() {
                    self.assignments[client_id] = label.min(self.k - 1);
                }
            }
        }

        Ok(())
    }

    /// Get the cluster assignment for a client.
    pub fn assignment(&self, client_id: usize) -> Option<usize> {
        self.assignments.get(client_id).copied()
    }

    /// Get all client IDs assigned to a given cluster.
    pub fn clients_in_cluster(&self, cluster_id: usize) -> Vec<usize> {
        self.assignments
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == cluster_id)
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute cluster stability: fraction of clients that kept the same
    /// assignment compared to `previous_assignments`.
    pub fn assignment_stability(&self, previous_assignments: &[usize]) -> f32 {
        if previous_assignments.len() != self.assignments.len() || self.assignments.is_empty() {
            return 0.0;
        }
        let stable = self
            .assignments
            .iter()
            .zip(previous_assignments.iter())
            .filter(|(a, b)| a == b)
            .count();
        stable as f32 / self.assignments.len() as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ClusteredFLMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Metrics for evaluating clustered federated learning.
#[derive(Debug, Clone, Default)]
pub struct ClusteredFLMetrics {
    /// Fraction of clients that maintained the same cluster assignment
    /// between consecutive rounds (range [0, 1]).
    pub assignment_stability: f32,
    /// Average cosine distance between clients in different clusters.
    pub inter_cluster_diversity: f32,
    /// Average cosine distance between clients in the same cluster.
    pub intra_cluster_compactness: f32,
    /// Number of active (non-empty) clusters.
    pub active_clusters: usize,
    /// Per-cluster average loss.
    pub cluster_losses: Vec<f32>,
}

impl ClusteredFLMetrics {
    /// Create a new zeroed metrics report.
    pub fn new() -> Self {
        ClusteredFLMetrics::default()
    }

    /// Compute inter- and intra-cluster diversity metrics.
    ///
    /// `updates`: client updates; `assignments[i]` is the cluster of client `i`.
    pub fn compute_diversity(&mut self, updates: &[ClientUpdate], assignments: &[usize]) {
        if updates.len() < 2 {
            return;
        }

        let flat: Vec<Vec<f32>> = updates.iter().map(|u| flatten(&u.params)).collect();
        let n = updates.len();

        let mut inter_sum = 0.0_f32;
        let mut inter_count = 0_usize;
        let mut intra_sum = 0.0_f32;
        let mut intra_count = 0_usize;

        for i in 0..n {
            let ci = if i < assignments.len() {
                assignments[i]
            } else {
                0
            };
            for j in (i + 1)..n {
                let cj = if j < assignments.len() {
                    assignments[j]
                } else {
                    0
                };
                let dist = cosine_distance(&flat[i], &flat[j]);
                if ci == cj {
                    intra_sum += dist;
                    intra_count += 1;
                } else {
                    inter_sum += dist;
                    inter_count += 1;
                }
            }
        }

        self.inter_cluster_diversity = if inter_count > 0 {
            inter_sum / inter_count as f32
        } else {
            0.0
        };
        self.intra_cluster_compactness = if intra_count > 0 {
            intra_sum / intra_count as f32
        } else {
            0.0
        };
    }

    /// Compute per-cluster loss averages.
    pub fn compute_cluster_losses(
        &mut self,
        updates: &[ClientUpdate],
        assignments: &[usize],
        num_clusters: usize,
    ) {
        let mut sums = vec![0.0_f32; num_clusters];
        let mut counts = vec![0_usize; num_clusters];
        for (update, &cluster) in updates.iter().zip(assignments.iter()) {
            if cluster < num_clusters {
                sums[cluster] += update.loss;
                counts[cluster] += 1;
            }
        }
        self.cluster_losses = sums
            .iter()
            .zip(counts.iter())
            .map(|(&s, &c)| if c > 0 { s / c as f32 } else { 0.0 })
            .collect();
        self.active_clusters = counts.iter().filter(|&&c| c > 0).count();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cluster model ensemble
// ─────────────────────────────────────────────────────────────────────────────

/// Ensemble of cluster models for inference.
///
/// Combines predictions from all cluster models using weighted averaging,
/// where weights are given by a client's similarity to each cluster's model.
#[derive(Debug, Clone)]
pub struct ClusterEnsemble {
    /// Cluster model parameters.
    pub cluster_params: Vec<ModelParams>,
}

impl ClusterEnsemble {
    /// Create from an [`IfcaAlgorithm`]'s cluster models.
    pub fn from_ifca(ifca: &IfcaAlgorithm) -> Self {
        ClusterEnsemble {
            cluster_params: ifca.cluster_params.clone(),
        }
    }

    /// Compute softmax weights for a query based on cosine similarity to each
    /// cluster mean, then return the mixture model.
    ///
    /// `query_params`: the client's local model parameters.
    /// Returns a weighted average of cluster models.
    pub fn mixture_model(&self, query_params: &ModelParams) -> ModelParams {
        if self.cluster_params.is_empty() {
            return Vec::new();
        }
        let q_flat = flatten(query_params);

        // Compute cosine similarities to each cluster model
        let sims: Vec<f32> = self
            .cluster_params
            .iter()
            .map(|cp| {
                let c_flat = flatten(cp);
                let dot: f32 = q_flat.iter().zip(c_flat.iter()).map(|(&a, &b)| a * b).sum();
                let na = q_flat
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(f32::EPSILON);
                let nb = c_flat
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(f32::EPSILON);
                (dot / (na * nb)).clamp(-1.0, 1.0)
            })
            .collect();

        // Softmax weights
        let max_sim = sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = sims.iter().map(|&s| (s - max_sim).exp()).collect();
        let exp_sum: f32 = exps.iter().sum::<f32>().max(f32::EPSILON);
        let weights: Vec<f32> = exps.iter().map(|&e| e / exp_sum).collect();

        weighted_model_avg(&self.cluster_params, &weights)
    }
}
