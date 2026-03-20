//! Prototypical Networks (Snell et al., 2017).

// ─────────────────────────────────────────────────────────────────────────────
// Distance functions
// ─────────────────────────────────────────────────────────────────────────────

/// Euclidean (L2) distance between two vectors.
///
/// Returns 0 if either vector is empty.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Cosine similarity between two vectors, clipped to `[-1, 1]`.
///
/// Returns 0 if either vector has zero norm.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Dot product between two vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// DistanceMetric and PrototypicalNetwork
// ─────────────────────────────────────────────────────────────────────────────

/// Distance metric used by a [`PrototypicalNetwork`].
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceMetric {
    /// Standard Euclidean (L2) distance.
    Euclidean,
    /// Cosine similarity (converted to a pseudo-distance as 1 − similarity).
    Cosine,
    /// Raw dot product (higher is closer; negated before softmax).
    DotProduct,
}

/// A Prototypical Network for N-way K-shot classification.
///
/// The network assumes that embeddings are already computed externally (e.g.
/// by a CNN encoder).  The class prototype is the mean embedding of all
/// support examples belonging to that class, and queries are classified by
/// proximity to the nearest prototype.
#[derive(Debug, Clone)]
pub struct PrototypicalNetwork {
    /// Dimensionality of the embedding space.
    pub embedding_dim: usize,
    /// Distance metric used to compare embeddings to prototypes.
    pub distance: DistanceMetric,
}

impl PrototypicalNetwork {
    /// Create a new `PrototypicalNetwork`.
    pub fn new(embedding_dim: usize, distance: DistanceMetric) -> Self {
        Self {
            embedding_dim,
            distance,
        }
    }

    /// Compute one prototype (mean embedding) per class from the support set.
    ///
    /// # Arguments
    ///
    /// * `support_embeddings` – `[n_way × k_shot, embedding_dim]` embeddings.
    /// * `labels` – integer class label for each support embedding (values in
    ///   `0..n_way`).
    /// * `n_way` – number of classes in this episode.
    ///
    /// # Returns
    ///
    /// A `Vec` of `n_way` prototype vectors, each of length `embedding_dim`.
    pub fn compute_prototypes(
        &self,
        support_embeddings: &[Vec<f32>],
        labels: &[usize],
        n_way: usize,
    ) -> Vec<Vec<f32>> {
        let dim = self.embedding_dim;
        let mut sums: Vec<Vec<f32>> = (0..n_way).map(|_| vec![0.0; dim]).collect();
        let mut counts: Vec<usize> = vec![0; n_way];

        for (emb, &lbl) in support_embeddings.iter().zip(labels.iter()) {
            if lbl < n_way {
                counts[lbl] += 1;
                for (s, e) in sums[lbl].iter_mut().zip(emb.iter()) {
                    *s += e;
                }
            }
        }

        sums.iter_mut()
            .zip(counts.iter())
            .map(|(sum, &cnt)| {
                let n = cnt.max(1) as f32;
                sum.iter().map(|s| s / n).collect()
            })
            .collect()
    }

    /// Compute a score matrix `[n_query, n_way]` between queries and
    /// prototypes.
    ///
    /// The values represent *negative distances* (or raw similarities for
    /// `DotProduct`) so that `softmax` gives the predicted class distribution.
    ///
    /// # Arguments
    ///
    /// * `query_embeddings` – `[n_query, embedding_dim]`.
    /// * `prototypes` – `[n_way, embedding_dim]`, from [`compute_prototypes`].
    ///
    /// [`compute_prototypes`]: PrototypicalNetwork::compute_prototypes
    pub fn query_distances(
        &self,
        query_embeddings: &[Vec<f32>],
        prototypes: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        query_embeddings
            .iter()
            .map(|q| {
                prototypes
                    .iter()
                    .map(|p| match self.distance {
                        DistanceMetric::Euclidean => -euclidean_distance(q, p),
                        DistanceMetric::Cosine => cosine_similarity(q, p) - 1.0,
                        DistanceMetric::DotProduct => dot_product(q, p),
                    })
                    .collect()
            })
            .collect()
    }

    /// Predict the class for each query embedding.
    ///
    /// Returns the index of the class (prototype) with the highest score.
    pub fn predict(&self, query_embeddings: &[Vec<f32>], prototypes: &[Vec<f32>]) -> Vec<usize> {
        let scores = self.query_distances(query_embeddings, prototypes);
        scores
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Compute the prototypical cross-entropy loss.
    ///
    /// For each query, a softmax distribution is formed over the negative
    /// distances to prototypes.  The loss is the mean negative log-probability
    /// of the true class.
    ///
    /// # Arguments
    ///
    /// * `query_embeddings` – `[n_query, embedding_dim]`.
    /// * `prototypes` – `[n_way, embedding_dim]`.
    /// * `query_labels` – true class index `(0..n_way)` for each query.
    pub fn loss(
        &self,
        query_embeddings: &[Vec<f32>],
        prototypes: &[Vec<f32>],
        query_labels: &[usize],
    ) -> f32 {
        if query_embeddings.is_empty() {
            return 0.0;
        }
        let scores = self.query_distances(query_embeddings, prototypes);
        let mut total_loss = 0.0_f32;

        for (row, &lbl) in scores.iter().zip(query_labels.iter()) {
            let max_score = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row.iter().map(|s| (s - max_score).exp()).sum();
            let log_softmax = if lbl < row.len() {
                (row[lbl] - max_score) - exp_sum.ln()
            } else {
                f32::NEG_INFINITY
            };
            total_loss -= log_softmax;
        }

        total_loss / query_embeddings.len() as f32
    }

    /// Compute accuracy as the fraction of correct predictions.
    pub fn accuracy(&self, predictions: &[usize], targets: &[usize]) -> f32 {
        if predictions.is_empty() {
            return 0.0;
        }
        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(p, t)| p == t)
            .count();
        correct as f32 / predictions.len() as f32
    }
}
