//! Neural isolation forest anomaly detection.

use super::building_blocks::{percentile, AdMlp};
use scirs2_core::random::{rngs::StdRng, Rng};
use scirs2_core::RngExt;

/// Configuration for [`NeuralIsolationForest`].
#[derive(Clone)]
pub struct IsolationForestConfig {
    /// Number of trees.
    pub n_trees: usize,
    /// Subsample size per tree. Default: 256.
    pub subsample_size: usize,
    /// Maximum tree depth. Default: `floor(log2(subsample_size))`.
    pub max_depth: usize,
}

impl Default for IsolationForestConfig {
    fn default() -> Self {
        let ss = 256_usize;
        let md = (ss as f64).log2().floor() as usize;
        Self {
            n_trees: 100,
            subsample_size: ss,
            max_depth: md,
        }
    }
}

/// A single node in an isolation tree.
pub struct IsolationNode {
    /// Feature index used for splitting at this node.
    pub split_feature: usize,
    /// Split threshold value.
    pub split_value: f64,
    /// Left child (x\[split_feature\] < split_value).
    pub left: Option<Box<IsolationNode>>,
    /// Right child (x\[split_feature\] >= split_value).
    pub right: Option<Box<IsolationNode>>,
    /// Number of training samples that reached this node.
    pub size: usize,
    /// Depth of this node.
    pub depth: usize,
}

impl IsolationNode {
    /// Compute the expected path length of a point `x` through this tree.
    pub fn path_length(&self, x: &[f64]) -> f64 {
        let mut node = self;
        let mut depth = 0.0_f64;

        loop {
            if node.left.is_none() && node.right.is_none() {
                depth += correction(node.size);
                return depth;
            }

            depth += 1.0;
            if x[node.split_feature] < node.split_value {
                match &node.left {
                    Some(child) => node = child,
                    None => {
                        depth += correction(node.size);
                        return depth;
                    }
                }
            } else {
                match &node.right {
                    Some(child) => node = child,
                    None => {
                        depth += correction(node.size);
                        return depth;
                    }
                }
            }
        }
    }
}

/// Expected path-length correction term `c(n)` for isolation trees.
pub fn correction(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    if n == 2 {
        return 1.0;
    }
    let n_f = n as f64;
    2.0 * harmonic(n - 1) - 2.0 * (n_f - 1.0) / n_f
}

/// Harmonic number H(n) = Σ 1/k for k=1..=n (Euler–Mascheroni approximation).
pub fn harmonic(n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    n_f.ln() + 0.577_215_664_9 + 1.0 / (2.0 * n_f)
}

/// A single isolation tree.
pub struct IsolationTree {
    /// Root node (None if tree is empty).
    pub root: Option<IsolationNode>,
    /// Subsample size this tree was built with.
    pub subsample_size: usize,
}

impl IsolationTree {
    /// Fit an isolation tree to `data` with the given `max_depth`.
    pub fn fit(data: &[Vec<f64>], max_depth: usize, rng: &mut StdRng) -> Self {
        if data.is_empty() {
            return Self {
                root: None,
                subsample_size: 0,
            };
        }
        let root = build_node(data, 0, max_depth, rng);
        Self {
            root: Some(root),
            subsample_size: data.len(),
        }
    }

    /// Compute the path length for point `x`.
    pub fn score(&self, x: &[f64]) -> f64 {
        match &self.root {
            Some(node) => node.path_length(x),
            None => 0.0,
        }
    }
}

/// Recursively build an isolation tree node.
fn build_node(
    data: &[Vec<f64>],
    depth: usize,
    max_depth: usize,
    rng: &mut StdRng,
) -> IsolationNode {
    let n = data.len();
    let n_features = data.first().map(|r| r.len()).unwrap_or(1);

    if n <= 1 || depth >= max_depth {
        return IsolationNode {
            split_feature: 0,
            split_value: 0.0,
            left: None,
            right: None,
            size: n,
            depth,
        };
    }

    let feat = (rng.random::<f64>() * n_features as f64) as usize;
    let feat = feat.min(n_features - 1);

    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for row in data.iter() {
        if feat < row.len() {
            if row[feat] < min_val {
                min_val = row[feat];
            }
            if row[feat] > max_val {
                max_val = row[feat];
            }
        }
    }

    if (max_val - min_val).abs() < 1e-15 {
        return IsolationNode {
            split_feature: feat,
            split_value: min_val,
            left: None,
            right: None,
            size: n,
            depth,
        };
    }

    let u: f64 = rng.random();
    let split_val = min_val + u * (max_val - min_val);

    let left_data: Vec<Vec<f64>> = data
        .iter()
        .filter(|r| feat < r.len() && r[feat] < split_val)
        .cloned()
        .collect();
    let right_data: Vec<Vec<f64>> = data
        .iter()
        .filter(|r| feat >= r.len() || r[feat] >= split_val)
        .cloned()
        .collect();

    let left = if left_data.is_empty() {
        None
    } else {
        Some(Box::new(build_node(&left_data, depth + 1, max_depth, rng)))
    };
    let right = if right_data.is_empty() {
        None
    } else {
        Some(Box::new(build_node(&right_data, depth + 1, max_depth, rng)))
    };

    IsolationNode {
        split_feature: feat,
        split_value: split_val,
        left,
        right,
        size: n,
        depth,
    }
}

/// Neural Isolation Forest: combines a learned embedding with standard isolation forest.
pub struct NeuralIsolationForest {
    /// Embedding network mapping raw inputs to a lower-dimensional space.
    pub embedding: AdMlp,
    /// Isolation trees built in the embedding space.
    pub trees: Vec<IsolationTree>,
    /// Configuration.
    pub config: IsolationForestConfig,
    /// Anomaly threshold.
    pub threshold: f64,
}

impl NeuralIsolationForest {
    /// Construct a new neural isolation forest.
    pub fn new(input_dim: usize, embed_dim: usize, config: IsolationForestConfig) -> Self {
        let sizes = vec![input_dim, (input_dim + embed_dim) / 2 + 1, embed_dim];
        Self {
            embedding: AdMlp::new(&sizes),
            trees: Vec::new(),
            config,
            threshold: 0.5,
        }
    }

    /// Normalisation constant `c(n)` used in isolation score computation.
    ///
    /// `c(n) = 2·H(n-1) - 2·(n-1)/n` where H is the harmonic number.
    pub fn c_n(n: usize) -> f64 {
        if n <= 1 {
            return 1.0;
        }
        correction(n)
    }

    /// Compute the isolation anomaly score in \[0,1\].
    ///
    /// `score = 2^(-E[path_length] / c(subsample_size))`
    ///
    /// Scores > 0.5 are indicative of anomalies.
    pub fn anomaly_score(&self, x: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.5;
        }
        let z = self.embedding.forward(x);
        let mean_path: f64 =
            self.trees.iter().map(|t| t.score(&z)).sum::<f64>() / self.trees.len() as f64;
        let c = correction(self.config.subsample_size).max(1.0);
        2.0_f64.powf(-mean_path / c)
    }

    /// Fit the isolation forest (train embedding then build trees).
    pub fn fit(&mut self, x_train: &[Vec<f64>], n_embed_epochs: usize, lr: f64, rng: &mut StdRng) {
        let embed_dim = self.embedding.output_dim();
        let eps = 1e-5;

        for _ in 0..n_embed_epochs {
            for sample in x_train.iter() {
                let z = self.embedding.forward(sample);
                let mean: f64 = z.iter().sum::<f64>() / z.len().max(1) as f64;
                let var: f64 =
                    z.iter().map(|zi| (zi - mean).powi(2)).sum::<f64>() / z.len().max(1) as f64;
                let loss = -var + 1e-3 * z.iter().map(|zi| zi.powi(2)).sum::<f64>();

                for layer_idx in 0..self.embedding.layers.len() {
                    let out_dim = self.embedding.layers[layer_idx].w.len();
                    let in_dim = if out_dim > 0 {
                        self.embedding.layers[layer_idx].w[0].len()
                    } else {
                        0
                    };
                    for j in 0..out_dim {
                        for i in 0..in_dim {
                            let orig = self.embedding.layers[layer_idx].w[j][i];
                            self.embedding.layers[layer_idx].w[j][i] = orig + eps;
                            let z_p = self.embedding.forward(sample);
                            let mean_p: f64 = z_p.iter().sum::<f64>() / z_p.len().max(1) as f64;
                            let var_p: f64 =
                                z_p.iter().map(|zi| (zi - mean_p).powi(2)).sum::<f64>()
                                    / z_p.len().max(1) as f64;
                            let loss_p =
                                -var_p + 1e-3 * z_p.iter().map(|zi| zi.powi(2)).sum::<f64>();
                            let g = (loss_p - loss) / eps;
                            self.embedding.layers[layer_idx].w[j][i] = orig - lr * g;
                        }
                        let orig_b = self.embedding.layers[layer_idx].b[j];
                        self.embedding.layers[layer_idx].b[j] = orig_b + eps;
                        let z_p = self.embedding.forward(sample);
                        let mean_p: f64 = z_p.iter().sum::<f64>() / z_p.len().max(1) as f64;
                        let var_p: f64 = z_p.iter().map(|zi| (zi - mean_p).powi(2)).sum::<f64>()
                            / z_p.len().max(1) as f64;
                        let loss_p = -var_p + 1e-3 * z_p.iter().map(|zi| zi.powi(2)).sum::<f64>();
                        let g = (loss_p - loss) / eps;
                        self.embedding.layers[layer_idx].b[j] = orig_b - lr * g;
                    }
                }
            }
        }

        let embedded: Vec<Vec<f64>> = x_train.iter().map(|x| self.embedding.forward(x)).collect();

        let ss = self.config.subsample_size.min(embedded.len()).max(1);
        let max_depth = self.config.max_depth;

        self.trees = (0..self.config.n_trees)
            .map(|_| {
                let subsample: Vec<Vec<f64>> = (0..ss)
                    .map(|_| {
                        let idx = (rng.random::<f64>() * embedded.len() as f64) as usize;
                        embedded[idx.min(embedded.len() - 1)].clone()
                    })
                    .collect();
                IsolationTree::fit(&subsample, max_depth, rng)
            })
            .collect();

        let _ = embed_dim;
    }

    /// Set threshold from scores on normal data.
    pub fn set_threshold(&mut self, x_normal: &[Vec<f64>]) {
        let scores: Vec<f64> = x_normal.iter().map(|x| self.anomaly_score(x)).collect();
        if scores.is_empty() {
            return;
        }
        self.threshold = percentile(&scores, 95.0);
    }

    /// Predict whether `x` is an anomaly (score > threshold).
    pub fn predict(&self, x: &[f64]) -> bool {
        self.anomaly_score(x) > self.threshold
    }
}
