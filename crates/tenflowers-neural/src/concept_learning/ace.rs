//! ACE — Automatic Concept-based Explanations.

use super::helpers::{dot_f64, l2_norm_f64};
use scirs2_core::random::{rngs::StdRng, Rng};
use scirs2_core::RngExt;

/// Configuration for [`AceExplainer`].
#[derive(Debug, Clone)]
pub struct AceConfig {
    pub n_concept_clusters: usize,
    pub min_concept_size: usize,
    pub concept_importance_threshold: f64,
}

/// A discovered concept cluster.
#[derive(Debug, Clone)]
pub struct Concept {
    pub name: String,
    /// Cluster centroid in activation space.
    pub center: Vec<f64>,
    /// Indices of training examples belonging to this concept.
    pub examples: Vec<usize>,
    pub tcav_score: f64,
    pub importance: f64,
}

/// ACE-style concept explainer.
pub struct AceExplainer {
    pub concepts: Vec<Concept>,
    pub config: AceConfig,
}

impl AceExplainer {
    pub fn new(config: AceConfig) -> Self {
        Self {
            concepts: Vec::new(),
            config,
        }
    }

    /// K-means clustering (Lloyd's algorithm, Euclidean distance).
    pub fn k_means_clustering(
        data: &[Vec<f64>],
        k: usize,
        n_iters: usize,
        rng: &mut StdRng,
    ) -> Vec<usize> {
        let n = data.len();
        if n == 0 || k == 0 {
            return vec![0; n];
        }
        let k = k.min(n);
        let dim = data[0].len();

        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = rng.random_range(i..n);
            indices.swap(i, j);
        }
        let mut centers: Vec<Vec<f64>> = indices[..k].iter().map(|&i| data[i].clone()).collect();
        let mut assignments = vec![0_usize; n];

        for _iter in 0..n_iters {
            let mut changed = false;
            for (i, x) in data.iter().enumerate() {
                let best = centers
                    .iter()
                    .enumerate()
                    .map(|(ci, c)| {
                        let d: f64 = x.iter().zip(c.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                        (ci, d)
                    })
                    .min_by(|(_, da), (_, db)| {
                        da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(ci, _)| ci)
                    .unwrap_or(0);
                if assignments[i] != best {
                    assignments[i] = best;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
            let mut new_centers = vec![vec![0.0_f64; dim]; k];
            let mut counts = vec![0_usize; k];
            for (i, x) in data.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for (j, &xj) in x.iter().enumerate() {
                    new_centers[c][j] += xj;
                }
            }
            for (c, cnt) in counts.iter().enumerate() {
                if *cnt > 0 {
                    for v in new_centers[c].iter_mut() {
                        *v /= *cnt as f64;
                    }
                    centers[c] = new_centers[c].clone();
                }
            }
        }
        assignments
    }

    /// Discover concepts via k-means on the provided activations.
    pub fn discover_concepts(
        &mut self,
        activations: &[Vec<f64>],
        rng: &mut StdRng,
    ) -> Vec<Concept> {
        let k = self.config.n_concept_clusters;
        let assignments = Self::k_means_clustering(activations, k, 100, rng);

        let dim = if activations.is_empty() {
            0
        } else {
            activations[0].len()
        };
        let mut centers = vec![vec![0.0_f64; dim]; k];
        let mut member_lists: Vec<Vec<usize>> = vec![Vec::new(); k];

        for (i, &c) in assignments.iter().enumerate() {
            member_lists[c].push(i);
            for (j, &v) in activations[i].iter().enumerate() {
                if j < dim {
                    centers[c][j] += v;
                }
            }
        }
        for (c, members) in member_lists.iter().enumerate() {
            let cnt = members.len();
            if cnt > 0 {
                for v in centers[c].iter_mut() {
                    *v /= cnt as f64;
                }
            }
        }

        let mut concepts: Vec<Concept> = (0..k)
            .filter(|&c| member_lists[c].len() >= self.config.min_concept_size)
            .map(|c| Concept {
                name: format!("concept_{c}"),
                center: centers[c].clone(),
                examples: member_lists[c].clone(),
                tcav_score: 0.0,
                importance: 0.0,
            })
            .collect();

        if concepts.is_empty() {
            concepts = (0..k)
                .map(|c| Concept {
                    name: format!("concept_{c}"),
                    center: centers[c].clone(),
                    examples: member_lists[c].clone(),
                    tcav_score: 0.0,
                    importance: 0.0,
                })
                .collect();
        }

        self.concepts = concepts.clone();
        concepts
    }

    /// Rank stored concepts by TCAV score.
    pub fn rank_concepts_by_importance(
        &mut self,
        model_fn: &dyn Fn(&[f64]) -> f64,
        activations: &[Vec<f64>],
    ) {
        let eps = 1e-4;
        for concept in self.concepts.iter_mut() {
            let cav = &concept.center;
            if cav.is_empty() || activations.is_empty() {
                continue;
            }
            let mut pos_count = 0_usize;
            let mut total = 0_usize;
            for h in activations.iter() {
                if h.len() != cav.len() {
                    continue;
                }
                let h_plus: Vec<f64> = h
                    .iter()
                    .zip(cav.iter())
                    .map(|(hi, ci)| hi + eps * ci)
                    .collect();
                let h_minus: Vec<f64> = h
                    .iter()
                    .zip(cav.iter())
                    .map(|(hi, ci)| hi - eps * ci)
                    .collect();
                let dd = (model_fn(&h_plus) - model_fn(&h_minus)) / (2.0 * eps);
                if dd > 0.0 {
                    pos_count += 1;
                }
                total += 1;
            }
            let score = if total > 0 {
                pos_count as f64 / total as f64
            } else {
                0.0
            };
            concept.tcav_score = score;
            concept.importance = score;
        }
        self.concepts.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Explain one activation by cosine similarity to each concept centroid.
    pub fn explain_prediction(&self, activation: &[f64]) -> Vec<(String, f64)> {
        self.concepts
            .iter()
            .map(|c| {
                let sim = if c.center.is_empty() || activation.is_empty() {
                    0.0
                } else {
                    let na = l2_norm_f64(activation).max(1e-12);
                    let nc = l2_norm_f64(&c.center).max(1e-12);
                    dot_f64(activation, &c.center) / (na * nc)
                };
                (c.name.clone(), sim)
            })
            .collect()
    }
}
