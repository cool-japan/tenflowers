//! CdMetrics — Causal Discovery Evaluation metrics.

// ─────────────────────────────────────────────────────────────────────────────
// 10. CdMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Structural Hamming Distance between predicted and true DAGs.
///
/// SHD = # edge insertions + # edge deletions + # reversals.
pub fn shd(pred_adj: &[Vec<bool>], true_adj: &[Vec<bool>]) -> usize {
    let n = pred_adj.len().min(true_adj.len());
    let mut s = 0usize;
    for i in 0..n {
        let n2 = pred_adj[i].len().min(true_adj[i].len());
        for j in 0..n2 {
            if i == j {
                continue;
            }
            let p = pred_adj[i][j];
            let t = true_adj[i][j];
            let p_rev = if j < pred_adj.len() && i < pred_adj[j].len() {
                pred_adj[j][i]
            } else {
                false
            };
            let t_rev = if j < true_adj.len() && i < true_adj[j].len() {
                true_adj[j][i]
            } else {
                false
            };

            // Count each undirected pair once (i < j)
            if i < j {
                let pred_fwd = p;
                let pred_bwd = p_rev;
                let true_fwd = t;
                let true_bwd = t_rev;

                match (pred_fwd || pred_bwd, true_fwd || true_bwd) {
                    (false, false) => {}     // both absent
                    (true, false) => s += 1, // insertion
                    (false, true) => s += 1, // deletion
                    (true, true) => {
                        // Both present; check reversal
                        if pred_fwd != true_fwd {
                            s += 1; // reversal
                        }
                    }
                }
            }
        }
    }
    s
}

/// F1 score over the skeleton (undirected edges) of the DAG.
pub fn f1_skeleton(pred_adj: &[Vec<bool>], true_adj: &[Vec<bool>]) -> f32 {
    let n = pred_adj.len().min(true_adj.len());
    let mut tp = 0u32;
    let mut fp = 0u32;
    let mut fn_ = 0u32;

    for i in 0..n {
        for j in (i + 1)..n {
            let p_edge = (j < pred_adj[i].len() && pred_adj[i][j])
                || (i < pred_adj[j].len() && pred_adj[j][i]);
            let t_edge = (j < true_adj[i].len() && true_adj[i][j])
                || (i < true_adj[j].len() && true_adj[j][i]);
            match (p_edge, t_edge) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }
    }

    let prec = if tp + fp == 0 {
        0.0
    } else {
        tp as f32 / (tp + fp) as f32
    };
    let rec = if tp + fn_ == 0 {
        0.0
    } else {
        tp as f32 / (tp + fn_) as f32
    };
    if prec + rec < 1e-12 {
        0.0
    } else {
        2.0 * prec * rec / (prec + rec)
    }
}

/// Normalised SHD: divide raw SHD by the total number of possible edges n*(n-1)/2.
pub fn normalized_shd(pred_adj: &[Vec<bool>], true_adj: &[Vec<bool>]) -> f32 {
    let n = pred_adj.len().min(true_adj.len());
    if n < 2 {
        return 0.0;
    }
    let possible = (n * (n - 1) / 2) as f32;
    shd(pred_adj, true_adj) as f32 / possible
}

/// AUROC for edge presence, treating `pred_weights[i][j]` as a score and
/// `true_adj[i][j]` as the ground-truth label.
///
/// Computed via the trapezoidal approximation.
pub fn auroc_edges(pred_weights: &[Vec<f32>], true_adj: &[Vec<bool>]) -> f32 {
    let n = pred_weights.len().min(true_adj.len());
    if n == 0 {
        return 0.5;
    }

    // Collect (score, label) pairs for all i ≠ j
    let mut pairs: Vec<(f32, bool)> = Vec::new();
    for i in 0..n {
        let n2 = pred_weights[i].len().min(if i < true_adj.len() {
            true_adj[i].len()
        } else {
            0
        });
        for j in 0..n2 {
            if i == j {
                continue;
            }
            let score = pred_weights[i][j];
            let label = true_adj[i][j];
            pairs.push((score, label));
        }
    }

    if pairs.is_empty() {
        return 0.5;
    }

    // Sort by descending score
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos = pairs.iter().filter(|&&(_, l)| l).count();
    let n_neg = pairs.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }

    // Compute AUC via trapezoidal rule over FPR axis.
    // Each negative sample (FP) steps FPR by 1/n_neg; we integrate TPR.
    let mut auroc = 0.0f32;
    let mut prev_tpr = 0.0f32;
    let mut tp_count = 0u32;
    let dfpr = 1.0 / n_neg as f32;

    for &(_, label) in &pairs {
        if label {
            tp_count += 1;
        } else {
            let tpr = tp_count as f32 / n_pos as f32;
            auroc += (prev_tpr + tpr) / 2.0 * dfpr;
            prev_tpr = tpr;
        }
    }

    auroc.clamp(0.0, 1.0)
}
