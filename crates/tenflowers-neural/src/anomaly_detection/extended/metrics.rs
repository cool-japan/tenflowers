//! Evaluation metrics for neural anomaly detection (f64-based).

/// Summary anomaly detection evaluation metrics.
#[derive(Debug, Clone)]
pub struct AnomalyMetrics {
    /// Area under the ROC curve.
    pub auc_roc: f64,
    /// Average precision (area under PR curve).
    pub average_precision: f64,
    /// F1 score at the supplied threshold.
    pub f1_at_threshold: f64,
    /// Precision at the supplied threshold.
    pub precision: f64,
    /// Recall at the supplied threshold.
    pub recall: f64,
}

/// Compute area under the ROC curve via trapezoidal integration.
///
/// `scores`: anomaly scores (higher = more anomalous).
/// `labels`: `true` = positive (anomaly).
pub fn compute_auc_roc(scores: &[f64], labels: &[bool]) -> f64 {
    if scores.len() != labels.len() || scores.is_empty() {
        return 0.5;
    }
    let n_pos = labels.iter().filter(|&&l| l).count() as f64;
    let n_neg = labels.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }

    let mut pairs: Vec<(f64, bool)> = scores.iter().copied().zip(labels.iter().copied()).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut tp = 0.0_f64;
    let mut fp = 0.0_f64;
    let mut auc = 0.0_f64;
    let mut prev_fp = 0.0_f64;
    let mut prev_tp = 0.0_f64;

    for (_, label) in &pairs {
        if *label {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        auc += (fp / n_neg - prev_fp / n_neg) * (tp / n_pos + prev_tp / n_pos) * 0.5;
        prev_fp = fp;
        prev_tp = tp;
    }

    auc.clamp(0.0, 1.0)
}

/// Compute average precision (area under precision-recall curve).
///
/// `scores`: anomaly scores (higher = more anomalous).
/// `labels`: `true` = positive (anomaly).
pub fn compute_average_precision(scores: &[f64], labels: &[bool]) -> f64 {
    if scores.len() != labels.len() || scores.is_empty() {
        return 0.0;
    }
    let n_pos = labels.iter().filter(|&&l| l).count() as f64;
    if n_pos == 0.0 {
        return 0.0;
    }

    let mut pairs: Vec<(f64, bool)> = scores.iter().copied().zip(labels.iter().copied()).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut tp = 0.0_f64;
    let mut fp = 0.0_f64;
    let mut ap = 0.0_f64;
    let mut prev_recall = 0.0_f64;
    let mut prev_prec = 1.0_f64;

    for (_, label) in &pairs {
        if *label {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let prec = tp / (tp + fp);
        let recall = tp / n_pos;
        ap += (recall - prev_recall) * (prec + prev_prec) * 0.5;
        prev_recall = recall;
        prev_prec = prec;
    }

    ap.clamp(0.0, 1.0)
}

/// Compute a full set of anomaly detection metrics at a given threshold.
pub fn compute_anomaly_metrics(scores: &[f64], labels: &[bool], threshold: f64) -> AnomalyMetrics {
    let auc_roc = compute_auc_roc(scores, labels);
    let average_precision = compute_average_precision(scores, labels);

    let (mut tp, mut fp, mut fn_) = (0_usize, 0_usize, 0_usize);
    for (s, l) in scores.iter().zip(labels.iter()) {
        let pred = *s >= threshold;
        match (pred, *l) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => {}
        }
    }

    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let f1_at_threshold = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    AnomalyMetrics {
        auc_roc,
        average_precision,
        f1_at_threshold,
        precision,
        recall,
    }
}
