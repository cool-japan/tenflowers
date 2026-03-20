//! Concept evaluation metrics: completeness, MI, and full evaluation report.

use super::cbm::ConceptBottleneckModel;

/// Aggregate concept evaluation metrics.
#[derive(Debug, Clone)]
pub struct ConceptMetrics {
    pub concept_accuracy: Vec<f64>,
    pub concept_completeness: f64,
    pub concept_alignment: f64,
    pub disentanglement: f64,
}

/// Compute concept completeness as best single-feature R² from concepts → model logits.
pub fn compute_concept_completeness(
    concept_predictions: &[Vec<f64>],
    model_logits: &[Vec<f64>],
) -> f64 {
    let n = concept_predictions.len().min(model_logits.len());
    if n == 0 {
        return 0.0;
    }
    let y: Vec<f64> = model_logits
        .iter()
        .take(n)
        .map(|v| v.first().copied().unwrap_or(0.0))
        .collect();
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    if ss_tot < 1e-12 {
        return 1.0;
    }

    let n_feats = concept_predictions[0].len();
    if n_feats == 0 {
        return 0.0;
    }
    let mut best_r2 = 0.0_f64;
    for j in 0..n_feats {
        let x_j: Vec<f64> = concept_predictions.iter().take(n).map(|v| v[j]).collect();
        let xm = x_j.iter().sum::<f64>() / n as f64;
        let ym = y_mean;
        let cov: f64 = x_j
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - xm) * (yi - ym))
            .sum();
        let var_x: f64 = x_j
            .iter()
            .map(|xi| (xi - xm).powi(2))
            .sum::<f64>()
            .max(1e-12);
        let beta = cov / var_x;
        let ss_res: f64 = x_j
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (yi - (ym + beta * (xi - xm))).powi(2))
            .sum();
        let r2 = (1.0 - ss_res / ss_tot).clamp(0.0, 1.0);
        if r2 > best_r2 {
            best_r2 = r2;
        }
    }
    best_r2
}

/// Discretised mutual information between two concept activation vectors.
pub fn concept_mutual_information(concept_a: &[f64], concept_b: &[f64], n_bins: usize) -> f64 {
    let n = concept_a.len().min(concept_b.len());
    if n == 0 || n_bins == 0 {
        return 0.0;
    }
    let bins = n_bins.max(2);

    let min_a = concept_a[..n].iter().cloned().fold(f64::INFINITY, f64::min);
    let max_a = concept_a[..n]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_b = concept_b[..n].iter().cloned().fold(f64::INFINITY, f64::min);
    let max_b = concept_b[..n]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let range_a = (max_a - min_a).max(1e-12);
    let range_b = (max_b - min_b).max(1e-12);

    let mut joint = vec![vec![0_usize; bins]; bins];
    for i in 0..n {
        let ia = (((concept_a[i] - min_a) / range_a * bins as f64).floor() as usize).min(bins - 1);
        let ib = (((concept_b[i] - min_b) / range_b * bins as f64).floor() as usize).min(bins - 1);
        joint[ia][ib] += 1;
    }

    let n_f = n as f64;
    let p_joint: Vec<Vec<f64>> = joint
        .iter()
        .map(|row| row.iter().map(|&c| c as f64 / n_f).collect())
        .collect();

    let p_a: Vec<f64> = p_joint.iter().map(|row| row.iter().sum()).collect();
    let p_b: Vec<f64> = (0..bins)
        .map(|j| p_joint.iter().map(|row| row[j]).sum())
        .collect();

    let mut mi = 0.0_f64;
    for i in 0..bins {
        for j in 0..bins {
            let pij = p_joint[i][j];
            if pij > 1e-12 && p_a[i] > 1e-12 && p_b[j] > 1e-12 {
                mi += pij * (pij / (p_a[i] * p_b[j])).ln();
            }
        }
    }
    mi.max(0.0)
}

/// Compute a full [`ConceptMetrics`] report.
pub fn evaluate_concepts(
    model: &ConceptBottleneckModel,
    x_test: &[Vec<f64>],
    concept_test: &[Vec<f64>],
    label_test: &[usize],
) -> ConceptMetrics {
    let n = x_test.len().min(concept_test.len()).min(label_test.len());
    let n_concepts = model.config.n_concepts;

    let concept_accuracy = if n == 0 {
        vec![0.0; n_concepts]
    } else {
        model.concept_accuracy(x_test, concept_test)
    };

    let concept_preds: Vec<Vec<f64>> = x_test[..n]
        .iter()
        .map(|x| model.encode_concepts(x))
        .collect();
    let logits_all: Vec<Vec<f64>> = x_test[..n]
        .iter()
        .map(|x| {
            let (_, logits) = model.forward(x);
            logits
        })
        .collect();
    let concept_completeness =
        compute_concept_completeness(&concept_preds, &logits_all).clamp(0.0, 1.0);

    let concept_alignment = if n == 0 {
        0.0
    } else {
        let label_f: Vec<f64> = label_test[..n].iter().map(|&l| l as f64).collect();
        let lm = label_f.iter().sum::<f64>() / n as f64;
        let mut corrs = Vec::new();
        for k in 0..n_concepts {
            let ck: Vec<f64> = concept_preds
                .iter()
                .map(|cp| cp.get(k).copied().unwrap_or(0.0))
                .collect();
            let cm = ck.iter().sum::<f64>() / n as f64;
            let cov: f64 = ck
                .iter()
                .zip(label_f.iter())
                .map(|(ci, li)| (ci - cm) * (li - lm))
                .sum();
            let std_c = ck
                .iter()
                .map(|ci| (ci - cm).powi(2))
                .sum::<f64>()
                .sqrt()
                .max(1e-12);
            let std_l = label_f
                .iter()
                .map(|li| (li - lm).powi(2))
                .sum::<f64>()
                .sqrt()
                .max(1e-12);
            corrs.push((cov / (std_c * std_l)).clamp(-1.0, 1.0).abs());
        }
        if corrs.is_empty() {
            0.0
        } else {
            corrs.iter().sum::<f64>() / corrs.len() as f64
        }
    };

    let disentanglement = if n_concepts < 2 || concept_preds.is_empty() {
        1.0
    } else {
        let mut mi_sum = 0.0_f64;
        let mut count = 0_usize;
        for i in 0..n_concepts {
            for j in (i + 1)..n_concepts {
                let ca: Vec<f64> = concept_preds
                    .iter()
                    .map(|cp| cp.get(i).copied().unwrap_or(0.0))
                    .collect();
                let cb: Vec<f64> = concept_preds
                    .iter()
                    .map(|cp| cp.get(j).copied().unwrap_or(0.0))
                    .collect();
                let mi = concept_mutual_information(&ca, &cb, 10);
                mi_sum += mi;
                count += 1;
            }
        }
        if count == 0 {
            1.0
        } else {
            let max_mi = std::f64::consts::LN_10;
            (1.0 - (mi_sum / count as f64) / max_mi).clamp(0.0, 1.0)
        }
    };

    ConceptMetrics {
        concept_accuracy,
        concept_completeness,
        concept_alignment,
        disentanglement,
    }
}
