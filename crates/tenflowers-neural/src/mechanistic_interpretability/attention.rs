//! Attention head analysis: patterns, attribution, induction heads, rollout.

use super::helpers::{dot, layer_norm, mat_mul, matvec, softmax, vec_add, zeros_vec};
use super::transformer::MiTransformer;

/// Analyse per-head attention patterns and attribution scores.
pub struct AttentionAnalyzer {
    /// The model being analysed.
    pub model: MiTransformer,
}

impl AttentionAnalyzer {
    /// Wrap `model` in an `AttentionAnalyzer`.
    pub fn new(model: MiTransformer) -> Self {
        Self { model }
    }

    /// Compute per-head attention patterns for the given token sequence.
    ///
    /// Returns `[n_layers][n_heads][seq_len][seq_len]`.
    pub fn attention_patterns(&self, tokens: &[usize]) -> Vec<Vec<Vec<Vec<f64>>>> {
        let seq_len = tokens.len();
        let model = &self.model;
        let mut residual: Vec<Vec<f64>> = model.embed_tokens(tokens);
        let mut all_patterns: Vec<Vec<Vec<Vec<f64>>>> = Vec::with_capacity(model.n_layers);

        for l in 0..model.n_layers {
            let normed: Vec<Vec<f64>> = residual
                .iter()
                .map(|r| layer_norm(r, &model.layers[l].ln1_scale, &model.layers[l].ln1_bias))
                .collect();

            let head_dim = model.d_model / model.n_heads;
            let scale = 1.0 / (head_dim as f64).sqrt();
            let mut layer_patterns: Vec<Vec<Vec<f64>>> = Vec::with_capacity(model.n_heads);

            for h in 0..model.n_heads {
                let h_start = h * head_dim;
                let h_end = h_start + head_dim;

                let q_h: Vec<Vec<f64>> = normed
                    .iter()
                    .map(|r| matvec(&model.layers[l].w_q, r)[h_start..h_end].to_vec())
                    .collect();
                let k_h: Vec<Vec<f64>> = normed
                    .iter()
                    .map(|r| matvec(&model.layers[l].w_k, r)[h_start..h_end].to_vec())
                    .collect();

                let mut scores = vec![vec![0.0_f64; seq_len]; seq_len];
                for i in 0..seq_len {
                    for j in 0..=i {
                        scores[i][j] = dot(&q_h[i], &k_h[j]) * scale;
                    }
                    for j in (i + 1)..seq_len {
                        scores[i][j] = f64::NEG_INFINITY;
                    }
                }
                let attn: Vec<Vec<f64>> = scores.iter().map(|row| softmax(row)).collect();
                layer_patterns.push(attn);
                // h_end is used implicitly above via the slice
                let _ = h_end;
            }

            all_patterns.push(layer_patterns);

            let (attn_out, _) = model.attention(&normed, l);
            residual = residual
                .iter()
                .zip(attn_out.iter())
                .map(|(r, a)| vec_add(r, a))
                .collect();
            let normed2: Vec<Vec<f64>> = residual
                .iter()
                .map(|r| layer_norm(r, &model.layers[l].ln2_scale, &model.layers[l].ln2_bias))
                .collect();
            let mlp_outs: Vec<Vec<f64>> = normed2.iter().map(|r| model.mlp(r, l)).collect();
            residual = residual
                .iter()
                .zip(mlp_outs.iter())
                .map(|(r, m)| vec_add(r, m))
                .collect();
        }

        all_patterns
    }

    /// Attribution score for each attention head via direct logit attribution.
    ///
    /// Returns `[n_layers][n_heads]`.
    pub fn head_attribution(
        &self,
        tokens: &[usize],
        correct_idx: usize,
        incorrect_idx: usize,
    ) -> Vec<Vec<f64>> {
        let model = &self.model;
        let c = correct_idx.min(model.vocab_size - 1);
        let ic = incorrect_idx.min(model.vocab_size - 1);

        // Direction in residual-stream space that maximises logit-diff.
        let direction: Vec<f64> = model.unembed.iter().map(|row| row[c] - row[ic]).collect();

        let mut residual: Vec<Vec<f64>> = model.embed_tokens(tokens);
        let mut attributions: Vec<Vec<f64>> = Vec::with_capacity(model.n_layers);

        for l in 0..model.n_layers {
            let normed: Vec<Vec<f64>> = residual
                .iter()
                .map(|r| layer_norm(r, &model.layers[l].ln1_scale, &model.layers[l].ln1_bias))
                .collect();

            let head_dim = model.d_model / model.n_heads;
            let scale = 1.0 / (head_dim as f64).sqrt();
            let seq_len = tokens.len();
            let mut layer_attr: Vec<f64> = Vec::with_capacity(model.n_heads);

            for h in 0..model.n_heads {
                let h_start = h * head_dim;
                let h_end = h_start + head_dim;

                let q_h: Vec<Vec<f64>> = normed
                    .iter()
                    .map(|r| matvec(&model.layers[l].w_q, r)[h_start..h_end].to_vec())
                    .collect();
                let k_h: Vec<Vec<f64>> = normed
                    .iter()
                    .map(|r| matvec(&model.layers[l].w_k, r)[h_start..h_end].to_vec())
                    .collect();
                let v_h: Vec<Vec<f64>> = normed
                    .iter()
                    .map(|r| matvec(&model.layers[l].w_v, r)[h_start..h_end].to_vec())
                    .collect();

                let mut scores = vec![vec![0.0_f64; seq_len]; seq_len];
                for i in 0..seq_len {
                    for j in 0..=i {
                        scores[i][j] = dot(&q_h[i], &k_h[j]) * scale;
                    }
                    for j in (i + 1)..seq_len {
                        scores[i][j] = f64::NEG_INFINITY;
                    }
                }
                let attn: Vec<Vec<f64>> = scores.iter().map(|row| softmax(row)).collect();

                let last_pos = seq_len - 1;
                let mut h_out = zeros_vec(head_dim);
                for j in 0..seq_len {
                    let a = attn[last_pos][j];
                    for (k, &v) in h_out.iter_mut().zip(v_h[j].iter()) {
                        *k += a * v;
                    }
                }

                let mut full = zeros_vec(model.d_model);
                for (i, &v) in h_out.iter().enumerate() {
                    full[h_start + i] = v;
                }
                let projected = matvec(&model.layers[l].w_o, &full);

                let attr = dot(&projected, &direction);
                layer_attr.push(attr);

                let _ = h_end;
            }

            attributions.push(layer_attr);

            let (attn_out, _) = model.attention(&normed, l);
            residual = residual
                .iter()
                .zip(attn_out.iter())
                .map(|(r, a)| vec_add(r, a))
                .collect();
            let normed2: Vec<Vec<f64>> = residual
                .iter()
                .map(|r| layer_norm(r, &model.layers[l].ln2_scale, &model.layers[l].ln2_bias))
                .collect();
            let mlp_outs: Vec<Vec<f64>> = normed2.iter().map(|r| model.mlp(r, l)).collect();
            residual = residual
                .iter()
                .zip(mlp_outs.iter())
                .map(|(r, m)| vec_add(r, m))
                .collect();
        }

        attributions
    }

    /// Detect "induction heads" (copy-suppression pattern).
    ///
    /// Returns `(layer, head, induction_score)` for all heads, sorted descending.
    pub fn detect_induction_heads(&self, tokens: &[usize]) -> Vec<(usize, usize, f64)> {
        let patterns = self.attention_patterns(tokens);
        let seq_len = tokens.len();
        let mut results: Vec<(usize, usize, f64)> = Vec::new();

        for (l, layer_pats) in patterns.iter().enumerate() {
            for (h, head_pats) in layer_pats.iter().enumerate() {
                let mut score_sum = 0.0_f64;
                let mut count = 0_usize;

                for i in 2..seq_len {
                    let prev_tok = tokens[i - 1];
                    if let Some(prev_pos) = (0..i - 1).rev().find(|&p| tokens[p] == prev_tok) {
                        let attend_to = (prev_pos + 1).min(seq_len - 1);
                        score_sum += head_pats[i][attend_to];
                        count += 1;
                    }
                }

                let score = if count > 0 {
                    score_sum / count as f64
                } else {
                    0.0
                };
                results.push((l, h, score));
            }
        }

        results
            .sort_by(|(_, _, a), (_, _, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Attention rollout (Abnar & Zuidema 2020).
    ///
    /// Returns a `[seq_len][seq_len]` matrix.
    pub fn attention_rollout(&self, tokens: &[usize]) -> Vec<Vec<f64>> {
        let patterns = self.attention_patterns(tokens);
        let seq_len = tokens.len();

        let mut rollout: Vec<Vec<f64>> = (0..seq_len)
            .map(|i| {
                let mut row = zeros_vec(seq_len);
                row[i] = 1.0;
                row
            })
            .collect();

        for layer_pats in &patterns {
            let mut avg = vec![zeros_vec(seq_len); seq_len];
            let n_heads = layer_pats.len();
            for head_pat in layer_pats {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        avg[i][j] += head_pat[i][j] / n_heads as f64;
                    }
                }
            }

            let mut attn_with_res = vec![zeros_vec(seq_len); seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let id_val = if i == j { 1.0 } else { 0.0 };
                    attn_with_res[i][j] = 0.5 * avg[i][j] + 0.5 * id_val;
                }
                let row_sum: f64 = attn_with_res[i].iter().sum();
                if row_sum > 1e-10 {
                    for v in attn_with_res[i].iter_mut() {
                        *v /= row_sum;
                    }
                }
            }

            rollout = mat_mul(&attn_with_res, &rollout);
        }

        rollout
    }
}
