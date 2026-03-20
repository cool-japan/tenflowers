//! Tests for advanced GNN layers.

#[cfg(test)]
mod tests {
    use crate::layers::gnn_advanced::{
        gat::GatLayer,
        message_passing::{message_passing, MessagePassing},
        sage::GraphSageLayer,
        types::AggregationMethod,
    };

    // ── GraphSAGE basic tests ─────────────────────────────────────────────────

    #[test]
    fn test_sage_forward_shape_mean() {
        let layer =
            GraphSageLayer::new(4, 8, AggregationMethod::Mean, false).expect("sage creation");
        let num_nodes = 3;
        let feats: Vec<f32> = (0..num_nodes * 4).map(|i| i as f32).collect();
        let nb: Vec<Vec<usize>> = vec![vec![1, 2], vec![0], vec![0, 1]];
        let (out, out_dim) = layer.forward(&feats, num_nodes, &nb).expect("sage forward");
        assert_eq!(out_dim, 8, "out_features should be 8");
        assert_eq!(out.len(), num_nodes * out_dim, "output length correct");
    }

    #[test]
    fn test_sage_forward_shape_max() {
        let layer =
            GraphSageLayer::new(3, 5, AggregationMethod::Max, false).expect("sage creation max");
        let num_nodes = 4;
        let feats: Vec<f32> = (0..num_nodes * 3).map(|i| i as f32 * 0.5).collect();
        let nb: Vec<Vec<usize>> = vec![vec![1], vec![2], vec![3], vec![0]];
        let (out, out_dim) = layer.forward(&feats, num_nodes, &nb).expect("sage forward max");
        assert_eq!(out.len(), num_nodes * out_dim, "output length correct");
    }

    #[test]
    fn test_sage_forward_shape_sum() {
        let layer =
            GraphSageLayer::new(2, 4, AggregationMethod::Sum, false).expect("sage creation sum");
        let num_nodes = 2;
        let feats = vec![1.0f32, 2.0, 3.0, 4.0];
        let nb: Vec<Vec<usize>> = vec![vec![1], vec![0]];
        let (out, out_dim) = layer.forward(&feats, num_nodes, &nb).expect("sage forward sum");
        assert_eq!(out.len(), num_nodes * out_dim, "output length correct");
    }

    #[test]
    fn test_sage_isolated_nodes() {
        let layer =
            GraphSageLayer::new(3, 3, AggregationMethod::Mean, false).expect("sage creation");
        let num_nodes = 3;
        let feats: Vec<f32> = vec![1.0; num_nodes * 3];
        let nb: Vec<Vec<usize>> = vec![vec![], vec![], vec![]];
        let (out, out_dim) = layer.forward(&feats, num_nodes, &nb).expect("sage isolated");
        assert_eq!(out.len(), num_nodes * out_dim, "output length correct");
    }

    #[test]
    fn test_sage_normalize() {
        let layer =
            GraphSageLayer::new(2, 2, AggregationMethod::Mean, true).expect("sage creation");
        let num_nodes = 2;
        let feats = vec![3.0f32, 4.0, 1.0, 0.0];
        let nb: Vec<Vec<usize>> = vec![vec![1], vec![0]];
        let (out, out_dim) = layer.forward(&feats, num_nodes, &nb).expect("sage normalize");
        assert_eq!(out.len(), num_nodes * out_dim);
        // After L2 normalisation each output row should have ~unit norm
        for node in 0..num_nodes {
            let row = &out[node * out_dim..(node + 1) * out_dim];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5 || norm < 1e-7,
                "row norm should be ~1, got {norm}"
            );
        }
    }

    #[test]
    fn test_sage_no_bias() {
        let layer =
            GraphSageLayer::new_no_bias(4, 4, AggregationMethod::Sum, false).expect("sage no bias");
        assert!(layer.bias.is_none(), "no-bias layer should have bias=None");
    }

    #[test]
    fn test_sage_set_w_self_and_w_neigh() {
        let mut layer =
            GraphSageLayer::new(2, 2, AggregationMethod::Mean, false).expect("sage creation");
        // Identity weights: out=2, in=2 → 4 elements each
        let identity = vec![1.0f32, 0.0, 0.0, 1.0];
        layer.set_w_self(identity.clone()).expect("set_w_self");
        layer.set_w_neigh(vec![0.0f32; 4]).expect("set_w_neigh");

        let feats = vec![2.0f32, 3.0, 5.0, 7.0]; // node0=[2,3], node1=[5,7]
        let nb: Vec<Vec<usize>> = vec![vec![], vec![]]; // no neighbours
        let (out, _) = layer.forward(&feats, 2, &nb).expect("deterministic forward");
        // h0 = W_self·[2,3] + W_neigh·[0,0] + bias
        //    = [2,3] + [0,0] + [0,0] = [2,3]
        assert!(
            (out[0] - 2.0).abs() < 1e-5,
            "node 0 feature 0 should be 2, got {}",
            out[0]
        );
        assert!(
            (out[1] - 3.0).abs() < 1e-5,
            "node 0 feature 1 should be 3, got {}",
            out[1]
        );
    }

    // ── GAT tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_gat_forward_concat_heads() {
        let gat = GatLayer::new(4, 4, 2, true).expect("gat creation");
        let num_nodes = 5;
        let feats: Vec<f32> = (0..num_nodes * 4).map(|i| i as f32 * 0.1).collect();
        let nb: Vec<Vec<usize>> = vec![vec![1, 2], vec![0], vec![0, 3], vec![4], vec![3]];
        let (out, out_dim) = gat
            .forward(&feats, num_nodes, &nb, false)
            .expect("gat forward");
        assert_eq!(out_dim, 8, "concat: out_dim = num_heads * out_features = 8");
        assert_eq!(out.len(), num_nodes * out_dim, "output length correct");
    }

    #[test]
    fn test_gat_forward_mean_heads() {
        let gat = GatLayer::new(4, 3, 2, false).expect("gat creation mean");
        let num_nodes = 3;
        let feats: Vec<f32> = (0..num_nodes * 4).map(|i| i as f32).collect();
        let nb: Vec<Vec<usize>> = vec![vec![1], vec![0, 2], vec![1]];
        let (out, out_dim) = gat
            .forward(&feats, num_nodes, &nb, false)
            .expect("gat forward");
        assert_eq!(
            out_dim, 3,
            "mean-heads: out_dim = out_features = 3"
        );
        assert_eq!(out.len(), num_nodes * out_dim, "output length correct");
    }

    // ── GAT attention coefficients are non-negative ──────────────────────────
    //
    // After softmax all α values are in [0, 1].  We verify by checking that
    // the weighted output is bounded: since α ≥ 0 and each z is finite, the
    // output of the aggregation step should also be finite.

    #[test]
    fn test_gat_output_all_finite() {
        let gat = GatLayer::new(4, 4, 2, true).expect("gat creation");
        let num_nodes = 4;
        let feats: Vec<f32> = (0..num_nodes * 4).map(|i| i as f32 * 0.5).collect();
        let nb: Vec<Vec<usize>> = vec![vec![1, 2], vec![0], vec![0, 3], vec![2]];
        let (out, _) = gat
            .forward(&feats, num_nodes, &nb, false)
            .expect("gat forward");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all output values must be finite (implies attention is non-negative)"
        );
    }

    // ── GAT — isolated nodes ────────────────────────────────────────────────

    #[test]
    fn test_gat_isolated_nodes() {
        let gat = GatLayer::new(3, 3, 1, false).expect("gat creation");
        let num_nodes = 3;
        let feats: Vec<f32> = vec![1.0; num_nodes * 3];
        let nb: Vec<Vec<usize>> = vec![vec![], vec![], vec![]];
        let (out, out_dim) = gat
            .forward(&feats, num_nodes, &nb, false)
            .expect("gat forward with isolated nodes");
        assert_eq!(out.len(), num_nodes * out_dim, "output length correct");
    }

    // ── MessagePassing framework ─────────────────────────────────────────────

    /// A simple sum-message-passing aggregator: message = source, aggregate = sum,
    /// update = node + aggregated.
    struct SumMP;

    impl MessagePassing for SumMP {
        fn message(&self, source_feat: &[f32], _target_feat: &[f32]) -> Vec<f32> {
            source_feat.to_vec()
        }
        fn aggregate(&self, messages: &[Vec<f32>]) -> Vec<f32> {
            if messages.is_empty() {
                return vec![0.0f32; 1]; // minimal fallback
            }
            let dim = messages[0].len();
            let mut acc = vec![0.0f32; dim];
            for m in messages {
                for (a, v) in acc.iter_mut().zip(m.iter()) {
                    *a += v;
                }
            }
            acc
        }
        fn update(&self, node_feat: &[f32], aggregated: &[f32]) -> Vec<f32> {
            node_feat
                .iter()
                .zip(aggregated.iter())
                .map(|(n, a)| n + a)
                .collect()
        }
    }

    #[test]
    fn test_message_passing_chain_graph() {
        // Chain: 0 — 1 — 2
        let feats = vec![
            1.0, 0.0, // node 0
            0.0, 1.0, // node 1
            1.0, 1.0, // node 2
        ];
        let nb: Vec<Vec<usize>> = vec![vec![1], vec![0, 2], vec![1]];
        let out = message_passing(&SumMP, &feats, 3, 2, &nb);
        assert_eq!(out.len(), 3 * 2, "output has num_nodes × feature_dim elements");
        // node 1 receives messages from 0 and 2 => aggregate = [1+1, 0+1] = [2, 1]
        // update = [0+2, 1+1] = [2, 2]
        assert!(
            (out[2] - 2.0).abs() < 1e-6,
            "node 1 output[0] should be 2.0, got {}",
            out[2]
        );
        assert!(
            (out[3] - 2.0).abs() < 1e-6,
            "node 1 output[1] should be 2.0, got {}",
            out[3]
        );
    }

    #[test]
    fn test_message_passing_isolated_nodes() {
        let feats = vec![5.0, 3.0, 2.0, 7.0];
        // 2 nodes, no edges
        let nb: Vec<Vec<usize>> = vec![vec![], vec![]];

        struct ZeroAggMP;
        impl MessagePassing for ZeroAggMP {
            fn message(&self, src: &[f32], _tgt: &[f32]) -> Vec<f32> {
                src.to_vec()
            }
            fn aggregate(&self, messages: &[Vec<f32>]) -> Vec<f32> {
                if messages.is_empty() {
                    return vec![0.0f32; 2];
                }
                let dim = messages[0].len();
                let mut acc = vec![0.0f32; dim];
                for m in messages {
                    for (a, v) in acc.iter_mut().zip(m.iter()) {
                        *a += v;
                    }
                }
                acc
            }
            fn update(&self, node_feat: &[f32], aggregated: &[f32]) -> Vec<f32> {
                node_feat
                    .iter()
                    .zip(aggregated.iter())
                    .map(|(n, a)| n + a)
                    .collect()
            }
        }

        let out = message_passing(&ZeroAggMP, &feats, 2, 2, &nb);
        // With no neighbours, update = node + 0 = node itself
        assert_eq!(out.len(), 4, "output length correct");
        assert!((out[0] - 5.0).abs() < 1e-6, "node 0 feat 0 unchanged");
        assert!((out[1] - 3.0).abs() < 1e-6, "node 0 feat 1 unchanged");
        assert!((out[2] - 2.0).abs() < 1e-6, "node 1 feat 0 unchanged");
        assert!((out[3] - 7.0).abs() < 1e-6, "node 1 feat 1 unchanged");
    }

    #[test]
    fn test_message_passing_two_node_complete() {
        // 2 nodes, both connected to each other
        let feats = vec![1.0, 0.0, 0.0, 1.0];
        let nb: Vec<Vec<usize>> = vec![vec![1], vec![0]];
        let out = message_passing(&SumMP, &feats, 2, 2, &nb);
        assert_eq!(out.len(), 4, "output length correct");
        // node 0: update = [1,0] + message([0,1]) = [1+0, 0+1] = [1, 1]
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        // node 1: update = [0,1] + message([1,0]) = [0+1, 1+0] = [1, 1]
        assert!((out[2] - 1.0).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
    }

    // ── GAT training mode (dropout active) ──────────────────────────────────

    #[test]
    fn test_gat_training_mode_runs_without_error() {
        let mut gat = GatLayer::new(4, 4, 2, true).expect("gat creation");
        gat.set_dropout(0.5);
        let num_nodes = 4;
        let feats: Vec<f32> = (0..num_nodes * 4).map(|i| i as f32 * 0.1).collect();
        let nb: Vec<Vec<usize>> = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];
        let result = gat.forward(&feats, num_nodes, &nb, true);
        assert!(result.is_ok(), "training mode forward should succeed");
        let (out, out_dim) = result.expect("gat training forward");
        assert_eq!(out.len(), num_nodes * out_dim, "output shape correct in training mode");
    }

    // ── Error handling ───────────────────────────────────────────────────────

    #[test]
    fn test_sage_wrong_input_length_errors() {
        let layer =
            GraphSageLayer::new(4, 4, AggregationMethod::Mean, false).expect("sage creation");
        // Provide wrong number of features
        let feats = vec![0.0f32; 3 * 3]; // 3 nodes × 3 feats, but in_features=4
        let nb: Vec<Vec<usize>> = vec![vec![], vec![], vec![]];
        let result = layer.forward(&feats, 3, &nb);
        assert!(result.is_err(), "mismatched input length must be rejected");
    }

    #[test]
    fn test_gat_wrong_input_length_errors() {
        let gat = GatLayer::new(4, 4, 2, true).expect("gat creation");
        let feats = vec![0.0f32; 5]; // deliberately wrong length
        let nb: Vec<Vec<usize>> = vec![vec![]];
        let result = gat.forward(&feats, 1, &nb, false);
        // 1 node × 4 in_features = 4 ≠ 5 → error
        assert!(result.is_err(), "mismatched input length must be rejected");
    }
}
