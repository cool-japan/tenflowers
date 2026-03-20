//! Tests for graph-level GNN operations.

#[cfg(test)]
mod tests {
    use super::super::{
        layers::{Gatv2Layer, GinLayer},
        pooling::{
            global_add_pool, global_max_pool, global_mean_pool, global_sum_pool,
            DiffPool, MinCutPool,
        },
        types::{Graph, GnnError},
    };

    // Helper: small 3-node complete graph, features in R^4.
    fn triangle_graph() -> Graph {
        let feats = vec![
            vec![1.0, 0.0, 0.5, -1.0],
            vec![0.0, 1.0, -0.5, 2.0],
            vec![-1.0, -1.0, 1.0, 0.0],
        ];
        let adj = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        Graph::new(feats, adj).expect("valid graph")
    }

    // ── Graph creation ──────────────────────────────────────────────────────

    #[test]
    fn test_graph_creation_valid() {
        let g = triangle_graph();
        assert_eq!(g.num_nodes(), 3);
        assert_eq!(g.feat_dim(), 4);
    }

    #[test]
    fn test_graph_empty_error() {
        let result = Graph::new(vec![], vec![]);
        assert!(matches!(result, Err(GnnError::EmptyGraph)));
    }

    #[test]
    fn test_graph_adj_not_square_error() {
        let feats = vec![vec![1.0_f32]];
        let adj = vec![vec![0.0_f32, 1.0_f32]];
        let result = Graph::new(feats, adj);
        assert!(matches!(result, Err(GnnError::AdjacencyNotSquare { .. })));
    }

    #[test]
    fn test_graph_feature_mismatch_error() {
        let feats = vec![vec![1.0_f32], vec![2.0_f32]];
        let adj = vec![vec![0.0_f32; 3]; 3]; // 3×3 adj but 2 node features
        let result = Graph::new(feats, adj);
        assert!(matches!(result, Err(GnnError::FeatureCountMismatch { .. })));
    }

    // ── Degree & Laplacian ──────────────────────────────────────────────────

    #[test]
    fn test_degree_triangle() {
        let g = triangle_graph();
        let deg = g.degree();
        assert_eq!(deg, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_laplacian_structure() {
        let g = triangle_graph();
        let lap = g.laplacian();
        // Diagonal = degree, off-diagonal = -adj
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!((lap[i][j] - 2.0).abs() < 1e-6);
                } else {
                    assert!((lap[i][j] + 1.0).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_normalized_adj_row_sums() {
        let g = triangle_graph();
        let a_hat = g.normalized_adj();
        // For a regular graph each row sums to 1 (since D^{-1/2} A D^{-1/2}
        // with uniform degrees reduces to A / degree).
        for row in &a_hat {
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row sum {s} ≠ 1.0");
        }
    }

    #[test]
    fn test_sym_norm_laplacian_identity_minus_normalized_adj() {
        let g = triangle_graph();
        let l_sym = g.sym_norm_laplacian();
        let a_hat = g.normalized_adj();
        let n = g.num_nodes();
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 - a_hat[i][j] } else { -a_hat[i][j] };
                assert!((l_sym[i][j] - expected).abs() < 1e-6);
            }
        }
    }

    // ── Global pooling ───────────────────────────────────────────────────────

    #[test]
    fn test_global_mean_pool_correct() {
        let feats = vec![
            vec![0.0_f32, 2.0],
            vec![2.0_f32, 4.0],
            vec![4.0_f32, 6.0],
        ];
        let out = global_mean_pool(&feats);
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_global_max_pool_correct() {
        let feats = vec![
            vec![0.0_f32, 5.0],
            vec![3.0_f32, 2.0],
            vec![1.0_f32, 4.0],
        ];
        let out = global_max_pool(&feats);
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_global_sum_pool_correct() {
        let feats = vec![
            vec![1.0_f32, 2.0],
            vec![3.0_f32, 4.0],
        ];
        let out = global_sum_pool(&feats);
        assert!((out[0] - 4.0).abs() < 1e-6);
        assert!((out[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_global_add_pool_alias() {
        let feats = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
        assert_eq!(global_add_pool(&feats), global_sum_pool(&feats));
    }

    #[test]
    fn test_global_mean_pool_empty() {
        let out: Vec<f32> = global_mean_pool(&[]);
        assert!(out.is_empty());
    }

    // ── DiffPool ─────────────────────────────────────────────────────────────

    #[test]
    fn test_diffpool_assignment_shape() {
        let dp = DiffPool::new(4, 2);
        let g = triangle_graph();
        let s = dp.assignment(&g.node_features);
        assert_eq!(s.len(), 3); // num_nodes
        assert_eq!(s[0].len(), 2); // num_output_nodes
    }

    #[test]
    fn test_diffpool_assignment_rows_sum_to_one() {
        let dp = DiffPool::new(4, 2);
        let g = triangle_graph();
        let s = dp.assignment(&g.node_features);
        for row in &s {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_diffpool_pool_output_shapes() {
        let dp = DiffPool::new(4, 2);
        let g = triangle_graph();
        let (x_new, a_new) = dp.pool(&g.node_features, &g.adj).expect("pool ok");
        assert_eq!(x_new.len(), 2); // num_output_nodes
        assert_eq!(x_new[0].len(), 4); // feat_dim
        assert_eq!(a_new.len(), 2);
        assert_eq!(a_new[0].len(), 2);
    }

    #[test]
    fn test_diffpool_entropy_loss_nonneg() {
        let dp = DiffPool::new(4, 2);
        let g = triangle_graph();
        let s = dp.assignment(&g.node_features);
        let loss = dp.entropy_loss(&s);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_diffpool_link_prediction_loss_nonneg() {
        let dp = DiffPool::new(4, 2);
        let g = triangle_graph();
        let s = dp.assignment(&g.node_features);
        let loss = dp.link_prediction_loss(&g.adj, &s);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_diffpool_pool_error_empty() {
        let dp = DiffPool::new(4, 2);
        let result = dp.pool(&[], &[]);
        assert!(matches!(result, Err(GnnError::EmptyGraph)));
    }

    // ── MinCutPool ───────────────────────────────────────────────────────────

    #[test]
    fn test_mincutpool_assignment_shape() {
        let mcp = MinCutPool::new(4, 2);
        let g = triangle_graph();
        let s = mcp.assignment(&g.node_features);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].len(), 2);
    }

    #[test]
    fn test_mincutpool_pool_output_shapes() {
        let mcp = MinCutPool::new(4, 2);
        let g = triangle_graph();
        let (x_new, a_new) = mcp.pool(&g.node_features, &g.adj).expect("pool ok");
        assert_eq!(x_new.len(), 2);
        assert_eq!(x_new[0].len(), 4);
        assert_eq!(a_new.len(), 2);
        assert_eq!(a_new[0].len(), 2);
    }

    #[test]
    fn test_mincutpool_orthogonality_loss_finite() {
        let mcp = MinCutPool::new(4, 2);
        let g = triangle_graph();
        let s = mcp.assignment(&g.node_features);
        let loss = mcp.orthogonality_loss(&s);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_mincutpool_error_k_gt_n() {
        let mcp = MinCutPool::new(4, 10); // 10 clusters > 3 nodes
        let g = triangle_graph();
        let result = mcp.pool(&g.node_features, &g.adj);
        assert!(matches!(result, Err(GnnError::InvalidNumClusters { .. })));
    }

    // ── GATv2 ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_gatv2_forward_output_dim() {
        let layer = Gatv2Layer::new(4, 8, 1);
        let g = triangle_graph();
        let out = layer.forward(&g.node_features, &g.adj).expect("forward ok");
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_gatv2_attention_scores_shape() {
        let layer = Gatv2Layer::new(4, 8, 1);
        let g = triangle_graph();
        let alpha = layer.attention_scores(&g.node_features, &g.adj);
        assert_eq!(alpha.len(), 3);
        assert_eq!(alpha[0].len(), 3);
    }

    #[test]
    fn test_gatv2_attention_scores_nonneg() {
        let layer = Gatv2Layer::new(4, 8, 1);
        let g = triangle_graph();
        let alpha = layer.attention_scores(&g.node_features, &g.adj);
        for row in &alpha {
            for &v in row.iter() {
                assert!(v >= 0.0);
            }
        }
    }

    #[test]
    fn test_gatv2_forward_error_empty() {
        let layer = Gatv2Layer::new(4, 8, 1);
        let result = layer.forward(&[], &[]);
        assert!(matches!(result, Err(GnnError::EmptyGraph)));
    }

    // ── GIN ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_gin_forward_output_dim() {
        let layer = GinLayer::new(4, 16, 0.0);
        let g = triangle_graph();
        let out = layer.forward(&g.node_features, &g.adj).expect("forward ok");
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 16);
    }

    #[test]
    fn test_gin_forward_relu_activation() {
        let layer = GinLayer::new(4, 8, 0.0);
        let g = triangle_graph();
        let out = layer.forward(&g.node_features, &g.adj).expect("forward ok");
        for row in &out {
            for &v in row.iter() {
                assert!(v >= 0.0, "GIN output should be non-negative after ReLU");
            }
        }
    }

    #[test]
    fn test_gin_forward_error_empty() {
        let layer = GinLayer::new(4, 8, 0.0);
        let result = layer.forward(&[], &[]);
        assert!(matches!(result, Err(GnnError::EmptyGraph)));
    }

    // ── Error Display ──────────────────────────────────────────────────────

    #[test]
    fn test_gnnerror_display_empty() {
        let e = GnnError::EmptyGraph;
        assert!(e.to_string().contains("no nodes"));
    }

    #[test]
    fn test_gnnerror_display_dimension_mismatch() {
        let e = GnnError::DimensionMismatch { expected: 4, found: 3 };
        let s = e.to_string();
        assert!(s.contains("4") && s.contains("3"));
    }

    #[test]
    fn test_gnnerror_display_adj_not_square() {
        let e = GnnError::AdjacencyNotSquare { rows: 3, cols: 4 };
        let s = e.to_string();
        assert!(s.contains("3") && s.contains("4"));
    }

    #[test]
    fn test_gnnerror_display_feature_count_mismatch() {
        let e = GnnError::FeatureCountMismatch { adj_size: 5, feat_count: 3 };
        let s = e.to_string();
        assert!(s.contains("5") && s.contains("3"));
    }

    #[test]
    fn test_gnnerror_display_invalid_clusters() {
        let e = GnnError::InvalidNumClusters { k: 10, n: 3 };
        let s = e.to_string();
        assert!(s.contains("10") && s.contains("3"));
    }

    // ── Misc correctness ───────────────────────────────────────────────────

    #[test]
    fn test_global_max_pool_single_node() {
        let feats = vec![vec![3.0_f32, -1.0, 0.5]];
        let out = global_max_pool(&feats);
        assert_eq!(out, vec![3.0, -1.0, 0.5]);
    }

    #[test]
    fn test_diffpool_self_loop_graph() {
        // Graph with self-loops — should still work.
        let feats = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
        let adj = vec![vec![1.0_f32, 1.0], vec![1.0_f32, 1.0]];
        let g = Graph::new(feats, adj).expect("valid");
        let dp = DiffPool::new(2, 1);
        let (x_new, a_new) = dp.pool(&g.node_features, &g.adj).expect("pool ok");
        assert_eq!(x_new.len(), 1);
        assert_eq!(a_new.len(), 1);
    }
}
