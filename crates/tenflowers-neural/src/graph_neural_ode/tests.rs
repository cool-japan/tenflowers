//! Tests for graph_neural_ode module (original + continuous extensions).

use super::*;

fn sg() -> GnoGraph {
    let mut g = GnoGraph::new(4);
    g.add_edge(0, 1, 1.0).expect("test: operation should succeed");
    g.add_edge(1, 2, 1.0).expect("test: operation should succeed");
    g.add_edge(2, 3, 1.0).expect("test: operation should succeed");
    g.add_edge(0, 3, 0.5).expect("test: operation should succeed");
    g
}
fn f4() -> Vec<Vec<f64>> {
    vec![
        vec![1., 0., 0.],
        vec![0., 1., 0.],
        vec![0., 0., 1.],
        vec![1., 1., 0.],
    ]
}

// §1 GnoGraph
#[test]
fn t_graph_new() {
    assert_eq!(GnoGraph::new(5).num_nodes(), 5);
}
#[test]
fn t_graph_edge_err() {
    let mut g = GnoGraph::new(3);
    assert!(g.add_edge(0, 1, 1.).is_ok());
    assert!(g.add_edge(0, 5, 1.).is_err());
}
#[test]
fn t_graph_degree() {
    let mut g = GnoGraph::new(3);
    g.add_edge(0, 1, 2.).expect("test: operation should succeed");
    g.add_edge(0, 2, 3.).expect("test: operation should succeed");
    assert!((g.degree(0) - 5.).abs() < 1e-10);
}
#[test]
fn t_graph_adj_nonneg() {
    let g = sg();
    let a = g.normalize_adjacency();
    for i in 0..4 {
        for j in 0..4 {
            assert!(a[i][j] >= 0.);
        }
        assert!(a[i][i] > 0.);
    }
}
#[test]
fn t_graph_lap_rowsum() {
    let g = sg();
    let lap = g.laplacian();
    for i in 0..4 {
        let s: f64 = lap[i].iter().sum();
        assert!(s.abs() < 1e-10, "row {i}={s}");
    }
}
#[test]
fn t_graph_empty() {
    assert_eq!(GnoGraph::new(0).num_nodes(), 0);
}

// §2 GnoMessagePassing
#[test]
fn t_mp_agg_shape() {
    let g = sg();
    let a = g.normalize_adjacency();
    let o = GnoMessagePassing::new().aggregate(&f4(), &a).expect("test: operation should succeed");
    assert_eq!(o.len(), 4);
    assert_eq!(o[0].len(), 3);
}
#[test]
fn t_mp_empty_err() {
    let res = GnoMessagePassing::new().aggregate(&[], &[]);
    assert!(res.is_err());
}

// §3 GnoGcnLayer
#[test]
fn t_gcn_shape() {
    let l = GnoGcnLayer::new(3, 8, 42).expect("test: operation should succeed");
    assert_eq!(l.weight[0].len(), 8);
}
#[test]
fn t_gcn_relu() {
    let g = sg();
    let a = g.normalize_adjacency();
    let o = GnoGcnLayer::new(3, 5, 42)
        .expect("test: operation should succeed")
        .forward(&f4(), &a)
        .expect("test: operation should succeed");
    assert!(o.iter().flat_map(|r| r.iter()).all(|v| *v >= 0.));
}
#[test]
fn t_gcn_linear() {
    let g = sg();
    let a = g.normalize_adjacency();
    assert_eq!(
        GnoGcnLayer::new(3, 4, 7)
            .expect("test: operation should succeed")
            .forward_linear(&f4(), &a)
            .expect("test: operation should succeed")
            .len(),
        4
    );
}
#[test]
fn t_gcn_invalid() {
    assert!(GnoGcnLayer::new(0, 5, 0).is_err());
}

// §4 GnoGraphODE
#[test]
fn t_gno_ode_ok() {
    let g = sg();
    assert!(GnoGraphODE::new(3, 5, g.normalize_adjacency(), 42).is_ok());
}
#[test]
fn t_gno_ode_dynamics_shape() {
    let g = sg();
    let ode = GnoGraphODE::new(3, 5, g.normalize_adjacency(), 42).expect("test: operation should succeed");
    let dh = ode.dynamics(&f4(), 0.).expect("test: operation should succeed");
    assert_eq!(dh.len(), 4);
    assert_eq!(dh[0].len(), 5);
}

// §5 GraphOdeSolver
#[test]
fn t_solver_euler_step() {
    let g = sg();
    let ode = GnoGraphODE::new(3, 3, g.normalize_adjacency(), 42).expect("test: operation should succeed");
    let h = GraphOdeSolver::new(GnoOdeMethod::Euler)
        .step(&ode, &f4(), 0., 0.1)
        .expect("test: operation should succeed");
    assert_eq!(h[0].len(), 3);
}
#[test]
fn t_solver_rk4_step() {
    let g = sg();
    let ode = GnoGraphODE::new(3, 3, g.normalize_adjacency(), 42).expect("test: operation should succeed");
    let h = GraphOdeSolver::new(GnoOdeMethod::Rk4)
        .step(&ode, &f4(), 0., 0.1)
        .expect("test: operation should succeed");
    assert_eq!(h.len(), 4);
}
#[test]
fn t_solver_trajectory() {
    let g = sg();
    let ode = GnoGraphODE::new(3, 3, g.normalize_adjacency(), 42).expect("test: operation should succeed");
    let traj = GraphOdeSolver::new(GnoOdeMethod::Euler)
        .solve(&ode, f4(), 0., 0.5, 0.1)
        .expect("test: operation should succeed");
    assert_eq!(traj.len(), 6);
}
#[test]
fn t_solver_bad_span() {
    let g = sg();
    let ode = GnoGraphODE::new(3, 3, g.normalize_adjacency(), 42).expect("test: operation should succeed");
    assert!(GraphOdeSolver::new(GnoOdeMethod::Euler)
        .solve(&ode, f4(), 1., 0.5, 0.1)
        .is_err());
}

// §6 GrandModel
#[test]
fn t_grand_ok() {
    assert!(GrandModel::new(GrandConfig {
        num_nodes: 4,
        feat_dim: 3,
        hidden_dim: 8,
        num_classes: 3,
        t_end: 1.,
        dt: 0.5,
        seed: 42
    })
    .is_ok());
}
#[test]
fn t_grand_forward_shape() {
    let m = GrandModel::new(GrandConfig {
        num_nodes: 4,
        feat_dim: 3,
        hidden_dim: 8,
        num_classes: 3,
        t_end: 0.5,
        dt: 0.25,
        seed: 42,
    })
    .expect("test: operation should succeed");
    let l = m.forward(&f4()).expect("test: operation should succeed");
    assert_eq!(l.len(), 4);
    assert_eq!(l[0].len(), 3);
}
#[test]
fn t_grand_predict_bounds() {
    let m = GrandModel::new(GrandConfig {
        num_nodes: 4,
        feat_dim: 3,
        hidden_dim: 6,
        num_classes: 2,
        t_end: 0.5,
        dt: 0.25,
        seed: 7,
    })
    .expect("test: operation should succeed");
    for p in m.predict(&f4()).expect("test: operation should succeed") {
        assert!(p < 2);
    }
}
#[test]
fn t_grand_invalid_classes() {
    assert!(GrandModel::new(GrandConfig {
        num_nodes: 4,
        feat_dim: 3,
        hidden_dim: 6,
        num_classes: 1,
        t_end: 0.5,
        dt: 0.25,
        seed: 7
    })
    .is_err());
}

// §7 CgnnModel
#[test]
fn t_cgnn_ok() {
    let g = sg();
    assert!(CgnnModel::new(
        CgnnConfig {
            feat_dim: 3,
            hidden_dim: 8,
            num_layers: 2,
            t_end: 0.5,
            dt: 0.25,
            seed: 42
        },
        &g
    )
    .is_ok());
}
#[test]
fn t_cgnn_forward() {
    let g = sg();
    let m = CgnnModel::new(
        CgnnConfig {
            feat_dim: 3,
            hidden_dim: 6,
            num_layers: 2,
            t_end: 0.5,
            dt: 0.25,
            seed: 42,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let o = m.forward(&f4()).expect("test: operation should succeed");
    assert_eq!(o.len(), 4);
    assert_eq!(o[0].len(), 3);
}
#[test]
fn t_cgnn_single_layer() {
    let g = sg();
    assert_eq!(
        CgnnModel::new(
            CgnnConfig {
                feat_dim: 3,
                hidden_dim: 5,
                num_layers: 1,
                t_end: 0.2,
                dt: 0.1,
                seed: 3
            },
            &g
        )
        .expect("test: operation should succeed")
        .forward(&f4())
        .expect("test: operation should succeed")
        .len(),
        4
    );
}

// §8 GraphOdeModel
#[test]
fn t_gode_ok() {
    let g = sg();
    assert!(GraphOdeModel::new(
        GraphOdeConfig {
            feat_dim: 3,
            latent_dim: 6,
            num_classes: 3,
            t_end: 0.5,
            dt: 0.25,
            seed: 42
        },
        &g
    )
    .is_ok());
}
#[test]
fn t_gode_forward_shape() {
    let g = sg();
    let m = GraphOdeModel::new(
        GraphOdeConfig {
            feat_dim: 3,
            latent_dim: 6,
            num_classes: 3,
            t_end: 0.5,
            dt: 0.25,
            seed: 42,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let l = m.forward(&f4()).expect("test: operation should succeed");
    assert_eq!(l.len(), 4);
    assert_eq!(l[0].len(), 3);
}
#[test]
fn t_gode_predict_bounds() {
    let g = sg();
    let m = GraphOdeModel::new(
        GraphOdeConfig {
            feat_dim: 3,
            latent_dim: 4,
            num_classes: 2,
            t_end: 0.5,
            dt: 0.25,
            seed: 42,
        },
        &g,
    )
    .expect("test: operation should succeed");
    for p in m.predict(&f4()).expect("test: operation should succeed") {
        assert!(p < 2);
    }
}
#[test]
fn t_gode_euler_interp() {
    let g = sg();
    let m = GraphOdeModel::new(
        GraphOdeConfig {
            feat_dim: 3,
            latent_dim: 4,
            num_classes: 2,
            t_end: 1.,
            dt: 0.25,
            seed: 42,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let z0 = m.encoder.forward(&f4(), &m.a_hat).expect("test: operation should succeed");
    assert_eq!(
        m.forward_from_latent(&z0, 0.5, GnoOdeMethod::Euler)
            .expect("test: operation should succeed")
            .len(),
        4
    );
}
#[test]
fn t_gode_invalid() {
    let g = sg();
    assert!(GraphOdeModel::new(
        GraphOdeConfig {
            feat_dim: 3,
            latent_dim: 4,
            num_classes: 1,
            t_end: 0.5,
            dt: 0.25,
            seed: 42
        },
        &g
    )
    .is_err());
}

// §9 StGnnOde
#[test]
fn t_stgnn_ok() {
    let g = sg();
    assert!(StGnnOde::new(
        StGnnOdeConfig {
            num_nodes: 4,
            feat_dim: 3,
            spatial_hidden: 8,
            temporal_hidden: 6,
            t_end: 0.5,
            dt: 0.25,
            horizon: 3,
            seed: 42
        },
        &g
    )
    .is_ok());
}
#[test]
fn t_stgnn_forecast_shape() {
    let g = sg();
    let m = StGnnOde::new(
        StGnnOdeConfig {
            num_nodes: 4,
            feat_dim: 3,
            spatial_hidden: 6,
            temporal_hidden: 4,
            t_end: 0.5,
            dt: 0.25,
            horizon: 2,
            seed: 42,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let o = m.forward(&f4()).expect("test: operation should succeed");
    assert_eq!(o.len(), 4);
    assert_eq!(o[0].len(), 2);
}
#[test]
fn t_stgnn_horizon1() {
    let g = sg();
    let m = StGnnOde::new(
        StGnnOdeConfig {
            num_nodes: 4,
            feat_dim: 3,
            spatial_hidden: 4,
            temporal_hidden: 3,
            t_end: 0.1,
            dt: 0.1,
            horizon: 1,
            seed: 3,
        },
        &g,
    )
    .expect("test: operation should succeed");
    assert_eq!(m.forward(&f4()).expect("test: operation should succeed")[0].len(), 1);
}

// §10 LatentGraphOde
#[test]
fn t_lgode_ok() {
    let g = sg();
    assert!(LatentGraphOde::new(
        LatentGraphOdeConfig {
            feat_dim: 3,
            latent_dim: 6,
            hidden_dim: 8,
            num_classes: 3,
            t_end: 0.5,
            dt: 0.25,
            seed: 42
        },
        &g
    )
    .is_ok());
}
#[test]
fn t_lgode_forward() {
    let g = sg();
    let m = LatentGraphOde::new(
        LatentGraphOdeConfig {
            feat_dim: 3,
            latent_dim: 4,
            hidden_dim: 6,
            num_classes: 3,
            t_end: 0.5,
            dt: 0.25,
            seed: 42,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let (l, kl) = m.forward(&f4(), 42).expect("test: operation should succeed");
    assert_eq!(l[0].len(), 3);
    assert!(kl.is_finite());
}
#[test]
fn t_lgode_kl_nonneg() {
    let g = sg();
    let m = LatentGraphOde::new(
        LatentGraphOdeConfig {
            feat_dim: 3,
            latent_dim: 4,
            hidden_dim: 6,
            num_classes: 2,
            t_end: 0.2,
            dt: 0.1,
            seed: 11,
        },
        &g,
    )
    .expect("test: operation should succeed");
    assert!(m.forward(&f4(), 0).expect("test: operation should succeed").1 >= 0.);
}
#[test]
fn t_lgode_predict_bounds() {
    let g = sg();
    let m = LatentGraphOde::new(
        LatentGraphOdeConfig {
            feat_dim: 3,
            latent_dim: 4,
            hidden_dim: 6,
            num_classes: 2,
            t_end: 0.2,
            dt: 0.1,
            seed: 11,
        },
        &g,
    )
    .expect("test: operation should succeed");
    for p in m.predict(&f4()).expect("test: operation should succeed") {
        assert!(p < 2);
    }
}
#[test]
fn t_lgode_invalid() {
    let g = sg();
    assert!(LatentGraphOde::new(
        LatentGraphOdeConfig {
            feat_dim: 3,
            latent_dim: 4,
            hidden_dim: 6,
            num_classes: 1,
            t_end: 0.2,
            dt: 0.1,
            seed: 11
        },
        &g
    )
    .is_err());
}

// §11 GnoGatLayer
#[test]
fn t_gat_ok() {
    let l = GnoGatLayer::new(3, 4, 2, 42).expect("test: operation should succeed");
    assert_eq!(l.num_heads, 2);
    assert_eq!(l.head_weights.len(), 2);
}
#[test]
fn t_gat_forward_concat() {
    let g = sg();
    let o = GnoGatLayer::new(3, 4, 2, 42)
        .expect("test: operation should succeed")
        .forward(&f4(), &g)
        .expect("test: operation should succeed");
    assert_eq!(o[0].len(), 8);
}
#[test]
fn t_gat_single_head() {
    let g = sg();
    assert_eq!(
        GnoGatLayer::new(3, 5, 1, 99)
            .expect("test: operation should succeed")
            .forward(&f4(), &g)
            .expect("test: operation should succeed")[0]
            .len(),
        5
    );
}
#[test]
fn t_gat_invalid() {
    assert!(GnoGatLayer::new(0, 4, 2, 0).is_err());
    assert!(GnoGatLayer::new(3, 0, 2, 0).is_err());
    assert!(GnoGatLayer::new(3, 4, 0, 0).is_err());
}

// §12 GnoNodeClassifier
#[test]
fn t_clf_grand_ok() {
    assert!(GnoNodeClassifier::from_grand(GrandConfig {
        num_nodes: 4,
        feat_dim: 3,
        hidden_dim: 6,
        num_classes: 2,
        t_end: 0.5,
        dt: 0.25,
        seed: 42
    })
    .is_ok());
}
#[test]
fn t_clf_gode_ok() {
    let g = sg();
    assert!(GnoNodeClassifier::from_graph_ode(
        GraphOdeConfig {
            feat_dim: 3,
            latent_dim: 4,
            num_classes: 2,
            t_end: 0.5,
            dt: 0.25,
            seed: 42
        },
        &g
    )
    .is_ok());
}
#[test]
fn t_clf_predict_grand() {
    let clf = GnoNodeClassifier::from_grand(GrandConfig {
        num_nodes: 4,
        feat_dim: 3,
        hidden_dim: 6,
        num_classes: 3,
        t_end: 0.5,
        dt: 0.25,
        seed: 42,
    })
    .expect("test: operation should succeed");
    assert_eq!(clf.predict(&f4()).expect("test: operation should succeed").len(), 4);
}
#[test]
fn t_clf_predict_gode() {
    let g = sg();
    let clf = GnoNodeClassifier::from_graph_ode(
        GraphOdeConfig {
            feat_dim: 3,
            latent_dim: 4,
            num_classes: 3,
            t_end: 0.5,
            dt: 0.25,
            seed: 42,
        },
        &g,
    )
    .expect("test: operation should succeed");
    assert_eq!(clf.predict(&f4()).expect("test: operation should succeed").len(), 4);
}
#[test]
fn t_clf_accuracy_range() {
    let clf = GnoNodeClassifier::from_grand(GrandConfig {
        num_nodes: 4,
        feat_dim: 3,
        hidden_dim: 6,
        num_classes: 3,
        t_end: 0.5,
        dt: 0.25,
        seed: 42,
    })
    .expect("test: operation should succeed");
    let a = clf.accuracy(&f4(), &[0, 1, 2, 0]).expect("test: operation should succeed");
    assert!((0. ..=1.).contains(&a));
}

// §13 GraphOdeMetrics
#[test]
fn t_mae() {
    let mae = GraphOdeMetrics::mae_at_t(
        &[vec![1., 2.], vec![3., 4.]],
        &[vec![1.5, 2.5], vec![3.5, 4.5]],
    )
    .expect("test: operation should succeed");
    assert!((mae - 0.5).abs() < 1e-10);
}
#[test]
fn t_mae_perfect() {
    let h = vec![vec![1., 2.]];
    assert!(GraphOdeMetrics::mae_at_t(&h, &h).expect("test: operation should succeed").abs() < 1e-10);
}
#[test]
fn t_r2_perfect() {
    let h = vec![vec![1., 2.], vec![3., 4.]];
    assert!((GraphOdeMetrics::r2_score(&h, &h).expect("test: operation should succeed") - 1.).abs() < 1e-10);
}
#[test]
fn t_r2_zero() {
    assert!(
        (GraphOdeMetrics::r2_score(&[vec![2.], vec![2.]], &[vec![1.], vec![3.]]).expect("test: operation should succeed"))
            .abs()
            < 1e-8
    );
}
#[test]
fn t_node_acc() {
    let acc =
        GraphOdeMetrics::node_classification_accuracy(&[0, 1, 2, 0], &[0, 1, 1, 0]).expect("test: operation should succeed");
    assert!((acc - 0.75).abs() < 1e-10);
}
#[test]
fn t_node_acc_perfect() {
    let l = vec![0, 1, 2, 1, 0];
    assert!(
        (GraphOdeMetrics::node_classification_accuracy(&l, &l).expect("test: operation should succeed") - 1.).abs() < 1e-10
    );
}
#[test]
fn t_rte_zero() {
    let s = vec![vec![vec![1., 2.], vec![3., 4.]]];
    assert!(
        GraphOdeMetrics::relative_trajectory_error(&s, &s)
            .expect("test: operation should succeed")
            .abs()
            < 1e-10
    );
}
#[test]
fn t_rte_nonzero() {
    let p = vec![vec![vec![2.], vec![2.]]];
    let t = vec![vec![vec![1.], vec![1.]]];
    assert!(GraphOdeMetrics::relative_trajectory_error(&p, &t).expect("test: operation should succeed") > 0.);
}
#[test]
fn t_metrics_dim_err() {
    let a = vec![vec![1.]];
    let b = vec![vec![1.], vec![2.]];
    assert!(GraphOdeMetrics::mae_at_t(&a, &b).is_err());
    assert!(GraphOdeMetrics::r2_score(&a, &b).is_err());
}

// Utilities
#[test]
fn t_box_muller_finite() {
    assert!(box_muller(0.5, 0.5).is_finite());
}
#[test]
fn t_box_muller_denorm() {
    assert!(box_muller(1e-30, 0.3).is_finite());
}
#[test]
fn t_matmul_2x2() {
    let a = vec![vec![1., 2.], vec![3., 4.]];
    let b = vec![vec![5., 6.], vec![7., 8.]];
    let c = matmul(&a, &b).expect("test: operation should succeed");
    assert!((c[0][0] - 19.).abs() < 1e-10);
    assert!((c[1][1] - 50.).abs() < 1e-10);
}
#[test]
fn t_matmul_err() {
    assert!(matmul(&[vec![1., 2.]], &[vec![1.], vec![2.], vec![3.]]).is_err());
}

// Large graph end-to-end
#[test]
fn t_large_graph_finite() {
    let n = 10;
    let mut g = GnoGraph::new(n);
    for i in 0..n - 1 {
        g.add_edge(i, i + 1, 1.).expect("test: operation should succeed");
    }
    let m = GraphOdeModel::new(
        GraphOdeConfig {
            feat_dim: 4,
            latent_dim: 8,
            num_classes: 3,
            t_end: 0.5,
            dt: 0.25,
            seed: 77,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let feats: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 * 0.1; 4]).collect();
    assert!(m
        .forward(&feats)
        .expect("test: operation should succeed")
        .iter()
        .flat_map(|r| r.iter())
        .all(|v| v.is_finite()));
}
#[test]
fn t_stgnn_large() {
    let n = 6;
    let mut g = GnoGraph::new(n);
    for i in 0..n - 1 {
        g.add_edge(i, i + 1, 1.).expect("test: operation should succeed");
    }
    let m = StGnnOde::new(
        StGnnOdeConfig {
            num_nodes: n,
            feat_dim: 4,
            spatial_hidden: 6,
            temporal_hidden: 4,
            t_end: 0.3,
            dt: 0.1,
            horizon: 3,
            seed: 13,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let o = m
        .forward(&(0..n).map(|_| vec![0.5f64; 4]).collect::<Vec<_>>())
        .expect("test: operation should succeed");
    assert_eq!(o.len(), n);
    assert_eq!(o[0].len(), 3);
}

// ─── Continuous extensions tests ────────────────────────────────────────────

#[test]
fn t_temporal_node_embedding_encode() {
    let emb = TemporalNodeEmbedding::new(3, 8, 4, 42)
        .expect("test: operation should succeed");
    let feats = f4();
    let out = emb.encode(&feats, 0.5).expect("test: operation should succeed");
    assert_eq!(out.len(), 4);
    assert_eq!(out[0].len(), 8);
}

#[test]
fn t_tgode_model_forward() {
    let g = sg();
    let m = TgodeModel::new(
        TgodeConfig {
            feat_dim: 3,
            hidden_dim: 6,
            time_enc_dim: 4,
            t_end: 0.4,
            dt: 0.2,
            seed: 42,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let out = m.forward(&f4(), 0.0).expect("test: operation should succeed");
    assert_eq!(out.len(), 4);
    assert_eq!(out[0].len(), 6);
}

#[test]
fn t_event_based_ode_integrate() {
    let g = sg();
    let m = EventBasedOde::new(
        EventOdeConfig {
            feat_dim: 3,
            hidden_dim: 5,
            seed: 7,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let events = vec![0.1, 0.3, 0.5];
    let out = m.integrate(&f4(), &events).expect("test: operation should succeed");
    assert_eq!(out.len(), 4);
    assert!(out.iter().flat_map(|r| r.iter()).all(|v| v.is_finite()));
}

#[test]
fn t_reaction_diffusion_gnn_forward() {
    let g = sg();
    let m = ReactionDiffusionGnn::new(
        RdGnnConfig {
            feat_dim: 3,
            hidden_dim: 6,
            t_end: 0.3,
            dt: 0.1,
            diffusion_coeff: 0.5,
            seed: 42,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let out = m.forward(&f4()).expect("test: operation should succeed");
    assert_eq!(out.len(), 4);
    assert!(out.iter().flat_map(|r| r.iter()).all(|v| v.is_finite()));
}

#[test]
fn t_differential_graph_wiring_forward() {
    let g = sg();
    let m = DifferentialGraphWiring::new(
        DgwConfig {
            feat_dim: 3,
            hidden_dim: 4,
            t_end: 0.2,
            dt: 0.1,
            seed: 13,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let out = m.forward(&f4()).expect("test: operation should succeed");
    assert_eq!(out.len(), 4);
}

#[test]
fn t_stochastic_gno_forward() {
    let g = sg();
    let m = StochasticGnoModel::new(
        SgnoConfig {
            feat_dim: 3,
            hidden_dim: 5,
            t_end: 0.2,
            dt: 0.1,
            noise_scale: 0.01,
            seed: 99,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let out = m.forward(&f4(), 42).expect("test: operation should succeed");
    assert_eq!(out.len(), 4);
    assert!(out.iter().flat_map(|r| r.iter()).all(|v| v.is_finite()));
}

#[test]
fn t_latent_stochastic_graph_forward() {
    let g = sg();
    let m = LatentStochasticGraph::new(
        LsgConfig {
            feat_dim: 3,
            latent_dim: 4,
            num_classes: 2,
            t_end: 0.2,
            dt: 0.1,
            noise_scale: 0.01,
            seed: 5,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let (logits, kl) = m.forward(&f4(), 42).expect("test: operation should succeed");
    assert_eq!(logits[0].len(), 2);
    assert!(kl.is_finite());
}

#[test]
fn t_latent_stochastic_graph_kl_nonneg() {
    let g = sg();
    let m = LatentStochasticGraph::new(
        LsgConfig {
            feat_dim: 3,
            latent_dim: 4,
            num_classes: 2,
            t_end: 0.2,
            dt: 0.1,
            noise_scale: 0.005,
            seed: 17,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let (_, kl) = m.forward(&f4(), 0).expect("test: operation should succeed");
    assert!(kl >= 0.0);
}

#[test]
fn t_particle_sim_graph_forward() {
    let n = 4;
    let mut g = GnoGraph::new(n);
    g.add_edge(0, 1, 1.0).expect("test: add_edge");
    g.add_edge(1, 2, 1.0).expect("test: add_edge");
    g.add_edge(2, 3, 1.0).expect("test: add_edge");
    let m = ParticleSimGraph::new(
        PsgConfig {
            n_particles: n,
            state_dim: 6,
            t_end: 0.2,
            dt: 0.1,
            spring_k: 1.0,
            gravity: 0.1,
            seed: 3,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let states = (0..n).map(|i| vec![i as f64 * 0.1; 6]).collect::<Vec<_>>();
    let out = m.forward(&states).expect("test: operation should succeed");
    assert_eq!(out.len(), n);
    assert!(out.iter().flat_map(|r| r.iter()).all(|v| v.is_finite()));
}

#[test]
fn t_hamiltonian_gnn_energy_conserving() {
    let g = sg();
    let m = HamiltonianGnn::new(
        HgnConfig {
            state_dim: 4,
            hidden_dim: 6,
            t_end: 0.4,
            dt: 0.2,
            seed: 11,
        },
        &g,
    )
    .expect("test: operation should succeed");
    // q and p both dim 2 per node (state_dim=4 → q_dim=2, p_dim=2)
    let states = vec![
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.5, 0.6, 0.7, 0.8],
        vec![0.9, 0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6, 0.7],
    ];
    let out = m.forward(&states).expect("test: operation should succeed");
    assert_eq!(out.len(), 4);
    assert_eq!(out[0].len(), 4);
}

#[test]
fn t_lagrangian_gnn_forward() {
    let g = sg();
    let m = LagrangianGnn::new(
        LgnConfig {
            q_dim: 3,
            hidden_dim: 6,
            t_end: 0.2,
            dt: 0.1,
            seed: 77,
        },
        &g,
    )
    .expect("test: operation should succeed");
    let q = f4();
    let q_dot = f4();
    let out = m.forward(&q, &q_dot).expect("test: operation should succeed");
    assert_eq!(out.len(), 4);
    assert!(out.iter().flat_map(|r| r.iter()).all(|v| v.is_finite()));
}

#[test]
fn t_gno_sim_metrics_mse_zero() {
    let traj = vec![f4(), f4()];
    let mse = GnoSimMetrics::trajectory_mse(&traj, &traj).expect("test: mse");
    assert!(mse.abs() < 1e-12);
}

#[test]
fn t_gno_sim_metrics_mse_nonzero() {
    let a = vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]];
    let b = vec![vec![vec![2.0, 3.0], vec![4.0, 5.0]]];
    let mse = GnoSimMetrics::trajectory_mse(&a, &b).expect("test: mse");
    assert!(mse > 0.0);
}

#[test]
fn t_gno_sim_metrics_rollout_stability() {
    let traj = vec![
        vec![vec![1.0; 3]; 4],
        vec![vec![1.0; 3]; 4],
        vec![vec![1.0; 3]; 4],
    ];
    let stable = GnoSimMetrics::rollout_stability(&traj, 2.0);
    assert!(stable);
}

#[test]
fn t_gno_sim_metrics_rollout_unstable() {
    let traj = vec![
        vec![vec![1.0; 3]; 4],
        vec![vec![100.0; 3]; 4],
    ];
    let stable = GnoSimMetrics::rollout_stability(&traj, 2.0);
    assert!(!stable);
}

#[test]
fn t_temporal_node_embedding_time_varies() {
    let emb = TemporalNodeEmbedding::new(3, 8, 4, 10).expect("test");
    let feats = f4();
    let o1 = emb.encode(&feats, 0.0).expect("t1");
    let o2 = emb.encode(&feats, 1.0).expect("t2");
    // Different times should produce different outputs
    let diff: f64 = o1.iter().zip(o2.iter())
        .flat_map(|(r1, r2)| r1.iter().zip(r2.iter()).map(|(a, b)| (a - b).abs()))
        .sum();
    assert!(diff > 1e-10);
}

#[test]
fn t_event_ode_no_events() {
    let g = sg();
    let m = EventBasedOde::new(
        EventOdeConfig { feat_dim: 3, hidden_dim: 5, seed: 7 },
        &g,
    ).expect("test");
    // No events → returns initial state unchanged (no integration steps)
    let out = m.integrate(&f4(), &[]).expect("test");
    assert_eq!(out.len(), 4);
}

#[test]
fn t_reaction_diffusion_stable_coeffs() {
    let g = sg();
    let m = ReactionDiffusionGnn::new(
        RdGnnConfig {
            feat_dim: 3,
            hidden_dim: 4,
            t_end: 0.5,
            dt: 0.1,
            diffusion_coeff: 1.0,
            seed: 88,
        },
        &g,
    ).expect("test");
    let out = m.forward(&f4()).expect("test");
    assert!(out.iter().flat_map(|r| r.iter()).all(|v| v.is_finite()));
}

#[test]
fn t_stochastic_gno_reproducible() {
    let g = sg();
    let m = StochasticGnoModel::new(
        SgnoConfig {
            feat_dim: 3, hidden_dim: 5, t_end: 0.2, dt: 0.1, noise_scale: 0.05, seed: 42,
        },
        &g,
    ).expect("test");
    let o1 = m.forward(&f4(), 999).expect("o1");
    let o2 = m.forward(&f4(), 999).expect("o2");
    let diff: f64 = o1.iter().zip(o2.iter())
        .flat_map(|(r1, r2)| r1.iter().zip(r2.iter()).map(|(a, b)| (a - b).abs()))
        .sum();
    assert!(diff < 1e-12);
}

#[test]
fn t_hamiltonian_gnn_invalid() {
    let g = sg();
    // state_dim must be even (q_dim = p_dim = state_dim/2)
    assert!(HamiltonianGnn::new(
        HgnConfig { state_dim: 3, hidden_dim: 4, t_end: 0.2, dt: 0.1, seed: 0 },
        &g,
    ).is_err());
}

#[test]
fn t_lagrangian_gnn_invalid() {
    let g = sg();
    assert!(LagrangianGnn::new(
        LgnConfig { q_dim: 0, hidden_dim: 4, t_end: 0.2, dt: 0.1, seed: 0 },
        &g,
    ).is_err());
}
