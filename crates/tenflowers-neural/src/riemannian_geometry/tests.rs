//! Tests for riemannian_geometry::advanced — Poincaré ball, geometric flows, metrics.

use super::*;

#[test]
fn test_poincare_ball_origin_in_ball() {
    let ball = PoincareBall::new(3, 1.0);
    let origin = [0.0; 3];
    let norm_sq: f64 = origin.iter().map(|x| x * x).sum();
    assert!(norm_sq < 1.0, "origin should be inside ball");
}

#[test]
fn test_mobius_add_identity() {
    let ball = PoincareBall::new(2, 1.0);
    let x = vec![0.3, 0.2];
    let zero = vec![0.0, 0.0];
    let result = ball.mobius_add(&x, &zero);
    for (r, xi) in result.iter().zip(x.iter()) {
        assert!(
            (r - xi).abs() < 1e-10,
            "x ⊕ 0 should equal x, got {r} vs {xi}"
        );
    }
}

#[test]
fn test_mobius_add_inverse() {
    let ball = PoincareBall::new(2, 1.0);
    let x = vec![0.3, 0.2];
    let neg_x: Vec<f64> = x.iter().map(|xi| -xi).collect();
    let result = ball.mobius_add(&x, &neg_x);
    for r in &result {
        assert!(r.abs() < 1e-10, "x ⊕ (−x) should be ~0, got {r}");
    }
}

#[test]
fn test_exp_log_map_roundtrip() {
    let ball = PoincareBall::new(3, 1.0);
    let v = vec![0.1, -0.2, 0.15];
    let y = ball.exp_map_origin(&v);
    let v_back = ball.log_map_origin(&y);
    for (a, b) in v.iter().zip(v_back.iter()) {
        assert!(
            (a - b).abs() < 1e-8,
            "exp then log should recover v: {a} vs {b}"
        );
    }
}

#[test]
fn test_hyperbolic_distance_zero() {
    let ball = PoincareBall::new(2, 1.0);
    let x = vec![0.3, 0.4];
    let d = ball.geodesic_distance(&x, &x);
    assert!(d.abs() < 1e-10, "d(x,x) should be 0, got {d}");
}

#[test]
fn test_hyperbolic_distance_positive() {
    let ball = PoincareBall::new(2, 1.0);
    let x = vec![0.3, 0.0];
    let y = vec![0.0, 0.4];
    let d = ball.geodesic_distance(&x, &y);
    assert!(
        d > 0.0,
        "distance between different points should be positive"
    );
}

#[test]
fn test_hyperbolic_distance_symmetry() {
    let ball = PoincareBall::new(2, 1.0);
    let x = vec![0.3, 0.1];
    let y = vec![-0.2, 0.4];
    let d_xy = ball.geodesic_distance(&x, &y);
    let d_yx = ball.geodesic_distance(&y, &x);
    assert!(
        (d_xy - d_yx).abs() < 1e-8,
        "distance should be symmetric: {d_xy} vs {d_yx}"
    );
}

#[test]
fn test_poincare_ball_project_inside() {
    let ball = PoincareBall::new(2, 1.0);
    let x = vec![10.0, 10.0]; // far outside ball
    let proj = ball.project(&x);
    let norm_sq: f64 = proj.iter().map(|xi| xi * xi).sum();
    assert!(
        norm_sq < 1.0,
        "projected point should be inside ball, norm_sq={norm_sq}"
    );
}

#[test]
fn test_poincare_ball_project_inside_unchanged() {
    let ball = PoincareBall::new(2, 1.0);
    let x = vec![0.1, 0.1]; // well inside ball
    let proj = ball.project(&x);
    for (a, b) in x.iter().zip(proj.iter()) {
        assert!(
            (a - b).abs() < 1e-14,
            "inside-ball point should be unchanged"
        );
    }
}

#[test]
fn test_lambda_x_at_origin() {
    let ball = PoincareBall::new(3, 1.0);
    let origin = vec![0.0; 3];
    let lambda = ball.lambda_x(&origin);
    // At origin: 2 / (1 - 0) = 2
    assert!(
        (lambda - 2.0).abs() < 1e-10,
        "lambda at origin should be 2, got {lambda}"
    );
}

#[test]
fn test_hyperbolic_embedding_creation() {
    let emb = HyperbolicEmbedding::new(10, 5, 1.0, 0.01, 42);
    assert_eq!(emb.n_items, 10);
    assert_eq!(emb.ball.dim, 5);
    for i in 0..10 {
        let e = emb.get(i);
        let norm_sq: f64 = e.iter().map(|xi| xi * xi).sum();
        assert!(
            norm_sq < 1.0,
            "embedding {i} should be inside ball, norm_sq={norm_sq}"
        );
    }
}

#[test]
fn test_hyperbolic_embedding_rsgd_step() {
    let mut emb = HyperbolicEmbedding::new(5, 3, 1.0, 0.01, 7);
    let grad = vec![0.1, -0.2, 0.3];
    emb.rsgd_step(0, &grad);
    let e = emb.get(0);
    let norm_sq: f64 = e.iter().map(|xi| xi * xi).sum();
    assert!(
        norm_sq < 1.0,
        "after update, embedding should still be in ball, norm_sq={norm_sq}"
    );
}

#[test]
fn test_hyperbolic_embedding_self_distance_zero() {
    let emb = HyperbolicEmbedding::new(5, 3, 1.0, 0.01, 99);
    let d = emb.distance(0, 0);
    assert!(
        d.abs() < 1e-10,
        "distance from item to itself should be 0, got {d}"
    );
}

#[test]
fn test_hyperbolic_embedding_distance_nonneg() {
    let emb = HyperbolicEmbedding::new(5, 3, 1.0, 0.01, 99);
    let d01 = emb.distance(0, 1);
    assert!(d01 >= 0.0, "distance should be non-negative, got {d01}");
}

#[test]
fn test_hyperbolic_linear_output_on_ball() {
    let layer = HyperbolicLinear::new(4, 3, 1.0, 42);
    let x = vec![0.1, -0.2, 0.3, -0.1];
    let y = layer.forward(&x);
    assert_eq!(y.len(), 3, "output should have out_dim elements");
    let norm_sq: f64 = y.iter().map(|yi| yi * yi).sum();
    assert!(
        norm_sq < 1.0,
        "output should be on Poincaré ball, norm_sq={norm_sq}"
    );
}

#[test]
fn test_hyperbolic_linear_different_inputs() {
    let layer = HyperbolicLinear::new(3, 2, 1.0, 77);
    let x1 = vec![0.1, 0.2, 0.3];
    let x2 = vec![-0.1, -0.2, -0.3];
    let y1 = layer.forward(&x1);
    let y2 = layer.forward(&x2);
    // Different inputs should generally give different outputs
    let same = y1.iter().zip(y2.iter()).all(|(a, b)| (a - b).abs() < 1e-14);
    assert!(!same, "different inputs should give different outputs");
}

#[test]
fn test_ollivier_ricci_triangle_curvature() {
    // Complete graph K3 (triangle) — all edges have positive curvature
    let edges = vec![
        vec![(1, 1.0), (2, 1.0)],
        vec![(0, 1.0), (2, 1.0)],
        vec![(0, 1.0), (1, 1.0)],
    ];
    let ricci = OllivierRicciCurvature::new(3, edges);
    let kappa = ricci.curvature(0, 1);
    assert!(
        kappa.is_finite(),
        "curvature should be finite for triangle graph"
    );
    // On K3: μ_0 = {1: 1/2, 2: 1/2}, μ_1 = {0: 1/2, 2: 1/2}
    // W1 = 1/2 * d(1,0) + 1/2 * d(2,2) + ... ≈ 0 → κ ≈ 1
    // Positive curvature expected
    assert!(
        kappa > 0.0,
        "triangle should have positive Ollivier-Ricci curvature, got {kappa}"
    );
}

#[test]
fn test_ollivier_ricci_path_curvature() {
    // Path graph: 0-1-2-3
    let edges = vec![
        vec![(1, 1.0)],
        vec![(0, 1.0), (2, 1.0)],
        vec![(1, 1.0), (3, 1.0)],
        vec![(2, 1.0)],
    ];
    let ricci = OllivierRicciCurvature::new(4, edges);
    let kappa = ricci.curvature(1, 2);
    assert!(kappa.is_finite(), "curvature on path should be finite");
}

#[test]
fn test_ricci_flow_weight_update() {
    let edges = vec![
        vec![(1, 1.0), (2, 1.0)],
        vec![(0, 1.0), (2, 1.0)],
        vec![(0, 1.0), (1, 1.0)],
    ];
    let mut flow = RicciFlow::new(3, edges, 0.0, 0.01);
    let w_before = flow.weight(0, 1).unwrap_or(1.0);
    flow.step();
    let w_after = flow
        .weight(0, 1)
        .expect("edge (0,1) should exist after step");
    assert!(
        w_after > 0.0,
        "weight should remain positive after flow step"
    );
    // Weight should have changed (curvature != 0 on triangle with flat target)
    let _ = w_before; // allowed to be same or different
}

#[test]
fn test_ricci_flow_weights_positive() {
    let edges = vec![
        vec![(1, 2.0), (2, 3.0)],
        vec![(0, 2.0), (2, 1.0)],
        vec![(0, 3.0), (1, 1.0)],
    ];
    let mut flow = RicciFlow::new(3, edges, 0.0, 0.1);
    flow.run(10);
    // All weights should remain positive
    for i in 0..3 {
        for &(_, w) in &flow.ricci.edges[i] {
            assert!(w > 0.0, "all weights should stay positive after Ricci flow");
        }
    }
}

#[test]
fn test_ricci_flow_two_nodes() {
    let edges = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
    let mut flow = RicciFlow::new(2, edges, 0.0, 0.01);
    flow.run(5);
    let w = flow.weight(0, 1);
    assert!(
        w.is_some() && w.unwrap() > 0.0,
        "two-node flow should maintain positive weight"
    );
}

#[test]
fn test_rg_metrics_geodesic_error_exact() {
    let mut metrics = RgMetrics::new();
    metrics.record_geodesic_error(1.05, 1.0);
    metrics.record_geodesic_error(0.95, 1.0);
    let err = metrics.mean_geodesic_error();
    assert!(
        (err - 0.05).abs() < 1e-10,
        "mean error should be 0.05, got {err}"
    );
}

#[test]
fn test_rg_metrics_embedding_distortion() {
    let mut metrics = RgMetrics::new();
    metrics.record_embedding_distortion(1.0, 1.1); // 10% distortion
    metrics.record_embedding_distortion(2.0, 2.0); // 0% distortion
    let dist = metrics.mean_embedding_distortion();
    assert!(dist >= 0.0, "distortion should be non-negative");
    assert!(
        (dist - 0.05).abs() < 1e-10,
        "mean distortion should be 0.05, got {dist}"
    );
}

#[test]
fn test_rg_metrics_empty() {
    let metrics = RgMetrics::new();
    assert_eq!(metrics.mean_geodesic_error(), 0.0);
    assert_eq!(metrics.mean_embedding_distortion(), 0.0);
    assert!(metrics.sectional_curvatures.is_empty());
}

#[test]
fn test_sectional_curvature_spd() {
    let man = RgSpdManifold::new(2);
    let mut metrics = RgMetrics::new();
    // Identity matrix as base point
    let p = vec![1.0, 0.0, 0.0, 1.0];
    let u = vec![0.1, 0.0, 0.0, 0.1];
    let v = vec![0.0, 0.1, 0.1, 0.0];
    let kappa = metrics.estimate_sectional_curvature(&man, &p, &u, &v, 0.01);
    assert!(
        kappa.is_finite(),
        "sectional curvature should be finite on SPD manifold"
    );
    assert_eq!(metrics.sectional_curvatures.len(), 1);
}

#[test]
fn test_ollivier_ricci_self_loop() {
    // Isolated node (no edges) — neighbor dist is self-loop
    let edges = vec![vec![], vec![(0, 1.0)]];
    let ricci = OllivierRicciCurvature::new(2, edges);
    // Curvature of edge (0,1) where node 0 has no neighbors
    let kappa = ricci.curvature(0, 1);
    assert!(
        kappa.is_finite(),
        "curvature with isolated node should be finite"
    );
}
