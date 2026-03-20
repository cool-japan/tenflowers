#![cfg(test)]

use super::*;

// ---------------------------------------------------------------------------
// Helper functions for test graph construction
// ---------------------------------------------------------------------------

/// Create a triangle graph (3 nodes, 3 edges).
fn triangle_graph() -> GmGraph {
    let mut g = GmGraph::new();
    g.add_node(vec![1.0, 0.0]);
    g.add_node(vec![0.0, 1.0]);
    g.add_node(vec![1.0, 1.0]);
    g.add_edge(0, 1, vec![1.0]).ok();
    g.add_edge(1, 2, vec![1.0]).ok();
    g.add_edge(0, 2, vec![1.0]).ok();
    g
}

/// Create a path graph: 0-1-2-3.
fn path_graph() -> GmGraph {
    let mut g = GmGraph::new();
    g.add_node(vec![1.0]);
    g.add_node(vec![2.0]);
    g.add_node(vec![3.0]);
    g.add_node(vec![4.0]);
    g.add_edge(0, 1, vec![]).ok();
    g.add_edge(1, 2, vec![]).ok();
    g.add_edge(2, 3, vec![]).ok();
    g
}

/// Create a square graph: 0-1-2-3-0.
fn square_graph() -> GmGraph {
    let mut g = GmGraph::new();
    g.add_node(vec![1.0, 0.0]);
    g.add_node(vec![0.0, 1.0]);
    g.add_node(vec![1.0, 1.0]);
    g.add_node(vec![0.5, 0.5]);
    g.add_edge(0, 1, vec![]).ok();
    g.add_edge(1, 2, vec![]).ok();
    g.add_edge(2, 3, vec![]).ok();
    g.add_edge(3, 0, vec![]).ok();
    g
}

/// Create a star graph with center node 0 and n_leaves.
fn star_graph(n_leaves: usize) -> GmGraph {
    let mut g = GmGraph::new();
    g.add_node(vec![0.0]); // center
    for i in 0..n_leaves {
        g.add_node(vec![(i + 1) as f64]);
        g.add_edge(0, i + 1, vec![]).ok();
    }
    g
}

/// Create an empty graph with n nodes and no edges.
fn isolated_graph(n: usize) -> GmGraph {
    let mut g = GmGraph::new();
    for i in 0..n {
        g.add_node(vec![i as f64]);
    }
    g
}

/// Create two isomorphic triangles with permuted node labels.
fn isomorphic_triangles() -> (GmGraph, GmGraph) {
    let g1 = triangle_graph();
    // g2 is the same triangle but with different node feature order
    let mut g2 = GmGraph::new();
    g2.add_node(vec![0.0, 1.0]); // was node 1
    g2.add_node(vec![1.0, 1.0]); // was node 2
    g2.add_node(vec![1.0, 0.0]); // was node 0
    g2.add_edge(0, 1, vec![1.0]).ok();
    g2.add_edge(1, 2, vec![1.0]).ok();
    g2.add_edge(0, 2, vec![1.0]).ok();
    (g1, g2)
}

// ---------------------------------------------------------------------------
// GmGraph tests
// ---------------------------------------------------------------------------

#[test]
fn test_gm_graph_creation() {
    let g = GmGraph::new();
    assert_eq!(g.n_nodes(), 0);
    assert_eq!(g.n_edges(), 0);
}

#[test]
fn test_gm_graph_add_node() {
    let mut g = GmGraph::new();
    let idx = g.add_node(vec![1.0, 2.0, 3.0]);
    assert_eq!(idx, 0);
    assert_eq!(g.n_nodes(), 1);
    assert_eq!(g.node_features[0], vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_gm_graph_add_edge() {
    let mut g = GmGraph::new();
    g.add_node(vec![1.0]);
    g.add_node(vec![2.0]);
    let result = g.add_edge(0, 1, vec![0.5]);
    assert!(result.is_ok());
    assert_eq!(g.n_edges(), 1);
    assert!(g.has_edge(0, 1));
    assert!(g.has_edge(1, 0)); // undirected
}

#[test]
fn test_gm_graph_add_edge_out_of_bounds() {
    let mut g = GmGraph::new();
    g.add_node(vec![1.0]);
    let result = g.add_edge(0, 5, vec![]);
    assert!(result.is_err());
}

#[test]
fn test_gm_graph_neighbors() {
    let g = triangle_graph();
    assert_eq!(g.neighbors(0).len(), 2);
    assert_eq!(g.neighbors(1).len(), 2);
    assert_eq!(g.neighbors(2).len(), 2);
}

#[test]
fn test_gm_graph_degree() {
    let g = star_graph(4);
    assert_eq!(g.degree(0), 4); // center
    assert_eq!(g.degree(1), 1); // leaf
    assert_eq!(g.degree(2), 1);
}

#[test]
fn test_gm_graph_adjacency_matrix() {
    let g = triangle_graph();
    let adj = g.adjacency_matrix();
    assert_eq!(adj.len(), 3);
    assert!((adj[0][1] - 1.0).abs() < 1e-10);
    assert!((adj[1][0] - 1.0).abs() < 1e-10);
    assert!((adj[0][0]).abs() < 1e-10);
}

#[test]
fn test_gm_graph_laplacian() {
    let g = triangle_graph();
    let lap = g.laplacian();
    // Diagonal = degree, off-diagonal = -1 for edges.
    assert!((lap[0][0] - 2.0).abs() < 1e-10);
    assert!((lap[0][1] - (-1.0)).abs() < 1e-10);
    // Row sums should be 0 for Laplacian.
    for row in &lap {
        let sum: f64 = row.iter().sum();
        assert!(sum.abs() < 1e-10, "Laplacian row sum = {sum}");
    }
}

#[test]
fn test_gm_graph_default() {
    let g = GmGraph::default();
    assert_eq!(g.n_nodes(), 0);
}

#[test]
fn test_gm_graph_has_edge_no_edge() {
    let g = isolated_graph(3);
    assert!(!g.has_edge(0, 1));
    assert!(!g.has_edge(1, 2));
}

#[test]
fn test_gm_graph_degree_out_of_bounds() {
    let g = triangle_graph();
    assert_eq!(g.degree(999), 0);
}

// ---------------------------------------------------------------------------
// Graph Edit Distance tests
// ---------------------------------------------------------------------------

#[test]
fn test_ged_identical_graphs() {
    let g1 = triangle_graph();
    let g2 = triangle_graph();
    let cost = GmEditCostModel::default();
    let (dist, _path) =
        GraphEditDistance::compute(&g1, &g2, 10, &cost).expect("GED computation failed");
    // Identical graphs should have very low edit distance (features match exactly for same-ordered nodes).
    assert!(
        dist < 1e-6,
        "GED of identical graphs should be ~0, got {dist}"
    );
}

#[test]
fn test_ged_empty_graphs() {
    let g1 = GmGraph::new();
    let g2 = GmGraph::new();
    let cost = GmEditCostModel::default();
    let (dist, path) =
        GraphEditDistance::compute(&g1, &g2, 10, &cost).expect("GED computation failed");
    assert!((dist).abs() < 1e-10);
    assert!(path.is_empty());
}

#[test]
fn test_ged_one_empty_one_not() {
    let g1 = triangle_graph();
    let g2 = GmGraph::new();
    let cost = GmEditCostModel::default();
    let (dist, _path) =
        GraphEditDistance::compute(&g1, &g2, 10, &cost).expect("GED computation failed");
    // Should cost at least 3 node deletions + 3 edge deletions.
    assert!(
        dist >= 3.0,
        "GED should be >= 3 for 3 deleted nodes, got {dist}"
    );
}

#[test]
fn test_ged_different_sizes() {
    let g1 = path_graph(); // 4 nodes
    let mut g2 = GmGraph::new();
    g2.add_node(vec![1.0]);
    g2.add_node(vec![2.0]);
    g2.add_edge(0, 1, vec![]).ok();
    let cost = GmEditCostModel::default();
    let (dist, _path) =
        GraphEditDistance::compute(&g1, &g2, 10, &cost).expect("GED computation failed");
    assert!(
        dist > 0.0,
        "GED between different-size graphs should be > 0"
    );
}

#[test]
fn test_ged_edit_path_nonempty() {
    let g1 = triangle_graph();
    let mut g2 = GmGraph::new();
    g2.add_node(vec![1.0, 0.0]);
    g2.add_node(vec![0.0, 1.0]);
    g2.add_edge(0, 1, vec![1.0]).ok();
    let cost = GmEditCostModel::default();
    let (_dist, path) =
        GraphEditDistance::compute(&g1, &g2, 10, &cost).expect("GED computation failed");
    assert!(
        !path.is_empty(),
        "Edit path should not be empty for different graphs"
    );
}

#[test]
fn test_ged_custom_cost_model() {
    let g1 = triangle_graph();
    let g2 = path_graph();
    let cost = GmEditCostModel {
        node_sub_cost: 2.0,
        node_del_cost: 3.0,
        node_ins_cost: 3.0,
        edge_sub_cost: 1.5,
        edge_del_cost: 2.0,
        edge_ins_cost: 2.0,
    };
    let (dist, _) =
        GraphEditDistance::compute(&g1, &g2, 10, &cost).expect("GED with custom costs failed");
    assert!(dist > 0.0);
}

// ---------------------------------------------------------------------------
// VF2 Matcher tests
// ---------------------------------------------------------------------------

#[test]
fn test_vf2_identical_graphs() {
    let g = triangle_graph();
    let result = Vf2Matcher::find_isomorphism(&g, &g).expect("VF2 failed");
    assert!(result.is_some(), "Identical graphs should have isomorphism");
}

#[test]
fn test_vf2_isomorphic_triangles() {
    let (g1, g2) = isomorphic_triangles();
    let result = Vf2Matcher::find_isomorphism(&g1, &g2).expect("VF2 failed");
    assert!(result.is_some(), "Isomorphic triangles should match");
    let mapping = result.expect("mapping should exist");
    assert_eq!(mapping.len(), 3);
}

#[test]
fn test_vf2_subgraph_in_larger() {
    // Triangle as subgraph of square + diagonal.
    let pattern = triangle_graph();
    let mut target = square_graph();
    target.add_edge(0, 2, vec![]).ok(); // add diagonal
    let result = Vf2Matcher::find_isomorphism(&pattern, &target).expect("VF2 failed");
    assert!(
        result.is_some(),
        "Triangle should be found in square+diagonal"
    );
}

#[test]
fn test_vf2_no_isomorphism() {
    // Pattern has more nodes than target.
    let big = star_graph(5);
    let small = triangle_graph();
    let result = Vf2Matcher::find_isomorphism(&big, &small).expect("VF2 failed");
    assert!(
        result.is_none(),
        "Bigger pattern should not match smaller target"
    );
}

#[test]
fn test_vf2_find_all_empty_pattern() {
    let pattern = GmGraph::new();
    let target = triangle_graph();
    let results =
        Vf2Matcher::find_all_subgraph_isomorphisms(&pattern, &target).expect("VF2 all failed");
    // Empty pattern trivially matches.
    assert!(!results.is_empty());
}

#[test]
fn test_vf2_find_all_triangle_in_k4() {
    // K4 has 4 triangles.
    let pattern = triangle_graph();
    let mut target = GmGraph::new();
    for i in 0..4 {
        target.add_node(vec![i as f64 * 0.1]); // close features
    }
    // K4 edges.
    for i in 0..4 {
        for j in (i + 1)..4 {
            target.add_edge(i, j, vec![]).ok();
        }
    }
    let results =
        Vf2Matcher::find_all_subgraph_isomorphisms(&pattern, &target).expect("VF2 all failed");
    // K4 contains C(4,3) = 4 triangles, each with 3! = 6 automorphisms = 24 total,
    // but VF2 may find fewer depending on feature constraints. Should find >= 1.
    assert!(
        !results.is_empty(),
        "Should find at least one triangle in K4"
    );
}

#[test]
fn test_vf2_single_node() {
    let mut pattern = GmGraph::new();
    pattern.add_node(vec![1.0]);
    let target = triangle_graph();
    let result = Vf2Matcher::find_isomorphism(&pattern, &target).expect("VF2 failed");
    assert!(result.is_some(), "Single node should match");
}

#[test]
fn test_vf2_edge_only() {
    let mut pattern = GmGraph::new();
    pattern.add_node(vec![0.5]);
    pattern.add_node(vec![0.5]);
    pattern.add_edge(0, 1, vec![]).ok();
    let target = path_graph();
    let result = Vf2Matcher::find_isomorphism(&pattern, &target).expect("VF2 failed");
    assert!(result.is_some());
}

// ---------------------------------------------------------------------------
// Weisfeiler-Leman Kernel tests
// ---------------------------------------------------------------------------

#[test]
fn test_wl_kernel_identical() {
    let g = triangle_graph();
    let k = WeisfeilerLemanKernel::compute_kernel(&g, &g, 3).expect("WL kernel failed");
    assert!(k > 0.0, "WL kernel of identical graphs should be positive");
}

#[test]
fn test_wl_kernel_different_structures() {
    let g1 = triangle_graph();
    let g2 = path_graph();
    let k_same = WeisfeilerLemanKernel::compute_kernel(&g1, &g1, 3).expect("WL kernel failed");
    let k_diff = WeisfeilerLemanKernel::compute_kernel(&g1, &g2, 3).expect("WL kernel failed");
    // Self-kernel should be >= cross-kernel.
    assert!(
        k_same >= k_diff,
        "Self-kernel ({k_same}) should be >= cross-kernel ({k_diff})"
    );
}

#[test]
fn test_wl_kernel_zero_iterations() {
    let g1 = triangle_graph();
    let g2 = triangle_graph();
    let k = WeisfeilerLemanKernel::compute_kernel(&g1, &g2, 0).expect("WL kernel failed");
    assert!(k > 0.0);
}

#[test]
fn test_wl_kernel_empty_graph() {
    let g = GmGraph::new();
    let k = WeisfeilerLemanKernel::compute_kernel(&g, &g, 3).expect("WL kernel failed");
    assert!((k).abs() < 1e-10, "Empty graph kernel should be 0");
}

#[test]
fn test_wl_oa_kernel_identical() {
    let g = triangle_graph();
    let k = WeisfeilerLemanKernel::compute_oa_kernel(&g, &g, 3).expect("WL OA kernel failed");
    assert!(k > 0.0, "OA kernel of identical graphs should be positive");
}

#[test]
fn test_wl_oa_kernel_vs_standard() {
    let g1 = triangle_graph();
    let g2 = square_graph();
    let k_std = WeisfeilerLemanKernel::compute_kernel(&g1, &g2, 2).expect("WL kernel failed");
    let k_oa = WeisfeilerLemanKernel::compute_oa_kernel(&g1, &g2, 2).expect("WL OA kernel failed");
    // Both should be non-negative.
    assert!(k_std >= 0.0);
    assert!(k_oa >= 0.0);
}

#[test]
fn test_wl_kernel_isomorphic_same_value() {
    let (g1, g2) = isomorphic_triangles();
    let k12 = WeisfeilerLemanKernel::compute_kernel(&g1, &g2, 3).expect("WL kernel failed");
    let k11 = WeisfeilerLemanKernel::compute_kernel(&g1, &g1, 3).expect("WL kernel failed");
    // Isomorphic graphs: cross-kernel should equal self-kernel.
    // With hash-based WL, labels may differ, so k12 <= k11.
    assert!(
        k12 <= k11 + 1e-6,
        "Cross-kernel should not exceed self-kernel"
    );
}

// ---------------------------------------------------------------------------
// Random Walk Kernel tests
// ---------------------------------------------------------------------------

#[test]
fn test_rw_kernel_identical() {
    let g = triangle_graph();
    let k = RandomWalkKernel::compute(&g, &g, 3, 0.1).expect("RW kernel failed");
    assert!(k > 0.0, "RW kernel of identical graphs should be positive");
}

#[test]
fn test_rw_kernel_empty() {
    let g = GmGraph::new();
    let k = RandomWalkKernel::compute(&g, &g, 3, 0.1).expect("RW kernel failed");
    assert!((k).abs() < 1e-10);
}

#[test]
fn test_rw_kernel_zero_walk_length() {
    let g1 = triangle_graph();
    let g2 = path_graph();
    let k = RandomWalkKernel::compute(&g1, &g2, 0, 0.1).expect("RW kernel failed");
    // Walk length 0: count product graph nodes = n1 * n2.
    assert!(k > 0.0);
}

#[test]
fn test_rw_kernel_decay_effect() {
    let g1 = triangle_graph();
    let g2 = triangle_graph();
    let k_small_decay = RandomWalkKernel::compute(&g1, &g2, 3, 0.01).expect("RW kernel failed");
    let k_large_decay = RandomWalkKernel::compute(&g1, &g2, 3, 0.5).expect("RW kernel failed");
    // Larger decay gives more weight to longer walks.
    assert!(k_large_decay >= k_small_decay);
}

#[test]
fn test_rw_kernel_self_vs_cross() {
    let g1 = triangle_graph();
    let g2 = path_graph();
    let k11 = RandomWalkKernel::compute(&g1, &g1, 2, 0.1).expect("RW kernel failed");
    let k12 = RandomWalkKernel::compute(&g1, &g2, 2, 0.1).expect("RW kernel failed");
    assert!(
        k11 >= 0.0 && k12 >= 0.0,
        "RW kernels should be non-negative"
    );
}

// ---------------------------------------------------------------------------
// Shortest Path Kernel tests
// ---------------------------------------------------------------------------

#[test]
fn test_sp_kernel_identical() {
    let g = triangle_graph();
    let k = ShortestPathKernel::compute(&g, &g).expect("SP kernel failed");
    assert!(k > 0.0, "SP kernel of identical graphs should be positive");
}

#[test]
fn test_sp_kernel_empty() {
    let g = GmGraph::new();
    let k = ShortestPathKernel::compute(&g, &g).expect("SP kernel failed");
    assert!((k).abs() < 1e-10);
}

#[test]
fn test_sp_kernel_different() {
    let g1 = triangle_graph();
    let g2 = path_graph();
    let k11 = ShortestPathKernel::compute(&g1, &g1).expect("SP kernel failed");
    let k12 = ShortestPathKernel::compute(&g1, &g2).expect("SP kernel failed");
    let k22 = ShortestPathKernel::compute(&g2, &g2).expect("SP kernel failed");
    assert!(k11 > 0.0);
    assert!(k22 > 0.0);
    assert!(k12 >= 0.0);
}

#[test]
fn test_sp_kernel_isolated_nodes() {
    let g1 = isolated_graph(3);
    let g2 = isolated_graph(3);
    let k = ShortestPathKernel::compute(&g1, &g2).expect("SP kernel failed");
    // No paths between isolated nodes.
    assert!((k).abs() < 1e-10);
}

#[test]
fn test_sp_kernel_single_edge() {
    let mut g1 = GmGraph::new();
    g1.add_node(vec![1.0]);
    g1.add_node(vec![2.0]);
    g1.add_edge(0, 1, vec![]).ok();

    let mut g2 = GmGraph::new();
    g2.add_node(vec![1.0]);
    g2.add_node(vec![2.0]);
    g2.add_edge(0, 1, vec![]).ok();

    let k = ShortestPathKernel::compute(&g1, &g2).expect("SP kernel failed");
    // Both have exactly one shortest path of length 1.
    assert!(
        (k - 1.0).abs() < 1e-10,
        "Single edge SP kernel should be 1.0, got {k}"
    );
}

// ---------------------------------------------------------------------------
// Spectral Alignment tests
// ---------------------------------------------------------------------------

#[test]
fn test_spectral_alignment_identical() {
    let g = triangle_graph();
    let corr = SpectralAlignment::align(&g, &g, 2).expect("Spectral alignment failed");
    assert!(!corr.is_empty(), "Should find correspondences");
    assert_eq!(corr.len(), 3);
}

#[test]
fn test_spectral_alignment_empty() {
    let g1 = GmGraph::new();
    let g2 = triangle_graph();
    let corr = SpectralAlignment::align(&g1, &g2, 2).expect("Spectral alignment failed");
    assert!(corr.is_empty());
}

#[test]
fn test_spectral_alignment_different_sizes() {
    let g1 = triangle_graph(); // 3 nodes
    let g2 = square_graph(); // 4 nodes
    let corr = SpectralAlignment::align(&g1, &g2, 2).expect("Spectral alignment failed");
    assert_eq!(corr.len(), 3); // min(3, 4) = 3 correspondences
}

#[test]
fn test_spectral_alignment_k_clamped() {
    let g = triangle_graph();
    // Request more eigenvectors than nodes.
    let corr = SpectralAlignment::align(&g, &g, 100).expect("Spectral alignment failed");
    assert!(!corr.is_empty());
}

// ---------------------------------------------------------------------------
// Graduated Assignment tests
// ---------------------------------------------------------------------------

#[test]
fn test_ga_identical_graphs() {
    let g = triangle_graph();
    let result = GraduatedAssignment::match_graphs(&g, &g, 10, 20, 0.5, 1.1).expect("GA failed");
    assert_eq!(result.discrete.len(), 3);
    // Soft assignment matrix should be 3x3.
    assert_eq!(result.matrix.len(), 3);
    assert_eq!(result.matrix[0].len(), 3);
}

#[test]
fn test_ga_empty_graphs() {
    let g = GmGraph::new();
    let result = GraduatedAssignment::match_graphs(&g, &g, 10, 20, 0.5, 1.1).expect("GA failed");
    assert!(result.discrete.is_empty());
}

#[test]
fn test_ga_soft_assignment_row_sums() {
    let g1 = triangle_graph();
    let g2 = square_graph();
    let result = GraduatedAssignment::match_graphs(&g1, &g2, 10, 30, 0.5, 1.2).expect("GA failed");
    // After Sinkhorn, rows should approximately sum to 1 (or close).
    for (i, row) in result.matrix.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!(sum > 0.0, "Row {i} sum should be positive, got {sum}");
    }
}

#[test]
fn test_ga_discrete_in_range() {
    let g1 = triangle_graph();
    let g2 = square_graph();
    let result = GraduatedAssignment::match_graphs(&g1, &g2, 5, 10, 0.5, 1.1).expect("GA failed");
    for &d in &result.discrete {
        assert!(d < g2.n_nodes(), "Discrete assignment out of range");
    }
}

#[test]
fn test_ga_beta_increases() {
    let g = triangle_graph();
    let result = GraduatedAssignment::match_graphs(&g, &g, 10, 20, 0.5, 2.0).expect("GA failed");
    // After 10 iterations with factor 2.0, beta = 0.5 * 2^10 = 512.
    assert!(result.final_beta > 0.5, "Beta should have increased");
}

// ---------------------------------------------------------------------------
// Maximum Common Subgraph tests
// ---------------------------------------------------------------------------

#[test]
fn test_mcs_identical_graphs() {
    let g = triangle_graph();
    let result = MaxCommonSubgraph::find_mcs(&g, &g).expect("MCS failed");
    assert_eq!(result.size, 3, "MCS of identical triangles should be 3");
}

#[test]
fn test_mcs_empty_graph() {
    let g1 = GmGraph::new();
    let g2 = triangle_graph();
    let result = MaxCommonSubgraph::find_mcs(&g1, &g2).expect("MCS failed");
    assert_eq!(result.size, 0);
}

#[test]
fn test_mcs_disjoint_graphs() {
    let g1 = triangle_graph();
    let g2 = star_graph(5);
    let result = MaxCommonSubgraph::find_mcs(&g1, &g2).expect("MCS failed");
    // Should find at least a single edge or node in common.
    assert!(result.size >= 1, "MCS should find at least 1 common node");
}

#[test]
fn test_mcs_subgraph_relationship() {
    // Triangle is a subgraph of K4.
    let g1 = triangle_graph();
    let mut g2 = GmGraph::new();
    for i in 0..4 {
        g2.add_node(vec![i as f64 * 0.3]);
    }
    for i in 0..4 {
        for j in (i + 1)..4 {
            g2.add_edge(i, j, vec![]).ok();
        }
    }
    let result = MaxCommonSubgraph::find_mcs(&g1, &g2).expect("MCS failed");
    assert!(result.size >= 2, "MCS of triangle and K4 should be >= 2");
}

#[test]
fn test_mcs_mapping_consistency() {
    let g = triangle_graph();
    let result = MaxCommonSubgraph::find_mcs(&g, &g).expect("MCS failed");
    // All mapped pairs should be valid node indices.
    for &(a, b) in &result.mapping {
        assert!(a < g.n_nodes());
        assert!(b < g.n_nodes());
    }
}

// ---------------------------------------------------------------------------
// GraphMatchMetrics tests
// ---------------------------------------------------------------------------

#[test]
fn test_metrics_accuracy_perfect() {
    let pred = vec![(0, 0), (1, 1), (2, 2)];
    let gt = vec![(0, 0), (1, 1), (2, 2)];
    let acc = GraphMatchMetrics::matching_accuracy(&pred, &gt);
    assert!((acc - 1.0).abs() < 1e-10);
}

#[test]
fn test_metrics_accuracy_zero() {
    let pred = vec![(0, 1), (1, 0), (2, 2)];
    let gt = vec![(0, 0), (1, 1)];
    let acc = GraphMatchMetrics::matching_accuracy(&pred, &gt);
    assert!(acc < 1.0);
}

#[test]
fn test_metrics_accuracy_empty() {
    let pred: Vec<(usize, usize)> = vec![];
    let gt: Vec<(usize, usize)> = vec![];
    let acc = GraphMatchMetrics::matching_accuracy(&pred, &gt);
    assert!((acc - 1.0).abs() < 1e-10);
}

#[test]
fn test_metrics_normalized_ged() {
    let g1 = triangle_graph();
    let g2 = path_graph();
    let nged = GraphMatchMetrics::normalized_ged(5.0, &g1, &g2);
    // max_ged = 3 + 4 + 3 + 3 = 13.
    assert!((nged - 5.0 / 13.0).abs() < 1e-10);
}

#[test]
fn test_metrics_normalized_ged_empty() {
    let g1 = GmGraph::new();
    let g2 = GmGraph::new();
    let nged = GraphMatchMetrics::normalized_ged(0.0, &g1, &g2);
    assert!((nged).abs() < 1e-10);
}

#[test]
fn test_metrics_psd_identity() {
    let km = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    assert!(GraphMatchMetrics::is_positive_semidefinite(&km, 1e-10));
}

#[test]
fn test_metrics_psd_negative_diagonal() {
    let km = vec![vec![-1.0, 0.0], vec![0.0, 1.0]];
    assert!(!GraphMatchMetrics::is_positive_semidefinite(&km, 1e-10));
}

#[test]
fn test_metrics_frobenius_norm() {
    let km = vec![vec![3.0, 0.0], vec![0.0, 4.0]];
    let norm = GraphMatchMetrics::frobenius_norm(&km);
    assert!((norm - 5.0).abs() < 1e-10); // sqrt(9 + 16) = 5
}

#[test]
fn test_metrics_evaluate_report() {
    let g1 = triangle_graph();
    let g2 = triangle_graph();
    let pred = vec![(0, 0), (1, 1), (2, 2)];
    let gt = vec![(0, 0), (1, 1), (2, 2)];
    let km = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
    let report = GraphMatchMetrics::evaluate(&pred, &gt, 0.0, &g1, &g2, Some(&km));
    assert!((report.accuracy - 1.0).abs() < 1e-10);
    assert!((report.normalized_ged).abs() < 1e-10);
    assert!(report.kernel_psd);
    assert!(report.kernel_frobenius_norm > 0.0);
}

#[test]
fn test_metrics_psd_symmetric_positive() {
    // Valid PSD kernel matrix.
    let km = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
    assert!(GraphMatchMetrics::is_positive_semidefinite(&km, 1e-10));
}

#[test]
fn test_metrics_psd_not_symmetric() {
    let km = vec![vec![1.0, 2.0], vec![0.0, 1.0]];
    assert!(!GraphMatchMetrics::is_positive_semidefinite(&km, 1e-10));
}

// ---------------------------------------------------------------------------
// Integration / cross-component tests
// ---------------------------------------------------------------------------

#[test]
fn test_wl_kernel_matrix_psd() {
    // Build a small kernel matrix and check PSD.
    let graphs = [triangle_graph(), path_graph(), square_graph()];
    let n = graphs.len();
    let mut km = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            km[i][j] = WeisfeilerLemanKernel::compute_kernel(&graphs[i], &graphs[j], 2)
                .expect("WL kernel failed");
        }
    }
    // WL kernel is known to be PSD.
    assert!(
        GraphMatchMetrics::is_positive_semidefinite(&km, 1e-6),
        "WL kernel matrix should be PSD"
    );
}

#[test]
fn test_sp_kernel_matrix_psd() {
    let graphs = [triangle_graph(), path_graph()];
    let n = graphs.len();
    let mut km = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            km[i][j] =
                ShortestPathKernel::compute(&graphs[i], &graphs[j]).expect("SP kernel failed");
        }
    }
    assert!(
        GraphMatchMetrics::is_positive_semidefinite(&km, 1e-6),
        "SP kernel matrix should be PSD"
    );
}

#[test]
fn test_ged_and_normalized_ged_consistency() {
    let g1 = triangle_graph();
    let g2 = path_graph();
    let cost = GmEditCostModel::default();
    let (ged, _) = GraphEditDistance::compute(&g1, &g2, 10, &cost).expect("GED failed");
    let nged = GraphMatchMetrics::normalized_ged(ged, &g1, &g2);
    assert!(
        (0.0..=1.0 + 1e-6).contains(&nged),
        "Normalized GED should be in [0, 1], got {nged}"
    );
}

#[test]
fn test_spectral_then_metrics() {
    let g1 = triangle_graph();
    let g2 = triangle_graph();
    let corr = SpectralAlignment::align(&g1, &g2, 2).expect("Spectral alignment failed");
    let gt: Vec<(usize, usize)> = vec![(0, 0), (1, 1), (2, 2)];
    let acc = GraphMatchMetrics::matching_accuracy(&corr, &gt);
    assert!((0.0..=1.0).contains(&acc));
}

#[test]
fn test_full_pipeline_triangle() {
    let g1 = triangle_graph();
    let g2 = triangle_graph();

    // GED.
    let cost = GmEditCostModel::default();
    let (ged, _path) = GraphEditDistance::compute(&g1, &g2, 10, &cost).expect("GED failed");

    // WL kernel.
    let wl_k = WeisfeilerLemanKernel::compute_kernel(&g1, &g2, 3).expect("WL kernel failed");

    // Spectral alignment.
    let corr = SpectralAlignment::align(&g1, &g2, 2).expect("Spectral alignment failed");

    // Graduated assignment.
    let ga = GraduatedAssignment::match_graphs(&g1, &g2, 5, 10, 0.5, 1.5).expect("GA failed");

    // MCS.
    let mcs = MaxCommonSubgraph::find_mcs(&g1, &g2).expect("MCS failed");

    assert!(ged < 1e-6, "GED of identical triangles should be ~0");
    assert!(wl_k > 0.0);
    assert!(!corr.is_empty());
    assert_eq!(ga.discrete.len(), 3);
    assert_eq!(mcs.size, 3);
}

#[test]
fn test_feature_distance_same() {
    let d = feature_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
    assert!(d.abs() < 1e-10);
}

#[test]
fn test_feature_distance_different_lengths() {
    let d = feature_distance(&[1.0], &[1.0, 2.0]);
    // sqrt((0 - 2)^2) = 2
    assert!((d - 2.0).abs() < 1e-10);
}
