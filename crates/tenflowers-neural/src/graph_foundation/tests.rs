//! Tests for graph_foundation module.

use super::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

fn make_rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn simple_adj(n: usize) -> Vec<Vec<usize>> {
    (0..n)
        .map(|i| if i + 1 < n { vec![i + 1] } else { vec![] })
        .collect()
}

// ── Graph Transformers ──────────────────────────────────────────────────

#[test]
fn test_graphormer_bias_spatial() {
    let mut rng = make_rng();
    let n = 4;
    let bias = GraphormerBias::new(8, n, 0, &mut rng);
    let adj = simple_adj(n);
    let degrees = vec![1usize; n];
    let b = bias.compute_bias(&adj, &degrees);
    assert_eq!(b.len(), n);
    assert_eq!(b[0].len(), n);
    let d0 = GraphormerBias::bfs_distances(&adj, n);
    assert_eq!(d0[0][0], 0);
    assert_eq!(d0[0][1], 1);
}

#[test]
fn test_graphormer_bias_bfs() {
    let adj: Vec<Vec<usize>> = vec![vec![1, 2], vec![0, 3], vec![0], vec![1]];
    let dist = GraphormerBias::bfs_distances(&adj, 4);
    assert_eq!(dist[0][3], 2);
    assert_eq!(dist[0][0], 0);
}

#[test]
fn test_graphormer_layer_shape() {
    let mut rng = make_rng();
    let n = 5;
    let d = 16;
    let h = 4;
    let layer = GraphormerLayer::new(d, h, &mut rng).expect("GraphormerLayer creation failed");
    let feats: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..d).map(|_| rng.random::<f64>()).collect())
        .collect();
    let adj = simple_adj(n);
    let degrees = vec![1usize; n];
    let bias = GraphormerBias::new(8, n, 0, &mut rng);
    let out = layer.forward(&feats, &adj, &degrees, &bias);
    assert_eq!(out.len(), n);
    assert_eq!(out[0].len(), d);
}

#[test]
fn test_graphormer_layer_invalid_heads() {
    let mut rng = make_rng();
    let result = GraphormerLayer::new(15, 4, &mut rng);
    assert!(result.is_err());
}

#[test]
fn test_graphormer_model_construction() {
    let mut rng = make_rng();
    let model = GraphormerModel::new(2, 16, 4, 8, 5, &mut rng).expect("GraphormerModel creation failed");
    let feats: Vec<Vec<f64>> = (0..5)
        .map(|_| (0..16).map(|_| rng.random::<f64>()).collect())
        .collect();
    let adj = simple_adj(5);
    let degrees = vec![1usize; 5];
    let out = model.forward(&feats, &adj, &degrees);
    assert_eq!(out.len(), 16);
}

#[test]
fn test_graphormer_model_empty() {
    let mut rng = make_rng();
    let model = GraphormerModel::new(1, 8, 2, 4, 1, &mut rng).expect("GraphormerModel creation failed");
    let out = model.forward(&[], &[], &[]);
    assert_eq!(out.len(), 8);
}

#[test]
fn test_token_gt_layer() {
    let mut rng = make_rng();
    let layer = TokenGtLayer::new(8, 16, 4, &mut rng).expect("TokenGtLayer creation failed");
    let feats: Vec<Vec<f64>> = (0..4)
        .map(|_| (0..8).map(|_| rng.random::<f64>()).collect())
        .collect();
    let edges = vec![(0usize, 1usize), (1, 2), (2, 3)];
    let out = layer.forward(&feats, &edges);
    assert_eq!(out.len(), 7);
    assert_eq!(out[0].len(), 16);
}

#[test]
fn test_graph_gps_layer() {
    let mut rng = make_rng();
    let layer = GraphGPSLayer::new(16, 4, &mut rng).expect("GraphGPSLayer creation failed");
    let feats: Vec<Vec<f64>> = (0..5)
        .map(|_| (0..16).map(|_| rng.random::<f64>()).collect())
        .collect();
    let edges = vec![(0usize, 1usize), (1, 2), (2, 3), (3, 4)];
    let out = layer.forward(&feats, &edges);
    assert_eq!(out.len(), 5);
    assert_eq!(out[0].len(), 16);
}

#[test]
fn test_graph_gps_no_edges() {
    let mut rng = make_rng();
    let layer = GraphGPSLayer::new(8, 2, &mut rng).expect("GraphGPSLayer creation failed");
    let feats: Vec<Vec<f64>> = (0..3).map(|_| (0..8).map(|_| 1.0).collect()).collect();
    let out = layer.forward(&feats, &[]);
    assert_eq!(out.len(), 3);
}

// ── Pre-training ────────────────────────────────────────────────────────

#[test]
fn test_graph_mae_masking_count() {
    let mut rng = make_rng();
    let mae = GraphMaskedAutoencoder::new(8, 16, &mut rng);
    let (visible, masked) = mae.mask_nodes(10, 0.3, &mut rng);
    assert_eq!(visible.len() + masked.len(), 10);
    assert_eq!(masked.len(), 3);
}

#[test]
fn test_graph_mae_reconstruction_loss() {
    let mut rng = make_rng();
    let mae = GraphMaskedAutoencoder::new(8, 8, &mut rng);
    let feats: Vec<Vec<f64>> = (0..5)
        .map(|_| (0..8).map(|_| rng.random::<f64>()).collect())
        .collect();
    let (_, masked) = mae.mask_nodes(5, 0.4, &mut rng);
    let pred = mae.forward(&feats, &masked);
    let loss = mae.reconstruction_loss(&pred, &feats, &masked);
    assert!(loss >= 0.0);
}

#[test]
fn test_graph_mae_zero_mask() {
    let mut rng = make_rng();
    let mae = GraphMaskedAutoencoder::new(4, 8, &mut rng);
    let loss = mae.reconstruction_loss(&[], &[], &[]);
    assert_eq!(loss, 0.0);
}

#[test]
fn test_edge_prediction_loss() {
    let mut rng = make_rng();
    let epp = EdgePredictionPretraining::new(16, &mut rng);
    let embs: Vec<Vec<f64>> = (0..6)
        .map(|_| (0..16).map(|_| rng.random::<f64>()).collect())
        .collect();
    let pos = vec![(0usize, 1usize), (1, 2), (2, 3)];
    let neg = epp.sample_negatives(6, &pos, 3, &mut rng);
    let loss = epp.loss(&embs, &pos, &neg);
    assert!(loss >= 0.0);
}

#[test]
fn test_edge_prediction_score_range() {
    let mut rng = make_rng();
    let epp = EdgePredictionPretraining::new(8, &mut rng);
    let h: Vec<f64> = (0..8).map(|_| rng.random::<f64>()).collect();
    let t: Vec<f64> = (0..8).map(|_| rng.random::<f64>()).collect();
    let s = epp.score_edge(&h, &t);
    assert!((0.0..=1.0).contains(&s), "score should be in [0,1], got {s}");
}

#[test]
fn test_attr_masking() {
    let mut rng = make_rng();
    let am = AttrMasking::new(8, &mut rng);
    let feats: Vec<Vec<f64>> = (0..4)
        .map(|_| (0..8).map(|_| rng.random::<f64>()).collect())
        .collect();
    let (masked_feats, mask_indices) = am.mask_attrs(&feats, 0.5, &mut rng);
    assert_eq!(masked_feats.len(), feats.len());
    assert_eq!(mask_indices.len(), feats.len());
    for (node_idx, indices) in mask_indices.iter().enumerate() {
        for &fi in indices {
            assert_eq!(masked_feats[node_idx][fi], 0.0);
        }
    }
}

#[test]
fn test_attr_masking_predict_shape() {
    let mut rng = make_rng();
    let am = AttrMasking::new(8, &mut rng);
    let feats: Vec<Vec<f64>> = (0..3).map(|_| (0..8).map(|_| 1.0).collect()).collect();
    let (masked, _) = am.mask_attrs(&feats, 0.25, &mut rng);
    let preds = am.predict(&masked);
    assert_eq!(preds.len(), 3);
    assert_eq!(preds[0].len(), 8);
}

#[test]
fn test_graph_contrastive_pretrain() {
    let mut rng = make_rng();
    let gcp = GraphContrastivePretraining::new(16, 0.07, &mut rng);
    let z1: Vec<f64> = (0..16).map(|_| rng.random::<f64>()).collect();
    let z2: Vec<f64> = (0..16).map(|_| rng.random::<f64>()).collect();
    let p1 = gcp.project(&z1);
    let p2 = gcp.project(&z2);
    assert_eq!(p1.len(), 16);
    let negs: Vec<Vec<f64>> = (0..4)
        .map(|_| gcp.project(&(0..16).map(|_| rng.random::<f64>()).collect::<Vec<_>>()))
        .collect();
    let loss = gcp.info_nce_loss(&p1, &p2, &negs);
    assert!(loss.is_finite());
}

#[test]
fn test_context_prediction() {
    let mut rng = make_rng();
    let cp = ContextPrediction::new(2, 8, &mut rng);
    let adj: Vec<Vec<usize>> = vec![vec![1, 2], vec![0, 3], vec![0], vec![1]];
    let nbrs = cp.r_hop_neighbors(0, &adj);
    assert!(nbrs.contains(&1));
    assert!(nbrs.contains(&2));
}

// ── Few-Shot ────────────────────────────────────────────────────────────

#[test]
fn test_graph_proto_net_prototypes() {
    let mut rng = make_rng();
    let proto = GraphProtoNet::new(8, 16, &mut rng);
    let support: Vec<(Vec<f64>, usize)> = vec![
        ((0..8).map(|_| rng.random::<f64>()).collect(), 0),
        ((0..8).map(|_| rng.random::<f64>()).collect(), 0),
        ((0..8).map(|_| rng.random::<f64>()).collect(), 1),
    ];
    let protos = proto.compute_prototypes(&support);
    assert_eq!(protos.len(), 2);
    assert!(protos.contains_key(&0));
    assert!(protos.contains_key(&1));
}

#[test]
fn test_graph_proto_net_predict() {
    let mut rng = make_rng();
    let proto = GraphProtoNet::new(8, 16, &mut rng);
    let support: Vec<(Vec<f64>, usize)> = vec![
        ((0..8).map(|_| 0.0).collect(), 0),
        ((0..8).map(|_| 1.0).collect(), 1),
    ];
    let protos = proto.compute_prototypes(&support);
    let query: Vec<f64> = (0..8).map(|_| 0.1).collect();
    let pred = proto.predict(&query, &protos);
    assert!(pred == 0 || pred == 1);
}

#[test]
fn test_graph_matching_network() {
    let mut rng = make_rng();
    let gmn = GraphMatchingNetwork::new(16, &mut rng);
    let query: Vec<f64> = (0..16).map(|_| rng.random::<f64>()).collect();
    let support: Vec<Vec<f64>> = (0..3)
        .map(|_| (0..16).map(|_| rng.random::<f64>()).collect())
        .collect();
    let score = gmn.match_score(&query, &support);
    assert!(score.is_finite());
}

#[test]
fn test_graph_matching_empty_support() {
    let mut rng = make_rng();
    let gmn = GraphMatchingNetwork::new(8, &mut rng);
    let query: Vec<f64> = (0..8).map(|_| 1.0).collect();
    let score = gmn.match_score(&query, &[]);
    assert!(score.is_finite());
}

#[test]
fn test_meta_gnn_construction() {
    let mut rng = make_rng();
    let meta = MetaGnn::new(16, 0.01, 3, &mut rng);
    assert_eq!(meta.weights.len(), 16);
    let feats: Vec<Vec<f64>> = (0..4)
        .map(|_| (0..16).map(|_| rng.random::<f64>()).collect())
        .collect();
    let adapted = meta.adapt(&feats);
    assert_eq!(adapted.len(), 16);
}

#[test]
fn test_task_aware_gnn() {
    let mut rng = make_rng();
    let tgnn = TaskAwareGnn::new(16, 8, &mut rng);
    let support: Vec<Vec<f64>> = (0..3)
        .map(|_| (0..16).map(|_| rng.random::<f64>()).collect())
        .collect();
    let task_emb = tgnn.encode_task(&support);
    assert_eq!(task_emb.len(), 8);
    let node_feat: Vec<f64> = (0..16).map(|_| rng.random::<f64>()).collect();
    let out = tgnn.forward(&node_feat, &task_emb);
    assert_eq!(out.len(), 16);
}

#[test]
fn test_graph_episode_sampler() {
    let mut rng = make_rng();
    let sampler = GraphEpisodeSampler::new(2, 2, 1);
    let dataset: Vec<(Vec<f64>, usize)> = (0..20)
        .map(|i| ((0..8).map(|_| rng.random::<f64>()).collect(), i % 4))
        .collect();
    let (support, _query) = sampler.sample_episode(&dataset, &mut rng);
    assert!(support.len() <= 4);
}

// ── HGT ─────────────────────────────────────────────────────────────────

#[test]
fn test_hgt_layer_output_types() {
    use std::collections::HashMap;
    let mut rng = make_rng();
    let layer = HgtLayer::new(
        16,
        4,
        &["paper", "author"],
        &["paper-written_by-author"],
        &mut rng,
    )
    .expect("HgtLayer creation failed");
    let mut node_feats = HashMap::new();
    node_feats.insert("paper".to_string(), vec![vec![1.0f64; 16]; 3]);
    node_feats.insert("author".to_string(), vec![vec![0.5f64; 16]; 2]);
    let mut edges = HashMap::new();
    edges.insert(
        "paper-written_by-author".to_string(),
        vec![(0usize, 0usize), (1, 1)],
    );
    let out = layer.forward(&node_feats, &edges);
    assert!(out.contains_key("paper") || out.contains_key("author"));
}

#[test]
fn test_hgt_layer_invalid() {
    let mut rng = make_rng();
    let result = HgtLayer::new(15, 4, &["a"], &[], &mut rng);
    assert!(result.is_err());
}

#[test]
fn test_relational_gcn_shape() {
    let mut rng = make_rng();
    let rgcn = RelationalGcn::new(8, 16, 3, 2, &mut rng);
    let feats: Vec<Vec<f64>> = (0..5)
        .map(|_| (0..8).map(|_| rng.random::<f64>()).collect())
        .collect();
    let adj_per_rel: Vec<Vec<Vec<usize>>> = (0..3)
        .map(|_| {
            (0..5)
                .map(|i| if i > 0 { vec![i - 1] } else { vec![] })
                .collect()
        })
        .collect();
    let out = rgcn.forward(&feats, &adj_per_rel);
    assert_eq!(out.len(), 5);
    assert_eq!(out[0].len(), 16);
}

#[test]
fn test_comp_gcn_layer() {
    let mut rng = make_rng();
    let layer = CompGcnLayer::new(16, 3, &mut rng);
    let feats: Vec<Vec<f64>> = (0..4)
        .map(|_| (0..16).map(|_| rng.random::<f64>()).collect())
        .collect();
    let edges = vec![(0usize, 1usize, 0usize), (1, 2, 1), (2, 3, 2)];
    let out = layer.forward(&feats, &edges);
    assert_eq!(out.len(), 4);
    assert_eq!(out[0].len(), 16);
}

#[test]
fn test_hetero_sage() {
    let mut rng = make_rng();
    let hs = HeteroSage::new(8, &["user", "item"], &mut rng);
    let feats: Vec<Vec<f64>> = (0..3)
        .map(|_| (0..8).map(|_| rng.random::<f64>()).collect())
        .collect();
    let neighbor_feats: Vec<Vec<f64>> = (0..2)
        .map(|_| (0..8).map(|_| rng.random::<f64>()).collect())
        .collect();
    let out = hs.aggregate("user", &feats, &neighbor_feats);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].len(), 8);
}

#[test]
fn test_semantic_attention() {
    let mut rng = make_rng();
    let sa = SemanticAttention::new(16, &mut rng);
    let paths: Vec<Vec<f64>> = (0..3)
        .map(|_| (0..16).map(|_| rng.random::<f64>()).collect())
        .collect();
    let out = sa.aggregate(&paths);
    assert_eq!(out.len(), 16);
}

#[test]
fn test_semantic_attention_empty() {
    let mut rng = make_rng();
    let sa = SemanticAttention::new(8, &mut rng);
    let out = sa.aggregate(&[]);
    assert_eq!(out.len(), 8);
    assert!(out.iter().all(|&x| x == 0.0));
}

// ── Link Prediction / Graph Generation ──────────────────────────────────

#[test]
fn test_complex_link_score() {
    let mut rng = make_rng();
    let model = ComplexLinkPredictor::new(10, 5, 16, &mut rng);
    let s = model.score(0, 0, 1);
    assert!(s.is_finite());
    let s_oob = model.score(100, 0, 0);
    assert_eq!(s_oob, 0.0);
}

#[test]
fn test_complex_link_score_symmetry() {
    let mut rng = make_rng();
    let model = ComplexLinkPredictor::new(5, 3, 8, &mut rng);
    let s1 = model.score(0, 0, 1);
    let s2 = model.score(1, 0, 0);
    assert!(s1.is_finite() && s2.is_finite());
}

#[test]
fn test_rotate_link_score() {
    let mut rng = make_rng();
    let model = RotateLinkPredictor::new(10, 5, 16, 12.0, &mut rng);
    let s = model.score(0, 0, 1);
    assert!(s.is_finite());
}

#[test]
fn test_rotate_link_score_oob() {
    let mut rng = make_rng();
    let model = RotateLinkPredictor::new(5, 3, 8, 6.0, &mut rng);
    assert_eq!(model.score(100, 0, 0), 0.0);
    assert_eq!(model.score(0, 100, 0), 0.0);
}

#[test]
fn test_graph_rnn_node() {
    let mut rng = make_rng();
    let gen = GraphRnnNode::new(16, 5, &mut rng);
    let probs = gen.generate(4);
    assert_eq!(probs.len(), 4);
    for p in &probs {
        assert_eq!(p.len(), 5);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}

#[test]
fn test_graph_rnn_edge() {
    let mut rng = make_rng();
    let edge_rnn = GraphRnnEdge::new(16, &mut rng);
    let node_hidden: Vec<f64> = (0..16).map(|_| rng.random::<f64>()).collect();
    let probs = edge_rnn.predict_edges(4, &node_hidden);
    assert_eq!(probs.len(), 4);
    assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
}

#[test]
fn test_graph_rnn_edge_no_previous() {
    let mut rng = make_rng();
    let edge_rnn = GraphRnnEdge::new(8, &mut rng);
    let h: Vec<f64> = vec![0.5; 8];
    let probs = edge_rnn.predict_edges(0, &h);
    assert_eq!(probs.len(), 0);
}

#[test]
fn test_molecule_generator() {
    let mut rng = make_rng();
    let gen = MoleculeGenerator::new(16, 6, 4, &mut rng);
    let scaffold: Vec<Vec<f64>> = (0..3)
        .map(|_| (0..16).map(|_| rng.random::<f64>()).collect())
        .collect();
    let latent = gen.encode_scaffold(&scaffold);
    assert_eq!(latent.len(), 16);
    let atoms = gen.decode_atoms(&latent, 5);
    assert_eq!(atoms.len(), 5);
    assert_eq!(atoms[0].len(), 6);
    let bonds = gen.decode_bonds(&latent, 4);
    assert_eq!(bonds.len(), 4);
    assert_eq!(bonds[0].len(), 4);
}

#[test]
fn test_molecule_generator_empty_scaffold() {
    let mut rng = make_rng();
    let gen = MoleculeGenerator::new(8, 4, 3, &mut rng);
    let latent = gen.encode_scaffold(&[]);
    assert_eq!(latent.len(), 8);
    assert!(latent.iter().all(|&x| x == 0.0));
}

#[test]
fn test_graph_contrastive_drop_edges() {
    let mut rng = make_rng();
    let gcp = GraphContrastivePretraining::new(8, 0.07, &mut rng);
    let edges: Vec<(usize, usize)> = (0..10).map(|i| (i, i + 1)).collect();
    let dropped = gcp.drop_edges(&edges, 0.5, &mut rng);
    assert!(dropped.len() <= 10);
}

#[test]
fn test_graph_contrastive_drop_features() {
    let mut rng = make_rng();
    let gcp = GraphContrastivePretraining::new(8, 0.1, &mut rng);
    let feats: Vec<Vec<f64>> = (0..4).map(|_| vec![1.0f64; 8]).collect();
    let dropped = gcp.drop_features(&feats, 0.5, &mut rng);
    assert_eq!(dropped.len(), 4);
    let _ = dropped; // randomness, no strict assert needed
}

// ── Pretraining (new) ───────────────────────────────────────────────────

#[test]
fn test_gcl_augmentation_node_drop() {
    let mut rng = make_rng();
    let model = GraphCL::new(8, 16, 8, 0.2, 0.3, &mut rng);
    let adj = vec![vec![1usize, 2], vec![0, 2], vec![0, 1]];
    let feats: Vec<Vec<f64>> = (0..3).map(|_| vec![1.0f64; 8]).collect();
    let g = GclGraph::new(feats, adj);
    let aug = model.augment(&g, GclAugmentation::NodeDropping, &mut rng);
    assert!(aug.n_nodes() <= 3);
}

#[test]
fn test_gcl_augmentation_edge_perturb() {
    let mut rng = make_rng();
    let model = GraphCL::new(8, 16, 8, 0.2, 0.5, &mut rng);
    let adj = vec![vec![1usize, 2], vec![0, 2], vec![0, 1]];
    let feats: Vec<Vec<f64>> = (0..3).map(|_| vec![1.0f64; 8]).collect();
    let g = GclGraph::new(feats, adj);
    let aug = model.augment(&g, GclAugmentation::EdgePerturbation, &mut rng);
    assert_eq!(aug.n_nodes(), 3);
    // Some edges may have been dropped
    let total_edges: usize = aug.adj.iter().map(|v| v.len()).sum();
    assert!(total_edges <= 6);
}

#[test]
fn test_gcl_augmentation_attr_mask() {
    let mut rng = make_rng();
    let model = GraphCL::new(8, 16, 8, 0.2, 0.5, &mut rng);
    let adj = simple_adj(4);
    let feats: Vec<Vec<f64>> = (0..4).map(|_| vec![1.0f64; 8]).collect();
    let g = GclGraph::new(feats, adj);
    let aug = model.augment(&g, GclAugmentation::AttributeMasking, &mut rng);
    assert_eq!(aug.n_nodes(), 4);
    // Some features should be zeroed
    let has_zero = aug.node_feats.iter().any(|f| f.contains(&0.0));
    let _ = has_zero; // with 50% drop rate, likely true but not guaranteed for test
}

#[test]
fn test_gcl_augmentation_subgraph() {
    let mut rng = make_rng();
    let model = GraphCL::new(8, 16, 8, 0.2, 0.5, &mut rng);
    let adj: Vec<Vec<usize>> = vec![vec![1, 2], vec![0, 3], vec![0], vec![1]];
    let feats: Vec<Vec<f64>> = (0..4).map(|_| vec![1.0f64; 8]).collect();
    let g = GclGraph::new(feats, adj);
    let aug = model.augment(&g, GclAugmentation::SubgraphSampling, &mut rng);
    assert!(aug.n_nodes() >= 1 && aug.n_nodes() <= 4);
}

#[test]
fn test_gcl_encode_shape() {
    let mut rng = make_rng();
    let model = GraphCL::new(8, 16, 8, 0.2, 0.3, &mut rng);
    let adj = simple_adj(5);
    let feats: Vec<Vec<f64>> = (0..5).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect();
    let g = GclGraph::new(feats, adj);
    let enc = model.encode(&g);
    assert_eq!(enc.len(), 16);
}

#[test]
fn test_gcl_nt_xent_loss() {
    let mut rng = make_rng();
    let model = GraphCL::new(8, 16, 8, 0.1, 0.3, &mut rng);
    let z1: Vec<Vec<f64>> = (0..4).map(|_| {
        let h: Vec<f64> = (0..16).map(|_| rng.random::<f64>()).collect();
        model.project(&h)
    }).collect();
    let z2: Vec<Vec<f64>> = (0..4).map(|_| {
        let h: Vec<f64> = (0..16).map(|_| rng.random::<f64>()).collect();
        model.project(&h)
    }).collect();
    let loss = model.nt_xent_loss(&z1, &z2);
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

#[test]
fn test_graph_contraster_train_step() {
    let mut rng = make_rng();
    let gcl = GraphCL::new(8, 16, 8, 0.1, 0.3, &mut rng);
    let contraster = GraphContraster::new(gcl, 2);
    let graphs: Vec<GclGraph> = (0..4).map(|_| {
        let adj = simple_adj(5);
        let feats: Vec<Vec<f64>> = (0..5).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect();
        GclGraph::new(feats, adj)
    }).collect();
    let loss = contraster.train_step(&graphs, &mut rng);
    assert!(loss.is_finite());
}

#[test]
fn test_graph_contraster_hard_negatives() {
    let mut rng = make_rng();
    let gcl = GraphCL::new(8, 16, 8, 0.1, 0.3, &mut rng);
    let contraster = GraphContraster::new(gcl, 2);
    let query: Vec<f64> = (0..8).map(|_| rng.random::<f64>()).collect();
    let pool: Vec<Vec<f64>> = (0..5).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect();
    let hard_neg = contraster.hard_negatives(&query, &pool, 0);
    assert!(hard_neg.len() <= 2);
}

#[test]
fn test_graph_mae_encoder_shape() {
    let mut rng = make_rng();
    let encoder = GraphMaeEncoder::new(8, 16, &mut rng);
    let adj = simple_adj(5);
    let feats: Vec<Vec<f64>> = (0..5).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect();
    let masked = vec![0usize, 2];
    let latents = encoder.encode(&feats, &adj, &masked);
    assert_eq!(latents.len(), 5);
    assert_eq!(latents[0].len(), 16);
}

#[test]
fn test_graph_mae_encoder_degree_mask() {
    let mut rng = make_rng();
    let encoder = GraphMaeEncoder::new(8, 16, &mut rng);
    let adj: Vec<Vec<usize>> = vec![vec![1, 2, 3], vec![0], vec![0], vec![0]];
    let masked = encoder.select_mask(4, 0.5, &adj, MaeMaskStrategy::DegreeBased, &mut rng);
    assert_eq!(masked.len(), 2);
    // Node 0 has degree 3 (highest), should be in mask
    assert!(masked.contains(&0));
}

#[test]
fn test_graph_mae_decoder_shape() {
    let mut rng = make_rng();
    let decoder = GraphMaeDecoder::new(16, 8, &mut rng);
    let latents: Vec<Vec<f64>> = (0..4).map(|_| (0..16).map(|_| rng.random::<f64>()).collect()).collect();
    let adj = simple_adj(4);
    let recon = decoder.decode(&latents, &adj);
    assert_eq!(recon.len(), 4);
    assert_eq!(recon[0].len(), 8);
}

#[test]
fn test_graph_mae_model_forward() {
    let mut rng = make_rng();
    let model = GraphMaeModel::new(8, 16, 0.3, MaeMaskStrategy::UniformRandom, &mut rng);
    let adj = simple_adj(6);
    let feats: Vec<Vec<f64>> = (0..6).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect();
    let (recon, masked) = model.forward(&feats, &adj, &mut rng);
    assert_eq!(recon.len(), 6);
    assert!(!masked.is_empty());
}

#[test]
fn test_graph_mae_sce_loss() {
    let mut rng = make_rng();
    let model = GraphMaeModel::new(8, 16, 0.3, MaeMaskStrategy::UniformRandom, &mut rng);
    let adj = simple_adj(5);
    let feats: Vec<Vec<f64>> = (0..5).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect();
    let (recon, masked) = model.forward(&feats, &adj, &mut rng);
    let loss = model.sce_loss(&recon, &feats, &masked);
    assert!((0.0..=1.0).contains(&loss));
}

#[test]
fn test_gtp_tokenizer_bfs_order() {
    let adj: Vec<Vec<usize>> = vec![vec![1, 2], vec![0, 3], vec![0], vec![1]];
    let order = GtpTokenizer::bfs_order(&adj, 0);
    assert_eq!(order[0], 0);
    assert!(order.contains(&1) && order.contains(&2) && order.contains(&3));
}

#[test]
fn test_gtp_tokenizer_rwpe() {
    let tokenizer = GtpTokenizer::new(8, 4, 5);
    let adj: Vec<Vec<usize>> = vec![vec![1], vec![0, 2], vec![1]];
    let pe = tokenizer.rwpe(&adj);
    assert_eq!(pe.len(), 3);
    assert_eq!(pe[0].len(), 4);
}

#[test]
fn test_gtp_tokenizer_tokenize() {
    let mut rng = make_rng();
    let tokenizer = GtpTokenizer::new(8, 4, 3);
    let adj = simple_adj(4);
    let feats: Vec<Vec<f64>> = (0..4).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect();
    let tokens = tokenizer.tokenize(&feats, &adj, &mut rng);
    assert_eq!(tokens.len(), 4);
    assert_eq!(tokens[0].1.len(), 12); // d_feat + d_pe = 8 + 4
}

#[test]
fn test_gtp_model_graph_repr() {
    let mut rng = make_rng();
    let model = GtpModel::new(8, 4, 3, 16, &mut rng);
    let adj = simple_adj(5);
    let feats: Vec<Vec<f64>> = (0..5).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect();
    let repr = model.graph_repr(&feats, &adj, &mut rng);
    assert_eq!(repr.len(), 16);
}

#[test]
fn test_gtp_model_mgm_loss() {
    let mut rng = make_rng();
    let model = GtpModel::new(8, 4, 3, 16, &mut rng);
    let adj = simple_adj(6);
    let feats: Vec<Vec<f64>> = (0..6).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect();
    let loss = model.mgm_loss(&feats, &adj, 0.3, &mut rng);
    assert!(loss >= 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_gtp_pretrainer_classify() {
    let mut rng = make_rng();
    let model = GtpModel::new(8, 4, 3, 16, &mut rng);
    let pretrainer = GtpPretrainer::new(model, 3, 0.3, &mut rng);
    let adj = simple_adj(5);
    let feats: Vec<Vec<f64>> = (0..5).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect();
    let cls = pretrainer.classify(&feats, &adj, &mut rng);
    assert!(cls < 3);
}

#[test]
fn test_gtp_pretrainer_loss() {
    let mut rng = make_rng();
    let model = GtpModel::new(8, 4, 3, 16, &mut rng);
    let pretrainer = GtpPretrainer::new(model, 3, 0.3, &mut rng);
    let adj = simple_adj(5);
    let feats: Vec<Vec<f64>> = (0..5).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect();
    let loss = pretrainer.pretrain_loss(&feats, &adj, Some(1), &mut rng);
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

#[test]
fn test_fed_graph_encode() {
    let mut rng = make_rng();
    let fg = FedGraph::new(8, 16, 3, 0.01, 0.3, &mut rng);
    let adj = simple_adj(5);
    let feats: Vec<Vec<f64>> = (0..5).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect();
    let enc = fg.encode(&feats, &adj);
    assert_eq!(enc.len(), 16);
    // L2 norm should be ~1
    let norm: f64 = enc.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn test_fed_graph_local_update() {
    let mut rng = make_rng();
    let mut fg = FedGraph::new(8, 16, 3, 0.001, 0.3, &mut rng);
    let adj = simple_adj(5);
    let feats: Vec<Vec<f64>> = (0..5).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect();
    let loss = fg.local_update(0, &feats, &adj, &mut rng);
    assert!(loss.is_finite());
}

#[test]
fn test_fed_graph_fedavg() {
    let mut rng = make_rng();
    let mut fg = FedGraph::new(8, 16, 3, 0.001, 0.3, &mut rng);
    let adj = simple_adj(4);
    let feats: Vec<Vec<f64>> = (0..4).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect();
    fg.local_update(0, &feats, &adj, &mut rng);
    fg.local_update(1, &feats, &adj, &mut rng);
    fg.fedavg();
    // Global encoder should be updated
    assert!(!fg.global_encoder.is_empty());
}

#[test]
fn test_cross_graph_transfer_mmd() {
    let mut rng = make_rng();
    let cgt = CrossGraphTransfer::new(8, 16, 1.0, &mut rng);
    let src: Vec<Vec<Vec<f64>>> = (0..3).map(|_| {
        (0..4).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect()
    }).collect();
    let tgt: Vec<Vec<Vec<f64>>> = (0..3).map(|_| {
        (0..4).map(|_| (0..8).map(|_| rng.random::<f64>() + 1.0).collect()).collect()
    }).collect();
    let mmd = cgt.mmd_loss(&src, &tgt);
    assert!(mmd.is_finite());
    assert!(mmd >= 0.0);
}

#[test]
fn test_cross_graph_transfer_adapt() {
    let mut rng = make_rng();
    let mut cgt = CrossGraphTransfer::new(8, 16, 1.0, &mut rng);
    let src: Vec<Vec<Vec<f64>>> = (0..2).map(|_| {
        (0..3).map(|_| (0..8).map(|_| rng.random::<f64>()).collect()).collect()
    }).collect();
    let tgt: Vec<Vec<Vec<f64>>> = (0..2).map(|_| {
        (0..3).map(|_| (0..8).map(|_| rng.random::<f64>() + 2.0).collect()).collect()
    }).collect();
    let loss_before = cgt.adapt_step(&src, &tgt, 0.01);
    assert!(loss_before.is_finite());
}

#[test]
fn test_graph_metrics_auc_roc() {
    let mut metrics = GraphMetrics::new();
    let pos: Vec<f64> = vec![0.9, 0.8, 0.7, 0.85];
    let neg: Vec<f64> = vec![0.2, 0.1, 0.3, 0.15];
    metrics.add_link_scores(&pos, &neg);
    let auc = metrics.auc_roc();
    assert!(auc > 0.5);
    assert!(auc <= 1.0);
}

#[test]
fn test_graph_metrics_average_precision() {
    let mut metrics = GraphMetrics::new();
    let pos: Vec<f64> = vec![0.9, 0.8, 0.7];
    let neg: Vec<f64> = vec![0.3, 0.2, 0.1];
    metrics.add_link_scores(&pos, &neg);
    let ap = metrics.average_precision();
    assert!(ap > 0.0 && ap <= 1.0);
}

#[test]
fn test_graph_metrics_accuracy() {
    let mut metrics = GraphMetrics::new();
    let labels = vec![0usize, 1, 0, 1];
    let probs: Vec<Vec<f64>> = vec![
        vec![0.9, 0.1],
        vec![0.2, 0.8],
        vec![0.8, 0.2],
        vec![0.1, 0.9],
    ];
    metrics.add_node_preds(&labels, &probs);
    let acc = metrics.accuracy();
    assert_eq!(acc, 1.0); // all correct
}

#[test]
fn test_graph_metrics_hits_at_k() {
    let mut metrics = GraphMetrics::new();
    // One positive with score 0.9, one negative with score 0.5
    metrics.add_link_scores(&[0.9], &[0.5]);
    let h1 = metrics.hits_at_k(1);
    assert_eq!(h1, 1.0); // 0.9 > 0.5, so rank=1
}

#[test]
fn test_graph_metrics_reset() {
    let mut metrics = GraphMetrics::new();
    metrics.add_link_scores(&[0.8], &[0.2]);
    metrics.reset();
    assert!(metrics.pos_scores.is_empty());
    assert!(metrics.neg_scores.is_empty());
}

#[test]
fn test_gcl_empty_graph() {
    let mut rng = make_rng();
    let model = GraphCL::new(8, 16, 8, 0.2, 0.3, &mut rng);
    let g = GclGraph::new(vec![], vec![]);
    let enc = model.encode(&g);
    assert_eq!(enc.len(), 16);
    assert!(enc.iter().all(|&x| x == 0.0));
}

#[test]
fn test_graph_mae_sce_loss_zero_masked() {
    let mut rng = make_rng();
    let model = GraphMaeModel::new(8, 16, 0.3, MaeMaskStrategy::UniformRandom, &mut rng);
    let feats: Vec<Vec<f64>> = (0..3).map(|_| (0..8).map(|_| 0.5).collect()).collect();
    let loss = model.sce_loss(&feats, &feats, &[]);
    assert_eq!(loss, 0.0);
}

#[test]
fn test_fed_graph_oob_client() {
    let mut rng = make_rng();
    let mut fg = FedGraph::new(8, 16, 2, 0.01, 0.3, &mut rng);
    let feats: Vec<Vec<f64>> = (0..3).map(|_| (0..8).map(|_| 1.0).collect()).collect();
    let adj = simple_adj(3);
    let loss = fg.local_update(99, &feats, &adj, &mut rng);
    assert_eq!(loss, 0.0);
}
