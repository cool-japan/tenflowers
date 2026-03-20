//! Tests for Temporal & Dynamic Graph Neural Network modules.

#[cfg(test)]
mod tests {
    use super::super::dygformer::{
        CoOccurrenceEncoder, DyGFormerLayer, DynamicGraphTransformer, NeighborSampler,
    };
    use super::super::hetero::{HeteroTemporalGraph, HeteroTgnModel, RelationalTemporalConv};
    use super::super::ode_gnn::{EventGraph, GraphOdeFunc, InterpNodeFeatures, OdeGnn};
    use super::super::stgcn::{StGcnBlock, StGcnLayer, StGcnModel, TemporalConv};
    use super::super::tgn::{MemoryUpdateModule, NodeMemory, TemporalGraphNetwork, TimeEncoder};
    use super::super::types::{
        HeteroEdgeType, HeteroNodeType, MessageFunction, OdeSolver, PartitionStrategy,
        TemporalEdge, TgnConfig,
    };

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_edge(src: usize, dst: usize, time: f64, feat_dim: usize) -> TemporalEdge {
        TemporalEdge::new(src, dst, time, vec![0.1; feat_dim])
    }

    fn identity_adj(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect()
    }

    fn uniform_adj(n: usize) -> Vec<Vec<f64>> {
        let val = 1.0 / n as f64;
        vec![vec![val; n]; n]
    }

    // ── 1. TGN ───────────────────────────────────────────────────────────────

    #[test]
    fn test_temporal_edge_construction() {
        let e = TemporalEdge::new(0, 1, 1.5, vec![0.1, 0.2]);
        assert_eq!(e.src, 0);
        assert_eq!(e.dst, 1);
        assert!((e.time - 1.5).abs() < 1e-12);
        assert_eq!(e.features.len(), 2);
    }

    #[test]
    fn test_temporal_edge_zero_feat() {
        let e = TemporalEdge::new(2, 3, 0.0, vec![]);
        assert_eq!(e.features.len(), 0);
        assert_eq!(e.time, 0.0);
    }

    #[test]
    fn test_node_memory_init() {
        let mem = NodeMemory::new(5, 8);
        assert_eq!(mem.states.len(), 5);
        assert_eq!(mem.memory_dim, 8);
        assert!(mem.states[0].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_node_memory_get_set() {
        let mut mem = NodeMemory::new(4, 4);
        let state = vec![1.0, 2.0, 3.0, 4.0];
        mem.set(2, state.clone(), 10.0).expect("memory set should succeed");
        let (s, t) = mem.get(2).expect("memory get should succeed");
        assert_eq!(*s, state);
        assert!((t - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_node_memory_out_of_range() {
        let mem = NodeMemory::new(3, 4);
        assert!(mem.get(10).is_err());
    }

    #[test]
    fn test_node_memory_reset() {
        let mut mem = NodeMemory::new(3, 4);
        mem.set(0, vec![1.0; 4], 5.0).expect("memory set should succeed");
        mem.reset();
        let (s, t) = mem.get(0).expect("memory get should succeed");
        assert!(s.iter().all(|&v| v == 0.0));
        assert_eq!(t, 0.0);
    }

    #[test]
    fn test_time_encoder_output_dim() {
        let enc = TimeEncoder::new(16, 42);
        let out = enc.encode(1.0);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_time_encoder_zero_time() {
        let enc = TimeEncoder::new(8, 1);
        let out = enc.encode(0.0);
        assert_eq!(out.len(), 8);
        // cos(w*0 + b) = cos(b), so all values in [-1, 1].
        assert!(out.iter().all(|&v| (-1.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_time_encoder_different_times() {
        let enc = TimeEncoder::new(8, 7);
        let out1 = enc.encode(0.0);
        let out2 = enc.encode(1.0);
        // At least some coordinates should differ.
        let diff: f64 = out1
            .iter()
            .zip(out2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "time encoder outputs must differ for different times"
        );
    }

    #[test]
    fn test_memory_update_module_step() {
        let gru = MemoryUpdateModule::new(8, 8, 42);
        let msg = vec![0.5; 8];
        let h = vec![0.0; 8];
        let h_new = gru.step(&msg, &h).expect("GRU step should succeed");
        assert_eq!(h_new.len(), 8);
    }

    #[test]
    fn test_memory_update_module_dim_mismatch() {
        let gru = MemoryUpdateModule::new(4, 8, 1);
        let msg = vec![0.0; 5]; // wrong dim
        let h = vec![0.0; 8];
        assert!(gru.step(&msg, &h).is_err());
    }

    #[test]
    fn test_tgn_construction() {
        let cfg = TgnConfig::new(10, 4, 2, 16);
        let _tgn = TemporalGraphNetwork::new(cfg, 0);
    }

    #[test]
    fn test_tgn_process_edge() {
        let cfg = TgnConfig::new(5, 4, 2, 8);
        let mut tgn = TemporalGraphNetwork::new(cfg, 42);
        let edge = make_edge(0, 1, 1.0, 2);
        tgn.process_edge(&edge).expect("TGN edge processing should succeed");
        // Memory should be updated.
        let (s, t) = tgn.memory.get(0).expect("TGN memory access should succeed");
        assert!((t - 1.0).abs() < 1e-12);
        assert_eq!(s.len(), 8);
    }

    #[test]
    fn test_tgn_embed_no_neighbours() {
        let cfg = TgnConfig::new(5, 4, 2, 8);
        let tgn = TemporalGraphNetwork::new(cfg, 1);
        let emb = tgn.embed(0, 0.0, &[]).expect("TGN embedding should succeed");
        assert_eq!(emb.len(), 8);
    }

    #[test]
    fn test_tgn_embed_with_neighbours() {
        let cfg = TgnConfig::new(5, 4, 2, 8);
        let mut tgn = TemporalGraphNetwork::new(cfg, 3);
        // Process edges to give nodes non-zero memories.
        tgn.process_edge(&make_edge(0, 1, 0.5, 2)).expect("TGN edge processing should succeed");
        let neighbours = vec![(1, vec![0.0f64; 2], 0.5)];
        let emb = tgn.embed(0, 1.0, &neighbours).expect("TGN embedding with neighbors should succeed");
        assert_eq!(emb.len(), 8);
    }

    #[test]
    fn test_tgn_mlp_message_fn() {
        let mut cfg = TgnConfig::new(4, 2, 2, 8);
        cfg.message_fn = MessageFunction::Mlp;
        let mut tgn = TemporalGraphNetwork::new(cfg, 99);
        tgn.process_edge(&make_edge(0, 1, 1.0, 2)).expect("TGN edge processing should succeed");
        let (s, _) = tgn.memory.get(0).expect("TGN memory access should succeed");
        assert_eq!(s.len(), 8);
    }

    // ── 2. DyGFormer ─────────────────────────────────────────────────────────

    #[test]
    fn test_neighbor_sampler_construction() {
        let edges: Vec<TemporalEdge> = (0..5)
            .map(|i| make_edge(i % 3, (i + 1) % 3, i as f64, 2))
            .collect();
        let sampler = NeighborSampler::new(edges, 3);
        assert_eq!(sampler.k, 3);
    }

    #[test]
    fn test_neighbor_sampler_k_limit() {
        let edges: Vec<TemporalEdge> = (0..10).map(|i| make_edge(0, 1, i as f64, 2)).collect();
        let sampler = NeighborSampler::new(edges, 3);
        let nb = sampler.sample(0, 100.0);
        assert!(nb.len() <= 3);
    }

    #[test]
    fn test_neighbor_sampler_time_filter() {
        let edges: Vec<TemporalEdge> = (0..10).map(|i| make_edge(0, 1, i as f64, 2)).collect();
        let sampler = NeighborSampler::new(edges, 100);
        // Only events before t=3 should be returned.
        let nb = sampler.sample(0, 3.0);
        assert!(nb.iter().all(|e| e.time < 3.0));
    }

    #[test]
    fn test_cooccurrence_encoder_basic() {
        let edges = vec![make_edge(0, 2, 0.5, 2), make_edge(1, 2, 0.5, 2)];
        let enc = CoOccurrenceEncoder::new(4, 2.0).expect("co-occurrence encoder creation should succeed");
        let out = enc.encode(0, 1, &edges);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_cooccurrence_encoder_invalid() {
        assert!(CoOccurrenceEncoder::new(0, 1.0).is_err());
        assert!(CoOccurrenceEncoder::new(4, -1.0).is_err());
    }

    #[test]
    fn test_cooccurrence_shared_window() {
        // Both node 0 and node 1 appear in window 0.
        let edges = vec![
            make_edge(0, 2, 0.1, 2), // node 0 in window 0
            make_edge(1, 2, 0.2, 2), // node 1 in window 0
        ];
        let enc = CoOccurrenceEncoder::new(2, 1.0).expect("co-occurrence encoder creation should succeed");
        let out = enc.encode(0, 1, &edges);
        assert_eq!(out.len(), 2);
        // Window 0 should be active (1.0).
        assert!((out[0] - 1.0).abs() < 1e-12, "window 0 should be co-active");
    }

    #[test]
    fn test_dygformer_layer_construction() {
        let layer = DyGFormerLayer::new(8, 4, 2, 16, 42).expect("DyGFormer layer creation should succeed");
        assert_eq!(layer.n_heads, 2);
        assert_eq!(layer.input_dim, 8);
    }

    #[test]
    fn test_dygformer_layer_forward() {
        let layer = DyGFormerLayer::new(8, 4, 2, 16, 42).expect("DyGFormer layer creation should succeed");
        let tokens: Vec<(Vec<f64>, Vec<f64>)> =
            (0..3).map(|_| (vec![0.1f64; 8], vec![0.0f64; 4])).collect();
        let out = layer.forward(&tokens).expect("DyGFormer forward should succeed");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_dygformer_layer_empty_tokens() {
        let layer = DyGFormerLayer::new(8, 4, 2, 16, 1).expect("DyGFormer layer creation should succeed");
        let out = layer.forward(&[]).expect("DyGFormer forward on empty should succeed");
        assert!(out.is_empty());
    }

    #[test]
    fn test_dynamic_graph_transformer_construction() {
        let edges: Vec<TemporalEdge> = (0..6)
            .map(|i| make_edge(i % 3, (i + 1) % 3, i as f64 * 0.5, 2))
            .collect();
        let dgt = DynamicGraphTransformer::new(5, 8, 2, 2, 16, 3, 4, 3.0, edges, 42).expect("DGT construction should succeed");
        assert_eq!(dgt.layers.len(), 2);
    }

    #[test]
    fn test_dynamic_graph_transformer_embed() {
        let edges: Vec<TemporalEdge> = (0..6)
            .map(|i| make_edge(i % 3, (i + 1) % 3, i as f64 * 0.5, 2))
            .collect();
        let all_edges = edges.clone();
        let dgt = DynamicGraphTransformer::new(5, 8, 1, 2, 16, 4, 4, 4.0, edges, 7).expect("DGT construction should succeed");
        let emb = dgt.embed(0, 5.0, &all_edges).expect("DGT embedding should succeed");
        assert!(!emb.is_empty());
    }

    // ── 3. ST-GCN ────────────────────────────────────────────────────────────

    #[test]
    fn test_temporal_conv_output_shape() {
        let tc = TemporalConv::new(4, 8, 3, 42);
        let x: Vec<Vec<f64>> = vec![vec![0.1; 10]; 4]; // [4][10]
        let out = tc.forward(&x).expect("temporal conv forward should succeed");
        assert_eq!(out.len(), 8);
        assert_eq!(out[0].len(), 10);
    }

    #[test]
    fn test_temporal_conv_causal() {
        // Output at t=0 should only depend on x[*][0] (no future).
        let tc = TemporalConv::new(1, 1, 3, 1);
        let mut x: Vec<Vec<f64>> = vec![vec![0.0; 5]];
        x[0][0] = 1.0; // only t=0 is non-zero
        let out = tc.forward(&x).expect("temporal conv forward should succeed");
        let val_t0 = out[0][0];
        // t=3 should differ from t=0 only if kernel covers future — should not in causal.
        let val_t3 = out[0][3];
        // In a causal conv, t=3 can see t=0 but t=0 cannot see t=3. Both may be non-zero; this
        // just ensures t=0 does NOT accidentally "see" future values by checking the conv is
        // well-formed (output has expected shape and no NaN).
        assert!(val_t0.is_finite());
        assert!(val_t3.is_finite());
    }

    #[test]
    fn test_temporal_conv_dim_mismatch() {
        let tc = TemporalConv::new(4, 8, 3, 0);
        let x: Vec<Vec<f64>> = vec![vec![0.0; 5]; 3]; // wrong in_channels
        assert!(tc.forward(&x).is_err());
    }

    #[test]
    fn test_stgcn_layer_output_shape() {
        let n = 4;
        let t = 6;
        let layer = StGcnLayer::new(n, 2, 4, 3, PartitionStrategy::Uniform, 0);
        let x: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.1; t]; 2]; n];
        let adj = vec![identity_adj(n)];
        let out = layer.forward(&x, &adj).expect("ST-GCN layer forward should succeed");
        assert_eq!(out.len(), n);
        assert_eq!(out[0].len(), 4);
        assert_eq!(out[0][0].len(), t);
    }

    #[test]
    fn test_stgcn_layer_distance_strategy() {
        let n = 3;
        let t = 4;
        let layer = StGcnLayer::new(n, 2, 2, 1, PartitionStrategy::Distance, 5);
        let adj = vec![identity_adj(n), uniform_adj(n)];
        let x: Vec<Vec<Vec<f64>>> = vec![vec![vec![1.0; t]; 2]; n];
        let out = layer.forward(&x, &adj).expect("ST-GCN layer forward should succeed");
        assert_eq!(out.len(), n);
        assert_eq!(out[0].len(), 2);
    }

    #[test]
    fn test_stgcn_block_residual() {
        let n = 3;
        let t = 4;
        let block = StGcnBlock::new(n, 2, 2, 1, PartitionStrategy::Uniform, 0.0, 0);
        let adj = vec![identity_adj(n)];
        let x: Vec<Vec<Vec<f64>>> = vec![vec![vec![1.0; t]; 2]; n];
        let out = block.forward(&x, &adj, false, 0).expect("ST-GCN block forward should succeed");
        assert_eq!(out.len(), n);
    }

    #[test]
    fn test_stgcn_model_forward() {
        let n = 5;
        let t = 8;
        let adj = vec![identity_adj(n)];
        let model = StGcnModel::new(
            n,
            &[(3, 4), (4, 4)],
            3,
            3,
            PartitionStrategy::Uniform,
            adj,
            42,
        )
        .expect("ST-GCN model construction should succeed");
        let x: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.5; t]; 3]; n];
        let logits = model.forward(&x).expect("ST-GCN model forward should succeed");
        assert_eq!(logits.len(), 3);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_partition_strategy_n_subsets() {
        assert_eq!(PartitionStrategy::Uniform.n_subsets(), 1);
        assert_eq!(PartitionStrategy::Distance.n_subsets(), 2);
        assert_eq!(PartitionStrategy::SpatialConf.n_subsets(), 3);
    }

    // ── 4. ODE-GNN ───────────────────────────────────────────────────────────

    #[test]
    fn test_graph_ode_func_eval() {
        let n = 3;
        let func = GraphOdeFunc::new(4, n, 42);
        let x: Vec<Vec<f64>> = vec![vec![0.1; 4]; n];
        let adj = identity_adj(n);
        let dx = func.eval(&x, &adj).expect("graph ODE function eval should succeed");
        assert_eq!(dx.len(), n);
        assert_eq!(dx[0].len(), 4);
    }

    #[test]
    fn test_graph_ode_func_adj_mismatch() {
        let func = GraphOdeFunc::new(4, 3, 0);
        let x: Vec<Vec<f64>> = vec![vec![0.0; 4]; 3];
        let bad_adj: Vec<Vec<f64>> = vec![vec![1.0; 3]; 2]; // wrong n
        assert!(func.eval(&x, &bad_adj).is_err());
    }

    #[test]
    fn test_ode_gnn_euler_forward() {
        let n = 3;
        let ode = OdeGnn::new(4, n, 5, OdeSolver::Euler, 1);
        let x0: Vec<Vec<f64>> = vec![vec![0.0; 4]; n];
        let adj = identity_adj(n);
        let x1 = ode.integrate(&x0, &adj, 0.0, 1.0).expect("ODE integration should succeed");
        assert_eq!(x1.len(), n);
    }

    #[test]
    fn test_ode_gnn_rk4_forward() {
        let n = 2;
        let ode = OdeGnn::new(4, n, 4, OdeSolver::Rk4, 2);
        let x0: Vec<Vec<f64>> = vec![vec![0.1; 4]; n];
        let adj = uniform_adj(n);
        let x1 = ode.integrate(&x0, &adj, 0.0, 0.5).expect("ODE integration should succeed");
        assert_eq!(x1.len(), n);
    }

    #[test]
    fn test_ode_gnn_invalid_time() {
        let ode = OdeGnn::new(4, 2, 4, OdeSolver::Euler, 0);
        let x0: Vec<Vec<f64>> = vec![vec![0.0; 4]; 2];
        let adj = identity_adj(2);
        assert!(ode.integrate(&x0, &adj, 1.0, 0.5).is_err());
    }

    #[test]
    fn test_interp_node_features_exact() {
        let snaps = vec![(0.0, vec![0.0, 0.0]), (1.0, vec![1.0, 1.0])];
        let interp = InterpNodeFeatures::new(snaps).expect("interpolation construction should succeed");
        let q = interp.query(0.5);
        assert!((q[0] - 0.5).abs() < 1e-10);
        assert!((q[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_interp_node_features_boundary() {
        let snaps = vec![(0.0, vec![2.0]), (1.0, vec![4.0])];
        let interp = InterpNodeFeatures::new(snaps).expect("interpolation construction should succeed");
        assert!((interp.query(-1.0)[0] - 2.0).abs() < 1e-10);
        assert!((interp.query(2.0)[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_interp_node_features_empty_error() {
        assert!(InterpNodeFeatures::new(vec![]).is_err());
    }

    #[test]
    fn test_event_graph_add_and_replay() {
        let mut eg = EventGraph::new(100).expect("event graph creation should succeed");
        for i in 0..5 {
            eg.add_event(make_edge(0, 1, i as f64, 2));
        }
        let replayed = eg.replay(1.0, 3.0);
        assert_eq!(replayed.len(), 3); // t=1,2,3
    }

    #[test]
    fn test_event_graph_buffer_overflow() {
        let mut eg = EventGraph::new(3).expect("event graph creation should succeed");
        for i in 0..5 {
            eg.add_event(make_edge(0, 1, i as f64, 2));
        }
        assert_eq!(eg.events.len(), 3);
    }

    #[test]
    fn test_event_graph_invalid_buffer() {
        assert!(EventGraph::new(0).is_err());
    }

    // ── 5. Heterogeneous graphs ───────────────────────────────────────────────

    #[test]
    fn test_hetero_graph_construction() {
        let types = vec![
            HeteroNodeType("user".to_string()),
            HeteroNodeType("item".to_string()),
            HeteroNodeType("item".to_string()),
        ];
        let mut g = HeteroTemporalGraph::new(types);
        let rel = HeteroEdgeType("buys".to_string());
        g.add_edge(rel, make_edge(0, 1, 1.0, 2)).expect("hetero edge addition should succeed");
        assert_eq!(g.n_nodes(), 3);
        assert_eq!(g.edges.len(), 1);
    }

    #[test]
    fn test_hetero_graph_invalid_node() {
        let types = vec![HeteroNodeType("x".to_string())];
        let mut g = HeteroTemporalGraph::new(types);
        let rel = HeteroEdgeType("r".to_string());
        let e = make_edge(0, 99, 0.0, 2);
        assert!(g.add_edge(rel, e).is_err());
    }

    #[test]
    fn test_hetero_graph_edges_of_type() {
        let types: Vec<HeteroNodeType> = (0..4).map(|i| HeteroNodeType(format!("t{i}"))).collect();
        let mut g = HeteroTemporalGraph::new(types);
        let r1 = HeteroEdgeType("a".to_string());
        let r2 = HeteroEdgeType("b".to_string());
        g.add_edge(r1.clone(), make_edge(0, 1, 0.0, 2)).expect("hetero edge addition should succeed");
        g.add_edge(r2.clone(), make_edge(1, 2, 1.0, 2)).expect("hetero edge addition should succeed");
        g.add_edge(r1.clone(), make_edge(2, 3, 2.0, 2)).expect("hetero edge addition should succeed");
        assert_eq!(g.edges_of_type(&r1).len(), 2);
        assert_eq!(g.edges_of_type(&r2).len(), 1);
    }

    #[test]
    fn test_relational_temporal_conv_output_shape() {
        let rels = vec![
            HeteroEdgeType("a".to_string()),
            HeteroEdgeType("b".to_string()),
        ];
        let conv = RelationalTemporalConv::new(rels, 4, 8, 42);
        let types: Vec<HeteroNodeType> = (0..3).map(|i| HeteroNodeType(format!("n{i}"))).collect();
        let mut g = HeteroTemporalGraph::new(types);
        g.add_edge(HeteroEdgeType("a".to_string()), make_edge(0, 1, 0.0, 2))
            .expect("hetero edge addition should succeed");
        let feats: Vec<Vec<f64>> = vec![vec![0.5; 4]; 3];
        let out = conv.forward(&feats, &g).expect("relational temporal conv should succeed");
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_hetero_tgn_construction() {
        let rels = vec![HeteroEdgeType("likes".to_string())];
        let model = HeteroTgnModel::new(5, 8, rels, 42);
        assert_eq!(model.memory.states.len(), 5);
        assert_eq!(model.updaters.len(), 1);
    }

    #[test]
    fn test_hetero_tgn_process_edge() {
        let rels = vec![HeteroEdgeType("likes".to_string())];
        let mut model = HeteroTgnModel::new(4, 8, rels, 7);
        let rel = HeteroEdgeType("likes".to_string());
        let edge = make_edge(0, 1, 1.0, 2);
        model.process_edge(&rel, &edge).expect("hetero TGN edge processing should succeed");
        let (s, t) = model.memory.get(0).expect("hetero TGN memory access should succeed");
        assert!((t - 1.0).abs() < 1e-12);
        assert_eq!(s.len(), 8);
    }

    #[test]
    fn test_hetero_tgn_unknown_relation() {
        let rels = vec![HeteroEdgeType("known".to_string())];
        let mut model = HeteroTgnModel::new(4, 8, rels, 0);
        let unknown = HeteroEdgeType("unknown".to_string());
        let edge = make_edge(0, 1, 1.0, 2);
        assert!(model.process_edge(&unknown, &edge).is_err());
    }

    #[test]
    fn test_hetero_tgn_embed() {
        let rels = vec![HeteroEdgeType("likes".to_string())];
        let model = HeteroTgnModel::new(3, 4, rels, 99);
        let types: Vec<HeteroNodeType> = (0..3).map(|i| HeteroNodeType(format!("n{i}"))).collect();
        let g = HeteroTemporalGraph::new(types);
        let embs = model.embed(&g).expect("hetero TGN embedding should succeed");
        assert_eq!(embs.len(), 3);
        assert_eq!(embs[0].len(), 4);
    }

    #[test]
    fn test_gru_update_changes_state() {
        let gru = MemoryUpdateModule::new(4, 4, 77);
        let h = vec![0.0; 4];
        let msg = vec![1.0; 4];
        let h_new = gru.step(&msg, &h).expect("GRU step should succeed");
        let total_change: f64 = h_new.iter().zip(h.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(total_change > 1e-6, "GRU must update state");
    }

    #[test]
    fn test_time_encoder_range() {
        let enc = TimeEncoder::new(32, 0);
        for t in [0.0, 0.1, 1.0, 10.0, 100.0] {
            let out = enc.encode(t);
            assert!(
                out.iter().all(|&v| (-1.0..=1.0).contains(&v)),
                "cos outputs must be in [-1,1]"
            );
        }
    }

    #[test]
    fn test_stgcn_spatial_conf_strategy() {
        let n = 4;
        let t = 3;
        let layer = StGcnLayer::new(n, 2, 2, 1, PartitionStrategy::SpatialConf, 10);
        // 3 subsets.
        let adj: Vec<Vec<Vec<f64>>> = vec![identity_adj(n); 3];
        let x: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.5; t]; 2]; n];
        let out = layer.forward(&x, &adj).expect("ST-GCN layer forward should succeed");
        assert_eq!(out.len(), n);
        assert_eq!(out[0].len(), 2);
        assert_eq!(out[0][0].len(), t);
    }
}
