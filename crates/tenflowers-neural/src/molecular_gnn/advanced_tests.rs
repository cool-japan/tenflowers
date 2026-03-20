use super::advanced::*;
use super::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

fn make_mol(seed: u64) -> MolGraph {
    let mut rng = StdRng::seed_from_u64(seed);
    MolGraph::from_smiles_stub(8, &mut rng)
}

fn make_conformer(n: usize, seed: u64) -> MolConformer {
    let mut rng = StdRng::seed_from_u64(seed);
    MolConformer::random_stub(n, &mut rng)
}

// ── MolConformer ──────────────────────────────────────────────────────────

#[test]
fn test_conformer_distance_positive() {
    let conf = make_conformer(5, 42);
    for i in 0..conf.positions.len() {
        for j in (i + 1)..conf.positions.len() {
            let d = conf.distance(i, j);
            assert!(d >= 0.0, "negative distance between {i} and {j}");
        }
    }
}

#[test]
fn test_conformer_distance_self_zero() {
    let conf = make_conformer(4, 7);
    for i in 0..conf.positions.len() {
        let d = conf.distance(i, i);
        assert!(d.abs() < 1e-10, "self-distance != 0 for atom {i}");
    }
}

#[test]
fn test_conformer_bond_angle_range() {
    let conf = make_conformer(6, 99);
    // Angles should be in [0, π]
    if conf.positions.len() >= 3 {
        let angle = conf.bond_angle(0, 1, 2);
        assert!((0.0..=std::f64::consts::PI).contains(&angle),
                "bond angle out of [0, π]: {angle}");
    }
}

#[test]
fn test_conformer_torsion_range() {
    let conf = make_conformer(6, 55);
    if conf.positions.len() >= 4 {
        let tor = conf.torsion_angle(0, 1, 2, 3);
        assert!((-std::f64::consts::PI..=std::f64::consts::PI).contains(&tor),
                "torsion out of [-π, π]: {tor}");
    }
}

#[test]
fn test_conformer_random_stub_structure() {
    let conf = make_conformer(8, 1);
    assert_eq!(conf.positions.len(), 8);
    assert_eq!(conf.atom_types.len(), 8);
    // Spanning chain edges: (0,1), (1,0), (1,2), (2,1), ...
    assert!(!conf.edges.is_empty());
}

// ── BesselBasis ──────────────────────────────────────────────────────────

#[test]
fn test_bessel_basis_length() {
    let basis = BesselBasis::new(8, 5.0);
    let enc = basis.encode(2.0);
    assert_eq!(enc.len(), 8, "Bessel encoding length mismatch");
}

#[test]
fn test_bessel_basis_zero_at_cutoff() {
    let basis = BesselBasis::new(8, 5.0);
    let enc = basis.encode(5.0); // at cutoff, envelope = 0
    for (i, &v) in enc.iter().enumerate() {
        assert!(v.abs() < 1e-9, "Bessel[{i}] != 0 at cutoff: {v}");
    }
}

#[test]
fn test_bessel_basis_nonneg_inside_cutoff() {
    let basis = BesselBasis::new(6, 4.0);
    let enc = basis.encode(1.0);
    // Not requiring non-negativity (can be signed), but should be finite
    for (i, &v) in enc.iter().enumerate() {
        assert!(v.is_finite(), "Bessel[{i}] is not finite: {v}");
    }
}

// ── DimeNetLayer ──────────────────────────────────────────────────────────

#[test]
fn test_dimenet_forward_shape() {
    let layer = DimeNetLayer::new(8, 16, 5.0, 42);
    let conf = make_conformer(5, 10);
    let h = layer.forward(&conf).expect("DimeNet forward failed");
    assert_eq!(h.len(), conf.positions.len(), "wrong number of embeddings");
    for hi in &h {
        assert_eq!(hi.len(), 16, "wrong embedding dim");
    }
}

#[test]
fn test_dimenet_forward_empty_error() {
    let layer = DimeNetLayer::new(8, 16, 5.0, 42);
    let conf = MolConformer::new(vec![], vec![], vec![]);
    let result = layer.forward(&conf);
    assert!(result.is_err(), "expected error on empty conformer");
}

#[test]
fn test_dimenet_all_finite() {
    let layer = DimeNetLayer::new(8, 16, 5.0, 1);
    let conf = make_conformer(4, 20);
    let h = layer.forward(&conf).expect("DimeNet forward failed");
    for (i, hi) in h.iter().enumerate() {
        for (j, &v) in hi.iter().enumerate() {
            assert!(v.is_finite(), "h[{i}][{j}] is not finite: {v}");
        }
    }
}

// ── ComENetLayer ─────────────────────────────────────────────────────────

#[test]
fn test_comenet_forward_shape() {
    let layer = ComENetLayer::new(8, 12, 5.0, 7);
    let conf = make_conformer(6, 30);
    let h = layer.forward(&conf).expect("ComENet forward failed");
    assert_eq!(h.len(), conf.positions.len());
    for hi in &h {
        assert_eq!(hi.len(), 12);
    }
}

#[test]
fn test_comenet_forward_empty_error() {
    let layer = ComENetLayer::new(4, 8, 5.0, 1);
    let conf = MolConformer::new(vec![], vec![], vec![]);
    assert!(layer.forward(&conf).is_err());
}

#[test]
fn test_comenet_all_finite() {
    let layer = ComENetLayer::new(6, 10, 6.0, 99);
    let conf = make_conformer(5, 77);
    let h = layer.forward(&conf).expect("ComENet forward failed");
    for hi in &h {
        for &v in hi {
            assert!(v.is_finite(), "ComENet output contains non-finite value");
        }
    }
}

// ── EquivariantMolNet ─────────────────────────────────────────────────────

#[test]
fn test_equivariant_forward_shape() {
    let net = EquivariantMolNet::new(16, 2, 42);
    let conf = make_conformer(5, 5);
    let (total, per_atom) = net.forward(&conf, 16).expect("EquivariantMolNet failed");
    assert_eq!(per_atom.len(), conf.positions.len());
    assert!(total.is_finite(), "total energy is not finite");
}

#[test]
fn test_equivariant_energy_sum_of_per_atom() {
    let net = EquivariantMolNet::new(8, 1, 13);
    let conf = make_conformer(4, 13);
    let (total, per_atom) = net.forward(&conf, 8).expect("EquivariantMolNet failed");
    let sum: f64 = per_atom.iter().sum();
    assert!((total - sum).abs() < 1e-10, "total energy != sum of per-atom: {total} vs {sum}");
}

#[test]
fn test_equivariant_forward_empty_error() {
    let net = EquivariantMolNet::new(8, 1, 1);
    let conf = MolConformer::new(vec![], vec![], vec![]);
    assert!(net.forward(&conf, 8).is_err());
}

// ── JunctionTreeVae ───────────────────────────────────────────────────────

#[test]
fn test_jtvae_encode_shape() {
    let vae = JunctionTreeVae::new(20, 16, 8, 42);
    let frag_feat = vec![0.1f64; 16];
    let (mean, logvar) = vae.encode(&frag_feat).expect("encode failed");
    assert_eq!(mean.len(), 8);
    assert_eq!(logvar.len(), 8);
}

#[test]
fn test_jtvae_reparameterize_shape() {
    let vae = JunctionTreeVae::new(10, 16, 8, 1);
    let mean = vec![0.0f64; 8];
    let logvar = vec![0.0f64; 8];
    let mut rng = StdRng::seed_from_u64(99);
    let z = vae.reparameterize(&mean, &logvar, &mut rng);
    assert_eq!(z.len(), 8);
}

#[test]
fn test_jtvae_decode_tree_nonempty() {
    let vae = JunctionTreeVae::new(10, 16, 8, 42);
    let z = vec![0.5f64; 8];
    let seq = vae.decode_tree(&z, 4);
    assert!(!seq.is_empty(), "decoded sequence should not be empty");
    assert!(seq.len() <= 4, "decoded sequence too long: {}", seq.len());
}

#[test]
fn test_jtvae_kl_loss_nonneg() {
    let vae = JunctionTreeVae::new(5, 8, 4, 1);
    // For mean=0, logvar=0: KL = -0.5*(1+0-0-1) = 0
    let mean = vec![0.0; 4];
    let logvar = vec![0.0; 4];
    let kl = vae.kl_loss(&mean, &logvar);
    assert!(kl.abs() < 1e-10, "KL at N(0,1) prior should be ~0: {kl}");
}

#[test]
fn test_jtvae_kl_loss_positive_for_nonzero_mean() {
    let vae = JunctionTreeVae::new(5, 8, 4, 1);
    let mean = vec![2.0; 4];
    let logvar = vec![0.0; 4];
    let kl = vae.kl_loss(&mean, &logvar);
    assert!(kl > 0.0, "KL should be positive for mean!=0: {kl}");
}

// ── GraphVae ──────────────────────────────────────────────────────────────

#[test]
fn test_graphvae_encode_decode() {
    let gvae = GraphVae::new(5, 44, 16, 42);
    let feat = vec![0.1f64; 5 * 44];
    let (mean, logvar) = gvae.encode(&feat);
    assert_eq!(mean.len(), 16);
    assert_eq!(logvar.len(), 16);
    let z = vec![0.0f64; 16];
    let (adj_probs, node_feats) = gvae.decode(&z);
    assert_eq!(adj_probs.len(), 5 * 5);
    assert_eq!(node_feats.len(), 5 * 44);
}

#[test]
fn test_graphvae_adj_probs_in_range() {
    let gvae = GraphVae::new(4, 8, 8, 1);
    let z = vec![0.1f64; 8];
    let (adj_probs, _) = gvae.decode(&z);
    for (i, &p) in adj_probs.iter().enumerate() {
        assert!((0.0..=1.0).contains(&p), "adj_prob[{i}] = {p} out of [0,1]");
    }
}

#[test]
fn test_graphvae_sample_adjacency_symmetric() {
    let gvae = GraphVae::new(4, 8, 8, 99);
    let z = vec![0.5f64; 8];
    let (adj_probs, _) = gvae.decode(&z);
    let mut rng = StdRng::seed_from_u64(42);
    let adj = gvae.sample_adjacency(&adj_probs, &mut rng);
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(adj[i][j], adj[j][i], "adjacency not symmetric at ({i},{j})");
        }
    }
}

#[test]
fn test_graphvae_matching_loss_nonneg() {
    let gvae = GraphVae::new(3, 8, 8, 1);
    let z = vec![0.0f64; 8];
    let (adj_probs, _) = gvae.decode(&z);
    let target = vec![vec![false, true, false], vec![true, false, true], vec![false, true, false]];
    let loss = gvae.graph_matching_loss(&target, &adj_probs);
    assert!(loss >= 0.0, "graph matching loss must be non-negative: {loss}");
    assert!(loss.is_finite(), "graph matching loss is not finite");
}

// ── MolecularFlowModel ────────────────────────────────────────────────────

#[test]
fn test_flow_forward_shape() {
    let flow = MolecularFlowModel::new(16, 3, 8, 42);
    let fp = vec![0.5f64; 16];
    let (z, log_det) = flow.forward(&fp).expect("flow forward failed");
    assert_eq!(z.len(), 16, "z shape mismatch");
    assert!(log_det.is_finite(), "log_det is not finite: {log_det}");
}

#[test]
fn test_flow_forward_short_error() {
    let flow = MolecularFlowModel::new(16, 2, 8, 1);
    let fp = vec![0.5f64; 1];
    assert!(flow.forward(&fp).is_err());
}

#[test]
fn test_flow_log_likelihood_finite() {
    let flow = MolecularFlowModel::new(16, 2, 8, 1);
    let z = vec![0.1f64; 16];
    let ll = flow.log_likelihood(&z, 0.5);
    assert!(ll.is_finite(), "log-likelihood is not finite: {ll}");
}

// ── MolDruglikenessFilter ─────────────────────────────────────────────────

#[test]
fn test_druglikeness_ro5_pass() {
    let filter = MolDruglikenessFilter::new();
    // Perfect drug-like molecule
    assert!(filter.passes_ro5(400.0, 3, 7, 3.0));
}

#[test]
fn test_druglikeness_ro5_two_violations_fail() {
    let filter = MolDruglikenessFilter::new();
    // Two violations: MW > 500 and logP > 5
    assert!(!filter.passes_ro5(600.0, 3, 7, 6.0));
}

#[test]
fn test_druglikeness_ro5_one_violation_pass() {
    let filter = MolDruglikenessFilter::new();
    // One violation (MW > 500) — Lipinski lenient → still passes
    assert!(filter.passes_ro5(520.0, 3, 7, 3.0));
}

#[test]
fn test_druglikeness_veber_pass() {
    let filter = MolDruglikenessFilter::new();
    assert!(filter.passes_veber(5, 80.0));
}

#[test]
fn test_druglikeness_veber_fail() {
    let filter = MolDruglikenessFilter::new();
    assert!(!filter.passes_veber(15, 150.0));
}

#[test]
fn test_druglikeness_assess() {
    let filter = MolDruglikenessFilter::new();
    let mol = make_mol(10);
    let report = filter.assess(&mol, 2.5, 4, 80.0);
    assert!(report.mw > 0.0, "MW should be positive");
    assert!(report.passes_ro5, "small mol should pass RO5");
    assert!(report.passes_veber, "small mol should pass Veber");
}

#[test]
fn test_estimate_hbd_hba() {
    let mol = make_mol(20);
    let hbd = MolDruglikenessFilter::estimate_hbd(&mol);
    let hba = MolDruglikenessFilter::estimate_hba(&mol);
    // HBD ≤ HBA (donors are subset of acceptors)
    assert!(hbd <= hba, "HBD > HBA: {hbd} > {hba}");
}

// ── AttentiveFP ───────────────────────────────────────────────────────────

#[test]
fn test_attentivefp_forward_shape() {
    let afp = AttentiveFp::new(44, 16, 4, 2, 42);
    let mol = make_mol(30);
    let out = afp.forward(&mol).expect("AttentiveFP forward failed");
    assert_eq!(out.len(), 4, "output dim mismatch");
}

#[test]
fn test_attentivefp_forward_empty_error() {
    let afp = AttentiveFp::new(44, 8, 2, 1, 1);
    let mol = MolGraph { atoms: vec![], bonds: vec![] };
    assert!(afp.forward(&mol).is_err());
}

#[test]
fn test_attentivefp_all_finite() {
    let afp = AttentiveFp::new(44, 16, 4, 2, 77);
    let mol = make_mol(40);
    let out = afp.forward(&mol).expect("AttentiveFP forward failed");
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "out[{i}] is not finite");
    }
}

// ── MolBert ───────────────────────────────────────────────────────────────

#[test]
fn test_molbert_encode_shape() {
    let bert = MolBert::new(20, 8, 2, 2, 42);
    let tokens = vec![1usize, 5, 3, 8, 12];
    let embs = bert.encode(&tokens).expect("MolBert encode failed");
    assert_eq!(embs.len(), tokens.len(), "embedding count mismatch");
    for e in &embs {
        assert_eq!(e.len(), 8, "embedding dim mismatch");
    }
}

#[test]
fn test_molbert_encode_empty_error() {
    let bert = MolBert::new(10, 8, 1, 1, 1);
    assert!(bert.encode(&[]).is_err());
}

#[test]
fn test_molbert_predict_masked_shape() {
    let bert = MolBert::new(20, 8, 1, 2, 42);
    let tokens = vec![0usize, 1, 2, 3, 4];
    let masked_preds = bert.predict_masked(&tokens, &[1, 3]).expect("predict_masked failed");
    assert_eq!(masked_preds.len(), 2, "should return predictions for 2 positions");
    for pred in &masked_preds {
        assert_eq!(pred.len(), 20, "prediction should have vocab_size logits");
    }
}

#[test]
fn test_molbert_predict_masked_oob_error() {
    let bert = MolBert::new(10, 8, 1, 1, 1);
    let tokens = vec![0usize, 1, 2];
    assert!(bert.predict_masked(&tokens, &[10]).is_err());
}

// ── MultiTaskMolNet ───────────────────────────────────────────────────────

#[test]
fn test_multitask_predict_n_tasks() {
    let config = MolMpnnConfig {
        node_dim: 44, edge_dim: 7, hidden_dim: 16, output_dim: 8, n_layers: 1,
        dropout_rate: 0.0, seed: 42,
    };
    let net = MultiTaskMolNet::new(config);
    let mol = make_mol(50);
    let preds = net.predict(&mol).expect("MultiTaskMolNet predict failed");
    assert_eq!(preds.len(), 4, "expected 4 task predictions");
}

#[test]
fn test_multitask_loss_zero_for_perfect() {
    let config = MolMpnnConfig {
        node_dim: 44, edge_dim: 7, hidden_dim: 8, output_dim: 4, n_layers: 1,
        dropout_rate: 0.0, seed: 1,
    };
    let net = MultiTaskMolNet::new(config);
    let preds = vec![1.0, 2.0, 3.0, 4.0];
    let targets = vec![1.0, 2.0, 3.0, 4.0];
    let loss = net.loss(&preds, &targets);
    assert!(loss.abs() < 1e-10, "MSE of perfect predictions != 0: {loss}");
}

// ── UncertaintyMolPredictor ───────────────────────────────────────────────

#[test]
fn test_uncertainty_predictor_shapes() {
    let config = MolMpnnConfig {
        node_dim: 44, edge_dim: 7, hidden_dim: 16, output_dim: 8, n_layers: 1,
        dropout_rate: 0.0, seed: 42,
    };
    let pred = UncertaintyMolPredictor::new(config, 0.1, 10);
    let mol = make_mol(60);
    let mut rng = StdRng::seed_from_u64(99);
    let (mean, var) = pred.predict_with_uncertainty(&mol, &mut rng).expect("uncertainty pred failed");
    assert_eq!(mean.len(), 4);
    assert_eq!(var.len(), 4);
}

#[test]
fn test_uncertainty_variance_nonneg() {
    let config = MolMpnnConfig {
        node_dim: 44, edge_dim: 7, hidden_dim: 8, output_dim: 4, n_layers: 1,
        dropout_rate: 0.2, seed: 13,
    };
    let pred = UncertaintyMolPredictor::new(config, 0.2, 5);
    let mol = make_mol(70);
    let mut rng = StdRng::seed_from_u64(7);
    let (_, var) = pred.predict_with_uncertainty(&mol, &mut rng).expect("uncertainty pred failed");
    for (i, &v) in var.iter().enumerate() {
        assert!(v >= 0.0, "variance[{i}] is negative: {v}");
    }
}

// ── ReactionGraph ─────────────────────────────────────────────────────────

#[test]
fn test_reaction_graph_atom_count() {
    let r1 = make_mol(80);
    let r2 = make_mol(81);
    let p1 = make_mol(82);
    let n_r = r1.n_atoms() + r2.n_atoms();
    let rxn = ReactionGraph::new(vec![r1, r2], vec![], vec![p1]);
    assert_eq!(rxn.n_reactant_atoms(), n_r);
}

#[test]
fn test_reaction_fingerprint_length() {
    let r = make_mol(90);
    let p = make_mol(91);
    let rxn = ReactionGraph::new(vec![r], vec![], vec![p]);
    let fp = rxn.reaction_fingerprint(256);
    assert_eq!(fp.len(), 256);
}

#[test]
fn test_reaction_fingerprint_finite() {
    let r = make_mol(92);
    let p = make_mol(93);
    let rxn = ReactionGraph::new(vec![r], vec![], vec![p]);
    let fp = rxn.reaction_fingerprint(128);
    for &v in &fp {
        assert!(v.is_finite());
    }
}

// ── LocalMapper ───────────────────────────────────────────────────────────

#[test]
fn test_local_mapper_similarity_matrix_shape() {
    let mapper = LocalMapper::new(0.5);
    let r = make_mol(100);
    let p = make_mol(101);
    let sim = mapper.similarity_matrix(&r, &p);
    assert_eq!(sim.len(), r.n_atoms());
    for row in &sim {
        assert_eq!(row.len(), p.n_atoms());
    }
}

#[test]
fn test_local_mapper_similarity_in_range() {
    let mapper = LocalMapper::new(0.5);
    let r = make_mol(102);
    let p = make_mol(103);
    let sim = mapper.similarity_matrix(&r, &p);
    for row in &sim {
        for &s in row {
            assert!((-1.0 - 1e-9..=1.0 + 1e-9).contains(&s), "cosine sim out of range: {s}");
        }
    }
}

#[test]
fn test_local_mapper_mapping_nonempty() {
    let mapper = LocalMapper::new(0.0); // threshold=0 → accept all
    let r = make_mol(104);
    let p = make_mol(104); // same molecule → should map many atoms
    let mapping = mapper.map_atoms(&r, &p);
    assert!(!mapping.is_empty(), "mapping should not be empty for identical molecules");
}

// ── RetrosynthesisPredictor ───────────────────────────────────────────────

#[test]
fn test_retrosynthesis_rank_all_templates() {
    let pred = RetrosynthesisPredictor::new(10, 256, 42);
    let mol = make_mol(110);
    let ranked = pred.rank_templates(&mol);
    assert_eq!(ranked.len(), 10, "should rank all templates");
    // Should be sorted descending
    for i in 0..ranked.len().saturating_sub(1) {
        assert!(ranked[i].1 >= ranked[i + 1].1, "templates not sorted by score");
    }
}

#[test]
fn test_retrosynthesis_top_k() {
    let pred = RetrosynthesisPredictor::new(20, 128, 1);
    let mol = make_mol(111);
    let top3 = pred.top_k_templates(&mol, 3);
    assert_eq!(top3.len(), 3);
    // Probabilities should sum to approximately 1
    let sum: f64 = top3.iter().map(|(_, p)| p).sum();
    assert!(sum > 0.0 && sum <= 1.0 + 1e-9, "probabilities should be in (0,1]: {sum}");
}

// ── ReactionYieldPredictor ────────────────────────────────────────────────

#[test]
fn test_yield_predictor_in_range() {
    let pred = ReactionYieldPredictor::new(256, 64, 42);
    let r = make_mol(120);
    let p = make_mol(121);
    let rxn = ReactionGraph::new(vec![r], vec![], vec![p]);
    let y = pred.predict(&rxn).expect("yield prediction failed");
    assert!((0.0..=100.0).contains(&y), "yield out of [0,100]: {y}");
}

#[test]
fn test_yield_predictor_empty_reactants_error() {
    let pred = ReactionYieldPredictor::new(64, 32, 1);
    let rxn = ReactionGraph::new(vec![], vec![], vec![]);
    assert!(pred.predict(&rxn).is_err());
}

#[test]
fn test_yield_predictor_loss_nonneg() {
    let pred = ReactionYieldPredictor::new(64, 32, 1);
    let loss = pred.loss(75.0, 80.0);
    assert!(loss >= 0.0 && loss.is_finite());
}

#[test]
fn test_yield_predictor_loss_perfect_zero() {
    let pred = ReactionYieldPredictor::new(64, 32, 1);
    let loss = pred.loss(85.0, 85.0);
    assert!(loss.abs() < 1e-10, "perfect prediction loss != 0: {loss}");
}
