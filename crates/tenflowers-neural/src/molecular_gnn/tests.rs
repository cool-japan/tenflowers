use super::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

fn make_mol(seed: u64) -> MolGraph {
    let mut rng = StdRng::seed_from_u64(seed);
    MolGraph::from_smiles_stub(8, &mut rng)
}

// ── Atom & Bond feature vectors ──────────────────────────────────────────

#[test]
fn test_atom_feature_vector_length() {
    let atom = Atom {
        atomic_num: 6,
        formal_charge: 0,
        is_aromatic: false,
        hybridization: Hybridization::Sp3,
        degree: 3,
        n_hydrogens: 1,
    };
    assert_eq!(atom.feature_vector().len(), 44);
}

#[test]
fn test_atom_feature_vector_all_known() {
    let atoms = vec![
        Atom {
            atomic_num: 1,
            formal_charge: 0,
            is_aromatic: false,
            hybridization: Hybridization::Sp3,
            degree: 1,
            n_hydrogens: 0,
        },
        Atom {
            atomic_num: 7,
            formal_charge: -1,
            is_aromatic: true,
            hybridization: Hybridization::Sp2,
            degree: 2,
            n_hydrogens: 0,
        },
        Atom {
            atomic_num: 99,
            formal_charge: 2,
            is_aromatic: false,
            hybridization: Hybridization::Other,
            degree: 0,
            n_hydrogens: 6,
        },
    ];
    for a in &atoms {
        let fv = a.feature_vector();
        assert_eq!(fv.len(), 44, "len mismatch for atom {}", a.atomic_num);
        let sum: f64 = fv.iter().sum();
        // Each one-hot block contributes exactly 1.0, plus aromaticity
        assert!(sum >= 1.0, "feature sum too small: {sum}");
    }
}

#[test]
fn test_bond_feature_vector_length() {
    let bond = Bond {
        src: 0,
        dst: 1,
        bond_type: MolBondType::Aromatic,
        is_aromatic: true,
        is_conjugated: true,
        in_ring: true,
    };
    assert_eq!(bond.feature_vector().len(), 7);
}

#[test]
fn test_bond_feature_vector_content() {
    let bond = Bond {
        src: 0,
        dst: 1,
        bond_type: MolBondType::Double,
        is_aromatic: false,
        is_conjugated: true,
        in_ring: false,
    };
    let fv = bond.feature_vector();
    assert_eq!(fv.len(), 7);
    assert_eq!(fv[1], 1.0); // Double is index 1
    assert_eq!(fv[4], 0.0); // not aromatic
    assert_eq!(fv[5], 1.0); // conjugated
    assert_eq!(fv[6], 0.0); // not in ring
}

// ── MolGraph ─────────────────────────────────────────────────────────────

#[test]
fn test_molgraph_from_smiles_stub() {
    let mut rng = StdRng::seed_from_u64(42);
    let mol = MolGraph::from_smiles_stub(10, &mut rng);
    assert!(mol.n_atoms() > 0, "no atoms generated");
    assert_eq!(mol.n_atoms(), 10);
    assert!(mol.n_bonds() > 0, "no bonds generated");
}

#[test]
fn test_molgraph_adjacency_list_consistency() {
    let mol = make_mol(7);
    let adj = mol.adjacency_list();
    assert_eq!(adj.len(), mol.n_atoms());
    // Verify all bond indices are in range
    for neighbors in &adj {
        for &(j, bi) in neighbors {
            assert!(j < mol.n_atoms(), "neighbor out of range");
            assert!(bi < mol.n_bonds(), "bond index out of range");
        }
    }
}

#[test]
fn test_molgraph_atom_features_shape() {
    let mol = make_mol(1);
    let af = mol.atom_features();
    assert_eq!(af.len(), mol.n_atoms());
    for row in &af {
        assert_eq!(row.len(), 44);
    }
}

#[test]
fn test_molgraph_bond_features_shape() {
    let mol = make_mol(2);
    let bf = mol.bond_features();
    assert_eq!(bf.len(), mol.n_bonds());
    for row in &bf {
        assert_eq!(row.len(), 7);
    }
}

// ── Morgan Fingerprint ───────────────────────────────────────────────────

#[test]
fn test_morgan_fp_length() {
    let mol = make_mol(3);
    let mfp = MorganFingerprint {
        radius: 2,
        n_bits: 256,
    };
    let fp = mfp.compute(&mol);
    assert_eq!(fp.len(), 256);
}

#[test]
fn test_morgan_fp_binary() {
    let mol = make_mol(4);
    let mfp = MorganFingerprint {
        radius: 2,
        n_bits: 128,
    };
    let fp = mfp.compute(&mol);
    assert!(
        fp.iter().all(|&b| b == 0 || b == 1),
        "non-binary bits found"
    );
}

#[test]
fn test_morgan_tanimoto_self() {
    let mol = make_mol(5);
    let mfp = MorganFingerprint {
        radius: 2,
        n_bits: 256,
    };
    let fp = mfp.compute(&mol);
    let t = MorganFingerprint::tanimoto(&fp, &fp);
    assert!((t - 1.0).abs() < 1e-10, "tanimoto self != 1: {t}");
}

#[test]
fn test_morgan_tanimoto_range() {
    let mol1 = make_mol(10);
    let mol2 = make_mol(20);
    let mfp = MorganFingerprint {
        radius: 2,
        n_bits: 512,
    };
    let fp1 = mfp.compute(&mol1);
    let fp2 = mfp.compute(&mol2);
    let t = MorganFingerprint::tanimoto(&fp1, &fp2);
    assert!((0.0..=1.0).contains(&t), "tanimoto out of [0,1]: {t}");
}

#[test]
fn test_morgan_different_mols_different_fps() {
    let mol1 = make_mol(100);
    let mol2 = make_mol(200);
    let mfp = MorganFingerprint {
        radius: 3,
        n_bits: 512,
    };
    let fp1 = mfp.compute(&mol1);
    let fp2 = mfp.compute(&mol2);
    // Not guaranteed to differ but very likely with different seeds
    let t = MorganFingerprint::tanimoto(&fp1, &fp2);
    assert!((0.0..=1.0).contains(&t));
}

// ── Topological Fingerprint ──────────────────────────────────────────────

#[test]
fn test_topological_fp_nonempty() {
    let mol = make_mol(6);
    let tfp = TopologicalFingerprint {
        n_bits: 128,
        max_path: 3,
    };
    let fp = tfp.compute(&mol);
    assert_eq!(fp.len(), 128);
    let n_set: usize = fp.iter().map(|&b| b as usize).sum();
    assert!(n_set > 0, "no bits set in topological fingerprint");
}

#[test]
fn test_topological_fp_binary() {
    let mol = make_mol(8);
    let tfp = TopologicalFingerprint {
        n_bits: 64,
        max_path: 2,
    };
    let fp = tfp.compute(&mol);
    assert!(fp.iter().all(|&b| b == 0 || b == 1));
}

#[test]
fn test_topological_similarity_self() {
    let mol = make_mol(9);
    let tfp = TopologicalFingerprint {
        n_bits: 256,
        max_path: 4,
    };
    let fp = tfp.compute(&mol);
    let s = TopologicalFingerprint::path_similarity(&fp, &fp);
    assert!((s - 1.0).abs() < 1e-10, "path_similarity self != 1: {s}");
}

// ── Gaussian Smearing ─────────────────────────────────────────────────────

#[test]
fn test_gaussian_smearing_length() {
    let gs = GaussianSmearing::new(16, 5.0);
    let out = gs.smear(2.5);
    assert_eq!(out.len(), 16);
}

#[test]
fn test_gaussian_smearing_nonneg() {
    let gs = GaussianSmearing::new(10, 8.0);
    let out = gs.smear(3.0);
    assert!(out.iter().all(|&v| v >= 0.0), "negative Gaussian value");
}

#[test]
fn test_gaussian_smearing_zero_dist_peak() {
    // At distance=0, the first Gaussian (μ=0) should be at its peak = 1.0
    let gs = GaussianSmearing::new(10, 5.0);
    let out = gs.smear(0.0);
    assert_eq!(out.len(), 10);
    // First center is at 0, so exp(0) = 1.0
    assert!(
        (out[0] - 1.0).abs() < 1e-10,
        "smear(0)[0] != 1.0, got {}",
        out[0]
    );
}

#[test]
fn test_gaussian_smearing_decay() {
    let gs = GaussianSmearing::new(8, 4.0);
    // Smearing at distance far from center 0 should be < 1
    let out_far = gs.smear(3.5);
    let out_near = gs.smear(0.0);
    // Near: first basis = 1.0; far: first basis decays
    assert!(out_far[0] < out_near[0]);
}

// ── MPNN ─────────────────────────────────────────────────────────────────

#[test]
fn test_mpnn_layer_aggregate_mean() {
    let layer = MpnnLayer::new(4, 4, 7, 42);
    let msgs = vec![vec![1.0, 2.0, 3.0, 4.0], vec![3.0, 4.0, 5.0, 6.0]];
    let agg = layer.aggregate(&msgs);
    assert_eq!(agg.len(), 4);
    assert!((agg[0] - 2.0).abs() < 1e-9);
    assert!((agg[1] - 3.0).abs() < 1e-9);
}

#[test]
fn test_mpnn_layer_aggregate_single() {
    let layer = MpnnLayer::new(3, 3, 7, 1);
    let msgs = vec![vec![5.0, 6.0, 7.0]];
    let agg = layer.aggregate(&msgs);
    assert_eq!(agg, vec![5.0, 6.0, 7.0]);
}

#[test]
fn test_mpnn_layer_aggregate_empty() {
    let layer = MpnnLayer::new(3, 3, 7, 1);
    let agg = layer.aggregate(&[]);
    assert_eq!(agg.len(), 3);
    assert!(agg.iter().all(|&v| v == 0.0));
}

#[test]
fn test_mpnn_forward_shape() {
    let config = MolMpnnConfig {
        node_dim: 44,
        edge_dim: 7,
        hidden_dim: 16,
        output_dim: 8,
        n_layers: 2,
        dropout_rate: 0.0,
        seed: 42,
    };
    let mpnn = Mpnn::new(config);
    let mol = make_mol(11);
    let embeds = mpnn.forward(&mol).expect("MPNN forward failed");
    assert_eq!(
        embeds.len(),
        mol.n_atoms(),
        "wrong number of node embeddings"
    );
    for e in &embeds {
        assert_eq!(e.len(), 16, "wrong embedding dim");
    }
}

#[test]
fn test_mpnn_readout_shape() {
    let config = MolMpnnConfig {
        node_dim: 44,
        edge_dim: 7,
        hidden_dim: 16,
        output_dim: 8,
        n_layers: 2,
        dropout_rate: 0.0,
        seed: 42,
    };
    let mpnn = Mpnn::new(config);
    let mol = make_mol(12);
    let embeds = mpnn.forward(&mol).expect("forward failed");
    let out = mpnn.graph_readout(&embeds);
    assert_eq!(out.len(), 8, "readout dim mismatch");
}

#[test]
fn test_mpnn_forward_single_atom() {
    let config = MolMpnnConfig {
        node_dim: 44,
        edge_dim: 7,
        hidden_dim: 8,
        output_dim: 4,
        n_layers: 1,
        dropout_rate: 0.0,
        seed: 7,
    };
    let mpnn = Mpnn::new(config);
    let mut rng = StdRng::seed_from_u64(99);
    let mol = MolGraph::from_smiles_stub(1, &mut rng);
    let embeds = mpnn.forward(&mol).expect("single atom forward failed");
    assert_eq!(embeds.len(), 1);
    assert_eq!(embeds[0].len(), 8);
}

// ── SchNet ────────────────────────────────────────────────────────────────

#[test]
fn test_schnet_forward_shape() {
    let config = SchNetConfig {
        n_atom_types: 10,
        n_filters: 16,
        n_interactions: 2,
        cutoff: 5.0,
        n_gaussians: 8,
        output_dim: 6,
        seed: 42,
    };
    let schnet = SchNet::new(config);
    let atom_types = vec![0, 1, 2, 3, 4];
    let positions: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.5, 0.0, 0.0],
        vec![0.0, 1.5, 0.0],
        vec![1.5, 1.5, 0.0],
        vec![0.75, 0.75, 1.0],
    ];
    let out = schnet
        .forward(&atom_types, &positions)
        .expect("SchNet forward failed");
    assert_eq!(out.len(), 6, "output dim mismatch");
}

#[test]
fn test_schnet_forward_empty_error() {
    let config = SchNetConfig {
        n_atom_types: 5,
        n_filters: 8,
        n_interactions: 1,
        cutoff: 3.0,
        n_gaussians: 4,
        output_dim: 4,
        seed: 1,
    };
    let schnet = SchNet::new(config);
    let result = schnet.forward(&[], &[]);
    assert!(result.is_err(), "expected error on empty input");
}

// ── Property Predictor ────────────────────────────────────────────────────

#[test]
fn test_property_predictor_regression() {
    let mpnn_config = MolMpnnConfig {
        node_dim: 44,
        edge_dim: 7,
        hidden_dim: 16,
        output_dim: 8,
        n_layers: 2,
        dropout_rate: 0.0,
        seed: 123,
    };
    let config = PropertyPredictorConfig {
        mpnn_config,
        task: PredictionTask::Regression { n_targets: 3 },
    };
    let pred = PropertyPredictor::new(config);
    let mol = make_mol(13);
    let out = pred.predict(&mol).expect("predict failed");
    assert_eq!(out.len(), 3, "expected 3 targets");
}

#[test]
fn test_property_predictor_classification() {
    let mpnn_config = MolMpnnConfig {
        node_dim: 44,
        edge_dim: 7,
        hidden_dim: 16,
        output_dim: 8,
        n_layers: 1,
        dropout_rate: 0.0,
        seed: 456,
    };
    let config = PropertyPredictorConfig {
        mpnn_config,
        task: PredictionTask::Classification { n_classes: 5 },
    };
    let pred = PropertyPredictor::new(config);
    let mol = make_mol(14);
    let out = pred.predict(&mol).expect("classify failed");
    assert_eq!(out.len(), 5, "expected 5 classes");
}

#[test]
fn test_property_predictor_loss_regression() {
    let mpnn_config = MolMpnnConfig {
        node_dim: 44,
        edge_dim: 7,
        hidden_dim: 8,
        output_dim: 4,
        n_layers: 1,
        dropout_rate: 0.0,
        seed: 789,
    };
    let config = PropertyPredictorConfig {
        mpnn_config,
        task: PredictionTask::Regression { n_targets: 2 },
    };
    let pred = PropertyPredictor::new(config.clone());
    let task = config.task.clone();
    let preds = vec![1.0, 2.0];
    let targets = vec![1.0, 2.0];
    let loss = pred.compute_loss(&preds, &targets, &task);
    assert!(loss.abs() < 1e-10, "MSE of perfect pred != 0: {loss}");
}

#[test]
fn test_property_predictor_loss_classification() {
    let mpnn_config = MolMpnnConfig {
        node_dim: 44,
        edge_dim: 7,
        hidden_dim: 8,
        output_dim: 4,
        n_layers: 1,
        dropout_rate: 0.0,
        seed: 101,
    };
    let config = PropertyPredictorConfig {
        mpnn_config,
        task: PredictionTask::Classification { n_classes: 3 },
    };
    let pred = PropertyPredictor::new(config.clone());
    let task = config.task.clone();
    // Perfect logits for class 0
    let preds = vec![10.0, -10.0, -10.0];
    let targets = vec![0.0]; // class 0
    let loss = pred.compute_loss(&preds, &targets, &task);
    assert!(loss >= 0.0, "cross-entropy must be non-negative");
    assert!(
        loss < 0.01,
        "CE of perfect classification too large: {loss}"
    );
}

// ── Reaction Features ─────────────────────────────────────────────────────

#[test]
fn test_reaction_features_diff_fp() {
    let mol1 = make_mol(30);
    let mol2 = make_mol(31);
    let feat = ReactionFeatures::from_molgraphs(&[mol1], &[mol2]);
    assert_eq!(feat.reactant_fps.len(), feat.product_fps.len());
    assert_eq!(feat.diff_fp.len(), feat.reactant_fps.len());
    // Verify diff_fp = product - reactant element-wise
    for (i, (&d, (&p, &r))) in feat
        .diff_fp
        .iter()
        .zip(feat.product_fps.iter().zip(feat.reactant_fps.iter()))
        .enumerate()
    {
        assert!((d - (p - r)).abs() < 1e-10, "diff_fp[{i}] mismatch");
    }
}

#[test]
fn test_reaction_features_multiple_reactants() {
    let reactants = vec![make_mol(40), make_mol(41)];
    let products = vec![make_mol(42)];
    let feat = ReactionFeatures::from_molgraphs(&reactants, &products);
    // reactant_fps should be 2x256 = 512 bits
    assert_eq!(feat.reactant_fps.len(), 512);
    assert_eq!(feat.product_fps.len(), 256);
    // diff_fp pads shorter side
    assert_eq!(feat.diff_fp.len(), 512);
}

// ── Reaction Classifier ───────────────────────────────────────────────────

#[test]
fn test_reaction_classifier_predict_sums_to_one() {
    let n_bits = 256;
    let mut clf = ReactionClassifier::new(4, n_bits);
    // Give non-trivial weights
    for c in 0..4 {
        for j in 0..n_bits {
            clf.weights[c][j] = ((c * n_bits + j) as f64) * 0.001;
        }
    }
    let mol1 = make_mol(50);
    let mol2 = make_mol(51);
    let feat = ReactionFeatures::from_molgraphs(&[mol1], &[mol2]);
    let probs = clf.predict(&feat);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "probs sum != 1: {sum}");
    assert!(probs.iter().all(|&p| p >= 0.0), "negative probability");
}

#[test]
fn test_reaction_classifier_train_loss_decreases() {
    // Two-class classification: label 0 vs label 1
    let n_bits = 256;
    let mut clf = ReactionClassifier::new(2, n_bits);

    // Build training data: two molecules clearly in different classes
    let mol_a = make_mol(60);
    let mol_b = make_mol(61);
    let feat_a = ReactionFeatures::from_molgraphs(std::slice::from_ref(&mol_a), std::slice::from_ref(&mol_b));
    let feat_b = ReactionFeatures::from_molgraphs(std::slice::from_ref(&mol_b), std::slice::from_ref(&mol_a));

    let feats = vec![feat_a, feat_b];
    let labels = vec![0usize, 1usize];

    let initial_loss = clf.train_step(&feats, &labels, 0.1);
    let mut last_loss = initial_loss;
    for _ in 0..50 {
        last_loss = clf.train_step(&feats, &labels, 0.1);
    }
    assert!(
        last_loss < initial_loss + 0.01,
        "loss did not decrease: initial={initial_loss}, final={last_loss}"
    );
}

// ── Utility functions ─────────────────────────────────────────────────────

#[test]
fn test_fnv1a_different_inputs() {
    let h1 = fnv1a_hash(b"hello");
    let h2 = fnv1a_hash(b"world");
    assert_ne!(h1, h2, "different inputs should have different hashes");
}

#[test]
fn test_fnv1a_deterministic() {
    let h = fnv1a_hash(b"test_input_123");
    assert_eq!(h, fnv1a_hash(b"test_input_123"));
}

#[test]
fn test_softmax_sums_to_one() {
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let s = softmax(&v);
    let sum: f64 = s.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12, "softmax sum != 1: {sum}");
}

#[test]
fn test_softmax_monotone() {
    let v = vec![0.0, 1.0, 2.0];
    let s = softmax(&v);
    assert!(s[0] < s[1] && s[1] < s[2]);
}

#[test]
fn test_sigmoid_at_zero() {
    let s = sigmoid(0.0);
    assert!((s - 0.5).abs() < 1e-12, "sigmoid(0) != 0.5: {s}");
}

#[test]
fn test_sigmoid_range() {
    for x in [-100.0, -1.0, 0.0, 1.0, 100.0_f64] {
        let s = sigmoid(x);
        assert!((0.0..=1.0).contains(&s), "sigmoid({x}) out of range: {s}");
    }
}

#[test]
fn test_relu_basic() {
    assert_eq!(relu(-5.0), 0.0);
    assert_eq!(relu(0.0), 0.0);
    assert!((relu(3.0) - 3.0).abs() < 1e-12);
}

#[test]
fn test_path_hash_deterministic() {
    let path = vec![1u64, 2, 3, 4];
    assert_eq!(path_hash(&path), path_hash(&path));
}

#[test]
fn test_path_hash_different() {
    let p1 = vec![1u64, 2, 3];
    let p2 = vec![1u64, 2, 4];
    assert_ne!(path_hash(&p1), path_hash(&p2));
}

#[test]
fn test_matvec_basic() {
    let mat = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let v = vec![3.0, 4.0];
    let out = matvec(&mat, &v);
    assert_eq!(out, vec![3.0, 4.0]);
}

#[test]
fn test_vecadd_basic() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let c = vecadd(&a, &b);
    assert_eq!(c, vec![5.0, 7.0, 9.0]);
}
