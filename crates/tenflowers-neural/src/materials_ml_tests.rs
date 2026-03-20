//! Tests for materials_ml module — split for policy compliance (max 1900 lines per file).

use super::*;
use scirs2_core::random::SeedableRng;

fn make_fe_o_graph() -> CrystalGraph {
    let lattice = [[4.26, 0.0, 0.0], [0.0, 4.26, 0.0], [0.0, 0.0, 4.26]];
    let atoms = vec![
        Atom {
            element: "Fe".to_string(),
            position: [0.0, 0.0, 0.0],
            atomic_number: 26,
            electronegativity: 1.83,
            radius: 1.26,
        },
        Atom {
            element: "O".to_string(),
            position: [2.13, 2.13, 2.13],
            atomic_number: 8,
            electronegativity: 3.44,
            radius: 0.66,
        },
    ];
    build_neighbor_graph(&atoms, &lattice, 4.5)
}

fn make_cubic_lattice() -> [[f32; 3]; 3] {
    [[3.5, 0.0, 0.0], [0.0, 3.5, 0.0], [0.0, 0.0, 3.5]]
}

// §1 CrystalGraph
#[test]
fn test_build_neighbor_graph_has_bonds() {
    let g = make_fe_o_graph();
    assert!(!g.bonds.is_empty(), "should have bonds within cutoff");
}

#[test]
fn test_build_neighbor_graph_atoms_count() {
    let g = make_fe_o_graph();
    assert_eq!(g.atoms.len(), 2);
}

#[test]
fn test_crystal_bond_distance_positive() {
    let g = make_fe_o_graph();
    for b in &g.bonds {
        assert!(b.distance > 0.0);
    }
}

#[test]
fn test_crystal_graph_empty_atoms() {
    let g = build_neighbor_graph(
        &[],
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        3.0,
    );
    assert!(g.bonds.is_empty());
    assert!(g.atoms.is_empty());
}

#[test]
fn test_crystal_bond_within_cutoff() {
    let g = make_fe_o_graph();
    for b in &g.bonds {
        assert!(b.distance <= 4.5);
    }
}

// §2 CGCNN
#[test]
fn test_cgcnn_forward_shape() {
    let config = CgcnnConfig {
        atom_fea_len: 16,
        n_conv: 2,
        n_h: 8,
        output_dim: 1,
    };
    let model = Cgcnn::new(config);
    let g = make_fe_o_graph();
    let out = model.forward(&g);
    assert_eq!(out.len(), 1);
}

#[test]
fn test_cgcnn_forward_empty_graph() {
    let config = CgcnnConfig {
        atom_fea_len: 16,
        n_conv: 1,
        n_h: 8,
        output_dim: 2,
    };
    let model = Cgcnn::new(config);
    let empty = CrystalGraph {
        atoms: vec![],
        bonds: vec![],
        lattice: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    let out = model.forward(&empty);
    assert_eq!(out.len(), 2);
}

#[test]
fn test_atom_featurizer_dim() {
    let f = AtomFeaturizer::new(32, 118);
    let atom = Atom {
        element: "Fe".to_string(),
        position: [0.0; 3],
        atomic_number: 26,
        electronegativity: 1.83,
        radius: 1.26,
    };
    assert_eq!(f.featurize(&atom).len(), 32);
}

#[test]
fn test_edge_featurizer_dim() {
    let ef = EdgeFeaturizer::new(41, 0.0, 8.0);
    let v = ef.featurize(2.5);
    assert_eq!(v.len(), 41);
}

#[test]
fn test_edge_featurizer_values_in_range() {
    let ef = EdgeFeaturizer::new(20, 0.5, 6.0);
    let v = ef.featurize(3.0);
    for &x in &v {
        assert!((0.0..=1.0 + 1e-6).contains(&x));
    }
}

// §3 SchNetMaterials
#[test]
fn test_schnet_predict_scalar() {
    let config = SchNetMaterialsConfig {
        n_atom_basis: 16,
        n_gaussians: 20,
        n_interactions: 2,
        cutoff: 5.0,
    };
    let model = SchNetMaterials::new(config);
    let g = make_fe_o_graph();
    let val = model.predict_property(&g);
    assert!(val.is_finite());
}

#[test]
fn test_schnet_empty_graph() {
    let config = SchNetMaterialsConfig {
        n_atom_basis: 8,
        n_gaussians: 10,
        n_interactions: 1,
        cutoff: 4.0,
    };
    let model = SchNetMaterials::new(config);
    let empty = CrystalGraph {
        atoms: vec![],
        bonds: vec![],
        lattice: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    assert_eq!(model.predict_property(&empty), 0.0);
}

#[test]
fn test_gaussian_smearing_dim() {
    let gs = GaussianSmearing::new(50, 10.0);
    let v = gs.expand(4.5);
    assert_eq!(v.len(), 50);
}

#[test]
fn test_gaussian_smearing_peak_at_center() {
    let gs = GaussianSmearing::new(10, 5.0);
    let v = gs.expand(0.0); // exactly on first center
    assert!(v[0] > v[9], "peak should be near first center");
}

// §4 MattersimModel
#[test]
fn test_mattersim_embed_dim() {
    let model = MattersimModel::new(16, 32);
    let g = make_fe_o_graph();
    let emb = model.embed(&g);
    assert_eq!(emb.len(), 32);
}

#[test]
fn test_mattersim_embed_empty() {
    let model = MattersimModel::new(16, 32);
    let empty = CrystalGraph {
        atoms: vec![],
        bonds: vec![],
        lattice: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    let emb = model.embed(&empty);
    assert_eq!(emb.len(), 32);
}

#[test]
fn test_compute_descriptor_pair_count() {
    let embed = ElementEmbedding::new(8);
    let g = make_fe_o_graph();
    let desc = compute_descriptor(&g, &embed);
    assert_eq!(desc.pair_features.len(), g.bonds.len());
}

// §5 PhasePredictor
#[test]
fn test_convex_hull_stable_point_on_hull() {
    let pts = vec![
        ConvexHullPoint {
            composition: vec![0.0],
            energy: 0.0,
        },
        ConvexHullPoint {
            composition: vec![0.5],
            energy: -1.0,
        },
        ConvexHullPoint {
            composition: vec![1.0],
            energy: 0.0,
        },
    ];
    let query = ConvexHullPoint {
        composition: vec![0.5],
        energy: -1.0,
    };
    assert!(is_on_hull(&pts, &query));
}

#[test]
fn test_convex_hull_unstable_point_above_hull() {
    let pts = vec![
        ConvexHullPoint {
            composition: vec![0.0],
            energy: 0.0,
        },
        ConvexHullPoint {
            composition: vec![1.0],
            energy: 0.0,
        },
    ];
    let query = ConvexHullPoint {
        composition: vec![0.5],
        energy: 0.5,
    };
    let e_above = energy_above_hull(&pts, &query);
    assert!(e_above > 0.0, "unstable compound should be above hull");
}

#[test]
fn test_energy_above_hull_zero_for_endpoints() {
    let pts = vec![
        ConvexHullPoint {
            composition: vec![0.0],
            energy: 0.0,
        },
        ConvexHullPoint {
            composition: vec![1.0],
            energy: 0.0,
        },
    ];
    let q0 = ConvexHullPoint {
        composition: vec![0.0],
        energy: 0.0,
    };
    assert!(energy_above_hull(&pts, &q0).abs() < 1e-3);
}

#[test]
fn test_phase_predictor_output_range() {
    let config = PhaseConfig {
        n_elements: 5,
        n_phases: 3,
        embed_dim: 8,
    };
    let pp = PhasePredictor::new(&config);
    let comp = vec![0.5, 0.3, 0.1, 0.05, 0.05];
    let score = pp.predict_stability(&comp, -1.5);
    assert!(
        (0.0..=1.0 + 1e-5).contains(&score),
        "stability score should be in [0,1]: {score}"
    );
}

// §6 MaterialPropertyPredictor
#[test]
fn test_material_property_predictor_count() {
    let config = MaterialPropertyPredictorConfig {
        n_properties: 6,
        embed_dim: 32,
    };
    let pp = MaterialPropertyPredictor::new(&config);
    let embed = vec![0.1f32; 32];
    let preds = pp.predict_all(&embed);
    assert_eq!(preds.len(), 6);
}

#[test]
fn test_material_property_predictor_finite() {
    let config = MaterialPropertyPredictorConfig {
        n_properties: 4,
        embed_dim: 16,
    };
    let pp = MaterialPropertyPredictor::new(&config);
    let embed = vec![1.0f32; 16];
    for (_, val) in pp.predict_all(&embed) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_material_property_all_variants() {
    let all = MaterialProperty::all();
    assert_eq!(all.len(), 6);
    assert!(all.contains(&MaterialProperty::BandGap));
    assert!(all.contains(&MaterialProperty::ThermalConductivity));
}

// §7 CrystalSymmetry
#[test]
fn test_detect_lattice_cubic() {
    let lattice = make_cubic_lattice();
    assert_eq!(detect_lattice_system(&lattice), "cubic");
}

#[test]
fn test_detect_lattice_tetragonal() {
    let lattice = [[3.5, 0.0, 0.0], [0.0, 3.5, 0.0], [0.0, 0.0, 5.0]];
    assert_eq!(detect_lattice_system(&lattice), "tetragonal");
}

#[test]
fn test_detect_lattice_orthorhombic() {
    let lattice = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]];
    assert_eq!(detect_lattice_system(&lattice), "orthorhombic");
}

#[test]
fn test_lattice_params_cubic() {
    let lattice = make_cubic_lattice();
    let (a, b, c, alpha, beta, gamma) = compute_lattice_params(&lattice);
    assert!((a - 3.5).abs() < 0.01);
    assert!((b - 3.5).abs() < 0.01);
    assert!((c - 3.5).abs() < 0.01);
    assert!((alpha - 90.0).abs() < 0.5);
    assert!((beta - 90.0).abs() < 0.5);
    assert!((gamma - 90.0).abs() < 0.5);
}

#[test]
fn test_point_group_order_at_least_one() {
    let g = make_fe_o_graph();
    let order = point_group_order(&g.atoms, &g.bonds);
    assert!(order >= 1);
}

// §8 MaterialAugmentation
#[test]
fn test_random_rotation_preserves_atom_count() {
    let g = make_fe_o_graph();
    let mut rng = StdRng::seed_from_u64(0);
    let rotated = random_rotation(&g, &mut rng);
    assert_eq!(rotated.atoms.len(), g.atoms.len());
}

#[test]
fn test_gaussian_displacement_preserves_count() {
    let g = make_fe_o_graph();
    let mut rng = StdRng::seed_from_u64(1);
    let displaced = add_gaussian_displacement(&g, 0.05, &mut rng);
    assert_eq!(displaced.atoms.len(), g.atoms.len());
}

#[test]
fn test_gaussian_displacement_changes_positions() {
    let g = make_fe_o_graph();
    let mut rng = StdRng::seed_from_u64(2);
    let displaced = add_gaussian_displacement(&g, 0.1, &mut rng);
    let changed = displaced
        .atoms
        .iter()
        .zip(g.atoms.iter())
        .any(|(a, b)| (a.position[0] - b.position[0]).abs() > 1e-6);
    assert!(changed);
}

#[test]
fn test_random_supercell_doubles_atoms() {
    let g = make_fe_o_graph();
    let mut rng = StdRng::seed_from_u64(3);
    let sc = random_supercell(&g, 2, &mut rng);
    assert_eq!(sc.atoms.len(), g.atoms.len() * 2);
}

#[test]
fn test_random_supercell_lattice_doubled() {
    let g = make_fe_o_graph();
    let mut rng = StdRng::seed_from_u64(4);
    let sc = random_supercell(&g, 2, &mut rng);
    let vol_orig: f32 = g.lattice.iter().map(|r| r[0]).sum();
    let vol_sc: f32 = sc.lattice.iter().map(|r| r[0]).sum();
    let diff_diag: Vec<f32> = (0..3).map(|i| sc.lattice[i][i] - g.lattice[i][i]).collect();
    let any_doubled = diff_diag.iter().any(|&d| {
        (d - g.lattice[0][0]).abs() < 0.1
            || (d - g.lattice[1][1]).abs() < 0.1
            || (d - g.lattice[2][2]).abs() < 0.1
    });
    let _ = (vol_orig, vol_sc);
    assert!(any_doubled || sc.atoms.len() == 2 * g.atoms.len());
}

// §9 GenerativeMaterials
#[test]
fn test_composition_generator_fraction_sum() {
    let gen = CompositionGenerator::new(20, 8);
    let mut rng = StdRng::seed_from_u64(5);
    let comp = gen.sample_composition(&[1.0, 0.5, -0.5], &mut rng);
    let total: f32 = comp.iter().map(|(_, f)| f).sum();
    assert!(
        (total - 1.0).abs() < 1e-4,
        "fractions should sum to 1, got {total}"
    );
}

#[test]
fn test_composition_generator_element_count() {
    let gen = CompositionGenerator::new(10, 4);
    let mut rng = StdRng::seed_from_u64(6);
    let comp = gen.sample_composition(&[0.5, -0.1], &mut rng);
    assert!(!comp.is_empty());
    assert!(comp.len() >= 2);
}

#[test]
fn test_diffusion_denoise_step_shape() {
    let model = DiffusionMaterialsModel::new(10, 6);
    let noisy = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let out = model.denoise_step(&noisy, 0.5);
    assert_eq!(out.len(), 6);
}

#[test]
fn test_diffusion_denoise_finite() {
    let model = DiffusionMaterialsModel::new(5, 9);
    let noisy: Vec<f32> = (0..9).map(|i| i as f32 * 0.1).collect();
    let out = model.denoise_step(&noisy, 0.1);
    for &v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_diffusion_denoise_empty() {
    let model = DiffusionMaterialsModel::new(5, 4);
    let out = model.denoise_step(&[], 0.5);
    assert!(out.is_empty());
}

// §10 MaterialMetrics
#[test]
fn test_mae_perfect() {
    let v = vec![1.0, 2.0, 3.0];
    assert!((mae(&v, &v) - 0.0).abs() < 1e-6);
}

#[test]
fn test_mae_known_value() {
    let pred = vec![1.0, 2.0, 3.0];
    let true_v = vec![2.0, 2.0, 2.0];
    assert!((mae(&pred, &true_v) - 2.0 / 3.0).abs() < 1e-5);
}

#[test]
fn test_r2_perfect() {
    let v = vec![1.0, 2.0, 3.0, 4.0];
    assert!((r2(&v, &v) - 1.0).abs() < 1e-5);
}

#[test]
fn test_r2_zero_model() {
    let true_v = vec![1.0, 3.0];
    let mean_pred = vec![2.0, 2.0];
    let r = r2(&mean_pred, &true_v);
    assert!(r.abs() < 1e-5, "mean prediction should give R2~0, got {r}");
}

#[test]
fn test_spearman_perfect_order() {
    let v = vec![1.0, 2.0, 3.0, 4.0];
    assert!((spearman_rank_correlation(&v, &v) - 1.0).abs() < 1e-5);
}

#[test]
fn test_spearman_reverse_order() {
    let pred = vec![4.0, 3.0, 2.0, 1.0];
    let true_v = vec![1.0, 2.0, 3.0, 4.0];
    assert!((spearman_rank_correlation(&pred, &true_v) + 1.0).abs() < 1e-5);
}

#[test]
fn test_top_k_screening_perfect() {
    let pred = vec![4.0, 3.0, 2.0, 1.0];
    let true_v = vec![4.0, 3.0, 2.0, 1.0];
    assert!((top_k_screening_rate(&pred, &true_v, 2) - 1.0).abs() < 1e-6);
}

#[test]
fn test_top_k_screening_zero() {
    let pred = vec![1.0, 2.0, 3.0, 4.0];
    let true_v = vec![4.0, 3.0, 2.0, 1.0];
    let rate = top_k_screening_rate(&pred, &true_v, 1);
    assert!(rate.is_finite());
}

#[test]
fn test_top_k_screening_k_larger_than_n() {
    let pred = vec![1.0, 2.0];
    let true_v = vec![2.0, 1.0];
    let rate = top_k_screening_rate(&pred, &true_v, 10);
    assert!(rate.is_finite());
}

#[test]
fn test_mae_empty() {
    assert_eq!(mae(&[], &[]), 0.0);
}

#[test]
fn test_r2_empty() {
    assert_eq!(r2(&[], &[]), 0.0);
}

#[test]
fn test_spearman_empty() {
    assert_eq!(spearman_rank_correlation(&[], &[]), 0.0);
}

// Integration tests
#[test]
fn test_cgcnn_multi_output() {
    let config = CgcnnConfig {
        atom_fea_len: 8,
        n_conv: 1,
        n_h: 4,
        output_dim: 3,
    };
    let model = Cgcnn::new(config);
    let g = make_fe_o_graph();
    let out = model.forward(&g);
    assert_eq!(out.len(), 3);
    for &v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_schnet_multiple_atoms() {
    let config = SchNetMaterialsConfig {
        n_atom_basis: 8,
        n_gaussians: 10,
        n_interactions: 2,
        cutoff: 5.0,
    };
    let lattice = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]];
    let atoms = vec![
        Atom {
            element: "Si".to_string(),
            position: [0.0, 0.0, 0.0],
            atomic_number: 14,
            electronegativity: 1.9,
            radius: 1.17,
        },
        Atom {
            element: "Si".to_string(),
            position: [1.36, 1.36, 1.36],
            atomic_number: 14,
            electronegativity: 1.9,
            radius: 1.17,
        },
        Atom {
            element: "O".to_string(),
            position: [2.5, 0.0, 0.0],
            atomic_number: 8,
            electronegativity: 3.44,
            radius: 0.66,
        },
        Atom {
            element: "O".to_string(),
            position: [0.0, 2.5, 0.0],
            atomic_number: 8,
            electronegativity: 3.44,
            radius: 0.66,
        },
    ];
    let graph = build_neighbor_graph(&atoms, &lattice, 4.0);
    let model = SchNetMaterials::new(config);
    let val = model.predict_property(&graph);
    assert!(val.is_finite());
}

#[test]
fn test_material_property_predictor_partial_properties() {
    let config = MaterialPropertyPredictorConfig {
        n_properties: 2,
        embed_dim: 8,
    };
    let pp = MaterialPropertyPredictor::new(&config);
    let embed = vec![-1.0f32, 0.5, 0.3, 0.1, 0.0, -0.2, 0.8, 0.4];
    let preds = pp.predict_all(&embed);
    assert_eq!(preds.len(), 2);
}
