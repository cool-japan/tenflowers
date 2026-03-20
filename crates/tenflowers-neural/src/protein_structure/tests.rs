use super::*;

// ── PsAminoAcid ──────────────────────────────────────────────────────────

#[test]
fn test_ps_amino_acid_from_char_valid() {
    assert_eq!(PsAminoAcid::from_char('A').expect("from_char should succeed"), PsAminoAcid::Ala);
    assert_eq!(PsAminoAcid::from_char('a').expect("from_char should succeed"), PsAminoAcid::Ala);
    assert_eq!(PsAminoAcid::from_char('V').expect("from_char should succeed"), PsAminoAcid::Val);
    assert_eq!(PsAminoAcid::from_char('W').expect("from_char should succeed"), PsAminoAcid::Trp);
}

#[test]
fn test_ps_amino_acid_from_char_unk_gap() {
    assert_eq!(PsAminoAcid::from_char('X').expect("from_char should succeed"), PsAminoAcid::Unk);
    assert_eq!(PsAminoAcid::from_char('-').expect("from_char should succeed"), PsAminoAcid::Gap);
    assert_eq!(PsAminoAcid::from_char('.').expect("from_char should succeed"), PsAminoAcid::Gap);
}

#[test]
fn test_ps_amino_acid_from_char_invalid() {
    assert!(PsAminoAcid::from_char('Z').is_err());
    assert!(PsAminoAcid::from_char('1').is_err());
}

#[test]
fn test_ps_amino_acid_roundtrip() {
    for aa in [
        PsAminoAcid::Ala,
        PsAminoAcid::Arg,
        PsAminoAcid::Trp,
        PsAminoAcid::Val,
    ] {
        let c = aa.to_char();
        assert_eq!(PsAminoAcid::from_char(c).expect("from_char roundtrip should succeed"), aa);
    }
}

#[test]
fn test_ps_amino_acid_hydrophobicity() {
    assert!(PsAminoAcid::Ile.hydrophobicity() > 0.0);
    assert!(PsAminoAcid::Arg.hydrophobicity() < 0.0);
    assert_eq!(PsAminoAcid::Gap.hydrophobicity(), 0.0);
}

#[test]
fn test_ps_amino_acid_charge() {
    assert_eq!(PsAminoAcid::Arg.charge(), 1.0);
    assert_eq!(PsAminoAcid::Lys.charge(), 1.0);
    assert_eq!(PsAminoAcid::Asp.charge(), -1.0);
    assert_eq!(PsAminoAcid::Ala.charge(), 0.0);
}

#[test]
fn test_ps_amino_acid_volume() {
    assert!(PsAminoAcid::Trp.volume_angstrom3() > PsAminoAcid::Gly.volume_angstrom3());
    assert_eq!(PsAminoAcid::Gap.volume_angstrom3(), 0.0);
}

#[test]
fn test_ps_amino_acid_to_idx() {
    assert_eq!(PsAminoAcid::Ala.to_idx(), 0);
    assert_eq!(PsAminoAcid::Val.to_idx(), 19);
    assert_eq!(PsAminoAcid::Unk.to_idx(), 20);
    assert_eq!(PsAminoAcid::Gap.to_idx(), 20);
}

// ── ProteinSequence ───────────────────────────────────────────────────────

#[test]
fn test_protein_sequence_from_str() {
    let seq = ProteinSequence::from_str("ACGX123").unwrap_err();
    assert!(matches!(seq, PsError::InvalidSequence(_)));
    let seq = ProteinSequence::from_str("ACDEFGHIKLM").expect("protein sequence parsing should succeed");
    assert_eq!(seq.length(), 11);
}

#[test]
fn test_protein_sequence_one_hot() {
    let seq = ProteinSequence::from_str("AV").expect("protein sequence parsing should succeed");
    let oh = seq.one_hot_encode();
    assert_eq!(oh.len(), 2);
    assert_eq!(oh[0].len(), 21);
    assert_eq!(oh[0][0], 1.0); // Ala at index 0
    assert_eq!(oh[1][19], 1.0); // Val at index 19
    assert_eq!(oh[0].iter().sum::<f64>(), 1.0);
}

#[test]
fn test_protein_sequence_embedding() {
    let seq = ProteinSequence::from_str("ACDE").expect("protein sequence parsing should succeed");
    let emb = seq.embedding_features();
    assert_eq!(emb.len(), 4);
    assert_eq!(emb[0].len(), 4);
    // Ala hydrophobicity normalized
    let expected_h = PsAminoAcid::Ala.hydrophobicity() / 5.0;
    assert!((emb[0][0] - expected_h).abs() < 1e-10);
}

#[test]
fn test_protein_sequence_length() {
    let seq = ProteinSequence::from_str("MKTLLLTLVVVTIVCLDLGAVAGNPCNGPK").expect("protein sequence parsing should succeed");
    assert_eq!(seq.length(), 30);
}

// ── Residue3D ─────────────────────────────────────────────────────────────

#[test]
fn test_residue3d_creation() {
    let r = Residue3D::new(
        [0.0, 0.0, 0.0],
        [1.46, 0.0, 0.0],
        [2.5, 1.2, 0.0],
        [3.5, 1.2, 0.0],
    );
    assert_eq!(r.ca, [1.46, 0.0, 0.0]);
}

#[test]
fn test_peptide_bond_length() {
    let r = Residue3D::new([0.0; 3], [1.46, 0.0, 0.0], [2.5, 0.0, 0.0], [3.5, 0.0, 0.0]);
    let next_n = [3.82, 0.0, 0.0];
    let bond = r.peptide_bond_length(&next_n);
    assert!((bond - 1.32).abs() < 0.1);
}

#[test]
fn test_backbone_torsion_no_neighbors() {
    let r = Residue3D::new([0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]);
    let (phi, psi) = r.backbone_torsion_phi_psi(None, None);
    assert_eq!(phi, 0.0);
    assert_eq!(psi, 0.0);
}

#[test]
fn test_backbone_torsion_with_neighbors() {
    let r = Residue3D::new(
        [1.46, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    let prev_c = [2.5, 1.0, 0.0];
    let next_n = [-2.0, 0.5, 0.0];
    let (phi, psi) = r.backbone_torsion_phi_psi(Some(&prev_c), Some(&next_n));
    assert!(phi.is_finite());
    assert!(psi.is_finite());
}

// ── ProteinStructure ──────────────────────────────────────────────────────

fn make_linear_structure(n: usize) -> ProteinStructure {
    let residues = (0..n)
        .map(|i| {
            let x = i as f64 * 3.8;
            Residue3D::new(
                [x - 1.0, 0.0, 0.0],
                [x, 0.0, 0.0],
                [x + 1.0, 0.0, 0.0],
                [x + 1.0, 1.2, 0.0],
            )
        })
        .collect();
    ProteinStructure::new(residues)
}

#[test]
fn test_distogram_shape() {
    let s = make_linear_structure(5);
    let d = s.compute_distogram();
    assert_eq!(d.len(), 5);
    assert_eq!(d[0].len(), 5);
    assert_eq!(d[0][0], 0.0);
    assert!((d[0][1] - 3.8).abs() < 1e-10);
}

#[test]
fn test_distogram_symmetry() {
    let s = make_linear_structure(4);
    let d = s.compute_distogram();
    for i in 0..4 {
        for j in 0..4 {
            assert!((d[i][j] - d[j][i]).abs() < 1e-10);
        }
    }
}

#[test]
fn test_rmsd_identical() {
    let s = make_linear_structure(5);
    let rmsd = s.rmsd(&s).expect("rmsd computation should succeed");
    assert!(rmsd < 1e-10);
}

#[test]
fn test_rmsd_shifted() {
    let s1 = make_linear_structure(5);
    // Apply position-dependent distortion (not a pure translation, which
    // centroid alignment would remove entirely).
    let s2 = ProteinStructure::new(
        s1.residues
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let offset = (i as f64) * 0.5;
                Residue3D::new(r.n, [r.ca[0] + offset, r.ca[1], r.ca[2]], r.c, r.o)
            })
            .collect(),
    );
    let rmsd = s1.rmsd(&s2).expect("rmsd computation should succeed");
    assert!(rmsd > 0.0);
}

#[test]
fn test_tm_score_identical() {
    let s = make_linear_structure(30);
    let tm = s.tm_score(&s).expect("tm_score computation should succeed");
    assert!(tm > 0.8);
}

#[test]
fn test_tm_score_empty_error() {
    let s1 = make_linear_structure(10);
    let s2 = ProteinStructure::new(vec![]);
    assert!(s1.tm_score(&s2).is_err());
}

#[test]
fn test_secondary_structure_assignment() {
    let s = make_linear_structure(5);
    let ss = s.compute_secondary_structure();
    assert_eq!(ss.len(), 5);
    for label in &ss {
        assert!(["H", "E", "C"].contains(&label.as_str()));
    }
}

// ── MsaEncoder ────────────────────────────────────────────────────────────

#[test]
fn test_msa_encoder_encode_single_seq() {
    let enc = MsaEncoder::new(4, 8, 2, 42);
    let seqs = vec![ProteinSequence::from_str("ACDE").expect("protein sequence parsing should succeed")];
    let out = enc.encode_msa(&seqs);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), 4); // n_pos
}

#[test]
fn test_msa_encoder_encode_multi_seq() {
    let enc = MsaEncoder::new(4, 8, 2, 42);
    let seqs = vec![
        ProteinSequence::from_str("ACDE").expect("protein sequence parsing should succeed"),
        ProteinSequence::from_str("FGHI").expect("protein sequence parsing should succeed"),
    ];
    let out = enc.encode_msa(&seqs);
    assert_eq!(out.len(), 2);
}

#[test]
fn test_row_attention_shape() {
    let enc = MsaEncoder::new(4, 8, 2, 1);
    let msa = vec![vec![vec![0.1_f64; 4]; 5], vec![vec![0.2_f64; 4]; 5]];
    let out = enc.row_attention(&msa);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].len(), 5);
    assert_eq!(out[0][0].len(), 8);
}

#[test]
fn test_column_attention_shape() {
    let enc = MsaEncoder::new(4, 8, 2, 2);
    let msa = vec![
        vec![vec![0.1_f64; 4]; 3],
        vec![vec![0.2_f64; 4]; 3],
        vec![vec![0.3_f64; 4]; 3],
    ];
    let out = enc.column_attention(&msa);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].len(), 3);
}

// ── PairwiseRepresentation ────────────────────────────────────────────────

#[test]
fn test_outer_product_mean_shape() {
    let pr = PairwiseRepresentation::new(16, 10);
    let msa: Vec<Vec<Vec<f64>>> = vec![vec![vec![1.0_f64; 4]; 3], vec![vec![0.5_f64; 4]; 3]];
    let pair = pr.outer_product_mean(&msa);
    assert_eq!(pair.len(), 3);
    assert_eq!(pair[0].len(), 3);
}

#[test]
fn test_outer_product_mean_diagonal() {
    let pr = PairwiseRepresentation::new(16, 11);
    let msa = vec![vec![vec![1.0_f64; 2]; 2]];
    let pair = pr.outer_product_mean(&msa);
    // [0][0] and [1][1] should be equal (same sequence position patterns)
    assert_eq!(pair[0][0], pair[1][1]);
}

#[test]
fn test_triangle_update_preserves_shape() {
    let pr = PairwiseRepresentation::new(4, 20);
    let pair = vec![vec![vec![0.1_f64; 4]; 3]; 3];
    let updated = pr.update_pair_with_triangles(&pair);
    assert_eq!(updated.len(), 3);
    assert_eq!(updated[0].len(), 3);
    assert_eq!(updated[0][0].len(), 4);
}

// ── InvariantPointAttention ───────────────────────────────────────────────

#[test]
fn test_ipa_forward_shape() {
    let ipa = InvariantPointAttention::new(8, 4, 2, 42);
    let single: Vec<Vec<f64>> = vec![vec![0.1_f64; 8]; 5];
    let pair: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0_f64; 4]; 5]; 5];
    let id_rot = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let frames: Vec<([f64; 9], [f64; 3])> = (0..5)
        .map(|i| (id_rot, [i as f64 * 3.8, 0.0, 0.0]))
        .collect();
    let out = ipa.forward(&single, &pair, &frames);
    assert_eq!(out.len(), 5);
    assert_eq!(out[0].len(), 8);
}

#[test]
fn test_ipa_forward_empty() {
    let ipa = InvariantPointAttention::new(8, 4, 2, 1);
    let out = ipa.forward(&[], &[], &[]);
    assert!(out.is_empty());
}

// ── StructureModule ───────────────────────────────────────────────────────

#[test]
fn test_structure_module_forward() {
    let sm = StructureModule::new(8, 4, 42);
    let single: Vec<Vec<f64>> = vec![vec![0.1_f64; 8]; 4];
    let pair: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0_f64; 4]; 4]; 4];
    let id = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let frames: Vec<([f64; 9], [f64; 3])> =
        (0..4).map(|i| (id, [i as f64 * 3.8, 0.0, 0.0])).collect();
    let (backbone, angles) = sm.forward(&single, &pair, &frames);
    assert_eq!(backbone.len(), 4);
    assert_eq!(angles.len(), 4);
    assert_eq!(angles[0].len(), 8);
}

#[test]
fn test_structure_module_torsion_range() {
    let sm = StructureModule::new(8, 4, 7);
    let single: Vec<Vec<f64>> = vec![vec![0.5_f64; 8]; 3];
    let pair = vec![vec![vec![0.0_f64; 4]; 3]; 3];
    let id = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let frames: Vec<([f64; 9], [f64; 3])> =
        (0..3).map(|i| (id, [i as f64 * 3.8, 0.0, 0.0])).collect();
    let (_, angles) = sm.forward(&single, &pair, &frames);
    for angle_set in &angles {
        for &a in angle_set {
            assert!((-std::f64::consts::PI..=std::f64::consts::PI).contains(&a));
        }
    }
}

// ── AlphaFoldLite ─────────────────────────────────────────────────────────

#[test]
fn test_alphafold_lite_predict() {
    let af = AlphaFoldLite::new(8, 4, 42);
    let seqs = vec![ProteinSequence::from_str("ACDE").expect("protein sequence parsing should succeed")];
    let structure = af.predict(&seqs).expect("alphafold prediction should succeed");
    assert_eq!(structure.len(), 4);
}

#[test]
fn test_alphafold_lite_empty_sequences_error() {
    let af = AlphaFoldLite::new(8, 4, 1);
    assert!(af.predict(&[]).is_err());
}

#[test]
fn test_alphafold_lite_coords_finite() {
    let af = AlphaFoldLite::new(8, 4, 10);
    let seqs = vec![ProteinSequence::from_str("MKTLL").expect("protein sequence parsing should succeed")];
    let structure = af.predict(&seqs).expect("alphafold prediction should succeed");
    for r in &structure.residues {
        assert!(r.ca.iter().all(|v| v.is_finite()));
    }
}

// ── ProteinEvoformer ──────────────────────────────────────────────────────

#[test]
fn test_evoformer_forward_shape() {
    let evo = ProteinEvoformer::new(4, 4, 42);
    let msa = vec![vec![vec![0.1_f64; 4]; 3]; 2];
    let pair = vec![vec![vec![0.0_f64; 4]; 3]; 3];
    let (msa_out, pair_out) = evo.forward(&msa, &pair);
    assert_eq!(msa_out.len(), 2);
    assert_eq!(msa_out[0].len(), 3);
    assert_eq!(pair_out.len(), 3);
}

#[test]
fn test_evoformer_forward_empty() {
    let evo = ProteinEvoformer::new(4, 4, 1);
    let (msa_out, pair_out) = evo.forward(&[], &[]);
    assert!(msa_out.is_empty());
    assert!(pair_out.is_empty());
}

#[test]
fn test_evoformer_pair_output_finite() {
    let evo = ProteinEvoformer::new(4, 4, 99);
    let msa = vec![vec![vec![0.5_f64; 4]; 4]; 2];
    let pair = vec![vec![vec![0.1_f64; 4]; 4]; 4];
    let (_, pair_out) = evo.forward(&msa, &pair);
    for row in &pair_out {
        for cell in row {
            for &v in cell {
                assert!(v.is_finite(), "pair output contains non-finite value: {v}");
            }
        }
    }
}

// ── ContactMapPredictor ───────────────────────────────────────────────────

#[test]
fn test_contact_predictor_shape() {
    let cmp = ContactMapPredictor::new(4, 16, 42);
    let emb = vec![vec![0.1_f64; 4]; 5];
    let contacts = cmp.predict_contacts(&emb);
    assert_eq!(contacts.len(), 5);
    assert_eq!(contacts[0].len(), 5);
}

#[test]
fn test_contact_predictor_probabilities() {
    let cmp = ContactMapPredictor::new(4, 16, 1);
    let emb = vec![vec![0.5_f64; 4]; 4];
    let contacts = cmp.predict_contacts(&emb);
    for row in &contacts {
        for &p in row {
            assert!((0.0..=1.0).contains(&p));
        }
    }
}

#[test]
fn test_contact_predictor_auc() {
    let probs = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
    let contacts = vec![vec![true, false], vec![false, true]];
    let auc = ContactMapPredictor::auc(&probs, &contacts);
    assert!((0.0..=1.0).contains(&auc));
}

// ── ProteinLanguageModelEmbed ─────────────────────────────────────────────

#[test]
fn test_plm_embed_shape() {
    let plm = ProteinLanguageModelEmbed::new(16, 42);
    let seq = ProteinSequence::from_str("ACDE").expect("protein sequence parsing should succeed");
    let emb = plm.embed(&seq);
    assert_eq!(emb.len(), 4);
    assert_eq!(emb[0].len(), 16);
}

#[test]
fn test_plm_embed_finite() {
    let plm = ProteinLanguageModelEmbed::new(16, 7);
    let seq = ProteinSequence::from_str("MKTLLLTLVVV").expect("protein sequence parsing should succeed");
    let emb = plm.embed(&seq);
    for v in &emb {
        for &x in v {
            assert!(x.is_finite());
        }
    }
}

#[test]
fn test_plm_embed_different_seqs() {
    let plm = ProteinLanguageModelEmbed::new(16, 3);
    let seq1 = ProteinSequence::from_str("AAAA").expect("protein sequence parsing should succeed");
    let seq2 = ProteinSequence::from_str("CCCC").expect("protein sequence parsing should succeed");
    let e1 = plm.embed(&seq1);
    let e2 = plm.embed(&seq2);
    // Different sequences should give different embeddings
    let diff: f64 = e1
        .iter()
        .zip(e2.iter())
        .map(|(v1, v2)| {
            v1.iter()
                .zip(v2)
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
        })
        .sum();
    assert!(diff > 0.0);
}

// ── ProteinMetrics ────────────────────────────────────────────────────────

#[test]
fn test_gdt_ts_identical() {
    let s = make_linear_structure(20);
    let gdt = ProteinMetrics::gdt_ts(&s, &s);
    assert!((gdt - 1.0).abs() < 1e-10);
}

#[test]
fn test_gdt_ts_far_apart() {
    let s1 = make_linear_structure(10);
    let s2 = ProteinStructure::new(
        s1.residues
            .iter()
            .map(|r| Residue3D::new(r.n, [r.ca[0] + 100.0, r.ca[1], r.ca[2]], r.c, r.o))
            .collect(),
    );
    let gdt = ProteinMetrics::gdt_ts(&s1, &s2);
    assert_eq!(gdt, 0.0);
}

#[test]
fn test_lddt_identical() {
    let s = make_linear_structure(10);
    let lddt = ProteinMetrics::lddt_score(&s, &s);
    assert_eq!(lddt.len(), 10);
    for &v in &lddt {
        assert!((0.0..=1.0).contains(&v));
    }
}

#[test]
fn test_contact_precision_at_l() {
    let probs = vec![vec![0.0_f64; 10]; 10];
    let contacts = vec![vec![false; 10]; 10];
    // All zeros — precision is 0
    let p = ProteinMetrics::contact_precision_at_l(&probs, &contacts, 5);
    assert_eq!(p, 0.0);
}

#[test]
fn test_secondary_structure_accuracy_perfect() {
    let pred = vec!["H".to_string(), "E".to_string(), "C".to_string()];
    let true_ = vec!["H".to_string(), "E".to_string(), "C".to_string()];
    let acc = ProteinMetrics::secondary_structure_accuracy(&pred, &true_);
    assert!((acc - 1.0).abs() < 1e-10);
}

#[test]
fn test_secondary_structure_accuracy_zero() {
    let pred = vec!["H".to_string(), "H".to_string(), "H".to_string()];
    let true_ = vec!["E".to_string(), "C".to_string(), "E".to_string()];
    let acc = ProteinMetrics::secondary_structure_accuracy(&pred, &true_);
    assert_eq!(acc, 0.0);
}

#[test]
fn test_secondary_structure_accuracy_empty() {
    let acc = ProteinMetrics::secondary_structure_accuracy(&[], &[]);
    assert_eq!(acc, 0.0);
}

// ── Integration tests ─────────────────────────────────────────────────────

#[test]
fn test_full_pipeline_short_seq() {
    let af = AlphaFoldLite::new(8, 4, 100);
    let seqs = vec![
        ProteinSequence::from_str("MKTLL").expect("protein sequence parsing should succeed"),
        ProteinSequence::from_str("MKALL").expect("protein sequence parsing should succeed"),
    ];
    let structure = af.predict(&seqs).expect("alphafold prediction should succeed");
    assert_eq!(structure.len(), 5);
    let distogram = structure.compute_distogram();
    assert_eq!(distogram.len(), 5);
    let ss = structure.compute_secondary_structure();
    assert_eq!(ss.len(), 5);
}

#[test]
fn test_evoformer_then_contact() {
    let evo = ProteinEvoformer::new(4, 4, 50);
    let cmp = ContactMapPredictor::new(4, 8, 50);
    let msa = vec![vec![vec![0.2_f64; 4]; 4]; 2];
    let pair = vec![vec![vec![0.1_f64; 4]; 4]; 4];
    let (msa_out, _) = evo.forward(&msa, &pair);
    // Use first sequence from MSA as embeddings
    let emb = &msa_out[0];
    let contacts = cmp.predict_contacts(emb);
    assert_eq!(contacts.len(), 4);
}

#[test]
fn test_structure_metrics_gdt_range() {
    let pred = make_linear_structure(15);
    // Slightly perturbed reference
    let target = ProteinStructure::new(
        pred.residues
            .iter()
            .map(|r| Residue3D::new(r.n, [r.ca[0] + 0.5, r.ca[1] + 0.3, r.ca[2] + 0.1], r.c, r.o))
            .collect(),
    );
    let gdt = ProteinMetrics::gdt_ts(&pred, &target);
    assert!((0.0..=1.0).contains(&gdt));
    let tm = pred.tm_score(&target).expect("tm_score computation should succeed");
    assert!((0.0..=1.0).contains(&tm));
}

#[test]
fn test_plm_embed_empty_seq() {
    let plm = ProteinLanguageModelEmbed::new(16, 99);
    let seq = ProteinSequence { residues: vec![] };
    let emb = plm.embed(&seq);
    assert!(emb.is_empty());
}

#[test]
fn test_ps_error_display() {
    let e1 = PsError::InvalidSequence("bad char".to_string());
    let e2 = PsError::ComputationFailed("singular".to_string());
    let e3 = PsError::InvalidStructure("empty".to_string());
    assert!(e1.to_string().contains("InvalidSequence"));
    assert!(e2.to_string().contains("ComputationFailed"));
    assert!(e3.to_string().contains("InvalidStructure"));
}

#[test]
fn test_contact_precision_at_l_high() {
    let n = 12;
    let mut probs = vec![vec![0.0_f64; n]; n];
    let mut contacts = vec![vec![false; n]; n];
    // Set some contacts with high probability
    for i in 0..n {
        for j in (i + 6)..n {
            probs[i][j] = 0.9;
            contacts[i][j] = true;
        }
    }
    let p = ProteinMetrics::contact_precision_at_l(&probs, &contacts, 3);
    assert!((p - 1.0).abs() < 1e-10);
}
