//! Tests for bio_ml module (original + new genomics algorithms).

use super::*;
use std::collections::HashSet;

fn tok_seq(s: &str) -> Vec<usize> {
    NucleotideTokenizer::new().tokenize(s)
}

// ─── Original bio_ml tests ────────────────────────────────────────────────────

#[test]
fn test_amino_acid_from_char() {
    assert_eq!(AminoAcid::from_char('A'), Some(AminoAcid::Ala));
    assert_eq!(AminoAcid::from_char('a'), Some(AminoAcid::Ala));
    assert_eq!(AminoAcid::from_char('V'), Some(AminoAcid::Val));
    assert_eq!(AminoAcid::from_char('Z'), None);
    assert_eq!(AminoAcid::from_char('X'), None);
}

#[test]
fn test_amino_acid_to_idx() {
    assert_eq!(AminoAcid::Ala.to_idx(), 0);
    assert_eq!(AminoAcid::Val.to_idx(), 19);
}

#[test]
fn test_protein_tokenizer_length() {
    let tok = ProteinTokenizer::new(128);
    let tokens = tok.tokenize("ACDEFGHIKLMNPQRSTVWY");
    assert_eq!(tokens.len(), 20);
    for &t in &tokens {
        assert!((2..=21).contains(&t));
    }
}

#[test]
fn test_protein_tokenizer_padding() {
    let tok = ProteinTokenizer::new(10);
    let padded = tok.pad(&tok.tokenize("ACDE"));
    assert_eq!(padded.len(), 10);
    assert_eq!(padded[4], 0);
}

#[test]
fn test_protein_tokenizer_truncation() {
    assert_eq!(
        ProteinTokenizer::new(5)
            .tokenize("ACDEFGHIKLMNPQRSTVWY")
            .len(),
        5
    );
}

#[test]
fn test_protein_tokenizer_unknown_residues() {
    assert_eq!(ProteinTokenizer::new(100).tokenize("AXBZC").len(), 2);
}

#[test]
fn test_rotary_embedding_shape() {
    let rope = RotaryEmbedding::new(16, 64).expect("test value");
    assert_eq!(rope.apply(&[0.1_f64; 16], 5).expect("test value").len(), 16);
}

#[test]
fn test_esm_attention_block_shape() {
    let block = EsmAttentionBlock::new(16, 2, 32, 42).expect("test value");
    let seq: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1_f64; 16]).collect();
    let out = block.forward(&seq).expect("test value");
    assert_eq!(out.len(), 5);
    assert_eq!(out[0].len(), 16);
}

#[test]
fn test_contact_prediction_matrix_shape() {
    let head = ContactPredictionHead::new(16, 32, 99);
    let embeddings: Vec<Vec<f64>> = (0..8).map(|_| vec![0.1_f64; 16]).collect();
    let contact_map = head.forward(&embeddings);
    assert_eq!(contact_map.len(), 8);
    assert_eq!(contact_map[0].len(), 8);
    for row in &contact_map {
        for &v in row {
            assert!((0.0..=1.0).contains(&v));
        }
    }
}

#[test]
fn test_contact_prediction_symmetric_tendency() {
    let head = ContactPredictionHead::new(8, 16, 77);
    let embeddings: Vec<Vec<f64>> = (0..4).map(|i| vec![(i as f64) * 0.1; 8]).collect();
    let map = head.forward(&embeddings);
    for i in 0..4 {
        assert!(map[i][i] >= 0.0);
    }
}

#[test]
fn test_esm_construction() {
    let esm = EvolutionaryScaleModeling::new(2, 16, 2, 32, 42).expect("test value");
    let logits = esm.forward(&[2usize, 3, 4, 5]).expect("test value");
    assert_eq!(logits.len(), 4);
    assert_eq!(logits[0].len(), 22);
}

#[test]
fn test_distance_matrix_symmetric() {
    let coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mat = DistanceMatrix::compute(&coords);
    assert_eq!(mat.len(), 3);
    for i in 0..3 {
        for j in 0..3 {
            assert!((mat[i][j] - mat[j][i]).abs() < 1e-10);
        }
        assert!(mat[i][i].abs() < 1e-10);
    }
}

#[test]
fn test_distance_matrix_values() {
    let mat = DistanceMatrix::compute(&[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]);
    assert!((mat[0][1] - 5.0).abs() < 1e-10);
}

#[test]
fn test_secondary_structure_3class() {
    let pred = SecondaryStructurePredictor::new(16, 3, 32, 42);
    let embeddings: Vec<Vec<f64>> = (0..10).map(|_| vec![0.1_f64; 16]).collect();
    let classes = pred.predict(&embeddings);
    assert_eq!(classes.len(), 10);
    for &c in &classes {
        assert!(c <= 2);
    }
}

#[test]
fn test_torsion_angle_shape() {
    let predictor = TorsionAnglePredictor::new(16, 32, 42);
    let embeddings: Vec<Vec<f64>> = (0..6).map(|_| vec![0.2_f64; 16]).collect();
    let angles = predictor.predict(&embeddings);
    assert_eq!(angles.len(), 6);
    for ap in &angles {
        assert!(ap[0].abs() <= std::f64::consts::PI + 1e-9);
        assert!(ap[1].abs() <= std::f64::consts::PI + 1e-9);
    }
}

#[test]
fn test_structure_encoder_features() {
    let residues = vec![
        BackboneAtoms {
            n: [0.0, 0.0, 0.0],
            ca: [1.5, 0.0, 0.0],
            c: [2.0, 1.2, 0.0],
            o: [3.0, 1.5, 0.5],
        },
        BackboneAtoms {
            n: [3.8, 0.0, 0.0],
            ca: [5.3, 0.0, 0.0],
            c: [5.8, 1.2, 0.0],
            o: [6.8, 1.5, 0.5],
        },
    ];
    let feats = StructureEncoder::encode(&residues);
    assert_eq!(feats.len(), 2);
    assert_eq!(feats[0].len(), 6);
    for feat in &feats {
        for &d in feat {
            assert!(d >= 0.0);
        }
    }
}

#[test]
fn test_fape_loss_same_zero() {
    let bb = BackboneAtoms {
        n: [0.0, 0.0, 0.0],
        ca: [1.5, 0.0, 0.0],
        c: [2.0, 1.2, 0.0],
        o: [2.5, 1.5, 0.5],
    };
    let coords = vec![bb.clone(), bb.clone(), bb.clone()];
    let loss = AlphaFoldLoss::new(10.0).compute(&coords, &coords).expect("test value");
    assert!(loss < 1e-6, "FAPE for identical structures: {loss}");
}

#[test]
fn test_fape_loss_different() {
    let pred = vec![
        BackboneAtoms {
            n: [0.0, 0.0, 0.0],
            ca: [1.5, 0.0, 0.0],
            c: [2.0, 1.2, 0.0],
            o: [2.5, 1.5, 0.5],
        },
        BackboneAtoms {
            n: [3.8, 0.0, 0.0],
            ca: [5.3, 0.0, 0.0],
            c: [5.8, 1.2, 0.0],
            o: [6.8, 1.5, 0.5],
        },
    ];
    let tru = vec![
        BackboneAtoms {
            n: [0.0, 0.0, 0.0],
            ca: [1.5, 0.0, 0.0],
            c: [2.0, 1.2, 0.0],
            o: [2.5, 1.5, 0.5],
        },
        BackboneAtoms {
            n: [5.0, 2.0, 1.0],
            ca: [6.5, 2.0, 1.0],
            c: [7.0, 3.2, 1.0],
            o: [7.5, 3.5, 1.5],
        },
    ];
    assert!(AlphaFoldLoss::new(10.0).compute(&pred, &tru).expect("test value") > 0.0);
}

#[test]
fn test_nucleotide_tokenizer_atcg() {
    assert_eq!(
        NucleotideTokenizer::new().tokenize("ACGT"),
        vec![0, 1, 2, 3]
    );
}

#[test]
fn test_nucleotide_tokenizer_rna() {
    assert_eq!(
        NucleotideTokenizer::new().tokenize("ACGU"),
        vec![0, 1, 2, 3]
    );
}

#[test]
fn test_nucleotide_tokenizer_unknown() {
    assert_eq!(NucleotideTokenizer::new().tokenize("ACNGT")[2], 4);
}

#[test]
fn test_dna_embedding_one_hot() {
    let emb = DnaEmbedding::new_one_hot(5);
    let embedded = emb.embed(&[0usize, 1, 2, 3]).expect("test value");
    assert_eq!(embedded.len(), 4);
    assert_eq!(embedded[0].len(), 5);
    assert!((embedded[0][0] - 1.0).abs() < 1e-10);
    assert!(embedded[0][1].abs() < 1e-10);
}

#[test]
fn test_conv_motif_scanner_shape() {
    let scanner = ConvolutionalMotifScanner::new(3, 5, 8, 42);
    let emb = DnaEmbedding::new_one_hot(5);
    let embeddings = emb.embed(&tok_seq("ACGTACGT")).expect("test value");
    let out = scanner.scan(&embeddings).expect("test value");
    assert_eq!(out.len(), 6);
    assert_eq!(out[0].len(), 8);
}

#[test]
fn test_conv_motif_scanner_short_error() {
    let scanner = ConvolutionalMotifScanner::new(5, 5, 4, 1);
    let emb = DnaEmbedding::new_one_hot(5);
    assert!(scanner.scan(&emb.embed(&tok_seq("ACG")).expect("test value")).is_err());
}

#[test]
fn test_rna_fold_gc_pair() {
    let mfe_gc = RnaFoldingScore::mfe_approx("GCGCGCGCGCGCGCGC");
    let mfe_au = RnaFoldingScore::mfe_approx("AUAUAUAUAUAUAUAU");
    assert!(mfe_gc <= mfe_au, "gc={mfe_gc}, au={mfe_au}");
}

#[test]
fn test_rna_fold_empty() {
    assert_eq!(RnaFoldingScore::mfe_approx(""), 0.0);
}

#[test]
fn test_rna_fold_short() {
    assert!(RnaFoldingScore::mfe_approx("AU") <= 0.0);
}

#[test]
fn test_splice_site_predictor() {
    let predictor = SpliceSitePredictor::new(42);
    let tokens = NucleotideTokenizer::new().tokenize("ACGTACGTACGTACGT");
    let (donor, acceptor) = predictor.predict(&tokens).expect("test value");
    assert_eq!(donor.len(), tokens.len() - 4);
    assert_eq!(acceptor.len(), tokens.len() - 4);
    for &s in donor.iter().chain(acceptor.iter()) {
        assert!((0.0..=1.0).contains(&s));
    }
}

#[test]
fn test_splice_site_predictor_short_error() {
    assert!(SpliceSitePredictor::new(1).predict(&[0, 1, 2]).is_err());
}

#[test]
fn test_normalize_log1p() {
    let normalizer = ScRnaSeqNormalizer::new(10_000.0);
    let norm = normalizer.normalize(&[vec![100.0, 200.0, 300.0], vec![50.0, 50.0, 50.0]]);
    assert_eq!(norm.len(), 2);
    assert_eq!(norm[0].len(), 3);
    for row in &norm {
        for &v in row {
            assert!(v >= 0.0 && v.is_finite());
        }
    }
}

#[test]
fn test_normalize_zero_library() {
    let norm = ScRnaSeqNormalizer::new(10_000.0).normalize(&[vec![0.0, 0.0, 0.0]]);
    for &v in &norm[0] {
        assert!((v - 0.0_f64.ln_1p()).abs() < 1e-10);
    }
}

#[test]
fn test_pca_reducer_dims() {
    let mut pca = PcaReducer::new(3);
    let data: Vec<Vec<f64>> = (0..20)
        .map(|i| vec![(i as f64) * 0.1, (i as f64) * 0.2, (i as f64) * 0.3])
        .collect();
    pca.fit(&data, 42).expect("test value");
    let t = pca.transform(&data).expect("test value");
    assert_eq!(t.len(), 20);
    assert_eq!(t[0].len(), 3);
}

#[test]
fn test_pca_reducer_fewer_components() {
    let mut pca = PcaReducer::new(2);
    let data: Vec<Vec<f64>> = (0..15)
        .map(|i| vec![(i as f64) * 0.1, (i as f64) * 0.2, -(i as f64) * 0.1])
        .collect();
    pca.fit(&data, 7).expect("test value");
    assert_eq!(pca.transform(&data).expect("test value")[0].len(), 2);
}

#[test]
fn test_vae_encode_decode_shape() {
    let vae = VariationalAutoencoder::new(50, 10, 32, 42);
    let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.01).collect();
    let (mu, lv) = vae.encode(&x);
    assert_eq!(mu.len(), 10);
    let recon = vae.decode(&vae.reparameterize(&mu, &lv, 77));
    assert_eq!(recon.len(), 50);
    for &v in &recon {
        assert!(v >= 0.0);
    }
}

#[test]
fn test_cell_type_classifier_shape() {
    let clf = CellTypeClassifier::new(30, 64, 5, 42);
    let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.01).collect();
    let probs = clf.predict_proba(&x);
    assert_eq!(probs.len(), 5);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "probs sum: {sum}");
    assert!(clf.predict(&x) < 5);
}

#[test]
fn test_trajectory_bfs_pseudotime() {
    let cells = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![2.0, 0.0],
        vec![3.0, 0.0],
    ];
    let pt = TrajectoryInference::new(2)
        .compute_pseudotime(&cells, 0)
        .expect("test value");
    assert_eq!(pt.len(), 4);
    assert_eq!(pt[0], 0.0);
    for &v in &pt {
        assert!(v >= 0.0);
    }
}

#[test]
fn test_trajectory_root_oob_error() {
    assert!(TrajectoryInference::new(1)
        .compute_pseudotime(&[vec![0.0, 0.0]], 5)
        .is_err());
}

#[test]
fn test_tanimoto_identical() {
    let fp = vec![1.0, 0.0, 1.0, 1.0, 0.0];
    assert!((FingerprintSimilarity::tanimoto(&fp, &fp) - 1.0).abs() < 1e-10);
}

#[test]
fn test_tanimoto_disjoint_zero() {
    let t = FingerprintSimilarity::tanimoto(&[1.0, 1.0, 0.0, 0.0], &[0.0, 0.0, 1.0, 1.0]);
    assert!(t.abs() < 1e-10);
}

#[test]
fn test_tanimoto_partial() {
    let t = FingerprintSimilarity::tanimoto(&[1.0, 1.0, 0.0, 0.0], &[1.0, 0.0, 1.0, 0.0]);
    assert!((t - 1.0 / 3.0).abs() < 1e-9);
}

#[test]
fn test_dti_score_range() {
    let score =
        DrugTargetInteraction::new(8, 16, 32, 42).score(&[0.1_f64; 8], &[0.2_f64; 16]);
    assert!((0.0..=1.0).contains(&score));
}

#[test]
fn test_virtual_screening_sorted() {
    let dti = DrugTargetInteraction::new(4, 8, 16, 42);
    let compounds: Vec<Vec<f64>> = (0..5).map(|i| vec![(i as f64) * 0.1; 4]).collect();
    let target = vec![0.5_f64; 8];
    let ranked = VirtualScreening::new(dti.clone()).screen(&compounds, &target);
    assert_eq!(ranked.len(), 5);
    let scores: Vec<f64> = ranked
        .iter()
        .map(|&i| dti.score(&compounds[i], &target))
        .collect();
    for w in scores.windows(2) {
        assert!(w[0] >= w[1]);
    }
}

#[test]
fn test_admet_5_outputs() {
    let admet = AdmetPredictor::new(12, 32, 42).predict(&[0.1_f64; 12]);
    assert_eq!(admet.len(), 5);
    for (i, &v) in admet.iter().enumerate() {
        assert!((0.0..=1.0).contains(&v), "ADMET[{i}]={v}");
    }
}

#[test]
fn test_molecular_docking_scoring() {
    let docker = MolecularDocking::new();
    let score = docker.score(&[([0.0, 0.0, 0.0], 1.0)], &[([4.0, 0.0, 0.0], -1.0)]);
    assert!(score.is_finite());
}

#[test]
fn test_molecular_docking_attractive() {
    let docker = MolecularDocking::new();
    let score = docker.score(&[([0.0, 0.0, 0.0], 0.0)], &[([3.5, 0.0, 0.0], 0.0)]);
    assert!(score < 0.0, "LJ at r_min should be negative: {score}");
}

// ─── New genomics tests ───────────────────────────────────────────────────────

#[test]
fn test_scrna_matrix_new_ok() {
    let counts = vec![vec![10.0, 5.0, 0.0], vec![0.0, 2.0, 8.0]];
    let mat = ScRnaMatrix::new(counts).expect("test value");
    assert_eq!(mat.n_cells, 2);
    assert_eq!(mat.n_genes, 3);
}

#[test]
fn test_scrna_matrix_log1p_cpm_positive() {
    let counts = vec![vec![100.0, 200.0, 300.0]];
    let mat = ScRnaMatrix::new(counts).expect("test value");
    let norm = mat.log1p_cpm();
    for &v in &norm[0] {
        assert!(v > 0.0 && v.is_finite());
    }
}

#[test]
fn test_scrna_matrix_gene_stats() {
    let counts = vec![
        vec![10.0, 20.0],
        vec![30.0, 40.0],
        vec![50.0, 60.0],
    ];
    let mat = ScRnaMatrix::new(counts).expect("test value");
    let stats = mat.gene_stats();
    assert_eq!(stats.len(), 2);
    for (mean, var) in &stats {
        assert!(mean.is_finite() && var.is_finite() && *var >= 0.0);
    }
}

#[test]
fn test_scrna_matrix_ragged_error() {
    let counts = vec![vec![1.0, 2.0], vec![3.0]];
    assert!(ScRnaMatrix::new(counts).is_err());
}

#[test]
fn test_scvae_encoder_shape() {
    let enc = ScvaeEncoder::new(20, 64, 10, 42);
    let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
    let (mu, lv) = enc.encode(&x).expect("test value");
    assert_eq!(mu.len(), 10);
    assert_eq!(lv.len(), 10);
    // log_var clamped to [-10, 10]
    for &v in &lv {
        assert!((-10.0..=10.0).contains(&v));
    }
}

#[test]
fn test_scvae_encoder_reparameterize() {
    let enc = ScvaeEncoder::new(20, 32, 8, 7);
    let x = vec![0.1_f64; 20];
    let (mu, lv) = enc.encode(&x).expect("test value");
    let z = enc.reparameterize(&mu, &lv, 99);
    assert_eq!(z.len(), 8);
    assert!(z.iter().all(|v| v.is_finite()));
}

#[test]
fn test_scvae_decoder_output_shapes() {
    let dec = ScvaeDecoder::new(8, 32, 20, 42);
    let z = vec![0.1_f64; 8];
    let (mu, theta, pi) = dec.decode(&z).expect("test value");
    assert_eq!(mu.len(), 20);
    assert_eq!(theta.len(), 20);
    assert_eq!(pi.len(), 20);
    for &v in &mu { assert!(v > 0.0); }
    for &v in &theta { assert!(v > 0.0); }
    for &v in &pi { assert!((0.0..=1.0).contains(&v)); }
}

#[test]
fn test_scvae_decoder_zinb_loss_finite() {
    let dec = ScvaeDecoder::new(8, 32, 10, 42);
    let z = vec![0.1_f64; 8];
    let (mu, theta, pi) = dec.decode(&z).expect("test value");
    let x = vec![1.0_f64; 10];
    let loss = dec.zinb_loss(&x, &mu, &theta, &pi);
    assert!(loss.is_finite() && loss >= 0.0);
}

#[test]
fn test_scvae_model_forward() {
    let model = ScvaeModel::new(20, 32, 8, 42);
    let x: Vec<f64> = (0..20).map(|i| (i as f64) * 0.05).collect();
    let result = model.forward(&x, 123);
    assert!(result.is_ok());
    let (z, mu, lv, dec_mu, dec_theta, dec_pi) = result.expect("test value");
    assert_eq!(z.len(), 8);
    assert_eq!(mu.len(), 8);
    assert_eq!(lv.len(), 8);
    assert_eq!(dec_mu.len(), 20);
    assert_eq!(dec_theta.len(), 20);
    assert_eq!(dec_pi.len(), 20);
}

#[test]
fn test_scvae_elbo_finite() {
    let model = ScvaeModel::new(15, 32, 6, 77);
    let x = vec![2.0_f64; 15];
    let (_, mu, lv, dec_mu, dec_theta, dec_pi) = model.forward(&x, 42).expect("test value");
    let elbo = model.elbo(&x, &mu, &lv, &dec_mu, &dec_theta, &dec_pi);
    assert!(elbo.is_finite());
}

#[test]
fn test_leiden_clustering_labels() {
    let embeddings: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, 0.0]).collect();
    let labels = LeidenClustering::new(3, 5).fit(&embeddings).expect("test value");
    assert_eq!(labels.len(), 10);
    // All labels should be valid indices
    for &l in &labels {
        assert!(l < 10);
    }
}

#[test]
fn test_leiden_clustering_empty_error() {
    assert!(LeidenClustering::new(2, 5).fit(&[]).is_err());
}

#[test]
fn test_leiden_two_clusters() {
    // Two well-separated groups: should produce at least 1 cluster
    let mut embeddings: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.01]).collect();
    let far: Vec<Vec<f64>> = (0..5).map(|i| vec![100.0 + i as f64 * 0.01]).collect();
    embeddings.extend(far);
    let labels = LeidenClustering::new(2, 10).fit(&embeddings).expect("test value");
    assert_eq!(labels.len(), 10);
}

#[test]
fn test_dna_tokenizer_k3() {
    let tok = DnaTokenizer::new(3).expect("test value");
    assert_eq!(tok.vocab_size, 64);
    let tokens = tok.tokenize("ACGT");
    assert_eq!(tokens.len(), 2); // "ACG", "CGT"
}

#[test]
fn test_dna_tokenizer_k1() {
    let tok = DnaTokenizer::new(1).expect("test value");
    let tokens = tok.tokenize("ACGT");
    assert_eq!(tokens, vec![0, 1, 2, 3]);
}

#[test]
fn test_dna_tokenizer_k0_error() {
    assert!(DnaTokenizer::new(0).is_err());
}

#[test]
fn test_dna_tokenizer_one_hot() {
    let tok = DnaTokenizer::new(2).expect("test value");
    let oh = tok.one_hot(0);
    assert_eq!(oh.len(), 16);
    assert!((oh[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_dna_conv_net_forward_shape() {
    let net = DnaConvNet::new(64, 8, 16, 3, 2, 3, 42);
    let tok = DnaTokenizer::new(3).expect("test value");
    let tokens = tok.tokenize("ACGTACGTACGTACGT");
    let logits = net.forward(&tokens).expect("test value");
    assert_eq!(logits.len(), 3);
}

#[test]
fn test_dna_conv_net_empty_error() {
    let net = DnaConvNet::new(64, 8, 16, 3, 2, 2, 42);
    assert!(net.forward(&[]).is_err());
}

#[test]
fn test_chromatin_accessibility_range() {
    let model = ChromatinAccessibility::new(64, 8, 16, 3, 2, 42);
    let tok = DnaTokenizer::new(3).expect("test value");
    let tokens = tok.tokenize("ACGTACGTACGTACGT");
    let prob = model.predict(&tokens).expect("test value");
    assert!((0.0..=1.0).contains(&prob));
}

#[test]
fn test_variant_effect_predictor() {
    let vep = VariantEffectPredictor::new(3, 8, 16, 3, 2, 2, 42).expect("test value");
    let results = vep.mutagenesis("ACGTACGT").expect("test value");
    // 8 positions × 3 alts = up to 24 variants (some may be filtered)
    assert!(!results.is_empty());
    for (pos, alt, lfc) in &results {
        assert!(*pos < 8);
        assert!(['A', 'C', 'G', 'T'].contains(alt));
        assert_eq!(lfc.len(), 2);
    }
}

#[test]
fn test_kaplan_meier_fit() {
    let times = [1.0, 2.0, 3.0, 4.0, 5.0];
    let events = [1u8, 0, 1, 1, 0];
    let km = KaplanMeier::fit(&times, &events).expect("test value");
    assert!(!km.timeline.is_empty());
    // Survival should be non-increasing
    for w in km.survival.windows(2) {
        assert!(w[0] >= w[1]);
    }
}

#[test]
fn test_kaplan_meier_predict_at_zero() {
    let times = [1.0, 2.0, 3.0];
    let events = [1u8, 1, 1];
    let km = KaplanMeier::fit(&times, &events).expect("test value");
    assert!((km.predict(0.0) - 1.0).abs() < 1e-10);
}

#[test]
fn test_kaplan_meier_predict_after_all() {
    let times = [1.0, 2.0];
    let events = [1u8, 1];
    let km = KaplanMeier::fit(&times, &events).expect("test value");
    assert!(km.predict(10.0) < 1.0);
}

#[test]
fn test_cox_ph_likelihood_finite() {
    let mut cox = CoxPh::new(3);
    let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1, 0.5, 1.0]).collect();
    let times: Vec<f64> = (0..10).map(|i| (i + 1) as f64).collect();
    let events: Vec<u8> = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
    let ll = cox.neg_partial_likelihood(&x, &times, &events).expect("test value");
    assert!(ll.is_finite());
    // Fit for a few iterations
    cox.fit(&x, &times, &events, 0.01, 5).expect("test value");
    let log_hazards = cox.predict_log_hazard(&x);
    assert_eq!(log_hazards.len(), 10);
}

#[test]
fn test_cox_ph_c_index_range() {
    let cox = CoxPh::new(2);
    let log_risks = vec![0.5, 1.0, 0.2, 0.8];
    let times = vec![3.0, 1.0, 4.0, 2.0];
    let events = vec![1u8, 1, 0, 1];
    let ci = SurvivalMetrics::c_index(&log_risks, &times, &events).expect("test value");
    assert!((0.0..=1.0).contains(&ci));
}

#[test]
fn test_deepsurv_log_risk_shape() {
    let ds = DeepSurv::new(5, 16, 42);
    let x: Vec<Vec<f64>> = (0..4).map(|_| vec![0.1_f64; 5]).collect();
    let risks = ds.predict_log_risk(&x);
    assert_eq!(risks.len(), 4);
    for &r in &risks { assert!(r.is_finite()); }
}

#[test]
fn test_survival_brier_score() {
    let surv_probs = vec![0.8, 0.6, 0.3, 0.9];
    let times = vec![5.0, 2.0, 3.0, 8.0];
    let events = vec![0u8, 1, 1, 0];
    let bs = SurvivalMetrics::brier_score(&surv_probs, &times, &events, 4.0).expect("test value");
    assert!((0.0..=1.0).contains(&bs));
}

#[test]
fn test_integrated_brier_score() {
    let eval_times = vec![1.0, 3.0, 5.0];
    let surv_at_times = vec![
        vec![0.9, 0.8, 0.7, 0.95],
        vec![0.7, 0.5, 0.4, 0.8],
        vec![0.5, 0.3, 0.2, 0.6],
    ];
    let times = vec![2.0, 4.0, 6.0, 8.0];
    let events = vec![1u8, 1, 0, 0];
    let ibs = SurvivalMetrics::integrated_brier_score(
        &surv_at_times, &times, &events, &eval_times,
    ).expect("test value");
    assert!(ibs.is_finite() && ibs >= 0.0);
}

#[test]
fn test_omics_dataset_new_ok() {
    let g = vec![vec![1.0, 2.0]; 3];
    let t = vec![vec![0.5, 0.6, 0.7]; 3];
    let p = vec![vec![0.1]; 3];
    let ds = OmicsDataset::new(g, t, p).expect("test value");
    assert_eq!(ds.n_samples, 3);
}

#[test]
fn test_omics_dataset_mismatch_error() {
    let g = vec![vec![1.0]; 3];
    let t = vec![vec![1.0]; 2]; // wrong
    let p = vec![vec![1.0]; 3];
    assert!(OmicsDataset::new(g, t, p).is_err());
}

#[test]
fn test_omics_dataset_concat() {
    let g = vec![vec![1.0, 2.0]];
    let t = vec![vec![3.0]];
    let p = vec![vec![4.0, 5.0]];
    let ds = OmicsDataset::new(g, t, p).expect("test value");
    let cat = ds.concat_sample(0);
    assert_eq!(cat, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_mofa_fit_and_transform() {
    let dims = vec![5, 4, 3];
    let mut mofa = MoFa::new(dims.clone(), 2, 42);
    // 6 samples, 3 modalities
    let data: Vec<Vec<Vec<f64>>> = dims.iter().map(|&d| {
        (0..6).map(|s| (0..d).map(|g| (s * d + g) as f64 * 0.01).collect()).collect()
    }).collect();
    mofa.fit(&data, 5, 42).expect("test value");
    assert_eq!(mofa.factors.len(), 6);
    assert_eq!(mofa.factors[0].len(), 2);
    // Transform a new sample
    let new_sample: Vec<Vec<f64>> = dims.iter().map(|&d| vec![0.1_f64; d]).collect();
    let z = mofa.transform_sample(&new_sample).expect("test value");
    assert_eq!(z.len(), 2);
}

#[test]
fn test_omics_attention_fusion_shape() {
    let modality_dims = vec![5, 4, 3];
    let fusion = OmicsAttentionFusion::new(&modality_dims, 8, 2, 42).expect("test value");
    let features: Vec<Vec<f64>> = modality_dims.iter().map(|&d| vec![0.1_f64; d]).collect();
    let out = fusion.fuse(&features).expect("test value");
    assert_eq!(out.len(), 8);
}

#[test]
fn test_omics_attention_fusion_wrong_modalities_error() {
    let fusion = OmicsAttentionFusion::new(&[5, 4], 8, 2, 42).expect("test value");
    let features = vec![vec![0.1_f64; 5]]; // only 1 modality, expect 2
    assert!(fusion.fuse(&features).is_err());
}

#[test]
fn test_pathway_enrichment_score() {
    let pe = PathwayEnrichment::new(20);
    let ranked: Vec<usize> = (0..20).collect();
    let gene_set: HashSet<usize> = [0, 1, 2, 3, 4].iter().cloned().collect();
    let (es, leading_edge) = pe.enrichment_score(&ranked, &gene_set, 1.0).expect("test value");
    assert!(es.is_finite());
    assert!(leading_edge <= gene_set.len());
}

#[test]
fn test_pathway_enrichment_positive_es_for_top_genes() {
    let pe = PathwayEnrichment::new(10);
    // Top genes (0..3) all in gene set → should give positive ES
    let ranked: Vec<usize> = (0..10).collect();
    let gene_set: HashSet<usize> = [0, 1, 2].iter().cloned().collect();
    let (es, _) = pe.enrichment_score(&ranked, &gene_set, 1.0).expect("test value");
    assert!(es > 0.0);
}

#[test]
fn test_pathway_enrichment_all_pathways() {
    let pe = PathwayEnrichment::new(12);
    let ranked: Vec<usize> = (0..12).collect();
    let pathways: Vec<HashSet<usize>> = vec![
        [0, 2, 4].iter().cloned().collect(),
        [1, 3, 5].iter().cloned().collect(),
    ];
    let results = pe.compute_all(&ranked, &pathways, 1.0).expect("test value");
    assert_eq!(results.len(), 2);
}
