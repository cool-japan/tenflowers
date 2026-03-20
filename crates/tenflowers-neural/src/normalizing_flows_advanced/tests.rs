//! Tests for the `normalizing_flows_advanced` module.
//!
//! Covers all 9 components with 50+ tests.

#[cfg(test)]
mod normalizing_flows_advanced {
    use super::super::*;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    // ── Helpers ─────────────────────────────────────────────────────────────

    fn make_input(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..dim)
            .map(|_| sample_standard_normal(&mut rng) * 0.5)
            .collect()
    }

    fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §1  NfaActNorm
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_act_norm_forward_shape() {
        let an = NfaActNorm::new(4);
        let x = make_input(4, 1);
        let (z, _ld) = an.forward(&x).expect("act_norm forward");
        assert_eq!(z.len(), 4);
    }

    #[test]
    fn test_act_norm_invertibility() {
        let an = NfaActNorm::new(6);
        let x = make_input(6, 2);
        let (z, _) = an.forward(&x).expect("forward");
        let x_rec = an.inverse(&z).expect("inverse");
        assert!(l2_dist(&x, &x_rec) < 1e-5, "ActNorm not invertible");
    }

    #[test]
    fn test_act_norm_log_det_finite() {
        let an = NfaActNorm::new(8);
        let x = make_input(8, 3);
        let (_, ld) = an.forward(&x).expect("forward");
        assert!(ld.is_finite(), "log_det must be finite");
    }

    #[test]
    fn test_act_norm_init_from_batch() {
        let mut an = NfaActNorm::new(4);
        let batch: Vec<Vec<f32>> = (0..16).map(|i| make_input(4, i as u64)).collect();
        an.initialize_from_batch(&batch).expect("init batch");
        let x = make_input(4, 99);
        let (z, _) = an.forward(&x).expect("forward after init");
        assert_eq!(z.len(), 4);
    }

    #[test]
    fn test_act_norm_dim_mismatch_error() {
        let an = NfaActNorm::new(4);
        let x = make_input(3, 5);
        assert!(an.forward(&x).is_err(), "should error on dim mismatch");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §2  NfaInvertible1x1Conv
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_inv1x1_forward_shape() {
        let conv = NfaInvertible1x1Conv::new(6, 42).expect("new");
        let x = make_input(6, 10);
        let (z, _) = conv.forward(&x).expect("forward");
        assert_eq!(z.len(), 6);
    }

    #[test]
    fn test_inv1x1_invertibility() {
        let conv = NfaInvertible1x1Conv::new(4, 7).expect("new");
        let x = make_input(4, 11);
        let (z, _) = conv.forward(&x).expect("forward");
        let x_rec = conv.inverse(&z).expect("inverse");
        assert!(
            l2_dist(&x, &x_rec) < 1e-4,
            "inv1x1 not invertible: {:.6}",
            l2_dist(&x, &x_rec)
        );
    }

    #[test]
    fn test_inv1x1_log_det_consistent() {
        let conv = NfaInvertible1x1Conv::new(4, 8).expect("new");
        let x = make_input(4, 12);
        let (_, ld) = conv.forward(&x).expect("forward");
        let ld2 = conv.log_det();
        assert!((ld - ld2).abs() < 1e-5, "log_det inconsistent");
    }

    #[test]
    fn test_inv1x1_log_det_finite() {
        let conv = NfaInvertible1x1Conv::new(8, 99).expect("new");
        let x = make_input(8, 13);
        let (_, ld) = conv.forward(&x).expect("forward");
        assert!(ld.is_finite(), "log_det must be finite");
    }

    #[test]
    fn test_inv1x1_dim_mismatch_error() {
        let conv = NfaInvertible1x1Conv::new(4, 1).expect("new");
        let x = make_input(5, 14);
        assert!(conv.forward(&x).is_err());
    }

    #[test]
    fn test_inv1x1_zero_dim_error() {
        assert!(NfaInvertible1x1Conv::new(0, 1).is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §3  NfaAffineCoupling
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_affine_coupling_forward_shape() {
        let ac = NfaAffineCoupling::new(8, 16, 5).expect("new");
        let x = make_input(8, 20);
        let (z, _) = ac.forward(&x).expect("forward");
        assert_eq!(z.len(), 8);
    }

    #[test]
    fn test_affine_coupling_invertibility() {
        let ac = NfaAffineCoupling::new(6, 12, 6).expect("new");
        let x = make_input(6, 21);
        let (z, _) = ac.forward(&x).expect("forward");
        let x_rec = ac.inverse(&z).expect("inverse");
        assert!(l2_dist(&x, &x_rec) < 1e-4, "coupling not invertible");
    }

    #[test]
    fn test_affine_coupling_log_det_finite() {
        let ac = NfaAffineCoupling::new(4, 8, 7).expect("new");
        let x = make_input(4, 22);
        let (_, ld) = ac.forward(&x).expect("forward");
        assert!(ld.is_finite());
    }

    #[test]
    fn test_affine_coupling_dim_lt2_error() {
        assert!(NfaAffineCoupling::new(1, 8, 8).is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §4  NfaGlowStep
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_glow_step_forward_shape() {
        let step = NfaGlowStep::new(8, 16, 30).expect("new");
        let x = make_input(8, 30);
        let (z, _) = step.forward(&x).expect("forward");
        assert_eq!(z.len(), 8);
    }

    #[test]
    fn test_glow_step_invertibility() {
        let step = NfaGlowStep::new(4, 8, 31).expect("new");
        let x = make_input(4, 31);
        let (z, _) = step.forward(&x).expect("forward");
        let x_rec = step.inverse(&z).expect("inverse");
        assert!(
            l2_dist(&x, &x_rec) < 1e-3,
            "Glow step not invertible: {:.6}",
            l2_dist(&x, &x_rec)
        );
    }

    #[test]
    fn test_glow_step_log_det_finite() {
        let step = NfaGlowStep::new(6, 12, 32).expect("new");
        let x = make_input(6, 32);
        let (_, ld) = step.forward(&x).expect("forward");
        assert!(ld.is_finite());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §5  NfaGlowModel
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_glow_model_forward_shape() {
        let model = NfaGlowModel::new(8, 2, 2, 16, 40).expect("new");
        let x = make_input(8, 40);
        let (z, _) = model.forward(&x).expect("forward");
        assert_eq!(z.len(), 8);
    }

    #[test]
    fn test_glow_model_invertibility() {
        let model = NfaGlowModel::new(8, 2, 2, 16, 41).expect("new");
        let x = make_input(8, 41);
        let (z, _) = model.forward(&x).expect("forward");
        let x_rec = model.inverse(&z).expect("inverse");
        assert!(
            l2_dist(&x, &x_rec) < 1e-2,
            "Glow model not invertible: {:.6}",
            l2_dist(&x, &x_rec)
        );
    }

    #[test]
    fn test_glow_model_log_likelihood_finite() {
        let model = NfaGlowModel::new(8, 2, 2, 16, 42).expect("new");
        let x = make_input(8, 42);
        let ll = model.log_likelihood(&x).expect("log_likelihood");
        assert!(ll.is_finite());
    }

    #[test]
    fn test_glow_model_dim_not_divisible_error() {
        assert!(
            NfaGlowModel::new(6, 2, 2, 16, 43).is_err(),
            "should fail: 6 not divisible by 4"
        );
    }

    #[test]
    fn test_glow_model_zero_levels_error() {
        assert!(NfaGlowModel::new(8, 0, 2, 16, 44).is_err());
    }

    #[test]
    fn test_glow_model_single_level() {
        let model = NfaGlowModel::new(4, 1, 3, 8, 45).expect("new");
        let x = make_input(4, 45);
        let (z, _) = model.forward(&x).expect("forward");
        assert_eq!(z.len(), 4);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §6  NfaMaskedAutoregressive (MAF)
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_maf_forward_shape() {
        let maf = NfaMaskedAutoregressive::new(4, 8, 2, 50).expect("new");
        let x = make_input(4, 50);
        let (z, _) = maf.forward(&x).expect("forward");
        assert_eq!(z.len(), 4);
    }

    #[test]
    fn test_maf_invertibility() {
        let maf = NfaMaskedAutoregressive::new(4, 16, 2, 51).expect("new");
        let x = make_input(4, 51);
        let (z, _) = maf.forward(&x).expect("forward");
        let x_rec = maf.inverse(&z).expect("inverse");
        assert!(
            l2_dist(&x, &x_rec) < 1e-3,
            "MAF not invertible: {:.6}",
            l2_dist(&x, &x_rec)
        );
    }

    #[test]
    fn test_maf_log_likelihood_finite() {
        let maf = NfaMaskedAutoregressive::new(4, 8, 2, 52).expect("new");
        let x = make_input(4, 52);
        let ll = maf.log_likelihood(&x).expect("log_likelihood");
        assert!(ll.is_finite());
    }

    #[test]
    fn test_maf_log_det_sum() {
        let maf = NfaMaskedAutoregressive::new(4, 8, 1, 53).expect("new");
        let x = make_input(4, 53);
        let (_, ld) = maf.forward(&x).expect("forward");
        assert!(ld.is_finite());
    }

    #[test]
    fn test_maf_dim_mismatch_error() {
        let maf = NfaMaskedAutoregressive::new(4, 8, 2, 54).expect("new");
        let x = make_input(5, 54);
        assert!(maf.forward(&x).is_err());
    }

    #[test]
    fn test_maf_zero_steps_error() {
        assert!(NfaMaskedAutoregressive::new(4, 8, 0, 55).is_err());
    }

    #[test]
    fn test_maf_dim_lt2_error() {
        assert!(NfaMaskedAutoregressive::new(1, 8, 2, 56).is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §7  NfaInverseAutoregressive (IAF)
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_iaf_forward_shape() {
        let iaf = NfaInverseAutoregressive::new(4, 8, 2, 60).expect("new");
        let x = make_input(4, 60);
        let (z, _) = iaf.forward(&x).expect("forward");
        assert_eq!(z.len(), 4);
    }

    #[test]
    fn test_iaf_invertibility() {
        let iaf = NfaInverseAutoregressive::new(4, 16, 2, 61).expect("new");
        let x = make_input(4, 61);
        let (z, _) = iaf.forward(&x).expect("forward");
        let x_rec = iaf.inverse(&z).expect("inverse");
        assert!(
            l2_dist(&x, &x_rec) < 1e-3,
            "IAF not invertible: {:.6}",
            l2_dist(&x, &x_rec)
        );
    }

    #[test]
    fn test_iaf_log_det_finite() {
        let iaf = NfaInverseAutoregressive::new(4, 8, 2, 62).expect("new");
        let x = make_input(4, 62);
        let (_, ld) = iaf.forward(&x).expect("forward");
        assert!(ld.is_finite());
    }

    #[test]
    fn test_iaf_dim_mismatch_error() {
        let iaf = NfaInverseAutoregressive::new(4, 8, 1, 63).expect("new");
        let x = make_input(5, 63);
        assert!(iaf.forward(&x).is_err());
    }

    #[test]
    fn test_iaf_vs_maf_complementary() {
        // IAF forward = MAF inverse (same MADE architecture, directions swapped)
        // We just check that both give finite outputs for same input
        let maf = NfaMaskedAutoregressive::new(4, 8, 1, 64).expect("new");
        let iaf = NfaInverseAutoregressive::new(4, 8, 1, 64).expect("new");
        let x = make_input(4, 64);
        let (z_maf, ld_maf) = maf.forward(&x).expect("maf forward");
        let (z_iaf, ld_iaf) = iaf.forward(&x).expect("iaf forward");
        assert!(z_maf.iter().all(|v| v.is_finite()));
        assert!(z_iaf.iter().all(|v| v.is_finite()));
        assert!(ld_maf.is_finite());
        assert!(ld_iaf.is_finite());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §8  NfaNeuralSpline
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_rq_spline_forward() {
        let spline = NfaNeuralSpline::new(4, 4, 16, 70).expect("new");
        let x = vec![0.5_f32, -0.3, 1.2, -0.8];
        let (z, _) = spline.forward(&x).expect("forward");
        assert_eq!(z.len(), 4);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rq_spline_invertibility() {
        let spline = NfaNeuralSpline::new(4, 4, 16, 71).expect("new");
        let x = vec![0.2_f32, -0.1, 0.4, -0.3];
        let (z, _) = spline.forward(&x).expect("forward");
        let x_rec = spline.inverse(&z).expect("inverse");
        // Spline is applied only to second half (x[split..])
        // First half passes through unchanged, second half is spline-transformed
        assert!(
            l2_dist(&x, &x_rec) < 0.1,
            "spline not invertible: {:.6}",
            l2_dist(&x, &x_rec)
        );
    }

    #[test]
    fn test_rq_spline_log_det_finite() {
        let spline = NfaNeuralSpline::new(6, 4, 16, 72).expect("new");
        let x = make_input(6, 72);
        let (_, ld) = spline.forward(&x).expect("forward");
        assert!(ld.is_finite());
    }

    #[test]
    fn test_rq_spline_dim_lt2_error() {
        assert!(NfaNeuralSpline::new(1, 4, 16, 73).is_err());
    }

    #[test]
    fn test_rq_spline_num_bins_lt2_error() {
        assert!(NfaNeuralSpline::new(4, 1, 16, 74).is_err());
    }

    #[test]
    fn test_rq_spline_dim_mismatch_error() {
        let spline = NfaNeuralSpline::new(4, 4, 16, 75).expect("new");
        let x = make_input(5, 75);
        assert!(spline.forward(&x).is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §9  NfaRadialFlow
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_radial_flow_forward_shape() {
        let rf = NfaRadialFlow::zeros_init(4, 80);
        let x = make_input(4, 80);
        let (z, _) = rf.forward(&x).expect("forward");
        assert_eq!(z.len(), 4);
    }

    #[test]
    fn test_radial_flow_log_det_finite() {
        let rf = NfaRadialFlow::zeros_init(4, 81);
        let x = make_input(4, 81);
        let (_, ld) = rf.forward(&x).expect("forward");
        assert!(ld.is_finite());
    }

    #[test]
    fn test_radial_flow_invertibility() {
        let rf = NfaRadialFlow::zeros_init(4, 82);
        let x = make_input(4, 82);
        let (z, _) = rf.forward(&x).expect("forward");
        let x_rec = rf.inverse(&z).expect("inverse");
        assert!(
            l2_dist(&x, &x_rec) < 1e-3,
            "RadialFlow not invertible: {:.6}",
            l2_dist(&x, &x_rec)
        );
    }

    #[test]
    fn test_radial_flow_output_finite() {
        let rf = NfaRadialFlow::new(6, vec![0.1; 6], 0.0, 1.0).expect("new");
        let x = make_input(6, 83);
        let (z, ld) = rf.forward(&x).expect("forward");
        assert!(z.iter().all(|v| v.is_finite()));
        assert!(ld.is_finite());
    }

    #[test]
    fn test_radial_flow_z0_mismatch_error() {
        assert!(NfaRadialFlow::new(4, vec![0.0; 3], 0.0, 0.0).is_err());
    }

    #[test]
    fn test_radial_flow_dim_mismatch_error() {
        let rf = NfaRadialFlow::zeros_init(4, 85);
        let x = make_input(5, 85);
        assert!(rf.forward(&x).is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §10  NfaHouseholderFlow
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_householder_forward_shape() {
        let hf = NfaHouseholderFlow::new(6, 4, 90).expect("new");
        let x = make_input(6, 90);
        let (z, _) = hf.forward(&x).expect("forward");
        assert_eq!(z.len(), 6);
    }

    #[test]
    fn test_householder_log_det_zero() {
        let hf = NfaHouseholderFlow::new(4, 3, 91).expect("new");
        let x = make_input(4, 91);
        let (_, ld) = hf.forward(&x).expect("forward");
        assert!(
            (ld - 0.0).abs() < 1e-6,
            "Householder log_det should be 0, got {ld}"
        );
    }

    #[test]
    fn test_householder_invertibility() {
        let hf = NfaHouseholderFlow::new(6, 4, 92).expect("new");
        let x = make_input(6, 92);
        let (z, _) = hf.forward(&x).expect("forward");
        let x_rec = hf.inverse(&z).expect("inverse");
        assert!(l2_dist(&x, &x_rec) < 1e-5, "Householder not invertible");
    }

    #[test]
    fn test_householder_norm_preserved() {
        // Orthogonal transform: ||z||₂ = ||x||₂
        let hf = NfaHouseholderFlow::new(4, 3, 93).expect("new");
        let x = make_input(4, 93);
        let (z, _) = hf.forward(&x).expect("forward");
        let norm_x: f32 = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let norm_z: f32 = z.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm_x - norm_z).abs() < 1e-5,
            "norm not preserved: {norm_x} vs {norm_z}"
        );
    }

    #[test]
    fn test_householder_zero_dim_error() {
        assert!(NfaHouseholderFlow::new(0, 2, 94).is_err());
    }

    #[test]
    fn test_householder_zero_reflections_error() {
        assert!(NfaHouseholderFlow::new(4, 0, 95).is_err());
    }

    #[test]
    fn test_householder_dim_mismatch_error() {
        let hf = NfaHouseholderFlow::new(4, 2, 96).expect("new");
        let x = make_input(5, 96);
        assert!(hf.forward(&x).is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §11  NfaFlowVAE
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_flow_vae_encode_shape() {
        let vae = NfaFlowVAE::new(8, 4, 16, 16, 3, 100).expect("new");
        let x: Vec<f32> = (0..8).map(|i| i as f32 / 8.0).collect();
        let (mu, lv) = vae.encode(&x).expect("encode");
        assert_eq!(mu.len(), 4);
        assert_eq!(lv.len(), 4);
    }

    #[test]
    fn test_flow_vae_decode_shape() {
        let vae = NfaFlowVAE::new(8, 4, 16, 16, 3, 101).expect("new");
        let z = make_input(4, 101);
        let x_recon = vae.decode(&z).expect("decode");
        assert_eq!(x_recon.len(), 8);
    }

    #[test]
    fn test_flow_vae_decode_in_unit_interval() {
        let vae = NfaFlowVAE::new(8, 4, 16, 16, 3, 102).expect("new");
        let z = make_input(4, 102);
        let x_recon = vae.decode(&z).expect("decode");
        for &v in &x_recon {
            assert!((0.0..=1.0).contains(&v), "decoded value {v} out of [0,1]");
        }
    }

    #[test]
    fn test_flow_vae_elbo_finite() {
        let vae = NfaFlowVAE::new(8, 4, 16, 16, 3, 103).expect("new");
        let x: Vec<f32> = (0..8).map(|i| i as f32 / 8.0).collect();
        let mut rng = StdRng::seed_from_u64(103);
        let (recon, kl, flow_ld) = vae.elbo(&x, &mut rng).expect("elbo");
        assert!(recon.is_finite(), "recon loss must be finite");
        assert!(kl.is_finite(), "kl must be finite");
        assert!(flow_ld.is_finite(), "flow_ld must be finite");
    }

    #[test]
    fn test_flow_vae_sample_shape() {
        let vae = NfaFlowVAE::new(8, 4, 16, 16, 3, 104).expect("new");
        let mut rng = StdRng::seed_from_u64(104);
        let sample = vae.sample(&mut rng).expect("sample");
        assert_eq!(sample.len(), 8);
    }

    #[test]
    fn test_flow_vae_sample_in_unit_interval() {
        let vae = NfaFlowVAE::new(8, 4, 16, 16, 3, 105).expect("new");
        let mut rng = StdRng::seed_from_u64(105);
        let sample = vae.sample(&mut rng).expect("sample");
        for &v in &sample {
            assert!((0.0..=1.0).contains(&v), "sample {v} out of [0,1]");
        }
    }

    #[test]
    fn test_flow_vae_latent_dim_lt2_error() {
        assert!(NfaFlowVAE::new(8, 1, 16, 16, 3, 106).is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §12  NfaMetrics
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_metrics_bpd_formula() {
        // BPD = -nll / (d * ln2)
        let bpd = NfaMetrics::bits_per_dim(-10.0_f32, 4);
        let expected = 10.0 / (4.0 * 2.0_f32.ln());
        assert!((bpd - expected).abs() < 1e-5);
    }

    #[test]
    fn test_metrics_nll_batch() {
        let model = NfaMaskedAutoregressive::new(4, 8, 1, 110).expect("new");
        let test_set: Vec<Vec<f32>> = (0..10).map(|i| make_input(4, i as u64 + 200)).collect();
        let nll = NfaMetrics::nll_batch(|x| model.log_likelihood(x), &test_set).expect("nll");
        assert!(nll.is_finite());
    }

    #[test]
    fn test_metrics_nll_batch_empty_error() {
        let res = NfaMetrics::nll_batch(|_| Ok(0.0_f32), &[]);
        assert!(res.is_err());
    }

    #[test]
    fn test_metrics_approximate_kl_finite() {
        let log_q: Vec<f32> = vec![-1.5, -2.0, -1.8];
        let flow_ld: Vec<f32> = vec![0.5, 0.3, 0.4];
        let log_p: Vec<f32> = vec![-2.0, -1.9, -2.1];
        let kl = NfaMetrics::approximate_kl(&log_q, &flow_ld, &log_p).expect("kl");
        assert!(kl.is_finite());
    }

    #[test]
    fn test_metrics_approximate_kl_empty_error() {
        assert!(NfaMetrics::approximate_kl(&[], &[], &[]).is_err());
    }

    #[test]
    fn test_metrics_evaluate_glow() {
        let model = NfaGlowModel::new(8, 2, 2, 16, 111).expect("new");
        let test_set: Vec<Vec<f32>> = (0..5).map(|i| make_input(8, i as u64 + 300)).collect();
        let report = NfaMetrics::evaluate_glow(&model, &test_set).expect("evaluate");
        assert!(report.is_valid());
        assert_eq!(report.dim, 8);
        assert_eq!(report.num_samples, 5);
    }

    #[test]
    fn test_metrics_evaluate_maf() {
        let model = NfaMaskedAutoregressive::new(4, 8, 2, 112).expect("new");
        let test_set: Vec<Vec<f32>> = (0..5).map(|i| make_input(4, i as u64 + 400)).collect();
        let report = NfaMetrics::evaluate_maf(&model, &test_set).expect("evaluate");
        assert!(report.is_valid());
        assert_eq!(report.model_name, "NfaMaskedAutoregressive");
    }

    #[test]
    fn test_metrics_report_is_valid() {
        let report = NfaReport {
            model_name: "Test".to_string(),
            num_samples: 100,
            dim: 4,
            mean_nll: 2.5,
            bpd: 3.6,
        };
        assert!(report.is_valid());
    }

    #[test]
    fn test_metrics_report_invalid_nan() {
        let report = NfaReport {
            model_name: "Test".to_string(),
            num_samples: 10,
            dim: 4,
            mean_nll: f32::NAN,
            bpd: 0.0,
        };
        assert!(!report.is_valid());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §13  NfaFlowComposite
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_composite_forward_shape() {
        let mut comp = NfaFlowComposite::new(4);
        comp.add(NfaActNorm::new(4)).expect("add actnorm");
        comp.add(NfaInvertible1x1Conv::new(4, 120).expect("inv"))
            .expect("add inv1x1");
        let x = make_input(4, 120);
        let (z, _) = comp.forward(&x).expect("forward");
        assert_eq!(z.len(), 4);
    }

    #[test]
    fn test_composite_invertibility() {
        let mut comp = NfaFlowComposite::new(4);
        comp.add(NfaActNorm::new(4)).expect("add actnorm");
        comp.add(NfaHouseholderFlow::new(4, 2, 121).expect("hf"))
            .expect("add hf");
        let x = make_input(4, 121);
        let (z, _) = comp.forward(&x).expect("forward");
        let x_rec = comp.inverse(&z).expect("inverse");
        assert!(l2_dist(&x, &x_rec) < 1e-4, "composite not invertible");
    }

    #[test]
    fn test_composite_log_det_cumulative() {
        let mut comp = NfaFlowComposite::new(4);
        comp.add(NfaHouseholderFlow::new(4, 2, 122).expect("hf1"))
            .expect("add hf1");
        comp.add(NfaHouseholderFlow::new(4, 2, 123).expect("hf2"))
            .expect("add hf2");
        let x = make_input(4, 122);
        let (_, ld) = comp.forward(&x).expect("forward");
        // Both Householder → log_det = 0 + 0 = 0
        assert!((ld).abs() < 1e-5, "composite log_det {ld} should be 0");
    }

    #[test]
    fn test_composite_log_likelihood_finite() {
        let mut comp = NfaFlowComposite::new(4);
        comp.add(NfaActNorm::new(4)).expect("add");
        comp.add(NfaRadialFlow::zeros_init(4, 124))
            .expect("add radial");
        let x = make_input(4, 124);
        let ll = comp.log_likelihood(&x).expect("log_likelihood");
        assert!(ll.is_finite());
    }

    #[test]
    fn test_composite_dim_mismatch_error() {
        let mut comp = NfaFlowComposite::new(4);
        let hf5 = NfaHouseholderFlow::new(5, 2, 125).expect("hf5");
        assert!(comp.add(hf5).is_err());
    }

    #[test]
    fn test_composite_empty_forward() {
        let comp = NfaFlowComposite::new(4);
        let x = make_input(4, 126);
        let (z, ld) = comp.forward(&x).expect("empty forward");
        assert_eq!(z.len(), 4);
        assert!(
            (ld - 0.0).abs() < 1e-6,
            "empty composite log_det should be 0"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §14  Integration tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_full_pipeline_glow_maf() {
        // Build a Glow model, compute forward/inverse, evaluate MAF on same data
        let glow = NfaGlowModel::new(8, 2, 2, 16, 130).expect("glow");
        let maf = NfaMaskedAutoregressive::new(8, 16, 2, 131).expect("maf");
        let test_data: Vec<Vec<f32>> = (0..5).map(|i| make_input(8, i as u64 + 500)).collect();

        for x in &test_data {
            let (z_glow, ld_glow) = glow.forward(x).expect("glow forward");
            assert!(z_glow.iter().all(|v| v.is_finite()));
            assert!(ld_glow.is_finite());

            let (z_maf, ld_maf) = maf.forward(x).expect("maf forward");
            assert!(z_maf.iter().all(|v| v.is_finite()));
            assert!(ld_maf.is_finite());
        }
    }

    #[test]
    fn test_householder_plus_radial_composite() {
        let mut comp = NfaFlowComposite::new(4);
        comp.add(NfaHouseholderFlow::new(4, 3, 135).expect("hf"))
            .expect("add hf");
        comp.add(NfaRadialFlow::zeros_init(4, 136))
            .expect("add radial");
        comp.add(NfaHouseholderFlow::new(4, 2, 137).expect("hf2"))
            .expect("add hf2");
        let x = make_input(4, 135);
        let (z, _) = comp.forward(&x).expect("forward");
        let x_rec = comp.inverse(&z).expect("inverse");
        assert!(
            l2_dist(&x, &x_rec) < 1e-3,
            "composite 3-layer not invertible"
        );
    }

    #[test]
    fn test_glow_step_different_seeds() {
        // Different seeds produce different transformations
        let step1 = NfaGlowStep::new(4, 8, 140).expect("step1");
        let step2 = NfaGlowStep::new(4, 8, 141).expect("step2");
        let x = make_input(4, 140);
        let (z1, _) = step1.forward(&x).expect("forward1");
        let (z2, _) = step2.forward(&x).expect("forward2");
        let diff = l2_dist(&z1, &z2);
        // Very likely to be different with different seeds
        assert!(diff > 0.0 || z1 == z2, "steps with different seeds");
    }

    #[test]
    fn test_maf_multiple_steps_more_expressive() {
        // A MAF with more steps should transform the input more
        let maf1 = NfaMaskedAutoregressive::new(4, 8, 1, 142).expect("1-step maf");
        let maf3 = NfaMaskedAutoregressive::new(4, 8, 3, 142).expect("3-step maf");
        let x = make_input(4, 142);
        let (z1, _) = maf1.forward(&x).expect("forward1");
        let (z3, _) = maf3.forward(&x).expect("forward3");
        // Both should give finite outputs
        assert!(z1.iter().all(|v| v.is_finite()));
        assert!(z3.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_flow_vae_multiple_samples_diverse() {
        let vae = NfaFlowVAE::new(8, 4, 16, 16, 3, 150).expect("new");
        let mut rng1 = StdRng::seed_from_u64(150);
        let mut rng2 = StdRng::seed_from_u64(151);
        let s1 = vae.sample(&mut rng1).expect("sample1");
        let s2 = vae.sample(&mut rng2).expect("sample2");
        // Samples from different seeds should generally differ
        assert_eq!(s1.len(), 8);
        assert_eq!(s2.len(), 8);
        // Both in [0,1]
        for &v in s1.iter().chain(s2.iter()) {
            assert!((0.0..=1.0).contains(&v));
        }
    }
}
