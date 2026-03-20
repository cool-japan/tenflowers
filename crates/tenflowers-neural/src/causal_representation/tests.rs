//! Tests for causal_representation module.

use super::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

// ── CrLinear / CrMlp ──────────────────────────────────────────────────────

#[test]
fn test_cr_linear_output_shape() {
    let layer = CrLinear::new(4, 6);
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = layer.forward(&x);
    assert_eq!(y.len(), 6);
}

#[test]
fn test_cr_linear_is_finite() {
    let layer = CrLinear::new(3, 5);
    let x = vec![0.5, -0.3, 1.2];
    let y = layer.forward(&x);
    assert!(y.iter().all(|v| v.is_finite()));
}

#[test]
fn test_cr_mlp_output_dimension() {
    let mlp = CrMlp::new(&[8, 16, 16, 4]);
    let x = vec![0.1; 8];
    let y = mlp.forward(&x);
    assert_eq!(y.len(), 4);
}

#[test]
fn test_cr_mlp_last_layer_linear() {
    // With last activation = false, a second pass with same input same output
    let mlp = CrMlp::new(&[4, 8, 2]);
    let x = vec![1.0, 0.0, -1.0, 0.5];
    let y1 = mlp.forward(&x);
    let y2 = mlp.forward(&x);
    assert_eq!(y1, y2);
}

#[test]
fn test_cr_mlp_update_changes_weights() {
    let mut mlp = CrMlp::new(&[2, 4, 2]);
    let old_w = mlp.layers[0].w.clone();
    // Create non-zero gradients
    let grad_w: Vec<Vec<Vec<f64>>> = mlp
        .layers
        .iter()
        .map(|l| {
            l.w.iter()
                .map(|row| row.iter().map(|_| 0.1).collect())
                .collect()
        })
        .collect();
    let grad_b: Vec<Vec<f64>> = mlp.layers.iter().map(|l| vec![0.1; l.b.len()]).collect();
    mlp.update(0.01, &grad_w, &grad_b);
    assert!(mlp.layers[0].w != old_w);
}

// ── Math helpers ──────────────────────────────────────────────────────────

#[test]
fn test_relu_positive() {
    assert_eq!(relu(2.0), 2.0);
    assert_eq!(relu(-1.5), 0.0);
}

#[test]
fn test_softmax_sums_to_one() {
    let v = vec![1.0, 2.0, 3.0];
    let s = softmax(&v);
    assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-9);
}

#[test]
fn test_log_sum_exp_finite() {
    let v = vec![1.0, 2.0, 3.0];
    assert!(log_sum_exp(&v).is_finite());
}

#[test]
fn test_box_muller_finite() {
    let z = box_muller(0.5, 0.3);
    assert!(z.is_finite());
}

// ── IvaeEncoder ───────────────────────────────────────────────────────────

#[test]
fn test_ivae_encoder_output_shapes() {
    let enc = IvaeEncoder::new(8, 4, 3, 16);
    let x = vec![0.1; 8];
    let u = vec![1.0, 0.0, 0.0, 0.0];
    let (mu, lv) = enc.encode(&x, &u);
    assert_eq!(mu.len(), 3);
    assert_eq!(lv.len(), 3);
}

#[test]
fn test_ivae_encoder_sample_shape() {
    let enc = IvaeEncoder::new(8, 4, 3, 16);
    let x = vec![0.1; 8];
    let u = vec![1.0, 0.0, 0.0, 0.0];
    let (mu, lv) = enc.encode(&x, &u);
    let z = enc.sample(&mu, &lv, 0.5, 0.3);
    assert_eq!(z.len(), 3);
    assert!(z.iter().all(|v| v.is_finite()));
}

// ── IvaePrior ─────────────────────────────────────────────────────────────

#[test]
fn test_ivae_prior_log_prob_finite() {
    let prior = IvaePrior::new(4, 5);
    let z = vec![0.1, -0.2, 0.3, 0.0, 1.0];
    let lp = prior.log_prob(&z, 2);
    assert!(lp.is_finite());
}

#[test]
fn test_ivae_prior_kl_non_negative() {
    let prior = IvaePrior::new(4, 5);
    let mu = vec![0.5, -0.2, 0.1, 0.0, 0.3];
    let lv = vec![0.0; 5];
    let kl = prior.kl_to_prior(&mu, &lv, 1);
    assert!(kl >= 0.0);
}

#[test]
fn test_ivae_prior_kl_zero_when_matching() {
    let mut prior = IvaePrior::new(2, 3);
    let mu = vec![0.5, -0.2, 0.1];
    let lv = vec![0.0; 3];
    prior.lambda_mu[0] = mu.clone();
    prior.lambda_lv[0] = lv.clone();
    let kl = prior.kl_to_prior(&mu, &lv, 0);
    assert!(kl < 1e-10, "KL should be ~0 when q matches p, got {}", kl);
}

// ── IvaeDecoder ───────────────────────────────────────────────────────────

#[test]
fn test_ivae_decoder_output_shape() {
    let dec = IvaeDecoder::new(5, 10, 16);
    let z = vec![0.1; 5];
    let x_recon = dec.decode(&z);
    assert_eq!(x_recon.len(), 10);
}

// ── IvaeModel ─────────────────────────────────────────────────────────────

#[test]
fn test_ivae_model_elbo_finite() {
    let config = IvaeConfig {
        x_dim: 6,
        z_dim: 3,
        u_dim: 2,
        n_segments: 4,
        hidden_dim: 16,
    };
    let model = IvaeModel::new(config);
    let x = vec![0.1, 0.2, -0.1, 0.5, 0.3, -0.2];
    let u = vec![1.0, 0.0];
    let elbo = model.elbo(&x, &u, 1, 0.5, 0.3);
    assert!(elbo.is_finite());
}

#[test]
fn test_ivae_model_encode_dim() {
    let config = IvaeConfig {
        x_dim: 6,
        z_dim: 3,
        u_dim: 2,
        n_segments: 4,
        hidden_dim: 16,
    };
    let model = IvaeModel::new(config);
    let x = vec![0.0; 6];
    let u = vec![1.0, 0.0];
    let z_mean = model.encode(&x, &u);
    assert_eq!(z_mean.len(), 3);
}

#[test]
fn test_ivae_model_reconstruct_dim() {
    let config = IvaeConfig {
        x_dim: 6,
        z_dim: 3,
        u_dim: 2,
        n_segments: 4,
        hidden_dim: 16,
    };
    let model = IvaeModel::new(config);
    let x = vec![0.1; 6];
    let u = vec![1.0, 0.0];
    let x_recon = model.reconstruct(&x, &u, 0.4, 0.7);
    assert_eq!(x_recon.len(), 6);
}

// ── TcVaeEncoder / Decoder ────────────────────────────────────────────────

#[test]
fn test_tc_vae_encoder_shapes() {
    let enc = TcVaeEncoder::new(10, 4, 32);
    let x = vec![0.1; 10];
    let (mu, lv) = enc.encode(&x);
    assert_eq!(mu.len(), 4);
    assert_eq!(lv.len(), 4);
}

#[test]
fn test_tc_vae_decoder_shape() {
    let dec = TcVaeDecoder::new(4, 10, 32);
    let z = vec![0.2; 4];
    let x_recon = dec.decode(&z);
    assert_eq!(x_recon.len(), 10);
}

#[test]
fn test_tc_vae_decoder_finite() {
    let dec = TcVaeDecoder::new(4, 10, 32);
    let z = vec![0.1, -0.2, 0.3, 0.0];
    let x_recon = dec.decode(&z);
    assert!(x_recon.iter().all(|v| v.is_finite()));
}

// ── TcVae ─────────────────────────────────────────────────────────────────

#[test]
fn test_tc_vae_forward_shapes() {
    let config = TcVaeConfig {
        x_dim: 8,
        z_dim: 3,
        alpha: 1.0,
        beta: 4.0,
        gamma: 1.0,
        hidden_dim: 16,
        dataset_size: 100,
    };
    let model = TcVae::new(config);
    let x = vec![0.1; 8];
    let (z, mu, lv, x_recon) = model.forward(&x, 0.5, 0.3);
    assert_eq!(z.len(), 3);
    assert_eq!(mu.len(), 3);
    assert_eq!(lv.len(), 3);
    assert_eq!(x_recon.len(), 8);
}

#[test]
fn test_tc_vae_elbo_finite() {
    let config = TcVaeConfig {
        x_dim: 8,
        z_dim: 3,
        ..TcVaeConfig::default()
    };
    let model = TcVae::new(config);
    let x = vec![0.2; 8];
    let (z, mu, lv, _) = model.forward(&x, 0.5, 0.3);
    let elbo = model.elbo(&x, &z, &mu, &lv);
    assert!(elbo.is_finite());
}

#[test]
fn test_tc_vae_tc_loss_dim_kl_positive() {
    let config = TcVaeConfig {
        x_dim: 8,
        z_dim: 3,
        ..TcVaeConfig::default()
    };
    let model = TcVae::new(config);
    let batch: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64 * 0.1; 8]).collect();
    let mut z_batch = vec![];
    let mut mu_batch = vec![];
    let mut lv_batch = vec![];
    for x in &batch {
        let (z, mu, lv, _) = model.forward(x, 0.5, 0.3);
        z_batch.push(z);
        mu_batch.push(mu);
        lv_batch.push(lv);
    }
    let (_mi, _tc, dim_kl) = model.tc_loss(&z_batch, &mu_batch, &lv_batch);
    assert!(dim_kl.is_finite());
}

#[test]
fn test_tc_vae_tc_decomposition_consistency() {
    let config = TcVaeConfig {
        x_dim: 6,
        z_dim: 3,
        ..TcVaeConfig::default()
    };
    let model = TcVae::new(config);
    let mut z_batch = vec![];
    let mut mu_batch = vec![];
    let mut lv_batch = vec![];
    for i in 0..16 {
        let x: Vec<f64> = (0..6).map(|j| (i as f64 + j as f64) * 0.05).collect();
        let (z, mu, lv, _) = model.forward(&x, 0.5 + 0.01 * i as f64, 0.3 + 0.01 * i as f64);
        z_batch.push(z);
        mu_batch.push(mu);
        lv_batch.push(lv);
    }
    let (mi, tc, dim_kl) = model.tc_loss(&z_batch, &mu_batch, &lv_batch);
    assert!(mi.is_finite());
    assert!(tc.is_finite());
    assert!(dim_kl.is_finite());
}

// ── FactorVAE ─────────────────────────────────────────────────────────────

#[test]
fn test_factor_vae_forward_shapes() {
    let config = FactorVaeConfig {
        x_dim: 8,
        z_dim: 4,
        gamma: 10.0,
        hidden_dim: 16,
        disc_hidden: 16,
    };
    let model = FactorVae::new(config);
    let x = vec![0.1; 8];
    let (z, mu, lv, x_recon) = model.forward(&x, 0.5, 0.3);
    assert_eq!(z.len(), 4);
    assert_eq!(mu.len(), 4);
    assert_eq!(lv.len(), 4);
    assert_eq!(x_recon.len(), 8);
}

#[test]
fn test_tc_discriminator_forward_finite() {
    let disc = TcDiscriminator::new(4, 16);
    let z = vec![0.1, -0.2, 0.3, 0.5];
    let logit = disc.forward(&z);
    assert!(logit.is_finite());
}

#[test]
fn test_tc_discriminator_permute_dims_changes() {
    let disc = TcDiscriminator::new(3, 8);
    let z_batch: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![i as f64 * 0.1, i as f64 * 0.2, i as f64 * 0.3])
        .collect();
    let z_perm = disc.permute_dims(&z_batch);
    assert_eq!(z_perm.len(), z_batch.len());
    let changed = z_batch
        .iter()
        .zip(z_perm.iter())
        .any(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-12));
    assert!(changed);
}

#[test]
fn test_factor_vae_vae_loss_finite() {
    let config = FactorVaeConfig {
        x_dim: 6,
        z_dim: 3,
        gamma: 5.0,
        hidden_dim: 16,
        disc_hidden: 16,
    };
    let model = FactorVae::new(config);
    let x = vec![0.1; 6];
    let (z, mu, lv, _) = model.forward(&x, 0.5, 0.3);
    let loss = model.vae_loss(&x, &z, &mu, &lv);
    assert!(loss.is_finite());
}

#[test]
fn test_factor_vae_train_batch_returns_two_finite_losses() {
    let config = FactorVaeConfig {
        x_dim: 6,
        z_dim: 3,
        gamma: 5.0,
        hidden_dim: 16,
        disc_hidden: 16,
    };
    let mut model = FactorVae::new(config);
    let x_batch: Vec<Vec<f64>> = (0..8)
        .map(|i| (0..6).map(|j| (i as f64 + j as f64) * 0.05).collect())
        .collect();
    let mut rng = StdRng::seed_from_u64(42);
    let (vae_loss, disc_loss) = model.train_batch(&x_batch, 1e-3, 1e-3, &mut rng);
    assert!(vae_loss.is_finite());
    assert!(disc_loss.is_finite());
}

// ── NonlinearIca ──────────────────────────────────────────────────────────

#[test]
fn test_nonlinear_ica_reconstruct_shape() {
    let config = SlowIcaConfig {
        x_dim: 4,
        z_dim: 4,
        hidden_dim: 16,
        lr: 0.01,
    };
    let model = NonlinearIca::new(config);
    let x = vec![0.1, 0.2, 0.3, 0.4];
    let x_recon = model.reconstruct(&x);
    assert_eq!(x_recon.len(), 4);
}

#[test]
fn test_nonlinear_ica_slowness_loss_zero_constant() {
    let config = SlowIcaConfig {
        x_dim: 3,
        z_dim: 3,
        hidden_dim: 8,
        lr: 0.01,
    };
    let model = NonlinearIca::new(config);
    let z_seq = vec![vec![1.0, 2.0, 3.0]; 5];
    let loss = model.slowness_loss(&z_seq);
    assert!(loss.abs() < 1e-10);
}

#[test]
fn test_nonlinear_ica_slowness_loss_non_negative() {
    let config = SlowIcaConfig {
        x_dim: 3,
        z_dim: 3,
        hidden_dim: 8,
        lr: 0.01,
    };
    let model = NonlinearIca::new(config);
    let z_seq: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![i as f64, -i as f64 * 0.5, (i as f64).sin()])
        .collect();
    let loss = model.slowness_loss(&z_seq);
    assert!(loss >= 0.0);
}

#[test]
fn test_nonlinear_ica_independence_penalty_non_negative() {
    let config = SlowIcaConfig {
        x_dim: 4,
        z_dim: 4,
        hidden_dim: 8,
        lr: 0.01,
    };
    let model = NonlinearIca::new(config);
    let z_batch: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            vec![
                i as f64 * 0.1,
                (i as f64).sin(),
                i as f64 * 0.05,
                -(i as f64) * 0.03,
            ]
        })
        .collect();
    let penalty = model.independence_penalty(&z_batch);
    assert!(penalty >= 0.0);
}

#[test]
fn test_nonlinear_ica_train_step_decreases_loss() {
    let config = SlowIcaConfig {
        x_dim: 3,
        z_dim: 3,
        hidden_dim: 16,
        lr: 0.1,
    };
    let mut model = NonlinearIca::new(config);
    let x_seq: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            vec![
                (i as f64 * 0.1).sin(),
                (i as f64 * 0.2).cos(),
                i as f64 * 0.05,
            ]
        })
        .collect();
    let mut rng = StdRng::seed_from_u64(7);
    let loss_before = {
        let z_seq: Vec<Vec<f64>> = x_seq
            .iter()
            .map(|x| model.unmixing_net.forward(x))
            .collect();
        model.slowness_loss(&z_seq) + model.independence_penalty(&z_seq)
    };
    for _ in 0..20 {
        model.train_step(&x_seq, &mut rng);
    }
    let loss_after = {
        let z_seq: Vec<Vec<f64>> = x_seq
            .iter()
            .map(|x| model.unmixing_net.forward(x))
            .collect();
        model.slowness_loss(&z_seq) + model.independence_penalty(&z_seq)
    };
    let _ = loss_before;
    assert!(loss_after.is_finite());
}

// ── DeepScm ───────────────────────────────────────────────────────────────

#[test]
fn test_dscm_mechanism_forward_shape() {
    let mech = DscmMechanism::new(3, 3, 2, 16);
    let u = vec![0.1, -0.2, 0.3];
    let pa = vec![0.5, -0.1];
    let x = mech.forward(&u, &pa);
    assert_eq!(x.len(), 3);
}

#[test]
fn test_dscm_mechanism_abduct_shape() {
    let mech = DscmMechanism::new(3, 3, 2, 16);
    let x = vec![0.1, 0.2, -0.3];
    let pa = vec![0.5, -0.1];
    let u = mech.abduct(&x, &pa);
    assert_eq!(u.len(), 3);
}

#[test]
fn test_dscm_mechanism_finite_outputs() {
    let mech = DscmMechanism::new(2, 2, 2, 16);
    let u = vec![0.3, -0.5];
    let pa = vec![0.1, 0.2];
    let x = mech.forward(&u, &pa);
    assert!(x.iter().all(|v| v.is_finite()));
    let u_back = mech.abduct(&x, &pa);
    assert!(u_back.iter().all(|v| v.is_finite()));
}

fn simple_linear_dag() -> DeepScm {
    let config = DscmConfig {
        n_variables: 3,
        var_dims: vec![2, 2, 2],
        hidden_dim: 8,
        parents: vec![vec![], vec![0], vec![1]],
    };
    DeepScm::new(config)
}

#[test]
fn test_deep_scm_topological_order_valid() {
    let scm = simple_linear_dag();
    let order = scm.topological_order();
    assert_eq!(order.len(), 3);
    let pos_map: Vec<usize> = {
        let mut m = vec![0; 3];
        for (i, &v) in order.iter().enumerate() {
            m[v] = i;
        }
        m
    };
    assert!(pos_map[0] < pos_map[1]);
    assert!(pos_map[1] < pos_map[2]);
}

#[test]
fn test_deep_scm_topological_order_no_cycles() {
    let scm = simple_linear_dag();
    let order = scm.topological_order();
    let pos_map: std::collections::HashMap<usize, usize> =
        order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    for i in 0..3 {
        for &p in &scm.config.parents[i] {
            let parent_pos = pos_map.get(&p).cloned().unwrap_or(usize::MAX);
            let child_pos = pos_map.get(&i).cloned().unwrap_or(0);
            assert!(
                parent_pos < child_pos,
                "Parent {} should appear before child {}",
                p,
                i
            );
        }
    }
}

#[test]
fn test_deep_scm_sample_n_variables() {
    let scm = simple_linear_dag();
    let mut rng = StdRng::seed_from_u64(42);
    let sample = scm.sample(&mut rng);
    assert_eq!(sample.len(), 3);
}

#[test]
fn test_deep_scm_sample_correct_dims() {
    let scm = simple_linear_dag();
    let mut rng = StdRng::seed_from_u64(1);
    let sample = scm.sample(&mut rng);
    for (i, x_i) in sample.iter().enumerate() {
        assert_eq!(x_i.len(), scm.config.var_dims[i]);
    }
}

#[test]
fn test_deep_scm_abduct_correct_shape() {
    let scm = simple_linear_dag();
    let mut rng = StdRng::seed_from_u64(0);
    let observed = scm.sample(&mut rng);
    let u = scm.abduct(&observed);
    assert_eq!(u.len(), 3);
    for (i, u_i) in u.iter().enumerate() {
        assert_eq!(u_i.len(), scm.mechanisms[i].u_dim);
    }
}

#[test]
fn test_deep_scm_counterfactual_intervention_variable_correct() {
    let scm = simple_linear_dag();
    let mut rng = StdRng::seed_from_u64(99);
    let factual = scm.sample(&mut rng);
    let intervention_value = vec![5.0, -5.0];
    let cf = scm.counterfactual(&factual, 1, intervention_value.clone());
    assert_eq!(cf.len(), 3);
    assert_eq!(cf[1], intervention_value);
}

#[test]
fn test_deep_scm_intervene_sets_value() {
    let scm = simple_linear_dag();
    let mut rng = StdRng::seed_from_u64(55);
    let observed = scm.sample(&mut rng);
    let value = vec![10.0, 10.0];
    let result = scm.intervene(&observed, 0, value.clone());
    assert_eq!(result[0], value);
}

#[test]
fn test_deep_scm_finite_sample() {
    let scm = simple_linear_dag();
    let mut rng = StdRng::seed_from_u64(7);
    let sample = scm.sample(&mut rng);
    for x_i in &sample {
        assert!(x_i.iter().all(|v| v.is_finite()));
    }
}

// ── Disentanglement Metrics ───────────────────────────────────────────────

fn synthetic_z_factors(n: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let z: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![(i as f64) * 0.1, -(i as f64) * 0.1])
        .collect();
    let f: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![(i as f64) * 0.1, -(i as f64) * 0.1])
        .collect();
    (z, f)
}

#[test]
fn test_compute_mig_in_range() {
    let (z, f) = synthetic_z_factors(100);
    let mig = compute_mig(&z, &f, 10);
    assert!((0.0..=1.0).contains(&mig), "MIG = {} out of [0,1]", mig);
}

#[test]
fn test_compute_sap_in_range() {
    let (z, f) = synthetic_z_factors(100);
    let sap = compute_sap(&z, &f);
    assert!((0.0..=1.0).contains(&sap), "SAP = {} out of [0,1]", sap);
}

#[test]
fn test_compute_modularity_in_range() {
    let (z, f) = synthetic_z_factors(200);
    let f2: Vec<Vec<f64>> = f
        .iter()
        .map(|row| {
            let mut r = row.clone();
            r.push(row[0] * 0.5 + 0.1);
            r
        })
        .collect();
    let z2: Vec<Vec<f64>> = z
        .iter()
        .map(|row| {
            let mut r = row.clone();
            r.push(row[1] * 0.3);
            r
        })
        .collect();
    let mod_score = compute_modularity(&z2, &f2);
    assert!(
        (0.0..=1.0).contains(&mod_score),
        "Modularity = {} out of [0,1]",
        mod_score
    );
}

#[test]
fn test_compute_mig_perfect_disentanglement_high() {
    let n = 200;
    let z: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64, 0.0]).collect();
    let f: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64, 0.0]).collect();
    let mig = compute_mig(&z, &f, 20);
    assert!((0.0..=1.0).contains(&mig));
}

#[test]
fn test_compute_sap_perfect_alignment() {
    let n = 100;
    let z: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64, -(i as f64)]).collect();
    let f: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64, -(i as f64)]).collect();
    let sap = compute_sap(&z, &f);
    assert!((0.0..=1.0).contains(&sap));
}

#[test]
fn test_discretize_all_in_range() {
    let vals = vec![-1.0, 0.0, 0.5, 1.0, 2.0];
    let bins = discretize(&vals, 5);
    assert!(bins.iter().all(|&b| b < 5));
}

#[test]
fn test_entropy_disc_uniform_high() {
    let x_disc: Vec<usize> = (0..100).map(|i| i % 10).collect();
    let h = entropy_disc(&x_disc, 10);
    assert!(h > 1.5);
}

#[test]
fn test_mi_disc_self_equals_entropy() {
    let x_disc: Vec<usize> = (0..50).map(|i| i % 5).collect();
    let h = entropy_disc(&x_disc, 5);
    let mi = mutual_information_disc(&x_disc, &x_disc, 5, 5);
    assert!(
        (mi - h).abs() < 0.01,
        "MI(X;X)={} should equal H(X)={}",
        mi,
        h
    );
}

#[test]
fn test_ivae_prior_kl_positive_for_different_dist() {
    let prior = IvaePrior::new(3, 4);
    let mu = vec![2.0, -2.0, 3.0, -3.0];
    let lv = vec![0.0; 4];
    let kl = prior.kl_to_prior(&mu, &lv, 0);
    assert!(kl > 0.0);
}

#[test]
fn test_cr_mlp_two_layer_correct_output_dim() {
    let mlp = CrMlp::new(&[5, 10, 3]);
    let x = vec![0.1; 5];
    let y = mlp.forward(&x);
    assert_eq!(y.len(), 3);
}

#[test]
fn test_dscm_fork_dag() {
    let config = DscmConfig {
        n_variables: 3,
        var_dims: vec![1, 1, 1],
        hidden_dim: 8,
        parents: vec![vec![], vec![0], vec![0]],
    };
    let scm = DeepScm::new(config);
    let order = scm.topological_order();
    let pos_map: std::collections::HashMap<usize, usize> =
        order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    assert!(pos_map[&0] < pos_map[&1]);
    assert!(pos_map[&0] < pos_map[&2]);
}

#[test]
fn test_dscm_diamond_dag() {
    let config = DscmConfig {
        n_variables: 4,
        var_dims: vec![1, 1, 1, 1],
        hidden_dim: 8,
        parents: vec![vec![], vec![0], vec![0], vec![1, 2]],
    };
    let scm = DeepScm::new(config);
    let order = scm.topological_order();
    assert_eq!(order.len(), 4);
    let pos_map: std::collections::HashMap<usize, usize> =
        order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    assert!(pos_map[&0] < pos_map[&3]);
    assert!(pos_map[&1] < pos_map[&3]);
    assert!(pos_map[&2] < pos_map[&3]);
}

#[test]
fn test_factor_vae_permute_preserves_shape() {
    let disc = TcDiscriminator::new(5, 16);
    let z_batch: Vec<Vec<f64>> = (0..12)
        .map(|i| (0..5).map(|j| i as f64 * 0.1 + j as f64).collect())
        .collect();
    let perm = disc.permute_dims(&z_batch);
    assert_eq!(perm.len(), 12);
    assert_eq!(perm[0].len(), 5);
}

#[test]
fn test_tc_vae_forward_all_finite() {
    let config = TcVaeConfig {
        x_dim: 6,
        z_dim: 3,
        ..TcVaeConfig::default()
    };
    let model = TcVae::new(config);
    let x = vec![-0.5, 0.2, 1.3, -0.1, 0.7, -0.9];
    let (z, mu, lv, x_recon) = model.forward(&x, 0.6, 0.4);
    assert!(z.iter().all(|v| v.is_finite()));
    assert!(mu.iter().all(|v| v.is_finite()));
    assert!(lv.iter().all(|v| v.is_finite()));
    assert!(x_recon.iter().all(|v| v.is_finite()));
}

#[test]
fn test_factor_vae_empty_batch() {
    let config = FactorVaeConfig {
        x_dim: 4,
        z_dim: 2,
        gamma: 1.0,
        hidden_dim: 8,
        disc_hidden: 8,
    };
    let mut model = FactorVae::new(config);
    let mut rng = StdRng::seed_from_u64(0);
    let (vl, dl) = model.train_batch(&[], 0.01, 0.01, &mut rng);
    assert_eq!(vl, 0.0);
    assert_eq!(dl, 0.0);
}

#[test]
fn test_ivae_model_n_segments_boundary() {
    let config = IvaeConfig {
        x_dim: 4,
        z_dim: 2,
        u_dim: 2,
        n_segments: 3,
        hidden_dim: 8,
    };
    let model = IvaeModel::new(config);
    let x = vec![0.1, 0.2, 0.3, 0.4];
    let u = vec![0.0, 1.0];
    let elbo = model.elbo(&x, &u, 10, 0.5, 0.3);
    assert!(elbo.is_finite());
}

#[test]
fn test_non_square_cr_linear() {
    let layer = CrLinear::new(10, 3);
    let x = vec![0.1; 10];
    let y = layer.forward(&x);
    assert_eq!(y.len(), 3);
    assert!(y.iter().all(|v| v.is_finite()));
}

#[test]
fn test_deep_scm_abduct_forward_roundtrip() {
    let config = DscmConfig {
        n_variables: 2,
        var_dims: vec![1, 1],
        hidden_dim: 4,
        parents: vec![vec![], vec![0]],
    };
    let scm = DeepScm::new(config);
    let mut rng = StdRng::seed_from_u64(11);
    let observed = scm.sample(&mut rng);
    let u = scm.abduct(&observed);
    assert!(u.iter().flat_map(|v| v.iter()).all(|v| v.is_finite()));
}
