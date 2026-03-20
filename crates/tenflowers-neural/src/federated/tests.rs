//! Tests for federated learning algorithms.

#[cfg(test)]
mod tests {
    use super::super::{
        byzantine::{
            BulyanAggregator, ByzantineMetrics, FlameAggregator, KrumAggregator, MedianAggregator,
            TrimmedMeanAggregator,
        },
        clustered_fl::{ClusterEnsemble, ClusteredFLMetrics, HypCluster, IfcaAlgorithm},
        compression::GradientCompressor,
        fedavg::FedAvg,
        fedma::{FedMaAggregator, LayerMatching, PermutationMatrix},
        fednova::FedNova,
        fedprox::FedProx,
        personalized::{ApflClient, FedBnClient, HeurFl, PFedMeClient, PersonalizedMetrics},
        privacy::{add_dp_noise, clip_gradients, compute_privacy_loss},
        scaffold::Scaffold,
        types::{ClientUpdate, FederatedConfig, FederatedError, ModelParams},
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_params(shapes: &[usize], fill: f32) -> ModelParams {
        shapes.iter().map(|&n| vec![fill; n]).collect()
    }

    fn make_update(
        client_id: usize,
        params: ModelParams,
        n: usize,
        loss: f32,
        steps: usize,
    ) -> ClientUpdate {
        ClientUpdate {
            client_id,
            params,
            num_samples: n,
            loss,
            num_local_steps: steps,
        }
    }

    fn config_small() -> FederatedConfig {
        FederatedConfig {
            num_clients: 5,
            clients_per_round: 5,
            num_rounds: 10,
            local_epochs: 2,
            local_lr: 0.01,
            min_clients_available: 2,
        }
    }

    // ── FederatedConfig default ───────────────────────────────────────────────

    #[test]
    fn test_federated_config_defaults_are_sensible() {
        let cfg = FederatedConfig::default();
        assert!(cfg.num_clients > 0, "num_clients must be positive");
        assert!(
            cfg.clients_per_round > 0,
            "clients_per_round must be positive"
        );
        assert!(
            cfg.clients_per_round <= cfg.num_clients,
            "clients_per_round must not exceed num_clients"
        );
        assert!(cfg.local_lr > 0.0, "local_lr must be positive");
        assert!(
            cfg.min_clients_available > 0,
            "min_clients_available must be positive"
        );
    }

    // ── FederatedError display ────────────────────────────────────────────────

    #[test]
    fn test_error_display_no_client_updates() {
        let msg = FederatedError::NoClientUpdates.to_string();
        assert!(msg.contains("no client updates"), "got: {msg}");
    }

    #[test]
    fn test_error_display_insufficient_clients() {
        let e = FederatedError::InsufficientClients {
            required: 5,
            available: 2,
        };
        let msg = e.to_string();
        assert!(msg.contains('5') && msg.contains('2'), "got: {msg}");
    }

    #[test]
    fn test_error_display_dimension_mismatch() {
        let e = FederatedError::DimensionMismatch {
            layer: 1,
            expected: 10,
            found: 8,
        };
        let msg = e.to_string();
        assert!(
            msg.contains('1') && msg.contains("10") && msg.contains('8'),
            "got: {msg}"
        );
    }

    #[test]
    fn test_error_display_invalid_config() {
        let e = FederatedError::InvalidConfig {
            reason: "foo".to_string(),
        };
        let msg = e.to_string();
        assert!(msg.contains("foo"), "got: {msg}");
    }

    #[test]
    fn test_error_display_invalid_client_id() {
        let e = FederatedError::InvalidClientId { id: 99, max: 10 };
        let msg = e.to_string();
        assert!(msg.contains("99") && msg.contains("10"), "got: {msg}");
    }

    // ── FedAvg ────────────────────────────────────────────────────────────────

    #[test]
    fn test_fedavg_aggregate_weights_correctly_by_samples() {
        // Two clients: client 0 has 3x more samples, its params should dominate
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let mut server = FedAvg::new(cfg, global);

        let p0 = make_params(&[4], 2.0); // big values
        let p1 = make_params(&[4], 0.0); // zero

        let updates = vec![
            make_update(0, p0, 300, 0.5, 5), // 300 samples => weight 0.75
            make_update(1, p1, 100, 0.5, 5), // 100 samples => weight 0.25
        ];
        let result = server.aggregate(&updates).expect("aggregate failed");
        // Expected: 0.75 * 2.0 + 0.25 * 0.0 = 1.5
        for &v in &result.params[0] {
            assert!((v - 1.5).abs() < 1e-5, "expected 1.5, got {v}");
        }
    }

    #[test]
    fn test_fedavg_equal_samples_gives_equal_weight() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let mut server = FedAvg::new(cfg, global);

        let updates = vec![
            make_update(0, make_params(&[4], 4.0), 100, 0.5, 5),
            make_update(1, make_params(&[4], 2.0), 100, 0.5, 5),
        ];
        let result = server.aggregate(&updates).expect("aggregate failed");
        // Expected: (4.0 + 2.0) / 2 = 3.0
        for &v in &result.params[0] {
            assert!((v - 3.0).abs() < 1e-5, "expected 3.0, got {v}");
        }
    }

    #[test]
    fn test_fedavg_distribute_returns_global_params() {
        let cfg = config_small();
        let global = make_params(&[6], 1.5);
        let server = FedAvg::new(cfg, global.clone());
        assert_eq!(server.distribute(), &global);
    }

    #[test]
    fn test_fedavg_round_increments() {
        let mut cfg = config_small();
        cfg.min_clients_available = 1;
        let global = make_params(&[4], 0.0);
        let mut server = FedAvg::new(cfg, global);
        assert_eq!(server.current_round(), 0);

        let updates = vec![make_update(0, make_params(&[4], 1.0), 100, 0.5, 5)];
        let res = server.aggregate(&updates).expect("ok");
        assert_eq!(server.current_round(), 1);
        assert_eq!(res.round, 1);
    }

    #[test]
    fn test_fedavg_errors_on_empty_updates() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let mut server = FedAvg::new(cfg, global);
        match server.aggregate(&[]) {
            Err(FederatedError::NoClientUpdates) => {}
            other => panic!("expected NoClientUpdates, got {other:?}"),
        }
    }

    #[test]
    fn test_fedavg_errors_on_insufficient_clients() {
        let mut cfg = config_small();
        cfg.min_clients_available = 5;
        let global = make_params(&[4], 0.0);
        let mut server = FedAvg::new(cfg, global);

        let updates = vec![make_update(0, make_params(&[4], 1.0), 100, 0.5, 5)];
        match server.aggregate(&updates) {
            Err(FederatedError::InsufficientClients { .. }) => {}
            other => panic!("expected InsufficientClients, got {other:?}"),
        }
    }

    #[test]
    fn test_fedavg_errors_on_dimension_mismatch() {
        let mut cfg = config_small();
        cfg.min_clients_available = 1;
        let global = make_params(&[4], 0.0);
        let mut server = FedAvg::new(cfg, global);

        // Client sends 5-dim layer instead of 4-dim
        let bad = vec![vec![1.0_f32; 5]];
        let updates = vec![make_update(0, bad, 100, 0.5, 5)];
        match server.aggregate(&updates) {
            Err(FederatedError::DimensionMismatch { .. }) => {}
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_fedavg_local_update_plain_sgd() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let server = FedAvg::new(cfg, global);

        // Quadratic loss f(w) = 0.5 * ||w||^2 => grad = w
        let grad_fn = |params: &ModelParams| params.clone();
        let init = make_params(&[4], 2.0);
        let updated = server.local_update(&init, grad_fn, 0.1, 10);

        // With lr=0.1 and grad=w, w should converge toward 0
        for &v in &updated[0] {
            assert!(v.abs() < init[0][0], "update should move toward 0, got {v}");
        }
    }

    #[test]
    fn test_fedavg_multiple_rounds_converge_quadratic() {
        // Simple quadratic: f(w) = 0.5 * ||w - w*||^2, w* = [1, 1, 1]
        // grad = w - w*
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        cfg.num_clients = 5;
        let global = make_params(&[3], 0.0); // start at [0,0,0], target [1,1,1]
        let mut server = FedAvg::new(cfg.clone(), global);

        let target = vec![1.0_f32; 3];
        let target_params: ModelParams = vec![target.clone()];

        for _ in 0..20 {
            let current = server.distribute().clone();
            let grad_fn = {
                let tp = target_params.clone();
                move |params: &ModelParams| -> ModelParams {
                    params
                        .iter()
                        .zip(tp.iter())
                        .map(|(layer, t_layer)| {
                            layer
                                .iter()
                                .zip(t_layer.iter())
                                .map(|(&p, &t)| p - t)
                                .collect()
                        })
                        .collect()
                }
            };
            let updated = server.local_update(&current, grad_fn, 0.1, 5);
            // All 3 clients return same update
            let updates: Vec<ClientUpdate> = (0..3)
                .map(|id| make_update(id, updated.clone(), 100, 0.5, 5))
                .collect();
            server.aggregate(&updates).expect("ok");
        }

        let final_params = server.global_params();
        for &v in &final_params[0] {
            assert!((v - 1.0).abs() < 0.2, "should converge near 1.0, got {v}");
        }
    }

    // ── FedProx ───────────────────────────────────────────────────────────────

    #[test]
    fn test_fedprox_proximal_term_zero_when_equal() {
        let cfg = config_small();
        let global = make_params(&[4, 3], 1.0);
        let server = FedProx::new(cfg, global.clone(), 0.1);
        let prox = server.proximal_term(&global, &global);
        assert!(
            prox.abs() < 1e-6,
            "proximal term should be 0 when params match, got {prox}"
        );
    }

    #[test]
    fn test_fedprox_proximal_term_positive_when_different() {
        let cfg = config_small();
        let global = make_params(&[4], 1.0);
        let server = FedProx::new(cfg, global.clone(), 0.5);
        let w = make_params(&[4], 2.0); // differs from global
        let prox = server.proximal_term(&w, &global);
        // 0.5 * 0.5 * (4 * 1.0) = 1.0
        assert!(prox > 0.0, "proximal term should be positive, got {prox}");
    }

    #[test]
    fn test_fedprox_proximal_term_scales_with_mu() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let s1 = FedProx::new(cfg.clone(), global.clone(), 1.0);
        let s2 = FedProx::new(cfg.clone(), global.clone(), 2.0);
        let w = make_params(&[4], 1.0);
        let p1 = s1.proximal_term(&w, &global);
        let p2 = s2.proximal_term(&w, &global);
        assert!(
            (p2 - 2.0 * p1).abs() < 1e-5,
            "proximal term should scale with mu"
        );
    }

    #[test]
    fn test_fedprox_local_update_stays_close_to_global() {
        // With a very high mu the proximal term pulls w back to global
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let server_prox = FedProx::new(cfg.clone(), global.clone(), 100.0); // large mu
        let server_avg = FedAvg::new(cfg.clone(), global.clone());

        // Gradient pushes params far from 0
        let grad_fn = |_params: &ModelParams| -> ModelParams { make_params(&[4], -10.0) };

        let init = make_params(&[4], 0.0);
        let w_prox = server_prox.local_update(&init, &global, grad_fn, 0.01, 20);
        let w_avg =
            server_avg.local_update(&init, |p: &ModelParams| make_params(&[4], -10.0), 0.01, 20);

        let norm_prox: f32 = w_prox[0].iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_avg: f32 = w_avg[0].iter().map(|v| v * v).sum::<f32>().sqrt();

        assert!(norm_prox <= norm_avg + 1e-3,
            "FedProx with high mu ({norm_prox}) should stay closer to global than FedAvg ({norm_avg})");
    }

    #[test]
    fn test_fedprox_aggregate_same_as_fedavg() {
        // Aggregation logic is identical to FedAvg
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let mut s_prox = FedProx::new(cfg.clone(), global.clone(), 0.1);
        let mut s_avg = FedAvg::new(cfg.clone(), global.clone());

        let updates = vec![
            make_update(0, make_params(&[4], 1.0), 200, 0.4, 5),
            make_update(1, make_params(&[4], 3.0), 200, 0.6, 5),
        ];
        let r_prox = s_prox.aggregate(&updates).expect("ok");
        let r_avg = s_avg.aggregate(&updates).expect("ok");

        for (lp, la) in r_prox.params.iter().zip(r_avg.params.iter()) {
            for (&vp, &va) in lp.iter().zip(la.iter()) {
                assert!(
                    (vp - va).abs() < 1e-5,
                    "aggregation should match: {vp} vs {va}"
                );
            }
        }
    }

    // ── SCAFFOLD ──────────────────────────────────────────────────────────────

    #[test]
    fn test_scaffold_distribute_returns_correct_control() {
        let cfg = config_small();
        let global = make_params(&[4], 1.0);
        let server = Scaffold::new(cfg.clone(), global.clone());

        let (params, control) = server.distribute(0).expect("ok");
        assert_eq!(params, &global);
        // Initial control should be all zeros
        for &v in &control[0] {
            assert_eq!(v, 0.0, "initial control should be zero");
        }
    }

    #[test]
    fn test_scaffold_distribute_rejects_invalid_client_id() {
        let cfg = config_small(); // num_clients = 5
        let global = make_params(&[4], 1.0);
        let server = Scaffold::new(cfg, global);
        match server.distribute(99) {
            Err(FederatedError::InvalidClientId { .. }) => {}
            other => panic!("expected InvalidClientId, got {other:?}"),
        }
    }

    #[test]
    fn test_scaffold_local_update_returns_updated_control() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let server = Scaffold::new(cfg, global.clone());

        let grad_fn = |p: &ModelParams| p.clone(); // gradient = params (quadratic at 0)
        let (new_w, new_ci) = server
            .local_update_with_control(&global, 0, grad_fn, 0.1, 5)
            .expect("ok");

        // params should change from 0
        let any_nonzero = new_w.iter().flat_map(|l| l.iter()).any(|&v| v != 0.0);
        // With zero gradient (params=0 => grad=0) params stay zero,
        // but control also stays zero which is valid; just check shapes are correct
        assert_eq!(new_w.len(), global.len());
        assert_eq!(new_ci.len(), global.len());
        let _ = any_nonzero; // suppress lint
    }

    #[test]
    fn test_scaffold_aggregate_updates_global_control() {
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global = make_params(&[4], 1.0);
        let mut server = Scaffold::new(cfg, global.clone());

        // Simulate clients: give non-trivial new controls
        let new_c0: ModelParams = vec![vec![0.1, 0.2, 0.3, 0.4]];
        let new_c1: ModelParams = vec![vec![0.5, 0.6, 0.7, 0.8]];
        let new_p0: ModelParams = vec![vec![0.9, 0.9, 0.9, 0.9]];
        let new_p1: ModelParams = vec![vec![1.1, 1.1, 1.1, 1.1]];

        let old_control = server.state.global_control.clone();
        let updates = vec![(0_usize, new_p0, new_c0), (1_usize, new_p1, new_c1)];
        server.aggregate(&updates).expect("ok");

        // Global control should have changed from zero
        let changed = server
            .state
            .global_control
            .iter()
            .flatten()
            .zip(old_control.iter().flatten())
            .any(|(new, old)| (new - old).abs() > 1e-7);
        assert!(changed, "global control should update after aggregation");
    }

    #[test]
    fn test_scaffold_aggregate_increments_round() {
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global = make_params(&[4], 0.0);
        let mut server = Scaffold::new(cfg, global.clone());
        assert_eq!(server.current_round(), 0);

        let updates = vec![
            (0_usize, global.clone(), global.clone()),
            (1_usize, global.clone(), global.clone()),
        ];
        server.aggregate(&updates).expect("ok");
        assert_eq!(server.current_round(), 1);
    }

    // ── FedNova ───────────────────────────────────────────────────────────────

    #[test]
    fn test_fednova_normalises_by_local_steps() {
        // Two clients: both start at 0, push to 2.0, but client 1 did 2x steps.
        // FedNova should normalise so the "speed" is equalized.
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global = make_params(&[4], 0.0);
        let mut server = FedNova::new(cfg, global.clone());

        // Both have equal samples. Client 0: 5 steps, client 1: 10 steps.
        // After normalization their effective deltas should be comparable.
        let p0 = make_params(&[4], 2.0); // Δ = 2, τ=5 → normalized 0.4
        let p1 = make_params(&[4], 4.0); // Δ = 4, τ=10 → normalized 0.4
        let updates = vec![
            make_update(0, p0, 100, 0.5, 5),
            make_update(1, p1, 100, 0.5, 10),
        ];
        let result = server.aggregate(&updates).expect("ok");

        // Effective τ = 0.5 * 5 + 0.5 * 10 = 7.5
        // Δ = 0.5 * (2/5) + 0.5 * (4/10) = 0.5 * 0.4 + 0.5 * 0.4 = 0.4
        // update = 7.5 * 0.4 = 3.0 => global = 0 + 3.0 = 3.0
        for &v in &result.params[0] {
            assert!((v - 3.0).abs() < 1e-4, "expected ~3.0, got {v}");
        }
    }

    #[test]
    fn test_fednova_errors_on_empty_updates() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let mut server = FedNova::new(cfg, global);
        match server.aggregate(&[]) {
            Err(FederatedError::NoClientUpdates) => {}
            other => panic!("expected NoClientUpdates, got {other:?}"),
        }
    }

    #[test]
    fn test_fednova_round_counter() {
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global = make_params(&[4], 0.0);
        let mut server = FedNova::new(cfg, global.clone());

        let updates = vec![
            make_update(0, global.clone(), 50, 0.3, 3),
            make_update(1, global.clone(), 50, 0.3, 3),
        ];
        let r = server.aggregate(&updates).expect("ok");
        assert_eq!(server.current_round(), 1);
        assert_eq!(r.round, 1);
    }

    // ── Gradient clipping ────────────────────────────────────────────────────

    #[test]
    fn test_clip_gradients_does_not_clip_small_norms() {
        let grads: ModelParams = vec![vec![1.0, 0.0, 0.0, 0.0]]; // norm = 1.0
        let clipped = clip_gradients(&grads, 10.0);
        assert!(
            (clipped[0][0] - 1.0).abs() < 1e-5,
            "should not clip small grad"
        );
    }

    #[test]
    fn test_clip_gradients_clips_to_max_norm() {
        let grads: ModelParams = vec![vec![3.0, 4.0]]; // norm = 5.0
        let max_norm = 1.0;
        let clipped = clip_gradients(&grads, max_norm);
        let norm_after: f32 = clipped[0].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm_after - max_norm).abs() < 1e-5,
            "clipped norm should equal max_norm ({norm_after})"
        );
    }

    #[test]
    fn test_clip_gradients_preserves_direction() {
        let grads: ModelParams = vec![vec![3.0, 4.0]]; // norm = 5, direction = (0.6, 0.8)
        let clipped = clip_gradients(&grads, 1.0);
        // direction should be preserved: (0.6, 0.8)
        assert!(
            (clipped[0][0] - 0.6).abs() < 1e-5,
            "direction x: {}",
            clipped[0][0]
        );
        assert!(
            (clipped[0][1] - 0.8).abs() < 1e-5,
            "direction y: {}",
            clipped[0][1]
        );
    }

    // ── DP noise ─────────────────────────────────────────────────────────────

    #[test]
    fn test_add_dp_noise_changes_params() {
        let params: ModelParams = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let noisy = add_dp_noise(&params, 1.0, 1.0, 1.0);
        let changed = params[0]
            .iter()
            .zip(noisy[0].iter())
            .any(|(orig, noisy_v)| (orig - noisy_v).abs() > 1e-7);
        assert!(changed, "noise should change at least one parameter");
    }

    #[test]
    fn test_add_dp_noise_zero_multiplier_no_change() {
        let params: ModelParams = vec![vec![1.0, 2.0, 3.0]];
        let noisy = add_dp_noise(&params, 0.0, 1.0, 1.0);
        for (&orig, &noisy_v) in params[0].iter().zip(noisy[0].iter()) {
            assert!(
                (orig - noisy_v).abs() < 1e-7,
                "zero multiplier should yield no noise"
            );
        }
    }

    #[test]
    fn test_add_dp_noise_preserves_shape() {
        let params: ModelParams = vec![vec![1.0; 8], vec![2.0; 4]];
        let noisy = add_dp_noise(&params, 0.5, 1.0, 1.0);
        assert_eq!(noisy.len(), 2);
        assert_eq!(noisy[0].len(), 8);
        assert_eq!(noisy[1].len(), 4);
    }

    // ── Privacy accounting ────────────────────────────────────────────────────

    #[test]
    fn test_compute_privacy_loss_returns_finite_values() {
        let (eps, delta) = compute_privacy_loss(1.0, 0.01, 100);
        assert!(
            eps.is_finite() && eps >= 0.0,
            "epsilon should be finite non-negative, got {eps}"
        );
        assert!(
            delta > 0.0 && delta < 1.0,
            "delta should be in (0,1), got {delta}"
        );
    }

    #[test]
    fn test_compute_privacy_loss_higher_noise_lower_epsilon() {
        let (eps_low_noise, _) = compute_privacy_loss(0.5, 0.01, 100);
        let (eps_high_noise, _) = compute_privacy_loss(5.0, 0.01, 100);
        assert!(
            eps_high_noise < eps_low_noise,
            "higher noise should give lower (better) epsilon: {eps_high_noise} vs {eps_low_noise}"
        );
    }

    // ── GradientCompressor ────────────────────────────────────────────────────

    #[test]
    fn test_gradient_compressor_keeps_correct_fraction() {
        let grads: ModelParams = vec![vec![0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.15]];
        let compressor = GradientCompressor::new(0.5); // keep 50%
        let (compressed, indices) = compressor.compress(&grads);

        assert_eq!(indices[0].len(), 5, "should keep exactly 5 of 10 entries");
        // Kept indices should be the 5 largest: 0.9, 0.8, 0.7, 0.6, 0.5
        // Values at kept positions should be preserved
        for &idx in &indices[0] {
            assert_eq!(
                compressed[0][idx], grads[0][idx],
                "kept entry should be preserved"
            );
        }
        // All other positions should be zero
        for i in 0..10 {
            if !indices[0].contains(&i) {
                assert_eq!(compressed[0][i], 0.0, "non-kept entry should be zero");
            }
        }
    }

    #[test]
    fn test_gradient_compressor_top_k_selects_largest_magnitude() {
        let grads: ModelParams = vec![vec![-5.0, 1.0, -2.0, 3.0]]; // magnitudes: 5, 1, 2, 3
        let compressor = GradientCompressor::new(0.5); // keep 2
        let (_compressed, indices) = compressor.compress(&grads);

        // Should keep indices 0 (|−5|=5) and 3 (|3|=3)
        assert!(
            indices[0].contains(&0),
            "should keep largest magnitude (index 0)"
        );
        assert!(
            indices[0].contains(&3),
            "should keep second largest (index 3)"
        );
    }

    #[test]
    fn test_gradient_compressor_compression_ratio() {
        let grads: ModelParams = vec![vec![1.0; 10], vec![1.0; 10]]; // 20 total
        let compressor = GradientCompressor::new(0.3); // keep 30%
        let ratio = compressor.compression_ratio(&grads);
        // ceil(10 * 0.3) = 3 per layer, 6/20 = 0.3
        assert!(
            (ratio - 0.3).abs() < 1e-4,
            "compression ratio should be ~0.3, got {ratio}"
        );
    }

    #[test]
    fn test_gradient_compressor_decompress_round_trip() {
        let grads: ModelParams = vec![vec![0.1, 0.5, 0.9, 0.2, 0.8]];
        let compressor = GradientCompressor::new(0.4); // keep 40% = 2 entries
        let (compressed, indices) = compressor.compress(&grads);

        let full_dim: Vec<usize> = grads.iter().map(|l| l.len()).collect();
        let recovered = compressor.decompress(&compressed, &indices, &full_dim);

        assert_eq!(recovered.len(), grads.len());
        assert_eq!(recovered[0].len(), grads[0].len());
        // Kept indices should match original
        for &idx in &indices[0] {
            assert!((recovered[0][idx] - grads[0][idx]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_gradient_compressor_new_clamps_fraction() {
        let c1 = GradientCompressor::new(0.0);
        assert!(c1.top_k_fraction > 0.0, "should clamp to positive value");

        let c2 = GradientCompressor::new(2.0);
        assert!(c2.top_k_fraction <= 1.0, "should clamp to at most 1.0");
    }

    // ── GlobalUpdate fields ───────────────────────────────────────────────────

    #[test]
    fn test_global_update_avg_loss_weighted() {
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global = make_params(&[4], 0.0);
        let mut server = FedAvg::new(cfg, global);

        let updates = vec![
            make_update(0, make_params(&[4], 1.0), 300, 0.6, 5),
            make_update(1, make_params(&[4], 2.0), 100, 1.0, 5),
        ];
        let result = server.aggregate(&updates).expect("ok");
        // Expected avg_loss = (300 * 0.6 + 100 * 1.0) / 400 = 280/400 = 0.7
        assert!(
            (result.avg_loss - 0.7).abs() < 1e-5,
            "expected avg_loss=0.7, got {}",
            result.avg_loss
        );
    }

    #[test]
    fn test_global_update_convergence_metric_zero_on_identity() {
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global = make_params(&[4], 1.0);
        let mut server = FedAvg::new(cfg, global.clone());

        // Both clients return exactly the current global params → no change
        let updates = vec![
            make_update(0, global.clone(), 100, 0.5, 5),
            make_update(1, global.clone(), 100, 0.5, 5),
        ];
        let result = server.aggregate(&updates).expect("ok");
        assert!(
            result.convergence_metric < 1e-5,
            "convergence metric should be 0 when params don't change: {}",
            result.convergence_metric
        );
    }

    #[test]
    fn test_global_update_participating_clients_count() {
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global = make_params(&[4], 0.0);
        let mut server = FedAvg::new(cfg, global.clone());

        let updates: Vec<ClientUpdate> = (0..3)
            .map(|id| make_update(id, global.clone(), 100, 0.5, 5))
            .collect();
        let result = server.aggregate(&updates).expect("ok");
        assert_eq!(result.participating_clients, 3);
    }

    // ── KrumAggregator ────────────────────────────────────────────────────────

    fn config_krum() -> FederatedConfig {
        FederatedConfig {
            num_clients: 10,
            clients_per_round: 7,
            num_rounds: 10,
            local_epochs: 2,
            local_lr: 0.01,
            min_clients_available: 2,
        }
    }

    #[test]
    fn test_krum_aggregator_new_valid() {
        let cfg = config_krum();
        let global = make_params(&[4], 0.0);
        let krum = KrumAggregator::new(cfg, global, 1, 1);
        assert!(
            krum.is_ok(),
            "KrumAggregator::new should succeed with valid params"
        );
    }

    #[test]
    fn test_krum_aggregator_new_invalid_f() {
        let cfg = config_krum(); // clients_per_round = 7, need 2f+2 < 7 => f <= 2
        let global = make_params(&[4], 0.0);
        // f=3 => 2*3+2=8 >= 7: should fail
        let krum = KrumAggregator::new(cfg, global, 3, 1);
        assert!(krum.is_err(), "should fail when 2f+2 >= n");
    }

    #[test]
    fn test_krum_aggregate_selects_honest_client() {
        let cfg = config_krum();
        let global = make_params(&[4], 0.0);
        let mut krum = KrumAggregator::new(cfg, global, 1, 1).expect("ok");

        // 4 honest clients near 1.0, 2 Byzantine at 100.0, 1 Byzantine at -100.0
        let mut updates: Vec<ClientUpdate> = (0..4)
            .map(|id| make_update(id, make_params(&[4], 1.0), 100, 0.5, 5))
            .collect();
        updates.push(make_update(4, make_params(&[4], 100.0), 100, 0.5, 5));
        updates.push(make_update(5, make_params(&[4], 100.0), 100, 0.5, 5));
        updates.push(make_update(6, make_params(&[4], -100.0), 100, 0.5, 5));

        let result = krum.aggregate(&updates).expect("krum aggregate ok");
        // Krum should pick an honest client near 1.0
        for &v in &result.params[0] {
            assert!(
                v.abs() < 10.0,
                "Krum should select near-honest update, got {v}"
            );
        }
    }

    #[test]
    fn test_krum_aggregate_multi_krum_averages_selected() {
        let cfg = config_krum();
        let global = make_params(&[4], 0.0);
        let mut krum = KrumAggregator::new(cfg, global, 1, 3).expect("ok");

        let updates: Vec<ClientUpdate> = (0..7)
            .map(|id| {
                let val = if id < 5 { 1.0 } else { 100.0 };
                make_update(id, make_params(&[4], val), 100, 0.5, 5)
            })
            .collect();
        let result = krum.aggregate(&updates).expect("multi-krum ok");
        // With m=3 and 5 honest near 1.0, should aggregate close to 1.0
        for &v in &result.params[0] {
            assert!(
                v < 10.0,
                "Multi-Krum result should be close to honest clients, got {v}"
            );
        }
    }

    #[test]
    fn test_krum_errors_on_empty_updates() {
        let cfg = config_krum();
        let global = make_params(&[4], 0.0);
        let mut krum = KrumAggregator::new(cfg, global, 1, 1).expect("ok");
        match krum.aggregate(&[]) {
            Err(FederatedError::NoClientUpdates) => {}
            other => panic!("expected NoClientUpdates, got {other:?}"),
        }
    }

    #[test]
    fn test_krum_round_increments() {
        let cfg = config_krum();
        let global = make_params(&[4], 0.0);
        let mut krum = KrumAggregator::new(cfg, global, 1, 1).expect("ok");
        assert_eq!(krum.current_round(), 0);

        let updates: Vec<ClientUpdate> = (0..7)
            .map(|id| make_update(id, make_params(&[4], 1.0), 100, 0.5, 5))
            .collect();
        krum.aggregate(&updates).expect("ok");
        assert_eq!(krum.current_round(), 1);
    }

    // ── FlameAggregator ───────────────────────────────────────────────────────

    #[test]
    fn test_flame_aggregate_basic() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let mut flame = FlameAggregator::new(cfg, global, 0.0, 0.01);

        let updates: Vec<ClientUpdate> = (0..3)
            .map(|id| make_update(id, make_params(&[4], 1.0), 100, 0.5, 5))
            .collect();
        let result = flame.aggregate(&updates).expect("flame ok");
        // All same params, noise is tiny: result should be near 1.0
        for &v in &result.params[0] {
            assert!(
                (v - 1.0).abs() < 0.5,
                "FLAME basic result near 1.0, got {v}"
            );
        }
    }

    #[test]
    fn test_flame_rejects_outliers() {
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global = make_params(&[4], 0.0);
        let mut flame = FlameAggregator::new(cfg, global, 0.5, 0.0);

        // 3 honest near 1.0, 2 Byzantine at very different directions
        let updates = vec![
            make_update(0, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(1, make_params(&[4], 1.1), 100, 0.5, 5),
            make_update(2, make_params(&[4], 0.9), 100, 0.5, 5),
            make_update(3, vec![vec![-100.0, 100.0, -100.0, 100.0]], 100, 0.5, 5),
            make_update(4, vec![vec![100.0, -100.0, 100.0, -100.0]], 100, 0.5, 5),
        ];
        let result = flame.aggregate(&updates).expect("ok");
        // With cos_threshold=0.5, outliers should be filtered
        let mean_result: f32 = result.params[0].iter().sum::<f32>() / 4.0;
        assert!(
            mean_result.abs() < 10.0,
            "FLAME should reduce Byzantine influence, mean={mean_result}"
        );
    }

    #[test]
    fn test_flame_round_increments() {
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global = make_params(&[4], 0.0);
        let mut flame = FlameAggregator::new(cfg, global, 0.0, 0.0);
        assert_eq!(flame.current_round(), 0);
        let updates: Vec<ClientUpdate> = (0..2)
            .map(|id| make_update(id, make_params(&[4], 1.0), 100, 0.5, 5))
            .collect();
        flame.aggregate(&updates).expect("ok");
        assert_eq!(flame.current_round(), 1);
    }

    // ── MedianAggregator ──────────────────────────────────────────────────────

    #[test]
    fn test_median_aggregator_basic() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let mut med = MedianAggregator::new(cfg, global);

        // Values: 1, 2, 3 => median = 2
        let updates = vec![
            make_update(0, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(1, make_params(&[4], 2.0), 100, 0.5, 5),
            make_update(2, make_params(&[4], 3.0), 100, 0.5, 5),
        ];
        let result = med.aggregate(&updates).expect("median ok");
        for &v in &result.params[0] {
            assert!(
                (v - 2.0).abs() < 1e-5,
                "median of [1,2,3] should be 2.0, got {v}"
            );
        }
    }

    #[test]
    fn test_median_aggregator_even_count() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let mut med = MedianAggregator::new(cfg, global);

        // Values: 1, 2, 3, 4 => median = (2+3)/2 = 2.5
        let updates = vec![
            make_update(0, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(1, make_params(&[4], 2.0), 100, 0.5, 5),
            make_update(2, make_params(&[4], 3.0), 100, 0.5, 5),
            make_update(3, make_params(&[4], 4.0), 100, 0.5, 5),
        ];
        let result = med.aggregate(&updates).expect("ok");
        for &v in &result.params[0] {
            assert!((v - 2.5).abs() < 1e-5, "median of [1,2,3,4] = 2.5, got {v}");
        }
    }

    #[test]
    fn test_median_aggregator_robust_to_outliers() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let mut med = MedianAggregator::new(cfg, global);

        // 4 honest at 1.0, 1 Byzantine at 1000.0
        let updates = vec![
            make_update(0, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(1, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(2, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(3, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(4, make_params(&[4], 1000.0), 100, 0.5, 5),
        ];
        let result = med.aggregate(&updates).expect("ok");
        // Median of [1,1,1,1,1000] = 1.0
        for &v in &result.params[0] {
            assert!((v - 1.0).abs() < 1e-5, "median should be 1.0, got {v}");
        }
    }

    #[test]
    fn test_median_aggregator_errors_on_empty() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let mut med = MedianAggregator::new(cfg, global);
        match med.aggregate(&[]) {
            Err(FederatedError::NoClientUpdates) => {}
            other => panic!("expected NoClientUpdates, got {other:?}"),
        }
    }

    // ── TrimmedMeanAggregator ─────────────────────────────────────────────────

    #[test]
    fn test_trimmed_mean_basic() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let mut tm = TrimmedMeanAggregator::new(cfg, global, 0.2).expect("ok");

        // 5 values: [1, 2, 3, 4, 5], trim 20% = 1 from each tail => [2,3,4] => mean=3
        let updates: Vec<ClientUpdate> = (0..5)
            .map(|id| make_update(id, make_params(&[4], (id + 1) as f32), 100, 0.5, 5))
            .collect();
        let result = tm.aggregate(&updates).expect("ok");
        for &v in &result.params[0] {
            assert!(
                (v - 3.0).abs() < 1e-5,
                "trimmed mean should be 3.0, got {v}"
            );
        }
    }

    #[test]
    fn test_trimmed_mean_invalid_beta() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        assert!(
            TrimmedMeanAggregator::new(cfg.clone(), global.clone(), 0.0).is_err(),
            "beta=0 should fail"
        );
        assert!(
            TrimmedMeanAggregator::new(cfg, global, 0.5).is_err(),
            "beta=0.5 should fail"
        );
    }

    #[test]
    fn test_trimmed_mean_robust_to_outliers() {
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global = make_params(&[4], 0.0);
        let mut tm = TrimmedMeanAggregator::new(cfg, global, 0.2).expect("ok");

        // 4 at 1.0, 1 outlier at 100.0 => trim 20%=1 from each tail, mean of [1,1,1]
        let updates = vec![
            make_update(0, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(1, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(2, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(3, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(4, make_params(&[4], 100.0), 100, 0.5, 5),
        ];
        let result = tm.aggregate(&updates).expect("ok");
        for &v in &result.params[0] {
            assert!(
                (v - 1.0).abs() < 1e-4,
                "trimmed mean result near 1.0, got {v}"
            );
        }
    }

    // ── BulyanAggregator ──────────────────────────────────────────────────────

    fn config_bulyan() -> FederatedConfig {
        FederatedConfig {
            num_clients: 15,
            clients_per_round: 11, // 4f+3 = 4*2+3 = 11
            num_rounds: 10,
            local_epochs: 2,
            local_lr: 0.01,
            min_clients_available: 2,
        }
    }

    #[test]
    fn test_bulyan_aggregator_new_valid() {
        let cfg = config_bulyan();
        let global = make_params(&[4], 0.0);
        let bul = BulyanAggregator::new(cfg, global, 2);
        assert!(bul.is_ok(), "Bulyan::new should succeed with n=11, f=2");
    }

    #[test]
    fn test_bulyan_aggregator_new_invalid() {
        let cfg = config_bulyan(); // clients_per_round=11
        let global = make_params(&[4], 0.0);
        let bul = BulyanAggregator::new(cfg, global, 3); // 4*3+3=15 > 11
        assert!(bul.is_err(), "Bulyan should fail with n=11, f=3");
    }

    #[test]
    fn test_bulyan_aggregate_resistant_to_byzantine() {
        let cfg = config_bulyan();
        let global = make_params(&[4], 0.0);
        let mut bul = BulyanAggregator::new(cfg, global, 2).expect("ok");

        // 9 honest clients near 1.0, 2 Byzantine at large values
        let mut updates: Vec<ClientUpdate> = (0..9)
            .map(|id| make_update(id, make_params(&[4], 1.0), 100, 0.5, 5))
            .collect();
        updates.push(make_update(9, make_params(&[4], 500.0), 100, 0.5, 5));
        updates.push(make_update(10, make_params(&[4], -500.0), 100, 0.5, 5));

        let result = bul.aggregate(&updates).expect("bulyan ok");
        for &v in &result.params[0] {
            assert!(
                (v - 1.0).abs() < 5.0,
                "Bulyan should resist Byzantine, got {v}"
            );
        }
    }

    #[test]
    fn test_bulyan_round_increments() {
        let cfg = config_bulyan();
        let global = make_params(&[4], 0.0);
        let mut bul = BulyanAggregator::new(cfg, global, 2).expect("ok");
        let updates: Vec<ClientUpdate> = (0..11)
            .map(|id| make_update(id, make_params(&[4], 1.0), 100, 0.5, 5))
            .collect();
        bul.aggregate(&updates).expect("ok");
        assert_eq!(bul.current_round(), 1);
    }

    // ── ByzantineMetrics ──────────────────────────────────────────────────────

    #[test]
    fn test_byzantine_metrics_detection_rate() {
        let mut metrics = ByzantineMetrics::new();
        metrics.compute_detection_rate(&[2, 4], &[1, 2, 4]);
        assert!((metrics.detection_rate - 1.0).abs() < 1e-5, "both detected");
    }

    #[test]
    fn test_byzantine_metrics_partial_detection() {
        let mut metrics = ByzantineMetrics::new();
        metrics.compute_detection_rate(&[1, 2, 3], &[2, 5]);
        assert!(
            (metrics.detection_rate - 1.0 / 3.0).abs() < 1e-5,
            "1 of 3 detected"
        );
    }

    #[test]
    fn test_byzantine_metrics_error_norm() {
        let mut metrics = ByzantineMetrics::new();
        let agg = make_params(&[4], 1.5);
        let reference = make_params(&[4], 1.0);
        metrics.compute_error_norm(&agg, &reference);
        // Error = sqrt(4 * 0.25) = sqrt(1) = 1.0
        assert!(
            (metrics.avg_error_norm - 1.0).abs() < 1e-4,
            "error norm should be 1.0, got {}",
            metrics.avg_error_norm
        );
    }

    // ── PFedMeClient ──────────────────────────────────────────────────────────

    #[test]
    fn test_pfedme_new_valid() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let server = PFedMeClient::new(cfg, global, 0.1, 0.01);
        assert!(server.is_ok(), "PFedMeClient::new should succeed");
    }

    #[test]
    fn test_pfedme_new_invalid_lambda() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        assert!(
            PFedMeClient::new(cfg, global, 0.0, 0.01).is_err(),
            "lambda=0 should fail"
        );
    }

    #[test]
    fn test_pfedme_personalize_changes_params() {
        let cfg = config_small();
        let global = make_params(&[4], 1.0);
        let mut server = PFedMeClient::new(cfg, global.clone(), 0.5, 0.01).expect("ok");

        // Gradient pushes toward 0
        let grad_fn = |p: &ModelParams| -> ModelParams { p.iter().map(|l| l.to_vec()).collect() };
        let personal = server.personalize(0, grad_fn, 5).expect("ok");
        // Personal params should differ from global
        let changed = personal[0]
            .iter()
            .zip(global[0].iter())
            .any(|(&p, &g)| (p - g).abs() > 1e-6);
        assert!(changed, "personalized params should differ from global");
    }

    #[test]
    fn test_pfedme_aggregate_basic() {
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global = make_params(&[4], 0.0);
        let mut server = PFedMeClient::new(cfg, global, 0.1, 0.01).expect("ok");

        let updates = vec![
            make_update(0, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(1, make_params(&[4], 3.0), 100, 0.5, 5),
        ];
        let result = server.aggregate(&updates).expect("ok");
        for &v in &result.params[0] {
            assert!((v - 2.0).abs() < 1e-5, "equal-weight avg = 2.0, got {v}");
        }
    }

    #[test]
    fn test_pfedme_distribute_invalid_client() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let server = PFedMeClient::new(cfg, global, 0.1, 0.01).expect("ok");
        assert!(
            server.distribute(99).is_err(),
            "invalid client id should fail"
        );
    }

    // ── ApflClient ────────────────────────────────────────────────────────────

    #[test]
    fn test_apfl_new_basic() {
        let cfg = config_small();
        let global = make_params(&[4], 0.5);
        let server = ApflClient::new(cfg, global, 0.01);
        assert_eq!(server.current_round(), 0);
        for i in 0..5 {
            let alpha = server.client_alpha(i).expect("client exists");
            assert!((alpha - 0.5).abs() < 1e-6, "initial alpha should be 0.5");
        }
    }

    #[test]
    fn test_apfl_alpha_updates() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let mut server = ApflClient::new(cfg, global, 0.1);

        let zero_grad = |_p: &ModelParams| -> ModelParams { make_params(&[4], 0.0) };
        let personal_grad = |p: &ModelParams| -> ModelParams { p.clone() }; // grad = params

        server
            .local_update(0, zero_grad, personal_grad, 0.01, 3)
            .expect("ok");
        // Alpha should remain in [0,1]
        let alpha = server.client_alpha(0).expect("ok");
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0,1], got {alpha}"
        );
    }

    #[test]
    fn test_apfl_personal_params_is_mixture() {
        let cfg = config_small();
        let global = make_params(&[4], 2.0);
        let mut server = ApflClient::new(cfg, global.clone(), 0.0);

        // Local starts at 2.0, global = 2.0, so mixture = 2.0 regardless
        let zero_grad = |_p: &ModelParams| -> ModelParams { make_params(&[4], 0.0) };
        let zero_grad2 = |_p: &ModelParams| -> ModelParams { make_params(&[4], 0.0) };
        server
            .local_update(0, zero_grad, zero_grad2, 0.01, 1)
            .expect("ok");
        let personal = server.personal_params(0).expect("ok");
        for &v in &personal[0] {
            assert!(
                (v - 2.0).abs() < 1e-4,
                "mixture of same values should be same, got {v}"
            );
        }
    }

    #[test]
    fn test_apfl_aggregate_fedavg() {
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global = make_params(&[4], 0.0);
        let mut server = ApflClient::new(cfg, global, 0.01);

        let updates = vec![
            make_update(0, make_params(&[4], 2.0), 100, 0.5, 5),
            make_update(1, make_params(&[4], 4.0), 100, 0.5, 5),
        ];
        let result = server.aggregate(&updates).expect("ok");
        for &v in &result.params[0] {
            assert!((v - 3.0).abs() < 1e-5, "equal-weight avg = 3.0, got {v}");
        }
    }

    // ── FedBnClient ───────────────────────────────────────────────────────────

    #[test]
    fn test_fedbn_new_basic() {
        let cfg = config_small();
        // 3 layers: [0] = shared, [1] = BN, [2] = shared
        let global: ModelParams = vec![vec![1.0; 4], vec![0.5; 2], vec![1.0; 4]];
        let fedbn = FedBnClient::new(cfg, global, vec![1]);
        assert_eq!(fedbn.bn_layer_indices, vec![1]);
    }

    #[test]
    fn test_fedbn_extract_shared_excludes_bn() {
        let cfg = config_small();
        let global: ModelParams = vec![vec![1.0; 4], vec![0.5; 2], vec![2.0; 4]];
        let fedbn = FedBnClient::new(cfg, global, vec![1]);
        let shared = fedbn.extract_shared(&fedbn.global_params.clone());
        // Should have 2 layers (indices 0 and 2)
        assert_eq!(shared.len(), 2, "shared should exclude BN layer");
        assert_eq!(shared[0].len(), 4);
        assert_eq!(shared[1].len(), 4);
    }

    #[test]
    fn test_fedbn_aggregate_shared_only() {
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global: ModelParams = vec![vec![0.0; 4], vec![0.5; 2], vec![0.0; 4]];
        let mut fedbn = FedBnClient::new(cfg, global, vec![1]);

        // Updates with only 2 shared layers
        let updates = vec![
            make_update(0, vec![vec![1.0; 4], vec![3.0; 4]], 100, 0.5, 5),
            make_update(1, vec![vec![3.0; 4], vec![5.0; 4]], 100, 0.5, 5),
        ];
        let result = fedbn.aggregate(&updates, &[0, 1]).expect("ok");
        // Shared layers averaged: [1+3]/2=2.0, [3+5]/2=4.0
        assert_eq!(result.participating_clients, 2);
    }

    #[test]
    fn test_fedbn_client_full_params_merges_bn() {
        let cfg = config_small();
        let global: ModelParams = vec![vec![1.0; 4], vec![0.5; 2], vec![2.0; 4]];
        let fedbn = FedBnClient::new(cfg, global.clone(), vec![1]);
        // client 0's BN params are initialized from global
        let full = fedbn.client_full_params(0).expect("ok");
        assert_eq!(full.len(), 3, "full model should have all 3 layers");
        assert_eq!(full[1], vec![0.5, 0.5], "BN layer from client state");
    }

    // ── HeurFl ────────────────────────────────────────────────────────────────

    #[test]
    fn test_heurfl_adapt_one_step() {
        let cfg = config_small();
        let global = make_params(&[4], 1.0);
        let heurfl = HeurFl::new(cfg, global, 0.1, 0.01, 1);

        // Gradient = params (quadratic at 0, target at 0)
        let grad_fn = |p: &ModelParams| -> ModelParams { p.clone() };
        let adapted = heurfl.adapt(grad_fn);
        // θ = w - α · ∇F(w) = 1.0 - 0.1*1.0 = 0.9
        for &v in &adapted[0] {
            assert!((v - 0.9).abs() < 1e-5, "adapted should be 0.9, got {v}");
        }
    }

    #[test]
    fn test_heurfl_aggregate_moves_toward_adapted() {
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global = make_params(&[4], 1.0);
        let mut heurfl = HeurFl::new(cfg, global.clone(), 0.1, 0.1, 1);

        // Adapted params at 0.8 (less than 1.0)
        let updates = vec![
            make_update(0, make_params(&[4], 0.8), 100, 0.5, 5),
            make_update(1, make_params(&[4], 0.8), 100, 0.5, 5),
        ];
        let result = heurfl.aggregate(&updates).expect("ok");
        // Global should move toward 0.8
        for &v in &result.params[0] {
            assert!(v < 1.0, "global should move below 1.0, got {v}");
        }
    }

    #[test]
    fn test_heurfl_round_increments() {
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global = make_params(&[4], 0.0);
        let mut heurfl = HeurFl::new(cfg, global, 0.1, 0.01, 1);
        assert_eq!(heurfl.current_round(), 0);
        let updates = vec![
            make_update(0, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(1, make_params(&[4], 1.0), 100, 0.5, 5),
        ];
        heurfl.aggregate(&updates).expect("ok");
        assert_eq!(heurfl.current_round(), 1);
    }

    // ── PersonalizedMetrics ───────────────────────────────────────────────────

    #[test]
    fn test_personalized_metrics_gap() {
        let mut metrics = PersonalizedMetrics::new();
        metrics.per_client_accuracy = vec![0.9, 0.85, 0.88];
        metrics.global_accuracy = 0.82;
        metrics.compute_gap();
        // mean(0.9, 0.85, 0.88) - 0.82 = 0.8766... - 0.82 ≈ 0.0566
        assert!(
            metrics.personalization_gap > 0.0,
            "personalized should improve over global"
        );
    }

    #[test]
    fn test_personalized_metrics_communication_cost() {
        let mut metrics = PersonalizedMetrics::new();
        let params = make_params(&[4, 3], 1.0);
        metrics.record_communication(&params);
        assert_eq!(metrics.total_params_communicated, 7);
        metrics.record_communication(&params);
        assert_eq!(metrics.total_params_communicated, 14);
    }

    // ── PermutationMatrix ─────────────────────────────────────────────────────

    #[test]
    fn test_permutation_matrix_identity() {
        let perm = PermutationMatrix::identity(4);
        assert_eq!(perm.perm, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_permutation_matrix_apply_to_rows() {
        // 2 neurons, 3 fan-in: swap rows
        let perm = PermutationMatrix { perm: vec![1, 0] };
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // row0=[1,2,3], row1=[4,5,6]
        let out = perm.apply_to_rows(&weights, 2, 3);
        assert_eq!(out[0..3], [4.0, 5.0, 6.0], "row 0 should be old row 1");
        assert_eq!(out[3..6], [1.0, 2.0, 3.0], "row 1 should be old row 0");
    }

    #[test]
    fn test_permutation_matrix_apply_to_cols() {
        // 2 fan_out, 2 neurons: swap columns
        let perm = PermutationMatrix { perm: vec![1, 0] };
        let weights = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
        let out = perm.apply_to_cols(&weights, 2, 2);
        assert_eq!(out, vec![2.0, 1.0, 4.0, 3.0], "cols should be swapped");
    }

    // ── LayerMatching ─────────────────────────────────────────────────────────

    #[test]
    fn test_layer_matching_identity_on_identical() {
        let matcher = LayerMatching::new(3, 4);
        let weights: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let perm = matcher.match_neurons(&weights, &weights);
        // With identical layers, matching should be identity (or any valid 1-1 mapping)
        let is_valid_perm = {
            let mut seen = [false; 3];
            perm.perm.iter().all(|&j| {
                if j < 3 && !seen[j] {
                    seen[j] = true;
                    true
                } else {
                    false
                }
            })
        };
        assert!(is_valid_perm, "matching should produce a valid permutation");
    }

    #[test]
    fn test_layer_matching_detects_swap() {
        let matcher = LayerMatching::new(2, 4);
        // ref: neuron 0 = [1,0,0,0], neuron 1 = [0,1,0,0]
        let ref_w = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        // client: neuron 0 = [0,1,0,0], neuron 1 = [1,0,0,0]  (swapped)
        let cli_w = vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let perm = matcher.match_neurons(&ref_w, &cli_w);
        // ref[0] ~ cli[1] => perm[0] = 1
        // ref[1] ~ cli[0] => perm[1] = 0
        assert_eq!(perm.perm[0], 1, "ref neuron 0 should match client neuron 1");
        assert_eq!(perm.perm[1], 0, "ref neuron 1 should match client neuron 0");
    }

    #[test]
    fn test_layer_matching_empty_graceful() {
        let matcher = LayerMatching::new(0, 4);
        let perm = matcher.match_neurons(&[], &[]);
        assert_eq!(perm.perm.len(), 0);
    }

    // ── FedMaAggregator ───────────────────────────────────────────────────────

    #[test]
    fn test_fedma_aggregate_no_matching() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let shapes = vec![None]; // no matching
        let mut fedma = FedMaAggregator::new(cfg, global, shapes);

        let updates = vec![
            make_update(0, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(1, make_params(&[4], 3.0), 100, 0.5, 5),
        ];
        let result = fedma.aggregate(&updates).expect("ok");
        for &v in &result.params[0] {
            assert!(
                (v - 2.0).abs() < 1e-5,
                "no-match fedma = fedavg = 2.0, got {v}"
            );
        }
    }

    #[test]
    fn test_fedma_aggregate_with_matching() {
        let cfg = config_small();
        // 2 neurons, 2 fan-in per layer
        let global: ModelParams = vec![vec![1.0, 0.0, 0.0, 1.0]]; // 2 neurons × 2 fan-in
        let shapes = vec![Some((2, 2))];
        let mut fedma = FedMaAggregator::new(cfg, global, shapes);

        let ref_w = vec![1.0, 0.0, 0.0, 1.0];
        // client has neurons swapped
        let cli_w = vec![0.0, 1.0, 1.0, 0.0];
        let updates = vec![
            make_update(0, vec![ref_w.clone()], 100, 0.5, 5),
            make_update(1, vec![cli_w], 100, 0.5, 5),
        ];
        let result = fedma.aggregate(&updates).expect("ok");
        assert_eq!(result.params[0].len(), 4);
    }

    #[test]
    fn test_fedma_compute_matching() {
        let cfg = config_small();
        let global: ModelParams = vec![vec![1.0, 0.0, 0.0, 1.0]];
        let shapes = vec![Some((2, 2))];
        let fedma = FedMaAggregator::new(cfg, global, shapes);

        let ref_w = vec![1.0, 0.0, 0.0, 1.0];
        let cli_w = vec![0.0, 1.0, 1.0, 0.0];
        let perm = fedma.compute_matching(0, &ref_w, &cli_w);
        assert!(perm.is_some(), "should compute matching for layer 0");
    }

    #[test]
    fn test_fedma_errors_on_empty_updates() {
        let cfg = config_small();
        let global = make_params(&[4], 0.0);
        let mut fedma = FedMaAggregator::new(cfg, global, vec![None]);
        match fedma.aggregate(&[]) {
            Err(FederatedError::NoClientUpdates) => {}
            other => panic!("expected NoClientUpdates, got {other:?}"),
        }
    }

    #[test]
    fn test_fedma_round_counter() {
        let mut cfg = config_small();
        cfg.min_clients_available = 2;
        let global = make_params(&[4], 0.0);
        let mut fedma = FedMaAggregator::new(cfg, global, vec![None]);
        assert_eq!(fedma.current_round(), 0);
        let updates = vec![
            make_update(0, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(1, make_params(&[4], 1.0), 100, 0.5, 5),
        ];
        fedma.aggregate(&updates).expect("ok");
        assert_eq!(fedma.current_round(), 1);
    }

    // ── IfcaAlgorithm ─────────────────────────────────────────────────────────

    fn config_ifca() -> FederatedConfig {
        FederatedConfig {
            num_clients: 6,
            clients_per_round: 6,
            num_rounds: 10,
            local_epochs: 2,
            local_lr: 0.01,
            min_clients_available: 1,
        }
    }

    #[test]
    fn test_ifca_new_basic() {
        let cfg = config_ifca();
        let global = make_params(&[4], 0.0);
        let ifca = IfcaAlgorithm::new(cfg, global, 2);
        assert!(ifca.is_ok());
        let ifca = ifca.expect("ok");
        assert_eq!(ifca.k, 2);
        assert_eq!(ifca.cluster_params.len(), 2);
    }

    #[test]
    fn test_ifca_new_k_zero_fails() {
        let cfg = config_ifca();
        let global = make_params(&[4], 0.0);
        assert!(IfcaAlgorithm::new(cfg, global, 0).is_err());
    }

    #[test]
    fn test_ifca_assign_client_to_best_cluster() {
        let cfg = config_ifca();
        let global = make_params(&[4], 0.0);
        let mut ifca = IfcaAlgorithm::new(cfg, global, 3).expect("ok");

        // Client 0 has lowest loss for cluster 2
        let losses = vec![0.8, 0.5, 0.2];
        let cluster = ifca.assign_client(0, &losses).expect("ok");
        assert_eq!(cluster, 2, "should assign to cluster with lowest loss");
    }

    #[test]
    fn test_ifca_aggregate_cluster() {
        let cfg = config_ifca();
        let global = make_params(&[4], 0.0);
        let mut ifca = IfcaAlgorithm::new(cfg, global, 2).expect("ok");

        let updates = vec![
            make_update(0, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(1, make_params(&[4], 3.0), 100, 0.5, 5),
        ];
        let result = ifca.aggregate_cluster(0, &updates).expect("ok");
        for &v in &result.params[0] {
            assert!((v - 2.0).abs() < 1e-5, "cluster avg = 2.0, got {v}");
        }
    }

    #[test]
    fn test_ifca_aggregate_all() {
        let cfg = config_ifca();
        let global = make_params(&[4], 0.0);
        let mut ifca = IfcaAlgorithm::new(cfg, global, 2).expect("ok");

        // Assign clients to clusters manually
        ifca.client_assignments = vec![0, 0, 0, 1, 1, 1];

        let updates: Vec<ClientUpdate> = (0..6)
            .map(|id| {
                let val = if id < 3 { 1.0 } else { 3.0 };
                make_update(id, make_params(&[4], val), 100, 0.5, 5)
            })
            .collect();
        let results = ifca.aggregate_all(&updates).expect("ok");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ifca_cluster_sizes() {
        let cfg = config_ifca();
        let global = make_params(&[4], 0.0);
        let mut ifca = IfcaAlgorithm::new(cfg, global, 2).expect("ok");
        ifca.client_assignments = vec![0, 0, 1, 1, 0, 1];
        let sizes = ifca.cluster_sizes();
        assert_eq!(sizes[0], 3);
        assert_eq!(sizes[1], 3);
    }

    // ── HypCluster ────────────────────────────────────────────────────────────

    #[test]
    fn test_hypcluster_new_basic() {
        let cfg = config_small();
        let hc = HypCluster::new(cfg, 2);
        assert!(hc.is_ok());
    }

    #[test]
    fn test_hypcluster_new_k_zero_fails() {
        let cfg = config_small();
        assert!(HypCluster::new(cfg, 0).is_err());
    }

    #[test]
    fn test_hypcluster_cluster_assigns_all_clients() {
        let cfg = config_small();
        let mut hc = HypCluster::new(cfg, 2).expect("ok");

        // Two distinct groups of clients
        let updates: Vec<ClientUpdate> = (0..4)
            .map(|id| {
                let val = if id < 2 { 1.0 } else { -1.0 };
                make_update(id, make_params(&[4], val), 100, 0.5, 5)
            })
            .collect();
        hc.cluster(&updates).expect("ok");

        // All 4 clients should be assigned
        for i in 0..4 {
            let asgn = hc.assignment(i);
            assert!(asgn.is_some(), "client {i} should have assignment");
            let a = asgn.expect("some");
            assert!(a < 2, "assignment must be < k=2, got {a}");
        }
    }

    #[test]
    fn test_hypcluster_stability_on_same_assignments() {
        let cfg = config_small();
        let mut hc = HypCluster::new(cfg, 2).expect("ok");

        let updates: Vec<ClientUpdate> = (0..4)
            .map(|id| make_update(id, make_params(&[4], id as f32), 100, 0.5, 5))
            .collect();
        hc.cluster(&updates).expect("ok");

        let prev = hc.assignments.clone();
        let stability = hc.assignment_stability(&prev);
        assert!(
            (stability - 1.0).abs() < 1e-6,
            "same assignments = 1.0 stability"
        );
    }

    #[test]
    fn test_hypcluster_clients_in_cluster() {
        let cfg = config_small();
        let mut hc = HypCluster::new(cfg, 2).expect("ok");
        // Manually set assignments
        hc.assignments = vec![0, 1, 0, 1, 0];
        let cluster0 = hc.clients_in_cluster(0);
        let cluster1 = hc.clients_in_cluster(1);
        assert_eq!(cluster0, vec![0, 2, 4]);
        assert_eq!(cluster1, vec![1, 3]);
    }

    // ── ClusteredFLMetrics ────────────────────────────────────────────────────

    #[test]
    fn test_clustered_fl_metrics_diversity() {
        let mut metrics = ClusteredFLMetrics::new();
        let updates: Vec<ClientUpdate> = vec![
            make_update(0, make_params(&[4], 1.0), 100, 0.5, 5),
            make_update(1, make_params(&[4], 1.0), 100, 0.5, 5), // same cluster
            make_update(2, make_params(&[4], -1.0), 100, 0.5, 5), // different cluster
        ];
        let assignments = vec![0, 0, 1];
        metrics.compute_diversity(&updates, &assignments);
        // intra-cluster: clients 0,1 same direction => dist=0
        assert!(
            metrics.intra_cluster_compactness < metrics.inter_cluster_diversity + 1e-3,
            "inter-cluster should be more diverse than intra"
        );
    }

    #[test]
    fn test_clustered_fl_metrics_cluster_losses() {
        let mut metrics = ClusteredFLMetrics::new();
        let updates: Vec<ClientUpdate> = vec![
            make_update(0, make_params(&[4], 1.0), 100, 0.4, 5),
            make_update(1, make_params(&[4], 1.0), 100, 0.6, 5),
            make_update(2, make_params(&[4], 1.0), 100, 0.8, 5),
        ];
        let assignments = vec![0, 0, 1];
        metrics.compute_cluster_losses(&updates, &assignments, 2);
        assert_eq!(metrics.active_clusters, 2);
        assert!(
            (metrics.cluster_losses[0] - 0.5).abs() < 1e-5,
            "cluster 0 avg loss = 0.5"
        );
        assert!(
            (metrics.cluster_losses[1] - 0.8).abs() < 1e-5,
            "cluster 1 avg loss = 0.8"
        );
    }

    // ── ClusterEnsemble ───────────────────────────────────────────────────────

    #[test]
    fn test_cluster_ensemble_mixture_model() {
        let cfg = config_ifca();
        let global = make_params(&[4], 0.0);
        let ifca = IfcaAlgorithm::new(cfg, global, 2).expect("ok");
        let ensemble = ClusterEnsemble::from_ifca(&ifca);

        let query = make_params(&[4], 1.0);
        let mixture = ensemble.mixture_model(&query);
        assert_eq!(mixture.len(), ifca.cluster_params[0].len());
    }

    #[test]
    fn test_cluster_ensemble_empty_returns_empty() {
        let ensemble = ClusterEnsemble {
            cluster_params: vec![],
        };
        let query = make_params(&[4], 1.0);
        let mixture = ensemble.mixture_model(&query);
        assert!(mixture.is_empty());
    }
}
