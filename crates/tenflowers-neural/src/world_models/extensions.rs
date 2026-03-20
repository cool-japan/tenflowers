//! Legacy tests for world_models module (migrated from flat file).

use super::*;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::SeedableRng;

    fn make_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    // --- RsssmConfig ---

    #[test]
    fn test_rssm_config_fields() {
        let c = RsssmConfig::new(256, 32, 8, 512);
        assert_eq!(c.deter_dim, 256);
        assert_eq!(c.stoch_dim, 32);
        assert_eq!(c.n_classes, 8);
        assert_eq!(c.embed_dim, 512);
    }

    // --- RecurrentModel ---

    #[test]
    fn test_recurrent_model_step_shape() {
        let model = RecurrentModel::new(64, 48, 42);
        let h = vec![0.0_f32; 64];
        let x = vec![0.1_f32; 48];
        let h_new = model.step(&h, &x);
        assert_eq!(h_new.len(), 64);
    }

    #[test]
    fn test_recurrent_model_deterministic() {
        let model = RecurrentModel::new(32, 20, 7);
        let h = vec![0.5_f32; 32];
        let x = vec![0.3_f32; 20];
        let h1 = model.step(&h, &x);
        let h2 = model.step(&h, &x);
        assert_eq!(h1, h2);
    }

    // --- RepresentationModel ---

    #[test]
    fn test_repr_model_output_dim() {
        let model = RepresentationModel::new(64, 128, 32, 1);
        let deter = vec![0.0_f32; 64];
        let embed = vec![0.1_f32; 128];
        let logits = model.forward(&deter, &embed);
        assert_eq!(logits.len(), 32);
    }

    // --- TransitionModel ---

    #[test]
    fn test_transition_model_output_dim() {
        let model = TransitionModel::new(64, 32, 2);
        let deter = vec![0.0_f32; 64];
        let logits = model.forward(&deter);
        assert_eq!(logits.len(), 32);
    }

    // --- stoch_straight_through ---

    #[test]
    fn test_straight_through_shape() {
        let mut rng = make_rng(0);
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (oh, st) = stoch_straight_through(&logits, 3, &mut rng);
        assert_eq!(oh.len(), 6);
        assert_eq!(st.len(), 6);
    }

    #[test]
    fn test_straight_through_one_hot_sum() {
        let mut rng = make_rng(1);
        let logits = vec![1.0, 0.5, 0.2];
        let (oh, _) = stoch_straight_through(&logits, 3, &mut rng);
        let sum: f32 = oh.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // --- DreamerV3 ---

    #[test]
    fn test_dreamerv3_imagine_step() {
        let config = RsssmConfig::new(32, 8, 4, 64);
        let model = DreamerV3::new(config, 4, 10);
        let deter = vec![0.0_f32; 32];
        let stoch = vec![0.0_f32; 32]; // 8 * 4
        let action = vec![0.1_f32; 4];
        let (next_d, logits) = model.imagine_step(&deter, &stoch, &action);
        assert_eq!(next_d.len(), 32);
        assert_eq!(logits.len(), 32);
    }

    #[test]
    fn test_dreamerv3_observe_step() {
        let config = RsssmConfig::new(32, 8, 4, 64);
        let model = DreamerV3::new(config, 4, 11);
        let mut rng = make_rng(5);
        let deter = vec![0.0_f32; 32];
        let stoch = vec![0.0_f32; 32];
        let action = vec![0.1_f32; 4];
        let embed = vec![0.5_f32; 64];
        let (nd, ns) = model.observe_step(&deter, &stoch, &action, &embed, &mut rng);
        assert_eq!(nd.len(), 32);
        assert_eq!(ns.len(), 32);
    }

    // --- TwmConfig ---

    #[test]
    fn test_twm_config() {
        let c = TwmConfig::new(16, 128, 4, 2);
        assert_eq!(c.seq_len, 16);
        assert_eq!(c.d_model, 128);
    }

    // --- TwmStateEncoder ---

    #[test]
    fn test_state_encoder_output_dim() {
        let enc = TwmStateEncoder::new(32, 128, 16, 3);
        let obs = vec![0.1_f32; 32];
        let tok = enc.embed_token(&obs, 0);
        assert_eq!(tok.len(), 128);
    }

    // --- TransformerWorldModel ---

    #[test]
    fn test_twm_forward_empty_history() {
        let config = TwmConfig::new(8, 64, 4, 2);
        let model = TransformerWorldModel::new(config, 16, 4, 20);
        let action = vec![0.0_f32; 4];
        let result = model.forward(&[], &action);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_twm_forward_with_history() {
        let config = TwmConfig::new(8, 64, 4, 2);
        let model = TransformerWorldModel::new(config, 16, 4, 21);
        let history: Vec<Vec<f32>> = (0..4).map(|_| vec![0.1_f32; 16]).collect();
        let action = vec![0.5_f32; 4];
        let result = model.forward(&history, &action);
        assert_eq!(result.len(), 16);
    }

    // --- DiscreteTokenizer ---

    #[test]
    fn test_tokenizer_encode_decode_roundtrip_shape() {
        let tok = DiscreteTokenizer::new(16, 8, 42);
        let obs = vec![0.3_f32; 32]; // 4 patches of d_code=8
        let ids = tok.encode(&obs);
        assert_eq!(ids.len(), 4);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded.len(), 32);
    }

    #[test]
    fn test_tokenizer_encode_valid_indices() {
        let tok = DiscreteTokenizer::new(16, 4, 0);
        let obs = vec![0.5_f32; 20];
        let ids = tok.encode(&obs);
        assert!(ids.iter().all(|&i| i < 16));
    }

    #[test]
    fn test_commitment_loss_nonneg() {
        let tok = DiscreteTokenizer::new(8, 4, 1);
        let obs = vec![0.1_f32; 16];
        let ids = tok.encode(&obs);
        let loss = tok.commitment_loss(&obs, &ids, 0.25);
        assert!(loss >= 0.0);
    }

    // --- MuZeroConfig ---

    #[test]
    fn test_muzero_config() {
        let c = MuZeroConfig::new(128, 8, 51);
        assert_eq!(c.hidden_dim, 128);
        assert_eq!(c.action_dim, 8);
        assert_eq!(c.n_values, 51);
    }

    // --- MuZeroRepresentationNet ---

    #[test]
    fn test_repr_net_output_dim() {
        let net = MuZeroRepresentationNet::new(32, 64, 5);
        let obs = vec![0.1_f32; 32];
        let h = net.forward(&obs);
        assert_eq!(h.len(), 64);
    }

    #[test]
    fn test_repr_net_output_bounded() {
        let net = MuZeroRepresentationNet::new(16, 32, 6);
        let obs = vec![1.0_f32; 16];
        let h = net.forward(&obs);
        // tanh output should be in (-1, 1)
        assert!(h.iter().all(|&v| v > -1.0 && v < 1.0));
    }

    // --- MuZeroDynamicsNet ---

    #[test]
    fn test_dynamics_net_output() {
        let net = MuZeroDynamicsNet::new(64, 8, 7);
        let h = vec![0.1_f32; 64];
        let a = vec![0.0_f32; 8];
        let (next_h, _reward) = net.forward(&h, &a);
        assert_eq!(next_h.len(), 64);
    }

    // --- MuZeroPredictionNet ---

    #[test]
    fn test_prediction_net_shapes() {
        let net = MuZeroPredictionNet::new(64, 8, 51, 8);
        let h = vec![0.5_f32; 64];
        let (logits, value) = net.forward(&h);
        assert_eq!(logits.len(), 8);
        assert!(value.is_finite());
    }

    // --- muzero_loss ---

    #[test]
    fn test_muzero_loss_positive() {
        let policy = vec![0.5, 0.3, 0.2];
        let target_p = vec![0.33, 0.33, 0.34];
        let loss = muzero_loss(&policy, 0.8, &target_p, 1.0, 0.5, 0.5);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_muzero_loss_perfect() {
        // Perfect value and reward prediction, slightly imperfect policy
        let policy = vec![1.0, 0.0, 0.0];
        let target_p = vec![1.0, 0.0, 0.0];
        let loss = muzero_loss(&policy, 1.0, &target_p, 1.0, 0.5, 0.5);
        assert!(loss < 5.0); // Should be small
    }

    // --- PcLayer ---

    #[test]
    fn test_pc_layer_forward_shapes() {
        let layer = PcLayer::new(16, 32, 10);
        let x = vec![0.5_f32; 16];
        let h = vec![0.1_f32; 32];
        let (pred, err) = layer.forward_pass(&x, &h);
        assert_eq!(pred.len(), 16);
        assert_eq!(err.len(), 16);
    }

    #[test]
    fn test_pc_layer_backward_shape() {
        let layer = PcLayer::new(16, 32, 11);
        let err = vec![0.1_f32; 16];
        let update = layer.backward_pass(&err, 0.01);
        assert_eq!(update.len(), 32);
    }

    // --- PcNetwork ---

    #[test]
    fn test_pc_network_infer_shape() {
        let net = PcNetwork::new(&[64, 128, 64], 12);
        let obs = vec![0.1_f32; 64];
        let hiddens = net.infer(&obs, 5);
        assert_eq!(hiddens.len(), 2);
        assert_eq!(hiddens[0].len(), 128);
        assert_eq!(hiddens[1].len(), 64);
    }

    #[test]
    fn test_pc_network_infer_empty() {
        let net = PcNetwork::new(&[32], 13);
        let obs = vec![0.0_f32; 32];
        let hiddens = net.infer(&obs, 3);
        assert_eq!(hiddens.len(), 0);
    }

    // --- consistency_loss ---

    #[test]
    fn test_consistency_loss_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let loss = consistency_loss(&v, &v);
        assert!(
            loss.abs() < 1e-5,
            "Expected ~0 for identical vectors, got {}",
            loss
        );
    }

    #[test]
    fn test_consistency_loss_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let loss = consistency_loss(&a, &b);
        assert!(
            (loss - 4.0).abs() < 1e-5,
            "Expected 4 for opposite vectors, got {}",
            loss
        );
    }

    // --- augment_state ---

    #[test]
    fn test_augment_state_shape() {
        let mut rng = make_rng(3);
        let h = vec![0.5_f32; 32];
        let aug = augment_state(&h, 0.01, &mut rng);
        assert_eq!(aug.len(), 32);
    }

    #[test]
    fn test_augment_state_small_noise() {
        let mut rng = make_rng(4);
        let h = vec![0.5_f32; 32];
        let aug = augment_state(&h, 0.001, &mut rng);
        let max_diff = h
            .iter()
            .zip(aug.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff < 0.1, "Noise too large: {}", max_diff);
    }

    // --- EfficientZeroModel ---

    #[test]
    fn test_efficientzero_self_supervised_step() {
        let config = MuZeroConfig::new(64, 4, 51);
        let model = EfficientZeroModel::new(16, config, 32, 30);
        let mut rng = make_rng(99);
        let obs = vec![0.1_f32; 16];
        let next_obs = vec![0.2_f32; 16];
        let loss = model.self_supervised_step(&obs, 0, &next_obs, &mut rng);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    // --- TdmConfig ---

    #[test]
    fn test_tdm_config() {
        let c = TdmConfig::new(20, 16, 4, 8);
        assert_eq!(c.horizon, 20);
        assert_eq!(c.state_dim, 16);
    }

    // --- TdmNetwork ---

    #[test]
    fn test_tdm_q_value_finite() {
        let config = TdmConfig::new(20, 16, 4, 8);
        let net = TdmNetwork::new(config, 50);
        let state = vec![0.1_f32; 16];
        let action = vec![0.5_f32; 4];
        let goal = vec![1.0_f32; 8];
        let q = net.q_value(&state, &action, &goal, 5);
        assert!(q.is_finite());
    }

    #[test]
    fn test_tdm_policy_shape() {
        let config = TdmConfig::new(20, 16, 4, 8);
        let net = TdmNetwork::new(config, 51);
        let state = vec![0.0_f32; 16];
        let goal = vec![1.0_f32; 8];
        let action = net.policy(&state, &goal, 10);
        assert_eq!(action.len(), 4);
    }

    #[test]
    fn test_tdm_policy_bounded() {
        let config = TdmConfig::new(10, 8, 2, 4);
        let net = TdmNetwork::new(config, 52);
        let state = vec![0.3_f32; 8];
        let goal = vec![0.7_f32; 4];
        let action = net.policy(&state, &goal, 3);
        // tanh output in (-1, 1)
        assert!(action.iter().all(|&a| a > -1.0 && a < 1.0));
    }

    // --- DecoderConfig & ObsDecoder ---

    #[test]
    fn test_obs_decoder_config() {
        let c = DecoderConfig::new(32, vec![128, 64], 784);
        assert_eq!(c.latent_dim, 32);
        assert_eq!(c.obs_dim, 784);
    }

    #[test]
    fn test_obs_decoder_output_dim() {
        let config = DecoderConfig::new(16, vec![64], 32);
        let dec = ObsDecoder::new(&config, 60);
        let z = vec![0.5_f32; 16];
        let obs = dec.decode(&z);
        assert_eq!(obs.len(), 32);
    }

    #[test]
    fn test_reconstruction_loss_zero() {
        let config = DecoderConfig::new(8, vec![16], 8);
        let dec = ObsDecoder::new(&config, 61);
        let obs = vec![1.0_f32; 8];
        let loss = dec.reconstruction_loss(&obs, &obs);
        assert!(loss.abs() < 1e-8);
    }

    #[test]
    fn test_reconstruction_loss_positive() {
        let config = DecoderConfig::new(8, vec![16], 8);
        let dec = ObsDecoder::new(&config, 62);
        let pred = vec![0.0_f32; 8];
        let true_obs = vec![1.0_f32; 8];
        let loss = dec.reconstruction_loss(&pred, &true_obs);
        assert!(loss > 0.0);
    }

    // --- WmRewardPredictor ---

    #[test]
    fn test_wm_reward_predictor_finite() {
        let rp = WmRewardPredictor::new(16, 4, 32, 70);
        let state = vec![0.1_f32; 16];
        let action = vec![0.5_f32; 4];
        let r = rp.predict_reward(&state, &action);
        assert!(r.is_finite());
    }

    #[test]
    fn test_wm_done_probability_bounded() {
        let rp = WmRewardPredictor::new(8, 2, 16, 71);
        let state = vec![0.2_f32; 8];
        let action = vec![0.1_f32; 2];
        let p = rp.predict_done(&state, &action);
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn test_reward_loss_mse() {
        let rp = WmRewardPredictor::new(4, 2, 8, 72);
        let loss = rp.reward_loss(1.5, 1.0);
        assert!((loss - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_done_loss_bce() {
        let rp = WmRewardPredictor::new(4, 2, 8, 73);
        let loss_true = rp.done_loss(10.0, true); // logit=10 → prob≈1
        let loss_false = rp.done_loss(-10.0, false); // logit=-10 → prob≈0
        assert!(loss_true < 0.01);
        assert!(loss_false < 0.01);
    }

    // --- Cem ---

    #[test]
    fn test_cem_config() {
        let cem = Cem::new(200, 0.1, 10, 16);
        assert_eq!(cem.pop_size, 200);
        assert_eq!(cem.n_iters, 10);
    }

    // --- LatentPlanner ---

    #[test]
    fn test_latent_planner_plan_shape() {
        let cem = Cem::new(50, 0.2, 3, 8);
        let planner = LatentPlanner::new(cem, 4);
        let mut rng = make_rng(80);
        let init = vec![0.0_f32; 8];
        let goal = vec![1.0_f32; 8];
        let actions = planner.plan(&init, &goal, 8, &mut rng);
        assert_eq!(actions.len(), 8);
        assert_eq!(actions[0].len(), 4);
    }

    #[test]
    fn test_latent_planner_plan_finite() {
        let cem = Cem::new(30, 0.3, 2, 4);
        let planner = LatentPlanner::new(cem, 3);
        let mut rng = make_rng(81);
        let init = vec![0.5_f32; 4];
        let goal = vec![0.0_f32; 4];
        let actions = planner.plan(&init, &goal, 4, &mut rng);
        assert!(actions.iter().all(|a| a.iter().all(|&v| v.is_finite())));
    }

    #[test]
    fn test_mppi_plan_shape() {
        let cem = Cem::new(64, 0.2, 5, 10);
        let planner = LatentPlanner::new(cem, 4);
        let mut rng = make_rng(90);
        let init = vec![0.0_f32; 4];
        let goal = vec![1.0_f32; 4];
        let actions = planner.mppi_plan(&init, &goal, 10, 64, 1.0, &mut rng);
        assert_eq!(actions.len(), 10);
        assert_eq!(actions[0].len(), 4);
    }

    #[test]
    fn test_mppi_plan_finite() {
        let cem = Cem::new(32, 0.25, 3, 6);
        let planner = LatentPlanner::new(cem, 2);
        let mut rng = make_rng(91);
        let init = vec![0.1_f32; 2];
        let goal = vec![0.9_f32; 2];
        let actions = planner.mppi_plan(&init, &goal, 6, 32, 0.1, &mut rng);
        assert!(actions.iter().all(|a| a.iter().all(|&v| v.is_finite())));
    }

    // --- IrisWorldModel ---

    #[test]
    fn test_iris_encode_decode() {
        let iris = IrisWorldModel::new(16, 8, 24, 4, 8, 100);
        let obs = vec![0.2_f32; 24];
        let ids = iris.tokenizer.encode(&obs);
        assert_eq!(ids.len(), 3); // 24 / 8 = 3 patches
        let decoded = iris.tokenizer.decode(&ids);
        assert_eq!(decoded.len(), 24);
    }

    // --- MuZeroModel ---

    #[test]
    fn test_muzero_model_full_forward() {
        let config = MuZeroConfig::new(64, 4, 51);
        let model = MuZeroModel::new(16, config, 42);
        let obs = vec![0.5_f32; 16];
        let h = model.repr_net.forward(&obs);
        assert_eq!(h.len(), 64);
        let (policy, value) = model.prediction_net.forward(&h);
        assert_eq!(policy.len(), 4);
        assert!(value.is_finite());
    }

    // --- softmax helper ---

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // --- TwmTransitionHead ---

    #[test]
    fn test_twm_head_output_dim() {
        let head = TwmTransitionHead::new(128, 32, 15);
        let context = vec![0.3_f32; 128];
        let out = head.project(&context);
        assert_eq!(out.len(), 32);
    }
}
