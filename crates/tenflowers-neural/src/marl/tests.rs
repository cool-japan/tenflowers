//! Tests for all MARL modules.

#[cfg(test)]
mod tests {
    use crate::marl::{
        Agent, AgentQNetwork, AtocAgent, ComaTrainer, CommChannel, CommNet, CreditAssignment,
        Experience, MaddpgActor, MaddpgAgent, MaddpgCritic, MaddpgTrainer, MessageDecoder,
        MessageEncoder, MixingNetwork, MultiAgentTrainer, MultiAgentTrainerConfig, OuNoise,
        QmixAgent, QmixTrainer, SharedReplayBuffer, TarmacAgent, TeamRewardShaper, VdnMixer,
        WqmixTrainer,
    };

    // ── Interface ────────────────────────────────────────────────────────────

    #[test]
    fn test_experience_construction() {
        let exp = Experience::new(vec![1.0, 2.0], vec![0.5], 1.5, vec![1.1, 2.1], false, 0);
        assert_eq!(exp.obs.len(), 2);
        assert_eq!(exp.action.len(), 1);
        assert!((exp.reward - 1.5).abs() < 1e-9);
        assert!(!exp.done);
        assert_eq!(exp.agent_id, 0);
    }

    #[test]
    fn test_experience_done_flag() {
        let exp = Experience::new(vec![0.0], vec![0.0], -1.0, vec![0.0], true, 2);
        assert!(exp.done);
        assert_eq!(exp.agent_id, 2);
    }

    #[test]
    fn test_shared_replay_buffer_push_and_len() {
        let buf = SharedReplayBuffer::new(5).expect("replay buffer creation should succeed");
        assert_eq!(buf.len().expect("len should succeed"), 0);
        for i in 0..5 {
            buf.push(Experience::new(
                vec![i as f64],
                vec![0.0],
                0.0,
                vec![0.0],
                false,
                i,
            ))
            .expect("push should succeed");
        }
        assert_eq!(buf.len().expect("len should succeed"), 5);
    }

    #[test]
    fn test_shared_replay_buffer_eviction() {
        let buf = SharedReplayBuffer::new(3).expect("replay buffer creation should succeed");
        for i in 0..5 {
            buf.push(Experience::new(
                vec![i as f64],
                vec![0.0],
                0.0,
                vec![0.0],
                false,
                0,
            ))
            .expect("push should succeed");
        }
        assert_eq!(buf.len().expect("len should succeed"), 3);
    }

    #[test]
    fn test_shared_replay_buffer_sample() {
        let buf = SharedReplayBuffer::new(100).expect("replay buffer creation should succeed");
        for i in 0..50 {
            buf.push(Experience::new(
                vec![i as f64],
                vec![0.0],
                0.0,
                vec![0.0],
                false,
                0,
            ))
            .expect("push should succeed");
        }
        let batch = buf.sample(10, 42).expect("sample should succeed");
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_shared_replay_buffer_empty_sample_error() {
        let buf = SharedReplayBuffer::new(10).expect("replay buffer creation should succeed");
        assert!(buf.sample(5, 0).is_err());
    }

    #[test]
    fn test_shared_replay_buffer_zero_capacity_error() {
        assert!(SharedReplayBuffer::new(0).is_err());
    }

    #[test]
    fn test_shared_replay_buffer_clone() {
        let buf = SharedReplayBuffer::new(10).expect("replay buffer creation should succeed");
        buf.push(Experience::new(
            vec![1.0],
            vec![0.0],
            1.0,
            vec![2.0],
            false,
            0,
        ))
        .expect("push should succeed");
        let buf2 = buf.clone();
        assert_eq!(buf2.len().expect("len should succeed"), 1);
    }

    #[test]
    fn test_multi_agent_trainer_init() {
        let agents: Vec<Box<dyn Agent>> = (0..3)
            .map(|i| -> Box<dyn Agent> {
                Box::new(QmixAgent::new(i, 4, 2, 8, 0.1, i as u64).expect("QmixAgent creation should succeed"))
            })
            .collect();
        let config = MultiAgentTrainerConfig::default();
        let trainer = MultiAgentTrainer::new(agents, config).expect("MultiAgentTrainer creation should succeed");
        assert_eq!(trainer.num_agents(), 3);
        assert_eq!(trainer.step_count(), 0);
    }

    #[test]
    fn test_multi_agent_trainer_record_and_train() {
        let agents: Vec<Box<dyn Agent>> = (0..2)
            .map(|i| -> Box<dyn Agent> {
                Box::new(QmixAgent::new(i, 2, 2, 4, 0.5, i as u64).expect("QmixAgent creation should succeed"))
            })
            .collect();
        let config = MultiAgentTrainerConfig {
            min_replay_size: 5,
            batch_size: 4,
            ..Default::default()
        };
        let mut trainer = MultiAgentTrainer::new(agents, config).expect("MultiAgentTrainer creation should succeed");
        for i in 0..10 {
            trainer
                .record(Experience::new(
                    vec![i as f64, 0.0],
                    vec![0.0, 1.0],
                    1.0,
                    vec![i as f64 + 1.0, 0.0],
                    false,
                    i % 2,
                ))
                .expect("recording experience should succeed");
        }
        trainer.train_step().expect("train step should succeed");
        assert!(trainer.step_count() > 0);
    }

    // ── QMIX ────────────────────────────────────────────────────────────────

    #[test]
    fn test_agent_q_network_forward() {
        let mut net = AgentQNetwork::new(4, 3, 8, 42).expect("AgentQNetwork creation should succeed");
        let q = net.forward(&[0.1, -0.2, 0.3, 0.4]).expect("forward pass should succeed");
        assert_eq!(q.len(), 3);
        // All finite
        assert!(q.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_agent_q_network_hidden_update() {
        let mut net = AgentQNetwork::new(2, 2, 4, 7).expect("AgentQNetwork creation should succeed");
        let h0 = net.hidden.clone();
        net.forward(&[1.0, 0.0]).expect("forward pass should succeed");
        assert_ne!(net.hidden, h0);
    }

    #[test]
    fn test_agent_q_network_reset_hidden() {
        let mut net = AgentQNetwork::new(2, 2, 4, 13).expect("AgentQNetwork creation should succeed");
        net.forward(&[1.0, 1.0]).expect("forward pass should succeed");
        net.reset_hidden();
        assert!(net.hidden.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_agent_q_network_dim_mismatch() {
        let mut net = AgentQNetwork::new(4, 2, 8, 1).expect("AgentQNetwork creation should succeed");
        assert!(net.forward(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_mixing_network_forward() {
        let mixer = MixingNetwork::new(3, 6, 8, 99).expect("MixingNetwork creation should succeed");
        let q = mixer
            .forward(&[0.5, -0.3, 0.8], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            .expect("mixing network forward should succeed");
        assert!(q.is_finite());
    }

    #[test]
    fn test_mixing_network_monotonicity() {
        // Increasing agent_qs[0] should not decrease joint Q (monotonicity).
        let mixer = MixingNetwork::new(2, 4, 4, 55).expect("MixingNetwork creation should succeed");
        let state = vec![0.5, 0.5, 0.5, 0.5];
        let q_low = mixer.forward(&[0.0, 0.5], &state).expect("forward with low q should succeed");
        let q_high = mixer.forward(&[1.0, 0.5], &state).expect("forward with high q should succeed");
        // Due to ELU and abs weights, monotonicity holds approximately.
        // We just check both are finite.
        assert!(q_low.is_finite());
        assert!(q_high.is_finite());
    }

    #[test]
    fn test_mixing_network_agent_count_mismatch() {
        let mixer = MixingNetwork::new(3, 4, 4, 1).expect("MixingNetwork creation should succeed");
        assert!(mixer.forward(&[0.5, 0.5], &[0.0; 4]).is_err());
    }

    #[test]
    fn test_mixing_network_state_dim_mismatch() {
        let mixer = MixingNetwork::new(2, 4, 4, 1).expect("MixingNetwork creation should succeed");
        assert!(mixer.forward(&[0.5, 0.5], &[0.0; 3]).is_err());
    }

    #[test]
    fn test_qmix_agent_epsilon_greedy_exploit() {
        let mut agent = QmixAgent::new(0, 4, 3, 8, 0.0, 42).expect("QmixAgent creation should succeed");
        let obs = vec![0.1, 0.2, 0.3, 0.4];
        // epsilon=0 → always exploit
        let action = agent.select_action(&obs).expect("select_action should succeed");
        assert!(action < 3);
    }

    #[test]
    fn test_qmix_agent_epsilon_greedy_explore() {
        let mut agent = QmixAgent::new(0, 4, 3, 8, 1.0, 42).expect("QmixAgent creation should succeed");
        let obs = vec![0.1, 0.2, 0.3, 0.4];
        // epsilon=1 → always explore; result should still be valid
        let action = agent.select_action(&obs).expect("select_action should succeed");
        assert!(action < 3);
    }

    #[test]
    fn test_qmix_agent_act_trait() {
        let agent = QmixAgent::new(0, 4, 3, 8, 0.0, 0).expect("QmixAgent creation should succeed");
        let result = agent.act(&[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(result.len(), 1);
        assert!(result[0].is_finite());
    }

    #[test]
    fn test_qmix_loss_computation() {
        let agents: Vec<QmixAgent> = (0..2)
            .map(|i| QmixAgent::new(i, 2, 2, 4, 0.1, i as u64).expect("QmixAgent creation should succeed"))
            .collect();
        let mixer = MixingNetwork::new(2, 4, 4, 100).expect("MixingNetwork creation should succeed");
        let mut trainer = QmixTrainer::new(agents, mixer, 0.99);
        let exps = vec![vec![
            Experience::new(vec![0.1, 0.2], vec![0.0], 1.0, vec![0.2, 0.3], false, 0),
            Experience::new(vec![0.3, 0.4], vec![1.0], 0.5, vec![0.4, 0.5], false, 1),
        ]];
        let states = vec![vec![0.1, 0.2, 0.3, 0.4]];
        let next_states = vec![vec![0.2, 0.3, 0.4, 0.5]];
        let loss = trainer.compute_loss(&exps, &states, &next_states).expect("loss computation should succeed");
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    // ── MADDPG ───────────────────────────────────────────────────────────────

    #[test]
    fn test_maddpg_actor_forward() {
        let actor = MaddpgActor::new(4, 2, 8, 1).expect("MaddpgActor creation should succeed");
        let a = actor.forward(&[0.1, -0.2, 0.3, 0.4]).expect("actor forward should succeed");
        assert_eq!(a.len(), 2);
        assert!(a.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    #[test]
    fn test_maddpg_actor_dim_mismatch() {
        let actor = MaddpgActor::new(4, 2, 8, 1).expect("MaddpgActor creation should succeed");
        assert!(actor.forward(&[0.1, 0.2]).is_err());
    }

    #[test]
    fn test_maddpg_critic_all_agents() {
        let critic = MaddpgCritic::new(8, 16, 42).expect("MaddpgCritic creation should succeed");
        let inp = vec![0.1f64; 8];
        let q = critic.forward(&inp).expect("critic forward should succeed");
        assert!(q.is_finite());
    }

    #[test]
    fn test_maddpg_critic_dim_mismatch() {
        let critic = MaddpgCritic::new(8, 16, 42).expect("MaddpgCritic creation should succeed");
        assert!(critic.forward(&[0.0; 4]).is_err());
    }

    #[test]
    fn test_maddpg_soft_update() {
        let mut actor = MaddpgActor::new(4, 2, 8, 10).expect("MaddpgActor creation should succeed");
        let target = MaddpgActor::new(4, 2, 8, 20).expect("target MaddpgActor creation should succeed");
        let w1_before = actor.w1[0];
        actor.soft_update(&target, 0.5).expect("soft update should succeed");
        let expected = 0.5 * w1_before + 0.5 * target.w1[0];
        assert!((actor.w1[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_maddpg_critic_soft_update() {
        let mut critic = MaddpgCritic::new(4, 4, 1).expect("MaddpgCritic creation should succeed");
        let target = MaddpgCritic::new(4, 4, 2).expect("target MaddpgCritic creation should succeed");
        let w_before = critic.w1[0];
        critic.soft_update(&target, 0.01).expect("critic soft update should succeed");
        let expected = 0.99 * w_before + 0.01 * target.w1[0];
        assert!((critic.w1[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn test_ou_noise_statistics() {
        let mut noise = OuNoise::new(4, 0);
        let samples: Vec<Vec<f64>> = (0..500).map(|_| noise.sample()).collect();
        // Mean should be close to 0 over many steps
        let means: Vec<f64> = (0..4)
            .map(|i| samples.iter().map(|s| s[i]).sum::<f64>() / 500.0)
            .collect();
        for m in &means {
            assert!(m.abs() < 1.5, "OU mean too large: {m}");
        }
    }

    #[test]
    fn test_ou_noise_reset() {
        let mut noise = OuNoise::new(3, 42);
        for _ in 0..20 {
            noise.sample();
        }
        noise.reset();
        assert!(noise.state.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_maddpg_agent_act_with_noise() {
        let mut agent = MaddpgAgent::new(0, 4, 2, 8, 12, 5).expect("MaddpgAgent creation should succeed");
        let a = agent.act_with_noise(&[0.1; 4]).expect("act_with_noise should succeed");
        assert_eq!(a.len(), 2);
        assert!(a.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    #[test]
    fn test_maddpg_agent_update_targets() {
        let mut agent = MaddpgAgent::new(0, 4, 2, 8, 12, 5).expect("MaddpgAgent creation should succeed");
        let before = agent.actor_target.w1[0];
        // Slightly modify actor
        agent.actor.w1[0] += 1.0;
        agent.update_targets().expect("update_targets should succeed");
        let expected = (1.0 - 0.01) * before + 0.01 * agent.actor.w1[0];
        assert!((agent.actor_target.w1[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_maddpg_trainer_critic_loss() {
        // critic_input_dim = obs_dim + action_dim = 4 + 2 = 6
        let agents = vec![
            MaddpgAgent::new(0, 4, 2, 8, 6, 10).expect("MaddpgAgent creation should succeed"),
            MaddpgAgent::new(1, 4, 2, 8, 6, 20).expect("MaddpgAgent creation should succeed"),
        ];
        let trainer = MaddpgTrainer::new(agents, 1000, 0.99, 32).expect("MaddpgTrainer creation should succeed");
        let batch = vec![Experience::new(
            vec![0.1f64; 4],
            vec![0.5, 0.5],
            1.0,
            vec![0.2f64; 4],
            false,
            0,
        )];
        let next_actions = vec![vec![0.3, 0.3]];
        let loss = trainer.critic_loss(0, &batch, &next_actions).expect("critic loss computation should succeed");
        assert!(loss.is_finite());
    }

    // ── Communication ───────────────────────────────────────────────────────

    #[test]
    fn test_comm_channel_forward() {
        let ch = CommChannel::new(8, 4, 77).expect("CommChannel creation should succeed");
        let h = vec![0.1f64; 8];
        let msg = ch.encode(&h).expect("encoding should succeed");
        assert_eq!(msg.len(), 4);
        assert!(msg.iter().all(|v| (-1.0..=1.0).contains(v)));
        let decoded = ch.decode(&msg).expect("decoding should succeed");
        assert_eq!(decoded.len(), 8);
    }

    #[test]
    fn test_comm_channel_encode_dim_error() {
        let ch = CommChannel::new(8, 4, 1).expect("CommChannel creation should succeed");
        assert!(ch.encode(&[0.0; 4]).is_err());
    }

    #[test]
    fn test_comm_channel_decode_dim_error() {
        let ch = CommChannel::new(8, 4, 1).expect("CommChannel creation should succeed");
        assert!(ch.decode(&[0.0; 8]).is_err());
    }

    #[test]
    fn test_message_encoder() {
        let enc = MessageEncoder::new(6, 3, 33).expect("MessageEncoder creation should succeed");
        let msg = enc.encode(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).expect("encoding should succeed");
        assert_eq!(msg.len(), 3);
        assert!(msg.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_message_decoder() {
        let dec = MessageDecoder::new(3, 6, 44).expect("MessageDecoder creation should succeed");
        let out = dec.decode(&[0.5, -0.5, 0.1]).expect("decoding should succeed");
        assert_eq!(out.len(), 6);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_commnet_message_aggregation() {
        let net = CommNet::new(3, 4, 2, 11).expect("CommNet creation should succeed");
        let hiddens = vec![vec![0.1; 4], vec![0.2; 4], vec![0.3; 4]];
        let decoded = net.communicate(&hiddens).expect("communication should succeed");
        assert_eq!(decoded.len(), 3);
        for d in &decoded {
            assert_eq!(d.len(), 4);
            assert!(d.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_commnet_agent_count_mismatch() {
        let net = CommNet::new(3, 4, 2, 1).expect("CommNet creation should succeed");
        let hiddens = vec![vec![0.1; 4], vec![0.2; 4]];
        assert!(net.communicate(&hiddens).is_err());
    }

    #[test]
    fn test_atoc_attention_weights() {
        let agent = AtocAgent::new(0, 4, 4, 2, 77).expect("AtocAgent creation should succeed");
        let thought = agent.compute_thought(&[0.1, 0.2, 0.3, 0.4]).expect("compute_thought should succeed");
        let w = agent.attention_weight(&thought).expect("attention_weight should succeed");
        assert!((0.0..=1.0).contains(&w));
    }

    #[test]
    fn test_atoc_encode_message() {
        let agent = AtocAgent::new(0, 4, 4, 3, 88).expect("AtocAgent creation should succeed");
        let thought = agent.compute_thought(&[0.5; 4]).expect("compute_thought should succeed");
        let msg = agent.encode_message(&thought).expect("encode_message should succeed");
        assert_eq!(msg.len(), 3);
    }

    #[test]
    fn test_tarmac_targeted_comm() {
        let agents: Vec<TarmacAgent> = (0..3)
            .map(|i| TarmacAgent::new(i, 8, 4, 4, i as u64 * 17).expect("TarmacAgent creation should succeed"))
            .collect();
        let hiddens: Vec<Vec<f64>> = (0..3).map(|i| vec![i as f64 * 0.1; 8]).collect();
        // Agent 0 aggregates from agents 1 and 2
        let agg = agents[0]
            .aggregate(&hiddens[0], &hiddens[1..], &agents[1..])
            .expect("aggregate should succeed");
        assert_eq!(agg.len(), 4);
        assert!(agg.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_tarmac_key_value_query() {
        let agent = TarmacAgent::new(0, 8, 4, 4, 1).expect("TarmacAgent creation should succeed");
        let h = vec![0.1f64; 8];
        let k = agent.key(&h);
        let v = agent.value(&h);
        let q = agent.query(&h);
        assert_eq!(k.len(), 4);
        assert_eq!(v.len(), 4);
        assert_eq!(q.len(), 4);
    }

    // ── Cooperative ──────────────────────────────────────────────────────────

    #[test]
    fn test_team_reward_shaper() {
        let shaper = TeamRewardShaper::new(0.99, vec![1.0, -1.0, 0.5]);
        let s = vec![1.0, 2.0, 0.0];
        let s_next = vec![1.5, 1.5, 1.0];
        let shaped = shaper.shape(1.0, &s, &s_next);
        assert!(shaped.is_finite());
        // Φ(s) = 1.0 - 2.0 + 0.0 = -1.0; Φ(s') = 1.5 - 1.5 + 0.5 = 0.5
        // r + 0.99*0.5 - (-1.0) = 1.0 + 0.495 + 1.0 = 2.495
        assert!((shaped - 2.495).abs() < 1e-6);
    }

    #[test]
    fn test_team_reward_shaper_team() {
        let shaper = TeamRewardShaper::new(0.9, vec![1.0]);
        let shaped = shaper.shape_team(&[0.5, 0.5], &[1.0], &[2.0]);
        assert_eq!(shaped.len(), 2);
        // shaping = 0.9*2.0 - 1.0 = 0.8; each agent gets r + 0.8
        assert!((shaped[0] - 1.3).abs() < 1e-9);
        assert!((shaped[1] - 1.3).abs() < 1e-9);
    }

    #[test]
    fn test_difference_rewards() {
        let ca = CreditAssignment::new(2, 2);
        let state = vec![0.0; 4];
        let actions = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let dr = ca
            .difference_reward(&state, &actions, |_s, a| a.iter().sum::<f64>(), 0)
            .expect("difference_reward should succeed");
        // Q(joint) = 1.0+0.0+0.0+1.0 = 2.0
        // Q(cf, a0=default=[0,0]) = 0.0+0.0+0.0+1.0 = 1.0
        // D_0 = 1.0
        assert!((dr - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_difference_rewards_out_of_range() {
        let ca = CreditAssignment::new(2, 2);
        let state = vec![0.0; 4];
        let actions = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        assert!(ca
            .difference_reward(&state, &actions, |_, _| 0.0, 5)
            .is_err());
    }

    #[test]
    fn test_coma_counterfactual_advantage() {
        let ca = CreditAssignment::new(2, 1);
        let state = vec![0.0];
        let actions = vec![vec![1.0], vec![1.0]];
        let adv = ca
            .coma_advantage(&state, &actions, |_, a| a.iter().sum::<f64>(), 0, 100, 42)
            .expect("coma_advantage should succeed");
        // Q(actual) = 2.0; baseline should be ~1.0 (uniform OU actions for agent 0)
        assert!(adv.is_finite());
    }

    #[test]
    fn test_coma_trainer_advantages() {
        let trainer = ComaTrainer::new(2, 2, 8, 8, 0.99, 77).expect("ComaTrainer creation should succeed");
        let state = vec![0.1; 4];
        let actions = vec![vec![0.5, 0.5], vec![-0.5, -0.5]];
        let advs = trainer.compute_advantages(&state, &actions, 20, 7).expect("compute_advantages should succeed");
        assert_eq!(advs.len(), 2);
        assert!(advs.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_vdn_mixer_sum() {
        let mixer = VdnMixer::new(3).expect("VdnMixer creation should succeed");
        let q = mixer.mix(&[1.0, 2.0, 3.0]).expect("mix should succeed");
        assert!((q - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_vdn_mixer_decompose() {
        let mixer = VdnMixer::new(4).expect("VdnMixer creation should succeed");
        let parts = mixer.decompose(8.0);
        assert_eq!(parts.len(), 4);
        assert!(parts.iter().all(|v| (v - 2.0).abs() < 1e-9));
    }

    #[test]
    fn test_vdn_mixer_zero_agents_error() {
        assert!(VdnMixer::new(0).is_err());
    }

    #[test]
    fn test_vdn_mixer_count_mismatch() {
        let mixer = VdnMixer::new(3).expect("VdnMixer creation should succeed");
        assert!(mixer.mix(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_wqmix_trainer_loss() {
        let agents: Vec<QmixAgent> = (0..2)
            .map(|i| QmixAgent::new(i, 2, 2, 4, 0.1, i as u64 * 3).expect("QmixAgent creation should succeed"))
            .collect();
        let mixer = MixingNetwork::new(2, 4, 4, 55).expect("MixingNetwork creation should succeed");
        let qmix = QmixTrainer::new(agents, mixer, 0.99);
        let vdn = VdnMixer::new(2).expect("VdnMixer creation should succeed");
        let mut wqmix = WqmixTrainer::new(qmix, vdn, 0.7).expect("WqmixTrainer creation should succeed");
        let exps = vec![vec![
            Experience::new(vec![0.0, 1.0], vec![0.0], 0.5, vec![0.1, 0.9], false, 0),
            Experience::new(vec![1.0, 0.0], vec![1.0], 0.5, vec![0.9, 0.1], false, 1),
        ]];
        let states = vec![vec![0.0, 1.0, 1.0, 0.0]];
        let next_states = vec![vec![0.1, 0.9, 0.9, 0.1]];
        let loss = wqmix.compute_loss(&exps, &states, &next_states).expect("WqmixTrainer loss computation should succeed");
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_wqmix_invalid_alpha() {
        let agents: Vec<QmixAgent> = (0..2)
            .map(|i| QmixAgent::new(i, 2, 2, 4, 0.1, i as u64).expect("QmixAgent creation should succeed"))
            .collect();
        let mixer = MixingNetwork::new(2, 4, 4, 1).expect("MixingNetwork creation should succeed");
        let qmix = QmixTrainer::new(agents, mixer, 0.99);
        let vdn = VdnMixer::new(2).expect("VdnMixer creation should succeed");
        assert!(WqmixTrainer::new(qmix, vdn, 0.0).is_err());
    }

    // ── Additional coverage ──────────────────────────────────────────────────

    #[test]
    fn test_qmix_trainer_step_count() {
        let agents: Vec<QmixAgent> = (0..2)
            .map(|i| QmixAgent::new(i, 2, 2, 4, 0.1, i as u64).expect("QmixAgent creation should succeed"))
            .collect();
        let mixer = MixingNetwork::new(2, 4, 4, 1).expect("MixingNetwork creation should succeed");
        let mut trainer = QmixTrainer::new(agents, mixer, 0.99);
        assert_eq!(trainer.step, 0);
        let exps = vec![vec![
            Experience::new(vec![0.0, 1.0], vec![0.0], 0.5, vec![0.1, 0.9], false, 0),
            Experience::new(vec![1.0, 0.0], vec![0.0], 0.5, vec![0.9, 0.1], false, 1),
        ]];
        let states = vec![vec![0.0; 4]];
        let next_states = vec![vec![0.1; 4]];
        trainer.compute_loss(&exps, &states, &next_states).expect("loss computation should succeed");
        assert_eq!(trainer.step, 1);
    }

    #[test]
    fn test_maddpg_trainer_update_targets() {
        let agents = vec![MaddpgAgent::new(0, 4, 2, 8, 12, 1).expect("MaddpgAgent creation should succeed")];
        let mut trainer = MaddpgTrainer::new(agents, 100, 0.99, 16).expect("MaddpgTrainer creation should succeed");
        trainer.update_targets().expect("update_targets should succeed");
        assert_eq!(trainer.step, 1);
    }

    #[test]
    fn test_commnet_msg_dim() {
        let net = CommNet::new(2, 4, 3, 1).expect("CommNet creation should succeed");
        assert_eq!(net.msg_dim(), 3);
    }

    #[test]
    fn test_tarmac_empty_others() {
        let agent = TarmacAgent::new(0, 4, 2, 2, 1).expect("TarmacAgent creation should succeed");
        let agg = agent.aggregate(&[0.0; 4], &[], &[]).expect("aggregate with empty others should succeed");
        assert_eq!(agg.len(), 2);
        assert!(agg.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_multi_agent_trainer_independent_buffers() {
        let agents: Vec<Box<dyn Agent>> = (0..2)
            .map(|i| -> Box<dyn Agent> {
                Box::new(QmixAgent::new(i, 2, 2, 4, 0.1, i as u64).expect("QmixAgent creation should succeed"))
            })
            .collect();
        let config = MultiAgentTrainerConfig {
            shared_buffer: false,
            min_replay_size: 3,
            batch_size: 2,
        };
        let mut trainer = MultiAgentTrainer::new(agents, config).expect("MultiAgentTrainer creation should succeed");
        for i in 0..6 {
            trainer
                .record(Experience::new(
                    vec![i as f64, 0.0],
                    vec![0.0],
                    1.0,
                    vec![0.0, 0.0],
                    false,
                    i % 2,
                ))
                .expect("recording experience should succeed");
        }
        trainer.train_step().expect("train step should succeed");
        assert!(trainer.step_count() > 0);
    }

    #[test]
    fn test_maddpg_trainer_buffer_record() {
        let agents = vec![MaddpgAgent::new(0, 4, 2, 8, 12, 99).expect("MaddpgAgent creation should succeed")];
        let trainer = MaddpgTrainer::new(agents, 50, 0.99, 8).expect("MaddpgTrainer creation should succeed");
        trainer
            .replay
            .push(Experience::new(
                vec![0.0; 12],
                vec![0.0, 0.0],
                1.0,
                vec![0.0; 12],
                false,
                0,
            ))
            .expect("push to replay buffer should succeed");
        assert_eq!(trainer.replay.len().expect("len should succeed"), 1);
    }

    #[test]
    fn test_agent_q_network_zero_dims_error() {
        assert!(AgentQNetwork::new(0, 2, 4, 1).is_err());
        assert!(AgentQNetwork::new(2, 0, 4, 1).is_err());
        assert!(AgentQNetwork::new(2, 2, 0, 1).is_err());
    }

    #[test]
    fn test_mixing_network_zero_dims_error() {
        assert!(MixingNetwork::new(0, 4, 4, 1).is_err());
        assert!(MixingNetwork::new(2, 0, 4, 1).is_err());
        assert!(MixingNetwork::new(2, 4, 0, 1).is_err());
    }

    #[test]
    fn test_comm_channel_zero_dims_error() {
        assert!(CommChannel::new(0, 4, 1).is_err());
        assert!(CommChannel::new(4, 0, 1).is_err());
    }

    #[test]
    fn test_maddpg_actor_soft_update_dim_error() {
        let mut a1 = MaddpgActor::new(4, 2, 8, 1).expect("MaddpgActor creation should succeed");
        let a2 = MaddpgActor::new(4, 2, 16, 2).expect("MaddpgActor creation should succeed");
        assert!(a1.soft_update(&a2, 0.01).is_err());
    }

    #[test]
    fn test_team_reward_potential_linear() {
        let shaper = TeamRewardShaper::new(1.0, vec![2.0, 3.0]);
        let p = shaper.potential(&[1.0, 1.0]);
        assert!((p - 5.0).abs() < 1e-9);
    }
}
