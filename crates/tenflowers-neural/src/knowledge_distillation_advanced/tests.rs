//! Tests for knowledge_distillation_advanced — covers §§1-10 (mod + extensions) and
//! §§11-13 (advanced: token-level, contrastive, structured distillation).

#[cfg(test)]
mod kda_tests {
    use super::super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // §1  Online Distillation
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_online_distil_config_defaults() {
        let cfg = OnlineDistilConfig::default();
        assert!(cfg.temp > 0.0);
        assert!(cfg.alpha >= 0.0 && cfg.alpha <= 1.0);
        assert!(cfg.n_students > 0);
    }

    #[test]
    fn test_online_distil_config_validation() {
        assert!(OnlineDistilConfig::new(0, 2.0, 0.5).is_err());
        assert!(OnlineDistilConfig::new(2, -1.0, 0.5).is_err());
        assert!(OnlineDistilConfig::new(2, 2.0, 1.5).is_err());
        assert!(OnlineDistilConfig::new(2, 2.0, 0.5).is_ok());
    }

    #[test]
    fn test_online_distillation_ensemble_logits() {
        let cfg = OnlineDistilConfig::new(2, 2.0, 0.5).unwrap();
        let od = OnlineDistillation::new(cfg);
        let students = vec![
            vec![1.0_f32, 2.0, 3.0],
            vec![3.0_f32, 2.0, 1.0],
        ];
        let ensemble = od.compute_ensemble_logits(&students).unwrap();
        assert_eq!(ensemble.len(), 3);
        // Average should be 2.0 everywhere
        for &v in &ensemble {
            assert!((v - 2.0).abs() < 1e-5, "v={v}");
        }
    }

    #[test]
    fn test_online_distillation_mutual_loss_same_logits() {
        let cfg = OnlineDistilConfig::new(1, 2.0, 0.5).unwrap();
        let od = OnlineDistillation::new(cfg);
        let logits = vec![1.0_f32, 2.0, 3.0];
        let labels = vec![1.0_f32, 0.0, 0.0];
        let loss = od.mutual_loss(&logits, &logits, &labels, 2.0, 0.5);
        assert!(loss >= 0.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §2  Self-Distillation
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_self_distil_config_defaults() {
        let cfg = SelfDistilConfig::default();
        assert!(cfg.n_layers >= 2);
        assert!(cfg.temp > 0.0);
    }

    #[test]
    fn test_self_distillation_intra_layer_kl_same_vectors() {
        let cfg = SelfDistilConfig::new(4, 2.0).unwrap();
        let sd = SelfDistillation::new(cfg);
        let early = vec![1.0_f32, 2.0, 3.0];
        let late = vec![1.0_f32, 0.0, 0.0];
        // projection: 3×3 identity maps early -> late space (same dim here)
        let proj = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let loss = sd.intra_layer_kl(&early, &late, &proj).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_self_distillation_born_again_loss() {
        let cfg = SelfDistilConfig::default();
        let sd = SelfDistillation::new(cfg);
        // 1 sample, 3 classes
        let logits = vec![0.0_f32, 1.0, 2.0];
        let targets = vec![1usize]; // class index 1
        let loss = sd.born_again_loss(&logits, &logits, &targets, 2.0);
        assert!(loss.is_finite());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §3  Data-Free Distillation
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_generator_config_validation() {
        assert!(GeneratorConfig::new(0, 32, 16).is_err());
        assert!(GeneratorConfig::new(8, 32, 16).is_ok());
    }

    #[test]
    fn test_data_generator_generate_shape() {
        let cfg = GeneratorConfig::new(8, 16, 4).unwrap();
        let gen = DataGenerator::new(cfg);
        let z = vec![0.1_f32; 8];
        let out = gen.generate(&z).unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_data_generator_wrong_z_dim_error() {
        let cfg = GeneratorConfig::new(8, 16, 4).unwrap();
        let gen = DataGenerator::new(cfg);
        let z = vec![0.1_f32; 4]; // wrong dim
        assert!(gen.generate(&z).is_err());
    }

    #[test]
    fn test_dreem_loss_computation() {
        let dl = DreemLoss::new(0.5);
        let gen_data = vec![0.1_f32, 0.2, 0.3];
        let t_logits = vec![1.0_f32, -1.0, 0.5];
        let s_logits = vec![0.8_f32, -0.8, 0.4];
        let loss = dl.dreaming_loss(&gen_data, &t_logits, &s_logits);
        assert!(loss.is_finite());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §4  Task-Agnostic Distillation
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_unlabeled_distil_config_defaults() {
        let cfg = UnlabeledDistilConfig::default();
        assert!(cfg.batch_size > 0);
        assert!(cfg.temp > 0.0);
        assert!(cfg.top_k_tokens > 0);
    }

    #[test]
    fn test_task_agnostic_soft_targets() {
        let cfg = UnlabeledDistilConfig::new(4, 2.0, 2).unwrap();
        let tad = TaskAgnosticDistillation::new(cfg);
        let teacher_logits = vec![
            vec![2.0_f32, 0.5, -1.0, 3.0],
            vec![1.5_f32, 0.3, -0.5, 2.5],
        ];
        let soft = tad.soft_targets_from_teacher(&teacher_logits, 2.0);
        assert_eq!(soft.len(), 2);
        for dist in &soft {
            let s: f32 = dist.iter().sum();
            assert!((s - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_task_agnostic_top_k_soft_targets() {
        let cfg = UnlabeledDistilConfig::default();
        let tad = TaskAgnosticDistillation::new(cfg);
        let soft = vec![0.4_f32, 0.3, 0.2, 0.1];
        let top2 = tad.top_k_soft_targets(&soft, 2);
        assert_eq!(top2.len(), 2);
        // Top-2 should be index 0 (0.4) and index 1 (0.3)
        assert_eq!(top2[0].0, 0);
    }

    #[test]
    fn test_task_agnostic_kd_loss_unlabeled() {
        let cfg = UnlabeledDistilConfig::default();
        let tad = TaskAgnosticDistillation::new(cfg);
        let s_logits = vec![1.0_f32, 0.5, -1.0];
        let soft_targets = vec![(0usize, 0.7_f32), (1usize, 0.3_f32)];
        let loss = tad.kd_loss_unlabeled(&s_logits, &soft_targets);
        assert!(loss >= 0.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §5  Patch Distillation
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_patch_distil_config_defaults() {
        let cfg = PatchDistilConfig::default();
        assert!(cfg.n_patches > 0);
        assert!(cfg.patch_dim > 0);
    }

    #[test]
    fn test_patch_distillation_loss_identical() {
        let cfg = PatchDistilConfig::new(4, 3, 3).unwrap();
        let pd = PatchDistillation::new(cfg);
        let patches: Vec<Vec<f32>> = (0..4)
            .map(|i| vec![i as f32, i as f32 + 1.0, i as f32 + 2.0])
            .collect();
        let loss = pd.patch_similarity_loss(&patches, &patches).unwrap();
        assert!(loss.abs() < 1e-4, "loss={loss}");
    }

    #[test]
    fn test_patch_distillation_loss_count_mismatch_error() {
        let cfg = PatchDistilConfig::new(2, 3, 3).unwrap();
        let pd = PatchDistillation::new(cfg);
        let s: Vec<Vec<f32>> = vec![vec![1.0; 3]];
        let t: Vec<Vec<f32>> = vec![vec![1.0; 3], vec![0.0; 3]];
        assert!(pd.patch_similarity_loss(&s, &t).is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §6  Graph Distillation
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_compute_relation_matrix_shape() {
        let feats: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let rm = compute_relation_matrix(&feats).unwrap();
        assert_eq!(rm.n_samples, 3);
        assert_eq!(rm.similarities.len(), 9);
    }

    #[test]
    fn test_graph_distillation_rku_loss_identical() {
        let feats: Vec<Vec<f32>> = vec![
            vec![1.0_f32, 0.0, 0.0],
            vec![0.0_f32, 1.0, 0.0],
            vec![0.0_f32, 0.0, 1.0],
            vec![0.5_f32, 0.5, 0.0],
        ];
        let rm = compute_relation_matrix(&feats).unwrap();
        let gd = GraphDistillation;
        let loss = gd.rku_loss(&rm, &rm).unwrap();
        assert!(loss.abs() < 1e-4, "loss={loss}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §7  Progressive Distillation
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_progressive_distillation_stage_loss() {
        let cfg = ProgressiveConfig::new(2, 0.5).unwrap();
        let pd = ProgressiveDistillation::new(cfg);
        let teacher = vec![1.0_f32, 2.0, 3.0, 4.0];
        let student = vec![1.0_f32, 2.0, 3.0, 4.0];
        let loss = pd.stage_loss(&student, &teacher);
        assert!(loss.abs() < 1e-5, "loss={loss}");
    }

    #[test]
    fn test_staged_model_advance() {
        let mut sm = StagedModel::new(vec![64, 32, 16]);
        assert_eq!(sm.hidden_dims.len(), 3);
        sm.advance_stage(0.5);
        // current stage should advance
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §8  Attention Transfer
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_compute_attention_map_shape() {
        let n_heads = 2;
        let seq_len = 3;
        let head_dim = 4;
        let total = n_heads * seq_len * head_dim;
        let q: Vec<f32> = (0..total).map(|x| x as f32 * 0.1).collect();
        let map = compute_attention_map(&q, &q, n_heads, seq_len).unwrap();
        assert_eq!(map.n_heads, n_heads);
        assert_eq!(map.seq_len, seq_len);
        assert_eq!(map.values.len(), n_heads * seq_len * seq_len);
    }

    #[test]
    fn test_attention_transfer_loss_identical() {
        let n_heads = 2;
        let seq_len = 3;
        let head_dim = 4;
        let total = n_heads * seq_len * head_dim;
        let q: Vec<f32> = (0..total).map(|x| x as f32 * 0.1).collect();
        let map = compute_attention_map(&q, &q, n_heads, seq_len).unwrap();
        let at = AttentionTransfer;
        let loss = at.at_loss(&map, &map).unwrap();
        assert!(loss.abs() < 1e-5, "loss={loss}");
    }

    #[test]
    fn test_attention_transfer_gram_matrix_loss() {
        let at = AttentionTransfer;
        let feat = vec![1.0_f32, 0.0, 0.0];
        let loss = at.gram_matrix_loss(&feat, &feat).unwrap();
        assert!(loss.abs() < 1e-5);
    }

    #[test]
    fn test_attention_map_get_accessor() {
        let n_heads = 1;
        let seq_len = 2;
        let head_dim = 2;
        let total = n_heads * seq_len * head_dim;
        let q = vec![1.0_f32; total];
        let k = vec![0.5_f32; total];
        let map = compute_attention_map(&q, &k, n_heads, seq_len).unwrap();
        let v = map.get(0, 0, 0);
        assert!((0.0..=1.0).contains(&v));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §9  Distillation Scheduler
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_kd_scheduler_constant() {
        let mut sched = KdDistilScheduler::new(
            TempSchedule::Constant(4.0),
            AlphaSchedule::Constant(0.7),
            10,
        );
        let (t, a) = sched.step();
        assert!((t - 4.0).abs() < 1e-5);
        assert!((a - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_kd_scheduler_linear_decay() {
        let mut sched = KdDistilScheduler::new(
            TempSchedule::LinearDecay { start: 8.0, end: 1.0 },
            AlphaSchedule::Constant(0.5),
            100,
        );
        let (t0, _) = sched.step();
        assert!((t0 - 8.0).abs() < 0.5);
        for _ in 0..99 {
            sched.step();
        }
        let (t_end, _) = sched.step();
        assert!(t_end < 2.0, "t_end={t_end}");
    }

    #[test]
    fn test_kd_scheduler_cosine_annealing() {
        let mut sched = KdDistilScheduler::new(
            TempSchedule::CosineAnnealing { t_max: 10.0, t_min: 1.0 },
            AlphaSchedule::LinearRise { start: 0.0, end: 1.0 },
            10,
        );
        for _ in 0..10 {
            let (t, a) = sched.step();
            assert!((1.0..=10.0).contains(&t), "t={t}");
            assert!((0.0..=1.0).contains(&a), "a={a}");
        }
    }

    #[test]
    fn test_kd_scheduler_cyclic_warm() {
        let mut sched = KdDistilScheduler::new(
            TempSchedule::CyclicWarm { base: 1.0, peak: 5.0, cycle_len: 4 },
            AlphaSchedule::Constant(0.5),
            20,
        );
        for _ in 0..20 {
            let (t, _) = sched.step();
            assert!((1.0..=5.0).contains(&t), "t={t}");
        }
    }

    #[test]
    fn test_kd_scheduler_reset() {
        let mut sched = KdDistilScheduler::new(
            TempSchedule::Constant(3.0),
            AlphaSchedule::Constant(0.5),
            10,
        );
        sched.step();
        sched.step();
        assert_eq!(sched.current_step, 2);
        sched.reset();
        assert_eq!(sched.current_step, 0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §10  Efficient Transfer Learning
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_freeze_schedule_should_unfreeze() {
        let sched = FreezeSchedule::new(4, 10).unwrap();
        assert!(sched.should_unfreeze_layer(0, 0));
        assert!(!sched.should_unfreeze_layer(1, 9));
        assert!(sched.should_unfreeze_layer(1, 10));
    }

    #[test]
    fn test_efficient_transfer_apply_unfreeze() {
        let groups = vec![
            LayerGroup::new_frozen(100),
            LayerGroup::new_frozen(200),
            LayerGroup::new_frozen(300),
        ];
        let sched = FreezeSchedule::new(3, 5).unwrap();
        let mut etl = EfficientTransferLearning::new(groups, sched);
        let states = etl.apply_unfreeze(10);
        assert!(states[0]);
        assert!(states[1]);
        assert!(states[2]);
    }

    #[test]
    fn test_efficient_transfer_active_frozen_params() {
        let groups = vec![
            LayerGroup::new_unfrozen(100),
            LayerGroup::new_frozen(200),
        ];
        let sched = FreezeSchedule::new(2, 100).unwrap();
        let etl = EfficientTransferLearning::new(groups, sched);
        assert_eq!(etl.active_params(), 100);
        assert_eq!(etl.frozen_params(), 200);
    }

    #[test]
    fn test_efficient_transfer_discriminative_lr() {
        let groups = vec![
            LayerGroup::new_unfrozen(100),
            LayerGroup::new_unfrozen(100),
        ];
        let sched = FreezeSchedule::new(2, 1).unwrap();
        let etl = EfficientTransferLearning::new(groups, sched);
        let lr0 = etl.compute_layer_lr(0, 0.01, 0.5);
        let lr1 = etl.compute_layer_lr(1, 0.01, 0.5);
        assert!((lr0 - 0.01).abs() < 1e-6);
        assert!((lr1 - 0.005).abs() < 1e-6);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §11  Token-Level Distillation
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_token_embedding_distiller_construction() {
        let ted = TokenEmbeddingDistiller::new(4, 8).unwrap();
        assert_eq!(ted.student_dim, 4);
        assert_eq!(ted.teacher_dim, 8);
        assert_eq!(ted.projection.len(), 4 * 8);
    }

    #[test]
    fn test_token_embedding_distiller_project_shape() {
        let ted = TokenEmbeddingDistiller::new(4, 8).unwrap();
        let token: Vec<f32> = (0..4).map(|x| x as f32).collect();
        let projected = ted.project(&token);
        assert_eq!(projected.len(), 8);
    }

    #[test]
    fn test_token_embedding_distiller_loss_same_seqlen() {
        let dim = 4;
        let ted = TokenEmbeddingDistiller::new(dim, dim).unwrap();
        let tokens: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        let loss = ted.token_embedding_loss(&tokens, &tokens).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_token_embedding_distiller_mismatch_error() {
        let ted = TokenEmbeddingDistiller::new(4, 8).unwrap();
        let s_tokens = vec![vec![1.0_f32; 4]];
        let t_tokens = vec![vec![1.0_f32; 4], vec![0.0_f32; 4]];
        let result = ted.token_embedding_loss(&s_tokens, &t_tokens);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_map_distiller_loss_identical() {
        let n_heads = 2;
        let seq_len = 3;
        let amd = AttentionMapDistiller::new(n_heads, seq_len).unwrap();
        let n = n_heads * seq_len * seq_len;
        let maps: Vec<f32> = (0..n).map(|x| (x as f32) * 0.01).collect();
        let loss = amd.attention_map_loss(&maps, &maps).unwrap();
        assert!(loss.abs() < 1e-5, "loss={loss}");
    }

    #[test]
    fn test_attention_map_distiller_loss_different() {
        let n_heads = 1;
        let seq_len = 2;
        let amd = AttentionMapDistiller::new(n_heads, seq_len).unwrap();
        let a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let b = vec![0.5_f32, 0.5, 0.5, 0.5];
        let loss = amd.attention_map_loss(&a, &b).unwrap();
        assert!(loss > 0.0);
    }

    #[test]
    fn test_hidden_state_distiller_loss() {
        let student_dim = 4;
        let teacher_dim = 8;
        let n_layers = 2;
        let hsd = HiddenStateDistiller::new(n_layers, student_dim, teacher_dim).unwrap();
        let s_hiddens: Vec<Vec<f32>> = (0..3)
            .map(|i| (0..student_dim).map(|j| (i * student_dim + j) as f32 * 0.1).collect())
            .collect();
        let t_hiddens: Vec<Vec<f32>> = (0..3)
            .map(|i| (0..teacher_dim).map(|j| (i * teacher_dim + j) as f32 * 0.1).collect())
            .collect();
        let loss = hsd.hidden_state_loss(&s_hiddens, &t_hiddens).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_hidden_state_distiller_seq_mismatch_error() {
        let hsd = HiddenStateDistiller::new(1, 4, 8).unwrap();
        let s_hiddens = vec![vec![0.0_f32; 4]];
        let t_hiddens = vec![vec![0.0_f32; 8], vec![1.0_f32; 8]];
        let result = hsd.hidden_state_loss(&s_hiddens, &t_hiddens);
        assert!(result.is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §12  Contrastive Distillation
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_contrastive_distillation_loss_finite() {
        let cd = ContrastiveDistillationLoss::new(0.07).unwrap();
        let feats: Vec<Vec<f32>> = vec![
            vec![1.0_f32, 0.0],
            vec![0.0_f32, 1.0],
        ];
        let loss = cd.loss(&feats, &feats);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_contrastive_distillation_symmetric_loss() {
        let cd = ContrastiveDistillationLoss::new(0.1).unwrap();
        let s: Vec<Vec<f32>> = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        let t: Vec<Vec<f32>> = vec![vec![0.7_f32, 0.3], vec![0.3_f32, 0.7]];
        let sym = cd.symmetric_loss(&s, &t);
        assert!(sym.is_finite());
        assert!(sym >= 0.0);
    }

    #[test]
    fn test_contrastive_distillation_invalid_temp() {
        let result = ContrastiveDistillationLoss::new(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_semckd_distiller_loss_identical() {
        let dim = 3;
        let sd = SemckdDistiller::new(dim).unwrap();
        let feats: Vec<Vec<f32>> = vec![
            vec![1.0_f32, 0.0, 0.0],
            vec![0.0_f32, 1.0, 0.0],
        ];
        let loss = sd.semckd_loss(&feats, &feats).unwrap();
        assert!(loss.abs() < 1e-4, "loss={loss}");
    }

    #[test]
    fn test_semckd_distiller_loss_different() {
        let dim = 4;
        let sd = SemckdDistiller::new(dim).unwrap();
        let s_feats: Vec<Vec<f32>> = (0..3)
            .map(|i| (0..dim).map(|j| (i * dim + j) as f32 * 0.1).collect())
            .collect();
        let t_feats: Vec<Vec<f32>> = (0..3)
            .map(|i| (0..dim).map(|j| (i * dim + j + 1) as f32 * 0.1).collect())
            .collect();
        let loss = sd.semckd_loss(&s_feats, &t_feats).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_semckd_distiller_batch_mismatch_error() {
        let sd = SemckdDistiller::new(4).unwrap();
        let s = vec![vec![0.0_f32; 4]];
        let t = vec![vec![0.0_f32; 4], vec![1.0_f32; 4]];
        assert!(sd.semckd_loss(&s, &t).is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §13  Structured Distillation
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_relational_kd_loss_identical() {
        let rkd = RelationalKdLoss::new(1.0, 1.0);
        let feats: Vec<Vec<f32>> = vec![
            vec![1.0_f32, 0.0],
            vec![0.0_f32, 1.0],
            vec![0.5_f32, 0.5],
        ];
        let loss = rkd.combined_loss(&feats, &feats);
        assert!(loss.abs() < 1e-4, "loss={loss}");
    }

    #[test]
    fn test_relational_kd_loss_different() {
        let rkd = RelationalKdLoss::new(1.0, 1.0);
        let s = vec![
            vec![1.0_f32, 0.0],
            vec![0.0_f32, 1.0],
        ];
        let t = vec![
            vec![0.5_f32, 0.5],
            vec![0.5_f32, 0.5],
        ];
        let loss = rkd.combined_loss(&s, &t);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_relational_kd_distance_weight_only() {
        let rkd = RelationalKdLoss::new(1.0, 0.0);
        let feats: Vec<Vec<f32>> = vec![
            vec![1.0_f32, 0.0],
            vec![0.0_f32, 1.0],
        ];
        let loss = rkd.combined_loss(&feats, &feats);
        assert!(loss.abs() < 1e-4);
    }

    #[test]
    fn test_graph_distillation_layer_loss_identical() {
        let n_nodes = 3;
        let node_dim = 4;
        let gdl = GraphDistillationLayer::new(n_nodes, node_dim, 1.0).unwrap();
        let nodes: Vec<Vec<f32>> = (0..n_nodes)
            .map(|i| (0..node_dim).map(|j| (i * node_dim + j) as f32 * 0.1).collect())
            .collect();
        let loss = gdl.graph_distil_loss(&nodes, &nodes).unwrap();
        assert!(loss.abs() < 1e-4, "loss={loss}");
    }

    #[test]
    fn test_graph_distillation_layer_different_nodes() {
        let n_nodes = 4;
        let node_dim = 3;
        let gdl = GraphDistillationLayer::new(n_nodes, node_dim, 0.5).unwrap();
        let s: Vec<Vec<f32>> = (0..n_nodes)
            .map(|i| (0..node_dim).map(|j| i as f32 * 0.1 + j as f32).collect())
            .collect();
        let t: Vec<Vec<f32>> = (0..n_nodes)
            .map(|i| (0..node_dim).map(|j| (n_nodes - i) as f32 * 0.1 + j as f32).collect())
            .collect();
        let loss = gdl.graph_distil_loss(&s, &t).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_graph_distillation_layer_node_count_mismatch_error() {
        let gdl = GraphDistillationLayer::new(3, 4, 1.0).unwrap();
        let s: Vec<Vec<f32>> = vec![vec![1.0_f32; 4], vec![0.0_f32; 4]];
        let t: Vec<Vec<f32>> = vec![vec![1.0_f32; 4], vec![0.0_f32; 4], vec![0.5_f32; 4]];
        let result = gdl.graph_distil_loss(&s, &t);
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_distillation_layer_invalid_temp_error() {
        let result = GraphDistillationLayer::new(3, 4, 0.0);
        assert!(result.is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Utility function tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_softmax_temp_sums_to_one() {
        use super::super::softmax_temp;
        let logits = vec![1.0_f32, 2.0, 3.0];
        let sm = softmax_temp(&logits, 1.0);
        let s: f32 = sm.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum={s}");
    }

    #[test]
    fn test_kl_div_identical() {
        use super::super::kl_div;
        let p = vec![0.25_f32, 0.5, 0.25];
        let q = vec![0.25_f32, 0.5, 0.25];
        let kl = kl_div(&p, &q);
        assert!(kl.abs() < 1e-5, "kl={kl}");
    }
}
