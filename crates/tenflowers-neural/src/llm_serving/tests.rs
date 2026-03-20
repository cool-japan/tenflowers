use super::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

// ── KVCacheManager tests ────────────────────────────────────────────

#[test]
fn test_kv_cache_basic_append_and_get() {
    let mut cache = LsKvCacheManager::new(4, 8, 2, 10).expect("cache creation");
    let key = vec![1.0; 8];
    let val = vec![2.0; 8];
    cache.append_kv(0, &key, &val).expect("append");
    assert_eq!(cache.cached_tokens(0), 1);
    let (keys, vals) = cache.get_kv(0, 0, 1).expect("get");
    assert_eq!(keys.len(), 1);
    assert_eq!(vals.len(), 1);
    assert!((keys[0][0] - 1.0).abs() < 1e-10);
    assert!((vals[0][0] - 2.0).abs() < 1e-10);
}

#[test]
fn test_kv_cache_block_allocation() {
    let mut cache = LsKvCacheManager::new(4, 8, 1, 10).expect("cache");
    let indices = cache.allocate_blocks(0, 10).expect("alloc");
    // 10 tokens / 4 per block = 3 blocks
    assert_eq!(indices.len(), 3);
}

#[test]
fn test_kv_cache_lru_eviction() {
    let mut cache = LsKvCacheManager::new(2, 4, 1, 2).expect("cache");
    // Fill 2 blocks (max)
    for i in 0..4 {
        cache
            .append_kv(0, &[i as f64; 4], &[i as f64; 4])
            .expect("append");
    }
    assert_eq!(cache.cached_tokens(0), 4);
    // Append one more, triggers LRU eviction
    cache
        .append_kv(0, &[99.0; 4], &[99.0; 4])
        .expect("append after evict");
}

#[test]
fn test_kv_cache_clear() {
    let mut cache = LsKvCacheManager::new(4, 8, 2, 10).expect("cache");
    cache
        .append_kv(0, &[1.0; 8], &[1.0; 8])
        .expect("ok");
    cache
        .append_kv(1, &[1.0; 8], &[1.0; 8])
        .expect("ok");
    cache.clear_layer(0);
    assert_eq!(cache.cached_tokens(0), 0);
    assert_eq!(cache.cached_tokens(1), 1);
    cache.clear_all();
    assert_eq!(cache.cached_tokens(1), 0);
}

#[test]
fn test_kv_cache_invalid_layer() {
    let mut cache = LsKvCacheManager::new(4, 8, 2, 10).expect("cache");
    assert!(cache.append_kv(5, &[1.0; 8], &[1.0; 8]).is_err());
}

// ── PagedKVCache tests ──────────────────────────────────────────────

#[test]
fn test_paged_cache_allocate_and_append() {
    let mut paged = LsPagedKvCache::new(4, 8, 16).expect("paged");
    assert_eq!(paged.free_blocks(), 16);
    paged
        .append_token(1, &[1.0; 8], &[2.0; 8])
        .expect("append");
    assert_eq!(paged.free_blocks(), 15);
    assert_eq!(paged.active_sequences(), 1);
}

#[test]
fn test_paged_cache_get_sequence_kv() {
    let mut paged = LsPagedKvCache::new(4, 4, 16).expect("paged");
    for i in 0..5 {
        paged
            .append_token(42, &[i as f64; 4], &[(i * 2) as f64; 4])
            .expect("append");
    }
    let (keys, vals) = paged.get_sequence_kv(42).expect("get");
    assert_eq!(keys.len(), 5);
    assert_eq!(vals.len(), 5);
    assert!((keys[3][0] - 3.0).abs() < 1e-10);
}

#[test]
fn test_paged_cache_free_sequence() {
    let mut paged = LsPagedKvCache::new(4, 4, 8).expect("paged");
    for _ in 0..3 {
        paged
            .append_token(1, &[0.0; 4], &[0.0; 4])
            .expect("ok");
    }
    let free_before = paged.free_blocks();
    paged.free_sequence(1).expect("free");
    assert!(paged.free_blocks() > free_before);
    assert_eq!(paged.active_sequences(), 0);
}

#[test]
fn test_paged_cache_out_of_blocks() {
    let mut paged = LsPagedKvCache::new(2, 4, 2).expect("paged");
    // Fill all blocks (2 blocks * 2 tokens = 4 tokens)
    for _ in 0..4 {
        paged
            .append_token(1, &[0.0; 4], &[0.0; 4])
            .expect("ok");
    }
    // Next should fail
    assert!(paged.append_token(1, &[0.0; 4], &[0.0; 4]).is_err());
}

// ── SpeculativeDecoder tests ────────────────────────────────────────

#[test]
fn test_speculative_basic_decode() {
    let mut dec = LsSpeculativeDecoder::new(3, 1, 8, 42).expect("dec");
    let draft_tokens = vec![2, 3, 4];
    // Draft and target agree perfectly
    let draft_logits = vec![
        vec![0.0, 0.0, 10.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 10.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 10.0],
    ];
    let target_logits = draft_logits.clone();
    let result = dec
        .decode(&draft_tokens, &draft_logits, &target_logits)
        .expect("decode");
    assert_eq!(result.n_accepted, 3);
    assert!((result.acceptance_rate - 1.0).abs() < 1e-6);
}

#[test]
fn test_speculative_rejection() {
    let mut dec = LsSpeculativeDecoder::new(3, 1, 8, 42).expect("dec");
    let draft_tokens = vec![0, 0, 0];
    // Draft strongly prefers token 0; target strongly prefers token 1
    let draft_logits = vec![vec![10.0, -10.0], vec![10.0, -10.0], vec![10.0, -10.0]];
    let target_logits = vec![vec![-10.0, 10.0], vec![-10.0, 10.0], vec![-10.0, 10.0]];
    let result = dec
        .decode(&draft_tokens, &draft_logits, &target_logits)
        .expect("decode");
    // Should reject token 0 since target gives it near-zero probability
    assert_eq!(result.n_accepted, 0);
}

#[test]
fn test_speculative_adaptive_k() {
    let mut dec = LsSpeculativeDecoder::new(3, 1, 8, 42).expect("dec");
    dec.ema_acceptance = 0.9;
    dec.increase_threshold = 0.8;
    dec.adapt_k();
    assert_eq!(dec.k, 4);
    dec.ema_acceptance = 0.3;
    dec.decrease_threshold = 0.4;
    dec.adapt_k();
    assert_eq!(dec.k, 3);
}

#[test]
fn test_speculative_token_tree() {
    let dec = LsSpeculativeDecoder::new(3, 1, 5, 42).expect("dec");
    let candidates = vec![vec![(0, -0.5), (1, -1.0)], vec![(2, -0.3), (3, -0.8)]];
    let roots = dec.build_token_tree(&candidates).expect("tree");
    assert_eq!(roots.len(), 2);
    assert_eq!(roots[0].children.len(), 2);
}

#[test]
fn test_speculative_overall_rate() {
    let mut dec = LsSpeculativeDecoder::new(3, 1, 8, 42).expect("dec");
    dec.total_accepted = 7;
    dec.total_proposed = 10;
    assert!((dec.overall_acceptance_rate() - 0.7).abs() < 1e-10);
}

// ── InstructionTuner tests ──────────────────────────────────────────

#[test]
fn test_instruction_dataset_basic() {
    let mut ds = LsInstructionDataset::new(512);
    ds.add_example(LsInstructionExample {
        system: "You are helpful.".to_string(),
        user: "Hello".to_string(),
        assistant: "Hi there!".to_string(),
        task: None,
    });
    assert_eq!(ds.examples.len(), 1);
}

#[test]
fn test_instruction_format_loss_mask() {
    let mut ds = LsInstructionDataset::new(512);
    ds.add_example(LsInstructionExample {
        system: "Sys".to_string(),
        user: "Usr".to_string(),
        assistant: "Ans".to_string(),
        task: None,
    });
    let fmt = ds.format_example(0).expect("format");
    // Only assistant tokens + EOS should have loss_mask = true
    let loss_count = fmt.loss_mask.iter().filter(|&&m| m).count();
    // "Ans" = 3 chars + 1 EOS = 4
    assert_eq!(loss_count, 4);
}

#[test]
fn test_instruction_packing() {
    let mut ds = LsInstructionDataset::new(256);
    for i in 0..3 {
        ds.add_example(LsInstructionExample {
            system: format!("S{}", i),
            user: format!("U{}", i),
            assistant: format!("A{}", i),
            task: Some(format!("task{}", i % 2)),
        });
    }
    let (packed, mask) = ds.pack_examples(&[0, 1, 2]).expect("pack");
    assert_eq!(packed.len(), 256);
    assert_eq!(mask.len(), 256);
}

#[test]
fn test_instruction_multitask_sampling() {
    let mut ds = LsInstructionDataset::new(256);
    for i in 0..10 {
        ds.add_example(LsInstructionExample {
            system: "S".to_string(),
            user: format!("Q{}", i),
            assistant: format!("A{}", i),
            task: Some(if i < 5 { "math" } else { "code" }.to_string()),
        });
    }
    let mut rng = StdRng::seed_from_u64(42);
    let batch = ds.sample_batch(4, &mut rng).expect("sample");
    assert_eq!(batch.len(), 4);
}

#[test]
fn test_instruction_masked_loss() {
    let tuner = LsInstructionTuner::new(LsInstructionDataset::new(64), 0.001, 1, 4);
    let logits = vec![
        vec![0.0, 1.0, 2.0],
        vec![1.0, 0.0, 2.0],
        vec![2.0, 1.0, 0.0],
    ];
    let tokens = vec![2, 1, 0, 2];
    let mask = vec![false, true, true, true];
    let loss = tuner
        .compute_masked_loss(&logits, &tokens, &mask)
        .expect("loss");
    assert!(loss > 0.0);
}

// ── RLHFRewardModel tests ───────────────────────────────────────────

#[test]
fn test_reward_head_forward() {
    let head = LsRewardHead::new(16);
    let hidden = vec![0.5; 16];
    let r = head.forward(&hidden);
    assert!(r.is_finite());
}

#[test]
fn test_reward_normalizer() {
    let mut norm = LsRewardNormalizer::new();
    for i in 0..100 {
        norm.update(i as f64);
    }
    let normalized = norm.normalize(50.0);
    assert!(normalized.abs() < 1.0);
}

#[test]
fn test_rlhf_preference_loss() {
    let model = LsRlhfRewardModel::new(8, 0.5, 0.01);
    let chosen = vec![1.0; 8];
    let rejected = vec![-1.0; 8];
    let loss = model.preference_loss(&chosen, &rejected);
    assert!(loss >= 0.0);
}

#[test]
fn test_rlhf_margin_loss() {
    let model = LsRlhfRewardModel::new(8, 1.0, 0.01);
    let chosen = vec![1.0; 8];
    let rejected = vec![-1.0; 8];
    let loss = model.margin_loss(&chosen, &rejected);
    assert!(loss >= 0.0);
}

#[test]
fn test_rlhf_train_step() {
    let mut model = LsRlhfRewardModel::new(4, 0.5, 0.01);
    let chosen = vec![vec![1.0, 0.5, 0.3, 0.1]; 4];
    let rejected = vec![vec![-0.5, -0.3, 0.1, -0.2]; 4];
    let loss = model.train_step(&chosen, &rejected).expect("train");
    assert!(loss >= 0.0);
    assert_eq!(model.loss_history.len(), 1);
}

#[test]
fn test_rlhf_batch_rewards() {
    let mut model = LsRlhfRewardModel::new(4, 0.5, 0.01);
    let batch = vec![vec![1.0; 4], vec![0.0; 4], vec![-1.0; 4]];
    let rewards = model.compute_rewards(&batch);
    assert_eq!(rewards.len(), 3);
}

// ── DynamicBatcher tests ────────────────────────────────────────────

#[test]
fn test_batcher_basic_batch() {
    let mut batcher = LsDynamicBatcher::new(4, 100, 0);
    batcher.add_request(LsServingRequest {
        request_id: 1,
        token_ids: vec![10, 20, 30],
        max_new_tokens: 50,
        priority: LsRequestPriority::Normal,
        arrival_time: 0,
    });
    batcher.add_request(LsServingRequest {
        request_id: 2,
        token_ids: vec![40, 50],
        max_new_tokens: 50,
        priority: LsRequestPriority::High,
        arrival_time: 0,
    });
    let batch = batcher.form_batch().expect("batch").expect("some batch");
    assert_eq!(batch.request_ids.len(), 2);
    // High priority should come first
    assert_eq!(batch.request_ids[0], 2);
    assert_eq!(batch.max_seq_len, 3);
}

#[test]
fn test_batcher_padding() {
    let mut batcher = LsDynamicBatcher::new(8, 100, 0);
    batcher.add_request(LsServingRequest {
        request_id: 1,
        token_ids: vec![1; 10],
        max_new_tokens: 5,
        priority: LsRequestPriority::Normal,
        arrival_time: 0,
    });
    batcher.add_request(LsServingRequest {
        request_id: 2,
        token_ids: vec![2; 5],
        max_new_tokens: 5,
        priority: LsRequestPriority::Normal,
        arrival_time: 0,
    });
    let batch = batcher.form_batch().expect("batch").expect("some");
    // Both padded to 10
    assert_eq!(batch.padded_tokens[0].len(), 10);
    assert_eq!(batch.padded_tokens[1].len(), 10);
    // Shorter sequence left-padded
    assert_eq!(batch.attention_mask[1][0], 0); // pad
    assert_eq!(batch.attention_mask[1][9], 1); // real
}

#[test]
fn test_batcher_empty_queue() {
    let mut batcher = LsDynamicBatcher::new(4, 100, 0);
    let batch = batcher.form_batch().expect("batch");
    assert!(batch.is_none());
}

#[test]
fn test_batcher_metrics() {
    let mut batcher = LsDynamicBatcher::new(4, 100, 0);
    for i in 0..3 {
        batcher.add_request(LsServingRequest {
            request_id: i,
            token_ids: vec![0; (i + 1) as usize],
            max_new_tokens: 5,
            priority: LsRequestPriority::Normal,
            arrival_time: 0,
        });
    }
    let _batch = batcher.form_batch().expect("ok");
    let avg = batcher.average_metrics();
    assert!(avg.padding_ratio >= 0.0);
    assert!(avg.utilization > 0.0);
}

// ── ModelWarmup tests ───────────────────────────────────────────────

#[test]
fn test_warmup_profile() {
    let warmup = LsModelWarmup::new(3, 64, 1_000_000);
    let layers = vec![
        ("embed", 512, 768),
        ("attn", 768, 768),
        ("ffn", 768, 3072),
        ("head", 3072, 512),
    ];
    let report = warmup.profile_layers(&layers);
    assert_eq!(report.layer_profiles.len(), 4);
    assert!(report.total_latency_us > 0.0);
    assert!(report.recommended_batch_size >= 1);
}

#[test]
fn test_warmup_run() {
    let warmup = LsModelWarmup::new(5, 32, 500_000);
    let layers = vec![("layer0", 64, 128)];
    let total = warmup.run_warmup(&layers);
    assert!(total > 0.0);
}

#[test]
fn test_warmup_batch_search() {
    let warmup = LsModelWarmup::new(1, 128, 10_000_000);
    let optimal = warmup.find_optimal_batch_size(1000);
    assert!(optimal >= 1);
    assert!(optimal <= 128);
}

// ── TokenClassifier tests ───────────────────────────────────────────

#[test]
fn test_token_classifier_greedy() {
    let labels = vec!["O".into(), "B-PER".into(), "I-PER".into()];
    let clf = LsTokenClassifier::new(8, 3, false, LsTaggingScheme::Bio, labels).expect("clf");
    let hidden = vec![vec![0.5; 8]; 4];
    let logits = clf.forward(&hidden);
    assert_eq!(logits.len(), 4);
    let tags = clf.decode_greedy(&logits);
    assert_eq!(tags.len(), 4);
}

#[test]
fn test_token_classifier_viterbi() {
    let labels = vec!["O".into(), "B-LOC".into(), "I-LOC".into()];
    let clf = LsTokenClassifier::new(8, 3, true, LsTaggingScheme::Bio, labels).expect("clf");
    let logits = vec![
        vec![0.1, 2.0, 0.0],
        vec![0.0, 0.0, 2.5],
        vec![0.0, 0.0, 2.0],
        vec![2.0, 0.0, 0.0],
    ];
    let tags = clf.decode_viterbi(&logits).expect("viterbi");
    assert_eq!(tags.len(), 4);
}

#[test]
fn test_token_classifier_entity_extraction_bio() {
    let labels = vec!["O".into(), "B-PER".into(), "I-PER".into(), "B-ORG".into()];
    let clf = LsTokenClassifier::new(8, 4, false, LsTaggingScheme::Bio, labels).expect("clf");
    let tag_indices = vec![1, 2, 0, 3, 0];
    let tokens: Vec<String> = vec!["John", "Smith", "works", "at", "Google"]
        .into_iter()
        .map(String::from)
        .collect();
    let entities = clf.extract_entities(&tag_indices, &tokens);
    assert!(!entities.is_empty());
    assert_eq!(entities[0].entity_type, "PER");
    assert_eq!(entities[0].start, 0);
    assert_eq!(entities[0].end, 2);
}

#[test]
fn test_token_classifier_bilou() {
    let labels = vec![
        "O".into(),
        "B-PER".into(),
        "I-PER".into(),
        "L-PER".into(),
        "U-ORG".into(),
    ];
    let clf = LsTokenClassifier::new(8, 5, false, LsTaggingScheme::Bilou, labels).expect("clf");
    let tag_indices = vec![1, 3, 0, 4];
    let tokens: Vec<String> = vec!["John", "Smith", "at", "IBM"]
        .into_iter()
        .map(String::from)
        .collect();
    let entities = clf.extract_entities(&tag_indices, &tokens);
    assert_eq!(entities.len(), 2);
    assert_eq!(entities[0].entity_type, "PER");
    assert_eq!(entities[1].entity_type, "ORG");
}

#[test]
fn test_token_classifier_no_crf_error() {
    let clf = LsTokenClassifier::new(
        8,
        3,
        false,
        LsTaggingScheme::Bio,
        vec!["O".into(), "B".into(), "I".into()],
    )
    .expect("clf");
    let logits = vec![vec![1.0, 0.0, 0.0]];
    // Should succeed with greedy
    let result = clf.decode(&logits);
    assert!(result.is_ok());
}

// ── ServingMetrics tests ────────────────────────────────────────────

#[test]
fn test_serving_metrics_basic() {
    let mut metrics = LsServingMetrics::new();
    for i in 0..100 {
        metrics.record_ttft(i as f64 * 10.0);
        metrics.record_tps(50.0 + i as f64);
        metrics.record_latency(i as f64 * 100.0);
    }
    let report = metrics.report();
    assert!(report.avg_tps > 0.0);
    assert!(report.p50_ttft_us > 0.0);
    assert!(report.p95_latency_us > report.p50_latency_us);
}

#[test]
fn test_serving_metrics_cache_rate() {
    let mut metrics = LsServingMetrics::new();
    for _ in 0..80 {
        metrics.record_cache_hit();
    }
    for _ in 0..20 {
        metrics.record_cache_miss();
    }
    assert!((metrics.cache_hit_rate() - 0.8).abs() < 1e-10);
}

#[test]
fn test_serving_metrics_sla_check() {
    let mut metrics = LsServingMetrics::new();
    for i in 0..50 {
        metrics.record_ttft(100.0 + i as f64);
        metrics.record_tps(100.0);
        metrics.record_latency(200.0 + i as f64);
    }
    let report = metrics.report();
    let sla = report.check_sla(1000.0, 50.0, 500.0);
    assert!(sla.overall_ok);
}

#[test]
fn test_serving_metrics_queue_depth() {
    let mut metrics = LsServingMetrics::new();
    metrics.record_queue_depth(5);
    metrics.record_queue_depth(10);
    metrics.record_queue_depth(15);
    assert!((metrics.avg_queue_depth() - 10.0).abs() < 1e-10);
}

// ── ContinuousBatchProcessor tests ──────────────────────────────────

#[test]
fn test_cbp_add_and_step() {
    let mut cbp = LsContinuousBatchProcessor::new(4, 100, 1).expect("cbp");
    cbp.add_sequence(1, vec![10, 20, 30], 3, LsRequestPriority::Normal)
        .expect("add");
    assert_eq!(cbp.active_count(), 1);

    // First step: prefill -> decode
    let processed = cbp.step().expect("step");
    assert_eq!(processed.len(), 1);
    assert_eq!(
        cbp.sequences.get(&1).map(|s| s.state),
        Some(LsSequenceState::Decode)
    );

    // Three more steps to complete (max_new_tokens = 3)
    for _ in 0..3 {
        cbp.step().expect("step");
    }
    assert_eq!(cbp.completed_count(), 1);
}

#[test]
fn test_cbp_preemption() {
    // max_memory_blocks=8, blocks_per_token=1
    let mut cbp = LsContinuousBatchProcessor::new(4, 8, 1).expect("cbp");
    // seq1 uses 3 initial blocks
    cbp.add_sequence(1, vec![0; 3], 100, LsRequestPriority::Low)
        .expect("add");
    // prefill -> decode
    cbp.step().expect("step");
    // decode step: seq1 gains 1 block -> 4 blocks used
    cbp.step().expect("step");
    // decode step: seq1 gains 1 block -> 5 blocks used
    cbp.step().expect("step");

    // Now used=5, max=8. Adding seq2 with 6 tokens needs 6 blocks.
    // 5 + 6 = 11 > 8, so preemption of seq1 is triggered.
    cbp.add_sequence(2, vec![0; 6], 10, LsRequestPriority::High)
        .expect("add");
    // Sequence 1 should be preempted to make room
    let s1_state = cbp.sequences.get(&1).map(|s| s.state);
    assert_eq!(s1_state, Some(LsSequenceState::Preempted));
}

#[test]
fn test_cbp_resume() {
    let mut cbp = LsContinuousBatchProcessor::new(4, 200, 1).expect("cbp");
    cbp.add_sequence(1, vec![0; 3], 5, LsRequestPriority::Low)
        .expect("add");
    cbp.step().expect("step"); // prefill -> decode

    // Manually preempt by forcing state
    if let Some(seq) = cbp.sequences.get_mut(&1) {
        cbp.used_memory_blocks = cbp.used_memory_blocks.saturating_sub(seq.memory_blocks);
        seq.state = LsSequenceState::Preempted;
    }

    cbp.resume_sequence(1).expect("resume");
    assert_eq!(
        cbp.sequences.get(&1).map(|s| s.state),
        Some(LsSequenceState::Decode)
    );
}

#[test]
fn test_cbp_memory_utilization() {
    let mut cbp = LsContinuousBatchProcessor::new(4, 100, 1).expect("cbp");
    cbp.add_sequence(1, vec![0; 10], 5, LsRequestPriority::Normal)
        .expect("add");
    let util = cbp.memory_utilization();
    assert!(util > 0.0);
    assert!(util <= 1.0);
}

#[test]
fn test_cbp_admission_control() {
    let cbp = LsContinuousBatchProcessor::new(2, 5, 1).expect("cbp");
    assert!(cbp.can_admit(3));
    assert!(!cbp.can_admit(10)); // exceeds memory
}
