use super::*;
use scirs2_core::random::SeedableRng;


#[test]
fn test_lottery_ticket_sparsity() {
    let mut finder = LotteryTicketFinder::new();
    let weights = vec![vec![0.1, 0.5, 0.2, 0.9], vec![0.3, 0.05, 0.7, 0.01]];
    let pruned = finder.prune_to_sparsity(weights, 0.5);
    let zeros: usize = pruned
        .iter()
        .flat_map(|r| r.iter())
        .filter(|&&v| v == 0.0)
        .count();
    let total: usize = pruned.iter().map(|r| r.len()).sum();
    let actual_sparsity = zeros as f64 / total as f64;
    assert!(
        (actual_sparsity - 0.5).abs() <= 0.15,
        "sparsity {actual_sparsity:.2} not near 0.5"
    );
}

#[test]
fn test_lottery_ticket_find_ticket() {
    let mut finder = LotteryTicketFinder::new();
    let init = vec![vec![0.1, 0.5, 0.01, 0.9], vec![0.3, 0.05, 0.7, 0.02]];
    let schedule = vec![0.3, 0.5];
    let (mask, sparse) = finder.find_ticket(init.clone(), &schedule);
    for row in &mask {
        for &v in row {
            assert!(v == 0.0 || v == 1.0, "mask value {v} not binary");
        }
    }
    for (mr, sr) in mask.iter().zip(sparse.iter()) {
        for (&m, &s) in mr.iter().zip(sr.iter()) {
            if m == 0.0 {
                assert_eq!(s, 0.0, "pruned weight should be zero");
            }
        }
    }
}

#[test]
fn test_gradual_pruning_schedule() {
    let gmp = GradualMagnitudePruning::new(0.0, 0.9, 0, 1);
    let s0 = gmp.sparsity_at(0, 10, 0.0, 0.9);
    assert!((s0 - 0.9).abs() < 0.01 || (s0 - 0.0).abs() < 0.01);
    let sf = gmp.sparsity_at(10, 10, 0.0, 0.9);
    assert!((sf - 0.9).abs() < 0.01, "final sparsity {sf} not near 0.9");
    let gmp2 = GradualMagnitudePruning::new(0.0, 0.9, 10, 1);
    let s = gmp2.sparsity_at(5, 20, 0.0, 0.9);
    assert!(
        (s - 0.0).abs() < 1e-9,
        "before t_start should return s_i=0.0"
    );
}

#[test]
fn test_structured_channel_pruning() {
    let weights = vec![
        vec![1.0, 1.0, 1.0], // L2 = sqrt(3) ≈ 1.73
        vec![0.1, 0.1, 0.1], // L2 ≈ 0.17 — should be pruned
        vec![2.0, 2.0, 2.0], // L2 ≈ 3.46 — kept
    ];
    let importance = StructuredChannelPruning::channel_importance(&weights);
    assert!(importance[2] > importance[0]);
    assert!(importance[0] > importance[1]);

    let pruned = StructuredChannelPruning::prune_channels(weights, 0.67);
    let weak_sum: f64 = pruned[1].iter().sum();
    assert_eq!(weak_sum, 0.0, "weakest channel should be zeroed");
}

#[test]
fn test_movement_score_positive() {
    let w = vec![vec![1.0, -0.5], vec![0.3, 0.8]];
    let g = vec![vec![0.2, -0.1], vec![-0.4, 0.6]];
    let scores = MovementPruning::movement_score(&w, &g).expect("movement score failed");
    assert!((scores[0][0] - 0.2).abs() < 1e-9);
    assert!((scores[0][1] - 0.05).abs() < 1e-9);
    assert!((scores[1][1] - 0.48).abs() < 1e-9);
}

#[test]
fn test_sparsity_report() {
    let weights = vec![vec![1.0, 0.0, 0.0, 2.0]];
    let masks = vec![vec![1.0, 0.0, 0.0, 1.0]];
    let report = SparsityReport::from_weights_and_masks(&weights, &masks)
        .expect("sparsity report failed");
    assert_eq!(report.total_params, 4);
    assert_eq!(report.nonzero_params, 2);
    assert!((report.sparsity - 0.5).abs() < 1e-9);
}

#[test]
fn test_pkd_loss_positive() {
    let pkd = PkdDistillation::new(1.0);
    let student = vec![vec![1.0, 0.0], vec![0.5, 0.5]];
    let teacher = vec![vec![1.0, 0.0], vec![0.5, 0.5]];
    let loss = pkd.pkd_loss(&student, &teacher).expect("pkd loss failed");
    assert!((loss - 0.0).abs() < 1e-9, "identical layers → zero loss");

    let student2 = vec![vec![0.0, 1.0], vec![0.0, 0.0]];
    let loss2 = pkd.pkd_loss(&student2, &teacher).expect("pkd loss failed");
    assert!(loss2 > 0.0, "different layers → positive loss");
}

#[test]
fn test_rkd_distance_loss() {
    let rkd = RkdDistillation::new(1.0, 0.0);
    let embs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
    let loss = rkd.rkd_loss(&embs, &embs).expect("rkd loss failed");
    assert!(loss.abs() < 1e-9, "identical embeddings → zero RKD loss");
}

#[test]
fn test_rkd_angle_loss() {
    let rkd = RkdDistillation::new(0.0, 1.0);
    let embs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
    let loss = rkd.rkd_loss(&embs, &embs).expect("rkd angle loss failed");
    assert!(loss.abs() < 1e-9, "identical embeddings → zero angle loss");
}

#[test]
fn test_crd_loss() {
    let crd = CrdDistillation::new(0.07);
    let student = vec![1.0, 0.0, 0.0];
    let teacher = vec![1.0, 0.0, 0.0]; // perfect match
    let negatives = vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
    let loss = crd
        .crd_loss(&student, &teacher, &negatives)
        .expect("crd loss failed");
    assert!(loss >= 0.0, "CRD loss must be non-negative");
    assert!(loss < 2.0, "CRD loss {loss} unexpectedly large");
}

#[test]
fn test_distillation_scheduler_anneal() {
    let sched = DistillationScheduler::new(8.0, 1.0, 100);
    let t0 = sched.temperature(0);
    let t_end = sched.temperature(100);
    let a0 = sched.alpha(0);
    let a_end = sched.alpha(100);
    assert!(
        (t0 - 8.0).abs() < 1e-9,
        "temperature at step 0 should be 8.0, got {t0}"
    );
    assert!(
        (t_end - 1.0).abs() < 1e-9,
        "temperature at final step should be 1.0, got {t_end}"
    );
    assert!(
        (a0 - 1.0).abs() < 1e-9,
        "alpha at step 0 should be 1.0, got {a0}"
    );
    assert!(
        (a_end - 0.0).abs() < 1e-9,
        "alpha at final step should be 0.0, got {a_end}"
    );
    let t_mid = sched.temperature(50);
    assert!(t_mid > t_end && t_mid < t0);
}

#[test]
fn test_gptq_quantize_range() {
    let gptq = GptqQuantizer::new(8);
    let w = vec![vec![0.5, -0.3, 0.8], vec![-0.1, 0.9, -0.7]];
    let h = vec![1.0, 0.5, 2.0];
    let (qw, scales) = gptq
        .quantize_block(&w, &h, 8)
        .expect("gptq quantize failed");
    assert_eq!(qw.len(), 2);
    assert_eq!(scales.len(), 2);
    for (row, qrow) in w.iter().zip(qw.iter()) {
        let max_abs = row.iter().map(|&v| v.abs()).fold(0.0f64, f64::max);
        for &qv in qrow {
            assert!(qv.abs() <= max_abs * 1.01 + 0.01, "qv {qv} out of range");
        }
    }
}

#[test]
fn test_awq_scale_positive() {
    let awq = AwqQuantizer::new(0.5);
    let w = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
    let activations = vec![1.0, 2.0, 0.5];
    let scales = awq
        .compute_scale(&w, &activations)
        .expect("awq scale failed");
    assert_eq!(scales.len(), 3);
    for &s in &scales {
        assert!(s > 0.0, "scale must be positive, got {s}");
    }
}

#[test]
fn test_smoothquant_balance() {
    let sq = SmoothQuant::new(0.5);
    let w = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let act_scale = vec![8.0, 2.0]; // channel 0 has high activation → migrate difficulty
    let (smooth_w, scales) = sq.smooth(&w, &act_scale).expect("smoothquant failed");
    assert_eq!(scales.len(), 2);
    for &s in &scales {
        assert!(s > 0.0, "scale {s} not positive");
    }
    for row in &smooth_w {
        assert!(
            row[0].abs() >= w[0][0].abs(),
            "column 0 should be amplified"
        );
    }
}

#[test]
fn test_fp8_quantize_e4m3() {
    let x = vec![0.0, 1.5, -2.75, 100.0, 448.1];
    let q = Fp8Quantizer::quantize_fp8(&x, Fp8Format::E4M3);
    assert_eq!(q.len(), 5);
    assert_eq!(q[0], 0.0);
    assert!(q[4].abs() <= 448.0 + 1.0, "E4M3 max exceeded: {}", q[4]);
}

#[test]
fn test_fp8_quantize_e5m2() {
    let x = vec![1.0, -3.125, 57344.5];
    let q = Fp8Quantizer::quantize_fp8(&x, Fp8Format::E5M2);
    assert_eq!(q.len(), 3);
    assert!(q[2].abs() <= 57344.0 + 1.0, "E5M2 max exceeded: {}", q[2]);
}

#[test]
fn test_quantization_calibrator() {
    let data: Vec<f64> = (-50..=50).map(|i| i as f64).collect();
    let cal = QuantizationCalibrator::new(8, CalibrationMethod::MinMax);
    let res = cal.calibrate(&data).expect("calibration failed");
    assert!((res.range_min - (-50.0)).abs() < 1.0);
    assert!((res.range_max - 50.0).abs() < 1.0);
    assert!(res.scale > 0.0);
    let cal2 = QuantizationCalibrator::new(8, CalibrationMethod::Percentile(99.0));
    let res2 = cal2
        .calibrate(&data)
        .expect("percentile calibration failed");
    assert!(res2.scale > 0.0);
    let cal3 = QuantizationCalibrator::new(8, CalibrationMethod::MseSearch);
    let res3 = cal3.calibrate(&data).expect("mse calibration failed");
    assert!(res3.scale > 0.0);
}

#[test]
fn test_model_batcher_add() {
    let mut batcher = ModelBatcher::new();
    batcher.add_request(vec![1.0, 2.0]);
    batcher.add_request(vec![3.0, 4.0]);
    batcher.add_request(vec![5.0, 6.0]);
    assert_eq!(batcher.pending(), 3);
    let batch = batcher.get_batch(2, 0);
    assert_eq!(batch.len(), 2);
    assert_eq!(batcher.pending(), 1);
}

#[test]
fn test_throughput_monitor_record() {
    let empty = ThroughputMonitor::new();
    assert_eq!(empty.report().n_requests, 0);
    assert_eq!(empty.report().mean_latency_ms, 0.0);
    let mut mon = ThroughputMonitor::new();
    for i in 1..=10 {
        mon.record(i as f64 * 10.0);
    }
    let report = mon.report();
    assert_eq!(report.n_requests, 10);
    assert!((report.mean_latency_ms - 55.0).abs() < 1.0);
}

#[test]
fn test_throughput_percentiles() {
    let mut mon = ThroughputMonitor::new();
    for i in 1..=100 {
        mon.record(i as f64);
    }
    let report = mon.report();
    assert!(report.p50_latency_ms >= 49.0 && report.p50_latency_ms <= 51.0);
    assert!(report.p95_latency_ms >= 94.0 && report.p95_latency_ms <= 96.0);
    assert!(report.p99_latency_ms >= 98.0 && report.p99_latency_ms <= 100.0);
}

#[test]
fn test_cache_manager() {
    let mut cm = CacheManager::new(4, 8, 10);
    assert_eq!(cm.free_pages(), 10);
    let slot = cm.allocate_page().expect("allocate failed");
    assert_eq!(cm.free_pages(), 9);
    cm.write_kv(slot, vec![0.1; 8], vec![0.2; 8])
        .expect("write_kv failed");
    cm.free_page(slot).expect("free_page failed");
    assert_eq!(cm.free_pages(), 10);
    let mut cm2 = CacheManager::new(2, 4, 2);
    let s0 = cm2.allocate_page().expect("alloc 0");
    let _s1 = cm2.allocate_page().expect("alloc 1");
    assert!(cm2.allocate_page().is_err(), "should fail when full");
    cm2.free_page(s0).expect("free 0");
    let _s2 = cm2.allocate_page().expect("alloc after free");
}

#[test]
fn test_request_scheduler() {
    let mut sched = RequestScheduler::new();
    sched.submit(vec![1.0], 1);
    sched.submit(vec![2.0], 5); // highest priority
    sched.submit(vec![3.0], 3);
    assert_eq!(sched.len(), 3);
    let first = sched.next().expect("should have request");
    assert_eq!(first.priority, 5, "highest priority should come first");
    let second = sched.next().expect("should have request");
    assert_eq!(second.priority, 3);
}

#[test]
fn test_benchmark_runner() {
    let mut rng = StdRng::seed_from_u64(42);
    let stats = BenchmarkRunner::run(8, 20, &mut rng);
    assert_eq!(stats.n_iter, 20);
    assert!(stats.mean_us > 0.0, "mean latency must be positive");
    assert!(stats.std_us >= 0.0, "std must be non-negative");
    assert!(stats.max_us >= stats.min_us);
}

#[test]
fn test_torchscript_export() {
    let layers = vec![
        ExportLayer {
            layer_type: "linear".to_string(),
            weight: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            bias: vec![0.0, 0.0],
        },
        ExportLayer {
            layer_type: "relu".to_string(),
            weight: vec![],
            bias: vec![],
        },
    ];
    let tmp = std::env::temp_dir().join("ts_export_test.json");
    let path = tmp.to_str().expect("tmp path is valid utf-8");
    TorchScriptExporter::export(&layers, path).expect("export failed");
    let content = fs::read_to_string(path).expect("read export file");
    assert!(content.contains("\"linear\""));
    assert!(content.contains("\"relu\""));
}

#[test]
fn test_onnx_exporter_conv1d() {
    let mut exp = OnnxExporter::new();
    exp.add_conv1d("input", "conv_out", 3, 1);
    exp.add_gru("conv_out", "gru_out", 128);
    exp.add_batch_norm("gru_out", "bn_out", 128);
    assert_eq!(exp.nodes().len(), 3);
    assert_eq!(exp.nodes()[0].op_type, ExtOnnxOpType::Conv1d);
    assert_eq!(exp.nodes()[1].op_type, ExtOnnxOpType::Gru);
    assert_eq!(exp.nodes()[2].op_type, ExtOnnxOpType::BatchNorm);
    let tmp = std::env::temp_dir().join("onnx_test.json");
    exp.export_json(tmp.to_str().expect("tmp path"))
        .expect("onnx export failed");
}

#[test]
fn test_coreml_converter() {
    let info = vec![
        (
            "fc1".to_string(),
            "innerProduct".to_string(),
            784,
            256,
            true,
        ),
        ("fc2".to_string(), "innerProduct".to_string(), 256, 10, true),
    ];
    let spec = CoreMlConverter::convert(&info);
    assert_eq!(spec.spec_version, 5);
    assert_eq!(spec.layers.len(), 2);
    assert_eq!(spec.layers[0].input_channels, 784);
    assert_eq!(spec.layers[1].output_channels, 10);
    assert!(spec.layers[0].has_bias);
}

#[test]
fn test_tflite_converter() {
    let ops = vec![
        (TfliteOpCode::FullyConnected, 0, 1),
        (TfliteOpCode::Relu, 1, 2),
        (TfliteOpCode::Softmax, 2, 3),
    ];
    let shapes = vec![vec![1, 128], vec![1, 64], vec![1, 64], vec![1, 10]];
    let model = TfliteConverter::convert(&ops, shapes.clone());
    assert_eq!(model.version, 3);
    assert_eq!(model.ops.len(), 3);
    assert_eq!(model.ops[0].op_code, TfliteOpCode::FullyConnected);
    assert_eq!(model.ops[2].op_code, TfliteOpCode::Softmax);
    assert_eq!(model.tensor_shapes, shapes);
}

#[test]
fn test_benchmark_stats() {
    let mut rng = StdRng::seed_from_u64(7);
    let stats = BenchmarkRunner::run(16, 50, &mut rng);
    assert_eq!(stats.n_iter, 50);
    assert!(stats.mean_us > 0.0);
    assert!(stats.median_us >= stats.min_us);
    assert!(stats.median_us <= stats.max_us);
}

#[test]
fn test_feature_distillation_fit_loss() {
    let fd = FeatureDistillation::new(4, 4);
    let proj: Vec<Vec<f64>> = (0..4)
        .map(|i| (0..4).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let feat = vec![1.0, 2.0, 3.0, 4.0];
    let loss = fd.fit_loss(&feat, &feat, &proj).expect("fit loss");
    assert!(
        loss.abs() < 1e-9,
        "identity projection → zero loss, got {loss}"
    );
}

#[test]
fn test_model_profiler() {
    let profiler = ModelProfiler::new(1e-3);
    let sizes = vec![(128, 64), (64, 32), (32, 10)];
    let profiles = profiler.profile_forward(&sizes);
    assert_eq!(profiles.len(), 3);
    assert_eq!(profiles[0].flops, 16384);
    for p in &profiles {
        assert!(p.latency_us > 0.0);
    }
}

#[test]
fn test_cache_manager_full() {
    let mut cm = CacheManager::new(2, 4, 2);
    let s0 = cm.allocate_page().expect("allocate 0");
    let s1 = cm.allocate_page().expect("allocate 1");
    assert!(cm.allocate_page().is_err(), "should fail when full");
    cm.free_page(s0).expect("free 0");
    let _s2 = cm.allocate_page().expect("allocate after free");
}

#[test]
fn test_distillation_scheduler_mid() {
    let sched = DistillationScheduler::new(10.0, 2.0, 100);
    let t_mid = sched.temperature(50);
    assert!(
        (t_mid - 6.0).abs() < 0.1,
        "mid temperature should be ~6.0, got {t_mid}"
    );
    let a_mid = sched.alpha(50);
    assert!(
        (a_mid - 0.5).abs() < 0.01,
        "mid alpha should be 0.5, got {a_mid}"
    );
}

#[test]
fn test_request_scheduler_fcfs() {
    let mut sched = RequestScheduler::new();
    sched.submit(vec![1.0], 0);
    sched.submit(vec![2.0], 0);
    sched.submit(vec![3.0], 0);
    let first = sched.next().expect("first");
    let second = sched.next().expect("second");
    assert!(first.sequence < second.sequence);
}

#[test]
fn test_error_cases() {
    let w = vec![vec![1.0, 2.0]];
    let g = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
    assert!(MovementPruning::movement_score(&w, &g).is_err());
    let awq = AwqQuantizer::new(0.5);
    assert!(awq.compute_scale(&w, &[1.0]).is_err());
    let sq = SmoothQuant::new(0.5);
    assert!(sq.smooth(&w, &[1.0]).is_err());
    let masks = vec![vec![1.0], vec![0.0]];
    assert!(SparsityReport::from_weights_and_masks(&w, &masks).is_err());
}

#[test]
fn test_fp8_zero_passthrough() {
    let q = Fp8Quantizer::quantize_fp8(&[0.0], Fp8Format::E4M3);
    assert_eq!(q[0], 0.0);
}

// ── Advanced: HyperpriorModel ────────────────────────────────────────────────

#[test]
fn test_hyperprior_model_encode_shape() {
    let model = HyperpriorModel::new(16, 8, 4, 0.01, 1);
    let x = vec![0.5_f64; 16];
    let y = model.encode(&x);
    assert_eq!(y.len(), 8);
    assert!(y.iter().all(|v| v.is_finite()));
}

#[test]
fn test_hyperprior_model_quantize_rounds() {
    let y = vec![0.3_f64, 0.7_f64, -0.4_f64, 1.6_f64];
    let y_hat = HyperpriorModel::quantize(&y);
    assert_eq!(y_hat, vec![0.0, 1.0, 0.0, 2.0]);
}

#[test]
fn test_hyperprior_model_hyper_encode_shape() {
    let model = HyperpriorModel::new(16, 8, 4, 0.01, 2);
    let y = vec![0.5_f64; 8];
    let z = model.hyper_encode(&y);
    assert_eq!(z.len(), 4);
    assert!(z.iter().all(|v| v.is_finite()));
}

#[test]
fn test_hyperprior_model_hyper_decode_shapes() {
    let model = HyperpriorModel::new(16, 8, 4, 0.01, 3);
    let z = vec![1.0_f64; 4];
    let (mean, log_scale) = model.hyper_decode(&z);
    assert_eq!(mean.len(), 8);
    assert_eq!(log_scale.len(), 8);
    assert!(log_scale.iter().all(|&v| (-10.0..=10.0).contains(&v)));
}

#[test]
fn test_hyperprior_model_forward_returns_finite() {
    let model = HyperpriorModel::new(8, 4, 2, 0.01, 4);
    let x = vec![0.1_f64; 8];
    let (x_hat, rate, loss) = model.forward(&x);
    assert_eq!(x_hat.len(), 8);
    assert!(rate.is_finite());
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

#[test]
fn test_hyperprior_model_rd_loss_lambda_effect() {
    // Higher lambda should give higher loss if distortion is fixed
    let m1 = HyperpriorModel::new(8, 4, 2, 0.001, 5);
    let m2 = HyperpriorModel::new(8, 4, 2, 1.0, 5); // same seed, different lambda
    let x = vec![0.1_f64; 8];
    let x_hat = vec![0.2_f64; 8];
    let l1 = m1.rd_loss(&x, &x_hat, 1.0);
    let l2 = m2.rd_loss(&x, &x_hat, 1.0);
    assert!(l2 > l1, "higher lambda should give larger R-D loss");
}

// ── Advanced: ChannelConditionalModel ────────────────────────────────────────

#[test]
fn test_channel_conditional_model_predict_params() {
    let model = ChannelConditionalModel::new(8, 4, 10);
    let latents = vec![0.5_f64; 8];
    for c in 0..8 {
        let (mean, log_scale) = model.predict_params(&latents, c);
        assert!(mean.is_finite());
        assert!((-10.0..=10.0).contains(&log_scale));
    }
}

#[test]
fn test_channel_conditional_model_first_channel_zero_context() {
    // First channel has no previous channels as context
    let model = ChannelConditionalModel::new(6, 4, 11);
    let latents = vec![1.0_f64; 6];
    let (mean, log_scale) = model.predict_params(&latents, 0);
    assert!(mean.is_finite());
    assert!(log_scale.is_finite());
}

#[test]
fn test_channel_conditional_model_total_rate() {
    let model = ChannelConditionalModel::new(8, 3, 12);
    let latents = vec![0.1_f64; 8];
    let rate = model.total_rate(&latents);
    assert!(rate.is_finite());
    // Rate should be positive for non-trivial latents
}

// ── Advanced: RdOptimizer ────────────────────────────────────────────────────

#[test]
fn test_rd_optimizer_rd_curve_shape() {
    let opt = RdOptimizer::new(vec![0.001, 0.01, 0.1, 1.0]);
    let distortions = vec![0.001, 0.005, 0.02, 0.1];
    let rates = vec![2.0, 1.5, 1.0, 0.5];
    let curve = opt.rd_curve(&distortions, &rates);
    assert_eq!(curve.len(), 4);
    for (r, psnr) in &curve {
        assert!(r.is_finite() && *r >= 0.0);
        assert!(psnr.is_finite());
    }
}

#[test]
fn test_rd_optimizer_select_lambda() {
    let mut opt = RdOptimizer::new(vec![0.001, 0.01, 0.1, 1.0]);
    let rates = vec![3.0, 2.0, 1.0, 0.5];
    let idx = opt.select_lambda(2.0, &rates);
    assert_eq!(idx, 1); // rate 2.0 at index 1
}

#[test]
fn test_rd_optimizer_bd_rate_zero_same_curve() {
    let opt = RdOptimizer::new(vec![0.001, 0.01, 0.1]);
    let curve = vec![(0.5, 30.0), (1.0, 35.0), (2.0, 40.0)];
    let bd = opt.bd_rate(&curve, &curve);
    // Same curve → 0% BD-Rate
    assert!(bd.abs() < 1.0, "BD-Rate of same curve vs itself should be ~0, got {bd}");
}

#[test]
fn test_rd_optimizer_bd_rate_returns_finite() {
    let opt = RdOptimizer::new(vec![0.001, 0.01, 0.1, 1.0]);
    let anchor = vec![(0.5, 28.0), (1.0, 33.0), (2.0, 38.0), (4.0, 42.0)];
    let test = vec![(0.4, 28.0), (0.8, 33.0), (1.6, 38.0), (3.2, 42.0)];
    let bd = opt.bd_rate(&anchor, &test);
    assert!(bd.is_finite());
}

// ── Advanced: MagnitudePruner ────────────────────────────────────────────────

#[test]
fn test_magnitude_pruner_target_sparsity() {
    let mut pruner = MagnitudePruner::new(0.5);
    let weights = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.5, 1.5, 2.5, 3.5],
    ];
    let pruned = pruner.prune(weights);
    let sparsity = MagnitudePruner::achieved_sparsity(&pruned);
    assert!((sparsity - 0.5).abs() < 0.15, "sparsity should be ~0.5, got {sparsity}");
}

#[test]
fn test_magnitude_pruner_zero_sparsity() {
    let mut pruner = MagnitudePruner::new(0.0);
    let weights = vec![vec![1.0_f64, 2.0, 3.0]];
    let pruned = pruner.prune(weights.clone());
    let sparsity = MagnitudePruner::achieved_sparsity(&pruned);
    assert_eq!(sparsity, 0.0);
}

#[test]
fn test_magnitude_pruner_mask_from_weights() {
    let pruned = vec![vec![0.0_f64, 1.0, 0.0, 2.0]];
    let mask = MagnitudePruner::mask_from_weights(&pruned);
    assert_eq!(mask, vec![vec![0.0, 1.0, 0.0, 1.0]]);
}

// ── Advanced: MovementPrunerV2 ───────────────────────────────────────────────

#[test]
fn test_movement_pruner_v2_accumulate() {
    let mut pruner = MovementPrunerV2::new(0.5, 2, &[(1, 4), (1, 4)]);
    let w = vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.5, 1.5, 2.5, 3.5]];
    let g = vec![vec![0.1, 0.2, 0.1, 0.2], vec![0.3, 0.1, 0.2, 0.1]];
    assert!(pruner.accumulate(&w, &g).is_ok());
    assert_eq!(pruner.n_steps, 1);
}

#[test]
fn test_movement_pruner_v2_compute_mask() {
    let mut pruner = MovementPrunerV2::new(0.5, 1, &[(1, 8)]);
    let w = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0_f64]];
    let g = vec![vec![0.1; 8]];
    pruner.accumulate(&w, &g).expect("accumulate");
    let mask = pruner.compute_mask();
    assert_eq!(mask.len(), 1);
    assert_eq!(mask[0].len(), 8);
    let kept = mask[0].iter().filter(|&&v| v == 1.0).count();
    assert!((3..=5).contains(&kept), "~50% should be kept, got {kept}/8");
}

#[test]
fn test_movement_pruner_v2_shape_mismatch_error() {
    let mut pruner = MovementPrunerV2::new(0.5, 2, &[(1, 4), (1, 4)]);
    let w = vec![vec![1.0_f64; 4]]; // only 1 layer instead of 2
    let g = vec![vec![0.1_f64; 4]];
    assert!(pruner.accumulate(&w, &g).is_err());
}

// ── Advanced: MixedPrecisionSearch ──────────────────────────────────────────

#[test]
fn test_mixed_precision_search_returns_valid_bits() {
    let search = MixedPrecisionSearch::new(vec![2, 4, 8], 1000, 4, 20);
    let mut rng = StdRng::seed_from_u64(20);
    let layer_sizes = vec![256, 512, 256, 128];
    let assignment = search.search(&layer_sizes, 10, &mut rng);
    assert_eq!(assignment.len(), 4);
    for &b in &assignment {
        assert!(search.allowed_bits.contains(&b), "bit-width {b} not in allowed_bits");
    }
}

#[test]
fn test_mixed_precision_search_compute_sensitivity() {
    let mut search = MixedPrecisionSearch::new(vec![4, 8], 10000, 3, 21);
    let weights = vec![
        vec![1.0_f64; 100],
        vec![0.01_f64; 200],
        vec![10.0_f64; 50],
    ];
    search.compute_sensitivity(&weights);
    assert_eq!(search.sensitivity.len(), 3);
    // Higher norm → higher sensitivity
    assert!(search.sensitivity[2] > search.sensitivity[1]);
}

// ── Advanced: LayerDropper ───────────────────────────────────────────────────

#[test]
fn test_layer_dropper_select_layers_keeps_first_last() {
    let mut dropper = LayerDropper::new(8, 4);
    dropper.scores = vec![0.1, 0.8, 0.5, 0.3, 0.9, 0.2, 0.4, 0.7];
    let kept = dropper.select_layers();
    assert!(kept.contains(&0), "first layer must always be kept");
    assert!(kept.contains(&7), "last layer must always be kept");
    assert_eq!(kept.len(), 4);
}

#[test]
fn test_layer_dropper_compute_importance() {
    let mut dropper = LayerDropper::new(4, 3);
    // Identical adjacent layers → low importance (cos_sim ≈ 1 → importance ≈ 0)
    let outputs = vec![
        vec![1.0_f64, 0.0, 0.0],
        vec![1.0_f64, 0.0, 0.0], // same as previous → ~0 importance
        vec![0.0_f64, 1.0, 0.0], // orthogonal → high importance
        vec![0.0_f64, 1.0, 0.0],
    ];
    dropper.compute_importance(&outputs);
    assert!(dropper.scores[0] < dropper.scores[1], "identical layer should have lower importance");
}

#[test]
fn test_layer_dropper_drop_layers_output_count() {
    let dropper = LayerDropper::new(6, 3);
    let outputs: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64; 4]).collect();
    let kept = dropper.drop_layers(&outputs);
    assert_eq!(kept.len(), 3);
}

// ── Advanced: VocabPruner ────────────────────────────────────────────────────

#[test]
fn test_vocab_pruner_prune_by_frequency() {
    let mut pruner = VocabPruner::new(5.0);
    let vocab_size = 6;
    let embed_dim = 4;
    let embeddings = vec![0.1_f64; vocab_size * embed_dim];
    let frequencies = vec![1.0, 10.0, 2.0, 20.0, 0.5, 8.0]; // indices 1, 3, 5 above threshold
    // With importance=None, imp_ok=true for all tokens (unwrap_or(true)),
    // so all vocab_size tokens are kept
    let (pruned_emb, kept) = pruner.prune(&embeddings, vocab_size, embed_dim, &frequencies, None)
        .expect("prune should succeed");
    assert_eq!(kept.len(), vocab_size);
    assert_eq!(pruned_emb.len(), vocab_size * embed_dim);
    // The high-frequency tokens are still present
    assert!(kept.contains(&1) && kept.contains(&3) && kept.contains(&5));
}

#[test]
fn test_vocab_pruner_with_importance_scores() {
    let mut pruner = VocabPruner::new(100.0); // very high threshold, only importance matters
    let vocab_size = 4;
    let embed_dim = 2;
    let embeddings = vec![0.5_f64; vocab_size * embed_dim];
    let frequencies = vec![1.0; vocab_size]; // all below threshold
    let importance = vec![1.0, 0.0, 1.0, 0.0]; // indices 0, 2 have importance
    let (_, kept) = pruner.prune(&embeddings, vocab_size, embed_dim, &frequencies, Some(&importance))
        .expect("prune should succeed");
    assert!(kept.contains(&0));
    assert!(kept.contains(&2));
    assert!(!kept.contains(&1));
    assert!(!kept.contains(&3));
}

#[test]
fn test_vocab_pruner_coverage() {
    let kept = vec![0, 2, 4];
    let token_counts = vec![10.0, 5.0, 20.0, 3.0, 15.0];
    let coverage = VocabPruner::coverage(&kept, &token_counts);
    // kept tokens: counts[0]+counts[2]+counts[4] = 10+20+15=45 of total 53
    let expected = 45.0 / 53.0;
    assert!((coverage - expected).abs() < 1e-10);
}

#[test]
fn test_vocab_pruner_size_mismatch_error() {
    let mut pruner = VocabPruner::new(1.0);
    let embeddings = vec![0.1_f64; 4 * 2]; // 4 tokens
    let frequencies = vec![1.0; 5]; // wrong size
    assert!(pruner.prune(&embeddings, 4, 2, &frequencies, None).is_err());
}

// ── Advanced: CompMetrics ────────────────────────────────────────────────────

#[test]
fn test_comp_metrics_compression_ratio() {
    let ratio = CompMetrics::compression_ratio(1_000_000, 250_000, 4.0);
    // 1M * 32bits / (250K * 4bits) = 32M / 1M = 32x
    assert!((ratio - 32.0).abs() < 1e-6);
}

#[test]
fn test_comp_metrics_accuracy_degradation() {
    let drop = CompMetrics::accuracy_degradation(0.95, 0.92);
    assert!((drop - 3.0).abs() < 1e-10);
}

#[test]
fn test_comp_metrics_latency_reduction() {
    let red = CompMetrics::latency_reduction(100.0, 40.0);
    assert!((red - 60.0).abs() < 1e-10);
}

#[test]
fn test_comp_metrics_psnr_identical() {
    let x = vec![0.5_f64; 8];
    let psnr = CompMetrics::psnr(&x, &x);
    assert_eq!(psnr, 100.0); // perfect reconstruction
}

#[test]
fn test_comp_metrics_psnr_different() {
    let x = vec![0.0_f64; 8];
    let y = vec![1.0_f64; 8];
    let psnr = CompMetrics::psnr(&x, &y);
    assert!(psnr < 10.0 && psnr.is_finite());
}

#[test]
fn test_comp_metrics_bits_per_dim() {
    let bpd = CompMetrics::bits_per_dim(1.0, 1);
    // nll_nats=1 / (1 * ln2) = 1/0.693 ≈ 1.442
    assert!((bpd - 1.0 / std::f64::consts::LN_2).abs() < 1e-10);
}

#[test]
fn test_comp_metrics_model_size_mb() {
    let size = CompMetrics::model_size_mb(1024 * 1024, 8); // 1M params * 8bits = 1MB
    assert!((size - 1.0).abs() < 1e-10);
}

#[test]
fn test_comp_metrics_sparse_flops() {
    let dense = 1_000_000u64;
    let sparse = CompMetrics::sparse_flops(dense, 0.0);
    assert_eq!(sparse, dense); // 0 sparsity → no reduction
    let sparse_75 = CompMetrics::sparse_flops(dense, 0.75);
    assert!(sparse_75 < dense);
}

#[test]
fn test_comp_metrics_report_format() {
    let report = CompMetrics::report(1_000_000, 500_000, 4, 0.95, 0.92, 100.0, 60.0);
    assert!(report.contains("ratio="));
    assert!(report.contains("acc_drop="));
    assert!(report.contains("lat_reduction="));
}
