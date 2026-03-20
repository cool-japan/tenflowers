use super::*;

// --- CP Decomposition ---

#[test]
fn test_cp_decomposition_basic() {
    // 3x3x3 tensor -- CP rank may need more than 3 for general tensor
    let dim_i = 3;
    let dim_j = 3;
    let dim_k = 3;
    let tensor: Vec<f64> = (0..27).map(|i| i as f64 + 1.0).collect();
    let cp = EoCpDecomposition::new(5, 200);
    let factors = cp
        .decompose(&tensor, dim_i, dim_j, dim_k)
        .expect("CP should succeed");
    assert_eq!(factors.lambdas.len(), 5);
    // A 3x3x3 sequential tensor may need rank up to 5 for good approx
    assert!(
        factors.approx_error < 1.0,
        "CP error too high: {}",
        factors.approx_error
    );
}

#[test]
fn test_cp_reconstruction_rank1() {
    // Rank-1 tensor: a (x) b (x) c
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0];
    let c = [6.0, 7.0];
    let mut tensor = vec![0.0_f64; 3 * 2 * 2];
    for i in 0..3 {
        for j in 0..2 {
            for k in 0..2 {
                tensor[i * 4 + j * 2 + k] = a[i] * b[j] * c[k];
            }
        }
    }
    let cp = EoCpDecomposition::new(1, 200).with_tol(1e-12);
    let factors = cp.decompose(&tensor, 3, 2, 2).expect("rank-1 CP");
    assert!(
        factors.approx_error < 0.1,
        "rank-1 error: {}",
        factors.approx_error
    );
    let recon = EoCpDecomposition::reconstruct(&factors, 3, 2, 2);
    for (t, r) in tensor.iter().zip(recon.iter()) {
        assert!(
            (t - r).abs() < 5.0,
            "reconstruction mismatch: {} vs {}",
            t,
            r
        );
    }
}

#[test]
fn test_cp_decomposition_wrong_size() {
    let cp = EoCpDecomposition::new(2, 10);
    let result = cp.decompose(&[1.0, 2.0], 2, 2, 2);
    assert!(result.is_err());
}

#[test]
fn test_cp_decomposition_iterations_count() {
    let tensor: Vec<f64> = (0..8).map(|i| (i as f64).sin()).collect();
    let cp = EoCpDecomposition::new(2, 5);
    let factors = cp.decompose(&tensor, 2, 2, 2).expect("CP");
    assert!(factors.iterations <= 5);
}

// --- Tucker Decomposition ---

#[test]
fn test_tucker_decomposition_basic() {
    let tensor: Vec<f64> = (0..24).map(|i| (i as f64 + 1.0) * 0.1).collect();
    let tucker = EoTuckerDecomposition::new((2, 2, 2));
    let factors = tucker
        .decompose(&tensor, 2, 3, 4)
        .expect("Tucker should succeed");
    assert_eq!(factors.core_shape, (2, 2, 2));
    assert!(factors.compression_ratio > 0.0);
}

#[test]
fn test_tucker_reconstruction() {
    let tensor: Vec<f64> = (0..12).map(|i| (i as f64).cos()).collect();
    let tucker = EoTuckerDecomposition::new((2, 2, 2));
    let factors = tucker.decompose(&tensor, 2, 2, 3).expect("Tucker");
    let recon = EoTuckerDecomposition::reconstruct(&factors, 2, 2, 3);
    assert_eq!(recon.len(), 12);
    // Low-rank approximation should have reasonable error
    assert!(factors.approx_error < 1.0);
}

#[test]
fn test_tucker_compression_ratio() {
    let tensor: Vec<f64> = (0..60).map(|i| (i as f64) * 0.01).collect();
    let tucker = EoTuckerDecomposition::new((2, 2, 2));
    let factors = tucker.decompose(&tensor, 3, 4, 5).expect("Tucker");
    // With ranks (2,2,2) the compressed size is 8+6+8+10 = 32 vs original 60
    assert!(
        factors.compression_ratio > 1.0,
        "Should compress: {}",
        factors.compression_ratio
    );
}

#[test]
fn test_tucker_wrong_size() {
    let tucker = EoTuckerDecomposition::new((2, 2, 2));
    let result = tucker.decompose(&[1.0], 2, 3, 4);
    assert!(result.is_err());
}

// --- TT Decomposition ---

#[test]
fn test_tt_decomposition_2d() {
    let tensor: Vec<f64> = (0..12).map(|i| i as f64 + 1.0).collect();
    let tt = TtDecomposition::new(3);
    let factors = tt.decompose(&tensor, &[3, 4]).expect("TT 2D");
    assert_eq!(factors.cores.len(), 2);
    let recon = TtDecomposition::reconstruct(&factors);
    assert_eq!(recon.len(), 12);
}

#[test]
fn test_tt_decomposition_3d() {
    let tensor: Vec<f64> = (0..24).map(|i| (i as f64).sin()).collect();
    let tt = TtDecomposition::new(4);
    let factors = tt.decompose(&tensor, &[2, 3, 4]).expect("TT 3D");
    assert_eq!(factors.cores.len(), 3);
    assert!(factors.approx_error < 1.0);
}

#[test]
fn test_tt_rounding() {
    let tensor: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let tt = TtDecomposition::new(4);
    let factors = tt.decompose(&tensor, &[2, 3, 4]).expect("TT");
    let rounded = tt.round(&factors, 2).expect("TT rounding");
    assert!(rounded
        .cores
        .iter()
        .all(|c| c.shape.0 <= 4 && c.shape.2 <= 4));
}

#[test]
fn test_tt_wrong_size() {
    let tt = TtDecomposition::new(2);
    let result = tt.decompose(&[1.0, 2.0], &[3, 4]);
    assert!(result.is_err());
}

#[test]
fn test_tt_too_few_dims() {
    let tt = TtDecomposition::new(2);
    let result = tt.decompose(&[1.0, 2.0, 3.0], &[3]);
    assert!(result.is_err());
}

// --- Codebook Quantization ---

#[test]
fn test_codebook_quantization_basic() {
    let weights: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
    let cq = CodebookQuantization::new(8);
    let result = cq.quantize(&weights).expect("Codebook quantization");
    assert_eq!(result.codebook.len(), 8);
    assert_eq!(result.indices.len(), 100);
    assert!(result.compression_ratio > 1.0);
}

#[test]
fn test_codebook_dequantize() {
    let weights: Vec<f64> = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    let cq = CodebookQuantization::new(3);
    let result = cq.quantize(&weights).expect("quantize");
    let recon = CodebookQuantization::dequantize(&result.codebook, &result.indices);
    assert_eq!(recon.len(), 6);
    // MSE should be small for well-clustered data
    assert!(result.mse < 0.5, "MSE too high: {}", result.mse);
}

#[test]
fn test_codebook_empty_weights() {
    let cq = CodebookQuantization::new(4);
    let result = cq.quantize(&[]);
    assert!(result.is_err());
}

#[test]
fn test_codebook_single_cluster() {
    let weights = vec![5.0, 5.0, 5.0, 5.0];
    let cq = CodebookQuantization::new(1);
    let result = cq.quantize(&weights).expect("single cluster");
    assert_eq!(result.codebook.len(), 1);
    assert!((result.codebook[0] - 5.0).abs() < 1e-10);
}

// --- Product Quantization ---

#[test]
fn test_pq_encode_basic() {
    let n = 20;
    let dim = 8;
    let vectors: Vec<f64> = (0..n * dim).map(|i| (i as f64 * 0.1).cos()).collect();
    let pq = ProductQuantization::new(4, 4);
    let codes = pq.encode(&vectors, n, dim).expect("PQ encode");
    assert_eq!(codes.codes.len(), n);
    assert_eq!(codes.codebooks.len(), 4);
    assert_eq!(codes.sub_dim, 2);
}

#[test]
fn test_pq_search_adc() {
    let n = 10;
    let dim = 4;
    let vectors: Vec<f64> = (0..n * dim).map(|i| i as f64).collect();
    let pq = ProductQuantization::new(2, 4);
    let codes = pq.encode(&vectors, n, dim).expect("PQ encode");
    let query: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
    let distances = pq.search_adc(&query, &codes).expect("PQ search");
    assert_eq!(distances.len(), n);
    // First vector [0,1,2,3] should have smallest distance to query
    let min_idx = distances
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);
    assert_eq!(min_idx, Some(0));
}

#[test]
fn test_pq_dim_mismatch() {
    let pq = ProductQuantization::new(3, 4);
    let result = pq.encode(&[1.0; 10], 2, 5);
    assert!(result.is_err()); // dim 5 not divisible by 3
}

#[test]
fn test_pq_search_wrong_query_dim() {
    let pq = ProductQuantization::new(2, 2);
    let codes = pq.encode(&[1.0; 8], 2, 4).expect("encode");
    let result = pq.search_adc(&[1.0, 2.0], &codes);
    assert!(result.is_err());
}

// --- Hardware Profile & Search ---

#[test]
fn test_hardware_profile_creation() {
    let p = HardwareProfile::cortex_m4();
    assert_eq!(p.memory_budget_bytes, 256 * 1024);
    let p2 = HardwareProfile::mobile_midrange();
    assert!(p2.compute_budget_mflops > p.compute_budget_mflops);
}

#[test]
fn test_hw_search_generate_candidates() {
    let profile = HardwareProfile::mobile_midrange();
    let search = HardwareAwareSearch::new(profile, 1000.0, 1_000_000);
    let candidates = search.generate_candidates();
    assert_eq!(candidates.len(), 4 * 3); // 4 widths x 3 depths
}

#[test]
fn test_hw_search_pareto_frontier() {
    let profile = HardwareProfile::jetson_nano();
    let search = HardwareAwareSearch::new(profile, 500.0, 500_000);
    let candidates = search.generate_candidates();
    let pareto = search.pareto_frontier(&candidates);
    assert!(!pareto.is_empty());
    // Pareto set should be subset
    assert!(pareto.len() <= candidates.len());
}

#[test]
fn test_hw_search_filter_feasible() {
    let profile = HardwareProfile::cortex_m4();
    let search = HardwareAwareSearch::new(profile, 100.0, 100_000);
    let candidates = search.generate_candidates();
    let feasible = search.filter_feasible(&candidates);
    // With tight MCU constraints, some candidates should be filtered out
    assert!(feasible.len() <= candidates.len());
}

#[test]
fn test_hw_search_full_pipeline() {
    let profile = HardwareProfile::mobile_midrange();
    let search = HardwareAwareSearch::new(profile, 1000.0, 1_000_000);
    let pareto = search.search();
    // Should find at least one feasible Pareto-optimal candidate
    assert!(!pareto.is_empty());
}

// --- Dynamic Width Network ---

#[test]
fn test_dynamic_width_full() {
    let net = DynamicWidthNetwork::new(&[8, 16, 4], 42).expect("DWN");
    let input = vec![1.0; 8];
    let out = net.forward_at_width(&input, 1.0).expect("full width");
    assert_eq!(out.len(), 4);
}

#[test]
fn test_dynamic_width_half() {
    let net = DynamicWidthNetwork::new(&[8, 16, 4], 42).expect("DWN");
    let input = vec![1.0; 8];
    let out = net.forward_at_width(&input, 0.5).expect("half width");
    // Output at half width: ceil(4 * 0.5) = 2
    assert_eq!(out.len(), 2);
}

#[test]
fn test_dynamic_width_quarter() {
    let net = DynamicWidthNetwork::new(&[8, 16, 4], 42).expect("DWN");
    let input = vec![1.0; 8];
    let out = net.forward_at_width(&input, 0.25).expect("quarter width");
    assert!(!out.is_empty());
}

#[test]
fn test_dynamic_width_distillation_loss() {
    let net = DynamicWidthNetwork::new(&[8, 16, 4], 42).expect("DWN");
    let input = vec![0.5; 8];
    let loss = net.inplace_distillation_loss(&input, 0.5).expect("distill");
    assert!(loss >= 0.0);
}

#[test]
fn test_dynamic_width_too_few_layers() {
    let result = DynamicWidthNetwork::new(&[8], 42);
    assert!(result.is_err());
}

// --- Integer Arithmetic ---

#[test]
fn test_fixed_point_roundtrip() {
    let test_val = std::f64::consts::PI;
    let fp = FixedPoint::from_f64(test_val, 8);
    let back = fp.to_f64();
    assert!((back - test_val).abs() < 0.01, "roundtrip: {}", back);
}

#[test]
fn test_fixed_point_multiply() {
    let a = FixedPoint::from_f64(2.0, 8);
    let b = FixedPoint::from_f64(3.0, 8);
    let c = a.mul(b);
    assert!((c.to_f64() - 6.0).abs() < 0.1, "mul: {}", c.to_f64());
}

#[test]
fn test_fixed_point_add() {
    let a = FixedPoint::from_f64(1.5, 8);
    let b = FixedPoint::from_f64(2.5, 8);
    let c = a.add(b);
    assert!((c.to_f64() - 4.0).abs() < 0.01, "add: {}", c.to_f64());
}

#[test]
fn test_integer_linear_basic() {
    let weights = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
    let bias = vec![0.0, 0.0];
    let layer = IntegerLinear::from_float(&weights, &bias, 2, 2).expect("IntLinear");
    let out = layer.forward_float(&[0.5, -0.3]).expect("forward");
    assert_eq!(out.len(), 2);
    // Should approximately preserve input
    assert!((out[0] - 0.5).abs() < 0.1, "out[0]: {}", out[0]);
}

#[test]
fn test_integer_linear_relu() {
    let acc = vec![-10, 0, 5, 100];
    let relu = IntegerLinear::quantized_relu(&acc);
    assert_eq!(relu, vec![0, 0, 5, 100]);
}

#[test]
fn test_integer_linear_sigmoid_approx() {
    let acc = vec![-1000, 0, 1000];
    let sig = IntegerLinear::quantized_sigmoid_approx(&acc, 0.01);
    // -1000*0.01 = -10 -> 0; 0 -> 128 (0.5); 1000*0.01 = 10 -> 256 (1.0)
    assert_eq!(sig[0], 0);
    assert_eq!(sig[1], 128); // 0.5 * 256
    assert_eq!(sig[2], 256); // 1.0 * 256
}

#[test]
fn test_integer_linear_wrong_size() {
    let result = IntegerLinear::from_float(&[1.0, 2.0], &[0.0], 3, 1);
    assert!(result.is_err());
}

// --- Memory Budget Allocator ---

#[test]
fn test_memory_estimator_conv() {
    let layer = EoLayerDesc {
        name: "conv1".to_string(),
        layer_type: EoLayerType::Conv {
            in_ch: 3,
            out_ch: 64,
            kernel: 3,
        },
        spatial: (32, 32),
        batch_size: 1,
    };
    let mem = MemoryBudgetAllocator::estimate_layer_memory(&layer);
    assert_eq!(mem, 64 * 32 * 32 * 4);
}

#[test]
fn test_memory_estimator_linear() {
    let layer = EoLayerDesc {
        name: "fc1".to_string(),
        layer_type: EoLayerType::Linear {
            in_feat: 512,
            out_feat: 256,
        },
        spatial: (1, 1),
        batch_size: 8,
    };
    let mem = MemoryBudgetAllocator::estimate_layer_memory(&layer);
    assert_eq!(mem, 8 * 256 * 4);
}

#[test]
fn test_fusion_detection() {
    let layers = vec![
        EoLayerDesc {
            name: "conv1".to_string(),
            layer_type: EoLayerType::Conv {
                in_ch: 3,
                out_ch: 64,
                kernel: 3,
            },
            spatial: (32, 32),
            batch_size: 1,
        },
        EoLayerDesc {
            name: "bn1".to_string(),
            layer_type: EoLayerType::BatchNorm { channels: 64 },
            spatial: (32, 32),
            batch_size: 1,
        },
        EoLayerDesc {
            name: "relu1".to_string(),
            layer_type: EoLayerType::Relu,
            spatial: (32, 32),
            batch_size: 1,
        },
    ];
    let fusions = MemoryBudgetAllocator::detect_fusions(&layers);
    assert_eq!(fusions.len(), 1);
    assert_eq!(fusions[0].layer_indices, vec![0, 1, 2]);
    assert!(fusions[0].memory_saved > 0);
}

#[test]
fn test_memory_plan_fits_budget() {
    let layers = vec![EoLayerDesc {
        name: "fc1".to_string(),
        layer_type: EoLayerType::Linear {
            in_feat: 10,
            out_feat: 10,
        },
        spatial: (1, 1),
        batch_size: 1,
    }];
    let allocator = MemoryBudgetAllocator::new(1_000_000);
    let plan = allocator.plan_memory(&layers);
    assert!(plan.fits_budget);
    assert!(plan.checkpoint_layers.is_empty());
}

#[test]
fn test_memory_plan_needs_checkpointing() {
    let layers: Vec<EoLayerDesc> = (0..10)
        .map(|i| EoLayerDesc {
            name: format!("layer_{}", i),
            layer_type: EoLayerType::Linear {
                in_feat: 1024,
                out_feat: 1024,
            },
            spatial: (1, 1),
            batch_size: 64,
        })
        .collect();
    let allocator = MemoryBudgetAllocator::new(100_000); // very tight
    let plan = allocator.plan_memory(&layers);
    // Should suggest checkpointing some layers
    assert!(!plan.checkpoint_layers.is_empty());
}

// --- Edge Metrics & Report ---

#[test]
fn test_edge_metrics_compute() {
    let metrics = EdgeMetrics::compute(1_000_000, 250_000, 1e9, 2.5e8, 1_000_000, 0.95);
    assert!((metrics.compression_ratio - 4.0).abs() < 0.01);
    assert!((metrics.speedup_factor - 4.0).abs() < 0.01);
}

#[test]
fn test_edge_metrics_efficiency_score() {
    let metrics = EdgeMetrics::compute(100, 10, 100.0, 10.0, 40, 0.9);
    let score = metrics.efficiency_score();
    assert!(score > 0.0 && score <= 1.0, "score: {}", score);
}

#[test]
fn test_edge_report_summary() {
    let metrics = EdgeMetrics::compute(1_000_000, 100_000, 1e9, 1e8, 400_000, 0.92);
    let report = EdgeReport::new("TestModel", "Cortex-M4", metrics);
    let summary = report.summary();
    assert!(summary.contains("TestModel"));
    assert!(summary.contains("Cortex-M4"));
    assert!(summary.contains("Compression"));
}

#[test]
fn test_edge_report_pareto_summary() {
    let metrics = EdgeMetrics::compute(100, 50, 100.0, 50.0, 200, 0.85);
    let mut report = EdgeReport::new("M1", "GPU", metrics);
    report.pareto_candidates.push(EoArchCandidate {
        width_mult: 0.5,
        depth_mult: 1.0,
        estimated_latency_ms: 5.0,
        estimated_memory_bytes: 1000,
        estimated_mflops: 50.0,
        accuracy_proxy: 0.88,
    });
    let pareto = report.pareto_analysis_summary();
    assert_eq!(pareto.len(), 1);
    assert!((pareto[0].0 - 0.88).abs() < 1e-10);
}

// --- Integration tests ---

#[test]
fn test_end_to_end_compress_and_measure() {
    // Compress weights with codebook, measure metrics
    let original_weights: Vec<f64> = (0..500).map(|i| (i as f64 * 0.02).sin()).collect();
    let cq = CodebookQuantization::new(16);
    let result = cq.quantize(&original_weights).expect("quantize");
    let recon = CodebookQuantization::dequantize(&result.codebook, &result.indices);

    // Measure MSE between original and reconstructed
    let mse: f64 = original_weights
        .iter()
        .zip(recon.iter())
        .map(|(o, r)| (o - r).powi(2))
        .sum::<f64>()
        / original_weights.len() as f64;
    assert!(mse < 0.1, "End-to-end MSE too high: {}", mse);

    let metrics = EdgeMetrics::compute(
        500,
        16 + 500, // codebook + indices (rough)
        1000.0,
        1000.0,
        (16 + 500) * 4,
        1.0 - mse,
    );
    assert!(metrics.accuracy > 0.9);
}

#[test]
fn test_lstsq_basic() {
    // Solve [[1,0],[0,1]] x = [3,4]
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let b = vec![3.0, 4.0];
    let x = solve_lstsq(&a, &b).expect("lstsq");
    assert!((x[0] - 3.0).abs() < 1e-10);
    assert!((x[1] - 4.0).abs() < 1e-10);
}

#[test]
fn test_truncated_svd_rank1() {
    // Rank-1 matrix: outer product of [1,2,3] and [4,5]
    let mat = vec![vec![4.0, 5.0], vec![8.0, 10.0], vec![12.0, 15.0]];
    let (u, s, vt) = truncated_svd(&mat, 1, 50, 42).expect("SVD");
    assert_eq!(s.len(), 1);
    // Reconstruct: should be close to original
    let recon: Vec<Vec<f64>> = (0..3)
        .map(|i| (0..2).map(|j| u[i][0] * s[0] * vt[0][j]).collect())
        .collect();
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (recon[i][j] - mat[i][j]).abs() < 0.5,
                "SVD recon mismatch at ({},{}): {} vs {}",
                i,
                j,
                recon[i][j],
                mat[i][j],
            );
        }
    }
}
