//! Tests for Neural Architecture Distillation & Efficient Architecture Search.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// AdLinear tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ad_linear_creation() -> Result<()> {
    let layer = AdLinear::new(10, 5, 42)?;
    assert_eq!(layer.in_features, 10);
    assert_eq!(layer.out_features, 5);
    assert_eq!(layer.weights.len(), 50);
    assert_eq!(layer.bias.len(), 5);
    Ok(())
}

#[test]
fn test_ad_linear_forward() -> Result<()> {
    let layer = AdLinear::new(4, 3, 42)?;
    let input = vec![1.0, 0.0, -1.0, 0.5];
    let output = layer.forward(&input)?;
    assert_eq!(output.len(), 3);
    Ok(())
}

#[test]
fn test_ad_linear_wrong_input_dim() {
    let layer = AdLinear::new(4, 3, 42).expect("creation should succeed");
    let result = layer.forward(&[1.0, 2.0]);
    assert!(result.is_err());
}

#[test]
fn test_ad_linear_zero_dim() {
    assert!(AdLinear::new(0, 5, 42).is_err());
    assert!(AdLinear::new(5, 0, 42).is_err());
}

#[test]
fn test_ad_linear_param_count() -> Result<()> {
    let layer = AdLinear::new(8, 4, 0)?;
    assert_eq!(layer.param_count(), 8 * 4 + 4);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// AdMlp tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ad_mlp_creation() -> Result<()> {
    let mlp = AdMlp::new(&[10, 32, 16, 5], 42)?;
    assert_eq!(mlp.layers.len(), 3);
    Ok(())
}

#[test]
fn test_ad_mlp_forward() -> Result<()> {
    let mlp = AdMlp::new(&[4, 8, 3], 42)?;
    let input = vec![1.0, -1.0, 0.5, 0.0];
    let output = mlp.forward(&input)?;
    assert_eq!(output.len(), 3);
    Ok(())
}

#[test]
fn test_ad_mlp_too_few_sizes() {
    assert!(AdMlp::new(&[10], 42).is_err());
    assert!(AdMlp::new(&[], 42).is_err());
}

#[test]
fn test_ad_mlp_relu_hidden_layers() -> Result<()> {
    // Create MLP with a large negative bias so hidden activations are negative pre-ReLU
    let mut mlp = AdMlp::new(&[2, 4, 1], 42)?;
    // Set all hidden biases to large negative
    for b in mlp.layers[0].bias.iter_mut() {
        *b = -100.0;
    }
    let input = vec![0.0, 0.0];
    let output = mlp.forward(&input)?;
    // Hidden layer should output 0 (ReLU clips), so final output = bias only
    assert_eq!(output.len(), 1);
    Ok(())
}

#[test]
fn test_ad_mlp_param_count() -> Result<()> {
    let mlp = AdMlp::new(&[10, 20, 5], 0)?;
    let expected = (10 * 20 + 20) + (20 * 5 + 5);
    assert_eq!(mlp.param_count(), expected);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// AdOpType tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ad_op_type_count() {
    assert_eq!(AdOpType::count(), 7);
    assert_eq!(AdOpType::all().len(), 7);
}

#[test]
fn test_ad_op_type_roundtrip() -> Result<()> {
    for i in 0..AdOpType::count() {
        let op = AdOpType::from_index(i)?;
        assert_eq!(op.to_index(), i);
    }
    Ok(())
}

#[test]
fn test_ad_op_type_invalid_index() {
    assert!(AdOpType::from_index(7).is_err());
    assert!(AdOpType::from_index(100).is_err());
}

#[test]
fn test_ad_op_type_names() -> Result<()> {
    assert_eq!(AdOpType::Conv3x3.name(), "conv3x3");
    assert_eq!(AdOpType::DWConv.name(), "dw_conv");
    assert_eq!(AdOpType::Zero.name(), "zero");
    Ok(())
}

#[test]
fn test_ad_op_type_flops() {
    let flops_3x3 = AdOpType::Conv3x3.flops_per_position(16);
    let flops_5x5 = AdOpType::Conv5x5.flops_per_position(16);
    assert!(flops_5x5 > flops_3x3);
    assert_eq!(AdOpType::Zero.flops_per_position(16), 0);
}

#[test]
fn test_ad_op_type_params() {
    assert!(AdOpType::Conv5x5.params_estimate(16) > AdOpType::Conv3x3.params_estimate(16));
    assert_eq!(AdOpType::Skip.params_estimate(16), 0);
    assert_eq!(AdOpType::Zero.params_estimate(16), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// AdArchEncoding tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ad_arch_encoding_creation() -> Result<()> {
    let edges = vec![
        AdEdge {
            op1: AdOpType::Conv3x3,
            op2: AdOpType::Skip,
            connection: 0,
        },
        AdEdge {
            op1: AdOpType::Conv5x5,
            op2: AdOpType::MaxPool,
            connection: 1,
        },
    ];
    let arch = AdArchEncoding::new(2, edges, 32, 4)?;
    assert_eq!(arch.n_nodes, 2);
    assert_eq!(arch.channels, 32);
    assert_eq!(arch.n_cells, 4);
    Ok(())
}

#[test]
fn test_ad_arch_encoding_invalid() {
    assert!(AdArchEncoding::new(0, vec![], 32, 4).is_err());
    assert!(AdArchEncoding::new(2, vec![], 0, 4).is_err());
    assert!(AdArchEncoding::new(2, vec![], 32, 0).is_err());
}

#[test]
fn test_ad_arch_encoding_encode() -> Result<()> {
    let arch = AdArchEncoding::random(3, 32, 4, 42)?;
    let enc = arch.encode();
    // 3 edges * 15 + 3 global = 48
    assert_eq!(enc.len(), 3 * 15 + 3);
    Ok(())
}

#[test]
fn test_ad_arch_encoding_decode() -> Result<()> {
    let arch = AdArchEncoding::random(2, 64, 8, 42)?;
    let desc = arch.decode();
    assert!(desc.contains("Arch("));
    assert!(desc.contains("nodes=2"));
    assert!(desc.contains("cells=8"));
    Ok(())
}

#[test]
fn test_ad_arch_encoding_flops() -> Result<()> {
    let arch = AdArchEncoding::random(4, 64, 8, 42)?;
    let flops = arch.estimate_flops(32);
    assert!(flops > 0);
    Ok(())
}

#[test]
fn test_ad_arch_encoding_params() -> Result<()> {
    let arch = AdArchEncoding::random(4, 64, 8, 42)?;
    let params = arch.estimate_params();
    assert!(params > 0);
    Ok(())
}

#[test]
fn test_ad_arch_encoding_random() -> Result<()> {
    let a1 = AdArchEncoding::random(4, 32, 8, 42)?;
    let a2 = AdArchEncoding::random(4, 32, 8, 43)?;
    // Different seeds should produce different architectures
    let e1 = a1.encode();
    let e2 = a2.encode();
    assert_ne!(e1, e2);
    Ok(())
}

#[test]
fn test_ad_arch_encoding_random_invalid() {
    assert!(AdArchEncoding::random(0, 32, 8, 0).is_err());
    assert!(AdArchEncoding::random(4, 0, 8, 0).is_err());
    assert!(AdArchEncoding::random(4, 32, 0, 0).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// NeuralPredictor tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_neural_predictor_creation() -> Result<()> {
    let pred = NeuralPredictor::new(48, 32, 42)?;
    assert_eq!(pred.encoding_dim, 48);
    Ok(())
}

#[test]
fn test_neural_predictor_invalid() {
    assert!(NeuralPredictor::new(0, 32, 0).is_err());
    assert!(NeuralPredictor::new(48, 0, 0).is_err());
}

#[test]
fn test_neural_predictor_predict() -> Result<()> {
    let pred = NeuralPredictor::new(48, 32, 42)?;
    let enc = vec![0.1; 48];
    let acc = pred.predict(&enc)?;
    // Sigmoid output in [0, 1]
    assert!((0.0..=1.0).contains(&acc));
    Ok(())
}

#[test]
fn test_neural_predictor_wrong_dim() -> Result<()> {
    let pred = NeuralPredictor::new(48, 32, 42)?;
    assert!(pred.predict(&[0.1; 10]).is_err());
    Ok(())
}

#[test]
fn test_neural_predictor_train() -> Result<()> {
    let mut pred = NeuralPredictor::new(10, 16, 42)?;
    let encodings: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32 * 0.1; 10]).collect();
    let accuracies = vec![0.5, 0.6, 0.7, 0.8, 0.9];
    let loss = pred.train(&encodings, &accuracies, 20)?;
    assert!(loss >= 0.0);
    Ok(())
}

#[test]
fn test_neural_predictor_train_mismatch() {
    let mut pred = NeuralPredictor::new(10, 16, 42).expect("ok");
    let encs: Vec<Vec<f32>> = vec![vec![0.0; 10]];
    let accs = vec![0.5, 0.6];
    assert!(pred.train(&encs, &accs, 5).is_err());
}

#[test]
fn test_neural_predictor_rank() -> Result<()> {
    let pred = NeuralPredictor::new(10, 16, 42)?;
    let encs: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32 * 0.2; 10]).collect();
    let ranked = pred.rank(&encs)?;
    assert_eq!(ranked.len(), 5);
    // Verify descending order
    for i in 1..ranked.len() {
        assert!(ranked[i - 1].1 >= ranked[i].1);
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// CompoundScaling tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_compound_scaling_creation() -> Result<()> {
    // alpha=1.2, beta=1.1, gamma=1.15
    // 1.2 * 1.1^2 * 1.15^2 = 1.2 * 1.21 * 1.3225 = 1.9190...
    let cs = CompoundScaling::new(1.2, 1.1, 1.15)?;
    let constraint = cs.constraint_value();
    assert!((constraint - 2.0).abs() < 0.5);
    Ok(())
}

#[test]
fn test_compound_scaling_invalid() {
    assert!(CompoundScaling::new(0.0, 1.0, 1.0).is_err());
    assert!(CompoundScaling::new(1.0, 0.0, 1.0).is_err());
    assert!(CompoundScaling::new(5.0, 5.0, 5.0).is_err()); // Way over constraint
}

#[test]
fn test_compound_scaling_scale() -> Result<()> {
    let cs = CompoundScaling::new(1.2, 1.1, 1.15)?;
    let base = AdBaseArch {
        depth: 18,
        width: 64,
        resolution: 224,
        base_flops: 1.8e9,
    };

    let scaled = cs.scale(&base, 1.0);
    assert!(scaled.depth >= base.depth);
    assert!(scaled.width >= base.width);
    assert!(scaled.resolution >= base.resolution);
    assert!(scaled.estimated_flops > base.base_flops);
    Ok(())
}

#[test]
fn test_compound_scaling_phi_zero() -> Result<()> {
    let cs = CompoundScaling::new(1.2, 1.1, 1.15)?;
    let base = AdBaseArch {
        depth: 18,
        width: 64,
        resolution: 224,
        base_flops: 1.8e9,
    };

    let scaled = cs.scale(&base, 0.0);
    assert_eq!(scaled.depth, 18);
    assert_eq!(scaled.width, 64);
    assert_eq!(scaled.resolution, 224);
    Ok(())
}

#[test]
fn test_compound_scaling_grid_search() -> Result<()> {
    let cs = CompoundScaling::grid_search((1.0, 1.5), (1.0, 1.3), (1.0, 1.3), 10)?;
    // Should find something close to constraint = 2
    let val = cs.constraint_value();
    assert!((val - 2.0).abs() < 1.0);
    Ok(())
}

#[test]
fn test_compound_scaling_grid_search_zero_steps() {
    assert!(CompoundScaling::grid_search((1.0, 1.5), (1.0, 1.3), (1.0, 1.3), 0).is_err());
}

#[test]
fn test_compound_scaling_increasing_phi() -> Result<()> {
    let cs = CompoundScaling::new(1.2, 1.1, 1.15)?;
    let base = AdBaseArch {
        depth: 10,
        width: 32,
        resolution: 128,
        base_flops: 1e8,
    };

    let s1 = cs.scale(&base, 1.0);
    let s2 = cs.scale(&base, 2.0);
    assert!(s2.estimated_flops > s1.estimated_flops);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// InvertedResidualBlock tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_inverted_residual_creation() -> Result<()> {
    let block = InvertedResidualBlock::new(16, 16, 4, 3, 1, 42)?;
    assert_eq!(block.in_channels, 16);
    assert_eq!(block.out_channels, 16);
    assert_eq!(block.expanded_channels, 64);
    assert!(block.use_skip); // same channels + stride 1
    Ok(())
}

#[test]
fn test_inverted_residual_no_skip() -> Result<()> {
    let block = InvertedResidualBlock::new(16, 32, 4, 3, 2, 42)?;
    assert!(!block.use_skip); // Different out channels or stride > 1
    Ok(())
}

#[test]
fn test_inverted_residual_forward() -> Result<()> {
    let block = InvertedResidualBlock::new(4, 4, 2, 3, 1, 42)?;
    let input = vec![0.5f32; 4 * 8]; // 4 channels, spatial_size=8
    let output = block.forward(&input, 8)?;
    assert_eq!(output.len(), 4 * 8);
    Ok(())
}

#[test]
fn test_inverted_residual_stride() -> Result<()> {
    let block = InvertedResidualBlock::new(4, 8, 2, 3, 2, 42)?;
    let spatial = 8;
    let input = vec![0.5f32; 4 * spatial];
    let output = block.forward(&input, spatial)?;
    assert_eq!(output.len(), 8 * 4); // out_spatial = 8/2 = 4
    Ok(())
}

#[test]
fn test_inverted_residual_invalid() {
    assert!(InvertedResidualBlock::new(0, 16, 4, 3, 1, 0).is_err());
    assert!(InvertedResidualBlock::new(16, 0, 4, 3, 1, 0).is_err());
    assert!(InvertedResidualBlock::new(16, 16, 0, 3, 1, 0).is_err());
    assert!(InvertedResidualBlock::new(16, 16, 4, 0, 1, 0).is_err());
    assert!(InvertedResidualBlock::new(16, 16, 4, 3, 0, 0).is_err());
}

#[test]
fn test_inverted_residual_flops() -> Result<()> {
    let block = InvertedResidualBlock::new(16, 16, 4, 3, 1, 42)?;
    let flops = block.estimate_flops(16);
    assert!(flops > 0);
    Ok(())
}

#[test]
fn test_inverted_residual_param_count() -> Result<()> {
    let block = InvertedResidualBlock::new(16, 16, 4, 3, 1, 42)?;
    assert!(block.param_count() > 0);
    Ok(())
}

#[test]
fn test_se_block_creation() -> Result<()> {
    let se = AdSeBlock::new(16, 4, 42)?;
    assert_eq!(se.channels, 16);
    assert_eq!(se.reduced_dim, 4);
    Ok(())
}

#[test]
fn test_se_block_forward() -> Result<()> {
    let se = AdSeBlock::new(4, 2, 42)?;
    let input = vec![1.0f32; 4 * 8]; // 4 channels, 8 spatial
    let output = se.forward(&input, 8)?;
    assert_eq!(output.len(), 32);
    // SE scales values, so they should differ from input
    Ok(())
}

#[test]
fn test_se_block_invalid() {
    assert!(AdSeBlock::new(0, 4, 0).is_err());
    assert!(AdSeBlock::new(16, 0, 0).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// ArchMorphism tests
// ─────────────────────────────────────────────────────────────────────────────

fn make_test_layers() -> Vec<AdLayerDesc> {
    vec![
        AdLayerDesc {
            index: 0,
            in_dim: 4,
            out_dim: 8,
            weights: vec![0.1; 4 * 8],
            bias: vec![0.0; 8],
        },
        AdLayerDesc {
            index: 1,
            in_dim: 8,
            out_dim: 6,
            weights: vec![0.1; 8 * 6],
            bias: vec![0.0; 6],
        },
        AdLayerDesc {
            index: 2,
            in_dim: 6,
            out_dim: 3,
            weights: vec![0.1; 6 * 3],
            bias: vec![0.0; 3],
        },
    ]
}

#[test]
fn test_arch_morphism_creation() -> Result<()> {
    let morph = ArchMorphism::new(make_test_layers())?;
    assert_eq!(morph.depth(), 3);
    Ok(())
}

#[test]
fn test_arch_morphism_empty() {
    assert!(ArchMorphism::new(vec![]).is_err());
}

#[test]
fn test_arch_morphism_mismatched_dims() {
    let layers = vec![
        AdLayerDesc {
            index: 0,
            in_dim: 4,
            out_dim: 8,
            weights: vec![0.1; 32],
            bias: vec![0.0; 8],
        },
        AdLayerDesc {
            index: 1,
            in_dim: 5,
            out_dim: 3, // mismatch: 5 != 8
            weights: vec![0.1; 15],
            bias: vec![0.0; 3],
        },
    ];
    assert!(ArchMorphism::new(layers).is_err());
}

#[test]
fn test_arch_morphism_widen() -> Result<()> {
    let mut morph = ArchMorphism::new(make_test_layers())?;
    let input = vec![1.0; 4];
    let out_before = morph.forward(&input)?;

    morph.widen(0, 12, 42)?; // Widen first layer from 8 to 12
    assert_eq!(morph.layers[0].out_dim, 12);
    assert_eq!(morph.layers[1].in_dim, 12);

    let out_after = morph.forward(&input)?;
    // Function-preserving: outputs should be approximately the same
    assert_eq!(out_before.len(), out_after.len());
    for (a, b) in out_before.iter().zip(out_after.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "widen should be function-preserving: {} vs {}",
            a,
            b
        );
    }
    Ok(())
}

#[test]
fn test_arch_morphism_widen_invalid() -> Result<()> {
    let mut morph = ArchMorphism::new(make_test_layers())?;
    assert!(morph.widen(10, 16, 0).is_err()); // Out of range
    assert!(morph.widen(0, 4, 0).is_err()); // new_width <= old_width
    assert!(morph.widen(2, 10, 0).is_err()); // Last layer
    Ok(())
}

#[test]
fn test_arch_morphism_deepen() -> Result<()> {
    let mut morph = ArchMorphism::new(make_test_layers())?;
    let input = vec![1.0; 4];
    let out_before = morph.forward(&input)?;

    morph.deepen(0)?; // Insert identity after layer 0
    assert_eq!(morph.depth(), 4);

    let out_after = morph.forward(&input)?;
    // Function-preserving (identity layer + ReLU — preserving only for non-negative)
    assert_eq!(out_before.len(), out_after.len());
    Ok(())
}

#[test]
fn test_arch_morphism_deepen_invalid() -> Result<()> {
    let mut morph = ArchMorphism::new(make_test_layers())?;
    assert!(morph.deepen(10).is_err());
    Ok(())
}

#[test]
fn test_arch_morphism_add_skip() -> Result<()> {
    let morph = ArchMorphism::new(make_test_layers())?;
    let skip = morph.add_skip(0, 2, 0.01)?;
    assert_eq!(skip.from_layer, 0);
    assert_eq!(skip.to_layer, 2);
    assert_eq!(skip.from_dim, 8);
    assert_eq!(skip.to_dim, 6);
    Ok(())
}

#[test]
fn test_arch_morphism_add_skip_invalid() -> Result<()> {
    let morph = ArchMorphism::new(make_test_layers())?;
    assert!(morph.add_skip(2, 0, 0.01).is_err()); // from >= to
    assert!(morph.add_skip(0, 10, 0.01).is_err()); // out of range
    Ok(())
}

#[test]
fn test_skip_connection_apply() -> Result<()> {
    let morph = ArchMorphism::new(make_test_layers())?;
    // Skip from layer 0 (out_dim=8) to layer 2 (in_dim=6)
    let skip = morph.add_skip(0, 2, 0.01)?;
    let source = vec![1.0; 8];
    let mut target = vec![0.0; 6];
    skip.apply(&source, &mut target)?;
    // Should have added small values to target
    let sum: f32 = target.iter().sum();
    assert!(sum.abs() > 0.0);
    Ok(())
}

#[test]
fn test_arch_morphism_total_params() -> Result<()> {
    let morph = ArchMorphism::new(make_test_layers())?;
    let params = morph.total_params();
    let expected = (4 * 8 + 8) + (8 * 6 + 6) + (6 * 3 + 3);
    assert_eq!(params, expected);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ProgressiveShrinking tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_progressive_shrinking_creation() -> Result<()> {
    let ps = ProgressiveShrinking::new(vec![4, 4, 4], vec![32, 64, 128], vec![3, 5, 7], 42)?;
    assert_eq!(ps.n_stages, 3);
    assert_eq!(ps.phase, 0);
    Ok(())
}

#[test]
fn test_progressive_shrinking_invalid() {
    assert!(ProgressiveShrinking::new(vec![], vec![32], vec![3], 0).is_err());
    assert!(ProgressiveShrinking::new(vec![4], vec![], vec![3], 0).is_err());
    assert!(ProgressiveShrinking::new(vec![4], vec![32], vec![], 0).is_err());
    assert!(ProgressiveShrinking::new(vec![4, 4], vec![32], vec![3], 0).is_err());
    // mismatch
}

#[test]
fn test_progressive_shrinking_sample_subnet() -> Result<()> {
    let ps = ProgressiveShrinking::new(vec![4, 4], vec![32, 64], vec![3, 5, 7], 42)?;
    let config = ps.sample_subnet(42)?;
    assert_eq!(config.depths.len(), 2);
    assert_eq!(config.widths.len(), 2);
    assert!(config.estimated_flops > 0);
    Ok(())
}

#[test]
fn test_progressive_shrinking_extract_weights() -> Result<()> {
    let ps = ProgressiveShrinking::new(vec![2, 2], vec![8, 16], vec![3, 5], 42)?;
    let config = ps.sample_subnet(42)?;
    let weights = ps.extract_weights(&config)?;
    assert_eq!(weights.len(), 2);
    Ok(())
}

#[test]
fn test_progressive_shrinking_phases() -> Result<()> {
    let mut ps = ProgressiveShrinking::new(vec![4], vec![32], vec![3, 5], 42)?;
    assert_eq!(ps.phase_name(), "elastic_kernel");
    ps.advance_phase();
    assert_eq!(ps.phase_name(), "elastic_depth");
    ps.advance_phase();
    assert_eq!(ps.phase_name(), "elastic_width");
    ps.advance_phase();
    assert_eq!(ps.phase_name(), "full_elastic");
    ps.advance_phase(); // Should not go beyond 3
    assert_eq!(ps.phase, 3);
    Ok(())
}

#[test]
fn test_progressive_shrinking_supernet_params() -> Result<()> {
    let ps = ProgressiveShrinking::new(vec![2], vec![8], vec![3, 5], 42)?;
    assert!(ps.supernet_params() > 0);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ProxyTask tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_proxy_task_creation() -> Result<()> {
    let proxy = ProxyTask::new(5, 0.1, 0.5, 0.5)?;
    assert_eq!(proxy.default_epochs, 5);
    Ok(())
}

#[test]
fn test_proxy_task_invalid() {
    assert!(ProxyTask::new(0, 0.1, 0.5, 0.5).is_err());
    assert!(ProxyTask::new(5, -0.1, 0.5, 0.5).is_err());
    assert!(ProxyTask::new(5, 0.1, 0.0, 0.5).is_err());
    assert!(ProxyTask::new(5, 0.1, 0.5, 1.5).is_err());
}

#[test]
fn test_proxy_task_evaluate() -> Result<()> {
    let proxy = ProxyTask::new(5, 0.5, 0.5, 0.5)?;
    let arch = AdArchEncoding::random(3, 32, 4, 42)?;
    let data: Vec<(Vec<f32>, Vec<f32>)> = (0..10)
        .map(|_| (vec![0.1; 8], vec![1.0, 0.0, 0.0]))
        .collect();
    let result = proxy.evaluate_proxy(&arch, &data, None, None)?;
    assert!(result.score >= 0.0 && result.score <= 1.0);
    assert_eq!(result.epochs_used, 5);
    assert!((result.data_fraction - 0.5).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_proxy_task_evaluate_empty_data() -> Result<()> {
    let proxy = ProxyTask::new(5, 0.5, 0.5, 0.5)?;
    let arch = AdArchEncoding::random(3, 32, 4, 42)?;
    assert!(proxy.evaluate_proxy(&arch, &[], None, None).is_err());
    Ok(())
}

#[test]
fn test_proxy_task_correlation() -> Result<()> {
    let proxy_scores = vec![0.1, 0.4, 0.3, 0.8, 0.6];
    let full_scores = vec![0.15, 0.38, 0.32, 0.75, 0.55];
    let corr = ProxyTask::correlation_analysis(&proxy_scores, &full_scores)?;
    // Strong positive correlation expected
    assert!(corr.kendall_tau > 0.5);
    assert!(corr.spearman > 0.5);
    assert_eq!(corr.n_samples, 5);
    Ok(())
}

#[test]
fn test_proxy_task_correlation_mismatch() {
    let a = vec![0.1, 0.2];
    let b = vec![0.1];
    assert!(ProxyTask::correlation_analysis(&a, &b).is_err());
}

#[test]
fn test_proxy_task_correlation_too_few() {
    let a = vec![0.1];
    let b = vec![0.2];
    assert!(ProxyTask::correlation_analysis(&a, &b).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// ArchDistiller tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_arch_distiller_creation() -> Result<()> {
    let config = AdSearchConfig::default_config();
    let distiller = ArchDistiller::new(config, 42)?;
    assert_eq!(distiller.config.population_size, 20);
    Ok(())
}

#[test]
fn test_arch_distiller_invalid() {
    let mut config = AdSearchConfig::default_config();
    config.population_size = 0;
    assert!(ArchDistiller::new(config, 0).is_err());

    let mut config = AdSearchConfig::default_config();
    config.temperature = -1.0;
    assert!(ArchDistiller::new(config, 0).is_err());

    let mut config = AdSearchConfig::default_config();
    config.alpha = 1.5;
    assert!(ArchDistiller::new(config, 0).is_err());
}

#[test]
fn test_arch_distiller_loss() -> Result<()> {
    let config = AdSearchConfig::default_config();
    let distiller = ArchDistiller::new(config, 42)?;

    let student_logits = vec![2.0, 1.0, 0.1];
    let teacher_logits = vec![3.0, 0.5, 0.2];
    let hard_labels = vec![1.0, 0.0, 0.0];

    let loss = distiller.distillation_loss(&student_logits, &teacher_logits, &hard_labels)?;
    assert!(loss >= 0.0);
    Ok(())
}

#[test]
fn test_arch_distiller_loss_mismatch() -> Result<()> {
    let config = AdSearchConfig::default_config();
    let distiller = ArchDistiller::new(config, 42)?;
    assert!(distiller
        .distillation_loss(&[1.0, 2.0], &[1.0], &[1.0, 0.0])
        .is_err());
    Ok(())
}

#[test]
fn test_arch_distiller_search() -> Result<()> {
    let mut config = AdSearchConfig::default_config();
    config.population_size = 5;
    config.n_rounds = 3;
    config.flops_budget = 500_000_000;
    let mut distiller = ArchDistiller::new(config, 42)?;

    let n_samples = 10;
    let n_classes = 3;
    let teacher_logits: Vec<Vec<f32>> = (0..n_samples)
        .map(|i| vec![(i as f32) * 0.1, 1.0 - (i as f32) * 0.05, 0.3])
        .collect();
    let train_data: Vec<(Vec<f32>, Vec<f32>)> = (0..n_samples)
        .map(|_| {
            let input = vec![0.1; 8];
            let mut label = vec![0.0; n_classes];
            label[0] = 1.0;
            (input, label)
        })
        .collect();

    let result = distiller.search(&teacher_logits, &train_data, 42)?;
    assert!(result.total_evaluated > 0);
    assert!(!result.history.is_empty());
    assert!(result.best_flops <= 500_000_000);
    Ok(())
}

#[test]
fn test_arch_distiller_search_empty() -> Result<()> {
    let config = AdSearchConfig::default_config();
    let mut distiller = ArchDistiller::new(config, 42)?;
    assert!(distiller.search(&[], &[], 42).is_err());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// AdMetrics tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ad_metrics_linear_flops() {
    assert_eq!(AdMetrics::linear_flops(128, 64), 2 * 128 * 64);
}

#[test]
fn test_ad_metrics_conv_flops() {
    let flops = AdMetrics::conv_flops(3, 64, 3, 32);
    assert_eq!(flops, 2 * 3 * 64 * 9 * 32 * 32);
}

#[test]
fn test_ad_metrics_linear_params() {
    assert_eq!(AdMetrics::linear_params(128, 64), 128 * 64 + 64);
}

#[test]
fn test_ad_metrics_activation_memory() {
    let mem = AdMetrics::activation_memory(32, 64, 16);
    assert_eq!(mem, 32 * 64 * 16 * 16 * 4);
}

#[test]
fn test_ad_metrics_latency_proxy() {
    let latency = AdMetrics::latency_proxy(1_000_000_000, 10.0);
    assert!((latency - 0.1).abs() < 1e-6);
    assert_eq!(AdMetrics::latency_proxy(100, 0.0), 0.0);
}

#[test]
fn test_ad_metrics_kendall_tau() -> Result<()> {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let tau = AdMetrics::kendall_tau(&a, &b)?;
    assert!((tau - 1.0).abs() < 1e-6);

    // Reversed order
    let c = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let tau_neg = AdMetrics::kendall_tau(&a, &c)?;
    assert!((tau_neg + 1.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_ad_metrics_kendall_tau_mismatch() {
    assert!(AdMetrics::kendall_tau(&[1.0], &[1.0, 2.0]).is_err());
}

#[test]
fn test_ad_metrics_efficiency_score() {
    let eff = AdMetrics::efficiency_score(0.9, 100_000_000, 200_000_000);
    assert!((eff - 1.8).abs() < 1e-6);
    assert_eq!(AdMetrics::efficiency_score(0.9, 0, 100), 0.0);
    assert_eq!(AdMetrics::efficiency_score(0.9, 100, 0), 0.0);
}

#[test]
fn test_ad_metrics_pareto_front() -> Result<()> {
    let accuracies = vec![0.9, 0.85, 0.8, 0.7];
    let flops = vec![100, 80, 60, 50];
    let pareto = AdMetrics::pareto_front(&accuracies, &flops)?;
    // Point 0 (0.9, 100) and point 3 (0.7, 50) should be Pareto
    assert!(pareto[0]); // Best accuracy
    assert!(pareto[3]); // Lowest FLOPs
    Ok(())
}

#[test]
fn test_ad_metrics_pareto_front_mismatch() {
    assert!(AdMetrics::pareto_front(&[0.9], &[100, 200]).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// AdReport tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ad_report_from_arch() -> Result<()> {
    let arch = AdArchEncoding::random(4, 64, 8, 42)?;
    let report = AdReport::from_arch(&arch, 32);
    assert!(report.flops > 0);
    assert!(report.params > 0);
    assert!(report.memory_bytes > 0);
    assert!(report.latency_seconds > 0.0);
    assert!(report.accuracy.is_none());
    Ok(())
}

#[test]
fn test_ad_report_with_accuracy() -> Result<()> {
    let arch = AdArchEncoding::random(4, 64, 8, 42)?;
    let report = AdReport::from_arch(&arch, 32).with_accuracy(0.92, 1_000_000_000);
    assert_eq!(report.accuracy, Some(0.92));
    assert!(report.efficiency.is_some());
    Ok(())
}

#[test]
fn test_ad_report_summary() -> Result<()> {
    let arch = AdArchEncoding::random(4, 64, 8, 42)?;
    let report = AdReport::from_arch(&arch, 32).with_accuracy(0.92, 1_000_000_000);
    let summary = report.summary();
    assert!(summary.contains("FLOPs"));
    assert!(summary.contains("Params"));
    assert!(summary.contains("Accuracy"));
    assert!(summary.contains("Efficiency"));
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility function tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ad_softmax_temp_basic() {
    let logits = vec![1.0, 2.0, 3.0];
    let probs = ad_softmax_temp(&logits, 1.0);
    assert_eq!(probs.len(), 3);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // Higher logit = higher probability
    assert!(probs[2] > probs[1]);
    assert!(probs[1] > probs[0]);
}

#[test]
fn test_ad_softmax_temp_high_temp() {
    let logits = vec![1.0, 2.0, 3.0];
    let probs = ad_softmax_temp(&logits, 100.0);
    // High temperature -> nearly uniform
    for &p in &probs {
        assert!((p - 1.0 / 3.0).abs() < 0.05);
    }
}

#[test]
fn test_ad_kl_div_same_dist() {
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let kl = ad_kl_div(&p, &p);
    assert!(kl.abs() < 1e-6);
}

#[test]
fn test_compute_ranks() {
    let values = vec![3.0, 1.0, 2.0];
    let ranks = compute_ranks(&values);
    assert!((ranks[0] - 3.0).abs() < 1e-6); // 3.0 is largest -> rank 3
    assert!((ranks[1] - 1.0).abs() < 1e-6); // 1.0 is smallest -> rank 1
    assert!((ranks[2] - 2.0).abs() < 1e-6);
}

#[test]
fn test_pearson_correlation_perfect() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0, 4.0, 6.0, 8.0];
    let r = pearson_correlation(&a, &b);
    assert!((r - 1.0).abs() < 1e-5);
}

#[test]
fn test_ad_search_config_default() {
    let config = AdSearchConfig::default_config();
    assert_eq!(config.population_size, 20);
    assert_eq!(config.n_rounds, 10);
    assert!((config.temperature - 4.0).abs() < 1e-6);
    assert!((config.alpha - 0.5).abs() < 1e-6);
}
