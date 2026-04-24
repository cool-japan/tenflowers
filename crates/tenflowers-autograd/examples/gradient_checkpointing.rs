//! Example: Gradient checkpointing for memory-efficient training.
//!
//! Checkpointing trades recomputation for memory: only every N-th intermediate
//! activation is stored during the forward pass; the others are recomputed on
//! the fly during backpropagation.
//!
//! This example demonstrates:
//! 1. [`CheckpointManager`] with [`CheckpointStrategy::EveryNLayers`].
//! 2. [`checkpoint_sequence`] — apply a sequence of closures with selective storage.
//! 3. [`ActivationRecomputeManager`] — policy-based per-layer recomputation control.

use tenflowers_autograd::{
    checkpoint_sequence, ActivationCheckpointPolicy, ActivationRecomputeManager, CheckpointManager,
    CheckpointStrategy, GradientTape, LayerMetadata, TrackedTensor,
};
use tenflowers_core::Tensor;

type LayerFn = Box<dyn Fn(&TrackedTensor<f32>) -> tenflowers_core::Result<TrackedTensor<f32>>>;

// TensorError is the framework's canonical error type; boxing would break the checkpoint_sequence API signature.
#[allow(clippy::result_large_err)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Gradient checkpointing example\n");

    // -------------------------------------------------------------------------
    // Part 1: CheckpointManager — track which tensors are stored
    // -------------------------------------------------------------------------
    println!("Part 1: CheckpointManager with EveryNLayers(2) strategy");
    println!("----------------------------------------------------------");

    let manager = CheckpointManager::new(CheckpointStrategy::EveryNLayers(2));

    // Simulate 5 layers; layers 0, 2, 4 will be checkpointed
    for layer_idx in 0..5_usize {
        let should = manager.should_checkpoint(layer_idx, "relu", layer_idx);
        println!("  layer {layer_idx}: checkpoint = {should}");
    }
    println!("  Memory used by manager: {} bytes", manager.memory_usage());
    println!();

    // -------------------------------------------------------------------------
    // Part 2: checkpoint_sequence — wrap layer closures
    // -------------------------------------------------------------------------
    println!("Part 2: checkpoint_sequence over 4 ReLU layers");
    println!("-----------------------------------------------");

    let tape = GradientTape::new();
    let input = tape.watch(Tensor::<f32>::ones(&[4, 16]));

    // Each closure is one "layer" in the sequence. Only even-indexed ones are
    // stored; the odd ones are discarded and will be recomputed on backward.
    let layers: Vec<LayerFn> = vec![
        Box::new(|t| t.relu()),
        Box::new(|t| t.relu()),
        Box::new(|t| t.relu()),
        Box::new(|t| t.relu()),
    ];

    let output = checkpoint_sequence(layers, CheckpointStrategy::EveryNLayers(2), input)?;
    println!("  Output shape: {:?}", output.tensor.shape().dims());
    println!("  Tape node count: {}", tape.node_count());
    println!();

    // -------------------------------------------------------------------------
    // Part 3: ActivationRecomputeManager — policy-based per-layer control
    // -------------------------------------------------------------------------
    println!("Part 3: ActivationRecomputeManager — EveryNLayers policy");
    println!("----------------------------------------------------------");

    // Checkpoint every 2nd layer; recompute the rest.
    let policy = ActivationCheckpointPolicy::EveryNLayers(2);
    let mut recompute_mgr = ActivationRecomputeManager::new(policy);

    // Register some layers with size and cost metadata
    let layer_info: [(usize, &str, usize); 4] = [
        (0, "conv2d", 1024),
        (1, "linear", 4096),
        (2, "relu", 512),
        (3, "linear", 8192),
    ];
    for (idx, layer_type, mem) in layer_info {
        recompute_mgr.register_layer(LayerMetadata {
            name: format!("layer_{idx}"),
            layer_type: layer_type.to_string(),
            layer_index: idx,
            computation_cost: mem as f64 * 1e-9,
            memory_requirement: mem,
            is_memory_intensive: mem > 2048,
            is_compute_intensive: layer_type == "linear",
        });
    }

    for (idx, layer_type, mem) in layer_info {
        let store = recompute_mgr.should_checkpoint_activation(idx, layer_type, mem);
        let action = if store { "store" } else { "recompute" };
        println!("  layer {idx} (type={layer_type:<8} mem={mem:>5}B): {action}");
    }

    println!("\nGradient checkpointing example complete.");
    Ok(())
}
