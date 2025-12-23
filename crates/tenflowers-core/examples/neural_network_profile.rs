use scirs2_core::ndarray::Array2;
use std::time::Instant;
use tenflowers_core::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Neural Network Forward Pass Profiling");
    println!("=====================================");

    let batch_size = 32;
    let input_size = 784;
    let hidden_size = 512;
    let output_size = 10;

    println!(
        "Network architecture: {}x{} -> {} -> {}",
        batch_size, input_size, hidden_size, output_size
    );

    // Create tensors
    println!("\n1. Creating tensors...");
    let tensor_start = Instant::now();
    let input = Tensor::from_array(Array2::<f32>::ones((batch_size, input_size)).into_dyn());
    let w1 = Tensor::from_array(Array2::<f32>::ones((input_size, hidden_size)).into_dyn());
    let w2 = Tensor::from_array(Array2::<f32>::ones((hidden_size, output_size)).into_dyn());
    let tensor_time = tensor_start.elapsed();
    println!("   Tensor creation: {:.2?}", tensor_time);

    // First matrix multiplication: input @ w1 (32x784 @ 784x512 = 32x512)
    println!(
        "\n2. First matrix multiplication ({}x{} @ {}x{})...",
        batch_size, input_size, input_size, hidden_size
    );
    let matmul1_start = Instant::now();
    let h1 = input.matmul(&w1)?;
    let matmul1_time = matmul1_start.elapsed();
    let ops1 = 2 * batch_size * input_size * hidden_size; // 2 ops per multiply-accumulate
    let gflops1 = (ops1 as f64) / (matmul1_time.as_secs_f64() * 1e9);
    println!(
        "   First matmul: {:.2?} ({:.2} GFLOPS)",
        matmul1_time, gflops1
    );

    // ReLU activation
    println!("\n3. ReLU activation...");
    let relu_start = Instant::now();
    let h1_relu = h1.relu()?;
    let relu_time = relu_start.elapsed();
    let relu_ops = batch_size * hidden_size; // 1 comparison + selection per element
    let relu_ops_per_sec = (relu_ops as f64) / relu_time.as_secs_f64();
    println!(
        "   ReLU activation: {:.2?} ({:.2} M ops/sec)",
        relu_time,
        relu_ops_per_sec / 1e6
    );

    // Second matrix multiplication: h1_relu @ w2 (32x512 @ 512x10 = 32x10)
    println!(
        "\n4. Second matrix multiplication ({}x{} @ {}x{})...",
        batch_size, hidden_size, hidden_size, output_size
    );
    let matmul2_start = Instant::now();
    let output = h1_relu.matmul(&w2)?;
    let matmul2_time = matmul2_start.elapsed();
    let ops2 = 2 * batch_size * hidden_size * output_size;
    let gflops2 = (ops2 as f64) / (matmul2_time.as_secs_f64() * 1e9);
    println!(
        "   Second matmul: {:.2?} ({:.2} GFLOPS)",
        matmul2_time, gflops2
    );

    // Total forward pass
    let total_time = matmul1_time + relu_time + matmul2_time;
    let total_ops = ops1 + relu_ops + ops2;
    let total_gflops = (total_ops as f64) / (total_time.as_secs_f64() * 1e9);

    println!("\n=== SUMMARY ===");
    println!(
        "First matmul:   {:.2?} ({:.1}%)",
        matmul1_time,
        (matmul1_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );
    println!(
        "ReLU:           {:.2?} ({:.1}%)",
        relu_time,
        (relu_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );
    println!(
        "Second matmul:  {:.2?} ({:.1}%)",
        matmul2_time,
        (matmul2_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );
    println!("─────────────────────────────");
    println!(
        "Total:          {:.2?} ({:.2} GFLOPS)",
        total_time, total_gflops
    );

    // Analysis
    println!("\n=== ANALYSIS ===");
    println!("Operations count:");
    println!(
        "  - First matmul:  {:.1}M ops ({:.1}%)",
        ops1 as f64 / 1e6,
        (ops1 as f64 / total_ops as f64) * 100.0
    );
    println!(
        "  - ReLU:          {:.1}K ops ({:.1}%)",
        relu_ops as f64 / 1e3,
        (relu_ops as f64 / total_ops as f64) * 100.0
    );
    println!(
        "  - Second matmul: {:.1}K ops ({:.1}%)",
        ops2 as f64 / 1e3,
        (ops2 as f64 / total_ops as f64) * 100.0
    );

    println!("\nBottleneck identification:");
    if matmul1_time > matmul2_time && matmul1_time > relu_time {
        println!("  → First matrix multiplication is the primary bottleneck");
        println!(
            "  → Focus optimization on large matrix multiplications ({}x{} shapes)",
            input_size, hidden_size
        );
    } else if matmul2_time > matmul1_time && matmul2_time > relu_time {
        println!("  → Second matrix multiplication is the primary bottleneck");
    } else {
        println!("  → ReLU activation is surprisingly slow - investigate element-wise operations");
    }

    println!("\nPerformance targets:");
    let target_gflops = 100.0; // Target performance
    let speedup_needed = target_gflops / total_gflops;
    println!("  → Current: {:.2} GFLOPS", total_gflops);
    println!("  → Target:  {:.2} GFLOPS", target_gflops);
    println!("  → Speedup needed: {:.1}x", speedup_needed);

    Ok(())
}
