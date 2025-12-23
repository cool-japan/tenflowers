use scirs2_core::ndarray::{Array1, Array2};
use std::time::Instant;
use tenflowers_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("TenfloweRS Simple Performance Test");
    println!("=================================");

    // Test basic tensor operations
    let sizes = [100, 500, 1000];

    for &size in &sizes {
        println!("\nTesting size: {}x{}", size, size);

        // Matrix multiplication test
        let a = Tensor::from_array(Array2::<f32>::ones((size, size)).into_dyn());
        let b = Tensor::from_array(Array2::<f32>::ones((size, size)).into_dyn());

        let start = Instant::now();
        let _c = a.matmul(&b)?;
        let matmul_time = start.elapsed();
        println!("  Matrix multiply: {:.2?}", matmul_time);

        // Element-wise operations test
        let x = Tensor::from_array(Array1::<f32>::ones(size * size).into_dyn());
        let y = Tensor::from_array(Array1::<f32>::ones(size * size).into_dyn());

        let start = Instant::now();
        let _z = x.add(&y)?;
        let add_time = start.elapsed();
        println!("  Element-wise add: {:.2?}", add_time);

        let start = Instant::now();
        let _z = x.mul(&y)?;
        let mul_time = start.elapsed();
        println!("  Element-wise mul: {:.2?}", mul_time);
    }

    // Test neural network-like operations
    println!("\nTesting neural network operations:");
    let batch_size = 32;
    let input_size = 784; // MNIST-like
    let hidden_size = 512;
    let output_size = 10;

    // Forward pass simulation
    let input = Tensor::from_array(Array2::<f32>::ones((batch_size, input_size)).into_dyn());
    let w1 = Tensor::from_array(Array2::<f32>::ones((input_size, hidden_size)).into_dyn());
    let w2 = Tensor::from_array(Array2::<f32>::ones((hidden_size, output_size)).into_dyn());

    let start = Instant::now();
    let h1 = input.matmul(&w1)?;
    let h1_relu = h1.relu()?;
    let output = h1_relu.matmul(&w2)?;
    let forward_time = start.elapsed();

    println!(
        "  Forward pass ({}x{}->{}->{}): {:.2?}",
        batch_size, input_size, hidden_size, output_size, forward_time
    );

    println!("\nPerformance test completed!");
    Ok(())
}
