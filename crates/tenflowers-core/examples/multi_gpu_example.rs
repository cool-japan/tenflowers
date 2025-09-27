use tenflowers_core::{
    collective::{broadcast, create_process_group, init_collective},
    Device, Result, Tensor,
};

fn main() -> Result<()> {
    println!("TenfloweRS Multi-GPU Example");
    println!("============================");

    // Initialize collective communication
    init_collective()?;

    // Create a tensor on CPU
    let cpu_tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    println!("Original CPU tensor:");
    if let Some(data) = cpu_tensor.as_slice() {
        println!("  {:?}", data);
    }
    println!("  Device: {}", cpu_tensor.device());

    // Test device transfer capabilities
    println!("\n1. Testing Device Transfer:");
    println!("   CPU -> CPU transfer:");
    let cpu_tensor2 = cpu_tensor.to_cpu()?;
    println!(
        "   ✓ CPU transfer successful, device: {}",
        cpu_tensor2.device()
    );

    // Test GPU transfer (if available)
    #[cfg(feature = "gpu")]
    {
        println!("   CPU -> GPU transfer:");
        match cpu_tensor.to_gpu(0) {
            Ok(gpu_tensor) => {
                println!(
                    "   ✓ GPU transfer successful, device: {}",
                    gpu_tensor.device()
                );

                // Test GPU -> CPU transfer
                println!("   GPU -> CPU transfer:");
                let cpu_back = gpu_tensor.to_cpu()?;
                println!(
                    "   ✓ GPU->CPU transfer successful, device: {}",
                    cpu_back.device()
                );

                // Test GPU -> GPU transfer (different device)
                println!("   GPU -> GPU transfer (device 0 -> 1):");
                match gpu_tensor.to_gpu(1) {
                    Ok(gpu_tensor2) => {
                        println!(
                            "   ✓ GPU->GPU transfer successful, device: {}",
                            gpu_tensor2.device()
                        );
                    }
                    Err(e) => {
                        println!(
                            "   ⚠ GPU->GPU transfer failed (likely only 1 GPU available): {}",
                            e
                        );
                    }
                }
            }
            Err(e) => {
                println!("   ⚠ GPU transfer failed (GPU not available): {}", e);
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("   ⚠ GPU features not compiled");
    }

    // Test collective operations
    println!("\n2. Testing Collective Operations:");

    // Create communication group with available devices
    let devices = vec![Device::Cpu];
    create_process_group("default".to_string(), devices)?;
    println!("   ✓ Created communication group 'default'");

    // Test broadcast operation
    println!("   Testing broadcast:");
    let broadcast_results = broadcast(&cpu_tensor, Device::Cpu, Some("default"))?;
    println!(
        "   ✓ Broadcast successful, got {} tensors",
        broadcast_results.len()
    );

    // Skip all-reduce for now due to trait bound complexity
    println!("   ⚠ All-reduce tests skipped (requires Float trait bounds)");

    // Demonstrate device placement capabilities
    println!("\n3. Testing Device Placement:");
    let tensor1 = Tensor::<f32>::ones(&[3, 3]);
    let tensor2 = Tensor::<f32>::zeros(&[3, 3]);

    println!("   tensor1 device: {}", tensor1.device());
    println!("   tensor2 device: {}", tensor2.device());

    // Test device compatibility
    println!("   Testing device compatibility:");
    println!(
        "   Can CPU tensor transfer to CPU? {}",
        tensor1.can_transfer_to(Device::Cpu)
    );
    #[cfg(feature = "gpu")]
    {
        println!(
            "   Can CPU tensor transfer to GPU? {}",
            tensor1.can_transfer_to(Device::Gpu(0))
        );
    }

    println!("\n✓ Multi-GPU example completed successfully!");

    Ok(())
}
