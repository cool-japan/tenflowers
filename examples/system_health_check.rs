use tenflowers_core::{
    run_system_health_check, run_quick_health_check,
    SystemHealthChecker, HealthCheckConfig, HealthStatus
};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 TenfloweRS System Health Check Example");
    println!("==========================================\n");
    
    // Run a quick health check (no performance tests)
    println!("⚡ Running quick health check...\n");
    let quick_info = run_quick_health_check()?;
    
    println!("\n" + "=".repeat(50).as_str());
    
    // Run a comprehensive health check with performance tests
    println!("\n🏃 Running comprehensive health check with performance benchmarks...\n");
    let comprehensive_info = run_system_health_check()?;
    
    println!("\n" + "=".repeat(50).as_str());
    
    // Custom health check configuration
    println!("\n⚙️  Running custom health check...\n");
    let custom_config = HealthCheckConfig {
        run_performance_tests: true,
        test_duration: Duration::from_secs(3),
        memory_threshold_warning: 0.7,
        memory_threshold_critical: 0.9,
        performance_threshold_warning: 0.5,
    };
    
    let custom_checker = SystemHealthChecker::with_config(custom_config);
    let custom_info = custom_checker.check_system_health()?;
    
    // Analyze results
    println!("\n📊 Health Check Analysis");
    println!("========================");
    
    println!("\nQuick check status: {:?}", quick_info.health_status);
    println!("Comprehensive check status: {:?}", comprehensive_info.health_status);
    println!("Custom check status: {:?}", custom_info.health_status);
    
    // Show device comparison
    println!("\n🖥️  Device Information:");
    println!("Available devices: {}", comprehensive_info.available_devices.len());
    for device in &comprehensive_info.available_devices {
        println!("  • {}", device);
    }
    
    // Performance summary
    if comprehensive_info.performance_benchmarks.cpu_add_throughput > 0.0 {
        println!("\n⚡ Performance Summary:");
        println!("  CPU Performance:");
        println!("    Add operations: {:.2} GFLOPS", comprehensive_info.performance_benchmarks.cpu_add_throughput);
        println!("    Matrix multiply: {:.2} GFLOPS", comprehensive_info.performance_benchmarks.cpu_matmul_throughput);
        
        if let Some(gpu_add) = comprehensive_info.performance_benchmarks.gpu_add_throughput {
            println!("  GPU Performance:");
            println!("    Add operations: {:.2} GFLOPS", gpu_add);
            if let Some(gpu_matmul) = comprehensive_info.performance_benchmarks.gpu_matmul_throughput {
                println!("    Matrix multiply: {:.2} GFLOPS", gpu_matmul);
                
                let speedup = gpu_matmul / comprehensive_info.performance_benchmarks.cpu_matmul_throughput;
                println!("    GPU Speedup: {:.1}x", speedup);
            }
        }
        
        println!("  Tensor creation latency: {:?}", comprehensive_info.performance_benchmarks.tensor_creation_latency);
        
        if let Some(bandwidth) = comprehensive_info.performance_benchmarks.device_transfer_bandwidth {
            println!("  Device transfer bandwidth: {:.2} GB/s", bandwidth);
        }
    }
    
    // Feature status
    println!("\n🔧 Feature Status:");
    let features = &comprehensive_info.features_enabled;
    println!("  GPU Support: {}", if features.gpu_support { "✅ Enabled" } else { "❌ Disabled" });
    println!("  CUDA: {}", if features.cuda_available { "✅ Available" } else { "❌ Not available" });
    println!("  Metal: {}", if features.metal_available { "✅ Available" } else { "❌ Not available" });
    println!("  ROCm: {}", if features.rocm_available { "✅ Available" } else { "❌ Not available" });
    println!("  BLAS Acceleration: {}", if features.blas_acceleration { "✅ Enabled" } else { "❌ Disabled" });
    println!("  Mixed Precision: {}", if features.mixed_precision { "✅ Enabled" } else { "❌ Disabled" });
    
    // Health status interpretation
    println!("\n🏥 Overall System Assessment:");
    match &comprehensive_info.health_status {
        HealthStatus::Excellent => {
            println!("  🌟 Your TenfloweRS installation is running at peak performance!");
            println!("     All features are optimally configured for maximum efficiency.");
        }
        HealthStatus::Good => {
            println!("  👍 Your TenfloweRS installation is working well.");
            println!("     Consider enabling additional features for better performance.");
        }
        HealthStatus::Warning(warnings) => {
            println!("  ⚠️  Your TenfloweRS installation has some issues that should be addressed:");
            for warning in warnings {
                println!("     • {}", warning);
            }
            println!("     These issues may impact performance but won't prevent usage.");
        }
        HealthStatus::Critical(issues) => {
            println!("  🚨 Your TenfloweRS installation has critical issues:");
            for issue in issues {
                println!("     • {}", issue);
            }
            println!("     These issues require immediate attention for optimal performance.");
        }
    }
    
    // Final recommendations
    println!("\n💡 Next Steps:");
    
    if !features.gpu_support && comprehensive_info.available_devices.iter().any(|d| d.is_cpu()) {
        println!("  1. Consider recompiling with GPU support for better performance");
        println!("     cargo build --features gpu");
    }
    
    if !features.blas_acceleration {
        println!("  2. Enable BLAS acceleration for improved CPU linear algebra");
        println!("     cargo build --features blas-openblas");
    }
    
    if comprehensive_info.performance_benchmarks.cpu_matmul_throughput < 10.0 {
        println!("  3. CPU performance seems low - check system load and consider upgrading");
    }
    
    println!("  • Run this health check regularly to monitor system performance");
    println!("  • Use 'run_quick_health_check()' for fast system verification");
    println!("  • Use 'run_system_health_check()' for detailed performance analysis");
    
    println!("\n✨ Happy computing with TenfloweRS! ✨");
    
    Ok(())
}