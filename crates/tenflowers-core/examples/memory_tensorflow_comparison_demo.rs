//! Memory Usage Profiling vs TensorFlow Demo
//!
//! This example demonstrates the memory usage profiling and optimization features
//! designed to achieve memory usage within 10% of TensorFlow.

use std::time::Instant;
use tenflowers_core::{
    DType, Device, MemoryProfilingConfig, MemorySnapshot, TensorFlowMemoryProfiler, MEMORY_PROFILER,
};

fn main() {
    println!("📊 TenfloweRS Memory Usage Profiling vs TensorFlow Demo");
    println!("======================================================");

    // Create custom memory profiling configuration
    let config = MemoryProfilingConfig {
        enable_memory_tracking: true,
        enable_tensorflow_comparison: true,
        target_efficiency_ratio: 0.9, // Within 10% of TensorFlow
        python_executable: "python3".to_string(),
        enable_detailed_tracking: true,
        enable_optimization_suggestions: true,
        sampling_interval_ms: 50, // Sample every 50ms
    };

    println!("Configuration:");
    println!("  • Memory tracking: {}", config.enable_memory_tracking);
    println!(
        "  • TensorFlow comparison: {}",
        config.enable_tensorflow_comparison
    );
    println!(
        "  • Target efficiency: ≥{:.1}% of TensorFlow",
        config.target_efficiency_ratio * 100.0
    );
    println!("  • Detailed tracking: {}", config.enable_detailed_tracking);
    println!(
        "  • Optimization suggestions: {}",
        config.enable_optimization_suggestions
    );
    println!();

    // Create memory profiler with custom config
    let profiler = TensorFlowMemoryProfiler::new(config);

    // Simulate memory profiling for various operations
    println!("🔄 Simulating Memory Profiling Operations:");

    let operations = [
        (
            "add",
            vec![vec![1000, 1000], vec![1000, 1000]],
            15.0,
            Some(12.0),
        ),
        (
            "mul",
            vec![vec![1000, 1000], vec![1000, 1000]],
            16.0,
            Some(13.5),
        ),
        (
            "matmul",
            vec![vec![512, 512], vec![512, 512]],
            42.0,
            Some(38.0),
        ),
        ("conv2d", vec![vec![32, 224, 224, 3]], 85.0, Some(78.0)),
        (
            "large_matmul",
            vec![vec![2048, 2048], vec![2048, 2048]],
            180.0,
            Some(150.0),
        ),
    ];

    let mut simulated_snapshots = Vec::new();

    for (op_name, input_shapes, tf_rs_memory, tf_memory) in &operations {
        println!("  Testing {op_name} with shapes {:?}", input_shapes);

        // Create simulated memory snapshot (in real use this would be from actual profiling)
        let efficiency = tf_memory.map(|tf| tf / tf_rs_memory).unwrap_or(1.0);
        let meets_target = efficiency >= 0.9;

        let snapshot = MemorySnapshot {
            timestamp: Instant::now(),
            operation: op_name.to_string(),
            tenflowers_memory_mb: *tf_rs_memory,
            tensorflow_memory_mb: *tf_memory,
            pytorch_memory_mb: tf_memory.map(|tf| tf * 1.05), // Simulate PyTorch slightly higher
            input_shapes: input_shapes.clone(),
            dtype: DType::Float32,
            device: Device::Cpu,
            memory_efficiency: efficiency,
            meets_target,
        };

        if meets_target {
            println!(
                "    ✅ Memory usage: {:.1}MB (efficiency: {:.1}%)",
                tf_rs_memory,
                efficiency * 100.0
            );
        } else {
            println!(
                "    ❌ Memory usage: {:.1}MB (efficiency: {:.1}%)",
                tf_rs_memory,
                efficiency * 100.0
            );
        }

        simulated_snapshots.push(snapshot);
    }

    println!();

    // Generate comprehensive memory comparison report
    let mut total_ops = simulated_snapshots.len();
    let mut ops_meeting_target = simulated_snapshots
        .iter()
        .filter(|s| s.meets_target)
        .count();
    let success_rate = ops_meeting_target as f64 / total_ops as f64;

    println!("📈 Memory Comparison Summary:");
    println!("  • Total operations: {}", total_ops);
    println!(
        "  • Operations meeting target: {}/{}",
        ops_meeting_target, total_ops
    );
    println!("  • Success rate: {:.1}%", success_rate * 100.0);
    println!();

    // Calculate average efficiency
    let tf_snapshots: Vec<_> = simulated_snapshots
        .iter()
        .filter(|s| s.tensorflow_memory_mb.is_some())
        .collect();

    if !tf_snapshots.is_empty() {
        let avg_efficiency = tf_snapshots
            .iter()
            .map(|s| s.memory_efficiency)
            .sum::<f64>()
            / tf_snapshots.len() as f64;

        let avg_tf_rs_memory = tf_snapshots
            .iter()
            .map(|s| s.tenflowers_memory_mb)
            .sum::<f64>()
            / tf_snapshots.len() as f64;

        let avg_tf_memory = tf_snapshots
            .iter()
            .filter_map(|s| s.tensorflow_memory_mb)
            .sum::<f64>()
            / tf_snapshots.len() as f64;

        println!("💾 Memory Usage Analysis:");
        println!("  • Average TenfloweRS memory: {:.1} MB", avg_tf_rs_memory);
        println!("  • Average TensorFlow memory: {:.1} MB", avg_tf_memory);
        println!(
            "  • Average memory efficiency: {:.1}%",
            avg_efficiency * 100.0
        );

        if avg_efficiency >= 0.9 {
            println!("  ✅ Overall memory efficiency meets target!");
        } else {
            let overhead = avg_tf_rs_memory - avg_tf_memory;
            let overhead_percentage = (overhead / avg_tf_memory) * 100.0;
            println!(
                "  ❌ Memory overhead: {:.1} MB ({:.1}%)",
                overhead, overhead_percentage
            );
        }
    }

    println!();

    // Demonstrate optimization features
    println!("⚡ Memory Optimization Features Implemented:");
    println!("  ✅ Real-time memory usage tracking and comparison vs TensorFlow");
    println!("  ✅ Automatic detection of memory efficiency issues");
    println!("  ✅ TensorFlow and PyTorch memory usage measurement via Python scripts");
    println!("  ✅ Memory optimization suggestions with priority levels");
    println!("  ✅ Detailed per-operation memory profiling");
    println!("  ✅ Integration with existing PerformanceMonitor infrastructure");
    println!("  ✅ Support for both CPU and GPU memory tracking");
    println!("  ✅ Configurable target efficiency thresholds");
    println!();

    // Demonstrate global memory profiler
    println!("🌍 Global Memory Profiler:");
    println!("  The global MEMORY_PROFILER is available for memory profiling");
    println!("  Use profiler.profile_operation_vs_tensorflow() for comparisons");
    println!();

    // Show detailed breakdown table
    println!("📋 Detailed Memory Usage Breakdown:");
    println!("{:-<80}", "");
    println!(
        "| {:^12} | {:^18} | {:^12} | {:^12} | {:^10} |",
        "Operation", "Shapes", "TF RS (MB)", "TensorFlow (MB)", "Efficiency"
    );
    println!("{:-<80}", "");

    for snapshot in &simulated_snapshots {
        let shapes_str = snapshot
            .input_shapes
            .iter()
            .map(|s| {
                format!(
                    "[{}]",
                    s.iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("×")
                )
            })
            .collect::<Vec<_>>()
            .join(" ");

        let tf_memory_str = snapshot
            .tensorflow_memory_mb
            .map(|m| format!("{:.1}", m))
            .unwrap_or_else(|| "N/A".to_string());

        let efficiency_str = if snapshot.tensorflow_memory_mb.is_some() {
            format!("{:.1}%", snapshot.memory_efficiency * 100.0)
        } else {
            "N/A".to_string()
        };

        println!(
            "| {:^12} | {:^18} | {:^12.1} | {:^12} | {:^10} |",
            snapshot.operation,
            if shapes_str.len() > 18 {
                format!("{}...", &shapes_str[..15])
            } else {
                shapes_str
            },
            snapshot.tenflowers_memory_mb,
            tf_memory_str,
            efficiency_str
        );
    }
    println!("{:-<80}", "");

    println!();
    println!("🎯 Target Achievement:");
    if success_rate >= 0.8 {
        println!("  ✅ Good progress toward memory efficiency target!");
    } else {
        println!("  ⚠️  More optimization needed to reach target efficiency");
    }

    println!("  💡 The implementation provides infrastructure to achieve");
    println!("     memory usage within 10% of TensorFlow as specified in TODO.md");

    println!();
    println!("======================================================");
}
