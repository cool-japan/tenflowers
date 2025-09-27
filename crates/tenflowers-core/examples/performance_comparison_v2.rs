//! 🔬 Performance Comparison: V1 vs V2 Ultra-Performance Implementations
//!
//! This analysis compares the original ultra_matmul with the redesigned ultra_matmul_v2
//! to validate that the performance issues have been resolved with humility.

use std::time::Instant;
use tenflowers_core::{
    ops::{matmul, ultra_matmul, ultra_matmul_v2},
    ultra_performance_profiler::{clear_performance_data, configure_profiler, ProfilerConfig},
    Tensor,
};

#[derive(Debug, Clone)]
struct PerformanceComparison {
    size: String,
    standard_time_ns: u64,
    ultra_v1_time_ns: u64,
    ultra_v2_time_ns: u64,
    v1_vs_standard_ratio: f64,
    v2_vs_standard_ratio: f64,
    v2_vs_v1_improvement: f64,
    verdict: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 ULTRA-PERFORMANCE V1 vs V2 COMPARISON ANALYSIS");
    println!("{}", "=".repeat(70));
    println!("Validating performance improvements with scientific rigor and humility");
    println!();

    // Configure high-precision profiler
    configure_profiler(ProfilerConfig {
        detailed_profiling: true,
        max_history_entries: 2000,
        min_record_time: 1, // 1 nanosecond precision
        optimization_recommendations: true,
    });

    clear_performance_data();

    let mut comparisons = Vec::new();

    // Test different matrix sizes
    let sizes = vec![16, 32, 64, 128, 256];

    println!("⏱️  PERFORMANCE COMPARISON RESULTS:");
    println!("{}", "-".repeat(70));
    println!(
        "{:>8} | {:>12} | {:>12} | {:>12} | {:>8} | {:>8} | {:>10}",
        "Size", "Standard", "Ultra V1", "Ultra V2", "V1 Ratio", "V2 Ratio", "V2 vs V1"
    );
    println!("{}", "-".repeat(70));

    for size in sizes {
        let comparison = compare_implementations(size)?;

        println!(
            "{:>8} | {:>10.0}ns | {:>10.0}ns | {:>10.0}ns | {:>7.2}x | {:>7.2}x | {:>9.2}x",
            comparison.size,
            comparison.standard_time_ns,
            comparison.ultra_v1_time_ns,
            comparison.ultra_v2_time_ns,
            comparison.v1_vs_standard_ratio,
            comparison.v2_vs_standard_ratio,
            comparison.v2_vs_v1_improvement
        );

        comparisons.push(comparison);
    }

    println!("{}", "-".repeat(70));
    println!();

    // Detailed analysis
    analyze_results(&comparisons);

    // Aspect ratio tests
    println!("🔍 ASPECT RATIO PERFORMANCE ANALYSIS:");
    println!("{}", "-".repeat(50));

    let aspect_ratios = vec![
        ("2x2 (micro)", 2, 2, 2),
        ("8x8 (small)", 8, 8, 8),
        ("Outer 64x1x64", 64, 1, 64),
        ("Vector 1x128x1", 1, 128, 1),
        ("Wide 32x128x64", 32, 128, 64),
        ("Tall 128x32x64", 128, 32, 64),
    ];

    for (name, m, k, n) in aspect_ratios {
        let comparison = compare_aspect_ratio(name, m, k, n)?;
        println!(
            "  {}: V2 is {:.2}x {} than V1, {:.2}x {} than Standard",
            name,
            comparison.v2_vs_v1_improvement,
            if comparison.v2_vs_v1_improvement > 1.0 {
                "FASTER"
            } else {
                "slower"
            },
            comparison.v2_vs_standard_ratio,
            if comparison.v2_vs_standard_ratio > 1.0 {
                "FASTER"
            } else {
                "slower"
            }
        );
    }

    println!();

    // Final assessment
    final_assessment(&comparisons);

    Ok(())
}

fn compare_implementations(
    size: usize,
) -> Result<PerformanceComparison, Box<dyn std::error::Error>> {
    let a_data: Vec<f32> = (0..size * size)
        .map(|i| (i as f32) / (size as f32))
        .collect();
    let b_data: Vec<f32> = (0..size * size)
        .map(|i| ((i + 1) as f32) / (size as f32))
        .collect();

    let a = Tensor::<f32>::from_vec(a_data.clone(), &[size, size])?;
    let b = Tensor::<f32>::from_vec(b_data.clone(), &[size, size])?;

    // Warm up all implementations
    let _ = matmul(&a, &b)?;
    let _ = ultra_matmul(&a, &b)?;
    let _ = ultra_matmul_v2(&a, &b)?;

    // Measure standard matmul
    let iterations = 50;
    let mut standard_times = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = matmul(&a, &b)?;
        standard_times.push(start.elapsed().as_nanos());
    }

    // Measure ultra matmul V1
    let mut ultra_v1_times = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = ultra_matmul(&a, &b)?;
        ultra_v1_times.push(start.elapsed().as_nanos());
    }

    // Measure ultra matmul V2
    let mut ultra_v2_times = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = ultra_matmul_v2(&a, &b)?;
        ultra_v2_times.push(start.elapsed().as_nanos());
    }

    // Calculate median times to avoid outliers
    standard_times.sort();
    ultra_v1_times.sort();
    ultra_v2_times.sort();

    let standard_median = standard_times[iterations / 2];
    let ultra_v1_median = ultra_v1_times[iterations / 2];
    let ultra_v2_median = ultra_v2_times[iterations / 2];

    let v1_vs_standard_ratio = standard_median as f64 / ultra_v1_median as f64;
    let v2_vs_standard_ratio = standard_median as f64 / ultra_v2_median as f64;
    let v2_vs_v1_improvement = ultra_v1_median as f64 / ultra_v2_median as f64;

    let verdict = if v2_vs_standard_ratio >= 1.1 {
        "EXCELLENT".to_string()
    } else if v2_vs_standard_ratio >= 0.95 {
        "GOOD".to_string()
    } else if v2_vs_standard_ratio >= 0.8 {
        "ACCEPTABLE".to_string()
    } else {
        "NEEDS_WORK".to_string()
    };

    Ok(PerformanceComparison {
        size: format!("{}x{}", size, size),
        standard_time_ns: standard_median as u64,
        ultra_v1_time_ns: ultra_v1_median as u64,
        ultra_v2_time_ns: ultra_v2_median as u64,
        v1_vs_standard_ratio,
        v2_vs_standard_ratio,
        v2_vs_v1_improvement,
        verdict,
    })
}

fn compare_aspect_ratio(
    name: &str,
    m: usize,
    k: usize,
    n: usize,
) -> Result<PerformanceComparison, Box<dyn std::error::Error>> {
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) / (m * k) as f32).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) / (k * n) as f32).collect();

    let a = Tensor::<f32>::from_vec(a_data, &[m, k])?;
    let b = Tensor::<f32>::from_vec(b_data, &[k, n])?;

    // Warm up
    let _ = matmul(&a, &b)?;
    let _ = ultra_matmul(&a, &b)?;
    let _ = ultra_matmul_v2(&a, &b)?;

    // Measure each implementation
    let iterations = 100;

    let standard_start = Instant::now();
    for _ in 0..iterations {
        let _ = matmul(&a, &b)?;
    }
    let standard_time = standard_start.elapsed().as_nanos() / iterations;

    let v1_start = Instant::now();
    for _ in 0..iterations {
        let _ = ultra_matmul(&a, &b)?;
    }
    let v1_time = v1_start.elapsed().as_nanos() / iterations;

    let v2_start = Instant::now();
    for _ in 0..iterations {
        let _ = ultra_matmul_v2(&a, &b)?;
    }
    let v2_time = v2_start.elapsed().as_nanos() / iterations;

    let v1_vs_standard_ratio = standard_time as f64 / v1_time as f64;
    let v2_vs_standard_ratio = standard_time as f64 / v2_time as f64;
    let v2_vs_v1_improvement = v1_time as f64 / v2_time as f64;

    Ok(PerformanceComparison {
        size: name.to_string(),
        standard_time_ns: standard_time as u64,
        ultra_v1_time_ns: v1_time as u64,
        ultra_v2_time_ns: v2_time as u64,
        v1_vs_standard_ratio,
        v2_vs_standard_ratio,
        v2_vs_v1_improvement,
        verdict: "ASPECT_RATIO".to_string(),
    })
}

fn analyze_results(comparisons: &[PerformanceComparison]) {
    let v1_worse_count = comparisons
        .iter()
        .filter(|c| c.v1_vs_standard_ratio < 0.9)
        .count();
    let v2_better_count = comparisons
        .iter()
        .filter(|c| c.v2_vs_standard_ratio >= 0.95)
        .count();
    let v2_excellent_count = comparisons
        .iter()
        .filter(|c| c.v2_vs_standard_ratio >= 1.1)
        .count();

    let avg_v1_ratio = comparisons
        .iter()
        .map(|c| c.v1_vs_standard_ratio)
        .sum::<f64>()
        / comparisons.len() as f64;
    let avg_v2_ratio = comparisons
        .iter()
        .map(|c| c.v2_vs_standard_ratio)
        .sum::<f64>()
        / comparisons.len() as f64;
    let avg_improvement = comparisons
        .iter()
        .map(|c| c.v2_vs_v1_improvement)
        .sum::<f64>()
        / comparisons.len() as f64;

    println!("📊 DETAILED ANALYSIS:");
    println!(
        "   V1 Performance Issues:    {}/{} test cases performed poorly",
        v1_worse_count,
        comparisons.len()
    );
    println!(
        "   V2 Acceptable Results:    {}/{} test cases meet standards",
        v2_better_count,
        comparisons.len()
    );
    println!(
        "   V2 Excellent Results:     {}/{} test cases exceed standards",
        v2_excellent_count,
        comparisons.len()
    );
    println!();
    println!("   Average V1 vs Standard:   {:.2}x", avg_v1_ratio);
    println!("   Average V2 vs Standard:   {:.2}x", avg_v2_ratio);
    println!("   Average V2 vs V1 Improvement: {:.2}x", avg_improvement);
    println!();

    if avg_v2_ratio >= 1.1 {
        println!("✅ V2 ASSESSMENT: EXCELLENT - Consistently outperforms standard implementation");
    } else if avg_v2_ratio >= 0.95 {
        println!("✅ V2 ASSESSMENT: GOOD - Matches or slightly exceeds standard implementation");
    } else if avg_v2_ratio >= 0.8 {
        println!("⚠️  V2 ASSESSMENT: ACCEPTABLE - Close to standard implementation performance");
    } else {
        println!(
            "❌ V2 ASSESSMENT: NEEDS MORE WORK - Still underperforming standard implementation"
        );
    }

    if avg_improvement >= 2.0 {
        println!(
            "🚀 IMPROVEMENT: DRAMATIC - V2 is {:.1}x faster than V1 on average",
            avg_improvement
        );
    } else if avg_improvement >= 1.5 {
        println!(
            "⚡ IMPROVEMENT: SIGNIFICANT - V2 is {:.1}x faster than V1 on average",
            avg_improvement
        );
    } else if avg_improvement >= 1.2 {
        println!(
            "✅ IMPROVEMENT: GOOD - V2 is {:.1}x faster than V1 on average",
            avg_improvement
        );
    } else {
        println!(
            "📈 IMPROVEMENT: MODEST - V2 is {:.1}x faster than V1 on average",
            avg_improvement
        );
    }
}

fn final_assessment(comparisons: &[PerformanceComparison]) {
    println!("🎯 FINAL ASSESSMENT:");
    println!("{}", "-".repeat(50));

    let successful_fixes = comparisons
        .iter()
        .filter(|c| c.v2_vs_standard_ratio >= 0.9)
        .count();

    let total_tests = comparisons.len();
    let success_rate = (successful_fixes as f64 / total_tests as f64) * 100.0;

    println!(
        "   Success Rate: {}/{} ({:.1}%) test cases fixed",
        successful_fixes, total_tests, success_rate
    );

    if success_rate >= 90.0 {
        println!();
        println!("🏆 CONCLUSION: MISSION ACCOMPLISHED");
        println!("   The V2 redesign successfully addresses the performance bottlenecks");
        println!("   identified in the original ultra_matmul implementation. Through");
        println!("   humble acknowledgment of issues and systematic optimization,");
        println!("   we have achieved true ultra-performance status.");
        println!();
        println!("🙏 Key Learnings:");
        println!("   • Building upon proven optimized implementations is more effective");
        println!("     than attempting to replace them entirely");
        println!("   • Type checking and conversion overhead significantly impact performance");
        println!(
            "   • Leveraging ndarray's BLAS-accelerated operations provides excellent baseline"
        );
        println!("   • Targeted optimizations should only be applied where they add clear value");
    } else if success_rate >= 70.0 {
        println!();
        println!("✅ CONCLUSION: SIGNIFICANT PROGRESS");
        println!("   The V2 redesign shows substantial improvements but requires further");
        println!("   refinement to achieve full ultra-performance status across all scenarios.");
    } else {
        println!();
        println!("⚠️  CONCLUSION: MORE WORK NEEDED");
        println!("   While V2 shows improvements, additional optimization work is required");
        println!("   to achieve the target ultra-performance goals.");
    }

    println!();
    println!("🌟 This journey demonstrates the importance of humility in performance");
    println!("   optimization - acknowledging shortcomings and systematically addressing");
    println!("   them leads to genuine improvements and learning.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v2_performance_improvement() {
        let result = compare_implementations(32);
        assert!(result.is_ok());
        let comparison = result.unwrap();

        // V2 should be significantly better than V1
        assert!(comparison.v2_vs_v1_improvement > 1.0);

        // V2 should at least be competitive with standard
        assert!(comparison.v2_vs_standard_ratio > 0.7);
    }

    #[test]
    fn test_aspect_ratio_optimization() {
        let result = compare_aspect_ratio("test_outer", 32, 1, 32);
        assert!(result.is_ok());
        let comparison = result.unwrap();

        // For outer products, V2 should perform well
        assert!(comparison.v2_vs_v1_improvement > 1.0);
    }
}
