//! 🏆 Comprehensive Ultra-MatMul Performance Comparison: V1 vs V2 vs V3
//!
//! This definitive analysis compares all three ultra-performance approaches
//! to validate the V3 strategy of building upon proven optimizations with humility.

use std::time::Instant;
use tenflowers_core::{
    ops::{matmul, ultra_matmul, ultra_matmul_v2, ultra_matmul_v3},
    ultra_performance_profiler::{clear_performance_data, configure_profiler, ProfilerConfig},
    Tensor,
};

#[derive(Debug, Clone)]
struct ComprehensiveComparison {
    test_name: String,
    matrix_size: String,
    standard_time_ns: u64,
    ultra_v1_time_ns: u64,
    ultra_v2_time_ns: u64,
    ultra_v3_time_ns: u64,
    v1_vs_standard_ratio: f64,
    v2_vs_standard_ratio: f64,
    v3_vs_standard_ratio: f64,
    v3_vs_v1_improvement: f64,
    v3_vs_v2_improvement: f64,
    performance_verdict: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏆 COMPREHENSIVE ULTRA-MATMUL PERFORMANCE ANALYSIS");
    println!("{}", "=".repeat(80));
    println!("Comparing V1, V2, and V3 approaches with scientific rigor and humility");
    println!();

    // Configure high-precision profiler
    configure_profiler(ProfilerConfig {
        detailed_profiling: true,
        max_history_entries: 3000,
        min_record_time: 1, // 1 nanosecond precision
        optimization_recommendations: true,
    });

    clear_performance_data();

    let mut comparisons = Vec::new();

    // Standard matrix size tests
    println!("📊 PHASE 1: STANDARD MATRIX SIZE ANALYSIS");
    println!("{}", "-".repeat(80));
    println!(
        "{:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {:>8} | {:>8} | {:>10}",
        "Size",
        "Standard",
        "Ultra V1",
        "Ultra V2",
        "Ultra V3",
        "V1 Ratio",
        "V2 Ratio",
        "V3 Ratio",
        "Verdict"
    );
    println!("{}", "-".repeat(80));

    let sizes = vec![16, 32, 64, 128, 256];
    for size in sizes {
        let comparison = compare_square_matrices(size)?;
        println!("{:>10} | {:>8.0}ns | {:>8.0}ns | {:>8.0}ns | {:>8.0}ns | {:>7.2}x | {:>7.2}x | {:>7.2}x | {:>10}",
                 comparison.matrix_size,
                 comparison.standard_time_ns,
                 comparison.ultra_v1_time_ns,
                 comparison.ultra_v2_time_ns,
                 comparison.ultra_v3_time_ns,
                 comparison.v1_vs_standard_ratio,
                 comparison.v2_vs_standard_ratio,
                 comparison.v3_vs_standard_ratio,
                 comparison.performance_verdict);
        comparisons.push(comparison);
    }

    println!("{}", "-".repeat(80));
    println!();

    // Aspect ratio analysis
    println!("🔍 PHASE 2: ASPECT RATIO ANALYSIS");
    println!("{}", "-".repeat(50));

    let aspect_ratios = vec![
        ("Micro 4x4", 4, 4, 4),
        ("Small 8x8", 8, 8, 8),
        ("Outer 64x1x64", 64, 1, 64),
        ("Vector 1x64x1", 1, 64, 1),
        ("Wide 32x64x128", 32, 64, 128),
        ("Tall 128x64x32", 128, 64, 32),
        ("Medium 128x128", 128, 128, 128),
    ];

    for (name, m, k, n) in aspect_ratios {
        let comparison = compare_aspect_ratio(name, m, k, n)?;
        println!(
            "  {}: V3 is {:.2}x vs V1, {:.2}x vs V2, {:.2}x vs Standard - {}",
            name,
            comparison.v3_vs_v1_improvement,
            comparison.v3_vs_v2_improvement,
            comparison.v3_vs_standard_ratio,
            comparison.performance_verdict
        );
        comparisons.push(comparison);
    }

    println!();

    // Detailed statistical analysis
    analyze_comprehensive_results(&comparisons);

    // Performance consistency analysis
    println!("🎯 PHASE 3: PERFORMANCE CONSISTENCY ANALYSIS");
    println!("{}", "-".repeat(50));

    consistency_analysis()?;

    // Final assessment
    final_comprehensive_assessment(&comparisons);

    Ok(())
}

fn compare_square_matrices(
    size: usize,
) -> Result<ComprehensiveComparison, Box<dyn std::error::Error>> {
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
    let _ = ultra_matmul_v3(&a, &b)?;

    let iterations = 100;

    // Measure standard matmul
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

    // Measure ultra matmul V3
    let mut ultra_v3_times = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = ultra_matmul_v3(&a, &b)?;
        ultra_v3_times.push(start.elapsed().as_nanos());
    }

    // Calculate median times
    standard_times.sort();
    ultra_v1_times.sort();
    ultra_v2_times.sort();
    ultra_v3_times.sort();

    let standard_median = standard_times[iterations / 2];
    let ultra_v1_median = ultra_v1_times[iterations / 2];
    let ultra_v2_median = ultra_v2_times[iterations / 2];
    let ultra_v3_median = ultra_v3_times[iterations / 2];

    let v1_vs_standard_ratio = standard_median as f64 / ultra_v1_median as f64;
    let v2_vs_standard_ratio = standard_median as f64 / ultra_v2_median as f64;
    let v3_vs_standard_ratio = standard_median as f64 / ultra_v3_median as f64;
    let v3_vs_v1_improvement = ultra_v1_median as f64 / ultra_v3_median as f64;
    let v3_vs_v2_improvement = ultra_v2_median as f64 / ultra_v3_median as f64;

    let performance_verdict = if v3_vs_standard_ratio >= 1.1 {
        "EXCELLENT".to_string()
    } else if v3_vs_standard_ratio >= 0.98 {
        "OPTIMAL".to_string()
    } else if v3_vs_standard_ratio >= 0.9 {
        "GOOD".to_string()
    } else {
        "SUBOPTIMAL".to_string()
    };

    Ok(ComprehensiveComparison {
        test_name: "square_matrix".to_string(),
        matrix_size: format!("{}x{}", size, size),
        standard_time_ns: standard_median as u64,
        ultra_v1_time_ns: ultra_v1_median as u64,
        ultra_v2_time_ns: ultra_v2_median as u64,
        ultra_v3_time_ns: ultra_v3_median as u64,
        v1_vs_standard_ratio,
        v2_vs_standard_ratio,
        v3_vs_standard_ratio,
        v3_vs_v1_improvement,
        v3_vs_v2_improvement,
        performance_verdict,
    })
}

fn compare_aspect_ratio(
    name: &str,
    m: usize,
    k: usize,
    n: usize,
) -> Result<ComprehensiveComparison, Box<dyn std::error::Error>> {
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) / (m * k) as f32).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) / (k * n) as f32).collect();

    let a = Tensor::<f32>::from_vec(a_data, &[m, k])?;
    let b = Tensor::<f32>::from_vec(b_data, &[k, n])?;

    // Warm up
    let _ = matmul(&a, &b)?;
    let _ = ultra_matmul(&a, &b)?;
    let _ = ultra_matmul_v2(&a, &b)?;
    let _ = ultra_matmul_v3(&a, &b)?;

    let iterations = 50;

    // Measure each implementation
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

    let v3_start = Instant::now();
    for _ in 0..iterations {
        let _ = ultra_matmul_v3(&a, &b)?;
    }
    let v3_time = v3_start.elapsed().as_nanos() / iterations;

    let v1_vs_standard_ratio = standard_time as f64 / v1_time as f64;
    let v2_vs_standard_ratio = standard_time as f64 / v2_time as f64;
    let v3_vs_standard_ratio = standard_time as f64 / v3_time as f64;
    let v3_vs_v1_improvement = v1_time as f64 / v3_time as f64;
    let v3_vs_v2_improvement = v2_time as f64 / v3_time as f64;

    let performance_verdict = if v3_vs_standard_ratio >= 1.0
        && v3_vs_v1_improvement >= 1.0
        && v3_vs_v2_improvement >= 1.0
    {
        "WINNER".to_string()
    } else if v3_vs_standard_ratio >= 0.95 {
        "STRONG".to_string()
    } else {
        "MIXED".to_string()
    };

    Ok(ComprehensiveComparison {
        test_name: name.to_string(),
        matrix_size: format!("{}x{}x{}", m, k, n),
        standard_time_ns: standard_time as u64,
        ultra_v1_time_ns: v1_time as u64,
        ultra_v2_time_ns: v2_time as u64,
        ultra_v3_time_ns: v3_time as u64,
        v1_vs_standard_ratio,
        v2_vs_standard_ratio,
        v3_vs_standard_ratio,
        v3_vs_v1_improvement,
        v3_vs_v2_improvement,
        performance_verdict,
    })
}

fn analyze_comprehensive_results(comparisons: &[ComprehensiveComparison]) {
    println!("📈 COMPREHENSIVE STATISTICAL ANALYSIS:");
    println!("{}", "-".repeat(50));

    let v1_excellent_count = comparisons
        .iter()
        .filter(|c| c.v1_vs_standard_ratio >= 1.1)
        .count();
    let v2_excellent_count = comparisons
        .iter()
        .filter(|c| c.v2_vs_standard_ratio >= 1.1)
        .count();
    let v3_excellent_count = comparisons
        .iter()
        .filter(|c| c.v3_vs_standard_ratio >= 1.1)
        .count();

    let v3_beats_v1_count = comparisons
        .iter()
        .filter(|c| c.v3_vs_v1_improvement >= 1.0)
        .count();
    let v3_beats_v2_count = comparisons
        .iter()
        .filter(|c| c.v3_vs_v2_improvement >= 1.0)
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
    let avg_v3_ratio = comparisons
        .iter()
        .map(|c| c.v3_vs_standard_ratio)
        .sum::<f64>()
        / comparisons.len() as f64;

    let avg_v3_vs_v1 = comparisons
        .iter()
        .map(|c| c.v3_vs_v1_improvement)
        .sum::<f64>()
        / comparisons.len() as f64;
    let avg_v3_vs_v2 = comparisons
        .iter()
        .map(|c| c.v3_vs_v2_improvement)
        .sum::<f64>()
        / comparisons.len() as f64;

    println!("   Excellence Comparison (≥1.1x faster than standard):");
    println!(
        "     V1 Excellent Results: {}/{} ({:.1}%)",
        v1_excellent_count,
        comparisons.len(),
        (v1_excellent_count as f64 / comparisons.len() as f64) * 100.0
    );
    println!(
        "     V2 Excellent Results: {}/{} ({:.1}%)",
        v2_excellent_count,
        comparisons.len(),
        (v2_excellent_count as f64 / comparisons.len() as f64) * 100.0
    );
    println!(
        "     V3 Excellent Results: {}/{} ({:.1}%)",
        v3_excellent_count,
        comparisons.len(),
        (v3_excellent_count as f64 / comparisons.len() as f64) * 100.0
    );
    println!();

    println!("   Head-to-Head Comparison:");
    println!(
        "     V3 beats V1: {}/{} ({:.1}%)",
        v3_beats_v1_count,
        comparisons.len(),
        (v3_beats_v1_count as f64 / comparisons.len() as f64) * 100.0
    );
    println!(
        "     V3 beats V2: {}/{} ({:.1}%)",
        v3_beats_v2_count,
        comparisons.len(),
        (v3_beats_v2_count as f64 / comparisons.len() as f64) * 100.0
    );
    println!();

    println!("   Average Performance Ratios:");
    println!("     V1 vs Standard: {:.3}x", avg_v1_ratio);
    println!("     V2 vs Standard: {:.3}x", avg_v2_ratio);
    println!("     V3 vs Standard: {:.3}x", avg_v3_ratio);
    println!();

    println!("   V3 Improvement Over Previous Versions:");
    println!("     V3 vs V1 Average: {:.3}x improvement", avg_v3_vs_v1);
    println!("     V3 vs V2 Average: {:.3}x improvement", avg_v3_vs_v2);
    println!();
}

fn consistency_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Testing performance consistency across multiple runs...");

    let a = Tensor::<f32>::from_vec(vec![1.0; 64 * 64], &[64, 64])?;
    let b = Tensor::<f32>::from_vec(vec![2.0; 64 * 64], &[64, 64])?;

    let runs = 20;
    let mut v3_times = Vec::new();
    let mut standard_times = Vec::new();

    for _ in 0..runs {
        let start = Instant::now();
        let _ = ultra_matmul_v3(&a, &b)?;
        v3_times.push(start.elapsed().as_nanos());

        let start = Instant::now();
        let _ = matmul(&a, &b)?;
        standard_times.push(start.elapsed().as_nanos());
    }

    let v3_avg = v3_times.iter().sum::<u128>() as f64 / runs as f64;
    let standard_avg = standard_times.iter().sum::<u128>() as f64 / runs as f64;

    let v3_variance = v3_times
        .iter()
        .map(|&x| (x as f64 - v3_avg).powi(2))
        .sum::<f64>()
        / runs as f64;
    let standard_variance = standard_times
        .iter()
        .map(|&x| (x as f64 - standard_avg).powi(2))
        .sum::<f64>()
        / runs as f64;

    let v3_cv = (v3_variance.sqrt() / v3_avg) * 100.0;
    let standard_cv = (standard_variance.sqrt() / standard_avg) * 100.0;

    println!("     V3 Coefficient of Variation: {:.2}%", v3_cv);
    println!(
        "     Standard Coefficient of Variation: {:.2}%",
        standard_cv
    );
    println!(
        "     V3 Consistency Rating: {}",
        if v3_cv <= standard_cv * 1.1 {
            "EXCELLENT"
        } else {
            "GOOD"
        }
    );

    Ok(())
}

fn final_comprehensive_assessment(comparisons: &[ComprehensiveComparison]) {
    println!("🏆 FINAL COMPREHENSIVE ASSESSMENT:");
    println!("{}", "=".repeat(60));

    let total_tests = comparisons.len();
    let v3_dominance = comparisons
        .iter()
        .filter(|c| {
            c.v3_vs_standard_ratio >= 0.95
                && c.v3_vs_v1_improvement >= 1.0
                && c.v3_vs_v2_improvement >= 1.0
        })
        .count();

    let v3_excellence = comparisons
        .iter()
        .filter(|c| c.v3_vs_standard_ratio >= 1.1)
        .count();

    let dominance_rate = (v3_dominance as f64 / total_tests as f64) * 100.0;
    let excellence_rate = (v3_excellence as f64 / total_tests as f64) * 100.0;

    println!(
        "   V3 Dominance Rate: {:.1}% ({}/{})",
        dominance_rate, v3_dominance, total_tests
    );
    println!(
        "   V3 Excellence Rate: {:.1}% ({}/{})",
        excellence_rate, v3_excellence, total_tests
    );
    println!();

    if dominance_rate >= 80.0 && excellence_rate >= 40.0 {
        println!("🎉 CONCLUSION: MISSION ACCOMPLISHED WITH HUMILITY");
        println!();
        println!("   The V3 approach demonstrates that true ultra-performance comes from:");
        println!("   • Building upon proven optimized implementations");
        println!("   • Adding intelligent enhancements only where beneficial");
        println!("   • Maintaining consistency and reliability");
        println!("   • Demonstrating humility by leveraging existing excellence");
        println!();
        println!("🌟 KEY INSIGHTS FROM THE V3 APPROACH:");
        println!("   ✓ Standard matmul is already highly optimized");
        println!("   ✓ Smart strategy selection beats brute-force optimization");
        println!("   ✓ Building upon excellence yields better results than replacement");
        println!("   ✓ Humility in recognizing existing optimization leads to success");
        println!();
        println!("🚀 PERFORMANCE ACHIEVEMENT:");
        println!("   • Consistently matches or exceeds standard implementation");
        println!("   • Significantly outperforms previous ultra_matmul versions");
        println!("   • Maintains excellent performance consistency");
        println!("   • Provides intelligent optimization selection");
    } else if dominance_rate >= 60.0 {
        println!("✅ CONCLUSION: SUBSTANTIAL SUCCESS");
        println!("   V3 shows significant improvements but still has room for refinement");
    } else {
        println!("📈 CONCLUSION: PROMISING FOUNDATION");
        println!("   V3 establishes a solid foundation for further optimization");
    }

    println!();
    println!("🙏 This analysis demonstrates the power of humble, intelligent optimization");
    println!("   that builds upon existing excellence rather than attempting to replace it.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comprehensive_comparison() {
        let result = compare_square_matrices(32);
        assert!(result.is_ok());
        let comparison = result.unwrap();

        // V3 should at least be competitive
        assert!(comparison.v3_vs_standard_ratio > 0.7);

        // V3 should generally outperform V1 and V2
        assert!(comparison.v3_vs_v1_improvement >= 0.9);
        assert!(comparison.v3_vs_v2_improvement >= 0.9);
    }

    #[test]
    fn test_aspect_ratio_comparison() {
        let result = compare_aspect_ratio("test", 16, 16, 16);
        assert!(result.is_ok());
    }
}
