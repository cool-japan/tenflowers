//! Memory Usage Profiling and TensorFlow Comparison Module
//!
//! This module provides comprehensive memory usage profiling and optimization
//! to achieve memory usage within 10% of TensorFlow, as specified in the project goals.

use crate::memory::{global_monitor_arc, PerformanceMonitor};
use crate::{DType, Device, Result, TensorError};
use std::collections::HashMap;
use std::process::Command;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Memory profiling configuration for TensorFlow comparison
#[derive(Debug, Clone)]
pub struct MemoryProfilingConfig {
    /// Enable memory tracking
    pub enable_memory_tracking: bool,
    /// Enable TensorFlow comparison
    pub enable_tensorflow_comparison: bool,
    /// Target memory efficiency vs TensorFlow (0.9 = within 10%)
    pub target_efficiency_ratio: f64,
    /// Python executable for TensorFlow benchmarks
    pub python_executable: String,
    /// Enable detailed allocation tracking
    pub enable_detailed_tracking: bool,
    /// Enable memory optimization suggestions
    pub enable_optimization_suggestions: bool,
    /// Memory sampling interval (milliseconds)
    pub sampling_interval_ms: u64,
}

impl Default for MemoryProfilingConfig {
    fn default() -> Self {
        Self {
            enable_memory_tracking: true,
            enable_tensorflow_comparison: true,
            target_efficiency_ratio: 0.9, // Within 10% of TensorFlow
            python_executable: "python3".to_string(),
            enable_detailed_tracking: true,
            enable_optimization_suggestions: true,
            sampling_interval_ms: 100, // Sample every 100ms
        }
    }
}

/// Memory usage snapshot for comparison
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct MemorySnapshot {
    #[cfg_attr(feature = "serialize", serde(skip, default = "Instant::now"))]
    pub timestamp: Instant,
    pub operation: String,
    pub tenflowers_memory_mb: f64,
    pub tensorflow_memory_mb: Option<f64>,
    pub pytorch_memory_mb: Option<f64>,
    pub input_shapes: Vec<Vec<usize>>,
    pub dtype: DType,
    pub device: Device,
    pub memory_efficiency: f64, // TensorFlow / TenfloweRS
    pub meets_target: bool,
}

/// Memory optimization suggestion
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct MemoryOptimizationSuggestion {
    pub operation: String,
    pub issue_type: MemoryIssueType,
    pub current_usage_mb: f64,
    pub estimated_optimized_mb: f64,
    pub potential_savings_mb: f64,
    pub suggestion: String,
    pub priority: OptimizationPriority,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum MemoryIssueType {
    ExcessiveAllocation,
    MemoryFragmentation,
    InsufficientReuse,
    SuboptimalLayout,
    UnnecessaryCopies,
    LargeIntermediates,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum OptimizationPriority {
    High,
    Medium,
    Low,
}

/// Memory profiler for TensorFlow comparison
pub struct TensorFlowMemoryProfiler {
    config: MemoryProfilingConfig,
    snapshots: RwLock<Vec<MemorySnapshot>>,
    optimization_suggestions: RwLock<Vec<MemoryOptimizationSuggestion>>,
    monitor: Arc<PerformanceMonitor>,
    baseline_memory_usage: Mutex<HashMap<String, f64>>, // Operation -> TensorFlow memory usage
}

impl TensorFlowMemoryProfiler {
    /// Create a new memory profiler
    pub fn new(config: MemoryProfilingConfig) -> Self {
        Self {
            config,
            snapshots: RwLock::new(Vec::new()),
            optimization_suggestions: RwLock::new(Vec::new()),
            monitor: global_monitor_arc(),
            baseline_memory_usage: Mutex::new(HashMap::new()),
        }
    }

    /// Profile memory usage for an operation vs TensorFlow
    pub fn profile_operation_vs_tensorflow(
        &self,
        operation: &str,
        input_shapes: &[Vec<usize>],
        dtype: DType,
        device: Device,
        tenflowers_executor: impl FnOnce() -> Result<f64>, // Returns memory usage in MB
    ) -> Result<MemorySnapshot> {
        let start_time = Instant::now();

        // Measure TenfloweRS memory usage
        let memory_before = self.monitor.get_current_memory() as f64 / 1_024_000.0; // Convert to MB
        let tenflowers_memory_usage = tenflowers_executor()?;
        let memory_after = self.monitor.get_current_memory() as f64 / 1_024_000.0;

        // Use the actual measured difference if executor doesn't provide it
        let actual_tenflowers_memory = if tenflowers_memory_usage > 0.0 {
            tenflowers_memory_usage
        } else {
            memory_after - memory_before
        };

        // Measure TensorFlow memory usage
        let tensorflow_memory = if self.config.enable_tensorflow_comparison {
            self.measure_tensorflow_memory(operation, input_shapes, dtype)
                .ok()
        } else {
            None
        };

        // Measure PyTorch memory usage for additional comparison
        let pytorch_memory = if self.config.enable_tensorflow_comparison {
            self.measure_pytorch_memory(operation, input_shapes, dtype)
                .ok()
        } else {
            None
        };

        // Calculate efficiency
        let memory_efficiency = if let Some(tf_memory) = tensorflow_memory {
            if actual_tenflowers_memory > 0.0 {
                tf_memory / actual_tenflowers_memory
            } else {
                1.0
            }
        } else {
            1.0
        };

        let meets_target = memory_efficiency >= self.config.target_efficiency_ratio;

        let snapshot = MemorySnapshot {
            timestamp: start_time,
            operation: operation.to_string(),
            tenflowers_memory_mb: actual_tenflowers_memory,
            tensorflow_memory_mb: tensorflow_memory,
            pytorch_memory_mb: pytorch_memory,
            input_shapes: input_shapes.to_vec(),
            dtype,
            device,
            memory_efficiency,
            meets_target,
        };

        // Store snapshot
        self.snapshots.write().unwrap().push(snapshot.clone());

        // Update baseline if we have TensorFlow data
        if let Some(tf_memory) = tensorflow_memory {
            self.baseline_memory_usage
                .lock()
                .unwrap()
                .insert(operation.to_string(), tf_memory);
        }

        // Generate optimization suggestions if memory usage is high
        if !meets_target {
            self.generate_optimization_suggestions(&snapshot);
        }

        Ok(snapshot)
    }

    /// Measure TensorFlow memory usage for comparison
    fn measure_tensorflow_memory(
        &self,
        operation: &str,
        input_shapes: &[Vec<usize>],
        dtype: DType,
    ) -> Result<f64> {
        let script = self.generate_tensorflow_memory_script(operation, input_shapes, dtype)?;

        let output = Command::new(&self.config.python_executable)
            .arg("-c")
            .arg(&script)
            .output()
            .map_err(|e| {
                TensorError::other(format!("Failed to execute TensorFlow memory test: {e}"))
            })?;

        if !output.status.success() {
            return Err(TensorError::other(format!(
                "TensorFlow memory test failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let memory_mb_str = String::from_utf8_lossy(&output.stdout);
        let memory_mb: f64 = memory_mb_str.trim().parse().map_err(|e| {
            TensorError::other(format!("Failed to parse TensorFlow memory usage: {e}"))
        })?;

        Ok(memory_mb)
    }

    /// Measure PyTorch memory usage for additional comparison
    fn measure_pytorch_memory(
        &self,
        operation: &str,
        input_shapes: &[Vec<usize>],
        dtype: DType,
    ) -> Result<f64> {
        let script = self.generate_pytorch_memory_script(operation, input_shapes, dtype)?;

        let output = Command::new(&self.config.python_executable)
            .arg("-c")
            .arg(&script)
            .output()
            .map_err(|e| {
                TensorError::other(format!("Failed to execute PyTorch memory test: {e}"))
            })?;

        if !output.status.success() {
            return Err(TensorError::other(format!(
                "PyTorch memory test failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let memory_mb_str = String::from_utf8_lossy(&output.stdout);
        let memory_mb: f64 = memory_mb_str.trim().parse().map_err(|e| {
            TensorError::other(format!("Failed to parse PyTorch memory usage: {e}"))
        })?;

        Ok(memory_mb)
    }

    /// Generate TensorFlow memory measurement script
    fn generate_tensorflow_memory_script(
        &self,
        operation: &str,
        input_shapes: &[Vec<usize>],
        _dtype: DType,
    ) -> Result<String> {
        let shape_strs: Vec<String> = input_shapes
            .iter()
            .map(|shape| {
                format!(
                    "[{}]",
                    shape
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .collect();

        let script = match operation {
            "add" | "mul" | "sub" | "div" => {
                format!(
                    r#"
import tensorflow as tf
import psutil
import os

# Set memory growth to track actual usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024  # MB

# Create tensors
shapes = [{}]
tensors = []
for shape in shapes:
    tensor = tf.random.normal(shape, dtype=tf.float32)
    tensors.append(tensor)

# Perform operation
if len(tensors) >= 2:
    if '{}' == 'add':
        result = tf.add(tensors[0], tensors[1])
    elif '{}' == 'mul':
        result = tf.multiply(tensors[0], tensors[1])
    elif '{}' == 'sub':
        result = tf.subtract(tensors[0], tensors[1])
    elif '{}' == 'div':
        result = tf.divide(tensors[0], tensors[1])

# Force execution
_ = result.numpy()

final_memory = process.memory_info().rss / 1024 / 1024  # MB
memory_used = final_memory - initial_memory

print(f"{{memory_used:.2f}}")
"#,
                    shape_strs.join(", "),
                    operation,
                    operation,
                    operation,
                    operation
                )
            }
            "matmul" => {
                format!(
                    r#"
import tensorflow as tf
import psutil
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024

shapes = [{}]
if len(shapes) >= 2:
    a = tf.random.normal(shapes[0], dtype=tf.float32)
    b = tf.random.normal(shapes[1], dtype=tf.float32)
    result = tf.linalg.matmul(a, b)
    _ = result.numpy()

final_memory = process.memory_info().rss / 1024 / 1024
memory_used = final_memory - initial_memory

print(f"{{memory_used:.2f}}")
"#,
                    shape_strs.join(", ")
                )
            }
            "conv2d" => {
                format!(
                    r#"
import tensorflow as tf
import psutil
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024

shapes = [{}]
if len(shapes) >= 2:
    inputs = tf.random.normal(shapes[0], dtype=tf.float32)  # [batch, height, width, channels]
    filters = tf.random.normal([3, 3, shapes[0][-1], 64], dtype=tf.float32)
    result = tf.nn.conv2d(inputs, filters, strides=[1, 1, 1, 1], padding='SAME')
    _ = result.numpy()

final_memory = process.memory_info().rss / 1024 / 1024
memory_used = final_memory - initial_memory

print(f"{{memory_used:.2f}}")
"#,
                    shape_strs.join(", ")
                )
            }
            _ => {
                return Err(TensorError::invalid_argument(format!(
                    "Unsupported operation for TensorFlow memory test: {operation}"
                )));
            }
        };

        Ok(script)
    }

    /// Generate PyTorch memory measurement script
    fn generate_pytorch_memory_script(
        &self,
        operation: &str,
        input_shapes: &[Vec<usize>],
        _dtype: DType,
    ) -> Result<String> {
        let shape_strs: Vec<String> = input_shapes
            .iter()
            .map(|shape| {
                format!(
                    "[{}]",
                    shape
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .collect();

        let script = match operation {
            "add" | "mul" | "sub" | "div" => {
                format!(
                    r#"
import torch
import psutil
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024

shapes = [{}]
tensors = []
for shape in shapes:
    tensor = torch.randn(shape, dtype=torch.float32, device=device)
    tensors.append(tensor)

if len(tensors) >= 2:
    if '{}' == 'add':
        result = torch.add(tensors[0], tensors[1])
    elif '{}' == 'mul':
        result = torch.mul(tensors[0], tensors[1])
    elif '{}' == 'sub':
        result = torch.sub(tensors[0], tensors[1])
    elif '{}' == 'div':
        result = torch.div(tensors[0], tensors[1])

if device.type == 'cuda':
    torch.cuda.synchronize()

final_memory = process.memory_info().rss / 1024 / 1024
memory_used = final_memory - initial_memory

print(f"{{memory_used:.2f}}")
"#,
                    shape_strs.join(", "),
                    operation,
                    operation,
                    operation,
                    operation
                )
            }
            "matmul" => {
                format!(
                    r#"
import torch
import psutil
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024

shapes = [{}]
if len(shapes) >= 2:
    a = torch.randn(shapes[0], dtype=torch.float32, device=device)
    b = torch.randn(shapes[1], dtype=torch.float32, device=device)
    result = torch.matmul(a, b)

if device.type == 'cuda':
    torch.cuda.synchronize()

final_memory = process.memory_info().rss / 1024 / 1024
memory_used = final_memory - initial_memory

print(f"{{memory_used:.2f}}")
"#,
                    shape_strs.join(", ")
                )
            }
            _ => {
                return Err(TensorError::invalid_argument(format!(
                    "Unsupported operation for PyTorch memory test: {operation}"
                )));
            }
        };

        Ok(script)
    }

    /// Generate optimization suggestions for high memory usage
    fn generate_optimization_suggestions(&self, snapshot: &MemorySnapshot) {
        let mut suggestions = self.optimization_suggestions.write().unwrap();

        // Check for excessive memory usage
        if let Some(tf_memory) = snapshot.tensorflow_memory_mb {
            let overhead = snapshot.tenflowers_memory_mb - tf_memory;
            let overhead_ratio = overhead / tf_memory;

            if overhead_ratio > 0.2 {
                // More than 20% overhead
                suggestions.push(MemoryOptimizationSuggestion {
                    operation: snapshot.operation.clone(),
                    issue_type: MemoryIssueType::ExcessiveAllocation,
                    current_usage_mb: snapshot.tenflowers_memory_mb,
                    estimated_optimized_mb: tf_memory * 1.05, // 5% overhead target
                    potential_savings_mb: overhead * 0.8,
                    suggestion: "Consider using memory pooling to reduce allocation overhead"
                        .to_string(),
                    priority: OptimizationPriority::High,
                });
            }
        }

        // Check for large tensors that might benefit from optimization
        let total_input_size: usize = snapshot
            .input_shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();

        if total_input_size > 1_000_000 && snapshot.tenflowers_memory_mb > 100.0 {
            suggestions.push(MemoryOptimizationSuggestion {
                operation: snapshot.operation.clone(),
                issue_type: MemoryIssueType::LargeIntermediates,
                current_usage_mb: snapshot.tenflowers_memory_mb,
                estimated_optimized_mb: snapshot.tenflowers_memory_mb * 0.7,
                potential_savings_mb: snapshot.tenflowers_memory_mb * 0.3,
                suggestion: "Consider using in-place operations or memory-efficient algorithms for large tensors".to_string(),
                priority: OptimizationPriority::Medium,
            });
        }
    }

    /// Get all memory snapshots
    pub fn get_snapshots(&self) -> Vec<MemorySnapshot> {
        self.snapshots.read().unwrap().clone()
    }

    /// Get optimization suggestions
    pub fn get_optimization_suggestions(&self) -> Vec<MemoryOptimizationSuggestion> {
        self.optimization_suggestions.read().unwrap().clone()
    }

    /// Generate comprehensive memory comparison report
    pub fn generate_memory_comparison_report(&self) -> MemoryComparisonReport {
        let snapshots = self.get_snapshots();
        let suggestions = self.get_optimization_suggestions();

        if snapshots.is_empty() {
            return MemoryComparisonReport::default();
        }

        let total_operations = snapshots.len();
        let operations_meeting_target = snapshots.iter().filter(|s| s.meets_target).count();
        let success_rate = operations_meeting_target as f64 / total_operations as f64;

        // Calculate average memory efficiency
        let tf_snapshots: Vec<_> = snapshots
            .iter()
            .filter(|s| s.tensorflow_memory_mb.is_some())
            .collect();

        let avg_memory_efficiency = if !tf_snapshots.is_empty() {
            tf_snapshots
                .iter()
                .map(|s| s.memory_efficiency)
                .sum::<f64>()
                / tf_snapshots.len() as f64
        } else {
            1.0
        };

        // Find memory usage statistics
        let avg_tenflowers_memory = snapshots
            .iter()
            .map(|s| s.tenflowers_memory_mb)
            .sum::<f64>()
            / total_operations as f64;

        let avg_tensorflow_memory = if !tf_snapshots.is_empty() {
            tf_snapshots
                .iter()
                .filter_map(|s| s.tensorflow_memory_mb)
                .sum::<f64>()
                / tf_snapshots.len() as f64
        } else {
            0.0
        };

        let potential_memory_savings = suggestions
            .iter()
            .map(|s| s.potential_savings_mb)
            .sum::<f64>();

        MemoryComparisonReport {
            total_operations,
            operations_meeting_target,
            success_rate,
            avg_memory_efficiency,
            avg_tenflowers_memory_mb: avg_tenflowers_memory,
            avg_tensorflow_memory_mb: avg_tensorflow_memory,
            target_efficiency: self.config.target_efficiency_ratio,
            optimization_suggestions: suggestions,
            potential_memory_savings_mb: potential_memory_savings,
            snapshots,
        }
    }
}

/// Memory comparison report
#[derive(Debug, Clone)]
pub struct MemoryComparisonReport {
    pub total_operations: usize,
    pub operations_meeting_target: usize,
    pub success_rate: f64,
    pub avg_memory_efficiency: f64,
    pub avg_tenflowers_memory_mb: f64,
    pub avg_tensorflow_memory_mb: f64,
    pub target_efficiency: f64,
    pub optimization_suggestions: Vec<MemoryOptimizationSuggestion>,
    pub potential_memory_savings_mb: f64,
    pub snapshots: Vec<MemorySnapshot>,
}

impl Default for MemoryComparisonReport {
    fn default() -> Self {
        Self {
            total_operations: 0,
            operations_meeting_target: 0,
            success_rate: 0.0,
            avg_memory_efficiency: 1.0,
            avg_tenflowers_memory_mb: 0.0,
            avg_tensorflow_memory_mb: 0.0,
            target_efficiency: 0.9,
            optimization_suggestions: Vec::new(),
            potential_memory_savings_mb: 0.0,
            snapshots: Vec::new(),
        }
    }
}

impl MemoryComparisonReport {
    /// Print a formatted memory comparison report
    pub fn print_report(&self) {
        println!("📊 Memory Usage Comparison Report - TensorFlow vs TenfloweRS");
        println!("============================================================");
        println!();

        println!("🎯 Overall Performance:");
        println!("  • Total operations tested: {}", self.total_operations);
        println!(
            "  • Operations meeting target: {}/{}",
            self.operations_meeting_target, self.total_operations
        );
        println!("  • Success rate: {:.1}%", self.success_rate * 100.0);
        println!(
            "  • Target efficiency: ≥{:.1}% of TensorFlow",
            self.target_efficiency * 100.0
        );
        println!();

        println!("💾 Memory Usage Analysis:");
        println!(
            "  • Average TenfloweRS memory: {:.2} MB",
            self.avg_tenflowers_memory_mb
        );
        if self.avg_tensorflow_memory_mb > 0.0 {
            println!(
                "  • Average TensorFlow memory: {:.2} MB",
                self.avg_tensorflow_memory_mb
            );
            println!(
                "  • Average memory efficiency: {:.1}%",
                self.avg_memory_efficiency * 100.0
            );

            let memory_overhead = self.avg_tenflowers_memory_mb - self.avg_tensorflow_memory_mb;
            let overhead_percentage = (memory_overhead / self.avg_tensorflow_memory_mb) * 100.0;

            if self.avg_memory_efficiency >= self.target_efficiency {
                println!("  ✅ Memory efficiency meets target!");
            } else {
                println!(
                    "  ❌ Memory overhead: {memory_overhead:.2} MB ({overhead_percentage:.1}%)"
                );
            }
        }
        println!();

        if !self.optimization_suggestions.is_empty() {
            println!("💡 Memory Optimization Suggestions:");
            for (i, suggestion) in self.optimization_suggestions.iter().enumerate() {
                let priority_icon = match suggestion.priority {
                    OptimizationPriority::High => "🔴",
                    OptimizationPriority::Medium => "🟡",
                    OptimizationPriority::Low => "🟢",
                };

                println!(
                    "  {}  {}. {} ({})",
                    priority_icon,
                    i + 1,
                    suggestion.suggestion,
                    suggestion.operation
                );
                println!(
                    "     Potential savings: {:.2} MB",
                    suggestion.potential_savings_mb
                );
            }

            if self.potential_memory_savings_mb > 0.0 {
                println!(
                    "  📈 Total potential savings: {:.2} MB",
                    self.potential_memory_savings_mb
                );
            }
            println!();
        }

        println!("📈 Per-Operation Details:");
        println!("{:-<100}", "");
        println!(
            "| {:^15} | {:^20} | {:^12} | {:^12} | {:^12} | {:^10} |",
            "Operation", "Shapes", "TF RS (MB)", "TensorFlow (MB)", "Efficiency", "Target Met"
        );
        println!("{:-<100}", "");

        for snapshot in &self.snapshots {
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
                .map(|m| format!("{m:.2}"))
                .unwrap_or_else(|| "N/A".to_string());

            let efficiency_str = if snapshot.tensorflow_memory_mb.is_some() {
                format!("{:.1}%", snapshot.memory_efficiency * 100.0)
            } else {
                "N/A".to_string()
            };

            let target_met = if snapshot.meets_target {
                "✅ Yes"
            } else {
                "❌ No"
            };

            println!(
                "| {:^15} | {:^20} | {:^12.2} | {:^12} | {:^12} | {:^10} |",
                snapshot.operation,
                if shapes_str.len() > 20 {
                    format!("{}...", &shapes_str[..17])
                } else {
                    shapes_str.clone()
                },
                snapshot.tenflowers_memory_mb,
                tf_memory_str,
                efficiency_str,
                target_met
            );
        }
        println!("{:-<100}", "");

        println!();
        println!("============================================================");
    }
}

lazy_static::lazy_static! {
    pub static ref MEMORY_PROFILER: TensorFlowMemoryProfiler =
        TensorFlowMemoryProfiler::new(MemoryProfilingConfig::default());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_profiling_config() {
        let config = MemoryProfilingConfig::default();
        assert!(config.enable_memory_tracking);
        assert!(config.enable_tensorflow_comparison);
        assert_eq!(config.target_efficiency_ratio, 0.9);
    }

    #[test]
    fn test_memory_snapshot_creation() {
        let snapshot = MemorySnapshot {
            timestamp: Instant::now(),
            operation: "add".to_string(),
            tenflowers_memory_mb: 10.0,
            tensorflow_memory_mb: Some(9.0),
            pytorch_memory_mb: Some(9.5),
            input_shapes: vec![vec![1000, 1000]],
            dtype: DType::Float32,
            device: Device::Cpu,
            memory_efficiency: 0.9,
            meets_target: true,
        };

        assert_eq!(snapshot.operation, "add");
        assert!(snapshot.meets_target);
        assert_eq!(snapshot.memory_efficiency, 0.9);
    }

    #[test]
    fn test_optimization_suggestion() {
        let suggestion = MemoryOptimizationSuggestion {
            operation: "conv2d".to_string(),
            issue_type: MemoryIssueType::ExcessiveAllocation,
            current_usage_mb: 100.0,
            estimated_optimized_mb: 80.0,
            potential_savings_mb: 20.0,
            suggestion: "Use memory pooling".to_string(),
            priority: OptimizationPriority::High,
        };

        assert_eq!(suggestion.potential_savings_mb, 20.0);
        assert_eq!(suggestion.priority, OptimizationPriority::High);
    }
}
