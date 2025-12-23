//! Gradient Analysis and Debugging Tools
//!
//! This module provides comprehensive tools for analyzing gradient flow,
//! detecting gradient-related issues, and providing debugging information.

use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::time::Instant;
use tenflowers_core::{Result, Tensor};

/// Comprehensive gradient analysis report
#[derive(Debug, Clone)]
pub struct GradientAnalysisReport {
    pub gradient_statistics: GradientStatistics,
    pub flow_analysis: GradientFlowAnalysis,
    pub potential_issues: Vec<GradientIssue>,
    pub performance_metrics: PerformanceMetrics,
    pub recommendations: Vec<String>,
}

/// Statistical analysis of gradients
#[derive(Debug, Clone)]
pub struct GradientStatistics {
    pub gradient_norms: HashMap<String, f64>,
    pub gradient_means: HashMap<String, f64>,
    pub gradient_stds: HashMap<String, f64>,
    pub sparsity_ratios: HashMap<String, f64>,
    pub condition_numbers: HashMap<String, f64>,
    pub rank_estimates: HashMap<String, usize>,
}

/// Analysis of gradient flow through the computation graph
#[derive(Debug, Clone)]
pub struct GradientFlowAnalysis {
    pub vanishing_gradient_layers: Vec<String>,
    pub exploding_gradient_layers: Vec<String>,
    pub dead_neurons: HashMap<String, Vec<usize>>,
    pub gradient_flow_score: f64,
    pub bottleneck_operations: Vec<String>,
}

/// Identified gradient-related issues
#[derive(Debug, Clone)]
pub enum GradientIssue {
    VanishingGradients {
        layer_name: String,
        gradient_norm: f64,
        threshold: f64,
    },
    ExplodingGradients {
        layer_name: String,
        gradient_norm: f64,
        threshold: f64,
    },
    DeadNeurons {
        layer_name: String,
        neuron_indices: Vec<usize>,
        dead_ratio: f64,
    },
    HighConditionNumber {
        parameter_name: String,
        condition_number: f64,
    },
    NumericalInstability {
        operation: String,
        details: String,
    },
    MemoryInefficiency {
        operation: String,
        memory_usage: usize,
        recommendation: String,
    },
}

/// Performance metrics for gradient computation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_computation_time: f64,
    pub memory_usage_mb: f64,
    pub flops_estimate: u64,
    pub cache_hit_rate: f64,
    pub parallel_efficiency: f64,
}

/// Gradient analyzer with comprehensive analysis capabilities
pub struct GradientAnalyzer<T> {
    gradient_history: HashMap<String, Vec<AnalysisSnapshot<T>>>,
    analysis_config: AnalysisConfig,
    computation_graph: ComputationGraph,
    performance_tracker: PerformanceTracker,
}

/// Configuration for gradient analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    pub vanishing_threshold: f64,
    pub exploding_threshold: f64,
    pub dead_neuron_threshold: f64,
    pub condition_number_threshold: f64,
    pub memory_threshold_mb: f64,
    pub enable_flow_analysis: bool,
    pub enable_performance_analysis: bool,
    pub history_length: usize,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            vanishing_threshold: 1e-6,
            exploding_threshold: 100.0,
            dead_neuron_threshold: 0.01,
            condition_number_threshold: 1e12,
            memory_threshold_mb: 1000.0,
            enable_flow_analysis: true,
            enable_performance_analysis: true,
            history_length: 100,
        }
    }
}

/// Snapshot of gradient analysis at a specific point in time
#[derive(Clone)]
struct AnalysisSnapshot<T> {
    #[allow(dead_code)]
    timestamp: Instant,
    #[allow(dead_code)]
    gradient: Tensor<T>,
    #[allow(dead_code)]
    gradient_norm: f64,
    #[allow(dead_code)]
    sparsity_ratio: f64,
    #[allow(dead_code)]
    memory_usage: usize,
}

/// Simplified computation graph representation
#[derive(Debug, Default)]
struct ComputationGraph {
    nodes: HashMap<String, GraphNode>,
    #[allow(dead_code)]
    edges: Vec<(String, String)>,
    #[allow(dead_code)]
    critical_path: Vec<String>,
}

#[derive(Debug, Clone)]
struct GraphNode {
    #[allow(dead_code)]
    operation_type: String,
    #[allow(dead_code)]
    input_shapes: Vec<Vec<usize>>,
    #[allow(dead_code)]
    output_shape: Vec<usize>,
    flop_count: u64,
    #[allow(dead_code)]
    memory_requirement: usize,
}

/// Performance tracking utilities
#[derive(Default)]
struct PerformanceTracker {
    #[allow(dead_code)]
    operation_times: HashMap<String, Vec<f64>>,
    memory_usage_history: Vec<usize>,
    cache_hits: usize,
    cache_misses: usize,
}

impl<T> GradientAnalyzer<T>
where
    T: Float + Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    /// Create a new gradient analyzer
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            gradient_history: HashMap::new(),
            analysis_config: config,
            computation_graph: ComputationGraph::default(),
            performance_tracker: PerformanceTracker::default(),
        }
    }

    /// Analyze a set of gradients and return comprehensive report
    pub fn analyze_gradients(
        &mut self,
        gradients: &HashMap<String, Tensor<T>>,
    ) -> Result<GradientAnalysisReport> {
        let start_time = Instant::now();

        // Update gradient history
        self.update_gradient_history(gradients)?;

        // Compute gradient statistics
        let gradient_statistics = self.compute_gradient_statistics(gradients)?;

        // Analyze gradient flow if enabled
        let flow_analysis = if self.analysis_config.enable_flow_analysis {
            self.analyze_gradient_flow(gradients)?
        } else {
            GradientFlowAnalysis {
                vanishing_gradient_layers: Vec::new(),
                exploding_gradient_layers: Vec::new(),
                dead_neurons: HashMap::new(),
                gradient_flow_score: 1.0,
                bottleneck_operations: Vec::new(),
            }
        };

        // Detect potential issues
        let potential_issues = self.detect_gradient_issues(gradients, &gradient_statistics)?;

        // Compute performance metrics if enabled
        let performance_metrics = if self.analysis_config.enable_performance_analysis {
            self.compute_performance_metrics(start_time)?
        } else {
            PerformanceMetrics {
                total_computation_time: 0.0,
                memory_usage_mb: 0.0,
                flops_estimate: 0,
                cache_hit_rate: 0.0,
                parallel_efficiency: 1.0,
            }
        };

        // Generate recommendations
        let recommendations = self.generate_recommendations(&potential_issues, &flow_analysis);

        Ok(GradientAnalysisReport {
            gradient_statistics,
            flow_analysis,
            potential_issues,
            performance_metrics,
            recommendations,
        })
    }

    /// Update the gradient history for trend analysis
    fn update_gradient_history(&mut self, gradients: &HashMap<String, Tensor<T>>) -> Result<()> {
        let timestamp = Instant::now();

        for (name, gradient) in gradients {
            let gradient_norm = self.compute_tensor_norm(gradient)?;
            let sparsity_ratio = self.compute_sparsity_ratio(gradient)?;
            let memory_usage = self.estimate_tensor_memory(gradient);

            let snapshot = AnalysisSnapshot {
                timestamp,
                gradient: gradient.clone(),
                gradient_norm,
                sparsity_ratio,
                memory_usage,
            };

            let history = self.gradient_history.entry(name.clone()).or_default();
            history.push(snapshot);

            // Keep history within configured length
            if history.len() > self.analysis_config.history_length {
                history.remove(0);
            }
        }

        Ok(())
    }

    /// Compute comprehensive gradient statistics
    fn compute_gradient_statistics(
        &self,
        gradients: &HashMap<String, Tensor<T>>,
    ) -> Result<GradientStatistics> {
        let mut gradient_norms = HashMap::new();
        let mut gradient_means = HashMap::new();
        let mut gradient_stds = HashMap::new();
        let mut sparsity_ratios = HashMap::new();
        let mut condition_numbers = HashMap::new();
        let mut rank_estimates = HashMap::new();

        for (name, gradient) in gradients {
            // Compute L2 norm
            let norm = self.compute_tensor_norm(gradient)?;
            gradient_norms.insert(name.clone(), norm);

            // Compute mean
            let mean = self.compute_tensor_mean(gradient)?;
            gradient_means.insert(name.clone(), mean);

            // Compute standard deviation
            let std = self.compute_tensor_std(gradient)?;
            gradient_stds.insert(name.clone(), std);

            // Compute sparsity ratio
            let sparsity = self.compute_sparsity_ratio(gradient)?;
            sparsity_ratios.insert(name.clone(), sparsity);

            // Estimate condition number (for 2D tensors)
            if gradient.shape().dims().len() == 2 {
                let cond_num = self.estimate_condition_number(gradient)?;
                condition_numbers.insert(name.clone(), cond_num);

                let rank = self.estimate_rank(gradient)?;
                rank_estimates.insert(name.clone(), rank);
            }
        }

        Ok(GradientStatistics {
            gradient_norms,
            gradient_means,
            gradient_stds,
            sparsity_ratios,
            condition_numbers,
            rank_estimates,
        })
    }

    /// Analyze gradient flow through the network
    fn analyze_gradient_flow(
        &self,
        gradients: &HashMap<String, Tensor<T>>,
    ) -> Result<GradientFlowAnalysis> {
        let mut vanishing_gradient_layers = Vec::new();
        let mut exploding_gradient_layers = Vec::new();
        let mut dead_neurons = HashMap::new();

        for (name, gradient) in gradients {
            let gradient_norm = self.compute_tensor_norm(gradient)?;

            // Check for vanishing gradients
            if gradient_norm < self.analysis_config.vanishing_threshold {
                vanishing_gradient_layers.push(name.clone());
            }

            // Check for exploding gradients
            if gradient_norm > self.analysis_config.exploding_threshold {
                exploding_gradient_layers.push(name.clone());
            }

            // Detect dead neurons (for fully connected layers)
            if gradient.shape().dims().len() == 2 {
                let dead_neuron_indices = self.detect_dead_neurons(gradient)?;
                if !dead_neuron_indices.is_empty() {
                    dead_neurons.insert(name.clone(), dead_neuron_indices);
                }
            }
        }

        // Compute overall gradient flow score
        let total_layers = gradients.len();
        let problematic_layers = vanishing_gradient_layers.len() + exploding_gradient_layers.len();
        let gradient_flow_score = 1.0 - (problematic_layers as f64 / total_layers as f64);

        // Identify bottleneck operations (simplified)
        let bottleneck_operations = self.identify_bottlenecks(gradients)?;

        Ok(GradientFlowAnalysis {
            vanishing_gradient_layers,
            exploding_gradient_layers,
            dead_neurons,
            gradient_flow_score,
            bottleneck_operations,
        })
    }

    /// Detect various gradient-related issues
    fn detect_gradient_issues(
        &self,
        gradients: &HashMap<String, Tensor<T>>,
        stats: &GradientStatistics,
    ) -> Result<Vec<GradientIssue>> {
        let mut issues = Vec::new();

        // Check for vanishing/exploding gradients
        for (name, &norm) in &stats.gradient_norms {
            if norm < self.analysis_config.vanishing_threshold {
                issues.push(GradientIssue::VanishingGradients {
                    layer_name: name.clone(),
                    gradient_norm: norm,
                    threshold: self.analysis_config.vanishing_threshold,
                });
            } else if norm > self.analysis_config.exploding_threshold {
                issues.push(GradientIssue::ExplodingGradients {
                    layer_name: name.clone(),
                    gradient_norm: norm,
                    threshold: self.analysis_config.exploding_threshold,
                });
            }
        }

        // Check for high condition numbers
        for (name, &cond_num) in &stats.condition_numbers {
            if cond_num > self.analysis_config.condition_number_threshold {
                issues.push(GradientIssue::HighConditionNumber {
                    parameter_name: name.clone(),
                    condition_number: cond_num,
                });
            }
        }

        // Check for dead neurons
        for (name, gradient) in gradients {
            if gradient.shape().dims().len() == 2 {
                let dead_indices = self.detect_dead_neurons(gradient)?;
                if !dead_indices.is_empty() {
                    let total_neurons = gradient.shape().dims()[1];
                    let dead_ratio = dead_indices.len() as f64 / total_neurons as f64;

                    issues.push(GradientIssue::DeadNeurons {
                        layer_name: name.clone(),
                        neuron_indices: dead_indices,
                        dead_ratio,
                    });
                }
            }
        }

        Ok(issues)
    }

    /// Compute performance metrics
    fn compute_performance_metrics(&self, start_time: Instant) -> Result<PerformanceMetrics> {
        let computation_time = start_time.elapsed().as_secs_f64();

        // Estimate memory usage (simplified)
        let memory_usage_mb = self
            .performance_tracker
            .memory_usage_history
            .last()
            .unwrap_or(&0)
            / (1024 * 1024);

        // Estimate FLOPS (simplified)
        let flops_estimate = self
            .computation_graph
            .nodes
            .values()
            .map(|node| node.flop_count)
            .sum();

        // Compute cache hit rate
        let total_accesses =
            self.performance_tracker.cache_hits + self.performance_tracker.cache_misses;
        let cache_hit_rate = if total_accesses > 0 {
            self.performance_tracker.cache_hits as f64 / total_accesses as f64
        } else {
            0.0
        };

        Ok(PerformanceMetrics {
            total_computation_time: computation_time,
            memory_usage_mb: memory_usage_mb as f64,
            flops_estimate,
            cache_hit_rate,
            parallel_efficiency: 0.8, // Placeholder - would need actual parallel profiling
        })
    }

    /// Generate recommendations based on detected issues
    fn generate_recommendations(
        &self,
        issues: &[GradientIssue],
        flow_analysis: &GradientFlowAnalysis,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Recommendations for vanishing gradients
        if !flow_analysis.vanishing_gradient_layers.is_empty() {
            recommendations.push(
                "Consider using residual connections, batch normalization, or different activation functions to address vanishing gradients.".to_string()
            );
        }

        // Recommendations for exploding gradients
        if !flow_analysis.exploding_gradient_layers.is_empty() {
            recommendations.push(
                "Consider gradient clipping, reducing learning rate, or using batch normalization to address exploding gradients.".to_string()
            );
        }

        // Recommendations for dead neurons
        if !flow_analysis.dead_neurons.is_empty() {
            recommendations.push(
                "Consider using different initialization schemes (e.g., He, Xavier) or activation functions (e.g., ReLU alternatives) to address dead neurons.".to_string()
            );
        }

        // Recommendations for numerical issues
        for issue in issues {
            match issue {
                GradientIssue::HighConditionNumber { .. } => {
                    recommendations.push(
                        "Consider using regularization techniques or different optimization algorithms to handle ill-conditioned problems.".to_string()
                    );
                }
                GradientIssue::NumericalInstability { .. } => {
                    recommendations.push(
                        "Consider using mixed precision training or numerical stabilization techniques.".to_string()
                    );
                }
                _ => {}
            }
        }

        recommendations
    }

    /// Helper methods for gradient analysis
    fn compute_tensor_norm(&self, tensor: &Tensor<T>) -> Result<f64> {
        // Compute L2 norm
        let two_tensor = Tensor::from_scalar(T::from(2.0).unwrap());
        let squared = tensor.pow(&two_tensor)?;
        let sum = squared.sum(None, false)?.to_scalar()?;
        Ok(sum.sqrt().to_f64().unwrap_or(0.0))
    }

    fn compute_tensor_mean(&self, tensor: &Tensor<T>) -> Result<f64> {
        let sum = tensor.sum(None, false)?.to_scalar()?;
        let count = tensor.shape().dims().iter().product::<usize>();
        Ok(sum.to_f64().unwrap_or(0.0) / count as f64)
    }

    fn compute_tensor_std(&self, tensor: &Tensor<T>) -> Result<f64> {
        let mean = self.compute_tensor_mean(tensor)?;
        let mean_tensor = Tensor::from_scalar(T::from(mean).unwrap());
        let centered = tensor.sub(&mean_tensor)?;
        let two_tensor = Tensor::from_scalar(T::from(2.0).unwrap());
        let squared = centered.pow(&two_tensor)?;
        let variance = self.compute_tensor_mean(&squared)?;
        Ok(variance.sqrt())
    }

    fn compute_sparsity_ratio(&self, tensor: &Tensor<T>) -> Result<f64> {
        let total_elements = tensor.shape().dims().iter().product::<usize>();
        let zero_threshold = T::from(1e-10).unwrap();

        // Count near-zero elements (simplified)
        let tensor_data = tensor.to_vec()?;
        let near_zero_count = tensor_data
            .iter()
            .filter(|&&val| val.abs() < zero_threshold)
            .count();

        Ok(near_zero_count as f64 / total_elements as f64)
    }

    fn estimate_condition_number(&self, _tensor: &Tensor<T>) -> Result<f64> {
        // Simplified condition number estimation
        // In practice, would use SVD
        Ok(1e6) // Placeholder
    }

    fn estimate_rank(&self, tensor: &Tensor<T>) -> Result<usize> {
        // Simplified rank estimation
        let dims = tensor.shape().dims();
        if dims.len() == 2 {
            Ok(dims[0].min(dims[1]))
        } else {
            Ok(0)
        }
    }

    fn detect_dead_neurons(&self, gradient: &Tensor<T>) -> Result<Vec<usize>> {
        let mut dead_neurons = Vec::new();
        let threshold = T::from(self.analysis_config.dead_neuron_threshold).unwrap();

        if gradient.shape().dims().len() == 2 {
            let dims = gradient.shape().dims();
            let grad_data = gradient.to_vec()?;

            // Check each neuron (column) for consistently small gradients
            for neuron_idx in 0..dims[1] {
                let mut max_grad = T::zero();
                for batch_idx in 0..dims[0] {
                    let idx = batch_idx * dims[1] + neuron_idx;
                    if idx < grad_data.len() {
                        max_grad = max_grad.max(grad_data[idx].abs());
                    }
                }

                if max_grad < threshold {
                    dead_neurons.push(neuron_idx);
                }
            }
        }

        Ok(dead_neurons)
    }

    fn identify_bottlenecks(&self, gradients: &HashMap<String, Tensor<T>>) -> Result<Vec<String>> {
        // Simplified bottleneck identification
        let mut bottlenecks = Vec::new();

        for (name, gradient) in gradients {
            let memory_usage = self.estimate_tensor_memory(gradient);
            let memory_mb = memory_usage as f64 / (1024.0 * 1024.0);

            if memory_mb > self.analysis_config.memory_threshold_mb {
                bottlenecks.push(name.clone());
            }
        }

        Ok(bottlenecks)
    }

    fn estimate_tensor_memory(&self, tensor: &Tensor<T>) -> usize {
        let element_count: usize = tensor.shape().dims().iter().product();
        element_count * std::mem::size_of::<T>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_analyzer() {
        let config = AnalysisConfig::default();
        let mut analyzer = GradientAnalyzer::<f32>::new(config);

        // Create test gradients
        let mut gradients = HashMap::new();
        gradients.insert("layer1".to_string(), Tensor::ones(&[10, 20]));
        gradients.insert(
            "layer2".to_string(),
            Tensor::from_scalar(1e-8f32).broadcast_to(&[5, 10]).unwrap(),
        );

        let report = analyzer.analyze_gradients(&gradients).unwrap();

        // Should detect vanishing gradients in layer2
        assert!(report.potential_issues.iter().any(|issue| {
            matches!(issue, GradientIssue::VanishingGradients { layer_name, .. } if layer_name == "layer2")
        }));
    }

    #[test]
    fn test_sparsity_computation() {
        let config = AnalysisConfig::default();
        let analyzer = GradientAnalyzer::<f32>::new(config);

        let sparse_tensor = Tensor::zeros(&[10, 10]);
        let sparsity = analyzer.compute_sparsity_ratio(&sparse_tensor).unwrap();

        assert!((sparsity - 1.0).abs() < 1e-6); // Should be fully sparse
    }
}
