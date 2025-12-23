//! Model Inspection and Debugging Utilities
//!
//! This module provides comprehensive tools for inspecting, debugging, and profiling
//! neural network models. Features include:
//! - Parameter counting and memory estimation
//! - Layer-wise analysis and statistics
//! - Gradient flow diagnostics
//! - Model structure visualization
//! - Performance profiling

use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Information about a single layer in the model
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct LayerInfo {
    /// Layer name/identifier
    pub name: String,
    /// Layer type (e.g., "Dense", "Conv2D", "Attention")
    pub layer_type: String,
    /// Number of parameters in this layer
    pub num_parameters: usize,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Whether layer is trainable
    pub trainable: bool,
    /// Additional layer-specific metadata
    pub metadata: HashMap<String, String>,
}

impl LayerInfo {
    /// Create new layer info
    pub fn new(name: String, layer_type: String) -> Self {
        Self {
            name,
            layer_type,
            num_parameters: 0,
            memory_bytes: 0,
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            trainable: true,
            metadata: HashMap::new(),
        }
    }

    /// Set number of parameters
    pub fn with_num_parameters(mut self, num: usize) -> Self {
        self.num_parameters = num;
        self
    }

    /// Set memory usage
    pub fn with_memory_bytes(mut self, bytes: usize) -> Self {
        self.memory_bytes = bytes;
        self
    }

    /// Set input shape
    pub fn with_input_shape(mut self, shape: Vec<usize>) -> Self {
        self.input_shape = shape;
        self
    }

    /// Set output shape
    pub fn with_output_shape(mut self, shape: Vec<usize>) -> Self {
        self.output_shape = shape;
        self
    }

    /// Set trainable flag
    pub fn with_trainable(mut self, trainable: bool) -> Self {
        self.trainable = trainable;
        self
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get memory usage in MB
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get parameter density (params per output element)
    pub fn parameter_density(&self) -> f64 {
        let output_size: usize = self.output_shape.iter().product();
        if output_size == 0 {
            0.0
        } else {
            self.num_parameters as f64 / output_size as f64
        }
    }
}

/// Complete model summary
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ModelSummary {
    /// Model name
    pub model_name: String,
    /// Total number of parameters
    pub total_parameters: usize,
    /// Number of trainable parameters
    pub trainable_parameters: usize,
    /// Number of non-trainable parameters
    pub non_trainable_parameters: usize,
    /// Total memory usage in bytes
    pub total_memory_bytes: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Layer-wise information
    pub layers: Vec<LayerInfo>,
    /// Model input shape
    pub input_shape: Vec<usize>,
    /// Model output shape
    pub output_shape: Vec<usize>,
}

impl ModelSummary {
    /// Create new model summary
    pub fn new(model_name: String) -> Self {
        Self {
            model_name,
            total_parameters: 0,
            trainable_parameters: 0,
            non_trainable_parameters: 0,
            total_memory_bytes: 0,
            num_layers: 0,
            layers: Vec::new(),
            input_shape: Vec::new(),
            output_shape: Vec::new(),
        }
    }

    /// Add layer information
    pub fn add_layer(&mut self, layer: LayerInfo) {
        self.total_parameters += layer.num_parameters;
        self.total_memory_bytes += layer.memory_bytes;

        if layer.trainable {
            self.trainable_parameters += layer.num_parameters;
        } else {
            self.non_trainable_parameters += layer.num_parameters;
        }

        self.num_layers += 1;
        self.layers.push(layer);
    }

    /// Get total memory in MB
    pub fn total_memory_mb(&self) -> f64 {
        self.total_memory_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get total memory in GB
    pub fn total_memory_gb(&self) -> f64 {
        self.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get percentage of trainable parameters
    pub fn trainable_percentage(&self) -> f64 {
        if self.total_parameters == 0 {
            0.0
        } else {
            (self.trainable_parameters as f64 / self.total_parameters as f64) * 100.0
        }
    }

    /// Format as a readable table
    pub fn format_table(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("Model: {}\n", self.model_name));
        output.push_str(&"=".repeat(80));
        output.push('\n');

        output.push_str(&format!(
            "{:<30} {:<15} {:<15} {:<15}\n",
            "Layer", "Type", "Params", "Memory (MB)"
        ));
        output.push_str(&"-".repeat(80));
        output.push('\n');

        for layer in &self.layers {
            output.push_str(&format!(
                "{:<30} {:<15} {:<15} {:<15.2}\n",
                layer.name,
                layer.layer_type,
                Self::format_number(layer.num_parameters),
                layer.memory_mb()
            ));
        }

        output.push_str(&"=".repeat(80));
        output.push('\n');
        output.push_str(&format!(
            "Total Parameters: {} ({} trainable, {} non-trainable)\n",
            Self::format_number(self.total_parameters),
            Self::format_number(self.trainable_parameters),
            Self::format_number(self.non_trainable_parameters)
        ));
        output.push_str(&format!(
            "Total Memory: {:.2} MB ({:.2} GB)\n",
            self.total_memory_mb(),
            self.total_memory_gb()
        ));
        output.push_str(&format!(
            "Trainable %: {:.2}%\n",
            self.trainable_percentage()
        ));

        output
    }

    /// Format large numbers with commas
    fn format_number(n: usize) -> String {
        let s = n.to_string();
        let mut result = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                result.push(',');
            }
            result.push(c);
        }
        result.chars().rev().collect()
    }

    /// Get layer by name
    pub fn get_layer(&self, name: &str) -> Option<&LayerInfo> {
        self.layers.iter().find(|layer| layer.name == name)
    }

    /// Get layers by type
    pub fn get_layers_by_type(&self, layer_type: &str) -> Vec<&LayerInfo> {
        self.layers
            .iter()
            .filter(|layer| layer.layer_type == layer_type)
            .collect()
    }

    /// Get largest layers by parameter count
    pub fn largest_layers(&self, n: usize) -> Vec<&LayerInfo> {
        let mut sorted: Vec<_> = self.layers.iter().collect();
        sorted.sort_by(|a, b| b.num_parameters.cmp(&a.num_parameters));
        sorted.into_iter().take(n).collect()
    }
}

/// Gradient flow statistics for debugging
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct GradientFlowInfo {
    /// Layer name
    pub layer_name: String,
    /// Mean gradient magnitude
    pub mean_magnitude: f32,
    /// Max gradient magnitude
    pub max_magnitude: f32,
    /// Min gradient magnitude
    pub min_magnitude: f32,
    /// Standard deviation
    pub std_deviation: f32,
    /// Percentage of zero gradients
    pub zero_percentage: f32,
    /// Whether gradients are healthy
    pub is_healthy: bool,
}

impl GradientFlowInfo {
    /// Create from gradient values
    pub fn from_gradients(layer_name: String, gradients: &[f32]) -> Self {
        if gradients.is_empty() {
            return Self {
                layer_name,
                mean_magnitude: 0.0,
                max_magnitude: 0.0,
                min_magnitude: 0.0,
                std_deviation: 0.0,
                zero_percentage: 100.0,
                is_healthy: false,
            };
        }

        let mean = gradients.iter().sum::<f32>() / gradients.len() as f32;
        let max = gradients.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = gradients.iter().cloned().fold(f32::INFINITY, f32::min);

        let variance =
            gradients.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / gradients.len() as f32;
        let std_dev = variance.sqrt();

        let zero_count = gradients.iter().filter(|&&x| x.abs() < 1e-10).count();
        let zero_pct = (zero_count as f32 / gradients.len() as f32) * 100.0;

        let is_healthy = !max.is_infinite()
            && !max.is_nan()
            && max < 10.0
            && mean.abs() > 1e-7
            && zero_pct < 90.0;

        Self {
            layer_name,
            mean_magnitude: mean.abs(),
            max_magnitude: max,
            min_magnitude: min,
            std_deviation: std_dev,
            zero_percentage: zero_pct,
            is_healthy,
        }
    }

    /// Check if gradients are exploding
    pub fn is_exploding(&self) -> bool {
        self.max_magnitude > 10.0 || self.max_magnitude.is_infinite()
    }

    /// Check if gradients are vanishing
    pub fn is_vanishing(&self) -> bool {
        self.mean_magnitude < 1e-7 || self.zero_percentage > 90.0
    }
}

/// Model profiling information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ProfilingInfo {
    /// Layer name
    pub layer_name: String,
    /// Forward pass time in milliseconds
    pub forward_time_ms: f64,
    /// Backward pass time in milliseconds
    pub backward_time_ms: Option<f64>,
    /// Total time in milliseconds
    pub total_time_ms: f64,
    /// Number of FLOPs (floating point operations)
    pub flops: Option<u64>,
}

impl ProfilingInfo {
    /// Create new profiling info
    pub fn new(layer_name: String, forward_time_ms: f64) -> Self {
        Self {
            layer_name,
            forward_time_ms,
            backward_time_ms: None,
            total_time_ms: forward_time_ms,
            flops: None,
        }
    }

    /// Add backward pass time
    pub fn with_backward_time(mut self, backward_time_ms: f64) -> Self {
        self.backward_time_ms = Some(backward_time_ms);
        self.total_time_ms += backward_time_ms;
        self
    }

    /// Add FLOP count
    pub fn with_flops(mut self, flops: u64) -> Self {
        self.flops = Some(flops);
        self
    }

    /// Get FLOPS (FLOPs per second)
    pub fn get_flops(&self) -> Option<f64> {
        self.flops
            .map(|f| (f as f64) / (self.total_time_ms / 1000.0))
    }

    /// Get percentage of total time
    pub fn percentage_of_total(&self, total_time: f64) -> f64 {
        if total_time > 0.0 {
            (self.total_time_ms / total_time) * 100.0
        } else {
            0.0
        }
    }
}

/// Model inspector for comprehensive analysis
pub struct ModelInspector {
    summary: ModelSummary,
    gradient_flow: Vec<GradientFlowInfo>,
    profiling: Vec<ProfilingInfo>,
}

impl ModelInspector {
    /// Create new model inspector
    pub fn new(model_name: String) -> Self {
        Self {
            summary: ModelSummary::new(model_name),
            gradient_flow: Vec::new(),
            profiling: Vec::new(),
        }
    }

    /// Add layer to summary
    pub fn add_layer(&mut self, layer: LayerInfo) {
        self.summary.add_layer(layer);
    }

    /// Add gradient flow information
    pub fn add_gradient_flow(&mut self, info: GradientFlowInfo) {
        self.gradient_flow.push(info);
    }

    /// Add profiling information
    pub fn add_profiling(&mut self, info: ProfilingInfo) {
        self.profiling.push(info);
    }

    /// Get model summary
    pub fn summary(&self) -> &ModelSummary {
        &self.summary
    }

    /// Get gradient flow information
    pub fn gradient_flow(&self) -> &[GradientFlowInfo] {
        &self.gradient_flow
    }

    /// Get profiling information
    pub fn profiling(&self) -> &[ProfilingInfo] {
        &self.profiling
    }

    /// Check for gradient issues
    pub fn check_gradient_health(&self) -> Vec<String> {
        let mut issues = Vec::new();

        for flow in &self.gradient_flow {
            if flow.is_exploding() {
                issues.push(format!(
                    "Layer '{}': Exploding gradients detected (max: {:.2e})",
                    flow.layer_name, flow.max_magnitude
                ));
            }

            if flow.is_vanishing() {
                issues.push(format!(
                    "Layer '{}': Vanishing gradients detected (mean: {:.2e}, zero%: {:.1})",
                    flow.layer_name, flow.mean_magnitude, flow.zero_percentage
                ));
            }
        }

        issues
    }

    /// Get profiling report
    pub fn profiling_report(&self) -> String {
        let mut output = String::new();
        output.push_str("Profiling Report\n");
        output.push_str(&"=".repeat(80));
        output.push('\n');

        let total_time: f64 = self.profiling.iter().map(|p| p.total_time_ms).sum();

        output.push_str(&format!(
            "{:<30} {:<15} {:<15} {:<15}\n",
            "Layer", "Forward (ms)", "Backward (ms)", "% of Total"
        ));
        output.push_str(&"-".repeat(80));
        output.push('\n');

        for prof in &self.profiling {
            let backward_str = prof
                .backward_time_ms
                .map(|t| format!("{:.2}", t))
                .unwrap_or_else(|| "N/A".to_string());

            output.push_str(&format!(
                "{:<30} {:<15.2} {:<15} {:<15.1}%\n",
                prof.layer_name,
                prof.forward_time_ms,
                backward_str,
                prof.percentage_of_total(total_time)
            ));
        }

        output.push_str(&"=".repeat(80));
        output.push('\n');
        output.push_str(&format!("Total Time: {:.2} ms\n", total_time));

        output
    }

    /// Generate full diagnostic report
    pub fn full_report(&self) -> String {
        let mut output = String::new();

        // Model summary
        output.push_str(&self.summary.format_table());
        output.push('\n');

        // Gradient health
        if !self.gradient_flow.is_empty() {
            output.push_str("\nGradient Flow Analysis\n");
            output.push_str(&"=".repeat(80));
            output.push('\n');

            let issues = self.check_gradient_health();
            if issues.is_empty() {
                output.push_str("✓ All gradients are healthy\n");
            } else {
                output.push_str("⚠ Gradient Issues Detected:\n");
                for issue in issues {
                    output.push_str(&format!("  - {}\n", issue));
                }
            }
            output.push('\n');
        }

        // Profiling
        if !self.profiling.is_empty() {
            output.push_str(&self.profiling_report());
        }

        output
    }
}

/// Utilities for parameter analysis
pub mod utils {
    use super::*;

    /// Count parameters in a tensor
    pub fn count_parameters<T>(tensor: &Tensor<T>) -> usize {
        tensor.shape().dims().iter().product()
    }

    /// Estimate memory usage for a tensor
    pub fn estimate_memory<T>(tensor: &Tensor<T>) -> usize {
        let num_elements: usize = tensor.shape().dims().iter().product();
        num_elements * std::mem::size_of::<T>()
    }

    /// Calculate FLOPs for a dense layer
    pub fn dense_layer_flops(input_size: usize, output_size: usize, batch_size: usize) -> u64 {
        // Matrix multiplication: batch_size * input_size * output_size * 2 (multiply-add)
        (batch_size * input_size * output_size * 2) as u64
    }

    /// Calculate FLOPs for a convolution layer
    pub fn conv_layer_flops(
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        in_channels: usize,
        out_channels: usize,
        batch_size: usize,
    ) -> u64 {
        // Output dimensions (assuming stride=1, no padding)
        let output_h = input_h - kernel_h + 1;
        let output_w = input_w - kernel_w + 1;

        // FLOPs: batch * out_h * out_w * out_channels * in_channels * kernel_h * kernel_w * 2
        (batch_size * output_h * output_w * out_channels * in_channels * kernel_h * kernel_w * 2)
            as u64
    }

    /// Format bytes as human-readable string
    pub fn format_bytes(bytes: usize) -> String {
        const KB: f64 = 1024.0;
        const MB: f64 = KB * 1024.0;
        const GB: f64 = MB * 1024.0;

        let bytes_f = bytes as f64;

        if bytes_f >= GB {
            format!("{:.2} GB", bytes_f / GB)
        } else if bytes_f >= MB {
            format!("{:.2} MB", bytes_f / MB)
        } else if bytes_f >= KB {
            format!("{:.2} KB", bytes_f / KB)
        } else {
            format!("{} B", bytes)
        }
    }

    /// Calculate model compression ratio
    pub fn compression_ratio(original_params: usize, compressed_params: usize) -> f64 {
        if compressed_params == 0 {
            0.0
        } else {
            original_params as f64 / compressed_params as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_info_creation() {
        let layer = LayerInfo::new("conv1".to_string(), "Conv2D".to_string())
            .with_num_parameters(1000)
            .with_memory_bytes(4000)
            .with_trainable(true);

        assert_eq!(layer.name, "conv1");
        assert_eq!(layer.layer_type, "Conv2D");
        assert_eq!(layer.num_parameters, 1000);
        assert_eq!(layer.memory_bytes, 4000);
        assert!(layer.trainable);
    }

    #[test]
    fn test_layer_info_memory_mb() {
        let layer =
            LayerInfo::new("test".to_string(), "Dense".to_string()).with_memory_bytes(1024 * 1024); // 1 MB

        assert!((layer.memory_mb() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_layer_info_parameter_density() {
        let layer = LayerInfo::new("test".to_string(), "Dense".to_string())
            .with_num_parameters(100)
            .with_output_shape(vec![10, 10]); // 100 elements

        assert!((layer.parameter_density() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_model_summary_creation() {
        let mut summary = ModelSummary::new("TestModel".to_string());

        let layer1 = LayerInfo::new("layer1".to_string(), "Dense".to_string())
            .with_num_parameters(1000)
            .with_memory_bytes(4000)
            .with_trainable(true);

        let layer2 = LayerInfo::new("layer2".to_string(), "Conv2D".to_string())
            .with_num_parameters(500)
            .with_memory_bytes(2000)
            .with_trainable(false);

        summary.add_layer(layer1);
        summary.add_layer(layer2);

        assert_eq!(summary.total_parameters, 1500);
        assert_eq!(summary.trainable_parameters, 1000);
        assert_eq!(summary.non_trainable_parameters, 500);
        assert_eq!(summary.total_memory_bytes, 6000);
        assert_eq!(summary.num_layers, 2);
    }

    #[test]
    fn test_model_summary_trainable_percentage() {
        let mut summary = ModelSummary::new("TestModel".to_string());

        let layer1 = LayerInfo::new("layer1".to_string(), "Dense".to_string())
            .with_num_parameters(800)
            .with_trainable(true);

        let layer2 = LayerInfo::new("layer2".to_string(), "Dense".to_string())
            .with_num_parameters(200)
            .with_trainable(false);

        summary.add_layer(layer1);
        summary.add_layer(layer2);

        assert!((summary.trainable_percentage() - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_model_summary_get_layer() {
        let mut summary = ModelSummary::new("TestModel".to_string());

        let layer = LayerInfo::new("conv1".to_string(), "Conv2D".to_string());
        summary.add_layer(layer);

        let found = summary.get_layer("conv1");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "conv1");

        let not_found = summary.get_layer("nonexistent");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_model_summary_get_layers_by_type() {
        let mut summary = ModelSummary::new("TestModel".to_string());

        summary.add_layer(LayerInfo::new("conv1".to_string(), "Conv2D".to_string()));
        summary.add_layer(LayerInfo::new("dense1".to_string(), "Dense".to_string()));
        summary.add_layer(LayerInfo::new("conv2".to_string(), "Conv2D".to_string()));

        let conv_layers = summary.get_layers_by_type("Conv2D");
        assert_eq!(conv_layers.len(), 2);
    }

    #[test]
    fn test_model_summary_largest_layers() {
        let mut summary = ModelSummary::new("TestModel".to_string());

        summary.add_layer(
            LayerInfo::new("layer1".to_string(), "Dense".to_string()).with_num_parameters(100),
        );
        summary.add_layer(
            LayerInfo::new("layer2".to_string(), "Dense".to_string()).with_num_parameters(500),
        );
        summary.add_layer(
            LayerInfo::new("layer3".to_string(), "Dense".to_string()).with_num_parameters(200),
        );

        let largest = summary.largest_layers(2);
        assert_eq!(largest.len(), 2);
        assert_eq!(largest[0].name, "layer2"); // 500 params
        assert_eq!(largest[1].name, "layer3"); // 200 params
    }

    #[test]
    fn test_gradient_flow_healthy() {
        let gradients = vec![0.01, 0.02, 0.015, 0.018, 0.012];
        let flow = GradientFlowInfo::from_gradients("layer1".to_string(), &gradients);

        assert!(flow.is_healthy);
        assert!(!flow.is_exploding());
        assert!(!flow.is_vanishing());
    }

    #[test]
    fn test_gradient_flow_exploding() {
        let gradients = vec![1.0, 50.0, 2.0];
        let flow = GradientFlowInfo::from_gradients("layer1".to_string(), &gradients);

        assert!(flow.is_exploding());
        assert!(!flow.is_healthy);
    }

    #[test]
    fn test_gradient_flow_vanishing() {
        let gradients = vec![0.0, 0.0, 1e-10, 0.0];
        let flow = GradientFlowInfo::from_gradients("layer1".to_string(), &gradients);

        assert!(flow.is_vanishing());
        assert!(!flow.is_healthy);
    }

    #[test]
    fn test_gradient_flow_empty() {
        let gradients: Vec<f32> = vec![];
        let flow = GradientFlowInfo::from_gradients("layer1".to_string(), &gradients);

        assert!(!flow.is_healthy);
        assert_eq!(flow.zero_percentage, 100.0);
    }

    #[test]
    fn test_profiling_info_creation() {
        let prof = ProfilingInfo::new("layer1".to_string(), 10.5);

        assert_eq!(prof.layer_name, "layer1");
        assert_eq!(prof.forward_time_ms, 10.5);
        assert_eq!(prof.total_time_ms, 10.5);
        assert!(prof.backward_time_ms.is_none());
    }

    #[test]
    fn test_profiling_info_with_backward() {
        let prof = ProfilingInfo::new("layer1".to_string(), 10.0).with_backward_time(20.0);

        assert_eq!(prof.forward_time_ms, 10.0);
        assert_eq!(prof.backward_time_ms, Some(20.0));
        assert_eq!(prof.total_time_ms, 30.0);
    }

    #[test]
    fn test_profiling_info_percentage() {
        let prof = ProfilingInfo::new("layer1".to_string(), 25.0);
        let percentage = prof.percentage_of_total(100.0);

        assert!((percentage - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_model_inspector() {
        let mut inspector = ModelInspector::new("TestModel".to_string());

        let layer =
            LayerInfo::new("layer1".to_string(), "Dense".to_string()).with_num_parameters(1000);

        inspector.add_layer(layer);

        assert_eq!(inspector.summary().total_parameters, 1000);
        assert_eq!(inspector.summary().num_layers, 1);
    }

    #[test]
    fn test_model_inspector_gradient_health() {
        let mut inspector = ModelInspector::new("TestModel".to_string());

        // Add healthy gradients
        let healthy =
            GradientFlowInfo::from_gradients("healthy_layer".to_string(), &vec![0.01, 0.02, 0.015]);
        inspector.add_gradient_flow(healthy);

        // Add exploding gradients
        let exploding =
            GradientFlowInfo::from_gradients("exploding_layer".to_string(), &vec![100.0, 200.0]);
        inspector.add_gradient_flow(exploding);

        let issues = inspector.check_gradient_health();
        assert_eq!(issues.len(), 1);
        assert!(issues[0].contains("exploding_layer"));
        assert!(issues[0].contains("Exploding"));
    }

    #[test]
    fn test_utils_dense_layer_flops() {
        let flops = utils::dense_layer_flops(100, 50, 32);
        // batch=32, input=100, output=50, multiply-add = 32 * 100 * 50 * 2
        assert_eq!(flops, 320000);
    }

    #[test]
    fn test_utils_format_bytes() {
        assert_eq!(utils::format_bytes(512), "512 B");
        assert_eq!(utils::format_bytes(1024), "1.00 KB");
        assert_eq!(utils::format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(utils::format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_utils_compression_ratio() {
        let ratio = utils::compression_ratio(1000, 100);
        assert!((ratio - 10.0).abs() < 0.01);

        let zero_ratio = utils::compression_ratio(1000, 0);
        assert_eq!(zero_ratio, 0.0);
    }

    #[test]
    fn test_model_summary_format_number() {
        assert_eq!(ModelSummary::format_number(1000), "1,000");
        assert_eq!(ModelSummary::format_number(1000000), "1,000,000");
        assert_eq!(ModelSummary::format_number(123), "123");
    }
}
