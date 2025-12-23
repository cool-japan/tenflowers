//! Gradient Flow Visualizer Implementation
//!
//! This module contains the main GradientFlowVisualizer implementation for
//! analyzing and visualizing gradient flow through computation graphs.

use super::types::{
    EdgeType, GradientFlowAnalysis, GradientFlowEdge, GradientFlowIssue, GradientFlowNode,
    GradientStats, IssueType, NodeType, Severity, ValueStats, VisualizationSettings,
};
use crate::tape::{GradientTape, Operation, TensorId, TrackedTensor};
use scirs2_core::numeric::{Float, FromPrimitive, One, Signed, Zero};
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

/// Configuration for gradient flow analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Threshold for vanishing gradient detection
    pub vanishing_threshold: f64,
    /// Threshold for exploding gradient detection
    pub exploding_threshold: f64,
    /// Minimum gradient magnitude to consider
    pub min_gradient_threshold: f64,
    /// Whether to analyze dead neurons
    pub analyze_dead_neurons: bool,
    /// Whether to detect saturated activations
    pub analyze_saturation: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            vanishing_threshold: 1e-6,
            exploding_threshold: 1e2,
            min_gradient_threshold: 1e-8,
            analyze_dead_neurons: true,
            analyze_saturation: true,
        }
    }
}

/// Result of gradient flow visualization
#[derive(Debug, Clone)]
pub struct VisualizationResult<T> {
    /// Generated visualization data (SVG, JSON, etc.)
    pub output: String,
    /// Analysis results
    pub analysis: GradientFlowAnalysis<T>,
    /// Visualization metadata
    pub metadata: HashMap<String, String>,
}

impl<T> VisualizationResult<T> {
    /// Create a new visualization result
    pub fn new(output: String, analysis: GradientFlowAnalysis<T>) -> Self {
        Self {
            output,
            analysis,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the result
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Gradient flow visualizer for computation graphs
#[derive(Debug, Clone)]
pub struct GradientFlowVisualizer<T> {
    /// Node information for visualization
    nodes: HashMap<TensorId, GradientFlowNode<T>>,
    /// Edge information for visualization
    edges: Vec<GradientFlowEdge>,
    /// Flow analysis results
    flow_analysis: Option<GradientFlowAnalysis<T>>,
    /// Visualization settings
    settings: VisualizationSettings,
}

impl<T> GradientFlowVisualizer<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + std::cmp::PartialOrd
        + FromPrimitive
        + Signed
        + std::fmt::Debug
        + serde::Serialize
        + serde::de::DeserializeOwned
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new gradient flow visualizer
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            flow_analysis: None,
            settings: VisualizationSettings::default(),
        }
    }

    /// Create visualizer with custom settings
    pub fn with_settings(settings: VisualizationSettings) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            flow_analysis: None,
            settings,
        }
    }

    /// Get the current visualization settings
    pub fn settings(&self) -> &VisualizationSettings {
        &self.settings
    }

    /// Update visualization settings
    pub fn set_settings(&mut self, settings: VisualizationSettings) {
        self.settings = settings;
    }

    /// Get the nodes in the visualization graph
    pub fn nodes(&self) -> &HashMap<TensorId, GradientFlowNode<T>> {
        &self.nodes
    }

    /// Get the edges in the visualization graph
    pub fn edges(&self) -> &[GradientFlowEdge] {
        &self.edges
    }

    /// Get the flow analysis results
    pub fn flow_analysis(&self) -> Option<&GradientFlowAnalysis<T>> {
        self.flow_analysis.as_ref()
    }

    /// Analyze gradient flow from a computation graph
    pub fn analyze_flow(
        &mut self,
        tape: &GradientTape,
        target: &TrackedTensor<T>,
        sources: &[&TrackedTensor<T>],
    ) -> Result<()> {
        // Clear previous analysis
        self.nodes.clear();
        self.edges.clear();

        // Compute gradients
        let targets = std::slice::from_ref(target);
        let source_values: Vec<TrackedTensor<T>> = sources.iter().map(|&s| s.clone()).collect();
        let gradients = tape.gradient(targets, &source_values)?;

        // Convert gradients from Vec<Option<Tensor<T>>> to Vec<Tensor<T>>
        let concrete_gradients: Vec<Tensor<T>> = gradients.into_iter().flatten().collect();

        // Build the computation graph
        self.build_graph(tape, target, sources, &concrete_gradients)?;

        // Perform flow analysis
        self.flow_analysis = Some(self.analyze_gradient_flow()?);

        Ok(())
    }

    /// Build the computation graph for visualization
    fn build_graph(
        &mut self,
        tape: &GradientTape,
        target: &TrackedTensor<T>,
        sources: &[&TrackedTensor<T>],
        gradients: &[Tensor<T>],
    ) -> Result<()> {
        // Add target node
        let target_stats = self.compute_gradient_stats(&target.tensor)?;
        let target_value_stats = self.compute_value_stats(&target.tensor)?;

        let target_node = GradientFlowNode::new(
            target.id,
            "target".to_string(),
            "target".to_string(),
            target.tensor.shape().dims().to_vec(),
            NodeType::Output,
        );

        let mut target_node = target_node;
        target_node.gradient_stats = target_stats;
        target_node.value_stats = target_value_stats;

        self.nodes.insert(target.id, target_node);

        // Add source nodes with their gradients
        for (i, source) in sources.iter().enumerate() {
            let gradient_stats = if i < gradients.len() {
                self.compute_gradient_stats(&gradients[i])?
            } else {
                GradientStats::default()
            };

            let value_stats = self.compute_value_stats(&source.tensor)?;

            let source_node = GradientFlowNode::new(
                source.id,
                format!("source_{}", i),
                "parameter".to_string(),
                source.tensor.shape().dims().to_vec(),
                NodeType::Parameter,
            );

            let mut source_node = source_node;
            source_node.gradient_stats = gradient_stats;
            source_node.value_stats = value_stats;

            self.nodes.insert(source.id, source_node);

            // Create edge from source to target
            let gradient_magnitude = if i < gradients.len() {
                self.compute_tensor_magnitude(&gradients[i])?
            } else {
                0.0
            };

            let edge = GradientFlowEdge::new(source.id, target.id, EdgeType::Forward)
                .with_gradient_magnitude(gradient_magnitude);

            self.edges.push(edge);
        }

        // Build intermediate nodes from tape operations
        self.build_intermediate_nodes_from_tape(tape)?;

        Ok(())
    }

    /// Build intermediate nodes from tape operations
    fn build_intermediate_nodes_from_tape(&mut self, tape: &GradientTape) -> Result<()> {
        // Get tape nodes and build intermediate computation nodes
        let tape_nodes = tape.get_all_nodes();

        for tape_node in tape_nodes.iter() {
            let node_id = &tape_node.id;
            if !self.nodes.contains_key(node_id) {
                // Create intermediate node
                let (operation_name, node_type) = match &tape_node.operation {
                    Operation::Add { .. } => ("Add".to_string(), NodeType::Hidden),
                    Operation::Sub { .. } => ("Sub".to_string(), NodeType::Hidden),
                    Operation::Mul { .. } => ("Mul".to_string(), NodeType::Hidden),
                    Operation::Div { .. } => ("Div".to_string(), NodeType::Hidden),
                    Operation::Pow { .. } => ("Pow".to_string(), NodeType::Hidden),
                    Operation::MatMul { .. } => ("MatMul".to_string(), NodeType::Hidden),
                    Operation::Relu { .. } => ("ReLU".to_string(), NodeType::Hidden),
                    Operation::Sigmoid { .. } => ("Sigmoid".to_string(), NodeType::Hidden),
                    Operation::Softmax { .. } => ("Softmax".to_string(), NodeType::Hidden),
                    Operation::BatchNorm { .. } => ("BatchNorm".to_string(), NodeType::Hidden),
                    _ => ("Unknown".to_string(), NodeType::Hidden),
                };

                let intermediate_node = GradientFlowNode::new(
                    *node_id,
                    format!("node_{}", node_id),
                    operation_name,
                    tape_node.output_shape.clone(),
                    node_type,
                );

                self.nodes.insert(*node_id, intermediate_node);
            }
        }

        // Add edges based on tape dependencies
        for tape_node in tape_nodes.iter() {
            let node_id = tape_node.id;
            for &input_id in &tape_node.parents {
                if self.nodes.contains_key(&input_id) {
                    let edge = GradientFlowEdge::new(input_id, node_id, EdgeType::Forward);
                    self.edges.push(edge);
                }
            }
        }

        Ok(())
    }

    /// Analyze gradient flow and detect issues
    fn analyze_gradient_flow(&self) -> Result<GradientFlowAnalysis<T>> {
        let mut analysis = GradientFlowAnalysis::new();

        // Calculate flow statistics
        analysis.flow_statistics.total_nodes = self.nodes.len();
        analysis.flow_statistics.total_edges = self.edges.len();

        // Analyze each node for gradient issues
        let mut vanishing_count = 0;
        let mut exploding_count = 0;
        let mut total_gradient_magnitude = T::zero();
        let mut max_gradient = T::zero();
        let mut min_gradient = T::from_f64(f64::INFINITY).unwrap_or(T::zero());

        for (node_id, node) in &self.nodes {
            let grad_magnitude = self.compute_node_gradient_magnitude(&node.gradient_stats);

            total_gradient_magnitude = total_gradient_magnitude + grad_magnitude;
            max_gradient = if grad_magnitude > max_gradient {
                grad_magnitude
            } else {
                max_gradient
            };
            min_gradient = if grad_magnitude < min_gradient {
                grad_magnitude
            } else {
                min_gradient
            };

            // Check for vanishing gradients
            if grad_magnitude < T::from_f64(1e-6).unwrap_or(T::zero()) {
                vanishing_count += 1;
                let issue = GradientFlowIssue::new(
                    IssueType::VanishingGradients,
                    Severity::High,
                    format!("Node {} has vanishing gradients", node_id),
                    "Consider gradient clipping or different activation functions".to_string(),
                )
                .with_affected_nodes(vec![*node_id]);
                analysis.add_issue(issue);
            }

            // Check for exploding gradients
            if grad_magnitude > T::from_f64(10.0).unwrap_or(T::from_f64(f64::INFINITY).unwrap()) {
                exploding_count += 1;
                let issue = GradientFlowIssue::new(
                    IssueType::ExplodingGradients,
                    Severity::High,
                    format!("Node {} has exploding gradients", node_id),
                    "Consider gradient clipping or lower learning rate".to_string(),
                )
                .with_affected_nodes(vec![*node_id]);
                analysis.add_issue(issue);
            }

            // Check for dead neurons (all zero gradients)
            if node.gradient_stats.zero_percentage > 0.95 {
                let issue = GradientFlowIssue::new(
                    IssueType::DeadNeurons,
                    Severity::Medium,
                    format!("Node {} appears to be a dead neuron", node_id),
                    "Consider different initialization or activation functions".to_string(),
                )
                .with_affected_nodes(vec![*node_id]);
                analysis.add_issue(issue);
            }
        }

        // Update flow statistics
        if !self.nodes.is_empty() {
            analysis.flow_statistics.avg_gradient_magnitude =
                total_gradient_magnitude / T::from_usize(self.nodes.len()).unwrap_or(T::one());
        }
        analysis.flow_statistics.max_gradient_magnitude = max_gradient;
        analysis.flow_statistics.min_gradient_magnitude = min_gradient;
        analysis.flow_statistics.vanishing_percentage =
            (vanishing_count as f64 / self.nodes.len() as f64) * 100.0;
        analysis.flow_statistics.exploding_percentage =
            (exploding_count as f64 / self.nodes.len() as f64) * 100.0;

        // Calculate graph depth
        analysis.flow_statistics.graph_depth = self.calculate_graph_depth();

        // Find critical path
        analysis.set_critical_path(self.find_critical_path());

        // Calculate health score
        analysis.calculate_health_score();

        Ok(analysis)
    }

    /// Compute gradient statistics for a tensor
    fn compute_gradient_stats(&self, tensor: &Tensor<T>) -> Result<GradientStats<T>> {
        let data = tensor.data();
        let total_elements = data.len();

        if total_elements == 0 {
            return Ok(GradientStats::default());
        }

        // Calculate basic statistics
        let sum = data.iter().fold(T::zero(), |acc, &x| acc + x);
        let mean = sum / T::from_usize(total_elements).unwrap_or(T::one());

        let variance = data.iter().fold(T::zero(), |acc, &x| {
            let diff = x - mean;
            acc + diff * diff
        }) / T::from_usize(total_elements).unwrap_or(T::one());

        let std = variance.sqrt();

        let min = data.iter().fold(
            T::from_f64(f64::INFINITY).unwrap_or(T::zero()),
            |acc, &x| {
                if x < acc {
                    x
                } else {
                    acc
                }
            },
        );

        let max = data.iter().fold(
            T::from_f64(f64::NEG_INFINITY).unwrap_or(T::zero()),
            |acc, &x| {
                if x > acc {
                    x
                } else {
                    acc
                }
            },
        );

        // Calculate norms
        let l1_norm = data.iter().fold(T::zero(), |acc, &x| acc + x.abs());
        let l2_norm = data.iter().fold(T::zero(), |acc, &x| acc + x * x).sqrt();

        // Calculate percentages
        let zero_threshold = T::from_f64(1e-10).unwrap_or(T::zero());
        let zero_count = data.iter().filter(|&&x| x.abs() < zero_threshold).count();
        let zero_percentage = (zero_count as f64 / total_elements as f64) * 100.0;

        let invalid_count = data
            .iter()
            .filter(|&&x| x.is_infinite() || x.is_nan())
            .count();
        let invalid_percentage = (invalid_count as f64 / total_elements as f64) * 100.0;

        // Determine if vanishing or exploding
        let magnitude = l2_norm / T::from_usize(total_elements).unwrap_or(T::one()).sqrt();
        let is_vanishing = magnitude < T::from_f64(1e-6).unwrap_or(T::zero());
        let is_exploding =
            magnitude > T::from_f64(10.0).unwrap_or(T::from_f64(f64::INFINITY).unwrap());

        Ok(GradientStats {
            mean,
            std,
            min,
            max,
            l1_norm,
            l2_norm,
            zero_percentage,
            invalid_percentage,
            is_vanishing,
            is_exploding,
        })
    }

    /// Compute value statistics for a tensor
    fn compute_value_stats(&self, tensor: &Tensor<T>) -> Result<ValueStats<T>> {
        let data = tensor.data();
        let total_elements = data.len();

        if total_elements == 0 {
            return Ok(ValueStats::default());
        }

        // Calculate basic statistics
        let sum = data.iter().fold(T::zero(), |acc, &x| acc + x);
        let mean = sum / T::from_usize(total_elements).unwrap_or(T::one());

        let variance = data.iter().fold(T::zero(), |acc, &x| {
            let diff = x - mean;
            acc + diff * diff
        }) / T::from_usize(total_elements).unwrap_or(T::one());

        let std = variance.sqrt();

        let min = data.iter().fold(
            T::from_f64(f64::INFINITY).unwrap_or(T::zero()),
            |acc, &x| {
                if x < acc {
                    x
                } else {
                    acc
                }
            },
        );

        let max = data.iter().fold(
            T::from_f64(f64::NEG_INFINITY).unwrap_or(T::zero()),
            |acc, &x| {
                if x > acc {
                    x
                } else {
                    acc
                }
            },
        );

        // Calculate L2 norm
        let norm = data.iter().fold(T::zero(), |acc, &x| acc + x * x).sqrt();

        // Calculate sparsity (percentage of zero values)
        let zero_threshold = T::from_f64(1e-10).unwrap_or(T::zero());
        let zero_count = data.iter().filter(|&&x| x.abs() < zero_threshold).count();
        let sparsity = (zero_count as f64 / total_elements as f64) * 100.0;

        Ok(ValueStats {
            mean,
            std,
            min,
            max,
            norm,
            sparsity,
        })
    }

    /// Compute the magnitude of a tensor (L2 norm)
    fn compute_tensor_magnitude(&self, tensor: &Tensor<T>) -> Result<f64> {
        let data = tensor.data();
        let sum_of_squares = data.iter().fold(T::zero(), |acc, &x| acc + x * x);
        let magnitude = sum_of_squares.sqrt();
        Ok(magnitude.to_f64().unwrap_or(0.0))
    }

    /// Compute gradient magnitude for a node
    fn compute_node_gradient_magnitude(&self, stats: &GradientStats<T>) -> T {
        stats.l2_norm
    }

    /// Calculate the depth of the computation graph
    fn calculate_graph_depth(&self) -> usize {
        // Simplified depth calculation - in practice, this would use topological sort
        let mut max_depth = 0;

        // For each node, calculate its depth based on incoming edges
        for node_id in self.nodes.keys() {
            let depth = self.calculate_node_depth(*node_id, 0);
            max_depth = max_depth.max(depth);
        }

        max_depth
    }

    /// Calculate depth for a specific node (recursive)
    fn calculate_node_depth(&self, node_id: TensorId, current_depth: usize) -> usize {
        // Prevent infinite recursion
        if current_depth > 100 {
            return current_depth;
        }

        let mut max_incoming_depth = current_depth;

        for edge in &self.edges {
            if edge.to == node_id {
                let incoming_depth = self.calculate_node_depth(edge.from, current_depth + 1);
                max_incoming_depth = max_incoming_depth.max(incoming_depth);
            }
        }

        max_incoming_depth
    }

    /// Find the critical path in the computation graph
    fn find_critical_path(&self) -> Vec<TensorId> {
        // Simplified critical path finding - in practice, this would be more sophisticated
        let mut path = Vec::new();

        // Find the path with the highest gradient magnitudes
        if let Some(start_node) = self
            .nodes
            .values()
            .filter(|node| node.node_type == NodeType::Parameter)
            .max_by(|a, b| {
                self.compute_node_gradient_magnitude(&a.gradient_stats)
                    .partial_cmp(&self.compute_node_gradient_magnitude(&b.gradient_stats))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            path.push(start_node.id);

            // Follow the path forward
            let mut current_id = start_node.id;
            let mut visited = std::collections::HashSet::new();

            while let Some(edge) = self
                .edges
                .iter()
                .filter(|edge| edge.from == current_id && !visited.contains(&edge.to))
                .max_by(|a, b| {
                    a.gradient_magnitude
                        .partial_cmp(&b.gradient_magnitude)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                visited.insert(edge.to);
                path.push(edge.to);
                current_id = edge.to;
            }
        }

        path
    }

    /// Clear all analysis results
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.flow_analysis = None;
    }

    /// Get a summary of the gradient flow health
    pub fn get_health_summary(&self) -> Result<String> {
        if let Some(analysis) = &self.flow_analysis {
            let mut summary = String::new();

            summary.push_str(&format!(
                "Gradient Flow Health Score: {:.2}\n",
                analysis.health_score
            ));

            summary.push_str(&format!("Total Issues Found: {}\n", analysis.issues.len()));

            summary.push_str(&format!(
                "Vanishing Gradients: {:.1}% of nodes\n",
                analysis.flow_statistics.vanishing_percentage
            ));

            summary.push_str(&format!(
                "Exploding Gradients: {:.1}% of nodes\n",
                analysis.flow_statistics.exploding_percentage
            ));

            summary.push_str(&format!(
                "Graph Depth: {} layers\n",
                analysis.flow_statistics.graph_depth
            ));

            Ok(summary)
        } else {
            Err(TensorError::invalid_operation_simple(
                "No analysis available - run analyze_flow first".to_string(),
            ))
        }
    }
}

impl<T> Default for GradientFlowVisualizer<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + std::cmp::PartialOrd
        + FromPrimitive
        + Signed
        + std::fmt::Debug
        + serde::Serialize
        + serde::de::DeserializeOwned
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::GradientTape;
    use tenflowers_core::Tensor;

    #[test]
    fn test_visualizer_creation() {
        let visualizer = GradientFlowVisualizer::<f32>::new();
        assert!(visualizer.nodes.is_empty());
        assert!(visualizer.edges.is_empty());
        assert!(visualizer.flow_analysis.is_none());
    }

    #[test]
    fn test_visualizer_with_settings() {
        let settings = VisualizationSettings {
            show_gradient_flow: false,
            ..Default::default()
        };

        let visualizer = GradientFlowVisualizer::<f32>::with_settings(settings.clone());
        assert!(!visualizer.settings().show_gradient_flow);
    }

    #[test]
    fn test_compute_gradient_stats() {
        let visualizer = GradientFlowVisualizer::<f32>::new();
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let stats = visualizer.compute_gradient_stats(&tensor).unwrap();
        assert!(stats.mean > 0.0);
        assert!(stats.std > 0.0);
        assert!(stats.l2_norm > 0.0);
    }

    #[test]
    fn test_compute_value_stats() {
        let visualizer = GradientFlowVisualizer::<f32>::new();
        let tensor = Tensor::<f32>::from_vec(vec![0.0, 1.0, 0.0, 2.0], &[2, 2]).unwrap();

        let stats = visualizer.compute_value_stats(&tensor).unwrap();
        assert_eq!(stats.sparsity, 50.0); // 50% zeros
        assert!(stats.norm > 0.0);
    }

    #[test]
    fn test_compute_tensor_magnitude() {
        let visualizer = GradientFlowVisualizer::<f32>::new();
        let tensor = Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2]).unwrap();

        let magnitude = visualizer.compute_tensor_magnitude(&tensor).unwrap();
        assert!((magnitude - 5.0).abs() < 1e-6); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_clear() {
        let mut visualizer = GradientFlowVisualizer::<f32>::new();

        // Manually add some data
        let node = GradientFlowNode::new(
            0,
            "test".to_string(),
            "relu".to_string(),
            vec![10],
            NodeType::Hidden,
        );
        visualizer.nodes.insert(0, node);

        let edge = GradientFlowEdge::new(0, 1, EdgeType::Forward);
        visualizer.edges.push(edge);

        assert!(!visualizer.nodes.is_empty());
        assert!(!visualizer.edges.is_empty());

        visualizer.clear();

        assert!(visualizer.nodes.is_empty());
        assert!(visualizer.edges.is_empty());
        assert!(visualizer.flow_analysis.is_none());
    }
}
