//! Gradient Visualization Data Types
//!
//! This module defines the core data structures used for gradient flow visualization
//! and analysis, including nodes, edges, statistics, and configuration types.

use crate::tape::TensorId;
use serde::{Deserialize, Serialize};

/// Information about a single node in the gradient flow graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFlowNode<T> {
    /// Node ID
    pub id: TensorId,
    /// Node name (for debugging)
    pub name: String,
    /// Operation that produced this node
    pub operation: String,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Gradient magnitude statistics
    pub gradient_stats: GradientStats<T>,
    /// Forward pass value statistics
    pub value_stats: ValueStats<T>,
    /// Node position in the graph (for layout)
    pub position: Option<(f64, f64)>,
    /// Node type (input, hidden, output)
    pub node_type: NodeType,
}

impl<T: Default> GradientFlowNode<T> {
    /// Create a new gradient flow node
    pub fn new(
        id: TensorId,
        name: String,
        operation: String,
        shape: Vec<usize>,
        node_type: NodeType,
    ) -> Self {
        Self {
            id,
            name,
            operation,
            shape,
            gradient_stats: GradientStats::default(),
            value_stats: ValueStats::default(),
            position: None,
            node_type,
        }
    }

    /// Set the position of this node for layout
    pub fn with_position(mut self, x: f64, y: f64) -> Self {
        self.position = Some((x, y));
        self
    }

    /// Update gradient statistics for this node
    pub fn update_gradient_stats(&mut self, gradient_stats: GradientStats<T>) {
        self.gradient_stats = gradient_stats;
    }

    /// Update value statistics for this node
    pub fn update_value_stats(&mut self, value_stats: ValueStats<T>) {
        self.value_stats = value_stats;
    }
}

/// Information about edges in the gradient flow graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFlowEdge {
    /// Source node ID
    pub from: TensorId,
    /// Destination node ID
    pub to: TensorId,
    /// Gradient magnitude flowing through this edge
    pub gradient_magnitude: f64,
    /// Edge weight (influence on gradient flow)
    pub weight: f64,
    /// Edge type classification
    pub edge_type: EdgeType,
    /// Whether this edge is part of the critical path
    pub is_critical: bool,
}

impl GradientFlowEdge {
    /// Create a new gradient flow edge
    pub fn new(from: TensorId, to: TensorId, edge_type: EdgeType) -> Self {
        Self {
            from,
            to,
            gradient_magnitude: 0.0,
            weight: 1.0,
            edge_type,
            is_critical: false,
        }
    }

    /// Mark this edge as critical (important for gradient flow)
    pub fn mark_critical(mut self) -> Self {
        self.is_critical = true;
        self
    }

    /// Set the gradient magnitude flowing through this edge
    pub fn with_gradient_magnitude(mut self, magnitude: f64) -> Self {
        self.gradient_magnitude = magnitude;
        self
    }

    /// Set the weight of this edge
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

/// Statistics about gradients for a node or tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientStats<T> {
    /// Mean gradient magnitude
    pub mean: T,
    /// Standard deviation of gradients
    pub std: T,
    /// Minimum gradient value
    pub min: T,
    /// Maximum gradient value
    pub max: T,
    /// L1 norm of gradients
    pub l1_norm: T,
    /// L2 norm of gradients
    pub l2_norm: T,
    /// Percentage of zero gradients
    pub zero_percentage: f64,
    /// Percentage of infinite/NaN gradients
    pub invalid_percentage: f64,
    /// Whether gradients are vanishing (very small)
    pub is_vanishing: bool,
    /// Whether gradients are exploding (very large)
    pub is_exploding: bool,
}

impl<T> Default for GradientStats<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            mean: T::default(),
            std: T::default(),
            min: T::default(),
            max: T::default(),
            l1_norm: T::default(),
            l2_norm: T::default(),
            zero_percentage: 0.0,
            invalid_percentage: 0.0,
            is_vanishing: false,
            is_exploding: false,
        }
    }
}

/// Statistics about forward pass values for a node or tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueStats<T> {
    /// Mean value
    pub mean: T,
    /// Standard deviation of values
    pub std: T,
    /// Minimum value
    pub min: T,
    /// Maximum value
    pub max: T,
    /// L2 norm of values
    pub norm: T,
    /// Percentage of zero values (sparsity)
    pub sparsity: f64,
}

impl<T> Default for ValueStats<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            mean: T::default(),
            std: T::default(),
            min: T::default(),
            max: T::default(),
            norm: T::default(),
            sparsity: 0.0,
        }
    }
}

/// Type of node in the computation graph
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    /// Input to the computation
    Input,
    /// Hidden layer or intermediate computation
    Hidden,
    /// Output of the computation
    Output,
    /// Parameter (weights, biases)
    Parameter,
    /// Constant value
    Constant,
}

/// Type of edge in the computation graph
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Standard forward pass edge
    Forward,
    /// Skip connection or residual edge
    Skip,
    /// Attention weight edge
    Attention,
}

/// Analysis results for gradient flow
#[derive(Debug, Clone)]
pub struct GradientFlowAnalysis<T> {
    /// Overall health score (0-1, higher is better)
    pub health_score: f64,
    /// Detected issues with gradient flow
    pub issues: Vec<GradientFlowIssue<T>>,
    /// Flow statistics across the entire graph
    pub flow_statistics: FlowStatistics<T>,
    /// Critical path through the graph
    pub critical_path: Vec<TensorId>,
    /// Bottlenecks in gradient flow
    pub bottlenecks: Vec<TensorId>,
}

impl<T> GradientFlowAnalysis<T>
where
    T: Default,
{
    /// Create a new gradient flow analysis
    pub fn new() -> Self {
        Self {
            health_score: 0.0,
            issues: Vec::new(),
            flow_statistics: FlowStatistics::default(),
            critical_path: Vec::new(),
            bottlenecks: Vec::new(),
        }
    }

    /// Add an issue to the analysis
    pub fn add_issue(&mut self, issue: GradientFlowIssue<T>) {
        self.issues.push(issue);
    }

    /// Set the critical path
    pub fn set_critical_path(&mut self, path: Vec<TensorId>) {
        self.critical_path = path;
    }

    /// Add a bottleneck node
    pub fn add_bottleneck(&mut self, node_id: TensorId) {
        self.bottlenecks.push(node_id);
    }

    /// Calculate and update the health score based on issues
    pub fn calculate_health_score(&mut self) {
        if self.issues.is_empty() {
            self.health_score = 1.0;
            return;
        }

        let total_severity: f64 = self
            .issues
            .iter()
            .map(|issue| match issue.severity {
                Severity::Low => 0.1,
                Severity::Medium => 0.3,
                Severity::High => 0.6,
                Severity::Critical => 1.0,
            })
            .sum();

        // Normalize by number of issues and invert (higher score = better health)
        let max_possible_severity = self.issues.len() as f64 * 1.0;
        self.health_score = 1.0 - (total_severity / max_possible_severity).min(1.0);
    }
}

impl<T> Default for GradientFlowAnalysis<T>
where
    T: Default,
{
    fn default() -> Self {
        Self::new()
    }
}

/// A detected issue with gradient flow
#[derive(Debug, Clone)]
pub struct GradientFlowIssue<T> {
    /// Type of issue
    pub issue_type: IssueType,
    /// Severity of the issue
    pub severity: Severity,
    /// Node(s) affected by this issue
    pub affected_nodes: Vec<TensorId>,
    /// Human-readable description
    pub description: String,
    /// Suggested resolution
    pub suggestion: String,
    /// Associated statistics or data
    pub data: Option<T>,
}

impl<T> GradientFlowIssue<T> {
    /// Create a new gradient flow issue
    pub fn new(
        issue_type: IssueType,
        severity: Severity,
        description: String,
        suggestion: String,
    ) -> Self {
        Self {
            issue_type,
            severity,
            affected_nodes: Vec::new(),
            description,
            suggestion,
            data: None,
        }
    }

    /// Add affected nodes to this issue
    pub fn with_affected_nodes(mut self, nodes: Vec<TensorId>) -> Self {
        self.affected_nodes = nodes;
        self
    }

    /// Add associated data to this issue
    pub fn with_data(mut self, data: T) -> Self {
        self.data = Some(data);
        self
    }
}

/// Type of gradient flow issue
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IssueType {
    /// Gradients are vanishing (becoming very small)
    VanishingGradients,
    /// Gradients are exploding (becoming very large)
    ExplodingGradients,
    /// Dead neurons (always output zero)
    DeadNeurons,
    /// Saturated activations
    SaturatedActivations,
    /// Imbalanced gradient magnitudes
    GradientImbalance,
    /// Numerical instability
    NumericalInstability,
}

/// Severity level for issues
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Overall statistics about gradient flow
#[derive(Debug, Clone)]
pub struct FlowStatistics<T> {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Total number of edges
    pub total_edges: usize,
    /// Average gradient magnitude across all nodes
    pub avg_gradient_magnitude: T,
    /// Maximum gradient magnitude
    pub max_gradient_magnitude: T,
    /// Minimum gradient magnitude
    pub min_gradient_magnitude: T,
    /// Percentage of nodes with vanishing gradients
    pub vanishing_percentage: f64,
    /// Percentage of nodes with exploding gradients
    pub exploding_percentage: f64,
    /// Graph depth (longest path from input to output)
    pub graph_depth: usize,
}

impl<T> Default for FlowStatistics<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            total_nodes: 0,
            total_edges: 0,
            avg_gradient_magnitude: T::default(),
            max_gradient_magnitude: T::default(),
            min_gradient_magnitude: T::default(),
            vanishing_percentage: 0.0,
            exploding_percentage: 0.0,
            graph_depth: 0,
        }
    }
}

/// Configuration for visualization rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationSettings {
    /// Color scheme for visualization
    pub color_scheme: ColorScheme,
    /// Layout algorithm to use
    pub layout_algorithm: LayoutAlgorithm,
    /// Output format
    pub output_format: OutputFormat,
    /// Show gradient magnitudes as edge thickness
    pub show_gradient_flow: bool,
    /// Show node statistics
    pub show_node_stats: bool,
    /// Highlight critical path
    pub highlight_critical_path: bool,
    /// Minimum gradient magnitude to display
    pub min_gradient_threshold: f64,
    /// Maximum number of nodes to display
    pub max_nodes: Option<usize>,
}

impl Default for VisualizationSettings {
    fn default() -> Self {
        Self {
            color_scheme: ColorScheme::Viridis,
            layout_algorithm: LayoutAlgorithm::ForceDirected,
            output_format: OutputFormat::SVG,
            show_gradient_flow: true,
            show_node_stats: true,
            highlight_critical_path: true,
            min_gradient_threshold: 1e-6,
            max_nodes: Some(100),
        }
    }
}

/// Color schemes for visualization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorScheme {
    /// Viridis color scheme (good for gradients)
    Viridis,
    /// Plasma color scheme
    Plasma,
    /// Grayscale
    Grayscale,
    /// Custom RGB colors
    Custom {
        primary: (u8, u8, u8),
        secondary: (u8, u8, u8),
    },
}

/// Layout algorithms for graph visualization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayoutAlgorithm {
    /// Force-directed layout (good for general graphs)
    ForceDirected,
    /// Hierarchical layout (good for DAGs)
    Hierarchical,
    /// Circular layout
    Circular,
    /// Grid layout
    Grid,
}

/// Output formats for visualization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Scalable Vector Graphics
    SVG,
    /// JSON format for web visualization
    JSON,
    /// DOT format for Graphviz
    DOT,
    /// Plain text summary
    Text,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_flow_node_creation() {
        let node = GradientFlowNode::<f32>::new(
            0,
            "test_node".to_string(),
            "relu".to_string(),
            vec![32, 64],
            NodeType::Hidden,
        );

        assert_eq!(node.id, 0);
        assert_eq!(node.name, "test_node");
        assert_eq!(node.operation, "relu");
        assert_eq!(node.shape, vec![32, 64]);
        assert_eq!(node.node_type, NodeType::Hidden);
    }

    #[test]
    fn test_gradient_flow_edge_creation() {
        let edge = GradientFlowEdge::new(1, 2, EdgeType::Forward)
            .with_gradient_magnitude(0.5)
            .mark_critical();

        assert_eq!(edge.from, 1);
        assert_eq!(edge.to, 2);
        assert_eq!(edge.gradient_magnitude, 0.5);
        assert!(edge.is_critical);
        assert_eq!(edge.edge_type, EdgeType::Forward);
    }

    #[test]
    fn test_gradient_flow_analysis() {
        let mut analysis = GradientFlowAnalysis::<f32>::new();
        assert_eq!(analysis.health_score, 0.0);
        assert!(analysis.issues.is_empty());

        let issue = GradientFlowIssue::new(
            IssueType::VanishingGradients,
            Severity::High,
            "Gradients are too small".to_string(),
            "Use gradient clipping".to_string(),
        );

        analysis.add_issue(issue);
        analysis.calculate_health_score();

        assert_eq!(analysis.issues.len(), 1);
        assert!(analysis.health_score < 1.0);
    }

    #[test]
    fn test_visualization_settings_default() {
        let settings = VisualizationSettings::default();
        assert_eq!(settings.color_scheme, ColorScheme::Viridis);
        assert_eq!(settings.layout_algorithm, LayoutAlgorithm::ForceDirected);
        assert_eq!(settings.output_format, OutputFormat::SVG);
        assert!(settings.show_gradient_flow);
        assert!(settings.show_node_stats);
    }

    #[test]
    fn test_issue_severity_ordering() {
        assert!(Severity::Low < Severity::Medium);
        assert!(Severity::Medium < Severity::High);
        assert!(Severity::High < Severity::Critical);
    }
}
