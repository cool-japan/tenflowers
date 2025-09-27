//! Gradient Visualization Formatting and Output
//!
//! This module provides functionality for formatting and exporting gradient flow
//! visualizations in various output formats including SVG, JSON, DOT, and text.

use super::types::{
    ColorScheme, GradientFlowAnalysis, GradientFlowEdge, GradientFlowNode, LayoutAlgorithm,
    OutputFormat, VisualizationSettings,
};
use crate::tape::TensorId;
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Write;
use tenflowers_core::{Result, TensorError};

/// Export format for visualization output
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportFormat {
    /// Scalable Vector Graphics
    SVG,
    /// JSON format for web visualization
    JSON,
    /// DOT format for Graphviz
    DOT,
    /// Plain text summary
    Text,
}

/// Visualization exporter for different output formats
pub struct VisualizationExporter {
    format: ExportFormat,
}

impl VisualizationExporter {
    /// Create a new exporter with the specified format
    pub fn new(format: ExportFormat) -> Self {
        Self { format }
    }

    /// Check if the format is supported
    pub fn is_supported(&self) -> bool {
        true // All formats are supported
    }

    /// Export visualization data to string
    pub fn export<T>(
        &self,
        nodes: &HashMap<TensorId, GradientFlowNode<T>>,
        edges: &[GradientFlowEdge],
        analysis: Option<&GradientFlowAnalysis<T>>,
        settings: &VisualizationSettings,
    ) -> Result<String>
    where
        T: Float + std::fmt::Debug + std::fmt::Display + Clone,
    {
        match self.format {
            ExportFormat::SVG => GradientFlowFormatter::export_svg(nodes, edges, settings),
            ExportFormat::JSON => GradientFlowFormatter::export_json(nodes, edges, analysis),
            ExportFormat::DOT => GradientFlowFormatter::export_dot(nodes, edges, settings),
            ExportFormat::Text => GradientFlowFormatter::export_text(nodes, edges, analysis),
        }
    }
}

/// Formatter for gradient flow visualizations
pub struct GradientFlowFormatter;

impl GradientFlowFormatter {
    /// Export visualization in the specified format
    pub fn export<T>(
        nodes: &HashMap<TensorId, GradientFlowNode<T>>,
        edges: &[GradientFlowEdge],
        analysis: Option<&GradientFlowAnalysis<T>>,
        settings: &VisualizationSettings,
    ) -> Result<String>
    where
        T: Float + std::fmt::Debug + std::fmt::Display + Clone,
    {
        match settings.output_format {
            OutputFormat::SVG => Self::export_svg(nodes, edges, settings),
            OutputFormat::JSON => Self::export_json(nodes, edges, analysis),
            OutputFormat::DOT => Self::export_dot(nodes, edges, settings),
            OutputFormat::Text => Self::export_text(nodes, edges, analysis),
        }
    }

    /// Export as SVG format
    fn export_svg<T>(
        nodes: &HashMap<TensorId, GradientFlowNode<T>>,
        edges: &[GradientFlowEdge],
        settings: &VisualizationSettings,
    ) -> Result<String>
    where
        T: Float + std::fmt::Debug + std::fmt::Display + Clone,
    {
        let mut svg = String::new();

        // SVG header
        writeln!(&mut svg, r#"<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">"#)
            .map_err(Self::fmt_err_to_tensor_err)?;

        // Add styles
        writeln!(&mut svg, "<defs><style>").map_err(Self::fmt_err_to_tensor_err)?;
        writeln!(&mut svg, ".node {{ stroke: #333; stroke-width: 2; }}")
            .map_err(Self::fmt_err_to_tensor_err)?;
        writeln!(&mut svg, ".edge {{ stroke: #666; stroke-width: 1; }}")
            .map_err(Self::fmt_err_to_tensor_err)?;
        writeln!(
            &mut svg,
            ".critical-path {{ stroke: #ff0000; stroke-width: 3; }}"
        )
        .map_err(Self::fmt_err_to_tensor_err)?;
        writeln!(
            &mut svg,
            ".text {{ font-family: Arial, sans-serif; font-size: 12px; }}"
        )
        .map_err(Self::fmt_err_to_tensor_err)?;
        writeln!(&mut svg, "</style></defs>").map_err(Self::fmt_err_to_tensor_err)?;

        // Calculate layout positions
        let positions = Self::calculate_layout(nodes, edges, &settings.layout_algorithm)?;

        // Draw edges first (so they appear behind nodes)
        for edge in edges {
            if let (Some(from_pos), Some(to_pos)) =
                (positions.get(&edge.from), positions.get(&edge.to))
            {
                let stroke_width = if settings.show_gradient_flow {
                    (edge.gradient_magnitude * 5.0).clamp(1.0, 10.0)
                } else {
                    2.0
                };

                let class = if edge.is_critical && settings.highlight_critical_path {
                    "critical-path"
                } else {
                    "edge"
                };

                writeln!(
                    &mut svg,
                    r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="{}" stroke-width="{}" />"#,
                    from_pos.0, from_pos.1, to_pos.0, to_pos.1, class, stroke_width
                )?;
            }
        }

        // Draw nodes
        for (node_id, node) in nodes {
            if let Some(pos) = positions.get(node_id) {
                let color = Self::get_node_color(&node.gradient_stats, &settings.color_scheme);
                let radius = 15.0;

                writeln!(
                    &mut svg,
                    r#"<circle cx="{}" cy="{}" r="{}" fill="{}" class="node" />"#,
                    pos.0, pos.1, radius, color
                )?;

                if settings.show_node_stats {
                    writeln!(
                        &mut svg,
                        r#"<text x="{}" y="{}" class="text" text-anchor="middle">{}</text>"#,
                        pos.0,
                        pos.1 - radius - 5.0,
                        node.name
                    )?;
                }
            }
        }

        writeln!(&mut svg, "</svg>")?;
        Ok(svg)
    }

    /// Export as JSON format
    fn export_json<T>(
        nodes: &HashMap<TensorId, GradientFlowNode<T>>,
        edges: &[GradientFlowEdge],
        analysis: Option<&GradientFlowAnalysis<T>>,
    ) -> Result<String>
    where
        T: Float + std::fmt::Debug + std::fmt::Display + Clone,
    {
        let mut json = String::new();

        writeln!(&mut json, "{{")?;
        writeln!(&mut json, r#"  "nodes": ["#)?;

        let mut node_first = true;
        for (node_id, node) in nodes {
            if !node_first {
                writeln!(&mut json, ",")?;
            }
            writeln!(&mut json, "    {{")?;
            writeln!(&mut json, r#"      "id": {},"#, node_id)?;
            writeln!(&mut json, r#"      "name": "{}","#, node.name)?;
            writeln!(&mut json, r#"      "operation": "{}","#, node.operation)?;
            writeln!(&mut json, r#"      "type": "{:?}","#, node.node_type)?;
            writeln!(&mut json, r#"      "shape": {:?},"#, node.shape)?;

            if let Some(pos) = node.position {
                writeln!(&mut json, r#"      "position": [{}, {}]"#, pos.0, pos.1)?;
            } else {
                writeln!(&mut json, r#"      "position": null"#)?;
            }

            write!(&mut json, "    }}")?;
            node_first = false;
        }

        writeln!(&mut json)?;
        writeln!(&mut json, "  ],")?;
        writeln!(&mut json, r#"  "edges": ["#)?;

        let mut edge_first = true;
        for edge in edges {
            if !edge_first {
                writeln!(&mut json, ",")?;
            }
            writeln!(&mut json, "    {{")?;
            writeln!(&mut json, r#"      "from": {},"#, edge.from)?;
            writeln!(&mut json, r#"      "to": {},"#, edge.to)?;
            writeln!(
                &mut json,
                r#"      "gradient_magnitude": {},"#,
                edge.gradient_magnitude
            )?;
            writeln!(&mut json, r#"      "weight": {},"#, edge.weight)?;
            writeln!(&mut json, r#"      "type": "{:?}","#, edge.edge_type)?;
            writeln!(&mut json, r#"      "is_critical": {}"#, edge.is_critical)?;
            write!(&mut json, "    }}")?;
            edge_first = false;
        }

        writeln!(&mut json)?;
        writeln!(&mut json, "  ]")?;

        if let Some(analysis) = analysis {
            writeln!(&mut json, ",")?;
            writeln!(&mut json, r#"  "analysis": {{"#)?;
            writeln!(
                &mut json,
                r#"    "health_score": {},"#,
                analysis.health_score
            )?;
            writeln!(
                &mut json,
                r#"    "issues_count": {},"#,
                analysis.issues.len()
            )?;
            writeln!(
                &mut json,
                r#"    "graph_depth": {}"#,
                analysis.flow_statistics.graph_depth
            )?;
            writeln!(&mut json, "  }}")?;
        }

        writeln!(&mut json, "}}")?;
        Ok(json)
    }

    /// Export as DOT format for Graphviz
    fn export_dot<T>(
        nodes: &HashMap<TensorId, GradientFlowNode<T>>,
        edges: &[GradientFlowEdge],
        settings: &VisualizationSettings,
    ) -> Result<String>
    where
        T: Float + std::fmt::Debug + std::fmt::Display + Clone,
    {
        let mut dot = String::new();

        writeln!(&mut dot, "digraph GradientFlow {{")?;
        writeln!(&mut dot, "  rankdir=TB;")?;
        writeln!(&mut dot, "  node [shape=circle, style=filled];")?;

        // Add nodes
        for (node_id, node) in nodes {
            let color = Self::get_node_color_hex(&node.gradient_stats, &settings.color_scheme);
            writeln!(
                &mut dot,
                r#"  {} [label="{}", fillcolor="{}"];"#,
                node_id, node.name, color
            )?;
        }

        // Add edges
        for edge in edges {
            let style = if edge.is_critical && settings.highlight_critical_path {
                r#"style=bold, color=red"#
            } else {
                r#"color=gray"#
            };

            let width = if settings.show_gradient_flow {
                format!(
                    "penwidth={:.1}",
                    (edge.gradient_magnitude * 3.0).clamp(0.5, 5.0)
                )
            } else {
                "penwidth=1.0".to_string()
            };

            writeln!(
                &mut dot,
                "  {} -> {} [{}, {}];",
                edge.from, edge.to, style, width
            )?;
        }

        writeln!(&mut dot, "}}")?;
        Ok(dot)
    }

    /// Export as plain text summary
    fn export_text<T>(
        nodes: &HashMap<TensorId, GradientFlowNode<T>>,
        edges: &[GradientFlowEdge],
        analysis: Option<&GradientFlowAnalysis<T>>,
    ) -> Result<String>
    where
        T: Float + std::fmt::Debug + std::fmt::Display + Clone,
    {
        let mut text = String::new();

        writeln!(&mut text, "=== Gradient Flow Analysis Report ===")?;
        writeln!(&mut text)?;

        if let Some(analysis) = analysis {
            writeln!(&mut text, "Health Score: {:.2}", analysis.health_score)?;
            writeln!(&mut text, "Issues Found: {}", analysis.issues.len())?;
            writeln!(
                &mut text,
                "Graph Depth: {}",
                analysis.flow_statistics.graph_depth
            )?;
            writeln!(
                &mut text,
                "Total Nodes: {}",
                analysis.flow_statistics.total_nodes
            )?;
            writeln!(
                &mut text,
                "Total Edges: {}",
                analysis.flow_statistics.total_edges
            )?;
            writeln!(
                &mut text,
                "Vanishing Gradients: {:.1}%",
                analysis.flow_statistics.vanishing_percentage
            )?;
            writeln!(
                &mut text,
                "Exploding Gradients: {:.1}%",
                analysis.flow_statistics.exploding_percentage
            )?;
            writeln!(&mut text)?;

            if !analysis.issues.is_empty() {
                writeln!(&mut text, "=== Issues Detected ===")?;
                for (i, issue) in analysis.issues.iter().enumerate() {
                    writeln!(
                        &mut text,
                        "{}. [{:?}] {}",
                        i + 1,
                        issue.severity,
                        issue.description
                    )?;
                    writeln!(&mut text, "   Suggestion: {}", issue.suggestion)?;
                    if !issue.affected_nodes.is_empty() {
                        writeln!(&mut text, "   Affected nodes: {:?}", issue.affected_nodes)?;
                    }
                    writeln!(&mut text)?;
                }
            }
        }

        writeln!(&mut text, "=== Nodes ({}) ===", nodes.len())?;
        for (node_id, node) in nodes {
            writeln!(
                &mut text,
                "Node {}: {} [{}] - Shape: {:?}",
                node_id, node.name, node.operation, node.shape
            )?;
        }

        writeln!(&mut text)?;
        writeln!(&mut text, "=== Edges ({}) ===", edges.len())?;
        for edge in edges {
            writeln!(
                &mut text,
                "{} -> {} (magnitude: {:.4}, critical: {})",
                edge.from, edge.to, edge.gradient_magnitude, edge.is_critical
            )?;
        }

        Ok(text)
    }

    /// Calculate layout positions for nodes
    fn calculate_layout<T>(
        nodes: &HashMap<TensorId, GradientFlowNode<T>>,
        edges: &[GradientFlowEdge],
        algorithm: &LayoutAlgorithm,
    ) -> Result<HashMap<TensorId, (f64, f64)>>
    where
        T: Float + std::fmt::Debug + std::fmt::Display + Clone,
    {
        match algorithm {
            LayoutAlgorithm::ForceDirected => Self::force_directed_layout(nodes, edges),
            LayoutAlgorithm::Hierarchical => Self::hierarchical_layout(nodes, edges),
            LayoutAlgorithm::Circular => Self::circular_layout(nodes),
            LayoutAlgorithm::Grid => Self::grid_layout(nodes),
        }
    }

    /// Force-directed layout algorithm (simplified)
    fn force_directed_layout<T>(
        nodes: &HashMap<TensorId, GradientFlowNode<T>>,
        _edges: &[GradientFlowEdge],
    ) -> Result<HashMap<TensorId, (f64, f64)>>
    where
        T: Float + std::fmt::Debug + std::fmt::Display + Clone,
    {
        let mut positions = HashMap::new();
        let center_x = 400.0;
        let center_y = 300.0;
        let radius = 200.0;

        let node_count = nodes.len();
        for (i, (node_id, _node)) in nodes.iter().enumerate() {
            let angle = (i as f64 * 2.0 * std::f64::consts::PI) / node_count as f64;
            let x = center_x + radius * angle.cos();
            let y = center_y + radius * angle.sin();
            positions.insert(*node_id, (x, y));
        }

        Ok(positions)
    }

    /// Hierarchical layout algorithm
    fn hierarchical_layout<T>(
        nodes: &HashMap<TensorId, GradientFlowNode<T>>,
        edges: &[GradientFlowEdge],
    ) -> Result<HashMap<TensorId, (f64, f64)>>
    where
        T: Float + std::fmt::Debug + std::fmt::Display + Clone,
    {
        let mut positions = HashMap::new();
        let mut levels: HashMap<TensorId, usize> = HashMap::new();

        // Calculate node levels based on dependencies
        for node_id in nodes.keys() {
            let level = Self::calculate_node_level(*node_id, edges, &mut HashMap::new());
            levels.insert(*node_id, level);
        }

        let _max_level = levels.values().copied().max().unwrap_or(0);

        // Group nodes by level
        let mut level_groups: HashMap<usize, Vec<TensorId>> = HashMap::new();
        for (node_id, level) in levels {
            level_groups.entry(level).or_default().push(node_id);
        }

        // Position nodes
        for (level, node_ids) in level_groups {
            let y = 50.0 + (level as f64 * 100.0);
            let nodes_in_level = node_ids.len();

            for (i, node_id) in node_ids.iter().enumerate() {
                let x = if nodes_in_level == 1 {
                    400.0
                } else {
                    100.0 + (i as f64 * 600.0) / (nodes_in_level - 1) as f64
                };
                positions.insert(*node_id, (x, y));
            }
        }

        Ok(positions)
    }

    /// Circular layout algorithm
    fn circular_layout<T>(
        nodes: &HashMap<TensorId, GradientFlowNode<T>>,
    ) -> Result<HashMap<TensorId, (f64, f64)>>
    where
        T: Float + std::fmt::Debug + std::fmt::Display + Clone,
    {
        let mut positions = HashMap::new();
        let center_x = 400.0;
        let center_y = 300.0;
        let radius = 200.0;

        let node_count = nodes.len();
        for (i, (node_id, _node)) in nodes.iter().enumerate() {
            let angle = (i as f64 * 2.0 * std::f64::consts::PI) / node_count as f64;
            let x = center_x + radius * angle.cos();
            let y = center_y + radius * angle.sin();
            positions.insert(*node_id, (x, y));
        }

        Ok(positions)
    }

    /// Grid layout algorithm
    fn grid_layout<T>(
        nodes: &HashMap<TensorId, GradientFlowNode<T>>,
    ) -> Result<HashMap<TensorId, (f64, f64)>>
    where
        T: Float + std::fmt::Debug + std::fmt::Display + Clone,
    {
        let mut positions = HashMap::new();
        let cols = (nodes.len() as f64).sqrt().ceil() as usize;
        let cell_width = 800.0 / cols as f64;
        let cell_height = 600.0 / cols as f64;

        for (i, (node_id, _node)) in nodes.iter().enumerate() {
            let col = i % cols;
            let row = i / cols;
            let x = (col as f64 * cell_width) + (cell_width / 2.0);
            let y = (row as f64 * cell_height) + (cell_height / 2.0);
            positions.insert(*node_id, (x, y));
        }

        Ok(positions)
    }

    /// Calculate the level of a node in the hierarchy
    fn calculate_node_level(
        node_id: TensorId,
        edges: &[GradientFlowEdge],
        visited: &mut HashMap<TensorId, usize>,
    ) -> usize {
        if let Some(&level) = visited.get(&node_id) {
            return level;
        }

        let mut max_parent_level = 0;
        for edge in edges {
            if edge.to == node_id {
                let parent_level = Self::calculate_node_level(edge.from, edges, visited);
                max_parent_level = max_parent_level.max(parent_level);
            }
        }

        let level = max_parent_level + 1;
        visited.insert(node_id, level);
        level
    }

    /// Get color for a node based on gradient statistics
    fn get_node_color<T>(_stats: &super::types::GradientStats<T>, scheme: &ColorScheme) -> String
    where
        T: Float + std::fmt::Debug + std::fmt::Display + Clone,
    {
        match scheme {
            ColorScheme::Viridis => "#440154".to_string(),
            ColorScheme::Plasma => "#0d0887".to_string(),
            ColorScheme::Grayscale => "#808080".to_string(),
            ColorScheme::Custom { primary, .. } => {
                format!("rgb({}, {}, {})", primary.0, primary.1, primary.2)
            }
        }
    }

    /// Get hex color for a node based on gradient statistics
    fn get_node_color_hex<T>(
        _stats: &super::types::GradientStats<T>,
        scheme: &ColorScheme,
    ) -> String
    where
        T: Float + std::fmt::Debug + std::fmt::Display + Clone,
    {
        match scheme {
            ColorScheme::Viridis => "#440154".to_string(),
            ColorScheme::Plasma => "#0d0887".to_string(),
            ColorScheme::Grayscale => "#808080".to_string(),
            ColorScheme::Custom { primary, .. } => {
                format!("#{:02x}{:02x}{:02x}", primary.0, primary.1, primary.2)
            }
        }
    }

    /// Convert fmt::Error to TensorError
    fn fmt_err_to_tensor_err(e: std::fmt::Error) -> TensorError {
        TensorError::invalid_operation_simple(format!("Formatting error: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use super::*;
    use crate::tape::TensorId;

    #[test]
    fn test_grid_layout() {
        let mut nodes = HashMap::new();
        for i in 0..4 {
            let node = GradientFlowNode::<f32>::new(
                i,
                format!("node_{}", i),
                "test".to_string(),
                vec![1],
                NodeType::Hidden,
            );
            nodes.insert(i, node);
        }

        let positions = GradientFlowFormatter::grid_layout(&nodes).unwrap();
        assert_eq!(positions.len(), 4);

        // All positions should be valid
        for (_, pos) in positions {
            assert!(pos.0 >= 0.0 && pos.0 <= 800.0);
            assert!(pos.1 >= 0.0 && pos.1 <= 600.0);
        }
    }

    #[test]
    fn test_circular_layout() {
        let mut nodes = HashMap::new();
        for i in 0..3 {
            let node = GradientFlowNode::<f32>::new(
                i,
                format!("node_{}", i),
                "test".to_string(),
                vec![1],
                NodeType::Hidden,
            );
            nodes.insert(i, node);
        }

        let positions = GradientFlowFormatter::circular_layout(&nodes).unwrap();
        assert_eq!(positions.len(), 3);

        // Check that positions form a circle around (400, 300)
        for (_, pos) in positions {
            let distance = ((pos.0 - 400.0).powi(2) + (pos.1 - 300.0).powi(2)).sqrt();
            assert!((distance - 200.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_export_text() {
        let mut nodes = HashMap::new();
        let node = GradientFlowNode::<f32>::new(
            0,
            "test_node".to_string(),
            "relu".to_string(),
            vec![10, 20],
            NodeType::Hidden,
        );
        nodes.insert(0, node);

        let edges = vec![GradientFlowEdge::new(0, 1, EdgeType::Forward)];

        let text = GradientFlowFormatter::export_text(&nodes, &edges, None).unwrap();
        assert!(text.contains("Gradient Flow Analysis Report"));
        assert!(text.contains("test_node"));
        assert!(text.contains("relu"));
    }

    #[test]
    fn test_export_dot() {
        let mut nodes = HashMap::new();
        let node = GradientFlowNode::<f32>::new(
            0,
            "test_node".to_string(),
            "relu".to_string(),
            vec![10],
            NodeType::Hidden,
        );
        nodes.insert(0, node);

        let edges = vec![GradientFlowEdge::new(0, 1, EdgeType::Forward)];
        let settings = VisualizationSettings::default();

        let dot = GradientFlowFormatter::export_dot(&nodes, &edges, &settings).unwrap();
        assert!(dot.contains("digraph GradientFlow"));
        assert!(dot.contains("test_node"));
        assert!(dot.contains("0 -> 1"));
    }
}
