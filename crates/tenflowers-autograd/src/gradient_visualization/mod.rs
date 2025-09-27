//! Gradient Flow Visualization
//!
//! This module provides comprehensive gradient flow visualization and analysis
//! capabilities for understanding the behavior of automatic differentiation
//! computations and neural network training dynamics.
//!
//! The module is organized as follows:
//! - `types`: Core data structures for nodes, edges, statistics, and analysis
//! - `visualizer`: Main visualization implementation with gradient flow analysis
//! - `formatting`: Export functionality for various output formats (SVG, JSON, DOT, text)
//!
//! # Examples
//!
//! ```
//! use tenflowers_autograd::gradient_visualization::{
//!     GradientFlowVisualizer, VisualizationSettings, ColorScheme
//! };
//!
//! // Create a visualizer with custom settings
//! let mut settings = VisualizationSettings::default();
//! settings.color_scheme = ColorScheme::Plasma;
//! settings.show_gradient_flow = true;
//!
//! let visualizer = GradientFlowVisualizer::<f32>::with_settings(settings);
//!
//! // Analyze gradient flow from gradient tape
//! // let analysis = visualizer.analyze_gradient_flow(&tape)?;
//! ```

pub mod formatting;
pub mod types;
pub mod visualizer;

// Re-export main types for convenience
pub use formatting::{ExportFormat, VisualizationExporter};
pub use types::{
    ColorScheme, EdgeType, FlowStatistics, GradientFlowAnalysis, GradientFlowEdge,
    GradientFlowIssue, GradientFlowNode, GradientStats, IssueType, LayoutAlgorithm, NodeType,
    OutputFormat, Severity, ValueStats, VisualizationSettings,
};
pub use visualizer::{AnalysisConfig, GradientFlowVisualizer, VisualizationResult};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::tape::TensorId;

    #[test]
    fn test_end_to_end_visualization() {
        let settings = VisualizationSettings::default();
        let visualizer = GradientFlowVisualizer::<f32>::new();

        // Create a simple node for testing
        let node = GradientFlowNode::<f32>::new(
            0,
            "test_node".to_string(),
            "relu".to_string(),
            vec![32, 64],
            NodeType::Hidden,
        );

        // Test that basic structures work
        assert_eq!(node.id, 0);
        assert_eq!(node.name, "test_node");
        assert_eq!(node.operation, "relu");
    }

    #[test]
    fn test_visualization_settings_integration() {
        let mut settings = VisualizationSettings::default();
        settings.color_scheme = ColorScheme::Plasma;
        settings.show_gradient_flow = false;

        let visualizer = GradientFlowVisualizer::<f32>::with_settings(settings);

        // Test that settings are properly integrated
        assert_eq!(visualizer.settings().color_scheme, ColorScheme::Plasma);
        assert!(!visualizer.settings().show_gradient_flow);
    }

    #[test]
    fn test_export_format_compatibility() {
        // Test that all export formats are supported
        let formats = vec![
            ExportFormat::SVG,
            ExportFormat::JSON,
            ExportFormat::DOT,
            ExportFormat::Text,
        ];

        for format in formats {
            let exporter = VisualizationExporter::new(format.clone());
            assert!(
                exporter.is_supported(),
                "Format {:?} should be supported",
                format
            );
        }
    }

    #[test]
    fn test_analysis_result_health_score() {
        let mut analysis = GradientFlowAnalysis::<f32>::new();

        // Test perfect health (no issues)
        analysis.calculate_health_score();
        assert_eq!(analysis.health_score, 1.0);

        // Test with issues
        let issue = GradientFlowIssue::new(
            IssueType::VanishingGradients,
            Severity::High,
            "Test issue".to_string(),
            "Test suggestion".to_string(),
        );

        analysis.add_issue(issue);
        analysis.calculate_health_score();

        assert!(analysis.health_score < 1.0);
        assert!(analysis.health_score >= 0.0);
    }
}
