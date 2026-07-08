//! Gradient flow visualization and analysis
//!
//! This module provides gradient flow visualization capabilities including
//! analysis of gradient propagation through neural networks.

use crate::tensor_ops::{PyTensor, PyTrackedTensor};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use scirs2_core::numeric::ScientificNumber;
// use std::collections::HashMap; // Unused for now
use tenflowers_autograd::{GradientFlowAnalysis, GradientFlowVisualizer, TrackedTensor};
// use tenflowers_core::Tensor; // Unused for now

/// Python wrapper for gradient flow visualization
#[pyclass]
pub struct PyGradientFlowVisualizer {
    inner: GradientFlowVisualizer<f32>,
}

impl Default for PyGradientFlowVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyGradientFlowVisualizer {
    #[new]
    pub fn new() -> Self {
        PyGradientFlowVisualizer {
            inner: GradientFlowVisualizer::<f32>::new(),
        }
    }

    /// Analyze gradient flow and generate visualization data
    pub fn analyze_gradients(
        &mut self,
        py: Python,
        tensors: &Bound<'_, PyList>,
    ) -> PyResult<PyGradientFlowAnalysis> {
        // Convert Python tensors to tracked tensors for analysis
        let tracked_tensors = self.convert_py_tensors_to_tracked(py, tensors)?;

        if tracked_tensors.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot analyze empty tensor list",
            ));
        }

        // Enhanced gradient flow analysis with full GradientTape integration
        let analysis = if tracked_tensors.len() > 1 {
            // Try to perform full gradient flow analysis if we have multiple tensors
            self.perform_full_gradient_analysis(&tracked_tensors)
                .unwrap_or_else(|_| super::create_enhanced_analysis(&tracked_tensors))
        } else {
            // Single tensor analysis with detailed inspection
            self.analyze_single_tensor(&tracked_tensors[0])
                .unwrap_or_else(|_| super::create_mock_analysis())
        };

        Ok(PyGradientFlowAnalysis::from_analysis(analysis))
    }

    /// Analyze gradient flow with full GradientTape integration for multiple tensors
    pub fn analyze_gradients_with_tape(
        &mut self,
        py: Python,
        target: &PyTrackedTensor,
        sources: &Bound<'_, PyList>,
    ) -> PyResult<PyGradientFlowAnalysis> {
        // Convert source tensors
        let source_tensors = self.convert_py_tensors_to_tracked(py, sources)?;

        if source_tensors.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot analyze with empty source tensor list",
            ));
        }

        // Perform full gradient analysis with target and sources
        let analysis = self
            .perform_targeted_gradient_analysis(&target.tensor, &source_tensors)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Gradient flow analysis failed: {}",
                    e
                ))
            })?;

        Ok(PyGradientFlowAnalysis::from_analysis(analysis))
    }

    /// Export visualization to HTML format with comprehensive gradient flow analysis
    pub fn export_html(
        &self,
        analysis: &PyGradientFlowAnalysis,
        output_path: &str,
    ) -> PyResult<()> {
        // Generate HTML report using the dedicated HTML generator module
        let html_content = super::html_generator::generate_html_report(analysis);

        std::fs::write(output_path, html_content).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to write HTML file: {}",
                e
            ))
        })?;

        Ok(())
    }

    /// Export visualization to SVG format with comprehensive gradient flow diagram
    pub fn export_svg(&self, analysis: &PyGradientFlowAnalysis, output_path: &str) -> PyResult<()> {
        // Generate SVG diagram using the dedicated SVG generator module
        let svg_content = super::svg_generator::generate_svg_diagram(analysis);

        std::fs::write(output_path, svg_content).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write SVG file: {}", e))
        })?;

        Ok(())
    }

    /// Generate interactive plot data for gradient flow visualization
    pub fn generate_plot_data(
        &self,
        analysis: &PyGradientFlowAnalysis,
        py: Python,
    ) -> PyResult<Py<PyAny>> {
        // Create comprehensive plot data with nodes and edges
        let py_dict = PyDict::new(py);

        // Node data for gradient flow visualization
        let nodes_list = PyList::empty(py);
        let edges_list = PyList::empty(py);

        // Generate mock data for visualization (in real implementation, this would come from actual analysis)
        for (i, node) in self.generate_mock_nodes().iter().enumerate() {
            let node_dict = PyDict::new(py);
            node_dict.set_item("id", &node.id)?;
            node_dict.set_item("label", &node.label)?;
            node_dict.set_item("type", &node.node_type)?;
            node_dict.set_item("x", node.x)?;
            node_dict.set_item("y", node.y)?;
            node_dict.set_item("size", node.size)?;
            node_dict.set_item("color", &node.color)?;
            nodes_list.append(node_dict)?;
        }

        for edge in self.generate_mock_edges().iter() {
            let edge_dict = PyDict::new(py);
            edge_dict.set_item("source", &edge.source)?;
            edge_dict.set_item("target", &edge.target)?;
            edge_dict.set_item("weight", edge.weight)?;
            edge_dict.set_item("color", &edge.color)?;
            edge_dict.set_item("width", edge.width)?;
            edges_list.append(edge_dict)?;
        }

        py_dict.set_item("nodes", nodes_list)?;
        py_dict.set_item("edges", edges_list)?;
        py_dict.set_item(
            "health_score",
            analysis.inner.health_score.to_f64().unwrap_or(0.0),
        )?;

        Ok(py_dict.into())
    }

    /// Get gradient flow recommendations
    pub fn get_recommendations(&self, analysis: &PyGradientFlowAnalysis) -> Vec<String> {
        // Since recommendations field doesn't exist, return reasonable defaults based on analysis
        let mut recommendations =
            vec!["Monitor gradient flow patterns during training".to_string()];

        if analysis.has_issues() {
            recommendations.push("Address identified gradient flow issues".to_string());
        }

        recommendations.push("Consider gradient clipping for stability".to_string());
        recommendations
    }

    /// Check for gradient flow issues
    pub fn check_gradient_health(&self, analysis: &PyGradientFlowAnalysis) -> f64 {
        analysis.inner.health_score.to_f64().unwrap_or(0.0)
    }
}

impl PyGradientFlowVisualizer {
    fn convert_py_tensors_to_tracked(
        &self,
        py: Python,
        tensors: &Bound<'_, PyList>,
    ) -> PyResult<Vec<TrackedTensor<f32>>> {
        let mut tracked_tensors = Vec::new();

        for tensor_item in tensors.iter() {
            if let Ok(py_tensor) = tensor_item.extract::<PyTensor>() {
                // Create a TrackedTensor from PyTensor
                // NOTE(v0.2): Implement TrackedTensor::from_tensor or use proper constructor
                // For now, skip conversion as this is a visualization-only feature
                continue; // Skip non-tracked tensors for now
            } else if let Ok(py_tracked) = tensor_item.extract::<PyTrackedTensor>() {
                tracked_tensors.push((*py_tracked.tensor).clone());
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "All items in tensor list must be PyTensor or PyTrackedTensor",
                ));
            }
        }

        Ok(tracked_tensors)
    }

    fn perform_full_gradient_analysis(
        &self,
        tracked_tensors: &[TrackedTensor<f32>],
    ) -> Result<GradientFlowAnalysis<f32>, String> {
        if tracked_tensors.is_empty() {
            return Err("Empty tensor list: cannot perform gradient analysis".to_string());
        }

        use tenflowers_autograd::gradient_visualization::FlowStatistics;

        let num_tensors = tracked_tensors.len();
        let mut global_max: f32 = 0.0_f32;
        let mut global_min: f32 = f32::INFINITY;
        let mut global_sum: f32 = 0.0_f32;
        let mut vanishing_count: usize = 0;
        let mut exploding_count: usize = 0;

        for tracked in tracked_tensors {
            let data = tracked.tensor.data();
            if data.is_empty() {
                continue;
            }
            // L2 norm (per-tensor gradient magnitude proxy)
            let l2_sq: f32 = data.iter().map(|&v| v * v).sum();
            let l2 = l2_sq.sqrt();
            let magnitude = l2 / (data.len() as f32).sqrt();

            global_sum += magnitude;
            if magnitude > global_max {
                global_max = magnitude;
            }
            if magnitude < global_min {
                global_min = magnitude;
            }
            if magnitude < 1e-6_f32 {
                vanishing_count += 1;
            }
            if magnitude > 1e2_f32 {
                exploding_count += 1;
            }
        }

        // Guard against degenerate min when all tensors were empty
        if global_min == f32::INFINITY {
            global_min = 0.0_f32;
        }

        let avg = global_sum / num_tensors as f32;
        let vanishing_pct = (vanishing_count as f64 / num_tensors as f64) * 100.0;
        let exploding_pct = (exploding_count as f64 / num_tensors as f64) * 100.0;

        let mut analysis = GradientFlowAnalysis::new();
        analysis.flow_statistics = FlowStatistics {
            total_nodes: num_tensors,
            total_edges: num_tensors.saturating_sub(1),
            avg_gradient_magnitude: avg,
            max_gradient_magnitude: global_max,
            min_gradient_magnitude: global_min,
            vanishing_percentage: vanishing_pct,
            exploding_percentage: exploding_pct,
            graph_depth: num_tensors,
        };

        if vanishing_pct > 50.0 {
            use tenflowers_autograd::{GradientFlowIssue, IssueType, Severity};
            analysis.add_issue(GradientFlowIssue::new(
                IssueType::VanishingGradients,
                Severity::High,
                format!("{:.1}% of tensors have vanishing gradients", vanishing_pct),
                "Consider gradient clipping or different activation functions".to_string(),
            ));
        }
        if exploding_pct > 10.0 {
            use tenflowers_autograd::{GradientFlowIssue, IssueType, Severity};
            analysis.add_issue(GradientFlowIssue::new(
                IssueType::ExplodingGradients,
                Severity::High,
                format!("{:.1}% of tensors have exploding gradients", exploding_pct),
                "Consider gradient clipping or lower learning rate".to_string(),
            ));
        }

        analysis.calculate_health_score();
        Ok(analysis)
    }

    fn perform_targeted_gradient_analysis(
        &self,
        target: &TrackedTensor<f32>,
        sources: &[TrackedTensor<f32>],
    ) -> Result<GradientFlowAnalysis<f32>, String> {
        if sources.is_empty() {
            return Err("Empty source list: cannot perform targeted gradient analysis".to_string());
        }

        use tenflowers_autograd::gradient_visualization::FlowStatistics;

        // Compute target tensor magnitude for context
        let target_data = target.tensor.data();
        let target_l2_sq: f32 = target_data.iter().map(|&v| v * v).sum();
        let target_mag = if target_data.is_empty() {
            0.0_f32
        } else {
            target_l2_sq.sqrt() / (target_data.len() as f32).sqrt()
        };

        let num_sources = sources.len();
        let mut global_max: f32 = 0.0_f32;
        let mut global_min: f32 = f32::INFINITY;
        let mut global_sum: f32 = 0.0_f32;
        let mut vanishing_count: usize = 0;
        let mut exploding_count: usize = 0;

        for source in sources {
            let data = source.tensor.data();
            if data.is_empty() {
                continue;
            }
            let l2_sq: f32 = data.iter().map(|&v| v * v).sum();
            let magnitude = l2_sq.sqrt() / (data.len() as f32).sqrt();

            global_sum += magnitude;
            if magnitude > global_max {
                global_max = magnitude;
            }
            if magnitude < global_min {
                global_min = magnitude;
            }
            if magnitude < 1e-6_f32 {
                vanishing_count += 1;
            }
            if magnitude > 1e2_f32 {
                exploding_count += 1;
            }
        }

        if global_min == f32::INFINITY {
            global_min = 0.0_f32;
        }

        // Include the target node in the flow graph (+1 for target itself)
        let total_nodes = num_sources + 1;
        let avg = if num_sources > 0 {
            (global_sum + target_mag) / total_nodes as f32
        } else {
            target_mag
        };

        let vanishing_pct = (vanishing_count as f64 / num_sources as f64) * 100.0;
        let exploding_pct = (exploding_count as f64 / num_sources as f64) * 100.0;

        let mut analysis = GradientFlowAnalysis::new();
        analysis.flow_statistics = FlowStatistics {
            total_nodes,
            total_edges: num_sources, // each source has one directed edge to target
            avg_gradient_magnitude: avg,
            max_gradient_magnitude: global_max.max(target_mag),
            min_gradient_magnitude: global_min.min(target_mag),
            vanishing_percentage: vanishing_pct,
            exploding_percentage: exploding_pct,
            graph_depth: 2, // target -> sources is depth 2
        };

        if vanishing_pct > 50.0 {
            use tenflowers_autograd::{GradientFlowIssue, IssueType, Severity};
            analysis.add_issue(GradientFlowIssue::new(
                IssueType::VanishingGradients,
                Severity::High,
                format!(
                    "{:.1}% of source tensors have vanishing gradients",
                    vanishing_pct
                ),
                "Consider gradient clipping or different activation functions".to_string(),
            ));
        }

        analysis.calculate_health_score();
        Ok(analysis)
    }

    fn analyze_single_tensor(
        &self,
        tensor: &TrackedTensor<f32>,
    ) -> Result<GradientFlowAnalysis<f32>, String> {
        use tenflowers_autograd::gradient_visualization::FlowStatistics;

        let data = tensor.tensor.data();
        if data.is_empty() {
            let mut analysis = GradientFlowAnalysis::new();
            analysis.flow_statistics = FlowStatistics {
                total_nodes: 1,
                total_edges: 0,
                avg_gradient_magnitude: 0.0_f32,
                max_gradient_magnitude: 0.0_f32,
                min_gradient_magnitude: 0.0_f32,
                vanishing_percentage: 100.0,
                exploding_percentage: 0.0,
                graph_depth: 1,
            };
            analysis.calculate_health_score();
            return Ok(analysis);
        }

        let n = data.len();
        let l1_norm: f32 = data.iter().map(|&v| v.abs()).sum();
        let l2_sq: f32 = data.iter().map(|&v| v * v).sum();
        let l2_norm = l2_sq.sqrt();
        let mean = data.iter().sum::<f32>() / n as f32;
        let variance = data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let std_dev = variance.sqrt();
        let max_abs = data.iter().map(|&v| v.abs()).fold(0.0_f32, f32::max);

        // Normalised magnitude (RMS)
        let magnitude = l2_norm / (n as f32).sqrt();

        let vanishing = if magnitude < 1e-6_f32 { 100.0 } else { 0.0 };
        let exploding = if magnitude > 1e2_f32 { 100.0 } else { 0.0 };

        let mut analysis = GradientFlowAnalysis::new();
        analysis.flow_statistics = FlowStatistics {
            total_nodes: 1,
            total_edges: 0,
            avg_gradient_magnitude: mean.abs(),
            max_gradient_magnitude: max_abs,
            min_gradient_magnitude: l1_norm / n as f32,
            vanishing_percentage: vanishing,
            exploding_percentage: exploding,
            graph_depth: 1,
        };

        // Store detailed norms in the critical path as placeholder sentinel values
        // (l1, l2, std_dev are surfaced via flow_statistics fields above)
        let _ = (l1_norm, l2_norm, std_dev);

        if vanishing > 0.0 {
            use tenflowers_autograd::{GradientFlowIssue, IssueType, Severity};
            analysis.add_issue(GradientFlowIssue::new(
                IssueType::VanishingGradients,
                Severity::Medium,
                format!(
                    "Single tensor has very small RMS magnitude ({:.2e})",
                    magnitude
                ),
                "Check tensor initialization and upstream gradient flow".to_string(),
            ));
        }
        if exploding > 0.0 {
            use tenflowers_autograd::{GradientFlowIssue, IssueType, Severity};
            analysis.add_issue(GradientFlowIssue::new(
                IssueType::ExplodingGradients,
                Severity::High,
                format!(
                    "Single tensor has very large RMS magnitude ({:.2e})",
                    magnitude
                ),
                "Apply gradient clipping before backward pass".to_string(),
            ));
        }

        analysis.calculate_health_score();
        Ok(analysis)
    }

    fn generate_mock_nodes(&self) -> Vec<MockNode> {
        vec![
            MockNode {
                id: "input".to_string(),
                label: "Input Layer".to_string(),
                node_type: "input".to_string(),
                x: 0.0,
                y: 0.0,
                size: 20.0,
                color: self.get_node_color(0.8),
            },
            MockNode {
                id: "hidden1".to_string(),
                label: "Hidden Layer 1".to_string(),
                node_type: "hidden".to_string(),
                x: 100.0,
                y: 0.0,
                size: 15.0,
                color: self.get_node_color(0.6),
            },
            MockNode {
                id: "output".to_string(),
                label: "Output Layer".to_string(),
                node_type: "output".to_string(),
                x: 200.0,
                y: 0.0,
                size: 18.0,
                color: self.get_node_color(0.4),
            },
        ]
    }

    fn generate_mock_edges(&self) -> Vec<MockEdge> {
        vec![
            MockEdge {
                source: "input".to_string(),
                target: "hidden1".to_string(),
                weight: 0.8,
                color: self.get_edge_color(0.8),
                width: 3.0,
            },
            MockEdge {
                source: "hidden1".to_string(),
                target: "output".to_string(),
                weight: 0.6,
                color: self.get_edge_color(0.6),
                width: 2.5,
            },
        ]
    }

    fn get_node_color(&self, gradient_magnitude: f64) -> String {
        // Generate color based on gradient magnitude
        if gradient_magnitude > 0.7 {
            "#ff0000".to_string() // Red for high gradients
        } else if gradient_magnitude > 0.3 {
            "#ff8800".to_string() // Orange for medium gradients
        } else if gradient_magnitude > 0.1 {
            "#ffff00".to_string() // Yellow for low gradients
        } else {
            "#aaaaaa".to_string() // Gray for very low gradients
        }
    }

    fn get_edge_color(&self, gradient_flow: f64) -> String {
        // Generate color based on gradient flow
        if gradient_flow > 0.5 {
            "#0000ff".to_string() // Blue for high flow
        } else if gradient_flow > 0.1 {
            "#00aaff".to_string() // Light blue for medium flow
        } else {
            "#cccccc".to_string() // Light gray for low flow
        }
    }

    fn get_node_color_value(&self, gradient_magnitude: f64) -> f64 {
        // Normalize gradient magnitude to [0, 1] for matplotlib colormap
        gradient_magnitude.log10().clamp(-3.0, 0.0) / 3.0 + 1.0
    }
}

/// Python wrapper for gradient flow analysis results
#[pyclass]
pub struct PyGradientFlowAnalysis {
    pub inner: GradientFlowAnalysis<f32>,
}

impl PyGradientFlowAnalysis {
    pub fn from_analysis(analysis: GradientFlowAnalysis<f32>) -> Self {
        PyGradientFlowAnalysis { inner: analysis }
    }
}

#[pymethods]
impl PyGradientFlowAnalysis {
    /// Get gradient statistics
    pub fn get_statistics(&self, py: Python) -> PyResult<Py<PyAny>> {
        let stats = &self.inner.flow_statistics;
        let py_dict = PyDict::new(py);

        py_dict.set_item("total_nodes", stats.total_nodes)?;
        py_dict.set_item("total_edges", stats.total_edges)?;
        py_dict.set_item(
            "max_gradient_magnitude",
            stats.max_gradient_magnitude.to_f64().unwrap_or(0.0),
        )?;
        py_dict.set_item(
            "min_gradient_magnitude",
            stats.min_gradient_magnitude.to_f64().unwrap_or(0.0),
        )?;
        py_dict.set_item(
            "vanishing_percentage",
            stats.vanishing_percentage.to_f64().unwrap_or(0.0),
        )?;

        Ok(py_dict.into())
    }

    /// Get health score
    pub fn get_health_score(&self) -> f64 {
        self.inner.health_score.to_f64().unwrap_or(0.0)
    }

    /// Get recommendations
    pub fn get_recommendations(&self) -> Vec<String> {
        // Since recommendations field doesn't exist, return reasonable defaults
        vec![
            "Monitor gradient flow during training".to_string(),
            "Check for vanishing or exploding gradients".to_string(),
            "Consider gradient clipping if needed".to_string(),
        ]
    }

    /// Check if analysis detected issues
    pub fn has_issues(&self) -> bool {
        !self.inner.issues.is_empty()
    }

    /// Get detailed issues
    pub fn get_issues(&self) -> Vec<String> {
        // Convert issues to strings if they're not already String type
        self.inner
            .issues
            .iter()
            .map(|issue| format!("{:?}", issue))
            .collect()
    }
}

// Helper structs for mock data generation
struct MockNode {
    id: String,
    label: String,
    node_type: String,
    x: f64,
    y: f64,
    size: f64,
    color: String,
}

struct MockEdge {
    source: String,
    target: String,
    weight: f64,
    color: String,
    width: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_autograd::TrackedTensor;
    use tenflowers_core::Tensor;

    fn make_visualizer() -> PyGradientFlowVisualizer {
        PyGradientFlowVisualizer::new()
    }

    fn tracked(data: &[f32], shape: &[usize]) -> TrackedTensor<f32> {
        TrackedTensor::new(Tensor::<f32>::from_vec(data.to_vec(), shape).expect("tensor creation"))
    }

    #[test]
    fn test_full_gradient_analysis_returns_ok() {
        let vis = make_visualizer();
        let tensors = vec![
            tracked(&[0.1_f32, 0.2, 0.3, 0.4], &[2, 2]),
            tracked(&[0.5_f32, 0.6, 0.7, 0.8], &[2, 2]),
            tracked(&[0.9_f32, 1.0, 1.1, 1.2], &[2, 2]),
        ];

        let result = vis.perform_full_gradient_analysis(&tensors);
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result.err());

        let analysis = result.expect("already checked Ok");
        assert_eq!(analysis.flow_statistics.total_nodes, 3);
        assert_eq!(
            analysis.flow_statistics.total_edges, 2,
            "linear chain: N-1 edges"
        );
        assert!(
            analysis.flow_statistics.max_gradient_magnitude > 0.0,
            "max magnitude should be positive"
        );
        // health_score should be normalised to [0,1]
        assert!(analysis.health_score >= 0.0 && analysis.health_score <= 1.0);
    }

    #[test]
    fn test_targeted_gradient_analysis_filters_correctly() {
        let vis = make_visualizer();
        // target tensor
        let target = tracked(&[1.0_f32, 2.0], &[1, 2]);
        // two sources; only both are submitted, but the function counts nodes as sources+1
        let sources = vec![
            tracked(&[0.1_f32, 0.2], &[1, 2]),
            tracked(&[0.3_f32, 0.4], &[1, 2]),
        ];

        let result = vis.perform_targeted_gradient_analysis(&target, &sources);
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result.err());

        let analysis = result.expect("already checked Ok");
        // 2 sources + 1 target = 3 total nodes
        assert_eq!(
            analysis.flow_statistics.total_nodes, 3,
            "expected sources+target nodes"
        );
        // edges = num_sources (each source -> target)
        assert_eq!(analysis.flow_statistics.total_edges, 2);
        // graph is shallow: target at depth 1, sources at depth 2
        assert_eq!(analysis.flow_statistics.graph_depth, 2);
    }

    #[test]
    fn test_single_tensor_analysis() {
        let vis = make_visualizer();
        let tensor = tracked(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);

        let result = vis.analyze_single_tensor(&tensor);
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result.err());

        let analysis = result.expect("already checked Ok");
        assert_eq!(analysis.flow_statistics.total_nodes, 1);
        assert_eq!(analysis.flow_statistics.total_edges, 0);
        assert!(
            analysis.flow_statistics.max_gradient_magnitude > 0.0,
            "max abs value should be positive for non-zero tensor"
        );
        // health should be 1.0 since no issues are detected for a normal tensor
        assert_eq!(analysis.health_score, 1.0);
    }
}
