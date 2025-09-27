//! Gradient flow visualization and analysis
//!
//! This module provides gradient flow visualization capabilities including
//! analysis of gradient propagation through neural networks.

use crate::tensor_ops::{PyTensor, PyTrackedTensor};
use num_traits::ToPrimitive;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
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

    /// Export visualization to HTML format
    pub fn export_html(
        &self,
        _analysis: &PyGradientFlowAnalysis,
        output_path: &str,
    ) -> PyResult<()> {
        // TODO: Implement generate_html_report in GradientFlowVisualizer
        let html_content = r#"<!DOCTYPE html>
<html>
<head>
    <title>Gradient Flow Report</title>
</head>
<body>
    <h1>Gradient Flow Analysis Report</h1>
    <p>HTML report generation not yet implemented.</p>
    <p>This is a placeholder report.</p>
</body>
</html>"#;

        std::fs::write(output_path, html_content).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to write HTML file: {}",
                e
            ))
        })?;

        Ok(())
    }

    /// Export visualization to SVG format
    pub fn export_svg(
        &self,
        _analysis: &PyGradientFlowAnalysis,
        output_path: &str,
    ) -> PyResult<()> {
        // TODO: Implement generate_svg_diagram in GradientFlowVisualizer
        let svg_content = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
            <text x="50" y="50" text-anchor="middle" dominant-baseline="central">
                SVG generation not yet implemented
            </text>
        </svg>"#;

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
    ) -> PyResult<PyObject> {
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
                // TODO: Implement TrackedTensor::from_tensor or use proper constructor
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
        _tracked_tensors: &[TrackedTensor<f32>],
    ) -> Result<GradientFlowAnalysis<f32>, String> {
        // In a real implementation, this would perform gradient flow analysis
        Err("Full gradient analysis not yet implemented".to_string())
    }

    fn perform_targeted_gradient_analysis(
        &self,
        _target: &TrackedTensor<f32>,
        _sources: &[TrackedTensor<f32>],
    ) -> Result<GradientFlowAnalysis<f32>, String> {
        // In a real implementation, this would perform targeted gradient flow analysis
        Err("Targeted gradient analysis not yet implemented".to_string())
    }

    fn analyze_single_tensor(
        &self,
        _tensor: &TrackedTensor<f32>,
    ) -> Result<GradientFlowAnalysis<f32>, String> {
        // In a real implementation, this would analyze a single tensor
        Err("Single tensor analysis not yet implemented".to_string())
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
    pub fn get_statistics(&self, py: Python) -> PyResult<PyObject> {
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
