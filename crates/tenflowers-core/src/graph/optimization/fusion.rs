//! Operation fusion optimization pass
//!
//! This module provides operation fusion capabilities that combine multiple
//! operations into fused kernels for improved performance.

use super::passes::{get_node_outputs, OptimizationPass};
use crate::graph::{Graph, NodeId};
use crate::Result;

/// Operation fusion pass
/// Combines multiple operations into fused kernels
pub struct OperationFusionPass;

impl OptimizationPass for OperationFusionPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;
        let mut fusion_candidates = Vec::new();

        // Look for fusable patterns
        for node in graph.nodes() {
            if let crate::graph::NodeType::Operation(op_name) = &node.op_type {
                match op_name.as_str() {
                    "MatMul" => {
                        // Look for MatMul + Add (bias) pattern
                        if let Some(add_node) = self.find_add_after_matmul(graph, node.id) {
                            fusion_candidates.push(FusionCandidate {
                                pattern: FusionPattern::MatMulAdd,
                                nodes: vec![node.id, add_node],
                            });
                            changed = true;
                        }
                    }
                    "Add" => {
                        // Look for Add + ReLU pattern
                        if let Some(relu_node) = self.find_activation_after_add(graph, node.id) {
                            fusion_candidates.push(FusionCandidate {
                                pattern: FusionPattern::AddActivation,
                                nodes: vec![node.id, relu_node],
                            });
                            changed = true;
                        }
                    }
                    "Conv2D" => {
                        // Look for Conv + BatchNorm + ReLU pattern
                        if let Some(fusion) = self.find_conv_bn_relu_pattern(graph, node.id) {
                            fusion_candidates.push(fusion);
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Apply fusions
        for candidate in fusion_candidates {
            self.apply_fusion(graph, &candidate)?;
        }

        Ok(changed)
    }

    fn name(&self) -> &str {
        "OperationFusion"
    }

    fn is_applicable(&self, graph: &Graph) -> bool {
        graph.node_count() > 1
    }

    fn priority(&self) -> u32 {
        120 // Medium priority - after CSE but before layout optimization
    }
}

impl Default for OperationFusionPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OperationFusionPass {
    pub fn new() -> Self {
        Self
    }

    fn find_add_after_matmul(&self, graph: &Graph, matmul_node: NodeId) -> Option<NodeId> {
        let outputs = get_node_outputs(graph, matmul_node);
        for output_id in outputs {
            if let Some(node) = graph.get_node(output_id) {
                if let crate::graph::NodeType::Operation(op_name) = &node.op_type {
                    if op_name == "Add" {
                        return Some(output_id);
                    }
                }
            }
        }
        None
    }

    fn find_activation_after_add(&self, graph: &Graph, add_node: NodeId) -> Option<NodeId> {
        let outputs = get_node_outputs(graph, add_node);
        for output_id in outputs {
            if let Some(node) = graph.get_node(output_id) {
                if let crate::graph::NodeType::Operation(op_name) = &node.op_type {
                    if matches!(op_name.as_str(), "ReLU" | "Sigmoid" | "Tanh") {
                        return Some(output_id);
                    }
                }
            }
        }
        None
    }

    fn find_conv_bn_relu_pattern(
        &self,
        graph: &Graph,
        conv_node: NodeId,
    ) -> Option<FusionCandidate> {
        let mut current_node = conv_node;
        let mut pattern_nodes = vec![current_node];

        // Look for BatchNorm after Conv
        if let Some(bn_node) = self.find_next_node_with_type(graph, current_node, "BatchNorm") {
            pattern_nodes.push(bn_node);
            current_node = bn_node;

            // Look for ReLU after BatchNorm
            if let Some(relu_node) = self.find_next_node_with_type(graph, current_node, "ReLU") {
                pattern_nodes.push(relu_node);

                return Some(FusionCandidate {
                    pattern: FusionPattern::ConvBatchNormReLU,
                    nodes: pattern_nodes,
                });
            }
        }

        None
    }

    fn find_next_node_with_type(
        &self,
        graph: &Graph,
        node_id: NodeId,
        op_type: &str,
    ) -> Option<NodeId> {
        let outputs = get_node_outputs(graph, node_id);
        for output_id in outputs {
            if let Some(node) = graph.get_node(output_id) {
                if let crate::graph::NodeType::Operation(op_name) = &node.op_type {
                    if op_name == op_type {
                        return Some(output_id);
                    }
                }
            }
        }
        None
    }

    fn apply_fusion(&self, graph: &mut Graph, candidate: &FusionCandidate) -> Result<()> {
        match candidate.pattern {
            FusionPattern::MatMulAdd => self.fuse_matmul_add(graph, &candidate.nodes),
            FusionPattern::AddActivation => self.fuse_add_activation(graph, &candidate.nodes),
            FusionPattern::ConvBatchNormReLU => self.fuse_conv_bn_relu(graph, &candidate.nodes),
        }
    }

    fn fuse_matmul_add(&self, graph: &mut Graph, nodes: &[NodeId]) -> Result<()> {
        if nodes.len() != 2 {
            return Ok(());
        }

        let matmul_node = nodes[0];
        let add_node = nodes[1];

        // Replace the MatMul node with a Dense operation that includes bias
        if let Some(matmul_node_ref) = graph.get_node_mut(matmul_node) {
            matmul_node_ref.op_type = crate::graph::NodeType::Operation("Dense".to_string());
            // In a real implementation, we'd also update the attributes to include bias handling
        }

        // Remove the Add node and redirect its outputs to the MatMul node
        graph.redirect_node_outputs(add_node, matmul_node)?;
        graph.remove_node(add_node)?;

        Ok(())
    }

    fn fuse_add_activation(&self, graph: &mut Graph, nodes: &[NodeId]) -> Result<()> {
        if nodes.len() != 2 {
            return Ok(());
        }

        let add_node = nodes[0];
        let activation_node = nodes[1];

        // Get the activation type
        let activation_type = if let Some(node) = graph.get_node(activation_node) {
            if let crate::graph::NodeType::Operation(op_name) = &node.op_type {
                op_name.clone()
            } else {
                return Ok(());
            }
        } else {
            return Ok(());
        };

        // Replace the Add node with a fused Add+Activation operation
        if let Some(add_node_ref) = graph.get_node_mut(add_node) {
            add_node_ref.op_type =
                crate::graph::NodeType::Operation(format!("Add{activation_type}"));
        }

        // Remove the activation node and redirect its outputs to the add node
        graph.redirect_node_outputs(activation_node, add_node)?;
        graph.remove_node(activation_node)?;

        Ok(())
    }

    fn fuse_conv_bn_relu(&self, graph: &mut Graph, nodes: &[NodeId]) -> Result<()> {
        if nodes.len() != 3 {
            return Ok(());
        }

        let conv_node = nodes[0];
        let bn_node = nodes[1];
        let relu_node = nodes[2];

        // Replace the Conv node with a fused ConvBatchNormReLU operation
        if let Some(conv_node_ref) = graph.get_node_mut(conv_node) {
            conv_node_ref.op_type =
                crate::graph::NodeType::Operation("ConvBatchNormReLU".to_string());
            // In a real implementation, we'd merge the BatchNorm parameters into the conv attributes
        }

        // Remove the BatchNorm and ReLU nodes
        graph.redirect_node_outputs(relu_node, conv_node)?;
        graph.remove_node(bn_node)?;
        graph.remove_node(relu_node)?;

        Ok(())
    }
}

/// Fusion patterns that can be optimized
#[derive(Debug, Clone)]
pub enum FusionPattern {
    MatMulAdd,         // MatMul + Add -> Dense layer
    AddActivation,     // Add + Activation -> Fused add+activation
    ConvBatchNormReLU, // Conv + BatchNorm + ReLU -> Fused conv block
}

/// Candidate for operation fusion
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FusionCandidate {
    pub pattern: FusionPattern,
    pub nodes: Vec<NodeId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_fusion_pass() {
        let pass = OperationFusionPass::new();
        assert_eq!(pass.name(), "OperationFusion");
        assert_eq!(pass.priority(), 120);
    }

    #[test]
    fn test_fusion_patterns() {
        // Test that fusion patterns can be created
        let pattern1 = FusionPattern::MatMulAdd;
        let _pattern2 = FusionPattern::AddActivation;
        let _pattern3 = FusionPattern::ConvBatchNormReLU;

        // Test fusion candidate creation
        let candidate = FusionCandidate {
            pattern: pattern1,
            nodes: vec![1, 2],
        };

        assert_eq!(candidate.nodes.len(), 2);
    }
}
