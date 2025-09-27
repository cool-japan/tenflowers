//! Graph optimization utilities for ultra-gradient computation

use crate::tape::{Operation, TapeNode};
use std::collections::HashMap;

/// Graph optimization utilities
pub struct GraphOptimizer;

impl GraphOptimizer {
    /// Check if two operations can be fused together
    pub fn can_fuse_operations(op1: &TapeNode, op2: &TapeNode) -> bool {
        // Check if operations are compatible for fusion
        match (&op1.operation, &op2.operation) {
            (Operation::Add { .. }, Operation::Mul { .. })
            | (Operation::Mul { .. }, Operation::Add { .. }) => true,
            (Operation::MatMul { .. }, Operation::Add { .. }) => true,
            // Add more fusion patterns as needed
            _ => false,
        }
    }

    /// Analyze common fusion patterns (Conv+BN+ReLU, etc.)
    pub fn analyze_fusion_patterns(nodes: &[TapeNode]) -> Vec<Vec<usize>> {
        let mut fusion_groups = Vec::new();
        let mut visited = vec![false; nodes.len()];

        for i in 0..nodes.len() {
            if visited[i] {
                continue;
            }

            let mut current_group = vec![i];
            visited[i] = true;

            // Look for fusion patterns
            for j in (i + 1)..nodes.len() {
                if visited[j] {
                    continue;
                }

                if Self::can_fuse_operations(&nodes[i], &nodes[j]) {
                    current_group.push(j);
                    visited[j] = true;
                }
            }

            if current_group.len() > 1 {
                fusion_groups.push(current_group);
            }
        }

        fusion_groups
    }

    /// Check for convolution + batch normalization + activation pattern
    pub fn is_conv_bn_relu_pattern(nodes: &[TapeNode], start_idx: usize) -> bool {
        if start_idx + 2 >= nodes.len() {
            return false;
        }

        // Check for convolution + batch normalization + activation pattern
        // This is a simplified check - in a real implementation you'd match specific operation types
        matches!(
            (
                &nodes[start_idx].operation,
                &nodes[start_idx + 1].operation,
                &nodes[start_idx + 2].operation
            ),
            (
                Operation::Conv2D { .. },
                Operation::BatchNorm { .. },
                Operation::Relu { .. }
            )
        )
    }

    /// Check for attention mechanism patterns
    pub fn is_attention_pattern(nodes: &[TapeNode], start_idx: usize) -> bool {
        if start_idx + 3 >= nodes.len() {
            return false;
        }

        // Query, Key, Value computation pattern
        matches!(
            (
                &nodes[start_idx].operation,
                &nodes[start_idx + 1].operation,
                &nodes[start_idx + 2].operation,
                &nodes[start_idx + 3].operation
            ),
            (
                Operation::MatMul { .. },
                Operation::MatMul { .. },
                Operation::MatMul { .. },
                Operation::Softmax { .. }
            )
        )
    }

    /// Compute topological ordering for optimal gradient computation
    pub fn compute_topological_order(nodes: &[TapeNode]) -> Vec<usize> {
        let mut order = Vec::new();
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        let mut adj_list: HashMap<usize, Vec<usize>> = HashMap::new();

        // Build dependency graph
        for (i, node) in nodes.iter().enumerate() {
            in_degree.insert(i, 0);
            adj_list.insert(i, Vec::new());

            // Add dependencies based on parent relationships
            for &parent_id in &node.parents {
                if let Some(parent_idx) = nodes.iter().position(|n| n.id == parent_id) {
                    adj_list.get_mut(&parent_idx).unwrap().push(i);
                    *in_degree.get_mut(&i).unwrap() += 1;
                }
            }
        }

        // Kahn's algorithm for topological sorting
        let mut queue: std::collections::VecDeque<usize> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(&node, _)| node)
            .collect();

        while let Some(current) = queue.pop_front() {
            order.push(current);

            if let Some(neighbors) = adj_list.get(&current) {
                for &neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(&neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        order
    }
}
