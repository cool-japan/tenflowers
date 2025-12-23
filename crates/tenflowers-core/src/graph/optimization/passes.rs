//! Core optimization passes and trait definitions
//!
//! This module provides the fundamental optimization pass trait and basic passes
//! like constant folding, common subexpression elimination, and dead code elimination.

use crate::graph::{Graph, NodeId};
use crate::{Result, TensorError};
use std::collections::{HashMap, HashSet};

/// Helper function to get input node IDs for a given node
pub(crate) fn get_node_inputs(graph: &Graph, node_id: NodeId) -> Vec<NodeId> {
    if let Some(node) = graph.get_node(node_id) {
        node.inputs
            .iter()
            .filter_map(|&edge_id| graph.get_edge(edge_id).map(|edge| edge.from_node))
            .collect()
    } else {
        Vec::new()
    }
}

/// Helper function to get output node IDs for a given node
pub(crate) fn get_node_outputs(graph: &Graph, node_id: NodeId) -> Vec<NodeId> {
    if let Some(node) = graph.get_node(node_id) {
        node.outputs
            .iter()
            .filter_map(|&edge_id| graph.get_edge(edge_id).map(|edge| edge.to_node))
            .collect()
    } else {
        Vec::new()
    }
}

/// Graph optimization pass trait
pub trait OptimizationPass {
    /// Apply the optimization pass to the graph
    fn apply(&self, graph: &mut Graph) -> Result<bool>;

    /// Get the name of this optimization pass
    fn name(&self) -> &str;

    /// Check if this pass can be safely applied
    fn is_applicable(&self, graph: &Graph) -> bool;

    /// Get the pass priority (higher = run first)
    fn priority(&self) -> u32 {
        100
    }
}

/// Constant folding optimization pass
/// Evaluates constant expressions at compile time
pub struct ConstantFoldingPass;

impl OptimizationPass for ConstantFoldingPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;
        let mut nodes_to_remove = Vec::new();
        let mut constant_nodes = HashSet::new();

        // Find all constant nodes (nodes with no inputs or only constant inputs)
        for node in graph.nodes() {
            if self.is_constant_node(graph, node.id) {
                constant_nodes.insert(node.id);
            }
        }

        // For each constant node that has arithmetic operations, try to fold
        for &node_id in &constant_nodes {
            if let Some(node) = graph.get_node(node_id) {
                if let crate::graph::NodeType::Operation(op_name) = &node.op_type {
                    match op_name.as_str() {
                        "Add" | "Mul" | "Sub" | "Div" => {
                            if self.can_fold_binary_op(graph, node_id) {
                                // Replace the operation with its constant result
                                // In a real implementation, we'd evaluate the operation
                                // For now, just mark it as foldable
                                nodes_to_remove.push(node_id);
                                changed = true;
                            }
                        }
                        "MatMul" => {
                            if self.can_fold_matmul(graph, node_id) {
                                nodes_to_remove.push(node_id);
                                changed = true;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Evaluate and replace folded nodes with constants
        for node_id in nodes_to_remove {
            if let Some(result_tensor) = self.evaluate_constant_operation(graph, node_id)? {
                graph.replace_with_constant(node_id, result_tensor)?;
            }
        }

        Ok(changed)
    }

    fn name(&self) -> &str {
        "ConstantFolding"
    }

    fn is_applicable(&self, graph: &Graph) -> bool {
        // Always applicable if there are nodes
        graph.node_count() > 0
    }

    fn priority(&self) -> u32 {
        200 // High priority - run early
    }
}

impl Default for ConstantFoldingPass {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstantFoldingPass {
    pub fn new() -> Self {
        Self
    }

    #[allow(clippy::only_used_in_recursion)]
    fn is_constant_node(&self, graph: &Graph, node_id: NodeId) -> bool {
        if let Some(node) = graph.get_node(node_id) {
            // Check if node is a constant or all inputs are constants
            if matches!(node.op_type, crate::graph::NodeType::Constant) {
                return true;
            }

            let inputs = get_node_inputs(graph, node_id);
            for input_id in inputs {
                if !self.is_constant_node(graph, input_id) {
                    return false;
                }
            }
            !node.inputs.is_empty()
        } else {
            false
        }
    }

    fn can_fold_binary_op(&self, graph: &Graph, node_id: NodeId) -> bool {
        let inputs = get_node_inputs(graph, node_id);
        inputs.len() == 2 && inputs.iter().all(|&id| self.is_constant_node(graph, id))
    }

    fn can_fold_matmul(&self, graph: &Graph, node_id: NodeId) -> bool {
        let inputs = get_node_inputs(graph, node_id);
        inputs.len() == 2 && inputs.iter().all(|&id| self.is_constant_node(graph, id))
    }

    fn evaluate_constant_operation(
        &self,
        graph: &Graph,
        node_id: NodeId,
    ) -> Result<Option<crate::tensor::Tensor<f32>>> {
        use crate::ops::{binary, matmul};

        let node = graph.get_node(node_id).ok_or_else(|| {
            TensorError::invalid_argument(format!("Node {node_id} does not exist"))
        })?;

        if let crate::graph::NodeType::Operation(op_name) = &node.op_type {
            let input_node_ids = get_node_inputs(graph, node_id);

            // Get input tensors
            let input_tensors: std::result::Result<Vec<_>, crate::error::TensorError> =
                input_node_ids
                    .iter()
                    .map(|&input_id| self.get_constant_tensor(graph, input_id))
                    .collect();

            let inputs = input_tensors?;

            match op_name.as_str() {
                "Add" if inputs.len() == 2 => {
                    let result = binary::add(&inputs[0], &inputs[1])?;
                    Ok(Some(result))
                }
                "Sub" if inputs.len() == 2 => {
                    let result = binary::sub(&inputs[0], &inputs[1])?;
                    Ok(Some(result))
                }
                "Mul" if inputs.len() == 2 => {
                    let result = binary::mul(&inputs[0], &inputs[1])?;
                    Ok(Some(result))
                }
                "Div" if inputs.len() == 2 => {
                    let result = binary::div(&inputs[0], &inputs[1])?;
                    Ok(Some(result))
                }
                "MatMul" if inputs.len() == 2 => {
                    let result = matmul::matmul(&inputs[0], &inputs[1])?;
                    Ok(Some(result))
                }
                _ => Ok(None), // Operation not supported for constant folding
            }
        } else {
            Ok(None)
        }
    }

    fn get_constant_tensor(
        &self,
        graph: &Graph,
        node_id: NodeId,
    ) -> std::result::Result<crate::tensor::Tensor<f32>, crate::error::TensorError> {
        let node = graph.get_node(node_id).ok_or_else(|| {
            TensorError::invalid_argument(format!("Node {node_id} does not exist"))
        })?;

        if let crate::graph::NodeType::Constant = &node.op_type {
            if let Some(crate::graph::AttributeValue::Tensor(tensor)) = node.attributes.get("value")
            {
                Ok(tensor.clone())
            } else {
                Err(TensorError::invalid_argument(
                    "Constant node missing tensor value".to_string(),
                ))
            }
        } else {
            Err(TensorError::invalid_argument(format!(
                "Node {node_id} is not a constant"
            )))
        }
    }
}

/// Common Subexpression Elimination pass
/// Removes duplicate computations
pub struct CSEPass;

impl OptimizationPass for CSEPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;
        let mut expression_map: HashMap<String, NodeId> = HashMap::new();
        let mut nodes_to_redirect = Vec::new();

        // Find duplicate expressions
        for node in graph.nodes() {
            let expr_key = self.compute_expression_key(graph, node.id);

            if let Some(&existing_node_id) = expression_map.get(&expr_key) {
                // Found duplicate - redirect this node's outputs to the existing one
                nodes_to_redirect.push((node.id, existing_node_id));
                changed = true;
            } else {
                expression_map.insert(expr_key, node.id);
            }
        }

        // Redirect duplicate nodes
        for (duplicate_node, canonical_node) in nodes_to_redirect {
            // Redirect all outputs from duplicate to canonical node
            graph.redirect_node_outputs(duplicate_node, canonical_node)?;
            // Remove the duplicate node
            graph.remove_node(duplicate_node)?;
        }

        Ok(changed)
    }

    fn name(&self) -> &str {
        "CommonSubexpressionElimination"
    }

    fn is_applicable(&self, graph: &Graph) -> bool {
        graph.node_count() > 1
    }

    fn priority(&self) -> u32 {
        150 // Medium-high priority
    }
}

impl Default for CSEPass {
    fn default() -> Self {
        Self::new()
    }
}

impl CSEPass {
    pub fn new() -> Self {
        Self
    }

    #[allow(clippy::only_used_in_recursion)]
    fn compute_expression_key(&self, graph: &Graph, node_id: NodeId) -> String {
        if let Some(node) = graph.get_node(node_id) {
            let inputs = get_node_inputs(graph, node_id);
            let input_keys: Vec<String> = inputs
                .iter()
                .map(|&id| self.compute_expression_key(graph, id))
                .collect();

            format!("{:?}({})", node.op_type, input_keys.join(","))
        } else {
            format!("node_{node_id}")
        }
    }
}

/// Dead code elimination pass
/// Removes nodes that don't contribute to any output
pub struct DeadCodeEliminationPass;

impl OptimizationPass for DeadCodeEliminationPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;
        let mut reachable = HashSet::new();

        // Mark all nodes reachable from outputs
        for node in graph.nodes() {
            if self.is_output_node(graph, node.id) {
                self.mark_reachable(graph, node.id, &mut reachable);
            }
        }

        // Remove unreachable nodes
        let mut nodes_to_remove = Vec::new();
        for node in graph.nodes() {
            if !reachable.contains(&node.id) {
                nodes_to_remove.push(node.id);
                changed = true;
            }
        }

        for node_id in nodes_to_remove {
            graph.remove_node(node_id)?;
        }

        Ok(changed)
    }

    fn name(&self) -> &str {
        "DeadCodeElimination"
    }

    fn is_applicable(&self, graph: &Graph) -> bool {
        graph.node_count() > 0
    }

    fn priority(&self) -> u32 {
        50 // Low priority - run last
    }
}

impl Default for DeadCodeEliminationPass {
    fn default() -> Self {
        Self::new()
    }
}

impl DeadCodeEliminationPass {
    pub fn new() -> Self {
        Self
    }

    fn is_output_node(&self, graph: &Graph, node_id: NodeId) -> bool {
        // A node is an output if it has no successors or is explicitly marked as output
        let outputs = get_node_outputs(graph, node_id);
        outputs.is_empty() || self.is_marked_as_output(graph, node_id)
    }

    fn is_marked_as_output(&self, _graph: &Graph, _node_id: NodeId) -> bool {
        // In a real implementation, check if node is marked as a graph output
        false
    }

    #[allow(clippy::only_used_in_recursion)]
    fn mark_reachable(&self, graph: &Graph, node_id: NodeId, reachable: &mut HashSet<NodeId>) {
        if reachable.contains(&node_id) {
            return;
        }

        reachable.insert(node_id);

        // Mark all inputs as reachable
        for input_id in get_node_inputs(graph, node_id) {
            self.mark_reachable(graph, input_id, reachable);
        }
    }
}

/// Algebraic simplification pass
/// Simplifies algebraic expressions using mathematical identities
pub struct AlgebraicSimplificationPass;

impl OptimizationPass for AlgebraicSimplificationPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;
        let mut nodes_to_simplify = Vec::new();

        // Find nodes that can be simplified
        for node in graph.nodes() {
            if let Some(simplification) = self.find_simplification(graph, node.id) {
                nodes_to_simplify.push((node.id, simplification));
                changed = true;
            }
        }

        // Apply simplifications
        for (node_id, simplification) in nodes_to_simplify {
            self.apply_simplification(graph, node_id, simplification)?;
        }

        Ok(changed)
    }

    fn name(&self) -> &str {
        "AlgebraicSimplification"
    }

    fn is_applicable(&self, graph: &Graph) -> bool {
        graph.node_count() > 0
    }

    fn priority(&self) -> u32 {
        180 // High priority, after constant folding
    }
}

impl Default for AlgebraicSimplificationPass {
    fn default() -> Self {
        Self::new()
    }
}

impl AlgebraicSimplificationPass {
    pub fn new() -> Self {
        Self
    }

    fn find_simplification(&self, graph: &Graph, node_id: NodeId) -> Option<SimplificationType> {
        let node = graph.get_node(node_id)?;

        if let crate::graph::NodeType::Operation(op_name) = &node.op_type {
            let inputs = get_node_inputs(graph, node_id);

            match op_name.as_str() {
                "Add" if inputs.len() == 2 => {
                    // Check for x + 0 = x or 0 + x = x
                    if self.is_zero_constant(graph, inputs[1]) {
                        return Some(SimplificationType::ReplaceWithInput(0));
                    }
                    if self.is_zero_constant(graph, inputs[0]) {
                        return Some(SimplificationType::ReplaceWithInput(1));
                    }
                    // Check for x + x = 2*x
                    if inputs[0] == inputs[1] {
                        return Some(SimplificationType::ConvertToMultiply(2.0));
                    }
                }
                "Mul" if inputs.len() == 2 => {
                    // Check for x * 1 = x or 1 * x = x
                    if self.is_one_constant(graph, inputs[1]) {
                        return Some(SimplificationType::ReplaceWithInput(0));
                    }
                    if self.is_one_constant(graph, inputs[0]) {
                        return Some(SimplificationType::ReplaceWithInput(1));
                    }
                    // Check for x * 0 = 0 or 0 * x = 0
                    if self.is_zero_constant(graph, inputs[0])
                        || self.is_zero_constant(graph, inputs[1])
                    {
                        return Some(SimplificationType::ReplaceWithConstant(0.0));
                    }
                }
                "Sub" if inputs.len() == 2 => {
                    // Check for x - 0 = x
                    if self.is_zero_constant(graph, inputs[1]) {
                        return Some(SimplificationType::ReplaceWithInput(0));
                    }
                    // Check for x - x = 0
                    if inputs[0] == inputs[1] {
                        return Some(SimplificationType::ReplaceWithConstant(0.0));
                    }
                }
                "Div" if inputs.len() == 2 => {
                    // Check for x / 1 = x
                    if self.is_one_constant(graph, inputs[1]) {
                        return Some(SimplificationType::ReplaceWithInput(0));
                    }
                    // Check for x / x = 1
                    if inputs[0] == inputs[1] {
                        return Some(SimplificationType::ReplaceWithConstant(1.0));
                    }
                }
                "Pow" if inputs.len() == 2 => {
                    // Check for x^1 = x
                    if self.is_one_constant(graph, inputs[1]) {
                        return Some(SimplificationType::ReplaceWithInput(0));
                    }
                    // Check for x^0 = 1
                    if self.is_zero_constant(graph, inputs[1]) {
                        return Some(SimplificationType::ReplaceWithConstant(1.0));
                    }
                }
                _ => {}
            }
        }

        None
    }

    fn is_zero_constant(&self, graph: &Graph, node_id: NodeId) -> bool {
        self.is_scalar_constant(graph, node_id, 0.0)
    }

    fn is_one_constant(&self, graph: &Graph, node_id: NodeId) -> bool {
        self.is_scalar_constant(graph, node_id, 1.0)
    }

    fn is_scalar_constant(&self, graph: &Graph, node_id: NodeId, value: f32) -> bool {
        if let Some(node) = graph.get_node(node_id) {
            if let crate::graph::NodeType::Constant = &node.op_type {
                if let Some(crate::graph::AttributeValue::Tensor(tensor)) =
                    node.attributes.get("value")
                {
                    // Check if tensor is scalar with the expected value
                    if tensor.shape().size() == 1 {
                        if let Some(slice) = tensor.as_slice() {
                            return (slice[0] - value).abs() < 1e-6;
                        }
                    }
                }
            }
        }
        false
    }

    fn apply_simplification(
        &self,
        graph: &mut Graph,
        node_id: NodeId,
        simplification: SimplificationType,
    ) -> Result<()> {
        match simplification {
            SimplificationType::ReplaceWithInput(input_idx) => {
                let inputs = get_node_inputs(graph, node_id);
                if input_idx < inputs.len() {
                    graph.redirect_node_outputs(node_id, inputs[input_idx])?;
                    graph.remove_node(node_id)?;
                }
            }
            SimplificationType::ReplaceWithConstant(value) => {
                let const_tensor = crate::tensor::Tensor::from_scalar(value);
                graph.replace_with_constant(node_id, const_tensor)?;
            }
            SimplificationType::ConvertToMultiply(_scale) => {
                // This would convert x + x to 2*x
                // Implementation would require creating new nodes
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum SimplificationType {
    ReplaceWithInput(usize),
    ReplaceWithConstant(f32),
    ConvertToMultiply(f32),
}

/// Operation scheduling pass
/// Reorders operations for better performance and parallelism
pub struct OperationSchedulingPass {
    prefer_memory_locality: bool,
    enable_parallelization: bool,
}

impl OptimizationPass for OperationSchedulingPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;

        // Compute operation dependencies
        let dependencies = self.compute_dependencies(graph);

        // Find operations that can be reordered
        let reorderable_ops = self.find_reorderable_operations(graph, &dependencies);

        if !reorderable_ops.is_empty() {
            // Apply scheduling heuristics
            self.apply_scheduling_heuristics(graph, &reorderable_ops)?;
            changed = true;
        }

        Ok(changed)
    }

    fn name(&self) -> &str {
        "OperationScheduling"
    }

    fn is_applicable(&self, graph: &Graph) -> bool {
        graph.node_count() > 2
    }

    fn priority(&self) -> u32 {
        120 // Medium priority
    }
}

impl Default for OperationSchedulingPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OperationSchedulingPass {
    pub fn new() -> Self {
        Self {
            prefer_memory_locality: true,
            enable_parallelization: true,
        }
    }

    pub fn with_config(prefer_memory_locality: bool, enable_parallelization: bool) -> Self {
        Self {
            prefer_memory_locality,
            enable_parallelization,
        }
    }

    fn compute_dependencies(&self, graph: &Graph) -> HashMap<NodeId, Vec<NodeId>> {
        let mut deps = HashMap::new();

        for node in graph.nodes() {
            let inputs = get_node_inputs(graph, node.id);
            deps.insert(node.id, inputs);
        }

        deps
    }

    fn find_reorderable_operations(
        &self,
        graph: &Graph,
        dependencies: &HashMap<NodeId, Vec<NodeId>>,
    ) -> Vec<(NodeId, NodeId)> {
        let mut reorderable = Vec::new();

        // Find pairs of operations that can be swapped
        let nodes: Vec<_> = graph.nodes().collect();
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let node_a = nodes[i].id;
                let node_b = nodes[j].id;

                if self.can_reorder(node_a, node_b, dependencies) {
                    reorderable.push((node_a, node_b));
                }
            }
        }

        reorderable
    }

    fn can_reorder(
        &self,
        node_a: NodeId,
        node_b: NodeId,
        dependencies: &HashMap<NodeId, Vec<NodeId>>,
    ) -> bool {
        // Check if nodes have no dependency relationship
        let deps_a = dependencies
            .get(&node_a)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let deps_b = dependencies
            .get(&node_b)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        // A and B can be reordered if neither depends on the other
        !deps_a.contains(&node_b) && !deps_b.contains(&node_a)
    }

    fn apply_scheduling_heuristics(
        &self,
        _graph: &mut Graph,
        _reorderable: &[(NodeId, NodeId)],
    ) -> Result<()> {
        // Apply heuristics like:
        // 1. Group operations by device
        // 2. Minimize memory transfers
        // 3. Maximize parallelism
        // For now, this is a placeholder
        Ok(())
    }
}

/// Strength reduction pass
/// Replaces expensive operations with cheaper equivalents
pub struct StrengthReductionPass;

impl OptimizationPass for StrengthReductionPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;
        let mut reductions = Vec::new();

        // Find operations that can be reduced
        for node in graph.nodes() {
            if let Some(reduction) = self.find_reduction(graph, node.id) {
                reductions.push((node.id, reduction));
                changed = true;
            }
        }

        // Apply reductions
        for (node_id, reduction) in reductions {
            self.apply_reduction(graph, node_id, reduction)?;
        }

        Ok(changed)
    }

    fn name(&self) -> &str {
        "StrengthReduction"
    }

    fn is_applicable(&self, graph: &Graph) -> bool {
        graph.node_count() > 0
    }

    fn priority(&self) -> u32 {
        140 // Medium-high priority
    }
}

impl Default for StrengthReductionPass {
    fn default() -> Self {
        Self::new()
    }
}

impl StrengthReductionPass {
    pub fn new() -> Self {
        Self
    }

    fn find_reduction(&self, graph: &Graph, node_id: NodeId) -> Option<ReductionType> {
        let node = graph.get_node(node_id)?;

        if let crate::graph::NodeType::Operation(op_name) = &node.op_type {
            let inputs = get_node_inputs(graph, node_id);

            match op_name.as_str() {
                "Pow" if inputs.len() == 2 => {
                    // x^2 -> x * x (faster on some hardware)
                    if self.is_constant_value(graph, inputs[1], 2.0) {
                        return Some(ReductionType::Square);
                    }
                    // x^0.5 -> sqrt(x)
                    if self.is_constant_value(graph, inputs[1], 0.5) {
                        return Some(ReductionType::SquareRoot);
                    }
                }
                "Div" if inputs.len() == 2 => {
                    // x / constant -> x * (1/constant)
                    if self.is_constant_node(graph, inputs[1]) {
                        return Some(ReductionType::DivToMul);
                    }
                }
                "Exp" if inputs.len() == 1 => {
                    // exp(log(x)) -> x
                    if self.is_log_operation(graph, inputs[0]) {
                        return Some(ReductionType::ExpLogCancel);
                    }
                }
                "Log" if inputs.len() == 1 => {
                    // log(exp(x)) -> x
                    if self.is_exp_operation(graph, inputs[0]) {
                        return Some(ReductionType::LogExpCancel);
                    }
                }
                _ => {}
            }
        }

        None
    }

    fn is_constant_value(&self, graph: &Graph, node_id: NodeId, value: f32) -> bool {
        if let Some(node) = graph.get_node(node_id) {
            if let crate::graph::NodeType::Constant = &node.op_type {
                if let Some(crate::graph::AttributeValue::Tensor(tensor)) =
                    node.attributes.get("value")
                {
                    if tensor.shape().size() == 1 {
                        if let Some(slice) = tensor.as_slice() {
                            return (slice[0] - value).abs() < 1e-6;
                        }
                    }
                }
            }
        }
        false
    }

    fn is_constant_node(&self, graph: &Graph, node_id: NodeId) -> bool {
        if let Some(node) = graph.get_node(node_id) {
            matches!(node.op_type, crate::graph::NodeType::Constant)
        } else {
            false
        }
    }

    fn is_log_operation(&self, graph: &Graph, node_id: NodeId) -> bool {
        if let Some(node) = graph.get_node(node_id) {
            matches!(
                node.op_type,
                crate::graph::NodeType::Operation(ref op) if op == "Log"
            )
        } else {
            false
        }
    }

    fn is_exp_operation(&self, graph: &Graph, node_id: NodeId) -> bool {
        if let Some(node) = graph.get_node(node_id) {
            matches!(
                node.op_type,
                crate::graph::NodeType::Operation(ref op) if op == "Exp"
            )
        } else {
            false
        }
    }

    fn apply_reduction(
        &self,
        graph: &mut Graph,
        node_id: NodeId,
        reduction: ReductionType,
    ) -> Result<()> {
        match reduction {
            ReductionType::Square => {
                // Replace pow(x, 2) with mul(x, x)
                // Would require creating a new Mul node and rewiring
                // For now, just mark for future implementation
            }
            ReductionType::SquareRoot => {
                // Replace pow(x, 0.5) with sqrt(x)
                // Would require creating a new Sqrt node and rewiring
                // For now, just mark for future implementation
            }
            ReductionType::DivToMul => {
                // Replace div(x, c) with mul(x, 1/c)
                // Would require computing reciprocal constant and creating new node
            }
            ReductionType::ExpLogCancel | ReductionType::LogExpCancel => {
                // Replace exp(log(x)) or log(exp(x)) with x
                let inputs = get_node_inputs(graph, node_id);
                if !inputs.is_empty() {
                    let inner_inputs = get_node_inputs(graph, inputs[0]);
                    if !inner_inputs.is_empty() {
                        graph.redirect_node_outputs(node_id, inner_inputs[0])?;
                        graph.remove_node(node_id)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum ReductionType {
    Square,
    SquareRoot,
    DivToMul,
    ExpLogCancel,
    LogExpCancel,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_folding_pass() {
        let pass = ConstantFoldingPass::new();
        assert_eq!(pass.name(), "ConstantFolding");
        assert_eq!(pass.priority(), 200);

        let graph = Graph::new();
        assert!(!pass.is_applicable(&graph));
    }

    #[test]
    fn test_cse_pass() {
        let pass = CSEPass::new();
        assert_eq!(pass.name(), "CommonSubexpressionElimination");
        assert_eq!(pass.priority(), 150);
    }

    #[test]
    fn test_dead_code_elimination_pass() {
        let pass = DeadCodeEliminationPass::new();
        assert_eq!(pass.name(), "DeadCodeElimination");
        assert_eq!(pass.priority(), 50);
    }

    #[test]
    fn test_algebraic_simplification_pass() {
        let pass = AlgebraicSimplificationPass::new();
        assert_eq!(pass.name(), "AlgebraicSimplification");
        assert_eq!(pass.priority(), 180);
    }

    #[test]
    fn test_operation_scheduling_pass() {
        let pass = OperationSchedulingPass::new();
        assert_eq!(pass.name(), "OperationScheduling");
        assert_eq!(pass.priority(), 120);
        assert!(pass.prefer_memory_locality);
        assert!(pass.enable_parallelization);
    }

    #[test]
    fn test_strength_reduction_pass() {
        let pass = StrengthReductionPass::new();
        assert_eq!(pass.name(), "StrengthReduction");
        assert_eq!(pass.priority(), 140);
    }

    #[test]
    fn test_operation_scheduling_with_config() {
        let pass = OperationSchedulingPass::with_config(false, true);
        assert!(!pass.prefer_memory_locality);
        assert!(pass.enable_parallelization);
    }
}
