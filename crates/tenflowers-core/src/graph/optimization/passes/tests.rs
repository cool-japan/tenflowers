//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use crate::graph::{Graph, NodeId};
use std::collections::{HashMap, HashSet};

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
fn add_op(graph: &mut Graph, name: &str) -> NodeId {
    graph
        .add_node(
            name.to_string(),
            crate::graph::NodeType::Operation("Identity".to_string()),
            crate::Device::Cpu,
            std::collections::HashMap::new(),
        )
        .expect("test: add_node should succeed")
}
fn connect(graph: &mut Graph, from: NodeId, to: NodeId) {
    graph
        .add_edge(
            from,
            to,
            0,
            0,
            crate::dtype::DType::Float32,
            crate::shape::Shape::new(vec![]),
            false,
        )
        .expect("test: add_edge should succeed");
}
/// Add an operation node with an arbitrary op name (unlike `add_op`,
/// which always creates an `"Identity"` node).
fn op_node(graph: &mut Graph, name: &str, op_name: &str) -> NodeId {
    graph
        .add_node(
            name.to_string(),
            crate::graph::NodeType::Operation(op_name.to_string()),
            crate::Device::Cpu,
            HashMap::new(),
        )
        .expect("test: add_node should succeed")
}
/// Add a scalar placeholder node (used to model an opaque input `x`
/// whose value isn't known until execution).
fn placeholder_node(graph: &mut Graph, name: &str) -> NodeId {
    graph
        .add_node(
            name.to_string(),
            crate::graph::NodeType::Placeholder {
                dtype: crate::dtype::DType::Float32,
                shape: crate::shape::Shape::new(vec![]),
            },
            crate::Device::Cpu,
            HashMap::new(),
        )
        .expect("test: add_node should succeed")
}
/// Add a scalar `Constant` node holding `value`.
fn const_node(graph: &mut Graph, name: &str, value: f32) -> NodeId {
    let mut attrs = HashMap::new();
    attrs.insert(
        "value".to_string(),
        crate::graph::AttributeValue::Tensor(crate::tensor::Tensor::from_scalar(value)),
    );
    graph
        .add_node(
            name.to_string(),
            crate::graph::NodeType::Constant,
            crate::Device::Cpu,
            attrs,
        )
        .expect("test: add_node should succeed")
}
/// Read back the scalar value held by a `Constant` node.
fn constant_value(graph: &Graph, node_id: NodeId) -> f32 {
    let node = graph.get_node(node_id).expect("test: node exists");
    match node.attributes.get("value") {
        Some(crate::graph::AttributeValue::Tensor(tensor)) => {
            tensor.as_slice().expect("test: constant tensor has data")[0]
        }
        other => panic!("test: node has no constant tensor value, got {other:?}"),
    }
}
/// `0 -> 1`, `0 -> 2`, `1 -> 3`: depth-first ordering differs from
/// breadth-first, so a locality schedule genuinely reorders it.
fn branching_graph() -> Graph {
    let mut graph = Graph::new();
    let n0 = add_op(&mut graph, "op0");
    let n1 = add_op(&mut graph, "op1");
    let n2 = add_op(&mut graph, "op2");
    let n3 = add_op(&mut graph, "op3");
    connect(&mut graph, n0, n1);
    connect(&mut graph, n0, n2);
    connect(&mut graph, n1, n3);
    graph
}
/// `0 -> {1,2} -> 3`: a diamond whose every topological order shares the
/// same first and last node.
fn diamond_graph() -> Graph {
    let mut graph = Graph::new();
    let n0 = add_op(&mut graph, "op0");
    let n1 = add_op(&mut graph, "op1");
    let n2 = add_op(&mut graph, "op2");
    let n3 = add_op(&mut graph, "op3");
    connect(&mut graph, n0, n1);
    connect(&mut graph, n0, n2);
    connect(&mut graph, n1, n3);
    connect(&mut graph, n2, n3);
    graph
}
fn assert_valid_topological_order(graph: &Graph, order: &[NodeId]) {
    let position: HashMap<NodeId, usize> = order
        .iter()
        .enumerate()
        .map(|(idx, &id)| (id, idx))
        .collect();
    for edge in graph.edges() {
        if edge.is_control {
            continue;
        }
        let from_pos = position[&edge.from_node];
        let to_pos = position[&edge.to_node];
        assert!(
            from_pos < to_pos,
            "edge {} -> {} violates topological order",
            edge.from_node,
            edge.to_node
        );
    }
}
#[test]
fn test_scheduling_no_change_on_linear_chain() {
    let pass = OperationSchedulingPass::new();
    let mut graph = Graph::new();
    let n0 = add_op(&mut graph, "op0");
    let n1 = add_op(&mut graph, "op1");
    let n2 = add_op(&mut graph, "op2");
    let n3 = add_op(&mut graph, "op3");
    connect(&mut graph, n0, n1);
    connect(&mut graph, n1, n2);
    connect(&mut graph, n2, n3);
    let changed = pass.apply(&mut graph).expect("test: apply should succeed");
    assert!(
        !changed,
        "a linear chain cannot be reordered; the pass must not claim a change"
    );
}
#[test]
fn test_scheduling_disabled_reports_no_change() {
    let pass = OperationSchedulingPass::with_config(false, false);
    let mut graph = diamond_graph();
    let changed = pass.apply(&mut graph).expect("test: apply should succeed");
    assert!(
        !changed,
        "no heuristic enabled; the pass must report no change"
    );
}
#[test]
fn test_scheduling_locality_reorders_branching_graph() {
    let pass = OperationSchedulingPass::new();
    let mut graph = branching_graph();
    let changed = pass.apply(&mut graph).expect("test: apply should succeed");
    assert!(
        changed,
        "locality schedule should reorder a branching graph"
    );
    let order = graph
        .topological_order
        .clone()
        .expect("test: a schedule should be cached after a change");
    assert_eq!(order.len(), graph.node_count());
    assert_valid_topological_order(&graph, &order);
}
#[test]
fn test_scheduling_is_idempotent() {
    let pass = OperationSchedulingPass::new();
    let mut graph = branching_graph();
    let first = pass
        .apply(&mut graph)
        .expect("test: first apply should succeed");
    assert!(
        first,
        "first application should reorder the branching graph"
    );
    let second = pass
        .apply(&mut graph)
        .expect("test: second apply should succeed");
    assert!(!second, "second application must not fabricate a change");
}
#[test]
fn test_constant_folding_nested_subexpressions_need_multiple_iterations() {
    let pass = ConstantFoldingPass::new();
    let mut graph = Graph::new();
    let one = const_node(&mut graph, "one", 1.0);
    let two = const_node(&mut graph, "two", 2.0);
    let three = const_node(&mut graph, "three", 3.0);
    let four = const_node(&mut graph, "four", 4.0);
    let add1 = op_node(&mut graph, "add1", "Add");
    connect(&mut graph, one, add1);
    connect(&mut graph, two, add1);
    let add2 = op_node(&mut graph, "add2", "Add");
    connect(&mut graph, three, add2);
    connect(&mut graph, four, add2);
    let mul = op_node(&mut graph, "mul", "Mul");
    connect(&mut graph, add1, mul);
    connect(&mut graph, add2, mul);
    let first = pass
        .apply(&mut graph)
        .expect(
            "test: folding the inner sums must not error even though the outer product isn't foldable yet",
        );
    assert!(first, "the two inner sums should fold on the first pass");
    assert!(
        matches!(
            graph.get_node(mul).expect("test: mul node exists").op_type,
            crate::graph::NodeType::Operation(_)
        ),
        "the outer product must not be folded until its inputs are literal constants"
    );
    assert!((constant_value(&graph, add1) - 3.0).abs() < 1e-6);
    assert!((constant_value(&graph, add2) - 7.0).abs() < 1e-6);
    let second = pass
        .apply(&mut graph)
        .expect("test: folding the outer product should not error");
    assert!(second, "the outer product should fold on the second pass");
    assert!(matches!(
        graph.get_node(mul).expect("test: mul node exists").op_type,
        crate::graph::NodeType::Constant
    ));
    assert!((constant_value(&graph, mul) - 21.0).abs() < 1e-6);
}
#[test]
fn test_dead_code_elimination_keeps_marked_output_and_its_ancestors() {
    let mut graph = Graph::new();
    let a = add_op(&mut graph, "a");
    let b = add_op(&mut graph, "b");
    let sum = add_op(&mut graph, "sum");
    connect(&mut graph, a, sum);
    connect(&mut graph, b, sum);
    let mut outputs = HashSet::new();
    outputs.insert(sum);
    let pass = DeadCodeEliminationPass::with_outputs(outputs);
    pass.apply(&mut graph).expect("test: apply should succeed");
    assert!(graph.get_node(a).is_some(), "ancestor 'a' must survive");
    assert!(graph.get_node(b).is_some(), "ancestor 'b' must survive");
    assert!(
        graph.get_node(sum).is_some(),
        "the marked output must survive"
    );
}
#[test]
fn test_dead_code_elimination_set_outputs_takes_effect() {
    let mut graph = Graph::new();
    let lone = add_op(&mut graph, "lone");
    let pass = DeadCodeEliminationPass::new();
    let mut outputs = HashSet::new();
    outputs.insert(lone);
    pass.set_outputs(&outputs);
    pass.apply(&mut graph).expect("test: apply should succeed");
    assert!(graph.get_node(lone).is_some());
}
#[test]
fn test_algebraic_simplification_double_add_rewrites_to_scalar_multiply() {
    let pass = AlgebraicSimplificationPass::new();
    let mut graph = Graph::new();
    let x = placeholder_node(&mut graph, "x");
    let add = op_node(&mut graph, "double", "Add");
    connect(&mut graph, x, add);
    connect(&mut graph, x, add);
    let changed = pass.apply(&mut graph).expect("test: apply should succeed");
    assert!(changed, "x + x should be rewritten");
    let node = graph.get_node(add).expect("test: node id is preserved");
    match &node.op_type {
        crate::graph::NodeType::Operation(op) => assert_eq!(op, "Mul"),
        other => panic!("expected Mul, got {other:?}"),
    }
    let inputs = get_node_inputs(&graph, add);
    assert_eq!(inputs.len(), 2);
    assert!(inputs.contains(&x));
    let const_id = inputs
        .into_iter()
        .find(|&id| id != x)
        .expect("test: a constant input must exist");
    assert!((constant_value(&graph, const_id) - 2.0).abs() < 1e-6);
    for sample_value in [-1.5f32, 0.0, 3.25] {
        let sample = crate::tensor::Tensor::<f32>::from_scalar(sample_value);
        let original = crate::ops::binary::add(&sample, &sample).expect("test: add should succeed");
        let rewritten = crate::ops::binary::mul(
            &sample,
            &crate::tensor::Tensor::<f32>::from_scalar(constant_value(&graph, const_id)),
        )
        .expect("test: mul should succeed");
        assert_eq!(original.as_slice(), rewritten.as_slice());
    }
}
#[test]
fn test_algebraic_simplification_protects_marked_identity_node() {
    let pass = AlgebraicSimplificationPass::new();
    let mut graph = Graph::new();
    let x = placeholder_node(&mut graph, "x");
    let zero = const_node(&mut graph, "zero", 0.0);
    let add = op_node(&mut graph, "x_plus_zero", "Add");
    connect(&mut graph, x, add);
    connect(&mut graph, zero, add);
    let mut outputs = HashSet::new();
    outputs.insert(add);
    pass.set_outputs(&outputs);
    let changed = pass.apply(&mut graph).expect("test: apply should succeed");
    assert!(
        !changed,
        "a protected identity node must not be simplified away"
    );
    assert!(
        graph.get_node(add).is_some(),
        "the protected node must keep its identity"
    );
}
#[test]
fn test_strength_reduction_square_rewrites_to_self_multiply() {
    let pass = StrengthReductionPass::new();
    let mut graph = Graph::new();
    let x = placeholder_node(&mut graph, "x");
    let two = const_node(&mut graph, "two", 2.0);
    let pow = op_node(&mut graph, "pow", "Pow");
    connect(&mut graph, x, pow);
    connect(&mut graph, two, pow);
    let changed = pass.apply(&mut graph).expect("test: apply should succeed");
    assert!(changed, "pow(x, 2) should be rewritten");
    let node = graph.get_node(pow).expect("test: node id is preserved");
    match &node.op_type {
        crate::graph::NodeType::Operation(op) => assert_eq!(op, "Mul"),
        other => panic!("expected Mul, got {other:?}"),
    }
    assert_eq!(get_node_inputs(&graph, pow), vec![x, x]);
    for sample_value in [-3.5f32, 0.0, 2.25] {
        let sample = crate::tensor::Tensor::<f32>::from_scalar(sample_value);
        let via_pow = crate::ops::pow(&sample, &crate::tensor::Tensor::<f32>::from_scalar(2.0f32))
            .expect("test: pow should succeed");
        let via_mul = crate::ops::binary::mul(&sample, &sample).expect("test: mul should succeed");
        assert_eq!(
            via_pow.as_slice(),
            via_mul.as_slice(),
            "pow(x,2) must equal x*x for x={sample_value}"
        );
    }
}
#[test]
fn test_strength_reduction_div_by_power_of_two_rewrites_to_multiply() {
    let pass = StrengthReductionPass::new();
    let mut graph = Graph::new();
    let x = placeholder_node(&mut graph, "x");
    let four = const_node(&mut graph, "four", 4.0);
    let div = op_node(&mut graph, "div", "Div");
    connect(&mut graph, x, div);
    connect(&mut graph, four, div);
    let changed = pass.apply(&mut graph).expect("test: apply should succeed");
    assert!(
        changed,
        "x / 4.0 should be rewritten (4.0 is a power of two)"
    );
    let node = graph.get_node(div).expect("test: node id is preserved");
    match &node.op_type {
        crate::graph::NodeType::Operation(op) => assert_eq!(op, "Mul"),
        other => panic!("expected Mul, got {other:?}"),
    }
    let inputs = get_node_inputs(&graph, div);
    assert_eq!(inputs.len(), 2);
    assert_eq!(inputs[0], x);
    let const_id = inputs[1];
    assert!((constant_value(&graph, const_id) - 0.25).abs() < 1e-9);
    for sample_value in [-7.0f32, 0.0, 3.0, 1.0e10] {
        let sample = crate::tensor::Tensor::<f32>::from_scalar(sample_value);
        let via_div =
            crate::ops::binary::div(&sample, &crate::tensor::Tensor::<f32>::from_scalar(4.0f32))
                .expect("test: div should succeed");
        let via_mul =
            crate::ops::binary::mul(&sample, &crate::tensor::Tensor::<f32>::from_scalar(0.25f32))
                .expect("test: mul should succeed");
        assert_eq!(via_div.as_slice(), via_mul.as_slice());
    }
}
#[test]
fn test_strength_reduction_div_by_non_power_of_two_is_left_unreduced() {
    let pass = StrengthReductionPass::new();
    let mut graph = Graph::new();
    let x = placeholder_node(&mut graph, "x");
    let three = const_node(&mut graph, "three", 3.0);
    let div = op_node(&mut graph, "div", "Div");
    connect(&mut graph, x, div);
    connect(&mut graph, three, div);
    let changed = pass.apply(&mut graph).expect("test: apply should succeed");
    assert!(
        !changed,
        "division by a non-power-of-two constant must not be rewritten"
    );
    let node = graph.get_node(div).expect("test: node still exists");
    match &node.op_type {
        crate::graph::NodeType::Operation(op) => assert_eq!(op, "Div"),
        other => panic!("expected the node to remain a Div, got {other:?}"),
    }
}
#[test]
fn test_cse_protected_duplicate_is_not_merged() {
    let mut graph = Graph::new();
    let a = placeholder_node(&mut graph, "a");
    let b = placeholder_node(&mut graph, "b");
    let dup1 = add_op(&mut graph, "dup1");
    connect(&mut graph, a, dup1);
    connect(&mut graph, b, dup1);
    let dup2 = add_op(&mut graph, "dup2");
    connect(&mut graph, a, dup2);
    connect(&mut graph, b, dup2);
    let before_count = graph.node_count();
    let pass = CSEPass::new();
    let mut outputs = HashSet::new();
    outputs.insert(dup1);
    outputs.insert(dup2);
    pass.set_outputs(&outputs);
    let changed = pass.apply(&mut graph).expect("test: apply should succeed");
    assert!(
        !changed,
        "both duplicates are protected outputs; CSE must not merge either away"
    );
    assert_eq!(graph.node_count(), before_count);
    assert!(graph.get_node(dup1).is_some());
    assert!(graph.get_node(dup2).is_some());
}
#[test]
fn test_cse_merges_unprotected_duplicates() {
    let mut graph = Graph::new();
    let a = placeholder_node(&mut graph, "a");
    let b = placeholder_node(&mut graph, "b");
    let dup1 = add_op(&mut graph, "dup1");
    connect(&mut graph, a, dup1);
    connect(&mut graph, b, dup1);
    let dup2 = add_op(&mut graph, "dup2");
    connect(&mut graph, a, dup2);
    connect(&mut graph, b, dup2);
    let before_count = graph.node_count();
    let pass = CSEPass::new();
    let changed = pass.apply(&mut graph).expect("test: apply should succeed");
    assert!(changed, "two structurally identical Add nodes should merge");
    assert_eq!(
        graph.node_count(),
        before_count - 1,
        "exactly one merge should happen"
    );
    assert_ne!(
        graph.get_node(dup1).is_some(),
        graph.get_node(dup2).is_some(),
        "exactly one of the two duplicates should remain"
    );
    assert!(
        graph.get_node(a).is_some() && graph.get_node(b).is_some(),
        "distinct placeholders must never be merged with each other"
    );
}
#[test]
fn test_cse_does_not_merge_distinct_valued_constants() {
    let mut graph = Graph::new();
    let two = const_node(&mut graph, "two", 2.0);
    let five = const_node(&mut graph, "five", 5.0);
    let pass = CSEPass::new();
    let changed = pass.apply(&mut graph).expect("test: apply should succeed");
    assert!(
        !changed,
        "constants with different values must never be treated as duplicates"
    );
    assert!(graph.get_node(two).is_some());
    assert!(graph.get_node(five).is_some());
}
#[test]
fn test_cse_merges_identically_valued_constants() {
    let mut graph = Graph::new();
    let two_a = const_node(&mut graph, "two_a", 2.0);
    let two_b = const_node(&mut graph, "two_b", 2.0);
    let pass = CSEPass::new();
    let changed = pass.apply(&mut graph).expect("test: apply should succeed");
    assert!(changed, "identically valued constants should merge");
    assert_eq!(graph.node_count(), 1);
    assert_ne!(
        graph.get_node(two_a).is_some(),
        graph.get_node(two_b).is_some()
    );
}
