//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::graph::{Graph, NodeId};
use crate::{Result, TensorError};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use super::pass_support::{get_node_inputs, OptimizationPass};

/// Operation scheduling pass
/// Reorders operations for better performance and parallelism
pub struct OperationSchedulingPass {
    pub(super) prefer_memory_locality: bool,
    pub(super) enable_parallelization: bool,
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
    pub(super) fn compute_dependencies(&self, graph: &Graph) -> HashMap<NodeId, Vec<NodeId>> {
        let mut deps = HashMap::new();
        for node in graph.nodes() {
            let inputs = get_node_inputs(graph, node.id);
            deps.insert(node.id, inputs);
        }
        deps
    }
    pub(super) fn find_reorderable_operations(
        &self,
        graph: &Graph,
        dependencies: &HashMap<NodeId, Vec<NodeId>>,
    ) -> Vec<(NodeId, NodeId)> {
        let mut reorderable = Vec::new();
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
    pub(super) fn can_reorder(
        &self,
        node_a: NodeId,
        node_b: NodeId,
        dependencies: &HashMap<NodeId, Vec<NodeId>>,
    ) -> bool {
        let deps_a = dependencies
            .get(&node_a)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let deps_b = dependencies
            .get(&node_b)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        !deps_a.contains(&node_b) && !deps_b.contains(&node_a)
    }
    /// Select the scheduling policy implied by this pass's configuration.
    pub(super) fn schedule_policy(&self) -> SchedulePolicy {
        if self.prefer_memory_locality {
            SchedulePolicy::Locality
        } else if self.enable_parallelization {
            SchedulePolicy::Parallelism
        } else {
            SchedulePolicy::Lexical
        }
    }
    /// Reorder operations into a dependency-respecting execution schedule.
    ///
    /// Returns `Ok(true)` only when the resulting schedule actually differs from
    /// the order the graph would otherwise execute in. When the configured
    /// heuristic produces the same order as the canonical baseline (for example a
    /// straight-line graph, or when no heuristic is enabled), it reports
    /// `Ok(false)` and makes no false claim of optimization. The schedule is
    /// installed into the graph's cached topological order, which the session
    /// executor consumes as the execution order.
    pub(super) fn apply_scheduling_heuristics(&self, graph: &mut Graph) -> Result<bool> {
        let optimized = self.compute_schedule(graph, self.schedule_policy())?;
        let baseline = self.compute_schedule(graph, SchedulePolicy::Lexical)?;
        let current = graph
            .topological_order
            .clone()
            .unwrap_or_else(|| baseline.clone());
        if optimized == current {
            if graph.topological_order.is_none() {
                graph.topological_order = Some(optimized);
            }
            Ok(false)
        } else {
            graph.topological_order = Some(optimized);
            Ok(true)
        }
    }
    /// Compute a valid topological execution order using the given policy.
    ///
    /// Every policy is a variant of Kahn's algorithm, so the result always
    /// respects each data dependency; policies differ only in how they break ties
    /// among operations that are simultaneously ready to run. Control edges do not
    /// constrain ordering, matching [`Graph::compute_topological_order`].
    pub(super) fn compute_schedule(
        &self,
        graph: &Graph,
        policy: SchedulePolicy,
    ) -> Result<Vec<NodeId>> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for node in graph.nodes() {
            in_degree.insert(node.id, 0);
            adjacency.insert(node.id, Vec::new());
        }
        for edge in graph.edges() {
            if edge.is_control {
                continue;
            }
            if let Some(successors) = adjacency.get_mut(&edge.from_node) {
                successors.push(edge.to_node);
            }
            if let Some(degree) = in_degree.get_mut(&edge.to_node) {
                *degree += 1;
            }
        }
        for successors in adjacency.values_mut() {
            successors.sort_unstable();
        }
        let out_degree: HashMap<NodeId, usize> = adjacency
            .iter()
            .map(|(&id, successors)| (id, successors.len()))
            .collect();
        let node_count = graph.node_count();
        let mut result: Vec<NodeId> = Vec::with_capacity(node_count);
        match policy {
            SchedulePolicy::Lexical => {
                let mut ready: BinaryHeap<Reverse<NodeId>> = in_degree
                    .iter()
                    .filter(|&(_, &degree)| degree == 0)
                    .map(|(&id, _)| Reverse(id))
                    .collect();
                while let Some(Reverse(node_id)) = ready.pop() {
                    result.push(node_id);
                    for successor in
                        self.release_ready_successors(node_id, &adjacency, &mut in_degree)
                    {
                        ready.push(Reverse(successor));
                    }
                }
            }
            SchedulePolicy::Parallelism => {
                let mut ready: BinaryHeap<(usize, Reverse<NodeId>)> = in_degree
                    .iter()
                    .filter(|&(_, &degree)| degree == 0)
                    .map(|(&id, _)| (*out_degree.get(&id).unwrap_or(&0), Reverse(id)))
                    .collect();
                while let Some((_, Reverse(node_id))) = ready.pop() {
                    result.push(node_id);
                    for successor in
                        self.release_ready_successors(node_id, &adjacency, &mut in_degree)
                    {
                        let fan_out = *out_degree.get(&successor).unwrap_or(&0);
                        ready.push((fan_out, Reverse(successor)));
                    }
                }
            }
            SchedulePolicy::Locality => {
                let mut ready: Vec<NodeId> = in_degree
                    .iter()
                    .filter(|&(_, &degree)| degree == 0)
                    .map(|(&id, _)| id)
                    .collect();
                ready.sort_unstable_by_key(|&id| Reverse(id));
                while let Some(node_id) = ready.pop() {
                    result.push(node_id);
                    let mut newly =
                        self.release_ready_successors(node_id, &adjacency, &mut in_degree);
                    newly.sort_unstable_by_key(|&id| Reverse(id));
                    ready.extend(newly);
                }
            }
        }
        if result.len() != node_count {
            return Err(TensorError::invalid_argument(
                "Operation scheduling failed: the graph contains a cycle".to_string(),
            ));
        }
        Ok(result)
    }
    /// Decrement the in-degree of `node_id`'s successors, returning those that
    /// have just become ready (in-degree zero).
    pub(super) fn release_ready_successors(
        &self,
        node_id: NodeId,
        adjacency: &HashMap<NodeId, Vec<NodeId>>,
        in_degree: &mut HashMap<NodeId, usize>,
    ) -> Vec<NodeId> {
        let mut newly_ready = Vec::new();
        if let Some(successors) = adjacency.get(&node_id) {
            for &successor in successors {
                if let Some(degree) = in_degree.get_mut(&successor) {
                    *degree -= 1;
                    if *degree == 0 {
                        newly_ready.push(successor);
                    }
                }
            }
        }
        newly_ready
    }
}
impl OptimizationPass for OperationSchedulingPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let dependencies = self.compute_dependencies(graph);
        let reorderable_ops = self.find_reorderable_operations(graph, &dependencies);
        if reorderable_ops.is_empty() {
            return Ok(false);
        }
        self.apply_scheduling_heuristics(graph)
    }
    fn name(&self) -> &str {
        "OperationScheduling"
    }
    fn is_applicable(&self, graph: &Graph) -> bool {
        graph.node_count() > 2
    }
    fn priority(&self) -> u32 {
        120
    }
}
impl Default for OperationSchedulingPass {
    fn default() -> Self {
        Self::new()
    }
}
/// Tie-breaking strategy for the operation scheduler.
///
/// All variants yield a valid topological order; they differ only in which
/// ready operation is chosen next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SchedulePolicy {
    /// Smallest `NodeId` first — the deterministic canonical order.
    Lexical,
    /// Highest fan-out first — expose parallelism earlier.
    Parallelism,
    /// Depth-first — keep consumers adjacent to producers for locality.
    Locality,
}
