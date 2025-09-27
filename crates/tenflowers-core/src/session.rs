use crate::{
    dtype::DType,
    error::TensorError,
    graph::{AttributeValue, Graph, NodeId, NodeType},
    ops::registry::OpRegistry,
    tensor::Tensor,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Session configuration options
#[derive(Clone, Debug)]
pub struct SessionConfig {
    /// Allow soft device placement (fallback to CPU when GPU unavailable)
    pub allow_soft_placement: bool,
    /// Log device placement decisions
    pub log_device_placement: bool,
    /// GPU memory growth configuration
    pub gpu_memory_growth: bool,
    /// GPU memory limit (bytes)
    pub gpu_memory_limit: Option<usize>,
    /// Number of inter-op threads
    pub inter_op_parallelism_threads: usize,
    /// Number of intra-op threads
    pub intra_op_parallelism_threads: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            allow_soft_placement: true,
            log_device_placement: false,
            gpu_memory_growth: true,
            gpu_memory_limit: None,
            inter_op_parallelism_threads: 0, // Use system default
            intra_op_parallelism_threads: 0, // Use system default
        }
    }
}

/// Feed dictionary for providing input values to placeholders
pub type FeedDict = HashMap<String, Tensor<f32>>;

/// Fetch specification for requesting output values
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FetchSpec {
    /// Fetch by node name
    Name(String),
    /// Fetch by node ID
    NodeId(NodeId),
    /// Fetch by node name and output index
    NamedOutput(String, usize),
    /// Fetch by node ID and output index
    IndexedOutput(NodeId, usize),
}

/// Session for executing computation graphs
pub trait Session {
    /// Run the graph with the given feeds and fetches
    fn run(
        &mut self,
        fetches: &[FetchSpec],
        feed_dict: &FeedDict,
    ) -> Result<Vec<Tensor<f32>>, TensorError>;

    /// Partial run - allows incrementally feeding inputs and fetching outputs
    fn partial_run_setup(
        &mut self,
        feeds: &[String],
        fetches: &[FetchSpec],
        targets: &[String],
    ) -> Result<String, TensorError>; // Returns run handle

    /// Continue a partial run
    fn partial_run(
        &mut self,
        handle: &str,
        feed_dict: &FeedDict,
        fetches: &[FetchSpec],
    ) -> Result<Vec<Tensor<f32>>, TensorError>;

    /// Close the session and release resources
    fn close(&mut self) -> Result<(), TensorError>;
}

/// Variable store for managing session variables
pub type VariableStore = HashMap<String, Tensor<f32>>;

/// Default session implementation
#[allow(dead_code)]
pub struct DefaultSession {
    graph: Arc<RwLock<Graph>>,
    config: SessionConfig,
    op_registry: Arc<OpRegistry>,
    closed: bool,
    // Variable storage
    variables: VariableStore,
    // Cached execution plan
    execution_cache: HashMap<Vec<FetchSpec>, ExecutionPlan>,
    // Partial run state
    partial_runs: HashMap<String, PartialRunState>,
    next_partial_run_id: u64,
}

/// Execution plan for a set of fetches
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct ExecutionPlan {
    /// Nodes to execute in topological order
    execution_order: Vec<NodeId>,
    /// Input mapping (placeholder name -> node id)
    input_mapping: HashMap<String, NodeId>,
    /// Output mapping (fetch spec -> (node id, output index))
    output_mapping: HashMap<FetchSpec, (NodeId, usize)>,
}

/// State for partial runs
#[derive(Debug)]
#[allow(dead_code)]
struct PartialRunState {
    feeds: Vec<String>,
    fetches: Vec<FetchSpec>,
    targets: Vec<String>,
    plan: ExecutionPlan,
    // Intermediate values stored between partial runs
    intermediate_values: HashMap<NodeId, Vec<Tensor<f32>>>,
}

impl DefaultSession {
    /// Create a new session with the given graph and configuration
    pub fn new(
        graph: Arc<RwLock<Graph>>,
        config: SessionConfig,
        op_registry: Arc<OpRegistry>,
    ) -> Self {
        Self {
            graph,
            config,
            op_registry,
            closed: false,
            variables: HashMap::new(),
            execution_cache: HashMap::new(),
            partial_runs: HashMap::new(),
            next_partial_run_id: 0,
        }
    }

    /// Create execution plan for the given fetches
    fn create_execution_plan(&self, fetches: &[FetchSpec]) -> Result<ExecutionPlan, TensorError> {
        let graph = self.graph.read().unwrap();

        // Find all nodes that need to be executed
        let mut required_nodes = std::collections::HashSet::new();
        let mut output_mapping = HashMap::new();

        // Process each fetch specification
        for fetch in fetches {
            let (node_id, output_idx) = match fetch {
                FetchSpec::Name(name) => {
                    let node = graph.get_node_by_name(name).ok_or_else(|| {
                        TensorError::invalid_argument(format!("Node '{name}' not found"))
                    })?;
                    (node.id, 0)
                }
                FetchSpec::NodeId(id) => (*id, 0),
                FetchSpec::NamedOutput(name, idx) => {
                    let node = graph.get_node_by_name(name).ok_or_else(|| {
                        TensorError::invalid_argument(format!("Node '{name}' not found"))
                    })?;
                    (node.id, *idx)
                }
                FetchSpec::IndexedOutput(id, idx) => (*id, *idx),
            };

            // Verify node exists
            if graph.get_node(node_id).is_none() {
                return Err(TensorError::invalid_argument(format!(
                    "Node {node_id} not found"
                )));
            }

            required_nodes.insert(node_id);
            output_mapping.insert(fetch.clone(), (node_id, output_idx));
        }

        // Find all dependencies using DFS
        let mut stack = required_nodes.iter().cloned().collect::<Vec<_>>();
        while let Some(node_id) = stack.pop() {
            if let Some(node) = graph.get_node(node_id) {
                // Add all input nodes
                for &edge_id in &node.inputs {
                    if let Some(edge) = graph.get_edge(edge_id) {
                        if required_nodes.insert(edge.from_node) {
                            stack.push(edge.from_node);
                        }
                    }
                }
            }
        }

        // We need to access compute_topological_order on the graph
        // Since we can't clone RwLockReadGuard, we'll call it on the original graph
        let full_topo_order = {
            drop(graph); // Release the read lock
            let mut graph_write = self.graph.write().unwrap();
            graph_write.compute_topological_order()?.to_vec()
        };
        let execution_order: Vec<NodeId> = full_topo_order
            .iter()
            .filter(|&&node_id| required_nodes.contains(&node_id))
            .cloned()
            .collect();

        // Create input mapping (placeholders)
        let graph = self.graph.read().unwrap();
        let mut input_mapping = HashMap::new();
        for node in graph.nodes() {
            if let NodeType::Placeholder { .. } = node.op_type {
                input_mapping.insert(node.name.clone(), node.id);
            }
        }

        Ok(ExecutionPlan {
            execution_order,
            input_mapping,
            output_mapping,
        })
    }

    /// Execute a single node
    fn execute_node(
        &mut self,
        node_id: NodeId,
        node_values: &mut HashMap<NodeId, Vec<Tensor<f32>>>,
        feed_dict: &FeedDict,
    ) -> Result<(), TensorError> {
        let graph = self.graph.read().unwrap();
        let node = graph
            .get_node(node_id)
            .ok_or_else(|| TensorError::invalid_argument(format!("Node {node_id} not found")))?;

        match &node.op_type {
            NodeType::Placeholder { .. } => {
                // Look up value in feed_dict
                if let Some(value) = feed_dict.get(&node.name) {
                    node_values.insert(node_id, vec![value.clone()]);
                } else {
                    return Err(TensorError::invalid_argument(format!(
                        "No value provided for placeholder '{}'",
                        node.name
                    )));
                }
            }
            NodeType::Constant => {
                // Get constant value from attributes
                if let Some(AttributeValue::Tensor(tensor)) = node.attributes.get("value") {
                    node_values.insert(node_id, vec![tensor.clone()]);
                } else {
                    return Err(TensorError::invalid_argument(format!(
                        "Constant node '{}' has no value attribute",
                        node.name
                    )));
                }
            }
            NodeType::Variable { shape, dtype, .. } => {
                // Check if variable is already initialized
                if let Some(var_tensor) = self.variables.get(&node.name) {
                    // Use existing variable value
                    node_values.insert(node_id, vec![var_tensor.clone()]);
                } else {
                    // Initialize variable with zeros or from initializer attribute
                    let tensor = if let Some(AttributeValue::Tensor(init_tensor)) =
                        node.attributes.get("initializer")
                    {
                        init_tensor.clone()
                    } else {
                        // Default initialization with zeros
                        match dtype {
                            DType::Float32 => Tensor::<f32>::zeros(shape.dims()),
                            _ => {
                                return Err(TensorError::unsupported_operation_simple(format!(
                                    "Variable dtype {dtype:?} not supported"
                                )))
                            }
                        }
                    };

                    // Store the variable for future use
                    self.variables.insert(node.name.clone(), tensor.clone());
                    node_values.insert(node_id, vec![tensor]);
                }
            }
            NodeType::Operation(op_name) => {
                // Gather input tensors
                let mut input_tensors = Vec::new();
                for &edge_id in &node.inputs {
                    if let Some(edge) = graph.get_edge(edge_id) {
                        if let Some(from_outputs) = node_values.get(&edge.from_node) {
                            if edge.from_output < from_outputs.len() {
                                input_tensors.push(from_outputs[edge.from_output].clone());
                            } else {
                                return Err(TensorError::invalid_argument(format!(
                                    "Invalid output index {} for node {}",
                                    edge.from_output, edge.from_node
                                )));
                            }
                        } else {
                            return Err(TensorError::invalid_argument(format!(
                                "Input node {} has not been computed",
                                edge.from_node
                            )));
                        }
                    }
                }

                // Execute the operation
                let outputs = self.execute_operation(op_name, &input_tensors, &node.attributes)?;
                node_values.insert(node_id, outputs);
            }
        }

        Ok(())
    }

    /// Execute an operation with given inputs
    fn execute_operation(
        &self,
        op_name: &str,
        inputs: &[Tensor<f32>],
        _attributes: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        // This is a simplified implementation
        // In practice, we'd use the op registry to dispatch to the correct kernel
        match op_name {
            "Add" => {
                if inputs.len() != 2 {
                    return Err(TensorError::invalid_argument(
                        "Add operation requires 2 inputs".to_string(),
                    ));
                }
                Ok(vec![inputs[0].add(&inputs[1])?])
            }
            "Mul" => {
                if inputs.len() != 2 {
                    return Err(TensorError::invalid_argument(
                        "Mul operation requires 2 inputs".to_string(),
                    ));
                }
                Ok(vec![inputs[0].mul(&inputs[1])?])
            }
            "MatMul" => {
                if inputs.len() != 2 {
                    return Err(TensorError::invalid_argument(
                        "MatMul operation requires 2 inputs".to_string(),
                    ));
                }
                Ok(vec![inputs[0].matmul(&inputs[1])?])
            }
            "Identity" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Identity operation requires 1 input".to_string(),
                    ));
                }
                Ok(vec![inputs[0].clone()])
            }
            "Sub" => {
                if inputs.len() != 2 {
                    return Err(TensorError::invalid_argument(
                        "Sub operation requires 2 inputs".to_string(),
                    ));
                }
                Ok(vec![inputs[0].sub(&inputs[1])?])
            }
            "Div" => {
                if inputs.len() != 2 {
                    return Err(TensorError::invalid_argument(
                        "Div operation requires 2 inputs".to_string(),
                    ));
                }
                Ok(vec![inputs[0].div(&inputs[1])?])
            }
            "Pow" => {
                if inputs.len() != 2 {
                    return Err(TensorError::invalid_argument(
                        "Pow operation requires 2 inputs".to_string(),
                    ));
                }
                Ok(vec![crate::ops::pow(&inputs[0], &inputs[1])?])
            }
            "Exp" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Exp operation requires 1 input".to_string(),
                    ));
                }
                Ok(vec![crate::ops::exp(&inputs[0])?])
            }
            "Log" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Log operation requires 1 input".to_string(),
                    ));
                }
                Ok(vec![crate::ops::log(&inputs[0])?])
            }
            "Sin" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Sin operation requires 1 input".to_string(),
                    ));
                }
                Ok(vec![crate::ops::sin(&inputs[0])?])
            }
            "Cos" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Cos operation requires 1 input".to_string(),
                    ));
                }
                Ok(vec![crate::ops::cos(&inputs[0])?])
            }
            "Tanh" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Tanh operation requires 1 input".to_string(),
                    ));
                }
                Ok(vec![crate::ops::tanh(&inputs[0])?])
            }
            "Relu" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Relu operation requires 1 input".to_string(),
                    ));
                }
                Ok(vec![crate::ops::relu(&inputs[0])?])
            }
            "Sigmoid" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Sigmoid operation requires 1 input".to_string(),
                    ));
                }
                Ok(vec![crate::ops::sigmoid(&inputs[0])?])
            }
            "Softmax" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Softmax operation requires 1 input".to_string(),
                    ));
                }
                // Default to last axis (-1)
                Ok(vec![crate::ops::softmax(&inputs[0], Some(-1))?])
            }
            "Sum" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Sum operation requires 1 input".to_string(),
                    ));
                }
                Ok(vec![crate::ops::sum(&inputs[0], None, false)?])
            }
            "Mean" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Mean operation requires 1 input".to_string(),
                    ));
                }
                Ok(vec![crate::ops::mean(&inputs[0], None, false)?])
            }
            "Reshape" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Reshape operation requires 1 input (shape as attribute)".to_string(),
                    ));
                }
                // For session execution, we'll use a simple flattening reshape
                let total_elements = inputs[0].shape().dims().iter().product::<usize>();
                Ok(vec![inputs[0].reshape(&[total_elements])?])
            }
            "Transpose" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Transpose operation requires 1 input".to_string(),
                    ));
                }
                Ok(vec![crate::ops::transpose(&inputs[0])?])
            }
            "Conv2D" => {
                if inputs.len() < 2 {
                    return Err(TensorError::invalid_argument(
                        "Conv2D operation requires at least 2 inputs".to_string(),
                    ));
                }
                // Use default parameters for stride, padding
                Ok(vec![crate::ops::conv2d(
                    &inputs[0],
                    &inputs[1],
                    None,
                    (1, 1),
                    "VALID",
                )?])
            }
            "MaxPool2D" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "MaxPool2D operation requires 1 input".to_string(),
                    ));
                }
                // Use default 2x2 kernel with stride 2
                Ok(vec![crate::ops::max_pool2d(
                    &inputs[0],
                    (2, 2),
                    (2, 2),
                    "VALID",
                )?])
            }
            "AvgPool2D" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "AvgPool2D operation requires 1 input".to_string(),
                    ));
                }
                // Use default 2x2 kernel with stride 2
                Ok(vec![crate::ops::avg_pool2d(
                    &inputs[0],
                    (2, 2),
                    (2, 2),
                    "VALID",
                )?])
            }
            "Max" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Max operation requires 1 input".to_string(),
                    ));
                }
                Ok(vec![crate::ops::max(&inputs[0], None, false)?])
            }
            "Min" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Min operation requires 1 input".to_string(),
                    ));
                }
                Ok(vec![crate::ops::min(&inputs[0], None, false)?])
            }
            "Gelu" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Gelu operation requires 1 input".to_string(),
                    ));
                }
                Ok(vec![crate::ops::gelu(&inputs[0])?])
            }
            "Swish" => {
                if inputs.len() != 1 {
                    return Err(TensorError::invalid_argument(
                        "Swish operation requires 1 input".to_string(),
                    ));
                }
                Ok(vec![crate::ops::swish(&inputs[0])?])
            }
            _ => Err(TensorError::unsupported_operation_simple(format!(
                "Operation '{op_name}' not supported in session execution"
            ))),
        }
    }
}

impl Session for DefaultSession {
    fn run(
        &mut self,
        fetches: &[FetchSpec],
        feed_dict: &FeedDict,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        if self.closed {
            return Err(TensorError::invalid_argument(
                "Session is closed".to_string(),
            ));
        }

        // Get or create execution plan
        let plan = if let Some(cached_plan) = self.execution_cache.get(fetches) {
            cached_plan.clone()
        } else {
            let plan = self.create_execution_plan(fetches)?;
            self.execution_cache.insert(fetches.to_vec(), plan.clone());
            plan
        };

        // Execute nodes in topological order
        let mut node_values: HashMap<NodeId, Vec<Tensor<f32>>> = HashMap::new();

        for &node_id in &plan.execution_order {
            self.execute_node(node_id, &mut node_values, feed_dict)?;
        }

        // Collect results
        let mut results = Vec::new();
        for fetch in fetches {
            if let Some(&(node_id, output_idx)) = plan.output_mapping.get(fetch) {
                if let Some(outputs) = node_values.get(&node_id) {
                    if output_idx < outputs.len() {
                        results.push(outputs[output_idx].clone());
                    } else {
                        return Err(TensorError::invalid_argument(format!(
                            "Invalid output index {output_idx} for node {node_id}"
                        )));
                    }
                } else {
                    return Err(TensorError::invalid_argument(format!(
                        "Node {node_id} was not computed"
                    )));
                }
            } else {
                return Err(TensorError::invalid_argument(
                    "Invalid fetch specification".to_string(),
                ));
            }
        }

        Ok(results)
    }

    fn partial_run_setup(
        &mut self,
        feeds: &[String],
        fetches: &[FetchSpec],
        targets: &[String],
    ) -> Result<String, TensorError> {
        if self.closed {
            return Err(TensorError::invalid_argument(
                "Session is closed".to_string(),
            ));
        }

        // Create execution plan for fetches
        let plan = self.create_execution_plan(fetches)?;

        // Generate unique handle
        let handle = format!("partial_run_{}", self.next_partial_run_id);
        self.next_partial_run_id += 1;

        // Store partial run state
        let partial_state = PartialRunState {
            feeds: feeds.to_vec(),
            fetches: fetches.to_vec(),
            targets: targets.to_vec(),
            plan,
            intermediate_values: HashMap::new(),
        };

        self.partial_runs.insert(handle.clone(), partial_state);
        Ok(handle)
    }

    fn partial_run(
        &mut self,
        handle: &str,
        feed_dict: &FeedDict,
        fetches: &[FetchSpec],
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        if self.closed {
            return Err(TensorError::invalid_argument(
                "Session is closed".to_string(),
            ));
        }

        // Get the execution plan and intermediate values first
        let (execution_order, output_mapping, mut node_values) = {
            let partial_state = self.partial_runs.get(handle).ok_or_else(|| {
                TensorError::invalid_argument(format!("Invalid partial run handle: {handle}"))
            })?;
            (
                partial_state.plan.execution_order.clone(),
                partial_state.plan.output_mapping.clone(),
                partial_state.intermediate_values.clone(),
            )
        };

        // Execute nodes that aren't already computed
        for &node_id in &execution_order {
            if !node_values.contains_key(&node_id) {
                self.execute_node(node_id, &mut node_values, feed_dict)?;
            }
        }

        // Update intermediate values
        if let Some(partial_state) = self.partial_runs.get_mut(handle) {
            partial_state.intermediate_values = node_values.clone();
        }

        // Collect results
        let mut results = Vec::new();
        for fetch in fetches {
            if let Some(&(node_id, output_idx)) = output_mapping.get(fetch) {
                if let Some(outputs) = node_values.get(&node_id) {
                    if output_idx < outputs.len() {
                        results.push(outputs[output_idx].clone());
                    } else {
                        return Err(TensorError::invalid_argument(format!(
                            "Invalid output index {output_idx} for node {node_id}"
                        )));
                    }
                } else {
                    return Err(TensorError::invalid_argument(format!(
                        "Node {node_id} was not computed"
                    )));
                }
            } else {
                return Err(TensorError::invalid_argument(
                    "Invalid fetch specification".to_string(),
                ));
            }
        }

        Ok(results)
    }

    fn close(&mut self) -> Result<(), TensorError> {
        if self.closed {
            return Ok(());
        }

        // Clear caches and partial run state
        self.execution_cache.clear();
        self.partial_runs.clear();
        self.closed = true;

        Ok(())
    }
}

/// Convenience function to create a new session
pub fn create_session(
    graph: Arc<RwLock<Graph>>,
    config: Option<SessionConfig>,
    op_registry: Option<Arc<OpRegistry>>,
) -> DefaultSession {
    let config = config.unwrap_or_default();
    let op_registry = op_registry.unwrap_or_else(|| Arc::new(OpRegistry::new()));
    DefaultSession::new(graph, config, op_registry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device::Device,
        dtype::DType,
        graph::{AttributeValue, Graph, NodeType},
        shape::Shape,
        tensor::Tensor,
    };
    use std::collections::HashMap;

    #[test]
    fn test_session_creation() {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let session = create_session(graph, None, None);
        assert!(!session.closed);
    }

    #[test]
    fn test_simple_execution() {
        let mut graph = Graph::new();

        // Create placeholder
        let placeholder_id = graph
            .add_node(
                "input".to_string(),
                NodeType::Placeholder {
                    dtype: DType::Float32,
                    shape: Shape::new(vec![2, 2]),
                },
                Device::Cpu,
                HashMap::new(),
            )
            .unwrap();

        // Create identity operation
        let identity_id = graph
            .add_node(
                "output".to_string(),
                NodeType::Operation("Identity".to_string()),
                Device::Cpu,
                HashMap::new(),
            )
            .unwrap();

        // Connect them
        graph
            .add_edge(
                placeholder_id,
                identity_id,
                0,
                0,
                DType::Float32,
                Shape::new(vec![2, 2]),
                false,
            )
            .unwrap();

        let graph = Arc::new(RwLock::new(graph));
        let mut session = create_session(graph, None, None);

        // Create input tensor
        let input_tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let mut feed_dict = FeedDict::new();
        feed_dict.insert("input".to_string(), input_tensor.clone());

        // Run session
        let fetches = vec![FetchSpec::Name("output".to_string())];
        let results = session.run(&fetches, &feed_dict).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].shape(), input_tensor.shape());
    }

    #[test]
    fn test_addition_execution() {
        let mut graph = Graph::new();

        // Create two placeholders
        let input1_id = graph
            .add_node(
                "input1".to_string(),
                NodeType::Placeholder {
                    dtype: DType::Float32,
                    shape: Shape::new(vec![2]),
                },
                Device::Cpu,
                HashMap::new(),
            )
            .unwrap();

        let input2_id = graph
            .add_node(
                "input2".to_string(),
                NodeType::Placeholder {
                    dtype: DType::Float32,
                    shape: Shape::new(vec![2]),
                },
                Device::Cpu,
                HashMap::new(),
            )
            .unwrap();

        // Create add operation
        let add_id = graph
            .add_node(
                "add".to_string(),
                NodeType::Operation("Add".to_string()),
                Device::Cpu,
                HashMap::new(),
            )
            .unwrap();

        // Connect inputs to add
        graph
            .add_edge(
                input1_id,
                add_id,
                0,
                0,
                DType::Float32,
                Shape::new(vec![2]),
                false,
            )
            .unwrap();

        graph
            .add_edge(
                input2_id,
                add_id,
                0,
                1,
                DType::Float32,
                Shape::new(vec![2]),
                false,
            )
            .unwrap();

        let graph = Arc::new(RwLock::new(graph));
        let mut session = create_session(graph, None, None);

        // Create input tensors
        let input1 = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let input2 = Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2]).unwrap();

        let mut feed_dict = FeedDict::new();
        feed_dict.insert("input1".to_string(), input1);
        feed_dict.insert("input2".to_string(), input2);

        // Run session
        let fetches = vec![FetchSpec::Name("add".to_string())];
        let results = session.run(&fetches, &feed_dict).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].shape(), &Shape::new(vec![2]));

        // Check result values
        if let Some(result_slice) = results[0].as_slice() {
            assert!((result_slice[0] - 4.0).abs() < 1e-6); // 1.0 + 3.0
            assert!((result_slice[1] - 6.0).abs() < 1e-6); // 2.0 + 4.0
        } else {
            panic!("Failed to get tensor slice");
        }
    }

    #[test]
    fn test_session_close() {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let mut session = create_session(graph, None, None);

        session.close().unwrap();
        assert!(session.closed);

        // Trying to run after close should fail
        let feed_dict = FeedDict::new();
        let fetches = vec![];
        let result = session.run(&fetches, &feed_dict);
        assert!(result.is_err());
    }
}
