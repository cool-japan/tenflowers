use crate::layers::Layer;
use crate::model::Model;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tenflowers_core::{Result, Tensor, TensorError};

/// Unique identifier for nodes in the computation graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

/// Shared ID generator for all nodes and inputs
static GLOBAL_NODE_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Generate a unique NodeId
fn next_node_id() -> NodeId {
    NodeId(GLOBAL_NODE_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
}

/// Represents an input to the functional model
#[derive(Debug, Clone)]
pub struct Input<T> {
    id: NodeId,
    shape: Vec<usize>,
    name: Option<String>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Input<T> {
    /// Create a new input with given shape
    pub fn new(shape: Vec<usize>) -> Self {
        let id = next_node_id();

        Self {
            id,
            shape,
            name: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new input with given shape and name
    pub fn new_named(shape: Vec<usize>, name: String) -> Self {
        let mut input = Self::new(shape);
        input.name = Some(name);
        input
    }

    /// Get the shape of this input
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the name of this input
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get the unique ID of this input
    pub fn id(&self) -> NodeId {
        self.id
    }
}

/// Represents a node in the computation graph (output of a layer)
#[derive(Debug, Clone)]
pub struct Node<T> {
    id: NodeId,
    shape: Vec<usize>,
    inputs: Vec<NodeId>,
    layer_id: Option<usize>,
    name: Option<String>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Node<T> {
    /// Create a new node with given shape and inputs
    pub fn new(shape: Vec<usize>, inputs: Vec<NodeId>) -> Self {
        let id = next_node_id();

        Self {
            id,
            shape,
            inputs,
            layer_id: None,
            name: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new node from a layer applied to inputs
    pub fn from_layer(layer_id: usize, shape: Vec<usize>, inputs: Vec<NodeId>) -> Self {
        let mut node = Self::new(shape, inputs);
        node.layer_id = Some(layer_id);
        node
    }

    /// Create a new node with a name
    pub fn new_named(shape: Vec<usize>, inputs: Vec<NodeId>, name: String) -> Self {
        let mut node = Self::new(shape, inputs);
        node.name = Some(name);
        node
    }

    /// Get the shape of this node
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the input node IDs
    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    /// Get the layer ID if this node is the output of a layer
    pub fn layer_id(&self) -> Option<usize> {
        self.layer_id
    }

    /// Get the unique ID of this node
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Get the name of this node
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set the ID of this node (internal use only)
    fn set_id(&mut self, id: NodeId) {
        self.id = id;
    }
}

/// A shared layer that can be used multiple times in the computation graph
#[derive(Clone)]
pub struct SharedLayer<T> {
    id: usize,
    layer: Arc<Mutex<Box<dyn Layer<T>>>>,
    name: Option<String>,
}

impl<T> SharedLayer<T> {
    /// Create a new shared layer
    pub fn new(layer: Box<dyn Layer<T>>) -> Self {
        // Use separate ID space for layers (not NodeIds)
        static NEXT_LAYER_ID: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(100_000);
        let id = NEXT_LAYER_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Self {
            id,
            layer: Arc::new(Mutex::new(layer)),
            name: None,
        }
    }

    /// Create a new shared layer with a name
    pub fn new_named(layer: Box<dyn Layer<T>>, name: String) -> Self {
        let mut shared = Self::new(layer);
        shared.name = Some(name);
        shared
    }

    /// Get the unique ID of this shared layer
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the name of this shared layer
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Execute forward pass through the shared layer
    pub fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let layer = self
            .layer
            .lock()
            .map_err(|_| TensorError::InvalidArgument {
                operation: "SharedLayer::forward".to_string(),
                reason: "Failed to acquire lock on shared layer".to_string(),
                context: None,
            })?;
        layer.forward(input)
    }

    /// Get parameters from the shared layer (returns copies to avoid borrowing issues)
    pub fn parameters(&self) -> Result<Vec<Tensor<T>>>
    where
        T: Clone,
    {
        let layer = self
            .layer
            .lock()
            .map_err(|_| TensorError::InvalidArgument {
                operation: "SharedLayer::parameters".to_string(),
                reason: "Failed to acquire lock on shared layer".to_string(),
                context: None,
            })?;
        // Return cloned parameters to avoid borrowing issues
        Ok(layer.parameters().into_iter().cloned().collect())
    }

    /// Apply a function to the mutable layer (for parameter updates)
    pub fn with_layer_mut<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Box<dyn Layer<T>>) -> R,
    {
        let mut layer = self
            .layer
            .lock()
            .map_err(|_| TensorError::InvalidArgument {
                operation: "SharedLayer::with_layer_mut".to_string(),
                reason: "Failed to acquire lock on shared layer".to_string(),
                context: None,
            })?;
        Ok(f(&mut *layer))
    }

    /// Set training mode for the shared layer
    pub fn set_training(&self, training: bool) -> Result<()> {
        let mut layer = self
            .layer
            .lock()
            .map_err(|_| TensorError::InvalidArgument {
                operation: "SharedLayer::set_training".to_string(),
                reason: "Failed to acquire lock on shared layer".to_string(),
                context: None,
            })?;
        layer.set_training(training);
        Ok(())
    }

    /// Apply this shared layer to an input node
    pub fn call(&self, input: &Node<T>) -> Result<Node<T>> {
        // For shape inference, we'd typically need to call the layer
        // For now, we assume the output shape equals the input shape
        // In a real implementation, we'd need proper shape inference
        let output_shape = input.shape().to_vec();

        Ok(Node::from_layer(self.id, output_shape, vec![input.id()]))
    }

    /// Apply this shared layer to multiple input nodes (for layers that support multiple inputs)
    pub fn call_multi(&self, inputs: &[&Node<T>]) -> Result<Node<T>> {
        if inputs.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "SharedLayer::call_multi".to_string(),
                reason: "At least one input is required".to_string(),
                context: None,
            });
        }

        // For multi-input layers, we might need custom shape inference logic
        // For now, use the shape of the first input
        let output_shape = inputs[0].shape().to_vec();
        let input_ids = inputs.iter().map(|node| node.id()).collect();

        Ok(Node::from_layer(self.id, output_shape, input_ids))
    }
}

/// Layer operation types for the functional API
#[allow(clippy::type_complexity)]
pub enum LayerOp<T> {
    /// A regular layer with single input/output
    Single(Box<dyn Layer<T>>),
    /// A shared layer that can be reused
    Shared(SharedLayer<T>),
    /// A custom operation function
    Custom(Box<dyn Fn(&[&Tensor<T>]) -> Result<Tensor<T>> + Send + Sync>),
}

/// A functional model that supports multi-input/output and shared layers
pub struct FunctionalModel<T> {
    /// Input specifications
    inputs: Vec<Input<T>>,
    /// Output node IDs
    outputs: Vec<NodeId>,
    /// All nodes in the computation graph
    nodes: HashMap<NodeId, Node<T>>,
    /// Layers indexed by ID
    layers: HashMap<usize, LayerOp<T>>,
    /// Execution order (topologically sorted node IDs)
    execution_order: Vec<NodeId>,
    /// Training mode flag
    training: bool,
    /// Model name
    name: Option<String>,
}

impl<T> FunctionalModel<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new functional model
    pub fn new(
        inputs: Vec<Input<T>>,
        outputs: Vec<Node<T>>,
        layers: HashMap<usize, LayerOp<T>>,
    ) -> Result<Self> {
        let mut nodes = HashMap::new();
        let mut output_ids = Vec::new();

        // Add input nodes - create them with the input's original ID
        for input in &inputs {
            let mut input_node = Node::new(input.shape().to_vec(), vec![]);
            // Override the node's ID to match the input's ID
            input_node.set_id(input.id());
            nodes.insert(input.id(), input_node);
        }

        // Add output nodes and collect their dependencies
        let mut all_nodes = HashMap::new();
        for output in &outputs {
            output_ids.push(output.id());
        }

        // Collect all dependencies
        Self::collect_all_dependencies(&outputs, &mut all_nodes)?;

        // Ensure input nodes are available for dependency verification
        for (input_id, input_node) in &nodes {
            if input_node.inputs().is_empty() && input_node.layer_id().is_none() {
                all_nodes.insert(*input_id, input_node.clone());
            }
        }

        // Merge collected nodes - input nodes should never be replaced
        for (id, node) in all_nodes {
            // Check if this node would create a self-reference
            if node.inputs().contains(&id) {
                return Err(TensorError::InvalidArgument {
                    operation: "FunctionalModel::from_nodes".to_string(),
                    reason: format!("Node {id:?} cannot depend on itself"),
                    context: None,
                });
            }

            // If there's already an input node with this ID, don't replace it
            // Computation nodes should have different IDs from input nodes
            if let Some(existing_input_node) = nodes.get(&id) {
                if existing_input_node.inputs().is_empty()
                    && existing_input_node.layer_id().is_none()
                {
                    // This is an input node, keep it and don't replace
                    continue;
                }
            }

            // Add the computation node
            nodes.insert(id, node);
        }

        // Verify all dependencies are satisfied
        Self::verify_dependencies(&nodes)?;

        // Compute execution order (topological sort)
        let execution_order = Self::topological_sort(&nodes)?;

        Ok(Self {
            inputs,
            outputs: output_ids,
            nodes,
            layers,
            execution_order,
            training: true,
            name: None,
        })
    }

    /// Create a functional model with a name
    pub fn new_named(
        inputs: Vec<Input<T>>,
        outputs: Vec<Node<T>>,
        layers: HashMap<usize, LayerOp<T>>,
        name: String,
    ) -> Result<Self> {
        let mut model = Self::new(inputs, outputs, layers)?;
        model.name = Some(name);
        Ok(model)
    }

    /// Recursively collect all nodes in the computation graph
    fn collect_nodes(node: &Node<T>, collected: &mut HashMap<NodeId, Node<T>>) -> Result<()> {
        if collected.contains_key(&node.id()) {
            return Ok(()); // Already collected
        }

        collected.insert(node.id(), node.clone());
        Ok(())
    }

    /// Collect all nodes (when all intermediate nodes are provided)
    fn collect_all_dependencies(
        outputs: &[Node<T>],
        collected: &mut HashMap<NodeId, Node<T>>,
    ) -> Result<()> {
        // Simply collect all provided nodes
        for output in outputs {
            Self::collect_nodes(output, collected)?;
        }

        Ok(())
    }

    /// Verify that all dependencies are satisfied in the final node collection
    fn verify_dependencies(nodes: &HashMap<NodeId, Node<T>>) -> Result<()> {
        for (node_id, node) in nodes.iter() {
            for input_id in node.inputs() {
                if !nodes.contains_key(input_id) {
                    return Err(TensorError::InvalidArgument {
                        operation: "verify_dependencies".to_string(),
                        reason: format!(
                            "Node {node_id:?} references non-existent input node {input_id:?}"
                        ),
                        context: None,
                    });
                }
            }
        }

        Ok(())
    }

    /// Perform topological sort to determine execution order
    fn topological_sort(nodes: &HashMap<NodeId, Node<T>>) -> Result<Vec<NodeId>> {
        let mut in_degree = HashMap::new();
        let mut adj_list = HashMap::new();

        // Initialize in-degree and adjacency list
        for (node_id, node) in nodes {
            in_degree.insert(*node_id, node.inputs().len());
            adj_list.insert(*node_id, Vec::new());
        }

        // Build adjacency list
        for (node_id, node) in nodes {
            for input_id in node.inputs() {
                if let Some(successors) = adj_list.get_mut(input_id) {
                    successors.push(*node_id);
                } else {
                    // Referenced node doesn't exist in the graph
                    return Err(TensorError::InvalidArgument {
                        operation: "topological_sort".to_string(),
                        reason: format!(
                            "Node {node_id:?} references non-existent input node {input_id:?}"
                        ),
                        context: None,
                    });
                }
            }
        }

        // Topological sort using Kahn's algorithm
        let mut queue = Vec::new();
        let mut result = Vec::new();

        // Find nodes with no incoming edges
        for (node_id, degree) in &in_degree {
            if *degree == 0 {
                queue.push(*node_id);
            }
        }

        while let Some(node_id) = queue.pop() {
            result.push(node_id);

            if let Some(successors) = adj_list.get(&node_id) {
                for successor in successors {
                    if let Some(degree) = in_degree.get_mut(successor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push(*successor);
                        }
                    }
                }
            }
        }

        if result.len() != nodes.len() {
            return Err(TensorError::InvalidArgument {
                operation: "topological_sort".to_string(),
                reason: "Cycle detected in computation graph".to_string(),
                context: None,
            });
        }

        Ok(result)
    }

    /// Forward pass with multiple inputs
    pub fn forward_multi(&self, inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> {
        if inputs.len() != self.inputs.len() {
            return Err(TensorError::InvalidArgument {
                operation: "forward_multi".to_string(),
                reason: format!(
                    "Expected {} inputs, got {}",
                    self.inputs.len(),
                    inputs.len()
                ),
                context: None,
            });
        }

        let mut activations = HashMap::new();

        // Set input activations
        for (i, input) in self.inputs.iter().enumerate() {
            activations.insert(input.id(), inputs[i].clone());
        }

        // Execute nodes in topological order
        for node_id in &self.execution_order {
            if activations.contains_key(node_id) {
                continue; // Skip inputs
            }

            let node = self
                .nodes
                .get(node_id)
                .ok_or_else(|| TensorError::InvalidArgument {
                    operation: "forward_multi".to_string(),
                    reason: format!("Node {node_id:?} not found"),
                    context: None,
                })?;

            if let Some(layer_id) = node.layer_id() {
                let layer_op =
                    self.layers
                        .get(&layer_id)
                        .ok_or_else(|| TensorError::InvalidArgument {
                            operation: "forward_multi".to_string(),
                            reason: format!("Layer {layer_id} not found"),
                            context: None,
                        })?;

                // Collect input activations
                let input_tensors: Result<Vec<_>> = node
                    .inputs()
                    .iter()
                    .map(|input_id| {
                        activations
                            .get(input_id)
                            .ok_or_else(|| TensorError::InvalidArgument {
                                operation: "forward_multi".to_string(),
                                reason: format!("Input activation for {input_id:?} not found"),
                                context: None,
                            })
                    })
                    .collect();
                let input_tensors = input_tensors?;

                // Execute layer
                let output = match layer_op {
                    LayerOp::Single(layer) => {
                        if input_tensors.len() != 1 {
                            return Err(TensorError::InvalidArgument {
                                operation: "forward_multi".to_string(),
                                reason: "Single input layer received multiple inputs".to_string(),
                                context: None,
                            });
                        }
                        layer.forward(input_tensors[0])?
                    }
                    LayerOp::Shared(shared_layer) => {
                        if input_tensors.len() != 1 {
                            return Err(TensorError::InvalidArgument {
                                operation: "forward_multi".to_string(),
                                reason: "Shared layer received multiple inputs (multi-input not yet supported)".to_string(),
                                context: None,
                            });
                        }
                        let layer = shared_layer.layer.lock().unwrap();
                        layer.forward(input_tensors[0])?
                    }
                    LayerOp::Custom(custom_fn) => custom_fn(&input_tensors)?,
                };

                activations.insert(*node_id, output);
            }
        }

        // Collect outputs
        let outputs: Result<Vec<_>> = self
            .outputs
            .iter()
            .map(|output_id| {
                activations
                    .get(output_id)
                    .ok_or_else(|| TensorError::InvalidArgument {
                        operation: "forward_multi".to_string(),
                        reason: format!("Output activation for {output_id:?} not found"),
                        context: None,
                    })
                    .cloned()
            })
            .collect();

        outputs
    }

    /// Get model inputs
    pub fn inputs(&self) -> &[Input<T>] {
        &self.inputs
    }

    /// Get output node IDs
    pub fn output_ids(&self) -> &[NodeId] {
        &self.outputs
    }

    /// Get the model name
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get the number of inputs
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Get the number of outputs
    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Set the output nodes (for advanced use cases where all intermediate nodes
    /// are provided to build() but only some should be considered outputs)
    pub fn set_outputs(&mut self, output_ids: Vec<NodeId>) {
        self.outputs = output_ids;
    }
}

impl<T> Model<T> for FunctionalModel<T>
where
    T: Clone + Default + Send + Sync + 'static + num_traits::Zero,
{
    /// Forward pass with single input (for compatibility with Model trait)
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if self.inputs.len() != 1 {
            return Err(TensorError::InvalidArgument {
                operation: "Model::forward".to_string(),
                reason: "Single input forward called on multi-input model".to_string(),
                context: None,
            });
        }

        if self.outputs.len() != 1 {
            return Err(TensorError::InvalidArgument {
                operation: "Model::forward".to_string(),
                reason: "Single output forward called on multi-output model".to_string(),
                context: None,
            });
        }

        let outputs = self.forward_multi(&[input])?;
        Ok(outputs.into_iter().next().unwrap())
    }

    /// Get all parameters from all layers
    /// Note: This implementation has limitations with shared layers due to borrowing rules
    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();
        for layer_op in self.layers.values() {
            match layer_op {
                LayerOp::Single(layer) => {
                    params.extend(layer.parameters());
                }
                LayerOp::Shared(_shared_layer) => {
                    // Note: Shared layers cannot provide direct parameter references due to Arc<Mutex<>>
                    // Use get_shared_layer_parameters() method for shared layer parameter access
                    // This is a fundamental limitation of the Model trait design with shared ownership
                }
                LayerOp::Custom(_) => {
                    // Custom operations don't have parameters
                }
            }
        }
        params
    }

    /// Get all mutable parameters from all layers
    /// Note: This implementation has limitations with shared layers due to borrowing rules
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();
        for layer_op in self.layers.values_mut() {
            match layer_op {
                LayerOp::Single(layer) => {
                    params.extend(layer.parameters_mut());
                }
                LayerOp::Shared(_shared_layer) => {
                    // Note: Shared layers cannot provide direct parameter references due to Arc<Mutex<>>
                    // Use get_shared_layer_parameters() method for shared layer parameter access
                    // This is a fundamental limitation of the Model trait design with shared ownership
                }
                LayerOp::Custom(_) => {
                    // Custom operations don't have parameters
                }
            }
        }
        params
    }

    /// Set training mode for all layers
    fn set_training(&mut self, training: bool) {
        self.training = training;
        for layer_op in self.layers.values_mut() {
            match layer_op {
                LayerOp::Single(layer) => {
                    layer.set_training(training);
                }
                LayerOp::Shared(shared_layer) => {
                    let mut layer = shared_layer.layer.lock().unwrap();
                    layer.set_training(training);
                }
                LayerOp::Custom(_) => {
                    // Custom operations don't have training mode
                }
            }
        }
    }

    /// Zero gradients for all parameters
    fn zero_grad(&mut self) {
        for param in self.parameters_mut() {
            if param.requires_grad() {
                let zero_grad = Tensor::zeros(param.shape().dims());
                param.set_grad(Some(zero_grad));
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<T> FunctionalModel<T>
where
    T: Clone,
{
    /// Get parameters from all shared layers in the model
    /// Returns a vector of parameter tensors (cloned to avoid borrowing issues)
    pub fn get_shared_layer_parameters(&self) -> Result<Vec<Tensor<T>>> {
        let mut all_params = Vec::new();
        let mut processed_shared_layers = std::collections::HashSet::new();

        for layer_op in self.layers.values() {
            if let LayerOp::Shared(shared_layer) = layer_op {
                // Only process each unique shared layer once
                if !processed_shared_layers.contains(&shared_layer.id()) {
                    let params = shared_layer.parameters()?;
                    all_params.extend(params);
                    processed_shared_layers.insert(shared_layer.id());
                }
            }
        }

        Ok(all_params)
    }

    /// Update parameters for a specific shared layer
    /// This allows parameter updates while respecting the Arc<Mutex<>> ownership
    pub fn update_shared_layer_parameters<F>(
        &self,
        shared_layer_id: usize,
        updater: F,
    ) -> Result<()>
    where
        F: FnOnce(&mut Vec<&mut Tensor<T>>),
    {
        for layer_op in self.layers.values() {
            if let LayerOp::Shared(shared_layer) = layer_op {
                if shared_layer.id() == shared_layer_id {
                    return shared_layer.with_layer_mut(|layer| {
                        let mut params = layer.parameters_mut();
                        updater(&mut params);
                    });
                }
            }
        }

        Err(TensorError::InvalidArgument {
            operation: "get_shared_layer".to_string(),
            reason: format!("Shared layer with ID {shared_layer_id} not found"),
            context: None,
        })
    }

    /// Get all unique shared layer IDs in the model
    pub fn get_shared_layer_ids(&self) -> Vec<usize> {
        let mut ids = std::collections::HashSet::new();

        for layer_op in self.layers.values() {
            if let LayerOp::Shared(shared_layer) = layer_op {
                ids.insert(shared_layer.id());
            }
        }

        ids.into_iter().collect()
    }

    /// Apply a function to all shared layers (for operations like setting training mode)
    pub fn with_all_shared_layers<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&SharedLayer<T>) -> Result<()>,
    {
        let mut processed_shared_layers = std::collections::HashSet::new();

        for layer_op in self.layers.values() {
            if let LayerOp::Shared(shared_layer) = layer_op {
                if !processed_shared_layers.contains(&shared_layer.id()) {
                    f(shared_layer)?;
                    processed_shared_layers.insert(shared_layer.id());
                }
            }
        }

        Ok(())
    }
}

/// Builder for creating functional models
pub struct FunctionalModelBuilder<T> {
    inputs: Vec<Input<T>>,
    layers: HashMap<usize, LayerOp<T>>,
    name: Option<String>,
}

impl<T> FunctionalModelBuilder<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            layers: HashMap::new(),
            name: None,
        }
    }

    /// Add an input to the model
    pub fn add_input(mut self, input: Input<T>) -> Self {
        self.inputs.push(input);
        self
    }

    /// Add a single-use layer
    pub fn add_layer(mut self, id: usize, layer: Box<dyn Layer<T>>) -> Self {
        self.layers.insert(id, LayerOp::Single(layer));
        self
    }

    /// Add a shared layer
    pub fn add_shared_layer(mut self, shared_layer: SharedLayer<T>) -> Self {
        let id = shared_layer.id();
        self.layers.insert(id, LayerOp::Shared(shared_layer));
        self
    }

    /// Add a custom operation
    #[allow(clippy::type_complexity)]
    pub fn add_custom_op(
        mut self,
        id: usize,
        op: Box<dyn Fn(&[&Tensor<T>]) -> Result<Tensor<T>> + Send + Sync>,
    ) -> Self {
        self.layers.insert(id, LayerOp::Custom(op));
        self
    }

    /// Set the model name
    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Build the functional model
    pub fn build(self, outputs: Vec<Node<T>>) -> Result<FunctionalModel<T>> {
        if let Some(name) = self.name {
            FunctionalModel::new_named(self.inputs, outputs, self.layers, name)
        } else {
            FunctionalModel::new(self.inputs, outputs, self.layers)
        }
    }
}

impl<T> Default for FunctionalModelBuilder<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::dense::Dense;
    use tenflowers_core::Tensor;

    #[test]
    fn test_input_creation() {
        let input = Input::<f32>::new(vec![10, 64]);
        assert_eq!(input.shape(), &[10, 64]);
        assert!(input.name().is_none());

        let named_input = Input::<f32>::new_named(vec![5, 32], "input1".to_string());
        assert_eq!(named_input.shape(), &[5, 32]);
        assert_eq!(named_input.name(), Some("input1"));
    }

    #[test]
    fn test_node_creation() {
        let input = Input::<f32>::new(vec![10, 64]);
        let node = Node::<f32>::new(vec![10, 32], vec![input.id()]);

        assert_eq!(node.shape(), &[10, 32]);
        assert_eq!(node.inputs(), &[input.id()]);
        assert!(node.layer_id().is_none());
    }

    #[test]
    fn test_shared_layer() {
        let dense = Dense::<f32>::new(64, 32, true);
        let shared_layer = SharedLayer::new(Box::new(dense));

        let input = Input::<f32>::new(vec![10, 64]);
        let input_node = Node::<f32>::new(vec![10, 64], vec![]);

        let output_node = shared_layer.call(&input_node).unwrap();
        assert_eq!(output_node.layer_id(), Some(shared_layer.id()));
        assert_eq!(output_node.inputs(), &[input_node.id()]);
    }

    #[test]
    fn test_builder_pattern() {
        let input = Input::<f32>::new(vec![10, 64]);
        let dense = Dense::<f32>::new(64, 32, true);

        let builder = FunctionalModelBuilder::new()
            .add_input(input.clone())
            .add_layer(0, Box::new(dense))
            .name("test_model".to_string());

        // Create output node
        let output_node = Node::<f32>::from_layer(0, vec![10, 32], vec![input.id()]);

        let model = builder.build(vec![output_node]).unwrap();
        assert_eq!(model.name(), Some("test_model"));
        assert_eq!(model.num_inputs(), 1);
        assert_eq!(model.num_outputs(), 1);
    }

    #[test]
    fn test_multi_input_model() {
        // Create inputs
        let input1 = Input::<f32>::new(vec![10, 32]);
        let input2 = Input::<f32>::new(vec![10, 64]);

        // Create layers
        let dense1 = Dense::<f32>::new(32, 16, true);
        let dense2 = Dense::<f32>::new(64, 16, true);

        // Build model (though we can't easily test the full forward pass here)
        let builder = FunctionalModelBuilder::new()
            .add_input(input1.clone())
            .add_input(input2.clone())
            .add_layer(0, Box::new(dense1))
            .add_layer(1, Box::new(dense2))
            .name("multi_input_model".to_string());

        // Create intermediate nodes
        let node1 = Node::<f32>::from_layer(0, vec![10, 16], vec![input1.id()]);
        let node2 = Node::<f32>::from_layer(1, vec![10, 16], vec![input2.id()]);

        let model = builder.build(vec![node1, node2]).unwrap();
        assert_eq!(model.num_inputs(), 2);
        assert_eq!(model.num_outputs(), 2);
    }
}
