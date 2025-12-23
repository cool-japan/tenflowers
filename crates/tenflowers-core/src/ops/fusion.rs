/// Elementwise Operation Fusion Pass for TenfloweRS
///
/// This module implements automatic fusion of elementwise tensor operations to reduce
/// memory bandwidth requirements and improve performance.
use crate::{DType, Device, Result, Shape, Tensor, TensorError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Type of elementwise operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementwiseOpType {
    /// Addition (a + b)
    Add,
    /// Subtraction (a - b)
    Sub,
    /// Multiplication (a * b)
    Mul,
    /// Division (a / b)
    Div,
    /// Power (a ^ b)
    Pow,
    /// ReLU activation
    ReLU,
    /// Tanh activation
    Tanh,
    /// Sigmoid activation
    Sigmoid,
    /// GELU activation
    GELU,
    /// Negation (-a)
    Neg,
    /// Reciprocal (1/a)
    Reciprocal,
    /// Square root
    Sqrt,
    /// Exponential
    Exp,
    /// Natural logarithm
    Log,
    /// Sine
    Sin,
    /// Cosine
    Cos,
    /// Absolute value
    Abs,
}

impl ElementwiseOpType {
    /// Check if this is a unary operation
    pub fn is_unary(&self) -> bool {
        matches!(
            self,
            Self::ReLU
                | Self::Tanh
                | Self::Sigmoid
                | Self::GELU
                | Self::Neg
                | Self::Reciprocal
                | Self::Sqrt
                | Self::Exp
                | Self::Log
                | Self::Sin
                | Self::Cos
                | Self::Abs
        )
    }

    /// Check if this is a binary operation
    pub fn is_binary(&self) -> bool {
        !self.is_unary()
    }

    /// Get operation name for code generation
    pub fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Pow => "pow",
            Self::ReLU => "relu",
            Self::Tanh => "tanh",
            Self::Sigmoid => "sigmoid",
            Self::GELU => "gelu",
            Self::Neg => "neg",
            Self::Reciprocal => "reciprocal",
            Self::Sqrt => "sqrt",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Abs => "abs",
        }
    }
}

/// Node in the operation fusion graph
#[derive(Debug, Clone)]
pub struct FusionNode {
    /// Unique node ID
    pub id: usize,
    /// Operation type
    pub op_type: ElementwiseOpType,
    /// Input node IDs (0 or 1 for unary, 0 and 1 for binary)
    pub inputs: Vec<usize>,
    /// Output consumers (which nodes consume this output)
    pub consumers: Vec<usize>,
    /// Whether this node is a graph input
    pub is_input: bool,
    /// Whether this node is a graph output
    pub is_output: bool,
    /// Data type
    pub dtype: DType,
    /// Shape
    pub shape: Shape,
}

/// Fusion graph representing a chain of elementwise operations
#[derive(Debug, Clone)]
pub struct FusionGraph {
    /// Nodes in the graph
    pub nodes: Vec<FusionNode>,
    /// Input node IDs
    pub inputs: Vec<usize>,
    /// Output node IDs
    pub outputs: Vec<usize>,
    /// Device for execution
    pub device: Device,
}

impl FusionGraph {
    /// Create a new empty fusion graph
    pub fn new(device: Device) -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            device,
        }
    }

    /// Add an input node
    pub fn add_input(&mut self, shape: Shape, dtype: DType) -> usize {
        let id = self.nodes.len();
        self.nodes.push(FusionNode {
            id,
            op_type: ElementwiseOpType::Add, // Placeholder, unused for inputs
            inputs: Vec::new(),
            consumers: Vec::new(),
            is_input: true,
            is_output: false,
            dtype,
            shape,
        });
        self.inputs.push(id);
        id
    }

    /// Add an operation node
    pub fn add_op(
        &mut self,
        op_type: ElementwiseOpType,
        inputs: Vec<usize>,
        shape: Shape,
        dtype: DType,
    ) -> usize {
        let id = self.nodes.len();

        // Update consumers for input nodes
        for &input_id in &inputs {
            if let Some(input_node) = self.nodes.get_mut(input_id) {
                input_node.consumers.push(id);
            }
        }

        self.nodes.push(FusionNode {
            id,
            op_type,
            inputs,
            consumers: Vec::new(),
            is_input: false,
            is_output: false,
            dtype,
            shape,
        });
        id
    }

    /// Mark a node as output
    pub fn mark_output(&mut self, node_id: usize) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.is_output = true;
            if !self.outputs.contains(&node_id) {
                self.outputs.push(node_id);
            }
        }
    }

    /// Check if this graph can be fused
    pub fn is_fusible(&self) -> bool {
        // A graph is fusible if:
        // 1. All operations are elementwise
        // 2. All intermediate results have the same shape
        // 3. There are at least 2 operations to fuse

        if self.nodes.len() < 3 {
            // Need at least 1 input + 2 ops
            return false;
        }

        // Check all non-input nodes are elementwise (already ensured by construction)
        // Check shapes are compatible
        let first_shape = self.nodes.first().map(|n| &n.shape);
        for node in &self.nodes {
            if !node.is_input && first_shape != Some(&node.shape) {
                return false;
            }
        }

        true
    }

    /// Get estimated memory savings from fusion (in bytes)
    pub fn estimated_memory_savings(&self) -> usize {
        if !self.is_fusible() {
            return 0;
        }

        // Each intermediate result that doesn't need to be materialized saves memory
        let dtype_size = match self
            .nodes
            .first()
            .map(|n| n.dtype)
            .unwrap_or(DType::Float32)
        {
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Int8 => 1,
            DType::UInt8 => 1,
            _ => 4,
        };

        let shape_size: usize = self
            .nodes
            .first()
            .map(|n| n.shape.dims().iter().product())
            .unwrap_or(1);

        let intermediate_count = self
            .nodes
            .iter()
            .filter(|n| !n.is_input && !n.is_output)
            .count();

        intermediate_count * shape_size * dtype_size
    }

    /// Generate a human-readable description of the fused operation
    pub fn description(&self) -> String {
        let mut desc = String::from("Fused operation: ");
        for node in &self.nodes {
            if !node.is_input {
                desc.push_str(&format!("{} -> ", node.op_type.name()));
            }
        }
        desc.truncate(desc.len() - 4); // Remove trailing " -> "
        desc
    }
}

/// Fusion pass builder for creating fusion graphs
pub struct FusionPassBuilder<T> {
    graph: FusionGraph,
    tensor_to_node: HashMap<*const Tensor<T>, usize>,
}

impl<T> FusionPassBuilder<T> {
    /// Create a new fusion pass builder
    pub fn new(device: Device) -> Self {
        Self {
            graph: FusionGraph::new(device),
            tensor_to_node: HashMap::new(),
        }
    }

    /// Record an input tensor
    pub fn input(&mut self, tensor: &Tensor<T>) -> usize
    where
        T: scirs2_core::num_traits::Float + Default + 'static,
    {
        let ptr = tensor as *const Tensor<T>;
        if let Some(&node_id) = self.tensor_to_node.get(&ptr) {
            node_id
        } else {
            let node_id = self.graph.add_input(tensor.shape().clone(), tensor.dtype());
            self.tensor_to_node.insert(ptr, node_id);
            node_id
        }
    }

    /// Record a unary operation
    pub fn unary_op(
        &mut self,
        op_type: ElementwiseOpType,
        input: &Tensor<T>,
        output: &Tensor<T>,
    ) -> usize
    where
        T: scirs2_core::num_traits::Float + Default + 'static,
    {
        let input_node = self.input(input);
        let output_node = self.graph.add_op(
            op_type,
            vec![input_node],
            output.shape().clone(),
            output.dtype(),
        );

        let ptr = output as *const Tensor<T>;
        self.tensor_to_node.insert(ptr, output_node);
        output_node
    }

    /// Record a binary operation
    pub fn binary_op(
        &mut self,
        op_type: ElementwiseOpType,
        lhs: &Tensor<T>,
        rhs: &Tensor<T>,
        output: &Tensor<T>,
    ) -> usize
    where
        T: scirs2_core::num_traits::Float + Default + 'static,
    {
        let lhs_node = self.input(lhs);
        let rhs_node = self.input(rhs);
        let output_node = self.graph.add_op(
            op_type,
            vec![lhs_node, rhs_node],
            output.shape().clone(),
            output.dtype(),
        );

        let ptr = output as *const Tensor<T>;
        self.tensor_to_node.insert(ptr, output_node);
        output_node
    }

    /// Mark a tensor as a graph output
    pub fn output(&mut self, tensor: &Tensor<T>)
    where
        T: scirs2_core::num_traits::Float + Default + 'static,
    {
        let ptr = tensor as *const Tensor<T>;
        if let Some(&node_id) = self.tensor_to_node.get(&ptr) {
            self.graph.mark_output(node_id);
        }
    }

    /// Build the fusion graph
    pub fn build(self) -> FusionGraph {
        self.graph
    }
}

/// Execute a fused operation graph
pub fn execute_fused_graph<T>(graph: &FusionGraph, inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>>
where
    T: scirs2_core::num_traits::Float
        + Default
        + std::fmt::Debug
        + bytemuck::Pod
        + Send
        + Sync
        + 'static,
{
    if inputs.len() != graph.inputs.len() {
        return Err(TensorError::invalid_argument(format!(
            "Expected {} inputs, got {}",
            graph.inputs.len(),
            inputs.len()
        )));
    }

    // Try GPU fusion for supported patterns
    #[cfg(feature = "gpu")]
    if matches!(graph.device, Device::Gpu(_)) && graph.is_fusible() {
        if let Ok(result) = try_gpu_fusion(graph, inputs) {
            return Ok(result);
        }
    }

    // Fall back to CPU sequential execution
    // Store intermediate results in a Vec to avoid memory leaks
    let mut intermediate_results: Vec<Tensor<T>> = Vec::new();
    let mut node_value_indices: HashMap<usize, usize> = HashMap::new();

    // Map input nodes to their tensor indices (using special indices < inputs.len())
    for (i, &input_id) in graph.inputs.iter().enumerate() {
        node_value_indices.insert(input_id, i);
    }

    // Execute nodes in topological order
    let mut outputs = Vec::new();
    for node in &graph.nodes {
        if node.is_input {
            continue; // Already handled
        }

        // Get input tensors
        let input_tensors: Vec<&Tensor<T>> = node
            .inputs
            .iter()
            .map(|&id| {
                let idx = node_value_indices.get(&id).unwrap();
                if *idx < inputs.len() {
                    // This is an original input
                    inputs[*idx]
                } else {
                    // This is an intermediate result
                    &intermediate_results[*idx - inputs.len()]
                }
            })
            .collect();

        // Execute operation
        let result = match node.op_type {
            ElementwiseOpType::Add if input_tensors.len() == 2 => {
                crate::ops::binary::add(input_tensors[0], input_tensors[1])?
            }
            ElementwiseOpType::Mul if input_tensors.len() == 2 => {
                crate::ops::binary::mul(input_tensors[0], input_tensors[1])?
            }
            ElementwiseOpType::Sub if input_tensors.len() == 2 => {
                crate::ops::binary::sub(input_tensors[0], input_tensors[1])?
            }
            ElementwiseOpType::Div if input_tensors.len() == 2 => {
                crate::ops::binary::div(input_tensors[0], input_tensors[1])?
            }
            ElementwiseOpType::ReLU if input_tensors.len() == 1 => {
                crate::ops::activation::relu(input_tensors[0])?
            }
            ElementwiseOpType::Tanh if input_tensors.len() == 1 => {
                crate::ops::activation::tanh(input_tensors[0])?
            }
            ElementwiseOpType::Sigmoid if input_tensors.len() == 1 => {
                crate::ops::activation::sigmoid(input_tensors[0])?
            }
            _ => {
                return Err(TensorError::not_implemented_simple(format!(
                    "Fused operation {} not yet implemented",
                    node.op_type.name()
                )))
            }
        };

        // Store result for subsequent nodes to access
        // Store the index where this result is located (offset by input count)
        let result_index = inputs.len() + intermediate_results.len();
        node_value_indices.insert(node.id, result_index);

        // Collect outputs before storing in intermediate_results
        if node.is_output {
            outputs.push(result.clone());
        }

        // Store in intermediate results
        intermediate_results.push(result);
    }

    Ok(outputs)
}

/// Try to execute fusion on GPU using simple_elementwise_fusion kernel
#[cfg(feature = "gpu")]
fn try_gpu_fusion<T>(graph: &FusionGraph, inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>>
where
    T: scirs2_core::num_traits::Float
        + Default
        + std::fmt::Debug
        + bytemuck::Pod
        + Send
        + Sync
        + 'static,
{
    use crate::device::context::get_gpu_context;
    use crate::gpu::buffer::GpuBuffer;
    use wgpu::util::DeviceExt;

    // Only support simple 3-input patterns for MVP:  (a op1 b) op2 c [activation]
    if inputs.len() != 3 {
        return Err(TensorError::not_implemented_simple(
            "GPU fusion only supports 3-input patterns for MVP".to_string(),
        ));
    }

    // Get device context
    let device_id = match graph.device {
        Device::Gpu(id) => id,
        _ => {
            return Err(TensorError::invalid_operation_simple(
                "Not a GPU device".to_string(),
            ))
        }
    };

    let context = get_gpu_context(device_id)?;

    // Convert inputs to GPU buffers
    let gpu_a = GpuBuffer::<T>::from_slice(inputs[0].data(), &graph.device)?;
    let gpu_b = GpuBuffer::<T>::from_slice(inputs[1].data(), &graph.device)?;
    let gpu_c = GpuBuffer::<T>::from_slice(inputs[2].data(), &graph.device)?;

    let output_len = inputs[0].size();

    // Create output buffer
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fusion_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Encode operations into operation_mask
    let operation_mask = encode_fusion_operations(graph)?;

    // Create params uniform
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct FusionParams {
        batch_size: u32,
        channels: u32,
        height: u32,
        width: u32,
        operation_mask: u32,
        alpha: f32,
        beta: f32,
        gamma: f32,
    }

    let params = FusionParams {
        batch_size: 1,
        channels: 1,
        height: 1,
        width: output_len as u32,
        operation_mask,
        alpha: 1.0,
        beta: 0.0,
        gamma: 0.0,
    };

    let params_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fusion_params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Load shader
    let shader_source = include_str!("../gpu/shaders/fused_operations.wgsl");
    let shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("simple_fusion_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    // Create compute pipeline
    let pipeline = context
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("simple_fusion_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("simple_elementwise_fusion"),
            compilation_options: Default::default(),
            cache: None,
        });

    // Create bind group
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fusion_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu_b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu_c.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

    // Execute kernel
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fusion_encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fusion_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (output_len as u32 + 255) / 256;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }

    context.queue.submit(Some(encoder.finish()));

    // Read back result
    let result_buffer = GpuBuffer::<T>::from_wgpu_buffer(
        output_buffer,
        Arc::clone(&context.device),
        Arc::clone(&context.queue),
        graph.device.clone(),
        output_len,
    );

    let result_tensor = Tensor::from_gpu_buffer(result_buffer, inputs[0].shape().clone());

    Ok(vec![result_tensor])
}

/// Encode fusion graph operations into operation_mask
#[cfg(feature = "gpu")]
fn encode_fusion_operations(graph: &FusionGraph) -> Result<u32> {
    let mut mask = 0u32;

    // Simple pattern: find binary op -> binary op -> activation
    let compute_nodes: Vec<_> = graph.nodes.iter().filter(|n| !n.is_input).collect();

    if compute_nodes.is_empty() || compute_nodes.len() > 3 {
        return Err(TensorError::not_implemented_simple(
            "GPU fusion only supports 2-3 operation chains".to_string(),
        ));
    }

    // Encode first binary operation
    if compute_nodes.len() >= 1 {
        let op1_code = match compute_nodes[0].op_type {
            ElementwiseOpType::Add => 0,
            ElementwiseOpType::Mul => 1,
            ElementwiseOpType::Sub => 2,
            ElementwiseOpType::Div => 3,
            _ => {
                return Err(TensorError::not_implemented_simple(
                    "Unsupported op1".to_string(),
                ))
            }
        };
        mask |= op1_code;
    }

    // Encode second operation
    if compute_nodes.len() >= 2 {
        let op2_code = match compute_nodes[1].op_type {
            ElementwiseOpType::Add => 0,
            ElementwiseOpType::Mul => 1,
            ElementwiseOpType::Sub => 2,
            ElementwiseOpType::Div => 3,
            _ => {
                return Err(TensorError::not_implemented_simple(
                    "Unsupported op2".to_string(),
                ))
            }
        };
        mask |= op2_code << 4;
    } else {
        mask |= 15 << 4; // Skip second operation
    }

    // Encode activation
    if compute_nodes.len() >= 3 {
        let act_code = match compute_nodes[2].op_type {
            ElementwiseOpType::ReLU => 1,
            ElementwiseOpType::Tanh => 2,
            ElementwiseOpType::Sigmoid => 3,
            ElementwiseOpType::GELU => 4,
            _ => 0, // No activation
        };
        mask |= act_code << 8;
    }

    Ok(mask)
}

/// Statistics about fusion opportunities
#[derive(Debug, Clone, Default)]
pub struct FusionStats {
    /// Number of fusion opportunities identified
    pub opportunities_identified: usize,
    /// Number of fusions applied
    pub fusions_applied: usize,
    /// Total memory saved (bytes)
    pub memory_saved: usize,
    /// Average fusion chain length
    pub avg_chain_length: f64,
    /// Longest fusion chain
    pub max_chain_length: usize,
}

/// Global fusion statistics
lazy_static::lazy_static! {
    static ref GLOBAL_FUSION_STATS: Arc<Mutex<FusionStats>> = {
        Arc::new(Mutex::new(FusionStats::default()))
    };
}

/// Record a fusion opportunity
pub fn record_fusion_opportunity(graph: &FusionGraph) {
    let mut stats = GLOBAL_FUSION_STATS.lock().unwrap();
    stats.opportunities_identified += 1;
    if graph.is_fusible() {
        stats.fusions_applied += 1;
        stats.memory_saved += graph.estimated_memory_savings();
        let chain_length = graph.nodes.len();
        stats.max_chain_length = stats.max_chain_length.max(chain_length);

        // Update average
        let total = stats.fusions_applied as f64;
        stats.avg_chain_length =
            (stats.avg_chain_length * (total - 1.0) + chain_length as f64) / total;
    }
}

/// Get current fusion statistics
pub fn get_fusion_stats() -> FusionStats {
    GLOBAL_FUSION_STATS.lock().unwrap().clone()
}

/// Reset fusion statistics
pub fn reset_fusion_stats() {
    *GLOBAL_FUSION_STATS.lock().unwrap() = FusionStats::default();
}

/// Print fusion statistics report
pub fn print_fusion_report() {
    let stats = get_fusion_stats();
    println!("=== Operation Fusion Report ===");
    println!("Fusion Opportunities: {}", stats.opportunities_identified);
    println!("Fusions Applied:      {}", stats.fusions_applied);
    println!(
        "Memory Saved:         {:.2} MB",
        stats.memory_saved as f64 / 1_048_576.0
    );
    println!("Avg Chain Length:     {:.2}", stats.avg_chain_length);
    println!("Max Chain Length:     {}", stats.max_chain_length);
    println!("===============================");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_graph_creation() {
        let mut graph = FusionGraph::new(Device::Cpu);
        let shape = Shape::from_slice(&[10, 10]);

        let input1 = graph.add_input(shape.clone(), DType::Float32);
        let input2 = graph.add_input(shape.clone(), DType::Float32);
        let add_node = graph.add_op(
            ElementwiseOpType::Add,
            vec![input1, input2],
            shape.clone(),
            DType::Float32,
        );
        let relu_node = graph.add_op(
            ElementwiseOpType::ReLU,
            vec![add_node],
            shape.clone(),
            DType::Float32,
        );

        graph.mark_output(relu_node);

        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.nodes.len(), 4); // 2 inputs + add + relu
        assert!(graph.is_fusible());
    }

    #[test]
    fn test_fusion_memory_savings() {
        let mut graph = FusionGraph::new(Device::Cpu);
        let shape = Shape::from_slice(&[1000, 1000]); // 1M elements

        let input = graph.add_input(shape.clone(), DType::Float32);
        let op1 = graph.add_op(
            ElementwiseOpType::ReLU,
            vec![input],
            shape.clone(),
            DType::Float32,
        );
        let op2 = graph.add_op(
            ElementwiseOpType::Tanh,
            vec![op1],
            shape.clone(),
            DType::Float32,
        );
        graph.mark_output(op2);

        // Should save memory for op1's output (1M elements * 4 bytes)
        let savings = graph.estimated_memory_savings();
        assert_eq!(savings, 1_000_000 * 4); // 4MB
    }

    #[test]
    fn test_elementwise_op_classification() {
        assert!(ElementwiseOpType::ReLU.is_unary());
        assert!(!ElementwiseOpType::ReLU.is_binary());
        assert!(ElementwiseOpType::Add.is_binary());
        assert!(!ElementwiseOpType::Add.is_unary());
    }

    #[test]
    fn test_fusion_builder() {
        use scirs2_core::ndarray::array;

        // Create test tensors
        let a = Tensor::from_array(array![1.0, 2.0, 3.0, 4.0].into_dyn());
        let b = Tensor::from_array(array![1.0, 1.0, 1.0, 1.0].into_dyn());
        let c = Tensor::from_array(array![2.0, 2.0, 2.0, 2.0].into_dyn());

        // Build fusion graph
        let mut graph = FusionGraph::new(Device::Cpu);
        let shape = Shape::from_slice(&[4]);

        let i1 = graph.add_input(shape.clone(), DType::Float32);
        let i2 = graph.add_input(shape.clone(), DType::Float32);
        let i3 = graph.add_input(shape.clone(), DType::Float32);

        // (a + b) * c
        let add_node = graph.add_op(
            ElementwiseOpType::Add,
            vec![i1, i2],
            shape.clone(),
            DType::Float32,
        );
        let mul_node = graph.add_op(
            ElementwiseOpType::Mul,
            vec![add_node, i3],
            shape.clone(),
            DType::Float32,
        );

        graph.mark_output(mul_node);

        // Verify graph structure
        assert!(graph.is_fusible());
        assert_eq!(graph.inputs.len(), 3);
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.nodes.len(), 5); // 3 inputs + add + mul

        // Test execution (should fallback to CPU)
        let inputs = [&a, &b, &c];
        let result = execute_fused_graph(&graph, &inputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fusion_stats() {
        reset_fusion_stats();

        let graph = FusionGraph::new(Device::Cpu);
        record_fusion_opportunity(&graph);

        let stats = get_fusion_stats();
        assert_eq!(stats.opportunities_identified, 1);
    }

    #[test]
    fn test_encode_fusion_operations() {
        #[cfg(feature = "gpu")]
        {
            let mut graph = FusionGraph::new(Device::Gpu(0));
            let shape = Shape::from_slice(&[10]);

            let i1 = graph.add_input(shape.clone(), DType::Float32);
            let i2 = graph.add_input(shape.clone(), DType::Float32);
            let i3 = graph.add_input(shape.clone(), DType::Float32);

            // (a + b) * c with ReLU
            let add_node = graph.add_op(
                ElementwiseOpType::Add,
                vec![i1, i2],
                shape.clone(),
                DType::Float32,
            );
            let mul_node = graph.add_op(
                ElementwiseOpType::Mul,
                vec![add_node, i3],
                shape.clone(),
                DType::Float32,
            );
            let relu_node = graph.add_op(
                ElementwiseOpType::ReLU,
                vec![mul_node],
                shape.clone(),
                DType::Float32,
            );
            graph.mark_output(relu_node);

            let mask = encode_fusion_operations(&graph).unwrap();

            // Check encoding: Add (0) + Mul (1) + ReLU (1)
            assert_eq!(mask & 0xF, 0); // Add
            assert_eq!((mask >> 4) & 0xF, 1); // Mul
            assert_eq!((mask >> 8) & 0xF, 1); // ReLU
        }
    }
}
