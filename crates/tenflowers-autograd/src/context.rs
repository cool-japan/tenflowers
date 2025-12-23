use scirs2_autograd as ag;
use scirs2_autograd::prelude::AsGraph;
use scirs2_autograd::tensor_ops as ag_ops;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

/// AutogradContext provides a wrapper for SciRS2-autograd integration with TenfloweRS
///
/// Note: Due to lifetime constraints in SciRS2-autograd's API, this context
/// does not store the underlying ag::Context directly. Instead, it provides
/// methods to work with SciRS2-autograd within the ag::run() closure.
pub struct AutogradContext<T: ag::Float = f32> {
    /// Map from variable names to tensor names (for tracking)
    variable_names: HashMap<String, String>,
    /// Map from placeholder names to their shapes  
    placeholder_shapes: HashMap<String, Vec<usize>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: ag::Float + Default> AutogradContext<T> {
    pub fn new() -> Self {
        Self {
            variable_names: HashMap::new(),
            placeholder_shapes: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Run a computation within an autograd context
    ///
    /// This is a wrapper around ag::run that provides a safer interface
    pub fn run<F, R>(f: F) -> R
    where
        F: FnOnce(&mut ag::Context<T>) -> R,
    {
        ag::run(f)
    }

    /// Register a placeholder shape for later use
    pub fn register_placeholder(&mut self, name: &str, shape: &[usize]) {
        self.placeholder_shapes
            .insert(name.to_string(), shape.to_vec());
    }

    /// Register a variable name for later use
    pub fn register_variable(&mut self, name: &str) {
        self.variable_names
            .insert(name.to_string(), name.to_string());
    }

    /// Convert TenFlowers tensor to SciRS2-autograd tensor
    ///
    /// This function must be called within an autograd context (ag::run)
    /// Only CPU tensors are supported for now.
    pub fn from_tenflowers<'a, G: AsGraph<T>>(
        tensor: &Tensor<T>,
        ctx: &'a mut G,
    ) -> Result<ag::Tensor<'a, T>> {
        use tenflowers_core::tensor::TensorStorage;

        #[allow(unreachable_patterns)] // GPU pattern unreachable when gpu feature is disabled
        match &tensor.storage {
            TensorStorage::Cpu(array) => {
                // Convert to ag::Tensor using the context
                let ag_tensor = ag_ops::convert_to_tensor(array.clone(), ctx);
                Ok(ag_tensor)
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => Err(TensorError::unsupported_operation_simple(
                "GPU tensors not yet supported for autograd conversion".to_string(),
            )),
            #[cfg(not(feature = "gpu"))]
            _ => unreachable!("GPU variant should not exist without gpu feature"),
        }
    }

    /// Convert SciRS2-autograd tensor to TenFlowers tensor
    ///
    /// This evaluates the ag::Tensor to get concrete data.
    /// The resulting tensor will be a CPU tensor.
    /// Note: This requires specifically a Context, not just AsGraph
    pub fn to_tenflowers<'a>(
        ag_tensor: &ag::Tensor<'a, T>,
        ctx: &ag::Context<'a, T>,
    ) -> Result<Tensor<T>> {
        // Evaluate the tensor to get concrete data
        let array = ag_tensor.eval(ctx).map_err(|e| {
            TensorError::compute_error_simple(format!("Failed to evaluate ag::Tensor: {e:?}"))
        })?;

        // Create TenFlowers tensor from the resulting array
        Ok(Tensor::from_array(array))
    }

    /// Get registered placeholder shapes
    pub fn get_placeholder_shape(&self, name: &str) -> Option<&[usize]> {
        self.placeholder_shapes.get(name).map(|v| v.as_slice())
    }

    /// Check if a variable name is registered
    pub fn has_variable(&self, name: &str) -> bool {
        self.variable_names.contains_key(name)
    }
}

/// Static Shape Inference for SciRS2-Autograd Integration
///
/// Provides compile-time shape inference for operations to enable
/// better optimization in the SciRS2-autograd computation graph.
pub struct StaticShapeInference<T: ag::Float> {
    /// Map from tensor node IDs to their static shapes
    tensor_shapes: HashMap<u64, Vec<Option<usize>>>,
    /// Map from operation types to their shape inference rules
    shape_rules: HashMap<String, ShapeInferenceRule>,
    _phantom: std::marker::PhantomData<T>,
}

/// Shape inference rule for operations
pub enum ShapeInferenceRule {
    /// Element-wise operations preserve input shape
    ElementWise,
    /// Matrix multiplication: (m, k) @ (k, n) -> (m, n)
    MatMul,
    /// Reshape operation: input -> specified shape
    Reshape(Vec<Option<usize>>),
    /// Reduction along specified axes
    Reduction { axes: Vec<i32>, keep_dims: bool },
    /// Broadcasting rules
    Broadcast,
    /// Concatenation along an axis
    Concat { axis: i32 },
    /// Convolution operations
    Conv2D {
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize, usize, usize),
    },
    /// Custom shape inference function
    #[allow(clippy::type_complexity)]
    Custom(fn(&[Vec<Option<usize>>]) -> Result<Vec<Option<usize>>>),
}

impl<T: ag::Float + Default> StaticShapeInference<T> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let mut inference = Self {
            tensor_shapes: HashMap::new(),
            shape_rules: HashMap::new(),
            _phantom: std::marker::PhantomData,
        };

        // Register built-in shape inference rules
        inference.register_built_in_rules();
        inference
    }

    /// Register built-in shape inference rules for common operations
    fn register_built_in_rules(&mut self) {
        // Element-wise operations
        for op in &[
            "add", "sub", "mul", "div", "pow", "relu", "sigmoid", "tanh", "gelu",
        ] {
            self.shape_rules
                .insert((*op).to_string(), ShapeInferenceRule::ElementWise);
        }

        // Matrix multiplication
        self.shape_rules
            .insert("matmul".to_string(), ShapeInferenceRule::MatMul);

        // Broadcasting operations
        self.shape_rules
            .insert("broadcast".to_string(), ShapeInferenceRule::Broadcast);

        // Reduction operations
        self.shape_rules.insert(
            "sum".to_string(),
            ShapeInferenceRule::Reduction {
                axes: vec![], // Will be set per operation
                keep_dims: false,
            },
        );
        self.shape_rules.insert(
            "mean".to_string(),
            ShapeInferenceRule::Reduction {
                axes: vec![],
                keep_dims: false,
            },
        );
    }

    /// Infer the output shape for an operation
    pub fn infer_shape(
        &mut self,
        operation: &str,
        input_shapes: &[Vec<Option<usize>>],
        parameters: Option<&HashMap<String, String>>,
    ) -> Result<Vec<Option<usize>>> {
        match self.shape_rules.get(operation) {
            Some(rule) => self.apply_shape_rule(rule, input_shapes, parameters),
            None => {
                // Unknown operation - try to infer from context
                self.fallback_shape_inference(operation, input_shapes)
            }
        }
    }

    /// Apply a specific shape inference rule
    fn apply_shape_rule(
        &self,
        rule: &ShapeInferenceRule,
        input_shapes: &[Vec<Option<usize>>],
        _parameters: Option<&HashMap<String, String>>,
    ) -> Result<Vec<Option<usize>>> {
        match rule {
            ShapeInferenceRule::ElementWise => {
                if input_shapes.is_empty() {
                    return Err(TensorError::invalid_argument(
                        "Element-wise operation requires at least one input".to_string(),
                    ));
                }

                // For element-wise operations, output shape is broadcast result of inputs
                self.broadcast_shapes(
                    &input_shapes[0],
                    input_shapes.get(1).unwrap_or(&input_shapes[0]),
                )
            }

            ShapeInferenceRule::MatMul => {
                if input_shapes.len() != 2 {
                    return Err(TensorError::invalid_argument(
                        "Matrix multiplication requires exactly 2 inputs".to_string(),
                    ));
                }

                self.infer_matmul_shape(&input_shapes[0], &input_shapes[1])
            }

            ShapeInferenceRule::Reduction { axes, keep_dims } => {
                if input_shapes.is_empty() {
                    return Err(TensorError::invalid_argument(
                        "Reduction operation requires at least one input".to_string(),
                    ));
                }

                self.infer_reduction_shape(&input_shapes[0], axes, *keep_dims)
            }

            ShapeInferenceRule::Reshape(target_shape) => Ok(target_shape.clone()),

            ShapeInferenceRule::Broadcast => {
                if input_shapes.len() != 2 {
                    return Err(TensorError::invalid_argument(
                        "Broadcast operation requires exactly 2 inputs".to_string(),
                    ));
                }

                self.broadcast_shapes(&input_shapes[0], &input_shapes[1])
            }

            ShapeInferenceRule::Concat { axis } => {
                if input_shapes.is_empty() {
                    return Err(TensorError::invalid_argument(
                        "Concat operation requires at least one input".to_string(),
                    ));
                }

                self.infer_concat_shape(input_shapes, *axis)
            }

            ShapeInferenceRule::Conv2D {
                kernel_size,
                stride,
                padding,
            } => {
                if input_shapes.len() < 2 {
                    return Err(TensorError::invalid_argument(
                        "Conv2D operation requires input and weight tensors".to_string(),
                    ));
                }

                self.infer_conv2d_shape(
                    &input_shapes[0],
                    &input_shapes[1],
                    *kernel_size,
                    *stride,
                    *padding,
                )
            }

            ShapeInferenceRule::Custom(inference_fn) => inference_fn(input_shapes),
        }
    }

    /// Broadcast two shapes together
    fn broadcast_shapes(
        &self,
        shape1: &[Option<usize>],
        shape2: &[Option<usize>],
    ) -> Result<Vec<Option<usize>>> {
        let len1 = shape1.len();
        let len2 = shape2.len();
        let max_len = len1.max(len2);
        let mut result = vec![None; max_len];

        for i in 0..max_len {
            let dim1 = if i < len1 {
                shape1[len1 - 1 - i]
            } else {
                Some(1)
            };
            let dim2 = if i < len2 {
                shape2[len2 - 1 - i]
            } else {
                Some(1)
            };

            result[max_len - 1 - i] = match (dim1, dim2) {
                (Some(1), d) | (d, Some(1)) => d,
                (Some(a), Some(b)) if a == b => Some(a),
                (None, d) | (d, None) => d,
                (Some(a), Some(b)) => {
                    return Err(TensorError::shape_mismatch(
                        "broadcast_shapes",
                        "broadcastable shapes",
                        &format!("shapes {shape1:?} and {shape2:?} with conflicting dimensions {a} and {b}")
                    ));
                }
            };
        }

        Ok(result)
    }

    /// Infer shape for matrix multiplication
    fn infer_matmul_shape(
        &self,
        shape1: &[Option<usize>],
        shape2: &[Option<usize>],
    ) -> Result<Vec<Option<usize>>> {
        if shape1.len() < 2 || shape2.len() < 2 {
            return Err(TensorError::shape_mismatch(
                "infer_matmul_shape",
                "at least 2D tensors for matrix multiplication",
                &format!("shapes {shape1:?} and {shape2:?}"),
            ));
        }

        let m = shape1[shape1.len() - 2];
        let k1 = shape1[shape1.len() - 1];
        let k2 = shape2[shape2.len() - 2];
        let n = shape2[shape2.len() - 1];

        // Check that inner dimensions match
        match (k1, k2) {
            (Some(a), Some(b)) if a != b => {
                return Err(TensorError::shape_mismatch(
                    "infer_matmul_shape",
                    "matching inner dimensions",
                    &format!("dimensions {a} and {b}"),
                ));
            }
            _ => {} // Either they match or at least one is unknown
        }

        // Handle batch dimensions by broadcasting
        let batch1 = &shape1[..shape1.len() - 2];
        let batch2 = &shape2[..shape2.len() - 2];
        let batch_dims = self.broadcast_shapes(batch1, batch2)?;

        let mut result = batch_dims;
        result.push(m);
        result.push(n);

        Ok(result)
    }

    /// Infer shape for reduction operations
    fn infer_reduction_shape(
        &self,
        input_shape: &[Option<usize>],
        axes: &[i32],
        keep_dims: bool,
    ) -> Result<Vec<Option<usize>>> {
        if axes.is_empty() {
            // Reduce all dimensions
            if keep_dims {
                Ok(vec![Some(1); input_shape.len()])
            } else {
                Ok(vec![])
            }
        } else {
            let mut result = input_shape.to_vec();

            // Sort axes in descending order to remove from back to front
            let mut sorted_axes = axes.to_vec();
            sorted_axes.sort_by(|a, b| b.cmp(a));

            for &axis in &sorted_axes {
                let positive_axis = if axis < 0 {
                    (input_shape.len() as i32 + axis) as usize
                } else {
                    axis as usize
                };

                if positive_axis >= input_shape.len() {
                    return Err(TensorError::invalid_argument(format!(
                        "Axis {} out of bounds for tensor with {} dimensions",
                        axis,
                        input_shape.len()
                    )));
                }

                if keep_dims {
                    result[positive_axis] = Some(1);
                } else {
                    result.remove(positive_axis);
                }
            }

            Ok(result)
        }
    }

    /// Infer shape for concatenation
    fn infer_concat_shape(
        &self,
        input_shapes: &[Vec<Option<usize>>],
        axis: i32,
    ) -> Result<Vec<Option<usize>>> {
        if input_shapes.is_empty() {
            return Err(TensorError::invalid_argument(
                "Concat requires at least one input".to_string(),
            ));
        }

        let first_shape = &input_shapes[0];
        let positive_axis = if axis < 0 {
            (first_shape.len() as i32 + axis) as usize
        } else {
            axis as usize
        };

        if positive_axis >= first_shape.len() {
            return Err(TensorError::invalid_argument(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                first_shape.len()
            )));
        }

        let mut result = first_shape.clone();
        let mut concat_dim_size = first_shape[positive_axis];

        // Check that all other dimensions match and accumulate concat dimension size
        for shape in input_shapes.iter().skip(1) {
            if shape.len() != first_shape.len() {
                return Err(TensorError::shape_mismatch(
                    "infer_concat_shape",
                    &format!("{} dimensions", first_shape.len()),
                    &format!("{} dimensions", shape.len()),
                ));
            }

            for (i, (&dim1, &dim2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if i == positive_axis {
                    // Accumulate size for concat dimension
                    concat_dim_size = match (concat_dim_size, dim2) {
                        (Some(a), Some(b)) => Some(a + b),
                        _ => None, // Unknown if either is unknown
                    };
                } else {
                    // All other dimensions must match
                    match (dim1, dim2) {
                        (Some(a), Some(b)) if a != b => {
                            return Err(TensorError::shape_mismatch(
                                "infer_concat_shape",
                                &format!("dimension {i} to be {a}"),
                                &format!("dimension {i} is {b}"),
                            ));
                        }
                        _ => {} // Either they match or at least one is unknown
                    }
                }
            }
        }

        result[positive_axis] = concat_dim_size;
        Ok(result)
    }

    /// Infer shape for 2D convolution
    fn infer_conv2d_shape(
        &self,
        input_shape: &[Option<usize>],
        weight_shape: &[Option<usize>],
        _kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize, usize, usize),
    ) -> Result<Vec<Option<usize>>> {
        if input_shape.len() != 4 || weight_shape.len() != 4 {
            return Err(TensorError::shape_mismatch(
                "infer_conv2d_shape",
                "4D tensors for Conv2D (NCHW format)",
                &format!("input: {input_shape:?}, weight: {weight_shape:?}"),
            ));
        }

        let batch_size = input_shape[0];
        let output_channels = weight_shape[0];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let output_height = match input_height {
            Some(h) => {
                let padded_h = h + padding.0 + padding.1;
                let kernel_h = weight_shape[2].unwrap_or(1);
                Some((padded_h - kernel_h) / stride.0 + 1)
            }
            None => None,
        };

        let output_width = match input_width {
            Some(w) => {
                let padded_w = w + padding.2 + padding.3;
                let kernel_w = weight_shape[3].unwrap_or(1);
                Some((padded_w - kernel_w) / stride.1 + 1)
            }
            None => None,
        };

        Ok(vec![
            batch_size,
            output_channels,
            output_height,
            output_width,
        ])
    }

    /// Fallback shape inference for unknown operations
    fn fallback_shape_inference(
        &self,
        _operation: &str,
        input_shapes: &[Vec<Option<usize>>],
    ) -> Result<Vec<Option<usize>>> {
        if input_shapes.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot infer shape for operation with no inputs".to_string(),
            ));
        }

        // Default: assume output shape matches first input shape
        Ok(input_shapes[0].clone())
    }

    /// Register the shape of a tensor
    pub fn register_tensor_shape(&mut self, tensor_id: u64, shape: Vec<Option<usize>>) {
        self.tensor_shapes.insert(tensor_id, shape);
    }

    /// Get the registered shape of a tensor
    pub fn get_tensor_shape(&self, tensor_id: u64) -> Option<&[Option<usize>]> {
        self.tensor_shapes.get(&tensor_id).map(|v| v.as_slice())
    }

    /// Register a custom shape inference rule
    pub fn register_shape_rule(&mut self, operation: String, rule: ShapeInferenceRule) {
        self.shape_rules.insert(operation, rule);
    }
}

impl<T: ag::Float + Default> Default for AutogradContext<T> {
    fn default() -> Self {
        Self::new()
    }
}
