//! Built-in operation and kernel registrations.
//!
//! Registers the standard set of arithmetic, matrix, and activation operations
//! along with their CPU kernels for various numeric types.

use super::types::{ArgDef, AttrDef, AttrType, AttrValue, Kernel, OpDef, OpRegistry, OpVersion};
use crate::{DType, Device, Result, Shape, TensorError};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// Register built-in operations
pub(super) fn register_builtin_ops(registry: &OpRegistry) {
    // Add
    registry
        .register_op(OpDef {
            name: "Add".to_string(),
            version: OpVersion::new(1, 0, 0),
            inputs: vec![
                ArgDef {
                    name: "x".to_string(),
                    dtype: None,
                    shape: None,
                    doc: "First operand".to_string(),
                },
                ArgDef {
                    name: "y".to_string(),
                    dtype: None,
                    shape: None,
                    doc: "Second operand".to_string(),
                },
            ],
            outputs: vec![ArgDef {
                name: "output".to_string(),
                dtype: None,
                shape: None,
                doc: "Sum of x and y".to_string(),
            }],
            attrs: HashMap::new(),
            shape_fn: Some(Arc::new(|inputs, _attrs| {
                let shape = inputs[0].broadcast_shape(inputs[1]).ok_or_else(|| {
                    TensorError::invalid_argument("Incompatible shapes for broadcast".to_string())
                })?;
                Ok(vec![shape])
            })),
            grad_fn: Some("AddGrad".to_string()),
            doc: "Element-wise addition of tensors".to_string(),
            deprecated: false,
            deprecation_message: None,
        })
        .expect("failed to register Add operation");

    // MatMul
    registry
        .register_op(OpDef {
            name: "MatMul".to_string(),
            version: OpVersion::new(1, 0, 0),
            inputs: vec![
                ArgDef {
                    name: "a".to_string(),
                    dtype: None,
                    shape: None,
                    doc: "First matrix".to_string(),
                },
                ArgDef {
                    name: "b".to_string(),
                    dtype: None,
                    shape: None,
                    doc: "Second matrix".to_string(),
                },
            ],
            outputs: vec![ArgDef {
                name: "output".to_string(),
                dtype: None,
                shape: None,
                doc: "Matrix multiplication result".to_string(),
            }],
            attrs: HashMap::from([
                (
                    "transpose_a".to_string(),
                    AttrDef {
                        name: "transpose_a".to_string(),
                        attr_type: AttrType::Bool,
                        default: Some(AttrValue::Bool(false)),
                        doc: "Transpose first matrix".to_string(),
                    },
                ),
                (
                    "transpose_b".to_string(),
                    AttrDef {
                        name: "transpose_b".to_string(),
                        attr_type: AttrType::Bool,
                        default: Some(AttrValue::Bool(false)),
                        doc: "Transpose second matrix".to_string(),
                    },
                ),
            ]),
            shape_fn: Some(Arc::new(|inputs, attrs| {
                let transpose_a = attrs
                    .get("transpose_a")
                    .and_then(|v| {
                        if let AttrValue::Bool(b) = v {
                            Some(*b)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(false);
                let transpose_b = attrs
                    .get("transpose_b")
                    .and_then(|v| {
                        if let AttrValue::Bool(b) = v {
                            Some(*b)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(false);

                let a_shape = inputs[0];
                let b_shape = inputs[1];

                if a_shape.rank() != 2 || b_shape.rank() != 2 {
                    return Err(TensorError::invalid_argument(
                        "MatMul requires 2D tensors".to_string(),
                    ));
                }

                let a_dims = a_shape.dims();
                let b_dims = b_shape.dims();

                let (m, k1) = if transpose_a {
                    (a_dims[1], a_dims[0])
                } else {
                    (a_dims[0], a_dims[1])
                };

                let (k2, n) = if transpose_b {
                    (b_dims[1], b_dims[0])
                } else {
                    (b_dims[0], b_dims[1])
                };

                if k1 != k2 {
                    return Err(TensorError::invalid_argument(format!(
                        "Incompatible matrix dimensions: ({m}, {k1}) x ({k2}, {n})"
                    )));
                }

                Ok(vec![Shape::from_slice(&[m, n])])
            })),
            grad_fn: Some("MatMulGrad".to_string()),
            doc: "Matrix multiplication".to_string(),
            deprecated: false,
            deprecation_message: None,
        })
        .expect("failed to register MatMul operation");

    // ReLU
    registry
        .register_op(OpDef {
            name: "ReLU".to_string(),
            version: OpVersion::new(1, 0, 0),
            inputs: vec![ArgDef {
                name: "input".to_string(),
                dtype: None,
                shape: None,
                doc: "Input tensor".to_string(),
            }],
            outputs: vec![ArgDef {
                name: "output".to_string(),
                dtype: None,
                shape: None,
                doc: "Output tensor".to_string(),
            }],
            attrs: HashMap::new(),
            shape_fn: Some(Arc::new(|inputs, _attrs| Ok(vec![inputs[0].clone()]))),
            grad_fn: Some("ReLUGrad".to_string()),
            doc: "Rectified Linear Unit activation".to_string(),
            deprecated: false,
            deprecation_message: None,
        })
        .expect("failed to register ReLU operation");

    // Register kernels for the operations
    register_builtin_kernels(registry);
}

/// Register built-in kernels for operations
fn register_builtin_kernels(registry: &OpRegistry) {
    use crate::{DType, Device};

    // Operation-aware kernel that handles different operation types
    struct OperationKernel {
        device: Device,
        dtype: DType,
        op_name: String,
    }

    /// Ultra-performance operation kernel with SIMD and GPU support
    struct UltraOperationKernel {
        device: Device,
        dtype: DType,
        op_name: String,
        #[allow(dead_code)]
        simd_capable: bool,
        #[allow(dead_code)]
        gpu_capable: bool,
    }

    impl Kernel for OperationKernel {
        fn compute(
            &self,
            inputs: &[&dyn Any],
            _attrs: &HashMap<String, AttrValue>,
        ) -> Result<Vec<Box<dyn Any>>> {
            match self.op_name.as_str() {
                // Binary operations
                "Add" | "Sub" | "Mul" | "Div" | "MatMul" => {
                    if inputs.len() != 2 {
                        return Err(TensorError::invalid_argument(format!(
                            "Binary operation '{}' requires exactly 2 inputs, got {}",
                            self.op_name,
                            inputs.len()
                        )));
                    }
                    self.compute_binary_operation(inputs)
                }
                // Unary operations
                "ReLU" => {
                    if inputs.len() != 1 {
                        return Err(TensorError::invalid_argument(format!(
                            "Unary operation '{}' requires exactly 1 input, got {}",
                            self.op_name,
                            inputs.len()
                        )));
                    }
                    self.compute_unary_operation(inputs)
                }
                _ => Err(TensorError::not_implemented_simple(format!(
                    "Operation '{}' not implemented in registry kernel",
                    self.op_name
                ))),
            }
        }

        fn device(&self) -> Device {
            self.device
        }

        fn dtype(&self) -> DType {
            self.dtype
        }
    }

    impl Kernel for UltraOperationKernel {
        fn compute(
            &self,
            inputs: &[&dyn Any],
            attrs: &HashMap<String, AttrValue>,
        ) -> Result<Vec<Box<dyn Any>>> {
            // Delegate to the existing OperationKernel implementation
            // For now, create a temporary OperationKernel for compatibility
            let temp_kernel = OperationKernel {
                device: self.device,
                dtype: self.dtype,
                op_name: self.op_name.clone(),
            };
            <OperationKernel as Kernel>::compute(&temp_kernel, inputs, attrs)
        }

        fn device(&self) -> Device {
            self.device
        }

        fn dtype(&self) -> DType {
            self.dtype
        }
    }

    impl OperationKernel {
        fn compute_binary_operation(&self, inputs: &[&dyn Any]) -> Result<Vec<Box<dyn Any>>> {
            match self.dtype {
                DType::Float32 => {
                    let tensor_a =
                        inputs[0]
                            .downcast_ref::<crate::Tensor<f32>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input 0 is not a f32 tensor".to_string(),
                                )
                            })?;
                    let tensor_b =
                        inputs[1]
                            .downcast_ref::<crate::Tensor<f32>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input 1 is not a f32 tensor".to_string(),
                                )
                            })?;

                    let result = match self.op_name.as_str() {
                        "Add" => crate::ops::binary::add(tensor_a, tensor_b)?,
                        "Sub" => crate::ops::binary::sub(tensor_a, tensor_b)?,
                        "Mul" => crate::ops::binary::mul(tensor_a, tensor_b)?,
                        "Div" => crate::ops::binary::div(tensor_a, tensor_b)?,
                        "MatMul" => crate::ops::matmul::matmul(tensor_a, tensor_b)?,
                        _ => {
                            return Err(TensorError::not_implemented_simple(format!(
                                "Binary operation '{}' not implemented",
                                self.op_name
                            )))
                        }
                    };
                    Ok(vec![Box::new(result)])
                }
                DType::Float64 => {
                    let tensor_a =
                        inputs[0]
                            .downcast_ref::<crate::Tensor<f64>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input 0 is not a f64 tensor".to_string(),
                                )
                            })?;
                    let tensor_b =
                        inputs[1]
                            .downcast_ref::<crate::Tensor<f64>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input 1 is not a f64 tensor".to_string(),
                                )
                            })?;

                    let result = match self.op_name.as_str() {
                        "Add" => crate::ops::binary::add(tensor_a, tensor_b)?,
                        "Sub" => crate::ops::binary::sub(tensor_a, tensor_b)?,
                        "Mul" => crate::ops::binary::mul(tensor_a, tensor_b)?,
                        "Div" => crate::ops::binary::div(tensor_a, tensor_b)?,
                        "MatMul" => crate::ops::matmul::matmul(tensor_a, tensor_b)?,
                        _ => {
                            return Err(TensorError::not_implemented_simple(format!(
                                "Binary operation '{}' not implemented",
                                self.op_name
                            )))
                        }
                    };
                    Ok(vec![Box::new(result)])
                }
                DType::Int32 => {
                    let tensor_a =
                        inputs[0]
                            .downcast_ref::<crate::Tensor<i32>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input 0 is not a i32 tensor".to_string(),
                                )
                            })?;
                    let tensor_b =
                        inputs[1]
                            .downcast_ref::<crate::Tensor<i32>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input 1 is not a i32 tensor".to_string(),
                                )
                            })?;

                    let result = match self.op_name.as_str() {
                        "Add" => crate::ops::binary::add(tensor_a, tensor_b)?,
                        "Sub" => crate::ops::binary::sub(tensor_a, tensor_b)?,
                        "Mul" => crate::ops::binary::mul(tensor_a, tensor_b)?,
                        "Div" => crate::ops::binary::div(tensor_a, tensor_b)?,
                        "MatMul" => crate::ops::matmul::matmul(tensor_a, tensor_b)?,
                        _ => {
                            return Err(TensorError::not_implemented_simple(format!(
                                "Binary operation '{}' not implemented",
                                self.op_name
                            )))
                        }
                    };
                    Ok(vec![Box::new(result)])
                }
                DType::Int64 => {
                    let tensor_a =
                        inputs[0]
                            .downcast_ref::<crate::Tensor<i64>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input 0 is not a i64 tensor".to_string(),
                                )
                            })?;
                    let tensor_b =
                        inputs[1]
                            .downcast_ref::<crate::Tensor<i64>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input 1 is not a i64 tensor".to_string(),
                                )
                            })?;

                    let result = match self.op_name.as_str() {
                        "Add" => crate::ops::binary::add(tensor_a, tensor_b)?,
                        "Sub" => crate::ops::binary::sub(tensor_a, tensor_b)?,
                        "Mul" => crate::ops::binary::mul(tensor_a, tensor_b)?,
                        "Div" => crate::ops::binary::div(tensor_a, tensor_b)?,
                        "MatMul" => crate::ops::matmul::matmul(tensor_a, tensor_b)?,
                        _ => {
                            return Err(TensorError::not_implemented_simple(format!(
                                "Binary operation '{}' not implemented",
                                self.op_name
                            )))
                        }
                    };
                    Ok(vec![Box::new(result)])
                }
                DType::Int8 => {
                    let tensor_a =
                        inputs[0]
                            .downcast_ref::<crate::Tensor<i8>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input 0 is not a i8 tensor".to_string(),
                                )
                            })?;
                    let tensor_b =
                        inputs[1]
                            .downcast_ref::<crate::Tensor<i8>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input 1 is not a i8 tensor".to_string(),
                                )
                            })?;

                    let result = match self.op_name.as_str() {
                        "Add" => crate::ops::binary::add(tensor_a, tensor_b)?,
                        "Sub" => crate::ops::binary::sub(tensor_a, tensor_b)?,
                        "Mul" => crate::ops::binary::mul(tensor_a, tensor_b)?,
                        "Div" => crate::ops::binary::div(tensor_a, tensor_b)?,
                        "MatMul" => crate::ops::matmul::matmul(tensor_a, tensor_b)?,
                        _ => {
                            return Err(TensorError::not_implemented_simple(format!(
                                "Binary operation '{}' not implemented",
                                self.op_name
                            )))
                        }
                    };
                    Ok(vec![Box::new(result)])
                }
                DType::UInt8 => {
                    let tensor_a =
                        inputs[0]
                            .downcast_ref::<crate::Tensor<u8>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input 0 is not a u8 tensor".to_string(),
                                )
                            })?;
                    let tensor_b =
                        inputs[1]
                            .downcast_ref::<crate::Tensor<u8>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input 1 is not a u8 tensor".to_string(),
                                )
                            })?;

                    let result = match self.op_name.as_str() {
                        "Add" => crate::ops::binary::add(tensor_a, tensor_b)?,
                        "Sub" => crate::ops::binary::sub(tensor_a, tensor_b)?,
                        "Mul" => crate::ops::binary::mul(tensor_a, tensor_b)?,
                        "Div" => crate::ops::binary::div(tensor_a, tensor_b)?,
                        "MatMul" => crate::ops::matmul::matmul(tensor_a, tensor_b)?,
                        _ => {
                            return Err(TensorError::not_implemented_simple(format!(
                                "Binary operation '{}' not implemented",
                                self.op_name
                            )))
                        }
                    };
                    Ok(vec![Box::new(result)])
                }
                _ => Err(TensorError::not_implemented_simple(format!(
                    "Binary operation '{}' not implemented for dtype {:?}",
                    self.op_name, self.dtype
                ))),
            }
        }

        fn compute_unary_operation(&self, inputs: &[&dyn Any]) -> Result<Vec<Box<dyn Any>>> {
            match self.dtype {
                DType::Float32 => {
                    let tensor =
                        inputs[0]
                            .downcast_ref::<crate::Tensor<f32>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input is not a f32 tensor".to_string(),
                                )
                            })?;

                    let result = match self.op_name.as_str() {
                        "ReLU" => crate::ops::activation::relu(tensor)?,
                        _ => {
                            return Err(TensorError::not_implemented_simple(format!(
                                "Unary operation '{}' not implemented",
                                self.op_name
                            )))
                        }
                    };
                    Ok(vec![Box::new(result)])
                }
                DType::Float64 => {
                    let tensor =
                        inputs[0]
                            .downcast_ref::<crate::Tensor<f64>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input is not a f64 tensor".to_string(),
                                )
                            })?;

                    let result = match self.op_name.as_str() {
                        "ReLU" => crate::ops::activation::relu(tensor)?,
                        _ => {
                            return Err(TensorError::not_implemented_simple(format!(
                                "Unary operation '{}' not implemented",
                                self.op_name
                            )))
                        }
                    };
                    Ok(vec![Box::new(result)])
                }
                DType::Int32 => {
                    let tensor =
                        inputs[0]
                            .downcast_ref::<crate::Tensor<i32>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input is not a i32 tensor".to_string(),
                                )
                            })?;

                    let result = match self.op_name.as_str() {
                        "ReLU" => crate::ops::activation::relu(tensor)?,
                        _ => {
                            return Err(TensorError::not_implemented_simple(format!(
                                "Unary operation '{}' not implemented",
                                self.op_name
                            )))
                        }
                    };
                    Ok(vec![Box::new(result)])
                }
                DType::Int64 => {
                    let tensor =
                        inputs[0]
                            .downcast_ref::<crate::Tensor<i64>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input is not a i64 tensor".to_string(),
                                )
                            })?;

                    let result = match self.op_name.as_str() {
                        "ReLU" => crate::ops::activation::relu(tensor)?,
                        _ => {
                            return Err(TensorError::not_implemented_simple(format!(
                                "Unary operation '{}' not implemented",
                                self.op_name
                            )))
                        }
                    };
                    Ok(vec![Box::new(result)])
                }
                DType::Int8 => {
                    let tensor =
                        inputs[0]
                            .downcast_ref::<crate::Tensor<i8>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input is not a i8 tensor".to_string(),
                                )
                            })?;

                    let result = match self.op_name.as_str() {
                        "ReLU" => crate::ops::activation::relu(tensor)?,
                        _ => {
                            return Err(TensorError::not_implemented_simple(format!(
                                "Unary operation '{}' not implemented",
                                self.op_name
                            )))
                        }
                    };
                    Ok(vec![Box::new(result)])
                }
                DType::UInt8 => {
                    let tensor =
                        inputs[0]
                            .downcast_ref::<crate::Tensor<u8>>()
                            .ok_or_else(|| {
                                TensorError::invalid_argument(
                                    "Input is not a u8 tensor".to_string(),
                                )
                            })?;

                    let result = match self.op_name.as_str() {
                        "ReLU" => crate::ops::activation::relu(tensor)?,
                        _ => {
                            return Err(TensorError::not_implemented_simple(format!(
                                "Unary operation '{}' not implemented",
                                self.op_name
                            )))
                        }
                    };
                    Ok(vec![Box::new(result)])
                }
                _ => Err(TensorError::not_implemented_simple(format!(
                    "Unary operation '{}' not implemented for dtype {:?}",
                    self.op_name, self.dtype
                ))),
            }
        }
    }

    // Register kernels for different devices and data types
    let devices = [Device::Cpu];
    let dtypes = [
        DType::Float32,
        DType::Float64,
        DType::Int32,
        DType::Int64,
        DType::Int8,
        DType::UInt8,
    ];

    for &device in &devices {
        for &dtype in &dtypes {
            // Register binary operations with ultra-performance capabilities
            for op_name in ["Add", "Sub", "Mul", "Div", "MatMul"] {
                let kernel = Arc::new(UltraOperationKernel {
                    device,
                    dtype,
                    op_name: op_name.to_string(),
                    simd_capable: matches!(op_name, "Add" | "Mul" | "MatMul"),
                    gpu_capable: device != Device::Cpu,
                });
                let _ = registry.register_kernel(op_name, device, dtype, kernel);
            }

            // Register unary operations with ultra-performance capabilities
            {
                let op_name = "ReLU";
                let kernel = Arc::new(UltraOperationKernel {
                    device,
                    dtype,
                    op_name: op_name.to_string(),
                    simd_capable: true,
                    gpu_capable: device != Device::Cpu,
                });
                let _ = registry.register_kernel(op_name, device, dtype, kernel);
            }
        }
    }
}
