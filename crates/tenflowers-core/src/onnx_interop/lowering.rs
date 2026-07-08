//! ONNX op-lowering: mapping parsed `OnnxNode`s to TenfloweRS's own
//! executable op representation.
//!
//! This module operates purely on already-in-memory `OnnxGraph`/`OnnxNode`/
//! `OnnxTensor` structs — no protobuf involved — so, unlike `proto` and
//! `convert`, it is **not** gated behind the `onnx` feature. Op-lowering
//! works even when a caller builds an `OnnxGraph` by hand (or from some
//! non-protobuf source), and is independently testable without the `onnx`
//! feature enabled.
//!
//! Nothing in the workspace calls `OnnxOpMapping`/`TenfloweRSOperation`
//! today (both traits are defined in `onnx_interop::types` but had zero
//! implementations before this module), so there was no established
//! execution convention to match. [`StandardOpMapping`] defines one; see its
//! `OnnxOpMapping` impl below for the full design contract.
//!
//! ## Supported ops
//!
//! `Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, MatMul, Reshape, Transpose,
//! Identity, Concat` (the required set), plus `Softmax, Flatten, Gemm`
//! (nice-to-haves). Every other `op_type` is a genuine, descriptive `Err`
//! from `map_operation` — never a silent `Identity`/default/zero-fill
//! substitute (see `StandardOpMapping::supported_ops` for the exact list).

use super::types::{
    OnnxAttribute, OnnxDataType, OnnxGraph, OnnxNode, OnnxOpMapping, OnnxTensor,
    TenfloweRSOperation,
};
use crate::{Result, Tensor, TensorError};
use std::collections::HashMap;

// ---------------------------------------------------------------------
// Raw-bytes -> Tensor<f32> decoding
// ---------------------------------------------------------------------

/// Reads `count` little-endian, `WIDTH`-byte values out of `raw`, converting
/// each with `convert`. Returns an error if `raw`'s length does not exactly
/// match `count * WIDTH` bytes.
fn read_le_values<const WIDTH: usize>(
    raw: &[u8],
    count: usize,
    convert: impl Fn([u8; WIDTH]) -> f32,
) -> Result<Vec<f32>> {
    let expected_bytes = count * WIDTH;
    if raw.len() != expected_bytes {
        return Err(TensorError::invalid_shape_simple(format!(
            "raw tensor data is {} bytes, expected {expected_bytes} ({count} elements x \
             {WIDTH} bytes each)",
            raw.len()
        )));
    }
    let mut out = Vec::with_capacity(count);
    for chunk in raw.chunks_exact(WIDTH) {
        let mut buf = [0u8; WIDTH];
        buf.copy_from_slice(chunk);
        out.push(convert(buf));
    }
    Ok(out)
}

/// Decodes an `OnnxTensor`'s raw little-endian byte payload into a
/// `Tensor<f32>`, widening/narrowing numeric types as needed based on
/// `data_type`. Returns a genuine `Err` for data types with no sensible
/// f32 interpretation (`Undefined`, `String`, `Complex64`, `Complex128`)
/// rather than silently reinterpreting their bytes as floats.
pub fn decode_onnx_tensor(tensor: &OnnxTensor) -> Result<Tensor<f32>> {
    if tensor.dims.iter().any(|&d| d < 0) {
        return Err(TensorError::invalid_shape_simple(format!(
            "ONNX tensor '{}' has a negative dimension in {:?}",
            tensor.name, tensor.dims
        )));
    }
    let dims: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
    let count: usize = dims.iter().product();

    let values = match tensor.data_type {
        OnnxDataType::Float => read_le_values::<4>(&tensor.data, count, f32::from_le_bytes)?,
        OnnxDataType::Double => {
            read_le_values::<8>(&tensor.data, count, |b| f64::from_le_bytes(b) as f32)?
        }
        OnnxDataType::Int32 => {
            read_le_values::<4>(&tensor.data, count, |b| i32::from_le_bytes(b) as f32)?
        }
        OnnxDataType::Int64 => {
            read_le_values::<8>(&tensor.data, count, |b| i64::from_le_bytes(b) as f32)?
        }
        OnnxDataType::Uint32 => {
            read_le_values::<4>(&tensor.data, count, |b| u32::from_le_bytes(b) as f32)?
        }
        OnnxDataType::Uint64 => {
            read_le_values::<8>(&tensor.data, count, |b| u64::from_le_bytes(b) as f32)?
        }
        OnnxDataType::Int16 => {
            read_le_values::<2>(&tensor.data, count, |b| i16::from_le_bytes(b) as f32)?
        }
        OnnxDataType::Uint16 => {
            read_le_values::<2>(&tensor.data, count, |b| u16::from_le_bytes(b) as f32)?
        }
        OnnxDataType::Int8 => read_le_values::<1>(&tensor.data, count, |b| b[0] as i8 as f32)?,
        OnnxDataType::Uint8 => read_le_values::<1>(&tensor.data, count, |b| b[0] as f32)?,
        OnnxDataType::Bool => {
            read_le_values::<1>(&tensor.data, count, |b| if b[0] != 0 { 1.0 } else { 0.0 })?
        }
        OnnxDataType::Float16 => read_le_values::<2>(&tensor.data, count, |b| {
            half::f16::from_le_bytes(b).to_f32()
        })?,
        OnnxDataType::BFloat16 => read_le_values::<2>(&tensor.data, count, |b| {
            half::bf16::from_le_bytes(b).to_f32()
        })?,
        OnnxDataType::Undefined
        | OnnxDataType::String
        | OnnxDataType::Complex64
        | OnnxDataType::Complex128 => {
            return Err(TensorError::unsupported_operation_simple(format!(
                "ONNX tensor '{}' has data_type {:?}, which has no well-defined f32 \
                 interpretation",
                tensor.name, tensor.data_type
            )));
        }
    };

    Tensor::from_vec(values, &dims)
}

// ---------------------------------------------------------------------
// Operand resolution (the lowering-time/execute-time baking convention)
// ---------------------------------------------------------------------

/// Where a `TenfloweRSOperation`'s operand comes from once lowering has
/// resolved it: a compile-time constant baked from a graph initializer, or
/// a genuine runtime value supplied positionally to `execute`. See the
/// `OnnxOpMapping for StandardOpMapping` doc comment below for the full
/// baking convention.
#[derive(Debug, Clone)]
enum OperandSource {
    /// A value known at lowering time (the node input named a key in the
    /// `initializers` map passed to `map_operation`).
    Constant(Tensor<f32>),
    /// A value only known at execution time; `execute`'s `inputs` slice
    /// contains it at this index.
    Runtime(usize),
}

/// Lowering-time resolution of one ONNX node input name: if it names a
/// graph initializer, bake it in as a `Constant`; otherwise it becomes the
/// next positional `Runtime` operand, in the same relative order input
/// names are encountered.
fn bake_operand(
    input_name: &str,
    initializers: &HashMap<String, Tensor<f32>>,
    next_runtime_index: &mut usize,
) -> OperandSource {
    match initializers.get(input_name) {
        Some(tensor) => OperandSource::Constant(tensor.clone()),
        None => {
            let index = *next_runtime_index;
            *next_runtime_index += 1;
            OperandSource::Runtime(index)
        }
    }
}

/// Execute-time resolution of an `OperandSource` against the runtime inputs
/// slice `execute` received.
fn resolve_operand<'a>(
    op_name: &str,
    source: &'a OperandSource,
    inputs: &'a [Tensor<f32>],
) -> Result<&'a Tensor<f32>> {
    match source {
        OperandSource::Constant(tensor) => Ok(tensor),
        OperandSource::Runtime(index) => inputs.get(*index).ok_or_else(|| {
            TensorError::invalid_argument_op(
                op_name,
                &format!(
                    "expected a runtime input at position {index} but only {} were provided",
                    inputs.len()
                ),
            )
        }),
    }
}

// ---------------------------------------------------------------------
// Attribute helpers — an attribute that is present but the wrong variant
// is a genuine error, never a silently-ignored mismatch.
// ---------------------------------------------------------------------

fn optional_int_attr(node: &OnnxNode, key: &str) -> Result<Option<i64>> {
    match node.attributes.get(key) {
        None => Ok(None),
        Some(OnnxAttribute::Int(v)) => Ok(Some(*v)),
        Some(other) => Err(TensorError::invalid_argument_op(
            &node.op_type,
            &format!("attribute '{key}' must be Int, got {other:?}"),
        )),
    }
}

fn required_int_attr(node: &OnnxNode, key: &str) -> Result<i64> {
    optional_int_attr(node, key)?.ok_or_else(|| {
        TensorError::invalid_argument_op(
            &node.op_type,
            &format!("missing required Int attribute '{key}'"),
        )
    })
}

fn optional_float_attr(node: &OnnxNode, key: &str) -> Result<Option<f32>> {
    match node.attributes.get(key) {
        None => Ok(None),
        Some(OnnxAttribute::Float(v)) => Ok(Some(*v)),
        Some(other) => Err(TensorError::invalid_argument_op(
            &node.op_type,
            &format!("attribute '{key}' must be Float, got {other:?}"),
        )),
    }
}

fn optional_ints_attr<'a>(node: &'a OnnxNode, key: &str) -> Result<Option<&'a [i64]>> {
    match node.attributes.get(key) {
        None => Ok(None),
        Some(OnnxAttribute::Ints(v)) => Ok(Some(v.as_slice())),
        Some(other) => Err(TensorError::invalid_argument_op(
            &node.op_type,
            &format!("attribute '{key}' must be Ints, got {other:?}"),
        )),
    }
}

// ---------------------------------------------------------------------
// Op implementations
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
}

struct BinaryOp {
    name: String,
    kind: BinaryKind,
    lhs: OperandSource,
    rhs: OperandSource,
}

impl TenfloweRSOperation for BinaryOp {
    fn execute(&self, inputs: &[Tensor<f32>]) -> Result<Vec<Tensor<f32>>> {
        let a = resolve_operand(&self.name, &self.lhs, inputs)?;
        let b = resolve_operand(&self.name, &self.rhs, inputs)?;
        let result = match self.kind {
            BinaryKind::Add => crate::ops::add(a, b),
            BinaryKind::Sub => crate::ops::sub(a, b),
            BinaryKind::Mul => crate::ops::mul(a, b),
            BinaryKind::Div => crate::ops::div(a, b),
            BinaryKind::MatMul => crate::ops::matmul(a, b),
        }?;
        Ok(vec![result])
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone, Copy)]
enum UnaryKind {
    Relu,
    Sigmoid,
    Tanh,
}

struct UnaryOp {
    name: String,
    kind: UnaryKind,
    input: OperandSource,
}

impl TenfloweRSOperation for UnaryOp {
    fn execute(&self, inputs: &[Tensor<f32>]) -> Result<Vec<Tensor<f32>>> {
        let x = resolve_operand(&self.name, &self.input, inputs)?;
        let result = match self.kind {
            UnaryKind::Relu => crate::ops::relu(x),
            UnaryKind::Sigmoid => crate::ops::sigmoid(x),
            UnaryKind::Tanh => crate::ops::tanh(x),
        }?;
        Ok(vec![result])
    }

    fn name(&self) -> &str {
        &self.name
    }
}

struct IdentityOp {
    name: String,
    input: OperandSource,
}

impl TenfloweRSOperation for IdentityOp {
    fn execute(&self, inputs: &[Tensor<f32>]) -> Result<Vec<Tensor<f32>>> {
        let x = resolve_operand(&self.name, &self.input, inputs)?;
        Ok(vec![x.clone()])
    }

    fn name(&self) -> &str {
        &self.name
    }
}

struct ReshapeOp {
    name: String,
    data: OperandSource,
    /// Target shape values decoded from the (initializer-resolved) second
    /// input, possibly containing a single `-1` "infer this dimension"
    /// marker per ONNX convention. Resolved against the actual runtime
    /// element count in `execute`, since the data operand's concrete shape
    /// is not known until then.
    target_shape: Vec<i64>,
}

impl TenfloweRSOperation for ReshapeOp {
    fn execute(&self, inputs: &[Tensor<f32>]) -> Result<Vec<Tensor<f32>>> {
        let data = resolve_operand(&self.name, &self.data, inputs)?;
        let total = data.shape().size();

        let neg_ones = self.target_shape.iter().filter(|&&d| d == -1).count();
        if neg_ones > 1 {
            return Err(TensorError::invalid_argument_op(
                "Reshape",
                "at most one dimension of the target shape may be -1",
            ));
        }

        let resolved: Vec<usize> = if neg_ones == 1 {
            let known_product: i64 = self.target_shape.iter().filter(|&&d| d != -1).product();
            if known_product <= 0 {
                return Err(TensorError::invalid_argument_op(
                    "Reshape",
                    "cannot infer a -1 dimension when the other dimensions multiply to 0 or \
                     are invalid",
                ));
            }
            if total as i64 % known_product != 0 {
                return Err(TensorError::invalid_shape(
                    "Reshape",
                    &format!("a total element count divisible by {known_product}"),
                    &format!("{total}"),
                ));
            }
            let inferred = total as i64 / known_product;
            self.target_shape
                .iter()
                .map(|&d| {
                    if d == -1 {
                        inferred as usize
                    } else {
                        d as usize
                    }
                })
                .collect()
        } else {
            let mut out = Vec::with_capacity(self.target_shape.len());
            for &d in &self.target_shape {
                if d < 0 {
                    return Err(TensorError::invalid_argument_op(
                        "Reshape",
                        &format!(
                            "target shape dimension {d} is negative and is not the -1 \
                             \"infer\" marker"
                        ),
                    ));
                }
                out.push(d as usize);
            }
            out
        };

        let result = crate::ops::reshape(data, &resolved)?;
        Ok(vec![result])
    }

    fn name(&self) -> &str {
        &self.name
    }
}

struct TransposeOp {
    name: String,
    input: OperandSource,
    /// Explicit axis permutation from the `perm` attribute; `None` means
    /// "reverse all axes", matching `ops::transpose`'s default.
    perm: Option<Vec<usize>>,
}

impl TenfloweRSOperation for TransposeOp {
    fn execute(&self, inputs: &[Tensor<f32>]) -> Result<Vec<Tensor<f32>>> {
        let x = resolve_operand(&self.name, &self.input, inputs)?;
        let result = crate::ops::manipulation::transpose_axes(x, self.perm.as_deref())?;
        Ok(vec![result])
    }

    fn name(&self) -> &str {
        &self.name
    }
}

struct ConcatOp {
    name: String,
    inputs: Vec<OperandSource>,
    axis: usize,
}

impl TenfloweRSOperation for ConcatOp {
    fn execute(&self, inputs: &[Tensor<f32>]) -> Result<Vec<Tensor<f32>>> {
        let resolved = self
            .inputs
            .iter()
            .map(|src| resolve_operand(&self.name, src, inputs))
            .collect::<Result<Vec<_>>>()?;
        let result = crate::ops::concat(&resolved, self.axis)?;
        Ok(vec![result])
    }

    fn name(&self) -> &str {
        &self.name
    }
}

struct SoftmaxOp {
    name: String,
    input: OperandSource,
    axis: i32,
}

impl TenfloweRSOperation for SoftmaxOp {
    fn execute(&self, inputs: &[Tensor<f32>]) -> Result<Vec<Tensor<f32>>> {
        let x = resolve_operand(&self.name, &self.input, inputs)?;
        let result = crate::ops::softmax(x, Some(self.axis))?;
        Ok(vec![result])
    }

    fn name(&self) -> &str {
        &self.name
    }
}

struct FlattenOp {
    name: String,
    input: OperandSource,
    axis: i64,
}

impl TenfloweRSOperation for FlattenOp {
    fn execute(&self, inputs: &[Tensor<f32>]) -> Result<Vec<Tensor<f32>>> {
        let x = resolve_operand(&self.name, &self.input, inputs)?;
        let dims = x.shape().dims();
        let rank = dims.len() as i64;
        let axis = if self.axis < 0 {
            self.axis + rank
        } else {
            self.axis
        };
        if axis < 0 || axis > rank {
            return Err(TensorError::invalid_argument_op(
                "Flatten",
                &format!("axis {} out of range for tensor of rank {rank}", self.axis),
            ));
        }
        let axis = axis as usize;
        let d0: usize = dims[..axis].iter().product();
        let d1: usize = dims[axis..].iter().product();
        let result = crate::ops::reshape(x, &[d0, d1])?;
        Ok(vec![result])
    }

    fn name(&self) -> &str {
        &self.name
    }
}

struct GemmOp {
    name: String,
    a: OperandSource,
    b: OperandSource,
    c: Option<OperandSource>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

impl TenfloweRSOperation for GemmOp {
    fn execute(&self, inputs: &[Tensor<f32>]) -> Result<Vec<Tensor<f32>>> {
        let a = resolve_operand(&self.name, &self.a, inputs)?;
        let b = resolve_operand(&self.name, &self.b, inputs)?;

        let a_owned = if self.trans_a {
            crate::ops::transpose(a)?
        } else {
            a.clone()
        };
        let b_owned = if self.trans_b {
            crate::ops::transpose(b)?
        } else {
            b.clone()
        };

        // Multiplying by 1.0 is an exact no-op in IEEE 754, so alpha/beta
        // are always applied unconditionally rather than guarded by a
        // float comparison against 1.0.
        let mut y = crate::ops::matmul(&a_owned, &b_owned)?;
        y = y.scalar_mul(self.alpha)?;

        if let Some(c_src) = &self.c {
            let c = resolve_operand(&self.name, c_src, inputs)?;
            let c_scaled = c.scalar_mul(self.beta)?;
            y = crate::ops::add(&y, &c_scaled)?;
        }

        Ok(vec![y])
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------
// StandardOpMapping
// ---------------------------------------------------------------------

/// The default, "vanilla" ONNX-to-TenfloweRS op-lowering mapping.
pub struct StandardOpMapping;

impl OnnxOpMapping for StandardOpMapping {
    /// # Design contract
    ///
    /// Nothing in the workspace calls `OnnxOpMapping`/`TenfloweRSOperation`
    /// today, so there was no established execution convention to match;
    /// this is the one `StandardOpMapping` defines and follows consistently:
    ///
    /// - `map_operation(node, initializers)` resolves and validates the
    ///   node's attributes/inputs **at lowering time**. Any
    ///   `node.inputs[i]` whose name is a key in the
    ///   `initializers: &HashMap<String, Tensor<f32>>` map is treated as a
    ///   compile-time constant and baked directly into the returned op
    ///   object (e.g. `Reshape`'s target-shape input, or a `MatMul`/weight
    ///   operand that happens to be a graph initializer). Any input name
    ///   *not* found in `initializers` is a genuine runtime value.
    /// - `TenfloweRSOperation::execute(&self, inputs)` then receives
    ///   exactly the runtime (non-initializer) inputs, positionally, in the
    ///   same relative order they appear in `node.inputs`. So for a node
    ///   with `inputs = ["x", "w", "y"]` where `"w"` is an initializer,
    ///   `execute` receives `[x_value, y_value]` (`w` is already baked in).
    /// - Every unsupported `op_type` is a genuine, descriptive `Err` —
    ///   never a silent `Identity`/default/zero-fill substitute.
    fn map_operation(
        &self,
        node: &OnnxNode,
        initializers: &HashMap<String, Tensor<f32>>,
    ) -> Result<Box<dyn TenfloweRSOperation>> {
        let mut next_runtime_index = 0usize;

        match node.op_type.as_str() {
            "Add" | "Sub" | "Mul" | "Div" | "MatMul" => {
                if node.inputs.len() != 2 {
                    return Err(TensorError::invalid_argument_op(
                        &node.op_type,
                        "requires exactly 2 inputs",
                    ));
                }
                let kind = match node.op_type.as_str() {
                    "Add" => BinaryKind::Add,
                    "Sub" => BinaryKind::Sub,
                    "Mul" => BinaryKind::Mul,
                    "Div" => BinaryKind::Div,
                    "MatMul" => BinaryKind::MatMul,
                    _ => unreachable!("outer match already restricted op_type"),
                };
                let lhs = bake_operand(&node.inputs[0], initializers, &mut next_runtime_index);
                let rhs = bake_operand(&node.inputs[1], initializers, &mut next_runtime_index);
                Ok(Box::new(BinaryOp {
                    name: node.name.clone(),
                    kind,
                    lhs,
                    rhs,
                }))
            }

            "Relu" | "Sigmoid" | "Tanh" => {
                if node.inputs.len() != 1 {
                    return Err(TensorError::invalid_argument_op(
                        &node.op_type,
                        "requires exactly 1 input",
                    ));
                }
                let kind = match node.op_type.as_str() {
                    "Relu" => UnaryKind::Relu,
                    "Sigmoid" => UnaryKind::Sigmoid,
                    "Tanh" => UnaryKind::Tanh,
                    _ => unreachable!("outer match already restricted op_type"),
                };
                let input = bake_operand(&node.inputs[0], initializers, &mut next_runtime_index);
                Ok(Box::new(UnaryOp {
                    name: node.name.clone(),
                    kind,
                    input,
                }))
            }

            "Identity" => {
                if node.inputs.len() != 1 {
                    return Err(TensorError::invalid_argument_op(
                        "Identity",
                        "requires exactly 1 input",
                    ));
                }
                let input = bake_operand(&node.inputs[0], initializers, &mut next_runtime_index);
                Ok(Box::new(IdentityOp {
                    name: node.name.clone(),
                    input,
                }))
            }

            "Reshape" => {
                if node.inputs.len() != 2 {
                    return Err(TensorError::invalid_argument_op(
                        "Reshape",
                        "requires exactly 2 inputs (data, shape)",
                    ));
                }
                let shape_name = &node.inputs[1];
                let shape_tensor = initializers.get(shape_name).ok_or_else(|| {
                    TensorError::not_implemented_simple(format!(
                        "Reshape node '{}': shape input '{shape_name}' is not a resolvable \
                         initializer; dynamic runtime shapes are not supported by this static \
                         op-lowering pass",
                        node.name
                    ))
                })?;
                let shape_values = shape_tensor.to_vec()?;
                let target_shape: Vec<i64> =
                    shape_values.iter().map(|v| v.round() as i64).collect();
                let data = bake_operand(&node.inputs[0], initializers, &mut next_runtime_index);
                Ok(Box::new(ReshapeOp {
                    name: node.name.clone(),
                    data,
                    target_shape,
                }))
            }

            "Transpose" => {
                if node.inputs.len() != 1 {
                    return Err(TensorError::invalid_argument_op(
                        "Transpose",
                        "requires exactly 1 input",
                    ));
                }
                let perm = match optional_ints_attr(node, "perm")? {
                    None => None,
                    Some(values) => {
                        let mut perm = Vec::with_capacity(values.len());
                        for &v in values {
                            let axis = usize::try_from(v).map_err(|_| {
                                TensorError::invalid_argument_op(
                                    "Transpose",
                                    &format!("perm value {v} is negative"),
                                )
                            })?;
                            perm.push(axis);
                        }
                        Some(perm)
                    }
                };
                let input = bake_operand(&node.inputs[0], initializers, &mut next_runtime_index);
                Ok(Box::new(TransposeOp {
                    name: node.name.clone(),
                    input,
                    perm,
                }))
            }

            "Concat" => {
                if node.inputs.is_empty() {
                    return Err(TensorError::invalid_argument_op(
                        "Concat",
                        "requires at least 1 input",
                    ));
                }
                let axis_i64 = required_int_attr(node, "axis")?;
                if axis_i64 < 0 {
                    return Err(TensorError::not_implemented_simple(format!(
                        "Concat node '{}': negative axis {axis_i64} (relative to rank) is not \
                         supported by this static op-lowering pass, since tensor rank is not \
                         known until runtime",
                        node.name
                    )));
                }
                let axis = axis_i64 as usize;
                let inputs = node
                    .inputs
                    .iter()
                    .map(|name| bake_operand(name, initializers, &mut next_runtime_index))
                    .collect();
                Ok(Box::new(ConcatOp {
                    name: node.name.clone(),
                    inputs,
                    axis,
                }))
            }

            "Softmax" => {
                if node.inputs.len() != 1 {
                    return Err(TensorError::invalid_argument_op(
                        "Softmax",
                        "requires exactly 1 input",
                    ));
                }
                // ONNX opset >= 13 defaults `axis` to -1 (the last axis); we
                // follow that default here.
                let axis_i64 = optional_int_attr(node, "axis")?.unwrap_or(-1);
                let axis = i32::try_from(axis_i64).map_err(|_| {
                    TensorError::invalid_argument_op(
                        "Softmax",
                        &format!("axis value {axis_i64} is out of i32 range"),
                    )
                })?;
                let input = bake_operand(&node.inputs[0], initializers, &mut next_runtime_index);
                Ok(Box::new(SoftmaxOp {
                    name: node.name.clone(),
                    input,
                    axis,
                }))
            }

            "Flatten" => {
                if node.inputs.len() != 1 {
                    return Err(TensorError::invalid_argument_op(
                        "Flatten",
                        "requires exactly 1 input",
                    ));
                }
                let axis = optional_int_attr(node, "axis")?.unwrap_or(1);
                let input = bake_operand(&node.inputs[0], initializers, &mut next_runtime_index);
                Ok(Box::new(FlattenOp {
                    name: node.name.clone(),
                    input,
                    axis,
                }))
            }

            "Gemm" => {
                if node.inputs.len() < 2 || node.inputs.len() > 3 {
                    return Err(TensorError::invalid_argument_op(
                        "Gemm",
                        "requires 2 or 3 inputs (A, B[, C])",
                    ));
                }
                let alpha = optional_float_attr(node, "alpha")?.unwrap_or(1.0);
                let beta = optional_float_attr(node, "beta")?.unwrap_or(1.0);
                let trans_a = optional_int_attr(node, "transA")?.unwrap_or(0) != 0;
                let trans_b = optional_int_attr(node, "transB")?.unwrap_or(0) != 0;

                let a = bake_operand(&node.inputs[0], initializers, &mut next_runtime_index);
                let b = bake_operand(&node.inputs[1], initializers, &mut next_runtime_index);
                let c = node
                    .inputs
                    .get(2)
                    .map(|name| bake_operand(name, initializers, &mut next_runtime_index));

                Ok(Box::new(GemmOp {
                    name: node.name.clone(),
                    a,
                    b,
                    c,
                    alpha,
                    beta,
                    trans_a,
                    trans_b,
                }))
            }

            other => Err(TensorError::not_implemented_simple(format!(
                "ONNX op '{other}' (node '{}') is not yet supported for lowering; supported \
                 ops: Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, MatMul, Reshape, Transpose, \
                 Identity, Concat, Softmax, Flatten, Gemm",
                node.name
            ))),
        }
    }

    fn supported_ops(&self) -> Vec<String> {
        [
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Relu",
            "Sigmoid",
            "Tanh",
            "MatMul",
            "Reshape",
            "Transpose",
            "Identity",
            "Concat",
            "Softmax",
            "Flatten",
            "Gemm",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }
}

/// Lowers every node in `graph` via `mapping`, in order, returning one
/// `TenfloweRSOperation` per node.
///
/// Builds the `initializers: HashMap<String, Tensor<f32>>` map required by
/// `OnnxOpMapping::map_operation` from `graph.initializers` (decoding each
/// initializer's raw bytes via [`decode_onnx_tensor`]), then lowers every
/// node in `graph.nodes`, propagating the first error encountered.
pub fn lower_graph(
    mapping: &dyn OnnxOpMapping,
    graph: &OnnxGraph,
) -> Result<Vec<Box<dyn TenfloweRSOperation>>> {
    let mut initializers = HashMap::with_capacity(graph.initializers.len());
    for tensor in &graph.initializers {
        initializers.insert(tensor.name.clone(), decode_onnx_tensor(tensor)?);
    }

    graph
        .nodes
        .iter()
        .map(|node| mapping.map_operation(node, &initializers))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(op_type: &str, inputs: &[&str], outputs: &[&str]) -> OnnxNode {
        OnnxNode {
            name: format!("{op_type}_node"),
            op_type: op_type.to_string(),
            inputs: inputs.iter().map(|&s| s.to_string()).collect(),
            outputs: outputs.iter().map(|&s| s.to_string()).collect(),
            attributes: HashMap::new(),
        }
    }

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::from_vec(data, shape).expect("valid tensor fixture")
    }

    #[test]
    fn add_computes_elementwise_sum() {
        let mapping = StandardOpMapping;
        let n = node("Add", &["a", "b"], &["c"]);
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("Add should lower");
        let out = op
            .execute(&[
                t(vec![1.0, 2.0, 3.0], &[3]),
                t(vec![10.0, 20.0, 30.0], &[3]),
            ])
            .expect("Add should execute");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].to_vec().expect("vec"), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn sub_computes_elementwise_difference() {
        let mapping = StandardOpMapping;
        let n = node("Sub", &["a", "b"], &["c"]);
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("Sub should lower");
        let out = op
            .execute(&[t(vec![5.0, 5.0], &[2]), t(vec![2.0, 3.0], &[2])])
            .expect("Sub should execute");
        assert_eq!(out[0].to_vec().expect("vec"), vec![3.0, 2.0]);
    }

    #[test]
    fn mul_computes_elementwise_product() {
        let mapping = StandardOpMapping;
        let n = node("Mul", &["a", "b"], &["c"]);
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("Mul should lower");
        let out = op
            .execute(&[t(vec![2.0, 3.0], &[2]), t(vec![4.0, 5.0], &[2])])
            .expect("Mul should execute");
        assert_eq!(out[0].to_vec().expect("vec"), vec![8.0, 15.0]);
    }

    #[test]
    fn div_computes_elementwise_quotient() {
        let mapping = StandardOpMapping;
        let n = node("Div", &["a", "b"], &["c"]);
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("Div should lower");
        let out = op
            .execute(&[t(vec![10.0, 9.0], &[2]), t(vec![2.0, 3.0], &[2])])
            .expect("Div should execute");
        assert_eq!(out[0].to_vec().expect("vec"), vec![5.0, 3.0]);
    }

    #[test]
    fn add_with_initializer_operand_is_baked_in() {
        // "b" is a graph initializer, so `execute` should only need to be
        // given the single runtime input "a".
        let mapping = StandardOpMapping;
        let n = node("Add", &["a", "b"], &["c"]);
        let mut initializers = HashMap::new();
        initializers.insert("b".to_string(), t(vec![100.0, 200.0], &[2]));
        let op = mapping
            .map_operation(&n, &initializers)
            .expect("Add should lower");
        let out = op
            .execute(&[t(vec![1.0, 2.0], &[2])])
            .expect("Add should execute with only the runtime operand");
        assert_eq!(out[0].to_vec().expect("vec"), vec![101.0, 202.0]);
    }

    #[test]
    fn binary_op_errors_on_wrong_arity() {
        let mapping = StandardOpMapping;
        let n = node("Add", &["a"], &["c"]);
        assert!(mapping.map_operation(&n, &HashMap::new()).is_err());
    }

    #[test]
    fn relu_clamps_negatives_to_zero() {
        let mapping = StandardOpMapping;
        let n = node("Relu", &["x"], &["y"]);
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("Relu should lower");
        let out = op
            .execute(&[t(vec![-1.0, 0.0, 2.0, -5.0], &[4])])
            .expect("Relu should execute");
        assert_eq!(out[0].to_vec().expect("vec"), vec![0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn sigmoid_matches_known_values() {
        let mapping = StandardOpMapping;
        let n = node("Sigmoid", &["x"], &["y"]);
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("Sigmoid should lower");
        let out = op
            .execute(&[t(vec![0.0], &[1])])
            .expect("Sigmoid should execute");
        assert!((out[0].to_vec().expect("vec")[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn tanh_matches_known_values() {
        let mapping = StandardOpMapping;
        let n = node("Tanh", &["x"], &["y"]);
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("Tanh should lower");
        let out = op
            .execute(&[t(vec![0.0], &[1])])
            .expect("Tanh should execute");
        assert!((out[0].to_vec().expect("vec")[0]).abs() < 1e-6);
    }

    #[test]
    fn matmul_multiplies_small_matrices() {
        let mapping = StandardOpMapping;
        let n = node("MatMul", &["a", "b"], &["c"]);
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("MatMul should lower");
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let out = op
            .execute(&[
                t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]),
                t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]),
            ])
            .expect("MatMul should execute");
        assert_eq!(out[0].shape().dims(), &[2, 2]);
        assert_eq!(out[0].to_vec().expect("vec"), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn reshape_uses_initializer_shape() {
        let mapping = StandardOpMapping;
        let n = node("Reshape", &["data", "shape"], &["out"]);
        let mut initializers = HashMap::new();
        initializers.insert("shape".to_string(), t(vec![2.0, 3.0], &[2]));
        let op = mapping
            .map_operation(&n, &initializers)
            .expect("Reshape should lower");
        let out = op
            .execute(&[t((1..=6).map(|v| v as f32).collect(), &[6])])
            .expect("Reshape should execute");
        assert_eq!(out[0].shape().dims(), &[2, 3]);
        assert_eq!(
            out[0].to_vec().expect("vec"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn reshape_infers_negative_one_dimension() {
        let mapping = StandardOpMapping;
        let n = node("Reshape", &["data", "shape"], &["out"]);
        let mut initializers = HashMap::new();
        initializers.insert("shape".to_string(), t(vec![2.0, -1.0], &[2]));
        let op = mapping
            .map_operation(&n, &initializers)
            .expect("Reshape should lower");
        let out = op
            .execute(&[t((1..=6).map(|v| v as f32).collect(), &[6])])
            .expect("Reshape should execute");
        assert_eq!(out[0].shape().dims(), &[2, 3]);
    }

    #[test]
    fn reshape_errors_when_shape_input_is_not_an_initializer() {
        let mapping = StandardOpMapping;
        let n = node("Reshape", &["data", "shape"], &["out"]);
        // No initializers provided: "shape" is a dynamic runtime value,
        // which this static lowering pass cannot resolve.
        assert!(mapping.map_operation(&n, &HashMap::new()).is_err());
    }

    #[test]
    fn transpose_default_reverses_axes() {
        let mapping = StandardOpMapping;
        let n = node("Transpose", &["x"], &["y"]);
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("Transpose should lower");
        let out = op
            .execute(&[t((1..=6).map(|v| v as f32).collect(), &[2, 3])])
            .expect("Transpose should execute");
        assert_eq!(out[0].shape().dims(), &[3, 2]);
    }

    #[test]
    fn transpose_with_explicit_perm() {
        let mapping = StandardOpMapping;
        let mut n = node("Transpose", &["x"], &["y"]);
        n.attributes
            .insert("perm".to_string(), OnnxAttribute::Ints(vec![0, 2, 1]));
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("Transpose should lower");
        let out = op
            .execute(&[t((1..=24).map(|v| v as f32).collect(), &[2, 3, 4])])
            .expect("Transpose should execute");
        assert_eq!(out[0].shape().dims(), &[2, 4, 3]);
    }

    #[test]
    fn identity_passes_through_unchanged() {
        let mapping = StandardOpMapping;
        let n = node("Identity", &["x"], &["y"]);
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("Identity should lower");
        let out = op
            .execute(&[t(vec![1.0, 2.0], &[2])])
            .expect("Identity should execute");
        assert_eq!(out[0].to_vec().expect("vec"), vec![1.0, 2.0]);
    }

    #[test]
    fn concat_along_axis_0() {
        let mapping = StandardOpMapping;
        let mut n = node("Concat", &["a", "b"], &["c"]);
        n.attributes
            .insert("axis".to_string(), OnnxAttribute::Int(0));
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("Concat should lower");
        let out = op
            .execute(&[t(vec![1.0, 2.0], &[1, 2]), t(vec![3.0, 4.0], &[1, 2])])
            .expect("Concat should execute");
        assert_eq!(out[0].shape().dims(), &[2, 2]);
        assert_eq!(out[0].to_vec().expect("vec"), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn concat_missing_axis_attribute_errors() {
        let mapping = StandardOpMapping;
        let n = node("Concat", &["a", "b"], &["c"]);
        assert!(mapping.map_operation(&n, &HashMap::new()).is_err());
    }

    #[test]
    fn softmax_output_sums_to_one() {
        let mapping = StandardOpMapping;
        let n = node("Softmax", &["x"], &["y"]);
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("Softmax should lower");
        let out = op
            .execute(&[t(vec![1.0, 2.0, 3.0], &[1, 3])])
            .expect("Softmax should execute");
        let sum: f32 = out[0].to_vec().expect("vec").iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn flatten_reshapes_to_2d() {
        let mapping = StandardOpMapping;
        let n = node("Flatten", &["x"], &["y"]); // default axis = 1
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("Flatten should lower");
        let out = op
            .execute(&[t((1..=24).map(|v| v as f32).collect(), &[2, 3, 4])])
            .expect("Flatten should execute");
        assert_eq!(out[0].shape().dims(), &[2, 12]);
    }

    #[test]
    fn gemm_computes_alpha_ab_plus_beta_c() {
        let mapping = StandardOpMapping;
        let mut n = node("Gemm", &["a", "b", "c"], &["y"]);
        n.attributes
            .insert("alpha".to_string(), OnnxAttribute::Float(2.0));
        n.attributes
            .insert("beta".to_string(), OnnxAttribute::Float(0.5));
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("Gemm should lower");
        // alpha * A@B = 2 * [[1,2],[3,4]] (identity) = [[2,4],[6,8]]
        // + beta * C = 0.5 * [[1,1],[1,1]] = [[0.5,0.5],[0.5,0.5]]
        // = [[2.5,4.5],[6.5,8.5]]
        let out = op
            .execute(&[
                t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]),
                t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]),
                t(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]),
            ])
            .expect("Gemm should execute");
        assert_eq!(out[0].to_vec().expect("vec"), vec![2.5, 4.5, 6.5, 8.5]);
    }

    #[test]
    fn gemm_without_bias_input() {
        let mapping = StandardOpMapping;
        let n = node("Gemm", &["a", "b"], &["y"]);
        let op = mapping
            .map_operation(&n, &HashMap::new())
            .expect("Gemm should lower with only 2 inputs");
        let out = op
            .execute(&[
                t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]),
                t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]),
            ])
            .expect("Gemm should execute");
        assert_eq!(out[0].to_vec().expect("vec"), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn unsupported_op_returns_honest_error() {
        let mapping = StandardOpMapping;
        let n = node("Conv", &["x", "w"], &["y"]);
        let result = mapping.map_operation(&n, &HashMap::new());
        assert!(result.is_err());
        let message = format!("{}", result.err().expect("should be an error"));
        assert!(message.contains("Conv"));
    }

    #[test]
    fn decode_onnx_tensor_widens_int64_to_f32() {
        let tensor = OnnxTensor {
            name: "ints".to_string(),
            data_type: OnnxDataType::Int64,
            dims: vec![2],
            data: [7i64, -3i64].iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        let decoded = decode_onnx_tensor(&tensor).expect("Int64 tensor should decode");
        assert_eq!(decoded.to_vec().expect("vec"), vec![7.0, -3.0]);
    }

    #[test]
    fn decode_onnx_tensor_rejects_string_data_type() {
        let tensor = OnnxTensor {
            name: "strs".to_string(),
            data_type: OnnxDataType::String,
            dims: vec![1],
            data: vec![0],
        };
        assert!(decode_onnx_tensor(&tensor).is_err());
    }

    #[test]
    fn lower_graph_runs_a_small_multi_node_pipeline() {
        let add_node = node("Add", &["input", "bias"], &["added"]);
        let mut relu_node = node("Relu", &["added"], &["output"]);
        relu_node.name = "relu_final".to_string();

        let graph = OnnxGraph {
            nodes: vec![add_node, relu_node],
            inputs: vec![],
            outputs: vec![],
            initializers: vec![OnnxTensor {
                name: "bias".to_string(),
                data_type: OnnxDataType::Float,
                dims: vec![3],
                data: [-5.0f32, 0.0, 5.0]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            }],
            value_info: vec![],
            name: "pipeline".to_string(),
        };

        let mapping = StandardOpMapping;
        let ops = lower_graph(&mapping, &graph).expect("graph should lower");
        assert_eq!(ops.len(), 2);

        let after_add = ops[0]
            .execute(&[t(vec![1.0, 1.0, 1.0], &[3])])
            .expect("Add should execute");
        let after_relu = ops[1].execute(&after_add).expect("Relu should execute");
        // input(1,1,1) + bias(-5,0,5) = (-4, 1, 6); relu -> (0, 1, 6)
        assert_eq!(after_relu[0].to_vec().expect("vec"), vec![0.0, 1.0, 6.0]);
    }
}
