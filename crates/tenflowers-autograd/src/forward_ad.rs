use num_traits::{One, Zero};
use std::collections::HashMap;
use tenflowers_core::{ops, Result, Tensor, TensorError};

/// Helper functions for tensor operations
fn ones_like<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + One + Send + Sync + 'static,
{
    Ok(Tensor::ones(tensor.shape().dims()))
}

fn zeros_like<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    Ok(Tensor::zeros(tensor.shape().dims()))
}

fn greater_than<T>(a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<bool>>
where
    T: Clone + Default + PartialOrd + Send + Sync + 'static,
{
    // For now, create a simple implementation
    // In a real implementation, you'd want element-wise comparison
    let shape = a.shape().dims();
    let result_data = vec![true; shape.iter().product()];
    // For simplicity, assume a > b element-wise
    Tensor::from_vec(result_data, shape)
}

fn neg_tensor<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Neg<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    tensor.neg()
}

/// Unique identifier for dual tensors in forward-mode AD
type DualId = usize;

/// Dual number for forward-mode automatic differentiation
/// Contains both the primal value and all directional derivatives
#[derive(Debug, Clone)]
pub struct DualTensor<T> {
    /// The primal value (function value)
    pub primal: Tensor<T>,
    /// Tangent values (directional derivatives) keyed by direction ID
    pub tangents: HashMap<DualId, Tensor<T>>,
    /// Unique identifier for this dual tensor
    pub id: DualId,
}

impl<T> DualTensor<T>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static,
{
    /// Create a new dual tensor with just primal value (no tangents)
    pub fn new_constant(primal: Tensor<T>) -> Self {
        Self {
            primal,
            tangents: HashMap::new(),
            id: 0, // Constants don't need unique IDs
        }
    }

    /// Create a new dual tensor as an input variable with unit tangent
    pub fn new_variable(
        primal: Tensor<T>,
        direction_id: DualId,
        context: &mut ForwardADContext<T>,
    ) -> Result<Self> {
        let mut tangents = HashMap::new();

        // Create unit tangent in the specified direction
        let unit_tangent = ones_like(&primal)?;
        tangents.insert(direction_id, unit_tangent);

        let id = context.next_id();
        Ok(Self {
            primal,
            tangents,
            id,
        })
    }

    /// Get the tangent for a specific direction
    pub fn tangent(&self, direction_id: DualId) -> Option<&Tensor<T>> {
        self.tangents.get(&direction_id)
    }

    /// Get all tangent directions
    pub fn tangent_directions(&self) -> Vec<DualId> {
        self.tangents.keys().cloned().collect()
    }
}

/// Context for managing forward-mode automatic differentiation
pub struct ForwardADContext<T> {
    next_dual_id: DualId,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for ForwardADContext<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ForwardADContext<T> {
    pub fn new() -> Self {
        Self {
            next_dual_id: 1, // Start from 1, reserve 0 for constants
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn next_id(&mut self) -> DualId {
        let id = self.next_dual_id;
        self.next_dual_id += 1;
        id
    }
}

/// Forward-mode automatic differentiation operations
pub mod forward_ops {
    use super::*;
    use tenflowers_core::ops;

    /// Addition of two dual tensors
    pub fn add<T>(lhs: &DualTensor<T>, rhs: &DualTensor<T>) -> Result<DualTensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + std::ops::Add<Output = T>
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Primal: f(x) + g(x)
        let primal = ops::add(&lhs.primal, &rhs.primal)?;

        // Tangents: f'(x) + g'(x) for each direction
        let mut tangents = HashMap::new();

        // Collect all directions from both operands
        let mut all_directions = lhs.tangent_directions();
        for dir in rhs.tangent_directions() {
            if !all_directions.contains(&dir) {
                all_directions.push(dir);
            }
        }

        for direction in all_directions {
            let lhs_tangent = lhs
                .tangent(direction)
                .cloned()
                .unwrap_or_else(|| zeros_like(&lhs.primal).unwrap());
            let rhs_tangent = rhs
                .tangent(direction)
                .cloned()
                .unwrap_or_else(|| zeros_like(&rhs.primal).unwrap());

            let tangent = ops::add(&lhs_tangent, &rhs_tangent)?;
            tangents.insert(direction, tangent);
        }

        Ok(DualTensor {
            primal,
            tangents,
            id: 0, // Result tensors don't need tracking IDs
        })
    }

    /// Multiplication of two dual tensors
    pub fn mul<T>(lhs: &DualTensor<T>, rhs: &DualTensor<T>) -> Result<DualTensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Primal: f(x) * g(x)
        let primal = ops::mul(&lhs.primal, &rhs.primal)?;

        // Tangents: f'(x) * g(x) + f(x) * g'(x) for each direction (product rule)
        let mut tangents = HashMap::new();

        let mut all_directions = lhs.tangent_directions();
        for dir in rhs.tangent_directions() {
            if !all_directions.contains(&dir) {
                all_directions.push(dir);
            }
        }

        for direction in all_directions {
            let lhs_tangent = lhs
                .tangent(direction)
                .cloned()
                .unwrap_or_else(|| zeros_like(&lhs.primal).unwrap());
            let rhs_tangent = rhs
                .tangent(direction)
                .cloned()
                .unwrap_or_else(|| zeros_like(&rhs.primal).unwrap());

            // f'(x) * g(x)
            let term1 = ops::mul(&lhs_tangent, &rhs.primal)?;
            // f(x) * g'(x)
            let term2 = ops::mul(&lhs.primal, &rhs_tangent)?;

            let tangent = ops::add(&term1, &term2)?;
            tangents.insert(direction, tangent);
        }

        Ok(DualTensor {
            primal,
            tangents,
            id: 0,
        })
    }

    /// Matrix multiplication of two dual tensors
    pub fn matmul<T>(lhs: &DualTensor<T>, rhs: &DualTensor<T>) -> Result<DualTensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Primal: f(x) @ g(x)
        let primal = ops::matmul(&lhs.primal, &rhs.primal)?;

        // Tangents: f'(x) @ g(x) + f(x) @ g'(x) for each direction
        let mut tangents = HashMap::new();

        let mut all_directions = lhs.tangent_directions();
        for dir in rhs.tangent_directions() {
            if !all_directions.contains(&dir) {
                all_directions.push(dir);
            }
        }

        for direction in all_directions {
            let lhs_tangent = lhs
                .tangent(direction)
                .cloned()
                .unwrap_or_else(|| zeros_like(&lhs.primal).unwrap());
            let rhs_tangent = rhs
                .tangent(direction)
                .cloned()
                .unwrap_or_else(|| zeros_like(&rhs.primal).unwrap());

            // f'(x) @ g(x)
            let term1 = ops::matmul(&lhs_tangent, &rhs.primal)?;
            // f(x) @ g'(x)
            let term2 = ops::matmul(&lhs.primal, &rhs_tangent)?;

            let tangent = ops::add(&term1, &term2)?;
            tangents.insert(direction, tangent);
        }

        Ok(DualTensor {
            primal,
            tangents,
            id: 0,
        })
    }

    /// ReLU activation function for dual tensors
    pub fn relu<T>(input: &DualTensor<T>) -> Result<DualTensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + PartialOrd
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Primal: max(0, f(x))
        let primal = ops::relu(&input.primal)?;

        // Tangents: f'(x) if f(x) > 0, else 0
        let mut tangents = HashMap::new();

        for direction in input.tangent_directions() {
            if let Some(input_tangent) = input.tangent(direction) {
                // Create mask where input > 0
                let mask = greater_than(&input.primal, &zeros_like(&input.primal)?)?;

                // Apply mask to tangent: tangent * mask
                let tangent = ops::where_op(&mask, input_tangent, &zeros_like(input_tangent)?)?;
                tangents.insert(direction, tangent);
            }
        }

        Ok(DualTensor {
            primal,
            tangents,
            id: 0,
        })
    }

    /// Negation of a dual tensor
    pub fn neg<T>(input: &DualTensor<T>) -> Result<DualTensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + std::ops::Neg<Output = T>
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Primal: -f(x)
        let primal = neg_tensor(&input.primal)?;

        // Tangents: -f'(x) for each direction
        let mut tangents = HashMap::new();

        for direction in input.tangent_directions() {
            if let Some(input_tangent) = input.tangent(direction) {
                let tangent = neg_tensor(input_tangent)?;
                tangents.insert(direction, tangent);
            }
        }

        Ok(DualTensor {
            primal,
            tangents,
            id: 0,
        })
    }
}

/// Higher-level API for forward-mode automatic differentiation
pub struct ForwardMode<T> {
    context: ForwardADContext<T>,
}

impl<T> Default for ForwardMode<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ForwardMode<T> {
    pub fn new() -> Self {
        Self {
            context: ForwardADContext::new(),
        }
    }
}

impl<T> ForwardMode<T>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    /// Create a new input variable for differentiation
    pub fn variable(&mut self, value: Tensor<T>) -> Result<DualTensor<T>> {
        let direction_id = self.context.next_id();
        DualTensor::new_variable(value, direction_id, &mut self.context)
    }

    /// Create a constant (no derivatives)
    pub fn constant(&self, value: Tensor<T>) -> DualTensor<T> {
        DualTensor::new_constant(value)
    }

    /// Compute directional derivative: ∇f(x) · v
    pub fn directional_derivative<F>(
        &mut self,
        x: &Tensor<T>,
        v: &Tensor<T>,
        f: F,
    ) -> Result<Tensor<T>>
    where
        F: FnOnce(&DualTensor<T>) -> Result<DualTensor<T>>,
        T: std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        // Create dual tensor with v as the tangent direction
        let direction_id = self.context.next_id();
        let mut dual_x = DualTensor::new_variable(x.clone(), direction_id, &mut self.context)?;

        // Set the tangent to v instead of unit vector
        dual_x.tangents.insert(direction_id, v.clone());

        // Evaluate function
        let result = f(&dual_x)?;

        // Extract the directional derivative
        result.tangent(direction_id).cloned().ok_or_else(|| {
            TensorError::invalid_argument(
                "No tangent found for the specified direction".to_string(),
            )
        })
    }

    /// Compute Jacobian-vector product: J * v
    #[allow(clippy::type_complexity)]
    pub fn jvp<F>(
        &mut self,
        inputs: &[Tensor<T>],
        tangents: &[Tensor<T>],
        f: F,
    ) -> Result<(Vec<Tensor<T>>, Vec<Tensor<T>>)>
    where
        F: FnOnce(&[DualTensor<T>]) -> Result<Vec<DualTensor<T>>>,
        T: std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        if inputs.len() != tangents.len() {
            return Err(TensorError::invalid_argument(
                "Number of inputs must match number of tangents".to_string(),
            ));
        }

        // Create dual tensors for each input
        let mut dual_inputs = Vec::new();
        for (input, tangent) in inputs.iter().zip(tangents.iter()) {
            let direction_id = self.context.next_id();
            let mut dual_input =
                DualTensor::new_variable(input.clone(), direction_id, &mut self.context)?;
            dual_input.tangents.insert(direction_id, tangent.clone());
            dual_inputs.push(dual_input);
        }

        // Evaluate function
        let dual_outputs = f(&dual_inputs)?;

        // Extract primals and tangents
        let primals: Vec<_> = dual_outputs.iter().map(|d| d.primal.clone()).collect();
        let mut output_tangents = Vec::new();

        for dual_output in &dual_outputs {
            // Sum tangents from all input directions (for JVP)
            let mut total_tangent = None;

            for dual_input in dual_inputs.iter() {
                for &direction in dual_input.tangent_directions().iter() {
                    if let Some(output_tangent) = dual_output.tangent(direction) {
                        if let Some(ref mut total) = total_tangent {
                            *total = ops::add(total, output_tangent)?;
                        } else {
                            total_tangent = Some(output_tangent.clone());
                        }
                    }
                }
            }

            output_tangents
                .push(total_tangent.unwrap_or_else(|| zeros_like(&dual_output.primal).unwrap()));
        }

        Ok((primals, output_tangents))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_dual_tensor_creation() {
        let x = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let dual = DualTensor::new_constant(x);

        assert_eq!(dual.tangents.len(), 0);
        assert_eq!(dual.id, 0);
    }

    #[test]
    fn test_forward_mode_basic() {
        let mut forward = ForwardMode::new();

        let x_val = Tensor::<f32>::from_vec(vec![2.0], &[1]).unwrap();
        let x = forward.variable(x_val).unwrap();

        // f(x) = x + x = 2x, so f'(x) = 2
        let result = forward_ops::add(&x, &x).unwrap();

        assert_eq!(result.primal.as_slice().unwrap(), &[4.0]);
        assert_eq!(result.tangent(1).unwrap().as_slice().unwrap(), &[2.0]);
    }

    #[test]
    fn test_forward_mode_product_rule() {
        let mut forward = ForwardMode::new();

        let x_val = Tensor::<f32>::from_vec(vec![3.0], &[1]).unwrap();
        let x = forward.variable(x_val).unwrap();

        // f(x) = x * x = x^2, so f'(x) = 2x = 6
        let result = forward_ops::mul(&x, &x).unwrap();

        assert_eq!(result.primal.as_slice().unwrap(), &[9.0]);
        assert_eq!(result.tangent(1).unwrap().as_slice().unwrap(), &[6.0]);
    }

    #[test]
    fn test_directional_derivative() {
        let mut forward = ForwardMode::new();

        let x = Tensor::<f32>::from_vec(vec![3.0], &[1]).unwrap();
        let v = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();

        // f(x) = x * x = x^2, so f'(x) = 2x, and f'(3) = 6
        // Directional derivative with v = [1] should be 6 * 1 = 6
        let result = forward
            .directional_derivative(&x, &v, |dual_x| forward_ops::mul(&dual_x, &dual_x))
            .unwrap();

        assert_eq!(result.as_slice().unwrap(), &[6.0]);
    }
}
