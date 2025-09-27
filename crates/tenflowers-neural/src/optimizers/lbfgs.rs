use crate::model::Model;
use crate::optimizers::Optimizer;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

/// L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) optimizer
///
/// This is a quasi-Newton method that approximates the inverse Hessian matrix
/// using a limited history of gradients and parameter updates. It's particularly
/// effective for full-batch optimization problems.
///
/// # References
/// - Nocedal, J. (1980). "Updating Quasi-Newton Matrices with Limited Storage"
/// - Liu, D.C. and Nocedal, J. (1989). "On the Limited Memory Method for Large Scale Optimization"
pub struct LBFGS<T> {
    /// Learning rate (step size)
    learning_rate: f32,
    /// Maximum number of correction pairs to store (memory limit)
    max_iter: usize,
    /// Maximum number of function evaluations
    max_eval: Option<usize>,
    /// Gradient tolerance for convergence
    tolerance_grad: f32,
    /// Change in function value tolerance for convergence  
    tolerance_change: f32,
    /// History buffer size (number of {s_k, y_k} pairs to remember)
    history_size: usize,
    /// Current step number
    step_count: usize,
    /// Function evaluation count
    eval_count: usize,
    /// Previous parameters for each tensor
    prev_params: HashMap<*const Tensor<T>, Tensor<T>>,
    /// Previous gradients for each tensor
    prev_grads: HashMap<*const Tensor<T>, Tensor<T>>,
    /// Parameter differences history: s_k = x_{k+1} - x_k
    s_history: HashMap<*const Tensor<T>, Vec<Tensor<T>>>,
    /// Gradient differences history: y_k = g_{k+1} - g_k
    y_history: HashMap<*const Tensor<T>, Vec<Tensor<T>>>,
    /// Reciprocal of y_k^T s_k for each history entry
    rho_history: HashMap<*const Tensor<T>, Vec<T>>,
    /// Direction search result for line search
    search_direction: HashMap<*const Tensor<T>, Tensor<T>>,
    /// Current loss value (for convergence checking)
    current_loss: Option<f32>,
    /// Previous loss value
    prev_loss: Option<f32>,
}

impl<T> LBFGS<T> {
    /// Create a new L-BFGS optimizer
    ///
    /// # Arguments
    /// * `learning_rate` - Initial learning rate (default: 1.0)
    /// * `max_iter` - Maximum number of iterations (default: 20)
    /// * `history_size` - Number of {s_k, y_k} pairs to store (default: 10)
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            max_iter: 20,
            max_eval: None,
            tolerance_grad: 1e-7,
            tolerance_change: 1e-9,
            history_size: 10,
            step_count: 0,
            eval_count: 0,
            prev_params: HashMap::new(),
            prev_grads: HashMap::new(),
            s_history: HashMap::new(),
            y_history: HashMap::new(),
            rho_history: HashMap::new(),
            search_direction: HashMap::new(),
            current_loss: None,
            prev_loss: None,
        }
    }

    /// Set maximum number of iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set maximum number of function evaluations
    pub fn with_max_eval(mut self, max_eval: usize) -> Self {
        self.max_eval = Some(max_eval);
        self
    }

    /// Set gradient tolerance for convergence
    pub fn with_tolerance_grad(mut self, tolerance_grad: f32) -> Self {
        self.tolerance_grad = tolerance_grad;
        self
    }

    /// Set change tolerance for convergence
    pub fn with_tolerance_change(mut self, tolerance_change: f32) -> Self {
        self.tolerance_change = tolerance_change;
        self
    }

    /// Set history buffer size
    pub fn with_history_size(mut self, history_size: usize) -> Self {
        self.history_size = history_size;
        self
    }

    /// Get current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get current function evaluation count
    pub fn eval_count(&self) -> usize {
        self.eval_count
    }

    /// Set current loss value for convergence checking
    pub fn set_loss(&mut self, loss: f32) {
        self.prev_loss = self.current_loss;
        self.current_loss = Some(loss);
        self.eval_count += 1;
    }

    /// Check if optimization has converged
    pub fn has_converged(&self) -> bool {
        // Check maximum iterations
        if self.step_count >= self.max_iter {
            return true;
        }

        // Check maximum evaluations
        if let Some(max_eval) = self.max_eval {
            if self.eval_count >= max_eval {
                return true;
            }
        }

        // Check function value change tolerance
        if let (Some(current), Some(prev)) = (self.current_loss, self.prev_loss) {
            if (prev - current).abs() < self.tolerance_change {
                return true;
            }
        }

        false
    }
}

impl<T> Default for LBFGS<T> {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl<T> Optimizer<T> for LBFGS<T>
where
    T: Clone
        + Default
        + num_traits::Zero
        + num_traits::One
        + num_traits::Float
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::FromPrimitive
        + PartialOrd
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn step(&mut self, model: &mut dyn Model<T>) -> Result<()> {
        self.step_count += 1;

        // Check convergence before proceeding
        if self.has_converged() {
            return Ok(());
        }

        let lr_t = T::from(self.learning_rate).unwrap();

        // Process each parameter tensor
        for param in model.parameters_mut() {
            if let Some(grad) = param.grad() {
                let grad = grad.clone();
                let param_ptr = param as *const Tensor<T>;

                // Check gradient tolerance for convergence
                if let Ok(grad_norm) = self.compute_gradient_norm(&grad) {
                    if grad_norm < T::from(self.tolerance_grad).unwrap() {
                        continue; // Skip this parameter if gradient is small enough
                    }
                }

                // Get or initialize search direction
                let search_dir = if self.step_count == 1 {
                    // First step: use steepest descent direction
                    grad.mul(&Tensor::from_scalar(-lr_t))?
                } else {
                    // Compute L-BFGS search direction
                    self.compute_lbfgs_direction(param_ptr, &grad)?
                };

                // Store search direction for potential line search
                self.search_direction.insert(param_ptr, search_dir.clone());

                // Simple step without line search (can be extended)
                let step_size = lr_t;
                let step = search_dir.mul(&Tensor::from_scalar(step_size))?;
                let new_param = param.add(&step)?;

                // Update history if we have previous gradients
                if let Some(prev_grad) = self.prev_grads.get(&param_ptr).cloned() {
                    if let Some(prev_param) = self.prev_params.get(&param_ptr).cloned() {
                        self.update_history(param_ptr, param, &prev_param, &grad, &prev_grad)?;
                    }
                }

                // Store current parameter and gradient for next iteration
                self.prev_params.insert(param_ptr, param.clone());
                self.prev_grads.insert(param_ptr, grad);

                // Update parameter
                *param = new_param;
            }
        }

        Ok(())
    }

    fn zero_grad(&self, model: &mut dyn Model<T>) {
        model.zero_grad();
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }

    fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

impl<T> LBFGS<T>
where
    T: Clone
        + Default
        + num_traits::Zero
        + num_traits::One
        + num_traits::Float
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::FromPrimitive
        + PartialOrd
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Compute L-BFGS search direction using two-loop recursion
    fn compute_lbfgs_direction(
        &self,
        param_ptr: *const Tensor<T>,
        grad: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        let s_hist = self.s_history.get(&param_ptr).cloned().unwrap_or_default();
        let y_hist = self.y_history.get(&param_ptr).cloned().unwrap_or_default();
        let rho_hist = self
            .rho_history
            .get(&param_ptr)
            .cloned()
            .unwrap_or_default();

        if s_hist.is_empty() || y_hist.is_empty() || rho_hist.is_empty() {
            // No history available, use steepest descent
            return grad.mul(&Tensor::from_scalar(-T::one()));
        }

        let m = s_hist.len().min(y_hist.len()).min(rho_hist.len());
        if m == 0 {
            return grad.mul(&Tensor::from_scalar(-T::one()));
        }

        // Initialize q = gradient
        let mut q = grad.clone();

        // First loop (backward)
        let mut alphas = Vec::with_capacity(m);
        for i in (0..m).rev() {
            // alpha_i = rho_i * s_i^T * q
            let rho_i = rho_hist[i];
            let s_i_dot_q = self.compute_dot_product(&s_hist[i], &q)?;
            let alpha_i = rho_i * s_i_dot_q;
            alphas.push(alpha_i);

            // q = q - alpha_i * y_i
            let alpha_y = y_hist[i].mul(&Tensor::from_scalar(alpha_i))?;
            q = q.sub(&alpha_y)?;
        }
        alphas.reverse(); // Reverse to match loop order

        // Apply initial Hessian approximation H_0 = gamma * I
        // gamma = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
        let gamma = if let (Some(last_s), Some(last_y)) = (s_hist.last(), y_hist.last()) {
            let s_dot_y = self.compute_dot_product(last_s, last_y)?;
            let y_dot_y = self.compute_dot_product(last_y, last_y)?;
            if y_dot_y > T::zero() {
                s_dot_y / y_dot_y
            } else {
                T::one()
            }
        } else {
            T::one()
        };

        let mut r = q.mul(&Tensor::from_scalar(gamma))?;

        // Second loop (forward)
        for i in 0..m {
            // beta = rho_i * y_i^T * r
            let rho_i = rho_hist[i];
            let y_i_dot_r = self.compute_dot_product(&y_hist[i], &r)?;
            let beta = rho_i * y_i_dot_r;

            // r = r + s_i * (alpha_i - beta)
            let diff = alphas[i] - beta;
            let s_i_scaled = s_hist[i].mul(&Tensor::from_scalar(diff))?;
            r = r.add(&s_i_scaled)?;
        }

        // Return negative direction for minimization
        r.mul(&Tensor::from_scalar(-T::one()))
    }

    /// Update L-BFGS history with new parameter and gradient differences
    fn update_history(
        &mut self,
        param_ptr: *const Tensor<T>,
        current_param: &Tensor<T>,
        prev_param: &Tensor<T>,
        current_grad: &Tensor<T>,
        prev_grad: &Tensor<T>,
    ) -> Result<()> {
        // Compute s_k = x_{k+1} - x_k
        let s_k = current_param.sub(prev_param)?;

        // Compute y_k = g_{k+1} - g_k
        let y_k = current_grad.sub(prev_grad)?;

        // Compute rho_k = 1 / (y_k^T s_k)
        let y_dot_s = self.compute_dot_product(&y_k, &s_k)?;

        // Skip update if y_k^T s_k <= 0 (curvature condition)
        if y_dot_s <= T::zero() {
            return Ok(());
        }

        let rho_k = T::one() / y_dot_s;

        // Get or create history vectors
        let s_hist = self.s_history.entry(param_ptr).or_default();
        let y_hist = self.y_history.entry(param_ptr).or_default();
        let rho_hist = self.rho_history.entry(param_ptr).or_default();

        // Add new entries
        s_hist.push(s_k);
        y_hist.push(y_k);
        rho_hist.push(rho_k);

        // Maintain history size limit
        if s_hist.len() > self.history_size {
            s_hist.remove(0);
            y_hist.remove(0);
            rho_hist.remove(0);
        }

        Ok(())
    }

    /// Compute dot product of two tensors (sum of element-wise products)
    fn compute_dot_product(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<T> {
        if a.shape() != b.shape() {
            return Err(TensorError::shape_mismatch(
                "dot_product",
                &format!("{:?}", a.shape()),
                &format!("{:?}", b.shape()),
            ));
        }

        // Element-wise multiplication then sum
        let product = a.mul(b)?;
        let sum = product.sum(None, false)?;

        // Extract scalar value
        if sum.shape().rank() == 0 {
            // Scalar tensor - extract the single value
            let data = sum.to_vec()?;
            if !data.is_empty() {
                Ok(data[0])
            } else {
                Ok(T::zero())
            }
        } else {
            // Should not happen if sum works correctly, but handle gracefully
            Ok(T::zero())
        }
    }

    /// Compute gradient norm for convergence checking
    fn compute_gradient_norm(&self, grad: &Tensor<T>) -> Result<T> {
        let grad_squared = grad.mul(grad)?;
        let sum_squared = grad_squared.sum(None, false)?;

        if sum_squared.shape().rank() == 0 {
            let data = sum_squared.to_vec()?;
            if !data.is_empty() {
                Ok(data[0].sqrt())
            } else {
                Ok(T::zero())
            }
        } else {
            Ok(T::zero())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::dense::Dense;
    use crate::model::sequential::Sequential;
    use tenflowers_core::Tensor;

    #[test]
    fn test_lbfgs_creation() {
        let optimizer = LBFGS::<f32>::new(1.0);
        assert_eq!(optimizer.get_learning_rate(), 1.0);
        assert_eq!(optimizer.step_count(), 0);
        assert_eq!(optimizer.eval_count(), 0);
    }

    #[test]
    fn test_lbfgs_with_options() {
        let optimizer = LBFGS::<f32>::new(0.1)
            .with_max_iter(50)
            .with_max_eval(100)
            .with_tolerance_grad(1e-6)
            .with_tolerance_change(1e-8)
            .with_history_size(20);

        assert_eq!(optimizer.get_learning_rate(), 0.1);
        assert_eq!(optimizer.max_iter, 50);
        assert_eq!(optimizer.max_eval, Some(100));
        assert_eq!(optimizer.tolerance_grad, 1e-6);
        assert_eq!(optimizer.tolerance_change, 1e-8);
        assert_eq!(optimizer.history_size, 20);
    }

    #[test]
    fn test_lbfgs_convergence_checking() {
        let mut optimizer = LBFGS::<f32>::new(1.0).with_max_iter(2);

        assert!(!optimizer.has_converged());

        // Simulate reaching max iterations
        optimizer.step_count = 2;
        assert!(optimizer.has_converged());
    }

    #[test]
    fn test_lbfgs_loss_tracking() {
        let mut optimizer = LBFGS::<f32>::new(1.0).with_tolerance_change(0.1);

        optimizer.set_loss(1.0);
        assert_eq!(optimizer.current_loss, Some(1.0));
        assert_eq!(optimizer.eval_count(), 1);

        optimizer.set_loss(0.95);
        assert_eq!(optimizer.prev_loss, Some(1.0));
        assert_eq!(optimizer.current_loss, Some(0.95));
        assert_eq!(optimizer.eval_count(), 2);

        // Should converge (change = 0.05 < tolerance = 0.1)
        assert!(optimizer.has_converged());

        optimizer.set_loss(0.949);
        // Now change = 0.001 < tolerance = 0.1, should converge
        assert!(optimizer.has_converged());
    }

    #[test]
    fn test_lbfgs_basic_step() {
        // Create a simple model with one parameter
        let mut model = Sequential::<f32>::new(vec![]);
        model = model.add(Box::new(Dense::new(2, 1, true)));

        // Create some dummy gradients
        for param in model.parameters_mut() {
            let grad = Tensor::ones(param.shape().dims());
            param.set_grad(Some(grad));
        }

        let mut optimizer = LBFGS::<f32>::new(0.1);

        // First step should work (steepest descent)
        assert!(optimizer.step(&mut model).is_ok());
        assert_eq!(optimizer.step_count(), 1);
    }
}
