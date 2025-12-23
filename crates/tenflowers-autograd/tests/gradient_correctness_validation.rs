use scirs2_core::ndarray::{Array1, Array2, Array3, Array4, ArrayD};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::process::Command;
use tenflowers_autograd::{GradientTape, TrackedTensor};
use tenflowers_core::{DType, Tensor};

/// Test result comparing TenfloweRS gradient with reference implementation
#[derive(Debug, Clone)]
pub struct GradientComparisonResult {
    pub operation: String,
    pub input_shape: Vec<usize>,
    pub tenflowers_gradient: Vec<f32>,
    pub reference_gradient: Vec<f32>,
    pub max_absolute_error: f64,
    pub mean_absolute_error: f64,
    pub relative_error: f64,
    pub passed: bool,
    pub tolerance: f64,
}

impl GradientComparisonResult {
    pub fn new(
        operation: String,
        input_shape: Vec<usize>,
        tenflowers_grad: &[f32],
        reference_grad: &[f32],
        tolerance: f64,
    ) -> Self {
        let max_abs_error = tenflowers_grad
            .iter()
            .zip(reference_grad.iter())
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0f64, f64::max);

        let mean_abs_error = tenflowers_grad
            .iter()
            .zip(reference_grad.iter())
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / tenflowers_grad.len() as f64;

        let rel_error = if reference_grad.iter().any(|&x| x.abs() > 1e-8) {
            let reference_norm: f64 = reference_grad
                .iter()
                .map(|&x| (x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            max_abs_error / reference_norm
        } else {
            max_abs_error
        };

        let passed = max_abs_error <= tolerance && rel_error <= tolerance;

        Self {
            operation,
            input_shape,
            tenflowers_gradient: tenflowers_grad.to_vec(),
            reference_gradient: reference_grad.to_vec(),
            max_absolute_error: max_abs_error,
            mean_absolute_error: mean_abs_error,
            relative_error: rel_error,
            passed,
            tolerance,
        }
    }
}

/// Gradient correctness test suite for comparing with PyTorch
pub struct GradientCorrectnessTests {
    tolerance: f64,
    results: Vec<GradientComparisonResult>,
}

impl GradientCorrectnessTests {
    pub fn new(tolerance: f64) -> Self {
        Self {
            tolerance,
            results: Vec::new(),
        }
    }

    /// Test basic arithmetic operations
    pub fn test_basic_arithmetic(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Testing basic arithmetic operations...");

        // Test addition
        self.test_addition()?;

        // Test multiplication
        self.test_multiplication()?;

        // Test subtraction
        self.test_subtraction()?;

        // Test division
        self.test_division()?;

        // Test power
        self.test_power()?;

        Ok(())
    }

    fn test_addition(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let shapes = vec![vec![10], vec![5, 4], vec![2, 3, 4]];

        for shape in shapes {
            // TenfloweRS computation
            let tape = GradientTape::new();
            let x_data = create_test_data(&shape, 0.5, 2.0);
            let y_data = create_test_data(&shape, 1.0, 3.0);

            let x = tape.watch(Tensor::from_array(x_data.clone()));
            let y = tape.watch(Tensor::from_array(y_data.clone()));
            let z = x.add(&y)?;
            let loss = z.sum(None, false)?;

            let grads = tape.gradient(&[loss], &[x, y])?;
            let x_grad_tf = tensor_to_vec(grads[0].as_ref().unwrap())?;
            let y_grad_tf = tensor_to_vec(grads[1].as_ref().unwrap())?;

            // PyTorch computation via Python script
            let (x_grad_pt, y_grad_pt) =
                self.compute_pytorch_gradient_binary_op(&shape, &x_data, &y_data, "add")?;

            // Compare results
            let x_result = GradientComparisonResult::new(
                format!("add_x_{:?}", shape),
                shape.clone(),
                &x_grad_tf,
                &x_grad_pt,
                self.tolerance,
            );

            let y_result = GradientComparisonResult::new(
                format!("add_y_{:?}", shape),
                shape.clone(),
                &y_grad_tf,
                &y_grad_pt,
                self.tolerance,
            );

            println!(
                "  Add x gradient for shape {:?}: {} (max_err: {:.2e})",
                shape,
                if x_result.passed { "PASS" } else { "FAIL" },
                x_result.max_absolute_error
            );
            println!(
                "  Add y gradient for shape {:?}: {} (max_err: {:.2e})",
                shape,
                if y_result.passed { "PASS" } else { "FAIL" },
                y_result.max_absolute_error
            );

            self.results.push(x_result);
            self.results.push(y_result);
        }

        Ok(())
    }

    fn test_multiplication(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let shapes = vec![vec![10], vec![5, 4], vec![2, 3, 4]];

        for shape in shapes {
            let tape = GradientTape::new();
            let x_data = create_test_data(&shape, 0.5, 2.0);
            let y_data = create_test_data(&shape, 1.0, 3.0);

            let x = tape.watch(Tensor::from_array(x_data.clone()));
            let y = tape.watch(Tensor::from_array(y_data.clone()));
            let z = x.mul(&y)?;
            let loss = z.sum(None, false)?;

            let grads = tape.gradient(&[loss], &[x, y])?;
            let x_grad_tf = tensor_to_vec(grads[0].as_ref().unwrap())?;
            let y_grad_tf = tensor_to_vec(grads[1].as_ref().unwrap())?;

            let (x_grad_pt, y_grad_pt) =
                self.compute_pytorch_gradient_binary_op(&shape, &x_data, &y_data, "mul")?;

            let x_result = GradientComparisonResult::new(
                format!("mul_x_{:?}", shape),
                shape.clone(),
                &x_grad_tf,
                &x_grad_pt,
                self.tolerance,
            );

            let y_result = GradientComparisonResult::new(
                format!("mul_y_{:?}", shape),
                shape.clone(),
                &y_grad_tf,
                &y_grad_pt,
                self.tolerance,
            );

            println!(
                "  Mul x gradient for shape {:?}: {} (max_err: {:.2e})",
                shape,
                if x_result.passed { "PASS" } else { "FAIL" },
                x_result.max_absolute_error
            );
            println!(
                "  Mul y gradient for shape {:?}: {} (max_err: {:.2e})",
                shape,
                if y_result.passed { "PASS" } else { "FAIL" },
                y_result.max_absolute_error
            );

            self.results.push(x_result);
            self.results.push(y_result);
        }

        Ok(())
    }

    fn test_subtraction(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let shapes = vec![vec![10], vec![5, 4]];

        for shape in shapes {
            let tape = GradientTape::new();
            let x_data = create_test_data(&shape, 2.0, 4.0);
            let y_data = create_test_data(&shape, 0.5, 1.5);

            let x = tape.watch(Tensor::from_array(x_data.clone()));
            let y = tape.watch(Tensor::from_array(y_data.clone()));
            let z = x.sub(&y)?;
            let loss = z.sum(None, false)?;

            let grads = tape.gradient(&[loss], &[x, y])?;
            let x_grad_tf = tensor_to_vec(grads[0].as_ref().unwrap())?;
            let y_grad_tf = tensor_to_vec(grads[1].as_ref().unwrap())?;

            let (x_grad_pt, y_grad_pt) =
                self.compute_pytorch_gradient_binary_op(&shape, &x_data, &y_data, "sub")?;

            let x_result = GradientComparisonResult::new(
                format!("sub_x_{:?}", shape),
                shape.clone(),
                &x_grad_tf,
                &x_grad_pt,
                self.tolerance,
            );

            let y_result = GradientComparisonResult::new(
                format!("sub_y_{:?}", shape),
                shape.clone(),
                &y_grad_tf,
                &y_grad_pt,
                self.tolerance,
            );

            println!(
                "  Sub x gradient for shape {:?}: {} (max_err: {:.2e})",
                shape,
                if x_result.passed { "PASS" } else { "FAIL" },
                x_result.max_absolute_error
            );
            println!(
                "  Sub y gradient for shape {:?}: {} (max_err: {:.2e})",
                shape,
                if y_result.passed { "PASS" } else { "FAIL" },
                y_result.max_absolute_error
            );

            self.results.push(x_result);
            self.results.push(y_result);
        }

        Ok(())
    }

    fn test_division(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let shapes = vec![vec![10], vec![5, 4]];

        for shape in shapes {
            let tape = GradientTape::new();
            let x_data = create_test_data(&shape, 1.0, 3.0);
            let y_data = create_test_data(&shape, 0.5, 2.0); // Avoid division by zero

            let x = tape.watch(Tensor::from_array(x_data.clone()));
            let y = tape.watch(Tensor::from_array(y_data.clone()));
            let z = x.div(&y)?;
            let loss = z.sum(None, false)?;

            let grads = tape.gradient(&[loss], &[x, y])?;
            let x_grad_tf = tensor_to_vec(grads[0].as_ref().unwrap())?;
            let y_grad_tf = tensor_to_vec(grads[1].as_ref().unwrap())?;

            let (x_grad_pt, y_grad_pt) =
                self.compute_pytorch_gradient_binary_op(&shape, &x_data, &y_data, "div")?;

            let x_result = GradientComparisonResult::new(
                format!("div_x_{:?}", shape),
                shape.clone(),
                &x_grad_tf,
                &x_grad_pt,
                self.tolerance,
            );

            let y_result = GradientComparisonResult::new(
                format!("div_y_{:?}", shape),
                shape.clone(),
                &y_grad_tf,
                &y_grad_pt,
                self.tolerance,
            );

            println!(
                "  Div x gradient for shape {:?}: {} (max_err: {:.2e})",
                shape,
                if x_result.passed { "PASS" } else { "FAIL" },
                x_result.max_absolute_error
            );
            println!(
                "  Div y gradient for shape {:?}: {} (max_err: {:.2e})",
                shape,
                if y_result.passed { "PASS" } else { "FAIL" },
                y_result.max_absolute_error
            );

            self.results.push(x_result);
            self.results.push(y_result);
        }

        Ok(())
    }

    fn test_power(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let shapes = vec![vec![10], vec![5, 4]];

        for shape in shapes {
            let tape = GradientTape::new();
            let x_data = create_test_data(&shape, 0.5, 2.0);
            let exponent = 2.0f32;

            let x = tape.watch(Tensor::from_array(x_data.clone()));
            let exp_tensor = tape.watch(Tensor::from_scalar(exponent));
            let z = x.pow(&exp_tensor)?;
            let loss = z.sum(None, false)?;

            let grads = tape.gradient(&[loss], &[x])?;
            let x_grad_tf = tensor_to_vec(grads[0].as_ref().unwrap())?;

            let x_grad_pt =
                self.compute_pytorch_gradient_unary_op(&shape, &x_data, "pow", Some(exponent))?;

            let x_result = GradientComparisonResult::new(
                format!("pow_x_{:?}", shape),
                shape.clone(),
                &x_grad_tf,
                &x_grad_pt,
                self.tolerance,
            );

            println!(
                "  Pow gradient for shape {:?}: {} (max_err: {:.2e})",
                shape,
                if x_result.passed { "PASS" } else { "FAIL" },
                x_result.max_absolute_error
            );

            self.results.push(x_result);
        }

        Ok(())
    }

    /// Test activation functions
    pub fn test_activation_functions(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Testing activation functions...");

        let shapes = vec![vec![10], vec![5, 4]];
        let activations = vec!["relu", "sigmoid", "tanh"];

        for shape in &shapes {
            for &activation in &activations {
                let tape = GradientTape::new();
                let x_data = match activation {
                    "relu" => create_test_data(shape, -2.0, 2.0),
                    "sigmoid" => create_test_data(shape, -5.0, 5.0),
                    "tanh" => create_test_data(shape, -3.0, 3.0),
                    _ => create_test_data(shape, -1.0, 1.0),
                };

                let x = tape.watch(Tensor::from_array(x_data.clone()));
                let z = match activation {
                    "relu" => x.relu()?,
                    "sigmoid" => x.sigmoid()?,
                    "tanh" => x.tanh()?,
                    _ => return Err("Unknown activation".into()),
                };
                let loss = z.sum(None, false)?;

                let grads = tape.gradient(&[loss], &[x])?;
                let x_grad_tf = tensor_to_vec(grads[0].as_ref().unwrap())?;

                let x_grad_pt =
                    self.compute_pytorch_gradient_unary_op(shape, &x_data, activation, None)?;

                let result = GradientComparisonResult::new(
                    format!("{}_{:?}", activation, shape),
                    shape.clone(),
                    &x_grad_tf,
                    &x_grad_pt,
                    self.tolerance,
                );

                println!(
                    "  {} gradient for shape {:?}: {} (max_err: {:.2e})",
                    activation,
                    shape,
                    if result.passed { "PASS" } else { "FAIL" },
                    result.max_absolute_error
                );

                self.results.push(result);
            }
        }

        Ok(())
    }

    /// Test matrix operations
    pub fn test_matrix_operations(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Testing matrix operations...");

        // Test matrix multiplication
        self.test_matmul()?;

        // Test transpose
        self.test_transpose()?;

        Ok(())
    }

    fn test_matmul(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let configs = vec![(3, 4, 5), (2, 3, 2)];

        for (m, k, n) in configs {
            let tape = GradientTape::new();
            let x_data = create_test_data(&[m, k], 0.1, 2.0);
            let y_data = create_test_data(&[k, n], 0.1, 2.0);

            let x = tape.watch(Tensor::from_array(x_data.clone()));
            let y = tape.watch(Tensor::from_array(y_data.clone()));
            let z = x.matmul(&y)?;
            let loss = z.sum(None, false)?;

            let grads = tape.gradient(&[loss], &[x, y])?;
            let x_grad_tf = tensor_to_vec(grads[0].as_ref().unwrap())?;
            let y_grad_tf = tensor_to_vec(grads[1].as_ref().unwrap())?;

            let (x_grad_pt, y_grad_pt) =
                self.compute_pytorch_matmul_gradient(&[m, k], &x_data, &[k, n], &y_data)?;

            let x_result = GradientComparisonResult::new(
                format!("matmul_x_{}x{}x{}", m, k, n),
                vec![m, k],
                &x_grad_tf,
                &x_grad_pt,
                self.tolerance,
            );

            let y_result = GradientComparisonResult::new(
                format!("matmul_y_{}x{}x{}", m, k, n),
                vec![k, n],
                &y_grad_tf,
                &y_grad_pt,
                self.tolerance,
            );

            println!(
                "  Matmul x gradient {}x{}x{}: {} (max_err: {:.2e})",
                m,
                k,
                n,
                if x_result.passed { "PASS" } else { "FAIL" },
                x_result.max_absolute_error
            );
            println!(
                "  Matmul y gradient {}x{}x{}: {} (max_err: {:.2e})",
                m,
                k,
                n,
                if y_result.passed { "PASS" } else { "FAIL" },
                y_result.max_absolute_error
            );

            self.results.push(x_result);
            self.results.push(y_result);
        }

        Ok(())
    }

    fn test_transpose(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let shapes = vec![vec![3, 4], vec![5, 2]];

        for shape in shapes {
            let tape = GradientTape::new();
            let x_data = create_test_data(&shape, 0.1, 2.0);

            let x = tape.watch(Tensor::from_array(x_data.clone()));
            let z = x.transpose(None)?;
            let loss = z.sum(None, false)?;

            let grads = tape.gradient(&[loss], &[x])?;
            let x_grad_tf = tensor_to_vec(grads[0].as_ref().unwrap())?;

            let x_grad_pt =
                self.compute_pytorch_gradient_unary_op(&shape, &x_data, "transpose", None)?;

            let result = GradientComparisonResult::new(
                format!("transpose_{:?}", shape),
                shape.clone(),
                &x_grad_tf,
                &x_grad_pt,
                self.tolerance,
            );

            println!(
                "  Transpose gradient for shape {:?}: {} (max_err: {:.2e})",
                shape,
                if result.passed { "PASS" } else { "FAIL" },
                result.max_absolute_error
            );

            self.results.push(result);
        }

        Ok(())
    }

    /// Compute PyTorch gradient for binary operation via Python script
    fn compute_pytorch_gradient_binary_op(
        &self,
        shape: &[usize],
        x_data: &ArrayD<f32>,
        y_data: &ArrayD<f32>,
        operation: &str,
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        let script = format!(
            r#"
import torch
import json
import sys

# Create tensors
x_shape = {}
y_shape = {}
x_data = {}
y_data = {}

x = torch.tensor(x_data, dtype=torch.float32, requires_grad=True).reshape(x_shape)
y = torch.tensor(y_data, dtype=torch.float32, requires_grad=True).reshape(y_shape)

# Retain gradients for intermediate tensors
x.retain_grad()
y.retain_grad()

# Perform operation
if '{}' == 'add':
    z = x + y
elif '{}' == 'mul':
    z = x * y
elif '{}' == 'sub':
    z = x - y
elif '{}' == 'div':
    z = x / y
else:
    raise ValueError(f"Unknown operation: {}")

# Compute loss and gradients
loss = z.sum()
loss.backward()

# Return gradients
result = {{
    'x_grad': x.grad.flatten().tolist(),
    'y_grad': y.grad.flatten().tolist()
}}
print(json.dumps(result))
"#,
            serde_json::to_string(shape)?,
            serde_json::to_string(shape)?,
            serde_json::to_string(&array_to_vec(x_data))?,
            serde_json::to_string(&array_to_vec(y_data))?,
            operation,
            operation,
            operation,
            operation,
            operation
        );

        let output = Command::new("python3").arg("-c").arg(&script).output()?;

        if !output.status.success() {
            return Err(format!(
                "Python script failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        let result: Value = serde_json::from_slice(&output.stdout)?;
        let x_grad: Vec<f32> = serde_json::from_value(result["x_grad"].clone())?;
        let y_grad: Vec<f32> = serde_json::from_value(result["y_grad"].clone())?;

        Ok((x_grad, y_grad))
    }

    /// Compute PyTorch gradient for unary operation via Python script
    fn compute_pytorch_gradient_unary_op(
        &self,
        shape: &[usize],
        x_data: &ArrayD<f32>,
        operation: &str,
        param: Option<f32>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let param_str = param
            .map(|p| p.to_string())
            .unwrap_or_else(|| "None".to_string());

        let script = format!(
            r#"
import torch
import json

# Create tensor
x_shape = {}
x_data = {}
param = {}

x = torch.tensor(x_data, dtype=torch.float32, requires_grad=True).reshape(x_shape)

# Retain gradients for intermediate tensors
x.retain_grad()

# Perform operation
if '{}' == 'relu':
    z = torch.relu(x)
elif '{}' == 'sigmoid':
    z = torch.sigmoid(x)
elif '{}' == 'tanh':
    z = torch.tanh(x)
elif '{}' == 'transpose':
    z = x.t()
elif '{}' == 'pow':
    z = torch.pow(x, param)
else:
    raise ValueError(f"Unknown operation: {}")

# Compute loss and gradients
loss = z.sum()
loss.backward()

# Return gradient
result = x.grad.flatten().tolist()
print(json.dumps(result))
"#,
            serde_json::to_string(shape)?,
            serde_json::to_string(&array_to_vec(x_data))?,
            param_str,
            operation,
            operation,
            operation,
            operation,
            operation,
            operation
        );

        let output = Command::new("python3").arg("-c").arg(&script).output()?;

        if !output.status.success() {
            return Err(format!(
                "Python script failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        let result: Vec<f32> = serde_json::from_slice(&output.stdout)?;
        Ok(result)
    }

    /// Compute PyTorch gradient for matrix multiplication
    fn compute_pytorch_matmul_gradient(
        &self,
        x_shape: &[usize],
        x_data: &ArrayD<f32>,
        y_shape: &[usize],
        y_data: &ArrayD<f32>,
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        let script = format!(
            r#"
import torch
import json

# Create tensors
x_shape = {}
y_shape = {}
x_data = {}
y_data = {}

x = torch.tensor(x_data, dtype=torch.float32, requires_grad=True).reshape(x_shape)
y = torch.tensor(y_data, dtype=torch.float32, requires_grad=True).reshape(y_shape)

# Retain gradients for intermediate tensors
x.retain_grad()
y.retain_grad()

# Matrix multiplication
z = torch.matmul(x, y)

# Compute loss and gradients
loss = z.sum()
loss.backward()

# Return gradients
result = {{
    'x_grad': x.grad.flatten().tolist(),
    'y_grad': y.grad.flatten().tolist()
}}
print(json.dumps(result))
"#,
            serde_json::to_string(x_shape)?,
            serde_json::to_string(y_shape)?,
            serde_json::to_string(&array_to_vec(x_data))?,
            serde_json::to_string(&array_to_vec(y_data))?
        );

        let output = Command::new("python3").arg("-c").arg(&script).output()?;

        if !output.status.success() {
            return Err(format!(
                "Python script failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        let result: Value = serde_json::from_slice(&output.stdout)?;
        let x_grad: Vec<f32> = serde_json::from_value(result["x_grad"].clone())?;
        let y_grad: Vec<f32> = serde_json::from_value(result["y_grad"].clone())?;

        Ok((x_grad, y_grad))
    }

    /// Print summary of test results
    pub fn print_summary(&self) {
        let total_tests = self.results.len();
        let passed_tests = self.results.iter().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;

        println!(
            "\n╔══════════════════════════════════════════════════════════════════════════════╗"
        );
        println!(
            "║                        GRADIENT CORRECTNESS TEST RESULTS                        ║"
        );
        println!(
            "╠══════════════════════════════════════════════════════════════════════════════╣"
        );
        println!(
            "║ Total Tests:  {:<10} │ Passed: {:<10} │ Failed: {:<10}               ║",
            total_tests, passed_tests, failed_tests
        );
        println!(
            "║ Success Rate: {:<10.1}%                                                      ║",
            (passed_tests as f64 / total_tests as f64) * 100.0
        );
        println!(
            "╠══════════════════════════════════════════════════════════════════════════════╣"
        );

        if failed_tests > 0 {
            println!("║                                 FAILED TESTS                                    ║");
            println!(
                "╠══════════════════════════════════════════════════════════════════════════════╣"
            );

            for result in &self.results {
                if !result.passed {
                    println!(
                        "║ {:<30} │ Max Error: {:<12.2e} │ Rel Error: {:<12.2e} ║",
                        result.operation, result.max_absolute_error, result.relative_error
                    );
                }
            }
        }

        println!(
            "╚══════════════════════════════════════════════════════════════════════════════╝"
        );
    }

    /// Run all gradient correctness tests
    pub fn run_all_tests(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Starting TenfloweRS vs PyTorch gradient correctness validation...");
        println!("Tolerance: {:.2e}\n", self.tolerance);

        self.test_basic_arithmetic()?;
        self.test_activation_functions()?;
        self.test_matrix_operations()?;

        self.print_summary();
        Ok(())
    }
}

/// Helper function to create test data with specific range
fn create_test_data(shape: &[usize], min: f32, max: f32) -> ArrayD<f32> {
    let total_elements: usize = shape.iter().product();
    let mut data = Vec::with_capacity(total_elements);

    for i in 0..total_elements {
        let normalized = i as f32 / total_elements as f32;
        let value = min + normalized * (max - min);
        data.push(value);
    }

    ArrayD::from_shape_vec(shape, data).unwrap()
}

/// Convert ndarray to Vec<f32>
fn array_to_vec(arr: &ArrayD<f32>) -> Vec<f32> {
    arr.iter().cloned().collect()
}

/// Convert Tensor to Vec<f32>
fn tensor_to_vec(tensor: &Tensor<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Convert tensor data to vector using the as_slice method
    match tensor.as_slice() {
        Some(slice) => Ok(slice.to_vec()),
        None => {
            // For GPU tensors, we'd need to transfer to CPU first
            // For now, return an error since our test focuses on CPU
            Err("Cannot convert GPU tensor to vector directly".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "long-running"]
    fn test_gradient_correctness_basic() {
        let mut test_suite = GradientCorrectnessTests::new(1e-4);

        // Run a subset of tests to avoid long test times
        if let Err(e) = test_suite.test_basic_arithmetic() {
            println!("Test failed: {}", e);
            // Don't fail the test if PyTorch is not available
            if e.to_string().contains("python3") {
                println!("Skipping gradient correctness tests - PyTorch not available");
                return;
            }
            panic!("Gradient correctness test failed: {}", e);
        }

        let passed_tests = test_suite.results.iter().filter(|r| r.passed).count();
        let total_tests = test_suite.results.len();

        // Require at least 80% of tests to pass
        assert!(
            passed_tests as f64 / total_tests as f64 >= 0.8,
            "Too many gradient correctness tests failed: {}/{}",
            passed_tests,
            total_tests
        );
    }

    #[test]
    #[ignore = "long-running"]
    fn test_gradient_correctness_activations() {
        let mut test_suite = GradientCorrectnessTests::new(1e-4);

        if let Err(e) = test_suite.test_activation_functions() {
            if e.to_string().contains("python3") {
                println!("Skipping gradient correctness tests - PyTorch not available");
                return;
            }
            panic!("Activation gradient correctness test failed: {}", e);
        }

        let passed_tests = test_suite.results.iter().filter(|r| r.passed).count();
        let total_tests = test_suite.results.len();

        assert!(
            passed_tests as f64 / total_tests as f64 >= 0.8,
            "Too many activation gradient tests failed: {}/{}",
            passed_tests,
            total_tests
        );
    }

    #[test]
    #[ignore] // Ignore by default since it requires full test suite
    fn test_full_gradient_correctness_suite() {
        let mut test_suite = GradientCorrectnessTests::new(1e-4);

        if let Err(e) = test_suite.run_all_tests() {
            if e.to_string().contains("python3") {
                println!("Skipping gradient correctness tests - PyTorch not available");
                return;
            }
            panic!("Full gradient correctness test failed: {}", e);
        }

        let passed_tests = test_suite.results.iter().filter(|r| r.passed).count();
        let total_tests = test_suite.results.len();

        assert!(
            passed_tests as f64 / total_tests as f64 >= 0.9,
            "Too many gradient correctness tests failed: {}/{}",
            passed_tests,
            total_tests
        );
    }
}
