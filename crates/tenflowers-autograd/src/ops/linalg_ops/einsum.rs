use num_traits::{One, Zero};
use tenflowers_core::ops::broadcast_to;
use tenflowers_core::{Result, Tensor, TensorError};

/// Computes gradients for all inputs to an einsum operation
pub fn einsum_backward<T>(
    grad_output: &Tensor<T>,
    equation: &str,
    input_tensors: &[&Tensor<T>],
    _input_shapes: &[Vec<usize>],
) -> Result<Vec<Tensor<T>>>
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
    if input_tensors.is_empty() {
        return Err(TensorError::other(
            "No input tensors provided for einsum gradient".to_string(),
        ));
    }

    // Parse the einsum equation to understand the operation
    let (input_subscripts, output_subscript) = parse_einsum_equation(equation)?;

    if input_subscripts.len() != input_tensors.len() {
        return Err(TensorError::other(format!(
            "Equation expects {} inputs but {} provided",
            input_subscripts.len(),
            input_tensors.len()
        )));
    }

    let mut gradients = Vec::new();

    // Compute gradient for each input tensor
    for (i, _input_tensor) in input_tensors.iter().enumerate() {
        let grad_input = einsum_backward_single(
            grad_output,
            equation,
            i,
            input_tensors,
            &input_subscripts,
            &output_subscript,
        )?;
        gradients.push(grad_input);
    }

    Ok(gradients)
}

/// Compute gradient for a single input in an einsum operation
fn einsum_backward_single<T>(
    grad_output: &Tensor<T>,
    equation: &str,
    input_index: usize,
    input_tensors: &[&Tensor<T>],
    input_subscripts: &[String],
    output_subscript: &str,
) -> Result<Tensor<T>>
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
    // For einsum gradient, we need to construct a new einsum equation
    // that computes the gradient with respect to the input at input_index

    let target_subscript = &input_subscripts[input_index];

    // Handle specific common cases with optimized backward equations
    match input_tensors.len() {
        1 => {
            // Unary operations like transpose: "ij->ji"
            // The gradient flows back with the reverse operation
            if equation == "ij->ji" {
                tenflowers_core::ops::einsum("ji->ij", &[grad_output])
            } else if equation == "ij->" {
                // Sum operation: broadcast gradient to input shape
                let input_shape = input_tensors[0].shape().dims();
                broadcast_to(grad_output, input_shape)
            } else {
                // General case: reverse the transformation
                let backward_equation = format!("{output_subscript}->{target_subscript}");
                tenflowers_core::ops::einsum(&backward_equation, &[grad_output])
            }
        }
        2 => {
            // Binary operations like matrix multiplication: "ij,jk->ik"
            // Handle specific known patterns
            match equation {
                "ij,jk->ik" => {
                    // Matrix multiplication gradient
                    if input_index == 0 {
                        // grad_A = grad_output @ B^T = "ik,kj->ij"
                        let b_tensor = input_tensors[1];
                        tenflowers_core::ops::einsum("ik,kj->ij", &[grad_output, b_tensor])
                    } else {
                        // grad_B = A^T @ grad_output = "ji,ik->jk"
                        let a_tensor = input_tensors[0];
                        tenflowers_core::ops::einsum("ji,ik->jk", &[a_tensor, grad_output])
                    }
                }
                "ij,ij->ij" => {
                    // Element-wise multiplication gradient
                    if input_index == 0 {
                        // grad_A = grad_output * B = "ij,ij->ij"
                        let b_tensor = input_tensors[1];
                        tenflowers_core::ops::einsum("ij,ij->ij", &[grad_output, b_tensor])
                    } else {
                        // grad_B = grad_output * A = "ij,ij->ij"
                        let a_tensor = input_tensors[0];
                        tenflowers_core::ops::einsum("ij,ij->ij", &[grad_output, a_tensor])
                    }
                }
                "ij,ij->" => {
                    // Dot product gradient
                    if input_index == 0 {
                        // grad_A = grad_output * B, broadcast grad_output to input shape
                        let b_tensor = input_tensors[1];
                        let broadcasted_grad = broadcast_to(grad_output, b_tensor.shape().dims())?;
                        tenflowers_core::ops::einsum("ij,ij->ij", &[&broadcasted_grad, b_tensor])
                    } else {
                        // grad_B = grad_output * A, broadcast grad_output to input shape
                        let a_tensor = input_tensors[0];
                        let broadcasted_grad = broadcast_to(grad_output, a_tensor.shape().dims())?;
                        tenflowers_core::ops::einsum("ij,ij->ij", &[&broadcasted_grad, a_tensor])
                    }
                }
                _ => {
                    // Fall back to general case
                    if input_index == 0 {
                        let other_tensor = input_tensors[1];
                        let other_subscript = &input_subscripts[1];
                        let backward_equation = construct_binary_backward_equation(
                            output_subscript,
                            other_subscript,
                            target_subscript,
                            true,
                        )?;
                        tenflowers_core::ops::einsum(
                            &backward_equation,
                            &[grad_output, other_tensor],
                        )
                    } else {
                        let other_tensor = input_tensors[0];
                        let other_subscript = &input_subscripts[0];
                        let backward_equation = construct_binary_backward_equation(
                            other_subscript,
                            output_subscript,
                            target_subscript,
                            false,
                        )?;
                        tenflowers_core::ops::einsum(
                            &backward_equation,
                            &[other_tensor, grad_output],
                        )
                    }
                }
            }
        }
        _ => {
            // Multi-operand einsum - fall back to general case
            let mut other_subscripts = Vec::new();
            let mut other_tensors = Vec::new();

            // Add the gradient output with output subscript
            other_subscripts.push(output_subscript.to_string());
            other_tensors.push(grad_output);

            // Add other input tensors (not the one we're computing gradient for)
            for (i, (subscript, tensor)) in input_subscripts
                .iter()
                .zip(input_tensors.iter())
                .enumerate()
            {
                if i != input_index {
                    other_subscripts.push(subscript.clone());
                    other_tensors.push(*tensor);
                }
            }

            // Construct the backward einsum equation
            let backward_equation =
                construct_backward_equation(&other_subscripts, target_subscript)?;

            // Execute the backward einsum
            tenflowers_core::ops::einsum(&backward_equation, &other_tensors)
        }
    }
}

/// Construct the backward einsum equation for binary operations
fn construct_binary_backward_equation(
    first_subscript: &str,
    second_subscript: &str,
    target_subscript: &str,
    first_operand: bool,
) -> Result<String> {
    // For binary einsum backward pass, we need to construct the correct contraction
    // Examples:
    // - Forward: "ij,jk->ik", target: "ij", other: "jk", output: "ik"
    //   Backward for first: "ik,kj->ij" (grad_output @ other^T)
    // - Forward: "ij,jk->ik", target: "jk", other: "ij", output: "ik"
    //   Backward for second: "ji,ik->jk" (other^T @ grad_output)

    if first_operand {
        // We're computing gradient w.r.t. the first operand
        // We need to contract grad_output (first_subscript) with other tensor (second_subscript)
        // to produce target shape (target_subscript)

        // Find shared indices between output and other tensor
        let _output_chars: Vec<char> = first_subscript.chars().collect();
        let other_chars: Vec<char> = second_subscript.chars().collect();
        let _target_chars: Vec<char> = target_subscript.chars().collect();

        // Build the backward equation by determining the contraction pattern
        // For matrix multiplication ij,jk->ik: gradient ik,kj->ij
        let mut other_modified = String::new();
        for c in other_chars.iter().rev() {
            other_modified.push(*c);
        }

        Ok(format!(
            "{first_subscript},{other_modified}->{target_subscript}"
        ))
    } else {
        // We're computing gradient w.r.t. the second operand
        // For matrix multiplication ij,jk->ik: gradient ji,ik->jk
        let mut first_modified = String::new();
        for c in first_subscript.chars().rev() {
            first_modified.push(c);
        }

        Ok(format!(
            "{first_modified},{second_subscript}->{target_subscript}"
        ))
    }
}

/// Construct the backward einsum equation for gradient computation
fn construct_backward_equation(
    other_subscripts: &[String],
    target_subscript: &str,
) -> Result<String> {
    if other_subscripts.is_empty() {
        return Err(TensorError::other(
            "No other subscripts provided".to_string(),
        ));
    }

    // Join input subscripts with commas
    let input_part = other_subscripts.join(",");

    // The backward equation is: other_inputs -> target_output
    Ok(format!("{input_part}->{target_subscript}"))
}

/// Parse einsum equation like "ij,jk->ik" into input and output subscripts
fn parse_einsum_equation(equation: &str) -> Result<(Vec<String>, String)> {
    let parts: Vec<&str> = equation.split("->").collect();
    if parts.len() != 2 {
        return Err(TensorError::other(format!(
            "Invalid einsum equation: expected exactly one '->' separator, got {}",
            parts.len() - 1
        )));
    }

    let input_part = parts[0];
    let output_part = parts[1];

    // Split input subscripts by comma
    let input_subscripts: Vec<String> = input_part
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if input_subscripts.is_empty() {
        return Err(TensorError::other(
            "No input subscripts found in einsum equation".to_string(),
        ));
    }

    Ok((input_subscripts, output_part.trim().to_string()))
}
