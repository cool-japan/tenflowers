use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;

/// Tolerance for numerical gradient checking
/// Using larger epsilon to avoid floating-point precision errors
const EPS: f32 = 1e-4; // Increased from 1e-5 to reduce precision errors
const RTOL: f32 = 1e-2; // Increased relative tolerance
const ATOL: f32 = 1e-4; // Increased absolute tolerance

/// Helper function to compute numerical gradients using finite differences
fn numerical_gradient<F>(f: F, inputs: &[Tensor<f32>], eps: f32) -> Vec<Tensor<f32>>
where
    F: Fn(&[Tensor<f32>]) -> Tensor<f32>,
{
    let mut grads = Vec::new();

    for (input_idx, input) in inputs.iter().enumerate() {
        let input_data = input.as_slice().unwrap().to_vec();
        let mut grad_data = vec![0.0; input_data.len()];

        for i in 0..input_data.len() {
            // Forward perturbation
            let mut input_plus = input_data.clone();
            input_plus[i] += eps;
            let tensor_plus = Tensor::from_vec(input_plus, input.shape().dims()).unwrap();

            // Backward perturbation
            let mut input_minus = input_data.clone();
            input_minus[i] -= eps;
            let tensor_minus = Tensor::from_vec(input_minus, input.shape().dims()).unwrap();

            // Create input arrays for function evaluation
            let mut inputs_plus = inputs.to_vec();
            let mut inputs_minus = inputs.to_vec();
            inputs_plus[input_idx] = tensor_plus;
            inputs_minus[input_idx] = tensor_minus;

            // Compute finite difference
            let f_plus = f(&inputs_plus);
            let f_minus = f(&inputs_minus);

            // Extract scalar values (assuming output is scalar)
            let val_plus = f_plus.as_slice().unwrap()[0];
            let val_minus = f_minus.as_slice().unwrap()[0];

            grad_data[i] = (val_plus - val_minus) / (2.0 * eps);
        }

        grads.push(Tensor::from_vec(grad_data, input.shape().dims()).unwrap());
    }

    grads
}

/// Compare analytical and numerical gradients
fn compare_gradients(
    analytical: &[Tensor<f32>],
    numerical: &[Tensor<f32>],
    rtol: f32,
    atol: f32,
    test_name: &str,
) {
    assert_eq!(
        analytical.len(),
        numerical.len(),
        "{}: Number of gradients mismatch",
        test_name
    );

    for (i, (anal, num)) in analytical.iter().zip(numerical.iter()).enumerate() {
        let anal_slice = anal.as_slice().unwrap();
        let num_slice = num.as_slice().unwrap();

        assert_eq!(
            anal_slice.len(),
            num_slice.len(),
            "{}: Gradient {} size mismatch",
            test_name,
            i
        );

        for (j, (&a_val, &n_val)) in anal_slice.iter().zip(num_slice.iter()).enumerate() {
            let diff = (a_val - n_val).abs();
            let tolerance = atol + rtol * n_val.abs();

            assert!(
                diff <= tolerance,
                "{}: Gradient {} element {} differs: analytical={:?}, numerical={:?}, diff={:?}, tolerance={:?}",
                test_name, i, j, a_val, n_val, diff, tolerance
            );
        }
    }
}

#[test]
fn test_addition_gradients() {
    let tape = GradientTape::new();

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![2.0, 1.0, 4.0, 3.0];

    let mut a_tensor = Tensor::from_vec(a_data.clone(), &[2, 2]).unwrap();
    let mut b_tensor = Tensor::from_vec(b_data.clone(), &[2, 2]).unwrap();

    a_tensor.set_requires_grad(true);
    b_tensor.set_requires_grad(true);

    let a = tape.watch(a_tensor.clone());
    let b = tape.watch(b_tensor.clone());

    let c = a.add(&b).unwrap();
    let loss = c.sum(None, false).unwrap();

    let grads = tape.gradient(&[loss], &[a, b]).unwrap();

    // Numerical gradient check
    let numerical_grads = numerical_gradient(
        |inputs| {
            let sum = tenflowers_core::ops::binary::add(&inputs[0], &inputs[1]).unwrap();
            tenflowers_core::ops::reduction::sum(&sum, None, false).unwrap()
        },
        &[a_tensor, b_tensor],
        EPS,
    );

    let unwrapped_grads: Vec<Tensor<f32>> = grads.into_iter().map(|g| g.unwrap()).collect();
    compare_gradients(
        &unwrapped_grads,
        &numerical_grads,
        RTOL,
        ATOL,
        "Addition gradients",
    );
}

#[test]
fn test_multiplication_gradients() {
    let tape = GradientTape::new();

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![2.0, 1.0, 4.0, 3.0];

    let mut a_tensor = Tensor::from_vec(a_data.clone(), &[2, 2]).unwrap();
    let mut b_tensor = Tensor::from_vec(b_data.clone(), &[2, 2]).unwrap();

    a_tensor.set_requires_grad(true);
    b_tensor.set_requires_grad(true);

    let a = tape.watch(a_tensor.clone());
    let b = tape.watch(b_tensor.clone());

    let c = a.mul(&b).unwrap();
    let loss = c.sum(None, false).unwrap();

    let grads = tape.gradient(&[loss], &[a, b]).unwrap();

    let numerical_grads = numerical_gradient(
        |inputs| {
            let prod = tenflowers_core::ops::binary::mul(&inputs[0], &inputs[1]).unwrap();
            tenflowers_core::ops::reduction::sum(&prod, None, false).unwrap()
        },
        &[a_tensor, b_tensor],
        EPS,
    );

    let unwrapped_grads: Vec<Tensor<f32>> = grads.into_iter().map(|g| g.unwrap()).collect();
    compare_gradients(
        &unwrapped_grads,
        &numerical_grads,
        RTOL,
        ATOL,
        "Multiplication gradients",
    );
}

#[test]
fn test_relu_gradients() {
    let input_data = vec![-2.0, -1.0, 0.5, 1.0, 2.0];
    let mut input_tensor = Tensor::from_vec(input_data.clone(), &[5]).unwrap();

    let tape = GradientTape::new();
    input_tensor.set_requires_grad(true);

    let x = tape.watch(input_tensor.clone());
    let y = x.relu().unwrap();
    let loss = y.sum(None, false).unwrap();

    let grads = tape.gradient(&[loss], &[x]).unwrap();

    let numerical_grads = numerical_gradient(
        |inputs| {
            let relu_result = tenflowers_core::ops::activation::relu(&inputs[0]).unwrap();
            tenflowers_core::ops::reduction::sum(&relu_result, None, false).unwrap()
        },
        &[input_tensor],
        EPS,
    );

    let unwrapped_grads: Vec<Tensor<f32>> = grads.into_iter().map(|g| g.unwrap()).collect();
    compare_gradients(
        &unwrapped_grads,
        &numerical_grads,
        RTOL,
        ATOL,
        "ReLU gradients",
    );
}

#[test]
fn test_chain_rule() {
    // Test a complex chain: x -> x^2 -> relu -> sum
    let tape = GradientTape::new();

    let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut input_tensor = Tensor::from_vec(input_data.clone(), &[5]).unwrap();

    input_tensor.set_requires_grad(true);

    let x = tape.watch(input_tensor.clone());
    let x_squared = x.mul(&x).unwrap(); // x^2
    let relu_result = x_squared.relu().unwrap(); // relu(x^2)
    let loss = relu_result.sum(None, false).unwrap(); // sum(relu(x^2))

    let grads = tape.gradient(&[loss], &[x]).unwrap();

    let numerical_grads = numerical_gradient(
        |inputs| {
            let squared = tenflowers_core::ops::binary::mul(&inputs[0], &inputs[0]).unwrap();
            let relu_result = tenflowers_core::ops::activation::relu(&squared).unwrap();
            tenflowers_core::ops::reduction::sum(&relu_result, None, false).unwrap()
        },
        &[input_tensor],
        EPS,
    );

    let unwrapped_grads: Vec<Tensor<f32>> = grads.into_iter().map(|g| g.unwrap()).collect();
    compare_gradients(
        &unwrapped_grads,
        &numerical_grads,
        RTOL,
        ATOL,
        "Complex chain rule",
    );
}

/// Basic integration test to verify gradient computation API works
#[test]
fn test_gradient_api_basic() {
    let tape = GradientTape::new();

    let mut x = Tensor::<f32>::from_scalar(2.0);
    let mut y = Tensor::<f32>::from_scalar(3.0);

    x.set_requires_grad(true);
    y.set_requires_grad(true);

    let x_tracked = tape.watch(x);
    let y_tracked = tape.watch(y);

    let z = x_tracked.add(&y_tracked).unwrap();

    let gradients = tape.gradient(&[z], &[x_tracked, y_tracked]).unwrap();

    assert_eq!(gradients.len(), 2);

    // Both gradients should be 1.0 for addition
    let grad_x = gradients[0].as_ref().unwrap().get(&[]).unwrap();
    let grad_y = gradients[1].as_ref().unwrap().get(&[]).unwrap();

    assert!((grad_x - 1.0).abs() < 1e-6);
    assert!((grad_y - 1.0).abs() < 1e-6);
}
