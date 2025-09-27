use scirs2_autograd::ndarray::{ArrayD, IxDyn};
/// Basic Tensor Operations Demo using TenfloweRS
/// Simple demonstration of tensor creation and basic operations
use tenflowers_core::{DType, Device, Tensor};
use tenflowers_neural::layers::{Dense, Layer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌻 TenfloweRS Basic Tensor Demo 🌻");
    println!("==================================");

    let device = Device::default();
    println!("Using device: {:?}", device);

    // Create simple tensors using ndarray
    println!("\n📊 Creating tensors...");

    // Create a 1D tensor
    let tensor_1d =
        Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0f32, 2.0, 3.0])?);
    println!("1D tensor shape: {:?}", tensor_1d.shape());
    println!("1D tensor data: {:?}", tensor_1d.to_vec());

    // Create a 2D tensor
    let tensor_2d = Tensor::from_array(ArrayD::from_shape_vec(
        IxDyn(&[2, 3]),
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    )?);
    println!("2D tensor shape: {:?}", tensor_2d.shape());
    println!("2D tensor data: {:?}", tensor_2d.to_vec());

    // Create zero tensor
    let zeros = Tensor::<f32>::zeros(&[2, 4]);
    println!("Zeros tensor shape: {:?}", zeros.shape());

    // Create ones tensor
    let ones = Tensor::<f32>::ones(&[3, 2]);
    println!("Ones tensor shape: {:?}", ones.shape());

    // Test basic tensor operations
    println!("\n⚡ Testing basic operations...");

    // Addition
    let a = Tensor::from_array(ArrayD::from_shape_vec(
        IxDyn(&[2, 2]),
        vec![1.0f32, 2.0, 3.0, 4.0],
    )?);
    let b = Tensor::from_array(ArrayD::from_shape_vec(
        IxDyn(&[2, 2]),
        vec![5.0f32, 6.0, 7.0, 8.0],
    )?);

    let sum = a.add(&b)?;
    println!("Tensor addition result: {:?}", sum.to_vec());

    // Element-wise multiplication
    let product = a.mul(&b)?;
    println!("Element-wise multiplication result: {:?}", product.to_vec());

    // Matrix multiplication
    let weights = Tensor::from_array(ArrayD::from_shape_vec(
        IxDyn(&[2, 3]),
        vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6],
    )?);
    let matmul_result = a.matmul(&weights)?;
    println!(
        "Matrix multiplication result shape: {:?}",
        matmul_result.shape()
    );
    println!("Matrix multiplication result: {:?}", matmul_result.to_vec());

    // Test dense layer
    println!("\n🧠 Testing dense layer...");
    let dense = Dense::new(2, 3, true); // 2 inputs -> 3 outputs with bias

    // Test forward pass
    let layer_input =
        Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0f32, 2.0])?);
    println!("Layer input shape: {:?}", layer_input.shape());

    let layer_output = dense.forward(&layer_input)?;
    println!("Layer output shape: {:?}", layer_output.shape());
    println!("Layer output: {:?}", layer_output.to_vec());

    // Demonstrate different data types
    println!("\n🔢 Testing different data types...");

    let int_tensor = Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[3]), vec![1i32, 2, 3])?);
    println!("Integer tensor: {:?}", int_tensor.to_vec());

    let float64_tensor =
        Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.14f64, 2.71])?);
    println!("Float64 tensor: {:?}", float64_tensor.to_vec());

    // Show tensor properties
    println!("\n📏 Tensor properties...");
    println!("Tensor 2D shape: {:?}", tensor_2d.shape());
    println!("Tensor 2D dtype: {:?}", tensor_2d.dtype());
    println!("Device: {:?}", tensor_2d.device());

    println!("\n✅ Basic tensor operations demo completed successfully!");
    println!("🌻 TenfloweRS tensor system is working correctly! 🌻");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() -> Result<(), Box<dyn std::error::Error>> {
        let tensor = Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0f32, 2.0])?);
        assert_eq!(tensor.shape(), &[2]);
        assert_eq!(tensor.to_vec(), vec![1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_operations() -> Result<(), Box<dyn std::error::Error>> {
        let a = Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0f32, 2.0])?);
        let b = Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0f32, 4.0])?);

        let sum = a.add(&b)?;
        assert_eq!(sum.to_vec(), vec![4.0, 6.0]);

        let product = a.mul(&b)?;
        assert_eq!(product.to_vec(), vec![3.0, 8.0]);

        Ok(())
    }

    #[test]
    fn test_dense_layer() -> Result<(), Box<dyn std::error::Error>> {
        let layer = Dense::new(2, 3, true);
        let input = Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0f32, 2.0])?);

        let output = layer.forward(&input)?;
        assert_eq!(output.shape(), &[1, 3]);

        Ok(())
    }

    #[test]
    fn test_zero_ones_tensors() {
        let zeros = Tensor::<f32>::zeros(&[2, 3]);
        assert_eq!(zeros.shape(), &[2, 3]);
        assert_eq!(zeros.to_vec(), vec![0.0; 6]);

        let ones = Tensor::<f32>::ones(&[2, 3]);
        assert_eq!(ones.shape(), &[2, 3]);
        assert_eq!(ones.to_vec(), vec![1.0; 6]);
    }
}
