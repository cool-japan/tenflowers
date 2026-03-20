use tenflowers_core::{Tensor, Device, Shape};
use tenflowers_neural::{Sequential, Dense, Model};

#[test]
fn test_tensor_creation() {
    let tensor = Tensor::<f32>::zeros(&[2, 3]);
    assert_eq!(tensor.shape().dims(), &[2, 3]);
    assert_eq!(tensor.device(), &Device::Cpu);
}

#[test]
fn test_tensor_operations() {
    let a = Tensor::<f32>::ones(&[2, 3]);
    let b = Tensor::<f32>::ones(&[2, 3]);
    
    let c = a.add(&b).unwrap();
    assert_eq!(c.shape().dims(), &[2, 3]);
}

#[test]
fn test_sequential_model() {
    let model = Sequential::<f32>::new(vec![
        Box::new(Dense::new(10, 20, true)),
        Box::new(Dense::new(20, 10, true)),
    ]);
    
    let input = Tensor::<f32>::zeros(&[5, 10]);
    let output = model.forward(&input);
    
    assert!(output.is_ok());
    if let Ok(out) = output {
        assert_eq!(out.shape().dims(), &[5, 10]);
    }
}