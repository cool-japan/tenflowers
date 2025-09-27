use tenflowers_core::{Result, Tensor};

// Helper function to convert u8 tensor to bool tensor
pub fn convert_u8_to_bool_tensor(tensor: &Tensor<u8>) -> Result<Tensor<bool>> {
    if let Some(u8_data) = tensor.as_slice() {
        let shape = tensor.shape().dims().to_vec();
        let bool_data: Vec<bool> = u8_data.iter().map(|&u| u != 0).collect();

        Tensor::from_vec(bool_data, &shape)
    } else {
        Err(tenflowers_core::error::TensorError::InvalidArgument {
            operation: "convert_u8_to_bool_tensor".to_string(),
            reason: "Cannot access u8 tensor data".to_string(),
            context: None,
        })
    }
}
