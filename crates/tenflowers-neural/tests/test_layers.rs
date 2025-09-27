use tenflowers_core::{Result, Tensor};
use tenflowers_neural::layers::{AvgPool2D, BatchNorm, Layer, MaxPool2D};

#[test]
fn test_batch_norm_inference() -> Result<()> {
    let mut batch_norm = BatchNorm::<f32>::new(3);
    batch_norm.set_training(false);

    // Create input: (batch=2, features=3)
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

    // Forward pass in inference mode (uses default running stats)
    let output = batch_norm.forward(&input)?;

    // Check output shape
    assert_eq!(output.shape().dims(), &[2, 3]);

    Ok(())
}

#[test]
fn test_batch_norm_training() -> Result<()> {
    let mut batch_norm = BatchNorm::<f32>::new(3);
    batch_norm.set_training(true);

    // Create input: (batch=4, features=3)
    let input = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0],
        &[4, 3],
    )?;

    // Forward pass in training mode
    let output = batch_norm.forward(&input)?;

    // Check output shape
    assert_eq!(output.shape().dims(), &[4, 3]);

    // In training mode with gamma=1 and beta=0, the output should be normalized
    // Mean should be close to 0 and variance close to 1 for each feature

    Ok(())
}

#[test]
fn test_max_pool2d() -> Result<()> {
    let pool = MaxPool2D::new((2, 2), None);

    // Create 4D input: (batch=1, height=4, width=4, channels=1)
    let input = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[1, 4, 4, 1],
    )?;

    let output = pool.forward(&input)?;

    // With 2x2 kernel and stride=2, output should be 2x2
    assert_eq!(output.shape().dims(), &[1, 2, 2, 1]);

    // Check values - should be max of each 2x2 region
    if let Some(data) = output.as_slice() {
        assert_eq!(data, &[6.0, 8.0, 14.0, 16.0]);
    }

    Ok(())
}

#[test]
fn test_avg_pool2d() -> Result<()> {
    let pool = AvgPool2D::new((2, 2), None);

    // Create 4D input: (batch=1, height=4, width=4, channels=1)
    let input = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[1, 4, 4, 1],
    )?;

    let output = pool.forward(&input)?;

    // With 2x2 kernel and stride=2, output should be 2x2
    assert_eq!(output.shape().dims(), &[1, 2, 2, 1]);

    // Check values - should be average of each 2x2 region
    if let Some(data) = output.as_slice() {
        // (1+2+5+6)/4 = 3.5, (3+4+7+8)/4 = 5.5, etc.
        assert_eq!(data, &[3.5, 5.5, 11.5, 13.5]);
    }

    Ok(())
}

#[test]
fn test_max_pool2d_multichannel() -> Result<()> {
    let pool = MaxPool2D::new((2, 2), Some((1, 1))); // stride = 1

    // Create 4D input: (batch=1, height=3, width=3, channels=2)
    let input = Tensor::from_vec(
        vec![
            // Channel 0
            1.0, 2.0, // position (0,0)
            3.0, 4.0, // position (0,1)
            5.0, 6.0, // position (0,2)
            7.0, 8.0, // position (1,0)
            9.0, 10.0, // position (1,1)
            11.0, 12.0, // position (1,2)
            13.0, 14.0, // position (2,0)
            15.0, 16.0, // position (2,1)
            17.0, 18.0, // position (2,2)
        ],
        &[1, 3, 3, 2],
    )?;

    let output = pool.forward(&input)?;

    // With 2x2 kernel and stride=1 on 3x3 input, output should be 2x2
    assert_eq!(output.shape().dims(), &[1, 2, 2, 2]);

    Ok(())
}
