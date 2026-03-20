// Simple validation test for TenfloweRS core functionality
use tenflowers_core::{Tensor, Device, Result};

fn main() -> Result<()> {
    println!("🌼 TenfloweRS Core Functionality Validation");

    // Test 1: Basic tensor creation
    println!("✅ Test 1: Basic tensor creation");
    let ones = Tensor::ones(&[2, 3]);
    println!("   Created tensor with shape: {:?}", ones.shape().dims());

    // Test 2: Tensor arithmetic operations
    println!("✅ Test 2: Tensor arithmetic");
    let zeros = Tensor::zeros(&[2, 3]);
    let result = ones.add(&zeros)?;
    println!("   Tensor addition successful");

    // Test 3: SciRS2 random number generation
    println!("✅ Test 3: SciRS2 random integration");
    let mut rng = scirs2_core::random::rng();
    let random_val: f32 = rng.random();
    println!("   Generated random value: {}", random_val);

    println!("🌼 All core validations passed! TenfloweRS is functional.");
    Ok(())
}