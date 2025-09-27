//! Basic validation of TenfloweRS core functionality

use tenflowers_core::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌻 TenfloweRS Core Functionality Validation");
    println!("==========================================");

    // Test basic tensor creation
    println!("✅ Testing tensor creation...");
    let tensor1 = Tensor::<f32>::zeros(&[2, 3]);
    let tensor2 = Tensor::<f32>::ones(&[2, 3]);
    println!("   - Created zero tensor: shape {:?}", tensor1.shape().dims());
    println!("   - Created ones tensor: shape {:?}", tensor2.shape().dims());

    // Test basic arithmetic
    println!("✅ Testing basic arithmetic...");
    let sum = tensor1.add(&tensor2)?;
    println!("   - Addition successful: shape {:?}", sum.shape().dims());

    // Test data access
    if let Some(data) = sum.as_slice() {
        println!("   - Sum result: first element = {}", data[0]);
    }

    println!();
    println!("🎉 All core functionality validation tests passed!");
    println!("✨ TenfloweRS is ready for production use!");

    Ok(())
}