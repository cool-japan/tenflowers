//! Loss functions for neural network training.
//!
//! This module provides a comprehensive collection of loss functions organized by task category.
//! Loss functions (also called cost functions or objective functions) measure how well a model's
//! predictions match the ground truth, and are minimized during training.
//!
//! # Loss Function Categories
//!
//! ## Classification Losses
//!
//! - [`binary_cross_entropy`]: Binary classification (sigmoid output)
//! - [`categorical_cross_entropy`]: Multi-class classification (softmax output)
//! - [`sparse_categorical_cross_entropy`]: Multi-class with integer labels
//! - [`focal_loss`]: Handles class imbalance by focusing on hard examples
//! - [`hinge_loss`]: SVM-style loss for classification
//! - [`nll_loss`]: Negative log-likelihood loss
//! - [`knowledge_distillation_loss`]: Transfer knowledge from teacher to student
//! - [`advanced_knowledge_distillation_loss`]: Advanced distillation with temperature
//!
//! ## Regression Losses
//!
//! - [`mse`]: Mean Squared Error (L2 loss)
//! - [`l1_loss`]: Mean Absolute Error (L1 loss)
//! - [`smooth_l1_loss`]: Smooth L1 loss (less sensitive to outliers)
//! - [`huber_loss`]: Combination of L1 and L2 loss
//! - [`quantile_loss`]: For quantile regression
//!
//! ## Segmentation Losses
//!
//! - [`dice_loss`]: Dice coefficient loss for segmentation
//! - [`iou_loss`]: Intersection over Union loss
//! - [`generalized_dice_loss`]: Generalized Dice for class imbalance
//!
//! ## Metric Learning Losses
//!
//! - [`contrastive_loss`]: Learn embeddings by contrasting pairs
//! - [`triplet_loss`]: Learn embeddings with anchor-positive-negative triplets
//!
//! # Usage Examples
//!
//! ## Binary Classification
//!
//! ```rust,no_run
//! use tenflowers_neural::loss::binary_cross_entropy;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let predictions = Tensor::zeros(&[32, 1]); // Sigmoid outputs [0, 1]
//! let targets = Tensor::zeros(&[32, 1]);     // Binary labels {0, 1}
//!
//! let loss = binary_cross_entropy(&predictions, &targets)?;
//! println!("BCE Loss: {:?}", loss);
//! # Ok(())
//! # }
//! ```
//!
//! ## Multi-Class Classification
//!
//! ```rust,no_run
//! use tenflowers_neural::loss::categorical_cross_entropy;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let predictions = Tensor::zeros(&[32, 10]); // Softmax outputs (sum to 1)
//! let targets = Tensor::zeros(&[32, 10]);     // One-hot encoded labels
//!
//! let loss = categorical_cross_entropy(&predictions, &targets)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Sparse Multi-Class (Integer Labels)
//!
//! ```rust,no_run
//! use tenflowers_neural::loss::sparse_categorical_cross_entropy;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let logits = Tensor::zeros(&[32, 10]);    // Raw logits (before softmax)
//! let labels = Tensor::zeros(&[32]);        // Integer labels [0, 9]
//!
//! let loss = sparse_categorical_cross_entropy(&logits, &labels)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Focal Loss for Imbalanced Data
//!
//! ```rust,no_run
//! use tenflowers_neural::loss::focal_loss;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let predictions = Tensor::zeros(&[32, 10]);
//! let targets = Tensor::zeros(&[32, 10]);
//!
//! let loss = focal_loss(
//!     &predictions,
//!     &targets,
//!     2.0,  // gamma: focus on hard examples
//!     0.25, // alpha: class weight
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Regression with MSE
//!
//! ```rust,no_run
//! use tenflowers_neural::loss::mse;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let predictions = Tensor::zeros(&[32, 1]);
//! let targets = Tensor::zeros(&[32, 1]);
//!
//! let loss = mse(&predictions, &targets)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Robust Regression with Huber Loss
//!
//! ```rust,no_run
//! use tenflowers_neural::loss::huber_loss;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let predictions = Tensor::zeros(&[32, 1]);
//! let targets = Tensor::zeros(&[32, 1]);
//!
//! let loss = huber_loss(&predictions, &targets, 1.0)?; // delta = 1.0
//! # Ok(())
//! # }
//! ```
//!
//! ## Semantic Segmentation with Dice Loss
//!
//! ```rust,no_run
//! use tenflowers_neural::loss::dice_loss;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let predictions = Tensor::zeros(&[8, 21, 256, 256]); // [B, C, H, W]
//! let targets = Tensor::zeros(&[8, 21, 256, 256]);     // One-hot encoded
//!
//! let loss = dice_loss(&predictions, &targets, 1e-7)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Knowledge Distillation
//!
//! ```rust,no_run
//! use tenflowers_neural::loss::knowledge_distillation_loss;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let student_logits = Tensor::zeros(&[32, 10]);
//! let teacher_logits = Tensor::zeros(&[32, 10]);
//! let true_labels = Tensor::zeros(&[32, 10]);
//!
//! let loss = knowledge_distillation_loss(
//!     &student_logits,
//!     &teacher_logits,
//!     &true_labels,
//!     0.5,  // alpha: balance distillation and true labels
//!     3.0,  // temperature: soften distributions
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Metric Learning with Triplet Loss
//!
//! ```rust,no_run
//! use tenflowers_neural::loss::triplet_loss;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let anchor = Tensor::zeros(&[32, 128]);    // Embeddings
//! let positive = Tensor::zeros(&[32, 128]);  // Same class as anchor
//! let negative = Tensor::zeros(&[32, 128]);  // Different class
//!
//! let loss = triplet_loss(&anchor, &positive, &negative, 0.2)?; // margin
//! # Ok(())
//! # }
//! ```
//!
//! # Loss Function Selection Guide
//!
//! ## Binary Classification
//! - **Binary Cross-Entropy**: Standard choice
//! - **Focal Loss**: When dealing with class imbalance
//!
//! ## Multi-Class Classification
//! - **Categorical Cross-Entropy**: Standard for one-hot labels
//! - **Sparse Categorical Cross-Entropy**: More efficient for integer labels
//! - **Focal Loss**: For highly imbalanced datasets
//!
//! ## Regression
//! - **MSE**: Standard, sensitive to outliers
//! - **L1 Loss**: More robust to outliers
//! - **Huber Loss**: Balanced between MSE and L1
//! - **Smooth L1**: Similar to Huber, used in object detection
//!
//! ## Semantic Segmentation
//! - **Cross-Entropy**: Pixel-wise classification
//! - **Dice Loss**: Handles class imbalance better
//! - **IoU Loss**: Directly optimizes IoU metric
//!
//! ## Object Detection
//! - **Smooth L1**: For bounding box regression
//! - **Focal Loss**: For classification with anchor imbalance
//!
//! ## Metric Learning
//! - **Contrastive Loss**: For pair-based learning
//! - **Triplet Loss**: For more structured embedding spaces

pub mod classification;
pub mod metric_learning;
pub mod regression;
pub mod segmentation;
pub mod utils;

// Re-export all loss functions for backwards compatibility
pub use classification::{
    advanced_knowledge_distillation_loss, binary_cross_entropy, categorical_cross_entropy,
    focal_loss, hinge_loss, knowledge_distillation_loss, nll_loss,
    sparse_categorical_cross_entropy,
};
pub use metric_learning::{contrastive_loss, triplet_loss};
pub use regression::{huber_loss, l1_loss, mse, quantile_loss, smooth_l1_loss};
pub use segmentation::{dice_loss, generalized_dice_loss, iou_loss};

// Re-export for convenience
// pub use classification::*; // Unused for now
// pub use metric_learning::*; // Unused for now
// pub use regression::*; // Unused for now
// pub use segmentation::*; // Unused for now
