// Loss functions organized by category

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
