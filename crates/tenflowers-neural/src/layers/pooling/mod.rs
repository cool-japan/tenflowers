// Pooling layers organized by type

pub mod fractional;
pub mod global;
pub mod pool2d;
pub mod pool3d;
pub mod roi;

// Re-export all pooling types for backwards compatibility
pub use fractional::{FractionalAvgPool2D, FractionalMaxPool2D};
pub use global::{GlobalAvgPool2D, GlobalAvgPool3D, GlobalMaxPool2D, GlobalMaxPool3D};
pub use pool2d::{AdaptiveAvgPool2D, AdaptiveMaxPool2D, AvgPool2D, MaxPool2D};
pub use pool3d::{AvgPool3D, MaxPool3D};
pub use roi::{ROIAlign2D, ROIPool2D};

// Re-export for convenience
// pub use fractional::*; // Unused for now
// pub use global::*; // Unused for now
// pub use pool2d::*; // Unused for now
// pub use pool3d::*; // Unused for now
// pub use roi::*; // Unused for now
