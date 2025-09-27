pub mod adabelief;
pub mod adadelta;
pub mod adagrad;
pub mod adam;
pub mod adamw;
pub mod cosine_scheduler;
pub mod enhanced_accumulation;
pub mod gradient_centralization;
pub mod gradient_clipping;
pub mod lamb;
pub mod lbfgs;
pub mod lion;
pub mod lookahead;
pub mod nadam;
pub mod optimizer_with_accumulation;
pub mod parameter_groups;
pub mod radam;
pub mod rmsprop;
pub mod sam;
pub mod sgd;
pub mod soap;
pub mod sophia;
pub mod swa;

pub use adabelief::{AdaBelief, AdaBeliefConfig};
pub use adadelta::Adadelta;
pub use adagrad::Adagrad;
pub use adam::Adam;
pub use adamw::AdamW;
pub use cosine_scheduler::{
    create_cosine_schedule_for_epochs, CosineScheduler, CosineSchedulerConfig, SnapshotEnsemble,
};
pub use enhanced_accumulation::{
    AccumulationProgress, AccumulationStrategy, EnhancedGradientAccumulator,
    EnhancedOptimizerWithAccumulation, MemoryConfig, MemoryStats,
};
pub use gradient_centralization::{
    apply_gradient_centralization, GradientCentralizationConfig, GradientCentralizationWrapper,
    WithGradientCentralization,
};
pub use gradient_clipping::{
    clip_gradients_adaptive, clip_gradients_by_global_norm, clip_gradients_by_norm,
    clip_gradients_by_value,
};
pub use lamb::LAMB;
pub use lbfgs::LBFGS;
pub use lion::Lion;
pub use lookahead::Lookahead;
pub use nadam::Nadam;
pub use optimizer_with_accumulation::OptimizerWithAccumulation;
pub use parameter_groups::{ParameterGroup, ParameterGroupConfig, ParameterGroupOptimizer};
pub use radam::RAdam;
pub use rmsprop::RMSprop;
pub use sam::SAMOptimizer;
pub use sgd::SGD;
pub use soap::Soap;
pub use sophia::Sophia;
pub use swa::{ensemble_predict, SwaConfig, SWA};

use crate::model::Model;
use tenflowers_core::Result;

pub trait Optimizer<T> {
    fn step(&mut self, model: &mut dyn Model<T>) -> Result<()>;
    fn zero_grad(&self, model: &mut dyn Model<T>);
    fn set_learning_rate(&mut self, learning_rate: f32);
    fn get_learning_rate(&self) -> f32;
}
