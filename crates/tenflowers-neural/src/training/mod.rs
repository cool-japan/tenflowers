pub mod data_parallel;
/// Training utilities and advanced training strategies
///
/// This module provides high-level training utilities including:
/// - Gradient accumulation for large model training
/// - Memory-efficient training strategies
/// - Advanced optimization techniques
/// - Knowledge distillation teacher-student training
/// - Data parallel distributed training
pub mod gradient_accumulation_trainer;
pub mod knowledge_distillation;

pub use gradient_accumulation_trainer::{
    create_memory_efficient_trainer, create_trainer_for_large_model, AccumulationTrainingConfig,
    GradientAccumulationTrainer, TrainingStats,
};

pub use knowledge_distillation::{
    create_distillation_trainer, create_distillation_trainer_with_temperature, DistillationConfig,
    DistillationMetrics, DistillationTrainer, DistillationTrainerBuilder,
};

pub use data_parallel::{
    create_data_parallel_trainer, AllReduce, CommunicationBackend, CompressedGradient,
    DataParallelConfig, DataParallelStats, DataParallelTrainer, DataParallelTrainerBuilder,
    GradientCompressor, ThreadAllReduce,
};
