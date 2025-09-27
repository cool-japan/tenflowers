#![deny(unsafe_code)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_mut)]
#![allow(clippy::result_large_err)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
#![allow(clippy::vec_init_then_push)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::doc_overindented_list_items)]

pub mod activation_function;
pub mod backends;
pub mod benchmarks;
pub mod data;
pub mod deployment;
pub mod distributed;
pub mod layers;
pub mod loss;
pub mod metrics;
pub mod mixed_precision;
pub mod model;
pub mod model_parallel;
pub mod optimizers;
pub mod peft;
pub mod pipeline;
pub mod pretrained;
pub mod scheduler;
#[cfg(feature = "serialize")]
pub mod serialization;
pub mod trainer;
pub mod training;
pub mod training_pipeline;

#[cfg(feature = "onnx")]
pub mod onnx;

pub mod tensorflow_compat;

pub use activation_function::ActivationFunction;
pub use benchmarks::{
    compare_models, BenchmarkConfig, BenchmarkMetrics, BenchmarkResults, ModelBenchmark,
};
pub use distributed::{
    models::utils::{create_data_parallel, create_distributed_data_parallel, init_process_group},
    models::{DDPConfig, DataParallel, DistributedDataParallel, SynchronizationMode},
    BackendConfig, CollectiveOp, CollectiveResult, CommunicationBackend, CommunicationBackendImpl,
    CommunicationGroup, CommunicationMetrics, CommunicationRuntime, CompressionAlgorithm,
    ReductionOp,
};
pub use layers::{Conv2D, Dense, KVCache, Layer, RMSNorm};
pub use loss::{
    advanced_knowledge_distillation_loss, binary_cross_entropy, categorical_cross_entropy,
    focal_loss, hinge_loss, huber_loss, knowledge_distillation_loss, mse, quantile_loss,
    sparse_categorical_cross_entropy,
};
pub use metrics::{
    accuracy, confusion_matrix, f1_score, mean_absolute_percentage_error, precision, r_squared,
    recall, top_k_accuracy,
};
pub use mixed_precision::MixedPrecisionTrainer;
pub use model::{
    FunctionalModel, FunctionalModelBuilder, Input, Model, Node, Sequential, SharedLayer,
};
pub use model_parallel::{
    CommunicationPattern, MemoryRequirements, ModelParallelConfig, ModelParallelCoordinator,
    ParallelLayer, PipelineConfig, PlacementStrategy, SplitLayer, TensorParallelConfig,
};
pub use optimizers::{Adam, Lion, Optimizer, SGD};
pub use pipeline::{MicroBatch, PipelineMetrics, PipelineModelBuilder, PipelineParallelModel};
pub use scheduler::{
    ConstantLR, CosineAnnealingLR, ExponentialLR, LearningRateScheduler, PolynomialLR,
    ReduceLROnPlateau, StepLR, WarmupCosineDecayLR,
};
pub use trainer::{
    Callback, EarlyStopping, LearningRateReduction, ModelCheckpoint, Trainer, TrainingMetrics,
    TrainingState,
};
pub use training::{
    create_distillation_trainer, create_distillation_trainer_with_temperature,
    create_memory_efficient_trainer, create_trainer_for_large_model, AccumulationTrainingConfig,
    DistillationConfig, DistillationMetrics, DistillationTrainer, DistillationTrainerBuilder,
    GradientAccumulationTrainer, TrainingStats,
};
pub use training_pipeline::{
    quick_train, TrainingPipeline, TrainingPipelineConfig, TrainingResults,
};

#[cfg(feature = "tensorboard")]
pub use trainer::TensorboardCallback;

#[cfg(feature = "onnx")]
pub use onnx::{
    OnnxAttribute, OnnxDataType, OnnxExport, OnnxGraph, OnnxModel, OnnxNode, OnnxTensor,
    OnnxValueInfo,
};

pub use tensorflow_compat::{
    load_tensorflow_model, load_tensorflow_model_with_config, SavedModel, SavedModelLoader,
    SavedModelMetadata,
};

pub use data::{DataPipelineConfig, NeuralDataPipeline, NeuralTransforms, TrainingBatch};
pub use deployment::{
    conservative_pruning_config, edge_fusion_config, edge_pruning_config, edge_quantization_config,
    fuse_layers, mobile_fusion_config, mobile_pruning_config, mobile_quantization_config,
    optimize_for_deployment, prune_model, quantize_model, ultra_low_precision_config,
    DeploymentMetadata, DeploymentModel, FusedLayer, FusionConfig, FusionPattern, FusionStats,
    LayerFusion, ModelOptimizer, ModelPruner, ModelQuantizer, OptimizationConfig,
    OptimizationStats, PrunedLayer, PruningConfig, PruningMask, PruningScope, PruningStats,
    PruningStrategy, QuantizationConfig, QuantizationParams, QuantizationPrecision,
    QuantizationStats, QuantizationStrategy, QuantizedLayer,
};
pub use peft::{
    AdaLoRAAdapter, AdaLoRAConfig, AdaLoRAStats, IA3Adapter, IA3Config, IA3InitStrategy,
    IA3ScalingType, IA3Stats, ImportanceMetric, LoRAAdapter, LoRAConfig, LoRADense, LoRALayer,
    MultiIA3Adapter, MultiIA3Stats, PEFTAdapter, PEFTConfig, PEFTLayer, PEFTMethod, PEFTStats,
    PTuningTaskType, PTuningV2Adapter, PTuningV2Config, PTuningV2Stats, PrefixTaskType,
    PrefixTuningAdapter, PrefixTuningConfig, PrefixTuningStats, PromptLayerConfig, QLoRAAdapter,
    QLoRAConfig, QLoRAMemoryStats, QuantizationType, RankAdaptationStats, TokenPosition,
};
pub use pretrained::{
    BasicBlock, BottleneckBlock, EfficientNet, EfficientNetConfig, MBConvBlock, PatchEmbedding,
    ResNet, ResNetBlockType, SEBlock, VisionTransformer,
};
#[cfg(feature = "serialize")]
pub use serialization::{
    AdvancedModelState, AdvancedSerialization, CheckpointInfo, CheckpointLoadResult,
    CheckpointManager, CompressionAlgorithm as SerializationCompressionAlgorithm, CompressionInfo,
    LoadResult, ModelMetadata, ModelMigrator, ParameterInfo, SchemaValidator, SemanticVersion,
    SerializationConfig, ValidationResult,
};
