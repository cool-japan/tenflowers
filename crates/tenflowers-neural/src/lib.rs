//! # TenfloweRS Neural Network Framework
//!
//! TenfloweRS Neural is a comprehensive, production-ready deep learning library built in pure Rust.
//! It provides a high-level API for building, training, and deploying neural networks with a focus
//! on safety, performance, and ease of use.
//!
//! ## Features
//!
//! - **Comprehensive Layer Library**: Dense, convolutional, recurrent, attention, normalization, and more
//! - **Advanced Training**: Gradient accumulation, mixed precision, distributed training
//! - **Modern Architectures**: Transformers, ResNet, EfficientNet, Vision Transformers, BERT, GPT
//! - **PEFT Methods**: LoRA, QLoRA, Prefix Tuning, P-Tuning v2, IA³
//! - **Optimization**: SGD, Adam, AdamW, Lion, LAMB, AdaBelief with advanced scheduling
//! - **Deployment**: Model quantization, pruning, ONNX export, mobile optimization
//! - **SciRS2 Integration**: Built on the robust SciRS2 scientific computing ecosystem
//!
//! ## Quick Start
//!
//! ### Building a Simple Neural Network
//!
//! ```rust,no_run
//! use tenflowers_neural::{Sequential, Dense, ActivationFunction};
//! use tenflowers_core::{Tensor, Device};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a simple feedforward network
//! let mut model = Sequential::new();
//! model.add(Dense::new(784, 128)?);
//! model.add_activation(ActivationFunction::ReLU);
//! model.add(Dense::new(128, 10)?);
//! model.add_activation(ActivationFunction::Softmax);
//!
//! // Forward pass
//! let input = Tensor::zeros(&[32, 784]); // batch_size=32, features=784
//! let output = model.forward(&input)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Training with the High-Level API
//!
//! ```rust,no_run
//! use tenflowers_neural::{quick_train, Sequential, Dense, SGD};
//! use tenflowers_neural::loss::categorical_cross_entropy;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Build model
//! let mut model = Sequential::new();
//! model.add(Dense::new(10, 64)?);
//! model.add(Dense::new(64, 3)?);
//!
//! // Prepare data
//! let x_train = Tensor::zeros(&[100, 10]);
//! let y_train = Tensor::zeros(&[100, 3]);
//!
//! // Train with one line
//! let results = quick_train(
//!     model,
//!     &x_train,
//!     &y_train,
//!     Box::new(SGD::new(0.01)),
//!     categorical_cross_entropy,
//!     10, // epochs
//!     32, // batch_size
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Advanced Training with Callbacks
//!
//! ```rust,no_run
//! use tenflowers_neural::{Trainer, EarlyStopping, ModelCheckpoint};
//! use tenflowers_neural::{Sequential, Dense, Adam};
//! use tenflowers_neural::loss::mse;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Sequential::new();
//! let optimizer = Box::new(Adam::new(0.001));
//!
//! let mut trainer = Trainer::new(model, optimizer, mse);
//! trainer.add_callback(Box::new(EarlyStopping::new(5, 0.001)));
//! trainer.add_callback(Box::new(ModelCheckpoint::new("best_model.bin")?));
//!
//! let x_train = Tensor::zeros(&[1000, 10]);
//! let y_train = Tensor::zeros(&[1000, 1]);
//!
//! trainer.fit(&x_train, &y_train, 100, 32)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture Overview
//!
//! The crate is organized into the following modules:
//!
//! - [`layers`]: Neural network layer implementations (Dense, Conv, RNN, Attention, etc.)
//! - [`model`]: Model abstractions (Sequential, Functional, custom models)
//! - [`optimizers`]: Optimization algorithms (SGD, Adam, AdamW, Lion, etc.)
//! - [`loss`]: Loss functions (MSE, cross-entropy, focal loss, etc.)
//! - [`metrics`]: Evaluation metrics (accuracy, F1, precision, recall, etc.)
//! - [`trainer`]: High-level training API with callbacks and hooks
//! - [`scheduler`]: Learning rate scheduling strategies
//! - [`distributed`]: Distributed and data-parallel training
//! - [`peft`]: Parameter-efficient fine-tuning methods
//! - [`deployment`]: Model optimization and export utilities
//! - [`pretrained`]: Pretrained model architectures and weights
//!
//! ## GPU Acceleration
//!
//! TenfloweRS supports GPU acceleration through the SciRS2 ecosystem. GPU operations
//! are automatically dispatched when tensors are placed on GPU devices:
//!
//! ```rust,no_run
//! use tenflowers_core::{Tensor, Device};
//! use tenflowers_neural::Dense;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # #[cfg(feature = "gpu")]
//! # {
//! let device = Device::gpu(0)?; // Use GPU 0
//! let layer = Dense::new(128, 64)?;
//! let input = Tensor::zeros(&[32, 128]).to_device(&device)?;
//! let output = layer.forward(&input)?; // Runs on GPU
//! # }
//! # Ok(())
//! # }
//! ```
//!
//! ## Mixed Precision Training
//!
//! For faster training and reduced memory usage:
//!
//! ```rust,no_run
//! use tenflowers_neural::{MixedPrecisionTrainer, Sequential, Adam};
//! use tenflowers_neural::loss::mse;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Sequential::new();
//! let optimizer = Box::new(Adam::new(0.001));
//!
//! let mut trainer = MixedPrecisionTrainer::new(
//!     model,
//!     optimizer,
//!     mse,
//!     true, // enable loss scaling
//! );
//! # Ok(())
//! # }
//! ```
//!
//! ## Distributed Training
//!
//! Scale training across multiple GPUs:
//!
//! ```rust,no_run
//! use tenflowers_neural::{create_data_parallel, Sequential, Dense};
//! use tenflowers_core::Device;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # #[cfg(feature = "gpu")]
//! # {
//! let model = Sequential::new();
//! let devices = vec![Device::gpu(0)?, Device::gpu(1)?];
//! let parallel_model = create_data_parallel(model, devices)?;
//! # }
//! # Ok(())
//! # }
//! ```
//!
//! ## PEFT (Parameter-Efficient Fine-Tuning)
//!
//! Fine-tune large models efficiently:
//!
//! ```rust,no_run
//! use tenflowers_neural::peft::{LoRALayer, LoRAConfig};
//! use tenflowers_neural::Dense;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let base_layer = Dense::new(768, 768)?;
//! let lora_config = LoRAConfig {
//!     rank: 8,
//!     alpha: 16.0,
//!     dropout: 0.1,
//! };
//! let lora_layer = LoRALayer::wrap(base_layer, lora_config)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Model Deployment
//!
//! Optimize models for production:
//!
//! ```rust,no_run
//! use tenflowers_neural::deployment::{ModelOptimizer, OptimizationConfig};
//! use tenflowers_neural::Sequential;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Sequential::new();
//! let config = OptimizationConfig {
//!     quantize: true,
//!     prune_threshold: Some(0.01),
//!     fuse_operations: true,
//! };
//!
//! let optimizer = ModelOptimizer::new(config);
//! let optimized_model = optimizer.optimize(model)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Contributing
//!
//! TenfloweRS is part of the SciRS2 ecosystem. For contributions, issues, or questions,
//! please visit our GitHub repository.

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
pub mod utils;

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
pub use layers::{
    scaled_dot_product_attention, BahdanauAttention, Conv2D, Dense, Dropout, KVCache, Layer,
    LuongAttention, MultiHeadAttention, RMSNorm, TransformerDecoder, TransformerEncoder, GRU, LSTM,
    RNN,
};
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
pub use optimizers::{
    clip_gradients_adaptive, clip_gradients_by_global_norm, clip_gradients_by_norm,
    clip_gradients_by_value, AdaBelief, Adadelta, Adagrad, Adam, AdamW, Lion, Lookahead, Nadam,
    Optimizer, ParameterGroup, ParameterGroupOptimizer, RAdam, RMSprop, LAMB, SGD,
};
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
pub use utils::{
    check_parameters_finite, clip_parameters_by_value, count_parameters,
    count_trainable_parameters, get_parameter_shapes, he_init, one_init, parameter_norm,
    xavier_init, zero_init, AugmentationConfig, AugmentationPipeline, AugmentationStats,
    BatchConfig, BatchSampler, BatchStatistics, CollationStrategy, Collator, ConfusionMatrix,
    GradientFlowInfo, Histogram, ImageAugmentation, LayerInfo, LearningRateSchedule,
    ModelInspector, ModelStats, ModelSummary, PaddingStrategy, PlotData, ProfilingInfo,
    SamplingStrategy, SequenceAugmentation, TrainingCurve,
};
