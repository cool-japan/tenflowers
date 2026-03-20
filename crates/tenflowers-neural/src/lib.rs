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
//! ```rust,ignore
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
//! ```rust,ignore
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
//! ```rust,ignore
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
//! ```rust,ignore
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
//! ```rust,ignore
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
//! ```rust,ignore
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
//! ```rust,ignore
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
//! ```rust,ignore
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

pub mod active_learning;
pub mod adversarial;
pub mod anomaly_detection;
pub mod tensorflow_compat;
pub mod text_generation_pipelines;
pub use anomaly_detection::{
    compute_anomaly_metrics,
    compute_auc_roc,
    compute_average_precision,
    AdDeepSvdd,
    AdDeepSvddConfig,
    // Extended f64-based anomaly detection
    AdLinear,
    AdMlp,
    AeAnomaly,
    AeAnomalyConfig,
    AnomalyError,
    AnomalyEvaluationMetrics,
    AnomalyMemory,
    AnomalyMetrics,
    AnomalyThresholder,
    AnomalyTransformerModel,
    AnomalyVAE,
    AnomalyVaeConfig,
    CouplingLayer,
    DeepSVDD,
    DeepSvddConfig,
    FlowAnomaly,
    FlowAnomalyConfig,
    GaussianMixtureAnomaly,
    IsolationForestConfig,
    IsolationNode,
    IsolationTree,
    MemAeConfig,
    MemoryAugmentedAE,
    NeuralIsolationForest,
    PatchTSAD,
    PatchTsadConfig,
    RobustRCF,
    RobustRcfConfig,
    SpectralResidual,
    VaeAnomaly,
    VaeAnomalyConfig,
};
pub mod architecture_distillation;
pub mod bayesian;
pub mod bayesian_dl;
pub use bayesian_dl::{
    compute_calibration, nll_classification, BdlLinear, BdlMlp, CalibrationResult,
    DeepEnsemble as BdlDeepEnsemble, DeepEnsembleConfig as BdlDeepEnsembleConfig,
    LaplaceApproximation, SghcmConfig, SghcmSampler, SgldConfig, SgldSampler, SwagConfig,
    SwagModel, TemperatureScaling as BdlTemperatureScaling,
};
pub mod continual_learning;
pub mod contrastive;
pub mod curriculum_learning;
pub use curriculum_learning::{
    AutoCurriculum, ClBabyStepConfig, ClBanditArm, ClCompetenceStrategy, ClConfidenceMethod,
    ClCurriculumScheduler, ClDifficultyMethod, ClDifficultyScorer, ClEpochResult, ClLambdaSchedule,
    ClReport, ClSplWeightingStrategy, ClTrainingMode, CompetenceLearning, CurriculumMetrics,
    CurriculumTrainer, DataShapley, KnnShapley, LavaValuation, SelfPacedLearning,
};
pub mod lifelong_learning;
pub use lifelong_learning::{
    AgemOptimizer,
    ClMetrics,
    ContinualLearningMetrics,
    DarkExperienceReplay,
    EpisodeMemory,
    ExperienceReplay,
    ForgettingMeasure,
    GemConstraint,
    GemTrainer,
    GenerativeReplay,
    HatMask,
    LearningWithoutForgetting,
    LllAGemModel,
    LllCoPE,
    LllDer,
    LllDerEntry,
    LllER,
    LllErSample,
    LllGemModel,
    LllHAT,
    // Lll-prefixed extensions
    LllLinearLayer,
    LllMetrics,
    LllReport,
    LllTaskOracle,
    ModularNetwork,
    PackNetMasker,
    ProgressiveGrowth,
    ReplayBuffer,
    SynapticIntelligence,
    OWM,
};
pub mod lm_evaluation;
pub use lm_evaluation::{
    LmeBERTScore,
    // §4 BERTScore
    LmeBERTScoreResult,
    LmeBLEU,
    // §2 BLEU
    LmeBLEUResult,
    // §D Calibration
    LmeCalibration,
    // §C ChrF
    LmeChrF,
    LmeCodeEval,
    // §9 Code evaluation
    LmeCodeResult,
    // §A Distinct-N
    LmeDistinctN,
    LmeFewShotConfig,
    LmeFewShotEval,
    // §6 Few-shot evaluation
    LmeFewShotExample,
    LmeHarmEval,
    // §7 Harm evaluation
    LmeHarmResult,
    // §8 Truthfulness evaluation
    LmeMC1Result,
    LmeMC2Result,
    LmeMathEval,
    // §5 Math evaluation
    LmeMathResult,
    // §B METEOR
    LmeMeteor,
    LmePerplexity,
    // §1 Perplexity
    LmePerplexityConfig,
    LmeROUGE,
    // §3 ROUGE
    LmeROUGEScore,
    // §10 Report
    LmeReport,
    LmeTruthfulnessEval,
    // §E WER
    LmeWER,
};
pub mod differentiable_physics;
pub mod diffusion;
pub mod distillation;
pub mod document_understanding;
pub use document_understanding::{
    create_spans,
    detect_table_structure,
    exact_match,
    extract_answer,
    extractive_summarize,
    f1_answer,
    // DocumentMetrics
    f1_score_ner,
    merge_lines,
    mmr_summarize,
    normalize_bbox,
    parse_form,
    rouge_n,
    sort_reading_order,
    table_to_csv,
    table_to_json,
    validate_field,
    BboxEmbedding,
    // DocumentClassifier
    DocClass,
    DocClassifier,
    DocEmbedder,
    DocEmbedderConfig,
    DocEntity,
    // QuestionAnsweringDoc
    DocQaConfig,
    // InformationExtractor
    EntityType,
    // FormParser
    FieldType,
    FormField,
    InformationExtractor,
    InputSpan,
    LayoutLm,
    // LayoutLM
    LayoutLmConfig,
    LayoutLmLayer,
    // DocumentOCR
    OcrBox,
    // DocumentEmbedder
    PoolingStrategy,
    // DocumentSummarizer
    SentenceScorer,
    Table,
    // TableExtractor
    TableCell,
};
pub mod ensemble;
pub mod federated;
pub mod flows;
pub mod hierarchical_time_series;
pub mod hparam;
pub mod hyperdimensional;
pub use hyperdimensional::{
    bind_binary, bind_bipolar, bundle_binary, bundle_bipolar, compute_hdc_stats,
    orthogonality_test, permute_binary, permute_bipolar, unbind_binary, unbind_bipolar, BinaryHv,
    BipolarHv, HdClassifier, HdSequenceEncoder, HdcError, HdcStats, HvType, IdEncoder, ItemMemory,
    LevelEncoder, OnlineHdc, RealHv, SdmConfig, SparseSdm, ThermometerEncoder, HD_DIM,
};
pub mod hyperparameter_optimization;
pub use hyperparameter_optimization::{
    best_trial,
    hypervolume_contribution,
    importance_by_fanova,
    nsga2_select,
    pareto_front,
    // §4 BOHB
    BohbConfig,
    BohbOptimizer,
    // §6 CMA-ES
    CmaHpoConfig,
    CmaHpoState,
    EvolutionaryStrategy,
    // §2 GP surrogate & acquisition
    GpHpo,
    HbBracket,
    HpSpace,
    // §1 HpSpace
    HpType,
    HpoAcqFunction,
    HpoBayesianConfig,
    HpoBayesianOptimizer,
    HpoLogger,
    HpoStudy,
    HpoTrial,
    // §3 HyperBand
    HyperBandConfig,
    HyperBandScheduler,
    HyperParameter,
    KdeSampler,
    // §8 Early Termination
    MedianStopping,
    // §7 Multi-Objective
    MoHpoConfig,
    MoObservation,
    MultiObjectiveHpo,
    OptDirection,
    // §5 PBT
    PbtConfig,
    PbtMember,
    PbtPopulation,
    PercentileStop,
    PopulationBasedTraining,
    // §10 Warm Starting
    PreviousStudy,
    SuccessiveHalving,
    // §9 HpoStudy / Logger
    TrialStatus,
    WarmStartSampler,
};
pub mod lora_adapters;
pub use lora_adapters::{
    AdaLoraLayer, BitFit, BitFitMask, DoraLayer, Ia3Layer, LoftqInit, LoftqResult, LoraLayer,
    LoraPlus, PeftManager, PrefixTuning, PromptTuning, SingularValueImportance,
};
pub mod learning_to_learn;
pub use learning_to_learn::{
    compute_linear_mse_grad, GradientPreprocessor, L2lAttnBlock, L2lError, L2lHiddenState,
    L2lLstmCell, L2lMetrics, L2lScheduler, L2lTask, L2lTcBlock, L2lTensor, LstmMetaOptimizer,
    MetaDataset, MetaLearningTrainer, MetaSampledTask, MetaTaskType, OptimizerNetwork, SnailModel,
    WarmStartOptimizer,
};
pub mod lr_finder;
pub mod meta_learning;
pub mod model_utils;
pub mod music_generation;
pub use music_generation::{
    generate_drum_pattern,
    AdsrEnvelope,
    AudioSynthesizer,
    // §6 Rhythm
    BjorklundPattern,
    // §4 Chord Detector
    Chord,
    ChordDetection,
    ChordDetector,
    DrumStyle,
    // §1 MIDI
    EventType,
    GenerationConfig as MelodyGenerationConfig,
    MelodyGenerator,
    MidiSequence,
    // §3 MuseTransformer
    MuseTransformer,
    // §10 Metrics
    MusicEvalReport,
    MusicMetrics,
    MusicTheoryAnalyzer,
    // §8 Variation Generator
    MusicVariationGenerator,
    NoteEvent,
    // §2 Piano Roll
    PianoRollEncoder,
    RhythmGrid,
    // §5 Melody Generator
    ScaleType,
    SynthConfig,
    // §9 Theory Analyzer
    VoiceLeadingAnalysis,
    // §7 Audio Synthesizer
    WaveType,
};
pub mod multi_objective;
pub use multi_objective::{
    AugmentedLagrangian, CaGrad, GeneticOperators, GradNormOptimizer, HypervolumeIndicator, ImtlG,
    InteriorPointMethod, LinearScalarization, Moead, MooHeads, NsgaIii, PFLLayer, ParetoFront,
    PcGrad, PenaltyMethod, ProjectedGradient, R2Indicator, ReferencePointSampler,
    SelectionOperator, TchebycheffScalarization, UncertaintyWeighting,
};
pub mod multi_fidelity;
pub mod multi_task;
pub mod neural_ode;
pub mod neural_sde;
pub use neural_sde::{
    compute_sde_metrics,
    shuffle_product,
    // advanced
    AncestralSampler,
    CdeVectorField,
    ControlledSde,
    LatentSde,
    LatentSdeConfig,
    LogSignatureLayer,
    NaturalCubicSpline,
    NeuralCde,
    NeuralCdeConfig,
    NeuralRde,
    NsdeMlp,
    PathSignature,
    RoughPath,
    ScoreMatchingSde,
    SdeDecoder,
    SdeDiffusionNet,
    SdeDriftNet,
    SdeEncoder,
    SdeMetrics,
    SdeMetricsExtended,
    SdeTrainer,
    SignatureConfig,
    SignatureKernel,
    SignatureTransform,
    VeSde,
    VpSde,
};
pub mod energy_models;
pub mod quantization;
pub mod rl;
pub mod self_supervised;
pub mod signal;
pub mod speech_recognition;
pub use speech_recognition::{
    cer,
    // Metrics
    edit_distance,
    // Spectrogram
    log_mel_spectrogram,
    wer,
    AsrMetrics,
    // Encoder / Decoder
    AudioConvStem,
    // CTC
    BeamEntry,
    CtcBeamDecoder,
    CtcConfig,
    JointNetwork,
    LmRescorer,
    // LM
    NgramLm,
    // RNN-T
    PredictionNetwork,
    RnntDecoder,
    SpeakerDiarizer,
    // Diarization
    SpectralClustering,
    // Augmentation
    SpeechAugmentation,
    // Pipeline
    SpeechPipeline,
    VadConfig,
    // VAD
    VoiceActivityDetector,
    // Config types
    WhisperConfig,
    WhisperDecoder,
    WhisperDecoderLayer,
    WhisperEncoder,
    WhisperEncoderLayer,
};
pub mod simulation_based_inference;
pub use simulation_based_inference::{
    c2st_accuracy,
    simulation_based_calibration,
    tarp_test,
    // Advanced: ABC
    AbcRejection,
    AbcSmcSampler,
    // Advanced: SBI diagnostics
    ExpectedCoveragePlot,
    FlowPosteriorSampler,
    FlowSbiTrainer,
    GaussianSimulator,
    LocalPredictivePerformance,
    // MADE / NDE
    MadeLayer,
    NeuralDensityEstimator,
    NeuralLikelihood,
    // SNRE
    NeuralRatioEstimator,
    NlePosteriorSampler,
    NreRatioEstimator,
    RoundSummary,
    // Diagnostics
    SbcResult,
    // Advanced: NRE / NLE posterior samplers
    SbiClassifier,
    SbiExtendedReport,
    // Advanced: Normalizing Flow SBI
    SbiNormalizingFlow,
    SbiReport,
    SequentialNpe,
    // Simulator trait + Gaussian toy
    Simulator,
    // SNLE
    SnleConfig,
    SnleEstimator,
    SnlePosterior,
    // SNPE
    SnpeConfig,
    SnpePosterior,
    SummaryStatistics,
    TarpResult,
};

pub mod simulation_ml;
pub use simulation_ml::{
    AdaptiveSampler, DenseLayer, EnsembleSurrogate, GpSurrogate, LatinHypercubeSampler,
    MlCorrectionModel, NeuralSurrogate, PhysicsResidual, PiSurrogate, RbfKernel,
    ReynoldsStressTensor, SimError, SimulationDataAugmenter, SimulationMetrics, SmagorinskyModel,
    Spring, SpringMassSystem, SurrogateActivation, SurrogateConfig,
};
pub mod spectral;
pub mod ssm;
pub mod structured_prediction;
pub use structured_prediction::{
    ctc_loss,
    label_smoothing_loss,
    log_sum_exp as sp_log_sum_exp,
    ordered_prediction_loss,
    // §6 Loss functions
    sequence_cross_entropy,
    softmax as sp_softmax,
    BeliefPropagation,
    ConstituencyConfig,
    ConstituencyParser,
    ConstrainedDecoding,
    // Advanced: Graph-based structured prediction
    DependencyParser,
    // §4 Energy-Based Model
    EnergyNetConfig,
    EnergyNetwork,
    // §5 Belief Propagation
    FactorGraph,
    HammingLoss,
    // §1 Linear-Chain CRF
    LinearChainCrfConfig,
    // Advanced: Neural CRF++
    NeuralCrfConfig,
    NeuralCrfLayer,
    PartialCrfLoss,
    SecondOrderCrf,
    // §2 Second-Order CRF
    SecondOrderCrfConfig,
    SemanticRoleLabeler,
    // §0 Shared utilities
    SpLinear,
    SpLinearChainCrf,
    // Advanced: Metrics
    SpStructuredMetrics,
    // Advanced: Span-based models
    Span,
    SpanClassifier,
    SpanExtractor,
    SrlAnnotation,
    // §3 Structured SVM
    SsvmConfig,
    StructuredLoss,
    StructuredSvm,
};
pub mod symbolic_math;
pub use symbolic_math::{
    balance,
    crossover as sym_crossover,
    derivative as sym_derivative,
    dim_divide,
    dim_multiply,
    eval as sym_eval,
    evolve as sym_evolve,
    find_pi_groups,
    frobenius_derivative,
    init_population as sym_init_population,
    integrate as sym_integrate,
    is_dimensionless,
    mutate as sym_mutate,
    // Equation balancer
    parse_equation,
    parse_token_sequence,
    poly_add,
    poly_div_rem,
    poly_evaluate,
    poly_gcd,
    poly_mul,
    poly_sub,
    roots_companion_matrix,
    simplify as sym_simplify,
    tournament_select as sym_tournament_select,
    trace_derivative,
    // Dimensional analysis
    Dimension,
    // Expression tree
    Expr,
    // Regressor
    ExprBasis,
    // Synthesizer
    ExprToken,
    // Hasher
    ExpressionHasher,
    GradientSymbolicRegressor,
    // Genetic programming
    Individual,
    // Matrix calculus
    MatrixExpr,
    NeuralExpressionSynthesizer,
    // Polynomial arithmetic
    Polynomial,
    SynthesizerConfig,
};
pub mod tabular_learning;
pub use tabular_learning::{
    AttentiveTransformer, CatBoostEncoder, CyclicEncoder, FTTransformer, FTTransformerConfig,
    FeatureEncoder, MinMaxScaler, MixedInputHead, NodeModel, ObliviousTree, QuantileTransformer,
    SaintBlock, SaintModel, StandardScaler, TabNet, TabNetConfig, TabTransformer,
    TabTransformerConfig, TabularAugmentation, TabularMetrics,
};
pub mod time_series;
pub mod tokenizer;
pub mod vae;
pub mod variational_inference;
pub use variational_inference::{
    compute_pareto_k,
    diagnose_vi,
    effective_sample_size,
    vi_log_sum_exp,
    AdviModel,
    AdviVariable,
    // §4 BBVI
    BbviConfig,
    BbviResult,
    BlackBoxVi,
    FlowVi,
    // §3 FullRankGaussian
    FullRankGaussian,
    // §2 MeanFieldGaussian
    MeanFieldGaussian,
    // §5 ADVI
    ParameterConstraint,
    // §7 FlowVi / PlanarFlow
    PlanarFlowLayer,
    // §6 StructuredVi
    StructuredVi,
    // §9 Diagnostics
    ViDiagnostics,
    // §1 Variational family enum
    ViDistribution,
    ViSvgd,
    // §8 ViSvgd (separate from monte_carlo::SvgdOptimizer)
    ViSvgdConfig,
};
pub mod graph_generation;
pub mod graph_matching;
pub mod graph_transformer;
pub use graph_matching::{
    GmEdge, GmEditCostModel, GmEditOp, GmGraph, GmMcsResult, GmReport, GmSoftAssignment,
    GraduatedAssignment, GraphEditDistance, GraphMatchMetrics, MaxCommonSubgraph, RandomWalkKernel,
    ShortestPathKernel, SpectralAlignment, Vf2Matcher, WeisfeilerLemanKernel,
};
pub mod automl;
pub mod nas;
pub use automl::{
    // meta_features
    AlgorithmPerformancePredictor,
    AlgorithmSelector,
    ArchitectureBank,
    ArchitectureDecoder,
    ArchitectureEnsemble,
    ArchitecturePredictor,
    AutoFeaturePipeline,
    AutoMlPipeline,
    AutoMlReport,
    AutoNormalizer,
    BayesianOptimizer as AutomlBayesianOptimizer,
    CellBasedNas,
    CellOp,
    Config as AutomlConfig,
    ConfigSpace,
    DatasetMetaFeatures,
    DistributionType,
    EarlyStoppingRule,
    EditDistance,
    EfficientNasPredictor,
    FeatureInteractionSearch,
    FeatureSelector,
    GradNormScore,
    GradientBasedNas,
    GraphEncoding,
    HyperBand,
    HyperBandBracket,
    JacobianScore,
    LandmarkingFeatures,
    MetaFeatureNormalizer,
    MfHpType,
    MultiObjectiveOptimizer,
    NaswotScore,
    PathEncoding,
    PipelineOptimizer,
    PipelineStep,
    PolynomialFeatures,
    PortfolioSelector,
    ProxylessNas,
    SinglePathOneShot,
    SmacOptimizer,
    SupernetLayer,
    SupernetOp,
    SynflowScore,
    TpeSampler,
    TransferNasFeatures,
    TrialMetric,
    ZenScore,
};
pub mod robotics;
pub use robotics::{
    AStarPlanner, BehavioralCloning, DaggerPolicy, DemonstrationBuffer, DhParam, GailDiscriminator,
    GraspQuality, MotionPrimitive, OccupancyGrid, ParticleFilter, PotentialFieldNavigator,
    RecurrentStateSpaceModel, RewardPredictor, RobotKinematics, WorldModelDecoder,
    WorldModelEncoder,
};
pub mod riemannian_geometry;
pub use riemannian_geometry::{
    mat_inv_nn,
    // Linear algebra helpers
    mat_mul_nn,
    matrix_exp_sym,
    matrix_log_sym,
    symmetrize_nn,
    // Fréchet mean
    FrechetMean as RgFrechetMean,
    // Grassmann manifold
    RgGrassmannManifold,
    // SO(3)
    RgSo3Manifold,
    // SPD manifold
    RgSpdManifold,
    // Stiefel manifold
    RgStiefelManifold,
    RiemannianAdam,
    // Optimizers
    RiemannianAdamConfig,
    // Batch normalization
    RiemannianBatchNorm,
    // Core trait
    RiemannianManifold,
    RiemannianSgd,
};
pub mod causal_inference;
pub mod causal_representation;
pub mod memory_networks;
pub mod pinn;
pub mod point_processes;
pub use causal_representation::{
    compute_mig, compute_modularity, compute_sap, CrLinear, CrMlp, DeepScm, DisentanglementMetrics,
    DscmConfig, DscmMechanism, FactorVae, FactorVaeConfig, IvaeConfig, IvaeDecoder, IvaeEncoder,
    IvaeModel, IvaePrior, NonlinearIca, SlowIcaConfig, TcDiscriminator, TcVae, TcVaeConfig,
    TcVaeDecoder, TcVaeEncoder,
};
pub mod bayesian_opt;
pub mod multimodal;
pub mod probabilistic;
pub use probabilistic::{
    nig_loss, ActNorm as ProbActNorm, CalibrationEvaluator, DeepEnsemble, DirichletOutput,
    EnsembleMember, EvidentialClassLayer, EvidentialLayer, Flow, FlowModel, FlowResult,
    GaussianProcess as ProbGaussianProcess, GpKernel, GpPrediction, IsotonicCalibrator, NigOutput,
    PlattScaling, RealNvpCoupling, SnapshotEnsemble, TemperatureScaling,
};
pub mod reward_learning;
pub use reward_learning::{
    compute_returns, cross_entropy_binary, kendall_tau, sigmoid, BradleyTerryModel, Comparison,
    CuriosityShaper, GoalRewardShaper, IrlConfig, MaxCausalEntIrl, MaxEntIrl, PreferenceDataset,
    RewardModel, RewardModelConfig, RewardModelMetrics, RewardShaper, RlhfConfig, RlhfResult,
    RlhfTrainer,
};

pub use bayesian_opt::{
    AcquisitionFunction, AcquisitionResult, BayesOptConfig, BayesOptResult, BayesSearchSpace,
    BayesianOptimizer, CmaEs, CmaEsConfig, CmaEsResult, CmaEsState, FidelityLevel, GaussianProcess,
    GpConfig, KernelType, MultiFidelityConfig, MultiFidelityOptimizer, ObservationRecord,
};

pub use graph_transformer::{
    layer_norm as gt_layer_norm, matmul as gt_matmul, softmax as gt_softmax, ChebNetLayer,
    GatConfig, GraphAttentionTransformer, GraphTransformerError, GraphTransformerLayer,
    LaplacianPositionalEncoding, PosEncodingType, RandomWalkPositionalEncoding, ReadoutType,
};

pub use graph_generation::{
    AtomType, BondType, GraphGenError, GraphPropertyPredictor, GraphRnn, GraphRnnConfig,
    MolecularFingerprint, MolecularGraph, MpnnConfig, MpnnLayer, VgaeConfig, VgaeEncoder,
};

pub use nas::{
    decode_architecture_string,
    encode_architecture_string,
    network_stats,
    // Legacy NAS components
    AgingEvolutionNas,
    ArchitectureEvaluator,
    CellConfig,
    CellEdge,
    CellEncoding,
    DartsCell,
    DartsConfig,
    DartsOptimizer,
    DartsState,
    EvoNasConfig,
    EvolutionResult,
    EvolutionaryNas,
    GumbelSoftmax,
    LotteryTicketPruner,
    MixedOp,
    NasLogger,
    NasSummary,
    NetworkEncoding,
    NetworkStats,
    NodeConfig,
    OneShotConfig,
    OneShotNas,
    // New comprehensive NAS components
    OpType,
    OpsChoice,
    RandomNasConfig,
    RandomNasSearch,
    RandomSearchNas,
    SearchSpace,
    TicketState,
};

pub use pinn::{
    BoundaryCondition, BurgersEquation, CollocationSampler, HeatEquation, LossComponents,
    NumericalGradient, PdeResidual, PinnActivation, PinnConfig, PinnLoss, PinnNetwork,
    PinnSolution, PinnTrainer, PoissonEquation, WaveEquation,
};

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
    compute_slopes, naive_attention, scaled_dot_product_attention, AlibiAttention, AlibiMask,
    AlibiSlopes, BahdanauAttention, Conv2D, Dense, Dropout, FlashAttention, FlashConfig, KVCache,
    Layer, LuongAttention, MultiHeadAttention, OnlineSoftmax, RMSNorm, RopeConfig, RopeEmbedding,
    RotaryInterpolation, TransformerDecoder, TransformerEncoder, GRU, LSTM, RNN,
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
    clip_gradients_adaptive,
    clip_gradients_by_global_norm,
    clip_gradients_by_norm,
    clip_gradients_by_value,
    AdaBelief,
    Adadelta,
    Adagrad,
    Adam,
    AdamW,
    AnnealStrategy,
    CosineAnnealingScheduler,
    ExponentialDecayScheduler,
    LambConfig,
    LambOptimizer,
    LinearScheduler,
    Lion,
    // New flat vec-based optimizers
    LionConfig,
    LionOptimizer,
    Lookahead,
    // New stateful LR schedulers
    LrScheduler,
    MetricMode,
    MuonConfig,
    MuonOptimizer,
    Nadam,
    OneCycleLrScheduler,
    Optimizer,
    ParameterGroup,
    ParameterGroupOptimizer,
    PolynomialDecayScheduler,
    RAdam,
    RMSprop,
    SchedReduceLrOnPlateau,
    WarmupScheduler,
    LAMB,
    SGD,
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

pub use text_generation_pipelines::{
    CFGDecoder, CharTokenizer, EtaSampler, GenerationCache, GreedyDecoder, MinPSampler,
    PipelineResult, RepetitionPenaltyProcessor, SamplingStrategy as TgpSamplingStrategy,
    SimpleVocab, StreamingGenerator, TemperatureScaledDecoder, TgpConfig, TgpPipelineMetrics,
    Tokenizer as TgpTokenizer, TypicalSamplerTgp,
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
    BasicBlock, BottleneckBlock, EfficientNet, EfficientNetConfig, MBConvBlock,
    PatchEmbedding as PretrainedPatchEmbedding, ResNet, ResNetBlockType, SEBlock,
    VisionTransformer as PretrainedVisionTransformer,
};

// Additional module declarations and re-exports are in reexports_ext.rs
// to keep lib.rs under the 2000-line policy limit.
include!("reexports_ext.rs");
