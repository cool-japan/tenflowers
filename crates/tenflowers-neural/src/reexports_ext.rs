pub mod compression;
pub use compression::{
    AwqQuantizer, BenchmarkRunner, BenchmarkStats, ChannelConditionalModel, CompMetrics,
    CrdDistillation, DistillationScheduler, Fp8Quantizer, GptqQuantizer, GradualMagnitudePruning,
    HyperpriorModel, LayerDropper, LotteryTicketFinder, MagnitudePruner, MixedPrecisionSearch,
    ModelBatcher, ModelProfiler, MovementPruning, MovementPrunerV2, PkdDistillation,
    QuantizationCalibrator, RdOptimizer, RequestScheduler, RkdDistillation, SmoothQuant,
    SparsityReport, StructuredChannelPruning, ThroughputMonitor, TorchScriptExporter, VocabPruner,
};

pub mod vision_transformer;
#[cfg(feature = "serialize")]
pub use serialization::{
    AdvancedModelState, AdvancedSerialization, CheckpointInfo, CheckpointLoadResult,
    CheckpointManager, CompressionAlgorithm as SerializationCompressionAlgorithm, CompressionInfo,
    LoadResult, ModelMetadata, ModelMigrator, ParameterInfo, SchemaValidator, SemanticVersion,
    SerializationConfig, ValidationResult,
};
pub use ssm::{
    HyenaFilter, HyenaOperator, MambaBlock, MambaConfig, S4Config, S4Layer, SelectiveScan,
    SsmSequenceModel,
};
pub use vision_transformer::{
    BarlowTwins, Byol, DetectionHead, MaeModel, NmsProcessor, PatchEmbedding, PatchMerging,
    RandomMasking, SegmentationHead, SimCLRv2, SwinBlock, SwinStage, VicReg, VisionTransformer,
    VitBlock,
};
pub mod state_space_models;
pub use causal_inference::{
    CausalEstimator, CausalGraph, CausalQuery, CounterfactualEstimator, CounterfactualQuery,
    CounterfactualResult, DoubleML, DoubleMLConfig, DoubleMLResult, Intervention, IpwEstimator,
    NuisanceModel, PropensityModel, PropensityScoreConfig, RddConfig, RddEstimator, RddResult,
    StructuralEquation,
};
pub use energy_models::{
    Activation, ContrastiveDivergenceTrainer, EbmClassifier, EnergyFunction, HamiltonianMonteCarlo,
    LangevinDynamics, MetropolisHastings, NeuralEnergy, NeuralLayer, QuadraticEnergy,
    QuadraticEnergyClassifier, SliceSampler,
};
pub use memory_networks::{
    allocation_weighting, dnc_read, dnc_write, usage_update, DncConfig, DncReadModes, DncState,
    MemoryBank, NeuralTuringMachine, NtmAddressing, NtmAddressingConfig, NtmAddressingState,
    NtmConfig, NtmController, NtmReadHead, NtmState, NtmWriteHead,
};
pub use state_space_models::{
    ContinuousTimeRnn, GatedRecurrentUnit, HawkBlock, HiPPoMatrix, JambaBlock,
    LinearRecurrenceLayer, LiquidNeuralNetwork, MambaBlock as SsmMambaBlock, MambaModel,
    NeuralCde as SsmNeuralCde, ParallelScan, S4Discretization, S4Kernel, S4Layer as SsmS4Layer,
    S4Model, SelectiveScanCausal, SelectiveStateSpace, SpikingNeuralNetwork, SsmEvaluator,
    TransformerSsmHybrid, ZambaBlock,
};
pub use time_series::{
    BasisType, NBeats, NBeatsBlock, NBeatsConfig, NBeatsStack, TemporalFusionTransformer,
    TftConfig, TimeSeriesMetrics, GRN, VSN,
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

pub mod causal_discovery_advanced;
pub use causal_discovery_advanced::{
    acyclicity_penalty,
    auroc_edges,
    castle_acyclicity_loss,
    castle_reconstruction_loss,
    castle_step,
    causal_boost,
    dcdi_step,
    direct_lingam,
    doubly_robust_ate,
    entropy_ica,
    estimate_propensity,
    f1_skeleton,
    fci_rules,
    fit_stump,
    grandag_loss,
    h_constraint,
    icp_find_parents,
    interventional_likelihood,
    ipw_ate,
    is_invariant,
    mutual_information_approx,
    normalized_shd,
    notears_loss,
    notears_step,
    orient_v_structures,
    regression_residuals,
    sensitivity_analysis_bounds,
    // CdMetrics
    shd,
    skeleton_search,
    // CASTLE
    CastleConfig,
    CausalAutoEncoder,
    // CausalBoosting
    CausalBoostConfig,
    CausalDecisionStump,
    // ICP
    CdEnvironment,
    // GraNDAG
    GraNDagConfig,
    // DCDI
    InterventionData,
    // DirectLiNGAM
    LinGamConfig,
    MlpCausalModule,
    // NOTEARS
    NoTearsConfig,
    Pag,
    PagEdge,
    // CausalEffect
    PropensityScoreModel as CdPropensityScoreModel,
    // FCI
    SkeltonEdge,
};

pub mod causal_discovery_ts;
pub use causal_discovery_ts::{
    CdtsCausalGraph, CdtsCausalLink, CdtsCcmConfig, CdtsCcmPoint, CdtsCcmResult, CdtsConvergentCC,
    CdtsDiscoveryMetrics, CdtsEffectResult, CdtsError, CdtsGrangerResult, CdtsGrangerTest,
    CdtsInterventionConfig, CdtsInterventionEffect, CdtsInterventionMethod, CdtsLingamTs,
    CdtsLingamTsConfig, CdtsLingamTsResult, CdtsMetrics, CdtsPcmci, CdtsPcmciConfig,
    CdtsPcmciResult, CdtsReport, CdtsTransferEntropy, CdtsTransferEntropyConfig, CdtsVarConfig,
    CdtsVarModel,
};

pub mod causal_ts;
pub use causal_ts::{
    // core
    AnomalyTransformer, CausalAttentionMechanism, CausalImpactModel, ConvergentCrossMapping,
    DeepStateSpaceModel, DifferenceInDifferences, GpCounterfactual, GrangerCausalityTest,
    GrangerResult, ImpactSummary, InverseIntensityWeighting, LocalLevelModel, LocalLinearTrend,
    NeuralSyntheticControl, PropensityScoreTs, RdEstimate, RecurrentGanForTimeSeries,
    RegressionDiscontinuity, SeasonalKalmanFilter, SyntheticControl,
    TemporalConvNet as CausalTcn, TransferEntropyEstimator, UcComponents,
    UnobservedComponentsModel, VectorAutoregression, WaveNet,
    // extensions
    CausalTsGraph, CausalTsMetrics, LagSelectionResult, RollingGrangerTest, RollingGrangerWindow,
    TransferEntropyMatrix, VarLagSelector,
    // advanced
    CausalBanditEnv, CausalThompsonSampling, CausalTsExtMetrics, CausalUcbAgent,
    ChangePointCausalDetector, KernelGrangerTest, NeuralGrangerTest, SvarForecastErrorVarianceDecomp,
    SvarImpulseResponse, SvarModel, TransferEntropyNeural, TvGrangerTest, TvVarModel,
};

pub mod continuous_normalizing_flows;
pub use continuous_normalizing_flows::{
    evaluate_cnf, evaluate_flow_matching, CnfDynamics, CnfMetrics, CnfMlp,
    ContinuousNormalizingFlow, Ffjord, FfjordBlock, FfjordConfig, FlowMatchingConfig,
    FlowMatchingModel, OtCfmModel, RectifiedFlow as CnfRectifiedFlow, RectifiedFlowConfig,
};

pub mod normalizing_flows_advanced;
pub use normalizing_flows_advanced::{
    NfaActNorm, NfaAffineCoupling, NfaFlowComposite, NfaFlowLayer, NfaFlowVAE, NfaGlowModel,
    NfaGlowStep, NfaHouseholderFlow, NfaInverseAutoregressive, NfaInvertible1x1Conv,
    NfaMaskedAutoregressive, NfaMetrics, NfaNeuralSpline, NfaRadialFlow, NfaReport,
};

pub mod climate_ml;
pub use climate_ml::{
    anomaly_correlation,
    brier_score,
    crps_ensemble,
    reliability_diagram,
    // ClimateMetrics functions
    rmse_skill_score,
    AtmosphericEncoder,
    // ClimateDownscaler
    BicubicUpsample,
    CarbonFluxEstimator,
    // ClimateProjectionEnsemble
    ClimateModel,
    ClimateSampler,
    DownscalerModel,
    // PanGu
    EarthPositionBias,
    // CarbonFluxEstimator
    EcosystemFeatures,
    EmpiricalOrthogonalFunction,
    ExtremeEventModel,
    // ExtremeEventDetector
    FocalLoss,
    // FourCastNet
    FourCastConfig,
    FourCastNet,
    LstmLayer,
    OceanLstm,
    // OceanCurrentPredictor
    OceanLstmConfig,
    PanGuWeather,
    PhotosynthesisModel,
    PressureLevelAttention,
    ProjectionEnsemble,
    ResidualDenseBlock,
    RespirationModel,
    SphericalFourierLayer,
    // TeleconnectionAnalyzer
    TeleconnectionIndex,
    // AtmosphericEmbedding
    VariableEmbedding,
};

pub mod knowledge_graph;
pub use knowledge_graph::{
    ComplExConfig, ComplExModel, KgDataset, KgTriple, KgeEvaluator, KgeModel, KgeModelExt,
    KgeModelSummary, KgeModelType, KgeTrainer, KgeTrainingResult, RotatEConfig, RotatEModel,
    TransEConfig, TransEModel,
    // Temporal KGs
    TemporalTriple, TeRoModel, TntComplExModel, TemporalKgMetrics,
    // Hyper-Relational KGs
    HypRelQuadruple, StarEModel, NaLPModel,
    // KG + LLM
    KgTextualizer, KgEmbeddingAlignment, KgQaRanker,
    // KG Inference
    HornRule, RuleInduction as KgRuleInduction, PathReasoningModel, KgMetricsExtended,
};

pub mod knowledge_distillation_advanced;
pub use knowledge_distillation_advanced::{
    compute_attention_map,
    compute_relation_matrix,
    AlphaSchedule,
    // §8 Attention Transfer
    AttentionMap,
    AttentionTransfer,
    DataGenerator,
    DreemLoss,
    EfficientTransferLearning,
    FreezeSchedule,
    // §3 Data-Free Distillation
    GeneratorConfig,
    GraphDistillation,
    KdDistilScheduler,
    // §10 Efficient Transfer Learning
    LayerGroup,
    // §1 Online Distillation
    OnlineDistilConfig,
    OnlineDistillation,
    // §5 Patch Distillation
    PatchDistilConfig,
    PatchDistillation,
    // §7 Progressive Distillation
    ProgressiveConfig,
    ProgressiveDistillation,
    // §6 Graph Distillation
    RelationMatrix,
    // §2 Self-Distillation
    SelfDistilConfig,
    SelfDistillation,
    StagedModel,
    TaskAgnosticDistillation,
    // §9 Distillation Scheduler (aliased to avoid clash with compression::DistillationScheduler)
    TempSchedule,
    // §4 Task-Agnostic Distillation
    UnlabeledDistilConfig,
    // §11 Token-Level Distillation
    TokenEmbeddingDistiller,
    AttentionMapDistiller,
    HiddenStateDistiller,
    // §12 Contrastive Distillation
    ContrastiveDistillationLoss,
    SemckdDistiller,
    // §13 Structured Distillation
    RelationalKdLoss,
    GraphDistillationLayer,
};

pub mod conformal_prediction;
pub use conformal_prediction::{
    bonferroni_correction, compute_coverage_diagnostics, conformal_p_value, AbsoluteResidual,
    AdaptivePredictionSetConfig, AdaptivePredictionSets, ConformalPredictionInterval,
    ConformalizingQr, CoverageDiagnostics, CqrConfig, CqrInterval, CrossConformal,
    CrossConformalConfig, NormalizedResidual, PredictionSet, QuantileModel, ScoreFunction,
    SignedResidual, SplitConformal, SplitConformalConfig,
};

pub use multimodal::{
    compute_multimodal_metrics, cosine_similarity,
    cross_entropy_loss as multimodal_cross_entropy_loss, l2_normalize,
    softmax as multimodal_softmax, ClipConfig, ClipLoss, ClipModel, CrossModalAttention,
    CrossModalAttentionConfig, FusionResult, FusionStrategy, LinearEncoder, ModalFusionConfig,
    Modality, ModalityEncoder, MoeConfig, MoeLayer, MultimodalBatch, MultimodalFusion,
    MultimodalMetrics,
};

pub mod optimal_transport;
pub use optimal_transport::{
    cost_matrix_euclidean, emd_1d, fused_gromov_wasserstein, gromov_wasserstein_distance,
    log_sum_exp, normalize_weights, partial_sinkhorn, sinkhorn, sliced_wasserstein, wasserstein_1d,
    wasserstein_distance, DistributionPair, FrechetDistance as OtFrechetDistance, JdotOptimizer, OtConfig, OtLoss,
    OtMetrics, OtResult, OtDaTransport, OnlineSlicedWasserstein, PartialOtAdvancedConfig,
    PartialOtConfig, PartialOtSolver, SinkhornDivergence, SinkhornLoss, SubsampledSinkhorn,
    TreeEdge, TreeWasserstein, UnbalancedOtConfig, UnbalancedSinkhorn, WassersteinBarycenter,
};

pub mod monte_carlo;
pub use monte_carlo::{
    iwae_elbo, iwae_gradient_weights, BananaDistribution, HmcConfig, HmcSampler, ImportanceSampler,
    IsResult, LogDensity, McmcDiagnostics, McmcResult, MhConfig, MhSampler, MixtureOfGaussians,
    SmcConfig, SmcResult, SmcSampler, StandardNormal, SvgdConfig, SvgdOptimizer, SvgdResult,
};

pub mod mixture_of_experts_advanced;
pub use mixture_of_experts_advanced::{
    CoarseConfig, ConditionalCompute, ConfidenceScorer, DomainClassifier, ExitConfig,
    ExpertAdapter, ExpertLoraConfig, ExpertMerger, GqaAttention, HierarchicalMoe,
    LoadBalanceReport, MegaBlockConfig, MegaBlockMoe, MixtureOfDepths, MoELoadBalancer, ModConfig,
    MoeProfileReport, MoeProfiler, RoutingLevel, SparseConfig, SparseExpert, SparseMoeLayer,
    SparseRouter, SpecialistMoe, TransformerBlockMod,
};

pub mod mixture_of_depths;
pub use mixture_of_depths::{
    ModAdaptiveDepth, ModConditionalComputation, ModEarlyExit, ModEfficiencyMetrics, ModError,
    ModPonderNet, ModReport, ModRouter, ModTransformerLayer, ModUniversalTransformer,
};

pub mod optimal_control;
pub use optimal_control::{
    quadratic_cost, CartPole, CostConfig, CrossEntropyMpc, DirectCollocation, DoubleIntegrator,
    DynamicsModel, IlqrConfig, IlqrResult, IlqrSolver, LinearDynamics, LqrConfig, LqrController,
    LqrSolution, LqrSolver, MpcConfig, MpcPlan, MppiConfig, MppiController, PendulumDynamics,
    RandomShootingMpc, TrajectoryOptConfig, TrajectoryOptResult,
};

pub mod molecular_gnn;
pub use molecular_gnn::{
    Atom, AttentiveFp, BesselBasis, Bond, ComENetLayer, DimeNetLayer, DruglikenessReport,
    EquivariantMolNet, GaussianSmearing, GraphVae, Hybridization, JtVocabEntry, JunctionTreeVae,
    LocalMapper, MolBert, MolBondType, MolConformer, MolDruglikenessFilter, MolGraph,
    MolMpnnConfig, MolecularFlowModel, MorganFingerprint, Mpnn, MpnnLayer as MolMpnnLayer,
    MultiTaskMolNet, PredictionTask, PropertyPredictor, PropertyPredictorConfig,
    ReactionClassifier, ReactionFeatures, ReactionGraph, ReactionYieldPredictor,
    RetrosynthesisPredictor, SchNet, SchNetConfig, TopologicalFingerprint, UncertaintyMolPredictor,
};

pub mod tensor_decomp;
pub use tensor_decomp::{
    hosvd, khatri_rao, matrix_multiply, matrix_qr, matrix_transpose, pseudo_inverse_via_svd, CpAls,
    CpConfig, CpDecomposition, DenseTensor, NmfConfig, NmfDecomposition, NmfMu, RandomizedSvd,
    SvdResult, TtConfig, TtSvd, TtTensor, TuckerConfig, TuckerDecomposition, TuckerHooi,
    // Advanced tensor decompositions
    NtfBeta, NtfModel, NtfSemiNmf, RiemannianGradientCompletion, ScalableTcAlternating,
    TdMetrics, TensorCompletion, TensorFusionLayer, TensorRingCore, TensorRingDecomp,
    TensorRingDecompAlgo, TensorTrainLinear, TrCompressor, TtRnn,
};

pub mod audio_models;
pub use audio_models::{
    AddNoise, AudioAugmentPipeline, BeatTracker, ChordRecognizer, ConformerBlock, ConformerEncoder,
    ContrastiveWav2VecLoss, ConvModule, CtcDecoder, DacCodec, Data2VecAudio, FastSpeech2Duration,
    FastSpeechDuration, FeatureExtractor, GeSim, GriffinLimVocoder, HubertModel, KeyDetector,
    LengthRegulator, LengthRegulatorAdv, MelSpectrogram, MusicSeparator, QuantizerCodebook,
    RvqCodebook, SoundStreamCodec, SpeakerEmbedding, SpeakerVerifier, SpecAugment, TdnnLayer,
    TtsMetrics, UniSpeechEncoder, VocoderHiFi, Wav2Vec2Model, XVectorExtractor,
};

pub mod temporal_gnn;
pub use temporal_gnn::{
    DyGFormerLayer, DynamicGraphTransformer, EventGraph, GraphOdeFunc, HeteroTemporalGraph,
    HeteroTgnModel, NeighborSampler, NodeMemory, OdeGnn, PartitionStrategy, RelationalTemporalConv,
    StGcnBlock, StGcnLayer, StGcnModel, TemporalEdge, TemporalGraphNetwork, TimeEncoder,
};

pub mod operator_learning;
pub use operator_learning::{
    BranchNet, ContinuousSensorDeepONet, DeepONet, DeepONetTrainer, FnoBlock, FnoConfig, FnoLayer,
    FnoModel, GpSymbolicRegressor, HamiltonianNN, HnnTrainer, LagrangianNN, LangevinSampler, Mlp,
    NeuralSymbolicHybrid, ScoreMatchingLoss, ScoreNetwork, SlicedScoreMatching, SpectralConv1d,
    SpectralConv2d, SymbolicExpr, SymbolicNode, SymplecticIntegrator, TrunkNet,
    // Advanced neural operators
    GnoKernel, GnoLayer, GnoModel, NeuralOperatorMetrics, PdeType, PinoLoss, PinoTrainer,
    UnoDecoder, UnoEncoder, UnoModel, UnoSkipConnection, WnoLayer, WnoModel,
};

pub mod generation;
pub use generation::{
    // Core generation utilities
    BM25Retriever, BanWordConstraint, BeamSearchDecoder, ContrastiveDecoding, DocumentChunk,
    DraftModel, FewShotPrompter, GenerationConfig as TextGenerationConfig, HybridRetriever,
    KvCache, LogitProcessor, NucleusTopKSampler, PrefixConstraint, PrefixLmDecoder, PromptTemplate,
    RagPipeline, RepetitionPenalty, SpeculativeDecoder, VectorStore,
    // Advanced generation algorithms
    MedianDraftLength, RegexFsm, FsmState, CfgRule, GrammarSampler,
    RagIndex, RagEntry, BertScoreProxy, GenerationMetrics, GenerationReport,
    // JSON-constrained generation
    JsonSchemaConstraint, JsonState,
    // Instruction formatting
    InstructionTuningFormatter,
    // Typical sampling
    TypicalSampler,
};

pub mod marl;
pub use marl::{
    Agent, AtocAgent, ComaTrainer, CommChannel, CommNet, CreditAssignment, Experience, MaddpgActor,
    MaddpgAgent, MaddpgCritic, MaddpgTrainer, MixingNetwork, MultiAgentEnv, MultiAgentTrainer,
    QmixAgent, QmixTrainer, SharedReplayBuffer, TarmacAgent, TeamRewardShaper, VdnMixer,
};

pub mod materials_ml;
pub use materials_ml::{
    add_gaussian_displacement,
    build_neighbor_graph,
    compute_descriptor,
    compute_lattice_params,
    detect_lattice_system,
    energy_above_hull,
    is_on_hull,
    // §10 MaterialMetrics
    mae as material_mae,
    point_group_order,
    r2 as material_r2,
    // §8 MaterialAugmentation
    random_rotation,
    random_supercell,
    spearman_rank_correlation,
    top_k_screening_rate,
    // §1 CrystalGraph
    Atom as CrystalAtom,
    AtomFeaturizer,
    CfConv,
    Cgcnn,
    // §2 CGCNN
    CgcnnConfig,
    CgcnnLayer,
    // §9 GenerativeMaterials
    CompositionGenerator,
    ConvexHullPoint,
    CrystalBond,
    CrystalGraph,
    DiffusionMaterialsModel,
    EdgeFeaturizer,
    // §4 MattersimModel
    ElementEmbedding,
    GaussianSmearing as MatGaussianSmearing,
    // §6 MaterialPropertyPredictor (aliased to avoid collision with molecular_gnn::PropertyPredictor)
    MaterialProperty,
    MaterialPropertyPredictor,
    MaterialPropertyPredictorConfig,
    MattersimModel,
    // §5 PhasePredictor
    PhaseConfig,
    PhasePredictor,
    SchNetLayer,
    SchNetMaterials,
    // §3 SchNetMaterials
    SchNetMaterialsConfig,
    // §7 CrystalSymmetry
    SpaceGroup,
    StructureDescriptor,
};

pub mod training_dynamics;
pub use training_dynamics::{
    AntiCurriculumTrainer, AsymSam, CurriculumSampler, DifficultyScorer, FisherSam,
    FlatnessMeasure, GradientFlowChecker, GradientHealthReport, GradientMonitor, GradientStatus,
    GrokFastScheduler, HessianTrace, LayerWiseLrScheduler, LossLandscape, MixedCurriculumScheduler,
    SamConfig, SamOptimizer, SelfPacedLearning as TdSelfPacedLearning, SharpnessMetric,
    SignalToNoiseRatio, StochasticDepthScheduler, TdOneCycleLrScheduler,
    TdPolynomialDecayScheduler, WarmupCosineScheduler,
};

pub mod moe_scaling;
pub use moe_scaling::{
    ActivationQuant, ExpertChoiceRouter, FeedForwardExpert, HashRouter, KvQuantizer,
    LinearAttention, LocalWindowAttention, LongformerAttention, MemoryEstimator,
    MoeLayer as ScalingMoeLayer, MoeTransformerBlock, PipelineStage, RouterAuxLoss, SharedExpert,
    SoftMoeRouter, TensorParallelLinear, TopKRouter, WeightOnlyQuant,
};

pub mod efficient_transformers;
pub use efficient_transformers::{
    ActivationCheckpointingMgr, AliBiPositionBias, FNetLayer, GatedLinearAttention,
    GradientCompressor, GroupedQueryAttention, LowRankAttention, Mamba2Layer, MegaLayer,
    MixedPrecisionScaler, MixerLayer, MixtureOfDepthsLayer, MultiQueryAttention,
    RetentiveNetworkLayer, ReversibleLayer, RopeEncoding, StateSpaceAttention,
    SwitchTransformerLayer, Xpos, YarnRope, ZeroRedundancyOptimizer,
    // advanced: RetNet, SSD/Mamba-2, GQA/MQA, KV Cache
    EnhancedGroupedQueryAttention, EtMetrics, Mamba2Block, MultiQuerySingleHeadAttention,
    RetNetBlock, RetNetModel, RetentionLayer, SinkTokenCache, SlidingWindowKvCache, Ssd, SsdLayer,
};

pub mod safety_alignment;
pub use safety_alignment::{
    AdversarialTrainingAugmenter, AttentionExplainer, AutoencoderAnomaly, CertifiedRobustness,
    ConceptBottleneck, ConstitutionalAiFilter, ConstitutionalPrinciple, CounterfactualExplainer,
    DefenseEnsemble, DemographicParityChecker, DpoTrainer, EnergyOodDetector, EqualizedOdds,
    InputSmoothing, IsolationForest, LimeExplainer, MahalanobisDetector, OodBenchmark, PpoWithKl,
    ReweightingDebias, ShapValues,
};

pub mod safe_rl;
pub use safe_rl::{
    build_srl_report,
    compute_srl_metrics,
    compute_srl_pareto_front,
    SrlBarrierConfig,
    SrlBarrierFn,
    SrlBarrierFunction,
    // §8 Barrier Function
    SrlBarrierType,
    SrlCmdpConfig,
    SrlCmdpEnv,
    SrlConstraintModel,
    SrlCpoAgent,
    // §3 CPO
    SrlCpoConfig,
    SrlCpoUpdateResult,
    SrlEpisodeData,
    SrlHalfspaceBarrier,
    // §2 Lagrangian RL
    SrlLagrangianConfig,
    SrlLagrangianRl,
    SrlLagrangianStepResult,
    // §0 Utilities
    SrlLinear,
    // §9 Metrics
    SrlMetrics,
    SrlMlp,
    SrlReport,
    SrlRobustMdp,
    // §6 Robust MDP
    SrlRobustMdpConfig,
    SrlSafeCartPole,
    SrlSafeExplorer,
    // §5 Safe Explorer
    SrlSafeExplorerConfig,
    SrlSafeGridWorld,
    SrlSafetyLayer,
    // §4 Safety Layer
    SrlSafetyLayerConfig,
    // §7 Shielded Policy
    SrlShieldConfig,
    SrlShieldedPolicy,
    SrlSphereBarrier,
    // §1 CMDP
    SrlStepResult,
};

pub mod graph_foundation;
pub use graph_foundation::{
    CompGcnLayer, ComplexLinkPredictor, EdgePredictionPretraining, GraphContrastivePretraining,
    GraphEpisodeSampler, GraphGPSLayer, GraphMaskedAutoencoder, GraphMatchingNetwork,
    GraphProtoNet, GraphRnnNode, GraphormerBias, GraphormerLayer, GraphormerModel, HgtLayer,
    MetaGnn, MoleculeGenerator, RelationalGcn, RotateLinkPredictor, SemanticAttention,
    // GraphCL / GraphMAE pretraining
    GclAugmentation, GclGraph, GraphCL, GraphContraster,
    GraphMaeEncoder, GraphMaeDecoder, GraphMaeModel, MaeMaskStrategy,
    GtpTokenizer, GtpModel, GtpPretrainer,
    FedGraph, CrossGraphTransfer, GraphMetrics,
};

pub mod geometric_dl;
pub use geometric_dl::{
    ChebMeshConv, DgcnnLayer, EquivariantReadout, HyperbolicEmbedding, HyperbolicLinear, KnnGraph,
    LorentzModel, PersistenceDiagram, PointCloud, PointNetLayer, RipsFiltration, SO3Features,
    SchNetEquivariant, SimplicialComplex, SurfacePooling, TFNLayer, TopoLoss, TriangleMesh,
    Vector3,
};

pub mod diffusion_advanced;
pub use diffusion_advanced::{
    AdaptiveLayerNorm, CfmModel, ClassifierFreeGuidance, ConsistencyModel, CosineNoiseSchedule,
    CrossAttentionConditioning, DiffusionLoss, DpmSolverSampler, FlowMatchingIntegrator,
    FrechetInceptionDistance, GuidedDiffusionStep, InceptionScore, InpaintingMask,
    LatentDiffusionModel, OtFlowMatching, PndmSampler, RectifiedFlow as DiffusionRectifiedFlow,
    SdeBasedSampler, VariationalDecoder, VariationalEncoder,
};

pub mod financial_ml;
pub use financial_ml::{
    AlphaFactorNet, BacktestEngine, BacktestResult, ChangePointDetector, CvarCalculator,
    DeepLobModel, ForecastMetrics, HiddenMarkovModel, HistoricalVaR, MaxDrawdown,
    MeanReversionStrategy, MomentumStrategy, NHitsLayer, OrderBookEncoder, ParametricVaR, PatchTsT,
    PortfolioOptimizer, TimeMixer, VolatilityRegimeDetector,
};

pub mod multimodal_foundation;
pub use multimodal_foundation::{
    AudioSpectrogram, AudioVisualAttention, AvContrastiveLoss, AvEncoder,
    ChainOfThoughtMultimodal, CogVlmLayer, DocumentQaModel, DocumentTokenizer,
    GatedCrossAttention, GroundingMetrics, ImageInstructionFormatter, LanguageProjector,
    LayoutAwareAttention, LlavaLoss, LlavaModel, MlpProjector, ModalityGapAnalyzer,
    MmReasoningMetrics, MultimodalAligner, PaLiModel, PerceiverResampler, PhrasalGrounder,
    ReadingOrderPrediction, RecRefDecoder, SceneEdge, SceneNode, SpeechVisualGrounding,
    SymbolicVisualReasoner, TableParser, TemporalPositionEmbedding, TimesFormerBlock,
    UnifiedEmbeddingSpace, UnifiedIOModel, UnifiedModality, UnifiedToken, VideoQaModel,
    VideoSlowFast, VideoTextAlignment, VisionEncoder as MmVisionEncoder, VisualLanguageAligner,
    VisualTokenizer,
};

pub mod bio_ml;
pub use bio_ml::{
    AdmetPredictor, AlphaFoldLoss, AminoAcid, CellTypeClassifier, ContactPredictionHead,
    ConvolutionalMotifScanner, DistanceMatrix, DrugTargetInteraction, EsmAttentionBlock,
    FingerprintSimilarity, NucleotideTokenizer, PcaReducer, ProteinTokenizer, RnaFoldingScore,
    ScRnaSeqNormalizer, SecondaryStructurePredictor, SpliceSitePredictor, TorsionAnglePredictor,
    TrajectoryInference, VirtualScreening,
    // Genomics extensions
    ChromatinAccessibility, CoxPh, DeepSurv, DnaConvNet, DnaTokenizer, KaplanMeier,
    LeidenClustering, MoFa, OmicsAttentionFusion, OmicsDataset, PathwayEnrichment,
    ScRnaMatrix, ScvaeDecoder, ScvaeEncoder, ScvaeModel, SurvivalMetrics, VariantEffectPredictor,
};

pub mod medical_imaging;
pub use medical_imaging::{
    bce_loss, compute_segmentation_metrics, connected_components_2d, dice_loss,
    AdaptiveInstanceNorm, AttentionGate, DiceBCELoss, DoubleConv, HausdorffDistance,
    MedicalAugmentation, NnUNetNormalizer, SegMetrics, SegmentationMetrics, SlidingWindowInference,
    UNet2D,
};

pub mod graph_signal;
pub use graph_signal::{
    AdaptiveGraphConv,
    AsapPooling,
    BandpassGraphFilter,
    BandpassGraphFilter as GspBandpassFilter,
    ChebyshevConv,
    DiffPoolLayer,
    DiffusionProcess,
    EdgeFlow,
    EdgeWeightLearner,
    GraphCoarsening,
    GraphFilter,
    GraphLaplacian,
    GraphStructureLearning,
    GraphWienerFilter,
    HarmonicAnalysis,
    HierarchicalPool,
    HodgeLaplacian,
    IterativeGraphRefinement,
    NgramGraphBuilder,
    PersistencePair,
    PersistentHomologyLayer,
    PoolLevel,
    SagPoolLayer,
    SignalInterpolation,
    SimplicialComplexData,
    SimplicialSignalDenoise,
    SpectralConv,
    TdaFeatureExtractor,
    // Additional exports
    WaveletTransformGraph,
};

pub mod neuro_symbolic;
pub use neuro_symbolic::{
    ArcConsistency, CausalDiscovery, CspVariable, DistMultModel, FuzzyLogic, GnnCspSolver,
    InterventionLayer, LogicTensorNetwork, NeuralProgramSearcher, NeuralTheoremProver,
    PathQueryEmbedding, ProductFuzzy, ProgramAst, ProgramEmbedding, ProgramToken, RuleInduction,
    SatisfiabilityLoss, StructuralCausalModel, TransRModel,
};

pub mod neuromorphic;
pub use neuromorphic::{
    compute_membrane_stats, compute_snn_metrics, AdexConfig, AdexNeuron, FastSigmoid, LifConfig,
    LifNeuron, LiquidStateMachine, LsmConfig, PiecewiseLinear, PopulationEncoder, SnnMetrics,
    SpikeEncoder, SpikeEncoding, SpikingLinear, StdpConfig, StdpSynapse, SuperSpike,
    SurrogateGradient,
};

pub mod neural_rendering;
pub use neural_rendering::{
    CameraModel, DepthEstimationNet, Gaussian2D, Gaussian3D, GaussianOptimizer,
    GaussianSplatRenderer, InstantNgp, MultiViewConsistencyLoss, NeRFLoss, NeRFMlp, NeuralSdf,
    OccupancyNetwork, PoseEstimator, PositionalEncoding, QuaternionOps, RayMarcher,
    SemanticNerfDecoder, SirenLayer, SirenNetwork, SurfaceNormalEstimator, ViewInterpolator,
    VolumeRenderer,
    // advanced: 3DGS, NRC, Deformable NeRF, NrMetrics
    // Note: advanced::Gaussian3D is aliased to avoid conflict with the mod-level Gaussian3D
    DeformationField, DynamicNerf, GaussianDensification, GaussianSplatter,
    NrcCache, NrMetrics, Reservoir, ReSTIR,
};

pub mod quantum_ml;
pub use quantum_ml::{
    AnsatzCircuit, BitFlipChannel, CircuitLayer, Complex64, CostHamiltonian, DataEncodingLayer,
    DepolarizingNoise, GroverSearch, HybridQuantumClassical, NoisyQuantumSimulator,
    NaturalGradientQnn, ParameterShiftGradient, PauliHamiltonian, Qaoa, QuantumAnnealingSimulator,
    QuantumBackpropagation, QuantumCircuit, QuantumGate, QuantumKernel, QuantumLayer, QuantumSvm,
    QubitState, ReadoutErrorMitigation, VqeOptimizer, ZeroNoiseExtrapolation,
};
pub use quantum_ml::{
    IqpFeatureMap, MaxCutQaoa, MeasurementErrorMitigation, ProbabilisticErrorCancellation,
    QaoaLayer, QaoaMetrics, QaoaOptimizer, QbmModel, QbmTrainer, QuantumFeatureMapType,
    QuantumKernelAlignment, QuantumKernelFull, ZneExtrapolation, ZzFeatureMap,
};

pub mod privacy_ml;
pub use privacy_ml::{
    ClusterFederated, DpMechanism, DpSgdOptimizer, FedAvgAggregator, FedMetrics, FedNova,
    FedProxAggregator, FedYogi, FederatedBenchmark, FederatedDistillation, GradientInversionAttack,
    HomomorphicAdd, LaplaceMechanism, MaskedAggregation, MembershipInferenceAttack,
    ModelExtractionDefense, PerFedMetaLearning, PersonalizedFedAvg, PrivacyBudgetTracker,
    RenyiAccountant, SecretSharing, SecureAggregationProtocol, WatermarkDefense,
};

pub mod program_synthesis_ml;
pub use program_synthesis_ml::{
    ASTEncoder, AstNode, BugLocalizerGnn, CfgNode, CodeBert, CodeBertConfig, CodeContrastive,
    CodeMetrics, CodeSummarizer, CodeTokenizer, ControlFlowGraph, DifferentiableInterpreter,
    FlashFillProgram, FlashFillSolver, HalsteadMetrics, Instruction, Language, MutationOperator,
    OpCode, PointerGeneratorConfig, StringDsl, TestCase, TestCaseGenerator, Token, TokenKind,
    // neural_exec exports
    BeamCandidate, DifferentiableVm, DslValue, ExampleIo, IoEmbedder, NpiController,
    ObservationalEquivalence, PcfgRule, ProgramEntry, ProgramLibrary, ProgramSearchBeam,
    PsMetrics, RecursiveNpi, SimpleType, Substitution, SyntaxGuidedSearch, TapeLanguage,
    TapeOp, TypeGuidedEnumerator, TypedDsl, TypedDslExpr, unify,
};

pub mod recommendation_systems;
pub use recommendation_systems::{
    build_popularity_map, BERT4Rec, Bert4RecConfig, BprLoss, CosineSimilarity,
    DotProductSimilarity, LightGCN, LightGcnConfig, MatrixFactorization, MfConfig, NcfConfig,
    NegativeSampler, NegativeSamplingStrategy, NeuralCF, RecMetrics, RecSysError,
    RecommendationEvaluator, SasRec, SasRecConfig, SessionEncoder, SessionEncoderConfig,
};

pub mod network_science;
pub use network_science::{
    basic_reproduction_number, biased_random_walk, node2vec_train, sir_simulate, sir_step,
    skip_gram_update, CentralityMeasures, CommunityDetection, LinkPrediction, NetTemporalEdge,
    NetworkGraph, NetworkMotifFinder, NetworkRobustness, NetworkStatistics, Node2VecConfig,
    SirConfig, SirState, TemporalNetwork,
};

pub mod nlp_components;
pub use nlp_components::{
    AbstractiveSummarizer, BleurtProxy, CharacterNgram, ChunkingDecoder, CoReferenceResolver,
    // ConstituencyParser is re-exported from structured_prediction (lib.rs line 722)
    CrfLayer, DataCollator, ExtractiveSummarizer, NerModel, OpenDomainQa,
    ParaphraserModel, RetrieverReader, RougeMetric, SemanticSimilarity, SentenceEncoder,
    SpanExtractionQa, SubwordTokenizer, TextNormalizer, TextualEntailment, TriviaQaEvaluator,
};

pub mod online_learning;
pub use online_learning::{
    // Core online learning algorithms
    Adwin, BloomFilter, ConfidenceIntervalTracker, CountMinSketch, CumulativeRegretTracker,
    DdmDetector, EpsilonGreedy, FollowTheRegularizedLeader, HyperLogLog, KsTest, LinUcb,
    NeuralBandit, OnlineAdaGrad, OnlineAdam, OnlineMetricsTracker, OnlineRocAuc, OnlineSgd,
    PageHinkleyTest, ReservoirSampler, ThompsonSampling, Ucb1, WindowedStatistics,
    // Advanced bandit algorithms
    LinUcbBandit, ThompsonSamplingLinear, CascadeLinUcb, NeuralBanditUcb,
    // Online convex optimization
    OgdOptimizer, FtrlOptimizer, OnlineNewton, AdaptiveRegretOptimizer,
    // Streaming anomaly detection
    OnlineIsolationForest, HstTree, LodaDetector,
    // Concept drift detection
    AdwinDetector, PageHinkley, DriftMetrics,
};

pub mod synthetic_data;
pub use synthetic_data::{
    ArimaSimulator, BootstrapSampler, ColumnTransformer, CopulaModel, DiversityMetric,
    FidelityMetrics, FractionalBrownianMotion, GarchSimulator, GaussianMixture,
    KernelDensityEstimator, MissingValueImputer, MultivariateGbm, NoisyLabelGenerator,
    OrnsteinUhlenbeck, PerlinNoise, PoissonDiskSampling, PrivacyMetrics, SmoteOversampler,
    SyntheticShapeGenerator, UtilityEvaluator,
};

pub mod video_understanding;
pub use video_understanding::{
    ActionRecognitionHead, ConsistencyModel as VideoConsistencyModel, DeformableConv2D, Dino4Video, HeatmapPoseHead,
    Hiera, MemoryBank as VosMemoryBank, MemoryReader, OpticalFlowRAFT, RelativePositionBias3D,
    TemporalActionDetector, TemporalShiftModule, TimeSformer, VideoAugmentation, VideoContrastive,
    VideoLdmUnet, VideoMAE, VideoMaeV2, VideoSwinBlock, VideoSwinBlockV2, VideoTransformerBlock,
    VosDecoder, VuMetrics,
};

pub mod functional_data_analysis;
pub use functional_data_analysis::{
    BSplineBasis, ElasticRegistration, FdaBasis, FdaBasisKind, FdaDataset, FdaEvalReport,
    FdaMetrics, FdaObservation, FnnConfig, FnnLayer, FourierBasis, FpcaModel, FpcaResult,
    FrechetMean, FunctionOnScalar, FunctionalNeuralNetwork, LegendreBasis, ScalarOnFunction,
};

pub mod world_models;
pub use world_models::{
    augment_state,
    // EfficientZero
    consistency_loss,
    muzero_loss,
    stoch_straight_through,
    // LatentPlanner
    Cem,
    // ObsDecoder
    DecoderConfig,
    // IRIS
    DiscreteTokenizer,
    DreamerV3,
    EfficientZeroModel,
    IrisWorldModel,
    LatentPlanner,
    // MuZero
    MuZeroConfig,
    MuZeroDynamicsNet,
    MuZeroModel,
    MuZeroPredictionNet,
    MuZeroRepresentationNet,
    ObsDecoder,
    // Predictive Coding
    PcLayer,
    PcNetwork,
    RecurrentModel,
    RepresentationModel,
    // DreamerV3 / RSSM
    RsssmConfig,
    // TDM
    TdmConfig,
    TdmNetwork,
    TransformerWorldModel,
    TransitionModel,
    // Transformer World Model
    TwmConfig,
    TwmStateEncoder,
    TwmTransitionHead,
    // WmRewardPredictor (aliased to avoid collision with robotics::RewardPredictor)
    WmRewardPredictor,
};

pub use point_processes::{
    ContinuousLstmCell, Event, EventSequence, HawkesProcess, NhpModel, RmtppModel,
    TemporalEncoding, ThpModel, ThpOutput, TppEvalReport, TppLoss, TppMetrics,
};

pub mod emotion_recognition;
pub use emotion_recognition::{
    differential_entropy,
    discrete_to_va,
    evaluate_continuous,
    evaluate_discrete,
    extract_acoustic_features,
    extract_band_power,
    extract_features,
    extract_mfcc,
    extract_pitch,
    va_to_discrete,
    // §4 Speech emotion recognizer
    AcousticFeatures,
    // §2 Facial AU encoder
    ActionUnit,
    AuFacsRules,
    AuProfile,
    // §1 Taxonomy
    BasicEmotion,
    // §3 EEG emotion network
    EegBand,
    EegEmotionNet,
    EegFeatures,
    EmotionEvalReport,
    EmotionLabel,
    // §7 Evaluation
    EmotionMetrics,
    // §6 Continuous tracker
    EmotionState,
    EmotionTrajectory,
    EmotionTransitionModel,
    ErFusionStrategy,
    FacialAuEncoder,
    KalmanEmotionFilter,
    // §5 Multi-modal fusion
    ModalityPrediction,
    MultiModalEmotionFuser,
    SpeechEmotionRecognizer,
    ValenceArousal,
};

pub mod trajectory_prediction;
pub use trajectory_prediction::{
    evaluate as traj_evaluate,
    gmm_nll,
    sample_trajectory as traj_sample_trajectory,
    AgentState,
    // §6 GMM decoder
    BivariateGaussian,
    GmmDecoder,
    GmmTrajOutput,
    // §3 LSTM predictor
    LstmConfig,
    LstmTrajPredictor,
    // §5 Transformer predictor
    MotionEncoder,
    // §1 Core data structures
    Point2D,
    SceneAttention,
    SceneContext,
    // §2 Social Force Model
    SfmConfig,
    SocialForceModel,
    SocialLstm,
    // §4 Social LSTM
    SocialPooling,
    TrajEvalReport,
    TrajLstmCell,
    // §7 Metrics
    TrajMetrics,
    Trajectory,
    TransformerTrajPredictor,
    Velocity2D,
};

pub mod cross_modal_retrieval;
pub use cross_modal_retrieval::{
    cmr_cosine_similarity, l2_distance, CmrCrossModalAttention, CmrModalityEncoder, DistanceMetric,
    DualEncoder, EmbedderConfig, EncoderConfig, FlatIndex, ModalityType, MultiModalEmbedder,
    RetrievalEvalReport, RetrievalMetrics, RetrievalResult, ZeroShotClassifier,
};

pub mod topological_ml;
pub use topological_ml::{
    CoverInterval, MapperAlgorithm, MapperConfig, MapperGraph, MapperNode, PersistenceComputer,
    PersistenceImage, PersistenceLandscape, TdaEvalReport, TdaFeatures, TdaMetrics,
    TdaPersistenceDiagram, TdaPersistencePair, TdaPiConfig, TdaSimplex, TdaSimplicialComplex,
    TdaWeightFn, TopologicalFeatureExtractor, VietorisRipsComplex,
};

pub mod neural_combinatorial;
pub use neural_combinatorial::{
    knapsack_density,
    // §7 Metrics
    optimality_gap,
    tour_validity,
    // §3 Attention Model
    AmConfig,
    AttentionModel,
    AttentionModelEncoder,
    // §4 REINFORCE
    BaselineType,
    CityEmbedding,
    CombBeamSearchDecoder,
    // §6 Beam search
    CombBeamState,
    CombEvalReport,
    CompatibilityDecoder,
    EncoderLayer as CombEncoderLayer,
    GreedyKnapsack,
    KnapsackInstance,
    // §5 Greedy heuristics
    NearestNeighbor,
    PointerNetwork,
    PtrDecoder,
    PtrEncoder,
    // §2 Pointer Network
    PtrNetConfig,
    ReinforceConfig,
    ReinforceTrainer,
    SavingsAlgorithm,
    // §1 Problem representations
    TspInstance,
    TwoOpt,
    VrpInstance,
};

pub mod sparse_learning;
pub mod sparse_mixture_experts;
pub use sparse_learning::{
    coherence_bound, evaluate, generate_matrix, init_attention_matrices, measure,
    reconstruction_error, recovery_quality, rip_constant_estimate, sparsity_ratio, BasisPursuit,
    BasisPursuitDenoise, BigBirdAttention, ChannelImportanceCriterion, ChannelPruner, CoSaMP,
    CompressedSensingMatrix, CsMatrix, CsMeasurement, CsMatrixType, CsMetrics, Dictionary,
    DlConfig, ElasticNetRegression, GroupLasso, KSvd, LassoEncoder, LassoRegression,
    LayerPruner, ListaNetwork, MatchingPursuit, MeasurementMatrix, MpResult,
    OnlineDictionaryLearning, OrthogonalMatchingPursuit, PredictiveCodingLayer,
    RecoveryGuarantees, SaeConfig, SparseAttentionRouter, SparseAutoencoder, SparseCode,
    SparseCodingLayer, SparseGroupLasso, SparseLearningReport, SparsePositionEncoding,
    SparseSlidingWindowAttention, SparsityMeasure, StructuredPruningMask, SubspacePursuit,
    WeakMatchingPursuit,
};

pub mod image_generation_advanced;
pub mod implicit_neural_repr;
pub mod influence_functions;
pub mod information_theory;
pub mod inverse_rl;
pub use image_generation_advanced::{
    discriminator_loss, generator_loss, gradient_penalty, r1_regularization, AdaInLayer,
    BigGanResBlock, ConditionalBatchNorm, EmaWeights, FrechetDistance, GanLossType,
    ImageGenEvalReport, ImageTensor, InceptionScore as IgaInceptionScore, MappingNetwork,
    PathLengthRegularization, PerceptualLoss as IgaPerceptualLoss, SpectralNormLinear,
    StyleGanGenerator, SynthesisBlock, VqCodebook, VqGan, VqGanDecoder, VqGanEncoder,
};

pub mod domain_adaptation;
pub use domain_adaptation::{
    align_features, class_conditional_shift, compute_covariance, coral_loss, da_evaluate,
    domain_gap, domain_statistics, mmd_loss, multi_kernel_mmd, proxy_a_distance, update_bn_stats,
    BatchNormStats, CoralAdapter, DaEvalReport, DaFeatureExtractor, DaLabelPredictor, DaRbfKernel,
    DannDomainClassifier, DannModel, DomainDataset, DomainSample, DomainStats, EntropyMinimization,
    GradientReversalLayer, IrmTrainer, MixupDomain, MmdAdapter, StyleTransferDa, TentAdapter,
    TestTimePseudoLabeling,
};

pub mod graph_ml_sampling;
pub use graph_ml_sampling::{
    cluster_modularity, link_prediction_auc, node_classification_accuracy, ClusterConfig,
    ClusterGcn, EdgeSampler, GmsAggregatorType, GmsCsrGraph, GmsGcnLayer, GmsGraphSageLayer,
    GmsGraphSageModel, GmsNeighborSampler, GmsSubgraphSample, GraphBatch, GraphCluster,
    NodeFeatures, NodeSampler, NormalizationWeights, SaintSubgraph, SamplerConfig,
    ScalableGnnReport, SignConfig, SignModel,
};

pub mod drug_discovery;
pub use drug_discovery::{
    // §8 Metrics
    auc_roc_virtual_screening,
    boltzmann_enhanced_discrimination,
    dd_tanimoto,
    ecfp4,
    evaluate_campaign,
    scaffold_hop_rate,
    // §7 Activity cliffs
    ActivityCliff,
    ActivityCliffAnalyzer,
    AdmetDescriptors,
    AdmetModel,
    // §3 ADMET
    AdmetProperty,
    // §5 Scaffold analysis
    BemisMurckoScaffold,
    DdAtom,
    // §1 Molecular representation
    DdAtomType,
    DdBond,
    DdBondType,
    // §2 Fingerprints
    DdMolecularFingerprint,
    DdMolecule,
    DiversityPicker,
    // §4 Virtual screening
    DockingProxy,
    DrugDiscoveryReport,
    PropertyConditionedGeneration,
    ScaffoldCluster,
    SmileRnn,
    // §6 Generative models
    SmileTokenizer,
    VirtualScreener,
};

pub mod statistical_testing;
pub use statistical_testing::{
    chi2_p_value,
    normal_cdf,
    normal_quantile,
    t_distribution_p_value,
    Bootstrap,
    // Bootstrap
    BootstrapConfig,
    BootstrapResult,
    CalibrationAnalysis,
    CochranQ,
    // Multiple comparisons
    Correction,
    FTest,
    FriedmanTest,
    KruskalWallis,
    MannWhitneyU,
    // McNemar / Cochran
    McNemarTest,
    MultipleComparisonCorrection,
    // Non-parametric
    NonParametricTests,
    // Parametric
    ParametricTests,
    // Calibration
    ReliabilityDiagram,
    // Core result type
    StTestResult,
    StatTestMetrics,
    // Metrics
    StatisticalReport,
    TTest,
    WilcoxonSignedRank,
};

pub mod satellite_ml;
pub use satellite_ml::{
    change_detection_f1,
    haversine_distance,
    kappa_coefficient,
    // §7 SatelliteMetrics
    overall_accuracy,
    psnr,
    ssim,
    // §2 ChangeDetection
    ChangeMap,
    CvaChangeDetection,
    DifferenceCD,
    // §6 GeoAiUtils
    GeoBbox,
    GeoPoint,
    // §4 LandCoverClassification
    LandCoverClass,
    LandCoverMap,
    MultispectralImage,
    NeuralChangeDetector,
    ObjectBasedSegmentation,
    PanSharpening,
    PixelClassifier,
    PolarimetricDecomposition,
    SarClassifier,
    // §3 SarProcessor
    SarImage,
    SatResidualBlock,
    SatelliteEvalReport,
    SatelliteSrModel,
    SpeckleFilter as SatSpeckleFilter,
    // §1 SpectralData
    SpectralBand,
    SpectralIndices,
    // §5 SuperResolution
    SrConfig,
    TileIndex,
    UtmProjection,
};

pub mod hypernetworks;
pub use hypernetworks::{
    compression_ratio,
    // §7 HyperNetMetrics
    parameter_count_generated,
    task_adaptation_loss,
    // §3 ChunkedHyperNetwork
    ChunkConfig,
    ChunkedHyperNetwork,
    // §2 DynamicNetwork
    DynamicLayer,
    DynamicNetwork,
    FastWeightCell,
    // §6 FastWeightProgrammer
    FastWeightStore,
    HnFilmLayer,
    HnFilmNetwork,
    HnParamPredictor,
    HnSupportEncoder,
    // §5 TaskConditioned (FiLM)
    HnTaskVector,
    // §4 HyperTransformer
    HtConfig,
    // §1 HyperNetwork
    HyperConfig,
    HyperNetReport,
    HyperNetwork,
    HyperTransformer,
};

pub mod probabilistic_circuits;
pub use probabilistic_circuits::{
    evaluate_node,
    pc_ancestral_sample,
    pc_bic_score,
    pc_conditional_sample,
    pc_evaluate,
    pc_log_likelihood,
    pc_marginalize,
    // §7 PcMetrics
    pc_mean_log_likelihood,
    pc_mpe,
    pc_partition_function,
    pc_perplexity,
    pc_sample_batch,
    // §6 PcSampling
    pc_sample_leaf,
    ChowLiuTree,
    NaiveFactorization,
    // §5 PcLearning
    PcEmLearner,
    PcEvalReport,
    // §2 PcEval
    PcEvidence,
    PcGradientLearner,
    PcGraph,
    PcLeafDist,
    // §4 ChowLiuTree
    PcMutualInformation,
    PcNode,
    // §1 PcGraph
    PcNodeType,
    PcPrimMst,
    RandomStructure,
    RegionGraph,
    // §3 SPN constructions
    SpnConfig,
};

pub mod neural_process;
pub use neural_process::{
    evaluate_np,
    // §1 MLP
    ActivationFn,
    AttentiveNeuralProcess,
    // §4 ANP
    AttentiveNpConfig,
    // §3 CNP
    CnpConfig,
    ConditionalNeuralProcess,
    GaussianNeuralProcess,
    // §6 GNP
    GaussianNpConfig,
    LatentEncoder,
    LatentNeuralProcess,
    // §5 LNP
    LatentNpConfig,
    Mlp as NpMlp,
    MlpConfig,
    // §2 Context Encoder
    NpContextEncoder,
    // §7 Trainer
    NpEpisode,
    // §8 Metrics
    NpMetrics,
    NpTrainer,
};

pub mod embodied_ai;
pub use embodied_ai::{
    ContinuousNavEnv,
    Dmp,
    DmpConfig,
    EmbActionType,
    // §1 EmbodiedEnv
    EmbObsType,
    EmbodiedEvalReport,
    EmbodiedObservation,
    // §6 EmbodiedEval
    EpisodeResult,
    ForwardDynamicsModel,
    GoalEncoder,
    // §3 ManipulationLearning
    GraspCandidate,
    GraspPlanner,
    GridWorldEnv,
    HierarchicalPolicy,
    HighLevelPolicy,
    // §5 HierarchicalPolicy
    HrlConfig,
    // §4 EmbodiedPretraining
    InverseDynamicsModel,
    LowLevelPolicy,
    ManipulationMetrics,
    NavigationMetrics,
    ObjectNavPolicy,
    PointGoalPolicy,
    TemporalContrastive,
    // §2 VisualNavigationPolicy
    VisualEncoder,
};

pub mod kolmogorov_arnold;
pub use kolmogorov_arnold::{
    compute_kan_metrics, BSplineBasis as KanBSplineBasis, KanActivation, KanConfig, KanLayer,
    KanMetrics, KanModel, KanPinn, KanReport, KanSymbolicExtractor, KanTrainer, SymbolicFit,
    SymbolicLibrary,
};

pub mod mixture_of_modalities;
pub use mixture_of_modalities::{
    compute_mom_metrics, evaluate_cross_modal_alignment, AnyModalEmbedder, AnyModalEmbedderConfig,
    AudioFrameTokenizer, CrossModalGenerator, CrossModalGeneratorConfig, GenerationMode,
    ImagePatchTokenizer, ModalProjector, ModalSequence, ModalToken, ModalityData,
    ModalityEmbedding, ModalityExpert, ModalityRouter, ModalityRouterConfig, ModalityTokenizer,
    MomMetrics, MomModalityMetrics, MomModalityType, MomReport, MultiModalPretrainer,
    PretrainingConfig, TabularTokenizer, TextModalityTokenizer, UnifiedTransformer,
    UnifiedTransformerConfig, UnifiedTransformerLayer,
};

pub mod uncertainty_quantification;
pub use uncertainty_quantification::{
    // §7 Calibration
    UqCalibration,
    // §3 Conformal Regression
    UqConformalRegression,
    UqDeepEnsemble,
    // §2 Deep Ensemble
    UqDeepEnsembleConfig,
    UqEnsembleResult,
    // Error
    UqError,
    // §6 Evidential Deep Learning
    UqEvidentialDL,
    UqEvidentialResult,
    // §4 Laplace Approximation
    UqLaplaceApprox,
    UqMCDropout,
    // §1 MC Dropout
    UqMCDropoutConfig,
    UqMCDropoutResult,
    // §10 Metrics
    UqMetrics,
    UqOodDetector,
    UqOodEvalResult,
    // §8 OOD Detector
    UqOodMethod,
    UqPriorNetResult,
    // §5 Prior Networks
    UqPriorNetworks,
    UqReport,
    UqRiskControl,
    // §9 Risk Control
    UqRiskFn,
};

pub mod zero_shot_learning;
pub use zero_shot_learning::{
    BilinearCompatibility,
    // §1 SemanticSpace
    ClassAttributes,
    CompatibilityType,
    DeVise,
    DomainShiftCorrection,
    // §2 VisualSemanticMapping
    LinearCompatibility,
    SaeDecoder,
    // §4 SemanticAutoencoder
    SaeEncoder,
    SemanticAutoencoder,
    SemanticSpace,
    // §6 TransductiveZsl
    StructuredPrediction,
    WordVector,
    ZslClassifier,
    ZslEvalReport,
    // §7 ZslMetrics
    ZslMetrics,
    // §3 ZSL Classifier
    ZslPrediction,
    // §5 GenerativeZsl
    ZslVae,
};

pub mod mechanistic_interpretability;
pub use mechanistic_interpretability::{
    build_mi_report, compute_mi_metrics, ActivationCache, ActivationPatcher, AttentionAnalyzer,
    LayerLogitLensResult, LayerPatchResult, LogitLens, MiLayer, MiMetrics, MiReport, MiSaeConfig,
    MiSparseAutoencoder, MiTransformer, PatchResult, ProbeClassifier, ProbeConfig,
};

pub mod test_time_adaptation;
pub use test_time_adaptation::{
    correlation_alignment_loss, entropy_minimization_loss, evaluate_tta, maximum_mean_discrepancy,
    EpisodicAdaptConfig, EpisodicAdapter, OnlineBnConfig, OnlineBnLayer, OnlineFeatureStats,
    RotationLabel, TentConfig, TentModel, TtaBatchNormStats, TtaLinear, TtaMetrics, TtaMlp,
    TttConfig, TttModel, TttPlusConfig, TttPlusModel,
};

pub mod test_time_compute;
pub use test_time_compute::{
    compute_ttc_metrics, AnswerExtractor, BestOfNConfig, BestOfNResult, BestOfNSampler,
    BudgetConfig, BudgetForcer, ComputeUsage, ExactMatchVerifier, MctsConfig, MctsNode, MctsResult,
    MctsSolver, MctsTree, NeuralVerifier, PrmConfig, ProcessRewardModel, SelfConsistencyConfig,
    SelfConsistencyDecoder, SelfConsistencyResult, StepBeamConfig, StepBeamResult,
    StepBeamSearcher, TtcMetrics, TtcReport, Verifier, VerifierEnsemble,
};

pub mod concept_learning;
pub use concept_learning::{
    compute_concept_completeness, concept_mutual_information, evaluate_concepts, AceConfig,
    AceExplainer, CbmConfig, ClLinear, ClMlp, ClProbeConfig, Concept, ConceptActivation,
    ConceptActivationVector, ConceptBottleneckModel, ConceptMetrics, ConceptShap,
    ConceptShapConfig, LinearProbe, MultiProbe, TcavAnalyzer, TcavConfig,
};

pub mod cooperative_game_theory;
pub use cooperative_game_theory::{
    BanzhafIndex, CfrInfoSet, CfrNode, CfrSolver, CgtCooperativeGame, CgtError, CgtGame,
    CgtMetrics, CgtNeuralNash, CoreSolver, CorrelatedEquilibrium, EvolutionaryGameDynamics,
    MeanFieldEquilibrium, MechanismDesign, NashEquilibriumSolver, NashNet, RegretAlgorithm,
    RegretMinimizer, ShapleyValueCalculator, SymmetricGame,
};

pub mod graph_neural_ode;
pub use graph_neural_ode::{
    CgnnConfig, CgnnModel, DgwConfig, DifferentialGraphWiring, EventBasedOde, EventOdeConfig,
    GnoClassifierBackbone, GnoError, GnoGatLayer, GnoGcnLayer, GnoGraph, GnoGraphODE,
    GnoMessagePassing, GnoNodeClassifier, GnoOdeMethod, GnoSimMetrics, GrandConfig, GrandModel,
    GraphOdeConfig, GraphOdeMetrics, GraphOdeModel, GraphOdeSolver, HamiltonianGnn, HgnConfig,
    LagrangianGnn, LatentGraphOde, LatentGraphOdeConfig, LatentStochasticGraph, LgnConfig,
    LsgConfig, ParticleSimGraph, PsgConfig, RdGnnConfig, ReactionDiffusionGnn, SgnoConfig,
    StGnnOde, StGnnOdeConfig, StochasticGnoModel, TemporalNodeEmbedding, TgodeConfig, TgodeModel,
};

pub mod mean_field_games;
pub use mean_field_games::{
    DeepMfgSolver, ExtendedMfgGame, LinearQuadraticMfg, LqMfgParams, McKeanVlasovDynamics,
    MckeanVlasovConfig, MeanFieldNashSolver, MfgAction, MfgCostFunction, MfgDeepConfig,
    MfgDeepResult, MfgDistribution, MfgFokkerPlanckConfig, MfgFokkerPlanckSolver, MfgHjbConfig,
    MfgHjbSolver, MfgMetrics, MfgMultiPopulation, MfgNashConfig, MfgNashResult, MfgPolicyNetwork,
    MfgPopulationConfig, MfgPopulationSimulator, MfgQuadraticCost, MfgState,
    // advanced
    CVaRMfgObjective, ErgodConstantEstimator, ExponentialUtility, GraphonEquilibrium,
    GraphonKernel, GraphonMfgConfig, GraphonMfgSolver, MfcCostFunctional,
    MfcPolicyGradient, MfcPolicyGradientConfig, MfcTrainResult, MfcValueFunction,
    MfgMetricsExtended, RiskSensitiveMfgConfig, RiskSensitiveMfgSolver,
    StationaryMfgConfig, StationaryMfgSolver, default_quadratic_cost,
};

pub mod adaptive_computation;
pub use adaptive_computation::{
    top_k_indices_f64, AcEarlyExitNetwork, AcError, AcGatingMechanism, AcGatingStrategy, AcMetrics,
    AcMixtureOfDepthsLayer, AcMlpBlock, ActConfig, ActLayer, AdaptiveDepthNetwork,
    AdaptiveMixturLayer, ConditionalDepthRouter, DynamicSlimmingLayer, LayerDropNetwork,
    PonderNetBlock, PonderingState, SkimmingModel,
};

pub mod pomdp_planning;
pub use pomdp_planning::{
    AlphaVector, BeliefMdpSolver, BeliefState, BtNode, FibAlgorithm, FscNode,
    OnlineBeliefTreeSearch, PbviSolver, PerseusAlgorithm, PomcpActionNode, PomcpNode, PomcpSolver,
    PomdpError, PomdpGenerator, PomdpMetrics, PomdpModel, PomdpPolicyGraph, QmdpApproximation,
    SarsopAlgorithm,
};

pub mod protein_structure;
pub use protein_structure::{
    AlphaFoldLite, ContactMapPredictor, InvariantPointAttention, MsaEncoder,
    PairwiseRepresentation, ProteinEvoformer, ProteinLanguageModelEmbed, ProteinMetrics,
    ProteinSequence, ProteinStructure, PsAminoAcid, PsError, Residue3D, StructureModule,
};

pub mod mixture_density_networks;
pub use mixture_density_networks::{
    BayesianMdn, ConditionalMdn, ConditionalVaeMdn, DensityEstimationBenchmark, MdnError,
    MdnGaussianMixture, MdnLinear, MdnMadeNetwork, MdnMetrics, MdnOptimizer, MdnTrainer,
    MdnTrainerConfig, MixtureDensityNetwork, MixtureLstmModel, NormalizingFlowMdn,
    RnadeDensityEstimator,
};

pub mod edge_optimization;
pub use edge_optimization::{
    CodebookQuantization, CodebookResult, CpFactors, DynamicWidthNetwork, EdgeMetrics, EdgeReport,
    EoAllocationPlan, EoArchCandidate, EoCpDecomposition, EoFusionOp, EoLayerDesc, EoLayerType,
    EoSlimmableLinear, EoTuckerDecomposition, FixedPoint, HardwareAwareSearch, HardwareProfile,
    IntegerLinear, MemoryBudgetAllocator, PqCodes, ProductQuantization, TtCore, TtDecomposition,
    TtFactors, TuckerFactors,
};

pub mod depth_estimation;
pub use depth_estimation::{
    // §5 DepthCompletion
    DeCompletionFusion,
    DeConv3d,
    // §6 DeConv3d
    DeConv3dConfig,
    DeDepthMetrics,
    // §9 DeDepthMetrics
    DeDepthReport,
    // §1 DepthEncoder
    DeEncoderBlock,
    // §7 DeImplicitNeuralField
    DeImplicitFieldConfig,
    DeImplicitNeuralField,
    DeMultiScaleFeatures,
    DePanopticConfig,
    DePanopticHead,
    // §8 DePanopticHead
    DePanopticResult,
    DeReassembleLayer,
    DeRefineBlock,
    DepthCompletion,
    DepthCompletionConfig,
    DepthEncoder,
    DepthEncoderConfig,
    // §3 MonocularDepthEstimator
    DepthMode,
    DptDecoder,
    // §2 DptDecoder
    DptDecoderConfig,
    MonocularDepthEstimator,
    StereoMatcher,
    // §4 StereoMatcher
    StereoMatcherConfig,
};

pub mod llm_serving;
pub use llm_serving::{
    LsBatch, LsBatchMetrics, LsBlockTableEntry, LsContinuousBatchProcessor, LsCrfTransitions,
    LsDynamicBatcher, LsEntity, LsFormattedInstruction, LsInstructionDataset, LsInstructionExample,
    LsInstructionTuner, LsKvBlock, LsKvCacheManager, LsLayerProfile, LsLinear, LsManagedSequence,
    LsModelWarmup, LsPagedKvCache, LsRequestPriority, LsRewardHead, LsRewardNormalizer,
    LsRlhfRewardModel, LsSequenceState, LsServingMetrics, LsServingReport, LsServingRequest,
    LsSlaCfompliance, LsSpeculativeDecoder, LsSpeculativeResult, LsTaggingScheme,
    LsTokenClassifier, LsTokenTreeNode, LsWarmupReport,
};

pub mod protein_lm;
pub use protein_lm::{
    PlmContactPredictor, PlmEmbedding, PlmEncoder, PlmError, PlmFeedForward, PlmFitnessPredictor,
    PlmLayerNorm, PlmMaskedLMLoss, PlmMetrics, PlmMsaEncoder, PlmMultiHeadAttention, PlmTokenizer,
    PlmTransformerBlock, PLM_CLS, PLM_EOS, PLM_MASK, PLM_PAD,
};

pub mod active_inference;
pub use active_inference::{
    AiActiveInferenceAgent, AiBeliefState, AiError, AiExpectedFreeEnergy, AiGenerativeModel,
    AiHierarchicalGenerativeModel, AiHierarchicalLevel, AiMarkovBlanket, AiMetrics,
    AiParameterLearning, AiPolicySelection, AiVariationalInference,
};

pub mod tensor_networks;
pub use tensor_networks::{
    TnConvolutionalKernel, TnEntanglementMeasures, TnError, TnLowRankRnn, TnMatrixProductState,
    TnMeraLayer, TnMetrics, TnMpsSite, TnNeuralNetworkTN, TnQuantumInspiredLayer, TnTreeNode,
    TnTreeTensorNetwork, TnTuckerLayer,
};

pub mod reward_shaping;
pub use reward_shaping::{
    HerStrategy, RsBradleyTerryModel, RsCuriosityModule, RsEmpowermentIntrinsic, RsEpisode,
    RsError, RsGoalConditionedReward, RsMetrics, RsPotentialBasedShaping,
    RsRandomNetworkDistillation, RsRetroactiveLearning, RsRewardComponent, RsRewardDecomposition,
    RsRewardEnsemble,
};

pub mod evolutionary_computation;
pub use evolutionary_computation::{
    EcActivation, EcConnectionGene, EcDifferentialEvolution, EcError, EcEvolutionStrategies,
    EcGeneticProgramming, EcGenome, EcGpNode, EcGpTree, EcMapElites, EcMapElitesGrid, EcMetrics,
    EcNeat, EcNeatConfig, EcNodeGene, EcNodeType, EcNoveltySearch, EcNsga3, EcParticle,
    EcParticleSwarm, EcSolution, EcSpecies,
};

pub mod neural_compression;
pub use neural_compression::{
    NcArithmeticCoder, NcAutoencoder, NcEntropyModel, NcError, NcHyperprior, NcMetrics,
    NcPatchCompressor, NcProgressiveCoder, NcRateDistortionLoss, NcResidualQuantizer,
    NcVectorQuantizer,
};

pub mod scene_graph;
pub use scene_graph::{
    SgBoundingBox, SgError, SgMessagePassing, SgMetrics, SgObject, SgObjectDetector, SgQuery,
    SgQueryEngine, SgQueryResult, SgRelationship, SgRelationshipDetector, SgSceneComparison,
    SgSceneGraph, SgSceneGraphGeneration, SgSpatialRelation, SgVQA,
};

pub mod causal_rl;
pub use causal_rl::{
    CrlCausalCredit, CrlCausalCurriculum, CrlCausalMechanism, CrlCausalModelBasedRL,
    CrlCausalPolicyGradient, CrlCounterfactualDataAugmentation, CrlCurriculumStrategy,
    CrlEnvironment, CrlError, CrlInvariantPolicyLearning, CrlMetrics, CrlModularModel,
    CrlOffPolicyCounterfactual, CrlReplayBuffer, CrlSCMDynaQ, CrlStructuralCausalModel,
    CrlVariable,
};

pub mod audio_generation;
pub use audio_generation::{
    AgAudioCodec, AgDiffWave, AgDiffWaveLayer, AgDilatedCausalConv1D, AgError, AgGatedActivation,
    AgGriffinLim, AgHifiGanGenerator, AgMetrics, AgMrfBlock, AgNoiseSchedule,
    AgTextToSpeechPipeline, AgWaveNet, AgWaveNetConfig,
};

pub mod object_tracking;
pub use object_tracking::{
    MotAppearanceFeature, MotBoundingBox, MotByteTrack, MotDeepSort, MotDeepSortTrack,
    MotEloRating, MotError, MotHungarian, MotKalmanFilter, MotMetrics, MotSort, MotSortConfig,
    MotTrack, MotTrackState,
};

pub mod nn_verification;
pub use nn_verification::{
    NnvActivation, NnvAdversarialCertifier, NnvCertification, NnvCrownBounds,
    NnvDatasetCertification, NnvError, NnvInterval, NnvIntervalBoundPropagation, NnvLayer,
    NnvLinearBound, NnvMetrics, NnvMonotonicityCheck, NnvNetwork, NnvProperty, NnvPropertyChecker,
    NnvRandomizedSmoothing, NnvResult, NnvZonotope,
};

pub mod self_play;
pub use self_play::{
    sp_rand01, sp_randn, SpAlphaZeroConfig, SpAlphaZeroTrainer, SpEloSystem, SpError, SpGame,
    SpMctsConfig, SpMctsNode, SpMctsTree, SpMetrics, SpMinimax, SpNeuralNetwork, SpSelfPlayBuffer,
    SpSelfPlayExample, SpTicTacToe,
};

pub mod digital_pathology;
pub use digital_pathology::{
    DpAbmil, DpBiomarkerPredictor, DpCoxPh, DpDeepSurv, DpDsmil, DpError, DpMetrics,
    DpMilClassifier, DpPatch, DpPatchSampler, DpPooling, DpSamplingStrategy, DpTissueSegmenter,
    DpWholeSlideImage,
};

pub mod data_augmentation;
pub use data_augmentation::{
    da_rand01, da_randint, da_randn, DaAugment, DaAugmentationPipeline, DaAutoAugment,
    DaBrightness, DaContrast, DaCutmix, DaDiffAugment, DaDiffPolicy, DaError, DaGaussianBlur,
    DaGaussianNoise, DaHorizontalFlip, DaMetrics, DaMixup, DaMixupBatch, DaOperation,
    DaRandAugment, DaRandomCrop, DaRandomErasing, DaRandomRotation, DaSample, DaShear, DaSubpolicy,
    DaTrivialAugment, DaVerticalFlip,
};

pub mod model_merging;
pub use model_merging::{
    MmDare, MmError, MmFisherWeightedMerge, MmLinearInterpolation, MmLoraAdapter,
    MmLoraAggregation, MmMetrics, MmModelWeights, MmRegmean, MmSimpleAverage, MmTaskArithmetic,
    MmTaskVector, MmTiesMerging,
};

pub mod neural_collapse;
pub use neural_collapse::{
    NclDrLoss, NclEquiangularTightFrame, NclError, NclEtfClassifier, NclFeatureStats, NclFewShotNc,
    NclFisherRao, NclLayerAnalysis, NclMetrics, NclNeuralCollapseMetrics, NclPrototypeClassifier,
    NclSupcon,
};

pub mod geospatial_ml;
pub use geospatial_ml::{
    frechet_distance_km,
    haversine_km,
    morans_i,
    // §1 Spatial Feature Encoding
    GeoCoord,
    GeoTokenizer,
    H3GridEncoder,
    QuadkeyEncoder,
    SpatialSinusoidalEncoding,
    // §2 Spatial Graph Networks
    PoiCategory,
    SpatialAttentionLayer,
    SpatialEdge,
    SpatialGcnLayer,
    SpatialGraph,
    UrbanComputingGnn,
    UrbanRegion,
    // §3 Trajectory Analysis
    FrequencyMapEncoding,
    GeoTrajectory,
    MovementPatternDetector,
    MovementState,
    TrajectoryClusterer,
    TrajectoryEncoder,
    TrajectoryPoint,
    // §4 Spatial Interpolation
    IdwInterpolator,
    OrdinaryKriging,
    RadialBasisInterpolator,
    SpatialCrossValidation,
    // §5 Spatial-Temporal Models
    DiffusionConvLayer,
    GeoAttentionModel,
    StGcnLayer as GeoStGcnLayer,
    // §6 GeoMetrics
    GeoMetrics,
    SpatialEvalReport,
    // §A Urban Mobility (advanced)
    TaxiDemandPredictor,
    RideSharingOptimizer,
    RideMatch,
    UrbanFlowEstimator,
    // §B Map Matching (advanced)
    RoadNetwork,
    HmmMapMatcher,
    // §C Spatial Anomaly Detection (advanced)
    SpatialIsolationForest,
    GeofenceAlert,
    // §D GeoMetrics+ (advanced)
    SpatialPredictionInterval,
    trajectory_dtw_km,
    SpatioTemporalMetrics,
};

pub use robotics::advanced::{
    Aabb,
    AdvancedDomainRandomizer,
    CentroidalDynamics,
    ContactModel,
    DexterousGraspPlanner,
    GraspQualityMetric,
    HierarchicalQp,
    NeuralRrt,
    Prm,
    RoboticsPolicyMetrics,
    RrtStarPlanner,
    SimToRealAdapter,
    TactileSensorModel,
    WbcTask,
};
pub use robotics::extensions::{
    AdaptationModule,
    DomainAdaptationLoss,
    DomainRandomizer,
    PhysicsParams,
    SimToRealEvaluator,
};
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
