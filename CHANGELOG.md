# Changelog

All notable changes to TenfloweRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-04-24

### Added

#### Meta-Crate Ergonomics (`tenflowers`)
- `tensor![]` declarative macro for shape-inferred tensor creation (1-D, 2-D, nested, explicit dtype)
- Type aliases: `Tensor1D<T>`, `Tensor2D<T>`, `Tensor3D<T>`, `Tensor4D<T>`, `Vector`, `Matrix`, `BatchTensor`
- Expanded prelude: `MultiHeadAttention`, `RMSNorm`, `TransformerEncoder/Decoder`, `GRU`, `LSTM`, `RNN`, `Optimizer`, `RandomSampler`, `DType`
- `tenflowers::interop::ndarray` — `from_ndarray` / `to_ndarray` conversion utilities
- `tenflowers::io` — `save_tensor` / `load_tensor` convenience wrappers
- `tenflowers::onnx` — ONNX re-exports under `onnx` feature gate
- `deprecated_use!` macro for structured deprecation notices
- Feature presets: `experimental`, `minimal`, `standard`

#### Documentation
- Getting Started tutorial and PyTorch↔TenfloweRS API mapping table (35 rows) in README
- Mermaid architecture diagram (crate dependency graph with feature-gate annotations)
- `docs/QUICK_REFERENCE.md` — 10-section cheat sheet
- `tenflowers/docs/MIGRATION_FROM_TENSORFLOW.md` — 5 side-by-side TF→Rust scenarios
- `tenflowers/docs/TROUBLESHOOTING.md` — 10 symptom-fix triples
- `tenflowers/docs/PRELUDE_STABILITY.md` — semver stability policy for prelude

#### Release Tooling
- `scripts/publish_meta.sh` — dry-run publish ordering script
- `scripts/bump_version.sh` — workspace version sync script
- `docs/RELEASE_CHECKLIST.md` — pre-release validation checklist
- `scripts/run_miri.sh` + `docs/MEMORY_SAFETY.md` — Miri testing policy

#### Dataset Crate (`tenflowers-dataset`)
- `PidAdaptiveController` — PID-controlled prefetch depth driven by cache hit-rate signal
- Criterion throughput benchmark harness (`benches/throughput.rs`)
- Drift metrics: PSI, KS two-sample statistic, Jensen-Shannon divergence, `DriftReport`
- `PipelineInspector` with per-step latency, shape-in/out, and `PipelineInspectionReport`
- `SchemaValidator::validate_full` with per-field `FieldDiff` (TypeMismatch, Widening, MissingRequired, UnexpectedExtra)
- Expanded module docs and API stabilization

#### FFI Crate (`tenflowers-ffi`)
- Structured `TensorError → TenflowersError` variant mapping (all 23 core variants, exhaustive match)
- `into_py_err()` — maps `TenflowersError` to typed Python exceptions (`ValueError`, `RuntimeError`, `IndexError`, `MemoryError`, `NotImplementedError`)
- `PyDevice` / `PyDeviceKind` — GPU/ROCm/CPU device as Python classes with `Device.cpu()`, `Device.gpu(id)`, `Device.rocm(id)`
- `PyTensor.__repr__` fix: dtype now reflects actual tensor dtype (was hardcoded `float32`)
- `PyTensor.__len__`, `.ndim`, `.numel()` properties
- `build.rs` for opt-in cbindgen header regeneration (`TENFLOWERS_REGENERATE_C_HEADER=1`)
- `tests/conftest.py` — shared pytest fixtures and marker registration
- `docs/FFI_ERROR_MAPPING.md` — full error mapping reference table

### Fixed

- **wgpu v29 API compatibility** — 57 sites updated across `tenflowers-core` and `tenflowers-dataset`:
  - `InstanceDescriptor::default()` replaced with `InstanceDescriptor::new_without_display_handle()`
  - `bind_group_layouts: &[&layout]` updated to `&[Some(&layout)]` per wgpu 29 API
- `data_quality.rs` — 9 `&str`/`String` type mismatches in drift metric constructors

### Improved

- Autograd module docs expanded to ~180 lines: mixed-precision, checkpointing, higher-order grads, custom ops
- Neural module docs finalized (removed "IN PROGRESS" markers)
- 3 new autograd examples: `mixed_precision.rs`, `gradient_checkpointing.rs`, `higher_order_grads.rs`
- FFI package metadata: classifiers, project URLs, optional dependency groups in `pyproject.toml`

## [0.1.0] - 2026-03-20

### Summary

First release of TenfloweRS — a research-grade, pure-Rust machine learning framework built on the SciRS2 ecosystem.

**Release Status:** Production-ready (6 crates)
- **Tests:** 12,949 passing across the workspace (0 failures, 0 warnings)
- **Code:** ~790K lines of Rust across 1,446+ source files
- **Security:** 0 vulnerabilities
- **Quality:** Zero clippy warnings, full formatting compliance, no `unwrap()` usage

### Added

#### Workspace Crates
- **tenflowers-core**: Core tensor operations, GPU abstraction, operation registry, shape inference, kernel fusion, autocast, sparse tensors, fused ops
- **tenflowers-autograd**: Reverse-mode automatic differentiation, gradient accumulation, checkpointing, in-place ops, forward-mode gradients, Jacobian checks, interpretability utilities
- **tenflowers-dataset**: Data loading and preprocessing, distributed streaming, dataset core, cache telemetry
- **tenflowers-neural**: Comprehensive neural network layers, training utilities, and research-grade algorithm modules (see below)
- **tenflowers-ffi**: C FFI and Python bindings via PyO3 (`publish = false` — requires Python environment)
- **tenflowers**: Unified API and prelude, user-facing macros and re-exports

#### Neural Network Modules (tenflowers-neural)

- **Attention mechanisms**: Flash Attention, ALiBi, RoPE, transformer decoder, TCN
- **Optimizers**: LAMB, Lion, Muon, learning-rate schedulers, LR finder
- **Generative models**: Diffusion (DDPM/advanced), VAE, normalizing flows, continuous normalizing flows, flow matching
- **Graph neural networks**: GNN advanced layers, graph-level pooling, temporal GNN, graph signal processing, graph matching, graph transformer, graph foundation models, graph generation, GraphODE, molecular GNN
- **Geometric deep learning**: EGNN, SE(3)-Transformer, VNN, IPA (AlphaFold2-style)
- **Quantum ML**: QAOA, quantum kernels (IQP/ZZ feature maps), zero-noise extrapolation, probabilistic error cancellation, measurement error mitigation, quantum Boltzmann machines
- **Federated learning**: Byzantine-robust aggregation (Krum, FLAME, Bulyan), personalized FL (pFedMe, APFL, FedBN), FedMA, clustered FL (IFCA)
- **Operator learning**: FNO, WNO, GNO, PINO, UNO
- **Bio ML**: scVAE (ZINB), Leiden clustering, DNA conv nets, survival analysis (CoxPH, DeepSurv, Kaplan-Meier), pathway enrichment, multi-omics factor analysis
- **AutoML**: Dataset meta-features, landmarking, algorithm selection, SMAC optimizer, portfolio selection, efficient NAS predictor
- **Molecular GNN**: DimeNet, AttentiveFP, MolBERT, JunctionTreeVAE, GraphVAE, reaction yield prediction
- **Audio models**: HuBERT, Data2Vec-Audio, SoundStream codec, RVQ, beat tracking, chord recognition, FastSpeech2, HiFi-GAN vocoder
- **Sparse learning**: BigBird, sparse sliding window attention, Group Lasso, N:M structured pruning, LISTA, predictive coding, basis pursuit
- **Geospatial ML**: H3/Quadkey grid encoders, spatial GCN, spatial attention, kriging, ST-GCN, diffusion convolution, Moran's I
- **Neural SDE**: VP/VE SDE, score matching, rough paths, NeuralRDE
- **Simulation-based inference**: Flow-SBI, ABC-SMC, NRE
- **Structured prediction**: Neural CRF, span parsing, biaffine dependency, SRL
- **Efficient transformers**: RetNet, Mamba-2, GQA, KV-cache management
- **Neural rendering**: 3D Gaussian splatting, NRC, ReSTIR, DeformNeRF
- **Riemannian geometry**: Poincaré ball, Ollivier-Ricci, Ricci flow
- **World models**: TD-MPC2, GWM tokenizer, GPT imagination loop
- **Symbolic math**: ATP tactics, neural tactic selector, equation database
- **Knowledge graph**: Temporal KG (TeRo/TntComplEx), hyper-relational KG, KG+LLM, rule induction; advanced knowledge distillation
- **Robotics**: RRT*/NeuralRRT/PRM, Ferrari-Canny grasp, whole-body control, DANN sim2real
- **Video understanding**: VideoSwin-V2, TimeSformer, VideoMAE, VOS memory, ConsistencyModel
- **Compression**: Hyperprior model, RD optimizer, movement pruning, mixed-precision search
- **Online learning**: LinUCB, Thompson sampling, ADWIN, LODA, OGD/FTRL
- **Generation**: Speculative decoding, RegexFSM/CFG constrained generation, RAG, BLEU/ROUGE metrics
- **Multimodal foundation**: UnifiedIO, PaLI, visual grounding, symbolic visual reasoning
- **Optimal transport**: Unbalanced/partial OT, JDOT, online sliced-Wasserstein, tree-Wasserstein
- **Additional modules**: active_inference, active_learning, adaptive_computation, adversarial, anomaly_detection, architecture_distillation, audio_generation, bayesian, bayesian_dl, bayesian_opt, bio_ml, causal_discovery_advanced, causal_discovery_ts, causal_inference, causal_representation, causal_rl, causal_ts, checkpoint_advanced, climate_ml, concept_learning, conformal_prediction, continual_learning, contrastive, cooperative_game_theory, cross_modal_retrieval, curriculum_learning, data_augmentation, depth_estimation, differentiable_physics, digital_pathology, distillation, document_understanding, domain_adaptation, drug_discovery, edge_optimization, embodied_ai, emotion_recognition, energy_models, ensemble, evolutionary_computation, financial_ml, functional_data_analysis, hierarchical_time_series, hparam, hyperdimensional, hypernetworks, hyperparameter_optimization, image_generation_advanced, implicit_neural_repr, influence_functions, information_theory, inverse_rl, knowledge_distillation_advanced, kolmogorov_arnold, learning_to_learn, lifelong_learning, llm_serving, lm_evaluation, lora_adapters, marl, materials_ml, mean_field_games, mechanistic_interpretability, medical_imaging, memory_networks, meta_learning, mixture_density_networks, mixture_of_depths, mixture_of_experts_advanced, mixture_of_modalities, model_merging, monte_carlo, multi_fidelity, multi_objective, multi_task, multimodal, music_generation, nas, network_science, neural_collapse, neural_combinatorial, neural_compression, neural_ode, neural_process, neuro_symbolic, neuromorphic, nlp_components, nn_verification, object_tracking, optimal_control, pinn, point_processes, pomdp_planning, privacy_ml, probabilistic, probabilistic_circuits, protein_lm, protein_structure, quantum_ml, recommendation_systems, reward_learning, reward_shaping, rl, safe_rl, safety_alignment, satellite_ml, scene_graph, self_play, self_supervised, signal, simulation_ml, sparse_mixture_experts, spectral, speech_recognition, ssm, state_space_models, statistical_testing, synthetic_data, tabular_learning, tensor_networks, test_time_adaptation, test_time_compute, text_generation_pipelines, time_series, tokenizer, topological_ml, training_dynamics, trajectory_prediction, uncertainty_quantification, variational_inference, vision_transformer, zero_shot_learning, and more

### Crates Published

| Crate | Description |
|-------|-------------|
| tenflowers-core | Core tensor operations, GPU abstraction, autocast, sparse, fused ops |
| tenflowers-autograd | Automatic differentiation, checkpointing, gradient accumulation |
| tenflowers-dataset | Data loading, preprocessing, distributed streaming |
| tenflowers-neural | Research-grade neural network layers and ML algorithms (300+ modules) |
| tenflowers | Unified public API and prelude |

**Not published:** tenflowers-ffi (`publish = false` — requires Python development environment)

### Notes

- **Tensorboard integration** is excluded from this release due to a known upstream vulnerability (RUSTSEC-2024-0437 in protobuf 2.x). It will be restored once the upstream fix is available.
- **SciRS2 dependencies**: All scirs2-* crates at 0.3.0+

### Installation

```toml
[dependencies]
tenflowers = "0.1.0"

# Optional features
tenflowers = { version = "0.1.0", features = ["gpu", "simd"] }
```

### Contributors

Developed by COOLJAPAN OU (Team KitaSan).
Contact: contact@cooljapan.tech
