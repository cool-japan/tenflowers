# TenfloweRS TODO & Roadmap (v0.1.1 · 2026-04-24)

## Current Capabilities

### Project Status
- **Tests**: 13,484 passing, 41 skipped across all crates
- **Code Size**: ~643K SLoC Rust code (~772K total Rust lines, 1,465 files)
- **Warnings**: 0 compilation warnings, 0 clippy warnings
- **Vulnerabilities**: 0 known vulnerabilities
- **SciRS2 Integration**: Full migration to SciRS2 ecosystem

### TenfloweRS-Core (Tensor Engine)
- Complete eager tensor engine with arithmetic, reduction, manipulation, matrix multiplication
- Blocked matrix multiplication + outer product specialization + optional BLAS acceleration
- Comprehensive SIMD optimization (8-element chunking, unchecked fast paths, math functions)
- GPU support via WGSL compute kernels with safe CPU fallbacks (cross-platform WebGPU)
- Memory management with reference counting, buffer reuse metrics, allocation tracing

### TenfloweRS-Autograd (Automatic Differentiation)
- Full reverse-mode gradient tape with recording + backward traversal
- Optimized gradient computation (hashmap lookup reductions, allocation optimization)
- Integrated memory profiling on gradient path with usage tracking
- Basic GPU gradient support for selected core operations
- Experimental mixed precision gradient path with loss scaling

### TenfloweRS-Neural (Neural Networks)
- Core layers: Dense, Conv1D/2D, Embedding (basic/positional/sparse), Residual
- Normalization: BatchNorm, LayerNorm, GroupNorm, SyncBatchNorm (training/inference)
- Advanced layers: Mamba/SSM blocks, attention scaffolding, transformer building blocks
- Extended activations: ReLU, GELU, Swish, LeakyReLU, ELU, SELU, Hardswish, GLU variants
- Optimizers: SGD, Momentum, Adam, AdamW, RMSProp, AdaBelief with AMSGrad support
- Training pipeline with gradient accumulation, metric tracking, hook system
- Pretrained model architectures: ResNet, EfficientNet, ViT, BERT, GPT families

### TenfloweRS-Dataset (Data Loading)
- Dataset trait + composable transform pipeline with builder pattern
- SIMD transforms (stats, color conversion, histogram) with runtime fallback
- GPU-accelerated transforms (crop, rotate, jitter, blur, noise, resize, flip)
- Predictive smart cache with pattern-based prefetch and memory pool management
- Formats: JSON/JSONL, Text, Parquet, HDF5, Audio, TFRecord, WebDataset, Zarr, CSV, Image
- Memory-mapped file dataset for large file zero-copy access

### TenfloweRS-FFI (Language Bindings)
- Python bindings via PyO3 (tensors, gradient tape, Dense/Sequential, hooks)
- NumPy tensor conversion (f32), memory optimization utilities
- C API scaffolding (types, tensor creation)
- Hook system (forward/backward), benchmarking, visualization

## Known Limitations (v0.1.1)

- Graph optimizer passes not yet enabled (CSE, constant fold, scheduling)
- Many GPU operations still fall back to CPU; GPU memory management inconsistent
- Higher-order gradients unreliable for composite activation chains
- Advanced LR schedulers (cosine, one-cycle, warmup) absent
- No published Python wheels or packaging pipeline

## Roadmap

### v0.2.0 — Attention & Training Polish
- Multi-head + scaled dot-product attention implementation
- Learning rate schedulers: step, cosine, warmup, one-cycle
- Gradient clipping utilities and anomaly detection hooks
- Unified dispatch registry (CPU/GPU) with backend feature gating
- Consolidated shape inference + standardized error taxonomy
- GPU memory diagnostics: allocation tracing, pool diagnostics, usage reporting
- Elementwise fusion MVP for performance improvement
- Activation checkpointing API for memory-efficient training
- Deterministic mode: global seed + op-local seeds for reproducible training
- Mixed precision policy refinement + dynamic loss scaling
- Streaming data loaders with deterministic sharding for distributed training
- Python wheel builds (manylinux, macOS universal2, Windows)

### v0.3.0 — Scale & Distributed
- Multi-GPU data-parallel execution with optimizer state sync
- Graph optimizer passes (fusion, CSE, scheduling) enabled
- ONNX export/import for core operator subset
- Sequence parallel / model parallel experiments
- Parameter grouping & weight decay configurability
- Pretrained model export/import (JSON weights + binary format)
- Arrow zero-copy integration for dataset pipelines
- Stable ABI & semantic versioning for FFI

### v1.0.0 — Production & Advanced Features
- Model/pipeline parallel & gradient compression
- Auto kernel fusion + JIT compilation prototype
- Advanced linear algebra (batched factorizations, sparse primitives)
- Memory offload / spill strategy (host ↔ device)
- INT8/INT4 quantization toolchain
- Plugin system for external operators
- Comprehensive documentation (Getting Started, Performance, Safety, Migration)

## Future Directions

- Federated learning and distributed sharding beyond standard data-parallel
- Advanced sparsity & compression research implementations
- Additional language bindings (C++, Swift)
- Cloud-native deployment & scaling infrastructure
- Federated data loaders with privacy preservation

---

Copyright 2025-2026 COOLJAPAN OU (Team KitaSan) · contact@cooljapan.tech
