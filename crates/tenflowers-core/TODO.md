# TenfloweRS Core TODO & Roadmap (0.1.0-alpha.1)

Alpha.1 focus: current tensor engine capabilities and forward development plan. Historical logs removed.

## 1. Current Capabilities

### Tensor Engine Foundation
- **Eager Execution**: Complete tensor operations (creation, arithmetic, reduction, manipulation)
- **Matrix Operations**: Blocked multiplication + outer product specialization + optional BLAS acceleration
- **SIMD Optimization**: 8-element chunking, unchecked fast paths, comprehensive mathematical functions
- **Memory Management**: Reference counting, buffer reuse metrics, allocation tracing capabilities
- **Error Handling**: Zero-warning baseline, consolidated error patterns across operations

### Modular Architecture Success
- **16/21 Large Files Refactored**: Successfully modularized major components
  - ✅ ops/normalization.rs (3,053 lines → modular)
  - ✅ gpu.rs (2,968 lines → specialized GPU modules)
  - ✅ graph.rs (2,953 lines → 7 functional modules)
  - ✅ pretrained.rs (2,881 lines → domain-specific)
  - ✅ ops/manipulation.rs (2,746 lines → operation-type modules)
  - ✅ ops/reduction.rs (2,637 lines → statistical/argops/cumulative/boolean/segment)
  - ✅ neural/layers/conv.rs (2,595 lines → layer-type modules)
  - ✅ neural/layers/embedding.rs (2,376 lines → embedding-type modules)
  - ✅ Additional 8 files from previous phases

### GPU & Performance
- **GPU Support**: Partial WGSL compute kernels with safe CPU fallbacks
- **Cross-Platform**: WebGPU backend for broad hardware compatibility
- **Performance**: Efficient SIMD operations with mathematical function library
- **Optimization**: Gradient clipping, memory pool management

### SciRS2 Integration
- **Complete Migration**: 100% usage of scirs2-autograd::ndarray (with array! macro)
- **Foundation**: Built on scirs2-core for random number generation and scientific primitives
- **Ecosystem**: Fully integrated with SciRS2/NumRS2 scientific computing stack

## 2. Current Gaps & Limitations

### Core Infrastructure
- **Dispatch System**: No unified op/kernel dispatch registry (per-module logic duplication)
- **Shape Inference**: Inconsistent shape inference and diagnostic error messages across ops
- **GPU Coverage**: Many operations still fallback to CPU, partial kernel coverage only
- **Performance**: Some advanced linear algebra operations lack fused kernels

### Graph Execution
- **Optimizer Passes**: Graph execution optimizer passes currently disabled
- **Fusion**: Limited elementwise fusion capabilities
- **JIT**: No just-in-time compilation layer

### Memory & Diagnostics
- **GPU Memory**: Inconsistent GPU memory management, limited allocation tracing
- **Diagnostics**: GPU memory pool diagnostics need enhancement
- **Error Messages**: Shape inference error messages need standardization

## 3. Near-Term Roadmap (Beta Prep)

### Priority 1: Core Infrastructure
1. **Unified Dispatch Registry**: Design and implement op/kernel dispatch registry with CPU/GPU/backend feature gating
2. **Shape Inference Consolidation**: Standardize shape inference + error taxonomy across all operations
3. **GPU Memory Diagnostics**: Implement allocation tracing, memory pool diagnostics, usage reporting
4. **Elementwise Fusion**: Simple chain fusion MVP for performance improvement
5. **Performance Regression Gates**: Integrate criterion-based regression thresholds in CI

### Priority 2: GPU Enhancement
6. **Expanded GPU Kernels**: Increase GPU reduction + broadcast kernel coverage
7. **Memory Pool Optimization**: Enhanced GPU memory pool with diagnostic capabilities
8. **Kernel Coverage**: Systematic expansion of GPU operation coverage

### Priority 3: Advanced Features
9. **ONNX Tensor I/O**: Internal tensor serialization utilities (groundwork for ONNX)
10. **Reference Counting**: Enhanced buffer reuse metrics exposure API
11. **Advanced Error Handling**: Improved error taxonomy with user-facing messages

## 4. Mid-Term Roadmap (Post-Beta)

### Graph Optimization
- **Optimizer Passes**: Enable graph optimizer passes (CSE, constant fold, schedule)
- **JIT Exploration**: Just-in-time compilation and fusion exploration layer
- **Advanced Scheduling**: Operation scheduling optimization

### Performance & Memory
- **Mixed Precision**: Stability improvements + automatic casting policies
- **Advanced Linear Algebra**: Batched factorizations, sparse primitives
- **Memory Offload**: Host <-> device memory spill strategy

### Advanced Features
- **Graph Execution**: Full graph execution mode with optimization
- **Distributed Support**: Multi-GPU orchestration foundations
- **Quantization**: Quantization pipeline infrastructure

## 5. Active TODO Items

### Immediate Tasks
- [ ] **Dispatch Registry Design**: Create design document and prototype
- [ ] **Shape Error Taxonomy**: Draft standardized error message system
- [ ] **GPU Alloc Trace**: Prototype memory allocation tracing system
- [ ] **Fusion Pass**: Implement elementwise chain fusion prototype
- [ ] **Reduction Kernel List**: Define expansion priorities for GPU operations
- [ ] **Tensor Serialization**: Implement internal primitive for ONNX groundwork

### Development Infrastructure
- [ ] **Performance Gate CI**: Integrate criterion-based regression detection
- [ ] **GPU Memory Metrics**: Expose allocation and usage APIs
- [ ] **Error Message Standards**: Implement consistent error formatting
- [ ] **Documentation**: Core concepts and performance guide

## 6. Remaining Large File Refactoring (5/21 files)

Priority files for future modular extraction:
- Complex multi-thousand line files requiring systematic modularization
- Systematic approach established: analyze → modularize → test → backup
- Target: Complete remaining 5 files for 100% modular architecture

## 7. Deferred Items

### Advanced Research
- **Distributed Multi-GPU**: Handled at higher framework layer initially
- **Full Sparse Tensor**: Complete sparse tensor feature set
- **Advanced Quantization**: Full quantization pipeline
- **JIT Compilation**: Advanced just-in-time compilation system

### Infrastructure
- **Memory Spilling**: Complex host/device memory management
- **Graph Optimization**: Advanced compiler-like optimizations
- **Custom Backends**: Pluggable backend system

---

**Alpha.1 Status**: TenfloweRS Core provides a production-ready tensor engine with comprehensive operations, 75% modular refactoring complete, and solid SciRS2 integration. Ready for beta development phase focusing on unified dispatch and GPU enhancement.