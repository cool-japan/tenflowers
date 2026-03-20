# TenfloweRS Autograd API Stability

**Version:** 0.1.0  
**Date:** 2026-03-20  
**Status:** Stable

---

## Overview

This document defines the API stability guarantees for the TenfloweRS Autograd crate as of v0.1.0.

## Stable Core APIs

### GradientTape

```rust
pub struct GradientTape { /* ... */ }

impl GradientTape {
    pub fn new() -> Self;
    pub fn watch<T>(&self, tensor: Tensor<T>) -> TrackedTensor<T>;
    pub fn gradient<T>(self, targets: &[TrackedTensor<T>], sources: &[TrackedTensor<T>]) -> Result<Vec<Tensor<T>>>;
}
```

### TrackedTensor

All standard tensor operations are available on tracked tensors with automatic gradient tracking.

### gradient_ops

Reverse-mode automatic differentiation for all supported operations including arithmetic, reductions, activations, matrix operations, and FFT.

### Additional Stable APIs

- **Second-Order Derivatives**: Hessian, Jacobian, Laplacian, Hessian-vect- **Second-Order Derivatives**: Hessian, Jacobian, Laplacian, Hessian-vect- **Second-Order Derivatiio- **Second-Order Derivatives**: Het t- **Second-Ordenfigurable strategies
- **Num- **Num- **Num- *ali- tio- **Num- *e d- **Num- **Num- **Num- *ali- tio- **Num- *e d- **Num- **Num- **Num- *ali- tio- **Num- *e d- **Numgem- **Num- **Num- **le - **Num- **Num-emo- **Nufiling**: Real-time gradient memory track- **Num- **Num- **ti- **Num- **Num- **Num- *ee- **Num- **Num- **Num- *ali- tio- *public APIs follow SemVer from v0.1.0 onward
2. **No Silent Breakage**: Any breaking change will increment the minor version (pre-1.0)
3. **Deprecatio3.Policy**: Deprecated A3. **Deprecatio3.Policy**: Deprecated A3. **Deprecatio3.Policy**: Deprecated A3. **Deprecatio3.Pype3. **Deprecatioare part of the stable API

---

Copyright 2025-2026 COOLJAPAN OU (Team KitaSan)
