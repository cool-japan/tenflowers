# Contributing to TenfloweRS

First off, thank you for considering contributing to TenfloweRS! It's people like you that make TenfloweRS such a great tool for the Rust machine learning community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Project Architecture](#project-architecture)
- [Recognition](#recognition)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

1. **Rust**: Install Rust 1.70 or later from [rustup.rs](https://rustup.rs/)
2. **cargo-nextest**: Install with `cargo install cargo-nextest`
3. **GPU Support (Optional)**: 
   - For WGPU: Ensure you have appropriate graphics drivers
   - For CUDA: Install CUDA toolkit 11.0+
4. **Python (for FFI)**: Python 3.8+ with `maturin` (`pip install maturin`)

### Building the Project

```bash
# Clone the repository
git clone https://github.com/cool-japan/tenflowers
cd tenflowers

# Build all crates
cargo build --workspace

# Run tests (we use nextest)
cargo nextest run --workspace --no-fail-fast

# Check for warnings (MUST pass)
cargo check --workspace
cargo clippy --workspace -- -D warnings
```

### Development Environment Setup

We recommend using:
- **VS Code** with rust-analyzer extension
- **IntelliJ IDEA** with Rust plugin
- **Neovim** with rust.vim

### Understanding the Codebase

Read these files in order:
1. `README.md` - Project overview
2. `TODO.md` - Current development priorities
3. `CLAUDE.md` - Project-specific guidelines
4. Each crate's `README.md` and `TODO.md`

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Environment details** (OS, Rust version, GPU if applicable)
- **Minimal reproducible example**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Include:

- **Use case** - Why is this needed?
- **Proposed solution** - How should it work?
- **Alternatives considered**
- **Additional context** - Examples, mockups, etc.

### Your First Code Contribution

Look for issues labeled:
- `good first issue` - Simple fixes to get started
- `help wanted` - More involved but well-defined tasks
- `todo` - Items from our TODO.md file

### Pull Requests

1. **Check TODO.md** for priority tasks
2. **Open an issue first** to discuss significant changes
3. **Follow the coding standards** below
4. **Write tests** for new functionality
5. **Update documentation** as needed
6. **Ensure CI passes** before requesting review

## Development Process

### 1. Pick a Task

Check `TODO.md` for prioritized tasks. Focus on:
- Phase 1 items if you're implementing core infrastructure
- "Priority 1" items within each crate's TODO.md
- Issues labeled `high-priority`

### 2. Design First

For non-trivial features:
1. Open an issue describing your approach
2. Wait for feedback before implementing
3. Consider TensorFlow's design as a reference
4. Ensure Rust idioms are followed

### 3. Implementation Checklist

- [ ] Follow existing patterns in the codebase
- [ ] Add comprehensive tests
- [ ] Document public APIs
- [ ] No compiler warnings
- [ ] Run `cargo fmt`
- [ ] Run `cargo clippy`
- [ ] Update relevant documentation

### 4. Testing Requirements

- **Unit tests**: Test individual functions
- **Integration tests**: Test feature interactions
- **Gradient tests**: Verify autodiff correctness
- **Device parity**: CPU and GPU produce same results
- **Benchmarks**: For performance-critical code

## Coding Standards

### Rust Style Guide

```rust
// Good: Clear, idiomatic Rust
pub fn add_tensors<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    ensure!(a.shape() == b.shape(), "Shapes must match");
    // Implementation
}

// Bad: Non-idiomatic
pub fn add_tensors<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    if a.shape() != b.shape() { panic!("bad shape"); }
    // Implementation
}
```

### Key Principles

1. **No Warnings Policy**: Code must compile without warnings
2. **Error Handling**: Use `Result<T, Error>` instead of panics
3. **Documentation**: All public items need doc comments
4. **Tests**: Comprehensive test coverage required
5. **Safety**: Minimize `unsafe` code, document when necessary

### Code Organization

```rust
// Standard module organization
use std::collections::HashMap;  // std imports first
use std::sync::Arc;

use ndarray::Array;            // external crates
use num_traits::Float;

use crate::error::{Result, TensorError};  // internal imports
use crate::tensor::Tensor;

// Then: types, traits, implementations, tests
```

### Naming Conventions

- **Types**: `PascalCase` (e.g., `GradientTape`)
- **Functions/Methods**: `snake_case` (e.g., `compute_gradient`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `MAX_DIMS`)
- **Modules**: `snake_case` (e.g., `shape_inference`)

### Error Messages

```rust
// Good: Helpful error with context
bail!("Cannot reshape tensor of size {} to shape {:?}", 
      self.numel(), new_shape);

// Bad: Generic error
bail!("Invalid operation");
```

## Testing Guidelines

### Test Organization

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_operation_correctness() {
        // Arrange
        let input = Tensor::ones(&[2, 3]);
        
        // Act
        let result = input.sum(None, false).unwrap();
        
        // Assert
        assert_eq!(result.scalar(), 6.0);
    }
    
    #[test]
    fn test_gradient_correctness() {
        // Use finite differences to verify gradients
    }
}
```

### Test Categories

1. **Unit Tests**: In `src/` files, test individual functions
2. **Integration Tests**: In `tests/`, test feature interactions
3. **Benchmarks**: In `benches/`, measure performance
4. **Doc Tests**: In doc comments, provide examples

### Running Tests

```bash
# Run all tests
cargo nextest run --workspace --no-fail-fast

# Run specific test
cargo nextest run test_name

# Run with GPU features
cargo nextest run --features gpu

# Run benchmarks
cargo bench

# Check doc tests
cargo test --doc
```

## Documentation

### API Documentation

```rust
/// Computes the matrix multiplication of two tensors.
/// 
/// # Arguments
/// 
/// * `other` - The right-hand side tensor
/// 
/// # Returns
/// 
/// The resulting tensor with shape determined by matrix multiplication rules.
/// 
/// # Errors
/// 
/// Returns `TensorError::ShapeMismatch` if the tensors have incompatible shapes.
/// 
/// # Examples
/// 
/// ```
/// # use tenflowers_core::{Tensor, Device};
/// let a = Tensor::<f32>::ones(&[2, 3]);
/// let b = Tensor::<f32>::ones(&[3, 4]);
/// let c = a.matmul(&b)?;
/// assert_eq!(c.shape().dims(), &[2, 4]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
    // Implementation
}
```

### Guidelines

1. **Document all public items**
2. **Include examples in doc comments**
3. **Explain error conditions**
4. **Use standard sections**: Arguments, Returns, Errors, Examples, Panics
5. **Keep descriptions concise but complete**

## Submitting Changes

### Before Submitting

1. **Rebase on main**: `git rebase origin/main`
2. **Run full test suite**: `cargo nextest run --workspace`
3. **Check formatting**: `cargo fmt --all -- --check`
4. **Run clippy**: `cargo clippy --workspace -- -D warnings`
5. **Update documentation**: Including CHANGELOG.md if needed
6. **Squash commits**: Into logical units

### Pull Request Process

1. **Create PR with clear title** following conventional commits:
   - `feat: Add Conv2D layer implementation`
   - `fix: Correct gradient computation for ReLU`
   - `docs: Update neural network examples`
   - `perf: Optimize matrix multiplication`

2. **Fill out PR template** including:
   - What changes were made
   - Why they were necessary
   - How they were tested
   - Breaking changes (if any)

3. **Address review feedback** promptly

4. **Ensure CI passes** before merge

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Example:
```
feat(autograd): Add support for higher-order derivatives

Implement forward-mode automatic differentiation alongside the existing
reverse-mode to enable efficient computation of Jacobian-vector products
and higher-order derivatives.

Closes #123
```

## Project Architecture

### Crate Dependencies

```
tenflowers-core
├── tenflowers-autograd (depends on core)
├── tenflowers-neural (depends on core, autograd)
├── tenflowers-dataset (depends on core)
└── tenflowers-ffi (depends on all above)
```

### Key Design Patterns

1. **Trait-Based Abstraction**: Operations, layers, optimizers use traits
2. **Builder Pattern**: For complex object construction
3. **Type Safety**: Leverage Rust's type system for correctness
4. **Zero-Cost Abstractions**: Performance without runtime overhead

### Adding New Operations

1. Define op in `tenflowers-core/src/ops/`
2. Implement CPU kernel
3. Add GPU kernel if applicable
4. Register in operation registry
5. Add shape inference function
6. Implement gradient in `tenflowers-autograd`
7. Write comprehensive tests
8. Document with examples

### Performance Considerations

- Minimize allocations in hot paths
- Use SIMD where beneficial
- Implement specialized kernels for common cases
- Profile before optimizing
- Benchmark against TensorFlow/PyTorch

## Recognition

### Contributors

All contributors will be recognized in:
- The project README
- Release notes
- Annual contributor spotlight

### Code Ownership

- Major feature authors listed in module docs
- Significant contributors become codeowners
- Active maintainers join core team

## Getting Help

- **GitHub Discussions**: For design discussions
- **Issue Tracker**: For bugs and features
- **Email**: info@kitasan.io (Team KitaSan)

## Additional Resources

- [TensorFlow Design Docs](https://github.com/tensorflow/community/tree/master/rfcs) - For design inspiration
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) - For Rust best practices
- [NumPy API Reference](https://numpy.org/doc/stable/reference/) - For operation semantics

Thank you for contributing to TenfloweRS! Your efforts help build a safer, faster future for machine learning in Rust.