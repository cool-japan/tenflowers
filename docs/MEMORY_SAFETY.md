# Memory Safety Policy for TenfloweRS

## Overview

TenfloweRS is a pure-Rust framework and relies on Rust's ownership system for
its primary memory-safety guarantees. This document describes supplementary
dynamic analysis tooling (Miri, AddressSanitizer, LeakSanitizer) and the
policy for when each tool runs.

---

## 1. Static Safety Baseline

All production code in every TenfloweRS crate carries `#![deny(unsafe_code)]`
(where technically feasible) or limits unsafe to clearly reviewed, minimal
blocks. The full build must compile with zero warnings before any release.

---

## 2. Miri — Undefined-Behaviour Interpreter

[Miri](https://github.com/rust-lang/miri) runs the Rust MIR through an
interpreter that detects undefined behaviour (UB), invalid memory accesses,
and data races that Rust's type system cannot prevent at compile time.

### Scope

Miri is run on a **curated, C-free subset** of the workspace:

| Crate | Target | Features |
|-------|--------|----------|
| `tenflowers-core` | `--lib --tests` | `std` only (no blas, no gpu) |
| `tenflowers-autograd` | `--lib --tests` | `default` (no gpu, no parallel) |

The following are **excluded** from Miri runs:

| Exclusion | Reason |
|-----------|--------|
| `blas-*` features | Depends on C/Fortran BLAS libraries outside Miri's model |
| `gpu / cuda / metal / rocm` | GPU drivers are C FFI, not Miri-modelable |
| `tenflowers-ffi` | PyO3 is built on the Python C API |
| Tests calling into `scirs2-core` C glue | Same as gpu/blas |

### Running Miri Locally

```bash
# Install nightly + miri (once)
rustup toolchain install nightly
rustup component add miri --toolchain nightly
cargo +nightly miri setup

# Run the curated subset
bash scripts/run_miri.sh
```

### Known Issues

No UB has been detected in the curated subset to date. If Miri reports an
error that originates from a transitive dependency outside TenfloweRS control
(e.g. a scirs2-core internal), document it here with a link to the upstream
issue rather than silencing it.

### Cadence

Miri runs are expected **weekly** on the `master` / release branches during
active development. CI execution is tracked in issue backlog item
`ci-wheel-workflow` (blocked on CI infrastructure restoration).

---

## 3. AddressSanitizer (ASan)

ASan detects:
- Out-of-bounds heap/stack accesses
- Use-after-free
- Use-after-return
- Double-free

### Planned Usage

Once the GitHub Actions workflow files are restored from `.disabled` status,
ASan will run on:

```bash
RUSTFLAGS="-Z sanitizer=address" \
cargo +nightly test \
    -p tenflowers-core \
    -p tenflowers-autograd \
    --target x86_64-unknown-linux-gnu \
    -- --skip blas --skip gpu
```

ASan is a nightly-only feature and requires Linux (ASAN is not supported on
macOS for Rust without special configuration).

---

## 4. LeakSanitizer (LSan)

LSan detects memory leaks at process exit. On Linux, LSan is bundled with
ASan. On macOS, `--features lsan` is required in some toolchains.

LSan will run as part of the same CI job as ASan once the workflow is enabled.

---

## 5. Valgrind / Helgrind

Valgrind and Helgrind are not currently used for this project. They are slower
than ASan/TSan and do not understand Rust's memory model well. TSan
(ThreadSanitizer) is under evaluation for the `parallel` feature crate subset.

---

## 6. Audit Scope for Unsafe Code

All uses of `unsafe` must:
1. Be in a clearly labelled block with a `// SAFETY:` comment explaining the
   invariant that makes it sound.
2. Be reviewed in every PR that touches the containing file.
3. Not use `unwrap()` inside unsafe blocks (per CLAUDE.md No-Unwrap Policy).

Current production `unsafe` count: **0** (all `unsafe_code` denied via lint).

---

## 7. Related Documents

- [`scripts/run_miri.sh`](../scripts/run_miri.sh): Miri runner script
- [`docs/RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md): Pre-release verification steps
- [Miri Documentation](https://github.com/rust-lang/miri)
- [Rustonomicon — Unsafe Rust](https://doc.rust-lang.org/nomicon/)
