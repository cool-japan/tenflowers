#!/usr/bin/env bash
# run_miri.sh — Run Miri (the Rust interpreter) on memory-safety-critical
# subsets of the TenfloweRS workspace.
#
# Usage:
#   bash scripts/run_miri.sh
#
# Prerequisites:
#   rustup component add miri --toolchain nightly
#   cargo +nightly miri setup
#
# What this runs:
#   - tenflowers-core (lib + test targets), skipping BLAS/GPU paths that
#     require native extensions Miri cannot model.
#
# What is excluded (and why):
#   - blas / blas-openblas / blas-mkl / blas-accelerate features: depend on
#     libBLAS, which is C-native code outside Miri's model.
#   - gpu / cuda / metal / rocm / opencl features: GPU kernels require native
#     drivers and FFI that Miri does not support.
#   - tenflowers-ffi (PyO3): Python C API calls are C FFI, excluded by design.
#   - scirs2-core FFI glue: any test calling into C is skipped via --skip.
#
# Policy (see docs/MEMORY_SAFETY.md for full policy):
#   Miri runs are expected weekly on the curated subset below. AddressSanitizer
#   (asan) and LeakSanitizer (lsan) will be added to CI when the workflow
#   files are restored (currently .disabled).

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== TenfloweRS Miri safety check ==="
echo "Workspace: ${WORKSPACE_ROOT}"
echo "Toolchain: nightly + miri"
echo ""

cd "${WORKSPACE_ROOT}"

# -----------------------------------------------------------------------
# Ensure the nightly toolchain + miri component are present.
# -----------------------------------------------------------------------
if ! command -v rustup &>/dev/null; then
    echo "ERROR: rustup is not installed — required to run nightly + miri."
    exit 1
fi

if ! rustup toolchain list | grep -q "nightly"; then
    echo "ERROR: nightly toolchain not found.  Run:"
    echo "  rustup toolchain install nightly"
    echo "  rustup component add miri --toolchain nightly"
    exit 1
fi

# -----------------------------------------------------------------------
# Core crate — pure-Rust subset.
# -----------------------------------------------------------------------
echo "--- tenflowers-core (lib + tests, no blas/gpu) ---"
MIRIFLAGS="-Zmiri-disable-isolation" \
cargo +nightly miri test \
    -p tenflowers-core \
    --lib \
    --tests \
    --no-default-features \
    --features "std" \
    -- \
    --skip blas \
    --skip gpu \
    --skip cuda \
    --skip metal \
    --skip rocm \
    2>&1

# -----------------------------------------------------------------------
# Autograd crate — pure-Rust subset.
# -----------------------------------------------------------------------
echo ""
echo "--- tenflowers-autograd (lib + tests, no gpu/parallel) ---"
MIRIFLAGS="-Zmiri-disable-isolation" \
cargo +nightly miri test \
    -p tenflowers-autograd \
    --lib \
    --tests \
    --no-default-features \
    -- \
    --skip gpu \
    --skip cuda \
    --skip metal \
    2>&1

echo ""
echo "=== Miri check complete ==="
