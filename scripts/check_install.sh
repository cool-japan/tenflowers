#!/usr/bin/env bash
# Check that tenflowers-ffi Python wheel installs correctly.
# Requires maturin and Python 3.8+ in PATH.
#
# Usage: ./scripts/check_install.sh [--release]
#        PYTHON=python3.11 ./scripts/check_install.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FFI_DIR="${WORKSPACE_ROOT}/crates/tenflowers-ffi"

PYTHON="${PYTHON:-python3}"
BUILD_MODE="${1:-}"

echo "=== TenfloweRS Installation Check ==="
echo "Python:    $("${PYTHON}" --version)"
echo "Workspace: ${WORKSPACE_ROOT}"
echo ""

# Step 1: Check that Rust workspace compiles
echo "[1/4] Checking Rust workspace..."
cargo check --workspace --quiet 2>&1 | tail -5
echo "  ✓ Rust workspace compiles"

# Step 2: Build the wheel
TMPDIR=$(mktemp -d)
trap 'rm -rf "${TMPDIR}"' EXIT

echo "[2/4] Building Python wheel..."
MATURIN_ARGS="--out ${TMPDIR}"
if [[ "${BUILD_MODE}" == "--release" ]]; then
    MATURIN_ARGS="${MATURIN_ARGS} --release"
fi

(cd "${FFI_DIR}" && maturin build ${MATURIN_ARGS} 2>&1 | tail -10)
WHEEL=$(ls "${TMPDIR}"/*.whl 2>/dev/null | head -1)
if [[ -z "${WHEEL}" ]]; then
    echo "  ERROR: No wheel produced" >&2
    exit 1
fi
echo "  ✓ Wheel built: $(basename "${WHEEL}")"

# Step 3: Install wheel in a venv
echo "[3/4] Installing wheel..."
VENV="${TMPDIR}/venv"
"${PYTHON}" -m venv "${VENV}"
"${VENV}/bin/pip" install --quiet "${WHEEL}"
echo "  ✓ Wheel installed"

# Step 4: Smoke test
echo "[4/4] Running smoke test..."
"${VENV}/bin/python" - <<'EOF'
import tenflowers
t = tenflowers.Tensor([1.0, 2.0, 3.0])
assert t.numel() == 3, f"Expected numel=3, got {t.numel()}"
print(f"  ✓ Tensor created: numel={t.numel()}")
EOF

echo ""
echo "=== All checks passed ==="
