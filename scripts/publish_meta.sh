#!/usr/bin/env bash
# TenfloweRS — publish all crates in dependency order.
# Defaults to --dry-run. Set TENFLOWERS_PUBLISH_CONFIRM=1 to actually publish.
# Usage: ./scripts/publish_meta.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Safety guard: branch must look like a release branch (0.x.y) ──────────
BRANCH=$(git -C "${WORKSPACE_ROOT}" rev-parse --abbrev-ref HEAD)
if ! [[ "${BRANCH}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: branch '${BRANCH}' is not a release branch (expected 0.x.y format)." >&2
    echo "  Checkout the release branch before publishing." >&2
    exit 1
fi

# ── Dry-run by default ────────────────────────────────────────────────────
DRY_RUN_FLAG="--dry-run"
if [[ "${TENFLOWERS_PUBLISH_CONFIRM:-0}" == "1" ]]; then
    DRY_RUN_FLAG=""
    echo "WARNING: PUBLISH CONFIRM set — will publish for real."
fi

PUBLISH_ORDER=(
    "crates/tenflowers-core"
    "crates/tenflowers-autograd"
    "crates/tenflowers-dataset"
    "crates/tenflowers-neural"
    "crates/tenflowers-ffi"
    "tenflowers"
)

echo "Publishing TenfloweRS crates on branch '${BRANCH}'..."
for crate_path in "${PUBLISH_ORDER[@]}"; do
    echo ""
    echo "── ${crate_path} ─────────────────────────────────────"
    (cd "${WORKSPACE_ROOT}/${crate_path}" && cargo publish ${DRY_RUN_FLAG} --allow-dirty)
done

echo ""
echo "Done. Run with TENFLOWERS_PUBLISH_CONFIRM=1 to actually publish."
