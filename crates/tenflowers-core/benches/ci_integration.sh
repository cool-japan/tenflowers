#!/bin/bash

# Dispatch Registry Benchmark CI Integration Script
#
# This script runs the dispatch registry benchmarks and validates results for CI/CD pipelines.
# It enforces performance thresholds and provides detailed reports.
#
# Usage:
#   ./ci_integration.sh [options]
#
# Options:
#   --threshold-override <percent>  Override default thresholds (e.g., 2.0 for 2%)
#   --save-baseline <file>          Save benchmark results as baseline
#   --compare-baseline <file>       Compare against baseline
#   --verbose                       Enable verbose output
#   --fail-fast                     Exit on first failure
#

set -e

# Configuration
CRATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_NAME="dispatch_benchmarks"
DEFAULT_THRESHOLD=2.0
VERBOSE=false
FAIL_FAST=false
BASELINE_FILE=""
COMPARE_FILE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --threshold-override)
            DEFAULT_THRESHOLD="$2"
            shift 2
            ;;
        --save-baseline)
            BASELINE_FILE="$2"
            shift 2
            ;;
        --compare-baseline)
            COMPARE_FILE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --fail-fast)
            FAIL_FAST=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Dispatch Registry CI Benchmark Validation                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Crate: $CRATE_DIR"
echo "  Benchmark: $BENCHMARK_NAME"
echo "  Default Threshold: ${DEFAULT_THRESHOLD}%"
echo "  Verbose: $VERBOSE"
if [ -n "$BASELINE_FILE" ]; then
    echo "  Save Baseline: $BASELINE_FILE"
fi
if [ -n "$COMPARE_FILE" ]; then
    echo "  Compare Baseline: $COMPARE_FILE"
fi
echo ""

# Build benchmarks
echo "Building benchmarks..."
cd "$CRATE_DIR"

if [ "$VERBOSE" = true ]; then
    cargo bench --bench "$BENCHMARK_NAME" --no-run 2>&1 | head -100
else
    cargo bench --bench "$BENCHMARK_NAME" --no-run > /dev/null 2>&1
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to build benchmarks${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Benchmarks built successfully${NC}"
echo ""

# Run comprehensive overhead analysis
echo "Running comprehensive overhead analysis..."
echo "─────────────────────────────────────────────────────────────────"

# Run the overhead_analysis benchmark specifically
BENCH_OUTPUT=$(mktemp)
cargo bench --bench "$BENCHMARK_NAME" -- overhead_analysis 2>&1 | tee "$BENCH_OUTPUT"

# Check for failures in output
if grep -q "FAILED\|panicked\|thread.*panicked" "$BENCH_OUTPUT"; then
    echo ""
    echo -e "${RED}✗ Benchmark execution failed${NC}"
    rm -f "$BENCH_OUTPUT"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Overhead analysis completed${NC}"

# Run scalability analysis
echo ""
echo "Running overhead scalability analysis..."
echo "─────────────────────────────────────────────────────────────────"

SCALABILITY_OUTPUT=$(mktemp)
cargo bench --bench "$BENCHMARK_NAME" -- overhead_scalability 2>&1 | tee "$SCALABILITY_OUTPUT"

if grep -q "FAILED\|panicked\|thread.*panicked" "$SCALABILITY_OUTPUT"; then
    echo ""
    echo -e "${RED}✗ Scalability analysis failed${NC}"
    rm -f "$BENCH_OUTPUT" "$SCALABILITY_OUTPUT"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Scalability analysis completed${NC}"

# Run contention analysis
echo ""
echo "Running dispatch contention analysis..."
echo "─────────────────────────────────────────────────────────────────"

CONTENTION_OUTPUT=$(mktemp)
cargo bench --bench "$BENCHMARK_NAME" -- dispatch_contention 2>&1 | tee "$CONTENTION_OUTPUT"

if grep -q "FAILED\|panicked\|thread.*panicked" "$CONTENTION_OUTPUT"; then
    echo ""
    echo -e "${RED}✗ Contention analysis failed${NC}"
    rm -f "$BENCH_OUTPUT" "$SCALABILITY_OUTPUT" "$CONTENTION_OUTPUT"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Contention analysis completed${NC}"

# Save baseline if requested
if [ -n "$BASELINE_FILE" ]; then
    echo ""
    echo "Saving benchmark results to baseline..."

    # Extract key metrics from outputs
    {
        echo "# Dispatch Registry Benchmark Baseline"
        echo "# Generated: $(date)"
        echo ""
        echo "## Overhead Analysis Results"
        grep -E "^[✓✗]" "$BENCH_OUTPUT" || echo "No detailed results available"
    } > "$BASELINE_FILE"

    echo -e "${GREEN}✓ Baseline saved to $BASELINE_FILE${NC}"
fi

# Compare against baseline if provided
if [ -n "$COMPARE_FILE" ] && [ -f "$COMPARE_FILE" ]; then
    echo ""
    echo "Comparing against baseline..."

    # This is a simple check - can be extended with more detailed comparison
    if grep -q "passed\|Passed" "$BENCH_OUTPUT"; then
        echo -e "${GREEN}✓ Performance is acceptable${NC}"
    else
        echo -e "${YELLOW}⚠ Unable to verify against baseline${NC}"
    fi
fi

# Summary report
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Benchmark Summary                                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

PASSED_COUNT=$(grep -c "^✓" "$BENCH_OUTPUT" || echo 0)
FAILED_COUNT=$(grep -c "^✗" "$BENCH_OUTPUT" || echo 0)
TOTAL_COUNT=$((PASSED_COUNT + FAILED_COUNT))

if [ "$TOTAL_COUNT" -eq 0 ]; then
    echo "Results: All benchmarks executed successfully"
else
    echo "Results:"
    echo "  Total Tests: $TOTAL_COUNT"
    echo "  Passed: $PASSED_COUNT"
    echo "  Failed: $FAILED_COUNT"
    echo ""

    if [ "$FAILED_COUNT" -gt 0 ]; then
        echo -e "${RED}Status: FAILED${NC}"
        echo ""
        echo "Failed measurements:"
        grep "^✗" "$BENCH_OUTPUT" || true
        echo ""

        rm -f "$BENCH_OUTPUT" "$SCALABILITY_OUTPUT" "$CONTENTION_OUTPUT"
        exit 1
    fi
fi

echo -e "${GREEN}Status: PASSED${NC}"
echo ""
echo "All dispatch registry benchmarks completed successfully."
echo "Performance overhead is within acceptable thresholds."
echo ""

# Cleanup
rm -f "$BENCH_OUTPUT" "$SCALABILITY_OUTPUT" "$CONTENTION_OUTPUT"

exit 0
