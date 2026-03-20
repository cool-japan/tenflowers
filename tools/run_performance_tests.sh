#!/usr/bin/env bash
#
# Performance Testing Script for TenfloweRS
#
# This script provides convenient ways to run, manage, and analyze
# criterion benchmarks locally before pushing to CI.
#
# Usage:
#   ./tools/run_performance_tests.sh [command] [options]
#
# Commands:
#   baseline <name>    - Create a new baseline
#   compare <baseline> - Compare against a baseline
#   analyze           - Analyze regressions
#   list              - List available baselines
#   clean             - Clean benchmark artifacts
#   ci-test           - Run full CI test locally
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CRATES=("tenflowers-core" "tenflowers-autograd")
CORE_BENCHES=("dispatch_benchmarks" "ultra_performance_benchmark")
AUTOGRAD_BENCHES=("gradient_performance" "advanced_gradient_benchmarks")

echo_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

echo_success() {
    echo -e "${GREEN}✓${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to run benchmarks for a crate
run_crate_benchmarks() {
    local crate=$1
    local baseline=$2
    local comparison=$3

    cd "$PROJECT_ROOT"

    if [ -z "$baseline" ]; then
        echo_info "Running benchmarks for $crate..."
    else
        echo_info "Running benchmarks for $crate with baseline: $baseline..."
    fi

    case $crate in
        tenflowers-core)
            BENCHES=("${CORE_BENCHES[@]}")
            ;;
        tenflowers-autograd)
            BENCHES=("${AUTOGRAD_BENCHES[@]}")
            ;;
        *)
            echo_error "Unknown crate: $crate"
            return 1
            ;;
    esac

    cd "crates/$crate"

    for bench in "${BENCHES[@]}"; do
        echo_info "Running $bench..."

        if [ -z "$baseline" ]; then
            cargo bench --bench "$bench" -- --output-format bencher
        elif [ "$comparison" == "true" ]; then
            cargo bench --bench "$bench" -- --baseline "$baseline" 2>&1 | tee -a "bench_${baseline}.txt" || true
        else
            cargo bench --bench "$bench" -- --save-baseline "$baseline"
        fi
    done

    cd "$PROJECT_ROOT"
    echo_success "Completed benchmarks for $crate"
}

# Function to create a baseline
cmd_baseline() {
    local baseline_name=$1

    if [ -z "$baseline_name" ]; then
        echo_error "Baseline name required"
        echo "Usage: run_performance_tests.sh baseline <name>"
        return 1
    fi

    echo_info "Creating baseline: $baseline_name"

    for crate in "${CRATES[@]}"; do
        run_crate_benchmarks "$crate" "$baseline_name" false
    done

    echo_success "Baseline created: $baseline_name"
}

# Function to compare against a baseline
cmd_compare() {
    local baseline_name=$1

    if [ -z "$baseline_name" ]; then
        echo_error "Baseline name required"
        echo "Usage: run_performance_tests.sh compare <baseline>"
        return 1
    fi

    echo_info "Comparing against baseline: $baseline_name"

    for crate in "${CRATES[@]}"; do
        run_crate_benchmarks "$crate" "$baseline_name" true
    done

    echo_success "Comparison complete"
}

# Function to analyze regressions
cmd_analyze() {
    echo_info "Analyzing regressions..."

    python3 "$SCRIPT_DIR/performance_regression_detector.py" \
        --baseline-dir "$PROJECT_ROOT/target/criterion" \
        --current-dir "$PROJECT_ROOT/crates/tenflowers-core/target/criterion" \
        --config "$PROJECT_ROOT/.github/performance-gates.json" \
        --output-format markdown

    echo_success "Analysis complete"
}

# Function to list baselines
cmd_list() {
    echo_info "Available baselines:"
    echo ""

    for crate in "${CRATES[@]}"; do
        baseline_dir="$PROJECT_ROOT/crates/$crate/target/criterion"
        if [ -d "$baseline_dir" ]; then
            echo "  $crate:"
            find "$baseline_dir" -type d -name 'base' -o -name 'pr-*' 2>/dev/null | \
                sed 's|.*/||' | sort -u | \
                while read -r baseline; do
                    echo "    - $baseline"
                done || true
        fi
    done
}

# Function to clean artifacts
cmd_clean() {
    echo_warning "Cleaning benchmark artifacts..."

    for crate in "${CRATES[@]}"; do
        rm -rf "$PROJECT_ROOT/crates/$crate/target/criterion"
    done

    rm -f "$PROJECT_ROOT/crates/tenflowers-core/bench_*.txt"
    rm -f "$PROJECT_ROOT/crates/tenflowers-autograd/bench_*.txt"

    echo_success "Cleaned benchmark artifacts"
}

# Function to run CI test locally
cmd_ci_test() {
    echo_info "Running full CI performance test locally..."

    # Build in release mode
    echo_info "Building workspace in release mode..."
    cargo build --release --workspace

    # Run baseline benchmarks
    echo_info "Creating main baseline..."
    for crate in "${CRATES[@]}"; do
        run_crate_benchmarks "$crate" "main" false
    done

    # Run performance gate validation
    echo_info "Running performance gate validation..."
    cd "$PROJECT_ROOT/crates/tenflowers-core"
    cargo run --release --bin performance_gate_validation || true
    cd "$PROJECT_ROOT"

    echo_success "CI test complete"
}

# Main command handler
main() {
    local command=${1:-help}

    case $command in
        baseline)
            cmd_baseline "${2:-}"
            ;;
        compare)
            cmd_compare "${2:-}"
            ;;
        analyze)
            cmd_analyze
            ;;
        list)
            cmd_list
            ;;
        clean)
            cmd_clean
            ;;
        ci-test)
            cmd_ci_test
            ;;
        help|--help|-h)
            cat << EOF
TenfloweRS Performance Testing Tool

USAGE:
    run_performance_tests.sh [COMMAND] [OPTIONS]

COMMANDS:
    baseline <name>     Create a new baseline for comparison
    compare <baseline>  Compare current benchmarks against a baseline
    analyze            Analyze performance regressions
    list               List available baselines
    clean              Clean benchmark artifacts
    ci-test            Run full CI test locally

EXAMPLES:
    # Create a baseline before making changes
    ./tools/run_performance_tests.sh baseline before-optimization

    # Make changes to code...

    # Compare current performance against baseline
    ./tools/run_performance_tests.sh compare before-optimization

    # Analyze regressions
    ./tools/run_performance_tests.sh analyze

    # List all available baselines
    ./tools/run_performance_tests.sh list

    # Clean all benchmark artifacts
    ./tools/run_performance_tests.sh clean

    # Run complete CI test locally
    ./tools/run_performance_tests.sh ci-test

BENCHMARK LOCATIONS:
    - Core: crates/tenflowers-core/benches/
    - Autograd: crates/tenflowers-autograd/benches/

CONFIGURATION:
    - Thresholds: .github/performance-gates.json
    - Analysis: tools/performance_regression_detector.py

For more details, see PERFORMANCE_REGRESSION_TESTING.md
EOF
            ;;
        *)
            echo_error "Unknown command: $command"
            echo "Run './tools/run_performance_tests.sh help' for usage information"
            return 1
            ;;
    esac
}

# Run main function
main "$@"
