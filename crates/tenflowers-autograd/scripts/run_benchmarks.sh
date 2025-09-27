#!/bin/bash

# TenfloweRS Autograd Performance Benchmarking Script
# This script runs comprehensive benchmarks for gradient computation performance
# and includes comparison with PyTorch and TensorFlow

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}TenfloweRS Autograd Performance Benchmarking${NC}"
echo "=============================================="

# Check if criterion is available
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: cargo is not installed${NC}"
    exit 1
fi

# Create benchmark results directory
BENCHMARK_DIR="target/benchmark_results"
mkdir -p "$BENCHMARK_DIR"

# Get current timestamp for report naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="$BENCHMARK_DIR/benchmark_report_$TIMESTAMP.txt"

echo -e "${YELLOW}Starting benchmark run at $(date)${NC}"
echo "Results will be saved to: $REPORT_FILE"

# Function to run a benchmark and capture results
run_benchmark() {
    local benchmark_name=$1
    local description=$2
    
    echo -e "${YELLOW}Running $description...${NC}"
    echo "===========================================" >> "$REPORT_FILE"
    echo "Benchmark: $benchmark_name" >> "$REPORT_FILE"
    echo "Description: $description" >> "$REPORT_FILE"
    echo "Timestamp: $(date)" >> "$REPORT_FILE"
    echo "===========================================" >> "$REPORT_FILE"
    
    # Run the benchmark and capture output
    if cargo bench --bench "$benchmark_name" 2>&1 | tee -a "$REPORT_FILE"; then
        echo -e "${GREEN}✓ $description completed successfully${NC}"
    else
        echo -e "${RED}✗ $description failed${NC}"
        return 1
    fi
    
    echo "" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
}

# Function to run all benchmarks
run_all_benchmarks() {
    echo -e "${YELLOW}Running all gradient performance benchmarks...${NC}"
    
    # Basic gradient operations
    run_benchmark "gradient_performance" "Basic Gradient Operations"
    
    # Performance comparison and regression tests
    run_benchmark "performance_comparison" "Performance Comparison & Regression Tests"
    
    # PyTorch comparison benchmarks (if available)
    if command -v python3 &> /dev/null && python3 -c "import torch" 2>/dev/null; then
        run_benchmark "pytorch_comparison" "PyTorch Framework Comparison"
    else
        echo -e "${YELLOW}Skipping PyTorch comparison (PyTorch not available)${NC}"
    fi
    
    echo -e "${GREEN}All benchmarks completed!${NC}"
}

# Function to run specific benchmark categories
run_category_benchmarks() {
    local category=$1
    
    case $category in
        "basic")
            run_benchmark "gradient_performance" "Basic Gradient Operations"
            ;;
        "comparison")
            run_benchmark "performance_comparison" "Performance Comparison & Regression Tests"
            ;;
        "pytorch")
            run_pytorch_external_comparison
            ;;
        *)
            echo -e "${RED}Unknown category: $category${NC}"
            echo "Available categories: basic, comparison, pytorch"
            exit 1
            ;;
    esac
}

# Function to run external PyTorch comparison
run_pytorch_external_comparison() {
    echo -e "${YELLOW}Running external PyTorch comparison benchmarks...${NC}"
    
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python3 not found. Please install Python 3.${NC}"
        return 1
    fi
    
    if ! python3 -c "import torch" 2>/dev/null; then
        echo -e "${RED}PyTorch not installed. Install with: pip install torch${NC}"
        return 1
    fi
    
    # Get script directory
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    PYTORCH_SCRIPT="$SCRIPT_DIR/pytorch_benchmark.py"
    
    if [ ! -f "$PYTORCH_SCRIPT" ]; then
        echo -e "${RED}PyTorch benchmark script not found: $PYTORCH_SCRIPT${NC}"
        return 1
    fi
    
    echo "PyTorch External Benchmark Comparison" >> "$REPORT_FILE"
    echo "=====================================" >> "$REPORT_FILE"
    echo "Timestamp: $(date)" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Run PyTorch benchmarks on CPU
    echo -e "${YELLOW}Running PyTorch CPU benchmarks...${NC}"
    python3 "$PYTORCH_SCRIPT" --device cpu --iterations 50 2>&1 | tee -a "$REPORT_FILE"
    
    # Run PyTorch benchmarks on CUDA if available
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo -e "${YELLOW}Running PyTorch CUDA benchmarks...${NC}"
        python3 "$PYTORCH_SCRIPT" --device cuda --iterations 50 2>&1 | tee -a "$REPORT_FILE"
    else
        echo -e "${YELLOW}CUDA not available, skipping GPU benchmarks${NC}"
    fi
    
    echo "" >> "$REPORT_FILE"
    echo -e "${GREEN}✓ PyTorch external comparison completed${NC}"
}

# Function to generate performance summary
generate_summary() {
    echo -e "${YELLOW}Generating performance summary...${NC}"
    
    SUMMARY_FILE="$BENCHMARK_DIR/benchmark_summary_$TIMESTAMP.txt"
    
    echo "TenfloweRS Autograd Performance Summary" > "$SUMMARY_FILE"
    echo "=====================================" >> "$SUMMARY_FILE"
    echo "Generated: $(date)" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    # Extract key metrics from the benchmark results
    if [ -f "$REPORT_FILE" ]; then
        echo "Key Performance Metrics:" >> "$SUMMARY_FILE"
        echo "========================" >> "$SUMMARY_FILE"
        
        # Extract throughput information
        grep -A 2 -B 2 "Elements/sec" "$REPORT_FILE" | head -20 >> "$SUMMARY_FILE"
        
        echo "" >> "$SUMMARY_FILE"
        echo "Timing Information:" >> "$SUMMARY_FILE"
        echo "==================" >> "$SUMMARY_FILE"
        
        # Extract timing information
        grep -A 1 -B 1 "time:" "$REPORT_FILE" | head -20 >> "$SUMMARY_FILE"
        
        echo "" >> "$SUMMARY_FILE"
        echo "Full report available in: $REPORT_FILE" >> "$SUMMARY_FILE"
        
        echo -e "${GREEN}Summary generated: $SUMMARY_FILE${NC}"
    else
        echo -e "${RED}No benchmark results found to summarize${NC}"
    fi
}

# Function to run benchmarks with different configurations
run_configuration_benchmarks() {
    local config=$1
    
    case $config in
        "debug")
            echo -e "${YELLOW}Running benchmarks in debug mode...${NC}"
            RUSTFLAGS="-C opt-level=0" cargo bench --bench gradient_performance 2>&1 | tee -a "$REPORT_FILE"
            ;;
        "release")
            echo -e "${YELLOW}Running benchmarks in release mode...${NC}"
            RUSTFLAGS="-C opt-level=3" cargo bench --bench gradient_performance 2>&1 | tee -a "$REPORT_FILE"
            ;;
        "parallel")
            echo -e "${YELLOW}Running benchmarks with parallel features...${NC}"
            cargo bench --bench gradient_performance --features parallel 2>&1 | tee -a "$REPORT_FILE"
            ;;
        "gpu")
            echo -e "${YELLOW}Running benchmarks with GPU features...${NC}"
            cargo bench --bench gradient_performance --features gpu 2>&1 | tee -a "$REPORT_FILE"
            ;;
        *)
            echo -e "${RED}Unknown configuration: $config${NC}"
            echo "Available configurations: debug, release, parallel, gpu"
            exit 1
            ;;
    esac
}

# Function to compare with baseline
compare_with_baseline() {
    local baseline_file=$1
    
    if [ ! -f "$baseline_file" ]; then
        echo -e "${RED}Baseline file not found: $baseline_file${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Comparing with baseline...${NC}"
    
    COMPARISON_FILE="$BENCHMARK_DIR/comparison_$TIMESTAMP.txt"
    
    echo "Performance Comparison with Baseline" > "$COMPARISON_FILE"
    echo "====================================" >> "$COMPARISON_FILE"
    echo "Current run: $(date)" >> "$COMPARISON_FILE"
    echo "Baseline: $baseline_file" >> "$COMPARISON_FILE"
    echo "" >> "$COMPARISON_FILE"
    
    # This would need to be implemented based on the specific format of criterion output
    echo "Comparison functionality to be implemented based on criterion output format" >> "$COMPARISON_FILE"
    
    echo -e "${GREEN}Comparison generated: $COMPARISON_FILE${NC}"
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -a, --all               Run all benchmarks (default)"
    echo "  -c, --category CATEGORY Run specific category (basic, comparison, pytorch)"
    echo "  -f, --config CONFIG     Run with specific configuration (debug, release, parallel, gpu)"
    echo "  -b, --baseline FILE     Compare results with baseline file"
    echo "  -s, --summary           Generate performance summary after benchmarks"
    echo "  -o, --output DIR        Set output directory for results"
    echo ""
    echo "Examples:"
    echo "  $0 -a                   # Run all benchmarks"
    echo "  $0 -c basic             # Run only basic gradient benchmarks"
    echo "  $0 -c pytorch           # Run PyTorch comparison benchmarks"
    echo "  $0 -f parallel          # Run benchmarks with parallel features"
    echo "  $0 -s                   # Run all benchmarks and generate summary"
}

# Parse command line arguments
CATEGORY=""
CONFIG=""
BASELINE=""
GENERATE_SUMMARY=false
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -a|--all)
            # Default behavior, no action needed
            shift
            ;;
        -c|--category)
            CATEGORY="$2"
            shift 2
            ;;
        -f|--config)
            CONFIG="$2"
            shift 2
            ;;
        -b|--baseline)
            BASELINE="$2"
            shift 2
            ;;
        -s|--summary)
            GENERATE_SUMMARY=true
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Update output directory if specified
if [ -n "$OUTPUT_DIR" ]; then
    BENCHMARK_DIR="$OUTPUT_DIR"
    mkdir -p "$BENCHMARK_DIR"
    REPORT_FILE="$BENCHMARK_DIR/benchmark_report_$TIMESTAMP.txt"
fi

# Main execution
echo "Starting benchmark execution..." > "$REPORT_FILE"
echo "===============================" >> "$REPORT_FILE"
echo "System Information:" >> "$REPORT_FILE"
echo "  Date: $(date)" >> "$REPORT_FILE"
echo "  Host: $(hostname)" >> "$REPORT_FILE"
echo "  Rust version: $(rustc --version)" >> "$REPORT_FILE"
echo "  Cargo version: $(cargo --version)" >> "$REPORT_FILE"
echo "===============================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

if [ -n "$CONFIG" ]; then
    run_configuration_benchmarks "$CONFIG"
elif [ -n "$CATEGORY" ]; then
    run_category_benchmarks "$CATEGORY"
else
    run_all_benchmarks
fi

if [ -n "$BASELINE" ]; then
    compare_with_baseline "$BASELINE"
fi

if [ "$GENERATE_SUMMARY" = true ]; then
    generate_summary
fi

echo -e "${GREEN}Benchmark run completed!${NC}"
echo "Results saved to: $REPORT_FILE"

# Open the results directory if on macOS
if command -v open &> /dev/null; then
    open "$BENCHMARK_DIR"
fi