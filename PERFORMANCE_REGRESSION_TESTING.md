# Performance Regression Detection System

## Overview

The TenfloweRS performance regression detection system provides comprehensive monitoring of benchmark performance across code changes. It integrates criterion-based benchmarking with GitHub Actions CI pipeline to automatically detect and report performance regressions.

## Features

- **Automatic Baseline Management**: Maintains baseline benchmarks on the main branch
- **Cross-Branch Comparison**: Compares PR/feature branches against main baseline
- **Configurable Thresholds**: Different thresholds for critical vs non-critical operations
- **PR Comments**: Automatic GitHub PR comments with detailed regression analysis
- **Artifact Archival**: Stores benchmark results for historical trend analysis
- **Fail-on-Regression**: CI pipeline fails if critical regressions are detected
- **Local Testing**: Python tool for local regression analysis before pushing

## Architecture

### GitHub Actions Workflow (`performance-gate.yml`)

The workflow consists of two jobs:

#### 1. `benchmark-main` Job
- **Trigger**: Automatic on push to `main` branch
- **Purpose**: Establish baseline benchmarks
- **Actions**:
  - Runs criterion benchmarks for all crates
  - Saves results as "main" baseline
  - Archives results as artifacts for 90 days
  - Runs performance gate validation

#### 2. `benchmark-pr` Job
- **Trigger**: Automatic on pull requests, manual via `workflow_dispatch`
- **Purpose**: Detect regressions in feature branches
- **Actions**:
  - Runs criterion benchmarks on PR code
  - Downloads main baseline artifacts
  - Compares results using Python analysis script
  - Comments on PR with regression details
  - Fails workflow if critical regressions detected

### Regression Detection (`analyze_regressions.py`)

Embedded in the workflow, this script:
- Parses criterion JSON output files
- Identifies regressions (measured time > baseline time)
- Classifies by severity (warning vs critical)
- Applies threshold logic
- Generates markdown reports
- Sets GitHub workflow outputs for downstream steps

### Configuration (`performance-gates.json`)

Centralized configuration file defining:
- Global regression thresholds (default: 10%, critical: 5%)
- Critical operation list
- Per-benchmark thresholds
- Notification preferences
- Historical monitoring settings

## Performance Thresholds

### Default Settings

```yaml
default_regression_threshold: 10%    # General operations
critical_operation_threshold: 5%     # Core tensor operations
```

### Critical Operations

Operations receiving stricter thresholds (5%):
- `matmul` - Matrix multiplication
- `dispatch` - Operation dispatch overhead
- `add`, `mul` - Binary operations
- `sum`, `mean` - Reduction operations
- `gradient`, `backward` - Gradient computation

### Per-Benchmark Configuration

Custom thresholds can be defined in `performance-gates.json`:

```json
{
  "benchmark_groups": {
    "dispatch": {
      "operations": [
        {
          "name": "dispatch_binary/add_dispatch/tiny_10",
          "baseline_ns": 500,
          "threshold_percent": 5,
          "critical": true
        }
      ]
    }
  }
}
```

## Benchmark Organization

### tenflowers-core
- **dispatch_benchmarks.rs**: Dispatch registry overhead measurements
  - Binary operations (add, mul) across tensor sizes
  - Unary operations (abs)
  - Matrix operations (2D)
  - Chained operations
  - Dispatch overhead isolation

- **ultra_performance_benchmark.rs**: Core tensor operation performance
  - Matrix multiplication across sizes (8x8 to 512x512)
  - Different matrix aspect ratios
  - Batch operations
  - Memory patterns and cache efficiency

### tenflowers-autograd
- **gradient_performance.rs**: Automatic differentiation benchmarks
  - Simple backpropagation
  - Complex computation graphs
  - Memory efficiency during backward pass

- **advanced_gradient_benchmarks.rs**: Advanced gradient scenarios
  - Multiway merge operations
  - Higher-order gradients
  - Composite operation chains

## Workflow Execution

### On Main Branch Push

```
1. Checkout repository
2. Install dependencies
3. Build in release mode
4. Run criterion benchmarks for all crates
5. Save baselines as artifacts
6. Run performance gate validation
7. Archive results (90-day retention)
```

### On Pull Request

```
1. Checkout PR branch
2. Build in release mode
3. Run criterion benchmarks on PR code
4. Download main branch baseline artifacts
5. Merge baseline results
6. Run comparison benchmarks
7. Analyze regressions using Python script
8. Comment on PR with results
9. Archive results (30-day retention)
10. Fail workflow if critical regressions detected
```

## Local Testing

### Running Benchmarks Locally

```bash
# Run benchmarks for specific crate
cd crates/tenflowers-core
cargo bench --bench dispatch_benchmarks -- --save-baseline my-baseline

# Compare against baseline
cargo bench --bench dispatch_benchmarks -- --baseline my-baseline

# Run all benchmarks with detailed output
cargo bench --bench ultra_performance_benchmark -- --verbose
```

### Analyzing Regressions Locally

```bash
# Using the Python analysis tool
python3 tools/performance_regression_detector.py \
  --baseline-dir target/criterion \
  --current-dir crates/tenflowers-core/target/criterion \
  --config .github/performance-gates.json

# Generate JSON report
python3 tools/performance_regression_detector.py \
  --output-format json \
  --json-report report.json
```

## Interpreting Results

### GitHub PR Comment Example

```
## ⚡ Performance Benchmark Results

**Status**: ✅ No Significant Regressions

**Summary**:
- Found 3 regressions
- 0 critical regressions
- 3 within acceptable threshold

**Warnings (within threshold)**:
⚠️ dispatch_binary/add_dispatch/medium_1k: +2.50%
⚠️ matmul_sizes/standard_matmul/128: +1.75%
⚠️ gradient_performance/simple_backprop: +0.50%

**Thresholds**:
- General operations: 10% regression allowed
- Critical operations: 5% regression allowed
```

### Regression Classification

| Status | Icon | Meaning | Action |
|--------|------|---------|--------|
| Critical | ❌ | Exceeds threshold | Workflow fails, PR blocked |
| Warning | ⚠️ | Below threshold | Allowed, but noted |
| Improvement | ✅ | Performance improved | Recorded, no action |
| No change | ✓ | Within noise | Ignored |

## Interpreting Performance Metrics

### Measurement Units

- **baseline_ns**: Baseline execution time (nanoseconds)
- **measured_ns**: Current execution time (nanoseconds)
- **regression_pct**: Percentage increase from baseline

### Examples

```
Baseline: 100_000 ns
Current:  105_000 ns
Regression: +5.0%

matmul_64x64: +2.5% (100_000ns → 102_500ns, threshold: 10%)
```

## Troubleshooting

### Workflow Timeouts

If benchmarks take too long:
1. Increase `timeout-minutes` in workflow
2. Reduce sample size in `performance_gates.json`
3. Reduce number of benchmark sizes tested

### False Positives

Random variance in measurements can cause false positives:
1. Check that baseline was on same hardware
2. Verify consistent system load
3. Run locally to confirm regression
4. Consider increasing threshold temporarily

### Missing Baselines

If "criterion-baselines-main" artifact is not found:
1. Ensure main branch has pushed changes
2. Check artifact retention (90 days default)
3. Manually trigger benchmark-main job via workflow_dispatch
4. Download and re-upload baselines

### Performance Gate Validation Failures

The `performance_gate_validation` binary may fail due to:
1. Baselines not initialized in codebase
2. Hardware differences from baseline creation
3. Regression detected by safety checks

Solution: Run locally and adjust baselines in `performance_gates.rs`

## Maintenance

### Updating Baselines

When intentional performance changes are made:

```bash
# Locally establish new baseline
cd crates/tenflowers-core
cargo bench --bench dispatch_benchmarks -- --save-baseline new-baseline

# Review changes
cargo bench --bench dispatch_benchmarks -- --baseline new-baseline
```

After merging to main, the benchmark-main job automatically updates baselines.

### Adjusting Thresholds

Edit `.github/performance-gates.json`:

```json
{
  "global_settings": {
    "default_regression_threshold_percent": 15,  // Increase tolerance
    "critical_operation_threshold_percent": 8
  }
}
```

Commit and push changes; they take effect on next workflow run.

### Adding New Benchmarks

1. Add criterion benchmark in `crates/*/benches/`
2. Add entry to `Cargo.toml` `[[bench]]` section
3. Add to performance-gates.json configuration
4. Push to main branch (benchmark-main job will capture baseline)
5. Use in PRs (benchmark-pr job will compare)

Example Cargo.toml addition:
```toml
[[bench]]
name = "my_new_benchmark"
harness = false
```

## Integration with Other CI Checks

The performance gate workflow runs alongside:
- **Clippy**: Linting checks
- **Tests**: Functional correctness
- **Build**: Compilation verification

All must pass for PR approval. Performance gates can block merges if regressions are critical.

## Performance Optimization Workflow

1. **Benchmark locally** before changes
   ```bash
   cargo bench --bench dispatch_benchmarks -- --save-baseline before
   ```

2. **Make optimization changes**

3. **Benchmark after changes**
   ```bash
   cargo bench --bench dispatch_benchmarks -- --baseline before
   ```

4. **Review regression analysis**
   - If improvements: Great! Commit and push
   - If regressions: Investigate and optimize further

5. **Submit PR** - CI will run full suite

6. **Review feedback** - PR comment will show detailed results

## Historical Trend Monitoring

Currently, the system:
- Archives baselines for 90 days
- Stores detailed criterion HTML reports
- Can be extended with trend analysis

Future enhancements:
- Track performance over many commits
- Detect gradual degradation
- Automatic trend reports

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [TenfloweRS Benchmarking Guide](./PERFORMANCE_TUNING.md)

## Contributing

When contributing performance-critical code:

1. Run benchmarks locally before pushing
2. Document any intentional performance changes
3. Explain regressions in PR comments
4. Only merge if regressions are acceptable
5. Update baselines after merge if needed

## Questions?

For issues with the performance regression system:
1. Check workflow logs in GitHub Actions
2. Review performance analysis report in PR comments
3. Run local analysis with verbose output
4. File issues with detailed measurements
