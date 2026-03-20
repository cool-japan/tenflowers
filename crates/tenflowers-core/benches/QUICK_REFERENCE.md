# Dispatch Registry Benchmarks - Quick Reference

## Quick Start

### Run all benchmarks
```bash
cd crates/tenflowers-core
cargo bench --bench dispatch_benchmarks
```

### Run specific benchmark
```bash
cargo bench --bench dispatch_benchmarks -- overhead_analysis
```

### Run with CI script (recommended for local validation)
```bash
cd crates/tenflowers-core
bash benches/ci_integration.sh --verbose
```

## Performance Thresholds At a Glance

| Tensor Size | Max Overhead | Status |
|------------|-------------|--------|
| 10 elements | 5% | ✓ Small (rare in practice) |
| 100 elements | 5% | ✓ Small |
| 1,000 elements | 2% | ✓ Medium (typical) |
| 10,000 elements | 1% | ✓ Large |
| 100,000 elements | 1% | ✓ Extra-large |

## Understanding Benchmark Output

### ✓ PASS
```
✓ add_medium_1k: dispatch=12100ns, direct=12000ns, overhead=0.83% (threshold=2%)
```
Operation is within acceptable overhead threshold.

### ✗ FAIL
```
✗ add_tiny_10: dispatch=450ns, direct=400ns, overhead=12.5% (threshold=5%)
```
Overhead exceeds threshold - investigate performance regression.

### Summary Report
```
Total tests: 15
Passed:      15 ✓
Failed:      0 ✗

Average overhead by operation:
  Add:  1.23%
  Mul:  1.45%
  Abs:  0.89%
```

## Benchmark Groups

| Group | Purpose | Time |
|-------|---------|------|
| `dispatch_binary` | Binary operations (add, mul) | ~1-2 min |
| `dispatch_unary` | Unary operations (abs) | ~1 min |
| `dispatch_overhead` | Pure dispatch overhead | ~30 sec |
| `dispatch_matrix` | 2D matrix operations | ~1 min |
| `dispatch_chained` | Chained operations | ~30 sec |
| `overhead_analysis` | Comprehensive analysis | ~2 min |
| `overhead_scalability` | Scaling properties | ~2 min |
| `dispatch_contention` | Lock contention test | ~30 sec |

## Common Tasks

### Save performance baseline
```bash
cd crates/tenflowers-core
bash benches/ci_integration.sh --save-baseline baselines/myversion.baseline
git add baselines/myversion.baseline
git commit -m "Add performance baseline for version X.Y.Z"
```

### Compare against baseline
```bash
cd crates/tenflowers-core
bash benches/ci_integration.sh --compare-baseline baselines/main.baseline
```

### Run specific test size
```bash
# Edit SIZES array in dispatch_benchmarks.rs or run specific group
cargo bench --bench dispatch_benchmarks -- overhead_analysis
```

### Profile with perf (Linux)
```bash
cargo build --release --bench dispatch_benchmarks
perf record -g ./target/release/deps/dispatch_benchmarks--*
perf report
```

### Get detailed output
```bash
RUST_LOG=debug cargo bench --bench dispatch_benchmarks -- --verbose
```

## Troubleshooting

### Benchmarks too slow
- Run specific group: `cargo bench -- overhead_analysis`
- Reduce iterations in code (default 100)
- Use faster hardware if available

### Inconsistent results
- Close other applications
- Run multiple times and average
- Try on different hardware
- Check system load

### Compilation fails
- Update Rust: `rustup update`
- Clean: `cargo clean`
- Check feature flags: `cargo build --all-features`

### Lock contention detected
- Run on quieter system (fewer background processes)
- Use dedicated CI hardware
- Check if multiple benchmarks running simultaneously

## Performance Gates Overview

### What we're measuring
- **Dispatch time**: Time to lookup operation, select kernel, call function
- **Direct time**: Time for direct CPU function call
- **Overhead**: `(dispatch_time - direct_time) / direct_time * 100%`

### Why these thresholds?
- **5% for small**: Dispatch setup is significant, small tensors are rare
- **2% for medium**: Most common case, overhead is imperceptible
- **1% for large**: Computation dominates, dispatch is negligible
- **Scalability**: Overhead must decrease with tensor size

## Expected Results

For a healthy dispatch registry:
```
Overhead should follow this pattern:
  10 elements:    ~5.0%
  100 elements:   ~4.0%
  1,000 elements: ~1.0%
  10,000 elements:~0.5%
  100,000 elem:   ~0.2%
```

If overhead increases with size → lock contention issue
If overhead stays high for large tensors → implementation issue

## Files Reference

| File | Purpose |
|------|---------|
| `dispatch_benchmarks.rs` | Main benchmark code |
| `BENCHMARK_README.md` | Complete user guide |
| `PERFORMANCE_GATES.md` | Threshold definitions |
| `CI_INTEGRATION_GUIDE.md` | CI setup instructions |
| `ci_integration.sh` | Automated CI script |
| `github_actions_workflow.yml` | GitHub Actions config |
| `QUICK_REFERENCE.md` | This file |

## Key Metrics to Track

1. **Average Overhead by Operation**: Should stay < 1-2%
2. **Scaling Trend**: Should improve (decrease %) with larger tensors
3. **Pass Rate**: Should be 100% on main branch
4. **Regression Frequency**: Should be < 1 regression per 100 commits

## When to Investigate

- [ ] Any benchmark fails CI
- [ ] Overhead increases for small tensors
- [ ] Overhead doesn't decrease for large tensors
- [ ] Lock contention analysis shows issues
- [ ] New operation added, overhead > threshold
- [ ] Backend changes, re-run all benchmarks

## Contact & Questions

- Check BENCHMARK_README.md for detailed info
- Check PERFORMANCE_GATES.md for threshold details
- Check CI_INTEGRATION_GUIDE.md for CI questions
- Profile with perf/flamegraph for deep investigation

---

**Last Updated**: 2026-03-20
**Version**: 0.1.0
**Status**: Ready for production use
