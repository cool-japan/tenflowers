# Dispatch Registry Benchmarks - CI/CD Integration Guide

This guide explains how to integrate the dispatch registry benchmarks into your CI/CD pipeline.

## Table of Contents

1. [GitHub Actions Integration](#github-actions-integration)
2. [GitLab CI Integration](#gitlab-ci-integration)
3. [Jenkins Integration](#jenkins-integration)
4. [Local Pre-commit Hooks](#local-pre-commit-hooks)
5. [Performance Baseline Management](#performance-baseline-management)
6. [Troubleshooting](#troubleshooting)

## GitHub Actions Integration

### Quick Setup

1. Copy the workflow file:
```bash
mkdir -p .github/workflows
cp crates/tenflowers-core/benches/github_actions_workflow.yml .github/workflows/dispatch_benchmarks.yml
```

2. Commit and push:
```bash
git add .github/workflows/dispatch_benchmarks.yml
git commit -m "Add dispatch registry benchmark CI"
git push
```

3. The workflow will run on:
   - All PRs targeting `main`
   - All pushes to `main` and `develop`
   - Changes to dispatch registry or benchmark files

### Workflow Features

- **Multi-platform testing**: Runs on Ubuntu and macOS
- **Multiple Rust versions**: Tests with stable and nightly
- **Artifact upload**: Saves results for comparison
- **PR commenting**: Posts results directly on PRs
- **Caching**: Speeds up builds with cargo caching
- **Failure detection**: Catches performance regressions

### Customization

#### Change trigger branches:

```yaml
on:
  push:
    branches:
      - main
      - your-branch-name
  pull_request:
    branches:
      - main
```

#### Add more test platforms:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    rust: [stable, beta, nightly]
```

#### Modify failure behavior:

```yaml
# Allow failures on nightly (don't block PR)
continue-on-error: ${{ matrix.rust == 'nightly' }}
```

#### Set timeout for benchmarks:

```yaml
- name: Run dispatch benchmarks
  timeout-minutes: 30
  run: |
    cd crates/tenflowers-core
    cargo bench --bench dispatch_benchmarks
```

## GitLab CI Integration

### .gitlab-ci.yml Configuration

```yaml
stages:
  - build
  - benchmark
  - report

dispatch_benchmarks:
  stage: benchmark
  image: rust:latest
  variables:
    RUST_BACKTRACE: "1"
  script:
    - cd crates/tenflowers-core
    - cargo bench --bench dispatch_benchmarks --no-run
    - bash benches/ci_integration.sh --verbose
  artifacts:
    reports:
      performance: bench_results.json
    paths:
      - crates/tenflowers-core/bench_results.txt
    expire_in: 30 days
  only:
    - merge_requests
    - main
    - develop
  allow_failure: false

# Merge request comment with results
benchmark_report:
  stage: report
  image: rust:latest
  dependencies:
    - dispatch_benchmarks
  script:
    - |
      echo "## Dispatch Registry Benchmark Results"
      cat crates/tenflowers-core/bench_results.txt
  only:
    - merge_requests
  allow_failure: true
```

### Running Locally to Mimic CI

```bash
# Install docker
docker run --rm -it -v $PWD:/project rust:latest bash

# In container:
cd /project/crates/tenflowers-core
cargo bench --bench dispatch_benchmarks
```

## Jenkins Integration

### Jenkinsfile (Declarative Pipeline)

```groovy
pipeline {
    agent any

    options {
        timeout(time: 1, unit: 'HOURS')
        buildDiscarder(logRotator(numToKeepStr: '30'))
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup') {
            steps {
                sh '''
                    rustc --version
                    cargo --version
                '''
            }
        }

        stage('Build Benchmarks') {
            steps {
                dir('crates/tenflowers-core') {
                    sh 'cargo bench --bench dispatch_benchmarks --no-run'
                }
            }
        }

        stage('Run Benchmarks') {
            steps {
                dir('crates/tenflowers-core') {
                    sh '''
                        bash benches/ci_integration.sh \\
                            --verbose \\
                            --save-baseline build_${BUILD_NUMBER}_baseline.txt
                    '''
                }
            }
        }

        stage('Compare with Main') {
            when {
                changeRequest()
            }
            steps {
                dir('crates/tenflowers-core') {
                    sh '''
                        git fetch origin main:main
                        git show main:benches/dispatch_benchmarks.rs > \
                            /tmp/main_dispatch_benchmarks.rs
                        cp crates/tenflowers-core/benches/dispatch_benchmarks.rs \
                            /tmp/pr_dispatch_benchmarks.rs
                        diff -u /tmp/main_dispatch_benchmarks.rs \
                            /tmp/pr_dispatch_benchmarks.rs || true
                    '''
                }
            }
        }
    }

    post {
        always {
            dir('crates/tenflowers-core') {
                junit(testResults: 'target/test-results.xml',
                      allowEmptyResults: true)

                archiveArtifacts(artifacts: 'bench_results.txt',
                                allowEmptyArchive: true)

                publishHTML(target: [
                    reportDir: 'target/criterion',
                    reportFiles: 'report/index.html',
                    reportName: 'Benchmark Report',
                    allowMissing: false
                ])
            }
        }

        success {
            echo 'Dispatch benchmarks passed!'
        }

        failure {
            echo 'Dispatch benchmarks failed - performance regression detected!'
            // Send notifications, update issue, etc.
        }
    }
}
```

### Jenkins Configuration Files

Create `Jenkinsfile` in project root:
```bash
cp Jenkinsfile.template Jenkinsfile
git add Jenkinsfile
git commit -m "Add Jenkins pipeline for benchmarks"
```

## Local Pre-commit Hooks

### Setup Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash

# Dispatch Registry Benchmark Pre-commit Hook
# Runs overhead analysis before commit

CRATE_DIR="$(cd "$(dirname "$0")/../crates/tenflowers-core" && pwd)"

if [ ! -d "$CRATE_DIR" ]; then
    echo "Warning: tenflowers-core crate not found, skipping benchmark check"
    exit 0
fi

echo "Running dispatch registry overhead analysis..."

cd "$CRATE_DIR"

# Run quick overhead check
if ! cargo bench --bench dispatch_benchmarks -- overhead_analysis > /dev/null 2>&1; then
    echo ""
    echo "ERROR: Dispatch benchmarks failed!"
    echo "Please fix performance regressions before committing."
    exit 1
fi

echo "✓ Dispatch benchmarks passed - commit allowed"
exit 0
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

### Skip Hook When Necessary

If you need to skip the hook for a commit:
```bash
git commit --no-verify
```

## Performance Baseline Management

### Creating a Baseline

```bash
cd crates/tenflowers-core

# Create baseline from current code
bash benches/ci_integration.sh \
    --save-baseline baselines/v0.1.0.baseline

# Commit baseline
git add baselines/v0.1.0.baseline
git commit -m "Add performance baseline for v0.1.0"
```

### Comparing Against Baseline

```bash
# Compare current performance against baseline
bash benches/ci_integration.sh \
    --compare-baseline baselines/main.baseline

# Output shows performance delta
```

### Baseline File Format

Baseline files are simple text files with benchmark results:

```
# Dispatch Registry Benchmark Baseline
# Generated: 2024-02-06 10:30:45 UTC
# System: M1 MacBook Pro, Rust 1.75

## Overhead Analysis Results
✓ add_tiny_10: dispatch=450ns, direct=400ns, overhead=12.50% (threshold=5%)
✓ add_small_100: dispatch=1220ns, direct=1200ns, overhead=1.67% (threshold=5%)
✓ mul_medium_1k: dispatch=12150ns, direct=12000ns, overhead=1.25% (threshold=2%)
...
```

## Troubleshooting

### Issue: Benchmarks Timeout in CI

**Symptom**: GitHub Actions workflow times out

**Solutions**:
1. Increase timeout:
```yaml
- name: Run benchmarks
  timeout-minutes: 60
  run: cargo bench --bench dispatch_benchmarks
```

2. Skip detailed benchmarks in CI:
```yaml
run: cargo bench --bench dispatch_benchmarks -- overhead_analysis
```

3. Use faster runner (macOS runners are faster for some benchmarks)

### Issue: Inconsistent Results Between Runs

**Symptom**: Same code produces different overhead percentages

**Causes**:
- System load/background processes
- CPU frequency scaling
- Cache state
- Turbo boost interference

**Solutions**:
1. Disable CPU frequency scaling:
```bash
sudo cpupower frequency-set --governor performance
```

2. Run benchmarks in isolation:
```bash
killall -9 chrome firefox spotify # Close heavy apps
cargo bench --bench dispatch_benchmarks
```

3. Use dedicated CI hardware if possible

### Issue: Benchmarks Fail on CI but Pass Locally

**Symptom**: Performance regression detected in CI but not locally

**Causes**:
- Different hardware (CI runners vs developer machines)
- Different Rust version
- System library differences
- Background processes on CI

**Solutions**:
1. Run on exact same OS/compiler version:
```bash
rustup install 1.75.0
rustup default 1.75.0
cargo +1.75.0 bench --bench dispatch_benchmarks
```

2. Check CI runner hardware specs and match locally if possible

3. Relax thresholds slightly for CI (e.g., 2% -> 2.5%) if unavoidable

### Issue: Failing to Compile in CI

**Symptom**: Compilation error only appears in CI

**Causes**:
- Feature flag mismatch
- Platform-specific code
- Dependency version conflicts

**Solutions**:
1. Check feature flags match:
```yaml
- name: Build
  run: cargo build -p tenflowers-core --all-features
```

2. Test on multiple platforms:
```yaml
matrix:
  os: [ubuntu-latest, macos-latest, windows-latest]
```

3. Pin dependency versions if needed:
```toml
criterion = "=0.5.1"
```

### Issue: PR Benchmark Comments Not Appearing

**Symptom**: GitHub Actions runs successfully but no comment on PR

**Causes**:
- Insufficient permissions
- Workflow run on fork
- Parsing error in results

**Solutions**:
1. Check workflow permissions:
```yaml
permissions:
  contents: read
  pull-requests: write
  checks: write
```

2. Check that results file exists:
```yaml
- name: Debug
  if: always()
  run: ls -la crates/tenflowers-core/bench_results.txt
```

3. Test comment generation locally:
```bash
cd crates/tenflowers-core
bash benches/ci_integration.sh --verbose
```

## Best Practices

1. **Run benchmarks before pushing**: Use pre-commit hooks to catch regressions early

2. **Commit baselines**: Keep performance baselines in git for historical tracking

3. **Document threshold changes**: If you update thresholds, explain why in commit message

4. **Monitor trends**: Review benchmark history periodically for upward trends

5. **Profile when failing**: Use perf/flame graphs to understand regressions

6. **Keep CI configuration**: Regularly update CI workflow to match latest best practices

7. **Test on target hardware**: If possible, run benchmarks on target deployment hardware

## Integration Checklist

- [ ] Copy workflow file to `.github/workflows/`
- [ ] Enable Actions in repository settings
- [ ] Test workflow on a branch
- [ ] Set up baselines in `baselines/` directory
- [ ] Configure PR comments
- [ ] Set up artifact retention
- [ ] Test baseline comparison workflow
- [ ] Document in README for team
- [ ] Set up local pre-commit hook
- [ ] Monitor first 10 benchmark runs for stability

## References

- [Benchmark Documentation](./BENCHMARK_README.md)
- [Performance Gates](./PERFORMANCE_GATES.md)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitLab CI Documentation](https://docs.gitlab.com/ee/ci/)
- [Jenkins Documentation](https://www.jenkins.io/doc/)
