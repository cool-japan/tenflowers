"""
Performance Regression Testing Framework for TenfloweRS

This module implements comprehensive performance benchmarking and regression
testing for the Python bindings.
"""

import pytest
import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Callable, Optional, Any
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    operation: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    iterations: int
    throughput: Optional[float] = None  # Operations per second
    memory_mb: Optional[float] = None  # Peak memory usage
    metadata: Optional[Dict[str, Any]] = None


class PerformanceTester:
    """
    Performance testing and regression detection framework.
    """

    def __init__(
        self,
        warmup_iterations: int = 10,
        test_iterations: int = 100,
        regression_threshold: float = 1.10  # 10% slowdown threshold
    ):
        """
        Initialize performance tester.

        Args:
            warmup_iterations: Number of warmup runs before benchmarking
            test_iterations: Number of benchmark iterations
            regression_threshold: Threshold for regression detection (1.10 = 10% slower)
        """
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.regression_threshold = regression_threshold
        self.results: List[BenchmarkResult] = []
        self.baseline: Optional[Dict[str, BenchmarkResult]] = None

    def benchmark_operation(
        self,
        operation_name: str,
        operation_fn: Callable,
        setup_fn: Optional[Callable] = None,
        teardown_fn: Optional[Callable] = None,
        throughput_fn: Optional[Callable[[float], float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Benchmark a single operation.

        Args:
            operation_name: Name of the operation
            operation_fn: Function to benchmark
            setup_fn: Optional setup function (called before each iteration)
            teardown_fn: Optional teardown function (called after each iteration)
            throughput_fn: Optional function to calculate throughput from time
            metadata: Optional metadata about the benchmark

        Returns:
            BenchmarkResult with timing statistics
        """
        print(f"\nBenchmarking: {operation_name}")

        # Warmup
        print(f"  Warming up ({self.warmup_iterations} iterations)...")
        for _ in range(self.warmup_iterations):
            if setup_fn:
                setup_fn()
            operation_fn()
            if teardown_fn:
                teardown_fn()

        # Benchmark
        print(f"  Running benchmark ({self.test_iterations} iterations)...")
        times = []

        for i in range(self.test_iterations):
            if setup_fn:
                setup_fn()

            start_time = time.perf_counter()
            operation_fn()
            end_time = time.perf_counter()

            if teardown_fn:
                teardown_fn()

            times.append((end_time - start_time) * 1000)  # Convert to ms

            if (i + 1) % 20 == 0:
                print(f"    Progress: {i + 1}/{self.test_iterations}")

        # Calculate statistics
        times_array = np.array(times)
        mean_time = float(np.mean(times_array))
        std_time = float(np.std(times_array))
        min_time = float(np.min(times_array))
        max_time = float(np.max(times_array))

        # Calculate throughput if function provided
        throughput = None
        if throughput_fn:
            throughput = throughput_fn(mean_time / 1000)  # Pass time in seconds

        result = BenchmarkResult(
            operation=operation_name,
            mean_time_ms=mean_time,
            std_time_ms=std_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            iterations=self.test_iterations,
            throughput=throughput,
            metadata=metadata
        )

        print(f"  Results: {mean_time:.3f}ms ± {std_time:.3f}ms "
              f"(min: {min_time:.3f}ms, max: {max_time:.3f}ms)")
        if throughput:
            print(f"  Throughput: {throughput:.2f} ops/sec")

        self.results.append(result)
        return result

    def load_baseline(self, baseline_path: str):
        """Load baseline results from file."""
        path = Path(baseline_path)
        if path.exists():
            with open(path, 'r') as f:
                baseline_data = json.load(f)
                self.baseline = {
                    r['operation']: BenchmarkResult(**r)
                    for r in baseline_data
                }
            print(f"Loaded baseline from {baseline_path}")
        else:
            print(f"No baseline found at {baseline_path}")

    def save_baseline(self, baseline_path: str):
        """Save current results as baseline."""
        path = Path(baseline_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        print(f"Saved baseline to {baseline_path}")

    def detect_regressions(self) -> List[tuple]:
        """
        Detect performance regressions compared to baseline.

        Returns:
            List of (operation_name, current_time, baseline_time, ratio) tuples
        """
        if not self.baseline:
            print("No baseline loaded for regression detection")
            return []

        regressions = []

        for result in self.results:
            if result.operation in self.baseline:
                baseline = self.baseline[result.operation]
                ratio = result.mean_time_ms / baseline.mean_time_ms

                if ratio > self.regression_threshold:
                    regressions.append((
                        result.operation,
                        result.mean_time_ms,
                        baseline.mean_time_ms,
                        ratio
                    ))

        return regressions

    def generate_report(self) -> str:
        """Generate comprehensive performance report."""
        lines = [
            "=" * 80,
            "PERFORMANCE BENCHMARK REPORT",
            "=" * 80,
            f"Warmup iterations: {self.warmup_iterations}",
            f"Test iterations: {self.test_iterations}",
            "",
            "Results:",
            "-" * 80,
            f"{'Operation':<40} {'Mean (ms)':<12} {'Std (ms)':<12} {'Throughput':<15}",
            "-" * 80
        ]

        for result in self.results:
            throughput_str = f"{result.throughput:.2f} ops/s" if result.throughput else "N/A"
            lines.append(
                f"{result.operation:<40} "
                f"{result.mean_time_ms:>10.3f}   "
                f"{result.std_time_ms:>10.3f}   "
                f"{throughput_str:<15}"
            )

        # Add regression detection if baseline available
        if self.baseline:
            lines.extend([
                "",
                "Regression Analysis:",
                "-" * 80,
                f"Threshold: {(self.regression_threshold - 1) * 100:.0f}% slowdown",
                ""
            ])

            regressions = self.detect_regressions()
            if regressions:
                lines.append(f"⚠️  Found {len(regressions)} regression(s):")
                for op, current, baseline, ratio in regressions:
                    pct_change = (ratio - 1) * 100
                    lines.append(
                        f"  - {op}: {current:.3f}ms (baseline: {baseline:.3f}ms) "
                        f"[+{pct_change:.1f}%]"
                    )
            else:
                lines.append("✓ No regressions detected!")

        lines.append("=" * 80)

        return "\n".join(lines)


# Benchmark tests
@pytest.mark.benchmark
def test_tensor_creation_performance():
    """Benchmark tensor creation operations."""
    import tenflowers as tf

    tester = PerformanceTester(warmup_iterations=10, test_iterations=100)

    # Benchmark zeros
    tester.benchmark_operation(
        "zeros_2x2",
        lambda: tf.zeros([2, 2]),
        metadata={"shape": [2, 2]}
    )

    tester.benchmark_operation(
        "zeros_100x100",
        lambda: tf.zeros([100, 100]),
        metadata={"shape": [100, 100]}
    )

    tester.benchmark_operation(
        "zeros_1000x1000",
        lambda: tf.zeros([1000, 1000]),
        metadata={"shape": [1000, 1000]}
    )

    # Benchmark ones
    tester.benchmark_operation(
        "ones_1000x1000",
        lambda: tf.ones([1000, 1000]),
        metadata={"shape": [1000, 1000]}
    )

    # Benchmark rand
    tester.benchmark_operation(
        "rand_1000x1000",
        lambda: tf.rand([1000, 1000]),
        metadata={"shape": [1000, 1000]}
    )

    print(tester.generate_report())


@pytest.mark.benchmark
def test_tensor_operations_performance():
    """Benchmark tensor operations."""
    import tenflowers as tf

    tester = PerformanceTester(warmup_iterations=10, test_iterations=100)

    # Setup tensors
    a_small = tf.ones([100, 100])
    b_small = tf.ones([100, 100])

    a_large = tf.ones([1000, 1000])
    b_large = tf.ones([1000, 1000])

    # Benchmark addition
    tester.benchmark_operation(
        "add_100x100",
        lambda: tf.add(a_small, b_small),
        metadata={"shape": [100, 100]}
    )

    tester.benchmark_operation(
        "add_1000x1000",
        lambda: tf.add(a_large, b_large),
        metadata={"shape": [1000, 1000]}
    )

    # Benchmark multiplication
    tester.benchmark_operation(
        "mul_1000x1000",
        lambda: tf.mul(a_large, b_large),
        metadata={"shape": [1000, 1000]}
    )

    print(tester.generate_report())


@pytest.mark.benchmark
def test_matmul_performance():
    """Benchmark matrix multiplication with GFLOPS calculation."""
    import tenflowers as tf

    tester = PerformanceTester(warmup_iterations=5, test_iterations=50)

    def gflops_calculator(m, n, k):
        """Calculate GFLOPS for matrix multiplication."""
        def calc(time_seconds):
            flops = 2 * m * n * k  # 2*m*n*k FLOPs for m×k @ k×n
            return (flops / time_seconds) / 1e9
        return calc

    # Various matrix sizes
    sizes = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]

    for m, k, n in sizes:
        a = tf.ones([m, k])
        b = tf.ones([k, n])

        tester.benchmark_operation(
            f"matmul_{m}x{k}x{n}",
            lambda: tf.matmul(a, b),
            throughput_fn=gflops_calculator(m, n, k),
            metadata={"m": m, "k": k, "n": n}
        )

    print(tester.generate_report())


@pytest.mark.benchmark
def test_numpy_interop_performance():
    """Benchmark NumPy interoperability."""
    import tenflowers as tf
    import numpy as np

    tester = PerformanceTester(warmup_iterations=10, test_iterations=100)

    # Create numpy array
    np_array = np.random.randn(1000, 1000).astype(np.float32)

    # Benchmark numpy -> tensor conversion
    tester.benchmark_operation(
        "numpy_to_tensor_1000x1000",
        lambda: tf.tensor_from_numpy(np_array),
        metadata={"shape": [1000, 1000]}
    )

    # Create tensor
    tensor = tf.rand([1000, 1000])

    # Benchmark tensor -> numpy conversion
    tester.benchmark_operation(
        "tensor_to_numpy_1000x1000",
        lambda: tf.tensor_to_numpy(tensor),
        metadata={"shape": [1000, 1000]}
    )

    print(tester.generate_report())


@pytest.mark.benchmark
def test_comprehensive_performance_suite():
    """Run comprehensive performance benchmark suite."""
    test_tensor_creation_performance()
    test_tensor_operations_performance()
    test_matmul_performance()
    test_numpy_interop_performance()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "benchmark"])
