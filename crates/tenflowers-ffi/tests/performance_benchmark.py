#!/usr/bin/env python3
"""
Performance benchmark suite for TenfloweRS FFI.

This benchmark suite measures:
- Tensor operation performance
- Layer forward/backward pass speed
- Memory usage patterns
- Throughput metrics
- Comparison with NumPy where applicable
"""

import sys
import time
import numpy as np
from typing import Callable, Any, Dict, List, Tuple

try:
    import tenflowers as tf
except ImportError:
    print("ERROR: tenflowers module not found. Build and install the wheel first.")
    print("Run: cd crates/tenflowers-ffi && maturin develop")
    sys.exit(1)


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(
        self,
        name: str,
        elapsed_time: float,
        operations: int,
        throughput: float,
        memory_mb: float = 0.0,
        extra_info: Dict[str, Any] = None
    ):
        self.name = name
        self.elapsed_time = elapsed_time
        self.operations = operations
        self.throughput = throughput
        self.memory_mb = memory_mb
        self.extra_info = extra_info or {}

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Time: {self.elapsed_time*1000:.2f}ms\n"
            f"  Ops: {self.operations:,}\n"
            f"  Throughput: {self.throughput:.2f} ops/sec\n"
            f"  Memory: {self.memory_mb:.2f} MB"
        )


class BenchmarkSuite:
    """Performance benchmark suite."""

    def __init__(self, warmup_iters: int = 3, bench_iters: int = 10):
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
        self.results: List[BenchmarkResult] = []

    def benchmark(
        self,
        name: str,
        func: Callable,
        operations: int = 1
    ) -> BenchmarkResult:
        """Run a benchmark with warmup and multiple iterations."""
        print(f"\nBenchmarking: {name}")
        print(f"  Warmup iterations: {self.warmup_iters}")
        print(f"  Benchmark iterations: {self.bench_iters}")

        # Warmup
        for _ in range(self.warmup_iters):
            func()

        # Benchmark
        start_time = time.perf_counter()
        for _ in range(self.bench_iters):
            func()
        end_time = time.perf_counter()

        elapsed = (end_time - start_time) / self.bench_iters
        throughput = operations / elapsed if elapsed > 0 else 0

        result = BenchmarkResult(
            name=name,
            elapsed_time=elapsed,
            operations=operations,
            throughput=throughput
        )

        self.results.append(result)
        print(f"  ✓ {elapsed*1000:.2f}ms ({throughput:.2f} ops/sec)")
        return result

    def print_summary(self):
        """Print benchmark summary."""
        print(f"\n{'='*70}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"Total benchmarks: {len(self.results)}")
        print(f"\nResults (sorted by throughput):")
        print(f"{'-'*70}")

        sorted_results = sorted(self.results, key=lambda x: x.throughput, reverse=True)
        for i, result in enumerate(sorted_results, 1):
            print(f"{i}. {result.name}")
            print(f"   Time: {result.elapsed_time*1000:.3f}ms | "
                  f"Throughput: {result.throughput:.2f} ops/sec")

        print(f"{'='*70}")


# ============================================================================
# Tensor Operation Benchmarks
# ============================================================================

def bench_tensor_creation(suite: BenchmarkSuite):
    """Benchmark tensor creation operations."""
    print(f"\n{'='*70}")
    print("TENSOR CREATION BENCHMARKS")
    print(f"{'='*70}")

    shape = [1000, 1000]
    ops = 100

    suite.benchmark(
        "zeros(1000x1000) x100",
        lambda: [tf.zeros(shape) for _ in range(ops)],
        operations=ops
    )

    suite.benchmark(
        "ones(1000x1000) x100",
        lambda: [tf.ones(shape) for _ in range(ops)],
        operations=ops
    )

    suite.benchmark(
        "rand(1000x1000) x100",
        lambda: [tf.rand(shape) for _ in range(ops)],
        operations=ops
    )

    suite.benchmark(
        "randn(1000x1000) x100",
        lambda: [tf.randn(shape) for _ in range(ops)],
        operations=ops
    )


def bench_arithmetic_ops(suite: BenchmarkSuite):
    """Benchmark arithmetic operations."""
    print(f"\n{'='*70}")
    print("ARITHMETIC OPERATION BENCHMARKS")
    print(f"{'='*70}")

    size = 1000
    a = tf.rand([size, size])
    b = tf.rand([size, size])
    ops = 1000

    suite.benchmark(
        "add(1000x1000) x1000",
        lambda: [tf.add(a, b) for _ in range(ops)],
        operations=ops
    )

    suite.benchmark(
        "mul(1000x1000) x1000",
        lambda: [tf.mul(a, b) for _ in range(ops)],
        operations=ops
    )

    suite.benchmark(
        "sub(1000x1000) x1000",
        lambda: [tf.sub(a, b) for _ in range(ops)],
        operations=ops
    )

    suite.benchmark(
        "div(1000x1000) x1000",
        lambda: [tf.div(a, b) for _ in range(ops)],
        operations=ops
    )


def bench_matrix_ops(suite: BenchmarkSuite):
    """Benchmark matrix operations."""
    print(f"\n{'='*70}")
    print("MATRIX OPERATION BENCHMARKS")
    print(f"{'='*70}")

    sizes = [(100, 100), (500, 500), (1000, 1000)]

    for m, n in sizes:
        a = tf.rand([m, n])
        b = tf.rand([n, m])
        ops = 100 if m <= 500 else 10

        suite.benchmark(
            f"matmul({m}x{n}, {n}x{m}) x{ops}",
            lambda: [tf.matmul(a, b) for _ in range(ops)],
            operations=ops
        )


def bench_reduction_ops(suite: BenchmarkSuite):
    """Benchmark reduction operations."""
    print(f"\n{'='*70}")
    print("REDUCTION OPERATION BENCHMARKS")
    print(f"{'='*70}")

    size = 1000
    tensor = tf.rand([size, size])
    ops = 1000

    suite.benchmark(
        f"sum({size}x{size}) x{ops}",
        lambda: [tf.sum(tensor) for _ in range(ops)],
        operations=ops
    )

    suite.benchmark(
        f"mean({size}x{size}) x{ops}",
        lambda: [tf.mean(tensor) for _ in range(ops)],
        operations=ops
    )

    suite.benchmark(
        f"max({size}x{size}) x{ops}",
        lambda: [tf.max(tensor) for _ in range(ops)],
        operations=ops
    )

    suite.benchmark(
        f"min({size}x{size}) x{ops}",
        lambda: [tf.min(tensor) for _ in range(ops)],
        operations=ops
    )


def bench_shape_ops(suite: BenchmarkSuite):
    """Benchmark shape manipulation operations."""
    print(f"\n{'='*70}")
    print("SHAPE MANIPULATION BENCHMARKS")
    print(f"{'='*70}")

    tensor = tf.rand([100, 100, 10])
    ops = 1000

    suite.benchmark(
        f"reshape(100x100x10) x{ops}",
        lambda: [tensor.reshape([100, 1000]) for _ in range(ops)],
        operations=ops
    )

    suite.benchmark(
        f"transpose(100x100x10) x{ops}",
        lambda: [tensor.transpose(0, 2) for _ in range(ops)],
        operations=ops
    )

    suite.benchmark(
        f"squeeze(100x1x100) x{ops}",
        lambda: [tf.ones([100, 1, 100]).squeeze(1) for _ in range(ops)],
        operations=ops
    )


# ============================================================================
# Neural Network Layer Benchmarks
# ============================================================================

def bench_dense_layers(suite: BenchmarkSuite):
    """Benchmark Dense layer operations."""
    print(f"\n{'='*70}")
    print("DENSE LAYER BENCHMARKS")
    print(f"{'='*70}")

    layer_configs = [
        (128, 64),
        (512, 256),
        (1024, 512),
    ]

    batch_size = 32
    ops = 100

    for in_dim, out_dim in layer_configs:
        layer = tf.Dense(in_dim, out_dim, activation="relu")
        input_tensor = tf.rand([batch_size, in_dim])

        suite.benchmark(
            f"Dense({in_dim}->{out_dim}, batch={batch_size}) x{ops}",
            lambda: [layer.forward(input_tensor) for _ in range(ops)],
            operations=ops
        )


def bench_conv_layers(suite: BenchmarkSuite):
    """Benchmark convolutional layer operations."""
    print(f"\n{'='*70}")
    print("CONVOLUTIONAL LAYER BENCHMARKS")
    print(f"{'='*70}")

    batch_size = 16
    ops = 50

    configs = [
        (3, 16, 32, 32),    # (in_ch, out_ch, H, W)
        (16, 32, 32, 32),
        (32, 64, 16, 16),
    ]

    for in_ch, out_ch, h, w in configs:
        conv = tf.Conv2D(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        input_tensor = tf.rand([batch_size, in_ch, h, w])

        suite.benchmark(
            f"Conv2D({in_ch}->{out_ch}, {h}x{w}, batch={batch_size}) x{ops}",
            lambda: [conv.forward(input_tensor) for _ in range(ops)],
            operations=ops
        )


def bench_normalization_layers(suite: BenchmarkSuite):
    """Benchmark normalization layer operations."""
    print(f"\n{'='*70}")
    print("NORMALIZATION LAYER BENCHMARKS")
    print(f"{'='*70}")

    batch_size = 32
    num_features = 256
    ops = 200

    # BatchNorm1d
    bn = tf.BatchNorm1d(num_features)
    input_tensor = tf.rand([batch_size, num_features])

    suite.benchmark(
        f"BatchNorm1d(features={num_features}, batch={batch_size}) x{ops}",
        lambda: [bn.forward(input_tensor) for _ in range(ops)],
        operations=ops
    )

    # LayerNorm
    ln = tf.LayerNorm(num_features)

    suite.benchmark(
        f"LayerNorm(features={num_features}, batch={batch_size}) x{ops}",
        lambda: [ln.forward(input_tensor) for _ in range(ops)],
        operations=ops
    )


def bench_recurrent_layers(suite: BenchmarkSuite):
    """Benchmark recurrent layer operations."""
    print(f"\n{'='*70}")
    print("RECURRENT LAYER BENCHMARKS")
    print(f"{'='*70}")

    batch_size = 16
    seq_len = 20
    input_size = 64
    hidden_size = 128
    ops = 20

    # LSTM
    lstm = tf.LSTM(input_size, hidden_size, num_layers=1)
    input_tensor = tf.rand([seq_len, batch_size, input_size])

    suite.benchmark(
        f"LSTM(in={input_size}, hidden={hidden_size}, seq={seq_len}, batch={batch_size}) x{ops}",
        lambda: [lstm.forward(input_tensor) for _ in range(ops)],
        operations=ops
    )

    # GRU
    gru = tf.GRU(input_size, hidden_size, num_layers=1)

    suite.benchmark(
        f"GRU(in={input_size}, hidden={hidden_size}, seq={seq_len}, batch={batch_size}) x{ops}",
        lambda: [gru.forward(input_tensor) for _ in range(ops)],
        operations=ops
    )


def bench_optimizer_steps(suite: BenchmarkSuite):
    """Benchmark optimizer step operations."""
    print(f"\n{'='*70}")
    print("OPTIMIZER STEP BENCHMARKS")
    print(f"{'='*70}")

    # Create simple model
    layer = tf.Dense(256, 128)
    params = layer.parameters()
    ops = 1000

    optimizers = [
        ("SGD", tf.SGD(params, lr=0.01)),
        ("Adam", tf.Adam(params, lr=0.001)),
        ("AdamW", tf.AdamW(params, lr=0.001)),
        ("RMSprop", tf.RMSprop(params, lr=0.01)),
        ("AdaBelief", tf.AdaBelief(params, lr=0.001)),
        ("RAdam", tf.RAdam(params, lr=0.001)),
        ("Nadam", tf.Nadam(params, lr=0.002)),
    ]

    for name, optimizer in optimizers:
        suite.benchmark(
            f"{name}.zero_grad() x{ops}",
            lambda: [optimizer.zero_grad() for _ in range(ops)],
            operations=ops
        )


# ============================================================================
# NumPy Comparison Benchmarks
# ============================================================================

def bench_numpy_comparison(suite: BenchmarkSuite):
    """Compare TenfloweRS performance with NumPy."""
    print(f"\n{'='*70}")
    print("NUMPY COMPARISON BENCHMARKS")
    print(f"{'='*70}")

    size = 1000
    ops = 100

    # NumPy matrix multiplication
    np_a = np.random.rand(size, size).astype(np.float32)
    np_b = np.random.rand(size, size).astype(np.float32)

    suite.benchmark(
        f"NumPy matmul({size}x{size}) x{ops}",
        lambda: [np.matmul(np_a, np_b) for _ in range(ops)],
        operations=ops
    )

    # TenfloweRS matrix multiplication
    tf_a = tf.rand([size, size])
    tf_b = tf.rand([size, size])

    suite.benchmark(
        f"TenfloweRS matmul({size}x{size}) x{ops}",
        lambda: [tf.matmul(tf_a, tf_b) for _ in range(ops)],
        operations=ops
    )

    # NumPy element-wise operations
    suite.benchmark(
        f"NumPy add({size}x{size}) x{ops}",
        lambda: [np.add(np_a, np_b) for _ in range(ops)],
        operations=ops
    )

    # TenfloweRS element-wise operations
    suite.benchmark(
        f"TenfloweRS add({size}x{size}) x{ops}",
        lambda: [tf.add(tf_a, tf_b) for _ in range(ops)],
        operations=ops
    )


# ============================================================================
# Memory Benchmark
# ============================================================================

def bench_memory_patterns(suite: BenchmarkSuite):
    """Benchmark memory usage patterns."""
    print(f"\n{'='*70}")
    print("MEMORY USAGE BENCHMARKS")
    print(f"{'='*70}")

    sizes = [100, 500, 1000, 2000]

    for size in sizes:
        tensors = []

        def create_and_store():
            for _ in range(10):
                t = tf.rand([size, size])
                tensors.append(t)
            tensors.clear()

        suite.benchmark(
            f"Create/destroy 10x({size}x{size}) tensors",
            create_and_store,
            operations=10
        )


def main():
    """Run all performance benchmarks."""
    print("="*70)
    print("TenfloweRS FFI Performance Benchmark Suite")
    print(f"Version: {tf.version()}")
    print("="*70)

    # Create benchmark suite
    suite = BenchmarkSuite(warmup_iters=3, bench_iters=10)

    # Run benchmarks
    bench_tensor_creation(suite)
    bench_arithmetic_ops(suite)
    bench_matrix_ops(suite)
    bench_reduction_ops(suite)
    bench_shape_ops(suite)
    bench_dense_layers(suite)
    bench_conv_layers(suite)
    bench_normalization_layers(suite)
    bench_recurrent_layers(suite)
    bench_optimizer_steps(suite)
    bench_numpy_comparison(suite)
    bench_memory_patterns(suite)

    # Print summary
    suite.print_summary()

    print(f"\nBenchmark suite completed successfully!")


if __name__ == "__main__":
    main()
