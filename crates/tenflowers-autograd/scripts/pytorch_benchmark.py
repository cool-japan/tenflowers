#!/usr/bin/env python3
"""
PyTorch benchmark script for comparing with TenfloweRS autograd performance.
This script implements equivalent operations to the Rust benchmarks for fair comparison.

Usage:
    python3 pytorch_benchmark.py [--output-json] [--verbose]

Requirements:
    pip install torch numpy argparse json time
"""

import torch
import numpy as np
import time
import json
import argparse
import sys
from typing import List, Dict, Any, Tuple


class BenchmarkResult:
    """Results from a single benchmark run"""
    
    def __init__(self, framework: str, operation: str, input_size: int, 
                 batch_size: int = 1, total_time_ms: float = 0.0):
        self.framework = framework
        self.operation = operation
        self.input_size = input_size
        self.batch_size = batch_size
        self.forward_time_ms = 0.0
        self.backward_time_ms = 0.0
        self.total_time_ms = total_time_ms
        self.memory_usage_mb = None
        self.throughput_ops_per_sec = 1000.0 / total_time_ms if total_time_ms > 0 else 0.0
    
    def with_forward_backward_split(self, forward_ms: float, backward_ms: float):
        self.forward_time_ms = forward_ms
        self.backward_time_ms = backward_ms
        return self
    
    def with_memory_usage(self, memory_mb: float):
        self.memory_usage_mb = memory_mb
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "operation": self.operation,
            "input_size": self.input_size,
            "batch_size": self.batch_size,
            "forward_time_ms": self.forward_time_ms,
            "backward_time_ms": self.backward_time_ms,
            "total_time_ms": self.total_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "throughput_ops_per_sec": self.throughput_ops_per_sec
        }


class PyTorchBenchmarkSuite:
    """Comprehensive PyTorch benchmark suite matching TenfloweRS tests"""
    
    def __init__(self, device: str = "cpu", warmup_iterations: int = 10, 
                 measurement_iterations: int = 100):
        self.device = torch.device(device)
        self.warmup_iterations = warmup_iterations
        self.measurement_iterations = measurement_iterations
        self.results: List[BenchmarkResult] = []
        
        # Set deterministic behavior for reproducible results
        torch.manual_seed(42)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def benchmark_operation(self, operation_func, operation_name: str, 
                          input_size: int, batch_size: int = 1) -> BenchmarkResult:
        """Benchmark a single operation with proper timing and memory tracking"""
        
        # Warmup
        for _ in range(self.warmup_iterations):
            try:
                operation_func()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"Warmup failed for {operation_name}: {e}")
                return BenchmarkResult("PyTorch", operation_name, input_size, batch_size, float('inf'))
        
        # Memory measurement before
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        
        # Timing measurement
        start_time = time.perf_counter()
        
        forward_times = []
        backward_times = []
        
        for _ in range(self.measurement_iterations):
            # Forward pass timing
            forward_start = time.perf_counter()
            try:
                result = operation_func()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                forward_end = time.perf_counter()
                forward_times.append((forward_end - forward_start) * 1000)
                
                # Backward pass timing (if result requires grad)
                if hasattr(result, 'backward') and result.requires_grad:
                    backward_start = time.perf_counter()
                    # Create a dummy loss for backward pass
                    if result.dim() > 0:
                        loss = result.sum()
                    else:
                        loss = result
                    loss.backward()
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    backward_end = time.perf_counter()
                    backward_times.append((backward_end - backward_start) * 1000)
                
            except Exception as e:
                print(f"Operation failed for {operation_name}: {e}")
                return BenchmarkResult("PyTorch", operation_name, input_size, batch_size, float('inf'))
        
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        
        # Memory measurement after
        memory_usage_mb = None
        if self.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage_mb = (peak_memory - start_memory) / (1024 * 1024)
        
        # Create result
        avg_forward_time = np.mean(forward_times) if forward_times else 0.0
        avg_backward_time = np.mean(backward_times) if backward_times else 0.0
        avg_total_time = total_time_ms / self.measurement_iterations
        
        result = BenchmarkResult("PyTorch", operation_name, input_size, batch_size, avg_total_time)
        result.with_forward_backward_split(avg_forward_time, avg_backward_time)
        
        if memory_usage_mb is not None:
            result.with_memory_usage(memory_usage_mb)
        
        return result
    
    def bench_basic_operations(self):
        """Benchmark basic element-wise operations"""
        sizes = [1000, 10000, 100000]
        
        for size in sizes:
            # Element-wise addition
            def add_op():
                x = torch.linspace(0.0, 1.0, size, device=self.device, requires_grad=True)
                y = torch.linspace(1.0, 2.0, size, device=self.device, requires_grad=True)
                return x + y
            
            result = self.benchmark_operation(add_op, "add_elementwise", size)
            self.results.append(result)
            
            # Element-wise multiplication
            def mul_op():
                x = torch.linspace(0.1, 1.0, size, device=self.device, requires_grad=True)
                y = torch.linspace(1.0, 2.0, size, device=self.device, requires_grad=True)
                return x * y
            
            result = self.benchmark_operation(mul_op, "mul_elementwise", size)
            self.results.append(result)
            
            # Power operation
            def pow_op():
                x = torch.linspace(0.1, 2.0, size, device=self.device, requires_grad=True)
                return torch.pow(x, 2.0)
            
            result = self.benchmark_operation(pow_op, "pow", size)
            self.results.append(result)
    
    def bench_matrix_operations(self):
        """Benchmark matrix operations"""
        matrix_sizes = [64, 128, 256, 512]
        
        for size in matrix_sizes:
            # Matrix multiplication
            def matmul_op():
                x = torch.zeros(size, size, device=self.device, requires_grad=True)
                y = torch.eye(size, device=self.device, requires_grad=True)
                return torch.matmul(x, y)
            
            result = self.benchmark_operation(matmul_op, "matmul", size * size)
            self.results.append(result)
            
            # Matrix transpose
            def transpose_op():
                x = torch.zeros(size, size, device=self.device, requires_grad=True)
                return x.t()
            
            result = self.benchmark_operation(transpose_op, "transpose", size * size)
            self.results.append(result)
            
            # Matrix determinant (for smaller sizes)
            if size <= 256:
                def det_op():
                    x = torch.eye(size, device=self.device, requires_grad=True)
                    return torch.det(x)
                
                result = self.benchmark_operation(det_op, "determinant", size * size)
                self.results.append(result)
    
    def bench_activation_functions(self):
        """Benchmark activation functions"""
        sizes = [1000, 10000, 100000]
        
        for size in sizes:
            # ReLU
            def relu_op():
                x = torch.linspace(-1.0, 1.0, size, device=self.device, requires_grad=True)
                return torch.relu(x)
            
            result = self.benchmark_operation(relu_op, "relu", size)
            self.results.append(result)
            
            # Sigmoid
            def sigmoid_op():
                x = torch.linspace(-5.0, 5.0, size, device=self.device, requires_grad=True)
                return torch.sigmoid(x)
            
            result = self.benchmark_operation(sigmoid_op, "sigmoid", size)
            self.results.append(result)
            
            # Tanh
            def tanh_op():
                x = torch.linspace(-2.0, 2.0, size, device=self.device, requires_grad=True)
                return torch.tanh(x)
            
            result = self.benchmark_operation(tanh_op, "tanh", size)
            self.results.append(result)
            
            # Softmax
            def softmax_op():
                x = torch.linspace(-1.0, 1.0, size, device=self.device, requires_grad=True)
                return torch.softmax(x, dim=0)
            
            result = self.benchmark_operation(softmax_op, "softmax", size)
            self.results.append(result)
            
            # GELU
            def gelu_op():
                x = torch.linspace(-2.0, 2.0, size, device=self.device, requires_grad=True)
                return torch.nn.functional.gelu(x)
            
            result = self.benchmark_operation(gelu_op, "gelu", size)
            self.results.append(result)
    
    def bench_neural_networks(self):
        """Benchmark neural network operations"""
        configs = [
            (32, 784, 128, 10),    # MNIST-like
            (64, 2048, 512, 100),  # Larger network
            (128, 1024, 256, 50),  # Medium network
        ]
        
        for batch_size, input_size, hidden_size, output_size in configs:
            def mlp_op():
                # Create input
                x = torch.zeros(batch_size, input_size, device=self.device, requires_grad=True)
                
                # Create network parameters
                w1 = torch.zeros(input_size, hidden_size, device=self.device, requires_grad=True)
                b1 = torch.zeros(hidden_size, device=self.device, requires_grad=True)
                w2 = torch.zeros(hidden_size, output_size, device=self.device, requires_grad=True)
                b2 = torch.zeros(output_size, device=self.device, requires_grad=True)
                
                # Forward pass
                h1 = torch.matmul(x, w1) + b1
                h1_relu = torch.relu(h1)
                h2 = torch.matmul(h1_relu, w2) + b2
                output = torch.softmax(h2, dim=1)
                loss = output.sum()
                
                return loss
            
            total_params = batch_size * input_size
            result = self.benchmark_operation(mlp_op, "mlp_forward_backward", total_params, batch_size)
            self.results.append(result)
    
    def bench_convolution_operations(self):
        """Benchmark convolution operations"""
        conv_configs = [
            (8, 3, 32, 32, 16),   # Small conv
            (16, 3, 64, 64, 32),  # Medium conv  
            (32, 3, 128, 128, 64), # Large conv
        ]
        
        for batch_size, channels, height, width, filters in conv_configs:
            def conv_op():
                x = torch.zeros(batch_size, channels, height, width, device=self.device, requires_grad=True)
                conv_layer = torch.nn.Conv2d(channels, filters, kernel_size=3, padding=1).to(self.device)
                
                # Forward pass
                conv_out = conv_layer(x)
                activated = torch.relu(conv_out)
                loss = activated.sum()
                
                return loss
            
            total_elements = batch_size * channels * height * width
            result = self.benchmark_operation(conv_op, "conv2d_forward_backward", total_elements, batch_size)
            self.results.append(result)
    
    def bench_reduction_operations(self):
        """Benchmark reduction operations"""
        sizes = [1000, 10000, 100000]
        
        for size in sizes:
            # Sum
            def sum_op():
                x = torch.linspace(0.0, 1.0, size, device=self.device, requires_grad=True)
                return x.sum()
            
            result = self.benchmark_operation(sum_op, "sum", size)
            self.results.append(result)
            
            # Mean
            def mean_op():
                x = torch.linspace(0.0, 1.0, size, device=self.device, requires_grad=True)
                return x.mean()
            
            result = self.benchmark_operation(mean_op, "mean", size)
            self.results.append(result)
            
            # Max
            def max_op():
                x = torch.linspace(0.0, 1.0, size, device=self.device, requires_grad=True)
                return x.max()
            
            result = self.benchmark_operation(max_op, "max", size)
            self.results.append(result)
    
    def bench_advanced_math(self):
        """Benchmark advanced mathematical operations"""
        sizes = [1000, 10000, 100000]
        
        for size in sizes:
            # Trigonometric chain
            def trig_op():
                x = torch.linspace(-3.14, 3.14, size, device=self.device, requires_grad=True)
                y = torch.sin(x)
                z = torch.cos(y)
                w = y + z
                return w.sum()
            
            result = self.benchmark_operation(trig_op, "sin_cos_chain", size)
            self.results.append(result)
            
            # Exponential and logarithmic
            def exp_log_op():
                x = torch.linspace(0.1, 2.0, size, device=self.device, requires_grad=True)
                y = torch.exp(x)
                z = torch.log(y + 1e-8)
                w = torch.sqrt(z)
                return w.sum()
            
            result = self.benchmark_operation(exp_log_op, "exp_log_chain", size)
            self.results.append(result)
    
    def bench_higher_order_derivatives(self):
        """Benchmark higher-order derivatives"""
        sizes = [10, 100, 500]
        
        for size in sizes:
            def second_order_op():
                x = torch.linspace(0.1, 1.0, size, device=self.device, requires_grad=True)
                
                # First-order gradient
                y = x ** 3
                loss = y.sum()
                
                # Compute first gradient
                first_grad = torch.autograd.grad(loss, x, create_graph=True)[0]
                
                # Second-order gradient
                second_loss = first_grad.sum()
                second_grad = torch.autograd.grad(second_loss, x)[0]
                
                return second_grad.sum()
            
            result = self.benchmark_operation(second_order_op, "second_order", size)
            self.results.append(result)
    
    def run_all_benchmarks(self):
        """Run the complete benchmark suite"""
        print(f"Running PyTorch benchmarks on device: {self.device}")
        print(f"Warmup iterations: {self.warmup_iterations}")
        print(f"Measurement iterations: {self.measurement_iterations}")
        print()
        
        benchmark_functions = [
            ("Basic Operations", self.bench_basic_operations),
            ("Matrix Operations", self.bench_matrix_operations),
            ("Activation Functions", self.bench_activation_functions),
            ("Neural Networks", self.bench_neural_networks),
            ("Convolution Operations", self.bench_convolution_operations),
            ("Reduction Operations", self.bench_reduction_operations),
            ("Advanced Math", self.bench_advanced_math),
            ("Higher-Order Derivatives", self.bench_higher_order_derivatives),
        ]
        
        for name, func in benchmark_functions:
            print(f"Running {name}...")
            try:
                func()
                print(f"✓ {name} completed")
            except Exception as e:
                print(f"✗ {name} failed: {e}")
            print()
        
        return self.results


def print_results_table(results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table"""
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                            PYTORCH BENCHMARK RESULTS                            ║")
    print("╠══════════════════════════════════════════════════════════════════════════════╣")
    print("║ Operation                | Input Size | Forward (ms) | Backward (ms) | Total (ms) ║")
    print("╠══════════════════════════════════════════════════════════════════════════════╣")
    
    for result in results:
        if result.total_time_ms != float('inf'):
            print(f"║ {result.operation:<24} | {result.input_size:>10} | "
                  f"{result.forward_time_ms:>11.3f} | {result.backward_time_ms:>12.3f} | "
                  f"{result.total_time_ms:>9.3f} ║")
    
    print("╚══════════════════════════════════════════════════════════════════════════════╝")


def main():
    parser = argparse.ArgumentParser(description="PyTorch benchmark for TenfloweRS comparison")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                      help="Device to run benchmarks on")
    parser.add_argument("--output-json", action="store_true",
                      help="Output results in JSON format")
    parser.add_argument("--verbose", action="store_true",
                      help="Verbose output")
    parser.add_argument("--warmup", type=int, default=10,
                      help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=100,
                      help="Number of measurement iterations")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Create benchmark suite
    suite = PyTorchBenchmarkSuite(
        device=args.device,
        warmup_iterations=args.warmup,
        measurement_iterations=args.iterations
    )
    
    # Run benchmarks
    results = suite.run_all_benchmarks()
    
    # Output results
    if args.output_json:
        # Output JSON for machine consumption
        json_results = [result.to_dict() for result in results]
        print(json.dumps(json_results, indent=2))
    else:
        # Output formatted table for human consumption
        print_results_table(results)
        
        if args.verbose:
            print(f"\nTotal benchmarks run: {len(results)}")
            successful = len([r for r in results if r.total_time_ms != float('inf')])
            print(f"Successful benchmarks: {successful}")
            print(f"Failed benchmarks: {len(results) - successful}")


if __name__ == "__main__":
    main()