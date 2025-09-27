#!/usr/bin/env python3
"""
PyTorch Performance Comparison Tool

This script benchmarks PyTorch operations to compare with TenfloweRS performance.
It generates results that can be compared with the TenfloweRS benchmark output.
"""

import torch
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Any
import argparse
import platform
import sys

def time_operation(operation, warmup_iterations=10, measurement_iterations=50):
    """Time a PyTorch operation with proper warmup."""
    # Warmup
    for _ in range(warmup_iterations):
        operation()
    
    # Synchronize if using CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(measurement_iterations):
        start = time.perf_counter()
        result = operation()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'iterations': measurement_iterations
    }

def benchmark_binary_operations(device: str = 'cpu'):
    """Benchmark binary operations like Add, Mul, etc."""
    results = {}
    device_obj = torch.device(device)
    
    operations = ['add', 'mul', 'sub', 'div']
    shapes = [
        (1000,),           # Small vector
        (100, 100),        # Small matrix
        (1000, 1000),      # Large matrix
    ]
    
    for op_name in operations:
        results[op_name] = {}
        
        for shape in shapes:
            shape_str = 'x'.join(map(str, shape))
            
            # Create tensors
            a = torch.randn(shape, device=device_obj, dtype=torch.float32)
            b = torch.randn(shape, device=device_obj, dtype=torch.float32)
            
            # Define operation
            if op_name == 'add':
                operation = lambda: torch.add(a, b)
            elif op_name == 'mul':
                operation = lambda: torch.mul(a, b)
            elif op_name == 'sub':
                operation = lambda: torch.sub(a, b)
            elif op_name == 'div':
                operation = lambda: torch.div(a, b)
            
            # Benchmark
            timing_result = time_operation(operation)
            
            # Calculate throughput
            total_elements = np.prod(shape) * 2  # Two input tensors
            throughput = total_elements / timing_result['mean_time']
            
            results[op_name][shape_str] = {
                **timing_result,
                'shape': shape,
                'throughput_elem_per_sec': throughput,
                'device': device
            }
    
    return results

def benchmark_unary_operations(device: str = 'cpu'):
    """Benchmark unary operations like ReLU, Sigmoid, etc."""
    results = {}
    device_obj = torch.device(device)
    
    operations = ['relu', 'sigmoid', 'tanh']
    shapes = [
        (10000,),          # Large vector
        (100, 100),        # Square matrix
        (32, 32, 32),      # 3D tensor
    ]
    
    for op_name in operations:
        results[op_name] = {}
        
        for shape in shapes:
            shape_str = 'x'.join(map(str, shape))
            
            # Create tensor
            x = torch.randn(shape, device=device_obj, dtype=torch.float32)
            
            # Define operation
            if op_name == 'relu':
                operation = lambda: torch.relu(x)
            elif op_name == 'sigmoid':
                operation = lambda: torch.sigmoid(x)
            elif op_name == 'tanh':
                operation = lambda: torch.tanh(x)
            
            # Benchmark
            timing_result = time_operation(operation)
            
            # Calculate throughput
            total_elements = np.prod(shape)
            throughput = total_elements / timing_result['mean_time']
            
            results[op_name][shape_str] = {
                **timing_result,
                'shape': shape,
                'throughput_elem_per_sec': throughput,
                'device': device
            }
    
    return results

def benchmark_matmul_operations(device: str = 'cpu'):
    """Benchmark matrix multiplication operations."""
    results = {}
    device_obj = torch.device(device)
    
    sizes = [
        (64, 64, 64),      # Small
        (128, 128, 128),   # Medium
        (256, 256, 256),   # Large
        (512, 512, 512),   # Very large
        (1024, 1024, 1024), # Extra large
    ]
    
    for m, k, n in sizes:
        size_str = f"{m}x{k}x{n}"
        
        # Create tensors
        a = torch.randn(m, k, device=device_obj, dtype=torch.float32)
        b = torch.randn(k, n, device=device_obj, dtype=torch.float32)
        
        # Define operation
        operation = lambda: torch.matmul(a, b)
        
        # Benchmark
        timing_result = time_operation(operation)
        
        # Calculate FLOPS
        flops = 2 * m * k * n  # Multiply-add operations
        flops_per_sec = flops / timing_result['mean_time']
        
        results[size_str] = {
            **timing_result,
            'size': (m, k, n),
            'flops': flops,
            'flops_per_sec': flops_per_sec,
            'device': device
        }
    
    return results

def get_system_info():
    """Get system information for the benchmark report."""
    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'cpu_count': torch.get_num_threads(),
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
    
    return info

def generate_report(results: Dict[str, Any], output_file: str = None):
    """Generate a human-readable benchmark report."""
    report = []
    report.append("# PyTorch Performance Benchmark Report")
    report.append("=" * 50)
    report.append("")
    
    # System info
    system_info = get_system_info()
    report.append("## System Information")
    report.append(f"Platform: {system_info['platform']}")
    report.append(f"PyTorch Version: {system_info['pytorch_version']}")
    report.append(f"CPU Threads: {system_info['cpu_count']}")
    report.append(f"CUDA Available: {system_info['cuda_available']}")
    if system_info['cuda_available']:
        report.append(f"CUDA Device: {system_info['cuda_device_name']}")
    report.append("")
    
    # Binary operations
    if 'binary_ops' in results:
        report.append("## Binary Operations")
        report.append("")
        for op_name, op_results in results['binary_ops'].items():
            report.append(f"### {op_name.upper()}")
            report.append("| Shape | Device | Duration (μs) | Throughput (elem/s) |")
            report.append("|-------|--------|---------------|-------------------|")
            
            for shape_str, result in op_results.items():
                duration_us = result['mean_time'] * 1_000_000
                throughput = f"{result['throughput_elem_per_sec']:.2e}"
                report.append(f"| {shape_str} | {result['device']} | {duration_us:.1f} | {throughput} |")
            report.append("")
    
    # Unary operations
    if 'unary_ops' in results:
        report.append("## Unary Operations")
        report.append("")
        for op_name, op_results in results['unary_ops'].items():
            report.append(f"### {op_name.upper()}")
            report.append("| Shape | Device | Duration (μs) | Throughput (elem/s) |")
            report.append("|-------|--------|---------------|-------------------|")
            
            for shape_str, result in op_results.items():
                duration_us = result['mean_time'] * 1_000_000
                throughput = f"{result['throughput_elem_per_sec']:.2e}"
                report.append(f"| {shape_str} | {result['device']} | {duration_us:.1f} | {throughput} |")
            report.append("")
    
    # Matrix multiplication
    if 'matmul' in results:
        report.append("## Matrix Multiplication")
        report.append("| Size (M×K×N) | Device | Duration (μs) | FLOPS/sec |")
        report.append("|--------------|--------|---------------|-----------|")
        
        for size_str, result in results['matmul'].items():
            duration_us = result['mean_time'] * 1_000_000
            flops_per_sec = f"{result['flops_per_sec']:.2e}"
            report.append(f"| {size_str} | {result['device']} | {duration_us:.1f} | {flops_per_sec} |")
        report.append("")
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {output_file}")
    else:
        print(report_text)
    
    return report_text

def main():
    parser = argparse.ArgumentParser(description='PyTorch performance benchmark tool')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device to run benchmarks on')
    parser.add_argument('--output-json', type=str, 
                       help='Output file for JSON results')
    parser.add_argument('--output-report', type=str,
                       help='Output file for human-readable report')
    parser.add_argument('--binary-ops', action='store_true',
                       help='Run binary operation benchmarks')
    parser.add_argument('--unary-ops', action='store_true',
                       help='Run unary operation benchmarks')
    parser.add_argument('--matmul', action='store_true',
                       help='Run matrix multiplication benchmarks')
    parser.add_argument('--all', action='store_true',
                       help='Run all benchmarks')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    if not any([args.binary_ops, args.unary_ops, args.matmul, args.all]):
        args.all = True
    
    print(f"🔥 PyTorch Performance Benchmark")
    print(f"Device: {args.device}")
    print(f"PyTorch Version: {torch.__version__}")
    print("=" * 50)
    
    results = {'system_info': get_system_info()}
    
    if args.all or args.binary_ops:
        print("Running binary operations benchmark...")
        results['binary_ops'] = benchmark_binary_operations(args.device)
    
    if args.all or args.unary_ops:
        print("Running unary operations benchmark...")
        results['unary_ops'] = benchmark_unary_operations(args.device)
    
    if args.all or args.matmul:
        print("Running matrix multiplication benchmark...")
        results['matmul'] = benchmark_matmul_operations(args.device)
    
    # Save JSON results
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"JSON results saved to {args.output_json}")
    
    # Generate report
    generate_report(results, args.output_report)
    
    print("\n🎉 Benchmark completed!")
    print("\n💡 Comparison Tips:")
    print("   - Compare operations with similar shapes and data types")
    print("   - GPU operations show their advantage with larger tensors")
    print("   - Consider memory bandwidth vs compute bound operations")
    print("   - Framework overhead can affect small tensor operations")

if __name__ == '__main__':
    main()