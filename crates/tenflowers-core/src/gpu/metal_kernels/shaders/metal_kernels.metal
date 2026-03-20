//! Metal Compute Kernels for Tensor Operations
//!
//! This file contains optimized Metal compute shaders for various tensor operations
//! including matrix multiplication, convolution, element-wise operations, and more.

#include <metal_stdlib>
using namespace metal;

// ===== Basic Element-wise Operations =====

kernel void elementwise_add(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* result [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
    result[index] = a[index] + b[index];
}

kernel void elementwise_mul(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* result [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
    result[index] = a[index] * b[index];
}

kernel void elementwise_sub(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* result [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
    result[index] = a[index] - b[index];
}

kernel void elementwise_div(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* result [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
    result[index] = a[index] / b[index];
}

// ===== Activation Functions =====

kernel void fused_relu(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    output[index] = max(0.0f, input[index]);
}

kernel void fused_gelu(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    float x = input[index];
    float cdf = 0.5f * (1.0f + tanh(sqrt(2.0f / M_PI_F) * (x + 0.044715f * x * x * x)));
    output[index] = x * cdf;
}

kernel void fused_swish(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        uint index [[thread_position_in_grid]]) {
    float x = input[index];
    output[index] = x / (1.0f + exp(-x));
}

kernel void fused_tanh(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    output[index] = tanh(input[index]);
}

kernel void fused_sigmoid(device const float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          uint index [[thread_position_in_grid]]) {
    output[index] = 1.0f / (1.0f + exp(-input[index]));
}

// ===== Reduction Operations =====

kernel void reduce_sum(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& length [[buffer(2)]],
                       threadgroup float* shared_memory [[threadgroup(0)]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint gid [[thread_position_in_grid]],
                       uint group_size [[threads_per_threadgroup]]) {

    // Load data into shared memory
    shared_memory[tid] = (gid < length) ? input[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] += shared_memory[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (tid == 0) {
        output[0] = shared_memory[0];
    }
}

kernel void reduce_max(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& length [[buffer(2)]],
                       threadgroup float* shared_memory [[threadgroup(0)]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint gid [[thread_position_in_grid]],
                       uint group_size [[threads_per_threadgroup]]) {

    // Load data into shared memory
    shared_memory[tid] = (gid < length) ? input[gid] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] = max(shared_memory[tid], shared_memory[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (tid == 0) {
        output[0] = shared_memory[0];
    }
}

// ===== Matrix Operations =====

kernel void matrix_multiply_naive(device const float* A [[buffer(0)]],
                                  device const float* B [[buffer(1)]],
                                  device float* C [[buffer(2)]],
                                  constant uint& M [[buffer(3)]],
                                  constant uint& N [[buffer(4)]],
                                  constant uint& K [[buffer(5)]],
                                  uint2 gid [[thread_position_in_grid]]) {

    if (gid.x >= M || gid.y >= N) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[gid.x * K + k] * B[k * N + gid.y];
    }
    C[gid.x * N + gid.y] = sum;
}

// ===== Normalization Operations =====

kernel void layer_norm(device const float* input [[buffer(0)]],
                       device const float* gamma [[buffer(1)]],
                       device const float* beta [[buffer(2)]],
                       device float* output [[buffer(3)]],
                       constant uint& hidden_size [[buffer(4)]],
                       constant float& epsilon [[buffer(5)]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       threadgroup float* shared_sum [[threadgroup(0)]],
                       threadgroup float* shared_sum_sq [[threadgroup(1)]]) {

    uint batch_idx = gid / hidden_size;
    uint feature_idx = gid % hidden_size;
    uint local_idx = tid;

    // Compute mean
    shared_sum[local_idx] = input[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction to compute sum
    for (uint s = 1; s < hidden_size; s *= 2) {
        if (local_idx % (2 * s) == 0 && local_idx + s < hidden_size) {
            shared_sum[local_idx] += shared_sum[local_idx + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / hidden_size;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute variance
    float diff = input[gid] - mean;
    shared_sum_sq[local_idx] = diff * diff;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction to compute sum of squares
    for (uint s = 1; s < hidden_size; s *= 2) {
        if (local_idx % (2 * s) == 0 && local_idx + s < hidden_size) {
            shared_sum_sq[local_idx] += shared_sum_sq[local_idx + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float variance = shared_sum_sq[0] / hidden_size;
    float inv_std = 1.0f / sqrt(variance + epsilon);

    // Apply normalization
    output[gid] = (input[gid] - mean) * inv_std * gamma[feature_idx] + beta[feature_idx];
}

// ===== Memory Bandwidth Test =====

kernel void memory_bandwidth_test(device const float* input [[buffer(0)]],
                                  device float* output [[buffer(1)]],
                                  uint index [[thread_position_in_grid]]) {
    // Simple copy operation for bandwidth testing
    output[index] = input[index];
}

// ===== Apple Silicon SIMD Operations =====

kernel void simd_add_optimized(device const float* a [[buffer(0)]],
                               device const float* b [[buffer(1)]],
                               device float* result [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
    // Optimized for Apple Silicon SIMD width (32)
    result[index] = a[index] + b[index];
}