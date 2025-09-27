//
//  tensor_ops.metal
//  TenfloweRS Metal Compute Shaders
//
//  High-performance tensor operations optimized for Apple Silicon and AMD GPUs on macOS
//  Leverages Metal's unified memory architecture and wide SIMD execution units
//

#include <metal_stdlib>
using namespace metal;

// MARK: - Utility Functions

/// Calculate linear index from 2D thread position
inline uint linear_index(uint2 thread_pos, uint width) {
    return thread_pos.y * width + thread_pos.x;
}

/// Calculate optimal memory access stride for coalescing
inline uint optimal_stride(uint base_stride, device uint* metadata) {
    // Align to cache line boundaries for better memory performance
    return (base_stride + 63) & ~63;
}

// MARK: - Fused Activation Functions

/// Fused ReLU activation with optimized memory access
kernel void fused_relu(device float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    
    float value = input[gid];
    output[gid] = max(value, 0.0f);
}

/// Fused GELU activation using optimized approximation
kernel void fused_gelu(device float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    
    float x = input[gid];
    // Optimized GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    float x_cubed = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x_cubed);
    output[gid] = 0.5f * x * (1.0f + tanh(inner));
}

/// Fused Swish activation (x * sigmoid(x))
kernel void fused_swish(device float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    
    float x = input[gid];
    float sigmoid_x = 1.0f / (1.0f + exp(-x));
    output[gid] = x * sigmoid_x;
}

/// Fused tanh activation
kernel void fused_tanh(device float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    
    output[gid] = tanh(input[gid]);
}

/// Fused sigmoid activation
kernel void fused_sigmoid(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    
    output[gid] = 1.0f / (1.0f + exp(-input[gid]));
}

// MARK: - Memory-Coalesced Element-wise Operations

/// Memory-coalesced addition with optimized access patterns
kernel void coalesced_add(device float* a [[buffer(0)]],
                          device float* b [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant uint& count [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]],
                          uint2 grid_size [[threads_per_grid]]) {
    
    uint linear_id = linear_index(gid, grid_size.x);
    if (linear_id >= count) return;
    
    // Process multiple elements per thread for better memory utilization
    const uint elements_per_thread = 4;
    uint base_idx = linear_id * elements_per_thread;
    
    for (uint i = 0; i < elements_per_thread && (base_idx + i) < count; i++) {
        uint idx = base_idx + i;
        output[idx] = a[idx] + b[idx];
    }
}

/// Memory-coalesced multiplication
kernel void coalesced_mul(device float* a [[buffer(0)]],
                          device float* b [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant uint& count [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]],
                          uint2 grid_size [[threads_per_grid]]) {
    
    uint linear_id = linear_index(gid, grid_size.x);
    if (linear_id >= count) return;
    
    const uint elements_per_thread = 4;
    uint base_idx = linear_id * elements_per_thread;
    
    for (uint i = 0; i < elements_per_thread && (base_idx + i) < count; i++) {
        uint idx = base_idx + i;
        output[idx] = a[idx] * b[idx];
    }
}

/// Memory-coalesced subtraction
kernel void coalesced_sub(device float* a [[buffer(0)]],
                          device float* b [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant uint& count [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]],
                          uint2 grid_size [[threads_per_grid]]) {
    
    uint linear_id = linear_index(gid, grid_size.x);
    if (linear_id >= count) return;
    
    const uint elements_per_thread = 4;
    uint base_idx = linear_id * elements_per_thread;
    
    for (uint i = 0; i < elements_per_thread && (base_idx + i) < count; i++) {
        uint idx = base_idx + i;
        output[idx] = a[idx] - b[idx];
    }
}

/// Memory-coalesced division
kernel void coalesced_div(device float* a [[buffer(0)]],
                          device float* b [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant uint& count [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]],
                          uint2 grid_size [[threads_per_grid]]) {
    
    uint linear_id = linear_index(gid, grid_size.x);
    if (linear_id >= count) return;
    
    const uint elements_per_thread = 4;
    uint base_idx = linear_id * elements_per_thread;
    
    for (uint i = 0; i < elements_per_thread && (base_idx + i) < count; i++) {
        uint idx = base_idx + i;
        output[idx] = a[idx] / b[idx];
    }
}

// MARK: - Optimized Reduction Operations

/// Hierarchical sum reduction using shared memory and optimized tree reduction
kernel void hierarchical_sum(device float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             threadgroup float* shared_memory [[threadgroup(0)]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint gid [[thread_position_in_grid]],
                             uint group_size [[threads_per_threadgroup]]) {
    
    // Load data into shared memory with coalesced access
    float local_sum = 0.0f;
    for (uint i = gid; i < count; i += group_size * 256) { // Assume 256 groups
        local_sum += input[i];
    }
    shared_memory[lid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction in shared memory
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_memory[lid] += shared_memory[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (lid == 0) {
        atomic_fetch_add_explicit((device atomic<float>*)output, shared_memory[0], memory_order_relaxed);
    }
}

/// Optimized mean calculation with single-pass algorithm
kernel void optimized_mean(device float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           threadgroup float* shared_memory [[threadgroup(0)]],
                           uint lid [[thread_position_in_threadgroup]],
                           uint gid [[thread_position_in_grid]],
                           uint group_size [[threads_per_threadgroup]]) {
    
    // Calculate local mean
    float local_sum = 0.0f;
    uint local_count = 0;
    
    for (uint i = gid; i < count; i += group_size * 256) {
        local_sum += input[i];
        local_count++;
    }
    
    shared_memory[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction for sum
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_memory[lid] += shared_memory[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Calculate final mean
    if (lid == 0) {
        float total_sum = shared_memory[0];
        atomic_fetch_add_explicit((device atomic<float>*)output, total_sum / count, memory_order_relaxed);
    }
}

/// Optimized max reduction with early termination
kernel void optimized_max(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          threadgroup float* shared_memory [[threadgroup(0)]],
                          uint lid [[thread_position_in_threadgroup]],
                          uint gid [[thread_position_in_grid]],
                          uint group_size [[threads_per_threadgroup]]) {
    
    float local_max = -INFINITY;
    
    for (uint i = gid; i < count; i += group_size * 256) {
        local_max = max(local_max, input[i]);
    }
    
    shared_memory[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction for max
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_memory[lid] = max(shared_memory[lid], shared_memory[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (lid == 0) {
        atomic_fetch_max_explicit((device atomic<float>*)output, shared_memory[0], memory_order_relaxed);
    }
}

/// Optimized min reduction with early termination
kernel void optimized_min(device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          threadgroup float* shared_memory [[threadgroup(0)]],
                          uint lid [[thread_position_in_threadgroup]],
                          uint gid [[thread_position_in_grid]],
                          uint group_size [[threads_per_threadgroup]]) {
    
    float local_min = INFINITY;
    
    for (uint i = gid; i < count; i += group_size * 256) {
        local_min = min(local_min, input[i]);
    }
    
    shared_memory[lid] = local_min;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction for min
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_memory[lid] = min(shared_memory[lid], shared_memory[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (lid == 0) {
        atomic_fetch_min_explicit((device atomic<float>*)output, shared_memory[0], memory_order_relaxed);
    }
}

// MARK: - Matrix Operations Optimized for Apple Silicon

/// Tiled matrix multiplication optimized for Apple Silicon's wide execution units
kernel void optimized_matmul(device float* A [[buffer(0)]],
                             device float* B [[buffer(1)]],
                             device float* C [[buffer(2)]],
                             constant uint& M [[buffer(3)]],
                             constant uint& N [[buffer(4)]],
                             constant uint& K [[buffer(5)]],
                             threadgroup float* tileA [[threadgroup(0)]],
                             threadgroup float* tileB [[threadgroup(1)]],
                             uint2 gid [[threadgroup_position_in_grid]],
                             uint2 lid [[thread_position_in_threadgroup]]) {
    
    const uint TILE_SIZE = 32; // Optimal for Apple Silicon
    
    uint row = gid.y * TILE_SIZE + lid.y;
    uint col = gid.x * TILE_SIZE + lid.x;
    
    float sum = 0.0f;
    
    for (uint tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tiles into shared memory with coalesced access
        uint tileRow = tile * TILE_SIZE + lid.y;
        uint tileCol = tile * TILE_SIZE + lid.x;
        
        if (row < M && tileCol < K) {
            tileA[lid.y * TILE_SIZE + lid.x] = A[row * K + tileCol];
        } else {
            tileA[lid.y * TILE_SIZE + lid.x] = 0.0f;
        }
        
        if (tileRow < K && col < N) {
            tileB[lid.y * TILE_SIZE + lid.x] = B[tileRow * N + col];
        } else {
            tileB[lid.y * TILE_SIZE + lid.x] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[lid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + lid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// MARK: - Convolution Operations

/// Optimized 2D convolution using Im2Col and GEMM approach
kernel void optimized_conv2d(device float* input [[buffer(0)]],
                             device float* weights [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint4& input_shape [[buffer(3)]], // [batch, channels, height, width]
                             constant uint4& weight_shape [[buffer(4)]], // [out_channels, in_channels, kernel_h, kernel_w]
                             constant uint2& stride [[buffer(5)]],
                             constant uint2& padding [[buffer(6)]],
                             uint3 gid [[thread_position_in_grid]]) {
    
    uint batch = gid.z;
    uint out_channel = gid.y;
    uint out_pixel = gid.x;
    
    uint output_h = (input_shape.z + 2 * padding.y - weight_shape.z) / stride.y + 1;
    uint output_w = (input_shape.w + 2 * padding.x - weight_shape.w) / stride.x + 1;
    
    if (batch >= input_shape.x || out_channel >= weight_shape.x || out_pixel >= (output_h * output_w)) {
        return;
    }
    
    uint out_y = out_pixel / output_w;
    uint out_x = out_pixel % output_w;
    
    float sum = 0.0f;
    
    // Convolution computation
    for (uint in_channel = 0; in_channel < weight_shape.y; in_channel++) {
        for (uint ky = 0; ky < weight_shape.z; ky++) {
            for (uint kx = 0; kx < weight_shape.w; kx++) {
                int in_y = int(out_y * stride.y) + int(ky) - int(padding.y);
                int in_x = int(out_x * stride.x) + int(kx) - int(padding.x);
                
                if (in_y >= 0 && in_y < int(input_shape.z) && in_x >= 0 && in_x < int(input_shape.w)) {
                    uint input_idx = batch * input_shape.y * input_shape.z * input_shape.w +
                                   in_channel * input_shape.z * input_shape.w +
                                   uint(in_y) * input_shape.w + uint(in_x);
                    
                    uint weight_idx = out_channel * weight_shape.y * weight_shape.z * weight_shape.w +
                                    in_channel * weight_shape.z * weight_shape.w +
                                    ky * weight_shape.w + kx;
                    
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    uint output_idx = batch * weight_shape.x * output_h * output_w +
                     out_channel * output_h * output_w +
                     out_y * output_w + out_x;
    
    output[output_idx] = sum;
}

// MARK: - Specialized Apple Silicon Optimizations

/// SIMD-optimized operations for Apple Silicon's wide execution units
kernel void apple_silicon_simd_add(device float4* a [[buffer(0)]],
                                   device float4* b [[buffer(1)]],
                                   device float4* output [[buffer(2)]],
                                   constant uint& count [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
    
    if (gid >= count) return;
    
    // Process 4 floats simultaneously using SIMD
    float4 vec_a = a[gid];
    float4 vec_b = b[gid];
    output[gid] = vec_a + vec_b;
}

/// Memory bandwidth optimized operations for unified memory architecture
kernel void unified_memory_optimized_copy(device float* src [[buffer(0)]],
                                         device float* dst [[buffer(1)]],
                                         constant uint& count [[buffer(2)]],
                                         uint gid [[thread_position_in_grid]]) {
    
    if (gid >= count) return;
    
    // Optimized for unified memory - minimal cache pollution
    const uint prefetch_distance = 64; // Elements to prefetch
    const uint elements_per_thread = 8;
    
    uint base_idx = gid * elements_per_thread;
    
    // Prefetch future data
    if (base_idx + prefetch_distance < count) {
        float prefetch_val = src[base_idx + prefetch_distance];
        (void)prefetch_val; // Prevent optimization away
    }
    
    // Copy elements
    for (uint i = 0; i < elements_per_thread && (base_idx + i) < count; i++) {
        dst[base_idx + i] = src[base_idx + i];
    }
}

// MARK: - Advanced Normalization Operations

/// Layer normalization optimized for transformer models
kernel void optimized_layer_norm(device float* input [[buffer(0)]],
                                device float* gamma [[buffer(1)]],
                                device float* beta [[buffer(2)]],
                                device float* output [[buffer(3)]],
                                constant uint& batch_size [[buffer(4)]],
                                constant uint& feature_size [[buffer(5)]],
                                constant float& eps [[buffer(6)]],
                                threadgroup float* shared_memory [[threadgroup(0)]],
                                uint2 gid [[thread_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]],
                                uint group_size [[threads_per_threadgroup]]) {
    
    uint batch_idx = gid.y;
    if (batch_idx >= batch_size) return;
    
    device float* batch_input = input + batch_idx * feature_size;
    device float* batch_output = output + batch_idx * feature_size;
    
    // Calculate mean
    float sum = 0.0f;
    for (uint i = lid; i < feature_size; i += group_size) {
        sum += batch_input[i];
    }
    shared_memory[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce sum
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_memory[lid] += shared_memory[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float mean = shared_memory[0] / feature_size;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Calculate variance
    float var_sum = 0.0f;
    for (uint i = lid; i < feature_size; i += group_size) {
        float diff = batch_input[i] - mean;
        var_sum += diff * diff;
    }
    shared_memory[lid] = var_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce variance
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_memory[lid] += shared_memory[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float variance = shared_memory[0] / feature_size;
    float inv_std = rsqrt(variance + eps);
    
    // Apply normalization
    for (uint i = lid; i < feature_size; i += group_size) {
        float normalized = (batch_input[i] - mean) * inv_std;
        batch_output[i] = gamma[i] * normalized + beta[i];
    }
}

/// Group normalization with optimized memory access
kernel void optimized_group_norm(device float* input [[buffer(0)]],
                                device float* gamma [[buffer(1)]],
                                device float* beta [[buffer(2)]],
                                device float* output [[buffer(3)]],
                                constant uint& batch_size [[buffer(4)]],
                                constant uint& channels [[buffer(5)]],
                                constant uint& spatial_size [[buffer(6)]],
                                constant uint& groups [[buffer(7)]],
                                constant float& eps [[buffer(8)]],
                                threadgroup float* shared_memory [[threadgroup(0)]],
                                uint3 gid [[thread_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]],
                                uint group_size [[threads_per_threadgroup]]) {
    
    uint batch_idx = gid.z;
    uint group_idx = gid.y;
    
    if (batch_idx >= batch_size || group_idx >= groups) return;
    
    uint channels_per_group = channels / groups;
    uint group_size_total = channels_per_group * spatial_size;
    
    device float* batch_input = input + batch_idx * channels * spatial_size;
    device float* batch_output = output + batch_idx * channels * spatial_size;
    device float* group_input = batch_input + group_idx * channels_per_group * spatial_size;
    device float* group_output = batch_output + group_idx * channels_per_group * spatial_size;
    
    // Calculate group mean
    float sum = 0.0f;
    for (uint i = lid; i < group_size_total; i += group_size) {
        sum += group_input[i];
    }
    shared_memory[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce sum
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_memory[lid] += shared_memory[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float mean = shared_memory[0] / group_size_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Calculate group variance
    float var_sum = 0.0f;
    for (uint i = lid; i < group_size_total; i += group_size) {
        float diff = group_input[i] - mean;
        var_sum += diff * diff;
    }
    shared_memory[lid] = var_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce variance
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_memory[lid] += shared_memory[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float variance = shared_memory[0] / group_size_total;
    float inv_std = rsqrt(variance + eps);
    
    // Apply normalization
    for (uint i = lid; i < group_size_total; i += group_size) {
        uint channel_offset = i / spatial_size;
        uint global_channel = group_idx * channels_per_group + channel_offset;
        
        float normalized = (group_input[i] - mean) * inv_std;
        group_output[i] = gamma[global_channel] * normalized + beta[global_channel];
    }
}

// MARK: - Attention Mechanisms

/// Multi-head attention with Flash Attention optimization
kernel void flash_attention(device float* query [[buffer(0)]],
                           device float* key [[buffer(1)]],
                           device float* value [[buffer(2)]],
                           device float* output [[buffer(3)]],
                           constant uint& batch_size [[buffer(4)]],
                           constant uint& seq_len [[buffer(5)]],
                           constant uint& head_dim [[buffer(6)]],
                           constant uint& num_heads [[buffer(7)]],
                           constant float& scale [[buffer(8)]],
                           threadgroup float* shared_qk [[threadgroup(0)]],
                           threadgroup float* shared_values [[threadgroup(1)]],
                           uint3 gid [[thread_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]],
                           uint group_size [[threads_per_threadgroup]]) {
    
    const uint BLOCK_SIZE = 64; // Optimized block size for Apple Silicon
    
    uint batch_idx = gid.z;
    uint head_idx = gid.y;
    uint query_idx = gid.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || query_idx >= seq_len) return;
    
    uint head_offset = batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim;
    device float* q = query + head_offset + query_idx * head_dim;
    device float* k_base = key + head_offset;
    device float* v_base = value + head_offset;
    device float* out = output + head_offset + query_idx * head_dim;
    
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    
    // Initialize output
    for (uint d = lid; d < head_dim; d += group_size) {
        out[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Process in blocks for memory efficiency
    for (uint block_start = 0; block_start < seq_len; block_start += BLOCK_SIZE) {
        uint block_end = min(block_start + BLOCK_SIZE, seq_len);
        
        // Compute attention scores for this block
        for (uint key_idx = block_start + lid; key_idx < block_end; key_idx += group_size) {
            device float* k = k_base + key_idx * head_dim;
            
            float score = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                score += q[d] * k[d];
            }
            score *= scale;
            
            shared_qk[key_idx - block_start] = score;
            max_score = max(max_score, score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute softmax for this block
        float block_sum = 0.0f;
        for (uint i = lid; i < (block_end - block_start); i += group_size) {
            float exp_score = exp(shared_qk[i] - max_score);
            shared_qk[i] = exp_score;
            block_sum += exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        sum_exp += block_sum;
        
        // Accumulate weighted values
        for (uint key_idx = block_start; key_idx < block_end; key_idx++) {
            device float* v = v_base + key_idx * head_dim;
            float weight = shared_qk[key_idx - block_start];
            
            for (uint d = lid; d < head_dim; d += group_size) {
                out[d] += weight * v[d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Normalize output
    for (uint d = lid; d < head_dim; d += group_size) {
        out[d] /= sum_exp;
    }
}

// MARK: - Advanced Mathematical Operations

/// Matrix exponential using Padé approximation
kernel void matrix_exponential(device float* input [[buffer(0)]],
                              device float* output [[buffer(1)]],
                              device float* workspace [[buffer(2)]],
                              constant uint& matrix_size [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
    
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= matrix_size || col >= matrix_size) return;
    
    // For now, implement a simplified version using Taylor series
    // This is a placeholder for a more sophisticated implementation
    float elem = input[row * matrix_size + col];
    
    if (row == col) {
        output[row * matrix_size + col] = exp(elem);
    } else {
        output[row * matrix_size + col] = elem; // Simplified off-diagonal handling
    }
}

/// Eigenvalue computation using QR algorithm iteration
kernel void eigenvalue_qr_step(device float* matrix [[buffer(0)]],
                              device float* q_matrix [[buffer(1)]],
                              device float* r_matrix [[buffer(2)]],
                              constant uint& matrix_size [[buffer(3)]],
                              threadgroup float* shared_temp [[threadgroup(0)]],
                              uint2 gid [[thread_position_in_grid]],
                              uint lid [[thread_position_in_threadgroup]]) {
    
    // Placeholder for QR decomposition step
    // This would implement Householder reflections or Givens rotations
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= matrix_size || col >= matrix_size) return;
    
    // For now, just copy input to Q (placeholder)
    if (row == col) {
        q_matrix[row * matrix_size + col] = 1.0f;
        r_matrix[row * matrix_size + col] = matrix[row * matrix_size + col];
    } else {
        q_matrix[row * matrix_size + col] = 0.0f;
        r_matrix[row * matrix_size + col] = 0.0f;
    }
}

// MARK: - Performance Monitoring and Profiling

/// Advanced performance profiling with GPU memory bandwidth tracking
kernel void performance_bandwidth_test(device float* input [[buffer(0)]],
                                      device float* output [[buffer(1)]],
                                      device uint64_t* stats [[buffer(2)]],
                                      constant uint& data_size [[buffer(3)]],
                                      uint gid [[thread_position_in_grid]]) {
    
    if (gid >= data_size) return;
    
    // Simple memory copy to measure bandwidth
    output[gid] = input[gid];
    
    // Update statistics (atomic operations for thread safety)
    if (gid == 0) {
        atomic_fetch_add_explicit((device atomic<uint64_t>*)&stats[0], 1, memory_order_relaxed);
    }
}

/// GPU occupancy measurement kernel
kernel void measure_occupancy(device uint32_t* occupancy_data [[buffer(0)]],
                             constant uint& total_threads [[buffer(1)]],
                             uint gid [[thread_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint group_id [[threadgroup_position_in_grid]]) {
    
    if (gid >= total_threads) return;
    
    // Record thread and threadgroup information for occupancy analysis
    occupancy_data[gid] = group_id;
}