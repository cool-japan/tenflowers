/*
 * activation_kernels.hip
 * ROCm/HIP kernels for fused activation functions
 * Optimized for AMD RDNA and CDNA architectures
 */

#include <hip/hip_runtime.h>
#include <hip/hip_math_constants.h>

// Wavefront size on AMD GPUs (64 threads)
#define WAVEFRONT_SIZE 64

// Fused ReLU activation optimized for AMD wavefronts
__global__ void rocm_fused_relu(float* input, float* output, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process multiple elements per thread for better memory utilization
    const int elements_per_thread = 4;
    int base_idx = gid * elements_per_thread;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = base_idx + i;
        if (idx < count) {
            float value = input[idx];
            output[idx] = fmaxf(value, 0.0f);
        }
    }
}

// Fused GELU activation using optimized approximation for AMD GPUs
__global__ void rocm_fused_gelu(float* input, float* output, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int elements_per_thread = 4;
    int base_idx = gid * elements_per_thread;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = base_idx + i;
        if (idx < count) {
            float x = input[idx];
            
            // Optimized GELU approximation for AMD hardware
            float x_cubed = x * x * x;
            float inner = 0.7978845608f * (x + 0.044715f * x_cubed);
            output[idx] = 0.5f * x * (1.0f + tanhf(inner));
        }
    }
}

// Fused Swish/SiLU activation (x * sigmoid(x))
__global__ void rocm_fused_swish(float* input, float* output, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int elements_per_thread = 4;
    int base_idx = gid * elements_per_thread;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = base_idx + i;
        if (idx < count) {
            float x = input[idx];
            float sigmoid_x = 1.0f / (1.0f + expf(-x));
            output[idx] = x * sigmoid_x;
        }
    }
}

// Fused Mish activation (x * tanh(softplus(x)))
__global__ void rocm_fused_mish(float* input, float* output, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int elements_per_thread = 4;
    int base_idx = gid * elements_per_thread;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = base_idx + i;
        if (idx < count) {
            float x = input[idx];
            // Mish: x * tanh(ln(1 + exp(x)))
            float softplus = logf(1.0f + expf(x));
            output[idx] = x * tanhf(softplus);
        }
    }
}

// Advanced Layer Normalization optimized for AMD RDNA/CDNA
__global__ void rocm_layer_norm(
    float* input,
    float* gamma,
    float* beta,
    float* output,
    int batch_size,
    int feature_size,
    float eps
) {
    extern __shared__ float shared_memory[];
    
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    if (batch_idx >= batch_size) return;
    
    float* batch_input = input + batch_idx * feature_size;
    float* batch_output = output + batch_idx * feature_size;
    
    // Calculate mean using wavefront-level reduction
    float sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        sum += batch_input[i];
    }
    
    // Wavefront reduction for sum
    #pragma unroll
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    
    // Write wavefront sums to shared memory
    if (lane_id == 0) {
        shared_memory[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction
    if (warp_id == 0) {
        float warp_sum = (tid < blockDim.x / WAVEFRONT_SIZE) ? shared_memory[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down(warp_sum, offset);
        }
        if (tid == 0) {
            shared_memory[0] = warp_sum / feature_size; // mean
        }
    }
    __syncthreads();
    
    float mean = shared_memory[0];
    
    // Calculate variance
    float var_sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float diff = batch_input[i] - mean;
        var_sum += diff * diff;
    }
    
    // Wavefront reduction for variance
    #pragma unroll
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        var_sum += __shfl_down(var_sum, offset);
    }
    
    if (lane_id == 0) {
        shared_memory[warp_id] = var_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float warp_var = (tid < blockDim.x / WAVEFRONT_SIZE) ? shared_memory[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
            warp_var += __shfl_down(warp_var, offset);
        }
        if (tid == 0) {
            shared_memory[0] = rsqrtf(warp_var / feature_size + eps); // inv_std
        }
    }
    __syncthreads();
    
    float inv_std = shared_memory[0];
    
    // Apply normalization
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float normalized = (batch_input[i] - mean) * inv_std;
        batch_output[i] = gamma[i] * normalized + beta[i];
    }
}

// Flash Attention optimized for AMD GPUs
__global__ void rocm_flash_attention(
    float* query,
    float* key,
    float* value,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    extern __shared__ float shared_qk[];
    float* shared_values = shared_qk + 128; // Assuming max block size of 128
    
    const int BLOCK_SIZE = 64; // Optimal for AMD wavefronts
    
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int query_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || query_idx >= seq_len) return;
    
    int head_offset = batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim;
    float* q = query + head_offset + query_idx * head_dim;
    float* k_base = key + head_offset;
    float* v_base = value + head_offset;
    float* out = output + head_offset + query_idx * head_dim;
    
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    
    // Initialize output
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out[d] = 0.0f;
    }
    __syncthreads();
    
    // Process in blocks for memory efficiency
    for (int block_start = 0; block_start < seq_len; block_start += BLOCK_SIZE) {
        int block_end = min(block_start + BLOCK_SIZE, seq_len);
        
        // Compute attention scores
        for (int key_idx = block_start + tid; key_idx < block_end; key_idx += blockDim.x) {
            float* k = k_base + key_idx * head_dim;
            
            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < head_dim; d++) {
                score += q[d] * k[d];
            }
            score *= scale;
            
            shared_qk[key_idx - block_start] = score;
            max_score = fmaxf(max_score, score);
        }
        __syncthreads();
        
        // Wavefront reduction for max_score
        #pragma unroll
        for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
            max_score = fmaxf(max_score, __shfl_down(max_score, offset));
        }
        
        // Compute softmax
        float block_sum = 0.0f;
        for (int i = tid; i < (block_end - block_start); i += blockDim.x) {
            float exp_score = expf(shared_qk[i] - max_score);
            shared_qk[i] = exp_score;
            block_sum += exp_score;
        }
        __syncthreads();
        
        sum_exp += block_sum;
        
        // Accumulate weighted values
        for (int key_idx = block_start; key_idx < block_end; key_idx++) {
            float* v = v_base + key_idx * head_dim;
            float weight = shared_qk[key_idx - block_start];
            
            for (int d = tid; d < head_dim; d += blockDim.x) {
                out[d] += weight * v[d];
            }
        }
        __syncthreads();
    }
    
    // Normalize output
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out[d] /= sum_exp;
    }
}

// Fused tanh activation
__global__ void rocm_fused_tanh(float* input, float* output, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int elements_per_thread = 4;
    int base_idx = gid * elements_per_thread;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = base_idx + i;
        if (idx < count) {
            output[idx] = tanhf(input[idx]);
        }
    }
}

// Fused sigmoid activation
__global__ void rocm_fused_sigmoid(float* input, float* output, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int elements_per_thread = 4;
    int base_idx = gid * elements_per_thread;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = base_idx + i;
        if (idx < count) {
            output[idx] = 1.0f / (1.0f + expf(-input[idx]));
        }
    }
}