/*
 * elementwise_kernels.hip
 * ROCm/HIP kernels for memory-coalesced element-wise operations
 * Optimized for AMD GPU memory hierarchy and wavefront execution
 */

#include <hip/hip_runtime.h>

// Memory-coalesced addition optimized for AMD wavefronts
__global__ void rocm_coalesced_add(float* a, float* b, float* output, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process multiple elements per thread for better memory utilization
    const int elements_per_thread = 4;
    int base_idx = gid * elements_per_thread;
    
    // Use vectorized loads when possible (float4 for better memory bandwidth)
    if (base_idx + 4 <= count && (base_idx % 4) == 0) {
        float4 vec_a = reinterpret_cast<float4*>(a)[base_idx / 4];
        float4 vec_b = reinterpret_cast<float4*>(b)[base_idx / 4];
        
        float4 result;
        result.x = vec_a.x + vec_b.x;
        result.y = vec_a.y + vec_b.y;
        result.z = vec_a.z + vec_b.z;
        result.w = vec_a.w + vec_b.w;
        
        reinterpret_cast<float4*>(output)[base_idx / 4] = result;
    } else {
        // Fallback to scalar operations for boundary elements
        #pragma unroll
        for (int i = 0; i < elements_per_thread; i++) {
            int idx = base_idx + i;
            if (idx < count) {
                output[idx] = a[idx] + b[idx];
            }
        }
    }
}

// Memory-coalesced multiplication
__global__ void rocm_coalesced_mul(float* a, float* b, float* output, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int elements_per_thread = 4;
    int base_idx = gid * elements_per_thread;
    
    if (base_idx + 4 <= count && (base_idx % 4) == 0) {
        float4 vec_a = reinterpret_cast<float4*>(a)[base_idx / 4];
        float4 vec_b = reinterpret_cast<float4*>(b)[base_idx / 4];
        
        float4 result;
        result.x = vec_a.x * vec_b.x;
        result.y = vec_a.y * vec_b.y;
        result.z = vec_a.z * vec_b.z;
        result.w = vec_a.w * vec_b.w;
        
        reinterpret_cast<float4*>(output)[base_idx / 4] = result;
    } else {
        #pragma unroll
        for (int i = 0; i < elements_per_thread; i++) {
            int idx = base_idx + i;
            if (idx < count) {
                output[idx] = a[idx] * b[idx];
            }
        }
    }
}

// Memory-coalesced subtraction
__global__ void rocm_coalesced_sub(float* a, float* b, float* output, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int elements_per_thread = 4;
    int base_idx = gid * elements_per_thread;
    
    if (base_idx + 4 <= count && (base_idx % 4) == 0) {
        float4 vec_a = reinterpret_cast<float4*>(a)[base_idx / 4];
        float4 vec_b = reinterpret_cast<float4*>(b)[base_idx / 4];
        
        float4 result;
        result.x = vec_a.x - vec_b.x;
        result.y = vec_a.y - vec_b.y;
        result.z = vec_a.z - vec_b.z;
        result.w = vec_a.w - vec_b.w;
        
        reinterpret_cast<float4*>(output)[base_idx / 4] = result;
    } else {
        #pragma unroll
        for (int i = 0; i < elements_per_thread; i++) {
            int idx = base_idx + i;
            if (idx < count) {
                output[idx] = a[idx] - b[idx];
            }
        }
    }
}

// Memory-coalesced division
__global__ void rocm_coalesced_div(float* a, float* b, float* output, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int elements_per_thread = 4;
    int base_idx = gid * elements_per_thread;
    
    if (base_idx + 4 <= count && (base_idx % 4) == 0) {
        float4 vec_a = reinterpret_cast<float4*>(a)[base_idx / 4];
        float4 vec_b = reinterpret_cast<float4*>(b)[base_idx / 4];
        
        float4 result;
        result.x = vec_a.x / vec_b.x;
        result.y = vec_a.y / vec_b.y;
        result.z = vec_a.z / vec_b.z;
        result.w = vec_a.w / vec_b.w;
        
        reinterpret_cast<float4*>(output)[base_idx / 4] = result;
    } else {
        #pragma unroll
        for (int i = 0; i < elements_per_thread; i++) {
            int idx = base_idx + i;
            if (idx < count) {
                output[idx] = a[idx] / b[idx];
            }
        }
    }
}

// RDNA-optimized element-wise operations using wave intrinsics
__global__ void rocm_rdna_optimized_add(float* a, float* b, float* output, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int wavefront_id = gid / 64;  // RDNA wavefront size is 64
    int lane_id = gid % 64;
    
    // Use wave-level operations for better RDNA utilization
    if (gid < count) {
        float value_a = a[gid];
        float value_b = b[gid];
        float result = value_a + value_b;
        
        // Use wave shuffle for better memory access patterns
        // This is a simplified example - real implementation would use proper wave intrinsics
        output[gid] = result;
    }
}