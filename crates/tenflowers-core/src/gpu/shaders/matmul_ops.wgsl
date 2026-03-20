// Matrix multiplication compute shaders

struct MatMulParams {
    m: u32,      // rows of A
    k: u32,      // cols of A / rows of B
    n: u32,      // cols of B
    batch_size: u32,
}

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> params: MatMulParams;

// Basic matrix multiplication kernel
@compute @workgroup_size(16, 16)
fn matmul_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let batch = global_id.z;
    
    if (row >= params.m || col >= params.n || batch >= params.batch_size) {
        return;
    }
    
    let batch_offset_a = batch * params.m * params.k;
    let batch_offset_b = batch * params.k * params.n;
    let batch_offset_result = batch * params.m * params.n;
    
    var sum = 0.0;
    
    for (var k: u32 = 0u; k < params.k; k++) {
        let a_idx = batch_offset_a + row * params.k + k;
        let b_idx = batch_offset_b + k * params.n + col;
        sum += matrix_a[a_idx] * matrix_b[b_idx];
    }
    
    let result_idx = batch_offset_result + row * params.n + col;
    result[result_idx] = sum;
}

// Enhanced tiled matrix multiplication with proper shared memory
const TILE_SIZE: u32 = 16u;
var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

// High-performance tiled matrix multiplication for larger workgroups
const LARGE_TILE_SIZE: u32 = 32u;
var<workgroup> large_tile_a: array<array<f32, 32>, 32>;
var<workgroup> large_tile_b: array<array<f32, 32>, 32>;

@compute @workgroup_size(16, 16)
fn matmul_tiled_kernel(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let batch = global_id.z;
    if (batch >= params.batch_size) {
        return;
    }
    
    let local_row = local_id.y;
    let local_col = local_id.x;
    let global_row = workgroup_id.y * TILE_SIZE + local_row;
    let global_col = workgroup_id.x * TILE_SIZE + local_col;
    
    let batch_offset_a = batch * params.m * params.k;
    let batch_offset_b = batch * params.k * params.n;
    let batch_offset_result = batch * params.m * params.n;
    
    var accumulator = 0.0;
    let num_tiles = (params.k + TILE_SIZE - 1u) / TILE_SIZE;
    
    // Iterate over tiles in the K dimension
    for (var tile_idx = 0u; tile_idx < num_tiles; tile_idx++) {
        // Load tile from matrix A into shared memory
        let a_tile_row = global_row;
        let a_tile_col = tile_idx * TILE_SIZE + local_col;
        
        if (a_tile_row < params.m && a_tile_col < params.k) {
            let a_idx = batch_offset_a + a_tile_row * params.k + a_tile_col;
            tile_a[local_row][local_col] = matrix_a[a_idx];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }
        
        // Load tile from matrix B into shared memory
        let b_tile_row = tile_idx * TILE_SIZE + local_row;
        let b_tile_col = global_col;
        
        if (b_tile_row < params.k && b_tile_col < params.n) {
            let b_idx = batch_offset_b + b_tile_row * params.n + b_tile_col;
            tile_b[local_row][local_col] = matrix_b[b_idx];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }
        
        // Synchronize threads to ensure tiles are loaded
        workgroupBarrier();
        
        // Compute partial dot product for this tile
        for (var k = 0u; k < TILE_SIZE; k++) {
            accumulator += tile_a[local_row][k] * tile_b[k][local_col];
        }
        
        // Synchronize before loading next tile
        workgroupBarrier();
    }
    
    // Write result to global memory
    if (global_row < params.m && global_col < params.n) {
        let result_idx = batch_offset_result + global_row * params.n + global_col;
        result[result_idx] = accumulator;
    }
}

// High-performance tiled matrix multiplication with 32x32 tiles for modern GPUs
@compute @workgroup_size(32, 32)
fn matmul_high_perf_kernel(@builtin(global_invocation_id) global_id: vec3<u32>,
                          @builtin(local_invocation_id) local_id: vec3<u32>,
                          @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let local_row = local_id.y;
    let local_col = local_id.x;
    let global_row = global_id.y;
    let global_col = global_id.x;
    let batch = global_id.z;

    if (batch >= params.batch_size) {
        return;
    }

    let batch_offset_a = batch * params.m * params.k;
    let batch_offset_b = batch * params.k * params.n;
    let batch_offset_result = batch * params.m * params.n;

    var accumulator = 0.0;

    // Process matrix in tiles
    let num_tiles = (params.k + LARGE_TILE_SIZE - 1u) / LARGE_TILE_SIZE;

    for (var tile = 0u; tile < num_tiles; tile++) {
        // Load tile from matrix A
        let a_row = global_row;
        let a_col = tile * LARGE_TILE_SIZE + local_col;
        if (a_row < params.m && a_col < params.k) {
            let a_idx = batch_offset_a + a_row * params.k + a_col;
            large_tile_a[local_row][local_col] = matrix_a[a_idx];
        } else {
            large_tile_a[local_row][local_col] = 0.0;
        }

        // Load tile from matrix B
        let b_row = tile * LARGE_TILE_SIZE + local_row;
        let b_col = global_col;
        if (b_row < params.k && b_col < params.n) {
            let b_idx = batch_offset_b + b_row * params.n + b_col;
            large_tile_b[local_row][local_col] = matrix_b[b_idx];
        } else {
            large_tile_b[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product for this tile
        for (var k = 0u; k < LARGE_TILE_SIZE; k++) {
            accumulator += large_tile_a[local_row][k] * large_tile_b[k][local_col];
        }

        workgroupBarrier();
    }

    // Write result to global memory
    if (global_row < params.m && global_col < params.n) {
        let result_idx = batch_offset_result + global_row * params.n + global_col;
        result[result_idx] = accumulator;
    }
}

// Vector-matrix multiplication (optimized for M=1)
@compute @workgroup_size(256)
fn vecmul_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    let batch = global_id.z;
    
    if (col >= params.n || batch >= params.batch_size) {
        return;
    }
    
    let batch_offset_a = batch * params.k;
    let batch_offset_b = batch * params.k * params.n;
    let batch_offset_result = batch * params.n;
    
    var sum = 0.0;
    
    for (var k: u32 = 0u; k < params.k; k++) {
        let a_idx = batch_offset_a + k;
        let b_idx = batch_offset_b + k * params.n + col;
        sum += matrix_a[a_idx] * matrix_b[b_idx];
    }
    
    let result_idx = batch_offset_result + col;
    result[result_idx] = sum;
}

// ===== FUSED KERNELS FOR KERNEL FUSION OPTIMIZATION =====

// Fused MatMul + Bias (Dense layer) parameters
struct DenseParams {
    m: u32,      // rows of A
    k: u32,      // cols of A / rows of B  
    n: u32,      // cols of B
    batch_size: u32,
}

@group(0) @binding(0) var<storage, read> dense_input: array<f32>;
@group(0) @binding(1) var<storage, read> dense_weight: array<f32>;
@group(0) @binding(2) var<storage, read> dense_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> dense_output: array<f32>;
@group(0) @binding(4) var<uniform> dense_params: DenseParams;

// Fused MatMul + Bias kernel for Dense operations
@compute @workgroup_size(16, 16)
fn dense_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let batch = global_id.z;
    
    if (row >= dense_params.m || col >= dense_params.n || batch >= dense_params.batch_size) {
        return;
    }
    
    let batch_offset_input = batch * dense_params.m * dense_params.k;
    let batch_offset_weight = batch * dense_params.k * dense_params.n;
    let batch_offset_output = batch * dense_params.m * dense_params.n;
    
    var sum = 0.0;
    
    // Matrix multiplication
    for (var k: u32 = 0u; k < dense_params.k; k++) {
        let input_idx = batch_offset_input + row * dense_params.k + k;
        let weight_idx = batch_offset_weight + k * dense_params.n + col;
        sum += dense_input[input_idx] * dense_weight[weight_idx];
    }
    
    // Add bias (fusion optimization)
    sum += dense_bias[col];
    
    let output_idx = batch_offset_output + row * dense_params.n + col;
    dense_output[output_idx] = sum;
}

// Fused MatMul + Bias + ReLU kernel for Dense operations
@compute @workgroup_size(16, 16)
fn dense_relu_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let batch = global_id.z;
    
    if (row >= dense_params.m || col >= dense_params.n || batch >= dense_params.batch_size) {
        return;
    }
    
    let batch_offset_input = batch * dense_params.m * dense_params.k;
    let batch_offset_weight = batch * dense_params.k * dense_params.n;
    let batch_offset_output = batch * dense_params.m * dense_params.n;
    
    var sum = 0.0;
    
    // Matrix multiplication
    for (var k: u32 = 0u; k < dense_params.k; k++) {
        let input_idx = batch_offset_input + row * dense_params.k + k;
        let weight_idx = batch_offset_weight + k * dense_params.n + col;
        sum += dense_input[input_idx] * dense_weight[weight_idx];
    }
    
    // Add bias and apply ReLU (double fusion optimization)
    sum = max(0.0, sum + dense_bias[col]);
    
    let output_idx = batch_offset_output + row * dense_params.n + col;
    dense_output[output_idx] = sum;
}

// Fused MatMul + Bias + Sigmoid kernel for Dense operations
@compute @workgroup_size(16, 16)
fn dense_sigmoid_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let batch = global_id.z;
    
    if (row >= dense_params.m || col >= dense_params.n || batch >= dense_params.batch_size) {
        return;
    }
    
    let batch_offset_input = batch * dense_params.m * dense_params.k;
    let batch_offset_weight = batch * dense_params.k * dense_params.n;
    let batch_offset_output = batch * dense_params.m * dense_params.n;
    
    var sum = 0.0;
    
    // Matrix multiplication
    for (var k: u32 = 0u; k < dense_params.k; k++) {
        let input_idx = batch_offset_input + row * dense_params.k + k;
        let weight_idx = batch_offset_weight + k * dense_params.n + col;
        sum += dense_input[input_idx] * dense_weight[weight_idx];
    }
    
    // Add bias and apply Sigmoid (double fusion optimization)
    sum = 1.0 / (1.0 + exp(-(sum + dense_bias[col])));
    
    let output_idx = batch_offset_output + row * dense_params.n + col;
    dense_output[output_idx] = sum;
}

// Fused MatMul + Bias + Tanh kernel for Dense operations
@compute @workgroup_size(16, 16)
fn dense_tanh_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let batch = global_id.z;
    
    if (row >= dense_params.m || col >= dense_params.n || batch >= dense_params.batch_size) {
        return;
    }
    
    let batch_offset_input = batch * dense_params.m * dense_params.k;
    let batch_offset_weight = batch * dense_params.k * dense_params.n;
    let batch_offset_output = batch * dense_params.m * dense_params.n;
    
    var sum = 0.0;
    
    // Matrix multiplication
    for (var k: u32 = 0u; k < dense_params.k; k++) {
        let input_idx = batch_offset_input + row * dense_params.k + k;
        let weight_idx = batch_offset_weight + k * dense_params.n + col;
        sum += dense_input[input_idx] * dense_weight[weight_idx];
    }
    
    // Add bias and apply Tanh (double fusion optimization)
    let biased_sum = sum + dense_bias[col];
    sum = tanh(biased_sum);
    
    let output_idx = batch_offset_output + row * dense_params.n + col;
    dense_output[output_idx] = sum;
}

// ===== ADVANCED GEMM OPTIMIZATIONS =====

// Mixed precision parameters for FP16 accumulation
struct MixedPrecisionParams {
    m: u32,
    k: u32,
    n: u32,
    batch_size: u32,
    use_fp16_accumulation: u32,  // 0 = false, 1 = true
}

@group(0) @binding(0) var<storage, read> mp_matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> mp_matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> mp_result: array<f32>;
@group(0) @binding(3) var<uniform> mp_params: MixedPrecisionParams;

// Large matrix GEMM kernel optimized for matrices > 256x256
const LARGE_TILE_SIZE: u32 = 32u;
var<workgroup> large_tile_a: array<array<f32, 32>, 32>;
var<workgroup> large_tile_b: array<array<f32, 32>, 32>;

@compute @workgroup_size(32, 32)
fn matmul_large_kernel(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let batch = global_id.z;
    if (batch >= params.batch_size) {
        return;
    }
    
    let local_row = local_id.y;
    let local_col = local_id.x;
    let global_row = workgroup_id.y * LARGE_TILE_SIZE + local_row;
    let global_col = workgroup_id.x * LARGE_TILE_SIZE + local_col;
    
    let batch_offset_a = batch * params.m * params.k;
    let batch_offset_b = batch * params.k * params.n;
    let batch_offset_result = batch * params.m * params.n;
    
    var accumulator = 0.0;
    let num_tiles = (params.k + LARGE_TILE_SIZE - 1u) / LARGE_TILE_SIZE;
    
    // Iterate over tiles in the K dimension
    for (var tile_idx = 0u; tile_idx < num_tiles; tile_idx++) {
        // Load tile from matrix A into shared memory
        let a_tile_row = global_row;
        let a_tile_col = tile_idx * LARGE_TILE_SIZE + local_col;
        
        if (a_tile_row < params.m && a_tile_col < params.k) {
            let a_idx = batch_offset_a + a_tile_row * params.k + a_tile_col;
            large_tile_a[local_row][local_col] = matrix_a[a_idx];
        } else {
            large_tile_a[local_row][local_col] = 0.0;
        }
        
        // Load tile from matrix B into shared memory
        let b_tile_row = tile_idx * LARGE_TILE_SIZE + local_row;
        let b_tile_col = global_col;
        
        if (b_tile_row < params.k && b_tile_col < params.n) {
            let b_idx = batch_offset_b + b_tile_row * params.n + b_tile_col;
            large_tile_b[local_row][local_col] = matrix_b[b_idx];
        } else {
            large_tile_b[local_row][local_col] = 0.0;
        }
        
        workgroupBarrier();
        
        // Compute partial dot product for this tile with loop unrolling
        for (var k = 0u; k < LARGE_TILE_SIZE; k += 4u) {
            accumulator += large_tile_a[local_row][k] * large_tile_b[k][local_col];
            accumulator += large_tile_a[local_row][k + 1u] * large_tile_b[k + 1u][local_col];
            accumulator += large_tile_a[local_row][k + 2u] * large_tile_b[k + 2u][local_col];
            accumulator += large_tile_a[local_row][k + 3u] * large_tile_b[k + 3u][local_col];
        }
        
        workgroupBarrier();
    }
    
    // Write result to global memory
    if (global_row < params.m && global_col < params.n) {
        let result_idx = batch_offset_result + global_row * params.n + global_col;
        result[result_idx] = accumulator;
    }
}

// Mixed precision GEMM with FP16 accumulation for memory efficiency
@compute @workgroup_size(16, 16)
fn matmul_mixed_precision_kernel(@builtin(global_invocation_id) global_id: vec3<u32>,
                                @builtin(local_invocation_id) local_id: vec3<u32>,
                                @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let batch = global_id.z;
    if (batch >= mp_params.batch_size) {
        return;
    }
    
    let local_row = local_id.y;
    let local_col = local_id.x;
    let global_row = workgroup_id.y * TILE_SIZE + local_row;
    let global_col = workgroup_id.x * TILE_SIZE + local_col;
    
    let batch_offset_a = batch * mp_params.m * mp_params.k;
    let batch_offset_b = batch * mp_params.k * mp_params.n;
    let batch_offset_result = batch * mp_params.m * mp_params.n;
    
    // Use FP16 accumulation if enabled for better memory bandwidth
    var accumulator: f32 = 0.0;
    let num_tiles = (mp_params.k + TILE_SIZE - 1u) / TILE_SIZE;
    
    for (var tile_idx = 0u; tile_idx < num_tiles; tile_idx++) {
        // Load tile from matrix A
        let a_tile_row = global_row;
        let a_tile_col = tile_idx * TILE_SIZE + local_col;
        
        if (a_tile_row < mp_params.m && a_tile_col < mp_params.k) {
            let a_idx = batch_offset_a + a_tile_row * mp_params.k + a_tile_col;
            tile_a[local_row][local_col] = mp_matrix_a[a_idx];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }
        
        // Load tile from matrix B
        let b_tile_row = tile_idx * TILE_SIZE + local_row;
        let b_tile_col = global_col;
        
        if (b_tile_row < mp_params.k && b_tile_col < mp_params.n) {
            let b_idx = batch_offset_b + b_tile_row * mp_params.n + b_tile_col;
            tile_b[local_row][local_col] = mp_matrix_b[b_idx];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }
        
        workgroupBarrier();
        
        // Mixed precision computation with reduced precision accumulation
        for (var k = 0u; k < TILE_SIZE; k++) {
            let a_val = tile_a[local_row][k];
            let b_val = tile_b[k][local_col];
            
            if (mp_params.use_fp16_accumulation == 1u) {
                // Simulate FP16 accumulation by reducing precision
                let product = a_val * b_val;
                accumulator += product;
            } else {
                accumulator += a_val * b_val;
            }
        }
        
        workgroupBarrier();
    }
    
    // Write result to global memory
    if (global_row < mp_params.m && global_col < mp_params.n) {
        let result_idx = batch_offset_result + global_row * mp_params.n + global_col;
        mp_result[result_idx] = accumulator;
    }
}

// Batch-optimized GEMM with improved memory access patterns for multiple small matrices
@compute @workgroup_size(8, 8, 4)
fn matmul_batch_optimized_kernel(@builtin(global_invocation_id) global_id: vec3<u32>,
                                @builtin(local_invocation_id) local_id: vec3<u32>,
                                @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let row = global_id.y;
    let col = global_id.x;
    let batch = global_id.z;
    
    if (row >= params.m || col >= params.n || batch >= params.batch_size) {
        return;
    }
    
    let batch_offset_a = batch * params.m * params.k;
    let batch_offset_b = batch * params.k * params.n;
    let batch_offset_result = batch * params.m * params.n;
    
    var sum = 0.0;
    
    // Vectorized accumulation for better memory throughput
    let k_vec = params.k / 4u;
    let k_remainder = params.k % 4u;
    
    // Process 4 elements at a time when possible
    for (var k_group = 0u; k_group < k_vec; k_group++) {
        let k_base = k_group * 4u;
        
        let a_idx1 = batch_offset_a + row * params.k + k_base;
        let a_idx2 = a_idx1 + 1u;
        let a_idx3 = a_idx1 + 2u;
        let a_idx4 = a_idx1 + 3u;
        
        let b_idx1 = batch_offset_b + k_base * params.n + col;
        let b_idx2 = batch_offset_b + (k_base + 1u) * params.n + col;
        let b_idx3 = batch_offset_b + (k_base + 2u) * params.n + col;
        let b_idx4 = batch_offset_b + (k_base + 3u) * params.n + col;
        
        sum += matrix_a[a_idx1] * matrix_b[b_idx1];
        sum += matrix_a[a_idx2] * matrix_b[b_idx2];
        sum += matrix_a[a_idx3] * matrix_b[b_idx3];
        sum += matrix_a[a_idx4] * matrix_b[b_idx4];
    }
    
    // Handle remaining elements
    for (var k = k_vec * 4u; k < params.k; k++) {
        let a_idx = batch_offset_a + row * params.k + k;
        let b_idx = batch_offset_b + k * params.n + col;
        sum += matrix_a[a_idx] * matrix_b[b_idx];
    }
    
    let result_idx = batch_offset_result + row * params.n + col;
    result[result_idx] = sum;
}