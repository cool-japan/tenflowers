// Enhanced binary operation compute shaders with memory coalescing optimizations for f64
// This shader implements tiled memory access patterns for improved performance

@group(0) @binding(0) var<storage, read> input_a: array<f64>;
@group(0) @binding(1) var<storage, read> input_b: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

// Coalescing parameters
@group(0) @binding(3) var<storage, read> shape_metadata: array<u32>; // [width, height, depth, batch_size]

// Shared memory for tile-based processing (reduced size for f64)
var<workgroup> tile_a: array<f64, 128>; // 8x16 tile for f64
var<workgroup> tile_b: array<f64, 128>; // 8x16 tile for f64

// Optimized workgroup size for memory coalescing with f64
// Using 2D workgroups for better memory access patterns
@compute @workgroup_size(16, 8, 1)
fn add_op_coalesced_f64(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>,
                        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let width = shape_metadata[0];
    let height = shape_metadata[1];
    let depth = shape_metadata[2];
    let batch_size = shape_metadata[3];
    
    // 2D memory access pattern for better coalescing
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    // Calculate linear index with proper stride
    let index = z * width * height + y * width + x;
    
    // Check bounds
    if (x >= width || y >= height || z >= depth) {
        return;
    }
    
    // Coalesced memory access - adjacent threads access adjacent memory
    let local_index = local_id.y * 16u + local_id.x;
    
    // Load data into shared memory for cache efficiency
    if (index < arrayLength(&input_a)) {
        tile_a[local_index] = input_a[index];
    }
    if (index < arrayLength(&input_b)) {
        tile_b[local_index] = input_b[index];
    }
    
    // Synchronize workgroup
    workgroupBarrier();
    
    // Perform operation using cached data
    if (index < arrayLength(&output)) {
        output[index] = tile_a[local_index] + tile_b[local_index];
    }
}

// Vectorized memory access for better bandwidth utilization
@compute @workgroup_size(64, 1, 1)
fn add_op_vectorized_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_index = global_id.x * 2u; // Process 2 f64 elements at once
    
    // Check bounds for vectorized access
    if (base_index + 1u >= arrayLength(&output)) {
        // Fallback to scalar processing for remaining elements
        let index = base_index;
        if (index < arrayLength(&output)) {
            let a_idx = index % arrayLength(&input_a);
            let b_idx = index % arrayLength(&input_b);
            output[index] = input_a[a_idx] + input_b[b_idx];
        }
        return;
    }
    
    // Vectorized memory access - process 2 f64 elements at once
    let a_base = base_index % arrayLength(&input_a);
    let b_base = base_index % arrayLength(&input_b);
    
    // Load 2 f64 elements at once for better memory bandwidth
    output[base_index] = input_a[a_base] + input_b[b_base];
    output[base_index + 1u] = input_a[a_base + 1u] + input_b[b_base + 1u];
}

// Memory-coalesced subtraction
@compute @workgroup_size(16, 8, 1)
fn sub_op_coalesced_f64(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let width = shape_metadata[0];
    let height = shape_metadata[1];
    let depth = shape_metadata[2];
    
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    let index = z * width * height + y * width + x;
    
    if (x >= width || y >= height || z >= depth) {
        return;
    }
    
    let local_index = local_id.y * 16u + local_id.x;
    
    if (index < arrayLength(&input_a)) {
        tile_a[local_index] = input_a[index];
    }
    if (index < arrayLength(&input_b)) {
        tile_b[local_index] = input_b[index];
    }
    
    workgroupBarrier();
    
    if (index < arrayLength(&output)) {
        output[index] = tile_a[local_index] - tile_b[local_index];
    }
}

// Memory-coalesced multiplication
@compute @workgroup_size(16, 8, 1)
fn mul_op_coalesced_f64(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let width = shape_metadata[0];
    let height = shape_metadata[1];
    let depth = shape_metadata[2];
    
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    let index = z * width * height + y * width + x;
    
    if (x >= width || y >= height || z >= depth) {
        return;
    }
    
    let local_index = local_id.y * 16u + local_id.x;
    
    if (index < arrayLength(&input_a)) {
        tile_a[local_index] = input_a[index];
    }
    if (index < arrayLength(&input_b)) {
        tile_b[local_index] = input_b[index];
    }
    
    workgroupBarrier();
    
    if (index < arrayLength(&output)) {
        output[index] = tile_a[local_index] * tile_b[local_index];
    }
}

// Memory-coalesced division
@compute @workgroup_size(16, 8, 1)
fn div_op_coalesced_f64(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let width = shape_metadata[0];
    let height = shape_metadata[1];
    let depth = shape_metadata[2];
    
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    let index = z * width * height + y * width + x;
    
    if (x >= width || y >= height || z >= depth) {
        return;
    }
    
    let local_index = local_id.y * 16u + local_id.x;
    
    if (index < arrayLength(&input_a)) {
        tile_a[local_index] = input_a[index];
    }
    if (index < arrayLength(&input_b)) {
        tile_b[local_index] = input_b[index];
    }
    
    workgroupBarrier();
    
    if (index < arrayLength(&output)) {
        output[index] = tile_a[local_index] / tile_b[local_index];
    }
}

// Memory-coalesced power operation
@compute @workgroup_size(16, 8, 1)
fn pow_op_coalesced_f64(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let width = shape_metadata[0];
    let height = shape_metadata[1];
    let depth = shape_metadata[2];
    
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    let index = z * width * height + y * width + x;
    
    if (x >= width || y >= height || z >= depth) {
        return;
    }
    
    let local_index = local_id.y * 16u + local_id.x;
    
    if (index < arrayLength(&input_a)) {
        tile_a[local_index] = input_a[index];
    }
    if (index < arrayLength(&input_b)) {
        tile_b[local_index] = input_b[index];
    }
    
    workgroupBarrier();
    
    if (index < arrayLength(&output)) {
        output[index] = pow(tile_a[local_index], tile_b[local_index]);
    }
}

// Memory-coalesced PReLU operation
@compute @workgroup_size(16, 8, 1)
fn prelu_op_coalesced_f64(@builtin(global_invocation_id) global_id: vec3<u32>,
                          @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let width = shape_metadata[0];
    let height = shape_metadata[1];
    let depth = shape_metadata[2];
    
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    let index = z * width * height + y * width + x;
    
    if (x >= width || y >= height || z >= depth) {
        return;
    }
    
    let local_index = local_id.y * 16u + local_id.x;
    
    if (index < arrayLength(&input_a)) {
        tile_a[local_index] = input_a[index];
    }
    if (index < arrayLength(&input_b)) {
        tile_b[local_index] = input_b[index];
    }
    
    workgroupBarrier();
    
    if (index < arrayLength(&output)) {
        let x_val = tile_a[local_index];
        let alpha = tile_b[local_index];
        output[index] = select(alpha * x_val, x_val, x_val > 0.0);
    }
}