// Enhanced binary operation compute shaders with memory coalescing optimizations
// This shader implements tiled memory access patterns for improved performance

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Coalescing parameters
@group(0) @binding(3) var<storage, read> shape_metadata: array<u32>; // [width, height, depth, batch_size]

// Shared memory for tile-based processing
var<workgroup> tile_a: array<f32, 256>; // 16x16 tile
var<workgroup> tile_b: array<f32, 256>; // 16x16 tile

// Optimized workgroup size for memory coalescing
// Using 2D workgroups for better memory access patterns
@compute @workgroup_size(16, 16, 1)
fn add_op_coalesced(@builtin(global_invocation_id) global_id: vec3<u32>,
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
fn add_op_vectorized(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_index = global_id.x * 4u; // Process 4 elements at once
    
    // Check bounds for vectorized access
    if (base_index + 3u >= arrayLength(&output)) {
        // Fallback to scalar processing for remaining elements
        let index = base_index;
        if (index < arrayLength(&output)) {
            let a_idx = index % arrayLength(&input_a);
            let b_idx = index % arrayLength(&input_b);
            output[index] = input_a[a_idx] + input_b[b_idx];
        }
        return;
    }
    
    // Vectorized memory access - process 4 elements at once
    let a_base = base_index % arrayLength(&input_a);
    let b_base = base_index % arrayLength(&input_b);
    
    // Load 4 elements at once for better memory bandwidth
    output[base_index] = input_a[a_base] + input_b[b_base];
    output[base_index + 1u] = input_a[a_base + 1u] + input_b[b_base + 1u];
    output[base_index + 2u] = input_a[a_base + 2u] + input_b[b_base + 2u];
    output[base_index + 3u] = input_a[a_base + 3u] + input_b[b_base + 3u];
}

// Memory-coalesced subtraction
@compute @workgroup_size(16, 16, 1)
fn sub_op_coalesced(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let width = shape_metadata[0];
    let height = shape_metadata[1];
    let depth = shape_metadata[2];
    
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    let index = z * width * height + y * width + x;
    
    if (x >= width || y >= height || z >= depth || index >= arrayLength(&output)) {
        return;
    }
    
    let local_index = local_id.y * 16u + local_id.x;
    
    // Load into shared memory
    if (index < arrayLength(&input_a)) {
        tile_a[local_index] = input_a[index];
    }
    if (index < arrayLength(&input_b)) {
        tile_b[local_index] = input_b[index];
    }
    
    workgroupBarrier();
    
    output[index] = tile_a[local_index] - tile_b[local_index];
}

// Memory-coalesced multiplication
@compute @workgroup_size(16, 16, 1)
fn mul_op_coalesced(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let width = shape_metadata[0];
    let height = shape_metadata[1];
    let depth = shape_metadata[2];
    
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    let index = z * width * height + y * width + x;
    
    if (x >= width || y >= height || z >= depth || index >= arrayLength(&output)) {
        return;
    }
    
    let local_index = local_id.y * 16u + local_id.x;
    
    // Load into shared memory
    if (index < arrayLength(&input_a)) {
        tile_a[local_index] = input_a[index];
    }
    if (index < arrayLength(&input_b)) {
        tile_b[local_index] = input_b[index];
    }
    
    workgroupBarrier();
    
    output[index] = tile_a[local_index] * tile_b[local_index];
}

// Memory-coalesced division
@compute @workgroup_size(16, 16, 1)
fn div_op_coalesced(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let width = shape_metadata[0];
    let height = shape_metadata[1];
    let depth = shape_metadata[2];
    
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    let index = z * width * height + y * width + x;
    
    if (x >= width || y >= height || z >= depth || index >= arrayLength(&output)) {
        return;
    }
    
    let local_index = local_id.y * 16u + local_id.x;
    
    // Load into shared memory
    if (index < arrayLength(&input_a)) {
        tile_a[local_index] = input_a[index];
    }
    if (index < arrayLength(&input_b)) {
        tile_b[local_index] = input_b[index];
    }
    
    workgroupBarrier();
    
    // Avoid division by zero
    let b_val = tile_b[local_index];
    if (abs(b_val) < 1e-10) {
        output[index] = 0.0;
    } else {
        output[index] = tile_a[local_index] / b_val;
    }
}

// Adaptive workgroup size selection based on tensor size
@compute @workgroup_size(256, 1, 1)
fn add_op_adaptive(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let array_len = arrayLength(&output);
    
    // Use larger workgroup for large tensors
    if (index >= array_len) {
        return;
    }
    
    // Coalesced memory access - adjacent threads access adjacent memory
    let a_idx = index % arrayLength(&input_a);
    let b_idx = index % arrayLength(&input_b);
    
    output[index] = input_a[a_idx] + input_b[b_idx];
}

// Bank-conflict avoidance for shared memory access
@compute @workgroup_size(32, 8, 1)
fn add_op_bank_conflict_free(@builtin(global_invocation_id) global_id: vec3<u32>,
                             @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let width = shape_metadata[0];
    let height = shape_metadata[1];
    
    let x = global_id.x;
    let y = global_id.y;
    
    let index = y * width + x;
    
    if (x >= width || y >= height || index >= arrayLength(&output)) {
        return;
    }
    
    // Bank-conflict free indexing
    let local_x = local_id.x;
    let local_y = local_id.y;
    let local_index = local_y * 33u + local_x; // +1 to avoid bank conflicts
    
    // Load data avoiding bank conflicts
    if (index < arrayLength(&input_a)) {
        tile_a[local_index] = input_a[index];
    }
    if (index < arrayLength(&input_b)) {
        tile_b[local_index] = input_b[index];
    }
    
    workgroupBarrier();
    
    output[index] = tile_a[local_index] + tile_b[local_index];
}