// Enhanced binary operation compute shaders with memory coalescing optimizations for u64
// This shader implements tiled memory access patterns for improved performance

@group(0) @binding(0) var<storage, read> input_a: array<u64>;
@group(0) @binding(1) var<storage, read> input_b: array<u64>;
@group(0) @binding(2) var<storage, read_write> output: array<u64>;

// Coalescing parameters
@group(0) @binding(3) var<storage, read> shape_metadata: array<u32>; // [width, height, depth, batch_size]

// Shared memory for tile-based processing
var<workgroup> tile_a: array<u64, 256>; // 16x16 tile
var<workgroup> tile_b: array<u64, 256>; // 16x16 tile

// Optimized workgroup size for memory coalescing
// Using 2D workgroups for better memory access patterns
@compute @workgroup_size(16, 16, 1)
fn add_op_coalesced_u64(@builtin(global_invocation_id) global_id: vec3<u32>,
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
// Process 2 u64 elements at once (since u64 is larger than u32)
@compute @workgroup_size(64, 1, 1)
fn add_op_vectorized_u64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_index = global_id.x * 2u; // Process 2 u64 elements at once
    
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
    
    // Vectorized memory access - process 2 u64 elements at once
    let a_base = base_index % arrayLength(&input_a);
    let b_base = base_index % arrayLength(&input_b);
    
    // Load 2 u64 elements at once for better memory bandwidth
    output[base_index] = input_a[a_base] + input_b[b_base];
    output[base_index + 1u] = input_a[a_base + 1u] + input_b[b_base + 1u];
}

// Memory-coalesced subtraction
@compute @workgroup_size(16, 16, 1)
fn sub_op_coalesced_u64(@builtin(global_invocation_id) global_id: vec3<u32>,
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

@compute @workgroup_size(64, 1, 1)
fn sub_op_vectorized_u64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_index = global_id.x * 2u;
    
    if (base_index + 1u >= arrayLength(&output)) {
        let index = base_index;
        if (index < arrayLength(&output)) {
            let a_idx = index % arrayLength(&input_a);
            let b_idx = index % arrayLength(&input_b);
            output[index] = input_a[a_idx] - input_b[b_idx];
        }
        return;
    }
    
    let a_base = base_index % arrayLength(&input_a);
    let b_base = base_index % arrayLength(&input_b);
    
    output[base_index] = input_a[a_base] - input_b[b_base];
    output[base_index + 1u] = input_a[a_base + 1u] - input_b[b_base + 1u];
}

// Memory-coalesced multiplication
@compute @workgroup_size(16, 16, 1)
fn mul_op_coalesced_u64(@builtin(global_invocation_id) global_id: vec3<u32>,
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

@compute @workgroup_size(64, 1, 1)
fn mul_op_vectorized_u64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_index = global_id.x * 2u;
    
    if (base_index + 1u >= arrayLength(&output)) {
        let index = base_index;
        if (index < arrayLength(&output)) {
            let a_idx = index % arrayLength(&input_a);
            let b_idx = index % arrayLength(&input_b);
            output[index] = input_a[a_idx] * input_b[b_idx];
        }
        return;
    }
    
    let a_base = base_index % arrayLength(&input_a);
    let b_base = base_index % arrayLength(&input_b);
    
    output[base_index] = input_a[a_base] * input_b[b_base];
    output[base_index + 1u] = input_a[a_base + 1u] * input_b[b_base + 1u];
}

// Memory-coalesced division
@compute @workgroup_size(16, 16, 1)
fn div_op_coalesced_u64(@builtin(global_invocation_id) global_id: vec3<u32>,
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
        // Handle division by zero for unsigned integers
        if (tile_b[local_index] == 0u) {
            output[index] = 0u;
        } else {
            output[index] = tile_a[local_index] / tile_b[local_index];
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn div_op_vectorized_u64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_index = global_id.x * 2u;
    
    if (base_index + 1u >= arrayLength(&output)) {
        let index = base_index;
        if (index < arrayLength(&output)) {
            let a_idx = index % arrayLength(&input_a);
            let b_idx = index % arrayLength(&input_b);
            if (input_b[b_idx] == 0u) {
                output[index] = 0u;
            } else {
                output[index] = input_a[a_idx] / input_b[b_idx];
            }
        }
        return;
    }
    
    let a_base = base_index % arrayLength(&input_a);
    let b_base = base_index % arrayLength(&input_b);
    
    // Handle division by zero for vectorized operations
    if (input_b[b_base] == 0u) {
        output[base_index] = 0u;
    } else {
        output[base_index] = input_a[a_base] / input_b[b_base];
    }
    
    if (input_b[b_base + 1u] == 0u) {
        output[base_index + 1u] = 0u;
    } else {
        output[base_index + 1u] = input_a[a_base + 1u] / input_b[b_base + 1u];
    }
}

// Memory-coalesced power operation
@compute @workgroup_size(16, 16, 1)
fn pow_op_coalesced_u64(@builtin(global_invocation_id) global_id: vec3<u32>,
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
        let base = tile_a[local_index];
        let exponent = tile_b[local_index];
        
        // Handle special cases
        if (exponent == 0u) {
            output[index] = 1u;
        } else if (base == 0u) {
            output[index] = 0u;
        } else {
            var result = 1u;
            var current_base = base;
            var current_exp = exponent;
            
            // Fast exponentiation
            while (current_exp > 0u) {
                if ((current_exp & 1u) == 1u) {
                    result = result * current_base;
                }
                current_base = current_base * current_base;
                current_exp = current_exp >> 1u;
            }
            
            output[index] = result;
        }
    }
}

// Memory-coalesced PReLU operation (for unsigned integers, just returns input)
@compute @workgroup_size(16, 16, 1)
fn prelu_op_coalesced_u64(@builtin(global_invocation_id) global_id: vec3<u32>,
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
    
    workgroupBarrier();
    
    if (index < arrayLength(&output)) {
        // For unsigned integers, PReLU doesn't make sense since all values are >= 0
        // Just return the input value
        output[index] = tile_a[local_index];
    }
}