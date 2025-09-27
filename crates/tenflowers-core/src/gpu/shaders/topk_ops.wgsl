// GPU TopK operations using parallel sorting

// Input and output buffers
@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_indices: array<u32>;

// Metadata: [axis_size, k, num_slices, stride]
@group(0) @binding(3) var<storage, read> metadata: array<u32>;

// Workgroup shared memory for sorting
var<workgroup> shared_values: array<f32, 256>;
var<workgroup> shared_indices: array<u32, 256>;

// Bitonic sort for small arrays (up to 256 elements)
fn bitonic_sort_step(tid: u32, j: u32, k: u32) {
    let ixj = tid ^ j;
    
    if ixj > tid {
        // Determine sort direction
        let ascending = (tid & k) == 0u;
        
        let should_swap = if ascending {
            shared_values[tid] > shared_values[ixj]
        } else {
            shared_values[tid] < shared_values[ixj]
        };
        
        if should_swap {
            // Swap values
            let temp_val = shared_values[tid];
            shared_values[tid] = shared_values[ixj];
            shared_values[ixj] = temp_val;
            
            // Swap indices
            let temp_idx = shared_indices[tid];
            shared_indices[tid] = shared_indices[ixj];
            shared_indices[ixj] = temp_idx;
        }
    }
}

@compute @workgroup_size(256)
fn topk_bitonic_sort(@builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(local_invocation_id) local_id: vec3<u32>,
                     @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let tid = local_id.x;
    let slice_idx = workgroup_id.x;
    let axis_size = metadata[0];
    let k = metadata[1];
    let num_slices = metadata[2];
    let stride = metadata[3];
    
    if slice_idx >= num_slices {
        return;
    }
    
    // Calculate base offset for this slice
    let base_offset = slice_idx * axis_size;
    
    // Load data into shared memory
    if tid < axis_size {
        shared_values[tid] = input_values[base_offset + tid];
        shared_indices[tid] = tid;
    } else {
        // Fill remaining with minimum values for proper sorting
        shared_values[tid] = -3.402823e+38; // -FLT_MAX
        shared_indices[tid] = 0u;
    }
    
    workgroupBarrier();
    
    // Bitonic sort - works for power-of-2 sizes up to 256
    let sort_size = 256u; // Fixed workgroup size
    
    // Bitonic sort steps
    for (var k_step = 2u; k_step <= sort_size; k_step *= 2u) {
        for (var j = k_step / 2u; j > 0u; j /= 2u) {
            bitonic_sort_step(tid, j, k_step);
            workgroupBarrier();
        }
    }
    
    // Write top k results (sorted in descending order)
    if tid < k {
        let output_base = slice_idx * k;
        output_values[output_base + tid] = shared_values[tid];
        output_indices[output_base + tid] = shared_indices[tid];
    }
}

// Alternative implementation using heap-based partial sort for larger arrays
var<workgroup> heap_values: array<f32, 64>;
var<workgroup> heap_indices: array<u32, 64>;

fn heap_parent(i: u32) -> u32 {
    return (i - 1u) / 2u;
}

fn heap_left_child(i: u32) -> u32 {
    return 2u * i + 1u;
}

fn heap_right_child(i: u32) -> u32 {
    return 2u * i + 2u;
}

fn heap_swap(i: u32, j: u32) {
    let temp_val = heap_values[i];
    heap_values[i] = heap_values[j];
    heap_values[j] = temp_val;
    
    let temp_idx = heap_indices[i];
    heap_indices[i] = heap_indices[j];
    heap_indices[j] = temp_idx;
}

// Min-heap operations for maintaining top k elements
fn heapify_down(heap_size: u32, i: u32) {
    var largest = i;
    let left = heap_left_child(i);
    let right = heap_right_child(i);
    
    if left < heap_size && heap_values[left] < heap_values[largest] {
        largest = left;
    }
    
    if right < heap_size && heap_values[right] < heap_values[largest] {
        largest = right;
    }
    
    if largest != i {
        heap_swap(i, largest);
        heapify_down(heap_size, largest);
    }
}

fn heapify_up(i: u32) {
    if i > 0u {
        let parent = heap_parent(i);
        if heap_values[i] < heap_values[parent] {
            heap_swap(i, parent);
            heapify_up(parent);
        }
    }
}

@compute @workgroup_size(64)
fn topk_heap_sort(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let tid = local_id.x;
    let slice_idx = workgroup_id.x;
    let axis_size = metadata[0];
    let k = metadata[1];
    let num_slices = metadata[2];
    
    if slice_idx >= num_slices {
        return;
    }
    
    // Calculate base offset for this slice
    let base_offset = slice_idx * axis_size;
    
    // Initialize heap with first k elements
    if tid < k {
        heap_values[tid] = input_values[base_offset + tid];
        heap_indices[tid] = tid;
    }
    
    workgroupBarrier();
    
    // Only thread 0 processes the heap
    if tid == 0u {
        // Build min-heap from first k elements
        for (var i = k / 2u; i > 0u; i--) {
            heapify_down(k, i - 1u);
        }
        
        // Process remaining elements
        for (var i = k; i < axis_size; i++) {
            let value = input_values[base_offset + i];
            // If current element is larger than heap root (minimum)
            if value > heap_values[0] {
                heap_values[0] = value;
                heap_indices[0] = i;
                heapify_down(k, 0u);
            }
        }
        
        // Extract elements from heap and sort them in descending order
        var temp_values: array<f32, 64>;
        var temp_indices: array<u32, 64>;
        
        for (var i = 0u; i < k; i++) {
            temp_values[i] = heap_values[i];
            temp_indices[i] = heap_indices[i];
        }
        
        // Simple selection sort to arrange in descending order
        for (var i = 0u; i < k; i++) {
            var max_idx = i;
            for (var j = i + 1u; j < k; j++) {
                if temp_values[j] > temp_values[max_idx] {
                    max_idx = j;
                }
            }
            
            // Swap
            let temp_val = temp_values[i];
            temp_values[i] = temp_values[max_idx];
            temp_values[max_idx] = temp_val;
            
            let temp_idx = temp_indices[i];
            temp_indices[i] = temp_indices[max_idx];
            temp_indices[max_idx] = temp_idx;
        }
        
        // Write results
        let output_base = slice_idx * k;
        for (var i = 0u; i < k; i++) {
            output_values[output_base + i] = temp_values[i];
            output_indices[output_base + i] = temp_indices[i];
        }
    }
}

// Simple parallel selection for small k values
@compute @workgroup_size(256)
fn topk_selection(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let tid = local_id.x;
    let slice_idx = workgroup_id.x;
    let axis_size = metadata[0];
    let k = metadata[1];
    let num_slices = metadata[2];
    
    if slice_idx >= num_slices {
        return;
    }
    
    // Calculate base offset for this slice
    let base_offset = slice_idx * axis_size;
    
    // Each thread finds the tid-th largest element
    if tid < k {
        var max_val = -3.402823e+38; // -FLT_MAX
        var max_idx = 0u;
        
        // Find the tid-th largest element
        for (var rank = 0u; rank <= tid; rank++) {
            var current_max = -3.402823e+38;
            var current_idx = 0u;
            
            for (var i = 0u; i < axis_size; i++) {
                let value = input_values[base_offset + i];
                
                // Check if this value is the next largest
                var is_next_largest = true;
                var count_larger = 0u;
                
                for (var j = 0u; j < axis_size; j++) {
                    if input_values[base_offset + j] > value {
                        count_larger++;
                    }
                }
                
                if count_larger == rank && value > current_max {
                    current_max = value;
                    current_idx = i;
                }
            }
            
            if rank == tid {
                max_val = current_max;
                max_idx = current_idx;
            }
        }
        
        // Write result
        let output_base = slice_idx * k;
        output_values[output_base + tid] = max_val;
        output_indices[output_base + tid] = max_idx;
    }
}