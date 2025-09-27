// Linear Algebra: Matrix Transpose
// 
// Efficiently transposes a matrix using shared memory tiles for optimal memory coalescing.
// This implementation follows cuBLAS patterns adapted for WGPU.
//
// Input:  Matrix A [rows_a, cols_a]
// Output: Matrix A^T [cols_a, rows_a]

struct LinalgMetadata {
    rows_a: u32,
    cols_a: u32,
    rows_b: u32,
    cols_b: u32,
    batch_size: u32,
    tolerance: f32,
    max_iterations: u32,
    padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> metadata: LinalgMetadata;

// Shared memory tile for efficient transpose
var<workgroup> tile: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let rows = metadata.rows_a;
    let cols = metadata.cols_a;
    
    // Calculate global position in input matrix
    let input_row = workgroup_id.y * 16u + local_id.y;
    let input_col = workgroup_id.x * 16u + local_id.x;
    
    // Load data into shared memory tile
    if (input_row < rows && input_col < cols) {
        let input_idx = input_row * cols + input_col;
        tile[local_id.y][local_id.x] = input[input_idx];
    } else {
        tile[local_id.y][local_id.x] = 0.0;
    }
    
    // Synchronize workgroup
    workgroupBarrier();
    
    // Calculate output position (transposed)
    let output_row = workgroup_id.x * 16u + local_id.y;
    let output_col = workgroup_id.y * 16u + local_id.x;
    
    // Write transposed data from shared memory
    if (output_row < cols && output_col < rows) {
        let output_idx = output_row * rows + output_col;
        output[output_idx] = tile[local_id.x][local_id.y];
    }
}