// Linear Algebra: Optimized Matrix Multiplication
//
// High-performance matrix multiplication using tiled algorithms and shared memory.
// Optimized for linear algebra operations with better cache efficiency than basic matmul.
// This implementation follows cuBLAS GEMM patterns adapted for WGPU.
//
// Computes: C = A * B
// Where: A [rows_a, cols_a], B [rows_b, cols_b], C [rows_a, cols_b]
// Requires: cols_a == rows_b

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

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> metadata: LinalgMetadata;

// Shared memory tiles for efficient computation
const TILE_SIZE: u32 = 16u;
var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let rows_a = metadata.rows_a;
    let cols_a = metadata.cols_a;
    let cols_b = metadata.cols_b;
    
    // Global output position
    let row = workgroup_id.y * TILE_SIZE + local_id.y;
    let col = workgroup_id.x * TILE_SIZE + local_id.x;
    
    var sum = 0.0;
    
    // Tile over the K dimension (cols_a == rows_b)
    let num_tiles = (cols_a + TILE_SIZE - 1u) / TILE_SIZE;
    
    for (var tile_idx = 0u; tile_idx < num_tiles; tile_idx = tile_idx + 1u) {
        // Load tile from matrix A
        let a_tile_row = row;
        let a_tile_col = tile_idx * TILE_SIZE + local_id.x;
        
        if (a_tile_row < rows_a && a_tile_col < cols_a) {
            let a_idx = a_tile_row * cols_a + a_tile_col;
            tile_a[local_id.y][local_id.x] = matrix_a[a_idx];
        } else {
            tile_a[local_id.y][local_id.x] = 0.0;
        }
        
        // Load tile from matrix B
        let b_tile_row = tile_idx * TILE_SIZE + local_id.y;
        let b_tile_col = col;
        
        if (b_tile_row < cols_a && b_tile_col < cols_b) {
            let b_idx = b_tile_row * cols_b + b_tile_col;
            tile_b[local_id.y][local_id.x] = matrix_b[b_idx];
        } else {
            tile_b[local_id.y][local_id.x] = 0.0;
        }
        
        // Synchronize to ensure tiles are loaded
        workgroupBarrier();
        
        // Compute partial dot product for this tile
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[local_id.y][k] * tile_b[k][local_id.x];
        }
        
        // Synchronize before loading next tile
        workgroupBarrier();
    }
    
    // Write result to output matrix
    if (row < rows_a && col < cols_b) {
        let c_idx = row * cols_b + col;
        matrix_c[c_idx] = sum;
    }
}