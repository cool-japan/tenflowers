// Embedding operations compute shaders

// Parameters for embedding lookup
struct EmbeddingParams {
    num_embeddings: u32,
    embedding_dim: u32,
    batch_size: u32,
    sequence_length: u32,
}

@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read> embedding_table: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: EmbeddingParams;

// Embedding lookup kernel - optimized workgroup size
@compute @workgroup_size(256, 1, 1)
fn embedding_lookup_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_indices = params.batch_size * params.sequence_length;
    let index_id = global_id.x;
    
    if (index_id >= total_indices) {
        return;
    }
    
    // Get the embedding index for this position
    let embedding_index = indices[index_id];
    
    // Bounds check
    if (embedding_index >= params.num_embeddings) {
        return; // Invalid index, output will remain zero
    }
    
    // Copy embedding vector to output
    let embedding_start = embedding_index * params.embedding_dim;
    let output_start = index_id * params.embedding_dim;
    
    for (var i: u32 = 0u; i < params.embedding_dim; i++) {
        output[output_start + i] = embedding_table[embedding_start + i];
    }
}

// Optimized embedding lookup with coalesced memory access
@compute @workgroup_size(8, 8, 1)
fn embedding_lookup_coalesced_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let seq_idx = global_id.y;
    let embedding_dim_chunk = global_id.z;
    
    if (batch_idx >= params.batch_size || 
        seq_idx >= params.sequence_length) {
        return;
    }
    
    let index_pos = batch_idx * params.sequence_length + seq_idx;
    let embedding_index = indices[index_pos];
    
    // Bounds check
    if (embedding_index >= params.num_embeddings) {
        return;
    }
    
    // Process embedding dimensions in chunks for better memory access
    let chunk_size = 8u;
    let start_dim = embedding_dim_chunk * chunk_size;
    let end_dim = min(start_dim + chunk_size, params.embedding_dim);
    
    let embedding_base = embedding_index * params.embedding_dim;
    let output_base = index_pos * params.embedding_dim;
    
    for (var dim: u32 = start_dim; dim < end_dim; dim++) {
        output[output_base + dim] = embedding_table[embedding_base + dim];
    }
}

// Sparse embedding lookup for high-dimensional embeddings
struct SparseEmbeddingParams {
    num_embeddings: u32,
    embedding_dim: u32,
    batch_size: u32,
    sequence_length: u32,
    active_indices: u32,  // Number of non-zero indices
}

@group(0) @binding(4) var<uniform> sparse_params: SparseEmbeddingParams;
@group(0) @binding(5) var<storage, read> active_embedding_indices: array<u32>;

// Sparse embedding lookup - only processes non-zero indices
@compute @workgroup_size(32, 1, 1)
fn sparse_embedding_lookup_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let active_idx = global_id.x;
    
    if (active_idx >= sparse_params.active_indices) {
        return;
    }
    
    // Get the actual embedding index from the active indices list
    let embedding_index = active_embedding_indices[active_idx];
    
    if (embedding_index >= sparse_params.num_embeddings) {
        return;
    }
    
    // This kernel assumes the output indices correspond to the active indices
    let embedding_start = embedding_index * sparse_params.embedding_dim;
    let output_start = active_idx * sparse_params.embedding_dim;
    
    for (var i: u32 = 0u; i < sparse_params.embedding_dim; i++) {
        output[output_start + i] = embedding_table[embedding_start + i];
    }
}

// Embedding gradient accumulation for training
struct EmbeddingGradParams {
    num_embeddings: u32,
    embedding_dim: u32,
    batch_size: u32,
    sequence_length: u32,
    learning_rate: f32,
}

@group(0) @binding(6) var<uniform> grad_params: EmbeddingGradParams;
@group(0) @binding(7) var<storage, read> output_gradients: array<f32>;
@group(0) @binding(8) var<storage, read_write> embedding_gradients: array<f32>;

// Accumulate gradients for embedding training
@compute @workgroup_size(64, 1, 1)
fn embedding_gradient_accumulation_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_indices = grad_params.batch_size * grad_params.sequence_length;
    let index_id = global_id.x;
    
    if (index_id >= total_indices) {
        return;
    }
    
    let embedding_index = indices[index_id];
    
    if (embedding_index >= grad_params.num_embeddings) {
        return;
    }
    
    // Accumulate gradients for this embedding
    let grad_input_start = index_id * grad_params.embedding_dim;
    let grad_embedding_start = embedding_index * grad_params.embedding_dim;
    
    for (var i: u32 = 0u; i < grad_params.embedding_dim; i++) {
        // Atomic add for thread safety
        atomicAdd(&embedding_gradients[grad_embedding_start + i], 
                 output_gradients[grad_input_start + i]);
    }
}