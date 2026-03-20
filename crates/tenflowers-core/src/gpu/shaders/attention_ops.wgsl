// Multi-Head Attention GPU Compute Shaders for TenfloweRS
// Implements efficient scaled dot-product attention and variants

// Workgroup size for compute operations
const WORKGROUP_SIZE: u32 = 256u;

// Parameters structure for attention computation
struct AttentionParams {
    seq_len: u32,
    head_dim: u32,
    scale: f32,      // Scale factor (1.0 / sqrt(head_dim)) stored as bits
    use_mask: u32,   // Whether to apply attention mask
}

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: AttentionParams;

// Shared memory for workgroup-local computations
var<workgroup> shared_scores: array<f32, 1024>;  // Attention scores cache
var<workgroup> shared_weights: array<f32, 1024>; // Softmax weights cache

/// Scaled Dot-Product Attention Kernel
/// Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
@compute @workgroup_size(WORKGROUP_SIZE)
fn scaled_dot_product_attention_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let seq_idx = global_id.x;
    let seq_len = params.seq_len;
    let head_dim = params.head_dim;
    
    // Early exit for out-of-bounds threads
    if (seq_idx >= seq_len) {
        return;
    }
    
    let local_id = global_id.x % WORKGROUP_SIZE;
    let workgroup_id = global_id.x / WORKGROUP_SIZE;
    
    // Step 1: Compute attention scores (Q * K^T)
    // Each thread processes one query position against all key positions
    var max_score = -3.4028235e38; // -FLT_MAX for numerical stability
    
    for (var k_idx: u32 = 0u; k_idx < seq_len; k_idx++) {
        var score = 0.0;
        
        // Compute dot product between query[seq_idx] and key[k_idx]
        for (var d: u32 = 0u; d < head_dim; d++) {
            let q_val = query[seq_idx * head_dim + d];
            let k_val = key[k_idx * head_dim + d];
            score += q_val * k_val;
        }
        
        // Apply scale factor
        let scale_factor = bitcast<f32>(params.scale);
        score *= scale_factor;
        
        // Apply mask if enabled
        if (params.use_mask != 0u) {
            let mask_val = mask[seq_idx * seq_len + k_idx];
            score += mask_val;
        }
        
        // Store in shared memory and track maximum for numerical stability
        if (k_idx < 1024u) {
            shared_scores[k_idx] = score;
        }
        max_score = max(max_score, score);
    }
    
    workgroupBarrier();
    
    // Step 2: Compute softmax weights
    var sum_weights = 0.0;
    
    for (var k_idx: u32 = 0u; k_idx < seq_len; k_idx++) {
        let score = select(
            shared_scores[k_idx],
            query[seq_idx * head_dim + k_idx % head_dim] * key[k_idx * head_dim],
            k_idx < 1024u
        );
        
        // Numerically stable softmax: exp(score - max_score)
        let weight = exp(score - max_score);
        sum_weights += weight;
        
        if (k_idx < 1024u) {
            shared_weights[k_idx] = weight;
        }
    }
    
    workgroupBarrier();
    
    // Step 3: Apply attention weights to values
    for (var d: u32 = 0u; d < head_dim; d++) {
        var attended_value = 0.0;
        
        for (var k_idx: u32 = 0u; k_idx < seq_len; k_idx++) {
            let weight = select(
                shared_weights[k_idx],
                exp(shared_scores[k_idx] - max_score),
                k_idx < 1024u
            ) / sum_weights;
            
            let v_val = value[k_idx * head_dim + d];
            attended_value += weight * v_val;
        }
        
        // Store final output
        output[seq_idx * head_dim + d] = attended_value;
    }
}

/// Multi-Head Attention Kernel
/// Processes multiple attention heads in parallel
@compute @workgroup_size(WORKGROUP_SIZE)
fn multi_head_attention_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // For now, delegate to scaled dot-product attention
    // In a full implementation, you'd want head-specific processing
    scaled_dot_product_attention_kernel(global_id);
}

/// Flash Attention Kernel (Memory-Efficient)
/// Implements block-wise attention computation for large sequences
@compute @workgroup_size(WORKGROUP_SIZE)
fn flash_attention_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let seq_idx = global_id.x;
    let seq_len = params.seq_len;
    let head_dim = params.head_dim;
    
    if (seq_idx >= seq_len) {
        return;
    }
    
    // Flash attention uses block-wise processing for memory efficiency
    // For simplicity, we'll use the same computation as scaled dot-product
    // In a full implementation, you'd want:
    // 1. Block-wise loading of K, V matrices
    // 2. Online softmax computation
    // 3. Incremental output accumulation
    
    let local_id = global_id.x % WORKGROUP_SIZE;
    let block_size = 256u; // Block size for Flash attention
    
    var running_max = -3.4028235e38;
    var running_sum = 0.0;
    
    // Process blocks of keys/values
    let num_blocks = (seq_len + block_size - 1u) / block_size;
    
    for (var block: u32 = 0u; block < num_blocks; block++) {
        let block_start = block * block_size;
        let block_end = min(block_start + block_size, seq_len);
        
        // Process this block
        var block_max = -3.4028235e38;
        
        for (var k_idx: u32 = block_start; k_idx < block_end; k_idx++) {
            var score = 0.0;
            
            // Compute attention score
            for (var d: u32 = 0u; d < head_dim; d++) {
                score += query[seq_idx * head_dim + d] * key[k_idx * head_dim + d];
            }
            
            score *= bitcast<f32>(params.scale);
            
            if (params.use_mask != 0u) {
                score += mask[seq_idx * seq_len + k_idx];
            }
            
            block_max = max(block_max, score);
        }
        
        // Update running statistics
        let old_max = running_max;
        running_max = max(running_max, block_max);
        
        // Correction factor for numerical stability
        let correction = exp(old_max - running_max);
        running_sum *= correction;
        
        // Add block contribution
        for (var k_idx: u32 = block_start; k_idx < block_end; k_idx++) {
            var score = 0.0;
            for (var d: u32 = 0u; d < head_dim; d++) {
                score += query[seq_idx * head_dim + d] * key[k_idx * head_dim + d];
            }
            score *= bitcast<f32>(params.scale);
            
            if (params.use_mask != 0u) {
                score += mask[seq_idx * seq_len + k_idx];
            }
            
            let weight = exp(score - running_max);
            running_sum += weight;
        }
    }
    
    // Final pass: compute output values
    for (var d: u32 = 0u; d < head_dim; d++) {
        var attended_value = 0.0;
        
        for (var k_idx: u32 = 0u; k_idx < seq_len; k_idx++) {
            var score = 0.0;
            for (var dim: u32 = 0u; dim < head_dim; dim++) {
                score += query[seq_idx * head_dim + dim] * key[k_idx * head_dim + dim];
            }
            score *= bitcast<f32>(params.scale);
            
            if (params.use_mask != 0u) {
                score += mask[seq_idx * seq_len + k_idx];
            }
            
            let weight = exp(score - running_max) / running_sum;
            attended_value += weight * value[k_idx * head_dim + d];
        }
        
        output[seq_idx * head_dim + d] = attended_value;
    }
}

/// Memory-Optimized Flash Attention Kernel
/// Implements true O(N) memory complexity Flash Attention with tiled computation
@compute @workgroup_size(128, 1, 1)
fn flash_attention_optimized(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let seq_idx = global_id.x;
    let seq_len = params.seq_len;
    let head_dim = params.head_dim;
    let block_size = 64u; // Tile size for memory efficiency
    
    if (seq_idx >= seq_len) {
        return;
    }
    
    // Local storage for query vector (reused across blocks)
    var query_vec: array<f32, 128>;
    
    // Load query vector for this sequence position
    for (var d = 0u; d < min(head_dim, 128u); d++) {
        if (seq_idx * head_dim + d < arrayLength(&query)) {
            query_vec[d] = query[seq_idx * head_dim + d];
        } else {
            query_vec[d] = 0.0;
        }
    }
    
    // Online softmax state
    var row_max = -3.4028235e38;
    var row_sum = 0.0;
    var output_acc: array<f32, 128>;
    
    // Initialize output accumulator
    for (var d = 0u; d < min(head_dim, 128u); d++) {
        output_acc[d] = 0.0;
    }
    
    let scale_factor = bitcast<f32>(params.scale);
    let num_blocks = (seq_len + block_size - 1u) / block_size;
    
    // Process each block of keys and values
    for (var block_idx = 0u; block_idx < num_blocks; block_idx++) {
        let block_start = block_idx * block_size;
        let block_end = min(block_start + block_size, seq_len);
        let current_block_size = block_end - block_start;
        
        // Local storage for current block scores
        var block_scores: array<f32, 64>;
        var block_max = -3.4028235e38;
        
        // Compute attention scores for current block
        for (var k_offset = 0u; k_offset < current_block_size; k_offset++) {
            let k_idx = block_start + k_offset;
            var score = 0.0;
            
            // Compute Q * K^T for this key
            for (var d = 0u; d < min(head_dim, 128u); d++) {
                if (k_idx * head_dim + d < arrayLength(&key)) {
                    score += query_vec[d] * key[k_idx * head_dim + d];
                }
            }
            
            score *= scale_factor;
            
            // Apply causal mask if needed
            if (params.use_mask != 0u) {
                if (k_idx > seq_idx) {
                    score = -3.4028235e38; // Negative infinity for causal masking
                } else if (k_idx * seq_len + seq_idx < arrayLength(&mask)) {
                    score += mask[seq_idx * seq_len + k_idx];
                }
            }
            
            block_scores[k_offset] = score;
            block_max = max(block_max, score);
        }
        
        // Update online softmax statistics
        let new_max = max(row_max, block_max);
        let exp_diff_old = exp(row_max - new_max);
        let exp_diff_block = exp(block_max - new_max);
        
        // Scale previous output accumulator
        for (var d = 0u; d < min(head_dim, 128u); d++) {
            output_acc[d] *= exp_diff_old;
        }
        
        // Compute block contributions
        var block_sum = 0.0;
        
        for (var k_offset = 0u; k_offset < current_block_size; k_offset++) {
            let k_idx = block_start + k_offset;
            let score_exp = exp(block_scores[k_offset] - new_max);
            block_sum += score_exp;
            
            // Accumulate weighted values
            for (var d = 0u; d < min(head_dim, 128u); d++) {
                if (k_idx * head_dim + d < arrayLength(&value)) {
                    output_acc[d] += score_exp * value[k_idx * head_dim + d];
                }
            }
        }
        
        // Update global statistics
        row_sum = row_sum * exp_diff_old + block_sum * exp_diff_block;
        row_max = new_max;
    }
    
    // Write normalized output
    for (var d = 0u; d < min(head_dim, 128u); d++) {
        let out_idx = seq_idx * head_dim + d;
        if (out_idx < arrayLength(&output)) {
            output[out_idx] = output_acc[d] / row_sum;
        }
    }
}

/// Grouped Query Attention (GQA) Kernel
/// Optimized for models with grouped key-value heads
@compute @workgroup_size(64, 1, 1)
fn grouped_query_attention(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let seq_idx = global_id.x;
    let head_idx = global_id.y;
    let seq_len = params.seq_len;
    let head_dim = params.head_dim;
    let num_query_heads = 32u; // Example: 32 query heads
    let num_kv_heads = 8u;     // Example: 8 key-value heads
    let group_size = num_query_heads / num_kv_heads;
    
    if (seq_idx >= seq_len || head_idx >= num_query_heads) {
        return;
    }
    
    // Find which KV head this query head belongs to
    let kv_head = head_idx / group_size;
    
    // Calculate offsets
    let q_offset = head_idx * seq_len * head_dim;
    let kv_offset = kv_head * seq_len * head_dim;
    let out_offset = head_idx * seq_len * head_dim;
    
    // Load query vector
    var query_vec: array<f32, 128>;
    for (var d = 0u; d < min(head_dim, 128u); d++) {
        let q_idx = q_offset + seq_idx * head_dim + d;
        if (q_idx < arrayLength(&query)) {
            query_vec[d] = query[q_idx];
        } else {
            query_vec[d] = 0.0;
        }
    }
    
    // Compute attention using shared KV head
    var max_score = -3.4028235e38;
    var sum_weights = 0.0;
    var output_vec: array<f32, 128>;
    
    for (var d = 0u; d < min(head_dim, 128u); d++) {
        output_vec[d] = 0.0;
    }
    
    let scale_factor = bitcast<f32>(params.scale);
    
    // First pass: compute max score for numerical stability
    for (var k_idx = 0u; k_idx < seq_len; k_idx++) {
        var score = 0.0;
        
        for (var d = 0u; d < min(head_dim, 128u); d++) {
            let k_global_idx = kv_offset + k_idx * head_dim + d;
            if (k_global_idx < arrayLength(&key)) {
                score += query_vec[d] * key[k_global_idx];
            }
        }
        
        score *= scale_factor;
        
        // Apply causal mask
        if (params.use_mask != 0u && k_idx > seq_idx) {
            score = -3.4028235e38;
        }
        
        max_score = max(max_score, score);
    }
    
    // Second pass: compute softmax and weighted sum
    for (var k_idx = 0u; k_idx < seq_len; k_idx++) {
        var score = 0.0;
        
        for (var d = 0u; d < min(head_dim, 128u); d++) {
            let k_global_idx = kv_offset + k_idx * head_dim + d;
            if (k_global_idx < arrayLength(&key)) {
                score += query_vec[d] * key[k_global_idx];
            }
        }
        
        score *= scale_factor;
        
        if (params.use_mask != 0u && k_idx > seq_idx) {
            score = -3.4028235e38;
        }
        
        let weight = exp(score - max_score);
        sum_weights += weight;
        
        // Accumulate weighted values
        for (var d = 0u; d < min(head_dim, 128u); d++) {
            let v_global_idx = kv_offset + k_idx * head_dim + d;
            if (v_global_idx < arrayLength(&value)) {
                output_vec[d] += weight * value[v_global_idx];
            }
        }
    }
    
    // Write normalized output
    for (var d = 0u; d < min(head_dim, 128u); d++) {
        let out_idx = out_offset + seq_idx * head_dim + d;
        if (out_idx < arrayLength(&output)) {
            output[out_idx] = output_vec[d] / sum_weights;
        }
    }
}

/// Utility function for stable softmax computation
fn stable_softmax(scores: ptr<function, array<f32, 1024>>, len: u32) -> array<f32, 1024> {
    var result: array<f32, 1024>;
    var max_val = -3.4028235e38;
    
    // Find maximum
    for (var i: u32 = 0u; i < len; i++) {
        max_val = max(max_val, (*scores)[i]);
    }
    
    // Compute exp and sum
    var sum_exp = 0.0;
    for (var i: u32 = 0u; i < len; i++) {
        let exp_val = exp((*scores)[i] - max_val);
        result[i] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize
    for (var i: u32 = 0u; i < len; i++) {
        result[i] /= sum_exp;
    }
    
    return result;
}