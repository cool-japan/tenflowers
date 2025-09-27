// GPU kernels for RNN operations (LSTM, GRU)

// LSTM cell computation with fused operations
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> hidden_data: array<f32>;
@group(0) @binding(2) var<storage, read> cell_data: array<f32>;
@group(0) @binding(3) var<storage, read> w_ih_data: array<f32>;
@group(0) @binding(4) var<storage, read> w_hh_data: array<f32>;
@group(0) @binding(5) var<storage, read> b_ih_data: array<f32>;
@group(0) @binding(6) var<storage, read> b_hh_data: array<f32>;
@group(0) @binding(7) var<storage, read_write> output_hidden: array<f32>;
@group(0) @binding(8) var<storage, read_write> output_cell: array<f32>;

// Uniform buffer for dimensions
struct LSTMParams {
    batch_size: u32,
    input_size: u32,
    hidden_size: u32,
    has_bias: u32,
}

@group(1) @binding(0) var<uniform> params: LSTMParams;

// Sigmoid activation function
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// Tanh activation function
fn tanh_activation(x: f32) -> f32 {
    let exp_2x = exp(2.0 * x);
    return (exp_2x - 1.0) / (exp_2x + 1.0);
}

// LSTM cell computation kernel
@compute @workgroup_size(256)
fn lstm_cell_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let hidden_idx = global_id.y;
    
    if (batch_idx >= params.batch_size || hidden_idx >= params.hidden_size) {
        return;
    }
    
    let batch_offset = batch_idx * params.hidden_size;
    let input_offset = batch_idx * params.input_size;
    
    // Compute input-to-hidden and hidden-to-hidden projections for all gates
    // Gates are arranged as: input_gate, forget_gate, cell_gate, output_gate
    var i_gate = 0.0;
    var f_gate = 0.0;
    var c_gate = 0.0;
    var o_gate = 0.0;
    
    // Input-to-hidden projection
    for (var i = 0u; i < params.input_size; i++) {
        let input_val = input_data[input_offset + i];
        let ih_base = i * params.hidden_size * 4u;
        
        i_gate += input_val * w_ih_data[ih_base + hidden_idx];
        f_gate += input_val * w_ih_data[ih_base + params.hidden_size + hidden_idx];
        c_gate += input_val * w_ih_data[ih_base + params.hidden_size * 2u + hidden_idx];
        o_gate += input_val * w_ih_data[ih_base + params.hidden_size * 3u + hidden_idx];
    }
    
    // Hidden-to-hidden projection
    for (var h = 0u; h < params.hidden_size; h++) {
        let hidden_val = hidden_data[batch_offset + h];
        let hh_base = h * params.hidden_size * 4u;
        
        i_gate += hidden_val * w_hh_data[hh_base + hidden_idx];
        f_gate += hidden_val * w_hh_data[hh_base + params.hidden_size + hidden_idx];
        c_gate += hidden_val * w_hh_data[hh_base + params.hidden_size * 2u + hidden_idx];
        o_gate += hidden_val * w_hh_data[hh_base + params.hidden_size * 3u + hidden_idx];
    }
    
    // Add bias if present
    if (params.has_bias == 1u) {
        i_gate += b_ih_data[hidden_idx] + b_hh_data[hidden_idx];
        f_gate += b_ih_data[params.hidden_size + hidden_idx] + b_hh_data[params.hidden_size + hidden_idx];
        c_gate += b_ih_data[params.hidden_size * 2u + hidden_idx] + b_hh_data[params.hidden_size * 2u + hidden_idx];
        o_gate += b_ih_data[params.hidden_size * 3u + hidden_idx] + b_hh_data[params.hidden_size * 3u + hidden_idx];
    }
    
    // Apply activation functions
    let i_activated = sigmoid(i_gate);      // Input gate
    let f_activated = sigmoid(f_gate);      // Forget gate
    let c_activated = tanh_activation(c_gate); // Cell gate
    let o_activated = sigmoid(o_gate);      // Output gate
    
    // Compute new cell state: C_t = f_t * C_{t-1} + i_t * g_t
    let prev_cell = cell_data[batch_offset + hidden_idx];
    let new_cell = f_activated * prev_cell + i_activated * c_activated;
    
    // Compute new hidden state: h_t = o_t * tanh(C_t)
    let new_hidden = o_activated * tanh_activation(new_cell);
    
    // Store results
    output_hidden[batch_offset + hidden_idx] = new_hidden;
    output_cell[batch_offset + hidden_idx] = new_cell;
}

// GRU cell computation kernel
@compute @workgroup_size(256)
fn gru_cell_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let hidden_idx = global_id.y;
    
    if (batch_idx >= params.batch_size || hidden_idx >= params.hidden_size) {
        return;
    }
    
    let batch_offset = batch_idx * params.hidden_size;
    let input_offset = batch_idx * params.input_size;
    
    // Compute input-to-hidden and hidden-to-hidden projections for gates
    // Gates are arranged as: reset_gate, update_gate, new_gate
    var r_gate = 0.0;
    var z_gate = 0.0;
    var n_gate = 0.0;
    
    // Input-to-hidden projection
    for (var i = 0u; i < params.input_size; i++) {
        let input_val = input_data[input_offset + i];
        let ih_base = i * params.hidden_size * 3u;
        
        r_gate += input_val * w_ih_data[ih_base + hidden_idx];
        z_gate += input_val * w_ih_data[ih_base + params.hidden_size + hidden_idx];
        n_gate += input_val * w_ih_data[ih_base + params.hidden_size * 2u + hidden_idx];
    }
    
    // Hidden-to-hidden projection for reset and update gates
    let prev_hidden = hidden_data[batch_offset + hidden_idx];
    for (var h = 0u; h < params.hidden_size; h++) {
        let hidden_val = hidden_data[batch_offset + h];
        let hh_base = h * params.hidden_size * 3u;
        
        r_gate += hidden_val * w_hh_data[hh_base + hidden_idx];
        z_gate += hidden_val * w_hh_data[hh_base + params.hidden_size + hidden_idx];
    }
    
    // Add bias if present
    if (params.has_bias == 1u) {
        r_gate += b_ih_data[hidden_idx] + b_hh_data[hidden_idx];
        z_gate += b_ih_data[params.hidden_size + hidden_idx] + b_hh_data[params.hidden_size + hidden_idx];
        n_gate += b_ih_data[params.hidden_size * 2u + hidden_idx];
    }
    
    // Apply activation functions
    let r_activated = sigmoid(r_gate);      // Reset gate
    let z_activated = sigmoid(z_gate);      // Update gate
    
    // Compute new gate with reset applied to hidden state
    let reset_hidden = r_activated * prev_hidden;
    for (var h = 0u; h < params.hidden_size; h++) {
        let hidden_val = reset_hidden;
        let hh_base = h * params.hidden_size * 3u;
        n_gate += hidden_val * w_hh_data[hh_base + params.hidden_size * 2u + hidden_idx];
    }
    
    if (params.has_bias == 1u) {
        n_gate += b_hh_data[params.hidden_size * 2u + hidden_idx];
    }
    
    let n_activated = tanh_activation(n_gate); // New gate
    
    // Compute new hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * n_t
    let new_hidden = (1.0 - z_activated) * prev_hidden + z_activated * n_activated;
    
    // Store result
    output_hidden[batch_offset + hidden_idx] = new_hidden;
}

// Batched LSTM computation for multiple time steps
@compute @workgroup_size(256)
fn lstm_sequence_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let hidden_idx = global_id.y;
    let seq_idx = global_id.z;
    
    if (batch_idx >= params.batch_size || hidden_idx >= params.hidden_size) {
        return;
    }
    
    // This kernel would process entire sequences in parallel
    // Implementation would be more complex as it needs to handle dependencies
    // For now, we'll use the single-step kernel in a loop
}

// Layer normalization kernel for RNN outputs
@compute @workgroup_size(256)
fn rnn_layer_norm_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let feature_idx = global_id.y;
    
    if (batch_idx >= params.batch_size || feature_idx >= params.hidden_size) {
        return;
    }
    
    let batch_offset = batch_idx * params.hidden_size;
    
    // Compute mean
    var sum = 0.0;
    for (var i = 0u; i < params.hidden_size; i++) {
        sum += output_hidden[batch_offset + i];
    }
    let mean = sum / f32(params.hidden_size);
    
    // Compute variance
    var var_sum = 0.0;
    for (var i = 0u; i < params.hidden_size; i++) {
        let diff = output_hidden[batch_offset + i] - mean;
        var_sum += diff * diff;
    }
    let variance = var_sum / f32(params.hidden_size);
    let std_dev = sqrt(variance + 1e-5);
    
    // Normalize
    let normalized = (output_hidden[batch_offset + feature_idx] - mean) / std_dev;
    output_hidden[batch_offset + feature_idx] = normalized;
}