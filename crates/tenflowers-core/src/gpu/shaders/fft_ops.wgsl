// FFT compute shaders for GPU operations
// Implements Cooley-Tukey FFT algorithm for parallel computation

// Input/output buffers for complex numbers (stored as f32 pairs: [real, imag, real, imag, ...])
@group(0) @binding(0) var<storage, read> input_complex: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_complex: array<f32>;

// FFT parameters
struct FFTInfo {
    n: u32,           // Size of FFT
    log2_n: u32,      // log2(n) for bit reversal
    batch_size: u32,  // Number of FFTs to compute
    is_inverse: u32,  // 1 for inverse FFT, 0 for forward FFT
}

@group(0) @binding(2) var<uniform> fft_info: FFTInfo;

// Twiddle factors (precomputed complex exponentials)
@group(0) @binding(3) var<storage, read> twiddle_factors: array<f32>;

// Bit reversal lookup table
@group(0) @binding(4) var<storage, read> bit_reversal_table: array<u32>;

// Complex number operations
fn complex_mul(a_real: f32, a_imag: f32, b_real: f32, b_imag: f32) -> vec2<f32> {
    return vec2<f32>(
        a_real * b_real - a_imag * b_imag,
        a_real * b_imag + a_imag * b_real
    );
}

fn complex_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x + b.x, a.y + b.y);
}

fn complex_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x - b.x, a.y - b.y);
}

// Helper function to compute log2 of a u32
fn log2_u32(x: u32) -> u32 {
    return 31u - countLeadingZeros(x);
}

// Helper function to compute bit reversal index
fn bit_reverse_index(x: u32, log2_n: u32) -> u32 {
    var result = 0u;
    var temp = x;
    
    for (var i = 0u; i < log2_n; i++) {
        result = (result << 1u) | (temp & 1u);
        temp >>= 1u;
    }
    
    return result;
}

// Radix-2 FFT butterfly operation
fn butterfly(data: ptr<function, array<vec2<f32>, 1024>>, stage: u32, k: u32) {
    let n = fft_info.n;
    let stride = 1u << stage;
    let half_stride = stride >> 1u;
    
    for (var i = 0u; i < n; i += stride) {
        let twiddle_idx = (k % half_stride) * (n / stride);
        let twiddle_real = twiddle_factors[twiddle_idx * 2u];
        let twiddle_imag = twiddle_factors[twiddle_idx * 2u + 1u];
        
        let idx1 = i + k;
        let idx2 = idx1 + half_stride;
        
        if (idx1 < n && idx2 < n) {
            let temp = complex_mul((*data)[idx2].x, (*data)[idx2].y, twiddle_real, twiddle_imag);
            (*data)[idx2] = complex_sub((*data)[idx1], temp);
            (*data)[idx1] = complex_add((*data)[idx1], temp);
        }
    }
}

// 1D FFT kernel (radix-2 decimation-in-time)
@compute @workgroup_size(32)
fn fft_1d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    
    if (batch_idx >= fft_info.batch_size) {
        return;
    }
    
    let n = fft_info.n;
    let batch_offset = batch_idx * n * 2u;
    
    // Shared memory for local FFT computation
    var local_data: array<vec2<f32>, 1024>;
    
    // Copy input data to local memory with bit reversal
    for (var i = 0u; i < n; i++) {
        let bit_rev_idx = bit_reversal_table[i];
        let input_idx = batch_offset + bit_rev_idx * 2u;
        
        if (input_idx + 1u < arrayLength(&input_complex)) {
            local_data[i] = vec2<f32>(
                input_complex[input_idx],
                input_complex[input_idx + 1u]
            );
        }
    }
    
    // FFT computation using Cooley-Tukey algorithm
    for (var stage = 1u; stage <= fft_info.log2_n; stage++) {
        let m = 1u << stage;
        let half_m = m >> 1u;
        
        for (var i = 0u; i < n; i += m) {
            for (var j = 0u; j < half_m; j++) {
                let idx1 = i + j;
                let idx2 = idx1 + half_m;
                
                if (idx1 < n && idx2 < n) {
                    // Twiddle factor index
                    let twiddle_idx = j * (n / m);
                    let twiddle_real = twiddle_factors[twiddle_idx * 2u];
                    let twiddle_imag = twiddle_factors[twiddle_idx * 2u + 1u];
                    
                    // Apply twiddle factor
                    let temp = complex_mul(local_data[idx2].x, local_data[idx2].y, twiddle_real, twiddle_imag);
                    
                    // Butterfly operation
                    let u = local_data[idx1];
                    local_data[idx1] = complex_add(u, temp);
                    local_data[idx2] = complex_sub(u, temp);
                }
            }
        }
    }
    
    // Copy result back to output buffer
    for (var i = 0u; i < n; i++) {
        let output_idx = batch_offset + i * 2u;
        
        if (output_idx + 1u < arrayLength(&output_complex)) {
            var result = local_data[i];
            
            // For inverse FFT, normalize by n
            if (fft_info.is_inverse != 0u) {
                let scale = 1.0 / f32(n);
                result = vec2<f32>(result.x * scale, result.y * scale);
            }
            
            output_complex[output_idx] = result.x;
            output_complex[output_idx + 1u] = result.y;
        }
    }
}

// 2D FFT kernel (applies 1D FFT along rows then columns)
@compute @workgroup_size(16, 16)
fn fft_2d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z;
    let row = global_id.y;
    let col = global_id.x;
    
    if (batch_idx >= fft_info.batch_size) {
        return;
    }
    
    let height = fft_info.n >> 16u;  // Upper 16 bits
    let width = fft_info.n & 0xFFFFu;  // Lower 16 bits
    
    if (row >= height || col >= width) {
        return;
    }
    
    let batch_offset = batch_idx * height * width * 2u;
    
    // Step 1: FFT along rows
    if (col == 0u) {
        var row_data: array<vec2<f32>, 1024>;
        
        // Load row data
        for (var w = 0u; w < width; w++) {
            let idx = batch_offset + (row * width + w) * 2u;
            row_data[w] = vec2<f32>(
                input_complex[idx],
                input_complex[idx + 1u]
            );
        }
        
        // Apply 1D FFT to row using Cooley-Tukey algorithm
        // Bit reversal
        var temp_row: array<vec2<f32>, 1024>;
        for (var w = 0u; w < width; w++) {
            let bit_rev_idx = bit_reverse_index(w, log2_u32(width));
            temp_row[w] = row_data[bit_rev_idx];
        }
        
        // FFT stages
        let log2_width = log2_u32(width);
        for (var stage = 1u; stage <= log2_width; stage++) {
            let m = 1u << stage;
            let half_m = m >> 1u;
            
            for (var i = 0u; i < width; i += m) {
                for (var j = 0u; j < half_m; j++) {
                    let idx1 = i + j;
                    let idx2 = idx1 + half_m;
                    
                    if (idx1 < width && idx2 < width) {
                        let twiddle_idx = j * (width / m);
                        let angle = -2.0 * 3.14159265359 * f32(twiddle_idx) / f32(width);
                        let twiddle = vec2<f32>(cos(angle), sin(angle));
                        
                        let temp = complex_mul(temp_row[idx2].x, temp_row[idx2].y, twiddle.x, twiddle.y);
                        let u = temp_row[idx1];
                        temp_row[idx1] = complex_add(u, temp);
                        temp_row[idx2] = complex_sub(u, temp);
                    }
                }
            }
        }
        
        // Store result
        for (var w = 0u; w < width; w++) {
            let idx = batch_offset + (row * width + w) * 2u;
            output_complex[idx] = temp_row[w].x;
            output_complex[idx + 1u] = temp_row[w].y;
        }
    }
    
    workgroupBarrier();
    
    // Step 2: FFT along columns
    if (row == 0u) {
        var col_data: array<vec2<f32>, 1024>;
        
        // Load column data from intermediate result
        for (var h = 0u; h < height; h++) {
            let idx = batch_offset + (h * width + col) * 2u;
            col_data[h] = vec2<f32>(
                output_complex[idx],
                output_complex[idx + 1u]
            );
        }
        
        // Apply 1D FFT to column using Cooley-Tukey algorithm
        // Bit reversal
        var temp_col: array<vec2<f32>, 1024>;
        for (var h = 0u; h < height; h++) {
            let bit_rev_idx = bit_reverse_index(h, log2_u32(height));
            temp_col[h] = col_data[bit_rev_idx];
        }
        
        // FFT stages
        let log2_height = log2_u32(height);
        for (var stage = 1u; stage <= log2_height; stage++) {
            let m = 1u << stage;
            let half_m = m >> 1u;
            
            for (var i = 0u; i < height; i += m) {
                for (var j = 0u; j < half_m; j++) {
                    let idx1 = i + j;
                    let idx2 = idx1 + half_m;
                    
                    if (idx1 < height && idx2 < height) {
                        let twiddle_idx = j * (height / m);
                        let angle = -2.0 * 3.14159265359 * f32(twiddle_idx) / f32(height);
                        let twiddle = vec2<f32>(cos(angle), sin(angle));
                        
                        let temp = complex_mul(temp_col[idx2].x, temp_col[idx2].y, twiddle.x, twiddle.y);
                        let u = temp_col[idx1];
                        temp_col[idx1] = complex_add(u, temp);
                        temp_col[idx2] = complex_sub(u, temp);
                    }
                }
            }
        }
        
        // Store final result
        for (var h = 0u; h < height; h++) {
            let idx = batch_offset + (h * width + col) * 2u;
            output_complex[idx] = temp_col[h].x;
            output_complex[idx + 1u] = temp_col[h].y;
        }
    }
}

// Real FFT kernel (for real input, only compute positive frequencies)
@compute @workgroup_size(32)
fn rfft_1d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    
    if (batch_idx >= fft_info.batch_size) {
        return;
    }
    
    let n = fft_info.n;
    let output_size = n / 2u + 1u;
    let batch_offset = batch_idx * n;
    let output_offset = batch_idx * output_size * 2u;
    
    // Shared memory for local FFT computation
    var local_data: array<vec2<f32>, 1024>;
    
    // Copy real input data to local memory (imaginary part = 0)
    for (var i = 0u; i < n; i++) {
        let input_idx = batch_offset + i;
        
        if (input_idx < arrayLength(&input_complex)) {
            local_data[i] = vec2<f32>(input_complex[input_idx], 0.0);
        }
    }
    
    // Perform FFT computation using Cooley-Tukey algorithm
    // Bit reversal
    var temp_data: array<vec2<f32>, 1024>;
    for (var i = 0u; i < n; i++) {
        let bit_rev_idx = bit_reverse_index(i, log2_u32(n));
        temp_data[i] = local_data[bit_rev_idx];
    }
    
    // FFT stages
    let log2_n = log2_u32(n);
    for (var stage = 1u; stage <= log2_n; stage++) {
        let m = 1u << stage;
        let half_m = m >> 1u;
        
        for (var i = 0u; i < n; i += m) {
            for (var j = 0u; j < half_m; j++) {
                let idx1 = i + j;
                let idx2 = idx1 + half_m;
                
                if (idx1 < n && idx2 < n) {
                    let twiddle_idx = j * (n / m);
                    let angle = -2.0 * 3.14159265359 * f32(twiddle_idx) / f32(n);
                    let twiddle = vec2<f32>(cos(angle), sin(angle));
                    
                    let temp = complex_mul(temp_data[idx2].x, temp_data[idx2].y, twiddle.x, twiddle.y);
                    let u = temp_data[idx1];
                    temp_data[idx1] = complex_add(u, temp);
                    temp_data[idx2] = complex_sub(u, temp);
                }
            }
        }
    }
    
    // Copy computed data back to local_data
    for (var i = 0u; i < n; i++) {
        local_data[i] = temp_data[i];
    }
    
    // Copy only positive frequencies to output
    for (var i = 0u; i < output_size; i++) {
        let output_idx = output_offset + i * 2u;
        
        if (output_idx + 1u < arrayLength(&output_complex)) {
            output_complex[output_idx] = local_data[i].x;
            output_complex[output_idx + 1u] = local_data[i].y;
        }
    }
}

// 3D FFT kernel (applies 1D FFT along each dimension)
@compute @workgroup_size(8, 8, 8)
fn fft_3d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z;
    let depth_idx = global_id.y;
    let height_idx = global_id.x;
    
    if (batch_idx >= fft_info.batch_size) {
        return;
    }
    
    // Extract dimensions from packed n value
    let depth = (fft_info.n >> 20u) & 0xFFFu;   // Bits 20-31
    let height = (fft_info.n >> 10u) & 0x3FFu;  // Bits 10-19
    let width = fft_info.n & 0x3FFu;            // Bits 0-9
    
    if (depth_idx >= depth || height_idx >= height) {
        return;
    }
    
    let batch_offset = batch_idx * depth * height * width * 2u;
    
    // Step 1: FFT along width (innermost dimension)
    var width_data: array<vec2<f32>, 1024>;
    
    // Load width slice
    for (var w = 0u; w < width; w++) {
        let idx = batch_offset + ((depth_idx * height + height_idx) * width + w) * 2u;
        width_data[w] = vec2<f32>(
            input_complex[idx],
            input_complex[idx + 1u]
        );
    }
    
    // Apply 1D FFT along width using Cooley-Tukey algorithm
    // Bit reversal
    var temp_width: array<vec2<f32>, 1024>;
    for (var w = 0u; w < width; w++) {
        let bit_rev_idx = bit_reverse_index(w, log2_u32(width));
        temp_width[w] = width_data[bit_rev_idx];
    }
    
    // FFT stages
    let log2_width = log2_u32(width);
    for (var stage = 1u; stage <= log2_width; stage++) {
        let m = 1u << stage;
        let half_m = m >> 1u;
        
        for (var i = 0u; i < width; i += m) {
            for (var j = 0u; j < half_m; j++) {
                let idx1 = i + j;
                let idx2 = idx1 + half_m;
                
                if (idx1 < width && idx2 < width) {
                    let twiddle_idx = j * (width / m);
                    let angle = -2.0 * 3.14159265359 * f32(twiddle_idx) / f32(width);
                    let twiddle = vec2<f32>(cos(angle), sin(angle));
                    
                    let temp = complex_mul(temp_width[idx2].x, temp_width[idx2].y, twiddle.x, twiddle.y);
                    let u = temp_width[idx1];
                    temp_width[idx1] = complex_add(u, temp);
                    temp_width[idx2] = complex_sub(u, temp);
                }
            }
        }
    }
    
    // Copy result back to width_data
    for (var w = 0u; w < width; w++) {
        width_data[w] = temp_width[w];
    }
    
    // Store intermediate result
    for (var w = 0u; w < width; w++) {
        let idx = batch_offset + ((depth_idx * height + height_idx) * width + w) * 2u;
        output_complex[idx] = width_data[w].x;
        output_complex[idx + 1u] = width_data[w].y;
    }
    
    workgroupBarrier();
    
    // Step 2: FFT along height
    if (depth_idx == 0u) {
        var height_data: array<vec2<f32>, 1024>;
        
        // Load height slice from intermediate result
        for (var h = 0u; h < height; h++) {
            let idx = batch_offset + ((depth_idx * height + h) * width + height_idx) * 2u;
            height_data[h] = vec2<f32>(
                output_complex[idx],
                output_complex[idx + 1u]
            );
        }
        
        // Apply 1D FFT along height
        var temp_height: array<vec2<f32>, 1024>;
        for (var h = 0u; h < height; h++) {
            let bit_rev_idx = bit_reverse_index(h, log2_u32(height));
            temp_height[h] = height_data[bit_rev_idx];
        }
        
        let log2_height = log2_u32(height);
        for (var stage = 1u; stage <= log2_height; stage++) {
            let m = 1u << stage;
            let half_m = m >> 1u;
            
            for (var i = 0u; i < height; i += m) {
                for (var j = 0u; j < half_m; j++) {
                    let idx1 = i + j;
                    let idx2 = idx1 + half_m;
                    
                    if (idx1 < height && idx2 < height) {
                        let twiddle_idx = j * (height / m);
                        let angle = -2.0 * 3.14159265359 * f32(twiddle_idx) / f32(height);
                        let twiddle = vec2<f32>(cos(angle), sin(angle));
                        
                        let temp = complex_mul(temp_height[idx2].x, temp_height[idx2].y, twiddle.x, twiddle.y);
                        let u = temp_height[idx1];
                        temp_height[idx1] = complex_add(u, temp);
                        temp_height[idx2] = complex_sub(u, temp);
                    }
                }
            }
        }
        
        // Store result
        for (var h = 0u; h < height; h++) {
            let idx = batch_offset + ((depth_idx * height + h) * width + height_idx) * 2u;
            output_complex[idx] = temp_height[h].x;
            output_complex[idx + 1u] = temp_height[h].y;
        }
    }
    
    workgroupBarrier();
    
    // Step 3: FFT along depth
    if (height_idx == 0u && depth_idx == 0u) {
        var depth_data: array<vec2<f32>, 1024>;
        
        // Load depth slice from intermediate result
        for (var d = 0u; d < depth; d++) {
            let idx = batch_offset + ((d * height + height_idx) * width + depth_idx) * 2u;
            depth_data[d] = vec2<f32>(
                output_complex[idx],
                output_complex[idx + 1u]
            );
        }
        
        // Apply 1D FFT along depth
        var temp_depth: array<vec2<f32>, 1024>;
        for (var d = 0u; d < depth; d++) {
            let bit_rev_idx = bit_reverse_index(d, log2_u32(depth));
            temp_depth[d] = depth_data[bit_rev_idx];
        }
        
        let log2_depth = log2_u32(depth);
        for (var stage = 1u; stage <= log2_depth; stage++) {
            let m = 1u << stage;
            let half_m = m >> 1u;
            
            for (var i = 0u; i < depth; i += m) {
                for (var j = 0u; j < half_m; j++) {
                    let idx1 = i + j;
                    let idx2 = idx1 + half_m;
                    
                    if (idx1 < depth && idx2 < depth) {
                        let twiddle_idx = j * (depth / m);
                        let angle = -2.0 * 3.14159265359 * f32(twiddle_idx) / f32(depth);
                        let twiddle = vec2<f32>(cos(angle), sin(angle));
                        
                        let temp = complex_mul(temp_depth[idx2].x, temp_depth[idx2].y, twiddle.x, twiddle.y);
                        let u = temp_depth[idx1];
                        temp_depth[idx1] = complex_add(u, temp);
                        temp_depth[idx2] = complex_sub(u, temp);
                    }
                }
            }
        }
        
        // Store final result
        for (var d = 0u; d < depth; d++) {
            let idx = batch_offset + ((d * height + height_idx) * width + depth_idx) * 2u;
            output_complex[idx] = temp_depth[d].x;
            output_complex[idx + 1u] = temp_depth[d].y;
        }
    }
}

// Specialized kernel for small FFTs that can fit in shared memory
@compute @workgroup_size(64)
fn fft_small(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    
    if (batch_idx >= fft_info.batch_size) {
        return;
    }
    
    let n = fft_info.n;
    let batch_offset = batch_idx * n * 2u;
    
    // For small FFTs (n <= 64), we can use a simpler approach
    var local_data: array<vec2<f32>, 64>;
    
    // Load input data
    for (var i = 0u; i < n; i++) {
        let input_idx = batch_offset + i * 2u;
        local_data[i] = vec2<f32>(
            input_complex[input_idx],
            input_complex[input_idx + 1u]
        );
    }
    
    // Simple DFT for small sizes
    for (var k = 0u; k < n; k++) {
        var sum = vec2<f32>(0.0, 0.0);
        
        for (var j = 0u; j < n; j++) {
            let angle = -2.0 * 3.14159265359 * f32(k * j) / f32(n);
            let twiddle = vec2<f32>(cos(angle), sin(angle));
            
            if (fft_info.is_inverse != 0u) {
                twiddle.y = -twiddle.y;
            }
            
            let product = complex_mul(local_data[j].x, local_data[j].y, twiddle.x, twiddle.y);
            sum = complex_add(sum, product);
        }
        
        if (fft_info.is_inverse != 0u) {
            sum = vec2<f32>(sum.x / f32(n), sum.y / f32(n));
        }
        
        let output_idx = batch_offset + k * 2u;
        output_complex[output_idx] = sum.x;
        output_complex[output_idx + 1u] = sum.y;
    }
}

// Utility kernel for bit reversal permutation
@compute @workgroup_size(64)
fn bit_reverse_permute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    
    if (batch_idx >= fft_info.batch_size) {
        return;
    }
    
    let n = fft_info.n;
    let batch_offset = batch_idx * n * 2u;
    
    // Apply bit reversal permutation
    for (var i = 0u; i < n; i++) {
        let bit_rev_idx = bit_reversal_table[i];
        
        if (i < bit_rev_idx) {
            let idx1 = batch_offset + i * 2u;
            let idx2 = batch_offset + bit_rev_idx * 2u;
            
            // Swap complex values
            let temp_real = output_complex[idx1];
            let temp_imag = output_complex[idx1 + 1u];
            
            output_complex[idx1] = output_complex[idx2];
            output_complex[idx1 + 1u] = output_complex[idx2 + 1u];
            
            output_complex[idx2] = temp_real;
            output_complex[idx2 + 1u] = temp_imag;
        }
    }
}

// In-place FFT along a specific axis with strided memory access
// Uses the regular FFT info structure, with stride=1 for simplicity in this version
@compute @workgroup_size(64)
fn fft_axis_inplace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    
    if (batch_idx >= fft_info.batch_size) {
        return;
    }
    
    let n = fft_info.n;
    let stride = 1u; // For now, assume unit stride (can be extended later)
    
    // Calculate base offset for this batch
    let base_offset = batch_idx * n * 2u;
    
    // Shared memory for local FFT computation
    var local_data: array<vec2<f32>, 1024>;
    
    // Copy input data to local memory with bit reversal 
    for (var i = 0u; i < n; i++) {
        let bit_rev_idx = bit_reversal_table[i];
        let input_idx = base_offset + bit_rev_idx * 2u;
        
        if (input_idx + 1u < arrayLength(&input_complex)) {
            local_data[i] = vec2<f32>(
                input_complex[input_idx],
                input_complex[input_idx + 1u]
            );
        }
    }
    
    // FFT computation using Cooley-Tukey algorithm
    for (var stage = 1u; stage <= fft_info.log2_n; stage++) {
        let m = 1u << stage;
        let half_m = m >> 1u;
        
        for (var i = 0u; i < n; i += m) {
            for (var j = 0u; j < half_m; j++) {
                let idx1 = i + j;
                let idx2 = idx1 + half_m;
                
                if (idx1 < n && idx2 < n) {
                    // Twiddle factor index
                    let twiddle_idx = j * (n / m) * 2u;
                    let twiddle = vec2<f32>(
                        twiddle_factors[twiddle_idx],
                        twiddle_factors[twiddle_idx + 1u]
                    );
                    
                    // Apply inverse sign for IFFT
                    var twiddle_corrected = twiddle;
                    if (fft_info.is_inverse != 0u) {
                        twiddle_corrected.y = -twiddle_corrected.y;
                    }
                    
                    // Butterfly operation
                    let product = complex_mul(
                        local_data[idx2].x, local_data[idx2].y,
                        twiddle_corrected.x, twiddle_corrected.y
                    );
                    
                    let temp = local_data[idx1];
                    local_data[idx1] = complex_add(temp, product);
                    local_data[idx2] = complex_sub(temp, product);
                }
            }
        }
        
        workgroupBarrier();
    }
    
    // Copy results back to global memory with strided access
    for (var i = 0u; i < n; i++) {
        let output_idx = base_offset + i * stride * 2u;
        
        var result = local_data[i];
        
        // Apply normalization for IFFT
        if (axis_fft_info.is_inverse != 0u) {
            result = vec2<f32>(result.x / f32(n), result.y / f32(n));
        }
        
        output_complex[output_idx] = result.x;
        output_complex[output_idx + 1u] = result.y;
    }
}