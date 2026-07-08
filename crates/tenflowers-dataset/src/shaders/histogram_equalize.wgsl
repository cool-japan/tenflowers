// GPU-accelerated per-channel histogram equalization shader
//
// This shader is invoked in a two-pass scheme:
//   Pass 1 (entry: build_histogram): atomically increment histogram bins
//   Pass 2 (entry: equalize):        apply the CDF-based remap to each pixel
//
// The histogram_data buffer holds NUM_BINS u32 values per channel
// (interleaved as channel-major: histogram[c * NUM_BINS + bin]).
// The cdf_data buffer holds the same shape as f32 (written by CPU prefix-scan,
// or by a third pass not shown here; for simplicity the CPU may compute the CDF
// and upload it).

const NUM_BINS: u32 = 256u;

struct EqualizeUniforms {
    width:    u32,
    height:   u32,
    channels: u32,
    padding:  u32,
}

// Bindings for histogram-build pass
@group(0) @binding(0) var<storage, read>            image_data      : array<f32>;
@group(0) @binding(1) var<storage, read_write>      histogram_data  : array<atomic<u32>>;
@group(0) @binding(2) var<uniform>                  uniforms        : EqualizeUniforms;

// --- Pass 1: build histogram (dispatch W*H per channel dimension) ---
@compute @workgroup_size(16, 16, 1)
fn build_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let c = gid.z;

    if (x >= uniforms.width || y >= uniforms.height || c >= uniforms.channels) {
        return;
    }

    let px_idx = c * uniforms.height * uniforms.width + y * uniforms.width + x;
    let value  = clamp(image_data[px_idx], 0.0, 1.0);
    let bin    = min(u32(value * f32(NUM_BINS)), NUM_BINS - 1u);

    atomicAdd(&histogram_data[c * NUM_BINS + bin], 1u);
}

// --- Pass 2 bindings: remap using pre-computed CDF ---
// (separate bind group layout; in practice caller uses two pipelines)

@group(0) @binding(0) var<storage, read>            src_image  : array<f32>;
@group(0) @binding(1) var<storage, read_write>      dst_image  : array<f32>;
@group(0) @binding(2) var<storage, read>            cdf_data   : array<f32>;
@group(0) @binding(3) var<uniform>                  eq_uniforms: EqualizeUniforms;

@compute @workgroup_size(16, 16, 1)
fn equalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let c = gid.z;

    if (x >= eq_uniforms.width || y >= eq_uniforms.height || c >= eq_uniforms.channels) {
        return;
    }

    let px_idx   = c * eq_uniforms.height * eq_uniforms.width + y * eq_uniforms.width + x;
    let value    = clamp(src_image[px_idx], 0.0, 1.0);
    let bin      = min(u32(value * f32(NUM_BINS)), NUM_BINS - 1u);
    let equalized = cdf_data[c * NUM_BINS + bin];

    dst_image[px_idx] = clamp(equalized, 0.0, 1.0);
}
