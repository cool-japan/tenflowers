// GPU-accelerated affine transform shader
// Applies a 2x3 affine warp matrix to each pixel, with bilinear interpolation.
// The matrix is encoded as [m00, m01, m02, m10, m11, m12] (row-major 2x3).
//
// For output pixel (x, y), the source coordinates are:
//   src_x = m00*x + m01*y + m02
//   src_y = m10*x + m11*y + m12

struct AffineUniforms {
    width:    u32,
    height:   u32,
    channels: u32,
    padding:  u32,
    // 2×3 matrix stored row-major: [m00, m01, m02, m10, m11, m12]
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    // two padding floats to keep 16-byte alignment
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<storage, read>       input_data  : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data : array<f32>;
@group(0) @binding(2) var<uniform>             uniforms    : AffineUniforms;

// Bilinear sample from input at (sx, sy) for a given channel
fn bilinear_sample(sx: f32, sy: f32, channel: u32) -> f32 {
    let w = uniforms.width;
    let h = uniforms.height;

    if (sx < 0.0 || sx >= f32(w) || sy < 0.0 || sy >= f32(h)) {
        return 0.0;
    }

    let x0 = u32(floor(sx));
    let y0 = u32(floor(sy));
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);

    let fx = sx - f32(x0);
    let fy = sy - f32(y0);

    let stride = w * h;
    let base   = channel * stride;

    let v00 = input_data[base + y0 * w + x0];
    let v01 = input_data[base + y0 * w + x1];
    let v10 = input_data[base + y1 * w + x0];
    let v11 = input_data[base + y1 * w + x1];

    let row0 = mix(v00, v01, fx);
    let row1 = mix(v10, v11, fx);
    return mix(row0, row1, fy);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let c = gid.z;

    if (x >= uniforms.width || y >= uniforms.height || c >= uniforms.channels) {
        return;
    }

    let fx = f32(x);
    let fy = f32(y);

    // Inverse-map: find source pixel for output (x, y)
    let src_x = uniforms.m00 * fx + uniforms.m01 * fy + uniforms.m02;
    let src_y = uniforms.m10 * fx + uniforms.m11 * fy + uniforms.m12;

    let out_idx = c * uniforms.width * uniforms.height + y * uniforms.width + x;
    output_data[out_idx] = bilinear_sample(src_x, src_y, c);
}
