// GPU-accelerated perspective (homography) transform shader
// Applies a 3x3 homography matrix to each pixel with bilinear interpolation.
//
// The matrix H is stored row-major as [h00..h22] (9 floats).
// For output pixel (x, y) in homogeneous coordinates:
//   w'  = h20*x + h21*y + h22
//   src_x = (h00*x + h01*y + h02) / w'
//   src_y = (h10*x + h11*y + h12) / w'

struct PerspectiveUniforms {
    width:    u32,
    height:   u32,
    channels: u32,
    padding:  u32,
    // 3×3 homography stored row-major
    h00: f32,
    h01: f32,
    h02: f32,
    h10: f32,
    h11: f32,
    h12: f32,
    h20: f32,
    h21: f32,
    h22: f32,
    // three padding floats for 16-byte alignment (total 4+9+3 = 16 floats = 64 bytes)
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<storage, read>       input_data  : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data : array<f32>;
@group(0) @binding(2) var<uniform>             uniforms    : PerspectiveUniforms;

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

    // Apply homography: H * [x, y, 1]^T
    let num_x = uniforms.h00 * fx + uniforms.h01 * fy + uniforms.h02;
    let num_y = uniforms.h10 * fx + uniforms.h11 * fy + uniforms.h12;
    let denom  = uniforms.h20 * fx + uniforms.h21 * fy + uniforms.h22;

    let out_idx = c * uniforms.width * uniforms.height + y * uniforms.width + x;

    if (abs(denom) < 1e-7) {
        output_data[out_idx] = 0.0;
        return;
    }

    let src_x = num_x / denom;
    let src_y = num_y / denom;

    output_data[out_idx] = bilinear_sample(src_x, src_y, c);
}
