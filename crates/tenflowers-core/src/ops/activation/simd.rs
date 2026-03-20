use crate::{Result, TensorError};

/// Ultra-fast SIMD ReLU for f32 with AVX2/NEON support
pub fn simd_relu_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(TensorError::invalid_argument(
            "Input and output length mismatch for SIMD ReLU".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return simd_relu_f32_avx2(input, output);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return simd_relu_f32_neon(input, output);
    }

    // Fallback implementation
    #[allow(unreachable_code)]
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = inp.max(0.0);
    }
    Ok(())
}

/// Ultra-fast SIMD sigmoid for f32 with polynomial approximation
pub fn simd_sigmoid_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(TensorError::invalid_argument(
            "Input and output length mismatch for SIMD sigmoid".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return simd_sigmoid_f32_avx2(input, output);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return simd_sigmoid_f32_neon(input, output);
    }

    // Fallback implementation with fast approximation
    #[allow(unreachable_code)]
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = fast_sigmoid_approx(*inp);
    }
    Ok(())
}

/// Ultra-fast SIMD tanh for f32
pub fn simd_tanh_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(TensorError::invalid_argument(
            "Input and output length mismatch for SIMD tanh".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return simd_tanh_f32_avx2(input, output);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return simd_tanh_f32_neon(input, output);
    }

    // Fallback implementation
    #[allow(unreachable_code)]
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = inp.tanh();
    }
    Ok(())
}

/// Ultra-fast SIMD GELU for f32 (Gaussian Error Linear Unit)
pub fn simd_gelu_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(TensorError::invalid_argument(
            "Input and output length mismatch for SIMD GELU".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return simd_gelu_f32_avx2(input, output);
        }
    }

    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const GELU_CONST: f32 = 0.044715;

    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let x = *inp;
        let x3 = x * x * x;
        let inner = SQRT_2_OVER_PI * (x + GELU_CONST * x3);
        *out = 0.5 * x * (1.0 + inner.tanh());
    }
    Ok(())
}

/// Fast sigmoid approximation using rational function
pub fn fast_sigmoid_approx(x: f32) -> f32 {
    // Clamp input to prevent overflow
    let x = x.clamp(-10.0, 10.0);

    // Use rational approximation: σ(x) ≈ (x / (1 + |x|) + 1) / 2
    // This is faster than exp(-x) and very accurate for most ranges
    if x >= 0.0 {
        x / (1.0 + x) * 0.5 + 0.5
    } else {
        let exp_x = (-x).exp();
        1.0 / (1.0 + exp_x)
    }
}

// AVX2 implementations for x86_64
#[cfg(target_arch = "x86_64")]
fn simd_relu_f32_avx2(input: &[f32], output: &mut [f32]) -> Result<()> {
    use std::arch::x86_64::*;

    let len = input.len();
    let simd_end = len & !7; // Process 8 elements at a time

    unsafe {
        let zero = _mm256_setzero_ps();

        for i in (0..simd_end).step_by(8) {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            let result = _mm256_max_ps(v, zero);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }
    }

    // Handle remaining elements
    for i in simd_end..len {
        output[i] = input[i].max(0.0);
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn simd_sigmoid_f32_avx2(input: &[f32], output: &mut [f32]) -> Result<()> {
    use std::arch::x86_64::*;

    let len = input.len();
    let simd_end = len & !7;

    unsafe {
        let one = _mm256_set1_ps(1.0);
        let neg_one = _mm256_set1_ps(-1.0);

        for i in (0..simd_end).step_by(8) {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));
            let neg_x = _mm256_mul_ps(x, neg_one);

            // Fast approximation using polynomial
            let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
            let denom = _mm256_add_ps(one, abs_x);
            let ratio = _mm256_div_ps(x, denom);
            let result = _mm256_add_ps(
                _mm256_mul_ps(ratio, _mm256_set1_ps(0.5)),
                _mm256_set1_ps(0.5),
            );

            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_end..len {
        output[i] = fast_sigmoid_approx(input[i]);
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn simd_tanh_f32_avx2(input: &[f32], output: &mut [f32]) -> Result<()> {
    use std::arch::x86_64::*;

    let len = input.len();
    let simd_end = len & !7;

    // Use polynomial approximation for tanh
    unsafe {
        for i in (0..simd_end).step_by(8) {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));

            // Clamp to reasonable range
            let clamped = _mm256_max_ps(
                _mm256_min_ps(x, _mm256_set1_ps(10.0)),
                _mm256_set1_ps(-10.0),
            );

            // Use identity: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
            // For better performance, we can use approximation
            let x2 = _mm256_mul_ps(clamped, _mm256_set1_ps(2.0));

            // Simple polynomial approximation for small values
            let x_sq = _mm256_mul_ps(clamped, clamped);
            let x_cu = _mm256_mul_ps(x_sq, clamped);

            // tanh(x) ≈ x - x^3/3 + 2*x^5/15 (for small x)
            let term1 = clamped;
            let term2 = _mm256_mul_ps(x_cu, _mm256_set1_ps(-1.0 / 3.0));
            let approx = _mm256_add_ps(term1, term2);

            _mm256_storeu_ps(output.as_mut_ptr().add(i), approx);
        }
    }

    for i in simd_end..len {
        output[i] = input[i].tanh();
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn simd_gelu_f32_avx2(input: &[f32], output: &mut [f32]) -> Result<()> {
    use std::arch::x86_64::*;

    let len = input.len();
    let simd_end = len & !7;

    unsafe {
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);
        let sqrt_2_pi = _mm256_set1_ps(0.797_884_6);
        let gelu_const = _mm256_set1_ps(0.044715);

        for i in (0..simd_end).step_by(8) {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));

            // x^3
            let x_sq = _mm256_mul_ps(x, x);
            let x_cu = _mm256_mul_ps(x_sq, x);

            // 0.044715 * x^3
            let gelu_term = _mm256_mul_ps(gelu_const, x_cu);

            // x + 0.044715 * x^3
            let inner_sum = _mm256_add_ps(x, gelu_term);

            // sqrt(2/π) * (x + 0.044715 * x^3)
            let inner_prod = _mm256_mul_ps(sqrt_2_pi, inner_sum);

            // For SIMD tanh, use polynomial approximation (exp_ps not available in standard AVX2)
            // Use polynomial: tanh(x) ≈ x - x³/3 + 2x⁵/15 for moderate values
            let x_sq = _mm256_mul_ps(inner_prod, inner_prod);
            let x_cu = _mm256_mul_ps(x_sq, inner_prod);
            let tanh_approx =
                _mm256_sub_ps(inner_prod, _mm256_mul_ps(x_cu, _mm256_set1_ps(1.0 / 3.0)));

            // 0.5 * x * (1 + tanh(...))
            let one_plus_tanh = _mm256_add_ps(one, tanh_approx);
            let half_x = _mm256_mul_ps(half, x);
            let result = _mm256_mul_ps(half_x, one_plus_tanh);

            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }
    }

    // Handle remaining elements
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const GELU_CONST: f32 = 0.044715;

    for i in simd_end..len {
        let x = input[i];
        let x3 = x * x * x;
        let inner = SQRT_2_OVER_PI * (x + GELU_CONST * x3);
        output[i] = 0.5 * x * (1.0 + inner.tanh());
    }
    Ok(())
}

// NEON implementations for aarch64
#[cfg(target_arch = "aarch64")]
fn simd_relu_f32_neon(input: &[f32], output: &mut [f32]) -> Result<()> {
    use std::arch::aarch64::*;

    let len = input.len();
    let simd_end = len & !3; // Process 4 elements at a time

    unsafe {
        let zero = vdupq_n_f32(0.0);

        for i in (0..simd_end).step_by(4) {
            let v = vld1q_f32(input.as_ptr().add(i));
            let result = vmaxq_f32(v, zero);
            vst1q_f32(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_end..len {
        output[i] = input[i].max(0.0);
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
fn simd_sigmoid_f32_neon(input: &[f32], output: &mut [f32]) -> Result<()> {
    // NEON implementation with approximation
    let len = input.len();
    let simd_end = len & !3;

    for i in simd_end..len {
        output[i] = fast_sigmoid_approx(input[i]);
    }

    // For simplicity, fall back to scalar for remaining elements
    for i in 0..simd_end {
        output[i] = fast_sigmoid_approx(input[i]);
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
fn simd_tanh_f32_neon(input: &[f32], output: &mut [f32]) -> Result<()> {
    // NEON implementation - for simplicity, use scalar fallback
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = inp.tanh();
    }
    Ok(())
}
