pub mod activation;
pub mod async_binary;
pub mod basic;
pub mod benchmark;
pub mod binary;
pub mod comparison;
pub mod conv;
pub mod einsum;
pub mod fft;
pub mod framework_comparison;
#[cfg(feature = "gpu")]
pub mod gpu_tensorflow_benchmark;
#[cfg(feature = "gpu")]
pub mod hybrid_scheduler;
pub mod lapack;
#[cfg(feature = "blas")]
pub mod lapack_f32;
#[cfg(feature = "blas")]
pub mod lapack_f64;
pub mod linalg;
pub mod logical;
pub mod manipulation;
pub mod matmul;
pub mod normalization;
pub mod numpy_array_creation;
pub mod numpy_compat;
pub mod optimized_binary;
pub mod optimized_matmul_v2;
pub mod performance_benchmark;
pub mod pooling;
pub mod random;
pub mod reduction;
pub mod registry;
pub mod shape_inference;
pub mod special;
pub mod stats;
pub mod ultra_matmul_v3;
pub mod ultra_performance_matmul;
pub mod unary;

#[cfg(all(test, feature = "gpu"))]
mod cpu_gpu_overlap_test;

pub use activation::{
    elu, gelu, hard_swish, hardswish, leaky_relu, log_softmax, mish, prelu, relu, relu6, sigmoid,
    softmax, swish, tanh,
};
pub use async_binary::{
    add_async, add_async_priority, batch_add_async, batch_mul_async, div_async,
    is_async_operations_idle, mul_async, mul_async_priority, pow_async, prelu_async, sub_async,
    synchronize_async_operations,
};
pub use basic::{add, clamp, div, mul, scalar_add, sub};
pub use binary::pow;
pub use comparison::{eq, ge, gt, le, lt, ne};
pub use conv::{conv1d, conv2d, conv3d, depthwise_conv2d};
pub use einsum::einsum;
pub use fft::{
    fft, fft2, fft2_bf16, fft2_f16, fft2_inplace, fft3, fft3_inplace, fft_bf16, fft_f16,
    fft_inplace, ifft, ifft2, ifft2_bf16, ifft2_f16, ifft2_inplace, ifft3, ifft3_inplace,
    ifft_bf16, ifft_f16, ifft_inplace, rfft,
};
pub use framework_comparison::{
    print_framework_comparison_results, run_framework_comparison_benchmark,
    FrameworkBenchmarkConfig, FrameworkComparisonResult,
};
#[cfg(feature = "gpu")]
pub use gpu_tensorflow_benchmark::{
    run_quick_gpu_tensorflow_benchmark, GpuBenchmarkConfig, GpuBenchmarkResult,
    GpuTensorFlowBenchmark,
};
pub use lapack::{
    cholesky_lapack, determinant_lapack, eigenvalues_lapack, inverse_lapack, is_lapack_available,
    lapack_provider, lstsq_lapack, lu_decompose_lapack, matmul_blas, qr_lapack, solve_lapack,
    svd_lapack,
};
pub use linalg::{cholesky, det, eig, inv, lu, svd};
pub use logical::{logical_and, logical_not, logical_or, logical_xor};
pub use manipulation::{
    // Shape operations
    broadcast_to,
    // Utility operations (extracted to utilities.rs)
    cast,
    // Concatenation operations (extracted to concatenation.rs)
    concat,
    expand_as,
    expand_dims,
    flatten,
    // Transposition operations (extracted to transpose.rs)
    flip,
    // Indexing operations (extracted to indexing.rs)
    gather,
    identity,
    one_hot,
    pad,
    repeat,
    reshape,
    roll,
    scatter,
    select,
    slice,
    slice_with_stride,
    split,
    squeeze,
    stack,
    tile,
    transpose,
    unsqueeze,
    where_op,
};
pub use matmul::{batch_matmul, dot, matmul, outer};
pub use normalization::{batch_norm, group_norm, layer_norm};
pub use numpy_array_creation::{
    arange, diag, diagonal, eye, from_range, fromfunction, full, geomspace,
    identity as np_identity, linspace, logspace, meshgrid, ones, tri, tril, triu, zeros,
};
pub use numpy_compat::{
    absolute, apply_ufunc, arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh, ceil, cos,
    cosh, create_f32_ufunc_registry, exp, exp2, expm1, fix, floor, fmax, fmin, fmod, isfinite,
    isinf, isnan, list_ufuncs, log, log10, log1p, log2, maximum, minimum, modulo, negative,
    numpy_broadcast_arrays, reciprocal, remainder, rint, sign, signbit, sin, sinh, sqrt, square,
    tan, tanh as np_tanh, trunc, UfuncRegistry,
};
pub use optimized_matmul_v2::ultra_matmul_v2;
pub use pooling::{
    adaptive_avg_pool2d, adaptive_max_pool2d, avg_pool2d, avg_pool3d, fractional_avg_pool2d,
    fractional_max_pool2d, global_avg_pool2d, global_avg_pool3d, global_max_pool2d,
    global_max_pool3d, max_pool2d, max_pool3d, roi_align2d, roi_pool2d,
};
pub use random::{
    multinomial_f32, rand_f32, rand_f64, randn_f32, randn_f64, random_normal_f32,
    random_normal_f32_device, random_normal_f64, random_uniform_f32, random_uniform_f32_device,
    random_uniform_f64, random_uniform_int,
};
pub use reduction::{argmax, argmin, cumprod, cumsum, max, mean, min, sum, topk};
pub use special::{
    bessel_j0, bessel_j1, bessel_y0, bessel_y1, digamma, erf, erfc, gamma, lgamma, smooth_l1_loss,
};
pub use stats::{
    correlation, covariance, histogram, kurtosis, median, moment, percentile, quantile, range,
    skewness,
};
pub use ultra_matmul_v3::{
    clear_performance_analytics, configure_ultra_performance, get_performance_analytics,
    ultra_matmul_v3, UltraPerformanceConfig,
};
pub use ultra_performance_matmul::ultra_matmul;
pub use unary::{abs, neg};
