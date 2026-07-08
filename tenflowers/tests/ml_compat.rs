//! ML ecosystem compatibility tests for the `tenflowers` meta-crate.
//!
//! These tests verify interoperability between TenfloweRS tensor types and the
//! common Rust ML/scientific-computing ecosystem:
//!
//! - `num-traits` (via `scirs2_core`): `Float` bound interoperability with generic
//!   functions over tensor element types.
//! - `bytemuck`: `Pod` type casting for efficient zero-copy tensor data access.
//! - `DType` exhaustiveness: all variants covered in match arms.
//! - Numeric stability: basic arithmetic on reasonable inputs produces finite results.
//!
//! # Design rationale
//!
//! These are compile-time and lightweight runtime checks — no external network
//! calls, no large allocations, no non-determinism.  File I/O uses
//! `std::env::temp_dir()` and cleans up after itself.

use tenflowers::core::{DType, Tensor};

// ─── 1. Slice access and bytemuck::cast_slice interop ─────────────────────────

/// Verify that `Tensor<f32>` data is accessible as a `&[f32]` slice.
///
/// This is the primary precondition for all `bytemuck`-based zero-copy
/// operations: if we can get a `&[f32]`, we can always reinterpret it as
/// `&[u8]` via `bytemuck::cast_slice`.
#[test]
fn bytemuck_pod_f32_slice_roundtrip() {
    // Create a known tensor.
    let data: Vec<f32> = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = [2_usize, 3];
    let tensor = Tensor::<f32>::from_vec(data.clone(), &shape)
        .expect("Tensor::from_vec should succeed for a valid shape");

    // Obtain an immutable slice of the raw element data.
    let elem_slice: &[f32] = tensor
        .as_slice()
        .expect("as_slice should return Some for a contiguous CPU tensor");

    // Verify roundtrip: slice contents match the original Vec.
    assert_eq!(
        elem_slice,
        data.as_slice(),
        "elem slice must match original data"
    );

    // Cast to &[u8] using bytemuck — this is the canonical way to obtain a
    // raw byte view for serialisation or transfer to foreign buffers.
    let byte_slice: &[u8] = bytemuck::cast_slice(elem_slice);
    assert_eq!(
        byte_slice.len(),
        std::mem::size_of_val(elem_slice),
        "byte slice length must equal element count × sizeof(f32)"
    );

    // Reconstruct the f32 slice from bytes and verify equality.
    let reconstructed: &[f32] = bytemuck::cast_slice(byte_slice);
    assert_eq!(
        reconstructed, elem_slice,
        "reconstructed slice after cast_slice roundtrip must match original"
    );
}

/// Verify that `Tensor<f64>` data undergoes the same byte-cast roundtrip.
///
/// `f64` is a different `bytemuck::Pod` type; this test catches any accidental
/// type-punning bugs that might be specific to 8-byte primitives.
#[test]
fn bytemuck_pod_f64_slice_roundtrip() {
    let data: Vec<f64> = (0..8).map(|i| i as f64 * 0.5).collect();
    let tensor =
        Tensor::<f64>::from_vec(data.clone(), &[2, 4]).expect("Tensor::from_vec should succeed");

    let elem_slice: &[f64] = tensor
        .as_slice()
        .expect("as_slice should return Some for contiguous f64 tensor");

    assert_eq!(elem_slice, data.as_slice());

    let byte_slice: &[u8] = bytemuck::cast_slice(elem_slice);
    assert_eq!(byte_slice.len(), std::mem::size_of_val(elem_slice));

    let reconstructed: &[f64] = bytemuck::cast_slice(byte_slice);
    assert_eq!(reconstructed, elem_slice);
}

/// Verify that `Tensor<u8>` (a non-floating-point `Pod` type) can also be
/// accessed as a byte slice — relevant for image pipeline and quantised models.
#[test]
fn bytemuck_pod_u8_is_its_own_byte_representation() {
    let data: Vec<u8> = (0_u8..16).collect();
    let tensor = Tensor::<u8>::from_vec(data.clone(), &[4, 4])
        .expect("Tensor::from_vec for u8 should succeed");

    let elem_slice: &[u8] = tensor
        .as_slice()
        .expect("as_slice should return Some for u8 tensor");

    // For u8, cast_slice is a no-op at the value level.
    let byte_slice: &[u8] = bytemuck::cast_slice(elem_slice);
    assert_eq!(byte_slice, elem_slice);
}

/// Verify that the data obtained from `Tensor::data()` can also be cast via
/// `bytemuck::cast_slice` (exercises the `data()` path, not just `as_slice()`).
#[test]
fn bytemuck_cast_slice_from_data_method() {
    let tensor = Tensor::<f32>::ones(&[3, 3]);

    // `data()` panics on GPU tensors but is safe for CPU tensors.
    let elem_slice: &[f32] = tensor.data();
    assert_eq!(elem_slice.len(), 9);

    let byte_slice: &[u8] = bytemuck::cast_slice(elem_slice);
    assert_eq!(byte_slice.len(), 9 * 4);

    // Every f32 value is 1.0 — verify the first float from bytes.
    let first_f32: f32 =
        f32::from_ne_bytes([byte_slice[0], byte_slice[1], byte_slice[2], byte_slice[3]]);
    assert!((first_f32 - 1.0_f32).abs() < f32::EPSILON);
}

// ─── 2. num_traits::Float bound interoperability ──────────────────────────────

/// Generic function that uses `scirs2_core::num_traits::Float` to compute the
/// sum of absolute values of tensor elements.
///
/// This proves the `Float` bound (from `num-traits`, re-exported by `scirs2-core`)
/// is compatible with the element types accepted by `Tensor<T>`.
fn sum_abs_via_float_bound<T>(tensor: &Tensor<T>) -> T
where
    T: scirs2_core::num_traits::Float + Default + Clone,
{
    let slice = tensor.data();
    slice.iter().fold(T::zero(), |acc, &x| {
        // Float::abs() and Float::zero() are from num_traits::Float.
        acc + x.abs()
    })
}

#[test]
fn num_traits_float_bound_f32_works_with_tensor() {
    let data: Vec<f32> = vec![-1.0, 2.0, -3.0, 4.0];
    let tensor = Tensor::<f32>::from_vec(data, &[4]).expect("from_vec for f32 should succeed");

    let result = sum_abs_via_float_bound(&tensor);
    // |-1| + |2| + |-3| + |4| = 10
    assert!(
        (result - 10.0_f32).abs() < 1e-5,
        "sum_abs should be 10.0, got {result}"
    );
}

#[test]
fn num_traits_float_bound_f64_works_with_tensor() {
    let data: Vec<f64> = vec![-1.5, 2.5, -3.5, 4.5];
    let tensor = Tensor::<f64>::from_vec(data, &[4]).expect("from_vec for f64 should succeed");

    let result = sum_abs_via_float_bound(&tensor);
    // 1.5 + 2.5 + 3.5 + 4.5 = 12.0
    assert!(
        (result - 12.0_f64).abs() < 1e-10,
        "sum_abs should be 12.0, got {result}"
    );
}

/// Generic function that uses `Float::nan()`, `Float::infinity()`, and
/// `Float::is_finite()` — all from `num_traits::Float` — to validate that
/// a tensor contains only finite values.
fn all_finite<T>(tensor: &Tensor<T>) -> bool
where
    T: scirs2_core::num_traits::Float + Clone,
{
    tensor.data().iter().all(|x| x.is_finite())
}

#[test]
fn num_traits_float_is_finite_check_on_tensor() {
    let tensor = Tensor::<f32>::ones(&[4, 4]);
    assert!(all_finite(&tensor), "ones tensor must be fully finite");
}

/// Generic function proving that `Float::sqrt`, `Float::ln`, `Float::exp` are
/// callable in a generic context with the same `Float` bound.
fn compute_log_exp_sqrt<T>(x: T) -> (T, T, T)
where
    T: scirs2_core::num_traits::Float,
{
    (x.sqrt(), x.ln(), x.exp())
}

#[test]
fn num_traits_float_sqrt_ln_exp_coherent_f32() {
    let (sqrt, ln, exp) = compute_log_exp_sqrt(4.0_f32);
    assert!((sqrt - 2.0_f32).abs() < 1e-5, "sqrt(4) should be 2");
    assert!(
        (ln - std::f32::consts::LN_2 * 2.0).abs() < 1e-5,
        "ln(4) should be ≈ 1.386"
    );
    assert!(exp > 50.0_f32, "exp(4) should be ≈ 54.6");
}

// ─── 3. DType exhaustiveness ──────────────────────────────────────────────────
//
// The `DType` enum has 17 variants (as of the dtype.rs definition).  Adding a
// new variant without updating match sites is a compile error when the match
// has no wildcard arm.  These tests serve as a canary for such additions.

/// Maps every `DType` variant to its expected byte size.  This match has no
/// wildcard arm — if a new variant is added to `DType` the test WILL NOT
/// compile until the arm is added here, making variant-exhaustiveness visible
/// at CI time.
fn dtype_to_expected_size(dt: DType) -> usize {
    match dt {
        DType::Float16 => 2,
        DType::BFloat16 => 2,
        DType::Float32 => 4,
        DType::Float64 => 8,
        DType::Int32 => 4,
        DType::Int64 => 8,
        DType::Int16 => 2,
        DType::Int8 => 1,
        DType::Int4 => 1, // minimum addressable unit
        DType::UInt64 => 8,
        DType::UInt32 => 4,
        DType::UInt16 => 2,
        DType::UInt8 => 1,
        DType::Bool => 1,
        DType::Complex32 => 8,  // 2 × f32
        DType::Complex64 => 16, // 2 × f64
        DType::String => 8,     // pointer-width placeholder
    }
}

/// Maps every `DType` variant to its canonical name string.  Exhaustive
/// without wildcard, matching `DType::name()` in the core crate.
fn dtype_to_name(dt: DType) -> &'static str {
    match dt {
        DType::Float16 => "float16",
        DType::BFloat16 => "bfloat16",
        DType::Float32 => "float32",
        DType::Float64 => "float64",
        DType::Int32 => "int32",
        DType::Int64 => "int64",
        DType::Int16 => "int16",
        DType::Int8 => "int8",
        DType::Int4 => "int4",
        DType::UInt64 => "uint64",
        DType::UInt32 => "uint32",
        DType::UInt16 => "uint16",
        DType::UInt8 => "uint8",
        DType::Bool => "bool",
        DType::Complex32 => "complex32",
        DType::Complex64 => "complex64",
        DType::String => "string",
    }
}

/// All `DType` variants must round-trip through `DType::size()` consistently
/// with our local exhaustive mapping.
#[test]
fn dtype_size_matches_exhaustive_map() {
    let all_dtypes = [
        DType::Float16,
        DType::BFloat16,
        DType::Float32,
        DType::Float64,
        DType::Int32,
        DType::Int64,
        DType::Int16,
        DType::Int8,
        DType::Int4,
        DType::UInt64,
        DType::UInt32,
        DType::UInt16,
        DType::UInt8,
        DType::Bool,
        DType::Complex32,
        DType::Complex64,
        DType::String,
    ];

    for dt in all_dtypes {
        let expected = dtype_to_expected_size(dt);
        let actual = dt.size();
        assert_eq!(
            actual, expected,
            "DType::{:?} size mismatch: expected {expected}, got {actual}",
            dt
        );
    }
}

/// All `DType` variants must produce a non-empty name string.
#[test]
fn dtype_name_exhaustive_all_non_empty() {
    let all_dtypes = [
        DType::Float16,
        DType::BFloat16,
        DType::Float32,
        DType::Float64,
        DType::Int32,
        DType::Int64,
        DType::Int16,
        DType::Int8,
        DType::Int4,
        DType::UInt64,
        DType::UInt32,
        DType::UInt16,
        DType::UInt8,
        DType::Bool,
        DType::Complex32,
        DType::Complex64,
        DType::String,
    ];

    for dt in all_dtypes {
        let name = dt.name();
        assert!(!name.is_empty(), "DType::{dt:?} must have a non-empty name");
        assert_eq!(
            name,
            dtype_to_name(dt),
            "DType::{dt:?} name '{name}' must match exhaustive map"
        );
    }
}

/// Tensor dtype accessor must reflect the correct static type for `f32` and
/// `f64` tensors.
#[test]
fn tensor_dtype_matches_element_type() {
    let f32_tensor = Tensor::<f32>::zeros(&[2, 2]);
    assert_eq!(
        f32_tensor.dtype(),
        DType::Float32,
        "f32 tensor dtype must be Float32"
    );

    let f64_tensor = Tensor::<f64>::zeros(&[2, 2]);
    assert_eq!(
        f64_tensor.dtype(),
        DType::Float64,
        "f64 tensor dtype must be Float64"
    );
}

// ─── 4. Numeric stability ─────────────────────────────────────────────────────
//
// These tests verify that basic tensor operations on reasonable (finite, normal)
// inputs do not accidentally produce NaN or Inf values.  They are not
// exhaustive — they catch obvious catastrophic regressions.

/// Addition of two finite tensors must produce finite output.
#[test]
fn numeric_stability_add_finite_inputs_produces_finite_output() {
    use tenflowers::core::ops;

    let a = Tensor::<f32>::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])
        .expect("from_vec");
    let b = Tensor::<f32>::from_vec(
        vec![-1.0_f32, -2.0, -3.0, -4.0, 10.0, 20.0, 30.0, 40.0],
        &[2, 4],
    )
    .expect("from_vec");

    let c = ops::add(&a, &b).expect("add should succeed on finite inputs");
    let data = c.to_vec().expect("to_vec");

    assert!(
        data.iter().all(|v| v.is_finite()),
        "add on finite inputs must not produce NaN or Inf; got {:?}",
        data
    );
}

/// Element-wise multiplication of two bounded tensors must stay finite.
#[test]
fn numeric_stability_mul_moderate_values_produces_finite_output() {
    use tenflowers::core::ops;

    let a =
        Tensor::<f32>::from_vec((0..16).map(|i| i as f32).collect(), &[4, 4]).expect("from_vec");
    let b =
        Tensor::<f32>::from_vec((0..16).map(|i| -(i as f32)).collect(), &[4, 4]).expect("from_vec");

    let c = ops::mul(&a, &b).expect("mul should succeed");
    let data = c.to_vec().expect("to_vec");

    assert!(
        data.iter().all(|v| v.is_finite()),
        "mul on moderate values must not produce Inf; got {:?}",
        data
    );
}

/// Subtraction of a tensor from itself must produce all-zero output (no NaN
/// from catastrophic cancellation on exact values).
#[test]
fn numeric_stability_sub_self_gives_zero() {
    use tenflowers::core::ops;

    let a = Tensor::<f32>::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4]).expect("from_vec");

    let c = ops::sub(&a, &a).expect("sub(a, a) should succeed");
    let data = c.to_vec().expect("to_vec");

    assert!(
        data.iter().all(|v| v.is_finite()),
        "sub(a, a) must produce finite values; got {:?}",
        data
    );
    assert!(
        data.iter().all(|v| v.abs() < f32::EPSILON),
        "sub(a, a) must produce zero-vector; got {:?}",
        data
    );
}

/// Ones tensor stays finite after a round-trip through `ops::add` then
/// `ops::sub`, verifying the pipeline does not accumulate NaN.
#[test]
fn numeric_stability_add_then_sub_roundtrip_finite() {
    use tenflowers::core::ops;

    let a = Tensor::<f32>::ones(&[8, 8]);
    let b = Tensor::<f32>::ones(&[8, 8]);

    let sum = ops::add(&a, &b).expect("add");
    let diff = ops::sub(&sum, &b).expect("sub");
    let data = diff.to_vec().expect("to_vec");

    assert!(
        data.iter().all(|v| v.is_finite()),
        "add+sub roundtrip must stay finite; got NaN/Inf"
    );
    // sum - b = (a+b) - b = a = ones → every element ≈ 1.0
    assert!(
        data.iter().all(|v| (v - 1.0_f32).abs() < 1e-5),
        "add+sub roundtrip must recover the original ones tensor"
    );
}

/// Verify that zero-initialised tensors produce zero after multiplication.
#[test]
fn numeric_stability_mul_by_zero_tensor_gives_zero() {
    use tenflowers::core::ops;

    let a = Tensor::<f64>::ones(&[6]);
    let zero = Tensor::<f64>::zeros(&[6]);

    let c = ops::mul(&a, &zero).expect("mul with zero tensor");
    let data = c.to_vec().expect("to_vec");

    assert!(
        data.iter().all(|v| v.abs() < f64::EPSILON),
        "ones × zeros must yield all zeros; got {:?}",
        data
    );
}

/// Verify that the `utils::softmax` utility (if available via `tenflowers::utils`)
/// does not produce NaN or Inf for reasonable logit vectors.
#[test]
fn numeric_stability_utils_softmax_finite_output() {
    let logits: Vec<f32> = vec![1.0_f32, 2.0, 3.0, 4.0];
    let probs = tenflowers::utils::softmax(&logits);

    assert!(
        probs.iter().all(|v| v.is_finite()),
        "softmax of finite logits must be finite; got {:?}",
        probs
    );
    // Probabilities must sum to 1.
    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0_f32).abs() < 1e-5,
        "softmax output must sum to 1.0; got {sum}"
    );
    // All probabilities must be non-negative.
    assert!(
        probs.iter().all(|v| *v >= 0.0_f32),
        "softmax output must be non-negative"
    );
}

/// Verify that `utils::sigmoid` stays in [0, 1] for all inputs (including
/// extreme values) and never produces NaN or Inf.
///
/// For very large positive/negative inputs f32 arithmetic saturates sigmoid to
/// exactly 1.0 or 0.0 — that is correct numeric behaviour, not a bug.  We
/// therefore test the closed interval [0, 1] rather than the open interval.
#[test]
fn numeric_stability_utils_sigmoid_bounded_output() {
    let inputs: [f32; 8] = [-1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0];

    for &x in &inputs {
        let y = tenflowers::utils::sigmoid(x);
        assert!(y.is_finite(), "sigmoid({x}) must be finite, got {y}");
        assert!(
            (0.0_f32..=1.0_f32).contains(&y),
            "sigmoid({x}) must be in [0, 1], got {y}"
        );
    }

    // Verify the sigmoid(0) = 0.5 property exactly.
    let mid = tenflowers::utils::sigmoid(0.0_f32);
    assert!(
        (mid - 0.5_f32).abs() < 1e-5,
        "sigmoid(0) must equal 0.5, got {mid}"
    );

    // Verify monotonicity: sigmoid is non-decreasing.
    let sorted_inputs = [-10.0_f32, -1.0, 0.0, 1.0, 10.0];
    let values: Vec<f32> = sorted_inputs
        .iter()
        .map(|&x| tenflowers::utils::sigmoid(x))
        .collect();
    for window in values.windows(2) {
        assert!(
            window[1] >= window[0],
            "sigmoid must be non-decreasing; violation: {} → {}",
            window[0],
            window[1]
        );
    }
}

// ─── 5. Temp-dir file I/O sanity (ensures std::env::temp_dir() is usable) ────

/// Write a trivial file to temp_dir and read it back — confirms the test
/// environment grants write access to the temporary directory.
///
/// This is a guard test: if the temp dir is not writable the serialisation
/// tests in `tests/io.rs` would also fail, so we fail fast here with a
/// clear diagnostic.
#[test]
fn temp_dir_is_writable() {
    let mut path = std::env::temp_dir();
    path.push("tenflowers_ml_compat_sanity_check.txt");

    std::fs::write(&path, b"tenflowers_ml_compat_ok")
        .expect("temp_dir must be writable for test infrastructure");

    let contents = std::fs::read(&path).expect("must be able to read back what was just written");
    assert_eq!(contents, b"tenflowers_ml_compat_ok");

    // Always clean up.
    let _ = std::fs::remove_file(&path);
}
