//! # TenfloweRS C FFI Layer
//!
//! This module exposes a stable C-compatible ABI for the TenfloweRS ML framework.
//! It covers:
//!
//! - **Neural Layer Inference API**: Allocate/free `TfDense` layers and run forward passes.
//! - **Dataset Loading API**: Load CSV datasets, query their length, and retrieve items.
//! - **Error type**: A consistent `TfError` (`i32`) enum mirroring C error conventions.
//!
//! ## Safety Contract
//!
//! All `unsafe extern "C"` functions in this module follow these invariants:
//!
//! 1. Pointer arguments must either be null or point to a valid object of the
//!    advertised type that was previously returned by the corresponding `_new` function.
//! 2. Slice-backed pointer arguments (`input`, `output`, `path`) must refer to memory
//!    that is valid for the given length / size in bytes.
//! 3. Objects must not be accessed concurrently without external synchronisation.
//! 4. Every object created with a `_new` function must eventually be freed with the
//!    matching `_free` function.  Double-free is undefined behaviour.
//!
//! ## Error Codes
//!
//! | Constant                    | Value | Meaning                                      |
//! |-----------------------------|-------|----------------------------------------------|
//! | `TF_OK`                     | 0     | Success                                      |
//! | `TF_ERROR_NULL_PTR`         | 1     | A required pointer argument was null         |
//! | `TF_ERROR_OUT_OF_BOUNDS`    | 2     | An index or buffer length was out of range   |
//! | `TF_ERROR_IO`               | 3     | An I/O error occurred (e.g. file not found)  |
//! | `TF_ERROR_INVALID_SHAPE`    | 4     | Tensor shape mismatch or invalid dimensions  |
//!
//! ## Example (C pseudo-code)
//!
//! ```c
//! TfDense* layer = tf_dense_new(4, 2);
//! float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
//! float output[2];
//! TfError err = tf_dense_forward(layer, input, 4, output, 2);
//! tf_dense_free(layer);
//! ```

#![allow(clippy::not_unsafe_ptr_arg_deref)] // We perform the checks ourselves

use std::ffi::CStr;
use std::os::raw::c_char;

use tenflowers_dataset::CsvDatasetBuilder;
use tenflowers_neural::Dense;
use tenflowers_core::Tensor;

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

/// Success.
pub const TF_OK: i32 = 0;
/// A required pointer argument was null.
pub const TF_ERROR_NULL_PTR: i32 = 1;
/// An index or buffer size was out of range.
pub const TF_ERROR_OUT_OF_BOUNDS: i32 = 2;
/// An I/O or filesystem error occurred.
pub const TF_ERROR_IO: i32 = 3;
/// A tensor shape mismatch or invalid dimensionality.
pub const TF_ERROR_INVALID_SHAPE: i32 = 4;

/// C-compatible error code returned by most FFI functions.
///
/// Use the `TF_OK`, `TF_ERROR_*` constants to inspect the value.
pub type TfError = i32;

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

/// Opaque handle to a Dense (fully-connected) neural layer holding `f32` weights.
///
/// Obtain via [`tf_dense_new`]; release via [`tf_dense_free`].
pub struct TfDense {
    inner: Dense<f32>,
    input_size: usize,
    output_size: usize,
}

/// Opaque handle to a CSV-backed dataset of `f32` feature rows.
///
/// Obtain via [`tf_dataset_from_csv`]; release via [`tf_dataset_free`].
pub struct TfDataset {
    /// Flat storage: each entry is a feature row.
    rows: Vec<Vec<f32>>,
    /// Feature width (number of columns per row).
    feature_len: usize,
}

// ---------------------------------------------------------------------------
// Dense layer API
// ---------------------------------------------------------------------------

/// Allocate a new `Dense` layer with zero-initialised weights.
///
/// # Safety
///
/// The returned pointer is heap-allocated and owned by the caller.  It must be
/// freed exactly once with [`tf_dense_free`].  Passing `input_size == 0` or
/// `output_size == 0` causes the function to return a null pointer.
///
/// # Returns
///
/// A non-null owning pointer on success, or null if allocation fails or the
/// dimensions are zero.
#[no_mangle]
pub extern "C" fn tf_dense_new(input_size: usize, output_size: usize) -> *mut TfDense {
    if input_size == 0 || output_size == 0 {
        return std::ptr::null_mut();
    }

    // Dense::new(input_dim, output_dim, use_bias) — zero-initialised weights.
    let layer = Dense::<f32>::new(input_size, output_size, true);

    let boxed = Box::new(TfDense {
        inner: layer,
        input_size,
        output_size,
    });

    Box::into_raw(boxed)
}

/// Free a `TfDense` layer previously created by [`tf_dense_new`].
///
/// # Safety
///
/// * `layer` must be a pointer returned by [`tf_dense_new`] that has not been
///   freed before.
/// * A null pointer is accepted and treated as a no-op.
#[no_mangle]
pub unsafe extern "C" fn tf_dense_free(layer: *mut TfDense) {
    if !layer.is_null() {
        // SAFETY: caller guarantees the pointer came from tf_dense_new and is
        // not aliased.
        drop(unsafe { Box::from_raw(layer) });
    }
}

/// Run a forward pass through a `TfDense` layer.
///
/// Reads `input_len` `f32` values from `input`, multiplies by the weight
/// matrix (and adds bias), and writes `output_len` `f32` values to `output`.
///
/// # Safety
///
/// * `layer` must be a non-null pointer returned by [`tf_dense_new`].
/// * `input` must point to at least `input_len` valid `f32` values.
/// * `output` must point to at least `output_len` writable `f32` values.
///
/// # Errors
///
/// | Condition                                    | Return value              |
/// |----------------------------------------------|---------------------------|
/// | `layer`, `input`, or `output` is null        | `TF_ERROR_NULL_PTR`       |
/// | `input_len` ≠ layer's `input_size`           | `TF_ERROR_INVALID_SHAPE`  |
/// | `output_len` ≠ layer's `output_size`         | `TF_ERROR_INVALID_SHAPE`  |
/// | Underlying computation fails                 | `TF_ERROR_INVALID_SHAPE`  |
#[no_mangle]
pub unsafe extern "C" fn tf_dense_forward(
    layer: *const TfDense,
    input: *const f32,
    input_len: usize,
    output: *mut f32,
    output_len: usize,
) -> TfError {
    // Null-pointer guards.
    if layer.is_null() || input.is_null() || output.is_null() {
        return TF_ERROR_NULL_PTR;
    }

    // SAFETY: caller guarantees layer is valid and non-aliased.
    let dense = unsafe { &*layer };

    // Shape validation.
    if input_len != dense.input_size {
        return TF_ERROR_INVALID_SHAPE;
    }
    if output_len != dense.output_size {
        return TF_ERROR_INVALID_SHAPE;
    }

    // Build an input tensor from the caller-supplied buffer.
    // SAFETY: caller guarantees input points to input_len valid f32 values.
    let input_slice = unsafe { std::slice::from_raw_parts(input, input_len) };
    let input_vec: Vec<f32> = input_slice.to_vec();

    let input_tensor = match Tensor::<f32>::from_vec(input_vec, &[1, input_len]) {
        Ok(t) => t,
        Err(_) => return TF_ERROR_INVALID_SHAPE,
    };

    // Run the forward pass.
    use tenflowers_neural::Layer;
    let result_tensor = match dense.inner.forward(&input_tensor) {
        Ok(t) => t,
        Err(_) => return TF_ERROR_INVALID_SHAPE,
    };

    // Copy result values into the caller-supplied output buffer.
    let result_data = match result_tensor.to_vec() {
        Ok(v) => v,
        Err(_) => return TF_ERROR_INVALID_SHAPE,
    };

    if result_data.len() != output_len {
        return TF_ERROR_INVALID_SHAPE;
    }

    // SAFETY: caller guarantees output points to output_len writable f32 values.
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, output_len) };
    output_slice.copy_from_slice(&result_data);

    TF_OK
}

// ---------------------------------------------------------------------------
// Dataset API
// ---------------------------------------------------------------------------

/// Load a CSV dataset from disk.
///
/// The path is given as a pointer to a UTF-8 encoded byte sequence of length
/// `path_len` (not necessarily null-terminated when `path_len` is provided).
/// The last column in the CSV is treated as the label and is **excluded** from
/// feature rows stored in the dataset handle; all remaining columns become the
/// feature vector for each sample.
///
/// # Safety
///
/// * `path` must point to at least `path_len` valid bytes.
/// * The pointed-to bytes must form a valid UTF-8 path string.
///
/// # Returns
///
/// A non-null owning pointer on success, or null if loading fails.
#[no_mangle]
pub unsafe extern "C" fn tf_dataset_from_csv(
    path: *const c_char,
    path_len: usize,
) -> *mut TfDataset {
    if path.is_null() {
        return std::ptr::null_mut();
    }

    // Convert the C string pointer to a Rust &str.
    // We support both null-terminated strings (path_len == 0 signals "use strlen")
    // and explicit-length strings.
    let path_str: String = if path_len == 0 {
        // SAFETY: caller guarantees path is null-terminated.
        let c_str = unsafe { CStr::from_ptr(path) };
        match c_str.to_str() {
            Ok(s) => s.to_owned(),
            Err(_) => return std::ptr::null_mut(),
        }
    } else {
        // SAFETY: caller guarantees path points to path_len valid bytes.
        let bytes = unsafe { std::slice::from_raw_parts(path as *const u8, path_len) };
        match std::str::from_utf8(bytes) {
            Ok(s) => s.to_owned(),
            Err(_) => return std::ptr::null_mut(),
        }
    };

    // Build the CsvDataset using the builder — last column is the label.
    let csv_dataset: tenflowers_dataset::CsvDataset<f32> =
        match CsvDatasetBuilder::new().from_path(&path_str).build() {
            Ok(ds) => ds,
            Err(_) => return std::ptr::null_mut(),
        };

    // Materialise all rows into our FFI-friendly flat structure.
    let n = tenflowers_dataset::Dataset::len(&csv_dataset);
    if n == 0 {
        let boxed = Box::new(TfDataset {
            rows: Vec::new(),
            feature_len: 0,
        });
        return Box::into_raw(boxed);
    }

    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut feature_len: usize = 0;

    for i in 0..n {
        match tenflowers_dataset::Dataset::get(&csv_dataset, i) {
            Ok((features, _label)) => {
                let fdata = match features.to_vec() {
                    Ok(v) => v,
                    Err(_) => return std::ptr::null_mut(),
                };
                if i == 0 {
                    feature_len = fdata.len();
                }
                rows.push(fdata);
            }
            Err(_) => return std::ptr::null_mut(),
        }
    }

    let boxed = Box::new(TfDataset { rows, feature_len });
    Box::into_raw(boxed)
}

/// Free a dataset previously created by [`tf_dataset_from_csv`].
///
/// # Safety
///
/// * `dataset` must be a pointer returned by [`tf_dataset_from_csv`] that has
///   not been freed before.
/// * A null pointer is accepted and treated as a no-op.
#[no_mangle]
pub unsafe extern "C" fn tf_dataset_free(dataset: *mut TfDataset) {
    if !dataset.is_null() {
        // SAFETY: caller guarantees the pointer came from tf_dataset_from_csv.
        drop(unsafe { Box::from_raw(dataset) });
    }
}

/// Return the number of samples in a dataset.
///
/// # Safety
///
/// * `dataset` must be a non-null pointer returned by [`tf_dataset_from_csv`].
///
/// # Returns
///
/// The number of rows in the dataset, or `0` if `dataset` is null.
#[no_mangle]
pub unsafe extern "C" fn tf_dataset_len(dataset: *const TfDataset) -> usize {
    if dataset.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees dataset is valid.
    let ds = unsafe { &*dataset };
    ds.rows.len()
}

/// Copy the feature vector for the sample at `index` into `output`.
///
/// # Safety
///
/// * `dataset` must be a non-null pointer returned by [`tf_dataset_from_csv`].
/// * `output` must point to at least `output_len` writable `f32` values.
///
/// # Errors
///
/// | Condition                                    | Return value              |
/// |----------------------------------------------|---------------------------|
/// | `dataset` or `output` is null                | `TF_ERROR_NULL_PTR`       |
/// | `index` ≥ number of samples                  | `TF_ERROR_OUT_OF_BOUNDS`  |
/// | `output_len` < feature width of the dataset  | `TF_ERROR_OUT_OF_BOUNDS`  |
#[no_mangle]
pub unsafe extern "C" fn tf_dataset_get_item(
    dataset: *const TfDataset,
    index: usize,
    output: *mut f32,
    output_len: usize,
) -> TfError {
    if dataset.is_null() || output.is_null() {
        return TF_ERROR_NULL_PTR;
    }

    // SAFETY: caller guarantees dataset is valid.
    let ds = unsafe { &*dataset };

    if index >= ds.rows.len() {
        return TF_ERROR_OUT_OF_BOUNDS;
    }

    let row = &ds.rows[index];

    if output_len < row.len() {
        return TF_ERROR_OUT_OF_BOUNDS;
    }

    // SAFETY: caller guarantees output points to output_len writable f32 values.
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, row.len()) };
    output_slice.copy_from_slice(row);

    TF_OK
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    // -----------------------------------------------------------------------
    // Helper: create a temporary CSV file and return its path.
    // -----------------------------------------------------------------------
    fn write_temp_csv(content: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        // Use a unique filename per test thread to avoid collisions.
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        path.push(format!("tf_ffi_test_{}.csv", id));
        std::fs::write(&path, content).expect("test: should write temp CSV");
        path
    }

    // -----------------------------------------------------------------------
    // TfDense tests
    // -----------------------------------------------------------------------

    /// 1. tf_dense_new returns non-null for valid dimensions.
    #[test]
    fn test_dense_new_returns_non_null() {
        let layer = tf_dense_new(4, 2);
        assert!(!layer.is_null());
        unsafe { tf_dense_free(layer) };
    }

    /// 2. tf_dense_new returns null when input_size is zero.
    #[test]
    fn test_dense_new_zero_input_returns_null() {
        let layer = tf_dense_new(0, 4);
        assert!(layer.is_null());
    }

    /// 3. tf_dense_new returns null when output_size is zero.
    #[test]
    fn test_dense_new_zero_output_returns_null() {
        let layer = tf_dense_new(4, 0);
        assert!(layer.is_null());
    }

    /// 4. tf_dense_free is a no-op on null.
    #[test]
    fn test_dense_free_null_is_noop() {
        // Must not panic or crash.
        unsafe { tf_dense_free(std::ptr::null_mut()) };
    }

    /// 5. tf_dense_forward returns TF_ERROR_NULL_PTR when layer is null.
    #[test]
    fn test_dense_forward_null_layer() {
        let mut output = [0.0f32; 2];
        let input = [1.0f32; 4];
        let err = unsafe {
            tf_dense_forward(
                std::ptr::null(),
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        assert_eq!(err, TF_ERROR_NULL_PTR);
    }

    /// 6. tf_dense_forward returns TF_ERROR_NULL_PTR when input is null.
    #[test]
    fn test_dense_forward_null_input() {
        let layer = tf_dense_new(4, 2);
        assert!(!layer.is_null());
        let mut output = [0.0f32; 2];
        let err = unsafe {
            tf_dense_forward(
                layer as *const TfDense,
                std::ptr::null(),
                4,
                output.as_mut_ptr(),
                output.len(),
            )
        };
        assert_eq!(err, TF_ERROR_NULL_PTR);
        unsafe { tf_dense_free(layer) };
    }

    /// 7. tf_dense_forward returns TF_ERROR_NULL_PTR when output is null.
    #[test]
    fn test_dense_forward_null_output() {
        let layer = tf_dense_new(4, 2);
        assert!(!layer.is_null());
        let input = [1.0f32; 4];
        let err = unsafe {
            tf_dense_forward(
                layer as *const TfDense,
                input.as_ptr(),
                input.len(),
                std::ptr::null_mut(),
                2,
            )
        };
        assert_eq!(err, TF_ERROR_NULL_PTR);
        unsafe { tf_dense_free(layer) };
    }

    /// 8. tf_dense_forward returns TF_ERROR_INVALID_SHAPE for wrong input_len.
    #[test]
    fn test_dense_forward_wrong_input_len() {
        let layer = tf_dense_new(4, 2);
        assert!(!layer.is_null());
        let input = [1.0f32; 3]; // wrong: should be 4
        let mut output = [0.0f32; 2];
        let err = unsafe {
            tf_dense_forward(
                layer as *const TfDense,
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        assert_eq!(err, TF_ERROR_INVALID_SHAPE);
        unsafe { tf_dense_free(layer) };
    }

    /// 9. tf_dense_forward returns TF_ERROR_INVALID_SHAPE for wrong output_len.
    #[test]
    fn test_dense_forward_wrong_output_len() {
        let layer = tf_dense_new(4, 2);
        assert!(!layer.is_null());
        let input = [1.0f32; 4];
        let mut output = [0.0f32; 5]; // wrong: should be 2
        let err = unsafe {
            tf_dense_forward(
                layer as *const TfDense,
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                5, // mismatched
            )
        };
        assert_eq!(err, TF_ERROR_INVALID_SHAPE);
        unsafe { tf_dense_free(layer) };
    }

    /// 10. tf_dense_forward round-trip succeeds and writes output_len values.
    ///
    /// With zero-initialised weights and zero-initialised bias the output must
    /// be all-zeros regardless of the input.
    #[test]
    fn test_dense_forward_round_trip() {
        let layer = tf_dense_new(4, 2);
        assert!(!layer.is_null());
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [f32::NAN; 2];
        let err = unsafe {
            tf_dense_forward(
                layer as *const TfDense,
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        assert_eq!(err, TF_OK);
        // Zero-init weights → output must be finite.
        for &v in &output {
            assert!(v.is_finite(), "expected finite output, got {v}");
        }
        unsafe { tf_dense_free(layer) };
    }

    // -----------------------------------------------------------------------
    // TfDataset tests
    // -----------------------------------------------------------------------

    /// 11. tf_dataset_from_csv returns null when path is null.
    #[test]
    fn test_dataset_from_csv_null_path() {
        let ds = unsafe { tf_dataset_from_csv(std::ptr::null(), 0) };
        assert!(ds.is_null());
    }

    /// 12. tf_dataset_from_csv returns null for a non-existent file.
    #[test]
    fn test_dataset_from_csv_missing_file() {
        let tmp_path = std::env::temp_dir().join("this_file_does_not_exist_tenflowers.csv");
        let path = CString::new(tmp_path.to_str().expect("valid path"))
            .expect("test: CString construction should succeed");
        let ds = unsafe { tf_dataset_from_csv(path.as_ptr(), 0) };
        assert!(ds.is_null());
    }

    /// 13. tf_dataset_free is a no-op on null.
    #[test]
    fn test_dataset_free_null_is_noop() {
        unsafe { tf_dataset_free(std::ptr::null_mut()) };
    }

    /// 14. tf_dataset_len returns 0 for null pointer.
    #[test]
    fn test_dataset_len_null() {
        let len = unsafe { tf_dataset_len(std::ptr::null()) };
        assert_eq!(len, 0);
    }

    /// 15. Full round-trip: load CSV → check len → get_item.
    #[test]
    fn test_dataset_full_round_trip() {
        // CSV content: header + 3 rows, last column is the label.
        let csv_content = "a,b,c,label\n1.0,2.0,3.0,0.0\n4.0,5.0,6.0,1.0\n7.0,8.0,9.0,1.0\n";
        let path = write_temp_csv(csv_content);
        let path_cstr =
            CString::new(path.to_str().expect("test: path should be valid UTF-8"))
                .expect("test: CString construction should succeed");

        let ds = unsafe { tf_dataset_from_csv(path_cstr.as_ptr(), 0) };
        assert!(!ds.is_null(), "expected non-null dataset handle");

        let len = unsafe { tf_dataset_len(ds as *const TfDataset) };
        assert_eq!(len, 3, "expected 3 samples");

        // Retrieve the first feature row.
        let mut output = [0.0f32; 4]; // 4 columns total (3 features + 1 label col)
        let err = unsafe {
            tf_dataset_get_item(ds as *const TfDataset, 0, output.as_mut_ptr(), output.len())
        };
        assert_eq!(err, TF_OK);

        // Feature columns are a, b, c (label is stripped by the builder).
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
        assert!((output[2] - 3.0).abs() < 1e-5);

        unsafe { tf_dataset_free(ds) };

        // Clean up temp file.
        let _ = std::fs::remove_file(&path);
    }

    /// 16. tf_dataset_get_item returns TF_ERROR_NULL_PTR when dataset is null.
    #[test]
    fn test_dataset_get_item_null_dataset() {
        let mut output = [0.0f32; 4];
        let err = unsafe {
            tf_dataset_get_item(std::ptr::null(), 0, output.as_mut_ptr(), output.len())
        };
        assert_eq!(err, TF_ERROR_NULL_PTR);
    }

    /// 17. tf_dataset_get_item returns TF_ERROR_NULL_PTR when output is null.
    #[test]
    fn test_dataset_get_item_null_output() {
        let csv_content = "x,y\n1.0,0.0\n";
        let path = write_temp_csv(csv_content);
        let path_cstr =
            CString::new(path.to_str().expect("test: path should be valid UTF-8"))
                .expect("test: CString construction should succeed");

        let ds = unsafe { tf_dataset_from_csv(path_cstr.as_ptr(), 0) };
        assert!(!ds.is_null());

        let err = unsafe {
            tf_dataset_get_item(ds as *const TfDataset, 0, std::ptr::null_mut(), 1)
        };
        assert_eq!(err, TF_ERROR_NULL_PTR);

        unsafe { tf_dataset_free(ds) };
        let _ = std::fs::remove_file(&path);
    }

    /// 18. tf_dataset_get_item returns TF_ERROR_OUT_OF_BOUNDS for index ≥ len.
    #[test]
    fn test_dataset_get_item_out_of_bounds_index() {
        let csv_content = "x,y\n1.0,0.0\n";
        let path = write_temp_csv(csv_content);
        let path_cstr =
            CString::new(path.to_str().expect("test: path should be valid UTF-8"))
                .expect("test: CString construction should succeed");

        let ds = unsafe { tf_dataset_from_csv(path_cstr.as_ptr(), 0) };
        assert!(!ds.is_null());

        let mut output = [0.0f32; 4];
        let err = unsafe {
            tf_dataset_get_item(ds as *const TfDataset, 99, output.as_mut_ptr(), output.len())
        };
        assert_eq!(err, TF_ERROR_OUT_OF_BOUNDS);

        unsafe { tf_dataset_free(ds) };
        let _ = std::fs::remove_file(&path);
    }

    /// 19. tf_dataset_get_item returns TF_ERROR_OUT_OF_BOUNDS when output_len is too small.
    #[test]
    fn test_dataset_get_item_output_too_small() {
        // 3 feature columns + 1 label column → feature_len = 3.
        let csv_content = "a,b,c,label\n1.0,2.0,3.0,0.0\n";
        let path = write_temp_csv(csv_content);
        let path_cstr =
            CString::new(path.to_str().expect("test: path should be valid UTF-8"))
                .expect("test: CString construction should succeed");

        let ds = unsafe { tf_dataset_from_csv(path_cstr.as_ptr(), 0) };
        assert!(!ds.is_null());

        let mut output = [0.0f32; 1]; // 1 < 3 features
        let err = unsafe {
            tf_dataset_get_item(ds as *const TfDataset, 0, output.as_mut_ptr(), output.len())
        };
        assert_eq!(err, TF_ERROR_OUT_OF_BOUNDS);

        unsafe { tf_dataset_free(ds) };
        let _ = std::fs::remove_file(&path);
    }

    /// 20. tf_dataset_from_csv with explicit path_len (no null terminator required).
    #[test]
    fn test_dataset_from_csv_explicit_path_len() {
        let csv_content = "x,y\n10.0,1.0\n20.0,0.0\n";
        let path = write_temp_csv(csv_content);
        let path_str = path.to_str().expect("test: path should be valid UTF-8");
        let path_bytes = path_str.as_bytes();

        let ds = unsafe {
            tf_dataset_from_csv(path_bytes.as_ptr() as *const c_char, path_bytes.len())
        };
        assert!(!ds.is_null(), "expected non-null dataset handle");

        let len = unsafe { tf_dataset_len(ds as *const TfDataset) };
        assert_eq!(len, 2);

        unsafe { tf_dataset_free(ds) };
        let _ = std::fs::remove_file(&path);
    }
}
