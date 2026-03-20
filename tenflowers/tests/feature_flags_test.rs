//! Feature-flag tests for the `tenflowers` meta crate.
//!
//! These tests verify that the crate's feature-gated paths compile correctly
//! and that utilities such as `version_info()` work under all build configurations.

// ---------------------------------------------------------------------------
// version_info() — always available, no feature flags required
// ---------------------------------------------------------------------------

#[test]
fn test_version_info_returns_non_empty_version() {
    let info = tenflowers::version_info();
    assert!(
        !info.version.is_empty(),
        "version_info().version must not be empty"
    );
}

#[test]
fn test_version_info_pkg_name() {
    let info = tenflowers::version_info();
    assert_eq!(
        info.pkg_name, "tenflowers",
        "pkg_name must match the crate name"
    );
}

#[test]
fn test_version_info_description_non_empty() {
    let info = tenflowers::version_info();
    assert!(
        !info.description.is_empty(),
        "version_info().description must not be empty"
    );
}

#[test]
fn test_version_info_display_contains_version() {
    let info = tenflowers::version_info();
    let display = info.to_string();
    assert!(
        display.contains(info.version),
        "Display output must contain the version string"
    );
}

#[test]
fn test_version_info_display_contains_name() {
    let info = tenflowers::version_info();
    let display = info.to_string();
    assert!(
        display.contains(info.pkg_name),
        "Display output must contain the package name"
    );
}

#[test]
fn test_version_info_equality() {
    // Two separate calls must return identical structs.
    let a = tenflowers::version_info();
    let b = tenflowers::version_info();
    assert_eq!(a, b);
}

#[test]
fn test_version_info_debug_impl() {
    let info = tenflowers::version_info();
    let debug = format!("{:?}", info);
    assert!(
        debug.contains("VersionInfo"),
        "Debug should contain type name"
    );
}

// ---------------------------------------------------------------------------
// version() function — thin wrapper, always available
// ---------------------------------------------------------------------------

#[test]
fn test_version_function_matches_constant() {
    assert_eq!(
        tenflowers::version(),
        tenflowers::VERSION,
        "version() must equal the VERSION constant"
    );
}

#[test]
fn test_version_function_non_empty() {
    assert!(!tenflowers::version().is_empty());
}

// ---------------------------------------------------------------------------
// Default features — std + parallel
// ---------------------------------------------------------------------------

#[test]
fn test_default_features_tensor_creation() {
    // With default features (std, parallel), basic tensor creation must work.
    use tenflowers::core::Tensor;
    let t = Tensor::<f32>::zeros(&[4, 4]);
    assert_eq!(t.shape().dims(), &[4, 4]);
}

#[test]
fn test_default_features_ops_available() {
    // ops module must be accessible through the prelude.
    use tenflowers::core::{ops, Tensor};
    let a = Tensor::<f32>::ones(&[2, 2]);
    let b = Tensor::<f32>::ones(&[2, 2]);
    let _ = ops::add(&a, &b).expect("add must succeed");
}

// ---------------------------------------------------------------------------
// Module aliases compile-time smoke tests
// ---------------------------------------------------------------------------

#[test]
fn test_macros_module_is_accessible() {
    // The macros module is declared as pub and should be accessible.
    // There are no items to call directly since all are declarative macros,
    // but we can verify the path resolves by using the macro.
    let t = tenflowers::tensor![1.0f32, 2.0, 3.0];
    assert_eq!(t.shape().dims(), &[3]);
}

#[test]
fn test_common_module_result_type() {
    // common::Result<T> should be usable as a type alias.
    fn _returns_result() -> tenflowers::common::Result<u32> {
        Ok(42)
    }
    assert_eq!(_returns_result().expect("_returns_result must succeed"), 42);
}
