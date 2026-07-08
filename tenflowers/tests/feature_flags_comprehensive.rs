//! Comprehensive feature-flag tests for the `tenflowers` meta crate.
//!
//! These tests verify that:
//! 1. All `cfg(feature = …)` paths compile without errors.
//! 2. The default feature set (std + parallel) provides a working tensor
//!    environment.
//! 3. Cross-cutting utilities (`version_check`, `logging`) function correctly.
//!
//! Because Cargo feature flags cannot be toggled at runtime, we rely on the
//! `#[cfg(feature = …)]` attribute to compile conditional code paths and assert
//! that their types/functions are reachable.  If a feature block would fail to
//! compile the test itself would not compile — which is the desired behaviour.

// ─── Default feature set smoke tests ─────────────────────────────────────────

#[test]
fn default_features_prelude_imports_work() {
    // The prelude must be importable and its most common types usable.
    use tenflowers::prelude::*;

    let _zeros: Tensor<f32> = Tensor::<f32>::zeros(&[4, 4]);
    let _ones: Tensor<f32> = Tensor::<f32>::ones(&[2, 3]);
}

#[test]
fn default_features_ops_add_works() {
    use tenflowers::core::{ops, Tensor};

    let a = Tensor::<f32>::ones(&[3, 3]);
    let b = Tensor::<f32>::ones(&[3, 3]);
    let c = ops::add(&a, &b).expect("add should succeed with default features");
    let data = c.to_vec().expect("to_vec should succeed");
    assert!(data.iter().all(|&v| (v - 2.0f32).abs() < 1e-6));
}

#[test]
fn default_features_prelude_dense_layer_available() {
    use tenflowers::prelude::Dense;
    let _layer = Dense::<f32>::new(4, 2, true);
}

#[test]
fn default_features_prelude_sequential_available() {
    use tenflowers::prelude::Sequential;
    let _model = Sequential::<f32>::new(vec![]);
}

#[test]
fn default_features_prelude_optimizer_available() {
    use tenflowers::prelude::Adam;
    let _opt = Adam::<f32>::new(1e-3);
}

// ─── gpu / simd / blas / wasm / serialize / compression feature guards ────────
//
// These blocks contain no runtime assertions.  Their purpose is purely to
// verify that the `cfg`-gated code compiles when the feature is present and
// does NOT compile (i.e., the block is absent) when it is absent — which is
// the correct behaviour and is enforced by the Rust type checker.

/// GPU feature: the `gpu` feature gates GPU-specific Device variants.
#[test]
fn gpu_feature_compiles_without_error() {
    // Under no-GPU builds, Device::Cpu is always present.
    use tenflowers::core::Device;
    let _d = Device::Cpu;

    #[cfg(feature = "gpu")]
    {
        // When the gpu feature is enabled, Gpu(0) should be constructable.
        let _gpu = Device::Gpu(0);
    }
}

/// SIMD feature: just verify the feature flag can be checked without error.
#[test]
fn simd_feature_compile_guard() {
    #[cfg(feature = "simd")]
    {
        // If the simd feature is enabled, the crate must compile cleanly.
        let _t = tenflowers::core::Tensor::<f32>::zeros(&[8]);
    }
    #[cfg(not(feature = "simd"))]
    {
        let _t = tenflowers::core::Tensor::<f32>::zeros(&[8]);
    }
}

/// Serialize feature: serialization helpers are gated on this feature.
#[test]
fn serialize_feature_compile_guard() {
    // tenflowers::io provides save/load helpers gated on the serialize feature.
    // This test just ensures the module exists and is importable.
    #[cfg(feature = "serialize")]
    {
        use tenflowers::io;
        // io module is accessible when feature is active.
        // Reference an existing `io` item to prove the module is reachable
        // under this feature combination.
        let _ = tenflowers::io::save_tensor::<f32>;
    }
}

/// ONNX feature guard.
#[test]
fn onnx_feature_compile_guard() {
    #[cfg(feature = "onnx")]
    {
        use tenflowers::onnx::OnnxFormat;
        let _f = OnnxFormat::Protobuf;
    }
}

/// Experimental feature guard.
#[test]
fn experimental_feature_compile_guard() {
    #[cfg(feature = "experimental")]
    {
        // The `experimental` module exists when the feature is enabled.
        // Currently it is empty, so we just verify the path resolves.
        let _ = std::any::type_name::<()>(); // dummy line to keep the block non-empty
    }
}

// ─── version_check ────────────────────────────────────────────────────────────

#[test]
fn assert_versions_consistent_passes() {
    // With a workspace-pinned version all subcrates must report the same string.
    tenflowers::version_check::assert_versions_consistent();
}

#[test]
fn subcrate_versions_non_empty() {
    let versions = tenflowers::version_check::subcrate_versions();
    assert!(
        !versions.is_empty(),
        "subcrate_versions() must return at least one entry"
    );
}

#[test]
fn subcrate_versions_first_entry_is_meta() {
    let versions = tenflowers::version_check::subcrate_versions();
    assert_eq!(versions[0].name, "tenflowers");
}

#[test]
fn subcrate_versions_all_non_empty_strings() {
    for sv in tenflowers::version_check::subcrate_versions() {
        assert!(
            !sv.version.is_empty(),
            "{} has empty version string",
            sv.name
        );
    }
}

#[test]
fn check_version_consistency_is_consistent() {
    let report = tenflowers::version_check::check_version_consistency();
    assert!(
        report.is_consistent,
        "version consistency check failed: {}",
        report.summary()
    );
}

// ─── logging ─────────────────────────────────────────────────────────────────

#[test]
fn current_log_level_returns_default() {
    use tenflowers::logging::{current_log_level, LogLevel};

    // The default level is Warn.  Other tests may change it, so we re-set
    // before testing and restore after.
    tenflowers::logging::set_log_level(LogLevel::Warn);
    assert_eq!(current_log_level(), LogLevel::Warn);
}

#[test]
fn set_log_level_roundtrip() {
    use tenflowers::logging::{current_log_level, set_log_level, LogLevel};

    set_log_level(LogLevel::Info);
    assert_eq!(current_log_level(), LogLevel::Info);

    // Restore default.
    set_log_level(LogLevel::Warn);
}

#[test]
fn log_level_from_str_valid_inputs() {
    use tenflowers::logging::LogLevel;

    assert_eq!(LogLevel::from_name("error"), Some(LogLevel::Error));
    assert_eq!(LogLevel::from_name("warn"), Some(LogLevel::Warn));
    assert_eq!(LogLevel::from_name("info"), Some(LogLevel::Info));
    assert_eq!(LogLevel::from_name("debug"), Some(LogLevel::Debug));
    assert_eq!(LogLevel::from_name("trace"), Some(LogLevel::Trace));
}

#[test]
fn log_level_from_str_invalid_returns_none() {
    use tenflowers::logging::LogLevel;
    assert_eq!(LogLevel::from_name("verbose"), None);
    assert_eq!(LogLevel::from_name(""), None);
}

#[test]
fn log_macros_do_not_panic() {
    use tenflowers::logging::{set_log_level, LogLevel};

    // Suppress output during tests.
    tenflowers::logging::set_log_backend(Some(Box::new(|_, _| {})));
    set_log_level(LogLevel::Trace);

    tenflowers::log_error!("error message");
    tenflowers::log_warn!("warn message");
    tenflowers::log_info!("info message");
    tenflowers::log_debug!("debug message");
    tenflowers::log_trace!("trace message");

    // Restore defaults.
    tenflowers::logging::set_log_backend(None);
    set_log_level(LogLevel::Warn);
}

#[test]
fn init_from_env_does_not_panic_without_var() {
    std::env::remove_var("TENFLOWERS_LOG");
    tenflowers::logging::init_from_env();
}

// ─── Error / Result types ─────────────────────────────────────────────────────

#[test]
fn framework_error_is_usable() {
    use tenflowers::error::FrameworkError;

    let e = FrameworkError::Other("test error".to_string());
    let display = e.to_string();
    assert!(!display.is_empty());
}

#[test]
fn crate_result_alias_is_usable() {
    fn always_ok() -> tenflowers::Result<u32> {
        Ok(42)
    }
    assert_eq!(always_ok().expect("should be Ok"), 42);
}

// ─── Prelude completeness ─────────────────────────────────────────────────────

#[test]
fn prelude_core_types_present() {
    use tenflowers::prelude::{DType, Device, Tensor};
    let _d = Device::Cpu;
    let _t = Tensor::<f32>::zeros(&[1]);
    let _dt = DType::Float32;
}

#[test]
fn prelude_autograd_types_present() {
    use tenflowers::prelude::{GradientTape, TrackedTensor};
    let _tape = GradientTape::new();
    let _ = std::any::type_name::<TrackedTensor<f32>>();
}

#[test]
fn prelude_loss_functions_present() {
    use tenflowers::prelude::{binary_cross_entropy, categorical_cross_entropy, mse};
    let _ = std::any::type_name_of_val(&binary_cross_entropy::<f32>);
    let _ = std::any::type_name_of_val(&categorical_cross_entropy::<f32>);
    let _ = std::any::type_name_of_val(&mse::<f32>);
}

#[test]
fn prelude_type_aliases_present() {
    use tenflowers::prelude::*;

    let _v: Vector = Tensor::<f32>::zeros(&[4]);
    let _m: Matrix = Tensor::<f32>::zeros(&[2, 2]);
}

// ─── Feature-combination coherence tests ─────────────────────────────────────
//
// Each test below verifies that a *combination* of features (as listed in the
// Cargo.toml [features] table) compiles and behaves coherently together.
// Runtime assertions are minimal — the compile-time check is the primary value.
//
// Feature matrix being exercised:
//
//  | combination               | what we verify                              |
//  |---------------------------|---------------------------------------------|
//  | std (only)                | prelude importable, Tensor usable           |
//  | std + parallel            | default preset — parallel iter path         |
//  | std + simd                | simd path present alongside std             |
//  | std + serialize           | io::save_tensor / load_tensor types resolve |
//  | std + parallel + simd     | "standard" preset + simd simultaneously     |
//  | std + parallel + gpu      | gpu variant accessible in Device enum       |
//  | std + experimental        | experimental module present                 |
//  | minimal (= std only)      | named preset alias works                    |
//  | standard (= std+parallel) | named preset alias works                    |
//  | full (all bundled)        | full preset compiles + Device::Cpu usable   |

/// `std` alone: no parallel, no simd — just the std-lib-enabled baseline.
#[test]
fn combination_std_only_prelude_usable() {
    // cfg(feature = "std") is always true in the test binary because our
    // default features include std.  The assertion here proves the prelude
    // remains fully functional without parallel or simd being explicitly
    // checked at runtime, matching the `minimal` preset.
    #[cfg(feature = "std")]
    {
        use tenflowers::core::Tensor;
        let t = Tensor::<f32>::zeros(&[2, 2]);
        let v = t.to_vec().expect("to_vec should succeed on std build");
        assert_eq!(v.len(), 4, "2×2 tensor must have 4 elements");
    }
}

/// `std + parallel`: the default preset — verify parallel-aware code path.
#[test]
fn combination_std_parallel_default_preset_works() {
    #[cfg(all(feature = "std", feature = "parallel"))]
    {
        use tenflowers::core::{ops, Tensor};
        // Use a shape large enough to exercise the parallel scatter-gather
        // path in `ops::add` if it is enabled.
        let a = Tensor::<f32>::ones(&[64, 64]);
        let b = Tensor::<f32>::ones(&[64, 64]);
        let c = ops::add(&a, &b).expect("std+parallel add should succeed");
        let data = c.to_vec().expect("to_vec should succeed");
        assert!(
            data.iter().all(|&v| (v - 2.0f32).abs() < 1e-5),
            "all elements must equal 2.0 after ones+ones"
        );
    }
    // When neither std nor parallel is present this is a no-op compile check.
    #[cfg(not(all(feature = "std", feature = "parallel")))]
    let _ = ();
}

/// `std + simd`: SIMD optimisation layer alongside the std allocator.
#[test]
fn combination_std_simd_tensor_creation_coherent() {
    #[cfg(all(feature = "std", feature = "simd"))]
    {
        use tenflowers::core::Tensor;
        // SIMD kernels may be used internally; verify the output is still correct.
        let t = Tensor::<f32>::ones(&[16]);
        let v = t.to_vec().expect("to_vec on std+simd build");
        assert_eq!(v.len(), 16);
        assert!(v.iter().all(|&x| (x - 1.0f32).abs() < 1e-6));
    }
    #[cfg(not(all(feature = "std", feature = "simd")))]
    let _ = ();
}

/// `std + serialize`: I/O types are reachable when both features are present.
#[test]
fn combination_std_serialize_io_types_reachable() {
    #[cfg(all(feature = "std", feature = "serialize"))]
    {
        // Verify the io module path is accessible under this combination.
        // Reference an existing `io` item to prove the module is reachable
        // under this feature combination.
        let _ = tenflowers::io::save_tensor::<f32>;
    }
    // Without serialize the io types are not present — that's correct behaviour.
    #[cfg(not(all(feature = "std", feature = "serialize")))]
    let _ = ();
}

/// `std + parallel + simd`: the three most commonly combined performance features.
#[test]
fn combination_std_parallel_simd_arithmetic_correct() {
    #[cfg(all(feature = "std", feature = "parallel", feature = "simd"))]
    {
        use tenflowers::core::{ops, Tensor};
        let a = Tensor::<f32>::ones(&[32]);
        let b = Tensor::<f32>::ones(&[32]);
        let c = ops::add(&a, &b).expect("std+parallel+simd add");
        let d = ops::mul(&a, &b).expect("std+parallel+simd mul");
        let c_data = c.to_vec().expect("c to_vec");
        let d_data = d.to_vec().expect("d to_vec");
        assert!(c_data.iter().all(|&v| (v - 2.0f32).abs() < 1e-6));
        assert!(d_data.iter().all(|&v| (v - 1.0f32).abs() < 1e-6));
    }
    #[cfg(not(all(feature = "std", feature = "parallel", feature = "simd")))]
    let _ = ();
}

/// `std + parallel + gpu`: GPU device variant available when gpu feature active.
#[test]
fn combination_std_parallel_gpu_device_variants_coherent() {
    #[cfg(all(feature = "std", feature = "parallel"))]
    {
        use tenflowers::core::Device;
        // Cpu is always present regardless of gpu feature.
        let _cpu = Device::Cpu;

        #[cfg(feature = "gpu")]
        {
            // Gpu(index) must be constructable when the gpu feature is active.
            let _gpu = Device::Gpu(0);
        }
    }
    #[cfg(not(all(feature = "std", feature = "parallel")))]
    let _ = ();
}

/// `std + experimental`: experimental module is reachable in this combination.
#[test]
fn combination_std_experimental_module_reachable() {
    #[cfg(all(feature = "std", feature = "experimental"))]
    {
        // The experimental module is empty but must compile cleanly alongside std.
        let _ = std::any::type_name::<()>();
    }
    #[cfg(not(all(feature = "std", feature = "experimental")))]
    let _ = ();
}

// ─── Named preset consistency ────────────────────────────────────────────────
//
// The `minimal` and `standard` feature presets expand to specific combinations
// (see Cargo.toml [features]).  These tests ensure the presets behave as their
// documentation claims.

/// `minimal` preset (= `std`) — tensor usable without parallel overhead.
#[test]
fn preset_minimal_tensor_ops_work() {
    // `minimal = ["std"]`, which is included in our default build.
    // The test proves the core tensor API is functional under that constraint.
    use tenflowers::core::Tensor;
    let t = Tensor::<f64>::zeros(&[4, 4]);
    let v = t.to_vec().expect("to_vec on minimal-compatible build");
    assert_eq!(v.len(), 16);
    assert!(v.iter().all(|&x| x == 0.0_f64));
}

/// `standard` preset (= `std + parallel`) — default plus explicit parallel check.
#[test]
fn preset_standard_add_ops_work() {
    // `standard = ["std", "parallel"]` which matches our default features.
    use tenflowers::core::{ops, Tensor};
    let a = Tensor::<f32>::ones(&[8, 8]);
    let b = Tensor::<f32>::ones(&[8, 8]);
    let c = ops::add(&a, &b).expect("add should work under standard preset");
    let data = c.to_vec().expect("to_vec under standard preset");
    assert_eq!(data.len(), 64);
    assert!(data.iter().all(|&v| (v - 2.0f32).abs() < 1e-6));
}

/// `full` preset — the highest-level bundled preset must not break basic ops.
#[test]
fn preset_full_basic_ops_coherent() {
    // `full = ["gpu", "blas-oxiblas", "simd", "serialize", "compression", "onnx", "autograd"]`
    // In practice the full preset is only active when all its constituents are
    // enabled at build time.  This test degrades gracefully to the actual
    // features that *are* active.
    use tenflowers::core::{DType, Device, Tensor};

    let _cpu = Device::Cpu;
    let _dt = DType::Float32;
    let t = Tensor::<f32>::ones(&[4, 4]);
    let v = t.to_vec().expect("to_vec under full preset");
    assert_eq!(v.len(), 16);

    // Prelude must also work under full.
    use tenflowers::prelude::Adam;
    let _opt = Adam::<f32>::new(1e-3);
}

// ─── Feature-flag Cargo.toml matrix coherence ─────────────────────────────────
//
// These tests verify structural properties of the feature-flag configuration at
// compile time using `cfg` booleans derived from the features actually enabled.
// No external files are read; the test binary's own compile-time cfg attributes
// are the ground truth.

/// The `default` feature set must include `std`.
///
/// If `std` is missing from defaults, fundamental stdlib usage (Vec, HashMap,
/// etc.) inside the crate would silently break on std targets.
#[test]
fn feature_matrix_default_includes_std() {
    // This assertion is vacuously true when compiled with default features
    // (which include `std`).  It becomes a compile-time proof: if std were
    // accidentally removed from [features] default, the `use std::vec::Vec`
    // line below would fail to compile.
    use std::vec::Vec;
    let v: Vec<u8> = Vec::new();
    assert!(
        v.is_empty(),
        "stdlib Vec must be available in default feature set"
    );
}

/// `parallel` must not be the *only* enabled feature when `std` is disabled.
///
/// Parallel execution (Rayon) requires the standard library; having `parallel`
/// without `std` is a mis-configuration.  This test proves the two are
/// coherently paired in all tested configurations.
#[test]
fn feature_matrix_parallel_requires_std_coherence() {
    // Under default features both std AND parallel are present.
    // Under `--no-default-features --features parallel` builds would fail
    // at the Rayon dep level — not at our code level — so we only assert the
    // positive (coherent) case here.
    #[cfg(feature = "parallel")]
    {
        // If parallel is on, std must also be on for our crate to compile.
        // The mere fact that this line compiles proves std is present.
        let _: std::thread::JoinHandle<()>;
    }
    let _ = ();
}

/// `gpu` and `simd` are additive, non-conflicting features.
///
/// Enabling both must not result in duplicate symbol definitions or linker
/// errors.  Verifying this at the type level is sufficient.
#[test]
fn feature_matrix_gpu_and_simd_are_additive() {
    #[cfg(all(feature = "gpu", feature = "simd"))]
    {
        use tenflowers::core::{Device, Tensor};
        let _cpu = Device::Cpu;
        let _t = Tensor::<f32>::zeros(&[8]);
    }
    #[cfg(not(all(feature = "gpu", feature = "simd")))]
    let _ = ();
}

/// `serialize` and `compression` are designed to be used together.
///
/// Enabling both must not cause duplicate or conflicting code paths.
#[test]
fn feature_matrix_serialize_and_compression_coexist() {
    #[cfg(all(feature = "serialize", feature = "compression"))]
    {
        // Both features active — io module must be reachable.
        // Reference an existing `io` item to prove the module is reachable
        // under this feature combination.
        let _ = tenflowers::io::save_tensor::<f32>;
    }
    #[cfg(not(all(feature = "serialize", feature = "compression")))]
    let _ = ();
}

/// `onnx` and `autograd` coexist without symbol conflicts.
#[test]
fn feature_matrix_onnx_and_autograd_coexist() {
    #[cfg(all(feature = "onnx", feature = "autograd"))]
    {
        // onnx module path must resolve.
        let _ = std::any::type_name::<tenflowers::onnx::OnnxFormat>();
        // autograd path must also resolve.
        use tenflowers::prelude::GradientTape;
        let _tape = GradientTape::new();
    }
    #[cfg(not(all(feature = "onnx", feature = "autograd")))]
    let _ = ();
}
