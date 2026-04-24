/// Build script for tenflowers-ffi.
///
/// When the `c-header-generate` feature is enabled and the
/// `TENFLOWERS_REGENERATE_C_HEADER` environment variable is set to `1`,
/// this script invokes cbindgen to regenerate `include/tenflowers.h` from
/// the Rust source.
///
/// By default (without the env var) the build is a no-op, preserving the
/// committed header as the source of truth during regular development builds.
fn main() {
    pyo3_build_config::use_pyo3_cfgs();
    // pyo3's own build.rs suppresses Python library linking when extension-module
    // feature is enabled (correct for .so files, but test binaries need Python linked).
    // Emit link args directly here to cover both cases on macOS.
    let config = pyo3_build_config::get();
    if let Some(lib_dir) = &config.lib_dir {
        println!("cargo:rustc-link-search=native={lib_dir}");
    }
    if let Some(lib_name) = &config.lib_name {
        println!("cargo:rustc-link-lib={lib_name}");
    }
    #[cfg(feature = "c-header-generate")]
    regenerate_c_header();
}

#[cfg(feature = "c-header-generate")]
fn regenerate_c_header() {
    // Opt-in only — skip regeneration unless the caller sets this env var.
    // This ensures the committed header is preserved during normal builds.
    if std::env::var("TENFLOWERS_REGENERATE_C_HEADER")
        .map(|v| v != "1")
        .unwrap_or(true)
    {
        return;
    }

    let crate_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by cargo");

    let crate_path = std::path::Path::new(&crate_dir);
    let config_path = crate_path.join("cbindgen.toml");

    let config =
        cbindgen::Config::from_file(&config_path).unwrap_or_else(|_| cbindgen::Config::default());

    let output_path = crate_path.join("include").join("tenflowers.h");

    match cbindgen::Builder::new()
        .with_crate(crate_path)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            let written = bindings.write_to_file(&output_path);
            if !written {
                eprintln!(
                    "cbindgen: failed to write header to {}",
                    output_path.display()
                );
                std::process::exit(1);
            }
        }
        Err(err) => {
            eprintln!("cbindgen: header generation failed: {}", err);
            std::process::exit(1);
        }
    }

    // Tell cargo to re-run this build script when relevant inputs change.
    println!("cargo:rerun-if-env-changed=TENFLOWERS_REGENERATE_C_HEADER");
    println!("cargo:rerun-if-changed=src/c_ffi.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
