fn main() {
    // Select BLAS backend based on priority when multiple are enabled
    // Priority: MKL > Accelerate > OpenBLAS

    let mkl_enabled = cfg!(feature = "blas-mkl");
    let accelerate_enabled = cfg!(feature = "blas-accelerate");
    let openblas_enabled = cfg!(feature = "blas-openblas");

    if mkl_enabled {
        println!("cargo:rustc-cfg=selected_blas_backend=\"mkl\"");
        println!("cargo:rustc-cfg=has_blas_backend");
    } else if accelerate_enabled {
        println!("cargo:rustc-cfg=selected_blas_backend=\"accelerate\"");
        println!("cargo:rustc-cfg=has_blas_backend");
    } else if openblas_enabled {
        println!("cargo:rustc-cfg=selected_blas_backend=\"openblas\"");
        println!("cargo:rustc-cfg=has_blas_backend");
    }

    // CUDA platform validation
    let cuda_enabled = cfg!(feature = "cuda");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if cuda_enabled && !(target_os == "linux" || target_os == "windows") {
        println!("cargo:warning=CUDA features are only supported on Linux and Windows platforms");
        println!(
            "cargo:warning=CUDA will be disabled on this platform ({})",
            target_os
        );
    }
}
