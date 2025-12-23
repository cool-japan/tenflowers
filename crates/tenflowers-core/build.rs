use std::path::Path;

fn main() {
    // Inform cargo about our custom cfg flags (using single colon for MSRV compatibility)
    println!("cargo:rustc-check-cfg=cfg(cuda_available)");
    println!("cargo:rustc-check-cfg=cfg(selected_blas_backend, values(\"mkl\", \"accelerate\", \"openblas\"))");
    println!("cargo:rustc-check-cfg=cfg(has_blas_backend)");
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

    // CUDA platform validation and library detection
    let cuda_enabled = cfg!(feature = "cuda");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if cuda_enabled && !(target_os == "linux" || target_os == "windows") {
        println!("cargo:warning=CUDA features are only supported on Linux and Windows platforms");
        println!(
            "cargo:warning=CUDA will be disabled on this platform ({})",
            target_os
        );
    }

    // Check if CUDA libraries are actually available for linking
    if cuda_enabled && (target_os == "linux" || target_os == "windows") {
        let cuda_available = detect_cuda_libraries(&target_os);

        if cuda_available {
            // CUDA libraries found - enable linking and set cfg flag
            println!("cargo:rustc-cfg=cuda_available");

            // Add CUDA library search paths
            if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
                if target_os == "windows" {
                    println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
                } else {
                    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
                }
            }

            // Standard CUDA paths
            if target_os == "linux" {
                println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
                println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
            }

            // Link CUDA runtime and driver libraries
            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rustc-link-lib=dylib=cuda");
        } else {
            println!("cargo:warning=CUDA feature enabled but CUDA libraries not found");
            println!("cargo:warning=CUDA functionality will be disabled at runtime");
            println!("cargo:warning=Install CUDA toolkit or set CUDA_PATH environment variable");
        }
    }
}

/// Detect if CUDA libraries are available on the system
fn detect_cuda_libraries(target_os: &str) -> bool {
    // Check CUDA_PATH environment variable first
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let lib_path = if target_os == "windows" {
            format!("{}/lib/x64/cudart.lib", cuda_path)
        } else {
            format!("{}/lib64/libcudart.so", cuda_path)
        };
        if Path::new(&lib_path).exists() {
            return true;
        }
    }

    // Check standard Linux paths
    if target_os == "linux" {
        let standard_paths = [
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/lib/x86_64-linux-gnu/libcudart.so",
            "/usr/lib64/libcudart.so",
        ];

        for path in &standard_paths {
            if Path::new(path).exists() {
                return true;
            }
        }

        // Also check via pkg-config
        if std::process::Command::new("pkg-config")
            .args(["--exists", "cudart"])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return true;
        }
    }

    // Check standard Windows paths
    if target_os == "windows" {
        let standard_paths = [
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64/cudart.lib",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64/cudart.lib",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/lib/x64/cudart.lib",
        ];

        for path in &standard_paths {
            if Path::new(path).exists() {
                return true;
            }
        }
    }

    false
}
