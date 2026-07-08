#!/usr/bin/env python3
"""
C Header Generator for TenfloweRS FFI

This script generates C header files from the Rust FFI bindings,
making it easier to use TenfloweRS from C/C++ code.

Usage:
    python generate_c_header.py [--output path/to/output.h]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Root workspace Cargo.toml, relative to this script:
# crates/tenflowers-ffi/scripts/generate_c_header.py -> ../../../Cargo.toml
_WORKSPACE_CARGO_TOML = Path(__file__).resolve().parents[3] / "Cargo.toml"

# Fallback used only if the workspace Cargo.toml cannot be read (e.g. this
# script is copied out of the repo). Kept in the historical 0.1.0 form so a
# missing manifest fails loudly via an obviously-a-fallback value rather than
# silently emitting a plausible-but-wrong version.
_FALLBACK_VERSION = "0.0.0"


def _read_workspace_version(cargo_toml_path: Path = _WORKSPACE_CARGO_TOML) -> str:
    """Read `[workspace.package] version = "X.Y.Z"` from the root Cargo.toml.

    This avoids hardcoding a version string in this script: every crate in
    the workspace inherits its version from this single source of truth via
    `version.workspace = true`, so the generated C header should too.
    """
    try:
        text = cargo_toml_path.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK_VERSION

    in_workspace_package = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_workspace_package = stripped == "[workspace.package]"
            continue
        if in_workspace_package:
            match = re.match(r'version\s*=\s*"([^"]+)"', stripped)
            if match:
                return match.group(1)
    return _FALLBACK_VERSION


def _parse_semver(version: str) -> Tuple[int, int, int]:
    """Parse a `major.minor.patch[-pre][+build]` string into an int triple."""
    core = version.split("-", 1)[0].split("+", 1)[0]
    parts = core.split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return major, minor, patch


class CHeaderGenerator:
    """Generator for C header files from Rust FFI"""

    def __init__(self, output_path: Optional[Path] = None):
        self.output_path = output_path or Path("tenflowers.h")
        self.types: List[str] = []
        self.functions: List[str] = []
        self.constants: Dict[str, str] = {}
        self.version_string: str = _read_workspace_version()
        self.version_major, self.version_minor, self.version_patch = _parse_semver(
            self.version_string
        )

    def generate_header(self) -> str:
        """Generate the complete C header file content"""
        header = []

        # Header guard
        header.append("#ifndef TENFLOWERS_H")
        header.append("#define TENFLOWERS_H")
        header.append("")

        # Standard includes
        header.append("#include <stddef.h>")
        header.append("#include <stdint.h>")
        header.append("#include <stdbool.h>")
        header.append("")

        # C++ compatibility
        header.append("#ifdef __cplusplus")
        header.append("extern \"C\" {")
        header.append("#endif")
        header.append("")

        # Version information (read from the workspace Cargo.toml so this
        # never drifts from the real crate version — see _read_workspace_version).
        header.append("/* TenfloweRS FFI Version Information */")
        header.append(f"#define TENFLOWERS_VERSION_MAJOR {self.version_major}")
        header.append(f"#define TENFLOWERS_VERSION_MINOR {self.version_minor}")
        header.append(f"#define TENFLOWERS_VERSION_PATCH {self.version_patch}")
        header.append(f"#define TENFLOWERS_VERSION_STRING \"{self.version_string}\"")
        header.append("")

        # Opaque types
        header.append("/* Opaque Types */")
        header.append("typedef struct TenflowersTensor TenflowersTensor;")
        header.append("typedef struct TenflowersDevice TenflowersDevice;")
        header.append("typedef struct TenflowersGradientTape TenflowersGradientTape;")
        header.append("typedef struct TenflowersLayer TenflowersLayer;")
        header.append("typedef struct TenflowersOptimizer TenflowersOptimizer;")
        header.append("typedef struct TenflowersError TenflowersError;")
        header.append("")

        # Result type
        header.append("/* Result Type for Error Handling */")
        header.append("typedef enum {")
        header.append("    TENFLOWERS_OK = 0,")
        header.append("    TENFLOWERS_ERROR_INVALID_ARGUMENT = 1,")
        header.append("    TENFLOWERS_ERROR_OUT_OF_MEMORY = 2,")
        header.append("    TENFLOWERS_ERROR_SHAPE_MISMATCH = 3,")
        header.append("    TENFLOWERS_ERROR_DEVICE_ERROR = 4,")
        header.append("    TENFLOWERS_ERROR_RUNTIME_ERROR = 5,")
        header.append("    TENFLOWERS_ERROR_NOT_IMPLEMENTED = 6,")
        header.append("} TenflowersResult;")
        header.append("")

        # Device type
        header.append("/* Device Types */")
        header.append("typedef enum {")
        header.append("    TENFLOWERS_DEVICE_CPU = 0,")
        header.append("    TENFLOWERS_DEVICE_GPU = 1,")
        header.append("} TenflowersDeviceType;")
        header.append("")

        # Data types
        header.append("/* Data Types */")
        header.append("typedef enum {")
        header.append("    TENFLOWERS_DTYPE_FLOAT32 = 0,")
        header.append("    TENFLOWERS_DTYPE_FLOAT64 = 1,")
        header.append("    TENFLOWERS_DTYPE_FLOAT16 = 2,")
        header.append("    TENFLOWERS_DTYPE_BFLOAT16 = 3,")
        header.append("    TENFLOWERS_DTYPE_INT8 = 4,")
        header.append("    TENFLOWERS_DTYPE_INT16 = 5,")
        header.append("    TENFLOWERS_DTYPE_INT32 = 6,")
        header.append("    TENFLOWERS_DTYPE_INT64 = 7,")
        header.append("    TENFLOWERS_DTYPE_UINT8 = 8,")
        header.append("    TENFLOWERS_DTYPE_UINT16 = 9,")
        header.append("    TENFLOWERS_DTYPE_UINT32 = 10,")
        header.append("    TENFLOWERS_DTYPE_UINT64 = 11,")
        header.append("    TENFLOWERS_DTYPE_BOOL = 12,")
        header.append("} TenflowersDType;")
        header.append("")

        # Tensor creation functions
        header.append("/* Tensor Creation Functions */")
        header.append("TenflowersResult tenflowers_tensor_zeros(")
        header.append("    const size_t* shape,")
        header.append("    size_t ndim,")
        header.append("    TenflowersDType dtype,")
        header.append("    TenflowersDeviceType device,")
        header.append("    TenflowersTensor** out_tensor")
        header.append(");")
        header.append("")

        header.append("TenflowersResult tenflowers_tensor_ones(")
        header.append("    const size_t* shape,")
        header.append("    size_t ndim,")
        header.append("    TenflowersDType dtype,")
        header.append("    TenflowersDeviceType device,")
        header.append("    TenflowersTensor** out_tensor")
        header.append(");")
        header.append("")

        header.append("TenflowersResult tenflowers_tensor_from_data(")
        header.append("    const void* data,")
        header.append("    const size_t* shape,")
        header.append("    size_t ndim,")
        header.append("    TenflowersDType dtype,")
        header.append("    TenflowersDeviceType device,")
        header.append("    TenflowersTensor** out_tensor")
        header.append(");")
        header.append("")

        # Tensor operations
        header.append("/* Tensor Operations */")
        header.append("TenflowersResult tenflowers_tensor_add(")
        header.append("    const TenflowersTensor* a,")
        header.append("    const TenflowersTensor* b,")
        header.append("    TenflowersTensor** out_tensor")
        header.append(");")
        header.append("")

        header.append("TenflowersResult tenflowers_tensor_mul(")
        header.append("    const TenflowersTensor* a,")
        header.append("    const TenflowersTensor* b,")
        header.append("    TenflowersTensor** out_tensor")
        header.append(");")
        header.append("")

        header.append("TenflowersResult tenflowers_tensor_matmul(")
        header.append("    const TenflowersTensor* a,")
        header.append("    const TenflowersTensor* b,")
        header.append("    TenflowersTensor** out_tensor")
        header.append(");")
        header.append("")

        # Tensor properties
        header.append("/* Tensor Properties */")
        header.append("TenflowersResult tenflowers_tensor_shape(")
        header.append("    const TenflowersTensor* tensor,")
        header.append("    size_t* out_shape,")
        header.append("    size_t* out_ndim")
        header.append(");")
        header.append("")

        header.append("TenflowersResult tenflowers_tensor_dtype(")
        header.append("    const TenflowersTensor* tensor,")
        header.append("    TenflowersDType* out_dtype")
        header.append(");")
        header.append("")

        header.append("TenflowersResult tenflowers_tensor_device(")
        header.append("    const TenflowersTensor* tensor,")
        header.append("    TenflowersDeviceType* out_device")
        header.append(");")
        header.append("")

        # Memory management
        header.append("/* Memory Management */")
        header.append("void tenflowers_tensor_free(TenflowersTensor* tensor);")
        header.append("")

        # Gradient tape
        header.append("/* Gradient Tape */")
        header.append("TenflowersResult tenflowers_gradient_tape_create(")
        header.append("    TenflowersGradientTape** out_tape")
        header.append(");")
        header.append("")

        header.append("TenflowersResult tenflowers_gradient_tape_backward(")
        header.append("    TenflowersGradientTape* tape,")
        header.append("    const TenflowersTensor* loss")
        header.append(");")
        header.append("")

        header.append("TenflowersResult tenflowers_gradient_tape_gradient(")
        header.append("    const TenflowersGradientTape* tape,")
        header.append("    const TenflowersTensor* tensor,")
        header.append("    TenflowersTensor** out_gradient")
        header.append(");")
        header.append("")

        header.append("void tenflowers_gradient_tape_free(TenflowersGradientTape* tape);")
        header.append("")

        # Error handling
        header.append("/* Error Handling */")
        header.append("const char* tenflowers_error_message(const TenflowersError* error);")
        header.append("void tenflowers_error_free(TenflowersError* error);")
        header.append("")

        # Device management
        header.append("/* Device Management */")
        header.append("TenflowersResult tenflowers_set_default_device(TenflowersDeviceType device);")
        header.append("TenflowersResult tenflowers_get_default_device(TenflowersDeviceType* out_device);")
        header.append("")

        # Utility functions
        header.append("/* Utility Functions */")
        header.append("const char* tenflowers_version();")
        header.append("bool tenflowers_is_gpu_available();")
        header.append("")

        # Close extern C
        header.append("#ifdef __cplusplus")
        header.append("}")
        header.append("#endif")
        header.append("")

        # Close header guard
        header.append("#endif /* TENFLOWERS_H */")

        return "\n".join(header)

    def write_header(self):
        """Write the generated header to file"""
        content = self.generate_header()

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Generated C header: {self.output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate C header file for TenfloweRS FFI"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("tenflowers.h"),
        help="Output header file path (default: tenflowers.h)",
    )

    args = parser.parse_args()

    generator = CHeaderGenerator(output_path=args.output)
    generator.write_header()

    return 0


if __name__ == "__main__":
    sys.exit(main())
