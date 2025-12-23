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
from typing import List, Dict, Optional


class CHeaderGenerator:
    """Generator for C header files from Rust FFI"""

    def __init__(self, output_path: Optional[Path] = None):
        self.output_path = output_path or Path("tenflowers.h")
        self.types: List[str] = []
        self.functions: List[str] = []
        self.constants: Dict[str, str] = {}

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

        # Version information
        header.append("/* TenfloweRS FFI Version Information */")
        header.append("#define TENFLOWERS_VERSION_MAJOR 0")
        header.append("#define TENFLOWERS_VERSION_MINOR 1")
        header.append("#define TENFLOWERS_VERSION_PATCH 0")
        header.append("#define TENFLOWERS_VERSION_STRING \"0.1.0-alpha.1\"")
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
