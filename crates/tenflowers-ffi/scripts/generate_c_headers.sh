#!/bin/bash
# Generate C headers for TenfloweRS FFI
#
# This script generates C headers from the Rust code to provide
# a stable C API for TenfloweRS.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HEADERS_DIR="$PROJECT_ROOT/include"
VERSION="0.1.0-alpha.2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Generating C headers for TenfloweRS FFI${NC}"
echo "Project root: $PROJECT_ROOT"
echo "Headers directory: $HEADERS_DIR"
echo "Version: $VERSION"
echo ""

# Create headers directory
mkdir -p "$HEADERS_DIR"

# Check if cbindgen is installed
if ! command -v cbindgen &> /dev/null; then
    echo -e "${RED}Error: cbindgen is not installed${NC}"
    echo "Install it with: cargo install cbindgen"
    exit 1
fi

# Generate main header
echo -e "${YELLOW}Generating tenflowers.h...${NC}"
cbindgen \
    --config "$PROJECT_ROOT/cbindgen.toml" \
    --crate tenflowers-ffi \
    --output "$HEADERS_DIR/tenflowers.h" \
    --lang c \
    "$PROJECT_ROOT"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Generated tenflowers.h${NC}"
else
    echo -e "${RED}✗ Failed to generate tenflowers.h${NC}"
    exit 1
fi

# Add version information to header
echo -e "${YELLOW}Adding version information...${NC}"
sed -i.bak "1i\\
/* TenfloweRS C API - Version $VERSION */\\
/* Generated on $(date) */\\
" "$HEADERS_DIR/tenflowers.h"
rm -f "$HEADERS_DIR/tenflowers.h.bak"

# Generate platform-specific headers if needed
# Add support for different platforms here

echo ""
echo -e "${GREEN}Header generation complete!${NC}"
echo "Headers location: $HEADERS_DIR"
echo ""
echo "Next steps:"
echo "  1. Review generated headers for correctness"
echo "  2. Test C API with example programs"
echo "  3. Update documentation with C API usage"
echo "  4. Consider versioning and ABI stability"
