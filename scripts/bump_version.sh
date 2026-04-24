#!/usr/bin/env bash
# Bump the workspace version in Cargo.toml and doc-version strings in lib.rs files.
# Usage: ./scripts/bump_version.sh <new-version>
# Example: ./scripts/bump_version.sh 0.1.1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NEW_VERSION="${1:?Usage: bump_version.sh <new-version>}"

# Validate semver format
if ! [[ "${NEW_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: '${NEW_VERSION}' is not a valid semver (expected X.Y.Z)." >&2
    exit 1
fi

WORKSPACE_TOML="${WORKSPACE_ROOT}/Cargo.toml"
OLD_VERSION=$(grep -m1 '^version = ' "${WORKSPACE_TOML}" | sed 's/version = "\(.*\)"/\1/')

if [[ -z "${OLD_VERSION}" ]]; then
    echo "ERROR: Could not determine current version from ${WORKSPACE_TOML}" >&2
    exit 1
fi

echo "Bumping ${OLD_VERSION} -> ${NEW_VERSION}"
echo ""

# Update workspace Cargo.toml
sed -i.bak "s/^version = \"${OLD_VERSION}\"/version = \"${NEW_VERSION}\"/" "${WORKSPACE_TOML}"
rm -f "${WORKSPACE_TOML}.bak"
echo "Updated ${WORKSPACE_TOML}"

# Update doc version strings in lib.rs files (//! version comments)
for lib_rs in $(find "${WORKSPACE_ROOT}" -name "lib.rs" -not -path "*/target/*" | sort); do
    if grep -q "${OLD_VERSION}" "${lib_rs}"; then
        sed -i.bak "s/${OLD_VERSION}/${NEW_VERSION}/g" "${lib_rs}"
        rm -f "${lib_rs}.bak"
        echo "Updated ${lib_rs}"
    fi
done

echo ""
echo "Done. Review the diff with: git diff"
echo "Then commit: git commit -am 'chore: bump version to ${NEW_VERSION}'"
