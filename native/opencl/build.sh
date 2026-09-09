#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
build_dir="${1:-$repo_root/target/release}"
cxx="${CXX:-c++}"

mkdir -p "$build_dir"

"$cxx" \
  -std=c++17 \
  -O3 \
  -DNDEBUG \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Wconversion \
  -Wshadow \
  -Werror \
  -DCL_TARGET_OPENCL_VERSION=120 \
  "$script_dir/janus_bt4_opencl.cpp" \
  -lOpenCL \
  -o "$build_dir/janus-bt4-opencl"

echo "built $build_dir/janus-bt4-opencl"
