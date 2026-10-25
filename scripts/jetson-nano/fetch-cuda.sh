#!/bin/bash
# Download the CUDA 10.2 toolkit pieces of JetPack 4.6 (L4T R32.6) from NVIDIA's Jetson apt repo and
# unpack them into $1 (default: ./cuda-10.2-sysroot). Only headers, nvcc tools and the libraries ggml links.
set -euo pipefail
OUT=${1:-cuda-10.2-sysroot}
REPO=https://repo.download.nvidia.com/jetson/common
PKGS='cuda-(nvcc|cudart|cudart-dev|driver-dev|nvrtc|nvrtc-dev|cuobjdump|nvprune|nvdisasm)-10-2_|libcublas10_|libcublas-dev_|libcurand-dev-10-2_'
fetch() { # fetch URL [OUTFILE]; OUTFILE "-" or empty means stdout
    if command -v curl >/dev/null; then curl -fsSL -o "${2:--}" "$1"; else wget -q -O "${2:--}" "$1"; fi
}
mkdir -p "$OUT" debs
fetch "$REPO/dists/r32.6/main/binary-arm64/Packages" | awk '/^Filename:/ {print $2}' | grep -E "/($PKGS)" | while read -r f; do
    fetch "$REPO/$f" "debs/$(basename "$f")"
done
for d in debs/*.deb; do
    dpkg-deb -x "$d" "$OUT"
done
echo "CUDA sysroot: $OUT/usr/local/cuda-10.2"
