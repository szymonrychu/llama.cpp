#!/bin/bash
# Configure and build llama.cpp for the Jetson Nano inside the scripts/jetson-nano Docker image.
# Expects the repo at /src and the CUDA 10.2 sysroot mounted at /usr/local/cuda; output in /src/build-jetson-nano.
set -euo pipefail
cd /src
export PATH=/usr/local/cuda/bin:$PATH TMPDIR=${TMPDIR:-/src/.tmp} CCACHE_DIR=${CCACHE_DIR:-/src/.ccache}
mkdir -p "$TMPDIR"
cmake -B build-jetson-nano -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=clang++-18 -DCMAKE_CUDA_ARCHITECTURES=53 \
    -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_FLAGS="--cuda-path=/usr/local/cuda" \
    -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 \
    -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a -DGGML_OPENMP=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=ON \
    -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld-18 -pthread" "$@"
cmake --build build-jetson-nano -j"$(nproc)" --target llama-server llama-mtmd-cli llama-bench llama-cli test-backend-ops
