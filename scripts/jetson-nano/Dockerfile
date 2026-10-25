# Build environment for the Jetson Nano (L4T R32.x, CUDA 10.2, sm_53): Ubuntu 20.04 arm64 + clang 18.
# Run on any arm64 host (e.g. an Apple Silicon Mac with Docker); see docs/build-jetson-nano.md.
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends gcc-9 g++-9 git make ninja-build python3-pip wget gnupg ca-certificates curl ccache pkg-config && pip3 install cmake==3.27.9 && rm -rf /var/lib/apt/lists/*
RUN echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-18 main" > /etc/apt/sources.list.d/llvm.list && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key > /etc/apt/trusted.gpg.d/llvm.asc && apt-get update && apt-get install -y --no-install-recommends clang-18 lld-18 && rm -rf /var/lib/apt/lists/*
