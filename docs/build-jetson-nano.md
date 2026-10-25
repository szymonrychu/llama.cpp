# llama.cpp on the Jetson Nano (CUDA 10.2, sm_53)

This branch is a hobby fork. It is not meant for upstream. It runs current llama.cpp, including multimodal models,
on the GPU of a 2019 Jetson Nano 4 GB. It is a for-fun project: a Nano draws 5-10 W and generates about 5 tokens/s
on a 2B model. Any current single-board computer with a newer GPU will do better.

## Hardware and software limits

| | |
|---|---|
| Board | Jetson Nano 4 GB (B01), Tegra X1 GPU: Maxwell, compute capability 5.3, 128 cores, one SM |
| OS | L4T R32.6.1 (JetPack 4.6), Ubuntu 20.04 user space |
| CUDA | 10.2 (the newest NVIDIA ships for this board) |
| Memory | 4 GB shared between CPU and GPU, about 23 GB/s GPU read bandwidth (measured) |

Upstream llama.cpp does not build for this board:
- ggml-cuda includes `cuda_bf16.h`, which CUDA 11 added.
- The CUDA sources use C++17, and nvcc 10.2 only supports C++14.

## How it is built

The build runs on an arm64 host in Docker, not on the Nano. clang 18 compiles the CUDA code in C++17 mode against the
CUDA 10.2 headers and libraries from NVIDIA's Jetson apt repository. The output is binaries that run on the Nano with
its stock driver.

```sh
docker build -t llama-jetson-nano scripts/jetson-nano
docker run --rm -v $PWD:/src -w /src llama-jetson-nano scripts/jetson-nano/fetch-cuda.sh /src/cuda-10.2-sysroot
docker run --rm -v $PWD:/src -v $PWD/cuda-10.2-sysroot/usr/local/cuda-10.2:/usr/local/cuda llama-jetson-nano \
    /src/scripts/jetson-nano/build.sh
# copy build-jetson-nano/bin/* to the Nano (binaries use $ORIGIN rpath for the shared libs)
```

On the Nano, run with `-fa off` (see Known issues).

## Changes

Every change is gated on one of three conditions, so other builds keep upstream behaviour:
- CUDA < 11 (`CUDART_VERSION < 11000`), which no other supported target uses
- clang as the CUDA compiler
- compute capability exactly 5.3 (`GGML_CUDA_CC_TEGRA_X1`), which only the Tegra X1 (Jetson Nano / TX1) has

### Build with clang and CUDA 10.2

- **`ggml-cuda/CMakeLists.txt`:** when `CMAKE_CUDA_COMPILER_ID` is Clang, the nvcc-only flags (`-use_fast_math`, `-extended-lambda`, `-Xcompiler`) become their clang equivalents.
- **`ggml-cuda/vendors/cuda10-bf16.h`:** CUDA 10.2 has no bf16 type, so this minimal `nv_bfloat16` / `nv_bfloat162` keeps the real bf16 bit pattern and does its arithmetic in float. It is included only when CUDA < 11.
- **`ggml-cuda/vendors/cuda.h`:** CUDA 10 versions of two CUDA 11 API defaults (`cudaStreamWaitEvent` without flags), plus `CUBLAS_GEMM_DEFAULT_TENSOR_OP` mapped to `CUBLAS_GEMM_DEFAULT`. cuBLAS 10.2 on Maxwell fails some fp16 `cublasGemmEx` calls with the tensor-op algorithm.
- **`ggml-cuda/softmax.cu`:** the cooperative-groups grid-sync kernel compiles to `NO_DEVICE_CODE` under CUDA 10. CUDA 10's `grid_group` needs relocatable device code, and the kernel is only launched on GPUs with cooperative launch, which the Nano lacks.

### Tegra X1 fixes

- **Register cap:** sm_53 allows 32K registers per block, so a 1024-thread block can use at most 32 registers per thread. clang allocated up to 64 for some norm kernels, which then failed with "too many resources requested for launch". Builds that target only `CMAKE_CUDA_ARCHITECTURES=53` now pass `--maxrregcount=32` to ptxas. Kernels with `__launch_bounds__` (MMQ, MMVQ, flash attention) keep their own register budget.
- **fp16:** the Tegra X1 has native fp16 at twice the fp32 rate, but ggml only enabled fp16 from Pascal. `fp16_available`, `fast_fp16_hardware_available`, `FP16_AVAILABLE` and the `dequantize_block_q8_0_f16` guard now include cc 5.3. Prompt processing uses fp16 cuBLAS GEMMs and fp16 dequantization: 22.9 -> 81 tokens/s at 256-token batches (Qwen3-VL-2B Q4_K_M).

### Single-token K-quant mat-vec (`ggml-cuda/mmv-sm53.cu`)

Generation is mostly Q4_K and Q6_K matrix-vector products. MMVQ relies on `__dp4a`, which sm_53 lacks. ggml emulates it with byte extracts and Maxwell's slow integer multiplies, and gets about 3 GB/s.

The new kernel works with y in float:
- **Dequantization:** a 4-bit value q is OR-ed into the mantissa of 1.0f, which gives exactly f = 1 + q/16. Then q·y = 16·(f·y − y), so each weight costs one shift, one LOP3 and one FMA.
- **y correction:** a small pre-pass sums y over 16-element groups once per mat-vec. Each sub-block applies the `−16·y` and `dmin·m` terms once.
- **Q4_K loads:** 8 lanes cover one 256-weight super-block. Each lane loads 16 bytes of `qs`, and one broadcast 16-byte load fetches d, dmin and all 12 scale bytes.
- **Q6_K:** the same float trick on 6-bit values with 16-bit loads, because its blocks are 210 bytes.
- **Row reuse:** each warp computes 4 rows from one load of y. The one-row-per-warp form was bound by y re-reads through L2/texture cache, not by DRAM or ALU. 8 rows per warp was slower because of register pressure.

It is used only for single-token, non-batched, contiguous Q4_K/Q6_K mat-vecs on cc 5.3. Everything else, including other GPUs without `__dp4a`, still goes through MMVQ.

Test: `tests/test-backend-ops.cpp` gains MUL_MAT n=1 cases at real model shapes (m 2048/6273, k 2048/6144, Q4_K and Q6_K). The `MUL_MAT ... n=1` Q4_K/Q6_K set passes 28/28 on the Nano.

## Measured (Jetson Nano 4 GB, MAXN, jetson_clocks, Qwen3-VL-2B-Instruct Q4_K_M, llama-bench, `-fa 0`)

| Build | tg (tokens/s) | pp32 (tokens/s) | pp256 (tokens/s) |
|---|---|---|---|
| CUDA, fp16 gated off for cc 5.3 (upstream behaviour) | 2.81 | 18.1 | not run (pp512 hit the GPU watchdog) |
| + fp16 on cc 5.3 | 2.81 | 23.0 | 78.6 |
| + mmv-sm53 kernel | **5.04** | - | 81.3 |

The average Q4_K mat-vec call went from 1.31 ms (MMVQ) to 0.59 ms. The Q6_K output layer (151936x2048) went from 64 ms to 40 ms per
token. The kernels now read about 6.5 GB/s of weights; a plain read probe on the same GPU reaches 23 GB/s.
In llama-server with JSON-schema sampling, an image + short JSON answer takes 15-25 s (about 4 tokens/s), down from 25-40 s.

## Known issues

- Flash attention: the tile kernel hangs at 512-token batches on sm_53. Use `-fa off`.
- The nvgpu channel watchdog (5 s) can kill multimodal requests: image encoding makes multi-second waits across CUDA
  streams. Raise it before starting the server: `echo 60000 > /sys/kernel/debug/gpu.0/ch_wdt_timeout_ms`.
- BF16 GGUFs compute through the float shim; use F16/Q8_0/K-quant files.
- Only checked on this one board.
