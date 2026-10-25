#include "common.cuh"

// Single-token mat-vec for K-quants on the Tegra X1 (Jetson Nano, sm_53), which has no __dp4a.
// MMVQ emulates dp4a there with byte extracts and slow integer multiplies;
// this path dequantizes nibbles with float bit tricks and keeps y in float.

bool ggml_cuda_should_use_mmv_sm53(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_vec_sm53(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
