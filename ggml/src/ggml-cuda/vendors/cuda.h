#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#else
#include "cuda10-bf16.h" // CUDA 10.2 (Jetson Nano) has no cuda_bf16.h
#endif // CUDART_VERSION >= 11000

#ifdef GGML_USE_NCCL
#include <nccl.h>
#endif // GGML_USE_NCCL

#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#define FP8_AVAILABLE
#endif // CUDART_VERSION >= 11080

#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif // CUDART_VERSION >= 12080

#if CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020

#if CUDART_VERSION < 11000
// CUDA 10.2 (Jetson Nano, JetPack 4): API defaults added in CUDA 11.
static inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event) {
    return cudaStreamWaitEvent(stream, event, 0);
}
// cuBLAS 10.2 on Maxwell (no tensor cores) fails some fp16 GemmEx calls with the TENSOR_OP algo.
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP CUBLAS_GEMM_DEFAULT
#endif // CUDART_VERSION < 11000
