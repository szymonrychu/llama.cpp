// Minimal bfloat16 for CUDA 10.2 (Jetson Nano), which has no cuda_bf16.h (added in CUDA 11).
// Storage is the real bf16 bit pattern; arithmetic goes through float.
#pragma once
#include <cuda_runtime.h>
#include <cstring>
#include <cuda_fp16.h>

struct __nv_bfloat16 {
    unsigned short x;
    __nv_bfloat16() = default;
    __host__ __device__ __nv_bfloat16(float f) {
        unsigned int u; memcpy(&u, &f, 4);
        if ((u & 0x7fffffffu) > 0x7f800000u) { x = (unsigned short)((u >> 16) | 0x40); }
        else { u += 0x7fffu + ((u >> 16) & 1u); x = (unsigned short)(u >> 16); }
    }
    __host__ __device__ __nv_bfloat16(double d) : __nv_bfloat16((float) d) {}
    __host__ __device__ __nv_bfloat16(int i) : __nv_bfloat16((float) i) {}
    __host__ __device__ __nv_bfloat16(half h) : __nv_bfloat16(__half2float(h)) {}
    __host__ __device__ operator float() const {
        unsigned int u = ((unsigned int) x) << 16; float f; memcpy(&f, &u, 4); return f;
    }
};
struct __nv_bfloat162 {
    __nv_bfloat16 x, y;
    __nv_bfloat162() = default;
    __host__ __device__ __nv_bfloat162(__nv_bfloat16 a, __nv_bfloat16 b) : x(a), y(b) {}
};
typedef __nv_bfloat16 nv_bfloat16;
typedef __nv_bfloat162 nv_bfloat162;

static __host__ __device__ inline float __bfloat162float(const __nv_bfloat16 a) { return (float) a; }
static __host__ __device__ inline __nv_bfloat16 __float2bfloat16(const float a) { return __nv_bfloat16(a); }
static __host__ __device__ inline __nv_bfloat16 __float2bfloat16_rn(const float a) { return __nv_bfloat16(a); }
static __host__ __device__ inline float2 __bfloat1622float2(const __nv_bfloat162 a) { return make_float2((float) a.x, (float) a.y); }
static __host__ __device__ inline __nv_bfloat162 __floats2bfloat162_rn(const float a, const float b) { return __nv_bfloat162(__nv_bfloat16(a), __nv_bfloat16(b)); }
static __host__ __device__ inline __nv_bfloat162 __float22bfloat162_rn(const float2 a) { return __floats2bfloat162_rn(a.x, a.y); }
static __host__ __device__ inline __nv_bfloat16 __low2bfloat16(const __nv_bfloat162 a) { return a.x; }
static __host__ __device__ inline __nv_bfloat16 __high2bfloat16(const __nv_bfloat162 a) { return a.y; }

#ifndef CUDA_R_16BF
#define CUDA_R_16BF ((cudaDataType_t) 14)
#endif
