#include "mmv-sm53.cuh"

// MMV_SM53_RPW rows per warp share each y load (y re-reads, not weights, bound the one-row-per-warp form),
// MMV_SM53_WARPS warps per block.
#define MMV_SM53_WARPS 4
#define MMV_SM53_RPW   2

// 2^23 as a float: OR-ing an integer q < 2^23 into its mantissa gives 2^23 + q exactly.
#define MMV_SM53_MAGIC_BITS 0x4B000000u
#define MMV_SM53_MAGIC      8388608.0f

static __device__ __forceinline__ float mmv_sm53_u2f(const uint32_t q) {
    return __uint_as_float(q | MMV_SM53_MAGIC_BITS) - MMV_SM53_MAGIC;
}

// Same, minus the Q6_K offset of 32 in the one subtraction (exact: both operands are integers below 2^24).
static __device__ __forceinline__ float mmv_sm53_u2f_m32(const uint32_t q) {
    return __uint_as_float(q | MMV_SM53_MAGIC_BITS) - (MMV_SM53_MAGIC + 32.0f);
}

static __device__ __forceinline__ float mmv_sm53_warp_sum(float v) {
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset, WARP_SIZE);
    }
    return v;
}

// Bits 19..22 of a float with exponent 0 hold a nibble q as the value 1 + q/16 (exact), so
// q*y = 16*(f*y - y): one shift, one LOP3 and one FMA per weight. The -16*y part is applied once per
// sub-block from the precomputed 16-element sums of y (ys16).
#define MMV_SM53_ONE_BITS 0x3F800000u
#define MMV_SM53_NIB_MASK 0x00780000u

static __device__ __forceinline__ float mmv_sm53_nib(const uint32_t shifted) {
    return __uint_as_float((shifted & MMV_SM53_NIB_MASK) | MMV_SM53_ONE_BITS);
}

// Q4_K with native sm_53 fp16: a nibble OR-ed into bits 6..9 of half 1.0 is exactly 1 + q/16, so one shift +
// LOP3 builds two weights, HSUB2 makes them q/16 (exact) and HFMA2 multiplies both by y. y comes as halves in
// groups of 4 stored (y0, y2, y1, y3), so each half2 of y lines up with one weight pair. 16 products are summed
// in half (|q/16 * y| <= |y|), then converted to float.
#define MMV_SM53_H2_ONE  0x3C003C00u
#define MMV_SM53_H2_NIB  0x03C003C0u

static __device__ __forceinline__ half2 mmv_sm53_h2(const uint32_t shifted) {
    const uint32_t bits = (shifted & MMV_SM53_H2_NIB) | MMV_SM53_H2_ONE;
    return __hsub2(*(const half2 *) &bits, __float2half2_rn(1.0f));
}

static __device__ __forceinline__ float mmv_sm53_dot_q4_K_h(
        const block_q4_K * __restrict__ b, const uint4 (&ylo)[2], const uint4 (&yhi)[2],
        const float s_lo, const float s_hi, const int i) {
    const uint4 hdr = __ldg((const uint4 *) b);
    const uint4 q   = __ldg((const uint4 *) (b->qs + 16*i));

    const int c  = i / 2;
    const int j0 = (i % 2) * 16;

    const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
    const uint32_t yl[8] = {ylo[0].x, ylo[0].y, ylo[0].z, ylo[0].w, ylo[1].x, ylo[1].y, ylo[1].z, ylo[1].w};
    const uint32_t yh[8] = {yhi[0].x, yhi[0].y, yhi[0].z, yhi[0].w, yhi[1].x, yhi[1].y, yhi[1].z, yhi[1].w};
    half2 acc_lo = __float2half2_rn(0.0f), acc_hi = __float2half2_rn(0.0f);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t v = qw[w];
        acc_lo = __hfma2(mmv_sm53_h2(v << 6), *(const half2 *) &yl[2*w],     acc_lo);  // weights 4w+0, 4w+2
        acc_lo = __hfma2(mmv_sm53_h2(v >> 2), *(const half2 *) &yl[2*w + 1], acc_lo);  // weights 4w+1, 4w+3
        acc_hi = __hfma2(mmv_sm53_h2(v << 2), *(const half2 *) &yh[2*w],     acc_hi);
        acc_hi = __hfma2(mmv_sm53_h2(v >> 6), *(const half2 *) &yh[2*w + 1], acc_hi);
    }
    const float2 flo = __half22float2(acc_lo);
    const float2 fhi = __half22float2(acc_hi);

    const int      sh = 16*(c & 1);
    const uint32_t Y  = hdr.y >> sh, Z = hdr.z >> sh, W = hdr.w >> sh;
    int sc_lo, sc_hi, m_lo, m_hi;
    if (c < 2) {
        sc_lo = Y & 63; sc_hi = (Y >> 8) & 63;
        m_lo  = Z & 63; m_hi  = (Z >> 8) & 63;
    } else {
        sc_lo = ( W        & 0xF) | ((Y >>  2) & 0x30); sc_hi = ((W >>  8) & 0xF) | ((Y >> 10) & 0x30);
        m_lo  = ((W >>  4) & 0xF) | ((Z >>  2) & 0x30); m_hi  = ((W >> 12) & 0xF) | ((Z >> 10) & 0x30);
    }
    const float2 dm = __half22float2(*(const half2 *) &hdr.x);
    float r = 16.0f*dm.x*(sc_lo*(flo.x + flo.y) + sc_hi*(fhi.x + fhi.y));
    if (j0 == 0) {
        r -= dm.y*(m_lo*s_lo + m_hi*s_hi);
    }
    return r;
}

// Q6_K: lane l takes half h = l/16 (128 weights) and positions p = 2*(l%16), p+1 of that half.
// Each position yields 4 weights at p, p+32, p+64, p+96 with scales sc[h*8 + p/16 + {0,2,4,6}].
// Blocks are 210 bytes, so loads stay 16-bit.
static __device__ __forceinline__ float mmv_sm53_dot_q6_K(
        const block_q6_K * __restrict__ b, const int h, const int p,
        const float2 y0, const float2 y1, const float2 y2, const float2 y3) {
    const uint32_t ql0 = __ldg((const unsigned short *) (b->ql + h*64 + p));       // weights p,    p+64 (low/high nibble)
    const uint32_t ql1 = __ldg((const unsigned short *) (b->ql + h*64 + 32 + p));  // weights p+32, p+96
    const uint32_t qh  = __ldg((const unsigned short *) (b->qh + h*32 + p));       // 2 bits each for p, p+32, p+64, p+96

    // byte k of ql0/ql1/qh belongs to position p + k
    const float a0 = mmv_sm53_u2f_m32(( ql0        & 0xF) | ((qh <<  4) & 0x30));
    const float a1 = mmv_sm53_u2f_m32(((ql0 >>  8) & 0xF) | ((qh >>  4) & 0x30));
    const float b0 = mmv_sm53_u2f_m32(( ql1        & 0xF) | ((qh <<  2) & 0x30));
    const float b1 = mmv_sm53_u2f_m32(((ql1 >>  8) & 0xF) | ((qh >>  6) & 0x30));
    const float c0 = mmv_sm53_u2f_m32(((ql0 >>  4) & 0xF) | ( qh        & 0x30));
    const float c1 = mmv_sm53_u2f_m32(((ql0 >> 12) & 0xF) | ((qh >>  8) & 0x30));
    const float d0 = mmv_sm53_u2f_m32(((ql1 >>  4) & 0xF) | ((qh >>  2) & 0x30));
    const float d1 = mmv_sm53_u2f_m32(((ql1 >> 12) & 0xF) | ((qh >> 10) & 0x30));

    const signed char * sc = b->scales + h*8 + p/16;
    const unsigned short d_bits = __ldg((const unsigned short *) &b->d);
    return __half2float(*(const half *) &d_bits) *
        ( __ldg(sc + 0)*(a0*y0.x + a1*y0.y) + __ldg(sc + 2)*(b0*y1.x + b1*y1.y)
        + __ldg(sc + 4)*(c0*y2.x + c1*y2.y) + __ldg(sc + 6)*(d0*y3.x + d1*y3.y));
}

// Sums of y over consecutive 16-element groups, shared by all rows of one mat-vec. Also writes y as halves in
// groups of 4 ordered (y0, y2, y1, y3) for the fp16 Q4_K path.
static __global__ void mul_mat_vec_sm53_ysum16(const float * __restrict__ y, float * __restrict__ ys16, uint2 * __restrict__ yh, const int n16) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n16) {
        return;
    }
    const float4 * y4 = (const float4 *) (y + 16*i);
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float4 v = y4[k];
        s += (v.x + v.y) + (v.z + v.w);
        const half2 a = __floats2half2_rn(v.x, v.z);
        const half2 b = __floats2half2_rn(v.y, v.w);
        yh[4*i + k] = make_uint2(*(const uint32_t *) &a, *(const uint32_t *) &b);
    }
    ys16[i] = s;
}

template <ggml_type type>
static __global__ void __launch_bounds__(WARP_SIZE*MMV_SM53_WARPS)
mul_mat_vec_sm53(const void * __restrict__ vx, const float * __restrict__ y, const float * __restrict__ ys16, const uint2 * __restrict__ yh,
                 float * __restrict__ dst, const int ncols, const int nrows, const int64_t stride_row) {
    const int row0 = (blockIdx.x*MMV_SM53_WARPS + threadIdx.y)*MMV_SM53_RPW;
    if (row0 >= nrows) {
        return;
    }
    const int lane = threadIdx.x;
    const int nb   = ncols / QK_K;
    const char * x0 = (const char *) vx + row0*stride_row;

    float acc[MMV_SM53_RPW] = {0.0f};
    if constexpr (type == GGML_TYPE_Q4_K) {
        const int i  = lane % 8;
        const int c  = i / 2;
        const int j0 = (i % 2) * 16;
        for (int ib = lane / 8; ib < nb; ib += WARP_SIZE / 8) {
            const uint4 * yg = (const uint4 *) (yh + (ib*QK_K + c*64 + j0)/4);  // 2 groups of 4 per uint4
            const uint4 ylo[2] = {__ldg(yg),     __ldg(yg + 1)};
            const uint4 yhi[2] = {__ldg(yg + 4), __ldg(yg + 5)};               // +32 weights = +8 groups
            float s_lo = 0.0f, s_hi = 0.0f;
            if (j0 == 0) {
                const float * s16 = ys16 + ib*(QK_K/16) + 4*c;
                s_lo = __ldg(s16)     + __ldg(s16 + 1);
                s_hi = __ldg(s16 + 2) + __ldg(s16 + 3);
            }
#pragma unroll
            for (int r = 0; r < MMV_SM53_RPW; ++r) {
                if (row0 + r < nrows) {
                    acc[r] += mmv_sm53_dot_q4_K_h((const block_q4_K *) (x0 + r*stride_row) + ib, ylo, yhi, s_lo, s_hi, i);
                }
            }
        }
    } else {
        const int h = lane / 16;
        const int p = 2*(lane % 16);
        for (int ib = 0; ib < nb; ++ib) {
            const float * yh = y + ib*QK_K + h*128 + p;
            const float2 y0 = __ldg((const float2 *) (yh));
            const float2 y1 = __ldg((const float2 *) (yh + 32));
            const float2 y2 = __ldg((const float2 *) (yh + 64));
            const float2 y3 = __ldg((const float2 *) (yh + 96));
#pragma unroll
            for (int r = 0; r < MMV_SM53_RPW; ++r) {
                if (row0 + r < nrows) {
                    acc[r] += mmv_sm53_dot_q6_K((const block_q6_K *) (x0 + r*stride_row) + ib, h, p, y0, y1, y2, y3);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < MMV_SM53_RPW; ++r) {
        const float sum = mmv_sm53_warp_sum(acc[r]);
        if (lane == 0 && row0 + r < nrows) {
            dst[row0 + r] = sum;
        }
    }
}

bool ggml_cuda_should_use_mmv_sm53(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    // Measured only on the Jetson Nano; other dp4a-less GPUs (desktop Maxwell, P100) keep MMVQ.
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc != GGML_CUDA_CC_TEGRA_X1 || ggml_cuda_highest_compiled_arch(cc) != GGML_CUDA_CC_TEGRA_X1) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q6_K) {
        return false;
    }
    // plain 2D single-token case: no broadcasting, contiguous rows, aligned y for vector loads
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src0->ne[0] % QK_K == 0 && src0->nb[0] == ggml_type_size(src0->type) &&
           ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
           (uintptr_t) src1->data % 16 == 0 && (uintptr_t) src0->data % 16 == 0 && src0->nb[1] % 16 == 0;
}

void ggml_cuda_mul_mat_vec_sm53(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int ncols = src0->ne[0];
    const int nrows = src0->ne[1];

    constexpr int rows_per_block = MMV_SM53_WARPS*MMV_SM53_RPW;
    const dim3 block_dims(WARP_SIZE, MMV_SM53_WARPS, 1);
    const dim3 block_nums((nrows + rows_per_block - 1) / rows_per_block, 1, 1);
    cudaStream_t stream = ctx.stream();

    const float * y = (const float *) src1->data;
    float       * d = (float       *) dst->data;

    const int n16 = ncols / 16;
    ggml_cuda_pool_alloc<float> ys16(ctx.pool(), n16);
    ggml_cuda_pool_alloc<uint2> yh(ctx.pool(), ncols/4);
    mul_mat_vec_sm53_ysum16<<<(n16 + 127) / 128, 128, 0, stream>>>(y, ys16.get(), yh.get(), n16);

    switch (src0->type) {
        case GGML_TYPE_Q4_K:
            mul_mat_vec_sm53<GGML_TYPE_Q4_K><<<block_nums, block_dims, 0, stream>>>(src0->data, y, ys16.get(), yh.get(), d, ncols, nrows, src0->nb[1]);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_sm53<GGML_TYPE_Q6_K><<<block_nums, block_dims, 0, stream>>>(src0->data, y, ys16.get(), yh.get(), d, ncols, nrows, src0->nb[1]);
            break;
        default:
            GGML_ABORT("unsupported type for mmv_sm53");
    }
    CUDA_CHECK(cudaGetLastError());
}
