/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef FASTMATH_H_
#define FASTMATH_H_

// ARM NEON code here is experimental -> not necessarily faster!!!
#ifdef __ARM_NEON__
//#define FM_OPTIMIZE_FOR_ARM_NEON
#endif

#define FM_OPTIMIZE_FOR_X86

#ifdef ARCH_X86
#include <emmintrin.h>
#define _USE_MATH_DEFINES
#include <math.h>
#ifdef _MSC_VER
#include <stdlib.h>
#endif
#endif

#ifdef ARCH_ARM
#define _USE_MATH_DEFINES
#include <math.h>
#ifdef FM_OPTIMIZE_FOR_ARM_NEON
#include <arm_neon.h>
#endif
#endif
#include "Base.h"

#define FP16_MULT(x,fs)        ((x)*((int)((fs)*65536)))
#define FP16_ROUND(x)          (((x)+0x8000)>>16)
#define FP16_ROUND_TO_FPN(x,n) (((x)+(1<<(15-(n))))>>(16-(n)))
#define FLOAT_TO_FPN(x,n)      ((int)((x)*((1<<(n))-1)))

#define CLAMP_AT_ZERO(x)       ((x)&~((int)(x)>>31))
#define CLAMP_AT_BASE2(x,n)    min_i32(x,(1<<(n))-1)
#define ROUNDED_HALF(x)        (((x)>>1)+((x)&0x01))
#define IS_ODD(x)              (((x)&0x01)!=0)
#define IS_EVEN(x)             (((x)&0x01)==0)

// ARM assembly for ARMv6+
// x86 assembly for SSE2

// these guys should remain constant (we use them to generate LUTs in FastMath.cpp)
#define RECP_MANTISSA_BITS 16
#define RECP_LUT_BITS      9

extern ushort const RecpLUT[1 << RECP_LUT_BITS];

// valid fractional bit count: 0-15
template<int fracBits> FORCE_INLINE uint recp_ufp(uint value)
{
    if (value == 0)
    {
        return 0;
    }
#ifdef ARCH_X86
#ifdef FM_OPTIMIZE_FOR_X86
    // idiv on x86 nowadays is fast enough
    return (1 << (fracBits << 1)) / value;
#else
    int shift = __builtin_clz( value ) + 1;
    int dindex = ( value << shift ) >> ( 32 - RECP_LUT_BITS );
    return ( (uint) RecpLUT[dindex] << ( 32 - RECP_MANTISSA_BITS ) ) >> ( ( ( ( 32 - fracBits ) << 1 ) - 1 ) - shift );
#endif
#endif

#ifdef ARCH_ARM
#ifdef FM_OPTIMIZE_FOR_ARM_NEON
    float32x2_t recp = vrecpe_f32(vcvt_n_f32_u32(vdup_n_u32(value), fracBits));
    return vget_lane_u32(vcvt_n_u32_f32(recp, fracBits), 0);
#else
    asm volatile(
            "\nclz r0, %[tmp] \n"
            "add r0, r0, #1 \n"
            "lsl %[tmp], %[tmp], r0 \n"
            "lsr %[tmp], %[tmp], %[lutNorm] \n"
            "lsl %[tmp], %[tmp], #1 \n"
            "ldrh %[tmp], [%[RecpLUT], %[tmp]] \n"
            "rsb r0, r0, %[fracNorm] \n"
            "lsl %[tmp], %[tmp], %[recpNorm] \n"
            "lsr %[tmp], %[tmp], r0 \n"
            : [tmp]"+r"(value) : [RecpLUT]"r"(RecpLUT), [lutNorm]"I"(32 - RECP_LUT_BITS),
            [recpNorm]"I"(32 - RECP_MANTISSA_BITS), [fracNorm]"I"(((32 - fracBits) << 1) - 1) : "r0");
    return value;
#endif
#endif
}

FORCE_INLINE int div_i32_u32(int a, uint b)
{
    if (b == 0)
    {
        return 0;
    }
#ifdef ARCH_X86
#ifdef FM_OPTIMIZE_FOR_X86
    // idiv on x86 nowadays is fast enough
    return a / (int)b;
#else
    int shift = __builtin_clz( b ) + 1;
    int dindex = ( b << shift ) >> ( 32 - RECP_LUT_BITS );
    return (int) ( ( (int64) ( a << 2 )
                    * ( ( (uint) RecpLUT[dindex] << ( 32 - RECP_MANTISSA_BITS ) ) >> ( 33 - shift ) ) )
            >> 32 );
#endif
#endif

#ifdef ARCH_ARM
#ifdef FM_OPTIMIZE_FOR_ARM_NEON
    float32x2_t recp = vrecpe_f32(vcvt_f32_u32(vdup_n_u32(b)));
    return vget_lane_s32(vcvt_s32_f32(vmul_f32(vcvt_f32_s32(vdup_n_s32(a)), recp)), 0);

//  XXX: remove, this turned out to be much slower than floating-point reciprocal
//    asm volatile(
//            "\nclz r0, %[tmp] \n"
//            "lsl %[tmp], %[tmp], r0 \n"
//            "vdup.u32 d0, %[tmp] \n"
//            "vdup.s32 d1, %[dividend] \n"
//            "vrecpe.u32 d0, d0 \n"
//            "vshr.u32 d0, d0, #1 \n"
//            "vqdmulh.s32 d0, d1, d0[0] \n"
//            "vmov.u32 %[tmp], d0[1] \n"
//            "rsb r0, r0, #30 \n"
//            "asr %[tmp], %[tmp], r0 \n"
//            "add %[tmp], %[tmp], #1 \n"
//            "asr %[tmp], %[tmp], #1 \n"
//            : [tmp]"+r"(b) : [dividend]"r"(a) : "r0", "d0", "d1");
//    return b;
#else
    // XXX: limitations: no support for negative divisors, max 29 bits signed dividend, result
    // is computed via reciprocal with RECP_MANTISSA_BITS precision
    asm volatile(
            "\nclz r0, %[tmp] \n"
            "add r0, r0, #1 \n"
            "lsl %[tmp], %[tmp], r0 \n"
            "lsr %[tmp], %[tmp], %[lutNorm] \n"
            "lsl %[tmp], %[tmp], #1 \n"
            "ldrh %[tmp], [%[RecpLUT], %[tmp]] \n"
            "rsb r0, r0, #33 \n"
            "lsl %[tmp], %[tmp], %[recpNorm] \n"
            "lsr %[tmp], %[tmp], r0 \n"
            "lsl r0, %[dividend], #2 \n"
            "smmul %[tmp], r0, %[tmp] \n"
            : [tmp]"+r"(b) : [dividend]"r"(a), [RecpLUT]"r"(RecpLUT),
            [lutNorm]"I"(32 - RECP_LUT_BITS), [recpNorm]"I"(32 - RECP_MANTISSA_BITS) : "r0");

    return b;
#endif
#endif
}

FORCE_INLINE int mulhi_i32(int a, int b)
{
#ifdef ARCH_X86
    return (int)((int64)a * b >> 32);
#endif
#ifdef ARCH_ARM
    asm volatile("smmul %[a], %[a], %[b]": [a]"+r"(a) : [b]"r"(b));
    return a;
#endif
}

template<int fracBits> FORCE_INLINE ushort pack16_usat8_ifp(int a, int b)
{
#ifdef ARCH_X86
    a >>= fracBits;
    b >>= fracBits;
    a = a < 0 ? 0 : (a > 0xff ? 0xff : a);
    b = b < 0 ? 0 : (b > 0xff ? 0xff : b);
    return (a << 8) | b;
#endif
#ifdef ARCH_ARM
    // XXX: limitations: only least significant 16-bits are taken into account after shift!
    if (fracBits != 0)
    {
        asm volatile(
                "\nasr %[b], %[b], %[shift1] \n"
                "pkhbt %[a], %[b], %[a], lsl %[shift2] \n"
                "usat16 %[b], #8, %[a] \n"
                "orr %[a], %[b], %[b], lsr#8 \n"
                : [a]"+r"(a), [b]"+r"(b) : [shift1]"I"(fracBits), [shift2]"I"(16 - fracBits));
    }
    else
    {
        asm volatile(
                "\npkhbt %[a], %[b], %[a], lsl#16 \n"
                "usat16 %[a], #8, %[a] \n"
                "orr %[a], %[a], %[a], lsr#8 \n"
                : [a]"+r"(a) : [b]"r"(b));
    }
    return a;
#endif
}

template<int fracBits> FORCE_INLINE uchar usat8_ifp(int a)
{
    a >>= fracBits;
    return a < 0 ? 0 : (a > 0xff ? 0xff : a);
}

FORCE_INLINE int abs_i32(int x)
{
    // mask will be 0 for positive, -1 (0xffffffff) for negative numbers
    int mask = x >> 31;
    return (x ^ mask) - mask;
}

FORCE_INLINE int sign_i32(int x)
{
    return 1 - ((x >> 31) & 0x2);
}

FORCE_INLINE int copysign_i32(int value, int signSource)
{
    int mask = signSource >> 31;
    return (value ^ mask) - mask;
}

#if defined(ARCH_X86)
FORCE_INLINE __m128i abd_uint8x16(__m128i a, __m128i b)
{
    return _mm_sub_epi8(_mm_max_epu8(a, b), _mm_min_epu8(a, b));
}

FORCE_INLINE __m128i abd_int16x8(__m128i a, __m128i b)
{
    return _mm_sub_epi16(_mm_max_epi16(a, b), _mm_min_epi16(a, b));
}

#endif

FORCE_INLINE uint byte_swap_u32(uint x)
{
#ifdef __GNUC__
    return __builtin_bswap32(x);
#elif defined(_MSC_VER)
    return _byteswap_ulong( x );
#endif
}

FORCE_INLINE uint bit_reverse_u32(uint x)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return ((x >> 16) | (x << 16));
}

#define INT_SQRT_ITER(n) \
    temp = root + (1 << (n)); \
    if (x >= temp << (n))   \
    {   x -= temp << (n);   \
        root |= 2 << (n); \
    }

FORCE_INLINE int sqrt_u16(uint x)
{
    uint root = 0, temp;
    INT_SQRT_ITER( 7);
    INT_SQRT_ITER( 6);
    INT_SQRT_ITER( 5);
    INT_SQRT_ITER( 4);
    INT_SQRT_ITER( 3);
    INT_SQRT_ITER( 2);
    INT_SQRT_ITER( 1);
    INT_SQRT_ITER( 0);
    return root >> 1;
}

FORCE_INLINE int max_i32(int a, int b)
{
    int t = a - b;
    return a - (t & (t >> 31));
}

FORCE_INLINE int min_i32(int a, int b)
{
    int t = a - b;
    return b + (t & (t >> 31));
}

FORCE_INLINE float recp_f32(float x)
{
#ifdef ARCH_X86
    return _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x)));
#else
    return 1.0f / x;
#endif
}

FORCE_INLINE float sqrt_f32(float x)
{
#ifdef ARCH_X86
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
#else
    return sqrtf(x);
#endif
}

FORCE_INLINE float rsqrt_f32(float x)
{
#ifdef ARCH_X86
    return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
#else
    return 1.0f / sqrtf(x);
#endif
}

FORCE_INLINE float max_f32(float a, float b)
{
#ifdef ARCH_X86
    return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(a), _mm_set_ss(b)));
#else
    return a > b ? a : b;
#endif
}

FORCE_INLINE float min_f32(float a, float b)
{
#ifdef ARCH_X86
    return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(a), _mm_set_ss(b)));
#else
    return a < b ? a : b;
#endif
}

FORCE_INLINE float log_f32(float y)
{
    return logf(y);
}

FORCE_INLINE float log2_f32(float y)
{
    return logf(y) * M_LOG2E;
}

FORCE_INLINE float exp_f32(float y)
{
#ifdef ARCH_X86
    const float powPoly[] =
    { 0.3371894217f, 0.6576362914f, 1.0017247597f };
    __m128 rval, fvec;
    __m128i ivec;

    // rval = y * log2(e)
    rval = _mm_mul_ss(_mm_set_ss(y), _mm_set_ss(1.4426950409f));

    // 2^rval => e^y
    // FIXME: SSE unit goes to denorm mode and slows down insanely for exponents close to the limit
    rval = _mm_min_ss(rval, _mm_set_ss(128.0f));
    rval = _mm_max_ss(rval, _mm_set_ss(-126.0f));

    ivec = _mm_add_epi32(_mm_cvttps_epi32(rval), _mm_srai_epi32(_mm_castps_si128(rval), 31));
    rval = _mm_sub_ss(rval, _mm_cvtepi32_ps(ivec));
    fvec = _mm_add_ss(_mm_mul_ss(_mm_set_ss(powPoly[0]), rval), _mm_set_ss(powPoly[1]));
    fvec = _mm_add_ss(_mm_mul_ss(fvec, rval), _mm_set_ss(powPoly[2]));
    rval = _mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(fvec), _mm_slli_epi32(ivec, 23)));

    return _mm_cvtss_f32(rval);
#else
    return expf(y);
#endif
}

FORCE_INLINE float ldexp_f32(float scale, int factor)
{
    // fast ldexp implementation (no support for denorms, negative numbers, inf/nans)
    union
    {
        int i;
        float f;
    } fi;

    fi.f = scale;
    int e = (fi.i >> 23) + factor;
    if (e <= 0)
    {
        fi.i = 0;
    }
    else if (e >= 255)
    {
        fi.i = 0x7f000000;
    }
    else
    {
        fi.i = (fi.i & 0x7fffff) | (e << 23);
    }

    return fi.f;
}

FORCE_INLINE float abs_f32(float const value)
{
#ifdef ARCH_X86
    return _mm_cvtss_f32(_mm_and_ps(_mm_set_ss(value), _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff))));
#else
    return value < 0 ? -value : value;
#endif
}

void init_recp_lut(ushort *recpLut);

#endif /* FASTMATH_H_ */
