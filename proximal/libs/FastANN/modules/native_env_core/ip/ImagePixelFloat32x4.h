/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef IMAGEPIXELFLOAT32X4_H_
#define IMAGEPIXELFLOAT32X4_H_

#include "Base.h"

#define IPFF32X4_SUPPORTED

#define IP_MATH_EPSILON 1e-20f

class ImagePixelUInt8x4;
class ImagePixelUInt16x4;
class ImagePixelUInt16x8;
class ImagePixelUInt32x4;
class ImagePixelFloat16x4;

#ifdef ARCH_X86

#include <smmintrin.h>

class ImagePixelFloat32x4
{
    friend class ImagePixelUInt8x4;
    friend class ImagePixelUInt8x16;
    friend class ImagePixelUInt16x4;
    friend class ImagePixelUInt16x8;
    friend class ImagePixelUInt32x4;
public:
    ImagePixelFloat32x4(void)
    {
    }

    ImagePixelFloat32x4(const __m128 &t)
    : mData(t)
    {
    }

    ImagePixelFloat32x4(int t)
    : mData(_mm_set1_ps(t))
    {
    }

    ImagePixelFloat32x4(float t)
    : mData(_mm_set1_ps(t))
    {
    }

    ImagePixelFloat32x4(ImagePixelFloat32x4 const &t)
    : mData(t.mData)
    {
    }

    ImagePixelFloat32x4(float r, float g, float b, float a = 0.0f)
    : mData(_mm_set_ps(a, r, g, b))
    {
    }

    explicit ImagePixelFloat32x4(ImagePixelUInt8x4 const &t);

    explicit ImagePixelFloat32x4(ImagePixelUInt16x4 const &t);

    explicit ImagePixelFloat32x4(ImagePixelUInt32x4 const &t);

    ImagePixelFloat32x4 &operator=(ImagePixelFloat32x4 const &t)
    {
        mData = t.mData;
        return *this;
    }

    void operator/=(ImagePixelFloat32x4 const &value)
    {
        mData = _mm_mul_ps(mData, _mm_rcp_ps(value.mData));
    }

    void operator*=(ImagePixelFloat32x4 const &value)
    {
        mData = _mm_mul_ps(mData, value.mData);
    }

    void operator+=(ImagePixelFloat32x4 const &value)
    {
        mData = _mm_add_ps(mData, value.mData);
    }

    void operator-=(ImagePixelFloat32x4 const &value)
    {
        mData = _mm_sub_ps(mData, value.mData);
    }

    ImagePixelFloat32x4 operator+(ImagePixelFloat32x4 const &value) const
    {
        return _mm_add_ps(mData, value.mData);
    }

    ImagePixelFloat32x4 operator-(ImagePixelFloat32x4 const &value) const
    {
        return _mm_sub_ps(mData, value.mData);
    }

    ImagePixelFloat32x4 operator*(ImagePixelFloat32x4 const &value) const
    {
        return _mm_mul_ps(mData, value.mData);
    }

    ImagePixelFloat32x4 operator/(ImagePixelFloat32x4 const &value) const
    {
        return _mm_mul_ps(mData, _mm_rcp_ps(value.mData));
    }

    ImagePixelFloat32x4 operator>>(int shift) const
    {
        int const value = (127 - shift) << 23;
        return _mm_mul_ps(mData, _mm_castsi128_ps(_mm_set1_epi32(value)));
    }

    ImagePixelFloat32x4 operator<<(int shift) const
    {
        int const value = (127 + shift) << 23;
        return _mm_mul_ps(mData, _mm_castsi128_ps(_mm_set1_epi32(value)));
    }

    ImagePixelFloat32x4 broadcastA(void) const
    {
        return _mm_shuffle_ps( mData, mData, _MM_SHUFFLE(3, 3, 3, 3) );
    }

    ImagePixelFloat32x4 broadcastR(void) const
    {
        return _mm_shuffle_ps( mData, mData, _MM_SHUFFLE(2, 2, 2, 2) );
    }

    ImagePixelFloat32x4 broadcastG(void) const
    {
        return _mm_shuffle_ps( mData, mData, _MM_SHUFFLE(1, 1, 1, 1) );
    }

    ImagePixelFloat32x4 broadcastB(void) const
    {
        return _mm_shuffle_ps( mData, mData, _MM_SHUFFLE(0, 0, 0, 0) );
    }

    ImagePixelFloat32x4 cloneWithAlpha(float alpha) const
    {
        return _mm_insert_ps( mData, _mm_set_ss( alpha ), _MM_MK_INSERTPS_NDX(0, 3, 0) );
    }

    float sum(void) const
    {
        __m128 temp = _mm_hadd_ps(mData, mData);
        return _mm_cvtss_f32(_mm_hadd_ps(temp, temp));
    }

    float squaredSum(void) const
    {
        return _mm_cvtss_f32(_mm_dp_ps( mData, mData, 0xff ));
    }

    float squaredSum3(void) const
    {
        return _mm_cvtss_f32(_mm_dp_ps( mData, mData, 0x77 ));
    }

    float weightedSum(ImagePixelFloat32x4 const &weights) const
    {
        __m128 temp = _mm_hadd_ps(weights.mData, weights.mData);
        // XXX: full division has variable run-time (up to 2x slower than constant-time reciprocal)
//        return _mm_cvtss_f32( _mm_div_ss( _mm_dp_ps( m_vdata, weights.m_vdata, 0xff ), _mm_hadd_ps( temp, temp ) ) );
        temp = _mm_rcp_ss(_mm_hadd_ps(temp, temp));
        return _mm_cvtss_f32(_mm_mul_ss(_mm_dp_ps( mData, weights.mData, 0xff ), temp));
    }

    float dot4(ImagePixelFloat32x4 const &value) const
    {
        return _mm_cvtss_f32(_mm_dp_ps( mData, value.mData, 0xff ));
//        __m128 temp = _mm_mul_ps(mData, value.mData); // SSE2 version below
//        temp = _mm_hadd_ps(temp, temp);
//        return _mm_cvtss_f32(_mm_hadd_ps(temp, temp));
    }

    float dot3(ImagePixelFloat32x4 const &value) const
    {
        return _mm_cvtss_f32(_mm_dp_ps( mData, value.mData, 0x77 ));
    }

    ImagePixelFloat32x4 cross3(ImagePixelFloat32x4 const &value) const
    {
        return _mm_sub_ps(
                _mm_mul_ps(_mm_shuffle_ps( mData, mData, _MM_SHUFFLE(3, 0, 2, 1) ),
                        _mm_shuffle_ps( value.mData, value.mData, _MM_SHUFFLE(3, 1, 0, 2) )),
                _mm_mul_ps(_mm_shuffle_ps( mData, mData, _MM_SHUFFLE(3, 1, 0, 2) ),
                        _mm_shuffle_ps( value.mData, value.mData, _MM_SHUFFLE(3, 0, 2, 1) )));
    }

    // XXX: normalization will fail if input is very small (epsilon will bias the result)
    ImagePixelFloat32x4 normalize3(void) const
    {
        __m128 temp = _mm_rsqrt_ss(_mm_add_ss(_mm_dp_ps( mData, mData, 0x77 ), _mm_set_ss(IP_MATH_EPSILON)));
        return _mm_mul_ps(mData, _mm_shuffle_ps( temp, temp, _MM_SHUFFLE(3, 0, 0, 0) ));
    }

    ImagePixelFloat32x4 normalize4(void) const
    {
        __m128 temp = _mm_rsqrt_ss(_mm_add_ss(_mm_dp_ps( mData, mData, 0xff ), _mm_set_ss(IP_MATH_EPSILON)));
        return _mm_mul_ps(mData, _mm_shuffle_ps( temp, temp, 0 ));
    }

    float length3(void) const
    {
        return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps( mData, mData, 0x77 )));
    }

    float invLength3(void) const
    {
        return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_add_ss(_mm_dp_ps( mData, mData, 0x77 ), _mm_set_ss(IP_MATH_EPSILON))));
    }

    ImagePixelFloat32x4 recp(void) const
    {
        return _mm_rcp_ps(mData);
    }

    float min(void) const
    {
        __m128 temp = _mm_min_ps(mData, _mm_shuffle_ps( mData, mData, _MM_SHUFFLE(1, 0, 3, 2) ));
        return _mm_cvtss_f32(_mm_min_ss(temp, _mm_shuffle_ps( temp, temp, _MM_SHUFFLE(2, 3, 0, 1) )));
    }

    float max(void) const
    {
        __m128 temp = _mm_max_ps(mData, _mm_shuffle_ps( mData, mData, _MM_SHUFFLE(1, 0, 3, 2) ));
        return _mm_cvtss_f32(_mm_max_ss(temp, _mm_shuffle_ps( temp, temp, _MM_SHUFFLE(2, 3, 0, 1) )));
    }

    friend float min(ImagePixelFloat32x4 const &value)
    {
        return value.min();
    }

    friend float max(ImagePixelFloat32x4 const &value)
    {
        return value.max();
    }

    friend ImagePixelFloat32x4 max(ImagePixelFloat32x4 const &a, ImagePixelFloat32x4 const &b)
    {
        return _mm_max_ps(a.mData, b.mData);
    }

    friend ImagePixelFloat32x4 min(ImagePixelFloat32x4 const &a, ImagePixelFloat32x4 const &b)
    {
        return _mm_min_ps(a.mData, b.mData);
    }

    friend float dot3(ImagePixelFloat32x4 const &a, ImagePixelFloat32x4 const &b)
    {
        return a.dot3(b);
    }

    friend float dot4(ImagePixelFloat32x4 const &a, ImagePixelFloat32x4 const &b)
    {
        return a.dot4(b);
    }

    friend ImagePixelFloat32x4 cross3(ImagePixelFloat32x4 const &a, ImagePixelFloat32x4 const &b)
    {
        return a.cross3(b);
    }

    friend ImagePixelFloat32x4 abs(ImagePixelFloat32x4 const &value)
    {
        return _mm_and_ps(value.mData, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)));
    }

    friend ImagePixelFloat32x4 sign(ImagePixelFloat32x4 const &value)
    {
        return _mm_or_ps(_mm_set1_ps(1.0f), _mm_and_ps(value.mData, _mm_castsi128_ps(_mm_set1_epi32(0x80000000))));
    }

    friend ImagePixelFloat32x4 sqrt(ImagePixelFloat32x4 const &value)
    {
        return _mm_sqrt_ps(value.mData);
    }

    friend ImagePixelFloat32x4 rsqrt(ImagePixelFloat32x4 const &value)
    {
        return _mm_rsqrt_ps(value.mData);
    }

    // Compares cmp1 and cmp2. Returns 0 if cmp1 is bigger, 1 if cmp2 is bigger.
    friend ImagePixelFloat32x4 cmpgt(ImagePixelFloat32x4 const &cmp1, ImagePixelFloat32x4 const &cmp2)
    {
        return _mm_cmpgt_ps(cmp2.mData, cmp1.mData);
    }

    friend ImagePixelFloat32x4 select(ImagePixelFloat32x4 const &cmp, ImagePixelFloat32x4 const &v1,
            ImagePixelFloat32x4 const &v2)
    {
        return _mm_or_ps(_mm_andnot_ps(cmp.mData, v1.mData), _mm_and_ps(cmp.mData, v2.mData));
    }

    friend ImagePixelFloat32x4 log2(ImagePixelFloat32x4 const &x)
    {
        const float logPoly[] =
        {   0.204204359088f, -1.252546896816f, 3.331021458582f, -2.282678920620f};

        const __m128i mask = _mm_set1_epi32(0xff << 23);
        const __m128i bias = _mm_set1_epi32(0x7f << 23);
        __m128 rval, fvec;

        rval = _mm_castsi128_ps(_mm_or_si128(_mm_andnot_si128(mask, _mm_castps_si128(x.mData)), bias));
        fvec = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(logPoly[0]), rval), _mm_set1_ps(logPoly[1]));
        fvec = _mm_add_ps(_mm_mul_ps(fvec, rval), _mm_set1_ps(logPoly[2]));
        fvec = _mm_add_ps(_mm_mul_ps(fvec, rval), _mm_set1_ps(logPoly[3]));
        fvec = _mm_add_ps(
                fvec,
                _mm_cvtepi32_ps(
                        _mm_srai_epi32(_mm_sub_epi32(_mm_and_si128(_mm_castps_si128(x.mData), mask), bias), 23)));

        return fvec;
    }

    friend ImagePixelFloat32x4 pow2_fast(ImagePixelFloat32x4 const &y)
    {
        const float powPoly[] =
        {   0.3371894217f, 0.6576362914f, 1.0017247597f};
        __m128 rval, fvec;
        __m128i ivec;

        ivec = _mm_add_epi32(_mm_cvttps_epi32(y.mData), _mm_srai_epi32(_mm_castps_si128(y.mData), 31));
        rval = _mm_sub_ps(y.mData, _mm_cvtepi32_ps(ivec));
        fvec = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(powPoly[0]), rval), _mm_set1_ps(powPoly[1]));
        fvec = _mm_add_ps(_mm_mul_ps(fvec, rval), _mm_set1_ps(powPoly[2]));
        rval = _mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(fvec), _mm_slli_epi32(ivec, 23)));

        return rval;
    }

    friend ImagePixelFloat32x4 pow2(ImagePixelFloat32x4 const &y)
    {
        return pow2_fast(
                ImagePixelFloat32x4(_mm_min_ps(_mm_max_ps(y.mData, _mm_set1_ps(-126.0f)), _mm_set1_ps(128.0f))));
    }

    FORCE_INLINE
    friend void vstore(ImagePixelFloat32x4 *dst, ImagePixelFloat32x4 const &t)
    {
        *dst = t;
    }

    FORCE_INLINE
    friend ImagePixelFloat32x4 vload(ImagePixelFloat32x4 const *src)
    {
        return src[0];
    }

    float r(void) const
    {
        return _mm_cvtss_f32(_mm_shuffle_ps( mData, mData, _MM_SHUFFLE(2, 2, 2, 2) ));
    }

    float g(void) const
    {
        return _mm_cvtss_f32(_mm_shuffle_ps( mData, mData, _MM_SHUFFLE(1, 1, 1, 1) ));
    }

    float b(void) const
    {
        return _mm_cvtss_f32(mData);
    }

    float a(void) const
    {
        return _mm_cvtss_f32(_mm_shuffle_ps( mData, mData, _MM_SHUFFLE(3, 3, 3, 3) ));
    }

    float operator[](uint i) const
    {
        union
        {
            __m128 vec;
            float scalar[4];
        }temp;

        temp.vec = mData;
        return temp.scalar[i];
    }

    const __m128 &vdata(void) const
    {
        return mData;
    }

private:
    __m128 mData;
};

#endif

#ifdef ARCH_ARM

#include <arm_neon.h>

class ImagePixelFloat32x4_Channel0;
class ImagePixelFloat32x4_Channel1;
class ImagePixelFloat32x4_Channel2;
class ImagePixelFloat32x4_Channel3;

class ImagePixelFloat32x4
{
    friend class ImagePixelUInt8x4;
    friend class ImagePixelUInt8x16;
    friend class ImagePixelUInt16x4;
    friend class ImagePixelUInt16x8;
    friend class ImagePixelUInt32x4;
    friend class ImagePixelFloat16x4;

public:
    ImagePixelFloat32x4(void)
    {
    }

    ImagePixelFloat32x4(const float32x4_t &t)
            : mData(t)
    {
    }

    ImagePixelFloat32x4(int t)
            : mData(vdupq_n_f32(static_cast<float>(t)))
    {
    }

    ImagePixelFloat32x4(float32_t t)
            : mData(vdupq_n_f32(t))
    {
    }

    ImagePixelFloat32x4(ImagePixelFloat32x4 const &t)
            : mData(t.mData)
    {
    }

    ImagePixelFloat32x4(float32_t r, float32_t g, float32_t b, float32_t a = 0.0f)
    {
        union
        {
            float32x4_t v;
            float32_t f[4];
        } val;

        val.f[0] = b;
        val.f[1] = g;
        val.f[2] = r;
        val.f[3] = a;

        mData = val.v;
    }

    explicit ImagePixelFloat32x4(ImagePixelUInt8x4 const &t);

    explicit ImagePixelFloat32x4(ImagePixelUInt16x4 const &t);

    explicit ImagePixelFloat32x4(ImagePixelUInt32x4 const &t);

    explicit ImagePixelFloat32x4(ImagePixelFloat16x4 const &t);

    ImagePixelFloat32x4 &operator=(ImagePixelFloat32x4 const &t)
    {
        mData = t.mData;
        return *this;
    }

    void operator/=(ImagePixelFloat32x4 const &value)
    {
        mData = vmulq_f32(mData, vrecpeq_f32(value.mData));
    }

    void operator*=(ImagePixelFloat32x4 const &value)
    {
        mData = vmulq_f32(mData, value.mData);
    }

    void operator+=(ImagePixelFloat32x4 const &value)
    {
        mData = vaddq_f32(mData, value.mData);
    }

    void operator-=(ImagePixelFloat32x4 const &value)
    {
        mData = vsubq_f32(mData, value.mData);
    }

    ImagePixelFloat32x4 operator+(ImagePixelFloat32x4 const &value) const
    {
        return vaddq_f32(mData, value.mData);
    }

    ImagePixelFloat32x4 operator-(ImagePixelFloat32x4 const &value) const
    {
        return vsubq_f32(mData, value.mData);
    }

    ImagePixelFloat32x4 operator*(ImagePixelFloat32x4 const &value) const
    {
        return vmulq_f32(mData, value.mData);
    }

    ImagePixelFloat32x4 operator*(ImagePixelFloat32x4_Channel0 const &value) const;

    ImagePixelFloat32x4 operator*(ImagePixelFloat32x4_Channel1 const &value) const;

    ImagePixelFloat32x4 operator*(ImagePixelFloat32x4_Channel2 const &value) const;

    ImagePixelFloat32x4 operator*(ImagePixelFloat32x4_Channel3 const &value) const;

    ImagePixelFloat32x4 operator/(ImagePixelFloat32x4 const &value) const
    {
        return vmulq_f32(mData, vrecpeq_f32(value.mData));
    }

    ImagePixelFloat32x4 operator>>(int shift) const
    {
        int const value = (127 - shift) << 23;
        return vmulq_f32(mData, vreinterpretq_f32_s32(vdupq_n_s32(value)));
    }

    ImagePixelFloat32x4 operator<<(int shift) const
    {
        int const value = (127 + shift) << 23;
        return vmulq_f32(mData, vreinterpretq_f32_s32(vdupq_n_s32(value)));
    }

    float32_t sum(void) const
    {
        float32x2_t temp = vpadd_f32(vget_low_f32(mData), vget_high_f32(mData));
        return vget_lane_f32(vpadd_f32(temp, temp), 0);
    }

    float32_t squaredSum(void) const
    {
        float32x4_t prod = vmulq_f32(mData, mData);
        float32x2_t temp = vpadd_f32(vget_low_f32(prod), vget_high_f32(prod));
        return vget_lane_f32(vpadd_f32(temp, temp), 0);
    }
//
//    float weightedSum( ImagePixelFloat32x4 const &weights ) const
//    {
//        __m128 temp = _mm_hadd_ps( weights.mData, weights.mData );
//        temp = _mm_rcp_ss( _mm_hadd_ps( temp, temp ) );
//        return _mm_cvtss_f32( _mm_mul_ss( _mm_dp_ps( mData, weights.mData, 0xff ), temp ) );
//    }

    float32_t dot4(ImagePixelFloat32x4 const &value) const
    {
        float32x4_t prod = vmulq_f32(mData, value.mData);
        float32x2_t temp = vpadd_f32(vget_low_f32(prod), vget_high_f32(prod));
        return vget_lane_f32(vpadd_f32(temp, temp), 0);
    }
//
//    float dot3( ImagePixelFloat32x4 const &value ) const
//    {
//        return _mm_cvtss_f32( _mm_dp_ps( mData, value.mData, 0x77 ) );
//    }
//
    float32_t min(void) const
    {
        float32x2_t temp = vpmin_f32(vget_low_f32(mData), vget_high_f32(mData));
        return vget_lane_f32(vpmin_f32(temp, temp), 0);
    }

    float32_t max(void) const
    {
        float32x2_t temp = vpmax_f32(vget_low_f32(mData), vget_high_f32(mData));
        return vget_lane_f32(vpmax_f32(temp, temp), 0);
    }

    ImagePixelFloat32x4_Channel0 b(void) const;

    ImagePixelFloat32x4_Channel1 g(void) const;

    ImagePixelFloat32x4_Channel2 r(void) const;

    ImagePixelFloat32x4_Channel3 a(void) const;

    float32_t operator[](uint i) const
    {
        return vgetq_lane_f32(mData, i);
    }

    friend ImagePixelFloat32x4 sqrt(ImagePixelFloat32x4 const &value)
    {
        return vrecpeq_f32(vrsqrteq_f32(value.mData));
    }

    friend ImagePixelFloat32x4 log2(ImagePixelFloat32x4 const &x)
    {
        static const float32_t weight[] =
        { 0.204204359088f, -1.252546896816f, 3.331021458582f, -2.282678920620f };
        static const int32_t biasAndMask[] =
        { 0x7f << 23, 0xff << 23 };

        float32x4_t const wv = vld1q_f32(weight);
        int32x2_t const bmv = vld1_s32(biasAndMask);

        int32x4_t bias = vdupq_lane_s32(bmv, 0);
        int32x4_t ivec = vreinterpretq_s32_f32(x.mData);

        float32x4_t rval, fvec;
        rval = vreinterpretq_f32_s32(vbslq_s32(vreinterpretq_u32_s32(vdupq_lane_s32(bmv, 1)), bias, ivec));
        fvec = vmlaq_lane_f32(vdupq_lane_f32(vget_low_f32(wv), 1), rval, vget_low_f32(wv), 0);
        fvec = vmlaq_f32(vdupq_lane_f32(vget_high_f32(wv), 0), rval, fvec);
        fvec = vmlaq_f32(vdupq_lane_f32(vget_high_f32(wv), 1), rval, fvec);
        fvec = vaddq_f32(fvec, vcvtq_f32_s32(vshrq_n_s32(vsubq_s32(ivec, bias), 23)));

        return fvec;
    }

    friend ImagePixelFloat32x4 pow2_fast(ImagePixelFloat32x4 const &y)
    {
        static const float32_t weight[] =
        { 0.3371894217f, 0.6576362914f, 1.0017247597f, 0.0f };

        float32x4_t const wv = vld1q_f32(weight);
        float32x4_t rval, fvec;
        int32x4_t ivec;

        ivec = vaddq_s32(vcvtq_s32_f32(y.mData), vshrq_n_s32(vreinterpretq_s32_f32(y.mData), 31));
        rval = vsubq_f32(y.mData, vcvtq_f32_s32(ivec));
        fvec = vmlaq_lane_f32(vdupq_lane_f32(vget_low_f32(wv), 1), rval, vget_low_f32(wv), 0);
        fvec = vmlaq_f32(vdupq_lane_f32(vget_high_f32(wv), 0), rval, fvec);
        rval = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(fvec), vshlq_n_s32(ivec, 23)));

        return rval;
    }

    friend ImagePixelFloat32x4 pow2(ImagePixelFloat32x4 const &y)
    {
        static const float32_t clamp[] =
        { 128.0f, -126.0f };
        float32x2_t const clampv = vld1_f32(clamp);

        return pow2_fast(
                ImagePixelFloat32x4(
                        vmaxq_f32(vminq_f32(y.mData, vdupq_lane_f32(clampv, 0)), vdupq_lane_f32(clampv, 1))));
    }

    friend ImagePixelFloat32x4 max(ImagePixelFloat32x4 const &a, ImagePixelFloat32x4 const &b)
    {
        return vmaxq_f32(a.mData, b.mData);
    }

    friend float32_t max(ImagePixelFloat32x4 const &a)
    {
        return a.max();
    }

    friend ImagePixelFloat32x4 min(ImagePixelFloat32x4 const &a, ImagePixelFloat32x4 const &b)
    {
        return vminq_f32(a.mData, b.mData);
    }

    friend float32_t min(ImagePixelFloat32x4 const &a)
    {
        return a.min();
    }

    friend ImagePixelFloat32x4 abs(ImagePixelFloat32x4 const &value)
    {
        return vabsq_f32(value.mData);
    }

    friend ImagePixelFloat32x4 sign(ImagePixelFloat32x4 const &value)
    {
        return vbslq_f32(vdupq_n_u32(0x80000000), value.mData, vdupq_n_f32(1.0f));
    }

    // Compares cmp1 and cmp2. Returns 0 if cmp1 is bigger, 1 if cmp2 is bigger.
    friend ImagePixelFloat32x4 compare(ImagePixelFloat32x4 const &cmp1, ImagePixelFloat32x4 const &cmp2)
    {
        return vreinterpretq_f32_u32(vcgtq_f32(cmp2.mData, cmp1.mData));
    }

    friend ImagePixelFloat32x4 selection(ImagePixelFloat32x4 const &cmp, ImagePixelFloat32x4 const &v1,
                                         ImagePixelFloat32x4 const &v2)
    {
        return vbslq_f32(vreinterpretq_u32_f32(cmp.mData), v2.mData, v1.mData);
    }

    const float32x4_t &vdata(void) const
    {
        return mData;
    }

private:
    float32x4_t mData;
};

// using C++ overloading to implement faster vector-scalar multiplications
class ImagePixelFloat32x4_Channel0
{
    friend class ImagePixelFloat32x4;
public:
    ImagePixelFloat32x4_Channel0(float32x4_t const &t)
            : mData(t)
    {
    }

    operator float(void) const
    {
        return vgetq_lane_f32(mData, 0);
    }
private:
    float32x4_t mData;
};

class ImagePixelFloat32x4_Channel1
{
    friend class ImagePixelFloat32x4;
public:
    ImagePixelFloat32x4_Channel1(float32x4_t const &t)
            : mData(t)
    {
    }

    operator float(void) const
    {
        return vgetq_lane_f32(mData, 1);
    }
private:
    float32x4_t mData;
};

class ImagePixelFloat32x4_Channel2
{
    friend class ImagePixelFloat32x4;
public:
    ImagePixelFloat32x4_Channel2(float32x4_t const &t)
            : mData(t)
    {
    }

    operator float(void) const
    {
        return vgetq_lane_f32(mData, 2);
    }
private:
    float32x4_t mData;
};

class ImagePixelFloat32x4_Channel3
{
    friend class ImagePixelFloat32x4;
public:
    ImagePixelFloat32x4_Channel3(float32x4_t const &t)
            : mData(t)
    {
    }

    operator float(void) const
    {
        return vgetq_lane_f32(mData, 3);
    }
private:
    float32x4_t mData;
};

FORCE_INLINE ImagePixelFloat32x4_Channel0 ImagePixelFloat32x4::b(void) const
{
    return mData;
}

FORCE_INLINE ImagePixelFloat32x4_Channel1 ImagePixelFloat32x4::g(void) const
{
    return mData;
}

FORCE_INLINE ImagePixelFloat32x4_Channel2 ImagePixelFloat32x4::r(void) const
{
    return mData;
}

FORCE_INLINE ImagePixelFloat32x4_Channel3 ImagePixelFloat32x4::a(void) const
{
    return mData;
}

FORCE_INLINE ImagePixelFloat32x4 ImagePixelFloat32x4::operator*(ImagePixelFloat32x4_Channel0 const &value) const
{
    return vmulq_lane_f32(mData, vget_low_f32(value.mData), 0);
}

FORCE_INLINE ImagePixelFloat32x4 ImagePixelFloat32x4::operator*(ImagePixelFloat32x4_Channel1 const &value) const
{
    return vmulq_lane_f32(mData, vget_low_f32(value.mData), 1);
}

FORCE_INLINE ImagePixelFloat32x4 ImagePixelFloat32x4::operator*(ImagePixelFloat32x4_Channel2 const &value) const
{
    return vmulq_lane_f32(mData, vget_high_f32(value.mData), 0);
}

FORCE_INLINE ImagePixelFloat32x4 ImagePixelFloat32x4::operator*(ImagePixelFloat32x4_Channel3 const &value) const
{
    return vmulq_lane_f32(mData, vget_high_f32(value.mData), 1);
}

#endif

// global operator overloads
template<typename T> FORCE_INLINE ImagePixelFloat32x4 operator+(T const &a, ImagePixelFloat32x4 const &b)
{
    return ImagePixelFloat32x4(a) + b;
}

template<typename T> FORCE_INLINE ImagePixelFloat32x4 operator-(T const &a, ImagePixelFloat32x4 const &b)
{
    return ImagePixelFloat32x4(a) - b;
}

template<typename T> FORCE_INLINE ImagePixelFloat32x4 operator*(T const &a, ImagePixelFloat32x4 const &b)
{
    return ImagePixelFloat32x4(a) * b;
}

template<typename T> FORCE_INLINE ImagePixelFloat32x4 operator/(T const &a, ImagePixelFloat32x4 const &b)
{
    return ImagePixelFloat32x4(a) / b;
}

#include "ImagePixelCommon.h"

#endif /* IMAGEPIXELARGB128F_H_ */
