/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef IMAGEPIXELUINT8X16_H_
#define IMAGEPIXELUINT8X16_H_

#include "Base.h"
#include "ImagePixelUInt16x8.h"

#define IPFUI8X16_SUPPORTED

#ifdef ARCH_X86

class ImagePixelUInt8x16
{
public:
    ImagePixelUInt8x16( void )
    {
    }

    ImagePixelUInt8x16( const __m128i &t )
    : mData( t )
    {
    }

    ImagePixelUInt8x16( ImagePixelUInt8x16 const &t )
    : mData( t.mData )
    {
    }

    ImagePixelUInt8x16( ImagePixelUInt16x8 const &hi, ImagePixelUInt16x8 const &lo )
    : mData( _mm_packus_epi16( lo.mData, hi.mData ) )
    {
    }

    ImagePixelUInt8x16( ImagePixelFloat32x4 const &r, ImagePixelFloat32x4 const &g, ImagePixelFloat32x4 const &b )
    {
        __m128i pc = _mm_packus_epi16( _mm_packs_epi32( _mm_cvtps_epi32( r.mData ), _mm_cvtps_epi32( g.mData ) ),
                _mm_packs_epi32( _mm_cvtps_epi32( b.mData ), _mm_setzero_si128() ) );
        __m128i t0 = _mm_unpacklo_epi8( pc, _mm_srli_si128( pc, 8 ) );
        __m128i t1 = _mm_unpacklo_epi8( _mm_srli_si128( pc, 4 ), _mm_setzero_si128() );
        mData = _mm_unpacklo_epi8( t0, t1 );
    }

    ImagePixelUInt8x16 &operator=( ImagePixelUInt8x16 const &t )
    {
        mData = t.mData;
        return *this;
    }

    ImagePixelUInt16x8 low() const
    {
        return _mm_unpacklo_epi8( mData, _mm_setzero_si128() );
    }

    ImagePixelUInt16x8 high() const
    {
        return _mm_unpackhi_epi8( mData, _mm_setzero_si128() );
    }

    uchar operator[]( uint i ) const
    {
        union
        {
            __m128i vec;
            uchar scalar[16];
        }temp;

        temp.vec = mData;
        return temp.scalar[i];
    }

    const __m128i &vdata( void ) const
    {
        return mData;
    }

private:
    __m128i mData;
};

#endif

#ifdef ARCH_ARM

#include <arm_neon.h>

class ImagePixelUInt8x16
{
public:
    ImagePixelUInt8x16(void)
    {
    }

    ImagePixelUInt8x16(const uint8x16_t &t)
            : mData(t)
    {
    }

    ImagePixelUInt8x16(ImagePixelUInt8x16 const &t)
            : mData(t.mData)
    {
    }

    ImagePixelUInt8x16(ImagePixelUInt16x8 const &hi, ImagePixelUInt16x8 const &lo)
            : mData(vcombine_u8(vqmovn_u16(lo.mData), vqmovn_u16(hi.mData)))
    {
    }

    ImagePixelUInt8x16(ImagePixelFloat32x4 const &r, ImagePixelFloat32x4 const &g, ImagePixelFloat32x4 const &b)
    {
        uint8x8_t p0 = vqmovn_u16(vcombine_u16(vqmovn_u32(vcvtq_u32_f32(r.mData)), vqmovn_u32(vcvtq_u32_f32(g.mData))));
        uint8x8_t p1 = vqmovn_u16(vcombine_u16(vqmovn_u32(vcvtq_u32_f32(b.mData)), vdup_n_u16(0)));
        uint8x8x2_t t = vzip_u8(p0, p1);
        t = vzip_u8(t.val[0], t.val[1]);
        mData = vcombine_u8(t.val[0], t.val[1]);
    }

    ImagePixelUInt8x16 &operator=(ImagePixelUInt8x16 const &t)
    {
        mData = t.mData;
        return *this;
    }

    ImagePixelUInt16x8 low() const
    {
        return vmovl_u8(vget_low_u8(mData));
    }

    ImagePixelUInt16x8 high() const
    {
        return vmovl_u8(vget_high_u8(mData));
    }

    uchar operator[](uint i) const
    {
        return vgetq_lane_u8(mData, i);
    }

    const uint8x16_t &vdata(void) const
    {
        return mData;
    }

    operator uint8x16_t(void) const {
    	return mData;
    }

private:
    uint8x16_t mData;
};

#endif

#include "ImagePixelCommon.h"

#endif /* IMAGEPIXELUINT8X16_H_ */
