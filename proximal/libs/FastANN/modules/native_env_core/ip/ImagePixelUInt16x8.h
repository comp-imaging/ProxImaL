/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef IMAGEPIXELUINT16X8_H_
#define IMAGEPIXELUINT16X8_H_

#include "Base.h"
#include "ImagePixelFloat32x4.h"

#define IPFUI16X8_SUPPORTED

#ifdef ARCH_X86

class ImagePixelUInt16x8
{
    friend class ImagePixelUInt8x16;
    friend class ImagePixelUInt16x4;
public:
    ImagePixelUInt16x8( void )
    {
    }

    ImagePixelUInt16x8( const __m128i &t )
    : mData( t )
    {
    }

    ImagePixelUInt16x8( ImagePixelUInt16x8 const &t )
    : mData( t.mData )
    {
    }

#ifdef USE_SSE4
    ImagePixelUInt16x8( ImagePixelFloat32x4 const &hi, ImagePixelFloat32x4 const &lo )
    : mData( _mm_packus_epi32( _mm_cvtps_epi32( lo.mData ), _mm_cvtps_epi32( hi.mData ) ) )
    {
    }
#else
    ImagePixelUInt16x8( ImagePixelFloat32x4 const &hi, ImagePixelFloat32x4 const &lo )
    : mData(
            _mm_add_epi16(
                    _mm_packs_epi32( _mm_sub_epi32( _mm_cvtps_epi32( lo.mData ), _mm_set1_epi32( 32768 ) ),
                            _mm_sub_epi32( _mm_cvtps_epi32( hi.mData ), _mm_set1_epi32( 32768 ) ) ),
                    _mm_set1_epi16( 32768 ) ) )
    {
    }
#endif

    ImagePixelUInt16x8 &operator=( ImagePixelUInt16x8 const &t )
    {
        mData = t.mData;
        return *this;
    }

    ImagePixelFloat32x4 low() const
    {
        return _mm_cvtepi32_ps( _mm_unpacklo_epi16( mData, _mm_setzero_si128() ) );
    }

    ImagePixelFloat32x4 high() const
    {
        return _mm_cvtepi32_ps( _mm_unpackhi_epi16( mData, _mm_setzero_si128() ) );
    }

    ushort operator[]( uint i ) const
    {
        union
        {
            __m128i vec;
            ushort scalar[8];
        }temp;

        temp.vec = mData;
        return temp.scalar[i];
    }
private:
    __m128i mData;
};

#endif

#ifdef ARCH_ARM

#include <arm_neon.h>

class ImagePixelUInt16x8
{
    friend class ImagePixelUInt8x16;
    friend class ImagePixelUInt16x4;
public:
    ImagePixelUInt16x8(void)
    {
    }

    ImagePixelUInt16x8(const uint16x8_t &t)
            : mData(t)
    {
    }

    ImagePixelUInt16x8(ImagePixelUInt16x8 const &t)
            : mData(t.mData)
    {
    }

    ImagePixelUInt16x8(ImagePixelFloat32x4 const &hi, ImagePixelFloat32x4 const &lo)
            : mData(vcombine_u16(vqmovn_u32(vcvtq_u32_f32(lo.mData)), vqmovn_u32(vcvtq_u32_f32(hi.mData))))
    {
    }

    ImagePixelUInt16x8 &operator=(ImagePixelUInt16x8 const &t)
    {
        mData = t.mData;
        return *this;
    }

    ImagePixelFloat32x4 low() const
    {
        return vcvtq_f32_u32(vmovl_u16(vget_low_u16(mData)));
    }

    ImagePixelFloat32x4 high() const
    {
        return vcvtq_f32_u32(vmovl_u16(vget_high_u16(mData)));
    }

    ushort operator[](uint i) const
    {
        return vgetq_lane_u16(mData, i);
    }
private:
    uint16x8_t mData;
};

#endif

#include "ImagePixelCommon.h"

#endif /* IMAGEPIXELUINT16X8_H_ */
