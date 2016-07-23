/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef IMAGEPIXELUINT16X4_H_
#define IMAGEPIXELUINT16X4_H_

#include "Base.h"
#include "ImagePixelFloat32x4.h"

#define IPFUI16X4_SUPPORTED

#ifdef ARCH_X86

class ImagePixelUInt16x4
{
    friend class ImagePixelFloat32x4;
public:
    ImagePixelUInt16x4( void )
    {
    }

#ifdef USE_SSE4
    ImagePixelUInt16x4( ImagePixelFloat32x4 const &t )
    {
        _mm_storel_epi64(reinterpret_cast<__m128i *>( mData ), _mm_packus_epi32( _mm_cvtps_epi32( t.mData ), _mm_setzero_si128() ) );
    }
#else
    ImagePixelUInt16x4( ImagePixelFloat32x4 const &t )
    {
        _mm_storel_epi64(
                reinterpret_cast<__m128i *>( mData ),
                _mm_add_epi16(
                        _mm_packs_epi32( _mm_sub_epi32( _mm_cvtps_epi32( t.mData ), _mm_set1_epi32( 32768 ) ),
                                _mm_setzero_si128() ),
                        _mm_set1_epi16( 32768 ) ) );
    }
#endif

    ImagePixelUInt16x4 &operator=( ImagePixelUInt16x4 const &t )
    {
        _mm_storel_epi64( reinterpret_cast<__m128i *>( mData ),
                _mm_loadl_epi64( reinterpret_cast<const __m128i *>( t.mData ) ) );
        return *this;
    }

    ushort operator[]( uint i ) const
    {
        return mData[i];
    }

private:
    ushort mData[4];
};

#endif

#ifdef ARCH_ARM

class ImagePixelUInt16x4
{
    friend class ImagePixelFloat32x4;
public:
    ImagePixelUInt16x4(void)
    {
    }

    ImagePixelUInt16x4(uint16x4_t const &t)
            : mData(t)
    {
    }

    ImagePixelUInt16x4(ImagePixelFloat32x4 const &t)
            : mData(vqmovn_u32(vcvtq_u32_f32(t.mData)))
    {
    }

    ImagePixelUInt16x4 &operator=(ImagePixelUInt16x4 const &t)
    {
        mData = t.mData;
        return *this;
    }

    ushort operator[](uint i) const
    {
        return vget_lane_u16(mData, i);
    }

    FORCE_INLINE friend ImagePixelUInt16x4 vload(ImagePixelUInt16x4 const *src)
    {
        return vld1_u16(reinterpret_cast<uint16_t const *>(src));
    }

    FORCE_INLINE friend void vstore(ImagePixelUInt16x4 *dst, ImagePixelUInt16x4 const &t)
    {
        vst1_u16(reinterpret_cast<uint16_t *>(dst), t.mData);
    }

private:
    uint16x4_t mData;
};

#endif

#include "ImagePixelCommon.h"

#endif /* IMAGEPIXELUINT16X4_H_ */
