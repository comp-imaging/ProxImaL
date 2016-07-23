/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef IMAGEPIXELUINT32X4_H_
#define IMAGEPIXELUINT32X4_H_

#include "Base.h"
#include "ImagePixelFloat32x4.h"

#define IPFUI32X4_SUPPORTED

#ifdef ARCH_X86

class ImagePixelUInt32x4
{
    friend class ImagePixelFloat32x4;
public:
    ImagePixelUInt32x4( void )
    {
    }

    ImagePixelUInt32x4( ImagePixelFloat32x4 const &t )
    : mData( _mm_cvtps_epi32( t.mData ) )
    {
    }

    ImagePixelUInt32x4( uint t )
    : mData( _mm_set1_epi32( t ) )
    {
    }

    ImagePixelUInt32x4( __m128i const &t )
    : mData( t )
    {
    }

    ImagePixelUInt32x4 &operator=( ImagePixelUInt32x4 const &t )
    {
        mData = t.mData;
        return *this;
    }

    ImagePixelUInt32x4 operator &( ImagePixelUInt32x4 const &t ) const
    {
        return _mm_and_si128( mData, t.mData );
    }

    ushort operator[]( uint i ) const
    {
        union
        {
            uint i[4];
            __m128i v;
        }value;

        value.v = mData;
        return value.i[i];
    }

private:
    __m128i mData;
};

#endif

#ifdef ARCH_ARM

class ImagePixelUInt32x4
{
    friend class ImagePixelFloat32x4;
public:
    ImagePixelUInt32x4(void)
    {
    }

    ImagePixelUInt32x4(uint t)
            : mData(vdupq_n_u32(t))
    {
    }

    ImagePixelUInt32x4(uint32x4_t const &t)
            : mData(t)
    {
    }

    ImagePixelUInt32x4(ImagePixelFloat32x4 const &t)
            : mData(vcvtq_u32_f32(t.mData))
    {
    }

    ImagePixelUInt32x4 &operator=(ImagePixelUInt32x4 const &t)
    {
        mData = t.mData;
        return *this;
    }

    ImagePixelUInt32x4 operator &(ImagePixelUInt32x4 const &t) const
    {
        return vandq_u32(mData, t.mData);
    }

    uint operator[](uint i) const
    {
        return vgetq_lane_u32(mData, i);
    }

    FORCE_INLINE friend ImagePixelUInt32x4 vload(ImagePixelUInt32x4 const *src)
    {
        return vld1q_u32(reinterpret_cast<uint32_t const *>(src));
    }

    FORCE_INLINE friend void vstore(ImagePixelUInt16x4 *dst, ImagePixelUInt32x4 const &t)
    {
        vst1q_u32(reinterpret_cast<uint32_t *>(dst), t.mData);
    }

private:
    uint32x4_t mData;
};

#endif

#include "ImagePixelCommon.h"

#endif /* IMAGEPIXELUINT16X4_H_ */
