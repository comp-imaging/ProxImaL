/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef IMAGEPIXELFLOAT16X4_H_
#define IMAGEPIXELFLOAT16X4_H_

#include "Base.h"
#include "ImagePixelFloat32x4.h"

#define IPFF16X4_SUPPORTED

#ifdef ARCH_ARM

class ImagePixelFloat16x4
{
    friend class ImagePixelFloat32x4;
public:
    ImagePixelFloat16x4(void)
    {
    }

    ImagePixelFloat16x4(int16x4_t const &t)
            : mData(t)
    {
    }

    ImagePixelFloat16x4(ImagePixelFloat32x4 const &t)
    {
        asm volatile ("vcvt.f16.f32 %P0, %q1 \n" : "=w"(mData) : "w"(t.mData));
    }

    ImagePixelFloat16x4 &operator=(ImagePixelFloat16x4 const &t)
    {
        mData = t.mData;
        return *this;
    }

    short operator[](uint i) const
    {
        return vget_lane_s16(mData, i);
    }

    FORCE_INLINE friend ImagePixelFloat16x4 vload(ImagePixelFloat16x4 const *src)
    {
        return vld1_s16(reinterpret_cast<int16_t const *>(src));
    }

    FORCE_INLINE friend void vstore(ImagePixelFloat16x4 *dst, ImagePixelFloat16x4 const &t)
    {
        vst1_s16(reinterpret_cast<int16_t *>(dst), t.mData);
    }

private:
    int16x4_t mData;
};

#endif

#include "ImagePixelCommon.h"

#endif /* IMAGEPIXELFLOAT16X4_H_ */
