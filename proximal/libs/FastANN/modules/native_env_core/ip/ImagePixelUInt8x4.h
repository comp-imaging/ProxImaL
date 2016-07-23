/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef IMAGEPIXELUINT8X4_H_
#define IMAGEPIXELUINT8X4_H_

#include "Base.h"

#define IPFUI8X4_SUPPORTED

class ImagePixelFloat32x4;

class ImagePixelUInt8x4
{
    friend class ImagePixelFloat32x4;
public:
    ImagePixelUInt8x4(void)
    {
    }

    ImagePixelUInt8x4(int t)
            : mData(t)
    {
    }

    ImagePixelUInt8x4(ImagePixelUInt8x4 const &t)
            : mData(t.mData)
    {
    }

    ImagePixelUInt8x4(ImagePixelFloat32x4 const &t);

    ImagePixelUInt8x4(int r, int g, int b, int a = 0)
            : mData((a << 24) | (r << 16) | (g << 8) | b)
    {
    }

    ImagePixelUInt8x4 &operator=(ImagePixelUInt8x4 const &t)
    {
        mData = t.mData;
        return *this;
    }

    uchar operator[](uint i) const
    {
        return mData >> (i << 3);
    }

    FORCE_INLINE friend void vstore(ImagePixelUInt8x4 *dst, ImagePixelUInt8x4 const &t)
    {
        *dst = t;
    }

    FORCE_INLINE friend ImagePixelUInt8x4 vload(ImagePixelUInt8x4 const *src)
    {
        return src[0];
    }

    FORCE_INLINE uint vdata(void)
    {
        return mData;
    }

private:
    uint mData;
};

#include "ImagePixelCommon.h"

#endif /* IMAGEPIXELUINT8X4_H_ */
