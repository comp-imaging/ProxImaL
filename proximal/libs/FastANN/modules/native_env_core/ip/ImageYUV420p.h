/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef IMAGEYUV420P_H_
#define IMAGEYUV420P_H_

#include "Image.h"
#include "ManagedPtr.h"

class ImageYUV420p
{
public:
    ImageYUV420p(int width, int height, uchar *ydata, int ystride, uchar *udata, uchar *vdata, int uvstride)
            : mWidth(width), mHeight(height), mStrideY(ystride), mStrideUV(uvstride), mDataOwner(false), mDataY(ydata),
              mDataU(udata), mDataV(vdata)
    {
    }

    ImageYUV420p(int width, int height, int rowAlignment = 1);

    ImageYUV420p(Image<IPF_uint8x4> const &image, int rowAlignSize = 1);

    ~ImageYUV420p(void);

    int getWidth() const
    {
        return mWidth;
    }

    int getHeight() const
    {
        return mHeight;
    }

    int getStrideY() const
    {
        return mStrideY;
    }

    int getStrideUV() const
    {
        return mStrideUV;
    }

    uchar *getY(void) const
    {
        return mDataY;
    }

    uchar *getU(void) const
    {
        return mDataU;
    }

    uchar *getV(void) const
    {
        return mDataV;
    }

    void copyTo(Image<IPF_uint8x4> &image) const;

private:
    // prevent copy construction and assignment
    ImageYUV420p(ImageYUV420p const &instance);
    ImageYUV420p &operator=(ImageYUV420p const &instance);

    int const mWidth, mHeight, mStrideY, mStrideUV;
    bool const mDataOwner;
    uchar *mDataY, *mDataU, *mDataV;
};

#endif /* IMAGEYUV420P_H_ */
