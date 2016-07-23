/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _IMAGE_H
#define _IMAGE_H

#include "SystemCore.h"

class ImagePixelUInt16x2;
class ImagePixelUInt8x4;
class ImagePixelUInt16x4;
class ImagePixelFloat16x4;
class ImagePixelFloat32x4;
class ImagePixelUInt16x8;

typedef enum
{
    IPF_int8,
    IPF_int16,
    IPF_int32,
    IPF_float32,
    IPF_float64,
    IPF_uint8,
    IPF_uint16,
    IPF_uint32,
    IPF_uint16x2,
    IPF_uint8x4,
    IPF_uint16x4,
    IPF_float16x4,
    IPF_float32x4,
    IPF_uint16x8,
} ImagePixelFormat;

template<ImagePixelFormat> struct ImagePixelFormatTrait;
template<> struct ImagePixelFormatTrait<IPF_int8>
{
    typedef signed char type;
};
template<> struct ImagePixelFormatTrait<IPF_uint8>
{
    typedef uchar type;
};
template<> struct ImagePixelFormatTrait<IPF_uint8x4>
{
    typedef ImagePixelUInt8x4 type;
};
template<> struct ImagePixelFormatTrait<IPF_int16>
{
    typedef short type;
};
template<> struct ImagePixelFormatTrait<IPF_uint16>
{
    typedef ushort type;
};
template<> struct ImagePixelFormatTrait<IPF_uint16x2>
{
    typedef ImagePixelUInt16x2 type;
};
template<> struct ImagePixelFormatTrait<IPF_uint16x4>
{
    typedef ImagePixelUInt16x4 type;
};
template<> struct ImagePixelFormatTrait<IPF_uint16x8>
{
    typedef ImagePixelUInt16x8 type;
};
template<> struct ImagePixelFormatTrait<IPF_int32>
{
    typedef int type;
};
template<> struct ImagePixelFormatTrait<IPF_uint32>
{
    typedef uint type;
};
template<> struct ImagePixelFormatTrait<IPF_float32>
{
    typedef float type;
};
template<> struct ImagePixelFormatTrait<IPF_float16x4>
{
    typedef ImagePixelFloat16x4 type;
};
template<> struct ImagePixelFormatTrait<IPF_float32x4>
{
    typedef ImagePixelFloat32x4 type;
};
template<> struct ImagePixelFormatTrait<IPF_float64>
{
    typedef double type;
};

// ======================================================================================================
// IMAGE CONTAINER
// ======================================================================================================

template<ImagePixelFormat T> class Image
{
public:
    typedef typename ImagePixelFormatTrait<T>::type PixelDataType;

    static ImagePixelFormat getPixelType()
    {
        return T;
    }

    Image(void)
            : mWidth(0), mHeight(0), mSize(0), mData(0), mDataSize(0), mIsDataOwner(true)
    {
    }

    Image(const Image<T> &t)
            : mWidth(0), mHeight(0), mSize(0), mData(0), mDataSize(0), mIsDataOwner(true)
    {
        (*this) = t;
    }

    Image(int x, int y)
            : mWidth(0), mHeight(0), mSize(0), mData(0), mDataSize(0), mIsDataOwner(true)
    {
        set(x, y);
    }

    Image(uint width, uint height, PixelDataType *ptr)
            : mWidth(width), mHeight(height), mSize(width * height), mData(ptr), mDataSize(width * height),
              mIsDataOwner(false)
    {
    }

    ~Image(void)
    {
        if (mData != 0 && mIsDataOwner)
        {
            System::MemoryFree(mData);
        }
    }

    Image<T> &operator=(Image<T> const &t)
    {
        set(t.mWidth, t.mHeight);

        if (t.mData != 0)
        {
            System::MemoryCopy(mData, t.mData, mSize * sizeof(PixelDataType));
        }

        return *this;
    }

    PixelDataType &operator()(int x, int y)
    {
        return mData[y * mWidth + x];
    }

    const PixelDataType &operator()(int x, int y) const
    {
        return mData[y * mWidth + x];
    }

    PixelDataType &operator[](uint index)
    {
        return mData[index];
    }

    const PixelDataType &operator[](uint index) const
    {
        return mData[index];
    }

    void operator*=(PixelDataType const &value)
    {
        PixelDataType *__restrict dptr = static_cast<PixelDataType *>(__builtin_assume_aligned(mData,
                CACHELINE_ALIGNMENT));
        for (uint i = 0; i < mSize; i++)
        {
            dptr[i] *= value;
        }
    }

    void operator+=(PixelDataType const &value)
    {
        PixelDataType *__restrict dptr = static_cast<PixelDataType *>(__builtin_assume_aligned(mData,
                CACHELINE_ALIGNMENT));
        for (uint i = 0; i < mSize; i++)
        {
            dptr[i] += value;
        }
    }

    void operator-=(PixelDataType const &value)
    {
        PixelDataType *__restrict dptr = static_cast<PixelDataType *>(__builtin_assume_aligned(mData,
                CACHELINE_ALIGNMENT));
        for (uint i = 0; i < mSize; i++)
        {
            dptr[i] -= value;
        }
    }

    void operator*=(const Image<T> &t)
    {
        PixelDataType *__restrict dptr = static_cast<PixelDataType *>(__builtin_assume_aligned(mData,
                CACHELINE_ALIGNMENT));
        PixelDataType const *__restrict sptr = static_cast<PixelDataType const *>(__builtin_assume_aligned(
                t.mData, CACHELINE_ALIGNMENT));
        for (uint i = 0; i < mSize; i++)
        {
            dptr[i] *= sptr[i];
        }
    }

    void operator+=(const Image<T> &t)
    {
        PixelDataType *__restrict dptr = static_cast<PixelDataType *>(__builtin_assume_aligned(mData,
                CACHELINE_ALIGNMENT));
        PixelDataType const *__restrict sptr = static_cast<PixelDataType const *>(__builtin_assume_aligned(
                t.mData, CACHELINE_ALIGNMENT));
        for (uint i = 0; i < mSize; i++)
        {
            dptr[i] += sptr[i];
        }
    }

    void operator-=(const Image<T> &t)
    {
        PixelDataType *__restrict dptr = static_cast<PixelDataType *>(__builtin_assume_aligned(mData,
                CACHELINE_ALIGNMENT));
        PixelDataType const *__restrict sptr = static_cast<PixelDataType const *>(__builtin_assume_aligned(
                t.mData, CACHELINE_ALIGNMENT));
        for (uint i = 0; i < mSize; i++)
        {
            dptr[i] -= sptr[i];
        }
    }

    void set(uint x, uint y)
    {
        uint newImageSize = x * y;
        if (newImageSize > mDataSize)
        {
            if (mData != 0 && mIsDataOwner)
            {
                System::MemoryFree(mData);
            }

            mIsDataOwner = true;
            mDataSize = newImageSize;
            mData = static_cast<PixelDataType *>(__builtin_assume_aligned(
                    System::Internal::MemoryAlloc(sizeof(PixelDataType) * newImageSize, CACHELINE_ALIGNMENT), CACHELINE_ALIGNMENT));
        }

        mWidth = x;
        mHeight = y;
        mSize = newImageSize;
    }

    void clear(PixelDataType const &value)
    {
        PixelDataType *__restrict dptr = static_cast<PixelDataType *>(__builtin_assume_aligned(mData,
                CACHELINE_ALIGNMENT));
        for (uint i = 0; i < mSize; i++)
        {
            dptr[i] = value;
        }
    }

    void subImage(Image<T> const &src, int srcPosX = 0, int srcPosY = 0, int dstWidth = 0, int dstHeight = 0)
    {
        if (dstWidth == 0)
        {
            dstWidth = src.mWidth;
        }

        if (dstHeight == 0)
        {
            dstHeight = src.mHeight;
        }

        set(dstWidth, dstHeight);

        PixelDataType const * srcPtr = &src(srcPosX, srcPosY);
        PixelDataType *dstPtr = mData;

        for (int y = 0; y < dstHeight; y++)
        {
            System::MemoryCopy(dstPtr, srcPtr, sizeof(PixelDataType) * dstWidth);
            srcPtr += src.mWidth;
            dstPtr += dstWidth;
        }
    }

    void blt(Image<T> const &src, int dstPosX = 0, int dstPosY = 0, int dstWidth = 0, int dstHeight = 0)
    {
        if (dstWidth == 0)
        {
            dstWidth = src.mWidth;
        }

        if (dstHeight == 0)
        {
            dstHeight = src.mHeight;
        }

        double const fdx = 65536.0 * src.mWidth / dstWidth;
        double const fdy = 65536.0 * src.mHeight / dstHeight;

        int offsetSrcX = 0;
        if (dstPosX < 0)
        {
            offsetSrcX = (int)(-dstPosX * fdx);
            dstWidth += dstPosX;
            dstPosX = 0;
        }

        if (dstPosX + dstWidth > (int)mWidth)
        {
            dstWidth = mWidth - dstPosX;
        }

        int offsetSrcY = 0;
        if (dstPosY < 0)
        {
            offsetSrcY = (int)(-dstPosY * fdy);
            dstHeight += dstPosY;
            dstPosY = 0;
        }

        if (dstPosY + dstHeight > (int)mHeight)
        {
            dstHeight = mHeight - dstPosY;
        }

        if (dstWidth <= 0 || dstHeight <= 0)
        {
            return;
        }

        int const dx = (int)fdx;
        int const dy = (int)fdy;
        PixelDataType *dstData = mData + dstPosY * mWidth + dstPosX;
        int ay = offsetSrcY;
        for (int y = 0; y < dstHeight; y++)
        {
            int ax = offsetSrcX;
            PixelDataType const *srcData = src.mData + (ay >> 16) * src.mWidth;

            for (int x = 0; x < dstWidth; x++)
            {
                dstData[x] = srcData[ax >> 16];
                ax += dx;
            }

            dstData += mWidth;
            ay += dy;
        }
    }

    void mirrorX(void)
    {
        uint hh = mHeight >> 1;
        PixelDataType *buf = new PixelDataType[mWidth];

        for (uint i = 0; i < hh; i++)
        {
            PixelDataType *l1 = mData + i * mWidth;
            PixelDataType *l2 = mData + (mHeight - 1 - i) * mWidth;

            System::MemoryCopy(buf, l1, mWidth * sizeof(PixelDataType));
            System::MemoryCopy(l1, l2, mWidth * sizeof(PixelDataType));
            System::MemoryCopy(l2, buf, mWidth * sizeof(PixelDataType));
        }

        delete[] buf;
    }

    void swapSurface(Image<T> &b)
    {
        NVR::xchg(mWidth, b.mWidth);
        NVR::xchg(mHeight, b.mHeight);
        NVR::xchg(mSize, b.mSize);
        NVR::xchg(mData, b.mData);
        NVR::xchg(mDataSize, b.mDataSize);
        NVR::xchg(mIsDataOwner, b.mIsDataOwner);
    }

    bool loadImage(const char *name);
    bool writeImage(const char *name);

    const PixelDataType *getRawPointer(void) const
    {
        return mData;
    }

    PixelDataType *getRawPointer(void)
    {
        return mData;
    }

    uint getAllocatedDataSize(void) const
    {
        return mDataSize * sizeof(PixelDataType);
    }

    uint getDataHash(void) const
    {
        if (mData == 0)
        {
            return 0;
        }

        return System::MemoryHash(mData, sizeof(PixelDataType) * mSize);
    }

    int getWidth(void) const
    {
        return mWidth;
    }

    int getHeight(void) const
    {
        return mHeight;
    }

    int getSize(void) const
    {
        return mSize;
    }

protected:
    uint mWidth, mHeight;
    uint mSize;
    PixelDataType *__restrict mData;
    uint mDataSize;
    bool mIsDataOwner;
};

#endif
