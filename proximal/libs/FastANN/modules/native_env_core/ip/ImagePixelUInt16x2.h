/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef IMAGEPIXELUINT16X2_H_
#define IMAGEPIXELUINT16X2_H_

#define IPUI16X2_SUPPORTED

#include "Base.h"

// Packed 2x16bits pixel format type.
// XXX: all operators except subtraction assume no carry/overflow will ever happen

class ImagePixelUInt16x2
{
public:
    ImagePixelUInt16x2(void)
    {
    }

    ImagePixelUInt16x2(uint const &t)
            : mData(t)
    {
    }

    ImagePixelUInt16x2(ImagePixelUInt16x2 const &t)
            : mData(t.mData)
    {
    }

    operator ushort(void) const
    {
        return mData;
    }

    ImagePixelUInt16x2 &operator=(ImagePixelUInt16x2 const &t)
    {
        mData = t.mData;
        return *this;
    }

    ImagePixelUInt16x2 operator+(ImagePixelUInt16x2 const &value) const
    {
        return mData + value.mData;
    }

    ImagePixelUInt16x2 operator-(ImagePixelUInt16x2 const &value) const
    {
#ifdef _WIN32
        return ImagePixelUInt16x2(
                ( ( ( mData >> 16 ) - ( value.mData >> 16 ) ) << 16 ) | ( ( mData - value.mData ) & 0xffff ) );
#else
        uint rval;
        asm volatile("usub16 %[out], %[a], %[b]" : [out]"=r"(rval): [a]"r"(mData), [b]"r"(value.mData));
        return rval;
#endif
    }

    ImagePixelUInt16x2 operator>>(int const &shift) const
    {
        return (mData >> shift) & ~(((1 << shift) - 1) << (16 - shift));
    }

    ImagePixelUInt16x2 operator<<(int const &shift) const
    {
        return mData << shift;
    }

    ushort operator[](int const &i) const
    {
        return mData >> (i << 4);
    }

    template<int fracBits> friend FORCE_INLINE uint pack16_usat8_ifp(ImagePixelUInt16x2 const &a,
                                                                     ImagePixelUInt16x2 const &b)
    {
#ifdef _WIN32
        int l, r;
        l = static_cast<int>( a.mData ) >> ( 16 + fracBits );
        r = static_cast<short>( a.mData );
        r >>= fracBits;
        l = l < 0 ? 0 : ( l > 0xff ? 0xff : l );
        r = r < 0 ? 0 : ( r > 0xff ? 0xff : r );
        uint tmp = ( l << 24 ) | ( r << 8 );
        l = static_cast<int>( b.mData ) >> ( 16 + fracBits );
        r = static_cast<short>( b.mData );
        r >>= fracBits;
        l = l < 0 ? 0 : ( l > 0xff ? 0xff : l );
        r = r < 0 ? 0 : ( r > 0xff ? 0xff : r );
        return tmp | ( l << 16 ) | r;
#else
        ImagePixelUInt16x2 tmpa, tmpb;
        if (fracBits != 0)
        {
            tmpa = a >> fracBits;
            tmpb = b >> fracBits;
        }
        else
        {
            tmpa = a;
            tmpb = b;
        }
        asm volatile(
                "\nusat16 %[tmpa], #8, %[tmpa] \n"
                "usat16 %[tmpb], #8, %[tmpb] \n"
                "orr %[tmpa], %[tmpb], %[tmpa], lsl#8 \n"
                : [tmpa]"+r"(tmpa.mData), [tmpb]"+r"(tmpb.mData));

        return tmpa.mData;
#endif
    }

    template<int fracBits> friend FORCE_INLINE uint usat8_ifp(ImagePixelUInt16x2 const &a)
    {
#ifdef _WIN32
        int l, r;
        l = static_cast<int>( a.mData ) >> ( 16 + fracBits );
        r = static_cast<short>( a.mData );
        r >>= fracBits;
        l = l < 0 ? 0 : ( l > 0xff ? 0xff : l );
        r = r < 0 ? 0 : ( r > 0xff ? 0xff : r );
        return ( l << 16 ) | r;
#else
        ImagePixelUInt16x2 tmpa;
        if (fracBits != 0)
        {
            tmpa = a >> fracBits;
        }
        else
        {
            tmpa = a;
        }
        asm volatile("usat16 %[tmpa], #8, %[tmpa]": [tmpa]"+r"(tmpa.mData));
        return tmpa.mData;
#endif
    }

protected:
    uint mData;
};

#include "ImagePixelCommon.h"

#endif /* IMAGEPIXELINT16X2_H_ */
