/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include "ip/ImageYUV420p.h"
#include "ip/ImagePixelUInt8x4.h"
#include "math/FastMath.h"
#ifdef ARCH_ARM
#include <arm_neon.h>
#endif

#define RGB_TO_Y(r,g,b) FP16_ROUND(FP16_MULT(r,0.299)+FP16_MULT(g,0.587)+FP16_MULT(b,0.114))
#define RGB_TO_U(r,g,b) FP16_ROUND(FP16_MULT(r,-0.168736)+FP16_MULT(g,-0.331264)+FP16_MULT(b,0.5))
#define RGB_TO_V(r,g,b) FP16_ROUND(FP16_MULT(r,0.5)+FP16_MULT(g,-0.418688)+FP16_MULT(b,-0.081312))

#define YUV_TO_R(y,u,v) ((y)+FP16_ROUND(FP16_MULT(v,1.402)))
#define YUV_TO_G(y,u,v) ((y)-FP16_ROUND(FP16_MULT(u,0.34414)+FP16_MULT(v,0.71414)))
#define YUV_TO_B(y,u,v) ((y)+FP16_ROUND(FP16_MULT(u,1.772)))

ImageYUV420p::ImageYUV420p(int width, int height, int rowAlignment)
        : mWidth(width), mHeight(height), mStrideY(ALIGNED_SIZE(width, rowAlignment)),
          mStrideUV(ALIGNED_SIZE(width >> 1, rowAlignment)), mDataOwner(true),
          mDataY(System::MemoryAlloc<uchar>(mStrideY * mHeight + mStrideUV * (mHeight >> 1) * 2, CACHELINE_ALIGNMENT)),
          mDataU(mDataY + mStrideY * mHeight), mDataV(mDataU + mStrideUV * (mHeight >> 1))
{
}

ImageYUV420p::ImageYUV420p(Image<IPF_uint8x4> const &image, int rowAlignment)
        : mWidth(image.getWidth()), mHeight(image.getHeight()), mStrideY(ALIGNED_SIZE(image.getWidth(), rowAlignment)),
          mStrideUV(ALIGNED_SIZE(image.getWidth() >> 1, rowAlignment)), mDataOwner(true),
          mDataY(System::MemoryAlloc<uchar>(mStrideY * mHeight + mStrideUV * (mHeight >> 1) * 2, CACHELINE_ALIGNMENT)),
          mDataU(mDataY + mStrideY * mHeight), mDataV(mDataU + mStrideUV * (mHeight >> 1))
{
    uchar *ydata = mDataY;
    uchar *udata = mDataU;
    uchar *vdata = mDataV;
    uint *argbdata = (uint *)image.getRawPointer();

    for (int i = 0; i < mHeight; i += 2)
    {
        for (int j = 0; j < mWidth; j += 2)
        {
            uint p = argbdata[j];
            int cr = (p >> 16) & 0xff;
            int cg = (p >> 8) & 0xff;
            int cb = p & 0xff;

            ydata[j] = RGB_TO_Y(cr, cg, cb);
            udata[j >> 1] = 128 + RGB_TO_U(cr, cg, cb);
            vdata[j >> 1] = 128 + RGB_TO_V(cr, cg, cb);

            p = argbdata[j + 1];
            ydata[j + 1] = RGB_TO_Y((p >> 16) & 0xff, (p >> 8) & 0xff, p & 0xff);

            p = argbdata[j + mWidth];
            ydata[j + mStrideY] = RGB_TO_Y((p >> 16) & 0xff, (p >> 8) & 0xff, p & 0xff);

            p = argbdata[j + mWidth + 1];
            ydata[j + mStrideY + 1] = RGB_TO_Y((p >> 16) & 0xff, (p >> 8) & 0xff, p & 0xff);
        }

        argbdata += mWidth << 1;
        ydata += mStrideY << 1;
        udata += mStrideUV;
        vdata += mStrideUV;
    }
}

ImageYUV420p::~ImageYUV420p(void)
{
    if (mDataOwner)
    {
        System::MemoryFree(mDataY);
    }
}

void ImageYUV420p::copyTo(Image<IPF_uint8x4> &image) const
{
    image.set(mWidth, mHeight);

    uchar const *ydata = mDataY;
    uchar const *udata = mDataU;
    uchar const *vdata = mDataV;
    uint *rgbadata = (uint *)image.getRawPointer();

    int const strideY = mStrideY;
    int const strideUV = mStrideUV;
    int const width = mWidth;
    int const height = mHeight;
#ifdef ARCH_ARM
    bool const alignedRow = width != 0 && (width & 0xf) == 0 && (strideY & 0xf) == 0 && (strideUV & 0x7) == 0;
#endif

    for (int i = 0; i < height; i++)
    {
#ifdef ARCH_ARM
        if (alignedRow)
        {
            uchar const *py = ydata;
            uchar const *pu = udata;
            uchar const *pv = vdata;
            int length = width;

            asm volatile(" \
            movw        r0, #22968 \n\
            vdup.s16    q15, r0 \n\
            movw        r0, #59898 \n\
            vdup.s16    q14, r0 \n\
            movw        r0, #53837 \n\
            vdup.s16    q13, r0 \n\
            movw        r0, #29030 \n\
            vdup.s16    q12, r0 \n\
            mov         r0, #128 \n\
            vdup.s8     d23, r0 \n\
            mov         r0, #255 \n\
            vdup.s8     d11, r0 \n\
         2: \n\
            vld1.8      d0, [%[udata],:64]! \n\
            vld1.8      d2, [%[vdata],:64]! \n\
            vadd.s8     d0, d0, d23 \n\
            vadd.s8     d2, d2, d23 \n\
            pld         [%[ydata], %[PLD_LOOKAHEAD]] \n\
            vmovl.s8    q0, d0\n\
            vmovl.s8    q1, d2\n\
            vshl.s16    q0, q0, #7 \n\
            vshl.s16    q1, q1, #7 \n\
            vqdmulh.s16 q10, q1, q15 \n\
            vqdmulh.s16 q9, q0, q14 \n\
            vqdmulh.s16 q8, q1, q13 \n\
            vqdmulh.s16 q7, q0, q12 \n\
            vqadd.s16   q9, q9, q8 \n\
            vld1.8      {d12-d13}, [%[ydata],:128]! \n\
            vmov        q0, q10 \n\
            vmov        q1, q9 \n\
            vmov        q2, q7 \n\
            pld         [%[udata], %[PLD_LOOKAHEAD]] \n\
            vzip.16     q0, q10 \n\
            vzip.16     q1, q9 \n\
            vzip.16     q2, q7 \n\
            \
            vmovl.u8    q8, d12 \n\
            vshl.s16    q8, q8, #6 \n\
            pld         [%[vdata], %[PLD_LOOKAHEAD]] \n\
            vqadd.s16   q0, q0, q8 \n\
            vqadd.s16   q1, q1, q8 \n\
            vqadd.s16   q2, q2, q8 \n\
            vshr.s16    q0, q0, #6 \n\
            vshr.s16    q1, q1, #6 \n\
            vshr.s16    q2, q2, #6 \n\
            vqmovun.s16 d10, q0 \n\
            vqmovun.s16 d9, q1 \n\
            vqmovun.s16 d8, q2 \n\
            vst4.8      {d8-d11}, [%[rgbadata],:256]! \n\
            subs        %[length], %[length], #16 \n\
            \
            vmovl.u8    q8, d13 \n\
            vshl.s16    q8, q8, #6 \n\
            vqadd.s16   q10, q10, q8 \n\
            vqadd.s16   q9, q9, q8 \n\
            vqadd.s16   q7, q7, q8 \n\
            vshr.s16    q10, q10, #6 \n\
            vshr.s16    q9, q9, #6 \n\
            vshr.s16    q7, q7, #6 \n\
            vqmovun.s16 d10, q10 \n\
            vqmovun.s16 d9, q9 \n\
            vqmovun.s16 d8, q7 \n\
            vst4.8      {d8-d11}, [%[rgbadata],:256]! \n\
            bhi         2b \n\
        " : [length]"+r"(length), [ydata]"+r"(py), [udata]"+r"(pu), [vdata]"+r"(pv), [rgbadata]"+r"(rgbadata)
                    : [PLD_LOOKAHEAD]"I"(32) : "cc", "q4", "q5", "q6", "q7", "r0");
        }
        else
        {
#endif
            for (int j = 0; j < width; j++)
            {
                int y = ydata[j];
                int u = udata[j >> 1] - 128;
                int v = vdata[j >> 1] - 128;

                int r = YUV_TO_R(y, u, v);
                r = min_i32(255, CLAMP_AT_ZERO(r));
                int g = YUV_TO_G(y, u, v);
                g = min_i32(255, CLAMP_AT_ZERO(g));
                int b = YUV_TO_B(y, u, v);
                b = min_i32(255, CLAMP_AT_ZERO(b));

                *rgbadata++ = (r << 16) | (g << 8) | b;
            }
#ifdef ARCH_ARM
        }
#endif
        ydata += strideY;
        if (IS_ODD(i))
        {
            udata += strideUV;
            vdata += strideUV;
        }
    }
}

