/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation and
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution
 * of this software and related documentation without an express license
 * agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

/*
    Bicubic filtering
    See GPU Gems 2: "Fast Third-Order Texture Filtering", Sigg & Hadwiger
    http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter20.html

    Reformulation thanks to Keenan Crane
*/

#ifndef _BICUBICTEXTURE_KERNEL_CUH_
#define _BICUBICTEXTURE_KERNEL_CUH_


// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__host__ __device__
float w0(float a)
{
//    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

__host__ __device__
float w1(float a)
{
//    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
    return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__host__ __device__
float w2(float a)
{
//    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__host__ __device__
float w3(float a)
{
    return (1.0f/6.0f)*(a*a*a);
}

// g0 and g1 are the two amplitude functions
__device__ float g0(float a)
{
    return w0(a) + w1(a);
}

__device__ float g1(float a)
{
    return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
__device__ float h0(float a)
{
    // note +0.5 offset to compensate for CUDA linear filtering convention
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

__device__ float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

// filter 4 values using cubic splines
template<class T>
__device__
T cubicFilter(float x, T c0, T c1, T c2, T c3)
{
    T r;
    r = c0 * w0(x);
    r += c1 * w1(x);
    r += c2 * w2(x);
    r += c3 * w3(x);
    return r;
}

// slow but precise bicubic lookup using 16 texture lookups
template<class T>  // return type
__device__
T tex2DBicubic(cudaTextureObject_t texref, float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    return cubicFilter<T>(fy,
                          cubicFilter<T>(fx, tex2D<T>(texref, px-1, py-1), tex2D<T>(texref, px, py-1), tex2D<T>(texref, px+1, py-1), tex2D<T>(texref, px+2,py-1)),
                          cubicFilter<T>(fx, tex2D<T>(texref, px-1, py),   tex2D<T>(texref, px, py),   tex2D<T>(texref, px+1, py),   tex2D<T>(texref, px+2, py)),
                          cubicFilter<T>(fx, tex2D<T>(texref, px-1, py+1), tex2D<T>(texref, px, py+1), tex2D<T>(texref, px+1, py+1), tex2D<T>(texref, px+2, py+1)),
                          cubicFilter<T>(fx, tex2D<T>(texref, px-1, py+2), tex2D<T>(texref, px, py+2), tex2D<T>(texref, px+1, py+2), tex2D<T>(texref, px+2, py+2))
                          );
}

// fast bicubic texture lookup using 4 bilinear lookups
template<class T>  // return type
__device__
T tex2DFastBicubic(cudaTextureObject_t texref, float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    // note: we could store these functions in a lookup table texture, but maths is cheap
    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    T r = g0(fy) * ( g0x * tex2D<T>(texref, px + h0x, py + h0y)   +
                     g1x * tex2D<T>(texref, px + h1x, py + h0y) ) +
          g1(fy) * ( g0x * tex2D<T>(texref, px + h0x, py + h1y)   +
                     g1x * tex2D<T>(texref, px + h1x, py + h1y) );
    return r;
}

#endif // _BICUBICTEXTURE_KERNEL_CUH_
