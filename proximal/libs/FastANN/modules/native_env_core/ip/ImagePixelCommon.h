/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

// pixel-type conversion constructors
#if defined(IPFF32X4_SUPPORTED) && defined(IPFUI8X4_SUPPORTED) && !defined(CONV_F32X4_UI8X4)
#define CONV_F32X4_UI8X4

#ifdef ARCH_X86
FORCE_INLINE ImagePixelUInt8x4::ImagePixelUInt8x4(ImagePixelFloat32x4 const &t) :
mData( _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(t.mData), _mm_setzero_si128()), _mm_setzero_si128())))
{
}

FORCE_INLINE ImagePixelFloat32x4::ImagePixelFloat32x4(ImagePixelUInt8x4 const &t) :
mData(
        _mm_cvtepi32_ps(
                _mm_unpacklo_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128(t.mData), _mm_setzero_si128()), _mm_setzero_si128())))
{
}
#endif

#ifdef ARCH_ARM
FORCE_INLINE ImagePixelUInt8x4::ImagePixelUInt8x4(ImagePixelFloat32x4 const &t) :
mData( vget_lane_u32( vreinterpret_u32_u8( vqmovun_s16( vcombine_s16( vqmovn_s32( vcvtq_s32_f32( t.mData ) ), vdup_n_s16(0) ) ) ), 0) )

{
}

FORCE_INLINE ImagePixelFloat32x4::ImagePixelFloat32x4(ImagePixelUInt8x4 const &t) :
mData( vcvtq_f32_u32( vmovl_u16( vget_low_u16( vmovl_u8( vreinterpret_u8_u32( vdup_n_u32( t.mData ) ) ) ) ) ) )

{
}

#endif

#endif

#if defined(IPFF32X4_SUPPORTED) && defined(IPFF16X4_SUPPORTED) && !defined(CONV_F32X4_F16X4)
#define CONV_F32X4_F16X4

#ifdef ARCH_ARM

FORCE_INLINE ImagePixelFloat32x4::ImagePixelFloat32x4(ImagePixelFloat16x4 const &t)
{
    asm volatile ("vcvt.f32.f16 %q0, %P1 \n" : "=w"(mData) : "w"(t.mData));
}

#endif

#endif

#if defined(IPFF32X4_SUPPORTED) && defined(IPFUI16X4_SUPPORTED) && !defined(CONV_F32X4_UI16X4)
#define CONV_F32X4_UI16X4

#ifdef ARCH_X86
FORCE_INLINE ImagePixelFloat32x4::ImagePixelFloat32x4( ImagePixelUInt16x4 const &t ) :
mData( _mm_cvtepi32_ps( _mm_unpacklo_epi16( _mm_loadl_epi64( reinterpret_cast<const __m128i *>( t.mData ) ), _mm_setzero_si128() ) ) )
{

}
#endif

#ifdef ARCH_ARM
FORCE_INLINE ImagePixelFloat32x4::ImagePixelFloat32x4( ImagePixelUInt16x4 const &t ) :
mData( vcvtq_f32_u32( vmovl_u16(t.mData) ) )
{
}
#endif

#endif

#if defined(IPFF32X4_SUPPORTED) && defined(IPFUI32X4_SUPPORTED) && !defined(CONV_F32X4_UI32X4)
#define CONV_F32X4_UI32X4

#ifdef ARCH_X86
FORCE_INLINE ImagePixelFloat32x4::ImagePixelFloat32x4( ImagePixelUInt32x4 const &t ) :
mData( _mm_cvtepi32_ps( t.mData ) )
{

}
#endif

#ifdef ARCH_ARM
FORCE_INLINE ImagePixelFloat32x4::ImagePixelFloat32x4( ImagePixelUInt32x4 const &t ) :
mData( vcvtq_f32_u32( t.mData ) )
{
}
#endif

#endif

