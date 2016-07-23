/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _BASE_H
#define _BASE_H

/**
 * \file
 * Application-wide definitions.
 */

typedef unsigned long ulong;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

#ifdef __GNUC__

// GCC compiler
#define FORCE_INLINE __attribute__((always_inline)) inline
#define NO_INLINE __attribute__((noinline))
#define UNROLL_LOOPS __attribute__((optimize("unroll-loops")))
typedef long long int64;
typedef unsigned long long uint64;

#define GCC_VERSION (__GNUC__*10000+__GNUC_MINOR__*100+__GNUC_PATCHLEVEL__)
/* Test for GCC > 4.7.0 */
#if GCC_VERSION < 40700
#define __builtin_assume_aligned(x,s) (x)
#endif

#elif defined(_MSC_VER)

// MSVC compiler
#define FORCE_INLINE __forceinline
#define NO_INLINE __declspec(noinline)
#define UNROLL_LOOPS
typedef __int64 int64;
typedef unsigned __int64 uint64;

#define __builtin_assume_aligned(x,s) (x)
#undef USE_SSE4

#else

// TODO: RVCT?

#endif

// TODO: consider ARM arch setup
#ifdef ANDROID
#define CACHELINE_ALIGNMENT 32
#define SIMD_WORD_ALIGNMENT 16
#else
#define CACHELINE_ALIGNMENT 64
#define SIMD_WORD_ALIGNMENT 16
#endif

#define ALIGNED_SIZE(size,alignment) (((size)+((alignment)-1))&~((alignment)-1))

#define BIT_CLEAR(x,b) ((x)&=~(b))
#define BIT_SET(x,b)   ((x)|=(b))
#define BIT_ISSET(x,b) (((x)&(b))!=0)

#ifndef NULLPTR
#define NULLPTR ((void*)0)
#endif

template<typename T, typename U> FORCE_INLINE T *unsafe_pointer_cast(U *ptr)
{
    return static_cast<T *>(static_cast<void *>(ptr));
}

// vector load/store overloads
template<typename T> FORCE_INLINE void vstore(T *dst, T const &t)
{
    *dst = t;
}

template<typename T> FORCE_INLINE T vload(T const *src)
{
    return src[0];
}

namespace NVR
{
template<typename T> FORCE_INLINE void xchg(T &a, T &b)
{
    T tmp(a);
    a = b;
    b = tmp;
}
}

#endif

