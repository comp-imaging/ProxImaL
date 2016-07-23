/*
 * Memory.h
 *
 *  Created on: Mar 7, 2013
 *      Author: ytsai
 */

#ifndef MEMORY_H_
#define MEMORY_H_

#include <Cuda/Memory/DevPtr.h>
#include <Cuda/Memory/SharedMemory.h>
#include <Cuda/Arithmetic.h>

namespace nv {
template<typename T>
__device__  __inline__ T Read2D(cudaTextureObject_t ref, int x, int y) {
    return tex2D < T > (ref, x, y);
}

template<typename T>
__device__  __inline__ T Read2D(cudaTextureObject_t ref, float x, float y) {
    return tex2D < T > (ref, x, y);
}

template<>
__device__ __inline__ float Read2D(cudaTextureObject_t ref, int x, int y) {
   float4 tmp;
   asm volatile ("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];" : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w) : "l"(ref), "r"(x), "r"(y));
   return (float)(tmp.x);
}

template<typename T>
__device__  __inline__ T Read2D(void const* in, int x, int y, int width_stride) {
    int idx = y * width_stride + sizeof(T) * x;
    return reinterpret_cast<T const*>(reinterpret_cast<char const*>(in) + idx)[0];
}

__device__  __inline__ float4 Read2D(void const* in, int x, int y, int width_stride, int width, int height) {
	int offset = width * height * sizeof(float);
    int idx = (y * width + x) * sizeof(float);

    float4 val;
    val.x = reinterpret_cast<float const*>(reinterpret_cast<char const*>(in) + idx)[0];
    idx += offset;
    val.y = reinterpret_cast<float const*>(reinterpret_cast<char const*>(in) + idx)[0];
    idx += offset;
    val.z = reinterpret_cast<float const*>(reinterpret_cast<char const*>(in) + idx)[0];
    idx += offset;
    val.w = reinterpret_cast<float const*>(reinterpret_cast<char const*>(in) + idx)[0];

    return val;
}



template<typename T>
__device__  __inline__ T Read2D(void const volatile* in, int x, int y, int width_stride) {
    int idx = y * width_stride + sizeof(T) * x;

    T val;
    val.x = reinterpret_cast<T const volatile*>(reinterpret_cast<char const volatile*>(in) + idx)[0].x;
    val.y = reinterpret_cast<T const volatile*>(reinterpret_cast<char const volatile*>(in) + idx)[0].y;
    val.z = reinterpret_cast<T const volatile*>(reinterpret_cast<char const volatile*>(in) + idx)[0].z;
    val.w = reinterpret_cast<T const volatile*>(reinterpret_cast<char const volatile*>(in) + idx)[0].w;

    return val;
}

template<typename FloatType, typename IntType>
__device__ __inline__ void Write2D(cudaSurfaceObject_t ref,
        const FloatType& val, int x, int y) {
    IntType ival = Denorm<FloatType, IntType>(val);
    surf2Dwrite(ival, ref, x * sizeof(IntType), y);
}

template<typename T>
__device__ __inline__ void Write2D(cudaSurfaceObject_t ref,
        const T& val, int x, int y) {
    surf2Dwrite(val, ref, x * sizeof(T), y);
}

template<typename T>
__device__ __inline__ void Write2D(void* out, const T& val, int x, int y, int width_stride) {
    int idx = y * width_stride + sizeof(T) * x;
    reinterpret_cast<T*>(reinterpret_cast<char*>(out) + idx)[0] = val;
}

__device__ __inline__ void Write2D(void* out, const float4& val, int x, int y, int width_stride, int width, int height) {
	int offset = width * height * sizeof(float);
    int idx = (y * width + x) * sizeof(float);
    reinterpret_cast<float*>(reinterpret_cast<char*>(out) + idx)[0] = val.x;
    idx += offset;
    reinterpret_cast<float*>(reinterpret_cast<char*>(out) + idx)[0] = val.y;
    idx += offset;
    reinterpret_cast<float*>(reinterpret_cast<char*>(out) + idx)[0] = val.z;
    idx += offset;
    reinterpret_cast<float*>(reinterpret_cast<char*>(out) + idx)[0] = val.w;
}

template<typename FloatType, typename IntType>
__device__ __inline__ void Write2D(void *out, const FloatType& val, int x, int y, int width_stride) {
    IntType ival = Denorm<FloatType, IntType>(val);
    int idx = y * width_stride + sizeof(IntType) * x;
    reinterpret_cast<IntType*>(reinterpret_cast<char *>(out) + idx)[0] = ival;
}

template<typename FloatType, typename IntType>
__device__ __inline__ void Write2D(void volatile* out, const FloatType& val, int x, int y, int width_stride) {
    IntType ival = Denorm<FloatType, IntType>(val);
    int idx = y * width_stride + sizeof(IntType) * x;
    reinterpret_cast<IntType volatile*>(reinterpret_cast<char volatile*>(out) + idx)[0].x = ival.x;
    reinterpret_cast<IntType volatile*>(reinterpret_cast<char volatile*>(out) + idx)[0].y = ival.y;
    reinterpret_cast<IntType volatile*>(reinterpret_cast<char volatile*>(out) + idx)[0].z = ival.z;
    reinterpret_cast<IntType volatile*>(reinterpret_cast<char volatile*>(out) + idx)[0].w = ival.w;
}

template<typename TIn>
__device__ __inline__ void AtomicAdd2D(void* out, const TIn& val, int x, int y, int width_stride) {
    printf("Type is not supported!\n");
}

template<>
__device__ __inline__ void AtomicAdd2D(void* out, const float& val, int x, int y, int width_stride)
{
    int idx = y * width_stride + sizeof(float) * x;
    atomicAdd(reinterpret_cast<float*>(reinterpret_cast<char*>(out) + idx), val);
}

//template<>
//__device__ __inline__ void Write2D(void volatile* out, const float4& val, int x, int y, int width_stride) {
//    volatile uchar4 ival = Denorm<float4, uchar4>(val);
//    int idx = y * width_stride + sizeof(IntType) * x;
//    reinterpret_cast<uchar4 volatile*>(reinterpret_cast<char volatile*>(out) + idx)[0].x = ival.x;
//}
} // namespace nv


#endif /* MEMORY_H_ */
