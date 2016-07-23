/*
 * Vecs.h
 *
 *  Created on: Mar 7, 2013
 *      Author: ytsai
 */

#ifndef VECS_H_
#define VECS_H_

#include <cuda_runtime.h>
#include <Core/Types.h>

__host__  __device__  __forceinline__ short4 operator+(const short4& lhs,
        const int& s) {
    short4 result;
    result.x = lhs.x + s;
    result.y = lhs.y + s;
    result.z = lhs.z + s;
    result.w = lhs.w + s;

    return result;
}

__host__  __device__  __forceinline__ short4 operator+(const short4& lhs,
        const short4& rhs) {
    short4 result;
    result.x = lhs.x + rhs.x;
    result.y = lhs.y + rhs.y;
    result.z = lhs.z + rhs.z;
    result.w = lhs.w + rhs.w;

    return result;
}

__host__  __device__  __forceinline__ short4 operator-(const short4& lhs,
        const short4& rhs) {
    short4 result;
    result.x = lhs.x - rhs.x;
    result.y = lhs.y - rhs.y;
    result.z = lhs.z - rhs.z;
    result.w = lhs.w - rhs.w;

    return result;
}

__host__  __device__  __forceinline__ short4 operator*(const short4& lhs,
        const short4& rhs) {
    short4 result;
    result.x = lhs.x * rhs.x;
    result.y = lhs.y * rhs.y;
    result.z = lhs.z * rhs.z;
    result.w = lhs.w * rhs.w;

    return result;
}

__host__  __device__  __forceinline__ short4 operator*(const short4& lhs,
        const int& s) {
    short4 result;
    result.x = lhs.x * s;
    result.y = lhs.y * s;
    result.z = lhs.z * s;
    result.w = lhs.w * s;

    return result;
}

__host__  __device__  __forceinline__ short4 operator/(const short4& lhs,
        const int& s) {
    short4 result;
    result.x = lhs.x / s;
    result.y = lhs.y / s;
    result.z = lhs.z / s;
    result.w = lhs.w / s;

    return result;
}

//////////

__host__  __device__  __forceinline__ float4 operator+(const float4& lhs,
        const float4& rhs) {
    float4 result;
    result.x = lhs.x + rhs.x;
    result.y = lhs.y + rhs.y;
    result.z = lhs.z + rhs.z;
    result.w = lhs.w + rhs.w;

    return result;
}

__host__  __device__  __forceinline__ float4 operator-(const float4& lhs,
        const float4& rhs) {
    float4 result;
    result.x = lhs.x - rhs.x;
    result.y = lhs.y - rhs.y;
    result.z = lhs.z - rhs.z;
    result.w = lhs.w - rhs.w;

    return result;
}

__host__  __device__  __forceinline__ float4 operator*(const float4& lhs,
        const float& s) {
    float4 result;
    result.x = lhs.x * s;
    result.y = lhs.y * s;
    result.z = lhs.z * s;
    result.w = lhs.w * s;

    return result;
}

__host__  __device__  __forceinline__ float4 operator/(const float4& lhs,
        const float& s) {
    float4 result;
    result.x = lhs.x / s;
    result.y = lhs.y / s;
    result.z = lhs.z / s;
    result.w = lhs.w / s;

    return result;
}

//////////

__host__  __device__  __forceinline__ float2 operator+(const float2& lhs,
        const float2& rhs) {
    float2 result;
    result.x = lhs.x + rhs.x;
    result.y = lhs.y + rhs.y;

    return result;
}

__host__  __device__  __forceinline__ float2 operator-(const float2& lhs,
        const float2& rhs) {
    float2 result;
    result.x = lhs.x - rhs.x;
    result.y = lhs.y - rhs.y;

    return result;
}

__host__  __device__  __forceinline__ float2 operator*(const float2& lhs,
        const float& s) {
    float2 result;
    result.x = lhs.x * s;
    result.y = lhs.y * s;

    return result;
}

__host__  __device__  __forceinline__ float2 operator/(const float2& lhs,
        const float& s) {
    float2 result;
    result.x = lhs.x / s;
    result.y = lhs.y / s;

    return result;
}

//////////

__host__  __device__  __forceinline__ float3 operator+(const float3& lhs,
        const float3& rhs) {
    float3 result;
    result.x = lhs.x + rhs.x;
    result.y = lhs.y + rhs.y;
    result.z = lhs.z + rhs.z;

    return result;
}

__host__  __device__  __forceinline__ float3 operator-(const float3& lhs,
        const float3& rhs) {
    float3 result;
    result.x = lhs.x - rhs.x;
    result.y = lhs.y - rhs.y;
    result.z = lhs.z - rhs.z;

    return result;
}

__host__  __device__  __forceinline__ float3 operator*(const float3& lhs,
        const float& s) {
    float3 result;
    result.x = lhs.x * s;
    result.y = lhs.y * s;
    result.z = lhs.z * s;

    return result;
}

__host__  __device__  __forceinline__ float3 operator/(const float3& lhs,
        const float& s) {
    float3 result;
    result.x = lhs.x / s;
    result.y = lhs.y / s;
    result.z = lhs.z / s;

    return result;
}


#endif /* VECS_H_ */
