/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include "BaseMath.h"
#include <math.h>

#define IDX_3X3(r,c) ((r)*3+(c))

// ================================================================
// BASIC MATH FUNCTIONS IMPLEMENTATION
// ================================================================

float Math::Floorf(float v)
{
    return floorf(v);
}

float Math::Ceilf(float v)
{
    return ceilf(v);
}

float Math::Roundf(float v)
{
    return (v > 0.0f) ? floorf(v + 0.5f) : ceilf(v - 0.5f);
}

float Math::Absf(float v)
{
    return fabsf(v);
}

float Math::Modf(float n, float d)
{
    return fmodf(n, d);
}

float Math::Powf(float x, float n)
{
    return powf(x, n);
}

float Math::Sinf(float angle)
{
    return sinf(angle);
}

float Math::Cosf(float angle)
{
    return cosf(angle);
}

float Math::Acosf(float cvalue)
{
    return acosf(cvalue);
}

float Math::Tanf(float angle)
{
    return tanf(angle);
}

float Math::Atan2f(float y, float x)
{
    return atan2f(y, x);
}

float Math::Sqrtf(float v)
{
    return sqrtf(v);
}

float Math::InvSqrtf(float v)
{
    float x2 = v * 0.5f;
    float y = v;
    // evil approximation hack
    int i = *(int *)&y;
    i = 0x5f3759df - (i >> 1);
    y = *(float *)&i;

    // 2 newton iteration
    y = y * (1.5f - (x2 * y * y));
    y = y * (1.5f - (x2 * y * y));

    return y;
}

float Math::FastSqrtf(float v)
{
    float x2 = v * 0.5f;
    float y = v;
    // evil approximation hack
    int i = *(int *)&y;
    i = 0x5f3759df - (i >> 1);
    y = *(float *)&i;

    // newton iteration
    y = y * (1.5f - (x2 * y * y));

    return v * y;
}

float Math::FastInvSqrtf(float v)
{
    float x2 = v * 0.5f;
    float y = v;
    // evil approximation hack
    int i = *(int *)&y;
    i = 0x5f3759df - (i >> 1);
    y = *(float *)&i;

    // newton iteration
    y = y * (1.5f - (x2 * y * y));

    return y;
}

// ================================================================
// 3x2 MATRIX IMPLEMENTATION
// ================================================================

using namespace Math;

void Matrix3x2f::setSRT(const Vec2f &scale, float rotation, const Vec2f &position)
{
    float msin = sinf(rotation * MATH_IRAD);
    float mcos = cosf(rotation * MATH_IRAD);

    mData[0] = mcos * scale.x;
    mData[1] = msin * scale.x;
    mData[2] = -msin * scale.y;
    mData[3] = mcos * scale.y;
    mData[4] = position.x;
    mData[5] = position.y;
}

void Matrix3x2f::setRotate(float angle)
{
    float msin = sinf(angle * MATH_IRAD);
    float mcos = cosf(angle * MATH_IRAD);

    mData[0] = mcos;
    mData[1] = msin;
    mData[2] = -msin;
    mData[3] = mcos;
    mData[4] = 0.0f;
    mData[5] = 0.0f;
}

void Matrix3x2f::setTranslate(const Vec2f &vec)
{
    mData[0] = 1.0f;
    mData[1] = 0.0f;
    mData[2] = 0.0f;
    mData[3] = 1.0f;
    mData[4] = vec.x;
    mData[5] = vec.y;
}

void Matrix3x2f::setPretranslate(const Matrix3x2f &matrix, const Vec2f &vec)
{
    mData[0] = matrix.mData[0];
    mData[1] = matrix.mData[1];
    mData[2] = matrix.mData[2];
    mData[3] = matrix.mData[3];
    mData[4] = vec.x * mData[0] + vec.y * mData[2] + matrix.mData[4];
    mData[5] = vec.x * mData[1] + vec.y * mData[3] + matrix.mData[5];
}

void Matrix3x2f::setScale(const Vec2f &vec)
{
    mData[0] = vec.x;
    mData[1] = 0.0f;
    mData[2] = 0.0f;
    mData[3] = vec.y;
    mData[4] = 0.0f;
    mData[5] = 0.0f;
}

// ================================================================
// 3x3 MATRIX IMPLEMENTATION
// ================================================================

void Matrix3x3f::applyTranspose(void)
{
    int *idata = (int *)mData;
    int t = idata[1];
    idata[1] = idata[3];
    idata[3] = t;

    t = idata[2];
    idata[2] = idata[6];
    idata[6] = t;

    t = idata[5];
    idata[5] = idata[7];
    idata[7] = t;
}

void Matrix3x3f::setTranslate(const Vec2f &vec)
{
    mData[0] = 1.0f;
    mData[1] = 0.0f;
    mData[2] = 0.0f;
    mData[3] = 0.0f;
    mData[4] = 1.0f;
    mData[5] = 0.0f;
    mData[6] = vec.x;
    mData[7] = vec.y;
    mData[8] = 1.0f;
}

void Matrix3x3f::setScale(const Vec2f &vec)
{
    mData[0] = vec.x;
    mData[1] = 0.0f;
    mData[2] = 0.0f;
    mData[3] = 0.0f;
    mData[4] = vec.y;
    mData[5] = 0.0f;
    mData[6] = 0.0f;
    mData[7] = 0.0f;
    mData[8] = 1.0f;
}

void Matrix3x3f::applyScale(const Vec2f &vec)
{
    mData[0] *= vec.x;
    mData[1] *= vec.y;

    mData[3] *= vec.x;
    mData[4] *= vec.y;

    mData[6] *= vec.x;
    mData[7] *= vec.y;
}

void Matrix3x3f::applyTranslate(const Vec2f &vec)
{
    mData[0] += mData[2] * vec.x;
    mData[1] += mData[2] * vec.y;

    mData[3] += mData[5] * vec.x;
    mData[4] += mData[5] * vec.y;

    mData[6] += mData[8] * vec.x;
    mData[7] += mData[8] * vec.y;
}

void Matrix3x3f::operator *=(const Matrix3x3f &mat)
{
    for (int y = 0; y < 3; y++)
    {
        int index = y * 3;
        float m1 = mData[index];
        float m2 = mData[index + 1];
        float m3 = mData[index + 2];
        mData[index] = m1 * mat.mData[0] + m2 * mat.mData[3] + m3 * mat.mData[6];
        mData[index + 1] = m1 * mat.mData[1] + m2 * mat.mData[4] + m3 * mat.mData[7];
        mData[index + 2] = m1 * mat.mData[2] + m2 * mat.mData[5] + m3 * mat.mData[8];
    }
}

void Matrix3x3f::invert(const Matrix3x3f &mat)
{
    float determinant = mat.mData[IDX_3X3(0,0)]
                        * (mat.mData[IDX_3X3(1,1)] * mat.mData[IDX_3X3(2,2)]
                           - mat.mData[IDX_3X3(2,1)] * mat.mData[IDX_3X3(1,2)])
                        - mat.mData[IDX_3X3(0,1)]
                          * (mat.mData[IDX_3X3(1,0)] * mat.mData[IDX_3X3(2,2)]
                             - mat.mData[IDX_3X3(1,2)] * mat.mData[IDX_3X3(2,0)])
                        + mat.mData[IDX_3X3(0,2)]
                          * (mat.mData[IDX_3X3(1,0)] * mat.mData[IDX_3X3(2,1)]
                             - mat.mData[IDX_3X3(1,1)] * mat.mData[IDX_3X3(2,0)]);

    float invdet = 1 / determinant;
    mData[IDX_3X3( 0, 0 )] = (mat.mData[IDX_3X3(1,1)] * mat.mData[IDX_3X3(2,2)]
                              - mat.mData[IDX_3X3(2,1)] * mat.mData[IDX_3X3(1,2)])
                             * invdet;

    mData[IDX_3X3( 0, 1 )] = -(mat.mData[IDX_3X3(0,1)] * mat.mData[IDX_3X3(2,2)]
                               - mat.mData[IDX_3X3(0,2)] * mat.mData[IDX_3X3(2,1)])
                             * invdet;

    mData[IDX_3X3( 0, 2 )] = (mat.mData[IDX_3X3(0,1)] * mat.mData[IDX_3X3(1,2)]
                              - mat.mData[IDX_3X3(0,2)] * mat.mData[IDX_3X3(1,1)])
                             * invdet;

    mData[IDX_3X3( 1, 0 )] = -(mat.mData[IDX_3X3(1,0)] * mat.mData[IDX_3X3(2,2)]
                               - mat.mData[IDX_3X3(1,2)] * mat.mData[IDX_3X3(2,0)])
                             * invdet;

    mData[IDX_3X3( 1, 1 )] = (mat.mData[IDX_3X3(0,0)] * mat.mData[IDX_3X3(2,2)]
                              - mat.mData[IDX_3X3(0,2)] * mat.mData[IDX_3X3(2,0)])
                             * invdet;

    mData[IDX_3X3( 1, 2 )] = -(mat.mData[IDX_3X3(0,0)] * mat.mData[IDX_3X3(1,2)]
                               - mat.mData[IDX_3X3(1,0)] * mat.mData[IDX_3X3(0,2)])
                             * invdet;

    mData[IDX_3X3( 2, 0 )] = (mat.mData[IDX_3X3(1,0)] * mat.mData[IDX_3X3(2,1)]
                              - mat.mData[IDX_3X3(2,0)] * mat.mData[IDX_3X3(1,1)])
                             * invdet;

    mData[IDX_3X3( 2, 1 )] = -(mat.mData[IDX_3X3(0,0)] * mat.mData[IDX_3X3(2,1)]
                               - mat.mData[IDX_3X3(2,0)] * mat.mData[IDX_3X3(0,1)])
                             * invdet;

    mData[IDX_3X3( 2, 2 )] = (mat.mData[IDX_3X3(0,0)] * mat.mData[IDX_3X3(1,1)]
                              - mat.mData[IDX_3X3(1,0)] * mat.mData[IDX_3X3(0,1)])
                             * invdet;
}

// ================================================================
// 4x4 MATRIX IMPLEMENTATION
// XXX: Open GL style matrix format (column-wise)
// ================================================================

void Matrix4x4f::setIdentity(void)
{
    mData[0] = 1.0f;
    mData[1] = 0.0f;
    mData[2] = 0.0f;
    mData[3] = 0.0f;
    mData[4] = 0.0f;
    mData[5] = 1.0f;
    mData[6] = 0.0f;
    mData[7] = 0.0f;
    mData[8] = 0.0f;
    mData[9] = 0.0f;
    mData[10] = 1.0f;
    mData[11] = 0.0f;
    mData[12] = 0.0f;
    mData[13] = 0.0f;
    mData[14] = 0.0f;
    mData[15] = 1.0f;
}

void Matrix4x4f::setScale(const Vec3f &vec)
{
    mData[0] = vec.x;
    mData[1] = 0.0f;
    mData[2] = 0.0f;
    mData[3] = 0.0f;
    mData[4] = 0.0f;
    mData[5] = vec.y;
    mData[6] = 0.0f;
    mData[7] = 0.0f;
    mData[8] = 0.0f;
    mData[9] = 0.0f;
    mData[10] = vec.z;
    mData[11] = 0.0f;
    mData[12] = 0.0f;
    mData[13] = 0.0f;
    mData[14] = 0.0f;
    mData[15] = 1.0f;
}

void Matrix4x4f::applyScale(const Vec3f &vec)
{
    mData[0] *= vec.x;
    mData[1] *= vec.y;
    mData[2] *= vec.z;

    mData[4] *= vec.x;
    mData[5] *= vec.y;
    mData[6] *= vec.z;

    mData[8] *= vec.x;
    mData[9] *= vec.y;
    mData[10] *= vec.z;

    mData[12] *= vec.x;
    mData[13] *= vec.y;
    mData[14] *= vec.z;
}

void Matrix4x4f::setTranslate(const Vec3f &vec)
{
    mData[0] = 1.0f;
    mData[1] = 0.0f;
    mData[2] = 0.0f;
    mData[3] = 0.0f;
    mData[4] = 0.0f;
    mData[5] = 1.0f;
    mData[6] = 0.0f;
    mData[7] = 0.0f;
    mData[8] = 0.0f;
    mData[9] = 0.0f;
    mData[10] = 1.0f;
    mData[11] = 0.0f;
    mData[12] = vec.x;
    mData[13] = vec.y;
    mData[14] = vec.z;
    mData[15] = 1.0f;
}

void Matrix4x4f::applyTranslate(const Vec3f &vec)
{
    mData[0] += mData[3] * vec.x;
    mData[1] += mData[3] * vec.y;
    mData[2] += mData[3] * vec.z;

    mData[4] += mData[7] * vec.x;
    mData[5] += mData[7] * vec.y;
    mData[6] += mData[7] * vec.z;

    mData[8] += mData[11] * vec.x;
    mData[9] += mData[11] * vec.y;
    mData[10] += mData[11] * vec.z;

    mData[12] += mData[15] * vec.x;
    mData[13] += mData[15] * vec.y;
    mData[14] += mData[15] * vec.z;
}

void Matrix4x4f::setTranspose(const Matrix4x4f &mat)
{
    int *srcdata = (int *)mat.mData;
    int *dstdata = (int *)mData;
    dstdata[0] = srcdata[0];
    dstdata[1] = srcdata[4];
    dstdata[2] = srcdata[8];
    dstdata[3] = srcdata[12];
    dstdata[4] = srcdata[1];
    dstdata[5] = srcdata[5];
    dstdata[6] = srcdata[9];
    dstdata[7] = srcdata[13];
    dstdata[8] = srcdata[2];
    dstdata[9] = srcdata[6];
    dstdata[10] = srcdata[10];
    dstdata[11] = srcdata[14];
    dstdata[12] = srcdata[3];
    dstdata[13] = srcdata[7];
    dstdata[14] = srcdata[11];
    dstdata[15] = srcdata[15];
}

void Matrix4x4f::applyTranspose(void)
{
    int *idata = (int *)mData;
    int t = idata[1];
    idata[1] = idata[4];
    idata[4] = t;

    t = idata[2];
    idata[2] = idata[8];
    idata[8] = t;

    t = idata[3];
    idata[3] = idata[12];
    idata[12] = t;
    t = idata[6];
    idata[6] = idata[9];
    idata[9] = t;

    t = idata[7];
    idata[7] = idata[13];
    idata[13] = t;

    t = idata[11];
    idata[11] = idata[14];
    idata[14] = t;
}

void Matrix4x4f::setRotateByAxis(EAxes axis, float angle)
{
    setIdentity();

    angle = angle * MATH_IRAD;
    float msin = Sinf(angle);
    float mcos = Cosf(angle);

    switch (axis)
    {
    case kAxisX:
        mData[5] = mcos;
        mData[6] = msin;
        mData[9] = -msin;
        mData[10] = mcos;
        break;

    case kAxisY:
        mData[0] = mcos;
        mData[2] = -msin;
        mData[8] = msin;
        mData[10] = mcos;
        break;

    case kAxisZ:
        mData[0] = mcos;
        mData[1] = msin;
        mData[4] = -msin;
        mData[5] = mcos;
        break;
    }
}

void Matrix4x4f::setRotate(const Quat &quat)
{
    float mx, my, mz;
    float xx, xy, xz;
    float yy, yz, zz;
    float wx, wy, wz;

    mx = quat.x + quat.x;
    my = quat.y + quat.y;
    mz = quat.z + quat.z;

    xx = quat.x * mx;
    xy = quat.x * my;
    xz = quat.x * mz;

    yy = quat.y * my;
    yz = quat.y * mz;
    zz = quat.z * mz;

    wx = quat.w * mx;
    wy = quat.w * my;
    wz = quat.w * mz;

    mData[0] = 1.0f - (yy + zz);
    mData[1] = xy + wz;
    mData[2] = xz - wy;
    mData[3] = 0.0f;

    mData[4] = xy - wz;
    mData[5] = 1.0f - (xx + zz);
    mData[6] = yz + wx;
    mData[7] = 0.0f;

    mData[8] = xz + wy;
    mData[9] = yz - wx;
    mData[10] = 1.0f - (xx + yy);
    mData[11] = 0.0f;

    mData[12] = 0.0f;
    mData[13] = 0.0f;
    mData[14] = 0.0f;
    mData[15] = 1.0f;
}

void Matrix4x4f::setOrtho(float l, float r, float b, float t, float n, float f)
{
    float rl = 1.0f / (r - l);
    float tb = 1.0f / (t - b);
    float fn = 1.0f / (f - n);

    mData[0] = 2.0f * rl;
    mData[1] = 0.0f;
    mData[2] = 0.0f;
    mData[3] = 0.0f;

    mData[4] = 0.0f;
    mData[5] = 2.0f * tb;
    mData[6] = 0.0f;
    mData[7] = 0.0f;

    mData[8] = 0.0f;
    mData[9] = 0.0f;
    mData[10] = -2.0f * fn;
    mData[11] = 0.0f;

    mData[12] = -(r + l) * rl;
    mData[13] = -(t + b) * tb;
    mData[14] = -(f + n) * fn;
    mData[15] = 1.0f;
}

void Matrix4x4f::setFrustum(float l, float r, float b, float t, float n, float f)
{
    float rl = 1.0f / (r - l);
    float tb = 1.0f / (t - b);
    float fn = 1.0f / (f - n);

    mData[0] = 2.0f * n * rl;
    mData[1] = 0.0f;
    mData[2] = 0.0f;
    mData[3] = 0.0f;

    mData[4] = 0.0f;
    mData[5] = 2.0f * n * tb;
    mData[6] = 0.0f;
    mData[7] = 0.0f;

    mData[8] = (r + l) * rl;
    mData[9] = (t + b) * tb;
    mData[10] = -(f + n) * fn;
    mData[11] = -1.0f;

    mData[12] = 0.0f;
    mData[13] = 0.0f;
    mData[14] = -2.0f * f * n * fn;
    mData[15] = 0.0f;
}

void Matrix4x4f::setPerspective(float fovy, float aspect, float znear, float zfar)
{
    float ymax = znear * Tanf(fovy * MATH_IRAD2);
    float ymin = -ymax;
    float xmin = ymin * aspect;
    float xmax = ymax * aspect;
    setFrustum(xmin, xmax, ymin, ymax, znear, zfar);
}

void Matrix4x4f::operator *=(const Matrix4x4f &mat)
{
    for (int y = 0; y < 4; y++)
    {
        int index = y << 2;
        float m1 = mData[index];
        float m2 = mData[index + 1];
        float m3 = mData[index + 2];
        float m4 = mData[index + 3];
        mData[index] = m1 * mat.mData[0] + m2 * mat.mData[4] + m3 * mat.mData[8] + m4 * mat.mData[12];
        mData[index + 1] = m1 * mat.mData[1] + m2 * mat.mData[5] + m3 * mat.mData[9] + m4 * mat.mData[13];
        mData[index + 2] = m1 * mat.mData[2] + m2 * mat.mData[6] + m3 * mat.mData[10] + m4 * mat.mData[14];
        mData[index + 3] = m1 * mat.mData[3] + m2 * mat.mData[7] + m3 * mat.mData[11] + m4 * mat.mData[15];
    }
}

void Matrix4x4f::setSRT(const Vec3f &scale, const Quat &rotation, const Vec3f &position)
{
    setRotate(rotation);

    mData[0] *= scale.x;
    mData[1] *= scale.x;
    mData[2] *= scale.x;

    mData[4] *= scale.y;
    mData[5] *= scale.y;
    mData[6] *= scale.y;

    mData[8] *= scale.z;
    mData[9] *= scale.z;
    mData[10] *= scale.z;

    mData[12] = position.x;
    mData[13] = position.y;
    mData[14] = position.z;
}

// ================================================================
// QUATERNION CLASS IMPLEMENTATION
// ================================================================

void Quat::slerp(const Quat &q1, const Quat &q2, float t)
{
    Quat to(q2);

    float mcos = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;
    if (mcos < 0.0f)
    {
        mcos = -mcos;
        to.x = -to.x;
        to.y = -to.y;
        to.z = -to.z;
        to.w = -to.w;
    }

    float s0, s1;
    if ((1.0f - mcos) > MATH_DELTA)
    {
        float omega = Acosf(mcos);
        float imsin = 1.0f / Sinf(omega);
        s0 = Sinf((1.0f - t) * omega) * imsin;
        s1 = Sinf(t * omega) * imsin;
    }
    else
    {
        s0 = 1.0f - t;
        s1 = t;
    }

    x = q1.x * s0 + to.x * s1;
    y = q1.y * s0 + to.y * s1;
    z = q1.z * s0 + to.z * s1;
    w = q1.w * s0 + to.w * s1;
}
