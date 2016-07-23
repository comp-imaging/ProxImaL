/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _BASEMATH_H
#define _BASEMATH_H

#define MATH_PI_D    (3.1415926535897932384626433832795)
#define MATH_PI      ((float)MATH_PI_D)
#define MATH_RAD     ((float)(180.0/MATH_PI_D))
#define MATH_IRAD    ((float)(MATH_PI_D/180.0))
#define MATH_IRAD2   ((float)(MATH_PI_D/360.0))
#define MATH_DELTA   (0.000001f)
#define MATH_FLT_MAX (1e37f)

namespace Math
{

class Vec2f;
class Vec3f;
class Vec4f;
class Quat;

float Sinf(float angle);
float Cosf(float angle);
float Acosf(float cvalue);
float Tanf(float angle);
float Atan2f(float y, float x);
float Sqrtf(float v);
float InvSqrtf(float v);
float Floorf(float v);
float Ceilf(float v);
float Roundf(float v);
float Absf(float v);
float Modf(float n, float d);
float Powf(float x, float n);

float FastSqrtf(float v);
float FastInvSqrtf(float v);

class Matrix2x2f
{
public:
    Matrix2x2f(void)
    {
    }

    Matrix2x2f &setIdentity(void)
    {
        mData[0] = 1.0f;
        mData[1] = 0.0f;
        mData[2] = 0.0f;
        mData[3] = 1.0f;

        return *this;
    }

    float mData[4];
};

class Matrix3x2f
{
public:
    Matrix3x2f(void)
    {
    }

    Matrix3x2f &setIdentity(void)
    {
        mData[0] = 1.0f;
        mData[1] = 0.0f;
        mData[2] = 0.0f;
        mData[3] = 1.0f;
        mData[4] = 0.0f;
        mData[5] = 0.0f;

        return *this;
    }

    void invert(const Matrix3x2f &mat)
    {
        float invdet = 1.0f / (mat.mData[0] * mat.mData[3] - mat.mData[1] * mat.mData[2]);
        mData[0] = mat.mData[3] * invdet;
        mData[1] = -mat.mData[1] * invdet;
        mData[2] = -mat.mData[2] * invdet;
        mData[3] = mat.mData[0] * invdet;
        mData[4] = (mat.mData[2] * mat.mData[5] - mat.mData[4] * mat.mData[3]) * invdet;
        mData[5] = -(mat.mData[0] * mat.mData[5] - mat.mData[4] * mat.mData[1]) * invdet;
    }

    void operator *=(const Matrix3x2f &mat)
    {
        float m1, m2;

        m1 = mData[0];
        m2 = mData[1];
        mData[0] = m1 * mat.mData[0] + m2 * mat.mData[2];
        mData[1] = m1 * mat.mData[1] + m2 * mat.mData[3];

        m1 = mData[2];
        m2 = mData[3];
        mData[2] = m1 * mat.mData[0] + m2 * mat.mData[2];
        mData[3] = m1 * mat.mData[1] + m2 * mat.mData[3];

        m1 = mData[4];
        m2 = mData[5];
        mData[4] = m1 * mat.mData[0] + m2 * mat.mData[2] + mat.mData[4];
        mData[5] = m1 * mat.mData[1] + m2 * mat.mData[3] + mat.mData[5];
    }

    void setScale(const Vec2f &vec);
    void setRotate(float angle);
    void setTranslate(const Vec2f &vec);
    void setPretranslate(const Matrix3x2f &mat, const Vec2f &vec);

    void setSRT(const Vec2f &scale, float rotation, const Vec2f &position);

    float mData[6];
};

class Matrix3x3f
{
public:
    Matrix3x3f(void)
    {
    }

    void invert(const Matrix3x3f &mat);

    void operator *=(const Matrix3x3f &mat);

    void setTranslate(const Vec2f &vec);
    void setScale(const Vec2f &vec);
    void applyScale(const Vec2f &vec);
    void applyTranslate(const Vec2f &vec);

    Matrix3x3f &setIdentity(void)
    {
        mData[0] = 1.0f;
        mData[1] = 0.0f;
        mData[2] = 0.0f;
        mData[3] = 1.0f;
        mData[4] = 1.0f;
        mData[5] = 0.0f;
        mData[6] = 1.0f;
        mData[7] = 0.0f;
        mData[8] = 1.0f;

        return *this;
    }

    void applyTranspose(void);

    float mData[9];
};

class Matrix4x4f
{
public:
    enum EAxes
    {
        kAxisX, kAxisY, kAxisZ
    };

    Matrix4x4f(void)
    {
    }

    void operator *=(const Matrix4x4f &mat);

    void setIdentity(void);
    void setOrtho(float left, float right, float bottom, float top, float near, float far);
    void setFrustum(float left, float right, float bottom, float top, float near, float far);
    void setPerspective(float fovy, float aspect, float znear, float zfar);

    void setRotateByAxis(EAxes axis, float angle);
    void setRotate(const Quat &quat);
    void setScale(const Vec3f &vec);
    void applyScale(const Vec3f &vec);
    void setTranslate(const Vec3f &vec);
    void applyTranslate(const Vec3f &vec);
    void setTranspose(const Matrix4x4f &mat);
    void applyTranspose(void);

    void setSRT(const Vec3f &scale, const Quat &rotation, const Vec3f &position);

    float mData[16];
};

class Vec2b
{
public:
    Vec2b(void)
            : x(0), y(0)
    {
    }

    bool x, y;
};

class Vec3b
{
public:
    Vec3b(void)
            : x(0), y(0), z(0)
    {
    }

    bool x, y, z;
};

class Vec4b
{
public:
    Vec4b(void)
            : x(0), y(0), z(0), w(0)
    {
    }

    bool x, y, z, w;
};

class Vec2i
{
public:
    Vec2i(void)
            : x(0), y(0)
    {
    }
    Vec2i(int v)
            : x(v), y(v)
    {
    }
    Vec2i(int x, int y)
            : x(x), y(y)
    {
    }
    Vec2i(const Vec2i &v)
            : x(v.x), y(v.y)
    {
    }

    Vec2i &operator =(int v)
    {
        x = y = v;
        return *this;
    }

    Vec2i &operator =(const Vec2i &v)
    {
        x = v.x;
        y = v.y;
        return *this;
    }

    void operator +=(int v)
    {
        x += v;
        y += v;
    }

    void operator +=(const Vec2i &v)
    {
        x += v.x;
        y += v.y;
    }

    void operator -=(int v)
    {
        x -= v;
        y -= v;
    }

    void operator -=(const Vec2i &v)
    {
        x -= v.x;
        y -= v.y;
    }

    void operator *=(int v)
    {
        x *= v;
        y *= v;
    }

    void operator *=(const Vec2i &v)
    {
        x *= v.x;
        y *= v.y;
    }

    void operator /=(int v)
    {
        x /= v;
        y /= v;
    }

    void operator /=(const Vec2i &v)
    {
        x /= v.x;
        y /= v.y;
    }

    Vec2i operator +(int v) const
    {
        return Vec2i(x + v, y + v);
    }

    Vec2i operator +(const Vec2i &v) const
    {
        return Vec2i(x + v.x, y + v.y);
    }

    Vec2i operator -(void) const
    {
        return Vec2i(-x, -y);
    }

    Vec2i operator -(int v) const
    {
        return Vec2i(x - v, y - v);
    }

    Vec2i operator -(const Vec2i &v) const
    {
        return Vec2i(x - v.x, y - v.y);
    }

    Vec2i operator *(int v) const
    {
        return Vec2i(x * v, y * v);
    }

    Vec2i operator *(const Vec2i &v) const
    {
        return Vec2i(x * v.x, y * v.y);
    }

    Vec2i operator /(int v) const
    {
        return Vec2i(x / v, y / v);
    }

    Vec2i operator /(const Vec2i &v) const
    {
        return Vec2i(x / v.x, y / v.y);
    }

    int x, y;
};

class Vec3i
{
public:
    Vec3i(void)
            : x(0), y(0), z(0)
    {
    }

    int x, y, z;
};

class Vec4i
{
public:
    Vec4i(void)
            : x(0), y(0), z(0), w(0)
    {
    }

    int x, y, z, w;
};

class Vec2f
{
public:
    Vec2f(void)
            : x(0.0f), y(0.0f)
    {
    }
    Vec2f(float v)
            : x(v), y(v)
    {
    }
    Vec2f(float x, float y)
            : x(x), y(y)
    {
    }
    Vec2f(const Vec2f &v)
            : x(v.x), y(v.y)
    {
    }
    Vec2f(const Vec2i &v)
            : x((float)v.x), y((float)v.y)
    {
    }

    void set(float vx, float vy)
    {
        x = vx;
        y = vy;
    }

    Vec2f &operator =(float v)
    {
        x = y = v;
        return *this;
    }

    Vec2f &operator =(const Vec2f &v)
    {
        x = v.x;
        y = v.y;
        return *this;
    }

    void operator +=(float v)
    {
        x += v;
        y += v;
    }

    void operator +=(const Vec2f &v)
    {
        x += v.x;
        y += v.y;
    }

    void operator -=(float v)
    {
        x -= v;
        y -= v;
    }

    void operator -=(const Vec2f &v)
    {
        x -= v.x;
        y -= v.y;
    }

    void operator *=(float v)
    {
        x *= v;
        y *= v;
    }

    void operator *=(const Vec2f &v)
    {
        x *= v.x;
        y *= v.y;
    }

    void operator /=(float v)
    {
        x /= v;
        y /= v;
    }

    void operator /=(int v)
    {
        x /= v;
        y /= v;
    }

    void operator /=(const Vec2f &v)
    {
        x /= v.x;
        y /= v.y;
    }

    Vec2f operator +(float v) const
    {
        return Vec2f(x + v, y + v);
    }

    Vec2f operator +(const Vec2f &v) const
    {
        return Vec2f(x + v.x, y + v.y);
    }

    Vec2f operator -(void) const
    {
        return Vec2f(-x, -y);
    }

    Vec2f operator -(float v) const
    {
        return Vec2f(x - v, y - v);
    }

    Vec2f operator -(const Vec2f &v) const
    {
        return Vec2f(x - v.x, y - v.y);
    }

    Vec2f operator *(float v) const
    {
        return Vec2f(x * v, y * v);
    }

    Vec2f operator *(const Vec2f &v) const
    {
        return Vec2f(x * v.x, y * v.y);
    }

    Vec2f operator *(const Matrix3x2f &mat) const
    {
        return Vec2f(x * mat.mData[0] + y * mat.mData[2] + mat.mData[4],
                     x * mat.mData[1] + y * mat.mData[3] + mat.mData[5]);
    }

    Vec2f operator /(float v) const
    {
        return Vec2f(x / v, y / v);
    }

    Vec2f operator /(int v) const
    {
        return Vec2f(x / v, y / v);
    }

    Vec2f operator /(const Vec2f &v) const
    {
        return Vec2f(x / v.x, y / v.y);
    }

    float dot(Vec2f &v) const
    {
        return x * v.x + y * v.y;
    }

    Vec2f &normalize(void)
    {
        float scale = x * x + y * y;
        if (scale > 0.0f)
        {
            scale = FastInvSqrtf(scale);
            x *= scale;
            y *= scale;
        }
        else
        {
            x = y = 0.0f;
        }
        return *this;
    }

    float length(void) const
    {
        return FastSqrtf(x * x + y * y);
    }

    float length2(void) const
    {
        return x * x + y * y;
    }

    float distance(const Vec2f &v) const
    {
        return FastSqrtf((x - v.x) * (x - v.x) + (y - v.y) * (y - v.y));
    }

    float distance2(const Vec2f &v) const
    {
        return (x - v.x) * (x - v.x) + (y - v.y) * (y - v.y);
    }

    float x, y;
};

class Vec3f
{
public:
    Vec3f(void)
            : x(0.0f), y(0.0f), z(0.0f)
    {
    }
    Vec3f(float v)
            : x(v), y(v), z(v)
    {
    }
    Vec3f(const Vec2f &v)
            : x(v.x), y(v.y), z(0.0f)
    {
    }
    Vec3f(const Vec2f &v, float z)
            : x(v.x), y(v.y), z(z)
    {
    }
    Vec3f(float x, float y, float z)
            : x(x), y(y), z(z)
    {
    }
    Vec3f(const Vec3f &v)
            : x(v.x), y(v.y), z(v.z)
    {
    }

    Vec3f &operator =(float v)
    {
        x = y = z = v;
        return *this;
    }

    Vec3f &operator =(const Vec3f &v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }

    void set(float vx, float vy, float vz)
    {
        x = vx;
        y = vy;
        z = vz;
    }

    void operator +=(float v)
    {
        x += v;
        y += v;
        z += v;
    }

    void operator +=(const Vec3f &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
    }

    void operator -=(float v)
    {
        x -= v;
        y -= v;
        z -= v;
    }

    void operator -=(const Vec3f &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
    }

    void operator *=(float v)
    {
        x *= v;
        y *= v;
        z *= v;
    }

    void operator *=(const Vec3f &v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
    }

    void operator /=(float v)
    {
        float iv = 1.0f / v;
        x *= iv;
        y *= iv;
        z *= iv;
    }

    void operator /=(const Vec3f &v)
    {
        x /= v.x;
        y /= v.y;
        z /= v.z;
    }

    Vec3f operator +(float v) const
    {
        return Vec3f(x + v, y + v, z + v);
    }

    Vec3f operator +(const Vec3f &v) const
    {
        return Vec3f(x + v.x, y + v.y, z + v.z);
    }

    Vec3f operator -(void) const
    {
        return Vec3f(-x, -y, -z);
    }

    Vec3f operator -(float v) const
    {
        return Vec3f(x - v, y - v, z - v);
    }

    Vec3f operator -(const Vec3f &v) const
    {
        return Vec3f(x - v.x, y - v.y, z - v.z);
    }

    Vec3f operator *(float v) const
    {
        return Vec3f(x * v, y * v, z * v);
    }

    Vec3f operator *(const Vec3f &v) const
    {
        return Vec3f(x * v.x, y * v.y, z * v.z);
    }

    Vec3f operator *(const Matrix4x4f &mat) const
    {
        return Vec3f(x * mat.mData[0] + y * mat.mData[4] + z * mat.mData[8] + mat.mData[12],
                     x * mat.mData[1] + y * mat.mData[5] + z * mat.mData[9] + mat.mData[13],
                     x * mat.mData[2] + y * mat.mData[6] + z * mat.mData[10] + mat.mData[14]);
    }

    Vec3f operator /(float v) const
    {
        float iv = 1.0f / v;
        return Vec3f(x * iv, y * iv, z * iv);
    }

    Vec3f operator /(const Vec3f &v) const
    {
        return Vec3f(x / v.x, y / v.y, z / v.z);
    }

    float dot(const Vec3f &v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }

    Vec3f &normalize(void)
    {
        float scale = x * x + y * y + z * z;
        if (scale > 0.0f)
        {
            scale = FastInvSqrtf(scale);
            x *= scale;
            y *= scale;
            z *= scale;
        }
        else
        {
            x = y = z = 0.0f;
        }
        return *this;
    }

    float length(void) const
    {
        return FastSqrtf(x * x + y * y + z * z);
    }

    float length2(void) const
    {
        return x * x + y * y + z * z;
    }

    float distance(const Vec3f &v) const
    {
        return FastSqrtf((x - v.x) * (x - v.x) + (y - v.y) * (y - v.y) + (z - v.z) * (z - v.z));
    }

    float distance2(const Vec3f &v) const
    {
        return (x - v.x) * (x - v.x) + (y - v.y) * (y - v.y) + (z - v.z) * (z - v.z);
    }

    float x, y, z;
};

class Vec4f
{
public:
    Vec4f(void)
            : x(0.0f), y(0.0f), z(0.0f), w(0.0f)
    {
    }
    Vec4f(float x, float y, float z, float w)
            : x(x), y(y), z(z), w(w)
    {
    }
    Vec4f(const Vec4f &v)
            : x(v.x), y(v.y), z(v.z), w(v.w)
    {
    }

    Vec4f &operator =(const Vec4f &v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        w = v.w;
        return *this;
    }

    float x, y, z, w;
};

class Quat: public Vec4f
{
public:
    Quat(void)
            : Vec4f(0.0f, 0.0f, 0.0f, 1.0f)
    {
    }
    Quat(const Vec3f &v, float w)
            : Vec4f(v.x, v.y, v.z, w)
    {
        toQuaternion();
    }
    Quat(float x, float y, float z, float w)
            : Vec4f(x, y, z, w)
    {
    }
    Quat(const Quat &v)
            : Vec4f(v.x, v.y, v.z, v.w)
    {
    }

    void toQuaternion(void)
    {
        float angle = w * MATH_IRAD2;
        float d = Sinf(angle);
        x *= d;
        y *= d;
        z *= d;
        w = Cosf(angle);
    }

    void toAngleVector(void)
    {
        float ha = Acosf(w);
        float sinval2 = 1.0f - w * w;
        float isinval = FastInvSqrtf(sinval2);
        w = (2.0f * MATH_RAD) * ha;
        if (sinval2 * isinval < MATH_DELTA)
        {
            x = 1.0f;
            y = 0.0f;
            z = 0.0f;
        }
        else
        {
            x *= isinval;
            y *= isinval;
            z *= isinval;
        }
    }

    void inverse(void)
    {
        float d = x * x + y * y + z * z + w * w;
        if (d == 0.0f)
        {
            d = 1.0f;
        }
        else
        {
            d = 1.0f / d;
        }
        x *= -d;
        y *= -d;
        z *= -d;
        w *= d;
    }

    void operator *=(const Quat &q)
    {
        float nx = w * q.x + x * q.w + y * q.z - z * q.y;
        float ny = w * q.y + y * q.w + z * q.x - x * q.z;
        float nz = w * q.z + z * q.w + x * q.y - y * q.x;
        float nw = w * q.w - x * q.x - y * q.y - z * q.z;
        x = nx;
        y = ny;
        z = nz;
        w = nw;
    }

    Quat operator *(const Quat &q)
    {
        return Quat(w * q.x + x * q.w + y * q.z - z * q.y, w * q.y + y * q.w + z * q.x - x * q.z,
                    w * q.z + z * q.w + x * q.y - y * q.x, w * q.w - x * q.x - y * q.y - z * q.z);
    }

    void slerp(const Quat &q1, const Quat &q2, float t);
};
}

#endif

