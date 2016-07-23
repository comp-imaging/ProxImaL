/*
 * Arithmetic.h
 *
 *  Created on: Mar 7, 2013
 *      Author: ytsai
 */

#ifndef ARITHMETIC_H_
#define ARITHMETIC_H_

#define DIVUP(X, Y) ((X) + (Y) - 1) / (Y)

namespace nv {

/// credit to Dawid and Norbert

template<typename IntType>
class IntTypeTraits {
public:
    __host__ __device__
    static int scale();
};

template<>
class IntTypeTraits<unsigned char> {
public:
    __host__ __device__
    static float scale() {
        return 255.0f;
    }
};

template<>
class IntTypeTraits<signed char> {
public:
    __host__ __device__
    static float scale() {
        return 127.0f;
    }
};

template<>
class IntTypeTraits<short> {
public:
    __host__ __device__
    static float scale() {
        return 32767.0f;
    }
};

template<>
class IntTypeTraits<unsigned short> {
public:
    __host__ __device__
    static float scale() {
        return 65535.0f;
    }
};

template<>
class IntTypeTraits<int> {
public:
    __host__ __device__
    static float scale() {
        return 2147483647.0f;
    }
};

template<>
class IntTypeTraits<unsigned int> {
public:
    __host__ __device__
    static float scale() {
        return 4294967295.0f;
    }
};

template<typename IntType>
__device__  __inline__ IntType fast_cvt_rn_halfup(float f) {
    float val;

    float const scale = IntTypeTraits<IntType>::scale();
    float const magic = 8388608.0f + 4194304.0f; // 2^23 (shift the mantissa) + 2^22 (shift the sign bits if it is set)
    val = __fmaf_rn(f, scale, 0.5);
    val = __fadd_rz(val, magic);

    return (IntType) __float_as_int(val);
}

template<typename FloatType, typename IntType>
__device__  __inline__ IntType Denorm(FloatType f);

template<>
__device__ __inline__ unsigned char Denorm(float f) {
    return fast_cvt_rn_halfup<unsigned char>(f);
}

template<>
__device__  __inline__ uchar4 Denorm(float4 f) {
    uchar4 val;
    val.x = fast_cvt_rn_halfup<unsigned char>(f.x);
    val.y = fast_cvt_rn_halfup<unsigned char>(f.y);
    val.z = fast_cvt_rn_halfup<unsigned char>(f.z);
    val.w = fast_cvt_rn_halfup<unsigned char>(f.w);

    return val;
}

template<>
__device__ __inline__ short Denorm(float f) {
    return fast_cvt_rn_halfup<short>(f);
}

template<>
__device__ __inline__ unsigned short Denorm(float f) {
    return fast_cvt_rn_halfup<unsigned short>(f);
}

template<>
__device__ __inline__ int Denorm(float f) {
    return fast_cvt_rn_halfup<int>(f);
}

template<>
__device__ __inline__ unsigned int Denorm(float f) {
    return fast_cvt_rn_halfup<unsigned int>(f);
}

template<>
__device__ __inline__ float Denorm(float f) {
    return f;
}

template<>
__device__ __inline__ nv::half Denorm(float f) {
    unsigned short v = __float2half_rn(f);
    return *reinterpret_cast<nv::half*>(&v);
}

template<>
__device__  __inline__ short4 Denorm(float4 f) {
    short4 val;
    val.x = fast_cvt_rn_halfup<short>(f.x);
    val.y = fast_cvt_rn_halfup<short>(f.y);
    val.z = fast_cvt_rn_halfup<short>(f.z);
    val.w = fast_cvt_rn_halfup<short>(f.w);

    return val;
}

}//namespace nv


#endif /* ARITHMETIC_H_ */
