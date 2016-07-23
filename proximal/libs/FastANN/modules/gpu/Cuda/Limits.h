#ifndef LIMITS_H_
#define LIMITS_H_

#include <limits>

namespace nv {
template<class T> struct numeric_limits
{
	typedef T type;
	__device__  __forceinline__  static type min() {
		return type();
	}
	;
	__device__  __forceinline__  static type max() {
		return type();
	}
	;
	__device__  __forceinline__  static type epsilon() {
		return type();
	}
	__device__  __forceinline__  static type round_error() {
		return type();
	}
	__device__  __forceinline__  static type denorm_min() {
		return type();
	}
	__device__  __forceinline__  static type infinity() {
		return type();
	}
	__device__  __forceinline__  static type quiet_NaN() {
		return type();
	}
	__device__  __forceinline__  static type signaling_NaN() {
		return T();
	}
	static const bool is_signed;
};

template<> struct numeric_limits<bool>
{
	typedef bool type;
	__device__  __forceinline__  static type min() {
		return false;
	}
	;
	__device__  __forceinline__  static type max() {
		return true;
	}
	;
	__device__  __forceinline__  static type epsilon();
	__device__  __forceinline__  static type round_error();
	__device__  __forceinline__  static type denorm_min();
	__device__  __forceinline__  static type infinity();
	__device__  __forceinline__  static type quiet_NaN();
	__device__  __forceinline__  static type signaling_NaN();
	static const bool is_signed = false;
};

template<> struct numeric_limits<char>
{
	typedef char type;
	__device__  __forceinline__  static type min() {
		return CHAR_MIN;
	}
	;
	__device__  __forceinline__  static type max() {
		return CHAR_MAX;
	}
	;
	__device__  __forceinline__  static type epsilon();
	__device__  __forceinline__  static type round_error();
	__device__  __forceinline__  static type denorm_min();
	__device__  __forceinline__  static type infinity();
	__device__  __forceinline__  static type quiet_NaN();
	__device__  __forceinline__  static type signaling_NaN();
	static const bool is_signed = (char) -1 == -1;
};

template<> struct numeric_limits<signed char>
{
	typedef char type;
	__device__  __forceinline__  static type min() {
		return SCHAR_MIN;
	}
	;
	__device__  __forceinline__  static type max() {
		return SCHAR_MAX;
	}
	;
	__device__  __forceinline__  static type epsilon();
	__device__  __forceinline__  static type round_error();
	__device__  __forceinline__  static type denorm_min();
	__device__  __forceinline__  static type infinity();
	__device__  __forceinline__  static type quiet_NaN();
	__device__  __forceinline__  static type signaling_NaN();
	static const bool is_signed = (signed char) -1 == -1;
};

template<> struct numeric_limits<unsigned char>
{
	typedef unsigned char type;
	__device__  __forceinline__  static type min() {
		return 0;
	}
	;
	__device__  __forceinline__  static type max() {
		return UCHAR_MAX;
	}
	;
	__device__  __forceinline__  static type epsilon();
	__device__  __forceinline__  static type round_error();
	__device__  __forceinline__  static type denorm_min();
	__device__  __forceinline__  static type infinity();
	__device__  __forceinline__  static type quiet_NaN();
	__device__  __forceinline__  static type signaling_NaN();
	static const bool is_signed = false;
};

template<> struct numeric_limits<short>
{
	typedef short type;
	__device__  __forceinline__  static type min() {
		return SHRT_MIN;
	}
	;
	__device__  __forceinline__  static type max() {
		return SHRT_MAX;
	}
	;
	__device__  __forceinline__  static type epsilon();
	__device__  __forceinline__  static type round_error();
	__device__  __forceinline__  static type denorm_min();
	__device__  __forceinline__  static type infinity();
	__device__  __forceinline__  static type quiet_NaN();
	__device__  __forceinline__  static type signaling_NaN();
	static const bool is_signed = true;
};

template<> struct numeric_limits<unsigned short>
{
	typedef unsigned short type;
	__device__  __forceinline__  static type min() {
		return 0;
	}
	;
	__device__  __forceinline__  static type max() {
		return USHRT_MAX;
	}
	;
	__device__  __forceinline__  static type epsilon();
	__device__  __forceinline__  static type round_error();
	__device__  __forceinline__  static type denorm_min();
	__device__  __forceinline__  static type infinity();
	__device__  __forceinline__  static type quiet_NaN();
	__device__  __forceinline__  static type signaling_NaN();
	static const bool is_signed = false;
};

template<> struct numeric_limits<int>
{
	typedef int type;
	__device__  __forceinline__  static type min() {
		return INT_MIN;
	}
	;
	__device__  __forceinline__  static type max() {
		return INT_MAX;
	}
	;
	__device__  __forceinline__  static type epsilon();
	__device__  __forceinline__  static type round_error();
	__device__  __forceinline__  static type denorm_min();
	__device__  __forceinline__  static type infinity();
	__device__  __forceinline__  static type quiet_NaN();
	__device__  __forceinline__  static type signaling_NaN();
	static const bool is_signed = true;
};

template<> struct numeric_limits<unsigned int>
{
	typedef unsigned int type;
	__device__  __forceinline__  static type min() {
		return 0;
	}
	;
	__device__  __forceinline__  static type max() {
		return UINT_MAX;
	}
	;
	__device__  __forceinline__  static type epsilon();
	__device__  __forceinline__  static type round_error();
	__device__  __forceinline__  static type denorm_min();
	__device__  __forceinline__  static type infinity();
	__device__  __forceinline__  static type quiet_NaN();
	__device__  __forceinline__  static type signaling_NaN();
	static const bool is_signed = false;
};

template<> struct numeric_limits<long>
{
	typedef long type;
	__device__  __forceinline__  static type min() {
		return LONG_MIN;
	}
	;
	__device__  __forceinline__  static type max() {
		return LONG_MAX;
	}
	;
	__device__  __forceinline__  static type epsilon();
	__device__  __forceinline__  static type round_error();
	__device__  __forceinline__  static type denorm_min();
	__device__  __forceinline__  static type infinity();
	__device__  __forceinline__  static type quiet_NaN();
	__device__  __forceinline__  static type signaling_NaN();
	static const bool is_signed = true;
};

template<> struct numeric_limits<unsigned long>
{
	typedef unsigned long type;
	__device__  __forceinline__  static type min() {
		return 0;
	}
	;
	__device__  __forceinline__  static type max() {
		return ULONG_MAX;
	}
	;
	__device__  __forceinline__  static type epsilon();
	__device__  __forceinline__  static type round_error();
	__device__  __forceinline__  static type denorm_min();
	__device__  __forceinline__  static type infinity();
	__device__  __forceinline__  static type quiet_NaN();
	__device__  __forceinline__  static type signaling_NaN();
	static const bool is_signed = false;
};

template<> struct numeric_limits<float>
{
	typedef float type;
	__device__  __forceinline__  static type min() {
		return 1.175494351e-38f/*FLT_MIN*/;
	}
	;
	__device__  __forceinline__  static type max() {
		return 3.402823466e+38f/*FLT_MAX*/;
	}
	;
	__device__  __forceinline__  static type epsilon() {
		return 1.192092896e-07f/*FLT_EPSILON*/;
	}
	;
	__device__  __forceinline__  static type round_error();
	__device__  __forceinline__  static type denorm_min();
	__device__  __forceinline__  static type infinity();
	__device__  __forceinline__  static type quiet_NaN();
	__device__  __forceinline__  static type signaling_NaN();
	static const bool is_signed = true;
};

template<> struct numeric_limits<double>
{
	typedef double type;
	__device__  __forceinline__  static type min() {
		return 2.2250738585072014e-308/*DBL_MIN*/;
	}
	;
	__device__  __forceinline__  static type max() {
		return 1.7976931348623158e+308/*DBL_MAX*/;
	}
	;
	__device__  __forceinline__  static type epsilon() {
		return 2.2204460492503131e-16;
	}
	__device__  __forceinline__  static type round_error();
	__device__  __forceinline__  static type denorm_min();
	__device__  __forceinline__  static type infinity();
	__device__  __forceinline__  static type quiet_NaN();
	__device__  __forceinline__  static type signaling_NaN();
	static const bool is_signed = true;
};

} // namepsace nv

#endif
