/*
 * Types.h
 *
 *  Created on: Jan 19, 2013
 *      Author: ytsai
 */

#ifndef TYPES_H_
#define TYPES_H_

#include <cuda_runtime.h>

#include <Log/Log.h>

#ifdef __CUDACC__
#define __NV_HOST_AND_DEVICE__ __host__ __device__
#else
#define __NV_HOST_AND_DEVICE__
#endif

enum NV_STATUS {
	NV_SUCCESS, NV_FAILED
};

enum PixType {
	UCHAR,
	SHORT,
	USHORT,
	INT,
	FLOAT,
	HALF,
	FLOAT3,
	UCHAR4,
	SHORT4,
	FLOAT4
};

enum TexReadMode {
    RD_NORMALIZED_FLOAT,
    RD_ELEMENT_TYPE
};

enum TexFilterMode {
    FL_POINT,
    FL_LINEAR
};

enum MemoryFormat {
    BLOCK_LINEAR,
    PITCH2D
};

enum IOType {
	NV_INPUT, NV_OUTPUT
};

namespace nv {

struct half
{
    unsigned short data;
    half() {}
    half(float f)
    {
        //TODO: adjust coversion to mirror CUDA __float2half_rn
        unsigned int floatval = *reinterpret_cast<unsigned int*>(&f);
        unsigned int halfval = ((floatval>>16)&0x8000)|((((floatval&0x7f800000)-0x38000000)>>13)&0x7c00)|((floatval>>13)&0x03ff);
        data = halfval;
    }
    operator float() const
    {
        //TODO: adjust coversion to mirror CUDA __half2float
        unsigned int floatval = ((data&0x8000)<<16) | (((data&0x7c00)+0x1C000)<<13) | ((data&0x03FF)<<13);
        return *reinterpret_cast<float*>(&floatval);
    }
};

struct half2
{
    short2 data;
    half2() {}
    half2(float2 f)
    {
        //TODO: adjust coversion to mirror CUDA __float2half_rn
        unsigned int floatval = *reinterpret_cast<unsigned int*>(&f.x);
        unsigned int halfval = ((floatval>>16)&0x8000)|((((floatval&0x7f800000)-0x38000000)>>13)&0x7c00)|((floatval>>13)&0x03ff);
        data.x = halfval;
        floatval = *reinterpret_cast<unsigned int*>(&f.y);
        halfval = ((floatval>>16)&0x8000)|((((floatval&0x7f800000)-0x38000000)>>13)&0x7c00)|((floatval>>13)&0x03ff);
        data.y = halfval;
    }
    operator float2() const
    {
        //TODO: adjust coversion to mirror CUDA __half2float
        float2 o;
        unsigned int floatval = ((data.x&0x8000)<<16) | (((data.x&0x7c00)+0x1C000)<<13) | ((data.x&0x03FF)<<13);
        o.x = *reinterpret_cast<float*>(&floatval);
        floatval = ((data.y&0x8000)<<16) | (((data.y&0x7c00)+0x1C000)<<13) | ((data.y&0x03FF)<<13);
        o.y = *reinterpret_cast<float*>(&floatval);
        return o;
    }
};

struct half4
{
    short4 data;
    half4() {}
    half4(float4 f)
    {
        //TODO: adjust coversion to mirror CUDA __float2half_rn
        unsigned int floatval = *reinterpret_cast<unsigned int*>(&f.x);
        unsigned int halfval = ((floatval>>16)&0x8000)|((((floatval&0x7f800000)-0x38000000)>>13)&0x7c00)|((floatval>>13)&0x03ff);
        data.x = halfval;
        floatval = *reinterpret_cast<unsigned int*>(&f.y);
        halfval = ((floatval>>16)&0x8000)|((((floatval&0x7f800000)-0x38000000)>>13)&0x7c00)|((floatval>>13)&0x03ff);
        data.y = halfval;
        floatval = *reinterpret_cast<unsigned int*>(&f.z);
        halfval = ((floatval>>16)&0x8000)|((((floatval&0x7f800000)-0x38000000)>>13)&0x7c00)|((floatval>>13)&0x03ff);
        data.z = halfval;
        floatval = *reinterpret_cast<unsigned int*>(&f.w);
        halfval = ((floatval>>16)&0x8000)|((((floatval&0x7f800000)-0x38000000)>>13)&0x7c00)|((floatval>>13)&0x03ff);
        data.w = halfval;
    }
    operator float4() const
    {
        //TODO: adjust coversion to mirror CUDA __half2float
        float4 o;
        unsigned int floatval = ((data.x&0x8000)<<16) | (((data.x&0x7c00)+0x1C000)<<13) | ((data.x&0x03FF)<<13);
        o.x = *reinterpret_cast<float*>(&floatval);
        floatval = ((data.y&0x8000)<<16) | (((data.y&0x7c00)+0x1C000)<<13) | ((data.y&0x03FF)<<13);
        o.y = *reinterpret_cast<float*>(&floatval);
        floatval = ((data.z&0x8000)<<16) | (((data.z&0x7c00)+0x1C000)<<13) | ((data.z&0x03FF)<<13);
        o.z = *reinterpret_cast<float*>(&floatval);
        floatval = ((data.w&0x8000)<<16) | (((data.w&0x7c00)+0x1C000)<<13) | ((data.w&0x03FF)<<13);
        o.w = *reinterpret_cast<float*>(&floatval);
        return o;
    }
};



template<typename T>
class Type2PixType {
};

template<>
class Type2PixType<unsigned char> {
public:
    static const PixType PixT = UCHAR;
};

template<>
class Type2PixType<short> {
public:
    static const PixType PixT = SHORT;
};

template<>
class Type2PixType<unsigned short> {
public:
    static const PixType PixT = USHORT;
};

template<>
class Type2PixType<int> {
public:
    static const PixType PixT = INT;
};

template<>
class Type2PixType<float> {
public:
    static const PixType PixT = FLOAT;
};

template<>
class Type2PixType<half> {
public:
    static const PixType PixT = HALF;
};

template<>
class Type2PixType<uchar4> {
public:
    static const PixType PixT = UCHAR4;
};

template<>
class Type2PixType<short4> {
public:
    static const PixType PixT = SHORT4;
};

template<>
class Type2PixType<float3> {
public:
    static const PixType PixT = FLOAT3;
};

template<>
class Type2PixType<float4> {
public:
    static const PixType PixT = FLOAT4;
};

template<PixType T>
class PixTypeTraits {
public:
	static inline cudaChannelFormatDesc GetChannelDesc();
	static inline int GetChannels();
	static inline int GetChannelBitDepth();
	static inline int SizeOf();
	static inline float NormalizedActualValue();
};

template<>
class PixTypeTraits<UCHAR> {
public:
	static inline cudaChannelFormatDesc GetChannelDesc() {
		int channel_bit_depth = GetChannelBitDepth();
		return cudaCreateChannelDesc(channel_bit_depth, 0, 0, 0,
				cudaChannelFormatKindUnsigned);
	}

	static inline int GetChannels() {
		return 1;
	}
	static inline int GetChannelBitDepth() {
		return sizeof(unsigned char) * 8;
	}
	static inline int SizeOf() {
		return sizeof(unsigned char);
	}
	static inline float NormalizedActualValue()
	{
	    return 255.0f;
	}
};

template<>
class PixTypeTraits<SHORT> {
public:
	static inline cudaChannelFormatDesc GetChannelDesc() {
		int channel_bit_depth = GetChannelBitDepth();
		return cudaCreateChannelDesc(channel_bit_depth, 0, 0, 0,
				cudaChannelFormatKindSigned);
	}

	static inline int GetChannels() {
		return 1;
	}
	static inline int GetChannelBitDepth() {
		return sizeof(short) * 8;
	}
	static inline int SizeOf() {
		return sizeof(short);
	}
	static inline float NormalizedActualValue()
    {
        return 32767.0f;
    }
};

template<>
class PixTypeTraits<USHORT> {
public:
    static inline cudaChannelFormatDesc GetChannelDesc() {
        int channel_bit_depth = GetChannelBitDepth();
        return cudaCreateChannelDesc(channel_bit_depth, 0, 0, 0,
                cudaChannelFormatKindUnsigned);
    }

    static inline int GetChannels() {
        return 1;
    }
    static inline int GetChannelBitDepth() {
        return sizeof(unsigned short) * 8;
    }
    static inline int SizeOf() {
        return sizeof(unsigned short);
    }
    static inline float NormalizedActualValue()
    {
        return 65535.0f;
    }
};

template<>
class PixTypeTraits<INT> {
public:
    static inline cudaChannelFormatDesc GetChannelDesc() {
        int channel_bit_depth = GetChannelBitDepth();
        return cudaCreateChannelDesc(channel_bit_depth, 0, 0, 0,
                cudaChannelFormatKindSigned);
    }

    static inline int GetChannels() {
        return 1;
    }
    static inline int GetChannelBitDepth() {
        return sizeof(int) * 8;
    }
    static inline int SizeOf() {
        return sizeof(int);
    }
    static inline float NormalizedActualValue()
    {
        return 2147483647.0f;
    }
};

template<>
class PixTypeTraits<FLOAT> {
public:
    static inline cudaChannelFormatDesc GetChannelDesc() {
        int channel_bit_depth = GetChannelBitDepth();
        return cudaCreateChannelDesc(channel_bit_depth, 0, 0, 0,
                cudaChannelFormatKindFloat);
    }

    static inline int GetChannels() {
        return 1;
    }
    static inline int GetChannelBitDepth() {
        return sizeof(float) * 8;
    }
    static inline int SizeOf() {
        return sizeof(float);
    }
    static inline float NormalizedActualValue()
    {
        return 1.0f;
    }
};


template<>
class PixTypeTraits<HALF> {
public:
    static inline cudaChannelFormatDesc GetChannelDesc() {
        return cudaCreateChannelDescHalf();
    }

    static inline int GetChannels() {
        return 1;
    }
    static inline int GetChannelBitDepth() {
        return sizeof(unsigned short) * 8;
    }
    static inline int SizeOf() {
        return sizeof(unsigned short);
    }
    static inline float NormalizedActualValue()
    {
        return 1.0f;
    }
};

template<>
class PixTypeTraits<FLOAT3> {
public:
    static inline cudaChannelFormatDesc GetChannelDesc() {
        int channel_bit_depth = GetChannelBitDepth();
        return cudaCreateChannelDesc(channel_bit_depth, channel_bit_depth,
                channel_bit_depth, 0,
                cudaChannelFormatKindFloat);
    }

    static inline int GetChannels() {
        return 3;
    }
    static inline int GetChannelBitDepth() {
        return sizeof(float) * 8;
    }
    static inline int SizeOf() {
        return sizeof(float3);
    }
    static inline float NormalizedActualValue()
    {
        return 1.0f;
    }
};

template<>
class PixTypeTraits<UCHAR4> {
public:
    static inline cudaChannelFormatDesc GetChannelDesc() {
        int channel_bit_depth = GetChannelBitDepth();
        return cudaCreateChannelDesc(channel_bit_depth, channel_bit_depth,
                channel_bit_depth, channel_bit_depth,
                cudaChannelFormatKindUnsigned);
    }

    static inline int GetChannels() {
        return 4;
    }
    static inline int GetChannelBitDepth() {
        return sizeof(unsigned char) * 8;
    }
    static inline int SizeOf() {
        return sizeof(uchar4);
    }
    static inline float NormalizedActualValue()
    {
        return 255.0f;
    }

};

template<>
class PixTypeTraits<SHORT4> {
public:
	static inline cudaChannelFormatDesc GetChannelDesc() {
		int channel_bit_depth = GetChannelBitDepth();
		return cudaCreateChannelDesc(channel_bit_depth, channel_bit_depth,
				channel_bit_depth, channel_bit_depth,
				cudaChannelFormatKindSigned);
	}

	static inline int GetChannels() {
		return 4;
	}
	static inline int GetChannelBitDepth() {
		return sizeof(short) * 8;
	}
	static inline int SizeOf() {
		return sizeof(short4);
	}
	static inline float NormalizedActualValue()
    {
        return 32767.0f;
    }
};

template<>
class PixTypeTraits<FLOAT4> {
public:
    static inline cudaChannelFormatDesc GetChannelDesc() {
        int channel_bit_depth = GetChannelBitDepth();
        return cudaCreateChannelDesc(channel_bit_depth, channel_bit_depth,
                channel_bit_depth, channel_bit_depth,
                cudaChannelFormatKindFloat);
    }

    static inline int GetChannels() {
        return 4;
    }
    static inline int GetChannelBitDepth() {
        return sizeof(float) * 8;
    }
    static inline int SizeOf() {
        return sizeof(float4);
    }
    static inline float NormalizedActualValue()
    {
        return 1.0f;
    }
};

struct LaunchParams {
	// address offset from the origin
	dim3 offset;
	// cta size
	dim3 block_size;
//	// maximum concurrent cta
//	dim3 max_concurrent_blocks;
//	// stream id
//	cudaStream_t stream;
};

}

#endif /* TYPES_H_ */
