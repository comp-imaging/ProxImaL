/*
 * FrameBuffer.cpp
 *
 *  Created on: Feb 1, 2013
 *      Author: ytsai
 */

#include "FrameBuffer.h"
#include <Log/Log.h>
#include <string.h>
#include <assert.h>

namespace nv {
namespace internal {

inline cudaChannelFormatDesc CreateChannelDesc(PixType type) {
    cudaChannelFormatDesc channel_desc;
    switch (type) {
    case UCHAR: {
        channel_desc = PixTypeTraits<UCHAR>::GetChannelDesc();
        break;
    }
    case SHORT: {
        channel_desc = PixTypeTraits<SHORT>::GetChannelDesc();
        break;
    }
    case USHORT: {
        channel_desc = PixTypeTraits<USHORT>::GetChannelDesc();
        break;
    }
    case INT: {
        channel_desc = PixTypeTraits<INT>::GetChannelDesc();
        break;
    }
    case HALF: {
        channel_desc = PixTypeTraits<HALF>::GetChannelDesc();
        break;
    }
    case FLOAT: {
        channel_desc = PixTypeTraits<FLOAT>::GetChannelDesc();
        break;
    }
    case FLOAT3: {
        channel_desc = PixTypeTraits<FLOAT3>::GetChannelDesc();
        break;
    }
    case UCHAR4: {
        channel_desc = PixTypeTraits<UCHAR4>::GetChannelDesc();
        break;
    }
    case SHORT4: {
        channel_desc = PixTypeTraits<SHORT4>::GetChannelDesc();
        break;
    }
    case FLOAT4: {
        channel_desc = PixTypeTraits<FLOAT4>::GetChannelDesc();
        break;
    }
    }
    return channel_desc;
}

inline size_t GetSizeOf(PixType type) {
    switch (type) {
    case UCHAR: {
        return PixTypeTraits<UCHAR>::SizeOf();
    }
    case SHORT: {
        return PixTypeTraits<SHORT>::SizeOf();
    }
    case USHORT: {
    	return PixTypeTraits<USHORT>::SizeOf();
    }
    case INT: {
        return PixTypeTraits<INT>::SizeOf();
    }
    case FLOAT: {
        return PixTypeTraits<FLOAT>::SizeOf();
    }
    case HALF: {
        return PixTypeTraits<HALF>::SizeOf();
    }
    case FLOAT3: {
        return PixTypeTraits<FLOAT3>::SizeOf();
    }
    case UCHAR4: {
        return PixTypeTraits<UCHAR4>::SizeOf();
    }
    case SHORT4: {
        return PixTypeTraits<SHORT4>::SizeOf();
    }
    case FLOAT4: {
        return PixTypeTraits<FLOAT4>::SizeOf();
    }
    }

    return 0;
}

inline NV_STATUS CreatePitch2D(void** ptr, size_t* pitch, int width, int height,
                               PixType type) {
    cudaError_t error;
    error = cudaMallocPitch(ptr, pitch, width * GetSizeOf(type), height);

    if (error != cudaSuccess) {
        NVLOG(NVLOG_ERROR) << __FUNCTION__
                << ": unable to allocate a pitch2D with width=" << width
                << ", height=" << height;
        return NV_FAILED;
    }
    return NV_SUCCESS;
}

inline NV_STATUS CreateCudaArray(cudaArray_t* array,
                                 const cudaChannelFormatDesc* channel_desc,
                                 int width, int height) {
    cudaError_t error;
    error = cudaMallocArray(array, channel_desc, width, height,
            cudaArraySurfaceLoadStore);

    if (error != cudaSuccess) {
        NVLOG(NVLOG_ERROR) << __FUNCTION__
                << ": unable to allocate a cuda array with width=" << width
                << ", height=" << height;
        return NV_FAILED;
    }
    return NV_SUCCESS;
}

inline NV_STATUS BindTexture(cudaTextureObject_t* tex_obj, void* ptr,
                             size_t pitch, int width, int height, PixType type,
                             const cudaChannelFormatDesc& channel_desc,
                             TexReadMode read_mode, TexFilterMode filter_mode) {
    cudaError_t error;

    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypePitch2D;
    res_desc.res.pitch2D.devPtr = ptr;
    res_desc.res.pitch2D.desc = channel_desc;
    res_desc.res.pitch2D.height = height;
    res_desc.res.pitch2D.width = width;// * GetSizeOf(type);
    res_desc.res.pitch2D.pitchInBytes = pitch;

    struct cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    if(filter_mode == FL_LINEAR)
        tex_desc.filterMode = cudaFilterModeLinear;
    else
        tex_desc.filterMode = cudaFilterModePoint;
    if (read_mode == RD_NORMALIZED_FLOAT)
        tex_desc.readMode = cudaReadModeNormalizedFloat;
    else
        tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 0;

    error = cudaCreateTextureObject(tex_obj, &res_desc, &tex_desc, NULL);

    if (error != cudaSuccess) {
        NVLOG(NVLOG_ERROR) << __FUNCTION__
                << ": unable to bind texture, error = " << cudaGetErrorString(error);
        return NV_FAILED;
    }

    return NV_SUCCESS;
}

inline NV_STATUS BindTexture(cudaTextureObject_t* tex_obj,
                             const cudaArray_t array, TexReadMode read_mode,
                             TexFilterMode filter_mode ) {
    cudaError_t error;

    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array;

    struct cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeMirror;
    tex_desc.addressMode[1] = cudaAddressModeMirror;
    if(filter_mode == FL_POINT)
        tex_desc.filterMode = cudaFilterModePoint;
    else
        tex_desc.filterMode = cudaFilterModeLinear;
    if (read_mode == RD_NORMALIZED_FLOAT)
        tex_desc.readMode = cudaReadModeNormalizedFloat;
    else
        tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 0;

    error = cudaCreateTextureObject(tex_obj, &res_desc, &tex_desc, NULL);

    if (error != cudaSuccess) {
        NVLOG(NVLOG_ERROR) << __FUNCTION__
                << ": unable to bind texture, error = " << error;
        return NV_FAILED;
    }

    return NV_SUCCESS;
}

inline NV_STATUS BindSurface(cudaSurfaceObject_t* surf_obj,
                             const cudaArray_t array) {
    cudaError_t error;

    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array;

    error = cudaCreateSurfaceObject(surf_obj, &res_desc);

    if (error != cudaSuccess) {
        NVLOG(NVLOG_ERROR) << __FUNCTION__ << ": unable to bind surface";
        return NV_FAILED;
    }

    return NV_SUCCESS;
}

FrameBuffer::FrameBuffer(int width, int height, PixType type,
                         MemoryFormat format, TexReadMode read_mode,
                         TexFilterMode filter_mode) :
        width_(width), height_(height), type_(type), format_(format),
                cu_array_(0), ptr_(0) {
    NV_STATUS status;
    // 1. create cuda channel desc
    cu_channel_desc_ = CreateChannelDesc(type);

    if (format == BLOCK_LINEAR) {
        // 2. create cuda array
        status = CreateCudaArray(&cu_array_, &cu_channel_desc_, width_,
                height_);
        assert(status == NV_SUCCESS);
        // 3. map to texture
        status = BindTexture(&cu_tex_obj_, cu_array_, read_mode, filter_mode);
        assert(status == NV_SUCCESS);
        // 4. map to surface
        status = BindSurface(&cu_surf_obj_, cu_array_);
        assert(status == NV_SUCCESS);
    } else {
        // 2. create pitch linear
        status = CreatePitch2D(reinterpret_cast<void**>(&ptr_), &pitch_, width_,
                height_, type);
        assert(status == NV_SUCCESS);
        // 3. map to texture
        status = BindTexture(&cu_tex_obj_, ptr_, pitch_, width_, height_, type_,
                cu_channel_desc_, read_mode, filter_mode);
        assert(status == NV_SUCCESS);
    }
}

FrameBuffer::~FrameBuffer() {

    cudaError_t error;

    // 1. destroy texture object
    error = cudaDestroyTextureObject(cu_tex_obj_);
    assert(error == cudaSuccess);

    if (format_ == BLOCK_LINEAR) {
        // 2. destroy surface object
        error = cudaDestroySurfaceObject(cu_surf_obj_);
        assert(error == cudaSuccess);

        // 3. destroy cuda array
        error = cudaFreeArray(cu_array_);
        assert(error == cudaSuccess);
    } else {
        error = cudaFree(ptr_);
        assert(error == cudaSuccess);
    }
}

NV_STATUS FrameBuffer::Upload(const void* buf, size_t pitch, int x_offset,
                              int y_offset, unsigned int width,
                              unsigned int height) {

    cudaError_t error;
    if (format_ == BLOCK_LINEAR) {
        error = cudaMemcpy2DToArray(cu_array_, x_offset, y_offset, buf, pitch,
                width, height, cudaMemcpyHostToDevice);
    } else {
        error = cudaMemcpy2D(ptr_, pitch_, buf, pitch, width, height,
                cudaMemcpyHostToDevice);
    }

    if (error != cudaSuccess) {
        NVLOG(NVLOG_ERROR) << __FUNCTION__ << ": unable to upload data.";
        return NV_FAILED;
    }

    return NV_SUCCESS;
}

NV_STATUS FrameBuffer::Download(void* buf, size_t pitch, int x_offset,
                                int y_offset, unsigned int width,
                                unsigned int height) const {
    cudaError_t error;
    if (format_ == BLOCK_LINEAR) {
        error = cudaMemcpy2DFromArray(buf, pitch, cu_array_, x_offset, y_offset,
                width, height, cudaMemcpyDeviceToHost);
    } else {
        error = cudaMemcpy2D(buf, pitch, ptr_, pitch_, width, height,
                cudaMemcpyDeviceToHost);
    }

    if (error != cudaSuccess) {
        NVLOG(NVLOG_ERROR) << __FUNCTION__ << ": unable to download data.";
        return NV_FAILED;
    }

    return NV_SUCCESS;
}
NV_STATUS FrameBuffer::CopyTo(FrameBuffer* fb) const {
    cudaError_t error;

    if (format_ == BLOCK_LINEAR) {
        int element_bytes = (cu_channel_desc_.x + cu_channel_desc_.y
                + cu_channel_desc_.z + cu_channel_desc_.w) / 8;
        error = cudaMemcpyArrayToArray(fb->cu_array_, 0, 0, cu_array_, 0, 0,
                element_bytes * width() * height());
    } else {
        error = cudaMemcpy2D(fb->ptr_, fb->pitch_, ptr_, pitch_,
                width_ * GetSizeOf(type_), height_, cudaMemcpyDeviceToDevice);
    }

    if (error != cudaSuccess) {
        NVLOG(NVLOG_ERROR) << __FUNCTION__ << ": unable to copy data.";
        return NV_FAILED;
    }

    return NV_SUCCESS;
}
} /* namespace internal */
} /* namespace nv */
