/*
 * FrameBufferHandle.cpp
 *
 *  Created on: Jan 31, 2013
 *      Author: ytsai
 */

#include "FrameBufferHandle.h"
#include <Core/Internal/FrameBuffer.h>
#include <Core/Internal/FrameBufferImpl.h>

namespace nv
{
FrameBufferHandle::FrameBufferHandle()
        : id_(0), fb_(0)
{
}

FrameBufferHandle::FrameBufferHandle(int id, internal::FrameBuffer* fb)
        : id_(id), fb_(fb)
{

}

FrameBufferHandle::FrameBufferHandle(const FrameBufferHandle& handle)
{
    id_ = handle.id_;
    fb_ = handle.fb_;
}

const FrameBufferHandle& FrameBufferHandle::operator=(const FrameBufferHandle& handle)
{
    id_ = handle.id_;
    fb_ = handle.fb_;
    return *this;
}

NV_STATUS FrameBufferHandle::Upload(const void* buf, size_t pitch, int x_offset, int y_offset, unsigned int width,
                                    unsigned int height)
{
    return fb_->Upload(buf, pitch, x_offset, y_offset, width, height);
}

NV_STATUS FrameBufferHandle::Download(void* buf, size_t pitch, int x_offset, int y_offset, unsigned int width,
                                      unsigned int height) const
{
    return fb_->Download(buf, pitch, x_offset, y_offset, width, height);
}

NV_STATUS FrameBufferHandle::CopyTo(FrameBufferHandle& handle) const
{
    if (handle.Size() != this->Size())
        return NV_FAILED;
    if (handle.Type() != this->Type())
        return NV_FAILED;

    return fb_->CopyTo(handle.fb_);
}

NV_STATUS FrameBufferHandle::CopyTo(FrameBufferHandle& handle, int widthOffset, int heightOffset, int nWidth, int nHeight) const
{
    nv::internal::CopyToRegion(handle, *this, widthOffset, heightOffset, nWidth, nHeight);
    return NV_SUCCESS;
}

unsigned int FrameBufferHandle::Width() const
{
    return fb_->width();
}

unsigned int FrameBufferHandle::Height() const
{
    return fb_->height();
}

nv::Size FrameBufferHandle::Size() const
{
    return nv::Size(fb_->width(), fb_->height());
}

PixType FrameBufferHandle::Type() const
{
    return fb_->type();
}

MemoryFormat FrameBufferHandle::Format() const
{
	return fb_->format();
}

cudaTextureObject_t FrameBufferHandle::Texture() const
{
    return fb_->cu_tex_obj();
}

cudaSurfaceObject_t FrameBufferHandle::Surface() const
{
    return fb_->cu_surf_obj();
}

cudaArray* FrameBufferHandle::cu_array()
{
    return fb_->cu_array();
}

void* FrameBufferHandle::Ptr()
{
    return fb_->ptr();
}
const void* FrameBufferHandle::Ptr() const
{
    return fb_->ptr();
}

size_t FrameBufferHandle::Pitch() const
{
    return fb_->pitch();
}

}
/* namespace nv */
