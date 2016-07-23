/*
 * FrameBufferHandle.h
 *
 *  Created on: Jan 31, 2013
 *      Author: ytsai
 */

#ifndef FRAMEBUFFERHANDLE_H_
#define FRAMEBUFFERHANDLE_H_

#include <Core/Types.h>
#include <cuda_runtime.h>

namespace nv {

namespace internal {
class FrameBuffer;
}

struct Size {
	Size() : width(0), height(0) {}
	Size(int w, int h) : width(w) , height(h) { }

	inline Size operator /(const unsigned int& scalar) {
		return Size(width / scalar, height / scalar);
	}

	inline bool operator ==(const Size& rhs) {
		return IsEqual(rhs);
	}

	inline bool operator !=(const Size& rhs) {
		return !IsEqual(rhs);
	}

	inline bool IsEqual(const Size& rhs) {
		return ((rhs.width == this->width) &&
				(rhs.height == this->height));
	}

	unsigned int width;
	unsigned int height;
};

class FrameBufferHandle {
public:
	FrameBufferHandle();
	FrameBufferHandle(int id, internal::FrameBuffer* fb);
	FrameBufferHandle(const FrameBufferHandle& handle);
	const FrameBufferHandle& operator= (const FrameBufferHandle& handle);

	// it has the same interface as FrameBuffer.
	NV_STATUS Upload(const void* buf, size_t pitch, int x_offset, int y_offset,
					 unsigned int width, unsigned int height);
	NV_STATUS Download(void* buf, size_t pitch, int x_offset, int y_offset,
					   unsigned int width, unsigned int height) const;
	NV_STATUS CopyTo(FrameBufferHandle& handle) const;
	NV_STATUS CopyTo(FrameBufferHandle& handle, int widthOffset, int heightOffset, int nWidth, int nHeight) const;

	bool IsValid() const { return (fb_ != NULL); }
	unsigned int Width() const;
	unsigned int Height() const;
	nv::Size Size() const;
	PixType Type() const;
	MemoryFormat Format() const;

	cudaTextureObject_t Texture() const;
	cudaSurfaceObject_t Surface() const;

	void* Ptr();
	const void* Ptr() const;
	size_t Pitch() const;

	cudaArray* cu_array();

	int id() const {
		return id_;
	}

private:
	int id_;
	internal::FrameBuffer* fb_;

	friend class FrameBufferManager;
};

} /* namespace nv */
#endif /* FRAMEBUFFERHANDLE_H_ */
