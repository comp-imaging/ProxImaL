/*
 * FrameBuffer.h
 *
 *  Created on: Feb 1, 2013
 *      Author: ytsai
 */

#ifndef FRAMEBUFFER_H_
#define FRAMEBUFFER_H_

#include <Core/Types.h>
#include <cuda_runtime.h>

namespace nv {
namespace internal {
class FrameBuffer {
public:
	FrameBuffer(int width, int height, PixType type, MemoryFormat format, TexReadMode read_mode, TexFilterMode filter_mode = FL_POINT);
	virtual ~FrameBuffer();

	// data transfer
	NV_STATUS Upload(const void* buf, size_t pitch, int x_offset, int y_offset,
			unsigned int width, unsigned int height);
	NV_STATUS Download(void* buf, size_t pitch, int x_offset, int y_offset,
			unsigned int width, unsigned int height) const;
	NV_STATUS CopyTo(FrameBuffer* fb) const;

	// getter
	unsigned int width() const { return width_; }
	unsigned int height() const { return height_; }
	PixType type() const { return type_; }
	MemoryFormat format() const { return format_; }

	cudaArray* cu_array() { return cu_array_; }
	const cudaArray* cu_array() const { return cu_array_; }

	void* ptr() { return reinterpret_cast<void*>(ptr_); }
	const void* ptr() const { return reinterpret_cast<const void*>(ptr_); }
	size_t pitch() const { return pitch_; }

	cudaTextureObject_t cu_tex_obj() const { return cu_tex_obj_; }
	cudaSurfaceObject_t cu_surf_obj() const { return cu_surf_obj_; }


private:
	unsigned int width_;
	unsigned int height_;
	PixType type_;
	MemoryFormat format_;

	cudaArray* cu_array_;

	unsigned char* ptr_;
	size_t pitch_;

	cudaChannelFormatDesc cu_channel_desc_;
	cudaTextureObject_t cu_tex_obj_;
	cudaSurfaceObject_t cu_surf_obj_;
};

} /* internal */
} /* namespace nv */
#endif /* FRAMEBUFFER_H_ */
