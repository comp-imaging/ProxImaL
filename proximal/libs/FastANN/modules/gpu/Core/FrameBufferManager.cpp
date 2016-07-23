/*
 * FrameBufferManager.cpp
 *
 *  Created on: Jan 31, 2013
 *      Author: ytsai
 */

#include "FrameBufferManager.h"
#include <Core/Internal/FrameBuffer.h>
#include <Log/Log.h>
#include <assert.h>

static int idx_counter = 0;
namespace nv {

FrameBufferManager::FrameBufferManager() {

}

FrameBufferManager::~FrameBufferManager() {
	ShutDown();
}
FrameBufferManager& FrameBufferManager::GetInstance() {
	static FrameBufferManager instance;
	return instance;
}

NV_STATUS FrameBufferManager::ShutDown() {
	//NVLOG(NVLOG_INFO) << __FUNCTION__ << ": shutting down framebuffer manager.";
	FrameBufferMap::iterator it = map_.begin();
	while(it != map_.end()) {
		delete it->second;
		it++;
	}

	map_.clear();
	return NV_SUCCESS;
}

FrameBufferHandle FrameBufferManager::Create(int width, int height,
		PixType type, MemoryFormat format, TexReadMode read_mode, TexFilterMode filter_mode) {

	int id = idx_counter;
	internal::FrameBuffer* fb = new internal::FrameBuffer(width, height, type, format, read_mode, filter_mode);
	map_[id] = fb;

	idx_counter++;
	return FrameBufferHandle(id, fb);
}

NV_STATUS FrameBufferManager::Destroy(FrameBufferHandle& handle) {
	if(!map_.count(handle.id_)) {
		NVLOG(NVLOG_ERROR) << __FUNCTION__ << ": unable to find framebuffer handle in the record.";
		return NV_FAILED;
	}

	// FIXME: this may not be thread-safe.
	delete handle.fb_;
	map_.erase(handle.id_);
	handle.fb_ = 0;
	handle.id_ = -1;

	return NV_SUCCESS;
}

} /* namespace nv */
