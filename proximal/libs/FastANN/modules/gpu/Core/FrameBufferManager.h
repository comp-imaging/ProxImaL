/*
 * FrameBufferManager.h
 *
 *  Created on: Jan 31, 2013
 *      Author: ytsai
 */

#ifndef FRAMEBUFFERMANAGER_H_
#define FRAMEBUFFERMANAGER_H_

#include <Core/Types.h>
#include <Core/FrameBufferHandle.h>
#include <map>
namespace nv {
namespace internal {
	class FrameBuffer;
} /* namespace internal */
class FrameBufferManager {
	typedef std::map<int, internal::FrameBuffer*> FrameBufferMap;
public:
	FrameBufferManager();
	~FrameBufferManager();

	// common manager interface
	static FrameBufferManager& GetInstance();
	NV_STATUS ShutDown();

	// specific operations
	FrameBufferHandle Create(int width, int height, PixType type, MemoryFormat format = BLOCK_LINEAR, TexReadMode read_mode = RD_NORMALIZED_FLOAT, TexFilterMode filter_mode = FL_POINT);
	NV_STATUS Destroy(FrameBufferHandle& handle);

private:
	FrameBufferManager(const FrameBufferManager&);
	FrameBufferManager& operator =(const FrameBufferManager&);

private:
	FrameBufferMap map_;
};

} /* namespace nv */
#endif /* FRAMEBUFFERMANAGER_H_ */
