/*
 * FrameBufferImpl.h
 *
 *  Created on: Nov 26, 2013
 *      Author: ytsai
 */

#ifndef FRAMEBUFFERIMPL_H_
#define FRAMEBUFFERIMPL_H_

namespace nv {
namespace internal {
void CopyToRegion(nv::FrameBufferHandle& out, const nv::FrameBufferHandle& in, int widthOffset, int heightOffset,
                  int nWidth, int nHeight);
} // namesapce internal
} // namespace nv


#endif /* FRAMEBUFFERIMPL_H_ */
