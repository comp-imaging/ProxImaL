/*
 * SharedMemory.h
 *
 *  Created on: Mar 7, 2013
 *      Author: ytsai
 */

#ifndef SHAREDMEMORY_H_
#define SHAREDMEMORY_H_

namespace nv {
template<typename T>
struct shared_memory
{
    __device__ inline operator T*() {
        extern __shared__ unsigned char __smem[];
        return (T*) __smem;
    }

    __device__ inline operator const T*() const {
        extern __shared__ unsigned char __smem[];
        return (T*) __smem;
    }
};
} // namespace nv

#endif /* SHAREDMEMORY_H_ */
