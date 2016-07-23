/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef MEMALLOC_H_
#define MEMALLOC_H_

#include "Base.h"
#include "ManagedPtr.h"

namespace System
{

// initial block capacity for fixed size block allocator
#define BLOCK_ALLOCATOR_MIN_CAPACITY        16

class BlockAllocator: public ManagedAbstractObject
{
    typedef struct BlockHeaderStruct
    {
        struct BlockHeaderStruct *next;
    } BlockHeader;
public:
    BlockAllocator(uint blockSize, uint initialCapacity = BLOCK_ALLOCATOR_MIN_CAPACITY);
    ~BlockAllocator(void);

    uchar *allocate(void)
    {
        // can we reuse already allocated block?
        if (mFreeRoot != 0)
        {
            BlockHeader *rval = mFreeRoot;
            mFreeRoot = mFreeRoot->next;
            return reinterpret_cast<uchar *>(rval);
        }

        return allocateNewBlock();
    }

    void deallocate(uchar *ptr)
    {
        BlockHeader *header = unsafe_pointer_cast<BlockHeader>(ptr);
        header->next = mFreeRoot;
        mFreeRoot = header;
    }

    int getCapacity(void) const
    {
        return ((1 << mSegmentCount) - 1) * mFirstSegmentCapacity;
    }

    void deallocateAll(void);

private:
    // prevent copy construction and assignment
    BlockAllocator(BlockAllocator const &instance);
    BlockAllocator &operator=(BlockAllocator const &instance);

    uchar *allocateNewBlock(void);

    const int mFirstSegmentCapacity;
    const int mBlockSize;

    uchar **mBlockData;
    int mSegmentPointerArraySize;
    int mSegmentCount;
    int mCurrentSegmentIndex;
    int mCurrentSegmentCapacity;
    int mCurrentBlockIndex;

    BlockHeader *mFreeRoot;
};

template<typename T> class TypedBlockAllocator: protected BlockAllocator
{
public:
    TypedBlockAllocator(uint initialCapacity = BLOCK_ALLOCATOR_MIN_CAPACITY)
            : BlockAllocator(sizeof(T), initialCapacity)
    {
    }

    T *allocate(void)
    {
        // allocate
        T *instance = reinterpret_cast<T *>(BlockAllocator::allocate());
        // construct instance
        new (static_cast<void *>(instance)) T();
        return instance;
    }

    void deallocate(T *ptr)
    {
        // destroy instance
        ptr->~T();
        // deallocate
        BlockAllocator::deallocate(reinterpret_cast<uchar *>(ptr));
    }

    int getCapacity(void) const
    {
        return BlockAllocator::getCapacity();
    }

    void deallocateAll(void)
    {
        BlockAllocator::deallocateAll();
    }

private:
    // prevent copy construction and assignment
    TypedBlockAllocator(TypedBlockAllocator const &instance);
    TypedBlockAllocator &operator=(TypedBlockAllocator const &instance);
};

}

#endif /* MEMALLOC_H_ */
