/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include "SystemCore.h"
#include "MemAlloc.h"

// default ManagedAbstractObject destructor
ManagedAbstractObject::~ManagedAbstractObject(void)
{
}

// initial number of memory segment slots for fixed size block allocator
#define FSBA_INITIAL_SEGMENT_NUM 4

System::BlockAllocator::BlockAllocator(uint size, uint capacity)
        : mFirstSegmentCapacity(capacity < 1 ? 1 : capacity),
          mBlockSize(size > sizeof(BlockHeader) ? size : sizeof(BlockHeader)),
          mBlockData(MemoryAlloc<uchar *>(FSBA_INITIAL_SEGMENT_NUM)),
          mSegmentPointerArraySize(FSBA_INITIAL_SEGMENT_NUM), mSegmentCount(1), mCurrentSegmentIndex(0),
          mCurrentSegmentCapacity(mFirstSegmentCapacity * mBlockSize), mCurrentBlockIndex(0), mFreeRoot(0)
{
    mBlockData[0] = MemoryAlloc<uchar>(mCurrentSegmentCapacity, CACHELINE_ALIGNMENT);
}

System::BlockAllocator::~BlockAllocator(void)
{
    for (int i = 0; i < mSegmentCount; i++)
    {
        MemoryFree(mBlockData[i]);
    }

    MemoryFree(mBlockData);
}

void System::BlockAllocator::deallocateAll(void)
{
    mFreeRoot = 0;
    mCurrentBlockIndex = 0;
    mCurrentSegmentIndex = 0;
    mCurrentSegmentCapacity = mFirstSegmentCapacity * mBlockSize;
}

uchar *System::BlockAllocator::allocateNewBlock(void)
{
    // try to get new block
    if (mCurrentBlockIndex == mCurrentSegmentCapacity)
    {
        // segment full
        if (mCurrentSegmentIndex + 1 == mSegmentCount)
        {
            // out of memory, need new segment
            if (mSegmentCount == mSegmentPointerArraySize)
            {
                // segment array too small, reallocate
                uchar **newBlockData = MemoryAlloc<uchar *>(mSegmentPointerArraySize << 1);
                MemoryCopy(newBlockData, mBlockData, sizeof(uchar *) * mSegmentPointerArraySize);
                MemoryFree(mBlockData);
                mBlockData = newBlockData;
                mSegmentPointerArraySize <<= 1;
            }

            mBlockData[mSegmentCount] = MemoryAlloc<uchar>(mCurrentSegmentCapacity << 1, CACHELINE_ALIGNMENT);
            mSegmentCount++;
        }

        mCurrentBlockIndex = 0;
        mCurrentSegmentIndex++;
        mCurrentSegmentCapacity <<= 1;
    }

    uchar *ptr = mBlockData[mCurrentSegmentIndex] + mCurrentBlockIndex;
    mCurrentBlockIndex += mBlockSize;

    return ptr;
}
