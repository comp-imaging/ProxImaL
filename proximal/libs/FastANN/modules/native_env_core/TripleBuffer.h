/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _TRIPLEBUFFER_H
#define _TRIPLEBUFFER_H

/**
 * @file
 *
 * Definition of TripleBuffer
 */

#include <pthread.h>

namespace System
{

/**
 * Provides a thread-safe implementation of triple buffering mechanism.
 * We use triple buffering to minimize the synchronization between Java
 * UI and image capture code. It allows us to capture images and perform UI
 * refresh always at their full speeds (30hz camera, ~60hz UI).
 */

template<class T> class TripleBuffer
{
public:
    /**
     * Default constructor.
     */
    template<typename U> TripleBuffer(U const &front, U const &back, U const &spare)
            : mFrontBuffer(front), mBackBuffer(back), mSpareBuffer(spare)
    {
        pthread_mutex_init(&mSwapWaitLock, 0);
        pthread_cond_init(&mSwapSignal, 0);

        mFinalized = false;

        mBufferPtr[0] = &mFrontBuffer;
        mBufferPtr[1] = &mBackBuffer;
        mBufferPtr[2] = &mSpareBuffer;

        mSpareBufferOrigin = 2;
    }

    /**
     * Default destructor.
     */
    ~TripleBuffer(void)
    {
        finalize();

        // XXX: dirty hack
        pthread_mutex_lock(&mSwapWaitLock);
        pthread_mutex_unlock(&mSwapWaitLock);

        pthread_mutex_destroy(&mSwapWaitLock);
        pthread_cond_destroy(&mSwapSignal);
    }

    /*
     * Cancel all pending consume requests, flush queue,
     * and stop taking new tasks.
     */
    void finalize()
    {
        if (mFinalized)
        {
            return;
        }

        pthread_mutex_lock(&mSwapWaitLock);
        mFinalized = true;
        pthread_cond_broadcast(&mSwapSignal);
        pthread_mutex_unlock(&mSwapWaitLock);
    }

    /**
     * Swaps front buffer with an empty buffer.
     * @return pointer to the active front buffer
     */
    T const &swapFrontBuffer(void)
    {
        return swapBuffer(0);
    }

    /**
     * Gets a pointer to active front buffer. The pointer is valid
     * till next call to swapFrontBuffer().
     * @return pointer to the front buffer
     */
    T const &getFrontBuffer(void)
    {
        return getBuffer(0);
    }

    T const &trySwapAndGetFrontBuffer(bool blocking)
    {
        return trySwapAndGetBuffer(0, blocking);
    }

    /**
     * Swaps back buffer with an empty buffer.
     * @return pointer to the active back buffer
     */
    T const &swapBackBuffer(void)
    {
        return swapBuffer(1);
    }

    /**
     * Gets a pointer to active back buffer. The pointer is valid
     * till next call to swapBackBuffer().
     * @return pointer to the back buffer
     */
    T const &getBackBuffer(void)
    {
        return getBuffer(1);
    }

    T const &trySwapAndGetBackBuffer(bool blocking)
    {
        return trySwapAndGetBuffer(1, blocking);
    }

private:
    T const &getBuffer(int bufferIndex)
    {
        if (mFinalized)
        {
            return *mBufferPtr[bufferIndex];
        }

        pthread_mutex_lock(&mSwapWaitLock);
        T const *rval = mBufferPtr[bufferIndex];
        pthread_mutex_unlock(&mSwapWaitLock);

        return *rval;
    }

    T const &trySwapAndGetBuffer(int bufferIndex, bool blocking)
    {
        if (mFinalized)
        {
            return *mBufferPtr[bufferIndex];
        }
        pthread_mutex_lock(&mSwapWaitLock);
        if (blocking)
        {
            while (mSpareBufferOrigin != (1 - bufferIndex))
            {
                if (mFinalized)
                {
                    T const *rval = mBufferPtr[bufferIndex];
                    pthread_mutex_unlock(&mSwapWaitLock);
                    return *rval;
                }
                pthread_cond_wait(&mSwapSignal, &mSwapWaitLock);
            }
            NVR::xchg(mBufferPtr[bufferIndex], mBufferPtr[2]);
            mSpareBufferOrigin = bufferIndex;
            pthread_cond_broadcast(&mSwapSignal);
        }
        else
        {
            if (!mFinalized && mSpareBufferOrigin == (1 - bufferIndex))
            {
                NVR::xchg(mBufferPtr[bufferIndex], mBufferPtr[2]);
                mSpareBufferOrigin = bufferIndex;
                pthread_cond_broadcast(&mSwapSignal);
            }
        }

        T const *rval = mBufferPtr[bufferIndex];
        pthread_mutex_unlock(&mSwapWaitLock);

        return *rval;
    }

    T const &swapBuffer(int bufferIndex)
    {
        if (mFinalized)
        {
            return *mBufferPtr[bufferIndex];
        }

        pthread_mutex_lock(&mSwapWaitLock);

        if (!mFinalized)
        {
            NVR::xchg(mBufferPtr[bufferIndex], mBufferPtr[2]);
            mSpareBufferOrigin = bufferIndex;
            pthread_cond_broadcast(&mSwapSignal);
        }

        T const *rval = mBufferPtr[bufferIndex];
        pthread_mutex_unlock(&mSwapWaitLock);

        return *rval;
    }

    T const mFrontBuffer;
    T const mBackBuffer;
    T const mSpareBuffer;
    int mSpareBufferOrigin; // index of the buffer that spare buffer is from (most recently)

    T const *mBufferPtr[3];

    pthread_mutex_t mSwapWaitLock;
    pthread_cond_t mSwapSignal;
    bool mFinalized;
};

}

#endif

