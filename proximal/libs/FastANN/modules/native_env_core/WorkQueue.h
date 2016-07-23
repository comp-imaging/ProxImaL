/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _WORKQUEUE_H
#define _WORKQUEUE_H

/**
 * @file
 * Definition of WorkQueue.
 */

#include <pthread.h>
#include <queue>

namespace System
{

/**
 * This class provides a thread-safe implementation of a work queue (FIFO).
 */
template<class T> class WorkQueue
{
public:
    /**
     * Default constructor.
     */
    WorkQueue(void)
            : mFinalized(false), mDataQueue(), mAccessLock(), mEnqueueSignal()
    {
        // TODO: consider failure cases
        pthread_mutex_init(&mAccessLock, 0);
        pthread_cond_init(&mEnqueueSignal, 0);
    }

    /**
     * Default destructor.
     */
    ~WorkQueue(void)
    {
        finalize();

        // XXX: dirty hack
        // we want to make sure that we are executing the destructor
        // code *after* all threads calling produce()/consume()
        // have become aware of finalization
        pthread_mutex_lock(&mAccessLock);
        pthread_mutex_unlock(&mAccessLock);

        pthread_cond_destroy(&mEnqueueSignal);
        pthread_mutex_destroy(&mAccessLock);
    }

    /**
     * Cancel all pending consume requests, flush queue,
     * and stop taking new tasks.
     */
    void finalize(void)
    {
        if (mFinalized)
        {
            return;
        }

        pthread_mutex_lock(&mAccessLock);
        mFinalized = true;
        mDataQueue = std::queue<T>();
        pthread_cond_broadcast(&mEnqueueSignal);
        pthread_mutex_unlock(&mAccessLock);
    }

    /**
     * Resets the queue to its initial state. Calling this function right after
     * a finalize() call might prevent some threads from exiting the wait loop
     * in consume() function.
     */
    void reset(void)
    {
        pthread_mutex_lock(&mAccessLock);
        mFinalized = false;
        mDataQueue = std::queue<T>();
        pthread_mutex_unlock(&mAccessLock);
    }

    /**
     * Returns true is queue is in finalized state, false otherwise.
     */
    bool isFinalized(void)
    {
        return mFinalized;
    }

    /**
     * Adds a new element to the end of the queue.
     * @param elem value to be copied to the queue
     */
    void produce(T const &elem)
    {
        // TODO: add std::forward() functionality
        // TODO: add emplace() like functionality
        if (mFinalized)
        {
            return;
        }

        pthread_mutex_lock(&mAccessLock);
        if (!mFinalized)
        {
            bool const wasEmpty = mDataQueue.empty();
            mDataQueue.push(elem);
            if (wasEmpty)
            {
                // if the queue was empty let consumers know
                // TODO: investigate why pthread_cond_signal() is slower on Tegra
                pthread_cond_broadcast(&mEnqueueSignal);
            }
        }
        pthread_mutex_unlock(&mAccessLock);
    }

    /**
     * Adds a vector of new elements to the end of the queue.
     * @param elem vector of values to be copied to the queue
     */
    void produce(std::vector<T> const &array)
    {
        // TODO: add std::forward() functionality
        // TODO: add emplace() like functionality
        int const arraySize = array.size();
        if (mFinalized || arraySize == 0)
        {
            return;
        }

        pthread_mutex_lock(&mAccessLock);
        if (!mFinalized)
        {
            bool const wasEmpty = mDataQueue.empty();

            for (int i = 0; i < arraySize; i++)
            {
                mDataQueue.push(array[i]);
            }

            if (wasEmpty)
            {
                // if the queue was empty let consumers know
                pthread_cond_broadcast(&mEnqueueSignal);
            }
        }
        pthread_mutex_unlock(&mAccessLock);
    }

    /**
     * Removes an element from the queue. The consumed element is copied to the object passed
     * in the function call. If the queue is empty and blocking is set to true (default) then
     * the function blocks the execution till it can fill in the output. If blocking is set
     * to false, the function always returns immediately.
     * @param elem a location where the object from top of the queue should be put
     * @param blocking determines whether the consume call should block until it is able
     * to fill in the output object
     * @return false if failed to get element from queue (empty queue for non-blocking calls, or
     * queue has been finalized)
     */
    bool consume(T &elem, bool blocking = true)
    {
        if (mFinalized)
        {
            return false;
        }

        pthread_mutex_lock(&mAccessLock);
        while (mDataQueue.empty())
        {
            if (!blocking || mFinalized)
            {
                pthread_mutex_unlock(&mAccessLock);
                return false;
            }
            pthread_cond_wait(&mEnqueueSignal, &mAccessLock);
        }

        elem = mDataQueue.front();
        mDataQueue.pop();
        pthread_mutex_unlock(&mAccessLock);

        return true;
    }

    /**
     * Copies the contents of this queue to an instance of std::queue.
     * The function is non-blocking.
     */
    void consumeAll(std::queue<T> &queue)
    {
        if (mFinalized)
        {
            return;
        }

        pthread_mutex_lock(&mAccessLock);
        queue = mDataQueue;
        mDataQueue = std::queue<T>();
        pthread_mutex_unlock(&mAccessLock);
    }

    /**
     * Returns the number of elements in this work queue.
     * @return number of elements in this work queue
     */
    int size(void)
    {
        return mDataQueue.size();
    }

private:
    // prevent copy construction and assignment
    WorkQueue(WorkQueue const &instance);
    WorkQueue &operator=(WorkQueue const &instance);

    bool mFinalized;
    std::queue<T> mDataQueue;
    pthread_mutex_t mAccessLock;
    pthread_cond_t mEnqueueSignal;
};

}

#endif

