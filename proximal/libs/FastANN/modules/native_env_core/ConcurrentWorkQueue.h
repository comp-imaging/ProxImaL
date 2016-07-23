/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _CONCURRENT_WORK_QUEUE_H_
#define _CONCURRENT_WORK_QUEUE_H_

#ifdef __GXX_EXPERIMENTAL_CXX0X__
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif
#include "WorkQueue.h"
#include "Base.h"

namespace System {

/*
 * A class that represents a single job. It may be split into
 * multiple tasks.
 */
class TaskExecutor
{
public:
    virtual ~TaskExecutor(void)
    {
    }

    // Perform task #index out of #total. index is zero-based.
    virtual void run(uint const taskIndex, uint const totalTasks) = 0;

    // Returns the maximum number of tasks supported by the job.
    // Note that runTask may be called with any value of total
    // less than this number, depending on pool size.
    virtual uint getMaximumTaskCount() const
    {
        return 1;
    }
};

/*
 * A utility class that represents a thread pool.
 * Supports a number of blocking waits, synchronization, etc.
 */
class ConcurrentWorkQueue
{
public:
    ConcurrentWorkQueue(const uint poolSize);

    ~ConcurrentWorkQueue(void);

    /*
     * Waits for all jobs to finish, and terminates the worker threads.
     * Future calls to enqueue will fail.
     */
    void finalize(void);

    /**
     * Returns true is queue is in finalized state, false otherwise.
     */
    bool isFinalized(void) const
    {
        return mFinalized;
    }

    /*
     * Returns the size of the thread pool.
     */
    uint getSize() const
    {
        return mThreadPoolSize;
    }

    /*
     * Enqueues the given job. Tasks in jobs are scheduled
     * sequentially. e.g. it is guaranteed that
     * task #x is scheduled before task #y if x<y. Also it is
     * guaranteed that all tasks of a job enqueued before
     * another job are scheduled before the tasks of the latter job.
     * Returns nonnegative job ID if successful. Use this to wait for
     * a particular job.
     */
    // TODO: implement clone() functionality
    uint enqueue(TaskExecutor *job);

    /*
     * Blocks until all tasks in the given job
     * are completed. Will return immediately if job is not
     * yet enqueued.
     */
    void waitForJob(uint jobId);

    /*
     * Blocks until all tasks in all enqueued jobs are
     * completed.
     */
    void waitForAllJobs(void);

private:
    // prevent copy construction and assignment
    ConcurrentWorkQueue(ConcurrentWorkQueue const &instance);
    ConcurrentWorkQueue &operator=(ConcurrentWorkQueue const &instance);

    class PendingJob
    {
    public:
        PendingJob(uint jobId, uint totalTaskCount, TaskExecutor *job)
                : mJobId(jobId), mRemainingTaskCount(totalTaskCount), mTotalTaskCount(totalTaskCount), mExecutor(job)
        {
        }

        uint getJobId(void) const
        {
            return mJobId;
        }

        bool markTaskCompleted(void)
        {
            return --mRemainingTaskCount == 0;
        }

        uint getRemainingTaskCount(void) const
        {
            return mRemainingTaskCount;
        }

        uint getTotalTaskCount(void) const
        {
            return mTotalTaskCount;
        }

        TaskExecutor *getExecutor(void) const
        {
            return mExecutor;
        }

    private:
        uint const mJobId;
        uint mRemainingTaskCount;
        uint const mTotalTaskCount;
        TaskExecutor *mExecutor;
    };

    // state
    // TODO: replace with my BTrie implementation
#ifdef __GXX_EXPERIMENTAL_CXX0X__
    std::unordered_map<uint, PendingJob> mPendingJobs;
#else
    std::tr1::unordered_map<uint, PendingJob> mPendingJobs;
#endif
    bool mFinalized;
    uint mNextJobId;

    // task queue
    typedef struct
    {
        uint index;
        PendingJob *owner;
    } Task;

    WorkQueue<Task> mTaskQueue;

    // used to signal that a job is finished.
    pthread_cond_t mJobFinishedSignal;

    // mutex protecting instance variables.
    pthread_mutex_t mAccessLock;

    // per thread data
    struct ThreadDataStruct *mThreadData;
    uint const mThreadPoolSize;

    // pthread thread body
    static void *ThreadBody(void *arg);

    // called from ThreadBody()
    void run(int threadIndex);
};

}

#endif
