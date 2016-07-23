/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include "ConcurrentWorkQueue.h"
#include "SystemCore.h"

namespace System
{
typedef struct ThreadDataStruct
{
    int threadIndex;
    pthread_t threadId;
    ConcurrentWorkQueue *executor;
} ThreadData;
}

System::ConcurrentWorkQueue::ConcurrentWorkQueue(const uint poolSize)
        : mPendingJobs(), mFinalized(false), mNextJobId(1), mTaskQueue(), mJobFinishedSignal(), mAccessLock(),
          mThreadData(new ThreadData[poolSize]), mThreadPoolSize(poolSize)
{
    // initialize mutex and cond
    pthread_mutex_init(&mAccessLock, 0);
    pthread_cond_init(&mJobFinishedSignal, 0);

    // initialize threads
    for (uint i = 0; i < poolSize; i++)
    {
        mThreadData[i].executor = this;
        mThreadData[i].threadIndex = i;
        pthread_create(&mThreadData[i].threadId, 0, &ConcurrentWorkQueue::ThreadBody, &mThreadData[i]);
    }
}

System::ConcurrentWorkQueue::~ConcurrentWorkQueue()
{
    finalize();

    // gracefully destroy threads
    for (uint i = 0; i < mThreadPoolSize; i++)
    {
        pthread_join(mThreadData[i].threadId, 0);

        LOG_DEBUG( LOG_SYS, "thread %i joined...", mThreadData[i].threadIndex);
    }

    delete[] mThreadData;

    // destroy mutex and cond
    pthread_mutex_destroy(&mAccessLock);
    pthread_cond_destroy(&mJobFinishedSignal);
}

void System::ConcurrentWorkQueue::finalize(void)
{
    if (mFinalized)
    {
        return;
    }

    waitForAllJobs();
    mTaskQueue.finalize();

    mFinalized = true;
}

void System::ConcurrentWorkQueue::waitForAllJobs()
{
    if (mFinalized)
    {
        return;
    }

    // wait for the idle condition.
    pthread_mutex_lock(&mAccessLock);

    while (!mPendingJobs.empty())
    {
        pthread_cond_wait(&mJobFinishedSignal, &mAccessLock);
    }

    pthread_mutex_unlock(&mAccessLock);
}

void System::ConcurrentWorkQueue::waitForJob(uint jobId)
{
    if (mFinalized)
    {
        return;
    }

    // wait for the job-finished condition.
    pthread_mutex_lock(&mAccessLock);

    while (mPendingJobs.find(jobId) != mPendingJobs.end())
    {
        pthread_cond_wait(&mJobFinishedSignal, &mAccessLock);
    }

    pthread_mutex_unlock(&mAccessLock);
}

uint System::ConcurrentWorkQueue::enqueue(TaskExecutor *job)
{
    if (mFinalized)
    {
        return 0;
    }

    uint taskCount = job->getMaximumTaskCount();
    if (taskCount > mThreadPoolSize)
    {
        // divide the job into task count not larger than pool size
        taskCount = mThreadPoolSize;
    }

    // TODO: increment might be a very bad idea for long running programs
    // need to implement some 1D R-tree to efficiently deal with unique uint id
    // search problem

    pthread_mutex_lock(&mAccessLock);

    uint const jobId = mNextJobId++;
    PendingJob *jobData = &mPendingJobs.insert(std::pair<uint, PendingJob>(jobId, PendingJob(jobId, taskCount, job)))
            .first->second;

    pthread_mutex_unlock(&mAccessLock);

    // add tasks to the thread-safe queue
    std::vector<Task> taskArray(taskCount);
    for (uint i = 0; i < taskCount; i++)
    {
        taskArray[i].index = i;
        taskArray[i].owner = jobData;
    }

    mTaskQueue.produce(taskArray);

    return jobId;
}

void *System::ConcurrentWorkQueue::ThreadBody(void *arg)
{
    ThreadData *tdata = static_cast<ThreadData *>(arg);
    tdata->executor->run(tdata->threadIndex);
    return 0;
}

void System::ConcurrentWorkQueue::run(int threadIndex)
{
    (void)threadIndex;
    Task task;

    // loop infinitely, waiting for a task in the queue
    while (true)
    {
        // LOG("thread %i waiting for task...", threadIndex);
        // fetch a task
        if (!mTaskQueue.consume(task, true))
        {
            break;
        }

        // LOG("thread %i got task %i from job %i...", threadIndex, task.index, task.owner->getJobId());
        // execute task
        PendingJob *currentJob = task.owner;
        currentJob->getExecutor()->run(task.index, currentJob->getTotalTaskCount());
        // LOG("thread %i finished task %i from job %i...", threadIndex, task.index, task.owner->getJobId());

        pthread_mutex_lock(&mAccessLock);

        // mask task complete
        if (currentJob->markTaskCompleted())
        {
            // LOG("thread %i finished job %i...", threadIndex, task.owner->getJobId());

            // all tasks in this job are finished
            mPendingJobs.erase(currentJob->getJobId());
            // broadcast that a job is finished, in case someone is waiting on it
            pthread_cond_broadcast(&mJobFinishedSignal);
        }

        pthread_mutex_unlock(&mAccessLock);
    }
}
