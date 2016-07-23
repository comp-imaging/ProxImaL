/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#if defined(__APPLE__) && defined(__MACH__)
#else
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif
#endif
#include "SystemTimer.h"

#if !((defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32))
static void GetLocalTime(uint64_t &rval)
{
    timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    rval = tv.tv_sec;
    rval *= 1000000000;
    rval += tv.tv_nsec;
}
#endif

System::Timer::Timer(void)
        : mStartupTime(0),
#ifdef _WIN32
          mFrequency(),
#endif
          mTimeStampStack()
{
#if defined(__APPLE__) && defined(__MACH__)
    mach_timebase_info(&mTimerInfo);
    mStartupTime = mach_absolute_time();
#else
#ifdef _WIN32
    LARGE_INTEGER li;
    QueryPerformanceFrequency(&li);
    mFrequency = 1000.0 / li.QuadPart;
    QueryPerformanceCounter(&li);
    mStartupTime = li.QuadPart;
#else
    GetLocalTime(mStartupTime);
#endif
#endif
}

void System::Timer::tic(void)
{
#if defined(__APPLE__) && defined(__MACH__)
    uint64_t tv;
    tv = mach_absolute_time();
    mTimeStampStack.push(tv);
#else
#ifdef _WIN32
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    mTimeStampStack.push(li.QuadPart);
#else
    uint64_t tv;
    GetLocalTime(tv);
    mTimeStampStack.push(tv);
#endif
#endif
}

double System::Timer::toc(void)
{
#if defined(__APPLE__) && defined(__MACH__)
    uint64_t tv;
    tv = mach_absolute_time();
    uint64_t start = mTimeStampStack.top();
    mTimeStampStack.pop();
    tv -= start;
    tv *= mTimerInfo.numer;
    tv /= mTimerInfo.denom;
    return tv * (1.0 / 1000000.0);
#else
#ifdef _WIN32
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    int64_t start = mTimeStampStack.top();
    mTimeStampStack.pop();
    return (li.QuadPart - start) * mFrequency;
#else
    uint64_t tv;
    GetLocalTime(tv);
    uint64_t start = mTimeStampStack.top();
    mTimeStampStack.pop();
    return (tv - start) * (1.0 / 1000000.0);
#endif
#endif
}

double System::Timer::get(void)
{
#if defined(__APPLE__) && defined(__MACH__)
    uint64_t tv;
    tv = mach_absolute_time();
    tv -= mStartupTime;
    tv *= mTimerInfo.numer;
    tv /= mTimerInfo.denom;
    return tv * (1.0 / 1000000.0);
#else
#ifdef _WIN32
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (li.QuadPart - mStartupTime) * mFrequency;
#else
    uint64_t tv;
    GetLocalTime(tv);
    return (tv - mStartupTime) * (1.0 / 1000000.0);
#endif
#endif
}

