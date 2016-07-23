/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef TIMER_H_
#define TIMER_H_

#include <stack>
#include <stdint.h>
#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach_time.h>
#endif

/**
 * @file
 * Definition of Timer.
 */

namespace System
{

/**
 * High-precision timer implementation. The time between function calls
 * can be measured in two ways: by measuring absolute time increments
 * with get() function or by calling matlab style tic()/toc() functions.
 */
class Timer
{
public:
    /**
     * Default constructor.
     */
    Timer(void);

    /**
     * Default destructor.
     */
    ~Timer(void)
    {
    }
    /**
     * Gets current time value and pushes it on the stack.
     */
    void tic(void);

    /**
     * Measures time between toc() and last call to tic(). Technically the function
     * pops the tic() time from the stack and returns a different between current time
     * and tic() value.
     * @return time difference between toc() and last call to tic()
     */
    double toc(void);

    /**
     * Gets the absolute time that has passed since the
     * construction of this Timer object (in milliseconds).
     * @return time in milliseconds
     */
    double get(void);

private:
    uint64_t mStartupTime; /**< Start-up time */
#ifdef _WIN32
    double mFrequency; /**< Inverse of timer frequency */
#endif
#if defined(__APPLE__) && defined(__MACH__)
    mach_timebase_info_data_t mTimerInfo;
#endif
    std::stack<uint64_t> mTimeStampStack; /**< Stack storing time stamps of tic() calls */
};

}

#endif /* TIMER_H_ */
