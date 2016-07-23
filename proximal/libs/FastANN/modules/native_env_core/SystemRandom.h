/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef SYSTEMRANDOM_H_
#define SYSTEMRANDOM_H_

#include "Base.h"

namespace System
{

#define RAND48_SEED_0 0x330e
#define RAND48_MULT_0 0xe66d
#define RAND48_MULT_1 0xdeec
#define RAND48_MULT_2 0x0005
#define RAND48_ADD    0x000b

class Rand48
{
public:
    Rand48(void)
    {
        setSeed(0);
    }

    Rand48(int seedValue)
    {
        setSeed(seedValue);
    }

    /**
     * Sets the random number generator seed value.
     * @param seed - random number generator seed value
     */
    void setSeed(int seedValue)
    {
        mSeed[0] = RAND48_SEED_0;
        mSeed[1] = seedValue;
        mSeed[2] = seedValue >> 16;
    }

    /**
     * Gets the random number using rand48 algorithm.
     * @return random number value
     */
    int getInt(void)
    {
        ushort temp[2];
        uint accum;

        accum = RAND48_MULT_0 * mSeed[0] + RAND48_ADD;
        temp[0] = (ushort)accum;
        accum >>= 16;
        accum += RAND48_MULT_0 * mSeed[1] + RAND48_MULT_1 * mSeed[0];
        temp[1] = (ushort)accum;
        accum >>= 16;
        accum += RAND48_MULT_0 * mSeed[2] + RAND48_MULT_1 * mSeed[1] + RAND48_MULT_2 * mSeed[0];

        mSeed[0] = temp[0];
        mSeed[1] = temp[1];
        mSeed[2] = (ushort)accum;

        return (accum << 16) | temp[1];
    }

private:
    ushort mSeed[3];
};

}

#endif /* SYSTEMRANDOM_H_ */
