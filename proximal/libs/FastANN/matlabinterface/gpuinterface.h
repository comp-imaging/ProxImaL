
/*
 * gpuinterace.h
 *
 *  Created on: Feb 4, 2014
 *      Author: msteinberger
 *  Copyright 2008 NVIDIA Corporation.  All Rights Reserved
 */

#ifndef GPUINTERFACE_H_
#define GPUINTERFACE_H_


#include <NonLocalMean.h>
#include <structures/DataGroup.cuh>
#include <ostream>

struct Config
{
    NlmDenoiseAlgorithm Algorithm;
    NlmChannelInfo Format;
    int BlockSize;
    int TileSize;
    int ClusterSize;
    int NumCandidates;
};

bool runDenoiser(Config& config, float sigma, ConstChannels inputData, Channels outputData, float& memusage, float& exectime, std::ostream& message);


#endif //GPUINTERFACE_H_
