/*
 * gpuinterace.h
 *
 *  Created on: Feb 5, 2014
 *      Author: msteinberger
 *  Copyright 2008 NVIDIA Corporation.  All Rights Reserved
 */

#ifndef GPUINTERFACE_SEPARATE_H_
#define GPUINTERFACE_SEPARATE_H_


#include <NonLocalMean.h>
#include <Core/FrameBufferHandle.h>
#include <ostream>
#include <vector>

bool runIndexing(std::vector<int2>& clusteredPixCoords, std::vector<int>& clusterOffsets, std::vector<int>& clusterSizes, nv::FrameBufferHandle& in, int& TileSize, int& BlockSize, int& ClusterSize, float& memusage, float& runtime, std::ostream& messages);
bool runQuery(std::vector<int2>& candidates, nv::FrameBufferHandle& in, std::vector<int2> & clusteredPixCoords, std::vector<int> & clusterOffsets, std::vector<int> & clusterSizes, int& TileSize, int& BlockSize, int& ClusterSize, int& CandidatesSize, float& memusage, float& runtime, std::ostream& messages, bool PowTwoCandidates);


#endif //GPUINTERFACE_SEPARATE_H_
