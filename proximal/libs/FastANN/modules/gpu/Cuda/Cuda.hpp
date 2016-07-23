/*
 * Cuda.hpp
 *
 *  Created on: Mar 2, 2013
 *      Author: ytsai
 *	Copyright 2008 NVIDIA Corporation.  All Rights Reserved
 */

#ifndef CUDA_HPP_
#define CUDA_HPP_

#include <Cuda/Arithmetic.h>
#include <Cuda/Filters.h>
#include <Cuda/Limits.h>
#include <Cuda/Memory.h>
#include <Cuda/Vecs.h>

#define CUDA_SAFE_CALL(X) \
{\
    cudaError_t err = X; \
    if(cudaSuccess != (X)) { \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString( err) ); \
    } \
}

#endif /* CUDA_HPP_ */
