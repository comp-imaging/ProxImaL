/*
 * Filters.h
 *
 *  Created on: Mar 4, 2013
 *      Author: ytsai
 *	Copyright 2008 NVIDIA Corporation.  All Rights Reserved
 */

#ifndef FILTERS_H_
#define FILTERS_H_

#include <Cuda/Memory.h>

namespace nv {

template<typename FloatType>
__device__        __inline__ FloatType Convolution3x3(cudaTextureObject_t in,
		int ctr_t_x, int ctr_t_y) {
	/** Since Gaussian filter is symmetric - 1 2 1
	 2 4 2
	 1 2 1
	 l c r
	 We can spilt it into 3 columns: left, center, right, respectively.
	 As the convolution goes, we noticed that the result of r can be reused as l in the next convolution.
	 Why? Because of the downsampling game we play:

	 O O O O
	 O X O X
	 O O O O
	 O X O X

	 The above is 4x4 tile and X are the pixels we picked in the down sampling.
	 Only the convoluted result of 'X' is important, we skipped all the 'O' pixels. Note that between 'X' they shares the
	 same column of the pixels, just different side (right vs. left). */

	// now we have avoid expensive SHL, SHR
	int top_t_y = ctr_t_y - 1;
	int btm_t_y = ctr_t_y + 1;
	int lft_t_x = ctr_t_x - 1;
	int rgt_t_x = ctr_t_x + 1;

	FloatType lft, ctr, rgt; // left center right (columns)
	lft = Read2D<FloatType>(in, lft_t_x, top_t_y)
			+ (Read2D<FloatType>(in, lft_t_x, ctr_t_y) * 2.0)
			+ Read2D<FloatType>(in, lft_t_x, btm_t_y);
	ctr = Read2D<FloatType>(in, ctr_t_x, top_t_y)
			+ (Read2D<FloatType>(in, ctr_t_x, ctr_t_y) * 2.0)
			+ Read2D<FloatType>(in, ctr_t_x, btm_t_y);
	rgt = Read2D<FloatType>(in, rgt_t_x, top_t_y)
			+ (Read2D<FloatType>(in, rgt_t_x, ctr_t_y) * 2.0)
			+ Read2D<FloatType>(in, rgt_t_x, btm_t_y);

	return (lft + (ctr * 2.0) + rgt) / 16.0;
}

template<typename FloatType, typename IntType>
__global__ void DownSampleHalfScale3x3(cudaSurfaceObject_t out, cudaTextureObject_t in,
		int x_offset, int y_offset, int width, int height) {
	int x = x_offset + blockIdx.x * blockDim.x + threadIdx.x;
	int y = y_offset + blockIdx.y * blockDim.y + threadIdx.y;

	if (x < (width >> 1) && y < (height >> 1)) {
		FloatType val = Convolution3x3<FloatType>(in, (x << 1) + 1, (y << 1) + 1);
		Write2D<FloatType, IntType>(out, val, x, y);
	}
}

} // namespace nv


#endif /* FILTERS_H_ */
