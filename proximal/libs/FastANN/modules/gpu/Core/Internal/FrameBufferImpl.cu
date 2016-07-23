/*
 * FrameBufferImpl.cu
 *
 *  Created on: Nov 26, 2013
 *      Author: ytsai
 */

#include <assert.h>
#include <Core/FrameBufferHandle.h>
#include <Cuda/Cuda.hpp>
#include <Core/Internal/FrameBufferImpl.h>

template<typename Type>
__global__ void CopyToRegionKernel(void* out, int outPitch, const void* in, int inPitch, int widthOffset,
                                   int heightOffset, int nWidth, int nHeight)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= nWidth || y >= nHeight)
    {
        return;
    }

    Type v = nv::Read2D<Type>(in, x + widthOffset, y + heightOffset, inPitch);
    nv::Write2D<Type>(out, v, x, y, outPitch);
}

namespace nv {
namespace internal {
void CopyToRegion(nv::FrameBufferHandle& out, const nv::FrameBufferHandle& in, int widthOffset, int heightOffset,
                  int nWidth, int nHeight)
{
    assert(out.Type() == in.Type());
    assert(((widthOffset + nWidth) <= in.Width()) && ((heightOffset + nHeight) <= in.Height()));
    assert((nWidth <= out.Width()) && (nHeight <= out.Height()));

    const int kBlockSize = 16;
    dim3 threadBlock(kBlockSize, kBlockSize);
    dim3 grid(DIVUP(nWidth, kBlockSize), DIVUP(nHeight, kBlockSize));

    switch(out.Type())
    {
    case FLOAT:
        CopyToRegionKernel<float><<<grid, threadBlock>>>(out.Ptr(), out.Pitch(), in.Ptr(), in.Pitch(), widthOffset, heightOffset, nWidth, nHeight);
        break;
    case UCHAR:
        CopyToRegionKernel<unsigned char><<<grid, threadBlock>>>(out.Ptr(), out.Pitch(), in.Ptr(), in.Pitch(), widthOffset, heightOffset, nWidth, nHeight);
        break;
    case SHORT:
        CopyToRegionKernel<short><<<grid, threadBlock>>>(out.Ptr(), out.Pitch(), in.Ptr(), in.Pitch(), widthOffset, heightOffset, nWidth, nHeight);
        break;
    case USHORT:
        CopyToRegionKernel<unsigned short><<<grid, threadBlock>>>(out.Ptr(), out.Pitch(), in.Ptr(), in.Pitch(), widthOffset, heightOffset, nWidth, nHeight);
        break;
    case INT:
        CopyToRegionKernel<int><<<grid, threadBlock>>>(out.Ptr(), out.Pitch(), in.Ptr(), in.Pitch(), widthOffset, heightOffset, nWidth, nHeight);
        break;
    case HALF:
        CopyToRegionKernel<nv::half><<<grid, threadBlock>>>(out.Ptr(), out.Pitch(), in.Ptr(), in.Pitch(), widthOffset, heightOffset, nWidth, nHeight);
        break;
    }

}

} // namespace internal
} // namespce nv
