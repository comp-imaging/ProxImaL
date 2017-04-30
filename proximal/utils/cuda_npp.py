from ctypes import *
import numpy as np
from pycuda import gpuarray
import pycuda.autoinit

class NppiSize(Structure):
    _fields_ = [("width", c_int32),
                ("height", c_int32),
                ]

class NppiPoint(Structure):
    _fields_ = [("x", c_int32),
                ("y", c_int32),
                ]

NppStatus = c_int32

nppi = cdll.LoadLibrary("libnppi.so")
nppi.nppiFilter_32f_C1R.argtypes = [POINTER(c_float), c_int32, POINTER(c_float), c_int32, NppiSize, POINTER(c_float), NppiSize, NppiPoint]
nppi.nppiFilter_32f_C1R.restype = NppStatus

def nppiFilter(src, kernel, roi = None, dst = None):
    if len(src.shape) < 2 or len(src.shape) > 3 or (len(src.shape) == 3 and src.shape[-1] > 4):
        raise RuntimeError("Only 2D convolution supported")
    if len(kernel.shape) < 2 or len(kernel.shape) > 3 or (len(kernel.shape) == 3 and kernel.shape[-1] != 1):
        raise RuntimeError("Only 2D convolution supported")

    if roi is None:
        # x,y,width,height
        roi = [0,0,src.shape[1],src.shape[0]]

    if src.dtype == np.float32 and kernel.dtype == np.float32:
        if dst is None:
            dst = gpuarray.zeros(src.shape, src.dtype)
        if dst.dtype != np.float32 or dst.dtype != src.dtype:
            raise RuntimeError("Unsupported destination image.")
        srcp = int(src.gpudata)
        dstp = int(dst.gpudata)

        es = 4

        if len(src.shape) == 2 or (len(src.shape) == 3 and src.shape[-1] == 1):
            nppfunc = nppi.nppiFilter_32f_C1R
            srcp += (roi[0] + roi[1]*src.shape[1])*es
            dstp += (roi[0] + roi[1]*dst.shape[1])*es
            src_step = src.shape[1]*es
            dst_step = dst.shape[1]*es

        elif len(src.shape) == 3 and src.shape[-1] > 0:
            srcp += (roi[0] + roi[1]*src.shape[1])*src.shape[2]*es
            dstp += (roi[0] + roi[1]*dst.shape[1])*src.shape[2]*es
            src_step = src.shape[1]*src.shape[2]*es
            dst_step = dst.shape[1]*src.shape[2]*es
            if src.shape[-1] == 2:
                nppfunc = nppi.nppiFilter_32f_C2R
            elif src.shape[-1] == 3:
                nppfunc = nppi.nppiFilter_32f_C3R
            elif src.shape[-1] == 4:
                nppfunc = nppi.nppiFilter_32f_C4R
            else:
                raise RuntgimeError("Not supported")

        oSizeROI = NppiSize()
        oSizeROI.width = roi[2]
        oSizeROI.height = roi[3]

        kernelp = int(kernel.gpudata)
        kernelSize = NppiSize()
        kernelSize.width = kernel.shape[1]
        kernelSize.height = kernel.shape[0]

        anchor = NppiPoint()
        anchor.x = kernel.shape[1]//2
        anchor.y = kernel.shape[0]//2

        status = nppfunc(cast(srcp, POINTER(c_float)), src_step, cast(dstp, POINTER(c_float)), dst_step, oSizeROI, cast(kernelp, POINTER(c_float)), kernelSize, anchor)
        if status < 0:
            raise RuntimeError("Npp library returned %d" % status)
        return dst

    raise RuntimeError("Not supported")

if __name__ == "__main__":
    src = gpuarray.to_gpu(np.reshape(np.arange(100, dtype=np.float32), (10,10)))
    kernel = gpuarray.to_gpu( np.array([[-1, 0, 1]], dtype=np.float32 ))
    dst = nppiFilter(src, kernel, roi = [1,1,8,8])
    print(dst)

    src = gpuarray.to_gpu(np.reshape(np.arange(200, dtype=np.float32), (10,10,2)))
    kernel = gpuarray.to_gpu( np.array([[-1, 0, 1]], dtype=np.float32 ))
    dst = nppiFilter(src, kernel, roi = [1,1,8,8])
    print(dst[:,:,0])
    print(dst[:,:,1])
