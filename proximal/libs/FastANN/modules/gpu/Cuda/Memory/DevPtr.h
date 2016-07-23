/*
 * DevPtr.h
 *
 *  Created on: Mar 4, 2013
 *      Author: ytsai
 *	Copyright 2008 NVIDIA Corporation.  All Rights Reserved
 */

#ifndef DEVPTR_H_
#define DEVPTR_H_
//namespace cv {
//#ifdef __CUDACC__
//    #define __CV_GPU_HOST_DEVICE__ __host__ __device__ __forceinline__
//#else
//    #define __CV_GPU_HOST_DEVICE__
//#endif
//
//template<typename T> struct DevPtr
//{
//    typedef T elem_type;
//    typedef int index_type;
//
//    enum { elem_size = sizeof(elem_type) };
//
//    T* data;
//
//    __CV_GPU_HOST_DEVICE__ DevPtr() : data(0) {}
//    __CV_GPU_HOST_DEVICE__ DevPtr(T* data_) : data(data_) {}
//
//    __CV_GPU_HOST_DEVICE__ size_t elemSize() const { return elem_size; }
//    __CV_GPU_HOST_DEVICE__ operator       T*()       { return data; }
//    __CV_GPU_HOST_DEVICE__ operator const T*() const { return data; }
//};
//
//template<typename T> struct PtrSz : public DevPtr<T>
//{
//    __CV_GPU_HOST_DEVICE__ PtrSz() : size(0) {}
//    __CV_GPU_HOST_DEVICE__ PtrSz(T* data_, size_t size_) : DevPtr<T>(data_), size(size_) {}
//
//    size_t size;
//};
//
//template<typename T> struct PtrStep : public DevPtr<T>
//{
//    __CV_GPU_HOST_DEVICE__ PtrStep() : step(0) {}
//    __CV_GPU_HOST_DEVICE__ PtrStep(T* data_, size_t step_) : DevPtr<T>(data_), step(step_) {}
//
//    /** \brief stride between two consecutive rows in bytes. Step is stored always and everywhere in bytes!!! */
//    size_t step;
//
//    __CV_GPU_HOST_DEVICE__       T* ptr(int y = 0)       { return (      T*)( (      char*)DevPtr<T>::data + y * step); }
//    __CV_GPU_HOST_DEVICE__ const T* ptr(int y = 0) const { return (const T*)( (const char*)DevPtr<T>::data + y * step); }
//
//    __CV_GPU_HOST_DEVICE__       T& operator ()(int y, int x)       { return ptr(y)[x]; }
//    __CV_GPU_HOST_DEVICE__ const T& operator ()(int y, int x) const { return ptr(y)[x]; }
//};
//
//template <typename T> struct PtrStepSz : public PtrStep<T>
//{
//    __CV_GPU_HOST_DEVICE__ PtrStepSz() : cols(0), rows(0) {}
//    __CV_GPU_HOST_DEVICE__ PtrStepSz(int rows_, int cols_, T* data_, size_t step_)
//        : PtrStep<T>(data_, step_), cols(cols_), rows(rows_) {}
//
//    template <typename U>
//    explicit PtrStepSz(const PtrStepSz<U>& d) : PtrStep<T>((T*)d.data, d.step), cols(d.cols), rows(d.rows){}
//
//    int cols;
//    int rows;
//};
//
//typedef PtrStepSz<unsigned char> PtrStepSzb;
//typedef PtrStepSz<float> PtrStepSzf;
//typedef PtrStepSz<int> PtrStepSzi;
//
//typedef PtrStep<unsigned char> PtrStepb;
//typedef PtrStep<float> PtrStepf;
//typedef PtrStep<int> PtrStepi;
//}
namespace nv {
template<typename T> class DevPtr
{
    typedef T elem_type;
    enum {
        elem_size = sizeof(elem_type)
    };
public:
    __host__ __device__ __forceinline__ DevPtr() :
            data_(0) {
    }
    __host__ __device__ __forceinline__ DevPtr(T* data) :
            data_(data) {
    }

    __host__ __device__ __forceinline__ size_t ElemSize() const {
        return elem_size;
    }
    __host__ __device__ __forceinline__ operator T*() {
        return data_;
    }
    __host__ __device__ __forceinline__ operator const T*() const {
        return data_;
    }
protected:

    T* data_;
};

template<typename T> class Ptr2D: public DevPtr<T> {
public:
    __host__ __device__ __forceinline__ Ptr2D() :
            step_(0) {
    }
    __host__ __device__ __forceinline__ Ptr2D(T* data, size_t step) :
            DevPtr<T>(data), step_(step) {
    }

    __host__ __device__ __forceinline__ Ptr2D(const Ptr2D& ptr) :
            DevPtr<T>(ptr.data_), step_(ptr.step_) {

    }

    __host__  __device__ __forceinline__ T* ptr(int y = 0) {
        return (T*) ((char*) DevPtr<T>::data_ + y * step_);
    }
    __host__  __device__ __forceinline__
	 const T* ptr(int y = 0) const {
        return (const T*) ((const char*) DevPtr<T>::data_ + y * step_);
    }

    __host__  __device__ __forceinline__ T& operator ()(int x, int y) {
        return ptr(y)[x];
    }
    __host__  __device__ __forceinline__
	 const T& operator ()(int x, int y) const {
        return ptr(y)[x];
    }

private:
    /** \brief stride between two consecutive rows in bytes. Step is stored always and everywhere in bytes!!! */
    size_t step_;
};

} // namespace nvr

#endif /* DEVPTR_H_ */
