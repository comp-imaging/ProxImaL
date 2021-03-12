#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <complex>

#include "HalideBuffer.h"

namespace py = pybind11;
using array_float_t = py::array_t<float, py::array::f_style | py::array::forcecast>;
using array_complex_t = py::array_t<std::complex<float>, py::array::f_style | py::array::forcecast>;

namespace {

/** Return halide buffer, with broadcasting */
template <int N, typename T>
Halide::Runtime::Buffer<T>
getHalideBuffer(py::array_t<T, py::array::f_style | py::array::forcecast> input, bool host_dirty=true) {
    const int w = input.shape(1);
    const int h = input.shape(0);
    const int s = (input.ndim() >= 3) ? input.shape(2) : 1;
    const int t = (input.ndim() >= 4) ? input.shape(3) : 1;

    const auto buf = input.request();
    switch (N) {
        case 2: {
            Halide::Runtime::Buffer<T> b{static_cast<T*>(buf.ptr), w, h};
            if (host_dirty) b.set_host_dirty();
            return b;
        }
        case 3: {
            Halide::Runtime::Buffer<T> b{static_cast<T*>(buf.ptr), w, h, s};
            if (host_dirty) b.set_host_dirty();
            return b;
        }
        default: {
            Halide::Runtime::Buffer<T> b{static_cast<T*>(buf.ptr), w, h, s, t};
            if (host_dirty) b.set_host_dirty();
            return b;
        }
    }
}

/** Return halide buffer, with broadcasting */
template <int N, typename T>
Halide::Runtime::Buffer<T>
getHalideComplexBuffer(py::array_t<std::complex<T>, py::array::f_style | py::array::forcecast> input, bool host_dirty=true) {
    const int w = input.shape(1);
    const int h = input.shape(0);
    const int s = (input.ndim() >= 3) ? input.shape(2) : 1;

    const auto buf = input.request();
    switch (N) {
        case 3: {
            Halide::Runtime::Buffer<T> b{static_cast<T*>(buf.ptr), 2, w, h};
            if (host_dirty) b.set_host_dirty();
            return b;
        }
        default: {
            Halide::Runtime::Buffer<T> b{static_cast<T*>(buf.ptr), 2, w, h, s};
            if (host_dirty) b.set_host_dirty();
            return b;
        }
    }
}

}  // namespace