#include "util.hpp"
#include "fftR2CImg.h"

namespace proximal {

int fft2_r2c_glue(const array_float_t input, const int xshift, 
    const int yshift, array_complex_t output) {

        auto input_buf = getHalideBuffer<3>(input);
        auto output_buf = getHalideComplexBuffer<4>(output);

        return fftR2CImg(input_buf, xshift, yshift, output_buf);
    }

} // proximal

PYBIND11_MODULE(libfft2_r2c, m) {
    m.def("run", &proximal::fft2_r2c_glue, "Apply 2D adjoint convolution");
}