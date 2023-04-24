#include "util.hpp"
#include "fftR2CImg.h"

namespace proximal {

int fft2_r2c_glue(const array_float_t input, const int xshift,
    const int yshift, array_cxfloat_t output) {

        auto input_buf = getHalideBuffer<3>(input);
        auto output_buf = getHalideComplexBuffer<4>(output, true);

        return fftR2CImg(input_buf, xshift, yshift, output_buf);
    }

} // proximal

PYBIND11_MODULE(fft2_r2c, m) {
    m.def("run", &proximal::fft2_r2c_glue, "Apply 2D adjoint convolution");
}