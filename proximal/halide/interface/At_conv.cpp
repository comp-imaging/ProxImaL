#include "util.hpp"
#include "convImgT.h"

namespace proximal {

int At_conv_glue(const array_float_t input, const array_float_t K,
    array_float_t output) {

        auto input_buf = getHalideBuffer<3>(input);
        auto K_buf = getHalideBuffer<3>(K);
        auto output_buf = getHalideBuffer<3>(output, true);

        const int success = convImgT(input_buf, K_buf, output_buf);
        output_buf.copy_to_host();
        return success;
    }

} // proximal

PYBIND11_MODULE(libAt_conv, m) {
    m.def("run", &proximal::At_conv_glue, "Apply 2D adjoint convolution");
}