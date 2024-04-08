#include "util.hpp"
#include "ifftC2RImg.h"

namespace proximal {

constexpr int32_t wtarget{CONFIG_FFT_WIDTH};
constexpr int32_t htarget{CONFIG_FFT_HEIGHT};

int ifft2_c2r_glue(const array_cxfloat_t input, array_float_t output) {

        auto input_buf = getHalideComplexBuffer<4>(input);
        auto output_buf = getHalideBuffer<3>(output, true);

        return ifftC2RImg(input_buf, output_buf);
    }

} // proximal

PYBIND11_MODULE(ifft2_c2r, m) {
    m.def("run", &proximal::ifft2_c2r_glue, "Apply 2D ifft");
    m.attr("wtarget") = pybind11::int_(proximal::wtarget);
    m.attr("htarget") = pybind11::int_(proximal::htarget);
}