#include "util.hpp"
#include "warpImg.h"

namespace proximal {

int A_warp_glue(const array_float_t input, const array_float_t H,
    array_float_t output) {

        auto input_buf = getHalideBuffer<3>(input);
        auto H_buf = getHalideBuffer<3>(H);
        auto output_buf = getHalideBuffer<4>(output);

        return warpImg(input_buf, H_buf, output_buf);
    }

} // proximal

PYBIND11_MODULE(libA_warp, m) {
    m.def("run", &proximal::A_warp_glue, "Apply affine transform");
}
