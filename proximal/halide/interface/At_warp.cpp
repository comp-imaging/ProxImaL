#include "util.hpp"
#include "warpImgT.h"

namespace proximal {

int At_warp_glue(const array_float_t input, const array_float_t H,
    array_float_t output) {
    auto input_buf = getHalideBuffer<4>(input, false, false);
    auto H_buf = getHalideBuffer<3>(H);
    auto output_buf = getHalideBuffer<3>(output, true, false);

    const bool success = warpImgT(input_buf, H_buf, output_buf);
    output_buf.copy_to_host();
    return success;
    }

} // proximal

PYBIND11_MODULE(At_warp, m) {
    m.def("run", &proximal::At_warp_glue, "Apply inverse affine transform");
}
