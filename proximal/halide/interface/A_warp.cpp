#include "util.hpp"
#include "warpImg.h"

namespace proximal {

int A_warp_glue(const array_float_t input, const array_float_t H,
    array_float_t output) {

        auto input_buf = getHalideBuffer<3>(input);
        auto H_buf = getHalideBuffer<3>(H);
        auto output_buf = getHalideBuffer<4>(output, true);

        const bool success = warpImg(input_buf, H_buf, output_buf);
        output_buf.copy_to_host();
        return success;
    }

} // proximal

PYBIND11_MODULE(A_warp, m) {
    m.def("run", &proximal::A_warp_glue, "Apply affine transform");
}
