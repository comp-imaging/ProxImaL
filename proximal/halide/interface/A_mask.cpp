#include "util.hpp"
#include "WImg.h"

namespace proximal {

int A_mask(const array_float_t input, const array_float_t mask,
    array_float_t output) {

        auto input_buf = getHalideBuffer<3>(input);
        auto mask_buf = getHalideBuffer<3>(mask);
        auto output_buf = getHalideBuffer<3>(output, true);

        const bool success = WImg(input_buf, mask_buf, output_buf);
        output_buf.copy_to_host();
        return success;
    }

} // proximal

PYBIND11_MODULE(A_mask, m) {
    m.def("run", &proximal::A_mask, "Apply elementwise multiplication");
}