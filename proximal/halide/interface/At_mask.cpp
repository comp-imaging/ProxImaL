#include "util.hpp"
#include "WImg.h"

namespace proximal {

int At_mask(const array_float_t input, const array_float_t mask,
    array_float_t output) {

        auto input_buf = getHalideBuffer<3>(input);
        auto mask_buf = getHalideBuffer<3>(mask);
        auto output_buf = getHalideBuffer<3>(output);

        return WImg(input_buf, mask_buf, output_buf);
    }

} // proximal

PYBIND11_MODULE(libAt_mask, m) {
    m.def("run", &proximal::At_mask, "Apply conjugate elementwise multiplication");
}