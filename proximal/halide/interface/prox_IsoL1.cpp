#include "util.hpp"
#include "proxIsoL1.h"

namespace proximal {

int prox_IsoL1_glue(const array_float_t input, const float theta, array_float_t output) {

        auto input_buf = getHalideBuffer<4>(input);
        auto output_buf = getHalideBuffer<4>(output);

        return proxIsoL1(input_buf, theta, output_buf);
    }

} // proximal

PYBIND11_MODULE(libprox_IsoL1, m) {
    m.def("run", &proximal::prox_IsoL1_glue, "Apply soft thresholding");
}