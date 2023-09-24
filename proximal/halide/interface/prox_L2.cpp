#include "least_square_direct.h"
#include "util.hpp"

namespace proximal {

int
prox_L2_glue(const array_float_t input, const float theta, const array_float_t offset,
             const array_cxfloat_t freq_diag, const array_float_t output) {
    auto input_buf = getHalideBuffer<3>(input);
    auto offset_buf = getHalideBuffer<3>(offset);
    auto freq_diag_buf = getHalideComplexBuffer<4>(freq_diag);
    auto output_buf = getHalideBuffer<3>(output, true);

    const auto success = least_square_direct(input_buf, theta, offset_buf,
                                             freq_diag_buf, output_buf);
    output_buf.copy_to_host();
    return success;
}

}  // namespace proximal

PYBIND11_MODULE(prox_L2, m) {
    m.def("run", &proximal::prox_L2_glue, "Least square algorithm with direct FFT method");
}