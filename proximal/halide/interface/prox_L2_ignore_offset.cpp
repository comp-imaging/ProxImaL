#include "least_square_direct_ignore_offset.h"
#include "util.hpp"

namespace proximal {

int
prox_L2_ignore_offset_glue(const array_float_t input, 
             const array_cxfloat_t freq_diag, const array_float_t output) {
    auto input_buf = getHalideBuffer<3>(input);
    auto freq_diag_buf = getHalideComplexBuffer<4>(freq_diag);
    auto output_buf = getHalideBuffer<3>(output, true);

    constexpr float dont_care = 0;
    auto& dont_care_buf = input_buf;

    const uint64_t dont_care_hash = 0;

    const auto success = least_square_direct_ignore_offset(
        input_buf, dont_care, dont_care_buf, freq_diag_buf, dont_care_hash, output_buf);
    output_buf.copy_to_host();
    return success;
}

}  // namespace proximal

PYBIND11_MODULE(prox_L2_ignore_offset, m) {
    m.def("run", &proximal::prox_L2_ignore_offset_glue, "Least square algorithm with direct FFT method");
}