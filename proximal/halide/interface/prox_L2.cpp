#include "least_square_direct.h"
#include "util.hpp"

namespace proximal {

int
prox_L2_glue(const array_float_t input, const float theta, const array_float_t offset,
             const array_cxfloat_t freq_diag, const array_float_t output) {
    auto input_buf = getHalideBuffer<3>(input);
    auto offset_buf = getHalideBuffer<3>(offset);
    auto freq_diag_buf = getHalideComplexBuffer<4>(freq_diag);
    auto output_buf = getHalideBuffer<3>(output, false);

    // A hash to denote when the Fourier transformed offset signal should be
    // recomputed. TODO(Antony): It is more practical to marshal the
    // proximal.Problem instance hash from Python runtime to here.
    const auto input_buf_hash = reinterpret_cast<uintptr_t>(input_buf.begin());

    const auto has_error = least_square_direct(input_buf, theta, offset_buf, freq_diag_buf,
                                               input_buf_hash, output_buf);
    output_buf.copy_to_host();
    return has_error;
}

}  // namespace proximal

PYBIND11_MODULE(prox_L2, m) {
    m.def("run", &proximal::prox_L2_glue, "Least square algorithm with direct FFT method");
}