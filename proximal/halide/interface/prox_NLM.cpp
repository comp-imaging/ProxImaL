#include "proxNLM.h"
#include "util.hpp"

namespace proximal {

int
prox_NLM_glue(const array_float_t input, const float theta, const array_float_t params,
              array_float_t output) {
    auto input_buf = getHalideBuffer<3>(input);
    auto params_buf = getHalideBuffer<1>(params);

    auto output_buf = getHalideBuffer<3>(output, true);

    const auto success = proxNLM(input_buf, theta, params_buf, output_buf);
    output_buf.copy_to_host();
    return success;
}

}  // namespace proximal

PYBIND11_MODULE(prox_NLM, m) {
    m.def("run", &proximal::prox_NLM_glue, "Denoise image with non-local means");
}