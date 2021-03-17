#include "proxPoisson.h"
#include "util.hpp"

namespace proximal {

int
prox_Poisson_glue(const array_float_t input, const array_float_t M, const array_float_t b,
                  const float theta, array_float_t output) {
    auto input_buf = getHalideBuffer<3>(input);
    auto M_buf = getHalideBuffer<3>(M);
    auto b_buf = getHalideBuffer<3>(b);

    auto output_buf = getHalideBuffer<3>(output, true);

    const bool success = proxPoisson(input_buf, M_buf, b_buf, theta, output_buf);
    output_buf.copy_to_host();
    return success;
}

}  // namespace proximal

PYBIND11_MODULE(prox_Poisson, m) {
    m.def("run", &proximal::prox_Poisson_glue, "Apply proximal function of Poisson statistics");
}