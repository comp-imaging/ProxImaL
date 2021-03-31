#include "gradImg.h"
#include "util.hpp"

namespace proximal {

int
A_grad_glue(const array_float_t input, array_float_t output) {
    auto input_buf = getHalideBuffer<3>(input);
    auto output_buf = getHalideBuffer<4>(output, true);

    const int success = gradImg(input_buf, output_buf);
    output_buf.copy_to_host();
    return success;
}

}  // namespace proximal

PYBIND11_MODULE(A_grad, m) {
    m.def("run", &proximal::A_grad_glue, "Compute adjoint of gradient");
}