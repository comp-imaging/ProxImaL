#include "gradTransImg.h"
#include "util.hpp"

namespace proximal {

int
At_grad_glue(const array_float_t input, array_float_t output) {
    auto input_buf = getHalideBuffer<4>(input);
    auto output_buf = getHalideBuffer<3>(output);

    return gradTransImg(input_buf, output_buf);
}

}  // namespace proximal

PYBIND11_MODULE(libAt_grad, m) {
    m.def("run", &proximal::At_grad_glue, "Compute adjoint of gradient");
}