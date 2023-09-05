////////////////////////////////////////////////////////////////////////////////
// NLM proximal operator from "core/prox_operators.h"
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>

using namespace Halide;

#include "core/prox_operators.h"

namespace {

constexpr auto n_params = 4;

class proxNLM_extern_gen : public Generator<proxNLM_extern_gen> {
   public:
    Input<Buffer<float, 3>> input{"input"};
    Input<float> theta{"theta"};
    Input<Buffer<float, 1>> params{"params"};

    Output<Buffer<float, 3>> output{"output"};

    void generate() {
        const Expr width = input.width();
        const Expr height = input.height();
        const Expr channels = input.channels();

        // Schedule
        Func NLM_input{"NLM_input"};
        output = proxNLM(input, theta, params, width, height, channels);

        input.dim(0).set_min(0).set_stride(1);
        input.dim(1).set_min(0).set_stride(width);
        input.dim(2).set_min(0).set_stride(width * height);

        params.dim(0).set_bounds(0, n_params);

        output.dim(0).set_bounds(0, width).set_stride(1);
        output.dim(1).set_bounds(0, height).set_stride(width);
        output.dim(2).set_bounds(0, channels).set_stride(width * height);
    }

    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            output.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            params.set_estimates({{0, n_params}});
            theta.set_estimate(1.0f);
            return;
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(proxNLM_extern_gen, proxNLM);