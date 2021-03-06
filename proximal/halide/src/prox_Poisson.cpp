////////////////////////////////////////////////////////////////////////////////
// Test function for Poisson penalty proximal operator from "core/prox_operators.h"
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/prox_operators.h"

class proxPoisson_gen : public Generator<proxPoisson_gen> {
   public:
    Input<Buffer<float>> input{"input", 3};
    Input<Buffer<float>> M{"M", 3};
    Input<Buffer<float>> b{"b", 3};
    Input<float> theta{"theta"};

    Output<Buffer<float>> output{"output", 3};

    void generate() {
        // Inputs
        Expr width = input.width();
        Expr height = input.height();

        // Prox L1 function
        output(x, y, c) = proxPoisson(input, width, height, M, b, theta)(x, y, c);
    }

    void schedule() {
        if (auto_schedule) {
            input.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            M.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            b.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            output.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            return;
        }

        // Schedule
        output.parallel(y);
    }
};

HALIDE_REGISTER_GENERATOR(proxPoisson_gen, proxPoisson);
