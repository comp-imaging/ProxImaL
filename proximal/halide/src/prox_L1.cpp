////////////////////////////////////////////////////////////////////////////////
// L1 proximal operator from "core/prox_operators.h"
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/prox_operators.h"

class proxL1_gen : public Generator<proxL1_gen> {
public:

    Input<Buffer<float>> input{"input", 4};
    Input<float> theta{"theta"};
    Output<Buffer<float>> proxL1_input{"proxL1_input", 4};

    void generate() {
        Expr width = input.width();
        Expr height = input.height();

        // Prox L1 function
        proxL1_input(x, y, c, k) = proxL1(input, width, height, theta)(x, y, c, k);
    }

    void schedule() {
        if (auto_schedule) {
            input.set_estimates({{0, 512}, {0, 512}, {0, 1}, {0, 2}});
            proxL1_input.set_estimates({{0, 512}, {0, 512}, {0, 1}, {0, 2}});
            return;
        }
        // Schedule
        proxL1_input.parallel(y);
        proxL1_input.compute_root();

        //Allow for arbitrary strides
        //input.set_stride(0, Expr());
        //proxL1_input.output_buffer().set_stride(0, Expr()); 
    }
};

HALIDE_REGISTER_GENERATOR(proxL1_gen, proxL1);