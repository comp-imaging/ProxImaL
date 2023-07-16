////////////////////////////////////////////////////////////////////////////////
//Isotropic L1 proximal operator from "core/prox_operators.h"
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/prox_operators.h"

class proxIsoL1_gen : public Generator<proxIsoL1_gen> {
public:

    Input<Buffer<float, 4>> input{"input"};
    Input<float> theta{"theta"};
    Output<Buffer<float, 4>> output{"output"};

    void generate() {
        Expr width = input.width();
        Expr height = input.height();

        // Prox L1 function
        output(x, y, c, k) = proxIsoL1(input, width, height, theta)(x, y, c, k);
    }

    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{0, 512}, {0, 512}, {0, 1}, {0, 2}});
            output.set_estimates({{0, 512}, {0, 512}, {0, 1}, {0, 2}});
            theta.set_estimate(1.0f);
            return;
        }

        //Allow for arbitrary strides
        //input.set_stride(0, Expr());
        //proxIsoL1_input.output_buffer().set_stride(0, Expr()); 
	    output.parallel(y);
    }
};

HALIDE_REGISTER_GENERATOR(proxIsoL1_gen, proxIsoL1)