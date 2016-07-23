////////////////////////////////////////////////////////////////////////////////
// L1 proximal operator from "core/prox_operators.h"
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/prox_operators.h"

class proxL1_gen : public Generator<proxL1_gen> {
public:

    ImageParam input{Float(32), 4, "input"};
    Param<float> theta{"theta"};

    Func build() {
        Expr width = input.width();
        Expr height = input.height();

        Func input_func;
        input_func(x, y, c, k) = input(x, y, c, k);

        // Prox L1 function
        Func proxL1_input = proxL1(input_func, width, height, theta);

        //Allow for arbitrary strides
        input.set_stride(0, Expr());
        proxL1_input.output_buffer().set_stride(0, Expr()); 
        
        return proxL1_input;
    }
};

auto proxl1 = RegisterGenerator<proxL1_gen>("proxl1");