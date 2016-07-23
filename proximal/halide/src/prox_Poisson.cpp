////////////////////////////////////////////////////////////////////////////////
//Test function for Poisson penalty proximal operator from "core/prox_operators.h"
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/prox_operators.h"

class proxPoisson_gen : public Generator<proxPoisson_gen> {
public:

    ImageParam input{Float(32), 3, "input"};
    ImageParam M{Float(32), 3, "M"};
    ImageParam b{Float(32), 3, "b"};
    Param<float> theta{"theta"};

    Func build() {
        // Inputs
        Expr width = input.width();
        Expr height = input.height();

        Func input_func;
        input_func(x, y, c) = input(x, y, c);

        Func M_func;
        M_func(x, y, c) = M(x, y, c);

        Func b_func;
        b_func(x, y, c) = b(x, y, c);

        // Prox L1 function
        Func proxPoisson_input = proxPoisson(input_func, width, height, M_func, b_func, theta);
        
        //Allow for arbitrary strides
        input.set_stride(0, Expr());
        M.set_stride(0, Expr());
        b.set_stride(0, Expr());
        proxPoisson_input.output_buffer().set_stride(0, Expr()); 

        return proxPoisson_input;
    }
};

auto prox_poisson = RegisterGenerator<proxPoisson_gen>("prox_poisson");