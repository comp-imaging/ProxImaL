////////////////////////////////////////////////////////////////////////////////
//Isotropic L1 proximal operator from "core/prox_operators.h"
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/prox_operators.h"

class proxIsoL1_gen : public Generator<proxIsoL1_gen> {
public:

    ImageParam input{Float(32), 4, "input"};
    Param<float> theta{"theta"};

    Func build() {
        Expr width = input.width();
        Expr height = input.height();

        Func input_func;
        input_func(x, y, c, k) = input(x, y, c, k);

        // Prox L1 function
        Func proxIsoL1_input = proxIsoL1(input_func, width, height, theta);

        //Allow for arbitrary strides
        input.set_stride(0, Expr());
        proxIsoL1_input.output_buffer().set_stride(0, Expr()); 
        
        return proxIsoL1_input;
    }
};

auto proxIsol1 = RegisterGenerator<proxIsoL1_gen>("proxIsol1");