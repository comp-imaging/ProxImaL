////////////////////////////////////////////////////////////////////////////////
//Gradient as part of image formation.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/prior_transforms.h"

class grad_gen : public Generator<grad_gen> {
public:

    ImageParam input{Float(32), 3, "input"};
    
    Func build() {
        Expr width = input.width();
        Expr height = input.height();
        
        //Input
        Func input_func("in");
        input_func(x, y, c) = input(x, y, c);

        //Warping
        Func K_input = K_grad_mat(input_func, width, height);

        //Allow for arbitrary strides
        input.set_stride(0, Expr());
        K_input.output_buffer().set_stride(0, Expr()); 

        return K_input;
    }
};

auto gradImg = RegisterGenerator<grad_gen>("gradImg");