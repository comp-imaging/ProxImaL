////////////////////////////////////////////////////////////////////////////////
//Gradient trans as part of image formation.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/prior_transforms.h"

class grad_trans_gen : public Generator<grad_trans_gen> {
public:

    ImageParam input{Float(32), 4, "input"};
    
    Func build() {
        Expr width = input.width();
        Expr height = input.height();

        //Input
        Func input_func("in");
        input_func(x, y, c, g) = input(x, y, c, g);

        //Warping
        Func grad_trans_input = KT_grad_mat(input_func, width, height); 

        //Allow for arbitrary strides
        input.set_stride(0, Expr());
        grad_trans_input.output_buffer().set_stride(0, Expr()); 

        return grad_trans_input;
    }
};

auto gradTransImg = RegisterGenerator<grad_trans_gen>("gradTransImg");