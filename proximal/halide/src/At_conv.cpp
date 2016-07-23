////////////////////////////////////////////////////////////////////////////////
//Convolution as part of image formation.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/image_formation.h"

class conv_trans_gen : public Generator<conv_trans_gen> {
public:

    ImageParam input{Float(32), 3, "input"};
    ImageParam K{Float(32), 3, "K"};
    
    Func build() {
        Expr width = input.width();
        Expr height = input.height();
        
        Expr width_kernel = K.width();
        Expr height_kernel = K.height();

        //Input
        Func input_func("in");
        input_func(x, y, c) = input(x, y, c);
        
        //Input H
        Func K_func("K");
        K_func(i, j, c) = K(i, j, c);

        //Warping
        Func conv_trans_input = At_conv(input_func, width, height, K_func, width_kernel, height_kernel);

        //Allow for arbitrary strides
        input.set_stride(0, Expr());
        K.set_stride(0, Expr());
        conv_trans_input.output_buffer().set_stride(0, Expr()); 

        return conv_trans_input;
    }
};

auto convImg = RegisterGenerator<conv_trans_gen>("convImg");