////////////////////////////////////////////////////////////////////////////////
//Warp with n homographies as part of image formation.
//The different homographies are ordered in stack of matrices.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/image_formation.h"

class warp_gen : public Generator<warp_gen> {
public:

    ImageParam input{Float(32), 3, "input"};
    ImageParam H{Float(32), 3, "H"};
    
    Func build() {

        Expr width = input.width();
        Expr height = input.height();
        Expr nhom = H.channels();

        //Input
        Func input_func("in");
        input_func(x, y, c) = input(x, y, c);
        
        //Input H
        Func H_func("H");
        H_func(i, j, g) = H(i, j, g);

        //Warping
        Func warp_input = A_warpHomography(input_func, width, height, H_func, nhom);
       
        //Allow for arbitrary strides
        input.set_stride(0, Expr());
        H.set_stride(0, Expr());
        warp_input.output_buffer().set_stride(0, Expr()); 

        return warp_input;
    }
};

auto warpImg = RegisterGenerator<warp_gen>("warpImg");