////////////////////////////////////////////////////////////////////////////////
//Warp with n homographies as part of image formation.
//The different homographies are ordered in stack of matrices.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/image_formation.h"

class warp_trans_gen : public Generator<warp_trans_gen> {
public:

    ImageParam input{Float(32), 4, "input"};
    ImageParam Hinv{Float(32), 3, "H"};
    
    Func build() {
        Expr width = input.width();
        Expr height = input.height();
        Expr nhom = Hinv.channels();

        //Input
        Func input_func("in");
        input_func(x, y, c, g) = input(x, y, c, g);
        
        //Input H
        Func Hinv_func("Hinv");
        Hinv_func(i, j, g) = Hinv(i, j, g);

        //Warping
        Func warp_trans_input = At_warpHomography(input_func, width, height, Hinv_func, nhom);

        //Allow for arbitrary strides
        input.set_stride(0, Expr());
        Hinv.set_stride(0, Expr());
        warp_trans_input.output_buffer().set_stride(0, Expr()); 

        return warp_trans_input;
    }
};

auto warpImg = RegisterGenerator<warp_trans_gen>("warpImg");