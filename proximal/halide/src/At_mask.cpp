////////////////////////////////////////////////////////////////////////////////
// Weighting transpose (diagonal weighting matrix M) as part of image formation.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/image_formation.h"

class mask_trans_gen : public Generator<mask_trans_gen> {
public:

    ImageParam input{Float(32), 3, "input"};
    ImageParam mask{Float(32), 3, "mask"};
    
    Func build() {
        //Image dimensions
        Expr width = input.width();
        Expr height = input.height();
        
        //Input
        Func input_func("input");
        input_func(x, y, c) = input(x, y, c);
        
        //Input mask
        Func M_func("mask");
        M_func(x, y, c) = mask(x, y, c);

        //Warping
        Func M_trans_input = At_M(input_func, width, height, M_func);

        //Allow for arbitrary strides
        input.set_stride(0, Expr());
        mask.set_stride(0, Expr());
        M_trans_input.output_buffer().set_stride(0, Expr()); 

        return M_trans_input;
    }
};

auto WImg = RegisterGenerator<mask_trans_gen>("WImg");