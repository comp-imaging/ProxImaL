////////////////////////////////////////////////////////////////////////////////
//Convolution as part of image formation.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/image_formation.h"

class conv_trans_gen : public Generator<conv_trans_gen> {
public:

    Input<Buffer<float, 3>> input{"input"};
    Input<Buffer<float, 3>> K{"K"};
    Output<Buffer<float, 3>> conv_trans_input{"output"};
    
    void generate() {
        Expr width = input.width();
        Expr height = input.height();
        
        Expr width_kernel = K.width();
        Expr height_kernel = K.height();

        //Warping
        conv_trans_input(x, y, c) = At_conv(input, width, height, K, width_kernel, height_kernel)(x, y, c);
    }

    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            K.set_estimates({{0, 15}, {0, 15}, {0, 1}});
            conv_trans_input.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            return;
        }

        // Schedule
        conv_trans_input.reorder(c, x, y);

        const auto vec_width = natural_vector_size<float>();
        // Parallel
        conv_trans_input.vectorize(x, vec_width);
        conv_trans_input.parallel(y);
    }
};

HALIDE_REGISTER_GENERATOR(conv_trans_gen, convImgT);