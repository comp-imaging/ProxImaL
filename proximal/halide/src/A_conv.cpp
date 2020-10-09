////////////////////////////////////////////////////////////////////////////////
//Convolution as part of image formation.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/image_formation.h"

class conv_gen : public Generator<conv_gen> {
public:

    Input<Buffer<float>> input{"input", 3};
    Input<Buffer<float>> K{"K", 3};
    Output<Buffer<float>> conv_output{"output", 3};
    
    void generate () {
        Expr width = input.width();
        Expr height = input.height();
        
        Expr width_kernel = K.width();
        Expr height_kernel = K.height();

        //Warping
        conv_output(x, y, c) = A_conv(input, width, height, K, width_kernel, height_kernel)(x, y, c);
    }

    void schedule() {
        if (auto_schedule) {
            input.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            K.set_estimates({{0, 15}, {0, 15}, {0, 1}});
            conv_output.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            return;
        }
        // Schedule
        conv_output.reorder(c, x, y);

        // Parallel
        const auto vec_width = natural_vector_size<float>();
        conv_output.vectorize(x, vec_width);
        conv_output.parallel(y, 8);

        //Allow for arbitrary strides
        //input.set_stride(0, Expr());
        //K.set_stride(0, Expr());
        //conv_input.output_buffer().set_stride(0, Expr()); 
    }
};

HALIDE_REGISTER_GENERATOR(conv_gen, convImg);