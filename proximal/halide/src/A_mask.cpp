////////////////////////////////////////////////////////////////////////////////
// Weighting (diagonal weighting matrix M) as part of image formation.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/image_formation.h"

class mask_gen : public Generator<mask_gen> {
   public:
    Input<Buffer<float>> input{"input", 3};
    Input<Buffer<float>> mask{"mask", 3};
    Output<Buffer<float>> output{"output", 3};

    void generate() {
        // Image dimensions
        Expr width = input.width();
        Expr height = input.height();

        // Warping
        output(x, y, c) = A_M(input, width, height, mask)(x, y, c);
    }

    void schedule() {
        if (auto_schedule) {
            input.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            mask.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            output.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            return;
        }

        const auto vec_width = natural_vector_size<float>();

        output.reorder(c, y, x);
        output.vectorize(y, vec_width);
        //output.unroll(c,3);
        output.parallel(x);
    }
};

HALIDE_REGISTER_GENERATOR(mask_gen, WImg);