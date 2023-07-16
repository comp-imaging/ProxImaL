////////////////////////////////////////////////////////////////////////////////
// Gradient trans as part of image formation.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/prior_transforms.h"

class grad_trans_gen : public Generator<grad_trans_gen> {
   public:
    Input<Buffer<float, 4>> input{"input"};
    Output<Buffer<float, 3>> output{"output"};

    void generate() {
        Expr width = input.width();
        Expr height = input.height();

        // Warping
        output(x, y, c) = KT_grad_mat(input, width, height)(x, y, c);
    }

    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{0, 512}, {0, 512}, {0, 1}, {0, 2}});
            output.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            return;
        }

        // Schedule
        output.reorder(c, y, x);
        output.parallel(x);

        const auto vec_width = natural_vector_size<float>();
        output.vectorize(y, vec_width);
    }
};

HALIDE_REGISTER_GENERATOR(grad_trans_gen, gradTransImg);