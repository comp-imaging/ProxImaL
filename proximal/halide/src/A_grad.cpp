////////////////////////////////////////////////////////////////////////////////
// Gradient as part of image formation.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/prior_transforms.h"

class grad_gen : public Generator<grad_gen> {
   public:
    Input<Buffer<float>> input{"input", 3};
    Output<Buffer<float>> output{"output", 4};

    void generate() {
        Expr width = input.width();
        Expr height = input.height();

        // Warping
        output(x, y, c, k) = K_grad_mat(input, width, height)(x, y, c, k);
    }

    void schedule() {
        if (auto_schedule) {
            input.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            output.set_estimates({{0, 512}, {0, 512}, {0, 1}, {0, 2}});
            return;
        }

        const auto vec_width = natural_vector_size<float>();

        // Schedule
        output.reorder(k, c, y, x);
        output.vectorize(y, vec_width);
        output.parallel(x);
        output.unroll(k, 2);
    }
};

HALIDE_REGISTER_GENERATOR(grad_gen, gradImg);