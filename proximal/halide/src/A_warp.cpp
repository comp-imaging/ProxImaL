////////////////////////////////////////////////////////////////////////////////
// Warp with n homographies as part of image formation.
// The different homographies are ordered in stack of matrices.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/image_formation.h"

class warp_gen : public Generator<warp_gen> {
   public:
    Input<Buffer<float>> input{"input", 3};
    Input<Buffer<float>> H{"H", 3};
    Output<Buffer<float>> output{"output", 4};

    void generate() {
        Expr width = input.width();
        Expr height = input.height();
        Expr nhom = H.channels();

        // TODO: swap x-y axes here because of how Numpy defines an axis in Fortran order,
        // A better way is to either
        // (i) define stride in halide_dimension_t, or
        // (ii) use c-order in numpy
        Func input_swap_axes;
        input_swap_axes(y, x, c) = input(x, y, c);

        // Again, swap axes back
        output(y, x, c, k) = A_warpHomography(input_swap_axes, width, height, H, nhom)(x, y, c, k);
    }

    void schedule() {
        if (auto_schedule) {
            input.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            H.set_estimates({{0, 3}, {0, 3}, {0, 1}});
            output.set_estimates({{0, 512}, {0, 512}, {0, 1}, {0, 2}});
            return;
        }

        const auto vec_width = natural_vector_size<float>();
        output.vectorize(y, vec_width);

        Var xo, xi;
        output.split(x, xo, x, 32).parallel(xo);
    }
};

HALIDE_REGISTER_GENERATOR(warp_gen, warpImg);
