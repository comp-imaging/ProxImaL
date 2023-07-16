////////////////////////////////////////////////////////////////////////////////
// Warp with n homographies as part of image formation.
// The different homographies are ordered in stack of matrices.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/image_formation.h"

class warp_trans_gen : public Generator<warp_trans_gen> {
    Var xo, xi;

   public:
    Input<Buffer<float, 4>> input{"input"};
    Input<Buffer<float, 3>> Hinv{"H"};
    Output<Buffer<float, 3>> output{"output"};

    void generate() {
        Expr width = input.width();
        Expr height = input.height();
        Expr nhom = Hinv.channels();

        // TODO: swap x-y axes here because of how Numpy defines an axis in Fortran order,
        // A better way is to either
        // (i) define stride in halide_dimension_t, or
        // (ii) use c-order in numpy
        Func input_swap_axes;
        input_swap_axes(y, x, c, k) = input(x, y, c, k);

        // Again, swap axes back
        output(y, x, c) = At_warpHomography(input_swap_axes, width, height, Hinv, nhom)(x, y, c);
    }

    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{0, 512}, {0, 512}, {0, 1}, {0, 1}});
            Hinv.set_estimates({{0, 3}, {0, 3}, {0, 1}});
            output.set_estimates({{0, 512}, {0, 512}, {0, 1}});
            return;
        }
        const auto vec_width = natural_vector_size<float>();
        output.vectorize(y, 8);

        output.split(x, xo, x, 32).parallel(xo);
    }
};

HALIDE_REGISTER_GENERATOR(warp_trans_gen, warpImgT);