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

        output = At_warpHomography(input, width, height, Hinv, nhom);
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