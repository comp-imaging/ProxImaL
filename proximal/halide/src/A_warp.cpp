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
    Input<Buffer<float, 3>> input{"input"};
    Input<Buffer<float, 3>> H{"H"};
    Output<Buffer<float, 4>> output{"output"};

    void generate() {
        Expr width = input.width();
        Expr height = input.height();
        Expr nhom = H.channels();

        output = A_warpHomography(input, width, height, H, nhom);
    }

    void schedule() {
        if (using_autoscheduler()) {
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
