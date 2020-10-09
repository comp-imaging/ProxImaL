////////////////////////////////////////////////////////////////////////////////
//IFFT2 transform using halide
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "fft/fft.h"

namespace {
Var x("x"), y("y"), c("c"), k("k");

//Convolution
Func ifft2_c2r(Func input, int W, int H) {

    Target target = get_target_from_environment();

    Fft2dDesc inv_desc;
    inv_desc.gain = 1.0f/(W*H);

    //Make complex
    ComplexFunc input_complex;
    input_complex(x, y, c) = {input(0, x, y, c), input(1, x, y, c)};

    // Compute the inverse DFT
    Func res = fft2d_c2r(input_complex, W, H, target, inv_desc);
    res.compute_root();

    return res;
}

} // namespace

class ifft2_c2r_gen : public Generator<ifft2_c2r_gen> {
public:

    Input<Buffer<float>> input{"input", 4};
    Output<Buffer<float>> fftOut{"output", 3};

    GeneratorParam<int> wtarget{"wtarget", 512, 2, 1024};
    GeneratorParam<int> htarget{"htarget", 512, 2, 1024};
    
    void generate() {
        //Warping
        fftOut(x, y, c) = ifft2_c2r(input, (int)wtarget, (int)htarget)(x, y, c);
    }

    void schedule() {
    }
};

HALIDE_REGISTER_GENERATOR(ifft2_c2r_gen, ifftC2RImg);