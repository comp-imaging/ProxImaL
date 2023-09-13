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
Func ifft2_c2r(Func input, int W, int H, const Target& target) {

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

    Input<Buffer<float, 4>> input{"input"};
    Output<Buffer<float, 3>> fftOut{"output"};

    GeneratorParam<int> wtarget{"wtarget", 512, 2, 1024};
    GeneratorParam<int> htarget{"htarget", 512, 2, 1024};
    
    void generate() {
        const auto transformed = ifft2_c2r(input, (int)wtarget, (int)htarget, get_target());

        // Crop the image by the user-defined dimensions
        fftOut(x, y, c) = transformed(x, y, c);
    }

    void schedule() {
        assert(!using_autoscheduler() && "Auto-scheduler not required for FFT interface");
    }
};

HALIDE_REGISTER_GENERATOR(ifft2_c2r_gen, ifftC2RImg);