////////////////////////////////////////////////////////////////////////////////
//IFFT2 transform using halide
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

Var x("x"), y("y"), c("c"), k("k");

#include "fft/fft.h"

//Convolution
Func ifft2_c2r(Func input, int W, int H) {

    Target target = get_target_from_environment();

    Fft2dDesc fwd_desc;
    Fft2dDesc inv_desc;
    inv_desc.gain = 1.0f/(W*H);

    //Make complex
    ComplexFunc input_complex;
    input_complex(x, y, c) = {input(x, y, c, 0), input(x, y, c, 1)};

    // Compute the inverse DFT
    Func res = fft2d_c2r(input_complex, W, H, target, inv_desc);

    //Schedule
    res.compute_root();

    return res;
}

class ifft2_c2r_gen : public Generator<ifft2_c2r_gen> {
public:

    ImageParam input{Float(32), 4, "input"};
    
    Func build() {

        //Input
        Func input_func("in");
        input_func(x, y, c, k) = input(x, y, c, k);

        //Warping
        Func fftOut = ifft2_c2r(input_func, WTARGET, HTARGET);

        //Allow for arbitrary strides
        input.set_stride(0, Expr());
        fftOut.output_buffer().set_stride(0, Expr()); 

        return fftOut;
    }
};

auto ifftC2RImg = RegisterGenerator<ifft2_c2r_gen>("ifftC2RImg");