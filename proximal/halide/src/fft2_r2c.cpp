////////////////////////////////////////////////////////////////////////////////
//FFT2 transform using halide
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

Var x("x"), y("y"), c("c"), k("k");

#include "fft/fft.h"

//Convolution
Func fft2_r2c(Func input, int W, int H) {

    //Local vars
    Var h("h"), l("l"), m("m"); 

    Target target = get_target_from_environment();

    Fft2dDesc fwd_desc;
    Fft2dDesc inv_desc;
    inv_desc.gain = 1.0f/(W*H);

    // Compute the DFT of the input and the kernel.
    ComplexFunc dft_in = fft2d_r2c(input, W, H, target, fwd_desc);
    dft_in.compute_root();
  
    // Pure definition: do nothing.
    Func dft_in_flat("dft_in_flat");
    dft_in_flat(h, l, m, k) = undef<float>();
    dft_in_flat(h, l, m, 0) =  re( dft_in(h, l, m) );
    dft_in_flat(h, l, m, 1) =  im( dft_in(h, l, m) );

    //Schedule
    dft_in_flat.compute_root();
    return dft_in_flat;
}

class fft2_r2c_gen : public Generator<fft2_r2c_gen> {
public:

    ImageParam input{Float(32), 3, "input"};
    Param<int> shiftx{"shiftx"};
    Param<int> shifty{"shifty"};

    Func build() {

        //Input
        Func input_func("in");
        Func paddedInput("paddedInput");
        paddedInput = repeat_image( constant_exterior(input, 0.f), 0, WTARGET, 0, HTARGET);
        input_func(x, y, c) = paddedInput( x + shiftx, y + shifty, c );

        //Warping
        Func fftIn = fft2_r2c(input_func, WTARGET, HTARGET);

        //Allow for arbitrary strides
        input.set_stride(0, Expr());
        fftIn.output_buffer().set_stride(0, Expr()); 

        return fftIn;
    }
};

auto fftR2CImg = RegisterGenerator<fft2_r2c_gen>("fftR2CImg");