////////////////////////////////////////////////////////////////////////////////
//FFT2 transform using halide
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

namespace {

Var x("x"), y("y"), c("c"), k("k");

} // namespace


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
    dft_in_flat(k, h, l, m) = undef<float>();
    dft_in_flat(0, h, l, m) =  re( dft_in(h, l, m) );
    dft_in_flat(1, h, l, m) =  im( dft_in(h, l, m) );

    return dft_in_flat;
}

class fft2_r2c_gen : public Generator<fft2_r2c_gen> {
public:

    Input<Buffer<float>> input{"input", 3};
    Input<int> shiftx{"shiftx"};
    Input<int> shifty{"shifty"};
    GeneratorParam<int> wtarget{"wtarget", 512, 2, 4096};
    GeneratorParam<int> htarget{"htarget", 512, 2, 4096};
    Output<Buffer<float>> fftIn{"fftIn", 4};

    void generate() {

        //Input
        Func input_func("in");
        Func paddedInput("paddedInput");
        paddedInput = repeat_image( constant_exterior(input, 0.f), 0, wtarget, 0, htarget);
        input_func(x, y, c) = paddedInput( x + shiftx, y + shifty, c );

        //Warping
        fftIn(k, x, y, c) = fft2_r2c(input_func, (int)wtarget, (int)htarget)(k, x, y, c);
    }

    void schedule() {
        if (auto_schedule) {
            return;
        }

        //Allow for arbitrary strides
        //input.set_stride(0, Expr());
        //fftIn.output_buffer().set_stride(0, Expr()); 
    }
};

HALIDE_REGISTER_GENERATOR(fft2_r2c_gen, fftR2CImg);
