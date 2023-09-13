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

Func fft2_r2c(Func input, int W, int H, const Target& target) {

    //Local vars
    Var h("h"), l("l"), m("m"); 

    Fft2dDesc fwd_desc;

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

    Input<Buffer<float, 3>> input{"input"};
    Input<int> shiftx{"shiftx"};
    Input<int> shifty{"shifty"};
    GeneratorParam<int> wtarget{"wtarget", 512, 2, 4096};
    GeneratorParam<int> htarget{"htarget", 512, 2, 4096};
    Output<Buffer<float, 4>> fftIn{"fftIn"};

    void generate() {

        /** 
         * Circular boundary condition with zero padding. 
         *
         * If the input image dimensions are smaller than the FFT dimensions,
         * assume the end user requests a zero-value boundary condition in the
         * 2D convolution algorithm. Make it so.
         *
         * TODO(Antony): remove the redundant logic "repeat_image(...)" because the
         * circular boundary condition is already implied by the FFT algorithm.
        */
        Func input_func("in");
        Func paddedInput("paddedInput");
        paddedInput = repeat_image( constant_exterior(input, 0.f), {{0, wtarget}, {0, htarget}});

        /** Option to shift or center 2D convolution kernel relative to the
         * input image by a user-specified offset. */
        input_func(x, y, c) = paddedInput( x + shiftx, y + shifty, c );

        const auto transformed = fft2_r2c(input_func, (int)wtarget, (int)htarget, get_target());

        // Crop the FFT transformed signal by the user-defined output dimensions.
        fftIn(k, x, y, c) = transformed(k, x, y, c);
    }

    void schedule() {
        assert(!using_autoscheduler() && "Auto-scheduler not required for FFT interface");

        //Allow for arbitrary strides
        //input.set_stride(0, Expr());
        //fftIn.output_buffer().set_stride(0, Expr()); 
    }
};

HALIDE_REGISTER_GENERATOR(fft2_r2c_gen, fftR2CImg);
