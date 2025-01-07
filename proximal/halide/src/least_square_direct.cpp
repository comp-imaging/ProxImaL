////////////////////////////////////////////////////////////////////////////////
// IFFT2 transform using halide
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using Halide::BoundaryConditions::constant_exterior;
using Halide::BoundaryConditions::repeat_image;

#include "fft/fft.h"

namespace {

enum component_t : int32_t { RE = 0, IM = 1 };

class least_square_direct_gen : public Generator<least_square_direct_gen> {
   public:
    Input<Buffer<float, 3>> input{"input"};
    Input<float> rho{"rho"};
    Input<Buffer<float, 3>> offset{"offset"};
    Input<Buffer<float, 4>> freq_diag{"freq_diag"};
    Input<uint64_t> input_hash{"offset_hash"};

    Output<Buffer<float, 3>> output{"output"};

    GeneratorParam<int> wtarget{"wtarget", 512, 2, 1024};
    GeneratorParam<int> htarget{"htarget", 512, 2, 1024};
    GeneratorParam<bool> ignore_offset{"ignore_offset", false};

    void generate() {
        const int W = wtarget;
        const int H = htarget;

        // Zeropadding to fit the inputs to the user-provided FFT target width and height.
        padded_input = constant_exterior(input, 0.f, {{0, input.width()}, {0, input.height()}});

        padded_offset(x, y, c) = constant_exterior(offset, 0.f, {{0, W}, {0, H}})(x, y, c);

        // If the input buffer (e.g. blur kernel) is smaller than the FFT
        // target width, shift the image to the center.
        const Expr shiftx = select(input.width() < W, (W - input.width()) / 2, 0);
        const Expr shifty = select(input.height() < H, (H - input.height()) / 2, 0);
        shifted_input(x, y, c) = padded_input(x + shiftx, y + shifty, c);

        // Forward DFT
        Fft2dDesc fwd_desc{};
        fwd_desc.parallel = true;
        f_input_tmp = fft2d_r2c(shifted_input, W, H, target, fwd_desc);

        f_input_cached(x, y, k, c) =
            memoize_tag(mux(k, {f_input_tmp(x, y, c).re(), f_input_tmp(x, y, c).im()}), input_hash);
        f_input(x, y, c) = {f_input_cached(x, y, RE, c), f_input_cached(x, y, IM, c)};

        f_offset = fft2d_r2c(padded_offset, W, H, target, fwd_desc);

        // Cast freq_diag from pair<float> to std:::complex<float>
        diag(x, y, c) = {freq_diag(RE, x, y, c), freq_diag(IM, x, y, c)};

        if (ignore_offset) {
            weighted_average(x, y, c) =
                f_input(x, y, c) / diag(x, y, c);
        } else {
            weighted_average(x, y, c) =
                (f_input(x, y, c) / rho + f_offset(x, y, c)) / (diag(x, y, c) / rho + 1.0f);
        }

        // Inverse DFT
        Fft2dDesc inv_desc{};
        inv_desc.gain = 1.0f / (W * H);
        inv_desc.parallel = true;

        inversed = fft2d_c2r(weighted_average, W, H, target, inv_desc);
        output = inversed;
    }

    void validateDimensions() {
        // Number of channels must match.
        const auto n_channels = input.dim(2).extent();
        offset.dim(2).set_extent(n_channels);
        freq_diag.dim(3).set_extent(n_channels);
        output.dim(2).set_extent(n_channels);

        // Frequency domain diagonals size must match FFT target size
        freq_diag.dim(0).set_extent(2);
        freq_diag.dim(1).set_extent(wtarget);
        //freq_diag.dim(2).set_extent(htarget);

        // All buffers must start with index zero
        input.dim(0).set_min(0);
        offset.dim(0).set_min(0);
        freq_diag.dim(0).set_min(0);
        output.dim(0).set_min(0);

        input.dim(1).set_min(0);
        offset.dim(1).set_min(0);
        freq_diag.dim(1).set_min(0);
        output.dim(1).set_min(0);

        input.dim(2).set_min(0);
        offset.dim(2).set_min(0);
        freq_diag.dim(2).set_min(0);
        output.dim(2).set_min(0);

        freq_diag.dim(3).set_min(0);
    }

    void schedule() {
        assert(!using_autoscheduler() && "Auto-scheduler not possible with manual schedules in FFT");

        validateDimensions();

        const auto vfloat = natural_vector_size<float>();
        output //
        .vectorize(x, vfloat)
        .parallel(y)
        .parallel(c)
        ;

        inversed //
            .compute_root()
            .parallel(c);

        weighted_average //
            .compute_root()
            .vectorize(x, vfloat)
            .parallel(y)
            .parallel(c);

        if(!ignore_offset) {
            f_offset
                .compute_root()  //
                .parallel(c);

            padded_offset
                .compute_root()  //
                .vectorize(x, vfloat)
                .parallel(y)
                .parallel(c);
        }

        // Cache the Fourier-transformed input
        f_input_cached  //
            .compute_root()
            .bound(k, 0, 2)
            .unroll(k)
            .vectorize(x, vfloat)
            .parallel(y)
            .parallel(c)
            .memoize();

        // Compute FFT only when cache is evicted
        f_input_tmp.compute_at(f_input_cached, c);

        shifted_input
            .compute_at(f_input_cached, Var::outermost())  //
            .vectorize(x, vfloat)
            .parallel(y)
            .parallel(c);
    }

   private:
    // coordinates in the space domain
    Var x{"x"}, y{"y"}, c{"c"}, k{"k"};

    Func padded_input{"padded_input"};
    Func shifted_input{"shifted_input"};
    Func padded_offset{"padded_offset"};
    ComplexFunc diag{"diag"};

    ComplexFunc f_offset;

    Func f_input_cached{"f_input_cached"};
    ComplexFunc f_input_tmp;
    ComplexFunc f_input{"f_input"};

    ComplexFunc weighted_average{"weighted_average"};
    Func inversed{"inversed"};
};

}  // namespace
HALIDE_REGISTER_GENERATOR(least_square_direct_gen, least_square_direct);
