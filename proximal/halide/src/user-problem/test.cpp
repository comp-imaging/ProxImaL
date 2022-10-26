#include <HalideBuffer.h>

#include <iostream>

#include "halide_image_io.h"
#include "ladmm-runtime.h"
#include "problem-config.h"

using Halide::Runtime::Buffer;
using Halide::Tools::load_and_convert_image;
using proximal::runtime::ladmmSolver;

namespace {

constexpr auto W = problem_config::input_width;
constexpr auto H = problem_config::input_height;

#ifndef RAW_IMAGE_PATH
#error Path to the raw image must be defined with -DRAW_IMAGE_PATH="..." in the compile command.
#endif

constexpr char raw_image_path[]{RAW_IMAGE_PATH};

constexpr bool verbose = true;

}  // namespace

int
main() {
    Buffer<float> raw_image = load_and_convert_image(raw_image_path);

    raw_image.add_dimension();
    Buffer<const float> normalized = std::move(raw_image);

    const auto max_n_iter = 50;
    const auto [error_code, denoised, r, s, eps_pri, eps_dual] =
        ladmmSolver(normalized, max_n_iter);

    // TODO(Antony): use std::ranges::zip_view
    for (size_t i = 0; i < r.size(); i++) {
        const bool converged = (r[i] < eps_pri[i]) && (s[i] < eps_dual[i]);

        std::cout << "{r, eps_pri, s, eps_dual}[" << i << "] = " << r[i] << '\t' << eps_pri[i]
                  << '\t' << s[i] << '\t' << eps_dual[i] << (converged ? "\tconverged" : "")
                  << '\n';
    }

    std::cout << "Top-left pixel = " << denoised(0, 0, 0) << '\n';

    Buffer<float> output = std::move(denoised);
    Halide::Tools::convert_and_save_image(output, "denoised.png");

    return 0;
}