#include <HalideBuffer.h>

#include <iostream>

#include "halide_image_io.h"
#include "ladmm_iter.h"
#include "problem-config.h"

using Halide::Runtime::Buffer;
using Halide::Tools::load_and_convert_image;

namespace {

constexpr auto W = problem_config::input_width;
constexpr auto H = problem_config::input_height;

#ifndef RAW_IMAGE_PATH
#error Path to the raw image must be defined with -DRAW_IMAGE_PATH="..." in the compile command.
#endif

constexpr char raw_image_path[]{RAW_IMAGE_PATH};

constexpr bool verbose = true;

/** Runtime function to call (L-)ADMM, with early termination.
 *
 * Halide being a non-Turing complete language, is unable to dynamically
 * terminate a while-loop. Therefore, we generate the Halide-optimized AOT
 * pipeline with a fixed (e.g. 10) iterations, returning the convergence
 * metrics at iteration #10.
 *
 * Then, we check the convergence criteria, and terminate the for-loop when the
 * criteria are met. Otherwise, repeat for another (10) iterations.
 *
 * Reference: https://stackoverflow.com/a/33472074
 */
std::tuple<Buffer<float>, std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
ladmmSolver(Buffer<const float>& input, const size_t iter_max = 100, const float eps_abs = 1e-3,
            const float eps_rel = 1e-3) {
    Buffer<float> v(W, H, 1);
    Buffer<float> z0(W, H, 1, 2);
    Buffer<float> z1(W, H, 1);
    Buffer<float> u0(W, H, 1, 2);
    Buffer<float> u1(W, H, 1);

    // Set zeros
    for (auto* buf : {&v, &z0, &z1, &u0, &u1}) {
        buf->fill(0.0f);
    }

    Buffer<float> z0_new(W, H, 1, 2);
    Buffer<float> z1_new(W, H, 1);
    Buffer<float> u0_new(W, H, 1, 2);
    Buffer<float> u1_new(W, H, 1);

    Buffer<float> v_new(W, H, 1);

    std::vector<float> r(iter_max);
    std::vector<float> s(iter_max);
    std::vector<float> eps_pri(iter_max);
    std::vector<float> eps_dual(iter_max);

    for (size_t i = 0; i < iter_max; i++) {
        auto _r = Buffer<float>::make_scalar(r.data() + i);
        auto _s = Buffer<float>::make_scalar(s.data() + i);
        auto _eps_pri = Buffer<float>::make_scalar(eps_pri.data() + i);
        auto _eps_dual = Buffer<float>::make_scalar(eps_dual.data() + i);

        const auto error = ladmm_iter(input, v, z0, z1, u0, u1, v_new, z0_new, z1_new, u0_new,
                                      u1_new, _r, _s, _eps_pri, _eps_dual);

        if (error) {
            std::cout << "Error: " << error << '\n';
            return {};
        }

        // Terminate the algorithm early, if optimal solution is reached.
        const bool converged = (r[i] < eps_pri[i]) && (s[i] < eps_dual[i]);
        if (converged) {
            for (auto* v : {&r, &s, &eps_pri, &eps_dual}) {
                v->resize(i + 1);
            }
            break;
        }

        if (i != iter_max - 1) {
            // This iteration's v_new becomes current v in the next iteration.
            std::swap(v, v_new);
            std::swap(u0, u0_new);
            std::swap(u1, u1_new);
            std::swap(z0, z0_new);
            std::swap(z1, z1_new);
        }
    }

    return {v_new, r, s, eps_pri, eps_dual};
}

}  // namespace

int
main() {
    Buffer<float> raw_image = load_and_convert_image(raw_image_path);

    raw_image.add_dimension();
    Buffer<const float> normalized = std::move(raw_image);

    const auto max_n_iter = 50;
    const auto [denoised, r, s, eps_pri, eps_dual] = ladmmSolver(normalized, max_n_iter);

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