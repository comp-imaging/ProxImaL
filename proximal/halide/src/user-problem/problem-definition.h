#pragma once
#include <Halide.h>

#include <utility>

using namespace Halide;

#include "prior_transforms.h"
#include "problem-config.h"
#include "problem-interface.h"
#include "prox_operators.h"
#include "prox_parameterized.h"

namespace problem_definition {

using proximal::prox::ParameterizedProx;

/** Transform between variables z and u.
 *
 * This is a demonstration of how ProxImaL should generate Halide code for the
 * (L-)ADMM solver. In this example, the end-user defines the TV-regularizaed
 * denoising problem with non-negative constraints in Proximal language. Then,
 * the ProxImaL-Gen module exports the following functions automatically.
 *
 * Currently (Sept, 2022), ProxImaL has not yet implemented Halide-codegen
 * feature in ProxImal. Here, we manualy craft the transform in Halide code as
 * an example.
 */
struct Transform {
    constexpr static auto width = problem_config::output_width;
    constexpr static auto height = problem_config::output_height;
    constexpr static auto N = problem_config::psi_size;

    /** Compute dx, dy of a two dimensional image with c number of channels. */
    FuncTuple<N> forward(const Func& z) {
        /* Begin code-generation */
        return {K_grad_mat(z, width, height), z};
        /* End code-generation */
    }

    Func adjoint(const FuncTuple<N>& u) {
        /* Begin code-generation */
        const Func z0 = KT_grad_mat(u[0], width, height);
        const auto& z1 = u[1];
        Func s;
        s(x, y, c) = z0(x, y, c) + z1(x, y, c);
        return s;
        /* End code-generation */
    }
} K;

#if __cplusplus >= 202002L

// If C++20 is supported, validate the transform structure.
static_assert(LinOpGrah<Transform>);
#endif

/** List of functions in set Omega, after problem splitting by ProxImaL. */
const ParameterizedProx omega_fn{
    /* Begin code-generation */
    /* . prox = */ [](const Func& v, const Expr& rho) -> Func { return proxSumsq(v, rho, "vhat"); }
    /* .alpha = 1.0f, */
    /* .beta = 1.0f, */
    /* .gamma = 1.0f, */
    /* ._c = 0.0f, */
    /* .d = 0.0f, */
    /* .n_dim = 3, */
    /* End code-generation */
};

#if __cplusplus >= 202002L

// If C++20 is supported, validate the transform structure.
static_assert(Prox<ParameterizedProx>);
#endif

/** List of functions in set Psi, after problem splitting by ProxImaL.
 *
 * Note(Antony): these proximal functions can be parameterized with
 * proximal::prox::ParameterizedProx . But first the class needs to be generalized to 4D signal.
 *
 * TODO(Antony): Use C++20 reference designator to initialize fields. This requires Halide >= 14.0.
 */
const std::array<ParameterizedProx, problem_config::psi_size> psi_fns{
    /* Begin code generation */
    ParameterizedProx{/* .prox = */ [](const Func& u, const Expr& theta) -> Func {
                          using problem_config::output_width;
                          using problem_config::output_height;

                          return proxIsoL1(u, output_width, output_height, theta);
                      },
                      /* .alpha = */ 0.1f,
                      /* .beta = */ 1.0f,
                      /* .gamma = */ 0.0f,
                      /* ._c = */ 0.0f,
                      /* .d = */ 0.0f,
                      /*.n_dim = */ 4},
    ParameterizedProx{
        /* .prox = */ [](const Func& u, const Expr&) -> Func { return proxNonneg<3>(u); }
        /* .alpha = 1.0f, */
        /* .beta = 1.0f, */
        /* .gamma = 0.0f, */
        /* ._c = 0.0f, */
        /* .d = 0.0f, */
        /* .n_dim = 3 */
    }
    /* End code-generation */
};

}  // namespace problem_definition