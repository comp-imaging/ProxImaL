#include <utility>

#include "Halide.h"

// Back-porting of the <range> library from C++20 standard.
// Provides zip_view
#include <range/v3/algorithm/transform.hpp>
#include <range/v3/view/zip.hpp>

#include "problem-interface.h"
#include "vars.h"

using namespace Halide;
using ranges::zip_view;

namespace utils {
Func
normSquared(const Func& v, const RDom& r) {
    using halide_vars::x;

    Func sumsq{"sumsq"};
    sumsq(x) = 0.0f;

    if (v.dimensions() == 4) {
        sumsq(x) += v(r.x, r.y, r.z, r.w) * v(r.x, r.y, r.z, r.w);
    } else {  // n_dim == 3
        sumsq(x) += v(r.x, r.y, r.z) * v(r.x, r.y, r.z);
    }

    return sumsq;
}

template <size_t N>
Func
normSquared(const FuncTuple<N>& v, const RDom& r) {
    using halide_vars::x;

    Func sumsq{"sumsq"};
    sumsq(x) = 0.0f;

    for (const auto& _v : v) {
        if (_v.dimensions() == 4) {
            sumsq(x) += _v(r.x, r.y, r.z, r.w) * _v(r.x, r.y, r.z, r.w);
        } else {  // n_dim == 3
            sumsq(x) += _v(r.x, r.y, r.z) * _v(r.x, r.y, r.z);
        }
    }

    return sumsq;
}
}  // namespace utils

namespace algorithm {
namespace pock_chambolle {

template <size_t N, LinOpGraph G, Prox P, Prox P2>
std::tuple<Func, Func, FuncTuple<N>>
iterate(const Func& X, const Func& Xbar, const FuncTuple<N>& Y, G& K, const P& omega_fn,
        std::array<P2, N> psi_fns, const Expr& sigma, const Expr& tau) {
    using Vars = std::vector<Var>;

    // The variables xyz are taken by image coordinates, but Pock-Chambolle
    // needs the same alphabet for consistency. Use capital XYZ for now.
    using halide_vars::k;
    using halide_vars::x;
    using halide_vars::y;
    using halide_vars::z;

    // Compute z (discarded after one single iteration).
    FuncTuple<N> Z_new;
    {
        const FuncTuple<N> Kxbar = K.forward(Xbar);

        ranges::transform(
            zip_view{Kxbar, Y, psi_fns}, Z_new.begin(), [=](const auto& args) -> Func {
                const auto& [_Kxbar, _y, prox] = args;

                // If z_i is a 4D matrix, make it so. Otherwise, assume a 3D data.
                const auto vars = (prox.n_dim == 4) ? Vars{x, y, z, k} : Vars{x, y, z};

                Func _Z_new{"Z"};
                _Z_new(vars) = _y(vars) + sigma * _Kxbar(vars);

                return _Z_new;
            });
    }

    // Update Y
    FuncTuple<N> Y_new;
    ranges::transform(zip_view{Z_new, psi_fns}, Y_new.begin(), [=](const auto& args) -> Func {
        const auto& [_Z_new, prox] = args;

        // If z_i is a 4D matrix, make it so. Otherwise, assume a 3D data.
        const auto vars = (prox.n_dim == 4) ? Vars{x, y, z, k} : Vars{x, y, z};

        Func Z_scaled{"Z_scaled"};
        Z_scaled(vars) = _Z_new(vars) / sigma;

        // Moreau identity.
        Func _Y{"Y"};
        _Y(vars) = _Z_new(vars) - sigma * prox(Z_scaled, sigma)(vars);

        return _Y;
    });

    // Update X
    Func X_new;
    {
        const Func KTy = K.adjoint(Y_new);
        Func Xtmp{"Xtmp"};
        Xtmp(x, y, z) = X(x, y, z) - tau * KTy(x, y, z);

        X_new = omega_fn.prox(Xtmp, 1.0f / tau);
    }

    // Update vbar
    Func Xbar_new{"Xbar_new"};
    constexpr float theta = 1.0f;
    Xbar_new(x, y, z) = X_new(x, y, z) + theta * (X_new(x, y, z) - X(x, y, z));

    return {X_new, Xbar_new, Y_new};
}

template <size_t N, LinOpGraph G>
std::tuple<Expr, Expr, Expr, Expr>
computeConvergence(const Func& X, const Func& X_prev, const FuncTuple<N>& Y,
                   const FuncTuple<N>& Y_prev, G& K, const Expr& sigma, const uint32_t input_size,
                   const RDom& input_dimensions, const uint32_t output_size,
                   const RDom& output_dimensions, const float eps_abs, const float eps_rel) {
    using Vars = std::vector<Var>;
    using halide_vars::k;
    using halide_vars::x;
    using halide_vars::y;
    const Var& c = halide_vars::z;

    // Compute primal residual
    constexpr bool strict = false;
    static_assert(!strict, "Fatal: not exactly the residual-based convergence criterion.");
    Func r{"r"};
    {
        using halide_vars::z;
        r(x, y, z) = X(x, y, z) - X_prev(x, y, z);
    }

    // Compute dual residual
    FuncTuple<N> Y_diff;
    ranges::transform(zip_view{Y, Y_prev}, Y_diff.begin(), [=](const auto& args) -> Func {
        using halide_vars::z;

        const auto& [_Y, _Y_prev] = args;
        const auto vars = (_Y.dimensions() == 4) ? Vars{x, y, c, k} : Vars{x, y, c};

        Func _Y_diff{"Y_diff"};
        _Y_diff(vars) = _Y(vars) - _Y_prev(vars);

        return _Y_diff;
    });

    const Func s = K.adjoint(Y_diff);

    const Func KTy = K.adjoint(Y);

    // Compute convergence criteria
    using utils::normSquared;

    const Func X_norm = normSquared(X, input_dimensions);
    const Expr eps_pri = eps_rel * sqrt(X_norm(0)) + std::sqrt(float(output_size)) * eps_abs;

    const Func KTy_norm = normSquared(KTy, input_dimensions);
    const Expr eps_dual = sqrt(KTy_norm(0)) * eps_rel + std::sqrt(float(input_size)) * eps_abs;

    const Func r_norm = normSquared(r, input_dimensions);
    const Func s_norm = normSquared(s, input_dimensions);
    return {sqrt(r_norm(0)), sqrt(s_norm(0)), eps_pri, eps_dual};
}
}  // namespace pock_chambolle
}  // namespace algorithm
