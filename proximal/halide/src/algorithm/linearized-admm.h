#include <utility>

#include "Halide.h"

// Back-porting of the <range> library from C++20 standard.
// Provides zip_view
#include "problem-interface.h"
#include "range/v3/algorithm/transform.hpp"
#include "range/v3/view/zip.hpp"
#include "vars.h"

using namespace Halide;
using ranges::zip_view;

namespace utils {
Func
normSquared(const Func& v, const RDom& r) {
    Func sumsq{"sumsq"};
    sumsq() = 0.0f;

    if (v.dimensions() == 4) {
        sumsq() += v(r.x, r.y, r.z, r.w) * v(r.x, r.y, r.z, r.w);
    } else {  // n_dim == 3
        sumsq() += v(r.x, r.y, r.z) * v(r.x, r.y, r.z);
    }

    return sumsq;
}

template <size_t N>
Func
normSquared(const FuncTuple<N>& v, const RDom& r) {
    Func sumsq{"sumsq"};
    sumsq() = 0.0f;

    for (const auto& _v : v) {
        if (_v.dimensions() == 4) {
            sumsq() += _v(r.x, r.y, r.z, r.w) * _v(r.x, r.y, r.z, r.w);
        } else {  // n_dim == 3
            sumsq() += _v(r.x, r.y, r.z) * _v(r.x, r.y, r.z);
        }
    }

    return sumsq;
}
}  // namespace utils

namespace algorithm {
namespace linearized_admm {

template <size_t N, LinOpGraph G, Prox P, Prox P2>
std::tuple<Func, FuncTuple<N>, FuncTuple<N>>
iterate(const Func& v, const FuncTuple<N>& z, const FuncTuple<N>& u, G& K, const P& omega_fn,
        std::array<P2, N> psi_fns, const Expr& lmb, const Expr& mu, const Func& b) {
    using Vars = std::vector<Var>;

    // Update v
    Func v_new{"v_new"};
    {
        const FuncTuple<N> Kv = K.forward(v);

        FuncTuple<N> Kvzu;
        ranges::transform(zip_view{Kv, z, u, psi_fns}, Kvzu.begin(), [=](const auto& args) -> Func {
            const auto& [_Kv, _z, _u, prox] = args;

            // If z_i is a 4D matrix, make it so. Otherwise, assume a 3D data.
            const auto vars = (prox.n_dim == 4) ? Vars{x, y, c, k} : Vars{x, y, c};

            Func _Kvzu{"Kvzu"};
            _Kvzu(vars) = _Kv(vars) - _z(vars) + _u(vars);

            return _Kvzu;
        });

        const Func v2 = K.adjoint(Kvzu);
        Func v3;
        v3(x, y, c) = v(x, y, c) - (mu / lmb) * v2(x, y, c);

        v_new = omega_fn(v3, 1.0f / mu, b);
    }

    // Update z_i for i = 0..N .
    const FuncTuple<N> Kv2 = K.forward(v_new);
    FuncTuple<N> z_new;
    ranges::transform(zip_view{Kv2, u, psi_fns}, z_new.begin(), [=](const auto& args) -> Func {
        // We resort to the ranges::transform() syntax because MacOS+clang14
        // refuses to reference z_new using the structured binding syntax
        // `auto&& [...]`. Instead, the compiler makes a copy of z_new, so we
        // were unable to update the z value in the iteration.
        const auto& [_Kv2, _u, prox] = args;
        const auto vars = (prox.n_dim == 4) ? Vars{x, y, c, k} : Vars{x, y, c};

        Func Kv_u{"Kv_u"};
        Kv_u(vars) = _Kv2(vars) + _u(vars);

        return prox(Kv_u, 1.0f / lmb);
    });

    // Update u.
    FuncTuple<N> u_new;
    ranges::transform(zip_view{u, Kv2, z_new, psi_fns}, u_new.begin(), [=](const auto& args) -> Func {
        const auto& [_u, _Kv, _z, prox] = args;
        const auto vars = (prox.n_dim == 4) ? Vars{x, y, c, k} : Vars{x, y, c};

        Func _u_new{"u_new"};
        _u_new(vars) = _u(vars) + _Kv(vars) - _z(vars);

        return _u_new;
    });

    return {v_new, z_new, u_new};
}

template <size_t N, LinOpGraph G>
std::tuple<Expr, Expr, Expr, Expr>
computeConvergence(const Func& v, const FuncTuple<N>& z, const FuncTuple<N>& u,
                   const FuncTuple<N>& z_prev, G& K, const float lmb, const uint32_t input_size,
                   const RDom& input_dimensions, const uint32_t output_size,
                   const RDom& output_dimensions, const float eps_abs = 1e-3f,
                   const float eps_rel = 1e-3f) {
    using Vars = std::vector<Var>;

    const FuncTuple<N> Kv = K.forward(v);
    const Func KTu = K.adjoint(u);

    // Compute primal residual
    FuncTuple<N> r;
    ranges::transform(zip_view{Kv, z}, r.begin(), [](const auto& args) -> Func {
        const auto& [_Kv, _z] = args;
        const auto vars = (_z.dimensions() == 4) ? Vars{x, y, c, k} : Vars{x, y, c};

        Func _r{"r"};
        _r(vars) = _Kv(vars) - _z(vars);
        return _r;
    });

    FuncTuple<N> ztmp;
    ranges::transform(zip_view{z, z_prev}, ztmp.begin(), [=](const auto& args) -> Func {
        const auto& [_z, _z_prev] = args;
        const auto vars = (_z.dimensions() == 4) ? Vars{x, y, c, k} : Vars{x, y, c};

        Func _ztmp{"z_diff"};
        _ztmp(vars) = (_z(vars) - _z_prev(vars)) / lmb;

        return _ztmp;
    });

    // Compute dual residual
    const Func s = K.adjoint(ztmp);

    // Compute convergence criteria
    using utils::normSquared;

    const Func Kv_norm = normSquared(Kv, output_dimensions);
    const Func z_norm = normSquared(z, output_dimensions);
    const Expr eps_pri = eps_rel * sqrt(max(Kv_norm(), z_norm())) + std::sqrt(float(output_size)) * eps_abs;

    const Func KTu_norm = normSquared(KTu, input_dimensions);
    const Expr eps_dual =
        sqrt(KTu_norm()) * eps_rel / (1.0f / lmb) + std::sqrt(float(input_size)) * eps_abs;

    const Func r_norm = normSquared(r, output_dimensions);
    const Func s_norm = normSquared(s, input_dimensions);
    return {sqrt(r_norm()), sqrt(s_norm()), eps_pri, eps_dual};
}
}  // namespace linearized_admm
}  // namespace algorithm
