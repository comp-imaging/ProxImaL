#include <utility>

#include "Halide.h"

// Back-porting of the <range> library from C++20 standard.
// Provides zip_view
#include "problem-interface.h"
#include "range/v3/view/zip.hpp"
#include "vars.h"

using namespace Halide;
using ranges::zip_view;

namespace utils {
inline Expr
norm(const Func& v, const RDom r) {
    // TODO(Antony): n_channels.
    if (v.dimensions() == 4) {
        return sqrt(sum(v(r.x, r.y, r.z, r.w) * v(r.x, r.y, r.z, r.w)));
    } else {  // n_dim == 3
        return sqrt(sum(v(r.x, r.y, r.z) * v(r.x, r.y, r.z)));
    }
}

template <size_t N>
inline Expr
norm(const FuncTuple<N>& v, const RDom r) {
    Expr s = 0;

    // TODO(Antony): item specific n-dimensions
    for (auto&& _v : v) {
        if (_v.dimensions() == 4) {
            s += sum(_v(r.x, r.y, r.z, r.w) * _v(r.x, r.y, r.z, r.w));
        } else {  // n_dim == 3
            s += sum(_v(r.x, r.y, r.z) * _v(r.x, r.y, r.z));
        }
    }

    return sqrt(s);
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
        for (auto&& [_Kvzu, _Kv, _z, _u, prox] : zip_view(Kvzu, Kv, z, u, psi_fns)) {
            // Name the variable
            _Kvzu = Func{"Kvzu"};

            // If z_i is a 4D matrix, make it so. Otherwise, assume a 3D data.
            const auto vars = (prox.n_dim == 4) ? Vars{x, y, c, k} : Vars{x, y, c};

            _Kvzu(vars) = _Kv(vars) - _z(vars) + _u(vars);
        }

        const Func v2 = K.adjoint(Kvzu);
        Func v3;
        v3(x, y, c) = v(x, y, c) - (mu / lmb) * v2(x, y, c);

        v_new = omega_fn(v3, 1.0f / mu, b);
    }

    // Update z_i for i = 0..N .
    const FuncTuple<N> Kv2 = K.forward(v_new);
    FuncTuple<N> z_new;
    {
        for (auto&& [_z_new, _Kv2, _u, prox] : zip_view(z_new, Kv2, u, psi_fns)) {
            Func Kv_u{"Kv_u"};

            const auto vars = (prox.n_dim == 4) ? Vars{x, y, c, k} : Vars{x, y, c};

            Kv_u(vars) = _Kv2(vars) + _u(vars);

            _z_new = prox(Kv_u, 1.0f / lmb);
        }
    }

    // Update u.
    FuncTuple<N> u_new;
    for (auto&& [_u_new, _u, _Kv, _z, prox] : zip_view(u_new, u, Kv2, z_new, psi_fns)) {
        const auto vars = (prox.n_dim == 4) ? Vars{x, y, c, k} : Vars{x, y, c};

        _u_new(vars) = _u(vars) + _Kv(vars) - _z(vars);
    }

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
    {
        for (auto&& [_r, _Kv, _z] : zip_view(r, Kv, z)) {
            const auto vars = (_z.dimensions() == 4) ? Vars{x, y, c, k} : Vars{x, y, c};
            _r(vars) = _Kv(vars) - _z(vars);
        }
    }

    FuncTuple<N> ztmp;
    {
        for (auto&& [_ztmp, _z, _z_prev] : zip_view(ztmp, z, z_prev)) {
            const auto vars = (_z.dimensions() == 4) ? Vars{x, y, c, k} : Vars{x, y, c};
            _ztmp(vars) = (_z(vars) - _z_prev(vars)) / lmb;
        }
    }

    // Compute dual residual
    const Func s = K.adjoint(ztmp);

    // Compute convergence criteria
    using utils::norm;
    Expr eps_pri = eps_rel * max(norm(Kv, output_dimensions), norm(z, output_dimensions)) +
                   output_size * eps_abs;

    Expr eps_dual = norm(KTu, input_dimensions) * eps_rel / (1.0f / lmb) +
                    std::sqrt(float(input_size)) * eps_abs;

    return {norm(r, output_dimensions), norm(s, input_dimensions), eps_pri, eps_dual};
}
}  // namespace linearized_admm
}  // namespace algorithm