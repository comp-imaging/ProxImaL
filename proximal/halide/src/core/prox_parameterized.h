#pragma once

#include <optional>
#include <utility>

#include "Halide.h"
#include "vars.h"

namespace proximal {
namespace prox {

using namespace Halide;

/** Wrapper on the prox function to handle alpha etc.
 *
 * Solve the proximal of the parameterized form
 * g(x) = alpha * f(beta * x - b) + c^T * x + gamma x^T * x
 * with variable transformation.
 *
 * When all parameters are set at default values, e.g. alpha == 1.0f, Halide's
 * symbolic simplifications feature will eliminate all redundant
 * computations, e.g. 1.0f * x => x. We utilize this feature to express
 * parameterized proximal equations verbatim.
 */
struct ParameterizedProx {
    using ProxFn = Func (*)(const Func&, const Expr&);
    const ProxFn prox;
    const float alpha = 1.0f;
    const float beta = 1.0f;
    const float gamma = 0.0f;
    const float _c = 0.0f;
    const float d = 0.0f;
    const size_t n_dim = 3;

    /** Forward transform.
     *
     * Map variables v and rho into new domain, so we can compute prox with the
     * proximal operator f.
     */
    std::pair<Func, Expr> forward(const Func& v, const Expr& rho,
                                  const std::optional<Func>& _b = std::nullopt) const {
        const Expr rho_hat = (rho + gamma * 2) / (alpha * beta * beta);

        using Vars = std::vector<Var>;
        const Vars vars = (n_dim == 4) ? Vars{x, y, c, k} : Vars{x, y, c};

        Func v_hat;
        if (_b) {
            const auto& b = _b.value();
            v_hat(vars) = (v(vars) * rho - _c) * beta / (rho + gamma * 2) - b(vars);
        } else {
            v_hat(vars) = (v(vars) * rho - _c) * beta / (rho + gamma * 2);
        }

        return {v_hat, rho_hat};
    }

    /** Backward transform.
     * Map variable u_hat back to u.
     */
    Func backward(const Func& u_hat, const std::optional<Func>& _b = std::nullopt) const {
        using Vars = std::vector<Var>;
        const Vars vars = (n_dim == 4) ? Vars{x, y, c, k} : Vars{x, y, c};

        Func u;
        if (_b) {
            const auto& b = _b.value();
            u(vars) = (u_hat(vars) + b(vars)) / beta;
        } else {
            u(vars) = u_hat(vars) / beta;
        }

        return u;
    }

    /** Compute the parameterized proximal operation. */
    Func operator()(const Func& v, const Expr& rho,
                    const std::optional<Func> _b = std::nullopt) const {
        const auto [v_hat, rho_hat] = forward(v, rho, _b);

        // Note(Antony): In `halide/core/prox_operators.h`, all proximal functions are defined as
        // theta = 1 / rho. Make it so.
        const Func u_hat = prox(v_hat, 1.0f / rho_hat);
        return backward(u_hat, _b);
    }
};

}  // namespace prox
}  // namespace proximal