#include <Halide.h>
using namespace Halide;

#include "linearized-admm.h"
#include "problem-definition.h"

class LinearizedADMMIter : public Generator<LinearizedADMMIter> {
    static constexpr auto W = problem_config::output_width;
    static constexpr auto H = problem_config::output_height;

   public:
    /** User-provided distorted, and noisy image. */
    Input<Buffer<float, 3>> input{"input"};

    /** Initial estimate of the restored image. */
    Input<Buffer<float, 3>> v{"v"};

    // TODO(Antony): How do we determine the number of inputs z_i at run time? Generator::configure() ?
    // Does Buffer<Func[2]> results in terse code? How to set dimensions?
    Input<Buffer<float, 4>> z0{"z0"};
    Input<Buffer<float, 3>> z1{"z1"};

    Input<Buffer<float, 4>> u0{"u0"};
    Input<Buffer<float, 3>> u1{"u1"};

    /** Problem scaling factor.
     *
     * This influences the convergence rate of the (L-)ADMM algorithm.
     */
    GeneratorParam<float> lmb{"lmb", 1.0f, 0.0f, 1e3f};
    GeneratorParam<float> mu{"mu", 1.0f, 0.0f, 1e3f};

    /** Number of (L-)ADMM iterations before computing convergence metrics.
     *
     * This reduces the overhead of convengence check by a factor of n_iter
     * times. The end-users can decide whether to re-run this pipeline for
     * another n_iter times in their own runtime.
     */
    GeneratorParam<uint32_t> n_iter{"n_iter", 1ul, 1ul, 500ul};

    /** Optimal solution, after a hard termination after iterating for n_iter
     * times. */
    Output<Buffer<float, 3>> v_new{"v_new"};

    // TODO(Antony): How do we figure out the number of outputs z_i at run time? configure() ?
    Output<Buffer<float, 4>> z0_new{"z0_new"};
    Output<Buffer<float, 3>> z1_new{"z1_new"};

    Output<Buffer<float, 4>> u0_new{"u0_new"};
    Output<Buffer<float, 3>> u1_new{"u1_new"};

    // Convergence metrics
    Output<float> r{"r"};  //!< Primal residual
    Output<float> s{"s"};  //!< Dual residual
    Output<float> eps_pri{"eps_pri"};
    Output<float> eps_dual{"eps_dual"};

    void generate() {
        using problem_config::psi_size;
        using problem_definition::K;
        using problem_definition::omega_fn;
        using problem_definition::psi_fns;

        // How to generalize z_i as inputs?
        static_assert(psi_size == 2);

        std::vector<Func> v_list(n_iter);

        std::vector<FuncTuple<psi_size>> z_list(n_iter);
        std::vector<FuncTuple<psi_size>> u_list(n_iter);

        using problem_config::input_height;
        using problem_config::input_size;
        using problem_config::input_width;
        using problem_config::output_height;
        using problem_config::output_size;
        using problem_config::output_width;

        const RDom input_dimensions{0, input_width, 0, input_height, 0, 1};
        const RDom output_dimensions{0, output_width, 0, output_height, 0, 1, 0, 2};

        for (size_t i = 0; i < n_iter; i++) {
            const Func& v_prev =
                (i == 0) ? v : v_list[i - 1];
            const FuncTuple<psi_size>& z_prev =
                (i == 0) ? FuncTuple<psi_size>{z0, z1} : z_list[i - 1];
            const FuncTuple<psi_size>& u_prev =
                (i == 0) ? FuncTuple<psi_size>{u0, u1} : u_list[i - 1];

            std::tie(v_list[i], z_list[i], u_list[i]) = algorithm::linearized_admm::iterate(
                v_prev, z_prev, u_prev, K, omega_fn, psi_fns, lmb, mu, input);
        }

        const auto& z_prev = (n_iter > 1) ? *(z_list.rbegin() + 1) : FuncTuple<psi_size>{z0, z1};
        const auto [_r, _s, _eps_pri, _eps_dual] = algorithm::linearized_admm::computeConvergence(
            v_list.back(), z_list.back(), u_list.back(), z_prev, K, lmb, input_size,
            input_dimensions, output_size,
            output_dimensions);

        // Export data
        v_new = v_list.back();
        std::tie(z0_new, z1_new) = std::make_pair(z_list.back()[0], z_list.back()[1]);
        std::tie(u0_new, u1_new) = std::make_pair(u_list.back()[0], u_list.back()[1]);
        r() = _r;
        s() = _s;
        eps_pri() = _eps_pri;
        eps_dual() = _eps_dual;
    }

    /** Inform Halide of the fixed input and output image sizes. */
    void setBounds() {
        for (auto* a : {&input, &v, &z1, &u1}) {
            a->dim(0).set_bounds(0, W);
            a->dim(1).set_bounds(0, H);
            a->dim(2).set_bounds(0, 1);
        }

        for (auto* a : {&z0, &u0}) {
            a->dim(0).set_bounds(0, W);
            a->dim(1).set_bounds(0, H);
            a->dim(2).set_bounds(0, 1);
            a->dim(3).set_bounds(0, 2);
        }

        for (auto* a : {&v_new, &z1_new, &u1_new}) {
            a->dim(0).set_bounds(0, W);
            a->dim(1).set_bounds(0, H);
            a->dim(2).set_bounds(0, 1);
        }

        for (auto* a : {&z0_new, &u0_new}) {
            a->dim(0).set_bounds(0, W);
            a->dim(1).set_bounds(0, H);
            a->dim(2).set_bounds(0, 1);
            a->dim(3).set_bounds(0, 2);
        }
    }

    void scheduleForCPU() {
        const auto vec_width = natural_vector_size<float>();
        v_new.reorder(c, x, y).vectorize(x, vec_width).parallel(y);
        u0_new.reorder(c, k, x, y).vectorize(x, vec_width).parallel(y).unroll(k, 2);
        z0_new.reorder(c, k, x, y).vectorize(x, vec_width).parallel(y).unroll(k, 2);

        u1_new.reorder(c, x, y).vectorize(x, vec_width).parallel(y);
        z1_new.reorder(c, x, y).vectorize(x, vec_width).parallel(y);
    }

    void schedule() {
        setBounds();

        if (using_autoscheduler()) {
            // Estimate the image sizes of the inputs.
            for (auto* a : {&input, &v, &z1, &u1}) {
                a->set_estimates({{0, W}, {0, H}, {0, 1}});
            }

            for (auto* a : {&z0, &u0}) {
                a->set_estimates({{0, W}, {0, H}, {0, 1}, {0, 2}});
            }

            // Estimate the image sizes of the outputs.
            for (auto* a : {&v_new, &z1_new, &u1_new}) {
                a->set_estimates({{0, W}, {0, H}, {0, 1}});
            }

            for (auto* a : {&z0_new, &u0_new}) {
                a->set_estimates({{0, W}, {0, H}, {0, 1}, {0, 2}});
            }
            return;
        }

        // Schedule for CPU
        return scheduleForCPU();
    }
};

HALIDE_REGISTER_GENERATOR(LinearizedADMMIter, ladmm_iter);
