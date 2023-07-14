#pragma once
#include <array>

#include "Halide.h"

/** A proximal function takes a data plane/cube and a regularization variable,
 * and returns the proximal value. */
using ProxFn = Halide::Func (*)(const Halide::Func&, const Halide::Expr&);

/** A list of data planes/cubes.
 *
 * ProxImaL implements z = { z_i } with numpy.vstack() and numpy.ravel(). This
 *  destroys image locality. Here, we preserves the image dimensions of each z_i,
 *  using the std::array structure.
 */
template <size_t N>
using FuncTuple = std::array<Halide::Func, N>;

#if __cplusplus >= 202002L

#warning Halide 10.0 is incompatible to C++20. Upgrade to newer versions.
/** A compute graph composed of various LinOps written in Halide.
 *
 * The (L-)ADMM solvers expects the linear mapping between variable z and u, by
 * z_i = K_i * u.
 *
 * The Halide optimized functions for (L-)ADMM expects a struct that contains
 * two member functions: K.forward(), and K.adjoint().
 */
template <typename T, size_t N>
concept LinOpGraphImpl = requires(T a, const Halide::Func& f, const FuncTuple<N>& ft) {
    // forward function expects a Halide::Func as a data plane / data cube, and
    // returns a tuple of data planes.
    { a.forward(f) }
    -> std::same_as<FuncTuple<N>>;

    // adjoint function expects an array of Halide::Func, and returns
    // one single Func.
    { a.adjoint(ft) }
    -> std::same_as<Halide::Func>;
};

template <typename T>
concept Prox = requires(T a, const Halide::Func& f, const Halide::Expr& e) {
    { a.operator()(f, e, f) }
    -> std::same_as<Halide::Func>;
};

#define LinOpGraph LinOpGraphImpl<N>

#else

#define LinOpGraph class
#define Prox class

#endif