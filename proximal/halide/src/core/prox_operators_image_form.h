/** @file Conjugate descent algorithms.
 *
 * This file contains all proximal operators involving the image formation
 * operators. Currently all image formation proximal operators are computed via
 * CG. The two CG implementations in this file are identical except for the
 * function call of A^T*A, where A is here the concatenation of all image
 * formation matrices.
 *
 */

#pragma once
#warning Conjugate descent algorithms are no longer supported. Please contribute by recovering the code from Git history.

#include <vector>

#include "Halide.h"

namespace {

using namespace Halide;
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Solves the proximal of || M conv(h, x) - b ||_2^2 with conjugate gradient
 * descent algorithm.
 *
 * For masked convolution problems, the ProxImaL compiler by default absorbs
 * only the masking operater into the cost function, in order to ensure a
 * closed-formed proximal operator. However, it may result in the linear mapping
 * between the split variables (i.e. K of z = K x) having a high condition
 * number, causing slow ADMM convergence. So, it may be worth overriding
 * ProxImaL's default rules to approximate `prox(|| M conv(h, x) - b||_2^2)`.
 *
 * @deprecated
 */
[[deprecated]] std::vector<Func>
CG_conv(Func xin_func, Func Atb, Func K_func, Func MtM_func, Expr width, Expr height, Expr ch,
        Expr width_kernel, Expr height_kernel, Expr beta_cg, int cgiters, int outter_iter) {
    __builtin_unreachable();
    return {};
}

/**
 * Solves the proximal of || M warpHomography(H, x) - b ||_2^2 with conjugate
 * gradient descent algorithm.
 *
 * For masked convolution problems, the ProxImaL compiler by default absorbs
 * only the masking operater into the cost function, in order to ensure a
 * closed-formed proximal operator. However, it may result in the linear mapping
 * between the split variables (i.e. K of z = K x) having a high condition
 * number, causing slow ADMM convergence. So, it may be worth overriding
 * ProxImaL's default rules to approximate `prox(|| M warpHomography(H, x) -
 * b||_2^2)`.
 *
 * @deprecated
 */
[[deprecated]] std::vector<Func>
CG_warp(Func xin_func, Func Atb, Func H_func, Func Hinv_func, Func MtM_func, Expr width,
        Expr height, Expr ch, Expr nhom, Expr beta_cg, int cgiters, int outter_iter) {
    __builtin_unreachable();
    return {};
}

}  // namespace