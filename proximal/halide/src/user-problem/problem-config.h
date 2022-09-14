#pragma once

/** User-defined problem configurations. This is supposed to be generated automatically by
 * ProxImal-Gen */
namespace problem_config {

/** Number of functions in the set "psi", for (L-)ADMM solvers. */
constexpr auto psi_size = 2;

/** input data size of the user-provided (distorted and noisy) image. */
constexpr auto input_width = 512;
constexpr auto input_height = 512;
constexpr auto input_size = input_width * input_height;

/** output data size of the restored image. */
constexpr auto output_width = 512;
constexpr auto output_height = 512;
constexpr auto output_size = output_width * output_height;
}  // namespace problem_config