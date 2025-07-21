/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_grmonty/tetrads.hpp"
#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/ndarray.hpp"

namespace tetrads {

void lower(const double (&u_con)[consts::n_dim], const ndarray::NDArray<double> &g_cov,
           double (&u_cov)[consts::n_dim]) {
    /* clang-format off */
    u_cov[0] = (
        g_cov[{0, 0}].value() * u_con[0]
      + g_cov[{0, 1}].value() * u_con[1]
      + g_cov[{0, 2}].value() * u_con[2]
      + g_cov[{0, 3}].value() * u_con[3]
    );
    u_cov[1] = (
        g_cov[{1, 0}].value() * u_con[0]
      + g_cov[{1, 1}].value() * u_con[1]
      + g_cov[{1, 2}].value() * u_con[2]
      + g_cov[{1, 3}].value() * u_con[3]
    );
    u_cov[2] = (
        g_cov[{2, 0}].value() * u_con[0]
      + g_cov[{2, 1}].value() * u_con[1]
      + g_cov[{2, 2}].value() * u_con[2]
      + g_cov[{2, 3}].value() * u_con[3]
    );
    u_cov[3] = (
        g_cov[{3, 0}].value() * u_con[0]
      + g_cov[{3, 1}].value() * u_con[1]
      + g_cov[{3, 2}].value() * u_con[2]
      + g_cov[{3, 3}].value() * u_con[3]
    );
    /* clang-format off */
}

}; /* namespace tetrads */
