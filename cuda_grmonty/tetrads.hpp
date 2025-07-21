/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/ndarray.hpp"

namespace tetrads {

void lower(const double (&u_con)[consts::n_dim], const ndarray::NDArray<double> &g_cov, double (&u_cov)[consts::n_dim]);

}; /* namespace tetrads */
