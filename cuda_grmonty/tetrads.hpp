/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/ndarray.hpp"

namespace tetrads {

void coordinate_to_tetrad(const double (&e_cov)[consts::n_dim][consts::n_dim],
                          const double (&k)[consts::n_dim],
                          double (&k_tetrad)[consts::n_dim]);

void tetrad_to_coordinate(const double (&e_con)[consts::n_dim][consts::n_dim],
                          const double (&k_tetrad)[consts::n_dim],
                          double (&k)[consts::n_dim]);

void make_tetrad(const double (&u_con)[consts::n_dim],
                 double (&trial)[consts::n_dim],
                 const ndarray::NDArray<double> &g_cov,
                 double (&e_con)[consts::n_dim][consts::n_dim],
                 double (&e_cov)[consts::n_dim][consts::n_dim]);

void lower(const double (&u_con)[consts::n_dim], const ndarray::NDArray<double> &g_cov, double (&u_cov)[consts::n_dim]);

}; /* namespace tetrads */
