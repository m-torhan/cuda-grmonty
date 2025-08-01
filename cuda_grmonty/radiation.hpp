/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/ndarray.hpp"

namespace radiation {

double bk_angle(const double (&x)[consts::n_dim],
                const double (&k)[consts::n_dim],
                const double (&u_cov)[consts::n_dim],
                const double (&b_cov)[consts::n_dim],
                double b,
                double b_unit);

double
fluid_nu(const double (&x)[consts::n_dim], const double (&k)[consts::n_dim], const double (&u_cov)[consts::n_dim]);

double alpha_inv_scatt(double nu, double theta_e, double n_e, const ndarray::NDArray<double, 2> &hotcross_table);

double alpha_inv_abs(double nu,
                     double theta_e,
                     double n_e,
                     double b,
                     double theta,
                     const std::array<double, consts::n_e_samp + 1> &k2_table);

}; /* namespace radiation */
