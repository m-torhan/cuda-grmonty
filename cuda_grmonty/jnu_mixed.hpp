/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/ndarray.hpp"

namespace jnu_mixed {

void init_emiss_tables(ndarray::NDArray<double> &f, ndarray::NDArray<double> &k2);

double synch(double nu, double n_e, double theta_e, double b, double theta, const ndarray::NDArray<double> &k2_table);

double k2_eval(double theta_e, const ndarray::NDArray<double> &k2_table);

double f_eval(double theta_e, double b_mag, double nu, const ndarray::NDArray<double> &f_table);

}; /* namespace jnu_mixed */
