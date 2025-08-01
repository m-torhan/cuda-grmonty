/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>

#include "cuda_grmonty/consts.hpp"

namespace jnu_mixed {

void init_emiss_tables(std::array<double, consts::n_e_samp + 1> &f, std::array<double, consts::n_e_samp + 1> &k2);

double synch(double nu,
             double n_e,
             double theta_e,
             double b,
             double theta,
             const std::array<double, consts::n_e_samp + 1> &k2_table);

double k2_eval(double theta_e, const std::array<double, consts::n_e_samp + 1> &k2_table);

double f_eval(double theta_e, double b_mag, double nu, const std::array<double, consts::n_e_samp + 1> &f_table);

}; /* namespace jnu_mixed */
