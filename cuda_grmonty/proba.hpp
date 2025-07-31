/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <tuple>

#include "cuda_grmonty/consts.hpp"

namespace proba {

void sample_electron_distr_p(const double (&k)[consts::n_dim], double (&p)[consts::n_dim], double theta_e);

std::tuple<double, double> sample_beta_distr(double theta_e);

double sample_y_distr(double theta_e);

double sample_mu_distr(double beta_e);

double sample_klein_nishina(double k0);

double sample_thomson();

std::tuple<double, double, double> sample_rand_dir();

}; /* namespace proba */
