/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cmath>

namespace consts {

constexpr int n_dim = 4;
constexpr int n_prim = 8;

constexpr double eps = 1.0e-40;

constexpr int n_e_samp = 200;
constexpr int n_e_bins = 200;
constexpr int n_th_bins = 6;

/* range of initial superphoton frequencies */
constexpr double nu_min = 1.0e9;
constexpr double nu_max = 1.0e16;

constexpr double thetae_min = 1000.0;
constexpr double thetae_max = 0.3;
constexpr double tp_over_te = 3.0;

constexpr double weight_min = 1.0e31;

constexpr double electron_mass = 9.1093826e-28;
constexpr double photon_mass = 1.67262171e-24;

constexpr double sigma_thomson = 0.665245873e-24; /* Thomson cross section in cm^2 */

namespace hotcross {

constexpr double min_w = 1.0e-12;
constexpr double max_w = 1.0e6;
constexpr double min_t = 1.0e-4;
constexpr double max_t = 1.0e4;
constexpr int n_w = 220;
constexpr int n_t = 80;

constexpr double max_gamma = 12.0;
constexpr double d_mu_e = 0.05;
constexpr double d_gamma_e = 0.05;

}; /* namespace hotcross */

}; /* namespace consts */
