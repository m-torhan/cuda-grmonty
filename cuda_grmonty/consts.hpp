/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace consts {

constexpr int n_dim = 4;
constexpr int n_prim = 8;

/* range of initial superphoton frequencies */
constexpr double nu_min = 1.0e9;
constexpr double nu_max = 1.0e16;

constexpr double thetae_min = 1000.0;
constexpr double thetae_max = 0.3;
constexpr double tp_over_te = 3.0;

constexpr double weight_min = 1.0e31;

constexpr double electron_mass = 9.1093826e-28;
constexpr double photon_mass = 1.67262171e-24;

}; /* namespace consts */
