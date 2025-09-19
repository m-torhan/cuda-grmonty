/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>

#include "cuda_grmonty/consts.hpp"

namespace jnu_mixed {

/**
 * @brief Initialize tables for synchrotron emissivity calculations.
 *
 * @param f  Array to store f(theta_e) values.
 * @param k2 Array to store k2(theta_e) values.
 */
void init_emiss_tables(std::array<double, consts::n_e_samp + 1> &f, std::array<double, consts::n_e_samp + 1> &k2);

/**
 * @brief Compute synchrotron emissivity for a photon at a given frequency and fluid parameters.
 *
 * @param nu       Photon frequency.
 * @param n_e      Electron number density.
 * @param theta_e  Electron dimensionless temperature.
 * @param b        Magnetic field strength.
 * @param theta    Pitch angle.
 * @param k2_table Table of k2 values for interpolation.
 *
 * @return Synchrotron emissivity at the given frequency.
 */
double synch(double nu,
             double n_e,
             double theta_e,
             double b,
             double theta,
             const std::array<double, consts::n_e_samp + 1> &k2_table);

/**
 * @brief Evaluate k2 for a given electron dimensionless temperature using a lookup table.
 *
 * @param theta_e  Electron dimensionless temperature.
 * @param k2_table Table of k2 values for interpolation.
 *
 * @return Evaluated k2 value.
 */
double k2_eval(double theta_e, const std::array<double, consts::n_e_samp + 1> &k2_table);

/**
 * @brief Evaluate synchrotron function f(theta_e) for given parameters using table interpolation.
 *
 * @param theta_e Electron dimensionless temperature.
 * @param b_mag   Magnetic field strength.
 * @param nu      Photon frequency.
 * @param f_table Table of f(theta_e) values for interpolation.
 *
 * @return Evaluated synchrotron function f value.
 */
double f_eval(double theta_e, double b_mag, double nu, const std::array<double, consts::n_e_samp + 1> &f_table);

}; /* namespace jnu_mixed */
