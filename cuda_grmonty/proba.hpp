/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <tuple>

#include "cuda_grmonty/consts.hpp"

namespace proba {

/**
 * @brief Sample an electron momentum vector from a relativistic Maxwell-Jüttner distribution.
 *
 * @param[in] k       Incoming photon 4-momentum vector.
 * @param[out] p      Output electron 3-momentum vector.
 * @param[in] theta_e Electron dimensionless temperature.
 */
void sample_electron_distr_p(const double (&k)[consts::n_dim], double (&p)[consts::n_dim], double theta_e);

/**
 * @brief Sample electron Lorentz factor (gamma) and speed (beta) from a thermal distribution.
 *
 * @param theta_e Electron dimensionless temperature.
 *
 * @return Tuple of (gamma, beta) for the sampled electron.
 */
std::tuple<double, double> sample_beta_distr(double theta_e);

/**
 * @brief Sample the y parameter for the electron energy distribution.
 *
 * @param theta_e Electron dimensionless temperature.
 *
 * @return Sampled y value.
 */
double sample_y_distr(double theta_e);

/**
 * @brief Sample the cosine of the electron velocity direction.
 *
 * @param beta_e Electron speed (v/c).
 *
 * @return Sampled mu = cos(theta) of electron velocity.
 */
double sample_mu_distr(double beta_e);

/**
 * @brief Sample the outgoing photon energy using the Klein-Nishina cross-section.
 *
 * @param k0 Incoming photon energy.
 *
 * @return Sampled scattered photon energy.
 */
double sample_klein_nishina(double k0);

/**
 * @brief Sample the outgoing photon direction using the Thomson cross-section.
 *
 * @return Cosine of the scattering angle for the photon.
 */
double sample_thomson();

/**
 * @brief Sample a random 3D unit vector uniformly on the sphere.
 *
 * @return Tuple of (x, y, z) components of the sampled unit vector.
 */
std::tuple<double, double, double> sample_rand_dir();

}; /* namespace proba */
