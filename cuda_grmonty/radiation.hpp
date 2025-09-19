/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/ndarray.hpp"

namespace radiation {

/**
 * @brief Compute the angle between photon, fluid velocity, and magnetic field in the fluid frame.
 *
 * @param x      Photon position 4-vector.
 * @param k      Photon momentum 4-vector.
 * @param u_cov  Fluid 4-velocity covariant components.
 * @param b_cov  Magnetic field 4-vector covariant components.
 * @param b      Magnetic field strength.
 * @param b_unit Unit vector along magnetic field direction.
 *
 * @return Angle between photon and magnetic field in the fluid frame.
 */
double bk_angle(const double (&x)[consts::n_dim],
                const double (&k)[consts::n_dim],
                const double (&u_cov)[consts::n_dim],
                const double (&b_cov)[consts::n_dim],
                double b,
                double b_unit);

/**
 * @brief Compute the photon frequency in the local fluid frame.
 *
 * @param x     Photon position 4-vector.
 * @param k     Photon momentum 4-vector.
 * @param u_cov Fluid 4-velocity covariant components.
 *
 * @return Photon frequency in the fluid frame.
 */
double
fluid_nu(const double (&x)[consts::n_dim], const double (&k)[consts::n_dim], const double (&u_cov)[consts::n_dim]);

/**
 * @brief Compute inverse scattering opacity (alpha^{-1}) for given photon parameters.
 *
 * @param nu             Photon frequency.
 * @param theta_e        Electron dimensionless temperature.
 * @param n_e            Electron number density.
 * @param hotcross_table Precomputed table of total Compton cross-sections.
 *
 * @return Inverse scattering opacity at specified parameters.
 */
double alpha_inv_scatt(double nu, double theta_e, double n_e, const ndarray::NDArray<double, 2> &hotcross_table);

/**
 * @brief Compute inverse absorption opacity (alpha^{-1}) for given photon parameters.
 *
 * @param nu       Photon frequency.
 * @param theta_e  Electron dimensionless temperature.
 * @param n_e      Electron number density.
 * @param b        Magnetic field strength.
 * @param theta    Pitch angle between photon and magnetic field.
 * @param k2_table Precomputed table of k2(theta_e) values for interpolation.
 *
 * @return Inverse absorption opacity at specified parameters.
 */
double alpha_inv_abs(double nu,
                     double theta_e,
                     double n_e,
                     double b,
                     double theta,
                     const std::array<double, consts::n_e_samp + 1> &k2_table);

}; /* namespace radiation */
