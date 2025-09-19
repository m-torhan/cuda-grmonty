/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

#include "cuda_grmonty/hotcross.cuh"
#include "cuda_grmonty/jnu_mixed.cuh"

namespace cuda_radiation {

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
static __device__ double bk_angle(const double (&x)[consts::n_dim],
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
static __device__ double
fluid_nu(const double (&x)[consts::n_dim], const double (&k)[consts::n_dim], const double (&u_cov)[consts::n_dim]);

/**
 * @brief Compute inverse scattering opacity (alpha^{-1}) for given photon parameters.
 *
 * @param nu             Photon frequency.
 * @param theta_e        Electron dimensionless temperature.
 * @param n_e            Electron number density.
 * @param hotcross_table Pointer to precomputed hotcross table on device memory.
 *
 * @return Inverse scattering opacity at specified parameters.
 */
static __device__ double
alpha_inv_scatt(double nu, double theta_e, double n_e, const double *__restrict__ hotcross_table);

/**
 * @brief Compute inverse absorption opacity (alpha^{-1}) for given photon parameters.
 *
 * @param nu       Photon frequency.
 * @param theta_e  Electron dimensionless temperature.
 * @param n_e      Electron number density.
 * @param b        Magnetic field strength.
 * @param theta    Pitch angle between photon and magnetic field.
 * @param k2_table Pointer to precomputed k2 table on device memory.
 *
 * @return Inverse absorption opacity at specified parameters.
 */
static __device__ double
alpha_inv_abs(double nu, double theta_e, double n_e, double b, double theta, const double *__restrict__ k2_table);

/**
 * @brief Compute inverse Planck function B_ν^{-1}.
 *
 * @param nu      Photon frequency.
 * @param theta_e Electron dimensionless temperature.
 *
 * @return Inverse Planck function value at the given frequency and temperature.
 */
static __device__ double b_nu_inv(double nu, double theta_e);

/**
 * @brief Compute inverse synchrotron emissivity j_ν^{-1}.
 *
 * @param nu       Photon frequency.
 * @param theta_e  Electron dimensionless temperature.
 * @param n_e      Electron number density.
 * @param b        Magnetic field strength.
 * @param theta    Pitch angle between photon and magnetic field.
 * @param k2_table Pointer to precomputed k2 table on device memory.
 *
 * @return Inverse synchrotron emissivity at specified parameters.
 */
static __device__ double
jnu_inv(double nu, double theta_e, double n_e, double b, double theta, const double *__restrict__ k2_table);

/**
 * @brief Compute electron scattering opacity (Thomson/Compton) using precomputed hotcross table.
 *
 * @param nu             Photon frequency.
 * @param theta_e        Electron dimensionless temperature.
 * @param hotcross_table Pointer to precomputed hotcross table on device memory.
 *
 * @return Scattering opacity at specified frequency and temperature.
 */
static __device__ double kappa_es(double nu, double theta_e, const double *__restrict__ hotcross_table);

static __device__ double bk_angle(const double (&x)[consts::n_dim],
                                  const double (&k)[consts::n_dim],
                                  const double (&u_cov)[consts::n_dim],
                                  const double (&b_cov)[consts::n_dim],
                                  double b,
                                  double b_unit) {
    if (b == 0.0) {
        return CUDART_PI / 2.0;
    }

    /* clang-format off */
    double k_ = abs(
         k[0] * u_cov[0] +
         k[1] * u_cov[1] +
         k[2] * u_cov[2] +
         k[3] * u_cov[3]
    );
    double mu = (
        k[0] * b_cov[0] +
        k[1] * b_cov[1] +
        k[2] * b_cov[2] +
        k[3] * b_cov[3]
    ) / (k_ * b / b_unit);
    /* clang-format on */

    mu = mu < -1.0 ? -1.0 : (mu > 1.0 ? 1.0 : mu);

    return acos(mu);
}

static __device__ double
fluid_nu(const double (&x)[consts::n_dim], const double (&k)[consts::n_dim], const double (&u_cov)[consts::n_dim]) {
    /* clang-format off */
    double energy = -(
        k[0] * u_cov[0] +
        k[1] * u_cov[1] +
        k[2] * u_cov[2] +
        k[3] * u_cov[3]
    );
    /* clang-format on */

    return energy * consts::me * consts::cl * consts::cl / consts::hpl;
}

static __device__ double
alpha_inv_scatt(double nu, double theta_e, double n_e, const double *__restrict__ hotcross_table) {
    double kappa = kappa_es(nu, theta_e, hotcross_table);

    return nu * kappa * n_e * consts::mp;
}

static __device__ double
alpha_inv_abs(double nu, double theta_e, double n_e, double b, double theta, const double *__restrict__ k2_table) {
    double j = jnu_inv(nu, theta_e, n_e, b, theta, k2_table);
    double b_nu = b_nu_inv(nu, theta_e);

    return j / (b_nu + 1.0e-100);
}

static __device__ double b_nu_inv(double nu, double theta_e) {
    double x = consts::hpl * nu / (consts::me * consts::cl * consts::cl * theta_e);

    if (x < 1.0e-3) {
        return (2.0 * consts::hpl / (consts::cl * consts::cl)) / (x / 24.0 * (24.0 + x * (12.0 + x * (4.0 + x))));
    }

    return (2.0 * consts::hpl / (consts::cl * consts::cl)) / (std::exp(x) - 1.0);
}

static __device__ double
jnu_inv(double nu, double theta_e, double n_e, double b, double theta, const double *__restrict__ k2_table) {
    double j = cuda_jnu_mixed::synch(nu, n_e, theta_e, b, theta, k2_table);

    return j / (nu * nu);
}

static __device__ double kappa_es(double nu, double theta_e, const double *__restrict__ hotcross_table) {
    double e_g = consts::hpl * nu / (consts::me * consts::cl * consts::cl);

    return cuda_hotcross::total_compton_cross_lkup(e_g, theta_e, hotcross_table) / consts::mp;
}

}; /* namespace cuda_radiation */
