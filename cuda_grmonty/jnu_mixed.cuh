/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <math_constants.h>

#include "cuda_grmonty/consts.hpp"

namespace cuda_jnu_mixed {

/**
 * @brief Evaluate k2 for a given electron dimensionless temperature using device table.
 *
 * @param theta_e  Electron dimensionless temperature.
 * @param k2_table Pointer to k2 table in device memory.
 *
 * @return Evaluated k2 value.
 */
static __device__ double k2_eval(double theta_e, const double *__restrict__ k2_table);

/**
 * @brief Linearly interpolate k2 from a device table for a given electron temperature.
 *
 * @param theta_e  Electron dimensionless temperature.
 * @param k2_table Pointer to k2 table in device memory.
 *
 * @return Interpolated k2 value.
 */
static __device__ double linear_interp_k2(double theta_e, const double *__restrict__ k2_table);

/**
 * @brief Compute synchrotron emissivity for a photon at a given frequency and fluid parameters.
 *
 * @param nu       Photon frequency.
 * @param n_e      Electron number density.
 * @param theta_e  Electron dimensionless temperature.
 * @param b        Magnetic field strength.
 * @param theta    Pitch angle.
 * @param k2_table Pointer to k2 table in device memory.
 *
 * @return Synchrotron emissivity at the given frequency.
 */
static __device__ double
synch(double nu, double n_e, double theta_e, double b, double theta, const double *__restrict__ k2_table);

static __device__ double k2_eval(double theta_e, const double *__restrict__ k2_table) {
    if (theta_e < consts::theta_e_min) {
        return 0.0;
    }
    if (theta_e > consts::jnu::max_t) {
        return 2.0 * theta_e * theta_e;
    }

    return linear_interp_k2(theta_e, k2_table);
}

static __device__ double
synch(double nu, double n_e, double theta_e, double b, double theta, const double *__restrict__ k2_table) {
    if (theta_e < consts::theta_e_min) {
        return 0.0;
    }

    double k2 = k2_eval(theta_e, k2_table);
    double nu_c = consts::ee * b / (2.0 * CUDART_PI * consts::me * consts::cl);
    double sin_th = sin(theta);
    double nu_s = (2.0 / 9.0) * nu_c * theta_e * theta_e * sin_th;

    if (nu > 1.0e12 * nu_s) {
        return 0.0;
    }

    double x = nu / nu_s;
    double xp = pow(x, 1.0 / 3.0);
    double xx = sqrt(x) + consts::jnu::cst * sqrt(xp);
    double f = xx * xx;
    return (CUDART_SQRT_TWO * CUDART_PI * consts::ee * consts::ee * n_e * nu_s / (3.0 * consts::cl * k2)) * f *
           exp(-xp);
}

static __device__ double linear_interp_k2(double theta_e, const double *__restrict__ k2_table) {
    const double l_min_t = log(consts::jnu::min_t);
    const double d_l_t = log(consts::jnu::max_t / consts::jnu::min_t) / consts::n_e_samp;

    double l_t = log(theta_e);
    double d_i = (l_t - l_min_t) / d_l_t;
    int i = static_cast<int>(d_i);

    d_i -= i;

    return exp((1.0 - d_i) * k2_table[i] + d_i * k2_table[i + 1]);
}

}; /* namespace cuda_jnu_mixed */
