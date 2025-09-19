/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <curand_kernel.h>
#include <math_constants.h>

#include "cuda_grmonty/consts.hpp"

namespace cuda_proba {

/**
 * @brief Sample an electron momentum vector from a relativistic Maxwell-Jüttner distribution.
 *
 * @param rng_state Pointer to CUDA random number generator state.
 * @param k         Incoming photon 4-momentum vector.
 * @param p         Output electron 3-momentum vector.
 * @param theta_e   Electron dimensionless temperature.
 */
static __device__ void sample_electron_distr_p(curandStatePhilox4_32_10_t *rng_state,
                                               const double (&k)[consts::n_dim],
                                               double (&p)[consts::n_dim],
                                               double theta_e);

/**
 * @brief Sample electron Lorentz factor (gamma) and speed (beta) from a relativistic thermal distribution.
 *
 * @param rng_state Pointer to CUDA random number generator state.
 * @param theta_e   Electron dimensionless temperature.
 * @param gamma_e   Output Lorentz factor.
 * @param beta_e    Output speed (v/c).
 */
static __device__ void
sample_beta_distr(curandStatePhilox4_32_10_t *rng_state, double theta_e, double *gamma_e, double *beta_e);

/**
 * @brief Sample the y parameter for the electron energy distribution.
 *
 * @param rng_state Pointer to CUDA random number generator state.
 * @param theta_e   Electron dimensionless temperature.
 *
 * @return Sampled y value.
 */
static __device__ double sample_y_distr(curandStatePhilox4_32_10_t *rng_state, double theta_e);

/**
 * @brief Sample the cosine of the electron velocity direction.
 *
 * @param rng_state Pointer to CUDA random number generator state.
 * @param beta_e    Electron speed (v/c).
 *
 * @return Sampled mu = cos(theta) of electron velocity.
 */
static __device__ double sample_mu_distr(curandStatePhilox4_32_10_t *rng_state, double beta_e);

/**
 * @brief Sample the outgoing photon energy using the Klein-Nishina cross-section.
 *
 * @param rng_state Pointer to CUDA random number generator state.
 * @param k0        Incoming photon energy.
 *
 * @return Sampled scattered photon energy.
 */
static __device__ double sample_klein_nishina(curandStatePhilox4_32_10_t *rng_state, double k0);

/**
 * @brief Sample the outgoing photon direction using the Thomson cross-section.
 *
 * @param rng_state Pointer to CUDA random number generator state.
 *
 * @return Sampled cos(theta) for scattered photon direction.
 */
static __device__ double sample_thomson(curandStatePhilox4_32_10_t *rng_state);

/**
 * @brief Sample a random 3D unit vector uniformly on the sphere.
 *
 * @param rng_state Pointer to CUDA random number generator state.
 * @param x         Output x-component of unit vector.
 * @param y         Output y-component of unit vector.
 * @param z         Output z-component of unit vector.
 */
static __device__ void sample_rand_dir(curandStatePhilox4_32_10_t *rng_state, double *x, double *y, double *z);

/**
 * @brief Compute Klein-Nishina differential cross-section.
 *
 * @param a  Incoming photon energy.
 * @param ap Scattered photon energy.
 *
 * @return Klein-Nishina differential cross-section value.
 */
static __device__ double klein_nishina(double a, double ap);

static __device__ void sample_electron_distr_p(curandStatePhilox4_32_10_t *rng_state,
                                               const double (&k)[consts::n_dim],
                                               double (&p)[consts::n_dim],
                                               double theta_e) {
    double x1;
    double sigma_kn;
    double gamma_e;
    double beta_e;
    double mu;

    do {
        sample_beta_distr(rng_state, theta_e, &gamma_e, &beta_e);
        mu = sample_mu_distr(rng_state, beta_e);
        int sample_cnt = 0;

        if (mu > 1.0) {
            mu = 1.0;
        } else if (mu < -1.0) {
            mu = -1.0;
        }

        /* frequency in electron rest frame */
        double k_ = gamma_e * (1.0 - beta_e * mu) * k[0];
        if (k_ < 1.0e-3) {
            sigma_kn = 1.0 - 2.0 * k_;
        } else {
            sigma_kn = (3.0 / (4.0 * k_ * k_)) * (2.0 + k_ * k_ * (1.0 + k_) / ((1.0 + 2.0 * k_) * (1.0 + 2.0 * k_)) +
                                                  (k_ * k_ - 2.0 * k_ - 2.0) / (2.0 * k_) * log(1.0 + 2.0 * k_));
        }

        x1 = curand_uniform(rng_state);

        ++sample_cnt;

        if (sample_cnt > 10'000'000) {
            theta_e *= 0.5;
            sample_cnt = 0;
        }
    } while (x1 >= sigma_kn);

    /* first unit vector for coordinate system */
    double v0x = k[1];
    double v0y = k[2];
    double v0z = k[3];
    double v0 = sqrt(v0x * v0x + v0y * v0y + v0z * v0z);
    v0x /= v0;
    v0y /= v0;
    v0z /= v0;

    /* pick zero-angle for coordinate system */
    double n0x;
    double n0y;
    double n0z;
    sample_rand_dir(rng_state, &n0x, &n0y, &n0z);
    double n0dotv0 = v0x * n0x + v0y * n0y + v0z * n0z;

    /* second unit vector */
    double v1x = n0x - (n0dotv0)*v0x;
    double v1y = n0y - (n0dotv0)*v0y;
    double v1z = n0z - (n0dotv0)*v0z;

    /* normalize */
    double v1 = sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
    v1x /= v1;
    v1y /= v1;
    v1z /= v1;

    /* find one more unit vector using cross product this one is automatically normalized */
    double v2x = v0y * v1z - v0z * v1y;
    double v2y = v0z * v1x - v0x * v1z;
    double v2z = v0x * v1y - v0y * v1x;

    /* resolve new momentum vector along unit vectors and create a four-vector $p$ */
    double phi = curand_uniform(rng_state) * 2.0 * CUDART_PI; /* orient uniformly */
    double s_phi = sin(phi);
    double c_phi = cos(phi);

    double c_th = mu;
    double s_th = sqrt(1. - mu * mu);

    p[0] = gamma_e;
    p[1] = gamma_e * beta_e * (c_th * v0x + s_th * (c_phi * v1x + s_phi * v2x));
    p[2] = gamma_e * beta_e * (c_th * v0y + s_th * (c_phi * v1y + s_phi * v2y));
    p[3] = gamma_e * beta_e * (c_th * v0z + s_th * (c_phi * v1z + s_phi * v2z));

    if (beta_e < 0) {
        /* error */
    }
}

static __device__ void
sample_beta_distr(curandStatePhilox4_32_10_t *rng_state, double theta_e, double *gamma_e, double *beta_e) {
    double y = sample_y_distr(rng_state, theta_e);

    *gamma_e = y * y * theta_e + 1.0;
    *beta_e = sqrt(1.0 - 1.0 / ((*gamma_e) * (*gamma_e)));
}

static __device__ double sample_y_distr(curandStatePhilox4_32_10_t *rng_state, double theta_e) {
    double pi_3 = sqrt(CUDART_PI) / 4.0;
    double pi_4 = sqrt(0.5 * theta_e) / 2.0;
    double pi_5 = 3.0 * sqrt(CUDART_PI) * theta_e / 8.0;
    double pi_6 = theta_e * sqrt(0.5 * theta_e);

    double s_3 = pi_3 + pi_4 + pi_5 + pi_6;

    pi_3 /= s_3;
    pi_4 /= s_3;
    pi_5 /= s_3;
    pi_6 /= s_3;

    double y;
    double x2;
    double prob;

    do {
        double x1 = curand_uniform(rng_state);
        int dof;

        if (x1 < pi_3) {
            dof = 3;
        } else if (x1 < pi_3 + pi_4) {
            dof = 4;
        } else if (x1 < pi_3 + pi_4 + pi_5) {
            dof = 5;
        } else {
            dof = 6;
        }
        float x = 0.0f;

        /* TODO: check if matches chi2 */
        for (int i = 0; i < dof; i++) {
            float z = curand_normal(rng_state);
            x += z * z;
        }

        y = sqrt(x / 2.0);

        x2 = curand_uniform(rng_state);
        double num = sqrt(1.0 + 0.5 * theta_e * y * y);
        double den = (1.0 + y * sqrt(0.5 * theta_e));

        prob = num / den;
    } while (x2 >= prob);

    return y;
}

static __device__ double sample_mu_distr(curandStatePhilox4_32_10_t *rng_state, double beta_e) {
    double x1 = curand_uniform(rng_state);
    double det = 1.0 + 2.0 * beta_e + beta_e * beta_e - 4.0 * beta_e * x1;
    return (1.0 - sqrt(det)) / beta_e;
}

static __device__ double sample_klein_nishina(curandStatePhilox4_32_10_t *rng_state, double k0) {
    double k0pmin = k0 / (1.0 + 2.0 * k0);
    double k0pmax = k0;

    double x1;
    double k0p_tent;

    do {
        k0p_tent = k0pmin + (k0pmax - k0pmin) * curand_uniform(rng_state);

        x1 = 2.0 * (1.0 + 2.0 * k0 + 2.0 * k0 * k0) / (k0 * k0 * (1.0 + 2.0 * k0));
        x1 *= curand_uniform(rng_state);
    } while (x1 >= klein_nishina(k0, k0p_tent));

    return k0p_tent;
}

static __device__ double sample_thomson(curandStatePhilox4_32_10_t *rng_state) {
    double x1, x2;

    do {
        x1 = 2.0 * curand_uniform(rng_state) - 1.0;
        x2 = (3.0 / 4.0) * curand_uniform(rng_state);
    } while (x2 >= (3.0 / 8.0) * (1.0 + x1 * x1));

    return x1;
}

static __device__ void sample_rand_dir(curandStatePhilox4_32_10_t *rng_state, double *x, double *y, double *z) {
    *z = curand_uniform(rng_state) * 2.0 - 1.0;
    double phi = curand_uniform(rng_state) * 2.0 * CUDART_PI;

    *x = sqrt(1.0 - (*z) * (*z)) * cos(phi);
    *y = sqrt(1.0 - (*z) * (*z)) * sin(phi);
}

static __device__ double klein_nishina(double a, double ap) {
    double ch = 1.0 + 1.0 / a - 1.0 / ap;
    return (a / ap + ap / a - 1.0 + ch * ch) / (a * a);
}

}; /* namespace cuda_proba */
