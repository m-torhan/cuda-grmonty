/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/consts.hpp"

namespace cuda_tetrads {

static __device__ double delta(int i, int j) { return i == j ? 1.0 : 0.0; }

static __device__ void coordinate_to_tetrad(const double (&e_cov)[consts::n_dim][consts::n_dim],
                                            const double (&k)[consts::n_dim],
                                            double (&k_tetrad)[consts::n_dim]);

static __device__ void tetrad_to_coordinate(const double (&e_con)[consts::n_dim][consts::n_dim],
                                            const double (&k_tetrad)[consts::n_dim],
                                            double (&k)[consts::n_dim]);

static __device__ void make_tetrad(const double (&u_con)[consts::n_dim],
                                   double (&trial)[consts::n_dim],
                                   const double (&g_cov)[consts::n_dim][consts::n_dim],
                                   double (&e_con)[consts::n_dim][consts::n_dim],
                                   double (&e_cov)[consts::n_dim][consts::n_dim]);

static __device__ void lower(const double (&u_con)[consts::n_dim],
                             const double (&g_cov)[consts::n_dim][consts::n_dim],
                             double (&u_cov)[consts::n_dim]);

static __device__ void normalize(double (&v_con)[consts::n_dim], const double (&g_cov)[consts::n_dim][consts::n_dim]);

static __device__ void project_out(double (&v_con_a)[consts::n_dim],
                                   double (&v_con_b)[consts::n_dim],
                                   const double (&g_cov)[consts::n_dim][consts::n_dim]);

static __device__ void coordinate_to_tetrad(const double (&e_cov)[consts::n_dim][consts::n_dim],
                                            const double (&k)[consts::n_dim],
                                            double (&k_tetrad)[consts::n_dim]) {
    for (int i = 0; i < consts::n_dim; ++i) {
        k_tetrad[i] = 0.0;
        for (int j = 0; j < consts::n_dim; ++j) {
            k_tetrad[i] += e_cov[i][j] * k[j];
        }
    }
}

static __device__ void tetrad_to_coordinate(const double (&e_con)[consts::n_dim][consts::n_dim],
                                            const double (&k_tetrad)[consts::n_dim],
                                            double (&k)[consts::n_dim]) {
    for (int i = 0; i < consts::n_dim; ++i) {
        k[i] = 0.0;
        for (int j = 0; j < consts::n_dim; ++j) {
            k[i] += e_con[j][i] * k_tetrad[j];
        }
    }
}

static __device__ void make_tetrad(const double (&u_con)[consts::n_dim],
                                   double (&trial)[consts::n_dim],
                                   const double (&g_cov)[consts::n_dim][consts::n_dim],
                                   double (&e_con)[consts::n_dim][consts::n_dim],
                                   double (&e_cov)[consts::n_dim][consts::n_dim]) {
    for (int i = 0; i < consts::n_dim; ++i) {
        e_con[0][i] = u_con[i];
    }

    normalize(e_con[0], g_cov);

    double norm = 0.0;

    for (int i = 0; i < consts::n_dim; ++i) {
        for (int j = 0; j < consts::n_dim; ++j) {
            norm += trial[i] * trial[j] * g_cov[i][j];
        }
    }

    if (norm < 1.0e-30) {
        for (int i = 0; i < consts::n_dim; ++i) {
            trial[i] = delta(i, 1);
        }
    }

    for (int i = 0; i < consts::n_dim; ++i) {
        e_con[1][i] = trial[i];
    }

    project_out(e_con[1], e_con[0], g_cov);
    normalize(e_con[1], g_cov);

    for (int i = 0; i < consts::n_dim; ++i) {
        e_con[2][i] = delta(i, 2);
    }

    project_out(e_con[2], e_con[0], g_cov);
    project_out(e_con[2], e_con[1], g_cov);
    normalize(e_con[2], g_cov);

    for (int i = 0; i < consts::n_dim; ++i) {
        e_con[3][i] = delta(i, 3);
    }

    project_out(e_con[3], e_con[0], g_cov);
    project_out(e_con[3], e_con[1], g_cov);
    project_out(e_con[3], e_con[2], g_cov);
    normalize(e_con[3], g_cov);

    for (int i = 0; i < consts::n_dim; ++i) {
        lower(e_con[i], g_cov, e_cov[i]);
    }

    for (int i = 0; i < consts::n_dim; ++i) {
        e_cov[0][i] *= -1.0;
    }
}

static __device__ void lower(const double (&u_con)[consts::n_dim],
                             const double (&g_cov)[consts::n_dim][consts::n_dim],
                             double (&u_cov)[consts::n_dim]) {
    /* clang-format off */
    u_cov[0] = (
        g_cov[0][0] * u_con[0]
      + g_cov[0][1] * u_con[1]
      + g_cov[0][2] * u_con[2]
      + g_cov[0][3] * u_con[3]
    );
    u_cov[1] = (
        g_cov[1][0] * u_con[0]
      + g_cov[1][1] * u_con[1]
      + g_cov[1][2] * u_con[2]
      + g_cov[1][3] * u_con[3]
    );
    u_cov[2] = (
        g_cov[2][0] * u_con[0]
      + g_cov[2][1] * u_con[1]
      + g_cov[2][2] * u_con[2]
      + g_cov[2][3] * u_con[3]
    );
    u_cov[3] = (
        g_cov[3][0] * u_con[0]
      + g_cov[3][1] * u_con[1]
      + g_cov[3][2] * u_con[2]
      + g_cov[3][3] * u_con[3]
    );
    /* clang-format off */
}

static __device__ void normalize(double (&v_con)[consts::n_dim], const double (&g_cov)[consts::n_dim][consts::n_dim]) {
    double norm = 0.0;

    for (int i = 0; i < consts::n_dim; ++i) {
        for (int j = 0; j < consts::n_dim; ++j) {
            norm += v_con[i] * v_con[j] * g_cov[i][j];
        }
    }

    norm = std::sqrt(std::abs(norm));

    for (int i = 0; i < consts::n_dim; ++i) {
        v_con[i] /= norm;
    }
}

static __device__ void project_out(double (&v_con_a)[consts::n_dim],
                        double (&v_con_b)[consts::n_dim],
                        const double (&g_cov)[consts::n_dim][consts::n_dim]) {
    double v_con_b_sq = 0.0;

    for (int i = 0; i < consts::n_dim; ++i) {
        for (int j = 0; j < consts::n_dim; ++j) {
            v_con_b_sq += v_con_b[i] * v_con_b[j] * g_cov[i][j];
        }
    }

    double a_dot_b = 0.0;

    for (int i = 0; i < consts::n_dim; ++i) {
        for (int j = 0; j < consts::n_dim; ++j) {
            a_dot_b += v_con_a[i] * v_con_b[j] * g_cov[i][j];
        }
    }

    for (int i = 0; i < consts::n_dim; ++i) {
        v_con_a[i] -= v_con_b[i] * a_dot_b / v_con_b_sq;
    }
}

}; /* namespace cuda_tetrads */
