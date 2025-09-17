/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <math_constants.h>

#include "cuda_grmonty/tetrads.cuh"

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/harm_data.hpp"

namespace cuda_harm {

static __device__ void
gcon_func(const harm::Header *header, const double (&x)[consts::n_dim], double (&g_con)[consts::n_dim][consts::n_dim]);

static __device__ void
gcov_func(const harm::Header *header, const double (&x)[consts::n_dim], double (&g_cov)[consts::n_dim][consts::n_dim]);

static __device__ harm::FluidParams get_fluid_params(const harm::Header *header,
                                                     const harm::Units *units,
                                                     const double *__restrict__ k_rho,
                                                     const double *__restrict__ u,
                                                     const double *__restrict__ u_1,
                                                     const double *__restrict__ u_2,
                                                     const double *__restrict__ u_3,
                                                     const double *__restrict__ b_1,
                                                     const double *__restrict__ b_2,
                                                     const double *__restrict__ b_3,
                                                     const double (&x)[consts::n_dim],
                                                     const double (&g_cov)[consts::n_dim][consts::n_dim]);

static __device__ harm::BLCoord get_bl_coord(const harm::Header *header, const double (&x)[consts::n_dim]);

static __device__ void
x_to_ij(const harm::Header *header, const double (&x)[consts::n_dim], int &i, int &j, double &del_i, double &del_j);

static __device__ double
interp_scalar(const double *var, int var_n, int i, int j, const double (&coeff)[consts::n_dim]);

static __device__ void
gcon_func(const harm::Header *header, const double (&x)[consts::n_dim], double (&g_con)[consts::n_dim][consts::n_dim]) {
    harm::BLCoord bl_coord = get_bl_coord(header, x);

    double sin_theta = fabs(sin(bl_coord.theta)) + consts::eps;
    double cos_theta = cos(bl_coord.theta);

    double irho2 = 1.0 / (bl_coord.r * bl_coord.r + header->a * header->a * cos_theta * cos_theta);

    double hfac = CUDART_PI + (1.0 - header->h_slope) * CUDART_PI * std::cos(2.0 * CUDART_PI * x[2]);

    g_con[0][0] = -1.0 - 2.0 * bl_coord.r * irho2;
    g_con[0][1] = 2.0 * irho2;
    g_con[0][2] = 0.0;
    g_con[0][3] = 0.0;

    g_con[1][0] = g_con[0][1];
    g_con[1][1] = irho2 * (bl_coord.r * (bl_coord.r - 2.0) + header->a * header->a) / (bl_coord.r * bl_coord.r);
    g_con[1][2] = 0.0;
    g_con[1][3] = header->a * irho2 / bl_coord.r;

    g_con[2][0] = 0.0;
    g_con[2][1] = 0.0;
    g_con[2][2] = irho2 / (hfac * hfac);
    g_con[2][3] = 0.0;

    g_con[3][0] = 0.0;
    g_con[3][1] = g_con[1][3];
    g_con[3][2] = 0.0;
    g_con[3][3] = irho2 / (sin_theta * sin_theta);
}

static __device__ void
gcov_func(const harm::Header *header, const double (&x)[consts::n_dim], double (&g_cov)[consts::n_dim][consts::n_dim]) {
    harm::BLCoord bl_coord = get_bl_coord(header, x);

    double sin_theta = std::fabs(std::sin(bl_coord.theta)) + consts::eps;
    double cos_theta = std::cos(bl_coord.theta);

    double sin_theta_2 = sin_theta * sin_theta;
    double rho2 = bl_coord.r * bl_coord.r + header->a * header->a * cos_theta * cos_theta;

    double tfac = 1.0;
    double rfac = bl_coord.r - header->r_0;
    double hfac = CUDART_PI + (1.0 - header->h_slope) * CUDART_PI * std::cos(2.0 * CUDART_PI * x[2]);
    double pfac = 1.0;

    g_cov[0][0] = (-1.0 + 2.0 * bl_coord.r / rho2) * tfac * tfac;
    g_cov[0][1] = (2.0 * bl_coord.r / rho2) * tfac * rfac;
    g_cov[0][2] = 0.0;
    g_cov[0][3] = (-2.0 * header->a * bl_coord.r * sin_theta_2 / rho2) * tfac * pfac;

    g_cov[1][0] = g_cov[0][1];
    g_cov[1][1] = (1.0 + 2.0 * bl_coord.r / rho2) * rfac * rfac;
    g_cov[1][2] = 0.0;
    g_cov[1][3] = (-header->a * sin_theta_2 * (1.0 + 2.0 * bl_coord.r / rho2)) * rfac * pfac;

    g_cov[2][0] = 0.0;
    g_cov[2][1] = 0.0;
    g_cov[2][2] = rho2 * hfac * hfac;
    g_cov[2][3] = 0.0;

    g_cov[3][0] = g_cov[0][3];
    g_cov[3][1] = g_cov[1][3];
    g_cov[3][2] = 0.0;
    g_cov[3][3] =
        sin_theta_2 * (rho2 + header->a * header->a * sin_theta_2 * (1.0 + 2.0 * bl_coord.r / rho2)) * pfac * pfac;
}

static __device__ harm::FluidParams get_fluid_params(const harm::Header *header,
                                                     const harm::Units *units,
                                                     const double *__restrict__ k_rho,
                                                     const double *__restrict__ u,
                                                     const double *__restrict__ u_1,
                                                     const double *__restrict__ u_2,
                                                     const double *__restrict__ u_3,
                                                     const double *__restrict__ b_1,
                                                     const double *__restrict__ b_2,
                                                     const double *__restrict__ b_3,
                                                     const double (&x)[consts::n_dim],
                                                     const double (&g_cov)[consts::n_dim][consts::n_dim]) {
    struct harm::FluidParams fluid_params;

    if (x[1] < header->x_start[1] || x[1] > header->x_stop[1] || x[2] < header->x_start[2] ||
        x[2] > header->x_stop[2]) {
        fluid_params.n_e = 0.0;
        return fluid_params;
    }

    int i, j;
    double del_i, del_j;

    x_to_ij(header, x, i, j, del_i, del_j);

    double coeff[consts::n_dim] = {
        (1.0 - del_i) * (1.0 - del_j),
        (1.0 - del_i) * del_j,
        del_i * (1.0 - del_j),
        del_i * del_j,
    };

    double rho = interp_scalar(k_rho, header->n[1], i, j, coeff);
    double uu = interp_scalar(u, header->n[1], i, j, coeff);

    fluid_params.n_e = rho * units->n_e_unit;
    fluid_params.theta_e = uu / rho * units->theta_e_unit;

    double bp[consts::n_dim] = {
        0.0,
        interp_scalar(b_1, header->n[1], i, j, coeff),
        interp_scalar(b_2, header->n[1], i, j, coeff),
        interp_scalar(b_3, header->n[1], i, j, coeff),
    };

    double v_con[consts::n_dim] = {
        0.0,
        interp_scalar(u_1, header->n[1], i, j, coeff),
        interp_scalar(u_2, header->n[1], i, j, coeff),
        interp_scalar(u_3, header->n[1], i, j, coeff),
    };

    double g_con[consts::n_dim][consts::n_dim];

    gcon_func(header, x, g_con);

    double v_dot_v = 0.0;

    for (int i = 1; i < consts::n_dim; ++i) {
        for (int j = 1; j < consts::n_dim; ++j) {
            v_dot_v += g_cov[i][j] * v_con[i] * v_con[j];
        }
    }

    double v_fac = sqrt(-1.0 / g_con[0][0] * (1.0 + abs(v_dot_v)));

    fluid_params.u_con[0] = -v_fac * g_con[0][0];

    for (int i = 1; i < consts::n_dim; ++i) {
        fluid_params.u_con[i] = v_con[i] - v_fac * g_con[0][i];
    }
    cuda_tetrads::lower(fluid_params.u_con, g_cov, fluid_params.u_cov);

    double u_dot_bp = 0.0;
    for (int i = 1; i < consts::n_dim; ++i) {
        u_dot_bp += fluid_params.u_cov[i] * bp[i];
    }
    fluid_params.b_con[0] = u_dot_bp;
    for (int i = 1; i < consts::n_dim; ++i) {
        fluid_params.b_con[i] = (bp[i] + fluid_params.u_con[i] * u_dot_bp) / fluid_params.u_con[0];
    }
    cuda_tetrads::lower(fluid_params.b_con, g_cov, fluid_params.b_cov);

    fluid_params.b =
        sqrt(fluid_params.b_con[0] * fluid_params.b_cov[0] + fluid_params.b_con[1] * fluid_params.b_cov[1] +
                  fluid_params.b_con[2] * fluid_params.b_cov[2] + fluid_params.b_con[3] * fluid_params.b_cov[3]) *
        units->b_unit;

    return fluid_params;
}

static __device__ harm::BLCoord get_bl_coord(const harm::Header *header, const double (&x)[consts::n_dim]) {
    return harm::BLCoord{
        .r = exp(x[1]) + header->r_0,
        .theta = CUDART_PI * x[2] + ((1.0 - header->h_slope) / 2.0) * sin(2.0 * CUDART_PI * x[2]),
    };
}

static __device__ void
x_to_ij(const harm::Header *header, const double (&x)[consts::n_dim], int &i, int &j, double &del_i, double &del_j) {
    i = static_cast<int>((x[1] - header->x_start[1]) / header->dx[1] - 0.5 + 1000) - 1000;
    j = static_cast<int>((x[2] - header->x_start[2]) / header->dx[2] - 0.5 + 1000) - 1000;

    if (i < 0) {
        i = 0;
        del_i = 0.0;
    } else if (i > static_cast<int>(header->n[0]) - 2) {
        i = header->n[0] - 2;
        del_i = 1.0;
    } else {
        del_i = (x[1] - ((i + 0.5) * header->dx[1] + header->x_start[1])) / header->dx[1];
    }

    if (j < 0) {
        j = 0;
        del_j = 0.0;
    } else if (j > static_cast<int>(header->n[1]) - 2) {
        j = header->n[1] - 2;
        del_j = 1.0;
    } else {
        del_j = (x[2] - ((j + 0.5) * header->dx[2] + header->x_start[2])) / header->dx[2];
    }
}

static __device__ double
interp_scalar(const double *var, int var_n, int i, int j, const double (&coeff)[consts::n_dim]) {
    /* clang-format off */
    return (
        var[(i    ) * var_n + j    ] * coeff[0] +
        var[(i    ) * var_n + j + 1] * coeff[1] +
        var[(i + 1) * var_n + j    ] * coeff[2] +
        var[(i + 1) * var_n + j + 1] * coeff[3]
    );
    /* clang-format on */
}

}; /* namespace cuda_harm */
