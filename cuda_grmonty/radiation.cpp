/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>

#include "spdlog/spdlog.h"

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/hotcross.hpp"
#include "cuda_grmonty/jnu_mixed.hpp"
#include "cuda_grmonty/ndarray.hpp"
#include "cuda_grmonty/radiation.hpp"

namespace radiation {

static double b_nu_inv(double nu, double theta_e);

static double jnu_inv(double nu,
                      double theta_e,
                      double n_e,
                      double b,
                      double theta,
                      const std::array<double, consts::n_e_samp + 1> &k2_table);

static double kappa_es(double nu, double theta_e, const ndarray::NDArray<double, 2> &hotcross_table);

double bk_angle(const double (&x)[consts::n_dim],
                const double (&k)[consts::n_dim],
                const double (&u_cov)[consts::n_dim],
                const double (&b_cov)[consts::n_dim],
                double b,
                double b_unit) {
    if (b == 0.0) {
        return std::numbers::pi / 2.0;
    }

    /* clang-format off */
    double k_ = std::abs(
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

    mu = std::clamp(mu, -1.0, 1.0);

    return std::acos(mu);
}

double
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

double alpha_inv_scatt(double nu, double theta_e, double n_e, const ndarray::NDArray<double, 2> &hotcross_table) {
    double kappa = kappa_es(nu, theta_e, hotcross_table);

    return nu * kappa * n_e * consts::mp;
}

double alpha_inv_abs(double nu,
                     double theta_e,
                     double n_e,
                     double b,
                     double theta,
                     const std::array<double, consts::n_e_samp + 1> &k2_table) {
    double j = jnu_inv(nu, theta_e, n_e, b, theta, k2_table);
    double b_nu = b_nu_inv(nu, theta_e);

    return j / (b_nu + 1.0e-100);
}

static double b_nu_inv(double nu, double theta_e) {
    double x = consts::hpl * nu / (consts::me * consts::cl * consts::cl * theta_e);

    if (x < 1.0e-3) {
        return (2.0 * consts::hpl / (consts::cl * consts::cl)) / (x / 24.0 * (24.0 + x * (12.0 + x * (4.0 + x))));
    }

    return (2.0 * consts::hpl / (consts::cl * consts::cl)) / (std::exp(x) - 1.0);
}

static double jnu_inv(double nu,
                      double theta_e,
                      double n_e,
                      double b,
                      double theta,
                      const std::array<double, consts::n_e_samp + 1> &k2_table) {
    double j = jnu_mixed::synch(nu, n_e, theta_e, b, theta, k2_table);

    return j / (nu * nu);
}

static double kappa_es(double nu, double theta_e, const ndarray::NDArray<double, 2> &hotcross_table) {
    double e_g = consts::hpl * nu / (consts::me * consts::cl * consts::cl);

    return hotcross::total_compton_cross_lkup(e_g, theta_e, hotcross_table) / consts::mp;
}

}; /* namespace radiation */
