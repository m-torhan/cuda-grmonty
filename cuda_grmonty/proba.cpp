/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <random>
#include <tuple>

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/monty_rand.hpp"
#include "cuda_grmonty/proba.hpp"

namespace proba {

static double klein_nishina(double a, double ap);

void sample_electron_distr_p(const double (&k)[consts::n_dim], double (&p)[consts::n_dim], double theta_e) {
    double x1;
    double sigma_kn;
    double gamma_e;
    double beta_e;
    double mu;

    do {
        std::tie(gamma_e, beta_e) = sample_beta_distr(theta_e);
        mu = sample_mu_distr(beta_e);
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
                                                  (k_ * k_ - 2.0 * k_ - 2.0) / (2.0 * k_) * std::log(1.0 + 2.0 * k_));
        }

        x1 = monty_rand::rand();

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
    double v0 = std::sqrt(v0x * v0x + v0y * v0y + v0z * v0z);
    v0x /= v0;
    v0y /= v0;
    v0z /= v0;

    /* pick zero-angle for coordinate system */
    auto [n0x, n0y, n0z] = sample_rand_dir();
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
    double phi = monty_rand::rand() * 2.0 * std::numbers::pi; /* orient uniformly */
    double s_phi = std::sin(phi);
    double c_phi = std::cos(phi);

    double c_th = mu;
    double s_th = std::sqrt(1. - mu * mu);

    p[0] = gamma_e;
    p[1] = gamma_e * beta_e * (c_th * v0x + s_th * (c_phi * v1x + s_phi * v2x));
    p[2] = gamma_e * beta_e * (c_th * v0y + s_th * (c_phi * v1y + s_phi * v2y));
    p[3] = gamma_e * beta_e * (c_th * v0z + s_th * (c_phi * v1z + s_phi * v2z));

    if (beta_e < 0) {
        /* error */
    }
}

std::tuple<double, double> sample_beta_distr(double theta_e) {
    double y = sample_y_distr(theta_e);

    double gamma_e = y * y * theta_e + 1.0;
    double beta_e = std::sqrt(1.0 - 1.0 / (gamma_e * gamma_e));

    return {gamma_e, beta_e};
}

double sample_y_distr(double theta_e) {
    static std::mt19937 rd(123);

    double pi_3 = std::sqrt(std::numbers::pi) / 4.0;
    double pi_4 = std::sqrt(0.5 * theta_e) / 2.0;
    double pi_5 = 3.0 * std::sqrt(std::numbers::pi) * theta_e / 8.0;
    double pi_6 = theta_e * std::sqrt(0.5 * theta_e);

    double s_3 = pi_3 + pi_4 + pi_5 + pi_6;

    pi_3 /= s_3;
    pi_4 /= s_3;
    pi_5 /= s_3;
    pi_6 /= s_3;

    double y;
    double x2;
    double prob;

    do {
        double x1 = monty_rand::rand();
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

        std::chi_squared_distribution<double> chi_sq(dof);
        double x = chi_sq(rd);

        y = std::sqrt(x / 2.0);

        x2 = monty_rand::rand();
        double num = std::sqrt(1.0 + 0.5 * theta_e * y * y);
        double den = (1.0 + y * std::sqrt(0.5 * theta_e));

        prob = num / den;
    } while (x2 >= prob);

    return y;
}

double sample_mu_distr(double beta_e) {
    double x1 = monty_rand::rand();
    double det = 1.0 + 2.0 * beta_e + beta_e * beta_e - 4.0 * beta_e * x1;
    return (1.0 - std::sqrt(det)) / beta_e;
}

double sample_klein_nishina(double k0) {
    double k0pmin = k0 / (1.0 + 2.0 * k0);
    double k0pmax = k0;

    double x1;
    double k0p_tent;

    do {
        k0p_tent = k0pmin + (k0pmax - k0pmin) * monty_rand::rand();

        x1 = 2.0 * (1.0 + 2.0 * k0 + 2.0 * k0 * k0) / (k0 * k0 * (1.0 + 2.0 * k0));
        x1 *= monty_rand::rand();
    } while (x1 >= klein_nishina(k0, k0p_tent));

    return k0p_tent;
}

double sample_thomson() {
    double x1, x2;

    do {
        x1 = 2.0 * monty_rand::rand() - 1.0;
        x2 = (3.0 / 4.0) * monty_rand::rand();
    } while (x2 >= (3.0 / 8.0) * (1.0 + x1 * x1));

    return x1;
}

std::tuple<double, double, double> sample_rand_dir() {
    double z = monty_rand::rand() * 2.0 - 1.0;
    double phi = monty_rand::rand() * 2.0 * std::numbers::pi;

    double x = std::sqrt(1.0 - z * z) * std::cos(phi);
    double y = std::sqrt(1.0 - z * z) * std::sin(phi);

    return {x, y, z};
}

static double klein_nishina(double a, double ap) {
    double ch = 1.0 + 1.0 / a - 1.0 / ap;
    return (a / ap + ap / a - 1.0 + ch * ch) / (a * a);
}

}; /* namespace proba */
