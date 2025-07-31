/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <numbers>

#include "spdlog/spdlog.h"

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/integration.hpp"
#include "cuda_grmonty/jnu_mixed.hpp"
#include "cuda_grmonty/mathfn.hpp"
#include "cuda_grmonty/ndarray.hpp"

namespace jnu_mixed {

static double jnu_integrand(double th, double k);

static double emiss_table_f(double k);

static double linear_interp_k2(double theta_e, const ndarray::NDArray<double> &k2_table);

static double linear_interp_f(double k, const ndarray::NDArray<double> &f_table);

void init_emiss_tables(ndarray::NDArray<double> &f, ndarray::NDArray<double> &k2) {
    spdlog::info("Initializing HARM model emission tables");

    static const double l_min_k = std::log(consts::jnu::min_k);
    static const double l_min_t = std::log(consts::jnu::min_t);
    static const double d_l_k = std::log(consts::jnu::max_k / consts::jnu::min_k) / consts::n_e_samp;
    static const double d_l_t = std::log(consts::jnu::max_t / consts::jnu::min_t) / consts::n_e_samp;

    for (int i = 0; i <= consts::n_e_samp; ++i) {
        spdlog::debug("{} / {}", i, consts::n_e_samp);
        double k = std::exp(i * d_l_k + l_min_k);
        f[{i}] = jnu_mixed::emiss_table_f(k);
    }

    for (int i = 0; i <= consts::n_e_samp; ++i) {
        spdlog::debug("{} / {}", i, consts::n_e_samp);
        double t = std::exp(i * d_l_t + l_min_t);
        k2[{i}] = std::log(mathfn::bessel_Kn(2, 1.0 / t));
    }

    spdlog::info("Initializing HARM model emission tables done");
}

double synch(double nu, double n_e, double theta_e, double b, double theta, const ndarray::NDArray<double> &k2_table) {
    if (theta_e < consts::theta_e_min) {
        return 0.0;
    }

    double k2 = k2_eval(theta_e, k2_table);
    double nu_c = consts::ee * b / (2.0 * std::numbers::pi * consts::me * consts::cl);
    double sin_th = std::sin(theta);
    double nu_s = (2.0 / 9.0) * nu_c * theta_e * theta_e * sin_th;

    if (nu > 1.0e12 * nu_s) {
        return 0.0;
    }

    double x = nu / nu_s;
    double xp = std::pow(x, 1.0 / 3.0);
    double xx = std::sqrt(x) + consts::jnu::cst * std::sqrt(xp);
    double f = xx * xx;
    return (std::numbers::sqrt2 * std::numbers::pi * consts::ee * consts::ee * n_e * nu_s / (3.0 * consts::cl * k2)) *
           f * std::exp(-xp);
}

double k2_eval(double theta_e, const ndarray::NDArray<double> &k2_table) {
    if (theta_e < consts::theta_e_min) {
        return 0.0;
    }
    if (theta_e > consts::jnu::max_t) {
        return 2.0 * theta_e * theta_e;
    }

    return linear_interp_k2(theta_e, k2_table);
}

double f_eval(double theta_e, double b_mag, double nu, const ndarray::NDArray<double> &f_table) {
    double k = consts::jnu::k_fac * nu / (b_mag * theta_e * theta_e);

    if (k > consts::jnu::max_k) {
        return 0.0;
    }
    if (k < consts::jnu::min_k) {
        double x = std::pow(k, 1.0 / 3.0);
        return x * (37.67503800178 + 2.240274341836 * x);
    }

    return linear_interp_f(k, f_table);
}

static double jnu_integrand(double th, double k) {
    double sin_th = std::sin(th);
    double x = k / sin_th;

    if (sin_th < 1.0e-150 || x > 2.0e8) {
        return 0.0;
    }

    return sin_th * sin_th * std::pow(std::sqrt(x) + consts::jnu::cst * std::pow(x, 1.0 / 6.0), 2.0) *
           std::exp(-std::pow(x, 1.0 / 3.0));
}

static double emiss_table_f(double k) {
    double result = integration::gauss_kronrod_61([k](double th) { return jnu_integrand(th, k); },
                                                  0,
                                                  std::numbers::pi / 2.0,
                                                  consts::jnu::eps_abs,
                                                  consts::jnu::eps_rel,
                                                  1000);

    return std::log(4 * std::numbers::pi * result);
}

static double linear_interp_k2(double theta_e, const ndarray::NDArray<double> &k2_table) {
    static const double l_min_t = std::log(consts::jnu::min_t);
    static const double d_l_t = std::log(consts::jnu::max_t / consts::jnu::min_t) / consts::n_e_samp;

    double l_t = std::log(theta_e);
    double d_i = (l_t - l_min_t) / d_l_t;
    int i = static_cast<int>(d_i);

    d_i -= i;

    return std::exp((1.0 - d_i) * k2_table[{i}].value() + d_i * k2_table[{i + 1}].value());
}

static double linear_interp_f(double k, const ndarray::NDArray<double> &f_table) {
    static const double l_min_k = std::log(consts::jnu::min_k);
    static const double d_l_k = std::log(consts::jnu::max_k / consts::jnu::min_k) / consts::n_e_samp;

    double l_k = std::log(k);
    double d_i = (l_k - l_min_k) / d_l_k;
    int i = static_cast<int>(d_i);

    d_i -= i;

    return std::exp((1.0 - d_i) * f_table[{i}].value() + d_i * f_table[{i + 1}].value());
}

}; /* namespace jnu_mixed */
