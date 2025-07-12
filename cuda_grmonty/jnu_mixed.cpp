/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>

#include "spdlog/spdlog.h"

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/jnu_mixed.hpp"
#include "cuda_grmonty/mathfn.hpp"
#include "cuda_grmonty/ndarray.hpp"

namespace jnu_mixed {

static double jnu_integrand(double th, double k);

static double emiss_table_f(double k);

void init_emiss_tables(ndarray::NDArray<double> &f, ndarray::NDArray<double> &k2) {
    spdlog::info("Initializing HARM model emission tables");

    static const double l_min_k = std::log(consts::jnu::min_k);
    static const double l_min_t = std::log(consts::jnu::min_t);
    static const double d_l_k = std::log(consts::jnu::max_k / consts::jnu::min_k) / consts::n_e_samp;
    static const double d_l_t = std::log(consts::jnu::max_t / consts::jnu::min_t) / consts::n_e_samp;

    for (int i = 0; i <= consts::n_e_samp; ++i) {
        double k = std::exp(i * d_l_k + l_min_k);
        f[{i}] = jnu_mixed::emiss_table_f(k);
    }

    for (int i = 0; i <= consts::n_e_samp; ++i) {
        double t = std::exp(i * d_l_t + l_min_t);
        k2[{i}] = std::log(mathfn::bessel_Kn(2, 1.0 / t));
    }

    spdlog::info("Initializing HARM model emission tables done");
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

static double emiss_table_f(double k) { return 0.0; }

}; /* namespace jnu_mixed */
