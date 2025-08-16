/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>

#include "spdlog/spdlog.h"

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/hotcross.hpp"
#include "cuda_grmonty/ndarray.hpp"
#ifdef CUDA
#include "cuda_grmonty/hotcross.cuh"
#endif /* CUDA */

namespace hotcross {

/**
 * @brief Computes the total angle-averaged Compton scattering cross-section.
 *
 * @param w Dimensionless photon energy (w = h * nu / (m_e * c^2)).
 * @param thetae Dimensionless electron temperature (thetae = k * T_e / (m_e * c^2)).
 *
 * @return Total angle-averaged Compton scattering cross-section, in units of the Thomson cross-section (sigma_T).
 */
static double total_compton_cross_num(double w, double theta_e);

static double hc_klein_nishina(double w);

static double dnd_gamma_e(double theta_e, double gamma_e);

static double boostcross(double w, double mu_e, double gamma_e);

void init_table(ndarray::NDArray<double, 2> &table) {
    spdlog::info("Initializing HARM model hotcross");

/* TODO: Add some better switching between implementations */
#ifdef CUDA
    cuda_hotcross::init_table(table);
#else  /* CUDA */
    for (int i = 0; i <= consts::hotcross::n_w; ++i) {
        spdlog::debug("{} / {}", i, consts::hotcross::n_w);
        for (int j = 0; j <= consts::hotcross::n_t; ++j) {
            double l_w = consts::hotcross::l_min_w + i * consts::hotcross::d_l_w;
            double l_t = consts::hotcross::l_min_t + j * consts::hotcross::d_l_t;

            table(i, j) = std::log10(hotcross::total_compton_cross_num(std::pow(10.0, l_w), std::pow(10.0, l_t)));
        }
    }
#endif /* CUDA */

    spdlog::info("Initializing HARM model hotcross done");
}

double total_compton_cross_lkup(double w, double theta_e, const ndarray::NDArray<double, 2> &hotcross_table) {
    if (w * theta_e < 1.0e-6) {
        return consts::sigma_thomson;
    }

    if (theta_e < consts::hotcross::min_t) {
        return hc_klein_nishina(w) * consts::sigma_thomson;
    }

    if (w <= consts::hotcross::min_w || w >= consts::hotcross::max_w || theta_e <= consts::hotcross::min_t ||
        theta_e >= consts::hotcross::max_t) {
        return total_compton_cross_num(w, theta_e);
    }

    const double l_w = std::log10(w);
    const double l_t = std::log10(theta_e);
    int i = static_cast<int>((l_w - consts::hotcross::l_min_w) / consts::hotcross::d_l_w);
    int j = static_cast<int>((l_t - consts::hotcross::l_min_t) / consts::hotcross::d_l_t);
    double d_i = (l_w - consts::hotcross::l_min_w) / consts::hotcross::d_l_w - i;
    double d_j = (l_t - consts::hotcross::l_min_t) / consts::hotcross::d_l_t - j;

    double l_cross = (1.0 - d_i) * (1.0 - d_j) * hotcross_table(i, j) + d_i * (1.0 - d_j) * hotcross_table(i + 1, j) +
                     (1.0 - d_i) * d_j * hotcross_table(i, j + 1) + d_i * d_j * hotcross_table(i + 1, j + 1);

    return std::pow(10, l_cross);
}

static double total_compton_cross_num(double w, double theta_e) {
    if (std::isnan(w)) {
        spdlog::error("Compton cross is nan: %f %f", w, theta_e);
        return 0.0;
    }

    if (theta_e < consts::hotcross::min_t && w < consts::hotcross::min_w) {
        return consts::sigma_thomson;
    }
    if (theta_e < consts::hotcross::min_t) {
        return hc_klein_nishina(w) * consts::sigma_thomson;
    }

    /* integrate over mu_e, gamma_e, where mu_e is the cosine of the
       angle between k and u_e, and the angle k is assumed to lie,
       wlog, along the z axis */
    double cross = 0.0;

    for (double mu_e = -1.0 + 0.5 * consts::hotcross::d_mu_e; mu_e < 1.0; mu_e += consts::hotcross::d_mu_e) {
        for (double gamma_e = 1.0 + 0.5 * theta_e * consts::hotcross::d_gamma_e;
             gamma_e < 1.0 + consts::hotcross::max_gamma * theta_e;
             gamma_e += theta_e * consts::hotcross::d_gamma_e) {
            double f = 0.5 * dnd_gamma_e(theta_e, gamma_e);

            cross +=
                theta_e * consts::hotcross::d_mu_e * consts::hotcross::d_gamma_e * boostcross(w, mu_e, gamma_e) * f;

            if (std::isnan(cross)) {
                spdlog::error("cross is nan");
            }
        }
    }

    return cross * consts::sigma_thomson;
}

static double hc_klein_nishina(double w) {
    if (w < 1.0e-3) {
        return (1.0 - 2.0 * w);
    }

    return (3.0 / 4.0) * (2.0 / (w * w) + (1.0 / (2.0 * w) - (1.0 + w) / (w * w * w)) * std::log(1.0 + 2.0 * w) +
                          (1.0 + w) / ((1.0 + 2.0 * w) * (1.0 + 2.0 * w)));
}

static double dnd_gamma_e(double theta_e, double gamma_e) {
    double k2f;

    if (theta_e > 1.0e-2) {
        k2f = std::cyl_bessel_k(2, 1.0 / theta_e) * std::exp(1.0 / theta_e);
    } else {
        k2f = std::sqrt(std::numbers::pi * theta_e / 2.0);
    }

    return ((gamma_e * std::sqrt(gamma_e * gamma_e - 1.) / (theta_e * k2f)) * std::exp(-(gamma_e - 1.) / theta_e));
}

static double boostcross(double w, double mu_e, double gamma_e) {
    double we, boostcross, v;

    /* energy in electron rest frame */
    v = sqrt(gamma_e * gamma_e - 1.0) / gamma_e;
    we = w * gamma_e * (1.0 - mu_e * v);

    boostcross = hc_klein_nishina(we) * (1.0 - mu_e * v);

    if (std::isnan(boostcross)) {
        spdlog::error("boostcross({}, {}, {}) is nan", w, mu_e, gamma_e);
    }

    return boostcross;
}

}; /* namespace hotcross */
