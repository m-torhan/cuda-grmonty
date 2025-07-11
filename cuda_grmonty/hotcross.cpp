/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>

#include "spdlog/spdlog.h"

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/hotcross.hpp"
#include "cuda_grmonty/mathfn.hpp"

namespace hotcross {

static double hc_klein_nishina(double w);

static double dnd_gamma_e(double theta_e, double gamma_e);

static double boostcross(double w, double mu_e, double gamma_e);

double total_compton_cross_num(double w, double theta_e) {
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

    const double d_mu_e = consts::hotcross::d_mu_e;
    const double d_gamma_e = theta_e * consts::hotcross::d_gamma_e;

    /* integrate over mu_e, gamma_e, where mu_e is the cosine of the
       angle between k and u_e, and the angle k is assumed to lie,
       wlog, along the z axis */
    double cross = 0.0;

    for (double mu_e = -1.0 + 0.5 * d_mu_e; mu_e < 1.0; mu_e += d_mu_e) {
        for (double gamma_e = 1.0 + 0.5 * d_gamma_e; gamma_e < 1.0 + consts::hotcross::max_gamma * theta_e;
             gamma_e += d_gamma_e) {
            double f = 0.5 * dnd_gamma_e(theta_e, gamma_e);

            cross += d_mu_e * d_gamma_e * boostcross(w, mu_e, gamma_e) * f;

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
        k2f = mathfn::bessel_Kn(2, 1.0 / theta_e) * std::exp(1.0 / theta_e);
    } else {
        k2f = std::sqrt(std::numbers::pi * theta_e / 2.0);
    }

    return ((gamma_e * std::sqrt(gamma_e * gamma_e - 1.) / (theta_e * k2f)) * std::exp(-(gamma_e - 1.) / theta_e));
}

static double boostcross(double w, double mu_e, double gamma_e) {
    double we, boostcross, v;
    double hc_klein_nishina(double we);

    /* energy in electron rest frame */
    v = sqrt(gamma_e * gamma_e - 1.0) / gamma_e;
    we = w * gamma_e * (1.0 - mu_e * v);

    boostcross = hc_klein_nishina(we) * (1.0 - mu_e * v);

    // if (boostcross > 2) {
    //     fprintf(stderr, "w,mue,gammae: %g %g %g\n", w, mu_e, gamma_e);
    //     fprintf(stderr, "v,we, boostcross: %g %g %g\n", v, we, boostcross);
    //     fprintf(stderr, "kn: %g %g %g\n", v, we, boostcross);
    // }

    if (std::isnan(boostcross)) {
        spdlog::error("boostcross is nan: %f %f %f\n", w, mu_e, gamma_e);
    }

    return boostcross;
}

}; /* namespace hotcross */
