/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/consts.hpp"

namespace photon {

struct Photon {
    double x[consts::n_dim];
    double k[consts::n_dim];
    double dkdlam[consts::n_dim];
    double w;
    double e;
    double l;
    double x1i;
    double x2i;
    double tau_abs;
    double tau_scatt;
    double n_e_0;
    double theta_e_0;
    double b_0;
    double e_0;
    double e_0_s;
    int n_scatt;
};

}; /* namespace photon */
