/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuda_harm {

struct Data {
    double *k_rho; /* rest-mass density */
    double *u;     /* internal eneergy density */
    double *u_1;   /* covariant velocity components */
    double *u_2;
    double *u_3;
    double *b_1; /* contravariant magnetic field components */
    double *b_2;
    double *b_3;
};

struct Tables {
    double *hotcross_table;
    double *f;
    double *k2;
};

}; /* namespace cuda_harm */
