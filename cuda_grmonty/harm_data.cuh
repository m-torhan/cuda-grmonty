/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @brief CUDA data structures for GRMHD (HARM) simulations.
 */
namespace cuda_harm {

/**
 * @brief Device pointers to simulation field arrays.
 */
struct Data {
    double *k_rho; /* Rest-mass density */
    double *u;     /* Internal eneergy density */
    double *u_1;   /* Covariant velocity components */
    double *u_2;
    double *u_3;
    double *b_1; /* Contravariant magnetic field components */
    double *b_2;
    double *b_3;
};

/**
 * @brief Device pointers to tabulated data.
 */
struct Tables {
    double *hotcross_table; /* Hot cross-section lookup table */
    double *f;              /* Auxiliary function table */
    double *k2;             /* Precomputed k^2 values */
};

}; /* namespace cuda_harm */
