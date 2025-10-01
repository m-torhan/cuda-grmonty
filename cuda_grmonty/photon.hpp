/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/consts.hpp"

namespace photon {

/**
 * @brief Representation of a photon in the simulation.
 *
 * Stores position, momentum, weights, interaction history, and plasma properties at emission and scattering points.
 * Used as the fundamental particle in Monte Carlo radiative transfer.
 */
struct Photon {
    double x[consts::n_dim];      /* Photon position 4-vector (x^μ). */
    double k[consts::n_dim];      /* Photon momentum 4-vector (k^μ). */
    double dkdlam[consts::n_dim]; /* Derivative of momentum with respect to affine parameter (dk^μ/dλ). */
    double w;                     /* Photon statistical weight (number of physical photons represented). */
    double e;                     /* Photon energy measured in the coordinate frame. */
    double l;                     /* Photon angular momentum (conserved in axisymmetric spacetimes). */
    double x1i;                   /* Initial radial coordinate at emission. */
    double x2i;                   /* Initial polar coordinate at emission. */
    double tau_abs;               /* Accumulated absorption optical depth. */
    double tau_scatt;             /* Accumulated scattering optical depth. */
    double n_e_0;                 /* Electron number density at emission. */
    double theta_e_0;             /* Electron dimensionless temperature at emission. */
    double b_0;                   /* Magnetic field strength at emission. */
    double e_0;                   /* Photon energy at emission (fluid frame). */
    double e_0_s;                 /* Photon energy at emission (scaled/normalized for tables). */
    int n_scatt;                  /* Number of scatterings experienced by the photon. */
};

/**
 * @brief Reduced Photon struct including only the fields that are initialized during photon creation.
 */
struct InitPhoton {
    double x[consts::n_dim];
    double k[consts::n_dim];
    double w;
    double e;
    double l;
    double n_e_0;
    double theta_e_0;
    double b_0;
    double e_0;
    int n_scatt;
};

}; /* namespace photon */
