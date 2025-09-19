/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/ndarray.hpp"

/**
 * @brief General relativistic MHD (HARM) simulation data structures.
 */
namespace harm {

/**
 * @brief Simulation header and metadata.
 */
struct Header {
    double t;          /* Simulation time */
    int n[2];          /* Number of grid points in x1 and x2 directions  */
    double x_start[4]; /* Start coordinates of the grid */
    double x_stop[4];  /* Stop coordinates of the grid */
    double dx[4];      /* Grid spacing */
    double t_final;    /* Final simulation time */
    int n_step;        /* Number of simulation steps */
    double a;          /* Black hole spin parameter (dimensionless Kerr parameter) */
    double gamma;      /* Adiabatic index */
    double courant;    /* Courant number for time-stepping */
    double dt_dump;    /* Time interval between dumps */
    double dt_log;     /* Time interval for log outputs */
    double dt_img;     /* Time interval for image outputs */
    int dt_rdump;      /* Time interval of restart dumps written */
    int cnt_dump;      /* Counter for number of dumps written */
    int cnt_img;       /* Counter for number of images written */
    int cnt_rdump;     /* Counter for number of restart dumps written */
    double dt;         /* Time step size */
    int lim;           /* Slope limiter method used */
    int failed;        /* Number of failed steps in simulation */
    double r_in;       /* Inner radius of the simulation domain */
    double r_out;      /* Outer radius of the simulation domain */
    double h_slope;    /* Grid stretching parameters */
    double r_0;        /* Reference radius */
};

/**
 * @brief Hydrodynamic and magnetic field data arrays.
 */
struct Data {
    ndarray::NDArray<double, 2> k_rho; /* Rest-mass density */
    ndarray::NDArray<double, 2> u;     /* Internal eneergy density */
    ndarray::NDArray<double, 2> u_1;   /* Covariant velocity components */
    ndarray::NDArray<double, 2> u_2;
    ndarray::NDArray<double, 2> u_3;
    ndarray::NDArray<double, 2> b_1; /* Contravariant magnetic field components */
    ndarray::NDArray<double, 2> b_2;
    ndarray::NDArray<double, 2> b_3;
};

/**
 * @brief Unit system for physical quantities.
 */
struct Units {
    double mass_unit;    /* Mass unit */
    double l_unit;       /* Length unit */
    double t_unit;       /* Time unit */
    double rho_unit;     /* Density unit */
    double u_unit;       /* Energy density unit */
    double b_unit;       /* Magnetic field unit */
    double theta_e_unit; /* Electron temperature unit */
    double n_e_unit;     /* Electron number density unit */
};

/**
 * @brief Metric tensors and determinant for the simulation grid.
 */
struct Geometry {
    ndarray::NDArray<double, 4> cov; /* Covariant metric tensor g_μν */
    ndarray::NDArray<double, 4> con; /* Contravariant metric tensor g^μν */
    ndarray::NDArray<double, 2> det; /* Determinant of the metric */
};

/**
 * @brief Boyer-Lindquist coordinates (r, θ).
 */
struct BLCoord {
    double r;     /* Radial coordinate */
    double theta; /* Polar angle */
};

/**
 * @brief Fluid properties in a single simulation zone (contravariant only).
 */
struct FluidZone {
    double n_e;                  /* Electron number density */
    double theta_e;              /* Dimensionless electron temperature */
    double b;                    /* Magnetic field magnitude */
    double u_con[consts::n_dim]; /* Contravariant 4-velocity */
    double b_con[consts::n_dim]; /* Contravariant magnetic field */
};

/**
 * @brief Full fluid properties in a zone, including covariant vectors.
 */
struct FluidParams {
    double n_e;                  /* Electron number density */
    double theta_e;              /* Dimensionless electron temperature */
    double b;                    /* Magnetic field magnitude */
    double u_con[consts::n_dim]; /* Contravariant 4-velocity */
    double u_cov[consts::n_dim]; /* Covariant 4-velocity */
    double b_con[consts::n_dim]; /* Contravariant magnetic field */
    double b_cov[consts::n_dim]; /* Covariant magnetic field */
};

/**
 * @brief Monte Carlo sampling zone definition.
 */
struct Zone {
    int x_1;        /* Grid index in x1 direction */
    int x_2;        /* Grid index in x2 direction */
    int num_to_gen; /* Number of superphotons to generate */
    double dn_max;  /* Maximum differential number */
    bool quit_flag; /* Quit condition flag */
};

/**
 * @brief Photon spectrum diagnostics.
 */
struct Spectrum {
    double dn_dle;    /* dN/dlnE: photon count per logarithmic energy bin */
    double de_dle;    /* dE/dlnE: energy per logarithmic energy bin */
    double nph;       /* Number of photons */
    double nscatt;    /* Number of scatterings */
    double x1i_av;    /* Average X1 coordinate */
    double x2i_sq;    /* Square of X2 coordinate */
    double x3f_sq;    /* Square of X3 coordinate */
    double tau_abs;   /* Absorption optical depth */
    double tau_scatt; /* Scattering optical depth */
    double ne_0;      /* Reference electron density */
    double theta_e_0; /* Reference electron temperature */
    double b_0;       /* Reference magnetic field */
    double e_0;       /* Reference energy */
};

}; /* namespace harm */
