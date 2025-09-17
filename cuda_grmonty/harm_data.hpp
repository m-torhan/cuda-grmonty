/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/ndarray.hpp"

namespace harm {

struct Header {
    double t;          /* simulation time */
    int n[2];          /* number of grid points in x1 and x2 directions  */
    double x_start[4]; /* start coordinates of the grid */
    double x_stop[4];  /* stop coordinates of the grid */
    double dx[4];      /* grid spacing */
    double t_final;    /* final simulation time */
    int n_step;        /* number of simulation steps */
    double a;          /* black hole spin parameter (dimensionless Kerr parameter) */
    double gamma;      /* adiabatic index */
    double courant;    /* courant number for time-stepping */
    double dt_dump;    /* time interval between dumps */
    double dt_log;     /* time interval for log outputs */
    double dt_img;     /* time interval for image outputs */
    int dt_rdump;      /* time interval of restart dumps written */
    int cnt_dump;      /* counter for number of dumps written */
    int cnt_img;       /* counter for number of images written */
    int cnt_rdump;     /* counter for number of restart dumps written */
    double dt;         /* time step size */
    int lim;           /* slope limiter method used */
    int failed;        /* number of failed steps in simulation */
    double r_in;       /* inner radius of the simulation domain */
    double r_out;      /* outer radius of the simulation domain */
    double h_slope;    /* grid stretching parameters */
    double r_0;        /* reference radius */
};

struct Data {
    ndarray::NDArray<double, 2> k_rho; /* rest-mass density */
    ndarray::NDArray<double, 2> u;     /* internal eneergy density */
    ndarray::NDArray<double, 2> u_1;   /* covariant velocity components */
    ndarray::NDArray<double, 2> u_2;
    ndarray::NDArray<double, 2> u_3;
    ndarray::NDArray<double, 2> b_1; /* contravariant magnetic field components */
    ndarray::NDArray<double, 2> b_2;
    ndarray::NDArray<double, 2> b_3;
};

struct Units {
    double mass_unit;
    double l_unit;
    double t_unit;
    double rho_unit;
    double u_unit;
    double b_unit;
    double theta_e_unit;
    double n_e_unit;
};

struct Geometry {
    ndarray::NDArray<double, 4> cov;
    ndarray::NDArray<double, 4> con;
    ndarray::NDArray<double, 2> det;
};

struct BLCoord {
    double r;
    double theta;
};

struct FluidZone {
    double n_e;
    double theta_e;
    double b;
    double u_con[consts::n_dim];
    double b_con[consts::n_dim];
};

struct FluidParams {
    double n_e;
    double theta_e;
    double b;
    double u_con[consts::n_dim];
    double u_cov[consts::n_dim];
    double b_con[consts::n_dim];
    double b_cov[consts::n_dim];
};

struct Zone {
    int x_1;
    int x_2;
    int num_to_gen;
    double dn_max;
    bool quit_flag;
};

struct Spectrum {
    double dNdlE;
    double dEdlE;
    double nph;
    double nscatt;
    double X1iav;
    double X2isq;
    double X3fsq;
    double tau_abs;
    double tau_scatt;
    double ne0;
    double thetae0;
    double b0;
    double E0;
};

}; /* namespace harm */
