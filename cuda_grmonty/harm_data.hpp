/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <string>

#include "cuda_grmonty/ndarray.hpp"

namespace harm {

struct Header {
    double t;          /* simulation time */
    unsigned int n[2]; /* number of grid points in x1 and x2 directions  */
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
    ndarray::NDArray<double> p;   /* rest-mass density */
    ndarray::NDArray<double> u;   /* internal eneergy density */
    ndarray::NDArray<double> u_1; /* covariant velocity components */
    ndarray::NDArray<double> u_2;
    ndarray::NDArray<double> u_3;
    ndarray::NDArray<double> b_1; /* contravariant magnetic field components */
    ndarray::NDArray<double> b_2;
    ndarray::NDArray<double> b_3;
    double bias_norm;
};

class HARMData {
public:
    /**
     * Reads HARM data from file
     *
     * @param filepath Path to HARM dump
     */
    void read_file(std::string filepath);

    const struct Header *get_header() { return &header; }

    const struct Data *get_data() { return &data; }

private:
    struct Header header;
    struct Data data;
};

}; /* namespace harm */
