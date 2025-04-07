/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <sstream>

#include "consts.hpp"

#include "harm_data.hpp"

using namespace harm;

void HARMData::read_file(std::string filepath) {
    std::cout << "Reading file: " << filepath << std::endl;

    /* open file */
    if (!std::filesystem::exists(filepath)) {
        throw std::runtime_error("File does not exist " + filepath);
    }
    std::ifstream harm_file(filepath);
    if (!harm_file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    std::string line;

    /* read header */
    {
        std::getline(harm_file, line);
        std::istringstream iss(line);

        iss >> header.t;
        iss >> header.n[0];
        iss >> header.n[1];
        header.x_start[0] = 0.0;
        iss >> header.x_start[1];
        iss >> header.x_start[2];
        header.x_start[3] = 0.0;
        header.dx[0] = 1.0;
        iss >> header.dx[1];
        iss >> header.dx[2];
        header.dx[3] = 2.0 * std::numbers::pi;
        header.x_stop[0] = 1.0;
        header.x_stop[1] = header.x_start[1] + header.n[0] * header.dx[1];
        header.x_stop[2] = header.x_start[2] + header.n[1] * header.dx[2];
        header.x_stop[3] = 2.0 * std::numbers::pi;
        iss >> header.t_final;
        iss >> header.n_step;
        iss >> header.a;
        iss >> header.gamma;
        iss >> header.courant;
        iss >> header.dt_dump;
        iss >> header.dt_log;
        iss >> header.dt_img;
        iss >> header.dt_rdump;
        iss >> header.cnt_dump;
        iss >> header.cnt_img;
        iss >> header.cnt_rdump;
        iss >> header.dt;
        iss >> header.lim;
        iss >> header.failed;
        iss >> header.r_in;
        iss >> header.r_out;
        iss >> header.h_slope;
        iss >> header.r_0;
    }

    double two_temp_gamma =
        0.5 * ((1. + 2. / 3. * (consts::tp_over_te + 1.) / (consts::tp_over_te + 2.)) + header.gamma);
    double thetae_unit =
        (two_temp_gamma - 1.) * (consts::photon_mass / consts::electron_mass) / (1. + consts::tp_over_te);
    double d_mact = 0.0;
    double l_adv = 0.0;
    double v = 0.0;
    double d_v = header.dx[1] * header.dx[2] * header.dx[3];
    data.bias_norm = 0.0;

    double x[4]; /* coordinates */
    double r;    /* radial coordinate */
    double h;    /* angular coordinate */
    double div_b;
    double vmin[2];
    double vmax[2];
    double g_det;

    double u_con[4]; /* contravariant 4-velocity components */
    double u_cov[4]; /* covariant 4-velocity components */
    double b_con[4]; /* contravariant 4-magnetic field components */
    double b_cov[4]; /* convariant 4-magnetic field components */

    /* prepare space for data */
    data.p.resize(header.n[0], std::vector<double>(header.n[1]));
    data.u.resize(header.n[0], std::vector<double>(header.n[1]));
    data.u_1.resize(header.n[0], std::vector<double>(header.n[1]));
    data.u_2.resize(header.n[0], std::vector<double>(header.n[1]));
    data.u_3.resize(header.n[0], std::vector<double>(header.n[1]));
    data.b_1.resize(header.n[0], std::vector<double>(header.n[1]));
    data.b_2.resize(header.n[0], std::vector<double>(header.n[1]));
    data.b_3.resize(header.n[0], std::vector<double>(header.n[1]));

    /* read data */
    int n_cells = header.n[0] * header.n[1];

    for (int i = 0; i < n_cells; ++i) {
        std::getline(harm_file, line);
        std::istringstream iss(line);

        int x_1 = i % header.n[0];
        int x_2 = i / header.n[0];

        iss >> x[1];
        iss >> x[2];
        iss >> r;
        iss >> h;
        iss >> data.p[x_1][x_2];
        iss >> data.u[x_1][x_2];
        iss >> data.u_1[x_1][x_2];
        iss >> data.u_2[x_1][x_2];
        iss >> data.u_3[x_1][x_2];
        iss >> data.b_1[x_1][x_2];
        iss >> data.b_2[x_1][x_2];
        iss >> data.b_3[x_1][x_2];
        iss >> div_b;
        iss >> u_con[0] >> u_con[1] >> u_con[2] >> u_con[3];
        iss >> u_cov[0] >> u_cov[1] >> u_cov[2] >> u_cov[3];
        iss >> b_con[0] >> b_con[1] >> b_con[2] >> b_con[3];
        iss >> b_cov[0] >> b_cov[1] >> b_cov[2] >> b_cov[3];
        iss >> vmin[0] >> vmax[0];
        iss >> vmin[1] >> vmax[1];
        iss >> g_det;

        data.bias_norm += d_v * g_det * std::pow(data.u[x_1][x_2] / data.p[x_1][x_2] * thetae_unit, 2.);
        v += d_v * g_det;

        if (x_1 <= 20) {
            d_mact += g_det * data.p[x_1][x_2] * u_con[1];
        }
        if (x_1 >= 20 && x_1 < 40) {
            l_adv += g_det * data.u[x_1][x_2] * u_con[1] * u_con[0];
        }
    }

    data.bias_norm /= v;
    d_mact *= header.dx[3] * header.dx[2];
    d_mact /= 21.;
    l_adv *= header.dx[3] * header.dx[2];
    l_adv /= 21.;

    harm_file.close();
}
