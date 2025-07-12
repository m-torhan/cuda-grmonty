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
#include <string>

#include "cuda_grmonty/ndarray.hpp"
#include "spdlog/spdlog.h"

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/harm_model.hpp"
#include "cuda_grmonty/hotcross.hpp"
#include "cuda_grmonty/jnu_mixed.hpp"

namespace harm {

void HARMModel::read_file(std::string filepath) {
    spdlog::info("Reading file {}", filepath);

    /* open file */
    if (!std::filesystem::exists(filepath)) {
        spdlog::error("File does not exist {}", filepath);
        throw std::runtime_error("File does not exist " + filepath);
    }
    std::ifstream harm_file(filepath);
    if (!harm_file.is_open()) {
        spdlog::error("Cannot open file {}", filepath);
        throw std::runtime_error("Cannot open file " + filepath);
    }

    std::string line;

    /* read header */
    spdlog::debug("Reading file header");
    {
        std::getline(harm_file, line);
        std::istringstream iss(line);

        iss >> header_.t;
        iss >> header_.n[0];
        iss >> header_.n[1];
        header_.x_start[0] = 0.0;
        iss >> header_.x_start[1];
        iss >> header_.x_start[2];
        header_.x_start[3] = 0.0;
        header_.dx[0] = 1.0;
        iss >> header_.dx[1];
        iss >> header_.dx[2];
        header_.dx[3] = 2.0 * std::numbers::pi;
        header_.x_stop[0] = 1.0;
        header_.x_stop[1] = header_.x_start[1] + header_.n[0] * header_.dx[1];
        header_.x_stop[2] = header_.x_start[2] + header_.n[1] * header_.dx[2];
        header_.x_stop[3] = 2.0 * std::numbers::pi;
        iss >> header_.t_final;
        iss >> header_.n_step;
        iss >> header_.a;
        iss >> header_.gamma;
        iss >> header_.courant;
        iss >> header_.dt_dump;
        iss >> header_.dt_log;
        iss >> header_.dt_img;
        iss >> header_.dt_rdump;
        iss >> header_.cnt_dump;
        iss >> header_.cnt_img;
        iss >> header_.cnt_rdump;
        iss >> header_.dt;
        iss >> header_.lim;
        iss >> header_.failed;
        iss >> header_.r_in;
        iss >> header_.r_out;
        iss >> header_.h_slope;
        iss >> header_.r_0;
    }

    double two_temp_gamma =
        0.5 * ((1. + 2. / 3. * (consts::tp_over_te + 1.) / (consts::tp_over_te + 2.)) + header_.gamma);
    double thetae_unit =
        (two_temp_gamma - 1.) * (consts::photon_mass / consts::electron_mass) / (1. + consts::tp_over_te);
    double d_mact = 0.0;
    double l_adv = 0.0;
    double v = 0.0;
    double d_v = header_.dx[1] * header_.dx[2] * header_.dx[3];
    bias_norm_ = 0.0;

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
    data_.p = ndarray::NDArray<double>({header_.n[0], header_.n[1]});
    data_.u = ndarray::NDArray<double>({header_.n[0], header_.n[1]});
    data_.u_1 = ndarray::NDArray<double>({header_.n[0], header_.n[1]});
    data_.u_2 = ndarray::NDArray<double>({header_.n[0], header_.n[1]});
    data_.u_3 = ndarray::NDArray<double>({header_.n[0], header_.n[1]});
    data_.b_1 = ndarray::NDArray<double>({header_.n[0], header_.n[1]});
    data_.b_2 = ndarray::NDArray<double>({header_.n[0], header_.n[1]});
    data_.b_3 = ndarray::NDArray<double>({header_.n[0], header_.n[1]});

    /* read data */
    spdlog::debug("Reading file data");
    int n_cells = header_.n[0] * header_.n[1];

    for (int i = 0; i < n_cells; ++i) {
        spdlog::debug("{} / {}", i, n_cells);
        std::getline(harm_file, line);
        std::istringstream iss(line);

        int x_1 = i % header_.n[0];
        int x_2 = i / header_.n[0];

        iss >> x[1];
        iss >> x[2];
        iss >> r;
        iss >> h;
        iss >> data_.p[{x_1, x_2}];
        iss >> data_.u[{x_1, x_2}];
        iss >> data_.u_1[{x_1, x_2}];
        iss >> data_.u_2[{x_1, x_2}];
        iss >> data_.u_3[{x_1, x_2}];
        iss >> data_.b_1[{x_1, x_2}];
        iss >> data_.b_2[{x_1, x_2}];
        iss >> data_.b_3[{x_1, x_2}];
        iss >> div_b;
        iss >> u_con[0] >> u_con[1] >> u_con[2] >> u_con[3];
        iss >> u_cov[0] >> u_cov[1] >> u_cov[2] >> u_cov[3];
        iss >> b_con[0] >> b_con[1] >> b_con[2] >> b_con[3];
        iss >> b_cov[0] >> b_cov[1] >> b_cov[2] >> b_cov[3];
        iss >> vmin[0] >> vmax[0];
        iss >> vmin[1] >> vmax[1];
        iss >> g_det;

        bias_norm_ +=
            d_v * g_det *
            std::pow(static_cast<double>(data_.u[{x_1, x_2}]) / static_cast<double>(data_.p[{x_1, x_2}]) * thetae_unit,
                     2.);
        v += d_v * g_det;

        if (x_1 <= 20) {
            d_mact += g_det * static_cast<double>(data_.p[{x_1, x_2}]) * u_con[1];
        }
        if (x_1 >= 20 && x_1 < 40) {
            l_adv += g_det * static_cast<double>(data_.u[{x_1, x_2}]) * u_con[1] * u_con[0];
        }
    }

    harm_file.close();

    bias_norm_ /= v;
    d_mact *= header_.dx[3] * header_.dx[2];
    d_mact /= 21.;
    l_adv *= header_.dx[3] * header_.dx[2];
    l_adv /= 21.;

    spdlog::debug("d_mact={}", d_mact);
    spdlog::debug("l_adv={}", l_adv);

    rh_ = 1.0 + std::sqrt(1.0 - header_.a * header_.a);

    spdlog::info("Reading file done");
}

void HARMModel::init() {
    init_geometry();
    hotcross::init_table(hotcross_table_);
    jnu_mixed::init_emiss_tables(f_, k2_);
}

void HARMModel::init_geometry() {
    spdlog::info("Initializing HARM model geometry");

    geometry_.cov = ndarray::NDArray<double>({header_.n[0], header_.n[1], consts::n_dim, consts::n_dim});
    geometry_.con = ndarray::NDArray<double>({header_.n[0], header_.n[1], consts::n_dim, consts::n_dim});
    geometry_.det = ndarray::NDArray<double>({header_.n[0], header_.n[1]});

    for (int x_1 = 0; x_1 < static_cast<int>(header_.n[0]); ++x_1) {
        for (int x_2 = 0; x_2 < static_cast<int>(header_.n[1]); ++x_2) {
            std::array<double, consts::n_dim> x = get_coord(x_1, x_2);

            gcov_func(x.data(), geometry_.cov[{x_1, x_2}]);
            gcon_func(x.data(), geometry_.con[{x_1, x_2}]);

            geometry_.det[{x_1, x_2}] = geometry_.cov[{x_1, x_2}].det();
        }
    }

    spdlog::info("Initializing HARM model geometry done");
}

void HARMModel::gcon_func(double x[consts::n_dim], ndarray::NDArray<double> &&gcon) {
    gcon = 0.0;

    BLCoord bl_coord = get_bl_coord(x);

    double sin_theta = std::fabs(std::sin(bl_coord.theta)) + consts::eps;
    double cos_theta = std::cos(bl_coord.theta);

    double irho2 = 1.0 / (bl_coord.r * bl_coord.r + header_.a * header_.a * cos_theta * cos_theta);

    double hfac =
        std::numbers::pi + (1.0 - header_.h_slope) * std::numbers::pi * std::cos(2.0 * std::numbers::pi * x[2]);

    gcon[{0, 0}] = -1.0 - 2.0 * bl_coord.r * irho2;
    gcon[{0, 1}] = 2.0 * irho2;

    gcon[{1, 0}] = gcon[{0, 1}];
    gcon[{1, 1}] = irho2 * (bl_coord.r * (bl_coord.r - 2.0) + header_.a * header_.a) / (bl_coord.r * bl_coord.r);
    gcon[{1, 2}] = header_.a * irho2 / bl_coord.r;

    gcon[{2, 2}] = irho2 / (hfac * hfac);

    gcon[{3, 1}] = gcon[{1, 3}];
    gcon[{3, 3}] = irho2 / (sin_theta * sin_theta);
}

void HARMModel::gcov_func(double x[consts::n_dim], ndarray::NDArray<double> &&gcov) {
    gcov = 0.0;

    BLCoord bl_coord = get_bl_coord(x);

    double sin_theta = std::fabs(std::sin(bl_coord.theta)) + consts::eps;
    double cos_theta = std::cos(bl_coord.theta);

    double sin_theta_2 = sin_theta * sin_theta;
    double rho2 = bl_coord.r * bl_coord.r + header_.a * header_.a * cos_theta * cos_theta;

    double tfac = 1.0;
    double rfac = bl_coord.r - header_.r_0;
    double hfac =
        std::numbers::pi + (1.0 - header_.h_slope) * std::numbers::pi * std::cos(2.0 * std::numbers::pi * x[2]);
    double pfac = 1.0;

    gcov[{0, 0}] = (-1.0 + 2.0 * bl_coord.r / rho2) * tfac * tfac;
    gcov[{0, 1}] = (2.0 * bl_coord.r / rho2) * tfac * rfac;
    gcov[{0, 3}] = (-2.0 * header_.a * bl_coord.r * sin_theta_2 / rho2) * tfac * pfac;

    gcov[{1, 0}] = gcov[{0, 1}];
    gcov[{1, 1}] = (1.0 + 2.0 * bl_coord.r / rho2) * rfac * rfac;
    gcov[{1, 3}] = (-header_.a * sin_theta_2 * (1.0 + 2.0 * bl_coord.r / rho2)) * rfac * pfac;

    gcov[{2, 2}] = rho2 * hfac * hfac;

    gcov[{3, 0}] = gcov[{0, 3}];
    gcov[{3, 1}] = gcov[{1, 3}];
    gcov[{3, 3}] =
        sin_theta_2 * (rho2 + header_.a * header_.a * sin_theta_2 * (1.0 + 2.0 * bl_coord.r / rho2)) * pfac * pfac;
}

struct BLCoord HARMModel::get_bl_coord(double x[consts::n_dim]) {
    return BLCoord{
        .r = std::exp(x[1]) + header_.r_0,
        .theta = std::numbers::pi * x[2] + ((1.0 + header_.h_slope) / 2.0) * std::sin(2.0 * std::numbers::pi * x[2]),
    };
}

std::array<double, 4> HARMModel::get_coord(int x_1, int x_2) {
    return {
        header_.x_start[0],
        header_.x_start[1] + (x_1 * 0.5) * header_.dx[1],
        header_.x_start[2] + (x_2 * 0.5) * header_.dx[2],
        header_.x_start[3],
    };
}

}; /* namespace harm */
