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
#include "cuda_grmonty/tetrads.hpp"

namespace harm {

HARMModel::HARMModel(int photon_n, double mass_unit) : photon_n_(photon_n), mass_unit_(mass_unit) {
    l_unit_ = consts::g_newt * consts::m_bh / (consts::cl * consts::cl);
    t_unit_ = l_unit_ / consts::cl;
    rho_unit_ = mass_unit_ / std::pow(l_unit_, 3);
    u_unit_ = rho_unit_ * consts::cl * consts::cl;
    b_unit_ = consts::cl * std::sqrt(4.0 * std::numbers::pi * rho_unit_);
    n_e_unit_ = rho_unit_ / (consts::mp + consts::me);
    max_tau_scatt_ = 6.0 * l_unit_ * rho_unit_ * 0.4;
}

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
    theta_e_unit_ = (two_temp_gamma - 1.) * (consts::mp / consts::me) / (1. + consts::tp_over_te);
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
    data_.k_rho = ndarray::NDArray<double>({header_.n[0], header_.n[1]});
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
        if (i % 1024 == 0) {
            spdlog::debug("{} / {}", i, n_cells);
        }
        std::getline(harm_file, line);
        std::istringstream iss(line);

        int x_1 = i / header_.n[1];
        int x_2 = i % header_.n[1];

        iss >> x[1];
        iss >> x[2];
        iss >> r;
        iss >> h;
        iss >> data_.k_rho[{x_1, x_2}];
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
            d_v * g_det * std::pow(data_.u[{x_1, x_2}].value() / data_.k_rho[{x_1, x_2}].value() * theta_e_unit_, 2.);
        v += d_v * g_det;

        if (x_1 <= 20) {
            d_mact += g_det * data_.k_rho[{x_1, x_2}].value() * u_con[1];
        }
        if (x_1 >= 20 && x_1 < 40) {
            l_adv += g_det * data_.u[{x_1, x_2}].value() * u_con[1] * u_con[0];
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
    init_weight_table();
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

            geometry_.det[{x_1, x_2}] = std::sqrt(std::abs(geometry_.cov[{x_1, x_2}].det()));
        }
    }

    spdlog::info("Initializing HARM model geometry done");
}

void HARMModel::init_weight_table() {
    spdlog::info("Initializing super photon weight table");

    static const double l_nu_min = std::log(consts::nu_min);
    static const double l_nu_max = std::log(consts::nu_max);
    static const double d_l_nu = (l_nu_max - l_nu_min) / consts::n_e_samp;

    double sum[consts::n_e_samp + 1];
    double nu[consts::n_e_samp + 1];

    for (int i = 0; i <= consts::n_e_samp; ++i) {
        sum[i] = 0.0;
        nu[i] = std::exp(i * d_l_nu + l_nu_min);
    }

    double s_fac = header_.dx[1] * header_.dx[2] * header_.dx[3] * l_unit_ * l_unit_ * l_unit_;

    for (int i = 0; i < static_cast<int>(header_.n[0]); ++i) {
        spdlog::debug("{} / {}", i, header_.n[0]);
        for (int j = 0; j < static_cast<int>(header_.n[1]); ++j) {
            harm::FluidZone fluid_zone = get_fluid_zone(i, j);

            if (fluid_zone.n_e == 0.0 || fluid_zone.theta_e < consts::theta_e_min) {
                continue;
            }

            double k2 = jnu_mixed::k2_eval(fluid_zone.theta_e, k2_);
            double fac = (consts::super_photon::jcst * fluid_zone.n_e * fluid_zone.b * fluid_zone.theta_e *
                          fluid_zone.theta_e / k2) *
                         s_fac * geometry_.det[{i, j}].value();
            for (int k = 0; k <= consts::n_e_samp; ++k) {
                double f = jnu_mixed::f_eval(fluid_zone.theta_e, fluid_zone.b, nu[k], f_);
                sum[k] += fac * f;
            }
        }
    }

    spdlog::info("Initializing super photon weight table done");
}

void HARMModel::gcon_func(double x[consts::n_dim], ndarray::NDArray<double> &&gcon) const {
    gcon = 0.0;

    BLCoord bl_coord = get_bl_coord(x);

    double sin_theta = std::fabs(std::sin(bl_coord.theta)) + consts::eps;
    double cos_theta = std::cos(bl_coord.theta);

    double irho2 = 1.0 / (bl_coord.r * bl_coord.r + header_.a * header_.a * cos_theta * cos_theta);

    double hfac =
        std::numbers::pi + (1.0 - header_.h_slope) * std::numbers::pi * std::cos(2.0 * std::numbers::pi * x[2]);

    gcon[{0, 0}] = -1.0 - 2.0 * bl_coord.r * irho2;
    gcon[{0, 1}] = 2.0 * irho2;

    gcon[{1, 0}] = gcon[{0, 1}].value();
    gcon[{1, 1}] = irho2 * (bl_coord.r * (bl_coord.r - 2.0) + header_.a * header_.a) / (bl_coord.r * bl_coord.r);
    gcon[{1, 3}] = header_.a * irho2 / bl_coord.r;

    gcon[{2, 2}] = irho2 / (hfac * hfac);

    gcon[{3, 1}] = gcon[{1, 3}].value();
    gcon[{3, 3}] = irho2 / (sin_theta * sin_theta);
}

void HARMModel::gcov_func(double x[consts::n_dim], ndarray::NDArray<double> &&gcov) const {
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

    gcov[{1, 0}] = gcov[{0, 1}].value();
    gcov[{1, 1}] = (1.0 + 2.0 * bl_coord.r / rho2) * rfac * rfac;
    gcov[{1, 3}] = (-header_.a * sin_theta_2 * (1.0 + 2.0 * bl_coord.r / rho2)) * rfac * pfac;

    gcov[{2, 2}] = rho2 * hfac * hfac;

    gcov[{3, 0}] = gcov[{0, 3}].value();
    gcov[{3, 1}] = gcov[{1, 3}].value();
    gcov[{3, 3}] =
        sin_theta_2 * (rho2 + header_.a * header_.a * sin_theta_2 * (1.0 + 2.0 * bl_coord.r / rho2)) * pfac * pfac;
}

struct FluidZone HARMModel::get_fluid_zone(int x_1, int x_2) const {
    struct FluidZone result;

    double u_cov[consts::n_dim];
    double b_cov[consts::n_dim];

    double v_con[consts::n_dim] = {
        0.0,
        data_.u_1[{x_1, x_2}].value(),
        data_.u_2[{x_1, x_2}].value(),
        data_.u_3[{x_1, x_2}].value(),
    };
    double b[consts::n_dim] = {
        0.0,
        data_.b_1[{x_1, x_2}].value(),
        data_.b_2[{x_1, x_2}].value(),
        data_.b_3[{x_1, x_2}].value(),
    };

    result.n_e = data_.k_rho[{x_1, x_2}].value() * n_e_unit_;
    result.theta_e = (data_.u[{x_1, x_2}].value() / result.n_e) * n_e_unit_ * theta_e_unit_;

    double v_dot_v = 0.0;
    for (int i = 1; i < consts::n_dim; ++i) {
        for (int j = 1; j < consts::n_dim; ++j) {
            v_dot_v += geometry_.cov[{x_1, x_2, i, j}].value() * v_con[i] * v_con[j];
        }
    }

    double v_fac = std::sqrt(-1.0 / geometry_.con[{x_1, x_2, 0, 0}].value() * (1.0 + std::abs(v_dot_v)));

    result.u_con[0] = -v_fac * geometry_.con[{x_1, x_2, 0, 0}].value();
    for (int i = 1; i < consts::n_dim; ++i) {
        result.u_con[i] = v_con[i] - v_fac * geometry_.con[{x_1, x_2, 0, i}].value();
    }

    tetrads::lower(result.u_con, geometry_.cov[{x_1, x_2}], u_cov);

    double u_dot_b = 0.0;
    for (int i = 1; i < consts::n_dim; ++i) {
        u_dot_b += u_cov[i] * b[i];
    }

    result.b_con[0] = u_dot_b;
    for (int i = 1; i < consts::n_dim; ++i) {
        result.b_con[i] = (b[i] + result.u_con[i] * u_dot_b) / result.u_con[0];
    }

    tetrads::lower(result.b_con, geometry_.cov[{x_1, x_2}], b_cov);

    result.b = std::sqrt(result.b_con[0] * b_cov[0] + result.b_con[1] * b_cov[1] + result.b_con[2] * b_cov[2] +
                         result.b_con[3] * b_cov[3]) *
               b_unit_;

    return result;
}

struct BLCoord HARMModel::get_bl_coord(double x[consts::n_dim]) const {
    return BLCoord{
        .r = std::exp(x[1]) + header_.r_0,
        .theta = std::numbers::pi * x[2] + ((1.0 - header_.h_slope) / 2.0) * std::sin(2.0 * std::numbers::pi * x[2]),
    };
}

std::array<double, 4> HARMModel::get_coord(int x_1, int x_2) const {
    return {
        header_.x_start[0],
        header_.x_start[1] + (x_1 + 0.5) * header_.dx[1],
        header_.x_start[2] + (x_2 + 0.5) * header_.dx[2],
        header_.x_start[3],
    };
}

}; /* namespace harm */
