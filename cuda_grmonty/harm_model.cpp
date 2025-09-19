/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <semaphore>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>

#include "spdlog/spdlog.h"

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/harm_data.hpp"
#include "cuda_grmonty/harm_model.hpp"
#include "cuda_grmonty/hotcross.hpp"
#include "cuda_grmonty/jnu_mixed.hpp"
#include "cuda_grmonty/monty_rand.hpp"
#include "cuda_grmonty/ndarray.hpp"
#include "cuda_grmonty/photon.hpp"
#include "cuda_grmonty/photon_queue.hpp"
#include "cuda_grmonty/proba.hpp"
#include "cuda_grmonty/radiation.hpp"
#include "cuda_grmonty/tetrads.hpp"

#ifdef CUDA
#include "cuda_grmonty/super_photon.cuh"
#endif /* CUDA */

namespace harm {

/**
 * @brief Performs bilinear interpolation of a scalar field from a 2D array.
 *
 * @param var   2D scalar field array to interpolate from.
 * @param i     Radial grid index.
 * @param j     Polar grid index.
 * @param coeff Interpolation coefficients for each dimension.
 *
 * @return Interpolated scalar value at the given location.
 */
static double interp_scalar(const ndarray::NDArray<double, 2> &var, int i, int j, const double (&coeff)[consts::n_dim]);

/**
 * @brief Applies a Lorentz boost to transform a 4-vector.
 *
 * @param[in] v   Input 4-vector in the lab frame.
 * @param[in] u   4-velocity defining the boost (typically fluid frame).
 * @param[out] vp Boosted 4-vector in the new frame.
 */
static void boost(const double (&v)[consts::n_dim], const double (&u)[consts::n_dim], double (&vp)[consts::n_dim]);

HARMModel::HARMModel(int photon_n, double mass_unit) : photon_n_(photon_n) {
    units_.mass_unit = mass_unit;
    units_.l_unit = consts::g_newt * consts::m_bh / (consts::cl * consts::cl);
    units_.t_unit = units_.l_unit / consts::cl;
    units_.rho_unit = units_.mass_unit / std::pow(units_.l_unit, 3);
    units_.u_unit = units_.rho_unit * consts::cl * consts::cl;
    units_.b_unit = consts::cl * std::sqrt(4.0 * std::numbers::pi * units_.rho_unit);
    units_.n_e_unit = units_.rho_unit / (consts::mp + consts::me);
    max_tau_scatt_ = 6.0 * units_.l_unit * units_.rho_unit * 0.4;
    d_tau_k_ = 2.0 * std::numbers::pi * units_.l_unit / (consts::me * consts::cl * consts::cl / consts::hbar);
    for (auto &row : spectrum_) {
        for (auto &cell : row) {
            cell = {};
        }
    }
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
    units_.theta_e_unit = (two_temp_gamma - 1.) * (consts::mp / consts::me) / (1. + consts::tp_over_te);
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
    data_.k_rho = ndarray::NDArray<double, 2>({header_.n[0], header_.n[1]});
    data_.u = ndarray::NDArray<double, 2>({header_.n[0], header_.n[1]});
    data_.u_1 = ndarray::NDArray<double, 2>({header_.n[0], header_.n[1]});
    data_.u_2 = ndarray::NDArray<double, 2>({header_.n[0], header_.n[1]});
    data_.u_3 = ndarray::NDArray<double, 2>({header_.n[0], header_.n[1]});
    data_.b_1 = ndarray::NDArray<double, 2>({header_.n[0], header_.n[1]});
    data_.b_2 = ndarray::NDArray<double, 2>({header_.n[0], header_.n[1]});
    data_.b_3 = ndarray::NDArray<double, 2>({header_.n[0], header_.n[1]});

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
        iss >> data_.k_rho(x_1, x_2);
        iss >> data_.u(x_1, x_2);
        iss >> data_.u_1(x_1, x_2);
        iss >> data_.u_2(x_1, x_2);
        iss >> data_.u_3(x_1, x_2);
        iss >> data_.b_1(x_1, x_2);
        iss >> data_.b_2(x_1, x_2);
        iss >> data_.b_3(x_1, x_2);
        iss >> div_b;
        iss >> u_con[0] >> u_con[1] >> u_con[2] >> u_con[3];
        iss >> u_cov[0] >> u_cov[1] >> u_cov[2] >> u_cov[3];
        iss >> b_con[0] >> b_con[1] >> b_con[2] >> b_con[3];
        iss >> b_cov[0] >> b_cov[1] >> b_cov[2] >> b_cov[3];
        iss >> vmin[0] >> vmax[0];
        iss >> vmin[1] >> vmax[1];
        iss >> g_det;

        bias_norm_ += d_v * g_det * std::pow(data_.u(x_1, x_2) / data_.k_rho(x_1, x_2) * units_.theta_e_unit, 2.);
        v += d_v * g_det;

        if (x_1 <= 20) {
            d_mact += g_det * data_.k_rho(x_1, x_2) * u_con[1];
        }
        if (x_1 >= 20 && x_1 < 40) {
            l_adv += g_det * data_.u(x_1, x_2) * u_con[1] * u_con[0];
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
    x1_min_ = std::log(rh_);

    spdlog::info("Reading file done");
}

void HARMModel::init() {
    init_geometry();
    hotcross::init_table(hotcross_table_);
    jnu_mixed::init_emiss_tables(f_, k2_);
    init_weight_table();
    init_nint_table();
}

void HARMModel::init_geometry() {
    spdlog::info("Initializing HARM model geometry");

    geometry_.cov = ndarray::NDArray<double, 4>({header_.n[0], header_.n[1], consts::n_dim, consts::n_dim});
    geometry_.con = ndarray::NDArray<double, 4>({header_.n[0], header_.n[1], consts::n_dim, consts::n_dim});
    geometry_.det = ndarray::NDArray<double, 2>({header_.n[0], header_.n[1]});

    for (int x_1 = 0; x_1 < static_cast<int>(header_.n[0]); ++x_1) {
        spdlog::debug("{} / {}", x_1, header_.n[0]);
        for (int x_2 = 0; x_2 < static_cast<int>(header_.n[1]); ++x_2) {
            double x[consts::n_dim];
            get_coord(x_1, x_2, x);

            auto g_cov = geometry_.cov(x_1, x_2);
            auto g_con = geometry_.con(x_1, x_2);

            gcov_func(x, g_cov);
            gcon_func(x, g_con);

            geometry_.det(x_1, x_2) = std::sqrt(std::abs(geometry_.cov(x_1, x_2).det()));
        }
    }

    spdlog::info("Initializing HARM model geometry done");
}

void HARMModel::init_weight_table() {
    spdlog::info("Initializing super photon weight table");

    double sum[consts::n_e_samp + 1];
    double nu[consts::n_e_samp + 1];

    for (int i = 0; i <= consts::n_e_samp; ++i) {
        sum[i] = 0.0;
        nu[i] = std::exp(i * consts::d_l_nu + consts::l_nu_min);
    }

    double s_fac = header_.dx[1] * header_.dx[2] * header_.dx[3] * units_.l_unit * units_.l_unit * units_.l_unit;

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
                         s_fac * geometry_.det(i, j);
            for (int k = 0; k <= consts::n_e_samp; ++k) {
                double f = jnu_mixed::f_eval(fluid_zone.theta_e, fluid_zone.b, nu[k], f_);
                sum[k] += fac * f;
            }
        }
    }

    for (int i = 0; i <= consts::n_e_samp; ++i) {
        weight_[i] = std::log(sum[i] / (consts::hpl * photon_n_));
    }

    spdlog::info("Initializing super photon weight table done");
}

void HARMModel::init_nint_table() {
    spdlog::info("Initializing nint table");

    for (int i = 0; i <= consts::nint; ++i) {
        if (i % 1024 == 0) {
            spdlog::debug("{} / {}", i, consts::nint);
        }

        double nint = 0.0;
        double dndlnu_max = 0.0;
        double b_mag = std::exp(i * consts::d_l_b + consts::l_b_min);

        for (int j = 0; j < consts::n_e_samp; ++j) {
            double dn = jnu_mixed::f_eval(1.0, b_mag, std::exp(j * consts::d_l_nu + consts::l_nu_min), f_) /
                        (std::exp(weight_[j]) + 1.0e-100);

            if (dn > dndlnu_max) {
                dndlnu_max = dn;
            }
            nint += consts::d_l_nu * dn;
        }
        nint *= header_.dx[1] * header_.dx[2] * header_.dx[3] * units_.l_unit * units_.l_unit * units_.l_unit *
                std::numbers::sqrt2 * consts::ee * consts::ee * consts::ee /
                (27.0 * consts::me * consts::cl * consts::cl) * (1.0 / consts::hpl);

        nint_[i] = std::log(nint);
        dndlnu_max_[i] = std::log(dndlnu_max);
    }

    spdlog::info("Initializing nint table done");
}

void HARMModel::run_simulation() {
    auto start = std::chrono::system_clock::now();

    spdlog::info("Starting main loop");

#ifdef CUDA
    cuda_super_photon::alloc_memory(header_, data_, units_, hotcross_table_, f_, k2_);

    photon::PhotonQueue photon_queue(consts::cuda::threads_per_grid);
    std::binary_semaphore done_sem{0};

    std::thread make_super_photon_thread(
        &HARMModel::make_super_photon_async, this, std::ref(photon_queue), std::ref(done_sem));

    cuda_super_photon::track_super_photons(
        bias_norm_, max_tau_scatt_, photon_queue, done_sem, spectrum_, n_super_photon_recorded_, n_super_photon_scatt_);

    make_super_photon_thread.join();

    cuda_super_photon::free_memory();
#else  /* CUDA */
    int n_rate = 0;
    auto start_iter = start;

    while (true) {
        auto [photon, quit] = make_super_photon();

        if (quit) {
            break;
        }

        track_super_photon(photon);

        ++n_super_photon_created_;
        ++n_rate;
        std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start_iter;
        if (elapsed_seconds.count() > 1.0) {
            double rate = n_rate / elapsed_seconds.count();
            spdlog::info("Rate {:.2f} ph/s, zone ({}, {})", rate, zone_x_1_, zone_x_2_);
            n_rate = 0;
            start_iter = std::chrono::system_clock::now();
        }
    }
#endif /* CUDA */

    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = stop - start;
    spdlog::info("Final rate {:.2f} ph/s", n_super_photon_created_ / elapsed_seconds.count());
    spdlog::info("Super photons:");
    spdlog::info("\tcreated: {}", n_super_photon_created_);
    spdlog::info("\tscattered: {}", n_super_photon_scatt_);
    spdlog::info("\trecorded: {}", n_super_photon_recorded_);
}

void HARMModel::report_spectrum(std::string filepath) {
    const double dx2 = (header_.x_stop[2] - header_.x_start[2]) / (2 * consts::n_th_bins);

    spdlog::info("Writing spectrum to file {}", filepath);

    if (std::filesystem::exists(filepath)) {
        spdlog::warn("File {} already exists, overwriting", filepath);
    }
    std::ofstream spectrum_file(filepath);
    if (!spectrum_file.is_open()) {
        spdlog::error("Cannot open file {}", filepath);
        return;
    }

    double max_tau_scatt = 0.0;
    double l = 0.0;

    for (int i = 0; i < consts::n_e_bins; ++i) {
        spectrum_file << std::format("{:10.5g} ",
                                     (i * consts::spectrum::d_l_e + consts::spectrum::l_e_0) / std::numbers::ln10);

        for (int j = 0; j < consts::n_th_bins; ++j) {
            double d_omega = 2.0 * d_omega_func(j * dx2, (j + 1) * dx2);

            double nu_lnu = (consts::me * consts::cl * consts::cl) * (4.0 * std::numbers::pi / d_omega) *
                            (1.0 / consts::spectrum::d_l_e);

            nu_lnu *= spectrum_[j][i].de_dle;
            nu_lnu /= consts::l_sun;

            double tau_scatt = spectrum_[j][i].tau_scatt / (spectrum_[j][i].dn_dle + consts::eps);

            spectrum_file << std::format("{:10.5g} ", nu_lnu);
            spectrum_file << std::format("{:10.5g} ", spectrum_[j][i].tau_abs / (spectrum_[j][i].dn_dle + consts::eps));
            spectrum_file << std::format("{:10.5g} ", tau_scatt);
            spectrum_file << std::format("{:10.5g} ", spectrum_[j][i].x1i_av / (spectrum_[j][i].dn_dle + consts::eps));
            spectrum_file << std::format(
                "{:10.5g} ", std::sqrt(std::abs(spectrum_[j][i].x2i_sq / (spectrum_[j][i].dn_dle + consts::eps))));
            spectrum_file << std::format(
                "{:10.5g} ", std::sqrt(std::abs(spectrum_[j][i].x3f_sq / (spectrum_[j][i].dn_dle + consts::eps))));

            if (tau_scatt > max_tau_scatt) {
                max_tau_scatt = tau_scatt;
            }

            l += nu_lnu * d_omega * consts::spectrum::d_l_e;
        }
        spectrum_file << std::endl;
    }

    spectrum_file.close();

    spdlog::info("Writing spectrum done");
    spdlog::info("\tlumosity: {}", l);
    spdlog::info("\tmax_tau_scatt: {}", max_tau_scatt);
}

void HARMModel::gcon_func(const double (&x)[consts::n_dim], ndarray::NDArray<double, 2> &g_con) const {
    g_con = 0.0;

    BLCoord bl_coord = get_bl_coord(x);

    double sin_theta = std::fabs(std::sin(bl_coord.theta)) + consts::eps;
    double cos_theta = std::cos(bl_coord.theta);

    double irho2 = 1.0 / (bl_coord.r * bl_coord.r + header_.a * header_.a * cos_theta * cos_theta);

    double hfac =
        std::numbers::pi + (1.0 - header_.h_slope) * std::numbers::pi * std::cos(2.0 * std::numbers::pi * x[2]);

    g_con(0, 0) = -1.0 - 2.0 * bl_coord.r * irho2;
    g_con(0, 1) = 2.0 * irho2;

    g_con(1, 0) = g_con(0, 1);
    g_con(1, 1) = irho2 * (bl_coord.r * (bl_coord.r - 2.0) + header_.a * header_.a) / (bl_coord.r * bl_coord.r);
    g_con(1, 3) = header_.a * irho2 / bl_coord.r;

    g_con(2, 2) = irho2 / (hfac * hfac);

    g_con(3, 1) = g_con(1, 3);
    g_con(3, 3) = irho2 / (sin_theta * sin_theta);
}

void HARMModel::gcov_func(const double (&x)[consts::n_dim], ndarray::NDArray<double, 2> &g_cov) const {
    g_cov = 0.0;

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

    g_cov(0, 0) = (-1.0 + 2.0 * bl_coord.r / rho2) * tfac * tfac;
    g_cov(0, 1) = (2.0 * bl_coord.r / rho2) * tfac * rfac;
    g_cov(0, 3) = (-2.0 * header_.a * bl_coord.r * sin_theta_2 / rho2) * tfac * pfac;

    g_cov(1, 0) = g_cov(0, 1);
    g_cov(1, 1) = (1.0 + 2.0 * bl_coord.r / rho2) * rfac * rfac;
    g_cov(1, 3) = (-header_.a * sin_theta_2 * (1.0 + 2.0 * bl_coord.r / rho2)) * rfac * pfac;

    g_cov(2, 2) = rho2 * hfac * hfac;

    g_cov(3, 0) = g_cov(0, 3);
    g_cov(3, 1) = g_cov(1, 3);
    g_cov(3, 3) =
        sin_theta_2 * (rho2 + header_.a * header_.a * sin_theta_2 * (1.0 + 2.0 * bl_coord.r / rho2)) * pfac * pfac;
}

double HARMModel::d_omega_func(double x2i, double x2f) const {
    return 2.0 * std::numbers::pi *
           (-std::cos(std::numbers::pi * x2f + 0.5 * (1.0 - header_.h_slope) * std::sin(2 * std::numbers::pi * x2f)) +
            std::cos(std::numbers::pi * x2i + 0.5 * (1.0 - header_.h_slope) * std::sin(2 * std::numbers::pi * x2i)));
}

struct FluidZone HARMModel::get_fluid_zone(int x_1, int x_2) const {
    struct FluidZone result;

    double u_cov[consts::n_dim];
    double b_cov[consts::n_dim];

    double v_con[consts::n_dim] = {
        0.0,
        data_.u_1(x_1, x_2),
        data_.u_2(x_1, x_2),
        data_.u_3(x_1, x_2),
    };
    double b[consts::n_dim] = {
        0.0,
        data_.b_1(x_1, x_2),
        data_.b_2(x_1, x_2),
        data_.b_3(x_1, x_2),
    };

    result.n_e = data_.k_rho(x_1, x_2) * units_.n_e_unit;
    result.theta_e = (data_.u(x_1, x_2) / result.n_e) * units_.n_e_unit * units_.theta_e_unit;

    double v_dot_v = 0.0;
    for (int i = 1; i < consts::n_dim; ++i) {
        for (int j = 1; j < consts::n_dim; ++j) {
            v_dot_v += geometry_.cov(x_1, x_2, i, j) * v_con[i] * v_con[j];
        }
    }

    double v_fac = std::sqrt(-1.0 / geometry_.con(x_1, x_2, 0, 0) * (1.0 + std::abs(v_dot_v)));

    result.u_con[0] = -v_fac * geometry_.con(x_1, x_2, 0, 0);
    for (int i = 1; i < consts::n_dim; ++i) {
        result.u_con[i] = v_con[i] - v_fac * geometry_.con(x_1, x_2, 0, i);
    }

    tetrads::lower(result.u_con, geometry_.cov(x_1, x_2), u_cov);

    double u_dot_b = 0.0;
    for (int i = 1; i < consts::n_dim; ++i) {
        u_dot_b += u_cov[i] * b[i];
    }

    result.b_con[0] = u_dot_b;
    for (int i = 1; i < consts::n_dim; ++i) {
        result.b_con[i] = (b[i] + result.u_con[i] * u_dot_b) / result.u_con[0];
    }

    tetrads::lower(result.b_con, geometry_.cov(x_1, x_2), b_cov);

    result.b = std::sqrt(result.b_con[0] * b_cov[0] + result.b_con[1] * b_cov[1] + result.b_con[2] * b_cov[2] +
                         result.b_con[3] * b_cov[3]) *
               units_.b_unit;

    return result;
}

struct FluidParams HARMModel::get_fluid_params(const double (&x)[consts::n_dim],
                                               const ndarray::NDArray<double, 2> &g_cov) const {
    struct FluidParams fluid_params;

    if (x[1] < header_.x_start[1] || x[1] > header_.x_stop[1] || x[2] < header_.x_start[2] ||
        x[2] > header_.x_stop[2]) {
        fluid_params.n_e = 0.0;
        return fluid_params;
    }

    auto [i, j, del_i, del_j] = x_to_ij(x);

    double coeff[consts::n_dim] = {
        (1.0 - del_i) * (1.0 - del_j),
        (1.0 - del_i) * del_j,
        del_i * (1.0 - del_j),
        del_i * del_j,
    };

    double rho = interp_scalar(data_.k_rho, i, j, coeff);
    double uu = interp_scalar(data_.u, i, j, coeff);

    fluid_params.n_e = rho * units_.n_e_unit;
    fluid_params.theta_e = uu / rho * units_.theta_e_unit;

    double bp[consts::n_dim] = {
        0.0,
        interp_scalar(data_.b_1, i, j, coeff),
        interp_scalar(data_.b_2, i, j, coeff),
        interp_scalar(data_.b_3, i, j, coeff),
    };

    double v_con[consts::n_dim] = {
        0.0,
        interp_scalar(data_.u_1, i, j, coeff),
        interp_scalar(data_.u_2, i, j, coeff),
        interp_scalar(data_.u_3, i, j, coeff),
    };

    ndarray::NDArray<double, 2> g_con({consts::n_dim, consts::n_dim});

    gcon_func(x, g_con);

    double v_dot_v = 0.0;

    for (int i = 1; i < consts::n_dim; ++i) {
        for (int j = 1; j < consts::n_dim; ++j) {
            v_dot_v += g_cov(i, j) * v_con[i] * v_con[j];
        }
    }

    double v_fac = std::sqrt(-1.0 / g_con(0, 0) * (1.0 + std::abs(v_dot_v)));

    fluid_params.u_con[0] = -v_fac * g_con(0, 0);

    for (int i = 1; i < consts::n_dim; ++i) {
        fluid_params.u_con[i] = v_con[i] - v_fac * g_con(0, i);
    }
    tetrads::lower(fluid_params.u_con, g_cov, fluid_params.u_cov);

    double u_dot_bp = 0.0;
    for (int i = 1; i < consts::n_dim; ++i) {
        u_dot_bp += fluid_params.u_cov[i] * bp[i];
    }
    fluid_params.b_con[0] = u_dot_bp;
    for (int i = 1; i < consts::n_dim; ++i) {
        fluid_params.b_con[i] = (bp[i] + fluid_params.u_con[i] * u_dot_bp) / fluid_params.u_con[0];
    }
    tetrads::lower(fluid_params.b_con, g_cov, fluid_params.b_cov);

    fluid_params.b =
        std::sqrt(fluid_params.b_con[0] * fluid_params.b_cov[0] + fluid_params.b_con[1] * fluid_params.b_cov[1] +
                  fluid_params.b_con[2] * fluid_params.b_cov[2] + fluid_params.b_con[3] * fluid_params.b_cov[3]) *
        units_.b_unit;

    return fluid_params;
}

struct Zone HARMModel::get_zone() {
    struct Zone zone = {.quit_flag = true};
    int num_to_gen;

    ++zone_x_2_;

    if (zone_x_2_ >= static_cast<int>(header_.n[1])) {
        zone_x_2_ = 0;
        ++zone_x_1_;
        if (zone_x_1_ >= static_cast<int>(header_.n[0])) {
            zone.num_to_gen = 1;
            zone.x_1 = static_cast<int>(header_.n[0]);
            return zone;
        }
    }

    auto [d_num_to_gen, dn_max] = init_zone(zone_x_1_, zone_x_2_);

    zone.dn_max = dn_max;

    if (std::fmod(d_num_to_gen, 1.0) > monty_rand::rand()) {
        num_to_gen = static_cast<int>(d_num_to_gen) + 1;
    } else {
        num_to_gen = static_cast<int>(d_num_to_gen);
    }

    zone.x_1 = zone_x_1_;
    zone.x_2 = zone_x_2_;
    zone.num_to_gen = num_to_gen;

    return zone;
}

struct photon::Photon HARMModel::sample_zone_photon(struct Zone &zone) {
    static double e_con[consts::n_dim][consts::n_dim];
    static double e_cov[consts::n_dim][consts::n_dim];

    photon::Photon photon;

    get_coord(zone.x_1, zone.x_2, photon.x);

    auto fluid_zone = get_fluid_zone(zone.x_1, zone.x_2);

    double nu;
    double weight;

    do {
        nu = std::exp(monty_rand::rand() * consts::n_l_n + consts::l_nu_min);
        weight = linear_interp_weight(nu);
    } while (monty_rand::rand() >
             (jnu_mixed::f_eval(fluid_zone.theta_e, fluid_zone.b, nu, f_) / (weight + 1.0e-100)) / zone.dn_max);

    photon.w = weight;
    double j_max = jnu_mixed::synch(nu, fluid_zone.n_e, fluid_zone.theta_e, fluid_zone.b, std::numbers::pi / 2.0, k2_);

    double cos_th;
    double th;
    do {
        cos_th = 2.0 * monty_rand::rand() - 1.0;
        th = std::acos(cos_th);
    } while (monty_rand::rand() >
             (jnu_mixed::synch(nu, fluid_zone.n_e, fluid_zone.theta_e, fluid_zone.b, th, k2_) / j_max));

    double sin_th = std::sqrt(1.0 - cos_th * cos_th);
    double phi = 2.0 * std::numbers::pi * monty_rand::rand();
    double cos_phi = std::cos(phi);
    double sin_phi = std::sin(phi);

    double e = nu * consts::hpl / (consts::me * consts::cl * consts::cl);
    double k_tetrad[consts::n_dim] = {
        e,
        e * cos_th,
        e * sin_th * cos_phi,
        e * sin_th * sin_phi,
    };

    double b_hat[consts::n_dim];

    if (zone.quit_flag) {
        if (fluid_zone.b > 0.0) {
            for (int i = 0; i < consts::n_dim; ++i) {
                b_hat[i] = fluid_zone.b_con[i] * units_.b_unit / fluid_zone.b;
            }
        } else {
            for (int i = 1; i < consts::n_dim; ++i) {
                b_hat[i] = 0.0;
            }
            b_hat[0] = 1.0;
        }
        tetrads::make_tetrad(fluid_zone.u_con, b_hat, geometry_.cov(zone.x_1, zone.x_2), e_con, e_cov);
        zone.quit_flag = 0;
    }

    tetrads::tetrad_to_coordinate(e_con, k_tetrad, photon.k);

    k_tetrad[0] *= -1.0;

    double tmp_k[consts::n_dim];
    tetrads::tetrad_to_coordinate(e_cov, k_tetrad, tmp_k);

    photon.e = -tmp_k[0];
    photon.e_0 = -tmp_k[0];
    photon.e_0_s = -tmp_k[0];
    photon.l = tmp_k[3];
    photon.tau_scatt = 0.;
    photon.tau_abs = 0.;
    photon.x1i = photon.x[1];
    photon.x2i = photon.x[2];
    photon.n_scatt = 0;
    photon.n_e_0 = fluid_zone.n_e;
    photon.b_0 = fluid_zone.b;
    photon.theta_e_0 = fluid_zone.theta_e;

    return photon;
}

double HARMModel::linear_interp_weight(double nu) {
    double l_nu = std::log(nu);

    double d_i = (l_nu - consts::l_nu_min) / consts::d_l_nu;
    int i = static_cast<int>(d_i);
    d_i -= i;

    return std::exp((1.0 - d_i) * weight_[i] + d_i * weight_[i + 1]);
}

std::tuple<struct photon::Photon, bool> HARMModel::make_super_photon() {
    static Zone zone{.num_to_gen = -1};

    while (zone.num_to_gen <= 0) {
        zone = get_zone();
    }

    --zone.num_to_gen;

    bool quit = zone.x_1 == static_cast<int>(header_.n[0]);

    photon::Photon photon;
    if (!quit) {
        photon = sample_zone_photon(zone);
    }

    return {photon, quit};
}

void HARMModel::make_super_photon_async(photon::PhotonQueue &photon_queue, std::binary_semaphore &done_sem) {
    auto start_iter = std::chrono::system_clock::now();
    int n_rate = 0;

    while (true) {
        auto [photon, quit] = make_super_photon();

        photon_queue.enqueue(photon);

        ++n_super_photon_created_;
        ++n_rate;

        std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start_iter;
        if (elapsed_seconds.count() > 1.0) {
            double rate = n_rate / elapsed_seconds.count();
            spdlog::info("Rate {:.2f} ph/s, zone ({} {})", rate, zone_x_1_, zone_x_2_);
            n_rate = 0;
            start_iter = std::chrono::system_clock::now();
        }

        if (quit) {
            break;
        }
    }

    done_sem.release();
}

void HARMModel::track_super_photon(struct photon::Photon &photon) {
    if (std::isnan(photon.x[0]) || std::isnan(photon.x[1]) || std::isnan(photon.x[2]) || std::isnan(photon.x[3]) ||
        std::isnan(photon.k[0]) || std::isnan(photon.k[1]) || std::isnan(photon.k[2]) || std::isnan(photon.k[3]) ||
        photon.w == 0.0) {
        spdlog::error("Invalid photon provided");
        return;
    }

    ndarray::NDArray<double, 2> g_cov({consts::n_dim, consts::n_dim});

    gcov_func(photon.x, g_cov);
    auto fluid_params = get_fluid_params(photon.x, g_cov);

    double theta =
        radiation::bk_angle(photon.x, photon.k, fluid_params.u_cov, fluid_params.b_cov, fluid_params.b, units_.b_unit);
    double nu = radiation::fluid_nu(photon.x, photon.k, fluid_params.u_cov);
    double alpha_scatti = radiation::alpha_inv_scatt(nu, fluid_params.theta_e, fluid_params.n_e, hotcross_table_);
    double alpha_absi =
        radiation::alpha_inv_abs(nu, fluid_params.theta_e, fluid_params.n_e, fluid_params.b, theta, k2_);
    double bi = bias_func(fluid_params.theta_e, photon.w);

    init_dkdlam(photon.x, photon.k, photon.dkdlam);

    int n_step = 0;

    while (!stop_criterion(photon)) {
        struct photon::Photon photon_2;

        std::copy(std::begin(photon.x), std::end(photon.x), std::begin(photon_2.x));
        std::copy(std::begin(photon.k), std::end(photon.k), std::begin(photon_2.k));
        std::copy(std::begin(photon.dkdlam), std::end(photon.dkdlam), std::begin(photon_2.dkdlam));
        photon_2.e_0_s = photon.e_0_s;

        double dl = step_size(photon.x, photon.k);

        /* step the geodesic */
        push_photon(photon, dl, 0);

        if (stop_criterion(photon)) {
            break;
        }

        /* allow photon to interact with matter */
        if (alpha_absi > 0.0 || alpha_scatti > 0.0 || fluid_params.n_e > 0.0) {
            gcov_func(photon.x, g_cov);
            fluid_params = get_fluid_params(photon.x, g_cov);

            bool bound_flag = fluid_params.n_e == 0.0;

            if (!bound_flag) {
                theta = radiation::bk_angle(
                    photon.x, photon.k, fluid_params.u_cov, fluid_params.b_cov, fluid_params.b, units_.b_unit);
                nu = radiation::fluid_nu(photon.x, photon.k, fluid_params.u_cov);

                if (std::isnan(nu)) {
                    spdlog::error("nu is nan {} {} {} {}", photon.x[0], photon.x[1], photon.x[2], photon.x[3]);
                }
            }

            double d_tau_scatt;
            double d_tau_abs;
            double bias;

            if (bound_flag || nu < 0.0) {
                d_tau_scatt = 0.5 * alpha_scatti * d_tau_k_ * dl;
                d_tau_abs = 0.5 * alpha_absi * d_tau_k_ * dl;
                alpha_scatti = 0.0;
                alpha_absi = 0.0;
                bias = 0.0;
                bi = 0.0;
            } else {
                double alpha_scattf =
                    radiation::alpha_inv_scatt(nu, fluid_params.theta_e, fluid_params.n_e, hotcross_table_);
                d_tau_scatt = 0.5 * (alpha_scatti + alpha_scattf) * d_tau_k_ * dl;
                alpha_scatti = alpha_scattf;

                double alpha_absf =
                    radiation::alpha_inv_abs(nu, fluid_params.theta_e, fluid_params.n_e, fluid_params.b, theta, k2_);
                d_tau_abs = 0.5 * (alpha_absi + alpha_absf) * d_tau_k_ * dl;
                alpha_absi = alpha_absf;

                double bf = bias_func(fluid_params.theta_e, photon.w);
                bias = 0.5 * (bi + bf);
                bi = bf;
            }

            double x1 = -std::log(monty_rand::rand());

            struct photon::Photon photon_p;
            photon_p.w = photon.w / bias;

            if (bias * d_tau_scatt > x1 && photon_p.w > consts::weight_min) {
                double frac = x1 / (bias * d_tau_scatt);

                /* apply absorption until scattering event */
                d_tau_abs *= frac;

                if (d_tau_abs > 100) {
                    /* this photon has been absorbed before scattering */
                    return;
                }

                d_tau_scatt *= frac;
                double d_tau = d_tau_abs + d_tau_scatt;
                if (d_tau_abs < 1.0e-3) {
                    photon.w *= (1.0 - d_tau / 24.0 * (24.0 - d_tau * (12.0 - d_tau * (4.0 - d_tau))));
                } else {
                    photon.w *= std::exp(-d_tau);
                }

                /* interpolate position and wave vector to scattering event */
                push_photon(photon_2, dl * frac, 0);

                std::copy(std::begin(photon_2.x), std::end(photon_2.x), std::begin(photon.x));
                std::copy(std::begin(photon_2.k), std::end(photon_2.k), std::begin(photon.k));
                std::copy(std::begin(photon_2.dkdlam), std::end(photon_2.dkdlam), std::begin(photon.dkdlam));
                photon.e_0_s = photon_2.e_0_s;

                gcov_func(photon.x, g_cov);
                fluid_params = get_fluid_params(photon.x, g_cov);

                if (fluid_params.n_e > 0.0) {
                    scatter_super_photon(photon, photon_p, fluid_params, g_cov, units_.b_unit);

                    if (photon.w < 1.0e-100) {
                        /* must have been a problem popping k back onto light cone */
                        return;
                    }

                    track_super_photon(photon_p);
                }

                theta = radiation::bk_angle(
                    photon.x, photon.k, fluid_params.u_cov, fluid_params.b_cov, fluid_params.b, units_.b_unit);
                nu = radiation::fluid_nu(photon.x, photon.k, fluid_params.u_cov);

                if (nu < 0.0) {
                    alpha_scatti = 0.0;
                    alpha_absi = 0.0;
                } else {
                    alpha_scatti =
                        radiation::alpha_inv_scatt(nu, fluid_params.theta_e, fluid_params.n_e, hotcross_table_);
                    alpha_absi = radiation::alpha_inv_abs(
                        nu, fluid_params.theta_e, fluid_params.n_e, fluid_params.b, theta, k2_);
                }
                bi = bias_func(fluid_params.theta_e, photon.w);
            } else {
                if (d_tau_abs > 100) {
                    /* this photon has been absorbed */
                    return;
                }

                double d_tau = d_tau_abs + d_tau_scatt;
                if (d_tau < 1.0e-3) {
                    photon.w *= (1. - d_tau / 24. * (24. - d_tau * (12. - d_tau * (4. - d_tau))));
                } else {
                    photon.w *= std::exp(-d_tau);
                }
            }

            photon.tau_abs += d_tau_abs;
            photon.tau_scatt += d_tau_scatt;
        }

        ++n_step;

        if (n_step > consts::max_n_step) {
            spdlog::warn("max step reached in super photon tracking");
            break;
        }
    }

    if (record_criterion(photon) && n_step <= consts::max_n_step) {
        record_super_photon(photon, n_step);
    }
}

void HARMModel::scatter_super_photon(struct photon::Photon &photon,
                                     struct photon::Photon &photon_p,
                                     const struct FluidParams &fluid_params,
                                     const ndarray::NDArray<double, 2> &g_cov,
                                     double b_unit) const {
    if (photon.k[0] > 1.0e5 || photon.k[0] < 0.0 || std::isnan(photon.k[0]) || std::isnan(photon.k[1]) ||
        std::isnan(photon.k[3])) {
        photon.k[0] = std::abs(photon.k[0]);
        photon.w = 0.0;
        return;
    }

    double b_hat_con[consts::n_dim];

    if (fluid_params.b > 0.0) {
        for (int i = 0; i < consts::n_dim; ++i) {
            b_hat_con[i] = fluid_params.b_con[i] / (fluid_params.b / b_unit);
        }
    } else {
        for (int i = 0; i < consts::n_dim; ++i) {
            b_hat_con[i] = 0.0;
        }
        b_hat_con[1] = 1.0;
    }

    double e_con[consts::n_dim][consts::n_dim];
    double e_cov[consts::n_dim][consts::n_dim];

    /* local tetrad */
    tetrads::make_tetrad(fluid_params.u_con, b_hat_con, g_cov, e_con, e_cov);

    double k_tetrad[consts::n_dim];

    tetrads::coordinate_to_tetrad(e_cov, photon.k, k_tetrad);

    if (k_tetrad[0] > 1.0e5 || k_tetrad[0] < 0.0 || std::isnan(k_tetrad[1])) {
        return;
    }

    double p[consts::n_dim];
    proba::sample_electron_distr_p(k_tetrad, p, fluid_params.theta_e);

    double k_tetrad_p[consts::n_dim];
    sample_scattered_photon(k_tetrad, p, k_tetrad_p);

    tetrads::tetrad_to_coordinate(e_con, k_tetrad_p, photon_p.k);

    if (std::isnan(photon_p.k[1])) {
        photon_p.w = 0.0;
        return;
    }

    double tmp_k[consts::n_dim];
    k_tetrad_p[0] *= -1.0;
    tetrads::tetrad_to_coordinate(e_cov, k_tetrad_p, tmp_k);

    photon_p.e = -tmp_k[0];
    photon_p.e_0_s = -tmp_k[0];
    photon_p.l = tmp_k[3];
    photon_p.tau_abs = 0.0;
    photon_p.tau_scatt = 0.0;
    photon_p.b_0 = fluid_params.b;

    photon_p.x1i = photon.x[1];
    photon_p.x2i = photon.x[2];
    photon_p.x[0] = photon.x[0];
    photon_p.x[1] = photon.x[1];
    photon_p.x[2] = photon.x[2];
    photon_p.x[3] = photon.x[3];

    photon_p.n_e_0 = photon.n_e_0;
    photon_p.theta_e_0 = photon.theta_e_0;
    photon_p.e_0 = photon.e_0;
    photon_p.n_scatt = photon.n_scatt + 1;
}

void HARMModel::sample_scattered_photon(const double (&k)[consts::n_dim],
                                        double (&p)[consts::n_dim],
                                        double (&kp)[consts::n_dim]) const {
    double ke[consts::n_dim];

    boost(k, p, ke);

    double k0p;
    double c_th;

    if (ke[0] > 1.0e-4) {
        k0p = proba::sample_klein_nishina(ke[0]);
        c_th = 1.0 - 1.0 / k0p + 1.0 / ke[0];
    } else {
        k0p = ke[0];
        c_th = proba::sample_thomson();
    }
    double s_th = std::sqrt(std::abs(1.0 - c_th * c_th));

    double v0x = ke[1] / ke[0];
    double v0y = ke[2] / ke[0];
    double v0z = ke[3] / ke[0];

    auto [n0x, n0y, n0z] = proba::sample_rand_dir();

    double n0dotv0 = v0x * n0x + v0y * n0y + v0z * n0z;

    /* unit vector 2 */
    double v1x = n0x - (n0dotv0)*v0x;
    double v1y = n0y - (n0dotv0)*v0y;
    double v1z = n0z - (n0dotv0)*v0z;
    double v1 = std::sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
    v1x /= v1;
    v1y /= v1;
    v1z /= v1;

    /* find one more unit vector using cross product;
       this guy is automatically normalized */
    double v2x = v0y * v1z - v0z * v1y;
    double v2y = v0z * v1x - v0x * v1z;
    double v2z = v0x * v1y - v0y * v1x;

    /* now resolve new momentum vector along unit vectors */
    /* create a four-vector $p$ */
    /* solve for orientation of scattered photon */

    /* find phi for new photon */
    double phi = 2.0 * std::numbers::pi * monty_rand::rand();
    double s_phi = std::sin(phi);
    double c_phi = std::cos(phi);

    p[1] *= -1.;
    p[2] *= -1.;
    p[3] *= -1.;

    double dir1 = c_th * v0x + s_th * (c_phi * v1x + s_phi * v2x);
    double dir2 = c_th * v0y + s_th * (c_phi * v1y + s_phi * v2y);
    double dir3 = c_th * v0z + s_th * (c_phi * v1z + s_phi * v2z);

    double kpe[consts::n_dim] = {
        k0p,
        k0p * dir1,
        k0p * dir2,
        k0p * dir3,
    };

    /* transform k back to lab frame */
    boost(kpe, p, kp);
}

void HARMModel::push_photon(struct photon::Photon &photon, double dl, int n) {
    if (photon.x[1] < header_.x_start[1]) {
        return;
    }

    double x_cpy[consts::n_dim];
    double k_cpy[consts::n_dim];
    double dk_cpy[consts::n_dim];

    std::copy(std::begin(photon.x), std::end(photon.x), std::begin(x_cpy));
    std::copy(std::begin(photon.k), std::end(photon.k), std::begin(k_cpy));
    std::copy(std::begin(photon.dkdlam), std::end(photon.dkdlam), std::begin(dk_cpy));

    double dl_2 = 0.5 * dl;
    double k[consts::n_dim];

    for (int i = 0; i < consts::n_dim; ++i) {
        double dk = photon.dkdlam[i] * dl_2;
        photon.k[i] += dk;
        k[i] = photon.k[i] + dk;
        photon.x[i] += photon.k[i] * dl;
    }

    double lconn[consts::n_dim][consts::n_dim][consts::n_dim];

    get_connection(photon.x, lconn);

    double err;
    int iter = 0;

    do {
        ++iter;

        double k_cont[consts::n_dim];
        std::copy(std::begin(k), std::end(k), std::begin(k_cont));

        err = 0.0;

        for (int i = 0; i < consts::n_dim; ++i) {
            photon.dkdlam[i] =
                -2.0 *
                (k_cont[0] * (lconn[i][0][1] * k_cont[1] + lconn[i][0][2] * k_cont[2] + lconn[i][0][3] * k_cont[3]) +
                 k_cont[1] * (lconn[i][1][2] * k_cont[2] + lconn[i][1][3] * k_cont[3]) +
                 lconn[i][2][3] * k_cont[2] * k_cont[3]);
            photon.dkdlam[i] -= (lconn[i][0][0] * k_cont[0] * k_cont[0] + lconn[i][1][1] * k_cont[1] * k_cont[1] +
                                 lconn[i][2][2] * k_cont[2] * k_cont[2] + lconn[i][3][3] * k_cont[3] * k_cont[3]);

            k[i] = photon.k[i] + dl_2 * photon.dkdlam[i];
            err += std::abs((k_cont[i] - k[i]) / (k[i] + consts::eps));
        }
    } while (err > consts::e_tol && iter < consts::max_iter);

    std::copy(std::begin(k), std::end(k), std::begin(photon.k));

    ndarray::NDArray<double, 2> g_cov({consts::n_dim, consts::n_dim});
    gcov_func(photon.x, g_cov);

    double e_1 = -(photon.k[0] * g_cov(0, 0) + photon.k[1] * g_cov(0, 1) + photon.k[2] * g_cov(0, 2) +
                   photon.k[3] * g_cov(0, 3));

    double err_e = std::abs((e_1 - photon.e_0_s) / photon.e_0_s);

    if (n < 7 && (err_e > 1.0e-4 || err > consts::e_tol || std::isnan(err) || std::isinf(err))) {
        std::copy(std::begin(x_cpy), std::end(x_cpy), std::begin(photon.x));
        std::copy(std::begin(k_cpy), std::end(k_cpy), std::begin(photon.k));
        std::copy(std::begin(dk_cpy), std::end(dk_cpy), std::begin(photon.dkdlam));
        push_photon(photon, 0.5 * dl, n + 1);
        push_photon(photon, 0.5 * dl, n + 1);
        e_1 = photon.e_0_s;
    }

    photon.e_0_s = e_1;
}

void HARMModel::record_super_photon(const struct photon::Photon &photon, int n_step) {
    if (std::isnan(photon.w) || std::isnan(photon.e)) {
        return;
    }

    if (photon.tau_scatt > max_tau_scatt_) {
        max_tau_scatt_ = photon.tau_scatt;
    }

    double dx2 = (header_.x_stop[2] - header_.x_start[2]) / (2.0 * consts::n_th_bins);
    int ix2;
    if (photon.x[2] < 0.5 * (header_.x_start[2] + header_.x_stop[2])) {
        ix2 = static_cast<int>(photon.x[2] / dx2);
    } else {
        ix2 = static_cast<int>((header_.x_stop[2] - photon.x[2]) / dx2);
    }

    if (ix2 < 0 || ix2 >= consts::n_th_bins) {
        return;
    }

    double l_e = std::log(photon.e);
    int i_e = static_cast<int>((l_e - consts::spectrum::l_e_0) / consts::spectrum::d_l_e + 2.5) - 2;

    if (i_e < 0 || i_e >= consts::n_e_bins) {
        return;
    }

    ++n_super_photon_recorded_;
    n_super_photon_scatt_ += photon.n_scatt;

    /* sum in photon */
    spectrum_[ix2][i_e].dn_dle += photon.w;
    spectrum_[ix2][i_e].de_dle += photon.w * photon.e;
    spectrum_[ix2][i_e].tau_abs += photon.w * photon.tau_abs;
    spectrum_[ix2][i_e].tau_scatt += photon.w * photon.tau_scatt;
    spectrum_[ix2][i_e].x1i_av += photon.w * photon.x1i;
    spectrum_[ix2][i_e].x2i_sq += photon.w * (photon.x2i * photon.x2i);
    spectrum_[ix2][i_e].x3f_sq += photon.w * (photon.x[3] * photon.x[3]);
    spectrum_[ix2][i_e].ne_0 += photon.w * (photon.n_e_0);
    spectrum_[ix2][i_e].b_0 += photon.w * (photon.b_0);
    spectrum_[ix2][i_e].theta_e_0 += photon.w * (photon.theta_e_0);
    spectrum_[ix2][i_e].nscatt += photon.n_scatt;
    spectrum_[ix2][i_e].nph += 1.0;
}

std::tuple<double, double> HARMModel::init_zone(int x_1, int x_2) const {
    auto fluid_zone = get_fluid_zone(x_1, x_2);

    if (fluid_zone.n_e == 0.0 || fluid_zone.theta_e < consts::theta_e_min) {
        return {0.0, 0.0};
    }

    double l_bth = std::log(fluid_zone.b * fluid_zone.theta_e * fluid_zone.theta_e);

    double d_l = (l_bth - consts::l_b_min) / (consts::d_l_b);

    int l = static_cast<int>(d_l);
    d_l -= l;

    if (l < 0) {
        return {0.0, 0.0};
    }

    double ninterp = 0.0;
    double dn_max = 0.0;

    if (l >= consts::nint) {
        spdlog::warn("outside of nint table range: {}", fluid_zone.b * fluid_zone.theta_e * fluid_zone.theta_e);
        for (int i = 0; i <= consts::n_e_samp; ++i) {
            double dn = jnu_mixed::f_eval(
                            fluid_zone.theta_e, fluid_zone.b, std::exp(x_2 * consts::d_l_nu + consts::l_nu_min), f_) /
                        (std::exp(weight_[i]) + 1.0e-100);
            if (dn > dn_max) {
                dn_max = dn;
            }

            ninterp += consts::d_l_nu * dn;
        }
    } else if (!std::isinf(nint_[l]) && !std::isinf(nint_[l + 1])) {
        ninterp = std::exp((1.0 - d_l) * nint_[l] + d_l * nint_[l + 1]);
        dn_max = std::exp((1.0 - d_l) * dndlnu_max_[l] + d_l * dndlnu_max_[l + 1]);
    }

    double k2 = jnu_mixed::k2_eval(fluid_zone.theta_e, k2_);

    if (k2 == 0.0) {
        return {0.0, 0.0};
    }

    double nz = geometry_.det(x_1, x_2) * fluid_zone.n_e * fluid_zone.b * fluid_zone.theta_e * fluid_zone.theta_e *
                ninterp / k2;

    if (nz > photon_n_ * std::log(consts::nu_max / consts::nu_min)) {
        return {0.0, 0.0};
    }

    return {nz, dn_max};
}

double HARMModel::bias_func(double t_e, double w) const {
    double max = 0.5 * w / consts::weight_min;
    double avg_num_scatt = n_super_photon_scatt_ / (1.0 * n_super_photon_recorded_ + 1.0);
    double bias = 100.0 * t_e * t_e / (bias_norm_ * max_tau_scatt_ * (avg_num_scatt + 2.0));

    if (bias < consts::tp_over_te) {
        bias = consts::tp_over_te;
    }
    if (bias > max) {
        bias = max;
    }

    return bias / consts::tp_over_te;
}

std::tuple<int, int, double, double> HARMModel::x_to_ij(const double (&x)[consts::n_dim]) const {
    int i = static_cast<int>((x[1] - header_.x_start[1]) / header_.dx[1] - 0.5 + 1000) - 1000;
    int j = static_cast<int>((x[2] - header_.x_start[2]) / header_.dx[2] - 0.5 + 1000) - 1000;

    double del_i;
    double del_j;

    if (i < 0) {
        i = 0;
        del_i = 0.0;
    } else if (i > static_cast<int>(header_.n[0]) - 2) {
        i = header_.n[0] - 2;
        del_i = 1.0;
    } else {
        del_i = (x[1] - ((i + 0.5) * header_.dx[1] + header_.x_start[1])) / header_.dx[1];
    }

    if (j < 0) {
        j = 0;
        del_j = 0.0;
    } else if (j > static_cast<int>(header_.n[1]) - 2) {
        j = header_.n[1] - 2;
        del_j = 1.0;
    } else {
        del_j = (x[2] - ((j + 0.5) * header_.dx[2] + header_.x_start[2])) / header_.dx[2];
    }

    return {i, j, del_i, del_j};
}

void HARMModel::get_connection(const double (&x)[consts::n_dim],
                               double (&lconn)[consts::n_dim][consts::n_dim][consts::n_dim]) {
    double r1 = std::exp(x[1]);
    double r2 = r1 * r1;
    double r3 = r2 * r1;
    double r4 = r3 * r1;

    double s_x = std::sin(2.0 * std::numbers::pi * x[2]);
    double c_x = std::cos(2.0 * std::numbers::pi * x[2]);

    double th = std::numbers::pi * x[2] + 0.5 * (1.0 - header_.h_slope) * s_x;
    double dthdx2 = std::numbers::pi * (1.0 + (1.0 - header_.h_slope) * c_x);
    double d2thdx22 = -2.0 * std::numbers::pi * std::numbers::pi * (1.0 - header_.h_slope) * s_x;
    double dthdx22 = dthdx2 * dthdx2;

    double sth = std::sin(th);
    double cth = std::cos(th);

    double sth2 = sth * sth;
    double r1sth2 = r1 * sth2;
    double sth4 = sth2 * sth2;
    double cth2 = cth * cth;
    double cth4 = cth2 * cth2;
    double s2th = 2.0 * sth * cth;
    double c2th = 2.0 * cth2 - 1.0;

    double a = header_.a;
    double a2 = a * a;
    double a3 = a2 * a;
    double a4 = a3 * a;
    double a2sth2 = a2 * sth2;
    double a2cth2 = a2 * cth2;
    double a4cth4 = a4 * cth4;

    double rho2 = r2 + a2cth2;
    double rho22 = rho2 * rho2;
    double rho23 = rho22 * rho2;
    double irho2 = 1.0 / rho2;
    double irho22 = irho2 * irho2;
    double irho23 = irho22 * irho2;
    double irho23_dthdx2 = irho23 / dthdx2;

    double fac1 = r2 - a2cth2;
    double fac1_rho23 = fac1 * irho23;
    double fac2 = a2 + 2.0 * r2 + a2 * c2th;
    double fac3 = a2 + r1 * (-2.0 + r1);

    lconn[0][0][0] = 2.0 * r1 * fac1_rho23;
    lconn[0][0][1] = r1 * (2.0 * r1 + rho2) * fac1_rho23;
    lconn[0][0][2] = -a2 * r1 * s2th * dthdx2 * irho22;
    lconn[0][0][3] = -2.0 * a * r1sth2 * fac1_rho23;

    /* lconn[0][1][0] = lconn[0][0][1]; */
    lconn[0][1][1] = 2.0 * r2 * (r4 + r1 * fac1 - a4cth4) * irho23;
    lconn[0][1][2] = -a2 * r2 * s2th * dthdx2 * irho22;
    lconn[0][1][3] = a * r1 * (-r1 * (r3 + 2.0 * fac1) + a4cth4) * sth2 * irho23;

    /* lconn[0][2][0] = lconn[0][0][2]; */
    /* lconn[0][2][1] = lconn[0][1][2]; */
    lconn[0][2][2] = -2.0 * r2 * dthdx22 * irho2;
    lconn[0][2][3] = a3 * r1sth2 * s2th * dthdx2 * irho22;

    /* lconn[0][3][0] = lconn[0][0][3]; */
    /* lconn[0][3][1] = lconn[0][1][3]; */
    /* lconn[0][3][2] = lconn[0][2][3]; */
    lconn[0][3][3] = 2.0 * r1sth2 * (-r1 * rho22 + a2sth2 * fac1) * irho23;

    lconn[1][0][0] = fac3 * fac1 / (r1 * rho23);
    lconn[1][0][1] = fac1 * (-2.0 * r1 + a2sth2) * irho23;
    lconn[1][0][2] = 0.0;
    lconn[1][0][3] = -a * sth2 * fac3 * fac1 / (r1 * rho23);

    /* lconn[1][1][0] = lconn[1][0][1]; */
    lconn[1][1][1] = (r4 * (-2.0 + r1) * (1.0 + r1) + a2 * (a2 * r1 * (1.0 + 3.0 * r1) * cth4 + a4cth4 * cth2 +
                                                            r3 * sth2 + r1 * cth2 * (2.0 * r1 + 3.0 * r3 - a2sth2))) *
                     irho23;
    lconn[1][1][2] = -a2 * dthdx2 * s2th / fac2;
    lconn[1][1][3] = a * sth2 *
                     (a4 * r1 * cth4 + r2 * (2.0 * r1 + r3 - a2sth2) + a2cth2 * (2.0 * r1 * (-1.0 + r2) + a2sth2)) *
                     irho23;

    /* lconn[1][2][0] = lconn[1][0][2]; */
    /* lconn[1][2][1] = lconn[1][1][2]; */
    lconn[1][2][2] = -fac3 * dthdx22 * irho2;
    lconn[1][2][3] = 0.0;

    /* lconn[1][3][0] = lconn[1][0][3]; */
    /* lconn[1][3][1] = lconn[1][1][3]; */
    /* lconn[1][3][2] = lconn[1][2][3]; */
    lconn[1][3][3] = -fac3 * sth2 * (r1 * rho22 - a2 * fac1 * sth2) / (r1 * rho23);

    lconn[2][0][0] = -a2 * r1 * s2th * irho23_dthdx2;
    lconn[2][0][1] = r1 * lconn[2][0][0];
    lconn[2][0][2] = 0.0;
    lconn[2][0][3] = a * r1 * (a2 + r2) * s2th * irho23_dthdx2;

    /* lconn[2][1][0] = lconn[2][0][1]; */
    lconn[2][1][1] = r2 * lconn[2][0][0];
    lconn[2][1][2] = r2 * irho2;
    lconn[2][1][3] =
        (a * r1 * cth * sth * (r3 * (2.0 + r1) + a2 * (2.0 * r1 * (1.0 + r1) * cth2 + a2 * cth4 + 2.0 * r1sth2))) *
        irho23_dthdx2;

    /* lconn[2][2][0] = lconn[2][0][2]; */
    /* lconn[2][2][1] = lconn[2][1][2]; */
    lconn[2][2][2] = -a2 * cth * sth * dthdx2 * irho2 + d2thdx22 / dthdx2;
    lconn[2][2][3] = 0.0;

    /* lconn[2][3][0] = lconn[2][0][3]; */
    /* lconn[2][3][1] = lconn[2][1][3]; */
    /* lconn[2][3][2] = lconn[2][2][3]; */
    lconn[2][3][3] =
        -cth * sth * (rho23 + a2sth2 * rho2 * (r1 * (4.0 + r1) + a2cth2) + 2.0 * r1 * a4 * sth4) * irho23_dthdx2;

    lconn[3][0][0] = a * fac1_rho23;
    lconn[3][0][1] = r1 * lconn[3][0][0];
    lconn[3][0][2] = -2.0 * a * r1 * cth * dthdx2 / (sth * rho22);
    lconn[3][0][3] = -a2sth2 * fac1_rho23;

    /* lconn[3][1][0] = lconn[3][0][1]; */
    lconn[3][1][1] = a * r2 * fac1_rho23;
    lconn[3][1][2] = -2 * a * r1 * (a2 + 2.0 * r1 * (2.0 + r1) + a2 * c2th) * cth * dthdx2 / (sth * fac2 * fac2);
    lconn[3][1][3] = r1 * (r1 * rho22 - a2sth2 * fac1) * irho23;

    /* lconn[3][2][0] = lconn[3][0][2]; */
    /* lconn[3][2][1] = lconn[3][1][2]; */
    lconn[3][2][2] = -a * r1 * dthdx22 * irho2;
    lconn[3][2][3] = dthdx2 * (0.25 * fac2 * fac2 * cth / sth + a2 * r1 * s2th) * irho22;

    /* lconn[3][3][0] = lconn[3][0][3]; */
    /* lconn[3][3][1] = lconn[3][1][3]; */
    /* lconn[3][3][2] = lconn[3][2][3]; */
    lconn[3][3][3] = (-a * r1sth2 * rho22 + a3 * sth4 * fac1) * irho23;
}

void HARMModel::init_dkdlam(const double (&x)[consts::n_dim],
                            const double (&k_con)[consts::n_dim],
                            double (&d_k)[consts::n_dim]) {
    double lconn[consts::n_dim][consts::n_dim][consts::n_dim];

    get_connection(x, lconn);

    for (int i = 0; i < consts::n_dim; ++i) {
        d_k[i] =
            -2.0 *
            (k_con[0] * (lconn[i][0][1] * k_con[1] + lconn[i][0][2] * k_con[2] + lconn[i][0][3] * k_con[3]) +
             k_con[1] * (lconn[i][1][2] * k_con[2] + lconn[i][1][3] * k_con[3]) + lconn[i][2][3] * k_con[2] * k_con[3]);

        d_k[i] -= (lconn[i][0][0] * k_con[0] * k_con[0] + lconn[i][1][1] * k_con[1] * k_con[1] +
                   lconn[i][2][2] * k_con[2] * k_con[2] + lconn[i][3][3] * k_con[3] * k_con[3]);
    }
}

bool HARMModel::stop_criterion(struct photon::Photon &photon) const {
    if (photon.x[1] < x1_min_) {
        /* stop at event horizon */
        return true;
    }

    if (photon.x[1] > consts::x1_max) {
        /* stop at large distance */
        if (photon.w < consts::weight_min) {
            if (monty_rand::rand() <= 1.0 / consts::roulette) {
                photon.w *= consts::roulette;
            } else {
                photon.w = 0.0;
            }
        }
        return true;
    }

    if (photon.w < consts::weight_min) {
        if (monty_rand::rand() <= 1.0 / consts::roulette) {
            photon.w *= consts::roulette;
        } else {
            photon.w = 0.0;
            return true;
        }
    }
    return false;
}

bool HARMModel::record_criterion(const struct photon::Photon &photon) const { return (photon.x[1] > consts::x1_max); }

double HARMModel::step_size(const double (&x)[consts::n_dim], const double (&k)[consts::n_dim]) {
    double dl_x_1 = consts::step_eps * x[1] / (std::abs(k[1]) + consts::eps);
    double dl_x_2 = consts::step_eps * std::min(x[2], header_.x_stop[2] - x[2]) / (std::abs(k[2]) + consts::eps);
    double dl_x_3 = consts::step_eps / (std::abs(k[3]) + consts::eps);

    double i_dl_x_1 = 1.0 / (std::abs(dl_x_1) + consts::eps);
    double i_dl_x_2 = 1.0 / (std::abs(dl_x_2) + consts::eps);
    double i_dl_x_3 = 1.0 / (std::abs(dl_x_3) + consts::eps);

    return 1.0 / (i_dl_x_1 + i_dl_x_2 + i_dl_x_3);
}

struct BLCoord HARMModel::get_bl_coord(const double (&x)[consts::n_dim]) const {
    return BLCoord{
        .r = std::exp(x[1]) + header_.r_0,
        .theta = std::numbers::pi * x[2] + ((1.0 - header_.h_slope) / 2.0) * std::sin(2.0 * std::numbers::pi * x[2]),
    };
}

void HARMModel::get_coord(int x_1, int x_2, double (&x)[consts::n_dim]) const {
    x[0] = header_.x_start[0];
    x[1] = header_.x_start[1] + (x_1 + 0.5) * header_.dx[1];
    x[2] = header_.x_start[2] + (x_2 + 0.5) * header_.dx[2];
    x[3] = header_.x_start[3];
}

static double
interp_scalar(const ndarray::NDArray<double, 2> &var, int i, int j, const double (&coeff)[consts::n_dim]) {
    /* clang-format off */
    return (
        var(i    , j    ) * coeff[0] +
        var(i    , j + 1) * coeff[1] +
        var(i + 1, j    ) * coeff[2] +
        var(i + 1, j + 1) * coeff[3]
    );
    /* clang-format on */
}

static void boost(const double (&v)[consts::n_dim], const double (&u)[consts::n_dim], double (&vp)[consts::n_dim]) {
    double g = u[0];
    double v_ = std::sqrt(std::abs(1.0 - 1.0 / (g * g)));
    double n1 = u[1] / (g * v_ + consts::eps);
    double n2 = u[2] / (g * v_ + consts::eps);
    double n3 = u[3] / (g * v_ + consts::eps);
    double gm1 = g - 1.0;

    /* general Lorentz boost into frame u from lab frame */
    vp[0] = u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3];
    vp[1] = -u[1] * v[0] + (1.0 + n1 * n1 * gm1) * v[1] + n1 * n2 * gm1 * v[2] + n1 * n3 * gm1 * v[3];
    vp[2] = -u[2] * v[0] + n2 * n1 * gm1 * v[1] + (1.0 + n2 * n2 * gm1) * v[2] + n2 * n3 * gm1 * v[3];
    vp[3] = -u[3] * v[0] + n3 * n1 * gm1 * v[1] + n3 * n2 * gm1 * v[2] + (1.0 + n3 * n3 * gm1) * v[3];
}

}; /* namespace harm */
