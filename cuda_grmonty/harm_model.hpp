/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <semaphore>
#include <string>
#include <tuple>

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/harm_data.hpp"
#include "cuda_grmonty/ndarray.hpp"
#include "cuda_grmonty/photon.hpp"
#include "cuda_grmonty/photon_queue.hpp"

namespace harm {

class HARMModel {
public:
    explicit HARMModel(int photon_n, double mass_unit);

    HARMModel(const HARMModel &) = delete;

    HARMModel &operator=(const HARMModel &) = delete;

    /**
     * @brief Reads HARM data from file.
     *
     * @param filepath Path to HARM dump.
     */
    void read_file(std::string filepath);

    void init();

    void run_simulation();

    void report_spectrum(std::string filepath);

    const struct Header *get_header() const { return &header_; }

    const struct Data *get_data() const { return &data_; }

private:
    struct Header header_;
    struct Data data_;
    struct Units units_;

    double bias_norm_;
    double rh_;

    int photon_n_;
    double max_tau_scatt_;
    double d_tau_k_;
    double x1_min_;

    uint64_t n_super_photon_created_ = 0;
    uint64_t n_super_photon_scatt_ = 0;
    uint64_t n_super_photon_recorded_ = 0;

    int zone_x_1_ = 0;
    int zone_x_2_ = -1;

    struct Geometry geometry_;
    ndarray::NDArray<double, 2> hotcross_table_ =
        ndarray::NDArray<double, 2>({consts::hotcross::n_w + 1, consts::hotcross::n_t + 1});
    std::array<double, consts::n_e_samp + 1> f_;
    std::array<double, consts::n_e_samp + 1> k2_;
    std::array<double, consts::n_e_samp + 1> weight_;
    std::array<double, consts::nint + 1> nint_;
    std::array<double, consts::nint + 1> dndlnu_max_;

    struct Spectrum spectrum_[consts::n_th_bins][consts::n_e_bins];

    /**
     * @brief Initializes the metric.
     */
    void init_geometry();

    void init_weight_table();

    void init_nint_table();

    void gcon_func(const double (&x)[consts::n_dim], ndarray::NDArray<double, 2> &g_con) const;

    void gcov_func(const double (&x)[consts::n_dim], ndarray::NDArray<double, 2> &g_cov) const;

    double d_omega_func(double x2i, double x2f) const;

    struct FluidZone get_fluid_zone(int x_1, int x_2) const;

    struct FluidParams get_fluid_params(const double (&x)[consts::n_dim],
                                        const ndarray::NDArray<double, 2> &g_cov) const;

    /**
     * @brief Return the next zone and the number of superphotons that need to be generated in it.
     */
    struct Zone get_zone();

    struct photon::Photon sample_zone_photon(struct Zone &zone);

    double linear_interp_weight(double nu);

    std::tuple<struct photon::Photon, bool> make_super_photon();

    void make_super_photon_async(photon::PhotonQueue &photon_queue, std::binary_semaphore &done_sem);

    void track_super_photon(struct photon::Photon &photon);

    void scatter_super_photon(struct photon::Photon &photon,
                              struct photon::Photon &photon_p,
                              const struct FluidParams &fluid_params,
                              const ndarray::NDArray<double, 2> &g_cov,
                              double b_unit) const;

    void sample_scattered_photon(const double (&k)[consts::n_dim],
                                 double (&p)[consts::n_dim],
                                 double (&kp)[consts::n_dim]) const;

    void push_photon(struct photon::Photon &photon, double dl, int n);

    void record_super_photon(const struct photon::Photon &photon, int n_step);

    std::tuple<double, double> init_zone(int x_1, int x_2) const;

    double bias_func(double t_e, double w) const;

    std::tuple<int, int, double, double> x_to_ij(const double (&x)[consts::n_dim]) const;

    void get_connection(const double (&x)[consts::n_dim], double (&lconn)[consts::n_dim][consts::n_dim][consts::n_dim]);

    void
    init_dkdlam(const double (&x)[consts::n_dim], const double (&k_con)[consts::n_dim], double (&d_k)[consts::n_dim]);

    bool stop_criterion(struct photon::Photon &photon) const;

    bool record_criterion(const struct photon::Photon &photon) const;

    double step_size(const double (&x)[consts::n_dim], const double (&k)[consts::n_dim]);

    struct BLCoord get_bl_coord(const double (&x)[consts::n_dim]) const;

    void get_coord(int x_1, int x_2, double (&x)[consts::n_dim]) const;
};

}; /* namespace harm */
