/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <string>
#include <tuple>

#include "cuda_grmonty/consts.hpp"
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
    ndarray::NDArray<double> k_rho; /* rest-mass density */
    ndarray::NDArray<double> u;     /* internal eneergy density */
    ndarray::NDArray<double> u_1;   /* covariant velocity components */
    ndarray::NDArray<double> u_2;
    ndarray::NDArray<double> u_3;
    ndarray::NDArray<double> b_1; /* contravariant magnetic field components */
    ndarray::NDArray<double> b_2;
    ndarray::NDArray<double> b_3;
};

struct Geometry {
    ndarray::NDArray<double> cov;
    ndarray::NDArray<double> con;
    ndarray::NDArray<double> det;
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

struct Photon {
    double x[consts::n_dim];
    double k[consts::n_dim];
    double dkdlam[consts::n_dim];
    double w;
    double e;
    double l;
    double x1i;
    double x2i;
    double tau_abs;
    double tau_scatt;
    double n_e_0;
    double theta_e_0;
    double b_0;
    double e_0;
    double e_0_s;
    int n_scatt;
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

    /**
     * @brief Initializes the metric.
     */
    void init_geometry();

    void init_weight_table();

    void init_nint_table();

    void gcon_func(const double (&x)[consts::n_dim], ndarray::NDArray<double> &&gcon) const;

    void gcov_func(const double (&x)[consts::n_dim], ndarray::NDArray<double> &&gcov) const;

    struct FluidZone get_fluid_zone(int x_1, int x_2) const;

    struct FluidParams get_fluid_params(const double (&x)[consts::n_dim], const ndarray::NDArray<double> &g_cov) const;

    /**
     * @brief Return the next zone and the number of superphotons that need to be generated in it.
     */
    struct Zone get_zone();

    struct Photon sample_zone_photon(struct Zone &zone);

    double linear_interp_weight(double nu);

    std::tuple<struct Photon, bool> make_super_photon();

    void track_super_photon(struct Photon &photon);

    void scatter_super_photon(struct Photon &photon,
                              struct Photon &photon_2,
                              const struct FluidParams &fluid_params,
                              const ndarray::NDArray<double> &g_cov,
                              double b_unit) const;

    void
    sample_scattered_photon(const double (&k)[consts::n_dim], double (&p)[consts::n_dim], double (&kp)[consts::n_dim]);

    void push_photon(struct Photon &photon, double dl, int n);

    void record_super_photon(const struct Photon &photon);

    std::tuple<double, double> init_zone(int x_1, int x_2) const;

    double bias_func(double t_e, double w) const;

    std::tuple<int, int, double, double> x_to_ij(const double (&x)[consts::n_dim]) const;

    void get_connection(const double (&x)[consts::n_dim], double (&lconn)[consts::n_dim][consts::n_dim][consts::n_dim]);

    void
    init_dkdlam(const double (&x)[consts::n_dim], const double (&k_con)[consts::n_dim], double (&d_k)[consts::n_dim]);

    bool stop_criterion(struct Photon &photon) const;

    bool record_criterion(const struct Photon &photon) const;

    double step_size(const double (&x)[consts::n_dim], const double (&k)[consts::n_dim]);

    struct BLCoord get_bl_coord(const double (&x)[consts::n_dim]) const;

    void get_coord(int x_1, int x_2, double (&x)[consts::n_dim]) const;

    const struct Header *get_header() const { return &header_; }

    const struct Data *get_data() const { return &data_; }

    const int get_photon_n() const { return photon_n_; }

    const double get_l_unit() const { return l_unit_; }

    const Geometry &get_geometry() const { return geometry_; }

    const ndarray::NDArray<double> &get_k2_table() const { return k2_; }

private:
    struct Header header_;
    struct Data data_;

    double bias_norm_;
    double rh_;

    double n_scatt_;
    double n_super_photon_recorded_;

    /* spectral bin parameters */
    double d_l_e_ = 0.25;              /* bin width */
    double l_e_0_ = std::log(1.0e-12); /* location of first bin, in electron rest-mass units */

    int photon_n_;
    double mass_unit_;
    double l_unit_;
    double t_unit_;
    double rho_unit_;
    double u_unit_;
    double b_unit_;
    double theta_e_unit_;
    double n_e_unit_;
    double max_tau_scatt_;

    int zone_x_1_ = 0;
    int zone_x_2_ = -1;
    bool zone_flag_ = 0;

    struct Geometry geometry_;
    ndarray::NDArray<double> hotcross_table_ =
        ndarray::NDArray<double>({consts::hotcross::n_w + 1, consts::hotcross::n_t + 1});
    ndarray::NDArray<double> f_ = ndarray::NDArray<double>({consts::n_e_samp + 1});
    ndarray::NDArray<double> k2_ = ndarray::NDArray<double>({consts::n_e_samp + 1});
    ndarray::NDArray<double> weight_ = ndarray::NDArray<double>({consts::n_e_samp + 1});
    ndarray::NDArray<double> nint_ = ndarray::NDArray<double>({consts::nint + 1});
    ndarray::NDArray<double> dndlnu_max_ = ndarray::NDArray<double>({consts::nint + 1});

    struct Spectrum spectrum_[consts::n_th_bins][consts::n_e_bins];
};

}; /* namespace harm */
