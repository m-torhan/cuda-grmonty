/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cmath>
#include <numbers>

namespace consts {

constexpr int n_dim = 4;
constexpr int n_prim = 8;

constexpr double eps = 1.0e-40;

constexpr int n_e_samp = 200;
constexpr int n_e_bins = 200;
constexpr int n_th_bins = 6;

/* range of initial superphoton frequencies */
constexpr double nu_min = 1.0e9;
constexpr double nu_max = 1.0e16;

constexpr double theta_e_min = 0.3;
constexpr double theta_e_max = 1000.0;
constexpr double tp_over_te = 3.0;

constexpr double weight_min = 1.0e31;

constexpr double r_max = 100.0;
constexpr double roulette = 1.0e4;

constexpr double etol = 1.0e-3;
constexpr int max_iter = 2;
constexpr int max_n_step = 1280000;

constexpr double ee = 4.80320680e-10;                /* electron charge */
constexpr double cl = 2.99792458e10;                 /* speed of light */
constexpr double me = 9.1093826e-28;                 /* electron mass */
constexpr double mp = 1.67262171e-24;                /* proton mass */
constexpr double mn = 1.67492728e-24;                /* neutron mass */
constexpr double amu = 1.66053886e-24;               /* atomic mass unit */
constexpr double hpl = 6.6260693e-27;                /* Planck constant */
constexpr double hbar = hpl / 2. * std::numbers::pi; /* Planck's consant / 2pi */
constexpr double kbol = 1.3806505e-16;               /* Boltzmann constant */
constexpr double g_newt = 6.6742e-8;                 /* Gravitational constant */
constexpr double sif = 5.670400e-5;                  /* Stefan-Boltzmann constant */
constexpr double rgas = 8.3143e7;                    /* erg K^-1 mole^-1: ideal gas const */
constexpr double ev = 1.60217653e-12;                /* electron volt in erg */
constexpr double sigma_thomson = 0.665245873e-24;    /* Thomson cross section in cm^2 */
constexpr double jy = 1.e-23;                        /* Jansky flux/freq. uni; in cgs */

constexpr double pc = 3.085678e18;      /* parsec */
constexpr double au = 1.49597870691e13; /* Astronomical Unit */

constexpr double m_sun = 1.989e33;     /* solar mass */
constexpr double r_sun = 6.96e10;      /* Radius of Sun */
constexpr double l_sun = 3.827e33;     /* Luminousity of Sun */
constexpr double t_sun = 5.78e3;       /* Temperature of Sun's photosphere */
constexpr double m_bh = 4.0e6 * m_sun; /* black hole mass */

constexpr int nint = 20000;
constexpr double bthsq_min = 1.0e-4;
constexpr double bthsq_max = 1.0e8;

namespace hotcross {

constexpr double min_w = 1.0e-12;
constexpr double max_w = 1.0e6;
constexpr double min_t = 1.0e-4;
constexpr double max_t = 1.0e4;
constexpr int n_w = 220;
constexpr int n_t = 80;

constexpr double max_gamma = 12.0;
constexpr double d_mu_e = 0.05;
constexpr double d_gamma_e = 0.05;

}; /* namespace hotcross */

namespace jnu {

constexpr double eps_abs = 0.0;
constexpr double eps_rel = 1.0e-6;

constexpr double min_k = 0.002;
constexpr double max_k = 1.0e7;
constexpr double min_t = theta_e_min;
constexpr double max_t = 1.0e2;

constexpr double cst = 1.88774862536; /* 2^{11/12} */

constexpr double k_fac = 9 * std::numbers::pi * me * cl / ee;

}; /* namespace jnu */

namespace super_photon {

constexpr double jcst = std::numbers::sqrt2 * ee * ee * ee / (27.0 * me * cl * cl);

}; /* namespace super_photon */

}; /* namespace consts */
