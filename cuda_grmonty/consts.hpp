/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <numbers>

namespace consts {

/* Dimensional and primitive variable counts. */
constexpr int n_dim = 4;  /* Number of spacetime dimensions (t, r, θ, φ). */
constexpr int n_prim = 8; /* Number of primitive fluid variables. */

/* Numerical tolerances and tiny values. */
constexpr double eps = 1.0e-40; /* Machine-level epsilon used to avoid division by zero. */

/* Photon energy sampling and spectrum binning. */
constexpr int n_e_samp = 200; /* Number of electron energy samples for tables. */
constexpr int n_e_bins = 200; /* Number of photon energy bins in the spectrum. */
constexpr int n_th_bins = 6;  /* Number of observer inclination angle bins. */

/* Range of initial superphoton frequencies. */
constexpr double nu_min = 1.0e9;  /* Minimum photon frequency [Hz]. */
constexpr double nu_max = 1.0e16; /* Maximum photon frequency [Hz]. */

/* Precomputed logarithmic frequency range parameters. */
const double l_nu_min = std::log(nu_min);               /* log(ν_min). */
const double l_nu_max = std::log(nu_max);               /* log(ν_max). */
const double n_l_n = l_nu_max - l_nu_min;               /* Total log-frequency span. */
const double d_l_nu = (l_nu_max - l_nu_min) / n_e_samp; /* Logarithmic frequency step. */

/* Electron temperature limits and proton-to-electron temperature ratio. */
constexpr double theta_e_min = 0.3;    /* Minimum electron temperature in units of m_e c^2 / k_B. */
constexpr double theta_e_max = 1000.0; /* Maximum electron temperature in units of m_e c^2 / k_B. */
constexpr double tp_over_te = 3.0;     /* Ratio of proton to electron temperatures. */

/* Photon weight thresholds for Russian roulette termination. */
constexpr double weight_min = 1.0e31; /* Minimum photon weight before roulette. */
constexpr double roulette = 1.0e4;    /* Russian roulette survival factor. */

/* Spatial domain bounds. */
constexpr double r_max = 100.0;        /* Maximum radius [GM/c^2]. */
const double x1_max = std::log(r_max); /* Logarithmic maximum radius. */

/* Integration tolerances and iteration limits. */
constexpr double step_eps = 0.04;   /* Maximum fractional step size for geodesic integration. */
constexpr double e_tol = 1.0e-3;    /* Relative error tolerance for iterative solvers. */
constexpr int max_iter = 2;         /* Maximum iterations for root finding. */
constexpr int max_n_step = 1280000; /* Maximum number of integration steps per photon. */

/* Physical constants (CGS units). */
constexpr double ee = 4.80320680e-10;                  /* Electron charge [statC]. */
constexpr double cl = 2.99792458e10;                   /* Speed of light [cm/s]. */
constexpr double me = 9.1093826e-28;                   /* Electron mass [g]. */
constexpr double mp = 1.67262171e-24;                  /* Proton mass [g]. */
constexpr double mn = 1.67492728e-24;                  /* Neutron mass [g]. */
constexpr double amu = 1.66053886e-24;                 /* Atomic mass unit [g]. */
constexpr double hpl = 6.6260693e-27;                  /* Planck constant [erg·s]. */
constexpr double hbar = hpl / (2. * std::numbers::pi); /* Reduced Planck constant ħ = h / 2π. */
constexpr double kbol = 1.3806505e-16;                 /* Boltzmann constant [erg/K]. */
constexpr double g_newt = 6.6742e-8;                   /* Gravitational constant [cm^3/g/s^2]. */
constexpr double sif = 5.670400e-5;                    /* Stefan–Boltzmann constant [erg/cm^2/s/K^4]. */
constexpr double rgas = 8.3143e7;                      /* Ideal gas constant [erg/(K·mol)]. */
constexpr double ev = 1.60217653e-12;                  /* Electron volt [erg]. */
constexpr double sigma_thomson = 0.665245873e-24;      /* Thomson cross section [cm^2]. */
constexpr double jy = 1.e-23;                          /* Jansky flux density unit [erg/s/cm^2/Hz]. */

/* Astronomical distance units. */
constexpr double pc = 3.085678e18;      /* Parsec, in cm */
constexpr double au = 1.49597870691e13; /* Astronomical Unit, in cm */

/* Solar reference values and example black hole mass. */
constexpr double m_sun = 1.989e33;     /* Solar mass, in grams */
constexpr double r_sun = 6.96e10;      /* Solar radius, in cm */
constexpr double l_sun = 3.827e33;     /* Solar luminosity, in erg/s */
constexpr double t_sun = 5.78e3;       /* Solar photosphere temperature, in Kelvin */
constexpr double m_bh = 4.0e6 * m_sun; /* Example black hole mass, 4 million solar masses */

/* Integration parameters and b-threshold range. */
constexpr int nint = 20000;                                  /* Number of integration steps */
constexpr double bthsq_min = 1.0e-4;                         /* Minimum b^2 value */
constexpr double bthsq_max = 1.0e8;                          /* Maximum b^2 value */
const double l_b_min = std::log(bthsq_min);                  /* Logarithm of minimum b^2 */
const double d_l_b = std::log(bthsq_max / bthsq_min) / nint; /* Logarithmic step in b^2 */

/**
 * @brief Hot cross-section calculation parameters.
 */
namespace hotcross {

constexpr double min_w = 1.0e-12; /* Minimum photon energy in units of mec^2 */
constexpr double max_w = 1.0e6;   /* Maximum photon energy in units of mec^2 */
constexpr double min_t = 1.0e-4;  /* Minimum electron temperature (dimensionless) */
constexpr double max_t = 1.0e4;   /* Maximum electron temperature (dimensionless) */
constexpr int n_w = 220;          /* Number of photon energy bins */
constexpr int n_t = 80;           /* Number of temperature bins */

constexpr double max_gamma = 12.0; /* Maximum electron Lorentz factor */
constexpr double d_mu_e = 0.05;    /* Step size in electron pitch angle cosine */
constexpr double d_gamma_e = 0.05; /* Step size in electron Lorentz factor */

/* Precomputed logarithmic grid steps */
const double l_min_w = std::log10(hotcross::min_w);                                 /* Logarithm of min photon energy */
const double l_min_t = std::log10(hotcross::min_t);                                 /* Logarithm of min temperature */
const double d_l_w = std::log10(hotcross::max_w / hotcross::min_w) / hotcross::n_w; /* Step in log w */
const double d_l_t = std::log10(hotcross::max_t / hotcross::min_t) / hotcross::n_t; /* Step in log T */

} /* namespace hotcross */

/**
 * @brief Parameters for jν emissivity calculations.
 */
namespace jnu {

constexpr double eps_abs = 0.0;    /* Absolute tolerance for numerical integration */
constexpr double eps_rel = 1.0e-6; /* Relative tolerance for numerical integration */

/* Momentum (k) sampling range */
constexpr double min_k = 0.002;                                    /* Minimum momentum value */
constexpr double max_k = 1.0e7;                                    /* Maximum momentum value */
const double l_min_k = std::log(jnu::min_k);                       /* Logarithm of min momentum */
const double d_l_k = std::log(jnu::max_k / jnu::min_k) / n_e_samp; /* Logarithmic step in momentum */

/* Temperature (θe) sampling range */
constexpr double min_t = theta_e_min;                              /* Minimum electron temperature */
constexpr double max_t = 1.0e2;                                    /* Maximum electron temperature */
const double l_min_t = std::log(jnu::min_t);                       /* Logarithm of min temperature */
const double d_l_t = std::log(jnu::max_t / jnu::min_t) / n_e_samp; /* Logarithmic step in temperature */

constexpr double cst = 1.88774862536;                         /* Precomputed constant 2^{11/12} */
constexpr double k_fac = 9 * std::numbers::pi * me * cl / ee; /* Scaling factor for k */

} /* namespace jnu */

/**
 * @brief Constants for super-photon emission processes.
 */
namespace super_photon {

constexpr double jcst = std::numbers::sqrt2 * ee * ee * ee / (27.0 * me * cl * cl); /* Prefactor for emissivity */

} /* namespace super_photon */

/**
 * @brief Spectrum binning parameters.
 */
namespace spectrum {

constexpr double d_l_e = 0.25;          /* Logarithmic bin width in electron rest-mass units */
const double l_e_0 = std::log(1.0e-12); /* Location of first bin in electron rest-mass units */

} /* namespace spectrum */

/**
 * @brief CUDA kernel launch configuration.
 */
namespace cuda {

constexpr int grid_dim = 128;                           /* Grid dimension */
constexpr int block_dim = 128;                          /* Block dimension */
constexpr int threads_per_grid = block_dim * grid_dim; /* Total number of threads per grid */
constexpr int n_photons = threads_per_grid * 1;        /* Number of photons processed at once */

} /* namespace cuda */

}; /* namespace consts */
