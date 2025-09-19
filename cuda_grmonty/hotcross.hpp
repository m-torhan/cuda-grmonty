/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/ndarray.hpp"

namespace hotcross {

/**
 * @brief Initializes the look-up table for hot cross sections (angle-averaged Compton scattering).
 *
 * @param table 2D array to store precomputed cross-section values.
 */
void init_table(ndarray::NDArray<double, 2> &table);

/**
 * @brief Retrieves the total Compton scattering cross-section from the look-up table.
 *
 * @param w              Dimensionless photon energy (w = h * nu / (m_e * c^2)).
 * @param theta_e        Dimensionless electron temperature (theta_e = k_B * T_e / (m_e * c^2)).
 * @param hotcross_table Precomputed hot cross-section table.
 *
 * @return Total angle-averaged Compton scattering cross-section in units of the Thomson cross-section.
 */
double total_compton_cross_lkup(double w, double theta_e, const ndarray::NDArray<double, 2> &hotcross_table);

}; /* namespace hotcross */
