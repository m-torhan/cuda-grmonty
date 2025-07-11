/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace hotcross {

/**
 * @brief Computes the total angle-averaged Compton scattering cross-section.
 *
 * @param w Dimensionless photon energy (w = h * nu / (m_e * c^2)).
 * @param thetae Dimensionless electron temperature (thetae = k * T_e / (m_e * c^2)).
 *
 * @return Total angle-averaged Compton scattering cross-section, in units of the Thomson cross-section (sigma_T).
 */
double total_compton_cross_num(double w, double theta_e);

}; /* namespace hotcross */
