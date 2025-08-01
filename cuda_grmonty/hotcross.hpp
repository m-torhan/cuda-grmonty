/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/ndarray.hpp"

namespace hotcross {

/**
 * @brief Initializes look-up table for hot cross sections.
 */
void init_table(ndarray::NDArray<double, 2> &table);

double total_compton_cross_lkup(double w, double theta_e, const ndarray::NDArray<double, 2> &hotcross_table);

}; /* namespace hotcross */
