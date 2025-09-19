/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/ndarray.hpp"

namespace cuda_hotcross {

/**
 * @brief Initializes the hot cross-section table on the GPU.
 *
 * This function allocates and populates the device table used for fast lookup of angle-averaged Compton scattering
 * cross-sections over a grid of photon energies and electron temperatures.
 *
 * @param table Reference to an NDArray that will store the initialized table.
 */
void init_table(ndarray::NDArray<double, 2> &table);

}; /* namespace cuda_hotcross */
