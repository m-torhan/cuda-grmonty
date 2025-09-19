/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <semaphore>

#include "cuda_grmonty/harm_data.hpp"
#include "cuda_grmonty/photon.hpp"
#include "cuda_grmonty/photon_queue.hpp"

namespace cuda_super_photon {

/**
 * @brief Allocate GPU memory for super-photon propagation.
 *
 * Sets up device arrays for photon data, hotcross tables, and synchrotron emissivity tables.
 *
 * @param header         Simulation header containing grid and parameter information.
 * @param data           Simulation data (geometry, fluid, and zones).
 * @param units          Units conversion data.
 * @param hotcross_table Precomputed total Compton cross-section table.
 * @param f              Precomputed f(theta_e) table for synchrotron emissivity.
 * @param k2             Precomputed k2(theta_e) table for synchrotron emissivity.
 */
void alloc_memory(const harm::Header &header,
                  const harm::Data &data,
                  const harm::Units &units,
                  const ndarray::NDArray<double, 2> &hotcross_table,
                  const std::array<double, consts::n_e_samp + 1> &f,
                  const std::array<double, consts::n_e_samp + 1> &k2);

/**
 * @brief Free GPU memory allocated for super-photon propagation.
 */
void free_memory();

/**
 * @brief Track and propagate super-photons through the simulation.
 *
 * Uses Monte Carlo transport to evolve photons, accumulate spectrum contributions, and update scattering and recording
 * statistics.
 *
 * @param bias_norm               Bias normalization factor for photon weighting.
 * @param max_tau_scatt           Maximum scattering optical depth per photon.
 * @param photon_queue            Thread-safe queue for exchanging photons.
 * @param stop_sem                Semaphore to signal completion or termination.
 * @param spectrum                2D array to accumulate photon spectra (frequency × energy bins).
 * @param n_super_photon_recorded Output: number of super-photons recorded.
 * @param n_super_photon_scatt    Output: number of super-photons that scattered.
 */
void track_super_photons(double bias_norm,
                         double max_tau_scatt,
                         photon::PhotonQueue &photon_queue,
                         std::binary_semaphore &stop_sem,
                         harm::Spectrum (&spectrum)[consts::n_th_bins][consts::n_e_bins],
                         uint64_t &n_super_photon_recorded,
                         uint64_t &n_super_photon_scatt);

}; /* namespace cuda_super_photon */
