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

void alloc_memory(const harm::Header &header,
                  const harm::Data &data,
                  const harm::Units &units,
                  const ndarray::NDArray<double, 2> &hotcross_table,
                  const std::array<double, consts::n_e_samp + 1> &f,
                  const std::array<double, consts::n_e_samp + 1> &k2);

void free_memory();

void track_super_photons(double bias_norm,
                         double max_tau_scatt,
                         photon::PhotonQueue &photon_queue,
                         std::binary_semaphore &stop_sem,
                         harm::Spectrum (&spectrum)[consts::n_th_bins][consts::n_e_bins],
                         uint64_t &n_super_photon_recorded,
                         uint64_t &n_super_photon_scatt);

}; /* namespace cuda_super_photon */
