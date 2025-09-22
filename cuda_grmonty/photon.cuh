/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/consts.hpp"

namespace cuda_super_photon {

/**
 * @brief Represents an array of photons stored in a struct-of-arrays (SoA) layout.
 *
 * This structure provides GPU-friendly memory access by organizing photon attributes as separate contiguous arrays.
 * Each array holds one attribute across all photons, which improves memory coalescing during CUDA kernel execution.
 */
struct PhotonArray {
    double *x[consts::n_dim];
    double *k[consts::n_dim];
    double *dkdlam[consts::n_dim];
    double *w;
    double *e;
    double *l;
    double *x1i;
    double *x2i;
    double *tau_abs;
    double *tau_scatt;
    double *n_e_0;
    double *theta_e_0;
    double *b_0;
    double *e_0;
    double *e_0_s;
    int *n_scatt;
};

/**
 * @brief Allocate GPU memory for a PhotonArray.
 *
 * Each field in the PhotonArray is allocated as a contiguous GPU array of length @p n. The function must be paired
 * with free_photon_array() to avoid memory leaks.
 *
 * @param[out] photon_array  Struct of arrays to allocate memory for.
 * @param[in] n              Number of photons to allocate.
 */
void alloc_photon_array(PhotonArray &photon_array, size_t n);

/**
 * @brief Free GPU memory for a PhotonArray.
 *
 * Releases all memory previously allocated by alloc_photon_array(). After this call, all pointers inside
 * @p photon_array are invalid.
 *
 * @param[in,out] photon_array  Struct of arrays to deallocate.
 */
void free_photon_array(PhotonArray &photon_array);

}; /* namespace cuda_super_photon */
