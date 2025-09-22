/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_grmonty/photon.cuh"
#include "cuda_grmonty/utils.cuh"

#include "cuda_grmonty/consts.hpp"

namespace cuda_super_photon {

void alloc_photon_array(PhotonArray &photon_array, size_t n) {
    for (int i = 0; i < consts::n_dim; ++i) {
        gpuErrchk(cudaMalloc((void **)&photon_array.x[i], n * sizeof(double[consts::n_dim])));
        gpuErrchk(cudaMalloc((void **)&photon_array.k[i], n * sizeof(double[consts::n_dim])));
        gpuErrchk(cudaMalloc((void **)&photon_array.dkdlam[i], n * sizeof(double[consts::n_dim])));
    }
    gpuErrchk(cudaMalloc((void **)&photon_array.w, n * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&photon_array.e, n * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&photon_array.l, n * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&photon_array.x1i, n * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&photon_array.x2i, n * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&photon_array.tau_abs, n * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&photon_array.tau_scatt, n * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&photon_array.n_e_0, n * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&photon_array.theta_e_0, n * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&photon_array.b_0, n * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&photon_array.e_0, n * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&photon_array.e_0_s, n * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&photon_array.n_scatt, n * sizeof(int)));
}

void free_photon_array(PhotonArray &photon_array) {
    for (int i = 0; i < consts::n_dim; ++i) {
        gpuErrchk(cudaFree(photon_array.x[i]));
        gpuErrchk(cudaFree(photon_array.k[i]));
        gpuErrchk(cudaFree(photon_array.dkdlam[i]));
    }
    gpuErrchk(cudaFree(photon_array.w));
    gpuErrchk(cudaFree(photon_array.e));
    gpuErrchk(cudaFree(photon_array.l));
    gpuErrchk(cudaFree(photon_array.x1i));
    gpuErrchk(cudaFree(photon_array.x2i));
    gpuErrchk(cudaFree(photon_array.tau_abs));
    gpuErrchk(cudaFree(photon_array.tau_scatt));
    gpuErrchk(cudaFree(photon_array.n_e_0));
    gpuErrchk(cudaFree(photon_array.theta_e_0));
    gpuErrchk(cudaFree(photon_array.b_0));
    gpuErrchk(cudaFree(photon_array.e_0));
    gpuErrchk(cudaFree(photon_array.e_0_s));
    gpuErrchk(cudaFree(photon_array.n_scatt));
}

}; /* namespace cuda_super_photon */
