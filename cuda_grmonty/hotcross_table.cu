/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <math_constants.h>

#include "cuda_grmonty/hotcross.cuh"
#include "cuda_grmonty/hotcross_table.cuh"
#include "cuda_grmonty/utils.cuh"

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/ndarray.hpp"

namespace cuda_hotcross {

/**
 * @brief Initializes a portion of the hot cross-section table on the device.
 *
 * Each thread computes a single table entry corresponding to photon energy and electron temperature grid indices.
 *
 * @param table   Pointer to device array storing the table.
 * @param n       Number of photon energy bins.
 * @param m       Number of electron temperature bins.
 * @param l_min_w Logarithm of minimum photon energy.
 * @param d_l_w   Logarithmic step size in photon energy.
 * @param l_t     Logarithm of electron temperature for this entry.
 * @param d_l_t   Logarithmic step size in electron temperature.
 */
static __global__ void
init_table_entry(double *table, int n, int m, double l_min_w, double d_l_w, double l_t, double d_l_t);

void init_table(ndarray::NDArray<double, 2> &table) {
    double *dev_table;

    gpuErrchk(cudaMalloc((void **)&dev_table, table.size() * sizeof(double)));

    init_table_entry<<<dim3(16, 16), dim3(16, 16)>>>(dev_table,
                                                     table.shape()[0],
                                                     table.shape()[1],
                                                     consts::hotcross::l_min_w,
                                                     consts::hotcross::d_l_w,
                                                     consts::hotcross::l_min_t,
                                                     consts::hotcross::d_l_t);

    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(table.data(), dev_table, table.size() * sizeof(double), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(dev_table));
}

static __global__ void
init_table_entry(double *table, int n, int m, double l_min_w, double d_l_w, double l_min_t, double d_l_t) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        for (int j = threadIdx.y + blockIdx.y * blockDim.y; j < m; j += blockDim.y * gridDim.y) {
            double l_w = l_min_w + i * d_l_w;
            double l_t = l_min_t + j * d_l_t;

            table[i * m + j] = log10(total_compton_cross_num(pow(10.0, l_w), pow(10.0, l_t)));
        }
    }
}

}; /* namespace cuda_hotcross */
