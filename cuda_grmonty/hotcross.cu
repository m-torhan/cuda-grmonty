/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <math_constants.h>

#include "cuda_grmonty/hotcross.cuh"

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/ndarray.hpp"

namespace cuda_hotcross {

static __device__ double total_compton_cross_num(double w, double theta_e);

static __device__ double hc_klein_nishina(double w);

static __device__ double dnd_gamma_e(double theta_e, double gamma_e);

static __device__ double boostcross(double w, double mu_e, double gamma_e);

static __device__ double cyl_bessel_k0(double x);

static __device__ double cyl_bessel_k1(double x);

static __device__ double cyl_bessel_k2(double x);

static __global__ void
init_table_entry(double *table, int n, int m, double l_min_w, double d_l_w, double l_t, double d_l_t);

void init_table(ndarray::NDArray<double, 2> &table) {
    static const double l_min_w = std::log10(consts::hotcross::min_w);
    static const double l_min_t = std::log10(consts::hotcross::min_t);
    static const double d_l_w = std::log10(consts::hotcross::max_w / consts::hotcross::min_w) / consts::hotcross::n_w;
    static const double d_l_t = std::log10(consts::hotcross::max_t / consts::hotcross::min_t) / consts::hotcross::n_t;

    double *dev_table;

    cudaError_t code = cudaMalloc((void **)&dev_table, table.size() * sizeof(double));
    if (code != cudaSuccess) {
        /* error */
        return;
    }
    code = cudaMemcpy(dev_table, table.data(), table.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (code != cudaSuccess) {
        /* error */
        return;
    }

    init_table_entry<<<dim3(16, 16), dim3(16, 16)>>>(
        dev_table, table.shape()[0], table.shape()[1], l_min_w, d_l_w, l_min_t, d_l_t);

    code = cudaDeviceSynchronize();
    if (code != cudaSuccess) {
        /* error */
        return;
    }

    code = cudaMemcpy(table.data(), dev_table, table.size() * sizeof(double), cudaMemcpyDeviceToHost);
    if (code != cudaSuccess) {
        /* error */
        return;
    }
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

static __device__ double total_compton_cross_num(double w, double theta_e) {
    if (isnan(w)) {
        return 0.0;
    }

    if (theta_e < consts::hotcross::min_t && w < consts::hotcross::min_w) {
        return consts::sigma_thomson;
    }
    if (theta_e < consts::hotcross::min_t) {
        return hc_klein_nishina(w) * consts::sigma_thomson;
    }

    const double d_mu_e = consts::hotcross::d_mu_e;
    const double d_gamma_e = theta_e * consts::hotcross::d_gamma_e;

    /* integrate over mu_e, gamma_e, where mu_e is the cosine of the
       angle between k and u_e, and the angle k is assumed to lie,
       wlog, along the z axis */
    double cross = 0.0;

    for (double mu_e = -1.0 + 0.5 * d_mu_e; mu_e < 1.0; mu_e += d_mu_e) {
        for (double gamma_e = 1.0 + 0.5 * d_gamma_e; gamma_e < 1.0 + consts::hotcross::max_gamma * theta_e;
             gamma_e += d_gamma_e) {
            double f = 0.5 * dnd_gamma_e(theta_e, gamma_e);

            cross += d_mu_e * d_gamma_e * boostcross(w, mu_e, gamma_e) * f;

            if (isnan(cross)) {
                /* error */
            }
        }
    }

    return cross * consts::sigma_thomson;
}

static __device__ double hc_klein_nishina(double w) {
    if (w < 1.0e-3) {
        return (1.0 - 2.0 * w);
    }

    return (3.0 / 4.0) * (2.0 / (w * w) + (1.0 / (2.0 * w) - (1.0 + w) / (w * w * w)) * log(1.0 + 2.0 * w) +
                          (1.0 + w) / ((1.0 + 2.0 * w) * (1.0 + 2.0 * w)));
}

static __device__ double dnd_gamma_e(double theta_e, double gamma_e) {
    double k2f;

    if (theta_e > 1.0e-2) {
        k2f = cyl_bessel_k2(1.0 / theta_e) * exp(1.0 / theta_e);
    } else {
        k2f = sqrt(CUDART_PI * theta_e / 2.0);
    }

    return ((gamma_e * sqrt(gamma_e * gamma_e - 1.) / (theta_e * k2f)) * exp(-(gamma_e - 1.) / theta_e));
}

static __device__ double boostcross(double w, double mu_e, double gamma_e) {
    double we, boostcross, v;

    /* energy in electron rest frame */
    v = sqrt(gamma_e * gamma_e - 1.0) / gamma_e;
    we = w * gamma_e * (1.0 - mu_e * v);

    boostcross = hc_klein_nishina(we) * (1.0 - mu_e * v);

    if (isnan(boostcross)) {
        /* error */
    }

    return boostcross;
}

static __device__ double cyl_bessel_k0(double x) {
    if (x <= 0.0) {
        return CUDART_INF;
    }
    if (x <= 2.0) {
        /* Small x: K0(x) = -ln(x/2) * I0(x) + P(y), y=(x/2)^2 */
        const double y = 0.5 * x;
        const double y2 = y * y;
        /* clang-format off */
        double P = -0.57721566
                 + y2 * ( 0.42278420
                 + y2 * ( 0.23069756
                 + y2 * ( 0.03488590
                 + y2 * ( 0.00262698
                 + y2 * ( 0.00010750
                 + y2 *   0.00000740)))));
        /* clang-format on */
        return -log(y) * cyl_bessel_i0(x) + P;
    } else {
        /* Large x: K0(x) ~ sqrt(pi/(2x)) e^{-x} * R(2/x) */
        const double y = 2.0 / x;
        /* clang-format off */
        double R = 1.25331414
                 + y * (-0.07832358
                 + y * ( 0.02189568
                 + y * (-0.01062446
                 + y * ( 0.00587872
                 + y * (-0.00251540
                 + y *   0.00053208)))));
        /* clang-format on */
        return exp(-x) * R / sqrt(x);
    }
}

static __device__ double cyl_bessel_k1(double x) {
    if (x <= 0.0) {
        return CUDART_INF;
    }
    if (x <= 2.0) {
        /* Small x: K1(x) ~ ln(x/2)*I1(x) + (1/x) * [ 1 + y * R(y) ], y=(x/2)^2 */
        const double y = 0.5 * x;
        const double y2 = y * y;
        /* clang-format off */
        double R = 0.15443144
                 + y2 * (-0.67278579
                 + y2 * (-0.18156897
                 + y2 * (-0.01919402
                 + y2 * (-0.00110404
                 + y2 *  -0.00004686))));
        /* clang-format on */
        return log(y) * cyl_bessel_i1(x) + (1.0 / x) * (1.0 + y2 * R);
    } else {
        /* Large x: K1(x) ~ sqrt(pi/(2x)) e^{-x} * S(2/x) */
        const double y = 2.0 / x;
        /* clang-format off */
        double S = 1.25331414
                 + y * ( 0.23498619
                 + y * (-0.03655620
                 + y * ( 0.01504268
                 + y * (-0.00780353
                 + y * ( 0.00325614
                 + y *  -0.00068245)))));
        /* clang-format on */
        return exp(-x) * S / sqrt(x);
    }
}

static __device__ double cyl_bessel_k2(double x) {
    if (x <= 0.0)
        return CUDART_INF;

    const double k0 = cyl_bessel_k0(x);
    const double k1 = cyl_bessel_k1(x);

    return k0 + (2.0 * k1) / x;
}

}; /* namespace cuda_hotcross */
