/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <stdexcept>

#include "cuda_grmonty/mathfn.hpp"

namespace mathfn {

double bessel_k_n(int n, double x) {
    if (x <= 0.0) {
        throw std::invalid_argument("x must be > 0");
    }
    if (n < 0) {
        throw std::invalid_argument("n must be >= 0");
    }

    /* Base cases using standard math library */
    if (n == 0) {
        return std::cyl_bessel_k(0, x);
    }
    if (n == 1) {
        return std::cyl_bessel_k(1, x);
    }

    /* Recurrence:
     * K_{n+1}(x) = (2n / x) * K_n(x) + K_{n-1}(x)
     */
    double k_m2 = std::cyl_bessel_k(0, x); // K_0
    double k_m1 = std::cyl_bessel_k(1, x); // K_1
    double k_n = 0.0;

    for (int k = 1; k < n; ++k) {
        k_n = (2.0 * k) / x * k_m1 + k_m2;
        k_m2 = k_m1;
        k_m1 = k_n;
    }

    return k_n;
}

}; /* namespace mathfn */
