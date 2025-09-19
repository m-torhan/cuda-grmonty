/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>

namespace integration {

/**
 * @brief Compute the integral of a function over [a, b] using adaptive 61-point Gauss-Kronrod quadrature.
 *
 * This routine adaptively subdivides the integration interval to achieve the requested absolute and relative error
 * tolerances. The algorithm uses a max-heap to prioritize intervals with the largest estimated error.
 *
 * @param f             Function to integrate.
 * @param a             Lower limit of integration.
 * @param b             Upper limit of integration.
 * @param eps_abs       Absolute error tolerance.
 * @param eps_rel       Relative error tolerance.
 * @param max_intervals Maximum number of subintervals allowed.
 *
 * @return Approximated integral of the function over [a, b].
 */
double gauss_kronrod_61(
    const std::function<double(double)> &f, double a, double b, double eps_abs, double eps_rel, int max_intervals);

}; /* namespace integration */
