/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace mathfn {

/**
 * @brief Computes the modified Bessel function of the second kind K_n(x).
 *
 * @param n Non-negative integer order (n >= 0)
 * @param x Positive real argument (x > 0)
 * @return Value of K_n(x)
 *
 * @throws std::invalid_argument if x <= 0 or n < 0
 */
double bessel_k_n(int n, double x);

}; /* namespace mathfn */
