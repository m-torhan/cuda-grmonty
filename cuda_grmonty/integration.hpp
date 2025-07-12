/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>

namespace integration {

double gauss_kronrod_61(const std::function<double(double)> &f, double a, double b, double eps_abs, double eps_rel,
                        int max_depth, int max_intervals);

}; /* namespace integration */
