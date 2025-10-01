/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <random>

#include "cuda_grmonty/monty_rand.hpp"

namespace monty_rand {

/**
 * @brief Mersenne Twister random number engine.
 *
 * A globally accessible instance of std::mt19937 used for deterministic and reproducible random number generation
 * across Monte Carlo routines.
 */
thread_local static std::mt19937 rd;

void init(int seed) { rd = std::mt19937(seed); }

double uniform() {
    std::uniform_real_distribution<double> dist(0, 1);
    return dist(rd);
}

double chi_sq(int dof) {
    std::chi_squared_distribution<double> chi_sq(dof);
    return chi_sq(rd);
}

}; /* namespace monty_rand */
