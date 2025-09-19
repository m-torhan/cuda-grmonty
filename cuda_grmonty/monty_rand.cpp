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
static std::mt19937 rd;

void init(int seed) { rd = std::mt19937(seed); }

double rand() {
    std::uniform_real_distribution<double> dist(0, 1);
    return dist(rd);
    // return 0.1;
}

}; /* namespace monty_rand */
