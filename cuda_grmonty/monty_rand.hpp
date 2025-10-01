/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace monty_rand {

/**
 * @brief Initialize the random number generator with a given seed.
 *
 * @param seed Seed value for the Mersenne Twister engine. Using the same seed ensures reproducibility.
 */
void init(int seed);

/**
 * @brief Generate a random number uniformly distributed in the range [0, 1).
 *
 * @return Random double value in [0, 1).
 */
double uniform();

/**
 * @brief Generate a random number from a chi-squared distribution.
 *
 * @param dof Degrees of freedom of the chi-squared distribution (must be > 0).
 *
 * @return Random double value drawn from a chi-squared distribution.
 */
double chi_sq(int dof);

}; /* namespace monty_rand */
