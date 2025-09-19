/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/ndarray.hpp"

namespace tetrads {

/**
 * @brief Transform a contravariant vector from coordinate frame to tetrad frame.
 *
 * @param[in]  e_cov    Covariant tetrad basis vectors (e_mu^a).
 * @param[in]  k        Contravariant vector in coordinate frame.
 * @param[out] k_tetrad Resulting vector in tetrad frame.
 */
void coordinate_to_tetrad(const double (&e_cov)[consts::n_dim][consts::n_dim],
                          const double (&k)[consts::n_dim],
                          double (&k_tetrad)[consts::n_dim]);

/**
 * @brief Transform a tetrad-frame vector to the coordinate frame.
 *
 * @param[in]  e_con    Contravariant tetrad basis vectors (e^mu_a).
 * @param[in]  k_tetrad Vector in tetrad frame.
 * @param[out] k        Resulting vector in coordinate frame.
 */
void tetrad_to_coordinate(const double (&e_con)[consts::n_dim][consts::n_dim],
                          const double (&k_tetrad)[consts::n_dim],
                          double (&k)[consts::n_dim]);

/**
 * @brief Construct an orthonormal tetrad for a given 4-velocity.
 *
 * @param[in]  u_con Contravariant 4-velocity of the fluid.
 * @param[in]  trial Initial trial vector for tetrad construction (e.g., spatial direction).
 * @param[in]  g_cov Covariant metric tensor at the point.
 * @param[out] e_con Contravariant tetrad basis vectors.
 * @param[out] e_cov Covariant tetrad basis vectors.
 */
void make_tetrad(const double (&u_con)[consts::n_dim],
                 double (&trial)[consts::n_dim],
                 const ndarray::NDArray<double, 2> &g_cov,
                 double (&e_con)[consts::n_dim][consts::n_dim],
                 double (&e_cov)[consts::n_dim][consts::n_dim]);

/**
 * @brief Lower a contravariant vector using the metric tensor.
 *
 * @param[in]  u_con Input contravariant vector.
 * @param[in]  g_cov Covariant metric tensor.
 * @param[out] u_cov Resulting covariant vector.
 */
void lower(const double (&u_con)[consts::n_dim],
           const ndarray::NDArray<double, 2> &g_cov,
           double (&u_cov)[consts::n_dim]);

}; /* namespace tetrads */
