/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/ndarray.hpp"

namespace jnu_mixed {

void init_emiss_tables(ndarray::NDArray<double> &f, ndarray::NDArray<double> &k2);

}; /* namespace jnu_mixed */
