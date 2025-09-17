/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_grmonty/ndarray.hpp"

namespace cuda_hotcross {

void init_table(ndarray::NDArray<double, 2> &table);

}; /* namespace cuda_hotcross */
